"""Pre-run guard for `vega_snapshot` (selfbuild) fixtures: is the case gradable at all?

A selfbuild case passes only when its hidden check AND its regression command (Vega's whole
`tests` suite, run on the snapshot) pass. Audit 2026-10-27: the snapshot (`git archive` of
evals.SNAPSHOT_PATHS) lacked scripts/, release/, docs/, eval data and .git, so the regression
command failed on the UNMODIFIED fixture at bbf60af / a24fabe (4 tests) and 76b90c8 (15): no
attempt could pass, and 58 of 73 selfbuild attempts since 10-24 ran on such fixtures, scored as
model failures. Fixed for those tests in 694c376 (needs_checkout skips); this guard stops it
from recurring unseen.

Before a suite with vega_snapshot cases runs (`vega eval`, `vega eval --ab`, and so every night
runner step), `check` builds the fixture once per snapshot content and runs, model-free:

  regression   each distinct regression_command on the UNMODIFIED source (no mutation). Its
               failing tests are the case's BASELINE: grading then judges the case's regression
               run as "no new failures vs the unmodified snapshot" (evals.run_checks with
               `regression_baseline`), so one red test the case never touches does not make every
               case ungradable (11-03: one host-state test aborted all 15). Strict where it
               matters, aborted with "fixture red: ..." (a fixture error: not a model failure,
               not an attempt, left out of pass rates) when a failing test lives in a file the
               case mutates or expects to change, or when the run is red with no test named
  unmutated    each case's hidden check on the unmutated source: a hidden test the correct
               code fails makes the case unpassable -> aborted the same way
  floor        each case's hidden check on the untouched MUTATED fixture: hidden tests passed
               before the agent does anything. Recorded on the case (`hidden_floor`) so A/B
               summaries read progress relative to it ("broke fewer tests" is not "helps")

Results are cached in VEGA_HOME/evals/fixture_guard.json (the night runner points
VEGA_FIXTURE_GUARD_CACHE at VEGA_HOME/night/fixture_guard.json, which outlives its scratch home)
keyed by the snapshot's content (the git tree ids of SNAPSHOT_PATHS at HEAD), so a commit that
changes nothing in the snapshot reuses them and the guard runs once per snapshot. Cost: the
regression run is the whole suite (253-324 s per commit in the audit; 356 s at b8baf78 on
2026-10-27 with the research runner sharing the CPU) plus ~1-2 s per case for the two hidden
checks (27 s for the 15 selfbuild cases): 383 s uncached, a git ls-tree when cached. Bounded by
GUARD_TIMEOUT (VEGA_FIXTURE_GUARD_TIMEOUT): a regression run over it is not a verdict, so the
cases run unguarded with a note (cached too; VEGA_FIXTURE_GUARD=refresh re-checks).
VEGA_FIXTURE_GUARD=0 switches the guard off.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GUARD_ENV = "VEGA_FIXTURE_GUARD"  # "0"/"off": no guard; "refresh": ignore the cache
TIMEOUT_ENV = "VEGA_FIXTURE_GUARD_TIMEOUT"
CACHE_ENV = "VEGA_FIXTURE_GUARD_CACHE"
GUARD_TIMEOUT = 900  # seconds for one regression run (the audit measured 253-324 s)
RED_PREFIX = "fixture red:"
SHOWN_IDS = 5  # failing test ids named in an error (the rest counted)
OUTPUT_LIMIT = 2_000_000
_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.MULTILINE)
_SUMMARY = re.compile(r"^=*\s*(\d+ (?:failed|passed|errors?|skipped)[^\n]*?)\s*=*$", re.MULTILINE)


@dataclass
class GuardResult:
    errors: dict[str, str] = field(default_factory=dict)  # case id -> "fixture red: ..."
    floors: dict[str, list[int]] = field(default_factory=dict)  # case id -> [passed, total]
    # case id -> tests failing on the unmodified snapshot that the case does not touch: grading
    # passes its regression run when it fails only these (evals.run_checks `regression_baseline`)
    baselines: dict[str, list[str]] = field(default_factory=dict)
    summary: dict[str, Any] | None = None  # for the run record (None: no snapshot case / off)

    def note(self) -> str:
        return str((self.summary or {}).get("note") or "")


def enabled() -> bool:
    return os.environ.get(GUARD_ENV, "").strip().lower() not in ("0", "off", "false", "no")


def cache_path() -> Path:
    """VEGA_FIXTURE_GUARD_CACHE (the night runner points it at VEGA_HOME/night/, which outlives
    its per-night scratch home), else VEGA_HOME/evals/fixture_guard.json."""
    explicit = os.environ.get(CACHE_ENV)
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.environ.get("VEGA_HOME") or Path.home() / ".vega").expanduser() / "evals" / "fixture_guard.json"


def timeout() -> int:
    try:
        return max(1, int(os.environ.get(TIMEOUT_ENV) or GUARD_TIMEOUT))
    except ValueError:
        return GUARD_TIMEOUT


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def snapshot_key(repo: Path, paths: tuple[str, ...], exclude: tuple[str, ...], ref: str = "HEAD") -> tuple[str, str]:
    """(content key, short commit): the key is the git tree/blob ids of the snapshot paths at
    `ref`, so commits that change nothing in the snapshot share it."""
    listing = subprocess.run(["git", "-C", str(repo), "ls-tree", ref, "--", *paths],
                             capture_output=True, text=True, check=False)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", ref],
                            capture_output=True, text=True, check=False).stdout.strip()
    if listing.returncode or not listing.stdout.strip():
        raise RuntimeError(f"git ls-tree {ref} failed: {listing.stderr.strip()[:200]}")
    return _digest([listing.stdout, list(exclude)]), commit


def failing_ids(output: str) -> list[str]:
    return list(dict.fromkeys(_FAILED.findall(output)))


def red_error(failing: list[str], detail: str = "") -> str:
    if not failing:
        return f"{RED_PREFIX} the regression command fails on the unmodified snapshot ({detail[:200]})"
    shown = ", ".join(failing[:SHOWN_IDS]) + (f", +{len(failing) - SHOWN_IDS} more" if len(failing) > SHOWN_IDS else "")
    return f"{RED_PREFIX} {len(failing)} tests fail on the unmodified snapshot ({shown})"


def test_file(node: str) -> str:
    """The file of a pytest node id or collection-error path (`tests/a.py::test_x[1]` -> `tests/a.py`)."""
    return node.partition("::")[0]


def touched_files(case: dict[str, Any]) -> set[str]:
    """Files the case's change touches: those its mutation edits and those it expects the fix to
    change. A pre-existing failure in one of them would mix with the case's own verdict."""
    return ({str(m.get("path")) for m in case.get("mutation") or [] if m.get("path")}
            | set(case.get("expected_files") or []))


def baseline_or_error(case: dict[str, Any], failing: list[str], count: str) -> tuple[list[str], str]:
    """(baseline, '') when every failing test on the unmodified snapshot is one the case does not
    touch; ([], "fixture red: ...") when one is in a touched file or no failing test was named."""
    if not failing:
        return [], red_error([], count)
    touched = touched_files(case)
    mixed = [node for node in failing if test_file(node) in touched]
    if mixed:
        return [], red_error(mixed).replace("tests fail on the unmodified snapshot",
                                            "tests in files this case touches fail on the unmodified snapshot")
    return list(failing), ""


def run_regression(command: str, project: Path, limit: int) -> dict[str, Any]:
    """The regression command on an unmodified fixture, run as grading runs it (sandbox.run_check)
    plus `-rfE` so pytest names every failing test."""
    from vega import sandbox

    started = time.monotonic()
    extra = " -rfE --tb=no" if "pytest" in command else ""
    output = sandbox.run_check(command + extra, project, timeout=limit, output_limit=OUTPUT_LIMIT)
    seconds = round(time.monotonic() - started, 1)
    first = output.partition("\n")[0]
    summary = _SUMMARY.findall(output)
    row: dict[str, Any] = {"seconds": seconds, "count": summary[-1] if summary else first[:200]}
    if output.startswith("Check timed out"):
        row["status"] = "timeout"
    elif first == "exit code: 0":
        row["status"] = "green"
    elif first.startswith("exit code:"):
        row["status"] = "red"
        row["failing"] = failing_ids(output)
    else:  # the sandbox could not run it at all: no verdict on the fixture
        row["status"] = "unavailable"
        row["count"] = first[:200]
    return row


def _hidden(case: dict[str, Any], project: Path) -> list[int] | None:
    """The hidden counts grading would record for this tree (evals.run_checks without the
    regression command)."""
    from vega import evals

    outcome = evals.run_checks({**case, "regression_command": None}, project, "")
    return list(outcome.hidden) if outcome.hidden else None


def _case_digest(case: dict[str, Any]) -> str:
    return _digest({k: case.get(k) for k in ("mutation", "files", "hidden_files", "check_command",
                                              "check_timeout", "fixture_tree")})


def check_case(case: dict[str, Any], archive: bytes) -> dict[str, Any]:
    """Hidden check on the unmutated source and on the untouched mutated fixture."""
    from vega import evals

    started = time.monotonic()
    row: dict[str, Any] = {"digest": _case_digest(case)}
    for name, spec in (("unmutated", {**case, "mutation": []}), ("floor", case)):
        with tempfile.TemporaryDirectory(prefix="vega-guard-") as temp:
            project = Path(temp).resolve()
            try:
                evals.build_fixture(spec, project, archive)
            except evals.FixtureError as exc:
                row["fixture_error"] = str(exc)
                break
            row[name] = _hidden(spec, project)
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    tmp.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def check(cases: list[dict[str, Any]], archive: bytes | None, *, repo: Path, paths: tuple[str, ...],
          exclude: tuple[str, ...], ref: str = "HEAD", say: Any = None) -> GuardResult:
    """Errors (cases to abort), hidden floors and a summary for the vega_snapshot cases in `cases`."""
    snap = [c for c in cases if c.get("vega_snapshot")]
    if not snap or archive is None:
        return GuardResult()
    if not enabled():
        return GuardResult(summary={"status": "off", "note": f"fixture guard off ({GUARD_ENV}=0)"})
    say = say or (lambda text: None)
    try:
        key, commit = snapshot_key(repo, paths, exclude, ref)
    except RuntimeError as exc:
        return GuardResult(summary={"status": "unavailable", "note": f"fixture guard not run: {exc}"})
    path = cache_path()
    cache = _load(path)
    refresh = os.environ.get(GUARD_ENV, "").strip().lower() == "refresh"
    entry = {} if refresh else dict(cache.get(key) or {})
    entry.setdefault("commit", commit)
    regression = dict(entry.get("regression") or {})
    case_rows = dict(entry.get("cases") or {})
    limit = timeout()
    started = time.monotonic()
    ran = False
    for command in dict.fromkeys(c["regression_command"] for c in snap if c.get("regression_command")):
        rkey = _digest(command)
        if rkey in regression:
            continue
        say(f"Fixture guard: running the regression command on the unmodified snapshot {commit} "
            f"(once per snapshot; bounded at {limit}s)")
        with tempfile.TemporaryDirectory(prefix="vega-guard-") as temp:
            project = Path(temp).resolve()
            first = next(c for c in snap if c.get("regression_command") == command)
            from vega import evals
            try:
                evals.build_fixture({**first, "mutation": []}, project, archive)
                row = run_regression(command, project, limit)
            except evals.FixtureError as exc:
                row = {"status": "unavailable", "count": f"fixture error: {exc}"[:200], "seconds": 0}
        regression[rkey] = {**row, "command": command, "checked": _now(), "limit": limit}
        ran = True
    for case in snap:
        digest = _case_digest(case)
        if (case_rows.get(case["id"]) or {}).get("digest") == digest:
            continue
        case_rows[case["id"]] = {**check_case(case, archive), "checked": _now()}
        ran = True
    entry.update(regression=regression, cases=case_rows)
    if ran:
        cache[key] = entry
        _save(path, cache)

    result = GuardResult()
    notes = []
    for case in snap:
        reg = regression.get(_digest(case.get("regression_command"))) if case.get("regression_command") else None
        row = case_rows.get(case["id"]) or {}
        if reg and reg.get("status") == "red":
            baseline, error = baseline_or_error(case, list(reg.get("failing") or []), str(reg.get("count", "")))
            if error:
                result.errors[case["id"]] = error
            else:
                result.baselines[case["id"]] = baseline
        unmutated = row.get("unmutated")
        if case["id"] not in result.errors and unmutated and unmutated[0] < unmutated[1]:
            result.errors[case["id"]] = (f"{RED_PREFIX} the hidden check fails on the unmutated source "
                                         f"({unmutated[1] - unmutated[0]} of {unmutated[1]} hidden tests fail)")
        if row.get("floor"):
            result.floors[case["id"]] = list(row["floor"])
    statuses = {r.get("status") for r in regression.values()}
    for r in regression.values():
        if r.get("status") in ("timeout", "unavailable"):
            why = f"over {r.get('limit', limit)}s" if r["status"] == "timeout" else str(r.get("count", ""))
            notes.append(f"fixture guard skipped the regression check ({r['status']}: {why}); "
                         "the cases ran unguarded")
    if result.errors:
        notes.append(f"fixture red on {len(result.errors)} of {len(snap)} snapshot case(s) at {commit}: "
                     + next(iter(result.errors.values())).removeprefix(RED_PREFIX).strip())
    if result.baselines:
        failing = sorted({node for nodes in result.baselines.values() for node in nodes})
        shown = ", ".join(failing[:SHOWN_IDS]) + (f", +{len(failing) - SHOWN_IDS} more"
                                                  if len(failing) > SHOWN_IDS else "")
        notes.append(f"{len(failing)} test(s) fail on the unmodified snapshot at {commit} ({shown}); "
                     f"{len(result.baselines)} case(s) that do not touch them are graded as "
                     "no new failures vs the snapshot")
    status = ("red" if result.errors else "baseline" if result.baselines
              else "green" if statuses <= {"green"} else "skipped")
    result.summary = {
        "status": status, "commit": commit, "key": key, "cached": not ran,
        "seconds": round(time.monotonic() - started, 1),
        "regression": [{k: r.get(k) for k in ("status", "count", "seconds", "checked")}
                       | {"failing": len(r.get("failing") or [])} for r in regression.values()],
        "red_cases": sorted(result.errors), "floors": len(result.floors),
        "baseline_cases": sorted(result.baselines),
        "note": "; ".join(notes),
    }
    return result


def is_red(error: object) -> bool:
    return str(error or "").startswith(RED_PREFIX)
