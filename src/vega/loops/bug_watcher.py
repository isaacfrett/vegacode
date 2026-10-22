"""Bug Watcher (skill `diff-bug`): check one commit for bugs.

Deterministic pass (runs any time): the project's tests at the commit and `ruff check` on the
Python files the commit changed (only errors the parent did not have), both through
`sandbox.run_check`. Failures are findings with evidence `failing_test` / `lint`.

Model pass (only while idle): the model sees the commit's diff — never whole files — and writes
ONE pytest test that should fail if the change introduced a bug. The finding is kept only when the
test fails at the commit and passes at its parent ("no reproduction, no finding"). Everything runs
in a temporary `git worktree` (shares the object store) that is removed when the job ends.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import debuglog, resources, sandbox
from . import inbox
from .definitions import LoopDef
from .idle import DISK_MIN_GB, IdleCheck
from .state import root

if TYPE_CHECKING:
    from ..agent import CodingAgent

TEST_COMMAND = "python -m pytest -q -p no:cacheprovider"
MAX_DIFF_CHARS = 12000  # the 7B gets the diff only; bigger diffs are skipped, not truncated
MODEL_ATTEMPTS = 2  # one retry when the reply has no usable test
GENERATED_TEST = "test_vega_loop_{sha}.py"
_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_EXIT = re.compile(r"^exit code: (-?\d+)")
_LINT = re.compile(r"^(?P<path>[^\s:][^:]*\.pyi?):(?P<line>\d+):(?P<col>\d+): (?P<rest>.+)$")
_FAILED_ID = re.compile(r"^(?:FAILED|ERROR) (?P<id>[^\s]+\.py(?:::[^\s]+)?)")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<line>\d+)")

SYSTEM = """You check one commit of a Python project for bugs. You see only the commit's diff.
Write ONE pytest test that passes if the changed code is correct and FAILS if the change
introduced a bug. Test behaviour the diff changed, using the project's real modules and names as
they are after the commit. Use plain asserts and no network, no sleeping, no files outside
pytest's tmp_path. If the change looks correct and you cannot name a concrete bug, reply exactly
NO_BUG."""

ASK = """Commit {sha}: {subject}

```diff
{diff}
```

Reply with one line `TITLE: <the suspected bug in a few words>`, then the whole test file in one
```python block. Or reply NO_BUG."""

AgentFactory = Callable[[Path], "CodingAgent"]


def default_agent(project: Path) -> CodingAgent:
    from .. import server
    from ..agent import CodingAgent

    server.ensure_running()  # only reached while idle: starting the model server is fine then
    return CodingAgent(project, can_write=False, can_delegate=False, allow_commands=False)


@dataclass
class JobReport:
    commit: str
    filed: list[dict[str, Any]] = field(default_factory=list)  # new findings written to the inbox
    duplicates: int = 0  # findings that matched an open one (recorded as seen again)
    model: str = "not requested"  # what the model pass did: ran / deferred / skipped / dropped
    model_ran: bool = False  # the model pass finished (kept or dropped); False when deferred
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""  # the job itself failed (budget, git, sandbox crash); the loop shows "failing"
    debug_log: str | None = None

    def summary(self) -> str:
        found = f"{len(self.filed)} new finding(s)" if self.filed else "no new findings"
        if self.duplicates:
            found += f", {self.duplicates} already in the inbox"
        text = f"{self.commit[:8]}: {found}; model: {self.model}"
        return f"{text}; ERROR {self.error}" if self.error else text


class BudgetExceeded(RuntimeError):
    pass


class Deadline:
    def __init__(self, minutes: float):
        self.minutes = minutes
        self.end = time.monotonic() + minutes * 60

    def remaining(self) -> int:
        left = int(self.end - time.monotonic())
        if left <= 0:
            raise BudgetExceeded(f"hit the {self.minutes:g}-minute budget")
        return left


# -- git and worktrees -----------------------------------------------------------------------------

def git(repo: Path, *args: str, timeout: int = 60) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                            timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {(result.stderr or result.stdout).strip()}")
    return result.stdout


def parent_of(project: Path, commit: str) -> str | None:
    parents = git(project, "rev-list", "--parents", "-n", "1", commit).split()
    return parents[1] if len(parents) > 1 else None


def changed_python(project: Path, parent: str | None, commit: str) -> list[str]:
    """Python files added or modified by the commit (deleted files cannot be linted)."""
    if parent is None:
        out = git(project, "diff-tree", "--root", "--no-commit-id", "--name-only", "-r",
                  "--diff-filter=AM", commit)
    else:
        out = git(project, "diff", "--name-only", "--diff-filter=AM", parent, commit)
    return [line for line in out.splitlines() if line.endswith((".py", ".pyi"))]


def is_test_file(path: str) -> bool:
    name = Path(path).name
    return path.startswith(("tests/", "test/")) or name.startswith("test_") or name == "conftest.py"


def worktree_root() -> Path:
    return root() / "worktrees"


@contextmanager
def worktree(project: Path, commit: str) -> Iterator[Path]:
    """A detached worktree at `commit` (objects shared, only the checkout is new on disk),
    removed when the block ends whatever happens. The project's .venv is symlinked, not copied."""
    path = worktree_root() / f"{commit[:12]}-{os.getpid()}-{time.monotonic_ns() % 10**6}"
    path.parent.mkdir(parents=True, exist_ok=True)
    git(project, "worktree", "add", "--detach", "--quiet", str(path), commit, timeout=120)
    try:
        for name in (".venv", "venv"):
            if (project / name).is_dir() and not (path / name).exists():
                (path / name).symlink_to(project / name, target_is_directory=True)
        yield path
    finally:
        try:
            git(project, "worktree", "remove", "--force", str(path), timeout=120)
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        shutil.rmtree(path, ignore_errors=True)
        try:
            git(project, "worktree", "prune")
        except (RuntimeError, subprocess.TimeoutExpired):
            pass


# -- checks ----------------------------------------------------------------------------------------

def exit_code(result: str) -> int | None:
    match = _EXIT.match(result)
    return int(match.group(1)) if match else None


def has_tests(project: Path) -> bool:
    if (project / "tests").is_dir() or (project / "pytest.ini").is_file():
        return True
    pyproject = project / "pyproject.toml"
    return pyproject.is_file() and "[tool.pytest" in pyproject.read_text(encoding="utf-8")


def run_tests(wt: Path, deadline: Deadline, target: str = "") -> tuple[int | None, str]:
    command = f"{TEST_COMMAND} {target}".strip()
    result = sandbox.run_check(command, wt, timeout=deadline.remaining())
    return exit_code(result), result


def lint(wt: Path, files: list[str], deadline: Deadline) -> dict[str, list[tuple[str, str]]]:
    """ruff errors per file as (line, "CODE message"); empty when ruff is clean or missing."""
    present = [f for f in files if (wt / f).is_file()]
    if not present:
        return {}
    result = sandbox.run_check(
        "ruff check --no-cache --output-format concise " + " ".join(f"'{f}'" for f in present),
        wt, timeout=deadline.remaining())
    errors: dict[str, list[tuple[str, str]]] = {}
    for line in result.splitlines():
        match = _LINT.match(line.strip())
        if match:
            path = match.group("path").removeprefix("./")
            errors.setdefault(path, []).append((match.group("line"), match.group("rest")))
    return errors


def diff_locations(diff: str) -> list[dict[str, Any]]:
    """First changed line of each file in a unified diff, for a finding's `where`."""
    where: list[dict[str, Any]] = []
    current = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            current = line[4:].removeprefix("b/")
            continue
        match = _HUNK.match(line)
        if match and current and current != "/dev/null" and \
                not any(item["path"] == current for item in where):
            where.append({"path": current, "line": int(match.group("line"))})
    return where[:5]


# -- the model pass --------------------------------------------------------------------------------

@dataclass
class Suspicion:
    title: str
    test: str


def parse_reply(reply: str) -> Suspicion | str:
    """The model's test, or why the reply is unusable ("NO_BUG" when it saw no bug)."""
    if reply.strip().upper().startswith("NO_BUG") or (
            "NO_BUG" in reply and not _CODE_BLOCK.search(reply)):
        return "NO_BUG"
    blocks = _CODE_BLOCK.findall(reply)
    if not blocks:
        return "no ```python block with a test"
    code = max(blocks, key=len).strip("\n") + "\n"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"the test does not parse: {exc.msg} (line {exc.lineno})"
    if not any(isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test")
               for node in ast.walk(tree)):
        return "no test_ function in the code"
    title = next((line.split(":", 1)[1].strip() for line in reply.splitlines()
                  if line.strip().upper().startswith("TITLE:")), "") or "suspected bug in the change"
    return Suspicion(title=title[:120], test=code)


def ask_model(agent: CodingAgent, messages: list[dict[str, Any]], deadline: Deadline) -> str:
    reply = agent.client.chat.completions.create(model=agent.model, messages=messages,
                                                 timeout=deadline.remaining(), **agent.sampling)
    return reply.choices[0].message.content or ""


def model_pass(
    loop: LoopDef, project: Path, wt: Path, commit: str, parent: str | None,
    deadline: Deadline, gate: Callable[[], IdleCheck], agent_factory: AgentFactory,
    report: JobReport,
) -> dict[str, Any] | None:
    """Ask for one test, verify it at the commit and the parent; return a finding or None."""
    if parent is None:
        report.model, report.model_ran = "skipped: root commit has no parent to compare", True
        return None
    if not sandbox.available():
        report.model, report.model_ran = ("skipped: sandbox unavailable, so a test cannot be "
                                          "verified (an unverified test is never a finding)"), True
        return None
    sources = [f for f in changed_python(project, parent, commit) if not is_test_file(f)]
    if not sources:
        report.model, report.model_ran = "skipped: no changed Python source files", True
        return None
    diff = git(project, "diff", "--unified=3", parent, commit, "--", *sources)
    if len(diff) > MAX_DIFF_CHARS:
        report.model, report.model_ran = (f"skipped: diff is {len(diff):,} characters "
                                          f"(limit {MAX_DIFF_CHARS:,})"), True
        return None
    subject = git(project, "log", "-1", "--format=%s", commit).strip()
    agent = agent_factory(project)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ASK.format(sha=commit[:8], subject=subject, diff=diff)},
    ]
    suspicion: Suspicion | None = None
    for attempt in range(MODEL_ATTEMPTS):
        check = gate()  # idle is re-checked before every model step
        if not check.ok:
            report.model = f"deferred: {', '.join(check.reasons)}"
            return None
        reply = ask_model(agent, messages, deadline)
        parsed = parse_reply(reply)
        debuglog.record("stage", stage="loop.bug_watcher.reply", attempt=attempt + 1,
                        result=parsed if isinstance(parsed, str) else parsed.title)
        if parsed == "NO_BUG":
            report.model, report.model_ran = "ran: the model saw no bug", True
            return None
        if isinstance(parsed, Suspicion):
            suspicion = parsed
            break
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"That reply is unusable: {parsed}. Reply with "
                                                 "TITLE: ... and one ```python test file, or NO_BUG."}]
    if suspicion is None:
        report.model, report.model_ran = "dropped: no usable test after 2 replies", True
        return None
    folder = "tests" if (wt / "tests").is_dir() else "."
    relative = f"{folder}/{GENERATED_TEST.format(sha=commit[:8])}".removeprefix("./")
    (wt / relative).write_text(suspicion.test, encoding="utf-8")
    at_commit, output = run_tests(wt, deadline, relative)
    debuglog.record("stage", stage="loop.bug_watcher.at_commit", exit_code=at_commit, output=output)
    if at_commit != 1:
        why = "passes at the commit" if at_commit == 0 else f"does not run at the commit (exit {at_commit})"
        report.model, report.model_ran = f"dropped: the model's test {why}", True
        return None
    git(wt, "checkout", "--detach", "--quiet", parent)
    at_parent, parent_output = run_tests(wt, deadline, relative)
    debuglog.record("stage", stage="loop.bug_watcher.at_parent", exit_code=at_parent,
                    output=parent_output)
    if at_parent != 0:
        report.model, report.model_ran = (f"dropped: the model's test also fails at the parent "
                                          f"(exit {at_parent}), so it shows no regression"), True
        return None
    evidence_dir = inbox.tests_folder()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    kept = evidence_dir / f"{loop.slug}-{commit[:8]}-{time.time_ns() % 10**6}.py"
    kept.write_text(suspicion.test, encoding="utf-8")
    report.model, report.model_ran = "ran: reproduced (fails at the commit, passes at the parent)", True
    return inbox.make_finding(
        loop=loop.name, project=project, commit=commit[:12], kind="bug", title=suspicion.title,
        where=diff_locations(diff),
        evidence={"type": "failing_test", "path": str(kept), "output": output.strip(),
                  "fails_at": commit[:12], "passes_at": parent[:12], "test": relative},
        confidence="verified", route=_route(loop, "failing_test"), debug_log=report.debug_log)


def _route(loop: LoopDef, evidence_type: str) -> str:
    """The first stored route rule matching the evidence type (routing itself is phase 2)."""
    for rule in loop.route:
        when = rule.get("when", {})
        if all(key == "evidence" and value == evidence_type for key, value in when.items()):
            return str(rule["to"])
    return "user"


# -- the job ---------------------------------------------------------------------------------------

def disk_free_gb() -> float:
    return shutil.disk_usage(resources.DISK_PATH).free / 1024**3


@contextmanager
def _debug_mode(label: str) -> Iterator[debuglog.DebugLog | None]:
    """Every loop job writes a debug log (`vega debug watch` follows it), even when VEGA_DEBUG
    is off for the rest of the process; the setting is restored afterwards."""
    before = os.environ.get("VEGA_DEBUG")
    os.environ["VEGA_DEBUG"] = "1"
    log = debuglog.start(label)
    try:
        yield log
    finally:
        if before is None:
            os.environ.pop("VEGA_DEBUG", None)
        else:
            os.environ["VEGA_DEBUG"] = before
        if log is not None:
            log.close()
            debuglog._current = None  # the job's log ends with the job


def check_commit(
    loop: LoopDef, commit: str, *, deterministic: bool = True,
    gate: Callable[[], IdleCheck] | None = None,
    agent_factory: AgentFactory = default_agent,
) -> JobReport:
    """One bounded job on one commit. `gate` decides model work (None: no model pass)."""
    project = loop.project
    commit = git(project, "rev-parse", "--verify", f"{commit}^{{commit}}").strip()
    report = JobReport(commit=commit)
    started = time.monotonic()
    free = disk_free_gb()
    if free < DISK_MIN_GB:
        report.notes.append(f"skipped: free disk {free:.1f} GB < {DISK_MIN_GB:.0f} GB")
        report.model = "skipped: low disk" if gate else report.model
        return report
    deadline = Deadline(loop.minutes_per_run)
    with _debug_mode(f"loop-{loop.slug}-{commit[:8]}") as log:
        report.debug_log = str(log.path) if log else None
        debuglog.record("stage", stage="loop.job", loop=loop.name, commit=commit,
                        deterministic=deterministic, model=gate is not None)
        candidates: list[dict[str, Any]] = []
        try:
            parent = parent_of(project, commit)
            with worktree(project, commit) as wt:
                if deterministic:
                    candidates += _deterministic(loop, project, wt, commit, parent, deadline, report)
                if gate is not None:
                    git(wt, "checkout", "--detach", "--quiet", commit)  # lint may have moved it
                    finding = model_pass(loop, project, wt, commit, parent, deadline, gate,
                                         agent_factory, report)
                    if finding:
                        candidates.append(finding)
        except BudgetExceeded as exc:
            report.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - a loop job never takes the daemon down
            report.error = f"{type(exc).__name__}: {exc}"
        for finding in candidates:
            stored, is_new = inbox.add(finding)
            if is_new:
                report.filed.append(stored)
            else:
                report.duplicates += 1
        report.seconds = round(time.monotonic() - started, 1)
        debuglog.record("stage", stage="loop.job.done", summary=report.summary(),
                        notes=report.notes, seconds=report.seconds)
    return report


def failing_ids(output: str) -> list[str]:
    """Test ids from pytest's short summary: `FAILED tests/x.py::test_a - ...`, `ERROR tests/y.py`."""
    ids: list[str] = []
    for line in output.splitlines():
        match = _FAILED_ID.match(line.strip())
        if match and match.group("id") not in ids:
            ids.append(match.group("id"))
    return ids


def _deterministic(
    loop: LoopDef, project: Path, wt: Path, commit: str, parent: str | None,
    deadline: Deadline, report: JobReport,
) -> list[dict[str, Any]]:
    """Tests and lint at the commit; a failure counts only if the parent did not have it (tests
    failing at both are pre-existing or caused by the environment, and only noted)."""
    findings: list[dict[str, Any]] = []
    if not sandbox.available():
        report.notes.append("tests and lint not run: sandbox unavailable on this system")
        return findings
    short = commit[:12]
    test_code, test_output, failing = None, "", []
    if has_tests(wt):
        test_code, test_output = run_tests(wt, deadline)
        debuglog.record("stage", stage="loop.tests", exit_code=test_code, output=test_output)
        failing = failing_ids(test_output) if test_code in (1, 2, 3, 4) else []
    files = changed_python(project, parent, commit)
    lint_now = lint(wt, files, deadline)
    lint_before: dict[str, list[tuple[str, str]]] = {}
    still_failing: set[str] = set()
    if parent is not None and (failing or lint_now):
        git(wt, "checkout", "--detach", "--quiet", parent)
        lint_before = lint(wt, list(lint_now), deadline) if lint_now else {}
        existing = [i for i in failing if (wt / i.split("::")[0]).exists()]
        if existing:
            code, output = run_tests(wt, deadline, " ".join(f"'{i}'" for i in existing))
            debuglog.record("stage", stage="loop.tests.at_parent", exit_code=code, output=output)
            if code == 0:
                still_failing = set()
            elif code in (1, 2):
                still_failing = set(failing_ids(output)) & set(existing)
            else:  # the parent run itself broke (usage error, timeout): keep every failure
                still_failing = set()
    if test_code is None:
        report.notes.append("no tests to run")
    elif test_code == 0:
        report.notes.append("tests pass")
    elif test_code == 5:
        report.notes.append("no tests collected")
    elif test_code in (1, 2, 3, 4):
        new = [i for i in failing if i not in still_failing]
        if still_failing:
            report.notes.append(f"{len(still_failing)} test(s) also fail at the parent (not new): "
                                + ", ".join(sorted(still_failing)[:5]))
        if new or not failing:
            report.notes.append(f"tests fail (exit {test_code}): {len(new)} new failure(s)")
            summary = "\n".join(f"FAILED {i}" for i in new)
            files_failing = list(dict.fromkeys(i.split("::")[0] for i in new))
            findings.append(inbox.make_finding(
                loop=loop.name, project=project, commit=short, kind="bug",
                title=(f"{len(new)} test(s) fail at {commit[:8]}" if test_code == 1 else
                       f"test run errors at {commit[:8]} (exit {test_code})"),
                where=[{"path": path} for path in files_failing] or [{"path": "tests"}],
                evidence={"type": "failing_test", "path": " ".join(new) or "tests",
                          "output": test_output, "fails_at": short,
                          **({"passes_at": parent[:12]} if parent else {}),
                          "fingerprint": inbox.fingerprint({"type": "failing_test",
                                                            "output": summary or test_output})},
                route=_route(loop, "failing_test"), debug_log=report.debug_log))
    else:
        report.notes.append(f"tests did not finish: {test_output.splitlines()[0][:120]}")
    new_lint = {path: [e for e in errors if e[1] not in {b[1] for b in lint_before.get(path, [])}]
                for path, errors in lint_now.items()}
    new_lint = {path: errors for path, errors in new_lint.items() if errors}
    count = sum(map(len, new_lint.values()))
    debuglog.record("stage", stage="loop.lint", files=files, new_errors=count)
    report.notes.append(f"lint: {count} new error(s) in {len(files)} changed file(s)")
    for path, errors in new_lint.items():
        output = "\n".join(f"{path}:{line}: {rest}" for line, rest in errors)
        findings.append(inbox.make_finding(
            loop=loop.name, project=project, commit=short, kind="lint",
            title=f"{len(errors)} new ruff error(s) in {path}",
            where=[{"path": path, "line": int(errors[0][0])}],
            evidence={"type": "lint", "path": path, "output": output, "fails_at": short,
                      **({"passes_at": parent[:12]} if parent else {})},
            route=_route(loop, "lint"), debug_log=report.debug_log))
    return findings
