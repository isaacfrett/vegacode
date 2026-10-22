"""Check results as feedback for a small model: which command checks a project, a failing run
condensed for an 8k window, and whether a final answer's success claim matches the last run.

Why (failures seen with Granite 4.1 8B, 2026-10-15/16): after a failing check the agent rarely
recovered; many failed
attempts ended claiming success ("tests pass", "verified"); many agent-loop attempts ended after
their last edit without running anything. The harness never decides for the model: it runs the
check, shows the result and says what contradicts the answer; the model chooses what to do.
Switches: verify_before_finish, repair_feedback, claim_check (agent.py uses these helpers).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import features, sandbox
from .tools import _project_files

# -- which command checks this project -----------------------------------------------------------

_TEST_FILE = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py)$")
# A backticked command the task says to run as the check ("Run the tests with `python check.py`",
# "`pytest -q` must pass", "check with `go test ./...`").
_TASK_COMMAND = re.compile(
    r"\b(?:run|re-?run|check|verify|test)\w*\b[^`.\n]{0,40}?(?:with|using|via|by running|:)?\s*"
    r"`([^`\n]{2,120})`"
    r"|`([^`\n]{2,120})`\s*(?:must|should|has to|needs to)\s+(?:still\s+)?pass",
    re.IGNORECASE)
_RUNNERS = re.compile(
    r"^(?:pytest\b|python3? -m (?:pytest|unittest)\b|python3? [\w./-]*(?:test|check)[\w./-]*\.py\b"
    r"|go test\b|cargo test\b|npm test\b|make (?:test|check)\b|swift test\b|ruff check\b"
    r"|mypy \S|pyright\b|npx tsc --noEmit\b)")
_PLACEHOLDER = re.compile(r"<[^>]*>|\b[A-Z][A-Z_]{2,}\b|\.\.\.")


def task_command(project: Path, task: str) -> str | None:
    """A test/check command the task names as the way to check the work, if run_check runs it."""
    for match in _TASK_COMMAND.finditer(task):
        command = (match.group(1) or match.group(2) or "").strip()
        if not _RUNNERS.match(command) or _PLACEHOLDER.search(command):
            continue
        if command.startswith(("python ", "python3 ")) and " -m " not in f" {command} ":
            script = command.split()[1]
            if not (project / script).is_file():
                continue
        if isinstance(sandbox.parse_command(command, project), list):
            return command
    return None


def _pytest_config(project: Path) -> bool:
    if (project / "pytest.ini").is_file() or (project / "conftest.py").is_file():
        return True
    for name, marker in (("pyproject.toml", "[tool.pytest"), ("setup.cfg", "[tool:pytest]"),
                         ("tox.ini", "[pytest]")):
        path = project / name
        try:
            if path.is_file() and marker in path.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def project_command(project: Path, task: str = "") -> str | None:
    """The command that checks this project now: one the task names, else pytest (config or test
    files), an npm test script, a Makefile test/check target, go test, cargo test; None if none."""
    named = task_command(project, task) if task else None
    if named:
        return named
    files = [p.relative_to(project).as_posix() for p in _project_files(project)]
    if _pytest_config(project) or any(_TEST_FILE.search(path) for path in files):
        return "python -m pytest -q"
    package = project / "package.json"
    if package.is_file():
        try:
            script = str(json.loads(package.read_text(encoding="utf-8")).get("scripts", {})
                         .get("test", ""))
        except (OSError, ValueError, AttributeError):
            script = ""
        if script and "no test specified" not in script:
            return "npm test"
    makefile = project / "Makefile"
    if makefile.is_file():
        text = makefile.read_text(encoding="utf-8", errors="replace")
        for target in ("test", "check"):
            if re.search(rf"^{target}\s*:", text, re.MULTILINE):
                return f"make {target}"
    if (project / "go.mod").is_file():
        return "go test ./..."
    if (project / "Cargo.toml").is_file():
        return "cargo test"
    return None


_PYTEST_FAMILY = re.compile(r"(?:^|\s)(?:pytest|-m pytest)\b")


def short_tracebacks(command: str) -> str:
    """short_tracebacks: a pytest command without a --tb option gets --tb=short, so its failures
    keep their assertion lines within the sandbox's output limit (2026-10-20: pytest's
    default long tracebacks through argparse, or for a dozen failures, did not fit)."""
    if (features.on("short_tracebacks") and _PYTEST_FAMILY.search(command)
            and not re.search(r"(?:^|\s)--tb\b|--full-trace", command)):
        return f"{command} --tb=short"
    return command


def covers(ran: str, check: str) -> bool:
    """Whether a command the model ran counts as running `check` (any pytest run counts for
    pytest; otherwise the same program and subcommand)."""
    if _PYTEST_FAMILY.search(check):
        return bool(_PYTEST_FAMILY.search(ran))
    return ran.split()[:2] == check.split()[:2]


# -- exit codes ------------------------------------------------------------------------------------

_EXIT = re.compile(r"^exit code: (-?\d+)")
NO_TESTS = 5  # pytest: no tests were collected; not a failure of the code


def exit_code(result: str) -> int | None:
    """The exit code of a run_check/run_command result; None when it did not run (refused,
    timed out, stopped)."""
    match = _EXIT.match(result)
    return int(match.group(1)) if match else None


def failed(result: str) -> bool:
    code = exit_code(result)
    return code is not None and code not in (0, NO_TESTS)


# -- condensing a failing run (repair_feedback) ----------------------------------------------------

_SECTION = re.compile(r"^=+ (.+?) =+$")
_BLOCK = re.compile(r"^_{3,} (.+?) _{3,}$")
_SUMMARY_ID = re.compile(r"^(FAILED|ERROR) (\S+)(?: - (.*))?$")
_TOTALS = re.compile(r"\b\d+ (?:failed|passed|errors?|error)\b.*\bin [\d.]+s\b")
_FRAME = re.compile(r"^(\S+?\.py):(\d+): ?(?:in (\S+)|(\w*))?\s*$")
_FIXTURE_FRAME = re.compile(r"^file (\S+?\.py), line (\d+)$")
_PY_FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+)(?:, in (\S+))?')
_CAPTURED = re.compile(r"^-+ Captured (?:stdout|stderr|log) \w+ -+$")
_CUT = re.compile(r"^\.\.\. \[[\d,]+ characters of output cut here\] \.\.\.$")  # sandbox.CUT_LINE
MAX_BLOCKS = 5
MAX_E_LINES = 6
MAX_LINE = 220
MAX_CONDENSED = 2600
MIN_TO_CONDENSE = 900  # shorter output is already concise


def _own(path: str) -> bool:
    """A frame in the project's own code (the sandbox shows the project as `.`), not the
    standard library, site-packages or importlib's frozen frames."""
    return not (path.startswith(("/", "<", "\\")) or "site-packages" in path
                or re.match(r"^[A-Za-z]:\\", path))


def _clip(line: str) -> str:
    line = line.rstrip()
    return line if len(line) <= MAX_LINE else line[:MAX_LINE - 1] + "…"


@dataclass
class _Frame:
    path: str
    line: int
    func: str
    source: str = ""


@dataclass
class _Block:
    title: str
    frames: list[_Frame] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    captured: list[str] = field(default_factory=list)


def _parse_block(title: str, lines: list[str]) -> _Block:
    block = _Block(title)
    pending_gt = ""  # the `>` line of a long-style frame, before its location line
    capture = 0
    for index, raw in enumerate(lines):
        line = raw.rstrip()
        if _CAPTURED.match(line):
            capture = 5
            continue
        if capture:
            if line.strip():
                block.captured.append(_clip(line))
                capture -= 1
            continue
        if line.startswith(">"):
            if not line.startswith(">       available fixtures") and "pytest --fixtures" not in line:
                pending_gt = line[1:].strip()
            continue
        if line.startswith("E "):
            text = line[1:].strip()
            if text and len(block.errors) < MAX_E_LINES:
                block.errors.append(_clip(text))
            continue
        frame = _FRAME.match(line) or _FIXTURE_FRAME.match(line)
        if frame:
            path = frame.group(1).removeprefix("./")
            func = (frame.group(3) or "") if frame.re is _FRAME else ""
            source = pending_gt
            if not source:  # short style: the source line follows the location
                following = lines[index + 1].strip() if index + 1 < len(lines) else ""
                if following and not _FRAME.match(following) and not following.startswith(
                        ("E ", "_ _")):
                    source = following
            if frame.re is _FIXTURE_FRAME and index + 1 < len(lines):
                source = lines[index + 1].strip()
            block.frames.append(_Frame(path, int(frame.group(2)), func, _clip(source)))
            pending_gt = ""
    return block


def _block_id(block: _Block, ids: dict[str, tuple[str, str]]) -> str:
    """The summary id (`FAILED tests/x.py::test_y`) of a block titled `test_y` / `Cls.test_y` /
    `ERROR at setup of test_y` / `ERROR collecting tests/x.py`."""
    title = block.title
    name = title
    for prefix in ("ERROR at setup of ", "ERROR at teardown of ", "ERROR collecting "):
        name = name.removeprefix(prefix)
    tail = "::" + name.replace(".", "::")
    for test_id, (kind, _) in ids.items():
        if test_id == name or test_id.endswith((tail, "::" + name)):
            return f"{kind} {test_id}"
    kind = "ERROR " if title.startswith("ERROR") else "FAILED "
    test_file = next((f.path for f in block.frames if _own(f.path) and _TEST_FILE.search(f.path)),
                     "")
    if test_file and not title.startswith("ERROR collecting"):
        return kind + test_file + tail  # the summary was cut off: rebuild the id
    return kind + name


def _render_block(block: _Block, ids: dict[str, tuple[str, str]]) -> list[str]:
    out = [_block_id(block, ids)]
    own = [frame for frame in block.frames if _own(frame.path)]
    if len(own) > 3:
        own = [own[0], *own[-2:]]
    for frame in own:
        where = f"{frame.path}:{frame.line}" + (f" in {frame.func}" if frame.func else "")
        out.append(f"  {where}: {frame.source}" if frame.source else f"  {where}")
    if block.frames and not _own(block.frames[-1].path):
        last = block.frames[-1]  # raised inside a library: say where, without its source
        out.append(f"  (raised in {Path(last.path).name}:{last.line})")
    out += [f"  E {line}" for line in block.errors]
    if block.captured:
        out.append("  captured output:")
        out += [f"    {line}" for line in block.captured]
    return out


def _condense_pytest(lines: list[str]) -> list[str] | None:
    section = ""
    blocks: list[_Block] = []
    current: tuple[str, list[str]] | None = None
    ids: dict[str, tuple[str, str]] = {}  # test id -> (FAILED/ERROR, short message)
    totals = ""
    warnings = 0
    for line in lines:
        if _CUT.match(line):  # output_head_tail: the block before the cut ends there; what
            if current:       # follows up to the next header belongs to an unseen block
                blocks.append(_parse_block(*current))
                current = None
            continue
        header = _SECTION.match(line)
        if header and not _BLOCK.match(line):
            if current:
                blocks.append(_parse_block(*current))
                current = None
            section = header.group(1).lower()
            if _TOTALS.search(line):
                totals = header.group(1)
            continue
        if section in ("failures", "errors"):
            block = _BLOCK.match(line)
            if block and not line.startswith("_ _"):
                if current:
                    blocks.append(_parse_block(*current))
                current = (block.group(1), [])
                continue
            if current:
                current[1].append(line)
            continue
        if section.startswith("warnings summary"):
            warnings += 1 if line.strip() and not line.startswith(" ") else 0
            continue
        summary = _SUMMARY_ID.match(line)
        if summary:
            ids[summary.group(2)] = (summary.group(1), summary.group(3) or "")
        elif _TOTALS.search(line):
            totals = line.strip("= ").strip()
    if current:
        blocks.append(_parse_block(*current))
    if not blocks and not ids:
        return None
    out = [totals] if totals else []
    shown = 0
    listed: set[str] = set()
    rest: list[str] = []
    previous: list[str] | None = None
    for block in blocks:
        name = _block_id(block, ids)
        listed.add(name.split(" ", 1)[1])
        if previous is not None and block.errors and block.errors == previous:
            out.append(f"{name} (same error as above)")  # e.g. one missing fixture, five tests
            continue
        if shown >= MAX_BLOCKS:
            rest.append(name + (f" - {block.errors[-1]}" if block.errors else ""))
            continue
        out += _render_block(block, ids)
        previous = block.errors
        shown += 1
    rest += [f"{kind} {test_id}" + (f" - {message}" if message else "")
             for test_id, (kind, message) in ids.items() if test_id not in listed]
    if rest:
        out.append(f"Also failing ({len(rest)}):" if shown else "Failing:")
        out += [f"  {_clip(line)}" for line in rest[:20]]
        if len(rest) > 20:
            out.append(f"  … and {len(rest) - 20} more")
    if warnings:
        out.append(f"({warnings} warning(s) not shown)")
    return out


def _condense_traceback(lines: list[str]) -> list[str] | None:
    """A plain Python traceback: the project's own frames and the exception."""
    start = next((i for i, line in enumerate(lines)
                  if line.startswith("Traceback (most recent call last)")), None)
    if start is None:
        return None
    out = [line for line in lines[:start] if line.strip()][-5:]
    out.append("Traceback (your code's frames only):")
    frames = 0
    for index in range(start + 1, len(lines)):
        frame = _PY_FRAME.match(lines[index])
        if frame and _own(frame.group(1).removeprefix("./")):
            source = lines[index + 1].strip() if index + 1 < len(lines) else ""
            out.append(f"  {frame.group(1).removeprefix('./')}:{frame.group(2)}"
                       + (f" in {frame.group(3)}" if frame.group(3) else "")
                       + (f": {_clip(source)}" if source else ""))
            frames += 1
    tail = [line for line in lines[start + 1:] if line.strip() and not line.startswith(" ")]
    out += [_clip(line) for line in tail[-3:]]
    return out


def pytest_rerun(test_id: str) -> str:
    return f"python -m pytest -q --tb=long '{test_id}'"


def condense(result: str, command: str = "", note: bool = True) -> str:
    """A failing check's output cut to what a repair needs: the failing test ids, each one's
    assertion/exception lines (`E ...`), the failing line of the test and the nearest line of the
    project's own code from the traceback; library frames, fixture lists and pytest banners
    dropped. Short output, passing runs and runs narrowed to one test are returned unchanged."""
    if len(result) < MIN_TO_CONDENSE or not failed(result):
        return result
    if "::" in command or re.search(r"--tb=(?:long|auto|native)|--full-trace|-vv\b", command):
        return result  # the model asked for one test's whole traceback
    first, _, body = result.partition("\n")
    lines = body.splitlines()
    kept = _condense_pytest(lines) or _condense_traceback(lines)
    if not kept:
        return result
    text = "\n".join([first, *kept])
    if len(text) > MAX_CONDENSED:
        text = text[:MAX_CONDENSED].rsplit("\n", 1)[0] + "\n…"
    if len(text) >= len(result):
        return result
    if note:
        failing = re.search(r"^(?:FAILED|ERROR) (\S+\.py(?:::\S+)?)", text, re.MULTILINE)
        more = (f"For one test's whole traceback: run_check `{pytest_rerun(failing.group(1))}`."
                if failing else "Run a narrower command to see the whole output.")
        text += f"\n(Vega condensed this output from {len(result):,} characters. {more})"
    return text


# -- success claims (claim_check) ------------------------------------------------------------------

# The quality metrics count these as overclaims in failed attempts.
CLAIMS = re.compile(
    r"\b(?:all\s+)?(?:the\s+)?tests?\s+(?:now\s+|all\s+|still\s+)?pass(?:es|ed|ing)?\b|\bverified\b"
    r"|\bconfirmed\b|\bworks\s+(?:now|correctly|as expected)\b"
    r"|\bsuccessfully\s+(?:implemented|fixed|completed|tested)\b",
    re.IGNORECASE)
_NEGATED = re.compile(r"\b(?:not|n't|never|no longer|cannot|could not|unable to|unverified|"
                      r"still fail|fail(?:s|ed|ing)?)\b", re.IGNORECASE)


def success_claim(answer: str) -> str | None:
    """The phrase with which the answer says the work is checked ("tests pass", "verified",
    "works correctly"), from a sentence that does not negate or qualify it ("the tests do not
    pass", "could not verify"); None when it claims nothing."""
    for sentence in re.split(r"(?<=[.!?\n])\s+", answer or ""):
        match = CLAIMS.search(sentence)
        if match and not _NEGATED.search(sentence):
            return match.group(0)
    return None


def totals(result: str) -> str:
    """pytest's closing count line ("2 failed, 3 passed in 0.10s"), or ''."""
    for line in reversed(result.splitlines()):
        if _TOTALS.search(line):
            return line.strip("= ").strip()
    return ""


def excerpt(result: str, limit: int = 1500) -> str:
    """A failing result for a note, without its exit-code line: condensed (repair_feedback) and
    clipped at the end, else its last `limit` characters (where pytest's summary is)."""
    if features.on("repair_feedback"):
        body = condense(result, note=False).partition("\n")[2].strip()
        if len(body) > limit:
            body = body[:limit].rsplit("\n", 1)[0] + "\n…"
    else:
        body = result.partition("\n")[2].strip()
        if len(body) > limit:
            body = "…\n" + body[-limit:].split("\n", 1)[-1]
    return body or "(no output)"


def verify_note(command: str, result: str) -> str | None:
    """verify_before_finish: the result of the check Vega ran before accepting the answer."""
    head = f"(Vega harness) Before you finish: Vega ran `{command}` on your changes"
    code = exit_code(result)
    if code is None:
        if result.startswith("Check timed out"):
            return (f"{head}, but it did not finish ({result.strip()}). Something may loop or wait "
                    "forever. Fix it, or say in your answer that this is unresolved.")
        return None  # could not run here (no sandbox, program missing): the answer stands
    count = totals(result)
    if code == 0:
        return (f"{head} and it passed" + (f" ({count})" if count else "") + ". Give your final "
                "answer now; you can cite this run.")
    if code == NO_TESTS:
        return None
    body = excerpt(result)
    if count and body.startswith(count + "\n"):
        body = body[len(count) + 1:]  # already in the first line
    return (f"{head} → exit code {code}" + (f" ({count})" if count else "") + f":\n{body}\n"
            "Fix the failures, or explain in your answer what is left.")


def claim_note(claim: str, last: tuple[int, str, str] | None) -> str:
    """claim_check: the answer claims a check, but the last run after the last write failed (or
    nothing ran after it)."""
    if last is None:
        return (f'(Vega harness) Your answer says "{claim}", but nothing was run after your last '
                "edit, so no result shows that. Run the check, or correct the answer.")
    _, command, result = last
    code = exit_code(result)
    what = "found no tests" if code == NO_TESTS else "failed"
    return (f'(Vega harness) Your answer says "{claim}", but the last run (`{command}`) {what}:\n'
            f"{excerpt(result, 800)}\nCorrect the answer or keep fixing.")
