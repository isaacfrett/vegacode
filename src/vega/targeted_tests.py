"""targeted_tests: on a project whose whole pytest suite does not fit the check timeout, run the
tests related to what changed first, and the whole suite only as a final check with a longer,
bounded timeout; a whole-suite timeout is reported as slow, not failing.

Selfbuild run 2026-10-21 (Vega on its own source, 87 test files, ~1480 tests): acceptance ran
`python -m pytest -q` with run_check's 60 s limit, reported "Check timed out after 60s." as a
failing check, and the model spent its remaining steps chasing the timeout instead of the bug.

Small projects (fewer than LARGE_TEST_FILES pytest files) are unchanged. Rules only, no model call.
"""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path

from . import features
from .tools import _project_files

LARGE_TEST_FILES = 30  # a project with at least this many pytest files gets targeted runs
MAX_FILES = 12  # at most this many test files in one targeted run
FULL_FIRST_TIMEOUT = 240  # whole suite, time unknown and no targeted run to scale from
FULL_MIN_ESTIMATED = 120  # whole suite, time only estimated from the targeted run
FULL_MAX_TIMEOUT = 600  # the whole-suite run never waits longer than this
FULL_FACTOR = 1.5  # headroom over the measured / estimated suite time
FULL_MARGIN = 15  # seconds added for the sandbox clone and interpreter start

_TEST_FILE = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py)$")
_TIMED_OUT = "Check timed out after"
# Measured whole-suite seconds per project, for this process (a TUI session keeps it across turns).
_suite_seconds: dict[str, float] = {}


def test_files(project: Path) -> list[str]:
    """The project's pytest files, relative POSIX paths, sorted."""
    return sorted(path for path in (p.relative_to(project).as_posix()
                                    for p in _project_files(project))
                  if _TEST_FILE.search(path))


def whole_suite(command: str) -> bool:
    """A pytest command that runs the whole suite: no file, directory or node id named."""
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if not argv:
        return False
    if Path(argv[0]).name == "pytest":
        rest = argv[1:]
    elif len(argv) >= 3 and Path(argv[0]).name.startswith("python") and argv[1:3] == ["-m", "pytest"]:
        rest = argv[3:]
    else:
        return False
    skip_value = {"-k", "-m", "-p", "-c", "--tb", "-o", "--maxfail", "--rootdir", "--durations",
                  "--deselect", "--ignore", "--ignore-glob", "-W"}
    expect_value = False
    for arg in rest:
        if expect_value:
            expect_value = False
            continue
        if arg in skip_value:
            expect_value = True
            continue
        if not arg.startswith("-"):
            return False
    return True


def applies(project: Path, command: str | None) -> bool:
    """targeted_tests on, a whole-suite pytest command, and a large suite."""
    return bool(command and features.on("targeted_tests") and whole_suite(command)
                and len(test_files(project)) >= LARGE_TEST_FILES)


def _module(path: str) -> str:
    module = path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
    return module.removeprefix("src.")


def _imports(text: str) -> set[str]:
    """Dotted modules a test file imports (`from a.b import c` gives a.b and a.b.c)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def select(project: Path, changed: list[str]) -> tuple[list[str], int]:
    """(test files to run first, how many were related before the MAX_FILES cap). Order: test
    files written this turn, tests named after a changed module (test_tools*.py for tools.py),
    tests importing a changed module (the most focused first: fewest project imports), tests that
    mention a changed non-Python file by name."""
    tests = test_files(project)
    texts: dict[str, str] = {}

    def text(path: str) -> str:
        if path not in texts:
            try:
                texts[path] = (project / path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                texts[path] = ""
        return texts[path]

    ordered: list[str] = []

    def add(paths: list[str]) -> None:
        ordered.extend(p for p in paths if p not in ordered)

    add([p for p in changed if _TEST_FILE.search(p) and (project / p).is_file()])
    sources = [p for p in changed if p.endswith(".py") and not _TEST_FILE.search(p)]
    stems = {Path(p).stem for p in sources if Path(p).stem != "__init__"}
    add([t for t in tests if any(Path(t).stem == f"test_{s}" or Path(t).stem.startswith(f"test_{s}_")
                                 or Path(t).stem == f"{s}_test" for s in stems)])
    modules = {_module(p) for p in sources}
    importers = []
    for test in tests:
        imported = _imports(text(test))
        if modules & imported:
            roots = {m.split(".")[0] for m in modules}
            breadth = len({m for m in imported if m.split(".")[0] in roots})
            importers.append((breadth, test))
    add([t for _, t in sorted(importers)])
    others = [p for p in changed if not p.endswith(".py")]
    names = {Path(p).name for p in others} | set(others)
    add([t for t in tests if any(name in text(t) for name in names)])
    return ordered[:MAX_FILES], len(ordered)


def command_for(command: str, files: list[str]) -> str:
    return f"{command} " + " ".join(shlex.quote(f) for f in files)


def key(project: Path) -> str:
    return str(project.resolve())


def record_suite(project: Path, seconds: float) -> None:
    _suite_seconds[key(project)] = seconds


def full_timeout(project: Path, targeted: tuple[float, int] | None = None) -> int:
    """The whole-suite timeout: FULL_FACTOR x its last measured time; else scaled from the
    targeted run (seconds for n of the project's test files; at least FULL_MIN_ESTIMATED, since
    test files differ widely in cost); else FULL_FIRST_TIMEOUT. Never below sandbox.TIMEOUT,
    never above FULL_MAX_TIMEOUT."""
    from .sandbox import TIMEOUT

    measured = _suite_seconds.get(key(project))
    floor = TIMEOUT
    if measured is None and targeted and targeted[1] > 0:
        measured = targeted[0] * len(test_files(project)) / targeted[1]
        floor = FULL_MIN_ESTIMATED
    if measured is None:
        return FULL_FIRST_TIMEOUT
    return int(min(FULL_MAX_TIMEOUT, max(floor, measured * FULL_FACTOR + FULL_MARGIN)))


def timed_out(output: str) -> bool:
    return output.startswith(_TIMED_OUT)


def run_check_hint(command: str, project: Path, timeout: float) -> str:
    """The note after run_check's timeout message when a large suite's whole run did not fit:
    a time limit, not a test failure, and what to run instead ('' when targeted_tests does not
    apply)."""
    try:
        if not applies(project, command):
            return ""
        count = len(test_files(project))
    except OSError:
        return ""
    return (f" This is run_check's time limit, not a test failure: this project has {count} test "
            f"files and its whole suite does not finish in {int(timeout)}s. Run the test files "
            f"for what you changed instead (`{command} tests/test_<module>.py ...`); Vega runs "
            "the whole suite itself at finish with a longer limit.")
