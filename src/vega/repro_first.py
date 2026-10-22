"""repro_first (opt-in): on a bug report handled by the agent loop, reproduce the bug with one
small failing test before the first edit of the project's source.

Why (2026-10-21): bug fixes on the agent loop on a larger codebase failed mostly by editing the
wrong file or not editing at all. The first write (Vega's .vega/ notes aside) was a source file
early in the turn, almost always with no check run before it, and no attempt wrote a test that
failed on the unchanged code: fixes were checked against the existing suite, never against the
report. The project pipeline already has a task-to-tests step (opt-in spec_tests, off: no
measured gain and slower), so this is for the agent loop only.

Rules only, no model call:
- applies(): a change task (agent.asks_for_change) that reads as a bug report (a symptom word:
  fix, bug, broken, error, wrong, "doesn't", "never", ...), on a Python project with a pytest
  test directory and at least MIN_SOURCE_FILES non-test .py files, that neither asks for tests
  itself nor limits which files may change; the repro file must not exist yet.
- The turn's first write to a non-test file is not run once (hold_note): the model writes
  REPRO_NAME in the test directory, Vega runs it in the sandbox when it is written (before any
  source edit; at most AUTO_RUNS times) and says whether it fails, i.e. reproduces the report.
  Repeating the held call runs it: one hold per turn, so at most one extra model step.
- finish(): at the end of the turn the repro file (when this turn created it) is run once more;
  it stays only when it passes (a regression test for the fix, visible in the diff), and is
  removed when it fails or does not run: a failing test Vega asked for must not be left in the
  user's suite (or an eval's regression run), and its failure may be the test's own mistake. It
  is a new file: the project's own tests are never replaced; the note says not to change them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import check_feedback, routing

REPRO_NAME = "test_vega_repro.py"
MIN_SOURCE_FILES = 3  # a tiny project is read whole anyway; a repro step costs more than it finds
AUTO_RUNS = 2  # sandbox runs of the repro file when it is written before any source edit
TEST_DIRS = ("tests", "test")
_TEST_PATH = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)")
# words that describe broken behaviour; strong ones count anywhere outside code
_STRONG = re.compile(
    r"\b(?:fix(?:es|ed)?|bugs?|buggy|broken|breaks?|crash(?:es|ed|ing)?|tracebacks?|"
    r"exceptions?|errors?|wrong(?:ly)?|incorrect(?:ly)?|no longer|instead of|regress(?:ion|ed)?|"
    r"fails?|failing|failed)\b", re.IGNORECASE)
# negations describe the current behaviour ("the total never includes tax") unless the sentence
# states a requirement ("The loader must never write the file"): counted outside "must" sentences
_WEAK = re.compile(r"\b(?:doesn't|does not|don't|didn't|isn't|aren't|won't|can't|cannot|never|"
                   r"only (?:says|shows|prints|returns))\b", re.IGNORECASE)
_REQUIREMENT = re.compile(r"\bmust\b", re.IGNORECASE)


def is_test_path(relative: str) -> bool:
    return bool(_TEST_PATH.search(relative.replace("\\", "/").lstrip("./")))


def bug_report(task: str) -> bool:
    """The task describes behaviour that is wrong now (outside code spans)."""
    text = routing.strip_code(task)
    if _STRONG.search(text):
        return True
    return any(_WEAK.search(sentence) and not _REQUIREMENT.search(sentence)
               for sentence in re.split(r"(?<=[.!?;:])\s+|\n", text))


def test_dir(project: Path) -> str | None:
    """The project's pytest directory (tests/ or test/ holding a test_*.py file), or None."""
    for name in TEST_DIRS:
        folder = project / name
        if folder.is_dir() and any(folder.rglob("test_*.py")):
            return name
    return None


def _source_files(project: Path) -> int:
    from .tools import _project_files

    count = 0
    for path in _project_files(project):
        relative = path.relative_to(project).as_posix()
        if path.suffix == ".py" and not is_test_path(relative):
            count += 1
            if count >= MIN_SOURCE_FILES:
                break
    return count


def applies(task: str, project: Path, route: dict[str, Any] | None,
            scoped: bool = False) -> tuple[str | None, str]:
    """(the repro file's project-relative path, why) when the step applies, else (None, why
    not). `scoped`: the finish gate read a "change only X" / "do not change other files" limit."""
    from . import agent, project_pipeline

    if not isinstance(route, dict) or not agent.asks_for_change(route, task):
        return None, "not a change task"
    if not bug_report(task):
        return None, "no bug-report words"
    if scoped or routing.read_only(task):
        return None, "the task limits which files may change"
    if project_pipeline._ASKS_FOR_TESTS.search(routing.strip_code(task)):
        return None, "the task asks for tests itself"
    folder = test_dir(project)
    if folder is None:
        return None, "no pytest test directory"
    if _source_files(project) < MIN_SOURCE_FILES:
        return None, f"fewer than {MIN_SOURCE_FILES} source files"
    path = f"{folder}/{REPRO_NAME}"
    if (project / path).exists():
        return None, f"{path} exists already"
    return path, "bug report on a project with tests"


def hold_note(path: str) -> str:
    return ("(Vega harness) Not run yet: reproduce the reported bug before changing the source. "
            f"Write ONE small pytest test in `{path}` (a new file; do not change the project's "
            "existing tests) that calls the code the report is about the way the report "
            "describes and asserts what the report says should happen. It must fail on the code "
            "as it is now; Vega runs it when you write it. Then make your fix and run "
            f"`python -m pytest -q {path}` again. If the bug cannot be shown in a test, repeat "
            "your call unchanged to go on without one.")


def command(path: str) -> str:
    return f"python -m pytest -q {path}"


def written_note(path: str, result: str) -> str | None:
    """The result of the repro run Vega made when the file was written (no source edit yet)."""
    code = check_feedback.exit_code(result)
    if code is None:
        return None  # no sandbox here, or it did not finish: nothing to say
    head = f"(Vega harness) Vega ran `{command(path)}` on the unchanged source"
    count = check_feedback.totals(result)
    counted = f" ({count})" if count else ""
    if code == 1:
        return (f"{head}: it fails{counted}, so it reproduces the report if this is the failure "
                f"the report describes:\n{check_feedback.excerpt(result, 1200)}\nNow fix the source "
                "(not the test) and run it again.")
    if code == 0:
        return (f"{head}: it passes{counted}, so it does not reproduce the report yet. Make it "
                "exercise the reported case (the input, call or command the report names), or go "
                "on with the fix if the bug cannot be shown in a test.")
    return (f"{head}: it did not run as a test (exit code {code}{counted}):\n"
            f"{check_feedback.excerpt(result, 800)}\nFix the test file itself first.")


@dataclass
class Repro:
    """One turn's repro step (agent loop)."""

    path: str
    held: bool = False  # the first source write was held (once per turn)
    test_written: bool = False  # a test file was written this turn (no hold after that)
    source_written: bool = False  # a non-test file was written this turn
    auto_runs: int = 0
    runs: list[int | None] = field(default_factory=list)  # exit codes of Vega's runs

    def should_hold(self, relative: str | None) -> bool:
        return (not self.held and not self.test_written and not self.source_written
                and bool(relative) and not is_test_path(relative or "")
                and not (relative or "").startswith(".vega/"))  # Vega's notes, not the code

    def wrote(self, relative: str | None) -> None:
        if not relative:
            return
        if is_test_path(relative):
            self.test_written = True
        else:
            self.source_written = True

    def wants_run(self, relative: str | None) -> bool:
        return (relative == self.path and not self.source_written
                and self.auto_runs < AUTO_RUNS)


def finish(project: Path, repro: Repro, stop: Any = None) -> tuple[str, str] | None:
    """End of the turn: (what happened, the run's result) for a repro file this turn created —
    'kept' when it passes, 'removed' otherwise (deleted here) — or None when there is none."""
    from . import sandbox

    target = project / repro.path
    if not target.is_file():
        return None
    result = sandbox.run_check(command(repro.path), project, stop=stop)
    if check_feedback.exit_code(result) == 0:
        return "kept", result
    try:
        target.unlink()
    except OSError:
        return "left", result
    return "removed", result
