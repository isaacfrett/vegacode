"""baseline_aware_gate (opt-in): tests that already fail before Vega's first edit are not the
change's failures.

Why (eval diagnosis 2026-10-22): when a project's suite is already red before Vega edits anything,
the finish gate, verify_before_finish and the model's own test runs showed every failure as
something to fix, so Vega edited unrelated files chasing failures that were there before it
started (scored wrong_file / regression).

With the switch on, the agent records which tests fail before its first edit in the session:
the model's own whole-suite pytest run made before any edit is reused; otherwise Vega runs the
project's test command once in the sandbox, at the first write (or before a shell command that
may write files), with the sandbox's usual timeout. Afterwards each pytest result is split into:

- new failures: failing now and not failing at the baseline (a test that went from passing to
  failing is always one of these; so is any test the baseline did not name);
- targeted: failing at the baseline and named by the task (its file, module, test name, or a
  code name of the task inside the test's name), still shown as work to do;
- pre-existing: failing at the baseline and not named by the task. Listed once by id, then
  referred to as "listed earlier"; never pushed as "fix these".

A result whose failures are all pre-existing counts as passing for the finish gate only when the
run is complete: it went to the end (exit code 1, not stopped early or interrupted) and every
failure in pytest's count line is named in its summary. Anything that cannot be read that way
(not pytest, a timeout, a clipped summary, a collection error that stopped the run) keeps the old
behaviour.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field

from . import check_feedback, targeted_tests

SHOWN = 6  # test ids listed in one note
_SUMMARY = re.compile(r"^(FAILED|ERROR) (?!collecting\b)(.+?)(?: - .*)?\s*$")
_COUNT = re.compile(r"(\d+) (failed|errors?)\b")
_STOPPED = re.compile(r"!{3,} (?:stopping after|Interrupted:|KeyboardInterrupt)|^Interrupted: ",
                      re.MULTILINE)
# selection flags: a run with these is not the whole suite even though it names no path
_SELECTING = {"-k", "-m", "-x", "--exitfirst", "--lf", "--last-failed", "--ff", "--sw",
              "--stepwise", "--deselect", "--ignore", "--ignore-glob", "--maxfail"}
_CODE_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_GENERIC = {"test", "tests", "self", "none", "true", "false", "pytest", "python", "init", "main",
            "conftest", "assert", "print", "return", "import"}


@dataclass
class Run:
    """A pytest result's failures."""

    ids: list[str]  # failing test ids (and test files that failed to load), in order, unique
    complete: bool  # every failure in the count line is named in the summary
    finished: bool  # the run went to the end (exit code 0/1/5, not stopped early)


def parse(output: str) -> Run | None:
    """The failures a pytest run_check/run_command result names; None when the result is not
    one that names them (it did not run, or failed without a pytest summary)."""
    code = check_feedback.exit_code(output)
    if code is None:
        return None
    if code in (0, check_feedback.NO_TESTS):
        return Run([], True, True)
    lines = []
    for line in output.splitlines():
        match = _SUMMARY.match(line.rstrip())
        if match:
            lines.append(match.group(2).strip())
    if not lines:
        return None
    ids = list(dict.fromkeys(lines))
    counted = sum(int(n) for n, _ in _COUNT.findall(check_feedback.totals(output)))
    complete = counted > 0 and len(lines) >= counted
    finished = code == 1 and not _STOPPED.search(output)
    return Run(ids, complete, finished)


def suite_command(command: str) -> bool:
    """A pytest command that runs the whole suite (no path, node id or selection flag)."""
    if not targeted_tests.whole_suite(command):
        return False
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    rest = argv[argv.index("pytest") + 1:] if "pytest" in argv else argv[1:]
    return not any(arg.split("=", 1)[0] in _SELECTING for arg in rest)


@dataclass
class Baseline:
    """The tests failing before Vega's first edit (failing is empty when the suite passed)."""

    command: str
    failing: frozenset[str]
    how: str  # "reused": the model's own run before any edit; "run": Vega ran it; "rerun": lazy
    reported: set[str] = field(default_factory=set)  # pre-existing ids already listed to the model
    # A lazy baseline (an extension's test_baseline, for a suite too slow to run before the first
    # edit): `resolve(ids)` returns which of these test ids fail on the project as it was before
    # the first edit (None: could not tell). Each id is resolved once; `checked` holds them.
    resolve: Callable[[list[str]], set[str] | None] | None = None
    checked: set[str] = field(default_factory=set)

    def settle(self, ids: list[str]) -> None:
        """A lazy baseline: find out, once per id, whether these failing ids failed before the
        first edit. An id that cannot be told stays new (shown as work, the old behaviour)."""
        if self.resolve is None:
            return
        unknown = [i for i in ids if i not in self.checked and i not in self.failing]
        if not unknown:
            return
        self.checked.update(unknown)
        found = self.resolve(unknown)
        if found:
            self.failing = self.failing | frozenset(i for i in found if i in unknown)


def from_output(command: str, output: str, how: str) -> Baseline | None:
    """The baseline a whole-suite result gives; None when it names no result."""
    run = parse(output)
    if run is None:
        return None
    return Baseline(command, frozenset(run.ids), how)


def _code_words(task: str) -> set[str]:
    """Lower-cased code names the task uses: inside backticks, `name(`, `snake_case`, `x.py`."""
    words: set[str] = set()
    for span in re.findall(r"`([^`\n]{1,200})`", task):
        words.update(_CODE_WORD.findall(span))
    words.update(re.findall(r"([A-Za-z_]\w*)\s*\(", task))
    words.update(w for w in _CODE_WORD.findall(task) if "_" in w.strip("_"))
    words.update(re.findall(r"([A-Za-z_]\w*)\.py\b", task))
    return {w.lower() for w in words if len(w) >= 3 and w.lower() not in _GENERIC}


def targets(test_id: str, task: str) -> bool:
    """Whether the task is about this test: it names the test's file or module, the test (or
    its class), or a code name of the task appears as a whole part of the test's name."""
    path, _, rest = test_id.partition("::")
    lowered = task.lower()
    filename = path.rsplit("/", 1)[-1].lower()
    if path.lower() in lowered or (filename.endswith(".py") and filename in lowered):
        return True
    module = filename.removesuffix(".py").removeprefix("test_").removesuffix("_test")
    words = _code_words(task)
    if module and module in words:
        return True
    for part in (p.split("[", 1)[0].lower() for p in rest.split("::") if p):
        if part in words:
            return True
        name = part.removeprefix("test_").removeprefix("test")
        if any(re.search(rf"(?:^|_){re.escape(w)}(?:_|$)", name) for w in words):
            return True
    return False


@dataclass
class Split:
    new: list[str]  # failing now, not at the baseline
    targeted: list[str]  # failing at the baseline, named by the task
    pre: list[str]  # failing at the baseline, not named by the task
    run: Run

    @property
    def only_pre(self) -> bool:
        """Every failure is pre-existing and the run shows that completely."""
        return (bool(self.pre) and not self.new and not self.targeted and self.run.complete
                and self.run.finished)


def classify(baseline: Baseline | None, output: str, task: str) -> Split | None:
    """The result's failures split against the baseline; None without a baseline or ids."""
    if baseline is None:
        return None
    run = parse(output)
    if run is None or not run.ids:
        return None
    baseline.settle(run.ids)
    new = [i for i in run.ids if i not in baseline.failing]
    old = [i for i in run.ids if i in baseline.failing]
    targeted = [i for i in old if targets(i, task)]
    pre = [i for i in old if i not in targeted]
    return Split(new, targeted, pre, run)


def listed(ids: list[str]) -> str:
    shown = ", ".join(ids[:SHOWN])
    return shown + (f" (+{len(ids) - SHOWN} more)" if len(ids) > SHOWN else "")


def pre_text(baseline: Baseline, pre: list[str]) -> str:
    """The pre-existing failures: their ids the first time, "listed earlier" after that."""
    unreported = [i for i in pre if i not in baseline.reported]
    baseline.reported.update(pre)
    if unreported:
        return f"{len(pre)} test(s) that already failed before your first edit: {listed(pre)}"
    return f"{len(pre)} test(s) that already failed before your first edit (listed earlier)"


def note(baseline: Baseline, split: Split, before_edit: bool = False) -> str:
    """The line appended to a test result the model sees ('' when nothing is pre-existing)."""
    if not split.pre:
        return ""
    if before_edit:
        first = (f"(Vega harness) Before any edit, {pre_text(baseline, split.pre)}. They are "
                 "pre-existing: fix them only if the task is about them; do not edit other files "
                 "to chase them. Vega will not count them against your change.")
    else:
        first = (f"(Vega harness) Pre-existing, not caused by your changes: "
                 f"{pre_text(baseline, split.pre)}. Fix them only if the task is about them; do "
                 "not edit other files to chase them.")
    lines = [first]
    if split.new:
        lines.append(f"New since your first edit (fix these): {listed(split.new)}.")
    if split.targeted:
        lines.append(f"Failing before your edit and named by the task (fix these): "
                     f"{listed(split.targeted)}.")
    return "\n".join(lines)


def finish_report(baseline: Baseline, split: Split, ran: str) -> str:
    """At finish, when every failure is pre-existing: reported once, not sent back as work.
    `ran`: what Vega ran, as shown (a command in backticks)."""
    return (f"(Vega harness) Before you finish: Vega ran {ran}. No test that passed before "
            f"your first edit fails now; the only failures are {pre_text(baseline, split.pre)}. "
            "They are not caused by your change. Give your final answer now, and mention them as "
            "pre-existing failures rather than claiming every test passes.")


def item_detail(baseline: Baseline, split: Split, detail: str) -> str:
    """A failing acceptance tests item's detail, its first line naming only what is new."""
    head, _, body = detail.partition("\n")
    work = split.new + split.targeted
    first = (f"{head}: {len(work)} failure(s) to fix: {listed(work)}" if work else head)
    extra = f"(Also failing, pre-existing: {pre_text(baseline, split.pre)}; not caused by you.)" \
        if split.pre else ""
    return "\n".join(part for part in (first, extra, body) if part)
