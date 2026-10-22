"""The Tests pane: the last test run seen this session, its counts and its failing tests.

A run comes from Vega's own run_check / run_command results whose command is a test runner, or
from `r` in the pane (the same command again through the sandbox, stoppable with Esc).

Design (2026-10-15): the failed count is the one bold element (error colour); passed is in the
success colour, everything else dim. Each failing test is a two-line row like a Loops row: the
test id (its end kept when clipped, since the test name is what matters) and the first line of
its message. Option ids: `Tsummary` the run, `T<n>` a failing test.
"""

from __future__ import annotations

import contextlib
import re
import shlex
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

from rich.cells import cell_len
from rich.text import Text
from textual.widgets import OptionList
from textual.widgets.option_list import Option

INDENT = "  "  # see tui_loops.INDENT: leading spaces after a newline are dropped
EMPTY = "No test run yet. Test commands Vega runs (pytest, npm test, go test, cargo test) show here."
COUNT_WORDS = ("failed", "passed", "skipped", "errors", "xfailed", "xpassed")


@dataclass
class Failure:
    test: str  # pytest node id, go test name, cargo test path, ...
    message: str = ""


@dataclass
class TestRun:
    command: str
    output: str
    exit_code: int | None = None
    counts: dict[str, int] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)
    duration: float | None = None  # the runner's own figure when it prints one, else wall time
    wall: float | None = None
    turn: int | None = None  # None: run from the pane
    at: float = field(default_factory=time.time)
    stopped: bool = False

    @property
    def ok(self) -> bool:
        if self.stopped:
            return False
        if self.counts.get("failed") or self.counts.get("errors"):
            return False
        if self.exit_code is not None:
            return self.exit_code == 0
        return not self.failures

    def exit_line(self) -> str:
        for line in self.output.splitlines():
            if line.strip():
                return line.strip()
        return "(no output)"


def runner(command: str) -> str | None:
    """The test runner a command invokes (`pytest`, `npm`, `go`, `cargo`, ...), else None."""
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    words = [w for w in words if "=" not in w or w.startswith("-")]  # FOO=1 pytest
    if not words:
        return None
    first = words[0].rsplit("/", 1)[-1]
    rest = words[1:]
    if first in {"pytest", "py.test"} or first.startswith("pytest"):
        return "pytest"
    if first.startswith("python") or first == "uv":
        if "pytest" in rest:
            return "pytest"
        if "unittest" in rest:
            return "unittest"
        return None
    if first in {"npm", "yarn", "pnpm", "bun"}:
        if rest[:1] == ["test"] or rest[:2] in (["run", "test"],) or rest[:1] == ["t"]:
            return first
        return None
    if first == "npx" and rest[:1] and rest[0] in {"jest", "vitest", "mocha"}:
        return rest[0]
    if first in {"jest", "vitest", "mocha", "tox", "nox"}:
        return first
    if first == "go" and rest[:1] == ["test"]:
        return "go"
    if first == "cargo" and rest[:1] in (["test"], ["nextest"]):
        return "cargo"
    if first == "make" and rest[:1] == ["test"]:
        return "make"
    if first in {"swift", "dotnet"} and rest[:1] == ["test"]:
        return first
    if first in {"mvn", "gradle", "./gradlew", "gradlew"} and "test" in rest:
        return first
    return None


_EXIT = re.compile(r"^exit code: (-?\d+)", re.MULTILINE)
_PYTEST_SUMMARY = re.compile(
    r"^=*\s*((?:\d+ (?:failed|passed|skipped|errors?|xfailed|xpassed|deselected|warnings?)"
    r"(?:, )?)+)\s+in ([\d.]+)s", re.MULTILINE)
_PYTEST_NONE = re.compile(r"^=*\s*no tests ran in ([\d.]+)s", re.MULTILINE)
_PYTEST_FAILED = re.compile(r"^(FAILED|ERROR) (\S+)(?: - (.*))?$", re.MULTILINE)
_GO_FAIL = re.compile(r"^\s*--- FAIL: (\S+)", re.MULTILINE)
_GO_PASS = re.compile(r"^\s*--- PASS: ", re.MULTILINE)
_GO_SKIP = re.compile(r"^\s*--- SKIP: ", re.MULTILINE)
_GO_DURATION = re.compile(r"^(?:ok|FAIL)\s+\S+\s+([\d.]+)s", re.MULTILINE)
_CARGO = re.compile(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored.*?"
                    r"finished in ([\d.]+)s")
_CARGO_FAIL = re.compile(r"^test (\S+) \.\.\. FAILED", re.MULTILINE)
_JEST = re.compile(r"^Tests:\s+(.*?)(\d+) total", re.MULTILINE)
_VITEST = re.compile(r"^\s*Tests\s+(.*?)\((\d+)\)", re.MULTILINE)
_JEST_FAIL = re.compile(r"^\s*● (.+?)\s*$", re.MULTILINE)
_RULE = re.compile(r"^[_=]{3,} (.+?) [_=]{3,}$")
_PIPELINE = re.compile(r"^(?:all tests pass|(\d+) project test\(s\) failing)", re.MULTILINE)


def parse(command: str, result: str, *, turn: int | None = None,
          wall: float | None = None) -> TestRun:
    """A TestRun from a run_check / run_command result (`exit code: N` then the output)."""
    run = TestRun(command=command, output=result, turn=turn, wall=wall, duration=wall)
    match = _EXIT.search(result)
    if match:
        run.exit_code = int(match.group(1))
    if result.startswith("Stopped by you"):
        run.stopped = True
    kind = runner(command) or ""
    summary = None
    for summary in _PYTEST_SUMMARY.finditer(result):
        pass
    if summary is not None:
        for number, word in re.findall(r"(\d+) (\w+)", summary.group(1)):
            key = "errors" if word.startswith("error") else word
            if key in COUNT_WORDS:
                run.counts[key] = run.counts.get(key, 0) + int(number)
        run.duration = float(summary.group(2))
        for verdict, test, message in _PYTEST_FAILED.findall(result):
            run.failures.append(Failure(test, (message or verdict.lower()).strip()))
        return run
    nothing = _PYTEST_NONE.search(result)
    if nothing:
        run.duration = float(nothing.group(1))
        return run
    cargo = _CARGO.findall(result)
    if cargo:
        for passed, failed, ignored, seconds in cargo:
            run.counts["passed"] = run.counts.get("passed", 0) + int(passed)
            run.counts["failed"] = run.counts.get("failed", 0) + int(failed)
            run.counts["skipped"] = run.counts.get("skipped", 0) + int(ignored)
        run.duration = sum(float(seconds) for *_, seconds in cargo)
        run.failures = [Failure(name) for name in _CARGO_FAIL.findall(result)]
        return run
    if kind == "go" or _GO_FAIL.search(result):
        fails = _GO_FAIL.findall(result)
        passes, skips = len(_GO_PASS.findall(result)), len(_GO_SKIP.findall(result))
        if fails or passes or skips:
            run.counts = {"failed": len(fails), "passed": passes, "skipped": skips}
        run.failures = [Failure(name) for name in fails]
        durations = [float(s) for s in _GO_DURATION.findall(result)]
        if durations:
            run.duration = sum(durations)
        return run
    jest = _JEST.search(result) or _VITEST.search(result)
    if jest:
        for number, word in re.findall(r"(\d+) (failed|passed|skipped|todo)", jest.group(1)):
            key = "skipped" if word == "todo" else word
            run.counts[key] = run.counts.get(key, 0) + int(number)
        run.failures = [Failure(name) for name in _JEST_FAIL.findall(result)][:50]
        return run
    pipeline = _PIPELINE.search(result)
    if pipeline:
        failing = int(pipeline.group(1) or 0)
        if failing:
            run.counts["failed"] = failing
        return run
    return run


def section(run: TestRun, test: str) -> str:
    """The output that belongs to one failing test: pytest's `___ name ___` block, go's
    `--- FAIL: name` block, cargo's `---- name stdout ----` block; else the whole output."""
    lines = run.output.splitlines()
    name = test.split("::", 1)[1].replace("::", ".") if "::" in test else test
    header = re.compile(r"^_{3,} (.+?) _{3,}$")
    for start, line in enumerate(lines):
        found = header.match(line)
        if found and found.group(1) == name:
            end = start + 1
            while end < len(lines) and not header.match(lines[end]) and \
                    not re.match(r"^={3,} ", lines[end]):
                end += 1
            return "\n".join(lines[start:end])
        if line.strip().startswith(f"--- FAIL: {test}") or line.strip() == f"---- {test} stdout ----":
            end = start + 1
            while end < len(lines) and not lines[end].strip().startswith(("--- ", "---- ")):
                end += 1
            return "\n".join(lines[start:end])
    return run.output


def _clip(text: str, width: int, ellipsis: str, *, keep_end: bool = False) -> str:
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    if keep_end:
        while text and cell_len(text) > width - 1:
            text = text[1:]
        return ellipsis + text
    while text and cell_len(text) > width - 1:
        text = text[:-1]
    return text + ellipsis


def short_test(test: str) -> str:
    """`tests/unit/test_calc.py::test_add` -> `test_calc.py::test_add` (the folder is in the
    preview; the pane keeps the file and the name)."""
    path, sep, rest = test.partition("::")
    return (path.rsplit("/", 1)[-1] + sep + rest) if sep else test


SHORT = {"failed": "fail", "passed": "pass", "skipped": "skip", "errors": "err",
         "xfailed": "xfail", "xpassed": "xpass"}


def verdict_line(run: TestRun, colours: dict[str, str], g: SimpleNamespace,
                 width: int = 80) -> Text:
    """`✗ 2 failed  12 passed  1 skipped`, or `✓ 14 passed`, or the exit line. Narrower than
    that, the gaps close and then the words shorten (`✗ 2 fail 12 pass 1 skip`)."""
    text = Text()
    if run.stopped:
        text.append(_clip(f"{g.warn} stopped before it finished", width, g.ellipsis),
                    style=colours["warning"])
        return text
    mark, mark_style = (g.done, colours["success"]) if run.ok else ("✗" if g.ellipsis == "…"
                                                                   else "x", colours["error"])
    text.append(mark + " ", style=f"bold {mark_style}")
    styles = {"failed": f"bold {colours['error']}", "errors": f"bold {colours['error']}",
              "passed": colours["success"]}
    parts = [(key, run.counts[key]) for key in COUNT_WORDS if run.counts.get(key)]
    if not parts:
        line = f"exit code {run.exit_code}" if run.exit_code is not None else run.exit_line()
        text.append(_clip(line, max(4, width - 2), g.ellipsis), style=colours["fg"])
        return text
    for gap, words in (("  ", None), (" ", None), (" ", SHORT)):
        size = 2 + sum(len(f"{n} {words[k] if words else k}") for k, n in parts) + \
            len(gap) * (len(parts) - 1)
        if size <= width:
            break
    for index, (key, number) in enumerate(parts):
        if index:
            text.append(gap)
        text.append(str(number), style=styles.get(key, colours["fg"]))
        text.append(" " + (words[key] if words else key), style=colours["muted"])
    return text


def meta_line(run: TestRun, colours: dict[str, str]) -> Text:
    stamp = time.strftime("%H:%M", time.localtime(run.at))
    parts = []
    if run.duration is not None:
        parts.append(f"{run.duration:.1f} s" if run.duration < 100 else f"{run.duration:.0f} s")
    parts.append(f"turn {run.turn}" if run.turn is not None else "run from this pane")
    parts.append(stamp)
    return Text("  ".join(parts), style=colours["muted"])


def failure_option(failure: Failure, width: int, colours: dict[str, str],
                   g: SimpleNamespace) -> Text:
    mark = "✗" if g.ellipsis == "…" else "x"
    text = Text(mark + " ", style=colours["error"])
    name = short_test(failure.test)
    if cell_len(name) > width - 2 and "::" in name:  # too narrow for the file: the test alone
        name = name.split("::", 1)[1]
    text.append(_clip(name, max(4, width - 2), g.ellipsis, keep_end=True), style=colours["fg"])
    if failure.message:
        text.append("\n" + INDENT)
        text.append(_clip(failure.message, max(4, width - 2), g.ellipsis), style=colours["muted"])
    return text


def fill(options: OptionList, run: TestRun | None, running: str | None, width: int,
         colours: dict[str, str], g: SimpleNamespace, frame: int) -> None:
    """Rebuild the pane. `running` is the command being re-run from the pane, if any."""
    keep = options.highlighted_option.id if options.highlighted_option else None
    options.clear_options()
    if running is not None:
        spin = g.spinner[frame % len(g.spinner)]
        line = Text(spin + " ", style=colours["accent"])
        line.append(_clip(running, max(4, width - 14), g.ellipsis), style=colours["fg"])
        line.append("  Esc stops", style=colours["muted"])
        options.add_option(Option(line, disabled=True))
    if run is None:
        if running is None:
            options.add_option(Option(Text(EMPTY, style=colours["muted"]), disabled=True))
        return
    command = Text(_clip(run.command, width, g.ellipsis), style=colours["muted"])
    options.add_option(Option(command, disabled=True))
    summary = verdict_line(run, colours, g, width)
    summary.append("\n")
    summary.append_text(meta_line(run, colours))
    options.add_option(Option(summary, id="Tsummary"))
    for number, failure in enumerate(run.failures):
        options.add_option(Option(failure_option(failure, width, colours, g), id=f"T{number}"))
    if keep:
        with contextlib.suppress(Exception):  # the row is gone
            options.highlighted = options.get_option_index(keep)


def preview(run: TestRun, key: str, width: int, colours: dict[str, str],
            g: SimpleNamespace, limit: int) -> tuple[str, Text]:
    """(title, body) for the summary row (the whole output) or a failing test (its section)."""
    text = Text()
    if key == "Tsummary" or not key[1:].isdigit() or int(key[1:]) >= len(run.failures):
        title = _clip(run.command, max(10, width - 4), g.ellipsis)
        text.append_text(verdict_line(run, colours, g))
        text.append("\n")
        text.append_text(meta_line(run, colours))
        text.append("\nr runs it again (sandboxed; Esc stops)\n\n", style=colours["muted"])
        body = run.output
    else:
        failure = run.failures[int(key[1:])]
        title = _clip(failure.test, max(10, width - 4), g.ellipsis, keep_end=True)
        if failure.message:
            text.append(failure.message + "\n\n", style=colours["error"])
        body = section(run, failure.test)
    for number, line in enumerate(body.splitlines()[:limit]):
        if number:
            text.append("\n")
        rule = _RULE.match(line)  # pytest's `____ name ____` / `==== FAILURES ====` rules
        if rule:
            heading = rule.group(1).strip()
            text.append(heading, style=f"bold {colours['fg']}" if line.startswith("_")
                        else colours["muted"])
            continue
        error = line.startswith(("E ", "E\t", "FAILED", "ERROR", "--- FAIL")) or \
            "Error" in line or "panicked" in line
        text.append(line.expandtabs(4), style=colours["error"] if error else colours["fg"])
    return title, text


__all__ = ["EMPTY", "Failure", "TestRun", "fill", "parse", "preview", "runner", "section"]
