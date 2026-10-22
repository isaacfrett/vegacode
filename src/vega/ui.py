"""Terminal rendering for the interactive chat: progress lines, streamed text and edit review."""

from __future__ import annotations

import time

import difflib
import re
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

from .tools import WRITE_TOOLS, WritePlan
from .web import FAILURE_PREFIXES as _WEB_FAILURES

console = Console()
SUMMARY_WIDTH = 72
OUTCOME_WIDTH = 80
_FAILURE_PREFIXES = (
    "Edit not applied", "Write not applied", "File not found", "Tool error", "Invalid tool arguments",
    "Invalid line range", "Unknown tool", "The old_text block", "This delegated session",
    "User declined", "Commands are disabled", "Invalid plan", "Note not saved", *_WEB_FAILURES,
)
_PLAN_STYLES = {"done": ("✓", "dim green"), "in_progress": ("▸", "bold"), "pending": ("·", "dim")}


def plan_lines(items: list[dict[str, str]]) -> Text:
    """The plan as a compact checklist: ✓ done, ▸ in progress, · pending."""
    if not items:
        return Text("  plan cleared", style="dim")
    lines = Text()
    for index, item in enumerate(items):
        mark, style = _PLAN_STYLES.get(item.get("status", ""), ("·", "dim"))
        if index:
            lines.append("\n")
        lines.append(f"  {mark} {_shorten(item.get('text', ''), SUMMARY_WIDTH)}", style=style)
    return lines


def _shorten(text: str, width: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def _summary(name: str, args: dict[str, Any]) -> str:
    """The one argument that identifies a call: its path, query, command or task."""
    if "_error" in args:
        return "(unparseable arguments)"
    if name == "replace_lines":
        return f"{args.get('path', '')}:{args.get('start_line', '?')}-{args.get('end_line', '?')}"
    for key in ("path", "command", "query", "url", "task", "note"):
        if args.get(key):
            return str(args[key])
    return ""


def _outcome(name: str, result: str) -> tuple[str, str]:
    """(text, style) to append to a tool line; empty text when the result is not worth a mention."""
    first = result.strip().split("\n", 1)[0]
    if result.startswith(_FAILURE_PREFIXES):
        return "→ " + _shorten(first, OUTCOME_WIDTH), "yellow"
    if name in WRITE_TOOLS:
        note = ""
        match = re.search(r"\(([^()]*)\)\.?$", first)
        if match:
            note = f"({match.group(1)})"
        warning = re.search(r"WARNING: (.*)", result)
        if warning:
            return _shorten(f"{note} ⚠ {warning.group(1)}", OUTCOME_WIDTH), "yellow"
        return note, ""
    if name in {"run_check", "run_command"}:
        code = re.match(r"exit code: (-?\d+)", first)
        if code:
            return f"→ exit {code.group(1)}", "green" if code.group(1) == "0" else "red"
        return "→ " + _shorten(first, OUTCOME_WIDTH), "yellow"
    return "", ""


_TEST_SUMMARY = re.compile(
    r"\b(\d+ (?:passed|failed|errors?)(?:, \d+ (?:passed|failed|skipped|errors?|warnings?))*)"
)


def step_summary(name: str, args: dict[str, Any]) -> str:
    """The target column of a tool step: the path, query, command or task it acted on."""
    return re.sub(r"\s+", " ", _summary(name, args)).strip()


def step_outcome(name: str, result: str) -> tuple[str, str]:
    """(outcome text, level) for a finished tool step, without arrows or styling. level is "" for
    plain, "ok", "warning" (a problem the agent can work around) or "error" (a failed check)."""
    first = result.strip().split("\n", 1)[0]
    if result.startswith(_FAILURE_PREFIXES):
        return _shorten(first, OUTCOME_WIDTH), "warning"
    if name in WRITE_TOOLS or name == "remember":
        warning = re.search(r"WARNING: (.*)", result)
        if warning:
            return _shorten(warning.group(1), OUTCOME_WIDTH), "warning"
        match = re.search(r"\(([^()]*)\)\.?$", first)
        if name == "remember":
            return "saved", ""
        return (match.group(1) if match else ""), ""
    if name in {"run_check", "run_command"}:
        code = re.match(r"exit code: (-?\d+)", first)
        if not code:
            return _shorten(first, OUTCOME_WIDTH), "warning"
        tests = _TEST_SUMMARY.findall(result)
        detail = f", {tests[-1]}" if tests else ""
        return f"exit {code.group(1)}{detail}", "ok" if code.group(1) == "0" else "error"
    return _web_outcome(name, result), ""


_RESULT_COUNT = re.compile(r'^Search results for ".*" \((\d+)\):')


def _web_outcome(name: str, result: str) -> str:
    """web_search: how many results; fetch_page: the page's title and date (Steps)."""
    if name == "web_search":
        if result.startswith("No results"):
            return "no results"
        count = _RESULT_COUNT.match(result)
        return f"{count.group(1)} results" if count else ""
    if name == "fetch_page":
        title = re.search(r"^Title: (.*)$", result, re.MULTILINE)
        date = re.search(r"^Published: (.*)$", result, re.MULTILINE)
        if title:
            return _shorten(f"{title.group(1)} ({date.group(1) if date else 'unknown'})",
                            OUTCOME_WIDTH)
    return ""


def undo_note(changed: list[str]) -> dict[str, str]:
    """The message that tells the model its last turn's file changes were undone."""
    return {
        "role": "user",
        "content": "(Vega harness) The user undid your file changes from the previous turn "
        f"({', '.join(changed)}). Those files are back to their earlier content; re-read them "
        "before editing.",
    }


def tool_line(name: str, args: dict[str, Any], result: str) -> Text:
    """One dim line describing a finished tool call, e.g. `· run_check pytest -q → exit 0`."""
    line = Text("· ", style="dim")
    line.append(name, style="dim bold")
    summary = _shorten(_summary(name, args), SUMMARY_WIDTH)
    if summary:
        line.append(" " + summary, style="dim")
    outcome, style = _outcome(name, result)
    if outcome:
        line.append(" " + outcome, style=style or "dim")
    return line


def write_diff(plan: WritePlan) -> str:
    """Unified diff of a planned write (empty string when it changes nothing)."""
    before = (plan.before or "").splitlines(keepends=True)
    after = plan.after.splitlines(keepends=True)
    source = "/dev/null" if plan.before is None else f"a/{plan.relative}"
    lines = difflib.unified_diff(before, after, fromfile=source, tofile=f"b/{plan.relative}")
    no_newline = "\n\\ No newline at end of file\n"
    return "".join(line if line.endswith("\n") else line + no_newline for line in lines)


def ask_approval(plan: WritePlan, out: Console | None = None) -> bool:
    """Show the planned change as a coloured diff and ask whether to apply it."""
    out = out or console
    diff = write_diff(plan)
    verb = "create" if plan.before is None else plan.tool.replace("_", " ")
    out.print(f"\n[bold yellow]Vega wants to {verb}[/bold yellow] {plan.relative}")
    if diff:
        out.print(Syntax(diff.rstrip("\n"), "diff", theme="ansi_dark", background_color="default"))
    else:
        out.print("[dim](no textual change)[/dim]")
    try:
        answer = out.input("[yellow]Apply this change? \\[y/N] [/yellow]")
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in {"y", "yes"}


class TerminalRenderer:
    """on_event handler for CodingAgent that draws progress in the terminal as it happens."""

    def __init__(self, out: Console | None = None, *, stream_text: bool = False):
        # By default the model's raw text is held back until it is known to be the final answer:
        # a 7B often "thinks out loud", writes tool calls as JSON text or asks questions that the
        # harness sends back, and printing that live buried the progress lines (2026-10-13).
        self.console = out or console
        self.stream_text = stream_text
        self._streaming = False
        self._status = None
        self._working = None  # spinner shown while the model reads/writes
        self._turn_started = 0.0
        self._last_label = 0.0

    def __call__(self, kind: str, data: dict[str, Any]) -> None:
        handler = getattr(self, f"_on_{kind}", None)
        if handler is not None:
            handler(**data)

    def _end_stream(self) -> None:
        if self._streaming:
            self.console.print()
            self._streaming = False

    def _start_working(self) -> None:
        if self._working is None and self.console.is_terminal:
            self._working = self.console.status(Text(self._working_label(), style="dim"))
            self._working.start()

    def _stop_working(self) -> None:
        if self._working is not None:
            self._working.stop()
            self._working = None

    def _working_label(self) -> str:
        elapsed = time.monotonic() - self._turn_started if self._turn_started else 0.0
        return f"Vega is working… {elapsed:.0f}s"

    def _on_request(self, step: int = 0) -> None:
        if step == 0 or not self._turn_started:
            self._turn_started = time.monotonic()
        self._start_working()

    def _on_text(self, delta: str) -> None:
        if not self.stream_text:
            now = time.monotonic()
            if self._working is not None and now - self._last_label > 0.5:
                self._working.update(Text(self._working_label(), style="dim"))
                self._last_label = now
            return
        self._stop_working()
        self._streaming = True
        self.console.print(delta, end="", markup=False, highlight=False, soft_wrap=True)

    def _on_text_end(self) -> None:
        self._end_stream()

    def _on_discarded(self, reason: str, retry: bool) -> None:
        self._stop_working()
        self._end_stream()
        label = f"(retrying: {reason})" if retry else f"({reason}; not the answer)"
        self.console.print(Text(label, style="dim italic"))

    def _on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        self._stop_working()
        self._end_stream()
        if name == "run_check" and self.console.is_terminal:
            # Slow and never prompts, so it gets a transient spinner; the summary line replaces it.
            label = Text(f"{name} {_shorten(_summary(name, args), 60)}", style="dim")
            self._status = self.console.status(label)
            self._status.start()

    def _on_tool_result(self, name: str, args: dict[str, Any], result: str) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None
        if name == "update_plan" and not result.startswith(_FAILURE_PREFIXES):
            return  # the plan event that preceded this result already drew the checklist
        self.console.print(tool_line(name, args, result))

    def _on_plan(self, items: list[dict[str, str]]) -> None:
        self._stop_working()
        self._end_stream()
        self.console.print(Text("· plan", style="dim bold"))
        self.console.print(plan_lines(items))

    def _on_teach(self, role: str, text: str) -> None:
        """Teach mode: the model's plan, why and fix lines, in a quiet italic under the steps."""
        self._stop_working()
        self._end_stream()
        label = {"plan": "plan", "why": "why", "failure": "fix"}.get(role, role)
        line = Text(f"  {label:<4} ", style="bold magenta")
        line.append(text, style="italic" if role == "plan" else "dim italic")
        self.console.print(line)

    def _on_notice(self, message: str) -> None:
        self._stop_working()
        self._end_stream()
        self.console.print(Text(message, style="yellow"))

    def _on_autopilot(self, event: str, message: str, **_: Any) -> None:
        """Autopilot progress (round start/end, check results, commits) as one dim line each."""
        self._end_stream()
        self.console.print(Text(message, style="dim"))

    def _on_final(self, text: str, streamed: bool) -> None:
        self._stop_working()
        self._end_stream()
        self._turn_started = 0.0
        if not (streamed and self.stream_text):  # held-back text is shown here, formatted
            self.console.print(Markdown(text))
