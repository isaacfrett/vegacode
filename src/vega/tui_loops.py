"""The Loops pane: one two-line row per loop, its open findings on Enter, evidence in the preview.

Design (2026-10-14): the needs-you count in parentheses is the one emphatic element (accent,
bold); informational statuses are dim; a failing loop shows `!` and its error in the error colour;
a running loop's symbol is the spinner, the pane's only motion. The name truncates, the status
never does. Option ids: `L<loop name>` for a loop, `F<finding id>` for a finding.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

from rich.text import Text
from textual.binding import Binding, BindingType
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

ASCII_SYMBOLS = {"active": "*", "idle": "o", "paused": "=", "failing": "!"}
EMPTY = "No loops yet. `vega loops start` sets up Bug Watcher for this repo."
DEBUG_LINES = 200
# the second line's indent: non-breaking, because leading spaces after a newline are dropped when
# an option prompt is drawn
INDENT = "\u00a0\u00a0"


def _clip(text: str, width: int, ellipsis: str) -> str:
    return text if len(text) <= width else text[:max(0, width - 1)] + ellipsis


def header_text(header: Any, colours: dict[str, str], g: SimpleNamespace) -> Text:
    """'screen locked ✓ · model allowed' / 'model held: <reason>' / 'loops not running'."""
    if header is None:
        return Text("checking…", style=colours["muted"])
    text = Text()
    if not header.daemon_running:
        text.append("stopped: vega loops start", style=colours["muted"])
        return text
    text.append("screen locked " + g.done if header.screen_locked else "screen unlocked",
                style=colours["muted"])
    if header.model_allowed:
        text.append(" · model allowed", style=colours["muted"])
    else:
        reason = header.reasons[0] if header.reasons else "not idle"
        text.append(f" · model held: {reason}", style=colours["warning"])
    return text


def loop_option(row: Any, width: int, colours: dict[str, str], g: SimpleNamespace, frame: int,
                ascii_only: bool) -> Text:
    """Two lines: `● Name (status)` and the dim trigger/time (or the error when failing)."""
    if row.running_for is not None and row.state != "paused":
        symbol, symbol_style = g.spinner[frame % len(g.spinner)], colours["accent"]
    else:
        symbol = (ASCII_SYMBOLS if ascii_only else {"active": "●", "idle": "○", "paused": "‖",
                                                    "failing": "!"})[row.state]
        symbol_style = (colours["error"] if row.state == "failing" else
                        colours["muted"] if row.state in {"paused", "idle"} else colours["fg"])
    status = f"({row.status})"
    if row.state == "failing":
        status_style = colours["error"]
    elif row.needs_you:
        status_style = f"bold {colours['accent']}"
    else:
        status_style = colours["muted"]
    name_style = colours["muted"] if row.state == "paused" else colours["fg"]
    second = row.error if row.state == "failing" and row.error else row.when()
    second_style = colours["error"] if row.state == "failing" else colours["muted"]
    text = Text()
    text.append(symbol + " ", style=symbol_style)
    if 2 + len(row.name) + 1 + len(status) <= width:  # name and status on one line
        text.append(row.name + " ", style=name_style)
        text.append(status, style=status_style)
        text.append("\n" + INDENT)
        text.append(_clip(second, max(4, width - 2), g.ellipsis), style=second_style)
    else:  # narrow: the whole name, then the status leads the second line (never truncated)
        text.append(_clip(row.name, max(4, width - 2), g.ellipsis), style=name_style)
        text.append("\n" + INDENT)
        text.append(status, style=status_style)
        rest = max(0, width - 2 - len(status) - 3)
        if rest >= 4:
            text.append(" · " + _clip(second, rest, g.ellipsis), style=second_style)
    return text


def finding_option(finding: dict[str, Any], state: str, width: int, colours: dict[str, str],
                   g: SimpleNamespace) -> Text:
    text = Text(INDENT * 2)
    text.append(_clip(finding.get("title") or finding.get("id", ""),
                      max(4, width - 4 - len(state) - 1), g.ellipsis) + " ", style=colours["fg"])
    text.append(state, style=f"bold {colours['accent']}" if state in {"new", "proposed"}
                else colours["muted"])
    return text


def fill(options: OptionList, header: Any, rows: list[Any], errors: list[str], expanded: set[str],
         findings: dict[str, list[tuple[dict[str, Any], str]]], width: int,
         colours: dict[str, str], g: SimpleNamespace, frame: int, ascii_only: bool) -> None:
    """Rebuild the pane (header line, then the loops), keeping the highlighted row."""
    keep = options.highlighted_option.id if options.highlighted_option else None
    options.clear_options()
    line = header_text(header, colours, g)
    line.truncate(width, overflow="ellipsis")
    options.add_option(Option(line, disabled=True))
    if not rows and not errors:
        options.add_option(Option(Text(EMPTY, style=colours["muted"]), disabled=True))
    for row in rows:
        options.add_option(Option(loop_option(row, width, colours, g, frame, ascii_only),
                                  id=f"L{row.name}"))
        if row.name in expanded:
            items = findings.get(row.name, [])
            if not items:
                options.add_option(Option(Text(INDENT * 2 + "nothing open", style=colours["muted"]),
                                          disabled=True))
            for finding, state in items:
                options.add_option(Option(finding_option(finding, state, width, colours, g),
                                          id=f"F{finding['id']}"))
    for error in errors:
        name, _, detail = error.partition(":")
        text = Text("! ", style=colours["error"])
        text.append(_clip(Path(name).name, max(4, width - 13), g.ellipsis) + " ",
                    style=colours["fg"])
        text.append("(invalid)", style=colours["error"])
        text.append("\n" + INDENT + _clip(detail.strip(), max(4, width - 2), g.ellipsis),
                    style=colours["error"])
        options.add_option(Option(text, disabled=True))
    if keep:
        with contextlib.suppress(Exception):  # the row is gone
            options.highlighted = options.get_option_index(keep)


def finding_preview(finding: dict[str, Any], state: str, width: int, colours: dict[str, str],
                    g: SimpleNamespace) -> tuple[str, Text]:
    """(preview title, body): what was found, where, and the evidence that reproduces it."""
    evidence = finding.get("evidence") or {}
    title = _clip(finding.get("title") or finding.get("id", "finding"), max(10, width - 12),
                  g.ellipsis) + f"  {state}"
    text = Text()
    where = ", ".join(f"{w.get('path')}:{w.get('line')}" if w.get("line") else str(w.get("path"))
                      for w in finding.get("where") or [])
    meta = [part for part in (finding.get("loop"), where,
                              f"commit {finding['commit'][:8]}" if finding.get("commit") else "")
            if part]
    text.append(" · ".join(meta) + "\n", style=colours["muted"])
    kind = evidence.get("type", "evidence")
    proof = {"failing_test": "a test that fails", "lint": "lint findings"}.get(kind, kind)
    if evidence.get("fails_at"):
        proof += f" at {str(evidence['fails_at'])[:8]}"
    if evidence.get("passes_at"):
        proof += f", passes at {str(evidence['passes_at'])[:8]}"
    text.append(f"Evidence: {proof}\n", style=colours["fg"])
    if evidence.get("path"):
        text.append(f"Test: {evidence['path']}\n", style=colours["muted"])
    test_file = Path(str(evidence.get("path", "")))
    if test_file.is_file():
        text.append("\n")
        for line in test_file.read_text(encoding="utf-8", errors="replace").splitlines()[:80]:
            text.append(line.expandtabs(4) + "\n", style=colours["fg"])
    if evidence.get("output"):
        text.append("\n")
        for line in str(evidence["output"]).splitlines()[-60:]:
            error = line.startswith(("E ", "E\t")) or "Error" in line
            text.append(line.expandtabs(4) + "\n", style=colours["error"] if error
                        else colours["muted"])
    text.append("\na accept · x reject · l debug log", style=colours["muted"])
    return title, text


def loop_preview(row: Any, items: list[tuple[dict[str, Any], str]], width: int,
                 colours: dict[str, str], g: SimpleNamespace) -> tuple[str, Text]:
    title = f"{row.name}  ({row.status})"
    text = Text(row.when() + "\n", style=colours["muted"])
    if row.error:
        text.append(f"Last job failed: {row.error}\n", style=colours["error"])
    text.append("\n")
    if not items:
        text.append("Nothing open.\n", style=colours["muted"])
    for finding, state in items:
        text.append(f"{state:9}", style=f"bold {colours['accent']}" if state in {"new", "proposed"}
                    else colours["muted"])
        text.append(_clip(finding.get("title") or finding["id"], max(10, width - 10), g.ellipsis)
                    + "\n", style=colours["fg"])
    text.append("\nEnter shows findings · p pause/resume · r run now · l debug log",
                style=colours["muted"])
    return title, text


def debug_preview(loop_name: str, colours: dict[str, str]) -> tuple[str, Text]:
    """The newest debug log of this loop's jobs, formatted like `vega debug watch`."""
    from . import debuglog

    slug = "".join(ch if ch.isalnum() else "-" for ch in loop_name.lower())
    logs = sorted(debuglog.folder().glob(f"*loop-{slug}*.jsonl")) if debuglog.folder().is_dir() else []
    if not logs:
        return f"{loop_name}  debug log", Text("No job log yet: it appears after the loop's "
                                               "first run.", style=colours["muted"])
    text = Text()
    lines = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()[-DEBUG_LINES:]
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        for style, body in debuglog.format_entry(entry):
            text.append(body + "\n", style=colours["error"] if "red" in style else
                        colours["muted"] if style == "dim" else colours["fg"])
    return f"{loop_name}  {logs[-1].name}", text


def run_in_background(loop_name: str) -> None:
    """`r`: run the loop now in its own process (the chat stays responsive; the idle rule still
    decides whether the model pass runs)."""
    subprocess.Popen([sys.executable, "-c", "from vega.cli import app; app()", "loops", "run",
                      loop_name],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


class LoopsList(OptionList):
    """The Loops pane. Enter expands a loop's open findings; the keys act on the highlighted row."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("p", "loop('pause')", "Pause/resume", show=False),
        Binding("r", "loop('run')", "Run now", show=False),
        Binding("l", "loop('log')", "Debug log", show=False),
        Binding("a", "finding('accepted')", "Accept", show=False),
        Binding("x", "finding('rejected')", "Reject", show=False),
    ]

    class Action(Message):
        def __init__(self, action: str, option_id: str) -> None:
            super().__init__()
            self.action = action
            self.option_id = option_id

    def _current(self) -> str | None:
        option = self.highlighted_option
        return option.id if option is not None else None

    def _on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Enter (or a click) on a loop row expands or collapses its findings."""
        if event.option.id:
            self.post_message(self.Action("toggle", event.option.id))

    def action_loop(self, action: str) -> None:
        current = self._current()
        if current:
            self.post_message(self.Action(action, current))

    def action_finding(self, state: str) -> None:
        current = self._current()
        if current and current.startswith("F"):
            self.post_message(self.Action(state, current))
