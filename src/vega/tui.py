"""Full-screen terminal app for `vega` / `vega chat` (Textual).

The agent is synchronous, so each turn runs in a Textual worker thread. Agent events are forwarded
to the UI thread with call_from_thread; approvals (edit review, run_command) block the worker on a
Future until the modal on the UI thread answers. Ctrl+C asks the agent to stop after its current
step (CodingAgent.request_stop). `vega --plain` keeps the line-mode chat in cli.py.

Four layouts share one widget tree and one behaviour; CSS (tui*.tcss, scoped by an App class
`layout-<name>`) decides what is shown and where. VEGA_LAYOUT or /layout picks one (workbench
is the default; what its columns hold is the user's choice, see view.py, /view and F2):
classic (conversation and side panel), quiet (one centred column), observatory (bordered panes and
an instrument stack), workbench (changes, conversation and a diff preview in three columns).
"""

from __future__ import annotations

import contextlib
import difflib
import json
import os
import re
import sys
import time
import webbrowser
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widget import Widget
from textual.widgets import Button, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from . import (
    clipboard,
    context,
    features,
    models,
    product,
    resources,
    server,
    teaching,
    tui_context,
    tui_git,
    tui_loops,
    tui_models,
    tui_problems,
    tui_product,
    tui_tests,
    usage,
    view,
    web,
)
from .agent import _WRITE_OK, STOPPED, CodingAgent, FileSnapshot
from .tools import WRITE_TOOLS, WritePlan, execute, is_project, tool_specs
from .transcripts import SessionLog
from .ui import step_outcome, step_summary, undo_note, write_diff

LAYOUTS = ("classic", "quiet", "observatory", "workbench")
WIDE_AT = {"classic": 100, "quiet": 100, "observatory": 100, "workbench": 110}
SIDE_WIDTH = 34  # the classic side panel, and the overlay panel in every layout
GAUGE_CELLS = 20
PLACEHOLDER = "Ask Vega to change, explain or check something"
NOT_A_PROJECT = (
    "This folder doesn't look like a project (no Git repository or project file), so Vega only "
    "sees a partial view. Start Vega inside a project, or use vega -p <path>."
)
MEMORY_EVERY = 5.0  # seconds between model-memory readings
PREVIEW_LINES = 400  # most lines of a step's output shown in the workbench preview


def layout_from_env() -> str:
    wanted = os.environ.get("VEGA_LAYOUT", "").strip().lower()
    return wanted if wanted in LAYOUTS else "workbench"


# -- glyphs and palette ----------------------------------------------------------------------------

def glyphs(ascii_only: bool | None = None) -> SimpleNamespace:
    """Symbols used across the app; VEGA_ASCII=1 swaps them for plain ASCII."""
    if ascii_only is None:
        ascii_only = os.environ.get("VEGA_ASCII", "").strip() not in {"", "0"}
    if ascii_only:
        return SimpleNamespace(
            done="+", doing=">", pending="-", closed=">", opened="v", fixed="=", used="#",
            empty=".", tick="|", warn="!", prompt=">", minus="-", spinner="|/-\\", star="*",
            ellipsis="~", spark=" .:-=+*#", bullet="-", fail="x", open="o",
        )
    return SimpleNamespace(
        done="✓", doing="▸", pending="·", closed="▸", opened="▾", fixed="█", used="█",
        empty="░", tick="│", warn="⚠", prompt="›", minus="−", spinner="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏", star="✦",
        ellipsis="…", spark="▁▂▃▄▅▆▇█", bullet="·", fail="✗", open="○",
    )


def _theme(name: str, dark: bool, *, fg: str, bg: str, panel: str, accent: str, muted: str,
           fixed: str, rule: str, track: str, success: str, warning: str, warning_text: str,
           error: str, overlay: str, readout: str, add_bg: str, del_bg: str,
           teach: str = "") -> Theme:
    return Theme(
        name=name, dark=dark, primary=accent, accent=accent, secondary=muted,
        foreground=fg, background=bg, surface=bg, panel=panel,
        success=success, warning=warning, error=error,
        variables={
            "vega-muted": muted, "vega-fixed": fixed, "vega-rule": rule, "vega-track": track,
            "vega-warning-text": warning_text, "vega-overlay": overlay, "vega-readout": readout,
            "vega-add-bg": add_bg, "vega-del-bg": del_bg,
            # teach mode's notes: one tint in every layout, apart from the accent and warnings
            "vega-teach": teach or ("#B9A6EC" if dark else "#6A44C2"),
            "text-muted": muted, "button-focus-text-style": "bold underline",
        },
    )


# Each layout has its own palette in dark and light. Text colours meet 4.5:1 on background, panel
# and overlay; the gauge's fixed segment is a graphic (3:1) and differs from the accent in hue.
LAYOUT_THEMES: dict[str, dict[str, Theme]] = {
    # classic: deep blue night, pale blue accent (the first version, kept as the default)
    "classic": {
        "dark": _theme(
            "vega-dark", True, fg="#E3E9F3", bg="#0F1626", panel="#16203A", accent="#9CC9FF",
            muted="#7D8BA3", fixed="#5C6F8E", rule="#2A3650", track="#3A4866",
            success="#7FD1B9", warning="#E8C170", warning_text="#E8C170", error="#F28B82",
            overlay="#16203A", readout="#E3E9F3", add_bg="#173A33", del_bg="#3D2030"),
        "light": _theme(
            "vega-light", False, fg="#18202E", bg="#F3F6FA", panel="#EBF0F6", accent="#2A5DB0",
            muted="#5B6880", fixed="#4A5262", rule="#C9D2DF", track="#C3CCD9",
            success="#1F7A63", warning="#9A6A00", warning_text="#7F5700", error="#B3261E",
            overlay="#EBF0F6", readout="#18202E", add_bg="#DCEFE7", del_bg="#F6E0DE"),
    },
    # quiet: neutral ink and paper that sit close to a terminal's own colours; one accent, the
    # blue-white of the star
    "quiet": {
        "dark": _theme(
            "vega-quiet-dark", True, fg="#DADDE3", bg="#121417", panel="#1B1E23",
            accent="#AFCBFF", muted="#8A919D", fixed="#6E7684", rule="#262A31", track="#3A3F48",
            success="#8FD0B0", warning="#E0B860", warning_text="#E0B860", error="#F08A80",
            overlay="#1B1E23", readout="#DADDE3", add_bg="#18302A", del_bg="#3A2024"),
        "light": _theme(
            "vega-quiet-light", False, fg="#1C2027", bg="#FBFBFC", panel="#F0F1F4",
            accent="#2D5BC9", muted="#5E6673", fixed="#6B7280", rule="#E1E4E9", track="#CDD2DA",
            success="#1E7A5A", warning="#A66F00", warning_text="#875800", error="#B3261E",
            overlay="#F0F1F4", readout="#1C2027", add_bg="#E2F2EA", del_bg="#FBE5E3"),
    },
    # observatory: night sky and chart paper; the star's blue-white leads, amber reads out
    "observatory": {
        "dark": _theme(
            "vega-observatory-dark", True, fg="#D6E2F5", bg="#0A0F1C", panel="#111829",
            accent="#C3D8FF", muted="#8090AC", fixed="#8A7650", rule="#2A3550", track="#2C3854",
            success="#86D1B4", warning="#F0B35A", warning_text="#F0B35A", error="#F38B84",
            overlay="#111829", readout="#F0B35A", add_bg="#16332E", del_bg="#3A1E28"),
        "light": _theme(
            "vega-observatory-light", False, fg="#111A2B", bg="#F4F6FA", panel="#EAEEF5",
            accent="#2348A0", muted="#566178", fixed="#A07E3E", rule="#C3CCDA", track="#CBD3E0",
            success="#1D7356", warning="#8A5300", warning_text="#8A5300", error="#B0261E",
            overlay="#EAEEF5", readout="#8A5300", add_bg="#DDEFE6", del_bg="#F6E0DE"),
    },
    # workbench: ink on paper, and the same ink on a dark desk; diff tints carry the colour
    "workbench": {
        "dark": _theme(
            "vega-workbench-dark", True, fg="#E4E6EB", bg="#16181D", panel="#1F2228",
            accent="#82A8FF", muted="#8E95A1", fixed="#757C88", rule="#30343C", track="#3A3F48",
            success="#6CCB8A", warning="#E0B050", warning_text="#E0B050", error="#F2857A",
            overlay="#1F2228", readout="#E4E6EB", add_bg="#1A3324", del_bg="#3F2023"),
        "light": _theme(
            "vega-workbench-light", False, fg="#16181D", bg="#FFFFFF", panel="#F3F4F6",
            accent="#1F4FD1", muted="#5F6570", fixed="#7A808A", rule="#D3D6DC", track="#D9DCE1",
            success="#1A7F37", warning="#9A6200", warning_text="#7E5000", error="#C2261D",
            overlay="#F3F4F6", readout="#16181D", add_bg="#E3F4E8", del_bg="#FCE6E4"),
    },
}
THEMES = LAYOUT_THEMES["classic"]  # vega-dark / vega-light


def palette(app: App) -> dict[str, str]:
    theme = app.current_theme
    variables = theme.variables
    return {
        "fg": theme.foreground or "", "accent": theme.accent or "", "bg": theme.background or "",
        "success": theme.success or "", "error": theme.error or "", "tick": theme.warning or "",
        "warning": variables.get("vega-warning-text", theme.warning or ""),
        "muted": variables.get("vega-muted", ""), "fixed": variables.get("vega-fixed", ""),
        "rule": variables.get("vega-rule", ""), "track": variables.get("vega-track", ""),
        "readout": variables.get("vega-readout", theme.foreground or ""),
        "add_bg": variables.get("vega-add-bg", ""), "del_bg": variables.get("vega-del-bg", ""),
        "panel": theme.panel or "",
        "teach": variables.get("vega-teach", theme.accent or ""),
    }


def short_path(path: Path | str) -> str:
    """The path with the home folder shortened to ~ (any platform)."""
    text = str(path)
    home = str(Path.home())
    if text == home:
        return "~"
    if text.startswith(home + os.sep):
        return "~" + text[len(home):]
    return text


def _clip(text: str, width: int, *, keep_end: bool = False) -> str:
    """Fit text in `width` cells with an ellipsis; keep_end keeps the tail (file names)."""
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    mark = glyphs().ellipsis
    if keep_end:
        while text and cell_len(text) > width - 1:
            text = text[1:]
        return mark + text
    while text and cell_len(text) > width - 1:
        text = text[:-1]
    return text + mark


def clip_middle(path: str, width: int) -> str:
    """Shorten a path in the middle, keeping its first folder and the end of the file name:
    `tests/test_sub.py` in 14 cells is `tests/…_sub.py`."""
    if cell_len(path) <= width:
        return path
    if width < 4:
        return _clip(path, width, keep_end=True)
    mark = glyphs().ellipsis
    head = path.split("/", 1)[0] + "/" if "/" in path else ""
    if cell_len(head) > width // 2:
        head = ""
    tail_width = width - cell_len(head) - 1
    tail = path
    while tail and cell_len(tail) > tail_width:
        tail = tail[1:]
    return head + mark + tail


def window_label(window: int) -> str:
    return f"{window // 1024}K" if window % 1024 == 0 else f"{round(window / 1000)}K"


def tokens_label(tokens: float) -> str:
    return f"{tokens / 1000:.1f}K" if tokens >= 1000 else f"{tokens:.0f}"


def seconds_label(seconds: float) -> str:
    return f"{seconds:.1f} s"


def transcript_label(path: Path) -> str:
    """`13 Oct 21:29` (local time) from a transcript named `20261013T212916-<project>.json`."""
    match = re.match(r"(\d{8}T\d{6})", path.name)
    if match:
        stamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        return stamp.astimezone().strftime("%d %b %H:%M").lstrip("0")
    return _clip(path.name, 20)


def memory_label(used_mb: float, total_gb: float | None) -> str:
    used = f"{used_mb / 1024:.1f}"
    if total_gb:
        total = f"{total_gb:.0f}" if total_gb >= 10 else f"{total_gb:.1f}"
        return f"{used} of {total} GB"
    return f"{used} GB"


def sparkline(values: list[float], g: SimpleNamespace) -> str:
    if not values:
        return ""
    top = max(values) or 1.0
    marks = g.spark
    return "".join(marks[min(len(marks) - 1, round(value / top * (len(marks) - 1)))]
                   for value in values)


# -- context usage ---------------------------------------------------------------------------------

@dataclass
class Usage:
    fixed: int  # system message + tool definitions
    total: int  # what the next request would hold
    window: int
    measured: bool = False  # the server has reported a prompt size (estimates are scaled to it)
    parts: Any = field(default=None, compare=False)  # tui_context.Breakdown, for the Context pane


def context_usage(agent: CodingAgent, window: int | None = None) -> Usage:
    """Fixed prompt and total prompt size, from the agent's budget (server-measured when it has
    reported usage, else estimated). Never contacts the server: an unknown window is the default."""
    window = window or agent._num_ctx or context.DEFAULT_NUM_CTX
    budget = agent._budget
    scale = budget.scale if budget else 1.0
    if budget is not None:
        tools = budget.fixed_tokens
    else:
        tools = context.estimate_text_tokens(json.dumps(tool_specs(
            include_delegate=agent.can_delegate,
            include_remember=agent.can_write and features.on("remember"),
            include_plan=agent.planning == "on" and features.on("plan"),
            include_index=features.on("code_index"),
        )))
    system = context.estimate_text_tokens(str(agent.messages[0].get("content") or ""))
    fixed = int((system + tools) * scale)
    if budget is not None:
        total = budget.projected(agent.messages)
    else:
        total = int((context.estimate_tokens(agent.messages) + tools) * scale)
    measured = budget is not None and (budget.prompt_tokens is not None or budget.scale != 1.0)
    parts = tui_context.breakdown(agent.messages, tools, agent._plan_section())
    return Usage(fixed=fixed, total=max(total, fixed), window=window, measured=measured,
                 parts=parts)


def gauge_cells(fixed: float, used: float, window: int, cells: int) -> tuple[int, int]:
    """(fixed cells, conversation cells) for a gauge `cells` wide. Each segment gets at least one
    cell when it holds any tokens, so a short conversation is never invisible."""
    window = max(1, window)
    fixed_cells = min(cells, round(fixed / window * cells))
    if fixed > 0:
        fixed_cells = max(1, fixed_cells)
    used_cells = min(cells, round(used / window * cells))
    conversation = max(0, used_cells - fixed_cells)
    if used > fixed and conversation == 0 and fixed_cells < cells:
        conversation = 1
    return fixed_cells, min(conversation, cells - fixed_cells)


def gauge_bar(fixed: float, used: float, window: int, cells: int, colours: dict[str, str],
              g: SimpleNamespace) -> Text:
    """The bar alone: fixed prompt, conversation, empty track, and the compaction tick."""
    fixed_cells, conversation = gauge_cells(fixed, used, window, cells)
    tick = round(context.COMPACT_AT * cells)
    bar = Text()
    for index in range(cells):
        if index == tick:
            bar.append(g.tick, style=colours["tick"])
        if index < fixed_cells:
            bar.append(g.fixed, style=colours["fixed"])
        elif index < fixed_cells + conversation:
            bar.append(g.used, style=colours["accent"])
        else:
            bar.append(g.empty, style=colours["track"])
    return bar


# -- small widgets ---------------------------------------------------------------------------------

class ContextGauge(Widget):
    """Segmented gauge: fixed prompt, conversation (accent), empty track, tick at the compaction
    point. The fixed segment is frozen once the server has measured it, so it never jumps."""

    fixed: reactive[float] = reactive(0.0)
    used: reactive[float] = reactive(0.0)  # total tokens; animated once when context is compacted
    window: reactive[int] = reactive(context.DEFAULT_NUM_CTX)
    cells: reactive[int] = reactive(GAUGE_CELLS)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.frozen = False  # fixed segment measured and frozen
        self.remeasured = False  # the one change from estimate to measurement has happened
        self.parts: Any = None  # the latest tui_context.Breakdown (the Context pane splits it)

    def set_usage(self, usage: Usage, *, settle: bool = False) -> None:
        self.window = usage.window
        if usage.parts is not None:
            self.parts = usage.parts
        if not self.frozen:
            if usage.measured and self.fixed and abs(usage.fixed - self.fixed) > 1:
                # the single re-measure: estimate -> server measurement, shown once and marked
                self.remeasured = True
                if self.app.animation_level != "none":
                    self.animate("fixed", float(usage.fixed), duration=0.6, easing="out_cubic",
                                 level="basic")
                else:
                    self.fixed = float(usage.fixed)
            else:
                self.fixed = float(usage.fixed)
            self.frozen = usage.measured
        if settle and usage.total < self.used and self.app.animation_level != "none":
            # after compaction the bar settles back once
            self.animate("used", float(usage.total), duration=0.6, easing="out_cubic",
                         level="basic")
        else:
            self.used = float(usage.total)

    def watch_used(self) -> None:
        self._changed()

    def watch_fixed(self) -> None:
        self._changed()

    def _changed(self) -> None:
        if self.is_mounted and isinstance(self.app, VegaApp):
            self.app.gauge_changed()

    def render(self) -> Text:
        colours = palette(self.app)
        text = Text("context ", style=colours["muted"])
        text.append_text(gauge_bar(self.fixed, self.used, self.window, self.cells, colours,
                                   glyphs()))
        percent = round(100 * self.used / max(1, self.window))
        text.append(f"  {percent}% of {window_label(self.window)}", style=colours["readout"])
        return text

    def get_content_width(self, container: Any, viewport: Any) -> int:
        return cell_len(self.render().plain)


class PromptInput(TextArea):
    """Multi-line input: Enter sends; Shift+Enter (Ctrl+J where the terminal sends Shift+Enter as
    Enter; Alt+Enter only where it is reported as one key) adds a line; ↑/↓ at the
    first/last line walk through earlier messages."""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(placeholder=PLACEHOLDER, soft_wrap=True, show_line_numbers=False,
                         tab_behavior="focus", **kwargs)
        self.sent: list[str] = []
        self._browse: int | None = None
        self._draft = ""

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return
        if event.key in {"shift+enter", "ctrl+j", "alt+enter"}:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if event.key == "up" and self.cursor_location[0] == 0 and self.sent:
            event.stop()
            event.prevent_default()
            if self._browse is None:
                self._draft, self._browse = self.text, len(self.sent)
            self._browse = max(0, self._browse - 1)
            self._show(self.sent[self._browse])
            return
        if (event.key == "down" and self._browse is not None
                and self.cursor_location[0] == self.document.line_count - 1):
            event.stop()
            event.prevent_default()
            self._browse += 1
            if self._browse >= len(self.sent):
                self._browse = None
                self._show(self._draft)
            else:
                self._show(self.sent[self._browse])
            return
        await super()._on_key(event)

    def _show(self, text: str) -> None:
        self.load_text(text)
        self.move_cursor(self.document.end)

    def remember(self, text: str) -> None:
        if text and (not self.sent or self.sent[-1] != text):
            self.sent.append(text)
        self._browse = None
        self._draft = ""


GUTTERS = {
    "classic": {"you": "you", "vega": "vega"},
    "quiet": {"you": "›", "vega": ""},
    "observatory": {"you": "you", "vega": "vega"},
    "workbench": {"you": "you", "vega": "vega"},
}


class Gutter(Static):
    """The speaker column; its label depends on the layout (quiet marks only your messages)."""

    def __init__(self, speaker: str = "", **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self.speaker = speaker

    def render(self) -> str:
        if not self.speaker:
            return ""
        layout = getattr(self.app, "layout_name", "classic")
        label = GUTTERS.get(layout, GUTTERS["classic"]).get(self.speaker, self.speaker)
        if label == "›" and glyphs().prompt != "›":
            label = glyphs().prompt
        return label


class UserMessage(Horizontal):
    def __init__(self, text: str) -> None:
        super().__init__(classes="message user")
        self._text = text

    def compose(self) -> ComposeResult:
        yield Gutter("you", classes="gutter you")
        yield Static(self._text, markup=False, classes="user-text")


class Note(Horizontal):
    """A line outside a turn: command output, undo results, notices. kind: info|warning|output."""

    def __init__(self, body: Text | str, kind: str = "info", *, dismissible: bool = False) -> None:
        classes = f"message note {kind}" + (" dismissible" if dismissible else "")
        super().__init__(classes=classes)
        self._body = body

    def compose(self) -> ComposeResult:
        yield Gutter("", classes="gutter")
        yield Static(self._body, markup=False, classes="note-body")

    def on_click(self) -> None:
        if self.has_class("dismissible"):
            self.remove()


@dataclass
class Entry:
    kind: str  # step | note | notice | plan
    name: str = ""
    target: str = ""
    outcome: str = ""
    level: str = ""
    running: bool = False
    items: list[dict[str, str]] = field(default_factory=list)
    result: str = ""  # a step's full tool result (the workbench preview shows it)


def plan_position(items: list[dict[str, str]]) -> tuple[str, str]:
    """(item, position) for a plan's one-row summary: the item in progress and `step 2 of 3`."""
    if not items:
        return "plan cleared", ""
    count = len(items)
    for index, item in enumerate(items):
        if item.get("status") == "in_progress":
            return item.get("text", ""), f"step {index + 1} of {count}"
    if all(item.get("status") == "done" for item in items):
        return items[-1].get("text", ""), f"all {count} done"
    for index, item in enumerate(items):
        if item.get("status") != "done":
            return item.get("text", ""), f"step {index + 1} of {count}, not started"
    return items[-1].get("text", ""), ""


def write_lines(args: dict[str, Any]) -> int | None:
    content = args.get("content")
    if not isinstance(content, str):
        return None
    return len(content.splitlines())


class Steps(Static):
    """A turn's steps as aligned columns (tool, target, outcome), plus retry notes and plan rows.
    Rebuilt on every change, resize and theme switch, so columns always fit the pane."""

    def __init__(self) -> None:
        super().__init__("", classes="steps")
        self.entries: list[Entry] = []

    @property
    def step_count(self) -> int:
        return sum(entry.kind == "step" for entry in self.entries)

    @property
    def failed(self) -> bool:
        return any(entry.kind == "step" and entry.level == "error" for entry in self.entries)

    def first_warning_row(self) -> int | None:
        for index, entry in enumerate(self.entries):
            if entry.kind == "step" and entry.level in {"warning", "error"}:
                return index
        return None

    def add(self, entry: Entry) -> None:
        self.entries.append(entry)
        self.rebuild()

    def finish(self, name: str, args: dict[str, Any], result: str) -> Entry:
        outcome, level = step_outcome(name, result)
        lines = write_lines(args) if name == "write_file" and level == "" else None
        if lines is not None:
            outcome = f"{lines} line{'s' if lines != 1 else ''}"
        for entry in reversed(self.entries):
            if entry.kind == "step" and entry.running and entry.name == name:
                entry.running, entry.outcome, entry.level = False, outcome, level
                entry.result = result
                break
        else:
            entry = Entry("step", name, step_summary(name, args), outcome, level, result=result)
            self.entries.append(entry)
        self.rebuild()
        return entry

    def on_resize(self) -> None:
        self.rebuild()

    def rebuild(self) -> None:
        self.update(self.render_entries(self.content_region.width or 72))

    def render_entries(self, width: int) -> Text:
        g = glyphs()
        colours = palette(self.app)
        rows = [entry for entry in self.entries if entry.kind in {"step", "plan"}]
        names = [entry.name if entry.kind == "step" else "plan" for entry in rows]
        tool_w = min(14, max((cell_len(name) for name in names), default=8))
        rest = max(10, width - tool_w - 4)
        targets = [self._target(entry) for entry in rows]
        longest_target = max((cell_len(target) for target in targets), default=0)
        longest_outcome = max((cell_len(self._outcome(entry, g)) for entry in rows), default=0)
        target_w = min(longest_target, max(12, rest - min(longest_outcome, rest // 2)))
        outcome_w = max(0, rest - target_w)
        styles = {"": colours["muted"], "ok": colours["success"], "warning": colours["warning"],
                  "error": colours["error"], "waiting": colours["warning"]}
        text = Text()
        for index, entry in enumerate(self.entries):
            if index:
                text.append("\n")
            if entry.kind in {"step", "plan"}:
                plan = entry.kind == "plan"
                name = "plan" if plan else entry.name
                name_style = colours["muted"] if plan else f"bold {colours['fg']}"
                text.append(name.ljust(tool_w), style=name_style)
                text.append("  ")
                text.append(_clip(self._target(entry), target_w, keep_end=not plan)
                            .ljust(target_w), style=colours["fg"])
                text.append("  ")
                outcome = _clip(self._outcome(entry, g), outcome_w)
                text.append(outcome, style=colours["muted"] if plan
                            else styles.get(entry.level, colours["muted"]))
            elif entry.kind == "note":
                text.append(_clip(entry.outcome, width), style=f"italic {colours['muted']}")
            elif entry.kind == "notice":
                text.append(entry.outcome, style=colours["warning"])
        return text

    @staticmethod
    def _target(entry: Entry) -> str:
        return plan_position(entry.items)[0] if entry.kind == "plan" else entry.target

    @staticmethod
    def _outcome(entry: Entry, g: SimpleNamespace) -> str:
        if entry.kind == "plan":
            return plan_position(entry.items)[1]
        if entry.running and entry.level == "waiting":
            return "waiting for you"
        if entry.running:
            return f"running{g.ellipsis if g.ellipsis == '…' else '...'}"
        if entry.level in {"warning", "error"}:
            return f"{g.warn} {entry.outcome}"
        return entry.outcome


def plan_item(item: dict[str, str], width: int, g: SimpleNamespace, colours: dict[str, str],
              *, failed: bool = False) -> Text:
    status = item.get("status", "pending")
    mark = {"done": g.done, "in_progress": g.doing}.get(status, g.pending)
    style = {"done": colours["muted"], "in_progress": f"bold {colours['fg']}"}.get(
        status, colours["fg"])
    mark_style = (colours["success"] if status == "done" else
                  colours["accent"] if status == "in_progress" else colours["muted"])
    if failed:
        mark, mark_style, style = g.warn, colours["error"], colours["error"]
    line = Text()
    line.append(mark, style=mark_style)
    label = item.get("text", "")
    line.append(" " + _clip(label, max(8, width - 2)), style=style)
    return line


class StepsToggle(Static):
    """`▸ 3 steps, 9.4 s` under a finished answer; Enter or a click shows the steps."""

    can_focus = True
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter,space", "toggle", "Show steps", show=False),
    ]

    def __init__(self) -> None:
        super().__init__("", classes="toggle")
        self.count = 0
        self.seconds = 0.0
        self.expanded = False

    def set_summary(self, count: int, seconds: float) -> None:
        self.count, self.seconds = count, seconds
        self.refresh_label()

    def refresh_label(self) -> None:
        g = glyphs()
        if not self.count:
            self.update(f"{seconds_label(self.seconds)}, no tools used")
            return
        mark = g.opened if self.expanded else g.closed
        noun = "step" if self.count == 1 else "steps"
        self.update(f"{mark} {self.count} {noun}, {seconds_label(self.seconds)}")

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self.refresh_label()
        turn = next((node for node in self.ancestors if isinstance(node, Turn)), None)
        if turn is not None:
            turn.steps.display = self.expanded

    def action_toggle(self) -> None:
        if self.count:
            self.set_expanded(not self.expanded)

    def on_click(self) -> None:
        self.action_toggle()


class Turn(Horizontal):
    """One agent turn: live steps and a working line, then the answer with the steps collapsed
    (expanded when a check failed). `changes` is the compact changed-files row (quiet layout)."""

    def __init__(self, number: int = 0) -> None:
        super().__init__(classes="message turn")
        self.number = number
        self.steps = Steps()
        self.working = Static("", classes="working")
        self.answer = Markdown("", classes="answer")
        self.lesson = Static("", classes="lesson")  # teach mode: plan, why and fix lines
        self.lessons: list[tuple[str, str]] = []
        self.takeaway = Markdown("", classes="takeaway")  # teach mode: the answer's sections
        self.toggle = StepsToggle()
        self.changes = Static("", classes="turn-changes -empty")
        self.changed_files: list[tuple[str, int, int]] = []
        self.started = time.monotonic()
        self.done = False
        self.reply = ""  # the answer as the model wrote it (Markdown); /copy copies it
        self.request = ""  # the first line of the message (the Product pane's session list)

    def compose(self) -> ComposeResult:
        yield Gutter("vega", classes="gutter vega")
        with Vertical(classes="turn-body"):
            yield self.lesson
            yield self.answer
            yield self.takeaway
            yield self.toggle
            yield self.steps
            yield self.changes
            yield self.working

    def on_mount(self) -> None:
        self.lesson.display = bool(self.lessons)
        if self.done:  # a reply that arrived before the turn was mounted keeps its answer
            return
        self.takeaway.display = False
        self.answer.display = False
        self.toggle.display = False
        self.steps.display = False

    def add(self, entry: Entry) -> None:
        self.steps.display = step_detail(self) != "hidden"
        self.steps.add(entry)

    def add_lesson(self, role: str, text: str) -> None:
        self.lessons.append((role, text))
        self.render_lesson()

    def render_lesson(self) -> None:
        """Teach mode's lines: a short label in the teaching tint, the model's words in the muted
        text colour, wrapped under themselves."""
        if not self.lessons:
            return
        self.lesson.display = True
        self.lesson.update(lesson_table(self.lessons, palette(self.app)))

    def finish(self, text: str, seconds: float) -> None:
        self.done = True
        self.reply = text
        self.working.remove()
        self.answer.display = True
        body, sections = teaching.split(text)
        self.answer.update(body if sections and body else text)
        if sections and body:
            self.takeaway.update(sections)
            self.takeaway.display = True
        if text == STOPPED:
            self.answer.add_class("stopped")
        self.toggle.set_summary(self.steps.step_count, seconds)
        self.apply_detail(step_detail(self))

    def apply_detail(self, detail: str) -> None:
        """collapsed: a summary line (a failed check stays open); expanded: every step; hidden:
        neither (the workbench Steps pane lists them)."""
        if not self.done:
            self.steps.display = detail != "hidden" and bool(self.steps.entries)
            return
        self.toggle.display = detail != "hidden"
        self.toggle.set_expanded(detail == "expanded" or (detail == "collapsed"
                                                           and self.steps.failed))
        if detail == "hidden":
            self.steps.display = False


LESSON_LABELS = {"plan": "plan", "why": "why", "failure": "fix"}


def lesson_table(lessons: list[tuple[str, str]], colours: dict[str, str]) -> Any:
    from rich.table import Table

    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=4, no_wrap=True)
    grid.add_column(ratio=1)
    for role, text in lessons:
        grid.add_row(Text(LESSON_LABELS.get(role, role), style=f"bold {colours['teach']}"),
                     Text(text, style=colours["muted"] if role != "plan" else colours["fg"]))
    return grid


def step_detail(widget: Widget) -> str:
    """The app's step-detail setting (collapsed outside a running app, e.g. in unit tests)."""
    try:
        return getattr(widget.app, "step_detail", "collapsed")
    except Exception:  # noqa: BLE001 - no active app
        return "collapsed"


_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[^\n]*\n(.*?)^[ \t]{0,3}\1[`~]*[ \t]*$",
                    re.MULTILINE | re.DOTALL)


def code_blocks(markdown: str) -> list[str]:
    """The fenced code blocks of a reply, without their fences."""
    return [match.group(2).rstrip("\n") for match in _FENCE.finditer(markdown)]


def characters_label(count: int) -> str:
    return f"{count:,} character{'s' if count != 1 else ''}"


# -- column dividers -------------------------------------------------------------------------------

class ColumnDivider(Widget):
    """The workbench conversation's left or right edge, drawn over its border (an overlay, so it
    costs no width) so it can be dragged: mouse down and drag resizes that side column live
    (clamped to view.py's limits), release saves the width like /width, double-click resets it. At
    rest it looks like the border (rule colour, accent while the conversation has focus); under the
    mouse and while dragging it turns into a heavy accent line."""

    ALLOW_SELECT = False

    def __init__(self, side: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.side = side  # "left" | "right"
        self.hover = False
        self.dragging = False
        self._start_x = 0
        self._start_width = 0

    def render(self) -> Text:
        colours = palette(self.app)
        app = self.app
        focused = isinstance(app, VegaApp) and app.query_one("#conversation").has_focus_within
        active = self.hover or self.dragging
        colour = colours["accent"] if active or focused else colours["rule"]
        if active:
            top, mid, bottom = ("┎", "┃", "┖") if self.side == "left" else ("┒", "┃", "┚")
        else:
            top, mid, bottom = ("┌", "│", "└") if self.side == "left" else ("┐", "│", "┘")
        height = max(1, self.size.height)
        rows = [top] + [mid] * max(0, height - 2) + ([bottom] if height > 1 else [])
        style = f"bold {colour}" if active else colour
        return Text("\n".join(rows), style=style)

    def on_enter(self, event: events.Enter) -> None:
        self.hover = True
        self.refresh()
        if isinstance(self.app, VegaApp) and not self.dragging:
            self.app.column_hint(self.side)

    def on_leave(self, event: events.Leave) -> None:
        self.hover = False
        self.refresh()
        if isinstance(self.app, VegaApp) and not self.dragging:
            self.app.column_hint(None)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 1 or not isinstance(self.app, VegaApp):
            return
        event.stop()
        self.dragging = True
        self._start_x = event.screen_x
        self._start_width = self.app.column_width(self.side)
        self.capture_mouse()
        self.app.column_drag_started(self.side)
        self.refresh()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if not self.dragging or not isinstance(self.app, VegaApp):
            return
        event.stop()
        moved = event.screen_x - self._start_x
        self.app.drag_column(self.side, self._start_width + (moved if self.side == "left"
                                                             else -moved))

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if not self.dragging:
            return
        event.stop()
        self.dragging = False
        self.release_mouse()
        if isinstance(self.app, VegaApp):
            self.app.column_drag_finished(self.side)
            self.app.column_hint(self.side if self.hover else None)
        self.refresh()

    def on_click(self, event: events.Click) -> None:
        event.stop()
        if event.chain == 2 and isinstance(self.app, VegaApp):
            self.app.reset_column(self.side)


# -- modal screens ---------------------------------------------------------------------------------

def diff_text(diff: str, colours: dict[str, str]) -> Text:
    """A unified diff coloured by line type; the +/- markers carry the meaning without colour."""
    text = Text()
    for index, line in enumerate(diff.rstrip("\n").splitlines()):
        if index:
            text.append("\n")
        if line.startswith(("+++", "---")):
            style = f"bold {colours['muted']}"
        elif line.startswith("@@"):
            style = colours["accent"]
        elif line.startswith("+"):
            style = colours["success"]
        elif line.startswith("-"):
            style = colours["error"]
        else:
            style = colours["fg"]
        text.append(line, style=style)
    return text


def diff_rows(before: list[str], after: list[str], width: int, colours: dict[str, str],
              g: SimpleNamespace) -> Text:
    """A side-by-side-free diff for the preview pane: line number, marker, text, with added and
    removed rows tinted across the full width. The marker column carries the meaning."""
    text = Text()
    number_w = max(3, len(str(max(len(before), len(after), 1))))
    body_w = max(10, width - number_w - 3)
    groups = list(difflib.SequenceMatcher(None, before, after).get_grouped_opcodes(3))
    if not groups:
        return Text("No textual change", style=colours["muted"])

    def row(number: int | None, mark: str, line: str, style: str, bg: str) -> None:
        text.append("\n")
        num = str(number).rjust(number_w) if number is not None else " " * number_w
        text.append(num + " ", style=f"{colours['muted']} on {bg}" if bg else colours["muted"])
        text.append(mark + " ", style=f"bold {style} on {bg}" if bg else style)
        content = _clip(line.expandtabs(4), body_w).ljust(body_w)
        text.append(content, style=f"{colours['fg']} on {bg}" if bg else colours["fg"])

    for number, group in enumerate(groups):
        first, last = group[0][3] + 1, group[-1][4]
        if number:
            text.append("\n\n")
        dash = "–" if g.ellipsis == "…" else "-"
        text.append(f"lines {first}{dash}{last}", style=colours["accent"])
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for offset, line in enumerate(after[j1:j2]):
                    row(j1 + offset + 1, " ", line, colours["muted"], "")
                continue
            for offset, line in enumerate(before[i1:i2]):
                row(i1 + offset + 1, g.minus if g.minus != "-" else "-", line, colours["error"],
                    colours["del_bg"])
            for offset, line in enumerate(after[j1:j2]):
                row(j1 + offset + 1, "+", line, colours["success"], colours["add_bg"])
    return text


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def unified_rows(diff: str, width: int, colours: dict[str, str], g: SimpleNamespace) -> Text:
    """A unified diff (git's) drawn like diff_rows: `lines a–b` heads, line numbers, tinted added
    and removed rows. Lines before the first file header (a commit's message and stat) are kept
    as they are; per-file headers become one bold path line."""
    text = Text()
    lines = diff.splitlines()
    numbers = [int(m.group(3)) + int(m.group(4) or 1) for m in map(_HUNK.match, lines) if m]
    number_w = max(3, len(str(max(numbers, default=1))))
    body_w = max(10, width - number_w - 3)
    dash = "–" if g.ellipsis == "…" else "-"
    old = new = 0
    in_patch = False

    def row(number: int | None, mark: str, line: str, style: str, bg: str) -> None:
        text.append("\n")
        num = str(number).rjust(number_w) if number is not None else " " * number_w
        text.append(num + " ", style=f"{colours['muted']} on {bg}" if bg else colours["muted"])
        text.append(mark + " ", style=f"bold {style} on {bg}" if bg else style)
        content = _clip(line.expandtabs(4), body_w)
        text.append(content.ljust(body_w) if bg else content,
                    style=f"{colours['fg']} on {bg}" if bg else colours["fg"])

    for line in lines:
        if line.startswith("diff --git "):
            in_patch = True
            path = line.split(" b/", 1)[-1]
            text.append(("\n\n" if text else "") + path, style=f"bold {colours['fg']}")
            continue
        if not in_patch:
            text.append(("\n" if text else "") + line, style=colours["fg"])
            continue
        hunk = _HUNK.match(line)
        if hunk:
            old, new = int(hunk.group(1)), int(hunk.group(3))
            last = new + max(0, int(hunk.group(4) or 1) - 1)
            text.append(f"\nlines {new}{dash}{last}", style=colours["accent"])
        elif line.startswith(("index ", "--- ", "+++ ", "new file", "deleted file",
                              "similarity", "rename ", "old mode", "new mode")):
            if line.startswith(("new file", "deleted file", "rename ")):
                text.append("\n" + line, style=colours["muted"])
        elif line.startswith("+"):
            row(new, "+", line[1:], colours["success"], colours["add_bg"])
            new += 1
        elif line.startswith("-"):
            row(old, g.minus if g.minus != "-" else "-", line[1:], colours["error"],
                colours["del_bg"])
            old += 1
        elif line.startswith("\\"):
            text.append("\n" + line, style=colours["muted"])
        else:
            row(new, " ", line[1:], colours["muted"], "")
            old += 1
            new += 1
    return text


class ChoiceScreen(ModalScreen[bool]):
    """Centered y/n dialog with two buttons. Escape answers no."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "answer(True)", "Yes", show=False),
        Binding("n,escape", "answer(False)", "No", show=False),
    ]

    def __init__(self, title: str, body: Text | str, yes: str, no: str, *,
                 detail: Callable[[dict[str, str]], Text] | None = None,
                 hint: str = "y yes   n no") -> None:
        super().__init__(classes="choice")
        self.title_text, self.body, self.yes, self.no = title, body, yes, no
        self.detail, self.hint = detail, hint  # detail is drawn from the current theme

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.title_text, id="dialog-title", markup=False)
            if self.body:
                yield Static(self.body, id="dialog-body", markup=False)
            if self.detail is not None:
                with VerticalScroll(id="dialog-detail"):
                    yield Static("", id="dialog-detail-text", markup=False)
            with Horizontal(id="dialog-buttons"):
                yield Button(self.yes, id="yes", classes="primary", compact=True)
                yield Button(self.no, id="no", classes="secondary", compact=True)
                yield Static(self.hint, id="dialog-hint")

    def on_mount(self) -> None:
        self.restyle()
        self.query_one("#yes", Button).focus()

    def restyle(self) -> None:
        if self.detail is not None:
            self.query_one("#dialog-detail-text", Static).update(self.detail(palette(self.app)))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(event.button.id == "yes")

    def action_answer(self, value: bool) -> None:
        self.dismiss(value)


def review_verb(plan: WritePlan) -> str:
    if plan.before is None:
        return "Create"
    return "Overwrite" if plan.tool == "write_file" else "Edit"


def review_screen(plan: WritePlan) -> ChoiceScreen:
    diff = write_diff(plan)

    def detail(colours: dict[str, str]) -> Text:
        return diff_text(diff, colours) if diff else Text("(no textual change)", colours["muted"])

    return ChoiceScreen(
        f"{review_verb(plan)} {plan.relative}?",
        "Vega wants to make this change. Review it before it is written.",
        "Apply change", "Skip", detail=detail, hint="y apply   n skip",
    ).add_class("review")


def command_screen(command: str, project: Path) -> ChoiceScreen:
    return ChoiceScreen(
        "Run this command?",
        f"Vega wants to run it in {short_path(project)}. It is not on the read-only list, so it "
        "could change files or reach the network.",
        "Run command", "Don't run",
        detail=lambda colours: Text(command, style=f"bold {colours['fg']}"),
        hint="y run   n don't run",
    ).add_class("command")


class WebAccessScreen(ChoiceScreen):
    """Web research permission, asked once per session: allow for this session, always for this
    project, or no. Dismisses with "session", "project" or "no"."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("s,y", "web('session')", "This session", show=False),
        Binding("p,a", "web('project')", "Always for this project", show=False),
        Binding("n,escape", "web('no')", "No", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.title_text, id="dialog-title", markup=False)
            yield Static(self.body, id="dialog-body", markup=False)
            with Horizontal(id="dialog-buttons"):
                yield Button("Allow this session", id="yes", classes="primary", compact=True)
                yield Button("Always for this project", id="always", classes="secondary",
                             compact=True)
                yield Button("No", id="no", classes="secondary", compact=True)
                yield Static(self.hint, id="dialog-hint")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss({"yes": "session", "always": "project"}.get(event.button.id or "", "no"))

    def action_answer(self, value: bool) -> None:
        self.dismiss("session" if value else "no")

    def action_web(self, choice: str) -> None:
        self.dismiss(choice if choice in web.ANSWERS else "no")


class AdapterScreen(ChoiceScreen):
    """web_adapter_builder: approve a verified site adapter before its first use. Approve, view
    the code (shown in the detail panel), or deny. Dismisses with True (approve) or False."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("a,y", "answer(True)", "Approve", show=False),
        Binding("v", "view", "View code", show=False),
        Binding("n,escape", "answer(False)", "Deny", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.title_text, id="dialog-title", markup=False)
            yield Static(self.body, id="dialog-body", markup=False)
            with VerticalScroll(id="dialog-detail"):
                yield Static("", id="dialog-detail-text", markup=False)
            with Horizontal(id="dialog-buttons"):
                yield Button("Approve", id="yes", classes="primary", compact=True)
                yield Button("View code", id="view", classes="secondary", compact=True)
                yield Button("Deny", id="no", classes="secondary", compact=True)
                yield Static(self.hint, id="dialog-hint")

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#dialog-detail").display = False

    def action_view(self) -> None:
        panel = self.query_one("#dialog-detail")
        panel.display = not panel.display

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "view":
            self.action_view()
            return
        self.dismiss(event.button.id == "yes")


def adapter_screen(candidate: Any) -> AdapterScreen:
    from . import adapter_builder

    title, body = adapter_builder.approval_text(candidate)
    return AdapterScreen(title, body, "Approve", "Deny",
                         detail=lambda colours: Text(candidate.code, style=colours["fg"]),
                         hint="a approve   v view code   n deny").add_class("command")


def web_screen(kind: str, detail: str, backend: str) -> WebAccessScreen:
    title, body = web.prompt_text(kind, detail, backend)
    return WebAccessScreen(title, body, "Allow this session", "No",
                           hint="s session   p project   n no").add_class("web")


HELP_KEYS = [
    ("Enter", "send"),
    ("Shift+Enter", "new line (Ctrl+J where Shift+Enter is not passed through)"),
    ("↑ / ↓", "earlier messages, from the first or last line of the input"),
    ("Esc / Ctrl+C", "stop now (twice quits); Ctrl+C copies a selection first"),
    ("Mouse drag", "select text (double-click a block, triple-click a reply)"),
    ("Option+drag", "the terminal's own selection, for Cmd+C (Shift+drag on Linux)"),
    ("Cmd+V", "paste into the message; pasted newlines never send it"),
    ("Ctrl+U", "undo the last turn's file changes"),
    ("Ctrl+R", "review edits on or off"),
    ("Ctrl+P", "show or hide the panel (Esc also closes it)"),
    ("Ctrl+T", "light or dark theme"),
    ("Ctrl+L", "clear the view (Vega keeps the conversation)"),
    ("Ctrl+Q", "quit"),
    ("Tab", "next pane (workbench), otherwise the next control"),
    ("u", "on a changed file or turn: undo back to that turn"),
    ("Enter", "Git, Tests, Problems: show the diff, failure or line in the Preview"),
    ("r", "Git, Models: read again; Tests pane: run the last tests again (Esc stops)"),
    ("l", "Problems pane: run ruff check on the files changed this session"),
    ("Enter / d / x", "Models pane: switch to, download or remove a model (each asks first)"),
    ("Enter", "on a step summary: show or hide the steps"),
    ("F1", "this help"),
    ("F2", "customize the workbench: panes, widths, steps, status bar"),
    ("F3", "product view or code view (narrow terminals: the product screen; 1 and 5 pick a tab)"),
    ("f", "Product, Release: put a prepared fix request in the message box (never sent)"),
    ("Enter", "on a Release check: what it found and where that came from"),
    ("Drag an edge", "workbench: drag the conversation's edge to resize a column"),
    ("Double-click it", "put that column back to its default width"),
]
HELP_COMMANDS = [
    ("/status", "Git status"),
    ("/diff", "tracked changes against HEAD"),
    ("/undo [--force]", "restore the files changed in the last turn"),
    ("/review on|off", "ask before applying each edit"),
    ("/teach on|off", "teach mode: Vega says why as it works and ends with what to take away"),
    ("/layout [name]", "workbench (default), classic, quiet or observatory"),
    ("/view [preset]", ("what the workbench shows; presets product, workbench, focus, "
                        "review, monitor, steps")),
    ("/product", "the same as F3"),
    ("/ship", "not built yet: the Release checklist shows what is left"),
    ("/usage on|off", "the local usage log (off by default; events only, never sent)"),
    ("/pane NAME WHERE", "move a pane: left, right, hide, up or down"),
    ("/width SIDE N", "column width: left 24-64, right 30-100 or auto"),
    ("/steps DETAIL", "steps in the conversation: collapsed, expanded or hidden"),
    ("/learn SOURCE", "learn a web page, PDF, document or folder into Vega's library"),
    ("/open [URL]", "open a URL (default: the last page a site refused to Vega) in your browser"),
    ("/copy [N]", "copy the last reply (or turn N's) as Markdown"),
    ("/copy code [N]", "copy only the code blocks of the last reply (or turn N's)"),
    ("/transcript", "show this session's transcript path and copy it"),
    ("/reset", "clear the conversation context"),
    ("/help", "this help"),
    ("/exit", "leave"),
]


class HelpScreen(ModalScreen[None]):
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape,f1,q", "close", "Close", show=False)]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Help", id="dialog-title")
            with VerticalScroll(id="dialog-detail"):
                yield Static("", id="help-body")
            yield Static("Esc closes", id="dialog-hint")

    def on_mount(self) -> None:
        self.restyle()

    def restyle(self) -> None:
        colours = palette(self.app)
        body = Text()
        for title, rows in (("Keys", HELP_KEYS), ("Commands", HELP_COMMANDS)):
            if body:
                body.append("\n\n")
            body.append(title, style=f"bold {colours['fg']}")
            for key, what in rows:
                body.append("\n  ")
                body.append(key.ljust(17), style=f"bold {colours['accent']}")
                body.append(what, style=colours["fg"])
        self.query_one("#help-body", Static).update(body)

    def action_close(self) -> None:
        self.dismiss(None)


class ViewScreen(ModalScreen[None]):
    """F2: customize the workbench. Changes show live behind the dialog; Enter keeps them (saved
    for the next start), Esc puts the previous view back."""

    PLACES = ("left", "right", "hide")
    ROWS: ClassVar[list[tuple[str, str]]] = [
        *(("pane", name) for name in view.PANES),
        ("width", "left"), ("width", "right"), ("steps", ""), ("status", ""),
    ]
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up,k", "move(-1)", show=False),
        Binding("down,j", "move(1)", show=False),
        Binding("left,h", "change(-1)", show=False),
        Binding("right,l", "change(1)", show=False),
        Binding("left_square_bracket", "reorder(-1)", show=False),
        Binding("right_square_bracket", "reorder(1)", show=False),
        Binding("g", "status('gauge')", show=False),
        Binding("m", "status('memory')", show=False),
        Binding("r", "status('reply')", show=False),
        Binding("y", "status('keys')", show=False),
        Binding("enter", "keep", show=False),
        Binding("escape,f2", "cancel", show=False),
    ]

    def __init__(self, current: view.View) -> None:
        super().__init__()
        self.original = current
        self.view = current
        self.row = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="view-dialog"):
            yield Static("", id="view-body")

    def on_mount(self) -> None:
        self.query_one("#view-dialog").border_title = "Customize the workbench"
        self._fit()
        self.restyle()

    def on_resize(self) -> None:
        self._fit()

    def _fit(self) -> None:
        """80% of the screen keeps the workbench in view above the dialog; on a short terminal
        (24 rows) every row of the dialog matters more, so it may take the whole height."""
        dialog = self.query_one("#view-dialog")
        dialog.styles.max_height = "100%" if self.app.size.height < 30 else "80%"

    def on_key(self, event: events.Key) -> None:
        if event.character and event.character in "123456"[:len(view.PRESETS)]:
            names = list(view.PRESETS)
            index = int(event.character) - 1
            if index < len(names):
                self._set(view.preset(names[index]))
            event.stop()

    def _set(self, new: view.View) -> None:
        self.view = new
        self.app.apply_view(new, save=False)  # type: ignore[attr-defined]
        self.restyle()

    def action_move(self, step: int) -> None:
        self.row = (self.row + step) % len(self.ROWS)
        self.restyle()

    def action_change(self, step: int) -> None:
        kind, key = self.ROWS[self.row]
        current = self.view
        if kind == "pane":
            where = current.column(key) or "hide"
            index = max(0, min(2, self.PLACES.index(where) + step))
            if self.PLACES[index] != where:
                self._set(view.place(current, key, self.PLACES[index]))
        elif kind == "width" and key == "left":
            self._set(view.resize(current, "left", current.left_width + 2 * step))
        elif kind == "width":
            width = current.right_width
            if width == 0:
                width = view.RIGHT_WIDTHS[0] if step > 0 else 0
            elif width == view.RIGHT_WIDTHS[0] and step < 0:
                width = 0
            else:
                width += 2 * step
            self._set(view.resize(current, "right", width))
        elif kind == "steps":
            details = list(view.STEP_DETAIL)
            index = (details.index(current.steps) + step) % len(details)
            self._set(view.with_steps(current, details[index]))
        elif kind == "status":
            return

    def action_reorder(self, step: int) -> None:
        kind, key = self.ROWS[self.row]
        if kind == "pane":
            self._set(view.shift(self.view, key, step))

    def action_status(self, item: str) -> None:
        self._set(view.toggle_status(self.view, item))

    def action_keep(self) -> None:
        self.app.apply_view(self.view, save=True)  # type: ignore[attr-defined]
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.app.apply_view(self.original, save=False)  # type: ignore[attr-defined]
        self.dismiss(None)

    def restyle(self) -> None:
        colours, g = palette(self.app), glyphs()
        fg, muted, accent = colours["fg"], colours["muted"], colours["accent"]
        text = Text()
        text.append("Presets  ", style=muted)
        for number, name in enumerate(view.PRESETS, 1):
            style = f"bold {accent}" if name == self.view.preset else fg
            text.append(f"{number} ", style=muted)
            text.append(name, style=style)
            text.append("   ")
        text.append("\n\n" + " " * 14, style=muted)
        for place in self.PLACES:
            text.append(("hidden" if place == "hide" else place).ljust(8), style=muted)
        selected_about = ""
        for index, (kind, key) in enumerate(self.ROWS):
            chosen = index == self.row
            text.append("\n")
            text.append(f"{g.prompt} " if chosen else "  ", style=accent)
            style = f"bold {fg}" if chosen else fg
            if kind == "pane":
                text.append(view.PANE_TITLES[key].ljust(12), style=style)
                where = self.view.column(key) or "hide"
                for place in self.PLACES:
                    mark = "●" if place == where else "·"
                    if g.ellipsis != "…":
                        mark = "*" if place == where else "."
                    text.append(mark.ljust(8), style=accent if place == where else muted)
                if where in {"left", "right"}:
                    column = getattr(self.view, where)
                    text.append(f"{column.index(key) + 1} of {len(column)}", style=muted)
                if chosen:
                    selected_about = view.PANES[key]
            elif kind == "width":
                text.append(f"{key} width".ljust(12), style=style)
                if key == "left":
                    value = f"{self.view.left_width} columns"
                else:
                    value = (f"{self.view.right_width} columns" if self.view.right_width
                             else "auto (shares with the conversation)")
                text.append(value, style=accent)
                if chosen:
                    selected_about = "Left and right change the width."
            elif kind == "steps":
                text.append("steps".ljust(12), style=style)
                text.append(self.view.steps, style=accent)
                if chosen:
                    selected_about = "Steps in the conversation: " + \
                        view.STEP_DETAIL[self.view.steps] + "."
            else:
                text.append("status bar".ljust(12), style=style)
                for item, letter in zip(view.STATUS_ITEMS, "gmry", strict=True):
                    on = item in self.view.status
                    text.append(f"[{'x' if on else ' '}] ", style=accent if on else muted)
                    text.append(f"{view.STATUS_ITEMS[item]} ({letter})   ",
                                style=fg if on else muted)
                if chosen:
                    selected_about = "g m r y switch status bar items on or off."
        text.append("\n\n" + selected_about, style=muted)
        keys = "↑↓ choose   ←→ change" if g.ellipsis == "…" else \
            "up/down choose   left/right change"
        text.append(f"\n{keys}   [ ] reorder   1-{len(view.PRESETS)} preset   Enter keep   Esc cancel",
                    style=muted)
        self.query_one("#view-body", Static).update(text)


# -- side panel ------------------------------------------------------------------------------------

@dataclass
class ChangeRecord:
    turn: int
    checkpoint: dict[Path, FileSnapshot]
    files: list[tuple[str, int, int]]  # (relative path, lines added, lines removed)


def _lines(data: bytes | None) -> list[str]:
    return [] if data is None else data.decode("utf-8", errors="replace").splitlines()


def change_counts(snapshot: FileSnapshot) -> tuple[int, int]:
    added = removed = 0
    for line in difflib.unified_diff(_lines(snapshot.original), _lines(snapshot.written),
                                     lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


class ChangesList(OptionList):
    """Files changed this session. Option ids are `t<record>` (a turn) or `f<record>:<file>`
    (a file in the workbench); `u` undoes back to the selected row's turn."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("u", "undo_selected", "Undo to here", show=False),
    ]

    class UndoRequested(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index  # position in VegaApp.changes

    def record_index(self, option_index: int | None) -> int | None:
        if option_index is None:
            return None
        option_id = self.get_option_at_index(option_index).id or ""
        match = re.match(r"[tf](\d+)", option_id)
        return int(match.group(1)) if match else None

    def action_undo_selected(self) -> None:
        index = self.record_index(self.highlighted)
        if index is not None:
            self.post_message(self.UndoRequested(index))


class StepsList(OptionList):
    """Workbench: every step of the session under `turn N` heads; the preview shows its output."""


class GitList(OptionList):
    """Workbench Git pane (tui_git): Enter previews a file's diff or a commit; r reads it again."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("r", "reread", "Refresh", show=False)]

    def action_reread(self) -> None:
        self.app.read_git()  # type: ignore[attr-defined]


class TestsList(OptionList):
    """Workbench Tests pane (tui_tests): Enter previews a failing test's output; r re-runs."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("r", "rerun", "Run again", show=False)]

    def action_rerun(self) -> None:
        self.app.rerun_tests()  # type: ignore[attr-defined]


class ProblemsList(OptionList):
    """Workbench Problems pane (tui_problems): Enter shows the file at that line; l runs ruff."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("l", "lint", "Run ruff", show=False)]

    def action_lint(self) -> None:
        self.app.run_lint()  # type: ignore[attr-defined]


# -- workbench panes -------------------------------------------------------------------------------

PANE_IDS = {"changes": "wb-changes", "steps": "wb-steps", "preview": "preview", "plan": "wb-plan",
            "machine": "wb-machine", "session": "wb-session", "loops": "wb-loops",
            "git": "wb-git", "tests": "wb-tests", "problems": "wb-problems",
            "context": "wb-context", "models": "wb-models", "product": "wb-product",
            "release": "wb-release"}
LIST_PANES = {"wb-changes", "wb-steps", "wb-loops", "wb-git", "wb-tests", "wb-problems",
              "wb-models"}
LOOPS_EVERY = 5.0  # seconds between readings of the loops' state and inbox


def make_pane(name: str) -> Widget:
    """A workbench pane by name (view.PANES). Lists are focusable OptionLists; the text panes are
    scrollable so a narrow column still shows everything."""
    pane: Widget
    if name == "changes":
        pane = ChangesList(id=PANE_IDS[name], classes="wb-pane")
    elif name == "steps":
        pane = StepsList(id=PANE_IDS[name], classes="wb-pane")
    elif name == "loops":
        pane = tui_loops.LoopsList(id=PANE_IDS[name], classes="wb-pane wb-loops")
    elif name == "models":
        pane = tui_models.ModelsList(id=PANE_IDS[name], classes="wb-pane wb-models")
    elif name == "product":
        pane = tui_product.ProductPane(id=PANE_IDS[name], classes="wb-pane wb-text wb-product")
    elif name == "release":
        pane = tui_product.ReleaseList(id=PANE_IDS[name], classes="wb-pane wb-release")
    elif name in {"git", "tests", "problems"}:
        kind = {"git": GitList, "tests": TestsList, "problems": ProblemsList}[name]
        pane = kind(id=PANE_IDS[name], classes=f"wb-pane wb-{name}")
    else:
        pane = VerticalScroll(Static("", id=f"{PANE_IDS[name]}-body"), id=PANE_IDS[name],
                              classes=f"wb-pane wb-text wb-{name}")
    pane.border_title = view.PANE_TITLES[name]
    return pane


# -- the app ---------------------------------------------------------------------------------------

def _total_memory() -> float | None:
    try:
        return resources.total_memory_gb()
    except Exception:  # noqa: BLE001 - optional figure
        return None


class VegaApp(App[None]):
    CSS_PATH: ClassVar[list[str]] = [  # type: ignore[misc]
        "tui.tcss", "tui_quiet.tcss", "tui_observatory.tcss", "tui_workbench.tcss",
    ]
    ENABLE_COMMAND_PALETTE = False
    TITLE = "vega"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "stop", "Stop", priority=True, show=False),
        Binding("ctrl+q", "quit", "Quit", priority=True, show=False),
        Binding("ctrl+u", "undo", "Undo", priority=True, show=False),
        Binding("ctrl+r", "toggle_review", "Review", priority=True, show=False),
        Binding("ctrl+p", "toggle_panel", "Panel", priority=True, show=False),
        Binding("ctrl+t", "toggle_theme", "Theme", priority=True, show=False),
        Binding("ctrl+l", "clear_view", "Clear", priority=True, show=False),
        Binding("tab", "cycle_pane(1)", "Next pane", priority=True, show=False),
        Binding("shift+tab", "cycle_pane(-1)", "Previous pane", priority=True, show=False),
        Binding("f1", "help", "Help", show=False),
        Binding("f2", "customize", "Customize", priority=True, show=False),
        Binding("f3", "product_view", "Product view", priority=True, show=False),
        Binding("escape", "dismiss_notices", "Dismiss", show=False),
    ]

    def __init__(
        self, project: Path, *, review: bool = False, undo_depth: int = 20,
        agent: CodingAgent | None = None, warm_up: bool = True,
        memory_probe: Callable[[], float] | None = None, session_log: SessionLog | None = None,
        layout: str | None = None, total_memory_gb: float | None = None,
        swap_probe: Callable[[], float | None] | None = None, transparent: bool | None = None,
        teach: bool | None = None,
    ) -> None:
        super().__init__()
        for themes in LAYOUT_THEMES.values():
            for theme in themes.values():
                self.register_theme(theme)
        self.layout_name = layout if layout in LAYOUTS else layout_from_env()
        self.view = view.load()
        wanted = os.environ.get("VEGA_THEME", "dark").strip().lower()
        self.mode = wanted if wanted in {"dark", "light"} else "dark"
        self.theme = LAYOUT_THEMES[self.layout_name][self.mode].name
        if transparent is None:
            transparent = os.environ.get("VEGA_OPAQUE", "").strip() in {"", "0"}
        self.transparent_ok = transparent
        self.project = project.resolve()
        self.agent = agent or CodingAgent(self.project, stream=True, review=review,
                                          undo_depth=undo_depth, web_research=True)
        self.agent.on_event = self._agent_event
        self.agent.approve = self._approve_write
        self.agent.approve_command = self._approve_command
        self.agent.approve_web = self._approve_web
        self.agent.web.approve_adapter = self._approve_adapter
        if agent is not None and review:
            self.agent.review = True
        # teach mode: `vega --teach` (or VEGA_TEACH=1) for this session, else the saved setting
        wanted_teach = teach if teach is not None else (teaching.from_env() or self.view.teach)
        if wanted_teach != self.agent.teach:
            self.agent.set_teach(wanted_teach)
        self.session_log = session_log or SessionLog(server.STATE_DIR / "sessions", self.project)
        self.warm_up = warm_up
        if memory_probe is None:
            def probe() -> tuple[float, float | None]:
                snap = resources.snapshot()
                return snap.model_rss_mb, snap.swap_used_mb
        else:
            def probe() -> tuple[float, float | None]:
                return memory_probe(), swap_probe() if swap_probe else None
        self.machine_probe = probe
        self.total_memory_gb = total_memory_gb if total_memory_gb is not None else _total_memory()
        self.working = False
        self.awaiting_approval = False
        self.turn: Turn | None = None
        self.turn_list: list[Turn] = []
        self.turns = 0
        self.model_calls = 0
        self.request_seconds: list[float] = []  # every model request this session (sparkline)
        self.changes: list[ChangeRecord] = []
        self.pending: list[Future[bool]] = []
        self.panel_open = True
        self.last_reply: tuple[float, float | None] | None = None
        self.memory_mb: float | None = None
        self.swap_mb: float | None = None
        self.failed_item: str | None = None  # plan item whose last check failed
        self.preview_key: str | None = None
        # loops pane: read in a worker every LOOPS_EVERY seconds (see _read_loops)
        self.loop_rows: list[Any] = []
        self.loop_errors: list[str] = []
        self.loop_header: Any = None
        self.loop_findings: dict[str, list[tuple[dict[str, Any], str]]] = {}
        self.loops_expanded: set[str] = set()
        self._frame = 0
        self._drag_from: view.View | None = None  # the view when a column drag began
        # developer panes (git, tests, problems, context); see tui_git, tui_tests, ...
        self.git_state: tui_git.GitState | None = None
        self.tests_run: tui_tests.TestRun | None = None
        self.tests_running: str | None = None  # the command `r` is re-running
        self._tests_stop: Any = None  # threading.Event for that run
        self._command_started: dict[str, float] = {}
        self.problems: dict[str, list[tui_problems.Problem]] = {}
        self.problem_rows: list[tui_problems.Problem] = []
        self.lint_note: str | None = None
        self.lint_running = False
        self.last_compaction: tui_context.Compaction | None = None
        self.async_previews: dict[str, tuple[str, str, str]] = {}  # key: (title, kind, body)
        self.model_inventory: Any = None  # model_store.Inventory for the Models pane
        self.models_busy: str | None = None  # a download or removal in progress
        # product view (tui_product, product.py): the last readiness report, read in a worker
        self.product_report: product.Report | None = None
        self.product_turns: list[tuple[str, str]] = []  # (request's first line, pass|fail|none)
        self.started_at = time.monotonic()
        self.tests_stamp: str | None = None  # tree fingerprint when tests_run ran
        self.task_stamp: str | None = None  # ... when agent.last_acceptance ran
        self._acceptance_seen: Any = None
        self.pending_fix: tuple[str, str] | None = None  # (check key, prepared text) in the box
        self._blocking: set[str] = set()  # blocking checks in the last report (usage log)
        self._code_view: view.View | None = None  # the code view F3 returns to
        self._product_switched = 0.0  # when F3 last switched views (the F3-back measure)
        self.usage_session = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    # layout

    def compose(self) -> ComposeResult:
        with Horizontal(id="header"):
            yield Static("vega", id="brand")
            yield Static(short_path(self.project), id="path", markup=False)
            yield Static("", id="mode")
            yield Static(self._model_label(), id="model", markup=False)
            yield Static("ready", id="state")
        yield Static("", id="strip")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield from (make_pane(name) for name in self.view.left)
            yield ColumnDivider("left", id="divider-left", classes="divider")
            yield VerticalScroll(id="conversation")
            yield ColumnDivider("right", id="divider-right", classes="divider")
            with Vertical(id="side"):
                with Vertical(id="plan-pane", classes="pane"):
                    yield Static("Plan", classes="panel-title")
                    yield Static("", id="plan", classes="panel-body")
                with Vertical(id="changes-pane", classes="pane"):
                    yield Static("Changed this session", classes="panel-title")
                    yield Static("No file changes yet", id="changes-empty",
                                 classes="panel-empty panel-body")
                    yield ChangesList(id="changes")
                    yield Static("", id="changes-hint", classes="panel-hint")
                with Vertical(id="machine-pane", classes="pane"):
                    yield Static("Machine", classes="panel-title")
                    yield Static("", id="machine", classes="panel-body")
                with Vertical(id="loops-side-pane", classes="pane"):
                    yield Static("Loops", classes="panel-title")
                    yield Static("", id="loops-side", classes="panel-body")
                with Vertical(id="session-pane", classes="pane"):
                    yield Static("Session", classes="panel-title")
                    yield Static("", id="session", classes="panel-body")
                yield Static("^P or Esc to close", id="overlay-hint")
            with Vertical(id="right"):
                yield from (make_pane(name) for name in self.view.right)
        with Horizontal(id="input-row"):
            yield Static(glyphs().prompt, id="prompt-mark")
            yield PromptInput(id="prompt")
        with Horizontal(id="status"):
            yield ContextGauge(id="gauge")
            yield Static("", id="readouts")
            yield Static("", id="hints")

    def on_mount(self) -> None:
        self.query_one("#changes", ChangesList).display = False
        self.query_one("#plan-pane").border_title = "Plan"
        self.query_one("#changes-pane").border_title = "Changes"
        self.query_one("#machine-pane").border_title = "Machine"
        self.query_one("#session-pane").border_title = "Session"
        self.set_layout(self.layout_name, announce=False)
        self.query_one("#gauge", ContextGauge).set_usage(context_usage(self.agent))
        if not is_project(self.project):
            self._add(Note(NOT_A_PROJECT + "  (Esc dismisses)", "warning", dismissible=True))
        self.query_one("#prompt", PromptInput).focus()
        self.set_interval(0.1, self._tick)
        self.set_interval(MEMORY_EVERY, self._read_memory)
        self._read_memory()
        self.set_interval(LOOPS_EVERY, self._read_loops)
        self._read_loops()
        self.read_git()
        self.read_models()
        self.read_product()
        self.product_usage("session_start", view=self.view.mode, preset=self.view.preset)
        self.run_worker(self._discover, thread=True, group="startup", exit_on_error=False)

    def set_layout(self, name: str, *, announce: bool = True) -> None:
        """Switch layout in place: CSS classes, the layout's palette (same light/dark mode),
        terminal-background transparency for quiet, and every pane redrawn."""
        self.layout_name = name
        for other in LAYOUTS:
            self.set_class(other == name, f"layout-{other}")
        transparent = name == "quiet" and self.transparent_ok
        self.set_class(transparent, "transparent")
        self.ansi_color = True if transparent else None
        self.theme = LAYOUT_THEMES[name][self.mode].name
        self.default_screen.remove_class("panel-overlay", "panel-hidden")
        boxed = name in {"observatory", "workbench"}
        self.query_one("#conversation").border_title = "Conversation" if boxed else None
        self.query_one("#input-row").border_title = "Message" if boxed else None
        self._apply_view_styles()
        self._apply_width(self.size.width)
        for gutter in self.query(Gutter):
            gutter.refresh()
        self._restyle()
        self.call_after_refresh(self._after_layout)
        if announce:
            self._add(Note(f"Layout {name}. /layout lists the others; VEGA_LAYOUT={name} starts "
                           "with it."))

    @property
    def step_detail(self) -> str:
        return self.view.steps if self.layout_name == "workbench" else "collapsed"

    def apply_view(self, new: view.View, *, save: bool = True) -> None:
        """Show `new` now (panes rebuilt only if their placement changed); `save` keeps it for
        the next start."""
        new = view.with_teach(new, self.view.teach)  # presets do not switch teach mode
        old, self.view = self.view, new
        if save:
            try:
                view.save(new)
            except OSError as exc:
                self.notify(f"Couldn't save the view: {exc}", severity="warning")
        if (old.left, old.right) != (new.left, new.right):
            self.run_worker(self._rebuild_columns(), group="view", exclusive=True,
                            exit_on_error=False)
        self._apply_view_styles()
        self._apply_width(self.size.width)
        for turn in self.query(Turn):
            turn.apply_detail(self.step_detail)
        self.call_after_refresh(self._after_layout)

    async def _rebuild_columns(self) -> None:
        for column, names in (("#left", self.view.left), ("#right", self.view.right)):
            box = self.query_one(column)
            await box.remove_children()
            await box.mount(*(make_pane(name) for name in names))
        self.read_git()
        self.read_models()
        self.read_product()
        self.call_after_refresh(self._after_layout)

    def _apply_view_styles(self) -> None:
        self.query_one("#left").styles.width = self.view.left_width
        self.query_one("#right").styles.width = self.view.right_width or "1fr"
        screen = self.default_screen
        screen.set_class(not self.view.left, "no-left")
        screen.set_class(not self.view.right, "no-right")
        product_mode = self.layout_name == "workbench" and self.view.mode == "product"
        screen.set_class(product_mode, "product-mode")
        self.query_one("#header").set_class(product_mode, "with-mode")
        self._update_dividers()

    def _update_dividers(self) -> None:
        """The draggable conversation edges show only where a side column is on screen (wide
        workbench, panel not hidden); there they are drawn over the conversation's own border."""
        screen = self.default_screen
        shown = (self.layout_name == "workbench" and not screen.has_class("narrow")
                 and not screen.has_class("panel-hidden"))
        for side, present in (("left", bool(self.view.left)), ("right", bool(self.view.right))):
            visible = shown and present
            self.query_one(f"#divider-{side}", ColumnDivider).display = visible

    def _model_label(self) -> str:
        model = self.agent.model
        if model == server.model_name() == "vega":
            try:
                return models.active_profile() or model
            except Exception:  # noqa: BLE001 - a label is never worth failing start-up over
                return model
        return model

    def _discover(self) -> None:
        """Background start-up: load the model and cache the fixed prompt (#2), learn the window."""
        if self.warm_up:
            self.agent.warm_up()
        window = self.agent.num_ctx
        usage = context_usage(self.agent, window)
        self.call_from_thread(self.query_one("#gauge", ContextGauge).set_usage, usage)

    def on_resize(self, event: events.Resize) -> None:
        self._apply_width(event.size.width)
        self.call_after_refresh(self._after_layout)

    def _after_layout(self) -> None:
        self._refresh_panels()
        self._refresh_status()

    @property
    def narrow(self) -> bool:
        return self.default_screen.has_class("narrow")

    def _wide_at(self) -> int:
        return self.view.wide_at() if self.layout_name == "workbench" else \
            WIDE_AT[self.layout_name]

    def _apply_width(self, width: int) -> None:
        narrow = width < self._wide_at()
        screen = self.default_screen
        screen.set_class(narrow, "narrow")
        screen.set_class(not narrow, "wide")
        if self.layout_name == "quiet" or narrow:
            self.panel_open = screen.has_class("panel-overlay")
        else:
            self.panel_open = not screen.has_class("panel-hidden")
        self.query_one("#gauge", ContextGauge).cells = GAUGE_CELLS
        self._update_dividers()
        self._refresh_status()

    def _add(self, widget: Widget) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        conversation.mount(widget)
        conversation.scroll_end(animate=False)

    # input

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        text = event.text.strip()
        if not text:
            return
        prompt = self.query_one("#prompt", PromptInput)
        if text.startswith("/"):
            prompt.remember(text)
            prompt.load_text("")
            self._command(text)
            return
        if self.working:
            self.notify("Vega is still working. Wait for the answer, or press Ctrl+C to stop.",
                        severity="warning")
            return
        prompt.remember(text)
        prompt.load_text("")
        if self.pending_fix is not None:  # the user sent (maybe edited) a prepared fix request
            self.product_usage("fix_sent", check=self.pending_fix[0],
                               edited=text != self.pending_fix[1].strip())
            self.pending_fix = None
        self.start_turn(text)

    def _command(self, text: str) -> None:
        command, _, argument = text.partition(" ")
        command, argument = command.lower(), argument.strip().lower()
        if command in {"/exit", "/quit"}:
            self._quit()
            return
        if command == "/help":
            self.action_help()
            return
        if command == "/layout":
            if not argument:
                others = ", ".join(name for name in LAYOUTS if name != self.layout_name)
                self._add(Note(f"Layout {self.layout_name}. Others: {others}. Switch with "
                               "/layout <name>; VEGA_LAYOUT=<name> sets the one Vega starts "
                               "with."))
            elif argument in LAYOUTS:
                self.set_layout(argument)
            else:
                self._add(Note(f"Unknown layout {argument}. Choose one of "
                               f"{', '.join(LAYOUTS)}.", "warning"))
            return
        if command == "/learn":
            target = text.partition(" ")[2].strip()  # keep the original case of paths and URLs
            if not target:
                self._add(Note("Usage: /learn <url | file | folder>. Vega reads it into its "
                               "knowledge library and can then search it (vega learn list shows "
                               "what it knows).", "warning"))
                return
            self._add(Note(f"Learning {target}…"))
            self.run_worker(lambda: self._learn(target), thread=True, group="learn",
                            exit_on_error=False)
            return
        if command in {"/view", "/pane", "/width", "/steps"}:
            self._view_command(command, argument)
            return
        if command == "/copy":
            self._copy_command(argument)
            return
        if command == "/open":
            self._open_command(text.partition(" ")[2].strip())  # keep the URL's case
            return
        if command == "/transcript":
            path = str(self.session_log.path)
            try:
                self.copy_to_clipboard(path)
                copied = " (copied)"
            except Exception:  # noqa: BLE001 - clipboard support depends on the terminal
                copied = ""
            self._add(Note(f"Transcript{copied}: {path}", "output"))
            return
        if command == "/teach":
            self._teach_command(argument)
            return
        if command == "/product":
            self.action_product_view()
            return
        if command == "/ship":
            self._add(Note(tui_product.SHIP_LATER + " The Release checklist (F3) shows what is "
                           "left.", "warning"))
            return
        if command == "/usage":
            self._usage_command(argument)
            return
        if self.working and command != "/review":
            self.notify(f"{command} is available when Vega has finished this turn.",
                        severity="warning")
            return
        if command == "/reset":
            self.agent.reset()
            self._refresh_panels()
            self.query_one("#gauge", ContextGauge).set_usage(context_usage(self.agent))
            self._add(Note("Conversation context cleared. Files and undo history are kept."))
        elif command == "/status":
            self._add(Note(execute("git_status", {}, self.project), "output"))
        elif command == "/diff":
            diff = execute("git_diff", {}, self.project)
            body = diff_text(diff, palette(self)) if diff.startswith("diff ") else diff
            self._add(Note(body, "output"))
        elif command == "/undo":
            if argument not in {"", "--force"}:
                self._add(Note("Usage: /undo [--force]", "warning"))
                return
            self.undo(1, force=argument == "--force", confirm=False)
        elif command == "/review":
            if argument in {"on", "off"}:
                self.agent.review = argument == "on"
            elif argument:
                self._add(Note("Usage: /review on|off", "warning"))
                return
            self._review_changed()
        else:
            self._add(Note(f"Unknown command {command}. Press F1 or type /help for the list.",
                           "warning"))

    def _open_command(self, target: str) -> None:
        """Open a page in the user's default browser, only on their command: for a site that
        refuses automated reading, the user reads it and tells Vega what it says."""
        url = target or self.agent.web.blocked_url
        if not url:
            self._add(Note("Usage: /open <url>. With no URL it opens the last page a site "
                           "refused to Vega (none this session).", "warning"))
            return
        if not re.match(r"(?i)^https?://", url):
            self._add(Note(f"/open only opens http(s) web pages, not {url}.", "warning"))
            return
        opened = False
        with contextlib.suppress(Exception):  # no browser available (e.g. a server session)
            opened = webbrowser.open(url)
        if opened:
            self._add(Note(f"Opened {url} in your browser. Paste or summarise what it says and "
                           "Vega will use it."))
        else:
            self._add(Note(f"Could not open a browser here. The address is {url}", "warning"))

    def _teach_command(self, argument: str) -> None:
        if argument not in {"", "on", "off"}:
            self._add(Note("Usage: /teach on|off", "warning"))
            return
        if argument and self.working:
            self.notify("/teach is available when Vega has finished this turn.",
                        severity="warning")
            return
        if argument:
            on = argument == "on"
            self.agent.set_teach(on)
            self.view = view.with_teach(self.view, on)
            try:
                view.save(self.view)
            except OSError as exc:
                self.notify(f"Couldn't save the setting: {exc}", severity="warning")
            self._refresh_status()
        if self.agent.teach:
            self._add(Note("Teach mode on. Vega states its plan, says why before each change or "
                           "check, explains failing checks, and ends with what to take away. "
                           "These notes are shown here and kept out of the model's later "
                           "context. /teach off stops it."))
        else:
            self._add(Note("Teach mode off. /teach on makes Vega explain its decisions as it "
                           "works."))

    def _usage_command(self, argument: str) -> None:
        if argument not in {"", "on", "off"}:
            self._add(Note("Usage: /usage on|off", "warning"))
            return
        if argument:
            try:
                usage.set_enabled(argument == "on")
            except OSError as exc:
                self._add(Note(f"Couldn't save the setting: {exc}", "warning"))
                return
            if argument == "on":
                self.product_usage("session_start", view=self.view.mode,
                                   preset=self.view.preset)
        state = "on" if usage.enabled() else "off"
        self._add(Note(f"Usage log {state}. When on, Vega appends events only (a view opened, a "
                       "key used, a fix request sent; never code or text) to "
                       f"{usage.path()}. Nothing is sent anywhere. `vega usage` summarises it; "
                       "/usage on|off switches it."))

    def _learn(self, target: str) -> None:
        from . import knowledge

        try:
            added = knowledge.learn(target)
        except knowledge.LearnError as exc:
            self.call_from_thread(self._add, Note(str(exc), "warning"))
            return
        except Exception as exc:  # noqa: BLE001 - report any reader failure in the chat
            self.call_from_thread(self._add, Note(f"Could not learn {target}: {exc}", "warning"))
            return
        self.agent.refresh_knowledge()
        lines = "; ".join(f"{s.title} ({s.kind}, {s.chunks} sections)" for s in added[:5])
        more = f" and {len(added) - 5} more" if len(added) > 5 else ""
        self.call_from_thread(self._add, Note(
            f"Learned {lines}{more}. Vega can now look this up while it works."))

    def _view_command(self, command: str, argument: str) -> None:
        current, words = self.view, argument.split()
        new: view.View | None = None
        usage = {
            "/view": "/view [product|workbench|focus|review|monitor|steps|edit]",
            "/pane": f"/pane <{'|'.join(view.PANES)}> left|right|hide|up|down",
            "/width": "/width left <24-64> | /width right <30-100|auto>",
            "/steps": "/steps collapsed|expanded|hidden",
        }[command]
        if command == "/view":
            if not words:
                presets = "; ".join(f"{name}: {about}" for name, (_, about)
                                    in view.PRESETS.items())
                self._add(Note(f"View ({current.preset}): {current.describe()}.\nPresets: "
                               f"{presets}.\nF2 customizes; /pane, /width and /steps change one "
                               "thing.", "output"))
                return
            if words[0] in {"edit", "customize"}:
                self.action_customize()
                return
            if words[0] in view.PRESETS:
                new = view.preset(words[0])
            elif words[0] == "reset":
                new = view.View()
        elif command == "/pane" and len(words) == 2 and words[0] in view.PANES:
            name, where = words
            if where in {"left", "right", "hide"}:
                new = view.place(current, name, where)
            elif where in {"up", "down"}:
                new = view.shift(current, name, -1 if where == "up" else 1)
        elif command == "/width" and len(words) == 2 and words[0] in view.COLUMNS:
            value = words[1]
            if value == "auto" and words[0] == "right":
                new = view.resize(current, "right", 0)
            elif value.isdigit():
                new = view.resize(current, words[0], int(value))
        elif command == "/steps" and words and words[0] in view.STEP_DETAIL:
            new = view.with_steps(current, words[0])
        if new is None:
            self._add(Note(f"Usage: {usage}", "warning"))
            return
        self.apply_view(new)
        where = "" if self.layout_name == "workbench" else \
            " It applies to the workbench layout (/layout workbench)."
        self._add(Note(f"View saved: {new.describe()}.{where}"))

    # turns

    def start_turn(self, message: str) -> None:
        self._add(UserMessage(message))
        self.turn = Turn(self.turns + 1)
        self.turn.request = message.strip().splitlines()[0] if message.strip() else ""
        self.turn_list.append(self.turn)
        self._add(self.turn)
        self.working = True
        self._set_state()
        self.run_worker(lambda: self._run_turn(message), thread=True, group="turn",
                        exclusive=True, exit_on_error=False)

    def _run_turn(self, message: str) -> None:
        """Worker thread: one agent turn, then the transcript."""
        self.agent.request_log.clear()
        started = time.monotonic()
        try:
            answer = self.agent.run(message)
        except Exception as exc:  # noqa: BLE001 - never leave the UI stuck in "working"
            answer = f"Vega hit an error: {type(exc).__name__}: {exc}"
            self.call_from_thread(self._handle_event, "final", {"text": answer}, None)
        seconds = time.monotonic() - started
        requests = list(self.agent.request_log)
        self.session_log.record_turn(message, answer, seconds, requests)
        self.session_log.save(self.agent.messages)
        usage = context_usage(self.agent)
        self.call_from_thread(self._turn_done, seconds, requests, usage)

    def _agent_event(self, kind: str, data: dict[str, Any]) -> None:
        """Worker thread: measure the context here (agent state is only touched by this thread
        during a turn), then hand the event to the UI thread and wait until it is drawn."""
        if kind in {"text", "text_end"}:
            return  # raw model text is held back (as in line mode); skip a round trip per token
        usage = context_usage(self.agent) if kind in {"request", "tool_call", "notice",
                                                     "final"} else None
        self.call_from_thread(self._handle_event, kind, data, usage)

    def _handle_event(self, kind: str, data: dict[str, Any], usage: Usage | None) -> None:
        turn = self.turn
        if usage is not None:
            settle = kind == "notice" and str(data.get("message", "")).startswith(
                "Context compacted")
            self.query_one("#gauge", ContextGauge).set_usage(usage, settle=settle)
        if turn is None or turn.done:
            return
        if kind == "discarded":
            reason = data.get("reason", "")
            label = f"(retrying: {reason})" if data.get("retry") else f"({reason}; not the answer)"
            turn.add(Entry("note", outcome=label))
        elif kind == "tool_call" and data.get("name") != "update_plan":
            name, args = data["name"], data.get("args", {})
            if name in {"run_check", "run_command"}:
                self._command_started[str(args.get("command", ""))] = time.monotonic()
            turn.add(Entry("step", name, step_summary(name, args), running=True))
            self._refresh_steps_list()
        elif kind == "tool_result":
            name, result = data["name"], data.get("result", "")
            if name != "update_plan" or not result.startswith(("Plan updated", "Plan saved", "Plan cleared")):
                turn.steps.display = True
                entry = turn.steps.finish(name, data.get("args", {}), result)
                if name == "run_check":
                    self._check_finished(entry)
                self._refresh_steps_list()
                self._dev_result(name, data.get("args", {}), result, turn.number)
        elif kind == "plan":
            turn.add(Entry("plan", items=list(data.get("items", []))))
            self._refresh_panels()
        elif kind == "teach":
            turn.add_lesson(str(data.get("role", "why")), str(data.get("text", "")))
        elif kind == "notice":
            turn.add(Entry("notice", outcome=str(data.get("message", ""))))
            compaction = tui_context.parse_compaction(str(data.get("message", "")))
            if compaction is not None:
                self.last_compaction = compaction
                self._refresh_text_panes()
        elif kind == "final":
            turn.finish(str(data.get("text", "")), time.monotonic() - turn.started)
            self._scroll_to_turn(turn)

    def _check_finished(self, entry: Entry) -> None:
        """A run_check result: if it failed, the plan item in progress (else the first one not
        done, else the last) is shown as failed until a later check passes."""
        if entry.level != "error":
            self.failed_item = None
            return
        plan = self.agent.plan or []
        chosen = next((item for item in plan if item.get("status") == "in_progress"), None)
        chosen = chosen or next((item for item in plan if item.get("status") != "done"), None)
        chosen = chosen or (plan[-1] if plan else None)
        self.failed_item = chosen.get("text") if chosen else None

    def _turn_done(self, seconds: float, requests: list[dict[str, Any]], usage: Usage) -> None:
        self.working = False
        self.turns += 1
        self.model_calls += len(requests)
        self.request_seconds.extend(float(r.get("seconds") or 0.0) for r in requests)
        first = requests[0].get("seconds") if requests else None
        self.last_reply = (seconds, first)
        turn = self.turn
        if turn is not None and not turn.done:
            turn.finish(STOPPED, seconds)
        if turn is not None:
            turn.toggle.set_summary(turn.steps.step_count, seconds)
        self.query_one("#gauge", ContextGauge).set_usage(usage)
        before = {id(record.checkpoint) for record in self.changes}
        self._sync_changes()
        if turn is not None:
            new = [r for r in self.changes if id(r.checkpoint) not in before]
            self._show_turn_changes(turn, new)
        self._refresh_panels()
        self._refresh_status()
        self._set_state()
        self.read_git()
        latest = self._latest_file_key()
        if latest and (self.preview_key is None or self.preview_key.startswith("f")):
            self._select_preview(latest)
        if turn is not None:
            self._record_product_turn(turn, wrote=bool(new))
            self._scroll_to_turn(turn)
        self.read_product()

    def _show_turn_changes(self, turn: Turn, records: list[ChangeRecord]) -> None:
        """The quiet layout's compact row under a turn that changed files."""
        turn.changed_files = [entry for record in records for entry in record.files]
        self._render_turn_changes(turn)

    def _render_turn_changes(self, turn: Turn) -> None:
        files = turn.changed_files
        if not files:
            return
        colours, g = palette(self), glyphs()
        row = Text("changed  ", style=colours["muted"])
        for index, (path, added, removed) in enumerate(files):
            if index:
                row.append(",  ", style=colours["muted"])
            row.append(clip_middle(path, 28), style=colours["fg"])
            row.append(f" +{added} {g.minus}{removed}", style=colours["muted"])
        turn.changes.update(row)
        turn.changes.remove_class("-empty")

    # scrolling

    def _scroll_to_turn(self, turn: Turn) -> None:
        """Scroll to the end once the finished turn is laid out; if that hides the start of the
        reply, scroll back to it, keeping the first ⚠ step in view when the steps are open."""
        conversation = self.query_one("#conversation", VerticalScroll)

        def settle() -> None:
            conversation.scroll_end(animate=False, immediate=True)
            self.call_after_refresh(keep_reply)

        def keep_reply() -> None:
            if not turn.is_mounted:
                return
            top = self._offset_in(conversation, turn.answer)
            if top is None or top >= conversation.scroll_y:
                return
            view = conversation.scrollable_content_region.height
            target = top
            row = turn.steps.first_warning_row() if turn.steps.display else None
            steps_top = self._offset_in(conversation, turn.steps)
            if row is not None and steps_top is not None and steps_top + row >= target + view:
                target = steps_top + row - view + 1
            conversation.scroll_to(y=target, animate=False, immediate=True)

        self.call_after_refresh(settle)

    @staticmethod
    def _offset_in(container: Widget, widget: Widget) -> int | None:
        """The widget's top row in the container's scrollable content."""
        y = 0
        node: Any = widget
        while node is not None and node is not container:
            y += node.virtual_region.y
            node = node.parent
        return y if node is container else None

    # approvals (called on the worker thread; block until the modal answers)

    def _approve_write(self, plan: WritePlan) -> bool:
        return self._ask(lambda: review_screen(plan))

    def _approve_command(self, command: str) -> bool:
        return self._ask(lambda: command_screen(command, self.project))

    def _approve_web(self, kind: str, detail: str) -> str:
        """web_research: the once-per-session question ("session", "project" or "no")."""
        backend = str(getattr(self.agent.web.backend, "base_url", ""))
        future: Future[Any] = Future()
        self.call_from_thread(self._open_modal, lambda: web_screen(kind, detail, backend),
                              future, lambda value: value)
        try:
            answer = future.result()
        except Exception:  # noqa: BLE001 - a question that cannot be asked is a "no"
            return "no"
        return answer if answer in web.ANSWERS else "no"

    def _approve_adapter(self, candidate: Any) -> bool:
        """web_adapter_builder: approve a new site adapter before its first use."""
        return self._ask(lambda: adapter_screen(candidate))

    def _ask(self, make: Callable[[], ChoiceScreen]) -> bool:
        future: Future[bool] = Future()
        self.call_from_thread(self._open_modal, make, future)
        try:
            return bool(future.result())
        except Exception:  # noqa: BLE001 - an approval that cannot be asked is a "no"
            return False

    def _open_modal(self, make: Callable[[], ChoiceScreen], future: Future[Any],
                    convert: Callable[[Any], Any] = bool) -> None:
        if self.agent.stop_requested:
            future.set_result(False)
            return
        self.pending.append(future)
        if self.turn is not None:
            running = [e for e in self.turn.steps.entries if e.kind == "step" and e.running]
            if running:
                running[-1].level = "waiting"
                self.turn.steps.rebuild()
        self.awaiting_approval = True
        self._set_state()

        def answered(value: bool | None) -> None:
            if future in self.pending:
                self.pending.remove(future)
            self.awaiting_approval = bool(self.pending)
            self._set_state()
            if not future.done():
                future.set_result(convert(value))

        self.push_screen(make(), answered)

    def _decline_pending(self) -> None:
        for future in list(self.pending):
            if not future.done():
                future.set_result(False)
        self.pending.clear()
        self.awaiting_approval = False
        while isinstance(self.screen, ChoiceScreen):
            self.pop_screen()

    # undo

    def _sync_changes(self) -> None:
        """Mirror agent.checkpoints (oldest first) as ChangeRecords, labelled with their turn."""
        known = {id(record.checkpoint): record for record in self.changes}
        records = []
        for checkpoint in self.agent.checkpoints:
            record = known.get(id(checkpoint))
            if record is None:
                files = []
                for path, snapshot in checkpoint.items():
                    added, removed = change_counts(snapshot)
                    files.append((path.relative_to(self.agent.project).as_posix(), added,
                                  removed))
                record = ChangeRecord(self.turns, checkpoint, sorted(files))
            records.append(record)
        self.changes = records

    def undo(self, count: int, *, force: bool = False, confirm: bool = True) -> None:
        """Undo the last `count` file-changing turns, newest first (agent.undo semantics). On a
        conflict: confirm=True asks in a modal before forcing; False explains /undo --force."""
        for done in range(count):
            report = self.agent.undo(force=force)
            if report is None:
                self._add(Note("Nothing to undo: no earlier turn in this session changed files."))
                break
            if not report.applied:
                files = ", ".join(report.conflicts)
                if confirm:
                    left = count - done
                    self.push_screen(ChoiceScreen(
                        "Undo anyway?",
                        f"These files changed after Vega last wrote them: {files}. Undoing "
                        "restores Vega's earlier copy and discards those later edits.",
                        "Undo anyway", "Keep them", hint="y undo   n keep",
                    ), lambda ok, left=left: self.undo(left, force=True, confirm=False)
                        if ok else None)
                else:
                    self._add(Note(
                        f"Not undone. These files changed after Vega last wrote them, so "
                        f"restoring them would discard those later edits: {files}. Use "
                        "/undo --force to restore them anyway.", "warning"))
                break
            self._report_undo(report.restored, report.removed, report.conflicts)
        self._sync_changes()
        if self.preview_key and self.preview_key.startswith("f"):
            self.preview_key = None
        self._refresh_panels()
        self.query_one("#gauge", ContextGauge).set_usage(context_usage(self.agent))
        self.read_git()
        self._recheck_problems(list(self.problems))

    def _report_undo(self, restored: list[str], removed: list[str], overwritten: list[str]) -> None:
        g, colours = glyphs(), palette(self)
        body = Text()
        for path in restored:
            body.append(f"{g.done} restored  ", style=colours["success"])
            body.append(path + "\n", style=colours["fg"])
        for path in removed:
            body.append(f"{g.done} removed   ", style=colours["success"])
            body.append(path, style=colours["fg"])
            body.append("  (created by Vega)\n", style=colours["muted"])
        if overwritten:
            body.append(f"{g.warn} overwrote later changes in {', '.join(overwritten)}\n",
                        style=colours["warning"])
        left = len(self.agent.checkpoints)
        body.append(f"{left} earlier turn{'s' if left != 1 else ''} can still be undone.",
                    style=colours["muted"])
        self._add(Note(body))
        self.agent.messages.append(undo_note(restored + removed))

    def on_changes_list_undo_requested(self, event: ChangesList.UndoRequested) -> None:
        if self.working:
            self.notify("Undo is available when Vega has finished this turn.", severity="warning")
            return
        if not 0 <= event.index < len(self.changes):
            return
        count = len(self.changes) - event.index
        turn = self.changes[event.index].turn
        if count == 1:
            self.undo(1)
            return
        later = [str(record.turn) for record in self.changes[event.index:]]
        self.push_screen(ChoiceScreen(
            f"Undo back to turn {turn}?",
            f"Undo goes back one turn at a time, so turns {', '.join(later)} are all undone.",
            "Undo turns", "Cancel", hint="y undo   n cancel",
        ), lambda ok: self.undo(count) if ok else None)

    # panels and status

    def _panel_width(self) -> int:
        side = self.query_one("#side")
        width = side.content_region.width if side.display else 0
        for pane in ("#plan", "#session"):
            node = self.query_one(pane)
            if node.content_region.width:
                width = node.content_region.width
                break
        # one cell is kept free for the overlay's scrollbar
        return max(20, (width or SIDE_WIDTH - 3) - 1)

    def _plan_text(self, width: int) -> Text:
        g, colours = glyphs(), palette(self)
        plan = self.agent.plan
        texts = {item.get("text") for item in plan or []}
        if self.failed_item not in texts:
            self.failed_item = None
        if not plan:
            return Text("No plan yet", style=colours["muted"])
        text = Text()
        for index, item in enumerate(plan):
            if index:
                text.append("\n")
            failed = item.get("text") == self.failed_item
            text.append_text(plan_item(item, width, g, colours, failed=failed))
        if self.failed_item:
            text.append(f"\n{g.warn} the last check failed", style=colours["error"])
        return text

    def _refresh_panels(self) -> None:
        g, colours = glyphs(), palette(self)
        width = self._panel_width()
        self.query_one("#plan", Static).update(self._plan_text(width))
        self._refresh_changes(width, g, colours)
        self._refresh_session(width, colours)
        self._refresh_machine()
        self._refresh_workbench()

    def _refresh_changes(self, width: int, g: SimpleNamespace, colours: dict[str, str]) -> None:
        changes = self.query_one("#changes", ChangesList)
        highlighted = changes.highlighted
        changes.clear_options()
        counts_w = max((len(f"+{a} {g.minus}{r}") for record in self.changes
                        for _, a, r in record.files), default=0)
        for index, record in enumerate(self.changes):
            changes.add_option(Option(change_prompt(record, width - 1, counts_w, g, colours),
                                      id=f"t{index}"))
        changes.display = bool(self.changes)
        self.query_one("#changes-empty").display = not self.changes
        hint = self.query_one("#changes-hint", Static)
        hint.display = bool(self.changes)
        hint.update(Text("u undo back to the selected turn" if changes.has_focus
                         else "Tab to select a turn to undo", style=colours["muted"]))
        if self.changes:
            changes.highlighted = min(highlighted if highlighted is not None else 0,
                                      len(self.changes) - 1)

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        if isinstance(event.widget, ChangesList):
            self._refresh_changes_hint()
        if isinstance(event.widget, GitList):
            self.read_git()
        if isinstance(event.widget, tui_models.ModelsList) and self.models_busy is None:
            self.read_models()
        self._refresh_dividers()

    def on_descendant_blur(self, event: events.DescendantBlur) -> None:
        if isinstance(event.widget, ChangesList):
            self._refresh_changes_hint()
        self._refresh_dividers()

    def _refresh_dividers(self) -> None:
        """Redraw the conversation edges (they follow its focus colour and the theme)."""
        for divider in self.query(ColumnDivider):
            divider.refresh()

    def _refresh_changes_hint(self) -> None:
        colours = palette(self)
        changes = self.query_one("#changes", ChangesList)
        self.query_one("#changes-hint", Static).update(Text(
            "u undo back to the selected turn" if changes.has_focus
            else "Tab to select a turn to undo", style=colours["muted"]))

    def _refresh_session(self, width: int, colours: dict[str, str]) -> None:
        self.query_one("#session", Static).update(self._session_text(width, colours))

    def _session_text(self, width: int, colours: dict[str, str]) -> Text:
        turns = f"{self.turns} turn{'s' if self.turns != 1 else ''}"
        calls = f"{self.model_calls} model call{'s' if self.model_calls != 1 else ''}"
        session = Text(f"{turns}, {calls}", style=colours["fg"])
        gauge = self.query_one("#gauge", ContextGauge)
        rows = [
            ("model", _clip(self._model_label(), width - 12)),
            ("review", "on" if self.agent.review else "off"),
            ("teach", "on" if self.agent.teach else "off"),
        ]
        if self.last_reply is not None:
            seconds, first = self.last_reply
            rows.append(("last reply", seconds_label(seconds)))
            if first is not None:
                rows.append(("first call", seconds_label(first)))
        fixed = tokens_label(gauge.fixed)
        rows.append(("fixed", f"{fixed} re-measured" if gauge.remeasured
                     else f"{fixed} tokens"))
        rows.append(("transcript", transcript_label(self.session_log.path)))
        for label, value in rows:
            session.append("\n" + label.ljust(11), style=colours["muted"])
            session.append(_clip(value, width - 11), style=colours["fg"])
        session.append("\n/transcript shows the path", style=colours["muted"])
        return session

    def gauge_changed(self) -> None:
        self._refresh_machine()

    def _refresh_machine(self) -> None:
        """The instrument pane (observatory's Machine, or the workbench's when shown)."""
        if self.layout_name == "observatory":
            width = max(20, (self.query_one("#machine").content_region.width or 40) - 1)
            self.query_one("#machine", Static).update(self._machine_text(width))
        elif self._pane("machine") is not None or self._pane("context") is not None:
            self._refresh_text_panes()

    def _machine_text(self, width: int) -> Text:
        """Large gauge, memory of total, swap, latency and a sparkline of model request times."""
        width = max(20, width)
        g, colours = glyphs(), palette(self)
        gauge = self.query_one("#gauge", ContextGauge)
        label, value = colours["muted"], colours["readout"]
        text = Text()
        percent = f"{round(100 * gauge.used / max(1, gauge.window))}% of " \
                  f"{window_label(gauge.window)}"
        text.append("context".ljust(width - len(percent)), style=label)
        text.append(percent, style=f"bold {value}")
        cells = width - 1
        bar = gauge_bar(gauge.fixed, gauge.used, gauge.window, cells, colours, g)
        text.append("\n")
        text.append_text(bar)
        text.append("\n")
        text.append_text(bar)
        conversation = max(0.0, gauge.used - gauge.fixed)
        text.append("\n")
        text.append(g.fixed, style=colours["fixed"])
        text.append(" fixed ", style=label)
        text.append(tokens_label(gauge.fixed), style=value)
        text.append("   ")
        text.append(g.used, style=colours["accent"])
        text.append(" conversation ", style=label)
        text.append(tokens_label(conversation), style=value)
        text.append(f"\n{g.tick}", style=colours["tick"])
        text.append(f" compacts at {round(context.COMPACT_AT * 100)}%", style=label)

        def row(name: str, reading: str, note: str = "") -> None:
            text.append("\n" + name.ljust(12), style=label)
            text.append(reading, style=value)
            if note:
                text.append("  " + note, style=label)

        text.append("\n")
        if self.memory_mb:
            row("memory", memory_label(self.memory_mb, self.total_memory_gb), "model server")
        else:
            row("memory", "model not loaded" if self.memory_mb is not None else "reading")
        row("swap", f"{self.swap_mb / 1024:.1f} GB" if self.swap_mb is not None
            else "not reported")
        if self.last_reply is not None:
            seconds, first = self.last_reply
            row("last reply", seconds_label(seconds),
                f"first call {seconds_label(first)}" if first is not None else "")
        else:
            row("last reply", "none yet")
        recent = self.request_seconds[-max(4, width - 24):]
        if recent:
            spread = f"{min(recent):.1f}–{max(recent):.1f} s" if g.ellipsis == "…" else \
                f"{min(recent):.1f}-{max(recent):.1f} s"
            text.append("\n" + "requests".ljust(12), style=label)
            text.append(sparkline(recent, g), style=colours["accent"])
            text.append(f"  {spread}", style=value)
        else:
            row("requests", "none yet")
        turns = f"{self.turns} turn{'s' if self.turns != 1 else ''}"
        row("session", f"{turns}, {self.model_calls} calls")
        return text

    # workbench

    def _pane(self, name: str) -> Any:
        """The workbench pane `name` if the current view shows it, else None."""
        if self.layout_name != "workbench":
            return None
        return next(iter(self.query(f"#{PANE_IDS[name]}")), None)

    def _refresh_workbench(self) -> None:
        if self.layout_name != "workbench":
            return
        self._refresh_files_pane()
        self._refresh_steps_list()
        self._refresh_loops_pane()
        self._refresh_git_pane()
        self._refresh_models_pane()
        self._refresh_tests_pane()
        self._refresh_problems_pane()
        self._refresh_text_panes()
        self._render_preview()
        self._refresh_product_panes()

    def _refresh_text_panes(self) -> None:
        """Plan, Machine, Session and Context when the view puts them in a workbench column."""
        colours = palette(self)
        for name in ("plan", "machine", "session", "context"):
            pane = self._pane(name)
            if pane is None:
                continue
            width = max(16, (pane.scrollable_content_region.width or 30) - 1)
            text = (self._plan_text(width) if name == "plan" else
                    self._machine_text(width) if name == "machine" else
                    self._context_text(width) if name == "context" else
                    self._session_text(width, colours))
            body = next(iter(pane.query(Static)), None)
            if body is not None:  # None while F3 or /view is still rebuilding the column
                body.update(text)

    def _refresh_files_pane(self) -> None:
        files = self._pane("changes")
        if files is None:
            return
        g, colours = glyphs(), palette(self)
        width = max(16, files.content_region.width or 26)
        keep = files.highlighted_option.id if files.highlighted_option else None
        files.clear_options()
        if not self.changes:
            files.add_option(Option(Text("No file changes yet", style=colours["muted"]),
                                    disabled=True))
        counts_w = max((len(f"+{a} {g.minus}{r}") for record in self.changes
                        for _, a, r in record.files), default=0)
        for index, record in enumerate(self.changes):
            files.add_option(Option(Text(f"turn {record.turn}", style=colours["muted"]),
                                    disabled=True))
            for number, (path, added, removed) in enumerate(record.files):
                files.add_option(Option(file_row(path, added, removed, width, counts_w, g,
                                                 colours), id=f"f{index}:{number}"))
        if keep:
            self._highlight(files, keep)

    def _refresh_steps_list(self) -> None:
        steps = self._pane("steps")
        if steps is None:
            return
        g, colours = glyphs(), palette(self)
        width = max(16, steps.content_region.width or 26)
        keep = steps.highlighted_option.id if steps.highlighted_option else None
        steps.clear_options()
        if not self.turn_list:
            steps.add_option(Option(Text("No steps yet", style=colours["muted"]), disabled=True))
        for t_index, turn in enumerate(self.turn_list):
            steps.add_option(Option(Text(f"turn {turn.number}", style=colours["muted"]),
                                    disabled=True))
            for e_index, entry in enumerate(turn.steps.entries):
                if entry.kind != "step":
                    continue
                row = Text()
                if entry.level in {"warning", "error"}:
                    row.append(g.warn + " ", style=colours["error"] if entry.level == "error"
                               else colours["warning"])
                elif entry.running:
                    row.append(g.doing + " ", style=colours["accent"])
                else:
                    row.append("  ")
                row.append(entry.name.ljust(11), style=colours["fg"])
                row.append(_clip(entry.target, max(4, width - 13), keep_end=True),
                           style=colours["muted"])
                steps.add_option(Option(row, id=f"s{t_index}:{e_index}"))
        if keep:
            self._highlight(steps, keep)

    @staticmethod
    def _highlight(options: OptionList, option_id: str) -> None:
        with contextlib.suppress(Exception):  # the option is gone (undo): no highlight
            options.highlighted = options.get_option_index(option_id)

    def _latest_file_key(self) -> str | None:
        if not self.changes:
            return None
        index = len(self.changes) - 1
        return f"f{index}:0" if self.changes[index].files else None

    def _select_preview(self, key: str) -> None:
        self.preview_key = key
        target = self._pane("changes" if key.startswith("f") else "steps")
        if target is not None:
            self._highlight(target, key)
        self._render_preview()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if (event.option_list.id in LIST_PANES and event.option.id
                and (event.option_list.has_focus or self.preview_key is None)):
            self._show_preview(event.option.id)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Enter on a Git, Tests or Problems row shows it in the preview (the other lists
        preview on highlight and give Enter their own meaning)."""
        if event.option_list.id in {"wb-git", "wb-tests", "wb-problems"} and event.option.id:
            self._show_preview(event.option.id, force=True)

    def _show_preview(self, key: str, *, force: bool = False) -> None:
        """Preview `key`; a Git row first reads its diff in a worker."""
        if key[:1] in {"G", "C"} and (force or key != self.preview_key
                                      or key not in self.async_previews):
            self._read_git_preview(key)
        self.preview_key = key
        self._render_preview()

    def _render_preview(self) -> None:
        preview = self._pane("preview")
        if preview is None:
            return
        body = next(iter(preview.query("#preview-body").results(Static)), None)
        if body is None:  # F3 or /view is still rebuilding the column: render once it is in
            self.call_after_refresh(self._render_preview)
            return
        g, colours = glyphs(), palette(self)
        width = max(20, preview.scrollable_content_region.width or 40)
        key = self.preview_key or ""
        file_match = re.match(r"f(\d+):(\d+)$", key)
        step_match = re.match(r"s(\d+):(\d+)$", key)
        if file_match and int(file_match.group(1)) < len(self.changes):
            record = self.changes[int(file_match.group(1))]
            number = int(file_match.group(2))
            if number < len(record.files):
                path, added, removed = record.files[number]
                snapshot = record.checkpoint.get(self.agent.project / path)
                if snapshot is None:
                    snapshot = next((snap for p, snap in record.checkpoint.items()
                                     if p.as_posix().endswith(path)), None)
                preview.border_title = f"{clip_middle(path, max(10, width - 24))}  " \
                                       f"turn {record.turn}  +{added} {g.minus}{removed} lines"
                if snapshot is not None:
                    body.update(diff_rows(_lines(snapshot.original), _lines(snapshot.written),
                                          width, colours, g))
                    return
        if step_match and int(step_match.group(1)) < len(self.turn_list):
            turn = self.turn_list[int(step_match.group(1))]
            index = int(step_match.group(2))
            if index < len(turn.steps.entries):
                entry = turn.steps.entries[index]
                preview.border_title = f"{entry.name}  {_clip(entry.target, max(10, width - 20))}"
                text = Text()
                outcome = Steps._outcome(entry, g)
                if outcome:
                    style = colours["error"] if entry.level == "error" else \
                        colours["warning"] if entry.level == "warning" else colours["muted"]
                    text.append(outcome + "\n\n", style=style)
                lines = (entry.result or "(no output yet)").splitlines()[:PREVIEW_LINES]
                for number, line in enumerate(lines):
                    if number:
                        text.append("\n")
                    error = line.startswith(("E ", "E\t")) or "Error" in line
                    text.append(line.expandtabs(4), style=colours["error"] if error
                                else colours["fg"])
                body.update(text)
                return
        dev_preview = self._dev_preview(key, width, colours, g)
        if dev_preview is not None:
            preview.border_title, text = dev_preview
            body.update(text)
            return
        if key[:1] in {"L", "F", "D"}:
            loop_preview = self._loop_preview(key, width, colours, g)
            if loop_preview is not None:
                preview.border_title, text = loop_preview
                body.update(text)
                return
        preview.border_title = "Preview"
        body.update(Text("Select a changed file or a step on the left to see its diff or "
                         "output here. Tab moves between panes.", style=colours["muted"]))

    def _loop_preview(self, key: str, width: int, colours: dict[str, str],
                      g: SimpleNamespace) -> tuple[str, Text] | None:
        kind, name = key[:1], key[1:]
        if kind == "D":
            return tui_loops.debug_preview(name, colours)
        if kind == "L":
            row = next((r for r in self.loop_rows if r.name == name), None)
            if row is None:
                return None
            return tui_loops.loop_preview(row, self.loop_findings.get(name, []), width, colours, g)
        for items in self.loop_findings.values():
            for finding, state in items:
                if finding["id"] == name:
                    return tui_loops.finding_preview(finding, state, width, colours, g)
        return None

    def action_cycle_pane(self, step: int) -> None:
        """Tab: in the workbench, move between the panes (each shows a focus ring); elsewhere
        and in dialogs, the usual next/previous control."""
        if self.screen is not self.default_screen or self.layout_name != "workbench" \
                or self.narrow or self.default_screen.has_class("panel-hidden"):
            if step > 0:
                self.screen.focus_next()
            else:
                self.screen.focus_previous()
            return
        order = ["#prompt", *(f"#{PANE_IDS[p]}" for p in self.view.left), "#conversation",
                 *(f"#{PANE_IDS[p]}" for p in self.view.right)]
        focused = self.focused
        current = 0
        for index, selector in enumerate(order):
            node = self.query_one(selector)
            if focused is not None and (focused is node or node in focused.ancestors):
                current = index
                break
        self.query_one(order[(current + step) % len(order)]).focus()

    # product view (tui_product.py; the readiness report comes from product.py in a worker)

    def _product_shown(self) -> bool:
        return (self.layout_name == "workbench"
                and (self.view.mode == "product" or self._pane("product") is not None
                     or self._pane("release") is not None)) \
            or isinstance(self.screen, tui_product.ProductScreen)

    def read_product(self) -> None:
        """Gather the Product facts and the release checklist in a worker (git, file reads):
        at start, after each turn and test run, and when the product view opens."""
        if not self._product_shown():
            return
        project, run, accepted = self.project, self.tests_run, self.agent.last_acceptance
        tests_stamp, task_stamp = self.tests_stamp, self.task_stamp

        def work() -> None:
            now = product.tree_fingerprint(project)
            tests = task = None
            if run is not None:
                failed = run.counts.get("failed", 0) + run.counts.get("errors", 0)
                tests = product.Tests(
                    command=run.command, ok=run.ok, passed=run.counts.get("passed"),
                    failed=failed or (len(run.failures) if not run.ok else 0),
                    failures=[f.test for f in run.failures], current=tests_stamp == now)
            if accepted is not None:
                task = product.TaskChecks(
                    [(item.label, item.ok, (item.detail or "").splitlines()[0]
                      if item.detail else "") for item in accepted.items],
                    current=task_stamp == now)
            try:
                report = product.gather(project, tests, task)
            except Exception:  # noqa: BLE001 - the panes keep their last good content
                return
            self.call_from_thread(self._product_read, report)

        self.run_worker(work, thread=True, group="product", exclusive=True, exit_on_error=False)

    def _product_read(self, report: product.Report) -> None:
        blocking = {check.key for check in product.blockers(report.checks)}
        for key in self._blocking - blocking:
            self.product_usage("blocker_cleared", check=key)
        self._blocking = blocking
        self.product_report = report
        self._refresh_product_panes()

    def _product_session(self) -> tui_product.Session:
        files: dict[str, tuple[int, int]] = {}
        for record in self.changes:
            for path, added, removed in record.files:
                old = files.get(path, (0, 0))
                files[path] = (old[0] + added, old[1] + removed)
        return tui_product.Session(
            turns=list(self.product_turns), minutes=(time.monotonic() - self.started_at) / 60,
            files=len(files), added=sum(a for a, _ in files.values()),
            removed=sum(r for _, r in files.values()))

    def _tests_figure(self) -> tuple[str, str] | None:
        """The strip's tests item: (value, tone)."""
        run = self.tests_run
        if run is None:
            return ("not run", "muted")
        passed = run.counts.get("passed", 0)
        failed = run.counts.get("failed", 0) + run.counts.get("errors", 0)
        if run.ok:
            return (f"{passed}/{passed + failed}" if passed else "pass", "success")
        return (f"{failed} fail" if failed else "fail", "error")

    def _product_dirty(self, session: tui_product.Session) -> bool | None:
        report = self.product_report
        if report is None or not report.git.repo or not session.files:
            return None
        return bool(report.git.dirty)

    def _fill_product_pane(self, pane: Any) -> None:
        g, colours = glyphs(), palette(self)
        width = max(16, (pane.scrollable_content_region.width or 38) - 1)
        session = self._product_session()
        body = next(iter(pane.query(Static)), None)
        if body is None:  # still mounting
            return
        body.update(tui_product.product_text(
            self.product_report, session, width, colours, g,
            dirty=self._product_dirty(session)))

    def _fill_release_pane(self, pane: Any) -> None:
        g, colours = glyphs(), palette(self)
        width = max(16, (pane.scrollable_content_region.width or 38) - 1)
        tui_product.fill_release(pane, self.product_report, width, colours, g,
                                 keys=not isinstance(pane.screen, tui_product.ProductScreen))
        pane.border_title = tui_product.release_title(self.product_report, colours)

    def _refresh_product_panes(self) -> None:
        if (pane := self._pane("product")) is not None:
            self._fill_product_pane(pane)
        if (pane := self._pane("release")) is not None:
            self._fill_release_pane(pane)
        strip = self.query_one("#strip", Static)
        if self.layout_name == "workbench" and self.view.mode == "product":
            strip.update(tui_product.strip_text(self.product_report, self._tests_figure(),
                                                max(10, self.size.width - 2), palette(self)))
        self.query_one("#mode", Static).update(
            Text("product" if self.narrow else "product view", style=palette(self)["muted"]))
        for screen in self.screen_stack:
            if isinstance(screen, tui_product.ProductScreen):
                self.fill_product_screen(screen)

    def fill_product_screen(self, screen: tui_product.ProductScreen) -> None:
        colours = palette(self)
        screen.query_one("#path", Static).update(short_path(self.project))
        screen.query_one("#model", Static).update(self._model_label())
        screen.query_one("#state", Static).update(self.query_one("#state", Static).render())
        tabs = Text()
        for key, name in tui_product.TABS:
            current = key == screen.tab
            tabs.append(f"{key} ", style=f"bold {colours['accent']}" if current
                        else colours["muted"])
            tabs.append(name, style=f"bold {colours['fg']}" if current else colours["muted"])
            tabs.append("   ")
        hint = "Esc back"
        tabs.append(" " * max(1, self.size.width - 2 - cell_len(tabs.plain) - len(hint)))
        tabs.append(hint, style=colours["muted"])
        screen.query_one("#ps-tabs", Static).update(tabs)
        if screen.tab == "1":
            self._fill_product_pane(screen.query_one("#ps-product"))
        self._fill_release_pane(screen.query_one("#ps-release"))
        keys = "Tab next pane  Enter evidence  f ask Vega to fix  " + tui_product.SHIP_ROW
        screen.query_one("#ps-keys", Static).update(Text(keys, style=colours["muted"]))
        gauge = self.query_one("#gauge", ContextGauge)
        screen.query_one("#ps-gauge", Static).update(gauge.render())
        screen.query_one("#ps-hint", Static).update(Text("F3 conversation",
                                                         style=colours["muted"]))

    def product_usage(self, event: str, **fields: Any) -> None:
        usage.record(event, session=self.usage_session, **fields)

    def _record_product_turn(self, turn: Turn, *, wrote: bool) -> None:
        """This turn's line in the Product pane: ✗ if its checks failed, ✓ if they passed,
        · if none ran."""
        outcomes: list[bool] = []
        run = self.tests_run
        if run is not None and run.turn == turn.number:
            outcomes.append(run.ok)
        accepted = self.agent.last_acceptance
        if accepted is not None and accepted is not self._acceptance_seen:
            self._acceptance_seen = accepted
            self.task_stamp = product.tree_fingerprint(self.project)
            outcomes.append(accepted.ok)
        outcome = "none" if not outcomes else "pass" if all(outcomes) else "fail"
        if turn.request:
            self.product_turns.append((turn.request, outcome))
        self.product_usage("turn_done", wrote=wrote, checks=outcome)

    def action_product_view(self) -> None:
        """F3 or /product: swap the product view and the code view where the product view fits
        beside the conversation; below that, open (or close) the product screen."""
        if isinstance(self.screen, tui_product.ProductScreen):
            self.pop_screen()
            self.product_usage("product_screen_close")
            return
        if self.screen is not self.default_screen:
            return
        to_product = self.view.mode != "product"
        target = view.preset("product") if to_product else (self._code_view or view.View())
        quick = time.monotonic() - self._product_switched < 10
        if self.layout_name == "workbench" \
                and self.size.width >= view.preset("product").wide_at():
            if to_product:
                self._code_view = self.view
            self.apply_view(target)
            self._product_switched = time.monotonic()
            self.product_usage("view_switch", to=target.mode, within_10s=quick)
            name = "product view" if to_product else "code view"
            self.notify(f"{name.capitalize()}. F3 switches back.", timeout=2)
            return
        self.push_screen(tui_product.ProductScreen())
        self.product_usage("product_screen_open", view=self.view.mode)
        self.read_product()

    def on_product_action(self, event: tui_product.ProductAction) -> None:
        event.stop()
        report = self.product_report
        if event.kind == "ship":
            self.notify(tui_product.SHIP_LATER, severity="warning")
            self.product_usage("ship_pressed")
            return
        if report is None:
            self.notify("The checklist is still being read.")
            return
        checks = {check.key: check for check in report.checks}
        if event.kind == "evidence" and event.key in checks:
            check = checks[event.key]
            self.product_usage("evidence_open", check=check.key)
            self.push_screen(tui_product.EvidenceScreen(check),
                             lambda answer: self.prefill_fix(check) if answer == "fix" else None)
            return
        if event.kind == "fix":
            check = checks.get(event.key) if event.key else next(
                iter(product.blockers(report.checks)), None)
            if check is None:
                self.notify("Nothing blocks shipping, so there is nothing to fix.")
                return
            self.prefill_fix(check)

    def prefill_fix(self, check: product.Check) -> None:
        """f: put the check's prepared request in the message box. It is never sent from here;
        the user edits it and presses Enter."""
        if not check.fix:
            self.notify(f"Nothing to ask Vega for: {check.label} "
                        f"{'passes' if check.mark == 'pass' else check.detail}.")
            return
        while isinstance(self.screen, (tui_product.ProductScreen, tui_product.EvidenceScreen)):
            self.pop_screen()
        prompt = self.query_one("#prompt", PromptInput)
        prompt.load_text(check.fix)
        prompt.move_cursor(prompt.document.end)
        prompt.focus()
        self.pending_fix = (check.key, check.fix)
        self.product_usage("fix_prefilled", check=check.key)
        self.notify("Fix request in the message box. Edit it, then press Enter to send.",
                    timeout=4)

    # status

    def _refresh_status(self) -> None:
        """The status bar fits what it can, in priority order: gauge, memory, the ^P panel hint
        (right-aligned), the last reply time, then the other key hints."""
        colours = palette(self)
        width = self.size.width - 2  # #status has one cell of padding each side
        if self.layout_name == "quiet":
            width = min(width, 98)
        items = set(self.view.status) if self.layout_name == "workbench" \
            else set(view.STATUS_ITEMS)
        gauge = self.query_one("#gauge", ContextGauge)
        gauge_shown = not (self.layout_name == "observatory" and not self.narrow) \
            and "gauge" in items
        gauge.display = gauge_shown
        room = width - (cell_len(gauge.render().plain) if gauge_shown else 0)
        readouts = Text()
        segments: list[tuple[str, str]] = []
        if "memory" in items and (gauge_shown or self.layout_name == "workbench"):
            if self.memory_mb:
                segments.append(("memory ", memory_label(self.memory_mb, self.total_memory_gb)))
            elif self.memory_mb is not None:
                segments.append(("model not loaded", ""))
        needs = sum(row.needs_you for row in self.loop_rows)
        if needs and (self.narrow or self._pane("loops") is None):
            segments.insert(0, ("loops ", f"{needs} waiting"))  # actionable: shown first
        hints = ["^P panel"] if "keys" in items else []
        f3 = ""
        if self.layout_name == "workbench":
            f3 = "F3 product" if self.narrow else (
                "F3 code view" if self.view.mode == "product" else "F3 product view")
        if f3 and hints and self.view.mode == "product":
            hints.append(f3)  # in the product view F3 is the way out: it outranks memory
        used = sum(3 + len(a) + len(b) for a, b in segments) + sum(len(h) for h in hints)
        if used > room - 2 and len(segments) > 1:  # too full: the loops count outranks memory
            dropped = segments.pop()
            used -= 3 + len(dropped[0]) + len(dropped[1])
        if f3 and f3 in hints:
            used += 2  # the gap between the two hints
            while segments and segments[-1][0] != "loops " and used > room - 2:
                dropped = segments.pop()  # memory goes; "loops N waiting" (actionable) stays
                used -= 3 + len(dropped[0]) + len(dropped[1])
        if "reply" in items and (gauge_shown or self.layout_name == "workbench"):
            reply = (("last reply ", seconds_label(self.last_reply[0]))
                     if self.last_reply is not None else ("no replies yet", ""))
            if used + 3 + len(reply[0]) + len(reply[1]) <= room - 2:
                segments.insert(0, reply)
                used += 3 + len(reply[0]) + len(reply[1])
        extras = ("^U undo", "^R review", "F1 help")
        if f3 and f3 not in hints:
            extras = (f3, *extras)  # the code view: the first extra hint, after the readouts
        for extra in extras if hints else ():
            if used + 2 + len(extra) <= room - 2:
                hints.append(extra)
                used += 2 + len(extra)
        if self.agent.teach:  # the badge outranks every readout
            badge = " teach "
            while segments and used + len(badge) + 3 > room - 2:
                dropped = segments.pop()
                used -= 3 + len(dropped[0]) + len(dropped[1])
            if gauge_shown:
                readouts.append("   ")
            readouts.append(badge, style=f"bold {colours['bg']} on {colours['teach']}")
            used += len(badge) + 3
        for label, value in segments:
            if readouts or gauge_shown:
                readouts.append("   ")
            readouts.append(label, style=colours["muted"])
            readouts.append(value, style=colours["readout"])
        self.query_one("#readouts", Static).update(readouts)
        ordered = [hint for hint in (f3, "^U undo", "^R review", "F1 help")
                   if hint and hint in hints]
        if hints:
            ordered.append("^P panel")
        self.query_one("#hints", Static).update(Text("  ".join(ordered), style=colours["muted"]))

    def _read_memory(self) -> None:
        def read() -> None:
            try:
                rss, swap = self.machine_probe()
                value = float(rss)
            except Exception:  # noqa: BLE001 - the memory figure is optional
                return
            self.call_from_thread(self._memory_read, value, swap)

        self.run_worker(read, thread=True, group="memory", exclusive=True, exit_on_error=False)

    def _read_loops(self) -> None:
        """Every LOOPS_EVERY seconds: rows for the Loops pane, the ^P panel and the status bar."""
        def read() -> None:
            from .loops import inbox, status

            try:
                rows, errors = status.rows()
                header = status.header(rows)
                findings = {row.name: [(item, inbox.state_of(item))
                                       for item in inbox.items(row.name, open_only=True)]
                            for row in rows}
            except Exception:  # noqa: BLE001 - the pane is optional; never break the chat
                return
            self.call_from_thread(self._loops_read, rows, errors, header, findings)

        self.run_worker(read, thread=True, group="loops", exclusive=True, exit_on_error=False)

    def _loops_read(self, rows: list[Any], errors: list[str], header: Any,
                    findings: dict[str, list[tuple[dict[str, Any], str]]]) -> None:
        self.loop_rows, self.loop_errors, self.loop_header = rows, errors, header
        self.loop_findings = findings
        self._refresh_loops_pane()
        self._refresh_loops_side()
        self._refresh_status()
        if (self.preview_key or "")[:1] in {"L", "F"}:
            self._render_preview()

    def _refresh_loops_pane(self) -> None:
        pane = self._pane("loops")
        if pane is None:
            return
        g, colours = glyphs(), palette(self)
        width = max(16, pane.content_region.width or 28)
        ascii_only = os.environ.get("VEGA_ASCII", "").strip() not in {"", "0"}
        tui_loops.fill(pane, self.loop_header, self.loop_rows, self.loop_errors,
                       self.loops_expanded, self.loop_findings, width, colours, g,
                       self._frame // 5, ascii_only)
        needs = sum(row.needs_you for row in self.loop_rows)
        pane.border_title = f"Loops [bold {colours['accent']}]{needs}[/]" if needs else "Loops"

    def _refresh_loops_side(self) -> None:
        """The ^P panel's Loops section (the narrow layout's view of the loops; read-only)."""
        g, colours = glyphs(), palette(self)
        width = max(16, self._panel_width() - 2)  # the panel body's own indent
        ascii_only = os.environ.get("VEGA_ASCII", "").strip() not in {"", "0"}
        text = tui_loops.header_text(self.loop_header, colours, g)
        text.truncate(width, overflow="ellipsis")
        if not self.loop_rows and not self.loop_errors:
            text.append("\n" + tui_loops.EMPTY, style=colours["muted"])
        for row in self.loop_rows:
            text.append("\n")
            text.append_text(tui_loops.loop_option(row, width, colours, g, self._frame // 5,
                                                   ascii_only))
        if self.loop_errors:
            text.append(f"\n{len(self.loop_errors)} loop file(s) invalid: vega loops list",
                        style=colours["error"])
        self.query_one("#loops-side", Static).update(text)

    # Models pane (tui_models; downloads and removals through model_store)

    def read_models(self) -> None:
        """Read profiles, Ollama's local list and the catalogue in a worker (only while the
        Models pane is shown): on start, when the pane gets focus, with r, and after a change."""
        if self._pane("models") is None:
            return
        ram = self.total_memory_gb

        def work() -> None:
            from . import model_store

            state = model_store.inventory(total_ram_gb=ram)
            self.call_from_thread(self._models_read, state)

        self.run_worker(work, thread=True, group="models", exclusive=True, exit_on_error=False)

    def _models_read(self, state: Any) -> None:
        self.model_inventory = state
        self._refresh_models_pane()
        if (self.preview_key or "").startswith("M"):
            self._render_preview()

    def _refresh_models_pane(self, again: bool = True) -> None:
        """Draw the rows at the width they get: the list's width less its scrollbar, which only
        appears once the rows overflow, so the first drawing is checked once after layout."""
        pane = self._pane("models")
        if pane is None:
            return
        width = pane.scrollable_content_region.width or pane.size.width
        g, colours = glyphs(), palette(self)
        drawn = max(16, width or 28)
        ascii_only = os.environ.get("VEGA_ASCII", "").strip() not in {"", "0"}
        tui_models.fill(pane, self.model_inventory, drawn, colours, g, ascii_only,
                        self.models_busy)
        if again:
            def check() -> None:
                now = pane.scrollable_content_region.width
                if now and now != drawn:
                    self._refresh_models_pane(False)

            self.call_after_refresh(check)

    def _models_note(self, text: str, level: str = "info") -> None:
        self.notify(text, severity="warning" if level == "warning" else "information",
                    timeout=8)

    def on_models_list_action(self, event: tui_models.ModelsList.Action) -> None:
        if event.action == "refresh":
            self.read_models()
            return
        state = self.model_inventory
        row = state.row(event.name) if state is not None and event.name else None
        if row is None:
            return
        if self.models_busy:
            self._models_note(f"Wait for {self.models_busy} to finish.", "warning")
            return
        if event.action == "switch":
            self._ask_switch(row)
        elif event.action == "download":
            self._ask_download(row)
        elif event.action == "remove":
            self._ask_remove(row)

    def _ask_switch(self, row: Any) -> None:
        if row.active:
            self._models_note(f"{row.name} is already the model Vega runs.")
            return
        if not row.downloaded:
            self._models_note(f"{row.name} is not downloaded; press d to download it first.",
                              "warning")
            return
        if row.kind != "profile":
            self._models_note(f"{row.name} has no Vega profile yet; add one with "
                              f"`vega model add NAME --base {row.tag}`.", "warning")
            return
        if self.working:
            self._models_note("Vega is answering; switch models after this turn.", "warning")
            return
        body = (f"Vega will run {row.name} ({row.tag}, {tui_models.size_text(row)}, "
                f"{row.fit} in {self.total_memory_gb or 0:.0f} GB RAM) from the next message. "
                "This rebuilds the `vega` model; nothing is downloaded.")
        self.push_screen(ChoiceScreen(f"Switch to {row.name}?", body, "Switch", "Keep current",
                                      hint="y switch   n keep"),
                         lambda ok, name=row.name: self._switch_model(name) if ok else None)

    def _switch_model(self, name: str) -> None:
        self.models_busy = f"switching to {name}"
        self._refresh_models_pane()

        def work() -> None:
            try:
                code = models.use(name)
                error = "" if code == 0 else f"building the model exited with {code}"
            except (ValueError, RuntimeError, OSError) as exc:
                error = str(exc)
            self.call_from_thread(self._switched, name, error)

        self.run_worker(work, thread=True, group="models-change", exclusive=True,
                        exit_on_error=False)

    def _switched(self, name: str, error: str) -> None:
        self.models_busy = None
        if error:
            self._add(Note(f"Could not switch to {name}: {error}", "warning"))
        else:
            models.apply_feature_defaults(self.agent.model)
            self.query_one("#model", Static).update(self._model_label())
            self._add(Note(f"Vega now runs {name}. The next message uses it."))
        self.read_models()

    def _ask_download(self, row: Any) -> None:
        if row.downloaded or row.endpoint:
            self._models_note(f"{row.name} is already on this machine.")
            return
        self.models_busy = f"checking {row.name}"
        self._refresh_models_pane()

        def work() -> None:
            from . import model_store

            try:
                plan = model_store.plan_pull(row.name)
            except (ValueError, OSError) as exc:
                plan, error = None, str(exc)
            else:
                error = ""
            self.call_from_thread(self._download_planned, row.name, plan, error)

        self.run_worker(work, thread=True, group="models-change", exclusive=True,
                        exit_on_error=False)

    def _download_planned(self, name: str, plan: Any, error: str) -> None:
        from . import model_store

        self.models_busy = None
        self._refresh_models_pane()
        if plan is None or plan.refusal:
            self._add(Note(f"Not downloading {name}: {error or plan.refusal}", "warning"))
            return
        size = "unknown size" if plan.size_gb is None else f"{plan.size_gb:.1f} GB"
        body = (f"Download {plan.tag} ({size}) through Ollama. Free disk {plan.free_gb:.1f} GB "
                f"now, {plan.free_after_gb:.1f} GB after (Vega keeps {model_store.min_free_gb():.0f}"
                " GB free)." + (f" Then build vega-{plan.profile}." if plan.profile else "")
                + "".join(f" {w}" for w in plan.warnings))
        self.push_screen(ChoiceScreen(f"Download {name}?", body, "Download", "Cancel",
                                      hint="y download   n cancel"),
                         lambda ok: self._download(plan) if ok else None)

    def _download(self, plan: Any) -> None:
        from . import model_store

        self.models_busy = f"downloading {plan.tag}"
        self._refresh_models_pane()
        last = [0.0]

        def progress(event: Any) -> None:
            if event.total and time.monotonic() - last[0] > 0.25:
                last[0] = time.monotonic()
                percent = 100 * event.completed / event.total
                text = f"downloading {plan.tag} {percent:3.0f}%  " \
                       f"{event.completed / 1e9:.1f}/{event.total / 1e9:.1f} GB"
                self.call_from_thread(self._download_progress, text)

        def work() -> None:
            try:
                model_store.pull(plan, progress)
                self.call_from_thread(self._download_progress, f"building vega-{plan.profile}")
                built = model_store.finish_pull(plan)
                error = ""
            except (model_store.OllamaError, ValueError, RuntimeError, OSError) as exc:
                built, error = None, str(exc)
            self.call_from_thread(self._downloaded, plan, built, error)

        self.run_worker(work, thread=True, group="models-change", exclusive=True,
                        exit_on_error=False)

    def _download_progress(self, text: str) -> None:
        self.models_busy = text
        self._refresh_models_pane()

    def _downloaded(self, plan: Any, built: str | None, error: str) -> None:
        self.models_busy = None
        if error:
            self._add(Note(f"Download of {plan.tag} failed: {error}", "warning"))
        else:
            ready = f"; built {built}, Enter on it switches" if built else ""
            self._add(Note(f"Downloaded {plan.tag}{ready}."))
        self.read_models()

    def _ask_remove(self, row: Any) -> None:
        from . import model_store

        if row.active or row.default:
            self._models_note(f"{row.name} is the {'active' if row.active else 'default'} model; "
                              "switch to another before removing it.", "warning")
            return
        try:
            plan = model_store.plan_remove(row.name)
        except (ValueError, OSError) as exc:
            self._models_note(str(exc), "warning")
            return
        if plan.refusal:
            self._models_note(plan.refusal, "warning")
            return
        freed = plan.freed
        amount = "?" if freed is None else str(freed)
        body = f"Delete {', '.join(plan.names)} from Ollama: frees {amount}."
        if plan.kept:
            body += " Keeping " + "; ".join(plan.kept) + "."
        if freed and freed.shared_with:
            body += f" Weights shared with {', '.join(freed.shared_with)} stay on disk."
        if plan.affects:
            body += f" {', '.join(plan.affects)} will need a download again."
        self.push_screen(ChoiceScreen(f"Remove {row.name}?", body, "Remove", "Keep",
                                      hint="y remove   n keep"),
                         lambda ok: self._remove_models(plan) if ok else None)

    def _remove_models(self, plan: Any) -> None:
        from . import model_store

        self.models_busy = f"removing {plan.target}"
        self._refresh_models_pane()

        def work() -> None:
            try:
                removed, error = model_store.remove(plan), ""
            except (model_store.OllamaError, ValueError) as exc:
                removed, error = [], str(exc)
            self.call_from_thread(self._removed, plan, removed, error)

        self.run_worker(work, thread=True, group="models-change", exclusive=True,
                        exit_on_error=False)

    def _removed(self, plan: Any, removed: list[str], error: str) -> None:
        self.models_busy = None
        if error:
            self._add(Note(f"Removing {plan.target} stopped: {error}", "warning"))
        else:
            self._add(Note(f"Removed {', '.join(removed)}."))
        self.read_models()

    # developer panes: git, tests, problems, context

    def read_git(self) -> None:
        """Read git status and the last commits in a worker (only while the Git pane is shown):
        after each turn, on undo, when the pane gets focus, and with r in the pane."""
        if self._pane("git") is None:
            return
        project = self.project

        def work() -> None:
            state = tui_git.read_state(project)
            self.call_from_thread(self._git_read, state)

        self.run_worker(work, thread=True, group="git", exclusive=True, exit_on_error=False)

    def _git_read(self, state: tui_git.GitState) -> None:
        self.git_state = state
        for key in [k for k in self.async_previews if k[:1] in {"G", "C"}]:
            del self.async_previews[key]
        self._refresh_git_pane()
        key = self.preview_key or ""
        if key[:1] in {"G", "C"}:
            self._read_git_preview(key)

    def _refresh_git_pane(self) -> None:
        pane = self._pane("git")
        if pane is None:
            return
        g, colours = glyphs(), palette(self)
        width = max(16, pane.content_region.width or 28)
        ascii_only = os.environ.get("VEGA_ASCII", "").strip() not in {"", "0"}
        tui_git.fill(pane, self.git_state, width, colours, g, ascii_only)

    def _read_git_preview(self, key: str) -> None:
        state, project = self.git_state, self.project
        if state is None:
            return
        if key.startswith("G"):
            number = int(key[1:]) if key[1:].isdigit() else -1
            if not 0 <= number < len(state.files):
                return
            file = state.files[number]
            title = f"{file.path}  {tui_git.describe(file)}"

            def read() -> tuple[str, str]:
                return tui_git.file_diff(project, file)
        else:
            sha = key[1:]
            subject = next((c[2] for c in state.commits if c[0] == sha), "")
            title = f"{sha}  {subject}"

            def read() -> tuple[str, str]:
                return tui_git.commit_show(project, sha)

        def work() -> None:
            kind, body = read()
            self.call_from_thread(self._git_preview_read, key, title, kind, body)

        self.run_worker(work, thread=True, group="git-preview", exclusive=True,
                        exit_on_error=False)

    def _git_preview_read(self, key: str, title: str, kind: str, body: str) -> None:
        self.async_previews[key] = (title, kind, body)
        if self.preview_key == key:
            self._render_preview()

    def _dev_result(self, name: str, args: dict[str, Any], result: str, turn: int) -> None:
        """A finished tool step: test runs feed the Tests pane, writes the Problems pane."""
        if name in {"run_check", "run_command"}:
            command = str(args.get("command", ""))
            if tui_tests.runner(command) is None:
                return
            started = self._command_started.pop(command, None)
            wall = time.monotonic() - started if started is not None else None
            self.tests_run = tui_tests.parse(command, result, turn=turn, wall=wall)
            # stamped now: the agent waits for this handler, so nothing has changed since
            self.tests_stamp = product.tree_fingerprint(self.project)
            self._refresh_tests_pane()
            if (self.preview_key or "").startswith("T"):
                self._render_preview()
        elif name in WRITE_TOOLS and result.startswith(_WRITE_OK):
            path = self._relative(args.get("path"))
            if path is None:
                return
            had_ruff = any(p.source == "ruff" for p in self.problems.get(path, []))
            self._set_problems(path, tui_problems.from_write(path, result))
            if had_ruff:  # its last ruff findings are stale: check that file again
                self.run_lint([path])

    def _relative(self, raw: Any) -> str | None:
        if not raw:
            return None
        path = Path(str(raw))
        if path.is_absolute():
            try:
                return path.resolve().relative_to(self.project).as_posix()
            except ValueError:
                return None
        return path.as_posix().removeprefix("./")

    def _refresh_tests_pane(self) -> None:
        pane = self._pane("tests")
        if pane is None:
            return
        g, colours = glyphs(), palette(self)
        width = max(16, pane.content_region.width or 28)
        tui_tests.fill(pane, self.tests_run, self.tests_running, width, colours, g,
                       self._frame // 2)
        run = self.tests_run
        failed = (run.counts.get("failed", 0) + run.counts.get("errors", 0)) if run else 0
        failed = failed or (len(run.failures) if run else 0)
        pane.border_title = f"Tests [bold {colours['error']}]{failed}[/]" if failed else "Tests"

    def rerun_tests(self) -> None:
        """`r` in the Tests pane: the last test command again through the sandbox, in a worker;
        Esc or Ctrl+C stops it. Not while a turn is running."""
        if self.working:
            self.notify("Tests can be re-run when Vega has finished this turn.",
                        severity="warning")
            return
        if self.tests_running is not None:
            self.notify("The tests are already running. Esc stops them.")
            return
        if self.tests_run is None:
            self.notify("No test run to repeat yet: ask Vega to run the tests first.")
            return
        import threading

        from . import sandbox

        command, project = self.tests_run.command, self.project
        stop = threading.Event()
        self._tests_stop, self.tests_running = stop, command

        def work() -> None:
            started = time.monotonic()
            try:
                result = sandbox.run_check(command, project, stop=stop)
            except Exception as exc:  # noqa: BLE001 - shown in the pane, never raised
                result = f"Could not run the tests: {type(exc).__name__}: {exc}"
            run = tui_tests.parse(command, result, wall=time.monotonic() - started)
            self.call_from_thread(self._tests_done, run)

        self.run_worker(work, thread=True, group="tests", exclusive=True, exit_on_error=False)
        self._refresh_tests_pane()

    def _tests_done(self, run: tui_tests.TestRun) -> None:
        self.tests_running, self._tests_stop = None, None
        self.tests_run = run
        self.tests_stamp = product.tree_fingerprint(self.project)
        self._refresh_tests_pane()
        self.read_product()
        if (self.preview_key or "").startswith("T"):
            self._render_preview()

    def _set_problems(self, path: str, problems: list[tui_problems.Problem]) -> None:
        if problems:
            self.problems[path] = problems
        else:
            self.problems.pop(path, None)
        self._refresh_problems_pane()

    def _refresh_problems_pane(self) -> None:
        self.problem_rows = tui_problems.ordered(self.problems)
        pane = self._pane("problems")
        if pane is None:
            return
        g, colours = glyphs(), palette(self)
        width = max(16, pane.content_region.width or 28)
        tui_problems.fill(pane, self.problem_rows, self.lint_note, self.lint_running, width,
                          colours, g, self._frame // 2)
        errors = sum(p.severity == "error" for p in self.problem_rows)
        warnings = len(self.problem_rows) - errors
        title = "Problems"  # one count: in the error colour when any is an error
        if errors or warnings:
            style = f"bold {colours['error']}" if errors else colours["warning"]
            title += f" [{style}]{errors + warnings}[/]"
        pane.border_title = title
        key = self.preview_key or ""
        if key.startswith("P") and key[1:].isdigit() and int(key[1:]) >= len(self.problem_rows):
            self.preview_key = None  # fixed: the row is gone

    def _recheck_problems(self, paths: list[str]) -> None:
        """After an undo: the write checks again on the files that had problems (worker)."""
        if not paths:
            return
        project = self.project

        def work() -> None:
            found = {path: tui_problems.recheck(project, path) for path in paths}
            self.call_from_thread(self._problems_rechecked, found)

        self.run_worker(work, thread=True, group="problems", exit_on_error=False)

    def _problems_rechecked(self, found: dict[str, list[tui_problems.Problem]]) -> None:
        for path, problems in found.items():
            self._set_problems(path, problems)

    def run_lint(self, paths: list[str] | None = None) -> None:
        """`l` in the Problems pane: the project's `ruff check` on the files changed this session
        (and any with problems), in a worker with a timeout."""
        if self.lint_running:
            return
        if paths is None:
            changed = {path for record in self.changes for path, _, _ in record.files}
            paths = sorted(changed | set(self.problems))
        project, targets = self.project, list(paths)
        self.lint_running = True
        self._refresh_problems_pane()

        def work() -> None:
            problems, note = tui_problems.ruff_check(project, targets)
            self.call_from_thread(self._lint_done, targets, problems, note)

        self.run_worker(work, thread=True, group="lint", exit_on_error=False)

    def _lint_done(self, paths: list[str], problems: list[tui_problems.Problem],
                   note: str) -> None:
        self.lint_running = False
        self.lint_note = f"ruff {tui_problems.stamp()}  {note}"
        by_path: dict[str, list[tui_problems.Problem]] = {}
        for problem in problems:
            by_path.setdefault(problem.path, []).append(problem)
        for path in paths:
            if Path(path).suffix.lower() not in tui_problems.LINTED or \
                    not (self.project / path).is_file():
                continue
            # parse errors from the last write stay; ruff's findings replace the rest
            kept = [p for p in self.problems.get(path, [])
                    if p.source == "write" and p.code in {"syntax", "json", "toml", "yaml",
                                                          "ini"}]
            seen = {(p.line, p.code) for p in kept}
            kept += [p for p in by_path.get(path, []) if (p.line, p.code) not in seen]
            self._set_problems(path, kept)
        self._refresh_problems_pane()
        if (self.preview_key or "").startswith("P"):
            self._render_preview()

    def _context_text(self, width: int) -> Text:
        g, colours = glyphs(), palette(self)
        gauge = self.query_one("#gauge", ContextGauge)
        bar = gauge_bar(gauge.fixed, gauge.used, gauge.window, max(8, width - 1), colours, g)
        return tui_context.render(gauge.parts, gauge.fixed, gauge.used, gauge.window,
                                  self.last_compaction, width, colours, g, bar)

    def _dev_preview(self, key: str, width: int, colours: dict[str, str],
                     g: SimpleNamespace) -> tuple[str, Text] | None:
        kind = key[:1]
        if kind in {"G", "C"} and self.git_state is not None:
            if key not in self.async_previews:
                return "Git", Text("reading" + g.ellipsis, style=colours["muted"])
            title, body_kind, body = self.async_previews[key]
            lines = body.splitlines()[:PREVIEW_LINES]
            if body_kind == "diff":
                text = unified_rows("\n".join(lines), width, colours, g)
            elif body_kind == "new":
                text = diff_rows([], lines, width, colours, g)
            else:
                text = Text(body, style=colours["muted"])
            return _clip(title, max(10, width - 4)), text
        if kind == "T" and self.tests_run is not None and (key == "Tsummary"
                                                           or key[1:].isdigit()):
            return tui_tests.preview(self.tests_run, key, width, colours, g, PREVIEW_LINES)
        if kind == "P" and key[1:].isdigit() and int(key[1:]) < len(self.problem_rows):
            return tui_problems.preview(self.project, self.problem_rows[int(key[1:])], width,
                                        colours, g)
        if kind == "M" and self.model_inventory is not None:
            return tui_models.preview(self.model_inventory, key[1:], width, colours)
        return None

    def on_loops_list_action(self, event: tui_loops.LoopsList.Action) -> None:
        from .loops import inbox, state
        from .loops.definitions import load_all

        kind, key = event.option_id[:1], event.option_id[1:]
        if kind == "F":
            finding = next((f for items in self.loop_findings.values() for f, _ in items
                            if f["id"] == key), None)
            loop_name = finding.get("loop", "") if finding else ""
        else:
            finding, loop_name = None, key
        if event.action == "toggle" and kind == "L":
            self.loops_expanded ^= {loop_name}
        elif event.action == "pause" and loop_name:
            paused = state.is_paused(loop_name)
            state.set_paused(loop_name, not paused)
            self.notify(f"{loop_name} {'resumed' if paused else 'paused'}.")
        elif event.action == "run" and loop_name:
            loops, _ = load_all()
            if any(loop.name == loop_name for loop in loops):
                tui_loops.run_in_background(loop_name)
                self.notify(f"{loop_name}: running now (the model pass waits for the idle rule).")
        elif event.action == "log" and loop_name:
            self.preview_key = f"D{loop_name}"
            self._render_preview()
            return
        elif event.action in {"accepted", "rejected"} and finding is not None:
            try:
                inbox.transition(key, event.action, "from the chat")
            except Exception as exc:  # noqa: BLE001 - e.g. a transition the inbox does not allow
                self.notify(f"Could not mark it {event.action}: {exc}", severity="warning")
                return
            self.notify(f"Finding {event.action}: {finding.get('title', key)}")
        self._read_loops()

    def _memory_read(self, value: float, swap: float | None = None) -> None:
        self.memory_mb = value
        self.swap_mb = swap
        self._refresh_status()
        self._refresh_machine()

    def _set_state(self) -> None:
        state = self.query_one("#state", Static)
        colours = palette(self)
        if self.awaiting_approval:
            state.update(Text("waiting for you", style=f"bold {colours['warning']}"))
        elif self.working:
            g = glyphs()
            seconds = time.monotonic() - (self.turn.started if self.turn else time.monotonic())
            frame = g.spinner[self._frame % len(g.spinner)]
            if self.animation_level == "none":
                frame = g.doing
            dots = g.ellipsis if g.ellipsis == "…" else "..."
            verb = f"stopping after this step{dots}" if self.agent.stop_requested \
                else f"working{dots}"
            label = f"{frame} {verb} {seconds:.0f} s"
            state.update(Text(label, style=colours["accent"]))
            if self.turn is not None and not self.turn.done:
                self.turn.working.update(Text(label, style=colours["muted"]))
        else:
            state.update(Text("ready", style=colours["success"]))

    def _tick(self) -> None:
        if self.tests_running is not None or self.lint_running:
            if not self.working:
                self._frame += 1
            if self._frame % 2 == 0:
                self._refresh_tests_pane()
                self._refresh_problems_pane()
        if self.working:
            self._frame += 1
            self._set_state()
        elif any(row.running_for is not None for row in self.loop_rows):
            self._frame += 1
        if self._frame % 5 == 0 and any(row.running_for is not None for row in self.loop_rows):
            self._refresh_loops_pane()

    def _review_changed(self) -> None:
        state = "on: each edit is shown as a diff for approval" if self.agent.review else "off"
        self._add(Note(f"Review mode {state}."))
        self._refresh_panels()

    def _restyle(self) -> None:
        """Re-render everything that bakes theme colours into its text."""
        for steps in self.query(Steps):
            steps.rebuild()
        for turn in self.query(Turn):
            self._render_turn_changes(turn)
            turn.render_lesson()
        self._refresh_status()
        self.query_one("#gauge", ContextGauge).refresh()
        self._refresh_dividers()
        self.query_one("#brand", Static).update(
            f"{glyphs().star} vega" if self.layout_name == "observatory" else "vega")
        for screen in self.screen_stack:
            if isinstance(screen, (ChoiceScreen, HelpScreen, ViewScreen)):
                screen.restyle()
        self._refresh_panels()
        self._refresh_status()
        self._set_state()

    # actions

    FORCE_QUIT_WINDOW = 2.0  # seconds: a second Ctrl+C within this while still stopping quits

    def action_stop(self) -> None:
        """Ctrl+C: copy the selected text when there is any (the selection is then cleared, so
        the next Ctrl+C stops); otherwise stop the turn (stop_turn)."""
        if self._copy_selection():
            return
        self.stop_turn()

    def stop_turn(self) -> None:
        """Stop the running turn now; a second press within FORCE_QUIT_WINDOW quits Vega."""
        if not self.working and self._tests_stop is not None:
            self._tests_stop.set()
            self.notify("Stopping the test run.")
            return
        if not self.working:
            self.notify("Nothing is running. Ctrl+Q quits.")
            return
        now = time.monotonic()
        if now - getattr(self, "_stop_pressed_at", -10.0) <= self.FORCE_QUIT_WINDOW:
            self.force_quit()
            return
        self._stop_pressed_at = now
        self.agent.request_stop()
        self._decline_pending()
        self._set_state()
        self.notify("Stopping now. Press Ctrl+C again to quit Vega.", timeout=self.FORCE_QUIT_WINDOW)

    def force_quit(self) -> None:
        """Quit even while a model reply or command is still running: run_tui ends the process once
        the terminal is restored, instead of waiting for the worker thread."""
        self.forced_exit = True
        self.agent.request_stop()
        self._decline_pending()
        self.exit()

    def _copy_selection(self) -> bool:
        focused = self.focused
        if isinstance(focused, TextArea) and focused.selected_text:
            self.copy_to_clipboard(focused.selected_text)
            return True
        selected = self.screen.get_selected_text()
        if selected:
            self.copy_to_clipboard(selected)
            self.screen.clear_selection()
            return True
        return False

    def copy_to_clipboard(self, text: str) -> None:
        """Copy through the terminal (OSC 52, Textual's own) and the system clipboard tool
        (clipboard.py; Terminal.app ignores OSC 52). The tool runs in a worker with a timeout,
        so the UI never waits on it; a notice says how many characters were copied."""
        super().copy_to_clipboard(text)
        count = len(text)

        def write() -> None:
            done = clipboard.copy(text)
            with contextlib.suppress(Exception):  # the app may have closed meanwhile
                self.call_from_thread(self._copied, count, done)

        self.run_worker(write, thread=True, group="clipboard", exit_on_error=False)

    def _copied(self, count: int, system: bool) -> None:
        if system:
            self.notify(f"Copied {characters_label(count)}", timeout=2)
        else:
            self.notify(f"Sent {characters_label(count)} to the terminal's clipboard. If paste "
                        "comes up empty, install a clipboard tool (wl-copy, xclip or xsel) or "
                        "select with Option+drag or Shift+drag instead.", timeout=6)

    def on_paste(self, event: events.Paste) -> None:
        """A paste while a pane (not a text box) has focus goes into the message box."""
        if isinstance(self.focused, TextArea) or self.screen is not self.default_screen:
            return
        prompt = self.query_one("#prompt", PromptInput)
        prompt.focus()
        prompt.insert(event.text)

    def _copy_command(self, argument: str) -> None:
        words = argument.split()
        code = bool(words) and words[0] == "code"
        if code:
            words = words[1:]
        finished = [turn for turn in self.turn_list if turn.done and turn.reply]
        if len(words) > 1 or (words and not words[0].isdigit()):
            self._add(Note("Usage: /copy [N] or /copy code [N] (N is a turn number; without "
                           "it, the last reply)", "warning"))
            return
        if words:
            turn = next((t for t in finished if t.number == int(words[0])), None)
            if turn is None:
                self._add(Note(f"Turn {words[0]} has no reply to copy.", "warning"))
                return
        elif finished:
            turn = finished[-1]
        else:
            self._add(Note("No reply to copy yet.", "warning"))
            return
        text = turn.reply
        if code:
            blocks = code_blocks(text)
            if not blocks:
                self._add(Note(f"Turn {turn.number}'s reply has no code block.", "warning"))
                return
            text = "\n\n".join(blocks)
        self.copy_to_clipboard(text)

    # column dividers (ColumnDivider calls these)

    def column_width(self, side: str) -> int:
        """The side column's width on screen now (an `auto` right column has no set width)."""
        return self.query_one(f"#{side}").outer_size.width

    def _column_bounds(self, side: str) -> tuple[int, int] | None:
        """(least, most) columns for `side`: view.py's limits, and never so wide that the
        conversation drops below view.MIN_CONVERSATION (the layout would collapse mid-drag)."""
        least, most = view.LEFT_WIDTHS if side == "left" else view.RIGHT_WIDTHS
        others = view.MIN_CONVERSATION + 2
        if side == "left" and self.view.right:
            others += self.view.right_width or view.MIN_CONVERSATION
        elif side == "right" and self.view.left:
            others += self.view.left_width
        most = min(most, self.size.width - others)
        return (least, most) if most >= least else None

    def column_drag_started(self, side: str) -> None:
        self._drag_from = self.view
        self.column_hint(side, dragging=True)

    def drag_column(self, side: str, width: int) -> None:
        bounds = self._column_bounds(side)
        if bounds is None:
            return
        width = max(bounds[0], min(bounds[1], width))
        current = self.view.left_width if side == "left" else self.view.right_width
        if width != current:
            self._resize_column(view.resize(self.view, side, width), save=False)
        self.column_hint(side, dragging=True)

    def _resize_column(self, new: view.View, *, save: bool) -> None:
        """A width-only view change: columns re-laid out and panes redrawn for the new width,
        without apply_view's pass over every turn's step detail (it would fold steps the user
        opened, on every mouse move)."""
        self.view = new
        if save:
            try:
                view.save(new)
            except OSError as exc:
                self.notify(f"Couldn't save the view: {exc}", severity="warning")
        self._apply_view_styles()
        self._apply_width(self.size.width)
        self.call_after_refresh(self._after_layout)

    def column_drag_finished(self, side: str) -> None:
        start, self._drag_from = self._drag_from, None
        if start is None or start == self.view:
            return
        self._resize_column(self.view, save=True)
        width = self.view.left_width if side == "left" else self.view.right_width
        self.notify(f"{side.capitalize()} column {width} columns, saved. Double-click the edge "
                    "to reset it.", timeout=3)

    def reset_column(self, side: str) -> None:
        default = view.View()
        width = default.left_width if side == "left" else default.right_width
        self._resize_column(view.resize(self.view, side, width), save=True)
        about = f"{width} columns" if width else "auto (shares the space with the conversation)"
        self.notify(f"{side.capitalize()} column reset to {about}.", timeout=3)

    def column_hint(self, side: str | None, *, dragging: bool = False) -> None:
        """The conversation's bottom border says what the edge under the mouse does, and
        while dragging, the width and its limits."""
        conversation = self.query_one("#conversation")
        if side is None:
            conversation.border_subtitle = None
            return
        if dragging:
            width = self.view.left_width if side == "left" else self.view.right_width
            width = width or self.column_width(side)
            least, most = view.LEFT_WIDTHS if side == "left" else view.RIGHT_WIDTHS
            dash = "–" if glyphs().ellipsis == "…" else "-"
            conversation.border_subtitle = Text.assemble(
                (f"{side} column {width}", "bold"), f" of {least}{dash}{most}")
        else:
            conversation.border_subtitle = "drag to resize, double-click to reset"

    def action_undo(self) -> None:
        if self.working or self.screen is not self.screen_stack[0]:
            if self.working:
                self.notify("Undo is available when Vega has finished this turn.",
                            severity="warning")
            return
        self.undo(1)

    def action_toggle_review(self) -> None:
        self.agent.review = not self.agent.review
        self._review_changed()

    def action_toggle_panel(self) -> None:
        screen = self.default_screen
        if self.narrow or self.layout_name == "quiet":
            screen.toggle_class("panel-overlay")
            self.panel_open = screen.has_class("panel-overlay")
        else:
            screen.toggle_class("panel-hidden")
            self.panel_open = not screen.has_class("panel-hidden")
        self._update_dividers()
        self.call_after_refresh(self._refresh_panels)

    def action_toggle_theme(self) -> None:
        self.mode = "light" if self.mode == "dark" else "dark"
        self.theme = LAYOUT_THEMES[self.layout_name][self.mode].name
        self._restyle()

    def action_clear_view(self) -> None:
        for child in list(self.query_one("#conversation").children):
            if child is not self.turn or not self.working:
                child.remove()

    def action_customize(self) -> None:
        if self.layout_name != "workbench":
            self.notify("Customizing applies to the workbench layout: /layout workbench first.",
                        severity="warning")
            return
        if self.screen is self.default_screen:
            self.push_screen(ViewScreen(self.view))

    def action_help(self) -> None:
        if not isinstance(self.screen, HelpScreen):
            self.push_screen(HelpScreen())

    def action_dismiss_notices(self) -> None:
        if self.working or self._tests_stop is not None:  # Esc stops, like Ctrl+C (never copies)
            self.stop_turn()
            return
        if self.default_screen.has_class("panel-overlay"):
            self.default_screen.remove_class("panel-overlay")
            self.panel_open = False
            return
        for note in self.query(".dismissible"):
            note.remove()

    async def action_quit(self) -> None:
        self._quit()

    def _quit(self) -> None:
        if self.working:
            self.force_quit()  # do not wait for a reply or command still running
            return
        self.exit()

    def on_unmount(self) -> None:
        if self.working:
            self.agent.request_stop()
        if self._tests_stop is not None:
            self._tests_stop.set()
        for future in self.pending:
            if not future.done():
                future.set_result(False)


def change_prompt(record: ChangeRecord, width: int, counts_w: int, g: SimpleNamespace,
                  colours: dict[str, str]) -> Text:
    """One turn in Changed this session: a `turn N` sub-head, then its files with +/- lines."""
    text = Text(f"turn {record.turn}", style=colours["muted"])
    for path, added, removed in record.files:
        text.append("\n")
        text.append_text(file_row(path, added, removed, width, counts_w, g, colours))
    return text


def file_row(path: str, added: int, removed: int, width: int, counts_w: int, g: SimpleNamespace,
             colours: dict[str, str]) -> Text:
    counts = f"+{added} {g.minus}{removed}"
    counts_w = max(counts_w, len(counts))
    path_w = max(6, width - counts_w - 1)
    text = Text(clip_middle(path, path_w).ljust(path_w), style=colours["fg"])
    text.append(" " + " " * (counts_w - len(counts)))
    text.append(f"+{added}", style=colours["success"])
    text.append(" ")
    text.append(f"{g.minus}{removed}", style=colours["error"])
    return text


# xterm window operations (CSI 8 ; rows ; cols t): Terminal.app, iTerm2 and most Linux terminals resize
# the window; terminals that do not support it ignore the sequence. A very large size is clamped to the
# screen, which fills it like a desktop app.
_RESIZE = "\x1b[8;{rows};{cols}t"
_FILL_SCREEN = (999, 999)


def _window_resizable(stream: Any = None) -> bool:
    """Resize only a real terminal we can talk to directly: not when switched off (VEGA_WINDOW=keep),
    piped, inside tmux/screen (they swallow or misroute it) or on Windows consoles."""
    stream = stream or sys.__stdout__
    if os.environ.get("VEGA_WINDOW", "").strip().lower() in {"keep", "off", "0", "no"}:
        return False
    if os.name == "nt" or os.environ.get("TMUX") or os.environ.get("STY"):
        return False
    try:
        return bool(stream and stream.isatty())
    except (AttributeError, ValueError):
        return False


def _resize_window(rows: int, cols: int, stream: Any = None) -> None:
    stream = stream or sys.__stdout__
    stream.write(_RESIZE.format(rows=rows, cols=cols))
    stream.flush()


def _wait_for_new_size(before: tuple[int, int], fileno: int, limit: float = 1.5,
                       settle: float = 0.15, size: Any = os.get_terminal_size,
                       sleep: Any = time.sleep, clock: Any = time.monotonic) -> tuple[int, int]:
    """The terminal resizes asynchronously. Starting the app before it is done made Textual lay out
    for the old, small window (side columns collapsed, stale borders) and it never saw the change:
    wait until the size differs from `before` and has stopped changing, or `limit` passes."""
    deadline = clock() + limit
    current = before
    changed_at = None
    while clock() < deadline:
        now = size(fileno)
        now = (now.lines, now.columns)
        if now != current:
            current, changed_at = now, clock()
        elif changed_at is not None and clock() - changed_at >= settle:
            break
        sleep(0.03)
    return current


def run_tui(project: Path, *, review: bool = False, undo_depth: int = 20,
            teach: bool | None = None) -> None:
    """Open the full-screen chat for `project` (blocks until the user quits). The terminal window
    grows to fill the screen while Vega runs and returns to its size afterwards."""
    original = None
    if _window_resizable():
        try:
            fileno = sys.__stdout__.fileno()
            size = os.get_terminal_size(fileno)
            original = (size.lines, size.columns)
            _resize_window(*_FILL_SCREEN)
            _wait_for_new_size(original, fileno)
        except (OSError, ValueError):
            original = None
    app = VegaApp(project, review=review, undo_depth=undo_depth, teach=teach)
    try:
        app.run()
    finally:
        if original is not None:
            try:
                _resize_window(*original)
            except (OSError, ValueError):
                pass
    if getattr(app, "forced_exit", False):
        # A model reply or command may still be running in the worker thread: do not wait for it.
        sys.stdout.flush()
        os._exit(130)
