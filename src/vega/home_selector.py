"""The TUI selector for home_start: one question, docked at the bottom of the screen so the
conversation stays in view above it.

Each option is a row: its letter, its label, then (muted, in aligned columns) where it is and when
it was last worked on. The recommended option starts highlighted and carries a star and its
reason. Keys: up/down (or k/j) move, a-h or 1-8 pick, Enter takes the highlighted option, r takes
the recommended one, Esc closes it (the user types instead). The last row, "Something else…",
opens a line for the user's own words (a project, a path or a request).

Dismissed with the option's index, the typed text, or None (Esc).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from .tui import glyphs, palette

LETTERS = "abcdefgh"
OTHER_HINT = "Something else… (type a project, a path or a request)"


@dataclass(frozen=True)
class Question:
    """The selector's question: its options and the recommended one (None: no basis) with its
    one-line reason."""

    text: str
    options: list[str] = field(default_factory=list)
    recommended: int | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.recommended is not None and not 0 <= self.recommended < len(self.options):
            object.__setattr__(self, "recommended", None)
        if self.recommended is None:
            object.__setattr__(self, "reason", "")


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


class HomeScreen(ModalScreen[Any]):
    DEFAULT_CSS = """
    HomeScreen { align: center bottom; background: transparent; }
    #decision-dialog {
        width: 100%;
        height: auto;
        max-height: 80%;
        padding: 0 2;
        border-top: solid $accent;
        border-title-color: $accent;
        border-title-style: bold;
        background: $vega-overlay;
    }
    #decision-body { width: 100%; height: auto; }
    #decision-other {
        display: none;
        height: 1;
        margin: 0 0 0 4;
        padding: 0;
        border: none;
        background: $background;
    }
    #decision-other:focus { border: none; }
    #decision-other.shown { display: block; }
    #decision-keys { height: 1; margin-top: 1; color: $vega-muted; }
    """
    AUTO_FOCUS = None  # keys go to the screen; the typing line takes focus only when open
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up,k", "move(-1)", show=False),
        Binding("down,j", "move(1)", show=False),
        Binding("enter", "accept", show=False),
        Binding("r", "accept_all", show=False),
        Binding("escape", "close", show=False),
    ]

    def __init__(self, question: Question, title: str,
                 details: list[tuple[str, str]] | None = None) -> None:
        super().__init__()
        self.question = question
        self.title_text = title
        self.details = details or []  # per option: (where, when); ("", "") for none
        self.cursor = question.recommended if question.recommended is not None else 0
        self.moved = False  # the cursor left its start (Enter with no recommendation needs it)
        self.notice = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="decision-dialog"):
            yield Static("", id="decision-body")
            yield Input(placeholder="A project, a path or what you want to do, then Enter "
                        "(Esc: back)", id="decision-other", disabled=True)
            yield Static("", id="decision-keys")

    def on_mount(self) -> None:
        # inline: the app's `Screen { background }` rule outranks this class's DEFAULT_CSS, and
        # the conversation must stay visible above the panel
        self.styles.background = "transparent"
        self.restyle()

    def _rows(self) -> int:
        return len(self.question.options) + 1  # the options, then Something else…

    def restyle(self) -> None:
        self.query_one("#decision-dialog").border_title = self.title_text
        colours, g = palette(self.app), glyphs()
        fg, muted, accent = colours["fg"], colours["muted"], colours["accent"]
        question = self.question
        text = Text()
        text.append(question.text, style=f"bold {fg}")
        text.append("\n")
        options = [*question.options, OTHER_HINT]
        label_width = min(max(len(o) for o in question.options) + 2, 34)
        where_width = min(max((len(w) for w, _ in self.details), default=0) + 2, 32)
        for index, option in enumerate(options):
            chosen = index == self.cursor
            text.append("\n")
            text.append(f"  {g.prompt} " if chosen else "    ", style=f"bold {accent}")
            text.append(f"{LETTERS[index]}  ", style=accent if chosen else muted)
            other = index == len(question.options)
            style = f"bold {fg}" if chosen else (muted if other else fg)
            where, when = self.details[index] if index < len(self.details) else ("", "")
            if where or when:
                text.append(_clip(option, label_width - 2).ljust(label_width), style=style)
                text.append(_clip(where, where_width - 2).ljust(where_width), style=muted)
                text.append(when.ljust(12), style=muted)
            else:
                text.append(option + "  ", style=style)
            if index == question.recommended:
                text.append(f"{g.star} recommended", style=f"bold {accent}")
                text.append("\n" + " " * 7)
                text.append(question.reason, style=muted)
        if self.notice:
            text.append(f"\n\n{g.warn} {self.notice}", style=colours["warning"])
        self.query_one("#decision-body", Static).update(text)
        arrows = "↑↓" if g.ellipsis == "…" else "up/down"
        last = LETTERS[len(options) - 1]
        self.query_one("#decision-keys", Static).update(
            f"{arrows} choose   a–{last} pick   Enter take   Esc type instead")

    # keys

    def _other(self) -> Input:
        return self.query_one("#decision-other", Input)

    def on_key(self, event: events.Key) -> None:
        if self._other().has_class("shown"):
            if event.key == "escape":
                self._close_other()
                event.stop()
            return
        char = (event.character or "").lower()
        rows = self._rows()
        index = LETTERS.index(char) if char and char in LETTERS[:rows] else -1
        if char.isdigit() and 1 <= int(char) <= rows:
            index = int(char) - 1
        if index >= 0:
            event.stop()
            self.cursor, self.moved = index, True
            self.action_accept()

    def action_move(self, step: int) -> None:
        self.cursor = (self.cursor + step) % self._rows()
        self.moved = True
        self.notice = ""
        self.restyle()

    def action_accept(self) -> None:
        if self.cursor == len(self.question.options):  # Something else…: the typing line
            other = self._other()
            other.disabled = False
            other.add_class("shown")
            other.focus()
            return
        if self.question.recommended is None and not self.moved:
            self.notice = "No recommendation here: choose with the arrows or a letter."
            self.restyle()
            return
        self.dismiss(self.cursor)

    def action_accept_all(self) -> None:
        if self.question.recommended is not None:
            self.dismiss(self.question.recommended)

    def action_close(self) -> None:
        self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        value = event.value.strip()
        if value:
            self.dismiss(value)

    def _close_other(self) -> None:
        other = self._other()
        other.value = ""
        other.remove_class("shown")
        other.disabled = True
        self.set_focus(None)


def open_picker(app: App[Any], question: Question, title: str, done: Callable[[Any], None],
                details: list[tuple[str, str]] | None = None) -> HomeScreen:
    screen = HomeScreen(question, title, details)
    app.push_screen(screen, done)
    return screen
