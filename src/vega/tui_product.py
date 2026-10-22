"""The product view (product workbench, phase 1): the Product and Release panes, the one-line
product strip at narrow widths, the F3 product screen and the evidence sheet.

Design (product workbench plan and mockups): the verdict ("Not ready: 2
checks block shipping") is the one bold element, on the diff tints (vega-del-bg when blocked,
vega-add-bg when ready). Everything else is quiet: muted labels, status glyphs in their colours,
every number with a denominator. No new colours: the workbench palette only. Data comes from
product.py; this module only draws it and turns keys into messages for the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, ClassVar

from rich.cells import cell_len
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from .product import Check, Report, blockers, score, verdict

LOADING = "Reading the project…"
NOT_READ = ("Vega has not read this project yet. It does on the first message; "
            "`vega context init` reads it now.")
NO_TURNS = "Nothing changed this session. Ask for a change below."
SHIP_LATER = "Shipping from Vega comes later. Ship with your usual command for now."
SHIP_ROW = "s ship: not built yet"
TABS = (("1", "Overview"), ("5", "Release"))  # 2-4 (Preview, Security, Reliability) come later


@dataclass
class Session:
    """What the Product pane shows about this session (the app keeps it)."""
    turns: list[tuple[str, str]] = field(default_factory=list)  # (request, pass|fail|none)
    minutes: float = 0.0
    files: int = 0
    added: int = 0
    removed: int = 0


def _fit(text: str, width: int, ellipsis: str = "…") -> str:
    if cell_len(text) <= width:
        return text
    out = ""
    for char in text:
        if cell_len(out + char) > max(0, width - 1):
            break
        out += char
    return out + ellipsis


def mark_glyph(mark: str, g: SimpleNamespace) -> str:
    return {"pass": g.done, "block": g.fail, "warn": g.warn, "unchecked": g.open}[mark]


def mark_colour(mark: str, colours: dict[str, str]) -> str:
    return {"pass": colours["success"], "block": colours["error"], "warn": colours["warning"],
            "unchecked": colours["muted"]}[mark]


def verdict_style(tone: str, colours: dict[str, str]) -> str:
    if tone == "blocked":
        return f"bold {colours['error']} on {colours['del_bg']}"
    if tone == "ready":
        return f"bold {colours['success']} on {colours['add_bg']}"
    return f"bold {colours['warning']} on {colours['panel']}"


def _split_row(left: Text, right: str, right_style: str, width: int) -> Text:
    """`left` then `right` flush with the right edge; the right side is clipped first."""
    room = width - cell_len(left.plain) - 1
    right = _fit(right, max(0, room)) if room > 0 else ""
    row = Text()
    row.append_text(left)  # a fresh base style: the right side must not inherit left's bold
    row.append(" " *max(1, width - cell_len(left.plain) - cell_len(right)))
    row.append(right, style=right_style)
    return row


# -- Product pane --------------------------------------------------------------------------------

def product_text(report: Report | None, session: Session, width: int, colours: dict[str, str],
                 g: SimpleNamespace, *, dirty: bool | None = None) -> Text:
    """The Product pane: what it is, the verdict and counts, this session, Next and files."""
    muted, fg = colours["muted"], colours["fg"]
    width = max(16, width)
    text = Text()
    if report is None:
        return Text(LOADING, style=muted)
    facts = report.facts
    text.append(_fit(facts.name, width), style=f"bold {fg}")
    if facts.description:
        text.append("\n" + facts.description, style=fg)
    if facts.known:
        what = facts.kind + (f": {', '.join(facts.languages[:3])}" if facts.languages else "")
        text.append("\n" + _fit(what, width), style=muted)
    else:
        text.append("\n" + NOT_READ, style=muted)
    words, tone = verdict(report.checks)
    if cell_len(words) + 2 > width and words.endswith(" shipping"):
        words = words.removesuffix(" shipping")  # "Not ready: 2 checks block" at 80 columns
    text.append("\n\n")
    text.append(_fit(f" {words} ", width), style=verdict_style(tone, colours))
    text.append("\n" + score(report.checks).counts(), style=muted)

    text.append("\n\n")
    right = f"{len(session.turns)} turn{'s' if len(session.turns) != 1 else ''}"
    if session.turns:
        right += f", {max(1, round(session.minutes))} min"
    text.append_text(_split_row(Text("This session", style=f"bold {muted}"), right, muted, width))
    if not session.turns:
        text.append("\n" + NO_TURNS, style=muted)
    for request, outcome in session.turns[-6:]:
        glyph, style = {"pass": (g.done, colours["success"]),
                        "fail": (g.fail, colours["error"])}.get(outcome, (g.bullet, muted))
        text.append("\n" + glyph + " ", style=style)
        text.append(_fit(request, width - 2), style=fg)

    todo = blockers(report.checks)
    if todo:
        text.append("\n\n")
        text.append_text(_split_row(Text("Next", style=f"bold {muted}"), "f asks Vega", muted,
                                    width))
        for check in todo[:4]:
            text.append("\n" + _fit(check.next or check.label, width), style=fg)

    text.append("\n\n")
    text.append("Files ", style=f"bold {muted}")
    if session.files:
        text.append(f"{session.files} changed ", style=fg)
        text.append(f"+{session.added}", style=colours["success"])
        text.append(" ")
        text.append(f"{g.minus}{session.removed}", style=colours["error"])
        if dirty is not None:
            text.append("  " + ("uncommitted" if dirty else "committed"), style=muted)
    else:
        text.append("none changed this session", style=muted)
    return text


class ProductPane(VerticalScroll):
    """The Product pane (a scrollable text pane). f puts the first blocker's fix request in the
    message box."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("f", "fix", "Fix", show=False)]

    def compose(self) -> ComposeResult:
        yield Static("", classes="product-body")

    def action_fix(self) -> None:
        self.post_message(ProductAction("fix", ""))


# -- Release pane --------------------------------------------------------------------------------

def check_row(check: Check, width: int, colours: dict[str, str], g: SimpleNamespace) -> Text:
    mark = check.mark
    left = Text(mark_glyph(mark, g) + " ", style=mark_colour(mark, colours))
    left.append(_fit(check.label, max(4, width - 2)), style=colours["fg"])
    detail_style = colours["muted"] if mark in {"pass", "unchecked"} else mark_colour(mark, colours)
    return _split_row(left, check.detail, detail_style, width)


def release_title(report: Report | None, colours: dict[str, str]) -> str:
    if report is None:
        return "Release"
    s = score(report.checks)
    if s.blocking:
        return f"Release [bold {colours['error']}]{s.blocking} block[/]"
    if s.unchecked_blocking:
        return f"Release [{colours['warning']}]not verified[/]"
    return f"Release [{colours['success']}]ready[/]"


class ReleaseList(OptionList):
    """The release checklist. Enter shows a check's evidence; f puts its fix request in the
    message box (never sent); s says shipping is not built yet. Option ids: R<check key>."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("f", "fix", "Fix", show=False),
        Binding("s", "ship", "Ship", show=False),
    ]

    def selected_key(self) -> str:
        index = self.highlighted
        if index is None:
            return ""
        option_id = self.get_option_at_index(index).id or ""
        return option_id[1:] if option_id.startswith("R") else ""

    def action_fix(self) -> None:
        self.post_message(ProductAction("fix", self.selected_key()))

    def action_ship(self) -> None:
        self.post_message(ProductAction("ship", ""))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if (event.option.id or "").startswith("R"):
            self.post_message(ProductAction("evidence", event.option.id[1:]))


def fill_release(pane: ReleaseList, report: Report | None, width: int,
                 colours: dict[str, str], g: SimpleNamespace, *, keys: bool = True) -> None:
    highlighted = pane.highlighted
    pane.clear_options()
    if report is None:
        pane.add_option(Option(Text(LOADING, style=colours["muted"]), disabled=True))
        return
    for check in report.checks:
        pane.add_option(Option(check_row(check, width, colours, g), id=f"R{check.key}"))
    if keys:
        pane.add_option(Option(Text(""), disabled=True))
        pane.add_option(Option(Text(_fit("Enter evidence  f fix", width),
                                    style=colours["muted"]), disabled=True))
        pane.add_option(Option(Text(_fit(SHIP_ROW, width), style=colours["muted"]),
                               disabled=True))
    count = len(report.checks)
    pane.highlighted = min(highlighted, count - 1) if highlighted is not None else 0


class ProductAction(Message):
    """f (fix), Enter (evidence) or s (ship) in the Product or Release pane. `key` is the
    check's key ("" = the first blocker)."""

    def __init__(self, kind: str, key: str) -> None:
        super().__init__()
        self.kind = kind
        self.key = key


# -- the 80-column strip -------------------------------------------------------------------------

def strip_text(report: Report | None, tests: tuple[str, str] | None, width: int,
               colours: dict[str, str], hint: str = "F3") -> Text:
    """One row: name, the verdict, then tests and secrets. Items drop from the right as the
    terminal narrows (secrets, then tests); the verdict always stays, the name is clipped first.
    `tests` is (value, tone) with tone success, error or muted."""
    muted = colours["muted"]
    if report is None:
        return Text(_fit(LOADING, width), style=muted)
    tone = verdict(report.checks)[1]
    s = score(report.checks)
    chip = {"blocked": f"{s.blocking} block shipping", "ready": "ready to ship",
            "unverified": "not verified"}[tone]
    items: list[tuple[str, str, str]] = []
    if tests is not None:
        items.append(("tests ", tests[0], colours.get(tests[1], muted)))
    secrets = report.secrets
    if secrets is not None:
        exposed = len(secrets[0]) + len(secrets[1])
        items.append(("secrets ", f"{exposed} exposed" if exposed else "ok",
                      colours["error"] if exposed else colours["success"]))
    name = report.facts.name
    hint_w = cell_len(hint) + 2 if hint else 0

    def used(n: int) -> int:
        return (cell_len(name) + 3 + cell_len(chip) + 2
                + sum(2 + cell_len(a) + cell_len(b) for a, b, _ in items[:n]) + hint_w)

    shown = len(items)
    while shown and used(shown) > width:
        shown -= 1
    spare = width - used(shown)
    if spare < 0:  # even alone the name does not fit: clip it, the verdict stays whole
        name = _fit(name, max(1, cell_len(name) + spare))
    text = Text(name, style=f"bold {colours['fg']}")
    text.append("  ")
    text.append(f" {chip} ", style=verdict_style(tone, colours))
    for label, value, style in items[:shown]:
        text.append("  ")
        text.append(label, style=muted)
        text.append(value, style=style)
    if hint:
        pad = width - cell_len(text.plain) - cell_len(hint)
        if pad >= 1:
            text.append(" " * pad)
            text.append(hint, style=muted)
    return text


# -- the product screen (F3 below the wide width) and the evidence sheet --------------------------

class ProductScreen(Screen[None]):
    """The product view at full width for narrow terminals: tab 1 Overview (Product and
    Release side by side), tab 5 Release alone. Esc or F3 goes back to the conversation."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "back", "Back", show=False),
        Binding("1", "tab('1')", show=False),
        Binding("5", "tab('5')", show=False),
    ]

    def __init__(self, tab: str = "1") -> None:
        super().__init__(classes="product-screen")
        self.tab = tab

    def compose(self) -> ComposeResult:
        with Horizontal(id="header", classes="with-mode"):
            yield Static("vega", id="brand")
            yield Static("", id="path", markup=False)
            yield Static("product", id="mode")
            yield Static("", id="model", markup=False)
            yield Static("", id="state")
        yield Static("", id="ps-tabs")
        with Horizontal(id="ps-body"):
            yield ProductPane(id="ps-product", classes="wb-pane wb-text wb-product")
            yield ReleaseList(id="ps-release", classes="wb-pane wb-release")
        yield Static("", id="ps-keys")
        with Horizontal(id="ps-status"):
            yield Static("", id="ps-gauge")
            yield Static("", id="ps-hint")

    def on_mount(self) -> None:
        self.query_one("#ps-product").border_title = "Product"
        self.query_one("#ps-release").border_title = "Release"
        self.action_tab(self.tab)

    def on_resize(self) -> None:
        self.call_after_refresh(self.refill)

    def action_tab(self, tab: str) -> None:
        self.tab = tab
        self.query_one("#ps-product").display = tab == "1"
        release = self.query_one("#ps-release", ReleaseList)
        release.focus()
        self.call_after_refresh(self.refill)
        app: Any = self.app
        app.product_usage("product_tab", tab=tab)

    def action_back(self) -> None:
        app: Any = self.app
        app.action_product_view()  # closes this screen (and logs it when the usage log is on)

    def refill(self) -> None:
        app: Any = self.app
        app.fill_product_screen(self)


class EvidenceScreen(ModalScreen[str]):
    """Enter on a checklist row: what the check found and where it came from. f prepares the
    fix request (dismisses with "fix"); Esc closes."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape,enter,q", "close", "Close", show=False),
        Binding("f", "fix", "Fix", show=False),
    ]

    def __init__(self, check: Check) -> None:
        super().__init__()
        self.check = check

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("", id="dialog-title")
            with VerticalScroll(id="dialog-detail"):
                yield Static("", id="evidence-body")
            yield Static("", id="dialog-hint")

    def on_mount(self) -> None:
        from .tui import glyphs, palette

        colours, g = palette(self.app), glyphs()
        check = self.check
        title = Text(mark_glyph(check.mark, g) + " ", style=mark_colour(check.mark, colours))
        title.append(check.label, style=f"bold {colours['fg']}")
        if check.detail:
            title.append("  " + check.detail, style=colours["muted"])
        what = {"pass": "passes", "block": "blocks shipping", "warn": "a warning; does not block",
                "unchecked": "not checked; never counted as a pass"}[check.mark]
        title.append(f"  ({what})", style=colours["muted"])
        self.query_one("#dialog-title", Static).update(title)
        self.query_one("#evidence-body", Static).update(Text(check.evidence or check.detail,
                                                             style=colours["fg"]))
        hint = "f prepare a fix request   Esc close" if check.fix else "Esc close"
        self.query_one("#dialog-hint", Static).update(hint)

    def action_close(self) -> None:
        self.dismiss("")

    def action_fix(self) -> None:
        self.dismiss("fix" if self.check.fix else "")


__all__ = [
    "EvidenceScreen", "ProductAction", "ProductPane", "ProductScreen", "ReleaseList", "Session",
    "check_row", "fill_release", "product_text", "strip_text",
]
