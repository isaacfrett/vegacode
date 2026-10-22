"""The Models pane: the models this machine has and can get, whether each fits its RAM, which one
Vega runs, and how each scored where scores exist.

Design (2026-10-16): two lines per model. The first is the name with the size right-aligned; the
running model is the pane's one bold name, marked with the accent dot. The second line is the fit
verdict in its own colour (fits = success, tight = warning, won't fit = error) followed by the
scores, labels muted and numbers in the foreground, as many as the width allows (fix and goal
first: they are Vega's own suites). Models on disk come first, then the catalogue's models that
are not downloaded (their names muted, size marked as a download). The header line is the machine:
RAM and free disk, the two numbers every decision here depends on. A download in progress
replaces the header with its percentage. Option ids: `M<name>`; header rows are disabled.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import ClassVar

from rich.cells import cell_len
from rich.text import Text
from textual.binding import Binding, BindingType
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from .model_store import FIT_RULE, SUITE_LABELS, Inventory, ModelRow, ram_needed_gb

SCORE_ORDER = ("fixes", "goals", "humaneval40", "mbpp40")
FIT_STYLE = {"fits": "success", "tight": "warning", "won't fit": "error"}


class ModelsList(OptionList):
    """Workbench Models pane: Enter switches to the highlighted model (asks), d downloads it,
    x removes its downloads (asks; never the active model), r reads the list again."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("d", "act('download')", "Download", show=False),
        Binding("x", "act('remove')", "Remove", show=False),
        Binding("r", "act('refresh')", "Refresh", show=False),
    ]

    class Action(Message):
        def __init__(self, action: str, name: str | None) -> None:
            super().__init__()
            self.action = action
            self.name = name  # the model row's name (None for refresh or a header row)

    def _current(self) -> str | None:
        option = self.highlighted_option
        return option.id[1:] if option is not None and option.id else None

    def _on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.post_message(self.Action("switch", event.option.id[1:]))

    def action_act(self, action: str) -> None:
        self.post_message(self.Action(action, self._current()))


def _clip(text: str, width: int, ellipsis: str) -> str:
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    while text and cell_len(text) > width - 1:
        text = text[:-1]
    return text + ellipsis


def size_text(row: ModelRow) -> str:
    if row.endpoint:
        return "endpoint"
    if row.size_gb is None:
        return "?"
    return f"{'~' if row.size_estimated else ''}{row.size_gb:.1f} GB"


def header(state: Inventory, width: int, colours: dict[str, str], busy: str | None) -> Text:
    text = Text()
    if busy:
        text.append(_clip(busy, width, "…"), style=f"bold {colours['accent']}")
        return text
    parts = []
    if state.total_ram_gb:
        parts.append((f"{state.total_ram_gb:.0f} GB", " RAM"))
    if state.free_disk_gb is not None:
        parts.append((f"{state.free_disk_gb:.1f} GB", " free"))
    for index, (number, label) in enumerate(parts):
        if index:
            text.append("  ")
        text.append(number, style=colours["fg"])
        text.append(label, style=colours["muted"])
    if not state.ollama_up and cell_len(text.plain) + 12 <= width:
        text.append("  " if text else "")
        text.append("Ollama off", style=colours["warning"])
    return text


def name_line(row: ModelRow, width: int, colours: dict[str, str], g: SimpleNamespace,
              ascii_only: bool) -> Text:
    """`● granite-8b         5.3 GB`: marker, name, size right-aligned."""
    text = Text()
    dot = "*" if ascii_only else "●"
    if row.active:
        text.append(dot + " ", style=f"bold {colours['accent']}")
    else:
        text.append("  ")
    size = size_text(row)
    tags = []
    if row.default and not row.active:
        tags.append("default")
    if row.over_limit:
        tags.append("over 10B")
    suffix = ("  " + ", ".join(tags)) if tags else ""
    room = width - 2 - cell_len(size) - 1
    name_style = f"bold {colours['fg']}" if row.active else (
        colours["fg"] if row.downloaded else colours["muted"])
    name = _clip(row.name, max(4, room), g.ellipsis)
    text.append(name, style=name_style)
    if suffix and cell_len(name) + cell_len(suffix) <= room:
        text.append(suffix, style=colours["warning"] if row.over_limit else colours["muted"])
    gap = max(1, width - cell_len(text.plain) - cell_len(size))
    text.append(" " * gap)
    text.append(size, style=colours["fg"] if row.downloaded else colours["muted"])
    return text


def detail_line(row: ModelRow, width: int, colours: dict[str, str]) -> Text:
    """`  fits  fix 7/10  goal 3/8`: the fit verdict, then the scores that fit (if any)."""
    text = Text("  ")
    verdict = row.fit if row.fit != "?" else ("remote" if row.endpoint else "size ?")
    text.append(verdict, style=colours[FIT_STYLE[row.fit]] if row.fit in FIT_STYLE
                else colours["muted"])
    if row.kind == "profile" and row.downloaded and not row.built and not row.endpoint:
        state = "  not built"  # (models to download sit under their own heading)
    else:
        state = ""
    if state and cell_len(text.plain) + cell_len(state) <= width:
        text.append(state, style=colours["muted"])
    for suite in SCORE_ORDER:
        if suite not in row.scores:
            continue
        label, value = SUITE_LABELS[suite], row.scores[suite]
        piece = f"  {label} {value}"
        if cell_len(text.plain) + cell_len(piece) > width:
            break
        text.append(f"  {label} ", style=colours["muted"])
        text.append(value, style=colours["fg"])
    return text


def model_option(row: ModelRow, width: int, colours: dict[str, str], g: SimpleNamespace,
                 ascii_only: bool) -> Text:
    text = name_line(row, width, colours, g, ascii_only)
    text.append("\n")
    text.append_text(detail_line(row, width, colours))
    return text


def ordered(state: Inventory) -> tuple[list[ModelRow], list[ModelRow]]:
    """(on this machine, catalogue to download): the active model first, then by name order."""
    have = [r for r in state.rows if r.downloaded or r.endpoint]
    have.sort(key=lambda r: (not r.active, not r.default, r.kind != "profile"))
    get = [r for r in state.rows if not (r.downloaded or r.endpoint)]
    get.sort(key=lambda r: (r.over_limit, r.kind != "catalogue", not r.scores))
    return have, get


def fill(options: OptionList, state: Inventory | None, width: int, colours: dict[str, str],
         g: SimpleNamespace, ascii_only: bool, busy: str | None = None) -> None:
    """Rebuild the pane, keeping the highlighted model."""
    keep = options.highlighted_option.id if options.highlighted_option else None
    options.clear_options()
    if state is None:
        options.add_option(Option(Text("reading models" + g.ellipsis, style=colours["muted"]),
                                  disabled=True))
        return
    options.add_option(Option(header(state, width, colours, busy), disabled=True))
    if state.error:
        options.add_option(Option(Text(_clip(f"{g.warn} {state.error}", width * 3, g.ellipsis),
                                       style=colours["warning"]), disabled=True))
    have, get = ordered(state)
    if not have and not state.ollama_up:
        options.add_option(Option(Text("Start Ollama (vega server start) to see downloads.",
                                       style=colours["muted"]), disabled=True))
    for row in have:
        options.add_option(Option(model_option(row, width, colours, g, ascii_only),
                                  id=f"M{row.name}"))
    if get:
        options.add_option(Option(Text("to download", style=colours["muted"]), disabled=True))
        for row in get:
            options.add_option(Option(model_option(row, width, colours, g, ascii_only),
                                      id=f"M{row.name}"))
    if keep:
        with contextlib.suppress(Exception):  # the row is gone
            options.highlighted = options.get_option_index(keep)


def preview(state: Inventory, name: str, width: int, colours: dict[str, str]
            ) -> tuple[str, Text] | None:
    """Preview pane for a model: what it is, what it needs, how it scored, and the keys."""
    row = state.row(name)
    if row is None:
        return None
    text = Text()

    def line(label: str, value: str, style: str | None = None) -> None:
        text.append(f"{label:<10}", style=colours["muted"])
        text.append(value + "\n", style=style or colours["fg"])

    line("Ollama", row.tag)
    if row.endpoint:
        line("Server", row.endpoint)
    if row.params_b:
        line("Params", f"{row.params_b:g}B" + ("  (over Vega's 10B limit)" if row.over_limit
                                                  else ""), colours["warning"] if row.over_limit
             else None)
    if row.size_gb:
        disk = "on disk" if row.downloaded else "to download"
        line("Size", f"{size_text(row)} {disk}")
        line("RAM", f"needs ~{ram_needed_gb(row.size_gb):.1f} GB of "
             f"{state.total_ram_gb or 0:.0f} GB: {row.fit}",
             colours[FIT_STYLE[row.fit]] if row.fit in FIT_STYLE else None)
    if row.licence:
        line("Licence", row.licence)
    marks = [m for m, on in (("running now", row.active), ("default", row.default),
                             ("shipped", row.shipped_default)) if on]
    if marks:
        line("Role", ", ".join(marks))
    if row.scores:
        text.append("\n")
        for suite in (*SCORE_ORDER, *sorted(set(row.scores) - set(SCORE_ORDER))):
            if suite in row.scores:
                line(SUITE_LABELS.get(suite, suite), row.scores[suite])
    if row.note:
        text.append("\n" + row.note + "\n", style=colours["muted"])
    text.append("\n")
    keys = [] if row.active else [
        ("Enter", "switch to it" if row.downloaded else "download it first (d)")]
    if not row.downloaded and not row.endpoint:
        keys.append(("d", "download"))
    if row.downloaded and not row.active and not row.default and not row.endpoint:
        keys.append(("x", "remove its downloads"))
    keys.append(("r", "refresh"))
    for key, meaning in keys:
        text.append(f"{key:<6}", style=f"bold {colours['accent']}")
        text.append(meaning + "\n", style=colours["muted"])
    text.append("\n" + FIT_RULE[0].upper() + FIT_RULE[1:] + ".", style=colours["muted"])
    return _clip(f"{row.name}  {row.tag}", max(10, width - 4), "…"), text


__all__ = ["ModelsList", "detail_line", "fill", "header", "model_option", "name_line",
           "preview"]
