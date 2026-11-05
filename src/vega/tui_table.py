"""Markdown tables in replies as real terminal tables (2026-10-28 mockup): thin rounded borders, a
bold header row with a rule under it, aligned columns, numbers right-aligned. A table wider than
the reading column wraps its cells; one that cannot fit even wrapped (long words, many columns)
scrolls sideways inside itself, so the conversation never scrolls horizontally.

Drawn with Rich's Table inside Textual's Markdown (the same renderer the rest of the reply uses):
TableBlock replaces Textual's grid-based MarkdownTable, which cut cells off with an ellipsis."""

from __future__ import annotations

import re
from typing import Any

from rich import box
from rich.measure import Measurement
from rich.style import Style as RichStyle
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import HorizontalScroll
from textual.content import Content
from textual.widgets import Static
from textual.widgets._markdown import MarkdownBlock, MarkdownTable, MarkdownTH

# a cell that reads as a number: optional sign or currency, digits with , or . separators, then an
# optional %, x or a short unit ("12", "-3.5%", "$1,200", "4.1 GB", "120 ms", "2x")
_NUMBER = re.compile(r"^[-+±~≈]?[$€£]?\d[\d,_]*(\.\d+)?\s?(%|x|×|[kKMGT]?B|ms|s|h|[kKMB])?$")

_INLINE = {".strong": "bold", ".em": "italic", ".s": "strike"}


def is_number(text: str) -> bool:
    return bool(_NUMBER.match(text.strip()))


def numeric_columns(rows: list[list[str]], width: int) -> list[bool]:
    """A column is numeric when every non-empty body cell in it is a number."""
    out = []
    for index in range(width):
        cells = [row[index].strip() for row in rows if index < len(row) and row[index].strip()]
        out.append(bool(cells) and all(is_number(cell) for cell in cells))
    return out


def _alignment(block: MarkdownBlock) -> str | None:
    style = str(block._token.attrs.get("style", "")) if block._token.attrs else ""
    match = re.search(r"text-align:\s*(left|right|center)", style)
    return match.group(1) if match else None


def cell_text(content: Content, accent: str) -> Text:
    """A cell's inline Markdown as Rich text: bold, italic and strike kept, inline code in the
    accent colour (as in paragraphs)."""
    text = Text(content.plain)
    for span in content.spans:
        style = span.style
        if isinstance(style, str):
            name = _INLINE.get(style) or (accent if style == ".code_inline" else None)
            if name:
                text.stylize(name, span.start, span.end)
    return text


def build_table(headers: list[Text], rows: list[list[Text]], aligns: list[str | None],
                colours: dict[str, str]) -> Table:
    width = len(headers)
    numeric = numeric_columns([[cell.plain for cell in row] for row in rows], width)
    table = Table(box=box.ROUNDED, border_style=colours.get("rule") or None,
                  header_style=RichStyle(bold=True), show_edge=True, pad_edge=True,
                  expand=False, padding=(0, 1))
    for index, header in enumerate(headers):
        align = aligns[index] if index < len(aligns) and aligns[index] else (
            "right" if numeric[index] else "left")
        table.add_column(header, justify=align, overflow="fold")  # type: ignore[arg-type]
    for row in rows:
        cells = list(row[:width]) + [Text("")] * (width - len(row))
        table.add_row(*cells)
    return table


class TableBlock(MarkdownTable):
    """A Markdown table drawn by Rich, scrolling sideways inside itself when it cannot fit."""

    DEFAULT_CSS = """
    TableBlock {
        width: 1fr;
        height: auto;
        margin: 0 0 1 0;
        background: transparent;
        &:light { background: transparent; }
    }
    TableBlock > HorizontalScroll {
        width: 1fr;
        height: auto;
        overflow-y: hidden;
        scrollbar-size-horizontal: 1;
    }
    TableBlock .vega-table { width: auto; height: auto; }
    """

    def compose(self) -> ComposeResult:
        self._collect()
        scroller = HorizontalScroll()
        scroller.can_focus = False  # the prompt keeps the focus; the wheel still scrolls it
        with scroller:
            yield Static("", classes="vega-table")

    def _collect(self) -> None:
        def walk(block: MarkdownBlock) -> Any:
            for child in block._blocks:
                if child._blocks:
                    yield from walk(child)
                yield child

        self._aligns = [_alignment(b) for b in walk(self) if isinstance(b, MarkdownTH)]
        self._headers, self._rows = self._get_headers_and_rows()

    def on_mount(self) -> None:
        self.call_after_refresh(self.draw)

    def on_resize(self) -> None:
        self.draw()

    def table(self) -> Table:
        from .tui import palette

        colours = palette(self.app)
        accent = colours.get("accent", "")
        headers = [cell_text(h, accent) for h in self._headers]
        rows = [[cell_text(c, accent) for c in row] for row in self._rows]
        return build_table(headers, rows, self._aligns, colours)

    def draw(self) -> None:
        """Fit the table to the column: its natural width when that fits, wrapped cells when it
        does not, and its narrowest wrapped width (scrolling sideways) below that."""
        available = self.content_region.width or self.size.width
        if available <= 0:
            return
        table = self.table()
        console = self.app.console
        measure = Measurement.get(console, console.options.update_width(10_000), table)
        width = min(measure.maximum, available) if measure.minimum <= available \
            else measure.minimum
        static = self.query_one(".vega-table", Static)
        static.styles.width = width
        static.update(table)

    async def _update_from_block(self, block: MarkdownBlock) -> None:
        await MarkdownBlock._update_from_block(self, block)
