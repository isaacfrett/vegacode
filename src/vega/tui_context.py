"""The Context pane: what the model's context holds right now, part by part, read-only.

The totals are the status bar gauge's own (fixed prompt, total, window); this pane splits them.
Parts are estimated from characters (context.estimate_*) and then scaled so the fixed parts sum
to the gauge's fixed figure and the rest to its conversation figure, so the pane never disagrees
with the gauge. Files are those whose read_file results are still in the history (a compaction
folds them into the summary).

Design (2026-10-15): one line per part, a swatch in the gauge's own two colours (fixed prompt,
conversation) so the rows read as a legend for the bar, tokens right-aligned, share of the window
right-aligned after them. The headline figure is the one bold element.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from rich.cells import cell_len
from rich.text import Text

from . import context

SNAPSHOT_MARK = "\nProject root:"
EXTRA_MARKS = ("\n\nProject notes (from", "\n\nKnowledge library (material")
FILE_TOOLS = {"read_file"}
SHORT_NAMES = {"tool definitions": "tool specs", "project snapshot": "snapshot",
               "notes, knowledge": "notes", "summary of earlier work": "summary"}
_COMPACTED = re.compile(r"Context compacted: (\d+) older messages summarised "
                        r"\(~([\d,]+) → ~([\d,]+) tokens")


@dataclass
class Breakdown:
    """Estimated tokens per part (unscaled), measured on the agent's thread."""
    fixed: list[tuple[str, int]] = field(default_factory=list)
    conversation: list[tuple[str, int]] = field(default_factory=list)
    files: list[tuple[str, int]] = field(default_factory=list)  # (path, tokens), newest last
    messages: int = 0


@dataclass
class Compaction:
    at: float
    dropped: int
    before: int
    after: int


def parse_compaction(message: str) -> Compaction | None:
    match = _COMPACTED.search(message)
    if not match:
        return None
    return Compaction(time.time(), int(match.group(1)), int(match.group(2).replace(",", "")),
                      int(match.group(3).replace(",", "")))


def _split_system(content: str) -> tuple[int, int, int]:
    """(prompt, project snapshot, notes and knowledge) in estimated tokens."""
    cut = content.find(SNAPSHOT_MARK)
    prompt, rest = (content, "") if cut == -1 else (content[:cut], content[cut:])
    extra_at = min((i for i in (rest.find(mark) for mark in EXTRA_MARKS) if i != -1),
                   default=-1)
    snapshot, extra = (rest, "") if extra_at == -1 else (rest[:extra_at], rest[extra_at:])
    return (context.estimate_text_tokens(prompt), context.estimate_text_tokens(snapshot),
            context.estimate_text_tokens(extra))


def breakdown(messages: list[dict[str, Any]], tool_tokens: int, plan_text: str) -> Breakdown:
    """Split the messages into parts. Call on the thread that owns the agent's messages."""
    out = Breakdown(messages=len(messages))
    if not messages:
        return out
    prompt, snapshot, extra = _split_system(str(messages[0].get("content") or ""))
    out.fixed = [("system prompt", prompt), ("tool definitions", tool_tokens),
                 ("project snapshot", snapshot)]
    if extra:
        out.fixed.append(("notes, knowledge", extra))
    overhead = context.MESSAGE_OVERHEAD_TOKENS
    summary = conversation = results = 0
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    files: dict[str, int] = {}
    for message in messages[1:]:
        tokens = context.estimate_tokens([message])
        role = message.get("role")
        for call in message.get("tool_calls") or []:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            try:
                args = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                args = {}
            calls[str(call.get("id", ""))] = (str(function.get("name", "")),
                                              args if isinstance(args, dict) else {})
        if role == "tool":
            results += tokens
            name, args = calls.get(str(message.get("tool_call_id", "")), ("", {}))
            if name in FILE_TOOLS and args.get("path"):
                content = str(message.get("content") or "")
                if not content.lower().startswith(("error", "file not found", "no such")):
                    path = str(args["path"]).removeprefix("./")
                    files.pop(path, None)  # newest read last
                    files[path] = files.get(path, 0) + max(0, tokens - overhead)
        elif role == "user" and str(message.get("content", "")).startswith(context.SUMMARY_PREFIX):
            summary += tokens
        else:
            conversation += tokens
    out.conversation = [("summary of earlier work", summary), ("conversation", conversation),
                        ("tool results", results),
                        ("plan", context.estimate_text_tokens(plan_text))]
    out.files = list(files.items())
    return out


def _tokens(value: float) -> str:
    return f"{value / 1000:.1f}K" if value >= 1000 else f"{value:.0f}"


def _scaled(parts: list[tuple[str, int]], target: float) -> list[tuple[str, float]]:
    total = sum(tokens for _, tokens in parts)
    if total <= 0:
        return [(name, 0.0) for name, _ in parts]
    return [(name, tokens * target / total) for name, tokens in parts]


def _clip(text: str, width: int, ellipsis: str) -> str:
    if cell_len(text) <= width:
        return text
    while text and cell_len(text) > width - 1:
        text = text[1:]
    return ellipsis + text


def render(parts: Breakdown | None, fixed: float, used: float, window: int,
           compaction: Compaction | None, width: int, colours: dict[str, str],
           g: SimpleNamespace, bar: Text) -> Text:
    """The pane's text. fixed/used/window are the gauge's numbers; `bar` is its bar drawn at
    this width."""
    label, value = colours["muted"], colours["readout"]
    window = max(1, window)
    text = Text()
    size = f"{window // 1024}K" if window % 1024 == 0 else _tokens(window)  # as the gauge says it
    headline = f"{_tokens(used)} of {size}"
    percent = f"{round(100 * used / window)}%"
    text.append(headline, style=f"bold {value}")
    text.append(" tokens".ljust(max(1, width - len(headline) - len(percent))), style=label)
    text.append(percent, style=f"bold {value}")
    text.append("\n")
    text.append_text(bar)
    room = context.COMPACT_AT * window - used
    text.append("\n")
    if room > 0:
        text.append(f"{g.tick}", style=colours["tick"])
        text.append(f" compacts in {_tokens(room)} more", style=label)
    else:
        text.append(f"{g.tick} at the compaction point", style=colours["warning"])
    if parts is None:
        text.append("\n\nmeasuring" + g.ellipsis, style=label)
        return text
    number_w = 6  # "12.3K" plus a space
    share_w = 4  # " 61%"
    name_w = max(6, width - 2 - number_w - share_w)

    def row(swatch: str, swatch_style: str, name: str, tokens: float) -> None:
        if cell_len(name) > name_w:
            name = SHORT_NAMES.get(name, name)
        text.append("\n" + swatch + " ", style=swatch_style)
        text.append(_cut(name, name_w, g.ellipsis).ljust(name_w), style=colours["fg"])
        text.append(_tokens(tokens).rjust(number_w), style=value)
        text.append(f"{round(100 * tokens / window)}%".rjust(share_w), style=label)

    text.append("\n")
    for name, tokens in _scaled(parts.fixed, fixed):
        if tokens >= 1:
            row(g.fixed, colours["fixed"], name, tokens)
    for name, tokens in _scaled(parts.conversation, max(0.0, used - fixed)):
        if tokens >= 1:
            row(g.used, colours["accent"], name, tokens)
    row(g.empty, colours["track"], "free", max(0.0, window - used))
    text.append("\n\n")
    if parts.files:
        text.append(f"files in context  {len(parts.files)}", style=label)
        scale = (used - fixed) / max(1, sum(t for _, t in parts.conversation)) \
            if used > fixed else 1.0
        path_w = max(6, width - number_w)
        for path, tokens in reversed(parts.files):  # newest first
            text.append("\n" + _clip(path, path_w, g.ellipsis).ljust(path_w), style=colours["fg"])
            text.append(_tokens(tokens * scale).rjust(number_w), style=value)
    else:
        text.append("no file contents in context", style=label)
    text.append("\n\n")
    if compaction is None:
        text.append("not compacted this session", style=label)
    else:
        stamp = time.strftime("%H:%M", time.localtime(compaction.at))
        text.append("last compaction ", style=label)
        text.append(stamp, style=value)
        arrow = "→" if g.ellipsis == "…" else "->"
        text.append(f"\n{compaction.dropped} messages summarised", style=label)
        text.append(f"\n{_tokens(compaction.before)} {arrow} {_tokens(compaction.after)} tokens",
                    style=value)
    return text


def _cut(text: str, width: int, ellipsis: str) -> str:
    if cell_len(text) <= width:
        return text
    while text and cell_len(text) > width - 1:
        text = text[:-1]
    return text + ellipsis


__all__ = ["Breakdown", "Compaction", "breakdown", "parse_compaction", "render"]
