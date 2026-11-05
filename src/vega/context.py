"""Context-window budget: know the window, measure the prompt, cap tool results and compact history.

Ollama silently drops the oldest tokens when a prompt outgrows num_ctx (keeping only a few at the
start), which loses the system prompt mid-task with no error. This module keeps every request under
the window instead: before a request whose projected prompt passes compact_at() of the window, older
messages are replaced by one deterministic summary built from the history itself (no model call).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

Messages = list[dict[str, Any]]

DEFAULT_NUM_CTX = 8192
CHARS_PER_TOKEN = 3.5
MESSAGE_OVERHEAD_TOKENS = 4  # role markers and separators the chat template adds per message
COMPACT_AT = 0.65  # compact when the projected prompt passes this share of the window
KEEP_RECENT = 0.35  # most recent exchanges kept verbatim, at most this share of the window
HEADROOM = 0.10  # ...and fewer if needed so the next tool result fits before compacting again
TOOL_RESULT_SHARE = 0.20  # cap on any single tool result
SUMMARY_SHARE = 0.08  # cap on the summary message
SUMMARY_PREFIX = "(Vega harness) Summary of earlier work"
HARNESS_PREFIX = "(Vega harness)"


# -- window size ----------------------------------------------------------------------------------

def resolve_num_ctx(model: str) -> tuple[int, str]:
    """(window in tokens, where it came from): VEGA_NUM_CTX, else the Ollama model, else 8192."""
    raw = os.getenv("VEGA_NUM_CTX", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value, "VEGA_NUM_CTX"
    from . import server  # local import keeps this module importable without network helpers

    profile = server._endpoint_profile(model)
    if profile is not None:  # another server: its window is its own setting; trust the profile
        return profile.num_ctx, "profile"
    try:
        discovered = server.context_length(model)
    except Exception:
        discovered = None
    if discovered and discovered > 0:
        return discovered, "server"
    return DEFAULT_NUM_CTX, "default"


def compact_at(num_ctx: int, reply_tokens: int | None = None, projected: int | None = None) -> float:
    """The share of the window at which history is compacted: COMPACT_AT, unless an extension
    sets another (extensions.compact_share; `projected`: the prompt about to be measured)."""
    from . import extensions

    return extensions.compact_share(COMPACT_AT, num_ctx=num_ctx, reply_tokens=reply_tokens,
                                    projected=projected)


# -- the server's window ----------------------------------------------------------------------------

@dataclass
class WindowCheck:
    """Vega's budget against the window the server actually runs (match_window)."""

    model: str  # the model to run: the one asked for, or its window alias (vega-granite-8b-16k)
    budget: int | None  # VEGA_NUM_CTX, when set
    server: int | None  # the served model's num_ctx (/api/show), None when unknown
    error: str = ""  # non-empty: the run must not start


def match_window(model: str) -> WindowCheck:
    """Make the server's window match Vega's. Only VEGA_NUM_CTX can make them differ (otherwise
    the budget is read from the server). When it differs from `model`'s num_ctx, the run uses the
    window alias of the same weights (models.window_alias) if the server has it; otherwise it must
    not start: the server would silently cut every prompt past its own window.

    Unknown windows (server unreachable, an endpoint profile, a Modelfile without num_ctx) pass:
    there is nothing to compare."""
    raw = os.getenv("VEGA_NUM_CTX", "").strip()
    try:
        budget = int(raw) if raw else 0
    except ValueError:
        budget = 0
    if budget <= 0:
        return WindowCheck(model, None, None)
    from . import models, server

    if server._endpoint_profile(model) is not None:
        return WindowCheck(model, budget, None)
    try:
        served = server.context_length(model)
    except Exception:  # noqa: BLE001 - an unreachable server is checked when requests fail
        served = None
    if served is None or served == budget:
        return WindowCheck(model, budget, served)
    alias = models.window_alias(model, budget)
    if alias is not None and alias != model:
        try:
            alias_window = server.context_length(alias)
        except Exception:  # noqa: BLE001
            alias_window = None
        if alias_window == budget:
            return WindowCheck(alias, budget, alias_window)
    profile = models.window_profile_name(model)
    if alias is not None and profile is not None:
        fix = (f"build the matching model once (same weights, no download): `vega model fit-window` "
               f"or `vega model build {profile} --num-ctx {budget}` (creates {alias}); Vega then "
               f"uses it whenever VEGA_NUM_CTX={budget}")
    else:
        fix = (f"create an Ollama model from {model!r} with `PARAMETER num_ctx {budget}` and run it "
               "with --model")
    return WindowCheck(model, budget, served, error=(
        f"VEGA_NUM_CTX={budget} but the server runs {model!r} with num_ctx {served}: "
        f"prompts past {served:,} tokens would be cut silently (the start of the prompt, with the "
        f"system prompt and tools, is lost). To fix: {fix}; or unset VEGA_NUM_CTX to use the "
        "model's own window."))


# -- truncation tripwire --------------------------------------------------------------------------

TRUNCATION_MARGIN = 2  # a prompt this close to the window was cut by the server (max seen: 8,191)


def truncated(prompt_tokens: int | None, window: int | None) -> bool:
    """A reply whose prompt_tokens reached window - TRUNCATION_MARGIN: the server dropped part of
    the prompt (Ollama context shift keeps the first few tokens and the end; no error is sent)."""
    return bool(prompt_tokens and window and prompt_tokens >= window - TRUNCATION_MARGIN)


class TruncationWatch:
    """Wraps an OpenAI client: every reply whose usage shows a cut prompt (truncated) is logged
    as a `truncated_prompt` debug stage and reported through on_truncated(prompt_tokens, window).
    `window` is a callable so the window is discovered lazily (the agent's server window)."""

    def __init__(self, client: Any, window: Any, on_truncated: Any = None):
        self._client, self._window, self._on = client, window, on_truncated
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Completions", (), {})()
        self.chat.completions.create = self._create

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _check(self, response: Any, kwargs: dict[str, Any]) -> None:
        prompt = getattr(getattr(response, "usage", None), "prompt_tokens", None)
        if not isinstance(prompt, int):
            return
        try:
            window = self._window()
        except Exception:  # noqa: BLE001 - instrumentation never fails a request
            return
        if not truncated(prompt, window):
            return
        try:
            from . import debuglog

            debuglog.record("stage", stage="truncated_prompt", prompt_tokens=prompt, window=window,
                            model=kwargs.get("model"), messages=len(kwargs.get("messages") or []))
        except Exception:  # noqa: BLE001, S110
            pass
        if self._on is not None:
            self._on(prompt, window)

    def _create(self, **kwargs: Any) -> Any:
        response = self._client.chat.completions.create(**kwargs)
        if kwargs.get("stream"):
            return self._stream(response, kwargs)
        self._check(response, kwargs)
        return response

    def _stream(self, chunks: Any, kwargs: dict[str, Any]) -> Any:
        try:
            for chunk in chunks:
                if getattr(chunk, "usage", None) is not None:
                    self._check(chunk, kwargs)
                yield chunk
        finally:
            close = getattr(chunks, "close", None)
            if callable(close):
                close()


# -- measuring ------------------------------------------------------------------------------------

def _message_chars(message: dict[str, Any]) -> int:
    size = len(str(message.get("content") or ""))
    for call in message.get("tool_calls") or []:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        size += len(str(function.get("name", ""))) + len(str(function.get("arguments", "")))
    return size


def estimate_tokens(messages: Messages) -> int:
    """Rough prompt size (chars / 3.5 plus per-message overhead) for when no usage is reported."""
    chars = sum(_message_chars(message) for message in messages)
    return int(chars / CHARS_PER_TOKEN) + MESSAGE_OVERHEAD_TOKENS * len(messages)


def estimate_text_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def tool_result_budget(num_ctx: int) -> int:
    """Largest single tool result, in characters."""
    return int(num_ctx * TOOL_RESULT_SHARE * CHARS_PER_TOKEN)


@dataclass
class Budget:
    """Prompt size for one conversation: the last measured size plus what was added since."""

    num_ctx: int
    prompt_tokens: int | None = None  # from the server's usage for the last request
    sent_count: int = 0  # how many messages that request contained
    fixed_tokens: int = 0  # tool specs and other per-request overhead not in the message list
    scale: float = 1.0  # measured tokens / estimated tokens, so estimates match the tokenizer

    def observe(self, prompt_tokens: int | None, sent_count: int, sent_estimate: int = 0) -> None:
        """Record the server's prompt_tokens for a request of sent_count messages, whose own
        estimate (messages plus fixed overhead) was sent_estimate."""
        self.prompt_tokens = prompt_tokens if prompt_tokens and prompt_tokens > 0 else None
        self.sent_count = sent_count
        if self.prompt_tokens is not None and sent_estimate > 0:
            self.scale = min(4.0, max(0.5, self.prompt_tokens / sent_estimate))

    def forget(self) -> None:
        """The message list was rewritten, so the last measurement no longer applies."""
        self.prompt_tokens = None
        self.sent_count = 0

    def projected(self, messages: Messages, extra_tokens: int = 0) -> int:
        """Expected prompt tokens if `messages` were sent now. extra_tokens is text added to each
        request outside the list (the plan); a server measurement already includes it."""
        if self.prompt_tokens is not None and 0 < self.sent_count <= len(messages):
            added = estimate_tokens(messages[self.sent_count:])
            return self.prompt_tokens + int(added * self.scale)
        return int((estimate_tokens(messages) + self.fixed_tokens + extra_tokens) * self.scale)

    def over(self, messages: Messages, extra_tokens: int = 0) -> bool:
        projected = self.projected(messages, extra_tokens)
        return projected > compact_at(self.num_ctx, projected=projected) * self.num_ctx


# -- capping tool results -------------------------------------------------------------------------

_COMMAND_TOOLS = {"run_check", "run_command"}


def cap_tool_result(name: str, text: str, max_chars: int) -> str:
    """Keep a tool result under max_chars, cutting at line boundaries with a note saying so.

    Command output keeps its first line (the exit code) and its tail, where test summaries are;
    everything else keeps its head.
    """
    if len(text) <= max_chars:
        return text
    lines = text.split("\n")
    if name in _COMMAND_TOOLS and len(lines) > 2:
        head, rest = lines[0], lines[1:]
        tail: list[str] = []
        used = len(head) + 80
        for line in reversed(rest):
            if used + len(line) + 1 > max_chars:
                break
            tail.insert(0, line)
            used += len(line) + 1
        omitted = len(rest) - len(tail)
        return "\n".join([head, f"[truncated: {omitted} earlier lines omitted]", *tail])
    kept: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > max_chars - 80:
            break
        kept.append(line)
        used += len(line) + 1
    if not kept:  # one enormous line
        kept = [lines[0][: max_chars - 80]]
    omitted = len(lines) - len(kept)
    hint = "; use read_file with start_line/end_line" if name == "read_file" else ""
    return "\n".join(kept) + f"\n[truncated: {omitted} more lines{hint}]"


# -- grouping -------------------------------------------------------------------------------------

def group_units(messages: Messages) -> list[Messages]:
    """Split messages into units that must stay together: an assistant message with tool calls
    plus the tool results that answer it. Every other message is a unit of its own."""
    units: list[Messages] = []
    for message in messages:
        if message.get("role") == "tool" and units and (
            units[-1][0].get("tool_calls") and units[-1][0].get("role") == "assistant"
        ):
            units[-1].append(message)
        else:
            units.append([message])
    return units


# -- summarising ----------------------------------------------------------------------------------

def _shorten(text: str, width: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def _args(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("function", {}).get("arguments", "")
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _add(items: list[str], item: str, limit: int = 40) -> None:
    if item in items:
        items.remove(item)  # keep the most recent position
    items.append(item)
    del items[:-limit]


LEDGER_TITLE = "Already done in this turn (tool, arguments → outcome)"
LEDGER_FOOTER = ("Do not repeat these calls: with nothing written since, a repeat returns the same "
                 "line. Build on these results or try something different.")


@dataclass
class WorkLog:
    """Facts pulled out of dropped messages. It accumulates across compactions in one conversation,
    so a second compaction does not forget what the first one summarised."""

    requests: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    edits: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    dropped: int = 0

    def absorb(self, messages: Messages) -> None:
        calls: dict[str, tuple[str, dict[str, Any]]] = {}
        for message in messages:
            self.dropped += 1
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "user":
                if not content.startswith(HARNESS_PREFIX):
                    _add(self.requests, _shorten(content, 160), 6)
            elif role == "assistant":
                for call in message.get("tool_calls") or []:
                    name = call.get("function", {}).get("name", "")
                    calls[call.get("id", "")] = (name, _args(call))
                if content.strip():
                    _add(self.findings, _shorten(content, 240), 8)
            elif role == "tool":
                name, args = calls.get(message.get("tool_call_id", ""), ("", {}))
                self._absorb_result(name, args, content)

    def _absorb_result(self, name: str, args: dict[str, Any], result: str) -> None:
        first = result.strip().split("\n", 1)[0]
        path = str(args.get("path", ""))
        if name == "read_file":
            start, end = args.get("start_line"), args.get("end_line")
            label = f"{path}:{start or 1}-{end or 'end'}" if start or end else path
            if not first.startswith(("File not found", "Found multiple", "Tool error")):
                _add(self.files_read, label)
        elif name in {"edit_file", "replace_lines", "write_file",
                      "replace_definition", "remember"}:
            target = path or ".vega/notes.md"
            outcome = _shorten(first, 120)
            warning = re.search(r"WARNING: [^\n]*", result)
            if warning:
                outcome += " " + _shorten(warning.group(0), 100)
            elif "Syntax check: OK" in result:
                outcome += " Syntax OK."
            _add(self.edits, f"{name} {target} → {outcome}", 30)
        elif name in {"run_check", "run_command"}:
            command = _shorten(args.get("command", ""), 100)
            code = re.match(r"exit code: (-?\d+)", first)
            outcome = f"exit {code.group(1)}" if code else _shorten(first, 100)
            lines = [line for line in result.strip().split("\n")[1:] if line.strip()]
            if code and code.group(1) != "0" and lines:
                outcome += f" ({_shorten(lines[-1], 120)})"  # e.g. pytest's "1 failed, 3 passed"
            _add(self.checks, f"{name} `{command}` → {outcome}", 20)
        elif name in {"search_text", "search_code", "find_files"}:
            hits = 0 if first.startswith("No ") else len(result.strip().split("\n"))
            query = _shorten(args.get("query", ""), 60)
            _add(self.searches, f"{name} {query!r} ({hits} lines)", 12)
        elif name == "delegate_task":
            _add(self.findings, "Delegated investigation found: " + _shorten(result, 240), 8)

    def render(self, max_chars: int, ledger: list[str] | None = None) -> str:
        """`ledger`: this turn's read-only calls with their outcomes (agent turn_ledger). When given
        it replaces the Searches section, which has no outcomes."""
        header = (
            f"{SUMMARY_PREFIX} ({self.dropped} older messages were removed to fit the context "
            "window; this summary was built from them. Re-read any file before editing it):"
        )
        sections = [
            ("Earlier requests", self.requests),
            ("Files read", self.files_read),
            ("Edits (tool and outcome)", self.edits),
            ("Checks run (command and exit code)", self.checks),
            ("Searches", self.searches),
            ("Your earlier notes and findings", self.findings),
        ]
        if ledger:
            sections[4] = (LEDGER_TITLE, ledger)
        # Trim the oldest entries of the longest sections until the summary fits.
        lists = {title: list(items) for title, items in sections}
        while True:
            parts = [header]
            for title, _ in sections:
                items = lists[title]
                if not items:
                    continue
                if title == "Files read":
                    parts.append(f"{title}: " + ", ".join(items))
                elif title == LEDGER_TITLE:
                    parts.append(f"{title}:\n" + "\n".join(f"- {item}" for item in items)
                                 + "\n" + LEDGER_FOOTER)
                else:
                    parts.append(f"{title}:\n" + "\n".join(f"- {item}" for item in items))
            text = "\n".join(parts)
            if len(text) <= max_chars:
                return text
            longest = max(lists.values(), key=lambda items: sum(len(item) for item in items))
            if not longest:
                return text[:max_chars]
            longest.pop(0)


# -- compaction -----------------------------------------------------------------------------------

@dataclass
class Compaction:
    messages: Messages
    dropped: int
    before_tokens: int
    after_tokens: int


def compact(
    messages: Messages,
    goal_index: int,
    num_ctx: int,
    log: WorkLog,
    *,
    extra_tokens: int = 0,
    scale: float = 1.0,
    ledger: list[str] | None = None,
    pinned: str = "",
    keep_tokens: int | None = None,
) -> Compaction | None:
    """Rewrite messages as [system, summary, goal, recent...]; None when nothing can be dropped.

    messages[0] is the system message and messages[goal_index] the user request for the current
    turn. Everything between them (earlier turns, earlier summaries) and the oldest exchanges of the
    current turn are folded into `log` and rendered as one summary message. The newest exchanges
    are kept verbatim within KEEP_RECENT of the window, or less when the fixed parts are large, so
    the compacted prompt stays HEADROOM below the compact_at() trigger. They are kept as whole units,
    so a tool call is never separated from its results, and the newest unit is always kept.
    extra_tokens counts text sent with every request outside the message list (tool specs, plan);
    scale converts estimates to the server's measured token counts (Budget.scale).
    pinned (pin_editing_target): the current text of the definitions being edited this turn,
    appended to the summary as it is (not trimmed with the summary; its size counts as fixed).
    keep_tokens: at most this many tokens of recent history are kept (lab grow_context: a grown
    window keeps the starting window's share, so its larger window becomes headroom).
    """
    system, goal = messages[0], messages[goal_index]
    earlier = [m for m in messages[1:goal_index] if not _is_summary(m)]
    units = group_units(messages[goal_index + 1:])
    summary_chars = int(num_ctx * SUMMARY_SHARE * CHARS_PER_TOKEN)
    fixed = estimate_tokens([system, goal]) + extra_tokens + int(summary_chars / CHARS_PER_TOKEN) \
        + int(len(pinned) / CHARS_PER_TOKEN)
    room = int((compact_at(num_ctx) - HEADROOM) * num_ctx) - int(fixed * scale)
    keep_budget = max(0, min(int(KEEP_RECENT * num_ctx), room))
    if keep_tokens is not None:
        keep_budget = min(keep_budget, max(0, keep_tokens))
    kept: list[Messages] = []
    used = 0
    for unit in reversed(units):
        size = int(estimate_tokens(unit) * scale)
        if kept and used + size > keep_budget:
            break
        kept.insert(0, unit)  # the newest unit is always kept, whatever its size
        used += size
    dropped_units = units[: len(units) - len(kept)]
    dropped = earlier + [message for unit in dropped_units for message in unit]
    if not dropped:
        return None
    before = int((estimate_tokens(messages) + extra_tokens) * scale)
    log.absorb(dropped)
    text = log.render(summary_chars, ledger)
    summary = {"role": "user", "content": text + ("\n" + pinned if pinned else "")}
    rebuilt = [system, summary, goal] + [message for unit in kept for message in unit]
    after = int((estimate_tokens(rebuilt) + extra_tokens) * scale)
    return Compaction(rebuilt, len(dropped), before, after)


def _is_summary(message: dict[str, Any]) -> bool:
    content = str(message.get("content", ""))
    return message.get("role") == "user" and content.startswith(SUMMARY_PREFIX)
