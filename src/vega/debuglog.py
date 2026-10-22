"""Debug mode: a live log of everything a chat or eval does, to watch from a second terminal.

Turn it on with VEGA_DEBUG=1 (or `vega --debug`). Each session writes one JSON line per event to
~/.vega/debug/<time>-<label>.jsonl as it happens: every model request (the messages the model has
not seen yet in this log, including notes the harness adds), every reply (text, tool calls, finish
reason, tokens, seconds), tool calls and results, and pipeline stages. `vega debug watch` follows
the newest log and prints it readably; it switches to a newer log when one starts (eval attempts
each get their own).

Local diagnostic data only. Prompts and file contents are cut to TEXT_LIMIT characters per entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import server

TEXT_LIMIT = 6000
_lock = threading.Lock()
_current: DebugLog | None = None


def enabled() -> bool:
    return os.environ.get("VEGA_DEBUG", "").strip().lower() not in {"", "0", "false", "off", "no"}


def folder() -> Path:
    return server.STATE_DIR / "debug"


def latest_pointer() -> Path:
    return folder() / "latest"


def _clip(text: Any) -> str:
    text = "" if text is None else str(text)
    if len(text) <= TEXT_LIMIT:
        return text
    return text[:TEXT_LIMIT] + f"\n… ({len(text) - TEXT_LIMIT} more characters)"


class DebugLog:
    """One JSONL file; each line is flushed at once so a watcher sees it immediately."""

    def __init__(self, label: str):
        folder().mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in label)[:60] or "session"
        self.path = folder() / f"{stamp}-{safe}.jsonl"
        self.started = time.monotonic()
        self.seen: set[str] = set()  # hashes of messages already written to this log
        self._file = self.path.open("a", encoding="utf-8")
        latest_pointer().write_text(str(self.path), encoding="utf-8")

    def write(self, kind: str, **data: Any) -> None:
        entry = {"t": round(time.monotonic() - self.started, 2),
                 "at": datetime.now(UTC).isoformat(timespec="seconds"), "kind": kind, **data}
        with _lock:
            self._file.write(json.dumps(entry, default=str) + "\n")
            self._file.flush()

    def new_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The messages this log has not shown yet, compacted (the history is resent every step)."""
        out = []
        for message in messages:
            message = message if isinstance(message, dict) else _as_dict(message)
            key = hashlib.sha1(json.dumps(message, sort_keys=True, default=str).encode()).hexdigest()
            if key in self.seen:
                continue
            self.seen.add(key)
            entry: dict[str, Any] = {"role": message.get("role")}
            if message.get("content"):
                entry["text"] = _clip(message["content"])
            if message.get("tool_calls"):
                entry["calls"] = [_call_dict(call) for call in message["tool_calls"]]
            out.append(entry)
        return out

    def close(self) -> None:
        with _lock:
            self._file.close()


def _as_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return {"role": getattr(message, "role", None), "content": getattr(message, "content", None)}


def _call_dict(call: Any) -> dict[str, str]:
    if isinstance(call, dict):
        function = call.get("function", {})
        return {"name": function.get("name", ""), "args": _clip(function.get("arguments", ""))}
    function = getattr(call, "function", None)
    return {"name": getattr(function, "name", ""), "args": _clip(getattr(function, "arguments", ""))}


def start(label: str) -> DebugLog | None:
    """Begin a new log (closing the previous one) when debug mode is on; None otherwise."""
    global _current
    if not enabled():
        return None
    if _current is not None:
        _current.close()
    _current = DebugLog(label)
    return _current


def current() -> DebugLog | None:
    global _current
    if _current is None and enabled():
        _current = DebugLog("session")
    return _current


def record(kind: str, **data: Any) -> None:
    """Write one event to the current log; does nothing unless debug mode is on."""
    log = current()
    if log is not None:
        log.write(kind, **{key: _clip_value(value) for key, value in data.items()})


def _clip_value(value: Any) -> Any:
    if isinstance(value, str):
        return _clip(value)
    if isinstance(value, dict):
        return {key: _clip_value(item) for key, item in value.items()}
    return value


class DebugClient:
    """Wraps an OpenAI client: logs each chat request and its reply, passes both through unchanged."""

    def __init__(self, client: Any, role: str = "main"):
        self._client = client
        self._role = role
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Completions", (), {})()
        self.chat.completions.create = self._create

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _create(self, **kwargs: Any) -> Any:
        log = current()
        if log is None:
            return self._client.chat.completions.create(**kwargs)
        request: dict[str, Any] = {"role": self._role, "new": log.new_messages(kwargs.get("messages", []))}
        if kwargs.get("tools"):
            request["tools"] = [tool.get("function", {}).get("name") for tool in kwargs["tools"]]
        if kwargs.get("response_format"):
            request["format"] = kwargs["response_format"].get("type")
        if kwargs.get("max_tokens"):
            request["max_tokens"] = kwargs["max_tokens"]
        log.write("request", **request)
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            log.write("error", role=self._role, error=f"{type(exc).__name__}: {exc}",
                      seconds=round(time.monotonic() - started, 2))
            raise
        if kwargs.get("stream"):
            return self._stream(log, response, started)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        log.write("reply", role=self._role, seconds=round(time.monotonic() - started, 2),
                  finish=getattr(choice, "finish_reason", None),
                  text=_clip(choice.message.content),
                  calls=[_call_dict(call) for call in choice.message.tool_calls or []],
                  prompt_tokens=getattr(usage, "prompt_tokens", None),
                  reply_tokens=getattr(usage, "completion_tokens", None))
        return response

    def _stream(self, log: DebugLog, chunks: Any, started: float) -> Iterator[Any]:
        text: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish = prompt_tokens = reply_tokens = None
        first = None
        for chunk in chunks:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                reply_tokens = getattr(usage, "completion_tokens", None)
            for choice in getattr(chunk, "choices", None) or []:
                finish = getattr(choice, "finish_reason", None) or finish
                delta = choice.delta
                if delta is None:
                    continue
                if first is None and (getattr(delta, "content", None) or getattr(delta, "tool_calls", None)):
                    first = round(time.monotonic() - started, 2)
                if getattr(delta, "content", None):
                    text.append(delta.content)
                for fragment in getattr(delta, "tool_calls", None) or []:
                    entry = calls.setdefault(getattr(fragment, "index", 0) or 0, {"name": "", "args": ""})
                    function = getattr(fragment, "function", None)
                    entry["name"] += getattr(function, "name", None) or ""
                    entry["args"] += getattr(function, "arguments", None) or ""
            yield chunk
        log.write("reply", role=self._role, seconds=round(time.monotonic() - started, 2),
                  first_token_seconds=first, finish=finish, text=_clip("".join(text)),
                  calls=[{"name": c["name"], "args": _clip(c["args"])} for _, c in sorted(calls.items())],
                  prompt_tokens=prompt_tokens, reply_tokens=reply_tokens)


# -- watching ---------------------------------------------------------------------------------------

def _short(text: Any, limit: int | None) -> str:
    text = "" if text is None else str(text).strip()
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f" … (+{len(text) - limit} chars)"


def format_entry(entry: dict[str, Any], *, full: bool = False) -> list[tuple[str, str]]:
    """One log entry as (style, text) lines for the watcher; empty for entries it hides."""
    limit = None if full else 500
    t = f"{entry.get('t', 0):>7.1f}s "
    kind = entry.get("kind")
    pad = " " * len(t)
    if kind == "turn":
        on = ", ".join(entry.get("features_on") or []) or "none"
        off = ", ".join(entry.get("features_off") or []) or "none"
        return [("bold cyan", f"{t}━━ turn ━━ model {entry.get('model')} · opt-in on: {on} · off: {off}"),
                ("cyan", pad + _short(entry.get("task"), None if full else 800))]
    if kind == "request":
        lines = [("blue", f"{t}→ request ({entry.get('role')})"
                  + (f" · json {entry['format']}" if entry.get("format") else "")
                  + (f" · {len(entry['tools'])} tools" if entry.get("tools") else ""))]
        for message in entry.get("new") or []:
            role = message.get("role")
            if role == "system" and not full:
                lines.append(("dim", f"{pad}  system prompt ({len(message.get('text', ''))} chars; --full shows it)"))
                continue
            style = {"user": "white", "assistant": "green", "tool": "magenta"}.get(role, "dim")
            if message.get("text"):
                lines.append((style, f"{pad}  + {role}: {_short(message['text'], limit)}"))
            for call in message.get("calls") or []:
                lines.append((style, f"{pad}  + {role} calls {call['name']} {_short(call['args'], limit)}"))
        return lines
    if kind == "reply":
        stats = [f"{entry.get('seconds')}s"]
        if entry.get("first_token_seconds") is not None:
            stats.append(f"first token {entry['first_token_seconds']}s")
        if entry.get("prompt_tokens") is not None:
            stats.append(f"{entry['prompt_tokens']} prompt tok")
        if entry.get("reply_tokens") is not None:
            stats.append(f"{entry['reply_tokens']} reply tok")
        stats.append(f"finish {entry.get('finish')}")
        lines = [("green", f"{t}← reply ({entry.get('role')}) · " + " · ".join(stats))]
        if entry.get("text"):
            lines.append(("green", f"{pad}  {_short(entry['text'], limit)}"))
        for call in entry.get("calls") or []:
            lines.append(("bold green", f"{pad}  call {call['name']} {_short(call['args'], limit)}"))
        return lines
    if kind == "event":
        event, data = entry.get("event"), entry.get("data") or {}
        if event == "request":
            return []  # the request entry itself is logged
        if event == "tool_call":
            return [("magenta", f"{t}⚙ {data.get('name')} {_short(json.dumps(data.get('args'), default=str), 200)}")]
        if event == "tool_result":
            return [("magenta", f"{t}  ↳ {_short(data.get('result'), limit)}")]
        if event == "plan":
            items = "; ".join(f"[{i.get('status')}] {i.get('text') or i.get('step')}" for i in data.get("items") or [])
            return [("yellow", f"{t}☰ plan: {_short(items, limit)}")]
        if event == "discarded":
            return [("yellow", f"{t}✗ discarded ({data.get('reason')}; retry={data.get('retry')})")]
        if event == "notice":
            return [("yellow", f"{t}! {data.get('message')}")]
        if event == "final":
            return [("bold white", f"{t}■ final answer: {_short(data.get('text'), limit)}")]
        return [("dim", f"{t}{event}: {_short(json.dumps(data, default=str), 300)}")]
    if kind == "stage":
        rest = {k: v for k, v in entry.items() if k not in {"t", "at", "kind", "stage"}}
        return [("yellow", f"{t}▸ {entry.get('stage')} {json.dumps(rest, default=str)}")]
    if kind == "outcome":
        style = "bold green" if entry.get("passed") else "bold red"
        verdict = "PASS" if entry.get("passed") else "FAIL"
        return [(style, (f"{t}{verdict} {entry.get('case')} in {entry.get('seconds')}s "
                        f"{_short(entry.get('detail'), limit)}"))]
    if kind == "error":
        return [("bold red", f"{t}error ({entry.get('role')}): {entry.get('error')}")]
    return [("dim", f"{t}{kind}: {_short(json.dumps(entry, default=str), 300)}")]


def _latest() -> Path | None:
    try:
        path = Path(latest_pointer().read_text(encoding="utf-8").strip())
    except OSError:
        return None
    return path if path.is_file() else None


def watch(path: Path | None = None, *, full: bool = False, follow: bool = True,
          out: Any = None, poll: float = 0.3) -> None:
    """Print a log as it grows. Without `path`, follow the newest log and switch to newer ones."""
    from rich.console import Console
    from rich.text import Text

    console = out or Console(highlight=False)
    pinned = path is not None
    current_path = path or _latest()
    while current_path is None:
        if not follow:
            console.print("No debug logs yet. Start Vega with VEGA_DEBUG=1 or `vega --debug`.")
            return
        console.print(Text("Waiting for a debug log (start Vega with --debug or VEGA_DEBUG=1)…", style="dim"))
        while current_path is None:
            time.sleep(1)
            current_path = _latest()
    while True:
        console.print(Text(f"── {current_path.name} ──", style="bold"))
        with current_path.open(encoding="utf-8") as handle:
            last_check = time.monotonic()
            while True:
                line = handle.readline()
                if line:
                    if not line.endswith("\n"):  # a line still being written
                        handle.seek(handle.tell() - len(line.encode("utf-8")))
                        time.sleep(poll)
                        continue
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    for style, text in format_entry(entry, full=full):
                        first, *rest = text.split("\n")
                        # continuation lines go under the text, clear of the time column
                        body = "\n".join([first, *(" " * 13 + line for line in rest)])
                        console.print(Text(body, style=style), soft_wrap=True)
                    continue
                if not follow:
                    return
                if not pinned and time.monotonic() - last_check > 1:
                    last_check = time.monotonic()
                    newest = _latest()
                    if newest is not None and newest != current_path:
                        current_path = newest
                        break
                time.sleep(poll)
