"""Window composition: what each model request's prompt is made of, in estimated tokens per part.

One debug stage per model request (debuglog.DebugClient writes it just before the `request` row,
only with VEGA_DEBUG=1):

    {"kind": "stage", "stage": "window", "role": "main", "hinted": true,
     "parts": {"system": 812, "tools": 1460, "project_context": 0, "memory": 0, "history": 0,
               "tool_results": 2210, "request": 64, "notes": 120},
     "messages": {"system": 1, ...}, "total": 4666, "estimate": 4668}

Counts only, never text (systems review 2026-10-29, context manager C0: the window's contents per
request were not logged, so a context design could not be based on data).

Parts:
  system           the system message minus the project context section in it
  tools            the tool schemas sent with the request (JSON)
  project_context  VEGA.md, session digests and notes in the system message (agent loop)
  memory           extension context outside the history: file cards, capability card...
  history          earlier user and assistant messages (assistant tool calls included)
  tool_results     tool messages (and constrained mode's `Tool result (name):` user messages)
  request          the user's request this turn (a pipeline: its first prompt, inputs included)
  notes            harness notes: send-backs, plan notes, repair feedback after the request

The agent loop names its parts with hint() right before each request; an unhinted request (the
pipelines, clarify, planners) is split by role: its first user message is the request and later
user messages are notes. Tokens use the estimator the budget uses (context.estimate_tokens: chars
/ 3.5 plus 4 per message), so `total` (the sum of the parts) is within a few tokens of
`estimate`, and both can be set against the reply's measured prompt_tokens.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from . import context

PARTS = ("system", "tools", "project_context", "memory", "history", "tool_results", "request",
         "notes")
HARNESS_PREFIXES = ("(Vega harness)", "(Vega)")
TOOL_RESULT_PREFIX = "Tool result ("  # constrained mode's tool results (agent.to_action_messages)

_local = threading.local()


def hint(*, task: str = "", project_context: str = "", memory: list[str] | None = None,
         notes: list[str] | None = None, step: int | None = None) -> None:
    """Name the parts of this thread's next model request (consumed by it). Texts are compared
    with the request's messages, never logged."""
    _local.hint = {"task": task, "project_context": project_context,
                   "memory": [m for m in memory or [] if m], "notes": [n for n in notes or [] if n],
                   "step": step}


def take_hint() -> dict[str, Any] | None:
    found = getattr(_local, "hint", None)
    _local.hint = None
    return found


def _content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _role(message: Any) -> str:
    return str(message.get("role") if isinstance(message, dict) else getattr(message, "role", ""))


def _chars(message: Any) -> int:
    if not isinstance(message, dict):
        message = {"content": _content(message)}
    return context._message_chars(message)


def compose(messages: list[Any], tools: list[dict[str, Any]] | None = None,
            found: dict[str, Any] | None = None) -> dict[str, Any]:
    """{parts, messages, total, estimate, hinted[, step]} for one request (see the module doc)."""
    found = found or {}
    task = found.get("task") or ""
    section = found.get("project_context") or ""
    memory = set(found.get("memory") or ())
    notes = set(found.get("notes") or ())
    roles = [_role(m) for m in messages]
    texts = [_content(m) for m in messages]
    request_at = -1
    if task:  # the agent loop: the newest user message holding the task
        request_at = next((i for i in range(len(messages) - 1, -1, -1)
                           if roles[i] == "user" and texts[i].startswith(task)), -1)
    hinted = request_at >= 0
    if not hinted:  # by role: the first user message that is not a harness note
        request_at = next((i for i, (r, t) in enumerate(zip(roles, texts, strict=True))
                           if r == "user" and not t.startswith(HARNESS_PREFIXES)
                           and t not in memory and t not in notes), -1)
    chars = dict.fromkeys(PARTS, 0)
    counts = dict.fromkeys(PARTS, 0)
    overhead = dict.fromkeys(PARTS, 0)
    for i, message in enumerate(messages):
        role, text, size = roles[i], texts[i], _chars(message)
        if role == "system":
            part = "system"
            if section and section in text:
                chars["project_context"] += len(section)
                size -= len(section)
        elif role == "tool":
            part = "tool_results"
        elif role == "assistant":
            part = "history"
        elif i == request_at:
            part = "request"
        elif text in memory:
            part = "memory"
        elif text in notes or text.startswith(HARNESS_PREFIXES) or (0 <= request_at < i):
            part = "tool_results" if text.startswith(TOOL_RESULT_PREFIX) else "notes"
        else:
            part = "tool_results" if text.startswith(TOOL_RESULT_PREFIX) else "history"
        chars[part] += size
        counts[part] += 1
        overhead[part] += context.MESSAGE_OVERHEAD_TOKENS
    parts = {part: int(chars[part] / context.CHARS_PER_TOKEN) + overhead[part] for part in PARTS}
    if tools:
        parts["tools"] = context.estimate_text_tokens(json.dumps(tools))
        counts["tools"] = len(tools)
    out: dict[str, Any] = {
        "hinted": hinted, "parts": parts, "messages": counts, "total": sum(parts.values()),
        "estimate": context.estimate_tokens([m if isinstance(m, dict) else {"content": _content(m)}
                                             for m in messages]) + parts["tools"]}
    if found.get("step") is not None:
        out["step"] = found["step"]
    return out
