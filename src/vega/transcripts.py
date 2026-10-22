"""Compact conversation transcripts, shared by eval attempts and chat sessions."""

from __future__ import annotations

import contextlib
import json
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import features

TEXT_LIMIT = 1200
ARGS_LIMIT = 400


def compact(messages: list[dict[str, Any]], include_system: bool = False) -> list[dict[str, Any]]:
    """Every message, truncated: enough to see what the model did without storing whole files."""
    out = []
    for message in messages if include_system else messages[1:]:
        entry: dict[str, Any] = {"role": message.get("role")}
        if message.get("content"):
            entry["text"] = str(message["content"])[:TEXT_LIMIT]
        if message.get("tool_calls"):
            entry["calls"] = [
                {"name": call.get("function", {}).get("name"),
                 "args": str(call.get("function", {}).get("arguments", ""))[:ARGS_LIMIT]}
                for call in message["tool_calls"]
            ]
        out.append(entry)
    return out


class SessionLog:
    """One chat session's transcript, rewritten after every turn (local diagnostic data).

    With the `project_context` feature it also starts the project's context files (drafts VEGA.md
    on a project's first chat session) and keeps this session's digest in .vega/sessions.md:
    rewritten every `project_context.digest_every()` turns and once more when the session ends
    (close(), else at interpreter exit or when the log is garbage-collected)."""

    def __init__(self, folder: Path, project: Path):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.path = folder / f"{stamp}-{project.name or 'root'}.json"
        self.project = project
        self.turns: list[dict[str, Any]] = []
        self.record: Any = None
        if features.on("project_context"):
            from . import project_context

            project_context.start_session(project)
            if project_context.eligible(project):
                self.record = project_context.SessionRecord.begin(project, stamp)
                self._finalizer = weakref.finalize(self, _write_digest, self.record, True)

    def record_turn(self, message: str, answer: str, seconds: float,
                    requests: list[dict[str, Any]]) -> None:
        self.turns.append({"user": message[:TEXT_LIMIT], "answer": answer[:TEXT_LIMIT * 2],
                           "seconds": round(seconds, 2), "requests": requests})
        if self.record is not None:
            self.record.record_turn(message, answer)

    def close(self) -> None:
        """The session ended: write its final digest (idempotent)."""
        if self.record is not None:
            self._finalizer()

    def save(self, messages: list[dict[str, Any]]) -> None:
        if self.record is not None:
            from . import project_context

            self.record.observe(messages)
            if self.record.turns - self.record.written_turns >= project_context.digest_every():
                _write_digest(self.record)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({
                "project": str(self.project), "turns": self.turns,
                "messages": compact(messages, include_system=True),
            }, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass  # a transcript is a diagnostic aid; never break the chat over it


def _write_digest(record: Any, final: bool = False) -> None:
    # a digest is a convenience: never break the chat, or the interpreter's exit, over it
    with contextlib.suppress(Exception):
        record.write(force=final)
