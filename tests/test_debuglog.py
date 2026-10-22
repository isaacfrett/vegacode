"""Debug mode: a live JSONL log of requests, replies and events, and the watcher that prints it."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from vega import debuglog
from vega.agent import CodingAgent


def _entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _reply(text: str) -> SimpleNamespace:
    message = SimpleNamespace(content=text, tool_calls=None)
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=8)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)


@pytest.fixture
def debug_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEGA_DEBUG", "1")
    monkeypatch.setattr(debuglog, "_current", None)


def test_off_by_default_writes_nothing() -> None:
    assert debuglog.start("x") is None
    debuglog.record("stage", stage="nothing")
    assert not debuglog.folder().exists()


def test_requests_log_only_messages_the_log_has_not_shown(debug_on: None) -> None:
    log = debuglog.start("unit")
    inner = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: _reply("ok"))))
    client = debuglog.DebugClient(inner)
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "first"}]
    client.chat.completions.create(model="m", messages=history)
    history += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "second"}]
    client.chat.completions.create(model="m", messages=history)
    requests = [e for e in _entries(log.path) if e["kind"] == "request"]
    assert [m["text"] for m in requests[0]["new"]] == ["sys", "first"]
    assert [m["text"] for m in requests[1]["new"]] == ["ok", "second"]
    reply = next(e for e in _entries(log.path) if e["kind"] == "reply")
    assert reply["text"] == "ok" and reply["prompt_tokens"] == 120 and reply["finish"] == "stop"
    assert debuglog.latest_pointer().read_text() == str(log.path)


def test_streamed_replies_are_logged_after_the_last_chunk(debug_on: None) -> None:
    log = debuglog.start("stream")

    def chunk(content: str | None, finish: str | None = None) -> SimpleNamespace:
        delta = SimpleNamespace(content=content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None)

    inner = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: iter([chunk("Hel"), chunk("lo", "stop")]))))
    chunks = debuglog.DebugClient(inner).chat.completions.create(model="m", messages=[], stream=True)
    assert "".join(c.choices[0].delta.content for c in chunks) == "Hello"
    reply = next(e for e in _entries(log.path) if e["kind"] == "reply")
    assert reply["text"] == "Hello" and reply["finish"] == "stop"


def test_agent_turn_and_events_are_logged(debug_on: None, tmp_path: Path) -> None:
    log = debuglog.start("agent")
    agent = CodingAgent(tmp_path, can_write=False)
    assert isinstance(agent.client, debuglog.DebugClient)
    agent._emit("notice", message="Context compacted")
    agent._emit("text", delta="tok")  # streamed tokens are not logged one by one
    kinds = [(e["kind"], e.get("event")) for e in _entries(log.path)]
    assert kinds == [("event", "notice")]


def test_long_text_is_clipped(debug_on: None) -> None:
    log = debuglog.start("clip")
    debuglog.record("stage", stage="big", text="x" * (debuglog.TEXT_LIMIT + 50))
    entry = _entries(log.path)[0]
    assert entry["text"].endswith("(50 more characters)")


def test_watch_prints_a_readable_log(debug_on: None) -> None:
    log = debuglog.start("watch")
    debuglog.record("turn", task="Add a CLI flag", model="vega", features_on=[], features_off=[])
    debuglog.record("event", event="tool_call", data={"name": "read_file", "args": {"path": "a.py"}})
    debuglog.record("outcome", case="c1", passed=False, detail="2 failed", seconds=3.0)
    out = io.StringIO()
    debuglog.watch(log.path, follow=False, out=Console(file=out, width=200))
    text = out.getvalue()
    assert "Add a CLI flag" in text and "read_file" in text and "FAIL c1" in text


def test_pipeline_notices_use_the_renderer_keyword() -> None:
    from vega.ui import TerminalRenderer

    # pipelines once sent notice(text=...), which crashed `vega run` (the renderer takes message=)
    TerminalRenderer(Console(file=io.StringIO()))("notice", {"message": "Direct pipeline"})
    root = Path(__file__).resolve().parents[1] / "src" / "vega"
    for name in ("pipeline.py", "project_pipeline.py"):
        assert '_emit("notice", text=' not in (root / name).read_text(encoding="utf-8")
