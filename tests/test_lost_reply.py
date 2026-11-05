"""Lost replies: completion tokens but no text and no tool call (Ollama dropped a tool call it could
not parse). The loop recognises them, the debug log keeps the raw reply, and the lost_reply hook
may send a note back (vega_lab.lost_output); without a hook the handling is unchanged."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from test_agent_loop import FakeMessage, call, text

from vega import debuglog, extensions
from vega.agent import EMPTY_FINAL, CodingAgent


def lost() -> FakeMessage:
    return FakeMessage(content=None, tool_calls=None)


def agent_with_usage(project: Path, replies: list[tuple[FakeMessage, int]]):
    """Each reply carries usage.completion_tokens, like Ollama's OpenAI endpoint."""
    agent = CodingAgent(project, allow_commands=False)
    seen: list[list[dict]] = []

    def create(**kwargs):
        seen.append(list(kwargs["messages"]))
        message, tokens = replies.pop(0)
        usage = SimpleNamespace(prompt_tokens=2123, completion_tokens=tokens)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=usage)

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, seen


def with_hook(monkeypatch: pytest.MonkeyPatch, fn) -> None:
    module = ModuleType(extensions.PACKAGE)
    module.lost_reply = fn
    monkeypatch.setitem(sys.modules, extensions.PACKAGE, module)
    monkeypatch.delenv(extensions.DISABLE_ENV, raising=False)
    extensions.reset()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from switches import switch_off

    switch_off(monkeypatch, "question_pipeline")  # the agent loop answers; these test its reply handling
    (tmp_path / "a.py").write_text("x = 1\n")
    return tmp_path


def test_without_a_hook_a_lost_reply_is_handled_as_before(project: Path) -> None:
    # 200 tokens (a write_file call with `\$1` in its JSON, dropped by the server): as before, the
    # empty reply is sent back once with EMPTY_FINAL
    agent, seen = agent_with_usage(project, [
        (call("read_file", path="a.py"), 20), (lost(), 200), (text("a.py sets x to 1."), 9)])
    assert agent.run("What does a.py set?") == "a.py sets x to 1."
    assert seen[2][-1] == {"role": "user", "content": EMPTY_FINAL}


def test_the_hook_gets_the_lost_reply_and_its_note_is_sent_back(project: Path, monkeypatch) -> None:
    calls: list[dict] = []

    def hook(agent, task, **context):
        calls.append(context)
        return "NOTE: your call was lost"

    with_hook(monkeypatch, hook)
    agent, seen = agent_with_usage(project, [
        (call("read_file", path="a.py"), 20), (lost(), 200), (text("a.py sets x to 1."), 9)])
    assert agent.run("What does a.py set?") == "a.py sets x to 1."
    assert seen[2][-1] == {"role": "user", "content": "NOTE: your call was lost"}
    assert calls[0]["reply_tokens"] == 200 and calls[0]["count"] == 1
    assert calls[0]["finish"] == "stop" and "write_file" in calls[0]["tools"]


def test_a_truly_empty_reply_is_not_a_lost_reply(project: Path, monkeypatch) -> None:
    calls: list[dict] = []
    with_hook(monkeypatch, lambda agent, task, **context: calls.append(context) or "NOTE")
    agent, seen = agent_with_usage(project, [
        (call("read_file", path="a.py"), 20), (lost(), 1), (text("a.py sets x to 1."), 9)])
    assert agent.run("What does a.py set?") == "a.py sets x to 1."
    assert calls == [] and seen[2][-1]["content"] == EMPTY_FINAL


def test_a_hook_returning_none_keeps_the_old_handling(project: Path, monkeypatch) -> None:
    with_hook(monkeypatch, lambda agent, task, **context: None)
    agent, seen = agent_with_usage(project, [
        (call("read_file", path="a.py"), 20), (lost(), 300), (lost(), 300)])
    assert agent.run("What does a.py set?") == "(The model returned no text.)"
    assert len(seen) == 3


# -- debug log: the raw reply of a lost reply ----------------------------------------------------

@pytest.fixture
def debug_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEGA_DEBUG", "1")
    monkeypatch.setattr(debuglog, "_current", None)


def _entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class RawMessage(SimpleNamespace):
    def model_dump(self) -> dict:
        return dict(vars(self))


def test_a_lost_reply_logs_the_raw_reply(debug_on: None) -> None:
    log = debuglog.start("lost")
    message = RawMessage(role="assistant", content="", tool_calls=None, reasoning=None)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=2123, completion_tokens=200))
    inner = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: response)))
    tools = [{"type": "function", "function": {"name": "write_file"}}]
    debuglog.DebugClient(inner).chat.completions.create(
        model="m", messages=[{"role": "user", "content": "write it"}], tools=tools, max_tokens=2048)
    entry = next(e for e in _entries(log.path) if e["kind"] == "lost_reply")
    assert entry["reply_tokens"] == 200 and entry["finish"] == "stop"
    assert entry["tools"] == 1 and entry["max_tokens"] == 2048 and entry["prompt_chars"] == 8
    assert json.loads(entry["raw"])["content"] == "" and "reasoning" in json.loads(entry["raw"])


def test_a_streamed_lost_reply_logs_chunks_and_extra_fields(debug_on: None) -> None:
    log = debuglog.start("lost-stream")

    def chunk(finish=None, usage=None, **extra) -> SimpleNamespace:
        delta = SimpleNamespace(content="", tool_calls=None, model_extra=extra)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)],
                               usage=usage)

    stream = [chunk(), chunk(reasoning="hm"), chunk("stop"),
              SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=10,
                                                                completion_tokens=37))]
    inner = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: iter(stream))))
    list(debuglog.DebugClient(inner).chat.completions.create(model="m", messages=[], stream=True))
    entry = next(e for e in _entries(log.path) if e["kind"] == "lost_reply")
    assert entry["reply_tokens"] == 37 and entry["chunks"] == 4 and entry["stream"] is True
    assert entry["extras"] == {"reasoning": "hm"}


def test_a_reply_with_text_or_a_call_is_not_logged_as_lost(debug_on: None) -> None:
    log = debuglog.start("not-lost")
    for message in (RawMessage(content="done", tool_calls=None),
                    RawMessage(content=None, tool_calls=[SimpleNamespace(
                        function=SimpleNamespace(name="read_file", arguments="{}"))])):
        response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                                   usage=SimpleNamespace(prompt_tokens=5, completion_tokens=30))
        inner = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda response=response, **kwargs: response)))
        debuglog.DebugClient(inner).chat.completions.create(model="m", messages=[])
    assert not [e for e in _entries(log.path) if e["kind"] == "lost_reply"]


def test_the_loop_records_a_lost_reply_stage(debug_on: None, project: Path) -> None:
    log = debuglog.start("loop")
    agent, _ = agent_with_usage(project, [
        (call("read_file", path="a.py"), 20), (lost(), 200), (text("a.py sets x to 1."), 9)])
    agent.run("What does a.py set?")
    stage = next(e for e in _entries(log.path) if e.get("stage") == "lost_reply")
    assert stage["reply_tokens"] == 200 and stage["count"] == 1
