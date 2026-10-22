"""Chat session transcripts and per-request timing."""

from __future__ import annotations

import json
from pathlib import Path

from switches import switch_off

from vega import server
from vega.transcripts import SessionLog, compact

from test_agent_loop import agent_with, call, text


def test_agent_records_timing_per_request(tmp_path: Path, monkeypatch) -> None:
    # a question: the question pipeline would answer it; this tests the agent loop
    switch_off(monkeypatch, "question_pipeline")
    (tmp_path / "a.txt").write_text("hi")
    agent, _ = agent_with(tmp_path, [call("read_file", path="a.txt"), text("It says hi.")])
    agent.run("what is in a.txt?")
    assert len(agent.request_log) == 2
    assert all(entry["seconds"] >= 0 for entry in agent.request_log)


def test_session_log_saves_turns_requests_and_messages(tmp_path: Path, monkeypatch) -> None:
    # a question: the question pipeline would answer it; this tests the agent loop
    switch_off(monkeypatch, "question_pipeline")
    (tmp_path / "a.txt").write_text("hi")
    agent, _ = agent_with(tmp_path, [call("read_file", path="a.txt"), text("It says hi.")])
    log = SessionLog(server.STATE_DIR / "sessions", tmp_path)
    answer = agent.run("what is in a.txt?")
    log.record_turn("what is in a.txt?", answer, 1.5, list(agent.request_log))
    log.save(agent.messages)
    saved = json.loads(log.path.read_text())
    assert log.path.parent == server.STATE_DIR / "sessions"  # isolated by tests/conftest.py
    assert saved["turns"][0]["answer"] == "It says hi." and len(saved["turns"][0]["requests"]) == 2
    roles = [m["role"] for m in saved["messages"]]
    assert roles[0] == "system" and "tool" in roles


def test_compact_truncates_long_content() -> None:
    entries = compact([{"role": "system", "content": "s"},
                       {"role": "tool", "content": "x" * 5000}])
    assert len(entries) == 1 and len(entries[0]["text"]) == 1200


def test_warm_up_sends_fixed_prompt_with_one_token(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from vega.agent import CodingAgent

    sent: list[dict] = []
    agent = CodingAgent(tmp_path)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: sent.append(kw))))
    assert agent.warm_up() is True
    assert sent[0]["max_tokens"] == 1 and sent[0]["messages"][0] == agent.messages[0]
    assert sent[0]["tools"]  # same tool definitions as a first turn, so the prefix matches

    def broken(**kw):
        raise ConnectionError("server down")

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=broken)))
    assert agent.warm_up() is False  # never raises
