"""search_streak and re-reads, with the scripted fake model from test_agent_loop. The opt-in
repeat_unchanged was retired on 2026-10-17 (turn_ledger, a default, replaced it)."""

from __future__ import annotations

from pathlib import Path

import pytest
from switches import switch_off
from test_agent_loop import agent_with, call, text

from vega import features
from vega.agent import SEARCH_STREAK


@pytest.fixture(autouse=True)
def guards_on(monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "search_streak")


def tool_results(agent) -> list[str]:
    return [m["content"] for m in agent.messages if m.get("role") == "tool"]


def test_empty_search_streak_offers_the_file_list(tmp_path: Path, monkeypatch) -> None:
    # a question: the question pipeline would answer it; this tests the agent loop
    switch_off(monkeypatch, "question_pipeline")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "settings.py").write_text("CACHE = os.environ.get('APP_CACHE_HOME')\n")
    guesses = [call("search_text", query=q) for q in ("CACHETIER", "cache_folder", "CACHIER",
                                                       "cache_directory")]
    agent, _ = agent_with(tmp_path, [*guesses, call("read_file", path="src/settings.py"),
                                     text("APP_CACHE_HOME")])
    assert agent.run("Which environment variable controls the cache?") == "APP_CACHE_HOME"
    results = tool_results(agent)
    assert all("files" not in r for r in results[:SEARCH_STREAK - 1])
    assert "searches in a row found nothing" in results[SEARCH_STREAK - 1]
    assert "src/settings.py" in results[SEARCH_STREAK - 1]


def test_a_hit_resets_the_streak(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    replies = [call("search_text", query="nope1"), call("search_text", query="nope2"),
               call("search_text", query="nope3"), call("search_text", query="x = 1"),
               call("search_text", query="nope4"), text("done")]
    agent, _ = agent_with(tmp_path, replies)
    agent.run("find it")
    assert not any("found nothing" in r for r in tool_results(agent))


def test_rereading_a_file_always_runs(tmp_path: Path) -> None:
    (tmp_path / "w.py").write_text("x = 1\n")
    read = call("read_file", path="w.py")
    agent, _ = agent_with(tmp_path, [read, read, text("ok")])
    agent.run("look")
    assert all("x = 1" in r for r in tool_results(agent))


def test_guard_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv(features.ENV_ON)
    assert not features.on("search_streak")


def test_search_for_a_definition_that_does_not_exist_says_so(tmp_path: Path, monkeypatch) -> None:
    from vega.tools import execute

    monkeypatch.setenv(features.ENV_ON, "missing_symbol_hint")
    (tmp_path / "catalog.py").write_text("class Catalog:\n    def books(self):\n        return []\n")
    missing = execute("search_code", {"query": "def export(fmt, *, author=None)"}, tmp_path)
    assert missing.startswith("Note: nothing in this project defines export yet")
    present = execute("search_code", {"query": "class Catalog"}, tmp_path)
    assert not present.startswith("Note:")


def test_a_note_alone_does_not_count_as_looking_at_the_project(tmp_path: Path,
                                                                monkeypatch) -> None:
    switch_off(monkeypatch, "finish_gate_no_edit")  # the no-edit answer would go back once more
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, _ = agent_with(tmp_path, [call("remember", note="prompt shows y/N"),
                                     text("No change needed."),
                                     call("read_file", path="a.py"), text("x is 1, fine.")])
    assert agent.run("The prompt in a.py is missing its [y/N]; fix it.") == "x is 1, fine."
    results = [m["content"] for m in agent.messages if m.get("role") == "user"]
    assert any("tool" in r.lower() for r in results[1:])  # the act nudge was sent once
