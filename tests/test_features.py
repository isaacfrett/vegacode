"""Feature switches: each one removes exactly its behaviour, and unknown names are rejected."""

from __future__ import annotations

from pathlib import Path

import pytest

from vega import features, server
from vega.agent import (
    _INDEX_TEXT, _PLAN_LINE, _REMEMBER_TEXT, ACT_NUDGE, SYSTEM, system_prompt,
)
from vega.diagnostics import after_write
from vega.tools import tool_specs

from test_agent_loop import agent_with, text


def names(**kwargs) -> set[str]:
    return {spec["function"]["name"] for spec in tool_specs(**kwargs)}


def test_prompt_fragments_exist_in_full_prompt() -> None:
    # If the prompt is reworded these must be updated, or switching a feature off would do nothing.
    for fragment in (_PLAN_LINE, _REMEMBER_TEXT, _INDEX_TEXT):
        assert fragment in SYSTEM


def test_all_on_by_default(monkeypatch) -> None:
    monkeypatch.delenv(features.ENV, raising=False)
    assert features.disabled() == frozenset()
    # edit_by_name (on) recommends replace_definition in rule 3
    from vega.agent import _EDIT_BY_NAME_RULE, _EDIT_RULE

    assert system_prompt() == SYSTEM.replace(_EDIT_RULE, _EDIT_BY_NAME_RULE)


def test_unknown_feature_is_an_error(monkeypatch) -> None:
    monkeypatch.setenv(features.ENV, "lint,typo")
    with pytest.raises(ValueError, match="typo"):
        features.disabled()


def test_switching_off_plan_remember_index_trims_prompt(monkeypatch) -> None:
    monkeypatch.setenv(features.ENV, "plan,remember,code_index")
    prompt = system_prompt()
    assert "update_plan" not in prompt and "remember" not in prompt
    assert "repo_map" not in prompt and "search_code" not in prompt
    assert "search_text for an exact string" in prompt


def test_tool_specs_filtering() -> None:
    full = names()
    assert {"update_plan", "repo_map", "search_code"} <= full
    assert "update_plan" not in names(include_plan=False)
    assert not {"repo_map", "search_code"} & names(include_index=False)


def test_agent_offers_fewer_tools_when_switched_off(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV, "plan,remember,code_index")
    agent, seen = agent_with(tmp_path, [text("done")])
    offered: list[set[str]] = []
    original = agent.client.chat.completions.create

    def capture(**kwargs):
        offered.append({t["function"]["name"] for t in kwargs["tools"]})
        return original(**kwargs)

    agent.client.chat.completions.create = capture
    monkeypatch.setenv(features.ENV, "plan,remember,code_index,nudge_act")
    agent.run("hi")
    assert offered and not {"update_plan", "remember", "repo_map", "search_code"} & offered[0]


def test_nudge_act_switch(tmp_path: Path, monkeypatch) -> None:
    # clarify_model (default on) would spend a scripted reply on "Fix the bug" (an unsure verdict)
    monkeypatch.setenv(features.ENV, "nudge_act,clarify_model")
    agent, seen = agent_with(tmp_path, [text("Which file?")])
    assert agent.run("Fix the bug") == "Which file?"
    assert all(m.get("content") != ACT_NUDGE for m in seen[0])


def test_lint_switch(tmp_path: Path, monkeypatch) -> None:
    source = "import os\n"
    target = tmp_path / "m.py"
    monkeypatch.delenv(features.ENV, raising=False)
    with_lint = after_write(tmp_path, target, source)
    monkeypatch.setenv(features.ENV, "lint")
    without = after_write(tmp_path, target, source)
    assert "Syntax check: OK." in without and "Lint" not in without
    if "Lint" in with_lint:  # ruff installed
        assert with_lint != without


def test_kv_q8_switch_controls_server_env(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    class FakePopen:
        pid = 1

        def __init__(self, argv, env, **kwargs):
            captured.update(env)

        def poll(self):
            return 0

    monkeypatch.setattr(server, "is_running", lambda: False)
    monkeypatch.setattr(server, "STATE_DIR", tmp_path)
    monkeypatch.setattr(server, "PID_FILE", tmp_path / "pid")
    monkeypatch.setattr(server, "LOG_FILE", tmp_path / "log")
    monkeypatch.setattr(server.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr(server.subprocess, "Popen", FakePopen)
    for name in ("OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(features.ENV, "kv_q8")
    with pytest.raises(RuntimeError):
        server.start(timeout=0.01)
    assert "OLLAMA_KV_CACHE_TYPE" not in captured
    captured.clear()
    monkeypatch.delenv(features.ENV)
    with pytest.raises(RuntimeError):
        server.start(timeout=0.01)
    assert captured["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
