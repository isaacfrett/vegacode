"""The server's window matches Vega's (#0), from the context-efficiency review of 2026-10-28, and the
core's own compaction trigger and tool list at the hooks an extension may fill (compact_share,
tool_list, system_prompt, allow_command)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from test_agent_loop import agent_with, call, text
from typer.testing import CliRunner

from vega import cli, context, extensions, models, server
from vega.agent import SYSTEM, CodingAgent, _DELEGATE_TEXT, _REMEMBER_TEXT, system_prompt
from vega.tools import tool_specs

PROFILES = """
default = "granite-8b"

[profiles.granite-8b]
base = "granite4.1:8b"
num_ctx = 8192
features_off = ["plan"]

[profiles.remote]
base = "some-model"
endpoint = "http://127.0.0.1:1234/v1"
num_ctx = 8192
"""


@pytest.fixture
def profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "models.toml"
    path.write_text(PROFILES)
    monkeypatch.setattr(models, "PROFILES_FILE", path)
    monkeypatch.setattr(server, "STATE_DIR", tmp_path / "state")
    return path


def fake_server(monkeypatch: pytest.MonkeyPatch, windows: dict[str, int]) -> None:
    """Ollama's /api/show: the num_ctx each installed model was built with."""
    monkeypatch.setattr(server, "context_length",
                        lambda model=None: windows.get((model or server.model_name()).removesuffix(":latest")))


# -- the compaction trigger: the core's 65% (0.3.1), an extension's share through compact_share --

def test_core_compacts_at_65_percent_of_any_window() -> None:
    assert context.COMPACT_AT == 0.65
    for window in (4096, context.DEFAULT_NUM_CTX, 16384):
        assert context.compact_at(window) == 0.65
    budget = context.Budget(10000, prompt_tokens=7000, sent_count=1)
    assert budget.over([{"role": "system", "content": "x"}])  # 70% > 65%


def test_compact_share_hook_sets_the_trigger_and_bad_shares_are_ignored(monkeypatch) -> None:
    seen: list[dict] = []

    def share(value, **context_):
        seen.append(context_)
        return 0.75

    monkeypatch.setattr(extensions, "_impl", lambda name: share if name == "compact_share" else None)
    budget = context.Budget(10000, prompt_tokens=7000, sent_count=1)
    assert not budget.over([{"role": "system", "content": "x"}])  # 70% < 75%
    assert seen[-1] == {"num_ctx": 10000, "reply_tokens": None, "projected": 7000}
    for bad in (1.5, 0, "0.7", None):
        monkeypatch.setattr(extensions, "_impl",
                            lambda name, bad=bad: (lambda *a, **k: bad) if name == "compact_share" else None)
        assert context.compact_at(8192) == 0.65


# -- #0 the server's window -------------------------------------------------------------------------

def test_window_alias_names_and_profiles(profiles: Path) -> None:
    assert models.window_alias("vega", 16384) == "vega-granite-8b-16k"
    assert models.window_alias("vega-granite-8b", 32768) == "vega-granite-8b-32k"
    assert models.window_alias("vega-granite-8b-16k", 12000) == "vega-granite-8b-ctx12000"
    assert models.window_alias("vega", 8192) == "vega-granite-8b"  # the profile's own window
    assert models.window_alias("qwen2.5-coder:7b", 16384) is None  # not a profile's model
    assert models.window_alias("vega-remote", 16384) is None  # an endpoint: nothing to build
    alias = models.profile_for_model("vega-granite-8b-16k")
    assert alias is not None and alias.num_ctx == 16384 and alias.features_off == ["plan"]
    assert models.profile_for_model("vega-granite-8b").num_ctx == 8192


def test_build_window_alias_writes_the_same_profile_with_another_window(profiles: Path, monkeypatch) -> None:
    built: list[tuple[str, str]] = []
    monkeypatch.setattr(models, "build", lambda profile, target: built.append((target, profile.modelfile())) or 0)
    assert models.build_window_alias("granite-8b", 16384) == ("vega-granite-8b-16k", 0)
    _, modelfile = built[0]
    assert "FROM granite4.1:8b" in modelfile and "PARAMETER num_ctx 16384" in modelfile
    assert "8192" not in modelfile
    with pytest.raises(ValueError, match="endpoint|1234"):
        models.build_window_alias("remote", 16384)


def test_match_window_without_vega_num_ctx_is_the_servers(profiles: Path, monkeypatch) -> None:
    fake_server(monkeypatch, {"vega": 8192})
    check = context.match_window("vega")
    assert (check.model, check.error) == ("vega", "")


def test_match_window_uses_the_alias_or_refuses(profiles: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_NUM_CTX", "16384")
    fake_server(monkeypatch, {"vega": 8192})
    check = context.match_window("vega")
    assert check.error and check.model == "vega"
    assert "VEGA_NUM_CTX=16384" in check.error and "num_ctx 8192" in check.error
    assert "vega model build granite-8b --num-ctx 16384" in check.error and "fit-window" in check.error
    fake_server(monkeypatch, {"vega": 8192, "vega-granite-8b-16k": 16384})
    check = context.match_window("vega")
    assert (check.model, check.server, check.error) == ("vega-granite-8b-16k", 16384, "")
    # the same window, an unknown window and an endpoint all pass
    monkeypatch.setenv("VEGA_NUM_CTX", "8192")
    assert context.match_window("vega").model == "vega"
    fake_server(monkeypatch, {})
    monkeypatch.setenv("VEGA_NUM_CTX", "16384")
    assert context.match_window("vega").error == ""
    assert context.match_window("vega-remote").error == ""


def test_eval_refuses_a_window_the_server_does_not_run(profiles: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_NUM_CTX", "16384")
    fake_server(monkeypatch, {"vega": 8192})
    monkeypatch.setattr(cli, "_ensure_server", lambda *values: None)
    ran: list[str] = []
    monkeypatch.setattr(cli, "run_evaluations", lambda **kw: ran.append("eval"))
    result = CliRunner().invoke(cli.app, ["eval", "--suite", "fixes", "--case", "x"])
    assert result.exit_code == 2 and ran == []
    assert "VEGA_NUM_CTX=16384" in result.output and "fit-window" in result.output
    # with the alias on the server the eval runs it
    fake_server(monkeypatch, {"vega": 8192, "vega-granite-8b-16k": 16384})
    seen: list[str] = []
    monkeypatch.setattr(cli, "run_evaluations", lambda **kw: seen.append(server.model_name()))
    result = CliRunner().invoke(cli.app, ["eval", "--suite", "fixes", "--case", "x"])
    assert result.exit_code == 0, result.output
    assert seen == ["vega-granite-8b-16k"]


def test_fit_window_builds_the_alias_once(profiles: Path, monkeypatch) -> None:
    windows = {"vega": 8192}
    fake_server(monkeypatch, windows)
    monkeypatch.setattr(cli, "_ensure_server", lambda *values: None)
    built: list[str] = []

    def build(profile, target):
        built.append(target)
        windows[target] = profile.num_ctx  # what `ollama create` would serve
        return 0

    monkeypatch.setattr(models, "build", build)
    monkeypatch.setenv("VEGA_NUM_CTX", "16384")
    result = CliRunner().invoke(cli.app, ["model", "fit-window"])
    assert result.exit_code == 0, result.output
    assert built == ["vega-granite-8b-16k"] and "vega-granite-8b-16k" in result.output
    result = CliRunner().invoke(cli.app, ["model", "fit-window"])
    assert result.exit_code == 0 and built == ["vega-granite-8b-16k"]  # already there: no build
    result = CliRunner().invoke(cli.app, ["model", "build", "granite-8b", "--num-ctx", "32768"])
    assert result.exit_code == 0, result.output
    assert built[-1] == "vega-granite-8b-32k" and "VEGA_NUM_CTX=32768" in result.output


# -- #0 truncation tripwire -------------------------------------------------------------------------

def usage_reply(prompt_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=5),
                           choices=[SimpleNamespace(message=text("done"), finish_reason="stop")])


def test_truncated_is_window_minus_two() -> None:
    assert context.truncated(8190, 8192) and context.truncated(8191, 8192)
    assert not context.truncated(8189, 8192) and not context.truncated(None, 8192)


def test_truncation_watch_reports_cut_prompts_in_replies_and_streams() -> None:
    seen: list[tuple[int, int]] = []
    replies = iter([usage_reply(8191), usage_reply(4000)])
    inner = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: iter([SimpleNamespace(usage=None), SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=8190))]) if kw.get("stream") else next(replies))))
    watch = context.TruncationWatch(inner, lambda: 8192, lambda p, w: seen.append((p, w)))
    watch.chat.completions.create(model="vega", messages=[])
    watch.chat.completions.create(model="vega", messages=[])
    assert seen == [(8191, 8192)]
    chunks = list(watch.chat.completions.create(model="vega", messages=[], stream=True))
    assert len(chunks) == 2 and seen[-1] == (8190, 8192)


def test_agent_reports_a_truncated_prompt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(server, "context_length", lambda model=None: 8192)
    events: list[tuple[str, dict]] = []
    agent = CodingAgent(tmp_path, allow_commands=False, on_event=lambda k, d: events.append((k, d)))
    replies = iter([usage_reply(8191)])
    inner = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: next(replies))))
    agent.client = context.TruncationWatch(inner, agent.served_window, agent._prompt_truncated)
    agent.run("hi")
    assert agent.truncated_prompts == 1
    assert ("truncated_prompt", {"prompt_tokens": 8191, "window": 8192}) in events


def test_served_window_is_the_smaller_of_budget_and_server(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(server, "context_length", lambda model=None: 8192)
    monkeypatch.setenv("VEGA_NUM_CTX", "16384")
    assert CodingAgent(tmp_path).served_window() == 8192  # a mismatch past the guard still trips
    monkeypatch.setattr(server, "context_length", lambda model=None: None)
    assert CodingAgent(tmp_path).served_window() == 16384


def test_eval_meter_counts_truncated_prompts() -> None:
    from vega.evals import _Meter

    meter = _Meter()
    meter.on_event("truncated_prompt", {"prompt_tokens": 8191, "window": 8192})
    assert meter.truncated_prompts == 1


# -- the agent loop's tools: the core offers the full list (0.3.1); tool_list / system_prompt /
# allow_command are an extension's

def tool_names(tools: list[dict]) -> set[str]:
    return {spec["function"]["name"] for spec in tools}


def test_core_offers_every_tool_and_keeps_their_prompt_lines(tmp_path: Path) -> None:
    full = tool_specs()
    assert extensions.tool_list(full, "Fix the bug in pager.py") == full
    prompt = system_prompt()
    assert _REMEMBER_TEXT in SYSTEM and _DELEGATE_TEXT in SYSTEM
    assert "Use delegate_task only" in prompt and "Use remember" in prompt
    offered: list[set[str]] = []
    agent, _ = agent_with(tmp_path, [text("done")])
    original = agent.client.chat.completions.create

    def capture(**kwargs):
        offered.append(tool_names(kwargs["tools"]))
        return original(**kwargs)

    agent.client.chat.completions.create = capture
    agent.run("Fix the bug in pager.py")
    assert offered and {"git_status", "git_diff", "remember"} <= offered[0]


def test_tool_list_hook_may_leave_out_or_reword_never_add(monkeypatch) -> None:
    full = tool_specs()
    lean = [spec for spec in full if spec["function"]["name"] != "remember"]
    monkeypatch.setattr(extensions, "_impl", lambda name: (lambda tools, task, **k: lean)
                        if name == "tool_list" else None)
    assert extensions.tool_list(full, "t") == lean
    extra = [*full, {"type": "function", "function": {"name": "new_tool"}}]
    monkeypatch.setattr(extensions, "_impl", lambda name: (lambda tools, task, **k: extra)
                        if name == "tool_list" else None)
    assert extensions.tool_list(full, "t") == full


def test_disabled_run_command_stays_disabled_without_the_hook(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    assert extensions.allow_command("git status", tmp_path) is False
    agent, seen = agent_with(tmp_path, [call("run_command", command="git status --short"), text("done")])
    agent.run("What changed in the working tree? Check git status.")
    result = next(m for m in seen[-1] if m.get("role") == "tool")["content"]
    assert "disabled" in result
