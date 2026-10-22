"""Context budget: window discovery, tool-result caps, read_file ranges and compaction."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from switches import switch_off

from vega import context, server
from vega.agent import CodingAgent
from vega.context import SUMMARY_PREFIX, WorkLog, cap_tool_result, compact, group_units
from vega.server import context_length as real_context_length  # bound before conftest stubs it
from vega.tools import execute

_ids = itertools.count()


class FakeMessage(SimpleNamespace):
    def model_dump(self, exclude_none: bool = False) -> dict:
        data = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in self.tool_calls
            ]
        return data


def text(content: str) -> FakeMessage:
    return FakeMessage(content=content, tool_calls=None)


def call(name: str, **args) -> FakeMessage:
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    return FakeMessage(content=None, tool_calls=[SimpleNamespace(id=f"c{next(_ids)}", function=fn)])


def agent_with(project: Path, replies: list[tuple[FakeMessage, int | None]], **options):
    options.setdefault("planning", "on")  # these tests exercise planning
    """Agent whose fake server answers with (message, usage.prompt_tokens) pairs."""
    events: list[tuple[str, dict]] = []
    agent = CodingAgent(
        project, allow_commands=False, on_event=lambda k, d: events.append((k, d)), **options,
    )
    seen: list[list[dict]] = []

    def create(**kwargs):
        seen.append(json.loads(json.dumps(kwargs["messages"])))
        message, prompt_tokens = replies.pop(0)
        usage = SimpleNamespace(prompt_tokens=prompt_tokens) if prompt_tokens else None
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, seen, events


def assert_pairs_intact(messages: list[dict]) -> None:
    """Every tool message answers a call in the assistant message just before its tool block."""
    open_ids: set[str] = set()
    for message in messages:
        if message["role"] == "tool":
            assert message["tool_call_id"] in open_ids, f"orphaned tool result: {message}"
            open_ids.discard(message["tool_call_id"])
        else:
            assert not open_ids, f"tool calls without results: {open_ids}"
            open_ids = {c["id"] for c in message.get("tool_calls") or []}


# -- window discovery -----------------------------------------------------------------------------

def test_num_ctx_prefers_env_then_server_then_default(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []
    monkeypatch.setattr(server, "context_length", lambda model=None: asked.append(model) or 16384)
    monkeypatch.setenv("VEGA_NUM_CTX", "4096")
    assert context.resolve_num_ctx("vega") == (4096, "VEGA_NUM_CTX")
    assert asked == []
    monkeypatch.setenv("VEGA_NUM_CTX", "not-a-number")
    assert context.resolve_num_ctx("vega") == (16384, "server")
    monkeypatch.delenv("VEGA_NUM_CTX")
    monkeypatch.setattr(server, "context_length", lambda model=None: None)
    assert context.resolve_num_ctx("vega") == (8192, "default")


def test_num_ctx_is_discovered_once_per_agent(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(server, "context_length", lambda model=None: calls.append(model) or 12000)
    agent = CodingAgent(tmp_path, planning="on")
    assert calls == []  # nothing is asked until the window is needed
    assert agent.num_ctx == 12000 and agent.num_ctx == 12000
    assert calls == [agent.model] and agent.num_ctx_source == "server"


def test_server_context_length_reads_modelfile_parameter(monkeypatch) -> None:
    payloads: list[tuple[str, dict]] = []

    def fake_post(path: str, payload: dict):
        payloads.append((path, payload))
        return {"parameters": "temperature 0.2\nnum_ctx                        8192\nstop \"<|im_end|>\""}

    monkeypatch.setattr(server, "_post_json", fake_post)
    assert real_context_length("vega") == 8192
    assert payloads == [("/api/show", {"model": "vega"})]
    monkeypatch.setattr(server, "_post_json", lambda path, payload: {"parameters": "temperature 0.2"})
    assert real_context_length("vega") is None
    monkeypatch.setattr(server, "_post_json", lambda path, payload: None)
    assert real_context_length("vega") is None


# -- tool-result caps and read_file ranges --------------------------------------------------------

def test_read_file_range_keeps_line_numbers(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    result = execute("read_file", {"path": "a.py", "start_line": 3, "end_line": 4}, tmp_path)
    assert result == "   3\tline3\n   4\tline4\n[lines 3-4 of 10]"
    tail = execute("read_file", {"path": "a.py", "start_line": "9", "end_line": 50}, tmp_path)
    assert tail.startswith("   9\tline9\n  10\tline10\n")
    assert execute("read_file", {"path": "a.py", "start_line": 11}, tmp_path).startswith(
        "Invalid line range 11-end; the file has 10 lines")
    assert execute("read_file", {"path": "a.py", "start_line": 5, "end_line": 2}, tmp_path).startswith(
        "Invalid line range")


def test_read_file_cuts_long_files_at_a_line_with_a_note(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text("".join(f"x_{i} = {i}\n" for i in range(1, 1001)))
    result = execute("read_file", {"path": "big.py"}, tmp_path, max_result_chars=200)
    lines = result.split("\n")
    assert lines[0] == "   1\tx_1 = 1"
    assert lines[-1].startswith("[truncated: ") and "use read_file with start_line/end_line" in lines[-1]
    shown = len(lines) - 1
    assert f"{1000 - shown} more lines" in lines[-1] and f"showing 1-{shown} of 1000" in lines[-1]


def test_cap_tool_result_keeps_head_or_command_tail() -> None:
    long = "\n".join(f"row {i}" for i in range(500))
    capped = cap_tool_result("search_text", long, 300)
    assert len(capped) <= 300 and capped.startswith("row 0\n")
    assert capped.endswith("more lines]") and "read_file" not in capped
    output = "exit code: 1\n" + long + "\n1 failed, 3 passed"
    capped = cap_tool_result("run_check", output, 300)
    assert capped.startswith("exit code: 1\n[truncated: ") and capped.endswith("1 failed, 3 passed")
    assert cap_tool_result("read_file", "short", 300) == "short"


def test_agent_caps_tool_results_by_window(tmp_path: Path, monkeypatch) -> None:
    switch_off(monkeypatch, "question_pipeline")  # a question: this tests the loop's result cap
    # .log, not .txt: a large .txt is a data file and gets a preview instead (data_preview)
    (tmp_path / "big.log").write_text("".join(f"{i:05d} some text on this line\n" for i in range(3000)))
    agent, seen, _ = agent_with(tmp_path, [
        (call("read_file", path="big.log"), None), (text("It is long."), None),
    ], num_ctx=4000)
    agent.run("What is in big.log?")
    tool_message = seen[1][-1]
    assert tool_message["role"] == "tool"
    assert len(tool_message["content"]) <= context.tool_result_budget(4000) + 200
    assert "use read_file with start_line/end_line" in tool_message["content"]


def test_default_max_steps_is_100(tmp_path: Path) -> None:
    assert CodingAgent(tmp_path).max_steps == 100


# -- compaction (pure) ----------------------------------------------------------------------------

def _exchange(name: str, args: dict, result: str, call_id: str) -> list[dict]:
    return [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}},
        ]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _history() -> list[dict]:
    filler = "\n".join(f"   {i}\tvalue_{i} = compute({i})" for i in range(120))
    messages = [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "Earlier: explain the config loader."},
        {"role": "assistant", "content": "The loader reads settings.toml and merges env vars."},
        {"role": "user", "content": "Fix the rounding bug in calc.py and run the tests."},
        *_exchange("read_file", {"path": "calc.py"}, filler, "r1"),
        *_exchange("edit_file", {"path": "calc.py", "old_text": "a", "new_text": "b"},
                   "Edited calc.py (exact match).\nSyntax check: OK.", "e1"),
        {"role": "assistant", "content": "Rounding used int(); switched to round()."},
        *_exchange("run_check", {"command": "python -m pytest -q"},
                   "exit code: 1\nF.\n1 failed, 1 passed in 0.1s", "t1"),
        *_exchange("read_file", {"path": "tests/test_calc.py"}, filler, "r2"),
    ]
    # A final multi-result unit: two calls answered by two tool messages.
    messages.append({"role": "assistant", "content": None, "tool_calls": [
        {"id": "s1", "type": "function", "function": {"name": "search_text", "arguments": '{"query": "round"}'}},
        {"id": "s2", "type": "function", "function": {"name": "git_diff", "arguments": "{}"}},
    ]})
    messages.append({"role": "tool", "tool_call_id": "s1", "content": "calc.py:3: return round(x)"})
    messages.append({"role": "tool", "tool_call_id": "s2", "content": filler})
    return messages


def test_compaction_keeps_system_goal_and_recent_whole_units() -> None:
    messages = _history()
    result = compact(messages, 3, 2000, WorkLog())
    assert result is not None
    out = result.messages
    assert out[0] == messages[0]
    assert out[1]["role"] == "user" and out[1]["content"].startswith(SUMMARY_PREFIX)
    assert out[2] == messages[3]  # the current request survives verbatim
    assert out[-3:] == messages[-3:]  # the newest unit (one call, two results) is kept whole
    assert_pairs_intact(out)
    assert result.after_tokens < result.before_tokens


def test_compaction_never_splits_tool_calls_at_any_budget() -> None:
    messages = _history()
    for num_ctx in range(500, 20000, 250):
        result = compact(messages, 3, num_ctx, WorkLog())
        if result is not None:
            assert_pairs_intact(result.messages)
            assert result.messages[2] == messages[3]
    assert [len(unit) for unit in group_units(messages[4:])] == [2, 2, 1, 2, 2, 3]


def test_summary_lists_reads_edits_checks_and_findings() -> None:
    result = compact(_history(), 3, 4000, WorkLog())
    summary = result.messages[1]["content"]
    assert "Earlier: explain the config loader." in summary  # earlier request
    assert "Files read: calc.py" in summary
    assert "edit_file calc.py → Edited calc.py (exact match). Syntax OK." in summary
    assert "run_check `python -m pytest -q` → exit 1 (1 failed, 1 passed in 0.1s)" in summary
    assert "switched to round()" in summary and "merges env vars" in summary
    assert "Fix the rounding bug" not in summary  # the goal is kept as itself, not summarised


def test_second_compaction_keeps_earlier_facts() -> None:
    log = WorkLog()
    first = compact(_history(), 3, 2000, log).messages
    more = first + _exchange("edit_file", {"path": "util.py", "old_text": "x", "new_text": "y"},
                             "Edited util.py (exact match).", "e2")
    more += _exchange("read_file", {"path": "z.py"}, "x" * 6000, "r9")
    second = compact(more, 2, 2000, log).messages
    summary = second[1]["content"]
    assert sum(m["content"].startswith(SUMMARY_PREFIX) for m in second if m["role"] == "user") == 1
    assert "calc.py" in summary and "util.py" in summary
    assert_pairs_intact(second)


def test_nothing_to_drop_returns_none() -> None:
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    assert compact(messages, 1, 8192, WorkLog()) is None


# -- compaction in the agent loop -----------------------------------------------------------------

def test_agent_compacts_at_threshold_and_reinjects_plan(tmp_path: Path, monkeypatch) -> None:

    # a one-target task: routing would send it to the direct pipeline; this tests the agent loop
    switch_off(monkeypatch, "direct_pipeline")
    (tmp_path / "calc.py").write_text("".join(f"v{i} = {i}\n" for i in range(200)))
    plan = [
        {"text": "Read calc.py", "status": "done"},
        {"text": "Fix rounding", "status": "in_progress"},
        {"text": "Run tests", "status": "pending"},
    ]
    done = [dict(item, status="done") for item in plan]
    agent, seen, events = agent_with(tmp_path, [
        (call("update_plan", items=plan), 1000),
        (call("read_file", path="calc.py"), 1500),
        (call("read_file", path="calc.py", start_line=1, end_line=50), 2500),
        (call("edit_file", path="calc.py", old_text="v1 = 1\n", new_text="v1 = 1.0\n"), 9000),
        (call("update_plan", items=done), 3000),
        (text("Changed v1 in calc.py."), 3200),
    ], num_ctx=10000)
    assert agent.run("Fix the rounding in calc.py") == "Changed v1 in calc.py."
    # Requests 1-4 were under 65% of 10,000 tokens; the 9,000-token reply pushed request 5 over.
    assert [k for k, _ in events].count("notice") == 1
    notice = next(d["message"] for k, d in events if k == "notice")
    assert notice.startswith("Context compacted:") and "window 10,000" in notice
    assert all(m[1]["content"] == "Fix the rounding in calc.py" for m in seen[:4])
    compacted = seen[4]
    assert compacted[0]["content"].startswith("You are Vega")
    assert "Current plan" in compacted[-1]["content"] and "[>] Fix rounding" in compacted[-1]["content"]
    assert compacted[1]["content"].startswith(SUMMARY_PREFIX)
    assert "Files read: calc.py" in compacted[1]["content"]
    assert compacted[2] == {"role": "user", "content": "Fix the rounding in calc.py"}
    assert compacted[-2]["role"] == "tool" and compacted[-2]["content"].startswith("Edited calc.py")
    assert_pairs_intact(compacted)
    assert len(compacted) < len(seen[3]) + 2
    assert agent.messages[2] is not None and agent.messages[1]["content"].startswith(SUMMARY_PREFIX)


def test_estimate_is_used_when_server_reports_no_usage(tmp_path: Path) -> None:
    (tmp_path / "a.log").write_text("word " * 3000)  # not .txt: that would be previewed
    agent, seen, events = agent_with(tmp_path, [
        (call("read_file", path="a.log"), None),
        (call("read_file", path="a.log"), None),
        (call("read_file", path="a.log"), None),
        (text("Done reading."), None),
    ], num_ctx=6000)
    agent.run("Read a.log three times")
    assert any(k == "notice" for k, _ in events)
    for request in seen:
        assert context.estimate_tokens(request) < 6000
        assert_pairs_intact(request)
