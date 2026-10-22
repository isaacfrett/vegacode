"""clarify_first (ask before building a vague request) and turn_ledger (searches survive
compaction), with scripted fake models only.

The motivating chat: "Could you build an LLM workflow, verify it end to end, and setup a loop
around watching and measuring it for data analysis." got no question; the model searched
"workflow", "LLM workflow" and "data analysis loop", and after compaction ran the same searches
again.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_agent_loop import agent_with, call, text

from vega import clarify, context, features
from vega.agent import LEDGER_STUCK, CodingAgent

LIVE = ("Could you build an LLM workflow, verify it end to end, and setup a loop around watching "
        "and measuring it for data analysis.")


@pytest.fixture
def gate_on(monkeypatch) -> None:
    # finish_gate_no_edit off: the scripted change requests finish with no edit, which that
    # gate would send back once (tests/test_no_edit_gate.py covers it)
    monkeypatch.setenv(features.ENV, "question_pipeline,project_pipeline,finish_gate_no_edit")


def no_model_agent(project: Path) -> tuple[CodingAgent, list]:
    calls: list = []

    def create(**kwargs):
        calls.append(kwargs)
        raise AssertionError("the clarify gate must not call the model for a vague request")

    agent = CodingAgent(project, allow_commands=False)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, calls


def sample_project(root: Path) -> Path:
    (root / "src" / "app").mkdir(parents=True)
    (root / "src" / "app" / "loader.py").write_text("def load(path):\n    return open(path).read()\n")
    (root / "src" / "app" / "report.py").write_text("def report(rows):\n    return len(rows)\n")
    (root / "data").mkdir()
    (root / "data" / "events.csv").write_text("id,value\n1,2\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_loader.py").write_text("def test_ok():\n    assert True\n")
    return root


# -- the rules ------------------------------------------------------------------------------------

def test_live_request_and_open_ended_builds_are_vague() -> None:
    for task in (LIVE, "build an application", "make me a dashboard", "Can you build me a game?",
                 "Make it faster."):
        assert clarify.classify(task).kind == clarify.VAGUE, task


def test_questions_conversation_and_anchored_requests_are_clear() -> None:
    for task in ("what can you do", "hi", "do you know python", "Which file defines Inventory?",
                 "Fix the bug in difference().", "Add `export_csv` to todo/store.py",
                 "Set up a GitHub Actions workflow that runs pytest on push",
                 "Change the default API port from 8080 to 3000 in the application."):
        assert clarify.classify(task).kind == clarify.CLEAR, task


def test_question_about_a_function_named_like_a_verb_is_clear() -> None:
    # "add" is the function's name here, not a change: no clarify model call (cycle 3)
    for task in ("Why is add wrong?", "what does add return?", "What does `add()` return?",
                 "Why does update fail on empty lists?", "Where is delete defined?",
                 "What port does the server use?"):
        verdict = clarify.classify(task)
        assert verdict.kind == clarify.CLEAR, task
        assert verdict.reasons[0] in {"no build or change verb", "question"}, task
    # the change counterparts still count as change requests
    assert clarify.classify("Add a function that sums a list.").kind == clarify.UNSURE
    assert clarify.classify("Update the remove function.").kind == clarify.UNSURE


def test_pronoun_request_in_a_conversation_is_not_asked() -> None:
    assert clarify.classify("Make it faster.", follow_up=True).kind == clarify.CLEAR


def test_short_unanchored_change_is_left_to_the_model_check() -> None:
    assert clarify.classify("Add low-stock alerts to the inventory.").kind == clarify.UNSURE


# -- the gate in the agent ------------------------------------------------------------------------

def test_vague_request_gets_one_clarifying_reply_and_no_tools(tmp_path: Path, gate_on) -> None:
    project = sample_project(tmp_path)
    before = sorted(p.as_posix() for p in project.rglob("*"))
    agent, calls = no_model_agent(project)
    events: list = []
    agent.on_event = lambda kind, data: events.append((kind, data))
    reply = agent.run(LIVE)
    assert calls == []  # no model call, so no tool call
    assert not any(kind == "tool_call" for kind, _ in events)
    assert sorted(p.as_posix() for p in project.rglob("*")) == before
    numbered = [line for line in reply.splitlines() if line[:2] in {"1.", "2.", "3.", "4."}]
    assert 2 <= len(numbered) <= 4
    assert reply.rstrip().endswith("?")
    assert "data/events.csv" in reply  # options come from the project
    assert "python -m pytest -q" in reply
    assert "loop" in reply.lower()
    assert agent.messages[-1] == {"role": "assistant", "content": reply}


def test_answer_to_the_questions_goes_to_the_agent_with_the_request(tmp_path: Path, gate_on) -> None:
    project = sample_project(tmp_path)
    agent, seen = agent_with(project, [call("list_files"), text("Here is the plan.")])
    first = clarify.run(agent, "build an application")
    assert first is not None and agent._clarify_pending == "build an application"
    assert agent.run("1a, 2b, 3a") == "Here is the plan."
    goal = seen[0][-1]["content"]
    assert goal.startswith("build an application") and goal.endswith("1a, 2b, 3a")
    assert agent._clarify_pending is None


def test_clear_request_is_not_asked_and_not_slowed(tmp_path: Path, gate_on) -> None:
    (tmp_path / "math_ops.py").write_text("def difference(a, b):\n    return a + b\n")
    agent, seen = agent_with(tmp_path, [
        call("edit_file", path="math_ops.py", old_text="a + b", new_text="a - b"),
        text("Fixed."),
    ])
    assert agent.run("Fix the bug in difference().") == "Fixed."
    assert len(seen) == 2  # no extra model call
    assert seen[0][-1] == {"role": "user", "content": "Fix the bug in difference()."}


def test_unsure_request_runs_on_with_the_model_check_off(tmp_path: Path, gate_on,
                                                         monkeypatch) -> None:
    monkeypatch.setenv(features.ENV,
                       "question_pipeline,project_pipeline,clarify_model,finish_gate_no_edit")
    (tmp_path / "inventory.py").write_text("STOCK = {}\n")
    agent, seen = agent_with(tmp_path, [call("read_file", path="inventory.py"), text("Done.")])
    assert agent.run("Add low-stock alerts to the inventory.") == "Done."
    assert seen[0][-1]["content"] == "Add low-stock alerts to the inventory."


def model_check_agent(project: Path, verdict: dict, after: list) -> tuple[CodingAgent, list]:
    seen: list = []
    replies = [json.dumps(verdict), *after]

    def create(**kwargs):
        seen.append(kwargs)
        reply = replies.pop(0)
        message = text(reply) if isinstance(reply, str) else reply
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=None)

    agent = CodingAgent(project, allow_commands=False)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, seen


def test_model_check_asks_when_the_project_does_not_settle_it(tmp_path: Path, gate_on,
                                                                monkeypatch) -> None:
    (tmp_path / "README.md").write_text("# spendbook\nAn expense ledger.\n")
    verdict = {"specific": False, "missing": "export format",
               "questions": [{"question": "Which file format should the export use?",
                              "options": ["CSV", "JSON", "Excel"]}]}
    agent, seen = model_check_agent(tmp_path, verdict, [])
    reply = agent.run("Add a way to export the expenses to a file.")
    assert len(seen) == 1 and seen[0]["response_format"]["type"] == "json_schema"
    assert "README" in seen[0]["messages"][1]["content"] and "spendbook" in seen[0]["messages"][1]["content"]
    assert "1. Which file format should the export use?" in reply and "a) CSV" in reply
    assert reply.rstrip().endswith("?")


def test_model_check_lets_a_settled_request_through(tmp_path: Path, gate_on, monkeypatch) -> None:
    (tmp_path / "README.md").write_text("Every export is a CSV file.\n")
    agent, seen = model_check_agent(tmp_path, {"specific": True, "questions": []},
                                    [call("read_file", path="README.md"), "Added the CSV export."])
    assert agent.run("Add a way to export the expenses to a file.") == "Added the CSV export."
    assert len(seen) == 3


def test_model_check_failure_never_blocks_the_turn(tmp_path: Path, gate_on, monkeypatch) -> None:
    agent, _ = model_check_agent(tmp_path, {"specific": False}, ["not json at all"])
    agent_reply = clarify.model_check(agent, "Add low-stock alerts.")
    assert agent_reply is None  # no questions given: go on


def test_model_check_is_on_by_default(tmp_path: Path, gate_on) -> None:
    """With the check off an "unsure" request was built on a guess, so it is on by default."""
    assert features.on("clarify_model")
    verdict = {"specific": False, "missing": "low-stock threshold",
               "questions": [{"question": "Below what quantity is stock low?",
                              "options": ["5", "10", "a per-item threshold"]}]}
    (tmp_path / "inventory.py").write_text("STOCK = {}\n")
    agent, seen = model_check_agent(tmp_path, verdict, [])
    reply = agent.run("Add low-stock alerts to the inventory.")
    assert len(seen) == 1 and "1. Below what quantity is stock low?" in reply
    assert not any(p.suffix == ".py" and p.name != "inventory.py" for p in tmp_path.iterdir())


def test_model_check_sees_the_loaded_project_context(tmp_path: Path, gate_on) -> None:
    """"The timestamp bug we found last session" is settled by the session digest the agent
    loaded, not by the README; the check is shown that context."""
    agent, seen = model_check_agent(tmp_path, {"specific": True, "questions": []}, [])
    agent.context_section = "\n\n## Last sessions\n- Open: the Z-suffix bug in parse_stamp"
    assert clarify.model_check(agent, "Please fix the timestamp bug we found last session.") is None
    shown = seen[0]["messages"][1]["content"]
    assert "Project context from earlier sessions:" in shown and "Z-suffix bug" in shown
    assert "project context from earlier sessions" in seen[0]["messages"][0]["content"]


def test_gate_is_off_when_switched_off(tmp_path: Path, monkeypatch) -> None:
    # finish_gate_no_edit off: "ok" with no edit to a build request would be sent back once
    monkeypatch.setenv(features.ENV, "clarify_first,finish_gate_no_edit")
    agent, _ = agent_with(tmp_path, [call("list_files"), text("ok")])
    assert agent.run("build an application") == "ok"


# -- the turn ledger: searches survive compaction ---------------------------------------------------

def compaction_agent(project: Path, replies: list) -> tuple[CodingAgent, list, list]:
    events: list = []
    agent = CodingAgent(project, allow_commands=False, num_ctx=6000,
                        on_event=lambda kind, data: events.append((kind, data)))
    seen: list = []

    def create(**kwargs):
        seen.append(json.loads(json.dumps(kwargs["messages"])))
        return SimpleNamespace(choices=[SimpleNamespace(message=replies.pop(0))], usage=None)

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, seen, events


def tool_results(agent: CodingAgent) -> list[str]:
    return [m["content"] for m in agent.messages if m.get("role") == "tool"]


def test_repeated_search_after_compaction_gets_the_ledger_line(tmp_path: Path) -> None:
    (tmp_path / "flows.py").write_text("def workflow():\n    return 1\n")
    (tmp_path / "big.log").write_text("word " * 3000)  # not .txt: that would be previewed
    search = call("search_text", query="workflow")
    agent, seen, events = compaction_agent(tmp_path, [
        search,
        call("read_file", path="big.log"),
        call("read_file", path="big.log"),
        call("read_file", path="big.log"),
        search,  # after compaction: the ledger answers, the search does not run again
        search,  # the second repeat: change approach
        text("Stopping to ask."),
    ])
    assert agent.run("Look for the workflow code") == "Stopping to ask."
    assert any(kind == "notice" and "compacted" in data["message"] for kind, data in events)
    # the request before the repeat no longer holds the first search's result...
    repeat_request = seen[4]
    assert not any(m["role"] == "tool" and m["content"].startswith("flows.py:1")
                   for m in repeat_request)
    # ...but the summary lists it, with its outcome
    summary = next(m["content"] for m in repeat_request
                   if m["role"] == "user" and m["content"].startswith(context.SUMMARY_PREFIX))
    assert context.LEDGER_TITLE in summary
    assert "search_text(query='workflow') → 1 result lines; top hit flows.py:1" in summary
    tool_events = [data for kind, data in events if kind == "tool_result"]
    first_repeat, second_repeat = tool_events[4]["result"], tool_events[5]["result"]
    assert first_repeat.startswith("(Vega harness) Not run again")
    assert "top hit flows.py:1" in first_repeat
    assert "Change approach" not in first_repeat
    assert LEDGER_STUCK == 2 and "Change approach" in second_repeat
    assert "ask the user" in second_repeat


def test_a_write_makes_the_same_search_run_again(tmp_path: Path) -> None:
    (tmp_path / "w.py").write_text("x = 1\n")
    search = call("search_text", query="x = ")
    agent, _ = agent_with(tmp_path, [
        search,
        call("edit_file", path="w.py", old_text="x = 1", new_text="x = 2"),
        search,
        text("ok"),
    ])
    agent.run("set x to 2 in w.py")
    results = tool_results(agent)
    assert results[2].startswith("w.py:1: x = 2")


def test_read_file_is_never_answered_from_the_ledger(tmp_path: Path) -> None:
    (tmp_path / "w.py").write_text("x = 1\n")
    read = call("read_file", path="w.py")
    agent, _ = agent_with(tmp_path, [read, read, text("ok")])
    agent.run("look at w.py")
    assert all("x = 1" in r for r in tool_results(agent))


def test_ledger_outcomes() -> None:
    from vega.agent import ledger_outcome

    assert ledger_outcome("search_text", "No text matches for 'x'.") == "no matches"
    hit = ledger_outcome("search_code", "pkg/jobs.py:360-399 (11.74)\n  361: def f():")
    assert hit == "2 result lines; top hit pkg/jobs.py:360-399"
    assert ledger_outcome("run_check", "Check timed out after 60s.") == "Check timed out after 60s."
