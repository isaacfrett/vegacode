"""Agent-loop behaviour with a scripted fake model, so no server is needed."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from switches import switch_off

from vega.agent import ACT_NUDGE, MALFORMED_CALL, MAX_MALFORMED_RETRIES, UNSAVED_CLAIM, CodingAgent
from vega.tools import project_overview


class FakeMessage(SimpleNamespace):
    def model_dump(self, exclude_none: bool = False) -> dict:
        data = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in self.tool_calls
            ]
        return data


def text(content: str) -> FakeMessage:
    return FakeMessage(content=content, tool_calls=None)


def call(name: str, **args) -> FakeMessage:
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    return FakeMessage(content=None, tool_calls=[SimpleNamespace(id=f"c-{name}", function=fn)])


def agent_with(
    project: Path, replies: list[FakeMessage], planning: str = "auto",
) -> tuple[CodingAgent, list[list[dict]]]:
    agent = CodingAgent(project, allow_commands=False, planning=planning)
    seen: list[list[dict]] = []

    def create(**kwargs):
        seen.append(list(kwargs["messages"]))
        return SimpleNamespace(choices=[SimpleNamespace(message=replies.pop(0))])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, seen


def test_nudges_once_when_model_asks_instead_of_acting(tmp_path: Path) -> None:
    (tmp_path / "math_ops.py").write_text("def difference(a, b):\n    return a + b\n")
    agent, seen = agent_with(tmp_path, [
        text("Could you tell me more about difference()?"),
        call("edit_file", path="math_ops.py", old_text="a + b", new_text="a - b"),
        text("Fixed."),
    ])
    assert agent.run("Fix the bug in difference().") == "Fixed."
    assert (tmp_path / "math_ops.py").read_text().endswith("a - b\n")
    assert seen[1][-1] == {"role": "user", "content": ACT_NUDGE}


def test_nudge_is_not_repeated(tmp_path: Path, monkeypatch) -> None:
    switch_off(monkeypatch, "finish_gate_no_edit")  # the second no-edit answer would go back
    agent, seen = agent_with(tmp_path, [text("Done."), text("Done again.")])
    assert agent.run("fix the off-by-one in pager.py") == "Done again."
    assert len(seen) == 2


def test_conversation_is_answered_without_a_nudge(tmp_path: Path) -> None:
    # "what can you do" once got pushed into tools and the model then edited files unasked
    for task in ("hi", "what can you do", "what is a closure in python?"):
        agent, seen = agent_with(tmp_path, [text("Here is what I can do.")])
        assert agent.run(task) == "Here is what I can do."
        assert len(seen) == 1, task


def test_project_question_nudge_forbids_changes(tmp_path: Path, monkeypatch) -> None:
    from vega.agent import QUESTION_NUDGE

    # this question reached the agent loop only through the "port" misroute (fixed in cycle 3);
    # with the question pipeline off it still tests the agent loop's nudge
    switch_off(monkeypatch, "question_pipeline")
    agent, seen = agent_with(tmp_path, [text("Probably 8080."), text("8080, from settings.py.")])
    assert agent.run("Which port does the server in settings.py use?") == "8080, from settings.py."
    assert any(m.get("content") == QUESTION_NUDGE for m in seen[-1])
    assert "Do not change any files" in QUESTION_NUDGE


def test_no_nudge_after_tools_were_used(tmp_path: Path, monkeypatch) -> None:
    switch_off(monkeypatch, "question_pipeline")  # a question: this tests the agent loop's nudge
    (tmp_path / "a.txt").write_text("x")
    agent, seen = agent_with(tmp_path, [call("read_file", path="a.txt"), text("It contains x.")])
    assert agent.run("What is in a.txt?") == "It contains x."
    assert all(m.get("content") != ACT_NUDGE for m in seen[-1])


def test_system_prompt_includes_project_snapshot(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("")
    agent = CodingAgent(tmp_path)
    assert "src/app.py" in agent.messages[0]["content"]


def test_overview_respects_gitignore(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("build/\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.bin").write_text("")
    (tmp_path / "main.py").write_text("")
    overview = project_overview(tmp_path)
    assert "main.py" in overview and "out.bin" not in overview


def test_overview_caps_long_listings(tmp_path: Path) -> None:
    # Small projects get the full list (the eval fixtures' prompts stay unchanged) ...
    for i in range(12):
        (tmp_path / f"f{i}.py").write_text("")
    small = project_overview(tmp_path)
    assert "Files (12)" in small and "f11.py" in small and "does not look like a project" not in small
    # ... larger ones get a compact map of folders and root files instead of every path.
    for i in range(60):
        (tmp_path / "pkg").mkdir(exist_ok=True)
        (tmp_path / "pkg" / f"m{i}.py").write_text("")
    large = project_overview(tmp_path)
    assert "Files (72)" in large and "pkg/ (60 files)" in large and "pkg/m59.py" not in large
    assert "repo_map" in large


def test_overview_walk_is_capped_and_skips_heavy_folders(tmp_path: Path, monkeypatch) -> None:
    from vega import tools

    (tmp_path / "node_modules" / "dep").mkdir(parents=True)
    (tmp_path / "node_modules" / "dep" / "index.js").write_text("")
    for i in range(30):
        (tmp_path / f"f{i}.txt").write_text("")
    monkeypatch.setattr(tools, "OVERVIEW_WALK_LIMIT", 10)
    overview = project_overview(tmp_path)
    assert "Files (10+)" in overview and "does not look like a project" in overview
    assert "index.js" not in overview


# Captured from a real baseline failure: a text "tool call" whose arguments are a Python expression.
BROKEN_CALL = '{"name": "edit_file", "arguments": {"path": "greeting.py", "old_text": \'x\', "new_text": \'y\'.replace("a", "b")}}'


def test_malformed_text_tool_call_is_retried_not_returned(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    agent, seen = agent_with(tmp_path, [
        call("read_file", path="a.txt"),
        text(BROKEN_CALL),
        call("edit_file", path="a.txt", old_text="x", new_text="y"),
        text("Done."),
    ])
    assert agent.run("Change x to y in a.txt") == "Done."
    assert (tmp_path / "a.txt").read_text() == "y"
    assert seen[2][-1]["content"] == MALFORMED_CALL


def test_malformed_retries_are_bounded(tmp_path: Path, monkeypatch) -> None:
    switch_off(monkeypatch, "finish_gate_no_edit")  # no edit is made: the gate would add a round
    (tmp_path / "a.txt").write_text("x")
    replies = [call("read_file", path="a.txt")] + [text(BROKEN_CALL)] * (MAX_MALFORMED_RETRIES + 1)
    agent, seen = agent_with(tmp_path, replies)
    # names the file: a bare "Change x to y" is a short request the clarify gate asks the model
    # about (clarify_model), which would take one of the scripted replies
    assert agent.run("Change x to y in a.txt") == BROKEN_CALL
    assert len(seen) == MAX_MALFORMED_RETRIES + 2


def test_claimed_change_without_edit_is_sent_back(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, seen = agent_with(tmp_path, [
        call("read_file", path="a.py"),
        text("The bug has been fixed in a.py."),
        call("edit_file", path="a.py", old_text="x = 1", new_text="x = 2"),
        text("I fixed it."),
    ])
    assert agent.run("Set x to 2") == "I fixed it."
    assert seen[2][-1]["content"] == UNSAVED_CLAIM
    assert (tmp_path / "a.py").read_text() == "x = 2\n"


def test_readonly_answer_is_not_treated_as_change_claim(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, seen = agent_with(tmp_path, [call("read_file", path="a.py"), text("x is 1. No files were modified.")])
    assert agent.run("What is x?") == "x is 1. No files were modified."
    assert len(seen) == 2


def test_unparseable_file_is_sent_back_once(tmp_path: Path, monkeypatch) -> None:
    # nudge_parse sees a broken file only when atomic_writes is off (it refuses the write);
    # the refusal is tested in tests/test_edit_transactions.py
    switch_off(monkeypatch, "atomic_writes")
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, seen = agent_with(tmp_path, [
        call("edit_file", path="a.py", old_text="x = 1", new_text="x = (1"),
        text("I updated a.py."),
        call("edit_file", path="./a.py", old_text="x = (1", new_text="x = (1)"),
        text("I updated a.py."),
    ])
    assert agent.run("wrap 1 in parens") == "I updated a.py."
    assert seen[2][-1]["content"].startswith("(Vega harness) These files still fail to parse")
    assert (tmp_path / "a.py").read_text() == "x = (1)\n"


def agent_with_reasons(project: Path, replies: list[tuple[FakeMessage, str | None]]):
    """Like agent_with, but each reply carries a finish_reason (e.g. "length" for a cut-off)."""
    agent = CodingAgent(project, allow_commands=False)
    seen: list[dict] = []

    def create(**kwargs):
        seen.append(kwargs)
        message, reason = replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=reason)])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, seen


def test_requests_carry_a_reply_token_cap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("VEGA_MAX_TOKENS", raising=False)
    agent, seen = agent_with_reasons(tmp_path, [(call("list_files"), "tool_calls"), (text("done"), "stop")])
    agent.run("list")
    assert all(request["max_tokens"] == 2048 for request in seen)
    monkeypatch.setenv("VEGA_MAX_TOKENS", "4096")
    assert CodingAgent(tmp_path).sampling["max_tokens"] == 4096


def test_cut_off_reply_is_sent_back_once_then_recovers(tmp_path: Path) -> None:
    from vega.agent import CUT_OFF

    runaway = text("x x x " * 50)
    agent, seen = agent_with_reasons(tmp_path, [
        (call("list_files"), "tool_calls"), (runaway, "length"), (text("All done."), "stop"),
    ])
    assert agent.run("do it") == "All done."
    contents = [m.get("content") for m in agent.messages]
    assert CUT_OFF in contents and contents[-1] == "All done."
    assert len(seen) == 3


def test_repeated_cut_off_stops_cleanly(tmp_path: Path) -> None:
    from vega.agent import CUT_OFF_ANSWER

    runaway = text("y " * 50)
    agent, seen = agent_with_reasons(tmp_path, [
        (call("list_files"), "tool_calls"), (runaway, "length"), (runaway, "length"),
    ])
    assert agent.run("do it") == CUT_OFF_ANSWER
    assert len(seen) == 3


def test_cut_off_with_a_complete_tool_call_still_runs_the_tool(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hi")
    agent, seen = agent_with_reasons(tmp_path, [
        (call("read_file", path="a.txt"), "length"), (text("It says hi."), "stop"),
    ])
    assert agent.run("read a.txt") == "It says hi."


def test_function_style_text_calls_are_parsed() -> None:
    from vega.agent import _parse_text_tool_calls

    # Captured from a 7B trace: the plan written as `update_plan[...]` text, then accepted as the answer.
    plan = 'update_plan[\n  {"text":"Replace CURRENCY in billing/config.py", "status":"in_progress"}\n]'
    assert _parse_text_tool_calls(plan) == [
        ("update_plan", {"items": [{"text": "Replace CURRENCY in billing/config.py", "status": "in_progress"}]})
    ]
    assert _parse_text_tool_calls('read_file({"path": "a.py"})') == [("read_file", {"path": "a.py"})]
    assert _parse_text_tool_calls('remember("tests: pytest -q")') == [("remember", {"note": "tests: pytest -q"})]


def test_function_style_mention_in_prose_is_not_run() -> None:
    from vega.agent import _parse_text_tool_calls, looks_like_failed_tool_call

    prose = 'Done. I changed it with edit_file{"path": "a.py", "old_text": "x", "new_text": "y"}.'
    assert _parse_text_tool_calls(prose) == [] and not looks_like_failed_tool_call(prose)


def test_unparseable_function_style_call_is_sent_back(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    agent, seen = agent_with(tmp_path, [
        call("read_file", path="a.txt"),
        text("update_plan[ not json ]"),
        call("edit_file", path="a.txt", old_text="x", new_text="y"),
        text("Done."),
    ])
    assert agent.run("change x to y") == "Done."
    assert (tmp_path / "a.txt").read_text() == "y"


def test_command_style_text_calls_are_parsed() -> None:
    from vega.agent import _parse_text_tool_calls

    # Captured from a 7B trace: a search written like a shell command, then accepted as the answer.
    assert _parse_text_tool_calls("search_text Currency setting in config files") == [
        ("search_text", {"query": "Currency setting in config files"})
    ]
    assert _parse_text_tool_calls("read_file src/billing/config.py") == [
        ("read_file", {"path": "src/billing/config.py"})
    ]
    # Tools needing several arguments, prose, and multi-line replies are left alone.
    for text in ("edit_file src/a.py", "I will search_text for it", "search_text foo\nthen edit"):
        assert _parse_text_tool_calls(text) == []


def test_plan_written_as_text_is_recorded_and_the_model_moves_on(tmp_path: Path) -> None:
    from vega.agent import PLAN_RECORDED

    (tmp_path / "a.txt").write_text("x")
    agent, seen = agent_with(tmp_path, [
        text("Plan:\n1. Find the file.\n2. Change x to y.\n3. Check."),
        call("edit_file", path="a.txt", old_text="x", new_text="y"),
        call("update_plan", items=[{"text": "Find the file.", "status": "done"},
                                   {"text": "Change x to y.", "status": "done"},
                                   {"text": "Check.", "status": "done"}]),
        text("Done."),
    ], planning="on")
    assert agent.run("change x to y in a.txt") == "Done."  # named: clarify_first has no question
    assert PLAN_RECORDED in [m.get("content") for m in agent.messages]
    assert (tmp_path / "a.txt").read_text() == "y"


def test_numbered_summary_after_an_edit_is_not_taken_as_a_plan(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    summary = "Plan carried out:\n1. Found a.txt.\n2. Changed x to y."
    agent, seen = agent_with(tmp_path, [
        call("edit_file", path="a.txt", old_text="x", new_text="y"), text(summary),
    ])
    assert agent.run("change x to y in a.txt") == summary  # named: clarify_first has no question


def test_strip_harness_echo() -> None:
    from vega.agent import strip_harness_echo

    echo = ("(Vega harness) Summary of earlier work: Files read: a.py\nEdits: none\n\n"
            "x is defined in a.py.")
    assert strip_harness_echo(echo) == ("x is defined in a.py.", True)
    assert strip_harness_echo("Plain answer.") == ("Plain answer.", False)
    assert strip_harness_echo("> (Vega harness) You replied without inspecting")[0] == ""


def test_pure_echo_is_sent_back_and_partial_echo_is_cleaned(tmp_path: Path) -> None:
    from vega.agent import HARNESS_ECHO

    (tmp_path / "a.py").write_text("x = 1\n")
    agent, seen = agent_with(tmp_path, [
        call("read_file", path="a.py"),
        text("(Vega harness) You replied without inspecting the project."),
        text("(Vega harness) Summary of earlier work: read a.py\n\nx is 1."),
    ])
    assert agent.run("what is x?") == "x is 1."
    assert HARNESS_ECHO in [m.get("content") for m in agent.messages]


def test_asking_for_code_after_a_search_is_sent_back(tmp_path: Path) -> None:
    from vega.agent import ASK_FOR_CONTENT, asks_for_project_content

    # Captured from a real chat: after one search the 7B asked the user to paste the function.
    assert asks_for_project_content("Please provide the code inside the `add()` function so I can fix it.")
    assert not asks_for_project_content("I fixed add() in src/calc.py; the tests pass.")
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    agent, _ = agent_with(tmp_path, [
        call("search_code", query="add"),
        text("Please provide the code inside the add() function so I can identify the bug."),
        call("edit_file", path="calc.py", old_text="a - b", new_text="a + b"),
        text("Fixed add() in calc.py."),
    ])
    assert agent.run("Fix the bug in add()") == "Fixed add() in calc.py."
    assert ASK_FOR_CONTENT in [m.get("content") for m in agent.messages]
    assert (tmp_path / "calc.py").read_text().endswith("a + b\n")


def test_repeated_failing_call_is_not_run_again(tmp_path: Path, monkeypatch) -> None:
    from vega.tools import execute

    # "fix the boundary" is a short request naming nothing: clarify_model would ask the scripted
    # model first; naming g.py instead would route it to the direct pipeline
    switch_off(monkeypatch, "clarify_first")

    (tmp_path / "g.py").write_text('def grade(s):\n    if s > 90:\n        return "A"\n')
    bad = dict(path="g.py", old_text="if s > 95:", new_text="if s >= 90:")
    agent, _ = agent_with(tmp_path, [
        call("edit_file", **bad), call("edit_file", **bad),
        call("edit_file", path="g.py", old_text="if s > 90:", new_text="if s >= 90:"),
        text("Fixed the boundary."),
    ])
    assert agent.run("fix the boundary") == "Fixed the boundary."
    tool_results = [m["content"] for m in agent.messages if m.get("role") == "tool"]
    assert tool_results[0].startswith("Edit not applied")
    assert tool_results[1].startswith("(Vega harness) Not run: you already made exactly this call")
    assert "if s >= 90:" in (tmp_path / "g.py").read_text()


def test_empty_old_text_appends_and_quote_style_is_tolerated(tmp_path: Path) -> None:
    from vega.tools import execute

    (tmp_path / "u.py").write_text("def shout(t):\n    return t.upper()\n")
    result = execute("edit_file", {"path": "u.py", "old_text": "", "new_text": "def whisper(t):\n    return t.lower()\n"}, tmp_path)
    assert "appended at the end" in result
    assert (tmp_path / "u.py").read_text() == "def shout(t):\n    return t.upper()\n\ndef whisper(t):\n    return t.lower()\n"
    (tmp_path / "g.py").write_text('def grade(s):\n    if s > 90:\n        return "A"\n')
    result = execute("edit_file", {"path": "g.py", "old_text": "    if s > 90:\n        return 'A'",
                                   "new_text": "    if s >= 90:\n        return 'A'"}, tmp_path)
    assert "quote style" in result and "if s >= 90:" in (tmp_path / "g.py").read_text()


def test_capabilities_answer_that_mentions_sharing_files_is_kept(tmp_path: Path) -> None:
    # Live session: the answer to "what can you do" ended "share the file you want changed" and was
    # sent back as "asked the user for project content"; the retry wandered for two minutes.
    reply = "I can read, search and edit files and run tests. Share the file you want changed."
    agent, seen = agent_with(tmp_path, [text(reply)])
    assert agent.run("what can you do") == reply
    assert len(seen) == 1


def test_conversation_requests_ask_for_a_brief_reply(tmp_path: Path) -> None:
    from vega.agent import BRIEF_REPLY

    agent, seen = agent_with(tmp_path, [text("I read, search, edit and test code.")])
    agent.run("what can you do")
    assert seen[0][-1]["content"] == BRIEF_REPLY
    assert all(m.get("content") != BRIEF_REPLY for m in agent.messages)  # not kept in history
    agent2, seen2 = agent_with(tmp_path, [text("Done.")])
    agent2.run("fix the off-by-one in pager.py")
    assert all(m.get("content") != BRIEF_REPLY for m in seen2[0])
