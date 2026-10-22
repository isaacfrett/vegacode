"""Terminal UX: streaming and progress events, /undo snapshots and review mode (no server needed)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from switches import switch_off

from vega.agent import ACT_NUDGE, DECLINED, QUESTION_NUDGE, CodingAgent, accumulate_tool_call
from vega.tools import WritePlan, plan_write
from vega.ui import TerminalRenderer, ask_approval, tool_line, write_diff

# -- fake streaming model ---------------------------------------------------------------------------

def chunk(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))])


def fragment(index: int, *, id: str | None = None, name: str | None = None, args: str | None = None):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=args))


def text_chunks(*pieces: str) -> list[SimpleNamespace]:
    return [chunk(piece) for piece in pieces]


def call_chunks(index: int, name: str, call_id: str, **args) -> list[SimpleNamespace]:
    """One tool call split into three fragments: id+name, then the arguments in two halves."""
    raw = json.dumps(args)
    half = len(raw) // 2
    return [
        chunk(tool_calls=[fragment(index, id=call_id, name=name, args="")]),
        chunk(tool_calls=[fragment(index, args=raw[:half])]),
        chunk(tool_calls=[fragment(index, args=raw[half:])]),
    ]


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


def make_agent(project: Path, replies: list, **options) -> tuple[CodingAgent, list[list[dict]], list[tuple]]:
    """Agent whose client replays `replies`: a list of chunks when streaming, a FakeMessage otherwise."""
    events: list[tuple[str, dict]] = []
    options.setdefault("on_event", lambda kind, data: events.append((kind, data)))
    agent = CodingAgent(project, allow_commands=False, **options)
    seen: list[list[dict]] = []

    def create(**kwargs):
        seen.append(list(kwargs["messages"]))
        reply = replies.pop(0)
        if kwargs.get("stream"):
            assert isinstance(reply, list), "streaming request got a non-streamed reply"
            return iter(reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=reply)])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, seen, events


def kinds(events: list[tuple[str, dict]]) -> list[str]:
    return [kind for kind, _ in events if kind != "request"]  # "request" marks each model call


# -- streaming ----------------------------------------------------------------------------------------

def test_accumulates_interleaved_tool_call_fragments() -> None:
    calls: dict[int, dict[str, str]] = {}
    for piece in [
        fragment(0, id="a", name="read_file", args='{"pa'),
        fragment(1, id="b", name="git_status", args=""),
        fragment(0, args='th": "x.py"}'),
        fragment(1, args="{}"),
    ]:
        accumulate_tool_call(calls, piece)
    assert calls == {
        0: {"id": "a", "name": "read_file", "arguments": '{"path": "x.py"}'},
        1: {"id": "b", "name": "git_status", "arguments": "{}"},
    }


def test_repeated_name_fragments_are_not_doubled() -> None:
    calls: dict[int, dict[str, str]] = {}
    accumulate_tool_call(calls, fragment(0, id="a", name="read_file", args='{"path"'))
    accumulate_tool_call(calls, fragment(0, name="read_file", args=': "x"}'))
    assert calls[0]["name"] == "read_file" and calls[0]["arguments"] == '{"path": "x"}'


def test_streamed_tool_calls_run_and_final_text_is_forwarded(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, seen, events = make_agent(tmp_path, [
        call_chunks(0, "read_file", "c1", path="a.py") + call_chunks(1, "git_status", "c2"),
        call_chunks(0, "edit_file", "c3", path="a.py", old_text="x = 1", new_text="x = 2"),
        text_chunks("I changed ", "x to 2 in ", "a.py."),
    ], stream=True)
    assert agent.run("Set x to 2") == "I changed x to 2 in a.py."
    assert (tmp_path / "a.py").read_text() == "x = 2\n"
    assistant = seen[1][-3]
    assert [c["function"]["name"] for c in assistant["tool_calls"]] == ["read_file", "git_status"]
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"}
    assert "content" not in assistant  # no text streamed with the calls
    assert "".join(data["delta"] for kind, data in events if kind == "text") == "I changed x to 2 in a.py."
    assert events[-1] == ("final", {"text": "I changed x to 2 in a.py.", "streamed": True})
    assert kinds(events).count("tool_result") == 3


def test_nudged_streamed_reply_is_marked_discarded(tmp_path: Path, monkeypatch) -> None:
    # a question: the question pipeline would answer it; this tests the agent loop
    switch_off(monkeypatch, "question_pipeline")
    (tmp_path / "a.txt").write_text("x")
    agent, seen, events = make_agent(tmp_path, [
        text_chunks("Could you tell ", "me which file?"),
        call_chunks(0, "read_file", "c1", path="a.txt"),
        text_chunks("It contains x."),
    ], stream=True)
    assert agent.run("What is in a.txt?") == "It contains x."
    assert seen[1][-1] == {"role": "user", "content": QUESTION_NUDGE}
    assert kinds(events)[:4] == ["text", "text", "text_end", "discarded"]
    assert [data for kind, data in events if kind == "discarded"][0]["retry"] is True
    assert events[-1][1]["streamed"] is True


def test_streamed_text_tool_call_is_marked_not_the_answer(tmp_path: Path, monkeypatch) -> None:
    # a question: the question pipeline would answer it; this tests the agent loop
    switch_off(monkeypatch, "question_pipeline")
    (tmp_path / "a.txt").write_text("x")
    agent, _, events = make_agent(tmp_path, [
        text_chunks('{"name": "read_file", ', '"arguments": {"path": "a.txt"}}'),
        text_chunks("It contains x."),
    ], stream=True)
    assert agent.run("What is in a.txt?") == "It contains x."
    discarded = [data for kind, data in events if kind == "discarded"]
    assert discarded == [{"reason": "text was a tool call", "retry": False}]


def test_falls_back_when_server_rejects_streaming(tmp_path: Path) -> None:
    agent, seen, events = make_agent(tmp_path, [
        call("git_status"), text("Clean."),
    ], stream=True)
    original = agent.client.chat.completions.create

    def create(**kwargs):
        if kwargs.get("stream"):
            raise RuntimeError("streaming with tools is not supported")
        return original(**kwargs)

    agent.client.chat.completions.create = create
    assert agent.run("status?") == "Clean."
    assert agent.stream is False
    assert [k for k in kinds(events) if k == "notice"] == ["notice"]
    assert events[-1][1]["streamed"] is False


def test_non_streaming_path_reports_events_without_text(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    agent, _, events = make_agent(tmp_path, [call("read_file", path="a.txt"), text("x.")])
    assert agent.run("read a.txt") == "x."
    assert kinds(events) == ["tool_call", "tool_result", "final"]
    assert events[-1][1] == {"text": "x.", "streamed": False}


# -- terminal rendering -------------------------------------------------------------------------------

def test_renderer_shows_one_line_per_tool_and_streamed_answer(tmp_path: Path, monkeypatch) -> None:
    # Default: the model's raw text is held back; only progress lines and the final answer show.
    # a short request naming nothing: clarify_model would ask the scripted model first
    switch_off(monkeypatch, "clarify_first")
    (tmp_path / "app.py").write_text("def f():\n    return 1\n")
    out = Console(record=True, width=100, force_terminal=False)
    agent, _, _ = make_agent(tmp_path, [
        text_chunks("Let me ", "ask first."),
        call_chunks(0, "read_file", "c1", path="app.py"),
        call_chunks(0, "edit_file", "c2", path="app.py", old_text="return 1", new_text="  return 2"),
        text_chunks("Changed f to ", "return 2."),
    ], stream=True, on_event=TerminalRenderer(out))
    agent.run("make f return 2")
    shown = [line.rstrip() for line in out.export_text().splitlines()]
    assert "Let me ask first." not in "\n".join(shown)  # the sent-back reply never reaches the screen
    assert shown == [
        "(retrying: replied without inspecting the project)",
        "· read_file app.py",
        "· edit_file app.py (exact match)",
        "Changed f to return 2.",
    ]


def test_renderer_can_still_stream_text_live(tmp_path: Path) -> None:
    out = Console(record=True, width=100, force_terminal=False)
    agent, _, _ = make_agent(tmp_path, [
        call_chunks(0, "list_files", "c1"), text_chunks("All ", "done."),
    ], stream=True, on_event=TerminalRenderer(out, stream_text=True))
    agent.run("list")
    assert out.export_text().rstrip().endswith("All done.")


def test_tool_lines_summarise_outcomes() -> None:
    assert str(tool_line("run_check", {"command": "python -m pytest -q"}, "exit code: 0\n3 passed")) == (
        "· run_check python -m pytest -q → exit 0"
    )
    edited = "Edited src/app.py (matched ignoring indentation).\nSyntax check: OK."
    assert str(tool_line("edit_file", {"path": "src/app.py"}, edited)) == (
        "· edit_file src/app.py (matched ignoring indentation)"
    )
    broken = "Edited a.py (exact match).\nWARNING: the file now has a SyntaxError at line 2: bad. Fix it."
    assert "⚠ the file now has a SyntaxError at line 2" in str(tool_line("edit_file", {"path": "a.py"}, broken))
    long = tool_line("search_text", {"query": "x" * 200}, "")
    assert len(str(long)) < 100 and str(long).endswith("…")
    assert str(tool_line("edit_file", {"path": "a.py"}, DECLINED)) == f"· edit_file a.py → {DECLINED}"


# -- undo ---------------------------------------------------------------------------------------------

def test_undo_restores_edits_and_deletes_created_files_over_two_turns(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, _, _ = make_agent(tmp_path, [
        call("edit_file", path="a.py", old_text="x = 1", new_text="x = 2"), text("Done one."),
        call("edit_file", path="a.py", old_text="x = 2", new_text="x = 3"),
        call("write_file", path="pkg/sub/b.py", content="y = 1\n"), text("Done two."),
    ], undo_depth=10)
    agent.run("turn one")
    agent.run("turn two")
    assert (tmp_path / "a.py").read_text() == "x = 3\n" and (tmp_path / "pkg/sub/b.py").exists()

    report = agent.undo()
    assert report.applied and report.restored == ["a.py"] and report.removed == ["pkg/sub/b.py"]
    assert (tmp_path / "a.py").read_text() == "x = 2\n"
    assert not (tmp_path / "pkg").exists()  # directories Vega created are removed too

    report = agent.undo()
    assert report.applied and report.restored == ["a.py"]
    assert (tmp_path / "a.py").read_text() == "x = 1\n"
    assert agent.undo() is None


def test_turns_without_file_changes_are_not_checkpointed(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, _, _ = make_agent(tmp_path, [
        call("edit_file", path="a.py", old_text="x = 1", new_text="x = 2"), text("Done."),
        call("read_file", path="a.py"), text("x is 2."),
    ], undo_depth=10)
    agent.run("edit")
    agent.run("question")
    assert len(agent.checkpoints) == 1
    agent.undo()
    assert (tmp_path / "a.py").read_text() == "x = 1\n"


def test_undo_depth_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # write_guard: this scripted model overwrites a fixture file it never read; the test is about
    # undo depth, not the overwrite hold (tests/test_write_guard.py)
    switch_off(monkeypatch, "write_guard")
    (tmp_path / "n.txt").write_text("0")
    replies = []
    for i in range(4):
        replies += [call("write_file", path="n.txt", content=str(i + 1)), text("ok")]
    agent, _, _ = make_agent(tmp_path, replies, undo_depth=2)
    for i in range(4):
        agent.run(f"turn {i}")
    assert len(agent.checkpoints) == 2
    agent.undo()
    agent.undo()
    assert (tmp_path / "n.txt").read_text() == "2" and agent.undo() is None


def test_undo_refuses_when_user_changed_file_and_force_overrides(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, _, _ = make_agent(tmp_path, [
        call("edit_file", path="a.py", old_text="x = 1", new_text="x = 2"), text("Done."),
    ], undo_depth=10)
    agent.run("edit")
    (tmp_path / "a.py").write_text("x = 2  # my own change\n")

    report = agent.undo()
    assert not report.applied and report.conflicts == ["a.py"]
    assert (tmp_path / "a.py").read_text() == "x = 2  # my own change\n"
    assert len(agent.checkpoints) == 1  # still available

    report = agent.undo(force=True)
    assert report.applied and report.restored == ["a.py"]
    assert (tmp_path / "a.py").read_text() == "x = 1\n"


def test_undo_is_off_by_default(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, _, _ = make_agent(tmp_path, [
        call("edit_file", path="a.py", old_text="x = 1", new_text="x = 2"), text("Done."),
    ])
    agent.run("edit")
    assert agent.checkpoints == [] and agent.undo() is None


# -- review mode --------------------------------------------------------------------------------------

def test_review_decline_leaves_file_untouched_and_tells_model(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    asked: list[WritePlan] = []

    def decline(plan: WritePlan) -> bool:
        asked.append(plan)
        return False

    agent, seen, _ = make_agent(tmp_path, [
        call("edit_file", path="a.py", old_text="x = 1", new_text="x = 2"),
        text("Understood, I left a.py unchanged."),
    ], review=True, approve=decline, undo_depth=10)
    agent.run("Set x to 2")
    assert (tmp_path / "a.py").read_text() == "x = 1\n"
    assert seen[1][-1] == {"role": "tool", "tool_call_id": "c-edit_file", "content": DECLINED}
    assert len(asked) == 1 and "-x = 1\n+x = 2\n" in write_diff(asked[0])
    assert agent.checkpoints == []


def test_review_accept_writes_file(tmp_path: Path) -> None:
    approved: list[str] = []

    def accept(plan: WritePlan) -> bool:
        approved.append(write_diff(plan))
        return True

    agent, seen, _ = make_agent(tmp_path, [
        call("write_file", path="new.py", content="y = 1\n"), text("Created new.py."),
    ], review=True, approve=accept)
    agent.run("create new.py")
    assert (tmp_path / "new.py").read_text() == "y = 1\n"
    assert seen[1][-1]["content"].startswith("Wrote new.py")
    assert approved[0].startswith("--- /dev/null\n+++ b/new.py\n") and "+y = 1\n" in approved[0]


def test_failed_edit_is_not_offered_for_review(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, seen, _ = make_agent(tmp_path, [
        call("edit_file", path="a.py", old_text="nope", new_text="z"), text("It did not match."),
    ], review=True, approve=lambda plan: (_ for _ in ()).throw(AssertionError("asked")))
    agent.run("edit")
    assert seen[1][-1]["content"].startswith("Edit not applied")


def test_plan_write_does_not_touch_disk(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    plan = plan_write("edit_file", {"path": "a.py", "old_text": "x = 1", "new_text": "x = 2"}, tmp_path)
    assert isinstance(plan, WritePlan) and plan.after == "x = 2\n"
    # "z = 1": a bare "z" is an undefined name, which atomic_writes (on by default) refuses
    assert plan_write("write_file", {"path": "d/b.py", "content": "z = 1\n"}, tmp_path).before is None
    assert (tmp_path / "a.py").read_text() == "x = 1\n" and not (tmp_path / "d").exists()


def test_approval_prompt_shows_diff_and_reads_answer(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    plan = plan_write("edit_file", {"path": "a.py", "old_text": "x = 1", "new_text": "x = 2"}, tmp_path)
    out = Console(record=True, width=80)
    prompts: list[str] = []
    out.input = lambda prompt="": prompts.append(prompt) or "y"
    assert ask_approval(plan, out) is True
    shown = out.export_text()
    assert "Vega wants to edit file a.py" in shown and "-x = 1" in shown and "+x = 2" in shown
    assert "\\[y/N]" in prompts[0]  # escaped so rich does not swallow it as a markup tag
    out.input = lambda prompt="": "n"
    assert ask_approval(plan, out) is False


def test_cli_undo_reports_files_and_tells_model(tmp_path: Path, monkeypatch) -> None:
    from vega import cli

    (tmp_path / "a.py").write_text("x = 1\n")
    agent, _, _ = make_agent(tmp_path, [
        call("edit_file", path="a.py", old_text="x = 1", new_text="x = 2"),
        call("write_file", path="b.py", content="y = 1\n"), text("Done."),
    ], undo_depth=10)
    agent.run("edit")
    out = Console(record=True, width=100)
    monkeypatch.setattr(cli, "console", out)
    cli._undo(agent, force=False)
    shown = out.export_text()
    assert "restored a.py" in shown and "removed  b.py" in shown
    assert not (tmp_path / "b.py").exists() and (tmp_path / "a.py").read_text() == "x = 1\n"
    assert agent.messages[-1]["role"] == "user" and "a.py, b.py" in agent.messages[-1]["content"]
    cli._undo(agent, force=False)
    assert "Nothing to undo" in out.export_text()


def test_stop_abandons_a_streaming_reply_at_once(tmp_path: Path) -> None:
    """Esc/Ctrl+C used to wait for the reply (and any command) to finish: the stop is now seen
    between streamed chunks and the turn ends without running the half-written reply."""
    holder: dict[str, CodingAgent] = {}

    class StoppingChunks(list):
        def __iter__(self):
            yield from text_chunks("Let me ")
            holder["agent"].request_stop()  # the user presses Esc while the model is still writing
            yield from text_chunks("search ", "everything ", "again.")

    agent, seen, events = make_agent(tmp_path, [StoppingChunks()], stream=True)
    holder["agent"] = agent
    answer = agent.run("tidy up the code")
    assert answer == "Stopped by you."
    texts = "".join(data.get("delta", "") for kind, data in events if kind == "text")
    assert "again" not in texts and len(seen) == 1


def test_a_refused_write_is_shown_as_a_failure() -> None:
    from vega.ui import step_outcome
    text, style = step_outcome("write_file", "Write not applied: app.py would not parse (x).")
    assert style == "warning" and text.startswith("Write not applied")
