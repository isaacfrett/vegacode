"""Breaking repeat loops (engineering diagnosis 2026-10-21, fix 5a): the agent loop's repeat guard
catches identical refused writes, and repeat_stop ends a turn that keeps repeating them."""

from __future__ import annotations

from pathlib import Path

from switches import switch_off
from test_agent_loop import agent_with, call, text

from vega import agent as agent_module
from vega.agent import REPEAT_STOP, REPEAT_STOP_ANSWER, CodingAgent
from vega.tools import execute

# bucketeer's refused write: a literal newline inside a string literal (does not parse)
BROKEN = 'def save(f):\n    f.write("\n")\n'
FIXED = 'def save(f):\n    f.write("\\n")\n'


def setup(tmp_path: Path, monkeypatch) -> None:
    # the agent loop with scripted calls: no clarify question, no single-file pipeline, and no
    # finish-gate send-back when every write was refused
    switch_off(monkeypatch, "clarify_first", "direct_pipeline", "finish_gate")
    (tmp_path / "other.txt").write_text("x\n")  # journal.py is new: no write_guard hold


def tool_results(agent: CodingAgent) -> list[str]:
    return [m["content"] for m in agent.messages if m.get("role") == "tool"]


def test_every_refusal_prefix_the_tools_return_is_a_failure(tmp_path: Path) -> None:
    (tmp_path / "journal.py").write_text("def save(f):\n    pass\n")
    results = [
        execute("write_file", {"path": "journal.py", "content": BROKEN}, tmp_path),
        execute("edit_file", {"path": "journal.py", "old_text": "nope", "new_text": "x"}, tmp_path),
        execute("remember", {"note": ""}, tmp_path),
        execute("search_text", {"query": ""}, tmp_path),
        execute("find_files", {"pattern": ""}, tmp_path),
    ]
    for result in results:
        assert result.startswith(agent_module._FAILED_RESULT_PREFIXES), result
    assert results[0].startswith("Write not applied")
    assert agent_module.DECLINED.startswith(agent_module._FAILED_RESULT_PREFIXES)


def test_an_identical_refused_write_is_not_run_again(tmp_path: Path, monkeypatch) -> None:
    setup(tmp_path, monkeypatch)
    bad = {"path": "journal.py", "content": BROKEN}
    agent, _ = agent_with(tmp_path, [
        call("write_file", **bad), call("write_file", **bad),
        call("write_file", path="journal.py", content=FIXED),
        text("Done."),
    ])
    assert agent.run("fix the save") == "Done."
    results = tool_results(agent)
    assert results[0].startswith("Write not applied")
    assert results[1].startswith("(Vega harness) Not run: you already made exactly this call")
    assert "write \\n inside the quotes" in results[1]
    assert (tmp_path / "journal.py").read_text() == FIXED


def test_repeating_it_ends_the_turn(tmp_path: Path, monkeypatch) -> None:
    setup(tmp_path, monkeypatch)
    bad = {"path": "journal.py", "content": BROKEN}
    replies = [call("write_file", **bad) for _ in range(REPEAT_STOP + 1)] + [text("never asked")]
    agent, seen = agent_with(tmp_path, replies)
    answer = agent.run("fix the save")
    assert answer.startswith("Stopped: the model kept repeating a call that had already failed")
    assert "Write not applied" in answer
    assert len(seen) == REPEAT_STOP + 1  # the first refusal, then REPEAT_STOP guarded repeats
    assert REPEAT_STOP_ANSWER.split("{")[0] in answer


def test_a_call_that_runs_between_resets_the_count(tmp_path: Path, monkeypatch) -> None:
    setup(tmp_path, monkeypatch)
    bad = {"path": "journal.py", "content": BROKEN}
    agent, _ = agent_with(tmp_path, [
        call("write_file", **bad), call("write_file", **bad), call("write_file", **bad),
        call("read_file", path="other.txt"),
        call("write_file", **bad), call("write_file", **bad),
        text("Gave up."),
    ])
    assert agent.run("fix the save") == "Gave up."


def test_repeat_stop_off_keeps_the_old_behaviour(tmp_path: Path, monkeypatch) -> None:
    setup(tmp_path, monkeypatch)
    switch_off(monkeypatch, "repeat_stop")
    bad = {"path": "journal.py", "content": BROKEN}
    replies = [call("write_file", **bad) for _ in range(REPEAT_STOP + 2)] + [text("Gave up.")]
    agent, _ = agent_with(tmp_path, replies)
    assert agent.run("fix the save") == "Gave up."


def test_a_refused_write_runs_again_after_another_write_saved(tmp_path: Path,
                                                              monkeypatch) -> None:
    # whether a write is refused depends on the file: after a saved write it may be accepted
    setup(tmp_path, monkeypatch)
    bad = {"path": "journal.py", "content": BROKEN}
    agent, _ = agent_with(tmp_path, [
        call("write_file", **bad),
        call("write_file", path="other.py", content="X = 1\n"),
        call("write_file", **bad),
        text("Gave up."),
    ])
    assert agent.run("fix the save") == "Gave up."
    results = tool_results(agent)
    assert results[0].startswith("Write not applied")
    assert results[2].startswith("Write not applied")  # ran again, not the repeat note
