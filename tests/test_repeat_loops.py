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


def test_reads_between_repeats_do_not_reset_the_count(tmp_path: Path, monkeypatch) -> None:
    # the logged loop (generation-tokens report 2026-10-27): write_file -> refused -> read_file ->
    # the same write_file, up to 49 times; a successful read between repeats used to reset the
    # count, so the stop never fired
    setup(tmp_path, monkeypatch)
    for i in range(REPEAT_STOP):
        (tmp_path / f"notes{i}.txt").write_text(f"note {i}\n")
    bad = {"path": "journal.py", "content": BROKEN}
    replies = [call("write_file", **bad)]
    for i in range(REPEAT_STOP):  # a different read each time: none is a ledger repeat
        replies += [call("read_file", path=f"notes{i}.txt"), call("write_file", **bad)]
    replies.append(text("never asked"))
    agent, seen = agent_with(tmp_path, replies)
    answer = agent.run("fix the save")
    assert answer.startswith("Stopped: the model kept repeating a call that had already failed")
    assert len(seen) == 1 + 2 * REPEAT_STOP  # stopped right after the REPEAT_STOP-th repeat
    results = tool_results(agent)
    assert results[0].startswith("Write not applied")
    assert all("note" in r for r in results[1::2])  # the reads ran
    assert all(r.startswith("(Vega harness) Not run") for r in results[2::2])


def test_repeat_stop_reads_off_restores_the_old_rule(tmp_path: Path, monkeypatch) -> None:
    # the A/B arm: any call that ran resets the count, so the read-between loop is never stopped
    setup(tmp_path, monkeypatch)
    switch_off(monkeypatch, "repeat_stop_reads")
    for i in range(REPEAT_STOP):
        (tmp_path / f"notes{i}.txt").write_text(f"note {i}\n")
    bad = {"path": "journal.py", "content": BROKEN}
    replies = [call("write_file", **bad)]
    for i in range(REPEAT_STOP):
        replies += [call("read_file", path=f"notes{i}.txt"), call("write_file", **bad)]
    replies.append(text("gave up"))
    agent, seen = agent_with(tmp_path, replies)
    assert agent.run("fix the save") == "gave up"
    assert len(seen) == 2 + 2 * REPEAT_STOP


def test_a_saved_write_between_repeats_resets_the_count(tmp_path: Path, monkeypatch) -> None:
    # progress elsewhere is not a stuck loop: the count restarts after a write that saved
    setup(tmp_path, monkeypatch)
    (tmp_path / "journal.py").write_text("def save(f):\n    pass\n")
    bad = {"path": "journal.py", "old_text": "nope", "new_text": "x"}  # fails: guarded always
    replies = [call("edit_file", **bad)]
    replies += [call("edit_file", **bad) for _ in range(REPEAT_STOP - 1)]
    replies.append(call("write_file", path="other.py", content="X = 1\n"))
    replies += [call("edit_file", **bad) for _ in range(REPEAT_STOP - 1)]
    replies.append(text("Gave up."))
    agent, _ = agent_with(tmp_path, replies)
    assert agent.run("fix the save") == "Gave up."
    assert (tmp_path / "other.py").read_text() == "X = 1\n"


def test_repeat_stop_waits_for_the_fourth_repeat() -> None:
    # 10-20..27 debug-log replay: stopping at the 3rd repeat would have ended 2 passing attempts,
    # at the 4th none (47 attempts stopped, all failing)
    assert REPEAT_STOP == 4


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


# -- noop_write_repeat (bigbuild 2026-10-28: one file rewritten unchanged ~36 times in a turn) ----

ERRORS = "class ValidationError(Exception):\n    pass\n"


def test_a_byte_identical_write_is_reported_unchanged(tmp_path: Path, monkeypatch) -> None:
    setup(tmp_path, monkeypatch)
    (tmp_path / "errors.py").write_text(ERRORS)
    agent, _ = agent_with(tmp_path, [
        call("write_file", path="errors.py", content=ERRORS), text("Done."),
    ])
    assert agent.run("tidy errors.py") == "Done."
    result = tool_results(agent)[0]
    assert result.startswith(agent_module.NOOP_WRITE_PREFIX)
    assert "errors.py already has this content" in result


def test_noop_writes_count_toward_repeat_stop(tmp_path: Path, monkeypatch) -> None:
    setup(tmp_path, monkeypatch)
    (tmp_path / "errors.py").write_text(ERRORS)
    replies = [call("write_file", path="errors.py", content=ERRORS)
               for _ in range(REPEAT_STOP + 2)] + [text("never asked")]
    agent, seen = agent_with(tmp_path, replies)
    answer = agent.run("tidy errors.py")
    assert answer.startswith("Stopped: the model kept repeating")
    assert "already has this content" in answer
    assert len(seen) == REPEAT_STOP


def test_a_refused_write_alternating_with_a_noop_write_is_stopped(tmp_path: Path,
                                                                  monkeypatch) -> None:
    # ledger 5 / tasks 5 on 10-28: an "undefined name" refusal, then errors.py rewritten
    # unchanged, then the same refused write (run again: a write had "saved"), for many steps
    setup(tmp_path, monkeypatch)
    (tmp_path / "errors.py").write_text(ERRORS)
    bad = {"path": "journal.py", "content": BROKEN}
    replies = []
    for _ in range(REPEAT_STOP + 2):
        replies += [call("write_file", **bad), call("write_file", path="errors.py", content=ERRORS)]
    replies.append(text("never asked"))
    agent, seen = agent_with(tmp_path, replies)
    assert agent.run("fix the save").startswith("Stopped: the model kept repeating")
    results = tool_results(agent)
    assert results[0].startswith("Write not applied")
    assert results[2].startswith("(Vega harness) Not run")  # the noop write saved nothing
    assert len(seen) == 1 + REPEAT_STOP


def test_a_changing_write_still_resets_after_noop_writes(tmp_path: Path, monkeypatch) -> None:
    setup(tmp_path, monkeypatch)
    (tmp_path / "errors.py").write_text(ERRORS)
    same = call("write_file", path="errors.py", content=ERRORS)
    replies = [same, same, same, call("write_file", path="other.py", content="X = 1\n"),
               same, same, same, text("Done.")]
    agent, _ = agent_with(tmp_path, replies)
    assert agent.run("tidy errors.py") == "Done."


def test_noop_write_repeat_off_restores_a_saved_write(tmp_path: Path, monkeypatch) -> None:
    setup(tmp_path, monkeypatch)
    switch_off(monkeypatch, "noop_write_repeat")
    (tmp_path / "errors.py").write_text(ERRORS)
    replies = [call("write_file", path="errors.py", content=ERRORS)
               for _ in range(REPEAT_STOP + 2)] + [text("Done.")]
    agent, _ = agent_with(tmp_path, replies)
    assert agent.run("tidy errors.py") == "Done."
    assert all(r.startswith("Wrote errors.py") for r in tool_results(agent))
