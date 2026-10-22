"""localize_ties (opt-in): Likely places that score about the same are marked as equally likely,
the turn's first source write is held once while such a file is unread, and the no-edit finish
gate names the Likely-places files.

The failure it prevents (selfbuild 2026-10-21): the right file was #2 in a near-tie (25.33 vs
27.04, 10.96 vs 12.09) and the model edited #1 without reading #2. Scripted fake models.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from switches import switch_off
from test_agent_loop import agent_with, call, text
from test_localize import filler, make

from vega import features, localize
from vega.agent import NO_EDIT_NOTE

TASK = 'merge_rows breaks: it says "rows merged without keys". Fix it.'
HOLD = "(Vega harness) Not written yet: the Likely places ranked"


@pytest.fixture
def ties_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "localize_ties")


@pytest.fixture
def agent_loop_only(monkeypatch: pytest.MonkeyPatch) -> None:
    switch_off(monkeypatch, "direct_pipeline", "deliverable_pipeline", "project_pipeline",
               "web_pipeline", "clarify_first", "finish_gate", "acceptance")


def tied_project(tmp_path: Path) -> Path:
    """Two modules hold the task's name and message equally: a tie (pkg/join.py ranks first)."""
    body = "def merge_rows(path, left, right):\n    return 'rows merged without keys'\n"
    return make(tmp_path, {**filler(30), "pkg/table.py": body, "pkg/join.py": body,
                           "tests/test_table.py": "def test_x():\n    pass\n"})


def clear_project(tmp_path: Path) -> Path:
    return make(tmp_path, {**filler(30), "pkg/table.py": "def merge_rows(path, left, right):\n"
                           "    return 'rows merged without keys'\n"})


def results(seen: list[list[dict]]) -> list[str]:
    return [m["content"] for m in seen[-1] if m["role"] == "tool"]


# -- the tie flag --------------------------------------------------------------------------------

def places(*scores: float) -> list[localize.Place]:
    return [localize.Place(f"m{i}.py", s, ["x"]) for i, s in enumerate(scores, 1)]


def test_near_ties_close_vs_far() -> None:
    assert localize.near_ties(places(27.04, 25.33, 4.0)) == ["m1.py", "m2.py"]  # the xml case
    assert localize.near_ties(places(12.09, 10.96)) == ["m1.py", "m2.py"]  # the overview case
    assert localize.near_ties(places(12.0, 10.3, 10.25)) == ["m1.py", "m2.py", "m3.py"]
    assert localize.near_ties(places(27.0, 20.0)) == []  # 0.74 x the top: #1 stands alone
    assert localize.near_ties(places(5.0)) == []


def test_block_marks_a_tie_only_when_switched_on(tmp_path: Path) -> None:
    project = tied_project(tmp_path)
    located = localize.locate(project, TASK, ties=True)
    assert located.tied == ["pkg/join.py", "pkg/table.py"]
    assert located.paths[:2] == located.tied
    assert "2. pkg/table.py (" in located.text and "[about as likely as 1]" in located.text
    assert located.text.endswith(localize.TIE_NOTE.format("1 and 2", "both"))
    assert len(located.text) <= localize.MAX_CHARS
    plain = localize.block(project, TASK)
    assert "about as likely" not in plain and "score about the same" not in plain


def test_no_tie_mark_when_the_top_file_stands_alone(tmp_path: Path) -> None:
    located = localize.locate(clear_project(tmp_path), TASK, ties=True)
    assert located.paths[0] == "pkg/table.py" and located.tied == []
    assert "score about the same" not in located.text


def test_tie_unread_is_for_source_files_only() -> None:
    tied = ["src/a.py", "src/b.py"]
    assert localize.tie_unread("src/a.py", tied, set()) == ["src/b.py"]
    assert localize.tie_unread("src/c.py", tied, {"src/a.py"}) == ["src/b.py"]
    assert localize.tie_unread("src/a.py", tied, {"src/b.py"}) == []
    assert localize.tie_unread("tests/test_a.py", tied, set()) == []
    assert localize.tie_unread("README.md", tied, set()) == []
    assert localize.tie_unread("src/a.py", ["src/a.py"], set()) == []


# -- the one-time send-back ----------------------------------------------------------------------

def test_first_source_write_is_held_while_a_tied_file_is_unread(
        tmp_path: Path, agent_loop_only: None, ties_on: None) -> None:
    edit = call("edit_file", path="pkg/table.py", old_text="without keys", new_text="with keys")
    agent, seen = agent_with(tied_project(tmp_path), [
        edit, call("read_file", path="pkg/join.py"), edit, text("Done.")])
    agent.run(TASK)
    held, _read, applied = results(seen)
    assert held.startswith(HOLD) and "`pkg/join.py`" in held and "1. `pkg/join.py`" in held
    assert "with keys" in (tmp_path / "pkg/table.py").read_text()
    assert not applied.startswith(HOLD)
    assert "score about the same" in seen[0][-1]["content"]


def test_not_held_when_the_tied_file_was_read(
        tmp_path: Path, agent_loop_only: None, ties_on: None) -> None:
    agent, seen = agent_with(tied_project(tmp_path), [
        call("read_file", path="pkg/join.py"),
        call("edit_file", path="pkg/table.py", old_text="without keys", new_text="with keys"),
        text("Done.")])
    agent.run(TASK)
    assert not any(r.startswith(HOLD) for r in results(seen))
    assert "with keys" in (tmp_path / "pkg/table.py").read_text()


def test_held_once_a_repeat_goes_through(
        tmp_path: Path, agent_loop_only: None, ties_on: None) -> None:
    edit = call("edit_file", path="pkg/table.py", old_text="without keys", new_text="with keys")
    agent, seen = agent_with(tied_project(tmp_path), [edit, edit, text("Done.")])
    agent.run(TASK)
    assert [r.startswith(HOLD) for r in results(seen)] == [True, False]
    assert "with keys" in (tmp_path / "pkg/table.py").read_text()


def test_a_test_file_write_is_not_held_nor_checked(
        tmp_path: Path, agent_loop_only: None, ties_on: None) -> None:
    agent, seen = agent_with(tied_project(tmp_path), [
        call("write_file", path="tests/test_table.py", content="def test_y():\n    pass\n"),
        call("edit_file", path="pkg/table.py", old_text="without keys", new_text="with keys"),
        text("Done.")])
    agent.run(TASK)
    test_write, source_write = results(seen)
    assert not test_write.startswith(HOLD)  # a test file: not the source write
    assert source_write.startswith(HOLD)  # the first source write is still checked


def test_switched_off_nothing_is_held(tmp_path: Path, agent_loop_only: None) -> None:
    agent, seen = agent_with(tied_project(tmp_path), [
        call("edit_file", path="pkg/table.py", old_text="without keys", new_text="with keys"),
        text("Done.")])
    agent.run(TASK)
    assert not any(r.startswith(HOLD) for r in results(seen))
    assert "score about the same" not in seen[0][-1]["content"]
    assert agent._likely_tied == []


def test_no_hold_without_a_tie(tmp_path: Path, agent_loop_only: None, ties_on: None) -> None:
    agent, seen = agent_with(clear_project(tmp_path), [
        call("edit_file", path="pkg/table.py", old_text="without keys", new_text="with keys"),
        text("Done.")])
    agent.run(TASK)
    assert not any(r.startswith(HOLD) for r in results(seen))


# -- the no-edit finish gate ---------------------------------------------------------------------

def gate_notes(seen: list[list[dict]]) -> list[str]:
    return [r[-1]["content"] for r in seen if r[-1]["role"] == "user"
            and r[-1]["content"].startswith("(Vega harness) Not finished")]


@pytest.mark.parametrize("on", [True, False])
def test_the_no_edit_gate_names_the_likely_places(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, on: bool) -> None:
    switch_off(monkeypatch, "direct_pipeline", "deliverable_pipeline", "project_pipeline",
               "web_pipeline", "clarify_first")
    if on:
        monkeypatch.setenv(features.ENV_ON, "localize_ties")
    agent, seen = agent_with(tied_project(tmp_path), [
        call("read_file", path="pkg/table.py"),
        text("merge_rows returns the message; that is the bug."),
        text("No change needed."),
    ])
    agent.run(TASK)
    [note] = gate_notes(seen)
    assert NO_EDIT_NOTE in note
    named = "The Likely places for this task were `pkg/join.py`, `pkg/table.py`"
    assert (named in note) is on


def test_the_switch_is_opt_in() -> None:
    assert "localize_ties" in features.OPT_IN and "localize_ties" not in features.FEATURES
