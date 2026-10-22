"""finish_gate_no_edit: a task routed as a change does not finish with no file edited without
being sent back once.

The failure it prevents: a bug report routed "agent" with change: true is answered in prose after
a few reads (it explains the code, or restates the problem) and nothing is edited. Scripted fake
models.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from switches import switch_off
from test_agent_loop import agent_with, call, text

from vega import features
from vega.agent import NO_EDIT_NOTE, asks_for_change, asks_the_user, command_writes, no_edit_problem

GATE = "(Vega harness) Not finished"
REPORT_TASK = ("The invoice total is wrong when a discount code is applied twice: the second "
               "code is subtracted again. The total should apply each code once.")
FIX_TASK = "Fix `area` in geo.py: it returns w + h instead of w * h."


@pytest.fixture(autouse=True)
def agent_loop_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scripted tasks go through the agent loop: the pipelines and the clarify gate that
    would take a short fix request are off."""
    switch_off(monkeypatch, "direct_pipeline", "deliverable_pipeline", "project_pipeline",
               "web_pipeline", "clarify_first")


def notes(seen: list[list[dict]]) -> list[str]:
    return [request[-1]["content"] for request in seen
            if request[-1]["role"] == "user" and request[-1]["content"].startswith(GATE)]


def project(tmp_path: Path) -> Path:
    (tmp_path / "geo.py").write_text("def area(w, h):\n    return w + h\n")
    return tmp_path


# -- what counts as a change -------------------------------------------------------------------

def route(change: bool, reason: str, question: bool = False) -> dict:
    return {"name": "agent", "change": change, "question": question, "reasons": [reason]}


def test_bug_reports_routed_as_changes_are_changes() -> None:
    assert asks_for_change(route(True, "no question form: treated as a change"), REPORT_TASK)
    assert asks_for_change(route(True, "asks for a change: fix"), "Fix the date parser.")
    assert asks_for_change(route(True, "asks for a change: build"),
                           "It should skip hidden folders like the importer (and build output).")


@pytest.mark.parametrize("task", ["read a.txt", "Look up how widget 2.0 sets a timeout.",
                                  "Say hi.", "list", "Look for the workflow code"])
def test_a_fallback_change_without_a_bug_report_is_not_gated(task: str) -> None:
    assert not asks_for_change(route(True, "no question form: treated as a change"), task)


def test_only_run_is_not_a_change_unless_something_is_broken() -> None:
    assert not asks_for_change(route(True, "asks for a change: run"), "Run git status.")
    assert asks_for_change(route(True, "asks for a change: run"), "Run the tests; they fail.")


def test_questions_are_not_changes() -> None:
    assert not asks_for_change(route(False, "a question with no change verb outside code",
                                     question=True), "Why does area() fail?")


def test_no_problem_after_a_write_or_a_writing_command_or_a_question_to_the_user() -> None:
    assert no_edit_problem(True, False, []) == NO_EDIT_NOTE
    assert no_edit_problem(False, False, []) is None
    assert no_edit_problem(True, True, []) is None
    assert no_edit_problem(True, False, ["sed -i 's/a/b/' geo.py"]) is None
    assert no_edit_problem(True, False, ["python -m pytest -q 2>&1"]) == NO_EDIT_NOTE
    assert no_edit_problem(True, False, [], "Which file do you mean?") is None


@pytest.mark.parametrize("command", ["sed -i '' 's/a/b/' x.py", "echo hi > out.txt", "mv a b",
                                     "cat a | tee b", "git apply fix.patch", "printf x >> log"])
def test_commands_that_write(command: str) -> None:
    assert command_writes(command)


@pytest.mark.parametrize("command", ["python -m pytest -q", "ls -la 2>&1", "grep -rn x . 2>/dev/null",
                                     "git status", "git diff --stat"])
def test_commands_that_only_read(command: str) -> None:
    assert not command_writes(command)


def test_asks_the_user() -> None:
    assert asks_the_user("I looked at geo.py.\n\nShould I change both functions?")
    assert not asks_the_user("Why? Because the total counts every code.\n\nDone.")


# -- the agent loop --------------------------------------------------------------------------

def test_a_prose_answer_to_a_fix_is_sent_back_once_then_the_edit_lands(tmp_path: Path) -> None:
    agent, seen = agent_with(project(tmp_path), [
        call("read_file", path="geo.py"),
        text("area() adds w and h; it should multiply them."),
        call("edit_file", path="geo.py", old_text="w + h", new_text="w * h"),
        text("Fixed area()."),
    ])
    assert agent.run(FIX_TASK) == "Fixed area()."
    assert (tmp_path / "geo.py").read_text().endswith("w * h\n")
    [note] = notes(seen)
    assert NO_EDIT_NOTE in note


def test_the_second_no_edit_answer_stands(tmp_path: Path) -> None:
    agent, seen = agent_with(project(tmp_path), [
        call("read_file", path="geo.py"),
        text("area() looks fine to me."),
        text("No change is needed: area() is only called with h = 1."),
    ])
    assert agent.run(FIX_TASK) == "No change is needed: area() is only called with h = 1."
    assert len(notes(seen)) == 1


def test_an_edit_finishes_without_the_note(tmp_path: Path) -> None:
    agent, seen = agent_with(project(tmp_path), [
        call("edit_file", path="geo.py", old_text="w + h", new_text="w * h"),
        text("Fixed area()."),
    ])
    assert agent.run(FIX_TASK) == "Fixed area()."
    assert not any(NO_EDIT_NOTE in note for note in notes(seen))


def test_a_question_route_is_not_sent_back(tmp_path: Path, monkeypatch) -> None:
    switch_off(monkeypatch, "question_pipeline")  # the question goes through the agent loop here
    agent, seen = agent_with(project(tmp_path), [
        call("read_file", path="geo.py"), text("It returns w + h."), text("again"),
    ])
    assert agent.run("What does `area` in geo.py return?") == "It returns w + h."
    assert notes(seen) == []


def test_a_lookup_request_is_not_sent_back(tmp_path: Path) -> None:
    agent, seen = agent_with(project(tmp_path), [
        call("read_file", path="geo.py"), text("def area(w, h): return w + h"), text("again"),
    ])
    assert agent.run("read geo.py") == "def area(w, h): return w + h"
    assert notes(seen) == []


def test_switched_off_the_prose_answer_stands(tmp_path: Path, monkeypatch) -> None:
    switch_off(monkeypatch, "finish_gate_no_edit")
    agent, seen = agent_with(project(tmp_path), [
        call("read_file", path="geo.py"), text("area() adds w and h."), text("again"),
    ])
    assert agent.run(FIX_TASK) == "area() adds w and h."
    assert notes(seen) == []


def test_the_switch_is_a_default() -> None:
    assert "finish_gate_no_edit" in features.FEATURES
