"""Failure classification and transcript saving for eval attempts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from switches import switch_off

from vega import evals
from vega.evals import classify_failure

EDIT_CASE = {"id": "e", "task": "fix it", "files": {"a.py": "x = 1\n"}, "python_check": "assert 1"}
READ_CASE = {"id": "r", "task": "what is x?", "files": {"a.py": "x = 1\n"}, "response_matches": ["1"],
             "file_unchanged": ["a.py"]}


def attempt(**fields) -> dict:
    base = {"passed": False, "detail": "python_check failed: AssertionError", "failure": "acceptance",
            "tool_calls": {"read_file": 1}, "nudges": {}, "unexpected_files": []}
    return {**base, **fields}


def test_classes_in_priority_order() -> None:
    assert classify_failure(EDIT_CASE, attempt(passed=True), "", 1, 0) == ""
    assert classify_failure(EDIT_CASE, attempt(model_error=True), "", 0, 0) == "model_error"
    assert classify_failure(EDIT_CASE, attempt(hit_step_limit=True), "", 0, 0) == "step_limit"
    assert classify_failure(EDIT_CASE, attempt(tool_calls={}), "Which file should I change?", 0, 0) == "asked_user"
    assert classify_failure(EDIT_CASE, attempt(tool_calls={}), "Done.", 0, 0) == "no_tool_use"
    malformed = attempt(nudges={"tool call written as text could not be parsed": 2})
    assert classify_failure(EDIT_CASE, malformed, "Done.", 0, 0) == "malformed_calls"
    assert classify_failure(EDIT_CASE, attempt(), "I looked at it.", 0, 0) == "no_edit"
    assert classify_failure(EDIT_CASE, attempt(), "Done.", 0, 2) == "edit_failed"
    broken = attempt(detail="python_check failed: SyntaxError: invalid syntax")
    assert classify_failure(EDIT_CASE, broken, "Done.", 1, 0) == "broken_code"
    assert classify_failure(EDIT_CASE, attempt(), "Done.", 1, 0) == "wrong_fix"
    assert classify_failure(EDIT_CASE, attempt(detail="b.py was modified"), "Done.", 1, 0) == "wrong_file"
    assert classify_failure(READ_CASE, attempt(detail="response does not match"), "x is 2", 0, 0) == "wrong_answer"
    assert classify_failure(EDIT_CASE, attempt(failure="regression"), "Done.", 1, 0) == "regression"


def test_attempt_saves_transcript_and_class(tmp_path: Path, monkeypatch) -> None:
    switch_off(monkeypatch, "clarify_first")  # "fix it" would get questions, not the scripted loop
    switch_off(monkeypatch, "finish_gate_no_edit")  # the no-edit answer would be sent back once
    from test_agent_loop import call, text

    replies = [call("read_file", path="a.py"), text("x is 1, nothing to change.")]

    def factory(project: Path, max_steps: int):
        from vega.agent import CodingAgent

        agent = CodingAgent(project, allow_commands=False, max_steps=max_steps)
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=replies.pop(0))], usage=None))))
        return agent

    path = tmp_path / "t" / "e-1.json"
    result = evals.run_attempt(EDIT_CASE | {"python_check": "assert open('a.py').read() == 'x = 2\\n'"},
                               agent_factory=factory, transcript=path)
    assert result["failure_class"] == "no_edit" and result["transcript"] == "e-1.json"
    saved = json.loads(path.read_text())
    assert saved["failure_class"] == "no_edit"
    assert saved["messages"][1]["calls"][0]["name"] == "read_file"
    assert saved["final_answer"].startswith("x is 1")
