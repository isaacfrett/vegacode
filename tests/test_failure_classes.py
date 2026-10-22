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


# Modelled on real eval records (2026-10-20/21 goals, engineering and selfbuild runs).
GOAL_CASE = EDIT_CASE | {"expected_files": ["app/http.py", "app/settings.py", "tests/test_app.py"]}
SELFBUILD_CASE = EDIT_CASE | {"expected_files": ["src/vega/agent.py", "tests/test_agent_loop.py"]}
QUILL_CASE = EDIT_CASE | {"expected_files": ["quillbook/model.py", "quillbook/posts.py",
                                              "quillbook/authors.py", "tests/test_quillbook.py"]}
CHECK_FAILED = "check_command exit code: 1: 5 failed, 3 passed in 0.1s"


def changed(case: dict, *files: str, **fields) -> dict:
    """An attempt whose diff touched `files`, with unexpected_files built as run_attempt does."""
    base = {"detail": CHECK_FAILED, "tool_calls": {"read_file": 2, "edit_file": 3},
            "diff": {"files": list(files), "added": 10, "removed": 2},
            "unexpected_files": evals._unexpected_files(case.get("expected_files"), list(files))}
    return attempt(**(base | fields))


def test_expected_source_edited_plus_an_extra_is_not_wrong_file() -> None:
    # a goals attempt: edited both expected sources and also the caller app/sync.py
    record = changed(GOAL_CASE, "app/http.py", "app/settings.py", "app/sync.py")
    assert record["unexpected_files"] == ["app/sync.py"]
    assert classify_failure(GOAL_CASE, record, "Done.", 11, 0) == "wrong_fix"
    assert evals._extra_files(GOAL_CASE, record) == ["app/sync.py"]
    broken = changed(GOAL_CASE, "app/http.py", "app/__init__.py",
                     detail="check_command exit code: 1: SyntaxError: invalid syntax")
    assert classify_failure(GOAL_CASE, broken, "Done.", 2, 0) == "broken_code"


def test_only_unexpected_files_edited_is_wrong_file() -> None:
    # a selfbuild attempt: expected src/vega/agent.py, the agent edited tools.py
    record = changed(SELFBUILD_CASE, "src/vega/tools.py")
    assert classify_failure(SELFBUILD_CASE, record, "Done.", 2, 0) == "wrong_file"
    assert evals._extra_files(SELFBUILD_CASE, record) == []
    # another: only eval data (and Vega's notes) changed
    data_only = changed(SELFBUILD_CASE, ".vega/notes.md", "data/cases.jsonl")
    assert data_only["unexpected_files"] == ["data/cases.jsonl"]
    assert classify_failure(SELFBUILD_CASE, data_only, "Done.", 2, 0) == "wrong_file"


def test_expected_test_file_alone_does_not_excuse_a_wrong_source_file() -> None:
    record = changed(SELFBUILD_CASE, "tests/test_agent_loop.py", "src/vega/tools.py")
    assert classify_failure(SELFBUILD_CASE, record, "Done.", 2, 0) == "wrong_file"


def test_new_module_with_a_near_name_stays_wrong_file() -> None:
    # an engineering attempt: expected quillbook/model.py, the agent created quillbook/models.py
    record = changed(QUILL_CASE, "quillbook/__init__.py", "quillbook/models.py")
    assert classify_failure(QUILL_CASE, record, "Done.", 12, 0) == "wrong_file"


def test_repro_first_test_is_never_unexpected() -> None:
    record = changed(SELFBUILD_CASE, "src/vega/agent.py", "tests/test_vega_repro.py")
    assert record["unexpected_files"] == []
    assert classify_failure(SELFBUILD_CASE, record, "Done.", 2, 0) == "wrong_fix"
    # an older record that still lists it, with nothing else changed but the repro test
    old = changed(SELFBUILD_CASE, "tests/test_vega_repro.py",
                  unexpected_files=["tests/test_vega_repro.py"])
    assert classify_failure(SELFBUILD_CASE, old, "Done.", 1, 0) == "wrong_fix"


def test_forbidden_file_modified_is_still_wrong_file() -> None:
    record = changed(GOAL_CASE, "app/http.py", detail="app/sync.py was modified")
    assert classify_failure(GOAL_CASE, record, "Done.", 1, 0) == "wrong_file"


def test_empty_diff_after_writes_is_no_edit() -> None:
    # a fixes attempt: two writes that left the project unchanged, labelled wrong_fix before
    record = attempt(detail="python_check failed: AssertionError: %m/%d/%Y",
                     tool_calls={"read_file": 1, "edit_file": 2},
                     diff={"files": [], "added": 0, "removed": 0})
    assert classify_failure(EDIT_CASE, record, "Fixed the locale.", 2, 0) == "no_edit"
    # the earlier, more specific classes still win
    assert classify_failure(EDIT_CASE, record | {"hit_step_limit": True}, "", 2, 0) == "step_limit"
    assert classify_failure(EDIT_CASE, record | {"model_error": True}, "", 2, 0) == "model_error"
    cut = "Stopped: the model's replies kept hitting the length limit"
    assert classify_failure(EDIT_CASE, record, cut, 2, 0) == "cut_off"
    assert classify_failure(EDIT_CASE, record | {"failure": "regression"}, "", 2, 0) == "regression"


def test_external_agent_uses_the_same_wrong_file_rule() -> None:
    record = changed(GOAL_CASE, "app/http.py", "app/sync.py", tool_calls={})
    assert evals._classify_external(GOAL_CASE, record, "Done.") == "wrong_fix"
    record = changed(GOAL_CASE, "app/sync.py", tool_calls={})
    assert evals._classify_external(GOAL_CASE, record, "Done.") == "wrong_file"
