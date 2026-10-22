"""Opt-in constrained actions (E11): every reply is one JSON action; no native tool calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer

from vega import evals, features, lab
from vega.agent import (
    ACT_NUDGE,
    CUT_OFF,
    FINAL_ACTION,
    INVALID_ACTION_ANSWER,
    INVALID_ACTION_REASON,
    MAX_INVALID_ACTIONS,
    PLAN_UNFINISHED,
    UNSAVED_CLAIM,
    CodingAgent,
    action_schema,
    actions_section,
    parse_action,
    to_action_messages,
)
from vega.context import estimate_text_tokens
from vega.tools import tool_specs


@pytest.fixture(autouse=True)
def constrained_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(features.ENV, raising=False)
    monkeypatch.setenv(features.ENV_ON, "constrained")


def act(action: str, **args: Any) -> str:
    return json.dumps({"action": action, "args": args})


def final(answer: str) -> str:
    return json.dumps({"action": FINAL_ACTION, "answer": answer})


def constrained_agent(
    project: Path, replies: list[str | tuple[str, str]], **options: Any,
) -> tuple[CodingAgent, list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    """An agent whose fake server enforces the constrained request shape on every call."""
    events: list[tuple[str, dict[str, Any]]] = []
    agent = CodingAgent(project, allow_commands=False, on_event=lambda k, d: events.append((k, d)),
                        **options)
    seen: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        assert "response_format" in kwargs and "tools" not in kwargs
        assert "tool_choice" not in kwargs and "stream" not in kwargs
        assert kwargs["response_format"]["json_schema"]["name"] == "vega_action"
        assert all(m["role"] in {"system", "user", "assistant"} for m in kwargs["messages"])
        seen.append(kwargs)
        reply = replies.pop(0)
        content, reason = reply if isinstance(reply, tuple) else (reply, "stop")
        message = SimpleNamespace(content=content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=reason)])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, seen, events


def test_multi_step_edit_trajectory_in_json_actions(tmp_path: Path) -> None:
    (tmp_path / "math_ops.py").write_text("def difference(a, b):\n    return a + b\n")
    agent, seen, events = constrained_agent(tmp_path, [
        act("search_text", query="difference"),
        act("read_file", path="math_ops.py"),
        act("edit_file", path="math_ops.py", old_text="a + b", new_text="a - b"),
        final("Fixed difference() in math_ops.py."),
    ])
    assert agent.run("Fix the bug in difference().") == "Fixed difference() in math_ops.py."
    assert (tmp_path / "math_ops.py").read_text().endswith("a - b\n")
    assert [d["name"] for kind, d in events if kind == "tool_call"] == [
        "search_text", "read_file", "edit_file"]
    last = seen[-1]["messages"]
    assert last[-2] == {"role": "assistant", "content": act(
        "edit_file", path="math_ops.py", old_text="a + b", new_text="a - b")}
    assert last[-1]["role"] == "user"
    assert last[-1]["content"].startswith("Tool result (edit_file):\nEdited math_ops.py")
    assert "Tool result (read_file):" in last[-3]["content"]
    # Stored history keeps the native shape, so compaction and transcripts read it unchanged.
    assert [m["role"] for m in agent.messages[-3:]] == ["assistant", "tool", "assistant"]
    assert agent.messages[-1] == {
        "role": "assistant", "content": "Fixed difference() in math_ops.py."}


def test_final_answer_ends_the_turn(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    agent, seen, events = constrained_agent(
        tmp_path, [act("read_file", path="a.txt"), final("It contains x.")])
    assert agent.run("What is in a.txt?") == "It contains x."
    assert len(seen) == 2
    assert ("final", {"text": "It contains x.", "streamed": False}) in events


def test_schema_has_a_branch_per_offered_tool(tmp_path: Path) -> None:
    agent, seen, _ = constrained_agent(tmp_path, [act("list_files"), final("Done.")])
    agent.run("list")
    response_format = seen[0]["response_format"]
    assert response_format["type"] == "json_schema" and response_format["json_schema"]["strict"]
    branches = response_format["json_schema"]["schema"]["anyOf"]
    names = [branch["properties"]["action"]["enum"][0] for branch in branches]
    offered = [spec["function"]["name"] for spec in
               tool_specs(include_plan=False, include_definition=features.on("edit_by_name"))]
    assert names == offered + [FINAL_ACTION]
    edit = branches[names.index("edit_file")]
    assert edit["required"] == ["action", "args"]
    assert edit["properties"]["args"]["required"] == ["path", "old_text", "new_text"]
    assert edit["properties"]["args"]["additionalProperties"] is False
    assert branches[-1]["required"] == ["action", "answer"]


def test_tool_list_replaces_native_specs_within_budget(tmp_path: Path) -> None:
    agent, seen, _ = constrained_agent(tmp_path, [final("Hello."), final("Hello again.")])
    agent.run("hi")
    system = seen[0]["messages"][0]["content"]
    tools = tool_specs(include_plan=False, include_definition=features.on("edit_by_name"))
    section = actions_section(tools)
    assert system.endswith(section)
    assert "- edit_file {path, old_text, new_text}:" in section
    assert "- read_file {path, start_line?: int, end_line?: int}:" in section
    assert estimate_text_tokens(section) <= estimate_text_tokens(json.dumps(tools))
    assert "update_plan {items: [{text, status: pending|in_progress|done}]}" in actions_section(
        tool_specs())
    assert agent.messages[0]["content"] + section == system  # stored prompt is unchanged


@pytest.mark.parametrize("bad, error", [
    ("edit the file please", "not valid JSON"),
    (json.dumps({"action": "teleport", "args": {}}), "unknown action 'teleport'"),
    (act("edit_file", path="a.txt"), "missing required argument(s): old_text, new_text"),
    (act("read_file", path="a.txt", lines=3), "has no argument 'lines'"),
    (json.dumps({"action": FINAL_ACTION, "answer": ""}), "non-empty"),
])
def test_invalid_actions_are_sent_back(tmp_path: Path, bad: str, error: str) -> None:
    (tmp_path / "a.txt").write_text("x")
    agent, seen, events = constrained_agent(tmp_path, [
        act("read_file", path="a.txt"), bad,
        act("edit_file", path="a.txt", old_text="x", new_text="y"), final("Done."),
    ])
    assert agent.run("Change x to y") == "Done."
    assert (tmp_path / "a.txt").read_text() == "y"
    note = seen[2]["messages"][-1]["content"]
    assert note.startswith("(Vega harness) Your last reply was not a valid action")
    assert error in note
    assert seen[2]["messages"][-2] == {"role": "assistant", "content": bad}  # resent verbatim
    assert ("discarded", {"reason": INVALID_ACTION_REASON, "retry": True}) in events


def test_invalid_actions_are_bounded_and_never_the_answer(tmp_path: Path) -> None:
    agent, seen, _ = constrained_agent(tmp_path, ["not json"] * (MAX_INVALID_ACTIONS + 1))
    assert agent.run("Change x to y") == INVALID_ACTION_ANSWER
    assert len(seen) == MAX_INVALID_ACTIONS + 1


def test_act_nudge_applies_once_to_a_final_answer_without_tools(tmp_path: Path,
                                                                 monkeypatch) -> None:
    # finish_gate_no_edit off: "Still unsure." with no edit would be sent back once more
    monkeypatch.setenv(features.ENV, os.environ.get(features.ENV, "") + ",finish_gate_no_edit")
    agent, seen, _ = constrained_agent(tmp_path, [final("Which file?"), final("Still unsure.")])
    assert agent.run("Fix the bug") == "Still unsure."
    assert seen[1]["messages"][-1] == {"role": "user", "content": ACT_NUDGE}
    assert seen[1]["messages"][-2] == {"role": "assistant", "content": final("Which file?")}


def test_unsaved_claim_still_sent_back(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, seen, _ = constrained_agent(tmp_path, [
        act("read_file", path="a.py"), final("I fixed it."),
        act("edit_file", path="a.py", old_text="x = 1", new_text="x = 2"), final("I fixed it."),
    ])
    assert agent.run("Set x to 2") == "I fixed it."
    assert seen[2]["messages"][-1]["content"] == UNSAVED_CLAIM


def test_plan_update_via_action_and_unfinished_plan_nudge(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    items = [{"text": "Change x to y", "status": "in_progress"},
             {"text": "Check", "status": "pending"}]
    agent, seen, events = constrained_agent(tmp_path, [
        act("update_plan", items=items),
        act("edit_file", path="a.txt", old_text="x", new_text="y"),
        final("Changed x to y."),
        act("update_plan", items=[{**item, "status": "done"} for item in items]),
        final("Changed x to y and checked."),
    ], planning="on")
    assert agent.run("change x to y") == "Changed x to y and checked."
    assert any(kind == "plan" for kind, _ in events)
    assert "update_plan" in json.dumps(seen[0]["response_format"])
    assert "Current plan" in seen[1]["messages"][-1]["content"]
    assert any(str(m.get("content", "")).startswith(PLAN_UNFINISHED) for m in seen[3]["messages"])
    assert all(item["status"] == "done" for item in agent.plan)


def test_review_and_undo_hooks_fire_on_writes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    reviewed: list[str] = []

    def approve(plan: Any) -> bool:
        reviewed.append(plan.relative)
        return True

    agent, _, _ = constrained_agent(tmp_path, [
        act("edit_file", path="a.txt", old_text="x", new_text="y"), final("Edited a.txt."),
    ], review=True, approve=approve, undo_depth=1)
    assert agent.run("change x to y") == "Edited a.txt."
    assert reviewed == ["a.txt"] and (tmp_path / "a.txt").read_text() == "y"
    report = agent.undo()
    assert report is not None and report.applied and (tmp_path / "a.txt").read_text() == "x"


def test_declined_write_is_reported_back(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    agent, seen, _ = constrained_agent(tmp_path, [
        act("edit_file", path="a.txt", old_text="x", new_text="y"), final("Not changed."),
    ], review=True, approve=lambda plan: False)
    agent.run("change x to y")
    assert (tmp_path / "a.txt").read_text() == "x"
    assert seen[1]["messages"][-1]["content"].startswith("Tool result (edit_file):\nUser declined")


def test_cut_off_reply_is_sent_back_verbatim(tmp_path: Path) -> None:
    runaway = '{"action": "write_file", "args": {"path": "a.py", "content": "x x x'
    agent, seen, _ = constrained_agent(tmp_path, [
        act("list_files"), (runaway, "length"), final("All done."),
    ])
    assert agent.run("do it") == "All done."
    assert seen[2]["messages"][-2:] == [
        {"role": "assistant", "content": runaway}, {"role": "user", "content": CUT_OFF}]


def test_streaming_is_not_used_and_sampling_overrides_pass_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VEGA_SEED", "7")
    monkeypatch.setenv("VEGA_TEMPERATURE", "0")
    monkeypatch.delenv("VEGA_MAX_TOKENS", raising=False)
    agent, seen, events = constrained_agent(
        tmp_path, [act("list_files"), final("Done.")], stream=True)
    assert agent.run("list") == "Done."
    assert all(r["seed"] == 7 and r["temperature"] == 0.0 and r["max_tokens"] == 2048 for r in seen)
    assert not any(kind == "text" for kind, _ in events)  # the JSON is never rendered as text


def test_to_action_messages_relabels_history() -> None:
    history = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "(Vega harness) Summary of earlier work"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "read_file", "arguments": '{"path": "a.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "1\tx"},
        {"role": "assistant", "content": "{not json", "vega_raw": True},
        {"role": "assistant", "content": "The answer."},
    ]
    assert to_action_messages(history) == [
        history[0], history[1],
        {"role": "assistant", "content": '{"action": "read_file", "args": {"path": "a.py"}}'},
        {"role": "user", "content": "Tool result (read_file):\n1\tx"},
        {"role": "assistant", "content": "{not json"},
        {"role": "assistant", "content": final("The answer.")},
    ]


def test_parse_action_details() -> None:
    tools = tool_specs()
    assert parse_action('```json\n{"action": "read_file", "args": {"path": "a.py", '
                        '"start_line": "3", "end_line": null}}\n```', tools) == (
        "read_file", {"path": "a.py", "start_line": 3})
    assert parse_action('{"action": "git_status"}', tools) == ("git_status", {})
    assert "type integer" in str(parse_action(act("read_file", path="a", start_line=True), tools))
    assert "type string" in str(parse_action(act("read_file", path=3), tools))
    assert "JSON object" in str(parse_action("[1]", tools))
    assert parse_action(final("ok"), tools) == (FINAL_ACTION, {"answer": "ok"})


def test_action_schema_is_valid_json_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = action_schema(tool_specs())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    assert validator.is_valid(json.loads(act("edit_file", path="a", old_text="b", new_text="c")))
    assert not validator.is_valid(json.loads(act("edit_file", path="a")))
    assert not validator.is_valid({"action": "teleport", "args": {}})
    assert validator.is_valid(json.loads(final("ok")))


# -- switch off: requests are exactly as before ---------------------------------------------------

def test_default_requests_send_tools_and_no_response_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(features.ENV_ON)
    agent = CodingAgent(tmp_path, allow_commands=False)
    seen: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        seen.append(kwargs)
        message = SimpleNamespace(content="Hi.", tool_calls=None, model_dump=lambda **_: {
            "role": "assistant", "content": "Hi."})
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    agent.run("hi")
    assert seen and all("response_format" not in r and r["tool_choice"] == "auto" for r in seen)
    assert "Reply format:" not in seen[0]["messages"][0]["content"]
    assert set(seen[0]) == {"model", "messages", "tools", "tool_choice", "max_tokens"}


# -- switches --------------------------------------------------------------------------------------

def test_opt_in_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(features.ENV_ON)
    assert features.enabled() == frozenset() and not features.on("constrained")
    assert features.on("lint")
    monkeypatch.setenv(features.ENV_ON, " constrained , ")
    assert features.enabled() == {"constrained"} and features.on("constrained")
    monkeypatch.setenv(features.ENV_ON, "constrained,lint")  # lint is a default: accepted, no-op
    assert features.enabled() == {"constrained"}
    monkeypatch.setenv(features.ENV_ON, "constrained,nonsense")
    with pytest.raises(ValueError, match="nonsense"):
        features.enabled()
    monkeypatch.setenv(features.ENV_ON, "")
    monkeypatch.setenv(features.ENV, "constrained")  # opt-in names are not valid OFF names
    with pytest.raises(ValueError, match="constrained"):
        features.disabled()
    with pytest.raises(KeyError):
        features.on("nope")


@pytest.fixture
def two_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cases = tmp_path / "fixes.jsonl"
    cases.write_text(
        '{"id": "a", "task": "t", "files": {"x.txt": "1"}, "response_contains": ["ok"]}\n'
        '{"id": "b", "task": "t", "files": {"x.txt": "1"}, "response_contains": ["ok"]}\n'
    )
    monkeypatch.setattr(evals, "suite_file", lambda suite: cases)
    monkeypatch.setattr(evals, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setattr(evals, "_git_commit", lambda: "abc1234")
    monkeypatch.setattr(evals, "_server_fingerprint", lambda model: {})
    monkeypatch.delenv(features.ENV_ON)
    return tmp_path


def test_run_ab_with_b_on(two_cases: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, bool]] = []

    def fake_attempt(case: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        on = features.on("constrained")
        seen.append((case["id"], on))
        return {"passed": on, "detail": "" if on else "x", "seconds": 1.0, "failure": ""}

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", [], repeat=2, b_on=["constrained"])
    assert seen == [("a", False), ("a", True), ("a", True), ("a", False),
                    ("b", False), ("b", True), ("b", True), ("b", False)]
    assert record["variants"]["A"]["features_on"] == []
    assert record["variants"]["B"]["features_on"] == ["constrained"]
    assert record["variants"]["B"]["features_off"] == []
    assert record["paired"]["b_only"] == 4
    assert features.ENV_ON not in os.environ and features.ENV not in os.environ  # restored
    saved = json.loads((two_cases / "runs.jsonl").read_text())
    assert saved["variants"]["B"]["features_on"] == ["constrained"]
    assert "B (feature on) better" in lab.verdict({**saved, "paired": {
        **saved["paired"], "mcnemar_p": 0.01, "diff_ci": [0.2, 0.9], "b_only": 9, "a_only": 0}})
    with pytest.raises(typer.BadParameter, match="opt-in"):
        evals.run_ab("fixes", [], b_on=["lint"])
    with pytest.raises(typer.BadParameter, match="B must"):
        evals.run_ab("fixes", [])


def test_run_suite_records_features_on(two_cases: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evals, "run_attempt", lambda case, **kwargs: {
        "passed": True, "detail": "", "seconds": 1.0, "failure": ""})
    monkeypatch.setenv(features.ENV_ON, "constrained")
    record = evals.run_suite("fixes", repeat=1)
    assert record["features_on"] == ["constrained"] and record["features_off"] == []


def test_lab_experiment_with_ab_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "experiments.toml").write_text(
        '[[experiment]]\nid = "constrained"\nhypothesis = "h"\nab_on = ["constrained"]\n')
    monkeypatch.setattr(lab, "EXPERIMENTS_FILE", tmp_path / "experiments.toml")
    monkeypatch.setattr(lab, "SCOREBOARD_FILE", tmp_path / "scoreboard.md")
    monkeypatch.setattr(evals, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.delenv("VEGA_EXPERIMENT", raising=False)
    calls: list[tuple[str, list[str], list[str]]] = []

    def fake_ab(suite: str, b_off: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append((suite, b_off, kwargs.get("b_on")))
        return {"stopped": "test"}  # stop the queue before the scoreboard

    monkeypatch.setattr(evals, "run_ab", fake_ab)
    [experiment] = lab.load()
    assert experiment.ab == [] and experiment.ab_on == ["constrained"]
    lab.run_queue()
    assert calls == [("fixes", [], ["constrained"])]


@pytest.mark.skipif(not lab.EXPERIMENTS_FILE.exists(), reason="eval data not present")
def test_repo_constrained_experiment_is_declared() -> None:
    experiments = {experiment.id: experiment for experiment in lab.load()}
    assert experiments["constrained"].ab_on == ["constrained"]
    assert experiments["constrained"].suite == "fixes"
    known = set(features.OPT_IN) | set(features.FEATURES)
    for experiment in experiments.values():  # retired switches stay in the records of past runs
        assert set(experiment.ab_on) <= known | set(features.RETIRED)


def test_a_retired_switch_is_a_named_error(monkeypatch) -> None:
    assert not set(features.RETIRED) & (set(features.OPT_IN) | set(features.FEATURES))
    monkeypatch.setenv(features.ENV_ON, "replace_function")
    with pytest.raises(ValueError, match="replace_function was retired on 2026-10-17: replaced by "
                                         "edit_by_name"):
        features.on("search_streak")
    monkeypatch.delenv(features.ENV_ON)
    monkeypatch.setenv(features.ENV, "shape_routing")
    with pytest.raises(ValueError, match="shape_routing was retired"):
        features.on("plan")
