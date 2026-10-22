"""plan_churn_guard: re-editing the plan with no work in between is met with "do the next step"."""

from __future__ import annotations

from pathlib import Path

from switches import switch_off
from test_agent_loop import agent_with, call, text

from vega import features


def plan(*items: tuple[str, str]):
    return call("update_plan", items=[{"text": t, "status": s} for t, s in items])


def tool_results(agent) -> list[str]:
    return [m["content"] for m in agent.messages if m.get("role") == "tool"]


def churn(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n")
    first = plan(("read a.py", "in_progress"), ("answer", "pending"))
    again = plan(("read a.py", "in_progress"), ("answer", "pending"))
    other = plan(("read a.py first", "in_progress"), ("answer", "pending"))
    return agent_with(tmp_path, [first, again, again, other, call("read_file", path="a.py"),
                                 plan(("read a.py", "done"), ("answer", "in_progress")),
                                 text("x is 1")])


def test_off_by_default_plans_are_just_updated(tmp_path: Path) -> None:
    agent, _ = churn(tmp_path)
    agent.run("What is x in a.py? Plan it first.")
    assert not any("Vega harness" in r for r in tool_results(agent))


def test_guard_notes_then_refuses_then_resets_after_work(tmp_path: Path, monkeypatch) -> None:
    # a question: the question pipeline would answer it; this tests the agent loop
    switch_off(monkeypatch, "question_pipeline")
    monkeypatch.setenv(features.ENV_ON, "plan_churn_guard")
    agent, _ = churn(tmp_path)
    assert agent.run("What is x in a.py? Plan it first.") == "x is 1"
    results = tool_results(agent)
    assert "Vega harness" not in results[0]
    assert "Do the next step now with the other tools: read a.py" in results[1]
    assert "Do the next step now" in results[2]
    assert results[3].startswith("(Vega harness) The plan was not changed")
    assert agent.plan[0]["text"] == "read a.py"  # the refused 4th update ("read a.py first") was dropped
    assert "x = 1" in results[4]
    assert "Vega harness" not in results[5]  # after real work, an update is just an update
