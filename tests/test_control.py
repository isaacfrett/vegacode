"""Stop/resume control gate: scripted interruptions of a CodingAgent turn (tests/control_scenarios.py)
graded from the run's record. No model is called. Known product bugs are strict xfails, so a fix
turns them into failures here until the `known_bug` note is removed."""

from __future__ import annotations

import time
from pathlib import Path

import control_scenarios as control
import pytest

from vega import server


@pytest.fixture(autouse=True)
def scratch_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "vega-home"
    monkeypatch.setenv("VEGA_HOME", str(home))
    monkeypatch.setattr(server, "STATE_DIR", home)


def _params() -> list:
    return [pytest.param(item, id=item.id, marks=[pytest.mark.xfail(
        strict=True, reason=item.known_bug)] if item.known_bug else [])
        for item in control.SCENARIOS]


@pytest.mark.parametrize("item", _params())
def test_scenario(item: control.Scenario) -> None:
    outcome = control.run_scenario(item)
    if outcome.skipped:
        pytest.skip(outcome.skipped)
    assert outcome.problems == []


def test_the_table_covers_every_point_and_level() -> None:
    ids = [item.id for item in control.SCENARIOS]
    assert len(ids) >= 20 and len(set(ids)) == len(ids)
    assert {item.point for item in control.SCENARIOS} == set(control.POINTS)
    levels = [item.complexity for item in control.SCENARIOS]
    assert all(levels.count(level) >= 5 for level in control.COMPLEXITY)


def _fingerprint(outcomes: list[control.Outcome]) -> list[tuple]:
    return [(o.id, o.passed, bool(o.skipped), tuple(o.tools), o.requests, tuple(o.problems))
            for o in outcomes]


def test_runs_are_deterministic() -> None:
    first = control.run_all()
    second = control.run_all()
    assert _fingerprint(first) == _fingerprint(second)


# -- the graders catch violations ----------------------------------------------------------------

def test_a_stop_the_agent_ignores_fails_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """An agent that keeps working after Esc: the next request and tool call are counted."""
    from vega.agent import CodingAgent

    monkeypatch.setattr(CodingAgent, "request_stop", lambda self: None)
    item = next(i for i in control.SCENARIOS if i.id == "ctl-stream-between")
    problems = " | ".join(control.run_scenario(item).problems)
    assert "model request(s) started after the stop" in problems
    assert "not STOPPED" in problems


def test_history_grader_finds_unpaired_calls() -> None:
    call = {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "read_file"}}]}
    assert control.history_problem([{"role": "system"}, call]) == "tool calls without a result: ['a']"
    assert "no open tool call" in control.history_problem(
        [{"role": "tool", "tool_call_id": "zz", "content": "x"}])
    assert "before results" in control.history_problem([call, {"role": "user", "content": "hi"}])
    assert control.history_problem([call, {"role": "tool", "tool_call_id": "a"}]) == ""


def test_file_grader_finds_a_partial_write(tmp_path: Path) -> None:
    run = control.Run(tmp_path, before={"a.py": "x = 1\n"})
    run.allowed["b.py"] = {"y = 2\nz = 3\n"}
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")  # half of the write
    assert "b.py: neither" in control.check_files(run)
    (tmp_path / "b.py").write_text("y = 2\nz = 3\n")
    assert control.check_files(run) == ""


def test_latency_grader_uses_the_stop_time(tmp_path: Path) -> None:
    run = control.Run(tmp_path, stop_at=time.monotonic() - 2.0, returned_at=time.monotonic())
    assert "limit" in control.check_latency(run)


# -- stop reaches delegate_task sub-agents --------------------------------------------------------

def test_a_child_whose_parent_is_already_stopped_sends_nothing(tmp_path: Path) -> None:
    """A sub-agent started after the parent's Esc (its own run() clears its flag) still stops."""
    from vega.agent import STOPPED, CodingAgent

    run = control.Run(tmp_path)
    parent = control.make_agent(run, [control.say("Done.")])
    child = CodingAgent(tmp_path, can_write=False, can_delegate=False, allow_commands=False)
    child.client = control.FakeClient(run, [control.say("Child done.")], label="child")
    child.route_requests = False
    child._parent_stop = parent._stop
    parent.request_stop()
    assert child.run("survey the repo") == STOPPED
    assert run.requests == []
    assert parent.stop_requested  # the child's run() never clears its parent's flag


def test_the_stopped_child_reports_stopped_and_is_released(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vega import agent as agent_module
    from vega.agent import STOPPED

    (tmp_path / "a.py").write_text("x = 1\n")
    run = control.Run(tmp_path)
    parent = control.make_agent(run, [control.calls(("delegate_task", {"task": "read a.py"})),
                                      control.say("Done.")])
    child_script = [control.Hook(control.calls(control.read("a.py")), before=lambda r: r.stop()),
                    control.say("Summary.")]
    monkeypatch.setattr(agent_module, "OpenAI",
                        lambda *a, **k: control.FakeClient(run, child_script, label="child"))
    assert run.turn("summarise") == STOPPED
    delegated = [r["result"] for r in run.results if r["name"] == "delegate_task"]
    assert delegated == [STOPPED]
    assert parent._child is None
    assert [r["who"] for r in run.requests] == ["main", "child"]
