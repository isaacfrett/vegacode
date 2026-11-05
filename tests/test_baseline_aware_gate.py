"""baseline_aware_gate (opt-in): tests already failing before Vega's first edit are reported once
as pre-existing, not fed back as work; a test that went from passing to failing still blocks.

Scripted fake models; the sandbox is replaced by a plain pytest run in the project (what is
tested is the baseline and the split of later results, not the sandbox).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from switches import switch_off
from test_agent_loop import agent_with, call, text

from vega import baseline_gate, features, sandbox

TASK = ("The basket total is wrong: `total([(\"pen\", 2.0, 3)])` returns 2.0 instead of 6.0. "
        "The quantity should count. Fix it in shop/prices.py.")
UNRELATED = "tests/test_names.py::test_label"
SUITE = "python -m pytest -q --tb=short"  # the project's test command as the sandbox runs it


@pytest.fixture(autouse=True)
def agent_loop_with_gate(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The scripted task goes through the agent loop (the pipelines and the clarify gate that
    would take it are off); baseline_aware_gate is on; checks run as plain pytest. Returns the
    commands the sandbox ran."""
    switch_off(monkeypatch, "direct_pipeline", "deliverable_pipeline", "project_pipeline",
               "web_pipeline", "clarify_first")
    monkeypatch.setenv(features.ENV_ON, "baseline_aware_gate")
    ran: list[str] = []

    def run_check(command: str, project: Path, **_kwargs) -> str:
        ran.append(command)
        argv = shlex.split(command)
        argv[0] = sys.executable
        env = {**os.environ, "PYTHONPATH": str(project), "PYTHONDONTWRITEBYTECODE": "1"}
        done = subprocess.run(argv + ["-p", "no:cacheprovider"], cwd=project, env=env,
                              capture_output=True, text=True, timeout=120, check=False)
        return f"exit code: {done.returncode}\n{(done.stdout + done.stderr).strip()}"

    monkeypatch.setattr(sandbox, "run_check", run_check)
    return ran


def shop(tmp_path: Path) -> Path:
    """A project whose suite is red before any edit: test_label fails for a reason of its own
    (unrelated to the task); test_rounding passes."""
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "prices.py").write_text(
        "def total(items):\n    return sum(price for _name, price, _qty in items)\n\n\n"
        "def rounded(value):\n    return round(value, 2)\n")
    (tmp_path / "shop" / "names.py").write_text("def label(name):\n    return name.lower()\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_names.py").write_text(
        "from shop.names import label\n\n\ndef test_label():\n    assert label(\"pen\") == \"Pen\"\n")
    (tmp_path / "tests" / "test_prices.py").write_text(
        "from shop.prices import rounded\n\n\ndef test_rounding():\n"
        "    assert rounded(1.005) in (1.0, 1.01)\n")
    return tmp_path


FIX = call("edit_file", path="shop/prices.py", old_text="sum(price for _name, price, _qty in items)",
           new_text="sum(price * qty for _name, price, qty in items)")
# fixes the task and breaks rounded(), which test_rounding covered (passing before the edit)
FIX_AND_BREAK = call("edit_file", path="shop/prices.py",
                     old_text="sum(price for _name, price, _qty in items)\n\n\n"
                              "def rounded(value):\n    return round(value, 2)",
                     new_text="sum(price * qty for _name, price, qty in items)\n\n\n"
                              "def rounded(value):\n    return None")


def user_notes(seen: list[list[dict]]) -> list[str]:
    """Every harness note sent back to the model, in order (after the task itself)."""
    last = seen[-1] if seen else []
    return [m["content"] for m in last if m.get("role") == "user"][1:]


def tool_results(seen: list[list[dict]]) -> list[str]:
    last = seen[-1] if seen else []
    return [m["content"] for m in last if m.get("role") == "tool"]


def answers(*replies: str) -> list:
    return [text(reply) for reply in replies]


# -- the agent loop ------------------------------------------------------------------------------

def test_red_baseline_unrelated_failure_is_reported_once_not_pushed(
        tmp_path: Path, agent_loop_with_gate: list[str]) -> None:
    project = shop(tmp_path)
    final = "Fixed total(). tests/test_names.py::test_label was failing before my change."
    agent, seen = agent_with(project, [FIX, *answers("Fixed total().", final, "Done.", "Done.")])
    answer = agent.run(TASK)
    notes = user_notes(seen)
    assert answer == final
    assert len(notes) == 1  # one report, then the answer is accepted
    assert ("the only failures are 1 test(s) that already failed before your first edit: "
            f"{UNRELATED}") in notes[0]
    assert "check(s) fail" not in notes[0] and "Not finished" not in notes[0]
    # the baseline is one run of the command acceptance uses, before the first edit
    assert agent_loop_with_gate[0] == SUITE
    assert agent._test_baseline is not None and agent._test_baseline.how == "run"
    assert agent._test_baseline.failing == {UNRELATED}
    assert agent.last_acceptance.ok
    assert (project / "shop" / "names.py").read_text().endswith("name.lower()\n")


def test_a_new_regression_still_blocks_the_finish(tmp_path: Path) -> None:
    project = shop(tmp_path)
    agent, seen = agent_with(project, [
        FIX_AND_BREAK, *answers("Fixed total().", "Fixed total().", "Fixed total().", "Done.",
                                "Done.", "Done."),
    ])
    agent.run(TASK)
    notes = user_notes(seen)
    assert "check(s) fail" in notes[0]  # sent back as work, as before
    assert "1 failure(s) to fix: tests/test_prices.py::test_rounding" in notes[0]
    assert (f"pre-existing: 1 test(s) that already failed before your first edit: {UNRELATED}"
            in notes[0])
    assert any("Not finished" in note and "test_rounding" in note for note in notes[1:])
    assert not agent.last_acceptance.ok


def test_switch_off_pushes_every_failure_as_before(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_loop_with_gate: list[str]) -> None:
    monkeypatch.setenv(features.ENV_ON, "")
    project = shop(tmp_path)
    agent, seen = agent_with(project, [
        FIX, *answers("Fixed total().", "Fixed total().", "Fixed total().", "Done.", "Done.",
                      "Done."),
    ])
    agent.run(TASK)
    notes = user_notes(seen)
    assert "1 of 4 check(s) fail" in notes[0] and "`python -m pytest -q`: exit code 1" in notes[0]
    assert not any("pre-existing" in note or "before your first edit" in note for note in notes)
    assert any("Not finished" in note for note in notes)
    assert agent._test_baseline is None
    assert agent_loop_with_gate.count(SUITE) == 1  # acceptance only: no baseline run
    assert not agent.last_acceptance.ok


def test_the_models_own_run_before_any_edit_is_reused(
        tmp_path: Path, agent_loop_with_gate: list[str]) -> None:
    project = shop(tmp_path)
    run = call("run_check", command="python -m pytest -q")
    agent, seen = agent_with(project, [
        run, FIX, run, *answers("Fixed total(); test_label failed before my change.", "Done.",
                                "Done."),
    ])
    agent.run(TASK)
    results = tool_results(seen)
    assert ("Before any edit, 1 test(s) that already failed before your first edit: "
            f"{UNRELATED}") in results[0]
    after = results[2].split("(Vega harness) Pre-existing")[1]
    assert "(listed earlier)" in after and UNRELATED not in after
    assert agent._test_baseline is not None and agent._test_baseline.how == "reused"
    # no extra baseline run: the model's two runs, then acceptance's
    assert agent_loop_with_gate.count("python -m pytest -q") == 2
    assert agent_loop_with_gate.count(SUITE) == 1


def test_a_baseline_failure_the_task_names_stays_work(tmp_path: Path) -> None:
    project = shop(tmp_path)
    task = ("`label` in shop/names.py should capitalise the name; test_label fails. Also fix "
            "total in shop/prices.py so the quantity counts.")
    agent, seen = agent_with(project, [
        FIX, *answers("Fixed total().", "Fixed total().", "Fixed total().", "Done.", "Done.",
                      "Done."),
    ])
    agent.run(task)
    notes = user_notes(seen)
    assert "check(s) fail" in notes[0]
    assert f"1 failure(s) to fix: {UNRELATED}" in notes[0]
    assert "pre-existing" not in notes[0]


def test_verify_before_finish_reports_pre_existing_failures_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # acceptance off: verify_before_finish is then the check that runs the tests at the finish
    switch_off(monkeypatch, "acceptance")
    project = shop(tmp_path)
    agent, seen = agent_with(project, [FIX, *answers("Fixed total().", "Fixed total(); "
                                                     "test_label was failing before.", "Done.")])
    answer = agent.run(TASK)
    notes = [note for note in user_notes(seen) if "Vega ran" in note]
    assert len(notes) == 1
    assert notes[0].startswith("(Vega harness) Before you finish: Vega ran `python -m pytest -q`."
                               " No test that passed before your first edit fails now")
    assert UNRELATED in notes[0] and "Fix the failures" not in notes[0]
    assert answer.startswith("Fixed total(); test_label")


def test_verify_before_finish_still_pushes_a_regression(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    switch_off(monkeypatch, "acceptance")
    project = shop(tmp_path)
    agent, seen = agent_with(project, [FIX_AND_BREAK, *answers("Fixed.", "Fixed.", "Fixed.",
                                                               "Done.", "Done.")])
    agent.run(TASK)
    note = next(note for note in user_notes(seen) if "Vega ran" in note)
    assert "Fix the failures" in note
    assert "New since your first edit (fix these): tests/test_prices.py::test_rounding" in note
    assert f"Pre-existing, not caused by your changes: 1 test(s) that already failed before " \
        f"your first edit: {UNRELATED}" in note


# -- the helpers ---------------------------------------------------------------------------------

RED = """exit code: 1
.F.F                                                                     [100%]
=========================== short test summary info ============================
FAILED tests/test_a.py::test_one - assert 1 == 2
FAILED tests/test_b.py::TestB::test_two[x-1] - KeyError: 'x'
2 failed, 2 passed in 0.10s"""


def test_parse_names_every_failure() -> None:
    run = baseline_gate.parse(RED)
    assert run is not None
    assert run.ids == ["tests/test_a.py::test_one", "tests/test_b.py::TestB::test_two[x-1]"]
    assert run.complete and run.finished
    green = baseline_gate.parse("exit code: 0\n4 passed in 0.1s")
    assert green is not None and green.ids == []
    assert baseline_gate.parse("Check timed out after 60s.") is None
    assert baseline_gate.parse("exit code: 1\nmake: *** [test] Error 1") is None


def test_parse_flags_incomplete_and_stopped_runs() -> None:
    clipped = baseline_gate.parse(
        RED.replace("FAILED tests/test_a.py::test_one - assert 1 == 2\n", ""))
    assert clipped is not None and not clipped.complete
    stopped = baseline_gate.parse(
        RED.replace("2 failed", "!!!!!!!!!! stopping after 2 failures !!!!!!!!!!\n2 failed"))
    assert stopped is not None and not stopped.finished
    interrupted = baseline_gate.parse(
        "exit code: 2\n=== short test summary info ===\nERROR tests/test_c.py\n"
        "!!!!!!!! Interrupted: 1 error during collection !!!!!!!!\n1 error in 0.1s")
    assert interrupted is not None
    assert interrupted.ids == ["tests/test_c.py"] and not interrupted.finished


@pytest.mark.parametrize("command,whole", [
    ("python -m pytest -q", True), ("pytest", True), ("pytest -q --tb=short", True),
    ("pytest tests/test_a.py", False), ("pytest -k total", False), ("pytest -x", False),
    ("python -m pytest --lf", False), ("python check.py", False), ("make test", False),
])
def test_suite_command(command: str, whole: bool) -> None:
    assert baseline_gate.suite_command(command) is whole


@pytest.mark.parametrize("test_id,named", [
    ("tests/test_names.py::test_label", False),
    ("tests/test_prices.py::test_total", True),  # module prices.py named
    ("tests/test_cart.py::test_parse_date_empty", True),  # `parse_date` inside the test name
    ("tests/test_cart.py::test_parse_dates", False),  # not a whole part of the name
    ("tests/test_cart.py::test_checkout", True),  # the test itself named
])
def test_targets(test_id: str, named: bool) -> None:
    task = ("`parse_date` crashes on an empty string; fix it in shop/prices.py. "
            "test_checkout should pass too.")
    assert baseline_gate.targets(test_id, task) is named


def test_classify_never_hides_a_new_failure() -> None:
    baseline = baseline_gate.Baseline("pytest", frozenset({"tests/test_a.py::test_one"}), "run")
    split = baseline_gate.classify(baseline, RED, "Fix the totals.")
    assert split is not None
    assert split.pre == ["tests/test_a.py::test_one"]
    assert split.new == ["tests/test_b.py::TestB::test_two[x-1]"]
    assert not split.only_pre
    note = baseline_gate.note(baseline, split)
    assert "New since your first edit (fix these): tests/test_b.py::TestB::test_two[x-1]" in note
    assert "tests/test_a.py::test_one" in note  # listed once ...
    assert "(listed earlier)" in baseline_gate.note(baseline, split)  # ... then referred to
    green = baseline_gate.Baseline("pytest", frozenset(), "run")
    unknown = baseline_gate.classify(green, RED, "Fix the totals.")
    assert unknown is not None and unknown.pre == [] and len(unknown.new) == 2
