"""repro_first (opt-in): a bug report on the agent loop is reproduced with one failing test before
the first source edit.

Without it, attempts on the agent loop typically edited source before any check and did not
write a test that reproduced the report. Scripted
fake models; the sandbox is replaced by a plain pytest run in the project (the hold, the run when
the repro is written, and the keep/remove decision are what is tested, not the sandbox).
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

from vega import features, repro_first, sandbox

TASK = ("The basket total is wrong: `total([(\"pen\", 2.0, 3)])` returns 2.0 instead of 6.0. "
        "The quantity should count. Fix it.")
REPRO = "tests/test_vega_repro.py"
REPRO_TEST = ("from shop.prices import total\n\n\n"
              "def test_quantity_counts():\n    assert total([(\"pen\", 2.0, 3)]) == 6.0\n")
HOLD = "Not run yet: reproduce the reported bug"


@pytest.fixture(autouse=True)
def agent_loop_with_repro_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scripted bug report goes through the agent loop (the pipelines and the clarify gate
    that would take it are off); repro_first is switched on; checks run as plain pytest."""
    switch_off(monkeypatch, "direct_pipeline", "deliverable_pipeline", "project_pipeline",
               "web_pipeline", "clarify_first")
    monkeypatch.setenv(features.ENV_ON, "repro_first")

    def run_check(command: str, project: Path, **_kwargs) -> str:
        argv = shlex.split(command)
        argv[0] = sys.executable
        env = {**os.environ, "PYTHONPATH": str(project), "PYTHONDONTWRITEBYTECODE": "1"}
        done = subprocess.run(argv + ["-p", "no:cacheprovider"], cwd=project, env=env,
                              capture_output=True, text=True, timeout=120, check=False)
        return f"exit code: {done.returncode}\n{(done.stdout + done.stderr).strip()}"

    monkeypatch.setattr(sandbox, "run_check", run_check)


def shop(tmp_path: Path) -> Path:
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "prices.py").write_text(
        "def total(items):\n    return sum(price for _name, price, _qty in items)\n")
    (tmp_path / "shop" / "names.py").write_text("def label(name):\n    return name.title()\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_names.py").write_text(
        "from shop.names import label\n\n\ndef test_label():\n    assert label(\"pen\") == \"Pen\"\n")
    return tmp_path


FIX = call("edit_file", path="shop/prices.py", old_text="sum(price for _name, price, _qty in items)",
           new_text="sum(price * qty for _name, price, qty in items)")


def tool_results(seen: list[list[dict]]) -> list[str]:
    """Every tool result the model was shown, in order (each request repeats the history)."""
    last = seen[-1] if seen else []
    return [m["content"] for m in last if m.get("role") == "tool"]


# -- when it applies ---------------------------------------------------------------------------

ROUTE = {"name": "agent", "change": True, "question": False, "reasons": ["asks for a change: fix"]}


@pytest.mark.parametrize("task", [
    TASK, "Saving a draft crashes with a KeyError when the title is empty.",
    "The export doesn't include the header row any more; it should.",
    "The status line never shows the branch name although the helper returns it.",
])
def test_bug_reports(task: str) -> None:
    assert repro_first.bug_report(task)


@pytest.mark.parametrize("task", [
    "Add a `--json` flag to the list command.",
    "Rename `load` to `read_config` everywhere.",
    "Loading must never modify the file; add a `migrate` command that converts it.",
])
def test_not_bug_reports(task: str) -> None:
    assert not repro_first.bug_report(task)


def test_applies_to_a_bug_report_on_a_project_with_tests(tmp_path: Path) -> None:
    assert repro_first.applies(TASK, shop(tmp_path), ROUTE)[0] == REPRO


def test_skipped_without_a_test_directory_or_on_a_tiny_project(tmp_path: Path) -> None:
    project = shop(tmp_path)
    (project / "shop" / "names.py").unlink()
    (project / "shop" / "__init__.py").unlink()
    assert repro_first.applies(TASK, project, ROUTE) == (None, "fewer than 3 source files")
    other = tmp_path / "flat"
    other.mkdir()
    for name in ("a.py", "b.py", "c.py"):
        (other / name).write_text("x = 1\n")
    assert repro_first.applies(TASK, other, ROUTE) == (None, "no pytest test directory")


def test_skipped_for_other_tasks(tmp_path: Path) -> None:
    project = shop(tmp_path)
    assert repro_first.applies("Add a `--json` flag to the list command.", project, ROUTE)[0] \
        is None
    assert repro_first.applies(TASK + " Add a regression test.", project, ROUTE)[1] == \
        "the task asks for tests itself"
    assert repro_first.applies(TASK, project, ROUTE, scoped=True)[0] is None
    question = {**ROUTE, "change": False, "question": True}
    assert repro_first.applies(TASK, project, question)[0] is None
    (project / REPRO).write_text("")
    assert repro_first.applies(TASK, project, ROUTE)[1] == f"{REPRO} exists already"


@pytest.mark.parametrize("path,test", [("tests/test_x.py", True), ("test_x.py", True),
                                       ("pkg/x_test.py", True), ("tests/conftest.py", True),
                                       ("shop/prices.py", False), ("src/vega/testing.py", False)])
def test_test_paths(path: str, test: bool) -> None:
    assert repro_first.is_test_path(path) is test


def test_hold_once_before_any_write_and_never_for_vega_notes() -> None:
    step = repro_first.Repro(REPRO)
    assert not step.should_hold(".vega/notes.md")
    assert not step.should_hold("tests/test_names.py")
    assert step.should_hold("shop/prices.py")
    step.wrote("tests/test_other.py")  # a test written first: the source edit is not held
    assert not step.should_hold("shop/prices.py")


# -- the agent loop ----------------------------------------------------------------------------

def test_the_first_source_edit_waits_for_a_failing_repro(tmp_path: Path) -> None:
    project = shop(tmp_path)
    agent, seen = agent_with(project, [
        FIX,  # held: not run
        call("write_file", path=REPRO, content=REPRO_TEST),  # run: it fails on the old code
        FIX,  # the same edit now runs
        text("Fixed total(): it multiplies by the quantity; the reproduction test passes."),
        text("Done."), text("Done."),
    ])
    agent.run(TASK)
    results = tool_results(seen)
    assert HOLD in results[0]
    assert "it fails (1 failed" in results[1]
    assert results[2].startswith("Edited ")
    assert "price * qty" in (project / "shop" / "prices.py").read_text()
    assert (project / REPRO).read_text() == REPRO_TEST  # passes with the fix: kept


def test_off_the_edit_runs_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "")
    project = shop(tmp_path)
    agent, seen = agent_with(project, [FIX, text("Fixed total()."), text("Done."), text("Done.")])
    agent.run(TASK)
    results = tool_results(seen)
    assert results[0].startswith("Edited ")
    assert not any(HOLD in r for r in results)
    assert not (project / REPRO).exists()


def test_the_held_call_repeated_runs_once_per_turn(tmp_path: Path) -> None:
    project = shop(tmp_path)
    agent, seen = agent_with(project, [FIX, FIX, text("Fixed total()."), text("Done."),
                                       text("Done.")])
    agent.run(TASK)
    results = tool_results(seen)
    assert HOLD in results[0] and results[1].startswith("Edited ")
    assert sum(HOLD in r for r in results) == 1


def test_a_repro_that_passes_on_the_old_code_is_called_out(tmp_path: Path) -> None:
    project = shop(tmp_path)
    passing = "def test_nothing():\n    assert True\n"
    agent, seen = agent_with(project, [
        call("write_file", path=REPRO, content=passing), FIX, text("Fixed."), text("Done."),
        text("Done."),
    ])
    agent.run(TASK)
    results = tool_results(seen)
    assert "it passes (1 passed" in results[0] and "does not reproduce" in results[0]
    assert results[1].startswith("Edited ")  # a test was written first: no hold


def test_a_repro_still_failing_at_the_end_is_removed(tmp_path: Path) -> None:
    project = shop(tmp_path)
    wrong = call("edit_file", path="shop/names.py", old_text="name.title()",
                 new_text="name.title().strip()")
    agent, _seen = agent_with(project, [
        call("write_file", path=REPRO, content=REPRO_TEST), wrong,
        text("Fixed."), text("Fixed."), text("Fixed."), text("Fixed."),
    ])
    events: list[tuple[str, dict]] = []
    agent.on_event = lambda kind, data: events.append((kind, data))
    agent.run(TASK)
    assert not (project / REPRO).exists()
    assert (project / "tests" / "test_names.py").exists()  # the project's own tests stay
    assert any(kind == "notice" and "removed it" in data.get("message", "")
               for kind, data in events)


def test_a_repro_alone_is_not_the_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # acceptance and verify_before_finish would send the answer back first (the repro fails)
    switch_off(monkeypatch, "acceptance", "verify_before_finish")
    project = shop(tmp_path)
    agent, seen = agent_with(project, [
        call("write_file", path=REPRO, content=REPRO_TEST),
        text("I wrote a test that shows the bug."), FIX, text("Fixed."), text("Done."),
        text("Done."),
    ])
    agent.run(TASK)
    notes = [m["content"] for request in seen for m in request
             if m.get("role") == "user" and "no file was written or edited" in str(m["content"])]
    assert notes  # finish_gate_no_edit still sends the repro-only answer back
    assert "price * qty" in (project / "shop" / "prices.py").read_text()


def test_the_switch_is_opt_in() -> None:
    assert "repro_first" in features.OPT_IN and "repro_first" not in features.FEATURES
