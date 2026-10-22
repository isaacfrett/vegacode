"""verify_before_finish, repair_feedback and claim_check (Granite baseline, 2026-10-16), and long
tasks. Each switch has a switched-off test. Scripted fake models only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_agent_loop import agent_with, call, text

from vega import check_feedback, features, project_pipeline, question_pipeline, sandbox
from vega.check_feedback import condense, project_command, success_claim


def switch_off(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    import os

    current = [n for n in os.environ.get(features.ENV, "").split(",") if n]
    monkeypatch.setenv(features.ENV, ",".join([*current, name]))


# A real failing run (paths shortened).
FAILING = """exit code: 1
.F                                                                       [100%]
=================================== FAILURES ===================================
________________________________ test_run_tool _________________________________

    def test_run_tool():
>       code, out = asyncio.run(handlers.run_tool([sys.executable, "-c", "print('hi')"]))
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/test_handlers.py:16:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/usr/lib/python3.12/asyncio/runners.py:194: in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
/usr/lib/python3.12/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/lib/python3.12/asyncio/base_events.py:691: in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
src/loginsvc/handlers.py:38: in run_tool
    completed = await asyncio.create_subprocess_exec(
/usr/lib/python3.12/asyncio/subprocess.py:224: in create_subprocess_exec
    transport, protocol = await loop.subprocess_exec(
/usr/lib/python3.12/asyncio/base_events.py:1756: in subprocess_exec
    transport = await self._make_subprocess_transport(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <_UnixSubprocessTransport closed not started>
args = ('/usr/bin/python', '-c', "print('hi')")

    def _start(self, args, shell, stdin, stdout, stderr, bufsize, **kwargs):
        stdin_w = None
        try:
>           self._proc = subprocess.Popen(
                args, shell=shell, stdin=stdin, stdout=stdout, stderr=stderr,
                universal_newlines=False, bufsize=bufsize, **kwargs)
E               TypeError: Popen.__init__() got an unexpected keyword argument 'timeout'

/usr/lib/python3.12/asyncio/unix_events.py:820: TypeError
=========================== short test summary info ============================
FAILED tests/test_handlers.py::test_run_tool - TypeError: Popen.__init__() go...
1 failed, 1 passed in 0.04s"""
PASSING = "exit code: 0\n..                                                   [100%]\n2 passed in 0.01s"


# -- repair_feedback: condensing ---------------------------------------------------------------------

def test_condensed_output_keeps_ids_assertions_and_the_project_line() -> None:
    out = condense(FAILING, "python -m pytest -q")
    assert out.startswith("exit code: 1\n1 failed, 1 passed in 0.04s")
    assert "FAILED tests/test_handlers.py::test_run_tool" in out
    assert "E TypeError: Popen.__init__() got an unexpected keyword argument 'timeout'" in out
    assert "tests/test_handlers.py:16: code, out = asyncio.run(" in out  # the test's line
    assert "src/loginsvc/handlers.py:38 in run_tool: completed = await" in out  # nearest own line
    assert "base_events.py" not in out and "stdin_w = None" not in out  # library frames dropped
    assert "--tb=long 'tests/test_handlers.py::test_run_tool'" in out  # how to see it all
    assert len(out) < len(FAILING) / 2


def test_many_failures_list_every_id_and_collapse_repeats() -> None:
    block = ("_____________ test_{n} _____________\n\n    def test_{n}():\n>       assert f({n}) == 1\n"
             "E       assert 0 == 1\n\ntests/test_f.py:{n}: AssertionError\n")
    body = "".join(block.format(n=n) for n in range(1, 10))
    summary = "".join(f"FAILED tests/test_f.py::test_{n} - assert 0 == 1\n" for n in range(1, 10))
    result = ("exit code: 1\n" + "=" * 20 + " FAILURES " + "=" * 20 + "\n" + body + "=" * 10
              + " short test summary info " + "=" * 10 + "\n" + summary + "9 failed in 0.1s")
    out = condense(result, "pytest")
    for n in range(1, 10):
        assert f"tests/test_f.py::test_{n}" in out
    assert "(same error as above)" in out and out.count("E assert 0 == 1") == 1


def test_short_passing_and_narrowed_runs_are_unchanged() -> None:
    assert condense(PASSING) == PASSING
    assert condense("exit code: 1\nE assert 1 == 2\n1 failed in 0.01s") .startswith("exit code: 1")
    narrowed = "python -m pytest -q --tb=long 'tests/test_handlers.py::test_run_tool'"
    assert condense(FAILING, narrowed) == FAILING


def test_plain_traceback_keeps_own_frames_and_the_exception() -> None:
    lines = ["exit code: 1", "Traceback (most recent call last):"]
    lines += [f'  File "/usr/lib/python3.12/lib{n}.py", line {n}, in f{n}\n    x = y()'
              for n in range(30)]
    lines += ['  File "./app/main.py", line 7, in run\n    total = parse(rows)',
              "ValueError: bad row 3"]
    out = condense("\n".join(lines), "python app/main.py")
    assert "app/main.py:7 in run: total = parse(rows)" in out
    assert "ValueError: bad row 3" in out and "lib12.py" not in out


def test_loop_shows_the_condensed_failure_and_logs_the_full_one(tmp_path: Path,
                                                                 monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "run_check", lambda *a, **k: FAILING)
    switch_off(monkeypatch, "verify_before_finish")
    switch_off(monkeypatch, "question_pipeline")  # a question: this tests the agent loop's results
    agent, seen = agent_with(tmp_path, [call("run_check", command="python -m pytest -q"),
                                        text("One test fails: run_tool passes timeout to Popen.")])
    events: list[tuple[str, dict]] = []
    agent.on_event = lambda kind, data: events.append((kind, data))
    agent.run("Why does run_tool fail?")
    tool_message = seen[1][-1]
    assert tool_message["role"] == "tool" and "Vega condensed this output" in tool_message["content"]
    assert "base_events.py" not in tool_message["content"]
    assert any(kind == "tool_result" and data["result"] == FAILING for kind, data in events)
    switch_off(monkeypatch, "repair_feedback")
    agent, seen = agent_with(tmp_path, [call("run_check", command="python -m pytest -q"),
                                        text("One test fails.")])
    agent.run("Why does run_tool fail?")
    assert "base_events.py" in seen[1][-1]["content"]


def test_project_pipeline_repair_prompt_uses_the_condensed_failure(monkeypatch) -> None:
    text_on = project_pipeline._failure_text(FAILING)
    assert "FAILED tests/test_handlers.py::test_run_tool" in text_on
    assert "base_events.py" not in text_on and "Vega condensed" not in text_on
    switch_off(monkeypatch, "repair_feedback")
    assert "base_events.py" in project_pipeline._failure_text(FAILING)


# -- verify_before_finish ------------------------------------------------------------------------------

@pytest.fixture
def tested_project(tmp_path: Path, monkeypatch) -> Path:
    # These script verify_before_finish in the agent loop. With visible tests, "Make add() return
    # the sum." routes to the project pipeline, and acceptance's checks would call the scripted
    # sandbox.run_check too (acceptance's own gate: tests/test_acceptance.py).
    switch_off(monkeypatch, "project_pipeline")
    switch_off(monkeypatch, "acceptance")
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    return tmp_path


def runs(monkeypatch, results: list[str]) -> list[str]:
    """Script sandbox.run_check: each call returns the next result; the commands are recorded."""
    commands: list[str] = []

    def run_check(command, project, **kwargs):
        commands.append(command)
        return results.pop(0)

    monkeypatch.setattr(sandbox, "run_check", run_check)
    return commands


EDIT = call("edit_file", path="calc.py", old_text="a - b", new_text="a + b")


def test_runs_the_check_once_after_a_write_and_feeds_the_result_back(tested_project: Path,
                                                                       monkeypatch) -> None:
    commands = runs(monkeypatch, [FAILING])
    agent, seen = agent_with(tested_project, [EDIT, text("Changed add()."),
                                              text("add() is fixed; test_run_tool still fails.")])
    answer = agent.run("Make add() return the sum.")
    assert answer == "add() is fixed; test_run_tool still fails."
    assert commands == ["python -m pytest -q"]
    note = seen[2][-1]["content"]
    assert note.startswith("(Vega harness) Before you finish: Vega ran `python -m pytest -q` on "
                           "your changes → exit code 1 (1 failed, 1 passed in 0.04s):")
    assert "FAILED tests/test_handlers.py::test_run_tool" in note
    assert "Fix the failures, or explain in your answer what is left." in note


def test_a_passing_check_is_reported_so_the_answer_can_cite_it(tested_project: Path,
                                                               monkeypatch) -> None:
    runs(monkeypatch, [PASSING])
    agent, seen = agent_with(tested_project, [EDIT, text("Changed add()."),
                                              text("Changed add(); pytest passed (2 tests).")])
    assert agent.run("Make add() return the sum.") == "Changed add(); pytest passed (2 tests)."
    assert "and it passed (2 passed in 0.01s). Give your final answer now" in seen[2][-1]["content"]


def test_no_run_when_the_model_already_checked_after_its_last_write(tested_project: Path,
                                                                    monkeypatch) -> None:
    commands = runs(monkeypatch, [PASSING])
    agent, _ = agent_with(tested_project, [EDIT, call("run_check", command="pytest tests"),
                                           text("Changed add(); 2 passed.")])
    assert agent.run("Make add() return the sum.") == "Changed add(); 2 passed."
    assert commands == ["pytest tests"]  # the model's own run only


def test_at_most_two_rounds_per_turn(tested_project: Path, monkeypatch) -> None:
    commands = runs(monkeypatch, [FAILING, FAILING, FAILING])
    edit_back = call("edit_file", path="calc.py", old_text="a + b", new_text="a - b")
    agent, _ = agent_with(tested_project, [EDIT, text("one"), edit_back, text("two"), EDIT,
                                           text("three")])
    assert agent.run("Make add() return the sum.") == "three"
    assert len(commands) == 2


def test_never_for_questions_or_without_a_change_or_a_check(tested_project: Path,
                                                            monkeypatch) -> None:
    commands = runs(monkeypatch, [])
    agent, _ = agent_with(tested_project, [EDIT, text("It subtracted; now it adds.")])
    assert question_pipeline.is_question("What is wrong in calc.py?")
    agent.run("What is wrong in calc.py?")  # a question, even if the model edits
    agent, _ = agent_with(tested_project, [call("read_file", path="calc.py"), text("No change.")])
    agent.run("Make add() return the sum.")  # nothing written
    (tested_project / "tests" / "test_calc.py").unlink()
    (tested_project / "tests").rmdir()
    agent, _ = agent_with(tested_project, [EDIT, text("Changed add().")])
    agent.run("Make add() return the sum.")  # no test or check command in the project
    assert commands == []


def test_verify_switched_off(tested_project: Path, monkeypatch) -> None:
    commands = runs(monkeypatch, [])
    switch_off(monkeypatch, "verify_before_finish")
    agent, _ = agent_with(tested_project, [EDIT, text("Changed add().")])
    assert agent.run("Make add() return the sum.") == "Changed add()."
    assert commands == []


def test_the_check_command_the_project_or_task_gives(tmp_path: Path) -> None:
    assert project_command(tmp_path) is None
    (tmp_path / "check.py").write_text("print('ok')\n")
    assert project_command(tmp_path, "Port it. Run the tests with `python check.py`.") == \
        "python check.py"
    assert project_command(tmp_path, "Usage: `python check.py INPUT_CSV`; run it.") is None
    assert project_command(tmp_path, "It must pass `mypy --strict`.") is None  # no target
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {"test": "node --test"}}))
    assert project_command(tmp_path) == "npm test"
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts = '-q'\n")
    assert project_command(tmp_path) == "python -m pytest -q"
    assert check_feedback.covers("pytest tests/test_x.py", "python -m pytest -q")
    assert not check_feedback.covers("python - <<'PY'", "python -m pytest -q")


# -- claim_check -------------------------------------------------------------------------------------

def test_claim_contradicted_by_the_last_run_is_sent_back_once(tmp_path: Path, monkeypatch) -> None:
    switch_off(monkeypatch, "acceptance")  # its finish gate would answer the claim first
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    runs(monkeypatch, [FAILING])
    agent, seen = agent_with(tmp_path, [EDIT, call("run_check", command="python check_add.py"),
                                        text("Fixed add(). All tests pass."),
                                        text("Fixed add(). All tests pass.")])
    assert agent.run("Make add() return the sum.") == "Fixed add(). All tests pass."
    note = seen[3][-1]["content"]
    assert note.startswith('(Vega harness) Your answer says "All tests pass", but the last run '
                           "(`python check_add.py`) failed:")
    assert note.endswith("Correct the answer or keep fixing.")
    assert len(seen) == 4  # once: the second claim is the model's decision


def test_claim_with_nothing_run_after_the_last_write(tmp_path: Path, monkeypatch) -> None:
    switch_off(monkeypatch, "acceptance")  # its finish gate would answer the claim first
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    runs(monkeypatch, [])
    agent, seen = agent_with(tmp_path, [EDIT, text("Fixed and verified."), text("Fixed add().")])
    assert agent.run("Make add() return the sum.") == "Fixed add()."
    assert 'Your answer says "verified", but nothing was run after your last edit' in \
        seen[2][-1]["content"]


def test_no_claim_check_when_the_claim_holds_or_is_not_made(tmp_path: Path, monkeypatch) -> None:
    # the second turn's EDIT no longer applies (the first turn made it): no write that turn
    switch_off(monkeypatch, "finish_gate_no_edit")
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    runs(monkeypatch, [PASSING, FAILING])
    agent, _ = agent_with(tmp_path, [EDIT, call("run_check", command="python check_add.py"),
                                     text("Fixed add(); the tests pass.")])
    assert agent.run("Make add() return the sum.") == "Fixed add(); the tests pass."
    agent, _ = agent_with(tmp_path, [EDIT, call("run_check", command="python check_add.py"),
                                     text("Changed add(), but the tests do not pass yet.")])
    assert agent.run("Make add() return the sum.") == "Changed add(), but the tests do not pass yet."


def test_claim_check_switched_off(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    runs(monkeypatch, [FAILING])
    switch_off(monkeypatch, "claim_check")
    agent, _ = agent_with(tmp_path, [EDIT, call("run_check", command="python check_add.py"),
                                     text("All tests pass.")])
    assert agent.run("Make add() return the sum.") == "All tests pass."


def test_success_claims() -> None:
    assert success_claim("Done. All tests now pass.") == "All tests now pass"
    assert success_claim("I verified it with a script.") == "verified"
    assert success_claim("It works correctly for the examples.") == "works correctly"
    assert success_claim("The tests do not pass yet.") is None
    assert success_claim("I could not verify the timing requirement.") is None
    assert success_claim("Changed calc.py.") is None


# -- long tasks (the direct_long_tasks switch went with the old routing, 2026-10-17) ------------------

LONG = ("Implement `stats.median(values)` in `stats.py` exactly as its docstring specifies. "
        + "Ties, empty input and NaN handling follow the docstring; keep the signature. " * 6)


def test_long_tasks_route_by_the_writer_budget_not_their_length(tmp_path: Path) -> None:
    from vega import routing

    (tmp_path / "stats.py").write_text('def median(values):\n    """Median."""\n    pass\n')
    assert len(LONG) > 400
    route = routing.route_task(LONG, tmp_path)
    assert (route.name, route.targets) == ("direct", ["stats.py"])
    (tmp_path / "check.py").write_text("print('runs the tests')\n")
    with_helper = routing.route_task(LONG + " Run the tests with `python check.py`.", tmp_path)
    assert (with_helper.name, with_helper.targets) == ("direct", ["stats.py"])  # never check.py
    port = routing.route_task(LONG + " Port it to `src/lib.rs`.", tmp_path)
    assert port.name != "direct" and port.other_targets == ["src/lib.rs"]


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_verify_runs_the_real_tests_in_the_sandbox(tested_project: Path) -> None:
    wrong = call("edit_file", path="calc.py", old_text="a - b", new_text="a * b")
    right = call("edit_file", path="calc.py", old_text="a * b", new_text="a + b")
    agent, seen = agent_with(tested_project, [wrong, text("Changed add()."), right,
                                              text("Changed add()."), text("Fixed add().")])
    assert agent.run("Make add() return the sum.") == "Fixed add()."
    failed_note, passed_note = seen[2][-1]["content"], seen[4][-1]["content"]
    assert "exit code 1" in failed_note and "tests/test_calc.py::test_add" in failed_note
    assert "assert 6 == 5" in failed_note  # short output is shown whole
    assert "and it passed (1 passed" in passed_note
