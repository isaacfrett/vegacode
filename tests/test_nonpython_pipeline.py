"""Single-file non-Python pipeline (write_loop config): routing, the named check, repairs."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from switches import switch_off

from vega import nonpython_pipeline, routing, sandbox
from vega.agent import CodingAgent

needs_node = pytest.mark.skipif(not (sandbox.available() and shutil.which("node")),
                                reason="needs the macOS sandbox and node")
needs_go = pytest.mark.skipif(not (sandbox.available() and shutil.which("go")),
                              reason="needs the macOS sandbox and go")

CHECK_PY = ("import subprocess, sys\n"
            "sys.exit(subprocess.run(['node', '--test', 'money.test.js']).returncode)\n")
TEST_JS = ("const test = require('node:test');\nconst assert = require('node:assert');\n"
           "const { add } = require('./money.js');\n"
           "test('cents', () => { assert.strictEqual(add(10, 20), 30); });\n")
TASK = ("`money.js` exports `add(a, b)` that returns the sum of two integer cent amounts, but it "
        "subtracts. Fix `money.js`. Run the tests with `python check.py`.")
WRONG = "File: money.js\n```javascript\nexports.add = (a, b) => a - b;\n```"
RIGHT = "File: money.js\n```javascript\nexports.add = (a, b) => a + b;\n```"


def project(root: Path) -> Path:
    (root / "money.js").write_text("exports.add = (a, b) => a - b;\n")
    (root / "money.test.js").write_text(TEST_JS)
    (root / "check.py").write_text(CHECK_PY)
    return root


def scripted(root: Path, replies: list[str]) -> tuple[CodingAgent, list[dict]]:
    agent = CodingAgent(root, allow_commands=False)
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content=replies.pop(0), tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, calls


def test_the_named_check_command_is_found_and_made_runnable(tmp_path: Path) -> None:
    project(tmp_path)
    assert nonpython_pipeline.check_command(tmp_path, TASK, "money.js") == "python check.py"
    assert nonpython_pipeline.check_command(tmp_path, "Fix `money.js`. Run `python nope.py`.",
                                            "money.js") is None  # no such script
    assert nonpython_pipeline.check_command(
        tmp_path, "Fix `money.js`. Do not run `python check.py` yet.", "money.js") is None
    (tmp_path / "queries").mkdir()
    (tmp_path / "runq.py").write_text("")
    task = "`python runq.py queries/<file>.sql [name=value ...]` runs a query."
    assert nonpython_pipeline.check_command(tmp_path, task, "queries/weekly.sql") == \
        "python runq.py queries/weekly.sql"
    both = "Fix `money.js`. `node --test` runs the tests; `python check.py` sets it up first."
    assert nonpython_pipeline.check_command(tmp_path, both, "money.js") == "python check.py"


def test_routing_sends_one_non_python_file_with_a_check_here(tmp_path: Path, monkeypatch) -> None:
    project(tmp_path)
    route = routing.route_task(TASK, tmp_path)
    assert route.name == routing.NONPYTHON
    assert route.other_targets == ["money.js"] and route.check_command == "python check.py"
    switch_off(monkeypatch, "nonpython_pipeline")
    assert routing.route_task(TASK, tmp_path).name == routing.AGENT  # the old route


def test_no_check_command_is_not_this_route(tmp_path: Path, monkeypatch) -> None:
    project(tmp_path)
    switch_off(monkeypatch, "web_pipeline")  # about the single-file gate; JS would go web
    route = routing.route_task("`money.js` subtracts in `add`. Fix `money.js`.", tmp_path)
    assert route.name != routing.NONPYTHON
    assert "not nonpython: the task names no check command to run" in route.reasons


@needs_node
def test_write_run_the_check_and_repair_in_the_sandbox(tmp_path: Path) -> None:
    project(tmp_path)
    agent, calls = scripted(tmp_path, [WRONG, RIGHT])
    answer = agent.run(TASK)
    assert (tmp_path / "money.js").read_text() == "exports.add = (a, b) => a + b;\n"
    assert answer == "Wrote money.js; 1 of 1 checks pass (`python check.py`)."
    repair = calls[1]["messages"][-1]["content"]
    assert "The checks fail:" in repair and "`python check.py` exit code 1" in repair
    assert agent.last_route["name"] == routing.NONPYTHON


@needs_node
def test_giving_up_keeps_the_best_round_and_quotes_the_failure(tmp_path: Path,
                                                               monkeypatch) -> None:
    switch_off(monkeypatch, "unchanged_rounds")  # every round here rewrites the same file
    project(tmp_path)
    broken = "File: money.js\n```javascript\nexports.add = (a, b) => { throw new Error('no'); };\n```"
    agent, calls = scripted(tmp_path, [broken] * (nonpython_pipeline.REPAIRS + 1))
    answer = agent.run(TASK)
    assert len(calls) == nonpython_pipeline.REPAIRS + 1
    assert answer.startswith("Wrote money.js, but only 0 of 1 checks pass (`python check.py`)")
    assert "Still failing:" in answer


def test_identical_rewrites_are_told_once_then_end_the_loop(tmp_path: Path) -> None:
    # unchanged_rounds: round 2 rewrites money.js byte-identically -> no check, one note; round 3
    # does it again -> the loop ends with round 1 kept (3 calls, not REPAIRS + 1)
    project(tmp_path)
    broken = "File: money.js\n```javascript\nexports.add = (a, b) => { throw new Error('no'); };\n```"
    agent, calls = scripted(tmp_path, [broken] * (nonpython_pipeline.REPAIRS + 1))
    answer = agent.run(TASK)
    assert len(calls) == 3
    note = calls[2]["messages"][-1]["content"]
    assert "Your rewrite was identical to the current file (money.js)" in note
    assert "Change the code that causes it." in note
    assert answer.startswith("Wrote money.js, but only 0 of 1 checks pass (`python check.py`)")
    assert "Still failing:" in answer


def test_a_round_that_changes_bytes_is_checked(tmp_path: Path) -> None:
    # unchanged_rounds compares bytes: round 2 differs from round 1 -> checked as before
    project(tmp_path)
    broken = "File: money.js\n```javascript\nexports.add = (a, b) => { throw new Error('no'); };\n```"
    other = "File: money.js\n```javascript\nexports.add = (a, b) => { throw new Error('nope'); };\n```"
    agent, calls = scripted(tmp_path, [broken, other, other, other])
    agent.run(TASK)
    assert len(calls) == 4  # 1, 2 checked; 3 unchanged (told); 4 unchanged again (ends)
    assert "Your rewrite was identical" not in calls[2]["messages"][-1]["content"]
    assert "Your rewrite was identical" in calls[3]["messages"][-1]["content"]


@needs_go
def test_a_tiny_go_module_with_go_test(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/sum\n\ngo 1.21\n")
    (tmp_path / "sum.go").write_text("package sum\n\nfunc Sum(xs []int) int { return 0 }\n")
    (tmp_path / "sum_test.go").write_text(
        'package sum\n\nimport "testing"\n\nfunc TestSum(t *testing.T) {\n'
        '\tif Sum([]int{1, 2, 3}) != 6 {\n\t\tt.Fatal("want 6")\n\t}\n}\n')
    task = "`Sum` in `sum.go` returns 0. Fix `sum.go` so it adds the numbers. Check with `go test ./...`."
    reply = ("File: sum.go\n```go\npackage sum\n\nfunc Sum(xs []int) int {\n\ttotal := 0\n"
             "\tfor _, x := range xs {\n\t\ttotal += x\n\t}\n\treturn total\n}\n```")
    agent, _ = scripted(tmp_path, [reply])
    assert agent.run(task) == "Wrote sum.go; 1 of 1 checks pass (`go test ./...`)."


def test_a_config_file_that_does_not_parse_never_lands(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text('[tool.lint]\nline = 80\n')
    (tmp_path / "check.py").write_text("print('ok')\n")
    task = "Set `line` to 100 in `ruff.toml`. Run `python check.py`."
    (tmp_path / "ruff.toml").write_text("line = 80\n")
    bad = "File: ruff.toml\n```toml\nline = = 100\n```"
    good = "File: ruff.toml\n```toml\nline = 100\n```"
    agent, calls = scripted(tmp_path, [bad, good])
    route = routing.route_task(task, tmp_path)
    assert route.name == routing.NONPYTHON
    answer = nonpython_pipeline.run(agent, task, route=route)
    assert "That file was not saved: Write not applied" in calls[1]["messages"][-1]["content"]
    assert (tmp_path / "ruff.toml").read_text() == "line = 100\n"
    if sandbox.available():
        assert answer == "Wrote ruff.toml; 2 of 2 checks pass (1 format check, `python check.py`)."
