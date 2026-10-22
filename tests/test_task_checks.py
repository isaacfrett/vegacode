"""Task checks: the task's own statements as small gated checks in the project pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import features, sandbox, task_checks
from vega.agent import CodingAgent
from vega.task_checks import Check, TaskChecks

needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")

CLI = '''"""`cfgtool show`."""
import argparse
import sys

from cfgtool.loader import read_settings


def main(argv=None, out=None):
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="cfgtool")
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show")
    show.add_argument("--file")
    args = parser.parse_args(argv)
    for name, value in sorted(read_settings(args.file).items()):
        print(f"{name} = {value!r}", file=out)
    return 0
'''
LOADER = '''FIELDS = {"port": (int, 8000), "debug": (bool, False)}


def read_settings(path=None):
    return {name: default for name, (_, default) in FIELDS.items()}
'''
LOADER_ENV = '''import os

FIELDS = {"port": (int, 8000), "debug": (bool, False)}


def read_settings(path=None, env=True):
    config = {name: default for name, (_, default) in FIELDS.items()}
    for name, (kind, _) in FIELDS.items():
        raw = os.environ.get("VEGA_" + name.upper())
        if env and raw is not None:
            config[name] = kind(raw)
    return config
'''
TASK = ("Support environment overrides: VEGA_PORT=9000 overrides `port`. `cfgtool show --no-env` "
        "prints the settings without them. `cfgtool show --file PATH` keeps working.")


def make_project(root: Path, loader: str = LOADER, cli: str = CLI) -> Path:
    (root / "cfgtool").mkdir(parents=True, exist_ok=True)
    (root / "cfgtool" / "__init__.py").write_text("")
    (root / "cfgtool" / "loader.py").write_text(loader)
    (root / "cfgtool" / "cli.py").write_text(cli)
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test_loader.py").write_text(
        "from cfgtool.loader import read_settings\n\n\ndef test_port():\n"
        "    assert read_settings()['port'] == 8000\n")
    return root


# -- deriving -------------------------------------------------------------------------------------

def test_cli_checks_take_backticked_commands_of_the_projects_prog(tmp_path: Path) -> None:
    make_project(tmp_path)
    checks = task_checks.cli_checks(tmp_path, TASK + " Run `python -m pytest` and `ls -la`.")
    assert [c.claim for c in checks] == ["cfgtool show --no-env", "cfgtool show --file PATH"]
    first, second = checks
    assert first.spec == {"module": "cfgtool.cli", "prog": "cfgtool", "argv": ["show", "--no-env"],
                          "fixed": ["show", "--no-env"], "placeholders": False}
    assert second.spec["argv"] == ["show", "--file", "x"] and second.spec["placeholders"]
    assert second.spec["fixed"] == ["show", "--file"]


def test_cli_checks_skip_optional_parts_and_unknown_programs(tmp_path: Path) -> None:
    make_project(tmp_path)
    checks = task_checks.cli_checks(tmp_path, "`cfgtool [--verbose] show …` and `other show`")
    assert [c.spec["argv"] for c in checks] == [["show", "x"]]


def test_env_checks_need_an_example_value_and_a_loader(tmp_path: Path) -> None:
    make_project(tmp_path)
    checks = task_checks.env_checks(tmp_path, TASK + " VEGA_DEBUG overrides `debug`.")
    assert [c.claim for c in checks] == ["VEGA_PORT=9000 overrides `port`"]
    assert checks[0].spec["loaders"] == ["cfgtool.loader:read_settings"]
    assert checks[0].spec["prefix"] == "VEGA_"
    (tmp_path / "cfgtool" / "loader.py").write_text("def parse(text):\n    return {}\n")
    assert task_checks.env_checks(tmp_path, TASK) == []  # no loader: nothing to run


def test_model_checks_must_quote_the_task_and_assert() -> None:
    task = "With `top` depending on `b, a` and `b` depending on `c`, the order is `c, b, a, top`."
    reply = json.dumps({"checks": [
        {"claim": "the order is `c, b, a, top`", "code": "assert order() == ['c']"},
        {"claim": "it is fast and robust", "code": "assert True"},
        {"claim": "the order is c, b, a, top", "code": "x = 1"},
        {"claim": "the order is c, b, a, top", "code": "import subprocess\nassert 1"},
        {"claim": "the order is c, b, a, top", "code": "def broken(:"},
    ]})
    checks, dropped = task_checks.model_checks(reply, task)
    assert [c.spec["code"] for c in checks] == ["assert order() == ['c']"]
    assert [d.split(":")[0] for d in dropped] == [
        "not the task's words", "asserts nothing", "uses subprocess, pytest or the tests",
        "does not parse"]
    assert task_checks.model_checks("not json", task) == ([], ["the reply was not JSON"])


def test_missing_names_count_only_where_the_task_puts_them() -> None:
    task = ("Add `Money` in `billing/money.py`. `CycleError` (defined in `choremate.graph`). Its "
            "`.cycle` attribute lists the loop.")
    named = task_checks.missing_by_the_task
    assert named("ImportError: cannot import name 'CycleError' from 'choremate.graph' (x)", task)
    assert named("ModuleNotFoundError: No module named 'billing.money'", task)
    assert named("AttributeError: 'CycleError' object has no attribute 'cycle'", task)
    assert not named("ImportError: cannot import name 'Money' from 'billing' (x)", task)
    assert not named("ModuleNotFoundError: No module named 'choremate.task'", task)
    assert not named("NameError: name 'helper' is not defined", task)


def test_checks_importing_modules_nobody_named_are_dropped(tmp_path: Path) -> None:
    (tmp_path / "choremate").mkdir()
    (tmp_path / "choremate" / "__init__.py").write_text("")
    (tmp_path / "choremate" / "graph.py").write_text("")
    task = "Add `CycleError` to `choremate.graph` and a `choremate.cycles` module."
    code = ("from choremate.graph import CycleError\nfrom choremate.task import Task\n"
            "from choremate.cycles import find\nimport json\nassert 1")
    assert task_checks.unknown_imports(code, tmp_path, task) == ["choremate.task"]
    checks = TaskChecks(task)
    prepared = task_checks.prepare(tmp_path, task, rules=False,
                                   model=[Check("model-1", "code", "CycleError", {"code": code})],
                                   runner=lambda command, project: "exit code: 0\n{}")
    assert prepared.checks == [] and "choremate.task" in prepared.notes[0]
    assert checks.active == []


# -- verdicts and the churn rules ------------------------------------------------------------------

def cli_check(placeholders: bool = False, baseline: dict | None = None) -> Check:
    return Check("cli-1", "cli", "cfgtool show --no-env",
                 {"module": "cfgtool.cli", "prog": "cfgtool", "argv": ["show", "--no-env"],
                  "fixed": ["show", "--no-env"], "placeholders": placeholders},
                 baseline=baseline or {})


def test_cli_verdicts() -> None:
    rejected = {"usage": "cfgtool: error: unrecognized arguments: --no-env", "traceback": ""}
    assert "argparse rejects --no-env" in task_checks.verdict(cli_check(), rejected)
    choice = {"usage": "cfgtool: error: argument command: invalid choice: 'show'", "traceback": ""}
    assert "argparse rejects show" in task_checks.verdict(cli_check(True), choice)
    crash = {"usage": "", "traceback": "KeyError: 'port' (in loader.py:3)"}
    assert "traceback" in task_checks.verdict(cli_check(), crash)
    assert task_checks.verdict(cli_check(True), crash) == ""  # made-up arguments: words only
    assert task_checks.verdict(cli_check(baseline=crash), crash) == ""  # not this change's doing
    missing = {"usage": "", "traceback": "FileNotFoundError: tasks.ini"}
    assert task_checks.verdict(cli_check(), missing) == ""  # the sandbox lacks a file
    required = {"usage": "cfgtool: error: the following arguments are required: x"}
    assert task_checks.verdict(cli_check(baseline=required), required) == ""
    assert "usage error" in task_checks.verdict(cli_check(), required)


def scripted_runner(rounds: list[dict[str, dict]]):
    def run(command: str, project: Path) -> str:
        return "exit code: 0\n" + json.dumps(rounds.pop(0))
    return run


def test_a_check_failing_while_all_else_passes_is_dropped_after_two_rounds(tmp_path) -> None:
    fail = {"status": "fail", "detail": "assert 1 == 2"}
    check = Check("model-1", "code", "the total is 3", {"code": "assert 1 == 2"})
    checks = TaskChecks("the total is 3", [check],
                        runner=scripted_runner([{"model-1": fail}] * 4))
    checks.baseline(tmp_path)
    assert checks.active == [check]
    expected = '"the total is 3": assert 1 == 2\n  check:\n    assert 1 == 2'
    assert checks.evaluate(tmp_path, project_failing=0) == [expected]
    second = checks.evaluate(tmp_path, project_failing=0)
    assert second[0].endswith("(no longer checked)") and check.dropped
    assert checks.active == [] and checks.evaluate(tmp_path, project_failing=0) == []


def test_failing_while_the_project_tests_fail_is_not_stale(tmp_path) -> None:
    fail = {"status": "fail", "detail": "nope"}
    check = Check("env-1", "env", "VEGA_PORT=9000 overrides `port`", {})
    checks = TaskChecks("t", [check], runner=scripted_runner([{"env-1": fail}] * 4))
    checks.baseline(tmp_path)
    for _ in range(3):
        assert len(checks.evaluate(tmp_path, project_failing=2)) == 1
    assert not check.dropped


def test_a_check_whose_first_pass_breaks_the_project_is_dropped(tmp_path) -> None:
    fail, ok = {"status": "fail", "detail": "nope"}, {"status": "pass"}
    check = Check("model-1", "code", "x", {"code": "assert 0"})
    checks = TaskChecks("x", [check],
                        runner=scripted_runner([{"model-1": fail}, {"model-1": fail},
                                                {"model-1": ok}]))
    checks.baseline(tmp_path)
    checks.evaluate(tmp_path, project_failing=0)  # round 1: tests pass, the check fails
    assert checks.evaluate(tmp_path, project_failing=3) == []  # its pass broke 3 tests
    assert "broke the project's own checks" in check.dropped


def test_baseline_keeps_only_checks_that_fail_because_the_behaviour_is_missing(tmp_path) -> None:
    results = {"a": {"status": "pass"}, "b": {"status": "crash", "detail": "TypeError: x"},
               "c": {"status": "fail", "detail": "assert"}, "d": {"status": "skip",
                                                                  "detail": "no loader"},
               "e": {"status": "missing", "detail": "No module named 'pkg.new'"},
               "f": {"usage": "error: unrecognized arguments: --new", "traceback": ""}}
    checks = [Check(i, "code", i, {"code": ""}) for i in "abce"]
    checks += [Check("d", "env", "d", {}), cli_check()]
    checks[-1].id = "f"
    prepared = TaskChecks("Add `pkg/new.py`.", checks,
                          runner=lambda command, project: "x\n" + json.dumps(results))
    prepared.baseline(tmp_path)
    assert [c.id for c in prepared.active] == ["c", "e", "f"]
    assert "already passes" in checks[0].dropped and "of its own" in checks[1].dropped


# -- in the sandbox --------------------------------------------------------------------------------

@needs_sandbox
def test_rule_checks_fail_before_the_change_and_pass_after_it(tmp_path: Path) -> None:
    make_project(tmp_path)
    checks = task_checks.prepare(tmp_path, TASK, rules=True)
    assert [c.claim for c in checks.active] == [
        "cfgtool show --no-env", "cfgtool show --file PATH", "VEGA_PORT=9000 overrides `port`"]
    lines = checks.evaluate(tmp_path, project_failing=0)
    assert len(lines) == 2  # --file PATH already works: a smoke check that passes
    assert "argparse rejects --no-env" in lines[0]
    assert "gives port=8000" in lines[1] and "expected 9000" in lines[1]
    cli = CLI.replace('show.add_argument("--file")',
                      'show.add_argument("--file")\n    show.add_argument("--no-env", '
                      'action="store_true")')
    cli = cli.replace("read_settings(args.file)", "read_settings(args.file, not args.no_env)")
    make_project(tmp_path, LOADER_ENV, cli)
    assert checks.evaluate(tmp_path, project_failing=0) == []


@needs_sandbox
def test_a_crash_in_the_new_code_is_reported(tmp_path: Path) -> None:
    make_project(tmp_path)
    checks = task_checks.prepare(tmp_path, TASK, rules=True)
    make_project(tmp_path, LOADER_ENV.replace("kind(raw)", "kind.convert(raw)"),
                 CLI.replace('show.add_argument("--file")', 'show.add_argument("--file")\n'
                             '    show.add_argument("--no-env", action="store_true")'))
    lines = checks.evaluate(tmp_path, project_failing=0)
    assert len(lines) == 1 and "raised AttributeError" in lines[0]


@needs_sandbox
def test_model_check_code_runs_and_reports_the_failing_assert(tmp_path: Path) -> None:
    make_project(tmp_path)
    code = ("import os\nfrom cfgtool.loader import read_settings\nos.environ['VEGA_PORT'] = "
            "'9000'\nassert read_settings()['port'] == 9000, 'port not overridden'")
    check = Check("model-1", "code", "VEGA_PORT=9000 overrides `port`", {"code": code})
    checks = task_checks.prepare(tmp_path, TASK, rules=False, model=[check])
    assert checks.active == [check]
    assert check.baseline == {"status": "fail", "detail": "assert read_settings()['port'] == "
                                                          "9000, 'port not overridden': port "
                                                          "not overridden"}


# -- in the pipeline -------------------------------------------------------------------------------

PLAN = json.dumps({"analysis": "env overrides", "files": [
    {"path": "cfgtool/loader.py", "change": "read VEGA_ variables"},
    {"path": "cfgtool/cli.py", "change": "add --no-env"}]})


def scripted(root: Path, replies: list[str]) -> tuple[CodingAgent, list[dict]]:
    agent = CodingAgent(root, allow_commands=False)
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        content = replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, calls


def block(text: str) -> str:
    return f"```python\n{text}```"


@needs_sandbox
def test_pipeline_repairs_against_a_failing_task_check(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "project_pipeline")
    make_project(tmp_path)
    good_cli = CLI.replace('show.add_argument("--file")',
                           'show.add_argument("--file")\n    show.add_argument("--no-env", '
                           'action="store_true")').replace(
        "read_settings(args.file)", "read_settings(args.file, not args.no_env)")
    agent, calls = scripted(tmp_path, [PLAN, block(LOADER_ENV), block(CLI), PLAN,
                                       block(LOADER_ENV), block(good_cli)])
    from vega import project_pipeline

    answer = project_pipeline.run(agent, TASK)
    assert answer is not None and "tests pass" in answer
    repair = calls[3]["messages"][-1]["content"]  # the replan after round 1
    assert "task's own statements" in repair and "argparse rejects --no-env" in repair
    assert "--no-env" in (tmp_path / "cfgtool" / "cli.py").read_text()


@needs_sandbox
def test_pipeline_without_the_switch_finishes_on_green_tests(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV, "task_checks")
    monkeypatch.setenv(features.ENV_ON, "project_pipeline")
    make_project(tmp_path)
    agent, calls = scripted(tmp_path, [PLAN, block(LOADER_ENV), block(CLI)])
    from vega import project_pipeline

    assert "tests pass" in (project_pipeline.run(agent, TASK) or "")
    assert len(calls) == 3


@needs_sandbox
def test_model_checks_are_asked_for_once_and_gated(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV, "task_checks")
    monkeypatch.setenv(features.ENV_ON, "project_pipeline,task_checks_model")
    make_project(tmp_path)
    code = ("import os\nfrom cfgtool.loader import read_settings\nos.environ['VEGA_PORT'] = "
            "'9000'\nassert read_settings()['port'] == 9000")
    checks = json.dumps({"checks": [{"claim": "VEGA_PORT=9000 overrides `port`", "code": code},
                                    {"claim": "VEGA_PORT=9000 overrides", "code": "assert True"}]})
    agent, calls = scripted(tmp_path, [PLAN, checks, block(LOADER_ENV), block(CLI)])
    from vega import project_pipeline

    project_pipeline.run(agent, TASK)
    assert calls[1]["response_format"]["json_schema"]["name"] == "checks"
    assert calls[1]["max_tokens"] <= task_checks.MODEL_MAX_TOKENS
    assert "test_loader.py" in calls[1]["messages"][1]["content"]  # the visible tests are shown
    trace = agent.messages[-1]["content"]
    assert "task checks: \"VEGA_PORT=9000 overrides `port`\"" in trace
    assert "already passes before the change" in trace


# -- task_module_commands: `python -m pkg ...` and the full invocation of a bare name -----------

ROWSIFT_CLI = '''import argparse
import csv
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="colsort")
    parser.add_argument("file")
    parser.add_argument("--where", action="append", default=[])
    parser.add_argument("--columns")
    args = parser.parse_args(argv)
    with open(args.file, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    for row in rows:
        print(",".join(row))
    return 0
'''
ROWSIFT_TEST = '''from colsort.cli import main


def test_main_filters_a_file(tmp_path):
    path = tmp_path / "stock.csv"
    path.write_text("name,qty\\nbolt,12\\nnut,2\\n", encoding="utf-8")
    assert main([str(path)]) == 0
'''
ROWSIFT_TASK = ("Make `python -m colsort` usable in shell pipelines such as `python -m colsort "
                "--where qty>=7 < stock.csv | python -m colsort --columns name | head -1`: with no "
                "FILE argument, or FILE `-`, it reads CSV from stdin.")


def make_colsort(root: Path, cli: str = ROWSIFT_CLI) -> Path:
    (root / "colsort").mkdir(exist_ok=True)
    (root / "colsort" / "__init__.py").write_text("")
    (root / "colsort" / "__main__.py").write_text("from colsort.cli import main\n\n"
                                                  "raise SystemExit(main())\n")
    (root / "colsort" / "cli.py").write_text(cli)
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test_colsort.py").write_text(ROWSIFT_TEST)
    return root


def test_module_checks_take_each_pipeline_segment_with_a_sample_stdin(tmp_path: Path) -> None:
    make_colsort(tmp_path)
    checks = task_checks.module_checks(tmp_path, ROWSIFT_TASK)
    assert [c.claim for c in checks] == ["python -m colsort --where qty>=7 < stock.csv",
                                         "... | python -m colsort --columns name"]
    for check in checks:  # stock.csv is not in the project: the test's CSV of that name
        assert check.spec["stdin"] == "name,qty\nbolt,12\nnut,2\n"
        assert check.spec["reads"] and check.spec["module"] == "colsort"
    assert checks[0].spec["argv"] == ["--where", "qty>=7"]
    (tmp_path / "stock.csv").write_text("name,qty\nwasher,30\n")
    assert task_checks.module_checks(tmp_path, ROWSIFT_TASK)[0].spec["stdin"] == (
        "name,qty\nwasher,30\n")


def test_module_checks_skip_bare_names_other_programs_and_switch_off(tmp_path: Path,
                                                                      monkeypatch) -> None:
    make_colsort(tmp_path)
    task = ("Make `python -m colsort` work, `python -m pytest -q` must pass and "
            "`python -m numpy x` is not ours. Run `python -m colsort train|test` later.")
    checks = task_checks.module_checks(tmp_path, task)
    assert [c.claim for c in checks] == ["python -m colsort train|test"]
    assert checks[0].spec["placeholders"]  # alternatives: only argparse's rejections count
    monkeypatch.setenv(features.ENV, "task_module_commands")
    prepared = task_checks.prepare(tmp_path, ROWSIFT_TASK, rules=True,
                                   runner=lambda command, project: "\n{}")
    assert prepared.checks == []


def test_segments_split_pipes_and_redirects() -> None:
    assert task_checks._segments("python -m a x < in.csv | python -m b > out.txt") == [
        {"words": ["python", "-m", "a", "x"], "stdin_file": "in.csv", "piped": False},
        {"words": ["python", "-m", "b"], "stdin_file": "", "piped": True}]
    assert task_checks._segments("python -m a && rm -rf x") is None


TICKWISE_MAIN = '''import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="clockwork")
    parser.add_argument("expression")
    parser.add_argument("--after", required=True)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args(argv)
    if "mon" in args.expression:
        print("error: invalid name 'mon-fri' for field", file=sys.stderr)
        return 2
    print("2024-05-06 09:00 Mon")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
TICKWISE_README = '''# clockwork

```
python -m clockwork "0 9 * * mon-fri" --after 2024-05-03T09:00 --count 3
```

Usage: `python -m clockwork EXPRESSION --after YYYY-MM-DDTHH:MM [--count N]`
'''
TICKWISE_TASK = ("clockwork is half built. Finish the library so it does everything "
                 "clockwork/README.md specifies (errors, next runs), and make `python -m clockwork` "
                 "work end to end. "
                 "The existing tests must keep passing.")


def make_clockwork(root: Path, main: str = TICKWISE_MAIN) -> Path:
    (root / "clockwork").mkdir(exist_ok=True)
    (root / "clockwork" / "__init__.py").write_text("")
    (root / "clockwork" / "__main__.py").write_text(main)
    (root / "clockwork" / "README.md").write_text(TICKWISE_README)
    return root


def test_full_invocations_come_from_the_task_then_its_readme(tmp_path: Path) -> None:
    make_clockwork(tmp_path)
    # the usage line with placeholders is left out
    assert task_checks.full_invocations(tmp_path, TICKWISE_TASK, "clockwork") == [
        'python -m clockwork "0 9 * * mon-fri" --after 2024-05-03T09:00 --count 3']
    task = TICKWISE_TASK + " E.g. `python -m clockwork '*/5 * * * *' --after 2024-01-01T00:00`."
    assert task_checks.full_invocations(tmp_path, task, "clockwork")[0] == (
        "python -m clockwork '*/5 * * * *' --after 2024-01-01T00:00")


def test_acceptance_checks_the_readme_example_instead_of_a_bare_name(tmp_path: Path,
                                                                      monkeypatch) -> None:
    from vega import acceptance

    make_clockwork(tmp_path)
    assert acceptance.extract(tmp_path, TICKWISE_TASK).commands == [
        'python -m clockwork "0 9 * * mon-fri" --after 2024-05-03T09:00 --count 3']
    monkeypatch.setenv(features.ENV, "task_module_commands")
    assert acceptance.extract(tmp_path, TICKWISE_TASK).commands == ["python -m clockwork"]


def test_acceptance_does_not_fail_a_bare_name_on_its_usage_exit(tmp_path: Path,
                                                                 monkeypatch) -> None:
    from vega import acceptance

    usage = ("exit code: 2\nusage: clockwork [-h] --after AFTER expression\n"
             "clockwork: error: the following arguments are required: expression, --after\n")
    monkeypatch.setattr(sandbox, "run_check", lambda command, project, **kw: usage)
    assert acceptance._command_item("python -m clockwork", tmp_path, None, "command").ok
    assert not acceptance._command_item("python -m clockwork x", tmp_path, None, "command").ok
    monkeypatch.setenv(features.ENV, "task_module_commands")
    assert not acceptance._command_item("python -m clockwork", tmp_path, None, "command").ok


def module_check(**spec) -> Check:
    return Check("module-1", "module", "python -m colsort --where qty>=7 < stock.csv",
                 {"module": "colsort", "prog": "colsort", "argv": ["--where", "qty>=7"],
                  "fixed": ["--where", "qty>=7"], "placeholders": False, "reads": True,
                  "stdin": "name,qty\nbolt,12\n", **spec})


def test_module_verdicts() -> None:
    check = module_check()
    check.baseline = {"exit": 2, "usage": "colsort: error: the following arguments are "
                      "required: file", "traceback": ""}
    assert "usage error with input on stdin" in task_checks.verdict(check, check.baseline)
    crash = {"exit": 0, "usage": "", "traceback": "TypeError: expected str, not NoneType"}
    assert task_checks.verdict(check, crash).startswith("ends in a traceback (stdin: a sample")
    assert task_checks.verdict(check, {"exit": 0, "usage": "", "traceback": ""}) == ""
    empty = module_check(stdin="")
    assert task_checks.verdict(empty, {"exit": 1, "usage": "", "traceback":
                                       "JSONDecodeError: Expecting value"}) == ""
    assert task_checks.verdict(empty, crash) != ""


def test_module_checks_are_kept_when_main_is_missing_but_not_a_dependency() -> None:
    check = module_check()
    missing_main = {"traceback": "ImportError: No module named colsort.__main__; 'colsort' is a "
                    "package and cannot be directly executed"}
    assert task_checks.keep_at_baseline(check, missing_main, "") == ""
    missing_dependency = {"traceback": "ModuleNotFoundError: No module named 'pandas'"}
    assert task_checks.keep_at_baseline(check, missing_dependency, "") != ""


@needs_sandbox
def test_a_stdin_crash_is_reported_and_a_fix_passes(tmp_path: Path) -> None:
    make_colsort(tmp_path)
    checks = task_checks.prepare(tmp_path, ROWSIFT_TASK, rules=True)
    assert len(checks.active) == 2
    # before the change the file argument is still required, which the task says it must not be
    assert all("usage error with input on stdin" in line
               for line in checks.evaluate(tmp_path, project_failing=1))
    broken = ROWSIFT_CLI.replace('parser.add_argument("file")',
                                 'parser.add_argument("file", nargs="?", const="-")')
    make_colsort(tmp_path, broken)  # stdin without FILE opens None
    lines = checks.evaluate(tmp_path, project_failing=1)
    assert len(lines) == 2 and all("TypeError" in line for line in lines)
    assert "stdin was: name,qty / bolt,12" in lines[0]
    fixed = broken.replace('const="-"', 'default="-"').replace(
        'with open(args.file, newline="", encoding="utf-8") as handle:\n        rows = list('
        'csv.reader(handle))',
        'handle = sys.stdin if args.file == "-" else open(args.file, encoding="utf-8")\n'
        '    rows = list(csv.reader(handle))')
    assert fixed != broken.replace('const="-"', 'default="-"')
    make_colsort(tmp_path, fixed)
    assert checks.evaluate(tmp_path, project_failing=1) == []


@needs_sandbox
def test_the_readme_example_fails_on_the_wrong_code_and_passes_on_the_right(tmp_path) -> None:
    from vega import acceptance

    make_clockwork(tmp_path)
    spec = acceptance.extract(tmp_path, TICKWISE_TASK)
    item = acceptance._command_item(spec.commands[0], tmp_path, None, "command")
    assert not item.ok and "invalid name 'mon-fri'" in item.detail
    make_clockwork(tmp_path, TICKWISE_MAIN.replace('"mon" in', '"never" in'))
    assert acceptance._command_item(spec.commands[0], tmp_path, None, "command").ok
