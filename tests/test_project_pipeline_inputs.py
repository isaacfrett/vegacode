"""Broad sample 2026-10-17, fix 7: the project pipeline's inputs, large files, honest answers,
pytest exit code 5 and the script the task runs. Scripted model, the real sandbox; each switch on
(the default) and off (the old behaviour)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import project_pipeline, routing, sandbox
from vega.agent import CodingAgent

pytestmark = pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")


@pytest.fixture(autouse=True)
def defaults_on(monkeypatch) -> None:
    monkeypatch.delenv("VEGA_FEATURES_OFF", raising=False)  # conftest turns some off


def scripted(root: Path, replies: list[str]) -> tuple[CodingAgent, list[dict]]:
    agent = CodingAgent(root, allow_commands=False)
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        content = replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, calls


def prompt(call: dict) -> str:
    return call["messages"][-1]["content"]


def plan(*items: tuple[str, str]) -> str:
    return json.dumps({"analysis": "x", "files": [{"path": p, "change": c} for p, c in items]})


# -- project_inputs ---------------------------------------------------------------------------------

COMMITS = "".join(f"commit {i:07x}\n    feat: feature number {i}\n\n" for i in range(40))
RULES = "# Releasing\nA `feat` bumps the minor version. Add a section to CHANGELOG.md.\n"


def release_project(root: Path) -> str:
    (root / "app").mkdir()
    (root / "app" / "__init__.py").write_text('__version__ = "1.2.0"\n')
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## [1.2.0]\n\n- old\n")
    (root / "commits.txt").write_text(COMMITS)
    (root / "RELEASING.md").write_text(RULES)
    (root / "tests").mkdir()
    (root / "tests" / "test_version.py").write_text(
        "import app\n\n\ndef test_version():\n    assert app.__version__.count('.') == 2\n")
    return ("`commits.txt` is the git log since the last release. Following `RELEASING.md`, add "
            "the new section to `CHANGELOG.md` and bump the version in `app/__init__.py`. Do not "
            "change `commits.txt` or `RELEASING.md`.")


RELEASE_PLAN = plan(("CHANGELOG.md", "add the 1.3.0 section"), ("app/__init__.py", "1.3.0"))
RELEASE_WRITES = [("File: CHANGELOG.md\n```markdown\n# Changelog\n\n## [1.3.0]\n\n- new\n\n"
                  "## [1.2.0]\n\n- old\n```"),
                  'File: app/__init__.py\n```python\n__version__ = "1.3.0"\n```']


def test_route_inputs_reach_the_planner_and_every_writer(tmp_path: Path) -> None:
    task = release_project(tmp_path)
    assert {"commits.txt", "RELEASING.md"} <= set(routing.route_task(task, tmp_path).inputs)
    agent, calls = scripted(tmp_path, [RELEASE_PLAN, *RELEASE_WRITES])
    answer = project_pipeline.run(agent, task)
    assert answer == "Changed CHANGELOG.md, app/__init__.py; the project's tests pass."
    assert len(calls) == 3
    for call in calls:  # the planner and both writers see the inputs, whole (they fit)
        assert "commit 0000027\n    feat: feature number 39" in prompt(call)
        assert "A `feat` bumps the minor version." in prompt(call)
    # a planned file is shown as the file being written, not again as an input
    assert prompt(calls[1]).count("## [1.2.0]") == 1
    assert (tmp_path / "commits.txt").read_text() == COMMITS  # never written


def test_inputs_switch_off_is_the_old_prompt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "project_inputs")
    task = release_project(tmp_path)
    agent, calls = scripted(tmp_path, [RELEASE_PLAN, *RELEASE_WRITES])
    project_pipeline.run(agent, task)
    assert not any("feature number 39" in prompt(call) for call in calls)


def test_inputs_fit_the_budget_shortest_first(tmp_path: Path) -> None:
    (tmp_path / "short.md").write_text("short rules\n")
    (tmp_path / "long.md").write_text("".join(f"line {i}\n" for i in range(3000)))
    text = project_pipeline.inputs_text(tmp_path, ["long.md", "short.md"], budget=2000)
    assert "short rules" in text  # whole: the short file takes only what it needs
    assert "[... cut: long.md has 28,890 characters]" in text
    assert len(text) < 2000 + 400  # the budget plus the headers and fences
    (tmp_path / "rows.csv").write_text("n\n" + "".join(f"{i}\n" for i in range(3000)))
    text = project_pipeline.inputs_text(tmp_path, ["rows.csv"], budget=2000)
    assert "Preview of rows.csv" in text  # a data file too long to show: its preview
    assert project_pipeline.inputs_text(tmp_path, ["short.md"], exclude={"short.md"}) == ""


def test_only_non_python_inputs_that_exist(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("x")
    (tmp_path / "helper.py").write_text("x = 1\n")
    route = {"inputs": ["notes.md", "helper.py", "missing.csv"]}
    assert project_pipeline.read_only_inputs(tmp_path, route) == ["notes.md"]


# -- project_large_files and project_honest_answer -----------------------------------------------------

def large_project(root: Path) -> tuple[str, str]:
    filler = "".join(f"def helper_{i}(x):\n    return x + {i}   \n\n\n" for i in range(400))
    source = filler + "def parse(text):\n    return int(text)\n"
    assert len(source) > project_pipeline.FILE_CHARS
    (root / "big.py").write_text(source)
    (root / "tests").mkdir()
    (root / "tests" / "test_big.py").write_text(
        "from big import parse\n\n\ndef test_parse():\n    assert parse('7') == 7\n")
    return source, "Make `big.parse` accept surrounding spaces: `parse(' 7 ')` returns 7."


LARGE_PLAN = plan(("big.py", "parse: strip the text first"))
LARGE_WRITE = "File: big.py\n```python\ndef parse(text):\n    return int(text.strip())\n```"


def test_large_python_file_is_written_by_definition(tmp_path: Path) -> None:
    source, task = large_project(tmp_path)
    agent, calls = scripted(tmp_path, [LARGE_PLAN, LARGE_WRITE])
    answer = project_pipeline.run(agent, task)
    assert answer == "Changed big.py; the project's tests pass."
    text = (tmp_path / "big.py").read_text()
    assert text == source.replace("return int(text)", "return int(text.strip())")  # the rest kept
    assert "is large" in prompt(calls[1]) and "def parse(text)" in prompt(calls[1])
    assert len(prompt(calls[1])) < project_pipeline.WRITER_PROMPT_CHARS


def test_large_file_switch_off_says_nothing_was_written(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "project_large_files")
    source, task = large_project(tmp_path)
    agent, _ = scripted(tmp_path, [LARGE_PLAN])
    answer = project_pipeline.run(agent, task)
    assert (tmp_path / "big.py").read_text() == source
    assert answer == ("No file was changed, so the task is not done. Left unchanged: big.py "
                      "(too large to rewrite whole).")


def test_both_off_is_the_old_false_claim(tmp_path: Path, monkeypatch) -> None:
    """The bug the honest answer fixes: nothing written, 'Changed'."""
    monkeypatch.setenv("VEGA_FEATURES_OFF", "project_large_files,project_honest_answer")
    source, task = large_project(tmp_path)
    agent, _ = scripted(tmp_path, [LARGE_PLAN])
    answer = project_pipeline.run(agent, task)
    assert (tmp_path / "big.py").read_text() == source
    assert answer == "Changed big.py; the project's tests pass."


def test_honest_answer_names_a_planned_file_that_was_never_saved(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(project_pipeline, "REPAIRS", 0)
    task = release_project(tmp_path)
    no_block = "I would add a section for 1.3.0."
    agent, _ = scripted(tmp_path, [RELEASE_PLAN, no_block, no_block, RELEASE_WRITES[1]])
    answer = project_pipeline.run(agent, task)
    assert answer.startswith("Changed app/__init__.py, but after 0 repair rounds these still fail:")
    assert answer.endswith("\nLeft unchanged: CHANGELOG.md (Your reply had no code block for "
                           "CHANGELOG.md).")
    assert "Changed CHANGELOG.md" not in answer


# -- project_no_tests ---------------------------------------------------------------------------------

def untested_project(root: Path) -> str:
    (root / "calc.py").write_text("def double(x):\n    return x\n")
    return "Fix `calc.double(x)` in `calc.py` so that it returns twice x."


CALC_PLAN = plan(("calc.py", "double returns 2 * x"))
CALC_WRITE = "File: calc.py\n```python\ndef double(x):\n    return 2 * x\n```"


def test_no_tests_collected_is_not_a_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "acceptance")
    task = untested_project(tmp_path)
    agent, calls = scripted(tmp_path, [CALC_PLAN, CALC_WRITE])
    answer = project_pipeline.run(agent, task)
    assert len(calls) == 2  # no repair round for "no tests ran"
    assert answer == "Changed calc.py; the project has no tests, so no test ran."


def test_no_tests_switch_off_spends_the_repair_rounds(tmp_path: Path, monkeypatch) -> None:
    # unchanged_rounds off too: every round here rewrites the same file
    monkeypatch.setenv("VEGA_FEATURES_OFF", "acceptance,project_no_tests,unchanged_rounds")
    task = untested_project(tmp_path)
    agent, calls = scripted(tmp_path, [CALC_PLAN, CALC_WRITE] * (project_pipeline.REPAIRS + 1))
    answer = project_pipeline.run(agent, task)
    assert len(calls) == 2 * (project_pipeline.REPAIRS + 1)
    assert "still fail" in answer


def test_an_unchanged_round_tells_the_writer_then_ends_the_loop(tmp_path: Path,
                                                                monkeypatch) -> None:
    # unchanged_rounds: round 2's write is byte-identical -> no checks and no replan, its writer
    # is told; round 3 identical again -> the loop ends (5 calls instead of 8)
    monkeypatch.setenv("VEGA_FEATURES_OFF", "acceptance,project_no_tests")
    task = untested_project(tmp_path)
    agent, calls = scripted(tmp_path, [CALC_PLAN, CALC_WRITE, CALC_PLAN, CALC_WRITE, CALC_WRITE])
    answer = project_pipeline.run(agent, task)
    assert len(calls) == 5  # plan, write (checked), replan, write (unchanged), write (unchanged)
    told = calls[4]["messages"][-1]["content"]
    assert "Your rewrite was identical to the current file" in told
    assert "Your rewrite was identical" not in calls[3]["messages"][-1]["content"]
    assert "still fail" in answer


# -- project_run_script -------------------------------------------------------------------------------

def report_project(root: Path) -> str:
    (root / "data").mkdir()
    (root / "data" / "nums.csv").write_text("n\n1\n2\n3\n4\n")
    (root / "stats.py").write_text("def mean(values):\n    raise NotImplementedError\n")
    return ("`data/nums.csv` holds numbers (`n`). 1. `stats.py`: `mean(values)` returns the "
            "arithmetic mean. 2. `report.py` reads `data/nums.csv`, writes `summary.json` and "
            "prints the same JSON: `mean`. Your tools cannot write files from a command, so run "
            "`python report.py` with run_check and save exactly the JSON it prints as "
            "`summary.json`. Do not change `data/nums.csv`.")


REPORT_PLAN = plan(("stats.py", "mean"), ("report.py", "write summary.json"),
                   ("summary.json", "the summary"))
MEAN = "File: stats.py\n```python\ndef mean(values):\n    return sum(values) / len(values)\n```"
REPORT = ("File: report.py\n```python\nimport csv\nimport json\n\nfrom stats import mean\n\n"
          "values = [float(r['n']) for r in csv.DictReader(open('data/nums.csv'))]\n"
          "result = {'mean': mean(values)}\nopen('summary.json', 'w').write(json.dumps(result))\n"
          "print(json.dumps(result))\n```")
HAND = 'File: summary.json\n```json\n{"mean": 99}\n```'


def test_route_names_the_script_and_its_output(tmp_path: Path) -> None:
    route = routing.route_task(report_project(tmp_path), tmp_path)
    assert route.helpers == ["report.py"] and route.other_targets == ["summary.json"]


def test_the_task_script_runs_and_its_output_is_saved(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "acceptance")
    task = report_project(tmp_path)
    agent, calls = scripted(tmp_path, [REPORT_PLAN, MEAN, REPORT])
    answer = project_pipeline.run(agent, task)
    assert json.loads((tmp_path / "summary.json").read_text()) == {"mean": 2.5}  # computed
    assert len(calls) == 3  # the writer was never asked for summary.json
    assert "do not list those files" in prompt(calls[0])
    assert answer == ("Changed stats.py, report.py, summary.json; the project has no tests, so no "
                      "test ran.\n`python report.py` ran in the sandbox and wrote summary.json.")
    assert (tmp_path / "data" / "nums.csv").read_text() == "n\n1\n2\n3\n4\n"


def test_a_failing_script_goes_back_to_the_writer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "acceptance")
    monkeypatch.setattr(project_pipeline, "REPAIRS", 1)
    task = report_project(tmp_path)
    broken = REPORT.replace("from stats import mean", "from stats import average as mean")
    agent, calls = scripted(tmp_path, [REPORT_PLAN, MEAN, broken, REPORT_PLAN, MEAN, REPORT])
    answer = project_pipeline.run(agent, task)
    assert "`python report.py` failed" in prompt(calls[3])  # the replan sees the traceback
    assert "ImportError" in prompt(calls[3])
    assert json.loads((tmp_path / "summary.json").read_text()) == {"mean": 2.5}
    assert answer.endswith("`python report.py` ran in the sandbox and wrote summary.json.")


def test_script_switch_off_lets_the_writer_hand_write_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "acceptance,project_run_script,project_no_tests")
    monkeypatch.setattr(project_pipeline, "REPAIRS", 0)
    task = report_project(tmp_path)
    agent, _ = scripted(tmp_path, [REPORT_PLAN, MEAN, REPORT, HAND])
    project_pipeline.run(agent, task)
    assert json.loads((tmp_path / "summary.json").read_text()) == {"mean": 99}


def test_run_script_takes_printed_json_when_no_file_is_written(tmp_path: Path) -> None:
    (tmp_path / "show.py").write_text("import json\nprint(json.dumps({'a': 1}))\n")
    files, problem = project_pipeline.run_script(tmp_path, "show.py", ["out.json"])
    assert problem == "" and json.loads(files["out.json"]) == {"a": 1}
    assert not (tmp_path / "out.json").exists()  # run_script only reports; the pipeline saves
    files, problem = project_pipeline.run_script(tmp_path, "missing.py", ["out.json"])
    assert files == {} and "missing.py does not exist" in problem
