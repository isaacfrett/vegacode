"""Project-diagnosis routing fixes (2026-10-20):

- fix 1, route_cli_target / route_prose_class: a single-target direct route also changes the CLI
  file a backticked command runs, and the file of a class named in prose; then it is a project
  route;
- fix 5, question_change_cues: "report" is a question cue only as a verb, and "must"
  requirements are a change; question_code_guard: a code answer to a change task falls back.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import features, question_pipeline, routing
from vega.agent import CodingAgent

BASE_OFF = "clarify_first,project_context"


@pytest.fixture(autouse=True)
def defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(features.ENV, BASE_OFF)


def _off(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    monkeypatch.setenv(features.ENV, ",".join([BASE_OFF, *names]))


def test_switches_are_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(features.ENV, raising=False)
    for name in ("route_cli_target", "route_prose_class", "question_change_cues",
                 "question_code_guard"):
        assert features.on(name)


# -- fix 1: the CLI file and the class named in prose -------------------------------------------


def _cli_project(root: Path, *, tests: bool = True, prog: str = 'prog="tool"') -> Path:
    (root / "tool").mkdir()
    (root / "tool" / "__init__.py").write_text("")
    (root / "tool" / "core.py").write_text("def total(items):\n    return sum(items)\n")
    (root / "tool" / "cli.py").write_text(
        "import argparse\nfrom .core import total\n\n\ndef main(argv=None):\n"
        f"    parser = argparse.ArgumentParser({prog})\n"
        "    parser.add_argument('numbers', nargs='*', type=int)\n"
        "    print(total(parser.parse_args(argv).numbers))\n")
    if tests:
        (root / "tests").mkdir()
        (root / "tests" / "test_core.py").write_text(
            "from tool.core import total\n\n\ndef test_total():\n    assert total([1, 2]) == 3\n")
    return root


TASK = ("Make `total(items)` in tool/core.py skip None values, and give `tool sum --skip-none` "
        "a flag for it.")


def test_cli_file_from_argparse_prog(tmp_path: Path) -> None:
    project = _cli_project(tmp_path)
    assert routing.cli_files(project, TASK, ["tool/__init__.py", "tool/core.py",
                                             "tool/cli.py"]) == ["tool/cli.py"]
    route = routing.route_task(TASK, project)
    assert (route.name, route.targets) == (routing.PROJECT, ["tool/core.py", "tool/cli.py"])


def test_cli_file_from_a_console_script(tmp_path: Path) -> None:
    project = _cli_project(tmp_path, prog="")
    (project / "pyproject.toml").write_text(
        '[project]\nname = "tool"\n\n[project.scripts]\ntool = "tool.cli:main"\n')
    files = ["tool/__init__.py", "tool/core.py", "tool/cli.py"]
    assert routing.cli_files(project, TASK, files) == ["tool/cli.py"]


def test_no_cli_file_for_a_tool_command_or_an_unknown_or_ambiguous_prog(tmp_path: Path) -> None:
    project = _cli_project(tmp_path)
    (project / "tool" / "other.py").write_text("import argparse\nP = argparse.ArgumentParser("
                                               "prog='tool')\n")
    files = ["tool/core.py", "tool/cli.py", "tool/other.py"]
    assert routing.cli_files(project, TASK, files) == []  # two files claim `tool`
    assert routing.cli_files(project, "Run `python -m pytest -q` after `git add x`.", files) == []
    assert routing.cli_files(project, "Make `tool` faster.", files) == []  # not a command line


def test_kept_direct_when_the_wider_route_is_not_the_project_pipeline(tmp_path: Path) -> None:
    project = _cli_project(tmp_path, tests=False)
    task = "In tool/core.py, make total skip None values so tool sum --skip-none works."
    assert routing.route_task(task, project).name == routing.DIRECT  # no backticked command
    route = routing.route_task("In tool/core.py, make total skip None values; `tool sum 1 2` "
                               "should still print 3.", project)
    # no tests, no acceptance checks: two targets would go to the agent loop, so it stays direct
    assert (route.name, route.targets) == (routing.DIRECT, ["tool/core.py"])
    assert "kept direct: tool/cli.py would route to agent" in route.reasons


def test_prose_class_needs_one_defining_file_and_camel_case(tmp_path: Path) -> None:
    project = _cli_project(tmp_path)
    (project / "tool" / "http.py").write_text("class HttpClient:\n    pass\n\n\nclass Store:\n"
                                              "    pass\n")
    files = ["tool/core.py", "tool/cli.py", "tool/http.py"]
    assert routing.prose_classes(project, "Retry in the HttpClient when it fails.", files) \
        == ["tool/http.py"]
    assert routing.prose_classes(project, "The Store must keep order.", files) == []  # one hump
    assert routing.prose_classes(project, "Do not touch HttpClient; fix core.", files) == []
    (project / "tool" / "copy.py").write_text("class HttpClient:\n    pass\n")
    assert routing.prose_classes(project, "Retry in the HttpClient.",
                                 [*files, "tool/copy.py"]) == []


# -- fix 5: "report" as a noun, "must" requirements, the code-answer guard -----------------------


@pytest.mark.parametrize("task, cue", [
    ("Find cli.py anywhere in this project and report the marker string inside it.", True),
    ("Review src/auth.py. Only report it; do not change any files.", True),
    ("Report every real bug in the diff.", True),
    ("The stock report must also mark every low item.", False),
    ("the report ends with a total row", False),
    ("histokit/report.py crashes on empty input", False),
])
def test_report_is_a_cue_only_as_a_verb(task: str, cue: bool) -> None:
    assert bool(routing._ASKS_VERB.search(routing.strip_code(task))) is cue


@pytest.mark.parametrize("task, found", [
    ("The stock report must also mark every item below its reorder level.", True),
    ("Totals must be rounded half-up. Send one alert.", True),
    ("Further sales must not send more alerts.", False),  # a constraint, negated
    ("Which value must `retries` have for the sync to pass?", False),  # a question
    ("Where is the cache folder set? Your answer must name the file.", False),  # about the reply
    ("Explain how `load()` works.", False),
])
def test_must_requirements(task: str, found: bool) -> None:
    assert bool(routing.requirements(task)) is found


def test_a_must_task_is_never_a_question_unless_read_only(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "shop.py").write_text("def describe_order(order):\n    return str(order)\n")
    task = "Describe orders by date: `describe_order()` must list the newest first."
    route = routing.route_task(task, tmp_path)
    assert route.change and route.name != routing.QUESTION
    assert "states requirements with must: treated as a change" in route.reasons
    read_only = ("Explain `describe_order()`; the answer must name the file. "
                 "Do not change any files.")
    assert routing.route_task(read_only, tmp_path).question
    _off(monkeypatch, "question_change_cues")
    assert routing.route_task(task, tmp_path).question


CODE_ANSWER = ("Add this to inventory/warehouse.py:\n\n```python\ndef sell(self, sku, qty):\n"
               "    item = self.items[sku]\n    item.qty -= qty\n```\n")
MUST_TASK = "The stock report must also mark every item below its reorder level with `LOW`."


def test_code_for_change() -> None:
    assert question_pipeline.code_for_change(MUST_TASK, CODE_ANSWER)
    assert not question_pipeline.code_for_change(MUST_TASK, "It is in inventory/report.py.")
    assert not question_pipeline.code_for_change(
        "Where is `sell` defined?", CODE_ANSWER)  # a question may quote code
    short = "```\nLOW\n```"
    assert not question_pipeline.code_for_change(MUST_TASK, short)
    long_block = "```\n" + "\n".join(f"x{i} = {i}" for i in range(6)) + "\n```"
    assert question_pipeline.code_for_change(MUST_TASK, long_block)


def _scripted(root: Path, reply: str) -> CodingAgent:
    agent = CodingAgent(root, allow_commands=False)

    def create(**kwargs):
        message = SimpleNamespace(content=reply, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent


def test_a_code_answer_to_a_change_task_falls_back(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "inventory").mkdir()
    (tmp_path / "inventory" / "warehouse.py").write_text("class Warehouse:\n    pass\n")
    agent = _scripted(tmp_path, CODE_ANSWER)
    seen = []
    agent.on_event = lambda kind, data: seen.append((kind, data.get("message", "")))
    assert question_pipeline.run(agent, MUST_TASK) is None
    assert ("notice", "The answer is code for a change; making the change instead.") in seen
    assert len(agent.messages) == 1  # nothing added: the agent loop takes the task

    _off(monkeypatch, "question_code_guard")
    agent = _scripted(tmp_path, CODE_ANSWER)
    assert question_pipeline.run(agent, MUST_TASK) == CODE_ANSWER.strip()
