"""Candidate collection, review and eval/train separation, with a scripted fake model."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from private_data import require_training_tasks
from typer.testing import CliRunner

from vega import evals, training
from vega.agent import ACT_NUDGE, CodingAgent

runner = CliRunner()

EVAL_CASE = {
    "id": "eval-only-case", "task": "Fix the obvious bug in difference(). Do not run commands.",
    "files": {"src/math_ops.py": "def difference(a, b):\n    return a + b\n"},
    "python_check": "from math_ops import difference\nassert difference(5, 3) == 2\n",
}
SUB_FIXTURE = {"calc.py": "def sub(a, b):\n    return a + b\n"}
SUB_CHECK = "from calc import sub\nassert sub(5, 3) == 2\n"
TASKS = [
    {"id": "fix-sub", "task": "sub() adds instead of subtracting. Fix it.", "files": SUB_FIXTURE,
     "python_check": SUB_CHECK},
    {"id": "fix-mul", "task": "sub() is wrong again. Fix it please.", "files": SUB_FIXTURE,
     "python_check": SUB_CHECK},
]


class FakeMessage(SimpleNamespace):
    def model_dump(self, exclude_none: bool = False) -> dict:
        data = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in self.tool_calls
            ]
        return data


def text(content: str) -> FakeMessage:
    return FakeMessage(content=content, tool_calls=None)


def call(name: str, call_id: str, **args) -> FakeMessage:
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    return FakeMessage(content=None, tool_calls=[SimpleNamespace(id=call_id, function=fn)])


def scripted_factory(scripts: list[list[FakeMessage]]):
    """Each new agent (one per attempt) gets the next scripted conversation."""

    def factory(project: Path) -> CodingAgent:
        replies = scripts.pop(0)
        agent = CodingAgent(project, allow_commands=False)

        def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=replies.pop(0))])

        completions = SimpleNamespace(create=create)
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        return agent

    return factory


def fixing_script() -> list[FakeMessage]:
    return [
        call("read_file", "c1", path="calc.py"),
        call("edit_file", "c2", path="calc.py", old_text="a + b", new_text="a - b"),
        text("Fixed sub() in calc.py: it now returns a - b."),
    ]


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_root))
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text("".join(json.dumps(t) + "\n" for t in TASKS))
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(json.dumps(EVAL_CASE) + "\n")
    monkeypatch.setattr(training, "TASKS_FILE", tasks_file)
    monkeypatch.setattr(training, "CANDIDATES_DIR", tmp_path / "candidates")
    monkeypatch.setattr(training, "EXAMPLES_FILE", tmp_path / "examples.jsonl")
    monkeypatch.setattr(training, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evals, "CASES_FILE", cases_file)
    return tmp_path


def candidate_files(workspace: Path) -> list[str]:
    return sorted(p.name for p in (workspace / "candidates").glob("*.json"))


def test_collect_writes_candidate_only_for_passing_attempt(workspace: Path) -> None:
    failing = [call("read_file", "c1", path="calc.py"), text("The code looks correct to me.")]
    rows = training.collect_candidates(1, None, scripted_factory([fixing_script(), failing]))
    summary = [(r["task"], r["passed"], r["written"]) for r in rows]
    assert summary == [("fix-sub", 1, 1), ("fix-mul", 0, 0)]
    assert candidate_files(workspace) == ["fix-sub-1.json"]

    example = json.loads((workspace / "candidates" / "fix-sub-1.json").read_text())
    training._validate_example(example, 1)
    roles = [m["role"] for m in example["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert example["messages"][1]["content"] == TASKS[0]["task"]
    call_message = example["messages"][4]
    assert json.loads(call_message["tool_calls"][0]["function"]["arguments"])["new_text"] == "a - b"
    assert example["messages"][-1]["content"].startswith("Fixed sub()")
    names = {tool["function"]["name"] for tool in example["tools"]}
    assert {"read_file", "edit_file", "search_text"} <= names


def test_candidate_uses_placeholder_not_temp_path(workspace: Path) -> None:
    training.collect_candidates(1, ["fix-sub"], scripted_factory([fixing_script()]))
    raw = (workspace / "candidates" / "fix-sub-1.json").read_text()
    example = json.loads(raw)
    assert f"Project root: {training.WORKSPACE_PLACEHOLDER}\n" in example["messages"][0]["content"]
    assert str(workspace) not in raw and "vega-train-" not in raw


def test_build_trace_replaces_paths_everywhere() -> None:
    root = "/private/var/folders/x/vega-train-abc"
    messages = [
        {"role": "system", "content": f"Project root: {root}"},
        {"role": "user", "content": "Read it."},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "read_file", "arguments": json.dumps({"path": f"{root}/a.py"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": f"{root}/a.py: x = 1"},
        {"role": "assistant", "content": "x is 1."},
    ]
    trace = training.build_trace(messages, [], [root, "/var/folders/x/vega-train-abc"])
    assert "/var/folders" not in json.dumps(trace)
    args = json.loads(trace["messages"][2]["tool_calls"][0]["function"]["arguments"])
    assert args == {"path": "/workspace/project/a.py"}
    assert trace["messages"][3]["content"] == "/workspace/project/a.py: x = 1"


def test_harness_nudge_and_provoking_turn_are_stripped(workspace: Path) -> None:
    script = [text("Which file should I look at?")] + fixing_script()
    rows = training.collect_candidates(1, ["fix-sub"], scripted_factory([script]))
    assert rows[0]["written"] == 1
    raw = (workspace / "candidates" / "fix-sub-1.json").read_text()
    assert "(Vega harness)" not in raw and "Which file" not in raw
    example = json.loads(raw)
    assert [m["role"] for m in example["messages"]][:3] == ["system", "user", "assistant"]
    assert example["messages"][2]["tool_calls"][0]["function"]["name"] == "read_file"


def test_trace_left_invalid_by_stripping_is_skipped() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": ACT_NUDGE},
    ]
    trace = training.build_trace(messages, [], [])
    assert [m["role"] for m in trace["messages"]] == ["system", "user"]
    assert training._candidate_problem(trace)


def test_passing_attempt_without_tool_calls_is_not_kept(workspace: Path) -> None:
    # The check passes (a no-op task), but a text-only answer teaches no tool use.
    task = {"id": "say-hi", "task": "Say hi.", "files": {"a.txt": "x"}, "response_contains": ["hi"]}
    training.TASKS_FILE.write_text(json.dumps(task) + "\n")
    rows = training.collect_candidates(1, None, scripted_factory([[text("hi"), text("hi")]]))
    assert rows[0]["passed"] == 1 and rows[0]["written"] == 0
    assert rows[0]["skipped"] == ["no tool calls"]
    assert candidate_files(workspace) == []


def test_repeat_numbers_candidates(workspace: Path) -> None:
    training.collect_candidates(2, ["fix-sub"], scripted_factory([fixing_script() for _ in "ab"]))
    assert candidate_files(workspace) == ["fix-sub-1.json", "fix-sub-2.json"]


def test_review_accept_and_reject_move_files_and_append_once(workspace: Path) -> None:
    training.collect_candidates(2, ["fix-sub"], scripted_factory([fixing_script() for _ in "ab"]))
    listing = runner.invoke(training.training_app, ["review"])
    assert listing.exit_code == 0 and "fix-sub-1.json" in listing.output

    accepted = runner.invoke(training.training_app, ["review", "--accept", "fix-sub-1.json"])
    assert accepted.exit_code == 0, accepted.output
    rejected = runner.invoke(training.training_app, ["review", "--reject", "fix-sub-2.json"])
    assert rejected.exit_code == 0, rejected.output

    lines = training.EXAMPLES_FILE.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["messages"][1]["content"] == TASKS[0]["task"]
    assert candidate_files(workspace) == []
    assert (workspace / "candidates" / "accepted" / "fix-sub-1.json").is_file()
    assert (workspace / "candidates" / "rejected" / "fix-sub-2.json").is_file()

    again = runner.invoke(training.training_app, ["review", "--accept", "fix-sub-1.json"])
    assert again.exit_code != 0
    assert len(training.EXAMPLES_FILE.read_text().splitlines()) == 1


def test_accept_refuses_candidate_that_leaks_an_eval_task(workspace: Path) -> None:
    training.collect_candidates(1, ["fix-sub"], scripted_factory([fixing_script()]))
    path = workspace / "candidates" / "fix-sub-1.json"
    example = json.loads(path.read_text())
    example["messages"][1]["content"] = EVAL_CASE["task"]
    path.write_text(json.dumps(example))
    result = runner.invoke(training.training_app, ["review", "--accept", path.name])
    assert result.exit_code != 0
    assert not training.EXAMPLES_FILE.exists() and path.is_file()


def test_validate_passes_without_overlap(workspace: Path) -> None:
    result = runner.invoke(training.training_app, ["validate"])
    assert result.exit_code == 0, result.output
    assert "No eval overlap" in result.output


@pytest.mark.parametrize("field", ["id", "task"])
def test_validate_rejects_training_task_that_matches_eval(workspace: Path, field: str) -> None:
    leaked = dict(TASKS[0], **{field: EVAL_CASE[field]})
    training.TASKS_FILE.write_text(json.dumps(leaked) + "\n")
    result = runner.invoke(training.training_app, ["validate"])
    assert result.exit_code == 1
    assert "eval-only-case" in result.output


def test_validate_rejects_example_whose_user_message_is_an_eval_task(workspace: Path) -> None:
    example = {"messages": [{"role": "user", "content": "  " + EVAL_CASE["task"].upper()},
                            {"role": "assistant", "content": "Done."}]}
    training.EXAMPLES_FILE.write_text(json.dumps(example) + "\n")
    result = runner.invoke(training.training_app, ["validate"])
    assert result.exit_code == 1
    assert "example 1" in result.output


def test_real_task_pool_is_separate_from_real_eval_cases() -> None:
    require_training_tasks()
    tasks = training._load_tasks()
    assert len(tasks) >= 40
    assert training._overlap_problems(tasks, [], evals._load_cases()) == []
