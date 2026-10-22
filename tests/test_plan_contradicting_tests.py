"""plan_contradicting_tests: a visible test that calls a signature the task replaces is planned,
to update only those calls, and a rewrite that weakens it is refused."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import acceptance, features, project_pipeline, sandbox
from vega.agent import CodingAgent

TASK = ("Make the storage swappable: `InventoryService(repository)` takes any object with "
        "`load()` and does all of its storage through it.")
SERVICE = '''class InventoryService:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def level(self):
        return 5
'''
TEST = '''import pytest

from stock.service import InventoryService
from stock.store import Store


def test_level(tmp_path):
    service = InventoryService(tmp_path)
    assert service.level() == 5
    with pytest.raises(TypeError):
        service.level(1)


def test_other():
    assert 1 + 1 == 2
'''


@pytest.fixture(autouse=True)
def defaults_on(monkeypatch) -> None:
    monkeypatch.delenv(features.ENV, raising=False)


def make_project(root: Path, test: str = TEST) -> None:
    (root / "stock").mkdir()
    (root / "stock" / "__init__.py").write_text("")
    (root / "stock" / "service.py").write_text(SERVICE)
    (root / "stock" / "store.py").write_text(
        "class Store:\n    def __init__(self, path):\n        self.path = path\n\n"
        "    def load(self):\n        return 5\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_stock.py").write_text(test)


def test_fires_on_a_test_calling_the_replaced_signature(tmp_path: Path) -> None:
    make_project(tmp_path)
    [hit] = project_pipeline.contradicting_tests(tmp_path, TASK)
    assert hit.path == "tests/test_stock.py"
    assert hit.names == ["InventoryService"]
    assert hit.changes == ["InventoryService(data_dir) -> InventoryService(repository)"]
    assert hit.calls == ["line 8: InventoryService(tmp_path)"]
    assert project_pipeline.CALL_TEST_NOTE[1:] in hit.change()


@pytest.mark.parametrize("task", [
    # the same parameters: no interface change
    "`InventoryService(data_dir)` must also cache the levels it loads.",
    # the task says to keep the signature
    "Keep the signature of `InventoryService(repository)` and add caching.",
    # `name()` in prose is a call, not a stated empty signature
    "The report runs when `InventoryService()` is created; make level() cached.",
    # no signature stated at all
    "Make `InventoryService` cache the levels it loads.",
])
def test_quiet_when_the_task_changes_no_interface(tmp_path: Path, task: str) -> None:
    make_project(tmp_path)
    assert project_pipeline.contradicting_tests(tmp_path, task) == []


def test_quiet_when_the_test_already_calls_the_new_signature(tmp_path: Path) -> None:
    make_project(tmp_path, TEST.replace("InventoryService(tmp_path)",
                                        "InventoryService(repository=Store(tmp_path))"))
    assert project_pipeline.contradicting_tests(tmp_path, TASK) == []


def _params(text: str) -> list[acceptance.Param]:
    params = acceptance.parse_params(text)
    assert params is not None
    return params


@pytest.mark.parametrize("call, old, new, breaks", [
    ("f(x)", "data_dir", "repository", True),  # same arity, the parameter is replaced
    ("f(x)", "path", "path, *, clock=None", False),  # an added optional parameter
    ("f(x)", "path", "path, clock", True),  # a new required parameter not passed
    ("f(x, y)", "a, b", "a", True),  # one argument too many
    ("f(a=1)", "a", "b", True),  # a keyword the new signature does not take
    ("f(a=1)", "a", "a, **kw", False),
    ("f(*args)", "a", "b", False),  # starred: not judged
])
def test_call_breaks(call: str, old: str, new: str, breaks: bool) -> None:
    node = ast.parse(call).body[0].value  # type: ignore[attr-defined]
    assert project_pipeline._call_breaks(node, _params(old), _params(new)) is breaks


def test_guard_allows_only_the_changed_calls() -> None:
    names = ["InventoryService"]
    fixed = TEST.replace("InventoryService(tmp_path)", "InventoryService(Store(tmp_path))")
    assert project_pipeline.weakened_test(TEST, fixed, names) is None
    # an assertion that calls the changed name may change its call
    inline = TEST.replace("    service = InventoryService(tmp_path)\n    assert service.level() == 5",
                          "    assert InventoryService(tmp_path).level() == 5")
    moved = inline.replace("InventoryService(tmp_path).level()",
                           "InventoryService(Store(tmp_path)).level()")
    assert project_pipeline.weakened_test(inline, moved, names) is None


@pytest.mark.parametrize("edit, why", [
    (("def test_other():\n    assert 1 + 1 == 2\n", ""), "test test_other was removed"),
    (("    with pytest.raises(TypeError):\n        service.level(1)\n", ""),
     "test_level has 1 assertion(s), it had 2"),
    (("== 5", "== 6"), "test_level changed or dropped `assert service.level() == 5`"),
    (("pytest.raises(TypeError)", "pytest.raises(Exception)"), "changed or dropped"),
])
def test_guard_refuses_a_weakened_test(edit: tuple[str, str], why: str) -> None:
    after = TEST.replace(*edit).replace("InventoryService(tmp_path)",
                                        "InventoryService(Store(tmp_path))")
    refusal = project_pipeline.weakened_test(TEST, after, ["InventoryService"])
    assert refusal is not None and why in refusal
    assert refusal.startswith("Not saved:")


def test_planner_lets_only_the_detected_test_through(tmp_path: Path) -> None:
    make_project(tmp_path)
    (tmp_path / "tests" / "test_other.py").write_text("def test_x():\n    assert True\n")
    tests = {t.path: t for t in project_pipeline.contradicting_tests(tmp_path, TASK)}
    plan = json.dumps({"analysis": "x", "files": [
        {"path": "stock/service.py", "change": "take a repository"},
        {"path": "tests/test_stock.py", "change": "rewrite the tests"},
        {"path": "tests/test_other.py", "change": "edit"}]})
    agent, _ = scripted(tmp_path, [plan])
    items = project_pipeline._plan(agent, TASK, "map", call_tests=tests)
    assert [i["path"] for i in items] == ["stock/service.py", "tests/test_stock.py"]
    assert items[1]["change"] == tests["tests/test_stock.py"].change()  # not the planner's text


def test_with_call_tests_appends_once() -> None:
    test = project_pipeline.CallTest("tests/t.py", ["F"], ["F(a) -> F(b)"], ["line 1: F(x)"])
    plan = [{"path": "pkg/f.py", "change": "c"}]
    new, added = project_pipeline.with_call_tests(plan, [test])
    assert [i["path"] for i in new] == ["pkg/f.py", "tests/t.py"] and added == ["tests/t.py"]
    assert project_pipeline.with_call_tests(new, [test])[1] == []
    assert project_pipeline.failing_call_tests({"tests/t.py": test},
                                               "FAILED tests/t.py::test_a") == [test]
    assert project_pipeline.failing_call_tests({"tests/t.py": test}, "all good") == []


# -- end to end, scripted model -------------------------------------------------------------------

PLAN = json.dumps({"analysis": "The service takes a repository.",
                   "files": [{"path": "stock/service.py", "change": "take a repository"}]})
NEW_SERVICE = ("File: stock/service.py\n```python\nclass InventoryService:\n"
               "    def __init__(self, repository):\n        self.repository = repository\n\n"
               "    def level(self):\n        return self.repository.load()\n```")
RIGHT_TEST = ("File: tests/test_stock.py\n```python\ndef test_level(tmp_path):\n"
              "    service = InventoryService(Store(tmp_path))\n    assert service.level() == 5\n"
              "    with pytest.raises(TypeError):\n        service.level(1)\n```")
WEAK_TEST = ("File: tests/test_stock.py\n```python\ndef test_level(tmp_path):\n"
             "    service = InventoryService(Store(tmp_path))\n```")


def scripted(root: Path, replies: list[str]) -> tuple[CodingAgent, list[dict]]:
    agent = CodingAgent(root, allow_commands=False)
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=replies.pop(0)))])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, calls


needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")


@needs_sandbox
def test_end_to_end_the_test_call_is_updated(tmp_path: Path) -> None:
    make_project(tmp_path)
    agent, calls = scripted(tmp_path, [PLAN, NEW_SERVICE, RIGHT_TEST])
    answer = project_pipeline.run(agent, TASK)
    assert answer is not None
    test = (tmp_path / "tests" / "test_stock.py").read_text()
    assert "InventoryService(Store(tmp_path))" in test and "def test_other" in test
    assert "pytest.raises(TypeError)" in test
    writer = calls[2]["messages"][-1]["content"]
    assert "Now write tests/test_stock.py" in writer and "keep every assertion" in writer
    assert len(calls) == 3  # one round: everything passes


@needs_sandbox
def test_end_to_end_a_weakened_test_is_refused_then_fixed(tmp_path: Path) -> None:
    make_project(tmp_path)
    agent, calls = scripted(tmp_path, [PLAN, NEW_SERVICE, WEAK_TEST, PLAN, NEW_SERVICE, RIGHT_TEST])
    project_pipeline.run(agent, TASK)
    test = (tmp_path / "tests" / "test_stock.py").read_text()
    assert "InventoryService(Store(tmp_path))" in test and "pytest.raises(TypeError)" in test
    assert "assert service.level() == 5" in test
    retry = calls[5]["messages"][-1]["content"]
    assert "weakens it" in retry  # the refusal goes back to the test's writer


@needs_sandbox
def test_switch_off_keeps_tests_out_of_the_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV, "plan_contradicting_tests")
    make_project(tmp_path)
    agent, calls = scripted(tmp_path, [PLAN, NEW_SERVICE] + [PLAN, NEW_SERVICE] * 4)
    project_pipeline.run(agent, TASK)
    assert "InventoryService(tmp_path)" in (tmp_path / "tests" / "test_stock.py").read_text()
    assert not any("Now write tests/" in c["messages"][-1]["content"] for c in calls)
