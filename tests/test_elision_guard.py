"""elision_guard: a write that stands a comment or `...` in for code the file had is refused.

The refused texts are shaped like the logged ones (debug logs 2026-10-15..27, elision-guard
review of 2026-10-27); the allowed ones are the legal uses that must pass."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import elision, features, pipeline, routing, sandbox
from vega.agent import CodingAgent
from vega.tools import execute, plan_write

STORE = '''"""Orders kept in memory."""

STATUSES = ("new", "paid", "shipped")


class Store:
    def __init__(self):
        self.orders = {}

    def add(self, order_id, total):
        if order_id in self.orders:
            raise KeyError(order_id)
        self.orders[order_id] = {"total": total, "status": "new"}

    def pay(self, order_id):
        order = self.orders[order_id]
        order["status"] = "paid"
        return order


def run_check(command, timeout=10):
    argv = command.split()
    if not argv:
        return "empty command"
    workdir = make_workdir()
    try:
        result = launch(argv, workdir, timeout)
        output = result.stdout
        if len(output) > 4000:
            output = output[:4000] + "[truncated]"
        return f"exit code: {result.returncode}\\n{output}"
    finally:
        cleanup(workdir)
'''


@pytest.fixture(autouse=True)
def guard_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(features.ENV, raising=False)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "store.py").write_text(STORE)
    return tmp_path


def store(project: Path) -> str:
    return (project / "shop" / "store.py").read_text()


def test_rest_unchanged_comment_in_a_whole_file_write_is_refused(project: Path) -> None:
    content = ('"""Orders kept in memory."""\n\nSTATUSES = ("new", "paid", "shipped", "void")\n\n'
               "# ... rest of the code unchanged\n")
    result = execute("write_file", {"path": "shop/store.py", "content": content}, project)
    assert result.startswith("Write not applied: line 5 of the new shop/store.py")
    assert "# ... rest of the code unchanged" in result and "`Store`" in result
    assert "replace_definition" in result
    assert store(project) == STORE


def test_original_body_continues_in_replace_definition_is_refused(project: Path) -> None:
    code = ("def run_check(command, timeout=10):\n    argv = command.split()\n"
            "    if not argv:\n        return 'no command given'\n"
            "    # ... original body continues ...\n")
    result = execute("replace_definition", {"path": "shop/store.py", "name": "run_check",
                                            "code": code}, project)
    assert result.startswith("Write not applied:")
    assert "in `run_check`" in result and "lines of `run_check`" in result
    assert "whole definition" in result
    assert store(project) == STORE


def test_class_body_elided_by_other_methods_comment_is_refused(project: Path) -> None:
    content = STORE.replace(
        '    def add(self, order_id, total):\n        if order_id in self.orders:\n'
        '            raise KeyError(order_id)\n'
        '        self.orders[order_id] = {"total": total, "status": "new"}\n\n'
        '    def pay(self, order_id):\n        order = self.orders[order_id]\n'
        '        order["status"] = "paid"\n        return order\n',
        "    def cancel(self, order_id):\n        del self.orders[order_id]\n\n"
        "    # Other methods and attributes...\n")
    result = execute("write_file", {"path": "shop/store.py", "content": content}, project)
    assert result.startswith("Write not applied:") and "`Store`" in result
    assert store(project) == STORE


def test_bare_ellipsis_body_over_real_code_is_refused(project: Path) -> None:
    result = execute("replace_definition", {"path": "shop/store.py", "name": "Store.pay",
                                            "code": "def pay(self, order_id):\n    ...\n"},
                     project)
    assert result.startswith("Write not applied:") and "body of `Store.pay`" in result
    assert store(project) == STORE


def test_marker_dropped_by_the_merge_still_counts(project: Path) -> None:
    """replace_definition keeps no trailing comment: the marker is only in the text sent."""
    code = ("def run_check(command, timeout=10):\n    argv = command.split()\n"
            "    workdir = make_workdir()\n    # Additional setup logic follows ...\n")
    result = execute("replace_definition", {"path": "shop/store.py", "name": "run_check",
                                            "code": code}, project)
    assert result.startswith("Write not applied: line 4 of the text you sent")
    assert store(project) == STORE


def test_new_file_with_stubs_is_allowed(project: Path) -> None:
    content = ("from typing import Protocol\n\n\nclass Clock(Protocol):\n"
               "    def now(self) -> float: ...\n\n\n"
               "def later():\n    pass  # ... filled in by the next task\n")
    result = execute("write_file", {"path": "shop/clock.py", "content": content}, project)
    assert result.startswith("Wrote shop/clock.py")


def test_protocol_and_overload_stubs_in_an_existing_file_are_allowed(project: Path) -> None:
    content = STORE.replace(
        "class Store:",
        "from typing import Protocol, overload\n\n\nclass Ledger(Protocol):\n"
        "    def record(self, order_id: str) -> None: ...\n\n\n"
        "@overload\ndef total(x: int) -> int: ...\n@overload\ndef total(x: str) -> str: ...\n"
        "def total(x):\n    return x\n\n\nclass Store:")
    result = execute("write_file", {"path": "shop/store.py", "content": content}, project)
    assert result.startswith("Wrote shop/store.py"), result


def test_ellipsis_in_strings_and_kept_code_comments_are_allowed(project: Path) -> None:
    content = STORE.replace(
        '"""Orders kept in memory."""',
        '"""Orders kept in memory (ids 1, 2, ...)."""\n\n'
        '# Existing imports (unchanged)\nimport json').replace(
        '        order["status"] = "paid"',
        '        # ... then mark it paid\n        order["status"] = "paid"')
    result = execute("write_file", {"path": "shop/store.py", "content": content}, project)
    assert result.startswith("Wrote shop/store.py"), result


def test_marker_that_drops_nothing_is_allowed(project: Path) -> None:
    """A partial edit that only says the rest is unchanged: the rest is still in the file."""
    result = execute("edit_file", {
        "path": "shop/store.py",
        "old_text": "    def pay(self, order_id):\n        order = self.orders[order_id]\n",
        "new_text": "    def pay(self, order_id):\n        order = self.orders[str(order_id)]\n"
                    "        # ... rest of the method unchanged\n"}, project)
    assert result.startswith("Edited shop/store.py"), result


def test_rewrite_with_a_trailing_note_and_no_lost_code_is_allowed(project: Path) -> None:
    content = STORE.replace('raise KeyError(order_id)', 'raise ValueError(order_id)') \
        + "\n# The rest of the file remains unchanged\n"
    result = execute("write_file", {"path": "shop/store.py", "content": content}, project)
    assert result.startswith("Wrote shop/store.py"), result


def test_switch_off_saves_it(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(features.ENV, "elision_guard")
    content = '"""Orders."""\n\n# ... rest of the code unchanged\n'
    result = execute("write_file", {"path": "shop/store.py", "content": content}, project)
    assert result.startswith("Wrote shop/store.py")


def test_pipeline_writes_are_not_checked(project: Path) -> None:
    """atomic=False: plan_write leaves it to the pipelines, which call elision.elided in
    write_loop.save (the pipeline tests below)."""
    content = '"""Orders."""\n\n# ... rest of the code unchanged\n'
    plan = plan_write("write_file", {"path": "shop/store.py", "content": content}, project,
                      atomic=False)
    assert not isinstance(plan, str)
    assert elision.elided("shop/store.py", STORE, content) is not None


def test_not_python_is_not_checked() -> None:
    assert elision.elided("notes.md", "# Notes\n\nlots\n", "# Notes\n\n... rest unchanged\n") \
        is None


# -- the pipelines (write_loop.save): a refused write goes back to the writer like any refusal ---

needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")


def scripted(root: Path, replies: list[str]) -> tuple[CodingAgent, list[dict]]:
    agent = CodingAgent(root, allow_commands=False)
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content=replies.pop(0), tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    agent.on_event = lambda kind, data: None
    return agent, calls


SOLUTION = '''def solution(lst):
    """Return the sum of the odd elements that are in even positions.

    Examples
    solution([5, 8, 7, 1]) ==> 12
    """
    total = 0
    for i, x in enumerate(lst):
        if i % 2 == 1 and x % 2 == 1:
            total += x
    return total
'''


@needs_sandbox
def test_direct_pipeline_refuses_an_elided_write_and_asks_for_the_full_file(tmp_path: Path) -> None:
    (tmp_path / "solution.py").write_text(SOLUTION)
    elided = ("```python\ndef solution(lst):\n    total = 0\n"
              "    # ... rest of the function unchanged\n    return total\n```")
    right = ("```python\ndef solution(lst):\n"
             "    return sum(x for i, x in enumerate(lst) if i % 2 == 0 and x % 2 == 1)\n```")
    agent, calls = scripted(tmp_path, [elided, right])
    task = "Fix solution() in solution.py so it does what its docstring says."
    answer = pipeline.run(agent, task, route=routing.route_task(task, tmp_path))
    assert answer is not None and answer.startswith("Implemented the code in solution.py; "), answer
    feedback = calls[1]["messages"][-1]["content"]
    assert "That code was not saved: Write not applied: line" in feedback
    assert "in `solution` stands in for code" in feedback
    assert "# ... rest of the function unchanged" in feedback
    assert "Reply with the complete file" in feedback
    assert "rest of the function unchanged" not in (tmp_path / "solution.py").read_text()


def make_shop(root: Path) -> None:
    (root / "shop").mkdir()
    (root / "shop" / "__init__.py").write_text("")
    (root / "shop" / "cart.py").write_text(
        "def count(items):\n    n = 0\n    for _ in items:\n        n += 1\n    return n\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_cart.py").write_text(
        "from shop.cart import count, total\n\n\ndef test_count():\n"
        "    assert count([(1, 2)]) == 1\n\n\ndef test_total():\n"
        "    assert total([(2.5, 2), (1, 3)]) == 8\n")


CART_PLAN = json.dumps({"analysis": "x", "files": [{"path": "shop/cart.py",
                                                    "change": "add total(items)"}]})


@needs_sandbox
def test_project_pipeline_refuses_an_elided_write_and_asks_for_the_full_file(tmp_path: Path
                                                                             ) -> None:
    make_shop(tmp_path)
    elided = ("File: shop/cart.py\n```python\ndef count(items):\n    n = 0\n"
              "    # ... rest of the function unchanged\n\n\ndef total(items):\n"
              "    return sum(price * qty for price, qty in items)\n```")
    right = ("File: shop/cart.py\n```python\ndef total(items):\n"
             "    return sum(price * qty for price, qty in items)\n```")
    agent, calls = scripted(tmp_path, [CART_PLAN, elided, CART_PLAN, right])
    answer = agent.run("Add total(items) to the shop: the sum of price times quantity.")
    assert answer.startswith("Changed shop/cart.py; the project's tests pass."), answer
    writer = calls[3]["messages"][-1]["content"]
    assert "Your last reply for shop/cart.py was not saved:\nWrite not applied: line" in writer
    assert "in `count` stands in for code" in writer and "Reply with the complete file" in writer
    cart = (tmp_path / "shop" / "cart.py").read_text()
    assert "for _ in items:" in cart and "rest of the function unchanged" not in cart
