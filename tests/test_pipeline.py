"""The direct pipeline: routing, example extraction, merging, and write → check → repair."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from switches import switch_off

from vega import codemerge, features, pipeline, routing, sandbox
from vega.agent import CodingAgent

STUB = '''def solution(lst):
    """Return the sum of the odd elements that are in even positions.

    Examples
    solution([5, 8, 7, 1]) ==> 12
    solution([3, 3, 3, 3, 3]) ==> 9
    """
'''
WRONG = "```python\ndef solution(lst):\n    return sum(x for i, x in enumerate(lst) if i % 2 == 1 and x % 2 == 1)\n```"
RIGHT = "```python\ndef solution(lst):\n    return sum(x for i, x in enumerate(lst) if i % 2 == 0 and x % 2 == 1)\n```"


def test_examples_from_arrows_doctests_inputs_and_asserts() -> None:
    source = STUB + '''

def maximum(arr, k):
    """Example 1:
        Input: arr = [-3, -4, 5], k = 3
        Output: [-4, -3, 5]
    >>> maximum([1], 1)
    [1]
    """
'''
    calls = [(e.call, e.expected) for e in pipeline.examples_from_source(source)]
    assert ("solution([5, 8, 7, 1])", "12") in calls
    assert ("maximum(arr=[-3, -4, 5], k=3)", "[-4, -3, 5]") in calls
    assert ("maximum([1], 1)", "[1]") in calls
    module_doc = '"""\nWrite a function.\nassert f(99)==101\n"""\n'
    assert [(e.call, e.expected) for e in pipeline.examples_from_source(module_doc)] == [("f(99)", "101")]


def test_routing(tmp_path: Path) -> None:
    (tmp_path / "solution.py").write_text(STUB)

    def route(task: str) -> tuple[str, list[str]]:
        found = routing.route_task(task, tmp_path)
        return found.name, found.targets

    assert route("Implement solution() in solution.py so it does what its docstring says.") == (
        "direct", ["solution.py"])
    assert route("What does solution() return for [1]?")[0] != "direct"


def test_merge_keeps_other_definitions_and_adds_imports() -> None:
    original = "import os\n\n\ndef a():\n    return 1\n\n\ndef b():\n    pass\n"
    merged = pipeline.merge_code(original, "import math\n\n\ndef b():\n    return math.pi\n")
    assert "def a():\n    return 1" in merged and "return math.pi" in merged
    assert merged.startswith("import math\n") and merged.count("def b") == 1


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_wrong_code_is_repaired_from_the_failing_example(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "solution.py").write_text(STUB)
    replies = [WRONG, RIGHT]
    seen: list[list[dict]] = []
    agent = CodingAgent(tmp_path, allow_commands=False)

    def create(**kwargs):
        seen.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=replies.pop(0)))])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer = agent.run("Implement solution() in solution.py so it does exactly what its docstring says.")
    # acceptance (on by default) adds its task check to the count
    assert "3 of 3 checks pass (2 docstring examples, 1 task check)" in answer
    feedback = seen[1][-1]["content"]
    assert "solution([5, 8, 7, 1]) returned 1, expected 12" in feedback
    assert "i % 2 == 0" in (tmp_path / "solution.py").read_text()
    assert "response_format" not in str(seen)  # free text, not constrained JSON


def test_pipeline_is_on_by_default_and_can_be_switched_off(monkeypatch) -> None:
    assert features.on("direct_pipeline")
    monkeypatch.setenv(features.ENV, "direct_pipeline")
    assert not features.on("direct_pipeline")


def test_merge_puts_imports_after_docstring_and_future_imports() -> None:
    original = ('"""Aggregation."""\nfrom __future__ import annotations\n\nimport csv\n\n\n'
                'def by_region(rows):\n    return {}\n')
    generated = "import json\nfrom decimal import Decimal\n\n\ndef monthly(rows):\n    return json.dumps(rows)\n"
    merged = pipeline.merge_code(original, generated)
    compile(merged, "aggregate.py", "exec")
    lines = merged.splitlines()
    assert lines[0] == '"""Aggregation."""' and lines[1] == "from __future__ import annotations"
    assert "import json" in lines[2:4] and "def by_region(rows):" in merged and "def monthly(rows):" in merged


def test_merge_keeps_the_users_docstrings_the_model_left_out() -> None:
    original = (
        '"""Calculator."""\n\n\ndef add(a, b):\n    """Return a + b.\n\n    >>> add(2, 3)\n    5\n'
        '    """\n\n\nclass Stack:\n    """A stack."""\n\n    def push(self, x):\n'
        '        """Push x."""\n        pass\n'
    )
    generated = (
        "def add(a, b):\n    return a + b\n\n\nclass Stack:\n    def __init__(self):\n"
        "        self.items = []\n\n    def push(self, x):\n        self.items.append(x)\n"
    )
    merged = pipeline.merge_code(original, generated)
    tree = ast.parse(merged)
    assert ast.get_docstring(tree) == "Calculator."
    defs = {getattr(node, "name", ""): node for node in tree.body}
    add = defs["add"]
    assert ast.get_docstring(add) == "Return a + b.\n\n>>> add(2, 3)\n5"
    assert ast.unparse(add.body[-1]) == "return a + b"
    stack = defs["Stack"]
    assert ast.get_docstring(stack) == "A stack."
    push = next(n for n in stack.body if getattr(n, "name", "") == "push")
    assert ast.get_docstring(push) == "Push x."
    assert "self.items.append(x)" in merged


def test_merge_keeps_a_docstring_the_model_wrote() -> None:
    original = 'def add(a, b):\n    """Return a + b."""\n'
    merged = pipeline.merge_code(original, 'def add(a, b):\n    """Sum two numbers."""\n    return a + b\n')
    assert '"""Sum two numbers."""' in merged and "Return a + b" not in merged


def test_keep_docstrings_can_be_switched_off(monkeypatch) -> None:
    monkeypatch.setenv(features.ENV, "keep_docstrings")
    original = 'def add(a, b):\n    """Return a + b."""\n'
    assert pipeline.merge_code(original, "def add(a, b):\n    return a + b\n") == \
        "def add(a, b):\n    return a + b\n"


def test_a_reply_cut_off_at_the_token_limit_is_continued(tmp_path: Path) -> None:
    agent = CodingAgent(tmp_path, allow_commands=False)
    parts = iter([("Here:\n```python\ndef f():\n    return [1,", "length"),
                  ("```python\n 2]\n```", "stop")])
    calls = []

    def create(**kwargs):
        calls.append(kwargs["messages"])
        text, finish = next(parts)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text), finish_reason=finish)])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    text = pipeline.ask_complete(agent, [{"role": "user", "content": "write f"}])
    assert text == "Here:\n```python\ndef f():\n    return [1, 2]\n```"  # no second fence
    assert calls[1][-2] == {"role": "assistant", "content": "Here:\n```python\ndef f():\n    return [1,"}
    assert calls[1][-1]["content"] == pipeline.CONTINUE_PROMPT


def test_a_task_naming_other_files_is_not_taken(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text('__version__ = "1.0.0"\n')
    task = ("Write the 2.0.0 release: bump the version in `pkg/__init__.py` and `pyproject.toml`, "
            "and add a section to `CHANGELOG.md`.")
    release = routing.route_task(task, tmp_path)
    assert release.name != "direct" and "CHANGELOG.md" in release.other_targets
    assert routing.route_task("Implement add() in pkg/__init__.py.", tmp_path).name == "direct"
    # the same file named by its short name is not "another" file
    short = routing.route_task("Write add in __init__.py", tmp_path)
    assert (short.name, short.targets, short.other_targets) == ("direct", ["pkg/__init__.py"], [])


def test_thinking_is_dropped_before_reading_code() -> None:
    text = "<think>try ```python\nx = 1\n``` maybe</think>\n```python\nx = 2\n```"
    assert pipeline.strip_thinking(text) == "```python\nx = 2\n```"
    assert pipeline.strip_thinking("<think>cut off mid-thought") == ""
    assert pipeline.strip_thinking("```python\nx = 3\n```") == "```python\nx = 3\n```"
    # the template opened the block, so the reply carries only the closing tag
    assert pipeline.strip_thinking("draft ```python\nx = 1\n```</think>\n```python\nx = 4\n```") == \
        "```python\nx = 4\n```"


# -- method-level class merging (large_projects) ---------------------------------------------------

STORE = '''"""Stock."""


class Store:
    """A shop's stock."""

    LIMIT = 10
    name: str = "shop"

    def __init__(self):
        self.items = {}

    @staticmethod
    def unit():
        return "pcs"

    def add(self, item, qty):
        self.items[item] = self.items.get(item, 0) + qty

    @property
    def size(self):
        return len(self.items)

    @size.setter
    def size(self, value):
        raise AttributeError("read only")

    def remove(self, item):
        del self.items[item]


def helper():
    return 1
'''


@pytest.fixture
def methods_on(monkeypatch):
    monkeypatch.setenv(features.ENV_ON, "large_projects")


def _class_source(source: str, name: str = "Store") -> str:
    tree = ast.parse(source)
    node = next(n for n in tree.body if getattr(n, "name", "") == name)
    return ast.get_source_segment(source, node) or ""


def test_partial_class_replaces_one_method_and_appends_a_new_one(methods_on) -> None:
    generated = ("class Store:\n    def add(self, item, qty):\n        if qty <= 0:\n"
                 "            raise ValueError(qty)\n        self.items[item] = self.items.get(item, 0) + qty\n\n"
                 "    def count(self, item):\n        return self.items.get(item, 0)\n")
    merged = pipeline.merge_code(STORE, generated)
    expected = STORE.replace(
        "    def add(self, item, qty):\n",
        "    def add(self, item, qty):\n        if qty <= 0:\n            raise ValueError(qty)\n",
    ).replace(
        "        del self.items[item]\n",
        "        del self.items[item]\n\n    def count(self, item):\n        return self.items.get(item, 0)\n",
    )
    assert merged == expected  # everything else byte for byte: docstring, attributes, other methods


def test_partial_class_decorators_follow_the_generated_method(methods_on) -> None:
    generated = "class Store:\n    @classmethod\n    def unit(cls):\n        return cls.LIMIT\n"
    merged = pipeline.merge_code(STORE, generated)
    assert "@staticmethod" not in merged
    assert "    @classmethod\n    def unit(cls):\n        return cls.LIMIT\n\n    def add" in merged
    assert "    @property\n    def size(self):" in merged and "def remove" in merged


def test_property_setter_replaced_alone_and_getter_alone(methods_on) -> None:
    setter = "class Store:\n    @size.setter\n    def size(self, value):\n        self.items = dict.fromkeys(range(value), 0)\n"
    merged = pipeline.merge_code(STORE, setter)
    assert "    @property\n    def size(self):\n        return len(self.items)\n" in merged  # getter kept
    assert "dict.fromkeys(range(value), 0)" in merged and "read only" not in merged
    getter = "class Store:\n    @property\n    def size(self):\n        return sum(self.items.values())\n"
    merged = pipeline.merge_code(STORE, getter)
    assert "return sum(self.items.values())" in merged and 'raise AttributeError("read only")' in merged
    assert merged.count("def size") == 2


def test_a_new_property_role_goes_right_after_its_group(methods_on) -> None:
    generated = "class Store:\n    @size.deleter\n    def size(self):\n        self.items.clear()\n"
    merged = pipeline.merge_code(STORE, generated)
    assert ('        raise AttributeError("read only")\n\n    @size.deleter\n    def size(self):\n'
            "        self.items.clear()\n\n    def remove") in merged


def test_overloads_are_replaced_as_a_group(methods_on) -> None:
    original = ("from typing import overload\n\n\nclass P:\n    @overload\n    def get(self, k: int) -> int: ...\n"
                "    @overload\n    def get(self, k: str) -> str: ...\n    def get(self, k):\n        return k\n\n"
                "    def other(self):\n        return 0\n")
    generated = ("class P:\n    @overload\n    def get(self, k: bytes) -> bytes: ...\n"
                 "    def get(self, k):\n        return bytes(k)\n")
    merged = pipeline.merge_code(original, generated)
    compile(merged, "p.py", "exec")
    assert merged.count("def get") == 2 and "k: int" not in merged and "k: str" not in merged
    assert "return bytes(k)" in merged and "def other(self):\n        return 0" in merged


def test_nested_definitions_keep_exact_indentation(methods_on) -> None:
    generated = ("class Store:\n    def remove(self, item):\n        def gone(key):\n"
                 "            return key not in self.items\n\n        class Missing(KeyError):\n"
                 "            pass\n\n        if gone(item):\n            raise Missing(item)\n"
                 "        del self.items[item]\n")
    merged = pipeline.merge_code(STORE, generated)
    assert _class_source(merged).endswith(generated.split("\n", 1)[1].rstrip("\n"))
    # a reply whose methods sit at two spaces is shifted to the class's four; the body keeps its
    # own relative indentation (shifted, never re-scaled: that could change multi-line strings)
    two = "class Store:\n  def remove(self, item):\n    if item in self.items:\n      del self.items[item]\n"
    merged = pipeline.merge_code(STORE, two)
    compile(merged, "store.py", "exec")
    assert ("\n\n    def remove(self, item):\n      if item in self.items:\n"
            "        del self.items[item]\n\n\ndef helper") in merged


def test_docstring_and_attributes_change_only_when_the_reply_has_them(methods_on) -> None:
    generated = ('class Store:\n    """Stock, by item."""\n\n    LIMIT = 20\n    MAX = 99\n\n'
                 "    def remove(self, item):\n        self.items.pop(item, None)\n")
    merged = pipeline.merge_code(STORE, generated)
    store = _class_source(merged)
    assert '"""Stock, by item."""' in store and '"""A shop\'s stock."""' not in store
    assert "    LIMIT = 20\n    name: str = \"shop\"\n    MAX = 99\n" in store
    assert "def add(self, item, qty)" in store and "self.items.pop(item, None)" in store
    untouched = pipeline.merge_code(STORE, "class Store:\n    def remove(self, item):\n        pass\n")
    assert '"""A shop\'s stock."""\n\n    LIMIT = 10\n' in untouched


def test_a_class_reply_with_every_method_still_replaces_the_class(methods_on) -> None:
    whole = ("class Store:\n    def __init__(self):\n        self.items = []\n\n    def unit(self):\n"
             "        return 1\n\n    def add(self, item, qty):\n        pass\n\n    def size(self):\n"
             "        return 0\n\n    def remove(self, item):\n        pass\n")
    merged = pipeline.merge_code(STORE, whole)
    assert "LIMIT = 10" not in merged and "@property" not in merged and "def helper()" in merged


def test_partial_class_only_with_the_switch_on() -> None:
    generated = "class Store:\n    def remove(self, item):\n        self.items.pop(item, None)\n"
    merged = pipeline.merge_code(STORE, generated)
    # the old rule: the class is replaced whole (keep_docstrings puts its docstring back)
    assert merged == ('"""Stock."""\n\n\nclass Store:\n    """A shop\'s stock."""\n\n'
                      "    def remove(self, item):\n        self.items.pop(item, None)\n\n\n"
                      "def helper():\n    return 1\n")


def test_a_partial_merge_that_does_not_parse_falls_back_to_whole_class(methods_on, monkeypatch) -> None:
    monkeypatch.setattr(codemerge, "_merge_class", lambda *args: ["class Store:", "    def ("])
    generated = "class Store:\n    def remove(self, item):\n        self.items.pop(item, None)\n"
    merged = pipeline.merge_code(STORE, generated)
    compile(merged, "store.py", "exec")
    assert "def add" not in merged and "self.items.pop(item, None)" in merged


def test_a_file_that_is_only_the_class_is_merged_not_replaced(methods_on) -> None:
    source = "class A:\n    def f(self):\n        return 1\n\n    def g(self):\n        return 2\n"
    merged = pipeline.merge_code(source, "class A:\n    def g(self):\n        return 3\n")
    assert merged == "class A:\n    def f(self):\n        return 1\n\n    def g(self):\n        return 3\n"


def test_failing_lines_quotes_the_first_three() -> None:
    quoted = pipeline.failing_lines([
        "import polyform.io: ModuleNotFoundError: No module named 'fastyaml'",
        "load_shapes(...) raised  NameError:\n name 'x' is not defined", "third", "fourth", "third"])
    assert quoted.startswith("\nStill failing:\n- import polyform.io: ModuleNotFoundError: No "
                             "module named 'fastyaml'\n")
    assert "- load_shapes(...) raised NameError: name 'x' is not defined" in quoted
    assert "- third" in quoted and "- fourth" not in quoted and "1 more" in quoted
    assert pipeline.failing_lines([]) == ""


def test_give_up_answer_quotes_the_failing_check(tmp_path: Path, monkeypatch) -> None:
    """The answer used to say only "1 check(s) still fail" and hide "No module named
    'fastyaml'"."""
    (tmp_path / "solution.py").write_text(STUB)
    failure = "import solution: ModuleNotFoundError: No module named 'fastyaml'"
    monkeypatch.setattr(pipeline, "run_checks",
                        lambda *a, **k: pipeline.CheckResult(total=1, failures=[failure]))
    agent = CodingAgent(tmp_path, allow_commands=False)
    replies = [RIGHT] * (pipeline.REPAIRS + 1)

    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=replies.pop(0)), finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    task = "Implement solution() in solution.py so it does exactly what its docstring says."
    answer = agent.run(task)
    # acceptance (on by default) adds its task check, which passes here
    assert "only 1 of 2 checks pass" in answer and "No module named 'fastyaml'" in answer
    switch_off(monkeypatch, "failing_checks_quoted")
    replies += [RIGHT] * (pipeline.REPAIRS + 1)
    off = CodingAgent(tmp_path, allow_commands=False)
    off.client = agent.client
    assert "fastyaml" not in off.run(task)


# -- the honest answer (systems review #3) ------------------------------------------------------------

def test_answer_counts_the_checks_by_kind() -> None:
    kinds = {"docstring examples": 2, "tests": 1, "task checks": 3}
    assert pipeline.honest_answer("calc.py", 0, kinds, changed=True) == (
        "Implemented the code in calc.py; 6 of 6 checks pass (2 docstring examples, the tests, "
        "3 task checks).")
    assert pipeline.honest_answer("calc.py", 2, kinds, changed=True).startswith(
        "Wrote calc.py, but only 4 of 6 checks pass (2 docstring examples, the tests, 3 task "
        "checks) after")
    one = {"docstring examples": 0, "tests": 0, "task checks": 1}
    assert "1 of 1 checks pass (1 task check)" in pipeline.honest_answer("a.py", 0, one,
                                                                          changed=True)


def test_answer_says_nothing_was_verified_when_no_check_ran() -> None:
    none = {"docstring examples": 0, "tests": 0, "task checks": 0}
    answer = pipeline.honest_answer("calc.py", 0, none, changed=True)
    assert answer == ("Wrote calc.py, but nothing was verified: there were no docstring examples, "
                      "tests or task checks to run.")
    assert "pass" not in answer
    unchanged = pipeline.honest_answer("calc.py", 0, none, changed=False)
    assert unchanged.startswith("No file was changed") and "nothing was verified" in unchanged
    assert pipeline.honest_answer("calc.py", 0, {"tests": 1}, changed=False).endswith(
        "same as before; 1 of 1 checks pass (the tests).")


def test_direct_pipeline_with_zero_checks_does_not_claim_they_pass(tmp_path: Path,
                                                                    monkeypatch) -> None:
    """12 of 115 direct answers said "its docstring examples and tests pass" with 0 checks run."""
    monkeypatch.setenv(features.ENV, "acceptance,question_pipeline,clarify_first,project_context")
    (tmp_path / "solution.py").write_text("def solution(lst):\n    raise NotImplementedError\n")
    monkeypatch.setattr(pipeline, "run_checks", lambda *a, **k: pipeline.CheckResult(total=0))
    agent = CodingAgent(tmp_path, allow_commands=False)
    replies = [RIGHT]

    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=replies.pop(0)), finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer = agent.run("Implement solution() in solution.py: sum the odd elements in even "
                       "positions.")
    assert answer.startswith("Wrote solution.py, but nothing was verified")
    assert "checks pass" not in answer
