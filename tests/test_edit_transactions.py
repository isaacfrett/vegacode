"""Redesign B: atomic writes, edits by name, fence stripping and the pinned editing target.

The broken edits are the Granite baseline's own (failure diagnosis, component "edit tools");
the agent-loop tests use a scripted fake model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import agent as agent_module
from vega import context, edits, features
from vega.agent import CodingAgent, system_prompt
from vega.context import SUMMARY_PREFIX, WorkLog, compact
from vega.tools import execute, tool_specs

SWITCHES = ("atomic_writes", "edit_by_name", "fence_strip", "pin_editing_target")


@pytest.fixture(autouse=True)
def switches_on(monkeypatch: pytest.MonkeyPatch) -> None:
    off = [name for name in os.environ.get(features.ENV, "").split(",") if name and name not in SWITCHES]
    # the scripted agent-loop tests must reach the loop (short write tasks go direct otherwise)
    monkeypatch.setenv(features.ENV, ",".join([*off, "direct_pipeline"]))


def switches_off(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    current = [n for n in os.environ.get(features.ENV, "").split(",") if n]
    monkeypatch.setenv(features.ENV, ",".join(current + list(names or SWITCHES)))


def write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


STUB = 'def plot_monthly(path):\n    """Plot monthly revenue."""\n    raise NotImplementedError\n'


# -- atomic_writes --------------------------------------------------------------------------------

def test_unclosed_paren_is_refused_with_line_column_and_lines(tmp_path: Path) -> None:
    # this edit used to be saved with only a WARNING, and the run lost many steps to it
    path = write(tmp_path, "sales_charts/monthly.py", "import pandas as pd\n\n\n" + STUB)
    result = execute("edit_file", {
        "path": "sales_charts/monthly.py", "old_text": "def plot_monthly(path):",
        "new_text": "def plot_monthly(path:"}, tmp_path)
    assert result.startswith("Write not applied: sales_charts/monthly.py would not parse "
                             "(SyntaxError at line 4, column 17: '(' was never closed). "
                             "The file is unchanged.")
    assert path.read_text() == "import pandas as pd\n\n\n" + STUB
    assert "  3\t\n> 4\tdef plot_monthly(path:\n  5\t    \"\"\"Plot monthly revenue.\"\"\"" in result
    assert "same old_text (it is still in the file)" in result
    assert "replace_definition with name='plot_monthly'" in result


def test_replace_lines_and_write_file_refusals_say_how_to_retry(tmp_path: Path) -> None:
    # a real failure: replace_lines 6-10 overwrote the ORDERS literal
    original = 'ORDERS = {\n    "a1": {"id": "a1"},\n    "a2": {"id": "a2"},\n}\n\n\ndef get(i):\n    return ORDERS[i]\n'
    path = write(tmp_path, "api.py", original)
    result = execute("replace_lines", {"path": "api.py", "start_line": 2, "end_line": 4,
                                       "new_text": '    "total": 1,'}, tmp_path)
    assert "'{' was never closed" in result and "line 1, column 10" in result
    assert "Send replace_lines again for lines 2-4" in result
    assert path.read_text() == original
    result = execute("write_file", {"path": "new.py", "content": "def f(:\n    pass\n"}, tmp_path)
    assert result.startswith("Write not applied: new.py would not parse") and "line 1" in result
    assert not (tmp_path / "new.py").exists()


def test_repair_on_retry_succeeds_in_the_agent_loop(tmp_path: Path) -> None:
    path = write(tmp_path, "monthly.py", STUB)
    agent, seen = agent_with(tmp_path, [
        call("edit_file", path="monthly.py", old_text="    raise NotImplementedError",
             new_text="    return (1 + 2"),
        call("edit_file", path="monthly.py", old_text="    raise NotImplementedError",
             new_text="    return 1 + 2"),
        text("Implemented plot_monthly."),
    ])
    assert agent.run("Implement plot_monthly in monthly.py") == "Implemented plot_monthly."
    first, second = (m["content"] for m in seen[-1] if m["role"] == "tool")
    assert first.startswith("Write not applied: monthly.py would not parse")
    assert second.startswith("Edited monthly.py (exact match).") and "Syntax check: OK" in second
    assert path.read_text().endswith("    return 1 + 2\n")
    # the refused call did not count as a broken file: no parse nudge followed
    assert not any("no longer parses" in str(m.get("content")) for m in seen[-1])


def test_pre_broken_file_accepts_writes_that_add_no_new_error(tmp_path: Path) -> None:
    broken = "def a(:\n    pass\n\n\ndef b():\n    return 1\n"
    path = write(tmp_path, "m.py", broken)
    ok = execute("edit_file", {"path": "m.py", "old_text": "return 1", "new_text": "return 2"},
                 tmp_path)
    assert ok.startswith("Edited m.py") and "did not parse before this write either" in ok
    assert "Still: SyntaxError at line 1" in ok and "WARNING" in ok
    worse = execute("edit_file", {"path": "m.py", "old_text": "return 2", "new_text": "return (2"},
                    tmp_path)
    assert worse.startswith("Write not applied: m.py already did not parse, and this write adds")
    assert path.read_text() == broken.replace("return 1", "return 2")
    fixed = execute("edit_file", {"path": "m.py", "old_text": "def a(:", "new_text": "def a():"},
                    tmp_path)
    assert fixed.startswith("Edited m.py") and "Syntax check: OK" in fixed


def test_new_undefined_names_and_redefinitions_are_refused_only_when_new(tmp_path: Path) -> None:
    # the loop variable `first` was edited away; cascade-breaker: `self`
    source = "def groups(items):\n    for first in items:\n        print(first)\n    return 1\n"
    path = write(tmp_path, "dupes.py", source)
    result = execute("edit_file", {"path": "dupes.py", "old_text": "    for first in items:\n        print(first)",
                                   "new_text": "    print(first)"}, tmp_path)
    assert result.startswith("Write not applied: it would add undefined name(s) first to dupes.py")
    assert "line 2, column 11: F821 Undefined name `first`" in result
    assert path.read_text() == source
    # a name that was already undefined before the edit is not new
    write(tmp_path, "old.py", "def f():\n    return missing\n")
    kept = execute("edit_file", {"path": "old.py", "old_text": "return missing",
                                 "new_text": "return missing + 1"}, tmp_path)
    assert kept.startswith("Edited old.py")
    # a second copy of a function (F811)
    write(tmp_path, "two.py", "def f():\n    return 1\n")
    twice = execute("write_file", {"path": "two.py",
                                   "content": "def f():\n    return 1\n\n\ndef f():\n    return 2\n"},
                    tmp_path)
    assert "a second definition of f" in twice and "change the existing definition" in twice


def test_json_and_toml_are_refused(tmp_path: Path) -> None:
    path = write(tmp_path, "c.json", '{"a": 1}\n')
    result = execute("edit_file", {"path": "c.json", "old_text": '"a": 1', "new_text": '"a": 1,'},
                     tmp_path)
    # the column json reports for a trailing comma differs by Python version (9 on 3.12, 8 on 3.13)
    assert result.startswith("Write not applied: c.json would not parse (invalid JSON at line 1, "
                             "column ")
    assert path.read_text() == '{"a": 1}\n'
    path = write(tmp_path, "p.toml", '[tool]\nname = "x"\n')
    result = execute("replace_lines", {"path": "p.toml", "start_line": 2, "end_line": 2,
                                       "new_text": "name = x y"}, tmp_path)
    assert result.startswith("Write not applied: p.toml would not parse (invalid TOML at line 2")
    assert path.read_text() == '[tool]\nname = "x"\n'
    # other languages are not checked
    assert execute("write_file", {"path": "a.js", "content": "function ("}, tmp_path).startswith("Wrote")


# -- fence_strip ----------------------------------------------------------------------------------

def test_fenced_write_is_cleaned_with_a_note(tmp_path: Path) -> None:
    # a model that wraps the whole file in a fence
    result = execute("write_file", {"path": "shop/orders.py",
                                    "content": "```python\ndef place_order():\n    return 1\n```\n"}, tmp_path)
    assert result.startswith("Wrote shop/orders.py") and "removed the ``` fence" in result
    assert (tmp_path / "shop/orders.py").read_text() == "def place_order():\n    return 1\n"
    named = execute("write_file", {"path": "a.py", "content": "a.py\nx = 1\n"}, tmp_path)
    assert "removed a first line that only names the file ('a.py')" in named
    assert (tmp_path / "a.py").read_text() == "x = 1\n"
    # a comment naming the file is valid Python and stays; Markdown keeps its fences
    execute("write_file", {"path": "b.py", "content": "# b.py\nx = 1\n"}, tmp_path)
    assert (tmp_path / "b.py").read_text() == "# b.py\nx = 1\n"
    execute("write_file", {"path": "README.md", "content": "```python\nx = 1\n```\n"}, tmp_path)
    assert (tmp_path / "README.md").read_text() == "```python\nx = 1\n```\n"


# -- edit_by_name ---------------------------------------------------------------------------------

CLASSES = '''"""Shapes."""
import math


def helper(x):
    return x


class Rect:
    """A rectangle."""

    def __init__(self, w, h):
        self.w = w
        self.h = h

    @property
    def area(self):
        """The area."""
        return 0

    class Meta:
        def describe(self):
            return "meta"
'''


def test_replace_function_by_name_keeps_docstring(tmp_path: Path) -> None:
    path = write(tmp_path, "sales_charts/monthly.py", "import os\n\n\n" + STUB)
    result = execute("replace_definition", {
        "path": "sales_charts/monthly.py", "name": "plot_monthly",
        "code": "def plot_monthly(path):\n    return os.path.basename(path)\n"}, tmp_path)
    assert result.startswith("Replaced definition plot_monthly in sales_charts/monthly.py "
                             "(was lines 4-6, now lines 4-6); kept its docstring.")
    assert path.read_text() == ('import os\n\n\ndef plot_monthly(path):\n    """Plot monthly '
                                'revenue."""\n    return os.path.basename(path)\n')


def test_replace_method_with_fence_and_wrong_indentation_keeps_decorator(tmp_path: Path) -> None:
    path = write(tmp_path, "shapes.py", CLASSES)
    result = execute("replace_definition", {
        "path": "shapes.py", "name": "Rect.area",
        "code": "```python\n        def area(self):\n            return self.w * self.h\n```"}, tmp_path)
    assert "removed the ``` fence" in result and "kept its decorators" in result
    assert "kept its docstring" in result
    text = path.read_text()
    assert ('    @property\n    def area(self):\n        """The area."""\n'
            '        return self.w * self.h\n') in text
    compile(text, "shapes.py", "exec")


def test_method_sent_inside_its_class_replaces_only_that_method(tmp_path: Path) -> None:
    path = write(tmp_path, "shapes.py", CLASSES)
    result = execute("replace_definition", {
        "path": "shapes.py", "name": "Rect.__init__",
        "code": "class Rect:\n    def __init__(self, w, h):\n        self.w, self.h = w, h\n"}, tmp_path)
    assert "took __init__ out of your class Rect" in result
    text = path.read_text()
    assert "        self.w, self.h = w, h\n" in text and "def area(self):" in text
    assert "class Meta:" in text and text.count("class Rect") == 1


def test_replace_nested_class_method_by_dotted_module(tmp_path: Path) -> None:
    path = write(tmp_path, "geo/shapes.py", CLASSES)
    result = execute("replace_definition", {
        "path": "geo.shapes", "name": "Rect.Meta.describe",
        "code": "def describe(self):\n  return 'm2'\n"}, tmp_path)
    assert result.startswith("Replaced definition Rect.Meta.describe in geo/shapes.py")
    assert "        def describe(self):\n          return 'm2'\n" in path.read_text()


def test_decorated_function_and_new_decorator_and_missing_method(tmp_path: Path) -> None:
    source = "import functools\n\n\n@functools.cache\ndef fib(n):\n    return n\n\n\nclass A:\n    x = 1\n"
    path = write(tmp_path, "f.py", source)
    execute("replace_definition", {"path": "f.py", "name": "fib",
                                   "code": "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)"},
            tmp_path)
    assert "@functools.cache\ndef fib(n):\n    return n if n < 2" in path.read_text()
    execute("replace_definition", {"path": "f.py", "name": "fib",
                                   "code": "@functools.lru_cache(None)\ndef fib(n):\n    return 0"}, tmp_path)
    assert "@functools.lru_cache(None)\ndef fib(n):" in path.read_text()
    assert "@functools.cache" not in path.read_text()
    added = execute("replace_definition", {"path": "f.py", "name": "A.double",
                                           "code": "def double(self):\n    return self.x * 2"}, tmp_path)
    assert added.startswith("Added definition A.double to f.py")
    assert "    x = 1\n\n    def double(self):\n        return self.x * 2\n" in path.read_text()
    missing = execute("replace_definition", {"path": "f.py", "name": "B.g", "code": "def g(self):\n    pass"},
                      tmp_path)
    assert missing.startswith("Edit not applied: f.py has no class B") and "fib" in missing


def test_definition_brings_imports_helpers_and_typevars(tmp_path: Path) -> None:
    # merge_code used to drop `K = TypeVar('K')`
    path = write(tmp_path, "cache.py", '"""Cache."""\n\n\nclass TTLCache:\n    pass\n')
    result = execute("replace_definition", {"path": "cache.py", "name": "TTLCache", "code": (
        "from typing import Generic, TypeVar\n\nK = TypeVar('K')\nV = TypeVar('V')\n\n\n"
        "class TTLCache(Generic[K, V]):\n    def get(self, key: K) -> V:\n        return _lookup(key)\n\n\n"
        "def _lookup(key):\n    return key\n")}, tmp_path)
    assert result.startswith("Replaced definition TTLCache") and "added _lookup too" in result
    text = path.read_text()
    assert "K = TypeVar('K')" in text and "from typing import Generic, TypeVar" in text
    assert text.index("K = TypeVar") < text.index("class TTLCache")
    compile(text, "cache.py", "exec")


def test_definition_that_does_not_parse_or_names_another_function_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, "m.py", STUB)
    bad = execute("replace_definition", {"path": "m.py", "name": "plot_monthly",
                                         "code": "def plot_monthly(path:\n    return 1"}, tmp_path)
    assert bad.startswith("Edit not applied: your code does not parse (SyntaxError at line 1")
    other = execute("replace_definition", {"path": "m.py", "name": "plot_monthly",
                                           "code": "def plot(path):\n    return 1"}, tmp_path)
    assert "your code defines plot, not plot_monthly" in other
    assert path.read_text() == STUB


def test_replace_definition_repairs_a_broken_file(tmp_path: Path) -> None:
    # the function header lost its closing paren
    broken = "import os\n\n\ndef plot_monthly(path:\n    return 1\n\n\ndef other():\n    return 2\n"
    path = write(tmp_path, "m.py", broken)
    result = execute("replace_definition", {"path": "m.py", "name": "plot_monthly",
                                            "code": "def plot_monthly(path):\n    return 1\n"}, tmp_path)
    assert result.startswith("Replaced definition plot_monthly") and "Syntax check: OK" in result
    assert path.read_text() == broken.replace("(path:", "(path):")


def test_tools_and_prompt_recommend_edit_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = {t["function"]["name"]: t["function"] for t in
             tool_specs(include_definition=features.on("edit_by_name"))}
    assert "replace_definition" in specs and "Class.method" in specs["replace_definition"]["description"]
    assert "use replace_definition instead" in specs["edit_file"]["description"]
    assert "use replace_definition" in system_prompt()
    switches_off(monkeypatch, "edit_by_name")
    assert "replace_definition" not in system_prompt()
    assert "Use edit_file for focused edits" in system_prompt()
    assert "replace_definition" not in [t["function"]["name"] for t in tool_specs()]


# -- pin_editing_target ---------------------------------------------------------------------------

def test_pinned_text_shows_the_current_definition_newest_first(tmp_path: Path) -> None:
    write(tmp_path, "shapes.py", CLASSES)
    pinned = edits.pinned_text([("replace_definition", {"path": "shapes.py", "name": "helper"}),
                                ("edit_file", {"path": "shapes.py", "old_text": "return 0",
                                               "new_text": "return 0"})], tmp_path)
    assert pinned.startswith(edits.PIN_TITLE)
    assert pinned.index("shapes.py, Rect.area (lines 16-19)") < pinned.index("shapes.py, helper")
    assert "19\t        return 0" in pinned
    big = "def big():\n" + "".join(f"    x{i} = {i} * 1000000\n" for i in range(300))
    write(tmp_path, "big.py", big)
    capped = edits.pinned_text([("replace_definition", {"path": "big.py", "name": "big"})], tmp_path)
    assert len(capped) <= edits.PIN_MAX_CHARS and "read_file big.py start_line=" in capped


def run_compacting_turn(tmp_path: Path) -> list[str]:
    write(tmp_path, "calc.py", "".join(f"v{i} = {i}\n" for i in range(200))
          + "\n\ndef rounding(x):\n    return round(x, 2)\n")
    replies = [
        (call("read_file", path="calc.py"), 1500),
        (call("edit_file", path="calc.py", old_text="return round(x, 2)",
              new_text="return round(x, 3)"), 9000),
        (text("Changed rounding."), 3200),
    ]
    agent, seen = agent_with(tmp_path, [m for m, _ in replies], usage=[u for _, u in replies],
                             num_ctx=10000)
    agent.run("Fix the rounding in calc.py")
    return [m["content"] for m in seen[-1] if str(m.get("content", "")).startswith(SUMMARY_PREFIX)]


def test_pin_survives_compaction(tmp_path: Path) -> None:
    summaries = run_compacting_turn(tmp_path)
    assert summaries and edits.PIN_TITLE in summaries[0]
    assert "calc.py, rounding (lines" in summaries[0] and "return round(x, 3)" in summaries[0]


def test_compaction_without_pin_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "goal"}]
    for i in range(30):
        messages += [{"role": "assistant", "content": f"note {i} " * 40}]
    plain = compact(messages, 1, 2000, WorkLog())
    pinned = compact(messages, 1, 2000, WorkLog(), pinned="PINNED")
    assert plain is not None and pinned is not None
    assert pinned.messages[1]["content"].endswith("\nPINNED")
    assert "PINNED" not in plain.messages[1]["content"]


# -- switches off restore the old behaviour -------------------------------------------------------

def test_switches_off_restore_the_old_behaviour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    switches_off(monkeypatch)
    write(tmp_path, "m.py", STUB)
    result = execute("edit_file", {"path": "m.py", "old_text": "def plot_monthly(path):",
                                   "new_text": "def plot_monthly(path:"}, tmp_path)
    assert result.startswith("Edited m.py (exact match).") and "WARNING: the file now has a SyntaxError" in result
    fenced = execute("write_file", {"path": "f.py", "content": "```python\nx = 1\n```\n"}, tmp_path)
    assert "WARNING" in fenced and (tmp_path / "f.py").read_text().startswith("```python")
    assert "replace_definition" not in [t["function"]["name"] for t in agent_module.tool_specs(
        include_definition=features.on("edit_by_name"))]
    write(tmp_path, "calc.py", "def rounding(x):\n    return x\n")
    assert edits.pinned_text([("edit_file", {"path": "calc.py", "old_text": "return x",
                                             "new_text": "return x"})], tmp_path)  # helper still works
    summaries = run_compacting_turn(tmp_path / "loop")
    assert summaries and edits.PIN_TITLE not in summaries[0]


def test_pipelines_write_without_the_atomic_check(tmp_path: Path) -> None:
    write(tmp_path, "m.py", STUB)
    result = execute("write_file", {"path": "m.py", "content": "def f(:\n"}, tmp_path, atomic=False)
    assert result.startswith("Wrote m.py") and "WARNING" in result


# -- helpers --------------------------------------------------------------------------------------

class FakeMessage(SimpleNamespace):
    def model_dump(self, exclude_none: bool = False) -> dict:
        data = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in self.tool_calls]
        return data


_calls = iter(range(10_000))


def text(content: str) -> FakeMessage:
    return FakeMessage(content=content, tool_calls=None)


def call(name: str, **args) -> FakeMessage:
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    return FakeMessage(content=None, tool_calls=[SimpleNamespace(id=f"c{next(_calls)}", function=fn)])


def agent_with(project: Path, replies: list[FakeMessage], usage: list[int] | None = None,
               **options) -> tuple[CodingAgent, list[list[dict]]]:
    agent = CodingAgent(project, allow_commands=False, planning="off", **options)
    seen: list[list[dict]] = []
    sizes = list(usage or [])

    def create(**kwargs):
        seen.append(json.loads(json.dumps(kwargs["messages"])))
        tokens = sizes.pop(0) if sizes else None
        return SimpleNamespace(choices=[SimpleNamespace(message=replies.pop(0))],
                               usage=SimpleNamespace(prompt_tokens=tokens) if tokens else None)

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, seen


def test_context_module_constant_is_shared() -> None:
    assert context.SUMMARY_PREFIX == SUMMARY_PREFIX
