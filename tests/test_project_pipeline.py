"""Project pipeline: plan → write → test → repair, with a scripted model and the real sandbox."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import features, project_pipeline, routing, sandbox
from vega.agent import CodingAgent

pytestmark = pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")


@pytest.fixture(autouse=True)
def pipeline_on(monkeypatch) -> None:
    monkeypatch.delenv("VEGA_FEATURES_OFF", raising=False)  # conftest turns it off for other tests

PLAN = json.dumps({"analysis": "Add total() to shop/cart.py and use it in shop/cli.py.",
                   "files": [{"path": "shop/cart.py", "change": "add total(items)"}]})
WRONG = "```python\ndef total(items):\n    return sum(price for price, qty in items)\n```"
RIGHT = "```python\ndef total(items):\n    return sum(price * qty for price, qty in items)\n```"


def make_project(root: Path) -> None:
    (root / "shop").mkdir()
    (root / "shop" / "__init__.py").write_text("")
    (root / "shop" / "cart.py").write_text("def count(items):\n    return len(items)\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_cart.py").write_text(
        "from shop.cart import count, total\n\n\ndef test_count():\n    assert count([(1, 2)]) == 1\n\n\n"
        "def test_total():\n    assert total([(2.5, 2), (1, 3)]) == 8\n")


def scripted(root: Path, replies: list[str]) -> tuple[CodingAgent, list[dict]]:
    agent = CodingAgent(root, allow_commands=False)
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=replies.pop(0)))])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, calls


def test_plan_write_test_repair(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "project_pipeline")
    make_project(tmp_path)
    agent, calls = scripted(tmp_path, [PLAN, WRONG, PLAN, RIGHT])
    answer = agent.run("Add total(items) to the shop: the sum of price times quantity.")
    assert answer.startswith("Changed shop/cart.py; the project's tests pass.")
    text = (tmp_path / "shop" / "cart.py").read_text()
    assert "def count(items)" in text and "price * qty" in text  # merged, not overwritten
    assert calls[0]["response_format"]["type"] == "json_schema"  # the plan is structured
    assert "response_format" not in calls[1]  # code is written as text
    assert "test_total" in calls[2]["messages"][-1]["content"]  # the failure goes back


def test_routing_sends_changes_with_tests_here_not_questions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV, "")  # every default on
    (tmp_path / "a.py").write_text("x = 1\n")
    assert routing.route_task("Add a function y to a.py", tmp_path).name != "project"  # no tests
    make_project(tmp_path)
    assert routing.route_task("What does count() return?", tmp_path).name != "project"
    assert routing.route_task("Add total() to the cart", tmp_path).name == "project"


def test_plan_never_rewrites_tests_unless_asked(tmp_path: Path, monkeypatch) -> None:
    make_project(tmp_path)
    plan = json.dumps({"analysis": "x", "files": [{"path": "tests/test_cart.py", "change": "edit"},
                                                   {"path": "shop/cart.py", "change": "add total"}]})
    agent, _ = scripted(tmp_path, [plan, plan])
    assert [f["path"] for f in project_pipeline._plan(agent, "Add total() to the cart", "map")] == ["shop/cart.py"]
    asked = project_pipeline._plan(agent, "Add total() and write tests for it", "map")
    assert [f["path"] for f in asked] == ["tests/test_cart.py", "shop/cart.py"]


def test_contract_names_and_check(tmp_path: Path) -> None:
    make_project(tmp_path)
    task = "Make it available as `shop.cart.total(items)` and keep `shop.cart.count`; see `os.path`."
    names = project_pipeline.contract_names(tmp_path, task)
    assert names == ["shop.cart.total", "shop.cart.count"]  # os is not this project's package
    missing = project_pipeline._check_contract(tmp_path, names)
    assert missing == ["shop.cart.total: shop.cart has no total"]


def test_contract_check_finds_src_layout_packages(tmp_path: Path) -> None:
    (tmp_path / "src" / "rollup").mkdir(parents=True)
    (tmp_path / "src" / "rollup" / "__init__.py").write_text("")
    (tmp_path / "src" / "rollup" / "aggregate.py").write_text("def by_region():\n    pass\n")
    names = project_pipeline.contract_names(tmp_path, "Expose `rollup.aggregate.monthly_summary(d)`.")
    assert project_pipeline._check_contract(tmp_path, names) == [
        "rollup.aggregate.monthly_summary: rollup.aggregate has no monthly_summary"]


def test_repeated_plan_entries_become_one_write(tmp_path: Path) -> None:
    make_project(tmp_path)
    plan = json.dumps({"analysis": "x", "files": [{"path": "shop/cart.py", "change": "add total"},
                                                   {"path": "shop/cart.py", "change": "add tax"}]})
    agent, _ = scripted(tmp_path, [plan])
    assert project_pipeline._plan(agent, "Add total()", "map") == [
        {"path": "shop/cart.py", "change": "add total; add tax"}]


def test_writer_reply_with_only_the_new_function_keeps_the_rest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "project_pipeline")
    make_project(tmp_path)
    agent, _ = scripted(tmp_path, [PLAN, RIGHT])
    agent.run("Add total(items) to the shop: the sum of price times quantity.")
    text = (tmp_path / "shop" / "cart.py").read_text()
    assert "def count(items):\n    return len(items)" in text and "def total(items)" in text


SPEC = ("```python\nfrom shop.cart import total\n\n\ndef test_total_uses_quantity():\n"
        "    assert total([(2.5, 2), (1, 3)]) == 8\n```")


def make_untested_feature_project(root: Path) -> None:
    """The project's own tests cover only existing behaviour, like the goals cases."""
    make_project(root)
    (root / "tests" / "test_cart.py").write_text(
        "from shop.cart import count\n\n\ndef test_count():\n    assert count([(1, 2)]) == 1\n")


def test_spec_tests_catch_a_change_the_project_tests_miss(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "project_pipeline,spec_tests")
    make_untested_feature_project(tmp_path)
    agent, calls = scripted(tmp_path, [PLAN, SPEC, WRONG, PLAN, RIGHT])
    answer = agent.run("Add total(items) to the shop: the sum of price times quantity.")
    assert answer == "Changed shop/cart.py; the project's tests pass."
    assert "price * qty" in (tmp_path / "shop" / "cart.py").read_text()  # repaired, not WRONG
    assert "Tests written from the task fail" in calls[3]["messages"][-1]["content"]
    assert not (tmp_path / project_pipeline.SPEC_DIR).exists()  # never left in the project
    assert "spec tests: written, 1 failing before the change" in agent.messages[-1]["content"]


def test_spec_tests_that_already_pass_are_discarded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "project_pipeline,spec_tests")
    make_untested_feature_project(tmp_path)
    passing = ("```python\nfrom shop.cart import count\n\n\ndef test_count_again():\n"
               "    assert count([]) == 0\n```")
    agent, calls = scripted(tmp_path, [PLAN, passing, WRONG])
    answer = agent.run("Add total(items) to the shop: the sum of price times quantity.")
    assert answer == "Changed shop/cart.py; the project's tests pass."
    assert len(calls) == 3  # one round: nothing left failing
    assert "discarded" in agent.messages[-1]["content"]


def test_breaking_project_tests_ranks_worse_than_failing_spec_tests(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "project_pipeline,spec_tests")
    monkeypatch.setattr(project_pipeline, "REPAIRS", 1)
    make_untested_feature_project(tmp_path)
    breaks = "```python\ndef count(items):\n    return 0\n\n\ndef total(items):\n    return 8\n```"
    agent, _ = scripted(tmp_path, [PLAN, SPEC, WRONG, PLAN, breaks])
    answer = agent.run("Add total(items) to the shop: the sum of price times quantity.")
    # round 2 passes the spec test but breaks test_count; round 1 (project tests pass) is kept
    assert answer.startswith("Changed shop/cart.py; the project's tests pass. 1 of the tests")
    assert "return len(items)" in (tmp_path / "shop" / "cart.py").read_text()


def test_rollback_puts_back_a_file_the_change_did_not_need(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "project_pipeline,rollback_breakers")
    make_project(tmp_path)
    (tmp_path / "shop" / "util.py").write_text("def money(x):\n    return f'{x:.2f}'\n")
    (tmp_path / "tests" / "test_util.py").write_text(
        "from shop.util import money\n\n\ndef test_money():\n    assert money(2) == '2.00'\n")
    plan = json.dumps({"analysis": "x", "files": [{"path": "shop/cart.py", "change": "add total"},
                                                  {"path": "shop/util.py", "change": "tidy"}]})
    bad_util = ("```python\nfrom shop.currency import symbol\n\n\ndef money(x):\n"
                "    return symbol + str(x)\n```")  # shop.currency does not exist: util no longer loads
    agent, calls = scripted(tmp_path, [plan, RIGHT, bad_util])
    answer = agent.run("Add total(items) to the shop: the sum of price times quantity.")
    # project_honest_answer: util.py was put back, so it is not reported as changed
    assert answer == ("Changed shop/cart.py; the project's tests pass.\nLeft unchanged: "
                      "shop/util.py (put back as it was: it broke tests/test_util.py).")
    assert "{x:.2f}" in (tmp_path / "shop" / "util.py").read_text()  # put back
    assert "price * qty" in (tmp_path / "shop" / "cart.py").read_text()  # the needed change stays
    assert len(calls) == 3  # no repair round was needed
    assert "put back shop/util.py (it broke tests/test_util.py)" in agent.messages[-1]["content"]


def test_rollback_keeps_an_intended_change_that_fails_old_assertions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "project_pipeline,rollback_breakers")
    monkeypatch.setattr(project_pipeline, "REPAIRS", 0)
    make_project(tmp_path)
    (tmp_path / "tests" / "test_cart.py").write_text(
        "from shop.cart import count\n\n\ndef test_count():\n    assert count([(1, 2)]) == 1\n")
    # the task wants count() to sum quantities, so the old test is meant to break
    changed = "```python\ndef count(items):\n    return sum(q for _, q in items)\n```"
    plan = json.dumps({"analysis": "x", "files": [{"path": "shop/cart.py", "change": "count qty"}]})
    agent, _ = scripted(tmp_path, [plan, changed])
    agent.run("Make count(items) in the shop return the total quantity.")
    # the old test fails an assertion, which a change may intend: only load errors trigger rollback
    assert "sum(q for _, q in items)" in (tmp_path / "shop" / "cart.py").read_text()


def test_spec_tests_protect_an_intended_change_from_rollback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "project_pipeline,rollback_breakers,spec_tests")
    monkeypatch.setattr(project_pipeline, "REPAIRS", 0)
    make_project(tmp_path)
    (tmp_path / "tests" / "test_cart.py").write_text(
        "from shop.cart import count\n\n\ndef test_count():\n    assert count([(1, 2)]) == 1\n")
    spec = ("```python\nfrom shop.cart import count\n\n\ndef test_count_sums_quantities():\n"
            "    assert count([(1, 2), (5, 3)]) == 5\n```")
    changed = "```python\ndef count(items):\n    return sum(q for _, q in items)\n```"
    plan = json.dumps({"analysis": "x", "files": [{"path": "shop/cart.py", "change": "count qty"}]})
    agent, _ = scripted(tmp_path, [plan, spec, changed])
    agent.run("Make count(items) in the shop return the total quantity.")
    # reverting would fail the spec test, so the intended change stays
    assert "sum(q for _, q in items)" in (tmp_path / "shop" / "cart.py").read_text()


@pytest.mark.parametrize("first", ["api/limits.py", "# api/limits.py", "File: api/limits.py",
                                   "**api/limits.py**", "limits.py"])
def test_a_leading_file_name_line_is_dropped(first: str) -> None:
    code = f"{first}\nimport os\n\nX = 1\n"
    assert project_pipeline._drop_path_line(code, "api/limits.py") == "import os\n\nX = 1\n"


def test_a_real_first_line_is_kept() -> None:
    code = "# Rate limits per API key.\nX = 1\n"
    assert project_pipeline._drop_path_line(code, "api/limits.py") == code


def test_unresolved_imports_name_missing_modules_and_names(tmp_path: Path) -> None:
    make_project(tmp_path)
    (tmp_path / "shop" / "sub").mkdir()
    (tmp_path / "shop" / "sub" / "__init__.py").write_text("")
    (tmp_path / "shop" / "sub" / "deep.py").write_text("LIMIT = 3\n")
    (tmp_path / "shop" / "api.py").write_text(
        "import os\nimport shop.cart\nimport shop.models\nfrom shop.cart import count, total\n"
        "from shop.services import Limiter\nfrom shop import cart, sub\nfrom .sub.deep import LIMIT\n"
        "from .sub.deep import RATE\nfrom . import nothing_here\n\n\ndef f():\n    from shop.cart import x\n")
    problems = project_pipeline.unresolved_imports(tmp_path, ["shop/api.py"])
    assert problems == [
        "shop/api.py: `import shop.models`: no such module",
        "shop/api.py: `from shop.cart import total`: shop.cart has no total",
        "shop/api.py: `from shop.services import ...`: no module shop.services",
        "shop/api.py: `from shop.sub.deep import RATE`: shop.sub.deep has no RATE",
        "shop/api.py: `from shop import nothing_here`: shop has no nothing_here",
        "shop/api.py: `from shop.cart import x`: shop.cart has no x",
    ]


def test_unknowable_modules_are_not_flagged(tmp_path: Path) -> None:
    make_project(tmp_path)
    (tmp_path / "shop" / "dyn.py").write_text("from shop.cart import *\n")
    (tmp_path / "shop" / "use.py").write_text("from shop.dyn import anything\nimport json\n")
    assert project_pipeline.unresolved_imports(tmp_path, ["shop/use.py"]) == []


def test_import_grounding_feeds_unresolved_imports_back(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "project_pipeline,import_grounding")
    make_project(tmp_path)
    invented = ("```python\nfrom shop.money import times\n\n\ndef total(items):\n"
                "    return sum(times(p, q) for p, q in items)\n```")
    agent, calls = scripted(tmp_path, [PLAN, invented, PLAN, RIGHT])
    answer = agent.run("Add total(items) to the shop: the sum of price times quantity.")
    assert answer.startswith("Changed shop/cart.py; the project's tests pass.")
    assert "Modules that exist in this project: shop, shop.cart" in calls[1]["messages"][-1]["content"]
    assert "no module shop.money" in calls[2]["messages"][-1]["content"]


def test_missing_stdlib_imports_are_found_conservatively() -> None:
    code = ("def build():\n    parser = argparse.ArgumentParser()\n    return json.dumps({}), re\n\n\n"
            "def other(json):\n    return json\n\n\nre = None\n")
    # json is bound (an argument), re is bound (an assignment): only argparse is missing
    assert project_pipeline.missing_stdlib_imports(code) == ["argparse"]
    assert project_pipeline.missing_stdlib_imports("import os\nprint(os.sep, len([]))\n") == []
    assert project_pipeline.missing_stdlib_imports("x = (\n") == []  # unparsable: leave alone


def test_import_grounding_adds_a_missing_stdlib_import(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "project_pipeline,import_grounding")
    make_project(tmp_path)
    uses_math = ("```python\ndef total(items):\n"
                 "    return math.fsum(price * qty for price, qty in items)\n```")
    agent, _ = scripted(tmp_path, [PLAN, uses_math])
    answer = agent.run("Add total(items) to the shop: the sum of price times quantity.")
    assert answer.startswith("Changed shop/cart.py; the project's tests pass.")
    assert (tmp_path / "shop" / "cart.py").read_text().startswith("import math\n")


# -- large projects (large_projects) ---------------------------------------------------------------

def make_big_project(root: Path, extra_modules: int = 0) -> None:
    """A shop whose store.py is far over FILE_CHARS: a Store class with many methods (one with a
    bug the tests catch), padding functions, and optionally many small modules."""
    (root / "shop").mkdir()
    (root / "shop" / "__init__.py").write_text("")
    methods = "".join(
        f"\n    def metric_{i}(self):\n        \"\"\"Metric {i}.\"\"\"\n"
        f"        return sum(self.items.values()) * {i} + UNRELATED_{i}\n" for i in range(60))
    padding = "".join(f"\n\nUNRELATED_{i} = {i}\n\n\ndef unrelated_{i}(x):\n    return x + UNRELATED_{i}\n"
                      for i in range(60))
    (root / "shop" / "store.py").write_text(
        '"""Stock."""\n\nfrom __future__ import annotations\n' + padding +
        "\n\nclass Store:\n    \"\"\"A shop's stock.\"\"\"\n\n    def __init__(self):\n        self.items = {}\n"
        "\n    def add(self, item, qty):\n        self.items[item] = self.items.get(item, 0) + qty\n"
        "\n    def restock(self, item, qty):\n        self.items[item] = qty  # bug: replaces instead of adding\n"
        + methods)
    for i in range(extra_modules):
        (root / "shop" / f"mod_{i}.py").write_text(f"VALUE = {i}\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_store.py").write_text(
        "from shop.store import Store\n\n\ndef test_add():\n    s = Store()\n    s.add('a', 2)\n"
        "    assert s.items == {'a': 2}\n\n\ndef test_restock_adds():\n    s = Store()\n"
        "    s.add('a', 2)\n    s.restock('a', 3)\n    assert s.items == {'a': 5}\n\n\n"
        "def test_metric():\n    s = Store()\n    s.add('a', 1)\n    assert s.metric_2() == 4\n")


def test_routes_big_projects_here_only_with_the_switch(tmp_path: Path, monkeypatch) -> None:
    make_big_project(tmp_path, extra_modules=90)
    assert not routing._project_size_ok(tmp_path)  # over 80 files
    monkeypatch.setenv(features.ENV_ON, "large_projects")
    assert routing._project_size_ok(tmp_path)
    monkeypatch.setattr(project_pipeline, "LARGE_MAX_PYTHON_FILES", 50)  # a limit remains
    assert not routing._project_size_ok(tmp_path)


def test_large_file_view_shows_outline_and_relevant_source_only(tmp_path: Path) -> None:
    make_big_project(tmp_path)
    source = (tmp_path / "shop" / "store.py").read_text()
    assert len(source) > project_pipeline.FILE_CHARS
    view = project_pipeline.focused_view("shop/store.py", source, "Fix restocking in the shop.",
                                         "make `Store.restock` add to the existing quantity")
    outline = view.split("Outline of shop/store.py", 1)[1].split("Imports of", 1)[0]
    lines = source.split("\n")
    start = next(n for n, line in enumerate(lines, 1) if "def restock" in line)
    assert f"    restock: {start}-{start + 1}" in outline
    assert all(f"    metric_{i}:" in outline for i in range(60)) and "def unrelated_59:" in outline
    relevant = view.split("most relevant to this change:", 1)[1]
    assert "self.items[item] = qty  # bug" in relevant  # the named method, complete
    assert "class Store:" in relevant  # inside its class header
    assert "+ UNRELATED_7\n" not in relevant  # unrelated bodies are left out
    assert "class Store:\n    def restock(self, ...):" in view  # the reply format, by example


def test_large_file_view_respects_the_budget(tmp_path: Path) -> None:
    make_big_project(tmp_path)
    source = (tmp_path / "shop" / "store.py").read_text()
    task = " ".join(f"metric_{i}()" for i in range(60))  # names far more than fit
    view = project_pipeline.focused_view("shop/store.py", source, task, "change the metrics",
                                         budget=800)
    block = view.split("most relevant to this change:\n```python\n", 1)[1].split("\n```", 1)[0]
    shown = [line for line in block.split("\n") if not line.strip().startswith(("# lines", "class "))]
    assert 0 < sum(len(line) + 1 for line in shown) <= 800
    assert "def metric_0" in block and "def metric_59" not in block


def test_end_to_end_one_method_changed_in_a_large_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "large_projects")
    monkeypatch.setenv(features.ENV, "clarify_model")  # its unsure-verdict call is not scripted
    make_big_project(tmp_path, extra_modules=90)
    (tmp_path / "tests" / "test_known.py").write_text(
        "def test_already_broken():\n    assert False\n")  # fails before the change too
    plan = json.dumps({"analysis": "restock replaces the quantity",
                       "files": [{"path": "shop/store.py",
                                  "change": "fix Store.restock to add qty to the existing count"}]})
    fixed = ("```python\nclass Store:\n    def restock(self, item, qty):\n"
             "        self.items[item] = self.items.get(item, 0) + qty\n```")
    agent, calls = scripted(tmp_path, [plan, fixed, plan, fixed])
    before = (tmp_path / "shop" / "store.py").read_text()
    answer = agent.run("Restocking an item replaces its quantity instead of adding to it; fix it.")
    after = (tmp_path / "shop" / "store.py").read_text()
    assert after == before.replace("self.items[item] = qty  # bug: replaces instead of adding",
                                   "self.items[item] = self.items.get(item, 0) + qty")
    assert answer == ("Changed shop/store.py; no test that passed before fails now, but 1 test(s) "
                      "that failed before the change still fail.")
    assert len(calls) == 4  # one repair round fixed none of the old failures: stop there
    writer = calls[1]["messages"][-1]["content"]
    assert "shop/store.py is large" in writer and "def metric_30" not in writer
    assert "Files most related to the task" in calls[0]["messages"][-1]["content"]
    assert "tests/test_known.py::test_already_broken" in calls[2]["messages"][-1]["content"]


def test_end_to_end_large_file_fix_then_the_tests_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "large_projects")
    monkeypatch.setenv(features.ENV, "clarify_model")  # its unsure-verdict call is not scripted
    make_big_project(tmp_path, extra_modules=90)
    plan = json.dumps({"analysis": "x", "files": [{"path": "shop/store.py",
                                                   "change": "fix Store.restock"}]})
    fixed = ("Here is the fix:\n```python\nclass Store:\n    def restock(self, item, qty):\n"
             "        self.items[item] = self.items.get(item, 0) + qty\n```")
    agent, calls = scripted(tmp_path, [plan, fixed])
    answer = agent.run("Restocking an item replaces its quantity instead of adding to it; fix it.")
    assert answer == "Changed shop/store.py; the project's tests pass."
    assert len(calls) == 2
    text = (tmp_path / "shop" / "store.py").read_text()
    assert text.count("    def metric_") == 60 and "def unrelated_59(x)" in text
