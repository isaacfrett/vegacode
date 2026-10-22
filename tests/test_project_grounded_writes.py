"""Project pipeline, project diagnosis 2026-10-20 fix 4 and the fix 1 follow-up: route targets in
the plan (project_route_targets), automatic imports (project_auto_imports), the re-export guard
(project_reexport_guard) and holding back files that import a refused write (project_hold_back).

The shapes are the ones the diagnosis reproduced: feeds (`TransportError`), billing (`pad_columns`,
`DecimalException`, `from .money import Money` while money.py was refused), choremate (`CycleError`,
`UnknownTaskError`), docs build.py (`re`, `sys`, `argparse`), orders (`from .records import ...
total_cents, to_dict, from_dict`), and todo (`todo export --csv` needs todo/cli.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import edits, features, project_pipeline, sandbox
from vega.agent import CodingAgent


@pytest.fixture(autouse=True)
def pipeline_on(monkeypatch) -> None:
    monkeypatch.delenv("VEGA_FEATURES_OFF", raising=False)


def write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def scripted(root: Path, replies: list[str]) -> tuple[CodingAgent, list[dict]]:
    agent = CodingAgent(root, allow_commands=False)
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=replies.pop(0)), finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, calls


def imports_ok(root: Path, module: str) -> bool:
    result = subprocess.run([sys.executable, "-c", f"import {module}"], cwd=root,
                            capture_output=True, text=True, check=False)
    return result.returncode == 0


def plan(*items: tuple[str, str]) -> str:
    return json.dumps({"analysis": "a", "files": [{"path": p, "change": c} for p, c in items]})


def block(path: str, code: str) -> str:
    return f"File: {path}\n```python\n{code}```"


# -- project_route_targets -------------------------------------------------------------------------

TODO_ROUTE = {"name": "project", "targets": ["todo/store.py", "todo/cli.py"],
              "reasons": ["asks for a change: make",
                          "also a target: todo/cli.py (runs the backticked command)",
                          "several code targets with visible tests"]}


def test_route_targets_carry_routing_reason(tmp_path: Path) -> None:
    seeds = project_pipeline.route_targets(tmp_path, TODO_ROUTE, "Add `todo export --csv`.")
    assert seeds == [{"path": "todo/store.py", "why": ""},
                     {"path": "todo/cli.py", "why": "runs the backticked command"}]
    route = {"targets": ["tests/test_cli.py", "../x.py", "notes.md", "pkg/a.py", "pkg/cli.py"],
             "reasons": ["also a target: pkg/cli.py (runs the backticked command)"]}
    assert [s["path"] for s in project_pipeline.route_targets(tmp_path, route, "Fix it.")] == \
        ["pkg/a.py", "pkg/cli.py"]  # no test file unless asked, nothing outside, only Python
    assert [s["path"] for s in project_pipeline.route_targets(
        tmp_path, route, "Fix it and add a regression test.")] == \
        ["tests/test_cli.py", "pkg/a.py", "pkg/cli.py"]


def test_a_plain_project_route_forces_nothing(tmp_path: Path) -> None:
    """A task routed with src/postfee/fees.py as a target although it says "do not change the
    module": only a route routing widened with a found file counts."""
    route = {"name": "project", "targets": ["src/postfee/fees.py", "tests/test_fees.py"],
             "reasons": ["asks for a change: write", "several code targets with visible tests"]}
    assert project_pipeline.route_targets(tmp_path, route, "Write `tests/test_fees.py`.") == []


def test_with_route_targets_matches_src_layout_paths() -> None:
    planned = [{"path": "src/quillnote/app.py", "change": "move"}]
    merged, added = project_pipeline.with_route_targets(planned, [{"path": "quillnote/app.py",
                                                                   "why": "x"}])
    assert added == [] and merged == planned


def test_with_route_targets_adds_only_what_the_planner_left_out() -> None:
    planned = [{"path": "todo/store.py", "change": "add export_csv"},
               {"path": "todo/models.py", "change": "add a csv row helper"}]
    seeds = [{"path": "todo/store.py", "why": ""},
             {"path": "todo/cli.py", "why": "runs the backticked command"}]
    merged, added = project_pipeline.with_route_targets(planned, seeds)
    assert added == ["todo/cli.py"]
    assert [p["path"] for p in merged] == ["todo/store.py", "todo/models.py", "todo/cli.py"]
    assert merged[0]["change"] == "add export_csv"  # the planner's own items are untouched
    assert "runs the backticked command" in merged[2]["change"]


TODO_FILES = {
    "todo/__init__.py": "",
    "todo/store.py": "class TodoStore:\n    def __init__(self):\n        self.items = []\n",
    "todo/cli.py": ("import argparse\n\n\ndef build_parser():\n"
                    "    parser = argparse.ArgumentParser(prog=\"todo\")\n"
                    "    sub = parser.add_subparsers(dest=\"cmd\")\n    sub.add_parser(\"list\")\n"
                    "    return parser\n"),
    "tests/test_store.py": ("from todo.store import TodoStore\n\n\ndef test_new():\n"
                            "    assert TodoStore().items == []\n"),
}
TODO_TASK = ("Add CSV export: `TodoStore.export_csv()` returns the items as CSV, and "
             "`todo export --csv` prints it.")


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_route_targets_reach_the_plan_and_get_written(tmp_path: Path) -> None:
    write(tmp_path, TODO_FILES)
    store = ("class TodoStore:\n    def __init__(self):\n        self.items = []\n\n"
             "    def export_csv(self):\n        return \"\"\n")
    cli = ("def build_parser():\n    parser = argparse.ArgumentParser(prog=\"todo\")\n"
           "    sub = parser.add_subparsers(dest=\"cmd\")\n    sub.add_parser(\"list\")\n"
           "    export = sub.add_parser(\"export\")\n    export.add_argument(\"--csv\", "
           "action=\"store_true\")\n    return parser\n")
    agent, calls = scripted(tmp_path, [plan(("todo/store.py", "add export_csv")),
                                       block("todo/store.py", store), block("todo/cli.py", cli)])
    agent.last_route = {**TODO_ROUTE, "inputs": [], "task_chars": len(TODO_TASK)}
    answer = project_pipeline.run(agent, TODO_TASK)
    planner = calls[0]["messages"][-1]["content"]
    assert "These files must change for this task" in planner
    assert "- todo/cli.py (runs the backticked command)" in planner
    assert "Now write todo/cli.py" in calls[2]["messages"][-1]["content"]
    assert "\"--csv\"" in (tmp_path / "todo" / "cli.py").read_text()
    assert answer.startswith("Changed todo/store.py, todo/cli.py")


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_route_targets_off_plans_from_the_planner_alone(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV, "project_route_targets")
    write(tmp_path, TODO_FILES)
    store = ("class TodoStore:\n    def __init__(self):\n        self.items = []\n\n"
             "    def export_csv(self):\n        return \"\"\n")
    agent, calls = scripted(tmp_path, [plan(("todo/store.py", "add export_csv")),
                                       block("todo/store.py", store)])
    agent.last_route = {**TODO_ROUTE, "inputs": [], "task_chars": len(TODO_TASK)}
    project_pipeline.run(agent, TODO_TASK)
    assert len(calls) == 2
    assert "must change for this task" not in calls[0]["messages"][-1]["content"]


# -- project_auto_imports --------------------------------------------------------------------------

FEEDS = {
    "feeds/__init__.py": "",
    "feeds/errors.py": ("class FeedError(Exception):\n    pass\n\n\n"
                        "class TransportError(FeedError):\n    pass\n"),
    "feeds/clock.py": "import time\n\n\ndef sleep(seconds):\n    time.sleep(seconds)\n",
}
RETRY = ("def call_with_retry(fn, attempts=3):\n    for attempt in range(attempts):\n"
         "        try:\n            return fn()\n        except TransportError:\n"
         "            if attempt == attempts - 1:\n                raise\n")


def test_auto_import_of_a_name_one_project_module_defines(tmp_path: Path) -> None:
    write(tmp_path, FEEDS)
    content, added = project_pipeline.auto_imports(tmp_path, "feeds/retry.py", RETRY)
    assert added == ["from .errors import TransportError"]
    assert content.startswith("from .errors import TransportError\n")
    assert edits.check("pipeline", "feeds/retry.py", None, RETRY, tmp_path).refusal is not None
    assert edits.check("pipeline", "feeds/retry.py", None, content, tmp_path).refusal is None


def test_auto_import_follows_the_files_absolute_style(tmp_path: Path) -> None:
    write(tmp_path, {
        "choremate/__init__.py": "",
        "choremate/errors.py": ("class CycleError(Exception):\n    pass\n\n\n"
                            "class UnknownTaskError(KeyError):\n    pass\n"),
        "choremate/graph.py": "def resolve_order(tasks):\n    return list(tasks)\n",
    })
    cli = ("from choremate.graph import resolve_order\n\n\ndef main(tasks):\n    try:\n"
           "        return resolve_order(tasks)\n    except (CycleError, UnknownTaskError) as exc:\n"
           "        return str(exc)\n")
    content, added = project_pipeline.auto_imports(tmp_path, "choremate/cli.py", cli)
    assert added == ["from choremate.errors import CycleError",
                     "from choremate.errors import UnknownTaskError"]
    assert edits.check("pipeline", "choremate/cli.py", None, content, tmp_path).refusal is None


def test_auto_import_of_a_helper_and_a_stdlib_name(tmp_path: Path) -> None:
    write(tmp_path, {
        "billing/__init__.py": "",
        "billing/formatting.py": "def pad_columns(rows):\n    return rows\n",
    })
    invoice = ("def render(rows):\n    return pad_columns(rows)\n")
    content, added = project_pipeline.auto_imports(tmp_path, "billing/invoice.py", invoice)
    assert added == ["from .formatting import pad_columns"]
    money = ("def parse(text):\n    try:\n        return Decimal(text)\n"
             "    except DecimalException:\n        raise ValueError(text)\n")
    content, added = project_pipeline.auto_imports(tmp_path, "billing/money.py", money)
    assert added == ["from decimal import Decimal", "from decimal import DecimalException"]
    assert edits.check("pipeline", "billing/money.py", None, content, tmp_path).refusal is None


def test_auto_import_of_stdlib_modules_hidden_by_a_function_local_import(tmp_path: Path) -> None:
    """docs build.py's shape: `re` is imported inside one function and used in another, so the
    whole-file check of missing_stdlib_imports sees it bound; ruff's F821 does not."""
    write(tmp_path, {"docs/__init__.py": ""})
    build = ("def slug(text):\n    import re\n    return re.sub(r'\\W+', '-', text)\n\n\n"
             "def anchors(lines):\n    return [re.sub(r'\\s', '', line) for line in lines]\n\n\n"
             "def main():\n    parser = argparse.ArgumentParser()\n"
             "    parser.parse_args(sys.argv[1:])\n")
    assert project_pipeline.missing_stdlib_imports(build) == ["argparse", "sys"]  # no `re`
    content, added = project_pipeline.auto_imports(tmp_path, "docs/build.py", build)
    if sandbox.find_python_tool(tmp_path, "ruff") is not None:  # scope-aware findings
        assert sorted(added) == ["import argparse", "import re", "import sys"]
    else:
        assert sorted(added) == ["import argparse", "import sys"]
    assert edits.check("pipeline", "docs/build.py", None, content, tmp_path).refusal is None


def test_auto_import_leaves_ambiguous_circular_and_unknown_names(tmp_path: Path) -> None:
    write(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/a.py": "class Thing:\n    pass\n",
        "pkg/b.py": "class Thing:\n    pass\n",
        "pkg/errors.py": "from .retry import call\n\n\nclass Boom(Exception):\n    pass\n",
    })
    code = ("def call():\n    raise Boom(Thing(), last_time)\n")
    content, added = project_pipeline.auto_imports(tmp_path, "pkg/retry.py", code)
    assert added == [] and content == code  # Thing: two homes; Boom: errors imports retry
    # through another module: errors -> util -> retry would be a cycle as well
    write(tmp_path, {"pkg/errors.py": "from . import util\n\n\nclass Boom(Exception):\n    pass\n",
                     "pkg/util.py": "def helper():\n    from pkg.retry import call\n    return call\n"})
    assert project_pipeline.auto_imports(tmp_path, "pkg/retry.py", code)[1] == []
    # a variable that is not used as a module is never imported as one
    content, added = project_pipeline.auto_imports(tmp_path, "pkg/c.py", "def f():\n    return string\n")
    assert added == []


def test_auto_import_never_crosses_into_a_sibling_package(tmp_path: Path) -> None:
    """The writer used `status_row` (truckyard.web) in truckyard/domain/planner.py; the task
    forbids the domain importing the web layer."""
    write(tmp_path, {
        "truckyard/__init__.py": "VERSION = '1'\n",
        "truckyard/web/render.py": "def status_row(v, s):\n    return f'{v} {s}'\n",
        "truckyard/domain/rules.py": "def status(v, today):\n    return 'ok'\n",
    })
    code = ("def due(vehicles, today):\n    return [status_row(v, status(v, today)) "
            "for v in vehicles], VERSION\n")
    _, added = project_pipeline.auto_imports(tmp_path, "truckyard/domain/planner.py", code)
    assert added == ["from .rules import status", "from .. import VERSION"]


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_auto_import_saves_the_write_feeds_round_one_lost(tmp_path: Path) -> None:
    write(tmp_path, {**FEEDS, "tests/test_retry.py": (
        "from feeds.errors import TransportError\nfrom feeds.retry import call_with_retry\n\n\n"
        "def test_retry():\n    calls = []\n\n    def flaky():\n        calls.append(1)\n"
        "        if len(calls) < 2:\n            raise TransportError()\n        return 'ok'\n\n"
        "    assert call_with_retry(flaky) == 'ok'\n")})
    task = "Add `feeds.retry.call_with_retry(fn, attempts=3)` that retries on TransportError."
    agent, _ = scripted(tmp_path, [plan(("feeds/retry.py", "add call_with_retry")),
                                   block("feeds/retry.py", RETRY)])
    answer = project_pipeline.run(agent, task)
    assert answer.startswith("Changed feeds/retry.py; the project's tests pass."), answer
    assert "from .errors import TransportError" in (tmp_path / "feeds" / "retry.py").read_text()


# -- project_reexport_guard ------------------------------------------------------------------------

RECORDS = ("from dataclasses import dataclass, field\n\n\n@dataclass\nclass OrderLine:\n"
           "    sku: str\n    cents: int\n    qty: int = 1\n\n    @property\n"
           "    def total_cents(self):\n        return self.cents * self.qty\n\n\n@dataclass\n"
           "class Order:\n    lines: list = field(default_factory=list)\n\n"
           "    def to_dict(self):\n        return {'lines': len(self.lines)}\n\n"
           "    @classmethod\n    def from_dict(cls, data):\n        return cls()\n")
ORDERS_INIT = ('"""Orders."""\nfrom .records import OrderLine, Order, total_cents, to_dict, '
               'from_dict\n\n__all__ = ["OrderLine", "Order", "total_cents", "to_dict", '
               '"from_dict"]\n')


def test_reexports_of_methods_are_dropped(tmp_path: Path) -> None:
    write(tmp_path, {"orders/records.py": RECORDS, "orders/__init__.py": ORDERS_INIT})
    assert not imports_ok(tmp_path, "orders")
    bad = project_pipeline.bad_reexports(tmp_path, "orders/__init__.py", ORDERS_INIT)
    assert bad == {"orders.records": ["total_cents", "to_dict", "from_dict"]}
    fixed = project_pipeline.drop_reexports(ORDERS_INIT, "orders/__init__.py", bad)
    assert fixed == ('"""Orders."""\nfrom .records import OrderLine, Order\n\n'
                     "__all__ = ['OrderLine', 'Order']\n")
    (tmp_path / "orders" / "__init__.py").write_text(fixed)
    assert imports_ok(tmp_path, "orders")


def test_reexport_guard_keeps_real_names_and_names_the_init_uses(tmp_path: Path) -> None:
    write(tmp_path, {"orders/records.py": RECORDS})
    init = "from .records import Order, helper\n\nDEFAULT = helper()\n"
    assert project_pipeline.bad_reexports(tmp_path, "orders/__init__.py", init) == {}
    assert project_pipeline.bad_reexports(tmp_path, "orders/api.py", ORDERS_INIT) == {}  # not __init__


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_reexport_guard_in_a_round_keeps_the_package_importable(tmp_path: Path) -> None:
    write(tmp_path, {
        "orders/__init__.py": '"""Orders."""\n',
        "orders/records.py": "class OrderLine:\n    pass\n\n\nclass Order:\n    pass\n",
        "tests/test_orders.py": ("from orders import Order, OrderLine\n\n\ndef test_line():\n"
                                 "    assert OrderLine('a', 5, 2).total_cents == 10\n"),
    })
    task = "Convert them to dataclasses in `orders/records.py`: `OrderLine` and `Order`."
    agent, _ = scripted(tmp_path, [
        plan(("orders/records.py", "dataclasses"), ("orders/__init__.py", "re-export")),
        block("orders/records.py", RECORDS),
        "File: orders/__init__.py\n```python\n" + ORDERS_INIT.split("\n", 1)[1] + "```"])
    answer = project_pipeline.run(agent, task)
    assert answer.startswith("Changed orders/records.py, orders/__init__.py; the project's tests "
                             "pass."), answer
    assert "total_cents" not in (tmp_path / "orders" / "__init__.py").read_text()


# -- project_hold_back -----------------------------------------------------------------------------

def test_hold_back_puts_back_a_file_importing_a_refused_new_module(tmp_path: Path) -> None:
    """billing round 1: money.py (new) refused, __init__ saved with `from .money import Money`."""
    write(tmp_path, {"billing/__init__.py": "from .money import Money\n",
                     "billing/invoice.py": "def total():\n    return 0\n"})
    before = {"billing/__init__.py": '"""Billing."""\n', "billing/money.py": None,
              "billing/invoice.py": "def total():\n    return 0\n"}
    held = project_pipeline.hold_back(tmp_path, ["billing/__init__.py", "billing/invoice.py"],
                                      ["billing/money.py"], before)
    assert held == {"billing/__init__.py": "billing/money.py"}
    assert (tmp_path / "billing" / "__init__.py").read_text() == '"""Billing."""\n'
    assert imports_ok(tmp_path, "billing")


def test_hold_back_is_transitive_and_spares_imports_that_resolve(tmp_path: Path) -> None:
    """feeds: client.py imports the refused feeds.retry; __init__ imports a name client.py had
    only in this round's version; errors.py imports nothing from them."""
    old_client = "class FeedClient:\n    pass\n"
    write(tmp_path, {
        "feeds/errors.py": "class TransportError(Exception):\n    pass\n",
        "feeds/client.py": ("from feeds.retry import call_with_retry\n\n\nclass FeedClient:\n"
                            "    pass\n\n\nDEFAULT_DELAY = 1\n"),
        "feeds/__init__.py": "from .client import FeedClient, DEFAULT_DELAY\n",
        "feeds/util.py": "from .client import FeedClient\n",
    })
    before = {"feeds/client.py": old_client, "feeds/__init__.py": "", "feeds/util.py": None,
              "feeds/errors.py": "", "feeds/retry.py": None}
    saved = ["feeds/errors.py", "feeds/client.py", "feeds/__init__.py", "feeds/util.py"]
    held = project_pipeline.hold_back(tmp_path, saved, ["feeds/retry.py"], before)
    assert held == {"feeds/client.py": "feeds/retry.py", "feeds/__init__.py": "feeds/client.py"}
    assert (tmp_path / "feeds" / "client.py").read_text() == old_client
    assert (tmp_path / "feeds" / "util.py").is_file()  # FeedClient still resolves
    assert imports_ok(tmp_path, "feeds.util")


def test_hold_back_leaves_an_existing_module_whose_names_still_resolve(tmp_path: Path) -> None:
    write(tmp_path, {"pkg/__init__.py": "", "pkg/money.py": "class Money:\n    pass\n",
                     "pkg/invoice.py": "from .money import Money\n"})
    held = project_pipeline.hold_back(tmp_path, ["pkg/invoice.py"], ["pkg/money.py"],
                                      {"pkg/invoice.py": None})
    assert held == {}


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_hold_back_in_a_round_keeps_the_package_importable(tmp_path: Path) -> None:
    write(tmp_path, {
        "billing/__init__.py": '"""Billing."""\n',
        "billing/statement.py": "def lines():\n    return []\n",
        "tests/test_statement.py": ("from billing.statement import lines\n\n\ndef test_lines():\n"
                                    "    assert lines() == []\n"),
    })
    money_bad = "class Money:\n    def __init__(self, cents):\n        self.cents = cents + offset\n"
    init = '"""Billing."""\nfrom .money import Money\n'
    agent, calls = scripted(tmp_path, [
        plan(("billing/money.py", "add Money"), ("billing/__init__.py", "re-export Money")),
        block("billing/money.py", money_bad), block("billing/__init__.py", init),
        plan(("billing/money.py", "fix")),  # the replan; the held __init__ stays in the plan
        block("billing/money.py", "class Money:\n    def __init__(self, cents):\n"
                                  "        self.cents = cents\n"),
        block("billing/__init__.py", init)])
    answer = project_pipeline.run(agent, "Add a `billing.money.Money` type and export it.")
    round_two = calls[5]["messages"][-1]["content"]
    assert "Held back, not saved: billing/__init__.py imports billing.money" in round_two
    assert answer.startswith("Changed billing/money.py, billing/__init__.py; the project's tests "
                             "pass."), answer
    assert imports_ok(tmp_path, "billing")


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_hold_back_off_saves_the_importer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV, "project_hold_back")
    write(tmp_path, {"billing/__init__.py": '"""Billing."""\n',
                     "tests/test_x.py": "import billing\n\n\ndef test_x():\n    assert billing\n"})
    agent, _ = scripted(tmp_path, [
        plan(("billing/money.py", "add Money"), ("billing/__init__.py", "re-export Money")),
        block("billing/money.py", "x = offset\n"), block("billing/__init__.py",
                                                         "from .money import Money\n"),
        *[plan(("billing/money.py", "fix")), block("billing/money.py", "x = offset\n"),
          block("billing/__init__.py", "from .money import Money\n")] * 3])
    project_pipeline.run(agent, "Add a `billing.money.Money` type and export it.")
    assert not imports_ok(tmp_path, "billing")
