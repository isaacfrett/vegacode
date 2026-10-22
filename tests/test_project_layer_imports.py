"""project_layer_imports (engineering diagnosis 2026-10-21, fix 1): the stdlib name table derived
from the stdlib, `datetime` as class or module, `import pytest` in test files, imports the task
forbids, and the refusal for a name that lives in a layer the file must not import.

Shapes: a domain planner.py (`date` in an annotation; then `status_row` from truckyard.web), a
test update using `pytest.raises`."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import edits, project_pipeline
from vega.agent import CodingAgent


@pytest.fixture(autouse=True)
def features_on(monkeypatch) -> None:
    monkeypatch.delenv("VEGA_FEATURES_OFF", raising=False)


def write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


FLEET_TASK = (
    "truckyard/domain is meant to be pure business logic, but domain/planner.py reads the CSV file "
    "itself (truckyard.storage) and formats text for the web layer (truckyard.web). Restructure it "
    "as ports and adapters:\n- No module under truckyard/domain may import truckyard.storage or "
    "truckyard.web (anywhere in the file, including inside functions), import csv, io, os, "
    "pathlib or importlib, or call open() or __import__().\n- Add truckyard/domain/ports.py.")
FLEET = {
    "truckyard/__init__.py": "",
    "truckyard/domain/__init__.py": "",
    "truckyard/domain/models.py": "class Vehicle:\n    id = 0\n",
    "truckyard/domain/rules.py": "def status(vehicle, today):\n    return 'ok'\n",
    "truckyard/storage/__init__.py": "",
    "truckyard/storage/csv_store.py": "def load_vehicles(path):\n    return []\n",
    "truckyard/web/__init__.py": "",
    "truckyard/web/render.py": "def status_row(vehicle, state):\n    return f'{vehicle} {state}'\n",
}
# round 1 of the logged run (refused for `date` only)
PLANNER_ROUND1 = (
    '"""Which vehicles need a workshop slot."""\nfrom typing import List\n'
    "from truckyard.domain.rules import status\nfrom truckyard.domain.models import Vehicle\n\n"
    'ORDER = {"overdue": 0, "soon": 1}\n\n\nclass Planner:\n'
    "    def __init__(self, repository):\n        self.repository = repository\n\n"
    "    def due(self, today: date) -> List[tuple[Vehicle, str]]:\n"
    "        found = [(v, status(v, today)) for v in self.repository.list_vehicles()]\n"
    "        return [(v, s) for v, s in found if s != 'ok']\n")
# rounds 2-4: a leftover report() still uses the web layer's status_row
PLANNER_ROUND2 = (
    "from truckyard.domain.rules import status\n\n\nclass Planner:\n"
    "    def __init__(self, repository):\n        self.repository = repository\n\n"
    "    def due(self, today):\n        return [(v, status(v, today)) for v in "
    "self.repository.list_vehicles()]\n\n    def report(self, today):\n"
    "        return '\\n'.join(status_row(v, s) for v, s in self.due(today))\n")


# -- the derived stdlib table ----------------------------------------------------------------------

def test_stdlib_table_covers_common_names_and_leaves_out_ambiguous_ones() -> None:
    table = project_pipeline.stdlib_names()
    expected = {"date": "datetime", "timedelta": "datetime", "Path": "pathlib", "List": "typing",
                "Dict": "typing", "Callable": "typing", "dataclass": "dataclasses",
                "Decimal": "decimal", "ROUND_HALF_UP": "decimal", "defaultdict": "collections",
                "Counter": "collections", "OrderedDict": "collections", "UUID": "uuid",
                "deepcopy": "copy", "ArgumentParser": "argparse", "TemporaryDirectory": "tempfile",
                "JSONDecodeError": "json", "MagicMock": "unittest.mock", "Fraction": "fractions",
                "TYPE_CHECKING": "typing"}
    assert {name: table.get(name) for name in expected} == expected
    # two homes (threading/asyncio Event, concurrent.futures/asyncio Future, typing/abc Set),
    # generic words a project means its own, lowercase helpers, stdlib modules and builtins
    for name in ("Event", "Future", "Set", "Template", "Overflow", "Action", "MISSING",
                 "product", "chain", "count", "replace", "merge", "datetime", "time", "copy",
                 "open", "ValueError"):
        assert name not in table, name
    # every hand-listed name kept, from the same module
    assert all(table[name] == module for name, module in project_pipeline.STDLIB_NAMES.items())
    for name, module in table.items():
        assert hasattr(importlib.import_module(module), name), (name, module)


def test_date_in_an_annotation_is_imported_fleet_round_one(tmp_path: Path, monkeypatch) -> None:
    write(tmp_path, FLEET)
    relative = "truckyard/domain/planner.py"
    assert edits.check("pipeline", relative, None, PLANNER_ROUND1, tmp_path).refusal is not None
    resolved = project_pipeline.resolve_imports(tmp_path, relative, PLANNER_ROUND1)
    assert resolved.added == ["from datetime import date"] and resolved.foreign == {}
    assert edits.check("pipeline", relative, None, resolved.content, tmp_path).refusal is None
    monkeypatch.setenv("VEGA_FEATURES_OFF", "project_layer_imports")
    assert project_pipeline.resolve_imports(tmp_path, relative, PLANNER_ROUND1).added == []


@pytest.mark.parametrize(("code", "line"), [
    ("def stamp():\n    return datetime.now().isoformat()\n", "from datetime import datetime"),
    ("def parse(t) -> datetime:\n    return datetime.strptime(t, '%Y')\n",
     "from datetime import datetime"),
    ("def first():\n    return datetime(2026, 1, 1)\n", "from datetime import datetime"),
    ("def later(d):\n    return d + datetime.timedelta(days=1)\n", "import datetime"),
    ("def today():\n    return datetime.date.today()\n", "import datetime"),
])
def test_bare_datetime_is_the_class_or_the_module_by_use(tmp_path: Path, code: str,
                                                         line: str) -> None:
    write(tmp_path, {"app/__init__.py": ""})
    assert project_pipeline.resolve_imports(tmp_path, "app/clock.py", code).added == [line]


def test_mixed_datetime_use_is_left_to_the_writer(tmp_path: Path) -> None:
    write(tmp_path, {"app/__init__.py": ""})
    code = "def f():\n    return datetime.now(), datetime.timedelta(days=1)\n"
    assert project_pipeline.resolve_imports(tmp_path, "app/clock.py", code).added == []


def test_pytest_and_mock_are_imported_in_a_test_file_only(tmp_path: Path) -> None:
    """A test update used pytest.raises without the import; another used @mock.patch without
    `from unittest import mock`."""
    write(tmp_path, {"billsplit/__init__.py": "", "billsplit/money.py": "class Money:\n    pass\n"})
    test = ("from billsplit.money import Money\n\n\ndef test_float():\n"
            "    with pytest.raises(TypeError):\n        Money(1.5)\n")
    resolved = project_pipeline.resolve_imports(tmp_path, "tests/test_billsplit.py", test)
    assert resolved.added == ["import pytest"]
    assert edits.check("pipeline", "tests/test_billsplit.py", None, resolved.content,
                       tmp_path).refusal is None
    assert project_pipeline.resolve_imports(tmp_path, "billsplit/check.py", test).added == []
    test = "@mock.patch('warnings.warn')\ndef test_warns(warn):\n    assert warn\n"
    assert project_pipeline.resolve_imports(tmp_path, "tests/test_w.py", test).added == [
        "from unittest import mock"]
    # a bare `mock` that is not used as a module is somebody's fixture, not unittest.mock
    test = "def test_x():\n    assert mock\n"
    assert project_pipeline.resolve_imports(tmp_path, "tests/test_w.py", test).added == []


# -- what the task forbids -------------------------------------------------------------------------

def test_forbidden_imports_from_the_task_text() -> None:
    forbidden = project_pipeline.forbidden_imports
    assert forbidden(FLEET_TASK, "truckyard/domain/planner.py") == ("truckyard.storage",
                                                                   "truckyard.web")
    assert forbidden(FLEET_TASK, "truckyard/web/app.py") == ()  # the rule is about the domain
    assert forbidden(FLEET_TASK, "tests/test_truckyard.py") == ()
    stock = ("- `InventoryService(repository)` takes any object. binstore/service.py must not "
             "import json, os, pathlib, binstore.ledger or anything in binstore.adapters, and "
             "the service must never touch the filesystem itself.")
    assert forbidden(stock, "binstore/service.py") == ("binstore.ledger", "binstore.adapters")
    assert forbidden(stock, "binstore/adapters/json_files.py") == ()
    # report.py must import from ledger.parsing; parsing.py must not
    # import ledger.report
    extract = ("Afterwards `ledger/report.py` defines only `build_report` (importing what it "
               "needs from `ledger.parsing`), `ledger/parsing.py` must not import "
               "`ledger.report`, and `ledger.report.build_report` behaves exactly the same.")
    assert forbidden(extract, "ledger/report.py") == ()
    assert forbidden(extract, "ledger/parsing.py") == ("ledger.report",)
    # "no cycle ... every import statement counts" forbids nothing
    quill = ("Afterwards the import graph of the penbook package must have no cycle at all, "
             "where every import statement in a penbook module counts, and `from penbook "
             "import x` counts as importing penbook.x. Don't use importlib to get around it.")
    assert forbidden(quill, "penbook/posts.py") == ()


def test_a_name_in_a_forbidden_ancestor_module_is_not_imported(tmp_path: Path) -> None:
    write(tmp_path, {"binstore/__init__.py": "",
                     "binstore/ledger.py": "def append_line(path, line):\n    pass\n"})
    code = "def receive(sku):\n    append_line('x', sku)\n"
    resolved = project_pipeline.resolve_imports(tmp_path, "binstore/service.py", code)
    assert resolved.added == ["from .ledger import append_line"]  # nothing forbids it
    resolved = project_pipeline.resolve_imports(tmp_path, "binstore/service.py", code,
                                                ("binstore.ledger",))
    assert resolved.added == []
    assert "binstore.ledger" in resolved.foreign["append_line"]
    assert "must not import binstore.ledger" in resolved.foreign["append_line"]


# -- the refusal ------------------------------------------------------------------------------------

def test_refusal_says_the_name_is_in_a_layer_the_file_must_not_import(tmp_path: Path) -> None:
    write(tmp_path, FLEET)
    relative = "truckyard/domain/planner.py"
    resolved = project_pipeline.resolve_imports(tmp_path, relative, PLANNER_ROUND2)
    assert resolved.added == []
    assert set(resolved.foreign) == {"status_row"}
    refusal = edits.check("pipeline", relative, None, resolved.content, tmp_path,
                          foreign=resolved.foreign).refusal
    assert refusal is not None
    # no task rule known: a sibling package; import it only if this layer may use it
    assert ("`status_row` is defined in truckyard.web.render, in truckyard.web, a sibling package "
            "of truckyard.domain") in refusal
    assert "import it yourself (`from truckyard.web.render import status_row`)" in refusal
    assert "define or import" not in refusal
    # with the task's rule known, the refusal quotes it
    forbidden = project_pipeline.forbidden_imports(FLEET_TASK, relative)
    resolved = project_pipeline.resolve_imports(tmp_path, relative, PLANNER_ROUND2, forbidden)
    refusal = edits.check("pipeline", relative, None, resolved.content, tmp_path,
                          foreign=resolved.foreign).refusal
    assert ("`status_row` is defined in truckyard.web.render, and the task says "
            "truckyard/domain/planner.py must not import truckyard.web: do not import it; remove "
            "the code that uses it") in refusal
    assert "import it yourself" not in refusal


def test_refusal_keeps_define_or_import_for_the_other_names(tmp_path: Path) -> None:
    write(tmp_path, FLEET)
    code = PLANNER_ROUND2 + "\n\ndef extra():\n    return helper_nobody_wrote()\n"
    resolved = project_pipeline.resolve_imports(tmp_path, "truckyard/domain/planner.py", code)
    refusal = edits.check("pipeline", "truckyard/domain/planner.py", None, resolved.content,
                          tmp_path, foreign=resolved.foreign).refusal
    assert "`status_row` is defined in truckyard.web.render, in truckyard.web" in refusal
    assert "for the other undefined names: define or import" in refusal


def test_switch_off_keeps_the_old_refusal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "project_layer_imports")
    write(tmp_path, FLEET)
    resolved = project_pipeline.resolve_imports(tmp_path, "truckyard/domain/planner.py",
                                                PLANNER_ROUND2)
    assert resolved.foreign == {}


def test_the_writer_step_returns_the_layering_refusal(tmp_path: Path) -> None:
    """Through _produce_file: the reply is refused before saving, with the layering advice."""
    write(tmp_path, {**FLEET, "truckyard/domain/planner.py": "class Planner:\n    pass\n"})
    agent = CodingAgent(tmp_path, allow_commands=False)
    reply = f"File: truckyard/domain/planner.py\n```python\n{PLANNER_ROUND2}```"

    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=reply), finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    item = {"path": "truckyard/domain/planner.py", "change": "take a repository"}
    produced = project_pipeline._produce_file(agent, FLEET_TASK, item, [item], "")
    assert produced.writes == {}
    assert "the task says truckyard/domain/planner.py must not import truckyard.web" \
        in produced.problem
    assert "define or import" not in produced.problem
