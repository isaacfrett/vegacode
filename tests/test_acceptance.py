"""acceptance: Vega checks the work against the task itself (failure diagnosis, redesign A).

Spec extraction on task texts, the checks on tiny fixture projects in the real sandbox,
and the agent loop / project pipeline wiring with scripted fake models. No model is run.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_agent_loop import agent_with, call, text

from vega import acceptance, features, sandbox
from vega.acceptance import Item, Result

ROOT = Path(__file__).resolve().parents[1]
needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")


@pytest.fixture(autouse=True)
def acceptance_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Acceptance is on (a shipped default). These tasks are short single-file writes scripted
    through the agent loop: the direct pipeline would take them, and with it off the routing table
    sends a one-target task with acceptance checks to the project pipeline, so both are off. The
    finish_gate's own re-send of a failing check is tested in test_finish_gate.py."""
    import os

    names = [n for n in os.environ.get(features.ENV, "").split(",") if n and n != "acceptance"]
    monkeypatch.setenv(features.ENV, ",".join([*names, "direct_pipeline", "project_pipeline",
                                               "finish_gate"]))


def switch_off(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    import os

    current = [n for n in os.environ.get(features.ENV, "").split(",") if n]
    monkeypatch.setenv(features.ENV, ",".join([*current, name]))


# -- extraction -------------------------------------------------------------------------------------

def test_library_names_and_mentions_are_not_required(tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text("from sklearn.ensemble import RandomForestClassifier\n")
    task = ("In `model.py` implement `fit(df)` using `RandomForestClassifier(n_estimators=10)`; "
            "never call `plt.show()` and return a `dict` with exactly the keys `a`, `b`.")
    spec = acceptance.extract(tmp_path, task)
    assert [r.attr for r in spec.names] == ["fit"]


# -- the checks, on tiny projects in the real sandbox --------------------------------------------------

def checks(tmp_path: Path, task: str, files: dict[str, str], changed: list[str] | None = None,
           **kwargs) -> Result:
    spec = acceptance.extract(tmp_path, task)  # before the "write", like Vega
    for rel, body in files.items():
        target = tmp_path / rel
        before = target.read_text() if target.exists() else None
        spec.remember(rel, before)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    return acceptance.run(spec, tmp_path, changed if changed is not None else list(files), **kwargs)


@needs_sandbox
def test_missing_function_is_reported(tmp_path: Path) -> None:
    (tmp_path / "geo.py").write_text("")
    result = checks(tmp_path, "Implement `area(w, h)` in `geo.py`.",
                    {"geo.py": "def perimeter(w, h):\n    return 2 * (w + h)\n"})
    assert [i.line() for i in result.failing] == [
        "`area` in geo.py: not defined (the task names it)"]


@needs_sandbox
def test_wrong_signature_is_reported(tmp_path: Path) -> None:
    (tmp_path / "geo.py").write_text("")
    result = checks(tmp_path, "Implement `area(w, h)` in `geo.py`.",
                    {"geo.py": "def area(width, h):\n    return width * h\n"})
    (item,) = result.failing
    assert item.kind == "signature"
    assert "parameters are (width, h); the task states (w, h): parameter 1 is `width`, not `w`" \
        in item.detail


@needs_sandbox
def test_crash_on_the_named_input_is_reported_with_the_project_line(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "orders.csv").write_text("id,amount\n1,2.5\n2,3\n")
    (tmp_path / "orders.py").write_text("")
    task = ("`data/orders.csv` has `id` and `amount`. Implement `total(path)` in `orders.py`: "
            "read the CSV and return a `dict` with exactly the keys `n` (int), `sum` (float).")
    result = checks(tmp_path, task, {"orders.py": (
        "import csv\n\n\ndef total(path):\n    rows = list(csv.DictReader(open(path)))\n"
        "    return {'n': len(rows), 'sum': sum(float(r['amt']) for r in rows)}\n")})
    (item,) = result.failing
    assert item.label == "total('data/orders.csv')"
    assert item.detail.startswith("raised KeyError: 'amt' (orders.py:6:")


@needs_sandbox
def test_stated_keys_and_plain_python_types_are_checked(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "orders.csv").write_text("id,amount\n1,2.5\n2,3\n")
    (tmp_path / "orders.py").write_text("")
    task = ("`data/orders.csv` has `id` and `amount`. Implement `total(path)` in `orders.py`: "
            "read the CSV and return a `dict` with exactly the keys `n` (int), `sum` (float).")
    result = checks(tmp_path, task, {"orders.py": (
        "def total(path):\n    return {'n': 2.0, 'sum': 5.5, 'extra': 1}\n")})
    (item,) = result.failing
    assert "keys differ from the task's (extra extra)" in item.detail
    assert "`n` is builtins.float, the task states int" in item.detail
    good = checks(tmp_path, task, {"orders.py": "def total(path):\n    return {'n': 2, 'sum': 5.5}\n"})
    assert good.ok and any(i.kind == "call" and i.detail == "returned dict with 2 key(s)"
                           for i in good.items)


@needs_sandbox
def test_a_named_mypy_strict_command_that_fails(tmp_path: Path) -> None:
    if sandbox.find_python_tool(tmp_path, "mypy") is None:
        pytest.skip("mypy is not installed")
    (tmp_path / "src" / "box").mkdir(parents=True)
    (tmp_path / "src" / "box" / "__init__.py").write_text("")
    (tmp_path / "src" / "box" / "core.py").write_text("")
    task = "Type `src/box/core.py`. It must pass `mypy --strict` with no errors."
    result = checks(tmp_path, task, {"src/box/core.py": "def double(x):\n    return x * 2\n"})
    (item,) = result.failing
    assert item.kind == "command" and item.label == "`mypy --strict src/box/core.py`"
    assert "exit code 1" in item.detail and "missing a type annotation" in item.detail
    typed = checks(tmp_path, task, {"src/box/core.py": "def double(x: int) -> int:\n    return x * 2\n"})
    assert typed.ok


@needs_sandbox
def test_static_gate_reports_new_undefined_names_only(tmp_path: Path) -> None:
    (tmp_path / "old.py").write_text("def f():\n    return missing_before\n")
    result = checks(tmp_path, "Fix the bug in old.py.", {
        "old.py": "def f():\n    return missing_before\n\n\ndef g():\n    return np.zeros(3)\n"})
    (item,) = result.failing
    assert item.kind == "static"
    assert item.detail == "line 6 F821 Undefined name `np`"  # the old one is not this turn's


def test_static_fallback_without_ruff(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "find_python_tool", lambda project, name: None)
    found = acceptance.static_findings(tmp_path, "m.py", (
        "import os\n\ndef f(x):\n    return os.sep + y\n\ndef f(x):\n    return x\n"))
    assert (4, "F821", "Undefined name `y`") in found
    assert any(code == "F811" and line == 6 for line, code, _ in found)


@needs_sandbox
def test_a_named_command_runs_and_fails(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("import sys\nfrom app import ok\nsys.exit(0 if ok() else 1)\n")
    (tmp_path / "app.py").write_text("def ok():\n    return False\n")
    result = checks(tmp_path, "Make `ok()` return True. Run the tests with `python check.py`.",
                    {"app.py": "def ok():\n    return False\n"})
    assert [i.label for i in result.failing] == ["`python check.py`"]


def test_do_not_run_commands_runs_in_the_sandbox_by_default(tmp_path: Path, monkeypatch) -> None:
    ran: list[str] = []
    monkeypatch.setattr(sandbox, "run_check",
                        lambda command, project, **kw: ran.append(command) or "exit code: 0\n")
    (tmp_path / "plan.py").write_text("def plan(spec):\n    return spec['a']\n")
    task = "Rewrite `plan` in `plan.py`. Keep the signature. Do not run commands."
    spec = acceptance.extract(tmp_path, task)
    assert spec.no_commands and spec.runs
    acceptance.run(spec, tmp_path, ["plan.py"])
    assert len(ran) == 1  # the import/name check, in the sandbox


def test_do_not_run_commands_setting_off_keeps_checks_static(tmp_path: Path, monkeypatch) -> None:
    switch_off(monkeypatch, "acceptance_ignores_no_commands")
    monkeypatch.setattr(sandbox, "run_check",
                        lambda *a, **k: pytest.fail("nothing may run in the sandbox"))
    (tmp_path / "plan.py").write_text("def plan(spec):\n    return spec['a']\n")
    task = "Rewrite `plan` in `plan.py`. Keep the signature. Do not run commands."
    spec = acceptance.extract(tmp_path, task)
    assert spec.no_commands and not spec.runs
    (tmp_path / "plan.py").write_text("def plan(spec, extra):\n    return spec['a']\n")
    result = acceptance.run(spec, tmp_path, ["plan.py"])
    assert [(i.kind, i.ok) for i in result.items] == [("static", True), ("name", True),
                                                     ("signature", False)]


def test_stoppable(tmp_path: Path, monkeypatch) -> None:
    stop = threading.Event()
    seen: list[object] = []
    monkeypatch.setattr(sandbox, "run_check",
                        lambda command, project, **kw: seen.append(kw.get("stop")) or
                        sandbox.STOPPED_CHECK)
    (tmp_path / "geo.py").write_text("def area(w, h):\n    return w * h\n")
    spec = acceptance.extract(tmp_path, "Implement `area(w, h)` in `geo.py`.")
    acceptance.run(spec, tmp_path, [], stop=stop)
    assert seen == [stop]


# -- agent loop: feedback goes to the model; the model decides ---------------------------------------

def fake_results(monkeypatch, results: list[Result]) -> list[dict]:
    """Script acceptance.run: each call returns the next result; the calls are recorded."""
    calls: list[dict] = []

    def run(spec, project, changed, **kwargs):
        calls.append({"changed": list(changed), **kwargs})
        return results.pop(0) if results else Result([Item("name", "`area` in geo.py", True)])

    monkeypatch.setattr(acceptance, "run", run)
    return calls


def failing(detail: str) -> Result:
    return Result([Item("name", "`area` in geo.py", True),
                   Item("call", "area('data/x.csv')", False, detail)])


TASK = "Implement `area(w, h)` in `geo.py`."
WRITE = call("write_file", path="geo.py", content="def area(w, h):\n    return w * h\n")


def test_after_a_write_one_line_of_failing_items_goes_to_the_model(tmp_path: Path,
                                                                   monkeypatch) -> None:
    (tmp_path / "geo.py").write_text("")
    calls = fake_results(monkeypatch, [failing("raised ZeroDivisionError (geo.py:2: w / 0)"),
                                       Result([Item("name", "`area` in geo.py", True)])])
    agent, seen = agent_with(tmp_path, [WRITE, text("Wrote area().")])
    assert agent.run(TASK) == "Wrote area()."
    tool_result = seen[1][-1]["content"]
    assert tool_result.endswith("(Vega acceptance) 1 of 2 task checks fail: area('data/x.csv') "
                                "raised ZeroDivisionError (geo.py:2: w / 0)")
    assert calls[0] == {"changed": ["geo.py"], "stop": agent._stop, "commands": False,
                        "tests": False}
    assert calls[1]["commands"] and calls[1]["tests"]  # the finish gate runs the full set


def test_finish_gate_sends_the_failures_back_and_the_model_decides(tmp_path: Path,
                                                                   monkeypatch) -> None:
    (tmp_path / "geo.py").write_text("")
    bad = failing("raised ZeroDivisionError (geo.py:2: w / 0)")
    fake_results(monkeypatch, [bad, bad, bad])
    agent, seen = agent_with(tmp_path, [WRITE, text("Done, verified."),
                                        text("area() still divides by zero on that input; "
                                             "left open.")])
    answer = agent.run(TASK)
    assert answer == "area() still divides by zero on that input; left open."  # the model's call
    note = seen[2][-1]["content"]
    assert note.startswith("(Vega harness) Before you finish: Vega checked your work against "
                           "the task in the sandbox; 1 of 2 check(s) fail:\n- area('data/x.csv'): "
                           "raised ZeroDivisionError")
    assert (tmp_path / "geo.py").read_text() == "def area(w, h):\n    return w * h\n"  # untouched


def test_the_same_failing_state_is_not_sent_twice_and_rounds_are_capped(tmp_path: Path,
                                                                        monkeypatch) -> None:
    (tmp_path / "geo.py").write_text("")
    # every write changes the failing state: 4 different states, 3 rounds at most
    states = [failing(f"raised Error{n}") for n in range(12)]
    fake_results(monkeypatch, states)
    rewrite = [call("write_file", path="geo.py", content=f"def area(w, h):\n    return {n}\n")
               for n in range(4)]
    replies = [rewrite[0], text("a"), rewrite[1], text("b"), rewrite[2], text("c"), rewrite[3],
               text("d")]
    agent, seen = agent_with(tmp_path, replies)
    assert agent.run(TASK) == "d"
    notes = [m["content"] for request in seen for m in request[-1:]
             if m["role"] == "user" and m["content"].startswith("(Vega harness) Before you finish")]
    assert len(set(notes)) == acceptance.FEEDBACK_ROUNDS == 3


def test_unchanged_state_is_not_resent(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "geo.py").write_text("")
    bad = failing("raised ZeroDivisionError")
    calls = fake_results(monkeypatch, [bad, bad])
    agent, _ = agent_with(tmp_path, [WRITE, text("first"), text("second")])
    assert agent.run(TASK) == "second"
    assert len(calls) == 2  # one after the write, one full run; not re-run without a new write


def test_claim_check_uses_the_acceptance_result_as_the_latest_evidence(tmp_path: Path,
                                                                       monkeypatch) -> None:
    (tmp_path / "geo.py").write_text("")
    bad = failing("raised ZeroDivisionError")
    fake_results(monkeypatch, [bad, bad])
    agent, seen = agent_with(tmp_path, [WRITE, text("one"), text("All tests pass, verified."),
                                        text("area() still fails on the input.")])
    assert agent.run(TASK) == "area() still fails on the input."
    claim = seen[3][-1]["content"]
    assert claim.startswith('(Vega harness) Your answer says "All tests pass", but the last run '
                            "(`vega acceptance`) failed:")
    assert "FAIL area('data/x.csv'): raised ZeroDivisionError" in claim


def test_switched_off_nothing_runs(tmp_path: Path, monkeypatch) -> None:
    switch_off(monkeypatch, "acceptance")
    (tmp_path / "geo.py").write_text("")
    monkeypatch.setattr(acceptance, "run", lambda *a, **k: pytest.fail("acceptance ran"))
    agent, seen = agent_with(tmp_path, [WRITE, text("Wrote area().")])
    assert agent.run(TASK) == "Wrote area()."
    assert "(Vega acceptance)" not in seen[1][-1]["content"]


@needs_sandbox
def test_real_sandbox_end_to_end_in_the_agent_loop(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.csv").write_text("a\n1\n")
    (tmp_path / "geo.py").write_text("")
    write = call("write_file", path="geo.py",
                 content="def load(path):\n    return open(path).read().split()[5]\n")
    agent, seen = agent_with(tmp_path, [write, text("Done."), text("load() fails on the input.")])
    task = "`data/x.csv` holds numbers. Implement `load(path)` in `geo.py`: read the file."
    assert agent.run(task) == "load() fails on the input."
    assert "load('data/x.csv') raised IndexError: list index out of range" in seen[1][-1]["content"]
    assert "load('data/x.csv'): raised IndexError" in seen[2][-1]["content"]


# -- project pipeline: visible tests passing is not enough -------------------------------------------

PLAN = json.dumps({"analysis": "Add total() to shop/cart.py.",
                   "files": [{"path": "shop/cart.py", "change": "add total(items)"}]})
RIGHT = "```python\ndef total(items):\n    return sum(price * qty for price, qty in items)\n```"


def pipeline_project(root: Path) -> None:
    (root / "shop").mkdir()
    (root / "shop" / "__init__.py").write_text("")
    (root / "shop" / "cart.py").write_text("def count(items):\n    return len(items)\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_cart.py").write_text(
        "from shop.cart import total\n\n\ndef test_total():\n    assert total([(2, 2)]) == 4\n")


def scripted(root: Path, replies: list[str]):
    from vega.agent import CodingAgent

    agent = CodingAgent(root, allow_commands=False)
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=replies.pop(0)))])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, calls


@needs_sandbox
def test_project_pipeline_keeps_repairing_while_acceptance_fails(tmp_path: Path,
                                                                 monkeypatch) -> None:
    # unchanged_rounds off: the faked acceptance result changes while the files do not
    monkeypatch.setenv(features.ENV, "question_pipeline,clarify_first,project_context,direct_pipeline,"
                       "unchanged_rounds")
    pipeline_project(tmp_path)
    bad = Result([Item("call", "total([])", False, "raised TypeError")])
    good = Result([Item("call", "total([])", True, "returned int")])
    calls = fake_results(monkeypatch, [bad, good])
    agent, model_calls = scripted(tmp_path, [PLAN, RIGHT, PLAN, RIGHT])
    answer = agent.run("Add `total(items)` to `shop/cart.py`: the sum of price times quantity.")
    assert answer.startswith("Changed shop/cart.py; the project's tests pass.")
    assert len(calls) == 2  # round 1: tests pass but acceptance fails, so round 2 ran
    repair = model_calls[2]["messages"][-1]["content"]
    assert "Vega's checks from the task's own words fail:\n- total([]): raised TypeError" in repair


@needs_sandbox
def test_project_pipeline_reports_what_still_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV, "question_pipeline,clarify_first,project_context,direct_pipeline")
    pipeline_project(tmp_path)
    fake_results(monkeypatch, [Result([Item("call", "total([])", False, "raised TypeError")])
                               for _ in range(4)])
    agent, _ = scripted(tmp_path, [PLAN, RIGHT, PLAN, RIGHT, PLAN, RIGHT, PLAN, RIGHT])
    answer = agent.run("Add `total(items)` to `shop/cart.py`: the sum of price times quantity.")
    assert answer == ("Changed shop/cart.py, but after 3 repair rounds these still fail:\n"
                      "- total([]): raised TypeError")


@needs_sandbox
def test_direct_pipeline_repairs_against_acceptance_failures(tmp_path: Path, monkeypatch) -> None:
    # unchanged_rounds off: the faked acceptance result changes while the files do not
    monkeypatch.setenv(features.ENV, "question_pipeline,clarify_first,project_context,"
                       "unchanged_rounds")
    (tmp_path / "geo.py").write_text("def area(w, h):\n    raise NotImplementedError\n")
    fake_results(monkeypatch, [Result([Item("signature", "`area` in geo.py", False,
                                            "parameters are (width, h)")]),
                               Result([Item("signature", "`area` in geo.py", True)])])
    code = "```python\ndef area(w, h):\n    return w * h\n```"
    agent, model_calls = scripted(tmp_path, [code, code])
    answer = agent.run("Implement `area(w, h)` in geo.py.")
    assert answer.startswith("Implemented the code in geo.py")
    feedback = model_calls[1]["messages"][-1]["content"]
    assert "- `area` in geo.py: parameters are (width, h)" in feedback


# -- acceptance_model (opt-in) -------------------------------------------------------------------------

def test_model_fills_the_spec_only_when_rules_find_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "acceptance_model")
    (tmp_path / "geo.py").write_text("")
    replies = [json.dumps({"names": [{"name": "area", "file": "geo.py", "params": "w, h"},
                                     {"name": "bogus", "file": "../x.txt"}],
                           "commands": ["rm -rf /", "python -m pytest -q"]})]
    agent, seen = agent_with(tmp_path, [])
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: seen.append(kw) or SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=replies.pop(0)))]))))
    spec = acceptance.for_turn(agent, "Make the geometry helper work for rectangles.")
    assert spec is not None and spec.summary()["names"] == ["area (geo.py)(w, h)"]
    assert spec.commands == ["python -m pytest -q"]  # only what run_check would run
    assert seen[0]["response_format"]["json_schema"]["name"] == "acceptance"
    spec = acceptance.for_turn(agent, "Implement `area(w, h)` in `geo.py`.")
    assert len(seen) == 1  # the rules found a name: no model call


def test_other_language_task_names_no_python_file(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("import subprocess\n")
    (tmp_path / "ticker.js").write_text("class Ticker {}\n")
    task = ("In `ticker.js`, `Ticker` loses `this` inside `tickAll`. Fix it. "
            "The check is `python check.py`.")
    assert [(r.path, r.attr) for r in acceptance.extract(tmp_path, task).names] == []


def test_a_script_the_task_only_runs_is_not_a_target(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("import shapes\n")
    task = "Add a function `area(width, height)`. Run `python check.py` to check."
    assert all(r.path != "check.py" for r in acceptance.extract(tmp_path, task).names)


# -- broad sample 2026-10-17: prose `f()` and stubs ---------------------------------------------------

def test_prose_call_with_empty_parens_is_not_a_zero_arg_signature(tmp_path: Path,
                                                                   monkeypatch) -> None:
    """A task saying "use it in `cashdesk/receipt.py`'s `total_line()`" was read as
    total_line() taking no parameters, failing a correct total_line(amounts)."""
    (tmp_path / "cashdesk").mkdir()
    (tmp_path / "cashdesk" / "__init__.py").write_text("")
    (tmp_path / "cashdesk" / "receipt.py").write_text(
        "def total_line(amounts: list[int]) -> str:\n    return str(sum(amounts))\n")
    task = ("Add a money formatter and use it in `cashdesk/receipt.py`'s `total_line()` so the "
            "total is formatted.")
    spec = acceptance.extract(tmp_path, task)
    named = {r.attr: r for r in spec.names}
    assert "total_line" in named and named["total_line"].params is None
    assert acceptance.extract(tmp_path, "Write `def reset()` in `cashdesk/receipt.py`."
                              ).names[0].params == []  # `def f()` still states no parameters
    (tmp_path / "cashdesk" / "app.py").write_text("def create_app(config=None):\n    return 1\n")
    kept = acceptance.extract(tmp_path, "Fix `create_app()` in `cashdesk/app.py`.").names[0]
    assert kept.params == []  # the code takes no required parameters: the statement stands
    switch_off(monkeypatch, "acceptance_prose_calls")
    off = {r.attr: r for r in acceptance.extract(tmp_path, task).names}
    assert off["total_line"].params == []


STUBBY = '''from abc import ABC, abstractmethod


def load_shapes(path):
    """Read the shapes."""
    raise NotImplementedError("load_shapes")


def todo_later(x):
    # TODO: implement
    ...


def fine(x):
    if x is None:
        raise NotImplementedError
    return x


def quiet(x):
    pass


class Shape(ABC):
    @abstractmethod
    def area(self):
        raise NotImplementedError


class Base:
    def size(self):
        raise NotImplementedError


class Box(Base):
    def size(self):
        return 1
'''


def test_stub_reason() -> None:
    import ast

    tree = ast.parse(STUBBY)
    lines = STUBBY.splitlines()
    reasons = {node.name: acceptance.stub_reason(node, lines) for node in tree.body
               if isinstance(node, ast.FunctionDef)}
    assert "NotImplementedError" in reasons["load_shapes"]
    assert "TODO" in reasons["todo_later"]
    assert reasons["fine"] is None  # raised only on one branch
    assert reasons["quiet"] is None  # a bare `pass` without a TODO may be intended
    shape = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Shape")
    assert acceptance.stub_reason(shape.body[0], lines) is None  # @abstractmethod


def test_a_task_named_stub_fails_acceptance(tmp_path: Path, monkeypatch) -> None:
    ran: list[str] = []
    monkeypatch.setattr(sandbox, "run_check",
                        lambda command, project, **kw: ran.append(command) or "exit code: 0\n")
    (tmp_path / "shapes.py").write_text(STUBBY)
    task = ("Implement `load_shapes(path)` in `shapes.py`, and keep `Base.size` and "
            "`Shape.area` as they are.")
    spec = acceptance.extract(tmp_path, task)
    result = acceptance.run(spec, tmp_path, ["shapes.py"])
    stubs = [i for i in result.failing if i.kind == "stub"]
    assert [i.label for i in stubs] == ["`load_shapes` in shapes.py"]
    assert "still a stub: it raises NotImplementedError" in stubs[0].line()
    assert "still a stub" in acceptance.finish_note(result)
    (tmp_path / "shapes.py").write_text(STUBBY.replace(
        '    raise NotImplementedError("load_shapes")', "    return []"))
    assert not [i for i in acceptance.run(spec, tmp_path, ["shapes.py"]).items if i.kind == "stub"]
    (tmp_path / "shapes.py").write_text(STUBBY)
    switch_off(monkeypatch, "acceptance_stubs")
    assert not [i for i in acceptance.run(spec, tmp_path, ["shapes.py"]).items if i.kind == "stub"]


def test_stub_check_holds_when_nothing_may_run(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "shapes.py").write_text(STUBBY)
    switch_off(monkeypatch, "acceptance_ignores_no_commands")
    spec = acceptance.extract(tmp_path, "Implement `load_shapes(path)` in `shapes.py`. Do not run "
                                        "commands.")
    assert not spec.runs
    assert any(i.kind == "stub" for i in acceptance.run(spec, tmp_path, ["shapes.py"]).failing)


def test_direct_tasks_get_no_smoke_call_without_input_files(tmp_path: Path) -> None:
    """Report item (c): `calls` is empty for direct tasks because _call_args only builds a call
    from input files the task names (paths / parsed JSON); a function of values gets none."""
    (tmp_path / "solution.py").write_text("def solution(lst):\n    raise NotImplementedError\n")
    spec = acceptance.extract(tmp_path, "Implement `solution(lst)` in solution.py.")
    assert spec.summary()["calls"] == [] and spec.inputs == []


# -- command_absent_inputs: an example input file the project does not have ------------------------

MODULE_ENTRY_COMMAND = "python -m wordfreq count notes.txt"


def test_a_command_that_reports_its_absent_example_input_passes(tmp_path: Path,
                                                                monkeypatch) -> None:
    import switches

    replies = {"missing": "exit code: 1\nwordfreq: notes.txt: No such file or directory\n",
               "no_main": "exit code: 1\n<venv>/bin/python: No module named wordfreq.__main__; "
                          "'wordfreq' is a package and cannot be directly executed\n",
               "usage": "exit code: 2\nusage: wordfreq [-h] {count,top} ...\n"
                        "wordfreq: error: unrecognized arguments: notes.txt\n"}
    reply = {"now": "missing"}
    monkeypatch.setattr(sandbox, "run_check", lambda command, project, **kw: replies[reply["now"]])
    absent = ["notes.txt"]
    item = acceptance._command_item(MODULE_ENTRY_COMMAND, tmp_path, None, "command", absent=absent)
    assert item.ok and "reports notes.txt missing" in item.detail
    for other in ("no_main", "usage"):  # anything but the missing example still fails
        reply["now"] = other
        assert not acceptance._command_item(MODULE_ENTRY_COMMAND, tmp_path, None, "command",
                                            absent=absent).ok
    reply["now"] = "missing"
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")  # it exists: its exit code counts
    assert not acceptance._command_item(MODULE_ENTRY_COMMAND, tmp_path, None, "command",
                                        absent=absent).ok
    (tmp_path / "notes.txt").unlink()
    switches.switch_off(monkeypatch, "command_absent_inputs")
    assert not acceptance._command_item(MODULE_ENTRY_COMMAND, tmp_path, None, "command",
                                        absent=absent).ok
