"""Fixes from the Granite 4.1 8B baseline (2026-10-15/16): data inputs, plan_pending_ok,
data_preview and empty_final. Each is a feature switch with a switched-off test, except data
inputs: its switch went with the old routing (2026-10-17); routing.route_task classifies them."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_agent_loop import agent_with, call, text

from vega import features, routing
from vega.agent import EMPTY_FINAL, CodingAgent
from vega.data_preview import is_data_path
from vega.tools import execute, validate_plan, validate_plan_started


def switch_off(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    import os

    current = [n for n in os.environ.get(features.ENV, "").split(",") if n]
    monkeypatch.setenv(features.ENV, ",".join([*current, name]))


# -- data inputs: data files the task reads do not make the direct pipeline decline ---------------

REPORT = 'def total(path):\n    """Sum of the amount column."""\n    raise NotImplementedError\n'


@pytest.fixture
def data_project(tmp_path: Path) -> Path:
    (tmp_path / "report.py").write_text(REPORT)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sales.csv").write_text("id,amount\n1,2.5\n2,3.0\n")
    return tmp_path


ROUTES = [
    # (task, route, read-only inputs, other files to change)
    (("Implement `total(path)` in `report.py`: `data/sales.csv` has `id` and `amount`; return the "
     "sum of amount."), "direct", ["data/sales.csv"], []),
    ("In report.py implement total(path), which reads data/sales.csv and sums amount.",
     "direct", ["data/sales.csv"], []),
    ("Write `total(path)` in report.py that loads `data/sales.csv`.", "direct",
     ["data/sales.csv"], []),
    # the data file is to be changed or produced: not an input, not the direct pipeline's job
    ("Update data/sales.csv with a third row and implement total() in report.py.", "agent", [],
     ["data/sales.csv"]),
    ("Implement total() in report.py and write the result to data/summary.json.", "agent", [],
     ["data/summary.json"]),
    ("Implement total() in report.py; data/sales.csv must be corrected first.", "agent", [],
     ["data/sales.csv"]),
    ("Implement total() in report.py. data/sales.csv has a typo in row 2.", "agent", [],
     ["data/sales.csv"]),
    # no data file named
    ("Implement total() in report.py.", "direct", [], []),
]


@pytest.mark.parametrize(("task", "name", "inputs", "others"), ROUTES)
def test_routing_with_data_inputs(data_project: Path, task, name, inputs, others) -> None:
    route = routing.route_task(task, data_project)
    assert (route.name, route.targets, route.inputs, route.other_targets) == (
        name, ["report.py"], inputs, others)


def test_data_inputs_need_one_clear_code_target(data_project: Path) -> None:
    (data_project / "other.py").write_text("def f():\n    pass\n")
    task = "Implement total() in report.py and f() in other.py from data/sales.csv."
    assert routing.route_task(task, data_project).name != "direct"


def test_data_paths() -> None:
    assert is_data_path("data/sales.csv") and is_data_path("x.jsonl") and is_data_path("db.sqlite")
    assert is_data_path("fixtures/case.cfg") and is_data_path("samples/a.md")
    assert not is_data_path("data/loader.py") and not is_data_path("README.md")
    assert not is_data_path("pyproject.toml")


def test_the_writer_is_shown_the_input_data(data_project: Path, monkeypatch) -> None:
    replies = [("```python\nimport csv\n\n\ndef total(path):\n    \"\"\"Sum of the amount column."
               "\"\"\"\n    with open(path) as f:\n        return sum(float(r['amount']) for r in "
               "csv.DictReader(f))\n```")]
    seen: list[list[dict]] = []
    agent = CodingAgent(data_project, allow_commands=False)

    def create(**kwargs):
        seen.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=replies.pop(0)))])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer = agent.run("Implement `total(path)` in `report.py`: `data/sales.csv` has `id` and "
                       "`amount`; return the sum of amount.")
    assert "report.py" in answer and len(seen) == 1
    prompt = seen[0][-1]["content"]
    assert "Input data the task names" in prompt and "id,amount\n1,2.5" in prompt
    assert (data_project / "data" / "sales.csv").read_text() == "id,amount\n1,2.5\n2,3.0\n"


# -- plan_pending_ok: an all-pending plan starts its first item -----------------------------------

ALL_PENDING = [{"text": "Read the data", "status": "pending"},
               {"text": "Write the function", "status": "pending"}]


def test_all_pending_plan_starts_the_first_item(monkeypatch) -> None:
    items, started = validate_plan_started(ALL_PENDING)
    assert started == 1 and [i["status"] for i in items] == ["in_progress", "pending"]
    done_first = [dict(ALL_PENDING[0], status="done"), ALL_PENDING[1]]
    items, started = validate_plan_started(done_first)
    assert started == 2 and items[1]["status"] == "in_progress"
    two = [dict(item, status="in_progress") for item in ALL_PENDING]
    assert "2 items are in_progress" in validate_plan(two)  # still rejected
    switch_off(monkeypatch, "plan_pending_ok")
    assert "no item is in_progress" in validate_plan(ALL_PENDING)


def test_update_plan_result_says_it_started_the_item(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, seen = agent_with(tmp_path, [
        call("update_plan", items=ALL_PENDING), text("Done."), text("Done."), text("Done."),
    ])
    agent.run("Fix the loader bug in a.py")
    result = seen[1][-1]["content"]
    assert result.startswith("Plan saved; started item 1 (no item was in_progress).")
    assert "[>] Read the data" in result
    assert agent.plan[0]["status"] == "in_progress"


# -- data_preview: a large data file is previewed; ranges and code files are untouched ------------

def write_csv(path: Path, rows: int = 100) -> None:
    lines = ["order_id,amount,placed,status"]
    lines += [f"O{i:04d},{i * 1.5 if i % 7 else ''},2026-01-{i % 28 + 1:02d},paid" for i in
              range(rows)]
    path.write_text("\n".join(lines) + "\n")


def read(project: Path, **args) -> str:
    return execute("read_file", args, project)


def test_large_csv_gets_a_structured_preview(tmp_path: Path) -> None:
    write_csv(tmp_path / "orders.csv")
    result = read(tmp_path, path="orders.csv")
    assert result.startswith("Preview of orders.csv (") and "101 lines" in result
    assert "order_id: str" in result and "amount: float, " in result and "blank" in result
    assert "placed: date" in result
    assert "  21\tO0019," in result and "  22\t" not in result  # header + first 20 rows
    assert "read_file with start_line/end_line" in result


def test_explicit_range_returns_exactly_those_lines(tmp_path: Path) -> None:
    write_csv(tmp_path / "orders.csv")
    result = read(tmp_path, path="orders.csv", start_line=30, end_line=32)
    lines = (tmp_path / "orders.csv").read_text().splitlines()
    assert result == "\n".join(f"{n:>4}\t{lines[n - 1]}" for n in (30, 31, 32)) + \
        "\n[lines 30-32 of 101]"


def test_small_data_and_code_files_are_read_whole(tmp_path: Path, monkeypatch) -> None:
    write_csv(tmp_path / "small.csv", rows=5)
    assert read(tmp_path, path="small.csv").startswith("   1\torder_id,amount")
    (tmp_path / "big.py").write_text("".join(f"x{i} = {i}\n" for i in range(500)))
    assert read(tmp_path, path="big.py").startswith("   1\tx0 = 0")
    write_csv(tmp_path / "orders.csv")
    switch_off(monkeypatch, "data_preview")
    assert read(tmp_path, path="orders.csv").startswith("   1\torder_id,amount")


def test_json_and_jsonl_previews(tmp_path: Path) -> None:
    records = [{"id": i, "name": f"n{i}", "tags": ["a"] * (i % 3)} for i in range(200)]
    (tmp_path / "items.json").write_text(json.dumps(records, indent=1))
    result = read(tmp_path, path="items.json")
    assert "Top level: list of 200 items" in result and "Keys seen" in result
    assert "id, name, tags" in result and '[0] {"id": 0' in result
    (tmp_path / "config.json").write_text(json.dumps({"rows": records, "version": 3}, indent=1))
    result = read(tmp_path, path="config.json")
    assert "Top level: object with 2 keys" in result and "rows: list of 200 items" in result
    (tmp_path / "log.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    result = read(tmp_path, path="log.jsonl")
    assert "200 lines" in result and "Keys seen" in result and "First 20 lines" in result


def test_tsv_and_sqlite_previews(tmp_path: Path) -> None:
    (tmp_path / "a.tsv").write_text("k\tv\n" + "".join(f"{i}\tx{i}\n" for i in range(80)))
    result = read(tmp_path, path="a.tsv")
    assert "k: int" in result and "v: str" in result
    db = tmp_path / "shop.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE orders (id INTEGER, amount REAL)")
        connection.executemany("INSERT INTO orders VALUES (?, ?)", [(i, i / 2) for i in range(9)])
    connection.close()
    result = read(tmp_path, path="shop.db")
    assert "Table orders (9 rows): id INTEGER, amount REAL" in result


# -- empty_final: an empty final reply is sent back once ------------------------------------------
# The task is a question the question pipeline would answer; these test the agent loop's reply.

def test_empty_final_reply_is_sent_back_once(tmp_path: Path, monkeypatch) -> None:
    switch_off(monkeypatch, "question_pipeline")
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, seen = agent_with(tmp_path, [
        call("read_file", path="a.py"), text(""), text("a.py sets x to 1."),
    ])
    assert agent.run("What does a.py set?") == "a.py sets x to 1."
    assert seen[2][-1] == {"role": "user", "content": EMPTY_FINAL}


def test_a_second_empty_reply_ends_as_before(tmp_path: Path, monkeypatch) -> None:
    switch_off(monkeypatch, "question_pipeline")
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, seen = agent_with(tmp_path, [call("read_file", path="a.py"), text(""), text("")])
    assert agent.run("What does a.py set?") == "(The model returned no text.)"
    assert len(seen) == 3


def test_empty_final_switched_off(tmp_path: Path, monkeypatch) -> None:
    switch_off(monkeypatch, "empty_final")
    switch_off(monkeypatch, "question_pipeline")
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, seen = agent_with(tmp_path, [call("read_file", path="a.py"), text("")])
    assert agent.run("What does a.py set?") == "(The model returned no text.)"
    assert len(seen) == 2
