"""Redesign C (`pipeline_signals`): the pipelines trust the task, not proxies. Scripted models,
the real sandbox where tests run."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import (
    deliverable_pipeline,
    features,
    pipeline,
    project_pipeline,
    routing,
    sandbox,
    write_loop,
)
from vega.agent import CodingAgent

needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")


@pytest.fixture(autouse=True)
def defaults(monkeypatch) -> None:
    monkeypatch.delenv("VEGA_FEATURES_OFF", raising=False)  # every default on, pipeline_signals too


def signals_off(monkeypatch) -> None:
    monkeypatch.setenv(features.ENV, "pipeline_signals")


def scripted(root: Path, replies: list[str]) -> tuple[CodingAgent, list[dict], list[tuple]]:
    agent = CodingAgent(root, allow_commands=False)
    calls: list[dict] = []
    events: list[tuple] = []

    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content=replies.pop(0), tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    agent.on_event = lambda kind, data: events.append((kind, data))
    return agent, calls, events


def plan(*items: tuple[str, str]) -> str:
    return json.dumps({"analysis": "x", "files": [{"path": p, "change": c} for p, c in items]})


def make_shop(root: Path, tests: str) -> None:
    (root / "shop").mkdir()
    (root / "shop" / "__init__.py").write_text("")
    (root / "shop" / "cart.py").write_text("def count(items):\n    return len(items)\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_cart.py").write_text(tests)


COUNT_ONLY = "from shop.cart import count\n\n\ndef test_count():\n    assert count([(1, 2)]) == 1\n"
BOTH = ("from shop.cart import count, total\n\n\ndef test_count():\n    assert count([(1, 2)]) == 1\n"
        "\n\ndef test_total():\n    assert total([(2.5, 2), (1, 3)]) == 8\n")
CART_PLAN = plan(("shop/cart.py", "add total(items)"))
RIGHT = ("File: shop/cart.py\n```python\ndef total(items):\n"
         "    return sum(price * qty for price, qty in items)\n```")


# -- 1. stop condition and ranking ------------------------------------------------------------------

@needs_sandbox
def test_green_visible_tests_with_failing_acceptance_take_another_round(tmp_path: Path,
                                                                       monkeypatch) -> None:
    # one resolved target: routing would send this to the direct pipeline; this test is about
    # the project pipeline's stop condition, so it runs that pipeline itself (every default on)
    monkeypatch.setenv(features.ENV, "")
    make_shop(tmp_path, COUNT_ONLY)  # the visible tests pass before and after any change
    no_total = "File: shop/cart.py\n```python\ndef subtotal(items):\n    return 0\n```"
    agent, calls, _ = scripted(tmp_path, [CART_PLAN, no_total, CART_PLAN, RIGHT])
    answer = project_pipeline.run(
        agent, "Add `shop.cart.total(items)`: the sum of price times quantity.")
    assert answer.startswith("Changed shop/cart.py; the project's tests pass.")
    assert len(calls) == 4  # round 1 was green on pytest but not done
    assert "total" in calls[2]["messages"][-1]["content"]  # the failing check went to the replan
    assert "def total(items)" in (tmp_path / "shop" / "cart.py").read_text()


LOAD_ERROR = ("File: shop/cart.py\n```python\nimport sortedcontainers\n\n\ndef count(items):\n"
              "    return len(items)\n\n\ndef total(items):\n"
              "    return sum(price * qty for price, qty in items)\n```")
TWO_WRONG = ("File: shop/cart.py\n```python\ndef count(items):\n    return 0\n\n\n"
             "def total(items):\n    return 0\n```")


@needs_sandbox
def test_a_round_that_cannot_import_ranks_below_failing_tests(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(project_pipeline, "REPAIRS", 1)
    make_shop(tmp_path, BOTH)
    agent, _, _ = scripted(tmp_path, [CART_PLAN, LOAD_ERROR, CART_PLAN, TWO_WRONG])
    answer = agent.run("Add total(items) to the shop: the sum of price times quantity.")
    text = (tmp_path / "shop" / "cart.py").read_text()
    assert "sortedcontainers" not in text and "return 0" in text  # 2 failing beats 'cannot load'
    assert "test tests/test_cart.py::test_total" in answer  # the open items, listed


@needs_sandbox
def test_switch_off_keeps_the_round_with_the_lower_count(tmp_path: Path, monkeypatch) -> None:
    signals_off(monkeypatch)
    monkeypatch.setattr(project_pipeline, "REPAIRS", 1)
    make_shop(tmp_path, BOTH)
    agent, _, _ = scripted(tmp_path, [CART_PLAN, LOAD_ERROR.removeprefix("File: shop/cart.py\n"),
                                      CART_PLAN, TWO_WRONG.removeprefix("File: shop/cart.py\n")])
    agent.run("Add total(items) to the shop: the sum of price times quantity.")
    assert "sortedcontainers" in (tmp_path / "shop" / "cart.py").read_text()  # the old ranking


def test_load_errors_from_pytest_output() -> None:
    collection = ("exit code: 2\n_ ERROR collecting tests/test_cart.py _\nE   ModuleNotFoundError\n"
                  "ERROR tests/test_cart.py\n!!! Interrupted: 1 error during collection !!!")
    assert project_pipeline.load_errors(collection) == [
        "tests/test_cart.py does not load (import or syntax error at collection)"]
    config = "exit code: 4\nERROR: ./pyproject.toml: Expected '=' after a key (at line 2, column 3)"
    assert project_pipeline.load_errors(config) == [
        "pytest could not start: ./pyproject.toml: Expected '=' after a key (at line 2, column 3)"]
    assert project_pipeline.load_errors("exit code: 1\nFAILED tests/a.py::test_x - assert") == []


# -- 2. one file per reply ----------------------------------------------------------------------------

# the shape of a real reply for pyproject.toml: three files, the
# longest one cli.py's Python source, which the old rule wrote into pyproject.toml
MULTI = (
    "```python\n# pyproject.toml\n[build-system]\nrequires = [\"setuptools>=68\"]\n"
    "build-backend = \"setuptools.build_meta\"\n```\n\n"
    "**src/memocount/__init__.py**\n```python\n__version__ = \"1.4.0\"\n```\n\n"
    "File: src/memocount/cli.py\n```python\nimport sys\n\n\ndef main(argv=None):\n"
    "    args = sys.argv[1:] if argv is None else argv\n    print(len(args))\n    return 0\n\n\n"
    "if __name__ == \"__main__\":\n    raise SystemExit(main())\n```\n"
)
PKG_PLAN = [{"path": "pyproject.toml", "change": "setuptools backend, console script"},
            {"path": "src/memocount/__init__.py", "change": "version"},
            {"path": "src/memocount/cli.py", "change": "main returns the status"}]


def test_reply_blocks_read_each_label() -> None:
    labels = [label for _, label, _ in write_loop.reply_blocks(MULTI)]
    assert labels == ["pyproject.toml", "src/memocount/__init__.py", "src/memocount/cli.py"]
    paths = [p["path"] for p in PKG_PLAN]
    code, problem = write_loop.pick_block(MULTI, "pyproject.toml", paths)
    assert problem == "" and "[build-system]" in code and "def main" not in code
    code, _ = write_loop.pick_block(MULTI, "src/memocount/cli.py", paths)
    assert "def main" in code
    # unlabelled: the block in the file's language, not the longest one
    reply = "```python\nimport x\n" + "y = 1\n" * 20 + "```\n```toml\n[project]\nname = \"a\"\n```"
    code, _ = write_loop.pick_block(reply, "pyproject.toml", ["pyproject.toml"])
    assert code.startswith("[project]")


def test_labelled_multi_block_reply_writes_the_right_block(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"memocount\"\n")
    agent, calls, _ = scripted(tmp_path, [MULTI])
    result = project_pipeline._write_file(agent, "Fix the packaging.", PKG_PLAN[0], PKG_PLAN, "")
    assert result.startswith("Wrote pyproject.toml")
    written = (tmp_path / "pyproject.toml").read_text()
    assert written.startswith("[build-system]") and "def main" not in written
    prompt = calls[0]["messages"]
    assert "File: pkg/module.py" in prompt[0]["content"]  # the format is stated
    assert "Reply with pyproject.toml only: the line `File: pyproject.toml`" in prompt[-1]["content"]


def test_switch_off_takes_the_longest_block(tmp_path: Path, monkeypatch) -> None:
    signals_off(monkeypatch)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"memocount\"\n")
    agent, _, _ = scripted(tmp_path, [MULTI])
    project_pipeline._write_file(agent, "Fix the packaging.", PKG_PLAN[0], PKG_PLAN, "")
    assert "def main" in (tmp_path / "pyproject.toml").read_text()  # the old failure


ONLY_CLI = ("Here is the command line:\n\nFile: src/memocount/cli.py\n```python\n"
            "def main(argv=None):\n    return 0\n```")
TOML = "File: pyproject.toml\n```toml\n[project]\nname = \"memocount\"\nversion = \"1.4.0\"\n```"


def test_a_reply_for_another_file_is_sent_back_with_the_reason(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"memocount\"\n")
    agent, calls, events = scripted(tmp_path, [ONLY_CLI, TOML])
    result = project_pipeline._write_file(agent, "Fix the packaging.", PKG_PLAN[0], PKG_PLAN, "")
    assert result.startswith("Wrote pyproject.toml")
    feedback = calls[1]["messages"][-1]["content"]
    assert feedback.startswith("You sent code for src/memocount/cli.py; this step writes "
                               "pyproject.toml.")
    assert calls[1]["messages"][-2] == {"role": "assistant", "content": ONLY_CLI}
    assert 'version = "1.4.0"' in (tmp_path / "pyproject.toml").read_text()
    assert ("discarded", {"reason": "You sent code for src/memocount/cli.py; this step writes "
                                    "pyproject.toml.", "retry": True}) in events


def test_wrong_file_twice_leaves_the_file_unchanged(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"memocount\"\n")
    agent, _, _ = scripted(tmp_path, [ONLY_CLI, ONLY_CLI])
    result = project_pipeline._write_file(agent, "Fix the packaging.", PKG_PLAN[0], PKG_PLAN, "")
    assert result.startswith("Not written: You sent code for src/memocount/cli.py")
    assert (tmp_path / "pyproject.toml").read_text() == "[project]\nname = \"memocount\"\n"


def test_a_python_block_for_a_toml_file_is_sent_back(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"memocount\"\n")
    agent, calls, _ = scripted(tmp_path, ["```python\ndef available():\n    return []\n```", TOML])
    project_pipeline._write_file(agent, "Fix the packaging.", PKG_PLAN[0], PKG_PLAN, "")
    assert "block must be toml" in calls[1]["messages"][-1]["content"]


# -- 3. plan hygiene ----------------------------------------------------------------------------------

@pytest.mark.parametrize("change", [
    "No changes needed; only re-export existing modules if necessary.",
    "No changes required; the domain classes stay in models.py.",
    "No direct changes needed; this file is the source of the requirements.",
    "No code change, but ensure this file is included in the package data.",
    "Unchanged.",
    "Keep the existing version definition: `__version__ = \"1.4.0\"`.",
    "This file does not need to change.",
])
def test_no_change_items(change: str) -> None:
    assert project_pipeline.no_change(change)


@pytest.mark.parametrize("change", [
    "Add total(items) and keep count() unchanged.",
    "Keep the public API but rewrite parse() to stream lines.",
    "Implement delete(word) so counts are unchanged for a missing word.",
    "Move the planner into domain/planner.py.",
])
def test_change_items_are_kept(change: str) -> None:
    assert not project_pipeline.no_change(change)


def test_no_change_items_are_dropped_before_writing(tmp_path: Path, monkeypatch) -> None:
    reply = plan(("src/diskguard/__init__.py", "No changes needed; only re-export if necessary."),
                 ("src/diskguard/files.py", "Fix atomic_write so a failure leaves no temp file."))
    agent, _, _ = scripted(tmp_path, [reply, reply])
    assert [p["path"] for p in project_pipeline._plan(agent, "Fix atomic_write", "map")] == [
        "src/diskguard/files.py"]
    signals_off(monkeypatch)
    assert len(project_pipeline._plan(agent, "Fix atomic_write", "map")) == 2


@needs_sandbox
def test_an_empty_plan_is_retried_with_the_reason(tmp_path: Path) -> None:
    make_shop(tmp_path, BOTH)
    empty = json.dumps({"analysis": "", "files": []})
    agent, calls, _ = scripted(tmp_path, [empty, CART_PLAN, RIGHT])
    answer = project_pipeline.run(agent, "Add total(items) to the shop: price times quantity.")
    assert answer.startswith("Changed shop/cart.py; the project's tests pass.")
    assert "Your plan listed no files." in calls[1]["messages"][-1]["content"]


def test_an_empty_plan_twice_falls_back_with_a_notice(tmp_path: Path, monkeypatch) -> None:
    make_shop(tmp_path, BOTH)
    nothing = plan(("shop/cart.py", "No change required."))
    agent, calls, events = scripted(tmp_path, [nothing, nothing])
    assert project_pipeline.run(agent, "Add total(items) to the shop.") is None
    assert "Your plan listed only files that need no change: shop/cart.py (No change required.)" \
        in calls[1]["messages"][-1]["content"]
    notices = [data["message"] for kind, data in events if kind == "notice"]
    assert any("listed no file to change (asked twice); using the agent loop" in n for n in notices)
    signals_off(monkeypatch)  # the old behaviour: one plan call, silent fallback
    agent, calls, events = scripted(tmp_path, [json.dumps({"analysis": "", "files": []})])
    assert project_pipeline.run(agent, "Add total(items) to the shop.") is None
    assert len(calls) == 1


# -- 4. atomic pipeline writes --------------------------------------------------------------------------

BROKEN = ("File: shop/cart.py\n```python\ndef total(items):\n"
          "    return sum(price * qty for price, qty in items\n```")


@needs_sandbox
def test_a_refused_pipeline_write_goes_back_to_the_writer(tmp_path: Path) -> None:
    make_shop(tmp_path, BOTH)
    agent, calls, _ = scripted(tmp_path, [CART_PLAN, BROKEN, CART_PLAN, RIGHT])
    answer = agent.run("Add total(items) to the shop: the sum of price times quantity.")
    assert answer.startswith("Changed shop/cart.py; the project's tests pass.")
    writer = calls[3]["messages"][-1]["content"]
    assert "Your last reply for shop/cart.py was not saved:\nWrite not applied: shop/cart.py " \
           "would not parse" in writer
    assert "Reply again with the corrected code for shop/cart.py" in writer
    assert "These planned writes were not saved" in calls[2]["messages"][-1]["content"]


@needs_sandbox
def test_a_refused_write_never_lands_even_after_the_last_round(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(project_pipeline, "REPAIRS", 0)
    make_shop(tmp_path, COUNT_ONLY)  # green tests: the old stop rule would call this done
    agent, _, _ = scripted(tmp_path, [CART_PLAN, BROKEN])
    answer = agent.run("Add total(items) to the shop: the sum of price times quantity.")
    assert (tmp_path / "shop" / "cart.py").read_text() == "def count(items):\n    return len(items)\n"
    assert "shop/cart.py: not saved — Write not applied" in answer


DIRECT_STUB = '''def solution(lst):
    """Return the sum of the odd elements that are in even positions.

    Examples
    solution([5, 8, 7, 1]) ==> 12
    """
'''


@needs_sandbox
def test_direct_pipeline_feeds_a_refused_write_back(tmp_path: Path) -> None:
    (tmp_path / "solution.py").write_text(DIRECT_STUB)
    undefined = "```python\ndef solution(lst):\n    return helper(lst)\n```"
    right = ("```python\ndef solution(lst):\n"
             "    return sum(x for i, x in enumerate(lst) if i % 2 == 0 and x % 2 == 1)\n```")
    agent, calls, _ = scripted(tmp_path, [undefined, right])
    task = "Implement solution() in solution.py as its docstring says."
    answer = pipeline.run(agent, task, route=routing.route_task(task, tmp_path))
    assert answer.startswith("Implemented the code in solution.py; ")
    feedback = calls[1]["messages"][-1]["content"]
    assert "That code was not saved: Write not applied: it would add undefined name(s) helper" \
        in feedback
    assert "helper" not in (tmp_path / "solution.py").read_text()


DELIVERABLE_TASK = ("Rank the backlog in `backlog.csv` by value / effort, highest first, and "
                    "write `ranking.json` as {\"ranking\": [ids]}.")
BAD_JSON = "File: ranking.json\n```json\n{ranking: [B]}\n```"


def test_deliverable_refusal_is_fed_back_and_never_written(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "deliverable_pipeline")
    (tmp_path / "backlog.csv").write_text("id,value,effort\nA,10,5\nB,9,1\n")
    agent, calls, events = scripted(tmp_path, [BAD_JSON] * 3)
    assert deliverable_pipeline.run(agent, DELIVERABLE_TASK) is None
    assert "Write not applied: ranking.json would not parse (invalid JSON at line 1" in \
        calls[1]["messages"][-1]["content"]
    assert not (tmp_path / "ranking.json").exists()
    assert any(kind == "notice" and "would parse" in data["message"] for kind, data in events)
    signals_off(monkeypatch)  # the old behaviour wrote it
    agent, _, _ = scripted(tmp_path, [BAD_JSON] * 3)
    deliverable_pipeline.run(agent, DELIVERABLE_TASK)
    assert (tmp_path / "ranking.json").read_text().startswith("{ranking")


# -- 5. merge_code keeps module-level assignments ------------------------------------------------------

ORIGINAL = ("from typing import Generic\n\nLIMIT = 5\n\n\ndef a():\n    return LIMIT\n\n\n"
            "class Cache:\n    def get(self):\n        return 1\n")
REPLY = ("from typing import Generic, TypeVar\n\nK = TypeVar(\"K\")\nLIMIT = 10\n\n\n"
         "class Cache(Generic[K]):\n    def get(self, key: K):\n        return 2\n")


def test_merge_code_keeps_typevars_and_constants() -> None:
    merged = pipeline.merge_code(ORIGINAL, REPLY)
    assert "K = TypeVar(\"K\")" in merged and merged.index("K = TypeVar") < merged.index("class Cache")
    assert "LIMIT = 10" in merged and "LIMIT = 5" not in merged
    assert "def a():\n    return LIMIT" in merged
    compile(merged, "m.py", "exec")


def test_merge_code_switch_off_drops_them(monkeypatch) -> None:
    signals_off(monkeypatch)
    merged = pipeline.merge_code(ORIGINAL, REPLY)
    assert "TypeVar(\"K\")" not in merged and "LIMIT = 5" in merged
