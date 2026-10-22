"""issue_localize: the "Likely places" block for change tasks on large projects.

Without it the model often edited the wrong file, although the issue text named something the
right file holds verbatim.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from switches import switch_off
from test_agent_loop import agent_with, call, text

from vega import localize


def kinds(task: str) -> dict[str, str]:
    return {lit.text: lit.kind for lit in localize.literals(task) if lit.kind != "word"}


# -- literals ------------------------------------------------------------------------------------

def test_literals_of_an_issue() -> None:
    found = kinds('The save dialog says "Keep this draft?" but `save_draft` never shows '
                  "[y/N]; /help is missing /undo. Try --dry-run in cli.py. TodoStore.export_csv() "
                  "and CodeIndex too.")
    assert found["Keep this draft?"] == "quoted"
    assert found["save_draft"] == "ident"
    assert found["/help"] == found["/undo"] == found["--dry-run"] == "command"
    assert found["cli.py"] == "path"
    assert found["TodoStore.export_csv"] == found["export_csv"] == "ident"
    assert found["CodeIndex"] == "ident"
    assert "py" not in found  # cli.py is a file name, not a dotted name


def test_commands_to_run_and_closing_tags_are_not_literals() -> None:
    found = kinds("Run the tests with `python check.py` and `pytest -q`. The text form "
                  "<job_request><job_kind>sync_files</job_kind></job_request> fails; "
                  "see src/app/worker.py")
    assert "python check.py" not in found and "pytest -q" not in found
    assert "/job_kind" not in found and "/job_request" not in found
    assert found["sync_files"] == "ident" and found["src/app/worker.py"] == "path"
    assert "/app" not in found and "/worker" not in found


def test_fragments_of_a_built_message() -> None:
    message = "export error: Report folder is outside the workspace: q3.csv"
    assert localize._fragments(message) == ["Report folder is outside the workspace"]


# -- ranking -------------------------------------------------------------------------------------

def make(root: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root


def filler(count: int) -> dict[str, str]:
    """Unrelated modules that mention a common name, so the project is 'large'."""
    return {f"pkg/mod{i:02}.py": f"from pkg.table import merge_rows\n\n\ndef helper{i}():\n"
            f"    return merge_rows('x', 'a', 'b')\n" for i in range(count)}


def test_a_definition_and_a_rare_message_outrank_mentions(tmp_path: Path) -> None:
    make(tmp_path, {
        **filler(30),
        "pkg/table.py": "def merge_rows(path, left, right):\n"
                        "    if left not in path:\n"
                        "        return 'rows merged without keys'\n",
        "pkg/agent.py": "from pkg.table import merge_rows\n\ndef run():\n    merge_rows(1, 2, 3)\n",
    })
    places = localize.rank(tmp_path, "merge_rows breaks: it says \"rows merged without keys\" "
                                     "and drops right rows.")
    assert places[0].path == "pkg/table.py"
    assert any(reason == "defines `merge_rows`" for reason in places[0].reasons)


def test_test_files_and_docs_come_after_the_code(tmp_path: Path) -> None:
    make(tmp_path, {
        "app/report.py": "def render_total(rows):\n    return 'Total due'\n",
        "tests/test_report.py": "def test_total():\n    assert 'Total due' in render_total([])\n",
        "README.md": "The report ends with `Total due`.\n",
    })
    places = localize.rank(tmp_path, 'The report prints "Total due" twice. Fix render_total.')
    assert places[0].path == "app/report.py"


def test_a_named_target_beats_a_named_script_to_run(tmp_path: Path) -> None:
    make(tmp_path, {"src/ring.c": "int ring_push(void) { return 0; }\n",
                    "check.py": "# builds src/ring.c and runs it\nprint('ring_push')\n"})
    route = {"targets": [], "other_targets": ["src/ring.c"], "helpers": ["check.py"]}
    task = "Fix `src/ring.c` so ring_push works. Run `python check.py`; also see check.py."
    assert localize.rank(tmp_path, task, route=route)[0].path == "src/ring.c"


def test_ignored_folders_are_not_candidates(tmp_path: Path) -> None:
    make(tmp_path, {"node_modules/lib/x.js": "const load_config = 1;\n",
                    ".venv/lib/y.py": "load_config = 1\n",
                    "app.py": "def load_config():\n    pass\n"})
    assert [p.path for p in localize.rank(tmp_path, "load_config hangs")] == ["app.py"]


# -- the block -----------------------------------------------------------------------------------

def test_block_names_files_and_the_functions_around_the_matches(tmp_path: Path) -> None:
    make(tmp_path, {
        **filler(30),
        "pkg/table.py": "import os\n\n\ndef tool_specs():\n    return []\n\n\n"
                        "def merge_rows(path, left, right):\n"
                        "    if left not in path:\n"
                        "        return 'rows merged without keys'\n",
    })
    block = localize.block(tmp_path, 'merge_rows says "rows merged without keys" and breaks.')
    lines = block.splitlines()
    assert lines[0].startswith("Likely places")
    assert lines[1].startswith("1. pkg/table.py (")
    assert "   - def merge_rows (lines 8-10): lines 8, 10" in lines
    assert sum(line[:1].isdigit() for line in lines) == localize.TOP_FILES
    assert len(block) <= localize.MAX_CHARS


def test_no_block_for_a_small_project_or_word_only_evidence(tmp_path: Path) -> None:
    make(tmp_path, {"calc.py": "def add(a, b):\n    return a - b\n"})
    assert localize.block(tmp_path, "Fix add_numbers in calc.py") == ""
    make(tmp_path, filler(30))
    assert localize.block(tmp_path, "Something about the helpers is off.") == ""


def test_block_stays_within_its_budget(tmp_path: Path) -> None:
    long_names = {f"pkg/very/deeply/nested/module_number_{i:02}.py":
                  "".join(f"def function_with_a_long_name_{j}():\n    return 'Total due now'\n"
                          for j in range(6)) for i in range(30)}
    make(tmp_path, long_names)
    block = localize.block(tmp_path, 'It prints "Total due now" and function_with_a_long_name_1 '
                                     "and function_with_a_long_name_2 and function_with_a_long_n"
                                     "ame_3 break.")
    assert block and len(block) <= localize.MAX_CHARS


# -- the agent loop ------------------------------------------------------------------------------

@pytest.fixture
def agent_loop_only(monkeypatch: pytest.MonkeyPatch) -> None:
    switch_off(monkeypatch, "direct_pipeline", "deliverable_pipeline", "project_pipeline",
               "web_pipeline", "clarify_first", "finish_gate", "acceptance")


TASK = 'merge_rows breaks: it says "rows merged without keys". Fix it.'


def large_project(tmp_path: Path) -> Path:
    return make(tmp_path, {
        **filler(30),
        "pkg/table.py": "def merge_rows(path, left, right):\n"
                        "    return 'rows merged without keys'\n",
    })


def test_the_first_request_of_a_change_turn_carries_the_block(
        tmp_path: Path, agent_loop_only: None) -> None:
    agent, seen = agent_with(large_project(tmp_path), [call("read_file", path="pkg/table.py"),
                                                       text("Done.")])
    agent.run(TASK)
    goal = seen[0][-1]["content"]
    assert goal.startswith(TASK) and "\n\nLikely places" in goal
    assert "1. pkg/table.py" in goal


def test_no_block_when_switched_off_or_for_a_question(
        tmp_path: Path, agent_loop_only: None, monkeypatch: pytest.MonkeyPatch) -> None:
    project = large_project(tmp_path)
    agent, seen = agent_with(project, [text("It returns a message.")])
    agent.last_route = {"name": "agent", "change": False, "question": True, "reasons": []}
    assert agent._likely_places("What does merge_rows return?") == ""
    switch_off(monkeypatch, "issue_localize")
    agent, seen = agent_with(project, [call("read_file", path="pkg/table.py"), text("Done.")])
    agent.run(TASK)
    assert seen[0][-1]["content"] == TASK
