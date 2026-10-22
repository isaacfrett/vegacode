"""Deliverable pipeline: inputs in one prompt; files written directly or by a sandboxed script."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import deliverable_pipeline, features, sandbox
from vega.agent import CodingAgent

TASK = ("Rank the backlog in `backlog.csv` by value / effort, highest first, and write "
        "`ranking.json` as {\"ranking\": [ids]}.")
ONE_FORMAT_CHECK = "Wrote {}; 1 of 1 checks pass (1 format check)."
# json_template: the task states `ranking.json as {"ranking": ...}`, so its keys are checked too
WITH_TEMPLATE = "Wrote {}; 2 of 2 checks pass (1 format check, 1 template key)."
CSV = "id,value,effort\nA,10,5\nB,9,1\nC,5,2\n"  # 2, 9, 2.5
SCRIPT = ("```python\nimport csv, json\nrows = list(csv.DictReader(open('backlog.csv')))\n"
          "rows.sort(key=lambda r: -int(r['value']) / int(r['effort']))\n"
          "json.dump({'ranking': [r['id'] for r in rows]}, open('ranking.json', 'w'))\n```")


def scripted(root: Path, replies: list[str]) -> tuple[CodingAgent, list[dict]]:
    agent = CodingAgent(root, allow_commands=False)
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content=replies.pop(0), tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, calls


@pytest.fixture
def on(monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "deliverable_pipeline")


def test_targets_are_new_files_the_task_names(tmp_path: Path) -> None:
    (tmp_path / "backlog.csv").write_text(CSV)
    assert deliverable_pipeline.targets(tmp_path, TASK) == ["ranking.json"]  # backlog.csv exists


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_a_script_computes_the_deliverable_in_the_sandbox(tmp_path: Path, on) -> None:
    (tmp_path / "backlog.csv").write_text(CSV)
    agent, calls = scripted(tmp_path, [SCRIPT])
    assert agent.run(TASK) == WITH_TEMPLATE.format("ranking.json")
    assert json.loads((tmp_path / "ranking.json").read_text()) == {"ranking": ["B", "C", "A"]}
    assert (tmp_path / "backlog.csv").read_text() == CSV  # inputs untouched
    assert not (tmp_path / deliverable_pipeline.SCRIPT_DIR).exists()
    assert "id,value,effort" in calls[0]["messages"][-1]["content"]  # inputs in the prompt


def test_files_can_be_written_directly(tmp_path: Path, on) -> None:
    (tmp_path / "notes.txt").write_text("PR 12 adds dark mode (user-visible)\n")
    task = "Write `RELEASE_NOTES.md` with one bullet per user-visible change."
    agent, _ = scripted(tmp_path, ["File: RELEASE_NOTES.md\n```markdown\n- Dark mode (#12)\n```"])
    assert agent.run(task) == ONE_FORMAT_CHECK.format("RELEASE_NOTES.md")
    assert (tmp_path / "RELEASE_NOTES.md").read_text() == "- Dark mode (#12)\n"


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_a_failing_script_is_repaired(tmp_path: Path, on) -> None:
    (tmp_path / "backlog.csv").write_text(CSV)
    broken = "```python\nimport json\nraise ValueError('bad column effortt')\n```"
    agent, calls = scripted(tmp_path, [broken, SCRIPT])
    assert agent.run(TASK) == WITH_TEMPLATE.format("ranking.json")
    assert "bad column effortt" in calls[1]["messages"][-1]["content"]


def test_invalid_json_is_sent_back(tmp_path: Path, on) -> None:
    (tmp_path / "backlog.csv").write_text(CSV)
    bad = "File: ranking.json\n```json\n{ranking: [B]}\n```"
    good = 'File: ranking.json\n```json\n{"ranking": ["B", "C", "A"]}\n```'
    agent, calls = scripted(tmp_path, [bad, good])
    agent.run(TASK)
    assert "is not valid JSON" in calls[1]["messages"][-1]["content"]
    assert json.loads((tmp_path / "ranking.json").read_text())["ranking"][0] == "B"


# -- deliverable_wide (default on) ----------------------------------------------------------------


def _off(monkeypatch, name: str) -> None:
    import os

    names = [n for n in os.environ.get(features.ENV, "").split(",") if n]
    monkeypatch.setenv(features.ENV, ",".join([*names, name]))


def test_existing_inputs_are_not_targets(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("x")
    (tmp_path / "out.json").write_text("{}")
    task = ("Read `notes.md` (do not change `out.json`) and write `summary.json`. "
            "Then update `out.json`.")
    assert deliverable_pipeline.targets(tmp_path, task) == ["summary.json", "out.json"]
    assert deliverable_pipeline.targets(tmp_path, "Use `notes.md` to write `a.md`.") == ["a.md"]


def test_the_regenerated_file_is_written_and_an_unchanged_copy_is_sent_back(tmp_path: Path) -> None:
    (tmp_path / "deps.in").write_text("parselite==2.0.0\n")
    (tmp_path / "deps.lock").write_text("parselite==1.2.0\n")
    (tmp_path / "wheels").mkdir()
    (tmp_path / "wheels" / "parselite-2.0.0-py3-none-any.whl").write_bytes(b"PK\x03\x04\xff\xfe")
    task = "I bumped `parselite` in `deps.in`. Update `deps.lock` to match."
    same = "File: deps.lock\n```\nparselite==1.2.0\n```"
    new = "File: deps.lock\n```\nparselite==2.0.0\n```"
    agent, calls = scripted(tmp_path, [same, new])
    assert deliverable_pipeline.run(agent, task) == ONE_FORMAT_CHECK.format("deps.lock")
    assert (tmp_path / "deps.lock").read_text() == "parselite==2.0.0\n"
    prompt = calls[0]["messages"][-1]["content"]
    assert "deps.lock (the deliverable's CURRENT version" in prompt
    assert "wheels/parselite-2.0.0-py3-none-any.whl (binary, 6 bytes" in prompt
    assert "deps.lock is unchanged: write its new version" in calls[1]["messages"][-1]["content"]
    assert (tmp_path / "deps.in").read_text() == "parselite==2.0.0\n"  # inputs untouched


def test_large_inputs_are_shown_shorter_to_fit_the_window(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    for n in range(6):
        (tmp_path / "logs" / f"h{n}.log").write_text(
            "".join(f"2026-10-01T03:{i:02d}:00Z host{n} INFO request ok id={i}\n" for i in range(60)))
    whole = deliverable_pipeline.inputs(tmp_path, set())
    fitted = deliverable_pipeline.inputs(tmp_path, set(), budget=4000)
    assert len(whole) > 15000 and len(fitted) <= 4000
    assert "first 10 of 60 lines; read the whole file in a script" in fitted


def test_a_review_that_reports_lines_sees_line_numbers(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("import os\nx = os.system(input())\n")
    shown = deliverable_pipeline.inputs(tmp_path, set(), numbered=True)
    assert "   2| x = os.system(input())" in shown


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_the_writer_may_run_the_named_command_first(tmp_path: Path) -> None:
    (tmp_path / "tool.py").write_text("import sys\nprint('diff for', ' '.join(sys.argv[1:]))\n")
    task = "Run `python tool.py <args>` to see the change, then write `msg.txt` describing it."
    agent, calls = scripted(tmp_path, ["Run: python tool.py staged",
                                       "File: msg.txt\n```\nfix: the staged change\n```"])
    assert deliverable_pipeline.run(agent, task) == ONE_FORMAT_CHECK.format("msg.txt")
    assert "`python tool.py <args>`" in calls[0]["messages"][-1]["content"]
    assert "diff for staged" in calls[1]["messages"][-1]["content"]
    assert (tmp_path / "msg.txt").read_text() == "fix: the staged change\n"


# -- on write_loop (systems review #8) and the 2026-10-18 deliverable diagnosis ---------------------

needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")


def _script(body: str) -> str:
    return f"```python\n{body}\n```"


@needs_sandbox
def test_a_script_sees_a_private_home_not_the_users(tmp_path: Path, on) -> None:
    """A deliverable once held /Users/<name>/... from `expanduser('~')`."""
    (tmp_path / "backlog.csv").write_text(CSV)
    script = _script("import json, os\njson.dump({'ranking': [os.path.expanduser('~')]}, "
                     "open('ranking.json', 'w'))")
    agent, _ = scripted(tmp_path, [script])
    agent.run(TASK)
    assert str(Path.home()) not in (tmp_path / "ranking.json").read_text()


@needs_sandbox
def test_the_script_runs_from_the_root_and_the_target_folder_exists(tmp_path: Path, on) -> None:
    (tmp_path / "backlog.csv").write_text(CSV)
    task = "Count the rows of `backlog.csv` and write `answers/count.json` as {\"rows\": n}."
    script = _script("import json\nfrom pathlib import Path\nhere = Path(__file__).parent\n"
                     "rows = (here / 'backlog.csv').read_text().splitlines()[1:]\n"
                     "json.dump({'rows': len(rows)}, open(here / 'answers' / 'count.json', 'w'))")
    agent, _ = scripted(tmp_path, [script])
    assert agent.run(task).startswith("Wrote answers/count.json; 2 of 2 checks pass")
    assert json.loads((tmp_path / "answers" / "count.json").read_text()) == {"rows": 3}


@needs_sandbox
def test_a_script_wins_over_file_blocks_for_what_it_writes(tmp_path: Path, on) -> None:
    (tmp_path / "backlog.csv").write_text(CSV)
    guessed = 'File: ranking.json\n```json\n{"ranking": ["A", "B", "C"]}\n```'
    agent, _ = scripted(tmp_path, [SCRIPT + "\n\n" + guessed])
    agent.run(TASK)
    assert json.loads((tmp_path / "ranking.json").read_text()) == {"ranking": ["B", "C", "A"]}


def test_file_label_inside_the_fence_and_a_lone_block_for_one_target() -> None:
    inside = "Here it is:\n```text\nFile: msg.txt\nfix: the change\n```"
    assert deliverable_pipeline.file_blocks(inside, ["msg.txt"]) == (
        {"msg.txt": "fix: the change\n"}, [])
    lone = "```\nfeat: add retry\n```"
    assert deliverable_pipeline.file_blocks(lone, ["msg.txt"]) == (
        {"msg.txt": "feat: add retry\n"}, [])
    assert deliverable_pipeline.file_blocks(lone, ["a.txt", "b.txt"]) == ({}, [])


def test_an_empty_list_fails_when_the_task_asks_to_report_every_finding() -> None:
    task = "Review `app.py` and report every finding in `review.json` as a list."
    assert any("lists nothing" in p for p in
               deliverable_pipeline.check({"review.json": "[]"}, ["review.json"], task=task))
    assert deliverable_pipeline.check({"review.json": "[]"}, ["review.json"],
                                      task="Write `review.json`.") == []
    assert any("lists nothing" in p for p in deliverable_pipeline.check(
        {"review.json": '{"findings": []}'}, ["review.json"], task=task))


@needs_sandbox
def test_a_repeated_failure_is_named_and_the_line_quoted(tmp_path: Path, on) -> None:
    (tmp_path / "backlog.csv").write_text(CSV)
    broken = _script("import json\nx = 1\nraise ValueError('bad column effortt')")
    other = _script("import json\n\nraise ValueError('bad column effortt')")
    agent, calls = scripted(tmp_path, [broken, broken, other, SCRIPT])
    agent.run(TASK)
    assert "Line 3 of your script is: `raise ValueError('bad column effortt')`" in \
        calls[1]["messages"][-1]["content"]
    assert "same as last round's" in calls[2]["messages"][-1]["content"]
    assert "Same failure as last round (ValueError: bad column effortt)" in \
        calls[3]["messages"][-1]["content"]
    assert json.loads((tmp_path / "ranking.json").read_text())["ranking"] == ["B", "C", "A"]


@needs_sandbox
def test_every_round_failing_on_the_script_is_an_honest_stop(tmp_path: Path, on) -> None:
    """7 of 19 logged attempts fell back to the agent loop here; 6 then wrote made-up content."""
    (tmp_path / "backlog.csv").write_text(CSV)
    broken = _script("raise KeyError('effortt')")
    agent, calls = scripted(tmp_path, [broken] * (deliverable_pipeline.SCRIPT_REPAIRS + 1))
    answer = deliverable_pipeline.run(agent, TASK)
    assert len(calls) == deliverable_pipeline.SCRIPT_REPAIRS + 1  # 4 rounds on script errors
    assert answer.startswith("Not written: ranking.json.") and "KeyError: 'effortt'" in answer
    assert not (tmp_path / "ranking.json").exists()


def test_the_round_with_the_fewest_failing_checks_is_kept(tmp_path: Path, on) -> None:
    (tmp_path / "app.py").write_text("x = 1\n")
    task = "Review `app.py`: report every finding in `review.json` and a summary in `summary.md`."
    first = 'File: review.json\n```json\n["x is unused"]\n```'
    worse = 'File: review.json\n```json\n[]\n```'
    agent, calls = scripted(tmp_path, [first, worse, worse])
    answer = deliverable_pipeline.run(agent, task)
    assert len(calls) == deliverable_pipeline.REPAIRS + 1
    assert json.loads((tmp_path / "review.json").read_text()) == ["x is unused"]  # round 1
    assert answer.startswith("Wrote review.json, but only 1 of 2 checks pass")
    assert "summary.md was not produced" in answer and "Not written: summary.md" in answer
