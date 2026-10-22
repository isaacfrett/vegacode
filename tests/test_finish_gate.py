"""finish_gate: the agent loop does not accept a final answer while the task is visibly undone.

Seen in practice: a run finished without the test file it was told to create, another with the
function it was told to implement still raising NotImplementedError, another after editing a file
outside "do not change any file except review.json", and another with its changes only in the
answer text. Scripted fake models only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from switches import switch_off
from test_agent_loop import agent_with, call, text

from vega import acceptance, features
from vega.acceptance import Item, Result
from vega.agent import FINISH_GATE_ROUNDS, FinishGate, unwritten_code

ROOT = Path(__file__).resolve().parents[1]
GATE = "(Vega harness) Not finished"


@pytest.fixture(autouse=True)
def agent_loop_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tasks are scripted through the agent loop, so the pipelines that would take them are
    off: direct and deliverable, the project pipeline, which the routing table picks for a
    one-target task with acceptance checks once the direct pipeline is off, and web_pipeline
    (a JS-only change such as "Convert `lib/text.js` to ES modules")."""
    import os

    names = [n for n in os.environ.get(features.ENV, "").split(",") if n]
    monkeypatch.setenv(features.ENV, ",".join([*names, "direct_pipeline", "deliverable_pipeline",
                                               "project_pipeline", "web_pipeline"]))


def notes(seen: list[list[dict]]) -> list[str]:
    return [request[-1]["content"] for request in seen
            if request[-1]["role"] == "user" and request[-1]["content"].startswith(GATE)]


# -- what the gate reads from the task text ----------------------------------------------------

def test_fixing_a_file_that_does_not_exist_is_not_a_file_to_create(tmp_path: Path) -> None:
    gate = FinishGate.for_task("fix the off-by-one in pager.py", tmp_path, None)
    assert gate.outputs == [] and not gate.scoped


def test_code_shown_in_the_answer_but_not_written() -> None:
    answer = ("### Proposed Changes\n\n1. **lib/text.js**\n\n```javascript\n// lib/text.js\n"
              "function slugify(t) {\n  return t.toLowerCase();\n}\nexport { slugify };\n```\n")
    assert unwritten_code(answer, ROOT / "tests") == []  # no such file there

    def check(root: Path, content: str) -> list[str]:
        (root / "lib").mkdir(exist_ok=True)
        (root / "lib" / "text.js").write_text(content)
        return unwritten_code(answer, root)

    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        assert check(root, "function slugify(t) {\n  return t;\n}\nmodule.exports = {slugify};\n") \
            == ["lib/text.js"]
        assert check(root, "function slugify(t) {\n  return t.toLowerCase();\n}\n"
                           "export { slugify };\n") == []


# -- in the agent loop -------------------------------------------------------------------------

R_TASK = ("Create `R/clamp_range.R` defining `clamp_range(x)`. Also add a testthat file "
          "`tests/testthat/test-clamp_range.R` with two expectations.")


def test_a_missing_file_sends_the_answer_back_with_its_name(tmp_path: Path) -> None:
    agent, seen = agent_with(tmp_path, [
        call("write_file", path="R/clamp_range.R", content="clamp_range <- function(x) x\n"),
        text("Done: clamp_range() is in R/clamp_range.R."),
        call("write_file", path="tests/testthat/test-clamp_range.R",
             content="test_that('w', { expect_equal(clamp_range(1), 1) })\n"),
        text("Done: both files written."),
    ])
    assert agent.run(R_TASK) == "Done: both files written."
    [note] = notes(seen)
    assert "- `tests/testthat/test-clamp_range.R` does not exist: the task says to write it." in note
    assert "R/clamp_range.R" not in note.split("\n", 1)[1]  # written: not listed


def test_at_most_two_rounds_per_turn(tmp_path: Path) -> None:
    agent, seen = agent_with(tmp_path, [
        call("list_files"), text("one"), text("two"), text("three"),
    ])
    assert agent.run(R_TASK) == "three"
    assert len(notes(seen)) == FINISH_GATE_ROUNDS == 2


def test_a_stub_left_in_place_is_sent_back(tmp_path: Path) -> None:
    (tmp_path / "geo.py").write_text("def area(w, h):\n    raise NotImplementedError\n")
    agent, seen = agent_with(tmp_path, [
        call("read_file", path="geo.py"),
        text("area() is ready."),
        call("write_file", path="geo.py", content="def area(w, h):\n    return w * h\n"),
        text("Implemented area()."),
    ])
    assert agent.run("Implement `area(w, h)` in `geo.py`.") == "Implemented area()."
    [note] = notes(seen)
    assert "`area` in `geo.py` is still a stub" in note


def test_a_change_outside_change_only_is_sent_back(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    # write_guard: this scripted model overwrites a fixture file it never read; the test is about
    # the change-only send-back, not the overwrite hold (tests/test_write_guard.py)
    switch_off(monkeypatch, "write_guard")
    (tmp_path / "a.py").write_text("X = 1\n")
    (tmp_path / "b.py").write_text("Y = 2\n")
    agent, seen = agent_with(tmp_path, [
        call("write_file", path="a.py", content="X = 10\n"),
        call("write_file", path="b.py", content="Y = 20\n"),
        text("Changed X and Y."),
        call("write_file", path="b.py", content="Y = 2\n"),
        text("Changed X; b.py restored."),
    ])
    assert agent.run("Set X to 10. Change only `a.py`.") == "Changed X; b.py restored."
    [note] = notes(seen)
    assert "`b.py` was changed, but the task says not to change it (allowed: `a.py`)" in note
    assert "Y = 2" in note  # the original, to write back


def test_code_only_in_the_answer_is_sent_back(tmp_path: Path) -> None:
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "text.js").write_text("function slugify(t) {\n  return t;\n}\n"
                                              "module.exports = { slugify };\n")
    esm = "function slugify(t) {\n  return t;\n}\nexport { slugify };\n"
    agent, seen = agent_with(tmp_path, [
        call("read_file", path="lib/text.js"),
        text(f"### Proposed Changes\n\n**lib/text.js**\n```javascript\n{esm}```"),
        call("write_file", path="lib/text.js", content=esm),
        text("Converted lib/text.js."),
    ])
    assert agent.run("Convert `lib/text.js` to ES modules.") == "Converted lib/text.js."
    [note] = notes(seen)
    assert "Your answer shows code for `lib/text.js` that is not in the file" in note


def _failing(detail: str) -> Result:
    return Result([Item("name", "`area` in geo.py", True),
                   Item("call", "area('data/x.csv')", False, detail)])


def test_a_failing_acceptance_check_is_sent_back_once_more(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV, monkeypatch_env_without("acceptance"))
    (tmp_path / "geo.py").write_text("")
    bad = _failing("raised ZeroDivisionError")
    results = [bad, bad, bad, bad]
    monkeypatch.setattr(acceptance, "run", lambda *a, **k: results.pop(0) if results else bad)
    agent, seen = agent_with(tmp_path, [
        call("write_file", path="geo.py", content="def area(w, h):\n    return w / 0\n"),
        text("Done."),  # the acceptance gate shows the failing check
        text("Done, really."),  # the finish gate: still failing, nothing written since
        text("area() still divides by zero; left open."),  # same state: the answer stands
    ])
    assert agent.run("Implement `area(w, h)` in `geo.py`.") == \
        "area() still divides by zero; left open."
    [note] = notes(seen)
    assert ("Vega's last check against the task fails and nothing was written since: "
            in note) and "ZeroDivisionError" in note
    assert "if you cannot, say in your answer exactly which check still fails" in note


def monkeypatch_env_without(name: str) -> str:
    import os

    return ",".join(n for n in os.environ.get(features.ENV, "").split(",") if n and n != name)


def test_switched_off_the_first_answer_stands(tmp_path: Path, monkeypatch) -> None:
    import os

    monkeypatch.setenv(features.ENV, os.environ.get(features.ENV, "") + ",finish_gate")
    agent, seen = agent_with(tmp_path, [call("list_files"), text("Done.")])
    assert agent.run(R_TASK) == "Done."
    assert notes(seen) == []


@pytest.mark.parametrize("task", ["What does `area` in `geo.py` return?", "hi"])
def test_questions_and_conversation_are_never_gated(tmp_path: Path, task: str) -> None:
    (tmp_path / "geo.py").write_text("def area(w, h):\n    raise NotImplementedError\n")
    agent, seen = agent_with(tmp_path, [call("read_file", path="geo.py"), text("It raises."),
                                        text("It raises.")])
    agent.run(task)
    assert notes(seen) == []
