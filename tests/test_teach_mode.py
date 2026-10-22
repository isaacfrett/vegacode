"""Teach mode (the user's request, 2026-10-16): the model says why while it works; the harness
shows it, keeps it out of the model's later context and checks it names only what the turn showed.
Scripted fake models only."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from switches import switch_off
from test_agent_loop import FakeMessage, agent_with, call, text
from tui_helpers import calls as streamed_calls
from tui_helpers import chunk, drive, make_app, scripted_agent, send

from vega import context, pipeline, project_pipeline, sandbox, server, teaching, view
from vega.agent import CodingAgent

FAILING = ("exit code: 1\nF\n=================================== FAILURES ===\n"
           "FAILED test_calc.py::test_add - assert -1 == 5\n1 failed in 0.01s")
PASSING = "exit code: 0\n.\n1 passed in 0.01s"
TAKEAWAY = ("## What to take away\n- Decision: change the operator in `add()` rather than "
            "special-casing the test's inputs; the trade-off is none here, the special case would "
            "hide the bug.\n- Concept: an operator typo (subtraction instead of addition).\n"
            "- Check: what would `add(-2, 2)` have returned before the fix?")


def said(content: str, *calls: FakeMessage) -> FakeMessage:
    """A reply with text alongside its tool calls (the plan or a why line)."""
    merged = [c for message in calls for c in message.tool_calls]
    return FakeMessage(content=content, tool_calls=merged)


def loop_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent loop's tests: the direct pipeline (on by default) would take these tasks, and
    with it off the project pipeline (the fixture has visible tests)."""
    switch_off(monkeypatch, "direct_pipeline", "project_pipeline")


@pytest.fixture
def calc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    loop_only(monkeypatch)
    # acceptance's checks would also call the scripted run_check below and take its two results
    switch_off(monkeypatch, "acceptance")
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    results = [FAILING, PASSING]
    monkeypatch.setattr(sandbox, "run_check", lambda *a, **k: results.pop(0))
    return tmp_path


def lesson_script() -> list[FakeMessage]:
    return [
        said("Plan: read calc.py, fix add(), then run the tests. Reading first because the "
             "request names add() but not the line.", call("read_file", path="calc.py")),
        said("Why: the first attempt keeps the subtraction to confirm the test catches it.",
             call("run_check", command="python -m pytest -q")),
        said("The failure means add(2, 3) returned -1: the operator subtracts. Changing - to + "
             "fixes it.",
             call("edit_file", path="calc.py", old_text="a - b", new_text="a + b")),
        said("Why: confirm the fix with the same test.",
             call("run_check", command="python -m pytest -q")),
        text("Fixed add() in calc.py; pytest passes (1 passed).\n\n" + TAKEAWAY),
    ]


def events_of(agent: CodingAgent) -> list[tuple[str, dict]]:
    seen: list[tuple[str, dict]] = []
    agent.on_event = lambda kind, data: seen.append((kind, data))
    return seen


# -- the switch -----------------------------------------------------------------------------------

def test_off_by_default_and_the_prompt_lines_only_when_on(tmp_path: Path, monkeypatch) -> None:
    agent = CodingAgent(tmp_path)
    assert agent.teach is False and teaching.SECTION not in agent.messages[0]["content"]
    agent.set_teach(True)
    system = agent.messages[0]["content"]
    assert system.count(teaching.SECTION) == 1
    for line in ("state your plan and why", 'starting\n  "Why:"', "failing check",
                 '"## What to take away"', "alternative you rejected", "named precisely"):
        assert line in system
    agent.refresh_knowledge()  # rebuilding the tail keeps the section once
    assert agent.messages[0]["content"].count(teaching.SECTION) == 1
    agent.set_teach(False)
    assert teaching.SECTION not in agent.messages[0]["content"]
    monkeypatch.setenv("VEGA_TEACH", "1")
    assert CodingAgent(tmp_path).teach is True
    assert CodingAgent(tmp_path, teach=False).teach is False


def test_the_setting_is_saved_with_the_view_and_kept_across_presets() -> None:
    assert view.load().teach is False
    view.save(view.with_teach(view.View(), True))
    assert view.load().teach is True
    assert view.with_teach(view.load(), True).preset == "workbench"  # not a layout change
    assert view.validate({"teach": "yes"}).teach is False  # only a real true counts


# -- the agent loop ---------------------------------------------------------------------------------

def test_plan_why_and_fix_lines_are_shown_and_why_lines_leave_the_context(calc: Path) -> None:
    agent, seen = agent_with(calc, lesson_script())
    agent.set_teach(True)
    events = events_of(agent)
    answer = agent.run("Make add() in calc.py return the sum.")
    assert answer.endswith(TAKEAWAY)  # the user sees the takeaway
    lessons = [(data["role"], data["text"]) for kind, data in events if kind == "teach"]
    assert [role for role, _ in lessons] == ["plan", "why", "failure", "why"]
    assert lessons[1][1].startswith("the first attempt")  # "Why:" label dropped for the view
    assert lessons[2][1].startswith("The failure means add(2, 3) returned -1")
    # later steps: the plan line stays, why/fix lines are gone
    last = json.dumps(seen[-1])
    assert "Plan: read calc.py" in last
    assert "the first attempt keeps" not in last and "The failure means" not in last
    # the history kept for the next turn has no takeaway either
    final = [m for m in agent.messages if m["role"] == "assistant" and not m.get("tool_calls")]
    assert final[-1]["content"] == "Fixed add() in calc.py; pytest passes (1 passed)."


def test_off_nothing_is_classified_or_stripped(calc: Path) -> None:
    agent, seen = agent_with(calc, lesson_script())
    events = events_of(agent)
    agent.run("Make add() in calc.py return the sum.")
    assert not [kind for kind, _ in events if kind == "teach"]
    assert "the first attempt keeps" in json.dumps(seen[-1])
    assert "What to take away" in agent.messages[-1]["content"]


def test_tokens_saved_on_a_scripted_transcript(calc: Path, monkeypatch) -> None:
    """The same scripted turn with and without stripping: the history the next turn starts from."""
    on, _ = agent_with(calc, lesson_script())
    on.set_teach(True)
    on.run("Make add() in calc.py return the sum.")
    (calc / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    results = [FAILING, PASSING]
    monkeypatch.setattr(sandbox, "run_check", lambda *a, **k: results.pop(0))
    monkeypatch.setattr(teaching, "role", lambda *a: "plan")  # every line kept, like the plan
    monkeypatch.setattr(teaching, "strip_history", lambda messages: 0)
    kept, _ = agent_with(calc, lesson_script())
    kept.set_teach(True)
    kept.run("Make add() in calc.py return the sum.")
    with_lines = context.estimate_tokens(kept.messages[1:])
    stripped = context.estimate_tokens(on.messages[1:])
    print(f"history after one turn: {with_lines} tokens unstripped, {stripped} stripped")
    assert with_lines - stripped >= 100  # why/fix lines and the takeaway, every later request


# -- the grounding guard -----------------------------------------------------------------------------

def test_guard_fires_once_on_an_invented_file(calc: Path) -> None:
    agent, seen = agent_with(calc, [
        said("Plan: read calc.py and fix add().", call("read_file", path="calc.py")),
        said("Why: helpers.py defines the shared operator table, so add() must change.",
             call("edit_file", path="calc.py", old_text="a - b", new_text="a + b")),
        said("Why: run the tests from mathlib.py.", call("run_check", command="python -m pytest -q")),
        call("run_check", command="python -m pytest -q"),
        text("Fixed add().\n\n## What to take away\n- `normalize_ops()` was rejected."),
    ])
    agent.set_teach(True)
    events = events_of(agent)
    agent.run("Make add() in calc.py return the sum.")
    notes = [m["content"] for request in seen for m in request
             if "never showed" in str(m.get("content"))]
    first = seen[2][-1]["content"]  # the edit's result carries the note
    assert first.startswith("Edited calc.py") and "names `helpers.py`, which this turn's tool calls" \
        in first
    assert len({n for n in notes}) == 1  # once per turn: not for mathlib.py, not for the takeaway
    assert sum(kind == "discarded" and "never showed" in data["reason"] for kind, data in events) == 1


def test_guard_sends_back_an_invented_takeaway_once(calc: Path) -> None:
    agent, seen = agent_with(calc, [
        call("edit_file", path="calc.py", old_text="a - b", new_text="a + b"),
        call("run_check", command="python -m pytest -q"),
        text("Fixed.\n\n## What to take away\n- We rejected rewriting `apply_operator()` in "
             "ops/table.py."),
        text("Fixed.\n\n## What to take away\n- Concept: operator typo in `add()`."),
    ])
    agent.set_teach(True)
    answer = agent.run("Make add() in calc.py return the sum.")
    assert answer.endswith("operator typo in `add()`.")
    note = seen[3][-1]["content"]
    assert note.startswith('(Vega harness) Your "What to take away" names `ops/table.py`, '
                           "`apply_operator`")


def test_guard_ignores_builtins_the_stdlib_and_what_the_turn_showed() -> None:
    seen = teaching.corpus(["Fix add() in calc.py", {"path": "calc.py"},
                            "1: def add(a, b):\n2:     return a - b"])
    text_ok = ("Use `sorted()` not `list.sort()`; `os.path.join` and `functools.lru_cache` are "
               "fine; add() in calc.py; see `json.loads()` and print().")
    assert teaching.ungrounded(text_ok, seen) == []
    assert teaching.ungrounded("Edit utils/helpers.py and test_totals", seen) == [
        "utils/helpers.py", "test_totals"]


# -- the pipelines -------------------------------------------------------------------------------------

STUB = 'def add(a, b):\n    """Return the sum.\n\n    >>> add(2, 3)\n    5\n    """\n'
WRONG = "```python\ndef add(a, b):\n    return a - b\n```"
RIGHT = "```python\ndef add(a, b):\n    return a + b\n```\n\n" + TAKEAWAY


def test_direct_pipeline_note_uses_only_its_real_stages(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "acceptance")  # its checks would add to the counts
    (tmp_path / "calc.py").write_text(STUB)
    results = [pipeline.CheckResult(total=1, failures=["add(2, 3) returned -1, expected 5"]),
               pipeline.CheckResult(total=1)]
    monkeypatch.setattr(pipeline, "run_checks", lambda *a, **k: results.pop(0))
    replies, seen = ["Here:\n" + WRONG + "\n\n" + TAKEAWAY, RIGHT], []
    agent = CodingAgent(tmp_path, allow_commands=False, teach=True)

    def create(**kwargs):
        seen.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=replies.pop(0)), finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer = agent.run("Implement add() in calc.py as its docstring says.")
    assert answer.startswith("Implemented the code in calc.py; 1 of 1 checks pass (1 docstring "
                             "example).\n\n## How this was done\n")
    assert "- Attempt 1: 0/1 checks passed; first failure: add(2, 3) returned -1, expected 5" \
        in answer
    assert "- Attempt 2: 1/1 checks passed (kept)" in answer
    assert answer.endswith(TAKEAWAY)  # the writer's own takeaway, after the stage note
    assert teaching.WRITER_LINE in seen[0][0]["content"]
    assert "What to take away" not in json.dumps(seen[1][1:])  # not resent in the repair prompt
    off = CodingAgent(tmp_path, allow_commands=False)
    assert off.teach is False and "nothing else" in pipeline.WRITER_SYSTEM


def test_project_pipeline_note_lists_planned_files_and_rounds(tmp_path: Path, monkeypatch) -> None:
    # unchanged_rounds off: the faked test run changes while the files do not
    monkeypatch.setenv("VEGA_FEATURES_OFF", "unchanged_rounds")
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "__init__.py").write_text("")
    (tmp_path / "shop" / "cart.py").write_text("def count(items):\n    return len(items)\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_cart.py").write_text("def test_total():\n    assert True\n")
    plan = json.dumps({"analysis": "", "files": [{"path": "shop/cart.py",
                                                   "change": "add total(items)"}]})
    code = "```python\ndef total(items):\n    return sum(p * q for p, q in items)\n```"
    replies = [plan, code, plan, code]
    runs = [(1, "FAILED tests/test_cart.py::test_total"), (0, "")]
    monkeypatch.setattr(project_pipeline, "_run_tests", lambda *a, **k: runs.pop(0))
    # pipeline_signals (default on) reads load errors and test ids from the same run
    monkeypatch.setattr(project_pipeline, "_run_tests_detail",
                        lambda *a, **k: (*runs.pop(0), [], []))
    agent = CodingAgent(tmp_path, allow_commands=False, teach=True)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **k: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=replies.pop(0)), finish_reason="stop")]))))
    answer = project_pipeline.run(agent, "Add total(items) to the shop: price times quantity, "
                                         "summed over the cart.")
    assert answer is not None
    body, note = teaching.split(answer)
    assert body == "Changed shop/cart.py; the project's tests pass."
    assert "- Planned by the model:\n  - `shop/cart.py`: add total(items)" in note
    assert "- Round 1: wrote shop/cart.py; 1 project test(s) failing" in note
    assert "- Round 2: wrote shop/cart.py; all tests pass" in note


def test_question_pipeline_note_names_the_excerpts_it_answered_from(tmp_path: Path,
                                                                      monkeypatch) -> None:
    monkeypatch.delenv("VEGA_FEATURES_OFF", raising=False)
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "pricing.py").write_text(
        "TIERS = {'gold': 20}\n\n\ndef discounted_price(price, tier):\n"
        "    return price - price * TIERS.get(tier, 0) / 100\n")
    replies, prompts = ["It returns 80.0 (shop/pricing.py)."], []

    def create(**kwargs):
        prompts.append(kwargs["messages"][-1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=replies.pop(0), tool_calls=None),
            finish_reason="stop")])

    agent = CodingAgent(tmp_path, allow_commands=False, teach=True)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer = agent.run("What does discounted_price(100, 'gold') in shop/pricing.py return?")
    body, note = teaching.split(answer)
    assert body == "It returns 80.0 (shop/pricing.py)."
    assert note.startswith("## How this was done\n- Answered in one model call from these "
                           "excerpts")
    assert "`shop/pricing.py` (whole file)" in note
    for place in note.split("`")[1::2]:
        assert place in prompts[0]  # only what the model was actually shown
    assert agent.messages[-1]["content"] == body  # the note never reaches the model


# -- the app ------------------------------------------------------------------------------------------

def test_teach_command_badge_and_a_teach_mode_turn(tmp_path: Path, monkeypatch) -> None:
    loop_only(monkeypatch)
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    edit = {"path": "calc.py", "old_text": "a - b", "new_text": "a + b"}
    replies = [
        [chunk("Plan: read calc.py, then fix the operator in add()."),
         *streamed_calls(("read_file", {"path": "calc.py"}))],
        [chunk("Why: add() subtracts; one operator changes."),
         *streamed_calls(("edit_file", edit))],
        [chunk("Fixed add() in calc.py.\n\n" + TAKEAWAY)],
    ]
    agent, seen = scripted_agent(tmp_path, replies)
    app = make_app(tmp_path, agent, tmp_path, layout="workbench")

    async def script(pilot) -> None:
        readouts = app.query_one("#readouts")
        assert "teach" not in str(readouts.render())
        await send(pilot, "/teach on")
        assert app.agent.teach and json.loads((server.STATE_DIR / "view.json").read_text())["teach"]
        assert " teach " in str(readouts.render())
        await send(pilot, "Make add() in calc.py return the sum.")
        turn = app.turn
        assert [role for role, _ in turn.lessons] == ["plan", "why"]
        assert turn.lessons[1] == ("why", "add() subtracts; one operator changes.")
        assert turn.lesson.display
        assert turn.takeaway.display and "Fixed add() in calc.py." == turn.reply.split("\n")[0]
        assert "add() subtracts" not in json.dumps(seen[-1])  # the why line left the context
        assert app.agent.messages[-1]["content"] == "Fixed add() in calc.py."  # and the takeaway
        await send(pilot, "/teach off")
        assert not app.agent.teach and " teach " not in str(readouts.render())

    drive(app, script)


def test_teach_flag_turns_it_on_for_the_session(tmp_path: Path) -> None:
    agent, _ = scripted_agent(tmp_path, [])
    app = make_app(tmp_path, agent, tmp_path, teach=True)
    assert app.agent.teach and teaching.SECTION in app.agent.messages[0]["content"]
    assert view.load().teach is False  # the flag is for this session, not saved


def test_vega_teach_flag_reaches_the_app_and_the_line_mode(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from vega import cli, tui

    seen: list[tuple[str, object]] = []
    monkeypatch.setattr(cli, "_ensure_server", lambda: None)
    monkeypatch.setattr(cli, "_interactive_session",
                        lambda project, review=False, teach=False: seen.append(("line", teach)))
    monkeypatch.setattr(tui, "run_tui", lambda project, **kw: seen.append(("tui", kw["teach"])))
    monkeypatch.setattr(cli, "_use_tui", lambda plain: not plain)
    runner = CliRunner()
    for args in (["--teach", "-p", str(tmp_path)], ["chat", "--teach", "-p", str(tmp_path)],
                 ["-p", str(tmp_path)], ["--plain", "--teach", "-p", str(tmp_path)]):
        assert runner.invoke(cli.app, args).exit_code == 0
    assert seen == [("tui", True), ("tui", True), ("tui", None), ("line", True)]


def test_the_eval_harness_turns_teach_mode_on_for_teach_mode_cases(tmp_path: Path) -> None:
    from vega import evals

    made: list[CodingAgent] = []

    def factory(project: Path, max_steps: int) -> CodingAgent:
        agent, _ = agent_with(project, [text("No change needed.")])
        made.append(agent)
        return agent

    case = {"id": "teach-mode-x", "task": "Is calc fine?", "files": {"calc.py": "x = 1\n"},
            "teach_mode": True, "response_contains": ["No change"]}
    evals.run_attempt(case, suite="teaching", agent_factory=factory)
    assert made[0].teach and teaching.SECTION in made[0].messages[0]["content"]
    evals.run_attempt({**case, "teach_mode": False}, suite="teaching", agent_factory=factory)
    assert made[1].teach is False


def test_takeaway_names_the_root_cause_first(monkeypatch) -> None:
    """teach_root_cause (a takeaway used to name the fix, "Decimal precision", as the concept and
    never the mechanism)."""
    line = teaching.WRITER_LINE
    assert line.index("Root cause:") < line.index("Fix:") < line.index("Rejected:")
    assert "mechanism that made the old code wrong" in line and "not the fix" in line
    assert "the root\n  cause" in teaching.SECTION and "not the fix" in teaching.SECTION
    monkeypatch.setenv("VEGA_FEATURES_OFF", "teach_root_cause")
    assert "Root cause" not in teaching.WRITER_LINE and "root" not in teaching.SECTION
    assert teaching.WRITER_LINE.startswith(" After the code block, add a short section headed")


def test_direct_pipeline_writer_gets_the_root_cause_line(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "acceptance")
    (tmp_path / "calc.py").write_text(STUB)
    monkeypatch.setattr(pipeline, "run_checks", lambda *a, **k: pipeline.CheckResult(total=1))
    lesson = ("## What to take away\nRoot cause: operator typo, `-` subtracts.\nFix: use `+`.\n"
              "Rejected: special-casing the example, because it hides the bug.")
    seen: list = []
    agent = CodingAgent(tmp_path, allow_commands=False, teach=True)

    def create(**kwargs):
        seen.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="```python\ndef add(a, b):\n    return a + b\n```\n\n" + lesson),
            finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    answer = agent.run("Fix add() in calc.py: add(2, 3) returns -1.")
    assert 'a line starting "Root cause:"' in seen[0][0]["content"]
    assert answer.endswith(lesson)
