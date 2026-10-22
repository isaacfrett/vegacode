"""Full-screen app (Textual) driven headless with a scripted model: no server, no real state."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from switches import switch_off
from tui_helpers import (
    answer,
    calls,
    click_ready,
    drive,
    make_app,
    scripted_agent,
    send,
    wait_until,
)
from typer.testing import CliRunner

from vega import cli, tools
from vega.agent import NOT_RUN, STOPPED, CodingAgent
from vega.tui import (
    THEMES,
    ChangesList,
    ChoiceScreen,
    ContextGauge,
    HelpScreen,
    PromptInput,
    Steps,
    StepsToggle,
    Turn,
    Usage,
    VegaApp,
    gauge_cells,
    glyphs,
)


@pytest.fixture(autouse=True)
def no_console_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app must never fall back to console.input (it would hang a full-screen terminal)."""
    def refuse(*_: object, **__: object) -> str:
        raise AssertionError("console.input called inside the TUI")

    monkeypatch.setattr(tools.console, "input", refuse)


def project_with(tmp_path: Path, **files: str) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'p'\n")
    for name, content in files.items():
        path = project / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return project


def edit(path: str, old: str, new: str) -> list:
    return calls(("edit_file", {"path": path, "old_text": old, "new_text": new}))


def history_is_valid(messages: list[dict]) -> bool:
    """Every assistant tool call is followed by exactly its tool results, in order."""
    for index, message in enumerate(messages):
        ids = [call["id"] for call in message.get("tool_calls") or []]
        if not ids:
            continue
        following = messages[index + 1: index + 1 + len(ids)]
        if [m.get("role") for m in following] != ["tool"] * len(ids):
            return False
        if [m.get("tool_call_id") for m in following] != ids:
            return False
    return True


# -- a turn ---------------------------------------------------------------------------------------

def test_turn_shows_live_steps_then_answer_with_steps_collapsed(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    (project / "a.py").write_text("x = 1\n")
    hold = threading.Event()
    replies = [
        calls(("read_file", {"path": "a.py"})),
        edit("a.py", "x = 1", "x = 2"),
        answer("Set **x** to 2 in `a.py`."),
    ]
    count = {"n": 0}

    def gate() -> None:
        count["n"] += 1
        if count["n"] == 3:
            hold.wait(10)

    agent, _ = scripted_agent(project, replies, gate=gate)
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        await send(pilot, "set x to 2", wait=False)
        turn = app.query_one(Turn)
        await wait_until(pilot, lambda: turn.steps.step_count == 2
                         and not any(e.running for e in turn.steps.entries))
        await pilot.pause(0.15)
        # live: steps visible with aligned columns, a working line, no answer yet
        assert turn.steps.display and not turn.answer.display
        rows = str(turn.steps.render_entries(60)).splitlines()
        assert rows[0].startswith("read_file  a.py") and rows[1].startswith("edit_file  a.py")
        assert "exact match" in rows[1]
        assert rows[0].index("a.py") == rows[1].index("a.py")  # aligned target column
        assert "working" in str(turn.working.render())
        assert "working" in str(app.query_one("#state").render())
        hold.set()
        await wait_until(pilot, lambda: not app.working)
        await pilot.pause()
        # finished: answer shown, steps collapsed into one line that expands
        assert turn.answer.display and turn.answer.source == "Set **x** to 2 in `a.py`."
        assert not turn.steps.display
        toggle = turn.query_one(StepsToggle)
        assert str(toggle.render()).startswith(f"{glyphs().closed} 2 steps, ")
        toggle.focus()
        await pilot.press("enter")
        assert turn.steps.display and str(toggle.render()).startswith(glyphs().opened)
        await pilot.click(StepsToggle)
        assert not turn.steps.display
        assert "ready" in str(app.query_one("#state").render())
        assert app.turns == 1 and app.model_calls == 3
        assert "last reply" in str(app.query_one("#readouts").render())

    drive(app, script)
    assert (project / "a.py").read_text() == "x = 2\n"
    saved = json.loads(app.session_log.path.read_text())
    assert saved["turns"][0]["user"] == "set x to 2" and len(saved["turns"][0]["requests"]) == 3
    assert app.session_log.path.parent == tmp_path / "sessions"


def test_retry_notes_and_failed_checks_are_shown(tmp_path: Path, monkeypatch) -> None:
    from vega import sandbox

    project = project_with(tmp_path)
    monkeypatch.setattr(sandbox, "run_check",
                        lambda command, root, **_: "exit code: 1\nstdout:\n2 failed, 1 passed in 0.1s\n")
    agent, _ = scripted_agent(project, [
        answer("Which file do you mean?"),  # sent back: no tool used yet
        calls(("run_check", {"command": "pytest -q"})),
        answer("Two tests fail."),
    ])
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        await send(pilot, "run the tests")
        steps = app.query_one(Steps)
        text = str(steps.render_entries(80))
        assert "(retrying: replied without inspecting the project)" in text
        assert f"{glyphs().warn} exit 1, 2 failed, 1 passed" in text
        assert [e.level for e in steps.entries if e.kind == "step"] == ["error"]

    drive(app, script)


def test_plan_panel_updates_and_plan_shows_inline(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    items = [{"text": "find add()", "status": "done"},
             {"text": "fix the operator", "status": "in_progress"},
             {"text": "run the tests", "status": "pending"}]
    agent, _ = scripted_agent(project, [
        calls(("update_plan", {"items": items})),
        answer("Planned."),
    ], planning="on")
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        g = glyphs()
        assert "No plan yet" in str(app.query_one("#plan").render())
        await send(pilot, "plan it")
        panel = str(app.query_one("#plan").render())
        assert f"{g.done} find add()" in panel and f"{g.doing} fix the operator" in panel
        assert f"{g.pending} run the tests" in panel
        inline = str(app.query_one(Steps).render_entries(70)).splitlines()
        # one grid row per plan update: the item in progress and where it is in the plan
        assert inline[0].split() == ["plan", "fix", "the", "operator", "step", "2", "of", "3"]
        assert "plan updated" not in "\n".join(inline)
        assert app.query_one(Steps).step_count == 0  # update_plan is not a counted step

    drive(app, script)


# -- approvals ------------------------------------------------------------------------------------

def test_review_modal_apply_and_skip(tmp_path: Path, monkeypatch) -> None:
    # a short request naming nothing: clarify_model would ask the scripted model first
    switch_off(monkeypatch, "clarify_first")
    project = project_with(tmp_path)
    (project / "a.py").write_text("x = 1\ny = 1\n")
    agent, _ = scripted_agent(project, [
        edit("a.py", "x = 1", "x = 2"),
        edit("a.py", "y = 1", "y = 2"),
        answer("Changed x; you skipped y."),
    ])
    app = make_app(project, agent, tmp_path, review=True)

    async def script(pilot) -> None:
        await send(pilot, "change x and y", wait=False)
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        screen = app.screen
        assert str(screen.query_one("#dialog-title").render()) == "Edit a.py?"
        detail = str(screen.query_one("#dialog-detail-text").render())
        assert "-x = 1" in detail and "+x = 2" in detail
        assert "waiting for you" in str(app.query_one("#state").render())
        await pilot.press("y")
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen)
                         and app.screen is not screen)
        assert (project / "a.py").read_text() == "x = 2\ny = 1\n"
        await click_ready(pilot, "#no")  # the "Skip" button
        await wait_until(pilot, lambda: not app.working)
        await pilot.pause()
        outcomes = [e.outcome for e in app.query_one(Steps).entries if e.kind == "step"]
        assert outcomes[0] == "exact match" and outcomes[1].startswith("User declined")

    drive(app, script)
    assert (project / "a.py").read_text() == "x = 2\ny = 1\n"


def test_review_buttons_are_labelled(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    (project / "a.py").write_text("x = 1\n")
    agent, _ = scripted_agent(project, [edit("a.py", "x = 1", "x = 2"), answer("ok")])
    app = make_app(project, agent, tmp_path, review=True)

    async def script(pilot) -> None:
        await send(pilot, "edit", wait=False)
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        labels = [str(button.label) for button in app.screen.query("Button")]
        assert labels == ["Apply change", "Skip"]
        await click_ready(pilot, "#yes")
        await wait_until(pilot, lambda: not app.working)

    drive(app, script)
    assert (project / "a.py").read_text() == "x = 2\n"


def test_run_command_asks_in_a_modal(tmp_path: Path, monkeypatch) -> None:
    # a short request naming nothing: clarify_model would ask the scripted model first
    switch_off(monkeypatch, "clarify_first")
    project = project_with(tmp_path)
    agent, seen = scripted_agent(project, [
        calls(("run_command", {"command": "echo made > made.txt"})),
        calls(("run_command", {"command": "echo nope > nope.txt"})),
        answer("Ran one, you declined the other."),
    ])
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        await send(pilot, "make files", wait=False)
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        assert str(app.screen.query_one("#dialog-title").render()) == "Run this command?"
        assert "echo made > made.txt" in str(app.screen.query_one("#dialog-detail-text").render())
        assert [str(b.label) for b in app.screen.query("Button")] == ["Run command", "Don't run"]
        first = app.screen
        await pilot.press("y")
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen)
                         and app.screen is not first)
        await pilot.press("n")
        await wait_until(pilot, lambda: not app.working)

    drive(app, script)
    assert (project / "made.txt").exists() and not (project / "nope.txt").exists()
    results = [m["content"] for m in seen[-1] if m.get("role") == "tool"]
    assert results[0].startswith("exit code: 0") and results[1].startswith("User declined")


def test_execute_uses_the_approval_callback_and_keeps_the_console_default(tmp_path: Path) -> None:
    asked: list[str] = []
    result = tools.execute("run_command", {"command": "echo hi > x.txt"}, tmp_path,
                           approve_command=lambda command: asked.append(command) or False)
    assert asked == ["echo hi > x.txt"] and result.startswith("User declined")
    assert not (tmp_path / "x.txt").exists()
    # default (line mode): the console prompt, which the fixture turns into an error here
    default = tools.execute("run_command", {"command": "echo hi > x.txt"}, tmp_path)
    assert "console.input called" in default and not (tmp_path / "x.txt").exists()


# -- stopping -------------------------------------------------------------------------------------

def test_ctrl_c_stops_after_the_current_step_and_history_stays_valid(tmp_path: Path) -> None:
    project = project_with(tmp_path, a_txt="a")
    released = threading.Event()
    count = {"n": 0}

    def gate() -> None:
        count["n"] += 1
        if count["n"] == 2:
            released.wait(10)  # the second model request is "in flight" when Ctrl+C arrives

    agent, seen = scripted_agent(project, [
        calls(("read_file", {"path": "a_txt"})),
        calls(("git_status", {}), ("read_file", {"path": "a_txt"})),
        answer("Hello again."), answer("Hello again."),  # a reply without tools is sent back once
    ], gate=gate)
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        await send(pilot, "look around", wait=False)
        await wait_until(pilot, lambda: count["n"] == 2)
        await pilot.press("ctrl+c")
        await pilot.pause(0.15)
        assert "stopping after this step" in str(app.query_one("#state").render())
        released.set()
        await wait_until(pilot, lambda: not app.working)
        await pilot.pause()
        turn = app.query_one(Turn)
        assert turn.answer.source == STOPPED
        assert history_is_valid(agent.messages)
        # the reply that was in flight when Ctrl+C arrived is discarded whole: none of its calls ran
        assert not any(m.get("tool_calls") and any(c["function"]["name"] == "git_status"
                                                   for c in m["tool_calls"]) for m in agent.messages)
        await send(pilot, "hello")  # the next turn starts cleanly
        assert list(app.query(Turn))[-1].answer.source == "Hello again."

    drive(app, script)
    assert history_is_valid(seen[-1])


def test_request_stop_between_tool_calls_without_the_app(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    agent, _ = scripted_agent(tmp_path, [
        calls(("read_file", {"path": "a.txt"}), ("git_status", {}),
              ("read_file", {"path": "a.txt"})),
        answer("again"), answer("again"),
    ])
    agent.on_event = lambda kind, data: agent.request_stop() if kind == "tool_result" else None
    assert agent.run("read") == STOPPED
    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert [m["content"] for m in tool_messages][1:] == [NOT_RUN, NOT_RUN]
    assert history_is_valid(agent.messages)
    agent.on_event = None
    assert agent.run("next") == "again"  # the stop flag is cleared at the start of each turn
    assert history_is_valid(agent.messages)


def test_stop_while_a_modal_is_open_declines_it(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    (project / "a.py").write_text("x = 1\n")
    agent, _ = scripted_agent(project, [edit("a.py", "x = 1", "x = 2"), answer("never")])
    app = make_app(project, agent, tmp_path, review=True)

    async def script(pilot) -> None:
        await send(pilot, "edit", wait=False)
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        await pilot.press("ctrl+c")
        await wait_until(pilot, lambda: not app.working)
        assert not isinstance(app.screen, ChoiceScreen)
        assert app.query_one(Turn).answer.source == STOPPED

    drive(app, script)
    assert (project / "a.py").read_text() == "x = 1\n"
    assert history_is_valid(agent.messages)


# -- undo -----------------------------------------------------------------------------------------

def three_edit_turns(project: Path) -> list:
    return [
        edit("n.py", "n = 0", "n = 1"), answer("one"),
        edit("n.py", "n = 1", "n = 2"), answer("two"),
        edit("n.py", "n = 2", "n = 3"), answer("three"),
    ]


def test_slash_undo_and_panel_undo_back_to_a_turn(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    (project / "n.py").write_text("n = 0\n")
    agent, _ = scripted_agent(project, three_edit_turns(project))
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        for message in ("one", "two", "three"):
            await send(pilot, message)
        changes = app.query_one("#changes", ChangesList)
        assert [record.turn for record in app.changes] == [1, 2, 3]
        first = str(changes.get_option_at_index(0).prompt).splitlines()
        assert first[0] == "turn 1" and first[1].startswith("n.py") and "+1" in first[1]
        await send(pilot, "/undo")
        assert (project / "n.py").read_text() == "n = 2\n"
        assert agent.messages[-1]["content"].startswith("(Vega harness) The user undid")
        assert [record.turn for record in app.changes] == [1, 2]
        # panel: select turn 1 and press u -> confirm (two turns) -> both undone
        changes.focus()
        changes.highlighted = 0
        await pilot.press("u")
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        assert "Undo back to turn 1?" in str(app.screen.query_one("#dialog-title").render())
        await pilot.press("y")
        await pilot.pause()
        assert (project / "n.py").read_text() == "n = 0\n"
        assert app.changes == [] and not changes.display
        await pilot.press("ctrl+u")
        await pilot.pause()
        notes = [str(n.query_one(".note-body").render()) for n in app.query(".note")]
        assert any("Nothing to undo" in note for note in notes)

    drive(app, script)


def test_undo_conflict_asks_before_forcing(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    (project / "n.py").write_text("n = 0\n")
    agent, _ = scripted_agent(project, three_edit_turns(project)[:2])
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        await send(pilot, "one")
        (project / "n.py").write_text("n = 1  # mine\n")
        await send(pilot, "/undo")  # typed: refuses and explains --force
        notes = " ".join(str(n.query_one(".note-body").render()) for n in app.query(".note"))
        assert "Not undone" in notes and "/undo --force" in notes
        assert (project / "n.py").read_text() == "n = 1  # mine\n"
        await pilot.press("ctrl+u")  # key: asks in a modal
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        assert "n.py" in str(app.screen.query_one("#dialog-body").render())
        await pilot.press("n")
        await pilot.pause()
        assert (project / "n.py").read_text() == "n = 1  # mine\n"
        await pilot.press("ctrl+u")
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        await pilot.press("y")
        await pilot.pause()
        assert (project / "n.py").read_text() == "n = 0\n"

    drive(app, script)


# -- layout, theme, keys --------------------------------------------------------------------------

def test_narrow_layout_hides_the_panel_and_ctrl_p_overlays_it(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        side = app.query_one("#side")
        assert not side.display and app.screen.has_class("narrow")
        await wait_until(pilot, lambda: app.memory_mb is not None)
        await pilot.pause()
        status = str(app.query_one("#readouts").render())
        hints = str(app.query_one("#hints").render())
        assert "memory 5.1 of 16 GB" in status and hints == "^P panel"  # nothing else fits
        assert "last reply" not in status and "first call" not in status
        assert app.query_one(ContextGauge).cells == 20
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert side.display and side.region.right <= 80 and side.region.width == 34
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert not side.display
        await pilot.resize_terminal(120, 36)
        await pilot.pause()
        assert side.display and not app.screen.has_class("narrow")
        hints = str(app.query_one("#hints").render())
        assert hints.endswith("^P panel") and "^U undo" in hints
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert not side.display
        # a resize while a dialog is open still switches the main screen's layout
        await pilot.press("ctrl+p")
        app.push_screen(ChoiceScreen("Title", "Body", "Yes", "No"))
        await pilot.pause()
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert app.default_screen.has_class("narrow") and not app.screen.has_class("narrow")

    drive(app, script, size=(80, 24))


def test_theme_toggle_and_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_THEME", "light")
    project = project_with(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        assert app.theme == "vega-light"
        assert app.current_theme.background == "#F3F6FA"
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert app.theme == "vega-dark" and app.current_theme.background == "#0F1626"
        await pilot.press("ctrl+t")
        assert app.theme == "vega-light"

    drive(app, script)


def contrast(first: str, second: str) -> float:
    def luminance(colour: str) -> float:
        channels = [int(colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("name", ["dark", "light"])
def test_text_colours_meet_contrast_on_background_and_panel(name: str) -> None:
    theme = THEMES[name]
    texts = [theme.foreground, theme.accent, theme.success, theme.error,
             theme.variables["vega-muted"], theme.variables["vega-warning-text"]]
    for surface in (theme.background, theme.panel, theme.variables["vega-overlay"]):
        for colour in texts:
            assert contrast(colour, surface) >= 4.5, (name, colour, surface)
    # the gauge segments are graphics: 3:1 against the background, and told apart by glyph too
    assert contrast(theme.variables["vega-fixed"], theme.background) >= 3
    assert contrast(theme.accent, theme.background) >= 3


def test_ascii_glyphs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VEGA_ASCII", "1")
    project = project_with(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        gauge = app.query_one(ContextGauge)
        gauge.set_usage(Usage(fixed=4000, total=8000, window=16384))
        bar = str(gauge.render())
        assert "=" in bar and "#" in bar and "|" in bar and "49% of 16K" in bar
        assert not any(ch in bar for ch in "▓█░│")
        assert str(app.query_one("#prompt-mark").render()) == ">"

    drive(app, script, size=(120, 36))


def test_gauge_segments_and_compaction_settles(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        g = glyphs()
        gauge = app.query_one(ContextGauge)
        gauge.set_usage(Usage(fixed=2048, total=8192, window=16384))
        bar = str(gauge.render())
        assert gauge_cells(2048, 8192, 16384, 20) == (2, 8)  # 2/20 fixed, 10/20 in all
        assert bar.count(g.used) == 10 and bar.count(g.empty) == 10
        assert bar.index(g.tick) == len("context ") + 13  # the 65% compaction point
        assert bar.endswith("50% of 16K")
        # fixed and conversation are told apart by colour and by the labelled readouts
        segments = gauge.render().spans
        colours = {span.style for span in segments}
        assert len(colours) >= 4  # label, fixed, conversation, track (+ tick, readout)
        app.animation_level = "none"  # TEXTUAL_ANIMATIONS=none: no transition at all
        gauge.set_usage(Usage(fixed=2048, total=4096, window=16384), settle=True)
        assert gauge.used == 4096
        app.animation_level = "full"
        gauge.set_usage(Usage(fixed=2048, total=12000, window=16384))
        gauge.set_usage(Usage(fixed=2048, total=5000, window=16384), settle=True)
        await pilot.pause()
        assert 5000 <= gauge.used < 12000  # moving or already there (a slow CI runner), once
        await wait_until(pilot, lambda: gauge.used == 5000, seconds=3)

    drive(app, script, size=(120, 36))


def test_compaction_notice_is_shown_in_the_turn(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    held = threading.Event()
    agent, _ = scripted_agent(project, [answer("ok"), answer("ok")], gate=lambda: held.wait(10))
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        await send(pilot, "x", wait=False)
        await wait_until(pilot, lambda: app.turn is not None)
        app._handle_event("notice", {"message": "Context compacted: 4 older messages"},
                          Usage(fixed=1000, total=2000, window=16384))
        held.set()
        await wait_until(pilot, lambda: not app.working)
        assert "Context compacted" in str(app.query_one(Steps).render_entries(80))

    drive(app, script, size=(120, 36))


def test_input_history_newlines_and_slash_commands(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    agent, _ = scripted_agent(project, [answer("first answer")])
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        prompt = app.query_one(PromptInput)
        assert prompt.placeholder == "Ask Vega to change, explain or check something"
        await send(pilot, "first")
        await pilot.press("a", "shift+enter", "b", "ctrl+j", "c")
        assert prompt.text == "a\nb\nc"
        prompt.load_text("")
        await pilot.press("up")
        assert prompt.text == "first"
        await pilot.press("down")
        assert prompt.text == ""
        await send(pilot, "/review on")
        assert agent.review is True
        await pilot.press("ctrl+r")
        assert agent.review is False
        await send(pilot, "/help")
        assert isinstance(app.screen, HelpScreen)
        await pilot.press("escape")
        await pilot.press("f1")
        assert isinstance(app.screen, HelpScreen)
        await pilot.press("escape")
        await send(pilot, "/reset")
        assert len(agent.messages) == 1
        await send(pilot, "/status")
        await send(pilot, "/bogus")
        notes = " ".join(str(n.query_one(".note-body").render()) for n in app.query(".note"))
        assert "Conversation context cleared" in notes and "Unknown command /bogus" in notes
        await pilot.press("ctrl+l")
        await pilot.pause()
        assert not list(app.query(".message"))
        assert len(agent.messages) == 1  # Ctrl+L clears the view, not the context

    drive(app, script, size=(120, 36))


def test_not_a_project_notice_dismisses(tmp_path: Path) -> None:
    folder = tmp_path / "loose"
    folder.mkdir()
    agent, _ = scripted_agent(folder, [])
    app = make_app(folder, agent, tmp_path)

    async def script(pilot) -> None:
        assert len(app.query(".dismissible")) == 1
        await pilot.press("escape")
        await pilot.pause()
        assert len(app.query(".dismissible")) == 0

    drive(app, script, size=(120, 36))


def test_slash_exit_quits(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        await send(pilot, "/exit", wait=False)
        await pilot.pause()

    drive(app, script)
    assert app.return_code == 0


# -- CLI: who gets the TUI ------------------------------------------------------------------------

@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from vega import tui

    seen: list[str] = []
    monkeypatch.setattr(cli, "_ensure_server", lambda: None)
    monkeypatch.setattr(cli, "_interactive_session",
                        lambda project, review=False, teach=False: seen.append("line"))
    monkeypatch.setattr(tui, "run_tui", lambda project, **_: seen.append("tui"))
    return seen


def test_non_tty_uses_line_mode(launches: list[str], tmp_path: Path) -> None:
    result = CliRunner().invoke(cli.app, ["-p", str(tmp_path)])
    assert result.exit_code == 0, result.output
    result = CliRunner().invoke(cli.app, ["chat", "-p", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert launches == ["line", "line"]


def test_tty_opens_the_tui_unless_plain(launches: list[str], tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_use_tui", lambda plain: not plain)
    runner = CliRunner()
    for args in (["-p", str(tmp_path)], ["chat", "-p", str(tmp_path)],
                 ["--plain", "-p", str(tmp_path)], ["chat", "--plain", "-p", str(tmp_path)],
                 ["--plain", "chat", "-p", str(tmp_path)]):
        result = runner.invoke(cli.app, args)
        assert result.exit_code == 0, result.output
    assert launches == ["tui", "tui", "line", "line", "line"]


def test_use_tui_needs_both_ends_to_be_terminals(monkeypatch) -> None:
    class Stream:
        def __init__(self, tty: bool) -> None:
            self.tty = tty

        def isatty(self) -> bool:
            return self.tty

    for stdin, stdout, expected in ((True, True, True), (True, False, False),
                                    (False, True, False)):
        monkeypatch.setattr(cli.sys, "stdin", Stream(stdin))
        monkeypatch.setattr(cli.sys, "stdout", Stream(stdout))
        assert cli._use_tui(False) is expected
    assert cli._use_tui(True) is False


def test_app_agent_defaults(tmp_path: Path, monkeypatch) -> None:
    """Built by the CLI path: streaming, undo on, approvals routed to the app (no console)."""
    project = project_with(tmp_path)
    app = VegaApp(project, review=True, warm_up=False, memory_probe=lambda: 0.0)
    assert isinstance(app.agent, CodingAgent) and app.agent.stream and app.agent.review
    assert app.agent.undo_depth == 20
    assert app.agent.approve == app._approve_write
    assert app.agent.approve_command == app._approve_command


def test_escape_stops_and_a_second_ctrl_c_force_quits(tmp_path: Path) -> None:
    project = project_with(tmp_path, a_txt="a")
    released = threading.Event()
    count = {"n": 0}

    def gate() -> None:
        count["n"] += 1
        if count["n"] == 2:
            released.wait(10)

    agent, _ = scripted_agent(project, [
        calls(("read_file", {"path": "a_txt"})), answer("done"), answer("done"),
    ], gate=gate)
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        await send(pilot, "look around", wait=False)
        await wait_until(pilot, lambda: count["n"] == 2)
        await pilot.press("escape")  # Esc stops a running turn, like Ctrl+C
        await pilot.pause(0.1)
        assert agent.stop_requested
        await pilot.press("ctrl+c")  # again within 2 s: quit without waiting for the reply
        await pilot.pause(0.1)
        assert getattr(app, "forced_exit", False)
        released.set()

    drive(app, script)
