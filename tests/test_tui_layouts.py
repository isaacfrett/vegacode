"""Layouts (classic, quiet, observatory, workbench) and the shared review fixes, headless with a
scripted model: no server, no real state."""

from __future__ import annotations

import re
import threading
from pathlib import Path

import pytest
from test_tui import contrast, edit, history_is_valid, project_with
from test_tui_screens import CALC, fake_check, session_replies
from tui_helpers import answer, calls, drive, make_app, scripted_agent, send, wait_until

from vega import sandbox, tools
from vega.agent import STOPPED
from vega.tui import (
    LAYOUT_THEMES,
    LAYOUTS,
    ChangesList,
    ChoiceScreen,
    ContextGauge,
    Steps,
    StepsToggle,
    Turn,
    Usage,
    clip_middle,
    gauge_cells,
    memory_label,
    plan_position,
    transcript_label,
)


@pytest.fixture(autouse=True)
def no_console_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **__: object) -> str:
        raise AssertionError("console.input called inside the TUI")

    monkeypatch.setattr(tools.console, "input", refuse)


def calc_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(sandbox, "run_check", fake_check)
    project = project_with(tmp_path)
    (project / "src").mkdir()
    (project / "src" / "calc.py").write_text(CALC)
    return project


# -- pure helpers ---------------------------------------------------------------------------------

def test_gauge_cells_keep_a_visible_conversation_segment() -> None:
    assert gauge_cells(2048, 8192, 16384, 20) == (2, 8)
    assert gauge_cells(3000, 3050, 16384, 20) == (4, 1)  # 50 tokens still get one accent cell
    assert gauge_cells(3000, 3000, 16384, 20) == (4, 0)  # nothing beyond the fixed prompt
    assert gauge_cells(100, 100, 16384, 20) == (1, 0)  # a tiny fixed prompt is still visible
    assert sum(gauge_cells(9000, 20000, 16384, 20)) == 20  # never longer than the bar


def test_path_time_memory_and_plan_labels() -> None:
    assert clip_middle("tests/test_sub.py", 14) == "tests/…_sub.py"
    assert clip_middle("src/calc.py", 20) == "src/calc.py"
    assert len(clip_middle("a/very/long/folder/name/module_file.py", 18)) == 18
    label = transcript_label(Path("/x/20261013T212916-todo-app.json"))
    assert re.fullmatch(r"\d{1,2} Oct \d\d:\d\d", label)
    assert memory_label(5.1 * 1024, 16.0) == "5.1 of 16 GB"
    assert memory_label(5.1 * 1024, None) == "5.1 GB"
    items = [{"text": "a", "status": "done"}, {"text": "b", "status": "in_progress"},
             {"text": "c", "status": "pending"}]
    assert plan_position(items) == ("b", "step 2 of 3")
    assert plan_position([dict(item, status="done") for item in items]) == ("c", "all 3 done")
    assert plan_position([]) == ("plan cleared", "")


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("mode", ["dark", "light"])
def test_every_palette_meets_contrast(layout: str, mode: str) -> None:
    theme = LAYOUT_THEMES[layout][mode]
    texts = [theme.foreground, theme.accent, theme.success, theme.error,
             theme.variables["vega-muted"], theme.variables["vega-warning-text"],
             theme.variables["vega-readout"]]
    for surface in (theme.background, theme.panel, theme.variables["vega-overlay"]):
        for colour in texts:
            assert contrast(colour, surface) >= 4.5, (layout, mode, colour, surface)
    for graphic in (theme.variables["vega-fixed"], theme.accent):
        assert contrast(graphic, theme.background) >= 3, (layout, mode, graphic)
    assert theme.variables["vega-fixed"].lower() != theme.accent.lower()
    # diff tints keep the text readable
    for tint in (theme.variables["vega-add-bg"], theme.variables["vega-del-bg"]):
        assert contrast(theme.foreground, tint) >= 4.5, (layout, mode, tint)


# -- the gauge ------------------------------------------------------------------------------------

def test_fixed_segment_is_frozen_after_one_marked_remeasure(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        app.animation_level = "none"
        gauge = app.query_one(ContextGauge)
        gauge.frozen, gauge.remeasured = False, False
        gauge.set_usage(Usage(fixed=1500, total=1600, window=16384))  # estimate
        assert gauge.fixed == 1500 and not gauge.frozen
        gauge.set_usage(Usage(fixed=3000, total=3500, window=16384, measured=True))
        assert gauge.fixed == 3000 and gauge.frozen and gauge.remeasured
        gauge.set_usage(Usage(fixed=6000, total=7000, window=16384, measured=True))
        assert gauge.fixed == 3000  # never silently doubles
        assert gauge.used == 7000
        app._refresh_panels()
        assert "3.0K re-measured" in str(app.query_one("#session").render())

    drive(app, script)


# -- a failing turn -------------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(120, 36), (80, 24)], ids=["120x36", "80x24"])
def test_failed_check_expands_marks_the_plan_and_stays_in_view(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]) -> None:
    project = calc_project(tmp_path, monkeypatch)
    agent, _ = scripted_agent(project, session_replies())
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        g = __import__("vega.tui", fromlist=["glyphs"]).glyphs()
        await send(pilot, "fix the bug in add() in src/calc.py")
        conversation = app.query_one("#conversation")
        await pilot.pause(0.2)
        assert conversation.scroll_y == conversation.max_scroll_y  # scrolled to the end
        await send(pilot, "Add a subtract function")
        await pilot.pause(0.3)
        first, second = app.turn_list
        assert not first.toggle.expanded and second.toggle.expanded  # the failure auto-expands
        assert second.steps.display
        # the model marked every item done, but the last check failed
        plan = str(app.query_one("#plan").render())
        assert f"{g.warn} add a test for it" in plan and "last check failed" in plan
        # in view: the reply's first line, or (when the steps are long) the first ⚠ step
        top, bottom = conversation.scroll_y, conversation.scroll_y + \
            conversation.scrollable_content_region.height
        reply = app._offset_in(conversation, second.answer)
        warn = app._offset_in(conversation, second.steps) + second.steps.first_warning_row()
        assert top <= warn < bottom
        assert top <= reply or warn == bottom - 1
        # units: lines for writes, `N s` for times
        rows = str(second.steps.render_entries(90))
        assert "5 lines" in rows and "characters" not in rows
        assert re.search(r"\d\.\d s$", str(second.toggle.render()))
        changes = app.query_one("#changes", ChangesList)
        prompt = str(changes.get_option_at_index(1).prompt).splitlines()
        assert prompt[0] == "turn 2" and any("test_sub.py" in row for row in prompt[1:])

    drive(app, script, size=size)


def test_passing_check_clears_the_failed_item(tmp_path: Path, monkeypatch) -> None:
    project = calc_project(tmp_path, monkeypatch)
    replies = session_replies() + [
        calls(("run_check", {"command": "pytest -q"}), prompt_tokens=8800),
        answer("All pass now.", prompt_tokens=9000),
    ]
    agent, _ = scripted_agent(project, replies)
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        await send(pilot, "one")
        await send(pilot, "two")
        assert app.failed_item == "add a test for it"
        await send(pilot, "three")
        assert app.failed_item is None
        assert "last check failed" not in str(app.query_one("#plan").render())
        assert not app.turn_list[-1].toggle.expanded

    drive(app, script)


def test_working_state_uses_spaced_seconds_and_one_waiting_phrase(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    (project / "a.py").write_text("x = 1\n")
    held = threading.Event()
    count = {"n": 0}

    def gate() -> None:
        count["n"] += 1
        if count["n"] == 1:
            held.wait(10)

    agent, _ = scripted_agent(project, [edit("a.py", "x = 1", "x = 2"), answer("ok")],
                              gate=gate, review=True)
    app = make_app(project, agent, tmp_path)

    async def script(pilot) -> None:
        await send(pilot, "edit", wait=False)
        await pilot.pause(0.3)
        assert re.search(r"working… \d+ s$", str(app.query_one("#state").render()))
        held.set()
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        assert str(app.query_one("#state").render()) == "waiting for you"
        assert "waiting for you" in str(app.query_one(Steps).render_entries(80))
        await pilot.press("y")
        await wait_until(pilot, lambda: not app.working)

    drive(app, script)


# -- layouts --------------------------------------------------------------------------------------

def test_layout_from_env_and_slash_layout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_LAYOUT", "observatory")
    project = project_with(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout=None)

    async def script(pilot) -> None:
        assert app.layout_name == "observatory" and app.has_class("layout-observatory")
        assert app.theme == "vega-observatory-dark"
        assert app.query_one("#machine-pane").display
        await send(pilot, "/layout")
        notes = " ".join(str(n.query_one(".note-body").render()) for n in app.query(".note"))
        assert "Layout observatory" in notes and "workbench" in notes
        await send(pilot, "/layout workbench")
        await pilot.pause()
        assert app.has_class("layout-workbench") and not app.has_class("layout-observatory")
        assert app.query_one("#preview").display and app.query_one("#left").display
        assert not app.query_one("#side").display
        await pilot.press("ctrl+t")
        assert app.theme == "vega-workbench-light"
        await send(pilot, "/layout quiet")
        await pilot.pause()
        assert app.theme == "vega-quiet-light"  # the mode survives a layout switch
        assert not app.query_one("#side").display
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert app.query_one("#side").display  # details as an overlay
        assert app.query_one("#overlay-hint").display
        await pilot.press("escape")
        await pilot.pause()
        assert not app.query_one("#side").display
        await send(pilot, "/layout nope")
        notes = " ".join(str(n.query_one(".note-body").render()) for n in app.query(".note"))
        assert "Unknown layout nope" in notes
        await send(pilot, "/layout classic")
        assert app.theme == "vega-light"

    drive(app, script)


def test_quiet_follows_the_terminal_background(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="quiet", transparent=True)

    async def script(pilot) -> None:
        assert app.ansi_color is True and app.has_class("transparent")
        assert app.screen.styles.background.ansi == -1  # ANSI default: the terminal's own
        await send(pilot, "/layout classic")
        await pilot.pause()
        assert app.ansi_color is None and not app.has_class("transparent")

    drive(app, script)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_each_layout_renders_at_both_sizes_and_themes(tmp_path: Path, monkeypatch,
                                                      layout: str) -> None:
    project = calc_project(tmp_path, monkeypatch)
    agent, _ = scripted_agent(project, session_replies()[:4])
    app = make_app(project, agent, tmp_path, layout=layout, transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "fix it")
        for size in ((120, 36), (80, 24)):
            await pilot.resize_terminal(*size)
            for _ in range(2):
                await pilot.press("ctrl+t")
                await pilot.pause()
                svg = app.export_screenshot()
                assert "fix&#160;it" in svg or size == (80, 24)
                status = str(app.query_one("#hints").render())
                assert status.endswith("^P panel")
                if size == (80, 24):
                    assert app.default_screen.has_class("narrow")
                    readouts = str(app.query_one("#readouts").render())
                    assert "memory 5.1 of 16 GB" in readouts

    drive(app, script, size=(120, 36))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_approve_undo_and_stop_work_in_every_layout(tmp_path: Path, layout: str) -> None:
    project = project_with(tmp_path)
    (project / "a.py").write_text("x = 1\n")
    released = threading.Event()
    count = {"n": 0}

    def gate() -> None:
        count["n"] += 1
        if count["n"] == 3:
            released.wait(10)

    agent, _ = scripted_agent(project, [
        edit("a.py", "x = 1", "x = 2"), answer("done"),
        calls(("read_file", {"path": "a.py"})), answer("never"), answer("never"),
    ], gate=gate, review=True)
    app = make_app(project, agent, tmp_path, layout=layout, transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "edit", wait=False)
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        await pilot.press("y")
        await wait_until(pilot, lambda: not app.working)
        assert (project / "a.py").read_text() == "x = 2\n"
        await pilot.press("ctrl+u")
        await pilot.pause()
        assert (project / "a.py").read_text() == "x = 1\n"
        await send(pilot, "look", wait=False)
        await wait_until(pilot, lambda: count["n"] == 3)
        await pilot.press("ctrl+c")
        released.set()
        await wait_until(pilot, lambda: not app.working)
        assert app.turn_list[-1].answer.source == STOPPED
        assert history_is_valid(agent.messages)

    drive(app, script, size=(120, 36))


def test_workbench_preview_and_tab_between_panes(tmp_path: Path, monkeypatch) -> None:
    project = calc_project(tmp_path, monkeypatch)
    agent, _ = scripted_agent(project, session_replies())
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "one")
        await send(pilot, "two")
        await pilot.pause()
        preview = app.query_one("#preview")
        body = str(app.query_one("#preview-body").render())
        # after a turn the preview shows the newest change: turn 2's src/calc.py
        assert "src/calc.py" in str(preview.border_title) and "turn 2" in str(preview.border_title)
        assert "+ def sub(a, b):" in body
        assert app.focused is app.query_one("#prompt")
        await pilot.press("tab")
        assert app.focused is app.query_one("#wb-changes")
        await pilot.press("down")  # next file: tests/test_sub.py
        await pilot.pause()
        assert "test_sub.py" in str(preview.border_title)
        await pilot.press("tab")
        assert app.focused is app.query_one("#wb-git")
        await pilot.press("tab")
        steps = app.query_one("#wb-steps")
        assert app.focused is steps
        index = next(i for i in range(steps.option_count)
                     if "run_check" in str(steps.get_option_at_index(i).prompt)
                     and "test_sub" in str(steps.get_option_at_index(i).prompt))
        steps.highlighted = index
        await pilot.pause()
        assert "ModuleNotFoundError" in str(app.query_one("#preview-body").render())
        await pilot.press("tab")
        assert app.focused is app.query_one("#wb-loops")
        await pilot.press("tab", "tab", "tab")
        assert app.focused is app.query_one("#prompt")  # around the ring and back
        await pilot.press("shift+tab")
        assert app.focused is preview

    drive(app, script, size=(120, 36))


def test_quiet_shows_changed_files_inline(tmp_path: Path, monkeypatch) -> None:
    project = calc_project(tmp_path, monkeypatch)
    agent, _ = scripted_agent(project, session_replies())
    app = make_app(project, agent, tmp_path, layout="quiet", transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "one")
        turn = app.query_one(Turn)
        assert turn.changes.display
        assert "changed  src/calc.py +1" in str(turn.changes.render())
        assert not app.query_one("#side").display
        await send(pilot, "/layout classic")
        await pilot.pause()
        assert not turn.changes.display  # the side panel carries it there

    drive(app, script)


def test_observatory_machine_pane(tmp_path: Path, monkeypatch) -> None:
    project = calc_project(tmp_path, monkeypatch)
    agent, _ = scripted_agent(project, session_replies()[:4])
    app = make_app(project, agent, tmp_path, layout="observatory", transparent=False,
                   swap_probe=lambda: 1.2 * 1024)

    async def script(pilot) -> None:
        await send(pilot, "one")
        await wait_until(pilot, lambda: app.swap_mb is not None)
        app._refresh_machine()
        machine = str(app.query_one("#machine").render())
        for reading in ("% of 16K", "fixed", "conversation", "compacts at 65%",
                        "5.1 of 16 GB", "swap", "1.2 GB", "last reply", "requests"):
            assert reading in machine, reading
        assert not app.query_one(ContextGauge).display  # the status bar leaves it to the pane
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert app.query_one(ContextGauge).display

    drive(app, script)


def test_steps_toggle_still_toggles(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    agent, _ = scripted_agent(project, [calls(("git_status", {})), answer("ok")])
    app = make_app(project, agent, tmp_path, layout="quiet", transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "status")
        toggle = app.query_one(StepsToggle)
        assert not toggle.expanded
        await pilot.click(StepsToggle)
        assert toggle.expanded and app.query_one(Turn).steps.display

    drive(app, script)
