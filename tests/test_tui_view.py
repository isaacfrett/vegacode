"""Workbench customization: the view settings (view.py), /view /pane /width /steps, and F2."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_tui_options import demo_project
from test_tui_screens import session_replies
from tui_helpers import drive, make_app, scripted_agent, send, wait_until

from vega import server, view
from vega.tui import StepsToggle, ViewScreen, layout_from_env

SCREENS = Path(__file__).resolve().parents[1] / "docs" / "screens" / "view"


def project_at(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    (project / "src").mkdir(parents=True)
    (project / "src" / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (project / "pyproject.toml").write_text("[project]\nname = 'calc'\n")
    return project


def ids(column) -> list[str]:
    return [child.id for child in column.children]


# -- settings ------------------------------------------------------------------------------------

def test_workbench_is_the_default_layout(monkeypatch) -> None:
    monkeypatch.delenv("VEGA_LAYOUT", raising=False)
    assert layout_from_env() == "workbench"


def test_validate_drops_unknown_and_duplicate_panes_and_clamps_widths() -> None:
    v = view.validate({"left": ["steps", "nope", "steps"], "right": ["steps", "plan"],
                       "left_width": 500, "right_width": 3, "steps": "loud",
                       "status": ["keys", "gauge", "x"]})
    assert v.left == ("steps",) and v.right == ("plan",)
    assert v.left_width == view.LEFT_WIDTHS[1] and v.right_width == view.RIGHT_WIDTHS[0]
    assert v.steps == "collapsed" and v.status == ("gauge", "keys")
    assert view.validate({"right_width": 0}).right_width == 0  # auto


def test_save_and_load_round_trip_and_bad_file_falls_back() -> None:
    custom = view.place(view.View(), "machine", "right")
    view.save(custom)
    assert view.load() == custom
    (server.STATE_DIR / "view.json").write_text("{not json")
    assert view.load() == view.View()


def test_place_shift_and_wide_at() -> None:
    v = view.place(view.View(), "plan", "left")
    assert v.left == ("changes", "git", "steps", "loops", "plan") and v.preset == "custom"
    v = view.shift(v, "plan", -1)
    assert v.left == ("changes", "git", "steps", "plan", "loops")
    v = view.place(v, "plan", "hide")
    assert v.column("plan") is None
    assert view.preset("focus").wide_at() < view.View().wide_at()


# -- in the app ----------------------------------------------------------------------------------

def test_pane_command_moves_panes_live_and_saves(tmp_path: Path) -> None:
    project = project_at(tmp_path)
    agent, _ = scripted_agent(project, session_replies())
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        assert ids(app.query_one("#left")) == ["wb-changes", "wb-git", "wb-steps", "wb-loops"]
        assert ids(app.query_one("#right")) == ["preview"]
        await send(pilot, "one")
        await send(pilot, "/pane machine right")
        await send(pilot, "/pane machine up")
        await pilot.pause(0.2)
        assert ids(app.query_one("#right")) == ["wb-machine", "preview"]
        machine = str(app.query_one("#wb-machine-body").render())
        assert "context" in machine and "memory" in machine
        # the preview still shows turn 1's change after the rebuild
        assert "src/calc.py" in str(app.query_one("#preview").border_title)
        await send(pilot, "/width left 40")
        await pilot.pause()
        assert app.query_one("#left").outer_size.width == 40
        saved = json.loads((server.STATE_DIR / "view.json").read_text())
        assert saved["right"] == ["machine", "preview"] and saved["left_width"] == 40
        await send(pilot, "/pane bogus left")
        notes = " ".join(str(n.query_one(".note-body").render()) for n in app.query(".note"))
        assert "Usage: /pane" in notes

    drive(app, script, size=(140, 40))


def test_focus_preset_hides_both_columns_and_tab_order_follows_the_view(tmp_path: Path) -> None:
    project = project_at(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "/view focus")
        await pilot.pause(0.2)
        assert not app.query_one("#left").display and not app.query_one("#right").display
        await send(pilot, "/view monitor")
        await pilot.pause(0.2)
        assert ids(app.query_one("#left")) == ["wb-plan", "wb-steps", "wb-loops"]
        assert ids(app.query_one("#right")) == ["wb-machine", "preview"]
        assert app.query_one("#left").display and app.query_one("#right").display
        await pilot.press("tab")
        assert app.focused is app.query_one("#wb-plan")

    drive(app, script, size=(150, 40))


def test_f2_changes_show_live_esc_restores_enter_keeps(tmp_path: Path) -> None:
    project = project_at(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await pilot.press("f2")
        assert isinstance(app.screen, ViewScreen)
        await pilot.press("left")  # Changes: left is already the first place
        await pilot.press("right")  # Changes -> right column
        await pilot.pause(0.2)
        assert ids(app.query_one("#right")) == ["preview", "wb-changes"]
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert ids(app.query_one("#left")) == ["wb-changes", "wb-git", "wb-steps", "wb-loops"]
        assert not (server.STATE_DIR / "view.json").exists()
        await pilot.press("f2")
        await pilot.press("3")  # review preset
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert app.view == view.preset("review")
        assert json.loads((server.STATE_DIR / "view.json").read_text())["preset"] == "review"

    drive(app, script, size=(140, 40))


def test_steps_hidden_and_status_items(tmp_path: Path) -> None:
    project = project_at(tmp_path)
    agent, _ = scripted_agent(project, session_replies())
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "/steps hidden")
        await send(pilot, "one")
        await wait_until(pilot, lambda: not app.working)
        turn = app.turn_list[-1]
        assert not turn.steps.display and not turn.query_one(StepsToggle).display
        await send(pilot, "/steps expanded")
        await pilot.pause()
        assert turn.steps.display and turn.query_one(StepsToggle).expanded
        app.apply_view(view.toggle_status(app.view, "keys"))
        await pilot.pause()
        assert str(app.query_one("#hints").render()) == ""

    drive(app, script, size=(140, 40))


def test_saved_view_is_used_at_start(tmp_path: Path) -> None:
    view.save(view.preset("steps"))
    project = project_at(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        assert not app.query_one("#left").display
        assert ids(app.query_one("#right")) == ["wb-steps", "preview"]
        assert app.query_one("#right").outer_size.width == 56

    drive(app, script, size=(140, 40))


# -- screenshots ---------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(view.PRESETS))
def test_preset_screens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Each preset on the scripted two-turn session at 140x40 dark, plus the F2 dialog once.
    Writes to tmp_path; with VEGA_WRITE_SCREENS=1 it refreshes docs/screens/view/ instead."""
    out = SCREENS if os.environ.get("VEGA_WRITE_SCREENS") == "1" else tmp_path / "view"
    out.mkdir(parents=True, exist_ok=True)
    project = demo_project(tmp_path, monkeypatch)
    view.save(view.preset(name))
    agent, _ = scripted_agent(project, session_replies())
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False,
                   swap_probe=lambda: 1.2 * 1024)

    async def script(pilot) -> None:
        await send(pilot, "fix the bug in add() in src/calc.py")
        await send(pilot, "Add a subtract function:\n- add sub(a, b) to src/calc.py\n"
                          "- add a test for it\n- run the tests")
        app.query_one("#prompt").load_text("")
        await wait_until(pilot, lambda: app.memory_mb is not None)
        await pilot.pause(0.3)
        app.save_screenshot(f"preset-{name}.svg", path=str(out))
        if name == "workbench":
            await pilot.press("f2")
            await pilot.press("down", "down", "down")  # Plan
            await pilot.pause(0.3)
            app.save_screenshot("customize-f2.svg", path=str(out))

    drive(app, script, size=(140, 40))
    assert (out / f"preset-{name}.svg").exists()


def test_window_fills_the_screen_only_on_a_real_terminal(monkeypatch) -> None:
    import io

    from vega import tui

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    for name in ("VEGA_WINDOW", "TMUX", "STY"):
        monkeypatch.delenv(name, raising=False)
    assert tui._window_resizable(Tty())
    assert not tui._window_resizable(io.StringIO())  # piped
    monkeypatch.setenv("VEGA_WINDOW", "keep")
    assert not tui._window_resizable(Tty())
    monkeypatch.delenv("VEGA_WINDOW")
    monkeypatch.setenv("TMUX", "/tmp/tmux-1/default,1,0")
    assert not tui._window_resizable(Tty())
    out = Tty()
    tui._resize_window(40, 120, out)
    assert out.getvalue() == "\x1b[8;40;120t"


def test_waits_for_the_terminal_to_finish_resizing() -> None:
    from types import SimpleNamespace

    from vega import tui

    t = {"now": 0.0}
    sizes = [(30, 100)] * 3 + [(50, 160)] * 2 + [(60, 220)] * 20  # grows in two steps, then settles

    def size(_fd):
        rows, cols = sizes.pop(0) if len(sizes) > 1 else sizes[0]
        return SimpleNamespace(lines=rows, columns=cols)

    def sleep(seconds: float) -> None:
        t["now"] += seconds

    final = tui._wait_for_new_size((30, 100), 1, size=size, sleep=sleep, clock=lambda: t["now"])
    assert final == (60, 220)
    # a terminal that ignores the request: gives up after the limit with the old size
    t["now"] = 0.0
    still = tui._wait_for_new_size((30, 100), 1, limit=0.3, size=lambda _fd: SimpleNamespace(lines=30, columns=100),
                                   sleep=sleep, clock=lambda: t["now"])
    assert still == (30, 100)
