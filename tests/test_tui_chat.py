"""The chat-first default (2026-10-24): one conversation column in the terminal's own colours,
panels opt-in from the / menu and remembered, no window resizing, NO_COLOR respected."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from test_tui_screens import CALC, fake_check
from tui_helpers import answer, calls, drive, scripted_agent, send, wait_until

from vega import sandbox, server, tui, view
from vega.transcripts import SessionLog
from vega.tui import (
    TERMINAL_THEMES,
    PromptInput,
    VegaApp,
    colorfgbg_appearance,
    menu_matches,
    options_as_lists,
    osc_appearance,
)


@pytest.fixture(autouse=True)
def new_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests describe what a new user gets (conftest starts older tests in the workbench)."""
    monkeypatch.setattr(view, "NEW_USER_PRESET", "chat")
    for name in ("VEGA_LAYOUT", "VEGA_THEME", "COLORFGBG", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)


def project_at(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    (project / "src").mkdir(parents=True)
    (project / "src" / "calc.py").write_text(CALC)
    (project / "pyproject.toml").write_text("[project]\nname = 'calc'\n")
    return project


def chat_app(project: Path, tmp_path: Path, replies: list | None = None, **options) -> VegaApp:
    """The app as `vega` starts it: no layout or theme passed in."""
    agent, _ = scripted_agent(project, replies or [])
    return VegaApp(project, agent=agent, session_log=SessionLog(tmp_path / "sessions", project),
                   warm_up=False, memory_probe=lambda: 5.1 * 1024, total_memory_gb=16.0,
                   **options)


# -- never resize the window ---------------------------------------------------------------------

def test_run_tui_never_writes_a_window_resize(monkeypatch: pytest.MonkeyPatch) -> None:
    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return 1

    out = Tty()
    started = []

    class FakeApp:
        def __init__(self, *args, **kwargs) -> None:
            started.append(kwargs)

        def run(self) -> None:
            pass

    monkeypatch.setattr(tui.sys, "__stdout__", out)
    monkeypatch.setattr(tui, "VegaApp", FakeApp)
    monkeypatch.setattr(tui, "query_terminal_appearance", lambda: None)
    tui.run_tui(Path("."))
    assert started and "\x1b[8;" not in out.getvalue()
    source = Path(tui.__file__).read_text(encoding="utf-8")
    assert "[8;" not in source and "_resize_window" not in source


def test_appearance_is_asked_only_when_the_terminal_theme_needs_it(monkeypatch) -> None:
    assert tui.needs_appearance(view.preset("chat"))
    assert not tui.needs_appearance(view.View(theme="classic"))
    assert not tui.needs_appearance(view.View(appearance="light"))
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert not tui.needs_appearance(view.preset("chat"))
    assert colorfgbg_appearance() == "light"
    assert colorfgbg_appearance("15;0") == "dark" and colorfgbg_appearance("x") is None
    assert osc_appearance("\x1b]11;rgb:ffff/ffff/ffff\x07") == "light"
    assert osc_appearance("\x1b]11;rgb:0000/0000/0000\x1b\\") == "dark"
    assert osc_appearance("") is None


# -- the default screen --------------------------------------------------------------------------

def test_default_is_chat_only_in_the_terminal_colours_with_a_one_time_hint(tmp_path: Path) -> None:
    project = project_at(tmp_path)
    app = chat_app(project, tmp_path)

    async def script(pilot) -> None:
        assert app.layout_name == "chat" and app.has_class("layout-chat")
        assert app.theme == "vega-terminal-dark" and app.current_theme.ansi
        assert app.has_class("ansi") and app.has_class("style-rail")
        for hidden in ("#header", "#left", "#right", "#side", "#menu"):
            assert not app.query_one(hidden).display, hidden
        assert app.query_one("#conversation").display and app.query_one("#prompt").has_focus
        hint = " ".join(str(n.query_one(".note-body").render()) for n in app.query(".note.hint"))
        assert "/ commands, panels and themes" in hint and "Esc stops Vega" in hint
        status = str(app.query_one("#readouts").render())
        assert status.startswith("ready") and "~" not in status or "proj" in status
        assert str(app.query_one("#hints").render()) == "/ commands"

    drive(app, script, size=(100, 30))
    assert view.load().hinted  # shown once
    again = chat_app(project, tmp_path)

    async def second(pilot) -> None:
        assert not list(again.query(".note.hint"))

    drive(again, second, size=(100, 30))


@pytest.mark.parametrize("size", [(80, 24), (120, 40)], ids=["80x24", "120x40"])
@pytest.mark.parametrize("appearance", ["dark", "light"])
def test_chat_fits_without_horizontal_overflow(tmp_path: Path, monkeypatch, size, appearance):
    project = project_at(tmp_path)
    monkeypatch.setattr(sandbox, "run_check", fake_check)
    replies = [calls(("read_file", {"path": "src/calc.py"}), prompt_tokens=3100),
               calls(("run_check", {"command": "pytest -q"}), prompt_tokens=3600),
               answer("Read `src/calc.py`; `pytest -q` passes.\n\n```python\n"
                      "def add(a, b):\n    return a + b\n```", prompt_tokens=4000)]
    app = chat_app(project, tmp_path, replies, appearance=appearance)

    async def script(pilot) -> None:
        await send(pilot, "check the calculator")
        await pilot.pause()
        conversation = app.query_one("#conversation")
        assert conversation.max_scroll_x == 0
        assert not app.narrow
        line = str(app.query_one("#readouts").render())
        assert len(line) + len(str(app.query_one("#hints").render())) <= size[0]
        svg = app.export_screenshot()
        assert "check&#160;the&#160;calculator" in svg
        expected = "vega-terminal-light" if appearance == "light" else "vega-terminal-dark"
        assert app.theme == expected

    drive(app, script, size=size)


# -- panels, the / menu, themes -------------------------------------------------------------------

def test_panel_toggles_open_close_and_persist(tmp_path: Path) -> None:
    project = project_at(tmp_path)
    app = chat_app(project, tmp_path)

    async def script(pilot) -> None:
        await send(pilot, "/panel changes")
        await pilot.pause()
        assert app.view.left == ("changes",) and app.query_one("#left").display
        await send(pilot, "/panel preview")
        await pilot.pause()
        assert app.view.right == ("preview",) and app.query_one("#preview").display
        await send(pilot, "/panel changes")
        await pilot.pause()
        assert app.view.left == () and not app.query_one("#left").display

    drive(app, script, size=(140, 40))
    saved = json.loads((server.STATE_DIR / "view.json").read_text())
    assert saved["right"] == ["preview"] and saved["left"] == [] and saved["version"] == 2
    again = chat_app(project, tmp_path)

    async def second(pilot) -> None:
        assert again.layout_name == "chat" and again.query_one("#preview").display

    drive(again, second, size=(140, 40))


def test_slash_menu_filters_fills_runs_and_closes(tmp_path: Path) -> None:
    project = project_at(tmp_path)
    app = chat_app(project, tmp_path)

    async def script(pilot) -> None:
        prompt = app.query_one(PromptInput)
        menu = app.query_one("#menu")
        await pilot.press("slash")
        await pilot.pause()
        assert app.menu_open and menu.display
        assert [e.command for e in menu.entries[:3]] == ["/panel", "/theme", "/view"]
        await pilot.press("enter")  # /panel takes a name: Enter fills it in
        assert prompt.text == "/panel " and app.menu_open
        assert all(e.command.startswith("/panel ") for e in menu.entries)
        await pilot.press("p", "l", "a")
        await pilot.press("enter")
        await pilot.pause()
        assert app.view.left == ("plan",) and prompt.text == "" and not app.menu_open
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert prompt.text == "/" and app.menu_open
        await pilot.press("escape")
        assert not app.menu_open and prompt.text == "/"
        prompt.load_text("")
        await pilot.press("slash", "t", "h", "e", "m", "e", "space", "q", "u")
        await pilot.pause()
        assert app.theme == "vega-quiet-dark"  # previewed behind the menu
        await pilot.press("escape")
        assert app.theme == "vega-terminal-dark"  # and put back on leaving
        prompt.load_text("")
        await pilot.press("slash", "u", "n", "d")
        await pilot.press("tab")
        assert prompt.text == "/undo "

    drive(app, script, size=(120, 36))


def test_menu_matching_and_typed_commands_still_send() -> None:
    entries = [tui.MenuEntry("/panel", "open a panel"), tui.MenuEntry("/review", "ask first"),
               tui.MenuEntry("/panel git", "Git: branch", top=False)]
    assert [e.command for e in menu_matches(entries, "/")] == ["/panel", "/review"]
    assert [e.command for e in menu_matches(entries, "/pa")] == ["/panel", "/panel git"]
    assert [e.command for e in menu_matches(entries, "/branch")] == ["/panel git"]


def test_theme_and_style_commands_persist(tmp_path: Path) -> None:
    project = project_at(tmp_path)
    app = chat_app(project, tmp_path)

    async def script(pilot) -> None:
        await send(pilot, "/theme")
        notes = " ".join(str(n.query_one(".note-body").render()) for n in app.query(".note"))
        assert "terminal" in notes and "observatory" in notes and "current" in notes
        await send(pilot, "/theme workbench")
        assert app.theme == "vega-workbench-dark" and not app.has_class("ansi")
        await send(pilot, "/theme light")
        assert app.theme == "vega-workbench-light"
        await send(pilot, "/style bubbles")
        assert app.has_class("style-bubbles") and not app.has_class("style-rail")

    drive(app, script, size=(100, 30))
    saved = view.load()
    assert (saved.theme, saved.appearance, saved.chat_style) == ("workbench", "light", "bubbles")


def test_no_color_keeps_the_terminal_colours_without_colour(tmp_path: Path, monkeypatch) -> None:
    from textual.filter import Monochrome, NoColor

    view.save(view.View(theme="observatory"))
    monkeypatch.setenv("NO_COLOR", "1")
    project = project_at(tmp_path)
    app = chat_app(project, tmp_path)
    assert app.theme_family == "terminal" and app.current_theme.ansi
    assert any(isinstance(f, NoColor) for f in app._filters)
    assert not any(isinstance(f, Monochrome) for f in app._filters)

    async def script(pilot) -> None:
        await send(pilot, "/theme classic")
        assert app.theme_family == "terminal"

    drive(app, script, size=(100, 30))


# -- existing users ------------------------------------------------------------------------------

def test_a_view_json_from_before_chat_is_migrated_once(tmp_path: Path) -> None:
    old = {"left": ["changes", "git"], "right": ["preview"], "preset": "workbench",
           "teach": True, "steps": "expanded", "mode": "code"}
    (server.STATE_DIR / "view.json").write_text(json.dumps(old))
    project = project_at(tmp_path)
    app = chat_app(project, tmp_path)

    async def script(pilot) -> None:
        assert app.layout_name == "chat" and app.view.left == () and app.view.right == ()
        assert app.view.teach and app.view.steps == "expanded"
        notes = " ".join(str(n.query_one(".note-body").render()) for n in app.query(".note"))
        assert "/view workbench" in notes and "view-v1.json" in notes
        await send(pilot, "/view workbench")
        await pilot.pause()
        assert app.layout_name == "workbench" and app.view.left[:2] == ("changes", "git")

    drive(app, script, size=(140, 40))
    assert json.loads((server.STATE_DIR / "view-v1.json").read_text()) == old
    assert view.load().layout == "workbench"  # the choice is kept
    assert view.migrated is False


# -- colours and display helpers -----------------------------------------------------------------

def _luminance(rgb: tuple[int, int, int]) -> float:
    def channel(value: int) -> float:
        c = value / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# macOS Terminal's ANSI table (Basic and Pro share it); the worst case for accents
MAC = {"black": (0, 0, 0), "red": (0x99, 0, 0), "green": (0, 0xA6, 0), "yellow": (0x99, 0x99, 0),
       "blue": (0, 0, 0xB2), "magenta": (0xB2, 0, 0xB2), "cyan": (0, 0xA6, 0xB2),
       "white": (0xBF, 0xBF, 0xBF), "bright_black": (0x66, 0x66, 0x66),
       "bright_red": (0xE5, 0, 0)}


@pytest.mark.parametrize("mode,background", [("dark", (0, 0, 0)), ("light", (255, 255, 255))])
def test_terminal_theme_accents_read_on_the_stock_mac_palettes(mode: str, background) -> None:
    theme = TERMINAL_THEMES[mode]
    assert theme.ansi and theme.foreground == theme.background == "ansi_default"
    for colour in (theme.accent, theme.error, theme.variables["vega-warning-text"]):
        name = colour.removeprefix("ansi_")
        assert _contrast(MAC[name], background) >= 4.0, (mode, colour)
    muted = theme.variables["vega-muted"].removeprefix("ansi_")
    assert _contrast(MAC[muted], background) >= 3.0  # secondary text and rules
    assert tui.rich_colour("ansi_bright_black") == "bright_black"


def test_clarify_options_are_shown_one_per_line() -> None:
    text = "Pick:\n\n1. What?\n   a) one\n   b) two\n2. Where?\n   a) here\n"
    shown = options_as_lists(text)
    assert "   - a) one\n   - b) two" in shown and "   - a) here" in shown
    assert options_as_lists("1. a list\n2. only") == "1. a list\n2. only"


def test_stopping_and_working_state_in_the_status_line(tmp_path: Path) -> None:
    import threading

    project = project_at(tmp_path)
    release = threading.Event()
    agent, _ = scripted_agent(project, [answer("done")], gate=release.wait)
    app = VegaApp(project, agent=agent, session_log=SessionLog(tmp_path / "s", project),
                  warm_up=False, memory_probe=lambda: 100.0, total_memory_gb=16.0)

    async def script(pilot) -> None:
        await send(pilot, "hello", wait=False)
        await wait_until(pilot, lambda: "working" in str(app.query_one("#readouts").render()))
        assert str(app.query_one("#hints").render()) == "esc stop"
        release.set()
        await wait_until(pilot, lambda: not app.working)

    drive(app, script, size=(100, 30))
