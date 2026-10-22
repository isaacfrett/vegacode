"""Copy, paste and column dragging in the full-screen app: the system clipboard backends
(clipboard.py), Ctrl+C on a selection, /copy, bracketed paste into the composer, and the
workbench's draggable conversation edges."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual import events
from tui_helpers import answer, calls, drive, make_app, scripted_agent, send, wait_until

from vega import clipboard, server, view
from vega.tui import ColumnDivider, PromptInput, code_blocks

ORIGINAL_COMMAND = clipboard.command  # conftest stubs it out for every test


def project_at(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    (project / "src").mkdir(parents=True)
    (project / "src" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (project / "pyproject.toml").write_text("[project]\nname = 'calc'\n")
    return project


class FakeRun:
    """Stands in for subprocess.run: records each call, returns exit code 0."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        self.calls.append({"argv": argv, **kwargs})
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(argv, 0)


@pytest.fixture
def pbcopy(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    fake = FakeRun()
    monkeypatch.setattr(clipboard, "command", lambda *a, **k: ["pbcopy"])
    fake_subprocess(monkeypatch, fake)
    return fake


def fake_subprocess(monkeypatch: pytest.MonkeyPatch, fake: FakeRun) -> None:
    """Replace subprocess for clipboard.py only (the agent's own commands still run)."""
    monkeypatch.setattr(clipboard, "subprocess", SimpleNamespace(
        run=fake, DEVNULL=subprocess.DEVNULL, SubprocessError=subprocess.SubprocessError))


def copied(fake: FakeRun) -> list[str]:
    return [call["input"].decode("utf-8") for call in fake.calls]


def notices(app) -> str:
    return " ".join(str(n.message) for n in app._notifications)


# -- clipboard.py --------------------------------------------------------------------------------

def test_command_picks_the_platform_tool() -> None:
    def which(found: set[str]):
        return lambda name: f"/usr/bin/{name}" if name in found else None

    assert ORIGINAL_COMMAND("darwin", {}, which({"pbcopy"})) == ["pbcopy"]
    assert ORIGINAL_COMMAND("darwin", {}, which(set())) is None
    assert ORIGINAL_COMMAND("win32", {}, which({"clip.exe"})) == ["/usr/bin/clip.exe"]
    every = which({"wl-copy", "xclip", "xsel"})
    assert ORIGINAL_COMMAND("linux", {"WAYLAND_DISPLAY": "wayland-0"}, every) == ["wl-copy"]
    assert ORIGINAL_COMMAND("linux", {"DISPLAY": ":0"}, every) == \
        ["xclip", "-selection", "clipboard"]
    assert ORIGINAL_COMMAND("linux", {}, which({"xsel"})) == ["xsel", "--clipboard", "--input"]
    assert ORIGINAL_COMMAND("linux", {}, which({"clip.exe"})) == ["clip.exe"]  # WSL
    assert ORIGINAL_COMMAND("linux", {}, which(set())) is None


def test_copy_writes_stdin_with_a_timeout_and_never_raises(monkeypatch) -> None:
    fake = FakeRun()
    fake_subprocess(monkeypatch, fake)
    assert clipboard.copy("héllo\nworld", argv=["pbcopy"])
    call = fake.calls[0]
    assert call["argv"] == ["pbcopy"] and call["input"] == "héllo\nworld".encode()
    assert call["timeout"] == clipboard.TIMEOUT and call["env"]["LC_CTYPE"]
    assert clipboard.copy("hi", argv=["clip.exe"])
    assert fake.calls[1]["input"] == "hi".encode("utf-16")  # clip.exe reads UTF-16 with a BOM
    fake_subprocess(monkeypatch, FakeRun(subprocess.TimeoutExpired("pbcopy", 2)))
    assert clipboard.copy("hi", argv=["pbcopy"]) is False
    fake_subprocess(monkeypatch, FakeRun(FileNotFoundError("pbcopy")))
    assert clipboard.copy("hi", argv=["pbcopy"]) is False
    assert clipboard.copy("hi") is False  # no tool found (conftest)


def test_code_blocks_strips_fences() -> None:
    text = "Run this:\n\n```bash\npytest -q\n```\n\nand\n\n~~~\nx = 1\ny = 2\n~~~\n"
    assert code_blocks(text) == ["pytest -q", "x = 1\ny = 2"]
    assert code_blocks("no code here") == []


# -- copying in the app --------------------------------------------------------------------------

def test_ctrl_c_copies_a_mouse_selection_then_falls_back_to_stop(tmp_path: Path,
                                                                  pbcopy: FakeRun) -> None:
    project = project_at(tmp_path)
    agent, _ = scripted_agent(project, [answer("Hello world from Vega.")])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "hi")
        await wait_until(pilot, lambda: app.turn_list[0].answer.region.height > 0)
        region = app.turn_list[0].answer.region
        await pilot.mouse_down(offset=(region.x, region.y))
        await pilot.hover(offset=(region.x + 10, region.y))
        await pilot.mouse_up(offset=(region.x + 10, region.y))
        await pilot.pause()
        assert app.screen.get_selected_text() == "Hello world"
        await pilot.press("ctrl+c")
        await wait_until(pilot, lambda: bool(pbcopy.calls))
        await pilot.pause()
        assert copied(pbcopy) == ["Hello world"]
        assert app._clipboard == "Hello world"  # OSC 52 path (Textual's own) still runs
        assert "Copied 11 characters" in notices(app)
        assert not app.screen.selections  # cleared, so the next Ctrl+C means stop
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert "Nothing is running" in notices(app)
        assert len(pbcopy.calls) == 1

    drive(app, script, size=(140, 36))


def test_triple_click_selects_a_whole_reply(tmp_path: Path, pbcopy: FakeRun) -> None:
    project = project_at(tmp_path)
    reply = "First paragraph.\n\nSecond paragraph."
    agent, _ = scripted_agent(project, [answer(reply)])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "hi")
        await wait_until(pilot, lambda: app.turn_list[0].answer.region.height > 0)
        region = app.turn_list[0].answer.region
        await pilot.click(offset=(region.x + 2, region.y), times=3)
        await pilot.pause()
        selected = app.screen.get_selected_text() or ""
        assert "First paragraph." in selected and "Second paragraph." in selected

    drive(app, script, size=(140, 36))


def test_ctrl_c_copies_the_composer_selection(tmp_path: Path, pbcopy: FakeRun) -> None:
    project = project_at(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        prompt = app.query_one(PromptInput)
        prompt.load_text("draft message")
        prompt.select_all()
        await pilot.press("ctrl+c")
        await wait_until(pilot, lambda: bool(pbcopy.calls))
        assert copied(pbcopy) == ["draft message"]
        assert prompt.text == "draft message"

    drive(app, script, size=(140, 36))


def test_copy_command_copies_a_reply_or_its_code(tmp_path: Path, pbcopy: FakeRun) -> None:
    project = project_at(tmp_path)
    first = "Example output:\n\n```text\nhello there\n```\n"
    agent, _ = scripted_agent(project, [answer(first), answer("Plain second reply.")])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "/copy")
        assert "No reply to copy yet" in str(app.query(".note-body").last().render())
        await send(pilot, "hi")
        await send(pilot, "hi")
        for command in ("/copy", "/copy 1", "/copy code 1"):
            await send(pilot, command)
        await wait_until(pilot, lambda: len(pbcopy.calls) == 3)
        assert sorted(copied(pbcopy)) == sorted(["Plain second reply.", first, "hello there"])
        await send(pilot, "/copy code")
        assert "has no code block" in str(app.query(".note-body").last().render())
        await send(pilot, "/copy 9")
        assert "Turn 9 has no reply" in str(app.query(".note-body").last().render())
        await send(pilot, "/copy everything")
        assert "Usage: /copy" in str(app.query(".note-body").last().render())

    drive(app, script, size=(140, 36))


def test_without_a_clipboard_tool_the_notice_says_what_to_do(tmp_path: Path) -> None:
    project = project_at(tmp_path)
    agent, _ = scripted_agent(project, [answer("Some reply.")])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "hi")
        await send(pilot, "/copy")
        await wait_until(pilot, lambda: "terminal's clipboard" in notices(app))
        assert app._clipboard == "Some reply."

    drive(app, script, size=(140, 36))


# -- paste -----------------------------------------------------------------------------------------

def test_paste_into_the_composer_keeps_newlines_and_never_sends(tmp_path: Path) -> None:
    project = project_at(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)
    multi = "line one\nline two\n\nline four\n"
    large = "".join(f"row {n}: {'x' * 70}\n" for n in range(3000))  # ~230 KB

    async def script(pilot) -> None:
        prompt = app.query_one(PromptInput)
        assert prompt.has_focus
        app.post_message(events.Paste(multi))
        await pilot.pause()
        assert prompt.text == multi
        assert not app.turn_list and not app.working
        prompt.load_text("")
        app.post_message(events.Paste(large))
        await pilot.pause()
        assert prompt.text == large and not app.turn_list
        # a paste while a pane has focus lands in the message box
        prompt.load_text("")
        app.query_one("#wb-changes").focus()
        await pilot.pause()
        app.post_message(events.Paste("from a pane"))
        await pilot.pause()
        assert prompt.text == "from a pane" and prompt.has_focus

    drive(app, script, size=(140, 36))


# -- dragging the column edges ---------------------------------------------------------------------

async def drag(pilot, divider: ColumnDivider, dx: int, *, steps: int = 3) -> None:
    region = divider.region
    y = region.y + region.height // 2
    await pilot.mouse_down(offset=(region.x, y))
    for step in range(1, steps + 1):
        await pilot.hover(offset=(region.x + round(dx * step / steps), y))
        await pilot.pause()
    await pilot.mouse_up(offset=(region.x + dx, y))
    await pilot.pause()


def saved() -> dict[str, Any]:
    return json.loads((server.STATE_DIR / "view.json").read_text())


def test_dragging_the_left_edge_resizes_live_clamps_and_saves(tmp_path: Path) -> None:
    project = project_at(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        divider = app.query_one("#divider-left", ColumnDivider)
        assert divider.display and divider.region.x == app.view.left_width
        region = divider.region
        y = region.y + 5
        await pilot.mouse_down(offset=(region.x, y))
        await pilot.hover(offset=(region.x + 6, y))
        await pilot.pause()
        assert app.query_one("#left").outer_size.width == 36  # live, before release
        assert divider.dragging and "left column 36" in str(
            app.query_one("#conversation").border_subtitle)
        assert not (server.STATE_DIR / "view.json").exists()  # saved on release only
        await pilot.mouse_up(offset=(region.x + 6, y))
        await pilot.pause()
        assert saved()["left_width"] == 36 and app.view.preset == "custom"
        assert "saved" in notices(app)
        # far past the limit: clamped to view.LEFT_WIDTHS[1]
        await drag(pilot, divider, 80)
        assert app.view.left_width == view.LEFT_WIDTHS[1] == saved()["left_width"]
        assert app.query_one("#left").outer_size.width == view.LEFT_WIDTHS[1]
        await drag(pilot, divider, -60)
        assert app.view.left_width == view.LEFT_WIDTHS[0] == saved()["left_width"]
        assert not app.narrow

    drive(app, script, size=(200, 40))


def test_drag_never_squeezes_the_conversation_below_its_minimum(tmp_path: Path) -> None:
    project = project_at(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await drag(pilot, app.query_one("#divider-left", ColumnDivider), 60)
        # 140 columns, right column auto (counts as MIN_CONVERSATION): 140 - 46 - 44 = 50
        assert app.view.left_width == 140 - (view.MIN_CONVERSATION + 2) - view.MIN_CONVERSATION
        assert not app.narrow
        assert app.query_one("#conversation").outer_size.width >= view.MIN_CONVERSATION

    drive(app, script, size=(140, 36))


def test_right_edge_drag_and_double_click_reset(tmp_path: Path) -> None:
    project = project_at(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        divider = app.query_one("#divider-right", ColumnDivider)
        assert app.view.right_width == 0  # auto
        start = app.column_width("right")
        await drag(pilot, divider, -10)  # leftwards widens the right column
        assert app.view.right_width == start + 10 == saved()["right_width"]
        await pilot.click(divider, offset=(0, 5), times=2)
        await pilot.pause()
        assert app.view.right_width == 0 and saved()["right_width"] == 0
        assert "reset to auto" in notices(app)
        left = app.query_one("#divider-left", ColumnDivider)
        await drag(pilot, left, 8)
        assert app.view.left_width == 38
        await pilot.click(left, offset=(0, 5), times=2)
        await pilot.pause()
        assert app.view.left_width == view.View().left_width == saved()["left_width"]

    drive(app, script, size=(160, 36))


def test_edges_hide_when_narrow_hidden_or_another_layout(tmp_path: Path) -> None:
    project = project_at(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    def shown() -> tuple[bool, bool]:
        return (app.query_one("#divider-left").display, app.query_one("#divider-right").display)

    async def script(pilot) -> None:
        assert shown() == (True, True)
        conversation = app.query_one("#conversation").region
        assert app.query_one("#divider-left").region.x == conversation.x  # on its border
        assert app.query_one("#divider-right").region.right == conversation.right
        await pilot.press("ctrl+p")  # wide: hides the columns
        await pilot.pause()
        assert shown() == (False, False)
        await pilot.press("ctrl+p")
        await pilot.pause()
        await send(pilot, "/view review")  # changes, git, problems | conversation | preview
        assert shown() == (True, True)
        await send(pilot, "/pane git hide")
        await send(pilot, "/pane problems hide")
        await send(pilot, "/pane changes hide")
        assert shown() == (False, True)
        await pilot.resize_terminal(80, 30)
        await pilot.pause()
        assert app.narrow and shown() == (False, False)
        await pilot.resize_terminal(160, 36)
        await pilot.pause()
        assert shown() == (False, True)
        await send(pilot, "/layout classic")
        assert shown() == (False, False)

    drive(app, script, size=(160, 36))


def test_while_working_ctrl_c_copies_a_selection_and_esc_still_stops(tmp_path: Path,
                                                                     pbcopy: FakeRun) -> None:
    project = project_at(tmp_path)
    released = threading.Event()
    count = {"n": 0}

    def gate() -> None:
        count["n"] += 1
        if count["n"] == 2:
            released.wait(10)

    agent, _ = scripted_agent(project, [
        calls(("read_file", {"path": "src/calc.py"})), answer("done"), answer("done"),
    ], gate=gate)
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def select_message(pilot) -> None:
        region = app.query_one(".user-text").region
        await pilot.mouse_down(offset=(region.x, region.y))
        await pilot.hover(offset=(region.x + 3, region.y))
        await pilot.mouse_up(offset=(region.x + 3, region.y))
        await pilot.pause()

    async def script(pilot) -> None:
        await send(pilot, "look around", wait=False)
        await wait_until(pilot, lambda: count["n"] == 2)
        await select_message(pilot)
        await pilot.press("ctrl+c")  # a selection: copied, the turn keeps running
        await wait_until(pilot, lambda: bool(pbcopy.calls))
        assert copied(pbcopy) == ["look"] and not agent.stop_requested
        await select_message(pilot)
        await pilot.press("escape")  # Esc always stops, selection or not
        await pilot.pause(0.1)
        assert agent.stop_requested and len(pbcopy.calls) == 1
        released.set()
        await wait_until(pilot, lambda: not app.working)

    drive(app, script, size=(140, 36))
