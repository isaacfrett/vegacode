"""Scripted two-turn session rendered at 120x36 and 80x24 in both themes (SVG screenshots).

Writes to tmp_path; with VEGA_WRITE_SCREENS=1 it refreshes the committed docs/screens/ instead.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
from switches import switch_off
from tui_helpers import answer, calls, drive, make_app, scripted_agent, send, wait_until

from vega import sandbox
from vega.tui import ChoiceScreen, StepsToggle

SCREENS = Path(__file__).resolve().parents[1] / "docs" / "screens"
CALC = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"


def session_replies() -> list:
    return [
        calls(("read_file", {"path": "src/calc.py"}), prompt_tokens=3100),
        calls(("edit_file", {"path": "src/calc.py", "old_text": "return a - b",
                             "new_text": "return a + b"}), prompt_tokens=3600),
        calls(("run_check", {"command": "pytest -q"}), prompt_tokens=4100),
        answer("Fixed `add()` in `src/calc.py`: it now returns `a + b`. Tests pass "
               "(`pytest -q`, 3 passed).", prompt_tokens=4500),
        calls(("update_plan", {"items": [
            {"text": "add subtract() to src/calc.py", "status": "in_progress"},
            {"text": "add a test for it", "status": "pending"},
            {"text": "run the tests", "status": "pending"},
        ]}), prompt_tokens=5600),
        calls(("edit_file", {"path": "src/calc.py", "old_text": "def mul(a, b):",
                             "new_text": "def sub(a, b):\n    return a - b\n\n\ndef mul(a, b):"}),
              prompt_tokens=6300),
        calls(("update_plan", {"items": [
            {"text": "add subtract() to src/calc.py", "status": "done"},
            {"text": "add a test for it", "status": "in_progress"},
            {"text": "run the tests", "status": "pending"},
        ]}), ("write_file", {"path": "tests/test_sub.py",
                             "content": "from calc import sub\n\n\ndef test_sub():\n"
                                        "    assert sub(5, 3) == 2\n"}), prompt_tokens=7000),
        calls(("run_check", {"command": "pytest -q tests/test_sub.py"}), prompt_tokens=7600),
        calls(("update_plan", {"items": [
            {"text": "add subtract() to src/calc.py", "status": "done"},
            {"text": "add a test for it", "status": "done"},
            {"text": "run the tests", "status": "done"},
        ]}), prompt_tokens=8000),
        answer("Added `sub(a, b)` to `src/calc.py` and a test in `tests/test_sub.py`. The new "
               "test fails to import `calc` because `src/` is not on the path; add a "
               "`conftest.py` or install the package before running it.", prompt_tokens=8500),
    ]


@pytest.fixture(autouse=True)
def agent_loop_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scripted replies drive the agent loop; the pipelines that would take these short
    tasks first expect a different reply shape and would print their fallback note in the shot
    (README screenshots, 2026-10-21)."""
    switch_off(monkeypatch, "direct_pipeline", "deliverable_pipeline", "project_pipeline",
               "web_pipeline", "clarify_first")


def fake_check(command: str, project: Path, **_: object) -> str:
    if "test_sub" in command:
        return ("exit code: 2\nstdout:\nE   ModuleNotFoundError: No module named 'calc'\n"
                "1 error in 0.05s\n")
    return "exit code: 0\nstdout:\n...\n3 passed in 0.02s\n"


@pytest.mark.parametrize("size", [(120, 36), (80, 24)], ids=["120x36", "80x24"])
def test_screenshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                     size: tuple[int, int]) -> None:
    home = tmp_path.resolve()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    project = home / "projects" / "todo-app"
    (project / "src").mkdir(parents=True)
    (project / "src" / "calc.py").write_text(CALC)
    (project / "pyproject.toml").write_text("[project]\nname = 'calc'\n")
    monkeypatch.setattr(sandbox, "run_check", fake_check)
    monkeypatch.setenv("VEGA_MODEL", "granite-8b")
    out = SCREENS if os.environ.get("VEGA_WRITE_SCREENS") == "1" else tmp_path / "screens"
    out.mkdir(parents=True, exist_ok=True)
    agent, _ = scripted_agent(project, session_replies())
    app = make_app(project, agent, tmp_path)
    width, height = size

    async def script(pilot) -> None:
        await send(pilot, "fix the bug in add() in src/calc.py")
        await send(pilot, "Add a subtract function:\n- add sub(a, b) to src/calc.py\n"
                          "- add a test for it\n- run the tests")
        toggles = list(app.query(StepsToggle))
        assert toggles[-1].expanded  # the failed check keeps the second turn's steps open
        app.query_one("#prompt").load_text("")
        await pilot.pause()
        await wait_until(pilot, lambda: app.memory_mb is not None)
        for theme in ("dark", "light"):
            if (theme == "light") != (app.theme == "vega-light"):
                await pilot.press("ctrl+t")
            await pilot.pause()
            app.save_screenshot(f"session-{width}x{height}-{theme}.svg", path=str(out))
        if width < 100:  # the side panel as an overlay
            await pilot.press("ctrl+p")
            await pilot.pause()
            app.save_screenshot(f"panel-overlay-{width}x{height}-light.svg", path=str(out))

    drive(app, script, size=size)
    names = sorted(path.name for path in out.glob(f"*{width}x{height}*.svg"))
    assert f"session-{width}x{height}-dark.svg" in names
    assert f"session-{width}x{height}-light.svg" in names


def test_dialog_screenshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Working state and the edit review modal (120x36 dark), the command modal (80x24 light)."""
    home = tmp_path.resolve()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("VEGA_MODEL", "granite-8b")
    monkeypatch.setattr(sandbox, "run_check", fake_check)
    project = home / "projects" / "todo-app"
    (project / "src").mkdir(parents=True)
    (project / "src" / "calc.py").write_text(CALC)
    (project / "pyproject.toml").write_text("[project]\nname = 'calc'\n")
    out = SCREENS if os.environ.get("VEGA_WRITE_SCREENS") == "1" else tmp_path / "screens"
    out.mkdir(parents=True, exist_ok=True)
    held = threading.Event()
    count = {"n": 0}

    def gate() -> None:
        count["n"] += 1
        if count["n"] == 2:
            held.wait(10)

    replies = [
        calls(("read_file", {"path": "src/calc.py"}), prompt_tokens=3100),
        calls(("edit_file", {"path": "src/calc.py", "old_text": "return a - b",
                             "new_text": "return a + b"}), prompt_tokens=3600),
        calls(("run_command", {"command": "make release VERSION=0.3.0"}), prompt_tokens=4100),
        answer("Fixed `add()`; you chose not to run the release.", prompt_tokens=4500),
    ]
    agent, _ = scripted_agent(project, replies, gate=gate)
    app = make_app(project, agent, tmp_path, review=True)

    async def script(pilot) -> None:
        await send(pilot, "fix the bug in add() in src/calc.py, then cut a release", wait=False)
        await wait_until(pilot, lambda: count["n"] == 2)
        await pilot.pause(1.2)
        app.save_screenshot("working-120x36-dark.svg", path=str(out))
        held.set()
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        await pilot.pause()
        app.save_screenshot("review-120x36-dark.svg", path=str(out))
        await pilot.press("y")
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen)
                         and app.screen.has_class("command"))
        await pilot.press("ctrl+t")
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        app.save_screenshot("command-80x24-light.svg", path=str(out))
        await pilot.press("n")
        await wait_until(pilot, lambda: not app.working)

    drive(app, script, size=(120, 36))
    assert (out / "review-120x36-dark.svg").exists() and (out / "command-80x24-light.svg").exists()
