"""The four layouts (classic, quiet, observatory, workbench) on the scripted two-turn session.

Each layout is rendered at 120x36 dark and 80x24 light, plus its edit-review dialog at 120x36 dark.
Writes to tmp_path; with VEGA_WRITE_SCREENS=1 it refreshes docs/screens/options/ instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from switches import switch_off
from test_tui_screens import CALC, fake_check, session_replies
from tui_helpers import answer, calls, drive, make_app, scripted_agent, send, wait_until

from vega import sandbox
from vega.tui import LAYOUTS, ChoiceScreen, Steps, StepsToggle

OPTIONS = Path(__file__).resolve().parents[1] / "docs" / "screens" / "options"


def screens_dir(tmp_path: Path) -> Path:
    out = OPTIONS if os.environ.get("VEGA_WRITE_SCREENS") == "1" else tmp_path / "options"
    out.mkdir(parents=True, exist_ok=True)
    return out


def demo_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path.resolve()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("VEGA_MODEL", "coder-7b")
    monkeypatch.setattr(sandbox, "run_check", fake_check)
    project = home / "projects" / "todo-app"
    (project / "src").mkdir(parents=True)
    (project / "src" / "calc.py").write_text(CALC)
    (project / "pyproject.toml").write_text("[project]\nname = 'calc'\n")
    return project


@pytest.mark.parametrize("layout", LAYOUTS)
def test_layout_session_screens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                layout: str) -> None:
    project = demo_project(tmp_path, monkeypatch)
    out = screens_dir(tmp_path)
    agent, _ = scripted_agent(project, session_replies())
    app = make_app(project, agent, tmp_path, layout=layout, transparent=False,
                   swap_probe=lambda: 1.2 * 1024)

    async def script(pilot) -> None:
        await send(pilot, "fix the bug in add() in src/calc.py")
        await send(pilot, "Add a subtract function:\n- add sub(a, b) to src/calc.py\n"
                          "- add a test for it\n- run the tests")
        app.query_one("#prompt").load_text("")
        await wait_until(pilot, lambda: app.memory_mb is not None)
        await pilot.pause(0.3)
        # the failed check opened the second turn's steps; the first turn's stay folded
        toggles = list(app.query(StepsToggle))
        assert toggles[-1].expanded and not toggles[0].expanded
        app.save_screenshot(f"{layout}-120x36-dark.svg", path=str(out))
        await pilot.press("ctrl+t")
        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.3)
        app._scroll_to_turn(app.turn_list[-1])
        await pilot.pause(0.3)
        app.save_screenshot(f"{layout}-80x24-light.svg", path=str(out))
        await pilot.press("ctrl+p")
        await pilot.pause(0.3)
        app.save_screenshot(f"{layout}-panel-80x24-light.svg", path=str(out))
        await pilot.press("escape")
        await pilot.pause()
        assert not app.default_screen.has_class("panel-overlay")

    drive(app, script, size=(120, 36))
    for name in (f"{layout}-120x36-dark.svg", f"{layout}-80x24-light.svg"):
        assert (out / name).exists()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_layout_review_screens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                               layout: str) -> None:
    project = demo_project(tmp_path, monkeypatch)
    # a one-target fix: routing would send it to the direct pipeline; this reviews agent-loop edits
    switch_off(monkeypatch, "direct_pipeline")
    out = screens_dir(tmp_path)
    replies = [
        calls(("read_file", {"path": "src/calc.py"}), prompt_tokens=3100),
        calls(("edit_file", {"path": "src/calc.py", "old_text": "return a - b",
                             "new_text": "return a + b"}), prompt_tokens=3600),
        answer("Fixed `add()`.", prompt_tokens=4100),
    ]
    agent, _ = scripted_agent(project, replies)
    app = make_app(project, agent, tmp_path, review=True, layout=layout, transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "fix the bug in add() in src/calc.py", wait=False)
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        await pilot.pause(0.3)
        assert str(app.screen.query_one("#dialog-title").render()) == "Edit src/calc.py?"
        assert "waiting for you" in str(app.query_one("#state").render())
        app.save_screenshot(f"{layout}-review-120x36-dark.svg", path=str(out))
        await pilot.press("y")
        await wait_until(pilot, lambda: not app.working)
        assert "return a + b" in (project / "src" / "calc.py").read_text()
        steps = app.query_one(Steps)
        assert [e.outcome for e in steps.entries if e.kind == "step"][1] == "exact match"

    drive(app, script, size=(120, 36))
    assert (out / f"{layout}-review-120x36-dark.svg").exists()
