"""The Loops pane in the workbench: rows with fixed status wording, the needs-you badge, findings on
Enter, evidence in the preview, accept/reject and pause from the keyboard."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from rich.text import Text
from test_tui_view import project_at
from tui_helpers import drive, make_app, scripted_agent, wait_until

from vega import tui_loops
from vega.loops import definitions, inbox, state, status
from vega.tui import glyphs

COLOURS = {"fg": "#ffffff", "muted": "#888888", "accent": "#00aaff", "error": "#ff0000",
           "warning": "#ffaa00"}


def make_loops(project: Path) -> dict:
    """Bug Watcher with one open finding, and a second loop that has never run."""
    path = definitions.ensure_default(project)
    assert path is not None
    nightly = definitions.user_dir() / "nightly.toml"
    nightly.write_text(path.read_text().replace('name = "Bug Watcher"', 'name = "Nightly Audit"')
                       .replace('trigger = { on = "commit", branch = "main" }',
                                'trigger = { every = "1d", at = "02:00" }'))
    finding = inbox.make_finding(
        loop="Bug Watcher", project=project, commit="9ed58d5aa", kind="bug",
        title="merge_code drops a decorator", where=[{"path": "src/calc.py", "line": 2}],
        evidence={"type": "failing_test", "path": "", "output": "E   assert 1 == 2\nFAILED t.py::t",
                  "fails_at": "9ed58d5aa", "passes_at": "976e102bb"})
    inbox.add(finding)
    return finding


@pytest.fixture
def quiet_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real ioreg/pmset/process scans in tests: the header says the daemon is stopped."""
    monkeypatch.setattr(status, "header", lambda rows, probe=None: status.Header(
        daemon_running=False, daemon_pid=None, screen_locked=False, model_allowed=False,
        reasons=("screen unlocked",), needs_you=sum(r.needs_you for r in rows)))


def test_a_row_keeps_name_and_status_whole() -> None:
    def row(name: str) -> status.LoopRow:
        return status.LoopRow(name=name, state="active", status="3 found", needs_you=3,
                              informational=False, trigger="on commit", last_run=None,
                              next_run=None, running_for=None, error="", project="", source="")

    wide = tui_loops.loop_option(row("Bug Watcher"), 30, COLOURS, glyphs(), 0, ascii_only=False)
    assert wide.plain == "● Bug Watcher (3 found)\n\u00a0\u00a0on commit"
    accent = [span for span in wide.spans if "bold" in str(span.style)]
    assert wide.plain[accent[0].start:accent[0].end] == "(3 found)"
    narrow = tui_loops.loop_option(row("Dependency Auditor"), 26, COLOURS, glyphs(), 0,
                                   ascii_only=False)
    assert narrow.plain == "● Dependency Auditor\n\u00a0\u00a0(3 found) · on commit"
    tiny = tui_loops.loop_option(row("A very long loop name for a narrow pane"), 20, COLOURS,
                                 glyphs(), 0, ascii_only=False)
    first, second = tiny.plain.split("\n")
    assert len(first) <= 20 and first.endswith("…") and second.startswith("\u00a0\u00a0(3 found)")


def test_a_failing_row_shows_its_error_and_ascii_symbols() -> None:
    row = status.LoopRow(name="Bug Watcher", state="failing", status="failing", needs_you=0,
                         informational=True, trigger="on commit", last_run=None, next_run=None,
                         running_for=None, error="git worktree failed: disk below 12 GB",
                         project="", source="")
    text = tui_loops.loop_option(row, 40, COLOURS, glyphs(), 0, ascii_only=True)
    assert text.plain.startswith("! Bug Watcher (failing)")
    assert "disk below 12 GB" in text.plain


def test_header_says_why_model_work_is_held() -> None:
    header = status.Header(daemon_running=True, daemon_pid=1, screen_locked=False,
                           model_allowed=False, reasons=("screen unlocked",), needs_you=0)
    assert isinstance(tui_loops.header_text(header, COLOURS, glyphs()), Text)
    assert tui_loops.header_text(header, COLOURS, glyphs()).plain == \
        "screen unlocked · model held: screen unlocked"
    stopped = status.Header(daemon_running=False, daemon_pid=None, screen_locked=True,
                            model_allowed=True, reasons=(), needs_you=0)
    assert tui_loops.header_text(stopped, COLOURS, glyphs()).plain == \
        "stopped: vega loops start"


def test_pane_rows_badge_findings_preview_and_keys(tmp_path: Path, quiet_probe: None) -> None:
    project = project_at(tmp_path)
    finding = make_loops(project)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    def pane_text() -> str:
        pane = app.query_one("#wb-loops")
        return "\n".join(str(pane.get_option_at_index(i).prompt) for i in range(pane.option_count))

    async def script(pilot) -> None:
        await wait_until(pilot, lambda: "Bug Watcher" in pane_text() and "(1 found)" in pane_text())
        text = pane_text()
        assert "stopped: vega loops start" in text
        assert "Nightly Audit" in text and "(not run yet)" in text and "nightly" in text
        assert "1" in str(app.query_one("#wb-loops").border_title)  # the needs-you badge
        if os.environ.get("VEGA_SCREENSHOTS"):
            app.save_screenshot("loops-120x36.svg", path=os.environ["VEGA_SCREENSHOTS"])
        pane = app.query_one("#wb-loops")
        pane.focus()
        pane.highlighted = 1  # Bug Watcher (row 0 is the header)
        await pilot.pause(0.1)
        assert "Bug Watcher" in str(app.query_one("#preview").border_title)
        await pilot.press("enter")
        await wait_until(pilot, lambda: "merge_code drops" in pane_text())  # clipped to the pane
        pane.highlighted = 2  # the finding
        await pilot.pause(0.1)
        preview = str(app.query_one("#preview-body").render())
        assert "a test that fails at 9ed58d5a, passes at 976e102b" in preview
        assert "assert 1 == 2" in preview
        if os.environ.get("VEGA_SCREENSHOTS"):
            app.save_screenshot("loops-finding-120x36.svg", path=os.environ["VEGA_SCREENSHOTS"])
        await pilot.press("x")
        await wait_until(pilot, lambda: inbox.state_of(inbox.get(finding["id"])) == "rejected")
        # nothing open; this loop has never recorded a run, so it does not claim "all clear"
        await wait_until(pilot, lambda: "(not run yet)" in pane_text().split("Nightly")[0])
        assert "Loops" == str(app.query_one("#wb-loops").border_title)  # badge gone
        pane.highlighted = 1
        await pilot.press("p")
        await wait_until(pilot, lambda: "(paused)" in pane_text())
        assert state.is_paused("Bug Watcher")

    drive(app, script, size=(120, 36))


def test_empty_pane_says_how_to_start(tmp_path: Path, quiet_probe: None) -> None:
    project = project_at(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        pane = app.query_one("#wb-loops")
        await wait_until(pilot, lambda: pane.option_count >= 2)
        assert "vega loops start" in str(pane.get_option_at_index(1).prompt)
        if os.environ.get("VEGA_SCREENSHOTS"):
            app.save_screenshot("loops-empty-120x36.svg", path=os.environ["VEGA_SCREENSHOTS"])

    drive(app, script, size=(120, 36))


def test_narrow_terminal_shows_loops_in_status_bar_and_panel(tmp_path: Path,
                                                             quiet_probe: None) -> None:
    project = project_at(tmp_path)
    make_loops(project)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        assert app.narrow  # 80 columns: one column, no Loops pane on screen
        await wait_until(pilot, lambda: "loops 1 waiting" in str(app.query_one("#readouts").render()))
        await pilot.press("ctrl+p")
        await pilot.pause(0.2)
        side = str(app.query_one("#loops-side").render())
        assert "Bug Watcher" in side and "(1 found)" in side and "Nightly Audit" in side
        if os.environ.get("VEGA_SCREENSHOTS"):
            app.save_screenshot("loops-panel-80x24.svg", path=os.environ["VEGA_SCREENSHOTS"])

    drive(app, script, size=(80, 24))
