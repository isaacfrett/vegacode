"""The chat view of the 2026-10-28 mockup: one centred reading column (at most 88 cells), Markdown
tables drawn as terminal tables, a boxed message box the column's width, and one dim status line
under it with live CPU, memory, swap and model readings (vega.live_stats)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from rich.text import Text
from textual.containers import HorizontalScroll
from textual.geometry import Region
from textual.widgets import Static
from tui_helpers import answer, drive, scripted_agent, send, wait_until

from vega import live_stats, tui, view
from vega.transcripts import SessionLog
from vega.tui import VegaApp
from vega.tui_table import TableBlock, build_table, is_number, numeric_columns

READING = live_stats.Reading(cpu_pct=23.0, mem_used_gb=11.4, mem_total_gb=16.0,
                             pressure="normal", swap_used_gb=0.4, model="qwen3:8b",
                             model_gb=6.1)
TABLE = """Timings:

| Stage | Before (ms) | After (ms) | Hit rate |
|-------|------------:|-----------:|---------:|
| parse | 412 | 398 | 71% |
| resolve imports | 1,280 | 340 | 88% |
"""
WIDE = ("Every column, side by side:\n\n| " +" | ".join(f"column_heading_{i}" for i in range(8)) + " |\n|"
        + "---|" * 8 + "\n| " + " | ".join(f"supercalifragilistic_{i}" for i in range(8)) + " |\n")


@pytest.fixture(autouse=True)
def new_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(view, "NEW_USER_PRESET", "chat")
    for name in ("VEGA_LAYOUT", "VEGA_THEME", "COLORFGBG", "NO_COLOR", "VEGA_RESOURCES"):
        monkeypatch.delenv(name, raising=False)


def chat_app(tmp_path: Path, replies: list | None = None, probe=lambda: READING,
             **options) -> VegaApp:
    project = tmp_path / "proj"
    project.mkdir(exist_ok=True)
    (project / "pyproject.toml").write_text("[project]\nname = 'p'\n")
    agent, _ = scripted_agent(project, replies or [])
    return VegaApp(project, agent=agent, session_log=SessionLog(tmp_path / "s", project),
                   warm_up=False, memory_probe=lambda: 5.1 * 1024, total_memory_gb=16.0,
                   resource_probe=probe, **options)


def readouts(app: VegaApp) -> str:
    return str(app.query_one("#readouts").render())


# -- the reading column -----------------------------------------------------------------------

@pytest.mark.parametrize("size", [(120, 40), (80, 30), (200, 50)])
def test_one_centred_column_with_the_message_box_under_it(tmp_path: Path, size) -> None:
    app = chat_app(tmp_path, [answer("A reply.\n\nA second paragraph.")])

    async def script(pilot) -> None:
        await send(pilot, "hello")
        await pilot.pause()
        width = size[0]
        box = app.query_one("#input-row").region
        status = app.query_one("#status").region
        assert box.width == status.width == min(width, tui.CHAT_COLUMN)
        assert box.x == status.x == (width - box.width) // 2  # centred
        assert box.height == 3  # one line inside a thin box
        for message in app.query(".message"):
            region = message.region
            assert region.width <= tui.CHAT_COLUMN
            if width >= tui.CHAT_COLUMN + 10:  # centred within a cell or two of the box
                assert abs(region.x - box.x) <= 3, (region, box)
        assert app.query_one("#conversation").max_scroll_x == 0
        assert str(app.query_one("#prompt-mark").render()) == ">"

    drive(app, script, size=size)


def test_panels_beside_the_chat_give_the_message_box_the_full_width(tmp_path: Path) -> None:
    app = chat_app(tmp_path)

    async def script(pilot) -> None:
        await send(pilot, "/panel changes")
        await pilot.pause()
        assert app.query_one("#left").display
        assert app.query_one("#input-row").region.width == 160
        assert app.chat_column_width() == 160

    drive(app, script, size=(160, 40))


# -- tables ------------------------------------------------------------------------------------

def test_numbers_are_detected_for_right_alignment() -> None:
    for number in ("412", "1,280", "-3.5%", "$1,200", "4.1 GB", "120 ms", "2x"):
        assert is_number(number), number
    for word in ("parse", "v1.2.3", "1st", "", "12 apples"):
        assert not is_number(word), word
    assert numeric_columns([["a", "1"], ["b", ""], ["c", "2%"]], 2) == [False, True]


def test_a_reply_table_is_a_bordered_table_with_numbers_on_the_right(tmp_path: Path) -> None:
    app = chat_app(tmp_path, [answer(TABLE)])

    async def script(pilot) -> None:
        await send(pilot, "timings?")
        await wait_until(pilot, lambda: bool(app.query(TableBlock)))
        await pilot.pause(0.2)
        static = app.query_one(TableBlock).query_one(".vega-table", Static)
        lines = [strip.text for strip in static.render_lines(
            Region(0, 0, static.size.width, static.size.height))]
        assert lines[0].startswith("╭") and lines[-1].startswith("╰")
        assert "Stage" in lines[1] and "Hit rate" in lines[1]
        assert lines[2].startswith("├")  # a rule under the header row
        parse = next(line for line in lines if "parse" in line)
        resolve = next(line for line in lines if "resolve" in line)
        # right-aligned numbers end in the same cell
        assert parse.index("412") + 3 == resolve.index("1,280") + 5
        assert parse.index("71%") == resolve.index("88%")
        assert static.size.width < app.query_one(TableBlock).size.width  # natural width
        assert app.query_one("#conversation").max_scroll_x == 0

    drive(app, script, size=(120, 40))


def test_a_table_too_wide_scrolls_inside_itself(tmp_path: Path) -> None:
    app = chat_app(tmp_path, [answer(WIDE)])

    async def script(pilot) -> None:
        await send(pilot, "what are the columns?")
        await wait_until(pilot, lambda: bool(app.query(TableBlock)))
        await pilot.pause(0.2)
        block = app.query_one(TableBlock)
        scroller = block.query_one(HorizontalScroll)
        assert scroller.max_scroll_x > 0  # the table scrolls ...
        assert app.query_one("#conversation").max_scroll_x == 0  # ... the page does not
        assert block.region.right <= app.query_one("#conversation").region.right
        assert app.query_one("#prompt").has_focus  # the table never takes the focus

    drive(app, script, size=(80, 30))


def test_tables_wrap_cells_before_they_scroll() -> None:
    from rich.console import Console

    table = build_table([Text("Name"), Text("About")],
                        [[Text("cache"), Text("a long description " * 6)]], [None, None], {})
    console = Console(width=40, record=True, file=None)
    with console.capture() as captured:
        console.print(table)
    lines = captured.get().splitlines()
    assert len(lines) > 5 and all(len(line) <= 40 for line in lines)


# -- the status line ---------------------------------------------------------------------------

def test_status_line_shows_live_resources(tmp_path: Path) -> None:
    app = chat_app(tmp_path)

    async def script(pilot) -> None:
        await wait_until(pilot, lambda: "cpu" in readouts(app))
        line = readouts(app)
        assert line.startswith("ready")
        for part in ("cpu 23%", "mem 11.4/16 GB normal", "swap 0.4 GB", "qwen3:8b 6.1 GB"):
            assert part in line, line
        assert len(line) + len(str(app.query_one("#hints").render())) <= tui.CHAT_COLUMN - 2

    drive(app, script, size=(120, 40))


def test_status_line_refreshes_without_moving_the_focus(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tui, "RESOURCES_EVERY", 0.05)
    readings = iter([READING] + [live_stats.Reading(cpu_pct=float(n), mem_used_gb=14.9,
                                                    mem_total_gb=16.0, pressure="warn",
                                                    swap_used_gb=2.0, model="")
                                 for n in range(60, 1000)])
    calls = []

    def probe() -> live_stats.Reading:
        calls.append(1)
        return next(readings)

    app = chat_app(tmp_path, probe=probe)

    async def script(pilot) -> None:
        await pilot.press("h", "i")
        await wait_until(pilot, lambda: "mem 14.9/16 GB warn" in readouts(app))
        assert len(calls) >= 2
        line = readouts(app)
        assert "not loaded" in line  # Ollama answered with nothing loaded
        assert app.query_one("#prompt").has_focus
        assert app.query_one("#prompt").text == "hi"  # typing was not interrupted

    drive(app, script, size=(120, 40))


def test_resources_can_be_turned_off(tmp_path: Path) -> None:
    app = chat_app(tmp_path)

    async def script(pilot) -> None:
        await wait_until(pilot, lambda: "cpu" in readouts(app))
        await send(pilot, "/resources off")
        await pilot.pause()
        assert "cpu" not in readouts(app) and not app.resources_shown()
        assert view.load().resources is False
        await send(pilot, "/resources on")
        await wait_until(pilot, lambda: "cpu" in readouts(app))

    drive(app, script, size=(120, 40))


def test_vega_resources_env_turns_it_off(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_RESOURCES", "0")
    calls = []
    app = chat_app(tmp_path, probe=lambda: calls.append(1) or READING)

    async def script(pilot) -> None:
        await pilot.pause(0.2)
        assert not calls and "cpu" not in readouts(app)

    drive(app, script, size=(120, 40))


# -- live_stats ----------------------------------------------------------------------------------

def test_segments_name_each_reading_and_flag_pressure() -> None:
    assert live_stats.segments(READING) == [
        ("cpu 23%", ""), ("mem 11.4/16 GB normal", ""), ("swap 0.4 GB", ""),
        ("qwen3:8b 6.1 GB", "")]
    tight = live_stats.Reading(mem_used_gb=15.2, mem_total_gb=16.0, pressure="critical",
                               swap_used_gb=3.0, model="")
    assert live_stats.segments(tight, "vega") == [
        ("mem 15.2/16 GB critical", "warn"), ("swap 3.0 GB", "warn"), ("vega not loaded", "")]
    assert live_stats.segments(live_stats.Reading()) == []  # Ollama silent: no model item


def test_loaded_model_reads_ollama_ps() -> None:
    data = {"models": [{"name": "qwen3:8b", "size": 6 * 1024**3, "size_vram": 6 * 1024**3}]}
    assert live_stats.loaded_model(data) == ("qwen3:8b", 6.0)
    assert live_stats.loaded_model({"models": []}) == ("", None)
    assert live_stats.loaded_model(None) == (None, None)


def test_sampler_asks_ollama_at_a_lower_rate() -> None:
    now = [0.0]
    asked = []
    sampler = live_stats.Sampler(ollama=lambda: asked.append(now[0]) or {"models": []},
                                 ollama_every=5.0, clock=lambda: now[0])
    for step in range(6):
        now[0] = step * 2.0
        sampler.read()
    assert asked == [0.0, 6.0]


@pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="kernel readings")
def test_sampler_reads_this_machine_without_a_subprocess(monkeypatch) -> None:
    import subprocess

    def refuse(*args, **kwargs):
        raise AssertionError("no subprocess per reading")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    sampler = live_stats.Sampler(ollama=lambda: None)
    first = sampler.read()
    second = sampler.read()
    assert first.mem_total_gb and 0 < first.mem_used_gb <= first.mem_total_gb * 1.05
    assert first.swap_used_gb is not None and first.swap_used_gb >= 0
    assert first.pressure in {"normal", "warn", "critical"} or sys.platform == "linux"
    assert second.cpu_pct is None or 0 <= second.cpu_pct <= 100
