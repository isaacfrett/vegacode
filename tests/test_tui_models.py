"""The Models pane in the workbench, against the fake Ollama server (tests/fake_ollama.py): rows,
fit badges, the running model, scores, and the keys (Enter switches, d downloads, x removes, r
refreshes), each asking first. No real model is pulled, deleted or run.

VEGA_SCREENSHOTS=<dir> saves SVGs at 120x36 dark and 80x24 light."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_model_store import fake, no_scores, scores  # noqa: F401 - fake Ollama, scores files
from tui_helpers import drive, make_app, scripted_agent, wait_until

from vega import models, server, view
from vega.tui import ChoiceScreen


def options_text(pane) -> str:
    return "\n".join(str(pane.get_option_at_index(i).prompt) for i in range(pane.option_count))


def index_of(pane, option_id: str) -> int:
    return pane.get_option_index(option_id)


def shoot(app, name: str) -> None:
    folder = os.environ.get("VEGA_SCREENSHOTS")
    if folder:
        app.save_screenshot(f"{name}.svg", path=folder)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    folder = tmp_path / "proj"
    folder.mkdir()
    (folder / "pyproject.toml").write_text("[project]\nname = 'p'\n")
    return folder


def test_models_pane_is_hidden_by_default_and_placeable() -> None:
    assert "models" in view.PANES and view.PANE_TITLES["models"] == "Models"
    assert view.View().column("models") is None
    placed = view.place(view.View(), "models", "right")
    assert placed.column("models") == "right"
    assert view.validate({"left": ["models"]}).left == ("models",)


@pytest.mark.parametrize(("theme", "size"), [("dark", (120, 36)), ("light", (80, 24))])
def test_models_pane_rows(fake, project: Path, tmp_path: Path,  # noqa: F811
                          monkeypatch: pytest.MonkeyPatch, theme: str, size) -> None:
    monkeypatch.setenv("VEGA_THEME", theme)
    # 120 columns: Models | conversation | Preview; 80 columns: Models | conversation
    right = ("preview",) if size[0] >= 120 else ()
    view.save(view.View(left=("models",), right=right, left_width=30))
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        pane = app.query_one("#wb-models")
        await wait_until(pilot, lambda: "granite-8b" in options_text(pane))
        # drawn at the list's width less its scrollbar (it overflows at 80x24)
        await wait_until(pilot, lambda: all(len(line) <= pane.scrollable_content_region.width
                                            for line in options_text(pane).splitlines()))
        text = options_text(pane)
        lines = text.splitlines()
        assert lines[0] == "16 GB RAM  40.0 GB free"
        assert lines[1].startswith("● granite-8b") and lines[1].endswith("5.3 GB")
        assert lines[2].startswith("  fits  fix 7/10")
        if size[0] >= 120:
            assert lines[2].startswith("  fits  fix 7/10  goal 3/8")
        assert "to download" in text and "phi4-mini" in text and "smollm2:1.7b" in text
        assert text.index("smollm2") < text.index("to download") < text.index("phi4-mini")
        pane.focus()
        pane.highlighted = index_of(pane, "Mgranite-8b")
        await pilot.pause()
        shoot(app, f"models-{theme}-{size[0]}x{size[1]}")
        if not right:
            return
        title = str(app.query_one("#preview").border_title)
        assert "granite-8b" in title and "granite4.1:8b" in title
        body = str(app.query_one("#preview-body").render())
        assert "needs ~8.7 GB of 16 GB: fits" in body and "Licence" in body

    drive(app, script, size=size)


def test_models_pane_keys_ask_first(fake, project: Path, tmp_path: Path,  # noqa: F811
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    view.save(view.View(left=("models",), right=("preview",), left_width=30))
    switched: list[str] = []

    def use(name: str) -> int:
        switched.append(name)
        (server.STATE_DIR / "active_model.json").write_text(json.dumps({"profile": name}))
        return 0

    monkeypatch.setattr(models, "use", use)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def answer(pilot, key: str) -> None:
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        await pilot.press(key)
        await wait_until(pilot, lambda: not isinstance(app.screen, ChoiceScreen))

    async def script(pilot) -> None:
        pane = app.query_one("#wb-models")
        await wait_until(pilot, lambda: "coder-7b" in options_text(pane))
        pane.focus()
        # Enter switches, after asking; n keeps the current model
        pane.highlighted = index_of(pane, "Mcoder-7b")
        await pilot.press("enter")
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        assert "Switch to coder-7b?" in str(app.screen.query_one("#dialog-title").render())
        shoot(app, "models-switch-120x36")
        await pilot.press("n")
        await wait_until(pilot, lambda: not isinstance(app.screen, ChoiceScreen))
        assert switched == []
        await pilot.press("enter")
        await answer(pilot, "y")
        await wait_until(pilot, lambda: switched == ["coder-7b"] and app.models_busy is None
                         and options_text(pane).splitlines()[1].startswith("● coder-7b"))
        # x refuses the active model without asking, and deletes nothing
        pane.highlighted = index_of(pane, "Mcoder-7b")
        await pilot.press("x")
        await pilot.pause()
        assert not isinstance(app.screen, ChoiceScreen)
        assert fake.calls("DELETE", "/api/delete") == []
        # x on another model asks, shows the space, then deletes through Ollama
        pane.highlighted = index_of(pane, "Mcoder-7b-16k")
        await pilot.press("x")
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        body = str(app.screen.query_one("#dialog-body").render())
        assert "vega-coder-7b-16k:latest" in body and "frees" in body
        await pilot.press("y")
        await wait_until(pilot, lambda: fake.calls("DELETE", "/api/delete") != [])
        assert [b["model"] for b in fake.calls("DELETE", "/api/delete")] == [
            "vega-coder-7b-16k:latest"]
        # d downloads a catalogue model after showing its size and the free disk
        await wait_until(pilot, lambda: "Mphi4-mini" in {
            pane.get_option_at_index(i).id for i in range(pane.option_count)}
            and app.models_busy is None)
        pane.highlighted = index_of(pane, "Mphi4-mini")
        await pilot.press("d")
        await wait_until(pilot, lambda: isinstance(app.screen, ChoiceScreen))
        body = str(app.screen.query_one("#dialog-body").render())
        assert "2.5 GB" in body and "40.0 GB now" in body and "37.5 GB after" in body
        await pilot.press("y")
        await wait_until(pilot, lambda: fake.built == ["vega-phi4-mini"]
                         and app.models_busy is None)
        assert fake.calls("POST", "/api/pull") == [{"model": "phi4-mini:3.8b", "stream": True}]
        # r reads the list again
        before = len(fake.calls("GET", "/api/tags"))
        await pilot.press("r")
        await wait_until(pilot, lambda: len(fake.calls("GET", "/api/tags")) > before)

    drive(app, script, size=(120, 36))


def test_download_refused_in_the_pane_when_disk_is_low(fake, project: Path,  # noqa: F811
                                                        tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    from vega import model_store

    monkeypatch.setattr(model_store, "free_disk_gb", lambda path=None: 13.0)
    view.save(view.View(left=("models",), right=("preview",), left_width=30))
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        pane = app.query_one("#wb-models")
        await wait_until(pilot, lambda: "llama31-8b" in options_text(pane))
        pane.focus()
        pane.highlighted = index_of(pane, "Mllama31-8b")
        await pilot.press("d")
        await wait_until(pilot, lambda: "Not downloading llama31-8b" in
                         str(app.query_one("#conversation").render()) or any(
                             "Not downloading" in str(n.render())
                             for n in app.query("#conversation *")))
        assert not isinstance(app.screen, ChoiceScreen)
        assert fake.calls("POST", "/api/pull") == []

    drive(app, script, size=(120, 36))


def test_models_pane_without_scores_shows_no_score_text(fake, no_scores) -> None:  # noqa: F811
    """No scores file (the public release): rows and the preview carry no suite labels or counts."""
    from vega import model_store, tui_models

    colours = {key: "white" for key in ("fg", "muted", "accent", "success", "warning", "error")}
    state = model_store.inventory()
    for row in state.rows:
        assert not row.scores, row.name
        detail = tui_models.detail_line(row, 120, colours).plain
        _, preview = tui_models.preview(state, row.name, 60, colours)
        for label in model_store.SUITE_LABELS.values():
            assert f"  {label} " not in detail, (row.name, label)
            assert f"\n{label} " not in preview.plain, (row.name, label)
