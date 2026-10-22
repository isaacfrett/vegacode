"""The product view in the app (product workbench, phase 1): the product preset for new users,
F3 between the product and code views, the 80-column strip, the product screen, the evidence
sheet, prepared fix requests (never sent) and the opt-in usage log. Renders the real app at
80x24 and 140x40 in both themes (SVG; VEGA_WRITE_SCREENS=1 writes docs/screens/product/)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from tui_helpers import answer, calls, drive, make_app, scripted_agent, send, wait_until

from vega import acceptance, product, sandbox, server, usage, view
from vega.tui import PromptInput
from vega.tui_product import EvidenceScreen, ProductScreen, ReleaseList, strip_text

SCREENS = Path(__file__).resolve().parents[1] / "docs" / "screens" / "product"

VEGA_MD = """# VEGA.md

<!-- vega:auto -->
## fernhill
Order-ahead site for Fernhill Bakery

- Languages: Python (4 files), JavaScript (2 files), HTML (3 files)
- Tests: `python -m pytest -q`
<!-- /vega:auto -->
"""


@pytest.fixture(autouse=True)
def product_for_new_users(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(view, "NEW_USER_PRESET", "product")


def git(project: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)


def fernhill(root: Path) -> Path:
    """A small website at release 0.3.0 (tagged), with the version bumped to 0.4.0 since."""
    project = root / "projects" / "fernhill"
    (project / "static").mkdir(parents=True)
    (project / "tests").mkdir()
    (project / "index.html").write_text("<h1>Fernhill</h1>\n")
    (project / "menu.html").write_text("<table class='prices'></table>\n")
    (project / "static" / "cart.js").write_text("export const total = (xs) => xs.reduce("
                                                "(a, b) => a + b, 0);\n")
    (project / "app.py").write_text("OPEN, CLOSE = 7, 15\n")
    (project / "tests" / "test_app.py").write_text("def test_hours():\n    assert True\n")
    (project / "pyproject.toml").write_text("[project]\nname = 'fernhill'\nversion = '0.3.0'\n")
    (project / "CHANGELOG.md").write_text("# Changelog\n\n## 0.3.0\n- Menu page\n")
    (project / "VEGA.md").write_text(VEGA_MD)
    git(project, "init", "-q")
    git(project, "config", "user.email", "t@example.com")
    git(project, "config", "user.name", "t")
    git(project, "add", "-A")
    git(project, "commit", "-qm", "0.3.0")
    git(project, "tag", "v0.3.0")
    (project / "pyproject.toml").write_text("[project]\nname = 'fernhill'\nversion = '0.4.0'\n")
    return project


def session(agent_box: list) -> list:
    return [
        calls(("write_file", {"path": "checkout.py", "content": "SLOTS = 15\n"}),
              prompt_tokens=3100),
        calls(("run_check", {"command": "python -m pytest -q"}), prompt_tokens=3600),
        answer("Checkout now asks for a pickup time in 15-minute slots from 7:00 to 14:45, and "
               "the order API refuses times outside opening hours. Tests pass (48 of 48).",
               prompt_tokens=4100),
        calls(("write_file", {"path": "static/cart.js",
                              "content": "export const total = (xs) => xs.reduce((a, b) => a + "
                                         "b, 0) / 100;\n"}), prompt_tokens=4600),
        calls(("run_check", {"command": "python -m pytest -q"}), prompt_tokens=5000),
        answer("Totals are now kept in cents and shown with two decimals. Tests pass.",
               prompt_tokens=5400),
        calls(("write_file", {"path": "static/menu.css",
                              "content": "@media (max-width: 480px) { .grid { columns: 1 } }\n"}),
              prompt_tokens=5900),
        calls(("run_check", {"command": "python -m pytest -q menu"}), prompt_tokens=6300),
        answer("The menu grid now drops to one column below 480 px. One check still fails: at "
               "375 px the page is 391 px wide because the price table does not wrap. Shall I "
               "stack the table on small screens?", prompt_tokens=6800),
    ]


def fake_check(agent_box: list):
    def check(command: str, project: Path, **_: object) -> str:
        if "menu" in command:  # the task checks of turn 3: one of five fails
            agent_box[0].last_acceptance = acceptance.Result([
                acceptance.Item("static", "static/menu.css", True),
                acceptance.Item("command", "menu.html loads", True),
                acceptance.Item("command", "menu.html at 1280 px", True),
                acceptance.Item("command", "menu grid one column at 375 px", True),
                acceptance.Item("command", "menu.html at 375 px", False,
                                "391 px wide; the price table does not wrap"),
            ])
        return "exit code: 0\nstdout:\n........\n48 passed in 12.4s\n"
    return check


def scripted_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path.resolve()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("VEGA_MODEL", "granite-4.1-8b")
    project = fernhill(home)
    box: list = []
    monkeypatch.setattr(sandbox, "run_check", fake_check(box))
    agent, _ = scripted_agent(project, session(box))
    box.append(agent)
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)
    return app, project


async def run_session(pilot) -> None:
    app = pilot.app
    await send(pilot, "Add a pickup time to checkout. We open 7:00 to 15:00, in 15-minute slots.")
    await send(pilot, "The cart shows 12.000000001 for three rolls.")
    await send(pilot, "The menu page scrolls sideways on my phone.")
    await wait_until(pilot, lambda: app.product_report is not None
                     and len(app.product_turns) == 3
                     and any(c.key == "task_checks" and c.status == product.FAIL
                             for c in app.product_report.checks))
    await wait_until(pilot, lambda: app.memory_mb is not None)
    await pilot.pause()


def release_rows(app, selector: str = "#wb-release") -> list[str]:
    pane = app.query_one(selector, ReleaseList)
    return [str(pane.get_option_at_index(i).prompt) for i in range(pane.option_count)]


# -- the preset and F3 -------------------------------------------------------------------------

def test_new_user_starts_in_the_product_view_and_a_saved_view_is_kept(tmp_path: Path) -> None:
    assert view.load() == view.preset("product")  # no view.json: a new user
    project = tmp_path / "p"
    project.mkdir()
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        left = [child.id for child in app.query_one("#left").children]
        assert left == ["wb-product", "wb-release"] and app.view.mode == "product"

    drive(app, script, size=(140, 40))
    assert not (server.STATE_DIR / "view.json").exists()  # nothing written for a new user

    saved = view.preset("review")
    view.save(saved)
    before = (server.STATE_DIR / "view.json").read_text()
    assert view.load() == saved
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def kept(pilot) -> None:
        assert app.view == saved and app.view.mode == "code"

    drive(app, kept, size=(140, 40))
    assert (server.STATE_DIR / "view.json").read_text() == before  # untouched


def test_f3_swaps_product_and_code_views_and_keeps_the_code_layout(tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    view.save(view.preset("monitor"))  # an existing user with their own code view
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        assert app.view.mode == "code" and "F3 product view" in str(
            app.query_one("#hints").render())
        await pilot.press("f3")
        await pilot.pause(0.2)
        assert app.view.mode == "product"
        assert [c.id for c in app.query_one("#left").children] == ["wb-product", "wb-release"]
        assert "F3 code view" in str(app.query_one("#hints").render())
        assert str(app.query_one("#mode").render()) == "product view"
        await pilot.press("f3")
        await pilot.pause(0.2)
        assert app.view == view.preset("monitor")  # the code view it came from
        await send(pilot, "/product")
        await pilot.pause(0.2)
        assert app.view.mode == "product"
        # every code pane is still available in the product view
        await send(pilot, "/pane git right")
        await pilot.pause(0.2)
        assert [c.id for c in app.query_one("#right").children] == ["wb-git"]

    drive(app, script, size=(140, 40))


def test_f3_on_a_narrow_terminal_opens_the_product_screen(tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        assert app.narrow and app.view.mode == "product"
        assert app.query_one("#strip").display
        assert "F3 product" in str(app.query_one("#hints").render())
        await pilot.press("f3")
        await pilot.pause(0.2)
        assert isinstance(app.screen, ProductScreen)
        assert app.screen.query_one("#ps-product").display
        await pilot.press("5")
        await pilot.pause(0.2)
        assert not app.screen.query_one("#ps-product").display  # Release alone
        await pilot.press("1")
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ProductScreen)
        await pilot.press("f3")
        await pilot.pause()
        await pilot.press("f3")  # F3 also goes back
        await pilot.pause()
        assert not isinstance(app.screen, ProductScreen)
        assert app.view.mode == "product"  # the preset did not change

    drive(app, script, size=(80, 24))


def test_code_view_has_no_strip(tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    view.save(view.View())
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        assert app.narrow and not app.query_one("#strip").display
        assert not app.query_one("#mode").display

    drive(app, script, size=(80, 24))


# -- the session, checklist, f and the evidence sheet --------------------------------------------

def test_session_checklist_and_fix_request_is_never_sent(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # finish_gate_no_edit off: a scripted turn makes no edit and would take an extra reply
    monkeypatch.setenv("VEGA_FEATURES_OFF", "finish_gate_no_edit")
    app, _ = scripted_session(tmp_path, monkeypatch)

    async def script(pilot) -> None:
        await run_session(pilot)
        assert [o for _, o in app.product_turns] == ["pass", "pass", "fail"]
        product_text = str(app.query_one("#wb-product .product-body").render())
        assert "Not ready: 2 checks block shipping" in product_text
        assert "3 of 9 pass, 1 warn, 3 not checked" in product_text
        assert "✗ The menu page scrolls sideways" in product_text
        rows = release_rows(app)
        assert rows[0].startswith("✓ Tests pass") and rows[0].endswith("48 of 48")
        assert rows[1].startswith("✗ Task checks") and rows[1].endswith("1 of 5 fail")
        assert any(r.startswith("✗ Changelog") and r.endswith("no 0.4.0 entry") for r in rows)
        assert any(r.startswith("○ Preview renders") for r in rows)
        # f on Task checks: the request lands in the message box and nothing is sent
        release = app.query_one("#wb-release", ReleaseList)
        release.focus()
        release.highlighted = 1
        turns = app.turns
        await pilot.press("f")
        await pilot.pause(0.3)
        prompt = app.query_one("#prompt", PromptInput)
        assert prompt.text.startswith("Fix what the task checks report: menu.html at 375 px")
        assert app.focused is prompt and app.turns == turns and not app.working
        assert app.pending_fix and app.pending_fix[0] == "task_checks"
        # Enter on a row opens its evidence; f there prepares the request too
        release.focus()
        release.highlighted = 5  # Changelog
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert isinstance(app.screen, EvidenceScreen)
        assert "CHANGELOG.md has not changed since v0.3.0" in str(
            app.screen.query_one("#evidence-body").render())
        await pilot.press("f")
        await pilot.pause(0.2)
        assert prompt.text.startswith("Write the 0.4.0 entry in CHANGELOG.md")
        assert app.turns == turns
        # s: shipping is not built; nothing runs
        release.focus()
        await pilot.press("s")
        await pilot.pause()
        assert app.turns == turns

    drive(app, script, size=(140, 40))


def test_strip_drops_items_from_the_right_and_keeps_the_verdict() -> None:
    colours = {k: "" for k in ("fg", "muted", "success", "error", "warning", "del_bg",
                               "add_bg", "panel")}
    facts = product.Facts(name="fernhill", kind="website", known=True)
    checks = [product.Check("tests", "Tests pass", product.FAIL, True, "1 fail"),
              product.Check("changelog", "Changelog", product.FAIL, True, "x")]
    report = product.Report(facts, checks, product.Git(repo=True), ([], []))
    tests = ("48/48", "success")
    wide = strip_text(report, tests, 80, colours).plain
    assert wide.startswith("fernhill   2 block shipping   tests 48/48  secrets ok")
    assert wide.endswith("F3") and len(wide) == 80
    mid = strip_text(report, tests, 50, colours).plain
    assert "tests 48/48" in mid and "secrets" not in mid and len(mid) <= 50
    narrow = strip_text(report, tests, 36, colours).plain
    assert "2 block shipping" in narrow and "tests" not in narrow and len(narrow) <= 36
    tiny = strip_text(report, tests, 22, colours).plain
    assert "2 block shipping" in tiny and len(tiny) <= 22  # the name is clipped, never the verdict


# -- the usage log -------------------------------------------------------------------------------

def test_usage_log_is_off_by_default_and_writes_only_when_on(tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        assert not usage.enabled()
        await pilot.press("f3")
        await pilot.pause(0.2)
        await pilot.press("escape")
        assert not usage.path().exists()
        await send(pilot, "/usage on")
        assert usage.enabled()
        await pilot.press("f3")
        await pilot.pause(0.2)
        await pilot.press("escape")
        await pilot.pause()
        await send(pilot, "/usage off")
        await pilot.press("f3")
        await pilot.pause(0.2)

    drive(app, script, size=(80, 24))
    rows = [json.loads(line) for line in usage.path().read_text().splitlines()]
    events = [row["event"] for row in rows]
    assert events == ["session_start", "product_screen_open", "product_tab",
                      "product_screen_close"]  # only while on: nothing before or after
    assert all(set(row) <= {"t", "event", "session", "view", "preset", "tab", "check", "to",
                            "within_10s", "wrote", "checks", "edited"} for row in rows)
    assert str(project) not in usage.path().read_text()  # never paths or text


def test_usage_record_is_a_no_op_when_off() -> None:
    assert usage.record("x") is False and not usage.path().exists()
    usage.set_enabled(True)
    assert usage.record("x", check="a" * 200, n=3) is True
    row = json.loads(usage.path().read_text())
    assert row["event"] == "x" and len(row["check"]) == usage.FIELD_CHARS and row["n"] == 3
    assert "1 events from 0 sessions" in usage.summary()


# -- screenshots ---------------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(140, 40), (80, 24)], ids=["140x40", "80x24"])
def test_screenshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                     size: tuple[int, int]) -> None:
    app, _ = scripted_session(tmp_path, monkeypatch)
    out = SCREENS if os.environ.get("VEGA_WRITE_SCREENS") == "1" else tmp_path / "screens"
    out.mkdir(parents=True, exist_ok=True)
    width, height = size

    async def script(pilot) -> None:
        await run_session(pilot)
        app.query_one("#prompt").focus()
        await pilot.pause()
        for theme in ("dark", "light"):
            if (theme == "light") != app.theme.endswith("light"):
                await pilot.press("ctrl+t")
            await pilot.pause(0.3)
            app.save_screenshot(f"product-{width}x{height}-{theme}.svg", path=str(out))
        if width < 100:
            await pilot.press("f3")
            await pilot.pause(0.5)
            await wait_until(pilot, lambda: "Not ready" in str(
                app.screen.query_one("#ps-product .product-body").render()))
            for theme in ("light", "dark"):
                if (theme == "light") != app.theme.endswith("light"):
                    await pilot.press("ctrl+t")
                await pilot.pause(0.3)
                app.save_screenshot(f"product-screen-{width}x{height}-{theme}.svg",
                                    path=str(out))

    drive(app, script, size=size)
    assert (out / f"product-{width}x{height}-dark.svg").exists()


def test_vega_usage_command_switches_and_summarises() -> None:
    from typer.testing import CliRunner

    from vega.cli import app as cli

    runner = CliRunner()
    result = runner.invoke(cli, ["usage"])
    assert result.exit_code == 0 and "Usage log is off" in result.output
    assert not usage.path().exists()
    result = runner.invoke(cli, ["usage", "on"])
    assert result.exit_code == 0 and "Usage log is on" in result.output and usage.enabled()
    usage.record("fix_prefilled", check="tests")
    result = runner.invoke(cli, ["usage", "off", "--days", "1"])
    assert "Last 1 days: 1 events" in result.output and "Fix requests: 0 sent of 1" in \
        result.output and not usage.enabled()
