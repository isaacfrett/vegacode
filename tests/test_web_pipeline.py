"""web_pipeline: the non-Python pipeline's CSS/HTML/JS config (routing, replies, checks, loop)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import features, page_render, routing, sandbox, web_pipeline
from vega.agent import CodingAgent

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
needs_webkit = pytest.mark.skipif(not page_render.available(),
                                  reason="needs macOS WebKit and the sandbox")

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <main>
    <div id="banner">Spring tides</div>
    <p class="card">Nelson</p>
    <button type="button" id="notify">Notify me</button>
  </main>
  <script src="app.js"></script>
</body>
</html>
"""
CSS = """main { max-width: 640px; margin: 0 auto; }
@keyframes slide-in { from { transform: translateY(-100%); } to { transform: none; } }
#banner { padding: 8px; animation: slide-in 400ms ease-out; }
.card { padding: 16px; transition: transform 200ms ease; }
"""
JS = "document.getElementById('notify').addEventListener('click', () => {});\n"
TASK = ("Make the page (`index.html`, `styles.css`) respect the `prefers-reduced-motion` setting: "
        "with it, nothing animates or transitions; without it, keep today's motion exactly. Edit "
        "`styles.css`; keep `index.html` unchanged.")


def switch(monkeypatch: pytest.MonkeyPatch, on: bool) -> None:
    """web_pipeline on or off for this test, whether it ships on or opt-in."""
    for env in (features.ENV, features.ENV_ON):
        names = [n for n in os.environ.get(env, "").split(",") if n and n != "web_pipeline"]
        monkeypatch.setenv(env, ",".join(names))
    if "web_pipeline" in features.FEATURES and not on:
        monkeypatch.setenv(features.ENV, ",".join(filter(None, [os.environ[features.ENV],
                                                                "web_pipeline"])))
    if "web_pipeline" in features.OPT_IN and on:
        monkeypatch.setenv(features.ENV_ON, ",".join(filter(None, [os.environ[features.ENV_ON],
                                                                   "web_pipeline"])))


def project(root: Path) -> Path:
    (root / "index.html").write_text(PAGE)
    (root / "styles.css").write_text(CSS)
    (root / "app.js").write_text(JS)
    return root


def scripted(root: Path, replies: list[str]) -> tuple[CodingAgent, list[dict]]:
    agent = CodingAgent(root, allow_commands=False)
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content=replies.pop(0), tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, calls


# -- routing --------------------------------------------------------------------------------------

def test_css_html_js_changes_route_here_behind_the_switch(tmp_path: Path, monkeypatch) -> None:
    project(tmp_path)
    switch(monkeypatch, True)
    route = routing.route_task(TASK, tmp_path)
    assert (route.name, route.pipeline) == (routing.NONPYTHON, "web")
    assert route.other_targets == ["styles.css"] and "index.html" in route.inputs
    switch(monkeypatch, False)
    assert routing.route_task(TASK, tmp_path).name == routing.AGENT  # the old route


def test_keep_unchanged_files_are_inputs_and_a_named_package_json_is_a_target(
        tmp_path: Path, monkeypatch) -> None:
    switch(monkeypatch, True)
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "text.js").write_text("module.exports = { a: 1 };\n")
    (tmp_path / "index.html").write_text(PAGE)
    (tmp_path / "package.json").write_text('{"name": "x"}\n')
    task = ("Convert `lib/text.js` to an ES module; `package.json` gets `\"type\": \"module\"`. "
            "Update `index.html` too (keep the markup unchanged).")
    route = routing.route_task(task, tmp_path)
    assert route.pipeline == "web"
    assert route.other_targets == ["lib/text.js", "package.json"]
    assert "index.html" in route.inputs


def test_python_targets_and_large_files_are_not_this_route(tmp_path: Path, monkeypatch) -> None:
    switch(monkeypatch, True)
    project(tmp_path)
    assert web_pipeline.blocker(tmp_path, ["styles.css", "tool.py"]).startswith("also changes")
    (tmp_path / "big.css").write_text("a { color: red; }\n" * 2000)
    assert "characters" in web_pipeline.blocker(tmp_path, ["big.css"])
    assert web_pipeline.blocker(tmp_path, ["package.json"]) == "no CSS/HTML/JS file to change"


# -- replies and static checks --------------------------------------------------------------------

def test_reply_blocks_are_matched_to_targets_by_label_comment_or_language() -> None:
    reply = ("File: styles.css\n```css\na { color: red; }\n```\n"
             "```javascript\n// app.js\nconsole.log(1);\n```\n"
             "**index.html**\n```html\n<p>x</p>\n```\n")
    found, strays = web_pipeline.reply_files(reply, ["styles.css", "app.js"])
    assert found == {"styles.css": "a { color: red; }\n", "app.js": "console.log(1);\n"}
    assert strays == ["index.html"]
    only, _ = web_pipeline.reply_files("```\nb { margin: 0; }\n```", ["styles.css"])
    assert only == {"styles.css": "b { margin: 0; }\n"}


def test_html_nesting_problems() -> None:
    assert web_pipeline.html_problems(PAGE) == []
    assert web_pipeline.html_problems("<ul><li>a<li>b</ul><p>x") == []  # optional end tags
    problems = web_pipeline.html_problems("<main><div id='a'><span>x</div></main><b id='a'>y</b>")
    assert any("`<span>` is not closed" in p for p in problems)
    assert any("id `a` is used twice" in p for p in problems)
    assert web_pipeline.html_problems("<div>open")[0].endswith("`<div>` is never closed")


@needs_node
def test_node_check_reports_a_syntax_error(tmp_path: Path) -> None:
    (tmp_path / "ok.js").write_text("export const a = 1;\n")
    (tmp_path / "bad.js").write_text("function f( {\n")
    (tmp_path / "package.json").write_text('{"type": "module"}\n')
    assert web_pipeline.js_problems(tmp_path, "ok.js") == []
    problems = web_pipeline.js_problems(tmp_path, "bad.js")
    assert problems and "bad.js: `node --check` fails" in problems[0] and "SyntaxError" in problems[0]


def test_probes_come_from_the_task_words() -> None:
    probes = web_pipeline.probes_for("Respect `prefers-reduced-motion`.")
    assert probes.motion and not probes.shift and probes.contrast is None
    probes = web_pipeline.probes_for("Follow `prefers-color-scheme` with a WCAG contrast ratio of "
                                     "at least 7:1.")
    assert probes.scheme and probes.contrast == 7.0
    assert web_pipeline.probes_for("Fix the low contrast text.").contrast == 4.5
    assert web_pipeline.probes_for("Remove the layout shift by reserving their space").shift
    assert not web_pipeline.probes_for("Support `prefers-contrast: more`.").contrast


def test_pages_and_timer_wait(tmp_path: Path) -> None:
    project(tmp_path)
    (tmp_path / "late.js").write_text("setTimeout(() => load(), 500);\n")
    assert web_pipeline.pages_for(tmp_path, ["styles.css"], []) == ["index.html"]
    assert web_pipeline.pages_for(tmp_path, ["styles.css"], ["index.html"]) == ["index.html"]
    assert web_pipeline.wait_ms(tmp_path, {"styles.css": CSS}) == 900
    assert web_pipeline.changed_declarations(CSS, CSS + "p { colr: red; --x: 1; }\n") == [
        ("colr", "red")]


# -- the loop ----------------------------------------------------------------------------------------

@needs_webkit
def test_rendered_checks_send_back_lost_motion_and_a_partial_css_reply_is_merged(
        tmp_path: Path, monkeypatch) -> None:
    switch(monkeypatch, True)
    project(tmp_path)
    # round 1: motion off for everyone (the 2026-10-20 agent-loop mistake); round 2 puts the
    # .card rule back and adds the media block: only those rules are sent, every other is kept
    wrong = "File: styles.css\n```css\n.card { padding: 16px; transition: none; }\n```"
    right = ("File: styles.css\n```css\n.card { padding: 16px; transition: transform 200ms ease; }"
             "\n@media (prefers-reduced-motion: reduce) {\n"
             "  #banner { animation: none; }\n  .card { transition: none; }\n}\n```")
    agent, calls = scripted(tmp_path, [wrong, right])
    answer = agent.run(TASK)
    css = (tmp_path / "styles.css").read_text()
    assert "@keyframes slide-in" in css and "transition: transform 200ms ease" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    repair = calls[1]["messages"][-1]["content"]
    assert "without the reduced-motion preference" in repair and ".card" in repair
    assert "still moving: #banner" in repair  # round 1 also left the banner animating
    assert answer.startswith("Wrote styles.css;") and "rendered check" in answer
    assert agent.last_route["pipeline"] == "web"


@needs_node
@needs_sandbox
def test_a_js_syntax_error_is_repaired(tmp_path: Path, monkeypatch) -> None:
    switch(monkeypatch, True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "sum.js").write_text("export function sum(a, b) { return a - b; }\n")
    task = "`sum` in `src/sum.js` subtracts. Fix `src/sum.js` so `sum(a, b)` returns a + b."
    broken = "File: src/sum.js\n```javascript\nexport function sum(a, b) { return a + b;\n```"
    fixed = "File: src/sum.js\n```javascript\nexport function sum(a, b) { return a + b; }\n```"
    agent, calls = scripted(tmp_path, [broken, fixed])
    answer = agent.run(task)
    assert (tmp_path / "src" / "sum.js").read_text() == \
        "export function sum(a, b) { return a + b; }\n"
    # node --check passes an unterminated ES module in a package without "type" (Node 24's
    # syntax detection); loading the module catches it
    assert "src/sum.js: loading it in node fails" in calls[1]["messages"][-1]["content"]
    assert answer.startswith("Wrote src/sum.js; 2 of 2 checks pass")
