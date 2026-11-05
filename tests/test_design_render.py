"""vega.design_render: building the Swift renderer, running it (a fake here), grading measured
pages, and the eval hook. One real WebKit render, skipped where swiftc is missing (CI, Linux)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import design_render as d
from vega import evals


def measure(**over):
    """A clean measured page (a 390px phone, light) that passes every check."""
    base = {
        "viewport": {"w": 390, "h": 844}, "dark": False, "scroll_width": 390, "overflowing": [],
        "page_bg": [255, 255, 255, 1], "overlaps": [], "clipped": [],
        "texts": [{"el": "h1", "text": "Hi", "size": 32, "weight": 700, "fg": [28, 31, 38, 1],
                   "bg": [[255, 255, 255, 1]], "bg_image": False, "disabled": False},
                  {"el": "p", "text": "Body", "size": 16, "weight": 400, "fg": [74, 81, 96, 1],
                   "bg": [[255, 255, 255, 1]], "bg_image": False, "disabled": False}],
        "focusables": [{"el": "button", "tag": "button", "text": "Go", "rect": {"x": 16, "y": 100, "w": 80, "h": 44},
                        "target": {"w": 80, "h": 44}, "inline_link": False, "tabindex": None, "disabled": False,
                        "name": "Go", "focus": {"changed": ["outline-style", "outline-width", "outline-color"],
                                                "outline": True, "outline_style": "solid",
                                                "outline_colour": [31, 79, 209, 1], "shadow": False,
                                                "bg": [255, 255, 255, 1]}}],
        "images": [], "headings": [{"el": "h1", "level": 1, "text": "Hi"}, {"el": "h2", "level": 2, "text": "Sub"}],
        "form_controls": [], "spacing": [], "probes": {}, "root_styles": {},
    }
    base.update(over)
    return base


SPEC = {"page": "index.html", "viewports": ["phone"], "schemes": ["light"],
        "checks": {"overflow": True, "overlap": True, "contrast": True, "tap_targets": {"min": 24},
                   "focus_visible": True, "headings": True, "images_alt": True, "labels": True}}


def graded(m, spec=SPEC, focus=None):
    return d.grade(spec, [{"name": "phone-light", "measure": m, "focus": focus or [], "error": None}])


def test_clean_page_passes():
    assert graded(measure()) == []


def test_contrast_maths():
    assert d.contrast([0, 0, 0], [255, 255, 255]) == pytest.approx(21.0)
    assert d.contrast([255, 255, 255], [255, 255, 255]) == pytest.approx(1.0)
    # #767676 on white is the classic 4.54:1 borderline
    assert d.contrast([118, 118, 118], [255, 255, 255]) == pytest.approx(4.54, abs=0.01)


@pytest.mark.parametrize("change, check", [
    ({"scroll_width": 640, "overflowing": [{"el": "table", "right": 640}]}, "overflow"),
    ({"overlaps": [{"a": "h2", "b": "span.price", "w": 40, "h": 20}]}, "overlap"),
    ({"clipped": [{"el": "p", "by": "section", "text": "Every change"}]}, "overlap"),
    ({"headings": [{"el": "h1", "level": 1, "text": "a"}, {"el": "h3", "level": 3, "text": "b"}]}, "headings"),
    ({"headings": [{"el": "h2", "level": 2, "text": "a"}]}, "headings"),
    ({"images": [{"el": "img", "tag": "img", "has_alt": False, "alt": "", "hidden": False}]}, "images_alt"),
    ({"form_controls": [{"el": "input#q", "name": "", "type": "text"}]}, "labels"),
])
def test_each_defect_fails_its_check(change, check):
    failures = graded(measure(**change))
    assert failures and all(f.startswith(f"[{check}] phone-light") for f in failures)


def test_contrast_large_text_threshold_and_gradient_worst_stop():
    grey = {"el": "p", "text": "x", "size": 16, "weight": 400, "fg": [150, 150, 150, 1],
            "bg": [[255, 255, 255, 1]], "bg_image": False, "disabled": False}
    assert graded(measure(texts=[grey]))[0].startswith("[contrast]")  # ~2.96:1
    big = {**grey, "fg": [130, 130, 130, 1], "size": 24}  # ~3.9:1, large text needs 3
    assert graded(measure(texts=[big])) == []
    stops = {**grey, "fg": [255, 255, 255, 1], "bg": [[29, 60, 143, 1], [224, 195, 252, 1]]}
    assert graded(measure(texts=[stops]))[0].startswith("[contrast]")
    assert graded(measure(texts=[{**grey, "bg_image": True}])) == []  # unknown: not graded


def test_tap_targets_exempt_inline_links_and_honour_min():
    f = measure()["focusables"][0]
    small = {**f, "target": {"w": 30, "h": 18}}
    assert graded(measure(focusables=[small]))[0].startswith("[tap_targets]")
    assert graded(measure(focusables=[{**small, "inline_link": True}])) == []
    spec = {**SPEC, "checks": {**SPEC["checks"], "tap_targets": {"min": 44}}}
    assert graded(measure(focusables=[{**f, "target": {"w": 40, "h": 40}}]), spec)[0].startswith("[tap_targets]")


def test_focus_visible_rules():
    f = measure()["focusables"][0]
    faint = {**f, "focus": {**f["focus"], "outline_colour": [233, 236, 239, 1]}}
    assert "contrast" in graded(measure(focusables=[faint]))[0]
    none = {**f, "focus": {**f["focus"], "changed": [], "outline": False}}
    assert "no visible focus" in graded(measure(focusables=[none]))[0]
    ring = {**f, "focus": {**f["focus"], "outline_style": "auto", "outline_colour": [0, 0, 0, 0]}}
    assert graded(measure(focusables=[ring])) == []
    shadow = {**f, "focus": {**none["focus"], "shadow": True}}
    assert graded(measure(focusables=[shadow])) == []


def test_focus_order_follows_rows():
    f = measure()["focusables"][0]
    a = {**f, "el": "#a", "rect": {"x": 16, "y": 100, "w": 80, "h": 44}}
    b = {**f, "el": "#b", "rect": {"x": 120, "y": 102, "w": 80, "h": 44}}
    c = {**f, "el": "#c", "rect": {"x": 16, "y": 200, "w": 80, "h": 44}}
    spec = {**SPEC, "checks": {"focus_order": True}}
    steps = lambda *ix: [{"index": i, "wrapped": False} for i in ix] + [{"index": -1, "wrapped": True}]
    assert graded(measure(focusables=[a, b, c]), spec, steps(0, 1, 2)) == []
    assert "before" in graded(measure(focusables=[a, b, c]), spec, steps(1, 0, 2))[0]
    assert "never reaches" in graded(measure(focusables=[a, b, c]), spec, steps(0, 1))[0]


def test_type_scale_and_spacing():
    texts = [{**measure()["texts"][1], "size": s} for s in (12, 14, 16, 18, 20)]
    spec = {**SPEC, "checks": {"font_sizes": {"max": 4}}}
    assert graded(measure(texts=texts), spec)[0].startswith("[font_sizes]")
    spec = {**SPEC, "checks": {"spacing": {"selector": ".grid *", "base": 8}}}
    ok = [{"el": ".card", "values": {"padding-top": "16px", "margin-top": "0px", "row-gap": "normal"}}]
    assert graded(measure(spacing=ok), spec) == []
    off = [{"el": ".card", "values": {"padding-top": "14px"}}]
    assert "off the 8px grid" in graded(measure(spacing=off), spec)[0]


def test_theme_and_tokens():
    spec = {**SPEC, "checks": {"theme": {"tokens": ["--bg"]}}}
    dark = measure(dark=True, page_bg=[255, 255, 255, 1], root_styles={"--bg": "#fff"})
    assert "dark scheme page background is light" in graded(dark, spec)[0]
    assert "--bg" in graded(measure(root_styles={}), spec)[0]
    assert graded(measure(root_styles={"--bg": "#fff"}), spec) == []


def item(el, x, y, w, h, **kw):
    return {"el": el, "shown": True, "rect": {"x": x, "y": y, "w": w, "h": h}, "text": kw.get("text", ""),
            "styles": kw.get("styles", {}), "attrs": kw.get("attrs", {}), "name": kw.get("name", "")}


def test_expect_assertions():
    probes = {".card": {"items": [item(".card", 0, 0, 100, 50), item(".card", 110, 1, 100, 50), item(".card", 0, 60, 100, 50)]},
              "h1": {"items": [item("h1", 0, 0, 300, 40, text="Pricing plans", styles={"font-size": "32px"})]},
              ".box": {"items": [item(".box", 95, 0, 200, 10)]},
              "button": {"items": [item("button", 0, 0, 40, 40, name="")]}}
    m = measure(probes=probes)
    ok = lambda e: d.expect_failure(m, e)
    assert ok({"count": ".card", "equals": 3}) == ""
    assert ok({"columns": ".card", "equals": 2}) == ""
    assert "column" in ok({"columns": ".card", "equals": 3})
    assert ok({"text": "h1", "contains": "pricing"}) == ""
    assert ok({"px_at_least": "h1", "prop": "font-size", "px": 32}) == ""
    assert ok({"centered": ".box"}) == ""
    assert "no accessible name" in ok({"name": "button"})
    assert ok({"above": ["h1", ".card"]}) != ""
    assert "not shown" in ok({"visible": ".missing"})
    with pytest.raises(ValueError):
        d._kind({"bogus": 1})


def test_options_collect_probe_selectors_and_styles():
    spec = {"checks": {"spacing": {"selector": ".grid *", "base": 8}, "images_alt": {"informative": ["img.p"]},
                       "theme": {"tokens": ["--bg"]}, "overlap": {"ignore": ["main"]}},
            "expect": [{"style": ".a", "prop": "color", "in": []}, {"above": [".x", ".y"]}, {"attr": "#e", "name": "type", "equals": "email"}]}
    opts = d.options(spec)
    by = {p["selector"]: p for p in opts["probes"]}
    assert by[".a"]["styles"] == ["color"] and by["#e"]["attrs"] == ["type"] and {".x", ".y", "img.p"} <= set(by)
    assert opts["spacing"] == ".grid *" and opts["root_props"] == ["--bg"] and opts["overlap_ignore"] == "main"
    script = d.measure_script(opts)
    assert "__OPTIONS__;" not in script and '".grid *"' in script and script.startswith("try {")


# -- the renderer process (a fake) -------------------------------------------------------------------

def test_renderer_runs_the_tool_with_a_job_and_names_renders(tmp_path):
    (tmp_path / "index.html").write_text("<h1>x</h1>")
    seen = {}

    def fake_run(argv, **kw):
        job = json.loads(Path(argv[1]).read_text())
        seen.update(job)
        renders = [{"width": r["width"], "height": r["height"], "dark": r["dark"], "measure": measure(),
                    "focus": [], "snapshot": r["snapshot"], "error": None} for r in job["renders"]]
        return SimpleNamespace(returncode=0, stdout=json.dumps({"renders": renders}), stderr="")

    spec = {**SPEC, "viewports": ["phone", "desktop"], "schemes": ["light", "dark"]}
    out = d.Renderer(Path("/bin/true"), run=fake_run).render(tmp_path, "index.html", spec, snapshots=tmp_path / "s")
    assert [r["name"] for r in out] == ["phone-light", "phone-dark", "desktop-light", "desktop-dark"]
    assert seen["root"] == str(tmp_path.resolve()) and seen["tabs"] == 0
    assert [(r["width"], r["dark"]) for r in seen["renders"]] == [(390, False), (390, True), (1280, False), (1280, True)]
    assert seen["renders"][0]["snapshot"].endswith("s/phone-light.png")


def test_renderer_reports_missing_page_crash_and_script_error(tmp_path):
    r = d.Renderer(Path("/bin/true"), run=lambda *a, **k: None)
    assert r.render(tmp_path, "index.html", SPEC)[0]["error"] == "no page index.html"
    assert r.render(tmp_path, "../etc/passwd", SPEC)[0]["error"].startswith("no page")
    (tmp_path / "index.html").write_text("x")
    crash = d.Renderer(Path("/bin/true"), run=lambda *a, **k: SimpleNamespace(returncode=2, stdout="", stderr="boom"))
    assert "renderer failed: boom" in crash.render(tmp_path, "index.html", SPEC)[0]["error"]
    bad = {"renders": [{"measure": {"__error": "TypeError"}, "focus": [], "error": None}]}
    broken = d.Renderer(Path("/bin/true"), run=lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(bad), stderr=""))
    assert "measure script failed: TypeError" in broken.render(tmp_path, "index.html", SPEC)[0]["error"]
    assert d.grade(SPEC, broken.render(tmp_path, "index.html", SPEC))[0].startswith("[render]")


# -- building --------------------------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "darwin", reason="the renderer is macOS-only")
def test_build_compiles_once_then_uses_the_stamp(tmp_path):
    env = {"VEGA_HOME": str(tmp_path)}
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        Path(argv[-1]).write_text("binary")
        return SimpleNamespace(returncode=0)

    exe = d.ensure_built(env, run=fake_run, swiftc=lambda: True)
    assert exe == tmp_path / "native" / "design-render" / "design-render" and exe.read_text() == "binary"
    assert calls[0][:2] == ["xcrun", "swiftc"] and "-warnings-as-errors" in calls[0]
    d.ensure_built(env, run=fake_run, swiftc=lambda: True)
    assert len(calls) == 1  # the stamp matches: no rebuild


def test_unavailable_without_swiftc_or_macos(tmp_path, monkeypatch):
    env = {"VEGA_HOME": str(tmp_path)}
    if sys.platform == "darwin":
        with pytest.raises(d.RendererUnavailable, match="no swiftc"):
            d.ensure_built(env, swiftc=lambda: False)
    monkeypatch.setattr(d.sys, "platform", "linux")
    with pytest.raises(d.RendererUnavailable, match="needs macOS"):
        d.ensure_built(env)

    def failing(argv, **kw):
        raise subprocess.CalledProcessError(1, argv, stderr="error: nope")

    monkeypatch.setattr(d.sys, "platform", "darwin")
    with pytest.raises(d.RendererUnavailable, match="build failed: error: nope"):
        d.ensure_built(env, run=failing, swiftc=lambda: True)


def test_eval_harness_excludes_design_cases_when_the_renderer_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "availability", lambda env=None: "renderer unavailable: no swiftc")
    with pytest.raises(evals.FixtureError, match="renderer unavailable"):
        evals.build_fixture({"id": "dsn-x", "task": "t", "files": {"a.txt": "a"}, "design": SPEC}, tmp_path)
    evals.build_fixture({"id": "plain", "task": "t", "files": {"a.txt": "a"}}, tmp_path)  # others unaffected


def test_eval_harness_runs_the_design_check(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "check", lambda case, project: "design: [overflow] phone-light: too wide")
    outcome = evals.run_checks({"id": "dsn-x", "task": "t", "files": {}, "design": SPEC}, tmp_path, "")
    assert outcome.acceptance == "design: [overflow] phone-light: too wide"
    assert evals.suite_file("design").name == "design.jsonl" and "design" in evals.SUITES


# -- a real render ------------------------------------------------------------------------------------

@pytest.mark.needs_unsandboxed("WebKit's services (the renderer cannot compile its content rule list)")
@pytest.mark.skipif(not d.has_swiftc(), reason="renderer unavailable: needs macOS with swiftc")
def test_real_render_measures_a_page(tmp_path, monkeypatch):
    monkeypatch.setenv("VEGA_HOME", str(tmp_path / "home"))
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("""<!doctype html><html lang="en"><head><meta name="viewport" content="width=device-width">
<style>body{margin:16px;font:16px/1.5 system-ui} .wide{width:600px} .faint{color:#bbb}
button{padding:12px 16px;font:inherit} button:focus{outline:none} @media (prefers-color-scheme: dark){body{background:#111;color:#eee}}</style>
</head><body><h1>Hello</h1><p class="faint">faint text</p><button>One</button><div class="wide">wide</div>
<img src="https://example.com/x.png">
<script>fetch('https://example.com/').then(() => document.title = 'NET', () => document.title = 'BLOCKED')</script>
</body></html>""")
    spec = {**SPEC, "viewports": ["phone", "desktop"], "schemes": ["light", "dark"], "checks": {**SPEC["checks"], "focus_order": True},
            "expect": [{"text": "h1", "contains": "Hello"}]}
    renders = d.Renderer().render(site, "index.html", spec, snapshots=tmp_path / "snaps")
    assert all(r["error"] is None for r in renders), renders
    m = {r["name"]: r["measure"] for r in renders}
    assert m["phone-light"]["scroll_width"] > 390 and m["desktop-light"]["scroll_width"] == 1280
    assert m["phone-dark"]["dark"] and not m["phone-light"]["dark"]
    assert m["phone-light"]["title"] == "BLOCKED"  # no network
    assert (tmp_path / "snaps" / "phone-dark.png").stat().st_size > 1000
    failures = d.grade(spec, renders)
    kinds = {f.split("]")[0] + "]" for f in failures}
    assert {"[overflow]", "[contrast]", "[focus_visible]", "[images_alt]"} <= kinds
    assert not any(f.startswith(("[expect]", "[focus_order]", "[headings]")) for f in failures)
