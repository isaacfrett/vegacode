"""Render a page in macOS WebKit and grade its design: the design suite's deterministic checks.

The renderer is an in-house Swift tool (native/macos/DesignRender.swift: AppKit + WKWebView, no
packages), built with `xcrun swiftc` into $VEGA_HOME/native/design-render on first use and
whenever its source or flags change (a sha256 stamp).
It loads one file:// page with read access to the project only and every non-file load blocked
(a WKContentRuleList), at the asked viewport sizes and colour schemes, runs design_measure.js in
an isolated script world, presses Tab through the page with real key events, and writes JSON
plus PNG snapshots for human review. This module turns that JSON into pass/fail checks:

    overflow       no horizontal scrolling: document width <= viewport width
    overlap        no text run or control drawn over an unrelated one, and no text cut off by an
                   ancestor that hides overflow
    contrast       WCAG AA text contrast: 4.5:1, 3:1 for large text (24px, or 18.66px bold)
    tap_targets    every control (not a link inside a sentence) at least `min` CSS px (24; 44 ideal)
    focus_visible  keyboard focus changes the control's look (outline >= 3:1 against what is
                   behind it, a box-shadow, or a border / background / colour / underline change)
    focus_order    Tab visits controls in reading order (rows top to bottom, left to right) and
                   reaches every one
    font_sizes     at most `max` distinct font sizes (a type scale, not ad-hoc sizes)
    spacing        margins (vertical), paddings and gaps of `selector` on a `base` px grid
    images_alt     every <img> has alt; `informative` ones non-empty, `decorative` ones empty
    headings       one h1, first; no level skipped going down
    labels         every form field has an accessible name (a placeholder is not one)
    theme          the page background is dark in the dark scheme and light in the light one

and the case's functional `expect` assertions (count, columns, visible, hidden, text, above,
left_of, min_width, max_width, same_top, style, attr, name, px_at_least, centered) on selectors,
per render (an item with `viewport` / `scheme` applies to those renders only).

Where the renderer cannot run (not macOS, no swiftc: CI, Linux) `RendererUnavailable` is raised;
the eval harness turns it into a fixture-style error before any attempt runs, so it never counts
as a model failure.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "native" / "macos" / "DesignRender.swift"
MEASURE_JS = HERE / "design_measure.js"
SWIFT_FLAGS = ("-O", "-swift-version", "6", "-strict-concurrency=complete", "-warnings-as-errors",
               "-parse-as-library")
EXE = "design-render"
STAMP = "source.sha256"
TIMEOUT = 25.0  # per render (load + measure + tabbing + snapshot)
TABS = 40  # Tab presses per render at most (focus order)
VIEWPORTS = {"phone": (390, 844), "tablet": (768, 1024), "desktop": (1280, 800)}
SCHEMES = ("light", "dark")
UNAVAILABLE = "renderer unavailable"
FOCUS_JS = """
const list = window.__vegaFocusables || [];
const seen = window.__vegaSeen || (window.__vegaSeen = []);
const a = document.activeElement;
const wrapped = !a || a === document.body || a === document.documentElement || seen.includes(a);
seen.push(a);
return {index: list.indexOf(a), el: a ? a.tagName.toLowerCase() + (a.id ? '#' + a.id : '') : null, wrapped};
"""
CHECKS = ("overflow", "overlap", "contrast", "tap_targets", "focus_visible", "focus_order", "font_sizes",
          "spacing", "images_alt", "headings", "labels", "theme")


class RendererUnavailable(Exception):
    """The renderer cannot run here (not macOS, no swiftc, the build failed)."""


Run = Callable[..., Any]


# -- building ---------------------------------------------------------------------------------------

def vega_home(env: dict[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("VEGA_HOME") or Path.home() / ".vega")


def binary(env: dict[str, str] | None = None) -> Path:
    return vega_home(env) / "native" / "design-render" / EXE


def source_hash(source: bytes) -> str:
    h = hashlib.sha256(source)
    h.update("\0".join(SWIFT_FLAGS).encode())
    return h.hexdigest()


def swiftc_command(source: Path, out: Path) -> list[str]:
    return ["xcrun", "swiftc", *SWIFT_FLAGS, str(source), "-o", str(out)]


def has_swiftc(run: Run = subprocess.run) -> bool:
    if sys.platform != "darwin" or not shutil.which("xcrun"):
        return False
    try:
        return run(["xcrun", "--find", "swiftc"], capture_output=True, stdin=subprocess.DEVNULL,
                   timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def needs_build(exe: Path, digest: str) -> bool:
    stamp = exe.parent / STAMP
    return not (exe.is_file() and stamp.is_file() and stamp.read_text().strip() == digest)


def ensure_built(env: dict[str, str] | None = None, *, run: Run = subprocess.run,
                 swiftc: Callable[[], bool] | None = None) -> Path:
    """The renderer binary, compiled first when missing or stale. Raises RendererUnavailable."""
    if sys.platform != "darwin":
        raise RendererUnavailable(f"{UNAVAILABLE}: WebKit rendering needs macOS (this is {sys.platform})")
    exe = binary(env)
    digest = source_hash(SOURCE.read_bytes())
    if not needs_build(exe, digest):
        return exe
    if not (swiftc() if swiftc else has_swiftc(run)):
        raise RendererUnavailable(f"{UNAVAILABLE}: no swiftc (install the Xcode command line tools)")
    exe.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".building-", dir=exe.parent))
    try:
        new = work / EXE
        try:
            run(swiftc_command(SOURCE, new), check=True, capture_output=True, text=True,
                stdin=subprocess.DEVNULL, timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            raise RendererUnavailable(f"{UNAVAILABLE}: build failed: {detail.strip()[-300:]}") from exc
        (exe.parent / STAMP).unlink(missing_ok=True)
        os.replace(new, exe)
        (exe.parent / STAMP).write_text(digest + "\n")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return exe


# -- rendering --------------------------------------------------------------------------------------

def render_specs(spec: dict[str, Any]) -> list[tuple[str, int, int, bool]]:
    """(name, width, height, dark) for every viewport x scheme the case asks for."""
    out = []
    for viewport in spec.get("viewports", ["phone", "desktop"]):
        width, height = VIEWPORTS[viewport]
        for scheme in spec.get("schemes", ["light"]):
            out.append((f"{viewport}-{scheme}", width, height, scheme == "dark"))
    return out


def options(spec: dict[str, Any]) -> dict[str, Any]:
    """What design_measure.js needs beyond its fixed measurements: probe selectors and styles."""
    checks = spec.get("checks", {})
    probes: dict[str, dict[str, set[str]]] = {}

    def want(selector: str, styles: Iterable[str] = (), attrs: Iterable[str] = ()) -> None:
        entry = probes.setdefault(selector, {"styles": set(), "attrs": set()})
        entry["styles"].update(styles)
        entry["attrs"].update(attrs)

    for item in spec.get("expect", []):
        kind = _kind(item)
        target = item[kind]
        for selector in target if isinstance(target, list) else [target]:
            want(selector, [item["prop"]] if kind in ("style", "px_at_least") else [],
             [item["name"]] if kind == "attr" else [])
    images = checks.get("images_alt")
    if isinstance(images, dict):
        for key in ("informative", "decorative"):
            for selector in images.get(key, []):
                want(selector, attrs=["alt", "aria-hidden", "role"])
    spacing = checks.get("spacing")
    overlap = checks.get("overlap")
    return {
        "probes": [{"selector": s, "styles": sorted(v["styles"]), "attrs": sorted(v["attrs"])}
                   for s, v in probes.items()],
        "spacing": spacing.get("selector") if isinstance(spacing, dict) else None,
        "overlap_ignore": ", ".join(overlap.get("ignore", [])) if isinstance(overlap, dict) else "",
        "root_props": list(checks.get("theme", {}).get("tokens", [])) if isinstance(checks.get("theme"), dict) else [],
    }


def measure_script(opts: dict[str, Any]) -> str:
    """design_measure.js with its options, wrapped so a page that breaks it reports why."""
    body = MEASURE_JS.read_text(encoding="utf-8").replace("= __OPTIONS__;", "= " + json.dumps(opts) + ";", 1)
    return ("try { return await (async () => {\n" + body + "\n})(); }\n"
            "catch (e) { return {__error: String(e) + ' ' + String(e && e.stack || '').slice(0, 300)}; }")


class Renderer:
    """Runs the built Swift tool on one page. `run` is injectable (tests use a fake)."""

    def __init__(self, exe: Path | None = None, *, run: Run = subprocess.run,
                 env: dict[str, str] | None = None) -> None:
        self.exe = exe
        self.run = run
        self.env = env

    def render(self, root: Path, page: str, spec: dict[str, Any], *,
               snapshots: Path | None = None) -> list[dict[str, Any]]:
        """Render `page` (relative to `root`) as `spec` asks; one dict per render with its name."""
        exe = self.exe or ensure_built(self.env, run=self.run)
        root = root.resolve()
        target = (root / page).resolve()
        if not target.is_file() or root not in target.parents:
            return [{"name": name, "error": f"no page {page}"} for name, *_ in render_specs(spec)]
        specs = render_specs(spec)
        if snapshots is not None:
            snapshots.mkdir(parents=True, exist_ok=True)
        job = {"page": str(target), "root": str(root), "timeout": TIMEOUT,
               "script": measure_script(options(spec)), "focus_script": FOCUS_JS,
               "tabs": TABS if "focus_order" in spec.get("checks", {}) else 0,
               "renders": [{"width": w, "height": h, "dark": dark,
                            "snapshot": str(snapshots / f"{name}.png") if snapshots else None}
                           for name, w, h, dark in specs]}
        with tempfile.TemporaryDirectory(prefix="vega-design-") as tmp:
            job_file = Path(tmp) / "job.json"
            job_file.write_text(json.dumps(job), encoding="utf-8")
            try:
                result = self.run([str(exe), str(job_file)], capture_output=True, text=True,
                                  stdin=subprocess.DEVNULL, timeout=TIMEOUT * len(specs) + 30)
            except subprocess.TimeoutExpired:
                return [{"name": name, "error": "renderer timed out"} for name, *_ in specs]
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-300:]
            return [{"name": name, "error": f"renderer failed: {detail}"} for name, *_ in specs]
        renders = json.loads(result.stdout)["renders"]
        out = []
        for (name, *_), r in zip(specs, renders, strict=True):
            measured = r.get("measure")
            if isinstance(measured, dict) and "__error" in measured:
                r = {**r, "error": f"measure script failed: {measured['__error']}"}
            out.append({"name": name, **r})
        return out


# -- grading ----------------------------------------------------------------------------------------

def luminance(rgb: list[float]) -> float:
    def channel(c: float) -> float:
        c = c / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: list[float], b: list[float]) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def over(top: list[float], bottom: list[float]) -> list[float]:
    alpha = top[3] if len(top) > 3 else 1
    return [top[i] * alpha + bottom[i] * (1 - alpha) for i in range(3)]


def text_contrast(item: dict[str, Any]) -> float:
    """The worst ratio of the text's colour against every candidate background behind it."""
    return min(contrast(over(item["fg"], bg), bg) for bg in item["bg"])


def large_text(item: dict[str, Any]) -> bool:
    return item["size"] >= 24 or (item["size"] >= 18.66 and item["weight"] >= 700)


def _px(value: str) -> float | None:
    match = re.fullmatch(r"(-?[\d.]+)px", (value or "").strip())
    return float(match.group(1)) if match else None


EXPECT_KINDS = ("count", "columns", "visible", "hidden", "text", "above", "left_of", "min_width",
                "max_width", "same_top", "style", "attr", "name", "px_at_least", "centered")


def _kind(item: dict[str, Any]) -> str:
    for kind in EXPECT_KINDS:
        if kind in item:
            return kind
    raise ValueError(f"unknown expect item {item}")


def _check_overflow(m: dict[str, Any], _: Any) -> list[str]:
    if m["scroll_width"] <= m["viewport"]["w"] + 1:
        return []
    widest = ", ".join(f"{o['el']} (right edge {o['right']})" for o in m["overflowing"][:2])
    return [f"page is {m['scroll_width']}px wide in a {m['viewport']['w']}px viewport"
            + (f"; widest: {widest}" if widest else "")]


def _check_overlap(m: dict[str, Any], _: Any) -> list[str]:
    out = [f"{o['a']} overlaps {o['b']} ({o['w']}x{o['h']}px)" for o in m["overlaps"][:2]]
    out += [f"{c['el']} \"{c['text']}\" is cut off by {c['by']} (overflow hidden)" for c in m.get("clipped", [])[:2]]
    return out


def _check_contrast(m: dict[str, Any], opts: Any) -> list[str]:
    opts = opts if isinstance(opts, dict) else {}
    normal, large = float(opts.get("min", 4.5)), float(opts.get("large_min", 3.0))
    bad = []
    for item in m["texts"]:
        if item.get("bg_image") or item.get("disabled"):
            continue
        need = large if large_text(item) else normal
        ratio = text_contrast(item)
        if ratio + 1e-9 < need:
            bad.append((ratio, f"{item['el']} \"{item['text'][:24]}\" {ratio:.2f}:1 (needs {need}:1)"))
    return [text for _, text in sorted(bad)[:3]]


def _check_tap_targets(m: dict[str, Any], opts: Any) -> list[str]:
    least = float(opts.get("min", 24)) if isinstance(opts, dict) else 24.0
    bad = [f"{f['el']} is {f['target']['w']:g}x{f['target']['h']:g}px (min {least:g})"
           for f in m["focusables"] if not f["inline_link"] and not f["disabled"]
           and min(f["target"]["w"], f["target"]["h"]) + 0.05 < least]
    return bad[:3]


def focus_problem(f: dict[str, Any], min_ratio: float = 3.0) -> str:
    """'' when keyboard focus visibly changes the control: an outline it did not have (the system
    ring, or an author outline with `min_ratio` contrast against what is behind it), a box-shadow,
    or a border, background, colour or underline change. Else what is wrong."""
    focus = f["focus"]
    changed = set(focus["changed"])
    if focus["shadow"] or changed & {"border-top-color", "border-bottom-color", "border-top-width",
                                     "border-bottom-width", "background-color", "color",
                                     "text-decoration-line", "text-decoration-thickness"}:
        return ""
    if focus["outline"] and changed & {"outline-style", "outline-width", "outline-color"}:
        if focus.get("outline_style") == "auto":
            return ""
        colour, behind = focus["outline_colour"], focus["bg"]
        ratio = contrast(over(colour, behind), behind) if colour[3] > 0 else 1.0
        if ratio >= min_ratio:
            return ""
        return f"focus outline contrast {ratio:.2f}:1 against its background (needs {min_ratio:g}:1)"
    return "no visible focus indicator"


def _check_focus_visible(m: dict[str, Any], opts: Any) -> list[str]:
    ratio = float(opts.get("min_ratio", 3.0)) if isinstance(opts, dict) else 3.0
    bad = [f"{f['el']}: {problem}" for f in m["focusables"]
           if not f["disabled"] and (problem := focus_problem(f, ratio))]
    return bad[:3]


def reading_order(boxes: list[dict[str, Any]], tolerance: float = 8.0) -> list[int]:
    """Indexes of `boxes` in reading order: rows (tops within `tolerance`) top to bottom, each left
    to right."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i]["y"], boxes[i]["x"]))
    rows: list[list[int]] = []
    for i in order:
        if rows and abs(boxes[i]["y"] - boxes[rows[-1][0]]["y"]) <= tolerance:
            rows[-1].append(i)
        else:
            rows.append([i])
    return [i for row in rows for i in sorted(row, key=lambda i: boxes[i]["x"])]


def _check_focus_order(m: dict[str, Any], _: Any, focus: list[Any]) -> list[str]:
    visited: list[int] = []
    for step in focus:
        if not isinstance(step, dict) or step.get("wrapped"):
            if visited:
                break
            continue
        if step["index"] >= 0 and step["index"] not in visited:
            visited.append(step["index"])
    fs = m["focusables"]
    reachable = [i for i, f in enumerate(fs) if not f["disabled"] and (f["tabindex"] is None or f["tabindex"] >= 0)]
    missing = [fs[i]["el"] for i in reachable if i not in visited]
    if missing and len(visited) < TABS - 1:
        return [f"Tab never reaches {', '.join(missing[:2])}"]
    expected = [reachable[i] for i in reading_order([fs[i]["rect"] for i in reachable])]
    expected = [i for i in expected if i in visited]
    for got, want in zip(visited, expected):
        if got != want:
            return [f"Tab goes to {fs[got]['el']} before {fs[want]['el']} (reading order)"]
    return []


def _check_font_sizes(m: dict[str, Any], opts: Any) -> list[str]:
    most = int(opts.get("max", 6)) if isinstance(opts, dict) else 6
    sizes = sorted({round(t["size"] * 2) / 2 for t in m["texts"]})
    if len(sizes) <= most:
        return []
    return [f"{len(sizes)} distinct font sizes (max {most}): {', '.join(f'{s:g}' for s in sizes)}px"]


def _check_spacing(m: dict[str, Any], opts: Any) -> list[str]:
    base = float(opts.get("base", 8))
    bad = []
    for item in m["spacing"]:
        for prop, value in item["values"].items():
            px = _px(value)
            if px is None or abs(px) < 0.01:
                continue
            rem = abs(px) % base
            if min(rem, base - rem) > 0.26:
                bad.append(f"{item['el']} {prop} {px:g}px is off the {base:g}px grid")
    if not m["spacing"]:
        return [f"nothing matches the spacing selector {opts.get('selector')!r}"]
    return bad[:3]


def _check_images_alt(m: dict[str, Any], opts: Any) -> list[str]:
    bad = [f"{img['el']} has no alt attribute" for img in m["images"]
           if img["tag"] == "img" and not img["has_alt"] and not img["hidden"]]
    if isinstance(opts, dict):
        for selector in opts.get("informative", []):
            for item in _items(m, selector):
                if not (item["attrs"].get("alt") or "").strip():
                    bad.append(f"{item['el']} needs a descriptive alt text")
        for selector in opts.get("decorative", []):
            for item in _items(m, selector):
                a = item["attrs"]
                if (a.get("alt") or "").strip() and a.get("aria-hidden") != "true" and a.get("role") != "presentation":
                    bad.append(f"{item['el']} is decorative: alt=\"\" (or aria-hidden) expected")
    return bad[:3]


def _check_headings(m: dict[str, Any], _: Any) -> list[str]:
    hs = m["headings"]
    if not hs:
        return ["no headings"]
    ones = [h for h in hs if h["level"] == 1]
    if len(ones) != 1:
        return [f"{len(ones)} h1 headings (exactly one expected)"]
    if hs[0]["level"] != 1:
        return [f"first heading is h{hs[0]['level']} \"{hs[0]['text'][:30]}\", not the h1"]
    for prev, cur in itertools.pairwise(hs):
        if cur["level"] > prev["level"] + 1:
            return [f"heading level skips from h{prev['level']} to h{cur['level']} at \"{cur['text'][:30]}\""]
    return []


def _check_labels(m: dict[str, Any], _: Any) -> list[str]:
    return [f"{c['el']} ({c['type']}) has no label" for c in m["form_controls"] if not c["name"]][:3]


def _check_theme(m: dict[str, Any], opts: Any) -> list[str]:
    opts = opts if isinstance(opts, dict) else {}
    lum = luminance(m["page_bg"])
    if m["dark"] and lum > float(opts.get("dark_max", 0.2)):
        return [f"dark scheme page background is light (luminance {lum:.2f})"]
    if not m["dark"] and lum < float(opts.get("light_min", 0.5)):
        return [f"light scheme page background is dark (luminance {lum:.2f})"]
    for token in opts.get("tokens", []):
        if not m["root_styles"].get(token):
            return [f"custom property {token} is not defined on :root"]
    return []


GRADERS: dict[str, Callable[..., list[str]]] = {
    "overflow": _check_overflow, "overlap": _check_overlap, "contrast": _check_contrast,
    "tap_targets": _check_tap_targets, "focus_visible": _check_focus_visible,
    "font_sizes": _check_font_sizes, "spacing": _check_spacing, "images_alt": _check_images_alt,
    "headings": _check_headings, "labels": _check_labels, "theme": _check_theme,
}


def _items(m: dict[str, Any], selector: str) -> list[dict[str, Any]]:
    return (m["probes"].get(selector) or {}).get("items", [])


def _shown(m: dict[str, Any], selector: str) -> list[dict[str, Any]]:
    return [item for item in _items(m, selector) if item["shown"]]


def columns(rects: list[dict[str, Any]], tolerance: float = 4.0) -> int:
    """How many items share the first row (tops within `tolerance` of the topmost)."""
    if not rects:
        return 0
    top = min(r["y"] for r in rects)
    return sum(abs(r["y"] - top) <= tolerance for r in rects)


def expect_failure(m: dict[str, Any], item: dict[str, Any]) -> str:
    """'' when the functional assertion holds on this render, else why not."""
    kind = _kind(item)
    target = item[kind]
    if kind in ("above", "left_of"):
        a, b = (_shown(m, s) for s in target)
        if not a or not b:
            return f"{target[0] if not a else target[1]} is not shown"
        if kind == "above":
            ok = max(r["rect"]["y"] + r["rect"]["h"] for r in a) <= min(r["rect"]["y"] for r in b) + 1
            return "" if ok else f"{target[0]} is not above {target[1]}"
        ok = max(r["rect"]["x"] + r["rect"]["w"] for r in a) <= min(r["rect"]["x"] for r in b) + 1
        return "" if ok else f"{target[0]} is not left of {target[1]}"
    items = _items(m, target)
    shown = [i for i in items if i["shown"]]
    if kind == "count":
        n = len(shown)
        if "equals" in item and n != item["equals"]:
            return f"{n} {target} shown (expected {item['equals']})"
        if n < item.get("min", 0):
            return f"{n} {target} shown (expected at least {item['min']})"
        return ""
    if kind == "hidden":
        return f"{target} is shown" if shown else ""
    if not shown:
        return f"{target} is not shown"
    if kind == "visible":
        return ""
    if kind == "columns":
        n = columns([i["rect"] for i in shown])
        return "" if n == item["equals"] else f"{target} in {n} column(s) (expected {item['equals']})"
    if kind == "text":
        joined = " ".join(i["text"] for i in shown).casefold()
        return "" if item["contains"].casefold() in joined else f"{target} does not say {item['contains']!r}"
    if kind in ("min_width", "max_width"):
        widths = [i["rect"]["w"] for i in shown]
        if kind == "min_width" and min(widths) + 0.5 < item["px"]:
            return f"{target} is {min(widths):g}px wide (min {item['px']})"
        if kind == "max_width" and max(widths) - 0.5 > item["px"]:
            return f"{target} is {max(widths):g}px wide (max {item['px']})"
        return ""
    if kind == "same_top":
        tops = [i["rect"]["y"] for i in shown]
        return "" if max(tops) - min(tops) <= 4 else f"{target} are not on one row"
    if kind == "style":
        values = {i["styles"].get(item["prop"], "") for i in shown}
        allowed = set(item.get("in", [item.get("equals")]))
        wrong = sorted(values - allowed)
        return f"{target} {item['prop']} is {wrong[0]!r} (expected {' or '.join(sorted(allowed))})" if wrong else ""
    if kind == "px_at_least":
        values = [(_px(i["styles"].get(item["prop"], "")) or 0.0, i["el"]) for i in shown]
        low, where = min(values)
        return "" if low + 0.05 >= item["px"] else f"{where} {item['prop']} is {low:g}px (min {item['px']:g})"
    if kind == "centered":
        width = m["viewport"]["w"]
        off = [(abs(i["rect"]["x"] + i["rect"]["w"] / 2 - width / 2), i["el"]) for i in shown]
        worst, where = max(off)
        return "" if worst <= 2 else f"{where} is {worst:.0f}px off the horizontal centre"
    if kind == "attr":
        values = {i["attrs"].get(item["name"]) for i in shown}
        return "" if values == {item["equals"]} else f"{target} [{item['name']}] is {sorted(map(str, values))}"
    if kind == "name":
        bad = [i["el"] for i in shown if not i["name"]]
        if bad:
            return f"{bad[0]} has no accessible name"
        want = item.get("contains", "").casefold()
        wrong = [i for i in shown if want not in i["name"].casefold()]
        return f"{wrong[0]['el']} is named {wrong[0]['name']!r} (should mention {item['contains']!r})" if wrong else ""
    raise ValueError(kind)


def _applies(item: dict[str, Any], name: str) -> bool:
    viewport, scheme = name.split("-")
    return item.get("viewport", viewport) == viewport and item.get("scheme", scheme) == scheme


def grade(spec: dict[str, Any], renders: list[dict[str, Any]]) -> list[str]:
    """Every failure as "[check] render: detail" (empty: the design passes)."""
    failures: list[str] = []
    checks = spec.get("checks", {})
    for r in renders:
        name = r["name"]
        if r.get("error"):
            failures.append(f"[render] {name}: {r['error']}")
            continue
        m = r["measure"]
        for check, opts in checks.items():
            if not opts:
                continue
            if check == "focus_order":
                problems = _check_focus_order(m, opts, r.get("focus") or [])
            else:
                problems = GRADERS[check](m, opts)
            failures += [f"[{check}] {name}: {p}" for p in problems]
        for item in spec.get("expect", []):
            if _applies(item, name):
                problem = expect_failure(m, item)
                if problem:
                    failures.append(f"[expect] {name}: {problem}")
    return failures


def snapshot_dir(case_id: str, env: dict[str, str] | None = None) -> Path:
    """Where a case's PNGs go for human review (overwritten by the next render of the case)."""
    env = os.environ if env is None else env
    base = env.get("VEGA_DESIGN_SNAPSHOTS") or str(vega_home(env) / "evals" / "design-snapshots")
    return Path(base) / re.sub(r"[^\w.-]", "_", case_id)


def check(case: dict[str, Any], project: Path, *, renderer: Renderer | None = None,
          snapshots: Path | None = None) -> str:
    """The `design` check of an eval case on `project`: '' passes, else up to three failures.
    Raises RendererUnavailable where WebKit rendering cannot run."""
    spec = case["design"]
    renderer = renderer or Renderer()
    if snapshots is None:
        snapshots = snapshot_dir(case.get("id", "case"))
    renders = renderer.render(project, spec.get("page", "index.html"), spec, snapshots=snapshots)
    failures = grade(spec, renders)
    if not failures:
        return ""
    more = f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""
    return "design: " + "; ".join(failures[:3]) + more


def availability(env: dict[str, str] | None = None) -> str:
    """'' when the renderer can run here (building it if needed), else why not."""
    try:
        ensure_built(env)
    except RendererUnavailable as exc:
        return str(exc)
    return ""
