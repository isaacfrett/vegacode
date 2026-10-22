"""Render a project's HTML page and measure it: the web pipeline's rendered checks.

The evals grade pages in macOS's own WebKit (headless Chrome cannot start in
the check sandbox, WebKit runs in-process with no sockets). This is the same engine host, cut to
what the web pipeline measures (load a page, evaluate script, let timers run, resize, light or
dark appearance), so Vega checks its own CSS/HTML/JS the way a browser lays it out.

Platform: WebKit through `osascript -l JavaScript` exists on macOS only. `available()` says
whether it can run here (macOS, osascript, and the sandbox, because the page runs model-written
script); elsewhere the web pipeline's static checks still run and its answer says the rendered
ones did not. A Linux/Windows port can add a headless engine behind the same `Renderer` methods.

Every render works on a throwaway copy of the project (sandbox._clone: copy-on-write, heavy
directories symlinked) in which the caller's file contents are placed, and runs inside sandbox-exec
with no network and writes only inside that copy. requestAnimationFrame never fires (the window is
off-screen), and page time advances only while a command runs (`wait`).
"""

from __future__ import annotations

import functools
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Self

from . import sandbox

OSASCRIPT = "/usr/bin/osascript"
TIMEOUT = 20.0
MAX_FILES = 400  # a project larger than this is not copied for rendering
MAX_BYTES = 30_000_000

SERVER = r"""
ObjC.import('WebKit'); ObjC.import('Cocoa');
const input = $.NSFileHandle.fileHandleWithStandardInput;
const output = $.NSFileHandle.fileHandleWithStandardOutput;
let win = null, view = null;
function spin(seconds) {
  $.NSRunLoop.currentRunLoop.runUntilDate($.NSDate.dateWithTimeIntervalSinceNow(seconds));
}
function send(obj) {
  output.writeData($(JSON.stringify(obj) + '\n').dataUsingEncoding($.NSUTF8StringEncoding));
}
function js(source) { return view.stringByEvaluatingJavaScriptFromString(source).js; }
function evaluate(expr, timeout) {
  js('window.__vegaResult = undefined; Promise.resolve().then(() => (' + expr + '))' +
     '.then(v => { window.__vegaResult = JSON.stringify({value: v === undefined ? null : v}); },' +
     ' e => { window.__vegaResult = JSON.stringify({error: String(e && e.stack || e)}); });');
  const deadline = Date.now() + timeout * 1000;
  for (;;) {
    const raw = js('window.__vegaResult');
    if (raw) { const result = JSON.parse(raw); if ('error' in result) throw new Error(result.error); return result.value; }
    if (Date.now() > deadline) throw new Error('page script did not finish in ' + timeout + 's');
    spin(0.005);
  }
}
function open(msg) {
  if (win) { win.orderOut(null); win.close; win = null; view = null; }
  $.NSApplication.sharedApplication;
  const rect = $.NSMakeRect(0, 0, msg.width, msg.height);
  view = $.WebView.alloc.initWithFrameFrameNameGroupName(rect, $(''), $('vega'));
  view.mainFrame.frameView.setAllowsScrolling(false);
  view.setAppearance($.NSAppearance.appearanceNamed(msg.dark ? $.NSAppearanceNameDarkAqua : $.NSAppearanceNameAqua));
  win = $.NSWindow.alloc.initWithContentRectStyleMaskBackingDefer(
    $.NSMakeRect(-20000, -20000, msg.width, msg.height), 0, $.NSBackingStoreBuffered, false);
  win.setReleasedWhenClosed(false);
  win.setContentView(view);
  win.orderBack(null);
  view.mainFrame.loadRequest($.NSURLRequest.requestWithURL($.NSURL.fileURLWithPath(msg.path)));
  const deadline = Date.now() + msg.timeout * 1000;
  while (view.isLoading || js('document.readyState') !== 'complete') {
    if (Date.now() > deadline) throw new Error('page did not load in ' + msg.timeout + 's');
    spin(0.005);
  }
  return evaluate('[innerWidth, innerHeight]', msg.timeout);
}
function resize(msg) {
  win.setContentSize($.NSMakeSize(msg.width, msg.height));
  view.setFrame($.NSMakeRect(0, 0, msg.width, msg.height));
  spin(0.02);
  return evaluate('[innerWidth, innerHeight]', 5);
}
const commands = {open, resize,
  eval: msg => evaluate(msg.expr, msg.timeout),
  wait: msg => { spin(msg.ms / 1000); return null; }};
function run() {
  let buffer = '';
  for (;;) {
    const data = input.availableData;
    if (data.length === 0) return '';
    buffer += $.NSString.alloc.initWithDataEncoding(data, $.NSUTF8StringEncoding).js;
    let newline;
    while ((newline = buffer.indexOf('\n')) >= 0) {
      const msg = JSON.parse(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
      if (msg.cmd === 'quit') return '';
      try { send({id: msg.id, value: commands[msg.cmd](msg)}); }
      catch (e) { send({id: msg.id, error: String(e && e.message || e)}); }
    }
  }
}
"""

# Runs first in every rendered page: uncaught errors and rejected promises are collected, and
# scrollIntoView calls are recorded (with their behavior) before they run.
PRELUDE = r"""window.__vegaErrors = [];
window.addEventListener('error', e => window.__vegaErrors.push(String(e.message || e.type) +
  (e.filename ? ' (' + String(e.filename).split('/').pop() + ':' + e.lineno + ')' : '')));
window.addEventListener('unhandledrejection', e => window.__vegaErrors.push('unhandled rejection: ' + String(e.reason)));
window.__vegaScrolls = [];
(() => { const real = Element.prototype.scrollIntoView;
  Element.prototype.scrollIntoView = function (options) {
    window.__vegaScrolls.push([this.id || this.tagName.toLowerCase(),
      options && typeof options === 'object' ? (options.behavior || 'auto') : 'auto']);
    return real.call(this, options && typeof options === 'object' ? {...options, behavior: 'auto'} : options);
  }; })();
"""


class RenderError(RuntimeError):
    pass


@functools.cache
def available() -> bool:
    """WebKit can render here: macOS with osascript, and the sandbox the page runs in."""
    return sys.platform == "darwin" and os.access(OSASCRIPT, os.X_OK) and sandbox.available()


def _small_enough(project: Path) -> bool:
    count = size = 0
    for root, dirs, files in os.walk(project):
        dirs[:] = [d for d in dirs if d not in sandbox.LINKED_DIRS | sandbox.SKIPPED_DIRS]
        for name in files:
            count += 1
            try:
                size += (Path(root) / name).stat().st_size
            except OSError:
                continue
            if count > MAX_FILES or size > MAX_BYTES:
                return False
    return True


def media_substitution(feature: str, on: bool) -> str:
    """A JS function `t => t'` that answers one media feature in a media query text: `(feature)`
    and `(feature: <its on value>)` become always-true (on) or always-false, and the feature's
    other value the opposite (WebKit cannot switch Reduce Motion per page; the evals' grader
    emulates it the same way)."""
    values = {"prefers-reduced-motion": ("reduce", "no-preference"),
              "prefers-contrast": ("more", "no-preference")}
    yes_value, no_value = values[feature]
    yes, no = ("(min-width: 0px)", "(max-width: 0px)") if on else ("(max-width: 0px)",
                                                                   "(min-width: 0px)")
    return (r"t => String(t).replace(/\(\s*" + feature + r"\s*(:\s*" + yes_value + r"\s*)?\)/gi, '"
            + yes + r"').replace(/\(\s*" + feature + r"\s*:\s*" + no_value + r"\s*\)/gi, '" + no
            + "')")


def emulation_script(feature: str, on: bool) -> str:
    """Run before the page's scripts: matchMedia answers `feature` as `on`."""
    sub = media_substitution(feature, on)
    return ("(() => { const sub = " + sub + "; const real = window.matchMedia.bind(window);"
            " window.matchMedia = q => real(sub(q)); })();\n")


def stylesheet_emulation(feature: str, on: bool) -> str:
    """Run after load: every stylesheet's media rules for `feature` answered as `on`."""
    sub = media_substitution(feature, on)
    return ("(() => { const sub = " + sub + "; const fix = rules => { for (const r of rules) {"
            " if (r.media && /" + feature + "/i.test(r.media.mediaText)) r.media.mediaText = "
            "sub(r.media.mediaText); if (r.cssRules) fix(r.cssRules); } };"
            " for (const s of document.styleSheets) { try { fix(s.cssRules); } catch (e) {} }"
            " return 1; })()")


class Renderer:
    """One WebKit host over one copy of the project with `files` (path -> content, None: absent)
    placed in it. Use as a context manager; `page()` opens a project HTML file."""

    def __init__(self, project: Path, files: dict[str, str | None], *,
                 timeout: float = TIMEOUT) -> None:
        if not available():
            raise RenderError("rendering needs macOS WebKit and the sandbox")
        if not _small_enough(project):
            raise RenderError(f"the project is larger than {MAX_FILES} files or "
                              f"{MAX_BYTES // 1_000_000} MB, so it is not copied for rendering")
        self.timeout = timeout
        self.dir = Path(tempfile.mkdtemp(prefix="vega-render-")).resolve()
        self.root = self.dir / "project"
        self.root.mkdir()
        self.proc: subprocess.Popen[bytes] | None = None
        try:
            sandbox._clone(project, self.root)
            for relative, content in files.items():
                path = self.root / relative
                if content is None:
                    if path.is_file():
                        path.unlink()
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.is_symlink():
                    path.unlink()
                path.write_text(content, encoding="utf-8")
            (self.dir / "tmp").mkdir()
            server = self.dir / "server.js"
            server.write_text(SERVER, encoding="utf-8")
            self._log = open(self.dir / "stderr.log", "wb")  # noqa: SIM115 - closed in close()
            argv = [sandbox.EXECUTABLE, "-p", sandbox._profile(self.dir), OSASCRIPT, "-l",
                    "JavaScript", str(server)]
            self.proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                start_new_session=True, cwd=self.root,
                env={**os.environ, "TMPDIR": str(self.dir / "tmp")})
        except Exception:
            self.close()
            raise
        self._buffer = b""
        self._next = 0
        self._variants = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def command(self, cmd: str, timeout: float | None = None, **fields: Any) -> Any:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise RenderError("the renderer is closed")
        timeout = timeout or self.timeout
        self._next += 1
        message = {"id": self._next, "cmd": cmd, "timeout": timeout, **fields}
        try:
            self.proc.stdin.write(json.dumps(message).encode() + b"\n")
            self.proc.stdin.flush()
        except OSError as exc:
            raise RenderError(f"WebKit host exited: {self._stderr()}") from exc
        deadline = time.monotonic() + timeout + 5
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buffer:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                raise RenderError(f"WebKit did not answer {cmd!r} in time")
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                raise RenderError(f"WebKit host exited: {self._stderr()}")
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        reply = json.loads(line)
        if "error" in reply:
            raise RenderError(f"{cmd}: {reply['error']}")
        return reply["value"]

    def _stderr(self) -> str:
        try:
            return (self.dir / "stderr.log").read_text(errors="replace")[-500:]
        except OSError:
            return ""

    def page(self, html: str, *, width: int = 1280, height: int = 800, dark: bool = False,
             head: str = "") -> Page:
        """Open the project's `html` file (a copy with PRELUDE and `head` run first in <head>)."""
        source = self.root / html
        text = source.read_text(encoding="utf-8")
        match = re.search(r"<head[^>]*>", text, flags=re.IGNORECASE)
        script = "<script>" + PRELUDE + head + "</script>"
        text = (text[:match.end()] + script + text[match.end():] if match
                else script + text)
        self._variants += 1
        variant = source.with_name(f".vega-render-{self._variants}-{source.name}")
        variant.write_text(text, encoding="utf-8")
        size = self.command("open", path=str(variant), width=width, height=height, dark=dark)
        if size != [width, height]:
            raise RenderError(f"viewport is {size}, expected {[width, height]}")
        return Page(self, width, height)

    def close(self) -> None:
        proc = self.proc
        if proc is not None:
            try:
                if proc.poll() is None and proc.stdin:
                    proc.stdin.write(b'{"cmd": "quit"}\n')
                    proc.stdin.flush()
                    proc.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            proc.wait()
            for stream in (proc.stdin, proc.stdout):
                if stream:
                    stream.close()
            self.proc = None
        log = getattr(self, "_log", None)
        if log is not None:
            log.close()
            self._log = None
        shutil.rmtree(self.dir, ignore_errors=True)


class Page:
    def __init__(self, renderer: Renderer, width: int, height: int) -> None:
        self.renderer, self.width, self.height = renderer, width, height

    def eval(self, expr: str) -> Any:
        """A JS expression's JSON value (a promise is awaited)."""
        return self.renderer.command("eval", expr=expr)

    def call(self, function: str, *args: Any) -> Any:
        return self.eval(f"({function})(...{json.dumps(list(args))})")

    def wait(self, ms: int) -> None:
        self.renderer.command("wait", ms=ms, timeout=ms / 1000 + 10)

    def resize(self, width: int, height: int | None = None) -> None:
        height = height or self.height
        self.renderer.command("resize", width=width, height=height)
        self.width, self.height = width, height

    def errors(self) -> list[str]:
        return self.eval("window.__vegaErrors || []")
