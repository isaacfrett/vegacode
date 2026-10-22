"""The Problems pane: what is wrong in the files Vega wrote, per file, cleared when fixed.

Sources: the diagnostics every write already runs (diagnostics.after_write: a parse check for
Python, JSON, TOML and YAML, plus pyflakes-level ruff rules), read from the write's tool result so
nothing runs twice; and `l` in the pane, which runs the project's own `ruff check` over the files
changed this session. A later write, an undo or a lint run replaces a file's problems.

Design (2026-10-15): grouped under a file row with the count right-aligned; each problem is one
line: line number right-aligned in four cells, the code (`syntax`, `F401`) in the error or warning
colour, then the message. The pane's title carries the error count in the error colour (the one
emphatic mark). Option ids: `P<n>` a problem.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from rich.cells import cell_len
from rich.text import Text
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from . import diagnostics, features, sandbox

RUFF_TIMEOUT = 20.0
LINTED = {".py", ".pyi"}
EMPTY = "No problems in files Vega wrote."
HINT = "l runs ruff on this session's files"
_PARSE = re.compile(r"^WARNING: the file (?:now has|is now) (a SyntaxError|invalid \w+)"
                    r"(?: at line (\d+))?: (.*?)(?:\. Fix it\.)?$", re.MULTILINE)
_LINT = re.compile(r"^(\d+):(\d+) (\S+) (.+)$")


@dataclass(frozen=True)
class Problem:
    path: str  # relative to the project, posix
    line: int | None
    column: int | None
    code: str  # "syntax", "json", "toml", "yaml", "ini" or a ruff rule ("F401")
    message: str
    severity: str  # "error" or "warning"
    source: str  # "write" (after each write) or "ruff" (the pane's lint run)


def from_write(path: str, result: str) -> list[Problem]:
    """The problems a write's tool result reports (diagnostics.after_write's notes)."""
    found: list[Problem] = []
    for kind, line, message in _PARSE.findall(result):
        code = "syntax" if kind == "a SyntaxError" else kind.split()[-1].lower()
        found.append(Problem(path, int(line) if line else None, None, code, message.strip(),
                             "error", "write"))
    if "Lint findings (advisory):" in result:
        block = result.split("Lint findings (advisory):", 1)[1]
        for raw in block.strip().splitlines():
            match = _LINT.match(raw.strip())
            if not match:
                break
            row, column, code, message = match.groups()
            found.append(Problem(path, int(row), int(column), code, message.strip(),
                                 _severity(code), "write"))
    return found


def _severity(code: str) -> str:
    return "error" if code in {"syntax", "E999"} or code.startswith("E9") else "warning"


def recheck(project: Path, path: str) -> list[Problem]:
    """Run the same checks as after a write, on the file as it is on disk now (after an undo).
    Never executes project code. Call from a worker: lint runs ruff with a timeout."""
    file = project / path
    try:
        text = file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []  # deleted (an undo removed a file Vega created): nothing to report
    problem = diagnostics.parse_problem(file, text)
    if problem is not None:
        message, line = problem
        message = re.sub(r"^a SyntaxError at line \d+: ", "", message)
        code = "syntax" if file.suffix == ".py" else file.suffix.lstrip(".").lower()
        return [Problem(path, line, None, code, message, "error", "write")]
    if file.suffix.lower() == ".py" and features.on("lint"):
        return from_write(path, diagnostics.after_write(project, file, text))
    return []


def ruff_check(project: Path, paths: list[str]) -> tuple[list[Problem], str]:
    """(problems, note) from the project's `ruff check` over `paths` (its own configuration and
    rules). Call from a worker; never raises."""
    files = [p for p in paths if Path(p).suffix.lower() in LINTED and (project / p).is_file()]
    if not files:
        return [], "no Python files changed this session"
    ruff = sandbox.find_python_tool(project, "ruff")
    if ruff is None:
        return [], "ruff not found (project .venv, Vega's environment or PATH)"
    argv = [ruff, "check", "--no-cache", "--quiet", "--output-format=json", "--", *files]
    argv = sandbox.readonly_argv(argv)
    try:
        result = subprocess.run(argv, check=False, cwd=project, capture_output=True, text=True,
                                timeout=RUFF_TIMEOUT, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return [], f"ruff did not finish in {RUFF_TIMEOUT:.0f} s"
    except OSError as exc:
        return [], f"ruff could not start: {exc.strerror or exc}"
    if result.returncode not in (0, 1):
        first = (result.stderr or result.stdout).strip().splitlines()
        return [], f"ruff failed: {first[0] if first else f'exit code {result.returncode}'}"
    try:
        items = json.loads(result.stdout or "[]")
    except ValueError:
        return [], "ruff output was not JSON"
    found = []
    root = project.resolve()
    for item in items:
        filename = Path(str(item.get("filename", "")))
        try:
            relative = filename.resolve().relative_to(root).as_posix()
        except ValueError:
            relative = filename.as_posix()
        location = item.get("location") or {}
        code = item.get("code") or "syntax"
        found.append(Problem(relative, location.get("row"), location.get("column"), code,
                             str(item.get("message", "")), _severity(code), "ruff"))
    return found, f"{len(found)} found in {len(files)} file{'s' if len(files) != 1 else ''}"


def ordered(problems: dict[str, list[Problem]]) -> list[Problem]:
    """Files with errors first, then by path; problems by line."""
    out: list[Problem] = []
    for path in sorted(problems, key=lambda p: (not any(x.severity == "error"
                                                        for x in problems[p]), p)):
        out.extend(sorted(problems[path], key=lambda x: (x.line or 0, x.column or 0, x.code)))
    return out


def _clip(text: str, width: int, ellipsis: str, *, keep_end: bool = False) -> str:
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    if keep_end:
        while text and cell_len(text) > width - 1:
            text = text[1:]
        return ellipsis + text
    while text and cell_len(text) > width - 1:
        text = text[:-1]
    return text + ellipsis


def file_line(path: str, problems: list[Problem], width: int, colours: dict[str, str],
              g: SimpleNamespace) -> Text:
    count = str(len(problems))
    errors = any(p.severity == "error" for p in problems)
    text = Text()
    text.append(_clip(path, max(4, width - len(count) - 1), g.ellipsis, keep_end=True)
                .ljust(max(4, width - len(count))), style=f"bold {colours['fg']}")
    text.append(count, style=colours["error"] if errors else colours["warning"])
    return text


def problem_option(problem: Problem, width: int, colours: dict[str, str],
                   g: SimpleNamespace) -> Text:
    """`   3 syntax invalid syntax`: line (four cells), code, message."""
    text = Text()
    text.append((str(problem.line) if problem.line else "-").rjust(4) + " ",
                style=colours["muted"])
    code_w = 6
    text.append(problem.code[:code_w].ljust(code_w) + " ",
                style=colours["error"] if problem.severity == "error" else colours["warning"])
    text.append(_clip(problem.message, max(4, width - 5 - code_w - 1), g.ellipsis),
                style=colours["fg"])
    return text


def fill(options: OptionList, rows: list[Problem], lint_note: str | None, running: bool,
         width: int, colours: dict[str, str], g: SimpleNamespace, frame: int) -> None:
    keep = options.highlighted_option.id if options.highlighted_option else None
    options.clear_options()
    if running:
        line = Text(g.spinner[frame % len(g.spinner)] + " ruff check" + g.ellipsis,
                    style=colours["accent"])
        options.add_option(Option(line, disabled=True))
    elif lint_note:
        options.add_option(Option(Text(_clip(lint_note, width, g.ellipsis),
                                       style=colours["muted"]), disabled=True))
    if not rows:
        text = Text(EMPTY, style=colours["muted"])
        text.append("\n" + HINT, style=colours["muted"])
        options.add_option(Option(text, disabled=True))
    current = None
    for number, problem in enumerate(rows):
        if problem.path != current:
            current = problem.path
            group = [p for p in rows if p.path == current]
            options.add_option(Option(file_line(current, group, width, colours, g),
                                      disabled=True))
        options.add_option(Option(problem_option(problem, width, colours, g), id=f"P{number}"))
    if keep:
        with contextlib.suppress(Exception):  # fixed: the row is gone
            options.highlighted = options.get_option_index(keep)


def preview(project: Path, problem: Problem, width: int, colours: dict[str, str],
            g: SimpleNamespace, around: int = 12) -> tuple[str, Text]:
    """The file around the problem's line: numbered, the line itself tinted, the message above."""
    where = f"{problem.path}:{problem.line}" if problem.line else problem.path
    title = f"{_clip(where, max(10, width - len(problem.code) - 4), g.ellipsis, keep_end=True)}" \
            f"  {problem.code}"
    text = Text()
    text.append(problem.message + "\n", style=colours["error"] if problem.severity == "error"
                else colours["warning"])
    source = "after Vega's write" if problem.source == "write" else "ruff check"
    text.append(f"{source}  l runs ruff again\n\n", style=colours["muted"])
    try:
        lines = (project / problem.path).read_text(encoding="utf-8",
                                                   errors="replace").splitlines()
    except OSError as exc:
        text.append(f"Could not read the file: {exc.strerror or exc}", style=colours["muted"])
        return title, text
    target = problem.line or 1
    first = max(1, target - around)
    last = min(len(lines), target + around)
    number_w = max(3, len(str(last)))
    body_w = max(10, width - number_w - 3)
    for number in range(first, last + 1):
        line = lines[number - 1].expandtabs(4)
        hit = number == target and problem.line is not None
        bg = colours.get("del_bg") if problem.severity == "error" else colours.get("panel")
        if hit and bg:
            text.append(str(number).rjust(number_w) + " ", style=f"bold {colours['fg']} on {bg}")
            text.append((g.prompt + " "), style=f"bold {colours['error']} on {bg}")
            text.append(_clip(line, body_w, g.ellipsis).ljust(body_w),
                        style=f"{colours['fg']} on {bg}")
        else:
            text.append(str(number).rjust(number_w) + "   ", style=colours["muted"])
            text.append(_clip(line, body_w, g.ellipsis), style=colours["fg"])
        if number != last:
            text.append("\n")
    if not lines:
        text.append("(the file is empty)", style=colours["muted"])
    return title, text


def stamp() -> str:
    return time.strftime("%H:%M")


__all__ = ["EMPTY", "Problem", "fill", "from_write", "ordered", "preview", "recheck",
           "ruff_check"]
