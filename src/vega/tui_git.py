"""The Git pane: branch and upstream, what is staged, unstaged and untracked, and the last commits.

Design (2026-10-15): the branch name is the one bold element; ahead/behind counts follow it in
the accent (ahead) and warning (behind) colours; file rows are `git status --short`'s own two
status letters (index letter in the success colour, work-tree letter in the warning colour,
conflicts in the error colour) so a Git user reads them without a legend. Commits are dim except
their subject. Reading runs `git status --porcelain=v2 --branch -z` and `git log -5` in a worker
with a timeout and GIT_OPTIONAL_LOCKS=0, so a refresh never takes the index lock or blocks the UI.
Option ids: `G<n>` a file, `C<hash>` a commit. Files whose only change is Vega's own (the VEGA.md
it drafted, its marked .gitignore block) are counted apart (`2 by Vega`) and listed last, not as
the user's changes (2026-10-17).
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from rich.cells import cell_len
from rich.text import Text
from textual.widgets import OptionList
from textual.widgets.option_list import Option

TIMEOUT = 5.0  # seconds for each git command
MAX_FILES = 60  # file rows shown; the rest are counted
COMMITS = 5
HASH_FROM = 36  # pane width (cells) from which commit rows show the short hash
PREVIEW_BYTES = 200_000
NOT_A_REPO = "Not a git repository. Run git init to see branch, changes and commits here."


@dataclass
class GitFile:
    xy: str  # git's two status letters: index, work tree ("??" untracked)
    path: str
    orig: str = ""  # the old path of a rename or copy
    vega: bool = False  # only Vega's own change (VEGA.md, its .gitignore block): listed apart

    @property
    def staged(self) -> bool:
        return self.xy[0] not in {".", "?", "!"} and not self.conflict

    @property
    def unstaged(self) -> bool:
        return self.xy[1] not in {".", "?", "!"} and not self.conflict

    @property
    def untracked(self) -> bool:
        return self.xy == "??"

    @property
    def conflict(self) -> bool:
        return "U" in self.xy or self.xy in {"AA", "DD"}


@dataclass
class GitState:
    repo: bool = True
    error: str = ""
    branch: str = ""
    detached: bool = False
    oid: str = ""
    upstream: str = ""
    ahead: int = 0
    behind: int = 0
    files: list[GitFile] = field(default_factory=list)
    commits: list[tuple[str, int, str]] = field(default_factory=list)  # (short hash, unix, subject)
    read_at: float = 0.0

    def counts(self) -> dict[str, int]:
        """The user's changes by kind; Vega's own files (see GitFile.vega) only under "vega"."""
        mine = [f for f in self.files if not f.vega]
        return {
            "staged": sum(f.staged for f in mine),
            "unstaged": sum(f.unstaged for f in mine),
            "new": sum(f.untracked for f in mine),
            "conflict": sum(f.conflict for f in mine),
            "vega": len(self.files) - len(mine),
        }


def _git(project: Path, *args: str, timeout: float = TIMEOUT) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C", "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.run(["git", "-C", str(project), *args], check=False, capture_output=True,
                          text=True,
                          timeout=timeout, env=env, encoding="utf-8", errors="replace",
                          stdin=subprocess.DEVNULL)


def parse_status(raw: str, state: GitState) -> None:
    """Fill `state` from `git status --porcelain=v2 --branch -z` output."""
    records = raw.split("\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if record.startswith("# "):
            key, _, value = record[2:].partition(" ")
            if key == "branch.oid":
                state.oid = "" if value == "(initial)" else value
            elif key == "branch.head":
                state.detached = value == "(detached)"
                state.branch = "" if state.detached else value
            elif key == "branch.upstream":
                state.upstream = value
            elif key == "branch.ab":
                for part in value.split():
                    if part.startswith("+"):
                        state.ahead = int(part[1:] or 0)
                    elif part.startswith("-"):
                        state.behind = int(part[1:] or 0)
            continue
        kind = record[:1]
        if kind == "1":
            parts = record.split(" ", 8)
            state.files.append(GitFile(parts[1], parts[8]))
        elif kind == "2":
            parts = record.split(" ", 9)
            orig = records[index] if index < len(records) else ""
            index += 1  # -z puts the original path in its own record
            state.files.append(GitFile(parts[1], parts[9], orig))
        elif kind == "u":
            parts = record.split(" ", 10)
            state.files.append(GitFile(parts[1], parts[10]))
        elif kind == "?":
            state.files.append(GitFile("??", record[2:]))


def read_state(project: Path) -> GitState:
    """Branch, files and recent commits. Never raises: a missing git, a timeout or a folder
    outside any repository come back as a state with `repo=False` or `error`."""
    state = GitState(read_at=time.time())
    try:
        status = _git(project, "status", "--porcelain=v2", "--branch", "-z",
                      "--untracked-files=normal")
    except FileNotFoundError:
        return GitState(repo=False, error="git is not installed", read_at=state.read_at)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return GitState(error=f"git status did not finish ({type(exc).__name__})",
                        read_at=state.read_at)
    if status.returncode:
        if "not a git repository" in status.stderr.lower():
            return GitState(repo=False, read_at=state.read_at)
        return GitState(error=status.stderr.strip().splitlines()[0] if status.stderr.strip()
                        else f"git status exit code {status.returncode}", read_at=state.read_at)
    parse_status(status.stdout, state)
    _mark_vega_files(project, state)
    if state.oid:
        try:
            log = _git(project, "log", f"-{COMMITS}", "--no-color", "--format=%h%x1f%ct%x1f%s")
        except (OSError, subprocess.TimeoutExpired):
            log = None
        if log is not None and log.returncode == 0:
            for line in log.stdout.splitlines():
                parts = line.split("\x1f", 2)
                if len(parts) == 3 and parts[1].isdigit():
                    state.commits.append((parts[0], int(parts[1]), parts[2]))
    return state


def _mark_vega_files(project: Path, state: GitState) -> None:
    """Mark the files whose only change is Vega's own (project_context.written_by_vega): the
    VEGA.md it drafted and its .gitignore block are not the user's changes."""
    from . import project_context

    names = {Path(name).name for name in project_context.VEGA_OWNED_FILES}
    candidates = [f for f in state.files if not f.conflict and Path(f.path).name in names]
    if not candidates:
        return
    try:
        prefix = _git(project, "rev-parse", "--show-prefix").stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return
    for file in candidates:
        if file.path.startswith(prefix):
            file.vega = project_context.written_by_vega(project, file.path[len(prefix):])


def describe(file: GitFile) -> str:
    """`staged`, `unstaged`, `staged and unstaged`, `untracked`, `conflict` (preview titles)."""
    if file.conflict:
        return "conflict"
    if file.vega:
        return "written by Vega" + (" (untracked)" if file.untracked else "")
    if file.untracked:
        return "untracked"
    if file.staged and file.unstaged:
        return "staged and unstaged"
    return "staged" if file.staged else "unstaged"


def age(seconds: float) -> str:
    """Compact age for a commit: `now`, `5m`, `3h`, `2d`, `3w`, `4mo`, `2y`."""
    seconds = max(0, int(seconds))
    for limit, size, unit in ((60, 1, "now"), (3600, 60, "m"), (86400, 3600, "h"),
                              (14 * 86400, 86400, "d"), (60 * 86400, 7 * 86400, "w"),
                              (365 * 86400, 30 * 86400, "mo")):
        if seconds < limit:
            return "now" if unit == "now" else f"{seconds // size}{unit}"
    return f"{seconds // (365 * 86400)}y"


def _clip(text: str, width: int, ellipsis: str) -> str:
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    while text and cell_len(text) > width - 1:
        text = text[:-1]
    return text + ellipsis


def _clip_path(path: str, width: int, ellipsis: str) -> str:
    """Keep the end of a path (the file name), marking the cut with an ellipsis."""
    if cell_len(path) <= width:
        return path
    while path and cell_len(path) > width - 1:
        path = path[1:]
    return ellipsis + path


def branch_line(state: GitState, width: int, colours: dict[str, str], g: SimpleNamespace,
                ascii_only: bool) -> Text:
    text = Text()
    if state.detached:
        text.append("detached ", style=f"bold {colours['warning']}")
        text.append(f"at {state.oid[:7]}", style=colours["muted"])
        return text
    up, down = ("+", "-") if ascii_only else ("↑", "↓")
    marks = Text()
    if state.upstream:
        if state.ahead:
            marks.append(f" {up}{state.ahead}", style=f"bold {colours['accent']}")
        if state.behind:
            marks.append(f" {down}{state.behind}", style=f"bold {colours['warning']}")
        if not state.ahead and not state.behind:
            marks.append(" " + ("=" if ascii_only else "≡"), style=colours["muted"])
    branch = state.branch or "(no branch)"
    room = width - cell_len(marks.plain)
    text.append(_clip(branch, max(4, room), g.ellipsis), style=f"bold {colours['fg']}")
    text.append_text(marks)
    rest = width - cell_len(text.plain) - 2
    if not state.upstream:
        if rest >= 11:
            text.append("  no upstream", style=colours["muted"])
    elif rest >= 6:
        text.append("  " + _clip_path(state.upstream, rest, g.ellipsis), style=colours["muted"])
    if not state.oid:
        text.append("\n" + "no commits yet", style=colours["muted"])
    return text


def counts_line(state: GitState, width: int, colours: dict[str, str], g: SimpleNamespace) -> Text:
    counts = state.counts()
    text = Text()
    if len(state.files) == counts["vega"]:
        text.append(f"{g.done} clean", style=colours["success"])
        if counts["vega"]:
            fits = cell_len(text.plain) + 2 + len(f"{counts['vega']} by Vega") <= width
            text.append("  " if fits else "\n")
            text.append(str(counts["vega"]), style=f"bold {colours['muted']}")
            text.append(" by Vega", style=colours["muted"])
        return text
    styles = {"conflict": colours["error"], "staged": colours["success"],
              "unstaged": colours["warning"], "new": colours["fg"]}
    labels = {"conflict": "conflict", "staged": "staged", "unstaged": "unstaged", "new": "new"}
    for key in ("conflict", "staged", "unstaged", "new"):
        if not counts[key]:
            continue
        part = f"{counts[key]} {labels[key]}"
        gap = "  " if text else ""
        if cell_len(text.plain) + len(gap) + len(part) > width:
            labels_short = {"conflict": "!", "staged": "s", "unstaged": "u", "new": "n"}
            part = f"{counts[key]}{labels_short[key]}"
            gap = " " if text else ""
        text.append(gap)
        text.append(str(counts[key]), style=f"bold {styles[key]}")
        text.append(part[len(str(counts[key])):], style=colours["muted"])
    if counts["vega"]:  # Vega's own files: counted apart, never as the user's changes
        fits = cell_len(text.plain) + 2 + len(f"{counts['vega']} by Vega") <= width
        text.append("  " if fits else "\n")
        text.append(str(counts["vega"]), style=f"bold {colours['muted']}")
        text.append(" by Vega", style=colours["muted"])
    return text


def file_option(file: GitFile, width: int, colours: dict[str, str], g: SimpleNamespace) -> Text:
    """`MM src/calc.py`: git's own two letters, then the path (its end kept when clipped)."""
    text = Text()
    if file.conflict:
        text.append(file.xy, style=f"bold {colours['error']}")
    elif file.untracked:
        text.append("??", style=colours["muted"])
    else:
        index, tree = file.xy[0].replace(".", " "), file.xy[1].replace(".", " ")
        text.append(index, style=f"bold {colours['success']}")
        text.append(tree, style=f"bold {colours['warning']}")
    text.append(" ")
    name = file.path if not file.orig else f"{file.orig} > {file.path}"
    tag = "  by Vega" if file.vega and width >= 24 else ""
    text.append(_clip_path(name, max(4, width - 3 - len(tag)), g.ellipsis),
                style=colours["muted"] if file.untracked or file.vega else colours["fg"])
    if tag:
        text.append(tag, style=colours["muted"])
    return text


def commit_option(commit: tuple[str, int, str], now: float, width: int,
                  colours: dict[str, str], g: SimpleNamespace) -> Text:
    """`a1b2c3d  2h  Fix the add sign`: hash, age right-aligned in three cells, subject. Below
    HASH_FROM cells the hash is left out (the subject matters more; Enter shows the hash)."""
    sha, stamp, subject = commit
    text = Text()
    if width >= HASH_FROM:
        text.append(sha[:7] + " ", style=colours["muted"])
    text.append(age(now - stamp).rjust(3) + "  ", style=colours["muted"])
    text.append(_clip(subject, max(4, width - cell_len(text.plain)), g.ellipsis),
                style=colours["fg"])
    return text


def fill(options: OptionList, state: GitState | None, width: int, colours: dict[str, str],
         g: SimpleNamespace, ascii_only: bool, now: float | None = None) -> None:
    """Rebuild the pane, keeping the highlighted row."""
    keep = options.highlighted_option.id if options.highlighted_option else None
    options.clear_options()
    if state is None:
        options.add_option(Option(Text("reading git status" + g.ellipsis, style=colours["muted"]),
                                  disabled=True))
        return
    if not state.repo:
        message = state.error or NOT_A_REPO
        options.add_option(Option(Text(message, style=colours["muted"]), disabled=True))
        return
    if state.error:
        options.add_option(Option(Text(f"{g.warn} {state.error}", style=colours["warning"]),
                                  disabled=True))
        return
    now = time.time() if now is None else now
    options.add_option(Option(branch_line(state, width, colours, g, ascii_only), disabled=True))
    options.add_option(Option(counts_line(state, width, colours, g), disabled=True))
    order = sorted(range(len(state.files)), key=lambda i: (
        4 if state.files[i].vega else  # Vega's own files after the user's changes
        0 if state.files[i].conflict else 1 if state.files[i].staged else
        2 if state.files[i].unstaged else 3, state.files[i].path))
    for shown, number in enumerate(order):
        if shown == MAX_FILES:
            options.add_option(Option(Text(f"{len(order) - MAX_FILES} more files",
                                           style=colours["muted"]), disabled=True))
            break
        options.add_option(Option(file_option(state.files[number], width, colours, g),
                                  id=f"G{number}"))
    if state.commits:
        options.add_option(Option(Text("recent commits", style=colours["muted"]), disabled=True))
        for commit in state.commits:
            options.add_option(Option(commit_option(commit, now, width, colours, g),
                                      id=f"C{commit[0]}"))
    if keep:
        with contextlib.suppress(Exception):  # the row is gone
            options.highlighted = options.get_option_index(keep)


# -- previews (run in a worker; they return plain data, coloured on the UI thread) ---------------

def file_diff(project: Path, file: GitFile) -> tuple[str, str]:
    """(kind, body) for a file row: ("diff", unified diff) or ("new", file text) for an untracked
    file, or ("note", message)."""
    if file.untracked:
        path = project / file.path
        if path.is_dir():
            return "note", f"{file.path} is an untracked folder."
        try:
            data = path.read_bytes()[:PREVIEW_BYTES]
        except OSError as exc:
            return "note", f"Could not read {file.path}: {exc.strerror or exc}"
        if b"\0" in data[:8000]:
            return "note", f"{file.path} is a binary file."
        return "new", data.decode("utf-8", errors="replace")
    args = ["diff", "--no-color", "--no-ext-diff", "--no-textconv"]
    if file.staged and not file.unstaged:
        args.append("--cached")
    elif file.staged and file.unstaged:
        args.append("HEAD")
    try:
        result = _git(project, *args, "--", file.path)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "note", f"git diff did not finish ({type(exc).__name__})."
    if result.returncode:
        return "note", result.stderr.strip() or f"git diff exit code {result.returncode}"
    return ("diff", result.stdout[:PREVIEW_BYTES]) if result.stdout.strip() else \
        ("note", "No textual change (a mode change, or the file is binary).")


def commit_show(project: Path, sha: str) -> tuple[str, str]:
    try:
        result = _git(project, "show", "--no-color", "--stat", "--patch",
                      "--format=%H%n%an  %ad%n%n%B", "--date=format:%d %b %Y %H:%M", sha)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "note", f"git show did not finish ({type(exc).__name__})."
    if result.returncode:
        return "note", result.stderr.strip() or f"git show exit code {result.returncode}"
    return "diff", result.stdout[:PREVIEW_BYTES]


__all__ = [
    "NOT_A_REPO",
    "GitFile",
    "GitState",
    "age",
    "commit_show",
    "file_diff",
    "fill",
    "parse_status",
    "read_state",
]
