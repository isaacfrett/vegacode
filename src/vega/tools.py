from __future__ import annotations

import difflib
import json
import os
import re
import shlex
import subprocess
import threading
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

from . import data_preview, features, sandbox
from .diagnostics import after_write

console = Console()
IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache", ".ruff_cache",
    ".pytest_cache", ".cache", ".tox", ".gradle", ".godot", ".next", "Pods", "DerivedData",
    # Vega's own per-project state (notes.md from `remember`, session digests, state.json): never
    # project content. Without this, search_text returned the agent's own note as the first hit and
    # it edited the note instead of the code (selfbuild, 2026-10-21). read_file of an explicit
    # .vega/ path still works; only search and listing skip it.
    ".vega",
}
# Skipped only at the top of the home folder (Vega started in ~): ~/Library is huge. Anywhere else
# a Library folder is ordinary content (a project's, or a snapshot's Users/<u>/Library to inspect).
HOME_ONLY_DIRS = {"Library", ".Trash"}


def ignored_parts(parts: tuple[str, ...] | list[str], project: Path) -> bool:
    """Whether a path inside `project`, given by its parts relative to `project`, is skipped."""
    if any(part in IGNORED_DIRS for part in parts):
        return True
    return bool(parts) and parts[0] in HOME_ONLY_DIRS and _is_home(project)


def _is_home(project: Path) -> bool:
    try:
        return project.resolve() == Path.home().resolve()
    except OSError:
        return False
WALK_LIMIT = 5000  # files any non-git walk will visit; keeps tools fast in huge folders
OVERVIEW_WALK_LIMIT = 2000
FULL_LISTING_MAX = 40  # projects up to this size get the full file list in the snapshot
PROJECT_MARKERS = (
    ".git", "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "package.json",
    "Cargo.toml", "go.mod", "Package.swift", "project.godot", "Makefile", "pom.xml",
    "build.gradle", "Gemfile", "composer.json", "CMakeLists.txt",
)


def _project_path(project: Path, raw: str) -> Path:
    target = (project / raw).resolve()
    try:
        target.relative_to(project)
    except ValueError as exc:
        raise ValueError("Path must stay inside the project directory.") from exc
    return target


def _project_files(project: Path, limit: int = WALK_LIMIT):
    """Files under the project, never entering ignored folders, stopping after `limit` files."""
    count = 0
    at_home = _is_home(project)
    for root, dirs, files in os.walk(project):
        top = at_home and Path(root) == project
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS and not (top and d in HOME_ONLY_DIRS))
        for name in sorted(files):
            yield Path(root) / name
            count += 1
            if count >= limit:
                return


def is_project(folder: Path) -> bool:
    """Whether the folder looks like a project (Git or a common project file)."""
    if folder.resolve() == Path.home().resolve():
        return False
    if any((folder / marker).exists() for marker in PROJECT_MARKERS):
        return True
    return any(child.suffix in {".xcodeproj", ".xcworkspace", ".sln"} for child in folder.iterdir())


def project_overview(project: Path, limit: int = 150) -> str:
    """File list and Git status, injected at session start so the model sees the project upfront."""
    tracked = subprocess.run(
        ["git", "-C", str(project), "ls-files", "--cached", "--others", "--exclude-standard"],
        text=True, capture_output=True, timeout=15,
    )
    capped = False
    if tracked.returncode == 0:  # respects .gitignore, so build output and caches stay out
        files = sorted(set(tracked.stdout.splitlines()))
    else:
        walked = [p.relative_to(project).as_posix()
                  for p in _project_files(project, limit=OVERVIEW_WALK_LIMIT)]
        capped = len(walked) >= OVERVIEW_WALK_LIMIT
        files = sorted(walked)
    files = [f for f in files if not ignored_parts(f.split("/"), project)]  # e.g. untracked .vega/
    total = f"{len(files)}+" if capped else str(len(files))
    if len(files) <= FULL_LISTING_MAX:
        listing = "\n".join(files) or "(no files)"
    else:
        listing = _scoped_listing(files, limit)
    status = subprocess.run(
        ["git", "-C", str(project), "status", "--short", "--branch"], text=True, capture_output=True, timeout=15,
    )
    if status.returncode == 0:
        lines = status.stdout.strip().splitlines()
        git = "\n".join(lines[:20]) + (f"\n... and {len(lines) - 20} more" if len(lines) > 20 else "")
    else:
        git = "not a Git repository"
    # Only worth saying where it matters: the home folder, or a walk that hit its cap. Small
    # folders without project files (e.g. eval fixtures) keep their plain listing.
    if (capped or project.resolve() == Path.home().resolve()) and not is_project(project):
        listing = ("(This folder does not look like a project: no Git repository or project file. "
                   "Only a partial view is shown.)\n" + listing)
    return f"Files ({total}):\n{listing}\n\nGit status:\n{git or 'clean'}"


def _scoped_listing(files: list[str], limit: int) -> str:
    """A compact map for larger projects: top-level folders with file counts and the root files.
    The full tree is one repo_map/list_files call away; listing it here would crowd the context."""
    folders: dict[str, int] = {}
    root_files = []
    for path in files:
        head, _, rest = path.partition("/")
        if rest:
            folders[head] = folders.get(head, 0) + 1
        else:
            root_files.append(path)
    lines = [f"{name}/ ({count} files)" for name, count in
             sorted(folders.items(), key=lambda item: -item[1])[:20]]
    if len(folders) > 20:
        lines.append(f"... and {len(folders) - 20} more folders")
    lines += root_files[:20]
    if len(root_files) > 20:
        lines.append(f"... and {len(root_files) - 20} more files at the top level")
    lines.append("Use repo_map or list_files to see inside folders.")
    return "\n".join(lines[:limit])


def tool_specs(
    include_delegate: bool = True, include_remember: bool = True,
    include_plan: bool = True, include_index: bool = True, include_knowledge: bool = False,
    include_definition: bool = False,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List project files. Omit path to list the project root. Use this without asking the user what to inspect.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_files",
                "description": "Find project files whose relative path contains a name or phrase, such as cli.py.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_text",
                "description": "Search text in project files for a phrase or code identifier.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "repo_map",
                "description": "Show a compact map of the project: each file with its classes, functions and methods and their line numbers. Optionally limit to a path prefix such as src/.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Ranked keyword search over code. Describe what you are looking for in words or identifiers (e.g. 'start ollama server'); returns the most relevant functions and code blocks with line ranges.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 project file. Output is prefixed with line numbers (number, then a tab); they are not part of the file. Optionally pass start_line and/or end_line (1-based, inclusive) to read part of a large file. If the supplied path is not rooted correctly, Vega searches for a unique matching filename automatically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": EDIT_BY_NAME_EDIT_FILE if include_definition else "Replace one unique text block in an existing UTF-8 file inside the project. Read the file first. Copy old_text from the file without the line-number prefixes. Small whitespace or indentation differences are tolerated; the edit fails if the block is absent or ambiguous. Prefer this for focused changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "replace_lines",
                "description": "Replace lines start_line..end_line (inclusive, 1-based, as numbered by read_file) of an existing file with new_text. Use when edit_file cannot match the text. Re-read the file after editing because line numbers shift.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                        "new_text": {"type": "string"},
                    },
                    "required": ["path", "start_line", "end_line", "new_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or replace a UTF-8 file inside the project. Supply the complete contents.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_status",
                "description": "Inspect the project's current Git branch and working-tree changes. Use before editing and before reporting completion to preserve and review user work.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_diff",
                "description": "Inspect tracked changes compared with HEAD in the project. Optionally provide a project-relative path to limit the diff. Use to review your work before finishing.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_check",
                "description": (
                    "Run a test or check command in an isolated copy of the project with no network "
                    "access. It cannot change project files and runs without approval. Allowed: "
                    + "; ".join(sandbox.RUN_CHECK_PROGRAMS)
                    + ". Use it to verify changes, e.g. `python -m pytest -q`, `ruff check .`, or "
                    "`python -c \"from mod import f; assert f(1) == 2\"`."
                ),
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command in the project. Read-only inspection commands run automatically; other commands require user approval.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
            },
        },
    ]
    tools.append({
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": "Record or update your step-by-step plan for a multi-step task. Pass the whole list every time. Each item has text and a status: pending, in_progress or done. Exactly one item is in_progress while work remains. The current plan is shown to you on every step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "status": {"type": "string", "enum": list(PLAN_STATUSES)},
                            },
                            "required": ["text", "status"],
                        },
                    },
                },
                "required": ["items"],
            },
        },
    })
    if include_remember:
        tools.append({
            "type": "function",
            "function": {
                "name": "remember",
                "description": "Save one durable fact about this project (how to run tests, a convention, a gotcha) to .vega/notes.md so future sessions see it. Not for task progress.",
                "parameters": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]},
            },
        })
    if include_delegate:
        tools.append({
            "type": "function",
            "function": {
                "name": "delegate_task",
                "description": "Delegate focused read-only investigation to a separate bounded model session.",
                "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
            },
        })
    if include_definition:
        tools.append(REPLACE_DEFINITION_SPEC)
    if include_knowledge:
        tools.append({
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "Search Vega's knowledge library: books, documentation and pages the user asked Vega to learn. Returns the best-matching sections with their source and location (page or section) to cite.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            },
        })
    dropped = set() if include_plan else {"update_plan"}
    if not include_index:
        dropped |= {"repo_map", "search_code"}
    return [tool for tool in tools if tool["function"]["name"] not in dropped]


# edit_by_name: the recommended edit for Python code
EDIT_BY_NAME_EDIT_FILE = (
    "Replace one unique text block in an existing UTF-8 file inside the project. Read the file "
    "first. Copy old_text from the file without the line-number prefixes. Use it for small edits "
    "(a line or two) and for non-Python files; to change a Python function, class or method, use "
    "replace_definition instead. The file is saved only if it still parses.")
REPLACE_DEFINITION_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "replace_definition",
        "description": (
            "The recommended way to change Python code: replace one function, class or method by "
            "name, or add it if it does not exist (a method goes at the end of its class). No "
            "old_text needed. path: the .py file (or a dotted module). name: `function`, `Class` "
            "or `Class.method`. code: the complete new definition (signature, docstring, body) as "
            "plain code, without line numbers; indentation is adjusted to the file. The original "
            "decorators and docstring are kept when your code has none. New imports and helper "
            "functions in code are added too. The file is saved only if it still parses."),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "name": {"type": "string"},
                "code": {"type": "string"},
            },
            "required": ["path", "name", "code"],
        },
    },
}

READ_FILE_MAX_CHARS = 30000


def _line_number(args: dict[str, Any], key: str) -> int | None:
    value = args.get(key)
    if value is None or value == "":
        return None
    return int(value)  # a ValueError becomes a tool error naming the bad value


def _read_file(
    project: Path,
    raw_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    max_chars: int = READ_FILE_MAX_CHARS,
    preview: bool = False,
) -> str:
    """Numbered lines of a file, optionally only start_line..end_line (1-based, inclusive).

    Output is cut at a whole line once it passes max_chars, with a note saying how to read on.
    With `preview` (data_preview), a whole-file read of a large data file returns a structured
    preview instead; a read with a line range always returns exactly those lines.
    """
    target = _project_path(project, raw_path)
    resolution_note = ""
    if not target.is_file():
        matches = [p for p in _project_files(project) if p.name.casefold() == Path(raw_path).name.casefold()]
        if len(matches) == 1:
            target = matches[0]
            resolution_note = f"Resolved {raw_path!r} to {target.relative_to(project).as_posix()}.\n\n"
        elif len(matches) > 1:
            choices = "\n".join(p.relative_to(project).as_posix() for p in matches[:20])
            return f"Found multiple files named {Path(raw_path).name!r}; choose one of these:\n{choices}"
        else:
            return f"File not found: {raw_path}. Use find_files or list_files to locate it."
    ranged = start_line is not None or end_line is not None
    if preview and not ranged:
        shown = data_preview.preview(target, target.relative_to(project).as_posix()
                                     if target.is_relative_to(project) else raw_path)
        if shown:
            return resolution_note + shown
    lines = target.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    first = 1 if start_line is None else start_line
    last = total if end_line is None else min(end_line, total)
    bad_end = end_line is not None and end_line < first
    if ranged and total and (first < 1 or first > total or bad_end):
        shown_end = "end" if end_line is None else end_line
        return f"Invalid line range {first}-{shown_end}; the file has {total} lines."
    numbered: list[str] = []
    used = 0
    for number in range(first, last + 1):
        entry = f"{number:>4}\t{lines[number - 1]}"
        if numbered and used + len(entry) + 1 > max_chars:
            break
        numbered.append(entry[:max_chars])
        used += len(entry) + 1
    body = "\n".join(numbered)
    shown_last = first + len(numbered) - 1
    if numbered and shown_last < last:
        body += (
            f"\n[truncated: {last - shown_last} more lines; use read_file with start_line/end_line"
            f" (showing {first}-{shown_last} of {total})]"
        )
    elif ranged and numbered:
        body += f"\n[lines {first}-{shown_last} of {total}]"
    return resolution_note + (body or "(empty file)")


_LINE_PREFIX = re.compile(r"^\s*\d+\t")


def _strip_line_numbers(text: str) -> str:
    """Remove read_file line-number prefixes if the model copied them into an edit."""
    lines = text.split("\n")
    content = [line for line in lines if line.strip()]
    if content and all(_LINE_PREFIX.match(line) for line in content):
        return "\n".join(_LINE_PREFIX.sub("", line, count=1) for line in lines)
    return text


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _find_line_block(file_lines: list[str], old_lines: list[str], normalise) -> list[int]:
    """Start indexes where old_lines matches file_lines under a per-line normalisation."""
    wanted = [normalise(line) for line in old_lines]
    size = len(wanted)
    return [
        start for start in range(len(file_lines) - size + 1)
        if [normalise(line) for line in file_lines[start:start + size]] == wanted
    ]


def _reindent(new_lines: list[str], old_first: str, file_first: str) -> list[str]:
    """Shift new_lines by the indentation difference between the model's old_text and the file."""
    old_indent, file_indent = _indent(old_first), _indent(file_first)
    if old_indent == file_indent:
        return new_lines
    result = []
    for line in new_lines:
        if not line.strip():
            result.append(line)
        elif line.startswith(old_indent):
            result.append(file_indent + line[len(old_indent):])
        else:
            result.append(file_indent + line.lstrip())
    return result


def _apply_edit(original: str, old_text: str, new_text: str) -> tuple[str | None, str]:
    """Return (updated text or None, note). Tries exact, then whitespace-tolerant line matching."""
    occurrences = original.count(old_text)
    if occurrences == 1:
        return original.replace(old_text, new_text, 1), "exact match"
    if occurrences > 1:
        return None, f"old_text matched {occurrences} times; include more surrounding lines to make it unique."
    file_lines = original.split("\n")
    old_lines = old_text.strip("\n").split("\n")
    new_lines = new_text.strip("\n").split("\n") if new_text.strip("\n") else []
    for label, normalise in (
        ("ignoring trailing whitespace", str.rstrip),
        ("ignoring indentation", str.strip),
        # The 7B often swaps ' and " (seen 2026-10-13: an edit retried 14 times on 'A' vs "A").
        ("ignoring indentation and quote style", lambda line: line.strip().replace('"', "'")),
    ):
        starts = _find_line_block(file_lines, old_lines, normalise)
        if len(starts) > 1:
            lines = ", ".join(str(start + 1) for start in starts[:10])
            return None, f"old_text matched {len(starts)} places ({label}) at lines {lines}; include more context."
        if len(starts) == 1:
            start = starts[0]
            replacement = _reindent(new_lines, old_lines[0], file_lines[start])
            updated = file_lines[:start] + replacement + file_lines[start + len(old_lines):]
            return "\n".join(updated), f"matched {label}"
    first = old_lines[0].strip()
    close = difflib.get_close_matches(first, [line.strip() for line in file_lines], n=1, cutoff=0.5)
    hint = ""
    if close:
        line_number = next(i for i, line in enumerate(file_lines, 1) if line.strip() == close[0])
        hint = f" The closest line is {line_number}: {close[0]!r}. Re-read the file and copy the block exactly."
    return None, "old_text was not found in the file." + hint


def _repo_map(project: Path, raw_path: str | None) -> str:
    from vega.index import build_index  # local import: index imports this module

    prefix = ""
    if raw_path and str(raw_path).strip() not in {"", "."}:
        target = _project_path(project.resolve(), str(raw_path))
        prefix = target.relative_to(project.resolve()).as_posix()
        if target.is_dir():
            prefix += "/"
    return build_index(project).repo_map(prefix=prefix)


_DEFINITION_QUERY = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")


def _missing_symbol_note(project: Path, query: str) -> str:
    """A search for `def name(...)` / `class Name` that no file defines: say so first. On
    pyprojects the 7B searched ~90 times for the function it was asked to write (2026-10-14)."""
    if not features.on("missing_symbol_hint"):
        return ""
    match = _DEFINITION_QUERY.match(query)
    if not match:
        return ""
    name = match.group(1)
    pattern = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b", re.MULTILINE)
    for path in _project_files(project):
        if path.suffix != ".py":
            continue
        try:
            if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                return ""
        except OSError:
            continue
    return (f"Note: nothing in this project defines {name} yet. If the task asks you to add it, "
            "stop searching and write it (edit_file or write_file).\n\n")


def _search_code(project: Path, query: str, limit: int = 10) -> str:
    from vega.index import build_index, format_hits

    if not query.strip():
        return "Provide words or identifiers describing the code to find."
    hits = build_index(project).search(query, limit=limit)
    return format_hits(hits) or f"No code matched {query!r}. Try search_text for an exact phrase."


# Options that make an allowlisted command write a file or run another program. Auto-run
# commands go through a shell, so these would otherwise run without the user's approval.
_UNSAFE_OPTIONS = {
    "git": {"--output", "--ext-diff"},
    "rg": {"--pre", "--pre-glob"},
}


def _is_auto_read_only(command: str, project: Path) -> bool:
    """Auto-run only simple, allowlisted inspection commands scoped to the project."""
    # `~` and `{a,b}` are expanded by the shell after this check, so `cat ~/.ssh/id_rsa` would
    # pass a path check done on the literal text.
    if any(char in command for char in (";", "&", "|", "<", ">", "`", "$", "\\", "\n", "~", "{")):
        return False
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    if not parts:
        return False
    name = Path(parts[0]).name
    args = parts[1:]
    if name in {"pwd"}:
        return True
    if name in {"cat", "head", "tail", "rg", "grep", "wc", "file", "stat"}:
        if not args:
            return False
    elif name == "ls":
        pass
    elif name == "git":
        # The subcommand comes first: global options (-C, -c, --git-dir, --work-tree, ...) can
        # point git at another repository or set a pager or hook to run.
        if not args or args[0] not in {"status", "diff", "log", "show"}:
            return False
    else:
        return False
    # Avoid shell commands that inspect paths outside the selected project.
    for arg in args:
        if arg.startswith("-"):
            flag, _, value = arg.partition("=")
            if flag in _UNSAFE_OPTIONS.get(name, ()):
                return False
            # a path given inside an option (`--file=/etc/x`, `-f/etc/x`) is checked like any other
            if not value and not arg.startswith("--"):
                value = arg[2:]
            if "/" in value and (Path(value).is_absolute() or ".." in Path(value).parts):
                return False
            continue
        if "/" not in arg and arg not in {".", ".."}:
            continue
        if Path(arg).is_absolute() or ".." in Path(arg).parts:
            return False
        try:
            _project_path(project, arg)
        except ValueError:
            return False
    return True


WRITE_TOOLS = {"edit_file", "replace_lines", "write_file", "replace_definition"}


@dataclass
class WritePlan:
    """A write tool's effect computed without touching disk, so it can be previewed or declined."""

    tool: str
    target: Path
    relative: str
    before: str | None  # None when the file does not exist yet
    after: str
    message: str  # result text on success, without the syntax note


def plan_write(name: str, args: dict[str, Any], project: Path, *,
               atomic: bool = True) -> WritePlan | str:
    """Compute edit_file / replace_lines / write_file / replace_definition
    without writing; a str is the tool's error (the file is unchanged).

    - fence_strip: a code file's text wrapped in ``` (or starting with a line naming the file) is
      cleaned first, with a note in the result.
    - atomic_writes (with `atomic`, which Vega's own pipeline writes turn off): the whole new file
      is checked and the write refused when it does not parse or adds undefined names or
      redefinitions (vega/edits.py)."""
    from . import edits

    notes: list[str] = []
    if features.on("fence_strip") and name in _FENCED_ARGS and "path" in args:
        args = dict(args)
        for key in _FENCED_ARGS[name]:
            if isinstance(args.get(key), str):
                args[key], found = edits.strip_wrapping(args[key], str(args["path"]))
                notes += [note for note in found if note not in notes]
    plan = _plan_write(name, args, project)
    fence_note = f" Note: Vega {'; '.join(notes)} before checking it." if notes else ""
    if isinstance(plan, str):
        return plan + fence_note if plan.startswith(("Edit not applied", "Write not applied")) \
            else plan
    if atomic and features.on("atomic_writes") and plan.tool in _TEXT_ARG:
        verdict = edits.check(plan.tool, plan.relative, plan.before, plan.after, project, args)
        if verdict.refusal is not None:
            return verdict.refusal + fence_note
        if verdict.note:
            plan = replace(plan, message=plan.message + verdict.note)
    if fence_note:
        plan = replace(plan, message=plan.message + fence_note)
    return plan


_TEXT_ARG = {"edit_file": "new_text", "replace_lines": "new_text", "write_file": "content",
             "replace_definition": "code"}
# fence_strip: the arguments whose text is cleaned (replace_definition cleans its own code)
_FENCED_ARGS = {"edit_file": ("old_text", "new_text"), "replace_lines": ("new_text",),
                "write_file": ("content",)}
def _top_level_names(source: str) -> dict[str, tuple[int, int]]:
    """Top-level def/class names -> (first line incl. decorators, last line); {} if unparsable."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    names = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            names[node.name] = (first, node.end_lineno or node.lineno)
    return names


def _duplicate_definitions(target: Path, original: str, new_text: str) -> str:
    """Refuse an append that would define an existing top-level function or class a second time
    (the 7B appended the same function ~20 times, 2026-10-13 learn runs); '' when fine."""
    if target.suffix != ".py":
        return ""
    existing = _top_level_names(original)
    repeated = [name for name in _top_level_names(new_text) if name in existing]
    if not repeated:
        return ""
    lines = original.split("\n")
    shown = []
    for name in repeated:
        first, last = existing[name]
        body = "\n".join(f"{n}\t{lines[n - 1]}" for n in range(first, last + 1))
        shown.append(f"{name} (lines {first}-{last}):\n{body}")
    return (
        f"Edit not applied: {target.name} already defines {', '.join(repeated)}, so appending would "
        "add a second copy. Change the existing definition instead: use it as old_text in "
        "edit_file (without the line numbers), or replace_lines with its line range.\n"
        + "\n".join(shown)
    )


SHOW_ON_MISS_LINES = 120


def _current_lines(text: str) -> str:
    """The file as numbered lines (the first SHOW_ON_MISS_LINES), appended to a failed edit so the
    model copies real text instead of guessing again (it guessed 13 times, 2026-10-13)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    shown = lines[:SHOW_ON_MISS_LINES]
    width = len(str(len(shown)))
    body = "\n".join(f"{n:>{width}}\t{line}" for n, line in enumerate(shown, 1))
    more = f"\n[{len(lines) - len(shown)} more lines; read_file with start_line to see them]" \
        if len(lines) > len(shown) else ""
    return (f"\nThe file currently reads (copy old_text from here, without the line numbers):"
            f"\n{body}{more}")


_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
REFERENCE_LIMIT = 10


def references_note(project: Path, edited: Path, old_text: str, new_text: str) -> str:
    """Where names that the edit replaced still appear, e.g. a rename that missed a caller."""
    import keyword

    removed = {name for name in set(_IDENTIFIER.findall(old_text)) - set(_IDENTIFIER.findall(new_text))
               if not keyword.iskeyword(name)}
    if not removed or not set(_IDENTIFIER.findall(new_text)):
        return ""
    patterns = {name: re.compile(rf"\b{re.escape(name)}\b") for name in sorted(removed)}
    found: dict[str, list[str]] = {name: [] for name in patterns}
    for path in _project_files(project):
        if path.suffix.lower() not in {".py", ".pyi", ".js", ".ts", ".md", ".toml", ".cfg", ".ini",
                                       ".json", ".yaml", ".yml", ".txt", ".rst", ".sh"}:
            continue
        try:
            lines = path.read_text(encoding="utf-8").split("\n")
        except (OSError, UnicodeDecodeError):
            continue
        relative = path.relative_to(project).as_posix()
        for number, line in enumerate(lines, 1):
            for name, pattern in patterns.items():
                if len(found[name]) < REFERENCE_LIMIT and pattern.search(line):
                    found[name].append(f"{relative}:{number}")
    parts = [f"{name} still appears at {', '.join(places)}" for name, places in found.items()
             if places]
    if not parts:
        return ""
    return ("\nNote: " + "; ".join(parts) + ". Update those too if the change should apply "
            "there.")


def _module_path(project: Path, raw: str) -> str:
    """replace_definition also takes a dotted module (`pkg.mod`) for its path."""
    if raw.endswith((".py", ".pyi")) or "/" in raw or (project / raw).is_file():
        return raw
    base = raw.replace(".", "/")
    for candidate in (f"{base}.py", f"src/{base}.py", f"{base}/__init__.py",
                      f"src/{base}/__init__.py"):
        if (project / candidate).is_file():
            return candidate
    return f"{base}.py"


def _plan_write(name: str, args: dict[str, Any], project: Path) -> WritePlan | str:
    if name == "replace_definition":
        from .edits import plan_definition

        raw = str(args.get("path") or args.get("module_path") or "").strip()
        if not raw:
            return "Give path (the file, or a dotted module), name and code."
        target = _project_path(project, _module_path(project, raw))
        relative = target.relative_to(project).as_posix()
        return plan_definition(target, relative, str(args.get("name", "")),
                               str(args.get("code", "")))
    target = _project_path(project, args["path"])
    relative = target.relative_to(project).as_posix()
    if name == "edit_file":
        if not target.is_file():
            return f"File not found: {args['path']}. Use find_files or list_files to locate it."
        old_text = _strip_line_numbers(args["old_text"])
        new_text = _strip_line_numbers(args["new_text"])
        original = target.read_text(encoding="utf-8")
        if not old_text.strip():
            # An empty old_text means "add this to the file" (the model's natural move when adding a
            # new function; rejecting it looped 14 times on 2026-10-13): append at the end.
            if not new_text.strip():
                return "Nothing to add: new_text is empty."
            duplicate = _duplicate_definitions(target, original, new_text) \
                if features.on("append_guard") else ""
            if duplicate:
                return duplicate
            separator = "" if not original.strip() else ("\n" if original.endswith("\n") else "\n\n")
            updated, note = original + separator + new_text.lstrip("\n"), "appended at the end"
            if not updated.endswith("\n"):
                updated += "\n"
        elif old_text == new_text and features.on("noop_edit_refused"):
            # Granite "fixed" a selfbuild bug with old_text == new_text and was told
            # "Edited (exact match)", then reported the bug fixed (2026-10-21).
            return ("Edit not applied: new_text is identical to old_text, so nothing would "
                    "change. Write the corrected code in new_text.")
        else:
            updated, note = _apply_edit(original, old_text, new_text)
        if updated is None:
            if features.on("show_on_miss") and "not found" in note:
                note += _current_lines(original)
            return f"Edit not applied: {note}"
        return WritePlan(name, target, relative, original, updated, f"Edited {relative} ({note}).")
    if name == "replace_lines":
        if not target.is_file():
            return f"File not found: {args['path']}. Use find_files or list_files to locate it."
        original = target.read_text(encoding="utf-8")
        lines = original.split("\n")
        start, end = int(args["start_line"]), int(args["end_line"])
        if not 1 <= start <= end <= len(lines):
            return f"Invalid line range {start}-{end}; the file has {len(lines)} lines."
        new_text = _strip_line_numbers(args["new_text"])
        replacement = new_text.split("\n") if new_text else []
        updated = "\n".join(lines[: start - 1] + replacement + lines[end:])
        message = f"Replaced lines {start}-{end} of {relative}."
        return WritePlan(name, target, relative, original, updated, message)
    if name == "write_file":
        content = _strip_line_numbers(args["content"])
        before = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else None
        message = f"Wrote {relative} ({len(content)} characters)."
        return WritePlan(name, target, relative, before, content, message)
    return f"Unknown tool: {name}"


NOTES_PATH = ".vega/notes.md"
NOTE_MAX_CHARS = 300


def plan_note(args: dict[str, Any], project: Path, today: str | None = None) -> WritePlan | str:
    """`remember` as a write: append one dated bullet to .vega/notes.md, creating it if needed.

    Going through WritePlan means review mode asks first and /undo can revert it.
    """
    note = re.sub(r"\s+", " ", str(args.get("note", ""))).strip()
    if not note:
        return "Note not saved: provide one short fact about the project in `note`."
    if len(note) > NOTE_MAX_CHARS:
        return f"Note not saved: keep it under {NOTE_MAX_CHARS} characters (one durable fact)."
    target = _project_path(project, NOTES_PATH)  # resolves symlinks, so .vega cannot lead outside
    before = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else None
    text = before if before is not None else "# Vega project notes\n\n"
    if not text.endswith("\n"):
        text += "\n"
    after = text + f"- {today or date.today().isoformat()}: {note}\n"
    message = f"Saved to {NOTES_PATH}: {note}"
    return WritePlan("remember", target, NOTES_PATH, before, after, message)


def load_notes(project: Path, max_chars: int = 2000) -> str | None:
    """.vega/notes.md for the system message, capped at max_chars; None when absent or unsafe."""
    try:
        target = _project_path(project, NOTES_PATH)
        if not target.is_file():
            return None
        text = target.read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, ValueError):
        return None
    if not text:
        return None
    if len(text) > max_chars:
        omitted = len(text) - max_chars
        more = f"\n[... {omitted} more characters; read_file {NOTES_PATH} if needed]"
        text = text[:max_chars] + more
    return text


PLAN_STATUSES = ("pending", "in_progress", "done")
PLAN_MAX_ITEMS = 20
PLAN_ITEM_MAX_CHARS = 200
_STATUS_ALIASES = {
    "todo": "pending", "not_started": "pending", "in-progress": "in_progress",
    "doing": "in_progress", "active": "in_progress", "current": "in_progress",
    "completed": "done", "complete": "done", "finished": "done",
}
_PLAN_MARKS = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]"}
_PLAN_SHAPE = 'Invalid plan: `items` must be a list of {"text": ..., "status": ...} objects.'


def validate_plan(raw: Any) -> list[dict[str, str]] | str:
    """Normalised plan items, or an error message saying what to fix."""
    items, _ = validate_plan_started(raw)
    return items


def validate_plan_started(raw: Any) -> tuple[list[dict[str, str]] | str, int | None]:
    """validate_plan, plus the 1-based number of the item it started: with `plan_pending_ok`, a
    plan whose open items are all pending is accepted and its first pending item set in_progress
    (Granite baseline 2026-10-15: 21 rejections of an all-pending first plan in the datasci logs,
    each costing a step). Two or more in_progress items are still rejected."""
    if isinstance(raw, str):  # small models often send the list as a JSON string
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return _PLAN_SHAPE, None
    if isinstance(raw, dict) and "items" in raw:
        raw = raw["items"]
    if not isinstance(raw, list):
        return _PLAN_SHAPE, None
    if len(raw) > PLAN_MAX_ITEMS:
        return f"Invalid plan: at most {PLAN_MAX_ITEMS} items; merge small steps.", None
    items: list[dict[str, str]] = []
    for number, entry in enumerate(raw, 1):
        if not isinstance(entry, dict):
            return f"Invalid plan: item {number} must be an object with text and status.", None
        text = re.sub(r"\s+", " ", str(entry.get("text", ""))).strip()
        if not text:
            return f"Invalid plan: item {number} has no text.", None
        status = str(entry.get("status", "")).strip().lower().replace(" ", "_")
        status = _STATUS_ALIASES.get(status, status)
        if status not in PLAN_STATUSES:
            return (
                f"Invalid plan: item {number} has status {entry.get('status')!r}; "
                "use pending, in_progress or done."
            ), None
        items.append({"text": text[:PLAN_ITEM_MAX_CHARS], "status": status})
    active = sum(item["status"] == "in_progress" for item in items)
    unfinished = any(item["status"] != "done" for item in items)
    if active > 1:
        return (
            f"Invalid plan: {active} items are in_progress; exactly one may be in_progress at a "
            "time. Mark the others pending or done."
        ), None
    if unfinished and not active:
        if features.on("plan_pending_ok"):
            started = next(index for index, item in enumerate(items) if item["status"] == "pending")
            items[started]["status"] = "in_progress"
            return items, started + 1
        return (
            "Invalid plan: no item is in_progress. Mark the item you are working on now as "
            "in_progress (exactly one while work remains)."
        ), None
    return items, None


def render_plan(items: list[dict[str, str]]) -> str:
    return "\n".join(f"{_PLAN_MARKS[item['status']]} {item['text']}" for item in items)


def apply_write(plan: WritePlan, project: Path, args: dict[str, Any] | None = None) -> str:
    plan.target.parent.mkdir(parents=True, exist_ok=True)
    plan.target.write_text(plan.after, encoding="utf-8")
    note = after_write(project, plan.target, plan.after)
    if args and plan.tool == "edit_file" and features.on("references_hint"):
        note += references_note(project, plan.target, str(args.get("old_text", "")),
                                str(args.get("new_text", "")))
    return plan.message + note


def execute(
    name: str,
    args: dict[str, Any],
    project: Path,
    *,
    can_write: bool = True,
    before_write: Callable[[WritePlan], str | None] | None = None,
    max_result_chars: int | None = None,
    approve_command: Callable[[str], bool] | None = None,
    stop: threading.Event | None = None,
    atomic: bool = True,
) -> str:
    """Run one tool. before_write sees each planned write and may return a refusal instead.

    atomic=False skips the atomic_writes check: Vega's pipelines write merged code through their
    own check-and-repair loops (and restore earlier versions) with it off.

    max_result_chars lets read_file stop at a whole line with an accurate "read on" note.
    approve_command(command) decides whether a run_command that is not read-only may run; None
    keeps the console prompt (line mode). With a callback nothing is printed to the console.
    """
    try:
        if name == "list_files":
            raw_path = args.get("path", ".")
            folder = _project_path(project, raw_path)
            if not folder.is_dir():
                return f"Not a directory: {raw_path}"
            entries = [p.relative_to(project).as_posix() for p in folder.rglob("*")
                       if not ignored_parts(p.relative_to(project).parts, project)]
            return "\n".join(entries[:500]) or "(empty directory)"
        if name == "find_files":
            query = str(args.get("query", "")).casefold().strip()
            if not query:
                return "Provide a filename or path fragment to search for."
            results = [p.relative_to(project).as_posix() for p in _project_files(project) if query in p.relative_to(project).as_posix().casefold()]
            return "\n".join(results[:100]) or f"No files matched {query!r}."
        if name == "search_text":
            query = str(args.get("query", ""))
            if not query:
                return "Provide text or an identifier to search for."
            hits: list[str] = []
            for path in _project_files(project):
                try:
                    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                        if query.casefold() in line.casefold():
                            hits.append(f"{path.relative_to(project).as_posix()}:{line_number}: {line.strip()[:300]}")
                            if len(hits) >= 100:
                                return "\n".join(hits)
                except OSError:
                    continue
            return "\n".join(hits) or f"No text matches for {query!r}."
        if name == "search_knowledge":
            from vega import knowledge

            return knowledge.search_text(str(args.get("query", "")))
        if name == "repo_map":
            return _repo_map(project, args.get("path"))
        if name == "search_code":
            query = str(args.get("query", ""))
            return _missing_symbol_note(project, query) + _search_code(project, query)
        if name == "read_file":
            return _read_file(
                project, args["path"], _line_number(args, "start_line"),
                _line_number(args, "end_line"), max_chars=max_result_chars or READ_FILE_MAX_CHARS,
                preview=features.on("data_preview"),
            )
        if name == "remember":
            if not can_write:
                return "This delegated session is read-only."
            plan = plan_note(args, project)
            if isinstance(plan, str):
                return plan
            if before_write is not None:
                refusal = before_write(plan)
                if refusal:
                    return refusal
            return apply_write(plan, project)
        if name in WRITE_TOOLS:
            if not can_write:
                return "This delegated session is read-only."
            plan = plan_write(name, args, project, atomic=atomic)
            if isinstance(plan, str):
                return plan
            if before_write is not None:
                refusal = before_write(plan)
                if refusal:
                    return refusal
            return apply_write(plan, project, args)
        if name == "git_status":
            result = subprocess.run(
                ["git", "-C", str(project), "status", "--short", "--branch", "--untracked-files=normal"],
                text=True, capture_output=True, timeout=15,
            )
            if result.returncode:
                return f"Git status unavailable (exit code {result.returncode}): {result.stderr.strip()}"
            return result.stdout.strip() or "Working tree clean."
        if name == "git_diff":
            command = ["git", "-C", str(project), "diff", "HEAD", "--no-ext-diff", "--no-textconv"]
            raw_path = args.get("path")
            if raw_path:
                relative = _project_path(project, str(raw_path)).relative_to(project).as_posix()
                command.extend(["--", relative])
            result = subprocess.run(command, text=True, capture_output=True, timeout=15)
            if result.returncode:
                return f"Git diff unavailable (exit code {result.returncode}): {result.stderr.strip()}"
            return result.stdout[-20000:] or "No tracked diff against HEAD. Untracked files are not included in git diff."
        if name == "run_check":
            return sandbox.run_check(str(args.get("command", "")), project, stop=stop)
        if name == "run_command":
            command = args["command"]
            # A callback with `sees_read_only` (evals' command recorder) decides every command.
            sees_all = getattr(approve_command, "sees_read_only", False)
            if _is_auto_read_only(command, project) and not sees_all:
                if approve_command is None:
                    console.print(f"[dim]Running read-only inspection command:[/dim] {command}")
            elif approve_command is not None:
                if not approve_command(command):
                    return "User declined to run the command. Do not retry unless the user asks."
            else:
                console.print(f"\n[bold yellow]Agent requests command:[/bold yellow] {command}")
                if console.input("[yellow]Run this command? \\[y/N] [/yellow]").strip().lower() not in {"y", "yes"}:
                    return "User declined to run the command. Do not retry unless the user asks."
            try:
                result = sandbox.run_stoppable(command, shell=True, cwd=project, timeout=120, stop=stop)
            except sandbox.StoppedRun:
                return sandbox.STOPPED_CHECK
            output = f"exit code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            return output[-20000:]
        return f"Unknown tool: {name}"
    except subprocess.TimeoutExpired:
        return "Command timed out after 120 seconds."
    except Exception as exc:
        return f"Tool error: {type(exc).__name__}: {exc}"


def parse_arguments(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError as exc:
        return {"_error": f"Invalid tool arguments: {exc}"}
