"""json_template: the keys a task states for a JSON file it asks for, checked on the written file.

Why (2026-10-18): a JSON file was written without keys the task's template named (`Write
result.json as {"choice": ..., "version": ..., "sources": [...]}`), and nothing noticed. The
task's own template is the spec: every key it states must be present, and `sources`, when
stated, must be a list of strings. Rules only, no model call; used by the agent loop's finish
gate and the deliverable pipeline's checks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import features

_JSON_PATH = r"`?((?:[\w.-]+/)*[\w.-]+\.json)`?"
_AS = re.compile(_JSON_PATH + r"\s+(?:as|like|shaped like|in the form|of the form)\s*:?\s*`?\{")
_WITH_KEYS = re.compile(_JSON_PATH + r"[^.\n{]{0,60}?\bwith (?:exactly )?(?:these |the following |the )?"
                        r"(?:string |top-level )?(?:keys|fields)\b(.{0,1500})", re.IGNORECASE | re.DOTALL)
_TOKEN = re.compile(r'"(?:[^"\\\n]|\\.)*"|[{}\[\]:]')
_NAME = re.compile(r"`([^`\s]+)`|\"([^\"\s]+)\"|'([^'\s]+)'|\b([A-Za-z_][\w-]*)\b")
_BULLET = re.compile(r"^\s*[-*•]\s+")
_FILLER = {"and", "or", "the", "keys", "fields", "plus"}
_STOP_LIST = re.compile(r"[.;](?:\s|$)|\b(?:where|that|which|then|or both|if|for|with)\b")


def _object_keys(text: str, start: int) -> list[str]:
    """The top-level keys of the `{...}` template starting at `start` (it may hold prose
    placeholders such as `"alphadb" or "betadb"` and `[...]`, so it is scanned, not parsed)."""
    depth, keys, last = 0, [], None
    for match in _TOKEN.finditer(text, start):
        token = match.group(0)
        if token in "{[":
            depth += 1
        elif token in "}]":
            depth -= 1
            if depth <= 0:
                break
        elif token == ":" and depth == 1 and last is not None \
                and not text[last.end():match.start()].strip():
            key = last.group(0)[1:-1]
            if key and key not in keys:
                keys.append(key)
        last = match if token.startswith('"') else None
    return keys


def _drop_parentheses(text: str) -> str:
    out, depth = [], 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        elif not depth:
            out.append(char)
    return "".join(out)


def _first_name(piece: str, quoted_only: bool) -> str | None:
    for match in _NAME.finditer(piece):
        if match.group(4) is None:
            return next(group for group in match.groups() if group).strip("\"'")
        if not quoted_only and match.group(4).lower() not in _FILLER:
            return match.group(4)
    return None


def _listed_keys(text: str) -> list[str]:
    """The names after "with these keys": a bulleted list (`- `a`: what it is`, the first name
    of each bullet) or an inline one (`a` (what it is), `b` and `c` / "a, b, c"); what each one
    is (in parentheses, or after a bullet's colon) and the prose after the list are left out."""
    lines = text.split("\n")
    if not lines[0].strip(" :") and len(lines) > 1 and _BULLET.match(lines[1]):
        names = []
        for line in lines[1:]:
            if not _BULLET.match(line):
                break
            name = _first_name(line, quoted_only=True)
            if name:
                names.append(name)
        return list(dict.fromkeys(names))
    line = _drop_parentheses(lines[0])
    colon = line.find(":")
    first_quote = min((line.find(c) for c in "`\"'" if c in line), default=-1)
    if 0 <= colon < 120 and (first_quote < 0 or colon < first_quote):
        line = line[colon + 1:]  # "keys, each mapped to its value: `A`, `B`"
    stop = _STOP_LIST.search(line)
    line = line[:stop.start()] if stop else line
    quoted_only = any(c in line for c in "`\"'")
    pieces = re.split(r",|\band\b|\bor\b", line)
    if not quoted_only and any(len(piece.split()) > 1 for piece in pieces):
        return []  # prose, not a list of names
    names = [_first_name(piece, quoted_only) for piece in pieces]
    return list(dict.fromkeys(name for name in names if name))


def stated(task: str) -> dict[str, list[str]]:
    """JSON file -> the keys the task states for it (only files with a stated template)."""
    found: dict[str, list[str]] = {}
    for match in _AS.finditer(task):
        keys = _object_keys(task, match.end() - 1)
        if keys:
            found.setdefault(match.group(1), keys)
    for match in _WITH_KEYS.finditer(task):
        if match.group(1) in found:
            continue
        keys = _listed_keys(match.group(2))
        if keys:
            found[match.group(1)] = keys
    return found


def problems(project: Path, task: str, paths: list[str] | None = None) -> list[str]:
    """What a written JSON file lacks of the task's template, one line per file ([] when it
    matches, the file is missing (other checks say so) or json_template is off). `paths`: only
    these files."""
    if not features.on("json_template"):
        return []
    found = []
    for path, keys in stated(task).items():
        if paths is not None and path not in paths:
            continue
        try:
            data = json.loads((Path(project) / path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue  # missing or not JSON: the format checks report it
        template = "{" + ", ".join(f'"{key}"' for key in keys) + "}"
        if not isinstance(data, dict):
            found.append(f"{path} must be a JSON object with the keys the task states: {template}")
            continue
        missing = [key for key in keys if key not in data]
        if missing:
            found.append(f"{path} lacks the key(s) the task states: "
                         + ", ".join(f'"{key}"' for key in missing)
                         + f" (the task's template: {template})")
        sources = data.get("sources")
        if "sources" in keys and "sources" in data and not (
                isinstance(sources, list) and all(isinstance(item, str) for item in sources)):
            found.append(f'{path}: "sources" must be a list of strings (the URLs relied on)')
    return found
