"""format_rules: checks built from the format rules a deliverable task states in its own words.

Why (2026-10-20): the deliverable pipeline reported its checks passed on paths that did not
start with the absolute prefix the task stated, on a text line longer than the task's "at most N
characters", and on `"lines": [null, null]` when the task's template was
`{"file": "<path>", "lines": [<first>, <last>], ...}` (a JSON list, one object per item).
json_template checks only the key names of an object template; these rules add:

- **path prefix**: a stated prefix for paths (`starting /Users/me/`, "paths start with ..."):
  every path-like string in a JSON deliverable must start with it (and not with `~` when the task
  says "no `~`");
- **line length**: "at most N characters" said of lines: every line of a text deliverable
  (.txt/.md) is at most the largest such N;
- **list-of-object shapes**: a "JSON list/array ... one object per ...: `{...}`" template: each
  item is an object with every stated key, a quoted placeholder is a string, a bare one is not null
  (a number when its name says so: `<int>`, `<first>`, `<line>`), `[<a>, <b>]` is a list of that
  many such values, and "at most N characters" in a placeholder caps that string.

Rules only, no model call. Each failure names the file, the item and the field, what it holds and
what the task says, so a repair round can act on it. Switch: `format_rules`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from . import features

_MAX_SHOWN = 6  # failures reported per rule and file

# --- path prefix ------------------------------------------------------------------------------
_PREFIX = re.compile(r"\b(?:start|begin)(?:s|ing)?\s+(?:with\s+)?`?((?:/|~/)[^`\s,;)]+)`?",
                     re.IGNORECASE)
_NO_TILDE = re.compile(r"\bno\s+`?~`?(?:[\s,;).]|$)|\bwithout\s+(?:a\s+)?`?~`?(?:[\s,;).]|$)",
                       re.IGNORECASE)
_PATHISH = re.compile(r"^[~\w.@+-]*(?:/[^/\n]*)+$")  # segments may hold spaces

# --- line length ------------------------------------------------------------------------------
_AT_MOST = re.compile(r"(?:at most|no (?:more|longer) than|up to|max(?:imum)?(?: of)?|≤|<=)\s*"
                      r"(\d{1,4})\s*(?:characters|chars)\b", re.IGNORECASE)
_TEXT_EXT = (".txt", ".md")

# --- list-of-object templates -----------------------------------------------------------------
_LIST_CUE = re.compile(r"\bJSON\s+(?:list|array)\b|\b(?:list|array)\s+of\s+(?:JSON\s+)?objects\b",
                       re.IGNORECASE)
_PER_OBJECT = re.compile(r"\bone\s+object\s+(?:per|for\s+each)\b|\beach\s+(?:item|entry|element)"
                         r"\s+(?:is\s+)?(?:an?\s+)?object\b|\bobjects?\b", re.IGNORECASE)
_JSON_FILE = re.compile(r"`((?:[\w.-]+/)*[\w.-]+\.json)`")
_NUMBER_NAME = re.compile(r"\b(?:int|integer|number|float|count|line|lines|first|last|start|end|"
                          r"total|n|num|index|row|col|column|port|pid|score|amount|size)\b",
                          re.IGNORECASE)


@dataclass
class Shape:
    """What a template placeholder asks for: kind is 'string', 'number', 'value' (anything but
    null), 'list' (items: element shapes, fixed: the length is fixed), 'object' or 'any'."""

    kind: str
    nullable: bool = False
    max_chars: int | None = None
    items: list[Shape] = field(default_factory=list)
    fixed: bool = False
    text: str = ""  # the placeholder as written, for the failure message


@dataclass
class Rules:
    prefix: str | None = None  # path prefix
    no_tilde: bool = False
    line_max: int | None = None  # text deliverables: characters per line
    lists: dict[str, dict[str, Shape]] = field(default_factory=dict)  # json path -> key shapes

    @property
    def count(self) -> int:
        return bool(self.prefix) + bool(self.line_max) + len(self.lists)


def _template_end(text: str, start: int) -> int:
    """Index just past the `}` closing the template at text[start] == '{' (strings honoured)."""
    depth, i, quoted = 0, start, False
    while i < len(text):
        char = text[i]
        if quoted:
            if char == "\\":
                i += 1
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
        elif char == "`":
            return -1  # the backticked template ended unbalanced
        i += 1
    return -1


def _split_top(body: str) -> list[str]:
    """`body` split on commas outside strings, brackets and braces."""
    parts, depth, quoted, current = [], 0, False, []
    i = 0
    while i < len(body):
        char = body[i]
        current.append(char)
        if quoted:
            if char == "\\" and i + 1 < len(body):
                current.append(body[i + 1])
                i += 1
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        elif char == "," and depth == 0:
            current.pop()
            parts.append("".join(current))
            current = []
        i += 1
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def _shape(value: str) -> Shape:
    value = value.strip()
    nullable = bool(re.search(r"\bnull\b|\bNone\b", value)) and value != "null"
    chars = _AT_MOST.search(value)
    max_chars = int(chars.group(1)) if chars else None
    if value.startswith('"'):
        return Shape("string", nullable, max_chars, text=value)
    if value.startswith("["):
        inner = value[1:-1] if value.endswith("]") else value[1:]
        pieces = _split_top(inner)
        if not pieces or any("..." in piece or "…" in piece for piece in pieces):
            return Shape("list", nullable, text=value)
        items = [_shape(piece) for piece in pieces]
        fixed = len(items) >= 2 and all(not piece.startswith('"') for piece in pieces)
        return Shape("list", nullable, items=items, fixed=fixed, text=value)
    if value.startswith("{"):
        return Shape("object", nullable, text=value)
    if value in {"true", "false"}:
        return Shape("value", text=value)
    if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        return Shape("number", text=value)
    if value.startswith("<"):
        kind = "number" if _NUMBER_NAME.search(value.strip("<>")) else "value"
        return Shape(kind, nullable, text=value)
    return Shape("any", text=value)


def _template_shapes(template: str) -> dict[str, Shape]:
    """{key: Shape} of a `{"key": <placeholder>, ...}` template; {} when it is not of that form."""
    shapes: dict[str, Shape] = {}
    for part in _split_top(template.strip()[1:-1]):
        match = re.match(r'"((?:[^"\\]|\\.)+)"\s*:\s*(.+)$', part, re.DOTALL)
        if not match:
            return {}
        shapes[match.group(1)] = _shape(match.group(2))
    return shapes


def _list_templates(task: str, json_paths: list[str]) -> dict[str, dict[str, Shape]]:
    """json deliverable -> key shapes, for each "JSON list ... one object per ...: `{...}`"."""
    found: dict[str, dict[str, Shape]] = {}
    for match in re.finditer(r"`\{", task):
        start = match.start() + 1
        end = _template_end(task, start)
        if end < 0:
            continue
        before = task[max(0, start - 300):start]
        sentence = re.split(r"(?<=[.!?])\s+(?=[A-Z])", before)[-1]
        if not (_LIST_CUE.search(sentence) and _PER_OBJECT.search(sentence)):
            continue
        named = _JSON_FILE.findall(sentence)
        path = next((p for p in reversed(named) if p in json_paths), None)
        if path is None and len(json_paths) == 1 and not named:
            path = json_paths[0]
        if path is None or path in found:
            continue
        shapes = _template_shapes(task[start:end])
        if shapes:
            found[path] = shapes
    return found


def rules(task: str, deliverables: list[str]) -> Rules:
    """The format rules `task` states for these deliverables."""
    found = Rules()
    for match in _PREFIX.finditer(task):
        clause = task[max(0, match.start() - 120):match.start()]
        clause = re.split(r"(?<=[.;!?])\s", clause)[-1]
        prefix = match.group(1)
        if len(prefix) >= 2 and re.search(r"\bpaths?\b", clause, re.IGNORECASE):
            found.prefix = prefix
            window = task[match.start():match.end() + 40]
            found.no_tilde = bool(_NO_TILDE.search(window)) or prefix.startswith("/")
            break
    if any(path.endswith(_TEXT_EXT) for path in deliverables):
        limits = []
        for match in _AT_MOST.finditer(task):
            clause = task[max(0, match.start() - 160):match.start()]
            clause = re.split(r"(?<=[.;!?])\s|\b\d\.\s", clause)[-1]
            if re.search(r"\blines?\b", clause, re.IGNORECASE) and "`{" not in clause:
                limits.append(int(match.group(1)))
        if limits:
            found.line_max = max(limits)
    json_paths = [path for path in deliverables if path.endswith(".json")]
    if json_paths:
        found.lists = _list_templates(task, json_paths)
    return found


def _short(value: Any, limit: int = 80) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _path_strings(value: Any, where: str = "") -> list[tuple[str, str]]:
    """(where, string) for every path-like string in a JSON value (no URLs)."""
    out: list[tuple[str, str]] = []
    if isinstance(value, str):
        if "://" not in value and _PATHISH.match(value):
            out.append((where, value))
    elif isinstance(value, dict):
        for key, item in value.items():
            out += _path_strings(item, f'"{key}"' if not where else f'{where}."{key}"')
    elif isinstance(value, list):
        for n, item in enumerate(value):
            out += _path_strings(item, f"{where}[{n}]" if where else f"item {n}")
    return out


def _shape_problem(value: Any, shape: Shape) -> str | None:
    if value is None:
        return None if shape.nullable or shape.kind == "any" else "is null"
    if shape.kind == "string":
        if not isinstance(value, str):
            return f"is {_short(value)}, not a string"
        if shape.max_chars is not None and len(value) > shape.max_chars:
            return f"is {len(value)} characters (at most {shape.max_chars})"
    elif shape.kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"is {_short(value)}, not a number"
    elif shape.kind == "list":
        if not isinstance(value, list):
            return f"is {_short(value)}, not a list"
        if shape.fixed and len(value) != len(shape.items):
            return f"has {len(value)} item(s), not {len(shape.items)}"
        for n, item in enumerate(value):
            element = shape.items[min(n, len(shape.items) - 1)] if shape.items else None
            problem = _shape_problem(item, element) if element is not None else None
            if problem:
                return f"item {n} {problem}"
    elif shape.kind == "object":
        if not isinstance(value, dict):
            return f"is {_short(value)}, not an object"
    return None


def _list_problems(path: str, data: Any, shapes: dict[str, Shape]) -> list[str]:
    template = "{" + ", ".join(f'"{key}": {shape.text}' for key, shape in shapes.items()) + "}"
    if not isinstance(data, list):
        return [(f"{path} must be a JSON list of objects, one per item, each "
                 f"{template} (it is a {type(data).__name__})")]
    problems = []
    for n, item in enumerate(data):
        if not isinstance(item, dict):
            problems.append(f"{path} item {n} is {_short(item)}, not an object {template}")
            continue
        missing = [key for key in shapes if key not in item]
        if missing:
            problems.append(f"{path} item {n} lacks " + ", ".join(f'"{k}"' for k in missing))
        for key, shape in shapes.items():
            if key in item:
                problem = _shape_problem(item[key], shape)
                if problem:
                    problems.append(f'{path} item {n} "{key}" {problem}: the task\'s template '
                                    f"says {shape.text}")
    if len(problems) > _MAX_SHOWN:
        problems = problems[:_MAX_SHOWN] + [f"{path}: {len(problems) - _MAX_SHOWN} more like these"]
    return problems


def problems(task: str, files: dict[str, str], found: Rules | None = None) -> list[str]:
    """What the deliverables `files` (path -> content) break of the task's own format rules, one
    concrete line each ([] when format_rules is off or nothing is broken). A file that is not
    valid JSON is left to the parse check."""
    if not features.on("format_rules"):
        return []
    found = found if found is not None else rules(task, list(files))
    out: list[str] = []
    for path, text in files.items():
        if path.endswith(".json") and (found.prefix or path in found.lists):
            try:
                data = json.loads(text)
            except ValueError:
                continue
            if found.prefix:
                bad = []
                for where, value in _path_strings(data):
                    if value.startswith("~") and found.no_tilde:
                        bad.append(f"{where} is {_short(value)}: no `~`, the task wants paths "
                                   f"starting {found.prefix}")
                    elif not value.startswith(found.prefix) and (
                            value.startswith(("/", "~")) or found.prefix.lstrip("/~") in value):
                        bad.append(f"{where} is {_short(value)}, but the task says paths start "
                                   f"with {found.prefix}")
                out += [f"{path}: {line}" for line in bad[:_MAX_SHOWN]]
            if path in found.lists:
                out += _list_problems(path, data, found.lists[path])
        elif path.endswith(_TEXT_EXT) and found.line_max:
            long = [(n, line) for n, line in enumerate(text.split("\n"), 1)
                    if len(line) > found.line_max]
            for n, line in long[:_MAX_SHOWN]:
                out.append(f"{path} line {n} is {len(line)} characters; the task says at most "
                           f"{found.line_max}: rewrap it (`{line[:60]}...`)")
    return out
