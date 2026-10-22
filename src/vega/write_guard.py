"""write_guard: hold an agent-loop write once when it would silently lose what a file held.

Why (2026-10-19): some research failures were a write tool wrecking an existing file (`write_file
requirements.txt "requests==2.32.3\\n"` on a file never read dropped its other packages; a
config.py rewrite opened a docstring that swallowed a constant, and syntax and lint both passed),
and requirements files often lost lines. A write is held (not applied; the result says what
would be lost and how to go on) when:

- write_file replaces an existing file that was not read (or written) this turn and some of its
  non-blank lines would be lost;
- any write tool removes entries of a dependency/config file (requirements*.txt, *.lock,
  pyproject.toml and other *.toml, setup.cfg/*.cfg/*.ini, package.json), compared by package
  name or key, so a version change is not a loss but replacing one package's line with another
  package is;
- any write tool makes top-level names of a .py file disappear (def, class, assignments): for
  write_file every such name; for an edit, a name whose definition the edit did not touch (code
  swallowed by an unclosed string) or whose definition text is still there but no longer at
  module level. Renaming a definition you edited is not held.

Re-issuing exactly the same write after the note runs it; a path is held at most twice per turn.
Only the agent loop uses this: the pipelines' own writes (write_loop, project pipeline) go through
the agent's `_before_write` directly and are never held.
"""

from __future__ import annotations

import ast
import configparser
import difflib
import hashlib
import json
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

HOLD_PREFIX = "(Vega harness) Not written yet"
MAX_HOLDS_PER_PATH = 2
SHOW_LINES = 40
SHOW_LOST = 12

_REQUIREMENTS = re.compile(r"^(requirements.*|constraints.*|.*requirements)\.(txt|in)$",
                           re.IGNORECASE)
_REQ_NAME = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_LINE_KEY = re.compile(r"\s*([^\s=:<>!~@\[;,]+)")
_DEP_KEYS = ("dependencies", "requires", "require")


def dependency_kind(relative: str) -> str | None:
    """'requirements', 'toml', 'ini', 'json' or 'lines' for a dependency/config file; else None."""
    path = PurePosixPath(relative)
    name = path.name.lower()
    if _REQUIREMENTS.match(name) or (path.parent.name.lower() == "requirements"
                                     and path.suffix.lower() in {".txt", ".in"}):
        return "requirements"
    if name == "package.json":
        return "json"
    suffix = path.suffix.lower()
    if suffix == ".toml":
        return "toml"
    if suffix in {".cfg", ".ini"} or name in {".flake8", ".pylintrc"}:
        return "ini"
    if suffix == ".lock" or name == "pipfile":
        return "lines"
    return None


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_keys(text: str) -> dict[str, str]:
    """key -> the line it came from: the package name (canonical), or a whole option line."""
    keys: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            keys.setdefault(line, raw.strip())
            continue
        found = _REQ_NAME.match(line)
        if found:
            keys.setdefault(_canonical(found.group(1)), raw.strip())
    return keys


def _line_keys(text: str) -> dict[str, str]:
    keys: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";", "//")):
            continue
        found = _LINE_KEY.match(line)
        if found:
            keys.setdefault(found.group(1).lower(), line)
    return keys


def _is_dep_key(key: str) -> bool:
    last = key.rsplit(".", 1)[-1].lower().replace("_", "-")
    return any(last.endswith(word) for word in _DEP_KEYS) or "optional-dependencies" in key


def _flatten(value: object, prefix: str, keys: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys[path] = path
            _flatten(item, path, keys)
    elif isinstance(value, list) and prefix and _is_dep_key(prefix):
        for item in value:
            if isinstance(item, str):
                found = _REQ_NAME.match(item)
                if found:
                    keys[f"{prefix}[{_canonical(found.group(1))}]"] = f"{prefix}: {item}"


def _toml_keys(text: str) -> dict[str, str] | None:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        return None
    try:
        data = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError):
        return None
    keys: dict[str, str] = {}
    _flatten(data, "", keys)
    return keys


def _json_keys(text: str) -> dict[str, str] | None:
    try:
        data = json.loads(text)
    except ValueError:
        return None
    keys: dict[str, str] = {}
    if isinstance(data, dict):  # two levels: "dependencies.react", "scripts.test"
        for key, value in data.items():
            keys[str(key)] = str(key)
            if isinstance(value, dict):
                for inner in value:
                    keys[f"{key}.{inner}"] = f"{key}: {inner}"
    return keys


def _ini_keys(text: str) -> dict[str, str] | None:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read_string(text)
    except configparser.Error:
        return None
    keys: dict[str, str] = {}
    for section in parser.sections():
        keys[f"[{section}]"] = f"[{section}]"
        for option, value in parser.items(section, raw=True):
            keys[f"{section}.{option}"] = f"[{section}] {option}"
            if _is_dep_key(option):
                for item in (value or "").splitlines():
                    found = _REQ_NAME.match(item.split("#", 1)[0])
                    if found and item.strip():
                        keys[f"{section}.{option}[{_canonical(found.group(1))}]"] = \
                            f"[{section}] {option}: {item.strip()}"
    return keys


def entry_keys(kind: str, text: str) -> dict[str, str]:
    """The entries of a dependency/config file: key -> how to show it."""
    if kind == "requirements":
        return _requirement_keys(text)
    parsed = {"toml": _toml_keys, "json": _json_keys, "ini": _ini_keys}.get(kind)
    keys = parsed(text) if parsed else None
    return keys if keys is not None else _line_keys(text)


def lost_entries(relative: str, before: str, after: str) -> list[str]:
    """Entries of a dependency/config file that the new text no longer has (shown form)."""
    kind = dependency_kind(relative)
    if kind is None:
        return []
    old, new = entry_keys(kind, before), entry_keys(kind, after)
    if kind in {"toml", "json", "ini"} and not new and after.strip():
        new = _line_keys(after)  # the new text does not parse: compare its lines instead
        old = _line_keys(before)
    return [shown for key, shown in old.items() if key not in new]


# -- Python: module-level names ---------------------------------------------------------------

def _target_names(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [name for element in node.elts for name in _target_names(element)]
    if isinstance(node, ast.Starred):
        return _target_names(node.value)
    return []


def _collect(body: list[ast.stmt], names: dict[str, tuple[int, int]]) -> None:
    for node in body:
        span = (node.lineno, node.end_lineno or node.lineno)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            names.setdefault(node.name, (first, span[1]))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _target_names(target):
                    names.setdefault(name, span)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            for name in _target_names(node.target):
                names.setdefault(name, span)
        elif isinstance(node, ast.If):
            _collect(node.body, names)
            _collect(node.orelse, names)
        elif isinstance(node, ast.Try):
            _collect(node.body, names)
            for handler in node.handlers:
                _collect(handler.body, names)
            _collect(node.orelse, names)
            _collect(node.finalbody, names)
        elif isinstance(node, ast.With):
            _collect(node.body, names)


def module_names(source: str) -> dict[str, tuple[int, int]] | None:
    """Module-level def/class/assignment names -> (first line, last line); None if unparsable."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    names: dict[str, tuple[int, int]] = {}
    _collect(tree.body, names)
    return names


def _touched_lines(before: str, after: str) -> set[int]:
    """1-based lines of `before` that the change replaced or deleted (plus the lines next to an
    insertion point)."""
    old, new = before.splitlines(), after.splitlines()
    touched: set[int] = set()
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i1 == i2:  # an insertion between lines i1 and i1+1
            touched.update({i1, i1 + 1})
        touched.update(range(i1 + 1, i2 + 1))
    return touched


def _still_written(name: str, after: str) -> bool:
    """The name's definition text is in the new file (just no longer at module level)."""
    pattern = rf"^[ \t]*(?:async[ \t]+def|def|class)[ \t]+{re.escape(name)}\b|^[ \t]*{re.escape(name)}[ \t]*(?::[^=\n]*)?[+\-*/|&]?="
    return re.search(pattern, after, re.MULTILINE) is not None


def lost_names(tool: str, relative: str, before: str, after: str) -> tuple[list[str], list[str]]:
    """(.py) Module-level names the change removes: (lost, of which still written but swallowed).
    For edits, a name whose definition the edit replaced is intended (a rename) unless its text
    is still there."""
    if not relative.endswith((".py", ".pyi")):
        return [], []
    old = module_names(before)
    new = module_names(after)
    if not old or new is None:
        return [], []  # nothing defined before, or the new text does not parse (atomic_writes)
    gone = [name for name in old if name not in new]
    if not gone:
        return [], []
    swallowed = [name for name in gone if _still_written(name, after)]
    if tool == "write_file":
        return gone, swallowed
    touched = _touched_lines(before, after)
    lost = [name for name in gone if name in swallowed
            or not touched.intersection(range(old[name][0], old[name][1] + 1))]
    return lost, [name for name in swallowed if name in lost]


def _lost_lines(before: str, after: str) -> list[str]:
    kept = {line.strip() for line in after.splitlines()}
    return [line.rstrip() for line in before.splitlines()
            if line.strip() and line.strip() not in kept]


@dataclass
class Hold:
    relative: str
    reasons: list[str]
    lost: list[str]
    before: str
    tool: str

    def note(self) -> str:
        lines = self.before.splitlines()
        width = len(str(min(len(lines), SHOW_LINES)))
        shown = "\n".join(f"{n:>{width}}\t{line}" for n, line in enumerate(lines[:SHOW_LINES], 1))
        more = (f"\n[{len(lines) - SHOW_LINES} more lines; read_file with start_line to see them]"
                if len(lines) > SHOW_LINES else "")
        lost = "\n".join(f"  - {item}" for item in self.lost[:SHOW_LOST])
        if len(self.lost) > SHOW_LOST:
            lost += f"\n  - ... and {len(self.lost) - SHOW_LOST} more"
        how = ("To change only part of it, use edit_file (old_text copied from the lines below, "
               "without the numbers) or replace_lines; to add an entry or definition, append it "
               "(edit_file with an empty old_text)")
        if self.tool != "write_file":
            how = ("Make old_text cover only what should change and keep every other line; to add "
                   "an entry or definition, append it (edit_file with an empty old_text)")
        return (
            f"{HOLD_PREFIX}: {'; '.join(self.reasons)}.\nThis {self.tool} would lose:\n{lost}\n"
            f"{how}. If losing these is intended, make exactly the same {self.tool} call again "
            f"and it will run.\n{self.relative} currently reads:\n{shown}{more}"
        )


@dataclass
class WriteGuard:
    """One agent-loop turn's state: the files read or written, and the writes already held."""

    read: set[str] = field(default_factory=set)
    held: set[tuple[str, str]] = field(default_factory=set)
    holds: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _key(relative: str) -> str:
        return posixpath.normpath(relative.replace("\\", "/")) if relative else relative

    def saw(self, relative: str) -> None:
        """The model has this file's current content (read_file, its own write, a hold note)."""
        self.read.add(self._key(relative))

    def assess(self, tool: str, relative: str, before: str | None, after: str) -> Hold | None:
        """The hold for this write, or None when it may run. Pure: does not record anything."""
        if before is None or not before.strip() or before == after:
            return None
        reasons: list[str] = []
        lost: list[str] = []
        entries = lost_entries(relative, before, after)
        names, swallowed = lost_names(tool, relative, before, after)
        if tool == "write_file" and self._key(relative) not in self.read:
            gone = _lost_lines(before, after)
            if gone:
                reasons.append(f"{relative} already exists and you have not read it this turn; "
                               "write_file replaces the whole file")
                if not entries and not names:
                    lost += gone
        if entries:
            what = ("replaces a different package's line" if tool != "write_file"
                    and len(entries) <= 2 and dependency_kind(relative) == "requirements"
                    else "removes entries")
            reasons.append(f"this {what} in the dependency/config file {relative} "
                           f"({', '.join(entries[:6])}{', ...' if len(entries) > 6 else ''} "
                           "would no longer be there)")
            lost += entries
        if names:
            detail = ""
            if swallowed:
                detail = (f"; {', '.join(swallowed)} still appear in the text but are no longer "
                          "module-level (inside a string or docstring that is not closed where "
                          "you meant, or indented into another block)")
            reasons.append(f"{relative} would no longer define {', '.join(names)} at module "
                           f"level{detail}")
            lost += [f"{name} (module-level name)" for name in names]
        if not reasons:
            return None
        return Hold(relative, reasons, lost, before, tool)

    def check(self, tool: str, relative: str, before: str | None, after: str) -> str | None:
        """The hold note for this write (recorded, so the same write runs next time), or None."""
        key = (self._key(relative), hashlib.sha1(after.encode("utf-8", "replace")).hexdigest())
        if key in self.held or self.holds.get(key[0], 0) >= MAX_HOLDS_PER_PATH:
            return None
        hold = self.assess(tool, relative, before, after)
        if hold is None:
            return None
        self.held.add(key)
        self.holds[key[0]] = self.holds.get(key[0], 0) + 1
        self.saw(relative)  # the note shows the current content
        return hold.note()
