"""Redesign B of the failure diagnosis (2026-10-16): edits as transactions.

A large share of failures with Granite were the edit tools: an edit that broke the file was saved
with a WARNING (one attempt then spent dozens of steps and compactions on one unclosed paren),
`write_file` saved a leading ```` ```python ```` fence, and after compaction the model rebuilt
`old_text` from memory (text mismatches). Each part has its own switch
(features.py), on by default so it can be A/B'd off:

- `fence_strip`: a code file's content (or new_text) wrapped in a ``` fence, or starting with a
  line that only names the file, is cleaned before anything else, with a note in the result.
- `atomic_writes`: every write tool computes the whole new file first and refuses it (file
  unchanged) when it does not parse (Python, JSON, TOML, INI; YAML when PyYAML is installed) or
  when it adds undefined names (F821) or redefinitions (F811) that the file did not have. The
  refusal names the line and column, shows the three lines around it and says how to retry. A
  file that was already broken may be written when the write adds no new parse error.
- `edit_by_name`: `replace_definition` replaces (or adds) a function, class or method by name,
  `Class.method` included: the model's code is cleaned (fences, line numbers, indentation),
  re-indented to the target, keeps the target's decorators and docstring unless it brings its
  own, and is merged with merge_code's semantics (new imports and helpers come along).
- `pin_editing_target`: context compaction keeps the current text of the definitions written
  this turn (from disk, capped) in the summary, so `old_text` never has to come from memory.

Nothing here calls a model; the model still decides every edit. A refusal is a tool result.
"""

from __future__ import annotations

import ast
import difflib
import io
import re
import textwrap
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import features

# -- fence_strip ----------------------------------------------------------------------------------

# Files whose text is code or structured data: a ``` fence in them is never intended. Markdown,
# reStructuredText and plain text are not here (fences are legitimate content there).
CODE_SUFFIXES = {
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".json", ".toml", ".yaml",
    ".yml", ".ini", ".cfg", ".sh", ".bash", ".rs", ".go", ".c", ".h", ".cc", ".cpp", ".hpp",
    ".java", ".kt", ".rb", ".php", ".sql", ".css", ".scss", ".html", ".xml", ".swift", ".cs",
    ".lua", ".r",
}
_HASH_COMMENT = {".py", ".pyi", ".sh", ".bash", ".toml", ".yaml", ".yml", ".rb", ".r", ".ini",
                 ".cfg"}
_SLASH_COMMENT = {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".rs", ".go", ".c", ".h", ".cc",
                  ".cpp", ".hpp", ".java", ".kt", ".php", ".swift", ".cs", ".scss"}
_FENCE_OPEN = re.compile(r"^\s*(```|~~~)[\w.+#-]*[ \t]*$")
_FENCE_CLOSE = re.compile(r"^\s*(```|~~~)\s*$")


def _names_file(line: str, relative: str) -> bool:
    """A line that only names the file (`shop/orders.py`, `File: orders.py`, `**orders.py**`),
    unless it is a valid comment in the file's own language (then it is harmless code)."""
    suffix = Path(relative).suffix.lower()
    stripped = line.strip()
    if (suffix in _HASH_COMMENT and stripped.startswith("#")) or \
            (suffix in _SLASH_COMMENT and stripped.startswith("//")):
        return False
    text = stripped.strip("`*").strip()
    for prefix in ("#", "//", "File:", "file:", "Path:", "path:", "Filename:", "filename:"):
        text = text.removeprefix(prefix).strip()
    text = text.strip("`*: ")
    return bool(text) and text in {relative, Path(relative).name}


def strip_wrapping(text: str, relative: str) -> tuple[str, list[str]]:
    """`text` without a wrapping ``` fence and without a first line naming the file, for code
    files; (cleaned text, notes saying what was removed). Other text comes back unchanged."""
    if Path(relative).suffix.lower() not in CODE_SUFFIXES or not text:
        return text, []
    notes: list[str] = []
    lines = text.split("\n")
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None:
        return text, []
    if _names_file(lines[first], relative):
        notes.append(f"removed a first line that only names the file ({lines[first].strip()!r})")
        del lines[first]
        first = next((i for i, line in enumerate(lines) if line.strip()), None)
        if first is None:
            return "\n".join(lines), notes
    if _FENCE_OPEN.match(lines[first]):
        close = next((j for j in range(first + 1, len(lines)) if _FENCE_CLOSE.match(lines[j])), None)
        body = lines[first + 1:close] if close is not None else lines[first + 1:]
        trailing = [line for line in lines[close + 1:] if line.strip()] if close is not None else []
        note = "removed the ``` fence around the code"
        if trailing:
            note += f" and {len(trailing)} non-code line(s) after it"
        notes.append(note)
        lines = body
        inner = next((i for i, line in enumerate(lines) if line.strip()), None)
        if inner is not None and _names_file(lines[inner], relative):
            notes.append(f"removed a first line that only names the file ({lines[inner].strip()!r})")
            del lines[inner]
        if text.endswith("\n") and (not lines or lines[-1] != ""):
            lines.append("")
    elif _FENCE_CLOSE.match(lines[-1] if lines[-1].strip() else (lines[-2] if len(lines) > 1 else "")):
        # only a stray closing fence at the end (the opening one was dropped elsewhere)
        end = len(lines) - 1 if lines[-1].strip() else len(lines) - 2
        if not any(_FENCE_OPEN.match(line) or _FENCE_CLOSE.match(line) for line in lines[:end]):
            notes.append("removed a stray ``` line at the end")
            del lines[end]
    return "\n".join(lines), notes


# -- atomic_writes --------------------------------------------------------------------------------


@dataclass
class Problem:
    """A parse error: what, where (1-based line and column; None when the parser does not say)."""

    message: str
    line: int | None
    column: int | None
    kind: str = "SyntaxError"


def parse_error(path: Path | str, text: str) -> Problem | None:
    """The first parse error of `text` as `path`'s kind of file; None when it parses or the kind
    is not checked (Python, JSON, TOML, INI/CFG, YAML when PyYAML is installed)."""
    import configparser
    import json
    import tomllib

    suffix = Path(path).suffix.lower()
    if suffix in {".py", ".pyi"}:
        try:
            compile(text, str(path), "exec", dont_inherit=True)
        except SyntaxError as exc:
            kind = type(exc).__name__  # IndentationError / TabError are SyntaxErrors too
            return Problem(str(exc.msg), exc.lineno, exc.offset, kind)
        except ValueError as exc:  # null bytes
            return Problem(str(exc), None, None, "ValueError")
        return None
    if suffix == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            return Problem(exc.msg, exc.lineno, exc.colno, "invalid JSON")
        return None
    if suffix == ".toml":
        try:
            tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            message = str(exc)
            where = re.search(r"\(at line (\d+), column (\d+)\)", message)
            message = re.sub(r"\s*\(at line \d+, column \d+\)", "", message)
            if where:
                return Problem(message, int(where.group(1)), int(where.group(2)), "invalid TOML")
            return Problem(message, None, None, "invalid TOML")
        return None
    if suffix in {".ini", ".cfg"}:
        try:
            configparser.ConfigParser(interpolation=None).read_string(text)
        except configparser.Error as exc:
            return Problem(str(exc).splitlines()[0], getattr(exc, "lineno", None), None,
                           "invalid INI")
        return None
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            return None
        try:
            list(yaml.safe_load_all(text))
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            line = mark.line + 1 if mark is not None else None
            column = mark.column + 1 if mark is not None else None
            return Problem(str(getattr(exc, "problem", None) or exc), line, column, "invalid YAML")
        return None
    return None


MAX_SCAN = 25  # error-scan rounds for a file that does not parse


def _python_errors(text: str) -> list[tuple[str, Problem]]:
    """Every syntax error Python reports when each reported line is blanked in turn (bounded),
    as messages with numbers removed, so two versions of a broken file compare by kind of error
    rather than by line: a write that only moves or rewrites an existing error adds nothing."""
    lines = text.split("\n")
    found: list[tuple[str, Problem]] = []
    seen_lines: set[int] = set()
    for _ in range(MAX_SCAN):
        problem = parse_error("x.py", "\n".join(lines))
        if problem is None or problem.line is None:
            break
        index = min(max(problem.line, 1), len(lines)) - 1
        found.append((re.sub(r"\d+", "N", problem.message), problem))
        if index in seen_lines:
            break
        seen_lines.add(index)
        lines[index] = ""
    return found


def changed_lines(before: str | None, after: str) -> tuple[int, int]:
    """1-based inclusive range of `after` lines that differ from `before` (the whole file for a
    new file); (n+1, n) when nothing changed."""
    new = after.split("\n")
    if before is None:
        return 1, len(new)
    old = before.split("\n")
    start = 0
    while start < min(len(old), len(new)) and old[start] == new[start]:
        start += 1
    end_old, end_new = len(old), len(new)
    while end_old > start and end_new > start and old[end_old - 1] == new[end_new - 1]:
        end_old -= 1
        end_new -= 1
    if end_new <= start:  # pure deletion: point at the join
        return start + 1, start + 1
    return start + 1, end_new


def _window(text: str, line: int, *, around: int = 1) -> str:
    lines = text.split("\n")
    centre = min(max(line, 1), len(lines)) - 1
    start, end = max(0, centre - around), min(len(lines), centre + around + 1)
    width = len(str(end))
    return "\n".join(f"{'>' if n == centre else ' '} {n + 1:>{width}}\t{lines[n]}"
                     for n in range(start, end))


def _backticked(message: str) -> str:
    match = re.search(r"`([^`]+)`", message)
    return match.group(1) if match else message


def new_static_findings(project: Path, relative: str, before: str | None,
                        after: str) -> list[tuple[int, str, str]]:
    """F821/F811 findings of `after` that `before` did not have, compared by code and name
    (acceptance's static check: ruff when available, else an ast stand-in). Redefinitions of
    imports are not counted (acceptance ignores them too)."""
    from .acceptance import static_findings

    if not relative.endswith((".py", ".pyi")):
        return []
    if before is not None and parse_error(relative, before) is not None:
        return []  # no baseline to compare with: the parse rule decides alone
    known: dict[tuple[str, str], int] = {}
    for _, code, message in (static_findings(project, relative, before) if before else []):
        key = (code, _backticked(message))
        known[key] = known.get(key, 0) + 1
    lines = after.split("\n")
    new = []
    for line, code, message in static_findings(project, relative, after):
        if code == "F811" and 0 < line <= len(lines) \
                and lines[line - 1].lstrip().startswith(("import ", "from ")):
            continue
        key = (code, _backticked(message))
        if known.get(key, 0) > 0:
            known[key] -= 1
            continue
        new.append((line, code, message))
    return new


@dataclass
class Verdict:
    """atomic check outcome: refusal text (None = write it) and a note for an allowed write."""

    refusal: str | None
    note: str = ""


def _retry_hint(tool: str, args: dict[str, Any], region: tuple[int, int], relative: str,
                before: str | None) -> str:
    first, last = region
    span = f"lines {first}-{last}" if last > first else f"line {first}"
    if tool == "edit_file":
        hint = (f"Your new_text became {span} of the would-be file (shown with `>` above). Send "
                "edit_file again with the same old_text (it is still in the file) and corrected "
                "new_text.")
    elif tool == "replace_lines":
        hint = (f"Send replace_lines again for lines {args.get('start_line')}-"
                f"{args.get('end_line')} (the file's line numbers have not changed) with corrected "
                "new_text.")
    elif tool == "replace_definition":
        hint = (f"Your code became {span} of the would-be file. Send replace_definition again "
                f"for {args.get('name')} with the corrected code.")
    elif tool == "pipeline":  # a pipeline writer replies with code, not tool calls
        hint = (f"Your code became {span} of the would-be {relative} (shown above). Reply again "
                f"with the corrected code for {relative} in the same form as before.")
    elif tool == "write_file":
        hint = ("Send write_file again with the corrected content"
                + (f" (your content's {span} differ from the file now)" if before else "") + ".")
    else:
        hint = "Correct the text and send the write again."
    if features.on("edit_by_name") and tool in {"edit_file", "replace_lines"} \
            and relative.endswith(".py") and before is not None:
        name = enclosing_definition(before, min(first, len(before.split("\n"))))
        if name:
            hint += (f" For code, replace_definition with name={name!r} and the whole new "
                     "definition needs no old_text.")
    return hint


def unparsable_reply(relative: str, code: str, project: Path) -> str | None:
    """A pipeline writer's Python code that does not parse by itself, as the refusal to send
    back (its lines numbered as in the reply); None when it parses or is not Python. merge_code
    cannot merge such code and kept the file as it was, so the broken reply was silently lost."""
    if not relative.endswith((".py", ".pyi")) or parse_error(relative, code) is None:
        return None
    return check("pipeline", relative, None, code, project).refusal


def check(tool: str, relative: str, before: str | None, after: str, project: Path,
          args: dict[str, Any] | None = None, foreign: dict[str, str] | None = None) -> Verdict:
    """The atomic_writes rule for one computed write. `foreign`: undefined names the file must not
    import as-is (project_layer_imports), name -> where it lives and what to do; the refusal says
    so instead of "define or import"."""
    args = args or {}
    region = changed_lines(before, after)
    problem = parse_error(relative, after)
    if problem is not None:
        old_problem = parse_error(relative, before) if before is not None else None
        if old_problem is not None:
            added = _added_parse_error(relative, before or "", after, problem, region)
            if added is not None:
                return Verdict(_parse_refusal(tool, relative, after, added, region, args, before,
                                              still_broken=True))
            remaining = (f"{problem.kind} at line {problem.line}: {problem.message}"
                         if problem.line else f"{problem.kind}: {problem.message}")
            return Verdict(None, f"\n(It did not parse before this write either; this write added "
                                 f"no new parse error. Still: {remaining}.)")
        return Verdict(_parse_refusal(tool, relative, after, problem, region, args, before))
    new = new_static_findings(project, relative, before, after)
    if new:
        return Verdict(_static_refusal(tool, relative, after, new, region, args, before, foreign))
    return Verdict(None)


def _added_parse_error(relative: str, before: str, after: str, problem: Problem,
                       region: tuple[int, int]) -> Problem | None:
    """The first parse error `after` has that `before` did not (None: nothing new)."""
    if relative.endswith((".py", ".pyi")):
        old = [kind for kind, _ in _python_errors(before)]
        for kind, found in _python_errors(after):
            if kind in old:
                old.remove(kind)
            else:
                return found
        return None
    # other formats report one error: new when it sits in the lines this write changed
    if problem.line is not None and region[0] - 1 <= problem.line <= region[1] + 1:
        return problem
    return None


def _parse_refusal(tool: str, relative: str, after: str, problem: Problem,
                   region: tuple[int, int], args: dict[str, Any], before: str | None,
                   still_broken: bool = False) -> str:
    where = f"line {problem.line}" + (f", column {problem.column}" if problem.column else "") \
        if problem.line else "an unknown line"
    if still_broken:
        head = (f"Write not applied: {relative} already did not parse, and this write adds a new "
                f"{problem.kind} at {where}: {problem.message}. The file is unchanged.")
    else:
        head = (f"Write not applied: {relative} would not parse ({problem.kind} at {where}: "
                f"{problem.message}). The file is unchanged.")
    shown = f"\nThe would-be lines:\n{_window(after, problem.line)}" if problem.line else ""
    outside = ""
    if problem.line and not region[0] - 1 <= problem.line <= region[1] + 1:
        outside = (f"\nThe error is reported at line {problem.line}, but it comes from the lines "
                   f"this write changed ({region[0]}-{region[1]}), e.g. an unclosed bracket or "
                   "string.")
    return head + shown + outside + "\n" + _retry_hint(tool, args, region, relative, before)


def _static_refusal(tool: str, relative: str, after: str, new: list[tuple[int, str, str]],
                    region: tuple[int, int], args: dict[str, Any], before: str | None,
                    foreign: dict[str, str] | None = None) -> str:
    lines = after.split("\n")
    parts = []
    for line, code, message in new[:3]:
        name = _backticked(message)
        text = lines[line - 1] if 0 < line <= len(lines) else ""
        column = text.find(name) + 1 if name in text else None
        where = f"line {line}" + (f", column {column}" if column else "")
        parts.append(f"{where}: {code} {message}\n{_window(after, line)}")
    more = f"\n(+{len(new) - 3} more)" if len(new) > 3 else ""
    undefined = [_backticked(m) for _, code, m in new if code == "F821"]
    redefined = [_backticked(m) for _, code, m in new if code == "F811"]
    what = []
    if undefined:
        what.append("undefined name(s) " + ", ".join(dict.fromkeys(undefined)))
    if redefined:
        what.append("a second definition of " + ", ".join(dict.fromkeys(redefined)))
    advice = []
    foreign = {name: why for name, why in (foreign or {}).items() if name in undefined}
    for name, why in foreign.items():
        advice.append(f"`{name}` {why}")
    if [name for name in undefined if name not in foreign]:
        advice.append(("for the other undefined names: " if foreign else "")
                      + "define or import each undefined name in the same write (or write the "
                      "helper first), or use the name the code already has")
    if redefined:
        advice.append("change the existing definition instead of adding another copy")
    return (f"Write not applied: it would add {' and '.join(what)} to {relative}. The file is "
            f"unchanged.\n" + "\n".join(parts) + more + "\nTo retry: " + "; ".join(advice)
            + ". " + _retry_hint(tool, args, region, relative, before))


# -- locating definitions -------------------------------------------------------------------------

_DEF_LINE = re.compile(r"^(\s*)(?:async\s+def|def|class)\s+([A-Za-z_]\w*)")


@dataclass
class Span:
    """A definition in a file: qualified name, first line (decorators included), def/class line,
    last line (1-based, inclusive), its indentation and the ast node when the file parses."""

    name: str
    start: int
    header: int
    end: int
    indent: str
    node: ast.AST | None = None


def definitions(source: str) -> list[Span]:
    """Every function, class and method with its qualified name (`Outer.Inner.method`), in file
    order. Uses ast when the file parses, otherwise indentation (so a broken file still has its
    definitions found)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return _text_definitions(source)
    found: list[Span] = []
    lines = source.split("\n")

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
                line = lines[node.lineno - 1]
                found.append(Span(prefix + node.name, start, node.lineno,
                                  node.end_lineno or node.lineno,
                                  line[:len(line) - len(line.lstrip())], node))
                visit(node.body, prefix + node.name + ".")

    visit(tree.body, "")
    found.sort(key=lambda span: span.start)
    return found


def _text_definitions(source: str) -> list[Span]:
    lines = source.split("\n")
    found: list[Span] = []
    stack: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = _DEF_LINE.match(line)
        if not match:
            continue
        indent = len(match.group(1))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        name = ".".join([*(n for _, n in stack), match.group(2)])
        stack.append((indent, match.group(2)))
        start = index
        while start > 0 and lines[start - 1].strip().startswith("@") and \
                len(lines[start - 1]) - len(lines[start - 1].lstrip()) == indent:
            start -= 1
        end = index
        for later in range(index + 1, len(lines)):
            text = lines[later]
            if not text.strip():
                continue
            if len(text) - len(text.lstrip()) <= indent and not text.lstrip().startswith((")", "]", "}")):
                break
            end = later
        found.append(Span(name, start + 1, index + 1, end + 1, match.group(1)))
    return found


def enclosing_definition(source: str, line: int) -> str | None:
    """Qualified name of the innermost function or class around `line`, or None."""
    inner = [span for span in definitions(source) if span.start <= line <= span.end]
    return max(inner, key=lambda span: span.start).name if inner else None


def resolve(spans: list[Span], name: str) -> tuple[Span | None, str]:
    """The span named `name` (exact qualified name, else a unique `.name` suffix); a note when
    it was found by suffix."""
    exact = [span for span in spans if span.name == name]
    if exact:
        return exact[0], ""
    suffix = [span for span in spans if span.name.endswith("." + name)]
    if len(suffix) == 1:
        return suffix[0], f"{name} is {suffix[0].name}"
    return None, ""


# -- edit_by_name: replace_definition -------------------------------------------------------------


def _unescape(code: str) -> tuple[str, str]:
    if "\n" not in code.strip("\n") and "\\n" in code:
        return code.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t"), \
            "turned literal \\n into line breaks"
    return code, ""


def _dedent(code: str) -> str:
    """The model's definition moved to column 0. Also handles a first line sent without its
    indentation while the rest kept it (copied from inside a class)."""
    code = textwrap.dedent(code.strip("\n"))
    lines = code.split("\n")
    if len(lines) > 1 and parse_error("snippet.py", code) is not None and not lines[0][:1].isspace():
        rest = textwrap.dedent("\n".join(lines[1:]))
        candidate = lines[0] + "\n" + textwrap.indent(rest, "    ", lambda line: bool(line.strip()))
        if parse_error("snippet.py", candidate) is None:
            return candidate
    return code


def _snippet_problem(code: str) -> str:
    problem = parse_error("snippet.py", code)
    assert problem is not None
    where = f"line {problem.line}" + (f", column {problem.column}" if problem.column else "")
    shown = f"\n{_window(code, problem.line)}" if problem.line else ""
    return (f"Edit not applied: your code does not parse ({problem.kind} at {where} of the code "
            f"you sent: {problem.message}). The file is unchanged.{shown}\nSend replace_definition "
            "again with the corrected code.")


def _string_continuations(code: str) -> set[int]:
    """1-based lines inside multi-line strings that are not docstrings: never re-indented,
    since that would change the string's value."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))
    keep: set[int] = set()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError):
        return keep
    doc_starts = {(n.lineno, n.col_offset) for n in ast.walk(tree)
                  if isinstance(n, ast.Constant) and id(n) in docstrings}
    for token in tokens:
        if token.type == tokenize.STRING and token.end[0] > token.start[0] \
                and token.start not in doc_starts:
            keep.update(range(token.start[0] + 1, token.end[0] + 1))
    return keep


def _reindent(block: list[str], source_indent: str, target_indent: str,
              first_line: int, frozen: set[int]) -> list[str]:
    """Lines of a definition moved from `source_indent` to `target_indent`; blank lines become
    empty and string continuation lines (`frozen`, 1-based in the snippet) stay as written."""
    out = []
    for offset, line in enumerate(block):
        number = first_line + offset
        if not line.strip():
            out.append("")
        elif number in frozen:
            out.append(line)
        elif line.startswith(source_indent):
            out.append(target_indent + line[len(source_indent):])
        else:
            out.append(target_indent + line.lstrip())
    return out


def _node_start(node: ast.AST) -> int:
    return min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])  # type: ignore[attr-defined]


def _docstring_node(node: ast.AST) -> ast.Expr | None:
    body = getattr(node, "body", None)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[0]
    return None


def _definition_lines(code: str, node: ast.AST, target_indent: str,
                      keep_decorators: list[str] | None, keep_docstring: list[str] | None,
                      frozen: set[int]) -> list[str]:
    """The snippet's definition, re-indented to target_indent, with the original decorators
    (when the snippet has none) and the original docstring (when the snippet has none)."""
    lines = code.split("\n")
    start = _node_start(node)
    source_indent = " " * node.col_offset  # type: ignore[attr-defined]
    block = _reindent(lines[start - 1:node.end_lineno], source_indent, target_indent,  # type: ignore[attr-defined]
                      start, frozen)
    if keep_docstring and _docstring_node(node) is None:
        body = node.body  # type: ignore[attr-defined]
        first = body[0]
        if first.lineno > node.lineno:  # type: ignore[attr-defined]
            body_indent = target_indent + " " * (first.col_offset - node.col_offset)  # type: ignore[attr-defined]
            old_indent = keep_docstring[0][:len(keep_docstring[0]) - len(keep_docstring[0].lstrip())]
            doc = [body_indent + line[len(old_indent):] if line.startswith(old_indent) else line
                   for line in keep_docstring]
            at = first.lineno - start
            # comments between the signature and the first statement stay above it
            block[at:at] = doc
    if keep_decorators is not None and not getattr(node, "decorator_list", []):
        old_indent = keep_decorators[0][:len(keep_decorators[0]) - len(keep_decorators[0].lstrip())] \
            if keep_decorators else ""
        decorators = [target_indent + line[len(old_indent):] if line.startswith(old_indent) else line
                      for line in keep_decorators]
        block = decorators + block
    return block


def _find_in_snippet(tree: ast.Module, name: str) -> tuple[ast.AST | None, str]:
    """The snippet's definition for `name` (last part of a qualified name): top-level, or inside
    a top-level class of the same parent name (the model wrapped the method in its class)."""
    parts = name.split(".")
    leaf = parts[-1]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == leaf:
            return node, ""
    if len(parts) > 1:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == parts[-2]:
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                            and member.name == leaf:
                        others = [m.name for m in node.body if m is not member
                                  and isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                     ast.ClassDef))]
                        note = f"took {leaf} out of your class {node.name}"
                        if others:
                            note += f" (ignored {', '.join(others)}: send each by name)"
                        return member, note
    return None, ""


def plan_definition(target: Path, relative: str, name: str, code: str):
    """replace_definition: the WritePlan (or refusal text) for replacing or adding `name`."""
    from .codemerge import _insert_imports, keep_docstrings
    from .tools import WritePlan, _strip_line_numbers

    if target.suffix not in {".py", ".pyi"}:
        return "replace_definition works on Python files; use edit_file for other files."
    name = name.strip().removeprefix("def ").removeprefix("class ").strip().rstrip(":()")
    if not name or not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", name):
        return ("Give name as `function`, `Class` or `Class.method` (letters, digits, _ and "
                "dots), and the complete code.")
    notes: list[str] = []
    cleaned, stripped = strip_wrapping(code, relative)
    notes += stripped
    cleaned, unescaped = _unescape(_strip_line_numbers(cleaned))
    if unescaped:
        notes.append(unescaped)
    cleaned = _dedent(cleaned)
    if not cleaned.strip():
        return "Give the complete code of the definition."
    if parse_error("snippet.py", cleaned) is not None:
        return _snippet_problem(cleaned)
    tree = ast.parse(cleaned)
    node, note = _find_in_snippet(tree, name)
    if note:
        notes.append(note)
    if node is None:
        defined = [n.name for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        what = f"defines {', '.join(defined)}" if defined else "defines no function or class"
        return (f"Edit not applied: your code {what}, not {name.split('.')[-1]}. Send the "
                f"complete definition of {name} (starting with `def` or `class`).")
    original = target.read_text(encoding="utf-8") if target.is_file() else ""
    spans = definitions(original)
    span, resolved = resolve(spans, name)
    if resolved:
        notes.append(resolved)
    lines = original.split("\n")
    frozen = _string_continuations(cleaned)
    parts = name.split(".")
    if span is not None:
        decorators = lines[span.start - 1:span.header - 1] if span.header > span.start else []
        old_node = span.node
        old_doc = None
        if old_node is not None:
            doc = _docstring_node(old_node)
            if doc is not None:
                old_doc = lines[doc.lineno - 1:doc.end_lineno]
        if isinstance(old_node, ast.ClassDef) and isinstance(node, ast.ClassDef) \
                and _drops_methods(old_node, node):
            from .codemerge import _merge_class

            kept = sorted(_member_names(old_node) - _member_names(node))
            gen_lines = cleaned.split("\n")
            block = _merge_class(old_node, node, lines, gen_lines, span.start)
            notes.append(f"merged member by member; kept {', '.join(kept)} which your code left "
                         "out")
        else:
            block = _definition_lines(cleaned, node, span.indent, decorators or None,
                                      old_doc, frozen)
            if decorators and not getattr(node, "decorator_list", []):
                notes.append("kept its decorators")
            if old_doc and _docstring_node(node) is None:
                notes.append("kept its docstring")
        updated_lines = lines[:span.start - 1] + block + lines[span.end:]
        verb = f"Replaced definition {span.name} in {relative} (was lines {span.start}-{span.end}"
        qualified = span.name
    elif len(parts) > 1:
        parent, _ = resolve(spans, ".".join(parts[:-1]))
        if parent is None or not (parent.node is None or isinstance(parent.node, ast.ClassDef)):
            available = ", ".join(s.name for s in spans[:30]) or "no definitions"
            return (f"Edit not applied: {relative} has no class {'.'.join(parts[:-1])} to add "
                    f"{parts[-1]} to. It defines: {available}.")
        body_indent = _body_indent(lines, parent)
        block = ["", *_definition_lines(cleaned, node, body_indent, None, None, frozen)]
        updated_lines = lines[:parent.end] + block + lines[parent.end:]
        verb = f"Added definition {parent.name}.{parts[-1]} to {relative} ("
        qualified = f"{parent.name}.{parts[-1]}"
    else:
        if original.strip() and not spans and parse_error(relative, original) is not None:
            problem = parse_error(relative, original)
            return (f"Edit not applied: {relative} does not parse (line {problem.line}: "  # type: ignore[union-attr]
                    f"{problem.message}), and no definition was found to put {name} next to. "  # type: ignore[union-attr]
                    "Fix the file with replace_lines or write_file first.")
        block = _definition_lines(cleaned, node, "", None, None, frozen)
        body = lines[:]
        while body and not body[-1].strip():
            body.pop()
        gap = ["", ""] if body else []
        updated_lines = body + gap + block + [""]
        verb = f"Added definition {name} to {relative} ("
        qualified = name
    updated = "\n".join(updated_lines)
    updated, extra_notes = _carry_module_level(updated, cleaned, tree, node, qualified,
                                               _insert_imports)
    notes += extra_notes
    if original and parse_error(relative, updated) is None:
        restored = keep_docstrings(original, updated)
        if restored != updated:
            updated = restored
    if not updated.endswith("\n"):
        updated += "\n"
    final, _ = resolve(definitions(updated), qualified)
    if final is not None:
        verb += (", now " if verb.endswith(tuple("0123456789")) else "") + \
            f"lines {final.start}-{final.end}"
    message = verb.removesuffix(" (") + ")" + (f"; {'; '.join(notes)}" if notes else "") + "."
    before = original if target.is_file() else None
    return WritePlan("replace_definition", target, relative, before, updated, message)


def _member_names(node: ast.ClassDef) -> set[str]:
    return {m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _drops_methods(old: ast.ClassDef, new: ast.ClassDef) -> bool:
    return bool(_member_names(old) - _member_names(new)) and bool(_member_names(new))


def _body_indent(lines: list[str], parent: Span) -> str:
    if parent.node is not None and getattr(parent.node, "body", None):
        first = parent.node.body[0]  # type: ignore[attr-defined]
        return " " * first.col_offset
    for line in lines[parent.header:parent.end]:
        if line.strip():
            return line[:len(line) - len(line.lstrip())]
    return parent.indent + "    "


def _carry_module_level(updated: str, code: str, tree: ast.Module, node: ast.AST,
                        qualified: str, insert_imports) -> tuple[str, list[str]]:
    """merge_code semantics for the rest of the snippet: new imports go in with the file's
    imports, other top-level functions/classes replace their namesakes or are added at the end,
    and new module-level assignments (e.g. `K = TypeVar('K')`, which merge_code used to
    drop) go just before the definition that was written."""
    notes: list[str] = []
    code_lines = code.split("\n")
    try:
        current = ast.parse(updated)
    except SyntaxError:
        return updated, notes
    imports = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    existing = {ast.unparse(n) for n in ast.walk(current) if isinstance(n, (ast.Import, ast.ImportFrom))}
    missing = [line for line in imports if line not in existing]
    assigns = new_assignments(code_lines, tree, current)
    helpers = [n for n in tree.body if n is not node
               and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               and not (isinstance(n, ast.ClassDef) and node in n.body)]  # a wrapping class
    top = qualified.split(".")[0]
    if assigns:
        updated = insert_before_definition(updated, assigns, top)
        notes.append(f"added {len(assigns)} module-level assignment(s) before {top}")
    for helper in helpers:
        block = _definition_lines(code, helper, "", None, None, _string_continuations(code))
        spans = {s.name: s for s in definitions(updated)}
        lines = updated.split("\n")
        if helper.name in spans:
            span = spans[helper.name]
            lines[span.start - 1:span.end] = block
            notes.append(f"replaced {helper.name} too")
        else:
            while lines and not lines[-1].strip():
                lines.pop()
            lines += ["", "", *block, ""]
            notes.append(f"added {helper.name} too")
        updated = "\n".join(lines)
    if missing:
        updated = insert_imports(updated, missing)
        notes.append("added " + "; ".join(missing))
    return updated, notes


def _assigned_names(node: ast.stmt) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]  # type: ignore[attr-defined]
    return [t.id for t in targets if isinstance(t, ast.Name)]


def new_assignments(code_lines: list[str], tree: ast.Module, current: ast.Module) -> list[str]:
    """The snippet's module-level assignments (`K = TypeVar('K')`, constants) whose names the
    file does not bind at top level yet, as source text."""
    bound = {name for n in current.body if isinstance(n, (ast.Assign, ast.AnnAssign))
             for name in _assigned_names(n)}
    assigns = []
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            names = _assigned_names(n)
            if names and not set(names) & bound:
                assigns.append("\n".join(code_lines[n.lineno - 1:n.end_lineno]))
    return assigns


def insert_before_definition(updated: str, assigns: list[str], top: str) -> str:
    """Put `assigns` just before the top-level definition `top` (at the end when it is absent)."""
    spans = {s.name: s for s in definitions(updated)}
    at = spans[top].start if top in spans else len(updated.split("\n")) + 1
    lines = updated.split("\n")
    lines[at - 1:at - 1] = [*assigns, "", ""]
    return "\n".join(lines)


# -- pin_editing_target ---------------------------------------------------------------------------

PIN_MAX_CHARS = 1500
PIN_TITLE = ("Current text of what you are editing (from the file now; copy old_text from here "
             "without the line numbers, or use the line numbers with replace_lines)")


def _anchor(tool: str, args: dict[str, Any], text: str) -> tuple[str | None, int | None]:
    """(qualified name, line) in the current `text` that a write call was about."""
    lines = text.split("\n")
    if tool == "replace_definition":
        return str(args.get("name", "")).strip() or None, None
    if tool == "replace_lines":
        try:
            return None, int(args.get("start_line"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None, None
    if tool == "edit_file":
        stripped = [line.strip() for line in lines]
        for key in ("new_text", "old_text"):
            candidates = [line.strip() for line in str(args.get(key, "")).split("\n") if line.strip()]
            for candidate in candidates[:3]:
                if candidate in stripped:
                    return None, stripped.index(candidate) + 1
        first = next((line.strip() for line in str(args.get("old_text", "")).split("\n")
                      if line.strip()), "")
        close = difflib.get_close_matches(first, stripped, n=1, cutoff=0.6) if first else []
        if close:
            return None, stripped.index(close[0]) + 1
    return None, None


def pinned_text(targets: list[tuple[str, dict[str, Any]]], project: Path,
                max_chars: int = PIN_MAX_CHARS) -> str:
    """The current text of the definitions this turn's write calls touched, newest first, as
    numbered lines under PIN_TITLE, within max_chars; "" when there is nothing to show.
    `targets`: (tool name, arguments) of each write call this turn, oldest first."""
    from .tools import _project_path

    blocks: list[str] = []
    seen: set[tuple[str, str]] = set()
    used = len(PIN_TITLE) + 2
    for tool, args in reversed(targets):
        raw = str(args.get("path", ""))
        try:
            path = _project_path(project, raw)
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(project.resolve()).as_posix()
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        name, line = _anchor(tool, args, text)
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        first = last = None
        label = ""
        if path.suffix in {".py", ".pyi"}:
            spans = definitions(text)
            span = resolve(spans, name)[0] if name else None
            if span is None and line is not None:
                inner = [s for s in spans if s.start <= line <= s.end]
                # a method rather than its whole class, but a short class whole
                span = max(inner, key=lambda s: s.start) if inner else None
            if span is not None:
                first, last, label = span.start, span.end, span.name
        if first is None and line is not None and lines:
            first, last = max(1, line - 8), min(len(lines), line + 12)
            label = f"around line {line}"
        if first is None or (relative, label) in seen:
            continue
        seen.add((relative, label))
        problem = parse_error(path, text)
        status = (f"; the file does not parse: line {problem.line}: {problem.message}"
                  if problem else "")
        head = f"{relative}, {label} (lines {first}-{last}{status}):"
        body_lines = []
        budget = max_chars - used - len(head) - 1
        width = len(str(last))
        for number in range(first, (last or first) + 1):
            entry = f"{number:>{width}}\t{lines[number - 1] if number <= len(lines) else ''}"
            if sum(len(x) + 1 for x in body_lines) + len(entry) + 80 > budget:
                body_lines.append(f"[... lines {number}-{last} not shown: read_file {relative} "
                                  f"start_line={number}]")
                break
            body_lines.append(entry)
        if budget < 200 or len(body_lines) < 2:
            break
        block = head + "\n" + "\n".join(body_lines)
        blocks.append(block)
        used += len(block) + 1
        if used >= max_chars - 200:
            break
    if not blocks:
        return ""
    return PIN_TITLE + ":\n" + "\n".join(blocks)
