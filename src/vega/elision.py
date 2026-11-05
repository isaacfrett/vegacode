"""elision_guard: refuse a Python write that stands a comment or `...` in for code the file had.

Why (2026-10-27): in the debug logs of 10-15..27, 12 saved Python writes (10 eval runs; 9 from
the agent loop, 3 from the project pipeline) stood a comment in for code the file had
(`# ... original body continues ...`, `# … (rest of the code unchanged)`), deleting it
(selfbuild: sandbox.py, tools.py, server.py, tui.py, cli.py). Such a write parses, so
atomic_writes and write-recovery cannot see it.
The elision-guard review (2026-10-27) has the counts and the replay.

The rule (`elided(relative, before, after, sent, target)`), Python files that exist already:

- a marker is a comment-only line (or the comment after a lone `pass`/`...`) that says code was
  left out: it starts with `...`/`…` (the finish gate's `_ELIDED` form), ends with `word...`, or
  names the omission (rest of the code/function/file, omitted, for brevity, same as before,
  existing code follows, placeholder for the rest, body continues). Weaker wording that also
  heads kept code (`# Existing imports (unchanged)`, `# Implementation remains unchanged`) is a
  marker only when no code follows it in its block. Commented-out code (`#     # Process...`),
  comments the file already had and comments inside strings never count. An Ellipsis `...`
  statement beside other statements in a body is a marker too;
- a marker counts only where code is gone: in its function (or, inside a new method, its class;
  or the module), the changed block around it drops old code lines that appear nowhere in the
  new version and are not rewritten as a similar line. A marker the merge dropped from the file
  (replace_definition keeps no trailing comment) is checked against the replaced definition;
- a definition whose whole body (docstring aside) is `...` counts when the old file's definition
  of the same name had real code (`@overload` / `@abstractmethod` stubs never count).

New files, Ellipsis stubs in protocols and abstract or overloaded methods, `...` in strings and
docstrings, and markers that drop nothing are allowed. Nothing here calls a model.

Checked: agent-loop writes (tools.plan_write with `atomic`) and, under pipeline_signals, the
writing pipelines' writes (write_loop.save: direct, project, deliverable, non-Python; the merged
file, plus the reply's own code where the pipeline keeps it), refused like their other refusals.
"""

from __future__ import annotations

import ast
import difflib
import io
import re
import tokenize
from dataclasses import dataclass

# a line standing for elided code: "...", "…", "// ... existing code ...", "# ... rest unchanged"
# (the finish gate's form for answer blocks, agent._ELIDED)
ELIDED_LINE = re.compile(r"^\s*(?:(?://|#|/\*|<!--|--)\s*)?(?:\.\.\.|…)")
_CODE = (r"(?:code|file|module|function|method|class|implementation|body|logic|tests?|"
         r"definitions?|functions|methods|helpers?|imports|setup|handling|attributes)")
# a comment that says code was left out, wherever it stands
_MARKER_WORDS = re.compile(
    r"^\s*(?:\.\.\.|…)"  # "# ... original body ..."
    r"|^(?!e\.g\.|example)[^=:\d]*\b[A-Za-z]{2,}\s*(?:\.\.\.|…)\s*\)?\s*$"  # "# Other methods..."
    r"|\brest\s+of\s+(?:the\s+)?(?:\w+\s+){0,2}" + _CODE + r"\b"
    r"|\bsame\s+as\s+before\b|\b" + _CODE + r"\b[^.]{0,30}\bomitted\b|\bfor\s+brevity\b"
    r"|\bplaceholder\s+for\s+the\s+rest\b"
    r"|\b(?:existing|original)\s+(?:\w+\s+){0,2}" + _CODE + r"\s+(?:follows|continues)\b"
    r"|\b(?:implementation|body|code)\s+continues\b",
    re.IGNORECASE)
# a comment that may also head code kept below it ("# Existing imports (unchanged)" over the
# imports): a marker only when no code follows it in its block
_WEAK_MARKER_WORDS = re.compile(
    r"\b" + _CODE + r"\b[^.]{0,40}\b(?:remains?|stays?|is|are|kept)\s+"
    r"(?:the\s+same|unchanged|as\s+(?:is|before))"
    r"|\b" + _CODE + r"\b[^.]{0,20}\(\s*unchanged\b"
    r"|^(?:other|additional|remaining)\s+(?:\w+\s+){0,2}(?:unchanged|omitted)\b"
    r"|\b(?:existing|original|previous)\s+" + _CODE + r"(?:\s+here)?\s*(?:\(.*\))?\s*$",
    re.IGNORECASE)
_STUB_LINE = re.compile(r"^\s*(?:pass|\.\.\.)\s*#")


@dataclass
class Elision:
    """Where a write elides code: the new file's line, that line's text, and what it drops."""

    line: int
    text: str
    lost: str  # e.g. "3 lines of `process`", "methods `a`, `b` of `Store`", "`main`"
    name: str  # the definition it sits in ("" at module level)
    sent: bool = False  # `line` counts in the text the model sent (the merge dropped the marker)


def _markers(after: str) -> list[int]:
    """1-based lines of `after` holding an elision marker comment (strings never count)."""
    lines = after.split("\n")
    found: list[int] = []
    tokens = tokenize.generate_tokens(io.StringIO(after).readline)
    while True:
        try:
            token = next(tokens)
        except (StopIteration, tokenize.TokenError, IndentationError, SyntaxError):
            return found  # a fragment (edit_file new_text) may end inside brackets
        if token.type != tokenize.COMMENT:
            continue
        row = token.start[0]
        line = lines[row - 1] if row <= len(lines) else ""
        alone = not line[:token.start[1]].strip()
        if not (alone or _STUB_LINE.match(line)):
            continue
        if re.match(r"#(?:\s{2,}|\s*#)", token.string):
            continue  # commented-out code (`#     # Process job...`), not a stand-in
        words = token.string.lstrip("#").strip()
        if _MARKER_WORDS.search(words) or (_WEAK_MARKER_WORDS.search(words)
                                           and not _code_below(lines, row)):
            found.append(row)
    return found


def _code_below(lines: list[str], row: int) -> bool:
    """The next code line after 1-based `row` (comments and blanks skipped) is in the same block
    (indented at least as deep) and is not a stub `pass` / `...`."""
    indent = len(lines[row - 1]) - len(lines[row - 1].lstrip())
    for line in lines[row:]:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        return len(line) - len(line.lstrip()) >= indent and text not in {"pass", "..."}
    return False


def _definitions(tree: ast.Module) -> dict[str, ast.AST]:
    """Qualified name ("Class.method", "outer.inner") -> def/class node: the last definition of
    a name (the one in effect), never an `@overload` signature."""
    out: dict[str, ast.AST] = {}

    def walk(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                if not _decorated(node, "overload") or name not in out:
                    out[name] = node
                walk(node.body, name + ".")
            elif isinstance(node, (ast.If, ast.Try)):  # `if TYPE_CHECKING:`, try/except imports
                walk(getattr(node, "body", []) + getattr(node, "orelse", []), prefix)
    walk(tree.body, "")
    return out


def _decorated(node: ast.AST, *names: str) -> bool:
    """`node` has a decorator named one of `names` (`@overload`, `@typing.overload`)."""
    for decorator in getattr(node, "decorator_list", []):
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        last = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if last in names:
            return True
    return False


def _span(node: ast.AST) -> tuple[int, int]:
    first = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
    return first, node.end_lineno or node.lineno


def _body(node: ast.AST) -> list[ast.stmt]:
    body = list(getattr(node, "body", []))
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None),
                                                             ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]  # docstring
    return body


def _is_ellipsis(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
        and stmt.value.value is Ellipsis


def _is_stub(node: ast.AST) -> bool:
    """A body of only `...`/`pass`/`raise NotImplementedError` (docstring aside)."""
    body = _body(node)
    return all(_is_ellipsis(s) or isinstance(s, ast.Pass)
               or (isinstance(s, ast.Raise) and "NotImplementedError" in ast.unparse(s))
               for s in body)


def _code_lines(source: list[str], node: ast.AST, inner: dict[str, ast.AST]) -> list[str]:
    """`node`'s own non-blank, non-comment lines, stripped (its nested definitions excluded:
    they are compared by name)."""
    first, last = _span(node)
    skip: set[int] = set()
    for child in inner.values():
        a, b = _span(child)
        if first < a and b <= last:
            skip.update(range(a, b + 1))
    out = []
    for number in range(first + 1, last + 1):
        text = source[number - 1].strip() if number <= len(source) else ""
        if number not in skip and text and not text.startswith("#"):
            out.append(text)
    return out


def _children(node: ast.AST) -> set[str]:
    """Names a class (or module) body defines directly: defs, classes, assignments."""
    names: set[str] = set()
    for stmt in getattr(node, "body", []):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in stmt.names)
    return names


def _similar(old: str, new: str) -> bool:
    """`new` reads as an edited `old` (difflib ratio >= 0.6)."""
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    return matcher.real_quick_ratio() >= 0.6 and matcher.quick_ratio() >= 0.6 \
        and matcher.ratio() >= 0.6


def _lost_at(name: str, old: ast.AST, new: ast.AST, old_src: list[str], new_src: list[str],
             row: int, new_end: int | None = None) -> str:
    """What the write dropped where the marker on new line `row` stands, in definition `name`
    (the module when name is ""): the old code lines in the changed block around the marker that
    appear nowhere in the new definition; "" if none (the marker is only a comment beside kept
    or rewritten code)."""
    if isinstance(old, ast.Module):
        old_first, old_last, new_first, new_last = 1, len(old_src), 1, len(new_src)
    else:
        (old_first, old_last), (new_first, new_last) = _span(old), _span(new)
        new_last = max(new_last, new_end or 0)
    old_rows = [n for n in range(old_first, old_last + 1) if old_src[n - 1].strip()]
    new_rows = [n for n in range(new_first, new_last + 1) if new_src[n - 1].strip()]
    old_text = [old_src[n - 1].strip() for n in old_rows]
    new_text = [new_src[n - 1].strip() for n in new_rows]
    if row not in new_rows:
        return ""
    at = new_rows.index(row)
    groups: list[list[tuple[str, int, int, int, int]]] = [[]]
    for op in difflib.SequenceMatcher(None, old_text, new_text, autojunk=False).get_opcodes():
        if op[0] == "equal":
            groups.append([])
        else:
            groups[-1].append(op)
    for group in groups:
        if group and group[0][3] <= at < max(group[-1][4], group[0][3] + 1):
            kept = set(new_text)
            dropped = [line for op in group for line in old_text[op[1]:op[2]]
                       if not line.startswith("#") and line not in kept]
            added = [line for op in group for line in new_text[op[3]:op[4]]
                     if not line.startswith("#") and line not in {"...", "pass"}]
            if len(dropped) * len(added) <= 40_000:  # a line rewritten is not a line left out
                dropped = [line for line in dropped if not any(_similar(line, new)
                                                               for new in added)]
            if not dropped:
                return ""
            label = f"`{name}`" if name else "the module"
            gone = sorted(_children(old) - _children(new)) \
                if isinstance(old, (ast.Module, ast.ClassDef)) else []
            shown = "".join(f", `{m}`" for m in gone[:6]) + (" …" if len(gone) > 6 else "")
            count = len(dropped)
            return f"{count} line{'s' * (count != 1)} of {label}{shown}"
    return ""


def _lost_in(name: str, old: ast.AST | None, new: ast.AST | None, old_src: list[str],
             new_src: list[str], old_defs: dict[str, ast.AST],
             new_defs: dict[str, ast.AST]) -> str:
    """What the new version of function `name` lost anywhere (for a marker the merge dropped,
    so it has no place in the new file); "" if nothing."""
    if old is None or new is None:
        return ""
    old_inner = {k: v for k, v in old_defs.items() if k.startswith(name + ".")}
    new_inner = {k: v for k, v in new_defs.items() if k.startswith(name + ".")}
    remaining = list(_code_lines(new_src, new, new_inner))
    lost = 0
    for line in _code_lines(old_src, old, old_inner):
        if line in remaining:
            remaining.remove(line)
        else:
            lost += 1
    missing = sorted(k[len(name) + 1:] for k in old_inner if k not in new_defs)
    if lost or missing:
        parts = [f"{lost} line{'s' * (lost != 1)}"] if lost else []
        parts += [f"`{m}`" for m in missing[:4]]
        return f"{', '.join(parts)} of `{name}`"
    return ""


def elided(relative: str, before: str | None, after: str, sent: str | None = None,
           target: str | None = None) -> Elision | None:
    """The first place where the write `before` -> `after` stands a marker or `...` in for code
    the file had; None when it does not (or the file is new, not Python, or does not parse).
    `sent`: the text the model sent (content / new_text / code), whose marker comments a merge
    may have dropped from `after` (replace_definition keeps no trailing comments); such a marker
    counts against `target` (replace_definition's name) or the definitions the write changed."""
    if before is None or not relative.endswith((".py", ".pyi")) or not before.strip():
        return None
    try:
        old_tree, new_tree = ast.parse(before), ast.parse(after)
    except (SyntaxError, ValueError):
        return None
    old_src, new_src = before.split("\n"), after.split("\n")
    old_defs, new_defs = _definitions(old_tree), _definitions(new_tree)

    def enclosing(row: int) -> tuple[str, int]:
        best, size, end = "", None, 0
        for name, node in new_defs.items():
            a, b = _span(node)
            indent = len(new_src[node.lineno - 1]) - len(new_src[node.lineno - 1].lstrip())
            while b < len(new_src) and (not new_src[b].strip() or (  # trailing comments
                    new_src[b].lstrip().startswith("#")
                    and len(new_src[b]) - len(new_src[b].lstrip()) > indent)):
                b += 1
            if a <= row <= b and (size is None or b - a < size):
                best, size, end = name, b - a, b
        return best, end

    had = {line.strip() for line in old_src}  # a marker the file already had is not this write's
    rows = {row for row in _markers(after) if new_src[row - 1].strip() not in had}
    for node in ast.walk(new_tree):  # `...` beside other statements in a body
        body = getattr(node, "body", None)
        if isinstance(body, list) and len(_body(node)) > 1:
            rows.update(s.lineno for s in _body(node) if _is_ellipsis(s))
    for row in sorted(rows):
        name, end = enclosing(row)
        if name and name not in old_defs:  # a new method: what its class held there counts
            name, end = name.rpartition(".")[0], 0
            if not isinstance(old_defs.get(name), ast.ClassDef):
                continue  # a new function: nothing it stands in for
        old = old_defs.get(name) if name else old_tree
        new = new_defs.get(name) if name else new_tree
        if old is None or new is None or type(old) is not type(new):
            continue
        lost = _lost_at(name, old, new, old_src, new_src, row, end)
        if lost:
            return Elision(row, new_src[row - 1].strip(), lost, name)
    sent_lines = sent.split("\n") if sent else []
    kept = {line.strip() for line in new_src}  # markers still in the file were checked above
    sent_rows = [row for row in _markers(sent)
                 if sent_lines[row - 1].strip() not in had | kept] if sent else []
    if sent_rows:
        changed: set[int] = set()
        matcher = difflib.SequenceMatcher(None, old_src, new_src, autojunk=False)
        for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                changed.update(range(j1 + 1, max(j1 + 1, j2) + 1))
        names = [target] if target in new_defs else sorted(
            (n for n, node in new_defs.items()
             if any(_span(node)[0] <= row <= _span(node)[1] for row in changed)),
            key=lambda n: _span(new_defs[n])[1] - _span(new_defs[n])[0])
        for name in names:
            if name in old_defs and not isinstance(new_defs[name], ast.ClassDef):
                lost = _lost_in(name, old_defs[name], new_defs[name], old_src, new_src,
                                old_defs, new_defs)
                if lost:
                    row = sent_rows[0]
                    return Elision(row, sent_lines[row - 1].strip(), lost, name, sent=True)
    for name, node in new_defs.items():  # a whole body replaced by `...`
        old = old_defs.get(name)
        if isinstance(node, ast.ClassDef) or old is None or isinstance(old, ast.ClassDef) \
                or _decorated(node, "overload", "abstractmethod", "abstractproperty"):
            continue
        body = _body(node)
        if body and all(_is_ellipsis(s) for s in body) and not _is_stub(old):
            row = body[0].lineno
            lines = len(_code_lines(old_src, old, {k: v for k, v in old_defs.items()
                                                   if k.startswith(name + ".")}))
            return Elision(row, new_src[row - 1].strip(), f"the {lines}-line body of `{name}`",
                           name)
    return None


def refusal(tool: str, relative: str, found: Elision) -> str:
    """The tool result for a refused write (the file is unchanged)."""
    where = f" in `{found.name}`" if found.name else ""
    if tool == "write_file":
        how = ("Send the whole file with every line written out, or change only the part you "
               "mean to with replace_definition (one function or class by name) or edit_file.")
    elif tool == "replace_definition":
        how = ("Send the whole definition with every line written out; to change part of it, "
               "use edit_file on just those lines.")
    elif tool == "pipeline":  # write_loop.save: the writer replies with code, no tools
        how = ("Reply with the complete file, every function and class written out in full; "
               "never a comment or `...` in place of code that stays.")
    else:
        how = ("Write out the code in full, or make new_text replace only the lines you change "
               "(edit_file) so the rest stays as it is.")
    line = (f"line {found.line} of the text you sent" if found.sent
            else f"line {found.line} of the new {relative}")
    return (f"Write not applied: {line}{where} stands in for "
            f"code instead of containing it ({found.text!r}), and the file would lose "
            f"{found.lost}. Vega saves exactly what you send: a comment like that deletes the "
            f"code it stands for. {how} The file is unchanged.")
