"""Code merging for the writing pipelines: apply a model's code to a file by whole definitions.

Moved out of pipeline.py (2026-10-17): the direct, project and
deliverable pipelines, the shared write loop (write_loop.py) and edits.py all merge through it, so
it should not live inside one pipeline's module. pipeline.py re-exports these names, so existing
imports keep working.
"""

from __future__ import annotations

import ast
from typing import Any

from . import features
from .tools import _top_level_names


def merge_code(original: str, generated: str, *, replace_whole: bool = True) -> str:
    """Apply generated code to a file: replace definitions it redefines, keep the rest, add new
    ones; a reply that redefines everything the file defines replaces the whole file (unless
    replace_whole is False: then definitions are always merged one by one)."""
    if features.on("large_projects"):
        merged = _merge(original, generated, replace_whole=replace_whole, methods=True)
    else:
        merged = _merge(original, generated, replace_whole=replace_whole)
    if features.on("pipeline_signals"):
        merged = carry_assignments(merged, generated)
    return keep_docstrings(original, merged) if features.on("keep_docstrings") else merged


def carry_assignments(merged: str, generated: str) -> str:
    """Module-level assignments of the reply that a merge by definitions dropped (pipeline_signals):
    a changed value (`LIMIT = 10` for `LIMIT = 5`) replaces the file's one in place, and a new
    one (`K = TypeVar('K')`, a constant) goes just before the first of the reply's definitions,
    as replace_definition does (edits.new_assignments). Unchanged when the result would not parse."""
    from .edits import _assigned_names, insert_before_definition, new_assignments

    if merged == generated:
        return merged
    try:
        gen_tree, tree = ast.parse(generated), ast.parse(merged)
    except SyntaxError:
        return merged
    gen_lines, lines = generated.split("\n"), merged.split("\n")
    assigned = (ast.Assign, ast.AnnAssign)
    current = {}
    for node in tree.body:
        if isinstance(node, assigned) and len(_assigned_names(node)) == 1:
            current.setdefault(_assigned_names(node)[0], node)
    replacements = []
    for node in gen_tree.body:
        if not isinstance(node, assigned) or len(_assigned_names(node)) != 1:
            continue
        old = current.get(_assigned_names(node)[0])
        text = gen_lines[node.lineno - 1:node.end_lineno]
        if old is not None and lines[old.lineno - 1:old.end_lineno] != text:
            replacements.append((old.lineno, old.end_lineno, text))
    for first, last, text in sorted(replacements, reverse=True):
        lines[first - 1:last] = text
    result = "\n".join(lines)
    try:
        assigns = new_assignments(gen_lines, gen_tree, ast.parse(result))
    except SyntaxError:
        return merged
    if assigns:
        names = set(_top_level_names(generated))
        placed = sorted((first, name) for name, (first, _) in _top_level_names(result).items()
                        if name in names)
        result = insert_before_definition(result, assigns, placed[0][1] if placed else "")
    return result if _parses(result) else merged


def _merge(original: str, generated: str, *, replace_whole: bool, methods: bool = False) -> str:
    new_names = _top_level_names(generated)
    old_names = _top_level_names(original)
    partial = _partial_classes(original, generated) if methods else {}
    if not old_names or (replace_whole and set(old_names) <= set(new_names) and not partial):
        return generated
    lines = original.split("\n")
    gen_lines = generated.split("\n")
    # replace from the bottom so earlier line numbers stay valid
    for name, (first, last) in sorted(old_names.items(), key=lambda item: -item[1][0]):
        if name in new_names:
            g_first, g_last = new_names[name]
            if name in partial:
                lines[first - 1:last] = partial[name]
            else:
                lines[first - 1:last] = gen_lines[g_first - 1:g_last]
    merged = "\n".join(lines)
    extra = [name for name in new_names if name not in old_names]
    for name in extra:
        g_first, g_last = new_names[name]
        merged = merged.rstrip("\n") + "\n\n\n" + "\n".join(gen_lines[g_first - 1:g_last]) + "\n"
    try:
        imports = [ast.unparse(node) for node in ast.parse(generated).body
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
    except SyntaxError:
        imports = []
    missing = [line for line in imports if line not in merged]
    if missing:
        merged = _insert_imports(merged, missing)
    if partial and not _parses(merged):  # retry with whole-class replacement (the old behaviour)
        return _merge(original, generated, replace_whole=replace_whole)
    if _parses(generated) and not _parses(merged):
        return generated  # a merge that does not parse is worse than the model's own file
    return merged


# -- method-level class merging (feature `large_projects`) -----------------------------------------
#
# A large file is shown to the writer in parts, so a change to one method comes back as
# `class Name:` holding only that method. Replacing the whole class with it would delete every
# other method, so a generated class that exists in the original and leaves out some of its
# methods is merged member by member:
#
# - a generated method (or nested class) replaces the original member of the same name, at the
#   original's position; the original's decorators go with it and the generated ones come in;
# - same-name groups: property accessors are matched by role (`@name.setter`, `@name.deleter`,
#   otherwise the getter/plain def), so a reply with only the setter replaces only the setter and
#   a new role goes right after the group; when either side has an `@overload` (or the reply
#   repeats a role), the generated group replaces the original group as a whole, at the position
#   of the first original;
# - new methods are appended at the end of the class body;
# - the class docstring and class attributes (`NAME = ...`, `name: T = ...`) stay as they are
#   unless the generated class contains them: a docstring replaces the docstring, an attribute
#   replaces the same-named attribute, a new attribute goes after the class's leading
#   docstring/attributes; other statements (`...`, `pass`) are placeholders and ignored;
# - the class header (decorators, bases) is kept from the original;
# - a generated member at another indentation (e.g. 2 spaces) is shifted to the original
#   members' indentation; its body keeps its own relative indentation (never re-scaled, which
#   could change multi-line strings).
#
# A generated class that redefines every original method keeps whole-class replacement. If the
# merged file does not parse, the merge is redone with whole-class replacement (the old rule).

_MEMBER = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _class_nodes(source: str) -> dict[str, ast.ClassDef]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _method_names(node: ast.ClassDef) -> set[str]:
    return {m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _partial_classes(original: str, generated: str) -> dict[str, list[str]]:
    """Class name -> its merged lines, for each generated class that exists in the original and
    leaves out some of the original's methods."""
    old_classes, new_classes = _class_nodes(original), _class_nodes(generated)
    old_names = _top_level_names(original)
    lines, gen_lines = original.split("\n"), generated.split("\n")
    merged = {}
    for name, new in new_classes.items():
        old = old_classes.get(name)
        if old is None or old_names.get(name) is None or _method_names(old) <= _method_names(new):
            continue
        merged[name] = _merge_class(old, new, lines, gen_lines, old_names[name][0])
    return merged


def _start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)])  # type: ignore[attr-defined]


def _role(node: ast.AST) -> str:
    for decorator in getattr(node, "decorator_list", []):
        text = ast.unparse(decorator)
        if text in {"overload", "typing.overload", "typing_extensions.overload"}:
            return "overload"
        if text.endswith(".setter"):
            return "setter"
        if text.endswith(".deleter"):
            return "deleter"
    return "def"


def _attribute_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0],
                                                                             ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _is_docstring(node: ast.stmt) -> bool:
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


def _merge_class(old: ast.ClassDef, new: ast.ClassDef, lines: list[str], gen_lines: list[str],
                 first: int) -> list[str]:
    """The original class (lines `first`..old.end_lineno) with the generated members merged in."""
    indent = " " * (old.body[0].col_offset if old.body else 4)

    def text(node: ast.AST) -> list[str]:
        block = gen_lines[_start(node) - 1:node.end_lineno]  # type: ignore[attr-defined]
        source = " " * node.col_offset  # type: ignore[attr-defined]
        if source == indent:
            return block
        return [indent + line[len(source):] if line.startswith(source) else line for line in block]

    # edits on original line numbers: (start, end, new lines); end == start - 1 is an insertion
    edits: list[tuple[int, int, list[str]]] = []
    end = old.end_lineno or old.lineno
    members: dict[str, list[ast.AST]] = {}
    for node in old.body:
        if isinstance(node, _MEMBER):
            members.setdefault(node.name, []).append(node)
    generated: dict[str, list[ast.AST]] = {}
    for node in new.body:
        if isinstance(node, _MEMBER):
            generated.setdefault(node.name, []).append(node)
    appended: list[str] = []
    for name, group in generated.items():
        originals = members.get(name, [])
        if not originals:
            appended += ["", *(line for node in group for line in text(node))]
            continue
        roles = [_role(node) for node in group]
        if "overload" in roles or any(_role(n) == "overload" for n in originals) \
                or len(set(roles)) < len(roles):
            whole = [line for node in group for line in _with_gaps(node, group, text)]
            edits.append((_start(originals[0]), originals[0].end_lineno, whole))  # type: ignore[attr-defined]
            edits += [(_start(n), n.end_lineno, []) for n in originals[1:]]  # type: ignore[attr-defined]
            continue
        by_role = {_role(node): node for node in originals}
        for node, role in zip(group, roles):
            if role in by_role:
                target = by_role[role]
                edits.append((_start(target), target.end_lineno, text(node)))  # type: ignore[attr-defined]
            else:
                after = max(n.end_lineno for n in originals)  # type: ignore[attr-defined]
                edits.append((after + 1, after, ["", *text(node)]))
    # docstring and class attributes, only when the generated class includes them
    head = [n for n in old.body if not isinstance(n, _MEMBER)]
    leading_end = _start(old.body[0]) - 1  # the end of the class header
    for node in old.body:
        if isinstance(node, _MEMBER):
            break
        leading_end = node.end_lineno or leading_end
    old_doc = old.body[0] if old.body and _is_docstring(old.body[0]) else None
    for node in new.body:
        if node is new.body[0] and _is_docstring(node):
            if old_doc is not None:
                edits.append((old_doc.lineno, old_doc.end_lineno, text(node)))  # type: ignore[arg-type]
            else:
                at = _start(old.body[0])
                edits.append((at, at - 1, [*text(node), ""]))
            continue
        name = _attribute_name(node)
        if name is None:
            continue  # `...`, `pass`, comments: placeholders for the omitted members
        same = next((n for n in head if _attribute_name(n) == name), None)
        if same is not None:
            edits.append((same.lineno, same.end_lineno, text(node)))  # type: ignore[arg-type]
        else:
            edits.append((leading_end + 1, leading_end, text(node)))
    if appended:
        edits.append((end + 1, end, appended))
    result = lines[first - 1:end]
    # apply from the bottom; at one line, replacements first, then insertions in reverse order
    # (each lands before the previous one, so they end up in the order they were made)
    order = sorted(range(len(edits)), key=lambda i: (edits[i][0], edits[i][1] >= edits[i][0], i),
                   reverse=True)
    for start, stop, block in (edits[i] for i in order):
        result[start - first:stop - first + 1] = block
    return result


def _with_gaps(node: ast.AST, group: list[ast.AST], text: Any) -> list[str]:
    """A member's lines, preceded by a blank line unless it is the first of its group."""
    return text(node) if node is group[0] else ["", *text(node)]


def _definitions(tree: ast.Module) -> dict[str, ast.AST]:
    """Functions and classes by qualified name (`Class.method`), nested ones included."""
    found: dict[str, ast.AST] = {}

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                found.setdefault(name, node)
                visit(node.body, name + ".")

    visit(tree.body, "")
    return found


def _docstring_lines(node: ast.AST, lines: list[str]) -> list[str] | None:
    body = getattr(node, "body", None)
    if not body or ast.get_docstring(node, clean=False) is None:
        return None
    return lines[body[0].lineno - 1:body[0].end_lineno]


def keep_docstrings(original: str, merged: str) -> str:
    """Put back the user's docstrings that the model's version of a definition (or of the whole
    file) left out: they often hold the examples that specify the behaviour (seen 2026-10-14 in
    debug mode). A docstring the model wrote itself is kept as written."""
    try:
        old_tree, new_tree = ast.parse(original), ast.parse(merged)
    except SyntaxError:
        return merged
    old_lines, lines = original.split("\n"), merged.split("\n")
    inserts: list[tuple[int, list[str]]] = []
    old_defs = _definitions(old_tree)
    for name, node in _definitions(new_tree).items():
        old_node = old_defs.get(name)
        doc = _docstring_lines(old_node, old_lines) if old_node is not None else None
        first = node.body[0]  # type: ignore[attr-defined]
        if doc is None or ast.get_docstring(node, clean=False) is not None \
                or first.lineno == node.lineno:  # type: ignore[attr-defined]
            continue  # nothing to restore, or a one-line `def f(): ...`
        old_indent = doc[0][:len(doc[0]) - len(doc[0].lstrip())]
        new_indent = " " * first.col_offset
        block = [new_indent + line[len(old_indent):] if line.startswith(old_indent) else line
                 for line in doc]
        inserts.append((first.lineno - 1, block + ([""] if isinstance(node, ast.ClassDef) else [])))
    module_doc = _docstring_lines(old_tree, old_lines)
    if module_doc is not None and ast.get_docstring(new_tree, clean=False) is None:
        position = new_tree.body[0].lineno - 1 if new_tree.body else len(lines)
        inserts.append((position, module_doc + ([""] if new_tree.body else [])))
    for position, block in sorted(inserts, key=lambda item: -item[0]):
        lines[position:position] = block
    restored = "\n".join(lines)
    return restored if _parses(restored) else merged


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def _insert_imports(source: str, imports: list[str]) -> str:
    """Add import lines after the module docstring and any `from __future__` imports (which
    must stay first); `__future__` imports themselves go right after the docstring."""
    future = [line for line in imports if line.startswith("from __future__")]
    others = [line for line in imports if not line.startswith("from __future__")]
    lines = source.split("\n")
    try:
        body = ast.parse(source).body
    except SyntaxError:
        return "\n".join(future + others) + "\n" + source
    position = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        position = body[0].end_lineno or 0
    after_docstring = position
    for node in body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            position = max(position, node.end_lineno or position)
    lines[after_docstring:after_docstring] = future
    position += len(future)
    lines[position:position] = others
    return "\n".join(lines)
