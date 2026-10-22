"""Acceptance: Vega checks the work against the task itself, in the sandbox (feature `acceptance`).

Before the first write Vega reads the task and the project, with rules and no model call, into an
acceptance spec:
- the functions and classes the task names, with the parameters it states (`clean_orders(path)`,
  `robot.grid.plan_moves(grid)`, "Keep the signature" → the signature the project has now);
- what the task says they return, where it is unambiguous ("returns a pandas DataFrame with exactly
  the columns `a`, `b`", "a dict with exactly these keys in order: `n` (int), ...");
- the input files it names that exist (the code is called on them when the call is unambiguous);
- the commands it names as the check (`mypy --strict`, `python check.py`, `cc -std=c11 ...`,
  `coverage run --branch`, `python -c "import pkg.mod"`);
- the project's own visible tests.

Vega then runs those checks in the sandbox (import, names and parameters with inspect.signature,
one call on the named input, the commands, the tests) plus a static gate on the files changed this
turn (undefined names F821 and redefinitions F811 that were not there before) and, with
`acceptance_stubs`, a static check that no task-named function is still a stub. With
`acceptance_prose_calls`, `name()` in prose is a name, not a zero-parameter signature. The results go to
the model as evidence; the model decides the fix. Vega never edits the code or overrules the answer.

Why (failures seen with Granite 4.1 8B, 2026-10-15/16): many failed attempts were the
verification and repair loop: agent-loop attempts almost never had visible tests, many tasks say
"Do not run commands", and code that crashes on its first call was finished as "verified".
Vega's own checks run in the sandbox even when the task says "Do not run commands" (the sandbox
never touches the user's machine); the feature `acceptance_ignores_no_commands` (default on) switches that off, leaving only static checks there.
"""

from __future__ import annotations

import ast
import builtins
import itertools
import json
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import check_feedback, data_preview, debuglog, features, sandbox, targeted_tests
from .tools import _project_files

if TYPE_CHECKING:
    from .agent import CodingAgent

FEEDBACK_ROUNDS = 3  # finish-gate feedback rounds per turn (one per distinct failing state)
MAX_PY_FILES = 400  # larger projects: names are still resolved, definitions are not scanned
MAX_SHOWN = 8  # failing items shown in one note
LINE_CHARS = 240
CALL_SECONDS = 20  # one smoke call, inside the sandbox run

# -- the spec ------------------------------------------------------------------------------------


@dataclass
class Param:
    name: str
    kind: str = "pos"  # pos | kwonly | star | dstar
    default: bool = False

    def text(self) -> str:
        star = {"star": "*", "dstar": "**"}.get(self.kind, "")
        return star + self.name + ("=…" if self.default else "")


@dataclass
class Returns:
    type: str  # DataFrame, dict, list, Figure, int, ...
    keys: list[str] = field(default_factory=list)  # dict keys / DataFrame columns
    ordered: bool = False
    key_types: dict[str, str] = field(default_factory=dict)  # key -> int|float|str|bool|list
    element_types: dict[str, str] = field(default_factory=dict)  # key -> type of list items

    def text(self) -> str:
        what = self.type + (f" with {'columns' if self.type == 'DataFrame' else 'keys'} "
                            + ", ".join(self.keys) if self.keys else "")
        return what + (" (in that order)" if self.ordered and self.keys else "")


@dataclass
class Required:
    """A function or class the task names: `attr` inside the module of `path`."""

    attr: str
    path: str  # project-relative .py file
    params: list[Param] | None = None  # as the task states them; None = not stated
    keep: list[Param] | None = None  # the signature the project has now, when the task says keep it
    returns: Returns | None = None
    call: list[dict[str, str]] | None = None  # arguments of a smoke call: {"kind", "value"}
    exists: bool = False  # defined in the project before the change

    @property
    def module(self) -> str:
        return module_name(self.path)

    @property
    def label(self) -> str:
        return f"{self.attr} ({self.path})"


@dataclass
class Spec:
    task: str
    names: list[Required] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)  # existing input files, in order of mention
    commands: list[str] = field(default_factory=list)
    # command -> its input file arguments the project does not have (command_absent_inputs)
    absent_inputs: dict[str, list[str]] = field(default_factory=dict)
    tests: str | None = None  # the project's own test command
    no_commands: bool = False  # the task says "Do not run commands"
    originals: dict[str, str | None] = field(default_factory=dict)  # file -> text before the turn
    suite_timed_out: int = 0  # targeted_tests: the whole suite did not finish in this many s

    @property
    def runs(self) -> bool:
        """Whether Vega's checks may execute code (else static checks only)."""
        return not self.no_commands or features.on("acceptance_ignores_no_commands")

    def empty(self) -> bool:
        return not (self.names or self.commands or self.tests)

    def remember(self, relative: str, before: str | None) -> None:
        """A file's text before its first write this turn (the static gate's baseline)."""
        self.originals.setdefault(relative, before)

    def summary(self) -> dict[str, Any]:
        return {"names": [r.label + (f"({', '.join(p.text() for p in r.params)})"
                                     if r.params is not None else "") for r in self.names],
                "calls": [r.attr for r in self.names if r.call],
                "returns": {r.attr: r.returns.text() for r in self.names if r.returns},
                "inputs": self.inputs, "commands": self.commands, "tests": self.tests,
                "no_commands": self.no_commands}


def module_name(path: str) -> str:
    module = path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
    return module.removeprefix("src.")


# -- extraction ----------------------------------------------------------------------------------

_BACKTICK = re.compile(r"`([^`\n]{1,300})`")
_CALL = re.compile(r"^(?:async\s+)?(?:def\s+)?([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\((.*)\)"
                   r"(?:\s*->\s*.+)?$", re.DOTALL)
_IDENT = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_PARAM = re.compile(r"^(\*{0,2})([A-Za-z_]\w*)\s*(?::\s*[^=]+?)?\s*(=\s*.+)?$")
# Words in a name's clause that make it something to write or keep, not just a mention.
_DEFINE_CUE = re.compile(
    r"\b(?:implement|implements|write|add|define|create|provide|finish|complete|rewrite|fix|make|"
    r"expose|export|keep|keeping|type|refactor|change)\b", re.IGNORECASE)
_KEEP_UNCHANGED = re.compile(r"\b(?:do not|don't|never|must not)\s+(?:change|modify|edit|touch|"
                             r"overwrite|rewrite|write to)\b[^.\n]*", re.IGNORECASE)
_DEFINE_RIGHT_BEFORE = re.compile(
    r"\b(?:implement|write|add|define|create|provide|finish|complete|rewrite|fix|keep|type|"
    r"refactor|expose|export)\s+(?:(?:the|a|an|new|async|public|helper|function|method|class|"
    r"coroutine)\s+)*$", re.IGNORECASE)
_FILE_SUFFIXES = {"py", "json", "csv", "md", "txt", "toml", "yaml", "yml", "cfg", "ini", "c", "h",
                  "js", "ts", "go", "rs", "sql", "png", "html", "jsonl", "parquet", "db"}
_TASK_CHANGES = re.compile(r"\b(?:implement|keep|keeping|rewrite|fix|finish|make|refactor|change|"
                           r"type|add|write|complete)\b", re.IGNORECASE)
_KIND_WORD = re.compile(r"\b(?:function|method|class|coroutine)\s*$", re.IGNORECASE)
_KIND_AFTER = re.compile(r"^\s*(?:function|method|class|coroutine)\b", re.IGNORECASE)
_NEGATED = re.compile(r"\b(?:never|not|don't|without|instead of|avoid|no)\b[^.;]*$", re.IGNORECASE)
_KEEP_SIGNATURE = re.compile(
    r"\bkeep(?:ing)?\s+(?:the|its|their|all|every|existing|same|public)?\s*(?:existing\s+)?"
    r"(?:function\s+|method\s+|class\s+)?signatures?\b|\bsignatures?\s+(?:unchanged|as (?:it is|they "
    r"are))|\bsame signatures?\b", re.IGNORECASE)
_NO_COMMANDS = re.compile(r"\b(?:do not|don't|never)\s+run\s+(?:any\s+)?(?:commands|shell commands"
                          r"|tests|the tests)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(`])|\n")
_CLAUSE_END = re.compile(r"[.;:!?](?:\s|$)|\n")
_PY_PATH = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w-]+\.py)\b")
_ANY_PATH = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:c|h|py|csv|tsv|jsonl|json|txt|xml"
                       r"|ya?ml|parquet|sql|db|sqlite|toml|ini|cfg|md))\b")
_INPUT_PATH = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:csv|tsv|jsonl|json|parquet|txt|xml"
                         r"|ya?ml|sql|db|sqlite))\b")
_DOTTED_PLAIN = re.compile(r"(?<![\w.`/])([a-z_]\w*(?:\.[A-Za-z_]\w*){1,4})(?![\w(/`])")

_RETURN_TYPES = ("DataFrame", "Series", "Figure", "ndarray", "dict", "list", "tuple", "set",
                 "int", "float", "str", "bool", "bytes", "Path")  # not None: "or None" is common
_RETURNS = re.compile(
    r"\breturn(?:s|ing)?\s+(?:a|an|the|one|its|exactly one)?\s*(?:new\s+|sorted\s+|plain\s+|"
    r"Python\s+|pandas\s+|matplotlib\s+|numpy\s+|single\s+|JSON-serializable\s+)*`?"
    r"(?:pd\.|np\.|matplotlib\.figure\.)?(" + "|".join(_RETURN_TYPES) + r")`?(?![\w-])",
    re.IGNORECASE)
_KEYS_INTRO = re.compile(r"\b(?:keys|columns)\b", re.IGNORECASE)
_EXACT = re.compile(r"\bexactly\b", re.IGNORECASE)
_ORDERED = re.compile(r"\bin (?:that|this|the|exactly this|exactly that) order\b|\bin order\b"
                      r"|\bordered\b", re.IGNORECASE)
_KEY_TYPE = re.compile(r"^\s*\(\s*(?:a |an )?(int|float|str|bool|list|dict|string|integer)\b"
                       r"(?:\s+of\s+[^)]*?\b(ints?|floats?|strs?|strings?|bools?)\b)?",
                       re.IGNORECASE)
_TYPE_WORDS = {"string": "str", "integer": "int", "ints": "int", "floats": "float", "strs": "str",
               "strings": "str", "bools": "bool"}
_PATH_PARAM = re.compile(r"(?:path|file|csv|json|filename|fname|src|source|input)s?$|^p$",
                         re.IGNORECASE)
_LIBRARY_ROOTS = {"np", "numpy", "pd", "pandas", "plt", "matplotlib", "json", "os", "sys", "re",
                  "math", "asyncio", "typing", "collections", "itertools", "functools", "datetime",
                  "pathlib", "scipy", "sklearn", "torch", "self", "cls", "subprocess", "time",
                  "random", "statistics", "decimal", "fractions", "logging", "argparse", "csv",
                  "sqlite3", "shutil", "tempfile", "io", "dataclasses", "enum", "abc", "string",
                  "hashlib", "uuid", "heapq", "bisect", "copy", "operator", "contextlib", "inspect",
                  "importlib", "unittest", "pytest", "seaborn", "sns", "joblib", "warnings", "df",
                  "fig", "ax", "axes", "requests", "http", "urllib", "socket", "threading",
                  "multiprocessing", "concurrent", "queue", "zoneinfo", "struct", "array", "e.g",
                  "i.e"}
_BUILTINS = set(dir(builtins))

# Commands: programs Vega runs as the task's check, and words that make a backticked command a check.
_COMMAND_PROGRAMS = {"python", "python3", "pytest", "mypy", "ruff", "pyright", "coverage", "cc",
                     "gcc", "clang", "go", "cargo", "make", "npm", "node", "swift", "npx"}
_COMMAND_CUE = re.compile(
    r"\b(?:pass(?:es)?|compile[sd]?|build|run|runs|running|test|tests|check|checks|measured|verify|"
    r"verified|succeed|succeeds|exit|exits|clean(?:ly)?|fails?)\b", re.IGNORECASE)
_COMMAND_NEGATED = re.compile(r"\b(?:never|do not|don't|not|without)\s+(?:\w+\s+){0,2}$",
                              re.IGNORECASE)
_COMMAND_PLACEHOLDER = re.compile(r"<[^>]*>|\.\.\.|…|\[[^\]]*\]|(?<![\w-])[A-Z][A-Z_]{2,}(?![\w])"
                                  r"|\|")


def _clause_before(text: str, start: int, span: int = 160) -> str:
    before = text[max(0, start - span):start]
    ends = list(_CLAUSE_END.finditer(before))
    return before[ends[-1].end():] if ends else before


def _sentence_around(text: str, start: int, end: int) -> tuple[int, int]:
    begin = 0
    for match in _SENTENCE_END.finditer(text, 0, start):
        begin = match.end()
    after = _SENTENCE_END.search(text, end)
    return begin, (after.start() if after else len(text))


def _prose_call(text: str, definitions: dict[str, list[_Definition]], path: str,
                attr: str) -> bool:
    """acceptance_prose_calls: `name()` in prose is how people name a function ("use it in
    `format_total()`", 2026-10-17: read as a zero-parameter requirement for a function that
    takes `amounts`). It states "no parameters" only as `def name()`, or when the project's
    definition of that name takes no required parameters already (`create_app()`)."""
    if not features.on("acceptance_prose_calls") or re.match(r"(?:async\s+)?def\s", text):
        return False
    defs = [d for d in definitions.get(attr, []) if d.path == path]
    if not defs:
        return True
    params = _params_of(defs[0].node)
    return params is not None and any(p.kind in ("pos", "kwonly") and not p.default
                                      for p in params)


def parse_params(text: str) -> list[Param] | None:
    """`a, b=None, *, c` → params; None for anything that is not a plain parameter list (example
    values like `f(1, 2)`, `f(...)`)."""
    text = text.strip()
    if not text:
        return []
    parts, depth, current = [], 0, ""
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    params: list[Param] = []
    keyword_only = False
    for raw in (part.strip() for part in parts):
        if raw == "*":
            keyword_only = True
            continue
        if raw == "/":
            continue
        match = _PARAM.match(raw)
        if not match or raw in ("...", "…"):
            return None
        stars, name, default = match.groups()
        if name in ("self", "cls") and not params and not stars:
            continue
        if stars:
            params.append(Param(name, "star" if stars == "*" else "dstar"))
            keyword_only = keyword_only or stars == "*"
            continue
        params.append(Param(name, "kwonly" if keyword_only else "pos", default is not None))
    return params


def _params_of(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[Param] | None:
    if isinstance(node, ast.ClassDef):
        init = next((n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == "__init__"), None)
        if init is None:
            return None
        node = init
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults = len(args.defaults)
    params = [Param(a.arg, "pos", i >= len(positional) - defaults)
              for i, a in enumerate(positional)]
    if params and params[0].name in ("self", "cls"):
        params = params[1:]
    if args.vararg:
        params.append(Param(args.vararg.arg, "star"))
    params += [Param(a.arg, "kwonly", d is not None)
               for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=False)]
    if args.kwarg:
        params.append(Param(args.kwarg.arg, "dstar"))
    return params


@dataclass
class _Definition:
    path: str
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _test_path(path: str) -> bool:
    return bool(re.search(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)", path))


def _definitions(project: Path) -> tuple[dict[str, list[_Definition]], set[str], set[str]]:
    """Top-level functions/classes (and Class.method) of the project's non-test Python files, the
    set of all its non-test .py paths, and the names its code imports from elsewhere (a task that
    says "use `RandomForestClassifier(...)`" names a library class, not one to write)."""
    found: dict[str, list[_Definition]] = {}
    paths: set[str] = set()
    imported: set[str] = set()
    count = 0
    for file in _project_files(project):
        if file.suffix != ".py":
            continue
        relative = file.relative_to(project).as_posix()
        if _test_path(relative) or relative.startswith(".vega"):
            continue
        paths.add(relative)
        count += 1
        if count > MAX_PY_FILES:
            continue
        try:
            tree = ast.parse(file.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root = node.module.split(".")[0]
                imported.update(f"{root}:{alias.asname or alias.name}" for alias in node.names)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.setdefault(node.name, []).append(_Definition(relative, node))
                if isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            found.setdefault(f"{node.name}.{child.name}", []).append(
                                _Definition(relative, child))
    local = {p.split("/")[0].removesuffix(".py") for p in paths}
    local |= {p.split("/")[1].removesuffix(".py") for p in paths if p.startswith("src/")}
    library = {entry.split(":", 1)[1] for entry in imported if entry.split(":", 1)[0] not in local}
    return found, paths, library - set(found)


def _module_file(project: Path, dotted: list[str], paths: set[str]) -> tuple[str, int] | None:
    """The longest prefix of `dotted` that is a module of the project: (its file, parts used)."""
    for cut in range(len(dotted), 0, -1):
        base = "/".join(dotted[:cut])
        for root in ("", "src/"):
            for candidate in (f"{root}{base}.py", f"{root}{base}/__init__.py"):
                if candidate in paths:
                    return candidate, cut
    return None


def _named_py_paths(task: str, paths: set[str]) -> list[str]:
    named = []
    for path in _PY_PATH.findall(task):
        for candidate in (path, f"src/{path}"):
            if candidate in paths and candidate not in named:
                named.append(candidate)
    return named


def _resolve(project: Path, dotted: str, task: str, start: int, end: int,
             definitions: dict[str, list[_Definition]], paths: set[str],
             named_files: list[str], call: bool = True) -> tuple[str, str] | None:
    """(file, attribute path) for a name the task gives, or None when it is not the project's.
    `call`: the task wrote it with parentheses."""
    parts = dotted.split(".")
    if parts[0] in _LIBRARY_ROOTS and not any(p.startswith(parts[0] + "/") for p in paths):
        return None
    if any(part.startswith("__") for part in parts):
        return None
    if len(parts) > 1:
        module = _module_file(project, parts, paths)
        if module is not None:
            path, cut = module
            # lower-case parts before the last one are modules still to be written
            # (`tabsplit.money.Money` → tabsplit/money.py)
            while len(parts) - cut > 1 and parts[cut][:1].islower():
                base = path.removesuffix("/__init__.py").removesuffix(".py")
                path, cut = f"{base}/{parts[cut]}.py", cut + 1
            if cut == len(parts):
                return None  # a module, not a name in it
            rest = ".".join(parts[cut:])
            if not call and not any(d.path == path for d in definitions.get(rest, [])):
                return None  # `pkg.thing` without parentheses: maybe a module
            return path, rest
        defined = definitions.get(".".join(parts[:2])) or (
            definitions.get(parts[0]) if parts[0][:1].isupper() else None)
        if defined and len(parts) == 2:
            return defined[0].path, dotted  # Class.method
        return None
    name = parts[0]
    if name in _BUILTINS or name in _LIBRARY_ROOTS or name.startswith(("test_", "__")):
        return None
    defined = [d.path for d in definitions.get(name, [])]
    if not defined and any(p.removeprefix("src/").split("/")[0].removesuffix(".py") == name
                           for p in paths):
        return None  # the package or module itself
    begin, finish = _sentence_around(task, start, end)
    sentence_files = _named_py_paths(task[begin:finish], paths)
    for path in sentence_files:
        if path in defined:
            return path, name
    if defined:
        preferred = [p for p in defined if p in named_files]
        return (preferred or defined)[0], name
    methods = [(key, d.path) for key, defs in definitions.items() if key.endswith("." + name)
               for d in defs]
    if len(methods) == 1:
        return methods[0][1], methods[0][0]  # the one class that has this method
    # Not defined yet: the file named after it in its sentence ("`f(x)` in `mod.py`"), else the
    # only Python file the task names.
    after = _PY_PATH.search(task, end, finish)
    if after:
        hits = _named_py_paths(after.group(1), paths)
        if hits:
            return hits[0], name
        candidate = after.group(1)
        if not _test_path(candidate):
            return candidate, name  # a module the task asks to create
    # A script the task only runs (`python check.py`) is never where a new name goes, and a
    # task about another language's files names its functions, not Python ones.
    if _OTHER_SOURCE.search(task):
        return None
    scripts = set(_RUN_SCRIPT.findall(task))
    sentence_files = [p for p in sentence_files if p not in scripts]
    if sentence_files:
        return sentence_files[0], name
    named_files = [p for p in named_files if p not in scripts]
    if len(named_files) == 1:
        return named_files[0], name
    return None


_RUN_SCRIPT = re.compile(r"\bpython3?\s+(?:-\S+\s+)*((?:[\w.-]+/)*[\w.-]+\.py)\b")
_OTHER_SOURCE = re.compile(
    r"\b[\w/.-]+\.(?:go|js|mjs|ts|tsx|java|kt|rs|c|h|cc|cpp|hpp|rb|swift|cs|php|scala|R)\b")


def _is_exception(name: str, definitions: dict[str, list[_Definition]]) -> bool:
    if name.endswith(("Error", "Exception", "Warning")):
        return True
    for definition in definitions.get(name, []):
        for base in getattr(definition.node, "bases", []):
            base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
            if base_name.endswith(("Error", "Exception", "Warning")):
                return True
    return False


def _returns_for(task: str, start: int, end: int, sole: bool = False) -> Returns | None:
    """What the task says the name at task[start:end] returns: from its own sentence, else the
    next sentence when it starts with "Return(s)", else (when it is the task's only function) the
    first later sentence that starts with "Return(s)". None when it does not say."""
    _, finish = _sentence_around(task, start, end)
    match = _RETURNS.search(task, end, finish)
    candidates = []
    for sentence in _SENTENCE_END.finditer(task, finish):
        opening = sentence.end()
        if re.match(r"(?:It\s+|This\s+|The function\s+)?[Rr]eturns?\b", task[opening:]):
            candidates.append(opening)
        if not sole or candidates:
            break  # only the next sentence, unless this is the task's only function
    for opening in ([] if match else candidates):
        after = _SENTENCE_END.search(task, opening + 1)
        match = _RETURNS.search(task, opening, after.start() if after else len(task))
        if match:
            break
    if not match:
        return None
    type_name = next(t for t in _RETURN_TYPES if t.lower() == match.group(1).lower())
    if type_name == "Path" and match.group(1) != "Path":
        return None
    returns = Returns(type_name)
    if type_name not in ("dict", "DataFrame"):
        return returns
    tail = task[match.end():match.end() + 1500]
    intro = _KEYS_INTRO.search(tail[:200])
    if not intro or not _EXACT.search(tail[:intro.end()]):
        return returns
    returns.ordered = bool(_ORDERED.search(tail[:intro.end() + 30]))
    position = intro.end()
    for key in _BACKTICK.finditer(tail, position):
        gap = tail[position:key.start()]
        if returns.keys and (re.search(r"[.;]\s+[A-Z(]|[.;]\s*$|\n", gap) or len(gap) > 260):
            break
        if not re.match(r"^[\w .-]+$", key.group(1)):
            break
        returns.keys.append(key.group(1))
        position = key.end()
        typed = _KEY_TYPE.match(tail[position:position + 80])
        if typed:
            kind = typed.group(1).lower()
            returns.key_types[key.group(1)] = _TYPE_WORDS.get(kind, kind)
            if typed.group(2):
                element = typed.group(2).lower()
                returns.element_types[key.group(1)] = _TYPE_WORDS.get(element, element)
    if len(returns.keys) < 2:
        returns.keys, returns.key_types, returns.element_types = [], {}, {}
    return returns


def _call_args(project: Path, required: Required, inputs: list[str], node: ast.AST | None
               ) -> list[dict[str, str]] | None:
    """Arguments for one smoke call on the task's input files, only when unambiguous: every
    required parameter takes a path (by its name, or by how the current code uses it) and there
    is exactly one input per such parameter; one parameter the current code uses as parsed data
    (subscripts it) takes a JSON input parsed; a `column` parameter after a CSV path takes the
    first numeric, non-id column of that CSV (any numeric column will do for a smoke call)."""
    params = required.params if required.params is not None else required.keep
    if params is None or not inputs:
        return None
    needed = [p for p in params if p.kind == "pos" and not p.default]
    if not needed or any(p.kind == "kwonly" and not p.default for p in params):
        return None
    usage = _param_usage(node) if node is not None else {}
    kinds = []
    for param in needed:
        if param.name.lower().startswith("out"):
            return None
        kind = usage.get(param.name) or ("path" if _PATH_PARAM.search(param.name) else None)
        if kind is None and _COLUMN_PARAM.match(param.name) and kinds == ["path"]:
            kind = "column"
        if kind is None:
            return None
        kinds.append(kind)
    files = [kind for kind in kinds if kind != "column"]
    if len(files) != len(inputs):
        return None
    args: list[dict[str, str]] = []
    remaining = iter(inputs)
    for kind in kinds:
        if kind == "column":
            column = _numeric_column(project / args[0]["value"])
            if column is None:
                return None
            args.append({"kind": "value", "value": column})
            continue
        path = next(remaining)
        if kind == "data" and not path.endswith(".json"):
            return None
        args.append({"kind": "json" if kind == "data" else "path", "value": path})
    return args


_COLUMN_PARAM = re.compile(r"^(?:col|column|column_name|colname|field|metric|value_column)$")


def _numeric_column(path: Path) -> str | None:
    import csv

    if path.suffix.lower() not in (".csv", ".tsv"):
        return None
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            rows = list(itertools.islice(csv.reader(handle, delimiter="\t" if path.suffix.lower()
                                                    == ".tsv" else ","), 21))
    except OSError:
        return None
    if len(rows) < 2:
        return None
    for index, name in enumerate(rows[0]):
        if re.search(r"(?:^|_)id$", name.strip().lower()):
            continue
        cells = [row[index].strip() for row in rows[1:] if index < len(row) and row[index].strip()]
        try:
            [float(cell) for cell in cells]
        except ValueError:
            continue
        if cells:
            return name
    return None


def _param_usage(node: ast.AST) -> dict[str, str]:
    """How a function's current body uses each parameter: 'path' (opened/read) or 'data'
    (subscripted, .items()/.get())."""
    usage: dict[str, str] = {}
    for child in ast.walk(node):
        if (isinstance(child, ast.Subscript) and isinstance(child.value, ast.Name)) or (
                isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name)
                and child.attr in ("items", "get", "keys", "values")):
            usage.setdefault(child.value.id, "data")
        elif isinstance(child, ast.Call):
            func = child.func
            callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if callee in ("open", "read_csv", "read_json", "Path", "read_text", "load", "loadtxt",
                          "read_parquet", "connect") and child.args \
                    and isinstance(child.args[0], ast.Name):
                usage[child.args[0].id] = "path"
    return usage


def _inputs(project: Path, task: str) -> list[str]:
    from .pipeline import is_data_input

    found: list[str] = []
    # "Do not change `data/x.csv`" protects an input; it does not make it a file to write
    task = _KEEP_UNCHANGED.sub("", task)
    for path in _INPUT_PATH.findall(task):
        target = project / path
        if path in found or not target.is_file():
            continue
        try:
            inside = target.resolve().is_relative_to(project.resolve())
        except OSError:
            inside = False
        if inside and data_preview.is_data_path(path) and is_data_input(task, path):
            found.append(path)
    return found


def _command(project: Path, raw: str, task: str, start: int, paths: set[str]) -> str | None:
    """A backticked command the task names as a check, made runnable: `cc` flags without sources
    get the C files the task names (-fsyntax-only), `coverage run` without a target runs pytest."""
    command = " ".join(raw.split())
    try:
        argv = shlex.split(command)
    except ValueError:
        return None
    if not argv or argv[0] not in _COMMAND_PROGRAMS or len(argv) < 2:
        return None
    if _COMMAND_PLACEHOLDER.search(command):
        return None
    clause = _clause_before(task, start, 120)
    begin, finish = _sentence_around(task, start, start + len(raw))
    sentence = task[begin:finish]
    if _COMMAND_NEGATED.search(clause) or not _COMMAND_CUE.search(sentence):
        return None
    if argv[0] in ("python", "python3"):
        if argv[1] == "-m":
            if len(argv) < 3:
                return None
            module = argv[2]
            if module not in ("pytest", "unittest", "mypy") and _module_file(
                    project, module.split("."), paths) is None:
                return None
        elif argv[1] == "-c":
            pass
        elif not (project / argv[1]).is_file():
            return None
    if argv[0] in ("cc", "gcc", "clang") and not any(a.endswith((".c", ".h")) for a in argv[1:]):
        sources = [p for p in dict.fromkeys(_ANY_PATH.findall(task))
                   if p.endswith(".c") and (project / p).is_file() and not _test_path(p)]
        headers = [p for p in dict.fromkeys(_ANY_PATH.findall(task))
                   if p.endswith(".h") and (project / p).is_file()]
        files = sources or headers
        if not files:
            return None
        command = f"{command} -fsyntax-only " + " ".join(shlex.quote(f) for f in files)
    if argv[0] == "coverage":
        if argv[1] != "run":
            return None
        if len(argv) == 2 or all(a.startswith("-") for a in argv[2:]):
            command += " -m pytest -q"
    if argv[0] == "mypy" and all(a.startswith("-") for a in argv[1:]):
        targets = [p for p in dict.fromkeys(_PY_PATH.findall(task)) if (project / p).is_file()
                   and not _test_path(p)]
        if not targets:
            return None
        command += " " + " ".join(shlex.quote(t) for t in targets)
    if isinstance(sandbox.parse_command(command, project, harness=True), str):
        return None
    return command


_BARE_MODULE = re.compile(r"^python3? -m ([\w.]+)$")


def _full_invocation(project: Path, task: str, bare: str) -> str:
    """task_module_commands: a bare `python -m pkg` ("make `python -m pkg` work") names the
    program; run with no arguments it only shows argparse's usage (2026-10-21: a false failure in
    every round). The first full invocation the task or its README
    writes out that runs here without a shell is checked instead; with none, the bare name stays
    (and `_command_item` does not count its usage exit as a failure)."""
    from . import task_checks

    match = _BARE_MODULE.match(bare)
    assert match is not None
    for example in task_checks.full_invocations(project, task, match.group(1)):
        if not isinstance(sandbox.parse_command(example, project, harness=True), str):
            return example
    return bare


_SHELL_WORDS = {"<", ">", ">>", "|", "&&", "||", ";", "2>", "2>&1"}
_OUTPUT_FLAGS = {"-o", "--out", "--output", "--output-file", "--dest", "--to"}
_PATH_LIKE = re.compile(r"(?:[\w.-]+/)*[\w-][\w.-]*\.[A-Za-z]\w{0,5}")


def absent_inputs(project: Path, task: str, command: str) -> list[str]:
    """The input file arguments of a `python -m pkg ...` / `python script.py ...` command that the
    project does not have and the task never mentions outside its backticked commands: an example
    name (`python -m pkg count notes.txt` in a bug report), not a file to create. A file
    after another file argument or an output flag (`data/in.csv out.json`, `-o out.json`) may be
    one the program writes, so it is left out. Empty for other programs."""
    try:
        argv = shlex.split(command)
    except ValueError:
        return []
    if len(argv) < 2 or argv[0] not in ("python", "python3") or argv[1] == "-c":
        return []
    rest = argv[3:] if argv[1] == "-m" else argv[2:]
    prose = _BACKTICK.sub(" ", task)
    found: list[str] = []
    after_file = after_output_flag = False
    for word in rest:
        if word in _SHELL_WORDS:
            break
        output_flag, after_output_flag = after_output_flag, word in _OUTPUT_FLAGS
        if word.startswith("-"):
            flag, _, word = word.partition("=")
            output_flag = flag in _OUTPUT_FLAGS
        # `prog data/in.csv out.json`, `-o out.json`: a later file may be one the program writes
        later, after_file = after_file, after_file or bool(_PATH_LIKE.fullmatch(word))
        if not word or not _INPUT_PATH.fullmatch(word) or word in found or later or output_flag:
            continue
        if (project / word).exists():
            continue
        if re.search(rf"(?<![\w/.-]){re.escape(word)}(?![\w/-])", prose):
            continue  # the task names it in prose: maybe a file the work should add
        found.append(word)
    return found


_MISSING_FILE = re.compile(r"no such file|not found|does not exist|doesn't exist|cannot|can't|"
                           r"could not|couldn't|unable|missing|FileNotFoundError|Errno 2",
                           re.IGNORECASE)


def _reports_absent(output: str, project: Path, absent: list[str]) -> str:
    """The absent input a failed command's output reports as missing ('' when none): the program
    got as far as opening it, which is all this command can show without the file."""
    lines = output.splitlines()[1:]  # past run_check's "exit code: N" line
    for name in absent:
        if (project / name).exists():
            continue
        if any(name in line and _MISSING_FILE.search(line) for line in lines):
            return name
    return ""


def extract(project: Path, task: str) -> Spec:
    """The acceptance spec of a task, by rules (no model call)."""
    project = project.resolve()
    spec = Spec(task=task, no_commands=bool(_NO_COMMANDS.search(task)))
    definitions, paths, library = _definitions(project)
    named_files = _named_py_paths(task, paths)
    keep = bool(_KEEP_SIGNATURE.search(task))
    spec.inputs = _inputs(project, task)
    seen: set[tuple[str, str]] = set()

    spans: dict[tuple[str, str], tuple[int, int]] = {}
    classes: dict[int, tuple[str, str]] = {}  # sentence start -> the last class named in it

    def add(path: str, attr: str, params: list[Param] | None, start: int, end: int) -> None:
        key = (path, attr)
        defs = [d for d in definitions.get(attr, []) if d.path == path]
        if key in seen:
            existing = next(r for r in spec.names if (r.path, r.attr) == key)
            if existing.params is None and params is not None:
                existing.params = params
                spans[key] = (start, end)
            return
        seen.add(key)
        required = Required(attr, path, params, exists=bool(defs))
        if keep and defs:
            required.keep = _params_of(defs[0].node)
        spans[key] = (start, end)
        spec.names.append(required)

    for match in _BACKTICK.finditer(task):
        text = match.group(1).strip()
        call = _CALL.match(text)
        if call:
            dotted, params = call.group(1), parse_params(call.group(2))
        elif _IDENT.match(text):
            dotted, params = text, None
        else:
            continue
        before = _clause_before(task, match.start())
        after = task[match.end():match.end() + 30]
        if _NEGATED.search(before[-60:]) or dotted.split(".")[-1] in _FILE_SUFFIXES:
            continue
        if "." not in dotted:
            if dotted in library:
                continue
            # one plain name the project does not define: a call-like `f(x)` after a writing
            # word in its clause, or a bare `f` right after one or called a function/class; a
            # name the project defines counts when the task is to change code at all
            if definitions.get(dotted):
                if not _TASK_CHANGES.search(task):
                    continue
            elif not (_KIND_WORD.search(before) or _KIND_AFTER.match(after)
                      or _DEFINE_RIGHT_BEFORE.search(before)
                      or (call and _DEFINE_CUE.search(before))):
                continue
        resolved = _resolve(project, dotted, task, match.start(), match.end(), definitions, paths,
                            named_files, call=bool(call))
        if resolved is None:
            continue
        path, attr = resolved
        if not path.endswith(".py") or _test_path(path) or attr == attr.upper():
            continue  # CONSTANTS are not checked
        sentence = _sentence_around(task, match.start(), match.end())[0]
        if attr[:1].isupper() and "." not in attr:
            if not _is_exception(attr, definitions):
                classes[sentence] = (path, attr)
        elif "." not in attr and not definitions.get(attr) and sentence in classes \
                and classes[sentence][0] == path:
            attr = f"{classes[sentence][1]}.{attr}"  # "keep `Pool(...)` ... add `close()`"
        if params == [] and _prose_call(text, definitions, path, attr):
            params = None  # "calls `format_total()`" names the function, not its signature
        add(path, attr, params, match.start(), match.end())
    # dotted names in plain text whose first part is a project package ("quillbook.posts.Post")
    for match in _DOTTED_PLAIN.finditer(task):
        dotted = match.group(1).rstrip(".")
        parts = dotted.split(".")
        if parts[0] in _LIBRARY_ROOTS or parts[-1] in _FILE_SUFFIXES:
            continue
        module = _module_file(project, parts, paths)
        if module is None or module[1] == len(parts) or len(parts) - module[1] != 1:
            continue
        if any(d.path == module[0] for d in definitions.get(parts[-1], [])):
            add(module[0], parts[-1], None, match.start(), match.end())  # kept public API
    functions = [r for r in spec.names if r.attr.split(".")[-1][:1].islower()]
    for required in functions:
        start, end = spans[(required.path, required.attr)]
        required.returns = _returns_for(task, start, end, sole=functions == [required])
    nodes = {(d.path, name): d.node for name, defs in definitions.items() for d in defs}
    for required in spec.names:
        required.call = _call_args(project, required, spec.inputs,
                                   nodes.get((required.path, required.attr)))
    for match in _BACKTICK.finditer(task):
        command = _command(project, match.group(1), task, match.start(), paths)
        if command and _BARE_MODULE.match(command) and features.on("task_module_commands"):
            command = _full_invocation(project, task, command)
        if command and command not in spec.commands:
            spec.commands.append(command)
            absent = absent_inputs(project, task, command)
            if absent:
                spec.absent_inputs[command] = absent
    tests = check_feedback.project_command(project)
    if tests and not any(check_feedback.covers(c, tests) for c in spec.commands):
        spec.tests = tests
    return spec


# -- model fill (opt-in acceptance_model) --------------------------------------------------------

MODEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "names": {"type": "array", "maxItems": 6, "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "file": {"type": "string"},
                           "params": {"type": "string"}},
            "required": ["name", "file"]}},
        "commands": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
    },
    "required": ["names", "commands"],
}
MODEL_SYSTEM = ("List what the request requires to exist when the work is done: the Python "
                "functions or classes it names (name, the .py file, the parameters it states as "
                "'a, b=None'), and the shell commands it says must pass. Only what the request "
                "states; empty lists if it states none. Reply with the JSON object only.")
MODEL_MAX_TOKENS = 300


def fill_with_model(agent: CodingAgent, spec: Spec) -> None:
    """acceptance_model: one short constrained-JSON call when the rules found no names or
    commands. What it lists is kept only if it resolves in the project like a rule-found item."""
    if spec.names or spec.commands:
        return
    project = agent.project
    files = [p.relative_to(project).as_posix() for p in _project_files(project, limit=60)]
    sampling = {**agent.sampling, "max_tokens": min(MODEL_MAX_TOKENS,
                                                     agent.sampling.get("max_tokens", 10**6))}
    try:
        reply = agent.client.chat.completions.create(
            model=agent.model, **sampling,
            messages=[{"role": "system", "content": MODEL_SYSTEM},
                      {"role": "user", "content": f"Request: {spec.task}\n\nProject files:\n"
                                                  + "\n".join(files)}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "acceptance", "schema": MODEL_SCHEMA}},
        )
        data = json.loads(reply.choices[0].message.content or "")
    except Exception as exc:  # noqa: BLE001 - the rules' spec stands
        debuglog.record("stage", stage="acceptance.model_failed", error=str(exc)[:200])
        return
    if not isinstance(data, dict):
        return
    _, paths, _ = _definitions(project)
    for item in data.get("names") or []:
        if not isinstance(item, dict):
            continue
        name, path = str(item.get("name", "")).strip(), str(item.get("file", "")).strip()
        if not _IDENT.match(name) or not path.endswith(".py") or _test_path(path):
            continue
        if path not in paths and not (project / path).parent.is_dir():
            continue
        params = parse_params(str(item["params"])) if item.get("params") else None
        if all((r.path, r.attr) != (path, name) for r in spec.names):
            spec.names.append(Required(name, path, params))
    for raw in data.get("commands") or []:
        command = str(raw).strip()
        if command and not isinstance(sandbox.parse_command(command, project, harness=True), str) \
                and command.split()[0] in _COMMAND_PROGRAMS and command not in spec.commands:
            spec.commands.append(command)


def ensure(agent: Any, task: str) -> Spec | None:
    """The agent's spec for this turn, building it when a pipeline runs without agent.run (which
    builds it before any pipeline); None when the feature is off."""
    spec = getattr(agent, "_acceptance", None)
    if spec is None and features.on("acceptance") and getattr(agent, "can_write", False):
        spec = for_turn(agent, task)
        agent._acceptance = spec
        if not isinstance(getattr(agent, "_acceptance_changed", None), list):
            agent._acceptance_changed = []
    return spec


def for_turn(agent: CodingAgent, task: str) -> Spec | None:
    """The turn's spec (None when the feature is off); recorded in the debug log."""
    if not features.on("acceptance") or not agent.can_write:
        return None
    try:
        spec = extract(agent.project, task)
        if features.on("acceptance_model"):
            fill_with_model(agent, spec)
    except Exception as exc:  # noqa: BLE001 - no spec is the old behaviour
        debuglog.record("stage", stage="acceptance.extract_failed", error=str(exc)[:200])
        return None
    debuglog.record("stage", stage="acceptance.spec", **spec.summary())
    return spec


# -- running the checks --------------------------------------------------------------------------


@dataclass
class Item:
    kind: str  # import | name | signature | call | returns | stub | command | tests | static | run
    label: str
    ok: bool
    detail: str = ""
    output: str = ""  # a failing command's whole result (for claim_check / condensing)
    note: bool = False  # a passing item the model should still read (targeted_tests: slow suite)

    def line(self) -> str:
        text = f"{self.label}: {self.detail}" if self.detail else self.label
        text = " ".join(text.split()) if self.kind not in ("command", "tests") else text
        return text if len(text) <= LINE_CHARS else text[:LINE_CHARS - 1] + "…"


@dataclass
class Result:
    items: list[Item] = field(default_factory=list)

    @property
    def failing(self) -> list[Item]:
        return [item for item in self.items if not item.ok]

    @property
    def ok(self) -> bool:
        return not self.failing

    def state(self) -> tuple[str, ...]:
        """The failing state: the same failures again are not new evidence."""
        return tuple(sorted(f"{i.kind}:{i.label}:{i.detail.splitlines()[0] if i.detail else ''}"
                            for i in self.failing))

    def as_run(self) -> str:
        """The result in run_check's shape (`exit code: N`), for claim_check's evidence."""
        notes = "".join(f"\nNOTE {item.line()}" for item in self.items if item.ok and item.note)
        if self.ok:
            return f"exit code: 0\nVega acceptance: {len(self.items)} check(s) pass." + notes
        return "exit code: 1\n" + "\n".join(f"FAIL {item.line()}" for item in self.failing) + notes


CHECK_SCRIPT = r'''
import importlib, inspect, json, os, signal, sys, traceback
sys.path[:0] = ["src", "."]
os.environ.setdefault("MPLBACKEND", "Agg")
CWD = os.getcwd()
items = json.loads(sys.argv[1])
SECONDS = int(sys.argv[2])
UNINSTALLED = len(sys.argv) > 3 and sys.argv[3] == "1"
out = []

def not_ours(exc):
    """A ModuleNotFoundError for a module the project does not hold (a third-party package
    this environment lacks), not a project module that is missing."""
    name = getattr(exc, "name", None) if isinstance(exc, ModuleNotFoundError) else None
    if not name:
        return False
    top = name.split(".")[0]
    return not any(os.path.exists(os.path.join(root, top)) or
                   os.path.exists(os.path.join(root, top + ".py")) for root in ("src", "."))

def where(exc):
    frames = [f for f in traceback.extract_tb(exc.__traceback__)
              if f.filename.startswith(CWD) and "site-packages" not in f.filename]
    if not frames:
        return ""
    f = frames[-1]
    return " (%s:%d: %s)" % (os.path.relpath(f.filename, CWD), f.lineno, (f.line or "").strip()[:100])

def error(exc):
    text = str(exc).replace(os.path.realpath(CWD), ".").replace(CWD, ".")
    text = text.splitlines()[0][:160] if text else ""
    return "%s%s%s" % (type(exc).__name__, ": " + text if text else "", where(exc))

def params_of(obj):
    sig = inspect.signature(obj)
    params = list(sig.parameters.values())
    if params and params[0].name in ("self", "cls") and inspect.isfunction(obj):
        params = params[1:]
    return params

def show(params):
    parts = []
    for p in params:
        star = "*" if p.kind is p.VAR_POSITIONAL else "**" if p.kind is p.VAR_KEYWORD else ""
        parts.append(star + p.name + ("=…" if p.default is not p.empty else ""))
    return "(" + ", ".join(parts) + ")"

def compare(params, stated, strict_names):
    problems = []
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    names = {p.name: p for p in params}
    var_kw = any(p.kind is p.VAR_KEYWORD for p in params)
    var_pos = any(p.kind is p.VAR_POSITIONAL for p in params)
    index = 0
    covered = set()
    for s in stated:
        if s["kind"] == "star":
            if not var_pos:
                problems.append("no *%s" % s["name"])
            continue
        if s["kind"] == "dstar":
            if not var_kw:
                problems.append("no **%s" % s["name"])
            continue
        if s["kind"] == "kwonly" or (s["default"] and s["name"] in names):
            p = names.get(s["name"])
            covered.add(s["name"])
            if p is None and not var_kw:
                problems.append("no parameter `%s`" % s["name"])
            elif p is not None and s["default"] and p.default is p.empty:
                problems.append("`%s` has no default" % s["name"])
            elif p is not None and s["kind"] == "kwonly" and p.kind is not p.KEYWORD_ONLY and strict_names:
                problems.append("`%s` is not keyword-only" % s["name"])
            continue
        if index >= len(positional):
            if not var_pos:
                problems.append("no parameter for `%s`" % s["name"])
            continue
        p = positional[index]
        index += 1
        covered.add(p.name)
        if p.name != s["name"]:
            problems.append("parameter %d is `%s`, not `%s`" % (index, p.name, s["name"]))
        if s["default"] and p.default is p.empty:
            problems.append("`%s` has no default" % p.name)
    for p in params:
        if p.name not in covered and p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            problems.append("also requires `%s`" % p.name)
    return problems

def describe(value):
    name = type(value).__name__
    if hasattr(value, "shape") and hasattr(value, "columns"):
        return "%s %dx%d" % (name, value.shape[0], value.shape[1])
    if isinstance(value, dict):
        return "dict with %d key(s)" % len(value)
    if isinstance(value, (list, tuple, set)):
        return "%s of %d" % (name, len(value))
    return name

PY = {"int": int, "float": float, "str": str, "bool": bool, "list": list, "dict": dict}

def exact(value, kind):
    if kind == "int":
        return type(value) is int
    if kind == "bool":
        return type(value) is bool
    if kind == "float":
        return isinstance(value, float)
    return isinstance(value, PY[kind])

def check_returns(value, r):
    problems = []
    mro = {c.__name__ for c in type(value).__mro__}
    t = r["type"]
    if t == "None":
        if value is not None:
            problems.append("returned %s, the task says it returns None" % describe(value))
        return problems
    if t not in mro and not (t == "Path" and "PurePath" in mro):
        problems.append("returned %s, the task states %s" % (describe(value), t))
        return problems
    if r["keys"]:
        actual = [str(k) for k in (value.columns if t == "DataFrame" else value.keys())]
        stated = r["keys"]
        what = "columns" if t == "DataFrame" else "keys"
        if r["ordered"] or set(actual) != set(stated):
            if actual != stated and (r["ordered"] or set(actual) != set(stated)):
                missing = [k for k in stated if k not in actual]
                extra = [k for k in actual if k not in stated]
                bits = []
                if missing:
                    bits.append("missing " + ", ".join(missing))
                if extra:
                    bits.append("extra " + ", ".join(extra[:6]))
                if not bits:
                    bits.append("order " + ", ".join(actual))
                problems.append("%s differ from the task's (%s)" % (what, "; ".join(bits)))
        if t == "dict":
            for key, kind in r["key_types"].items():
                if key in value and kind in PY and not exact(value[key], kind):
                    v = value[key]
                    problems.append("`%s` is %s.%s, the task states %s" % (
                        key, type(v).__module__, type(v).__name__, kind))
            for key, kind in r["element_types"].items():
                v = value.get(key)
                if isinstance(v, (list, tuple)) and kind in PY:
                    bad = [x for x in v if not exact(x, kind)]
                    if bad:
                        problems.append("`%s` holds %s.%s values, the task states %ss" % (
                            key, type(bad[0]).__module__, type(bad[0]).__name__, kind))
    return problems

class Timeout(Exception):
    pass

def alarm(signum, frame):
    raise Timeout("no result after %d s" % SECONDS)

signal.signal(signal.SIGALRM, alarm)
modules = {}
for item in items:
    label = item["label"]
    module = item["module"]
    if module not in modules:
        try:
            signal.alarm(SECONDS)
            modules[module] = importlib.import_module(module)
            signal.alarm(0)
            out.append({"kind": "import", "label": "import " + module, "ok": True, "detail": ""})
        except BaseException as exc:
            signal.alarm(0)
            modules[module] = None
            if UNINSTALLED and not_ours(exc):
                out.append({"kind": "import", "label": "import " + module, "ok": True,
                            "detail": "not checkable here (module not installed: %s)" % exc.name})
                continue
            out.append({"kind": "import", "label": "import " + module, "ok": False, "detail": error(exc)})
    obj = modules[module]
    if obj is None:
        continue
    try:
        for part in item["attr"].split("."):
            obj = getattr(obj, part)
    except AttributeError:
        out.append({"kind": "name", "label": label, "ok": False,
                    "detail": "not defined (the task names it)"})
        continue
    out.append({"kind": "name", "label": label, "ok": True, "detail": ""})
    params = None
    try:
        params = params_of(obj)
    except (TypeError, ValueError):
        pass
    for key, strict in (("params", True), ("keep", True)):
        stated = item.get(key)
        if stated is None or params is None:
            continue
        problems = compare(params, stated, strict)
        wanted = "(" + ", ".join(("*" if s["kind"] == "star" else "**" if s["kind"] == "dstar" else "") + s["name"] + ("=…" if s["default"] else "") for s in stated) + ")"
        source = "the task states" if key == "params" else "before the change (keep it)"
        out.append({"kind": "signature", "label": label, "ok": not problems,
                    "detail": ("parameters are %s; %s %s: %s" % (show(params), source, wanted, "; ".join(problems))) if problems else ""})
    call = item.get("call")
    if call is None or not callable(obj):
        continue
    args = []
    for arg in call:
        if arg["kind"] == "json":
            with open(arg["value"], encoding="utf-8") as handle:
                args.append(json.load(handle))
        else:
            args.append(arg["value"])
    shown = "%s(%s)" % (item["attr"], ", ".join("<%s parsed>" % a["value"] if a["kind"] == "json" else repr(a["value"]) for a in call))
    try:
        signal.alarm(SECONDS)
        value = obj(*args)
        if inspect.isawaitable(value):
            import asyncio
            value = asyncio.run(value)
        signal.alarm(0)
    except BaseException as exc:
        signal.alarm(0)
        out.append({"kind": "call", "label": shown, "ok": False, "detail": "raised " + error(exc)})
        continue
    returns = item.get("returns")
    problems = check_returns(value, returns) if returns else []
    out.append({"kind": "call", "label": shown, "ok": not problems,
                "detail": ("; ".join(problems)) if problems else "returned " + describe(value)})
print("VEGA-ACCEPTANCE " + json.dumps(out))
'''


def _payload(spec: Spec, calls: bool) -> list[dict[str, Any]]:
    def params(values: list[Param] | None) -> list[dict[str, Any]] | None:
        if values is None:
            return None
        return [{"name": p.name, "kind": p.kind, "default": p.default} for p in values]

    payload = []
    for required in spec.names:
        payload.append({
            "label": f"`{required.attr}` in {required.path}", "module": required.module,
            "attr": required.attr, "params": params(required.params),
            "keep": params(required.keep) if required.params is None else None,
            "call": required.call if calls else None,
            "returns": (None if required.returns is None else {
                "type": required.returns.type, "keys": required.returns.keys,
                "ordered": required.returns.ordered, "key_types": required.returns.key_types,
                "element_types": required.returns.element_types}),
        })
    return payload


def _run_names(spec: Spec, project: Path, stop: threading.Event | None,
               calls: bool) -> list[Item]:
    payload = _payload(spec, calls)
    if not payload:
        return []
    command = (f"python -c {shlex.quote(CHECK_SCRIPT)} {shlex.quote(json.dumps(payload))} "
               f"{CALL_SECONDS} {int(features.on('acceptance_uninstalled'))}")
    output = sandbox.run_check(command, project, stop=stop, harness=True)
    marker = output.rfind("VEGA-ACCEPTANCE ")
    if marker >= 0:
        try:
            line = output[marker + len("VEGA-ACCEPTANCE "):].splitlines()[0]
            return [Item(**entry) for entry in json.loads(line)]
        except (ValueError, TypeError, IndexError):
            pass
    if check_feedback.exit_code(output) is None and not output.startswith("Check timed out"):
        return []  # could not run here (no sandbox): no evidence either way
    tail = output.partition("\n")[2].strip().splitlines()[-3:] or [output.splitlines()[0]]
    return [Item("run", "Vega's import/call checks", False, " / ".join(tail))]


def _static_names(spec: Spec, project: Path) -> list[Item]:
    """Without running anything: is each required name defined in its file (ast)?"""
    items = []
    for required in spec.names:
        target = project / required.path
        try:
            tree = ast.parse(target.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            items.append(Item("name", f"`{required.attr}` in {required.path}", False,
                              "the file is missing or does not parse"))
            continue
        found = _find(tree, required.attr.split("."))
        items.append(Item("name", f"`{required.attr}` in {required.path}", found is not None,
                          "" if found is not None else "not defined (the task names it)"))
        stated = required.params if required.params is not None else required.keep
        if found is not None and stated is not None and not isinstance(found, ast.ClassDef):
            actual = _params_of(found) or []
            problems = _compare_static(actual, stated)
            source = "the task states" if required.params is not None else "before the change"
            items.append(Item("signature", f"`{required.attr}` in {required.path}", not problems,
                              (f"parameters are ({', '.join(p.text() for p in actual)}); {source} "
                               f"({', '.join(p.text() for p in stated)})") if problems else ""))
    return items


_TODO = re.compile(r"#.*\b(?:TODO|FIXME|XXX|implement)\b", re.IGNORECASE)


def _is_not_implemented(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Raise) or node.exc is None:
        return False
    exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
    return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"


def stub_reason(node: ast.AST, lines: list[str]) -> str | None:
    """Why a function is still a stub: its body (after the docstring) raises NotImplementedError
    at the top level, or is only `...`/`pass` with a TODO comment in it; None otherwise."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    if any(isinstance(d, ast.Name) and d.id == "abstractmethod"
           or isinstance(d, ast.Attribute) and d.attr == "abstractmethod"
           for d in node.decorator_list):
        return None
    body = node.body[1:] if _is_docstring(node.body[0]) else node.body
    if any(_is_not_implemented(statement) for statement in body):
        return "still a stub: it raises NotImplementedError"
    placeholder = all(isinstance(s, ast.Pass) or isinstance(s, ast.Expr)
                      and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis
                      for s in body)
    source = lines[node.lineno - 1:(node.end_lineno or node.lineno)]
    if body and placeholder and any(_TODO.search(line) for line in source):
        return "still a stub: its body is only a placeholder with a TODO"
    return None


def _is_docstring(node: ast.stmt) -> bool:
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


def _overridden(tree: ast.Module, parts: list[str]) -> bool:
    """A base-class method that a subclass in the same file overrides (an abstract pattern)."""
    if len(parts) != 2:
        return False
    base, method = parts
    return any(isinstance(node, ast.ClassDef)
               and any(isinstance(b, ast.Name) and b.id == base for b in node.bases)
               and any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method
                       for n in node.body)
               for node in tree.body)


def _stub_items(spec: Spec, project: Path) -> list[Item]:
    """acceptance_stubs (2026-10-17): acceptance passed a task-named function that
    still raised NotImplementedError. Static (ast), so it also holds where nothing may run."""
    items = []
    for required in spec.names:
        try:
            text = (project / required.path).read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (OSError, SyntaxError, ValueError):
            continue  # the name and import checks report a missing or broken file
        parts = required.attr.split(".")
        found = _find(tree, parts)
        if found is None or _overridden(tree, parts):
            continue
        reason = stub_reason(found, text.splitlines())
        if reason:
            items.append(Item("stub", f"`{required.attr}` in {required.path}", False, reason))
    return items


def _find(tree: ast.Module, parts: list[str]) -> ast.AST | None:
    body: list[ast.stmt] = tree.body
    node: ast.AST | None = None
    for part in parts:
        node = next((n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                       ast.ClassDef)) and n.name == part), None)
        if node is None:
            return None
        body = node.body if isinstance(node, ast.ClassDef) else []
    return node


def _compare_static(actual: list[Param], stated: list[Param]) -> bool:
    plain = [p for p in actual if p.kind in ("pos", "kwonly")]
    wanted = [p for p in stated if p.kind in ("pos", "kwonly")]
    return [p.name for p in plain] != [p.name for p in wanted]


# -- static gate: F821 / F811 introduced in changed files ------------------------------------------

STATIC_CODES = ("F821", "F811")


def _ruff_codes(ruff: str, project: Path, path: str, text: str) -> list[tuple[int, str, str]] | None:
    argv = [ruff, "check", "--no-cache", "--quiet", "--isolated", "--output-format=json",
            f"--select={','.join(STATIC_CODES)}", "--stdin-filename", path, "-"]
    argv = sandbox.readonly_argv(argv)
    try:
        result = subprocess.run(argv, input=text, cwd=project, text=True, capture_output=True,
                                timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode not in (0, 1):
        return None
    try:
        found = json.loads(result.stdout or "[]")
    except ValueError:
        return None
    # syntax errors come back too: the parse check and the import check report those already
    return [(int(f["location"]["row"]), str(f.get("code")), str(f["message"])) for f in found
            if f.get("code") in STATIC_CODES]


def _ast_codes(text: str) -> list[tuple[int, str, str]]:
    """A stdlib stand-in for ruff's F821/F811: names loaded that nothing in the module binds
    (over-approximating what is bound), and top-level functions/classes defined twice."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    bound: set[str] = set(_BUILTINS) | {"__file__", "__name__", "__doc__", "__spec__",
                                        "__builtins__", "__path__", "__package__"}
    star = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    star = True
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
    findings = []
    if not star:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound:
                findings.append((node.lineno, "F821", f"Undefined name `{node.id}`"))
    seen: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            decorated = any(isinstance(d, ast.Attribute) or getattr(d, "id", "") == "overload"
                            for d in node.decorator_list)
            if node.name in seen and not decorated:
                findings.append((node.lineno, "F811",
                                 f"Redefinition of unused `{node.name}` from line {seen[node.name]}"))
            seen[node.name] = node.lineno
    return findings


def static_findings(project: Path, path: str, text: str) -> list[tuple[int, str, str]]:
    ruff = sandbox.find_python_tool(project, "ruff")
    found = _ruff_codes(ruff, project, path, text) if ruff else None
    return _ast_codes(text) if found is None else found


def _static_gate(spec: Spec, project: Path, changed: list[str]) -> list[Item]:
    items = []
    for path in changed:
        if not path.endswith(".py"):
            continue
        try:
            text = (project / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        before = spec.originals.get(path)
        known = {(code, message) for _, code, message in
                 (static_findings(project, path, before) if before else [])}
        lines = text.splitlines()
        new = [(line, code, message) for line, code, message in static_findings(project, path, text)
               if (code, message) not in known
               and not (code == "F811" and 0 < line <= len(lines)
                        and lines[line - 1].lstrip().startswith(("import ", "from ")))]
        label = f"{path} (undefined names, redefinitions)"
        if new:
            shown = "; ".join(f"line {line} {code} {message}" for line, code, message in new[:4])
            more = f" (+{len(new) - 4} more)" if len(new) > 4 else ""
            items.append(Item("static", label, False, shown + more))
        else:
            items.append(Item("static", label, True))
    return items


# -- the whole check -----------------------------------------------------------------------------


def run(spec: Spec, project: Path, changed: list[str], *, stop: threading.Event | None = None,
        commands: bool = True, tests: bool = True, calls: bool = True) -> Result:
    """Run the spec's checks on the project as it is now. `changed`: files written this turn
    (the static gate's scope). Commands and tests are the slow part: per-write checks skip them."""
    project = project.resolve()
    result = Result()
    result.items += _static_gate(spec, project, changed)
    if features.on("acceptance_stubs"):
        result.items += _stub_items(spec, project)
    if not spec.runs:
        result.items += _static_names(spec, project)
        return result
    result.items += _run_names(spec, project, stop, calls)
    if stop is not None and stop.is_set():
        return result
    if commands:
        for command in spec.commands:
            result.items.append(_command_item(command, project, stop, "command",
                                              absent=spec.absent_inputs.get(command, [])))
    if tests and spec.tests:
        if targeted_tests.applies(project, spec.tests):
            result.items += _targeted_items(spec, project, changed, stop)
        else:
            result.items.append(_command_item(spec.tests, project, stop, "tests"))
    return result


def _targeted_items(spec: Spec, project: Path, changed: list[str],
                    stop: threading.Event | None) -> list[Item]:
    """targeted_tests (a large pytest suite): the test files related to this turn's changes
    first, within run_check's timeout; when they pass (or none relate), the whole suite with a
    longer timeout scaled to its measured time. A whole suite that does not finish is reported
    as slow, not failing, with the targeted result."""
    assert spec.tests
    files, related = targeted_tests.select(project, changed)
    debuglog.record("stage", stage="acceptance.targeted", files=files, related=related)
    targeted: tuple[float, int] | None = None
    summary = "no test file relates to the files changed this turn"
    items: list[Item] = []
    if files:
        shown = ", ".join(files[:3]) + (f" +{len(files) - 3} more" if len(files) > 3 else "")
        cap = f", first {len(files)} of {related} related" if related > len(files) else ""
        started = time.monotonic()
        item = _command_item(targeted_tests.command_for(spec.tests, files), project, stop, "tests")
        targeted = (time.monotonic() - started, len(files))
        item.label = f"`{spec.tests}` on the tests related to your changes ({shown}{cap})"
        items.append(item)
        if not item.ok or (stop is not None and stop.is_set()):
            return items  # the targeted failure is the evidence; the whole suite can wait
        summary = f"the targeted tests passed ({item.detail or 'exit code 0'})"
    label = f"`{spec.tests}` (whole suite)"
    timeout = targeted_tests.full_timeout(project, targeted)
    if spec.suite_timed_out and spec.suite_timed_out >= timeout:
        items.append(Item("tests", label, True, (
            f"not run again: it did not finish in {spec.suite_timed_out}s earlier this turn; the "
            f"full suite is slow, not failing; {summary}"), note=True))
        return items
    started = time.monotonic()
    item = _command_item(spec.tests, project, stop, "tests", timeout=timeout)
    if targeted_tests.timed_out(item.output):
        spec.suite_timed_out = timeout
        item = Item("tests", label, True, (
            f"the full suite is slow, not failing: it did not finish in {timeout}s (a time "
            f"limit, not a test result); {summary}"), note=True)
    else:
        if check_feedback.exit_code(item.output) in (0, 1):  # collected and ran (2+: cut short)
            targeted_tests.record_suite(project, time.monotonic() - started)
        item.label = label
    debuglog.record("stage", stage="acceptance.full_suite", timeout=timeout, ok=item.ok,
                    timed_out=item.note)
    items.append(item)
    return items


def _command_item(command: str, project: Path, stop: threading.Event | None, kind: str,
                  timeout: int = sandbox.TIMEOUT, absent: list[str] | None = None) -> Item:
    output = sandbox.run_check(check_feedback.short_tracebacks(command), project, stop=stop,
                               harness=True, timeout=timeout)
    code = check_feedback.exit_code(output)
    label = f"`{command}`"
    if code is None:
        if output.startswith("Check timed out"):
            return Item(kind, label, False, output.strip(), output)
        return Item(kind, label, True, f"not run here ({output.splitlines()[0][:120]})")
    if code == 0 or (kind == "tests" and code == check_feedback.NO_TESTS):
        return Item(kind, label, True, check_feedback.totals(output), output)
    if (code == 2 and kind == "command" and _BARE_MODULE.match(command) and "usage:" in output
            and features.on("task_module_commands")):  # a bare name, run without its arguments
        return Item(kind, label, True, "runs (shows its usage without arguments)", output)
    if absent and kind == "command" and features.on("command_absent_inputs"):
        name = _reports_absent(output, project, absent)
        if name:  # an example input the project does not have (engineering A/B 2026-10-21)
            return Item(kind, label, True, f"runs (exit code {code}: reports {name} missing, an "
                        "example input the project does not have)", output)
    body = check_feedback.excerpt(output, 900)
    return Item(kind, label, False, f"exit code {code}\n{body}", output)


# -- feedback --------------------------------------------------------------------------------------

def short_line(result: Result) -> str:
    """After a write: one condensed line with the failing items only ('' when all pass)."""
    failing = result.failing
    if not failing:
        return ""
    parts = []
    for item in failing[:4]:
        detail = item.detail.splitlines()[0] if item.detail else ""
        parts.append(" ".join(f"{item.label} {detail}".split())[:180])
    more = f" (+{len(failing) - 4} more)" if len(failing) > 4 else ""
    return (f"(Vega acceptance) {len(failing)} of {len(result.items)} task checks fail: "
            + "; ".join(parts) + more)


def finish_note(result: Result) -> str:
    """At finish: what failed, as evidence; the model decides what to do about it."""
    failing = result.failing
    passed = len(result.items) - len(failing)
    lines = [(f"(Vega harness) Before you finish: Vega checked your work against the task in the "
              f"sandbox; {len(failing)} of {len(result.items)} check(s) fail:")]
    for item in failing[:MAX_SHOWN]:
        lines.append(f"- {item.line()}")
    if len(failing) > MAX_SHOWN:
        lines.append(f"- … and {len(failing) - MAX_SHOWN} more")
    if passed:
        lines.append(f"({passed} passed.)")
    lines.append("These checks come from the task's own words. Fix what the task requires, or "
                 "say in your answer which of these is still open and why.")
    return "\n".join(lines)


def pipeline_failures(result: Result) -> list[str]:
    """Failing items as lines for a pipeline's repair prompt."""
    return [item.line() for item in result.failing]
