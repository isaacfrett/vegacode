"""Shape routing (redesign D): one decision per task, from the task's shape.

Failure diagnosis (2026-10-16, candidate D): the old gates
decide by task length and keywords. `pipeline.target_files` declines any task over 400
characters and any task that names a second file (a README, a spec), and cannot resolve dotted
names; `question_pipeline.applies` matches keywords anywhere, including inside backticks. So
single-file "implement this to this spec" tasks all ran in the agent loop, and a quoted CLI
example (`app add NAME VALUE`) turned a read-only question into a change.

`route_task` computes the shape once:

- the files the task resolves to: named paths, dotted `pkg.mod.func` names (acceptance's
  resolver, redesign A), "in X.py", and the one file that holds a stub the task names;
- which named files are read-only inputs (data, spec/README/doc files, other code it reads) and
  which are helpers it runs (`python check.py`), never targets;
- whether the project has visible tests, and acceptance checks the task states;
- whether it is a question or asks for a change, by words OUTSIDE code spans (backticks, fenced
  code and quoted CLI examples are blanked first);
- for the direct pipeline: whether the writer's prompt fits the context window, computed from
  the file sizes, not from the task's characters.

Then one decision table (see `decide`). It is the only router: the old per-pipeline gates and
the `shape_routing` switch were removed on 2026-10-17, after a measured comparison with the
old gates. The route and its
reasons go to the debug log (stage `route`) and to eval attempt records (`route`).
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import data_preview, features, pipeline, verbs
from .pipeline import (
    _CHANGE_AFTER,
    _CHANGE_CUE,
    _QUESTION,
    MAX_FILE_CHARS,
    _clause_before,
    is_data_input,
)
from .tools import _project_files

QUESTION, DIRECT, PROJECT, DELIVERABLE, AGENT = "question", "direct", "project", "deliverable", "agent"
NONPYTHON = "nonpython"

# -- the writer's budget (direct pipeline) --------------------------------------------------------
CHARS_PER_TOKEN = 3.5  # context.CHARS_PER_TOKEN
DEFAULT_NUM_CTX = 8192
WRITER_OVERHEAD_CHARS = 2500  # system prompt, project context and the wrapper text
REPLY_GROWTH = 1.3  # the reply is the whole file, a stub grows when implemented
REPLY_MIN_CHARS = 2000  # the code written into a small stub
FEEDBACK_CHARS = 1200  # one repair round's failure list
WINDOW_SHARE = 0.9  # leave room for the tokenizer's error
INPUT_CHARS = pipeline.INPUT_CHARS  # a data input as the writer sees it
DOC_CHARS = pipeline.DOC_CHARS  # a spec document shown whole up to this, else cut
INPUTS_MAX = pipeline.INPUTS_MAX + 1  # as pipeline.route_inputs_shown shows them

# -- what the task names ---------------------------------------------------------------------------
_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_SPAN = re.compile(r"`([^`\n]+)`")
# "quoted CLI examples": a quoted string that starts like a command or holds a flag
_QUOTED = re.compile(r"(?<![\w])(\"[^\"\n]{1,200}\"|“[^”\n]{1,200}”|'[^'\n]{1,200}')(?![\w])")
_PATH_ONLY = re.compile(r"^\s*(?:[\w.-]*(?:/[\w.-]+)*\.[A-Za-z0-9]{1,8}|[\w.-]+(?:/[\w.-]+)*/)\s*$")
_ANY_PATH = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w-][\w.-]*\.([A-Za-z][A-Za-z0-9]{0,7}))(?![\w/-])")
_COMMAND = re.compile(
    r"^\s*(?:\$\s*)?(?:python3?|pytest|Rscript|R|bash|sh|zsh|node|npm|npx|make|cargo|go|uv|pip3?|"
    r"ruff|mypy|coverage|gcc|cc|clang|java|javac|ruby|perl|php|swift|docker|kubectl|git|curl|"
    r"sqlite3|psql|vega)\b\s+\S")
_RUN_PY = re.compile(r"(?:^|\s)(?:\$\s*)?python3?\s+(?:-\w+\s+)*((?:[\w.-]+/)*[\w.-]+\.py)\b")
_PY_EXT = {"py"}
_DOC_EXT = {"md", "rst", "adoc", "txt", "pdf", "html"}
_CONFIG_EXT = {"toml", "cfg", "ini", "conf", "env", "lock"}
_OTHER_CODE_EXT = {"rs", "go", "js", "mjs", "cjs", "ts", "tsx", "jsx", "java", "cs", "rb", "c", "h",
                   "cc", "cpp", "hpp", "swift", "kt", "r", "sql", "sh", "php", "scala", "lua", "pl",
                   "m", "jl", "dart", "zig", "ex", "exs", "hs", "ml", "vue", "svelte", "css", "scss"}
_DOC_NAMES = re.compile(r"(?i)(?:^|/)(?:readme|spec|specification|design|requirements|icd|contract|"
                        r"changelog|contributing|notes|plan)[\w.-]*$")
_TEST_PY = re.compile(r"(^|/)(tests?/.*\.py|test_[^/]*\.py|[^/]*_test\.py|conftest\.py)$")
# clause words that make a non-data file one the task only reads ("see", "follow", "per")
_READ_CUE = re.compile(r"\b(?:read(?!-)|reads|reading|see|follow|follows|following|per|according to|"
                       r"described in|documented in|specified in|defined in|stated in|from|given|using|"
                       r"uses|like|based on|load|loads|parse|parses|open|opens|port|ports|"
                       r"translate|translates|mirror|mirrors|match|matches)\b", re.IGNORECASE)

# -- what the task asks -----------------------------------------------------------------------------
_CHANGE_VERBS = re.compile(
    r"\b(?:implement|write|complete|create|add|define|code up|fill in|finish|fix|rename|refactor|"
    r"move|remove|delete|update|change|make|run|install|rewrite|tidy|clean up|optimi[sz]e|"
    r"speed up|convert|port|replace|extend|produce|generate|build|save|edit|modify|correct|"
    r"repair|migrate|split|merge|wire|hook up|restructure|reorgani[sz]e|deduplicate|patch)\b",
    re.IGNORECASE)
# a negated clause runs to the end of its sentence (a dot inside a path does not end it)
_NEGATED = re.compile(r"\b(?:do not|don't|never|without|must not|should not|no need to)\b"
                      r"(?:[^.;\n]|\.(?=[\w/`]))*", re.IGNORECASE)
# a whole-scope "do not change anything": never a writing pipeline. "Do not modify anything
# else", "don't change what the functions do" are constraints on a change, not read-only.
_READ_ONLY_WHOLE = re.compile(
    r"(?:^|[.;:!?,(]\s*|\b(?:and|so|please|but)\s+)(?:do not|don't|never|without)\s+"
    r"(?:change|modify|edit|touch|write(?: to)?)\s+"
    r"(?:any(?:thing| files?| code)?|the (?:code|project|repo(?:sitory)?|files?)|files|code)\b",
    re.IGNORECASE)
# what makes it scoped: "any file except review.json", "anything under tests/", "any other file"
_SCOPED = re.compile(r"^\W{0,3}(?:(?:files?|code)\s+)?(?:else|other|outside|except|but|besides|"
                     r"apart|beyond|than|under|in|inside|within|other than|not|itself)\b", re.IGNORECASE)
_ASKS = re.compile(r"\bonly report\b|\btell me\b|\breport\b|\bexplain\b|\bdescribe\b|\bfind\b|"
                   r"\blocate\b", re.IGNORECASE)
# question_change_cues (2026-10-20): "report" only as a verb, i.e. where a request or clause
# starts ("Report every bug", "and report the marker", "Only report it"). "The weekly report must
# also mark ..." is a noun and asked for a change; `pkg/report.py` is a path.
_ASKS_VERB = re.compile(
    r"\btell me\b|\bexplain\b|\bdescribe\b|\bfind\b|\blocate\b"
    r"|(?:^|[.!?;:,(]\s*|\b(?:only|and|then|please|to|also|just|or|but)\s+)report\b(?![/.]\w)",
    re.IGNORECASE)
# a requirement stated with "must": "The stock report must also mark ...". Negated clauses are
# removed first; "you must answer" / "the answer must name" constrain the reply, not the code.
_MUST = re.compile(r"\bmust\b", re.IGNORECASE)
_MUST_OF_ANSWER = re.compile(r"\b(?:you|your (?:answer|reply)|the (?:answer|reply)|answers?|"
                             r"replies)\s+must\b", re.IGNORECASE)
_CROSS_FILE = re.compile(r"\brename\b", re.IGNORECASE)  # callers in other files change too


@dataclass
class Route:
    """Where a task goes and why. Paths are project-relative POSIX strings."""

    name: str
    reasons: list[str] = field(default_factory=list)
    question: bool = False
    change: bool = False
    targets: list[str] = field(default_factory=list)  # Python files the task changes
    other_targets: list[str] = field(default_factory=list)  # non-Python files to change/produce
    inputs: list[str] = field(default_factory=list)  # read-only inputs (data, docs, other code)
    context: list[str] = field(default_factory=list)  # Python files named only as context
    helpers: list[str] = field(default_factory=list)  # scripts the task runs (`python check.py`)
    has_tests: bool = False
    acceptance: bool = False  # the task states names or commands acceptance can check
    task_chars: int = 0
    writer_tokens: int | None = None  # the direct pipeline's estimated need, when computed
    check_command: str = ""  # nonpython: the command the task names to check its file
    # nonpython: "web" when its CSS/HTML/JS config runs (web_pipeline, dispatched by
    # nonpython_pipeline.run, so the route name and its switch stay `nonpython`)
    pipeline: str = ""
    # direct_context: project modules the target imports and docs naming a distinctive task
    # token, shown to the direct writer after the inputs, within what the window leaves
    related: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return asdict(self)

    def target_paths(self, project: Path) -> list[Path]:
        return [project / path for path in self.targets]


# -- masking ----------------------------------------------------------------------------------------

def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + re.sub(r"[^\n]", " ", text[start:end]) + text[end:]


def mask_code(task: str) -> str:
    """The task with fenced code, backticked code and quoted CLI examples blanked to spaces (same
    length, so offsets still match the task). A backticked path keeps its text: it names a file."""
    masked = task
    for match in _FENCE.finditer(task):
        masked = _blank(masked, match.start(), match.end())
    for match in _SPAN.finditer(masked):
        inner = match.group(1)
        if _PATH_ONLY.match(inner) and not _COMMAND.match(inner):
            continue
        masked = _blank(masked, match.start(), match.end())
    for match in _QUOTED.finditer(masked):
        inner = match.group(1)[1:-1]
        if _COMMAND.match(inner) or re.search(r"(?:^|\s)--?\w", inner):
            masked = _blank(masked, match.start(), match.end())
    return masked


def strip_code(task: str) -> str:
    """mask_code, with the blanks collapsed (for keyword matching)."""
    return re.sub(r"[ \t]{2,}", " ", mask_code(task))


# -- question or change -----------------------------------------------------------------------------

def read_only(task: str) -> bool:
    """The task says not to change anything at all ("Don't change any files")."""
    text = strip_code(task)
    return any(not _SCOPED.match(text[m.end():m.end() + 40]) and
               not re.search(r"\bother\b", m.group(0), re.IGNORECASE)
               for m in _READ_ONLY_WHOLE.finditer(text))


def change_verbs(task: str) -> list[str]:
    """Change verbs outside code spans and outside negated clauses, in request position (a
    question about a function named add, or "what port", asks for no change: see verbs)."""
    text = _NEGATED.sub(" ", strip_code(task))
    return sorted(set(verbs.requested(_CHANGE_VERBS, text)))


def asks(task: str) -> bool:
    from .question_pipeline import _ABOUT_VEGA

    text = strip_code(task)
    cues = _ASKS_VERB if features.on("question_change_cues") else _ASKS
    return bool(_QUESTION.search(text) or cues.search(text) or _ABOUT_VEGA.search(text))


def requirements(task: str) -> list[str]:
    """Sentences outside code that state a requirement with "must" ("The stock report must also
    mark every item ..."): not negated ("must not send more" is a constraint), not a question, and
    not about the reply ("your answer must name the file")."""
    text = _NEGATED.sub(" ", strip_code(task))
    found = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = sentence.strip()
        musts = len(_MUST.findall(sentence))
        if not musts or sentence.endswith("?") or _QUESTION.match(sentence):
            continue
        if musts > len(_MUST_OF_ANSWER.findall(sentence)):
            found.append(sentence)
    return found


# -- files ------------------------------------------------------------------------------------------

def _kind(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in _PY_EXT:
        return "test" if _TEST_PY.search(path) else "py"
    if data_preview.is_data_path(path) and ext not in _DOC_EXT - {"txt"}:
        return "data"
    if ext in _OTHER_CODE_EXT:
        return "code"
    if ext in _DOC_EXT or _DOC_NAMES.search(path):
        return "doc"
    if ext in _CONFIG_EXT:
        return "config"
    return ""


def _existing(project: Path, path: str, files: set[str]) -> str | None:
    """The project file `path` names (also under src/, or by its tail), or None."""
    if path in files:
        return path
    for candidate in (f"src/{path}",):
        if candidate in files:
            return candidate
    tails = [f for f in files if f.endswith("/" + path)]
    return tails[0] if len(tails) == 1 else None


def _local_clause(masked: str, start: int) -> str:
    """The clause before a path, from the previous named path on (its cues are that path's)."""
    clause = _clause_before(masked, start)
    head = clause.removesuffix("`")
    cut = head.rfind("`")
    return head[cut + 1:] if cut >= 0 else clause


def _changes_here(masked: str, start: int, end: int) -> bool:
    """A change cue is the last cue in the clause before the path, or one right after it."""
    clause = _local_clause(masked, start)
    changes = [m.end() for m in _PATH_CHANGE.finditer(clause)]
    inputs = [m.end() for m in _READ_CUE.finditer(clause)]
    if changes and (not inputs or max(changes) > max(inputs)):
        return True
    return bool(_CHANGE_AFTER.search(masked[end:end + 80]))


def _reads_here(masked: str, start: int) -> bool:
    clause = _local_clause(masked, start)
    changes = [m.end() for m in _PATH_CHANGE.finditer(clause)]
    inputs = [m.end() for m in _READ_CUE.finditer(clause)]
    return bool(inputs) and (not changes or max(inputs) > max(changes))


def _helpers(task: str) -> set[str]:
    """Python scripts the task runs (`python check.py`, `python run_r.py x.R`): never targets."""
    found: set[str] = set()
    for match in _SPAN.finditer(task):
        found.update(_RUN_PY.findall(" " + match.group(1)))
    for match in _QUOTED.finditer(task):
        found.update(_RUN_PY.findall(" " + match.group(1)[1:-1]))
    found.update(m.group(1) for m in re.finditer(
        r"\brun\s+(?:it\s+)?(?:with\s+)?python3?\s+((?:[\w.-]+/)*[\w.-]+\.py)\b", task, re.IGNORECASE))
    return found


def _named_paths(task: str, masked: str) -> list[tuple[str, int, int]]:
    """(path, start, end) for every file path the task names outside commands and code."""
    found = []
    for match in _ANY_PATH.finditer(masked):
        path = match.group(1).rstrip(".")
        if "." not in path or path.startswith("."):
            continue
        ext = match.group(2).lower()
        if ext.isdigit():
            continue
        if not (_kind(path) or "/" in path):
            continue
        if "/" not in path and len(path.split(".")[0]) < 2:
            continue  # "a.m.", "e.g."
        found.append((path, match.start(1), match.start(1) + len(path)))
    return found


def _stubs(source: str) -> set[str]:
    """Names of stub definitions: functions whose body is only a docstring, pass, ... or raise
    NotImplementedError; a class whose methods are all stubs (and those Class.method names)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()

    def trivial(node: ast.AST) -> bool:
        body = list(getattr(node, "body", []))
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        return all(isinstance(s, ast.Pass)
                   or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
                   or (isinstance(s, ast.Raise) and "NotImplementedError" in ast.unparse(s))
                   for s in body)

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and trivial(node):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            methods = [c for c in node.body if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))]
            stubbed = [m for m in methods if trivial(m) and m.name != "__init__"]
            names.update(f"{node.name}.{m.name}" for m in stubbed)
            if stubbed and len(stubbed) >= len([m for m in methods if m.name != "__init__"]):
                names.add(node.name)
    return names


_IDENTS = re.compile(r"`([A-Za-z_][\w.]*)(?:\([^`]*\))?`|\b([A-Za-z_]\w*)\(")


def _named_identifiers(task: str) -> set[str]:
    names: set[str] = set()
    for match in _IDENTS.finditer(task):
        dotted = match.group(1) or match.group(2)
        parts = dotted.split(".")
        names.update(parts)
        names.update(".".join(parts[i:i + 2]) for i in range(len(parts) - 1))
    return names


def writer_tokens(task: str, file_chars: int, input_chars: int) -> int:
    """The direct pipeline's need for its first write and one repair round, in tokens."""
    reply = max(int(file_chars * REPLY_GROWTH), file_chars + REPLY_MIN_CHARS)
    chars = (WRITER_OVERHEAD_CHARS + len(task) + file_chars + input_chars
             + reply + FEEDBACK_CHARS + reply)
    return int(chars / CHARS_PER_TOKEN)


def _input_chars(project: Path, inputs: list[str]) -> int:
    total = 0
    for path in inputs[:INPUTS_MAX]:
        try:
            size = (project / path).stat().st_size
        except OSError:
            continue
        cap = DOC_CHARS if _kind(path) in {"doc", "config", "code"} else INPUT_CHARS
        total += min(size, cap)
    return total


# -- the decision -----------------------------------------------------------------------------------

def route_task(task: str, project: Path, *, num_ctx: int | None = None,
               can_write: bool = True) -> Route:
    """The task's shape and the path it goes to (no model call)."""
    from . import project_pipeline, question_pipeline

    project = Path(project)
    route = Route(AGENT, task_chars=len(task))
    masked = mask_code(task)
    verbs = change_verbs(task)
    from . import deliverable_pipeline

    outputs = deliverable_pipeline.targets(project, task) if verbs else []
    if read_only(task) and not outputs:
        route.question = True
        route.reasons.append("says not to change any file")
    elif verbs:
        route.change = True
        route.reasons.append("asks for a change: " + ", ".join(verbs[:5]))
    elif asks(task) and features.on("question_change_cues") and requirements(task):
        route.change = True  # "The stock report must also mark ...": requirements, not a question
        route.reasons.append("states requirements with must: treated as a change")
    elif asks(task):
        route.question = True
        route.reasons.append("a question with no change verb outside code")
    else:
        route.change = True
        route.reasons.append("no question form: treated as a change")

    if route.question:
        if question_pipeline._ABOUT_VEGA.search(task) or not question_pipeline._CODE_ANCHOR.search(task):
            route.reasons.append("conversation: nothing in the project to look up")
            return route
        if question_pipeline.is_broad(task):
            route.reasons.append("broad question: the agent loop explores")
            return route
        if not features.on("question_pipeline"):
            route.reasons.append("specific question; question_pipeline is off")
            return route
        route.name = QUESTION
        route.reasons.append("specific question about the code")
        return route
    if not can_write:
        route.reasons.append("read-only session")
        return route

    try:
        all_files = [p.relative_to(project).as_posix() for p in _project_files(project)]
    except OSError:
        all_files = []
    files = set(all_files)
    route.has_tests = bool(project_pipeline._test_files(project))
    helpers = _helpers(task)
    route.helpers = sorted(h for h in (_existing(project, h, files) or h for h in helpers))

    candidates: list[str] = []
    cued: set[str] = set()
    used: set[str] = set()  # files only named as something the new code uses
    weak: set[str] = set()  # files holding an existing name the task mentions (may change)
    protected = _protected(masked)
    for path, start, end in _named_paths(task, masked):
        real = _existing(project, path, files)
        name = real or path
        if ".." in path:
            continue  # "../etc/passwd", "notes..txt": a payload, not a file
        if (path in helpers or name in route.helpers) and not (
                _changes_here(masked, start, end) and not _reads_here(masked, start)):
            continue  # a script the task runs (`python check.py`), not one it changes
        if _is_protected(name, protected) or _is_protected(path, protected):
            if real:
                _add(route.inputs, name)  # "Do not change `shopcart/` or the spec"
            continue
        kind = _kind(name)
        changes = _changes_here(masked, start, end)
        if kind == "py":
            if real and not changes and _reads_strongly(masked, start):
                _add(route.inputs, name)  # "Port `python/semver.py` to Go"
                continue
            if name not in candidates:
                candidates.append(name)
            if changes:
                cued.add(name)
        elif kind == "test":
            if not real or (changes and not _reads_here(masked, start)):
                _add(route.other_targets, name)  # tests to write
            else:
                _add(route.inputs, name)
        elif kind == "data":
            if real and (is_data_input(task, path) or not changes):
                _add(route.inputs, name)  # read, not changed
            elif real or changes:
                _add(route.other_targets, name)  # changed in place, or an output to produce
            # else: a file that does not exist, named with no change cue (a default path)
        elif kind in {"doc", "config"}:
            if real and not changes:
                _add(route.inputs, name)
            elif real or changes:
                _add(route.other_targets, name)
        elif kind == "code":
            if real and _reads_here(masked, start) and not changes:
                _add(route.inputs, name)
            else:
                _add(route.other_targets, name)
    # dotted and plain names the acceptance resolver ties to files (redesign A)
    from . import acceptance

    try:
        spec = acceptance.extract(project, task)
    except Exception:  # noqa: BLE001 - routing never fails a turn
        spec = None
    if spec is not None:
        route.acceptance = bool(spec.names or spec.commands) and features.on("acceptance")
        for required in spec.names:
            if _TEST_PY.search(required.path) or _is_protected(required.path, protected) \
                    or required.path in route.inputs:
                continue
            if required.path in route.helpers and not required.exists \
                    and required.path not in candidates:
                continue  # `Ticker` (in ticker.js) is not check.py's, whatever the resolver says
            if required.path not in candidates:
                candidates.append(required.path)
            if not required.exists:
                cued.add(required.path)  # a name to write there
            elif not _only_used(task, masked, required.attr):
                weak.add(required.path)  # an existing name: maybe the one to change
            else:
                used.add(required.path)  # "with `chunk_text(...)` from chunking.py"
    # `settings.MissingSecret` for payway_client/settings.py: a module called by its short name
    stems = {Path(c).stem: c for c in candidates if c.endswith(".py")}
    for match in _SPAN.finditer(task):
        short = re.match(r"^([A-Za-z_]\w*)\.([A-Za-z_]\w*)", match.group(1).strip())
        if short and short.group(1) in stems and short.group(1) != "__init__":
            path = stems[short.group(1)]
            try:
                defined = re.search(r"^(?:class|def|async def)\s+" + re.escape(short.group(2))
                                    + r"\b", (project / path).read_text(encoding="utf-8"),
                                    re.MULTILINE)
            except (OSError, UnicodeDecodeError):
                defined = None
            (weak if defined else cued).add(path)
    # `pkg.__all__`, `pkg.mod.__getattr__`: module attributes the resolver leaves out (dunders)
    py_paths = {f for f in files if f.endswith(".py") and not _TEST_PY.search(f)}
    for match in _SPAN.finditer(task):
        dotted = re.match(r"^([A-Za-z_]\w*(?:\.\w+)+)", match.group(1).strip())
        if not dotted or "__" not in dotted.group(1).split(".")[-1]:
            continue
        found = acceptance._module_file(project, dotted.group(1).split("."), py_paths)
        if found and found[1] == len(dotted.group(1).split(".")) - 1 \
                and not _is_protected(found[0], protected) and found[0] not in route.inputs:
            if found[0] not in candidates:
                candidates.append(found[0])
            cued.add(found[0])
    # stubs the task names, when they live in exactly one file
    named = _named_identifiers(task)
    stub_files: dict[str, set[str]] = {}
    python = [f for f in all_files if f.endswith(".py") and not _TEST_PY.search(f)
              and f not in route.helpers and not f.startswith(".vega")
              and not _is_protected(f, protected)]
    for path in python:
        try:
            source = (project / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        hits = _stubs(source) & named
        if hits:
            stub_files[path] = hits
    if not candidates and len(stub_files) == 1:
        only = next(iter(stub_files))
        candidates.append(only)
        route.reasons.append(f"the stub the task names is in {only}")
    elif not candidates and not route.other_targets and len(python) <= 2:
        with_stubs = [p for p in python if _stubs((project / p).read_text(encoding="utf-8",
                                                                            errors="replace"))]
        if len(with_stubs) == 1:
            candidates.append(with_stubs[0])
            route.reasons.append(f"tiny project: the only stub file is {with_stubs[0]}")
    # several candidates: the ones the task changes (a change cue, a stub it names, a name to
    # write); the rest are context
    if len(candidates) > 1:
        # Only when some file is clearly the target (a change cue, a stub it names, a name to
        # write) are the others set aside, and then only files named with no change cue or only
        # as something the code uses; a file holding an existing name the task mentions stays a
        # target (the CLI's `main()` may need the change too).
        strong = [c for c in candidates if c in cued or c in stub_files]
        keep = [c for c in candidates if c in strong or (c in weak and c not in used)]
        if strong and len(keep) < len(candidates):
            route.context = [c for c in candidates if c not in keep]
            route.reasons.append("context only: " + ", ".join(route.context))
            candidates = keep
    only = _only_paths(masked, files)
    if only:  # "Change only `payments/webhook.py`": the rest is context
        dropped = [c for c in candidates if c not in only]
        if dropped and [c for c in candidates if c in only]:
            route.context += [c for c in dropped if c not in route.context]
            candidates = [c for c in candidates if c in only]
            route.reasons.append("the task allows changes only to " + ", ".join(sorted(only)))
        kept = [o for o in route.other_targets if o in only or o not in files]
        route.inputs += [o for o in route.other_targets if o not in kept and o not in route.inputs]
        route.other_targets = kept
    if read_only(task) and candidates:  # "Do not change any file ... write the findings to x.json"
        route.inputs += [c for c in candidates if c not in route.inputs]
        route.reasons.append("read-only for existing files: " + ", ".join(candidates))
        candidates = []
    route.targets = candidates
    route.inputs = [i for i in route.inputs if i not in candidates and i not in route.other_targets]
    route = decide(route, task, project, files, num_ctx=num_ctx)
    if route.name == DIRECT and len(route.targets) == 1:
        route = _entry_targets(route, task, project, files, all_files, protected, num_ctx)
    if route.name == DIRECT and features.on("direct_context"):
        route.related = related_files(project, route.targets[0], task, all_files,
                                      exclude={*route.inputs, *route.targets})
        if route.related:
            route.reasons.append("context for the writer: " + ", ".join(route.related))
    return route


# -- route_cli_target / route_prose_class ---------------------------------------------------------
# Why (2026-10-20): goals that add a CLI subcommand or flag went direct with only the library
# file as the target, so the command's parser in cli.py was never written; and a goal that named
# the class to change in prose (CamelCase, no backticks) routed only to a settings module.

_PROG = re.compile(r"\bprog\s*=\s*[\"']([\w.-]+)[\"']")
_CAMEL = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+)\b")  # ApiClient, not Warehouse
_CLASS_DEF = re.compile(r"^class\s+([A-Za-z_]\w*)", re.MULTILINE)


def _console_scripts(project: Path, py_files: list[str]) -> dict[str, str]:
    """pyproject [project.scripts] `name = "pkg.cli:main"` -> the module's project file."""
    import tomllib

    try:
        data = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return {}
    scripts = (data.get("project") or {}).get("scripts") or {}
    found = {}
    for name, target in scripts.items() if isinstance(scripts, dict) else ():
        module = str(target).split(":", 1)[0].strip()
        path = _module_paths(module, py_files) if module else None
        if path:
            found[str(name)] = path
    return found


def cli_files(project: Path, task: str, py_files: list[str]) -> list[str]:
    """The project files that parse a backticked command the task names (`todo export --csv`):
    the one file whose argparse `prog=` is the command's first word, else the module the
    pyproject console script of that name runs."""
    spans = [m.group(1).split() for m in _SPAN.finditer(task)]
    words = {w[0] for w in spans if len(w) >= 2 and re.fullmatch(r"[a-z][\w-]*", w[0])
             and not _COMMAND.match(" ".join(w))}
    if not words:
        return []
    progs: dict[str, set[str]] = {}
    for path in py_files:
        try:
            source = (project / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for match in _PROG.finditer(source):
            progs.setdefault(match.group(1), set()).add(path)
    scripts = _console_scripts(project, py_files)
    found: list[str] = []
    for word in sorted(words):
        homes = progs.get(word, set())
        home = next(iter(homes)) if len(homes) == 1 else scripts.get(word) if not homes else None
        if home and home not in found:
            found.append(home)
    return found


def prose_classes(project: Path, task: str, py_files: list[str]) -> list[str]:
    """Files defining a CamelCase class the task names in prose (outside code spans and negated
    clauses), when exactly one project file defines it."""
    names = set(_CAMEL.findall(_NEGATED.sub(" ", strip_code(task))))
    if not names:
        return []
    homes: dict[str, set[str]] = {}
    for path in py_files:
        try:
            source = (project / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for name in set(_CLASS_DEF.findall(source)) & names:
            homes.setdefault(name, set()).add(path)
    return sorted({next(iter(paths)) for paths in homes.values() if len(paths) == 1})


# -- route_module_entry ----------------------------------------------------------------------------
# Why (2026-10-21): a task reporting "`python -m pkg ...` fails with No module named
# pkg.__main__" went direct with only cli.py: the direct pipeline writes one existing file, so
# pkg/__main__.py was never created and the same check failed every round.

_MODULE_RUN = re.compile(r"(?<![\w.-])python(?:3(?:\.\d+)?)?(?:\s+-[A-Za-z]+)*\s+-m\s+"
                         r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)")
_SUBPROCESS_MODULE = re.compile(r"""["']-m["']\s*,\s*["']([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)["']""")
_README_WORD = re.compile(r"\breadme\b", re.IGNORECASE)
MODULE_ENTRY_SOURCE_BYTES = 200_000  # a larger doc or test file is not searched


def _module_run_sources(project: Path, task: str) -> list[str]:
    """The texts that may say how a package is run: the task, the docs and tests it names (the
    README when it says "README"), and the project's visible tests (the checks that run)."""
    from . import project_pipeline

    texts = [task]
    paths: list[Path] = []
    for path, _start, _end in _named_paths(task, mask_code(task)):
        if _kind(path) in {"doc", "test"}:
            paths.append(project / path)
    if _README_WORD.search(task):
        paths += sorted(project.glob("README*"))
    try:
        paths += project_pipeline._test_files(project)
    except OSError:
        pass
    seen: set[Path] = set()
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            if path.stat().st_size <= MODULE_ENTRY_SOURCE_BYTES:
                texts.append(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    return texts


def module_entry_files(project: Path, task: str, py_files: list[str]) -> list[str]:
    """`pkg/__main__.py` for each project package that the task, a doc or test it names, or a
    visible test runs as `python -m pkg` (or `[sys.executable, "-m", "pkg"]`) and that has no
    __main__.py yet. A module file (`pkg.py`, `pkg/cli.py` for `-m pkg.cli`) runs itself, and a
    name that is not a project package (`pytest`, `http.server`) is not the project's to add."""
    names: list[str] = []
    for text in _module_run_sources(project, task):
        for match in (*_MODULE_RUN.finditer(text), *_SUBPROCESS_MODULE.finditer(text)):
            if match.group(1) not in names:
                names.append(match.group(1))
    py = set(py_files)
    found: list[str] = []
    for name in names:
        for base in ("", "src/"):
            home = base + name.replace(".", "/")
            if home + ".py" in py:
                break  # a module: `python -m` runs it as it is
            if any(f.startswith(home + "/") for f in py):
                main = home + "/__main__.py"
                if main not in py and main not in found:
                    found.append(main)
                break
    return found


def _entry_targets(route: Route, task: str, project: Path, files: set[str], all_files: list[str],
                   protected: set[str], num_ctx: int | None) -> Route:
    """A direct route's one target, plus the CLI file a backticked command runs
    (route_cli_target) and the file of a class named in prose (route_prose_class): with them the
    task goes to the project pipeline. Kept direct when nothing is added, or when the wider route
    would not be the project pipeline (no tests or acceptance checks, project too large).
    route_module_entry: the missing `pkg/__main__.py` a `python -m pkg` needs is also a target,
    and since the direct pipeline cannot create it, the task leaves direct whatever the wider
    route is (the agent loop when not the project pipeline)."""
    cues = []
    if features.on("route_cli_target"):
        cues.append(("runs the backticked command", cli_files))
    if features.on("route_prose_class"):
        cues.append(("defines a class the task names", prose_classes))
    if features.on("route_module_entry"):
        cues.append(("python -m needs it", module_entry_files))
    if not cues:
        return route
    py_files = [f for f in all_files if f.endswith(".py") and not _TEST_PY.search(f)
                and not f.startswith(".vega")]
    taken = {*route.targets, *route.inputs, *route.context, *route.helpers, *route.other_targets}
    added: list[tuple[str, str]] = []
    for why, find in cues:
        try:
            paths = find(project, task, py_files)
        except Exception:  # noqa: BLE001 - routing never fails a turn
            paths = []
        for path in paths:
            if path in taken or _is_protected(path, protected) or any(p == path for p, _ in added):
                continue
            added.append((path, why))
    if not added:
        return route
    wider = Route(**{**route.record(), "name": AGENT,
                     "targets": [*route.targets, *(p for p, _ in added)],
                     "reasons": [r for r in route.reasons if not r.startswith("one code target")]})
    wider.reasons.append("also a target: " + ", ".join(f"{p} ({why})" for p, why in added))
    wider = decide(wider, task, project, files, num_ctx=num_ctx)
    if wider.name == PROJECT or any(why == "python -m needs it" for _, why in added):
        return wider
    route.reasons.append("kept direct: " + ", ".join(p for p, _ in added)
                         + f" would route to {wider.name}")
    return route


# -- direct_context ---------------------------------------------------------------------------------
# Why (2026-10-18): tasks that build on a package of the repo ("... using our <pkg> package") or
# quote an error code ("fails with error AB-12") went to the direct writer with only the target
# file: it never saw the package's __init__.py or the doc that explains the code.

RELATED_MODULES = 3  # project modules the target imports, shown at most
RELATED_DOCS = 2  # docs naming a task token, shown at most
RELATED_DOC_BYTES = 60_000  # a larger doc is not searched for tokens
_ERROR_CODE = re.compile(r"\b[A-Z][A-Z0-9]{1,7}-\d{1,6}\b")
_RELATED_DOC_EXT = {"md", "rst", "txt", "adoc"}


def _module_paths(dotted: str, py_files: list[str]) -> str | None:
    """The project file of a dotted module name: at the root or under src/, else vendored
    (`vendor/x/__init__.py`, `third_party/x.py`): the shortest such path."""
    tail = dotted.replace(".", "/")
    wanted = (tail + ".py", tail + "/__init__.py")
    for prefix in ("", "src/"):
        for suffix in wanted:
            if prefix + suffix in py_files:
                return prefix + suffix
    nested = [f for f in py_files if f.endswith(tuple("/" + w for w in wanted))]
    return min(nested, key=len) if nested else None


def _imported_modules(project: Path, target: str, py_files: list[str]) -> list[str]:
    try:
        tree = ast.parse((project / target).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
        return []
    package = target.rsplit("/", 1)[0].replace("/", ".") if "/" in target else ""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                parts = parts[:len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join([*parts, *([base] if base else [])])
            if base:
                names.append(base)
            names += [f"{base}.{alias.name}" if base else alias.name for alias in node.names
                      if alias.name != "*"]
    found: list[str] = []
    for name in names:
        path = _module_paths(name, py_files)
        if path and path != target and path not in found:
            try:
                if not (project / path).read_text(encoding="utf-8").strip():
                    continue  # an empty __init__.py tells the writer nothing
            except (OSError, UnicodeDecodeError):
                continue
            found.append(path)
    return found


def _task_tokens(task: str, target: str, py_files: list[str]) -> tuple[set[str], set[str]]:
    """(error codes like AB-12, names of the project's own packages/modules the task mentions
    other than the target's own package)."""
    codes = set(_ERROR_CODE.findall(task))
    own = target.split("/", 1)[0] if "/" in target else Path(target).stem
    packages = {Path(f).parts[-2] for f in py_files if f.endswith("/__init__.py")}
    packages |= {Path(f).stem for f in py_files if "/" not in f}
    words = set(re.findall(r"[A-Za-z_][\w]{3,}", strip_code(task)) + re.findall(
        r"`([A-Za-z_]\w{3,})`", task))
    names = {word for word in words if word in packages and word != own}
    return codes, names


def related_files(project: Path, target: str, task: str, all_files: list[str], *,
                  exclude: set[str]) -> list[str]:
    """direct_context: the in-repo modules `target` imports (vendored packages too) and the
    docs/README files that mention a distinctive token of the task (an error code, a project
    package's name), at most RELATED_MODULES + RELATED_DOCS, none already shown."""
    py_files = [f for f in all_files if f.endswith(".py") and not _TEST_PY.search(f)
                and not f.startswith(".vega")]
    modules = [m for m in _imported_modules(project, target, py_files)
               if m not in exclude][:RELATED_MODULES]
    codes, names = _task_tokens(task, target, py_files)
    scored: list[tuple[int, int, str]] = []
    if codes or names:
        for path in all_files:
            if path in exclude or path.startswith(".vega") or \
                    path.rsplit(".", 1)[-1].lower() not in _RELATED_DOC_EXT:
                continue
            if not (path.startswith("docs/") or "/" not in path or
                    Path(path).name.lower().startswith("readme")):
                continue
            try:
                if (project / path).stat().st_size > RELATED_DOC_BYTES:
                    continue
                text = (project / path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            code_hits = sum(text.count(code) for code in codes)
            name_hits = sum(len(re.findall(r"\b" + re.escape(name) + r"\b", text))
                            for name in names)
            if code_hits or name_hits:
                scored.append((-code_hits, -name_hits, path))
    docs = [path for _, _, path in sorted(scored)][:RELATED_DOCS]
    return modules + docs


_NEGATED_PATH = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)+|(?:[\w.-]+/)*[\w-][\w.-]*\.[A-Za-z]\w{0,7})"
                           r"(?![\w/-])")


_KEEP_IT = re.compile(r"\b(?:do not|don't|never|must not)\s+(?:change|modify|edit|touch)\s+"
                      r"(?:it|this file|that file|them)\b", re.IGNORECASE)


def _protected(masked: str) -> set[str]:
    """Paths and folders named inside a "do not change ..." clause."""
    found: set[str] = set()
    for clause in _NEGATED.finditer(masked):
        if not re.search(r"\b(?:change|modify|edit|touch|write|delete|remove|rename)\b",
                         clause.group(0), re.IGNORECASE):
            continue
        text = re.split(r"\b(?:except|other than|but|apart from|besides|only)\b|\)", clause.group(0),
                        maxsplit=1, flags=re.IGNORECASE)[0]
        found.update(m.group(1).rstrip(".") for m in _NEGATED_PATH.finditer(text))
    # "the provided `chunking.py` (tested; do not change it)": the path just before "it"
    for match in _KEEP_IT.finditer(masked):
        before = list(_NEGATED_PATH.finditer(masked[max(0, match.start() - 120):match.start()]))
        if before:
            found.add(before[-1].group(1).rstrip("."))
    return found


def _is_protected(path: str, protected: set[str]) -> bool:
    return any(path == p or (p.endswith("/") and (path.startswith(p) or f"/{p}" in f"/{path}"))
               or path.endswith("/" + p) for p in protected)


# a change cue before a path: pipeline's, plus "put the message in", "store it in"
_PATH_CHANGE = re.compile(_CHANGE_CUE.pattern + r"|\b(?:put|puts|store|stores|record|records|emit|"
                          r"emits|dump|dumps|persist|persists|build|builds|implement|implements|"
                          r"finish|complete|define|refactor|optimi[sz]e|port to)\b", re.IGNORECASE)
# "Change only `a.py`", "Edit only a.py and b.py": every other file is read-only
_ONLY = re.compile(r"\b(?:(?:change|edit|modify|touch|write)\s+only|only\s+(?:change|edit|modify|"
                   r"touch|write(?: to)?))\b([^.;\n]*(?:\.(?=[\w/`])[^.;\n]*)*)", re.IGNORECASE)
_USE_NEAR = re.compile(r"\b(?:use|uses|using|provided)\W+(?:\w+\W+){0,2}$", re.IGNORECASE)
_STRONG_READ = re.compile(r"\b(?:read(?!-)|reads|reading|see|follow|follows|following|per|according to|"
                          r"described in|documented in|specified in|stated in|port|ports|porting|"
                          r"translate|translates|mirror|mirrors|like|based on)\b", re.IGNORECASE)
_USE_CUE = re.compile(r"\b(?:use|uses|using|used|with|through|via|from|call|calls|calling|keep|"
                      r"keeps|keeping|like|return|returns|returning|list of|instances? of|for|"
                      r"built on|wrap|wraps)\W*(?:\w+\W+){0,2}$", re.IGNORECASE)


def _only_paths(masked: str, files: set[str]) -> set[str]:
    """Project files named after "change only" / "edit only" ("the files named" is not a list)."""
    found: set[str] = set()
    for match in _ONLY.finditer(masked):
        for path in _NEGATED_PATH.findall(match.group(1)):
            path = path.rstrip(".")
            real = path if path in files else next(
                (f for f in files if f.endswith("/" + path) or f == f"src/{path}"), None)
            if real:
                found.add(real)
    return found


def _reads_strongly(masked: str, start: int) -> bool:
    """The last cue before a path is an explicit reading word ("port", "see", "follow")."""
    clause = _local_clause(masked, start)
    if _USE_NEAR.search(clause):
        return True  # "Use the provided `numwords.py`"
    reads = [m.end() for m in _STRONG_READ.finditer(clause)]
    changes = [m.end() for m in _PATH_CHANGE.finditer(clause)]
    return bool(reads) and (not changes or max(reads) > max(changes))


def _only_used(task: str, masked: str, attr: str) -> bool:
    """Every mention of an existing name follows a use cue ("with `chunk_text(...)`", "through
    `knobs.loader.read_source`", "returns `Shape(...)`"): the new code calls it, the task does
    not change it. Mentions are found in the task's code spans (blanked in `masked`)."""
    last = attr.split(".")[-1]
    word = re.compile(r"\b" + re.escape(last) + r"\b")
    spans = [m for m in _SPAN.finditer(task) if word.search(m.group(1))]
    if not spans:
        return False
    return all(_USE_CUE.search(_clause_before(masked, m.start())[-40:]) for m in spans)


def _add(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def decide(route: Route, task: str, project: Path, files: set[str], *,
           num_ctx: int | None = None) -> Route:
    """The decision table for a change request (questions are decided in route_task):

    1. exactly one Python target that exists, nothing else to change, and the writer's prompt
       fits the window (from the file sizes) → direct pipeline;
    2. only JSON/Markdown/CSV/TXT/lock deliverables (deliverable_wide: also an existing one the
       task asks to update, and a helper script it lets the model write and run is not a code
       target; the project has at most MAX_INPUT_FILES files and the deliverable is a file the
       task asks for, not a format it mentions) → deliverable;
    2b. no Python target and exactly one non-Python code/config file to change or create, and
       the task names a runnable check command (`python check.py`, `node --test`) →
       single-file non-Python pipeline (`nonpython_pipeline`);
    2c. no Python target and only CSS/HTML/JS files (and package.json) to change, at most 5,
       fitting the window (web_pipeline) → the non-Python pipeline's front-end config
       (route name `nonpython`, `pipeline` "web"); checks: syntax, rendered page, named commands;
    3. several targets, or none resolved, or a single one the direct pipeline cannot take, in a
       project with visible tests or acceptance checks (and of the size the project pipeline
       takes) → project pipeline;
    4. otherwise → agent loop.
    """
    from . import deliverable_pipeline

    window = num_ctx or DEFAULT_NUM_CTX
    targets = route.targets
    direct_blocker = ""
    if len(targets) == 1 and not route.other_targets:
        target = targets[0]
        path = project / target
        python_files = [f for f in files if f.endswith(".py") and not _TEST_PY.search(f)]
        if not path.is_file():
            direct_blocker = f"{target} does not exist yet"
        elif _CROSS_FILE.search(_NEGATED.sub(" ", strip_code(task))) and len(python_files) > 1:
            direct_blocker = "rename/move reaches other files"
        else:
            size = path.stat().st_size
            route.writer_tokens = writer_tokens(task, size, _input_chars(project, route.inputs))
            if size > MAX_FILE_CHARS:
                direct_blocker = f"{target} is {size} characters (> {MAX_FILE_CHARS})"
            elif route.writer_tokens > int(window * WINDOW_SHARE):
                direct_blocker = (f"writer needs ~{route.writer_tokens} tokens "
                                  f"(> {int(window * WINDOW_SHARE)} of the {window} window)")
        if not direct_blocker and not features.on("direct_pipeline"):
            direct_blocker = "direct_pipeline is off"
        if not direct_blocker:
            route.name = DIRECT
            route.reasons.append(f"one code target: {target}"
                                 + (f"; read-only inputs: {', '.join(route.inputs[:4])}"
                                    if route.inputs else "")
                                 + f"; writer ~{route.writer_tokens} tokens")
            return route
        route.reasons.append("not direct: " + direct_blocker)
    elif len(targets) == 1:
        route.reasons.append("not direct: also changes " + ", ".join(route.other_targets[:4]))
    if not targets and len(route.other_targets) == 1 and features.on("nonpython_pipeline"):
        blocker = _nonpython_blocker(route, task, project, window)
        if not blocker:
            route.name = NONPYTHON
            route.reasons.append(f"one non-Python file: {route.other_targets[0]}, checked with "
                                 f"`{route.check_command}`; writer ~{route.writer_tokens} tokens")
            return route
        route.reasons.append("not nonpython: " + blocker)
    if not targets and route.other_targets and features.on("web_pipeline") \
            and features.on("nonpython_pipeline"):
        web_targets, web_inputs = _web_files(route, task, project)
        blocker = _web_blocker(Route(**{**route.record(), "other_targets": web_targets,
                                        "inputs": web_inputs}), task, project, window)
        if not blocker:
            route.other_targets, route.inputs = web_targets, web_inputs
            size = sum((project / t).stat().st_size for t in web_targets
                       if (project / t).is_file())
            route.writer_tokens = writer_tokens(task, size, _input_chars(project, web_inputs))
            route.name = NONPYTHON
            route.pipeline = "web"
            route.reasons.append("front-end files: " + ", ".join(route.other_targets[:5])
                                 + f"; writer ~{route.writer_tokens} tokens")
            return route
        if any(web_pipeline_takes(t) for t in route.other_targets):
            route.reasons.append("not web: " + blocker)
    deliverables = deliverable_pipeline.targets(project, task)
    wide = features.on("deliverable_wide")
    # deliverable_wide: a helper script the task lets the model write and run (`you may write one
    # helper script at scratch/check.py and run it`) is the pipeline's own script, not a target
    helpers = [t for t in targets if t in route.helpers and not (project / t).exists()] \
        if wide else []
    shown_whole = len(files) <= deliverable_pipeline.MAX_INPUT_FILES  # the writer sees every file
    if (deliverables and features.on("deliverable_pipeline")
            and all(t in helpers for t in targets)
            and all(t in deliverables for t in route.other_targets)
            # wide: an output the task asks for (a change cue), not a format it mentions
            # (`metrics.json` fields in a package spec), and a project the prompt can show whole
            and (not wide or (route.other_targets and shown_whole))):
        route.name = DELIVERABLE
        route.targets = []
        route.reasons.append(("deliverable file(s): " if wide else "new deliverable file(s): ")
                             + ", ".join(deliverables[:4])
                             + (f"; helper script {', '.join(helpers)} is not a target"
                                if helpers else ""))
        return route
    if deliverables and wide and not shown_whole and features.on("deliverable_pipeline"):
        route.reasons.append(f"not deliverable: {len(files)} files "
                             f"(> {deliverable_pipeline.MAX_INPUT_FILES} the writer is shown)")
    code_change = bool(targets) or not route.other_targets
    if code_change and (route.has_tests or route.acceptance):
        if not features.on("project_pipeline"):
            route.reasons.append("project_pipeline is off")
        elif _project_size_ok(project):
            route.name = PROJECT
            route.reasons.append(
                ("several code targets" if len(targets) > 1 else
                 "code target not resolved" if not targets else "one target, not direct")
                + " with " + ("visible tests" if route.has_tests else "acceptance checks"))
            return route
        else:
            route.reasons.append("project too large for the project pipeline")
    elif not code_change:
        route.reasons.append("changes only non-Python files: " + ", ".join(route.other_targets[:4]))
    else:
        route.reasons.append("no visible tests or acceptance checks")
    return route


def _nonpython_blocker(route: Route, task: str, project: Path, window: int) -> str:
    """Why the single-file non-Python pipeline cannot take the task ('' when it can): the target
    must be a code/config file it writes, the task must name a runnable check command, and the
    writer's prompt must fit the window."""
    from . import nonpython_pipeline

    target = route.other_targets[0]
    if not nonpython_pipeline.takes(target):
        return f"{target} is not a code or config file"
    command = nonpython_pipeline.check_command(project, task, target)
    if not command:
        return "the task names no check command to run"
    path = project / target
    size = path.stat().st_size if path.is_file() else 0
    if size > nonpython_pipeline.MAX_FILE_CHARS:
        return f"{target} is {size} characters (> {nonpython_pipeline.MAX_FILE_CHARS})"
    route.writer_tokens = writer_tokens(task, size, _input_chars(project, route.inputs))
    if route.writer_tokens > int(window * WINDOW_SHARE):
        return (f"writer needs ~{route.writer_tokens} tokens "
                f"(> {int(window * WINDOW_SHARE)} of the {window} window)")
    route.check_command = command
    return ""


# "keep `index.html` unchanged", "keep both and the markup unchanged": files the web pipeline's
# writer must not change (the general _protected only reads "do not change ..." clauses)
_KEEP_UNCHANGED = re.compile(r"\bkeep\b((?:[^.;\n]|\.(?=[\w/`]))*?)\b(?:unchanged|as (?:it|they) "
                             r"(?:is|are))\b", re.IGNORECASE)


def _web_files(route: Route, task: str, project: Path) -> tuple[list[str], list[str]]:
    """The web pipeline's targets and inputs: files in a "keep ... unchanged" clause (and HTML
    files when the clause keeps "the markup") become read-only inputs, and a project
    `package.json` the task names becomes a target beside the code (an ES module conversion
    sets its "type")."""
    kept: set[str] = set()
    markup = False
    for match in _KEEP_UNCHANGED.finditer(task):
        clause = match.group(1)
        kept.update(m.group(1).rstrip(".") for m in _NEGATED_PATH.finditer(clause))
        markup = markup or bool(re.search(r"\b(?:markup|html)\b", clause, re.IGNORECASE))
    targets, inputs = [], list(route.inputs)
    for target in route.other_targets:
        if _is_protected(target, kept) or (markup and target.lower().endswith((".html", ".htm"))):
            _add(inputs, target)
        else:
            targets.append(target)
    for path in list(inputs):
        if Path(path).name == "package.json" and (project / path).is_file() \
                and re.search(r"(?<![\w/.-])" + re.escape(path) + r"(?![\w/-])", task) \
                and not _is_protected(path, kept) and targets:
            inputs.remove(path)
            targets.append(path)
    return targets, inputs


def web_pipeline_takes(path: str) -> bool:
    from . import web_pipeline

    return Path(path).suffix.lower() in web_pipeline.WEB


def _web_blocker(route: Route, task: str, project: Path, window: int) -> str:
    """Why the front-end config of the non-Python pipeline (web_pipeline) cannot take the task
    ('' when it can): only CSS/HTML/JS files (and package.json) to change, at most
    web_pipeline.MAX_TARGETS, and the writer's prompt fits the window. A deliverable the task
    asks for (a new report.html) stays with the deliverable pipeline."""
    from . import deliverable_pipeline, web_pipeline

    targets = route.other_targets
    blocker = web_pipeline.blocker(project, targets)
    if blocker:
        return blocker
    deliverables = deliverable_pipeline.targets(project, task)
    if deliverables and all(t in deliverables and not (project / t).exists() for t in targets):
        return "only new deliverable files"
    size = sum((project / t).stat().st_size for t in targets if (project / t).is_file())
    route.writer_tokens = writer_tokens(task, size, _input_chars(project, route.inputs))
    if route.writer_tokens > int(window * WINDOW_SHARE):
        return (f"writer needs ~{route.writer_tokens} tokens "
                f"(> {int(window * WINDOW_SHARE)} of the {window} window)")
    return ""


def _project_size_ok(project: Path) -> bool:
    """Whether the project is within the project pipeline's file limits."""
    from . import project_pipeline

    if features.on("large_projects"):
        from .tools import WALK_LIMIT

        files = list(_project_files(project))
        python = [p for p in files if p.suffix == ".py"]
        return bool(python) and len(python) <= project_pipeline.LARGE_MAX_PYTHON_FILES \
            and len(files) < WALK_LIMIT
    files = list(_project_files(project, limit=project_pipeline.MAX_PROJECT_FILES + 1))
    return bool(files) and len(files) <= project_pipeline.MAX_PROJECT_FILES
