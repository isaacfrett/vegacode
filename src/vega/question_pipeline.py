"""Question fast path: answer a question about the project in one model call.

Chat response time baseline (2026-10-14): a one-line question took ~9 s and 3 model calls of
~3,000 prompt tokens each (choose a tool, read the file, answer), while an edit through the
direct pipeline took ~5 s in one short call. This path finds the relevant code itself (files the
question names, then the search index's best chunks), and asks once with a short prompt. When
the excerpts do not hold the answer the model replies NOT_FOUND and the turn falls back to the
agent loop, which can search further.

Switch: `question_pipeline` (opt-in until measured on the read-only cases and `vega latency`).

Run the code (switch `question_run_code`, on; 2026-10-17): for "what does f return / what would
X be for input Y", reading the code is where a 7B slips (it read `500 < 500` as true). When the
question's inputs are literals (a call written out, `f(100, 'basic')`, or the one number in the
question, `an order of exactly 50 items`) and the functions resolve in the project (acceptance's resolver, plus the project
functions they call), Vega calls them on those inputs in the sandbox and gives the model the
actual results as evidence; the model still writes the answer.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import debuglog, features, teaching, verbs
from .pipeline import _QUESTION, _WRITE_VERBS, ask_complete, with_context

if TYPE_CHECKING:
    from .agent import CodingAgent

NOT_FOUND = "NOT_FOUND"
EXCERPT_CHARS = 7000  # total code shown (~2,000 tokens)
FILE_CHARS = 4000  # a file the question names is shown whole up to this size
HITS = 8
MAP_CHARS = 1500
_READ_ONLY = re.compile(r"\b(do not|don't|without) (modify|change|edit)|\bonly report\b|\btell me\b|"
                        r"\breport\b|\bexplain\b|\bdescribe\b|\bfind\b|\blocate\b", re.IGNORECASE)
_ACTION = re.compile(r"\b(fix|rename|refactor|move|remove|delete|update|change|make|run|install)\b"
                     r"(?![^.]*\b(do not|don't)\b)", re.IGNORECASE)
_PATH = re.compile(r"[\w./-]+\.(?:py|toml|cfg|ini|json|ya?ml|md|txt|js|ts)\b")
QA_SYSTEM = (
    "You answer questions about a code project from excerpts of its files. Answer directly and "
    "briefly, giving the exact value, name or path asked for, and name the file you read it from. "
    "Read the code carefully: follow defaults, branches and calls to work out values. If the "
    f"excerpts do not contain what is needed to answer, reply exactly {NOT_FOUND} and nothing else."
)


# Something in the question that only the project's code can answer: a path, a call, a backticked or
# code-shaped name, or words that point at this codebase.
_CODE_ANCHOR = re.compile(
    r"`[^`]+`|\b[\w./-]+\.(?:py|toml|json|ya?ml|cfg|ini|md|txt|js|ts|sh|sql)\b|\b\w+\((?:[^)]*\))?"
    r"|\b[a-z]+_[a-z0-9_]+\b|\b[A-Z][a-z0-9]+[A-Z]\w*\b|\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b"
    r"|(?i:\b(?:this|the|our|my)\s+(?:[\w-]+\s+){0,2}(?:code|codebase|project|repo|repository|app|module"
    r"|modules|package|file|files|function|functions|class|classes|method|methods|test|tests|config|cli"
    r"|script|loader|guide|docs|documentation|readme|logic|api|endpoint|schema|model|models)\b)"
    r"|(?i:\b(?:defined|implemented|declared|imported|raised|returned|handled)\b)"
    r"|(?i:\b(?:environment\s+variables?|env\s+vars?|settings?|options?|flags?|defaults?|ports?|commands?"
    r"|dependenc(?:y|ies)|version)\b)"
    r"|(?i:\b(?:which|what)\s+(?:file|files|module|function|class|method|test)\b|\bwhere\s+(?:is|are|does|do)\b)",
)
# Questions to Vega about itself or small talk: a conversation, not a lookup in the code index.
_ABOUT_VEGA = re.compile(
    r"^\W*(?:hi|hello|hey|thanks|thank you|help)\W*$|\b(?:what|how)\s+can\s+you\b|\bwho\s+are\s+you\b"
    r"|\bwhat\s+are\s+you\b|\byour\s+(?:capabilities|abilities|features|name)\b|\bhow\s+do\s+i\s+use\s+you\b",
    re.IGNORECASE,
)


# Whole-project questions (review, overview, what to improve): a few index excerpts cannot answer
# them — the live "what can be improved about this project?" got a critique of whichever file
# matched. The agent loop explores instead. A specific anchor (path, call, code name) still counts.
_BROAD = re.compile(
    r"\b(?:improve[ds]?|improvements?|review|weak(?:ness(?:es)?)?|strengths?|overview|architecture"
    r"|summari[sz]e|summary|recog\w*|tell me about|walk me through|assess|critique|audit"
    r"|what\s+(?:does|is)\s+(?:this|the)\s+(?:project|repo|repository|codebase|app)\b)",
    re.IGNORECASE,
)
_SPECIFIC = re.compile(
    r"`[^`]+`|\b[\w./-]+\.(?:py|toml|json|ya?ml|cfg|ini|md|txt|js|ts|sh|sql)\b|\b\w+\("
    r"|\b[a-z]+_[a-z0-9_]+\b|\b[A-Z][a-z0-9]+[A-Z]\w*\b|\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b",
)


def is_broad(task: str) -> bool:
    """A question about the whole project with nothing specific named in it."""
    return bool(_BROAD.search(task)) and not _SPECIFIC.search(task)


def is_question(task: str) -> bool:
    """A question or read-only request that asks for no change."""
    asks = bool(_QUESTION.search(task) or _READ_ONLY.search(task) or _ABOUT_VEGA.search(task))
    # in request position only: "why is add wrong?" asks for no change (see verbs)
    changes = bool(verbs.requested(_WRITE_VERBS, task)
                   or verbs.requested(_ACTION, _strip_negations(task)))
    return asks and not changes


def is_conversation(task: str) -> bool:
    """A question to Vega itself, small talk, or a general question with nothing in the project to
    look at: answered directly, never by inspecting or changing files."""
    return is_question(task) and (bool(_ABOUT_VEGA.search(task)) or not _CODE_ANCHOR.search(task))


def _strip_negations(task: str) -> str:
    """'Do not modify anything' is not a request to modify: drop negated clauses."""
    return re.sub(r"\b(do not|don't|never|without)\b[^.;]*", "", task, flags=re.IGNORECASE)


def _numbered(lines: list[str], start: int) -> str:
    return "\n".join(f"{start + i}: {line}" for i, line in enumerate(lines))


def excerpts(project: Path, task: str) -> str:
    """Files the question names (whole, if small), then the best search chunks, then a short map."""
    from .index import build_index

    index = build_index(project)
    parts: list[str] = []
    used = 0
    seen: set[tuple[str, int]] = set()
    files = {entry.path: entry for entry in index.files}
    for named in dict.fromkeys(_PATH.findall(task)):
        match = next((path for path in files if path == named or path.endswith("/" + named)), None)
        if match is None:
            continue
        text = "\n".join(files[match].lines)
        if len(text) > FILE_CHARS:
            continue
        parts.append(f"{match} (whole file):\n{_numbered(files[match].lines, 1)}")
        used += len(text)
        seen.add((match, 0))
    for hit in index.search(task, limit=HITS):
        if (hit.path, 0) in seen or (hit.path, hit.start_line) in seen:
            continue
        lines = files[hit.path].lines[hit.start_line - 1:hit.end_line]
        block = f"{hit.path} lines {hit.start_line}-{hit.end_line}:\n{_numbered(lines, hit.start_line)}"
        if used + len(block) > EXCERPT_CHARS:
            break
        parts.append(block)
        used += len(block)
        seen.add((hit.path, hit.start_line))
    parts.append("Project map:\n" + index.repo_map(max_chars=MAP_CHARS))
    return "\n\n".join(parts)


# -- question_run_code: run the function on the question's inputs ---------------------------------

MAX_CALLS = 6
CALL_DEPTH = 2  # the named functions, the project functions they call, and theirs
CALL_SECONDS = 5
RESULT_CHARS = 300
# a question (it ends in "?") that asks for a value: "what does f(x) return?", "what surcharge
# does a 500 km trip get?", "what would the total be for input 3?"
_EVALUATE = re.compile(
    r"(?:\b(?:what|which|how\s+much|how\s+many)\b[^?]*?\b(?:returns?|returned|gets?|gives?|be|"
    r"produces?|outputs?|yields?|evaluates?|computes?|comes?\s+out|charges?|costs?)\b"
    r"|\bfor\s+(?:the\s+|an?\s+)?inputs?\b|\bwhen\s+(?:called|given)\b|\bresult\s+of\b)"
    r"(?=[^?]*\?)", re.IGNORECASE)
_NAMED_CALL = re.compile(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\(([^()`]*)\)")
_NAMED = re.compile(r"`([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?:\(\))?`")
_SPAN = re.compile(r"`[^`]*`")
_LITERAL_SPAN = re.compile(r"`-?\d+(?:\.\d+)?`")
_NUMBER = re.compile(r"(?<![\w.,/:-])-?\d+(?:\.\d+)?(?![\w.,/:-]*\d)(?![\w(])")
_NOT_INPUT = re.compile(r"\b(?i:line|lines|version|v|python|step|steps|round|rounds|top|first|"
                        r"last)\s*$|\b[A-Z]\w*\s+$")  # ... or a name's version: SQLite 3.50
_CALL_MARKER = "<<<VEGA-CALL>>>"
CALL_SCRIPT = r'''
import ast, importlib, json, signal, sys
sys.path[:0] = ["src", "."]
def alarm(signum, frame):
    raise TimeoutError("took longer than SECONDS s")
signal.signal(signal.SIGALRM, alarm)
for call in json.loads(sys.argv[1]):
    entry = {"call": call["shown"]}
    try:
        obj = importlib.import_module(call["module"])
        for part in call["attr"].split("."):
            obj = getattr(obj, part)
        args = ast.literal_eval(call["args"])
        signal.alarm(SECONDS)
        try:
            value = obj(*args)
        finally:
            signal.alarm(0)
        entry["result"] = repr(value)[:RESULT_CHARS]
    except BaseException as exc:
        entry["error"] = (type(exc).__name__ + ": " + str(exc))[:RESULT_CHARS]
    print(MARKER + json.dumps(entry), flush=True)
'''.replace("SECONDS", str(CALL_SECONDS)).replace("RESULT_CHARS", str(RESULT_CHARS)).replace(
    "MARKER", repr(_CALL_MARKER))


def _literal_args(text: str) -> str | None:
    """`100, 'gold'` → "(100, 'gold',)"; None when they are not all literals (or there are none)."""
    if not text.strip():
        return None
    source = f"({text},)"
    try:
        ast.literal_eval(source)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    return source


def question_numbers(task: str) -> list[str]:
    """Distinct numbers in the question's prose (not in code spans or answer templates such as
    `<percent>%`, not line numbers or versions): the inputs it states."""
    prose = _SPAN.sub(lambda m: m.group(0).strip("`").center(len(m.group(0)))
                      if _LITERAL_SPAN.fullmatch(m.group(0)) else " " * len(m.group(0)), task)
    found: list[str] = []
    for match in _NUMBER.finditer(prose):
        if _NOT_INPUT.search(prose[max(0, match.start() - 12):match.start()]):
            continue
        if match.group(0) not in found:
            found.append(match.group(0))
    return found


def _one_input(node: ast.AST) -> bool:
    """A plain (not async) function that takes exactly one argument without a default."""
    if not isinstance(node, ast.FunctionDef):
        return False
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    required = len(positional) - len(args.defaults)
    kw_required = sum(1 for default in args.kw_defaults if default is None)
    return required == 1 and not kw_required


def _callees(project: Path, path: str, node: ast.AST, definitions: dict[str, list[Any]],
             paths: set[str]) -> list[tuple[str, str]]:
    """(file, name) of the project's top-level functions that `node` calls by plain name: in
    its own module, or imported into it from another project module."""
    from . import acceptance

    try:
        tree = ast.parse((project / path).read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return []
    imported: dict[str, str] = {}
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom) and statement.module and not statement.level:
            parts = statement.module.split(".")
            module = acceptance._module_file(project, parts, paths)
            if module is not None and module[1] == len(parts):
                for alias in statement.names:
                    imported[alias.asname or alias.name] = module[0]
    found: list[tuple[str, str]] = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        name = call.func.id
        local = any(d.path == path for d in definitions.get(name, []))
        home = path if local else imported.get(name)
        if home and (home, name) not in found:
            found.append((home, name))
    return found


def planned_calls(project: Path, task: str) -> list[dict[str, str]]:
    """The calls to run for an "evaluate" question: each call written out with literal arguments,
    else the question's one number given to each named function that takes one argument and to
    the project functions it calls (CALL_DEPTH deep). [] when the question asks for no value,
    states no literal input (or several numbers), or names nothing that resolves in the project."""
    from . import acceptance

    if not _EVALUATE.search(task):
        return []
    definitions, paths, _library = acceptance._definitions(project)
    named_files = acceptance._named_py_paths(task, paths)

    def resolve(dotted: str, start: int, end: int) -> tuple[str, str] | None:
        found = acceptance._resolve(project, dotted, task, start, end, definitions, paths,
                                    named_files, call=True)
        if found is None or "." in found[1]:
            return None  # a method needs an instance: not called
        if not any(d.path == found[0] for d in definitions.get(found[1], [])):
            return None  # not defined: nothing to run
        return found

    calls: list[dict[str, str]] = []

    def add(path: str, name: str, args: str) -> None:
        module = acceptance.module_name(path)
        shown = f"{module}.{name}({args[1:-2]})"
        if len(calls) < MAX_CALLS and all(c["shown"] != shown for c in calls):
            calls.append({"module": module, "attr": name, "args": args, "shown": shown})

    for match in _NAMED_CALL.finditer(task):
        literal = _literal_args(match.group(2))
        found = resolve(match.group(1), match.start(1), match.end(1)) if literal else None
        if found is not None and literal is not None:
            add(*found, literal)
    if calls:
        return calls
    numbers = question_numbers(task)
    if len(numbers) != 1:
        return []  # no stated input, or several: which goes where is not known
    literal = f"({numbers[0]},)"
    queue: list[tuple[str, str, int]] = []
    for pattern in (_NAMED_CALL, _NAMED):
        for match in pattern.finditer(task):
            found = resolve(match.group(1), match.start(1), match.end(1))
            if found is not None and (*found, 0) not in queue:
                queue.append((*found, 0))
    seen: set[tuple[str, str]] = set()
    while queue:
        path, name, depth = queue.pop(0)
        if (path, name) in seen:
            continue
        seen.add((path, name))
        node = next((d.node for d in definitions.get(name, []) if d.path == path), None)
        if node is None:
            continue
        if _one_input(node):
            add(path, name, literal)
        if depth < CALL_DEPTH:
            queue += [(home, callee, depth + 1)
                      for home, callee in _callees(project, path, node, definitions, paths)]
    return calls


def run_calls(project: Path, calls: list[dict[str, str]]) -> list[dict[str, str]]:
    """Run the calls in the sandbox: [{call, result | error}] for each that ran."""
    from . import sandbox

    command = f"python -c {shlex.quote(CALL_SCRIPT)} {shlex.quote(json.dumps(calls))}"
    output = sandbox.run_check(command, project, timeout=CALL_SECONDS * len(calls) + 20)
    results = []
    for line in output.splitlines():
        if line.startswith(_CALL_MARKER):
            try:
                results.append(json.loads(line[len(_CALL_MARKER):]))
            except ValueError:
                continue
    return results


def call_evidence(project: Path, task: str) -> str:
    """The evidence text for the model ('' when there is nothing to run, or nothing ran)."""
    from . import acceptance

    if acceptance._NO_COMMANDS.search(task) and not features.on("acceptance_ignores_no_commands"):
        return ""
    calls = planned_calls(project, task)
    if not calls:
        return ""
    results = run_calls(project, calls)
    debuglog.record("stage", stage="question_pipeline.run_code",
                    calls=[c["shown"] for c in calls], results=results)
    if not results:
        return ""
    lines = [f"- {entry['call']} returned {entry['result']}" if "result" in entry
             else f"- {entry['call']} raised {entry.get('error', 'an error')}" for entry in results]
    return ("Vega ran the project's code on the question's inputs in a sandbox. Actual results "
            "(this is what the code does; trust it over reading the code by eye):\n"
            + "\n".join(lines))


# question_code_guard (2026-10-20): a change request came here on the noun "report", the model
# answered with the code in prose, and no file was written. An answer that is code for a task
# that states requirements is a change, not an answer.
_CODE_FENCE = re.compile(r"^[ \t]*```[^\n]*\n(.*?)(?:^[ \t]*```|\Z)", re.MULTILINE | re.DOTALL)
_DEFINITION = re.compile(r"^\s*(?:async\s+def|def|class)\s+\w", re.MULTILINE)
CODE_LINES = 6


def code_for_change(task: str, answer: str) -> bool:
    """The answer holds a code block (a def/class, or CODE_LINES+ lines) and the task states
    "must" requirements or asks for a change: a change the question pipeline cannot make."""
    from . import routing

    blocks = [m.group(1) for m in _CODE_FENCE.finditer(answer)]
    if not any(_DEFINITION.search(block) or len([ln for ln in block.splitlines() if ln.strip()])
               >= CODE_LINES for block in blocks):
        return False
    return bool(routing.requirements(task) or routing.change_verbs(task))


def run(agent: CodingAgent, task: str) -> str | None:
    """Answer a task routing.route_task sent here; None falls back to the agent loop."""
    agent._emit("notice", message="Answering from the code index")
    try:
        shown = excerpts(agent.project, task)
    except Exception as exc:  # noqa: BLE001 - fall back to the agent loop
        debuglog.record("stage", stage="question_pipeline.excerpts_failed", error=str(exc))
        return None
    evidence = ""
    if features.on("question_run_code"):
        try:
            evidence = call_evidence(agent.project, task)
        except Exception as exc:  # noqa: BLE001 - no evidence is the old behaviour
            debuglog.record("stage", stage="question_pipeline.run_code_failed", error=str(exc))
        if evidence:
            agent._emit("notice", message="Ran the code on the question's inputs")
    user = f"Question: {task}\n\nExcerpts:\n{shown}"
    if evidence:
        user += f"\n\n{evidence}"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": with_context(agent, QA_SYSTEM)},
        {"role": "user", "content": user},
    ]
    try:
        answer = ask_complete(agent, messages).strip()
    except Exception as exc:  # noqa: BLE001 - fall back to the agent loop
        agent._emit("notice", message=f"Fast answer failed ({exc}); searching instead.")
        return None
    debuglog.record("stage", stage="question_pipeline.answer", found=NOT_FOUND not in answer[:40],
                    excerpt_chars=len(shown))
    if not answer or answer.startswith(NOT_FOUND) or answer == NOT_FOUND:
        agent._emit("notice", message="Not in the obvious places; searching the project.")
        return None
    if features.on("question_code_guard") and code_for_change(task, answer):
        debuglog.record("stage", stage="question_pipeline.code_for_change")
        agent._emit("notice", message="The answer is code for a change; making the change instead.")
        return None
    agent.messages += [{"role": "user", "content": task}, {"role": "assistant", "content": answer}]
    if getattr(agent, "teach", False):  # teach mode: where the answer came from (real excerpts)
        note = teaching.question_note(shown)
        if note:
            return f"{answer}\n\n{note}"
    return answer
