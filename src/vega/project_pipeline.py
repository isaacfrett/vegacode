"""Project pipeline for multi-file change tasks: plan → write files → run tests → repair.

pyprojects (2026-10-14): in the agent loop the 7B spent ~90 steps searching for the code it had
been asked to write and edited nothing. The direct pipeline (pipeline.py) showed the fix for
single files: a fixed sequence with whole-unit writes in plain code blocks and real test output
as feedback. This applies it to projects:

1. Plan: from the task and a compact repo map, the model names the files to change or create and
   what each needs (one JSON reply, with a free-text `analysis` field first).
2. Write: each planned file is written in full as a ```code block``` (free text, no JSON), merged
   by whole definitions for Python, through the normal write path (review, undo, parse guard).
3. Check: the project's tests run in the sandbox (pytest when there are test files).
4. Repair: failing output goes back; the model names the files to fix and rewrites them; at most
   REPAIRS rounds, keeping the version with the fewest failing tests.

Steps 2-4 run on the shared write → check → repair engine (write_loop.run_loop) as a config with
several targets per round (`_Rounds`): one writer call per planned file, the ranking tuple as its
`score`, the replan as its `between` step; the saves, the best-round restore and the round loop
are the engine's.

With `spec_tests` on, step 1b writes pytest tests from the task before any code. The project's
own tests only cover existing behaviour: in the goals A/B (2026-10-14) tasks often ended with
every project test passing and the hidden tests failing, so "tests pass" never confirmed the
feature. The spec tests live in .vega-spec/ (pytest skips dot-directories, so the project's runs
never collect them), are kept only if they fail before the change, rank below the project's tests
when choosing the best round (a 7B's tests can be wrong), and are always removed at the end.

routing.route_task decides which tasks come here (a code change with visible tests or acceptance
checks, in a project small enough). Switch: `project_pipeline`.

Large projects (opt-in switch `large_projects`, 2026-10-15): selfbuild (6 real Vega bugs) got no
help from the pipelines: Vega's own repo is over MAX_PROJECT_FILES and its core modules are
30-120k characters, over FILE_CHARS, so both pipelines declined. With the switch on:

- projects with up to LARGE_MAX_PYTHON_FILES Python files are taken (any number of other files,
  up to the walk limit);
- in a project over MAX_PROJECT_FILES files the planner's map is focused on the task: the files
  holding the best search hits with their symbols, then the other Python modules by path;
- a planned Python file over FILE_CHARS is shown as an outline (every top-level def/class with
  its line range, methods under their class) plus the complete source of the definitions most
  relevant to the change (named in the plan item's change, then in the task, then the file's best
  search hits) up to FOCUS_CHARS; the writer replies with only the changed definitions, a method
  inside its `class Name:` header, and pipeline.merge_code merges classes method by method;
- checks run with LARGE_TEST_TIMEOUT (Vega's suite takes ~85 s in the sandbox, over the 60 s
  default). The full suite runs every round (running only "related" tests would miss callers of
  a changed method in other modules), and once before the change: tests failing then (two of
  Vega's fail only inside the sandbox) rank after newly failing tests. They are still shown,
  labelled, since the task may be to fix them, and keep the repairs going only while a round
  fixes some of them; the answer says when any still fail.

Pipeline signals (switch `pipeline_signals`, on; redesign C, 2026-10-16):
the rounds answer "does this meet the task", not "do the old tests still pass". A round is done
only when the tests and acceptance pass; a test file or task module that will not import ranks
below any number of failing tests; the writer replies with one labelled file per step and the
block for the planned path is taken (a reply only for other files goes back once with the
reason); "no change" plan items are dropped and an empty plan is retried once; every write gets
the atomic check (edits.check) and a refusal is the writer's next repair input.

Added 2026-10-17, each behind its own switch, on by default:

- `project_inputs`: the route's non-Python read-only inputs (a commit list and a release guide
  that were never shown left the version and the entries to be guessed) go to the planner
  and every writer, cut to fit the window (INPUT_CHARS, and what is left of WRITER_PROMPT_CHARS).
- `project_large_files`: a Python file over FILE_CHARS is written through the large_projects
  focused view even with that switch off (a module of over 100 KB was "too large to
  rewrite whole", left unchanged, and the answer still claimed the change and passing tests).
- `project_honest_answer`: the answer names the files actually written (compared with their
  content before the change) and every planned file left unchanged with the reason; with nothing
  written it never says the tests pass.
- `project_no_tests`: pytest exit code 5 (no tests collected) is no tests, not one failing test
  (repair rounds were spent on it).
- `project_run_script`: a script the task runs to produce an output (`python report.py` writes
  out.json) is run in the sandbox after each round's writes and the files it produced are
  saved; the outputs are dropped from the plan, so the writer never hand-writes them.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import functools
import json
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import (
    acceptance,
    check_feedback,
    debuglog,
    edits,
    features,
    sandbox,
    task_checks,
    teaching,
    write_loop,
)
from .codemerge import _insert_imports, _parses, merge_code
from .pipeline import ask_complete, with_context
from .tools import _project_files, execute

if TYPE_CHECKING:
    from .agent import CodingAgent

REPAIRS = 3
MAX_PLAN_FILES = 6
MAX_PROJECT_FILES = 80
MAP_CHARS = 5000
FILE_CHARS = 9000  # a file larger than this is not rewritten whole (falls back to the agent)
# with `large_projects` on
LARGE_MAX_PYTHON_FILES = 600
LARGE_MAP_CHARS = 8000
LARGE_MAP_FILES = 6  # files shown with their symbols in a large project's planner map
LARGE_TEST_TIMEOUT = 300  # seconds for one sandboxed check
FOCUS_CHARS = 6000  # source of the relevant definitions shown for a large file
OUTLINE_CHARS = 8000
WINDOW_OUTLINE_CHARS = 2500  # project_large_files without large_projects: the 8k window
WINDOW_FOCUS_CHARS = 3500
IMPORT_CHARS = 1500
RELATED_CHARS = 2500
TEST_OUTPUT_LINES = 40
OPEN_SHOWN = 8  # open items listed in the final answer
# project_inputs: read-only inputs shown to the planner and the writer
INPUT_CHARS = 6000  # all inputs together, at most
INPUT_MIN_CHARS = 1200  # a writer prompt that is already long still gets this much of them
WRITER_PROMPT_CHARS = 16000  # ~4.5k tokens: the 8k window keeps room for the reply
NO_TESTS = 5  # pytest's exit code when it collected no tests
# project_run_script: a script the task runs, and the files it writes
SCRIPT_MARKER = "<<<VEGA-SCRIPT-OUTPUTS>>>"
SCRIPT_RUNNER = r'''
import json, os, runpy, sys
from pathlib import Path
targets, script = json.loads(sys.argv[1]), sys.argv[2]
for target in targets:  # the sandbox's copy: what exists afterwards, the script wrote
    Path(target).unlink(missing_ok=True)
sys.argv = [script, *sys.argv[3:]]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
sys.stdout.flush()
print(MARKER)
print(json.dumps({"exit": code, "files": {t: Path(t).read_text(encoding="utf-8")
                                          for t in targets if Path(t).is_file()}}))
'''.replace("MARKER", repr(SCRIPT_MARKER))
_CODE_SUFFIXES = {".py", ".pyi"}


@dataclass
class _Extras:
    """What the 2026-10-17 switches add to one run: the route's read-only inputs, the scripts
    the task runs and the outputs they write, and what the answer needs to be exact."""

    inputs: list[str] = field(default_factory=list)
    scripts: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    originals: dict[str, str | None] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)  # path -> its last write's result
    no_tests: bool = False  # the last test run collected no tests (pytest exit code 5)
    produced: dict[str, str] = field(default_factory=dict)  # output -> the script that wrote it
    # plan_contradicting_tests: visible test files that call a signature the task replaces
    call_tests: dict[str, CallTest] = field(default_factory=dict)
_ASKS_FOR_TESTS = re.compile(r"\b(add|write|update|create)\b[^.]*\btests?\b|\bregression test",
                             re.IGNORECASE)
_TEST_FILE = re.compile(r"(^|/)(tests?/.*\.py|test_[^/]*\.py|[^/]*_test\.py)$")

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "change": {"type": "string"}},
                "required": ["path", "change"],
            },
        },
    },
    "required": ["analysis", "files"],
}
PLANNER_SYSTEM = (
    "You plan code changes for a Python project. Read the task and the project map. First write a "
    "short analysis of what must change and where (which modules, functions, classes, and any "
    "new files). Then list every file to change or create, each with a precise description of "
    "the change. Include the files that must change for the feature to work end to end (models, "
    "services, CLI, configuration). Put code exactly where the task names it: if the task says "
    "`pkg.module.func`, it goes in that module (pkg/module.py), not in a new file. Do not list "
    "test files unless the task asks for tests."
)
WRITER_SYSTEM = (
    "You are an expert Python engineer. Make the requested change correctly and consistently with "
    "the rest of the project. For an existing Python file, reply with only the top-level "
    "functions and classes you add or change, each complete, plus the imports they need, in one "
    "code block: everything you leave out stays exactly as it is, so do not repeat unchanged "
    "code. For a new file, or a non-Python file, reply with the whole file. Nothing else."
)
# pipeline_signals: one file per reply, labelled, so the block for the planned file is found by
# its path (otherwise cli.py's source could land in pyproject.toml as the longest block)
WRITER_FORMAT = (
    " Write exactly one file per reply: the file this step names; every other file gets its own "
    "step. Put its path on the line before the code block, like\nFile: pkg/module.py\n```python\n"
    "<code>\n```"
)


SPEC_DIR = ".vega-spec"
SPEC_FILE = f"{SPEC_DIR}/test_task_spec.py"
SPEC_SYSTEM = (
    "You write pytest tests for a change that has not been made yet. Test only the new behaviour "
    "the task asks for, using exactly the module paths, names, signatures, outputs and errors the "
    "task states (or that already exist in the project). Do not guess at anything the task leaves "
    "open. Small, independent, deterministic tests; use the tmp_path fixture for files; no "
    "network. Reply with one ```python code block containing the whole test file."
)


def _test_files(project: Path) -> list[Path]:
    return [p for p in _project_files(project)
            if p.suffix == ".py" and _TEST_FILE.search(p.relative_to(project).as_posix())]


def _ask(agent: CodingAgent, messages: list[dict[str, Any]], **extra: Any) -> str:
    if extra.get("response_format"):  # a JSON plan: short, and continuing would break the schema
        reply = agent.client.chat.completions.create(model=agent.model, messages=messages,
                                                     **agent.sampling, **extra)
        return reply.choices[0].message.content or ""
    return ask_complete(agent, messages, **extra)


# pipeline_signals: plan items that say the file needs no change are not written (seen
# 2026-10-16: "No changes required" for __init__.py, which the writer then filled with a
# second copy of the domain classes). A text that leads with a no-change phrase is dropped; one
# that says it elsewhere, or starts with "keep", only when nothing else in it asks for a change.
_NO_CHANGE = re.compile(
    r"\b(?:no (?:code |direct |real |further |other |functional )?(?:changes?|modifications?|"
    r"edits?|updates?)(?: (?:is |are )?(?:needed|required|necessary))?|unchanged|"
    r"nothing to change|(?:does not|doesn't|do not|don't) need (?:to )?(?:be )?(?:changed?|"
    r"modified|edited))\b", re.IGNORECASE)
_LEADS_NO_CHANGE = re.compile(r"^\W*(?:no\b|unchanged|nothing to change|n/a\b|none\b|"
                              r"(?:this |the )?file (?:is |stays |remains )?unchanged)",
                              re.IGNORECASE)
_KEEP = re.compile(r"^\W*(?:keep|leave|retain|preserve)\b", re.IGNORECASE)
_CHANGE_VERB = re.compile(
    r"\b(?:add|adds|adding|change|update|modify|implement|rewrite|fix|replace|remove|delete|"
    r"rename|create|move|refactor|define|export|re-export|write|convert|use|make|set|"
    r"extract|split|raise|return|handle|register|expose|wire|declare|configure)\b",
    re.IGNORECASE)


def no_change(change: str) -> bool:
    """A plan item's change text says the file needs no change."""
    text = change.strip()
    if not text:
        return False
    if _LEADS_NO_CHANGE.match(text) and _NO_CHANGE.search(text.split(".")[0] + "."):
        return True  # "No changes needed; only re-export existing modules if necessary."
    if _NO_CHANGE.search(text) and not _CHANGE_VERB.search(_NO_CHANGE.sub(" ", text)):
        return True
    return bool(_KEEP.match(text)) and not _CHANGE_VERB.search(text)


def _plan(agent: CodingAgent, task: str, repo_map: str, failure: str = "", *, note: str = "",
          dropped: list[dict[str, str]] | None = None, inputs: str = "",
          outputs: list[str] | None = None,
          required: list[dict[str, str]] | None = None,
          call_tests: dict[str, CallTest] | None = None) -> list[dict[str, str]]:
    """`inputs`: the read-only inputs' text (project_inputs); `outputs`: files a script the task
    runs writes (project_run_script), never planned as writes; `required`: the route's targets
    (project_route_targets), named to the planner as files that must be in the plan;
    `call_tests`: test files plan_contradicting_tests lets through (with its narrow change)."""
    user = f"Task:\n{task}\n\nProject map:\n{repo_map}"
    if inputs:
        user += inputs
    if required:
        user += ("\n\nThese files must change for this task (Vega found them from the task's "
                 "own words), so list each of them, plus any other file the change needs:\n"
                 + "\n".join(f"- {item['path']}" + (f" ({item['why']})" if item.get("why") else "")
                             for item in required))
    if outputs:
        user += ("\n\nVega runs the task's script after writing the code and saves what it "
                 f"writes ({', '.join(outputs)}): do not list those files.")
    if failure:
        user += ("\n\nThe last attempt was written, but the project's tests fail:\n"
                 f"{failure}\n\nList only the files that must change to fix these failures.")
    if note:
        user += f"\n\n{note}"
    text = _ask(agent, [{"role": "system", "content": with_context(agent, PLANNER_SYSTEM)},
                        {"role": "user", "content": user}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "plan", "schema": PLAN_SCHEMA}})
    try:
        plan = json.loads(text)
    except ValueError:
        return []
    files = []
    tests_wanted = bool(_ASKS_FOR_TESTS.search(task))
    for item in plan.get("files", []):
        path = str(item.get("path", "")).strip().lstrip("./")
        if not path or ".." in Path(path).parts:
            continue
        if call_tests and path in call_tests:  # only its broken calls, never its assertions
            if not any(f["path"] == path for f in files):
                files.append({"path": path, "change": call_tests[path].change()})
            continue
        if _TEST_FILE.search(path) and not tests_wanted:
            continue  # the tests are the yardstick: never rewrite them unless asked to
        same = next((f for f in files if f["path"] == path), None)
        if same is not None:  # one write per file: combine repeated entries
            same["change"] += "; " + str(item.get("change", ""))
            continue
        files.append({"path": path, "change": str(item.get("change", ""))})
    if features.on("pipeline_signals"):
        kept = [f for f in files if not no_change(f["change"])]
        if len(kept) < len(files):
            gone = [f for f in files if f not in kept]
            if dropped is not None:
                dropped += gone
            debuglog.record("stage", stage="project_pipeline.plan_no_change",
                            dropped=[f"{f['path']}: {f['change'][:120]}" for f in gone])
        files = kept
    if outputs:  # the script writes these; a hand-written copy would only be overwritten
        files = [f for f in files if f["path"] not in outputs]
    return files[:MAX_PLAN_FILES]


# -- project_route_targets: the files routing resolved ---------------------------------------------
# Why (2026-10-20): routing finds the CLI file a backticked command runs (app/cli.py for
# `app export --csv`) and the file of a class named in prose (app/client.py for ApiClient), and
# sends the task here for them, but the planner planned from
# the repo map alone, so whether those files were written was its own choice.

_ALSO_TARGET = re.compile(r"^also a target: (.*)$")
_TARGET_WHY = re.compile(r"([^,()]+?) \(([^)]*)\)")


def route_targets(project: Path, route: dict[str, Any], task: str) -> list[dict[str, str]]:
    """The route's Python targets as plan seeds, [{path, why}], when routing widened a one-target
    route with a file it found (route_cli_target / route_prose_class: an "also a target" reason);
    why is routing's reason for an added file ('runs the backticked command'), else '' (the
    direct route's own target). [] for any other route: a plain project route's targets can
    include files the task says not to change ("do not change the
    module", or a file under "Do not change any other file"), so forcing
    them would write what the planner rightly left alone (offline replay, 2026-10-20). Test
    files only when the task asks for tests; never a path outside the project."""
    whys: dict[str, str] = {}
    for reason in route.get("reasons") or []:
        match = _ALSO_TARGET.match(str(reason))
        if match:
            for part in _TARGET_WHY.finditer(match.group(1)):
                whys[part.group(1).strip()] = part.group(2).strip()
    if not whys:
        return []
    tests_wanted = bool(_ASKS_FOR_TESTS.search(task))
    seeds: list[dict[str, str]] = []
    for target in route.get("targets") or []:
        relative = str(target).strip().removeprefix("./")
        if (not relative or Path(relative).suffix not in _CODE_SUFFIXES
                or Path(relative).is_absolute() or ".." in Path(relative).parts):
            continue
        if _TEST_FILE.search(relative) and not tests_wanted:
            continue
        if any(seed["path"] == relative for seed in seeds):
            continue
        seeds.append({"path": relative, "why": whys.get(relative, "")})
    return seeds


def with_route_targets(plan: list[dict[str, str]], required: list[dict[str, str]],
                       ) -> tuple[list[dict[str, str]], list[str]]:
    """The plan with every required file in it: the planner's items as they are (none dropped,
    their order kept), then each required file it left out, with a change text saying why.
    Returns (the plan, the paths added)."""
    listed = {item["path"] for item in plan}
    plan = list(plan)
    added = []
    for seed in required:
        path = seed["path"]
        if any(path == other or other.endswith("/" + path) or path.endswith("/" + other)
               for other in listed):  # src/pkg/a.py is pkg/a.py
            continue
        why = f" ({seed['why']})" if seed.get("why") else ""
        plan.append({"path": seed["path"],
                     "change": f"The task needs a change in this file{why}: make the part of the "
                               "task that belongs here, consistent with the other planned files."})
        added.append(seed["path"])
    return plan, added


# -- plan_contradicting_tests: a visible test that calls a signature the task replaces --------------
# Why (2026-10-21): a task that changes a constructor's parameters (`Service(data_dir)` to
# `Service(repository)`) can leave a visible test still calling the old signature. Test files
# never reached the plan (the planner is told not to list them, and _plan drops them), so later
# rounds spent themselves bending the code back towards the old call. A deterministic detector
# (no model call) finds such tests and adds the file with a narrow instruction; a guard refuses a write that drops a test or one
# of its assertions.

CALL_TEST_NOTE = "update only the calls that the task's change breaks; keep every assertion"


@dataclass
class CallTest:
    """A visible test file that calls a task-named callable whose stated parameters replace the
    ones it has now."""

    path: str
    names: list[str]  # the changed callables as the test calls them (Service)
    changes: list[str]  # "Service(data_dir) -> Service(repository)"
    calls: list[str]  # "line 28: Service(tmp_path)"

    def change(self) -> str:
        return (f"The task changes {'; '.join(self.changes)}, and this test file still calls the "
                f"old signature ({'; '.join(self.calls)}). {CALL_TEST_NOTE[0].upper()}"
                f"{CALL_TEST_NOTE[1:]}: rewrite those calls to the new signature, and leave "
                "every assert, pytest.raises and expected value as it is.")


def _plain(params: list[acceptance.Param]) -> list[tuple[str, str]]:
    return [(p.name, p.kind) for p in params]


def _sig_text(name: str, params: list[acceptance.Param]) -> str:
    return f"{name}({', '.join(p.text() for p in params)})"


def _call_breaks(call: ast.Call, old: list[acceptance.Param], new: list[acceptance.Param]) -> bool:
    """Whether a call written for `old` uses what `new` replaces: a positional argument whose
    parameter is renamed or gone, a keyword the new signature does not take, or a new required
    parameter the call does not pass. Starred arguments are not judged (False)."""
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None
                                                                 for k in call.keywords):
        return False
    old_pos = [p for p in old if p.kind == "pos"]
    new_pos = [p for p in new if p.kind == "pos"]
    new_star = any(p.kind == "star" for p in new)
    new_names = {p.name for p in new if p.kind in ("pos", "kwonly")}
    for index in range(len(call.args)):
        before = old_pos[index].name if index < len(old_pos) else None
        after = new_pos[index].name if index < len(new_pos) else None
        if after is None and not new_star:
            return True
        if before is not None and after is not None and before != after:
            return True
    keywords = {k.arg for k in call.keywords}
    if any(k not in new_names for k in keywords) and not any(p.kind == "dstar" for p in new):
        return True
    passed = {p.name for p in new_pos[:len(call.args)]} | keywords
    return any(not p.default and p.name not in passed for p in new if p.kind in ("pos", "kwonly"))


def _module_aliases(tree: ast.Module) -> tuple[dict[str, str], dict[str, str]]:
    """(name -> 'module:attr' for `from module import attr as name`, expression text -> module
    for module references: `import a.b as c` (c), `import a.b` (a.b), `from a import b` (a.b))."""
    names: dict[str, str] = {}
    modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}:{alias.name}"
                modules[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules[alias.asname or alias.name] = alias.name
    return names, modules


def _from_module(source: str, module: str) -> bool:
    """`source` is the defining module or a package above it (a re-export from __init__)."""
    return source == module or module.startswith(source + ".")


def _changed_signatures(project: Path, task: str) -> list[tuple[acceptance.Required,
                                                                 list[acceptance.Param]]]:
    """The task-named callables that exist now and whose stated parameters differ from the ones
    they have: [(required with the stated params, the params it has now)]. None when the task
    says to keep the signature. An empty list (`main()`) is not a stated signature: prose writes
    calls that way ("when `main()` runs")."""
    spec = acceptance.extract(project, task)
    definitions = acceptance._definitions(project)[0]
    changed = []
    for required in spec.names:
        if not required.exists or not required.params or required.keep is not None:
            continue
        node = next((d.node for d in definitions.get(required.attr, [])
                     if d.path == required.path), None)
        current = acceptance._params_of(node) if node is not None else None
        if current is None or _plain(current) == _plain(required.params):
            continue
        changed.append((required, current))
    return changed


def contradicting_tests(project: Path, task: str) -> list[CallTest]:
    """Visible test files that call a callable the task gives new parameters (stated as
    `Name(a, b)` and different from the definition's now) in a way the new signature breaks.
    No model call; [] when the task changes no signature or no test calls one."""
    changed = _changed_signatures(project, task)
    if not changed:
        return []
    found: list[CallTest] = []
    for file in sorted(_test_files(project)):
        relative = file.relative_to(project).as_posix()
        if relative.startswith(".vega"):
            continue
        try:
            tree = ast.parse(file.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError, OSError):
            continue
        names, modules = _module_aliases(tree)
        hit = CallTest(relative, [], [], [])
        for required, current in changed:
            parts = required.attr.split(".")
            module = required.module
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if len(parts) == 1 and isinstance(func, ast.Name):
                    source = names.get(func.id, "")
                    module_part, _, attr = source.partition(":")
                    matches = attr == parts[0] and _from_module(module_part, module)
                elif len(parts) == 1 and isinstance(func, ast.Attribute):
                    owner = modules.get(ast.unparse(func.value), ast.unparse(func.value))
                    matches = func.attr == parts[0] and _from_module(owner, module)
                elif len(parts) == 2 and isinstance(func, ast.Attribute):
                    # Class.method: a method call by that name in a file that imports the class
                    matches = func.attr == parts[1] and any(
                        source.endswith(":" + parts[0]) for source in names.values())
                else:
                    matches = False
                if not matches or not _call_breaks(node, current, required.params or []):
                    continue
                label = required.attr.split(".")[-1]
                if label not in hit.names:
                    hit.names.append(label)
                    hit.changes.append(f"{_sig_text(required.attr, current)} -> "
                                       f"{_sig_text(required.attr, required.params or [])}")
                hit.calls.append(f"line {node.lineno}: {ast.unparse(node)[:100]}")
        if hit.calls:
            found.append(hit)
    return found


def with_call_tests(plan: list[dict[str, str]], tests: list[CallTest]
                    ) -> tuple[list[dict[str, str]], list[str]]:
    """The plan with each contradicting test file in it (after the planner's files), its change
    the narrow instruction. Returns (the plan, the paths added)."""
    listed = {item["path"] for item in plan}
    plan = list(plan)
    added = []
    for test in tests:
        if test.path in listed:
            continue
        plan.append({"path": test.path, "change": test.change()})
        added.append(test.path)
    return plan, added


def failing_call_tests(call_tests: dict[str, CallTest], failure: str) -> list[CallTest]:
    """The contradicting test files a round's failing output names (by path or file name)."""
    return [test for path, test in call_tests.items()
            if path in failure or Path(path).name in failure]


_RAISES = {"raises", "warns", "deprecated_call", "assertRaises", "assertRaisesRegex",
           "assertWarns"}


def _test_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and child.name.startswith("test"):
                    found[f"{node.name}.{child.name}"] = child
    return found


def _callee(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        func = node.func
        return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    return ""


def _assertions(function: ast.AST) -> list[ast.AST]:
    """assert statements, pytest.raises/warns contexts and unittest self.assert* calls."""
    found: list[ast.AST] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assert):
            found.append(node)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            found += [item.context_expr for item in node.items
                      if _callee(item.context_expr) in _RAISES]
        elif isinstance(node, ast.Expr) and _callee(node.value).startswith("assert"):
            found.append(node.value)
    return found


def _calls_any(node: ast.AST, names: list[str]) -> bool:
    return any(isinstance(n, ast.Call) and _callee(n) in names for n in ast.walk(node))


def weakened_test(before: str, after: str, names: list[str]) -> str | None:
    """Why a rewrite of a test file weakens it, or None: a test function is gone, a function
    has fewer assertions than before, or an assertion that does not call one of `names` (the
    callables whose signature the task changes) is no longer there word for word."""
    try:
        old_tree, new_tree = ast.parse(before), ast.parse(after)
    except (SyntaxError, ValueError):
        return None  # the write check refuses what does not parse
    old_tests, new_tests = _test_functions(old_tree), _test_functions(new_tree)
    problems = []
    for name, old in old_tests.items():
        new = new_tests.get(name)
        if new is None:
            problems.append(f"test {name} was removed")
            continue
        old_asserts, new_asserts = _assertions(old), _assertions(new)
        if len(new_asserts) < len(old_asserts):
            problems.append(f"{name} has {len(new_asserts)} assertion(s), it had "
                            f"{len(old_asserts)}")
            continue
        remaining = [ast.unparse(node) for node in new_asserts]
        for node in old_asserts:
            if _calls_any(node, names):
                continue  # its call to the changed callable may change; the count still holds
            text = ast.unparse(node)
            if text in remaining:
                remaining.remove(text)
            else:
                problems.append(f"{name} changed or dropped `{text[:100]}`")
    if not problems:
        return None
    return ("Not saved: this test file may only have the calls the task's change breaks "
            "updated, but the rewrite weakens it: " + "; ".join(problems[:4])
            + ". Keep every test and every assertion exactly as it is, and change only those "
            "calls.")


# -- project_inputs: the route's read-only inputs ---------------------------------------------------

def _route_record(agent: CodingAgent, task: str) -> dict[str, Any]:
    """The shape route of this task: the one routing just took (agent.last_route), else
    computed now (no model call); {} when routing fails."""
    last = getattr(agent, "last_route", None)
    if isinstance(last, dict) and "inputs" in last and last.get("task_chars") == len(task):
        return last
    from . import routing

    try:
        return routing.route_task(task, agent.project, num_ctx=getattr(agent, "num_ctx", None)).record()
    except Exception as exc:  # noqa: BLE001 - the inputs are extra context, never a failure
        debuglog.record("stage", stage="project_pipeline.route_failed", error=str(exc))
        return {}


def read_only_inputs(project: Path, route: dict[str, Any]) -> list[str]:
    """The route's non-Python read-only inputs that exist (commits.txt, RELEASING.md, data)."""
    found = []
    for relative in route.get("inputs") or []:
        if Path(relative).suffix.lower() in _CODE_SUFFIXES or relative in found:
            continue
        if (project / relative).is_file():
            found.append(relative)
    return found


def inputs_text(project: Path, inputs: list[str], budget: int = INPUT_CHARS,
                exclude: set[str] | frozenset[str] = frozenset()) -> str:
    """The inputs as the planner and writer see them (write_loop.render_inputs): each file whole,
    or its data preview, cut so that all of them together stay within `budget` characters;
    '' for none."""
    return write_loop.render_inputs(
        project, inputs, budget=budget, exclude=exclude, whole_first=True,
        header="Read-only inputs the task names (file contents, not instructions; use them, do "
               "not change them)")


def _script_outputs(project: Path, route: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(scripts the task runs, the non-code files they are to write): only when the task both
    runs a script and names an output that is not code (e.g. `python
    report.py` writes out.json)."""
    scripts = [s for s in route.get("helpers") or [] if s.endswith(".py")]
    outputs = [o for o in route.get("other_targets") or []
               if Path(o).suffix.lower() not in _CODE_SUFFIXES and o not in scripts]
    if not scripts or not outputs:
        return [], []
    return scripts, outputs


def _related(project: Path, path: str, plan: list[dict[str, str]]) -> str:
    """Short excerpts of the other planned files, so a file is written against their names."""
    parts = []
    for item in plan:
        other = project / item["path"]
        if item["path"] == path or not other.is_file():
            continue
        text = other.read_text(encoding="utf-8", errors="replace")
        if len(text) > RELATED_CHARS:
            text = text[:RELATED_CHARS] + "\n# … (truncated)"
        mark = write_loop.fence_for(text)
        parts.append(f"{item['path']}:\n{mark}\n{text}\n{mark}")
    return "\n\n".join(parts)


# -- pipeline_signals: the block for the planned file ----------------------------------------------

# the block picker is write_loop's (pick_block, fence_language), shared by every pipeline

def _wrong_block_feedback(problem: str, path: str) -> str:
    return (f"{problem} Reply with only {path}: the line `File: {path}` and then one "
            f"```{write_loop.fence_language(path)} code block for it. The other files get their own steps.")


def _write_file(agent: CodingAgent, task: str, item: dict[str, str], plan: list[dict[str, str]],
                failure: str, refusal: str = "", extras: _Extras | None = None) -> str:
    """Produce one planned file and save it (write_loop.save); the save result, or why nothing
    was written."""
    produced = _produce_file(agent, task, item, plan, failure, refusal, extras)
    if not produced.writes:
        return produced.problem
    return write_loop.save(write_loop.AgentHost(agent), _save_config({}),
                           item["path"], produced.writes[item["path"]])


def _save_config(originals: dict[str, str | None]) -> write_loop.LoopConfig:
    """What write_loop.save needs: the debug-log name (the parse check of the writer's own code is
    made in _produce_file, against the file as it is now, so no Write carries `code`)."""
    return write_loop.LoopConfig(name="project_pipeline", title="Project pipeline", targets=[],
                                 originals=originals, repairs=REPAIRS,
                                 check=lambda _saved: write_loop.Checks())


def _produce_file(agent: CodingAgent, task: str, item: dict[str, str],
                  plan: list[dict[str, str]], failure: str, refusal: str = "",
                  extras: _Extras | None = None) -> write_loop.Produced:
    """One writer call (and one retry for a reply without the planned file's block) for one
    planned file: its merged content, or the problem (too large, no block, a reply that does
    not parse by itself)."""
    project = agent.project
    target = project / item["path"]
    current = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
    focused = ""
    if len(current) > FILE_CHARS:
        large = features.on("large_projects") or features.on("project_large_files")
        if not (large and target.suffix == ".py" and _parses(current)):
            return write_loop.Produced(
                problem=f"{item['path']} is too large to rewrite whole; left unchanged.")
        wide = features.on("large_projects")
        focused = focused_view(item["path"], current, task, item["change"],
                               budget=FOCUS_CHARS if wide else WINDOW_FOCUS_CHARS,
                               outline_chars=OUTLINE_CHARS if wide else WINDOW_OUTLINE_CHARS)
    plan_text = "\n".join(f"- {p['path']}: {p['change']}" for p in plan)
    mark = write_loop.fence_for(current)
    shown = focused or (f"Current content of {item['path']}:\n{mark}\n{current}\n{mark}"
                        if current else f"{item['path']} does not exist yet; create it.")
    related = _related(project, item["path"], plan)
    user = (f"Task:\n{task}\n\nPlan:\n{plan_text}\n\nNow write {item['path']}: {item['change']}"
            f"\n\n{shown}")
    if related:
        user += f"\n\nOther files in the plan (for names and signatures):\n{related}"
    if extras is not None and extras.inputs:  # project_inputs: what fits next to the rest
        budget = min(INPUT_CHARS, max(INPUT_MIN_CHARS, WRITER_PROMPT_CHARS - len(user)))
        user += inputs_text(project, extras.inputs, budget,
                            exclude={p["path"] for p in plan})
    if features.on("import_grounding"):
        user += ("\n\n" + _modules_line(project) + ". Import only these, the modules in the plan "
                 "and the standard library.")
    if failure:
        user += f"\n\nThe project's tests currently fail with:\n{failure}"
    signals = features.on("pipeline_signals")
    if signals:
        if refusal:  # the last round's write of this file was not saved: its exact reason
            user += f"\n\nYour last reply for {item['path']} was not saved:\n{refusal}"
        user += (f"\n\nReply with {item['path']} only: the line `File: {item['path']}` and then "
                 f"one ```{write_loop.fence_language(item['path'])} code block.")
    system = WRITER_SYSTEM + WRITER_FORMAT if signals else WRITER_SYSTEM
    messages = [{"role": "system", "content": with_context(agent, system)},
                {"role": "user", "content": user}]
    reply = _ask(agent, messages)
    if signals:
        plan_paths = [p["path"] for p in plan]
        picked, problem = write_loop.pick_block(reply, item["path"], plan_paths)
        if picked is None:  # told why, one retry
            feedback = _wrong_block_feedback(problem, item["path"])
            debuglog.record("stage", stage="project_pipeline.wrong_block", path=item["path"],
                            problem=problem)
            agent._emit("discarded", reason=problem, retry=True)
            reply = _ask(agent, [*messages, {"role": "assistant", "content": reply},
                                 {"role": "user", "content": feedback}])
            picked, problem = write_loop.pick_block(reply, item["path"], plan_paths)
            if picked is None:
                return write_loop.Produced(problem=f"Not written: {problem} (asked twice; "
                                                   f"{item['path']} is unchanged.)")
        blocks = [picked]
    else:
        blocks = write_loop.code_blocks(reply) or write_loop.code_blocks(reply, any_language=True)
    if not blocks:
        return write_loop.Produced(problem=f"No code block for {item['path']}; left unchanged.")
    code = _drop_path_line(max(blocks, key=len), item["path"]).strip("\n") + "\n"
    if signals and current and _parses(current):
        broken = edits.unparsable_reply(item["path"], code, project)
        if broken is not None:  # merge_code would silently keep the old file
            debuglog.record("stage", stage="project_pipeline.refused", path=item["path"],
                            refusal=broken[:600])
            return write_loop.Produced(problem=broken)
    content = merge_code(current, code) if target.suffix == ".py" and current else code
    if target.suffix == ".py" and current and _definitions_dropped(current, content):
        content = merge_code(current, code, replace_whole=False)
    if target.suffix == ".py" and features.on("import_grounding"):
        content = _drop_stale_imports(project, item["path"], content, code)
        stdlib = missing_stdlib_imports(content)
        if stdlib:
            content = _insert_imports(content, [f"import {name}" for name in stdlib])
    if target.suffix == ".py" and features.on("project_auto_imports"):
        forbidden = (forbidden_imports(task, item["path"])
                     if features.on("project_layer_imports") else ())
        resolved = resolve_imports(project, item["path"], content, forbidden)
        content, added = resolved.content, resolved.added
        if added:
            debuglog.record("stage", stage="project_pipeline.auto_imports", path=item["path"],
                            added=added)
        if resolved.foreign and signals:  # the write check would refuse with "define or import"
            refused = edits.check("pipeline", item["path"], current if target.is_file() else None,
                                  content, project, foreign=resolved.foreign).refusal
            if refused is not None:
                debuglog.record("stage", stage="project_pipeline.refused", path=item["path"],
                                refusal=refused[:600], foreign=resolved.foreign)
                args = {"path": item["path"], "content": content}  # shown as write_loop.save does
                agent._emit("tool_call", name="write_file", args=args)
                agent._emit("tool_result", name="write_file", args=args, result=refused)
                return write_loop.Produced(problem=refused)
    call_test = extras.call_tests.get(item["path"]) if extras is not None else None
    if call_test is not None and extras is not None:  # its assertions are the yardstick
        before = extras.originals.get(item["path"])
        weakened = weakened_test(before if before is not None else current, content,
                                 call_test.names)
        if weakened is not None:
            debuglog.record("stage", stage="project_pipeline.refused", path=item["path"],
                            refusal=weakened[:600])
            return write_loop.Produced(problem=weakened)
    return write_loop.Produced({item["path"]: write_loop.Write(content)})


_MEMBERS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


def _first_line(node: ast.AST) -> int:
    return min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])  # type: ignore[attr-defined]


def _outline(tree: ast.Module, keep_methods: set[str] | None = None) -> str:
    """Every top-level function and class with its line range, and each class's methods (all
    classes, or only those in `keep_methods`; the others show a method count)."""
    out = []
    for node in tree.body:
        if not isinstance(node, _MEMBERS):
            continue
        span = f"{_first_line(node)}-{node.end_lineno}"
        if not isinstance(node, ast.ClassDef):
            out.append(f"def {node.name}: {span}")
            continue
        members = [m for m in node.body if isinstance(m, _MEMBERS)]
        if keep_methods is not None and node.name not in keep_methods:
            out.append(f"class {node.name}: {span} ({len(members)} methods)")
            continue
        out.append(f"class {node.name}: {span}")
        out += [f"    {m.name}: {_first_line(m)}-{m.end_lineno}" for m in members]
    return "\n".join(out)


def _definitions_by_name(tree: ast.Module) -> dict[str, list[tuple[ast.ClassDef | None, ast.AST]]]:
    """Name -> (enclosing class or None, node) for top-level defs/classes and class members."""
    found: dict[str, list[tuple[ast.ClassDef | None, ast.AST]]] = {}
    for node in tree.body:
        if not isinstance(node, _MEMBERS):
            continue
        found.setdefault(node.name, []).append((None, node))
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, _MEMBERS):
                    found.setdefault(member.name, []).append((node, member))
    return found


def _class_head_end(node: ast.ClassDef) -> int:
    """Last line of a class's header, docstring and attributes (before its first member)."""
    end = _first_line(node.body[0]) - 1 if node.body else node.end_lineno or node.lineno
    for member in node.body:
        if isinstance(member, _MEMBERS):
            break
        end = member.end_lineno or end
    return end


def _containing(tree: ast.Module, line: int) -> tuple[ast.ClassDef | None, ast.AST] | None:
    """The method or top-level function holding `line`; a class's own head is the class."""
    for node in tree.body:
        if not isinstance(node, _MEMBERS) or not _first_line(node) <= line <= (node.end_lineno or 0):
            continue
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, _MEMBERS) and _first_line(member) <= line <= (member.end_lineno or 0):
                    return node, member
        return None, node
    return None


def _relevant(tree: ast.Module, source: str, path: str, task: str, change: str,
              ) -> list[tuple[ast.ClassDef | None, ast.AST]]:
    """Definitions most relevant to a change, best first: names written as code in the change
    text, then in the task (`Class.method`, snake_case, CamelCase, `name(` or in backticks;
    `Class.method` picks only that class's method), then the file's best search hits, then plain
    words that happen to name a definition."""
    from .index import CodeIndex, FileEntry, extract_symbols

    by_name = _definitions_by_name(tree)
    ranked: list[tuple[ast.ClassDef | None, ast.AST]] = []

    def add(entry: tuple[ast.ClassDef | None, ast.AST]) -> None:
        if all(entry[1] is not seen[1] for seen in ranked):
            ranked.append(entry)

    words: list[str] = []  # plain words that happen to name a definition (`add`, `help`)
    for text in (change, task):
        for match in _NAME.finditer(text):
            dotted = match.group(0).split(".")
            name, owner = dotted[-1], (dotted[-2] if len(dotted) > 1 else None)
            entries = [e for e in by_name.get(name, [])
                       if owner is None or (e[0] is not None and e[0].name == owner)]
            if not entries:
                continue
            code_like = (len(dotted) > 1 or "_" in name or name[1:] != name[1:].lower()
                         or text[match.end():match.end() + 1] == "(" or (
                             match.start() and text[match.start() - 1] == "`"))
            if code_like:
                for entry in entries:
                    add(entry)
            else:
                words.append(name)
    index = CodeIndex(Path("."), [FileEntry(path, source.splitlines(), extract_symbols(path, source))])
    for hit in index.search(f"{change} {task}", limit=12):
        entry = _containing(tree, hit.start_line)
        if entry is not None:
            add(entry)
    for name in words:
        for entry in by_name.get(name, []):
            add(entry)
    return ranked


def focused_view(path: str, source: str, task: str, change: str,
                 budget: int = FOCUS_CHARS, outline_chars: int = OUTLINE_CHARS) -> str:
    """What the writer sees of a Python file too large to show whole: an outline, the file's
    imports and the complete source of the definitions most relevant to the change (whole
    definitions only, up to `budget` characters; one that does not fit is skipped)."""
    tree = ast.parse(source)
    lines = source.split("\n")
    chosen: list[tuple[ast.ClassDef | None, ast.AST, int, int]] = []  # (class, node, first, last)
    used = 0
    for owner, node in _relevant(tree, source, path, task, change):
        first = _first_line(node)
        last = _class_head_end(node) if isinstance(node, ast.ClassDef) else node.end_lineno or first  # type: ignore[attr-defined]
        if isinstance(node, ast.ClassDef) and last < first:
            continue
        size = sum(len(line) + 1 for line in lines[first - 1:last])
        if used + size > budget:
            continue
        if any(first <= c_first and c_last <= last or c_first <= first and last <= c_last
               for _, _, c_first, c_last in chosen):
            continue  # already shown, or inside something shown
        chosen.append((owner, node, first, last))
        used += size
    classes = {owner.name for owner, *_ in chosen if owner is not None}
    classes |= {node.name for owner, node, *_ in chosen if owner is None and isinstance(node, ast.ClassDef)}
    outline = _outline(tree)
    if len(outline) > outline_chars:
        outline = _outline(tree, keep_methods=classes)
        if len(outline) > outline_chars:
            outline = outline[:outline_chars].rsplit("\n", 1)[0] + "\n# … (outline truncated)"
    imports = "\n".join(ast.get_source_segment(source, node) or "" for node in tree.body
                        if isinstance(node, (ast.Import, ast.ImportFrom)))
    if len(imports) > IMPORT_CHARS:
        imports = imports[:IMPORT_CHARS].rsplit("\n", 1)[0] + "\n# … (more imports)"
    parts: list[str] = []
    open_class: ast.ClassDef | None = None
    for owner, node, first, last in sorted(chosen, key=lambda c: c[2]):
        if owner is not None and owner is not open_class:
            header_end = _first_line(owner.body[0]) - 1 if owner.body else owner.lineno
            header = lines[_first_line(owner) - 1:max(owner.lineno, header_end)]
            parts.append(f"# lines {_first_line(owner)}-{owner.end_lineno}: class {owner.name} "
                         "(only the members below are shown)\n" + "\n".join(header))
        open_class = owner or (node if isinstance(node, ast.ClassDef) else None)
        indent = " " * getattr(node, "col_offset", 0)
        parts.append(f"{indent}# lines {first}-{last}\n" + "\n".join(lines[first - 1:last]))
    count = len(lines)
    if parts:
        body = "\n\n".join(parts)
        mark = write_loop.fence_for(body)
        relevant = ("The complete source of the definitions most relevant to this change:\n"
                    f"{mark}python\n{body}\n{mark}")
    else:
        relevant = "No definition in the file matched this change by name or search."
    example = next(((lines[owner.lineno - 1].strip(), node.name) for owner, node, *_ in chosen
                    if owner is not None and lines[owner.lineno - 1].rstrip().endswith(":")),
                   ("class Name:", "method"))
    return (
        f"{path} is large ({count} lines), so it is not shown whole: below are its outline, its "
        "imports and the parts most relevant to this change.\n\n"
        f"Outline of {path} (name: first-last line):\n{outline}\n\n"
        f"Imports of {path}:\n```python\n{imports}\n```\n\n{relevant}\n\n"
        "Reply with only the definitions you add or change, each complete, plus any new imports "
        "they need, in one ```python code block. A top-level function: the whole function. A "
        "method: the method inside its class header, with only the methods you add or change, "
        f"like\n```python\n{example[0]}\n    def {example[1]}(self, ...):\n        "
        "<the complete method>\n```\nEverything you leave out stays exactly as it is: never "
        "rewrite the whole file or class, and do not write placeholders for omitted code."
    )


def large_map(index: Any, task: str, budget: int = LARGE_MAP_CHARS) -> str:
    """The planner's map of a project too large to list whole (the plain map of Vega's repo is
    1,500 bare paths, none under src/): the Python files holding the task's best search hits,
    each with its symbols, then the project's other non-test Python modules by path."""
    related: list[str] = []
    for hit in index.search(task, limit=40):
        if hit.path.endswith(".py") and not _TEST_FILE.search(hit.path) and hit.path not in related:
            related.append(hit.path)
        if len(related) == LARGE_MAP_FILES:
            break
    parts = ["Files most related to the task (with their symbols and first lines):"]
    per_file = budget // 2 // max(1, len(related))
    for path in related:
        parts.append(index.repo_map(max_chars=per_file, prefix=path))
    used = sum(len(part) + 1 for part in parts)
    others = [f.path for f in index.files if f.path.endswith(".py") and f.path not in related
              and not _TEST_FILE.search(f.path)]
    parts.append("\nOther Python modules:")
    for number, path in enumerate(others):
        if used + len(path) + 40 > budget:
            parts.append(f"... and {len(others) - number} more")
            break
        parts.append(path)
        used += len(path) + 1
    return "\n".join(parts)


def _module_names(project: Path) -> dict[str, set[str] | None]:
    """Dotted name -> names the module defines at top level, for the project's own Python files
    (src/ layout included); None when the module's names cannot be known statically (a star
    import or a module __getattr__, or it does not parse)."""
    modules: dict[str, set[str] | None] = {}
    for path in _project_files(project):
        relative = path.relative_to(project)
        if path.suffix != ".py" or _TEST_FILE.search(relative.as_posix()):
            continue
        parts = list(relative.with_suffix("").parts)
        if parts and parts[0] == "src":
            parts = parts[1:]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            modules[".".join(parts)] = None
            continue
        names: set[str] | None = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                if node.name == "__getattr__":
                    names = None
                    break
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if any(alias.name == "*" for alias in node.names):
                    names = None
                    break
                names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
            elif isinstance(node, (ast.If, ast.Try)):
                names = None  # conditional definitions: do not guess
                break
        modules[".".join(parts)] = names
    return modules


def _import_problems(node: ast.AST, package: list[str], modules: dict[str, set[str] | None],
                     roots: set[str]) -> list[str]:
    """What is wrong with one import statement of this project's modules ([] if nothing)."""
    problems = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] in roots and alias.name not in modules:
                problems.append(f"`import {alias.name}`: no such module")
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            base = package[:len(package) - node.level + 1] if node.level > 1 else package
            target = ".".join([*base, *(node.module.split(".") if node.module else [])])
        else:
            target = node.module or ""
        if not target or target.split(".")[0] not in roots:
            return []
        if target not in modules:
            return [f"`from {target} import ...`: no module {target}"]
        defined = modules[target]
        for alias in node.names:
            if defined is None or alias.name in defined or f"{target}.{alias.name}" in modules:
                continue
            problems.append(f"`from {target} import {alias.name}`: {target} has no {alias.name}")
    return problems


def _package_of(relative: str) -> list[str]:
    parts = list(Path(relative).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    return parts[:-1]  # the containing package, for a module and for an __init__ alike


def unresolved_imports(project: Path, paths: list[str]) -> list[str]:
    """Imports in `paths` of this project's modules, or of names from them, that do not exist."""
    modules = _module_names(project)
    roots = {name.split(".")[0] for name in modules}
    problems = []
    for relative in paths:
        path = project / relative
        if path.suffix != ".py" or not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue  # the tests and the parse guard report this
        package = _package_of(relative)
        for node in ast.walk(tree):
            problems += [f"{relative}: {p}" for p in _import_problems(node, package, modules, roots)]
    return problems


def _drop_stale_imports(project: Path, relative: str, content: str, code: str) -> str:
    """Remove top-level imports of this project's modules that do not resolve and that the new
    code block does not contain: merging by definition keeps a file's old import lines, so an
    invented import outlived the code that used it (2026-10-14)."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content
    modules = _module_names(project)
    roots = {name.split(".")[0] for name in modules}
    package = _package_of(relative)
    lines = content.split("\n")
    for node in reversed(tree.body):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        segment = ast.get_source_segment(content, node) or ""
        if segment and segment not in code and _import_problems(node, package, modules, roots):
            del lines[node.lineno - 1:node.end_lineno]
    return "\n".join(lines)


def missing_stdlib_imports(source: str) -> list[str]:
    """Standard-library modules the code uses (`argparse.ArgumentParser`) but never binds: each
    is a NameError when that line runs (goals 2026-10-14: `argparse` used, not imported). A name
    bound anywhere in the file (import, def, assignment, argument, loop or with target) is left
    alone, so this never shadows or duplicates anything."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    bound: set[str] = set()
    used: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (used.append(node.id) if isinstance(node.ctx, ast.Load) else bound.add(node.id))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    missing = []
    for name in used:
        if (name in sys.stdlib_module_names and name not in bound and name not in missing
                and not name.startswith("_") and not hasattr(builtins, name)):
            missing.append(name)
    return missing


# -- project_auto_imports ---------------------------------------------------------------------------
# Why (2026-10-20): writes were refused only for a name the writer forgot to import: an
# exception class defined in a sibling module, a helper from another module of the package, a
# stdlib name such as `DecimalException`, or stdlib modules (`re`, `sys`, `argparse`: the stdlib
# auto-import ran only under the opt-in import_grounding, which was off by default).

# well-known stdlib names a writer uses bare: name -> the module to import it from
STDLIB_NAMES = {
    **dict.fromkeys(("Decimal", "DecimalException", "InvalidOperation", "ROUND_HALF_UP",
                     "ROUND_HALF_EVEN", "ROUND_DOWN", "ROUND_UP", "ROUND_CEILING", "ROUND_FLOOR",
                     "localcontext"), "decimal"),
    **dict.fromkeys(("dataclass", "field", "asdict", "astuple", "fields"), "dataclasses"),
    **dict.fromkeys(("defaultdict", "Counter", "OrderedDict", "deque", "namedtuple"),
                    "collections"),
    **dict.fromkeys(("Path", "PurePath"), "pathlib"),
    **dict.fromkeys(("Enum", "IntEnum", "StrEnum", "Flag", "auto"), "enum"),
    **dict.fromkeys(("Any", "Optional", "Union", "Callable", "Iterable", "Iterator", "Sequence",
                     "Mapping", "TypeVar", "Generic", "Protocol", "NamedTuple", "TypedDict",
                     "ClassVar", "Literal", "cast", "TYPE_CHECKING"), "typing"),
    **dict.fromkeys(("partial", "lru_cache", "wraps", "reduce", "cached_property",
                     "total_ordering"), "functools"),
    **dict.fromkeys(("timedelta", "timezone"), "datetime"),
    **dict.fromkeys(("contextmanager", "suppress"), "contextlib"),
    **dict.fromkeys(("ABC", "abstractmethod"), "abc"),
    **dict.fromkeys(("StringIO", "BytesIO"), "io"),
    "Fraction": "fractions", "dedent": "textwrap",
}

# -- project_layer_imports --------------------------------------------------------------------------
# Why (2026-10-21): a write was refused every round, first only for `date` (used in an
# annotation; STDLIB_NAMES had timedelta and timezone but not date), then for a name that lives
# in an outer layer: the refusal said "define or import" although importing it is what the task
# forbids. A test update was refused for a missing `import pytest`.
# The table below is derived from the stdlib instead of listed by hand.

# where a bare name is looked up, in priority order (collections.Counter before typing.Counter,
# typing.Callable before collections.abc.Callable); a name two of these export as different
# objects (not one an alias of the other) is ambiguous and left out
STDLIB_HOMES = ("collections", "typing", "collections.abc", "dataclasses", "datetime", "decimal",
                "fractions", "pathlib", "enum", "functools", "itertools", "contextlib", "abc",
                "io", "textwrap", "uuid", "json", "string", "operator", "copy", "heapq", "bisect",
                "tempfile", "argparse", "statistics", "unittest.mock", "concurrent.futures")
# lowercase names are kept only when distinctive (a model writing `mean(xs)` means statistics';
# `count`, `product`, `chain`, `merge`, `wrap`, `replace`, `cache` are as likely its own helper)
STDLIB_FUNCTIONS = frozenset((
    "dataclass", "field", "fields", "asdict", "astuple", "is_dataclass", "make_dataclass",
    "defaultdict", "deque", "namedtuple", "partial", "partialmethod", "lru_cache",
    "cached_property", "wraps", "reduce", "total_ordering", "singledispatch", "cmp_to_key",
    "contextmanager", "asynccontextmanager", "suppress", "nullcontext", "redirect_stdout",
    "redirect_stderr", "abstractmethod", "dedent", "cast", "overload", "runtime_checkable",
    "get_type_hints", "get_args", "get_origin", "localcontext", "getcontext", "date", "timedelta",
    "timezone", "deepcopy", "attrgetter", "itemgetter", "methodcaller", "heappush", "heappop",
    "heapify", "heappushpop", "nlargest", "nsmallest", "bisect_left", "bisect_right", "insort",
    "uuid4", "uuid1", "auto", "zip_longest", "groupby", "islice", "pairwise", "permutations",
    "combinations", "accumulate", "starmap", "takewhile", "dropwhile", "mkdtemp", "mkstemp",
    "gettempdir", "ascii_letters", "ascii_lowercase", "ascii_uppercase", "fmean", "mean",
    "median", "stdev", "pstdev", "variance", "quantiles", "patch", "mock_open",
    "create_autospec"))
# CamelCase names generic enough that a project means its own (decimal's signal classes
# Overflow/Rounded/Inexact, argparse's Action/Namespace, numbers' Real/Number, Context…)
STDLIB_GENERIC = frozenset((
    "Action", "Namespace", "Field", "Executor", "Number", "Real", "Complex", "Rational",
    "Integral", "Formatter", "Template", "Buffer", "Rounded", "Clamped", "Inexact", "Overflow",
    "Underflow", "Subnormal", "Context", "BasicContext", "DefaultContext", "ExtendedContext",
    "Text", "Error", "FileType", "Buffer"))
# other common modules exporting a name differently make it ambiguous too (asyncio.Future vs
# concurrent.futures.Future, threading.Event); the hand-listed STDLIB_NAMES are kept regardless
# (pathlib.Path, not zipfile.Path; typing.Protocol, not asyncio.Protocol)
STDLIB_RIVALS = ("asyncio", "threading", "multiprocessing", "queue", "zipfile", "logging", "re",
                 "os", "shutil", "subprocess", "csv", "socket", "selectors", "sqlite3", "time")
_STDLIB_CONSTANTS = re.compile(r"^(ROUND_[A-Z0-9_]+|TYPE_CHECKING|UTC)$")


def _same_object(a: object, b: object) -> bool:
    return a is b or getattr(a, "__origin__", None) is b or getattr(b, "__origin__", None) is a


@functools.cache
def stdlib_names() -> dict[str, str]:
    """project_layer_imports: bare name -> the stdlib module to import it from, derived from
    STDLIB_HOMES: each module's public names (`__all__`, else names defined in it), kept when
    CamelCase (classes, exceptions, typing forms; an all-caps class such as UUID) and not in
    STDLIB_GENERIC, when a distinctive
    function (STDLIB_FUNCTIONS) or a ROUND_*/TYPE_CHECKING/UTC constant, and not exported as
    something else by another home or by STDLIB_RIVALS; never a builtin or a stdlib module name (`datetime`, `time`, `copy` are handled as modules). Includes every
    STDLIB_NAMES entry."""
    import importlib

    found: dict[str, tuple[str, object]] = {}
    ambiguous: set[str] = set()
    for module_name in STDLIB_HOMES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        public = getattr(module, "__all__", None)
        if public is None:
            homes = {module_name, "_" + module_name, "_py" + module_name}
            public = [n for n, v in vars(module).items() if not n.startswith("_")
                      and getattr(v, "__module__", module_name) in homes]
        for name in public:
            if name.startswith("_") or hasattr(builtins, name) or name in sys.stdlib_module_names:
                continue
            value = getattr(module, name, None)
            camel = name[0].isupper() and (any(c.islower() for c in name)
                                           or (len(name) > 1 and isinstance(value, type)))
            if not ((camel and name not in STDLIB_GENERIC) or name in STDLIB_FUNCTIONS
                    or _STDLIB_CONSTANTS.match(name)):
                continue
            if value is None:
                continue
            if name in found:
                if not _same_object(found[name][1], value):
                    ambiguous.add(name)
                continue
            found[name] = (module_name, value)
    for module_name in STDLIB_RIVALS:
        try:
            rival = vars(importlib.import_module(module_name))
        except ImportError:
            continue
        ambiguous.update(name for name, (_, value) in found.items()
                         if name in rival and not _same_object(rival[name], value))
    names = {name: module for name, (module, _) in found.items() if name not in ambiguous}
    return {**names, **STDLIB_NAMES}


# modules a test file uses by attribute without importing them (pytest.raises; @mock.patch)
_TEST_MODULES = {"pytest": "import pytest", "mock": "from unittest import mock"}

# `datetime` used bare: the class (datetime.now(), annotations, datetime(2026, 1, 1)) or the
# module (datetime.date.today(), datetime.timedelta(days=1)); mixed use is left to the writer
_DATETIME_MODULE_ATTRS = frozenset(("datetime", "date", "time", "timedelta", "timezone", "tzinfo",
                                    "UTC", "MINYEAR", "MAXYEAR"))
_DATETIME_CLASS_ATTRS = frozenset((
    "now", "utcnow", "today", "fromisoformat", "strptime", "combine", "fromtimestamp",
    "utcfromtimestamp", "fromordinal", "fromisocalendar", "min", "max", "resolution"))


def _datetime_import(tree: ast.Module) -> str | None:
    used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == "datetime"}
    if used and used <= _DATETIME_MODULE_ATTRS:
        return "import datetime"
    if used <= _DATETIME_CLASS_ATTRS:
        return "from datetime import datetime"
    return None


# a negation governing "import": "must not import", "may not import", "never imports", "No module
# under app/domain may import" (not "no cycle ... where every import statement counts")
_NEGATED_IMPORT = re.compile(
    r"\b(?:must|may|should|can|shall|does|do)\s*(?:not|n't)\s+(?:\S+\s+){0,2}?imports?\b"
    r"|\b(?:cannot|never)\s+(?:\S+\s+){0,2}?imports?\b"
    r"|\bno\s+(?:\S+\s+){0,4}?(?:may|must|can|should|shall)\s+import\b", re.IGNORECASE)
_MODULE_PATH = re.compile(r"[A-Za-z_]\w*(?:/[A-Za-z_]\w*)+(?:\.py)?")
_DOTTED_MODULE = re.compile(r"(?<![\w/.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")


def _inside(module: str, scope: str) -> bool:
    return module == scope or module.startswith(scope + ".")


def forbidden_imports(task: str, relative: str) -> tuple[str, ...]:
    """Modules (dotted, two parts or more) the task says this file must not import: the dotted
    names after a negated import, up to the end of its clause ("No module under app/domain
    may import app.storage or app.web"; "shop/service.py must not import json, os,
    shop.ledger or anything in shop.adapters, and ..."). The last path written before
    the negation or in it (app/domain, shop/service.py) is its subject: the rule applies
    only to files under it; with no path it applies to every file but the tests. A file is never
    forbidden its own package."""
    own = _module_of(relative)
    forbidden: list[str] = []
    for sentence in re.split(r"(?<=[.;:])\s+|\n", task):
        for match in _NEGATED_IMPORT.finditer(sentence):
            subjects = _MODULE_PATH.findall(sentence[:match.end()])
            if subjects:
                subject = subjects[-1].removesuffix(".py").replace("/", ".").removeprefix("src.")
                if not _inside(own, subject):
                    continue
            elif _TEST_FILE.search(relative):
                continue  # "`import geokit` must not import geokit.plot": not about the tests
            clause = re.split(r",\s+(?:and|but|yet|so|while|which)\b|;",
                              sentence[match.end():])[0]
            forbidden += [d for d in _DOTTED_MODULE.findall(clause)
                          if not d.endswith((".py", ".txt", ".json", ".csv", ".md"))
                          and not _inside(own, d)]
    return tuple(dict.fromkeys(forbidden))


def _module_of(relative: str) -> str:
    """Dotted module name of a project file (src/ layout and __init__ handled)."""
    parts = list(Path(relative).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _defining_modules(project: Path) -> dict[str, list[tuple[str, str, ast.Module]]]:
    """Name -> [(dotted module, its path, its tree)] for every non-test project module that
    defines the name at module level (def, class or assignment, also inside a top-level if/try);
    imports and re-exports do not count."""
    found: dict[str, list[tuple[str, str, ast.Module]]] = {}
    for path in _project_files(project):
        relative = path.relative_to(project).as_posix()
        if path.suffix != ".py" or _TEST_FILE.search(relative) or relative.startswith(".vega"):
            continue
        module = _module_of(relative)
        if not module:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        names: set[str] = set()
        body = list(tree.body)
        while body:
            node = body.pop(0)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(t.id for t in targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.If):
                body += node.body + node.orelse
            elif isinstance(node, ast.Try):
                body += node.body + node.orelse + node.finalbody
        for name in names:
            found.setdefault(name, []).append((module, relative, tree))
    return found


def _import_graph(project: Path, extra: frozenset[str] = frozenset()) -> dict[str, set[str]]:
    """Project module -> the project modules it imports anywhere in the file (inside functions
    and `if TYPE_CHECKING:` too; `from pkg import x` counts pkg.x when that is a module). `extra`:
    modules to count although they are not on disk yet (the file being written)."""
    files = {}
    for path in _project_files(project):
        relative = path.relative_to(project).as_posix()
        if path.suffix == ".py" and not relative.startswith(".vega") and _module_of(relative):
            files[_module_of(relative)] = relative
    known = set(files) | extra
    graph: dict[str, set[str]] = {}
    for module, relative in files.items():
        try:
            tree = ast.parse((project / relative).read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        package = _package_of(relative)
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                target = _import_target(node, package)
                names = [target, *(f"{target}.{alias.name}" for alias in node.names)]
            else:
                continue
            for name in names:
                parts = name.split(".")
                found.update(".".join(parts[:cut]) for cut in range(1, len(parts) + 1)
                             if ".".join(parts[:cut]) in known)
        graph[module] = found - {module}
    return graph


def _reaches(graph: dict[str, set[str]], start: str, goal: str) -> bool:
    """Whether `start` imports `goal`, directly or through other project modules."""
    seen, todo = {start}, [start]
    while todo:
        for nxt in graph.get(todo.pop(), set()):
            if nxt == goal:
                return True
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return False


def _import_line(relative: str, content: str, source: str, name: str) -> str:
    """`from <source> import name`, relative when the file is in the same top-level package as
    `source` and does not already import its own package absolutely."""
    package = _package_of(relative)
    parts = source.split(".")
    try:
        tree = ast.parse(content)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
    froms = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
    absolute_style = any(n.level == 0 and (n.module or "").split(".")[0] == parts[0]
                         for n in froms) and not any(n.level for n in froms)
    if not package or package[0] != parts[0] or absolute_style:
        return f"from {source} import {name}"
    common = 0
    while common < min(len(package), len(parts)) and package[common] == parts[common]:
        common += 1
    level = len(package) - common + 1
    return f"from {'.' * level}{'.'.join(parts[common:])} import {name}"


@dataclass
class AutoImports:
    content: str
    added: list[str]  # the import lines added
    # project_layer_imports: undefined name -> why it was not imported and what to do instead
    # ("is defined in app.web.render, and the task says ... must not import app.web: ...");
    # the write check's refusal gives this instead of "define or import"
    foreign: dict[str, str] = field(default_factory=dict)


def auto_imports(project: Path, relative: str, content: str) -> tuple[str, list[str]]:
    """project_auto_imports, as (the content, the import lines added); see resolve_imports."""
    result = resolve_imports(project, relative, content)
    return result.content, result.added


def resolve_imports(project: Path, relative: str, content: str,
                    forbidden: tuple[str, ...] = ()) -> AutoImports:
    """project_auto_imports: the undefined names in `content` (F821, scope-aware with ruff, the
    same findings the write check refuses) that have exactly one home get their import: a stdlib
    module used as a module (`re.sub`), a name exactly one other project module defines, or a
    well-known stdlib name (STDLIB_NAMES). A project name is imported only from this file's own
    package or an ancestor package (pkg/domain/planner.py never gets an import from
    pkg.web: a layered design forbids exactly that), and never from a module that
    imports this file, directly or through others (a circular import).
    With project_layer_imports: stdlib names come from stdlib_names() (date, Path, List…), bare
    `datetime` is imported as the class or the module by how it is used, a test file gets
    `import pytest` / `from unittest import mock` for pytest.x / mock.x, a name in a module of
    `forbidden` (forbidden_imports: what the task says this file must not import) is not
    imported, and the names left undefined because their only home is a sibling package or a
    forbidden module are returned in `foreign`, with what to do instead."""
    from .acceptance import static_findings

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return AutoImports(content, [])
    undefined = list(dict.fromkeys(edits._backticked(message) for _, code, message
                                   in static_findings(project, relative, content)
                                   if code == "F821"))
    if not undefined:
        return AutoImports(content, [])
    layered = features.on("project_layer_imports")
    table = stdlib_names() if layered else STDLIB_NAMES
    attribute_bases = {node.value.id for node in ast.walk(tree)
                       if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)}
    own, own_package = _module_of(relative), _package_of(relative)
    homes = _defining_modules(project)
    roots = {module.split(".")[0] for entries in homes.values() for module, _, _ in entries}
    graph: dict[str, set[str]] | None = None
    lines: list[str] = []
    foreign: dict[str, str] = {}
    for name in undefined:
        if not name.isidentifier():
            continue
        if layered and name == "datetime" and name not in roots and not homes.get(name):
            line = _datetime_import(tree)
            if line:
                lines.append(line)
            continue
        if (name in sys.stdlib_module_names and not name.startswith("_") and name not in roots
                and name in attribute_bases):
            lines.append(f"import {name}")
            continue
        if (layered and name in _TEST_MODULES and name not in roots and not homes.get(name)
                and name in attribute_bases
                and (_TEST_FILE.search(relative) or Path(relative).name == "conftest.py")):
            lines.append(_TEST_MODULES[name])  # pytest.raises, @mock.patch in a test file
            continue
        sources = [entry for entry in homes.get(name, []) if entry[0] != own]
        if len(sources) == 1:
            module, source_relative, _ = sources[0]
            source_package = _package_of(source_relative)
            barred = [f for f in forbidden if _inside(module, f)] if layered else []
            if barred:
                foreign[name] = (f"is defined in {module}, and the task says {relative} must "
                                 f"not import {barred[0]}: do not import it; remove the code "
                                 f"that uses it from {relative}, or move that code to a layer "
                                 "that may use it")
                continue
            if own_package[:len(source_package)] != source_package:
                if layered:
                    theirs = ".".join(source_package) or "the top level"
                    ours = ".".join(own_package) or "the top level"
                    foreign[name] = (f"is defined in {module}, in {theirs}, a sibling package of "
                                     f"{ours} (not imported automatically: a layering choice). "
                                     f"If {ours} may use {theirs}, import it yourself (`from "
                                     f"{module} import {name}`); if not, remove the code that "
                                     f"uses it from {relative} or move it to {theirs}")
                continue  # a sibling package (domain <- web): a layering choice, not a typo
            graph = _import_graph(project, frozenset({own})) if graph is None else graph
            if not _reaches(graph, module, own):
                lines.append(_import_line(relative, content, module, name))
            continue
        if not sources and name in table:
            lines.append(f"from {table[name]} import {name}")
    if not lines:
        return AutoImports(content, [], foreign)
    return AutoImports(_insert_imports(content, lines), lines, foreign)


# -- project_reexport_guard / project_hold_back -----------------------------------------------------
# Why (2026-10-20): a package's `__init__.py` (the task never asked to change it) was rewritten
# to re-export names that are methods, and the package stopped importing for several rounds; an
# `__init__.py` was saved importing a module whose write was refused; saved files imported a
# sibling module that was never saved. Either way every test errors at collection.

def _import_target(node: ast.ImportFrom, package: list[str]) -> str:
    """The dotted module a `from ... import` statement reads from."""
    if node.level:
        base = package[:len(package) - node.level + 1] if node.level > 1 else package
        return ".".join([*base, *(node.module.split(".") if node.module else [])])
    return node.module or ""


def _render_from(node: ast.ImportFrom, aliases: list[ast.alias]) -> str:
    names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in aliases)
    return f"from {'.' * node.level}{node.module or ''} import {names}"


def bad_reexports(project: Path, relative: str, content: str) -> dict[str, list[str]]:
    """In a package __init__.py, module-level `from <project module> import a, b` names that the
    module does not define at module level (nor are submodules), and that the __init__ uses for
    nothing but the re-export: {module: [names]}. A module that does not exist, or whose names
    cannot be known statically, is left to the other checks."""
    if Path(relative).name != "__init__.py":
        return {}
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {}
    modules = _module_names(project)
    roots = {name.split(".")[0] for name in modules}
    package = _package_of(relative)
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)}
    found: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        target = _import_target(node, package)
        if not target or target.split(".")[0] not in roots or target not in modules:
            continue
        defined = modules[target]
        if defined is None:
            continue
        for alias in node.names:
            bound = alias.asname or alias.name
            if (alias.name == "*" or alias.name in defined or f"{target}.{alias.name}" in modules
                    or bound in used):
                continue
            found.setdefault(target, []).append(alias.name)
    return found


def drop_reexports(content: str, relative: str, bad: dict[str, list[str]]) -> str:
    """`content` without the re-exported names in `bad` ({module: [names]}): each import keeps its
    other names (or goes, when none is left), and a literal __all__ loses them too."""
    tree = ast.parse(content)
    package = _package_of(relative)
    gone: set[str] = set()
    edits_at: list[tuple[int, int, list[str]]] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            names = bad.get(_import_target(node, package), [])
            dropped = [a for a in node.names if a.name in names]
            if not dropped:
                continue
            gone.update(a.asname or a.name for a in dropped)
            kept = [a for a in node.names if a not in dropped]
            edits_at.append((node.lineno, node.end_lineno or node.lineno,
                             [_render_from(node, kept)] if kept else []))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "__all__"
                and isinstance(node.value, (ast.List, ast.Tuple))
                and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                        for e in node.value.elts)):
            names = [e.value for e in node.value.elts]  # type: ignore[attr-defined]
            kept_names = [n for n in names if n not in gone]
            if len(kept_names) < len(names):
                edits_at.append((node.lineno, node.end_lineno or node.lineno,
                                 ["__all__ = [" + ", ".join(repr(n) for n in kept_names) + "]"]))
    lines = content.split("\n")
    for start, end, replacement in sorted(edits_at, reverse=True):
        lines[start - 1:end] = replacement
    return "\n".join(lines)


def hold_back(project: Path, saved: list[str], refused: list[str],
              before: dict[str, str | None]) -> dict[str, str]:
    """project_hold_back: put back (to `before`, their content before the round) the files
    `saved` this round that import a module, or names from one, that does not resolve because
    its write in this round was refused; transitively, since a held file counts as refused for
    the others. Returns {held path: the refused file it depends on}."""
    blocked = {_module_of(path): path for path in refused if path.endswith(".py")}
    held: dict[str, str] = {}
    changed = bool(blocked)
    while changed:
        changed = False
        modules = _module_names(project)
        roots = {name.split(".")[0] for name in modules}
        for path in saved:
            if (path in held or path not in before or not path.endswith(".py")
                    or not (project / path).is_file()):
                continue
            try:
                tree = ast.parse((project / path).read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            package = _package_of(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    targets = [_import_target(node, package)]
                elif isinstance(node, ast.Import):
                    targets = [alias.name for alias in node.names]
                else:
                    continue
                hit = next((blocked[t] for t in targets if t in blocked), None)
                if hit and _import_problems(node, package, modules, roots):
                    held[path] = hit
                    break
            if path in held:
                _put_back(project, path, before.get(path))
                blocked[_module_of(path)] = path
                changed = True
    return held


def _modules_line(project: Path) -> str:
    return "Modules that exist in this project: " + ", ".join(sorted(_module_names(project)))


def _drop_path_line(code: str, path: str) -> str:
    """Remove a first line that only names the file (`api/limits.py`, `# api/limits.py`, `File:
    api/limits.py`): the 7B often starts a block with it, which is a syntax error in a .py file
    (seen in debug mode, goals 2026-10-14)."""
    lines = code.lstrip("\n").split("\n")
    first = lines[0].strip().strip("`*").strip()
    for prefix in ("#", "//", "File:", "file:", "Path:", "path:"):
        first = first.removeprefix(prefix).strip()
    if first.strip("`*: ") in {path, Path(path).name}:
        return "\n".join(lines[1:])
    return code


def _definitions_dropped(before: str, after: str) -> bool:
    from .tools import _top_level_names

    return bool(set(_top_level_names(before)) - set(_top_level_names(after)))


def _check(command: str, project: Path) -> str:
    if features.on("large_projects"):  # a large project's suite can take minutes
        return sandbox.run_check(command, project, timeout=LARGE_TEST_TIMEOUT)
    return sandbox.run_check(command, project)


def _no_tests(output: str) -> bool:
    """project_no_tests: pytest collected no tests (exit code 5): nothing failed."""
    return features.on("project_no_tests") and check_feedback.exit_code(output) == NO_TESTS


def _passed(output: str) -> bool:
    return output.startswith("exit code: 0") or _no_tests(output)


_SUMMARY_LINE = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.MULTILINE)
PYTEST = "python -m pytest -q -p no:cacheprovider"


def _pytest(project: Path, args: str = "-rfE", runs: list[str] | None = None) -> str:
    """One sandboxed pytest run: its output, or '' when it passed (or collected no tests with
    project_no_tests). `runs` gets the raw output."""
    output = _check(check_feedback.short_tracebacks(f"{PYTEST} {args}".strip()), project)
    if runs is not None:
        runs.append(output)
    return "" if _passed(output) else output


def _count_failing(output: str) -> int:
    """Failed plus errored tests in pytest's summary line (0 when there is none)."""
    lines = [line for line in output.splitlines() if line.strip()]
    summary = next((line for line in reversed(lines) if " failed" in line or " error" in line), "")
    failed = re.search(r"(\d+) failed", summary)
    errors = re.search(r"(\d+) error", summary)
    return (int(failed.group(1)) if failed else 0) + (int(errors.group(1)) if errors else 0)


def _failing_tests(project: Path, runs: list[str] | None = None) -> tuple[set[str], str]:
    """(ids of failing tests and test files that fail to load, the output); ({}, '') when all
    pass. A run that ends without naming any (a timeout, a crash) gives {'(the test run)'}.
    `runs` gets the raw output."""
    output = _pytest(project, runs=runs)
    if not output:
        return set(), ""
    return set(_SUMMARY_LINE.findall(output)) or {"(the test run)"}, output


def _run_new_failures(project: Path, known: set[str], new_ids: list[str] | None = None,
                      runs: list[str] | None = None) -> tuple[int, str, list[str]]:
    """Like _run_tests, but tests in `known` (failing before the change) are not counted as the
    change's failures: (newly failing, the output, known tests still failing). Their output is
    still shown, labelled, since the task may be to fix them. `new_ids` gets the newly failing
    ids (a test file that fails to load appears as its path)."""
    failing, output = _failing_tests(project, runs)
    new = sorted(failing - known)
    if new_ids is not None:
        new_ids += new
    still = sorted(failing & known)
    if not failing:
        return 0, "", []
    note = (f"\n(Failing before the change too: {', '.join(still)}. Fix them only if the task is "
            "about them.)" if still else "")
    return len(new), _failure_text(output) + note, still


_COLLECT_ERROR = re.compile(r"^(?:_+ )?ERROR (?:collecting )?(\S+?\.py)(?=\s|$)", re.MULTILINE)
_CONFIG_ERROR = re.compile(r"^ERROR: (.+)$", re.MULTILINE)


def load_errors(output: str) -> list[str]:
    """What in a pytest run could not even load: test files with an import or syntax error at
    collection, or a config pytest cannot read (`ERROR: ./pyproject.toml: ...`, exit code 4).
    One of these hides every test behind it, so it ranks below any number of failing tests
    (otherwise a module that would not import was kept as '1 failing')."""
    found = [f"{path} does not load (import or syntax error at collection)"
             for path in dict.fromkeys(_COLLECT_ERROR.findall(output))]
    code = check_feedback.exit_code(output)
    if code in (3, 4):
        found += [f"pytest could not start: {line.strip()[:200]}"
                  for line in _CONFIG_ERROR.findall(output)[:2]] or [
            f"pytest could not start (exit code {code})"]
    elif code == 2 and not found:
        found.append("the test run was interrupted at collection (exit code 2)")
    return found


def _run_tests_detail(project: Path, runs: list[str] | None = None,
                      ) -> tuple[int, str, list[str], list[str]]:
    """pipeline_signals: (failing count, repair text, load errors, failing test ids). `runs`
    gets the raw output."""
    output = _pytest(project, runs=runs)
    if not output:
        return 0, "", [], []
    loads = load_errors(output)
    failing = max(_count_failing(output), len(loads)) or 1
    ids = [i for i in dict.fromkeys(_SUMMARY_LINE.findall(output)) if "::" in i]
    return failing, _failure_text(output), loads, ids


def _run_tests(project: Path, target: str = "", runs: list[str] | None = None) -> tuple[int, str]:
    """(number of failing tests or 1 for a collection error, the tail of the output); (0, '')
    when everything passes. `target` limits the run to one path; `runs` gets the raw output."""
    output = _pytest(project, target, runs)
    if not output:
        return 0, ""
    return _count_failing(output) or 1, _failure_text(output)


def _failure_text(output: str) -> str:
    """What the repair prompt shows of a failing run: with repair_feedback, the failing ids,
    their assertion lines and the nearest project line (py baseline: the last 40 lines were often
    asyncio or importlib frames); else, or when that is no shorter, the last 40 lines."""
    if features.on("repair_feedback"):
        condensed = check_feedback.condense(output, note=False)
        if condensed != output:
            return condensed.partition("\n")[2]
    lines = [line for line in output.splitlines() if line.strip()]
    return "\n".join(lines[-TEST_OUTPUT_LINES:])


_LOAD_ERROR = re.compile(r"^ERROR (\S+?\.py)(?=\s|$)", re.MULTILINE)


def _failed_ids(project: Path) -> set[str]:
    """Test files that fail to load (`ERROR path.py`: an import or syntax error at collection).

    Only these count as damage: a change can be meant to make old assertions fail (a behaviour
    the task changes), but never to make a test module unimportable. Rolling back on assertion
    failures reverted needed changes (goals 2026-10-14: inventory lost hidden tests it had
    passed); on load errors it rescued a broken project (config-env-overrides)."""
    return set(_LOAD_ERROR.findall(_pytest(project)))


def run_script(project: Path, script: str, outputs: list[str]) -> tuple[dict[str, str], str]:
    """project_run_script: run `python <script>` in the sandbox; (the outputs it wrote, a
    problem or ''). An output it did not write as a file but printed (`save exactly the JSON
    it prints as out.json`) is taken from what it printed when that is one JSON value."""
    if not (project / script).is_file():
        return {}, f"`python {script}` cannot run: {script} does not exist (write it)."
    command = (f"python -c {shlex.quote(SCRIPT_RUNNER)} {shlex.quote(json.dumps(outputs))} "
               f"{shlex.quote(script)}")
    output = sandbox.run_check(command, project)
    if SCRIPT_MARKER not in output:
        tail = "\n".join(output.splitlines()[-TEST_OUTPUT_LINES:])
        return {}, f"`python {script}` failed:\n{tail}"
    printed, _, result = output.partition(SCRIPT_MARKER)
    try:
        report = json.loads(result.strip().splitlines()[0])
    except (ValueError, IndexError):
        return {}, f"`python {script}`: its result could not be read."
    files: dict[str, str] = report.get("files", {})
    printed = printed.split("\n", 1)[1] if printed.startswith("exit code:") else printed
    missing = [o for o in outputs if o not in files]
    if len(missing) == 1 and missing[0].endswith(".json"):
        with contextlib.suppress(ValueError):
            json.loads(printed)
            files[missing[0]] = printed.strip() + "\n"
            missing = []
    if report.get("exit"):
        return files, f"`python {script}` exited with code {report['exit']}:\n{printed[-1500:]}"
    if missing:
        return files, (f"`python {script}` ran but did not write {', '.join(missing)}. Its "
                       f"output:\n{printed[-1500:] or '(none)'}")
    return files, ""


def _run_scripts(host: write_loop.PipelineHost, extras: _Extras, touched: set[str],
                 originals: dict[str, str | None]) -> list[str]:
    """Run each script the task names and save the outputs it wrote; the problems, if any."""
    project = host.project
    problems = []
    for script in extras.scripts:
        command = f"python {script}"
        host.emit("tool_call", name="run_check", args={"command": command})
        files, problem = run_script(project, script, extras.outputs)
        saved = []
        for path, content in files.items():
            target = project / path
            before = target.read_text(encoding="utf-8") if target.is_file() else None
            originals.setdefault(path, before)
            if before == content:
                saved.append(path)
                extras.produced[path] = script
                continue
            result = execute("write_file", {"path": path, "content": content}, project,
                             before_write=host.before_write, atomic=False)
            extras.notes[path] = result
            if result.startswith(("Wrote", "Edited")):
                touched.add(path)
                saved.append(path)
                extras.produced[path] = script
        host.refresh_written()
        host.emit("tool_result", name="run_check", args={"command": command},
                    result=problem or f"wrote {', '.join(saved) or 'nothing'}")
        debuglog.record("stage", stage="project_pipeline.run_script", script=script,
                        saved=saved, problem=problem[:600])
        if problem:
            problems.append(problem)
    return problems


def _put_back(project: Path, path: str, content: str | None) -> None:
    target = project / path
    if content is None:
        target.unlink(missing_ok=True)
    else:
        target.write_text(content, encoding="utf-8")


def _rollback_breakers(host: write_loop.PipelineHost, originals: dict[str, str | None],
                       baseline: set[str], spec: bool, contract: list[str]) -> list[str]:
    """Test files that loaded before the change now fail to load: try each touched file back at
    its original content, newest first, and keep a revert when it reduces that breakage without
    failing more spec tests or task names. Returns what was reverted."""
    project = host.project
    broken = _failed_ids(project) - baseline
    if not broken:
        return []

    def side_costs() -> tuple[int, int]:
        spec_failing = _run_tests(project, SPEC_DIR)[0] if spec else 0
        return spec_failing, len(_check_contract(project, contract))

    costs = side_costs()
    reverted = []
    for path in reversed(list(originals)):
        current = (project / path).read_text(encoding="utf-8") if (project / path).is_file() else None
        if current == originals[path]:
            continue
        _put_back(project, path, originals[path])
        after = _failed_ids(project) - baseline
        after_costs = side_costs() if len(after) < len(broken) else costs
        if len(after) < len(broken) and after_costs[0] <= costs[0] and after_costs[1] <= costs[1]:
            reverted.append(f"{path} (it broke {', '.join(sorted(broken - after)[:3])})")
            broken, costs = after, after_costs
            if not broken:
                break
        else:
            _put_back(project, path, current)
    if reverted:
        host.refresh_written()
        debuglog.record("stage", stage="project_pipeline.rollback", reverted=reverted)
        host.emit("notice", message="Put back unchanged: " + "; ".join(reverted))
    return reverted


_DOTTED = re.compile(r"`([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?:\([^`]*\))?`")
CONTRACT_SCRIPT = r'''
import importlib, json, sys
sys.path[:0] = ["src", "."]  # src/ layout and flat projects
missing = []
for dotted in json.loads(sys.argv[1]):
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        try:
            for name in parts[cut:]:
                obj = getattr(obj, name)
        except AttributeError:
            missing.append(f"{dotted}: {'.'.join(parts[:cut])} has no {name}")
        break
    else:
        missing.append(f"{dotted}: no module {parts[0]} can be imported")
print(json.dumps(missing))
'''


def contract_names(project: Path, task: str) -> list[str]:
    """Dotted names the task gives in backticks whose first part is a package of this project
    (e.g. `rollup.aggregate.monthly_summary(directory)`), to check they exist after the change."""
    packages = {p.parent.name for p in _project_files(project) if p.name == "__init__.py"}
    names = []
    for dotted in _DOTTED.findall(task):
        if dotted.split(".")[0] in packages and dotted not in names:
            names.append(dotted)
    return names


def _check_contract(project: Path, names: list[str]) -> list[str]:
    if not names:
        return []
    command = f"python -c {shlex.quote(CONTRACT_SCRIPT)} {shlex.quote(json.dumps(names))}"
    output = _check(command, project)
    lines = [line for line in output.splitlines() if line.strip()]
    try:
        return json.loads(lines[-1])
    except (IndexError, ValueError):
        return [f"could not check the task's names: {output[-200:]}"]


def _write_spec_tests(agent: CodingAgent, task: str, repo_map: str) -> str:
    """Write tests for the task into .vega-spec/; keep them only if they fail on the unchanged
    project (tests that already pass say nothing about the change). Returns a note for the trace."""
    project = agent.project
    reply = _ask(agent, [{"role": "system", "content": SPEC_SYSTEM},
                         {"role": "user", "content": f"Task:\n{task}\n\nProject map:\n{repo_map}"}])
    blocks = write_loop.code_blocks(reply)
    if not blocks:
        return "spec tests: none written"
    path = project / SPEC_FILE
    path.parent.mkdir(exist_ok=True)
    path.write_text(max(blocks, key=len).strip("\n") + "\n", encoding="utf-8")
    failing, _ = _run_tests(project, SPEC_DIR)
    debuglog.record("stage", stage="project_pipeline.spec_tests", failing_before=failing,
                    tests=path.read_text(encoding="utf-8"))
    if not failing:
        _remove_spec(project)
        return "spec tests: discarded (they already pass before the change)"
    return f"spec tests: written, {failing} failing before the change"


def _remove_spec(project: Path) -> None:
    import shutil

    shutil.rmtree(project / SPEC_DIR, ignore_errors=True)


def _snapshot(project: Path, paths: set[str]) -> dict[str, str | None]:
    return {p: ((project / p).read_text(encoding="utf-8") if (project / p).is_file() else None)
            for p in paths}


def empty_plan_note(dropped: list[dict[str, str]]) -> str:
    """What the planner is told when its plan lists no file to write."""
    if dropped:
        listed = "; ".join(f"{d['path']} ({d['change'][:80]})" for d in dropped[:6])
        reason = f"Your plan listed only files that need no change: {listed}."
    else:
        reason = "Your plan listed no files."
    return (reason + " Vega writes only the files a plan lists, so nothing would change. List "
            "each file that must change or be created for this task, with the change it needs. "
            "List a file only if it changes.")


def _extras(agent: CodingAgent, task: str) -> _Extras:
    """The route's inputs (project_inputs) and the task's scripts and their outputs
    (project_run_script) for this run; empty when those switches are off."""
    extras = _Extras()
    want_inputs, want_scripts = features.on("project_inputs"), features.on("project_run_script")
    if not (want_inputs or want_scripts):
        return extras
    route = _route_record(agent, task)
    if want_scripts:
        extras.scripts, extras.outputs = _script_outputs(agent.project, route)
    if want_inputs:
        extras.inputs = [p for p in read_only_inputs(agent.project, route)
                         if p not in extras.outputs]
    debuglog.record("stage", stage="project_pipeline.extras", inputs=extras.inputs,
                    scripts=extras.scripts, outputs=extras.outputs)
    return extras


def _why_left(path: str, note: str) -> str:
    """A planned file's last write result, as the reason it is unchanged."""
    if not note:
        return "not written"
    if note.startswith(("Wrote", "Edited")):
        return "the version kept is the same as before"
    first = next((line.strip() for line in note.splitlines() if line.strip()), note)
    first = first.removeprefix("Not written: ").removeprefix(f"{path} ").removeprefix("is ")
    first = re.sub(r"\s*\(asked twice;[^)]*\)\s*$", "", first)
    first = first.removesuffix(".").removesuffix("; left unchanged")
    return first[:200].rstrip(".")


def _open_list(open_items: list[str]) -> str:
    more = (f"\n- … and {len(open_items) - OPEN_SHOWN} more" if len(open_items) > OPEN_SHOWN
            else "")
    return "\n".join(f"- {line}" for line in open_items[:OPEN_SHOWN]) + more


def _outcome(changed: str, project_failing: int, still: int, spec_note: str,
             open_items: list[str], no_tests: bool = False) -> str:
    """The answer's first part, for files changed: what the kept round's checks say (large_projects:
    never hide what still fails; acceptance and pipeline_signals: the open items, in the task's
    terms)."""
    if project_failing == 0 and still:
        return (f"Changed {changed}; no test that passed before fails now, but {still} test(s) "
                f"that failed before the change still fail.{spec_note}")
    if project_failing == 0 and no_tests:
        return f"Changed {changed}; the project has no tests, so no test ran.{spec_note}"
    if project_failing == 0:
        return f"Changed {changed}; the project's tests pass.{spec_note}"
    if open_items:
        return (f"Changed {changed}, but after {REPAIRS} repair rounds these still fail:\n"
                + _open_list(open_items) + spec_note)
    return (f"Changed {changed}, but {project_failing} test(s) still fail after {REPAIRS} "
            f"repair rounds; it needs another look.{spec_note}")


def honest_answer(project: Path, extras: _Extras, project_failing: int, spec_note: str,
                  still: int, open_items: list[str]) -> str:
    """project_honest_answer: the files written (their content differs from before the change)
    and every planned file left unchanged with the reason; no claim about the tests when nothing
    was written, and none that they pass when no test ran."""
    written = [p for p, before in extras.originals.items()
               if write_loop._read(project / p) != before]
    left = [p for p in extras.originals if p not in written]
    left_note = ("Left unchanged: " + "; ".join(f"{p} ({_why_left(p, extras.notes.get(p, ''))})"
                                                for p in left) + ".") if left else ""
    if not written:
        answer = "No file was changed, so the task is not done."
        if left_note:
            answer += " " + left_note
        if open_items:
            answer += "\nStill open:\n" + _open_list(open_items)
        return answer
    answer = _outcome(", ".join(written), project_failing, still, spec_note, open_items,
                      extras.no_tests)
    scripts: dict[str, list[str]] = {}
    for output, script in extras.produced.items():
        if output in written:
            scripts.setdefault(script, []).append(output)
    for script, outputs in scripts.items():
        answer += f"\n`python {script}` ran in the sandbox and wrote {', '.join(outputs)}."
    if left_note:
        answer += "\n" + left_note
    return answer


def run(agent: CodingAgent, task: str) -> str | None:
    """Run a task routing.route_task sent here; None falls back to the agent loop."""
    from .index import build_index

    project = agent.project
    index = build_index(project)
    if features.on("large_projects") and len(index.files) > MAX_PROJECT_FILES:
        repo_map = large_map(index, task)
    else:
        repo_map = index.repo_map(max_chars=MAP_CHARS)
    agent._emit("notice", message="Project pipeline: plan files, write them, run the tests, repair")
    signals = features.on("pipeline_signals")
    extras = _extras(agent, task)
    planner_inputs = inputs_text(project, extras.inputs)
    required = ([seed for seed in route_targets(project, _route_record(agent, task), task)
                 if seed["path"] not in extras.outputs]
                if features.on("project_route_targets") else [])
    if features.on("plan_contradicting_tests") and not _ASKS_FOR_TESTS.search(task):
        with contextlib.suppress(Exception):  # no detection is the old behaviour
            extras.call_tests = {test.path: test for test in contradicting_tests(project, task)}
    dropped: list[dict[str, str]] = []
    try:
        plan = _plan(agent, task, repo_map, dropped=dropped, inputs=planner_inputs,
                     outputs=extras.outputs, required=required, call_tests=extras.call_tests)
        if not plan and signals:  # told why, one retry (the planner replied '"files": []')
            note = empty_plan_note(dropped)
            debuglog.record("stage", stage="project_pipeline.empty_plan", note=note)
            agent._emit("discarded", reason="the plan listed no file to change", retry=True)
            plan = _plan(agent, task, repo_map, note=note, inputs=planner_inputs,
                         outputs=extras.outputs, required=required,
                         call_tests=extras.call_tests)
    except Exception as exc:  # noqa: BLE001 - fall back to the agent loop
        agent._emit("notice", message=f"Project pipeline stopped ({exc}); using the agent loop.")
        return None
    if required:
        plan, added = with_route_targets(plan, required)
        debuglog.record("stage", stage="project_pipeline.route_targets",
                        required=[seed["path"] for seed in required], added=added)
    if extras.call_tests and plan:  # never the only file: the task's own change comes first
        plan, added = with_call_tests(plan, list(extras.call_tests.values()))
        debuglog.record("stage", stage="project_pipeline.call_tests", added=added,
                        calls={t.path: t.calls for t in extras.call_tests.values()})
    if not plan:
        if signals:
            agent._emit("notice", message="Project pipeline: the planner listed no file to change "
                                          "(asked twice); using the agent loop.")
        return None
    agent._emit("plan", items=[{"text": f"{p['path']}: {p['change'][:80]}", "status": "pending"}
                               for p in plan])
    touched: set[str] = set()
    trace: list[str] = []
    contract = contract_names(project, task)
    debuglog.record("stage", stage="project_pipeline.contract", names=sorted(contract))
    spec = False
    if features.on("spec_tests"):
        with contextlib.suppress(Exception):  # no spec tests is the old behaviour
            trace.append(_write_spec_tests(agent, task, repo_map))
            spec = (project / SPEC_FILE).is_file()
    acceptance.ensure(agent, task)
    checks = _task_checks(agent, task, repo_map, trace)
    runner = _Rounds(agent, task, repo_map, plan, contract, spec, touched, trace, extras,
                     planner_inputs, checks)
    try:  # the best round (by its rank) is restored at the end
        result = write_loop.run_loop(runner.host, runner.config(), [])
    finally:
        _remove_spec(project)
    assert result is not None and result.kept is not None and result.kept.checks is not None
    kept = result.kept.checks
    assert isinstance(kept, _Checks)
    agent.messages += [{"role": "user", "content": task},
                       {"role": "assistant",
                        "content": "(project pipeline)\n" + "\n\n".join(trace)}]
    project_failing, spec_failing, still = kept.failing, kept.spec_failing, len(kept.still)
    what = ("tests written from the task" if spec and checks is None else
            "task checks" if not spec else "tests and checks written from the task")
    spec_note = (f" {spec_failing} of the {what} still fail (they may be wrong themselves)."
                 if spec_failing else "")
    if features.on("project_honest_answer"):
        answer = honest_answer(project, extras, project_failing, spec_note, still,
                               kept.open_items)
    else:
        answer = _outcome(", ".join(sorted(touched)) or "no file", project_failing, still,
                          spec_note, kept.open_items)
    if getattr(agent, "teach", False):  # teach mode: a note from the real stages only
        rounds = [entry.split("\n", 1)[0].rstrip(":") for entry in trace
                  if entry.startswith("round ")]
        answer += "\n\n" + teaching.project_note(plan, rounds)
    return answer


def _task_checks(agent: CodingAgent, task: str, repo_map: str,
                 trace: list[str]) -> task_checks.TaskChecks | None:
    """task_checks / task_checks_model: the task's own statements as checks, gated on the
    unchanged project before any write (None when neither is on or none is kept)."""
    rules, model = features.on("task_checks"), features.on("task_checks_model")
    if not rules and not model:
        return None
    found: list[task_checks.Check] = []
    dropped: list[str] = []
    if model:
        try:
            found, dropped = task_checks.ask_model(agent, task, repo_map)
        except Exception as exc:  # noqa: BLE001 - no model checks is the old behaviour
            dropped = [f"the model call failed ({exc})"]
    try:
        checks = task_checks.prepare(agent.project, task, rules=rules, model=found,
                                     model_dropped=dropped, runner=_check)
    except Exception:  # noqa: BLE001 - no task checks is the old behaviour
        return None
    trace.extend(checks.notes)
    if not checks.active:
        return None
    agent._emit("notice", message="Task checks: " + ", ".join(c.label() for c in checks.active))
    return checks


@dataclass
class _Checks(write_loop.Checks):
    """One round's check stage: the parts of its rank and its open items."""

    loads: list[str] = field(default_factory=list)  # test files / task modules that won't load
    failing: int = 0  # project tests, the task's scripts and names, acceptance, imports, refusals
    spec_failing: int = 0
    still: list[str] = field(default_factory=list)  # failing before the change and still failing
    open_items: list[str] = field(default_factory=list)  # for the answer

    @property
    def failed(self) -> int:
        return self.failing + self.spec_failing

    def score(self) -> tuple[int, int, int, int]:
        return len(self.loads), self.failing, self.spec_failing, len(self.still)


class _Rounds:
    """The rounds as a several-target config of write_loop.run_loop: each planned file is written
    by its own writer call; the check stage runs the tests (and the scripts, the task's names,
    acceptance and import checks); rounds rank by (load errors, project failing, spec failing,
    failing before the change and still failing); the planner replans between rounds.

    Breaking the project's tests or the task's names ranks worse than failing a spec test; the
    last count is 0 unless large_projects. Failing acceptance items count like failing tests.

    pipeline_signals: a test file or task module that will not import (or a config pytest cannot
    read) ranks worst, below any number of failing tests (the first count; 0 with it off); a
    planned write that was refused or never sent (wrong file, no block) counts as failing and its
    reason goes to that file's writer next round; the open items are every open item of the best
    round (refused writes, load errors, failing tests, acceptance), for the final answer."""

    def __init__(self, agent: CodingAgent, task: str, repo_map: str, plan: list[dict[str, str]],
                 contract: list[str], spec: bool, touched: set[str], trace: list[str],
                 extras: _Extras, planner_inputs: str,
                 checks: task_checks.TaskChecks | None = None):
        self.agent, self.project, self.task, self.repo_map = agent, agent.project, task, repo_map
        self.host = write_loop.AgentHost(agent)
        self.plan, self.contract, self.spec, self.touched = plan, contract, spec, touched
        self.trace, self.extras, self.planner_inputs = trace, extras, planner_inputs
        self.task_checks = checks
        self.signals = features.on("pipeline_signals")
        self.rollback = features.on("rollback_breakers")
        self.baseline = _failed_ids(self.project) if self.rollback else set()
        self.known: set[str] | None = None
        if features.on("large_projects"):
            self.known = _failing_tests(self.project)[0] - {"(the test run)"}
            if self.known:
                trace.append("failing before the change: " + ", ".join(sorted(self.known)))
        self.previous_still = len(self.known) if self.known else 0
        self.failure = ""  # the last round's failing output, for the writers and the planner
        self.refusals: dict[str, str] = {}  # path -> why its write this round was not saved
        self.last_refusals: dict[str, str] = {}
        self.round = -1  # the round being checked (targets() starts each)
        self.round_start: dict[str, str | None] = {}  # this round's targets before its writes

    def config(self) -> write_loop.LoopConfig:
        return write_loop.LoopConfig(
            name="project_pipeline", title="Project pipeline", targets=[p["path"] for p in self.plan],
            originals=self.extras.originals, repairs=REPAIRS, check=self.check,
            produce_each=self.produce, round_targets=self.targets, on_write=self.wrote,
            score=lambda checks: checks.score(),  # type: ignore[attr-defined]
            done=self.done, between=self.replan, check_call="pytest",
            snapshot=lambda: _snapshot(self.project, self.touched),
            on_unchanged=self.unchanged)

    def unchanged(self, note: str) -> None:
        """unchanged_rounds: a round left every file as it was; its writers are told so, ahead of
        the last check's failing output (still current, since nothing changed)."""
        self.failure = note + ("\n\n" + self.failure if self.failure else "")
        self.trace.append(f"round {self.round + 1}: every planned file rewritten unchanged; "
                          "checks not run again")

    def targets(self) -> list[str]:
        self.last_refusals, self.refusals = self.refusals, {}
        self.round += 1
        paths = [item["path"] for item in self.plan]
        self.round_start = _snapshot(self.project, set(paths))
        return paths

    def produce(self, path: str) -> write_loop.Produced:
        item = next(p for p in self.plan if p["path"] == path)
        target = self.project / path
        self.extras.originals.setdefault(path, target.read_text(encoding="utf-8")
                                         if target.is_file() else None)
        return _produce_file(self.agent, self.task, item, self.plan, self.failure,
                             self.last_refusals.get(path, ""), self.extras)

    def wrote(self, path: str, result: str) -> None:
        self.extras.notes[path] = result
        if self.signals and not result.startswith(("Wrote", "Edited")) \
                and "too large to rewrite" not in result:
            self.refusals[path] = result
            return
        self.touched.add(path)

    def _grounded_round(self, saved: list[str]) -> None:
        """project_hold_back and project_reexport_guard, after the round's writes and before its
        checks: files that import a module whose write was refused this round are put back and
        counted as not saved; a package __init__.py's re-exports of names its source module
        does not define are dropped."""
        project, extras, round_number = self.project, self.extras, self.round
        if features.on("project_hold_back") and self.signals and self.refusals:
            held = hold_back(project, saved, list(self.refusals), self.round_start)
            for path, cause in held.items():
                why = "held back" if cause in held else "refused"
                note = (f"Held back, not saved: {path} imports {_module_of(cause) or cause} "
                        f"({cause}), whose write this round was {why}, so saving it would leave "
                        f"the package unimportable; {path} is as it was before this round. "
                        f"Write it again so it works with {cause} as it is now.")
                self.refusals[path] = note
                extras.notes[path] = note
            if held:
                self.host.refresh_written()
                listed = ", ".join(f"{p} (imports {c})" for p, c in held.items())
                self.trace.append(f"round {round_number + 1}: held back {listed}")
                debuglog.record("stage", stage="project_pipeline.hold_back",
                                round=round_number + 1, held=held)
                self.host.emit("notice", message="Held back (they import a refused write): "
                                                 + listed)
        if features.on("project_reexport_guard"):
            for path in sorted(self.touched):
                content = write_loop._read(project / path)
                if content is None or Path(path).name != "__init__.py":
                    continue
                bad = bad_reexports(project, path, content)
                if not bad:
                    continue
                fixed = drop_reexports(content, path, bad)
                result = execute("write_file", {"path": path, "content": fixed}, project,
                                 before_write=self.host.before_write, atomic=False)
                if not result.startswith(("Wrote", "Edited")):
                    continue
                listed = "; ".join(f"{', '.join(names)} (not defined in {module})"
                                   for module, names in bad.items())
                self.host.refresh_written()
                self.trace.append(f"round {round_number + 1}: dropped from {path}'s re-exports: "
                                  + listed)
                debuglog.record("stage", stage="project_pipeline.reexport_guard",
                                round=round_number + 1, path=path, dropped=bad)
                self.host.emit("notice", message=f"Dropped from {path}'s re-exports: {listed}")

    def check(self, saved: list[str]) -> _Checks:
        project, extras, signals = self.project, self.extras, self.signals
        round_number = self.round
        self._grounded_round(saved)
        reverted = (_rollback_breakers(self.host, extras.originals, self.baseline, self.spec,
                                       self.contract) if self.rollback else [])
        if reverted:
            self.trace.append(f"round {round_number + 1}: put back " + "; ".join(reverted))
            for entry in reverted:
                path, _, why = entry.partition(" (")
                extras.notes[path] = "put back as it was: " + why.rstrip(")")
        script_problems = (_run_scripts(self.host, extras, self.touched, extras.originals)
                           if extras.scripts else [])
        still: list[str] = []
        loads: list[str] = []
        test_ids: list[str] = []
        runs: list[str] = []
        if self.known is None and signals:
            failing, failure, loads, test_ids = _run_tests_detail(project, runs)
        elif self.known is None:
            failing, failure = _run_tests(project, runs=runs)
        else:
            new_ids: list[str] = []
            failing, failure, still = _run_new_failures(project, self.known, new_ids, runs)
            loads = [f"{i} does not load (import or syntax error at collection)" for i in new_ids
                     if "::" not in i and i.endswith(".py")] if signals else []
            test_ids = [i for i in new_ids if "::" in i]
        extras.no_tests = bool(runs) and _no_tests(runs[-1])

        def lead(header: str, items: list[str], after: str = "") -> str:
            return (header + "\n".join(f"- {item}" for item in items) + after
                    + ("\n\n" + failure if failure else ""))

        if script_problems:  # the task's own script is a check too: it must run and write
            failing += len(script_problems)
            failure = lead("The task's script did not produce its output:\n", script_problems)
        missing = _check_contract(project, self.contract)
        if missing:
            failing += len(missing)
            failure = lead("The task names these, but they do not exist after the change:\n",
                           missing)
            if signals:  # a task name whose module will not import hides everything behind it
                loads += [m for m in missing if "can be imported" in m]
        accept_spec = getattr(self.agent, "_acceptance", None)
        lines = []
        if accept_spec is not None:  # passing the visible tests is not enough while these fail
            accepted = acceptance.run(accept_spec, project, sorted(self.touched), tests=False,
                                      stop=getattr(self.agent, "_stop", None))
            lines = acceptance.pipeline_failures(accepted)
            debuglog.record("stage", stage="project_pipeline.acceptance", round=round_number + 1,
                            checks=len(accepted.items), failing=lines)
            if lines:
                failing += len(lines)
                failure = lead("Vega's checks from the task's own words fail:\n", lines)
            if signals:
                loads += [item.line() for item in accepted.failing if item.kind == "import"]
        unresolved = (unresolved_imports(project, sorted(self.touched))
                      if features.on("import_grounding") else [])
        if unresolved:
            failing += len(unresolved)
            failure = lead("These imports do not resolve:\n", unresolved,
                           "\n" + _modules_line(project))
        refusals = self.refusals
        if refusals:  # never landed: the round failed whatever the tests say about the old file
            failing += len(refusals)
            failure = lead("These planned writes were not saved (the files are unchanged):\n",
                           [f"{path}: {text.splitlines()[0]}" for path, text in refusals.items()])
        spec_failing, spec_output = _run_tests(project, SPEC_DIR) if self.spec else (0, "")
        if spec_failing:
            failure += ("\n\n" if failure else "") + (
                "Tests written from the task fail (if one contradicts the task, the task wins):\n"
                + spec_output)
        task_lines: list[str] = []
        if self.task_checks is not None:  # ranked after the project's own checks, like spec tests
            notes = len(self.task_checks.notes)
            task_lines = self.task_checks.evaluate(project, failing)
            self.trace.extend(self.task_checks.notes[notes:])
            counted = [line for line in task_lines if not line.endswith("(no longer checked)")]
            spec_failing += len(counted)
            if task_lines:
                failure += ("\n\n" if failure else "") + (
                    "Vega ran the task's own statements as checks, and these fail (if one "
                    "contradicts the task, the task wins):\n"
                    + "\n".join(f"- {line}" for line in task_lines))
        summary = ("all tests pass" if not failing and not spec_failing and not still else
                   f"{failing} project test(s) failing" + (f", {spec_failing} spec test(s) failing"
                                                           if self.spec else ""))
        counted_checks = [line for line in task_lines if not line.endswith("(no longer checked)")]
        if counted_checks and not self.spec:
            summary += f", {len(counted_checks)} task check(s) failing"
        if loads:
            summary += f" ({len(loads)} cannot load)"
        if still:
            summary += f", plus {len(still)} that failed before the change too"
        wrote = [p["path"] for p in self.plan if p["path"] not in refusals]
        self.trace.append(f"round {round_number + 1}: wrote {', '.join(wrote) or 'nothing'}; "
                          + summary + (f":\n{failure}" if failure else ""))
        self.failure = failure
        if signals:
            open_items = list(dict.fromkeys(
                [f"{path}: not saved — {text.splitlines()[0]}" for path, text in refusals.items()]
                + loads + [f"test {test_id}" for test_id in test_ids] + missing + lines
                + unresolved + script_problems
                + [f"task check {line.splitlines()[0]}" for line in counted_checks]))
        else:
            open_items = lines
        return _Checks(report=summary + (f"\n{failure}" if failure else ""), loads=loads,
                       failing=failing, spec_failing=spec_failing, still=still,
                       open_items=open_items)

    def done(self, checks: write_loop.Checks, round_number: int) -> bool:
        """All pass; tests that failed before the change too (large_projects) keep the repairs
        going only while they are being fixed: once a repair round fixes none of them, stop."""
        assert isinstance(checks, _Checks)
        still = checks.still
        stalled = round_number > 0 and len(still) >= self.previous_still
        self.previous_still = len(still)
        return checks.score()[:3] == (0, 0, 0) and (not still or stalled)

    def replan(self, _round_number: int) -> None:
        with contextlib.suppress(Exception):  # a failed replan keeps the previous plan
            replan = _plan(self.agent, self.task, self.repo_map, self.failure,
                           inputs=self.planner_inputs, outputs=self.extras.outputs,
                           call_tests=self.extras.call_tests) or self.plan
            if self.extras.call_tests:  # a contradicting test that still fails stays planned
                replan = with_call_tests(replan, failing_call_tests(self.extras.call_tests,
                                                                    self.failure))[0]
            if self.signals:  # a refused write is still owed: keep its file in the plan
                listed = {p["path"] for p in replan}
                replan = replan + [p for p in self.plan if p["path"] in self.refusals
                                   and p["path"] not in listed]
            self.plan = replan
