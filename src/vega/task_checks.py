"""Task checks: small executable checks of what the task itself states, for the project pipeline.

The project's visible tests cover only existing behaviour, so "the tests pass" often ended a run
whose new logic was wrong (2026-10-20). These checks look at the task's own statements instead:

- `task_checks` (rules, no model call):
  - CLI smoke: a backticked command whose first word is an argparse `prog=` of the project
    (`app show --verbose`) is run through that module's `main(argv)` in the sandbox. It fails
    when argparse rejects a word the task wrote (invalid choice, unrecognized arguments), or, for a
    command without placeholders, when it ends in a traceback or usage error that the unchanged
    project did not have.
  - `python -m pkg ...` commands (sub-switch `task_module_commands`): a backticked command, or
    one segment of a backticked shell pipeline (`python -m pkg --where qty>=7 < items.csv |
    python -m pkg --columns name`), that runs a package or module of the project is run
    in-process with runpy and judged like a CLI smoke check. A segment that reads stdin (after
    `<` or `|`, or with a `-` argument) is fed a small input: the redirected file when the
    project has it, else a sample of the same format (CSV, TSV, JSON) from the visible tests
    (`path.write_text("name,qty\nnut,3\n...")`) or a fenced README block, else nothing. A bare
    `python -m pkg` is a name, not a command: acceptance (same switch) checks the full
    invocation the task or its README gives instead (`full_invocations`), and never counts
    argparse's usage exit of the bare name as a failure.
  - Environment examples: "APP_PORT=9000 overrides `port`" calls the project's settings loader
    (a module-level function with no required parameters whose result has that key) with only
    that variable set, and expects the value converted to the setting's type.
- `task_checks_model` (opt-in, one short constrained-JSON model call): up to 5 checks of a few
  lines each, every one quoting the task words it checks.

Why not spec_tests again: its whole-file tests (off since the 2026-10-14 A/Bs: no measured gain) were kept
whenever they failed before the change, and most failed for reasons of their own (a wrong fixture
signature, an import from `src.`, a setting the project does not have, `subprocess` on an
uninstalled script), so the repairs chased tests that could never pass. Here each check stands
alone and is gated:

- before any write, on the unchanged project, a check is kept only when it fails *because the
  behaviour is missing*: an assertion, or a missing name that the task itself mentions (model
  checks); an env example whose loader cannot be found, or a check that already passes, is
  dropped. CLI smoke checks that pass are kept (they guard the command against breakage);
- during the rounds a check is never obeyed for long: one that keeps failing for STALE_ROUNDS
  rounds in which everything else passes is dropped as possibly wrong, and one whose first pass
  came with the project's own checks breaking (they passed the round before) is dropped as
  contradicting them. Failing checks rank after the project's tests, like spec tests.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import debuglog, features, sandbox
from .tools import _project_files

if TYPE_CHECKING:
    from .agent import CodingAgent

STALE_ROUNDS = 2  # rounds a check may fail while everything else passes before it is dropped
MAX_MODEL_CHECKS = 5
MODEL_MAX_TOKENS = 1200
TESTS_CHARS = 4000  # visible test text shown to the check writer
DETAIL_CHARS = 300
CODE_CHARS = 700  # a model check's code shown in the repair text
CHECK_SECONDS = 10  # one check, inside the sandbox run

_BACKTICK = re.compile(r"`([^`\n]{1,200})`")
_PROG = re.compile(r"ArgumentParser\(\s*(?:[^()]*?,\s*)?prog\s*=\s*[\"']([\w.-]+)[\"']")
# Placeholders in a written command: <x>, ..., …, [optional parts], DIR / FILE / PATH words.
_PLACEHOLDER = re.compile(r"^(?:<[^>]*>|\.\.\.|…|[A-Z][A-Z_]{2,}|\[.*\])$")
_ENV_EXAMPLE = re.compile(
    r"(?<![\w$])([A-Z][A-Z0-9]*_[A-Z0-9_]+)=([^\s,;`'\"]+)[,;.]?\s+(?:overrides|sets|"
    r"replaces)\s+`([A-Za-z_]\w*)`")
_LOADER_NAME = re.compile(r"(?:^|_)(?:load|read|get|settings|config|conf|options)(?:_|$)")
_COMMANDS = ("cli", "module")  # check kinds that run a command line
_PYTHON = ("python", "python3")
_SHELL_JOIN = {";", "&&", "||", "&"}
_OUT_REDIRECT = {">", ">>", "2>", "2>>", "&>"}
_NOT_PROJECT = {"pytest", "unittest", "mypy", "pip", "venv", "doctest", "timeit", "http.server"}
# A placeholder anywhere in a written command line (acceptance's rule for commands it runs).
_LINE_PLACEHOLDER = re.compile(r"<[^>]*>|\.\.\.|…|\[[^\]]*\]|(?<![\w-])[A-Z][A-Z_]{2,}(?![\w])")
_DOC_COMMAND = re.compile(r"(?:^|`|\$\s)\s*(python3? -m [\w.]+(?:[ \t]+[^`\n]+)?)")
_DOC_SUFFIXES = (".md", ".rst", ".txt")
SAMPLE_CHARS = 2000  # stdin fed to a command


@dataclass
class Check:
    id: str
    kind: str  # cli | module | env | code
    claim: str  # the task's words it checks
    spec: dict[str, Any]  # the runner's input
    baseline: dict[str, Any] = field(default_factory=dict)  # its result on the unchanged project
    dropped: str = ""  # why it is no longer used ('' = active)
    green_fails: int = 0  # consecutive rounds failing while everything else passed
    failed_last: bool = False

    def label(self) -> str:
        return f"`{self.claim}`" if self.kind in _COMMANDS else f'"{self.claim}"'


# -- rules ---------------------------------------------------------------------------------------

def _source_files(project: Path) -> list[Path]:
    return [p for p in _project_files(project) if p.suffix == ".py"
            and not any(part in {"tests", "test", "hidden_tests"} for part in
                        p.relative_to(project).parts[:-1])
            and not p.name.startswith("test_")]


def _module(project: Path, path: Path) -> str:
    parts = list(path.relative_to(project).with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def cli_programs(project: Path) -> dict[str, str]:
    """argparse `prog=` name -> the module whose `main` runs it (the first one found)."""
    programs: dict[str, str] = {}
    for path in _source_files(project):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "def main" not in text:
            continue
        for prog in _PROG.findall(text):
            programs.setdefault(prog, _module(project, path))
    return programs


def cli_checks(project: Path, task: str) -> list[Check]:
    """The task's backticked commands whose first word is a prog of the project."""
    programs = cli_programs(project)
    checks: list[Check] = []
    seen: set[str] = set()
    for raw in _BACKTICK.findall(task):
        command = raw.strip()
        try:
            words = shlex.split(command)
        except ValueError:
            continue
        if len(words) < 2 or words[0] not in programs or command in seen:
            continue
        seen.add(command)
        argv, fixed, placeholders = [], [], False
        for word in words[1:]:
            if _PLACEHOLDER.match(word):
                placeholders = True
                if not word.startswith("["):  # an optional part is left out
                    argv.append("x")
            else:
                argv.append(word)
                fixed.append(word)
        checks.append(Check(f"cli-{len(checks) + 1}", "cli", command,
                            {"module": programs[words[0]], "prog": words[0], "argv": argv,
                             "fixed": fixed, "placeholders": placeholders}))
    return checks


def _segments(command: str) -> list[dict[str, Any]] | None:
    """A shell command line as its pipeline segments: each one's words, the file it reads stdin
    from (`< stock.csv`) and whether it reads a pipe. Output redirects are dropped (the output is
    captured anyway). None for `;`, `&&` and the like: several commands, not one."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    segments: list[dict[str, Any]] = [{"words": [], "stdin_file": "", "piped": False}]
    pending = ""
    for token in tokens:
        if token in _SHELL_JOIN:
            return None
        if pending:
            if pending == "<":
                segments[-1]["stdin_file"] = token
            pending = ""
        elif token == "|":
            segments.append({"words": [], "stdin_file": "", "piped": True})
        elif token == "<" or token in _OUT_REDIRECT:
            pending = token
        else:
            segments[-1]["words"].append(token)
    return [s for s in segments if s["words"]]


def _module_target(project: Path, words: list[str]) -> str:
    """The module of this project that `python -m <module> ...` runs ('' for anything else)."""
    if len(words) < 3 or words[0] not in _PYTHON or words[1] != "-m":
        return ""
    module = words[2]
    if module in _NOT_PROJECT or not re.fullmatch(r"[A-Za-z_][\w.]*", module):
        return ""
    return module if _project_module_exists(project, module) else ""


def doc_files(project: Path, task: str) -> list[Path]:
    """The docs the task points to (`pkg/README.md`), then README files at the top level
    and one folder down."""
    found: list[Path] = []
    for raw in re.findall(r"[\w./-]+\.(?:md|rst|txt)\b", task):
        path = project / raw
        if path.is_file() and path not in found:
            found.append(path)
    folders = [project, *sorted(p for p in project.iterdir()
                                if p.is_dir() and not p.name.startswith("."))]
    for folder in folders:
        for path in sorted(folder.glob("README*")):
            if path.suffix in ("", *_DOC_SUFFIXES) and path.is_file() and path not in found:
                found.append(path)
    return found


def full_invocations(project: Path, task: str, module: str) -> list[str]:
    """`python -m <module> ARGS...` written out in full by the task, then by the docs it points
    to. A bare `python -m pkg` names the program; pkg/README.md shows how it is run:
    `python -m pkg "input" --after 2024-05-03T09:00 --count 3`. Lines with
    placeholders (`<file>`, `FILE`, `[--count N]`, `...`) are left out."""
    texts = [task]
    for path in doc_files(project, task):
        try:
            texts.append(path.read_text(encoding="utf-8")[:20000])
        except (OSError, UnicodeDecodeError):
            continue
    found: list[str] = []
    for text in texts:
        for line in text.splitlines():
            for match in _DOC_COMMAND.finditer(line):
                command = " ".join(match.group(1).split())
                if _LINE_PLACEHOLDER.search(command) or command in found:
                    continue
                try:
                    words = shlex.split(command)
                except ValueError:
                    continue
                alternatives = any("|" in w and w != "|" for w in words)  # train|evaluate
                if len(words) >= 4 and words[2] == module and not alternatives:
                    found.append(command)
    return found


def _string_constants(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return []
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
            and isinstance(node.value, str) and "\n" in node.value.strip()]


def _fits(text: str, kind: str) -> bool:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if kind in ("csv", "tsv"):
        sep = "," if kind == "csv" else "\t"
        return len(lines) >= 2 and all(sep in line for line in lines[:3])
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def sample_input(project: Path, task: str, stdin_file: str) -> str:
    """A small stdin for a command: the redirected file when the project has it, else a sample of
    the same format (the file's suffix, or the task's own word for it: CSV, TSV, JSON) from the
    visible tests (a test that names the file first), then from fenced blocks of the docs. ''
    (an empty stdin) when there is none."""
    if stdin_file and (project / stdin_file).is_file():
        try:
            return (project / stdin_file).read_text(encoding="utf-8")[:SAMPLE_CHARS]
        except (OSError, UnicodeDecodeError):
            return ""
    kind = Path(stdin_file).suffix.lstrip(".").lower()
    if not kind:
        word = re.search(r"\b(csv|tsv|json)\b", task, re.IGNORECASE)
        kind = word.group(1).lower() if word else ""
    if kind not in ("csv", "tsv", "json"):
        return ""
    name = Path(stdin_file).name

    def names_file(path: Path) -> bool:
        try:
            return bool(name) and name in path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False

    tests = sorted((p for p in _project_files(project) if p.suffix == ".py"
                    and (p.name.startswith("test_") or p.name == "conftest.py")),
                   key=lambda p: not names_file(p))
    for path in tests:
        for text in _string_constants(path):
            if _fits(text, kind):
                return text[:SAMPLE_CHARS]
    for path in doc_files(project, task):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for block in re.findall(r"```[\w-]*\n(.*?)```", text, re.DOTALL):
            if _fits(block, kind):
                return block[:SAMPLE_CHARS]
    return ""


def _acceptance_commands(project: Path, task: str) -> set[str]:
    """The commands acceptance runs for this task: a plain `python -m pkg args` it already
    checks (by exit code) is not run twice."""
    from . import acceptance  # acceptance imports this module

    try:
        return set(acceptance.extract(project, task).commands)
    except Exception:  # noqa: BLE001 - at worst a command is checked twice
        return set()


def module_checks(project: Path, task: str) -> list[Check]:
    """The task's `python -m pkg ...` commands: every segment of a backticked command line that
    runs a module of the project, with the stdin it reads. For a bare `python -m pkg`, the full
    invocations its docs give that read stdin (the plain ones are acceptance's)."""
    spans: list[str] = []
    for raw in _BACKTICK.findall(task):
        command = " ".join(raw.split())
        spans.append(command)
        parts = _segments(command) or []
        if len(parts) == 1 and len(parts[0]["words"]) == 3 \
                and not parts[0]["stdin_file"]:
            module = _module_target(project, parts[0]["words"])
            if module:
                spans += [c for c in full_invocations(project, task, module)
                          if any(s["stdin_file"] or s["piped"] for s in _segments(c) or [])]
    checks: list[Check] = []
    seen: set[str] = set()
    accepted: set[str] | None = None
    for command in dict.fromkeys(spans):
        segments = _segments(command)
        if not segments:
            continue
        upstream = ""  # the file the pipeline starts from: its format is what flows through
        for segment in segments:
            words = segment["words"]
            upstream = segment["stdin_file"] or upstream
            module = _module_target(project, words)
            if not module:
                continue
            written = words[3:]
            reads = bool(segment["stdin_file"] or segment["piped"] or "-" in written)
            if not written and not reads:
                continue  # a bare name: acceptance checks its full invocation
            claim = " ".join(words)
            claim = (f"{claim} < {segment['stdin_file']}" if segment["stdin_file"] else
                     f"... | {claim}" if segment["piped"] else claim)
            if claim in seen:
                continue
            if not reads and len(segments) == 1:
                if accepted is None:
                    accepted = _acceptance_commands(project, task)
                if command in accepted:
                    continue
            seen.add(claim)
            argv, fixed, placeholders = [], [], False
            for word in written:
                if _PLACEHOLDER.match(word) or "|" in word:  # train|evaluate: alternatives
                    placeholders = True
                    if not word.startswith("["):
                        argv.append("x")
                else:
                    argv.append(word)
                    fixed.append(word)
            stdin = sample_input(project, task, segment["stdin_file"] or upstream) if reads \
                else ""
            checks.append(Check(f"module-{len(checks) + 1}", "module", claim,
                                {"module": module, "prog": module, "argv": argv, "fixed": fixed,
                                 "placeholders": placeholders, "reads": reads, "stdin": stdin}))
    return checks


def _loaders(project: Path) -> list[str]:
    """module:function for module-level functions with no required parameters whose name says
    they load settings (read_settings, load_config, get_settings, settings)."""
    found = []
    for path in _source_files(project):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            if not _LOADER_NAME.search(node.name):
                continue
            args = node.args
            required = len(args.posonlyargs) + len(args.args) - len(args.defaults)
            required += sum(1 for d in args.kw_defaults if d is None)
            if required == 0:
                found.append(f"{_module(project, path)}:{node.name}")
    return found


def env_checks(project: Path, task: str) -> list[Check]:
    """"APP_PORT=9000 overrides `port`": the loader sees only that variable of the prefix."""
    loaders = _loaders(project)
    if not loaders:
        return []
    checks = []
    for name, value, key in _ENV_EXAMPLE.findall(task):
        prefix = name.split("_", 1)[0] + "_"
        checks.append(Check(f"env-{len(checks) + 1}", "env", f"{name}={value} overrides `{key}`",
                            {"var": name, "value": value, "key": key, "prefix": prefix,
                             "loaders": loaders}))
    return checks


# -- the model's checks --------------------------------------------------------------------------

MODEL_SYSTEM = (
    "You turn a programming task's explicit statements into small executable checks. Each check "
    "is a few lines of plain Python (no pytest, no fixtures) that imports from the project the "
    "way its tests do, builds objects the way the visible tests do (copy their helper code into "
    "the check; never import from the tests), calls the code and uses assert to check exactly "
    "one statement of the task, with the task's own values. For a stated error use try/except "
    "and raise AssertionError when it is not raised. Check only what the task states with "
    "concrete values or exact rules (examples, inputs and outputs, error types, ordering); skip "
    "anything you would have to guess. No subprocess, no network, files only under tempfile. "
    f"At most {MAX_MODEL_CHECKS} checks. `claim` copies the task's exact words the check tests.")
MODEL_SCHEMA = {
    "type": "object",
    "properties": {"checks": {"type": "array", "maxItems": MAX_MODEL_CHECKS, "items": {
        "type": "object", "properties": {"claim": {"type": "string"}, "code": {"type": "string"}},
        "required": ["claim", "code"]}}},
    "required": ["checks"],
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def grounded(claim: str, task: str) -> bool:
    """The claim is the task's own words: a substring after normalising, or 80% of its words
    (at least 3) appear in the task."""
    words = _words(claim)
    if len(words) < 2:
        return False
    if " ".join(words) in " ".join(_words(task)):
        return True
    task_words = set(_words(task))
    return len(words) >= 3 and sum(w in task_words for w in words) >= 0.8 * len(words)


def model_checks(reply: str, task: str) -> tuple[list[Check], list[str]]:
    """(checks, why others were dropped) from the model's JSON reply."""
    try:
        items = json.loads(reply).get("checks", [])
    except (ValueError, AttributeError):
        return [], ["the reply was not JSON"]
    checks, dropped = [], []
    for item in items[:MAX_MODEL_CHECKS]:
        if not isinstance(item, dict):
            continue
        claim, code = str(item.get("claim", "")).strip(), str(item.get("code", "")).strip()
        if not grounded(claim, task):
            dropped.append(f"not the task's words: {claim[:80]!r}")
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            dropped.append(f"does not parse: {claim[:80]!r}")
            continue
        if not any(isinstance(n, ast.Assert) or (isinstance(n, ast.Raise) and "AssertionError"
                                                 in ast.unparse(n)) for n in ast.walk(tree)):
            dropped.append(f"asserts nothing: {claim[:80]!r}")
            continue
        if re.search(r"\bsubprocess\b|\bos\.system\b|\bimport pytest\b|from tests?\b", code):
            dropped.append(f"uses subprocess, pytest or the tests: {claim[:80]!r}")
            continue
        checks.append(Check(f"model-{len(checks) + 1}", "code", claim, {"code": code}))
    return checks, dropped


def _visible_tests(project: Path) -> str:
    texts = []
    for path in _project_files(project):
        relative = path.relative_to(project)
        if path.suffix == ".py" and (path.name.startswith("test_") or path.name == "conftest.py"):
            try:
                texts.append(f"# {relative.as_posix()}\n" + path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
    return "\n\n".join(texts)[:TESTS_CHARS]


def ask_model(agent: CodingAgent, task: str, repo_map: str) -> tuple[list[Check], list[str]]:
    """One constrained-JSON call for the model's checks (MODEL_MAX_TOKENS at most)."""
    tests = _visible_tests(agent.project)
    user = (f"Task:\n{task}\n\nProject map:\n{repo_map}"
            + (f"\n\nThe project's visible tests (how to set things up):\n{tests}" if tests else ""))
    sampling = {**agent.sampling, "max_tokens": min(MODEL_MAX_TOKENS,
                                                     agent.sampling.get("max_tokens", 10**6))}
    reply = agent.client.chat.completions.create(
        model=agent.model, messages=[{"role": "system", "content": MODEL_SYSTEM},
                                     {"role": "user", "content": user}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "checks", "schema": MODEL_SCHEMA}},
        **sampling)
    return model_checks(reply.choices[0].message.content or "", task)


# -- running them --------------------------------------------------------------------------------

RUNNER = r'''
import contextlib, importlib, inspect, io, json, os, signal, sys, traceback
sys.path[:0] = ["src", "."]
for name in [n for n in os.environ if n.startswith("VEGA_")]:  # Vega's own, not the project's
    del os.environ[name]
SECONDS = int(sys.argv[2])
class Timeout(Exception):
    pass
def alarm(signum, frame):
    raise Timeout(f"took over {SECONDS}s")
signal.signal(signal.SIGALRM, alarm)
def last(exc):
    return f"{type(exc).__name__}: {exc}"[:600]
def run_cli(spec):
    try:
        main = importlib.import_module(spec["module"]).main
    except Exception as exc:
        return {"exit": 0, "usage": "", "traceback": last(exc)}
    def call():
        takes = bool(inspect.signature(main).parameters)
        return main(list(spec["argv"])) if takes else main()
    return invoke(spec, call)
def run_module(spec):
    import runpy
    def call():
        runpy.run_module(spec["module"], run_name="__main__", alter_sys=True)
        return 0
    return invoke(spec, call)
def stream(text=""):
    return io.TextIOWrapper(io.BytesIO(text.encode("utf-8")), encoding="utf-8",
                            write_through=True)
def value(handle):
    try:
        handle.flush()
    except ValueError:  # closed by the program
        pass
    return handle.buffer.getvalue().decode("utf-8", "replace")
def invoke(spec, call):
    out, err = stream(), stream()
    result = {"exit": 0, "usage": "", "traceback": ""}
    saved = sys.argv, sys.stdin
    sys.argv, sys.stdin = [spec["prog"], *spec["argv"]], stream(spec.get("stdin", ""))
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = call()
        result["exit"] = code if isinstance(code, int) else 0
    except SystemExit as exc:
        result["exit"] = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        text = value(err)
        if result["exit"] == 2 and "usage:" in (text + value(out)):
            lines = ([l for l in text.splitlines() if "error:" in l]
                     or text.strip().splitlines()[-1:] or [""])
            result["usage"] = lines[-1]
    except Timeout as exc:
        result["traceback"] = last(exc)
    except Exception as exc:
        tb = traceback.extract_tb(exc.__traceback__)
        where = f" (in {tb[-1].filename.rsplit('/', 1)[-1]}:{tb[-1].lineno})" if tb else ""
        result["traceback"] = last(exc) + where
    finally:
        sys.argv, sys.stdin = saved
    result["output"] = (value(out) + value(err))[-400:]
    return result
def convert(raw, default):
    if isinstance(default, bool):
        low = raw.strip().lower()
        return {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}[low]
    if isinstance(default, (int, float)):
        return type(default)(raw)
    return raw
def lookup(value, key):
    if isinstance(value, dict):
        return value[key] if key in value else KeyError
    return getattr(value, key, KeyError)
def run_env(spec):
    saved = dict(os.environ)
    try:
        for name in list(os.environ):
            if name.startswith(spec["prefix"]):
                del os.environ[name]
        raised = ""
        for dotted in spec["loaders"]:
            module, _, name = dotted.partition(":")
            try:
                loader = getattr(importlib.import_module(module), name)
                before = lookup(loader(), spec["key"])
            except Exception as exc:
                raised = raised or f"{name}() raised {last(exc)}"
                continue
            if before is KeyError:
                continue
            os.environ[spec["var"]] = spec["value"]
            call = f"{name}() with {spec['var']}={spec['value']}"
            try:
                after = lookup(loader(), spec["key"])
            except Exception as exc:
                return {"status": "fail", "detail": f"{call} raised {last(exc)}"}
            finally:
                del os.environ[spec["var"]]
            try:
                expected = convert(spec["value"], before)
            except (KeyError, ValueError):
                expected = spec["value"]
            if after == expected and type(after) is type(expected):
                return {"status": "pass"}
            return {"status": "fail", "detail": f"{call} gives {spec['key']}={after!r}; the task "
                    f"says it overrides {spec['key']} (expected {expected!r})"}
        if raised:
            return {"status": "fail", "detail": raised}
        return {"status": "skip", "detail": "no settings loader returns " + spec["key"]}
    finally:
        os.environ.clear()
        os.environ.update(saved)
def run_code(spec):
    saved, cwd = dict(os.environ), os.getcwd()
    try:
        exec(compile(spec["code"], "<task check>", "exec"), {"__name__": "__task_check__"})
        return {"status": "pass"}
    except AssertionError as exc:
        tb = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == "<task check>"]
        line = spec["code"].splitlines()[tb[-1].lineno - 1].strip() if tb else ""
        return {"status": "fail", "detail": (f"{line}: " if line else "") + (str(exc) or
                                                                              "assertion failed")}
    except (ImportError, AttributeError, NameError) as exc:
        return {"status": "missing", "detail": last(exc)}
    except BaseException as exc:
        return {"status": "crash", "detail": last(exc)}
    finally:
        os.environ.clear()
        os.environ.update(saved)
        os.chdir(cwd)
results = {}
for check in json.loads(sys.argv[1]):
    signal.alarm(SECONDS)
    try:
        if check["kind"] == "cli":
            results[check["id"]] = run_cli(check["spec"])
        elif check["kind"] == "module":
            results[check["id"]] = run_module(check["spec"])
        elif check["kind"] == "env":
            results[check["id"]] = run_env(check["spec"])
        else:
            results[check["id"]] = run_code(check["spec"])
    except BaseException as exc:
        results[check["id"]] = {"status": "crash", "detail": last(exc)}
    finally:
        signal.alarm(0)
print("\n" + json.dumps(results))
'''


def run_all(project: Path, checks: list[Check],
            runner: Callable[[str, Path], str] | None = None) -> dict[str, dict[str, Any]]:
    """Each check's raw result, from one sandboxed run (an empty dict when the run failed)."""
    if not checks:
        return {}
    payload = json.dumps([{"id": c.id, "kind": c.kind, "spec": c.spec} for c in checks])
    command = (f"python -c {shlex.quote(RUNNER)} {shlex.quote(payload)} "
               f"{CHECK_SECONDS}")
    output = (runner or sandbox.run_check)(command, project)
    lines = [line for line in output.splitlines() if line.strip()]
    try:
        results = json.loads(lines[-1])
    except (IndexError, ValueError):
        return {}
    return results if isinstance(results, dict) else {}


def _rejected(usage: str, fixed: list[str]) -> list[str]:
    """Words of the command as written that argparse rejected."""
    words = re.findall(r"invalid choice: '([^']*)'", usage)
    match = re.search(r"unrecognized arguments: (.*)$", usage)
    if match:
        words += match.group(1).split()
    return [w for w in words if w in fixed]


# A command that needs a file or a service the sandbox does not have is not wrong code.
_ENVIRONMENTAL = ("FileNotFoundError", "PermissionError", "IsADirectoryError",
                  "NotADirectoryError", "ConnectionError", "ConnectionRefusedError", "OSError",
                  "TimeoutError", "Timeout")


def verdict(check: Check, result: dict[str, Any]) -> str:
    """'' when the check passes, else what failed (one line)."""
    if not result:
        return "did not run"
    if check.kind not in _COMMANDS:
        if result.get("status") == "pass":
            return ""
        return str(result.get("detail") or result.get("status"))[:DETAIL_CHARS]
    spec, base = check.spec, check.baseline
    usage, trace = result.get("usage", ""), result.get("traceback", "")
    rejected = _rejected(usage, spec["fixed"])
    if rejected:
        return f"argparse rejects {', '.join(rejected)}: {usage}"[:DETAIL_CHARS]
    if spec["placeholders"]:  # made-up arguments: only the written words are checked
        return ""
    if usage and usage != base.get("usage"):
        return f"usage error: {usage}"[:DETAIL_CHARS]
    if usage and spec.get("reads") and "arguments are required" in usage:
        # the task's command reads stdin; it still demands the argument stdin replaces
        return f"usage error with input on stdin: {usage}"[:DETAIL_CHARS]
    if trace and trace != base.get("traceback") and not trace.startswith(_ENVIRONMENTAL):
        if spec.get("reads") and not spec.get("stdin") and trace.startswith(_NO_DATA):
            return ""  # no sample input was found: an error about the empty input is fair
        stdin = " (stdin: a sample of the project's data)" if spec.get("stdin") else ""
        return f"ends in a traceback{stdin}: {trace}"[:DETAIL_CHARS]
    return ""


# What a program may fairly raise when it is fed an empty stdin.
_NO_DATA = ("ValueError", "JSONDecodeError", "StopIteration", "IndexError", "KeyError",
            "EOFError")


def keep_at_baseline(check: Check, result: dict[str, Any], task: str) -> str:
    """'' to keep the check, else why it is dropped before any write."""
    if not result:
        return "could not be run"
    if check.kind == "cli":
        if result.get("traceback", "").startswith("ModuleNotFoundError"):
            return "its module does not import"
        return ""
    if check.kind == "module":
        # A missing `pkg/__main__.py` is behaviour the task asks for; a missing dependency is not.
        missing = re.match(r"(?:ModuleNotFound|Import)Error: No module named '?([\w.]+)",
                           result.get("traceback", ""))
        if missing and missing.group(1) != f"{check.spec['module']}.__main__":
            return "its module does not import"
        return ""
    status = result.get("status")
    if status == "pass":
        return "already passes before the change"
    if status == "skip":
        return str(result.get("detail", "could not be run"))
    if status == "fail":
        return ""
    if status == "missing" and check.kind == "code":
        detail = str(result.get("detail", ""))
        if missing_by_the_task(detail, task):
            return ""
        return f"fails for a reason the task does not name: {detail[:120]}"
    return f"fails for a reason of its own: {str(result.get('detail', ''))[:120]}"


def _mentions_module(module: str, task: str) -> bool:
    return module in task or module.replace(".", "/") + ".py" in task


def missing_by_the_task(detail: str, task: str) -> bool:
    """A check that fails on the unchanged project because a name is missing says something only
    when the task puts that name there: `app.graph` for CycleError, `app/money.py` for the
    module. `from app import Money` (the task never says the package re-exports it) or a
    module the task never names (`app.task`) is the check's own mistake."""
    match = (re.search(r"cannot import name '(\w+)' from '([\w.]+)'", detail)
             or re.search(r"module '([\w.]+)' has no attribute '(\w+)'", detail))
    if match:
        name, module = (match.groups() if detail.count("cannot import") else
                        (match.group(2), match.group(1)))
        return f"{module}.{name}" in task or ("." in module and _mentions_module(module, task))
    match = re.search(r"No module named '([\w.]+)'", detail)
    if match:
        return "." in match.group(1) and _mentions_module(match.group(1), task)
    match = re.search(r"object has no attribute '(\w+)'", detail)
    if match:
        spans = " ".join(_BACKTICK.findall(task))
        return match.group(1) in re.findall(r"\w+", spans)
    return False


@dataclass
class TaskChecks:
    """The checks of one pipeline run: gated before any write, run after each round."""

    task: str
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # what was derived and dropped, for the trace
    previous_project_failing: int | None = None
    runner: Callable[[str, Path], str] | None = None

    @property
    def active(self) -> list[Check]:
        return [c for c in self.checks if not c.dropped]

    def baseline(self, project: Path) -> None:
        results = run_all(project, self.checks, self.runner)
        for check in self.checks:
            result = results.get(check.id, {})
            check.baseline = result
            check.dropped = keep_at_baseline(check, result, self.task)
            if not check.dropped and verdict(check, result):
                check.failed_last = True
        kept = [c.label() for c in self.active]
        dropped = [f"{c.label()} ({c.dropped})" for c in self.checks if c.dropped]
        self.notes.append("task checks: " + (", ".join(kept) if kept else "none kept")
                          + (f"; dropped {', '.join(dropped)}" if dropped else ""))
        debuglog.record("stage", stage="task_checks.baseline",
                        kept=[c.id for c in self.active],
                        dropped={c.id: c.dropped for c in self.checks if c.dropped},
                        checks={c.id: {"kind": c.kind, "claim": c.claim} for c in self.checks})

    def evaluate(self, project: Path, project_failing: int) -> list[str]:
        """Run the active checks after a round; `project_failing` is everything else that failed
        this round (the project's tests, the task's names, acceptance, refusals). Returns one
        line per failing check, after dropping the checks the churn rules drop."""
        active = self.active
        results = run_all(project, active, self.runner)
        lines: list[str] = []
        before_green = self.previous_project_failing == 0
        for check in active:
            problem = verdict(check, results.get(check.id, {}))
            if problem and project_failing == 0:
                check.green_fails += 1
            elif not problem:
                check.green_fails = 0
            if not problem and check.failed_last and before_green and project_failing > 0:
                check.dropped = ("its first pass broke the project's own checks, which passed "
                                 "the round before")
            elif problem and check.green_fails >= STALE_ROUNDS:
                check.dropped = (f"still failing after {STALE_ROUNDS} rounds in which everything "
                                 "else passed (it may be wrong)")
                lines.append(self._line(check, problem) + " (no longer checked)")
            elif problem:
                lines.append(self._line(check, problem))
            check.failed_last = bool(problem)
            if check.dropped:
                self.notes.append(f"task check dropped: {check.label()}: {check.dropped}")
        self.previous_project_failing = project_failing
        debuglog.record("stage", stage="task_checks.round", failing=len(
            [line for line in lines if not line.endswith("(no longer checked)")]),
            dropped=[c.id for c in active if c.dropped])
        return lines

    def _line(self, check: Check, problem: str) -> str:
        if check.kind == "cli":
            return f"`{check.claim}`: {problem}"
        if check.kind == "module":
            stdin = check.spec.get("stdin", "")
            shown = " / ".join(stdin.strip().splitlines()[:3])
            return f"`{check.claim}`: {problem}" + (f" (stdin was: {shown} ...)" if stdin else "")
        if check.kind == "env":
            return f"{check.claim}: {problem}"
        code = check.spec["code"]
        code = code if len(code) <= CODE_CHARS else code[:CODE_CHARS] + "\n..."
        return f'"{check.claim}": {problem}\n  check:\n' + "\n".join(
            "    " + line for line in code.splitlines())


def _project_module_exists(project: Path, module: str) -> bool:
    parts = module.split(".")
    for root in (project, project / "src"):
        base = root.joinpath(*parts)
        if base.with_suffix(".py").is_file() or (base / "__init__.py").is_file():
            return True
    return False


def unknown_imports(code: str, project: Path, task: str) -> list[str]:
    """Project modules a check imports that neither exist nor are named by the task
    (`from app.task import Task` when the project has app/config.py): the check's own
    mistake, which an assertion further down could never get past."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    packages = {p.name for root in (project, project / "src") if root.is_dir()
                for p in root.iterdir() if (p / "__init__.py").is_file()}
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
    return [m for m in dict.fromkeys(modules) if m.split(".")[0] in packages
            and not _project_module_exists(project, m) and not _mentions_module(m, task)]


def prepare(project: Path, task: str, *, rules: bool, model: list[Check] | None = None,
            model_dropped: list[str] | None = None,
            runner: Callable[[str, Path], str] | None = None) -> TaskChecks:
    """Derive the checks and gate them on the unchanged project."""
    checks: list[Check] = []
    if rules:
        checks += cli_checks(project, task) + env_checks(project, task)
        if features.on("task_module_commands"):
            checks += module_checks(project, task)
    model_dropped = list(model_dropped or [])
    for check in model or []:
        unknown = unknown_imports(check.spec["code"], project, task)
        if unknown:
            model_dropped.append(f"imports {', '.join(unknown)}, which the project does not "
                                 f"have and the task does not name: {check.claim[:80]!r}")
        else:
            checks.append(check)
    result = TaskChecks(task, checks, runner=runner)
    if model_dropped:
        result.notes.append("model checks dropped before running: " + "; ".join(model_dropped))
    if checks:
        result.baseline(project)
    return result
