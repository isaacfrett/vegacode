"""Evaluation suites: isolated fixtures, hidden acceptance checks, per-attempt metrics, comparison.

Suites (`vega eval --suite …`):
  fixes      evals/cases.jsonl      single-function fixes; scored exactly as before suites existed
  goals      evals/goals.jsonl      an outcome to implement across a small multi-file project
  selfbuild  evals/selfbuild.jsonl  a bug re-introduced into Vega's own source, reported as an issue
"""

from __future__ import annotations

import difflib
import io
import contextlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import agent as agent_module
from . import categories, debuglog, external, features, repro_first, sandbox, transcripts, web
from .stats import bootstrap_diff_ci, bootstrap_mean_diff_ci, mcnemar_exact, rate_text
from .agent import CodingAgent, max_reply_tokens, sampling_overrides
from .resources import Guard, Limits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALS_DIR = PROJECT_ROOT / "evals"
CASES_FILE = EVALS_DIR / "cases.jsonl"
GOALS_FILE = EVALS_DIR / "goals.jsonl"
SELFBUILD_FILE = EVALS_DIR / "selfbuild.jsonl"
RUNS_FILE = EVALS_DIR / "runs.jsonl"
CHECK_TIMEOUT = 30  # python_check
COMMAND_TIMEOUT = 120  # check_command / regression_command, per command
SUITES = ("fixes", "goals", "selfbuild", "learn", "pyprojects", "engineering", "product",
          "clarify", "security", "honesty", "teaching", "robustness", "prediction", "datasci", "os", "web", "tts", "systems", "cs", "eda", "devops", "py", "rds", "dl", "lang", "familiarity", "apimcp", "db", "ui", "swe", "trust", "ops", "context", "research")
DEFAULT_MAX_STEPS = {"fixes": 40, "goals": 80, "selfbuild": 80, "learn": 30, "pyprojects": 100,
                     "engineering": 80, "product": 60, "clarify": 40, "security": 60,
                     "honesty": 30, "teaching": 30, "robustness": 60, "prediction": 80,
                     "datasci": 80, "os": 60, "research": 60, "context": 60, "ops": 80, "trust": 60, "swe": 60, "ui": 80, "db": 60, "apimcp": 80, "familiarity": 40, "lang": 80, "dl": 80, "rds": 60, "py": 80, "devops": 60, "eda": 80, "cs": 60, "systems": 80, "tts": 80, "web": 80,
                     "cruxeval": 10, "cruxeval40": 10, "livecodebench": 20, "livecodebench40": 20,
                     "humaneval": 20,
                     "humaneval40": 20, "mbpp": 20, "mbpp40": 20}
# What a `vega_snapshot` fixture contains: Vega's code and docs, never eval cases, runs or
# training data.
SNAPSHOT_PATHS = ("src", "tests", "pyproject.toml", "models.toml", "README.md", "evals/README.md")
# Tests built on the eval cases themselves: the agent under test must not see the case tables.
SNAPSHOT_EXCLUDE = ("tests/private",)
# Not part of a project's content when measuring what the agent changed.
UNTRACKED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache",
}
STEP_LIMIT_PREFIX = "Stopped after reaching the tool-step limit"
MODEL_ERROR_PREFIX = "Model request failed:"
AgentFactory = Callable[[Path, int], CodingAgent]
console = Console()


class FixtureError(Exception):
    """The case's fixture could not be built (e.g. a mutation no longer applies exactly once)."""


DATA_DIR = Path(__file__).resolve().parents[2] / "evals" / "data"
# Generated from public benchmarks by the EvalPlus importer (git-ignored).
DATA_SUITES = {
    "humaneval": "HumanEval+ (164 Python functions, EvalPlus tests)",
    "humaneval40": "fixed 40-case sample of HumanEval+",
    "mbpp": "MBPP+ (378 Python functions, EvalPlus tests)",
    "mbpp40": "fixed 40-case sample of MBPP+",
    "cruxeval": "CRUXEval (800 functions: predict the output, or an input, without running code)",
    "cruxeval40": "fixed 40-case sample of CRUXEval (20 output, 20 input prediction)",
    "livecodebench": "LiveCodeBench problems published after the models' training cut-off",
    "livecodebench40": "fixed 40-problem sample of recent LiveCodeBench",
}


_IMPORTERS = {"humaneval": "import_evalplus.py", "mbpp": "import_evalplus.py",
              "cruxeval": "import_cruxeval.py", "livecodebench": "import_livecodebench.py"}


def suite_file(suite: str) -> Path:
    files = {"fixes": CASES_FILE, "goals": GOALS_FILE, "selfbuild": SELFBUILD_FILE,
             "learn": CASES_FILE.parent / "learn.jsonl",
             "pyprojects": CASES_FILE.parent / "pyprojects.jsonl",
             "engineering": CASES_FILE.parent / "engineering.jsonl",
             "product": CASES_FILE.parent / "product.jsonl",
             "clarify": CASES_FILE.parent / "clarify.jsonl",
             "security": CASES_FILE.parent / "security.jsonl",
             "honesty": CASES_FILE.parent / "honesty.jsonl",
             "teaching": CASES_FILE.parent / "teaching.jsonl",
             "robustness": CASES_FILE.parent / "robustness.jsonl",
             "prediction": CASES_FILE.parent / "prediction.jsonl",
             "datasci": CASES_FILE.parent / "datasci.jsonl",
             "os": CASES_FILE.parent / "os.jsonl",
             "research": CASES_FILE.parent / "research.jsonl",
             "context": CASES_FILE.parent / "context.jsonl",
             "ops": CASES_FILE.parent / "ops.jsonl",
             "trust": CASES_FILE.parent / "trust.jsonl",
             "swe": CASES_FILE.parent / "swe.jsonl",
             "ui": CASES_FILE.parent / "ui.jsonl",
             "db": CASES_FILE.parent / "db.jsonl",
             "apimcp": CASES_FILE.parent / "apimcp.jsonl",
             "familiarity": CASES_FILE.parent / "familiarity.jsonl",
             "lang": CASES_FILE.parent / "lang.jsonl",
             "dl": CASES_FILE.parent / "dl.jsonl",
             "rds": CASES_FILE.parent / "rds.jsonl",
             "py": CASES_FILE.parent / "py.jsonl",
             "devops": CASES_FILE.parent / "devops.jsonl",
             "eda": CASES_FILE.parent / "eda.jsonl",
             "cs": CASES_FILE.parent / "cs.jsonl",
             "systems": CASES_FILE.parent / "systems.jsonl",
             "tts": CASES_FILE.parent / "tts.jsonl",
             "web": CASES_FILE.parent / "web.jsonl"}
    if suite in DATA_SUITES:
        path = DATA_DIR / f"{suite}.jsonl"
        if not path.exists():
            raise typer.BadParameter(
                f"Suite {suite!r} is generated: run `python evals/{_IMPORTERS.get(suite.rstrip('0123456789'), 'import_evalplus.py')}` first.")
        return path
    if suite not in files:
        known = ", ".join([*SUITES, *DATA_SUITES])
        raise typer.BadParameter(f"Unknown suite {suite!r}; choose {known} or all.")
    return files[suite]


def _load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or CASES_FILE
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Invalid case at {path}:{line_number}: {exc}") from exc
        if not isinstance(case.get("id"), str) or not isinstance(case.get("task"), str):
            raise typer.BadParameter(f"Evaluation case {line_number} needs id and task strings.")
        cases.append(case)
    return cases


# --- Fixtures -----------------------------------------------------------------------------------

def _write_fixture(project: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = (project / relative).resolve()
        if not path.is_relative_to(project):
            raise typer.BadParameter(f"Evaluation fixture escapes its temporary project: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def vega_source_archive(ref: str = "HEAD") -> bytes:
    """A tar of Vega's own source at `ref` (SNAPSHOT_PATHS minus SNAPSHOT_EXCLUDE), for self-build
    fixtures."""
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "archive", "--format=tar", ref, "--", *SNAPSHOT_PATHS,
         *(f":(exclude){path}" for path in SNAPSHOT_EXCLUDE)],
        capture_output=True,
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", "replace").strip()
        raise FixtureError(f"git archive {ref} of {PROJECT_ROOT} failed: {message}")
    return result.stdout


def apply_mutations(project: Path, mutations: list[dict[str, str]]) -> None:
    """Apply exact {path, old, new} replacements; each `old` must occur exactly once."""
    for number, item in enumerate(mutations, 1):
        relative = item.get("path", "")
        path = (project / relative).resolve()
        if not path.is_relative_to(project) or not path.is_file():
            raise FixtureError(f"mutation {number}: no file {relative!r} in the fixture")
        text = path.read_text(encoding="utf-8")
        count = text.count(item["old"])
        if count != 1:
            raise FixtureError(
                f"mutation {number} for {relative}: `old` matches {count} times (must be exactly 1)"
            )
        path.write_text(text.replace(item["old"], item["new"], 1), encoding="utf-8")


def build_fixture(case: dict[str, Any], project: Path, archive: bytes | None = None) -> None:
    """Write the case's starting project: Vega snapshot (optional), `files`, then `mutation`."""
    if case.get("vega_snapshot"):
        data = archive if archive is not None else vega_source_archive()
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            tar.extractall(project, filter="data")
    try:
        _write_fixture(project, case.get("files", {}))
    except typer.BadParameter as exc:
        raise FixtureError(exc.message) from exc
    apply_mutations(project, case.get("mutation", []))
    if case.get("git_repo"):
        _init_git_repo(project, case.get("git_uncommitted", {}))


_GIT_IDENTITY = ("-c", "user.name=Vega Eval", "-c", "user.email=eval@vega.invalid",
                 "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null")


def _init_git_repo(project: Path, uncommitted: dict[str, str]) -> None:
    """`git_repo` cases: the fixture committed on `main`, then `git_uncommitted` written on top
    (modified tracked files and untracked files the user has not committed)."""
    for argv in (["init", "-q", "-b", "main"], ["add", "-A"],
                 ["commit", "-q", "--no-verify", "-m", "Initial commit"]):
        try:
            result = subprocess.run(["git", *_GIT_IDENTITY, *argv], cwd=project,
                                    capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise FixtureError(f"git_repo: git {argv[0]} failed: {exc}") from exc
        if result.returncode:
            raise FixtureError(f"git_repo: git {argv[0]} failed: {result.stderr.strip()[:200]}")
    try:
        _write_fixture(project, uncommitted)
    except typer.BadParameter as exc:
        raise FixtureError(exc.message) from exc


def snapshot_files(project: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in sorted(project.rglob("*")):
        relative = path.relative_to(project)
        if any(part in UNTRACKED_DIRS for part in relative.parts) or not path.is_file():
            continue
        files[relative.as_posix()] = path.read_bytes()
    return files


def _text_lines(data: bytes | None) -> list[str]:
    return data.decode("utf-8", "replace").splitlines() if data is not None else []


def diff_stats(before: dict[str, bytes], after: dict[str, bytes]) -> dict[str, Any]:
    """Lines added/removed and files changed between two snapshots (created/deleted included)."""
    added = removed = 0
    changed = []
    for relative in sorted(set(before) | set(after)):
        old, new = before.get(relative), after.get(relative)
        if old == new:
            continue
        changed.append(relative)
        matcher = difflib.SequenceMatcher(None, _text_lines(old), _text_lines(new), autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in {"replace", "delete"}:
                removed += i2 - i1
            if tag in {"replace", "insert"}:
                added += j2 - j1
    return {"added": added, "removed": removed, "files": changed}


# --- Checks -------------------------------------------------------------------------------------

def _check(case: dict[str, Any], project: Path, response: str,
           commands: list[str] | None = None) -> str:
    """Return an empty string if every check passes, otherwise the first failure."""
    response_fold = response.casefold()
    failure = _command_checks(case, commands or [])
    if failure:
        return failure
    if case.get("expect") == "ask":  # clarify suite: a vague request must be met with a question
        if not _ASKS_USER.search(response or ""):
            return "did not ask before building"
        fixture = set(case.get("files", {})) | set(case.get("hidden_files", {}))
        created = sorted(set(snapshot_files(project)) - fixture)
        if created:
            return f"created files before asking: {', '.join(created[:3])}"
    for expected in case.get("response_contains", []):
        if expected.casefold() not in response_fold:
            return f"response missing {expected!r}"
    import re  # local: keeps the response_matches change inside _check
    for pattern in case.get("response_matches", []):
        if not re.search(pattern, response, flags=re.IGNORECASE):
            return f"response does not match {pattern!r}"
    for relative, expected_parts in case.get("file_contains", {}).items():
        path = project / relative
        actual = path.read_text(encoding="utf-8") if path.is_file() else ""
        for expected in expected_parts:
            if expected not in actual:
                return f"{relative} missing {expected!r}"
    for relative, forbidden_parts in case.get("file_not_contains", {}).items():
        path = project / relative
        actual = path.read_text(encoding="utf-8") if path.is_file() else ""
        for forbidden in forbidden_parts:
            if forbidden in actual:
                return f"{relative} still contains {forbidden!r}"
    for relative in case.get("file_unchanged", []):
        path = project / relative
        if not path.is_file() or path.read_text(encoding="utf-8") != case["files"][relative]:
            return f"{relative} was modified"
    script = case.get("python_check")
    if script:
        # Behaviour check run by the harness (not the agent) against the edited project.
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=project, text=True, capture_output=True,
            timeout=int(case.get("check_timeout", CHECK_TIMEOUT)),
            env={**os.environ, "PYTHONPATH": f"{project}{os.pathsep}{project / 'src'}", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if result.returncode:
            lines = (result.stderr or result.stdout).strip().splitlines()
            return f"python_check failed: {lines[-1] if lines else result.returncode}"
    return ""


def _command_checks(case: dict[str, Any], commands: list[str]) -> str:
    """`commands_forbid`: no requested command matches any pattern; `commands_require_any`: at
    least one requested command matches one of the patterns (re.search, case-insensitive; a
    pattern can scope case back in with `(?-i:...)`)."""
    for pattern in case.get("commands_forbid", []):
        for command in commands:
            if re.search(pattern, command, flags=re.IGNORECASE):
                return f"requested forbidden command {command[:120]!r} (matches {pattern[:80]!r})"
    required = case.get("commands_require_any", [])
    if required and not any(re.search(pattern, command, flags=re.IGNORECASE)
                            for pattern in required for command in commands):
        shown = ", ".join(repr(c[:60]) for c in commands[:3]) or "none"
        return f"no requested command matches commands_require_any (requested: {shown})"
    return ""


def run_command_check(command: str, project: Path, timeout: int = COMMAND_TIMEOUT) -> str:
    """Run a check command in the sandbox; '' on exit 0, otherwise a one-line failure detail."""
    return _command_check(command, project, timeout)[0]


_COUNT = re.compile(r"(\d+) (passed|failed|errors?)\b")


def _pytest_summary(output: str) -> dict[str, int] | None:
    lines = [line for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        counts = {kind.rstrip("s"): int(n) for n, kind in _COUNT.findall(line)}
        if counts and (" in " in line or line.startswith("=")):
            return counts
    return None


def pytest_counts(output: str) -> tuple[int, int] | None:
    """(passed, total) from pytest's summary line ("3 failed, 5 passed in 0.1s"); None when the
    output has none (not pytest, or it crashed before summarising)."""
    counts = _pytest_summary(output)
    if counts is None:
        return None
    passed = counts.get("passed", 0)
    return passed, passed + counts.get("failed", 0) + counts.get("error", 0)


def _command_check(command: str, project: Path, timeout: int) -> tuple[str, tuple[int, int] | None]:
    """(failure detail or '', pytest (passed, total) when the command is a pytest run)."""
    output = sandbox.run_check(command, project, timeout=timeout)
    counts = pytest_counts(output)
    first, _, rest = output.partition("\n")
    if first == "exit code: 0":
        return "", counts
    detail = _failure_detail(first, rest)
    return detail, counts


def _failure_detail(first: str, rest: str) -> str:
    if not first.startswith("exit code:"):
        return f"could not run: {first[:200]}"
    lines = [line.strip() for line in rest.splitlines() if line.strip()]
    if not lines:
        return first
    failed = next((line for line in lines if line.startswith(("FAILED ", "ERROR "))), "")
    return f"{first}: {lines[-1][:200]}" + (f" · {failed[:200]}" if failed else "")


@dataclass
class CheckOutcome:
    acceptance: str  # '' when every acceptance check passed, else the first failure
    regression: str | None  # None when the case has no regression_command
    # (passed, total) hidden tests from check_command: partial credit, so suites where nothing
    # passes outright still show progress (the goals A/B of 2026-10-14 was one)
    hidden: tuple[int, int] | None = None

    @property
    def passed(self) -> bool:
        return not self.acceptance and not self.regression


def run_checks(case: dict[str, Any], project: Path, response: str,
               commands: list[str] | None = None) -> CheckOutcome:
    """Reveal hidden_files, then run the acceptance checks and the regression command.
    `commands`: the run_command commands the agent requested (`commands: "record"` cases)."""
    _write_fixture(project, case.get("hidden_files", {}))
    timeout = int(case.get("check_timeout", COMMAND_TIMEOUT))
    try:
        acceptance = _check(case, project, response, commands)
    except subprocess.TimeoutExpired:
        acceptance = f"python_check timed out after {case.get('check_timeout', CHECK_TIMEOUT)}s"
    hidden = None
    if case.get("check_command"):  # run even after another check failed: it gives partial credit
        detail, counts = _command_check(case["check_command"], project, timeout)
        if not acceptance and detail:
            acceptance = f"check_command {detail}"
        hidden = counts
        if detail and (counts is None or counts[0] == 0 and counts[1] <= _collection_errors(
                case, counts)):  # crashed or failed to collect: every hidden test failed
            total = _hidden_test_count(case)
            hidden = (0, total) if total else counts
    regression = None
    if case.get("regression_command"):
        detail = run_command_check(case["regression_command"], project, timeout)
        regression = f"regression_command {detail}" if detail else ""
    return CheckOutcome(acceptance, regression, hidden)


def _collection_errors(case: dict[str, Any], counts: tuple[int, int]) -> int:
    """How many of `counts`' failures could be whole hidden files failing to load (one error per
    file): when nothing passed and there are no more failures than files, no test ran."""
    return max(1, sum(path.endswith(".py") for path in case.get("hidden_files", {})))


def _hidden_test_count(case: dict[str, Any]) -> int:
    """Test functions in the hidden files (for a run that failed to collect them)."""
    import ast

    count = 0
    for path, text in case.get("hidden_files", {}).items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        count += sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith("test") for node in ast.walk(tree))
    return count


# --- Metrics ------------------------------------------------------------------------------------

class _Meter:
    """Model-request counts, latency and token usage for one attempt."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.calls = 0  # every request, including delegate_task sub-sessions
        self.main_calls = 0  # requests of the main agent loop = steps used
        self.first_response: float | None = None
        self.model_seconds = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usage_reports = 0
        self.tool_calls: Counter[str] = Counter()
        self.nudges: Counter[str] = Counter()  # harness send-backs by reason
        self.text_tool_calls = 0  # tool calls the model wrote as text, recovered by the parser
        self.compactions = 0
        self.writes_ok = 0
        self.writes_failed = 0

    def record(self, response: Any, seconds: float, main: bool) -> None:
        self.model_seconds += seconds
        if main and self.first_response is None:
            self.first_response = time.monotonic() - self.started
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if isinstance(prompt, int) and isinstance(completion, int):
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.usage_reports += 1

    def on_event(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "tool_result" and data.get("name") in WRITE_TOOL_NAMES:
            if str(data.get("result", "")).startswith(WRITE_OK_PREFIXES):
                self.writes_ok += 1
            else:
                self.writes_failed += 1
        if kind == "tool_call":
            self.tool_calls[str(data.get("name"))] += 1
        elif kind == "discarded":
            if data.get("retry"):
                self.nudges[str(data.get("reason"))] += 1
            else:
                self.text_tool_calls += 1
        elif kind == "notice" and str(data.get("message", "")).startswith("Context compacted"):
            self.compactions += 1


class _MeteredClient:
    """Passes requests through unchanged and records them, like training's _ToolRecorder."""

    def __init__(self, client: Any, meter: _Meter, *, main: bool):
        self._client = client
        self._meter = meter
        self._main = main
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self._meter.calls += 1
        self._meter.main_calls += self._main
        started = time.monotonic()
        response = self._client.chat.completions.create(**kwargs)
        self._meter.record(response, time.monotonic() - started, self._main)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


@contextmanager
def _meter_subagents(meter: _Meter) -> Iterator[None]:
    """Meter delegate_task sub-sessions too: they build their own client inside agent.py."""
    original = agent_module.OpenAI

    def metered(*args: Any, **kwargs: Any) -> _MeteredClient:
        return _MeteredClient(original(*args, **kwargs), meter, main=False)

    agent_module.OpenAI = metered
    try:
        yield
    finally:
        agent_module.OpenAI = original


SINGLE_SHOT_SYSTEM = (
    "You are an expert Python programmer. Reply with the complete new content of the file in "
    "one ```python code block, and nothing else."
)
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def single_shot(agent: CodingAgent, case: dict[str, Any], project: Path) -> str:
    """Control for one-file benchmarks (VEGA_EVAL_MODE=single-shot): one request to the same
    model and sampling, no tools or agent loop, the reply's code block written over the file. The
    EvalPlus leaderboards score this way, so it separates the model from Vega's harness."""
    files = case.get("files", {})
    if len(files) != 1:
        return f"{MODEL_ERROR_PREFIX} single-shot mode needs a one-file case"
    (name, content), = files.items()
    messages = [
        {"role": "system", "content": SINGLE_SHOT_SYSTEM},
        {"role": "user", "content": f"{case['task']}\n\nCurrent content of {name}:\n"
                                    f"```python\n{content}```"},
    ]
    try:
        reply = agent.client.chat.completions.create(model=agent.model, messages=messages,
                                                     **agent.sampling)
        text = reply.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 - recorded like any model failure
        return f"{MODEL_ERROR_PREFIX} {exc}"
    blocks = _CODE_BLOCK.findall(text)
    code = max(blocks, key=len) if blocks else text
    (project / name).write_text(code if code.endswith("\n") else code + "\n", encoding="utf-8")
    agent.messages = [*messages, {"role": "assistant", "content": text}]
    return text


_FAKEWEB: Any = None


def fake_web(web_files: dict[str, str]) -> Any:
    """evals/fakeweb.py's FakeWeb for a case with `"web": "fixture"` (the research suite): the
    deterministic offline web behind web_search/fetch_page in an eval run."""
    global _FAKEWEB
    if _FAKEWEB is None:
        import importlib.util

        path = EVALS_DIR / "fakeweb.py"
        spec = importlib.util.spec_from_file_location("vega_evals_fakeweb", path)
        if spec is None or spec.loader is None or not path.is_file():
            raise FixtureError(f"web fixture needs {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FAKEWEB = module
    try:
        return _FAKEWEB.FakeWeb(web_files)
    except ValueError as exc:
        raise FixtureError(str(exc)) from exc


def _default_agent(project: Path, max_steps: int, planning: str = "auto") -> CodingAgent:
    return CodingAgent(project, allow_commands=False, max_steps=max_steps, planning=planning)


# Option arguments a recorded read-only command may carry and still run. Others are declined:
# some write or execute despite an inspection command name (git diff --output=FILE, rg --pre=CMD,
# file -C), and _is_auto_read_only does not look at options.
_SAFE_LONG_OPTIONS = {
    "--stat", "--short", "--porcelain", "--oneline", "--name-only", "--name-status", "--cached",
    "--staged", "--branch", "--all", "--graph", "--decorate", "--untracked-files", "--no-color",
    "--line-number", "--count", "--files-with-matches", "--ignore-case", "--recursive",
}


class CommandRecorder:
    """approve_command for `commands: "record"` cases: records every run_command request
    ({step, command, ran}) and declines it, so the agent gets the normal "User declined" result
    and nothing it asked for is executed. The one exception is an inspection command Vega already
    auto-runs (tools._is_auto_read_only: ls, cat, grep, git status/diff/log/show... scoped to the
    project) with only plain options: it runs inside the attempt's temporary project, as it would
    in a session, and a git one only when the fixture is its own git repository (`git_repo`), so
    git never walks up into a repository outside it."""

    sees_read_only = True  # tools.execute: send auto-read-only commands here too

    def __init__(self, project: Path, step: Callable[[], int]) -> None:
        self.project = project
        self.step = step
        self.requested: list[dict[str, Any]] = []

    def __call__(self, command: str) -> bool:
        allowed = self.may_run(command)
        self.requested.append({"step": self.step(), "command": command, "ran": allowed})
        return allowed

    def may_run(self, command: str) -> bool:
        import shlex

        from .tools import _is_auto_read_only

        if not _is_auto_read_only(command, self.project):
            return False
        parts = shlex.split(command)
        name = Path(parts[0]).name
        if name == "git" and not (self.project / ".git").is_dir():
            return False
        options = [arg for arg in parts[1:] if arg.startswith("-")]
        if name == "file" and options:
            return False
        return all(re.fullmatch(r"-[A-Za-z0-9]+", arg) or arg in _SAFE_LONG_OPTIONS
                   for arg in options)

    @property
    def commands(self) -> list[str]:
        return [item["command"] for item in self.requested]


# Multi-step suites always offer planning; one-edit `fixes` cases leave it to "auto".
SUITE_PLANNING = {"goals": "on", "selfbuild": "on", "pyprojects": "on", "engineering": "on"}


def _failure_kind(outcome: CheckOutcome) -> str:
    kinds = [kind for kind, detail in (("acceptance", outcome.acceptance),
                                       ("regression", outcome.regression)) if detail]
    return "+".join(kinds)


WRITE_TOOL_NAMES = {"edit_file", "replace_lines", "write_file", "replace_function",
                    "replace_definition"}
WRITE_OK_PREFIXES = ("Edited ", "Replaced lines ", "Wrote ", "Replaced function ", "Added function ",
                     "Replaced definition ", "Added definition ")
TRANSCRIPTS_DIR = PROJECT_ROOT / "evals" / "transcripts"
_ASKS_USER = re.compile(
    r"\?\s*$|\b(?:could you|can you|please (?:provide|run|share|clarify|confirm)|which (?:file|one))\b",
    re.IGNORECASE)
FAILURE_CLASSES = (
    "model_error", "step_limit", "cut_off", "no_tool_use", "asked_user", "malformed_calls",
    "no_edit", "edit_failed", "broken_code", "wrong_file", "wrong_fix", "wrong_answer", "regression",
)


def _is_read_only(case: dict[str, Any]) -> bool:
    """A question to answer, not a change to make: checked by its answer, not by code."""
    answer_checks = case.get("response_contains") or case.get("response_matches")
    code_checks = (case.get("python_check") or case.get("check_command") or case.get("file_contains")
                   or case.get("file_not_contains") or case.get("hidden_files"))
    return bool(answer_checks) and not code_checks


def classify_failure(case: dict[str, Any], attempt: dict[str, Any], response: str,
                     writes_ok: int, writes_failed: int) -> str:
    """One deterministic label for why an attempt failed (\"\" when it passed)."""
    if attempt["passed"]:
        return ""
    detail = attempt.get("detail", "")
    if attempt.get("model_error"):
        return "model_error"
    if attempt.get("hit_step_limit"):
        return "step_limit"
    if response.startswith("Stopped: the model's replies kept hitting the length limit"):
        return "cut_off"
    if attempt.get("failure") == "regression":
        return "regression"
    if "was modified" in detail or _wrong_file(case, attempt):
        return "wrong_file"
    if _is_read_only(case):
        return "wrong_answer"
    if not attempt.get("tool_calls"):
        return "asked_user" if _ASKS_USER.search(response or "") else "no_tool_use"
    if writes_ok == 0:
        if writes_failed:
            return "edit_failed"
        nudges = attempt.get("nudges", {})
        if (nudges.get("tool call written as text could not be parsed")
                or nudges.get("action failed schema validation")):  # constrained mode's equivalent
            return "malformed_calls"
        if _ASKS_USER.search(response or ""):
            return "asked_user"
        return "no_edit"
    if "diff" in attempt and not attempt["diff"].get("files"):
        # writes that left no change behind (rewrote the same text, or undid their own edit)
        return "asked_user" if _ASKS_USER.search(response or "") else "no_edit"
    if any(marker in detail for marker in ("SyntaxError", "IndentationError")):
        return "broken_code"
    return "wrong_fix"


def _unexpected_files(expected: list[str] | None, changed: list[str]) -> list[str]:
    """Changed files outside the case's `expected_files` ([] when the case lists none). Vega's own
    bookkeeping and the repro_first test (`repro_first.REPRO_NAME`, written by the harness's
    opt-in reproduce-first step) are never unexpected."""
    if expected is None:
        return []
    return [path for path in changed
            if path not in expected and not _vega_own(path) and not _is_repro_test(path)]


def _is_repro_test(path: str) -> bool:
    return path.replace("\\", "/").rsplit("/", 1)[-1] == repro_first.REPRO_NAME


def _edited_expected_source(case: dict[str, Any], attempt: dict[str, Any]) -> bool:
    """The diff touches at least one of the case's expected files that is not a test file."""
    expected = case.get("expected_files") or []
    changed = set((attempt.get("diff") or {}).get("files") or [])
    return any(path in changed and not repro_first.is_test_path(path) for path in expected)


def _wrong_file(case: dict[str, Any], attempt: dict[str, Any]) -> bool:
    """Files outside `expected_files` changed and none of the expected source files was: the
    attempt worked in the wrong place. Extra files next to an edited expected source file
    (re-exports, a new helper module) are `extra_files`, and the attempt is labelled by what its
    edit did (wrong_fix, broken_code, ...)."""
    unexpected = [path for path in attempt.get("unexpected_files") or []
                  if not _is_repro_test(path)]
    return bool(unexpected) and not _edited_expected_source(case, attempt)


def _extra_files(case: dict[str, Any], attempt: dict[str, Any]) -> list[str]:
    """Unexpected files changed alongside an edited expected source file (not a wrong_file)."""
    if not _edited_expected_source(case, attempt):
        return []
    return list(attempt.get("unexpected_files") or [])


def _transcript_dir(suite: str, kind: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    part = shard()
    suffix = f"-shard{part[0]}of{part[1]}" if part else ""
    return TRANSCRIPTS_DIR / f"{stamp}-{suite}-{kind}{suffix}"


def _compact_transcript(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return transcripts.compact(messages)


def run_attempt(
    case: dict[str, Any], *, suite: str = "fixes", agent_factory: AgentFactory | None = None,
    archive: bytes | None = None, transcript: Path | None = None,
) -> dict[str, Any]:
    """Build the fixture, run the agent once, check the result and return the attempt record.

    Every attempt gets its own empty knowledge library, so the user's ~/.vega/knowledge never
    leaks into results; a case's `learn` files ({name: text}) are learned into it first when the
    `knowledge` feature is on (the learn suite's "after learning" condition)."""
    from . import knowledge

    saved = os.environ.get("VEGA_KNOWLEDGE_DIR")
    with tempfile.TemporaryDirectory(prefix="vega-eval-knowledge-") as library:
        os.environ["VEGA_KNOWLEDGE_DIR"] = library
        try:
            if case.get("learn") and features.on("knowledge"):
                material = Path(library).resolve() / "material"  # macOS: /var -> /private/var
                _write_fixture(material, case["learn"])
                knowledge.learn(str(material))
            return _run_attempt(case, suite=suite, agent_factory=agent_factory, archive=archive,
                                transcript=transcript)
        finally:
            if saved is None:
                os.environ.pop("VEGA_KNOWLEDGE_DIR", None)
            else:
                os.environ["VEGA_KNOWLEDGE_DIR"] = saved


def _classify_external(case: dict[str, Any], attempt: dict[str, Any], response: str) -> str:
    """classify_failure for an external agent, which has no tool calls to inspect: the same labels
    from its exit, its applied/failed edits and the diff."""
    if attempt["passed"]:
        return ""
    if attempt.get("model_error"):
        return "model_error"
    if attempt.get("hit_step_limit"):
        return "step_limit"
    if attempt.get("failure") == "regression":
        return "regression"
    if "was modified" in attempt.get("detail", "") or _wrong_file(case, attempt):
        return "wrong_file"
    if _is_read_only(case):
        return "wrong_answer"
    if not attempt["diff"]["files"]:
        if attempt.get("writes_failed"):
            return "edit_failed"
        return "asked_user" if _ASKS_USER.search(response or "") else "no_edit"
    if any(marker in attempt.get("detail", "") for marker in ("SyntaxError", "IndentationError")):
        return "broken_code"
    return "wrong_fix"


def _run_external_attempt(
    case: dict[str, Any], *, suite: str, agent_name: str, archive: bytes | None,
    transcript: Path | None,
) -> dict[str, Any]:
    """One attempt by an external agent (Aider): same fixture, sandbox copy, task text, model and
    checks as a Vega attempt; the record has the same fields, with what the agent cannot report
    (tool calls, nudges, time to first reply, model seconds) left empty."""
    max_steps = int(case.get("max_steps") or DEFAULT_MAX_STEPS.get(suite, 40))
    model = os.getenv("VEGA_MODEL", "vega")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vega-eval-") as temp_dir:
        project = Path(temp_dir).resolve()  # the attempt's own copy, symlinks resolved as Vega's
        try:
            build_fixture(case, project, archive)
        except FixtureError as exc:
            return {"passed": False, "seconds": round(time.monotonic() - started, 2),
                    "detail": f"fixture error: {exc}", "failure": "fixture"}
        before = snapshot_files(project)
        run = external.run_aider(project, case["task"], model=model, max_steps=max_steps,
                                 before=before)
        response = run.response
        diff = diff_stats(before, snapshot_files(project))  # before hidden files are revealed
        outcome = run_checks(case, project, response)
    expected = case.get("expected_files")
    stats = run.stats
    attempt: dict[str, Any] = {
        "passed": outcome.passed,
        "seconds": round(time.monotonic() - started, 2),
        "detail": outcome.acceptance or outcome.regression or "",
        "failure": _failure_kind(outcome),
        "acceptance_passed": not outcome.acceptance,
        "hidden": list(outcome.hidden) if outcome.hidden else None,
        "regression_passed": None if outcome.regression is None else not outcome.regression,
        "agent": agent_name,
        "agent_version": _agent_version(agent_name),
        "agent_seconds": round(run.seconds, 2),  # includes Aider's start-up (~2-3 s)
        "first_response_seconds": None,
        "model_seconds": None,
        "model_calls": stats["model_calls"],
        "steps": stats["model_calls"],
        "max_steps": max_steps,
        "time_cap_seconds": run.time_cap,
        "hit_step_limit": run.timed_out,
        "model_error": stats["model_error"],
        "exit_code": run.exit_code,
        "tool_calls": {},
        "nudges": {},
        "text_tool_calls": 0,
        "compactions": 0,
        "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
        "tokens_approx": stats["prompt_tokens"] is not None,  # Aider rounds above 1,000
        "edit_format": stats["edit_format"],
        "chat_files": len(run.files),
        "repo_map": run.repo_map,
        "num_ctx": run.setup.num_ctx,
        "diff": diff,
        "asked": bool(_ASKS_USER.search(response or "")) and not diff["files"],
        "expect": case.get("expect"),
        "unexpected_files": _unexpected_files(expected, diff["files"]),
        "writes_ok": stats["writes_ok"],
        "writes_failed": stats["writes_failed"],
    }
    if outcome.regression and outcome.acceptance:
        attempt["regression_detail"] = outcome.regression
    attempt["extra_files"] = _extra_files(case, attempt)
    attempt["failure_class"] = _classify_external(case, attempt, response)
    if transcript is not None:
        try:
            transcript.parent.mkdir(parents=True, exist_ok=True)
            transcript.write_text(json.dumps({
                "case": case["id"], "task": case["task"], "agent": agent_name,
                "passed": attempt["passed"], "detail": attempt["detail"],
                "failure_class": attempt["failure_class"], "final_answer": response[:2000],
                "command": run.command, "chat_files": run.files,
                "output": run.output[-20000:], "chat_history": run.chat_history[-50000:],
                "messages": [],
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            attempt["transcript"] = transcript.name
        except OSError:
            pass
    return attempt


def _run_attempt(
    case: dict[str, Any], *, suite: str, agent_factory: AgentFactory | None,
    archive: bytes | None, transcript: Path | None,
) -> dict[str, Any]:
    agent_name = external.current_agent()
    if agent_name != "vega" and agent_factory is None:
        return _run_external_attempt(case, suite=suite, agent_name=agent_name, archive=archive,
                                     transcript=transcript)
    max_steps = int(case.get("max_steps") or DEFAULT_MAX_STEPS.get(suite, 40))
    factory = agent_factory or _default_agent
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vega-eval-") as temp_dir:
        project = Path(temp_dir).resolve()  # macOS: /var -> /private/var
        try:
            build_fixture(case, project, archive)
            # research suite: an offline fake web behind the web tools
            fake = fake_web(case.get("web_files") or {}) if case.get("web") == "fixture" else None
        except FixtureError as exc:
            return {"passed": False, "seconds": round(time.monotonic() - started, 2),
                    "detail": f"fixture error: {exc}", "failure": "fixture"}
        before = snapshot_files(project)
        debuglog.start(f"{suite}-{case.get('id', 'case')}")  # one live log per attempt (VEGA_DEBUG)
        agent = (factory(project, max_steps, SUITE_PLANNING.get(suite, "auto"))
                 if factory is _default_agent else factory(project, max_steps))
        # Web tools only for a fake-web case, approved without asking (the web is local) and not
        # cached in the knowledge library; every other case runs with no web at all.
        agent.web = (web.WebSession(project, backend=web.FakeWebBackend(fake),
                                    permission="allow", cache=False) if fake is not None
                     else web.WebSession(project, backend=web.NoBackend()))
        if case.get("teach_mode") and hasattr(agent, "set_teach"):
            agent.set_teach(True)  # teaching suite's teach-mode- cases: the user turned it on
        meter = _Meter()
        recorder = None
        if case.get("commands") == "record":  # observe the commands it would run; run none
            recorder = CommandRecorder(project, lambda: meter.main_calls)
            agent.allow_commands, agent.approve_command = True, recorder
        agent.client = _MeteredClient(agent.client, meter, main=True)
        previous = agent.on_event

        def on_event(kind: str, data: dict[str, Any]) -> None:
            meter.on_event(kind, data)
            if previous is not None:
                previous(kind, data)

        agent.on_event = on_event
        meter.started = agent_started = time.monotonic()
        with _meter_subagents(meter):
            if os.environ.get("VEGA_EVAL_MODE") == "single-shot":
                response = single_shot(agent, case, project)
            else:
                response = agent.run(case["task"])
        agent_seconds = time.monotonic() - agent_started
        diff = diff_stats(before, snapshot_files(project))  # before hidden files are revealed
        if fake is not None:  # always (over)written, so the agent cannot forge it
            fake.write_log(project)
        outcome = run_checks(case, project, response, recorder.commands if recorder else None)
        debuglog.record("outcome", case=case.get("id"), passed=outcome.passed,
                        detail=outcome.acceptance or outcome.regression or "",
                        seconds=round(time.monotonic() - started, 2))
    expected = case.get("expected_files")
    used_tokens = meter.usage_reports > 0
    attempt: dict[str, Any] = {
        "passed": outcome.passed,
        "seconds": round(time.monotonic() - started, 2),
        "detail": outcome.acceptance or outcome.regression or "",
        "failure": _failure_kind(outcome),
        "acceptance_passed": not outcome.acceptance,
        "hidden": list(outcome.hidden) if outcome.hidden else None,
        "regression_passed": None if outcome.regression is None else not outcome.regression,
        "agent_seconds": round(agent_seconds, 2),
        "first_response_seconds": (
            round(meter.first_response, 2) if meter.first_response is not None else None),
        "model_seconds": round(meter.model_seconds, 2),
        "model_calls": meter.calls,
        "steps": meter.main_calls,
        "max_steps": max_steps,
        "hit_step_limit": response.startswith(STEP_LIMIT_PREFIX),
        "model_error": response.startswith(MODEL_ERROR_PREFIX),
        "tool_calls": dict(sorted(meter.tool_calls.items())),
        "nudges": dict(sorted(meter.nudges.items())),
        "text_tool_calls": meter.text_tool_calls,
        "compactions": meter.compactions,
        "prompt_tokens": meter.prompt_tokens if used_tokens else None,
        "completion_tokens": meter.completion_tokens if used_tokens else None,
        "diff": diff,
        # asked the user instead of building: a question and no file changed (clarify metrics)
        "asked": bool(_ASKS_USER.search(response or "")) and not diff["files"],
        "expect": case.get("expect"),
        "unexpected_files": _unexpected_files(expected, diff["files"]),
    }
    if outcome.regression and outcome.acceptance:
        attempt["regression_detail"] = outcome.regression
    if recorder is not None:
        attempt["commands_requested"] = recorder.requested
    if fake is not None:
        attempt["web_calls"] = list(fake.calls)
    attempt["writes_ok"], attempt["writes_failed"] = meter.writes_ok, meter.writes_failed
    route = getattr(agent, "last_route", None)  # where the task went, and why (routing)
    if isinstance(route, dict):
        attempt["route"] = {key: route[key] for key in ("name", "reasons", "targets", "inputs",
                                                         "fell_back") if key in route}
    attempt["extra_files"] = _extra_files(case, attempt)
    attempt["failure_class"] = classify_failure(
        case, attempt, response, meter.writes_ok, meter.writes_failed)
    if transcript is not None:
        try:
            transcript.parent.mkdir(parents=True, exist_ok=True)
            transcript.write_text(json.dumps({
                "case": case["id"], "task": case["task"], "passed": attempt["passed"],
                "detail": attempt["detail"], "failure_class": attempt["failure_class"],
                "final_answer": response[:2000], "messages": _compact_transcript(agent.messages),
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            attempt["transcript"] = transcript.name
        except OSError:
            pass  # a transcript is a diagnostic aid; never fail the attempt over it
    return attempt


def _total_tokens(attempt: dict[str, Any]) -> int | None:
    prompt, completion = attempt.get("prompt_tokens"), attempt.get("completion_tokens")
    if prompt is None or completion is None:
        return None
    return prompt + completion


def _median(values: list[float | int | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def summarise(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Run-level numbers over every attempt (fixture errors excluded by the caller)."""
    total = len(attempts)
    passed = sum(bool(item.get("passed")) for item in attempts)
    return {
        "attempts": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else None,
        "median_seconds": _median([item.get("seconds") for item in attempts]),
        "median_tokens": _median([_total_tokens(item) for item in attempts]),
        "median_model_calls": _median([item.get("model_calls") for item in attempts]),
        "acceptance_failures": sum(item.get("acceptance_passed") is False for item in attempts),
        "regressions": sum(item.get("regression_passed") is False for item in attempts),
        "step_limit_hits": sum(bool(item.get("hit_step_limit")) for item in attempts),
        "model_errors": sum(bool(item.get("model_error")) for item in attempts),
        "nudges": dict(sum((Counter(item.get("nudges") or {}) for item in attempts), Counter())),
        "compactions": sum(item.get("compactions") or 0 for item in attempts),
        "failure_classes": dict(Counter(
            item["failure_class"] for item in attempts if item.get("failure_class")).most_common()),
        **({"routes": dict(Counter(
            item["route"]["name"] + ("->agent" if item["route"].get("fell_back") else "")
            for item in attempts if isinstance(item.get("route"), dict)).most_common())}
           if any(isinstance(item.get("route"), dict) for item in attempts) else {}),
        **({"clarify": clarify_metrics(attempts)} if any(a.get("expect") for a in attempts) else {}),
    }


def clarify_metrics(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Asking, as precision and recall: on requests that need a question (expect "ask") and on
    clear requests that should simply be done (expect "act")."""
    vague = [a for a in attempts if a.get("expect") == "ask"]
    clear = [a for a in attempts if a.get("expect") == "act"]
    asked_vague = sum(bool(a.get("asked")) for a in vague)
    asked_clear = sum(bool(a.get("asked")) for a in clear)
    asked = asked_vague + asked_clear
    return {
        "vague": len(vague), "asked_on_vague": asked_vague,
        "clear": len(clear), "asked_on_clear": asked_clear,
        "ask_recall": round(asked_vague / len(vague), 4) if vague else None,
        "false_question_rate": round(asked_clear / len(clear), 4) if clear else None,
        "ask_precision": round(asked_vague / asked, 4) if asked else None,
    }


def consistency_metrics(cases: list[dict[str, Any]], records: list[dict[str, Any]]
                        ) -> dict[str, Any] | None:
    """Robustness: cases with the same `variant_of` are one task worded differently. A task is
    consistent when every wording got the same result; robust when every wording passed."""
    groups: dict[str, list[bool]] = {}
    for case, record in zip(cases, records):
        key = case.get("variant_of")
        if key and record.get("attempts"):
            groups.setdefault(key, []).append(record["passed"] == len(record["attempts"]))
    if not groups:
        return None
    consistent = sum(len(set(results)) == 1 for results in groups.values())
    robust = sum(all(results) for results in groups.values())
    some = sum(any(results) for results in groups.values())
    return {"tasks": len(groups), "consistent": consistent, "robust": robust,
            "passed_some_wording": some}


def _clarify_line(summary: dict[str, Any]) -> str | None:
    c = summary.get("clarify")
    if not c:
        return None
    precision = "—" if c["ask_precision"] is None else f"{c['ask_precision']:.0%}"
    return (f"Clarification: asked on {c['asked_on_vague']}/{c['vague']} vague requests (recall), "
            f"on {c['asked_on_clear']}/{c['clear']} clear ones (false questions); "
            f"precision of asking {precision}")


# --- Running suites -----------------------------------------------------------------------------

def _resolve_model(value: str) -> str:
    """A model profile name (built as vega-<profile> on demand) or an Ollama model name."""
    from . import models

    try:
        return models.resolve(value)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _server_fingerprint(model: str) -> dict[str, Any]:
    from . import server

    try:
        return server.fingerprint(model)
    except Exception as exc:  # recording context must never break a run
        return {"error": str(exc)}


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"], text=True, capture_output=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--", "src"], text=True, capture_output=True,
    ).stdout.strip()
    return (result.stdout.strip() or "unknown") + ("+dirty" if dirty else "")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if digits else f"{value:.0f}"


def shard() -> tuple[int, int] | None:
    """VEGA_EVAL_SHARD="k/n": this process runs every n-th case starting at k (1-based), so n
    processes can split a suite and run at the same time; `--merge` recombines their records."""
    raw = os.environ.get("VEGA_EVAL_SHARD", "").strip()
    if not raw:
        return None
    try:
        k, n = (int(part) for part in raw.split("/"))
    except ValueError:
        raise typer.BadParameter(f"Shard must look like 1/2, got {raw!r}.") from None
    if not 1 <= k <= n:
        raise typer.BadParameter(f"Shard {raw!r}: need 1 <= k <= n.")
    return k, n


SCREEN_MAX = 12  # a screen keeps the cases that disagreed most often
SCREEN_MIN = 8  # a screen runs at least this many cases
SCREEN_SEED = 7  # fixed: the top-up sample is the same every day, so screens compare like with like
SCREEN_NOTE = "screen: not a full result (cases chosen because they told setups apart before)"


def discriminating_cases(suite: str, runs: list[dict[str, Any]] | None = None) -> list[str]:
    """Sorted ids of `suite`'s cases that have ever told two setups apart in the recorded runs:
    a paired A/B where some pair disagreed (only A or only B passed), or plain runs where the case
    both passed and failed (in any attempt of any run). Cases that always pass or always fail
    cannot show a difference, so a screen skips them."""
    found: set[str] = set()
    seen_single: dict[str, set[bool]] = {}
    for run in runs if runs is not None else load_runs():
        if _run_suite_name(run) != suite:
            continue
        if run.get("mode") == "ab":
            variants = run.get("variants") or {}
            b_cases = {case.get("case"): case for case in (variants.get("B") or {}).get("cases", [])}
            for case_a in (variants.get("A") or {}).get("cases", []):
                case_b = b_cases.get(case_a.get("case"))
                if case_b is None:
                    continue
                if any(bool(a.get("passed")) != bool(b.get("passed"))
                       for a, b in zip(_attempts_of(case_a), _attempts_of(case_b))):
                    found.add(case_a["case"])
        else:
            for case in run.get("cases", []):
                if case.get("error") or "case" not in case:
                    continue
                results = seen_single.setdefault(case["case"], set())
                results.update(bool(a.get("passed")) for a in _attempts_of(case))
    found.update(case_id for case_id, results in seen_single.items() if len(results) > 1)
    return sorted(found)


def disagreement_rates(suite: str, runs: list[dict[str, Any]] | None = None
                       ) -> dict[str, tuple[int, int]]:
    """Per case: (paired attempts where A and B disagreed, paired attempts observed). Over many
    A/Bs nearly every case flips at least once, so a screen ranks by how often."""
    rates: dict[str, list[int]] = {}
    for run in runs if runs is not None else load_runs():
        if _run_suite_name(run) != suite or run.get("mode") != "ab":
            continue
        variants = run.get("variants") or {}
        b_cases = {case.get("case"): case for case in (variants.get("B") or {}).get("cases", [])}
        for case_a in (variants.get("A") or {}).get("cases", []):
            case_b = b_cases.get(case_a.get("case"))
            if case_b is None:
                continue
            for a, b in zip(_attempts_of(case_a), _attempts_of(case_b)):
                entry = rates.setdefault(case_a["case"], [0, 0])
                entry[0] += bool(a.get("passed")) != bool(b.get("passed"))
                entry[1] += 1
    return {case: (d, n) for case, (d, n) in rates.items()}


def screen_selection(
    suite: str, cases: list[dict[str, Any]], minimum: int = SCREEN_MIN,
    runs: list[dict[str, Any]] | None = None, maximum: int = SCREEN_MAX,
) -> tuple[list[dict[str, Any]], int]:
    """(screen cases in suite order, how many of them are discriminating). The `maximum` cases that
    disagreed most often between paired setups (rate, then count); fewer than `minimum` are topped
    up with a fixed seeded sample of the rest."""
    runs = runs if runs is not None else load_runs()
    rates = disagreement_rates(suite, runs)
    ever = set(discriminating_cases(suite, runs))
    ids = {case["id"] for case in cases}
    ranked = sorted((case_id for case_id in ever if case_id in ids),
                    key=lambda c: (-(rates.get(c, (0, 1))[0] / max(1, rates.get(c, (0, 1))[1])),
                                   -rates.get(c, (0, 0))[0], c))
    wanted = set(ranked[:maximum])
    chosen = {case["id"] for case in cases if case["id"] in wanted}
    discriminating = len(chosen)
    if len(chosen) < minimum:
        rest = sorted(case["id"] for case in cases if case["id"] not in chosen)
        chosen.update(random.Random(SCREEN_SEED).sample(rest, min(len(rest), minimum - len(chosen))))
    return [case for case in cases if case["id"] in chosen], discriminating


def _suite_cases(
    suite: str, only: list[str] | None, screen: bool = False, screen_min: int = SCREEN_MIN,
) -> tuple[list[dict[str, Any]], bytes | None, str]:
    """(cases, Vega source archive for snapshot cases or None, archive error)."""
    path = suite_file(suite)
    if not path.exists():
        raise typer.BadParameter(f"No cases for suite {suite!r}: {_display_path(path)} is missing.")
    cases = _load_cases(path)
    if only:
        cases = [case for case in cases if any(key in case["id"] for key in only)]
    area = os.environ.get("VEGA_EVAL_CATEGORY", "").strip()
    if area:
        if area not in categories.areas():
            raise typer.BadParameter(
                f"Unknown category {area!r}. Known: {', '.join(categories.areas())}.")
        cases = [case for case in cases if categories.matches(case["id"], area)]
    if screen:
        total = len(cases)
        cases, discriminating = screen_selection(suite, cases, screen_min)
        console.print(f"[yellow]Screen of {suite}: {len(cases)} of {total} cases "
                      f"({discriminating} that told setups apart before"
                      + (f", {len(cases) - discriminating} from a fixed sample" if len(cases) > discriminating
                         else "") + "). Not a full result.[/yellow]")
    part = shard()
    if part:
        cases = cases[part[0] - 1::part[1]]
    if not cases:
        raise typer.BadParameter(f"No evaluation cases matched in suite {suite!r}.")
    archive: bytes | None = None
    archive_error = ""
    if any(case.get("vega_snapshot") for case in cases):
        try:
            archive = vega_source_archive()
        except FixtureError as exc:
            archive_error = str(exc)
    return cases, archive, archive_error


def control_file(name: str) -> Path:
    """~/.vega/eval.pause and eval.stop: `vega lab pause|resume|stop` for any running eval."""
    from . import server

    return server.STATE_DIR / f"eval.{name}"


PAUSE_POLL_SECONDS = 5.0


class _Pacer:
    """Cooldown between attempts, the user's pause/stop switches and the resource guard."""

    def __init__(self, guard: Guard | None, cooldown: float, sleep: Callable[[float], None]):
        self.guard, self.cooldown, self.sleep = guard, cooldown, sleep
        self.started = False

    def before_attempt(self) -> str:
        """Empty string to go ahead, otherwise the reason the run must stop."""
        if self.started and self.cooldown > 0:
            self.sleep(self.cooldown)  # lets the GPU idle between attempts: less heat, steadier timings
        self.started = True
        if control_file("pause").exists():
            console.print("[yellow]Paused by you between attempts (vega lab resume continues; "
                          "vega lab stop ends the run and keeps its results).[/yellow]")
            while control_file("pause").exists() and not control_file("stop").exists():
                self.sleep(PAUSE_POLL_SECONDS)
            console.print("[yellow]Resumed.[/yellow]")
        if control_file("stop").exists():
            control_file("stop").unlink(missing_ok=True)
            control_file("pause").unlink(missing_ok=True)
            reason = "stopped by you (vega lab stop); partial results kept"
            console.print(f"[yellow]{escape(reason)}[/yellow]")
            return reason
        if self.guard is None:
            return ""
        stopped = self.guard.before_attempt() or ""
        if stopped:
            console.print(f"[yellow]{stopped}[/yellow]")
        return stopped


def _checked_agent(name: str) -> str:
    """A known agent whose executable is available (typer error otherwise)."""
    try:
        return external.check_agent(name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _agent_version(name: str) -> str | None:
    return external.agent_version(name)


# --- Pinned sampling ------------------------------------------------------------------------------
# Eval runs send an explicit temperature and seed with every model request (via VEGA_TEMPERATURE /
# VEGA_SEED, which CodingAgent.sampling and external agents read). Without them every call sampled
# at the server's default temperature and identical one-attempt runs flipped
# on a noticeable share of cases (2026-10-20), so a fix could not be measured.
EVAL_SEED = 7  # the seed lab experiments already use
EVAL_TEMPERATURE = 0.0  # when the model's profile documents no temperature of its own (Granite)
MODEL_TEMPERATURE = "model"  # VEGA_TEMPERATURE / --temperature value: send none (server default)


def eval_temperature(model: str) -> float:
    """The temperature evals pin for `model`: its profile's own `temperature` parameter when the
    profile documents one (coder-7b: 0.2), else EVAL_TEMPERATURE."""
    from . import models

    profile = models.profile_for_model(model)
    value = profile.parameters.get("temperature") if profile else None
    return float(value) if value is not None else EVAL_TEMPERATURE


@contextmanager
def pinned_sampling() -> Iterator[dict[str, Any]]:
    """Pin sampling for one eval run: VEGA_SEED defaults to EVAL_SEED and VEGA_TEMPERATURE to
    eval_temperature(the run's model) unless already set (--seed / --temperature). The value
    `model` sends no temperature (the server's own, as before). Both A/B arms run inside one pin,
    so they share it. The environment is restored afterwards: interactive use is unchanged.
    Yields the overrides every request carries (sampling_overrides())."""
    saved = {name: os.environ.get(name) for name in ("VEGA_SEED", "VEGA_TEMPERATURE")}
    try:
        if not (saved["VEGA_SEED"] or "").strip():
            os.environ["VEGA_SEED"] = str(EVAL_SEED)
        temperature = (saved["VEGA_TEMPERATURE"] or "").strip()
        if temperature.lower() == MODEL_TEMPERATURE:
            del os.environ["VEGA_TEMPERATURE"]
        elif not temperature:
            os.environ["VEGA_TEMPERATURE"] = str(eval_temperature(os.getenv("VEGA_MODEL", "vega")))
        overrides = sampling_overrides()
        console.print("Sampling: " + ", ".join(f"{key} {value}" for key, value in overrides.items())
                      + ("" if "temperature" in overrides else ", the model's own temperature")
                      + f", max_tokens {max_reply_tokens()}.")
        yield overrides
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def attempt_seed(index: int) -> Iterator[int | None]:
    """Seed attempt `index` (0-based) with the run's seed + index, so repeats are different but
    reproducible samples whenever the temperature is above 0 (one seed for every attempt would make
    them the same reply). Both arms of A/B pair `index` get the same seed. Yields the seed (None
    when no seed is set)."""
    base = (os.environ.get("VEGA_SEED") or "").strip()
    if not base:
        yield None
        return
    seed = int(base) + index
    os.environ["VEGA_SEED"] = str(seed)
    try:
        yield seed
    finally:
        os.environ["VEGA_SEED"] = base


def _sampling_record() -> dict[str, Any]:
    """The run record's `sampling`: what every request carried (attempt k's seed is seed + k)."""
    overrides = sampling_overrides()
    return {**overrides, "max_tokens": max_reply_tokens(),
            **({"attempt_seed": "seed + attempt index (0-based)"} if "seed" in overrides else {})}


def _pinned(run: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Run an eval entry point (run_suite, run_ab) inside pinned_sampling()."""
    import functools

    @functools.wraps(run)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        with pinned_sampling():
            return run(*args, **kwargs)

    return wrapper


@_pinned
def run_suite(
    suite: str, repeat: int = 3, only: list[str] | None = None,
    agent_factory: AgentFactory | None = None,
    guard: Guard | None = None, cooldown: float = 0.0,
    sleep: Callable[[float], None] = time.sleep, screen: bool = False, screen_min: int = SCREEN_MIN,
) -> dict[str, Any]:
    agent_name = _checked_agent(os.environ.get(external.AGENT_ENV) or "vega")
    cases, archive, archive_error = _suite_cases(suite, only, screen, screen_min)
    model = os.getenv("VEGA_MODEL", "vega")
    commit = _git_commit()
    transcripts = _transcript_dir(suite, "run")
    off = sorted(features.disabled())
    opted_in = sorted(features.enabled())
    console.print(f"Running {suite}: {len(cases)} cases × {repeat} attempts against {model!r} "
                  + (f"with {agent_name} {_agent_version(agent_name) or ''} "
                     if agent_name != "vega" else "")
                  + f"(code {commit}; features off: {', '.join(off) or 'none'}"
                  + (f"; opt-in on: {', '.join(opted_in)}" if opted_in else "") + ").")
    legacy = suite == "fixes"
    table = Table(title=f"Vega evaluation · {suite} · {model} · {commit}"
                  + (f" · agent {agent_name}" if agent_name != "vega" else ""))
    table.add_column("Case", overflow="fold")
    table.add_column("Passed", justify="right")
    if legacy:
        table.add_column("Avg s", justify="right")
    else:
        for column in ("Med s", "Med tok", "Calls", "Steps", "Diff ±", "Unlisted", "Failed"):
            table.add_column(column, justify="right")
    table.add_column("First failure")

    outcomes: list[dict[str, Any]] = []
    stopped = ""
    pacer = _Pacer(guard, cooldown, sleep)
    for case in cases:
        if stopped:
            break
        attempts: list[dict[str, Any]] = []
        error = ""
        if case.get("vega_snapshot") and archive_error:
            error = f"fixture error: {archive_error}"
        for attempt_number in range(repeat if not error else 0):
            stopped = pacer.before_attempt()
            if stopped:
                break
            with attempt_seed(attempt_number) as seed:
                result = run_attempt(
                    case, suite=suite, agent_factory=agent_factory, archive=archive,
                    transcript=transcripts / f"{case['id']}-{attempt_number + 1}.json")
            if seed is not None:
                result["seed"] = seed
            if result.get("failure") == "fixture":
                error = result["detail"]
                break
            attempts.append(result)
            outcome_text = "pass" if result["passed"] else "fail — " + escape(result["detail"])
            console.print(f"[dim]{case['id']} attempt {attempt_number + 1}/{repeat}: "
                          f"{outcome_text} · {_seconds_text(result.get('seconds'))}[/dim]")
        if error:
            # A broken fixture is a harness problem, not a model failure: excluded from the totals.
            console.print(f"[red]{case['id']}: {escape(error)}[/red]")
            outcomes.append({"case": case["id"], "passed": 0, "attempts": [], "error": error})
            table.add_row(case["id"], "[red]error[/red]", *([""] * (1 if legacy else 7)), escape(error))
            continue
        if not attempts:  # stopped by the guard before this case ran
            break
        passed = sum(item["passed"] for item in attempts)
        failure = next((item["detail"] for item in attempts if item["detail"]), "")
        outcomes.append({"case": case["id"], "passed": passed, "attempts": attempts})
        colour = "green" if passed == repeat else ("yellow" if passed else "red")
        cells = [case["id"], f"[{colour}]{passed}/{repeat}[/{colour}]"]
        if legacy:
            average = sum(item["seconds"] for item in attempts) / len(attempts)
            cells.append(f"{average:.1f}")
        else:
            stats = summarise(attempts)
            diff = _median([item["diff"]["added"] + item["diff"]["removed"] for item in attempts])
            unlisted = sum(bool(item["unexpected_files"]) for item in attempts)
            failed = ", ".join(
                f"{label} {count}" for label, count in (
                    ("acc", stats["acceptance_failures"]), ("reg", stats["regressions"]),
                    ("limit", stats["step_limit_hits"])) if count
            )
            cells += [
                _fmt(stats["median_seconds"]), _fmt(stats["median_tokens"], 0),
                _fmt(stats["median_model_calls"], 0),
                f"{_fmt(_median([item['steps'] for item in attempts]), 0)}"
                f"/{attempts[0]['max_steps']}",
                _fmt(diff, 0), f"{unlisted}/{len(attempts)}", failed or "—",
            ]
        table.add_row(*cells, escape(failure))

    scored = [case for case in outcomes if "error" not in case]
    by_id = {case["id"]: case for case in cases}
    consistency = consistency_metrics([by_id[o["case"]] for o in scored], scored)
    all_attempts = [attempt for case in scored for attempt in case["attempts"]]
    total_passed = sum(case["passed"] for case in scored)
    total_attempts = len(all_attempts)
    summary = summarise(all_attempts)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "model": model,
        "agent": agent_name,
        "agent_version": _agent_version(agent_name),
        "commit": commit,
        "repeat": repeat,
        "passed": total_passed,
        "total": total_attempts,
        "summary": summary,
        "areas": categories.breakdown(outcomes),
        "topics": categories.topic_breakdown(outcomes),
        **({"consistency": consistency} if consistency else {}),
        "features_off": sorted(features.disabled()),
        "features_on": opted_in,
        "sampling": _sampling_record(),
        "server": _server_fingerprint(model),
        "transcripts": _display_path(transcripts),
        "experiment": os.environ.get("VEGA_EXPERIMENT") or None,
        "eval_mode": os.environ.get("VEGA_EVAL_MODE") or "agent",
        "subset": bool(only) or screen or bool(os.environ.get("VEGA_EVAL_CATEGORY")),
        "shard": "/".join(map(str, shard())) if shard() else None,
        "cases": outcomes,
    }
    if screen:
        record["screen"] = True
    if guard is not None:
        record["resources"] = {**guard.report(), "cooldown_seconds": cooldown}
    if stopped:
        record["stopped"] = stopped
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    console.print(table)
    fully_passing = sum(case["passed"] == repeat for case in scored)
    console.print(
        f"Attempts passed: {rate_text(total_passed, total_attempts)} · "
        f"cases passing every attempt: {fully_passing}/{len(scored)}"
        + (" · [yellow]subset run: not comparable with full-suite scores[/yellow]" if only else "")
    )
    console.print(_summary_line(summary))
    if _clarify_line(summary):
        console.print(_clarify_line(summary))
    console.print(_areas_line(categories.breakdown(scored)))
    hidden = _hidden_line(scored)
    if hidden:
        console.print(hidden)
    if record["topics"]:
        console.print(_topics_line(record["topics"]))
    if consistency:
        console.print(f"Robustness: {consistency['consistent']}/{consistency['tasks']} tasks got the same "
                      f"result in every wording; {consistency['robust']} passed in every wording, "
                      f"{consistency['passed_some_wording']} in at least one")
    console.print(_times_line(scored))
    if "resources" in record:
        console.print(_resources_line(record["resources"]))
    if stopped:
        console.print(f"[yellow]Partial run ({len(scored)} of {len(cases)} cases): {stopped}[/yellow]")
    if screen:
        console.print(f"[yellow]{SCREEN_NOTE}; confirm with a full run.[/yellow]")
    errors = len(outcomes) - len(scored)
    if errors:
        console.print(
            f"[red]{errors} case(s) could not build their fixture and were not scored.[/red]")
    console.print(f"Recorded results in {_display_path(RUNS_FILE)}")
    return record


def install_stop_signal() -> None:
    """Turn SIGTERM (a plain `kill`) into KeyboardInterrupt, so a killed A/B still saves its
    finished cases. Main thread only; a no-op where signals are unavailable."""
    import signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        return

    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    with contextlib.suppress(ValueError, AttributeError):
        signal.signal(signal.SIGTERM, interrupt)


def _seconds_text(seconds: float | None) -> str:
    if seconds is None:
        return "time n/a"
    return f"{seconds:.1f} s" if seconds < 90 else f"{seconds / 60:.1f} min"


def _times_line(cases: list[dict[str, Any]]) -> str:
    """'Time: 41.3 min total over 40 attempts · median 14.1 s · slowest 10% from 2.9 min ·
    slowest: a 4.1 min, b 3.8 min, c 3.0 min' for one variant's per-case records."""
    timed = [(case["case"], a["seconds"]) for case in cases for a in case.get("attempts", [])
             if a.get("seconds") is not None]
    if not timed:
        return "Time: not recorded"
    values = sorted(seconds for _, seconds in timed)
    p90 = values[min(len(values) - 1, int(len(values) * 0.9))]
    slowest = sorted(timed, key=lambda item: -item[1])[:3]
    return (f"Time: {sum(values) / 60:.1f} min total over {len(values)} attempts · median "
            f"{_seconds_text(_median(values))} · slowest 10% from {_seconds_text(p90)} · slowest: "
            + ", ".join(f"{name} {_seconds_text(seconds)}" for name, seconds in slowest))


def _areas_line(by_area: dict[str, list[int]]) -> str:
    """'By area: Reasoning 4/13 (31%), ...' over primary areas."""
    if not by_area:
        return "By area: (no tagged cases)"
    parts = [f"{categories.name(area)} {passed}/{total} ({passed / total:.0%})"
             for area, (passed, total) in by_area.items() if total]
    return "By area: " + ", ".join(parts)


def hidden_totals(cases: list[dict[str, Any]]) -> tuple[int, int, int]:
    """(hidden tests passed, hidden tests run, attempts that ran hidden tests) over run cases."""
    passed = total = attempts = 0
    for case in cases:
        for attempt in case.get("attempts", []):
            if attempt.get("hidden"):
                passed += attempt["hidden"][0]
                total += attempt["hidden"][1]
                attempts += 1
    return passed, total, attempts


def _hidden_line(cases: list[dict[str, Any]]) -> str | None:
    """'Hidden tests: 23/64 passed (36%) over 8 attempts' — partial credit below the pass line."""
    passed, total, attempts = hidden_totals(cases)
    if not total:
        return None
    return f"Hidden tests: {passed}/{total} passed ({passed / total:.0%}) over {attempts} attempts"


def _hidden_fraction(attempt: dict[str, Any]) -> float | None:
    hidden = attempt.get("hidden")
    return hidden[0] / hidden[1] if hidden and hidden[1] else None


def _topics_line(by_topic: dict[str, list[int]]) -> str:
    """'By topic: Planning 2/4 (50%), ...' (engineering suite)."""
    return "By topic: " + ", ".join(f"{name} {passed}/{total} ({passed / total:.0%})"
                                    for name, (passed, total) in by_topic.items() if total)


def _summary_line(summary: dict[str, Any]) -> str:
    tokens = summary["median_tokens"]
    token_text = f"{tokens:.0f} tokens" if tokens is not None else "tokens n/a (no usage reported)"
    return (
        f"Per attempt (median of n={summary['attempts']}): {_fmt(summary['median_seconds'])} s · "
        f"{token_text} · {_fmt(summary['median_model_calls'], 0)} model calls · "
        f"acceptance failures {summary['acceptance_failures']}, regressions "
        f"{summary['regressions']}, step-limit hits {summary['step_limit_hits']}, "
        f"model errors {summary['model_errors']}"
        + (" · failure types: " + ", ".join(f"{name} {count}" for name, count in
                                           summary.get("failure_classes", {}).items())
           if summary.get("failure_classes") else "")
    )


def _resources_line(res: dict[str, Any]) -> str:
    free = f"{res['min_free_pct']}%" if res.get("min_free_pct") is not None else "n/a"
    return (
        f"Machine: lowest free memory {free} · peak swap growth {res['peak_swap_growth_mb']} MB · "
        f"peak model memory {res['peak_model_rss_mb'] / 1024:.1f} GB · lowest free disk "
        f"{res['min_disk_free_gb']} GB · paused {res['pauses']}× ({res['paused_seconds']} s) · "
        f"cooldown {res['cooldown_seconds']:g} s between attempts"
    )


def run_evaluations(
    repeat: int = 3, only: list[str] | None = None, suite: str = "fixes",
    agent_factory: AgentFactory | None = None,
    guard: bool = False, cooldown: float = 0.0, limits: Limits | None = None,
    screen: bool = False, screen_min: int = SCREEN_MIN,
) -> list[dict[str, Any]]:
    suites = [name for name in SUITES if suite_file(name).exists()] if suite == "all" else [suite]
    if not suites:
        raise typer.BadParameter("No suite files found.")
    records = []
    for name in suites:
        if suite == "all" and only:
            cases = _load_cases(suite_file(name))
            if not any(key in case["id"] for case in cases for key in only):
                continue
        watcher = Guard(limits, log=lambda text: console.print(f"[yellow]{escape(text)}[/yellow]")) if guard else None
        record = run_suite(
            name, repeat=repeat, only=only, agent_factory=agent_factory,
            guard=watcher, cooldown=cooldown, screen=screen, screen_min=screen_min,
        )
        records.append(record)
        if record.get("stopped"):
            break
    if not records:
        raise typer.BadParameter("No evaluation cases matched.")
    return records


# --- Paired A/B runs -----------------------------------------------------------------------------

AB_SERVER_FEATURES = {"kv_q8"}
EARLY_STOP_P = 0.005  # interim threshold; keeps the overall false-positive rate close to 5%
FUTILITY_P = 0.05  # the final significance level a futility look asks whether it can still reach


def futility_best_p(a_only: int, b_only: int, remaining: int) -> float:
    """The smallest exact McNemar p the run could still end with: every remaining pair goes the
    way of the side already ahead (B on a tie), as a discordant pair. Futility stops the run when
    even this is not below FUTILITY_P: no outcome of the remaining pairs could make it significant."""
    if a_only > b_only:
        return mcnemar_exact(a_only + remaining, b_only)
    return mcnemar_exact(a_only, b_only + remaining)


@_pinned
def run_ab(
    suite: str, b_off: list[str], repeat: int = 2, only: list[str] | None = None,
    agent_factory: AgentFactory | None = None, guard: Guard | None = None, cooldown: float = 0.0,
    sleep: Callable[[float], None] = time.sleep, early_stop: bool = False,
    b_on: list[str] | None = None, b_model: str | None = None,
    screen: bool = False, screen_min: int = SCREEN_MIN, futility: bool | None = None,
    b_agent: str | None = None,
) -> dict[str, Any]:
    """Paired comparison: A = current settings, B = A with `b_off` features switched off and the
    opt-in `b_on` features switched on (or another model `b_model`, or another agent `b_agent`,
    e.g. aider: the same cases, model and checks run through an external agent).

    Each case's attempts run as A/B pairs back to back, alternating which goes first, so drift, warm
    caches and luck hit both variants alike; the verdict comes from the pairs that disagree.
    When A and B use different models or agents, every A attempt runs first and then every B attempt (same
    cases, order and seeds), paired by case afterwards: alternating would reload the model on every
    attempt. `screen` runs only the discriminating cases; `futility` (default: on for screens only)
    stops at the halfway look when no outcome of the remaining pairs could reach p < 0.05."""
    b_on = list(b_on or [])
    # Opt-in features that are on (e.g. constrained as a profile default) can be switched off for B.
    b_off_opt_in = [name for name in b_off if name in features.OPT_IN]
    b_off = [name for name in b_off if name not in features.OPT_IN]
    unknown = set(b_off) - features.FEATURES.keys()
    if unknown:
        raise typer.BadParameter(f"Unknown feature(s): {', '.join(sorted(unknown))}.")
    b_on = [name for name in b_on if name not in features.FEATURES]  # defaults now: already on
    unknown_on = set(b_on) - features.OPT_IN.keys()
    if unknown_on:
        raise typer.BadParameter(
            f"Unknown opt-in feature(s): {', '.join(sorted(unknown_on))}. "
            f"Known: {', '.join(features.OPT_IN)}.")
    if set(b_off) & AB_SERVER_FEATURES:
        raise typer.BadParameter(
            "kv_q8 is a server setting and cannot alternate per attempt; compare it with two "
            "separate runs (restarting the server between them).")
    base = sorted(features.disabled())
    base_on = sorted(features.enabled())
    variants = {"A": base, "B": sorted(set(base) | set(b_off))}
    variants_on = {"A": base_on, "B": sorted((set(base_on) | set(b_on)) - set(b_off_opt_in))}
    models_by_variant = {"A": os.getenv("VEGA_MODEL", "vega"),
                         "B": _resolve_model(b_model) if b_model else os.getenv("VEGA_MODEL", "vega")}
    a_agent = _checked_agent(os.environ.get(external.AGENT_ENV) or "vega")
    agents_by_variant = {"A": a_agent, "B": _checked_agent(b_agent) if b_agent else a_agent}
    if (variants["A"] == variants["B"] and variants_on["A"] == variants_on["B"]
            and models_by_variant["A"] == models_by_variant["B"]
            and agents_by_variant["A"] == agents_by_variant["B"]):
        raise typer.BadParameter(
            "B must switch off at least one feature that A has on, switch on an opt-in "
            "feature that A has off, use a different model (--ab-model) or a different agent "
            "(--ab-agent).")
    cases, archive, archive_error = _suite_cases(suite, only, screen, screen_min)
    if futility is None:
        futility = screen and not shard()  # full runs stay full; a shard sees only part of the pairs
    grouped = (models_by_variant["A"] != models_by_variant["B"]
               or agents_by_variant["A"] != agents_by_variant["B"])
    interim_at = 0
    if early_stop:
        # One pre-registered interim look at half the pairs, stopping only on overwhelming evidence
        # (p < EARLY_STOP_P); shuffling spreads every kind of case across both halves.
        random.Random(int(os.environ.get("VEGA_SEED") or 0)).shuffle(cases)
    planned_pairs = len(cases) * repeat
    if early_stop or futility:
        interim_at = max(1, planned_pairs // 2)
    model = os.getenv("VEGA_MODEL", "vega")
    commit = _git_commit()
    transcripts = _transcript_dir(suite, "ab")
    console.print(
        f"A/B on {suite}: {len(cases)} cases × {repeat} pairs against {model!r} (code {commit}). "
        f"A off: {', '.join(variants['A']) or 'none'} · "
        f"B off: {', '.join(variants['B']) or 'none'}"
        + (f" · A model: {models_by_variant['A']} · B model: {models_by_variant['B']}"
           if models_by_variant["A"] != models_by_variant["B"] else "")
        + (f" · A agent: {agents_by_variant['A']} · B agent: {agents_by_variant['B']}"
           if agents_by_variant["A"] != agents_by_variant["B"] else "")
        + (f" · A on: {', '.join(variants_on['A']) or 'none'} · B on: "
           f"{', '.join(variants_on['B']) or 'none'}" if variants_on["A"] != variants_on["B"]
           or base_on else "") + "."
        + (" Models or agents differ: all A attempts run first, then all B (one model load "
           "each)." if grouped else "")
    )
    pacer = _Pacer(guard, cooldown, sleep)
    saved_env = os.environ.get(features.ENV)
    saved_env_on = os.environ.get(features.ENV_ON)
    saved_model = os.environ.get("VEGA_MODEL")
    saved_agent = os.environ.get(external.AGENT_ENV)
    per_variant: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    pairs_by_case: list[list[tuple[bool, bool]]] = []
    stopped = ""
    early_result: dict[str, Any] = {}
    a_pass: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []  # grouped: finished A attempts
    b_errors: set[str] = set()
    table = Table(title=f"A/B · {suite} · {model} · {commit}")
    for column in ("Case", "A", "B", "A med s", "B med s"):
        table.add_column(column, justify="left" if column == "Case" else "right")

    def run_one(case: dict[str, Any], index: int, name: str) -> dict[str, Any] | None:
        """One attempt of one variant; None when the run must stop (`stopped` says why)."""
        nonlocal stopped
        stopped = pacer.before_attempt()
        if stopped:
            return None
        os.environ[features.ENV] = ",".join(variants[name])
        os.environ[features.ENV_ON] = ",".join(variants_on[name])
        os.environ["VEGA_MODEL"] = models_by_variant[name]
        os.environ[external.AGENT_ENV] = agents_by_variant[name]
        with attempt_seed(index) as seed:  # pair `index`: the same seed for A and B
            result = run_attempt(
                case, suite=suite, agent_factory=agent_factory, archive=archive,
                transcript=transcripts / f"{case['id']}-{index + 1}-{name}.json")
        if seed is not None:
            result["seed"] = seed
        if result.get("failure") != "fixture":
            console.print(f"[dim]{case['id']} pair {index + 1}/{repeat} {name}: "
                          f"{'pass' if result['passed'] else 'fail'} · "
                          f"{_seconds_text(result.get('seconds'))}[/dim]")
        return result

    def finish_case(case: dict[str, Any], results: dict[str, list[dict[str, Any]]],
                    pairs: list[tuple[bool, bool]]) -> None:
        """Record one case's pairs, then take the interim look once half the pairs are in."""
        nonlocal early_result
        pairs_by_case.append(pairs)
        done = sum(len(case_pairs) for case_pairs in pairs_by_case)
        if interim_at and done >= interim_at and not early_result:
            flat_so_far = [pair for case_pairs in pairs_by_case for pair in case_pairs]
            a_so_far = sum(a and not b for a, b in flat_so_far)
            b_so_far = sum(b and not a for a, b in flat_so_far)
            interim_p = mcnemar_exact(a_so_far, b_so_far)
            early_result = {"at_pairs": done, "only_a": a_so_far, "only_b": b_so_far,
                            "mcnemar_p": round(interim_p, 4),
                            "stopped": early_stop and interim_p < EARLY_STOP_P}
            note = " — stopping early" if early_result["stopped"] else ""
            if futility and not early_result["stopped"]:
                best = futility_best_p(a_so_far, b_so_far, max(0, planned_pairs - done))
                early_result["best_possible_p"] = round(best, 4)
                early_result["futility"] = best >= FUTILITY_P
                if early_result["futility"]:
                    note = (f" — stopping for futility (even if all {planned_pairs - done} remaining "
                            f"pairs went one way, p could only reach {best:.3f})")
            console.print(f"[dim]Interim look at {done} pairs: only A {a_so_far}, only B "
                          f"{b_so_far}, p = {interim_p:.3f}{note}[/dim]")
        for name in ("A", "B"):
            per_variant[name].append({
                "case": case["id"], "passed": sum(r["passed"] for r in results[name]),
                "attempts": results[name],
            })
        table.add_row(
            case["id"], f"{sum(a for a, _ in pairs)}/{len(pairs)}",
            f"{sum(b for _, b in pairs)}/{len(pairs)}",
            _fmt(_median([r["seconds"] for r in results["A"]])),
            _fmt(_median([r["seconds"] for r in results["B"]])),
        )

    def halted() -> bool:
        return bool(stopped or early_result.get("stopped") or early_result.get("futility"))

    try:
        if not grouped:
            for case in cases:
                if halted():
                    break
                if case.get("vega_snapshot") and archive_error:
                    console.print(f"[red]{case['id']}: fixture error: {escape(archive_error)}[/red]")
                    continue
                results: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
                pairs: list[tuple[bool, bool]] = []
                error = ""
                for index in range(repeat):
                    order = ("A", "B") if index % 2 == 0 else ("B", "A")
                    pair: dict[str, dict[str, Any]] = {}
                    for name in order:
                        result = run_one(case, index, name)
                        if result is None:
                            break
                        if result.get("failure") == "fixture":
                            error = result["detail"]
                            break
                        pair[name] = result
                        results[name].append(result)
                    if error or stopped:
                        break
                    pairs.append((bool(pair["A"]["passed"]), bool(pair["B"]["passed"])))
                if error:
                    console.print(f"[red]{case['id']}: {escape(error)}[/red]")
                    continue
                if not pairs:
                    break
                finish_case(case, results, pairs)
        else:
            # Pass 1: every A attempt (one model load). Pass 2: every B attempt, same cases, order
            # and seeds; a case is paired once its B attempts are in.
            for case in cases:
                if stopped:
                    break
                if case.get("vega_snapshot") and archive_error:
                    console.print(f"[red]{case['id']}: fixture error: {escape(archive_error)}[/red]")
                    continue
                a_results: list[dict[str, Any]] = []
                error = ""
                for index in range(repeat):
                    result = run_one(case, index, "A")
                    if result is None:
                        break
                    if result.get("failure") == "fixture":
                        error = result["detail"]
                        break
                    a_results.append(result)
                if error:
                    console.print(f"[red]{case['id']}: {escape(error)}[/red]")
                    continue
                if not a_results:
                    break
                a_pass.append((case, a_results))
            for case, a_results in a_pass:
                if halted():
                    break
                b_results: list[dict[str, Any]] = []
                error = ""
                for index in range(len(a_results)):
                    result = run_one(case, index, "B")
                    if result is None:
                        break
                    if result.get("failure") == "fixture":
                        error = result["detail"]
                        break
                    b_results.append(result)
                if error:
                    console.print(f"[red]{case['id']}: {escape(error)}[/red]")
                    b_errors.add(case["id"])  # not scored on either side
                    continue
                if not b_results:
                    break
                pairs = [(bool(a["passed"]), bool(b["passed"])) for a, b in zip(a_results, b_results)]
                finish_case(case, {"A": a_results, "B": b_results}, pairs)
    except KeyboardInterrupt:
        # Ctrl-C or a kill (see install_stop_signal): keep every finished case instead of losing
        # the whole run; the case in progress is dropped.
        stopped = "interrupted; partial results kept"
        console.print(f"[yellow]{stopped}[/yellow]")
    finally:
        if saved_env is None:
            os.environ.pop(features.ENV, None)
        else:
            os.environ[features.ENV] = saved_env
        if saved_env_on is None:
            os.environ.pop(features.ENV_ON, None)
        else:
            os.environ[features.ENV_ON] = saved_env_on
        if saved_model is None:
            os.environ.pop("VEGA_MODEL", None)
        else:
            os.environ["VEGA_MODEL"] = saved_model
        if saved_agent is None:
            os.environ.pop(external.AGENT_ENV, None)
        else:
            os.environ[external.AGENT_ENV] = saved_agent

    # Grouped runs stopped before B caught up: A's finished cases are kept, unpaired and unscored.
    paired_ids = {case["case"] for case in per_variant["A"]}
    unpaired_a = [{"case": case["id"], "passed": sum(r["passed"] for r in results),
                   "attempts": results}
                  for case, results in a_pass if case["id"] not in paired_ids | b_errors]
    flat = [pair for pairs in pairs_by_case for pair in pairs]
    a_only = sum(a and not b for a, b in flat)
    b_only = sum(b and not a for a, b in flat)
    # partial credit: the share of hidden tests passed, paired attempt by attempt
    partial_by_case = []
    for case_a, case_b in zip(per_variant["A"], per_variant["B"]):
        pairs_frac = [(fa, fb) for fa, fb in zip(map(_hidden_fraction, case_a["attempts"]),
                                                   map(_hidden_fraction, case_b["attempts"]))
                      if fa is not None and fb is not None]
        if pairs_frac:
            partial_by_case.append(pairs_frac)
    partial_diff, partial_low, partial_high = bootstrap_mean_diff_ci(partial_by_case)
    p_value = mcnemar_exact(a_only, b_only)
    low, high = bootstrap_diff_ci(pairs_by_case)
    variant_rows = {}
    for name in ("A", "B"):
        attempts = [a for case in per_variant[name] for a in case["attempts"]]
        variant_rows[name] = {
            "features_off": variants[name], "features_on": variants_on[name],
            "model": models_by_variant[name],
            "agent": agents_by_variant[name], "agent_version": _agent_version(agents_by_variant[name]),
            "passed": sum(c["passed"] for c in per_variant[name]),
            "total": len(attempts), "summary": summarise(attempts), "cases": per_variant[name],
            "areas": categories.breakdown(per_variant[name]),
            "topics": categories.topic_breakdown(per_variant[name]),
        }
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "ab", "suite": suite, "model": model, "commit": commit, "repeat": repeat,
        "subset": bool(only) or screen or bool(os.environ.get("VEGA_EVAL_CATEGORY")),
        "shard": "/".join(map(str, shard())) if shard() else None,
        "order": "grouped" if grouped else "alternating",
        "sampling": _sampling_record(),
        "server": _server_fingerprint(model), "variants": variant_rows,
        "transcripts": _display_path(transcripts),
        "experiment": os.environ.get("VEGA_EXPERIMENT") or None,
        "eval_mode": os.environ.get("VEGA_EVAL_MODE") or "agent",
        "paired": {
            "pairs": len(flat), "a_only": a_only, "b_only": b_only,
            "both": sum(a and b for a, b in flat), "neither": sum(not a and not b for a, b in flat),
            "mcnemar_p": round(p_value, 4), "diff_b_minus_a": round(
                (sum(b for _, b in flat) - sum(a for a, _ in flat)) / len(flat), 4) if flat else 0.0,
            "diff_ci": [round(low, 4), round(high, 4)],
        },
        "partial": {  # share of hidden tests passed, B − A, paired by attempt
            "pairs": sum(len(case) for case in partial_by_case),
            "a": hidden_totals(per_variant["A"])[:2], "b": hidden_totals(per_variant["B"])[:2],
            "diff_b_minus_a": round(partial_diff, 4),
            "diff_ci": [round(partial_low, 4), round(partial_high, 4)],
        },
    }
    if guard is not None:
        record["resources"] = {**guard.report(), "cooldown_seconds": cooldown}
    if screen:
        record["screen"] = True
    if unpaired_a:
        record["unpaired"] = {"A": unpaired_a}
    if early_result.get("futility"):
        stopped = stopped or "futility"
    if stopped:
        record["stopped"] = stopped
    if early_result:
        record["interim"] = early_result
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    console.print(table)
    for name in ("A", "B"):
        row = variant_rows[name]
        opted = f"; on: {', '.join(row['features_on'])}" if row["features_on"] else ""
        agent_text = f"agent {row['agent']}; " if row["agent"] != "vega" else ""
        console.print(f"{name} ({agent_text}off: {', '.join(row['features_off']) or 'none'}{opted}): "
                      f"{rate_text(row['passed'], row['total'])}")
        console.print("   " + _summary_line(row["summary"]))
        if _clarify_line(row["summary"]):
            console.print("   " + _clarify_line(row["summary"]))
        console.print("   " + _areas_line(categories.breakdown(row["cases"])))
        hidden = _hidden_line(row["cases"])
        if hidden:
            console.print("   " + hidden)
        if row["topics"]:
            console.print("   " + _topics_line(row["topics"]))
        console.print("   " + _times_line(row["cases"]))
    paired = record["paired"]
    console.print(
        f"Pairs: {paired['pairs']} · only A passed {a_only} · only B passed {b_only} · both "
        f"{paired['both']} · neither {paired['neither']} · exact McNemar p = {p_value:.3f}"
    )
    console.print(
        f"B − A pass rate: {paired['diff_b_minus_a'] * 100:+.0f} points "
        f"(95% CI {low * 100:+.0f} to {high * 100:+.0f}, resampling cases)"
    )
    if partial_by_case:
        console.print(
            f"B − A hidden tests passed: {partial_diff * 100:+.0f} points per attempt (95% CI "
            f"{partial_low * 100:+.0f} to {partial_high * 100:+.0f}, resampling cases; "
            f"{sum(len(c) for c in partial_by_case)} pairs)")
    threshold = EARLY_STOP_P if early_result.get("stopped") else 0.05
    verdict = (
        "no detectable difference at this n" if p_value >= threshold or low <= 0 <= high
        else ("B is better" if b_only > a_only else "A is better")
    )
    console.print(f"Verdict: {verdict}." + (" (subset run)" if only else ""))
    if "resources" in record:
        console.print(_resources_line(record["resources"]))
    if stopped == "futility":
        console.print(f"[yellow]Stopped for futility at {early_result['at_pairs']} of {planned_pairs} "
                      f"pairs: no outcome of the rest could reach p < {FUTILITY_P}.[/yellow]")
    elif stopped:
        console.print(f"[yellow]Partial A/B run: {stopped}[/yellow]")
    if unpaired_a:
        console.print(f"[yellow]{len(unpaired_a)} case(s) ran only A before the stop; kept under "
                      f"'unpaired', not scored.[/yellow]")
    if screen:
        console.print(f"[yellow]{SCREEN_NOTE}; a default changes only on a full paired run.[/yellow]")
    console.print(f"Recorded results in {_display_path(RUNS_FILE)}")
    return record


def merge_runs(keys: list[str], path: Path | None = None) -> dict[str, Any]:
    """Combine shard runs (same suite, mode, model, code and settings) into one recorded run."""
    runs = load_runs(path)
    picked = [runs[select_run(runs, key)] for key in keys]
    if len(picked) < 2:
        raise typer.BadParameter("Give at least two runs to merge.")
    first = picked[0]
    for field in ("suite", "mode", "model", "commit", "repeat", "features_off", "features_on"):
        values = {json.dumps(run.get(field), sort_keys=True) for run in picked}
        if len(values) > 1:
            raise typer.BadParameter(f"Runs differ in {field!r}; only shards of one run can be merged.")
    merged: dict[str, Any] = {
        key: value for key, value in first.items()
        if key not in {"cases", "variants", "paired", "summary", "passed", "total", "resources",
                       "interim", "shard", "timestamp"}
    }
    merged.update({"timestamp": datetime.now(timezone.utc).isoformat(),
                   "merged_from": [run.get("timestamp") for run in picked],
                   "shards": [run.get("shard") for run in picked]})
    if first.get("mode") == "ab":
        variants = {}
        for name in ("A", "B"):
            cases = [case for run in picked for case in run["variants"][name]["cases"]]
            attempts = [a for case in cases for a in case["attempts"]]
            variants[name] = {"features_off": first["variants"][name]["features_off"],
                              "features_on": first["variants"][name].get("features_on", []),
                              "passed": sum(case["passed"] for case in cases),
                              "total": len(attempts), "summary": summarise(attempts), "cases": cases}
        pairs_by_case = []
        for case_a, case_b in zip(variants["A"]["cases"], variants["B"]["cases"]):
            pairs_by_case.append([(bool(a["passed"]), bool(b["passed"]))
                                  for a, b in zip(case_a["attempts"], case_b["attempts"])])
        flat = [pair for pairs in pairs_by_case for pair in pairs]
        a_only = sum(a and not b for a, b in flat)
        b_only = sum(b and not a for a, b in flat)
        low, high = bootstrap_diff_ci(pairs_by_case)
        merged["variants"] = variants
        merged["paired"] = {
            "pairs": len(flat), "a_only": a_only, "b_only": b_only,
            "both": sum(a and b for a, b in flat), "neither": sum(not a and not b for a, b in flat),
            "mcnemar_p": round(mcnemar_exact(a_only, b_only), 4),
            "diff_b_minus_a": round((sum(b for _, b in flat) - sum(a for a, _ in flat)) / len(flat), 4)
            if flat else 0.0,
            "diff_ci": [round(low, 4), round(high, 4)],
        }
        console.print(f"Merged {len(picked)} A/B shards: {len(flat)} pairs · only A {a_only} · only B "
                      f"{b_only} · p = {merged['paired']['mcnemar_p']:.3f} · B − A "
                      f"{merged['paired']['diff_b_minus_a'] * 100:+.0f} points "
                      f"(95% CI {low * 100:+.0f} to {high * 100:+.0f})")
        for name in ("A", "B"):
            console.print(f"  {name}: {rate_text(variants[name]['passed'], variants[name]['total'])}")
    else:
        cases = [case for run in picked for case in run["cases"]]
        attempts = [a for case in cases if not case.get("error") for a in case["attempts"]]
        merged.update({"cases": cases, "passed": sum(c["passed"] for c in cases if not c.get("error")),
                       "total": len(attempts), "summary": summarise(attempts)})
        console.print(f"Merged {len(picked)} shards: {rate_text(merged['passed'], merged['total'])}")
    with (path or RUNS_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(merged, ensure_ascii=False) + "\n")
    return merged


# --- Comparing runs -----------------------------------------------------------------------------

def load_runs(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or RUNS_FILE
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _run_suite_name(run: dict[str, Any]) -> str:
    return run.get("suite", "fixes")  # runs recorded before suites existed are all `fixes`


def select_run(runs: list[dict[str, Any]], key: str) -> int:
    """Index (0-based) of a run given a 1-based number, a negative index or a timestamp prefix."""
    try:
        number = int(key)
    except ValueError:
        matches = [
            index for index, run in enumerate(runs) if run.get("timestamp", "").startswith(key)
        ]
        if len(matches) != 1:
            raise typer.BadParameter(
                f"Timestamp {key!r} matches {len(matches)} runs; give a longer prefix or a number."
            ) from None
        return matches[0]
    index = number - 1 if number > 0 else len(runs) + number
    if number == 0 or not 0 <= index < len(runs):
        raise typer.BadParameter(
            f"Run {key} does not exist; there are {len(runs)} runs (1..{len(runs)}).")
    return index


def _vega_own(path: str) -> bool:
    """Vega's own bookkeeping (.vega/notes.md from `remember`, digests, state): not the agent's
    change to the project, so never an unexpected or wrong file."""
    return path == ".vega" or path.startswith(".vega/")


def _attempts_of(case: dict[str, Any]) -> list[dict[str, Any]]:
    if "attempts" in case:
        return case["attempts"]
    return [case]  # the earliest records stored one attempt per case inline


def _run_attempts(run: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cases = [case for case in run.get("cases", []) if not case.get("error")]
    return {case["case"]: _attempts_of(case) for case in cases}


def _pass_text(attempts: list[dict[str, Any]] | None) -> str:
    if not attempts:
        return "—"
    return f"{sum(bool(item.get('passed')) for item in attempts)}/{len(attempts)}"


def _delta(a: float | None, b: float | None, digits: int = 1, unit: str = "") -> str:
    if a is None or b is None:
        return "—"
    return f"{b - a:+.{digits}f}{unit}"


def compare_runs(
    first: str | None = None, second: str | None = None, suite: str | None = None,
    path: Path | None = None,
) -> None:
    """Print a per-case side-by-side of two runs (default: the last two of one suite)."""
    runs = load_runs(path)
    if first is not None and second is not None:
        index_a, index_b = select_run(runs, first), select_run(runs, second)
    elif first is not None or second is not None:
        raise typer.BadParameter("Give two runs to compare, or none for the last two.")
    else:
        if not runs:
            raise typer.BadParameter(f"No runs recorded in {_display_path(path or RUNS_FILE)}.")
        single = [run for run in runs if run.get("mode") != "ab"]
        wanted = suite or (_run_suite_name(single[-1]) if single else "fixes")
        matching = [
            index for index, run in enumerate(runs)
            if _run_suite_name(run) == wanted and run.get("mode") != "ab"
        ]
        if len(matching) < 2:
            raise typer.BadParameter(
                f"Need two recorded {wanted!r} runs to compare; found {len(matching)}.")
        index_a, index_b = matching[-2], matching[-1]
    run_a, run_b = runs[index_a], runs[index_b]
    for index, run in ((index_a, run_a), (index_b, run_b)):
        if run.get("mode") == "ab":
            raise typer.BadParameter(
                f"Run #{index + 1} is a paired A/B run; its own summary already compares A and B.")
    for label, index, run in (("A", index_a, run_a), ("B", index_b, run_b)):
        console.print(
            f"{label}: run #{index + 1} · {_run_suite_name(run)} · {run.get('model', '?')} · "
            f"code {run.get('commit', '?')} · {run.get('timestamp', '?')[:19]}"
        )
    if _run_suite_name(run_a) != _run_suite_name(run_b):
        console.print(
            "[yellow]These runs are from different suites; only shared cases line up.[/yellow]")
    cases_a, cases_b = _run_attempts(run_a), _run_attempts(run_b)
    table = Table(title="Vega evaluation comparison (A → B)")
    table.add_column("Case", overflow="fold")
    for column in ("A pass", "B pass", "A med s", "B med s", "A med tok", "B med tok"):
        table.add_column(column, justify="right")
    for name in list(cases_a) + [name for name in cases_b if name not in cases_a]:
        a, b = cases_a.get(name), cases_b.get(name)
        table.add_row(
            name, _pass_text(a), _pass_text(b),
            _fmt(_median([item.get("seconds") for item in a or []])),
            _fmt(_median([item.get("seconds") for item in b or []])),
            _fmt(_median([_total_tokens(item) for item in a or []]), 0),
            _fmt(_median([_total_tokens(item) for item in b or []]), 0),
        )
    console.print(table)
    shared = [name for name in cases_a if name in cases_b]
    summary_a = summarise([item for name in shared for item in cases_a[name]])
    summary_b = summarise([item for name in shared for item in cases_b[name]])
    rate_a, rate_b = summary_a["pass_rate"], summary_b["pass_rate"]
    pp = _delta(rate_a * 100 if rate_a is not None else None,
                rate_b * 100 if rate_b is not None else None, 1, " pp")
    console.print(
        f"Over {len(shared)} shared cases: pass {summary_a['passed']}/{summary_a['attempts']} "
        f"({_fmt(rate_a * 100 if rate_a is not None else None, 0)}%) → "
        f"{summary_b['passed']}/{summary_b['attempts']} "
        f"({_fmt(rate_b * 100 if rate_b is not None else None, 0)}%), {pp}"
    )
    console.print(
        f"Median s {_fmt(summary_a['median_seconds'])} → {_fmt(summary_b['median_seconds'])} "
        f"({_delta(summary_a['median_seconds'], summary_b['median_seconds'])}) · "
        f"median tokens {_fmt(summary_a['median_tokens'], 0)} → "
        f"{_fmt(summary_b['median_tokens'], 0)} "
        f"({_delta(summary_a['median_tokens'], summary_b['median_tokens'], 0)}) · "
        f"median model calls {_fmt(summary_a['median_model_calls'], 0)} → "
        f"{_fmt(summary_b['median_model_calls'], 0)}"
    )
    only_one = len(set(cases_a) ^ set(cases_b))
    note = f"; {only_one} case(s) appear in only one run and are excluded" if only_one else ""
    console.print(
        f"[dim]n = attempts per run shown above; no significance test — with small n, differences "
        f"of one or two attempts can be noise{note}.[/dim]"
    )
