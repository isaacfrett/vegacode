"""External coding agents run through Vega's eval harness (the market comparison plan).

`vega eval --agent aider` runs Aider in each attempt's sandbox copy of the fixture, on the same
Ollama model, task text and context window Vega gets; the harness's normal checks (hidden tests,
regression tests, file_unchanged, ...) then score the result exactly as they score Vega.

Aider is installed in its own venv outside the repo (~/.vega/tools/aider-venv)
or found via VEGA_AIDER or PATH. Nothing here talks to a model itself.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENTS = ("vega", "aider")
AGENT_ENV = "VEGA_EVAL_AGENT"  # which agent an eval attempt runs (default vega)
AIDER_ENV = "VEGA_AIDER"  # explicit path to the aider executable
AIDER_PIN = "0.86.2"  # the version installed in that venv
STEP_SECONDS_ENV = "VEGA_AGENT_STEP_SECONDS"
# Aider has no tool-step limit, so a case's max_steps becomes a wall-clock cap of this many seconds
# per step: Vega's slowest tasks ran ~30-40 s per step on a 16 GB Apple Silicon Mac (2026-10-15).
DEFAULT_STEP_SECONDS = 30
FILES_ENV = "VEGA_AIDER_FILES"  # auto (default) | all | none: which fixture files start in the chat
FILE_BUDGET_SHARE = 0.4  # auto: add every file only if they fit in this share of the context
CHARS_PER_TOKEN = 4
AIDER_ARTIFACTS = ".aider"  # Aider's own files (tags cache, histories) start with this
_TOKENS = re.compile(r"^Tokens: (?P<sent>[\d.]+k?) sent(?:, [\d.]+k? cache (?:write|hit))*, "
                     r"(?P<received>[\d.]+k?) received\.", re.MULTILINE)
_APPLIED = re.compile(r"^Applied edit to ", re.MULTILINE)
_EDIT_FAILED = re.compile(r"did not conform to the edit format|SearchReplaceNoExactMatch|"
                          r"Failed to apply edit", re.IGNORECASE)
_MODEL_ERROR = re.compile(r"\blitellm\.\w+(?:Error|Exception)\b")
_EDIT_FORMAT = re.compile(r"^(?:Main )?[Mm]odel: \S+ with (?P<format>[\w-]+) edit format", re.MULTILINE)


def current_agent() -> str:
    name = (os.environ.get(AGENT_ENV) or "vega").strip().lower()
    if name not in AGENTS:
        raise ValueError(f"Unknown agent {name!r}; choose one of: {', '.join(AGENTS)}.")
    return name


def check_agent(name: str) -> str:
    """Validate an agent name; for an external agent also that its executable can be found."""
    name = (name or "vega").strip().lower()
    if name not in AGENTS:
        raise ValueError(f"Unknown agent {name!r}; choose one of: {', '.join(AGENTS)}.")
    if name == "aider" and not aider_executable():
        raise ValueError(
            "Aider is not installed. Install it outside the repo: `uv venv "
            f"{default_aider_venv()} && uv pip install --python {default_aider_venv()} "
            f"aider-chat=={AIDER_PIN}` (or set {AIDER_ENV} to an aider executable).")
    return name


def default_aider_venv() -> Path:
    from . import server

    return server.STATE_DIR / "tools" / "aider-venv"


def aider_executable() -> str | None:
    """VEGA_AIDER, else the tools venv (~/.vega/tools/aider-venv), else `aider` on PATH."""
    explicit = os.environ.get(AIDER_ENV, "").strip()
    if explicit:
        return explicit if Path(explicit).is_file() else shutil.which(explicit)
    venv = default_aider_venv()
    for candidate in (venv / "bin" / "aider", venv / "Scripts" / "aider.exe"):
        if candidate.is_file():
            return str(candidate)
    return shutil.which("aider")


@functools.lru_cache(maxsize=8)
def _version_of(executable: str) -> str | None:
    try:
        result = subprocess.run([executable, "--version"], capture_output=True, text=True,
                                timeout=60, stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+\.\d+(?:\.\d+)?\S*)", result.stdout + result.stderr)
    return match.group(1) if match else None


def agent_version(name: str) -> str | None:
    """The external agent's version (e.g. "0.86.2"); None for Vega (its commit is recorded)."""
    if name != "aider":
        return None
    executable = aider_executable()
    return _version_of(executable) if executable else None


# --- Settings ------------------------------------------------------------------------------------

@dataclass
class ModelSetup:
    """The model side of an Aider run, matched to what Vega sends the same Ollama model."""
    model: str  # Ollama model name, e.g. vega-granite-8b
    num_ctx: int
    num_ctx_source: str
    ollama_base: str  # Ollama's native API root (OLLAMA_API_BASE), no /v1
    temperature: float | None = None  # None: the Modelfile's own (as Vega without VEGA_TEMPERATURE)
    seed: int | None = None
    request: dict[str, Any] = field(default_factory=dict)  # the profile's extra request fields

    @property
    def aider_name(self) -> str:
        return f"ollama_chat/{self.model}"


def model_setup(model: str) -> ModelSetup:
    from . import agent, context, models, server

    profile = models.endpoint_profile(model)
    if profile is not None:
        raise ValueError(
            f"The Aider adapter runs Ollama models only; profile {profile.name!r} is served by "
            f"{profile.endpoint}. Use an Ollama profile for --agent aider.")
    num_ctx, source = context.resolve_num_ctx(model)
    sampling = agent.sampling_overrides()
    url = server.base_url().rstrip("/")
    return ModelSetup(
        model=model, num_ctx=num_ctx, num_ctx_source=source,
        ollama_base=url.removesuffix("/v1"),
        temperature=sampling.get("temperature"), seed=sampling.get("seed"),
        request=models.request_options(model))


def model_settings(setup: ModelSetup) -> list[dict[str, Any]]:
    """Aider's --model-settings-file content (JSON is valid YAML). num_ctx is pinned: by default
    Aider sizes it per request (tokens x 1.25 + 8192), which would give the model a different,
    larger window than Vega's and make Ollama reload the model at every new size."""
    extra: dict[str, Any] = {"num_ctx": setup.num_ctx, **setup.request}
    if setup.seed is not None:
        extra["seed"] = setup.seed
    return [{
        "name": setup.aider_name,
        "extra_params": extra,
        # Aider sends temperature 0 unless told otherwise; Vega sends none (the Modelfile's) unless
        # VEGA_TEMPERATURE is set.
        "use_temperature": setup.temperature if setup.temperature is not None else False,
    }]


def model_metadata(setup: ModelSetup) -> dict[str, Any]:
    """Aider's --model-metadata-file: the real window (sizes the repo map and history summary)
    and zero cost, so it prints plain token counts."""
    return {setup.aider_name: {
        "max_input_tokens": setup.num_ctx, "max_tokens": setup.num_ctx,
        "max_output_tokens": setup.num_ctx, "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0, "litellm_provider": "ollama_chat", "mode": "chat",
    }}


def chat_files(files: dict[str, bytes], num_ctx: int, mode: str | None = None) -> list[str]:
    """Fixture files to put in Aider's chat at the start.

    Aider edits only files in the chat; its documented workflow is that the user adds them, and
    the repo map plus "add this file?" (answered yes) covers the rest. Choosing *which* files to add
    would leak the answer, so `auto` adds every visible text file when they fit in
    FILE_BUDGET_SHARE of the window (a user with a small project would add them all), and none
    otherwise (a large project: repo map only). `all` / `none` force either."""
    mode = (mode or os.environ.get(FILES_ENV) or "auto").strip().lower()
    if mode not in {"auto", "all", "none"}:
        raise ValueError(f"{FILES_ENV} must be auto, all or none, got {mode!r}.")
    if mode == "none":
        return []
    texts = {}
    for name, data in files.items():
        if name.split("/")[0].startswith(AIDER_ARTIFACTS):
            continue
        try:
            texts[name] = data.decode("utf-8")
        except UnicodeDecodeError:
            continue  # binary data is not something to put in a chat
    if mode == "all":
        return sorted(texts)
    tokens = sum(len(text) for text in texts.values()) / CHARS_PER_TOKEN
    return sorted(texts) if tokens <= num_ctx * FILE_BUDGET_SHARE else []


def time_cap(max_steps: int) -> int:
    raw = os.environ.get(STEP_SECONDS_ENV, "").strip()
    try:
        per_step = float(raw) if raw else DEFAULT_STEP_SECONDS
    except ValueError as exc:
        raise ValueError(f"{STEP_SECONDS_ENV} must be a number of seconds, got {raw!r}") from exc
    return max(60, int(max_steps * per_step))


def aider_command(executable: str, setup: ModelSetup, *, workdir: Path, files: list[str],
                  git: bool) -> list[str]:
    """The non-interactive Aider command line (pinned version: see AIDER_PIN). Settings, task and
    histories live in `workdir`, outside the project, so they never count as the agent's edits."""
    command = [
        executable,
        "--model", setup.aider_name, "--weak-model", setup.aider_name,
        "--model-settings-file", str(workdir / "model-settings.yml"),
        "--model-metadata-file", str(workdir / "model-metadata.json"),
        "--message-file", str(workdir / "task.txt"),
        "--yes-always",
        "--no-auto-commits", "--no-dirty-commits", "--no-gitignore",
        "--no-suggest-shell-commands",  # Vega's eval agent may not run commands either
        "--no-auto-test",
        "--no-detect-urls", "--no-check-update", "--no-show-release-notes", "--no-analytics",
        "--no-show-model-warnings", "--no-notifications",
        "--no-pretty", "--no-fancy-input", "--no-stream",  # --no-stream: token counts from Ollama
        "--no-restore-chat-history",
        "--chat-history-file", str(workdir / "chat.history.md"),
        "--input-history-file", str(workdir / "input.history"),
        "--encoding", "utf-8",
    ]
    if not git:
        command.append("--no-git")
    for name in files:
        command += ["--file", name]
    return command


# --- Running -------------------------------------------------------------------------------------

def _init_git(project: Path) -> bool:
    """A throwaway git repo of the fixture so Aider's repo map sees every file (without git it maps
    only the files in the chat). True when created here, so it can be removed afterwards."""
    if (project / ".git").exists():
        return False
    git = shutil.which("git")
    if not git:
        return False
    quiet = {"cwd": project, "capture_output": True, "stdin": subprocess.DEVNULL, "timeout": 60}
    identity = ["-c", "user.name=vega-eval", "-c", "user.email=vega-eval@localhost",
                "-c", "commit.gpgsign=false"]
    try:
        for args in (["init", "-q"], ["add", "-A"],
                     [*identity, "commit", "-q", "--allow-empty", "-m", "fixture"]):
            if subprocess.run([git, *args], check=False, **quiet).returncode:
                shutil.rmtree(project / ".git", ignore_errors=True)
                return False
    except (OSError, subprocess.SubprocessError):
        shutil.rmtree(project / ".git", ignore_errors=True)
        return False
    return True


def _remove_tree(path: Path) -> None:
    def make_writable(function: Any, target: str, _exc: Any) -> None:
        os.chmod(target, 0o700)  # Windows: git objects are read-only
        function(target)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=make_writable)
    else:  # pragma: no cover - 3.11
        shutil.rmtree(path, onerror=make_writable)


def cleanup(project: Path, before: set[str], created_git: bool) -> None:
    """Remove what Aider and this adapter added that is not project content."""
    if created_git and (project / ".git").exists():
        _remove_tree(project / ".git")
    for entry in project.iterdir():
        if entry.name.startswith(AIDER_ARTIFACTS) and entry.name not in before:
            if entry.is_dir():
                _remove_tree(entry)
            else:
                entry.unlink(missing_ok=True)


def _token_count(text: str) -> int:
    return round(float(text[:-1]) * 1000) if text.endswith("k") else int(float(text))


def parse_output(output: str) -> dict[str, Any]:
    """Model calls, tokens and edit outcomes from Aider's own report lines. Aider prints one
    "Tokens: N sent, M received." line per model call, rounded (1.2k above 1,000; 12k above 10,000),
    so the sums are approximate."""
    calls = list(_TOKENS.finditer(output))
    edit_format = _EDIT_FORMAT.search(output)
    return {
        "model_calls": len(calls),
        "prompt_tokens": sum(_token_count(m["sent"]) for m in calls) if calls else None,
        "completion_tokens": sum(_token_count(m["received"]) for m in calls) if calls else None,
        "writes_ok": len(_APPLIED.findall(output)),
        "writes_failed": len(_EDIT_FAILED.findall(output)),
        "model_error": bool(_MODEL_ERROR.search(output)),
        "edit_format": edit_format["format"] if edit_format else None,
    }


def assistant_text(chat_history: str) -> str:
    """The model's replies from Aider's chat history markdown: user messages are `#### ` lines and
    Aider's own output is quoted with `> `; what remains is the assistant's text. Used as the
    answer for answer-checked cases, so the echoed task never counts as the reply."""
    lines = []
    for line in chat_history.splitlines():
        if line.startswith(("#### ", "> ", "# aider chat started")) or line.strip() == ">":
            continue
        lines.append(line)
    return "\n".join(lines).strip()


@dataclass
class AgentRun:
    response: str
    seconds: float
    exit_code: int | None
    timed_out: bool
    output: str
    chat_history: str
    command: list[str]
    files: list[str]
    repo_map: bool
    setup: ModelSetup
    time_cap: int
    stats: dict[str, Any]


def _kill(process: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)  # Aider plus any linter it started
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass


def run_aider(project: Path, task: str, *, model: str, max_steps: int,
              before: dict[str, bytes]) -> AgentRun:
    """Run Aider once on `project` (the attempt's sandbox copy) and clean up after it."""
    executable = aider_executable()
    if not executable:
        check_agent("aider")  # raises with install instructions
        raise AssertionError("unreachable")
    setup = model_setup(model)
    files = chat_files(before, setup.num_ctx)
    cap = time_cap(max_steps)
    existing = {entry.name for entry in project.iterdir()}
    created_git = _init_git(project)
    git = created_git or (project / ".git").exists()
    with tempfile.TemporaryDirectory(prefix="vega-aider-") as temp:
        workdir = Path(temp)
        (workdir / "model-settings.yml").write_text(json.dumps(model_settings(setup), indent=1),
                                                    encoding="utf-8")
        (workdir / "model-metadata.json").write_text(json.dumps(model_metadata(setup), indent=1),
                                                     encoding="utf-8")
        (workdir / "task.txt").write_text(task, encoding="utf-8")
        command = aider_command(executable, setup, workdir=workdir, files=files, git=git)
        env = {**os.environ, "OLLAMA_API_BASE": setup.ollama_base, "PYTHONIOENCODING": "utf-8",
               "TERM": "dumb", "NO_COLOR": "1"}
        started = time.monotonic()
        timed_out = False
        try:
            process = subprocess.Popen(
                command, cwd=project, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
        except OSError as exc:
            output, exit_code = f"Could not start Aider: {exc}", None
        else:
            try:
                raw, _ = process.communicate(timeout=cap)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill(process)
                raw, _ = process.communicate()
            output = raw.decode("utf-8", "replace")
            exit_code = process.returncode
        seconds = time.monotonic() - started
        history_file = workdir / "chat.history.md"
        chat_history = history_file.read_text("utf-8", "replace") if history_file.exists() else ""
    cleanup(project, existing, created_git)
    stats = parse_output(output)
    if exit_code is None and not timed_out:
        stats["model_error"] = True
    return AgentRun(
        response=assistant_text(chat_history) or output[-2000:], seconds=seconds,
        exit_code=exit_code, timed_out=timed_out, output=output, chat_history=chat_history,
        command=command, files=files, repo_map=git, setup=setup, time_cap=cap, stats=stats)
