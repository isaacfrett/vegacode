"""Run test/check commands in an isolated, network-less copy of the project (macOS sandbox-exec)."""

from __future__ import annotations

import functools
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import features

EXECUTABLE = "sandbox-exec"  # macOS: the only sandbox so far (readonly_argv is the seam)
TIMEOUT = 60
OUTPUT_LIMIT = 8000
# Shared read-only instead of copied: large, and tests should not modify them.
LINKED_DIRS = {".git", ".venv", "venv", "node_modules"}
SKIPPED_DIRS = {"__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
PYTHON_NAMES = {"python", "python3"}
# Program -> allowed first arguments (None = any). Only test/check runners, not general shells.
ALLOWED = {
    "python": None,
    "python3": None,
    "pytest": None,
    "npm": {"test"},
    "go": {"test", "vet", "build"},
    "cargo": {"test", "check"},
    "swift": {"test", "build"},
    "ruff": {"check", "format"},
    "mypy": None,
    "pyright": None,
    "npx": {"tsc", "eslint"},
    "make": {"test", "check"},
    "node": {"--test"},
}
# sandbox_toolchains (2026-10-17): compilers and script runners for non-Python tasks.
# They run under the same sandbox (no network, writes only inside the clone). `node` also takes a
# script file inside the project (NODE_SCRIPTS); `Rscript` likewise takes an .R script.
TOOLCHAIN_ALLOWED: dict[str, set[str] | None] = {"javac": None, "java": None, "Rscript": None}
NODE_SCRIPTS = (".js", ".mjs", ".cjs")
R_SCRIPTS = (".r",)
# Human-readable allowlist for the run_check tool description and rejection messages.
RUN_CHECK_PROGRAMS = (
    "python / python3 ...",
    "pytest ...",
    "npm test",
    "go test|vet|build",
    "cargo test|check",
    "swift test|build",
    "ruff check ...",
    "ruff format --check ...",
    "mypy ...",
    "pyright ...",
    "npx tsc --noEmit ...",
    "npx eslint ...",
    "make test",
    "make check",
    "node --test ...",
)
TOOLCHAIN_PROGRAMS = ("javac ...", "java ...", "Rscript <script.R>", "node <script.js>")
# Checkers resolved from an environment: the project's venv first, then Vega's own.
PYTHON_TOOLS = {"ruff", "mypy", "pyright", "coverage"}
# Programs only Vega's own acceptance checks run (acceptance.py: commands a task names as its
# check, e.g. "must compile with `cc -std=c11 ...`", "measured with `coverage run --branch`").
HARNESS_ALLOWED: dict[str, set[str] | None] = {"cc": None, "gcc": None, "clang": None,
                                               "coverage": {"run"}}


@functools.cache
def available() -> bool:
    """True when sandbox-exec exists and can start a sandbox (it cannot nest inside another one)."""
    executable = shutil.which(EXECUTABLE) if sys.platform == "darwin" else None
    if executable is None:
        return False
    try:
        probe = subprocess.run(
            [executable, "-p", "(version 1)(allow default)", "/usr/bin/true"],
            capture_output=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def _quoted(path: Path | str) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _profile(workdir: Path, read_roots: list[Path] | None = None) -> str:
    """The clone's profile: no network, writes only inside the clone. `read_roots`
    (sandbox_read_guard): reads of GUARDED_READS are denied too, except under these roots."""
    profile = (
        "(version 1)(allow default)(deny network*)(deny file-write*)"
        f"(allow file-write* (subpath {_quoted(workdir)})"
        ' (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr"))'
    )
    if read_roots is not None:
        denied = " ".join(f"(subpath {_quoted(path)})" for path in GUARDED_READS)
        allowed = " ".join([*(f"(subpath {_quoted(root)})" for root in [workdir, *read_roots]),
                            *(f"(literal {_quoted(path)})" for path in READ_LITERALS),
                            *(f"(subpath {_quoted(path)})" for path in READ_SUBPATHS)])
        profile += f"(deny file-read-data {denied})(allow file-read-data {allowed})"
    return profile


# sandbox_read_guard (2026-10-20): a model-written deliverable script read the Mac's own /etc
# config file instead of the task's copy under root/etc/.
# Host areas whose contents a private_home run may not read (metadata stays readable, and
# `lsof`/`ps` ask the kernel, not these files) ...
GUARDED_READS = ("/Users", "/Volumes", "/private/etc", "/private/var", "/private/tmp",
                 "/Library/Keychains")
# ... except what the interpreter and the tools a script runs need there (the time zone, the
# developer folder xcode-select chose for git, the dyld cache on older macOS)
READ_LITERALS = ("/private/etc/localtime",)
READ_SUBPATHS = ("/private/var/db/timezone", "/private/var/select", "/private/var/db/dyld",
                 "/private/var/db/xcode_select_link")


def _interpreter_roots(argv: list[str]) -> list[Path]:
    """Folders a sandboxed Python (argv[0]) and Vega's own interpreter read from: prefixes, base
    prefixes and their resolved forms. A root that holds a whole home folder is left out."""
    roots: list[Path] = []
    for value in (sys.prefix, sys.base_prefix, sys.exec_prefix):
        roots += [Path(value), Path(value).resolve()]
    for executable in {argv[0], sys.executable} if argv else {sys.executable}:
        path = Path(executable)
        if path.is_absolute() and path.exists():
            roots += [path.parent.parent, path.resolve().parent.parent]
    homes = {Path.home().resolve(), Path.home()}
    keep: list[Path] = []
    for root in roots:
        if root in keep or len(root.parts) < 3 or any(
                home == root or root in home.parents for home in homes):
            continue
        keep.append(root)
    return keep


def _read_roots(workdir: Path, argv: list[str]) -> list[Path]:
    """sandbox_read_guard: what a private_home run may read under GUARDED_READS besides the clone:
    its linked folders' targets (.venv and the like) and the interpreter's folders."""
    linked = [entry.resolve() for entry in workdir.iterdir() if entry.is_symlink()]
    return [*linked, *_interpreter_roots(argv)]


def scrub_paths(output: str, workdir: Path, project: Path) -> str:
    """sandbox_path_scrub: host paths in command output replaced, so none reaches
    the model or an answer: the clone and the project by `.`, Vega's venv by <venv>, the
    interpreter's install (a uv or pyenv python) by <python>, the user's home folder by `~`.
    Only these real paths: a fixture's /Users/dana/... in output or a deliverable is left alone."""
    pairs: set[tuple[str, str]] = set()
    for path, label in ((workdir, "."), (project, "."), (Path(sys.prefix), "<venv>"),
                        (Path(sys.exec_prefix), "<venv>"), (Path(sys.base_prefix), "<python>"),
                        (Path.home(), "~")):
        for form in {str(path), str(path.resolve())}:
            if len(form) > 1:
                pairs.add((form, label))
                if form.startswith("/private/"):
                    pairs.add((form.removeprefix("/private"), label))
    for form, label in sorted(pairs, key=lambda pair: -len(pair[0])):
        output = output.replace(form, label)
    return output


def _clone(project: Path, workdir: Path) -> None:
    """Copy-on-write clone (APFS `cp -c`) of the project; heavy directories are symlinked."""
    for entry in project.iterdir():
        target = workdir / entry.name
        if entry.name in SKIPPED_DIRS:
            continue
        if entry.name in LINKED_DIRS or entry.is_symlink():
            target.symlink_to(entry.resolve() if entry.is_symlink() else entry)
            continue
        if entry.is_dir():
            result = subprocess.run(["cp", "-c", "-R", str(entry), str(target)], capture_output=True,
                                    check=False)
            if result.returncode:
                shutil.copytree(entry, target, symlinks=True, dirs_exist_ok=True)
        else:
            result = subprocess.run(["cp", "-c", str(entry), str(target)], capture_output=True,
                                    check=False)
            if result.returncode:
                shutil.copy2(entry, target)


def readonly_profile() -> str:
    """Profile for a read-only helper run in place (no clone): no network, no file writes."""
    return (
        "(version 1)(allow default)(deny network*)(deny file-write*)"
        '(allow file-write* (literal "/dev/null"))'
    )


def readonly_argv(argv: list[str]) -> list[str]:
    """`argv` to run a helper read-only in place (no network, no file writes): wrapped in the
    macOS sandbox when it is available. Elsewhere (Linux, Windows, or nested in another sandbox)
    `argv` is returned unchanged, and the callers' own limits are what remain: a fixed argv,
    input on stdin, a timeout. The one place a platform port adds its own sandbox for these."""
    if not available():
        return list(argv)
    return [EXECUTABLE, "-p", readonly_profile(), *argv]


def find_python_tool(project: Path, name: str) -> str | None:
    """Locate a Python-installed checker: project venv, then Vega's environment, then PATH."""
    for root in (project / ".venv", project / "venv", Path(sys.prefix)):
        candidate = root / "bin" / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(name)


def _not_found(tool: str, where: str) -> str:
    return (
        f"`{tool}` is not installed {where}. run_check has no network access, so checkers must "
        "already be installed locally; it was not run."
    )


def _python_for(project: Path) -> str:
    for candidate in (project / ".venv" / "bin" / "python", project / "venv" / "bin" / "python"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


SHELL_TOKENS = {"|", "||", "&&", ";", "<", ">", ">>", "2>", "2>&1", "&", "$(", "`"}


def parse_command(command: str, project: Path, harness: bool = False) -> list[str] | str:
    """Return argv for an allowed check command, or an error message string. `harness`: a check
    Vega itself runs (acceptance), which may also use the HARNESS_ALLOWED programs."""
    # argv is executed directly (no shell), so ;, |, && etc. are plain arguments, never chaining.
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return f"Could not parse command: {exc}"
    if not argv:
        return "Provide a command such as `python -m pytest` or `pytest tests/test_x.py`."
    # No shell runs these commands and stdin is empty: a heredoc, pipe or redirect would be passed
    # as plain arguments and `python -` would read nothing, exit 0 and look like a passing check
    # (2026-10-16: many failures "verified" work with checks that never ran).
    shell_syntax = next((arg for arg in argv if arg in SHELL_TOKENS or arg.startswith("<<")), None)
    if shell_syntax is not None:
        return (f"run_check runs one program without a shell, so `{shell_syntax}` would not work and "
                "nothing would run. Write the script to a file (e.g. check_tmp.py) and run "
                "`python check_tmp.py`, or use `python -c \"...\"` for one line.")
    if Path(argv[0]).name in PYTHON_NAMES and (len(argv) == 1 or argv[1] == "-"):
        return ("`python` with no script reads stdin, which is empty here, so nothing would run. "
                "Write the script to a file and run `python file.py`, or use `python -c \"...\"`.")
    program = Path(argv[0]).name
    toolchains = features.on("sandbox_toolchains")
    allowed_programs = {**ALLOWED, **(HARNESS_ALLOWED if harness else {}),
                        **(TOOLCHAIN_ALLOWED if toolchains else {})}
    if program not in allowed_programs:
        allowed = "; ".join(RUN_CHECK_PROGRAMS + (TOOLCHAIN_PROGRAMS if toolchains else ()))
        return (
            f"run_check only runs test/check programs ({allowed}). "
            "Use run_command for anything else."
        )
    script = _script_argument(argv, project) if toolchains else None
    if isinstance(script, str):
        return script
    subcommands = None if script else allowed_programs[program]
    if subcommands is not None and (len(argv) < 2 or argv[1] not in subcommands):
        return f"`{program}` is only allowed with: {', '.join(sorted(subcommands))}."
    if program == "make" and len(argv) != 2:
        return "`make` is only allowed as exactly `make test` or `make check`."
    if program == "ruff" and argv[1] == "format" and "--check" not in argv[2:]:
        return "`ruff format` is only allowed with --check (run_check never rewrites files)."
    if program == "npx" and argv[1] == "tsc" and "--noEmit" not in argv[2:]:
        return "`npx tsc` is only allowed with --noEmit."
    python = _python_for(project)
    if program in PYTHON_NAMES:
        argv[0] = python
    elif program == "pytest":
        argv = [python, "-m", "pytest", *argv[1:]]
    elif program in PYTHON_TOOLS:
        tool = find_python_tool(project, program)
        if tool is None:
            return _not_found(program, "in the project's .venv, Vega's environment or PATH")
        argv[0] = tool
    elif program == "npx":
        # npx would try to download a missing package; run the project's local copy directly.
        local = project / "node_modules" / ".bin" / argv[1]
        if not local.is_file():
            return _not_found(f"npx {argv[1]}", "in the project's node_modules (npm install it)")
        argv = [str(local), *argv[2:]]
    elif shutil.which(program) is None:
        return _not_found(program, "on PATH")
    return argv


def _script_argument(argv: list[str], project: Path) -> bool | str | None:
    """sandbox_toolchains: `node <script.js>` / `Rscript <script.R>`. True for a script file inside
    the project, an error message for a script path outside it or missing, None when the command
    is not a script run (node's other forms keep their own rules)."""
    program = Path(argv[0]).name
    suffixes = NODE_SCRIPTS if program == "node" else R_SCRIPTS if program == "Rscript" else ()
    if not suffixes or len(argv) < 2 or not argv[1].lower().endswith(suffixes):
        return None
    root = project.resolve()
    path = (root / argv[1]).resolve()
    if path != root and root not in path.parents:
        return f"`{program}` may only run a script inside the project, not {argv[1]}."
    if not path.is_file():
        return f"`{argv[1]}` does not exist in the project; write it first, then run it."
    return True


STOPPED_CHECK = "Stopped by you before the command finished."


class StoppedRun(Exception):
    """The user stopped the turn while a command was running; it was killed."""


def run_stoppable(argv: list[str] | str, *, cwd: Path, env: dict[str, str] | None = None,
                  timeout: float, stop: threading.Event | None = None,
                  shell: bool = False) -> subprocess.CompletedProcess[str]:
    """subprocess.run that also ends early when `stop` is set: the whole process group is killed at
    once (a test runner's children too), so Esc/Ctrl+C never waits out a long command.
    Raises subprocess.TimeoutExpired like subprocess.run, and StoppedRun when stopped."""
    process = subprocess.Popen(
        argv, cwd=cwd, env=env, shell=shell, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, start_new_session=(os.name != "nt"),
    )
    deadline = time.monotonic() + timeout
    outcome: str | None = None
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.2)
            return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            if stop is not None and stop.is_set():
                outcome = "stopped"
            elif time.monotonic() >= deadline:
                outcome = "timeout"
            if outcome:
                _kill_tree(process)
                process.communicate()
                if outcome == "stopped":
                    raise StoppedRun() from None
                raise subprocess.TimeoutExpired(argv, timeout) from None


def _kill_tree(process: subprocess.Popen[str]) -> None:
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()


def toolchain_env(private_tmp: Path) -> dict[str, str]:
    """sandbox_toolchains: writable caches inside the clone for toolchains that default to the
    home folder (2026-10-17: every `go test` failed with "failed to initialize build
    cache" because ~/Library/Caches/go-build is not writable in the sandbox). No network, so Go
    must not try to download modules or toolchains."""
    gopath = private_tmp / "gopath"
    return {"GOCACHE": str(private_tmp / "go-build"), "GOPATH": str(gopath),
            "GOMODCACHE": str(gopath / "pkg" / "mod"), "GOPROXY": "off", "GOTOOLCHAIN": "local",
            "GOTELEMETRY": "off"}


HEAD_SHARE = 0.4  # output_head_tail: the share of the limit kept from the start
CUT_LINE = "... [{:,} characters of output cut here] ..."


def clip_output(output: str, limit: int = OUTPUT_LIMIT) -> str:
    """Output over `limit` characters cut to about that: with output_head_tail its head (pytest's
    first failures and their assertion lines) and tail (the short summary and totals), whole lines
    on both sides of a CUT_LINE; else only the tail (2026-10-20: a long argparse
    traceback or a dozen failures pushed every failure header out of a tail-only cut). A larger limit
    (the deliverable pipeline's 400k, which reads a script's JSON at the end) keeps the tail."""
    if len(output) <= limit:
        return output
    if not features.on("output_head_tail") or limit > OUTPUT_LIMIT:
        return "... [earlier output truncated]\n" + output[-limit:]
    head = output[:int(limit * HEAD_SHARE)].rsplit("\n", 1)[0]
    tail = output[len(output) - (limit - len(head)):]
    tail = tail.split("\n", 1)[1] if "\n" in tail else tail
    cut = len(output) - len(head) - len(tail)
    return f"{head}\n{CUT_LINE.format(cut)}\n{tail}"


def _timeout_hint(command: str, project: Path, timeout: float, harness: bool) -> str:
    """targeted_tests: after a model's whole-suite pytest run on a large project times out, say it
    is a time limit and what to run instead (selfbuild 2026-10-21: the model chased "Check timed
    out after 60s." as if a test hung). Harness runs report their own timeouts."""
    if harness:
        return ""
    from .targeted_tests import run_check_hint  # imports tools, which imports this module

    return run_check_hint(command, project, timeout)


def run_check(command: str, project: Path, timeout: int = TIMEOUT,
              output_limit: int = OUTPUT_LIMIT, stop: threading.Event | None = None,
              harness: bool = False, private_home: bool = False) -> str:
    """Run one allowed command in a sandboxed clone of the project; `exit code: N` + output.
    `private_home`: HOME (and USERPROFILE) point at the clone, so a model-written script that
    expands `~` writes and reads inside it and never puts the user's home path into its output
    (a deliverable once held /Users/<name>/...)."""
    if not available():
        return "Sandboxed checks are unavailable on this system (macOS sandbox-exec not found)."
    argv = parse_command(command, project, harness=harness)
    if isinstance(argv, str):
        return argv
    workdir = Path(tempfile.mkdtemp(prefix="vega-check-")).resolve()
    try:
        _clone(project, workdir)
        private_tmp = workdir / ".vega-tmp"  # the only writable place besides the clone itself
        private_tmp.mkdir()
        inherited = os.environ
        if features.on("sandbox_env_scrub"):
            # Vega's own settings are not the project's: an eval's pinned VEGA_SEED /
            # VEGA_TEMPERATURE and its A/B VEGA_FEATURES_OFF reached Vega's own tests in the
            # selfbuild regression run and failed one in every attempt (2026-10-21).
            inherited = {k: v for k, v in os.environ.items() if not k.startswith("VEGA_")}
        env = {
            **inherited, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(private_tmp),
            "PYTHONPATH": os.pathsep.join([str(workdir), str(workdir / "src")]),
        }
        if features.on("sandbox_toolchains"):
            env.update(toolchain_env(private_tmp))
        if private_home:
            env.update(HOME=str(workdir), USERPROFILE=str(workdir))
        guard = private_home and features.on("sandbox_read_guard")
        profile = _profile(workdir, _read_roots(workdir, argv) if guard else None)
        try:
            result = run_stoppable([EXECUTABLE, "-p", profile, *argv],
                                   cwd=workdir, env=env, timeout=timeout, stop=stop)
        except subprocess.TimeoutExpired:
            return f"Check timed out after {timeout}s." + _timeout_hint(command, project, timeout,
                                                                         harness)
        except StoppedRun:
            return STOPPED_CHECK
        output = (result.stdout + ("\n" + result.stderr if result.stderr else "")).strip()
        output = output.replace(str(workdir), ".")
        if features.on("sandbox_path_scrub"):
            output = scrub_paths(output, workdir, project)
        output = clip_output(output, output_limit)
        return f"exit code: {result.returncode}\n{output or '(no output)'}"
    finally:
        _remove(workdir)


def _remove(workdir: Path) -> None:
    """Delete a clone, including read-only directories a toolchain made (Go's module cache is
    0555), by making them writable and retrying."""
    def retry(function, path, _exc) -> None:
        try:
            os.chmod(os.path.dirname(path), 0o700)
            os.chmod(path, 0o700)
            function(path)
        except OSError:
            pass

    handler = {"onexc": retry} if sys.version_info >= (3, 12) else {"onerror": retry}
    shutil.rmtree(workdir, **handler)
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
