"""Runtime state of loops on disk: per-loop state files, pause flags, the daemon pid file and
per-loop job locks. Everything lives under server.STATE_DIR / "loops" so tests can isolate it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import server
from .definitions import slugify

ALL = "*"  # pause flag name meaning every loop


def root() -> Path:
    return server.STATE_DIR / "loops"


def _state_dir() -> Path:
    return root() / "state"


def pid_file() -> Path:
    return root() / "daemon.pid"


def daemon_log() -> Path:
    return root() / "daemon.log"


def write_json(path: Path, data: Any) -> None:
    """Atomic write: readers (the panel, `vega loops list`) never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@dataclass
class LoopState:
    last_run: str | None = None  # ISO local time a job last finished
    last_seconds: float | None = None
    last_result: str = ""  # one line about the last job
    last_seen_commit: str | None = None  # commit triggers: newest commit already handled
    pending_model: list[str] = field(default_factory=list)  # commits whose model pass waits for idle
    runs: dict[str, int] = field(default_factory=dict)  # jobs per local date (runs_per_day budget)
    running_since: str | None = None
    running_pid: int | None = None
    error: str = ""  # non-empty when the loop's own last job errored (the panel shows "failing")

    def runs_today(self, now: datetime) -> int:
        return self.runs.get(now.date().isoformat(), 0)

    def count_run(self, now: datetime) -> None:
        today = now.date().isoformat()
        self.runs = {today: self.runs.get(today, 0) + 1}  # only today's count matters


def load_state(name: str) -> LoopState:
    data = read_json(_state_dir() / f"{slugify(name)}.json")
    if not isinstance(data, dict):
        return LoopState()
    known = {key: value for key, value in data.items() if key in LoopState.__dataclass_fields__}
    try:
        return LoopState(**known)
    except TypeError:
        return LoopState()


def save_state(name: str, state: LoopState) -> None:
    write_json(_state_dir() / f"{slugify(name)}.json", asdict(state))


# -- pause -----------------------------------------------------------------------------------------

def _pause_file() -> Path:
    return root() / "paused.json"


def paused_names() -> set[str]:
    data = read_json(_pause_file())
    return set(data) if isinstance(data, list) else set()


def set_paused(name: str | None, paused: bool) -> None:
    """Pause or resume one loop, or every loop when name is None (resume None clears all)."""
    names = paused_names()
    key = ALL if name is None else slugify(name)
    if paused:
        names.add(key)
    elif name is None:
        names.clear()
    else:
        names.discard(key)
    write_json(_pause_file(), sorted(names))


def is_paused(name: str) -> bool:
    names = paused_names()
    return ALL in names or slugify(name) in names


# -- processes -------------------------------------------------------------------------------------

def pid_alive(pid: int) -> bool:
    """True when a process with this pid exists. Never signals it (os.kill on Windows would
    terminate the process, so Windows asks OpenProcess instead)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _looks_like_daemon(pid: int) -> bool:
    """Guard against a recycled pid: the process must be a Python running the loops daemon.
    `-ww`: without a terminal, Linux ps cuts the command at 80 columns, and a long interpreter
    path then hides the `loops` argument."""
    if sys.platform == "win32":
        return True
    try:
        command = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "command="], capture_output=True,
                                 text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return True  # cannot tell; trust the pid file rather than start a second daemon
    return "loops" in command


def daemon_pid() -> int | None:
    """Pid of the running daemon, or None (a stale pid file is removed)."""
    data = read_json(pid_file())
    pid = data.get("pid") if isinstance(data, dict) else None
    if not isinstance(pid, int):
        return None
    if pid == os.getpid():
        return pid
    if pid_alive(pid) and _looks_like_daemon(pid):
        return pid
    pid_file().unlink(missing_ok=True)
    return None


class AlreadyRunning(RuntimeError):
    pass


def claim_daemon() -> None:
    """Write the pid file for this process; refuse when another daemon is alive."""
    running = daemon_pid()
    if running is not None and running != os.getpid():
        raise AlreadyRunning(f"The loops daemon is already running (pid {running}). "
                             "Stop it first with `vega loops stop`.")
    root().mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(pid_file(), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise AlreadyRunning("Another loops daemon started at the same moment; not starting a "
                             "second one.") from None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started": datetime.now().isoformat(timespec="seconds")},
                  handle)


def release_daemon() -> None:
    data = read_json(pid_file())
    if isinstance(data, dict) and data.get("pid") == os.getpid():
        pid_file().unlink(missing_ok=True)


@contextmanager
def job_lock(name: str) -> Iterator[bool]:
    """Hold the loop's job lock; yields False when another live process holds it (the job must
    not run twice at once, e.g. `vega loops run` while the daemon runs the same loop)."""
    path = _state_dir() / f"{slugify(name)}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = read_json(path)
            pid = holder.get("pid") if isinstance(holder, dict) else None
            if isinstance(pid, int) and pid_alive(pid):
                yield False
                return
            path.unlink(missing_ok=True)  # stale: its process is gone
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "at": time.time()}, handle)
        try:
            yield True
        finally:
            path.unlink(missing_ok=True)
        return
    yield False
