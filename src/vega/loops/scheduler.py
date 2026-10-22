"""The loops scheduler and daemon.

Each tick, every loop is asked whether it has a job: a commit trigger polls
`git rev-parse <branch>` (no hooks are installed), a schedule trigger compares the clock with its
next run. Deterministic work runs whenever a job is due; model work only when the idle check
passes (screen locked, resources fine, on power, not paused). A commit whose model pass could not
run is kept in the loop's `pending_model` list and handled once the machine is idle.
One job per loop per tick, each bounded by the loop's minutes_per_run, so idle is re-checked
between jobs and between model steps.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .. import server
from . import bug_watcher
from .definitions import LoopDef, load_all
from .idle import IdleCheck, IdleProbe
from .state import (
    LoopState,
    claim_daemon,
    daemon_log,
    daemon_pid,
    is_paused,
    job_lock,
    load_state,
    release_daemon,
    save_state,
)

POLL_SECONDS = 30
MAX_NEW_COMMITS = 5  # a burst of commits: only the newest five are checked
MAX_PENDING = 10  # commits waiting for a model pass; older ones are dropped (and noted)
MODEL_SKILLS = {"diff-bug"}
REPO_ERROR = "cannot read the repository"


@dataclass(frozen=True)
class Job:
    commit: str | None
    deterministic: bool
    model: bool
    reason: str  # why this job (or why none), for logs and `vega loops run`


def head_of(loop: LoopDef) -> str:
    return bug_watcher.git(loop.project, "rev-parse", "--verify",
                           f"{loop.trigger.branch}^{{commit}}").strip()


def new_commits(loop: LoopDef, state: LoopState) -> tuple[str, list[str]]:
    """The branch head and the commits after `last_seen_commit`, oldest first (at most five).
    The first time a loop sees a branch nothing is new: the current head is the baseline."""
    head = head_of(loop)
    last = state.last_seen_commit
    if last is None or last == head:
        return head, []
    try:
        out = bug_watcher.git(loop.project, "rev-list", "--reverse",
                              f"--max-count={MAX_NEW_COMMITS}", f"{last}..{head}")
    except RuntimeError:  # history rewritten or the old commit is gone: check the head only
        return head, [head]
    return head, out.split()


def decide(
    loop: LoopDef, state: LoopState, *, now: datetime, commits: list[str], head: str | None,
    idle: IdleCheck, paused: bool,
) -> Job | None:
    """Pure scheduling decision for one loop this tick."""
    if paused:
        return None
    if state.runs_today(now) >= loop.runs_per_day:
        return None
    wants_model = loop.skill in MODEL_SKILLS
    if loop.trigger.kind == "commit" and commits:
        return Job(commits[0], deterministic=True, model=wants_model and idle.ok,
                   reason=f"new commit {commits[0][:8]}")
    if loop.trigger.kind == "schedule" and head is not None:
        due = loop.trigger.next_run(_parse(state.last_run), now)
        if due is not None and due <= now:
            return Job(head, deterministic=True, model=wants_model and idle.ok,
                       reason=f"scheduled ({loop.trigger.describe()})")
    if wants_model and state.pending_model and idle.ok:
        return Job(state.pending_model[0], deterministic=False, model=True,
                   reason=f"idle: model pass for {state.pending_model[0][:8]}")
    return None


def _parse(stamp: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(stamp) if stamp else None
    except ValueError:
        return None


def execute(
    loop: LoopDef, job: Job, *, probe: IdleProbe,
    agent_factory: bug_watcher.AgentFactory = bug_watcher.default_agent,
    now: Callable[[], datetime] = datetime.now,
) -> bug_watcher.JobReport | None:
    """Run one job under the loop's lock and record the outcome; None if the loop is busy."""
    if job.commit is None:
        return None
    with job_lock(loop.name) as acquired:
        if not acquired:
            return None
        state = load_state(loop.name)
        state.running_since, state.running_pid = now().isoformat(timespec="seconds"), os.getpid()
        save_state(loop.name, state)

        def gate() -> IdleCheck:
            return probe.check(paused=is_paused(loop.name))

        report = None
        try:
            report = bug_watcher.check_commit(
                loop, job.commit, deterministic=job.deterministic,
                gate=gate if job.model else None, agent_factory=agent_factory)
            if not job.model and loop.skill in MODEL_SKILLS and not report.error:
                report.model = "pending: runs when the machine is idle"
        finally:
            state = load_state(loop.name)
            state.running_since = state.running_pid = None
            finished = now()
            if report is not None:
                state.last_run = finished.isoformat(timespec="seconds")
                state.last_seconds = report.seconds
                state.last_result = report.summary()
                state.error = report.error
                state.count_run(finished)
                _update_pending(loop, state, job, report)
            else:
                state.error = "the job crashed; see the daemon log"
            save_state(loop.name, state)
    return report


def _update_pending(loop: LoopDef, state: LoopState, job: Job, report: bug_watcher.JobReport) -> None:
    commit = report.commit
    pending = [c for c in state.pending_model if not commit.startswith(c) and not c.startswith(commit)]
    if loop.skill in MODEL_SKILLS and not report.model_ran and not report.error:
        pending.append(commit)  # deferred (not idle) or not requested: run when idle
    if len(pending) > MAX_PENDING:
        report.notes.append(f"dropped model pass for {len(pending) - MAX_PENDING} older commit(s)")
        pending = pending[-MAX_PENDING:]
    state.pending_model = pending


class Daemon:
    """Polls the loops; `tick()` is one pass, `serve()` loops until stopped."""

    def __init__(
        self, *, probe: IdleProbe | None = None,
        agent_factory: bug_watcher.AgentFactory = bug_watcher.default_agent,
        poll_seconds: float = POLL_SECONDS, log: Callable[[str], None] = print,
        unload: Callable[[], object] = server.unload,
        projects: list[Path] | None = None,
    ):
        self.probe = probe or IdleProbe()
        self.agent_factory = agent_factory
        self.poll_seconds = poll_seconds
        self.log = log
        self.unload = unload
        self.projects = projects
        self.stopping = threading.Event()

    def tick(self, now: datetime | None = None) -> list[bug_watcher.JobReport]:
        now = now or datetime.now()
        loops, errors = load_all(self.projects)
        for error in errors:
            self.log(f"definition error: {error}")
        reports = []
        used_model = False
        for loop in loops:
            if self.stopping.is_set():
                break
            paused = is_paused(loop.name)
            state = load_state(loop.name)
            commits: list[str] = []
            head = None
            try:
                if loop.trigger.kind == "commit":
                    head, commits = new_commits(loop, state)
                    if state.last_seen_commit is None:
                        state.last_seen_commit = head
                        save_state(loop.name, state)
                        self.log(f"{loop.name}: watching {loop.trigger.branch} from {head[:8]}")
                elif loop.trigger.kind == "schedule":
                    head = bug_watcher.git(loop.project, "rev-parse", "HEAD").strip()
                if state.error.startswith(REPO_ERROR):
                    state.error = ""
                    save_state(loop.name, state)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                state.error = f"{REPO_ERROR}: {exc}"
                save_state(loop.name, state)
                continue
            idle = self.probe.check(paused=paused)  # per loop: a job before may have run long
            job = decide(loop, state, now=now, commits=commits, head=head, idle=idle, paused=paused)
            if job is None:
                continue
            if job.deterministic and loop.trigger.kind == "commit":
                state.last_seen_commit = job.commit  # handled (model pass may still be pending)
                save_state(loop.name, state)
            waits = (f" (model pass waits: {', '.join(idle.reasons)})"
                     if not job.model and loop.skill in MODEL_SKILLS else "")
            self.log(f"{loop.name}: {job.reason}{waits}")
            report = execute(loop, job, probe=self.probe, agent_factory=self.agent_factory)
            if report is not None:
                reports.append(report)
                used_model = used_model or job.model
                self.log(f"{loop.name}: {report.summary()}")
        if used_model:
            self.unload()  # free the model's memory once the loops go quiet
        return reports

    def serve(self) -> None:
        """Run until `vega loops stop` (SIGTERM), Ctrl-C or stop(); holds the pid file."""
        claim_daemon()
        if hasattr(os, "nice"):
            try:
                os.nice(10)  # tests and lint yield the CPU to the user's own work
            except OSError:
                pass
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, lambda *_: self.stopping.set())
        self.log(f"loops daemon started (pid {os.getpid()}), polling every {self.poll_seconds:g}s")
        try:
            while not self.stopping.is_set():
                try:
                    self.tick()
                except Exception as exc:  # noqa: BLE001 - keep serving; the error is logged
                    self.log(f"tick failed: {type(exc).__name__}: {exc}")
                self.stopping.wait(self.poll_seconds)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            release_daemon()
            self.log("loops daemon stopped")


def start_background() -> int:
    """Start the daemon as a detached process (`python -m vega.loops serve`); returns its pid."""
    running = daemon_pid()
    if running is not None:
        from .state import AlreadyRunning

        raise AlreadyRunning(f"The loops daemon is already running (pid {running}).")
    log = daemon_log()
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "VEGA_HOME": str(server.STATE_DIR)}
    kwargs: dict[str, object] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "vega.loops", "serve"], stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env, **kwargs)  # type: ignore[call-overload]
    return process.pid


def stop_daemon(timeout: float = 30.0) -> str:
    pid = daemon_pid()
    if pid is None:
        return "The loops daemon is not running."
    os.kill(pid, signal.SIGTERM)  # on Windows this terminates the process, which is the intent
    stopped = threading.Event()
    for _ in range(int(timeout * 4)):
        if daemon_pid() is None:
            stopped.set()
            break
        stopped.wait(0.25)
    if not stopped.is_set():
        return (f"Asked the loops daemon (pid {pid}) to stop; it is finishing its current step "
                "(at most one job's budget).")
    return f"Stopped the loops daemon (pid {pid})."


def run_now(
    loop: LoopDef, *, commit: str | None = None, probe: IdleProbe | None = None,
    agent_factory: bug_watcher.AgentFactory = bug_watcher.default_agent,
) -> tuple[Job, bug_watcher.JobReport | None]:
    """`vega loops run`: a job on the given commit (default: the branch head) right now.
    Deterministic checks always run; the model pass only if the idle check passes."""
    probe = probe or IdleProbe()
    check = probe.check(paused=is_paused(loop.name))
    target = commit or (head_of(loop) if loop.trigger.kind == "commit"
                        else bug_watcher.git(loop.project, "rev-parse", "HEAD").strip())
    job = Job(target, deterministic=True, model=loop.skill in MODEL_SKILLS and check.ok,
              reason="run now" + ("" if check.ok else f" (model pass waits: {', '.join(check.reasons)})"))
    return job, execute(loop, job, probe=probe, agent_factory=agent_factory)
