"""Autopilot: work toward one goal on a fresh Git branch, committing each round that passes a check.

The user reviews one branch at the end instead of approving each step. Autopilot never pushes,
never rewrites history, never touches other branches and never discards work: it refuses to start
on a dirty tree and commits anything left at the end as a clearly-labelled WIP commit.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import sandbox
from .agent import CodingAgent, EventHandler

DEFAULT_IDENTITY = ("Vega autopilot", "vega@localhost")
TRAILER = "Generated-by: Vega autopilot"
SUBJECT_LIMIT = 72
FEEDBACK_CHARS = 3000
CHECK_TIMEOUT = 600  # seconds; a whole test suite, not one quick check
NO_CHECK = {"", "none", "off"}
MAX_SLUG = 40

# Autopilot only ever runs these Git subcommands (fixed argv, never a shell). Everything here either
# reads state or adds a commit to the autopilot branch; nothing rewrites history or touches remotes.
_GIT_ALLOWED = {"rev-parse", "status", "branch", "checkout", "add", "commit", "log",
                "diff", "config"}

AgentFactory = Callable[[Path, "EventHandler | None"], CodingAgent]

ROUND_ONE = """(Vega autopilot) You are working autonomously toward the goal below. Nobody will
answer questions or approve commands until you finish, so decide for yourself and keep going.

Goal: {goal}

Start by writing a plan with update_plan, then work through it, keeping the plan current. {check}
Each time you give a final answer, Vega {after} and commits your changes to the branch `{branch}`.
When the whole goal is done and verified, mark every plan item done and end your answer with
DONE."""

CONTINUE = """(Vega autopilot) Round {round} finished; {status}. Continue working toward the
goal: {goal}
If everything is already done and verified, mark every plan item done and reply with a one-line
summary ending in DONE."""

FAILED = """(Vega autopilot) The check `{command}` failed after your changes:
{output}
Fix it."""


class AutopilotRefused(RuntimeError):
    """Raised before any change is made when autopilot cannot safely start."""


@dataclass
class CheckResult:
    command: str
    passed: bool
    exit_code: int | None  # None: the check could not run (timeout, sandbox error)
    output: str

    def describe(self) -> str:
        if self.exit_code is None:
            first = self.output.strip().split("\n", 1)[0]
            return f"`{self.command}` did not run: {first}"
        verdict = "passed" if self.passed else "failed"
        return f"`{self.command}` {verdict} (exit {self.exit_code})"


@dataclass
class Commit:
    hash: str
    message: str  # subject line
    round: int
    wip: bool = False


@dataclass
class AutopilotReport:
    goal: str
    project: Path
    branch: str
    original_branch: str  # a branch name, or the start commit when HEAD was detached
    start_commit: str
    check_command: str | None
    commits: list[Commit] = field(default_factory=list)
    rounds: int = 0
    max_rounds: int = 0
    stop_reason: str = ""
    completed: bool = False
    baseline_check: CheckResult | None = None
    final_check: CheckResult | None = None
    diffstat: str = ""
    plan: list[dict[str, str]] = field(default_factory=list)
    wall_seconds: float = 0.0
    wip: bool = False
    last_answer: str = ""

    @property
    def ok(self) -> bool:
        """The goal was reported done, the last check passed and nothing was left as WIP."""
        passed = self.final_check is None or self.final_check.passed
        return self.completed and passed and not self.wip

    @property
    def review_commands(self) -> list[tuple[str, str]]:
        start = self.start_commit[:12]
        return [
            ("review", f"git diff {start}..{self.branch}"),
            ("go back", f"git checkout {self.original_branch}"),
            ("discard", f"git checkout {self.original_branch} && git branch -D {self.branch}"),
        ]


# -- git -----------------------------------------------------------------------------------------

def _git(
    project: Path, *args: str, identity: list[str] | None = None,
) -> subprocess.CompletedProcess:
    if not args or args[0] not in _GIT_ALLOWED:
        raise ValueError(f"autopilot does not run `git {' '.join(args[:1])}`")
    argv = ["git", *(identity or []), *args]
    return subprocess.run(argv, cwd=project, capture_output=True, text=True, check=False)


def _git_out(project: Path, *args: str) -> str:
    result = _git(project, *args)
    if result.returncode:
        raise AutopilotRefused(
            f"`git {' '.join(args)}` failed: {(result.stderr or result.stdout).strip()}"
        )
    return result.stdout.strip()


def _identity(project: Path) -> list[str]:
    """`-c` overrides for whichever of user.name / user.email the repository does not configure."""
    overrides: list[str] = []
    for key, value in zip(("user.name", "user.email"), DEFAULT_IDENTITY):
        if not _git(project, "config", "--get", key).stdout.strip():
            overrides += ["-c", f"{key}={value}"]
    return overrides


def _status_entries(project: Path) -> list[str]:
    """Paths with staged, unstaged or untracked changes (ignored files excluded)."""
    raw = _git(project, "status", "--porcelain", "-z", "--untracked-files=all").stdout
    records = raw.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if len(record) < 4:
            continue
        paths.append(record[3:])
        if record[0] in "RC":  # renames and copies carry the original path as the next record
            index += 1
    return paths


def _fingerprint(project: Path) -> dict[str, str]:
    """Each dirty path with a hash of its current content, to tell whether a round changed files."""
    prints: dict[str, str] = {}
    for path in _status_entries(project):
        target = project / path
        try:
            prints[path] = hashlib.sha1(target.read_bytes()).hexdigest()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            prints[path] = "missing"
    return prints


def _in_progress_operation(project: Path) -> str | None:
    markers = {
        "MERGE_HEAD": "a merge", "rebase-merge": "a rebase", "rebase-apply": "a rebase or am",
        "CHERRY_PICK_HEAD": "a cherry-pick", "REVERT_HEAD": "a revert", "BISECT_LOG": "a bisect",
    }
    for marker, label in markers.items():
        path = Path(_git_out(project, "rev-parse", "--git-path", marker))
        if not path.is_absolute():
            path = project / path
        if path.exists():
            return label
    return None


@dataclass
class _Start:
    root: Path
    original_branch: str
    start_commit: str


def check_preconditions(project: Path) -> _Start:
    """Refuse (AutopilotRefused) unless project is a clean Git work tree with a commit to start
    from."""
    project = project.expanduser().resolve()
    if not project.is_dir():
        raise AutopilotRefused(f"Not a directory: {project}")
    inside = _git(project, "rev-parse", "--is-inside-work-tree")
    if inside.returncode or inside.stdout.strip() != "true":
        raise AutopilotRefused(
            f"{project} is not inside a Git repository. Autopilot works on a new branch so you can "
            "review or discard everything at the end; run `git init` and commit first."
        )
    root = Path(_git_out(project, "rev-parse", "--show-toplevel")).resolve()
    head = _git(project, "rev-parse", "--verify", "--quiet", "HEAD")
    if head.returncode:
        raise AutopilotRefused("The repository has no commits yet; make an initial commit first.")
    operation = _in_progress_operation(root)
    if operation:
        raise AutopilotRefused(f"The repository is in the middle of {operation}; finish it first.")
    dirty = _status_entries(root)
    if dirty:
        shown = ", ".join(dirty[:5]) + (f" and {len(dirty) - 5} more" if len(dirty) > 5 else "")
        raise AutopilotRefused(
            f"The working tree has uncommitted or untracked changes ({shown}). Autopilot never "
            "stashes or discards your work: commit them (or add them to .gitignore) first."
        )
    branch = _git(root, "branch", "--show-current").stdout.strip()
    return _Start(root, branch or head.stdout.strip(), head.stdout.strip())


def slugify(goal: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")
    if len(slug) > MAX_SLUG:
        slug = slug[:MAX_SLUG].rsplit("-", 1)[0] or slug[:MAX_SLUG]
    return slug.strip("-") or "goal"


def default_branch(goal: str, now: datetime | None = None) -> str:
    return f"vega/{slugify(goal)}-{(now or datetime.now()).strftime('%Y%m%d-%H%M')}"


def _branch_exists(root: Path, name: str) -> bool:
    return _git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}").returncode == 0


# -- check ---------------------------------------------------------------------------------------

def detect_check(project: Path) -> str | None:
    """A sensible default check: pytest for a Python test setup, else `make test`, else none."""
    pytest_config = (project / "pytest.ini").is_file() or (project / "conftest.py").is_file()
    for name, marker in (("pyproject.toml", "[tool.pytest"), ("setup.cfg", "[tool:pytest]"),
                         ("tox.ini", "[pytest]")):
        path = project / name
        if path.is_file() and marker in path.read_text(errors="replace"):
            pytest_config = True
    if (project / "tests").is_dir() or pytest_config:
        return "python -m pytest -q"
    makefile = project / "Makefile"
    text = makefile.read_text(errors="replace") if makefile.is_file() else ""
    if re.search(r"^test\s*:", text, re.MULTILINE):
        return "make test"
    return None


def _resolve_check(project: Path, check: str | None) -> str | None:
    if check is not None and check.strip().lower() in NO_CHECK:
        return None
    command = check.strip() if check is not None else detect_check(project)
    if command is None:
        return None
    if not sandbox.available():
        raise AutopilotRefused(
            f"The check `{command}` needs the macOS sandbox ({sandbox.EXECUTABLE}), which is not "
            "available. "
            "Pass --check none to run without a check."
        )
    argv = sandbox.parse_command(command, project)
    if isinstance(argv, str):
        raise AutopilotRefused(f"The check `{command}` cannot run in the sandbox: {argv}")
    return command


def run_check(command: str, project: Path) -> CheckResult:
    output = sandbox.run_check(command, project, timeout=CHECK_TIMEOUT)
    match = re.match(r"exit code: (-?\d+)\n?", output)
    if not match:
        return CheckResult(command, False, None, output)
    code = int(match.group(1))
    return CheckResult(command, code == 0, code, output[match.end():])


# -- messages ------------------------------------------------------------------------------------

_DONE = re.compile(r"\bDONE\b")
_DONE_WORDS = re.compile(r"\b(done|complete|completed|finished)\b", re.IGNORECASE)


def says_done(answer: str) -> bool:
    return bool(_DONE.search(answer or "") or _DONE_WORDS.search(answer or ""))


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def summarize(answer: str, fallback: str, limit: int = SUBJECT_LIMIT - len("vega: ")) -> str:
    """A one-line commit summary from the agent's final answer (first line with some substance)."""
    for line in (answer or "").splitlines():
        line = re.sub(r"`|\*\*|^\s*#+\s*|^\s*>\s*", "", line)
        line = re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", line)
        line = _DONE.sub("", line)
        line = re.sub(r"\s+", " ", line).strip(" .:;-")
        if len(line.split()) >= 3:
            break
    else:
        line = re.sub(r"\s+", " ", fallback).strip()
    return _clip(line, limit)


def _tail(text: str, limit: int = FEEDBACK_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else "... [earlier output truncated]\n" + text[-limit:]


# -- the run -------------------------------------------------------------------------------------

def _default_factory(project: Path, on_event: EventHandler | None) -> CodingAgent:
    return CodingAgent(
        project, allow_commands=False, review=False, stream=on_event is not None, on_event=on_event,
        planning="on",  # autopilot works toward multi-step goals
    )


class _Runner:
    def __init__(self, report: AutopilotReport, root: Path, on_event: EventHandler | None):
        self.report = report
        self.root = root
        self.on_event = on_event
        self.identity = _identity(root)

    def emit(self, event: str, message: str, **data: Any) -> None:
        if self.on_event is not None:
            self.on_event("autopilot", {"event": event, "message": message, **data})

    def commit(
        self, subject: str, body: str, round_number: int, wip: bool = False,
    ) -> Commit | None:
        added = _git(self.root, "add", "-A")
        if not added.returncode:
            from .gitignore import unstage_sensitive

            held = unstage_sensitive(self.root)  # `add -A` would sweep up an unignored .env or key
            if held:
                self.emit("sensitive_skipped",
                          "autopilot: left out of the commit (likely secrets): " + ", ".join(held),
                          paths=held)
        result = added if added.returncode else _git(
            self.root, "commit", "-q", "-m", subject, "-m", body, "-m", TRAILER,
            identity=self.identity,
        )
        if result.returncode:
            error = (result.stderr or result.stdout).strip()
            self.emit("commit_failed", f"autopilot: commit failed: {error}", error=error)
            return None
        sha = _git_out(self.root, "rev-parse", "HEAD")
        commit = Commit(sha, subject, round_number, wip)
        self.report.commits.append(commit)
        self.emit("commit", f"autopilot: committed {sha[:8]} {subject}", hash=sha, subject=subject)
        return commit


def _body(round_number: int, check: CheckResult | None) -> str:
    if check is None:
        return f"Autopilot round {round_number}. Check: none configured."
    return f"Autopilot round {round_number}. Check: {check.describe()}."


def run_goal(
    project: Path,
    goal: str,
    *,
    branch: str | None = None,
    max_rounds: int = 8,
    check: str | None = None,
    agent_factory: AgentFactory | None = None,
    on_event: EventHandler | None = None,
) -> AutopilotReport:
    """Work toward goal on a new branch; see the module docstring. Raises AutopilotRefused before
    changing anything when the repository is not in a safe state to start.

    check: a run_check command; None detects a default; "none" disables the check.
    agent_factory(project, on_event) builds the agent (tests pass one with a fake model client).
    """
    started = time.monotonic()
    goal = goal.strip()
    if not goal:
        raise AutopilotRefused("Give autopilot a goal.")
    if max_rounds < 1:
        raise AutopilotRefused("max_rounds must be at least 1.")
    start = check_preconditions(project)
    root = start.root
    command = _resolve_check(root, check)
    name = branch or default_branch(goal)
    if _branch_exists(root, name):
        raise AutopilotRefused(f"Branch `{name}` already exists; pick another with --branch.")
    created = _git(root, "checkout", "-q", "-b", name)
    if created.returncode:
        raise AutopilotRefused(f"Could not create branch `{name}`: {created.stderr.strip()}")

    report = AutopilotReport(
        goal=goal, project=root, branch=name, original_branch=start.original_branch,
        start_commit=start.start_commit, check_command=command, max_rounds=max_rounds,
    )
    runner = _Runner(report, root, on_event)
    runner.emit(
        "start", f"autopilot: on branch {name} (from {start.original_branch} @ "
        f"{start.start_commit[:8]}); check: {command or 'none'}; up to {max_rounds} rounds",
        branch=name, check=command,
    )
    if command:
        report.baseline_check = run_check(command, root)
        runner.emit("baseline", f"autopilot: before any change, {report.baseline_check.describe()}",
                    passed=report.baseline_check.passed)

    agent = (agent_factory or _default_factory)(root, on_event)
    agent.allow_commands = False  # autopilot must never stop to ask the user for anything
    agent.review = False
    agent.planning = "on"  # goals are multi-step
    # autopilot runs the check after every round and feeds the result back itself
    agent.finish_checks = False
    # its messages are harness prompts for the plan-driven agent loop (plan, DONE, "the check
    # failed, fix it"): never the clarify gate or a pipeline, which would read them as the task
    agent.route_requests = False
    try:
        _rounds(runner, agent, goal, command)
    except KeyboardInterrupt:
        report.stop_reason = "interrupted"
        runner.emit("stop", "autopilot: interrupted")
    finally:
        _finish(runner, agent, started)
    return report


def _round_one(goal: str, command: str | None, branch: str, baseline: CheckResult | None) -> str:
    if command:
        check = f"Verify your work with run_check `{command}`."
        if baseline is not None and not baseline.passed:
            check += (
                f" Note: this check already fails before any change "
                f"({baseline.describe()}); it must pass before anything is committed."
            )
        after = f"runs `{command}`, and if it passes,"
    else:
        check, after = "There is no project check, so verify your work by re-reading it.", "then"
    return ROUND_ONE.format(goal=goal, check=check, after=after, branch=branch)


def _rounds(runner: _Runner, agent: CodingAgent, goal: str, command: str | None) -> None:
    report, root = runner.report, runner.root
    message = _round_one(goal, command, report.branch, report.baseline_check)
    before = _fingerprint(root)
    idle = 0
    for number in range(1, report.max_rounds + 1):
        report.rounds = number
        runner.emit("round_start", f"autopilot: round {number}/{report.max_rounds}", round=number)
        answer = agent.run(message)
        report.last_answer = answer
        if agent.plan:  # the agent clears a finished plan when the next round starts; keep a copy
            report.plan = [dict(item) for item in agent.plan]
        if answer.startswith("Model request failed"):
            error = answer.splitlines()[0].split(": ", 1)[-1]
            report.stop_reason = f"model request failed: {error}"
            return
        after = _fingerprint(root)
        changed = after != before
        result = run_check(command, root) if command else None
        report.final_check = result
        passed = result is None or result.passed
        if result is not None:
            runner.emit("check", f"autopilot: {result.describe()}", passed=passed, round=number)
        committed = changed and passed and runner.commit(
            f"vega: {summarize(answer, goal)}", _body(number, result), number,
        ) is not None
        idle = 0 if changed else idle + 1
        before = _fingerprint(root)
        unfinished = [item for item in agent.plan if item.get("status") != "done"]
        state = "changed files" if changed else "no file changes"
        runner.emit("round_end", f"autopilot: round {number} done ({state}, "
                    f"{len(unfinished)} plan item(s) open)", round=number, changed=changed)
        if passed and not unfinished and (not changed or says_done(answer)):
            report.completed = True
            report.stop_reason = "goal complete (plan finished, check passed)"
            return
        if idle >= 2:
            report.stop_reason = "stuck (two rounds in a row changed no files)"
            return
        if not passed:
            message = FAILED.format(command=command, output=_tail(result.output if result else ""))
        else:
            status = "no files changed"
            if changed:
                status = "your changes were committed" if committed else (
                    "committing your changes failed, so they are still uncommitted")
            message = CONTINUE.format(round=number, status=status, goal=goal)
    report.stop_reason = f"reached max rounds ({report.max_rounds})"


def _finish(runner: _Runner, agent: CodingAgent, started: float) -> None:
    report, root = runner.report, runner.root
    if _status_entries(root):
        check = report.final_check
        subject = f"vega: WIP (check failing) {summarize(report.last_answer, report.goal)}"
        if check is None or check.passed:
            subject = f"vega: WIP {summarize(report.last_answer, report.goal)}"
        subject = _clip(subject, SUBJECT_LIMIT)
        body = _body(report.rounds, check) + " Left uncommitted when autopilot stopped "
        body += f"({report.stop_reason or 'error'}); committed so nothing is lost."
        if check is not None and not check.passed:
            body += "\n\nCheck output (tail):\n" + _tail(check.output, 1500)
        if runner.commit(subject, body, report.rounds, wip=True):
            report.wip = True
    if agent.plan:
        report.plan = [dict(item) for item in agent.plan]
    report.diffstat = _git(root, "diff", "--stat", f"{report.start_commit}..HEAD").stdout.rstrip()
    report.wall_seconds = time.monotonic() - started
    runner.emit("stop", f"autopilot: stopped: {report.stop_reason}", reason=report.stop_reason)


# -- presentation --------------------------------------------------------------------------------

_PLAN_MARKS = {"done": "✓", "in_progress": "▸", "pending": "·"}


def _duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def format_report(report: AutopilotReport) -> str:
    verdict = "goal complete" if report.ok else "needs attention"
    lines = [f"Vega autopilot: {verdict}", ""]

    def row(label: str, value: str) -> None:
        lines.append(f"  {label:<9} {value}".rstrip())

    row("goal", report.goal)
    row("branch", f"{report.branch} (from {report.original_branch} @ {report.start_commit[:8]})")
    row("rounds", f"{report.rounds} of {report.max_rounds}; stopped: {report.stop_reason}")
    if report.check_command is None:
        row("check", "none")
    else:
        if report.baseline_check is not None:
            row("check", f"before: {report.baseline_check.describe()}")
            row("", f"after:  {report.final_check.describe() if report.final_check else 'not run'}")
        else:
            row("check", report.final_check.describe() if report.final_check else "not run")
    row("commits", str(len(report.commits)) if report.commits else "none")
    for commit in report.commits:
        lines.append(f"    {commit.hash[:8]} {commit.message}")
    if report.wip:
        lines.append("    (the last commit is WIP: its check failed; nothing was left uncommitted)")
    if report.plan:
        row("plan", "")
        for item in report.plan:
            mark = _PLAN_MARKS.get(item.get("status", ""), "·")
            lines.append(f"    {mark} {item.get('text', '')}")
    if report.diffstat:
        row("changes", "")
        lines.extend(f"   {line}" for line in report.diffstat.splitlines())
    else:
        row("changes", "none")
    row("time", _duration(report.wall_seconds))
    lines.append("")
    for label, command in report.review_commands:
        row(label, command)
    return "\n".join(lines)


def autopilot_command(
    goal: str, project: Path, branch: str | None, max_rounds: int, check: str | None,
) -> None:
    """Everything `vega auto` needs: preconditions, server, live progress, report, exit code."""
    import typer
    from rich.console import Console

    from . import server
    from .ui import TerminalRenderer

    console = Console()
    try:
        check_preconditions(project)  # refuse before starting a model server for nothing
    except AutopilotRefused as exc:
        console.print(f"[red]Autopilot refused:[/red] {exc}", highlight=False)
        raise typer.Exit(1) from exc
    try:
        server.ensure_running()
    except RuntimeError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
    try:
        report = run_goal(
            project, goal, branch=branch, max_rounds=max_rounds, check=check,
            on_event=TerminalRenderer(console),
        )
    except AutopilotRefused as exc:
        console.print(f"[red]Autopilot refused:[/red] {exc}", highlight=False)
        raise typer.Exit(1) from exc
    console.print()
    console.print(format_report(report), markup=False, highlight=False)
    raise typer.Exit(0 if report.ok else 1)
