"""Loop definitions: one TOML file per loop, validated on load.

User-wide loops live in `~/.vega/loops/*.toml` (server.STATE_DIR / "loops"), project loops in
`<project>/.vega/loops/*.toml`. A file that does not validate is reported with its path and the
reason; it never stops the other loops from loading.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .. import server

SKILLS = {"diff-bug": "diff-only bug check: a finding needs a test that fails at the commit "
                      "and passes at its parent"}
PERMISSIONS = {"read_project", "run_tests", "write_branch", "web"}
# Permissions each skill needs; a loop may not grant more than its skill can use (minimal grants).
SKILL_PERMISSIONS = {"diff-bug": {"read_project", "run_tests"}}
STATUS_KINDS = ("found", "approvals", "findings", "all_clear", "regressions")
TRIGGER_KINDS = ("commit", "schedule", "manual")
LATER_TRIGGERS = {"eval_run": "phase 3", "item": "phase 2", "test_failure": "phase 3",
                  "resource": "phase 3"}
FIELDS = {"name", "project", "trigger", "skill", "permissions", "budget", "status", "route"}
DEFAULT_MINUTES = 15
DEFAULT_RUNS_PER_DAY = 20
_UNITS = {"m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}
_EVERY = re.compile(r"^\s*(\d+)\s*([mhdw])\s*$")
_AT = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


class LoopDefinitionError(ValueError):
    """A loop file that does not validate; the message names the file and the field."""


@dataclass(frozen=True)
class Trigger:
    kind: str  # commit | schedule | manual
    branch: str = "main"  # commit triggers
    every_seconds: int = 0  # schedule triggers
    at: str | None = None  # schedule triggers: local time "HH:MM"

    def describe(self) -> str:
        """Short wording for the loops panel: "on commit", "nightly 02:00", "every 6h"."""
        if self.kind == "commit":
            return "on commit" if self.branch in {"main", "master"} else f"on commit to {self.branch}"
        if self.kind == "manual":
            return "manual"
        every = self.every_seconds
        names = {86400: "nightly" if self.at else "daily", 7 * 86400: "weekly"}
        if every in names:
            return f"{names[every]} {self.at}" if self.at else names[every]
        for unit, seconds in (("w", _UNITS["w"]), ("d", 86400), ("h", 3600), ("m", 60)):
            if every % seconds == 0:
                text = f"every {every // seconds}{unit}"
                return f"{text} at {self.at}" if self.at else text
        return f"every {every}s"

    def next_run(self, last_run: datetime | None, now: datetime) -> datetime | None:
        """When a schedule trigger is next due (local time); None for commit and manual."""
        if self.kind != "schedule":
            return None
        every = timedelta(seconds=self.every_seconds)
        if self.at is None:
            return now if last_run is None else last_run + every
        hour, minute = (int(part) for part in self.at.split(":"))
        # Earliest HH:MM after the anchor: a day before the interval ends (so a nightly loop that
        # ran at 02:41 is due again at 02:00 the next day), or now when it has never run.
        anchor = now if last_run is None else last_run + every - timedelta(days=1)
        candidate = anchor.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if last_run is None:
            return candidate if candidate >= now - timedelta(minutes=1) else candidate + timedelta(days=1)
        return candidate if candidate > anchor else candidate + timedelta(days=1)


@dataclass(frozen=True)
class LoopDef:
    name: str
    project: Path
    trigger: Trigger
    skill: str
    permissions: tuple[str, ...]
    minutes_per_run: float = DEFAULT_MINUTES
    runs_per_day: int = DEFAULT_RUNS_PER_DAY
    status: str = "found"
    route: tuple[dict[str, Any], ...] = ()  # stored for phase 2 (the router); not acted on yet
    source: Path | None = field(default=None, compare=False)

    @property
    def slug(self) -> str:
        return slugify(self.name)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "loop"


def user_dir() -> Path:
    return server.STATE_DIR / "loops"


def project_dir(project: Path) -> Path:
    return project / ".vega" / "loops"


def parse_every(text: str) -> int:
    match = _EVERY.match(text)
    if not match or int(match.group(1)) == 0:
        raise ValueError(f"every must look like 30m, 6h, 1d or 1w, got {text!r}")
    return int(match.group(1)) * _UNITS[match.group(2)]


def _trigger(raw: Any) -> Trigger:
    if not isinstance(raw, dict):
        raise TypeError('trigger must be a table, e.g. { on = "commit", branch = "main" }')
    if "every" in raw:
        unknown = set(raw) - {"every", "at"}
        if unknown:
            raise ValueError(f"schedule trigger has unknown keys: {', '.join(sorted(unknown))}")
        at = raw.get("at")
        if at is not None and (not isinstance(at, str) or not _AT.match(at)):
            raise ValueError(f'trigger.at must be a 24-hour "HH:MM", got {at!r}')
        if not isinstance(raw["every"], str):
            raise ValueError("trigger.every must be a string such as \"1d\"")
        every = parse_every(raw["every"])
        if at is not None and every < 86400:
            raise ValueError("trigger.at needs an interval of at least 1d")
        return Trigger("schedule", every_seconds=every, at=at)
    on = raw.get("on")
    if on in LATER_TRIGGERS:
        raise ValueError(f"trigger on = {on!r} is planned ({LATER_TRIGGERS[on]}), not built yet")
    if on == "commit":
        unknown = set(raw) - {"on", "branch"}
        if unknown:
            raise ValueError(f"commit trigger has unknown keys: {', '.join(sorted(unknown))}")
        branch = raw.get("branch", "main")
        if not isinstance(branch, str) or not branch.strip() or branch.startswith("-"):
            raise ValueError(f"trigger.branch must be a branch name, got {branch!r}")
        return Trigger("commit", branch=branch.strip())
    if on == "manual":
        if set(raw) - {"on"}:
            raise ValueError("manual trigger takes no other keys")
        return Trigger("manual")
    raise ValueError(
        'trigger must be { on = "commit", branch = "..." }, { every = "1d", at = "02:00" } or '
        f'{{ on = "manual" }}, got {raw!r}')


def parse(data: dict[str, Any], source: Path | None = None) -> LoopDef:
    """Validate one loop's table; raises LoopDefinitionError naming the file and the problem."""
    where = str(source) if source else "loop definition"

    def fail(message: str) -> LoopDefinitionError:
        return LoopDefinitionError(f"{where}: {message}")

    unknown = set(data) - FIELDS
    if unknown:
        raise fail(f"unknown field(s): {', '.join(sorted(unknown))} (known: {', '.join(sorted(FIELDS))})")
    for key in ("name", "project", "trigger", "skill", "permissions"):
        if key not in data:
            raise fail(f"missing required field {key!r}")
    name = data["name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 40:
        raise fail("name must be a non-empty string of at most 40 characters")
    if not isinstance(data["project"], str):
        raise fail("project must be a path string")
    project = Path(data["project"]).expanduser()
    if not project.is_absolute():
        raise fail(f"project must be an absolute path, got {data['project']!r}")
    if not project.is_dir():
        raise fail(f"project {project} does not exist")
    try:
        trigger = _trigger(data["trigger"])
    except (TypeError, ValueError) as exc:
        raise fail(str(exc)) from exc
    skill = data["skill"]
    if skill not in SKILLS:
        raise fail(f"unknown skill {skill!r} (known: {', '.join(sorted(SKILLS))})")
    permissions = data["permissions"]
    if not isinstance(permissions, list) or not all(isinstance(p, str) for p in permissions):
        raise fail("permissions must be a list of strings")
    bad = sorted(set(permissions) - PERMISSIONS)
    if bad:
        raise fail(f"unknown permission(s) {', '.join(bad)} (known: {', '.join(sorted(PERMISSIONS))})")
    wider = sorted(set(permissions) - SKILL_PERMISSIONS[skill])
    if wider:
        raise fail(f"skill {skill!r} does not use {', '.join(wider)}; grant only "
                   f"{', '.join(sorted(SKILL_PERMISSIONS[skill]))}")
    missing = sorted(SKILL_PERMISSIONS[skill] - set(permissions))
    if missing:
        raise fail(f"skill {skill!r} needs permission(s) {', '.join(missing)}")
    budget = data.get("budget", {})
    if not isinstance(budget, dict) or set(budget) - {"minutes_per_run", "runs_per_day"}:
        raise fail("budget must be { minutes_per_run = N, runs_per_day = N }")
    minutes = budget.get("minutes_per_run", DEFAULT_MINUTES)
    runs = budget.get("runs_per_day", DEFAULT_RUNS_PER_DAY)
    if isinstance(minutes, bool) or not isinstance(minutes, int | float) or not 0 < minutes <= 120:
        raise fail(f"budget.minutes_per_run must be a number from 1 to 120, got {minutes!r}")
    if isinstance(runs, bool) or not isinstance(runs, int) or not 0 < runs <= 500:
        raise fail(f"budget.runs_per_day must be a whole number from 1 to 500, got {runs!r}")
    status = data.get("status", "found")
    if status not in STATUS_KINDS:
        raise fail(f"status must be one of {', '.join(STATUS_KINDS)}, got {status!r}")
    route = data.get("route", [])
    if not isinstance(route, list) or not all(
            isinstance(rule, dict) and isinstance(rule.get("to"), str)
            and isinstance(rule.get("when", {}), dict) for rule in route):
        raise fail('route must be a list of { when = { ... }, to = "<loop or user>" }')
    return LoopDef(name=name.strip(), project=project.resolve(), trigger=trigger, skill=skill,
                   permissions=tuple(permissions), minutes_per_run=float(minutes),
                   runs_per_day=runs, status=status, route=tuple(route), source=source)


def load_file(path: Path) -> LoopDef:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise LoopDefinitionError(f"{path}: cannot read: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise LoopDefinitionError(f"{path}: not valid TOML: {exc}") from exc
    return parse(data, path)


def definition_dirs(projects: list[Path] | None = None) -> list[Path]:
    """User-wide folder first, then each project's (Vega's own repo only, for now)."""
    projects = [server.PROJECT_ROOT] if projects is None else projects
    return [user_dir(), *(project_dir(p) for p in projects)]


def load_all(projects: list[Path] | None = None) -> tuple[list[LoopDef], list[str]]:
    """Every valid loop and one message per file that failed. Names must be unique."""
    loops: list[LoopDef] = []
    errors: list[str] = []
    seen: dict[str, Path | None] = {}
    for folder in definition_dirs(projects):
        for path in sorted(folder.glob("*.toml")) if folder.is_dir() else []:
            try:
                loop = load_file(path)
            except LoopDefinitionError as exc:
                errors.append(str(exc))
                continue
            if loop.slug in seen:
                errors.append(f"{path}: a loop named {loop.name!r} is already defined in {seen[loop.slug]}")
                continue
            seen[loop.slug] = path
            loops.append(loop)
    return loops, errors


def find(name: str, projects: list[Path] | None = None) -> LoopDef:
    loops, _ = load_all(projects)
    wanted = slugify(name)
    for loop in loops:
        if loop.slug == wanted:
            return loop
    known = ", ".join(loop.name for loop in loops) or "none defined"
    raise LookupError(f"No loop named {name!r} (loops: {known}).")


BUG_WATCHER_TOML = """\
# Bug Watcher: checks each new commit on {branch}. Tests and ruff run at once; the diff-only model
# check runs only while the screen is locked and the machine is idle. Edit or delete this file to
# change it; `vega loops list` shows its status.
name = "Bug Watcher"
project = "{project}"
trigger = {{ on = "commit", branch = "{branch}" }}
skill = "diff-bug"
permissions = ["read_project", "run_tests"]
budget = {{ minutes_per_run = 15, runs_per_day = 20 }}
status = "found"
route = [{{ when = {{ evidence = "failing_test" }}, to = "Software Engineer" }}]
"""


def ensure_default(project: Path | None = None, branch: str = "main") -> Path | None:
    """Create the Bug Watcher definition for Vega's repo when no loop is defined anywhere.
    Returns the new file's path, or None when loops already exist."""
    project = (project or server.PROJECT_ROOT).resolve()
    if any(folder.is_dir() and any(folder.glob("*.toml")) for folder in definition_dirs([project])):
        return None
    path = user_dir() / "bug-watcher.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    escaped = str(project).replace("\\", "\\\\").replace('"', '\\"')
    path.write_text(BUG_WATCHER_TOML.format(project=escaped, branch=branch), encoding="utf-8")
    load_file(path)  # the default must validate; raise now rather than at the first tick
    return path
