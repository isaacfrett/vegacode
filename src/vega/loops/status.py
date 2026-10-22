"""Status lines and panel rows. The loops panel (and `vega loops list`) render `rows()`.

Status wording is fixed per kind (since 2026-10-14) so the panel stays consistent and
quiet: `{n} found`, `{n} approvals`, `new findings`, `all clear`, `{n} regressions`, plus
`paused`, `failing` and `not run yet` for every kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from . import inbox
from .definitions import LoopDef, load_all
from .idle import IdleProbe
from .state import LoopState, daemon_pid, is_paused, load_state, pid_alive

SYMBOLS = {"active": "●", "idle": "○", "paused": "‖", "failing": "!"}
# The status kinds whose count is something the user must act on (accent colour, tab badge);
# "findings" is information only.
NEEDS_YOU_KINDS = {"found", "approvals", "regressions", "all_clear"}


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def status_text(kind: str, pending: int, *, paused: bool = False, failing: bool = False,
                ran: bool = True) -> str:
    """The parenthesised status for a loop of this kind with `pending` open items."""
    if paused:
        return "paused"
    if failing:
        return "failing"
    if kind == "found":
        text = f"{pending} found"
    elif kind == "approvals":
        text = _plural(pending, "approval", "approvals")
    elif kind == "findings":
        text = "new findings"
    elif kind == "regressions":
        text = _plural(pending, "regression", "regressions")
    elif kind == "all_clear":
        text = f"{pending} found"
    else:
        raise ValueError(f"unknown status kind {kind!r}")
    if pending:
        return text
    return "all clear" if ran else "not run yet"


@dataclass(frozen=True)
class LoopRow:
    """One loop as the panel shows it."""

    name: str
    state: str  # active | idle (active, nothing pending) | paused | failing
    status: str  # the parenthesised text, e.g. "3 found"
    needs_you: int  # items awaiting the user (findings / approvals); counts toward the tab badge
    informational: bool  # True when the status is information only (dim), not a call to act
    trigger: str  # "on commit", "nightly 02:00"
    last_run: datetime | None
    next_run: datetime | None  # schedule triggers only
    running_for: float | None  # seconds the current job has been running; None when not running
    error: str  # the loop's own last job error (state "failing")
    project: str
    source: str  # the TOML file

    @property
    def symbol(self) -> str:
        return SYMBOLS[self.state]

    def when(self) -> str:
        """The right-hand column: "on commit · last 09:12" / "· running 4m" / "· next 02:00"."""
        parts = [self.trigger]
        if self.running_for is not None:
            parts.append(f"running {_duration(self.running_for)}")
        elif self.last_run is not None:
            parts.append(f"last {_clock(self.last_run)}")
        if self.next_run is not None and self.running_for is None and self.state != "paused":
            parts.append(f"next {_clock(self.next_run)}")
        return " · ".join(parts)


def _clock(moment: datetime) -> str:
    now = datetime.now()
    return moment.strftime("%H:%M" if moment.date() == now.date() else "%b %d %H:%M")


def _duration(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds < 60 else f"{seconds // 60:.0f}m"


def _parse(stamp: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(stamp) if stamp else None
    except ValueError:
        return None


def row(loop: LoopDef, state: LoopState | None = None, now: datetime | None = None) -> LoopRow:
    now = now or datetime.now()
    state = state or load_state(loop.name)
    paused = is_paused(loop.name)
    failing = bool(state.error)
    open_items = inbox.items(loop.name, open_only=True)
    waiting = sum(1 for item in open_items if inbox.state_of(item) in inbox.NEEDS_YOU)
    needs = waiting if loop.status in NEEDS_YOU_KINDS else 0
    pending = waiting if loop.status != "findings" else len(open_items)
    last = _parse(state.last_run)
    started = _parse(state.running_since)
    running = None
    if started is not None and state.running_pid and pid_alive(state.running_pid):
        running = max(0.0, (now - started).total_seconds())
    if paused:
        kind = "paused"
    elif failing:
        kind = "failing"
    elif pending or running is not None:
        kind = "active"
    else:
        kind = "idle"
    return LoopRow(
        name=loop.name, state=kind,
        status=status_text(loop.status, pending, paused=paused, failing=failing,
                           ran=last is not None),
        needs_you=needs, informational=loop.status == "findings" or not pending,
        trigger=loop.trigger.describe(), last_run=last,
        next_run=loop.trigger.next_run(last, now), running_for=running, error=state.error,
        project=str(loop.project), source=str(loop.source or ""),
    )


def rows(now: datetime | None = None) -> tuple[list[LoopRow], list[str]]:
    """Every defined loop's row, plus one message per definition file that failed to load."""
    loops, errors = load_all()
    return [row(loop, now=now) for loop in loops], errors


@dataclass(frozen=True)
class Header:
    """The panel's title line: the idle rule and whether model work may run right now."""

    daemon_running: bool
    daemon_pid: int | None
    screen_locked: bool
    model_allowed: bool
    reasons: tuple[str, ...]  # why model work is held back now (empty when allowed)
    needs_you: int  # sum over loops: the tab badge

    def text(self) -> str:
        lock = "screen locked ✓" if self.screen_locked else "screen unlocked"
        daemon = "" if self.daemon_running else " · daemon stopped"
        return f"idle-only · {lock}{daemon}"


def header(loop_rows: list[LoopRow], probe: IdleProbe | None = None) -> Header:
    from .state import ALL, paused_names

    check = (probe or IdleProbe()).check(paused=ALL in paused_names())
    pid = daemon_pid()
    return Header(daemon_running=pid is not None, daemon_pid=pid, screen_locked=check.screen_locked,
                  model_allowed=check.ok, reasons=tuple(check.reasons),
                  needs_you=sum(r.needs_you for r in loop_rows))
