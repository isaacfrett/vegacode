"""When may a loop use the model? Only while the machine is idle (since 2026-10-14):
the screen is locked, memory pressure is normal, swap is stable, disk has ≥ 12 GB free, there is
no thermal limit, the machine is not on battery and loops are not paused.

Each probe sits behind one function with a safe fallback: where a probe cannot answer, the answer
is the one that keeps model work off (screen "not locked"). Deterministic checks never ask.
"""

from __future__ import annotations

import os
import plistlib
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .. import resources

DISK_MIN_GB = 12.0
# Other Vega commands that keep the model server busy for a long time; a loop never competes with
# them for the one model (an eval's timings and results must not be disturbed).
_MODEL_COMMANDS = re.compile(r"[/\\\s]vega(?:\.exe)?\s+(?:eval|lab\s+run|auto|train)\b")


def _run(argv: list[str]) -> bytes:
    try:
        return subprocess.run(argv, capture_output=True, timeout=10, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return b""


def parse_ioreg_root(data: bytes, uid: int | None = None) -> bool:
    """Locked state from `ioreg -n Root -d1 -a` (a plist): macOS puts `CGSSessionScreenIsLocked
    = true` in the user's IOConsoleUsers session entry while the lock screen is up and removes
    it on unlock. (Checked on macOS 26, 2026-10-14: the key was absent while unlocked and true
    while locked; the top-level `IOConsoleLocked` stayed false while locked, so it is not used.)
    Unreadable output counts as not locked."""
    try:
        root = plistlib.loads(data)
    except Exception:  # noqa: BLE001 - any parse failure is "cannot tell", i.e. not locked
        return False
    if isinstance(root, list):
        root = root[0] if root else {}
    if not isinstance(root, dict):
        return False
    for session in root.get("IOConsoleUsers") or []:
        if not isinstance(session, dict):
            continue
        if uid is not None and session.get("kCGSSessionUserIDKey") not in (None, uid):
            continue  # another user's session (fast user switching)
        if session.get("CGSSessionScreenIsLocked") is True:
            return True
    return False


def screen_locked() -> bool:
    """True only when the screen is known to be locked. macOS reads the IORegistry root (no extra
    dependencies); every other OS returns False until it has a probe (phase 3)."""
    if sys.platform != "darwin":
        return False
    return parse_ioreg_root(_run(["ioreg", "-n", "Root", "-d1", "-a"]), os.getuid())


def on_battery() -> bool:
    """True when running on battery power. Unknown counts as False (desktops have no battery);
    the screen-locked rule stays the gate."""
    if sys.platform == "darwin":
        return b"'Battery Power'" in _run(["pmset", "-g", "batt"])
    if sys.platform == "win32":
        try:
            import ctypes

            class PowerStatus(ctypes.Structure):
                _fields_ = [("ac", ctypes.c_byte), ("flag", ctypes.c_byte),
                            ("percent", ctypes.c_byte), ("saver", ctypes.c_byte),
                            ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]

            status = PowerStatus()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):  # type: ignore[attr-defined]
                return status.ac == 0
        except (OSError, AttributeError):
            return False
        return False
    supplies = Path("/sys/class/power_supply")
    try:
        mains = [p for p in supplies.iterdir() if (p / "type").read_text().strip() == "Mains"]
        return bool(mains) and all((p / "online").read_text().strip() == "0" for p in mains)
    except OSError:
        return False


def other_model_work() -> str:
    """The command line of another running Vega eval / lab run / autopilot / training, or ""."""
    if sys.platform == "win32":
        return ""  # no probe yet; the screen-locked gate is already False on Windows
    listing = _run(["ps", "-axo", "pid=,command="]).decode("utf-8", "replace")
    for line in listing.splitlines():
        pid, _, command = line.strip().partition(" ")
        match = _MODEL_COMMANDS.search(command)
        if pid.isdigit() and int(pid) != os.getpid() and match:
            return command[match.start():].strip(" /\\")[:60]
    return ""


@dataclass
class IdleCheck:
    ok: bool
    screen_locked: bool
    reasons: list[str] = field(default_factory=list)  # why model work may not run now


class IdleProbe:
    """Holds the previous resource snapshot so swap growth can be judged between checks."""

    def __init__(
        self, *, locked: Callable[[], bool] = screen_locked,
        battery: Callable[[], bool] = on_battery,
        busy: Callable[[], str] = other_model_work,
        snapshot: Callable[[], resources.Snapshot] = resources.snapshot,
        limits: resources.Limits | None = None,
    ):
        self.locked, self.battery, self.busy, self.snapshot = locked, battery, busy, snapshot
        self.limits = limits or resources.Limits()
        self.last_swap_mb: float | None = None

    def check(self, *, paused: bool = False) -> IdleCheck:
        reasons = []
        if paused:
            reasons.append("paused")
        locked = self.locked()
        if not locked:
            reasons.append("screen not locked")
        if self.battery():
            reasons.append("on battery")
        if locked and not paused:  # the costlier probes only matter once the cheap gates pass
            other = self.busy()
            if other:
                reasons.append(f"model busy with another Vega job ({other})")
            reasons += self.resource_problems()
        return IdleCheck(ok=not reasons, screen_locked=locked, reasons=reasons)

    def resource_problems(self) -> list[str]:
        now = self.snapshot()
        problems = []
        if now.pressure_level is not None and now.pressure_level > 1:
            problems.append(f"memory pressure level {now.pressure_level} (1 is normal)")
        if now.swap_used_mb is not None:
            if self.last_swap_mb is not None and \
                    now.swap_used_mb - self.last_swap_mb > self.limits.max_swap_growth_mb:
                problems.append(f"swap grew {now.swap_used_mb - self.last_swap_mb:.0f} MB")
            self.last_swap_mb = now.swap_used_mb
        need = max(DISK_MIN_GB, self.limits.min_disk_gb)
        if now.disk_free_gb < need:
            problems.append(f"free disk {now.disk_free_gb:.1f} GB < {need:.0f} GB")
        if now.thermal_warning:
            problems.append(f"thermal warning ({now.thermal_warning})")
        return problems
