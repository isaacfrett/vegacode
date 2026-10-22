"""Keep long model runs from maxing out the machine: check memory, swap and disk between attempts.

macOS only for the memory figures (memory_pressure, vm.swapusage); elsewhere the guard only checks
disk space. Thresholds come from the defaults below or VEGA_GUARD_* environment variables.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

DISK_PATH = Path("/System/Volumes/Data") if Path("/System/Volumes/Data").exists() else Path.home()


PRESSURE_CRITICAL = 4  # kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical


@dataclass
class Snapshot:
    free_pct: int | None  # system-wide free memory percentage (memory_pressure)
    swap_used_mb: float | None
    disk_free_gb: float
    model_rss_mb: float  # resident memory of the model server process(es)
    pressure_level: int | None = None  # macOS memory pressure level (1, 2 or 4)
    thermal_warning: str = ""  # non-empty when macOS reports a thermal or performance warning


def _run(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, text=True, capture_output=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def is_model_process(command: str) -> bool:
    """Model server processes whose memory counts as the model's: Ollama and its llama-server
    runners, plus any names in VEGA_MODEL_PROCESSES (comma-separated substrings, e.g. "LM Studio"
    for an endpoint profile's server; without them that server's memory shows as 0 here, while
    the free-memory and swap checks still apply)."""
    extra = [name.strip() for name in os.environ.get("VEGA_MODEL_PROCESSES", "").split(",")
             if name.strip()]
    return ("llama-server" in command or command.endswith("ollama")
            or any(name in command for name in extra))


def snapshot() -> Snapshot:
    free = re.search(r"free percentage:\s*(\d+)%", _run(["memory_pressure"]))
    swap = re.search(r"used = ([\d.]+)M", _run(["sysctl", "-n", "vm.swapusage"]))
    rss_kb = 0
    for line in _run(["ps", "-axo", "rss=,comm="]).splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and is_model_process(parts[1]):
            rss_kb += int(parts[0])
    level = _run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip()
    return Snapshot(
        thermal_warning=thermal_warning(),
        pressure_level=int(level) if level.isdigit() else None,
        free_pct=int(free.group(1)) if free else None,
        swap_used_mb=float(swap.group(1)) if swap else None,
        disk_free_gb=shutil.disk_usage(DISK_PATH).free / 1024**3,
        model_rss_mb=rss_kb / 1024,
    )


def total_memory_gb() -> float | None:
    """Physical memory in GB on macOS, Linux and Windows; None when it cannot be read."""
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("free_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong),
                            ("free_virtual", ctypes.c_ulonglong),
                            ("free_extended", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.total / 1024**3
        except (OSError, AttributeError, ValueError):
            return None
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (AttributeError, OSError, ValueError):
        return None


def thermal_warning() -> str:
    """macOS thermal/performance warning text from `pmset -g therm`, or "" when none is recorded.
    A warning means the chip is hot enough that macOS is (or is about to start) throttling it."""
    output = _run(["pmset", "-g", "therm"])
    if not output:
        return ""
    problems = []
    for kind in ("thermal", "performance"):
        if f"No {kind} warning level has been recorded" not in output:
            match = re.search(rf"{kind} warning level[^\n]*", output, re.IGNORECASE)
            problems.append(match.group(0).strip() if match else f"{kind} warning")
    return "; ".join(problems)


def _env_number(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Limits:
    min_free_pct: float = field(default_factory=lambda: _env_number("VEGA_GUARD_MIN_FREE_PCT", 20))
    min_disk_gb: float = field(default_factory=lambda: _env_number("VEGA_GUARD_MIN_DISK_GB", 12))
    # Swap that grows this much between two checks means pressure is happening now. (Total growth
    # since the start is not used: macOS keeps swap allocated long after the pressure has passed.)
    max_swap_growth_mb: float = field(
        default_factory=lambda: _env_number("VEGA_GUARD_MAX_SWAP_GROWTH_MB", 1024))
    max_wait_seconds: float = field(default_factory=lambda: _env_number("VEGA_GUARD_MAX_WAIT", 300))
    poll_seconds: float = 15


class Guard:
    """Call `before_attempt()` before each model attempt. It pauses while the machine is under
    pressure and returns a stop reason if pressure does not ease within `max_wait_seconds`."""

    def __init__(
        self, limits: Limits | None = None, *, probe: Callable[[], Snapshot] = snapshot,
        sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print,
    ):
        self.limits = limits or Limits()
        self.probe, self.sleep, self.log = probe, sleep, log
        self.start = probe()
        self.min_free_pct = self.start.free_pct
        self.peak_swap_growth_mb = 0.0
        self.peak_model_rss_mb = self.start.model_rss_mb
        self.min_disk_gb = self.start.disk_free_gb
        self.last_swap_mb = self.start.swap_used_mb
        self.max_pressure = self.start.pressure_level
        self.thermal_events = 0
        self.pauses = 0
        self.paused_seconds = 0.0

    def _observe(self, now: Snapshot) -> list[str]:
        limits = self.limits
        if now.free_pct is not None:
            self.min_free_pct = now.free_pct if self.min_free_pct is None else min(
                self.min_free_pct, now.free_pct)
        self.min_disk_gb = min(self.min_disk_gb, now.disk_free_gb)
        self.peak_model_rss_mb = max(self.peak_model_rss_mb, now.model_rss_mb)
        problems = []
        if now.free_pct is not None and now.free_pct < limits.min_free_pct:
            problems.append(f"free memory {now.free_pct}% < {limits.min_free_pct:.0f}%")
        if now.disk_free_gb < limits.min_disk_gb:
            problems.append(f"free disk {now.disk_free_gb:.1f} GB < {limits.min_disk_gb:.0f} GB")
        if now.swap_used_mb is not None and self.start.swap_used_mb is not None:
            self.peak_swap_growth_mb = max(
                self.peak_swap_growth_mb, now.swap_used_mb - self.start.swap_used_mb)
            if self.last_swap_mb is not None:
                jump = now.swap_used_mb - self.last_swap_mb
                if jump > limits.max_swap_growth_mb:
                    problems.append(
                        f"swap grew {jump:.0f} MB since the last check > "
                        f"{limits.max_swap_growth_mb:.0f} MB")
            self.last_swap_mb = now.swap_used_mb
        if now.thermal_warning:
            self.thermal_events += 1
            problems.append(f"macOS heat warning ({now.thermal_warning})")
        if now.pressure_level is not None:
            self.max_pressure = max(self.max_pressure or 0, now.pressure_level)
            if now.pressure_level >= PRESSURE_CRITICAL:
                problems.append("macOS reports critical memory pressure")
        return problems

    def before_attempt(self) -> str | None:
        problems = self._observe(self.probe())
        if not problems:
            return None
        self.pauses += 1
        waited = 0.0
        self.log(f"Pausing: {'; '.join(problems)}. Waiting for the machine to recover…")
        while problems and waited < self.limits.max_wait_seconds:
            self.sleep(self.limits.poll_seconds)
            waited += self.limits.poll_seconds
            problems = self._observe(self.probe())
        self.paused_seconds += waited
        if problems:
            return f"stopped by resource guard after waiting {waited:.0f}s: {'; '.join(problems)}"
        self.log(f"Resumed after {waited:.0f}s.")
        return None

    def report(self) -> dict[str, float | int | None]:
        end = self.probe()
        self._observe(end)
        return {
            "min_free_pct": self.min_free_pct,
            "peak_swap_growth_mb": round(self.peak_swap_growth_mb),
            "peak_model_rss_mb": round(self.peak_model_rss_mb),
            "min_disk_free_gb": round(self.min_disk_gb, 1),
            "max_pressure_level": self.max_pressure,
            "thermal_warnings_seen": self.thermal_events,
            "pauses": self.pauses,
            "paused_seconds": round(self.paused_seconds),
        }
