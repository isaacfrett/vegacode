"""Live machine readings for the chat status line: CPU %, memory used/total and pressure, swap,
and the model Ollama has loaded with its memory.

Cheap enough to read every two seconds: no subprocess per reading. macOS asks the kernel through
ctypes (host_statistics for CPU ticks and VM page counts, sysctlbyname for swap and the memory
pressure level); Linux reads /proc (stat, meminfo, pressure/memory); elsewhere psutil is used when
it happens to be installed (it is not a dependency), else the figures are left out. The loaded
model comes from Ollama's /api/ps (one local HTTP request, at most every OLLAMA_EVERY seconds).
Every figure is optional: a reading that fails is None and the status line leaves it out.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OLLAMA_EVERY = 5.0  # seconds between /api/ps requests (the model changes rarely)
PRESSURE_NAMES = {1: "normal", 2: "warn", 4: "critical"}  # kern.memorystatus_vm_pressure_level


@dataclass(frozen=True)
class Reading:
    cpu_pct: float | None = None  # all cores, since the previous reading
    mem_used_gb: float | None = None  # app + wired + compressed (Activity Monitor's "used")
    mem_total_gb: float | None = None
    pressure: str | None = None  # normal | warn | critical (macOS level, Linux PSI)
    swap_used_gb: float | None = None
    model: str | None = None  # the model Ollama has loaded ("" = Ollama answered, none loaded)
    model_gb: float | None = None  # its memory as Ollama reports it (size)


# -- macOS: mach host statistics and sysctl through ctypes ----------------------------------------

class _CpuLoad(ctypes.Structure):
    _fields_ = [("ticks", ctypes.c_uint32 * 4)]  # user, system, idle, nice


class _VmStats64(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint32), ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32), ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64), ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64), ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64), ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64), ("hits", ctypes.c_uint64), ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32), ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64), ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64), ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32), ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32), ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


class _SwapUsage(ctypes.Structure):
    _fields_ = [("total", ctypes.c_uint64), ("avail", ctypes.c_uint64),
                ("used", ctypes.c_uint64), ("pagesize", ctypes.c_uint32),
                ("encrypted", ctypes.c_bool)]


HOST_CPU_LOAD_INFO = 3
HOST_VM_INFO64 = 4


class _Mac:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL(ctypes.util.find_library("System") or "/usr/lib/libSystem.B.dylib")
        self.lib.mach_host_self.restype = ctypes.c_uint32
        self.host = self.lib.mach_host_self()
        self.page = os.sysconf("SC_PAGE_SIZE")

    def sysctl(self, name: str, value: Any) -> bool:
        size = ctypes.c_size_t(ctypes.sizeof(value))
        return self.lib.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size),
                                     None, ctypes.c_size_t(0)) == 0

    def cpu_ticks(self) -> tuple[int, int] | None:
        load = _CpuLoad()
        count = ctypes.c_uint32(4)
        if self.lib.host_statistics(self.host, HOST_CPU_LOAD_INFO, ctypes.byref(load),
                                    ctypes.byref(count)) != 0:
            return None
        user, system, idle, nice = load.ticks
        return user + system + nice, user + system + idle + nice

    def memory(self) -> tuple[float | None, float | None]:
        stats = _VmStats64()
        count = ctypes.c_uint32(ctypes.sizeof(_VmStats64) // 4)
        total = ctypes.c_uint64()
        total_gb = total.value / 1024**3 if self.sysctl("hw.memsize", total) else None
        if self.lib.host_statistics64(self.host, HOST_VM_INFO64, ctypes.byref(stats),
                                      ctypes.byref(count)) != 0:
            return None, total_gb
        pages = (stats.internal_page_count - stats.purgeable_count + stats.wire_count
                 + stats.compressor_page_count)
        return pages * self.page / 1024**3, total_gb

    def pressure(self) -> str | None:
        level = ctypes.c_int()
        return PRESSURE_NAMES.get(level.value) if self.sysctl(
            "kern.memorystatus_vm_pressure_level", level) else None

    def swap_gb(self) -> float | None:
        usage = _SwapUsage()
        return usage.used / 1024**3 if self.sysctl("vm.swapusage", usage) else None


# -- Linux: /proc -----------------------------------------------------------------------------------

def _linux_cpu_ticks() -> tuple[int, int] | None:
    first = Path("/proc/stat").read_text().split("\n", 1)[0].split()
    if not first or first[0] != "cpu":
        return None
    values = [int(v) for v in first[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
    return sum(values) - idle, sum(values)


def _linux_memory() -> tuple[float | None, float | None, float | None]:
    info: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            info[name] = int(parts[0]) * 1024
    total = info.get("MemTotal")
    used = total - info["MemAvailable"] if total and "MemAvailable" in info else None
    swap = (info["SwapTotal"] - info.get("SwapFree", 0)) if "SwapTotal" in info else None
    gb = 1024**3
    return (used / gb if used is not None else None, total / gb if total else None,
            swap / gb if swap is not None else None)


def _linux_pressure() -> str | None:
    """Memory PSI: the share of the last 10 s some task waited on memory."""
    try:
        line = Path("/proc/pressure/memory").read_text().splitlines()[0]
        avg10 = float(line.split("avg10=")[1].split()[0])
    except (OSError, IndexError, ValueError):
        return None
    return "critical" if avg10 >= 10 else "warn" if avg10 >= 1 else "normal"


# -- Ollama ------------------------------------------------------------------------------------------

def ollama_ps() -> dict[str, Any] | None:
    """Ollama's /api/ps (None when it does not answer). Tests replace this."""
    from . import server

    return server._get_json("/api/ps")


def loaded_model(data: dict[str, Any] | None) -> tuple[str | None, float | None]:
    """The first loaded model's name and size in GB; ("", None) when none is loaded, (None, None)
    when Ollama did not answer."""
    if not isinstance(data, dict):
        return None, None
    models = [m for m in data.get("models") or [] if isinstance(m, dict)]
    if not models:
        return "", None
    first = models[0]
    size = first.get("size")
    return str(first.get("name") or first.get("model") or ""), (
        int(size) / 1024**3 if isinstance(size, int | float) and size else None)


class Sampler:
    """Call read() every couple of seconds (from a worker thread: /api/ps is a network call).
    CPU % is measured between consecutive calls, so the first reading has none."""

    def __init__(self, *, ollama: Callable[[], dict[str, Any] | None] | None = None,
                 ollama_every: float = OLLAMA_EVERY, clock: Callable[[], float] = time.monotonic):
        self.ollama = ollama or (lambda: ollama_ps())
        self.ollama_every = ollama_every
        self.clock = clock
        self._ticks: tuple[int, int] | None = None
        self._model: tuple[str | None, float | None] = (None, None)
        self._model_at: float | None = None
        self._mac: _Mac | None = None
        self._psutil: Any = None
        if sys.platform == "darwin":
            try:
                self._mac = _Mac()
            except (OSError, AttributeError, ValueError):
                self._mac = None
        elif not sys.platform.startswith("linux"):
            try:
                import psutil  # type: ignore[import-not-found]  # optional, not a dependency

                self._psutil = psutil
            except ImportError:
                self._psutil = None

    def _cpu(self, ticks: tuple[int, int] | None) -> float | None:
        previous, self._ticks = self._ticks, ticks
        if ticks is None or previous is None or ticks[1] <= previous[1]:
            return None
        return max(0.0, min(100.0, 100 * (ticks[0] - previous[0]) / (ticks[1] - previous[1])))

    def machine(self) -> Reading:
        cpu = used = total = swap = None
        pressure = None
        try:
            if self._mac is not None:
                cpu = self._cpu(self._mac.cpu_ticks())
                used, total = self._mac.memory()
                pressure = self._mac.pressure()
                swap = self._mac.swap_gb()
            elif sys.platform.startswith("linux"):
                cpu = self._cpu(_linux_cpu_ticks())
                used, total, swap = _linux_memory()
                pressure = _linux_pressure()
            elif self._psutil is not None:
                cpu = self._psutil.cpu_percent(interval=None)
                memory = self._psutil.virtual_memory()
                used, total = (memory.total - memory.available) / 1024**3, memory.total / 1024**3
                swap = self._psutil.swap_memory().used / 1024**3
        except (OSError, ValueError, IndexError, AttributeError):
            pass
        return Reading(cpu_pct=cpu, mem_used_gb=used, mem_total_gb=total, pressure=pressure,
                       swap_used_gb=swap)

    def read(self) -> Reading:
        reading = self.machine()
        now = self.clock()
        if self._model_at is None or now - self._model_at >= self.ollama_every:
            self._model_at = now
            try:
                self._model = loaded_model(self.ollama())
            except Exception:  # noqa: BLE001 - the model figure is optional
                self._model = (None, None)
        name, size = self._model
        return Reading(cpu_pct=reading.cpu_pct, mem_used_gb=reading.mem_used_gb,
                       mem_total_gb=reading.mem_total_gb, pressure=reading.pressure,
                       swap_used_gb=reading.swap_used_gb, model=name, model_gb=size)


def gb_label(value: float) -> str:
    return f"{value:.1f}" if value < 100 else f"{value:.0f}"


def segments(reading: Reading, model_hint: str = "") -> list[tuple[str, str]]:
    """The status line's readings in priority order, as (text, kind): kind is "" for the plain
    dim text, "warn" for a reading that needs attention (memory pressure, swap in use with
    pressure). model_hint names the configured model when Ollama has none loaded."""
    out: list[tuple[str, str]] = []
    if reading.cpu_pct is not None:
        out.append((f"cpu {reading.cpu_pct:.0f}%", ""))
    if reading.mem_used_gb is not None:
        total = f"/{reading.mem_total_gb:.0f}" if reading.mem_total_gb else ""
        text = f"mem {gb_label(reading.mem_used_gb)}{total} GB"
        if reading.pressure:
            text += f" {reading.pressure}"
        out.append((text, "warn" if reading.pressure in {"warn", "critical"} else ""))
    if reading.swap_used_gb is not None:
        busy = reading.swap_used_gb >= 1 and reading.pressure in {"warn", "critical"}
        out.append((f"swap {gb_label(reading.swap_used_gb)} GB", "warn" if busy else ""))
    if reading.model:
        name = reading.model.removesuffix(":latest")
        if len(name) > 28:
            name = name[:27] + "…"
        size = f" {gb_label(reading.model_gb)} GB" if reading.model_gb else ""
        out.append((f"{name}{size}", ""))
    elif reading.model == "":
        out.append((f"{model_hint} not loaded" if model_hint else "no model loaded", ""))
    return out
