"""Resource guard: pause under pressure, resume when it eases, stop when it does not."""

from __future__ import annotations

from vega.resources import Guard, Limits, Snapshot


def snap(free: int = 80, swap: float = 500, disk: float = 30, rss: float = 5000) -> Snapshot:
    return Snapshot(free_pct=free, swap_used_mb=swap, disk_free_gb=disk, model_rss_mb=rss)


def make_guard(readings: list[Snapshot], **limits) -> tuple[Guard, list[float], list[str]]:
    slept: list[float] = []
    logs: list[str] = []
    queue = list(readings)
    guard = Guard(
        Limits(min_free_pct=20, min_disk_gb=12, max_swap_growth_mb=2048, max_wait_seconds=60,
               poll_seconds=15, **limits),
        probe=lambda: queue.pop(0) if len(queue) > 1 else queue[0],
        sleep=slept.append, log=logs.append,
    )
    return guard, slept, logs


def test_healthy_machine_never_pauses() -> None:
    guard, slept, _ = make_guard([snap(), snap(), snap()])
    assert guard.before_attempt() is None and guard.before_attempt() is None
    assert slept == [] and guard.pauses == 0


def test_low_memory_pauses_then_resumes() -> None:
    guard, slept, logs = make_guard([snap(), snap(free=10), snap(free=12), snap(free=40)])
    assert guard.before_attempt() is None
    assert slept == [15, 15] and guard.pauses == 1
    assert "free memory 10%" in logs[0] and logs[-1].startswith("Resumed")


def test_swap_jump_between_checks_stops_after_max_wait() -> None:
    growing = [snap(swap=1000 + 3000 * step) for step in range(8)]  # still climbing at every check
    guard, slept, _ = make_guard(growing)
    reason = guard.before_attempt()
    assert reason and "swap grew 3000 MB since the last check" in reason
    assert sum(slept) == 60


def test_old_swap_growth_alone_does_not_pause() -> None:
    # Swap that grew earlier and then stayed flat is history, not pressure.
    guard, slept, _ = make_guard([snap(swap=500), snap(swap=1400), snap(swap=1400), snap(swap=1450)])
    assert guard.before_attempt() is None and guard.before_attempt() is None
    assert slept == [] and guard.report()["peak_swap_growth_mb"] == 950


def test_critical_pressure_pauses() -> None:
    critical = Snapshot(free_pct=60, swap_used_mb=500, disk_free_gb=30, model_rss_mb=0, pressure_level=4)
    guard, _, logs = make_guard([snap(), critical, snap()])
    assert guard.before_attempt() is None
    assert "critical memory pressure" in logs[0] and guard.report()["max_pressure_level"] == 4


def test_low_disk_is_reported() -> None:
    guard, _, _ = make_guard([snap(), snap(disk=8)])
    assert "free disk 8.0 GB < 12 GB" in guard.before_attempt()


def test_report_tracks_extremes() -> None:
    guard, _, _ = make_guard([snap(swap=500), snap(free=35, swap=900, rss=9000, disk=20), snap(free=60)])
    guard.before_attempt()
    report = guard.report()
    assert report["min_free_pct"] == 35 and report["peak_swap_growth_mb"] == 400
    assert report["peak_model_rss_mb"] == 9000 and report["min_disk_free_gb"] == 20


def test_swap_jump_resumes_once_swap_stops_growing() -> None:
    guard, slept, logs = make_guard([snap(swap=1000), snap(swap=4000), snap(swap=4100)])
    assert guard.before_attempt() is None
    assert slept == [15] and logs[-1].startswith("Resumed")


def test_thermal_warning_pauses_until_it_clears() -> None:
    hot = Snapshot(free_pct=80, swap_used_mb=500, disk_free_gb=30, model_rss_mb=0,
                   thermal_warning="CPU_Speed_Limit 80")
    guard, slept, logs = make_guard([snap(), hot, hot, snap()])
    assert guard.before_attempt() is None
    assert slept == [15, 15] and "heat warning" in logs[0]
    assert guard.report()["thermal_warnings_seen"] == 2


def test_thermal_warning_that_persists_stops_the_run() -> None:
    hot = Snapshot(free_pct=80, swap_used_mb=500, disk_free_gb=30, model_rss_mb=0, thermal_warning="hot")
    guard, _, _ = make_guard([snap(), hot])
    assert "heat warning" in guard.before_attempt()


def test_thermal_warning_parsing(monkeypatch) -> None:
    from vega import resources

    quiet = ("Note: No thermal warning level has been recorded\n"
             "Note: No performance warning level has been recorded\n")
    monkeypatch.setattr(resources, "_run", lambda argv: quiet)
    assert resources.thermal_warning() == ""
    monkeypatch.setattr(resources, "_run", lambda argv: (
        "Thermal warning level: 2\nNote: No performance warning level has been recorded\n"))
    assert "Thermal warning level: 2" in resources.thermal_warning()
