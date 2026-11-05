"""The test watchdog (tests/conftest.py): a test still running after VEGA_TEST_WATCHDOG_S seconds
gets every thread's stack dumped to the real terminal, past pytest's output capture, so a hang on
CI names itself in the log; @pytest.mark.watchdog(seconds) gives a known-slow test longer, and
VEGA_TEST_WATCHDOG_EXIT=1 (CI) ends the run after the dump."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

CONFTEST = Path(__file__).resolve().parent / "conftest.py"

SLOW = '''import time

import pytest


def test_sleeps_past_the_watchdog():
    time.sleep(8)


@pytest.mark.watchdog(60)
def test_marked_slow_is_given_longer():
    time.sleep(5)
'''


def run(tmp_path: Path, test: str, seconds: str, exit_after: str = "") -> subprocess.CompletedProcess[str]:
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "test_slow.py").write_text(SLOW, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env.update(VEGA_TEST_WATCHDOG_S=seconds, VEGA_TEST_WATCHDOG_EXIT=exit_after,
               VEGA_HOME=str(tmp_path / "home"))
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           f"test_slow.py::{test}"], cwd=tmp_path, env=env, capture_output=True,
                          text=True, timeout=300, check=False)


def test_a_slow_test_has_its_stack_dumped_and_still_passes(tmp_path):
    # 3 s, not 1: the timer is armed before setup, and on a cold CI runner setup's first imports took
    # over a second, so a 1 s dump named an import instead of the test (vegacode PR #10, macOS 3.12)
    done = run(tmp_path, "test_sleeps_past_the_watchdog", "3")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Timeout (0:00:03)!" in done.stderr
    assert "in test_sleeps_past_the_watchdog" in done.stderr  # the dump names the test


def test_exit_mode_ends_the_run_after_the_dump(tmp_path):
    done = run(tmp_path, "test_sleeps_past_the_watchdog", "3", exit_after="1")
    assert done.returncode != 0 and "in test_sleeps_past_the_watchdog" in done.stderr
    assert " passed" not in done.stdout  # killed mid-test: no summary


def test_the_marker_gives_longer_and_zero_switches_it_off(tmp_path):
    marked = run(tmp_path, "test_marked_slow_is_given_longer", "3")
    off = run(tmp_path, "test_sleeps_past_the_watchdog", "0")
    for done in (marked, off):
        assert done.returncode == 0, done.stdout + done.stderr
        assert "Timeout" not in done.stderr
