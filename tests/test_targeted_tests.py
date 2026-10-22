"""targeted_tests: related tests first on a large pytest suite, the whole suite with a scaled
timeout, and a whole-suite timeout reported as slow, not failing."""

from __future__ import annotations

from pathlib import Path

import pytest
from switches import switch_off

from vega import acceptance, sandbox, targeted_tests
from vega.acceptance import Spec

COMMAND = "python -m pytest -q"


@pytest.fixture(autouse=True)
def fresh_suite_times():
    targeted_tests._suite_seconds.clear()
    yield
    targeted_tests._suite_seconds.clear()


def large_project(root: Path, count: int = targeted_tests.LARGE_TEST_FILES) -> Path:
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text("")
    (root / "src" / "pkg" / "tools.py").write_text("def f():\n    return 1\n")
    (root / "src" / "pkg" / "other.py").write_text("def g():\n    return 2\n")
    (root / "data.json").write_text("{}")
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_tools.py").write_text("from pkg.tools import f\n\ndef test_f():\n    assert f()\n")
    (tests / "test_uses_tools.py").write_text("from pkg import tools\n\ndef test_t():\n    pass\n")
    (tests / "test_wide.py").write_text("from pkg import tools, other\n\ndef test_w():\n    pass\n")
    (tests / "test_data.py").write_text("PATH = 'data.json'\n\ndef test_d():\n    pass\n")
    for index in range(count - 4):
        (tests / f"test_unrelated_{index}.py").write_text(
            "from pkg.other import g\n\ndef test_g():\n    assert g()\n")
    return root


def off(monkeypatch, name: str) -> None:
    switch_off(monkeypatch, name)  # the switch under test, to show the old behaviour


# -- selection ----------------------------------------------------------------------------------


def test_whole_suite_commands():
    assert targeted_tests.whole_suite("python -m pytest -q")
    assert targeted_tests.whole_suite("pytest -q -k slow --tb=short")
    assert targeted_tests.whole_suite("python3 -m pytest -x -p no:cacheprovider")
    assert not targeted_tests.whole_suite("python -m pytest -q tests/test_x.py")
    assert not targeted_tests.whole_suite("pytest tests")
    assert not targeted_tests.whole_suite("npm test")
    assert not targeted_tests.whole_suite("python check.py")


def test_select_orders_named_then_importers_then_mentions(tmp_path):
    project = large_project(tmp_path)
    files, related = targeted_tests.select(project, ["src/pkg/tools.py"])
    assert files == ["tests/test_tools.py", "tests/test_uses_tools.py", "tests/test_wide.py"]
    assert related == 3
    files, _ = targeted_tests.select(project, ["data.json"])
    assert files == ["tests/test_data.py"]


def test_select_puts_edited_tests_first_and_caps(tmp_path, monkeypatch):
    project = large_project(tmp_path)
    (project / "tests" / "test_new.py").write_text("def test_n():\n    pass\n")
    files, related = targeted_tests.select(project, ["src/pkg/tools.py", "tests/test_new.py"])
    assert files[0] == "tests/test_new.py"
    monkeypatch.setattr(targeted_tests, "MAX_FILES", 2)
    files, related = targeted_tests.select(project, ["src/pkg/other.py"])
    assert len(files) == 2 and related == targeted_tests.LARGE_TEST_FILES - 4 + 1


def test_applies_only_to_large_whole_suite_pytest(tmp_path, monkeypatch):
    project = large_project(tmp_path)
    assert targeted_tests.applies(project, COMMAND)
    assert not targeted_tests.applies(project, "python -m pytest -q tests/test_tools.py")
    small = large_project(tmp_path / "small", count=5)
    assert not targeted_tests.applies(small, COMMAND)
    off(monkeypatch, "targeted_tests")
    assert not targeted_tests.applies(project, COMMAND)


def test_full_timeout_scales_and_is_bounded(tmp_path):
    project = large_project(tmp_path)
    assert targeted_tests.full_timeout(project) == targeted_tests.FULL_FIRST_TIMEOUT
    # 3 of 30 files in 10 s -> about 100 s for the suite -> 1.5 x 100 + 15
    assert targeted_tests.full_timeout(project, (10.0, 3)) == 165
    assert targeted_tests.full_timeout(project, (0.5, 3)) == targeted_tests.FULL_MIN_ESTIMATED
    targeted_tests.record_suite(project, 80.0)
    assert targeted_tests.full_timeout(project, (10.0, 3)) == 135
    targeted_tests.record_suite(project, 2.0)
    assert targeted_tests.full_timeout(project) == sandbox.TIMEOUT
    targeted_tests.record_suite(project, 5000.0)
    assert targeted_tests.full_timeout(project) == targeted_tests.FULL_MAX_TIMEOUT


# -- acceptance.run -------------------------------------------------------------------------------


def fake_checks(monkeypatch, replies):
    """sandbox.run_check stub: replies(command, timeout) -> output; records the calls."""
    calls: list[tuple[str, int]] = []

    def run_check(command, project, timeout=sandbox.TIMEOUT, **_):
        calls.append((command, timeout))
        return replies(command, timeout)

    monkeypatch.setattr(sandbox, "run_check", run_check)
    monkeypatch.setattr(acceptance, "_run_names", lambda *a, **k: [])
    return calls


def is_targeted(command: str) -> bool:
    return "tests/" in command


def test_slow_full_suite_is_a_note_not_a_failure(tmp_path, monkeypatch):
    project = large_project(tmp_path)
    calls = fake_checks(monkeypatch, lambda c, t: "exit code: 0\n3 passed in 0.4s" if is_targeted(c)
                        else f"Check timed out after {t}s.")
    spec = Spec(task="fix it", tests=COMMAND)
    result = acceptance.run(spec, project, ["src/pkg/tools.py"])
    assert result.ok
    assert len(calls) == 2
    targeted, full = calls
    assert targeted[1] == sandbox.TIMEOUT and "tests/test_tools.py" in targeted[0]
    assert not is_targeted(full[0]) and full[1] > sandbox.TIMEOUT
    run = result.as_run()
    assert "the full suite is slow, not failing" in run and "the targeted tests passed" in run
    assert "3 passed" in run
    # a later finish check in the same turn does not wait out the whole suite again
    calls.clear()
    again = acceptance.run(spec, project, ["src/pkg/tools.py"])
    assert again.ok and len(calls) == 1 and "not run again" in again.as_run()


def test_targeted_failure_stops_before_the_whole_suite(tmp_path, monkeypatch):
    project = large_project(tmp_path)
    calls = fake_checks(monkeypatch, lambda c, t: (
        "exit code: 1\nFAILED tests/test_tools.py::test_f - assert 0\n1 failed in 0.2s"))
    result = acceptance.run(Spec(task="fix it", tests=COMMAND), project, ["src/pkg/tools.py"])
    assert not result.ok and len(calls) == 1
    line = result.failing[0].line()
    assert "related to your changes" in line and "test_f" in line


def test_full_suite_failure_still_fails_and_its_time_is_recorded(tmp_path, monkeypatch):
    project = large_project(tmp_path)
    fake_checks(monkeypatch, lambda c, t: "exit code: 0\n3 passed" if is_targeted(c)
                else "exit code: 1\nFAILED tests/test_unrelated_0.py::test_g\n1 failed, 40 passed")
    result = acceptance.run(Spec(task="fix it", tests=COMMAND), project, ["src/pkg/tools.py"])
    assert not result.ok and "(whole suite)" in result.failing[0].label
    assert targeted_tests.key(project) in targeted_tests._suite_seconds


def test_no_related_tests_runs_whole_suite_with_longer_timeout(tmp_path, monkeypatch):
    project = large_project(tmp_path)
    (project / "notes.txt").write_text("x")
    calls = fake_checks(monkeypatch, lambda c, t: f"Check timed out after {t}s.")
    result = acceptance.run(Spec(task="fix it", tests=COMMAND), project, ["notes.txt"])
    assert result.ok and len(calls) == 1
    assert calls[0] == (COMMAND + " --tb=short", targeted_tests.FULL_FIRST_TIMEOUT) \
        or calls[0] == (COMMAND, targeted_tests.FULL_FIRST_TIMEOUT)
    assert "no test file relates" in result.as_run()


def test_small_project_and_switch_off_keep_old_behaviour(tmp_path, monkeypatch):
    calls = fake_checks(monkeypatch, lambda c, t: f"Check timed out after {t}s.")
    small = large_project(tmp_path / "small", count=5)
    result = acceptance.run(Spec(task="fix it", tests=COMMAND), small, ["src/pkg/tools.py"])
    assert not result.ok and calls == [(calls[0][0], sandbox.TIMEOUT)] and not is_targeted(calls[0][0])
    calls.clear()
    off(monkeypatch, "targeted_tests")
    large = large_project(tmp_path / "large")
    result = acceptance.run(Spec(task="fix it", tests=COMMAND), large, ["src/pkg/tools.py"])
    assert not result.ok and len(calls) == 1 and calls[0][1] == sandbox.TIMEOUT
    assert "Check timed out after 60s" in result.failing[0].line()


# -- the model's own run_check ------------------------------------------------------------------


def test_run_check_timeout_hint(tmp_path, monkeypatch):
    project = large_project(tmp_path)
    hint = targeted_tests.run_check_hint(COMMAND, project, 60)
    assert "not a test failure" in hint and f"{targeted_tests.LARGE_TEST_FILES} test files" in hint
    assert targeted_tests.run_check_hint(COMMAND + " tests/test_tools.py", project, 60) == ""
    assert sandbox._timeout_hint(COMMAND, project, 60, harness=True) == ""
    assert sandbox._timeout_hint(COMMAND, project, 60, harness=False) == hint
    small = large_project(tmp_path / "small", count=5)
    assert targeted_tests.run_check_hint(COMMAND, small, 60) == ""
    off(monkeypatch, "targeted_tests")
    assert targeted_tests.run_check_hint(COMMAND, project, 60) == ""


@pytest.mark.skipif(not sandbox.available(), reason="needs macOS sandbox-exec")
def test_real_sandbox_timeout_message_carries_the_hint(tmp_path, monkeypatch):
    project = large_project(tmp_path)
    (project / "tests" / "test_slow.py").write_text(
        "import time\n\ndef test_slow():\n    time.sleep(30)\n")
    output = sandbox.run_check(COMMAND, project, timeout=2)
    assert output.startswith("Check timed out after 2s.") and "not a test failure" in output
