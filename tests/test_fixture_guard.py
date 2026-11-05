"""vega.fixture_guard: a selfbuild case whose UNMODIFIED fixture fails its own checks is aborted as
"fixture red" before any model attempt (audit 2026-10-27: 58 of 73 selfbuild attempts ran on such
fixtures and were scored as model failures), and hidden floors are recorded on the cases."""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from vega import evals, fixture_guard


def _sandbox_works() -> bool:
    if not evals.sandbox.available():
        return False
    probe = subprocess.run(["sandbox-exec", "-p", "(version 1)(allow default)", "true"], capture_output=True, check=False)
    return probe.returncode == 0


def make_archive(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


PYTEST = "python -m pytest -q -p no:cacheprovider"
SOURCE = {"src/calc.py": "def add(a, b):\n    return a + b\n",
          "tests/test_calc.py": "from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"}
HIDDEN = {"hidden_tests/test_h.py": ("from calc import add\n\ndef test_one():\n    assert add(1, 1) == 2\n\n"
                                     "def test_zero():\n    assert add(0, 0) == 0\n")}
CASE = {"id": "sb-add", "vega_snapshot": True, "regression_command": f"{PYTEST} tests",
        "check_command": f"{PYTEST} hidden_tests", "hidden_files": HIDDEN,
        "mutation": [{"path": "src/calc.py", "old": "a + b", "new": "a - b"}]}


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VEGA_HOME", str(tmp_path / "home"))
    for name in (fixture_guard.GUARD_ENV, fixture_guard.CACHE_ENV, fixture_guard.TIMEOUT_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(fixture_guard, "snapshot_key", lambda *a, **k: ("key1", "abc1234"))
    return tmp_path / "home"


def guard(cases: list[dict], archive: bytes = b"x") -> fixture_guard.GuardResult:
    return fixture_guard.check(cases, archive, repo=Path("."), paths=("src",), exclude=())


def fake(monkeypatch: pytest.MonkeyPatch, regression: dict, case_row: dict | None = None) -> list[str]:
    calls: list[str] = []

    def run_regression(command: str, project: Path, limit: int) -> dict:
        calls.append(command)
        return dict(regression)

    monkeypatch.setattr(fixture_guard, "run_regression", run_regression)
    monkeypatch.setattr(evals, "build_fixture", lambda case, project, archive=None: None)
    monkeypatch.setattr(fixture_guard, "check_case", lambda case, archive: {
        "digest": fixture_guard._case_digest(case), **(case_row or {"unmutated": [2, 2], "floor": [0, 2]})})
    return calls


def test_red_tests_a_case_does_not_touch_become_its_baseline_not_an_abort(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """11-03: one host-state test failed on the unmodified snapshot and the guard aborted all 15
    cases. A failing test in a file the case neither mutates nor expects to change is recorded as
    the case's baseline (graded as no new failures), and only a case touching it is aborted."""
    failing = [f"tests/lab/test_night.py::test_{i}" for i in range(7)]
    calls = fake(monkeypatch, {"status": "red", "failing": failing, "count": "7 failed, 10 passed", "seconds": 1})
    other = {**CASE, "id": "sb-two"}
    touching = {**CASE, "id": "sb-night", "expected_files": ["src/calc.py", "tests/lab/test_night.py"]}
    plain = {"id": "plain"}  # not a snapshot case: never guarded
    result = guard([CASE, other, touching, plain])
    assert result.baselines == {"sb-add": failing, "sb-two": failing}
    assert set(result.errors) == {"sb-night"}
    error = result.errors["sb-night"]
    assert error.startswith("fixture red: 7 tests in files this case touches fail on the unmodified snapshot (")
    assert "tests/lab/test_night.py::test_0" in error and "+2 more" in error
    assert fixture_guard.is_red(error)
    assert result.floors == {"sb-add": [0, 2], "sb-two": [0, 2], "sb-night": [0, 2]}
    assert result.summary["status"] == "red" and result.summary["baseline_cases"] == ["sb-add", "sb-two"]
    assert "fixture red on 1 of 3 snapshot case(s) at abc1234" in result.note()
    assert "7 test(s) fail on the unmodified snapshot at abc1234" in result.note()
    assert "2 case(s) that do not touch them are graded as no new failures" in result.note()
    assert len(calls) == 1  # one regression run for the commands the cases share
    assert guard([CASE]).summary["status"] == "baseline"  # cached; nothing aborted


def test_a_red_run_that_names_no_test_still_aborts_every_case(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake(monkeypatch, {"status": "red", "failing": [], "count": "exit code: 2", "seconds": 1})
    result = guard([CASE, {**CASE, "id": "sb-two"}])
    assert set(result.errors) == {"sb-add", "sb-two"} and result.baselines == {}
    assert result.errors["sb-add"].startswith("fixture red: the regression command fails on the unmodified snapshot")


def run_output(monkeypatch: pytest.MonkeyPatch, output: str) -> list[str]:
    commands: list[str] = []
    monkeypatch.setattr(evals.sandbox, "run_check",
                        lambda command, project, **kw: commands.append(command) or output)
    return commands


BASE = ["tests/lab/test_hq_v2.py::test_loop_rows"]


@pytest.mark.parametrize("output, verdict", [
    ("exit code: 0\n5 passed in 1s", ""),
    ("exit code: 1\nFAILED tests/lab/test_hq_v2.py::test_loop_rows\n1 failed, 5 passed in 1s", ""),
    (("exit code: 1\nFAILED tests/lab/test_hq_v2.py::test_loop_rows\nFAILED tests/test_tools.py::test_a - x\n"
      "2 failed, 4 passed in 1s"),
     "exit code: 1: 1 new failure(s) vs the unmodified snapshot (tests/test_tools.py::test_a)"),
    ("exit code: 2\nERROR tests/test_tools.py\n1 error in 1s",
     "exit code: 2: 1 new failure(s) vs the unmodified snapshot (tests/test_tools.py)"),
    # a clipped list (fewer names than the summary counts) or a timeout cannot show "no new": strict
    ("exit code: 1\nFAILED tests/lab/test_hq_v2.py::test_loop_rows\n3 failed, 5 passed in 1s",
     "exit code: 1: 3 failed, 5 passed in 1s · FAILED tests/lab/test_hq_v2.py::test_loop_rows"),
    ("Check timed out after 900 seconds", "could not run: Check timed out after 900 seconds"),
])
def test_regression_is_graded_as_no_new_failures_vs_the_snapshot(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output: str, verdict: str) -> None:
    commands = run_output(monkeypatch, output)
    assert evals.baseline_regression_check(f"{PYTEST} tests", tmp_path, 60, BASE) == verdict
    assert commands == [f"{PYTEST} tests -rfE --tb=no"]


def test_run_checks_uses_the_baseline_only_when_the_case_carries_one(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_output(monkeypatch, "exit code: 1\nFAILED tests/lab/test_hq_v2.py::test_loop_rows\n1 failed in 1s")
    case = {"regression_command": f"{PYTEST} tests"}
    assert evals.run_checks({**case, "regression_baseline": BASE}, tmp_path, "").regression == ""
    assert evals.run_checks(case, tmp_path, "").regression.startswith("regression_command exit code: 1")
    guarded = fixture_guard.GuardResult(baselines={"sb-add": BASE})
    assert evals.with_baseline(CASE, guarded)["regression_baseline"] == BASE
    assert evals.with_baseline({**CASE, "id": "sb-two"}, guarded) == {**CASE, "id": "sb-two"}


def test_the_result_is_cached_per_snapshot_so_the_guard_runs_once(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake(monkeypatch, {"status": "green", "count": "10 passed", "seconds": 300})
    first = guard([CASE])
    assert first.errors == {} and first.summary["status"] == "green"
    assert (home / "evals" / "fixture_guard.json").is_file()
    monkeypatch.setattr(fixture_guard, "check_case", lambda case, archive: pytest.fail("not cached"))
    second = guard([CASE])
    assert second.summary["cached"] and second.floors == {"sb-add": [0, 2]}
    assert len(calls) == 1
    monkeypatch.setenv(fixture_guard.GUARD_ENV, "refresh")
    monkeypatch.setattr(fixture_guard, "check_case", lambda case, archive: {"digest": "d", "floor": [1, 2]})
    assert guard([CASE]).floors == {"sb-add": [1, 2]} and len(calls) == 2


def test_the_night_runner_cache_location_wins(home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(fixture_guard.CACHE_ENV, str(tmp_path / "night" / "fixture_guard.json"))
    fake(monkeypatch, {"status": "green", "count": "", "seconds": 1})
    guard([CASE])
    assert (tmp_path / "night" / "fixture_guard.json").is_file()
    assert not (home / "evals" / "fixture_guard.json").exists()


def test_a_regression_run_over_the_bound_is_skipped_with_a_note_not_an_abort(
        home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake(monkeypatch, {"status": "timeout", "count": "Check timed out", "seconds": 900, "limit": 900})
    result = guard([CASE])
    assert result.errors == {}
    assert result.summary["status"] == "skipped"
    assert "skipped the regression check (timeout: over 900s)" in result.note()


def test_a_hidden_check_failing_on_the_correct_source_is_red(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake(monkeypatch, {"status": "green", "count": "", "seconds": 1}, {"unmutated": [1, 3], "floor": [0, 3]})
    error = guard([CASE]).errors["sb-add"]
    assert error == "fixture red: the hidden check fails on the unmutated source (2 of 3 hidden tests fail)"


def test_switched_off(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fixture_guard.GUARD_ENV, "0")
    monkeypatch.setattr(fixture_guard, "run_regression", lambda *a: pytest.fail("guard is off"))
    result = guard([CASE])
    assert result.errors == {} and result.summary["status"] == "off"
    assert guard([{"id": "plain"}]).summary is None  # no snapshot case: nothing to say


@pytest.mark.skipif(not _sandbox_works(), reason="working macOS sandbox-exec required")
@pytest.mark.skipif(not (evals.PROJECT_ROOT / ".git").exists(), reason="needs Vega's git checkout")
def test_end_to_end_on_a_real_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real fixtures and sandboxed pytest runs: a green snapshot gives the floor; a snapshot whose
    own test fails unmodified is red with that test named."""
    monkeypatch.setenv("VEGA_HOME", str(tmp_path / "home"))
    monkeypatch.delenv(fixture_guard.GUARD_ENV, raising=False)
    monkeypatch.delenv(fixture_guard.CACHE_ENV, raising=False)
    green = fixture_guard.check([CASE], make_archive(SOURCE), repo=evals.PROJECT_ROOT, paths=("pyproject.toml",),
                                exclude=())
    assert green.errors == {}, green.summary
    assert green.floors == {"sb-add": [1, 2]}  # add(0, 0) == 0 still passes with a - b
    broken = {**SOURCE, "tests/test_calc.py": SOURCE["tests/test_calc.py"] + "\ndef test_needs_checkout():\n"
              "    assert __import__('pathlib').Path('scripts/kpis.py').is_file()\n"}
    monkeypatch.setattr(fixture_guard, "snapshot_key", lambda *a, **k: ("other", "def5678"))
    touching = {**CASE, "expected_files": ["src/calc.py", "tests/test_calc.py"]}
    red = fixture_guard.check([touching], make_archive(broken), repo=evals.PROJECT_ROOT, paths=(), exclude=())
    assert red.errors["sb-add"] == ("fixture red: 1 tests in files this case touches fail on the unmodified "
                                    "snapshot (tests/test_calc.py::test_needs_checkout)")
    # the same red test, for a case that does not touch its file: the baseline, and grading a
    # fixture that fails only it passes its regression check
    other = fixture_guard.check([CASE], make_archive(broken), repo=evals.PROJECT_ROOT, paths=(), exclude=())
    assert other.errors == {} and other.baselines == {"sb-add": ["tests/test_calc.py::test_needs_checkout"]}
    project = tmp_path / "graded"
    evals.build_fixture({**CASE, "mutation": []}, project, make_archive(broken))
    graded = evals.with_baseline(CASE, other)
    assert evals.run_checks({**graded, "check_command": None}, project, "").regression == ""
    assert evals.run_checks({**CASE, "check_command": None}, project, "").regression
    cache = json.loads((tmp_path / "home" / "evals" / "fixture_guard.json").read_text())
    assert set(cache) == {"other", green.summary["key"]}
