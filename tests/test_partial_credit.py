"""Partial credit: the share of hidden tests passed per attempt, so a suite where nothing passes
outright still shows which variant gets closer."""

from __future__ import annotations

from pathlib import Path

import pytest

from vega import evals, sandbox
from vega.stats import bootstrap_mean_diff_ci

HIDDEN = ("from calc import add\n\n\ndef test_one():\n    assert add(1, 1) == 2\n\n\n"
          "def test_two():\n    assert add(2, 2) == 4\n\n\ndef test_three():\n    assert add(1, 2) == 4\n")
CASE = {"id": "c", "files": {"calc.py": "def add(a, b):\n    return a + b\n"},
        "hidden_files": {"hidden_tests/test_h.py": HIDDEN},
        "check_command": "python -m pytest -q -p no:cacheprovider hidden_tests"}


@pytest.mark.parametrize("output, counts", [
    ("exit code: 1\n..F\n1 failed, 2 passed in 0.01s", (2, 3)),
    ("exit code: 0\n...\n3 passed in 0.01s", (3, 3)),
    ("exit code: 2\nE   ImportError\n1 error in 0.02s", (0, 1)),
    ("exit code: 1\n= 2 failed, 1 passed, 1 error in 0.1s =", (1, 4)),
    ("exit code: 1\nTraceback (most recent call last):", None),
])
def test_pytest_counts(output: str, counts) -> None:
    assert evals.pytest_counts(output) == counts


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_hidden_counts_recorded_for_partial_and_collection_failures(tmp_path: Path) -> None:
    project = tmp_path / "p"
    evals.build_fixture(CASE, project, None)
    outcome = evals.run_checks(CASE, project, "")
    assert outcome.acceptance and outcome.hidden == (2, 3)
    broken = tmp_path / "q"
    evals.build_fixture({**CASE, "files": {"calc.py": "def plus(a, b):\n    return a + b\n"}},
                        broken, None)
    outcome = evals.run_checks(CASE, broken, "")
    assert outcome.hidden == (0, 3)  # failed to collect: every hidden test counts as failed


def test_hidden_check_runs_even_when_an_earlier_check_failed(tmp_path: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(evals, "_command_check",
                        lambda command, project, timeout: (calls.append(command) or "", (3, 3), None))
    project = tmp_path / "p"
    evals.build_fixture(CASE, project, None)
    outcome = evals.run_checks({**CASE, "file_contains": {"calc.py": ["nope"]}}, project, "")
    assert outcome.acceptance and calls and outcome.hidden == (3, 3)


def test_mean_diff_interval_resamples_cases() -> None:
    mean, low, high = bootstrap_mean_diff_ci([[(0.0, 0.5)], [(0.2, 0.6)], [(0.1, 0.4)]])
    assert mean == pytest.approx(0.4) and 0.29 < low <= mean <= high < 0.51
    assert bootstrap_mean_diff_ci([]) == (0.0, 0.0, 0.0)
    totals = evals.hidden_totals([{"attempts": [{"hidden": [2, 3]}, {"hidden": None}]},
                                  {"attempts": [{"hidden": [0, 4]}]}])
    assert totals == (2, 7, 2)


KINDS_OUTPUT = """exit code: 1
..F.
=========================== short test summary info ============================
PASSED hidden_tests/test_h.py::test_func_adds
PASSED hidden_tests/test_h.py::test_func_zero[0-0]
FAILED hidden_tests/test_h.py::test_sec_rejects[payload1] - AssertionError
PASSED hidden_tests/test_h.py::test_sec_rejects[payload2]
ERROR hidden_tests/test_h.py::test_sec_other - fixture 'x' not found
1 failed, 3 passed, 1 error in 0.02s"""


def test_pytest_kinds_split_functional_and_security_results() -> None:
    assert evals.pytest_kinds(KINDS_OUTPUT) == {"func": [2, 2], "sec": [1, 3]}
    assert evals.pytest_kinds("exit code: 0\n...\n3 passed in 0.01s") is None
    assert evals.pytest_kinds("PASSED t.py::test_other\n1 passed in 0.1s") is None


def test_pytest_kinds_count_kept_correct_versions_and_caught_mutants() -> None:
    output = ("exit code: 1\nPASSED h/test_g.py::test_keep_reference\n"
              "PASSED h/test_g.py::test_keep_alt_loop\nPASSED h/test_g.py::test_catch_empty\n"
              "FAILED h/test_g.py::test_catch_rounding - AssertionError\n"
              "PASSED h/test_g.py::test_catch_order\n1 failed, 4 passed in 0.3s")
    assert evals.pytest_kinds(output) == {"keep": [2, 2], "catch": [2, 3]}
    cases = [{"attempts": [{"hidden": [4, 5], "hidden_kinds": {"keep": [2, 2], "catch": [2, 3]}}]}]
    assert evals._hidden_line(cases) == ("Hidden tests: 4/5 passed (80%) over 1 attempts"
                                         " · mutants caught 2/3 (67%)"
                                         " · correct versions passed 2/2 (100%)")


def test_hidden_line_shows_functional_and_security_rates() -> None:
    cases = [{"attempts": [{"hidden": [3, 5], "hidden_kinds": {"func": [2, 2], "sec": [1, 3]}},
                           {"hidden": [1, 5], "hidden_kinds": {"func": [1, 2], "sec": [0, 3]}}]}]
    assert evals.hidden_kind_totals(cases) == {"func": [3, 4, 2], "sec": [1, 6, 2]}
    assert evals._hidden_line(cases) == ("Hidden tests: 4/10 passed (40%) over 2 attempts"
                                         " · functional 3/4 (75%) · security 1/6 (17%)")
    plain = [{"attempts": [{"hidden": [2, 3]}]}]
    assert evals._hidden_line(plain) == "Hidden tests: 2/3 passed (67%) over 1 attempts"


KIND_HIDDEN = ("from calc import add\n\n\ndef test_func_add():\n    assert add(1, 1) == 2\n\n\n"
               "def test_sec_big():\n    assert add(10**6, 1) == 0\n")
KIND_CASE = {"id": "k", "files": {"calc.py": "def add(a, b):\n    return a + b\n"},
             "hidden_files": {"hidden_tests/test_k.py": KIND_HIDDEN},
             "check_command": "python -m pytest -q -p no:cacheprovider -rfEp hidden_tests"}


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_hidden_kinds_recorded_for_runs_and_collection_failures(tmp_path: Path) -> None:
    project = tmp_path / "p"
    evals.build_fixture(KIND_CASE, project, None)
    outcome = evals.run_checks(KIND_CASE, project, "")
    assert outcome.hidden == (1, 2)
    assert outcome.hidden_kinds == {"func": [1, 1], "sec": [0, 1]}
    broken = tmp_path / "q"
    evals.build_fixture({**KIND_CASE, "files": {"calc.py": "def plus(a, b):\n    return 0\n"}},
                        broken, None)
    outcome = evals.run_checks(KIND_CASE, broken, "")
    assert outcome.hidden_kinds == {"func": [0, 1], "sec": [0, 1]}
