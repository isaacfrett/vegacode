"""VEGA_EVAL_REVEAL_REPRO (vega.evals): the oracle ablation that shows a case's hidden test as a
visible failing test. Off by default; on, fixture building writes tests/test_repro.py and the
run's record carries `oracle_reveal_repro: true`. Grading is unchanged."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vega import evals, features

HIDDEN = "def test_double():\n    from calc import double\n    assert double(2) == 4\n"
CASE = {"id": "a", "task": "t", "files": {"calc.py": "def double(x):\n    return x + 1\n",
                                          "tests/test_calc.py": "def test_ok():\n    assert True\n"},
        "hidden_files": {"hidden_tests/test_hidden_double.py": HIDDEN}}


def test_off_by_default_builds_the_usual_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(evals.REVEAL_REPRO_ENV, raising=False)
    assert not evals.reveal_repro()
    evals.build_fixture(CASE, tmp_path)
    assert sorted(p.name for p in (tmp_path / "tests").iterdir()) == ["test_calc.py"]
    assert not (tmp_path / "hidden_tests").exists()


@pytest.mark.parametrize("value", ["1", "true", "on", "yes"])
def test_on_writes_the_hidden_test_as_a_visible_repro(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      value: str) -> None:
    monkeypatch.setenv(evals.REVEAL_REPRO_ENV, value)
    evals.build_fixture(CASE, tmp_path)
    assert (tmp_path / "tests" / "test_repro.py").read_text(encoding="utf-8") == HIDDEN
    assert not (tmp_path / "hidden_tests").exists()  # the hidden copy still comes only at grading


def test_several_hidden_tests_keep_their_names(monkeypatch: pytest.MonkeyPatch) -> None:
    case = {"hidden_files": {"hidden_tests/test_one.py": "1", "hidden_tests/test_two.py": "2"}}
    assert evals.reveal_repro_files(case) == {"tests/test_repro_one.py": "1", "tests/test_repro_two.py": "2"}
    assert evals.reveal_repro_files({"files": {}}) == {}


def test_hidden_helpers_or_a_taken_name_are_fixture_errors(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(evals.REVEAL_REPRO_ENV, "1")
    helper = {**CASE, "hidden_files": {**CASE["hidden_files"], "hidden_tests/sim.py": "x = 1\n"}}
    with pytest.raises(evals.FixtureError, match="hidden helpers"):
        evals.build_fixture(helper, tmp_path / "one")
    taken = {**CASE, "files": {**CASE["files"], "tests/test_repro.py": "# the project's own\n"}}
    with pytest.raises(evals.FixtureError, match="already exists"):
        evals.build_fixture(taken, tmp_path / "two")


@pytest.fixture
def one_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cases = tmp_path / "fixes.jsonl"
    cases.write_text('{"id": "a", "task": "t", "files": {"x.txt": "1"}, "response_contains": ["ok"]}\n')
    monkeypatch.setattr(evals, "suite_file", lambda suite: cases)
    monkeypatch.setattr(evals, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setattr(evals, "_git_commit", lambda: "abc1234")
    monkeypatch.setattr(evals, "_server_fingerprint", lambda model: {})
    monkeypatch.delenv(features.ENV_ON, raising=False)

    def attempt(case: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {"passed": True, "detail": "", "seconds": 1.0, "failure": ""}

    monkeypatch.setattr(evals, "run_attempt", attempt)
    return tmp_path


def test_the_record_is_tagged_only_when_on(one_case: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(evals.REVEAL_REPRO_ENV, raising=False)
    assert evals.REVEAL_REPRO_TAG not in evals.run_suite("fixes", repeat=1)
    monkeypatch.setenv(evals.REVEAL_REPRO_ENV, "1")
    assert evals.run_suite("fixes", repeat=1)[evals.REVEAL_REPRO_TAG] is True
