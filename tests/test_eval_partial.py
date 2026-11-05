"""A stopped `vega eval` keeps the cases it finished (2026-10-28: a paused research step lost two
finished bigbuild cases, ~5 h, because the record was only written at the end)."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import pytest
from rich.console import Console

from vega import evals, server

CASES = [{"id": c, "task": "t", "files": {"x.txt": "1"}, "response_contains": ["ok"]} for c in ("a", "b", "c")]


@pytest.fixture
def suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cases = tmp_path / "fixes.jsonl"
    cases.write_text("".join(json.dumps(c) + "\n" for c in CASES))
    monkeypatch.setattr(evals, "suite_file", lambda name: cases)
    monkeypatch.setattr(evals, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setattr(evals, "_git_commit", lambda: "abc1234")
    monkeypatch.setattr(evals, "console", Console(record=True, width=160))
    monkeypatch.setattr(server, "STATE_DIR", tmp_path / "state")  # vega lab pause/stop switches
    monkeypatch.setenv(evals.PARTIAL_ENV, str(tmp_path / "partial" / "step.json"))
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    restore = evals.install_stop_signal()
    yield tmp_path
    restore()
    assert {sig: signal.getsignal(sig) for sig in saved} == saved
    evals._STOP_SIGNAL.clear()


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_sigterm_after_two_cases_writes_a_stopped_record_of_two(suite: Path, monkeypatch) -> None:
    seen: list[str] = []
    sidecar = suite / "partial" / "step.json"

    def attempt(case, **kwargs):
        seen.append(case["id"])
        if case["id"] == "b":  # case a is finished: the sidecar holds it
            assert [c["case"] for c in json.loads(sidecar.read_text())["cases"]] == ["a"]
        if case["id"] == "c":
            try:
                os.kill(os.getpid(), signal.SIGTERM)  # the research runner stops the step
            except KeyboardInterrupt:
                os.kill(os.getpid(), signal.SIGTERM)  # and signals the process too: ignored
                raise
        return {"passed": True, "detail": "", "seconds": 1.0, "failure": ""}

    monkeypatch.setattr(evals, "run_attempt", attempt)
    record = evals.run_suite("fixes", repeat=1)
    assert seen == ["a", "b", "c"]
    assert [c["case"] for c in record["cases"]] == ["a", "b"]  # the case in progress is dropped
    assert record["stopped"].startswith("interrupted (SIGTERM)") and record["partial"] is True
    assert (record["passed"], record["total"]) == (2, 2)
    assert records(suite / "runs.jsonl") == [record]  # one record for the run
    assert not sidecar.exists()  # never both: the record holds the cases now


def test_an_attempt_that_swallows_the_interrupt_is_not_kept(suite: Path, monkeypatch) -> None:
    def attempt(case, **kwargs):
        if case["id"] == "b":
            try:
                os.kill(os.getpid(), signal.SIGINT)
            except KeyboardInterrupt:  # e.g. an agent loop that treats Ctrl-C as "stop this turn"
                pass
            return {"passed": False, "detail": "interrupted", "seconds": 1.0, "failure": ""}
        return {"passed": True, "detail": "", "seconds": 1.0, "failure": ""}

    monkeypatch.setattr(evals, "run_attempt", attempt)
    record = evals.run_suite("fixes", repeat=1)
    assert [c["case"] for c in record["cases"]] == ["a"] and record["stopped"].startswith("interrupted (SIGINT)")


def test_killed_outright_leaves_the_sidecar_of_finished_cases(suite: Path, monkeypatch) -> None:
    """SIGKILL cannot be caught: the sidecar, rewritten after every case, is what is left."""
    class Killed(BaseException):
        pass

    def attempt(case, **kwargs):
        if case["id"] == "c":
            raise Killed  # the process dies here: nothing after this line of run_suite runs
        return {"passed": case["id"] == "a", "detail": "", "seconds": 1.0, "failure": ""}

    monkeypatch.setattr(evals, "run_attempt", attempt)
    with pytest.raises(Killed):
        evals.run_suite("fixes", repeat=1)
    assert not (suite / "runs.jsonl").exists()
    side = json.loads((suite / "partial" / "step.json").read_text())
    assert [c["case"] for c in side["cases"]] == ["a", "b"] and (side["passed"], side["total"]) == (1, 2)
    assert side["partial"] is True and side["stopped"] == evals.SIDECAR_STOPPED


def test_a_finished_run_writes_one_final_record_and_no_sidecar(suite: Path, monkeypatch) -> None:
    monkeypatch.setattr(evals, "run_attempt",
                        lambda case, **kw: {"passed": True, "detail": "", "seconds": 1.0, "failure": ""})
    record = evals.run_suite("fixes", repeat=1)
    assert [c["case"] for c in record["cases"]] == ["a", "b", "c"]
    assert "stopped" not in record and "partial" not in record
    assert records(suite / "runs.jsonl") == [record] and not (suite / "partial" / "step.json").exists()


def test_no_sidecar_without_the_env(suite: Path, monkeypatch) -> None:
    monkeypatch.delenv(evals.PARTIAL_ENV)
    monkeypatch.setattr(evals, "run_attempt",
                        lambda case, **kw: {"passed": True, "detail": "", "seconds": 1.0, "failure": ""})
    evals.run_suite("fixes", repeat=1)
    assert not (suite / "partial").exists()
