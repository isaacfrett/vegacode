"""Experiment queue: loading, tagging, running and the scoreboard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vega import evals, features, lab

EXPERIMENTS = """
[[experiment]]
id = "lint"
hypothesis = "lint helps"
ab = ["lint"]

[[experiment]]
id = "baseline"
hypothesis = "plain run"
"""


@pytest.fixture
def setup(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / "experiments.toml").write_text(EXPERIMENTS)
    monkeypatch.setattr(lab, "EXPERIMENTS_FILE", tmp_path / "experiments.toml")
    monkeypatch.setattr(lab, "SCOREBOARD_FILE", tmp_path / "scoreboard.md")
    monkeypatch.setattr(evals, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.delenv("VEGA_EXPERIMENT", raising=False)
    monkeypatch.delenv(features.ENV, raising=False)
    return tmp_path


@pytest.mark.skipif(not lab.EXPERIMENTS_FILE.exists(), reason="eval data not present")
def test_repo_experiments_file_is_valid() -> None:
    ids = [experiment.id for experiment in lab.load()]
    assert ids and len(ids) == len(set(ids))
    for experiment in lab.load():
        assert set(experiment.ab) <= set(features.FEATURES) - {"kv_q8"}


def test_queue_runs_pending_experiments_tagged_and_writes_scoreboard(setup: Path, monkeypatch) -> None:
    calls: list[tuple] = []

    def fake_ab(suite, b_off, **kwargs):
        calls.append(("ab", suite, tuple(b_off), __import__("os").environ.get("VEGA_EXPERIMENT")))
        record = {"timestamp": "2026-10-13T20:00:00", "mode": "ab", "suite": suite,
                  "experiment": __import__("os").environ.get("VEGA_EXPERIMENT"),
                  "paired": {"pairs": 10, "a_only": 0, "b_only": 6, "both": 2, "neither": 2,
                             "mcnemar_p": 0.03, "diff_b_minus_a": 0.6, "diff_ci": [0.2, 0.9]},
                  "variants": {name: {"summary": {"median_seconds": 10.0, "failure_classes": {"no_edit": 1}}}
                               for name in ("A", "B")}}
        with evals.RUNS_FILE.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    def fake_suite(suite, **kwargs):
        calls.append(("suite", suite, (), __import__("os").environ.get("VEGA_EXPERIMENT")))
        record = {"timestamp": "2026-10-13T20:01:00", "suite": suite, "passed": 3, "total": 10,
                  "experiment": __import__("os").environ.get("VEGA_EXPERIMENT"), "summary": {}}
        with evals.RUNS_FILE.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    monkeypatch.setattr(evals, "run_ab", fake_ab)
    monkeypatch.setattr(evals, "run_suite", fake_suite)
    lab.run_queue()
    assert calls == [("ab", "fixes", ("lint",), "lint"), ("suite", "fixes", (), "baseline")]
    assert "VEGA_EXPERIMENT" not in __import__("os").environ
    board = (setup / "scoreboard.md").read_text()
    assert "B (feature off) better" in board and "3/10" in board
    lab.run_queue()  # nothing pending now
    assert len(calls) == 2


def test_tag_attaches_experiment_to_a_recorded_run(setup: Path) -> None:
    (setup / "runs.jsonl").write_text(json.dumps({"timestamp": "t1", "suite": "fixes"}) + "\n")
    lab.tag("1", "lint")
    assert json.loads((setup / "runs.jsonl").read_text())["experiment"] == "lint"
    with pytest.raises(ValueError):
        lab.tag("1", "nope")
