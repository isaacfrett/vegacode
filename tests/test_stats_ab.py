"""Statistics helpers, sampling overrides and the paired A/B runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from vega import evals, features
from vega.agent import sampling_overrides
from vega.stats import bootstrap_diff_ci, mcnemar_exact, rate_text, wilson


def test_wilson_interval_bounds() -> None:
    low, high = wilson(10, 60)
    assert 0.08 < low < 0.10 and 0.27 < high < 0.29
    assert wilson(0, 5)[0] == 0.0 and wilson(5, 5)[1] == 1.0
    assert rate_text(1, 2) == "1/2 (50%, 95% CI 9%–91%)"


def test_mcnemar_exact() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact(0, 10) < 0.01
    assert mcnemar_exact(3, 3) == 1.0


def test_bootstrap_is_deterministic_and_signed() -> None:
    better_b = [[(False, True)], [(False, True)], [(True, True)], [(False, False)]]
    low, high = bootstrap_diff_ci(better_b)
    assert 0 <= low <= high and bootstrap_diff_ci(better_b) == (low, high)


def test_sampling_overrides(monkeypatch) -> None:
    monkeypatch.delenv("VEGA_TEMPERATURE", raising=False)
    monkeypatch.delenv("VEGA_SEED", raising=False)
    assert sampling_overrides() == {}
    monkeypatch.setenv("VEGA_TEMPERATURE", "0")
    monkeypatch.setenv("VEGA_SEED", "42")
    assert sampling_overrides() == {"temperature": 0.0, "seed": 42}
    monkeypatch.setenv("VEGA_SEED", "x")
    with pytest.raises(ValueError):
        sampling_overrides()


@pytest.fixture
def two_cases(tmp_path: Path, monkeypatch) -> Path:
    cases = tmp_path / "fixes.jsonl"
    cases.write_text(
        '{"id": "a", "task": "t", "files": {"x.txt": "1"}, "response_contains": ["ok"]}\n'
        '{"id": "b", "task": "t", "files": {"x.txt": "1"}, "response_contains": ["ok"]}\n'
    )
    monkeypatch.setattr(evals, "suite_file", lambda suite: cases)
    monkeypatch.setattr(evals, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setattr(evals, "_git_commit", lambda: "abc1234")
    monkeypatch.delenv(features.ENV, raising=False)
    return tmp_path


def test_ab_switches_features_per_attempt_and_alternates_order(two_cases: Path, monkeypatch) -> None:
    seen: list[tuple[str, str]] = []

    def fake_attempt(case, **kwargs):
        off = features.disabled()
        seen.append((case["id"], "B" if "lint" in off else "A"))
        # B (lint off) passes everything; A passes only case "a".
        passed = "lint" in off or case["id"] == "a"
        return {"passed": passed, "detail": "" if passed else "x", "seconds": 1.0, "failure": ""}

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", ["lint"], repeat=2)
    assert seen == [("a", "A"), ("a", "B"), ("a", "B"), ("a", "A"),
                    ("b", "A"), ("b", "B"), ("b", "B"), ("b", "A")]
    assert record["order"] == "alternating"
    assert record["paired"] == {
        "pairs": 4, "a_only": 0, "b_only": 2, "both": 2, "neither": 0, "mcnemar_p": 0.5,
        "diff_b_minus_a": 0.5, "diff_ci": record["paired"]["diff_ci"],
    }
    assert record["variants"]["A"]["passed"] == 2 and record["variants"]["B"]["passed"] == 4
    assert features.ENV not in __import__("os").environ  # restored
    saved = json.loads((two_cases / "runs.jsonl").read_text())
    assert saved["mode"] == "ab" and saved["variants"]["B"]["features_off"] == ["lint"]


def test_ab_rejects_server_features_and_no_op(two_cases: Path) -> None:
    with pytest.raises(typer.BadParameter, match="server setting"):
        evals.run_ab("fixes", ["kv_q8"])
    with pytest.raises(typer.BadParameter, match="Unknown"):
        evals.run_ab("fixes", ["nope"])


def test_compare_refuses_ab_rows(two_cases: Path, monkeypatch) -> None:
    rows = [
        {"suite": "fixes", "model": "vega", "passed": 1, "total": 2, "cases": []},
        {"mode": "ab", "suite": "fixes", "model": "vega", "variants": {}},
    ]
    (two_cases / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(typer.BadParameter, match="paired A/B"):
        evals.compare_runs("1", "2")


@pytest.fixture
def twenty_cases(tmp_path: Path, monkeypatch) -> Path:
    cases = tmp_path / "fixes.jsonl"
    cases.write_text("".join(
        json.dumps({"id": f"c{i}", "task": "t", "files": {"x.txt": "1"}, "response_contains": ["ok"]}) + "\n"
        for i in range(20)))
    monkeypatch.setattr(evals, "suite_file", lambda suite: cases)
    monkeypatch.setattr(evals, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setattr(evals, "_git_commit", lambda: "abc1234")
    monkeypatch.setattr(evals, "_server_fingerprint", lambda model: {})
    monkeypatch.delenv(features.ENV, raising=False)
    return tmp_path


def test_early_stop_on_overwhelming_evidence(twenty_cases: Path, monkeypatch) -> None:
    runs: list[str] = []

    def fake_attempt(case, **kwargs):
        runs.append(case["id"])
        passed = "lint" in features.disabled()  # B always passes, A always fails
        return {"passed": passed, "detail": "" if passed else "x", "seconds": 1.0, "failure": ""}

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", ["lint"], repeat=1, early_stop=True)
    assert record["interim"]["stopped"] and record["interim"]["at_pairs"] == 10
    assert len(runs) == 20 and record["paired"]["pairs"] == 10


def test_no_early_stop_on_a_weak_effect(twenty_cases: Path, monkeypatch) -> None:
    def fake_attempt(case, **kwargs):
        b = "lint" in features.disabled()
        passed = b if case["id"] in {"c1", "c2", "c3"} else (case["id"] in {"c4", "c5"})
        return {"passed": passed, "detail": "" if passed else "x", "seconds": 1.0, "failure": ""}

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", ["lint"], repeat=1, early_stop=True)
    assert record["interim"]["stopped"] is False and record["paired"]["pairs"] == 20


def test_shards_merge_to_the_same_result_as_one_run(twenty_cases: Path, monkeypatch) -> None:
    def fake_attempt(case, **kwargs):
        index = int(case["id"][1:])
        b = "lint" in features.disabled()
        passed = (index % 3 == 0) or (b and index % 4 == 0)
        return {"passed": passed, "detail": "" if passed else "x", "seconds": 1.0, "failure": ""}

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    whole = evals.run_ab("fixes", ["lint"], repeat=1)
    monkeypatch.setenv("VEGA_EVAL_SHARD", "1/2")
    first = evals.run_ab("fixes", ["lint"], repeat=1)
    monkeypatch.setenv("VEGA_EVAL_SHARD", "2/2")
    second = evals.run_ab("fixes", ["lint"], repeat=1)
    monkeypatch.delenv("VEGA_EVAL_SHARD")
    assert first["shard"] == "1/2" and first["paired"]["pairs"] == 10
    merged = evals.merge_runs(["-2", "-1"])
    for key in ("pairs", "a_only", "b_only", "both", "neither", "mcnemar_p"):
        assert merged["paired"][key] == whole["paired"][key]
    assert merged["variants"]["B"]["passed"] == whole["variants"]["B"]["passed"]


def test_merge_refuses_unrelated_runs(twenty_cases: Path, monkeypatch) -> None:
    rows = [{"suite": "fixes", "model": "vega", "commit": "a", "repeat": 1, "features_off": [], "cases": []},
            {"suite": "goals", "model": "vega", "commit": "a", "repeat": 1, "features_off": [], "cases": []}]
    (twenty_cases / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(typer.BadParameter, match="suite"):
        evals.merge_runs(["1", "2"])


def test_ab_model_variant_runs_all_of_a_then_all_of_b(two_cases: Path, monkeypatch) -> None:
    seen: list[tuple[str, str]] = []
    monkeypatch.setenv("VEGA_MODEL", "vega")
    monkeypatch.setattr(evals, "_resolve_model", lambda value: f"vega-{value}")

    def fake_attempt(case, **kwargs):
        model = __import__("os").environ["VEGA_MODEL"]
        seen.append((case["id"], model))
        return {"passed": model == "vega-big", "detail": "", "seconds": 1.0, "failure": ""}

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", [], repeat=1, b_model="big")
    assert seen == [("a", "vega"), ("b", "vega"), ("a", "vega-big"), ("b", "vega-big")]
    assert record["order"] == "grouped"
    assert record["variants"]["B"]["model"] == "vega-big" and record["paired"]["b_only"] == 2
    assert __import__("os").environ["VEGA_MODEL"] == "vega"  # restored


def test_ab_can_switch_an_on_opt_in_feature_off_for_b(two_cases: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "constrained")  # e.g. the 7B profile default
    seen: list[str] = []

    def fake_attempt(case, **kwargs):
        seen.append("on" if features.on("constrained") else "off")
        return {"passed": True, "detail": "", "seconds": 1.0, "failure": ""}

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", ["constrained"], repeat=1)
    assert seen == ["on", "off", "on", "off"]
    assert record["variants"]["B"]["features_on"] == []


def test_times_line_reports_total_median_and_slowest() -> None:
    cases = [{"case": "a", "attempts": [{"seconds": 10.0}, {"seconds": 200.0}]},
             {"case": "b", "attempts": [{"seconds": 20.0}]}, {"case": "c", "attempts": []}]
    line = evals._times_line(cases)
    assert line.startswith("Time: 3.8 min total over 3 attempts · median 20.0 s")
    assert "slowest: a 3.3 min, b 20.0 s, a 10.0 s" in line
    assert evals._times_line([]) == "Time: not recorded"


def test_stop_switch_ends_the_ab_and_keeps_finished_cases(two_cases: Path, monkeypatch) -> None:
    calls = {"n": 0}

    def fake_attempt(case, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # after case a's pair, the user asks to stop
            evals.control_file("stop").write_text("stop\n")
        return {"passed": True, "detail": "", "seconds": 1.0, "failure": ""}

    evals.control_file("stop").parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", ["lint"], repeat=1, guard=None)
    assert record["stopped"].startswith("stopped by you")
    assert [c["case"] for c in record["variants"]["A"]["cases"]] == ["a"]
    assert not evals.control_file("stop").exists()
    line = json.loads(evals.RUNS_FILE.read_text().splitlines()[-1])
    assert line["stopped"] == record["stopped"]


def test_pause_switch_waits_until_resumed(monkeypatch) -> None:
    evals.control_file("pause").parent.mkdir(parents=True, exist_ok=True)
    evals.control_file("pause").write_text("paused\n")
    waits: list[float] = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) == 3:
            evals.control_file("pause").unlink()

    pacer = evals._Pacer(None, 0.0, sleep)
    assert pacer.before_attempt() == ""
    assert waits == [evals.PAUSE_POLL_SECONDS] * 3


def test_ctrl_c_keeps_finished_cases(two_cases: Path, monkeypatch) -> None:
    calls = {"n": 0}

    def fake_attempt(case, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return {"passed": True, "detail": "", "seconds": 1.0, "failure": ""}

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", ["lint"], repeat=1, guard=None)
    assert record["stopped"] == "interrupted; partial results kept"
    assert [c["case"] for c in record["variants"]["A"]["cases"]] == ["a"]


def test_single_shot_writes_the_code_block_and_is_checked(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    case = {"id": "he-x", "task": "Implement add().", "files": {"solution.py": "def add(a, b):\n    pass\n"},
            "python_check": "from solution import add\nassert add(2, 3) == 5\n"}
    reply = "Here it is:\n```python\ndef add(a, b):\n    return a + b\n```\n"

    class FakeAgent:
        model, sampling, messages = "m", {"max_tokens": 64}, []
        on_event = None
        def __init__(self) -> None:
            def create(**kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
            self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        def run(self, task):  # must not be called in single-shot mode
            raise AssertionError("agent loop used")

    monkeypatch.setenv("VEGA_EVAL_MODE", "single-shot")
    result = evals.run_attempt(case, agent_factory=lambda project, steps: FakeAgent())
    assert result["passed"], result["detail"]
