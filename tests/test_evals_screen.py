"""Eval speed-ups: model A/Bs grouped by model, the screening tier and futility stopping."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from rich.console import Console

from vega import evals, features, lab


def _write_cases(path: Path, ids: list[str]) -> Path:
    path.write_text("".join(
        json.dumps({"id": case_id, "task": "t", "files": {"x.txt": "1"},
                    "response_contains": ["ok"]}) + "\n" for case_id in ids))
    return path


@pytest.fixture
def suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A fake suite whose case ids the test chooses; runs are recorded in a temp runs.jsonl."""
    def make(ids: list[str]) -> Path:
        cases = _write_cases(tmp_path / "fixes.jsonl", ids)
        monkeypatch.setattr(evals, "suite_file", lambda name: cases)
        return cases

    monkeypatch.setattr(evals, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setattr(evals, "_git_commit", lambda: "abc1234")
    monkeypatch.setattr(evals, "_server_fingerprint", lambda model: {})
    monkeypatch.setattr(evals, "_resolve_model", lambda value: f"vega-{value}")
    monkeypatch.setattr(evals, "console", Console(record=True, width=200))
    monkeypatch.setenv("VEGA_MODEL", "vega")
    monkeypatch.delenv(features.ENV, raising=False)
    monkeypatch.delenv("VEGA_EVAL_SHARD", raising=False)
    evals.control_file("stop").parent.mkdir(parents=True, exist_ok=True)
    evals.control_file("stop").unlink(missing_ok=True)
    return make


def _result(passed: bool) -> dict:
    return {"passed": passed, "detail": "" if passed else "x", "seconds": 1.0, "failure": ""}


# --- Model comparisons grouped by model ---------------------------------------------------------

def test_model_ab_runs_every_a_attempt_before_any_b_and_pairs_by_case(suite, monkeypatch) -> None:
    suite(["a", "b", "c"])
    seen: list[tuple[str, int, str]] = []
    # A passes a and b; B passes b and c, so a is only-A and c is only-B on both attempts.
    passes = {"vega": {"a", "b"}, "vega-big": {"b", "c"}}

    def fake_attempt(case, transcript, **kwargs):
        model = os.environ["VEGA_MODEL"]
        seen.append((case["id"], int(transcript.stem.split("-")[1]), model))
        return _result(case["id"] in passes[model])

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", [], repeat=2, b_model="big")
    models = [model for _, _, model in seen]
    assert models == ["vega"] * 6 + ["vega-big"] * 6  # one switch: the model loads once per side
    order = [(case, index) for case, index, _ in seen]
    assert order[:6] == order[6:] == [("a", 1), ("a", 2), ("b", 1), ("b", 2), ("c", 1), ("c", 2)]
    assert record["order"] == "grouped"
    assert record["paired"]["pairs"] == 6
    assert (record["paired"]["a_only"], record["paired"]["b_only"]) == (2, 2)
    assert record["paired"]["both"] == 2 and record["paired"]["neither"] == 0
    assert [c["case"] for c in record["variants"]["A"]["cases"]] == ["a", "b", "c"]
    assert [c["passed"] for c in record["variants"]["B"]["cases"]] == [0, 2, 2]
    assert "unpaired" not in record and "stopped" not in record
    assert os.environ["VEGA_MODEL"] == "vega"  # restored


def test_stop_during_the_a_pass_keeps_a_results_unpaired(suite, monkeypatch) -> None:
    suite(["a", "b", "c"])
    calls = {"n": 0}

    def fake_attempt(case, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # A has finished a and b; the user asks to stop
            evals.control_file("stop").write_text("stop\n")
        return _result(True)

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", [], repeat=1, b_model="big")
    assert record["stopped"].startswith("stopped by you")
    assert record["paired"]["pairs"] == 0
    assert record["variants"]["A"]["cases"] == [] and record["variants"]["B"]["cases"] == []
    assert [c["case"] for c in record["unpaired"]["A"]] == ["a", "b"]
    assert record["unpaired"]["A"][0]["attempts"][0]["passed"] is True
    saved = json.loads(evals.RUNS_FILE.read_text().splitlines()[-1])
    assert [c["case"] for c in saved["unpaired"]["A"]] == ["a", "b"]


def test_stop_during_the_b_pass_pairs_only_cases_both_sides_ran(suite, monkeypatch) -> None:
    suite(["a", "b", "c"])
    calls = {"n": 0}

    def fake_attempt(case, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:  # all of A, then B on case a
            evals.control_file("stop").write_text("stop\n")
        return _result(os.environ["VEGA_MODEL"] == "vega-big")

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", [], repeat=1, b_model="big")
    assert record["paired"]["pairs"] == 1 and record["paired"]["b_only"] == 1
    assert [c["case"] for c in record["variants"]["B"]["cases"]] == ["a"]
    assert [c["case"] for c in record["unpaired"]["A"]] == ["b", "c"]


def test_ctrl_c_during_the_a_pass_keeps_finished_a_cases(suite, monkeypatch) -> None:
    suite(["a", "b", "c"])
    calls = {"n": 0}

    def fake_attempt(case, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return _result(True)

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", [], repeat=1, b_model="big")
    assert record["stopped"] == "interrupted; partial results kept"
    assert [c["case"] for c in record["unpaired"]["A"]] == ["a", "b"]
    assert record["paired"]["pairs"] == 0


def test_same_model_ab_still_alternates(suite, monkeypatch) -> None:
    suite(["a", "b"])
    seen: list[tuple[str, str]] = []

    def fake_attempt(case, **kwargs):
        seen.append((case["id"], "B" if "lint" in features.disabled() else "A"))
        return _result(True)

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", ["lint"], repeat=2)
    assert seen == [("a", "A"), ("a", "B"), ("a", "B"), ("a", "A"),
                    ("b", "A"), ("b", "B"), ("b", "B"), ("b", "A")]
    assert record["order"] == "alternating"


def test_grouped_efficacy_look_happens_in_the_b_pass(suite, monkeypatch) -> None:
    suite([f"c{i}" for i in range(20)])
    models: list[str] = []

    def fake_attempt(case, **kwargs):
        models.append(os.environ["VEGA_MODEL"])
        return _result(os.environ["VEGA_MODEL"] == "vega-big")  # B always passes, A never

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", [], repeat=1, b_model="big", early_stop=True)
    assert record["interim"]["stopped"] and record["interim"]["at_pairs"] == 10
    assert models.count("vega") == 20 and models.count("vega-big") == 10
    assert record["paired"]["pairs"] == 10 and len(record["unpaired"]["A"]) == 10
    assert "futility" not in record["interim"]  # futility is off outside screens


# --- Discriminating cases and the screen ---------------------------------------------------------

def _ab_run(suite_name: str, a: dict[str, list[bool]], b: dict[str, list[bool]]) -> dict:
    def cases(results: dict[str, list[bool]]) -> list[dict]:
        return [{"case": case_id, "passed": sum(values),
                 "attempts": [{"passed": value} for value in values]}
                for case_id, values in results.items()]
    return {"mode": "ab", "suite": suite_name,
            "variants": {"A": {"cases": cases(a)}, "B": {"cases": cases(b)}}}


def _single_run(suite_name: str | None, results: dict[str, list[bool]]) -> dict:
    run: dict = {"cases": [{"case": case_id, "passed": sum(values),
                            "attempts": [{"passed": value} for value in values]}
                           for case_id, values in results.items()]}
    if suite_name:
        run["suite"] = suite_name
    return run


RUNS_FIXTURE = [
    # goals A/B: g2 differs on its second pair; g1 always agrees; g3 is only in A (no pair)
    _ab_run("goals", {"g1": [True, True], "g2": [True, True], "g3": [False]},
            {"g1": [True, True], "g2": [True, False]}),
    # goals single runs: g4 passes in one run and fails in another; g5 always fails
    _single_run("goals", {"g4": [True], "g5": [False]}),
    _single_run("goals", {"g4": [False], "g5": [False]}),
    # flaky within one run counts too
    _single_run("goals", {"g6": [True, False]}),
    # a fixture error is not a result
    {"suite": "goals", "cases": [{"case": "g7", "passed": 0, "attempts": [], "error": "boom"}]},
    # the earliest fixes records: no suite field, one inline attempt per case
    {"cases": [{"case": "f1", "passed": True, "detail": "", "seconds": 1}]},
    {"cases": [{"case": "f1", "passed": False, "detail": "x", "seconds": 1}]},
    # another suite's differences do not leak in
    _ab_run("mbpp40", {"g5": [True]}, {"g5": [False]}),
]


def test_discriminating_cases_from_a_runs_fixture(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    path.write_text("".join(json.dumps(run) + "\n" for run in RUNS_FIXTURE))
    runs = evals.load_runs(path)
    assert evals.discriminating_cases("goals", runs) == ["g2", "g4", "g6"]
    assert evals.discriminating_cases("fixes", runs) == ["f1"]
    assert evals.discriminating_cases("mbpp40", runs) == ["g5"]
    assert evals.discriminating_cases("humaneval40", runs) == []


def test_screen_selection_tops_up_with_a_fixed_sample() -> None:
    cases = [{"id": f"c{i:02d}"} for i in range(30)]
    runs = [_ab_run("fixes", {"c05": [True], "c20": [True], "gone": [True]},
                    {"c05": [False], "c20": [False], "gone": [False]})]
    chosen, discriminating = evals.screen_selection("fixes", cases, 8, runs)
    ids = [case["id"] for case in chosen]
    assert discriminating == 2 and len(ids) == 8 and {"c05", "c20"} <= set(ids)
    assert ids == sorted(ids)  # suite order kept
    again, _ = evals.screen_selection("fixes", cases, 8, runs)
    assert again == chosen  # the same sample every time
    # enough discriminating cases: only those run
    many = [_ab_run("fixes", {f"c{i:02d}": [True] for i in range(10)},
                    {f"c{i:02d}": [False] for i in range(10)})]
    chosen, discriminating = evals.screen_selection("fixes", cases, 8, many)
    assert discriminating == 10 and [case["id"] for case in chosen] == [f"c{i:02d}" for i in range(10)]
    # a suite smaller than the minimum runs whole
    assert len(evals.screen_selection("fixes", cases[:5], 8, [])[0]) == 5


def test_screen_run_is_marked_and_says_it_is_not_a_full_result(suite, monkeypatch) -> None:
    suite([f"c{i:02d}" for i in range(20)])
    evals.RUNS_FILE.write_text(json.dumps(_single_run("fixes", {"c03": [True, False]})) + "\n")
    ran: list[str] = []

    def fake_attempt(case, **kwargs):
        ran.append(case["id"])
        return _result(True)

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_suite("fixes", repeat=1, screen=True, screen_min=4)
    assert len(ran) == 4 and "c03" in ran
    assert record["screen"] is True and record["subset"] is True
    text = evals.console.export_text()
    assert "screen: not a full result" in text and "Screen of fixes: 4 of 20 cases" in text
    assert lab.verdict(record).startswith("screen, not a full result")
    full = evals.run_suite("fixes", repeat=1)
    assert "screen" not in full and full["subset"] is False and len(ran) == 24


# --- Futility stopping ----------------------------------------------------------------------------

def test_futility_best_p_pushes_the_side_already_ahead() -> None:
    from vega.stats import mcnemar_exact

    assert evals.futility_best_p(0, 0, 4) == pytest.approx(0.125)  # 0 vs 4: cannot reach 0.05
    assert evals.futility_best_p(0, 0, 6) == pytest.approx(0.03125)  # 0 vs 6 can
    assert evals.futility_best_p(0, 2, 4) == mcnemar_exact(0, 6)
    assert evals.futility_best_p(2, 0, 4) == mcnemar_exact(6, 0)  # A ahead: A's way is best
    assert evals.futility_best_p(3, 3, 4) == pytest.approx(0.34375)
    assert evals.futility_best_p(3, 3, 10) < 0.05


def _ab_by_feature(passes_a: set[str], passes_b: set[str]):
    def fake_attempt(case, **kwargs):
        b = "lint" in features.disabled()
        return _result(case["id"] in (passes_b if b else passes_a))
    return fake_attempt


def test_screen_stops_for_futility_when_no_outcome_could_be_significant(suite, monkeypatch) -> None:
    ids = [f"c{i}" for i in range(8)]
    suite(ids)
    monkeypatch.setattr(evals, "run_attempt", _ab_by_feature(set(ids), set(ids)))  # all agree
    record = evals.run_ab("fixes", ["lint"], repeat=1, screen=True)
    assert record["stopped"] == "futility" and record["screen"] is True and record["subset"]
    assert record["interim"]["at_pairs"] == 4 and record["interim"]["futility"] is True
    assert record["interim"]["best_possible_p"] == 0.125
    assert record["paired"]["pairs"] == 4
    text = evals.console.export_text()
    assert "Stopped for futility at 4 of 8 pairs" in text and "screen: not a full result" in text
    assert "stopped for futility" in lab.verdict(record)


def test_screen_does_not_stop_when_significance_is_still_possible(suite, monkeypatch) -> None:
    ids = [f"c{i}" for i in range(8)]
    suite(ids)
    monkeypatch.setattr(evals, "run_attempt", _ab_by_feature(set(), set(ids)))  # B wins every pair
    record = evals.run_ab("fixes", ["lint"], repeat=1, screen=True)
    assert record["interim"]["futility"] is False and record["interim"]["only_b"] == 4
    assert "stopped" not in record and record["paired"]["pairs"] == 8


def test_full_runs_never_stop_for_futility(suite, monkeypatch) -> None:
    ids = [f"c{i}" for i in range(8)]
    suite(ids)
    monkeypatch.setattr(evals, "run_attempt", _ab_by_feature(set(ids), set(ids)))
    record = evals.run_ab("fixes", ["lint"], repeat=1)
    assert "interim" not in record and "stopped" not in record and record["paired"]["pairs"] == 8
    forced = evals.run_ab("fixes", ["lint"], repeat=1, futility=True)
    assert forced["stopped"] == "futility" and "screen" not in forced


def test_screened_model_ab_takes_the_futility_look_in_the_b_pass(suite, monkeypatch) -> None:
    suite([f"c{i}" for i in range(8)])
    models: list[str] = []

    def fake_attempt(case, **kwargs):
        models.append(os.environ["VEGA_MODEL"])
        return _result(True)

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    record = evals.run_ab("fixes", [], repeat=1, b_model="big", screen=True)
    assert record["stopped"] == "futility" and record["paired"]["pairs"] == 4
    assert models == ["vega"] * 8 + ["vega-big"] * 4


# --- Lab ------------------------------------------------------------------------------------------

def test_lab_screen_field_and_futility_does_not_stop_the_queue(tmp_path: Path, monkeypatch) -> None:
    toml = tmp_path / "experiments.toml"
    toml.write_text('[[experiment]]\nid = "one"\nhypothesis = "h"\nab = ["lint"]\nscreen = true\n'
                    '[[experiment]]\nid = "two"\nhypothesis = "h"\nab = ["lint"]\n')
    monkeypatch.setattr(evals, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setattr(lab, "write_scoreboard", lambda: None)
    calls: list[tuple[str, bool]] = []

    def fake_run_ab(suite_name, b_off, **kwargs):
        calls.append((os.environ["VEGA_EXPERIMENT"], kwargs["screen"]))
        return {"stopped": "futility"}

    monkeypatch.setattr(evals, "run_ab", fake_run_ab)
    assert lab.load(toml)[0].screen is True and lab.load(toml)[1].screen is False
    lab.run_queue(path=toml)
    assert calls == [("one", True), ("two", False)]
    calls.clear()
    lab.run_queue(["two"], path=toml, screen=True)
    assert calls == [("two", True)]


def test_a_screen_is_not_an_experiments_result() -> None:
    full = {"experiment": "x", "timestamp": "2026-10-15T01:00", "passed": 1, "total": 2}
    screen = {**full, "timestamp": "2026-10-15T02:00", "screen": True, "subset": True}
    assert lab.results([full, screen])["x"] is full
    assert lab.results([screen]) == {}
    assert lab.results([full, screen], screens=True)["x"] is screen


def test_screen_keeps_the_cases_that_disagree_most_often() -> None:
    def ab(results):  # results: {case: (a_passed, b_passed)}
        return {"mode": "ab", "suite": "s", "variants": {
            name: {"cases": [{"case": c, "passed": int(r[i]), "attempts": [{"passed": r[i]}]}
                             for c, r in results.items()]} for i, name in enumerate("AB")}}

    runs = [ab({"often": (True, False), "once": (True, False), "never": (True, True)}),
            ab({"often": (False, True), "once": (True, True), "never": (True, True)}),
            ab({"often": (True, False), "once": (True, True), "never": (True, True)})]
    rates = evals.disagreement_rates("s", runs)
    assert rates["often"] == (3, 3) and rates["once"] == (1, 3) and rates["never"] == (0, 3)
    cases = [{"id": c} for c in ("never", "once", "often")]
    chosen, discriminating = evals.screen_selection("s", cases, minimum=1, runs=runs, maximum=1)
    assert [c["id"] for c in chosen] == ["often"] and discriminating == 1
