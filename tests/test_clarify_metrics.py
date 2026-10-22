"""Clarify suite support: a vague request must be met with a question, and asking is measured
as precision and recall."""

from __future__ import annotations

from pathlib import Path

from vega import evals

CASE = {"id": "ask-x", "expect": "ask", "files": {"app.py": "x = 1\n"}}


def test_an_ask_case_needs_a_question_and_no_new_files(tmp_path: Path) -> None:
    evals.build_fixture(CASE, tmp_path, None)
    assert evals._check(CASE, tmp_path, "Done, I added CSV export.") == "did not ask before building"
    assert evals._check(CASE, tmp_path, "Which format should the export use: CSV or JSON?") == ""
    (tmp_path / "export.py").write_text("")
    assert evals._check(CASE, tmp_path, "Which format: CSV or JSON?").startswith(
        "created files before asking")


def test_clarify_metrics_precision_and_recall() -> None:
    attempts = [{"expect": "ask", "asked": True}, {"expect": "ask", "asked": False},
                {"expect": "act", "asked": True}, {"expect": "act", "asked": False},
                {"expect": "act", "asked": False}]
    m = evals.clarify_metrics(attempts)
    assert (m["asked_on_vague"], m["vague"], m["asked_on_clear"], m["clear"]) == (1, 2, 1, 3)
    assert m["ask_recall"] == 0.5 and m["false_question_rate"] == 0.3333 and m["ask_precision"] == 0.5
    assert "precision of asking 50%" in evals._clarify_line({"clarify": m})
    assert "clarify" not in evals.summarise([{"passed": True}])


def test_consistency_groups_rewordings() -> None:
    cases = [{"id": "a1", "variant_of": "a"}, {"id": "a2", "variant_of": "a"},
             {"id": "b1", "variant_of": "b"}, {"id": "b2", "variant_of": "b"}, {"id": "c"}]
    records = [{"passed": 1, "attempts": [{}]}, {"passed": 0, "attempts": [{}]},
               {"passed": 1, "attempts": [{}]}, {"passed": 1, "attempts": [{}]},
               {"passed": 1, "attempts": [{}]}]
    assert evals.consistency_metrics(cases, records) == {
        "tasks": 2, "consistent": 1, "robust": 1, "passed_some_wording": 2}
    assert evals.consistency_metrics([{"id": "c"}], [{"passed": 1, "attempts": [{}]}]) is None
