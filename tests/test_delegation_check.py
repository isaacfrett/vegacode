"""Delegation grading (delegate suite): the briefs an attempt records and `require_delegation` /
`delegate_brief_matches`."""

from __future__ import annotations

from vega import evals

CASE = {"id": "c", "files": {}, "require_delegation": True,
        "delegate_brief_matches": [r"legacy_auth", r"adapters?/"]}


def test_delegation_passes_when_a_brief_names_every_pattern(tmp_path):
    briefs = ["List every file under adapters/ that still calls legacy_auth()."]
    assert evals.run_checks(CASE, tmp_path, "", delegations=briefs).passed


def test_patterns_may_match_different_briefs(tmp_path):
    briefs = ["Which modules in adapters/ log in?", "Where is LEGACY_AUTH defined?"]
    assert evals.run_checks(CASE, tmp_path, "", delegations=briefs).passed


def test_no_delegation_fails(tmp_path):
    outcome = evals.run_checks(CASE, tmp_path, "", delegations=[])
    assert "did not delegate" in outcome.acceptance


def test_a_vague_brief_fails_the_patterns(tmp_path):
    outcome = evals.run_checks(CASE, tmp_path, "", delegations=["look around the code"])
    assert "no delegate_task brief matches 'legacy_auth'" in outcome.acceptance


def test_unknown_record_and_unrelated_cases_are_not_graded(tmp_path):
    assert evals.run_checks(CASE, tmp_path, "").passed  # external agent / validation
    assert evals.run_checks({"id": "d", "files": {}}, tmp_path, "", delegations=[]).passed
    only_required = {"id": "e", "files": {}, "require_delegation": True}
    assert evals.run_checks(only_required, tmp_path, "", delegations=["anything"]).passed
    assert not evals.run_checks(only_required, tmp_path, "", delegations=[]).passed


def test_meter_records_delegate_briefs_in_order():
    meter = evals._Meter()
    for name, args in (("delegate_task", {"task": "find A"}), ("read_file", {"path": "x"}),
                       ("delegate_task", {"task": "find B"}), ("delegate_task", "not a dict")):
        meter.on_event("tool_call", {"name": name, "args": args})
    assert meter.delegations == ["find A", "find B"]
    assert meter.tool_calls["delegate_task"] == 3
