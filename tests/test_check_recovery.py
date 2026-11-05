"""Execution-feedback grading: the check trace an attempt records and `require_check_recovery`."""

from __future__ import annotations

from vega import evals


def test_check_result_reads_sandbox_and_pipeline_summaries():
    assert evals.check_result("exit code: 0\n3 passed") == "P"
    assert evals.check_result("exit code: 1\nFAILED test_x") == "F"
    assert evals.check_result("Check timed out after 60s.") == "F"
    assert evals.check_result("4/4 checks passed") == "P"
    assert evals.check_result("3/4 checks passed\n- example 2") == "F"
    assert evals.check_result("2 project test(s) failing, 1 spec test(s) failing") == "F"
    assert evals.check_result("all tests pass") == "P"
    # did not run: refused, repeated, no shell
    assert evals.check_result("run_check runs one program without a shell, so `|` ...") == ""
    assert evals.check_result("(Vega harness) Not run again: you already made this call") == ""


def test_recovered_needs_a_pass_after_a_failure():
    assert evals.recovered("FP")
    assert evals.recovered("PFFP")
    assert not evals.recovered("PF")
    assert not evals.recovered("PPP")
    assert not evals.recovered("")
    assert not evals.recovered(None)


def test_require_check_recovery_grades_the_trace(tmp_path):
    case = {"id": "c", "files": {}, "require_check_recovery": True}
    assert evals.run_checks(case, tmp_path, "", check_trace="FFP").passed
    no_failure = evals.run_checks(case, tmp_path, "", check_trace="P")
    assert "no failing check" in no_failure.acceptance
    still_failing = evals.run_checks(case, tmp_path, "", check_trace="PF")
    assert "no passing check after" in still_failing.acceptance
    # no trace known (an external agent, case validation): not graded
    assert evals.run_checks(case, tmp_path, "").passed
    # cases without the field ignore the trace
    assert evals.run_checks({"id": "d", "files": {}}, tmp_path, "", check_trace="").passed


def test_meter_records_check_results_in_order():
    meter = evals._Meter()
    for name, result in (("run_check", "exit code: 1\nboom"), ("read_file", "x"),
                         ("run_command", "exit code: 0\nok"), ("run_check", "Not run: stopped")):
        meter.on_event("tool_result", {"name": name, "result": result})
    assert meter.check_trace == "FP"
