"""Chat response time benchmark: timings recorded per request, answers checked, rows saved."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import latency
from vega.agent import CodingAgent


def scripted(answer: str):
    def factory(project: Path) -> CodingAgent:
        agent = CodingAgent(project, stream=False, allow_commands=False)

        def create(**kwargs):
            usage = SimpleNamespace(prompt_tokens=400, completion_tokens=20)
            message = SimpleNamespace(content=answer, tool_calls=None, role="assistant")
            message.model_dump = lambda exclude_none=False: {"role": "assistant", "content": answer}
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                                   usage=usage)

        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=create)))
        return agent
    return factory


@pytest.fixture(autouse=True)
def plain_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "direct_pipeline,nudge_act")  # a scripted plain answer


def test_a_run_records_timings_tokens_and_correctness() -> None:
    run = latency.run_once(latency.REQUESTS[0], cold=False, agent_factory=scripted("It returns 7."))
    assert run.correct and run.first_feedback is not None and run.answer >= run.first_feedback
    assert run.model_calls >= 1 and run.prompt_tokens >= 400 and run.reply_tokens >= 20
    wrong = latency.run_once(latency.REQUESTS[0], cold=False, agent_factory=scripted("It is 8."))
    assert not wrong.correct  # a fast wrong answer is reported as wrong


def test_checks_reject_near_misses() -> None:
    check = latency.REQUESTS[0].check
    assert check("7", Path(".")) and check("It returns 7.", Path("."))
    assert not check("17", Path(".")) and not check("0.7", Path(".")) and not check("7.5", Path("."))
    assert latency.REQUESTS[2].check("It is in shop/inventory.py.", Path("."))


def test_summary_and_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(latency, "LATENCY_FILE", tmp_path / "latency.jsonl")
    runs = [latency.Run("quick-question", "question", False, first_feedback=1.0, answer=2.0,
                        correct=True, reply_tokens=40, model_seconds=2.0),
            latency.Run("quick-question", "question", False, first_feedback=3.0, answer=4.0,
                        correct=False, reply_tokens=40, model_seconds=2.0),
            latency.Run("quick-question", "question", True, first_feedback=9.0, answer=12.0)]
    record = latency._record(runs, repeat=2, cold=True)
    row = record["requests"]["quick-question"]
    assert row["runs"] == 2 and row["correct"] == 1 and row["answer_median"] == 3.0
    assert row["tokens_per_second"] == 20.0
    assert record["cold_start"]["answer"] == 12.0
    saved = json.loads((tmp_path / "latency.jsonl").read_text())
    assert saved["requests"]["quick-question"]["first_feedback_median"] == 2.0
