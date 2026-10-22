"""Evals pin sampling: every model request of an eval run carries an explicit temperature and seed
(unpinned runs flipped cases between otherwise identical runs), A/B arms carry the same ones, and clarify's ask/act check is sent at temperature 0.
Fake model clients only: the tests read the request payloads they receive."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_agent_loop import text
from typer.testing import CliRunner

from vega import clarify, evals, features, models
from vega.agent import CodingAgent

SAMPLING_KEYS = ("temperature", "seed", "max_tokens")


@pytest.fixture
def suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cases = tmp_path / "fixes.jsonl"
    cases.write_text(
        '{"id": "a", "task": "Say ok.", "files": {"x.txt": "1"}, "response_contains": ["ok"]}\n'
        '{"id": "b", "task": "Say ok.", "files": {"x.txt": "1"}, "response_contains": ["ok"]}\n'
    )
    monkeypatch.setattr(evals, "suite_file", lambda name: cases)
    monkeypatch.setattr(evals, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setattr(evals, "_git_commit", lambda: "abc1234")
    monkeypatch.setattr(evals, "_server_fingerprint", lambda model: {})
    monkeypatch.setattr(models, "profile_for_model", lambda model: None)  # no profile temperature
    for name in ("VEGA_SEED", "VEGA_TEMPERATURE", "VEGA_MAX_TOKENS", features.ENV):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def capturing_factory(seen: list[dict]) -> evals.AgentFactory:
    """A real CodingAgent whose model client records every request (with the model it was for)."""

    def factory(project: Path, max_steps: int) -> CodingAgent:
        agent = CodingAgent(project, allow_commands=False, max_steps=max_steps)

        def create(**kwargs):
            seen.append({**kwargs, "env_model": os.environ.get("VEGA_MODEL", "vega")})
            return SimpleNamespace(choices=[SimpleNamespace(message=text("ok"),
                                                            finish_reason="stop")], usage=None)

        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return agent

    return factory


def sampling_of(request: dict) -> dict:
    return {key: request.get(key) for key in SAMPLING_KEYS}


def test_eval_run_pins_temperature_and_seed_in_every_request(suite: Path) -> None:
    seen: list[dict] = []
    record = evals.run_suite("fixes", repeat=2, agent_factory=capturing_factory(seen))
    assert record["passed"] == 4 and seen
    # attempt k carries seed 7 + k; every request carries temperature 0 and the reply cap
    seeds = [request["seed"] for request in seen]
    assert set(seeds) == {7, 8}
    assert all(request["temperature"] == 0.0 and request["max_tokens"] == 2048 for request in seen)
    assert record["sampling"] == {"temperature": 0.0, "seed": 7, "max_tokens": 2048,
                                  "attempt_seed": "seed + attempt index (0-based)"}
    attempts = [attempt["seed"] for case in record["cases"] for attempt in case["attempts"]]
    assert attempts == [7, 8, 7, 8]
    saved = json.loads((suite / "runs.jsonl").read_text())
    assert saved["sampling"]["temperature"] == 0.0 and saved["sampling"]["seed"] == 7
    # restored afterwards: interactive use in the same process keeps its own sampling
    assert "VEGA_SEED" not in os.environ and "VEGA_TEMPERATURE" not in os.environ


def test_explicit_flags_win_and_model_sends_no_temperature(suite: Path, monkeypatch) -> None:
    seen: list[dict] = []
    monkeypatch.setenv("VEGA_TEMPERATURE", "0.5")
    monkeypatch.setenv("VEGA_SEED", "11")
    record = evals.run_suite("fixes", repeat=1, agent_factory=capturing_factory(seen))
    assert all(request["temperature"] == 0.5 and request["seed"] == 11 for request in seen)
    assert record["sampling"]["temperature"] == 0.5 and record["sampling"]["seed"] == 11

    seen.clear()
    monkeypatch.setenv("VEGA_TEMPERATURE", "model")
    record = evals.run_suite("fixes", repeat=1, agent_factory=capturing_factory(seen))
    assert seen and all("temperature" not in request and request["seed"] == 11 for request in seen)
    assert "temperature" not in record["sampling"]
    assert os.environ["VEGA_TEMPERATURE"] == "model"  # restored as the caller set it


def test_profile_temperature_is_the_eval_default(monkeypatch) -> None:
    profile = models.Profile(name="coder-7b", base="x", parameters={"temperature": 0.2})
    monkeypatch.setattr(models, "profile_for_model", lambda model: profile)
    assert evals.eval_temperature("vega-coder-7b") == 0.2
    monkeypatch.setattr(models, "profile_for_model", lambda model: models.Profile(name="g", base="x"))
    assert evals.eval_temperature("vega") == evals.EVAL_TEMPERATURE == 0.0


@pytest.mark.parametrize("b_model", [None, "big"])
def test_ab_arms_send_identical_sampling(suite: Path, monkeypatch, b_model) -> None:
    seen: list[dict] = []
    monkeypatch.setenv("VEGA_MODEL", "vega")
    monkeypatch.setattr(evals, "_resolve_model", lambda value: f"vega-{value}")
    off = [] if b_model else ["lint"]
    record = evals.run_ab("fixes", off, repeat=2, agent_factory=capturing_factory(seen),
                          b_model=b_model)
    assert record["paired"]["pairs"] == 4
    a_arm = [a for case in record["variants"]["A"]["cases"] for a in case["attempts"]]
    b_arm = [b for case in record["variants"]["B"]["cases"] for b in case["attempts"]]
    assert [a["seed"] for a in a_arm] == [b["seed"] for b in b_arm] == [7, 8, 7, 8]
    assert {(r["temperature"], r["max_tokens"]) for r in seen} == {(0.0, 2048)}
    if b_model:  # every request of each arm went to its own model, with the same options
        by_model = {model: sorted(sampling_of(r)["seed"] for r in seen if r["env_model"] == model)
                    for model in ("vega", "vega-big")}
        assert by_model["vega"] == by_model["vega-big"] and set(by_model["vega"]) == {7, 8}
    assert record["sampling"]["temperature"] == 0.0 and record["sampling"]["seed"] == 7
    assert "VEGA_SEED" not in os.environ and "VEGA_TEMPERATURE" not in os.environ


def clarify_agent(project: Path) -> tuple[CodingAgent, list[dict]]:
    seen: list[dict] = []

    def create(**kwargs):
        seen.append(kwargs)
        reply = json.dumps({"specific": True, "questions": []})
        return SimpleNamespace(choices=[SimpleNamespace(message=text(reply), finish_reason="stop")],
                               usage=None)

    agent = CodingAgent(project, allow_commands=False)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, seen


def test_clarify_check_is_sent_at_temperature_zero_with_the_eval_seed(tmp_path: Path,
                                                                       monkeypatch) -> None:
    monkeypatch.setenv("VEGA_TEMPERATURE", "0.8")  # the turn's own sampling
    monkeypatch.setenv("VEGA_SEED", "7")
    monkeypatch.delenv(features.ENV, raising=False)
    agent, seen = clarify_agent(tmp_path)
    assert agent.sampling["temperature"] == 0.8
    assert clarify.model_check(agent, "Add low-stock alerts.") is None
    assert len(seen) == 1
    assert seen[0]["temperature"] == 0.0 and seen[0]["seed"] == 7
    assert seen[0]["max_tokens"] == clarify.CHECK_MAX_TOKENS


def test_clarify_check_without_a_seed_and_with_clarify_greedy_off(tmp_path: Path,
                                                                 monkeypatch) -> None:
    monkeypatch.delenv("VEGA_SEED", raising=False)
    monkeypatch.delenv("VEGA_TEMPERATURE", raising=False)
    agent, seen = clarify_agent(tmp_path)
    clarify.model_check(agent, "Add low-stock alerts.")
    assert seen[0]["temperature"] == 0.0 and "seed" not in seen[0]  # interactive: no seed

    monkeypatch.setenv(features.ENV, "clarify_greedy")
    monkeypatch.setenv("VEGA_TEMPERATURE", "0.8")
    agent, seen = clarify_agent(tmp_path)
    clarify.model_check(agent, "Add low-stock alerts.")
    assert seen[0]["temperature"] == 0.8


def test_cli_rejects_a_temperature_that_is_not_a_number_or_model() -> None:
    from vega.cli import app

    result = CliRunner().invoke(app, ["eval", "--temperature", "warm"])
    assert result.exit_code != 0 and "number or `model`" in result.output
