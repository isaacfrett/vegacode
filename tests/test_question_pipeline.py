"""Question fast path: one call with retrieved excerpts; falls back when they lack the answer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import features, question_pipeline, routing, sandbox
from vega.agent import CodingAgent


def project(root: Path) -> Path:
    (root / "shop").mkdir()
    (root / "shop" / "__init__.py").write_text("")
    (root / "shop" / "pricing.py").write_text(
        "TIERS = {'gold': 20, 'silver': 10}\n\n\ndef discounted_price(price, tier):\n"
        "    return price - price * TIERS.get(tier, 0) / 100\n")
    (root / "shop" / "payments.py").write_text("class PaymentProcessor:\n    pass\n")
    return root


@pytest.fixture(autouse=True)
def fast_path_on(monkeypatch) -> None:
    monkeypatch.delenv("VEGA_FEATURES_OFF", raising=False)  # conftest turns it off for other tests


def scripted(root: Path, replies: list[str]) -> tuple[CodingAgent, list[dict]]:
    agent = CodingAgent(root, allow_commands=False)
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content=replies.pop(0), tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, calls


def asked(task: str, root: Path) -> bool:
    """Whether routing sends the task to this pipeline (the old applies() gate went with the old
    routing, 2026-10-17)."""
    return routing.route_task(task, root).name == routing.QUESTION


@pytest.mark.parametrize("task, applies", [
    ("What does discounted_price(100, 'gold') return?", True),
    ("Which file defines the PaymentProcessor class? Do not modify anything.", True),
    (("Review src/auth.py and tell me the security problem in it. Only report it; do not change "
     "any files."), True),
    ("Find cli.py anywhere in this project and report the marker string inside it.", True),
    ("Fix the off-by-one in pricing.py", False),
    ("Add a --verbose flag to the CLI", False),
    ("Can you rename discounted_price to final_price?", False),
])
def test_routes_questions_here_not_changes(tmp_path: Path, task: str, applies: bool) -> None:
    assert asked(task, project(tmp_path)) is applies


def test_one_call_with_the_relevant_code(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "question_pipeline")
    agent, calls = scripted(project(tmp_path), ["It returns 80.0 (shop/pricing.py)."])
    answer = agent.run("What does discounted_price(100, 'gold') return?")
    assert answer == "It returns 80.0 (shop/pricing.py)."
    assert len(calls) == 1 and "tools" not in calls[0]
    prompt = calls[0]["messages"][-1]["content"]
    assert "def discounted_price" in prompt and "TIERS = {'gold': 20" in prompt
    assert len(prompt) < 9000  # a short prompt is the point


def test_named_file_is_shown_whole(tmp_path: Path) -> None:
    shown = question_pipeline.excerpts(project(tmp_path), "What is in pricing.py?")
    assert shown.startswith("shop/pricing.py (whole file):\n1: TIERS")


def test_not_found_falls_back_to_the_agent_loop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV_ON, "question_pipeline")
    agent, _calls = scripted(project(tmp_path), [question_pipeline.NOT_FOUND])
    seen = []
    agent.on_event = lambda kind, data: seen.append((kind, data.get("message", "")))
    assert question_pipeline.run(agent, "Which environment variable sets the cache folder?") is None
    assert ("notice", "Not in the obvious places; searching the project.") in seen
    assert len(agent.messages) == 1  # nothing added: the agent loop takes the question


def test_on_by_default() -> None:
    assert features.on("question_pipeline")


def test_questions_to_vega_and_general_questions_are_not_code_lookups(tmp_path: Path) -> None:
    for task in ("what can you do", "hi", "hey", "What can you help me with?", "who are you",
                 "what is a closure in python?", "what's the difference between a list and a tuple?"):
        assert not asked(task, tmp_path), task
    for task in ("Which file defines the Inventory class?", "where is the retry logic?",
                 "how does the config loader work?", "explain the cache module",
                 "Which environment variable sets the cache folder?",
                 "In what order does tessellate's Grid.neighbors list a cell's neighbours?",
                 'hey the report crashes. `summary_line("Aisle 3", [])` is it the empty mean?'):
        assert asked(task, tmp_path), task
    # "port" here is a noun, not "port X to Go" (a routing quirk from 2026-10-16, fixed in cycle 3)
    assert asked("what port does the server use?", tmp_path)


def test_whole_project_questions_go_to_the_agent_loop(tmp_path: Path) -> None:
    # live: "what can be improved about this project?" got a critique of one matched file
    for task in ("what can be improved about this project?", "What about this project do you recogonize?",
                 "what does this project do?", "review the architecture", "give me an overview"):
        assert not asked(task, tmp_path), task
    for task in ("how could pager.py be improved?", "where is the retry logic?",
                 "what could go wrong in `parse_args`?"):
        assert asked(task, tmp_path), task


# -- question_run_code (broad sample 2026-10-17, fix 4) ----------------------------------------------

def cartage(root: Path) -> Path:
    """robust-fuel-surcharge: tiers are exclusive upper limits, so 500 km is in the 6% tier."""
    (root / "cartage").mkdir()
    (root / "cartage" / "__init__.py").write_text("")
    (root / "cartage" / "tiers.py").write_text(
        "SURCHARGE_TIERS = [(150, 0), (500, 4), (1200, 6), (None, 9)]\n")
    (root / "cartage" / "rates.py").write_text(
        "from decimal import Decimal\n\nfrom cartage.tiers import SURCHARGE_TIERS\n\n"
        "PER_KM = Decimal('1.10')\n\n\ndef surcharge_percent(distance_km):\n"
        "    for limit, percent in SURCHARGE_TIERS:\n"
        "        if limit is None or distance_km < limit:\n            return percent\n\n\n"
        "def quote(distance_km):\n    base = PER_KM * distance_km\n"
        "    return base * (100 + surcharge_percent(distance_km)) / 100\n")
    (root / "cartage" / "legacy.py").write_text(
        "def fuel_surcharge(distance_km):\n    return 4 if distance_km <= 500 else 8\n")
    return root


FUEL = ("Answer with one line in the form `<function name>: <percent>%`, and do not change any "
        "files. The trip is exactly 500 km: what surcharge percentage does it get, and which "
        "function decides the fuel surcharge percentage that `quote()` in `cartage/rates.py` "
        "applies?")


def test_named_function_and_its_callees_run_on_the_stated_input(tmp_path: Path) -> None:
    calls = question_pipeline.planned_calls(cartage(tmp_path), FUEL)
    assert [c["shown"] for c in calls] == ["cartage.rates.quote(500)",
                                           "cartage.rates.surcharge_percent(500)"]
    # legacy.fuel_surcharge is neither named nor called by quote(): never run (it says 4)


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_the_actual_result_goes_to_the_model(tmp_path: Path, monkeypatch) -> None:
    agent, calls = scripted(cartage(tmp_path), ["surcharge_percent: 6%"])
    answer = question_pipeline.run(agent, FUEL)
    assert answer == "surcharge_percent: 6%"  # the model still writes the answer
    shown = calls[0]["messages"][-1]["content"]
    assert "- cartage.rates.surcharge_percent(500) returned 6" in shown
    assert "- cartage.rates.quote(500) returned Decimal('583.00')" in shown
    assert len(calls) == 1
    assert (tmp_path / "cartage" / "rates.py").read_text().count("def ") == 2  # unchanged


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_switch_off_is_the_old_prompt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "question_run_code")
    agent, calls = scripted(cartage(tmp_path), ["surcharge_percent: 4%"])
    question_pipeline.run(agent, FUEL)
    assert "Vega ran the project's code" not in calls[0]["messages"][-1]["content"]


@pytest.mark.skipif(not sandbox.available(), reason="needs the macOS sandbox")
def test_a_call_written_out_runs_as_written_and_errors_are_reported(tmp_path: Path) -> None:
    root = project(tmp_path)
    assert question_pipeline.call_evidence(root, "What does discounted_price(100, 'gold') return?"
                                           ).endswith("- shop.pricing.discounted_price(100, 'gold') "
                                                      "returned 80.0")
    evidence = question_pipeline.call_evidence(root, "What does `discounted_price(None, 'x')` return?")
    assert evidence.endswith("raised TypeError: unsupported operand type(s) for *: 'NoneType' and "
                             "'int'")


@pytest.mark.parametrize("task", [
    "What does discounted_price(price, tier) return?",  # no literal input
    "What do 100 and 200 cost with discounted_price?",  # two numbers: which goes where?
    "Explain discounted_price for 100.",  # asks for no value
    "What does discounted_price(100, 'gold') return? Do not run any commands.",
    "Which version of Python 3.12 does `discounted_price` need?",  # a version, not an input
])
def test_nothing_runs_without_one_literal_input(tmp_path: Path, task: str, monkeypatch) -> None:
    root = project(tmp_path)
    if "Do not run" in task:  # as acceptance: only with acceptance_ignores_no_commands off
        monkeypatch.setenv("VEGA_FEATURES_OFF", "acceptance_ignores_no_commands")
        assert question_pipeline.call_evidence(root, task) == ""
    else:
        assert question_pipeline.planned_calls(root, task) == []


def test_question_numbers_skip_templates_lines_and_versions() -> None:
    assert question_pipeline.question_numbers(FUEL) == ["500"]
    assert question_pipeline.question_numbers("On SQLite 3.50, line 12: what is f for `7`?") == ["7"]
