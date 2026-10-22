"""Chat response time: how long everyday requests take, measured the way the chat runs them.

    vega latency                 # cold start, then each request 3 times, warm
    vega latency --repeat 5 --no-cold

Each request runs through `CodingAgent` with streaming on (the chat's path) in a small fixture
project. Per run it records the time to first feedback (the first thing the chat can show: a
step line, a plan, a notice or streamed text), the time to the final answer, the model's
generation speed and the prompt size; each answer is also checked, so a fast wrong answer is
reported as wrong. The cold run unloads the model first, so it includes loading it. Results
append to evals/latency.jsonl (one row per run of the benchmark).
"""

from __future__ import annotations

import json
import os
import re
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from . import features, server
from .agent import CodingAgent
from .resources import Guard

LATENCY_FILE = Path(__file__).resolve().parents[2] / "evals" / "latency.jsonl"
console = Console()

FIXTURE = {
    "shop/__init__.py": "",
    "shop/cart.py": (
        '"""A shopping cart."""\n\n\n'
        "def subtotal(items):\n"
        '    """Sum of price * quantity for (price, quantity) pairs."""\n'
        "    return sum(price * quantity for price, quantity in items)\n\n\n"
        "def discount(total, percent):\n"
        '    """`total` reduced by `percent` percent."""\n'
        "    return total - total * percent / 100\n"
    ),
    "shop/inventory.py": (
        '"""Stock levels."""\n\n\n'
        "class Inventory:\n"
        "    def __init__(self):\n"
        "        self.levels = {}\n\n"
        "    def add(self, sku, quantity):\n"
        "        self.levels[sku] = self.levels.get(sku, 0) + quantity\n\n"
        "    def remove(self, sku, quantity):\n"
        "        self.levels[sku] = self.levels.get(sku, 0) - quantity\n"
    ),
    "shop/fmt.py": (
        '"""Formatting."""\n\n\n'
        "def money(amount):\n"
        '    """Format an amount with two decimals and a leading $, e.g. money(3) == "$3.00".\n\n'
        "    >>> money(3)\n"
        "    '$3.00'\n"
        '    """\n'
        "    raise NotImplementedError\n"
    ),
    "tests/test_cart.py": (
        "from shop.cart import discount, subtotal\n\n\n"
        "def test_subtotal():\n    assert subtotal([(2, 3), (1, 1)]) == 7\n\n\n"
        "def test_discount():\n    assert discount(200, 10) == 180\n"
    ),
}


@dataclass
class Request:
    name: str
    kind: str  # what the chat request is like: question | explain | locate | edit
    message: str
    check: Callable[[str, Path], bool]  # (answer, project) -> correct?


def _file(project: Path, path: str) -> str:
    target = project / path
    return target.read_text(encoding="utf-8") if target.is_file() else ""


def _money_ok(project: Path) -> bool:
    namespace: dict[str, Any] = {}
    try:
        exec(compile(_file(project, "shop/fmt.py"), "fmt.py", "exec"), namespace)  # noqa: S102
        return namespace["money"](3) == "$3.00" and namespace["money"](2.5) == "$2.50"
    except Exception:  # noqa: BLE001 - any failure means the edit is wrong
        return False


REQUESTS = [
    Request("quick-question", "question",
            "What does subtotal([(2, 3), (1, 1)]) in shop/cart.py return? Just the number.",
            lambda answer, _: re.search(r"(?<![\d.])7(?!\d|\.\d)", answer) is not None),
    Request("explain", "explain",
            "In one or two sentences, what does discount() in shop/cart.py do?",
            lambda answer, _: "percent" in answer.lower() or "%" in answer),
    Request("locate", "locate",
            "Which file defines the Inventory class?",
            lambda answer, _: "inventory.py" in answer),
    Request("small-edit", "edit",
            "Implement money() in shop/fmt.py as its docstring describes.",
            lambda _, project: _money_ok(project)),
]


@dataclass
class Run:
    request: str
    kind: str
    cold: bool
    first_feedback: float | None = None  # seconds to the first event the chat can show
    answer: float | None = None  # seconds to the final answer
    correct: bool = False
    model_calls: int = 0
    prompt_tokens: int = 0
    reply_tokens: int = 0
    model_seconds: float = 0.0
    events: list[str] = field(default_factory=list)
    reply: str = ""  # the start of the final answer, to read a wrong one

    @property
    def tokens_per_second(self) -> float | None:
        return self.reply_tokens / self.model_seconds if self.model_seconds and self.reply_tokens \
            else None


class _Timing:
    """Wraps the agent's client: counts calls, tokens and model time (streamed or not)."""

    def __init__(self, client: Any, run: Run):
        self._client, self._run = client, run
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Completions", (), {})()
        self.chat.completions.create = self._create

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _create(self, **kwargs: Any) -> Any:
        self._run.model_calls += 1
        started = time.monotonic()
        if kwargs.get("stream"):
            kwargs.setdefault("stream_options", {"include_usage": True})
            return self._stream(self._client.chat.completions.create(**kwargs), started)
        response = self._client.chat.completions.create(**kwargs)
        self._account(getattr(response, "usage", None), started)
        return response

    def _stream(self, chunks: Any, started: float) -> Any:
        usage = None
        for chunk in chunks:
            usage = getattr(chunk, "usage", None) or usage
            yield chunk
        self._account(usage, started)

    def _account(self, usage: Any, started: float) -> None:
        self._run.model_seconds += time.monotonic() - started
        self._run.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self._run.reply_tokens += getattr(usage, "completion_tokens", 0) or 0


VISIBLE = {"text", "tool_call", "plan", "notice", "discarded", "final"}


def run_once(request: Request, *, cold: bool, agent_factory: Callable[[Path], CodingAgent] | None
             = None) -> Run:
    run = Run(request.name, request.kind, cold)
    with tempfile.TemporaryDirectory(prefix="vega-latency-") as temp:
        project = Path(temp).resolve()
        for path, text in FIXTURE.items():
            (project / path).parent.mkdir(parents=True, exist_ok=True)
            (project / path).write_text(text, encoding="utf-8")
        agent = (agent_factory or (lambda p: CodingAgent(p, stream=True, allow_commands=False)))(
            project)
        agent.client = _Timing(agent.client, run)
        started = time.monotonic()

        def on_event(kind: str, data: dict[str, Any]) -> None:
            if kind in VISIBLE and run.first_feedback is None:
                run.first_feedback = time.monotonic() - started
            if kind in VISIBLE and kind != "text":
                run.events.append(kind if kind != "tool_call" else f"tool:{data.get('name')}")

        agent.on_event = on_event
        try:
            answer = agent.run(request.message)
        except Exception as exc:  # noqa: BLE001 - a failed request is a result, not a crash
            answer = f"(error: {exc})"
        run.answer = time.monotonic() - started
        run.reply = answer[:300]
        run.correct = bool(request.check(answer, project))
    return run


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]


def _s(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}s"


def summarise(runs: list[Run]) -> dict[str, Any]:
    by_request: dict[str, Any] = {}
    for request in REQUESTS:
        warm = [r for r in runs if r.request == request.name and not r.cold]
        if not warm:
            continue
        speeds = [r.tokens_per_second for r in warm if r.tokens_per_second]
        by_request[request.name] = {
            "kind": request.kind, "runs": len(warm),
            "correct": sum(r.correct for r in warm),
            "first_feedback_median": _median([r.first_feedback for r in warm
                                              if r.first_feedback is not None]),
            "answer_median": _median([r.answer for r in warm if r.answer is not None]),
            "answer_p90": _p90([r.answer for r in warm if r.answer is not None]),
            "tokens_per_second": _median(speeds),
            "prompt_tokens_median": _median([r.prompt_tokens for r in warm]),
            "model_calls_median": _median([r.model_calls for r in warm]),
        }
    cold = next((r for r in runs if r.cold), None)
    return {"requests": by_request,
            "cold_start": None if cold is None else {
                "request": cold.request, "first_feedback": cold.first_feedback,
                "answer": cold.answer, "correct": cold.correct}}


def benchmark(repeat: int = 3, *, cold: bool = True, guard: Guard | None = None) -> dict[str, Any]:
    runs: list[Run] = []
    profile = server._endpoint_profile()
    if cold and profile is not None:
        console.print(f"[yellow]Cold start skipped: profile {profile.name!r} is served by "
                      f"{profile.endpoint}, which Vega cannot unload; warm runs only.[/yellow]")
        cold = False
    if cold:
        server.unload()
        time.sleep(2)
        runs.append(run_once(REQUESTS[0], cold=True))
        console.print(f"[dim]cold start ({REQUESTS[0].name}): first feedback "
                      f"{_s(runs[-1].first_feedback)}, answer {_s(runs[-1].answer)}[/dim]")
    for index in range(repeat):
        for request in REQUESTS:
            if guard is not None:
                stop = guard.before_attempt()
                if stop:
                    console.print(f"[yellow]Stopped: {stop}[/yellow]")
                    return _record(runs, repeat, cold, stopped=stop)
            run = run_once(request, cold=False)
            runs.append(run)
            console.print(f"[dim]{request.name} {index + 1}/{repeat}: first feedback "
                          f"{_s(run.first_feedback)} · answer {_s(run.answer)} · "
                          f"{'correct' if run.correct else 'WRONG'}[/dim]")
    return _record(runs, repeat, cold)


def _record(runs: list[Run], repeat: int, cold: bool, stopped: str | None = None) -> dict[str, Any]:
    from .evals import _git_commit

    record = {
        "timestamp": datetime.now(UTC).isoformat(), "model": server.model_name(),
        "commit": _git_commit(), "repeat": repeat, "cold": cold,
        "features_off": sorted(features.disabled()), "features_on": sorted(features.enabled()),
        "stopped": stopped, **summarise(runs),
        "runs": [{**{k: v for k, v in vars(r).items() if k != "events"},
                  "events": r.events[:20]} for r in runs],
    }
    LATENCY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LATENCY_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    show(record)
    return record


def show(record: dict[str, Any]) -> None:
    table = Table("request", "kind", "correct", "first feedback", "answer (median)",
                  "answer (p90)", "tokens/s", "prompt tokens", "model calls",
                  title=f"Chat response time · {record['model']} · {record['commit']}")
    for name, row in record["requests"].items():
        table.add_row(name, row["kind"], f"{row['correct']}/{row['runs']}",
                      _s(row["first_feedback_median"]), _s(row["answer_median"]),
                      _s(row["answer_p90"]),
                      "—" if row["tokens_per_second"] is None else f"{row['tokens_per_second']:.0f}",
                      "—" if row["prompt_tokens_median"] is None else f"{row['prompt_tokens_median']:.0f}",
                      "—" if row["model_calls_median"] is None else f"{row['model_calls_median']:.0f}")
    console.print(table)
    cold = record.get("cold_start")
    if cold:
        console.print(f"Cold start (model unloaded first): first feedback {_s(cold['first_feedback'])}"
                      f", answer {_s(cold['answer'])} ({'correct' if cold['correct'] else 'wrong'}).")
    if os.environ.get("VEGA_EVAL_MODE"):
        console.print("[yellow]VEGA_EVAL_MODE is set; it does not apply here.[/yellow]")
