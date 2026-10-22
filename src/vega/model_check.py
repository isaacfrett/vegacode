"""`vega model check <profile>`: does a model work through Vega's interface?

A fast conformance check (about a minute on a 7-8B model) to run after attaching a model and
before evaluating it. Each item exercises one thing Vega relies on, on the small `vega latency`
fixture project and its requests:

    chat         a plain streamed answer (also gives the first-reply latency)
    thinking     <think>...</think> reasoning is stripped from the answer
    question     the question pipeline answers from the code index in one call
    direct-edit  the direct pipeline implements a function in one file and its checks pass
    tool-call    the agent loop calls a tool and answers from its result
    truncation   a reply cut off at the token limit is reported (finish_reason "length") and
                 continued into complete code
    options      the profile's request fields went out with every request and were accepted;
                 on Ollama, the built model has the profile's num_ctx

Any failed item makes the command exit non-zero. Nothing is recorded to the eval logs.
"""

from __future__ import annotations

import ast
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import features, latency, models, server
from .agent import CodingAgent
from .pipeline import ask_complete, strip_thinking

CHAT_MAX_TOKENS = 512  # a plain answer needs a few tokens; a thinking model may use them all
TRUNCATE_AT = 32  # tokens per reply in the truncation item: the code needs several replies
TRUNCATE_CONTINUATIONS = 8
TOOL_STEPS = 8
CHAT_MESSAGES = [
    {"role": "system", "content": "Answer with just the number."},
    {"role": "user", "content": "What is 17 * 23?"},
]
TRUNCATE_PROMPT = (
    "Write a Python function squares(n) that returns the list of the first n square numbers, "
    "e.g. squares(3) == [1, 4, 9]. Give it a docstring of three sentences and one doctest. "
    "Reply with only one ```python block."
)
PIPELINES = ("question_pipeline", "direct_pipeline", "project_pipeline", "deliverable_pipeline")
_BLOCK = re.compile(r"```[\w.+-]*[ \t]*\n(.*?)```", re.DOTALL)

AgentFactory = Callable[[Path], CodingAgent]


@dataclass
class Item:
    name: str
    passed: bool | None  # None: not exercised / skipped (does not fail the check)
    seconds: float = 0.0
    detail: str = ""


@dataclass
class Report:
    model: str  # the name Vega runs it under (vega-<profile>, vega or an Ollama name)
    profile: str | None
    base_url: str
    served_as: str  # the name sent to the server
    ollama: bool
    items: list[Item] = field(default_factory=list)
    first_reply: float | None = None  # seconds to the first streamed token of the chat item
    memory: str = ""
    stopped: str = ""

    @property
    def ok(self) -> bool:
        return not self.stopped and all(item.passed is not False for item in self.items) and any(
            item.passed for item in self.items)


class Recorder:
    """Wraps an agent's client: keeps each request's fields, finish reason and error."""

    def __init__(self, client: Any, log: list[dict[str, Any]]):
        self._client, self.log = client, log
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Completions", (), {})()
        self.chat.completions.create = self._create

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _create(self, **kwargs: Any) -> Any:
        entry: dict[str, Any] = {"kwargs": {k: v for k, v in kwargs.items() if k != "messages"},
                                 "finish": None, "error": None, "tokens": None}
        self.log.append(entry)
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        if kwargs.get("stream"):
            return self._stream(response, entry)
        try:
            entry["finish"] = getattr(response.choices[0], "finish_reason", None)
        except (AttributeError, IndexError, TypeError):
            pass
        entry["tokens"] = getattr(getattr(response, "usage", None), "completion_tokens", None)
        return response

    def _stream(self, chunks: Any, entry: dict[str, Any]) -> Iterator[Any]:
        for chunk in chunks:
            choices = getattr(chunk, "choices", None) or []
            if choices and getattr(choices[0], "finish_reason", None):
                entry["finish"] = choices[0].finish_reason
            yield chunk


@contextmanager
def _features(*, off: tuple[str, ...] = (), on: tuple[str, ...] = ()) -> Iterator[None]:
    """Switch features for one item on top of the current settings; restored afterwards."""
    saved = {name: os.environ.get(name) for name in (features.ENV, features.ENV_ON)}
    current_off = {n for n in (saved[features.ENV] or "").split(",") if n.strip()}
    current_on = {n for n in (saved[features.ENV_ON] or "").split(",") if n.strip()}
    default_off = {name for name in off if name in features.FEATURES}
    os.environ[features.ENV] = ",".join(sorted((current_off | default_off) - set(on)))
    opt_in = {name for name in on if name in features.OPT_IN}
    os.environ[features.ENV_ON] = ",".join(sorted((current_on | opt_in) - set(off)))
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _project(root: Path) -> Path:
    for path, text in latency.FIXTURE.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    return root


def _default_factory(project: Path) -> CodingAgent:
    return CodingAgent(project, stream=False, allow_commands=False, max_steps=TOOL_STEPS)


class _Run:
    def __init__(self, factory: AgentFactory, requests: list[dict[str, Any]]):
        self.factory, self.requests = factory, requests

    def agent(self, project: Path, events: list[tuple[str, dict[str, Any]]]) -> CodingAgent:
        agent = self.factory(project)
        agent.client = Recorder(agent.client, self.requests)
        agent.on_event = lambda kind, data: events.append((kind, data))
        return agent


def _chat(run: _Run, project: Path, report: Report) -> tuple[Item, str]:
    agent = run.agent(project, [])
    started = time.monotonic()
    text = ""
    sampling = {**agent.sampling, "max_tokens": CHAT_MAX_TOKENS}
    finish = None
    for chunk in agent.client.chat.completions.create(
            model=agent.model, messages=CHAT_MESSAGES, stream=True, **sampling):
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        piece = getattr(choices[0].delta, "content", None) or ""
        if piece and report.first_reply is None:
            report.first_reply = time.monotonic() - started
        text += piece
        finish = getattr(choices[0], "finish_reason", None) or finish
    seconds = time.monotonic() - started
    answer = strip_thinking(text).strip()
    passed = re.search(r"(?<!\d)391(?!\d)", answer) is not None
    detail = f"answered {answer[:60]!r}"
    if not passed and finish == "length":
        detail += f" (hit the {CHAT_MAX_TOKENS}-token limit: thinking left on?)"
    return Item("chat", passed, seconds, detail), text


def _thinking(raw: str) -> Item:
    stripped = strip_thinking(raw)
    leftover = "<think>" in stripped or "</think>" in stripped
    if "<think>" in raw:
        detail = f"reply had <think> reasoning ({len(raw) - len(stripped):,} chars), stripped"
    elif "</think>" in raw:
        detail = (f"reply had reasoning closed by </think> with the opening tag in the template "
                  f"({len(raw) - len(stripped):,} chars), stripped")
    else:
        detail = "no thinking tags in the reply (nothing to strip)"
    if not stripped.strip():
        detail += "; nothing left after stripping"
    return Item("thinking", bool(stripped.strip()) and not leftover, 0.0, detail)


def _agent_item(run: _Run, project: Path, name: str, request: latency.Request, *,
                expect_notice: str = "", need_tool: bool = False) -> Item:
    events: list[tuple[str, dict[str, Any]]] = []
    agent = run.agent(project, events)
    started = time.monotonic()
    try:
        answer = agent.run(request.message)
    except Exception as exc:  # noqa: BLE001 - a failure here is the finding
        return Item(name, False, time.monotonic() - started, f"error: {exc}")
    seconds = time.monotonic() - started
    notices = [str(data.get("message", "")) for kind, data in events if kind == "notice"]
    tools = [str(data.get("name")) for kind, data in events if kind == "tool_call"]
    correct = bool(request.check(answer, project))
    routed = not expect_notice or any(expect_notice in notice for notice in notices)
    called = not need_tool or bool(tools)
    passed = correct and routed and called
    parts = ["correct" if correct else f"wrong answer {answer[:80]!r}"]
    if expect_notice and not routed:
        parts.append("did not go through the pipeline")
    if need_tool:
        parts.append(f"tools: {', '.join(tools)}" if tools else "no tool call")
    return Item(name, passed, seconds, "; ".join(parts))


def _truncation(run: _Run, project: Path) -> Item:
    agent = run.agent(project, [])
    agent.sampling = {**agent.sampling, "max_tokens": TRUNCATE_AT}
    first = len(run.requests)
    started = time.monotonic()
    try:
        text = ask_complete(agent, [{"role": "user", "content": TRUNCATE_PROMPT}],
                            continuations=TRUNCATE_CONTINUATIONS)
    except Exception as exc:  # noqa: BLE001
        return Item("truncation", False, time.monotonic() - started, f"error: {exc}")
    seconds = time.monotonic() - started
    calls = run.requests[first:]
    finishes = [call["finish"] for call in calls]
    if finishes and finishes[0] != "length":
        used = calls[0].get("tokens") or 0
        if finishes[0] == "stop" and used < TRUNCATE_AT:
            return Item("truncation", None, seconds,
                        f"not exercised: the whole reply fit in {TRUNCATE_AT} tokens")
        return Item("truncation", False, seconds,
                    f"the first reply used {used or '?'} of {TRUNCATE_AT} tokens but the server "
                    f"reported finish_reason {finishes[0]!r}, not 'length': Vega cannot tell a "
                    "cut-off reply from a finished one")
    blocks = _BLOCK.findall(text)
    try:
        parses = bool(blocks) and bool(ast.parse(max(blocks, key=len)))
    except SyntaxError:
        parses = False
    detail = (f"{len(calls)} replies of ≤{TRUNCATE_AT} tokens; "
              + ("joined into complete code" if parses else
                 "joined text has no complete, parsing code block"
                 + (f" (still cut off after {TRUNCATE_CONTINUATIONS} continuations)"
                    if finishes and finishes[-1] == "length" else "")))
    return Item("truncation", parses and len(calls) > 1, seconds, detail)


def _options(report: Report, requests: list[dict[str, Any]],
             context_length: Callable[[str], int | None]) -> Item:
    profile = models.profile_for_model(report.model)
    problems: list[str] = []
    notes: list[str] = []
    errors = [call["error"] for call in requests if call["error"]]
    if errors:
        problems.append(f"{len(errors)} request(s) failed, first: {errors[0][:120]}")
    wanted = dict(profile.request) if profile else {}
    if wanted:
        missing = [call for call in requests
                   if any(call["kwargs"].get(key) != value for key, value in wanted.items())]
        if missing:
            problems.append(f"{len(missing)}/{len(requests)} requests lacked "
                            f"{', '.join(wanted)}")
        else:
            notes.append(f"{', '.join(f'{k}={v}' for k, v in wanted.items())} sent with all "
                         f"{len(requests)} requests")
    else:
        notes.append("no request fields in the profile" if profile else "no profile")
    if profile is not None and report.ollama:
        configured = context_length(report.model)
        if configured is None:
            notes.append("num_ctx not reported by Ollama")
        elif configured != profile.num_ctx:
            problems.append(f"built model has num_ctx {configured}, profile says "
                            f"{profile.num_ctx} (rebuild: vega model build {profile.name})")
        else:
            notes.append(f"num_ctx {configured}")
    elif profile is not None:
        notes.append(f"num_ctx {profile.num_ctx} is assumed: the endpoint server sets its own "
                     "window")
    return Item("options", not problems, 0.0, "; ".join(problems or notes))


def _loaded() -> list[str]:
    data = server._get_json("/api/ps") or {}
    return [str(item.get("name", "")) for item in data.get("models", [])]


def _memory(report: Report) -> str:
    from . import resources

    parts: list[str] = []
    if report.ollama:
        for item in (server._get_json("/api/ps") or {}).get("models", []):
            if item.get("name") in {report.model, f"{report.model}:latest"}:
                parts.append(f"Ollama reports {int(item.get('size', 0)) / 1024**3:.1f} GB loaded "
                             f"({int(item.get('size_vram', 0)) / 1024**3:.1f} GB on the GPU)")
    try:
        snap = resources.snapshot()
    except Exception:  # noqa: BLE001 - memory is context, never a reason to fail
        return "; ".join(parts)
    if report.ollama:
        parts.append(f"model server RSS {snap.model_rss_mb:,.0f} MB")
    elif os.environ.get("VEGA_MODEL_PROCESSES", "").strip():
        parts.append(f"model server RSS {snap.model_rss_mb:,.0f} MB (Ollama + "
                     "VEGA_MODEL_PROCESSES)")
    else:  # Ollama's RSS would be someone else's model here
        parts.append("model server RSS unknown (set VEGA_MODEL_PROCESSES to its process name)")
    if snap.free_pct is not None:
        parts.append(f"free memory {snap.free_pct}%")
    if snap.swap_used_mb is not None:
        parts.append(f"swap used {snap.swap_used_mb:,.0f} MB")
    return "; ".join(parts)


def run(model: str, *, agent_factory: AgentFactory | None = None,
        before_item: Callable[[], str | None] | None = None,
        context_length: Callable[[str], int | None] | None = None,
        measure_memory: bool = True, on_item: Callable[[Item], None] | None = None) -> Report:
    """Run every item against `model` (the name Vega runs it under, as `--model` resolves it).
    `before_item` (e.g. the resource guard) can stop the check between items."""
    route = models.connection(model)
    profile = models.profile_for_model(model)
    report = Report(model=model, profile=profile.name if profile else None,
                    base_url=route.base_url, served_as=route.model, ollama=not route.is_endpoint)
    requests: list[dict[str, Any]] = []
    checker = _Run(agent_factory or _default_factory, requests)
    saved_model = os.environ.get("VEGA_MODEL")
    os.environ["VEGA_MODEL"] = model
    was_loaded = report.ollama and measure_memory and any(
        name in {model, f"{model}:latest"} for name in _loaded())

    def add(item: Item) -> None:
        report.items.append(item)
        if on_item is not None:
            on_item(item)

    def guarded(name: str, step: Callable[[Path], Item]) -> None:
        if report.stopped:
            add(Item(name, None, 0.0, "skipped: " + report.stopped))
            return
        stop = before_item() if before_item is not None else None
        if stop:
            report.stopped = stop
            add(Item(name, None, 0.0, "skipped: " + stop))
            return
        with tempfile.TemporaryDirectory(prefix="vega-check-") as temp:
            try:
                add(step(_project(Path(temp).resolve())))
            except Exception as exc:  # noqa: BLE001 - report it as this item's failure
                add(Item(name, False, 0.0, f"error: {type(exc).__name__}: {exc}"))

    raw: list[str] = [""]

    def chat_step(project: Path) -> Item:
        item, raw[0] = _chat(checker, project, report)
        return item

    try:
        guarded("chat", chat_step)
        if raw[0]:
            add(_thinking(raw[0]))
        else:
            add(Item("thinking", None, 0.0, "no chat reply to inspect"))
        with _features(on=("question_pipeline",)):
            guarded("question", lambda project: _agent_item(
                checker, project, "question", latency.REQUESTS[0],
                expect_notice="Answering from the code index"))
        with _features(off=tuple(p for p in PIPELINES if p != "direct_pipeline"),
                       on=("direct_pipeline",)):
            guarded("direct-edit", lambda project: _agent_item(
                checker, project, "direct-edit", latency.REQUESTS[3],
                expect_notice="Direct pipeline"))
        with _features(off=PIPELINES):
            guarded("tool-call", lambda project: _agent_item(
                checker, project, "tool-call", latency.REQUESTS[2], need_tool=True))
        guarded("truncation", lambda project: _truncation(checker, project))
        add(_options(report, requests, context_length or server.context_length))
        if measure_memory:
            report.memory = _memory(report)
    finally:
        if saved_model is None:
            os.environ.pop("VEGA_MODEL", None)
        else:
            os.environ["VEGA_MODEL"] = saved_model
        if report.ollama and measure_memory and not was_loaded:
            server.unload(model)  # it was not loaded before the check: free the memory again
    return report

