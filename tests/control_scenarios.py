"""Stop/resume control scenarios: scripted interruptions of a CodingAgent turn, graded from the
run's own record (model requests, tool calls, timings, the message history and the project files),
never from model output.

Used by tests/test_control.py (the release gate, run in CI and selfbuild snapshots, which carry
tests/ but not evals/) and by the private eval tooling, which prints the scenarios as a
table. No model is ever called: every scenario drives `CodingAgent.request_stop()` -- the entry
point the TUI's Esc and Ctrl+C use -- against a scripted fake client.

Interruption points (`point`): stream (while a reply streams), tool (while a tool or command
runs), write (between a write and its check), compact (around context compaction), delegate (a
delegate_task sub-session), continue (a second turn in the same session), restart (a new process
from the saved session) and lab/loops (`vega lab pause|resume|stop`, `vega loops pause|resume`).

Generic checks on every stopped turn:
- latency: run() returns within MAX_STOP_SECONDS of request_stop();
- after_stop: no model request is started and no tool call begins after the stop;
- history: every assistant tool call has exactly one tool result, and no result is orphaned;
- files: every project file is either its old content or a complete version the script wrote;
- answer: the turn ends with agent.STOPPED.
Scenario checks add the resume side: the task finishes and finished steps are not redone.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MAX_STOP_SECONDS = 1.0
COMPLEXITY = (1, 2, 3)
POINTS = ("stream", "tool", "write", "compact", "delegate", "continue", "restart", "lab", "loops")


# -- fake model ----------------------------------------------------------------------------------

def _chunk(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=content, tool_calls=tool_calls), finish_reason=None)])


def _fragment(index: int, *, id: str | None = None, name: str | None = None,
              args: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=args))


class Message(SimpleNamespace):
    """A complete scripted reply: text, or tool calls [(name, args), ...]."""

    def model_dump(self, exclude_none: bool = False) -> dict:
        data: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in self.tool_calls]
        return data


def say(content: str) -> Message:
    return Message(content=content, tool_calls=None)


def calls(*items: tuple[str, dict[str, Any]], tag: str = "c") -> Message:
    tool_calls = [SimpleNamespace(id=f"{tag}-{index}-{name}",
                                  function=SimpleNamespace(name=name, arguments=json.dumps(args)))
                  for index, (name, args) in enumerate(items)]
    return Message(content=None, tool_calls=tool_calls)


def chunks_of(message: Message) -> list[SimpleNamespace]:
    """A reply as streamed chunks: text in words, each tool call in three fragments."""
    out = [_chunk(word) for word in _words(message.content or "")]
    for index, call in enumerate(message.tool_calls or []):
        raw = call.function.arguments
        half = len(raw) // 2
        out += [_chunk(tool_calls=[_fragment(index, id=call.id, name=call.function.name, args="")]),
                _chunk(tool_calls=[_fragment(index, args=raw[:half])]),
                _chunk(tool_calls=[_fragment(index, args=raw[half:])])]
    return out


def _words(text: str) -> list[str]:
    parts = text.split(" ")
    return [part + (" " if index < len(parts) - 1 else "") for index, part in enumerate(parts) if part]


@dataclass
class Hook:
    """A scripted reply with side effects: `before(run)` when the request arrives; streamed chunks
    call `during(run, index)` before chunk `index` is handed over; `delay` seconds per chunk."""

    reply: Message
    before: Callable[[Run], None] | None = None
    during: Callable[[Run, int], None] | None = None
    delay: float = 0.0


class FakeClient:
    """Replays a script; once it runs out, every request gets `fallback` (a plain final answer), so a
    harness send-back (review, finish gates) never crashes the script."""

    def __init__(self, run: Run, script: list[Message | Hook], fallback: str = "Done.", *,
                 label: str = "main"):
        self.run, self.script, self.fallback, self.label = run, list(script), fallback, label
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Any:
        self.run.requests.append({"who": self.label, "after_stop": self.run.stopped(),
                                  "messages": list(kwargs.get("messages", []))})
        item = self.script.pop(0) if self.script else say(self.fallback)
        hook = item if isinstance(item, Hook) else Hook(item)
        if hook.before is not None:
            hook.before(self.run)
        if kwargs.get("stream"):
            return self._stream(hook)
        if hook.during is not None:
            hook.during(self.run, 0)
        return SimpleNamespace(choices=[SimpleNamespace(message=hook.reply, finish_reason="stop")])

    def _stream(self, hook: Hook) -> Iterator[SimpleNamespace]:
        for index, piece in enumerate(chunks_of(hook.reply)):
            if hook.during is not None:
                hook.during(self.run, index)
            if hook.delay:
                time.sleep(hook.delay)
            self.run.chunks_sent += 1
            yield piece


# -- one run -------------------------------------------------------------------------------------

@dataclass
class Run:
    """What one scenario did, recorded by the harness (never by the model)."""

    project: Path
    agent: Any = None
    requests: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)  # tool_call events
    results: list[dict[str, Any]] = field(default_factory=list)  # tool_result events
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    checks_run: list[dict[str, Any]] = field(default_factory=list)  # sandbox.run_check calls
    stop_at: float | None = None
    returned_at: float | None = None
    answers: list[str] = field(default_factory=list)
    chunks_sent: int = 0
    children: int = 0  # delegate_task sub-agents created
    allowed: dict[str, set[str]] = field(default_factory=dict)  # path -> complete contents
    before: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def stopped(self) -> bool:
        return self.stop_at is not None

    def stop(self) -> None:
        """Press Esc: what the TUI does (agent.request_stop), timed."""
        if self.stop_at is None:
            self.stop_at = time.monotonic()
        self.agent.request_stop()

    def stop_later(self, seconds: float) -> None:
        threading.Timer(seconds, self.stop).start()

    def on_event(self, kind: str, data: dict[str, Any]) -> None:
        self.events.append((kind, data))
        if kind == "tool_call":
            self.tools.append({"name": data.get("name"), "args": data.get("args"),
                               "after_stop": self.stopped()})
        elif kind == "tool_result":
            self.results.append({"name": data.get("name"), "result": str(data.get("result", ""))})

    def tool_names(self) -> list[str]:
        return [str(t["name"]) for t in self.tools]

    def writes_to(self, path: str) -> int:
        return sum(1 for t in self.tools if t["name"] in ("write_file", "edit_file")
                   and isinstance(t["args"], dict) and t["args"].get("path") == path)

    def turn(self, task: str) -> str:
        answer = self.agent.run(task)
        self.returned_at = time.monotonic()
        self.answers.append(answer)
        return answer


def snapshot(project: Path) -> dict[str, str]:
    out = {}
    for path in sorted(project.rglob("*")):
        relative = path.relative_to(project)
        if path.is_file() and not any(part in (".git", ".vega", "__pycache__") for part in relative.parts):
            out[relative.as_posix()] = path.read_text(encoding="utf-8", errors="replace")
    return out


# -- checks --------------------------------------------------------------------------------------

def check_latency(run: Run) -> str:
    if run.stop_at is None:
        return "the scenario never pressed stop"
    if run.returned_at is None:
        return "run() never returned"
    seconds = run.returned_at - run.stop_at
    return "" if seconds < MAX_STOP_SECONDS else f"stop took {seconds:.2f}s (limit {MAX_STOP_SECONDS}s)"


def check_after_stop(run: Run) -> str:
    late = [r["who"] for r in run.requests if r["after_stop"]]
    if late:
        return f"{len(late)} model request(s) started after the stop ({', '.join(late)})"
    tools = [t["name"] for t in run.tools if t["after_stop"]]
    if tools:
        return f"tool call(s) began after the stop: {', '.join(map(str, tools))}"
    checks = [c["command"] for c in run.checks_run if c["after_stop"]]
    if checks:
        return f"sandboxed check(s) ran after the stop: {', '.join(checks)}"
    return ""


def history_problem(messages: list[dict[str, Any]]) -> str:
    """Every assistant tool call has exactly one tool result after it; no orphan results."""
    open_ids: dict[str, int] = {}
    seen: set[str] = set()
    for index, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            call_id = call.get("id")
            if call_id in seen or call_id in open_ids:
                return f"message {index}: tool call id {call_id!r} used twice"
            open_ids[call_id] = index
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in open_ids:
                return f"message {index}: tool result {call_id!r} has no open tool call"
            open_ids.pop(call_id)
            seen.add(call_id)
        elif open_ids and message.get("role") in ("assistant", "user") and not message.get("tool_calls"):
            return f"message {index}: {message.get('role')} message before results for {sorted(open_ids)}"
    if open_ids:
        return f"tool calls without a result: {sorted(open_ids)}"
    return ""


def check_history(run: Run) -> str:
    problem = history_problem(run.agent.messages)
    return f"history: {problem}" if problem else ""


def check_files(run: Run) -> str:
    """Each file is its old content or a complete content the script wrote (nothing partial)."""
    after = snapshot(run.project)
    for path in sorted(set(after) | set(run.before)):
        content = after.get(path)
        options = {run.before.get(path)} | run.allowed.get(path, set())
        if content not in options:
            shown = "deleted" if content is None else f"{len(content)} chars"
            return f"{path}: neither its old content nor a complete written version ({shown})"
    return ""


def check_stopped_answer(run: Run, turn: int = 0) -> str:
    from vega.agent import STOPPED

    answer = run.answers[turn] if len(run.answers) > turn else None
    return "" if answer == STOPPED else f"turn {turn + 1} answered {str(answer)[:60]!r}, not STOPPED"


STOP_CHECKS = (check_latency, check_after_stop, check_history, check_files, check_stopped_answer)


# -- the scenario table --------------------------------------------------------------------------

@dataclass
class Scenario:
    id: str
    point: str
    complexity: int
    what: str
    body: Callable[[Run], list[str]]  # runs the scenario, returns its own problems
    needs_sandbox: bool = False
    stop_checks: bool = True  # also run STOP_CHECKS (agent scenarios)
    known_bug: str = ""  # a real defect this scenario exposes (xfail, strict) until fixed


SCENARIOS: list[Scenario] = []


def scenario(id: str, point: str, complexity: int, what: str, **options: Any):
    def register(body: Callable[[Run], list[str]]) -> Callable[[Run], list[str]]:
        if point not in POINTS or complexity not in COMPLEXITY or not id.startswith(f"ctl-{point}-"):
            raise ValueError(f"{id}: bad point/complexity/id")
        SCENARIOS.append(Scenario(id, point, complexity, what, body, **options))
        return body
    return register


def make_agent(run: Run, script: list[Message | Hook], *, stream: bool = True,
               **options: Any) -> Any:
    from vega.agent import CodingAgent

    options.setdefault("allow_commands", False)
    agent = CodingAgent(run.project, stream=stream, on_event=run.on_event, **options)
    agent.client = FakeClient(run, script)
    # Routing sends some requests to pipelines with their own prompts; these scenarios are about
    # the agent loop that every route ends in, so the loop is driven directly.
    agent.route_requests = False
    run.agent = agent
    run.before = snapshot(run.project)
    return agent


def write(path: str, content: str) -> tuple[str, dict[str, Any]]:
    return ("write_file", {"path": path, "content": content})


def read(path: str) -> tuple[str, dict[str, Any]]:
    return ("read_file", {"path": path})


def allow(run: Run, path: str, *contents: str) -> None:
    run.allowed.setdefault(path, set()).update(contents)


def when_tool(run: Run, name: str, action: Callable[[], None], *, on: str = "tool_result",
              nth: int = 1) -> None:
    """Call `action` when the nth `on` event (tool_call / tool_result) for tool `name` arrives."""
    previous = run.agent.on_event
    seen = [0]

    def handler(kind: str, data: dict[str, Any]) -> None:
        previous(kind, data)
        if kind == on and data.get("name") == name:
            seen[0] += 1
            if seen[0] == nth:
                action()

    run.agent.on_event = handler


def expect(condition: bool, problem: str) -> list[str]:
    return [] if condition else [problem]


BIG = "".join(f"VALUE_{n} = {n}\n" for n in range(6000))  # ~80 KB module


# == stream ======================================================================================

@scenario("ctl-stream-text", "stream", 1, "Esc while a text reply streams: the rest is dropped")
def _(run: Run) -> list[str]:
    hook = Hook(say("Let me search every file again and then rewrite the whole module."),
                during=lambda r, i: r.stop() if i == 3 else None)
    make_agent(run, [hook])
    run.turn("tidy up the code")
    shown = "".join(d.get("delta", "") for k, d in run.events if k == "text")
    return expect("rewrite" not in shown, f"text after the stop was shown: {shown!r}") + \
        expect(len(run.requests) == 1, f"{len(run.requests)} requests")


@scenario("ctl-stream-toolcall", "stream", 1,
          "Esc while a tool call streams: the half-received call never runs")
def _(run: Run) -> list[str]:
    hook = Hook(calls(write("notes.txt", "draft\n")), during=lambda r, i: r.stop() if i == 1 else None)
    make_agent(run, [hook])
    run.turn("write the notes")
    return expect(not (run.project / "notes.txt").exists(), "notes.txt was written") + \
        expect(not run.tools, f"tools ran: {run.tool_names()}")


@scenario("ctl-stream-slow", "stream", 2,
          "Esc from another thread during a slow stream (5 s if not stopped) returns at once")
def _(run: Run) -> list[str]:
    hook = Hook(say(" ".join(["word"] * 25)), delay=0.2,
                before=lambda r: r.stop_later(0.3))
    make_agent(run, [hook])
    run.turn("explain the project")
    return expect(run.chunks_sent < 10, f"{run.chunks_sent} of 25 chunks consumed after the stop")


@scenario("ctl-stream-between", "stream", 1,
          "Esc between steps (after a tool result): the next request is never sent")
def _(run: Run) -> list[str]:
    (run.project / "a.py").write_text("x = 1\n")
    make_agent(run, [calls(read("a.py")), calls(read("a.py"), tag="d"), say("Done.")])
    when_tool(run, "read_file", run.stop)
    run.turn("look at a.py")
    return expect(len(run.requests) == 1, f"{len(run.requests)} requests (expected 1)")


@scenario("ctl-stream-nonstream", "stream", 2,
          "Esc during a non-streamed reply (server without streaming): the reply is discarded")
def _(run: Run) -> list[str]:
    hook = Hook(calls(write("out.txt", "x\n")), before=lambda r: r.stop())
    make_agent(run, [hook], stream=False)
    run.turn("write out.txt")
    return expect(not (run.project / "out.txt").exists(), "the discarded reply's write ran")


# == tool ========================================================================================

SLEEP = 'python3 -c "import time; time.sleep(20)"'


@scenario("ctl-tool-command", "tool", 2,
          "Esc while run_command runs a 20 s command: it is killed and the turn ends in < 1 s")
def _(run: Run) -> list[str]:
    make_agent(run, [calls(("run_command", {"command": SLEEP})), say("Done.")],
               allow_commands=True, approve_command=lambda command: True)
    when_tool(run, "run_command", lambda: run.stop_later(0.3), on="tool_call")
    started = time.monotonic()
    run.turn("run the slow job")
    from vega import sandbox

    result = run.results[0]["result"] if run.results else ""
    return expect(time.monotonic() - started < 5, "the command was not killed") + \
        expect(result == sandbox.STOPPED_CHECK, f"command result {result[:60]!r}")


@scenario("ctl-tool-check", "tool", 2,
          "Esc while run_check runs a 20 s sandboxed check: it is killed in < 1 s",
          needs_sandbox=True)
def _(run: Run) -> list[str]:
    make_agent(run, [calls(("run_check", {"command": SLEEP})), say("Done.")])
    when_tool(run, "run_check", lambda: run.stop_later(0.3), on="tool_call")
    started = time.monotonic()
    run.turn("run the checks")
    return expect(time.monotonic() - started < 5, "the check was not killed")


@scenario("ctl-tool-parallel", "tool", 2,
          "Esc during the first of two calls in one reply: the second gets a 'not run' result")
def _(run: Run) -> list[str]:
    from vega.agent import NOT_RUN

    (run.project / "a.py").write_text("x = 1\n")
    make_agent(run, [calls(read("a.py"), write("b.py", "y = 2\n")), say("Done.")])
    when_tool(run, "read_file", run.stop, on="tool_call")
    run.turn("copy a into b")
    tool_texts = [m.get("content") for m in run.agent.messages if m.get("role") == "tool"]
    return expect(not (run.project / "b.py").exists(), "b.py was written after the stop") + \
        expect(tool_texts[-1:] == [NOT_RUN], f"last tool result {tool_texts[-1:]!r}")


@scenario("ctl-tool-three", "tool", 3,
          "Esc after the first of three writes: one complete file, two calls answered 'not run'")
def _(run: Run) -> list[str]:
    from vega.agent import NOT_RUN

    make_agent(run, [calls(write("one.py", "A = 1\n"), write("two.py", "B = 2\n"),
                           read("one.py")), say("Done.")])
    allow(run, "one.py", "A = 1\n")
    when_tool(run, "write_file", run.stop)
    run.turn("make the modules")
    not_run = sum(m.get("content") == NOT_RUN for m in run.agent.messages if m.get("role") == "tool")
    return expect((run.project / "one.py").exists(), "one.py missing") + \
        expect(not (run.project / "two.py").exists(), "two.py written after the stop") + \
        expect(not_run == 2, f"{not_run} 'not run' results (expected 2)")


# == write =======================================================================================

@scenario("ctl-write-beforecheck", "write", 1,
          "Esc right after a write: the check that would follow never runs")
def _(run: Run) -> list[str]:
    make_agent(run, [calls(write("calc.py", "def add(a, b):\n    return a + b\n")),
                     calls(("run_check", {"command": "python -m pytest -q"})), say("Done.")])
    allow(run, "calc.py", "def add(a, b):\n    return a + b\n")
    when_tool(run, "write_file", run.stop)
    run.turn("write calc.add")
    return expect("run_check" not in run.tool_names(), "run_check ran after the stop")


@scenario("ctl-write-edit", "write", 2,
          "Esc right after an edit: the file holds the whole edit, no check runs")
def _(run: Run) -> list[str]:
    source = "def area(w, h):\n    return w + h\n\n\ndef label():\n    return 'box'\n"
    (run.project / "geo.py").write_text(source)
    make_agent(run, [calls(read("geo.py")),
                     calls(("edit_file", {"path": "geo.py", "old_text": "return w + h",
                                          "new_text": "return w * h"}), tag="e"),
                     calls(("run_check", {"command": "python -m pytest -q"}), tag="k")])
    allow(run, "geo.py", source.replace("w + h", "w * h"))
    when_tool(run, "edit_file", run.stop)
    run.turn("fix area")
    return expect((run.project / "geo.py").read_text() == source.replace("w + h", "w * h"),
                  "geo.py does not hold the complete edit") + \
        expect("run_check" not in run.tool_names(), "run_check ran after the stop")


@scenario("ctl-write-big", "write", 3,
          "Esc as an 80 KB write starts: the file is absent or complete, never partial")
def _(run: Run) -> list[str]:
    make_agent(run, [calls(write("table.py", BIG)), say("Done.")])
    allow(run, "table.py", BIG)
    when_tool(run, "write_file", run.stop, on="tool_call")
    run.turn("generate the table module")
    return []  # check_files covers it


@scenario("ctl-write-finish", "write", 3,
          "Esc while the final answer streams after a write: no finish checks run after the stop")
def _(run: Run) -> list[str]:
    tests = "from calc2 import mul\n\n\ndef test_mul():\n    assert mul(2, 3) == 6\n"
    (run.project / "tests").mkdir()
    (run.project / "tests" / "test_calc2.py").write_text(tests)
    code = "def mul(a, b):\n    return a * b\n"
    final = Hook(say("I wrote mul and it multiplies the two numbers as the tests expect."),
                 during=lambda r, i: r.stop() if i == 2 else None)
    make_agent(run, [calls(write("calc2.py", code)), final])
    allow(run, "calc2.py", code)
    run.turn("add calc2.mul so tests/test_calc2.py passes")
    return []


# == compact =====================================================================================

def _filler(n: int) -> str:
    return "".join(f"# line {i}: " + "padding " * 12 + "\n" for i in range(n))


def _compacting_agent(run: Run, script: list[Message | Hook]) -> Any:
    for index in range(4):
        (run.project / f"part{index}.py").write_text(_filler(120))
    return make_agent(run, script, num_ctx=4096)


@scenario("ctl-compact-during", "compact", 2,
          "Esc while the history is being compacted: no model request follows")
def _(run: Run) -> list[str]:
    from vega import context

    script = [calls(read(f"part{i}.py"), tag=f"r{i}") for i in range(4)] + [say("Done.")]
    _compacting_agent(run, script)
    original = context.compact

    def compact(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if result is not None and not run.stopped():
            run.stop()  # the user presses Esc while compaction runs (before it returns)
        return result

    with _patched(context, "compact", compact):
        run.turn("read every part")
    compacted = any(k == "notice" and str(d.get("message", "")).startswith("Context compacted")
                    for k, d in run.events)
    return expect(compacted, "compaction never ran (the scenario did not reach its point)")


@scenario("ctl-compact-continue", "compact", 3,
          "Esc before compaction, then continue: the compacted history keeps every call paired",
          stop_checks=False)  # checks its stopped first turn itself
def _(run: Run) -> list[str]:
    script = [calls(read("part0.py"), tag="r0"), calls(read("part1.py"), read("part2.py"), tag="r1")]
    agent = _compacting_agent(run, script)
    when_tool(run, "read_file", run.stop, nth=2)
    run.turn("read every part")
    problems = [f"turn 1: {p}" for p in (c(run) for c in STOP_CHECKS) if p]
    notices = _compactions(run)
    agent.client = FakeClient(run, [calls(read("part2.py"), tag="s0"),
                                    calls(read("part3.py"), tag="s1"), say("All parts read.")])
    run.stop_at = None  # the next turn is not stopped
    for request in run.requests:
        request["after_stop"] = False
    agent.run("continue")
    compacted = _compactions(run) > notices
    second = history_problem(agent.messages)
    return problems + expect(compacted, "compaction never ran") + \
        expect(not second, f"history after continue: {second}")


def _compactions(run: Run) -> int:
    return sum(k == "notice" and str(d.get("message", "")).startswith("Context compacted")
               for k, d in run.events)


@scenario("ctl-compact-summary", "compact", 3,
          "Esc after compaction: the stopped turn still keeps its goal and a valid history")
def _(run: Run) -> list[str]:
    script = [calls(read(f"part{i}.py"), tag=f"r{i}") for i in range(4)] + [say("Done.")]
    agent = _compacting_agent(run, script)

    def on_notice(kind: str, data: dict[str, Any]) -> None:
        if kind == "tool_result" and any(
                k == "notice" and str(d.get("message", "")).startswith("Context compacted")
                for k, d in run.events) and not run.stopped():
            run.stop()

    previous = agent.on_event
    agent.on_event = lambda kind, data: (previous(kind, data), on_notice(kind, data))
    run.turn("read every part")
    goal = [m for m in agent.messages if m.get("role") == "user"
            and "read every part" in str(m.get("content", ""))]
    return expect(bool(goal), "the goal message was lost") + \
        expect(run.stopped(), "compaction never ran before the end (scenario did not reach it)")


# == delegate ====================================================================================

@contextlib.contextmanager
def _patched(target: Any, name: str, value: Any) -> Iterator[None]:
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


@contextlib.contextmanager
def _child_model(run: Run, script: list[Message | Hook]) -> Iterator[None]:
    """delegate_task builds its sub-agent's client with agent.OpenAI: hand it a scripted one."""
    from vega import agent as agent_module

    def factory(*args: Any, **kwargs: Any) -> FakeClient:
        run.children += 1
        return FakeClient(run, script, fallback="Child done.", label="child")

    with _patched(agent_module, "OpenAI", factory):
        yield


@scenario("ctl-delegate-before", "delegate", 2,
          "Esc before a delegate_task call in the same reply runs: no sub-agent is started")
def _(run: Run) -> list[str]:
    (run.project / "a.py").write_text("x = 1\n")
    make_agent(run, [calls(read("a.py"), ("delegate_task", {"task": "survey the repo"})),
                     say("Done.")])
    when_tool(run, "read_file", run.stop)
    with _child_model(run, [say("found nothing")]):
        run.turn("survey and fix")
    return expect(run.children == 0, f"{run.children} sub-agent(s) started after the stop")


@scenario("ctl-delegate-during", "delegate", 3,
          "Esc while a delegated sub-session works: the sub-session stops too")
def _(run: Run) -> list[str]:
    for name in ("a.py", "b.py", "c.py"):
        (run.project / name).write_text("x = 1\n")
    make_agent(run, [calls(("delegate_task", {"task": "read a, b and c and summarise"})),
                     say("Done.")])
    child = [Hook(calls(read("a.py"), tag="k0"), before=lambda r: r.stop()),
             calls(read("b.py"), tag="k1"), calls(read("c.py"), tag="k2"), say("Summary.")]
    with _child_model(run, child):
        run.turn("summarise the modules")
    return expect(run.children == 1, f"{run.children} sub-agents")


# == continue ====================================================================================

def _second_turn(run: Run, script: list[Message | Hook], task: str = "continue") -> str:
    run.agent.client = FakeClient(run, script)
    answer = run.agent.run(task)
    run.answers.append(answer)
    return answer


def _first_request_of_turn(run: Run, task: str) -> list[dict[str, Any]]:
    for request in run.requests:
        if any(m.get("role") == "user" and m.get("content") == task for m in request["messages"]):
            return request["messages"]
    return []


@scenario("ctl-continue-finish", "continue", 2,
          "Stop after a write, then 'continue': the task finishes and the write is not redone")
def _(run: Run) -> list[str]:
    make_agent(run, [calls(write("app.py", "def main():\n    return 1\n")),
                     calls(write("cli.py", "from app import main\n"), tag="w2")])
    allow(run, "app.py", "def main():\n    return 1\n")
    when_tool(run, "write_file", run.stop)
    run.turn("write app.py then cli.py")
    problems = [p for p in (c(run) for c in STOP_CHECKS) if p]
    _second_turn(run, [calls(write("cli.py", "from app import main\n"), tag="w3"), say("Done.")])
    history = _first_request_of_turn(run, "continue")
    carried = any(m.get("role") == "tool" and str(m.get("content", "")).startswith("Wrote")
                  for m in history)
    return problems + \
        expect((run.project / "cli.py").read_text() == "from app import main\n", "cli.py missing") + \
        expect(run.writes_to("app.py") == 1, f"app.py written {run.writes_to('app.py')} times") + \
        expect(carried, "the continue turn's request lacks the first turn's write result") + \
        expect(run.answers[-1] == "Done.", f"continue answered {run.answers[-1][:60]!r}")


@scenario("ctl-continue-notrun", "continue", 2,
          "Stopped calls are visible as 'not run' to the next turn, which runs only those")
def _(run: Run) -> list[str]:
    from vega.agent import NOT_RUN

    make_agent(run, [calls(write("one.py", "A = 1\n"), write("two.py", "B = 2\n"))])
    allow(run, "one.py", "A = 1\n")
    when_tool(run, "write_file", run.stop)
    run.turn("make one.py and two.py")
    problems = [p for p in (c(run) for c in STOP_CHECKS) if p]
    _second_turn(run, [calls(write("two.py", "B = 2\n"), tag="n"), say("Done.")])
    history = _first_request_of_turn(run, "continue")
    shows = any(m.get("role") == "tool" and m.get("content") == NOT_RUN for m in history)
    return problems + expect(shows, "the next turn does not see which call was not run") + \
        expect(run.writes_to("one.py") == 1, "one.py was written again") + \
        expect((run.project / "two.py").read_text() == "B = 2\n", "two.py missing")


@scenario("ctl-continue-twice", "continue", 3,
          "Stop, continue, stop again, continue: each file written once, history valid throughout")
def _(run: Run) -> list[str]:
    make_agent(run, [calls(write("s1.py", "S = 1\n"), tag="a"), calls(write("s2.py", "S = 2\n"), tag="b")])
    when_tool(run, "write_file", run.stop)
    run.turn("write s1, s2 and s3")
    problems = [f"turn 1: {p}" for p in (check_history(run), check_stopped_answer(run, 0)) if p]
    # turn 2: writes s2, stopped after it
    run.stop_at = None
    run.agent.client = FakeClient(run, [calls(write("s2.py", "S = 2\n"), tag="c"),
                                        calls(write("s3.py", "S = 3\n"), tag="d")])
    seen = [0]
    previous = run.agent.on_event

    def stop_on_s2(kind: str, data: dict[str, Any]) -> None:
        previous(kind, data)
        if kind == "tool_result" and data.get("name") == "write_file":
            seen[0] += 1
            if seen[0] == 1:
                run.stop()

    run.agent.on_event = stop_on_s2
    run.answers.append(run.agent.run("continue"))
    run.returned_at = time.monotonic()
    problems += [f"turn 2: {p}" for p in (check_history(run), check_stopped_answer(run, 1),
                                         check_after_stop(run), check_latency(run)) if p]
    run.agent.on_event = previous
    run.stop_at = None
    _second_turn(run, [calls(write("s3.py", "S = 3\n"), tag="e"), say("Done.")])
    files = {name: (run.project / name).read_text() if (run.project / name).exists() else None
             for name in ("s1.py", "s2.py", "s3.py")}
    counts = {name: run.writes_to(name) for name in files}
    return problems + \
        expect(files == {"s1.py": "S = 1\n", "s2.py": "S = 2\n", "s3.py": "S = 3\n"}, f"files {files}") + \
        expect(counts == {"s1.py": 1, "s2.py": 1, "s3.py": 1}, f"writes per file {counts}") + \
        expect(not history_problem(run.agent.messages), "final history invalid")


@scenario("ctl-continue-stale", "continue", 1,
          "A stop pressed between turns does not cancel the next turn", stop_checks=False)
def _(run: Run) -> list[str]:
    make_agent(run, [calls(write("ok.txt", "ok\n")), say("Done.")])
    run.agent.request_stop()  # a late Esc after the previous turn had already ended
    answer = run.turn("write ok.txt")
    return expect(answer == "Done.", f"answered {answer[:60]!r}") + \
        expect((run.project / "ok.txt").exists(), "ok.txt missing")


# == restart =====================================================================================

def _session_folder(run: Run) -> Path:
    from vega import server

    return Path(server.STATE_DIR) / "sessions"


@scenario("ctl-restart-transcript", "restart", 2,
          "The session saved after a stopped turn parses and pairs every call with a result")
def _(run: Run) -> list[str]:
    from vega.transcripts import SessionLog

    make_agent(run, [calls(write("one.py", "A = 1\n"), write("two.py", "B = 2\n"))])
    allow(run, "one.py", "A = 1\n")
    when_tool(run, "write_file", run.stop)
    log = SessionLog(_session_folder(run), run.project)
    started = time.monotonic()
    answer = run.turn("make one and two")
    log.record_turn("make one and two", answer, time.monotonic() - started, run.agent.request_log)
    log.save(run.agent.messages)
    log.close()
    problems = [p for p in (c(run) for c in STOP_CHECKS) if p]
    data = json.loads(log.path.read_text(encoding="utf-8"))
    messages = data["messages"]
    calls_made = sum(len(m.get("calls", [])) for m in messages)
    results = sum(m.get("role") == "tool" for m in messages)
    return problems + expect(calls_made == results == 2, f"saved {calls_made} calls, {results} results") + \
        expect(data["turns"][0]["answer"] == answer, "the saved turn lost its answer")


@scenario("ctl-restart-digest", "restart", 3,
          "Stop, end the process, start a new session: it is told what changed and finishes "
          "without rewriting the finished file")
def _(run: Run) -> list[str]:
    from vega.agent import CodingAgent
    from vega.transcripts import SessionLog

    (run.project / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    make_agent(run, [calls(write("one.py", "A = 1\n"), write("two.py", "B = 2\n"))])
    allow(run, "one.py", "A = 1\n")
    when_tool(run, "write_file", run.stop)
    log = SessionLog(_session_folder(run), run.project)
    answer = run.turn("make one.py and two.py")
    log.record_turn("make one.py and two.py", answer, 0.0, run.agent.request_log)
    log.save(run.agent.messages)
    log.close()
    problems = [p for p in (c(run) for c in STOP_CHECKS if c is not check_files) if p]
    # a new process: a fresh agent on the same project
    fresh = CodingAgent(run.project, stream=True, on_event=run.on_event, allow_commands=False)
    fresh.route_requests = False
    fresh.client = FakeClient(run, [calls(write("two.py", "B = 2\n"), tag="r"), say("Done.")])
    run.agent = fresh
    run.stop_at = None
    run.turn("finish making one.py and two.py")
    first = _first_request_of_turn(run, "finish making one.py and two.py")
    system = str(first[0].get("content", "")) if first else ""
    return problems + \
        expect("one.py" in system and "make one.py and two.py" in system,
               "the new session's prompt does not carry the stopped session's digest") + \
        expect(run.writes_to("one.py") == 1, "one.py was rewritten after the restart") + \
        expect((run.project / "two.py").read_text() == "B = 2\n", "two.py missing")


# == lab and loops (state switches, no agent) ====================================================

def _pacer(sleep: Callable[[float], None]) -> Any:
    from vega import evals

    return evals._Pacer(None, 0.0, sleep)


@scenario("ctl-lab-resume", "lab", 1, "vega lab pause holds the next attempt; resume releases it",
          stop_checks=False)
def _(run: Run) -> list[str]:
    from vega import evals

    pause = evals.control_file("pause")
    pause.parent.mkdir(parents=True, exist_ok=True)
    pause.write_text("paused\n")
    polls = []

    def sleep(seconds: float) -> None:
        polls.append(seconds)
        if len(polls) == 2:
            pause.unlink()  # `vega lab resume`

    with contextlib.redirect_stdout(open(os.devnull, "w")):
        reason = _pacer(sleep).before_attempt()
    return expect(reason == "", f"stopped: {reason}") + expect(len(polls) == 2, f"{len(polls)} polls")


@scenario("ctl-lab-stop", "lab", 2,
          "vega lab stop while paused ends the run and clears both switches", stop_checks=False)
def _(run: Run) -> list[str]:
    from vega import evals

    pause, stop = evals.control_file("pause"), evals.control_file("stop")
    pause.parent.mkdir(parents=True, exist_ok=True)
    pause.write_text("paused\n")
    reason = _pacer(lambda seconds: stop.write_text("stop\n")).before_attempt()
    return expect("stopped by you" in reason, f"reason {reason!r}") + \
        expect(not pause.exists() and not stop.exists(), "a switch file was left behind")


@scenario("ctl-loops-one", "loops", 1, "vega loops pause NAME pauses only that loop",
          stop_checks=False)
def _(run: Run) -> list[str]:
    from vega.loops import state

    state.set_paused("bug-watch", True)
    first = (state.is_paused("bug-watch"), state.is_paused("docs"))
    state.set_paused("bug-watch", False)
    return expect(first == (True, False), f"paused after pause: {first}") + \
        expect(not state.is_paused("bug-watch"), "still paused after resume")


@scenario("ctl-loops-all", "loops", 2,
          "Pause every loop: resuming one by name keeps it paused; plain resume clears all",
          stop_checks=False)
def _(run: Run) -> list[str]:
    from vega.loops import state

    state.set_paused(None, True)
    state.set_paused("bug-watch", False)
    still = state.is_paused("bug-watch")
    state.set_paused(None, False)
    return expect(still, "resuming one loop by name lifted the pause on every loop") + \
        expect(not state.paused_names(), f"left paused: {state.paused_names()}")


# -- running -------------------------------------------------------------------------------------

@dataclass
class Outcome:
    id: str
    point: str
    complexity: int
    problems: list[str]
    seconds: float
    stop_seconds: float | None
    tools: list[str]
    requests: int
    known_bug: str = ""
    skipped: str = ""

    @property
    def passed(self) -> bool:
        return not self.problems and not self.skipped


@contextlib.contextmanager
def isolated() -> Iterator[Path]:
    """A scratch VEGA_HOME and the test defaults (no extensions, no model discovery, no debug
    logs), for running scenarios outside pytest; everything is restored afterwards."""
    from vega import extensions, server

    with tempfile.TemporaryDirectory(prefix="vega-control-home-") as home:
        saved = {name: os.environ.get(name) for name in (
            "VEGA_HOME", "VEGA_DEBUG", "VEGA_FEATURES_ON", "VEGA_FEATURES_OFF", "VEGA_NUM_CTX",
            "VEGA_TEACH", extensions.DISABLE_ENV)}
        for name in saved:
            os.environ.pop(name, None)
        os.environ["VEGA_HOME"] = home
        os.environ[extensions.DISABLE_ENV] = "1"
        extensions.reset()
        old_state, old_ctx = server.STATE_DIR, server.context_length
        server.STATE_DIR = Path(home)
        server.context_length = lambda model=None: None
        try:
            yield Path(home)
        finally:
            server.STATE_DIR, server.context_length = old_state, old_ctx
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            extensions.reset()


def run_scenario(item: Scenario) -> Outcome:
    """Run one scenario in a fresh project; its problems are the scenario's own checks plus, for
    agent scenarios, STOP_CHECKS on the stopped turn."""
    from vega import sandbox

    if item.needs_sandbox and not sandbox.available():
        return Outcome(item.id, item.point, item.complexity, [], 0.0, None, [], 0,
                       item.known_bug, skipped="needs the macOS sandbox")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vega-control-") as temp:
        run = Run(Path(temp).resolve())
        original = sandbox.run_check

        def counted(command: str, *args: Any, **kwargs: Any) -> str:
            run.checks_run.append({"command": command, "after_stop": run.stopped()})
            return original(command, *args, **kwargs)

        with _patched(sandbox, "run_check", counted):
            crashed = False
            try:
                problems = item.body(run)
            except Exception as exc:  # noqa: BLE001  (a crash is the scenario's problem, shown)
                problems, crashed = [f"crashed: {type(exc).__name__}: {exc}"], True
            # continue/restart scenarios check their stopped turn themselves
            if (item.stop_checks and run.agent is not None and not crashed
                    and item.point not in ("continue", "restart")):
                problems += [p for p in (check(run) for check in STOP_CHECKS) if p]
        stop_seconds = (run.returned_at - run.stop_at
                        if run.stop_at is not None and run.returned_at is not None
                        and run.returned_at >= run.stop_at else None)
        return Outcome(item.id, item.point, item.complexity, problems,
                       round(time.monotonic() - started, 2),
                       round(stop_seconds, 3) if stop_seconds is not None else None,
                       run.tool_names(), len(run.requests), item.known_bug)


def run_all(only: str = "") -> list[Outcome]:
    with isolated():
        return [run_scenario(item) for item in SCENARIOS if only in item.id]
