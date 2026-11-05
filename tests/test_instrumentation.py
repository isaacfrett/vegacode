"""Instrumentation (systems review 2026-10-29, "do now" 1): `acted` marks once per turn per switch,
nothing logged without debug mode, switches marking where they act, and the window composition
record per model request. Logging only: nothing here may change what a turn does."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vega import context, debuglog, features, server, window_log
from vega.agent import CodingAgent


def _entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _acted(path: Path) -> list[dict]:
    return [e for e in _entries(path) if e.get("stage") == "acted"]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(server, "STATE_DIR", tmp_path / "home")
    monkeypatch.setattr(debuglog, "_current", None)
    features.reset_acted()
    yield tmp_path / "home"
    features.reset_acted()


@pytest.fixture
def debug_on(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VEGA_DEBUG", "1")
    return home


# -- the helper -------------------------------------------------------------------------------------

def test_acted_is_logged_once_per_turn_per_switch(debug_on: Path) -> None:
    log = debuglog.start("unit")
    features.mark_acted("lint", findings=2)
    features.mark_acted("lint", findings=5)
    features.mark_acted("nudge_act")
    rows = _acted(log.path)
    assert [(r["feature"], r["switch"]) for r in rows] == [("lint", "lint"), ("nudge_act", "nudge_act")]
    assert rows[0]["findings"] == 2
    features.new_turn()  # a new turn logs each switch once again; the attempt's set keeps them
    features.mark_acted("lint")
    assert [r["feature"] for r in _acted(log.path)] == ["lint", "nudge_act", "lint"]
    assert features.acted_names() == {"lint", "nudge_act"}


def test_acted_logs_nothing_without_debug_but_the_attempt_still_knows(home: Path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(debuglog, "record", lambda *a, **k: calls.append((a, k)))
    features.mark_acted("compaction", dropped=3)
    assert calls == []  # no record, no detail used
    assert not (home / "debug").exists()
    assert "compaction" in features.acted_names()  # evals._with_acted still says it acted


def test_reset_clears_the_attempt_and_the_turn(debug_on: Path) -> None:
    log = debuglog.start("unit")
    features.mark_acted("lint")
    features.reset_acted()
    assert features.acted_names() == frozenset()
    features.mark_acted("lint")
    assert len(_acted(log.path)) == 2


def test_every_tracked_switch_is_a_default_on_core_switch_and_marked_in_the_code() -> None:
    import re

    src = Path(features.__file__).parent
    marked = set()
    for path in src.rglob("*.py"):
        marked |= set(re.findall(r"""mark_acted\(\s*["']([a-z0-9_]+)["']""", path.read_text()))
    assert len(features.ACTED_TRACKED) == len(set(features.ACTED_TRACKED))
    for name in features.ACTED_TRACKED:
        assert name in features.FEATURES, name
        assert name in marked, f"{name} is tracked but no call site marks it"


# -- switches mark where they act ---------------------------------------------------------------

def test_a_sample_of_switches_marks_when_it_acts_and_not_when_it_does_not(debug_on: Path, tmp_path: Path) -> None:
    from vega import check_feedback, codemerge, sandbox, tools, write_loop

    log = debuglog.start("sample")
    project = tmp_path / "p"
    project.mkdir()
    (project / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    assert check_feedback.short_tracebacks("pytest -q --tb=long") == "pytest -q --tb=long"
    assert "short_tracebacks" not in features.acted_names()
    assert check_feedback.short_tracebacks("pytest -q").endswith("--tb=short")

    assert sandbox.clip_output("x" * 10) == "x" * 10  # short output: nothing cut
    assert "output_head_tail" not in features.acted_names()
    sandbox.clip_output("line\n" * 5000)

    tools.execute("edit_file", {"path": "a.py", "old_text": "return 1", "new_text": "return 1"}, project)
    tools.execute("search_code", {"query": "def missing_thing"}, project)
    write_loop.fence_for("plain text")
    assert "fence_lines" not in features.acted_names()
    write_loop.fence_for("has ``` inside")
    merged = codemerge.merge_code('def f():\n    """Doc."""\n    return 1\n', "def f():\n    return 2\n")
    assert '"""Doc."""' in merged

    acted = {r["feature"] for r in _acted(log.path)}
    assert {"short_tracebacks", "output_head_tail", "noop_edit_refused", "missing_symbol_hint",
            "fence_lines", "keep_docstrings"} <= acted


def test_a_switch_that_is_off_never_marks(debug_on: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vega import check_feedback

    monkeypatch.setenv(features.ENV, "short_tracebacks")
    log = debuglog.start("off")
    assert check_feedback.short_tracebacks("pytest -q") == "pytest -q"
    assert _acted(log.path) == []


def test_agent_loop_send_backs_mark_their_switch(debug_on: Path, tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    (project / "ops.py").write_text("def diff(a, b):\n    return a + b\n", encoding="utf-8")
    replies = [_message("Could you tell me more about diff()?"),
               _message(None, ("edit_file", {"path": "ops.py", "old_text": "a + b", "new_text": "a - b"})),
               _message("Fixed.")]
    agent, _seen = _scripted(project, replies)
    debuglog.start("loop")
    assert agent.run("Fix the bug in diff().") == "Fixed."
    acted = {r["feature"] for r in _acted(debuglog.current().path)}
    assert "nudge_act" in acted
    assert "code_index" in acted  # 2026-11-03: the index lines and tools change every agent turn


def test_acceptance_marks_when_it_acts(debug_on: Path, tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    from vega import acceptance

    log = debuglog.start("marks")
    agent = SimpleNamespace(project=tmp_path, can_write=True)
    monkeypatch.setattr(acceptance, "extract", lambda project, task: acceptance.Spec(task=task))
    assert acceptance.for_turn(agent, "say hello") is not None
    assert "acceptance" not in features.acted_names()  # an empty spec changes nothing
    monkeypatch.setattr(acceptance, "extract", lambda project, task: acceptance.Spec(task=task, commands=["python a.py"]))
    acceptance.for_turn(agent, "run python a.py")
    acted = {r["feature"] for r in _acted(log.path)}
    assert "acceptance" in acted


# -- window composition -------------------------------------------------------------------------

def test_window_parts_sum_to_the_total_and_match_the_estimate() -> None:
    section = "\n\nProject context:\nVEGA.md says hello." * 3
    messages = [
        {"role": "system", "content": "You are Vega." + section},
        {"role": "user", "content": "an earlier task"},
        {"role": "assistant", "content": "earlier answer"},
        {"role": "user", "content": "Fix the bug in diff()."},
        {"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "read_file", "arguments": "{\"path\": \"ops.py\"}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "def diff(a, b):\n    return a + b\n" * 20},
        {"role": "user", "content": "(Vega harness) Not finished: run the tests."},
        {"role": "user", "content": "File cards: ops.py defines diff"},
    ]
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    window_log.hint(task="Fix the bug in diff().", project_context=section,
                    memory=["File cards: ops.py defines diff"], step=2)
    record = window_log.compose(messages, tools, window_log.take_hint())
    parts = record["parts"]
    assert record["hinted"] and record["step"] == 2
    assert record["total"] == sum(parts.values())
    assert abs(record["total"] - record["estimate"]) <= len(window_log.PARTS)
    assert record["estimate"] == context.estimate_tokens(messages) + parts["tools"]
    assert record["messages"] == {"system": 1, "tools": 1, "project_context": 0, "memory": 1,
                                  "history": 3, "tool_results": 1, "request": 1, "notes": 1}
    assert parts["project_context"] == int(len(section) / context.CHARS_PER_TOKEN)
    assert parts["tool_results"] > parts["request"] > 0
    assert window_log.take_hint() is None  # consumed


def test_unhinted_requests_split_by_role() -> None:
    messages = [{"role": "system", "content": "writer"}, {"role": "user", "content": "Task: x\n\nInputs..."},
                {"role": "assistant", "content": "```python\nx = 1\n```"},
                {"role": "user", "content": "The tests fail with: ..."}]
    record = window_log.compose(messages)
    assert not record["hinted"]
    assert record["messages"]["request"] == 1 and record["messages"]["notes"] == 1
    assert record["messages"]["history"] == 1 and record["parts"]["tools"] == 0


def test_each_model_request_gets_one_window_record_with_counts_only(debug_on: Path, tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    (project / "ops.py").write_text("def diff(a, b):\n    return a + b\n", encoding="utf-8")
    replies = [_message(None, ("read_file", {"path": "ops.py"})), _message("It adds.")]
    agent, seen = _scripted(project, replies, wrap=True)
    log = debuglog.start("window")
    agent.run("What does diff in ops.py return? Do not change anything.")
    rows = _entries(log.path)
    windows = [r for r in rows if r.get("stage") == "window"]
    requests = [r for r in rows if r["kind"] == "request"]
    assert len(windows) == len(requests) == len(seen)
    for row in windows:
        assert row["total"] == sum(row["parts"].values())
        assert all(isinstance(v, int) for v in row["parts"].values())
        assert "text" not in row and "content" not in json.dumps(row)


# -- helpers ----------------------------------------------------------------------------------------

class _Msg(SimpleNamespace):
    def model_dump(self, exclude_none: bool = False) -> dict:
        data = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                "name": c.function.name, "arguments": c.function.arguments}} for c in self.tool_calls]
        return data


def _message(content: str | None, call: tuple[str, dict] | None = None) -> _Msg:
    calls = None
    if call:
        calls = [SimpleNamespace(id=f"c-{call[0]}", function=SimpleNamespace(
            name=call[0], arguments=json.dumps(call[1])))]
    return _Msg(content=content, tool_calls=calls)


def _scripted(project: Path, replies: list[_Msg], wrap: bool = False):
    agent = CodingAgent(project, allow_commands=False)
    seen: list[list[dict]] = []

    def create(**kwargs):
        seen.append(json.loads(json.dumps(kwargs["messages"], default=str)))
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=replies.pop(0), finish_reason="stop")],
                               usage=usage)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    agent.client = debuglog.DebugClient(client) if wrap else client
    agent.stream = False
    return agent, seen
