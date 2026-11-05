"""vega.extensions: the optional extension package (present, absent, broken, disabled) and the
default of every hook point. The package here is a fake module put in sys.modules."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from vega import clarify, extensions, write_loop


@pytest.fixture(autouse=True)
def fresh(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(extensions.DISABLE_ENV, raising=False)
    extensions.reset()
    yield
    extensions.reset()


def absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, extensions.PACKAGE, None)  # import raises ImportError
    extensions.reset()  # a package loaded earlier in the test (importing vega.cli loads it) is forgotten


def present(monkeypatch: pytest.MonkeyPatch, **hooks: Any) -> ModuleType:
    module = ModuleType(extensions.PACKAGE)
    for name, fn in hooks.items():
        setattr(module, name, fn)
    monkeypatch.setitem(sys.modules, extensions.PACKAGE, module)
    extensions.reset()  # as in absent(): the fake package, not one loaded earlier in the test
    return module


def test_absent_package_is_public_core(monkeypatch) -> None:
    absent(monkeypatch)
    assert extensions.load() is None and not extensions.active()
    assert extensions.load_error == ""  # absent is normal, not an error


def test_each_hook_default_without_the_package(monkeypatch) -> None:
    absent(monkeypatch)
    assert extensions.before_start(None, model_given=False) == 0
    assert extensions.before_start("run", model_given=True) == 0
    app = object()
    assert extensions.cli_commands(app) is None
    assert extensions.turn_started(object(), "task") is None
    assert extensions.checks_recorded("direct_pipeline", checks=None) is None
    assert extensions.final_answer("Done.", source="agent", agent=None, task="t") == "Done."
    verdict = clarify.Verdict(clarify.UNSURE, ["short"])
    assert extensions.clarify_verdict("t", verdict, agent=None, follow_up=False) is verdict
    assert extensions.keep_answer("Which one?", reason="the task is not done yet", agent=None,
                                  task="t", wrote=False) is False
    assert extensions.present_questions(object(), [("Which?", ["a", "b"])], text="Which?",
                                        agent=None, send=lambda reply: None) is None
    assert extensions.run_case({"id": "c", "task": "t"}, suite="fixes") is None
    assert extensions.write_findings("write_file", "a.py", None, "import x\n",
                                     project=".", pending=[]) == ("", "")
    assert extensions.writer_context(".", "a.py", "t", source="direct_pipeline", prompt="p",
                                     budget=1000) == ""
    assert extensions.system_context(object(), "what can you do", conversation=True,
                                     route=None) == ""


def test_a_broken_package_is_ignored_and_the_reason_kept(monkeypatch) -> None:
    def boom(name: str) -> ModuleType:
        raise RuntimeError("lab is half written")

    monkeypatch.setattr(extensions.importlib, "import_module", boom)
    assert extensions.load() is None
    assert extensions.load_error == "RuntimeError: lab is half written"
    assert extensions.final_answer("a", source="agent") == "a"


def test_a_missing_dependency_of_the_package_is_reported(monkeypatch) -> None:
    def missing(name: str) -> ModuleType:
        raise ModuleNotFoundError("No module named 'numpyish'", name="numpyish")

    monkeypatch.setattr(extensions.importlib, "import_module", missing)
    assert extensions.load() is None and "numpyish" in extensions.load_error


def test_disabled_by_the_environment(monkeypatch) -> None:
    present(monkeypatch, final_answer=lambda answer, **_: answer + "!")
    monkeypatch.setenv(extensions.DISABLE_ENV, "1")
    assert extensions.load() is None
    assert extensions.final_answer("a", source="agent") == "a"


def test_present_package_hooks_are_called_with_their_context(monkeypatch) -> None:
    calls: list[tuple] = []
    present(monkeypatch,
            before_start=lambda command, *, model_given: calls.append(
                ("start", command, model_given)) or 3,
            cli_commands=lambda app: calls.append(("cli", app)),
            turn_started=lambda agent, task: calls.append(("turn", task)),
            checks_recorded=lambda source, **ctx: calls.append(("checks", source, sorted(ctx))),
            final_answer=lambda answer, *, source, **ctx: f"{answer} [{source}]")
    assert extensions.active()
    assert extensions.before_start("run", model_given=True) == 3
    extensions.cli_commands("APP")
    extensions.turn_started(None, "fix it")
    extensions.checks_recorded("acceptance", agent=None, result=None, changed=[])
    assert extensions.final_answer("Done.", source="agent") == "Done. [agent]"
    assert calls == [("start", "run", True), ("cli", "APP"), ("turn", "fix it"),
                     ("checks", "acceptance", ["agent", "changed", "result"])]


def test_a_package_without_some_hooks_gets_the_defaults(monkeypatch) -> None:
    present(monkeypatch)  # defines nothing
    assert extensions.active()
    assert extensions.before_start(None, model_given=False) == 0
    assert extensions.final_answer("x", source="agent") == "x"


def test_a_failing_hook_never_fails_the_caller(monkeypatch, capsys) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("bad hook")

    present(monkeypatch, before_start=fail, cli_commands=fail, turn_started=fail,
            checks_recorded=fail, final_answer=fail, clarify_verdict=fail, keep_answer=fail,
            present_questions=fail)
    assert extensions.present_questions(object(), [("q", ["a"])], text="q") is None
    verdict = clarify.Verdict(clarify.CLEAR)
    assert extensions.clarify_verdict("t", verdict) is verdict
    assert extensions.keep_answer("a", reason="r") is False
    assert extensions.before_start(None, model_given=False) == 0
    extensions.cli_commands(object())
    extensions.turn_started(None, "t")
    extensions.checks_recorded("x")
    assert extensions.final_answer("kept", source="agent") == "kept"
    err = capsys.readouterr().err
    assert "extension hook before_start failed (ValueError: bad hook)" in err
    assert "extension hook cli_commands failed" in err
    assert "final_answer" not in err  # per-turn hooks only go to the debug log


def test_attempt_wait_default_results_and_bad_results(monkeypatch) -> None:
    absent(monkeypatch)
    assert extensions.attempt_wait(sleep=lambda s: None, first=True, log=print) == []
    seen: list[dict] = []
    present(monkeypatch, attempt_wait=lambda **ctx: seen.append(ctx) or {"kind": "heat", "seconds": 12})
    assert extensions.attempt_wait(sleep="SLEEP", first=False, log="LOG") == [{"kind": "heat", "seconds": 12.0}]
    assert seen == [{"sleep": "SLEEP", "first": False, "log": "LOG"}]
    present(monkeypatch, attempt_wait=lambda **ctx: [{"kind": "heat", "seconds": 1}, {"kind": "bad key", "seconds": 1},
                                                    {"kind": "x", "seconds": -1}, {"seconds": 3}, "junk"])
    assert extensions.attempt_wait(sleep=None, first=True, log=None) == [{"kind": "heat", "seconds": 1.0}]

    def fail(**ctx: Any) -> Any:
        raise RuntimeError("sensor tool crashed")

    present(monkeypatch, attempt_wait=fail)
    assert extensions.attempt_wait(sleep=None, first=True, log=None) == []  # never stops the run


def test_run_case_hook_result_and_bad_results(monkeypatch) -> None:
    seen: list[tuple] = []

    def run_case(case: dict, **context: Any) -> dict | None:
        seen.append((case["id"], sorted(context)))
        return {"passed": True} if case["id"] == "mine" else None

    present(monkeypatch, run_case=run_case)
    assert extensions.run_case({"id": "mine"}, suite="s", transcript=None) == {"passed": True}
    assert extensions.run_case({"id": "other"}, suite="s") is None
    assert seen == [("mine", ["suite", "transcript"]), ("other", ["suite"])]
    present(monkeypatch, run_case=lambda case, **_: "not a record")
    extensions.reset()
    assert extensions.run_case({"id": "x"}) is None


def test_writer_context_is_passed_its_context_and_adds_only_text(monkeypatch) -> None:
    calls: list[tuple] = []

    def context(project, target, task, *, source, **ctx):
        calls.append((project, target, task, source, sorted(ctx)))
        return "\n\nInterfaces: ..."

    present(monkeypatch, writer_context=context)
    assert extensions.writer_context("/p", "a.py", "t", source="project_pipeline", prompt="u",
                                     plan=[], budget=9) == "\n\nInterfaces: ..."
    assert calls == [("/p", "a.py", "t", "project_pipeline", ["budget", "plan", "prompt"])]
    present(monkeypatch, writer_context=lambda *a, **k: None)
    extensions.reset()
    assert extensions.writer_context("/p", "a.py", "t", source="direct_pipeline") == ""

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("bad hook")

    present(monkeypatch, writer_context=fail)
    extensions.reset()
    assert extensions.writer_context("/p", "a.py", "t", source="direct_pipeline") == ""


def test_a_non_string_final_answer_keeps_the_answer(monkeypatch) -> None:
    present(monkeypatch, final_answer=lambda answer, **_: None)
    assert extensions.final_answer("kept", source="agent") == "kept"


def test_a_non_verdict_clarify_verdict_and_a_truthy_keep_answer_are_ignored(monkeypatch) -> None:
    present(monkeypatch, clarify_verdict=lambda task, verdict, **_: "vague",
            keep_answer=lambda answer, **_: "yes")
    verdict = clarify.Verdict(clarify.CLEAR)
    assert extensions.clarify_verdict("t", verdict) is verdict
    assert extensions.keep_answer("a", reason="r") is False  # only True keeps


def test_the_clarify_gate_uses_the_hooks_verdict_and_its_questions(monkeypatch, tmp_path) -> None:
    from vega import features

    monkeypatch.setenv(features.ENV, "question_pipeline,project_pipeline")
    seen: dict[str, Any] = {}

    def clarify_verdict(task: str, verdict: Any, **ctx: Any) -> Any:
        seen.update(task=task, kind=verdict.kind, ctx=sorted(ctx))
        return clarify.Verdict(clarify.VAGUE, ["lab"], questions=[("Which colour?", ["red"])])

    present(monkeypatch, clarify_verdict=clarify_verdict)
    agent = _scripted(tmp_path, "never called")
    agent.client = None  # no model call: the hook's questions are asked as they are
    reply = agent.run("Hide the debug panel.")
    assert "1. Which colour?" in reply and "a) red" in reply
    assert seen == {"task": "Hide the debug panel.", "kind": "clear",
                    "ctx": ["agent", "follow_up"]}


def test_keep_answer_lets_the_agent_loop_end_on_the_answer(monkeypatch, tmp_path) -> None:
    from test_agent_loop import agent_with, call, text

    seen: list[tuple[str, list[str]]] = []

    def keep_answer(answer: str, *, reason: str, **ctx: Any) -> bool:
        seen.append((reason, sorted(ctx)))
        return True

    present(monkeypatch, keep_answer=keep_answer)
    (tmp_path / "a.py").write_text("X = 1\n")
    asked = "Could you provide the details of the new value: 2 or 3?"
    agent, calls = agent_with(tmp_path, [call("read_file", path="a.py"), text(asked)])
    assert agent.run("Set X in a.py to the new value.") == asked
    assert seen == [("asked the user for project content", ["agent", "task", "wrote"])]
    assert len(calls) == 2


def test_keep_answer_also_holds_against_finish_problems(monkeypatch, tmp_path) -> None:
    """A question kept by keep_answer is not sent back by an extension's finish_problems either
    (clarify's intent question vs the deliverables ledger, same reason as the finish gate)."""
    from test_agent_loop import agent_with, call, text

    seen: list[str] = []
    present(monkeypatch, keep_answer=lambda answer, *, reason, **ctx: seen.append(reason) or True,
            finish_problems=lambda agent, task, **ctx: ["`b.py` does not exist yet"])
    (tmp_path / "a.py").write_text("X = 1\n")
    asked = "Which value should X take: 2 or 3?"
    agent, calls = agent_with(tmp_path, [call("read_file", path="a.py"), text(asked)])
    assert agent.run("Set X in a.py to the new value and add b.py.") == asked
    assert seen == ["the task is not done yet"] and len(calls) == 2
    extensions.reset()
    present(monkeypatch, keep_answer=lambda answer, **ctx: False,
            finish_problems=lambda agent, task, **ctx: ["b.py is missing"] if ctx["round"] == 1
            else [])
    agent, calls = agent_with(tmp_path, [call("read_file", path="a.py"), text(asked),
                                         text("Done.")] + [text("Done.")] * 4)
    agent.run("Set X in a.py to the new value and add b.py.")
    assert len(calls) >= 3 and "- b.py is missing" in calls[2][-1]["content"]  # sent back


def _loop(tmp_path) -> tuple[write_loop.LoopResult, write_loop.LoopConfig]:
    checks = write_loop.Checks({"tests": 2})
    attempt = write_loop.Attempt(2, 2, [], {"tests": 2}, 0, "reply", checks)
    result = write_loop.LoopResult([attempt], 0, {"a.py": "new\n"}, [], {})
    config = write_loop.LoopConfig(name="demo_pipeline", title="Demo", targets=["a.py"],
                                   originals={"a.py": "old\n"}, repairs=1,
                                   check=lambda saved: checks)
    return result, config


def test_pipeline_answers_pass_through_final_answer(monkeypatch, tmp_path) -> None:
    result, config = _loop(tmp_path)
    absent(monkeypatch)
    plain = write_loop.answer_for(result, config)
    extensions.reset()
    seen: dict[str, Any] = {}

    def hook(answer: str, *, source: str, **ctx: Any) -> str:
        seen.update(source=source, changed=ctx["changed"], config=ctx["config"])
        return answer + "\nChecked."

    present(monkeypatch, final_answer=hook)
    assert write_loop.answer_for(result, config) == plain + "\nChecked."
    assert seen == {"source": "demo_pipeline", "changed": ["a.py"], "config": config}


def test_render_markdown_default_and_hook(monkeypatch) -> None:
    absent(monkeypatch)
    assert extensions.render_markdown("a\nb") == "a\nb"
    extensions.reset()
    present(monkeypatch, render_markdown=lambda text: text.replace("\n", "  \n"))
    assert extensions.render_markdown("a\nb") == "a  \nb"
    extensions.reset()
    present(monkeypatch, render_markdown=lambda text: None)  # not a string: the text stays
    assert extensions.render_markdown("a\nb") == "a\nb"


def _scripted(tmp_path, content: str) -> Any:
    from types import SimpleNamespace

    from vega.agent import CodingAgent

    agent = CodingAgent(tmp_path, allow_commands=False)
    reply = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=reply, finish_reason="stop")
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: SimpleNamespace(choices=[choice]))))
    return agent


def test_a_pipeline_passes_its_check_records_and_answer_context(monkeypatch, tmp_path) -> None:
    """checks_recorded gets write_loop.Checks with `items` (plain dicts naming the checker);
    final_answer gets `left` and `what` from write_loop.answer_for."""
    from vega import features

    monkeypatch.setenv(features.ENV_ON, "deliverable_pipeline")
    seen: dict[str, Any] = {}

    def checks_recorded(source: str, **ctx: Any) -> None:
        seen["items"] = [dict(i) for i in ctx["checks"].items]

    def final_answer(answer: str, *, source: str, **ctx: Any) -> str:
        seen.update(source=source, left=ctx["left"], what=ctx["what"])
        return answer

    present(monkeypatch, checks_recorded=checks_recorded, final_answer=final_answer)
    (tmp_path / "notes.txt").write_text("PR 12 adds dark mode (user-visible)\n")
    agent = _scripted(tmp_path, "File: RELEASE_NOTES.md\n```markdown\n- Dark mode\n```")
    answer = agent.run("Write `RELEASE_NOTES.md` with one bullet per user-visible change.")
    assert answer == "Wrote RELEASE_NOTES.md; 1 of 1 checks pass (1 format check)."
    assert seen["items"] == [{"source": "format", "label": "", "ok": True, "detail": ""}]
    assert seen == {**seen, "source": "deliverable_pipeline", "left": {}, "what": "content"}


def test_the_agent_loop_passes_its_turn_context(monkeypatch, tmp_path) -> None:
    seen: dict[str, Any] = {}

    def final_answer(answer: str, *, source: str, **ctx: Any) -> str:
        seen["final"] = sorted(ctx)
        return answer + "\nAdded."

    present(monkeypatch, final_answer=final_answer,
            turn_started=lambda agent, task, **ctx: seen.update(turn=sorted(ctx)))
    from test_agent_loop import agent_with, text

    (tmp_path / "a.py").write_text("X = 1\n")
    agent, _ = agent_with(tmp_path, [text("Done."), text("Done.")])
    events: list[tuple[str, dict]] = []
    agent.on_event = lambda kind, data: events.append((kind, data))
    assert agent.run("Set X to 2 in a.py.") == "Done.\nAdded."
    assert seen["turn"] == ["spec", "stop"]
    assert seen["final"] == ["agent", "baseline", "spec", "stop", "task", "written"]
    assert [d for k, d in events if k == "final"][-1]["addendum"] == "Added."


def test_write_findings_refuses_or_notes_through_the_write_check(monkeypatch, tmp_path) -> None:
    """edits.check gives a write that passed it to write_findings: a refusal refuses it (with the
    retry hint), a note is added to the allowed write, a bad result is ignored."""
    from vega import edits

    seen: list[tuple] = []

    def findings(tool, relative, before, after, *, project, **context):
        seen.append((tool, relative, before, after, project, context))
        return ("a.py imports nothing that exists.", "") if "bad" in after else ("", " (noted)")

    present(monkeypatch, write_findings=findings)
    refused = edits.check("pipeline", "a.py", None, "bad = 1\n", tmp_path, pending=["b.py"])
    assert refused.refusal is not None
    assert refused.refusal.startswith("Write not applied: a.py imports nothing that exists.\n")
    assert "Reply again with the corrected code for a.py" in refused.refusal
    assert seen[0] == ("pipeline", "a.py", None, "bad = 1\n", tmp_path, {"pending": ["b.py"]})
    allowed = edits.check("write_file", "a.py", "x = 1\n", "x = 2\n", tmp_path)
    assert allowed.refusal is None and allowed.note == " (noted)"
    assert edits.check("write_file", "a.py", None, "x = (\n", tmp_path).refusal  # parse first
    assert len(seen) == 2  # not called for a write the core refuses itself
    present(monkeypatch, write_findings=lambda *a, **k: "not a pair")
    extensions.reset()
    assert extensions.write_findings("pipeline", "a.py", None, "", project=tmp_path) == ("", "")
    assert edits.check("pipeline", "a.py", None, "x = 1\n", tmp_path).refusal is None


def test_without_the_package_the_write_check_is_unchanged(monkeypatch, tmp_path) -> None:
    from vega import edits

    absent(monkeypatch)
    verdict = edits.check("pipeline", "a.py", None, "from nowhere import thing\n", tmp_path)
    assert verdict.refusal is None and verdict.note == ""


def test_system_context_default_hook_and_non_string(monkeypatch) -> None:
    present(monkeypatch)  # a package without the hook
    assert extensions.system_context(object(), "t") == ""
    extensions.reset()
    present(monkeypatch, system_context=lambda agent, task, **ctx: f"[{task}|{sorted(ctx)}]")
    assert extensions.system_context(object(), "t", conversation=True) == "[t|['conversation']]"
    extensions.reset()
    present(monkeypatch, system_context=lambda agent, task, **ctx: None)
    assert extensions.system_context(object(), "t") == ""


def test_the_agent_loop_adds_system_context_to_each_request_not_the_history(
        monkeypatch, tmp_path) -> None:
    seen: dict[str, Any] = {}

    def system_context(agent: Any, task: str, **ctx: Any) -> str:
        seen.update(ctx)
        return "(note) facts for this turn" if "you" in task else ""

    present(monkeypatch, system_context=system_context)
    from test_agent_loop import agent_with, text

    agent, requests = agent_with(tmp_path, [text("I write and fix Python code."), text("Hi.")])
    assert agent.run("what can you do") == "I write and fix Python code."
    assert seen["conversation"] is True and seen["route"]["name"] == "agent"
    notes = [m for m in requests[0] if m["role"] == "user" and m["content"].startswith("(note)")]
    assert len(notes) == 1 and requests[0].index(notes[0]) == len(requests[0]) - 2  # before BRIEF
    assert all("(note)" not in str(m.get("content")) for m in agent.messages)
    agent.run("hello")  # the hook returns "": nothing added
    assert all("(note)" not in str(m.get("content")) for m in requests[1])


def test_without_the_package_requests_are_unchanged(monkeypatch, tmp_path) -> None:
    absent(monkeypatch)
    from test_agent_loop import agent_with, text

    agent, requests = agent_with(tmp_path, [text("I write and fix Python code.")])
    agent.run("what can you do")
    assert requests[0][-1]["content"].startswith("(Vega harness) This is a conversational")
    assert requests[0][-2]["content"] == "what can you do"


def test_agent_event_and_answer_context_defaults(monkeypatch) -> None:
    absent(monkeypatch)
    assert extensions.agent_event(object(), "final", {"text": "x"}) is None
    assert extensions.answer_context(object(), "did you test it?", route=None) == ""
    extensions.reset()
    present(monkeypatch, answer_context=lambda agent, task, **ctx: None)  # not a string: nothing
    assert extensions.answer_context(object(), "t", route=None) == ""


def test_the_agent_loop_passes_events_and_takes_answer_context(monkeypatch, tmp_path) -> None:
    """agent_event sees every non-text event (a tool result, the turn's final answer);
    answer_context's text is a trailing message on each request of the turn, not history."""
    events: list[tuple[str, Any]] = []
    asked: list[tuple[str, Any]] = []

    def answer_context(agent: Any, task: str, **ctx: Any) -> str:
        asked.append((task, ctx["route"]["name"]))
        return "Evidence: nothing ran."

    present(monkeypatch, agent_event=lambda agent, kind, data: events.append((kind, dict(data))),
            answer_context=answer_context)
    from test_agent_loop import agent_with, text

    agent, seen = agent_with(tmp_path, [text("Not tested.")])
    assert agent.run("what is a closure in python?") == "Not tested."
    assert asked == [("what is a closure in python?", "agent")]
    assert {"role": "user", "content": "Evidence: nothing ran."} in seen[0]
    assert all("Evidence" not in str(m.get("content")) for m in agent.messages)
    assert events[-1] == ("final", {"text": "Not tested.", "streamed": False})
    assert all(kind not in ("text", "text_end") for kind, _ in events)


def test_the_question_pipeline_takes_answer_context_and_final_answer(monkeypatch, tmp_path) -> None:
    seen: dict[str, Any] = {}

    def final_answer(answer: str, *, source: str, **ctx: Any) -> str:
        seen.update(source=source, ctx=sorted(ctx))
        return answer + " (checked)"

    present(monkeypatch, answer_context=lambda agent, task, **ctx: f"Extra for {ctx['route']}.",
            final_answer=final_answer)
    (tmp_path / "config.py").write_text("PORT = 8080\n")
    agent = _scripted(tmp_path, "PORT is 8080 (config.py).")
    sent: list[list[dict]] = []
    create = agent.client.chat.completions.create
    agent.client.chat.completions.create = lambda **kw: sent.append(kw["messages"]) or create(**kw)
    assert agent.run("What is the value of PORT in config.py?") == \
        "PORT is 8080 (config.py). (checked)"
    assert agent.last_route["name"] == "question"
    assert sent[0][-1]["content"].endswith("\n\nExtra for question.")
    assert seen == {"source": "question_pipeline", "ctx": ["agent", "task"]}
    assert agent.messages[-1]["content"] == "PORT is 8080 (config.py). (checked)"


def test_budget_default_and_hook(monkeypatch) -> None:
    class Agent:
        reads = 0

        @property
        def num_ctx(self) -> int:
            Agent.reads += 1
            return 32768

    absent(monkeypatch)
    assert extensions.budget("project_pipeline.FILE_CHARS", 9000, num_ctx=32768) == 9000
    assert extensions.budget("pipeline.MAX_FILE_CHARS", 12000, agent=Agent()) == 12000
    assert Agent.reads == 0  # without a hook the window is never resolved for this

    extensions.reset()
    seen: list[tuple[str, int, Any]] = []

    def budget(name: str, default: int, *, num_ctx: Any = None, **context: Any) -> Any:
        seen.append((name, default, num_ctx))
        return {"a": default * 2, "b": "big", "c": 0, "d": True}.get(name, default)

    present(monkeypatch, budget=budget)
    assert extensions.budget("a", 100, agent=Agent()) == 200
    assert Agent.reads == 1 and seen[-1] == ("a", 100, 32768)
    assert extensions.budget("a", 100, num_ctx=16384) == 200 and seen[-1][2] == 16384
    for name in ("b", "c", "d"):  # not a positive whole number: the default
        assert extensions.budget(name, 100, num_ctx=8192) == 100


def test_a_failing_budget_hook_keeps_the_default(monkeypatch) -> None:
    def budget(name: str, default: int, **context: Any) -> int:
        raise ValueError("no")

    present(monkeypatch, budget=budget)
    assert extensions.budget("x", 7, num_ctx=65536) == 7


def test_repair_write_hook_through_the_write_check(monkeypatch, tmp_path) -> None:
    """edits.check with repair=True gives a write that adds a parse error to repair_write: a
    repaired text that parses is checked like any write and returned as Verdict.after with the
    note; a refusal replaces the text; a repair that does not parse, or a caller without
    repair=True, gets the plain refusal; the agent's write tools write the repaired text."""
    from vega import edits, tools

    seen: list[tuple] = []

    def repair_write(tool, relative, before, after, *, project, problem, **context):
        seen.append((tool, relative, problem.line, sorted(context)))
        if "fixable" in after:
            return {"after": after.replace("x = (", "x = ()"), "note": " (repaired)"}
        if "broken-repair" in after:
            return {"after": "y = (\n"}
        return {"refusal": "Write not applied: custom."}

    present(monkeypatch, repair_write=repair_write)
    fixed = edits.check("write_file", "a.py", None, "# fixable\nx = (\n", tmp_path, repair=True)
    assert fixed.refusal is None and fixed.after == "# fixable\nx = ()\n"
    assert fixed.note == " (repaired)"
    assert seen[0] == ("write_file", "a.py", 2, ["args", "refusal", "region"])
    custom = edits.check("write_file", "a.py", None, "x = (\n", tmp_path, repair=True)
    assert custom.refusal == "Write not applied: custom." and custom.after is None
    plain = edits.check("write_file", "a.py", None, "# broken-repair\nx = (\n", tmp_path,
                        repair=True)
    assert plain.refusal is not None and "would not parse" in plain.refusal
    calls = len(seen)
    assert edits.check("pipeline", "a.py", None, "# fixable\nx = (\n", tmp_path).after is None
    assert len(seen) == calls  # without repair=True the hook is not asked
    result = tools.execute("write_file", {"path": "a.py", "content": "# fixable\nx = (\n"},
                           tmp_path)
    assert result.startswith("Wrote a.py") and "(repaired)" in result
    assert (tmp_path / "a.py").read_text() == "# fixable\nx = ()\n"


def test_resolve_undefined_hook_through_the_write_check(monkeypatch, tmp_path) -> None:
    """edits.check with repair=True gives a write whose only new findings are undefined names to
    resolve_undefined: a text that then passes the check is written with the note; one that
    still fails, an empty answer, or a caller without repair=True get the plain refusal."""
    from vega import edits

    seen: list[tuple] = []

    def resolve_undefined(tool, relative, before, after, *, project, names, offers, **context):
        seen.append((tool, relative, names, sorted(context)))
        if "fixable" in after:
            return {"after": "import json\n" + after, "note": " (imported)"}
        if "still" in after:
            return {"after": "x = undefined_too\n" + after}
        return {}

    present(monkeypatch, resolve_undefined=resolve_undefined)
    fixed = edits.check("write_file", "a.py", None, "# fixable\nx = json.dumps(1)\n", tmp_path,
                        repair=True)
    assert fixed.refusal is None and fixed.after == "import json\n# fixable\nx = json.dumps(1)\n"
    assert fixed.note.startswith(" (imported)")
    assert seen[0] == ("write_file", "a.py", ["json"], ["args"])
    still = edits.check("write_file", "a.py", None, "# still\nx = json.dumps(1)\n", tmp_path,
                        repair=True)
    assert still.refusal is not None and "undefined name" in still.refusal and still.after is None
    plain = edits.check("write_file", "a.py", None, "x = json.dumps(1)\n", tmp_path, repair=True)
    assert plain.refusal is not None and plain.after is None
    calls = len(seen)
    assert edits.check("pipeline", "a.py", None, "# fixable\nx = json.dumps(1)\n",
                       tmp_path).after is None
    assert len(seen) == calls  # without repair=True the hook is not asked


def test_write_base_hook_lets_replace_definition_edit_a_draft(monkeypatch, tmp_path) -> None:
    from vega import tools

    draft = "def a():\n    return 1\n\ndef b(:\n    return 2\n"
    present(monkeypatch, write_base=lambda tool, relative, *, project, **ctx:
            draft if relative == "m.py" and ctx.get("disk") is None else None)
    result = tools.execute("replace_definition",
                           {"path": "m.py", "name": "b", "code": "def b():\n    return 2\n"},
                           tmp_path)
    assert result.startswith("Replaced definition b"), result
    assert "draft" in result
    assert (tmp_path / "m.py").read_text() == "def a():\n    return 1\n\ndef b():\n    return 2\n"


def test_tool_refused_hook_sees_only_refused_calls_and_may_apply_or_reword(monkeypatch,
                                                                           tmp_path) -> None:
    from vega import tools

    (tmp_path / "a.py").write_text("x = 1\n")
    seen: list[tuple[str, str]] = []

    def hook(tool, args, refusal, *, project, **ctx):
        seen.append((tool, refusal.splitlines()[0]))
        if tool == "edit_file" and "not found" in refusal:
            target = project / args["path"]
            return {"plan": tools.WritePlan("edit_file", target, args["path"], "x = 1\n",
                                            "x = 2\n", "Edited a.py (by the hook).")}
        if refusal.startswith(("Tool error", "Unknown tool")):
            return {"refusal": f"reworded: {refusal}"}
        return {}

    present(monkeypatch, tool_refused=hook)
    assert tools.execute("edit_file", {"path": "a.py", "old_text": "x = 1", "new_text": "x = 3"},
                         tmp_path).startswith("Edited a.py (exact match)")
    assert seen == []  # an edit that applies never reaches the hook
    result = tools.execute("edit_file", {"path": "a.py", "old_text": "y = 1", "new_text": "y"},
                           tmp_path)
    assert result.startswith("Edited a.py (by the hook)"), result
    assert (tmp_path / "a.py").read_text() == "x = 2\n"
    assert tools.execute("edit_file", {"path": "a.py", "new_text": "y"}, tmp_path) == \
        "reworded: Tool error: KeyError: 'old_text'"
    assert tools.execute("nope", {}, tmp_path) == "reworded: Unknown tool: nope"
    # the computed write is still checked like any other
    seen.clear()
    monkeypatch.setattr(tools, "_plan_write", lambda *a: "Edit not applied: old_text was not found")
    present(monkeypatch, tool_refused=lambda tool, args, refusal, *, project, **ctx: {
        "plan": tools.WritePlan("edit_file", project / "a.py", "a.py", "x = 2\n", "x = (\n", "Edited")})
    extensions.reset()
    assert "parse" in tools.execute("edit_file", {"path": "a.py", "old_text": "q", "new_text": "r"},
                                    tmp_path)
    assert (tmp_path / "a.py").read_text() == "x = 2\n"


def test_tool_refused_default_keeps_every_refusal(monkeypatch, tmp_path) -> None:
    from vega import tools

    absent(monkeypatch)
    assert extensions.tool_refused("edit_file", {}, "x", project=tmp_path) == {}
    (tmp_path / "a.py").write_text("x = 1\n")
    assert tools.execute("edit_file", {"path": "a.py", "new_text": "y"}, tmp_path) == \
        "Tool error: KeyError: 'old_text'"
    assert tools.execute("nope", {}, tmp_path) == "Unknown tool: nope"
    assert tools.execute("edit_file", {"path": "a.py", "old_text": "zzz", "new_text": "y"},
                         tmp_path).startswith("Edit not applied: old_text was not found")


def test_repair_hooks_default_without_the_package(monkeypatch, tmp_path) -> None:
    from vega import edits

    absent(monkeypatch)
    assert extensions.repair_write("write_file", "a.py", None, "x = (", project=tmp_path,
                                   problem=None) == {}
    assert extensions.write_base("replace_definition", "a.py", project=tmp_path) is None
    verdict = edits.check("write_file", "a.py", None, "x = (\n", tmp_path, repair=True)
    assert verdict.refusal is not None and verdict.after is None
# -- present_questions: the answer's questions reach the UIs ---------------------------------------

def _asking_agent(monkeypatch, tmp_path) -> Any:
    from vega import features

    monkeypatch.setenv(features.ENV, "question_pipeline,project_pipeline")
    agent = _scripted(tmp_path, "never called")
    agent.client = None
    return agent


def test_a_turn_that_asks_questions_carries_them_in_the_final_event(monkeypatch, tmp_path) -> None:
    questions = [("Which colour?", ["red", "blue"])]
    present(monkeypatch, clarify_verdict=lambda task, verdict, **_: clarify.Verdict(
        clarify.VAGUE, ["lab"], questions=questions))
    agent = _asking_agent(monkeypatch, tmp_path)
    events: list[tuple[str, dict]] = []
    agent.on_event = lambda kind, data: events.append((kind, data))
    agent.run("Hide the debug panel.")
    final = [data for kind, data in events if kind == "final"]
    assert final[0]["questions"] == questions and agent.questions_asked == questions


def test_a_turn_without_questions_has_none(monkeypatch, tmp_path) -> None:
    from test_agent_loop import agent_with, text

    absent(monkeypatch)
    agent, _ = agent_with(tmp_path, [text("Hello.")])
    events: list[tuple[str, dict]] = []
    agent.on_event = lambda kind, data: events.append((kind, data))
    agent.run("Say hello.")
    final = [data for kind, data in events if kind == "final"]
    assert "questions" not in final[0] and agent.questions_asked is None


def test_line_chat_without_the_hook_shows_the_held_answer_as_before(monkeypatch, tmp_path) -> None:
    from io import StringIO

    from rich.console import Console

    from vega import cli
    from vega.ui import TerminalRenderer

    absent(monkeypatch)
    extensions.reset()  # importing cli loaded the lab (cli_commands)
    agent = _asking_agent(monkeypatch, tmp_path)
    plain, held = StringIO(), StringIO()
    TerminalRenderer(Console(file=plain, width=100))(
        "final", {"text": "1. Which?\n   a) red", "streamed": False,
                  "questions": [("Which?", ["red"])]})
    renderer = TerminalRenderer(Console(file=held, width=100), hold_questions=True)
    renderer("final", {"text": "1. Which?\n   a) red", "streamed": False,
                       "questions": [("Which?", ["red"])]})
    assert held.getvalue() == ""  # held back for the chat loop
    agent.on_event = renderer
    monkeypatch.setattr(cli, "console", renderer.console)
    assert cli._questions_reply(agent) == ""
    assert held.getvalue() == plain.getvalue() != ""
    assert cli._questions_reply(agent) == ""  # shown once


def test_line_chat_sends_the_hooks_reply(monkeypatch, tmp_path) -> None:
    from io import StringIO

    from rich.console import Console

    from vega import cli
    from vega.ui import TerminalRenderer

    seen: dict[str, Any] = {}

    def present_questions(ui: Any, questions: list, **ctx: Any) -> str:
        seen.update(questions=questions, ctx=sorted(ctx))
        return "1. Which? red"

    present(monkeypatch, present_questions=present_questions)
    agent = _asking_agent(monkeypatch, tmp_path)
    out = StringIO()
    renderer = TerminalRenderer(Console(file=out, width=100), hold_questions=True)
    renderer("final", {"text": "1. Which?", "streamed": False, "questions": [("Which?", ["red"])]})
    agent.on_event = renderer
    assert cli._questions_reply(agent) == "1. Which? red"
    assert seen == {"questions": [("Which?", ["red"])], "ctx": ["agent", "send", "text"]}
    assert out.getvalue() == ""  # the hook showed the questions its own way


def test_compact_history_default_hook_and_unusable_results(monkeypatch) -> None:
    from vega.context import Compaction, WorkLog

    system, goal = {"role": "system", "content": "s"}, {"role": "user", "content": "do it"}
    messages = [system, goal, {"role": "assistant", "content": "x"}]
    absent(monkeypatch)
    assert extensions.compact_history(messages, 1, 8192, WorkLog()) is None

    extensions.reset()
    results: list[Any] = [
        Compaction([system, goal], 1, 10, 5),  # usable: keeps the goal object
        Compaction([system, dict(goal)], 1, 10, 5),  # a copy of the goal: the agent cannot find it
        Compaction([], 0, 0, 0),
        "not a compaction",
    ]
    seen: list[dict[str, Any]] = []

    def compact_history(messages, goal_index, num_ctx, log, **context):
        seen.append(context)
        return results.pop(0)

    present(monkeypatch, compact_history=compact_history)
    first = extensions.compact_history(messages, 1, 8192, WorkLog(), scale=1.2, agent=None)
    assert first is not None and first.messages[1] is goal
    assert seen[0] == {"scale": 1.2, "agent": None}
    for _ in range(3):
        assert extensions.compact_history(messages, 1, 8192, WorkLog()) is None


def test_tool_shown_and_compaction_pins_defaults_and_bad_results(monkeypatch) -> None:
    """tool_shown: the core's text (None) without the package, for "" / non-str results and when
    the hook raises; compaction_pins: "" likewise."""
    absent(monkeypatch)
    assert extensions.tool_shown(None, "read_file", {"path": "a.py"}, "1\tx") is None
    assert extensions.compaction_pins(None, goal={}) == ""

    seen: list[tuple] = []
    results: list[Any] = ["shorter", "", 42]

    def tool_shown(agent, name, args, result, **context):
        seen.append((name, args, result, context))
        return results.pop(0)

    def compaction_pins(agent, **context):
        raise RuntimeError("broken")

    present(monkeypatch, tool_shown=tool_shown, compaction_pins=compaction_pins)
    assert extensions.tool_shown(None, "read_file", {"path": "a.py"}, "long", calls=3) == "shorter"
    assert seen[0] == ("read_file", {"path": "a.py"}, "long", {"calls": 3})
    assert extensions.tool_shown(None, "read_file", {}, "x") is None
    assert extensions.tool_shown(None, "read_file", {}, "x") is None
    assert extensions.compaction_pins(None, goal={}) == ""  # a failing hook: nothing pinned


def test_project_pipeline_trim_hooks_defaults_and_bad_results(monkeypatch) -> None:
    """unchanged_offer / unchanged_reply / planner_request: no offer, never a kept file and the
    planner's own request without the package, or when a hook returns something unusable."""
    schema, system = {"type": "object"}, "You plan code changes."
    absent(monkeypatch)
    assert extensions.unchanged_offer(".", "a.py", "t", source="project_pipeline",
                                      current="x", original="y", failure="f") == ""
    assert extensions.unchanged_reply("UNCHANGED: a.py", "a.py", source="project_pipeline") is False
    assert extensions.planner_request(schema, system, source="project_pipeline") == (schema, system)
    present(monkeypatch, unchanged_offer=lambda *a, **k: 3, unchanged_reply=lambda *a, **k: "yes",
            planner_request=lambda *a, **k: ({"x": 1}, ""))
    extensions.reset()
    assert extensions.unchanged_offer(".", "a.py", "t", source="project_pipeline") == ""
    assert extensions.unchanged_reply("UNCHANGED", "a.py", source="project_pipeline") is False
    assert extensions.planner_request(schema, system, source="project_pipeline") == (schema, system)
    brief = ({"type": "object", "maxLength": 1}, system + " Short.")
    present(monkeypatch, unchanged_offer=lambda *a, **k: "offer",
            unchanged_reply=lambda *a, **k: True, planner_request=lambda *a, **k: brief)
    extensions.reset()
    assert extensions.unchanged_offer(".", "a.py", "t", source="project_pipeline") == "offer"
    assert extensions.unchanged_reply("UNCHANGED", "a.py", source="project_pipeline") is True
    assert extensions.planner_request(schema, system, source="project_pipeline") == brief


def test_memory_context_default_hook_and_non_string(monkeypatch) -> None:
    absent(monkeypatch)
    assert extensions.memory_context(object(), "t", step=0) == ""
    extensions.reset()
    present(monkeypatch, memory_context=lambda agent, task, **ctx: None)
    assert extensions.memory_context(object(), "t", step=3) == ""


def test_the_agent_loop_asks_memory_context_before_each_request(monkeypatch, tmp_path) -> None:
    """memory_context is asked with the loop step before every request; its text is a trailing
    message of that request only (it may change during the turn), never in the history."""
    steps: list[int] = []

    def memory_context(agent: Any, task: str, *, step: int, **ctx: Any) -> str:
        steps.append(step)
        return "(memory) the map" if step >= 1 else ""

    present(monkeypatch, memory_context=memory_context)
    from test_agent_loop import agent_with, call, text

    (tmp_path / "a.py").write_text("x = 1\n")
    agent, requests = agent_with(tmp_path, [call("read_file", path="a.py"), text("Set x to 2."),
                                            call("write_file", path="a.py", content="x = 2\n"),
                                            text("Done: x is 2 in a.py.")] + [text("Done.")] * 4)
    agent.run("Change x in a.py to 2.")
    loop = [r for r in requests if str(r[0].get("content", "")).startswith("You are Vega")]
    assert len(loop) >= 2 and steps[:2] == [0, 1]
    assert all("(memory)" not in str(m.get("content")) for m in loop[0])
    assert loop[1][-1]["content"] == "(memory) the map"
    assert sum("(memory)" in str(m.get("content")) for m in loop[1]) == 1
    assert all("(memory)" not in str(m.get("content")) for m in agent.messages)


def test_offer_delegate_false_hides_delegate_task(monkeypatch, tmp_path) -> None:
    absent(monkeypatch)
    from test_agent_loop import agent_with, text

    agent, _ = agent_with(tmp_path, [text("Done.")])
    tools: list[list[str]] = []
    real = agent._request

    def spy(messages, available, step, response_format=None):
        tools.append([t["function"]["name"] for t in available])
        return real(messages, available, step, response_format)

    monkeypatch.setattr(agent, "_request", spy)
    agent.offer_delegate = False
    agent.run("add a docstring to nothing; just say done")
    assert tools and "delegate_task" not in tools[0]
    assert agent.can_delegate  # the agent keeps its role (clarify gate, routing)


def test_larger_window_default_and_bad_results(monkeypatch) -> None:
    """larger_window (lab grow_context): None without the package; a result is used only with a
    model name and a window larger than the agent's; a bad keep_tokens is dropped."""
    from types import SimpleNamespace

    agent = SimpleNamespace(num_ctx=8192)
    absent(monkeypatch)
    assert extensions.larger_window(agent, step=3) is None
    results: list[Any] = [
        {"model": "vega-x-16k", "num_ctx": 16384, "reason": "r", "keep_tokens": 2867},
        {"model": "vega-x-8k", "num_ctx": 8192}, {"num_ctx": 16384}, "no",
        {"model": "vega-x-16k", "num_ctx": 16384, "keep_tokens": -1},
    ]
    present(monkeypatch, larger_window=lambda *a, **k: results.pop(0))
    assert extensions.larger_window(agent, step=3) == {
        "model": "vega-x-16k", "num_ctx": 16384, "reason": "r", "keep_tokens": 2867}
    assert extensions.larger_window(agent, step=3) is None
    assert extensions.larger_window(agent, step=3) is None
    assert extensions.larger_window(agent, step=3) is None
    assert extensions.larger_window(agent, step=3)["keep_tokens"] is None


def test_attempt_environment_hooks_default_to_vegas_own_python(monkeypatch, tmp_path) -> None:
    """0.3.1: checks use the project's venv or Vega's own interpreter, and an attempt runs in no
    environment of an extension's (check_python / case_environment)."""
    import sys

    from vega import evals, sandbox

    absent(monkeypatch)
    assert extensions.check_python() is None
    assert sandbox._python_for(tmp_path) == sys.executable
    with extensions.case_environment({"id": "c"}, suite="fixes") as record:
        assert record == {}
    seen = []
    monkeypatch.setattr(evals, "_run_isolated_attempt", lambda case, **k: seen.append(case) or {"passed": True})
    assert evals.run_attempt({"id": "c", "task": "t"}, suite="fixes") == {"passed": True}


def test_case_environment_records_fields_and_its_error_is_a_fixture_error(monkeypatch, tmp_path) -> None:
    import contextlib

    from vega import evals, sandbox

    python = tmp_path / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")

    @contextlib.contextmanager
    def environment(case, *, suite):
        if case.get("id") == "broken":
            raise extensions.CaseEnvironmentError("no venv here")
        yield {"python_env": {"key": "k"}}

    present(monkeypatch, case_environment=environment, check_python=lambda: python)
    assert sandbox._python_for(tmp_path) == str(python)
    monkeypatch.setattr(evals, "_run_isolated_attempt", lambda case, **k: {"passed": True})
    assert evals.run_attempt({"id": "ok", "task": "t"}, suite="fixes") == {"passed": True,
                                                                        "python_env": {"key": "k"}}
    attempt = evals.run_attempt({"id": "broken", "task": "t"}, suite="fixes")
    assert attempt["failure"] == "fixture" and "no venv here" in attempt["detail"]
