"""Shape routing (redesign D): one decision per task from its shape, never from its length.

Each test builds a tiny project and routes a task against it (no model call).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vega import features, question_pipeline, routing

BASE_ON = "clarify_first,project_context"  # off; everything else at its default


@pytest.fixture(autouse=True)
def defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(features.ENV, BASE_ON)


# -- code spans ---------------------------------------------------------------------------------

def test_words_inside_code_spans_do_not_make_a_change() -> None:
    task = ("When someone runs `kite add NAME COLOR` (kiteshop/cli.py), what is on the "
            "call stack? Don't change any files. End with `PATH: a > b`.")
    assert routing.change_verbs(task) == []
    assert routing.read_only(task)
    masked = routing.mask_code(task)
    assert "add" not in masked and len(masked) == len(task)
    assert "kiteshop/cli.py" in masked  # a path still names a file


def test_fenced_code_and_quoted_commands_are_blanked() -> None:
    task = 'Explain this:\n```python\ndef fix(): write()\n```\nand what "git add -A" does.'
    assert routing.change_verbs(task) == []
    assert routing.asks(task)


def test_scoped_do_not_change_is_a_constraint_not_read_only() -> None:
    assert not routing.read_only("Tidy the API. Don't change what the functions do.")
    assert not routing.read_only("Fix it. Do not change any other file.")
    assert not routing.read_only("Review it: do not change any file except creating `r.json`.")
    assert not routing.read_only("Make tests pass; do not edit anything under tests/.")
    assert routing.read_only("Why is it slow? Don't change any files.")


def test_backticked_find_does_not_make_rewrite_a_question(tmp_path: Path) -> None:
    (tmp_path / "graphkit").mkdir()
    (tmp_path / "graphkit" / "__init__.py").write_text("")
    (tmp_path / "graphkit" / "components.py").write_text(
        "class Components:\n    def find(self, x):\n        return x\n")
    task = "`graphkit.components.Components`: `find` is slow. Rewrite it with a cache."
    route = routing.route_task(task, tmp_path)
    assert route.change and route.name == "direct"
    assert route.targets == ["graphkit/components.py"]


def _server(root: Path) -> None:
    (root / "app").mkdir()
    (root / "app" / "__init__.py").write_text("")
    (root / "app" / "server.py").write_text("PORT = 8080\n\n\ndef add(a, b):\n    return a - b\n")


@pytest.mark.parametrize("task", [
    "What port does the server use?",
    "Which port is the API on?",
    "Why is add wrong?",
    "what does add return?",
    "What does `add()` return?",
    "Why does update fail on empty lists?",
    "How does the remove function work?",
])
def test_question_with_a_noun_or_function_named_like_a_verb_is_a_question(
        tmp_path: Path, task: str) -> None:
    _server(tmp_path)
    assert routing.change_verbs(task) == []
    route = routing.route_task(task, tmp_path)
    assert route.question and not route.change, route.reasons
    assert question_pipeline.is_question(task)


@pytest.mark.parametrize(("task", "verb"), [
    ("Port `app/server.py` to Go.", "port"),
    ("Port the parser to Rust.", "port"),
    ("Please port app/server.py to TypeScript.", "port"),
    ("Can you port the server to Go?", "port"),
    ("Add a function that subtracts two numbers to app/server.py.", "add"),
    ("Can you add tests for add()?", "add"),
    ("Why is add wrong? Fix it.", "fix"),
    ("Could you update the port in app/server.py?", "update"),
    ("How do I add a new command?", "add"),
])
def test_request_with_the_same_verbs_is_still_a_change(tmp_path: Path, task: str,
                                                        verb: str) -> None:
    _server(tmp_path)
    assert verb in routing.change_verbs(task)
    route = routing.route_task(task, tmp_path)
    assert route.change and not route.question, route.reasons
    if verb != "port":  # is_question's own verb list has no "port" (routing decides above)
        assert not question_pipeline.is_question(task)


def test_port_as_a_noun_in_a_statement_is_not_a_verb() -> None:
    assert routing.change_verbs("The default port stays 8080. Fix the timeout.") == ["fix"]
    assert routing.change_verbs("PORT must be a whole number.") == []
    assert routing.change_verbs("Change the server port to 3000.") == ["change"]


# -- target resolution ------------------------------------------------------------------------------

def _package(root: Path) -> None:
    (root / "sundial").mkdir()
    (root / "sundial" / "__init__.py").write_text("")
    (root / "sundial" / "cycles.py").write_text(
        'def next_chime(cycles, after=0):\n    """Smallest t."""\n    raise NotImplementedError\n')
    (root / "sundial" / "report.py").write_text("def show(x):\n    return x\n")


def test_dotted_names_resolve_to_their_file(tmp_path: Path) -> None:
    _package(tmp_path)
    route = routing.route_task("Implement `sundial.cycles.next_chime(cycles, after=0)` exactly "
                               "as its docstring specifies." + " Detail." * 80, tmp_path)
    assert route.targets == ["sundial/cycles.py"]
    assert route.name == "direct"  # 700+ characters: length does not decide


def test_the_one_file_holding_a_named_stub_is_the_target(tmp_path: Path) -> None:
    _package(tmp_path)
    route = routing.route_task("Write next_chime(cycles, after) so it returns the smallest t.",
                               tmp_path)
    assert route.targets == ["sundial/cycles.py"]
    assert "stub the task names" in "; ".join(route.reasons)


def test_read_only_inputs_do_not_block_direct(tmp_path: Path) -> None:
    _package(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "plan.csv").write_text("period,offset\n4,1\n")
    (tmp_path / "SPEC.md").write_text("# Spec\nRules.\n")
    route = routing.route_task("Implement `next_chime` in `sundial/cycles.py`, following the "
                               "rules in `SPEC.md`; it reads `data/plan.csv`.", tmp_path)
    assert route.name == "direct"
    assert route.inputs == ["SPEC.md", "data/plan.csv"]


def test_a_data_file_to_change_is_not_a_direct_code_edit(tmp_path: Path) -> None:
    _package(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "plan.csv").write_text("period,offset\n4,1\n")
    route = routing.route_task("Fix the offsets in `data/plan.csv`: every offset must be "
                               "below its period.", tmp_path)
    assert route.name == "agent"
    assert route.other_targets == ["data/plan.csv"] and route.targets == []


def test_other_language_targets_are_excluded_from_direct(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("import subprocess\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text("pub fn wrap() {}\n")
    route = routing.route_task("Fix `wrap` in `src/lib.rs` so long words split. Run the tests "
                               "with `python check.py`.", tmp_path)
    assert route.name == "nonpython"  # nonpython_pipeline (on); with it off: the agent loop
    assert route.targets == [] and route.other_targets == ["src/lib.rs"]
    assert route.helpers == ["check.py"]


def test_a_port_reads_the_source_and_writes_the_python(tmp_path: Path) -> None:
    (tmp_path / "durations.js").write_text("function parse() {}\n")
    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "durations.py").write_text("def parse(text):\n    raise NotImplementedError\n")
    route = routing.route_task("Port `durations.js` to Python in `python/durations.py`.", tmp_path)
    assert route.name == "direct"
    assert route.targets == ["python/durations.py"] and route.inputs == ["durations.js"]


def test_files_the_task_protects_are_never_targets(tmp_path: Path) -> None:
    _package(tmp_path)
    route = routing.route_task("Implement `next_chime` in sundial/cycles.py using "
                               "`sundial.report.show` from `sundial/report.py` (tested; do not "
                               "change it).", tmp_path)
    assert route.targets == ["sundial/cycles.py"]
    assert "sundial/report.py" in route.inputs


# -- the writer's budget ------------------------------------------------------------------------

def test_length_cap_is_computed_from_file_size(tmp_path: Path) -> None:
    _package(tmp_path)
    long_task = "Implement `sundial.cycles.next_chime`. " + "Spec line. " * 300  # 3,300 chars
    assert routing.route_task(long_task, tmp_path).name == "direct"
    (tmp_path / "sundial" / "cycles.py").write_text(
        (tmp_path / "sundial" / "cycles.py").read_text() + "#" * 9000 + "\n")
    small = routing.route_task("Implement `sundial.cycles.next_chime`.", tmp_path, num_ctx=4096)
    assert small.name != "direct"
    assert "writer needs" in "; ".join(small.reasons)
    roomy = routing.route_task("Implement `sundial.cycles.next_chime`.", tmp_path, num_ctx=32768)
    assert roomy.name == "direct"


def test_writer_tokens_grow_with_the_file_not_the_task() -> None:
    small_file = routing.writer_tokens("x" * 100, 1000, 0)
    assert routing.writer_tokens("x" * 100, 9000, 0) > small_file + 4000
    assert routing.writer_tokens("x" * 2000, 1000, 0) - small_file < 600


# -- in agent.run ------------------------------------------------------------------------------------

def _agent(tmp_path: Path, monkeypatch, taken: list[str]):
    from vega import pipeline, project_pipeline, question_pipeline
    from vega.agent import CodingAgent

    def fake(name):  # records the path taken, no model call
        def run(agent, task, **kwargs):
            taken.append(name)
            return f"({name})"
        return run

    monkeypatch.setattr(question_pipeline, "run", fake("question"))
    monkeypatch.setattr(pipeline, "run", fake("direct"))
    monkeypatch.setattr(project_pipeline, "run", fake("project"))
    agent = CodingAgent(tmp_path, allow_commands=False)
    monkeypatch.setattr(agent, "_run", lambda task: ("(agent loop)", False))
    return agent


def test_run_takes_the_route_and_records_it(tmp_path: Path, monkeypatch) -> None:
    _package(tmp_path)
    taken: list[str] = []
    agent = _agent(tmp_path, monkeypatch, taken)
    task = "Implement `sundial.cycles.next_chime(cycles, after=0)`. " + "More spec. " * 60
    assert agent.run(task) == "(direct)"
    assert taken == ["direct"]
    assert agent.last_route["name"] == "direct"
    assert agent.last_route["targets"] == ["sundial/cycles.py"]


def test_a_pipeline_that_declines_falls_back_to_the_agent_loop(tmp_path: Path, monkeypatch) -> None:
    from vega import pipeline

    _package(tmp_path)
    taken: list[str] = []
    agent = _agent(tmp_path, monkeypatch, taken)
    monkeypatch.setattr(pipeline, "run", lambda agent, task, **kwargs: None)
    assert agent.run("Implement `sundial.cycles.next_chime`.") == "(agent loop)"
    assert agent.last_route["name"] == "direct" and agent.last_route["fell_back"]


def test_routing_records_a_debug_stage(tmp_path: Path, monkeypatch) -> None:
    from vega import debuglog

    _package(tmp_path)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(debuglog, "record", lambda kind, **data: events.append((kind, data)))
    agent = _agent(tmp_path, monkeypatch, [])
    agent.run("Implement `sundial.cycles.next_chime`.")
    stage = next(data for kind, data in events if kind == "stage" and data.get("stage") == "route")
    assert stage["name"] == "direct" and stage["targets"] == ["sundial/cycles.py"]
    assert stage["reasons"]


def test_switched_off_pipelines_are_skipped_by_the_table(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(features.ENV, BASE_ON + ",direct_pipeline")
    _package(tmp_path)
    route = routing.route_task("Implement `sundial.cycles.next_chime`.", tmp_path)
    assert route.name == "project"  # the next row: one target, with acceptance checks
    assert "direct_pipeline is off" in "; ".join(route.reasons)
