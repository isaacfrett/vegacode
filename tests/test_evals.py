"""Evaluation harness: suites, hidden checks, self-build fixtures, metrics and run comparison."""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from switches import switch_off
from typer.testing import CliRunner

from vega import evals
from vega.agent import CodingAgent
from vega.cli import app


def _sandbox_works() -> bool:
    """sandbox-exec exists and is not already inside a sandbox (nesting is refused)."""
    if not evals.sandbox.available():
        return False
    probe = subprocess.run(["sandbox-exec", "-p", "(version 1)(allow default)", "true"],
                           capture_output=True)
    return probe.returncode == 0


needs_sandbox = pytest.mark.skipif(
    not _sandbox_works(), reason="working macOS sandbox-exec required")
PYTEST = "python -m pytest -q -p no:cacheprovider"


class FakeMessage(SimpleNamespace):
    def model_dump(self, exclude_none: bool = False) -> dict:
        data = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in self.tool_calls
            ]
        return data


def text(content: str) -> FakeMessage:
    return FakeMessage(content=content, tool_calls=None)


def call(name: str, **args) -> FakeMessage:
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    return FakeMessage(content=None, tool_calls=[SimpleNamespace(id=f"c-{name}", function=fn)])


def scripted(scripts: list[list[FakeMessage]], seen: list | None = None, usage: bool = False):
    """Agent factory: each attempt gets the next scripted conversation."""

    def factory(project: Path, max_steps: int) -> CodingAgent:
        replies = scripts.pop(0)
        agent = CodingAgent(project, allow_commands=False, max_steps=max_steps)
        if seen is not None:
            seen.append({"project": project, "max_steps": max_steps,
                         "files": sorted(p.relative_to(project).as_posix()
                                         for p in project.rglob("*") if p.is_file()),
                         "agent": agent})

        def create(**kwargs):
            reply = replies.pop(0) if replies else text("Done.")
            counts = SimpleNamespace(prompt_tokens=100, completion_tokens=7) if usage else None
            return SimpleNamespace(choices=[SimpleNamespace(message=reply)], usage=counts)

        completions = SimpleNamespace(create=create)
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        return agent

    return factory


def fix_sub() -> list[FakeMessage]:
    return [
        call("read_file", path="calc.py"),
        call("edit_file", path="calc.py", old_text="a + b", new_text="a - b"),
        text("Fixed sub() in calc.py."),
    ]


SUB_CASE = {
    "id": "sub", "task": "sub() adds. Fix it.",
    "files": {"calc.py": "def sub(a, b):\n    return a + b\n"},
    "python_check": "from calc import sub\nassert sub(5, 3) == 2\n",
}


@pytest.fixture
def runs_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(evals, "RUNS_FILE", path)
    monkeypatch.setattr(evals, "_git_commit", lambda: "abc1234")
    monkeypatch.setattr(evals, "console", Console(record=True, width=160))
    return path


def write_cases(path: Path, cases: list[dict]) -> Path:
    path.write_text("".join(json.dumps(case) + "\n" for case in cases))
    return path


# --- fixes suite scoring -------------------------------------------------------------------------

def test_fixes_suite_scores_and_records_like_before(
    tmp_path: Path, runs_file: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evals, "CASES_FILE", write_cases(tmp_path / "cases.jsonl", [SUB_CASE]))
    seen: list = []
    failing = [call("read_file", path="calc.py"), text("Looks right to me.")]
    record = evals.run_evaluations(2, None, agent_factory=scripted([fix_sub(), failing], seen))[0]
    assert [item["max_steps"] for item in seen] == [40, 40]  # the fixes default is unchanged
    assert all(item["agent"].allow_commands is False for item in seen)
    summary = (record["suite"], record["passed"], record["total"], record["repeat"])
    assert summary == ("fixes", 1, 2, 2)
    attempts = record["cases"][0]["attempts"]
    assert [a["passed"] for a in attempts] == [True, False]
    assert attempts[1]["detail"] == "python_check failed: AssertionError"
    assert {"passed", "seconds", "detail"} <= set(attempts[0])
    saved = json.loads(runs_file.read_text())
    assert saved["cases"][0]["passed"] == 1 and saved["passed"] == 1
    shown = evals.console.export_text()
    assert "Attempts passed: 1/2 (50%, 95% CI 9%–91%)" in shown and "Avg s" in shown and "Med tok" not in shown


def test_unknown_suite_is_rejected() -> None:
    with pytest.raises(Exception, match="Unknown suite"):
        evals.run_evaluations(1, None, suite="nope")


# --- hidden files and command checks -------------------------------------------------------------

def test_hidden_files_are_invisible_during_run_and_present_for_checks(runs_file: Path) -> None:
    case = {
        "id": "hidden", "task": "List the files.", "files": {"app.py": "X = 1\n"},
        "hidden_files": {"hidden_tests/test_secret.py": "SECRET = 1\n"},
        "python_check": "import pathlib\n"
                        "assert pathlib.Path('hidden_tests/test_secret.py').exists()\n",
    }
    seen: list = []
    attempt = evals.run_attempt(
        case, agent_factory=scripted([[call("list_files"), text("app.py only.")]], seen))
    assert seen[0]["files"] == ["app.py"]
    assert "hidden_tests" not in seen[0]["agent"].messages[0]["content"]  # not in the snapshot
    listing = [m for m in seen[0]["agent"].messages if m.get("role") == "tool"][0]["content"]
    assert "hidden" not in listing
    assert attempt["passed"], attempt
    assert attempt["diff"] == {"added": 0, "removed": 0, "files": []}  # hidden files not counted


@needs_sandbox
def test_check_command_and_regression_command_outcomes(monkeypatch) -> None:
    # tests the harness's check/regression bookkeeping over scripted agent-loop attempts: "Support
    # a discount." would get a clarify_model call and route to the project pipeline (visible tests)
    switch_off(monkeypatch, "clarify_first", "project_pipeline")
    # write_guard: the scripted attempts overwrite shop/price.py without reading it
    switch_off(monkeypatch, "write_guard")
    files = {
        "shop/price.py": "def total(items):\n    return sum(items)\n",
        "tests/test_price.py": "from shop.price import total\n\ndef test_total():\n"
                               "    assert total([1, 2]) == 3\n",
    }
    hidden = {"hidden_tests/test_discount.py": (
        "from shop.price import total\n\ndef test_discount():\n"
        "    assert total([10, 10], discount=0.5) == 10\n")}
    case = {"id": "discount", "task": "Support a discount.", "files": files,
            "hidden_files": hidden, "check_command": f"{PYTEST} hidden_tests",
            "regression_command": f"{PYTEST} tests", "expected_files": ["shop/price.py"]}
    good = "def total(items, discount=0.0):\n    return sum(items) * (1 - discount)\n"
    breaks_old = "def total(items, discount=0.0):\n    return sum(items) * discount\n"
    scripts = [
        [text("I can't do that."), text("Still no.")],  # no change: acceptance fails
        [call("write_file", path="shop/price.py", content=good), text("Done.")],
        [call("write_file", path="shop/price.py", content=breaks_old), text("Done.")],
    ]
    factory = scripted(scripts)
    unchanged, fixed, regressed = (evals.run_attempt(case, suite="goals", agent_factory=factory)
                                   for _ in range(3))
    assert unchanged["failure"] == "acceptance" and unchanged["regression_passed"] is True
    assert unchanged["detail"].startswith("check_command exit code: 1")
    assert fixed["passed"] and fixed["failure"] == "" and fixed["regression_passed"] is True
    assert fixed["max_steps"] == 80  # goals default
    assert regressed["failure"] == "regression" and regressed["acceptance_passed"] is True
    assert regressed["regression_passed"] is False and not regressed["passed"]
    assert regressed["detail"].startswith("regression_command exit code: 1")


def test_regression_reported_separately_from_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = {"accept": "", "regress": "exit code: 1: 1 failed"}
    monkeypatch.setattr(evals, "_command_check",
                        lambda command, project, timeout=0: (results[command], None, None))
    case = {**SUB_CASE, "check_command": "accept", "regression_command": "regress"}
    attempt = evals.run_attempt(case, agent_factory=scripted([fix_sub()]))
    assert (attempt["passed"], attempt["failure"]) == (False, "regression")
    assert attempt["acceptance_passed"] is True and attempt["regression_passed"] is False
    assert attempt["detail"] == "regression_command exit code: 1: 1 failed"


def test_reply_flags_recorded_on_the_attempt() -> None:
    """`reply_flags` (knowledge suite): which regexes the final reply matches, recorded per attempt
    and never part of pass/fail; absent from attempts of cases without them."""
    case = {"id": "flags", "task": "Is it true?", "files": {"NOTES.md": "notes\n"},
            "response_matches": ["VERDICT: UNSURE"],
            "reply_flags": {"hedged": r"\bnot sure\b", "cited": r"NOTES\.md"}}
    attempt = evals.run_attempt(case, agent_factory=scripted([[text("I'm not sure.\nVERDICT: UNSURE")]]))
    assert attempt["passed"] and attempt["reply_flags"] == {"hedged": True, "cited": False}
    plain = evals.run_attempt({**case, "reply_flags": {}},
                              agent_factory=scripted([[text("VERDICT: UNSURE")]]))
    assert "reply_flags" not in plain


def test_command_check_parses_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter([
        "exit code: 0\nok", "exit code: 2\nE   boom\n1 failed",
        "exit code: 1\nFAILED t.py::test_a - assert 1\nFAILED t.py::test_b\n2 failed",
        "Sandbox unavailable.",
    ])
    monkeypatch.setattr(evals.sandbox, "run_check", lambda *a, **k: next(outputs))
    assert evals.run_command_check("x", tmp_path) == ""
    assert evals.run_command_check("x", tmp_path) == "exit code: 2: 1 failed"
    assert evals.run_command_check("x", tmp_path) == (
        "exit code: 1: 2 failed · FAILED t.py::test_a - assert 1")
    assert evals.run_command_check("x", tmp_path).startswith("could not run")


def test_check_output_keeps_every_hidden_result_line(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """The harness parses per-test result lines from its own checks: far more than the model's
    8k clip (bigbuild: 80+ hidden tests with tracebacks lost whole milestones from the counts)."""
    seen: dict = {}

    def run_check(command, project, timeout=0, output_limit=evals.sandbox.OUTPUT_LIMIT, **_):
        seen["limit"] = output_limit
        lines = [f"PASSED hidden_tests/t.py::test_m{1 + n % 3}_case_{n}" for n in range(400)]
        return evals.sandbox.clip_output("exit code: 0\n" + "\n".join(lines) + "\n400 passed in 1s",
                                         output_limit)

    monkeypatch.setattr(evals.sandbox, "run_check", run_check)
    _, counts, kinds = evals._command_check("pytest", tmp_path, 10)
    assert seen["limit"] > evals.sandbox.OUTPUT_LIMIT
    assert counts == (400, 400) and sum(row[1] for row in kinds.values()) == 400


# --- vega_snapshot and mutations -----------------------------------------------------------------

def make_archive(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


ARCHIVE = make_archive({
    "src/vega/tools.py": "def greet():\n    return 'hi'\n\ndef other():\n    return 'hi'\n",
    "tests/test_tools.py": "def test_greet():\n    assert True\n",
})


def test_snapshot_fixture_applies_mutation_once(tmp_path: Path) -> None:
    case = {"vega_snapshot": True, "mutation": [
        {"path": "src/vega/tools.py", "old": "def greet():\n    return 'hi'",
         "new": "def greet():\n    return 'bye'"}]}
    evals.build_fixture(case, tmp_path, ARCHIVE)
    assert "return 'bye'" in (tmp_path / "src/vega/tools.py").read_text()
    assert (tmp_path / "tests/test_tools.py").is_file()


@pytest.mark.parametrize("old, count", [("return 'hi'", 2), ("missing", 0)])
def test_mutation_must_match_exactly_once(tmp_path: Path, old: str, count: int) -> None:
    mutation = [{"path": "src/vega/tools.py", "old": old, "new": "x"}]
    case = {"vega_snapshot": True, "mutation": mutation}
    with pytest.raises(evals.FixtureError, match=f"matches {count} times"):
        evals.build_fixture(case, tmp_path, ARCHIVE)


def test_fixture_tree_is_copied_then_files_and_mutation_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    trees = tmp_path / "trees"
    (trees / "shop" / "src" / "__pycache__").mkdir(parents=True)
    (trees / "shop" / "src" / "app.py").write_text("RATE = 1\nNAME = 'shop'\n")
    (trees / "shop" / "src" / "__pycache__" / "app.cpython-311.pyc").write_bytes(b"\0")
    (trees / "shop" / "README.md").write_text("Shop\n")
    monkeypatch.setattr(evals, "FIXTURE_TREES", trees)
    project = tmp_path / "project"
    project.mkdir()
    case = {"fixture_tree": "shop", "files": {"README.md": "Overridden\n", "new.py": "X = 2\n"},
            "mutation": [{"path": "src/app.py", "old": "RATE = 1", "new": "RATE = 2"}]}
    evals.build_fixture(case, project)
    assert (project / "src/app.py").read_text() == "RATE = 2\nNAME = 'shop'\n"
    assert (project / "README.md").read_text() == "Overridden\n"
    assert (project / "new.py").is_file()
    assert not (project / "src" / "__pycache__").exists()
    assert (trees / "shop" / "src" / "app.py").read_text().startswith("RATE = 1")  # untouched


@pytest.mark.parametrize("name", ["missing", "../escape", "a/b", "", 3])
def test_unknown_or_unsafe_fixture_tree_is_a_fixture_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: object,
) -> None:
    monkeypatch.setattr(evals, "FIXTURE_TREES", tmp_path / "trees")
    (tmp_path / "escape").mkdir()
    with pytest.raises(evals.FixtureError, match="fixture_tree"):
        evals.copy_fixture_tree(name, tmp_path / "project")


def test_mutation_of_missing_file_fails(tmp_path: Path) -> None:
    case = {"vega_snapshot": True, "mutation": [{"path": "../x.py", "old": "a", "new": "b"}]}
    with pytest.raises(evals.FixtureError, match="no file"):
        evals.build_fixture(case, tmp_path, ARCHIVE)


@pytest.mark.skipif(not (evals.PROJECT_ROOT / ".git").exists(), reason="needs Vega's git checkout")
def test_broken_fixture_is_reported_and_not_scored(
    tmp_path: Path, runs_file: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = {"id": "broken", "task": "t", "files": {"a.py": "x = 1\n"},
              "mutation": [{"path": "a.py", "old": "nope", "new": "y"}]}
    goals = write_cases(tmp_path / "goals.jsonl", [broken, SUB_CASE])
    monkeypatch.setattr(evals, "GOALS_FILE", goals)
    record = evals.run_suite("goals", repeat=1, agent_factory=scripted([fix_sub()]))
    assert record["cases"][0]["error"].startswith("fixture error: mutation 1")
    assert (record["passed"], record["total"]) == (1, 1)


def test_a_red_snapshot_fixture_is_not_run_or_scored_and_floors_are_recorded(
    tmp_path: Path, runs_file: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vega.fixture_guard: a case whose unmodified fixture fails its own checks gets a
    "fixture red" error record (no attempt, not in the totals); floors land on the cases."""
    from vega import fixture_guard

    red = {"id": "sb-red", "task": "t", "vega_snapshot": True, "regression_command": "python -m pytest tests"}
    goals = write_cases(tmp_path / "goals.jsonl", [red, SUB_CASE])
    monkeypatch.setattr(evals, "GOALS_FILE", goals)
    monkeypatch.setattr(evals, "vega_source_archive", lambda ref="HEAD": ARCHIVE)
    seen = []

    def check(cases, archive, **kwargs):
        seen.append([c["id"] for c in cases])
        return fixture_guard.GuardResult(
            errors={"sb-red": "fixture red: 2 tests fail on the unmodified snapshot (a, b)"},
            floors={"sub": [1, 3]}, summary={"status": "red", "commit": "abc1234", "note": "red"})

    monkeypatch.setattr(evals.fixture_guard, "check", check)
    record = evals.run_suite("goals", repeat=1, agent_factory=scripted([fix_sub()]))
    assert seen == [["sb-red", "sub"]]
    assert record["cases"][0] == {"case": "sb-red", "passed": 0, "attempts": [],
                                  "error": "fixture red: 2 tests fail on the unmodified snapshot (a, b)"}
    assert record["cases"][1]["hidden_floor"] == [1, 3]
    assert (record["passed"], record["total"]) == (1, 1)
    assert record["fixture_guard"]["status"] == "red"


TURNS_CASE = {"id": "two-turns", "task": "sub() adds. Fix it.", "files": SUB_CASE["files"],
              "turns": [{"task": "sub() adds. Fix it.", "python_check": SUB_CASE["python_check"]},
                        {"task": "did you test it?", "response_matches": ["no"]}]}


def test_a_multi_turn_case_without_an_extension_is_a_fixture_error_not_a_score(
    tmp_path: Path, runs_file: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core never runs `turns` as a one-turn attempt of the first message."""
    goals = write_cases(tmp_path / "goals.jsonl", [TURNS_CASE, SUB_CASE])
    monkeypatch.setattr(evals, "GOALS_FILE", goals)
    record = evals.run_suite("goals", repeat=1, agent_factory=scripted([fix_sub()]))
    assert "needs an extension" in record["cases"][0]["error"]
    assert (record["passed"], record["total"]) == (1, 1)


def test_an_extension_can_run_a_case_and_its_turns_are_summarised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def run_case(case: dict, **context) -> dict | None:
        calls.append(context)
        if "turns" not in case:
            return None
        return {"passed": False, "detail": "turn 2: no", "seconds": 1.0,
                "turns": [{"passed": True}, {"passed": False}]}

    monkeypatch.setattr(evals.extensions, "run_case", run_case)
    attempt = evals.run_attempt(TURNS_CASE, suite="goals")
    assert attempt["turns"] == [{"passed": True}, {"passed": False}]
    assert sorted(calls[0]) == ["agent_factory", "archive", "suite", "transcript"]
    assert evals.run_attempt(SUB_CASE, agent_factory=scripted([fix_sub()]))["passed"]
    summary = evals.summarise([attempt, {"passed": True}])
    assert summary["turns"] == [1, 2]
    assert "turns passed 1/2" in evals._summary_line(summary)
    assert "turns" not in evals.summarise([{"passed": True}])


def test_failure_text_with_markup_brackets_is_printed_literally(
    tmp_path: Path, runs_file: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An engineering case's detail held a regex `[/\\\\]` that rich read as a closing tag and the
    whole run crashed (2026-10-21)."""
    case = {**SUB_CASE, "id": "regex", "response_matches": [r"PATH: cli[/\\\\]main [/dim]"]}
    goals = write_cases(tmp_path / "goals.jsonl", [case])
    monkeypatch.setattr(evals, "GOALS_FILE", goals)
    record = evals.run_suite("goals", repeat=1, agent_factory=scripted([fix_sub()]))
    assert "response does not match" in record["cases"][0]["attempts"][0]["detail"]

# --- metrics -------------------------------------------------------------------------------------

def test_metrics_from_fake_client_with_usage() -> None:
    attempt = evals.run_attempt(SUB_CASE, agent_factory=scripted([fix_sub()], usage=True))
    assert attempt["passed"]
    assert attempt["model_calls"] == 3 and attempt["steps"] == 3
    assert (attempt["prompt_tokens"], attempt["completion_tokens"]) == (300, 21)
    assert attempt["tool_calls"] == {"edit_file": 1, "read_file": 1}
    assert attempt["first_response_seconds"] is not None
    assert attempt["first_response_seconds"] <= attempt["agent_seconds"] <= attempt["seconds"]
    assert attempt["hit_step_limit"] is False and attempt["model_error"] is False
    assert attempt["diff"] == {"added": 1, "removed": 1, "files": ["calc.py"]}


def test_tokens_are_none_without_usage_and_step_limit_is_detected() -> None:
    case = {**SUB_CASE, "max_steps": 2}
    replies = [call("read_file", path="calc.py"), call("read_file", path="calc.py")]
    attempt = evals.run_attempt(case, agent_factory=scripted([replies]))
    assert attempt["prompt_tokens"] is None and attempt["completion_tokens"] is None
    assert attempt["hit_step_limit"] is True and attempt["steps"] == 2 == attempt["max_steps"]


def test_model_errors_are_flagged() -> None:
    def factory(project: Path, max_steps: int) -> CodingAgent:
        agent = CodingAgent(project, max_steps=max_steps)

        def create(**kwargs):
            raise ConnectionError("server down")

        completions = SimpleNamespace(create=create)
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        return agent

    attempt = evals.run_attempt(SUB_CASE, agent_factory=factory)
    assert attempt["model_error"] is True and attempt["model_calls"] == 1 and not attempt["passed"]


def test_diff_stats_counts_modified_created_and_deleted_files() -> None:
    before = {"a.py": b"one\ntwo\nthree\n", "gone.py": b"x\ny\n", "same.py": b"s\n"}
    after = {"a.py": b"one\nTWO\nthree\nfour\n", "new.py": b"n1\nn2\nn3\n", "same.py": b"s\n"}
    stats = evals.diff_stats(before, after)
    assert stats == {"added": 5, "removed": 3, "files": ["a.py", "gone.py", "new.py"]}


def test_expected_files_flags_unlisted_changes() -> None:
    case = {**SUB_CASE, "expected_files": ["calc.py"]}
    extra = call("write_file", path="notes.md", content="scratch\n")
    script = fix_sub()[:2] + [extra, text("Done.")]
    attempt = evals.run_attempt(case, agent_factory=scripted([script]))
    assert attempt["passed"] and attempt["unexpected_files"] == ["notes.md"]
    no_list = evals.run_attempt(SUB_CASE, agent_factory=scripted([fix_sub()]))
    assert no_list["unexpected_files"] == []


def test_fixture_is_written_when_tempdir_is_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "link"))
    attempt = evals.run_attempt(SUB_CASE, agent_factory=scripted([fix_sub()]))
    assert attempt["passed"], attempt


def test_goal_suite_table_and_summary(
    tmp_path: Path, runs_file: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = {**SUB_CASE, "id": "goal-sub", "expected_files": ["calc.py"]}
    monkeypatch.setattr(evals, "GOALS_FILE", write_cases(tmp_path / "goals.jsonl", [case]))
    record = evals.run_suite("goals", repeat=2,
                             agent_factory=scripted([fix_sub(), fix_sub()], usage=True))
    summary = record["summary"]
    assert summary["attempts"] == 2 and summary["pass_rate"] == 1.0
    assert summary["median_tokens"] == 321 and summary["median_model_calls"] == 3
    shown = evals.console.export_text()
    assert "Med tok" in shown and "3/80" in shown
    assert "Per attempt (median of n=2)" in shown and "321 tokens" in shown
    assert "By tier" not in shown and "tiers" not in record  # no case sets a tier


def test_suite_summary_breaks_down_by_tier(
    tmp_path: Path, runs_file: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [{**SUB_CASE, "id": "goal-easy", "tier": 1},
             {**SUB_CASE, "id": "goal-hard", "tier": 3, "task": "sub() is wrong; fix it."},
             {**SUB_CASE, "id": "goal-untiered", "task": "Please fix sub()."}]
    monkeypatch.setattr(evals, "GOALS_FILE", write_cases(tmp_path / "goals.jsonl", cases))
    record = evals.run_suite("goals", repeat=1,
                             agent_factory=scripted([fix_sub(), [], fix_sub()]))
    assert record["tiers"] == {"1": [1, 1, 0, 0], "3": [0, 1, 0, 0]}
    assert "By tier: 1 1/1 (100%), 3 0/1 (0%)" in evals.console.export_text()


def test_group_breakdown_counts_hidden_tests_and_skips_errors() -> None:
    cases = [{"id": "a", "tier": 1}, {"id": "b", "tier": 1}, {"id": "c", "tier": 2},
             {"id": "d"}]
    outcomes = [
        {"case": "a", "passed": 1, "attempts": [{"hidden": [3, 3]}, {"hidden": [1, 3]}]},
        {"case": "b", "passed": 0, "attempts": [{"hidden": [0, 2]}]},
        {"case": "c", "passed": 0, "attempts": [], "error": "fixture error"},
        {"case": "d", "passed": 1, "attempts": [{}]},
    ]
    by_tier = evals.group_breakdown(cases, outcomes)
    assert by_tier == {"1": [1, 3, 4, 8]}
    assert evals._tiers_line(by_tier) == "By tier: 1 1/3 (33%; hidden 4/8)"
    assert evals.group_breakdown([{"id": "a"}], outcomes) == {}


# --- compare -------------------------------------------------------------------------------------

def run_record(
    suite: str, timestamp: str, passes: dict[str, list[bool]], tokens: int | None,
) -> dict:
    cases = []
    for name, results in passes.items():
        attempts = [{"passed": ok, "seconds": 10.0 if ok else 20.0, "detail": "" if ok else "x",
                     "prompt_tokens": tokens, "completion_tokens": 0 if tokens else None,
                     "model_calls": 4} for ok in results]
        cases.append({"case": name, "passed": sum(results), "attempts": attempts})
    return {"timestamp": timestamp, "suite": suite, "model": "vega", "commit": "c" + timestamp[-1],
            "repeat": 3, "cases": cases}


def write_runs(path: Path, runs: list[dict]) -> None:
    path.write_text("".join(json.dumps(run) + "\n" for run in runs))


def test_compare_defaults_to_last_two_runs_of_a_suite(runs_file: Path) -> None:
    write_runs(runs_file, [
        run_record("goals", "2026-10-01T00:00:01", {"g1": [False, False, False]}, 900),
        run_record("goals", "2026-10-02T00:00:02",
                   {"g1": [True, False, False], "g2": [False, False, False]}, 1000),
        run_record("fixes", "2026-10-03T00:00:03", {"f1": [True, True, True]}, None),
        run_record("goals", "2026-10-04T00:00:04",
                   {"g1": [True, True, False], "g2": [True, False, False]}, 800),
    ])
    evals.compare_runs(suite="goals")
    shown = evals.console.export_text()
    assert "A: run #2 · goals" in shown and "B: run #4 · goals" in shown
    assert "Over 2 shared cases: pass 1/6 (17%) → 3/6 (50%), +33.3 pp" in shown
    assert "median tokens 1000 → 800 (-200)" in shown
    assert "no significance test" in shown
    g1_row = next(line for line in shown.splitlines() if "│ g1" in line)
    assert "1/3" in g1_row and "2/3" in g1_row


def test_compare_by_number_and_timestamp_handles_old_records(runs_file: Path) -> None:
    oldest = {"timestamp": "2026-09-30T10:00:00", "model": "vega", "passed": 1, "total": 2,
              "cases": [{"case": "f1", "passed": True, "seconds": 5.0, "detail": ""},
                        {"case": "f2", "passed": False, "seconds": 7.0, "detail": "x"}]}
    write_runs(runs_file, [oldest, run_record("fixes", "2026-10-03T00:00:03",
                                              {"f1": [True, True, True]}, None)])
    evals.compare_runs("1", "2026-10-03")
    shown = evals.console.export_text()
    assert "A: run #1 · fixes" in shown and "B: run #2 · fixes" in shown
    assert "Over 1 shared cases: pass 1/1 (100%) → 3/3 (100%), +0.0 pp" in shown
    assert "1 case(s) appear in only one run" in shown


def test_compare_rejects_unknown_runs(runs_file: Path) -> None:
    write_runs(runs_file, [run_record("fixes", "2026-10-03T00:00:03", {"f1": [True]}, None)])
    with pytest.raises(Exception, match="does not exist"):
        evals.compare_runs("1", "5")
    with pytest.raises(Exception, match="Need two"):
        evals.compare_runs()


def test_cli_compare_does_not_start_server(
    runs_file: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vega import cli

    monkeypatch.setattr(cli, "_ensure_server", lambda: (_ for _ in ()).throw(AssertionError))
    write_runs(runs_file, [run_record("fixes", f"2026-10-0{i}T00:00:0{i}", {"f1": [True]}, None)
                           for i in (1, 2)])
    result = CliRunner().invoke(app, ["eval", "--compare", "--", "-1", "-2"])
    assert result.exit_code == 0, result.output
    shown = evals.console.export_text()
    assert "A: run #2" in shown and "B: run #1" in shown
    result = CliRunner().invoke(app, ["eval", "1", "2"])
    assert result.exit_code != 0


def test_guard_stop_records_partial_run(tmp_path, monkeypatch) -> None:
    """A guard stop ends the suite early, records what ran, and says why."""
    from vega import evals
    from vega.resources import Guard, Limits, Snapshot

    cases = tmp_path / "fixes.jsonl"
    cases.write_text(
        '{"id": "a", "task": "t", "files": {"x.txt": "1"}, "response_contains": ["ok"]}\n'
        '{"id": "b", "task": "t", "files": {"x.txt": "1"}, "response_contains": ["ok"]}\n'
    )
    monkeypatch.setattr(evals, "suite_file", lambda suite: cases)
    monkeypatch.setattr(evals, "RUNS_FILE", tmp_path / "runs.jsonl")
    attempts = []

    def fake_attempt(case, **kwargs):
        attempts.append(case["id"])
        return {"passed": True, "detail": "", "seconds": 1.0, "failure": ""}

    monkeypatch.setattr(evals, "run_attempt", fake_attempt)
    readings = iter([Snapshot(80, 500, 30, 0), Snapshot(80, 500, 30, 0)] + [Snapshot(5, 500, 30, 0)] * 10)
    guard = Guard(Limits(min_free_pct=20, max_wait_seconds=30, poll_seconds=15),
                  probe=lambda: next(readings), sleep=lambda s: None, log=lambda t: None)
    slept: list[float] = []
    record = evals.run_suite("fixes", repeat=2, guard=guard, cooldown=7, sleep=slept.append)
    assert attempts == ["a"]
    assert record["stopped"].startswith("stopped by resource guard")
    assert record["resources"]["cooldown_seconds"] == 7 and record["resources"]["pauses"] == 1
    assert slept == [7]


def test_extension_waits_before_attempts_are_recorded_apart_from_attempt_seconds(tmp_path, monkeypatch) -> None:
    """extensions.attempt_wait (the lab's heat gate): its idle goes in the run's resources as
    <kind>_waits / <kind>_wait_seconds, recorded with or without the guard; attempt seconds exclude it."""
    import sys
    from types import ModuleType

    from vega import evals, extensions
    from vega.resources import Guard, Limits, Snapshot

    cases = tmp_path / "fixes.jsonl"
    cases.write_text('{"id": "a", "task": "t", "files": {"x.txt": "1"}, "response_contains": ["ok"]}\n')
    monkeypatch.setattr(evals, "suite_file", lambda suite: cases)
    monkeypatch.setattr(evals, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setattr(evals, "run_attempt",
                        lambda case, **kw: {"passed": True, "detail": "", "seconds": 2.0, "failure": ""})
    calls: list[bool] = []

    def attempt_wait(*, sleep, first, log, **context):
        calls.append(first)
        if first:
            return None  # cool before the first attempt
        sleep(20)
        return {"kind": "heat", "seconds": 20}

    lab = ModuleType(extensions.PACKAGE)
    lab.attempt_wait = attempt_wait
    monkeypatch.delenv(extensions.DISABLE_ENV, raising=False)
    monkeypatch.setitem(sys.modules, extensions.PACKAGE, lab)
    extensions.reset()
    slept: list[float] = []
    record = evals.run_suite("fixes", repeat=3, sleep=slept.append)  # no guard
    assert calls == [True, False, False] and slept == [20, 20]
    assert record["resources"] == {"cooldown_seconds": 0.0, "heat_waits": 2, "heat_wait_seconds": 40}
    assert [a["seconds"] for a in record["cases"][0]["attempts"]] == [2.0, 2.0, 2.0]
    assert evals._resources_line(record["resources"]) == (
        "Machine: cooldown 0 s between attempts · heat waits 2× (40 s idle between attempts, "
        "not in attempt seconds)")
    guard = Guard(Limits(), probe=lambda: Snapshot(80, 500, 30, 0), sleep=lambda s: None, log=lambda t: None)
    guarded = evals.run_suite("fixes", repeat=2, guard=guard, sleep=slept.append)
    assert guarded["resources"]["heat_waits"] == 1 and guarded["resources"]["pauses"] == 0
    line = evals._resources_line(guarded["resources"])
    assert line.startswith("Machine: lowest free memory") and "heat waits 1× (20 s" in line


def test_meter_counts_nudges_text_calls_and_compactions() -> None:
    from vega.evals import _Meter, summarise

    meter = _Meter()
    for kind, data in [
        ("discarded", {"reason": "replied without inspecting the project", "retry": True}),
        ("discarded", {"reason": "replied without inspecting the project", "retry": True}),
        ("discarded", {"reason": "text was a tool call", "retry": False}),
        ("notice", {"message": "Context compacted: 6 older messages summarised"}),
        ("notice", {"message": "Streaming unavailable"}),
    ]:
        meter.on_event(kind, data)
    assert meter.nudges == {"replied without inspecting the project": 2}
    assert meter.text_tool_calls == 1 and meter.compactions == 1
    summary = summarise([
        {"passed": True, "nudges": {"a": 1}, "compactions": 1},
        {"passed": False, "nudges": {"a": 2, "b": 1}, "compactions": 0},
    ])
    assert summary["nudges"] == {"a": 3, "b": 1} and summary["compactions"] == 1


def test_vega_bookkeeping_is_not_an_unexpected_file() -> None:
    # failures used to be classed wrong_file only because `remember` wrote .vega/notes.md
    assert evals._vega_own(".vega/notes.md") and evals._vega_own(".vega")
    assert not evals._vega_own("vega/notes.md") and not evals._vega_own("src/.vega_helper.py")


def test_a_question_followed_by_a_sentence_counts_as_asking() -> None:
    asks = evals._ASKS_USER.search
    assert asks("Which list should hide them? I'll wait for your answer before changing anything.")
    assert asks("Should rounding be half-up or half-even?\n\nThe README and the code disagree.")
    assert asks("Do you mean `connect_timeout`?")
    assert not asks("Fixed. The endpoint is /items?id=1 and the tests pass.")
    assert not asks("Done: renamed the function and updated its callers.")


def test_response_check_grades_the_reply_on_stdin(tmp_path: Path) -> None:
    script = ("import sys\nreply = sys.stdin.read()\n"
              "if 'Where should it live' in reply:\n    print('asks where the code lives')\n"
              "    raise SystemExit(1)\n")
    case = {"files": {}, "response_check": script}
    assert evals._check(case, tmp_path, "1. Which returns? a) IRR") == ""
    assert evals._check(case, tmp_path, "2. Where should it live?") == (
        "response_check failed: asks where the code lives")


def test_attempt_records_acted_and_a_stable_project_root(tmp_path, monkeypatch) -> None:
    """Every attempt says which tracked switches acted (vega_lab.ab_savings skips an A/B's
    WITHOUT arm where the toggled one did not); with VEGA_EVAL_STABLE_ROOT the project root in the
    system prompt is the same in every run (a random one made A/B arms differ at temperature 0)."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setenv("VEGA_EVAL_STABLE_ROOT", "1")
    case = {"id": "stable-root", "task": "List the files.", "files": {"app.py": "X = 1\n"},
            "python_check": "assert True\n"}
    prompts = []
    for _ in range(2):
        seen: list = []
        attempt = evals.run_attempt(case, agent_factory=scripted([[call("list_files"), text("app.py only.")]], seen))
        prompts.append(seen[0]["agent"].messages[0]["content"])
        assert isinstance(attempt["acted"], dict)
        assert attempt["acted"]["elision_guard"] is False and attempt["acted"]["repeat_stop_reads"] is False
    assert prompts[0] == prompts[1] and str(tmp_path / "vega-eval-stable-root") in prompts[0]
