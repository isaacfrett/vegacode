"""Autopilot: branch, rounds, check gating, commits and the report, with a scripted fake model.

Every repository here is a temporary one under tmp_path with its own local identity; the user's
global Git config is hidden so its hooks, signing or identity cannot leak in.
"""

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from rich.console import Console
from switches import switch_off

from vega import autopilot, sandbox, server
from vega.agent import CodingAgent
from vega.autopilot import (
    TRAILER, AutopilotRefused, detect_check, format_report, run_goal, slugify, summarize,
)
from vega.ui import TerminalRenderer

needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="needs macOS sandbox-exec")
_ids = itertools.count()
CHECK = "python -m pytest -q"
CALC = "def add(a, b):\n    return a + b\n"
CALC_WITH_SUB = CALC + "\n\ndef sub(a, b):\n    return a - b\n"
BROKEN = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"
TEST = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"


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
    return FakeMessage(content=None, tool_calls=[SimpleNamespace(id=f"a{next(_ids)}", function=fn)])


def plan(*items: tuple[str, str]) -> FakeMessage:
    return call("update_plan", items=[{"text": t, "status": s} for t, s in items])


class Script:
    """Agent factory whose model replies come from a list shared across all rounds."""

    def __init__(self, replies: list[FakeMessage]):
        self.replies = replies
        self.seen: list[list[dict]] = []
        self.agent: CodingAgent | None = None

    def __call__(self, project: Path, on_event) -> CodingAgent:
        # Deliberately permissive settings: autopilot must switch commands and review off itself.
        agent = CodingAgent(project, allow_commands=True, review=True, on_event=on_event)

        def create(**kwargs):
            self.seen.append(json.loads(json.dumps(kwargs["messages"])))
            return SimpleNamespace(choices=[SimpleNamespace(message=self.replies.pop(0))])

        completions = SimpleNamespace(create=create)
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        self.agent = agent
        return agent

    def user_messages(self) -> list[str]:
        return [m["content"] for m in self.seen[-1] if m["role"] == "user"]


@pytest.fixture(autouse=True)
def isolated_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "global.gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL",
                 "GIT_DIR", "GIT_WORK_TREE"):
        monkeypatch.delenv(name, raising=False)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def make_repo(tmp_path: Path, files: dict[str, str] | None = None, identity: bool = True) -> Path:
    repo = tmp_path / "proj"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    if identity:
        git(repo, "config", "user.name", "Test User")
        git(repo, "config", "user.email", "test@example.com")
    for name, content in (files or {"calc.py": CALC, "tests/test_calc.py": TEST}).items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(content)
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=Setup", "-c", "user.email=setup@example.com",
        "commit", "-q", "-m", "initial")
    return repo


def branch_commits(repo: Path, start: str) -> list[str]:
    return git(repo, "log", "--format=%H", f"{start}..HEAD").split()


# -- refusals ------------------------------------------------------------------------------------

def test_refuses_outside_a_git_repository(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(AutopilotRefused, match="not inside a Git repository"):
        run_goal(plain, "do things", check="none", agent_factory=Script([]))


def test_refuses_repository_without_commits(tmp_path: Path) -> None:
    repo = tmp_path / "empty"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    with pytest.raises(AutopilotRefused, match="no commits"):
        run_goal(repo, "do things", check="none", agent_factory=Script([]))


@pytest.mark.parametrize("dirty", ["untracked", "modified", "staged"])
def test_refuses_dirty_tree_and_leaves_it_alone(tmp_path: Path, dirty: str) -> None:
    repo = make_repo(tmp_path)
    if dirty == "untracked":
        (repo / "scratch.txt").write_text("mine")
    else:
        (repo / "calc.py").write_text(CALC + "# my edit\n")
        if dirty == "staged":
            git(repo, "add", "calc.py")
    status = git(repo, "status", "--porcelain")
    with pytest.raises(AutopilotRefused, match="never stashes or discards"):
        run_goal(repo, "do things", check="none", agent_factory=Script([]))
    assert git(repo, "status", "--porcelain") == status
    assert git(repo, "branch", "--show-current") == "main"
    assert git(repo, "branch", "--format=%(refname:short)") == "main"


def test_ignored_files_do_not_count_as_dirty(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": CALC, ".gitignore": "build/\n"})
    (repo / "build").mkdir()
    (repo / "build" / "out.bin").write_text("x")
    script = Script([call("read_file", path="calc.py"), text("Nothing needed; it is fine. DONE")])
    report = run_goal(repo, "check calc", check="none", agent_factory=script)
    assert report.completed and not report.commits


def test_refuses_existing_branch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    git(repo, "branch", "vega/taken")
    with pytest.raises(AutopilotRefused, match="already exists"):
        run_goal(repo, "x", branch="vega/taken", check="none", agent_factory=Script([]))
    assert git(repo, "branch", "--show-current") == "main"


def test_refuses_during_merge(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / ".git" / "MERGE_HEAD").write_text(git(repo, "rev-parse", "HEAD") + "\n")
    with pytest.raises(AutopilotRefused, match="middle of a merge"):
        run_goal(repo, "x", check="none", agent_factory=Script([]))


def test_git_helper_refuses_unlisted_commands(tmp_path: Path) -> None:
    for command in ("push", "reset", "stash", "rebase", "fetch"):
        with pytest.raises(ValueError):
            autopilot._git(tmp_path, command)


# -- rounds --------------------------------------------------------------------------------------

@needs_sandbox
def test_passing_round_is_committed_on_a_new_branch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    start = git(repo, "rev-parse", "HEAD")
    events: list[tuple[str, dict]] = []
    script = Script([
        plan(("Add sub() to calc.py", "in_progress")),
        call("write_file", path="calc.py", content=CALC_WITH_SUB),
        plan(("Add sub() to calc.py", "done")),
        text("Added `sub()` to calc.py and ran the tests. DONE"),
    ])
    report = run_goal(repo, "Add a subtract function!", agent_factory=script,
                      on_event=lambda k, d: events.append((k, d)))

    assert report.check_command == CHECK and report.baseline_check.passed
    assert report.branch.startswith("vega/add-a-subtract-function-")
    assert git(repo, "branch", "--show-current") == report.branch
    assert git(repo, "rev-parse", "main") == start  # the original branch never moves
    assert report.start_commit == start and report.original_branch == "main"
    assert report.completed and report.ok and report.rounds == 1
    assert report.stop_reason.startswith("goal complete")
    assert [c.message for c in report.commits] == [
        "vega: Added sub() to calc.py and ran the tests"
    ]
    assert branch_commits(repo, start) == [report.commits[0].hash]
    message = git(repo, "log", "-1", "--format=%B")
    assert "Autopilot round 1. Check: `python -m pytest -q` passed (exit 0)." in message
    assert message.splitlines()[-1] == TRAILER
    assert git(repo, "log", "-1", "--format=%an <%ae>") == "Test User <test@example.com>"
    assert git(repo, "status", "--porcelain") == ""
    assert (repo / "calc.py").read_text() == CALC_WITH_SUB
    assert report.plan == [{"text": "Add sub() to calc.py", "status": "done"}]
    assert "calc.py" in report.diffstat
    # The agent was forced into autonomous settings, and told to plan first.
    assert script.agent.allow_commands is False and script.agent.review is False
    # Its rounds are harness prompts for the agent loop: with the shipped defaults the project
    # pipeline (this repo has tests) would otherwise take them as the task (2026-10-17)
    assert script.agent.route_requests is False and script.agent.last_route["name"] == "agent"
    first = script.seen[0][1]["content"]
    assert "Add a subtract function!" in first and "update_plan" in first
    kinds = [d["event"] for k, d in events if k == "autopilot"]
    assert kinds == ["start", "baseline", "round_start", "check", "commit", "round_end", "stop"]


@needs_sandbox
def test_failing_check_is_fed_back_and_not_committed_until_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # write_guard: this scripted model overwrites a fixture file it never read; the test is about
    # autopilot's check feedback and commits, not the overwrite hold (tests/test_write_guard.py)
    switch_off(monkeypatch, "write_guard")
    repo = make_repo(tmp_path)
    start = git(repo, "rev-parse", "HEAD")
    commits_after_round: list[int] = []
    script = Script([
        call("write_file", path="calc.py", content=BROKEN),
        text("Added sub() to calc.py. DONE"),
        call("edit_file", path="calc.py", old_text="return a - b\n\n\ndef sub",
             new_text="return a + b\n\n\ndef sub"),
        text("Fixed add() so the tests pass again. DONE"),
    ])

    def on_event(kind: str, data: dict) -> None:
        if kind == "autopilot" and data["event"] == "round_end":
            commits_after_round.append(len(branch_commits(repo, start)))

    report = run_goal(repo, "add sub", agent_factory=script, on_event=on_event)

    assert commits_after_round == [0, 1]  # round 1 failed its check: nothing committed
    feedback = [m for m in script.user_messages() if m.startswith("(Vega autopilot) The check")]
    assert len(feedback) == 1
    expected = f"(Vega autopilot) The check `{CHECK}` failed after your changes:"
    assert feedback[0].startswith(expected)
    assert "assert" in feedback[0] and feedback[0].rstrip().endswith("Fix it.")
    assert report.ok and report.rounds == 2 and len(report.commits) == 1
    assert report.commits[0].message == "vega: Fixed add() so the tests pass again"
    assert "Autopilot round 2" in git(repo, "log", "-1", "--format=%B")
    assert git(repo, "rev-parse", "main") == start


@needs_sandbox
def test_still_failing_at_the_end_becomes_a_wip_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # write_guard: this scripted model overwrites a fixture file it never read; the test is about
    # the WIP commit, not the overwrite hold (tests/test_write_guard.py)
    switch_off(monkeypatch, "write_guard")
    repo = make_repo(tmp_path)
    start = git(repo, "rev-parse", "HEAD")
    script = Script([
        call("write_file", path="calc.py", content=BROKEN), text("Rewrote add(). DONE"),
    ])
    report = run_goal(repo, "add sub", max_rounds=1, agent_factory=script)

    assert report.stop_reason == "reached max rounds (1)"
    assert report.wip and not report.ok and not report.final_check.passed
    assert len(report.commits) == 1 and report.commits[0].wip
    assert report.commits[0].message.startswith("vega: WIP (check failing) ")
    assert git(repo, "status", "--porcelain") == ""  # nothing left uncommitted
    body = git(repo, "log", "-1", "--format=%B")
    assert "committed so nothing is lost" in body and "Check output (tail):" in body
    assert body.splitlines()[-1] == TRAILER
    assert (repo / "calc.py").read_text() == BROKEN
    assert git(repo, "rev-parse", "main") == start
    assert "WIP" in format_report(report)


def test_stops_when_two_rounds_change_nothing(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    script = Script([
        plan(("Investigate", "in_progress"), ("Change things", "pending")),
        call("list_files"),  # real work: a plan alone gets the act nudge first
        text("Working on it."),
        text("Still thinking."),  # the agent sends a final answer with open plan items back once
        call("read_file", path="calc.py"),
        text("Looked at calc.py."),
        text("Nothing to change yet."),
    ])
    report = run_goal(repo, "improve calc", check="none", agent_factory=script)
    assert report.stop_reason == "stuck (two rounds in a row changed no files)"
    assert report.rounds == 2 and not report.commits and not report.ok
    assert script.replies == []


def test_max_rounds_commits_each_round_and_stops(tmp_path: Path, monkeypatch) -> None:
    repo = make_repo(tmp_path)
    start = git(repo, "rev-parse", "HEAD")
    git_calls: list[list[str]] = []
    real_git = autopilot._git

    def recording_git(project, *args, identity=None):
        git_calls.append(list(args))
        return real_git(project, *args, identity=identity)

    monkeypatch.setattr(autopilot, "_git", recording_git)
    script = Script([
        plan(("Write first note", "in_progress"), ("Write second note", "pending"),
             ("Write third note", "pending")),
        call("write_file", path="notes/one.txt", content="one\n"),
        text("Working through the plan."),  # sent back once: the plan has open items
        text("Wrote the first note file."),
        call("write_file", path="notes/two.txt", content="two\n"),
        text("Working through the plan."),
        text("Wrote the second note file."),
    ])
    report = run_goal(repo, "write notes", max_rounds=2, check="none", agent_factory=script)

    assert report.stop_reason == "reached max rounds (2)" and not report.completed
    assert [c.message for c in report.commits] == [
        "vega: Wrote the first note file", "vega: Wrote the second note file",
    ]
    assert branch_commits(repo, start) == [c.hash for c in reversed(report.commits)]
    assert "Check: none configured." in git(repo, "log", "-1", "--format=%B")
    assert report.final_check is None and not report.wip
    assert git(repo, "rev-parse", "main") == start
    # Round 2's message tells the agent round 1's work was committed and to continue.
    continued = "(Vega autopilot) Round 1 finished; your changes were committed."
    assert any(m.startswith(continued) for m in script.user_messages())
    used = {args[0] for args in git_calls}
    assert used <= {"rev-parse", "status", "branch", "checkout", "add", "commit", "diff", "config"}
    assert [args for args in git_calls if args[0] == "checkout"] == [
        ["checkout", "-q", "-b", report.branch]
    ]

    text_report = format_report(report)
    assert f"git diff {start[:12]}..{report.branch}" in text_report
    assert "git checkout main" in text_report
    assert f"git branch -D {report.branch}" in text_report
    assert "reached max rounds (2)" in text_report and "needs attention" in text_report
    assert "▸ Write first note" in text_report and "· Write third note" in text_report
    assert all(line == line.rstrip() for line in text_report.splitlines())
    assert report.commits[0].hash[:8] in text_report and "2 files changed" in text_report


def test_model_failure_stops_without_commits(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    report = run_goal(repo, "anything", check="none", agent_factory=Script([]))
    assert report.stop_reason.startswith("model request failed: ")
    assert not report.commits and not report.ok
    assert git(repo, "branch", "--show-current") == report.branch


def test_uses_fallback_identity_only_when_repo_has_none(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, identity=False)
    script = Script([
        call("write_file", path="README.md", content="hello\n"),
        text("Wrote a README for the project. DONE"),
    ])
    report = run_goal(repo, "write a readme", branch="docs/readme", check="none",
                      agent_factory=script)
    assert report.branch == "docs/readme" and report.ok
    assert git(repo, "log", "-1", "--format=%an <%ae>") == "Vega autopilot <vega@localhost>"
    result = subprocess.run(["git", "config", "--get", "user.name"], cwd=repo, capture_output=True)
    assert result.returncode == 1  # nothing was written to the repository's config


# -- helpers -------------------------------------------------------------------------------------

def test_default_check_detection(tmp_path: Path) -> None:
    for name in ("tests_dir", "pyproject", "conftest", "makefile", "nothing", "makefile_no_test"):
        (tmp_path / name).mkdir()
    (tmp_path / "tests_dir" / "tests").mkdir()
    (tmp_path / "pyproject" / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tmp_path / "conftest" / "conftest.py").write_text("")
    (tmp_path / "makefile" / "Makefile").write_text("build:\n\tcc x.c\n\ntest: build\n\t./x\n")
    (tmp_path / "makefile_no_test" / "Makefile").write_text("build:\n\tcc x.c\n")
    (tmp_path / "nothing" / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    assert detect_check(tmp_path / "tests_dir") == CHECK
    assert detect_check(tmp_path / "pyproject") == CHECK
    assert detect_check(tmp_path / "conftest") == CHECK
    assert detect_check(tmp_path / "makefile") == "make test"
    assert detect_check(tmp_path / "makefile_no_test") is None
    assert detect_check(tmp_path / "nothing") is None
    assert autopilot._resolve_check(tmp_path / "tests_dir", "none") is None


def test_unrunnable_check_is_refused_before_branching(tmp_path: Path, monkeypatch) -> None:
    repo = make_repo(tmp_path)
    monkeypatch.setattr(sandbox, "available", lambda: True)
    with pytest.raises(AutopilotRefused, match="cannot run in the sandbox"):
        run_goal(repo, "x", check="rm -rf build", agent_factory=Script([]))
    monkeypatch.setattr(sandbox, "available", lambda: False)
    with pytest.raises(AutopilotRefused, match="--check none"):
        run_goal(repo, "x", agent_factory=Script([]))
    assert git(repo, "branch", "--format=%(refname:short)") == "main"


def test_slug_and_summary() -> None:
    assert slugify("Add a subtract function!") == "add-a-subtract-function"
    assert slugify("???") == "goal"
    assert len(slugify("word " * 30)) <= autopilot.MAX_SLUG
    assert summarize("**Summary**\n- Added `sub_numbers()` to calc.py. DONE", "x") == (
        "Added sub_numbers() to calc.py"
    )
    assert summarize("", "the goal itself") == "the goal itself"
    long = summarize("word " * 40, "x")
    assert len("vega: " + long) <= 72 and long.endswith("...")


def test_command_refuses_before_starting_the_server(tmp_path: Path, monkeypatch) -> None:
    def no_server() -> None:
        raise AssertionError("the model server must not be started for a refused run")

    monkeypatch.setattr(server, "ensure_running", no_server)
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(typer.Exit) as exit_info:
        autopilot.autopilot_command("x", plain, None, 8, None)
    assert exit_info.value.exit_code == 1


@pytest.mark.parametrize("completed", [True, False])
def test_command_exit_code_follows_the_report(tmp_path: Path, monkeypatch, completed) -> None:
    repo = make_repo(tmp_path)
    monkeypatch.setattr(server, "ensure_running", lambda: None)
    received: dict = {}

    def fake_run_goal(project, goal, **options):
        received.update(options, project=project, goal=goal)
        return autopilot.AutopilotReport(
            goal=goal, project=project, branch="vega/b", original_branch="main",
            start_commit="0" * 40, check_command=None, rounds=1, max_rounds=3,
            stop_reason="x", completed=completed,
        )

    monkeypatch.setattr(autopilot, "run_goal", fake_run_goal)
    with pytest.raises(typer.Exit) as exit_info:
        autopilot.autopilot_command("the goal", repo, "vega/b", 3, "none")
    assert exit_info.value.exit_code == (0 if completed else 1)
    assert received["goal"] == "the goal" and received["branch"] == "vega/b"
    assert received["max_rounds"] == 3 and received["check"] == "none"
    assert isinstance(received["on_event"], TerminalRenderer)


def test_renderer_prints_autopilot_events_as_lines() -> None:
    out = Console(record=True, width=100, force_terminal=False)
    renderer = TerminalRenderer(out)
    renderer("autopilot", {"event": "commit", "message": "autopilot: committed abc123 vega: x",
                           "hash": "abc123", "subject": "vega: x"})
    assert "autopilot: committed abc123 vega: x" in out.export_text()
