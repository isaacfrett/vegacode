"""Loops: definitions, status templates, the inbox, scheduling decisions, the commit trigger and
Bug Watcher's reproduction rule, with scripted fake models and temporary Git repositories.

No test calls a real model or touches ~/.vega (conftest points server.STATE_DIR at a temp dir).
"""

from __future__ import annotations

import plistlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from vega import sandbox, server
from vega.loops import bug_watcher, definitions, idle, inbox, scheduler, state, status
from vega.loops.idle import IdleCheck, IdleProbe
from vega.resources import Snapshot

needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="needs macOS sandbox-exec")
CALC = "def add(a, b):\n    return a + b\n"
BROKEN = "def add(a, b):\n    return a - b\n"
TEST = "from calc import add\n\n\ndef test_add_zero():\n    assert add(2, 0) == 2\n"
IDLE = IdleCheck(ok=True, screen_locked=True)
BUSY = IdleCheck(ok=False, screen_locked=False, reasons=["screen not locked"])


@pytest.fixture(autouse=True)
def isolated_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "global.gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL",
                 "GIT_DIR", "GIT_WORK_TREE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(bug_watcher, "disk_free_gb", lambda: 100.0)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()


def make_repo(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    repo = tmp_path / "proj"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "Test User")
    git(repo, "config", "user.email", "test@example.com")
    return commit(repo, files or {"calc.py": CALC, "tests/test_calc.py": TEST}, "initial")


def commit(repo: Path, files: dict[str, str], message: str) -> Path:
    for name, content in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(content)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return repo


def loop_for(repo: Path, **overrides) -> definitions.LoopDef:
    data = {"name": "Bug Watcher", "project": str(repo), "trigger": {"on": "commit", "branch": "main"},
            "skill": "diff-bug", "permissions": ["read_project", "run_tests"],
            "budget": {"minutes_per_run": 5, "runs_per_day": 20}, "status": "found",
            "route": [{"when": {"evidence": "failing_test"}, "to": "Software Engineer"}]}
    data.update(overrides)
    return definitions.parse(data)


def write_loop(folder: Path, repo: Path, name: str = "Bug Watcher", extra: str = "") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{definitions.slugify(name)}.toml"
    path.write_text(f'name = "{name}"\nproject = "{repo}"\ntrigger = {{ on = "commit" }}\n'
                    f'skill = "diff-bug"\npermissions = ["read_project", "run_tests"]\n{extra}')
    return path


def fake_agent(replies: list[str], calls: list[dict]):
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=replies.pop(0)))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return lambda project: SimpleNamespace(client=client, model="fake", sampling={})


def worktrees(repo: Path) -> list[str]:
    return [line for line in git(repo, "worktree", "list", "--porcelain").splitlines()
            if line.startswith("worktree ")]


# -- definitions -----------------------------------------------------------------------------------

def test_definition_parses_and_describes(tmp_path: Path) -> None:
    loop = loop_for(make_repo(tmp_path))
    assert loop.slug == "bug-watcher" and loop.trigger.kind == "commit"
    assert loop.trigger.describe() == "on commit"
    assert loop.route[0]["to"] == "Software Engineer"  # stored for phase 2
    assert definitions.Trigger("commit", branch="dev").describe() == "on commit to dev"
    nightly = definitions.parse({**_base(tmp_path), "trigger": {"every": "1d", "at": "02:00"}})
    assert nightly.trigger.describe() == "nightly 02:00"
    assert definitions.parse({**_base(tmp_path), "trigger": {"every": "6h"}}).trigger.describe() == "every 6h"
    assert definitions.parse({**_base(tmp_path), "trigger": {"on": "manual"}}).trigger.describe() == "manual"


def _base(tmp_path: Path) -> dict:
    return {"name": "X", "project": str(tmp_path), "trigger": {"on": "manual"}, "skill": "diff-bug",
            "permissions": ["read_project", "run_tests"]}


@pytest.mark.parametrize(("change", "message"), [
    ({"colour": "red"}, "unknown field"),
    ({"name": ""}, "name must be"),
    ({"project": "relative/path"}, "absolute path"),
    ({"project": "/no/such/dir/anywhere"}, "does not exist"),
    ({"trigger": {"on": "eval_run"}}, "planned (phase 3)"),
    ({"trigger": {"on": "push"}}, "trigger must be"),
    ({"trigger": "commit"}, "trigger must be a table"),
    ({"trigger": {"every": "1x"}}, "every must look like"),
    ({"trigger": {"every": "1d", "at": "25:00"}}, "HH:MM"),
    ({"trigger": {"every": "6h", "at": "02:00"}}, "at least 1d"),
    ({"skill": "mind-reader"}, "unknown skill"),
    ({"permissions": ["read_project", "run_tests", "web"]}, "does not use web"),
    ({"permissions": ["read_project"]}, "needs permission(s) run_tests"),
    ({"permissions": ["root"]}, "unknown permission"),
    ({"budget": {"minutes_per_run": 0}}, "minutes_per_run"),
    ({"budget": {"runs_per_day": 2.5}}, "runs_per_day"),
    ({"budget": {"hours": 1}}, "budget must be"),
    ({"status": "loud"}, "status must be one of"),
    ({"route": [{"when": {}}]}, "route must be"),
])
def test_definition_errors_are_clear(tmp_path: Path, change: dict, message: str) -> None:
    with pytest.raises(definitions.LoopDefinitionError, match=message.replace("(", r"\(").replace(")", r"\)")):
        definitions.parse({**_base(tmp_path), **change}, tmp_path / "x.toml")


def test_missing_field_names_it(tmp_path: Path) -> None:
    data = _base(tmp_path)
    del data["skill"]
    with pytest.raises(definitions.LoopDefinitionError, match="missing required field 'skill'"):
        definitions.parse(data)


def test_load_all_reports_bad_files_and_keeps_good_ones(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    folder = definitions.user_dir()
    write_loop(folder, repo)
    (folder / "broken.toml").write_text("name = [unclosed\n")
    write_loop(definitions.project_dir(repo), repo)  # same name in the project folder
    loops, errors = definitions.load_all([repo])
    assert [loop.name for loop in loops] == ["Bug Watcher"]
    assert any("broken.toml: not valid TOML" in e for e in errors)
    assert any("already defined" in e for e in errors)
    assert definitions.find("bug watcher", [repo]).name == "Bug Watcher"
    with pytest.raises(LookupError, match="No loop named"):
        definitions.find("Nope", [repo])


def test_default_bug_watcher_is_created_once(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    created = definitions.ensure_default(repo)
    assert created is not None and created.parent == server.STATE_DIR / "loops"
    loop = definitions.load_file(created)
    assert (loop.name, loop.skill, loop.project) == ("Bug Watcher", "diff-bug", repo.resolve())
    assert definitions.ensure_default(repo) is None


def test_schedule_next_run() -> None:
    nightly = definitions.Trigger("schedule", every_seconds=86400, at="02:00")
    now = datetime(2026, 10, 15, 1, 30)
    assert nightly.next_run(None, now) == datetime(2026, 10, 15, 2, 0)
    assert nightly.next_run(None, datetime(2026, 10, 15, 3, 0)) == datetime(2026, 10, 16, 2, 0)
    ran = datetime(2026, 10, 15, 2, 41)
    assert nightly.next_run(ran, datetime(2026, 10, 15, 9, 0)) == datetime(2026, 10, 16, 2, 0)
    hourly = definitions.Trigger("schedule", every_seconds=3600)
    assert hourly.next_run(ran, now) == datetime(2026, 10, 15, 3, 41)
    assert definitions.Trigger("commit").next_run(ran, now) is None


# -- status templates and rows ---------------------------------------------------------------------

def test_status_templates() -> None:
    assert status.status_text("found", 3) == "3 found"
    assert status.status_text("found", 0) == "all clear"
    assert status.status_text("found", 0, ran=False) == "not run yet"
    assert status.status_text("approvals", 2) == "2 approvals"
    assert status.status_text("approvals", 1) == "1 approval"
    assert status.status_text("findings", 4) == "new findings"
    assert status.status_text("regressions", 1) == "1 regression"
    assert status.status_text("all_clear", 0) == "all clear"
    assert status.status_text("found", 3, paused=True) == "paused"
    assert status.status_text("found", 3, failing=True) == "failing"
    with pytest.raises(ValueError, match="unknown status kind"):
        status.status_text("loud", 1)


def _finding(repo: Path, title: str = "tests fail", output: str = "FAILED tests/test_calc.py::test_add",
             path: str = "calc.py", commit_id: str = "abc123") -> dict:
    return inbox.make_finding(loop="Bug Watcher", project=repo, commit=commit_id, kind="bug",
                              title=title, where=[{"path": path, "line": 2}],
                              evidence={"type": "failing_test", "path": "tests/test_calc.py",
                                        "output": output, "fails_at": commit_id})


def test_rows_follow_inbox_pause_and_errors(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    loop = loop_for(repo)
    row = status.row(loop)
    assert (row.state, row.status, row.needs_you, row.symbol) == ("idle", "not run yet", 0, "○")
    inbox.add(_finding(repo, output="FAILED tests/test_calc.py::test_a"))
    inbox.add(_finding(repo, output="FAILED tests/test_calc.py::test_b", path="other.py"))
    st = state.LoopState(last_run=datetime(2026, 10, 15, 9, 12).isoformat())
    row = status.row(loop, st)
    assert (row.state, row.status, row.needs_you, row.informational) == ("active", "2 found", 2, False)
    assert row.symbol == "●" and row.trigger == "on commit" and row.last_run.hour == 9
    assert row.when().startswith("on commit · last ")
    first = inbox.items(loop.name)[0]
    inbox.transition(first["id"], "rejected", "not a bug")
    assert status.row(loop, st).status == "1 found"
    state.set_paused(loop.name, True)
    paused = status.row(loop, st)
    assert (paused.state, paused.status, paused.symbol) == ("paused", "paused", "‖")
    state.set_paused(loop.name, False)
    failing = status.row(loop, state.LoopState(error="git exploded"))
    assert (failing.state, failing.status, failing.symbol) == ("failing", "failing", "!")
    findings_loop = loop_for(repo, status="findings")
    info = status.row(findings_loop, st)
    assert (info.status, info.needs_you, info.informational) == ("new findings", 0, True)


def test_pause_all_and_resume_all() -> None:
    state.set_paused(None, True)
    assert state.is_paused("Anything")
    state.set_paused("Bug Watcher", False)
    assert state.is_paused("Bug Watcher")  # the global pause still holds
    state.set_paused(None, False)
    assert not state.is_paused("Bug Watcher")


def test_header_counts_needs_you() -> None:
    probe = IdleProbe(locked=lambda: False, battery=lambda: False, busy=lambda: "",
                      snapshot=lambda: Snapshot(None, None, 100.0, 0.0))
    row = SimpleNamespace(needs_you=2)
    head = status.header([row, SimpleNamespace(needs_you=1)], probe)
    assert head.needs_you == 3 and not head.model_allowed and not head.daemon_running
    assert head.reasons == ("screen not locked",)
    assert head.text() == "idle-only · screen unlocked · daemon stopped"


# -- inbox -----------------------------------------------------------------------------------------

def test_inbox_dedupes_open_and_rejected_findings(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    first, new = inbox.add(_finding(repo, output="FAILED tests/test_calc.py::test_add - in 0.12s"))
    assert new and inbox.state_of(first) == "new"
    again, new = inbox.add(_finding(repo, output="FAILED tests/test_calc.py::test_add - in 3.4s",
                                    commit_id="def456"))
    assert not new and again["id"] == first["id"]
    assert [s["commit"] for s in inbox.get(first["id"])["seen_again"]] == ["def456"]
    _, new = inbox.add(_finding(repo, output="FAILED tests/test_calc.py::test_sub"))
    assert new  # a different failing test is a different finding
    inbox.transition(first["id"], "rejected")
    _, new = inbox.add(_finding(repo, output="FAILED tests/test_calc.py::test_add"))
    assert not new  # rejected stays rejected
    assert len(inbox.items()) == 2 and len(inbox.items(open_only=True)) == 1


def test_inbox_files_a_recurrence_after_accept(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    first, _ = inbox.add(_finding(repo))
    for step in ("routed", "in_progress", "proposed", "accepted"):
        inbox.transition(first["id"], step)
    _, new = inbox.add(_finding(repo))
    assert new


def test_inbox_transitions_are_enforced(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    finding, _ = inbox.add(_finding(repo))
    with pytest.raises(ValueError, match="can move to"):
        inbox.transition(finding["id"], "proposed")  # only a worker's job proposes
    with pytest.raises(ValueError, match="unknown state"):
        inbox.transition(finding["id"], "maybe")
    inbox.transition(finding["id"], "duplicate", "same as another")
    with pytest.raises(ValueError, match="none \\(finished\\)"):
        inbox.transition(finding["id"], "routed")
    history = inbox.get(finding["id"][:20])["status"]
    assert [h["state"] for h in history] == ["new", "duplicate"] and history[1]["note"]


def test_fingerprint_ignores_line_numbers_and_timings() -> None:
    a = inbox.fingerprint({"type": "lint", "output": "calc.py:3:1: F401 `os` imported but unused"})
    b = inbox.fingerprint({"type": "lint", "output": "calc.py:9:1: F401 `os` imported but unused"})
    c = inbox.fingerprint({"type": "lint", "output": "calc.py:9:1: F841 local variable unused"})
    assert a == b != c


def test_finding_schema(tmp_path: Path) -> None:
    finding = _finding(make_repo(tmp_path))
    assert set(finding) >= {"id", "loop", "project", "commit", "kind", "title", "where",
                            "evidence", "confidence", "route", "status", "debug_log"}
    assert set(finding["evidence"]) >= {"type", "path", "output", "fails_at", "fingerprint"}
    with pytest.raises(ValueError, match="evidence type"):
        inbox.make_finding(loop="x", project=".", commit="c", kind="bug", title="t", where=[],
                           evidence={"type": "hunch"})


# -- idle ------------------------------------------------------------------------------------------

def _ioreg(sessions: list[dict], console_locked: bool = False) -> bytes:
    return plistlib.dumps({"IOConsoleLocked": console_locked, "IOConsoleUsers": sessions})


def test_screen_lock_parser() -> None:
    unlocked = {"kCGSSessionUserIDKey": 501, "kCGSSessionOnConsoleKey": True}
    locked = {**unlocked, "CGSSessionScreenIsLocked": True}
    assert idle.parse_ioreg_root(_ioreg([locked]), 501)
    assert not idle.parse_ioreg_root(_ioreg([unlocked]), 501)
    assert not idle.parse_ioreg_root(_ioreg([{**locked, "kCGSSessionUserIDKey": 502}]), 501)
    assert not idle.parse_ioreg_root(b"not a plist")  # cannot tell: not locked
    assert not idle.parse_ioreg_root(b"")


def test_screen_locked_is_false_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(idle.sys, "platform", "linux")
    assert idle.screen_locked() is False


def test_idle_probe_reasons() -> None:
    snap = {"value": Snapshot(free_pct=50, swap_used_mb=100.0, disk_free_gb=50.0, model_rss_mb=0,
                              pressure_level=1)}
    probe = IdleProbe(locked=lambda: True, battery=lambda: False, busy=lambda: "",
                      snapshot=lambda: snap["value"])
    assert probe.check().ok
    assert probe.check(paused=True).reasons == ["paused"]
    snap["value"] = Snapshot(50, 3000.0, 5.0, 0, pressure_level=2, thermal_warning="hot")
    reasons = probe.check().reasons
    assert any("memory pressure" in r for r in reasons) and any("swap grew" in r for r in reasons)
    assert any("free disk" in r for r in reasons) and any("thermal" in r for r in reasons)
    on_battery = IdleProbe(locked=lambda: True, battery=lambda: True, busy=lambda: "vega eval",
                           snapshot=lambda: Snapshot(50, 0.0, 50.0, 0, pressure_level=1))
    assert on_battery.check().reasons == ["on battery", "model busy with another Vega job (vega eval)"]
    unlocked = IdleProbe(locked=lambda: False, battery=lambda: False, busy=lambda: "",
                         snapshot=lambda: pytest.fail("no resource probe while unlocked"))
    assert unlocked.check().reasons == ["screen not locked"]


def test_other_model_work_pattern() -> None:
    pattern = idle._MODEL_COMMANDS
    assert pattern.search("/x/.venv/bin/python /Users/me/.local/bin/vega lab run goals")
    assert pattern.search("/x/bin/vega eval --suite fixes")
    assert not pattern.search("/x/bin/vega loops start --foreground")
    assert not pattern.search("vim evaluation.txt")


# -- scheduler -------------------------------------------------------------------------------------

def test_decide_runs_deterministic_always_and_model_only_when_idle(tmp_path: Path) -> None:
    loop = loop_for(make_repo(tmp_path))
    now = datetime(2026, 10, 15, 12, 0)
    fresh = state.LoopState()
    busy = scheduler.decide(loop, fresh, now=now, commits=["c1", "c2"], head="c2", idle=BUSY,
                            paused=False)
    assert (busy.commit, busy.deterministic, busy.model) == ("c1", True, False)
    ready = scheduler.decide(loop, fresh, now=now, commits=["c1"], head="c1", idle=IDLE, paused=False)
    assert (ready.deterministic, ready.model) == (True, True)
    waiting = state.LoopState(pending_model=["c0"])
    assert scheduler.decide(loop, waiting, now=now, commits=[], head="c0", idle=BUSY, paused=False) is None
    later = scheduler.decide(loop, waiting, now=now, commits=[], head="c0", idle=IDLE, paused=False)
    assert (later.commit, later.deterministic, later.model) == ("c0", False, True)
    assert scheduler.decide(loop, fresh, now=now, commits=["c1"], head="c1", idle=IDLE, paused=True) is None
    spent = state.LoopState(runs={"2026-10-15": 20})
    assert scheduler.decide(loop, spent, now=now, commits=["c1"], head="c1", idle=IDLE, paused=False) is None


def test_decide_schedule_trigger(tmp_path: Path) -> None:
    loop = loop_for(make_repo(tmp_path), trigger={"every": "1d", "at": "02:00"})
    ran = state.LoopState(last_run=datetime(2026, 10, 15, 2, 5).isoformat())
    assert scheduler.decide(loop, ran, now=datetime(2026, 10, 15, 9), commits=[], head="h",
                            idle=IDLE, paused=False) is None
    due = scheduler.decide(loop, ran, now=datetime(2026, 10, 16, 2, 0, 30), commits=[], head="h",
                           idle=BUSY, paused=False)
    assert (due.commit, due.deterministic, due.model) == ("h", True, False)


def test_commit_trigger_detects_new_commits(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    loop = loop_for(repo)
    head, new = scheduler.new_commits(loop, state.LoopState())
    assert new == [] and head == git(repo, "rev-parse", "HEAD")  # first sight: baseline
    seen = state.LoopState(last_seen_commit=head)
    assert scheduler.new_commits(loop, seen)[1] == []
    commit(repo, {"a.py": "x = 1\n"}, "one")
    first = git(repo, "rev-parse", "HEAD")
    commit(repo, {"b.py": "y = 2\n"}, "two")
    second = git(repo, "rev-parse", "HEAD")
    assert scheduler.new_commits(loop, seen) == (second, [first, second])  # oldest first
    git(repo, "checkout", "-q", "-b", "side")
    commit(repo, {"c.py": "z = 3\n"}, "side")
    assert scheduler.new_commits(loop, state.LoopState(last_seen_commit=second))[1] == []  # main only


def test_daemon_tick_defers_model_work_until_idle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_repo(tmp_path)
    write_loop(definitions.user_dir(), repo)
    calls: list[tuple[str, bool, bool]] = []

    def fake_check(loop, sha, *, deterministic=True, gate=None, agent_factory=None):
        model_ran = gate is not None and gate().ok
        calls.append((sha, deterministic, gate is not None))
        return bug_watcher.JobReport(commit=sha, model="ran" if model_ran else "not requested",
                                     model_ran=model_ran)

    monkeypatch.setattr(bug_watcher, "check_commit", fake_check)
    locked = {"value": False}
    probe = IdleProbe(locked=lambda: locked["value"], battery=lambda: False, busy=lambda: "",
                      snapshot=lambda: Snapshot(50, 0.0, 50.0, 0, pressure_level=1))
    unloads: list[bool] = []
    daemon = scheduler.Daemon(probe=probe, log=lambda text: None, projects=[],
                              unload=lambda: unloads.append(True))
    assert daemon.tick() == []  # baseline only
    commit(repo, {"calc.py": BROKEN}, "break add")
    sha = git(repo, "rev-parse", "HEAD")
    daemon.tick()
    assert calls == [(sha, True, False)]  # deterministic now, no model while unlocked
    st = state.load_state("Bug Watcher")
    assert st.pending_model == [sha] and st.last_seen_commit == sha and st.last_run
    daemon.tick()
    assert len(calls) == 1  # nothing new, still not idle
    locked["value"] = True
    daemon.tick()
    assert calls[-1] == (sha, False, True)  # model-only pass once idle
    assert state.load_state("Bug Watcher").pending_model == []
    assert unloads == [True]  # the model is unloaded once the loops go quiet
    state.set_paused(None, True)
    commit(repo, {"b.py": "y = 1\n"}, "more")
    daemon.tick()
    assert len(calls) == 2  # paused: nothing runs


def test_job_lock_and_pid_file(tmp_path: Path) -> None:
    with state.job_lock("Bug Watcher") as first:
        assert first
        with state.job_lock("Bug Watcher") as second:
            assert not second
    with state.job_lock("Bug Watcher") as again:
        assert again
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "loops"])
    try:
        state.write_json(state.pid_file(), {"pid": sleeper.pid})
        with pytest.raises(state.AlreadyRunning, match=f"pid {sleeper.pid}"):
            state.claim_daemon()
        with pytest.raises(state.AlreadyRunning):
            scheduler.start_background()
    finally:
        sleeper.kill()
        sleeper.wait()
    state.claim_daemon()  # the pid file is stale now: a new daemon may start
    assert state.daemon_pid() is not None
    state.release_daemon()
    assert not state.pid_file().exists()
    assert scheduler.stop_daemon() == "The loops daemon is not running."


def test_daemon_serve_stops_and_releases_pid(tmp_path: Path) -> None:
    daemon = scheduler.Daemon(probe=IdleProbe(locked=lambda: False, battery=lambda: False,
                                              busy=lambda: ""),
                              log=lambda text: None, projects=[], unload=lambda: None,
                              poll_seconds=0.01)
    ticks = {"n": 0}

    def tick(now=None):
        ticks["n"] += 1
        if ticks["n"] == 2:
            daemon.stopping.set()
        return []

    daemon.tick = tick
    daemon.serve()
    assert ticks["n"] == 2 and not state.pid_file().exists()


# -- Bug Watcher -----------------------------------------------------------------------------------

def test_parse_reply() -> None:
    good = "TITLE: add subtracts\n```python\nfrom calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n```"
    parsed = bug_watcher.parse_reply(good)
    assert parsed.title == "add subtracts" and "def test_add" in parsed.test
    assert bug_watcher.parse_reply("NO_BUG") == "NO_BUG"
    assert "no ```python block" in bug_watcher.parse_reply("I think it is fine")
    assert "does not parse" in bug_watcher.parse_reply("```python\ndef test_x(:\n```")
    assert "no test_ function" in bug_watcher.parse_reply("```python\nx = 1\n```")


def test_diff_locations() -> None:
    diff = ("diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n-    return a + b\n+    return a - b\n")
    assert bug_watcher.diff_locations(diff) == [{"path": "calc.py", "line": 1}]


def test_model_is_not_called_when_not_idle(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit(repo, {"calc.py": BROKEN}, "break add")
    calls: list[dict] = []
    report = bug_watcher.check_commit(loop_for(repo), "HEAD", deterministic=False,
                                      gate=lambda: BUSY, agent_factory=fake_agent([], calls))
    if sandbox.available():
        assert report.model == "deferred: screen not locked" and not report.model_ran
    else:
        assert report.model.startswith("skipped: sandbox unavailable")
    assert calls == [] and report.filed == []
    assert worktrees(repo) == [f"worktree {repo.resolve()}"]


KEEP = ("TITLE: add subtracts instead of adding\n```python\nfrom calc import add\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n```")
FAILS_BOTH = "TITLE: wrong\n```python\nfrom calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 6\n```"
PASSES_HEAD = "TITLE: zero\n```python\nfrom calc import add\n\n\ndef test_add():\n    assert add(2, 0) == 2\n```"


@needs_sandbox
def test_reproduction_rule_keeps_only_tests_that_fail_at_head_and_pass_at_parent(
        tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    commit(repo, {"calc.py": BROKEN}, "break add")
    sha, parent = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "HEAD~1")
    loop = loop_for(repo)
    calls: list[dict] = []
    kept = bug_watcher.check_commit(loop, sha, deterministic=False, gate=lambda: IDLE,
                                    agent_factory=fake_agent([KEEP], calls))
    assert kept.model.startswith("ran: reproduced") and len(kept.filed) == 1, kept
    finding = kept.filed[0]
    assert finding["confidence"] == "verified" and finding["route"] == "Software Engineer"
    assert finding["evidence"]["fails_at"] == sha[:12] and finding["evidence"]["passes_at"] == parent[:12]
    assert Path(finding["evidence"]["path"]).read_text().startswith("from calc import add")
    assert finding["where"] == [{"path": "calc.py", "line": 1}]
    assert finding["title"] == "add subtracts instead of adding"
    prompt = calls[0]["messages"][-1]["content"]
    assert "-    return a + b" in prompt and "tests/test_calc.py" not in prompt  # the diff only
    for reply, why in ((FAILS_BOTH, "also fails at the parent"), (PASSES_HEAD, "passes at the commit")):
        report = bug_watcher.check_commit(loop, sha, deterministic=False, gate=lambda: IDLE,
                                          agent_factory=fake_agent([reply], []))
        assert report.filed == [] and why in report.model, report
    no_bug = bug_watcher.check_commit(loop, sha, deterministic=False, gate=lambda: IDLE,
                                      agent_factory=fake_agent(["NO_BUG"], []))
    assert no_bug.model == "ran: the model saw no bug"
    assert len(inbox.items()) == 1
    assert worktrees(repo) == [f"worktree {repo.resolve()}"]
    assert not any((server.STATE_DIR / "loops" / "worktrees").iterdir())


@needs_sandbox
def test_deterministic_pass_reports_new_test_failures_and_lint(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": CALC, "tests/test_calc.py": TEST +
                                "\n\ndef test_add():\n    assert add(2, 3) == 5\n"})
    commit(repo, {"calc.py": "import os\n\n\n" + BROKEN}, "break add, unused import")
    report = bug_watcher.check_commit(loop_for(repo), "HEAD", deterministic=True, gate=None)
    assert report.error == "" and report.model == "not requested"
    kinds = sorted((f["evidence"]["type"], f["title"]) for f in report.filed)
    assert kinds[0][0] == "failing_test" and "1 test(s) fail" in kinds[0][1]
    assert kinds[1][0] == "lint" and "calc.py" in kinds[1][1]
    test_finding = next(f for f in report.filed if f["evidence"]["type"] == "failing_test")
    assert test_finding["evidence"]["path"] == "tests/test_calc.py::test_add"
    assert report.debug_log and Path(report.debug_log).is_file()  # every job has a debug log
    assert "VEGA_DEBUG" not in __import__("os").environ  # restored after the job
    again = bug_watcher.check_commit(loop_for(repo), "HEAD", deterministic=True, gate=None)
    assert again.filed == [] and again.duplicates == 2  # deduped, not filed twice
    assert worktrees(repo) == [f"worktree {repo.resolve()}"]


@needs_sandbox
def test_failures_already_at_the_parent_are_not_findings(tmp_path: Path) -> None:
    failing = "def test_always():\n    assert False\n"
    repo = make_repo(tmp_path, {"calc.py": CALC, "tests/test_bad.py": failing})
    commit(repo, {"notes.txt": "hello\n"}, "docs only")
    report = bug_watcher.check_commit(loop_for(repo), "HEAD", deterministic=True, gate=None)
    assert report.filed == [] and any("also fail at the parent" in n for n in report.notes)


def test_worktree_is_removed_even_when_the_job_fails(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    with pytest.raises(RuntimeError, match="boom"), bug_watcher.worktree(repo, "HEAD") as wt:
        assert (wt / "calc.py").read_text() == CALC
        raise RuntimeError("boom")
    assert not wt.exists()
    assert worktrees(repo) == [f"worktree {repo.resolve()}"]


def test_budget_exceeded_is_recorded_as_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_repo(tmp_path)

    def out_of_time(*args, **kwargs):
        raise bug_watcher.BudgetExceeded("hit the 5-minute budget")

    monkeypatch.setattr(bug_watcher, "_deterministic", out_of_time)
    report = bug_watcher.check_commit(loop_for(repo), "HEAD", deterministic=True, gate=None)
    assert report.error == "hit the 5-minute budget"
    assert worktrees(repo) == [f"worktree {repo.resolve()}"]


def test_low_disk_skips_the_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_repo(tmp_path)
    monkeypatch.setattr(bug_watcher, "disk_free_gb", lambda: 3.0)
    report = bug_watcher.check_commit(loop_for(repo), "HEAD", gate=lambda: IDLE)
    assert report.notes == ["skipped: free disk 3.0 GB < 12 GB"] and report.filed == []


# -- CLI -------------------------------------------------------------------------------------------

def test_cli_list_show_pause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vega.cli import app

    repo = make_repo(tmp_path)
    write_loop(definitions.user_dir(), repo)
    monkeypatch.setattr(idle, "screen_locked", lambda: False)
    monkeypatch.setattr(status, "IdleProbe", lambda: IdleProbe(
        locked=lambda: False, battery=lambda: False, busy=lambda: ""))
    runner = CliRunner()
    listed = runner.invoke(app, ["loops", "list"])
    assert listed.exit_code == 0 and "Bug Watcher" in listed.output and "not run yet" in listed.output
    assert runner.invoke(app, ["loops", "pause", "bug watcher"]).exit_code == 0
    assert "(paused)" in runner.invoke(app, ["loops"]).output
    assert runner.invoke(app, ["loops", "resume", "Bug Watcher"]).exit_code == 0
    inbox.add(_finding(repo))
    shown = runner.invoke(app, ["loops", "show", "Bug Watcher"])
    assert "tests fail" in shown.output and "new" in shown.output
    assert runner.invoke(app, ["loops", "show", "Nope"]).exit_code == 1
