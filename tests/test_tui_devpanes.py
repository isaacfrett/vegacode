"""The developer panes in the workbench: Git, Tests, Problems and Context. Headless, with a scripted
model (no server), a real temporary git repository and a fake sandbox for test runs.

VEGA_SCREENSHOTS=<dir> saves an SVG of each pane at 120x36 and 80x24, dark and light."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from switches import switch_off
from tui_helpers import answer, calls, drive, make_app, scripted_agent, send, wait_until

from vega import sandbox, tools, tui_context, tui_git, tui_problems, tui_tests, view
from vega.tui import ContextGauge, glyphs

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

PYTEST_FAIL = """exit code: 1
stdout:
..F.F.s                                                                  [100%]
=================================== FAILURES ===================================
___________________________________ test_add ___________________________________

    def test_add():
>       assert add(2, 2) == 4
E       assert 0 == 4
E        +  where 0 = add(2, 2)

tests/test_calc.py:5: AssertionError
______________________________ TestDiv.test_zero _______________________________

self = <test_calc.TestDiv object at 0x1>

    def test_zero(self):
>       div(1, 0)
E       ZeroDivisionError: division by zero

tests/test_calc.py:12: ZeroDivisionError
=========================== short test summary info ============================
FAILED tests/test_calc.py::test_add - assert 0 == 4
FAILED tests/test_calc.py::TestDiv::test_zero - ZeroDivisionError: division by zero
==================== 2 failed, 4 passed, 1 skipped in 0.31s ====================
"""
PYTEST_PASS = "exit code: 0\nstdout:\n......s\n6 passed, 1 skipped in 0.28s\n"


@pytest.fixture(autouse=True)
def no_console_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **__: object) -> str:
        raise AssertionError("console.input called inside the TUI")

    monkeypatch.setattr(tools.console, "input", refuse)


def project_with(tmp_path: Path, **files: str) -> Path:
    """A project folder; keyword names use `__` for `/` and the last `_` for `.`
    (src__calc_py is src/calc.py)."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'p'\n")
    for name, content in files.items():
        stem, _, suffix = name.replace("__", "/").rpartition("_")
        path = project / f"{stem}.{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return project


def shoot(app, name: str) -> None:
    folder = os.environ.get("VEGA_SCREENSHOTS")
    if folder:
        app.save_screenshot(f"{name}.svg", path=folder)


def use_view(left: tuple[str, ...], right: tuple[str, ...] = ("preview",), **options) -> None:
    view.save(view.View(left=left, right=right, **options))


def git(project: Path, *args: str) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.com",
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
    subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True, env=env)


def repo(tmp_path: Path) -> Path:
    """A repository with an upstream one commit behind, a staged, an unstaged and an untracked
    file, and a renamed one."""
    project = project_with(tmp_path, src__calc_py="def add(a, b):\n    return a - b\n",
                           src__util_py="X = 1\n", README_md="calc\n")
    git(project, "init", "-q", "-b", "main")
    git(project, "add", "-A")
    git(project, "commit", "-q", "-m", "Start the calculator")
    remote = tmp_path / "remote.git"
    git(project, "clone", "-q", "--bare", str(project), str(remote))
    git(project, "remote", "add", "origin", str(remote))
    git(project, "fetch", "-q", "origin")
    git(project, "branch", "-q", "-u", "origin/main")
    (project / "src" / "util.py").write_text("X = 2\n")
    git(project, "commit", "-q", "-am", "Raise X to two so the totals add up in every report")
    (project / "src" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    git(project, "add", "src/calc.py")
    (project / "README.md").write_text("calc\n\nAdds numbers.\n")
    (project / "notes.txt").write_text("todo\n")
    return project


def options_text(pane) -> str:
    return "\n".join(str(pane.get_option_at_index(i).prompt) for i in range(pane.option_count))


def index_of(pane, needle: str) -> int:
    return next(i for i in range(pane.option_count)
                if needle in str(pane.get_option_at_index(i).prompt))


# -- git ------------------------------------------------------------------------------------------

def test_parse_status_reads_branch_counts_renames_and_conflicts() -> None:
    records = [  # `git status --porcelain=v2 --branch -z`: NUL-separated records
        "# branch.oid 1234567890abcdef", "# branch.head main", "# branch.upstream origin/main",
        "# branch.ab +2 -1",
        "1 M. N... 100644 100644 100644 aaa bbb src/calc.py",
        "1 .M N... 100644 100644 100644 aaa bbb README.md",
        "2 R. N... 100644 100644 100644 aaa bbb R100 src/new name.py", "src/old.py",
        "u UU N... 100644 100644 100644 100644 a b c conflict.py",
        "? notes.txt", ""]
    raw = "\0".join(records)
    state = tui_git.GitState()
    tui_git.parse_status(raw, state)
    assert (state.branch, state.upstream, state.ahead, state.behind) == ("main", "origin/main", 2, 1)
    assert [(f.xy, f.path, f.orig) for f in state.files] == [
        ("M.", "src/calc.py", ""), (".M", "README.md", ""), ("R.", "src/new name.py", "src/old.py"),
        ("UU", "conflict.py", ""), ("??", "notes.txt", "")]
    assert state.counts() == {"staged": 2, "unstaged": 1, "new": 1, "conflict": 1, "vega": 0}
    assert tui_git.age(30) == "now" and tui_git.age(7200) == "2h" and tui_git.age(3 * 86400) == "3d"


def test_vegas_own_files_are_not_counted_as_the_users_changes(tmp_path: Path) -> None:
    """project_context (default on) drafts VEGA.md and adds Vega's block to .gitignore at session
    start. Those are Vega's changes, not the user's: counted apart until the user edits them."""
    from vega import project_context

    project = repo(tmp_path)
    project_context.start_session(project)
    state = tui_git.read_state(project)
    vega = sorted(f.path for f in state.files if f.vega)
    assert vega == [".gitignore", "VEGA.md"]
    assert state.counts() == {"staged": 1, "unstaged": 1, "new": 1, "conflict": 0, "vega": 2}
    assert tui_git.describe(next(f for f in state.files if f.path == "VEGA.md")) == \
        "written by Vega (untracked)"
    # the user's own edit makes the file theirs again
    (project / "VEGA.md").write_text((project / "VEGA.md").read_text() + "\nMy note.\n")
    (project / ".gitignore").write_text("dist/\n" + (project / ".gitignore").read_text())
    state = tui_git.read_state(project)
    assert not any(f.vega for f in state.files)
    assert state.counts()["new"] == 3
    # a tracked .gitignore whose only change is Vega's block stays Vega's
    git(project, "checkout", "-q", "--", ".")
    (project / ".gitignore").write_text("build/\n")
    git(project, "add", ".gitignore")
    git(project, "commit", "-q", "-m", "ignore build")
    gitignore_text = (project / ".gitignore").read_text()
    from vega import gitignore

    gitignore.write_block(project, [".vega/"])
    assert (project / ".gitignore").read_text() != gitignore_text
    state = tui_git.read_state(project)
    assert [f.path for f in state.files if f.vega] == [".gitignore"]


def test_not_a_repository_is_said_calmly(tmp_path: Path) -> None:
    project = project_with(tmp_path)
    state = tui_git.read_state(project)
    assert state.repo is False or shutil.which("git") is None
    use_view(("git",))
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        pane = app.query_one("#wb-git")
        await wait_until(pilot, lambda: "git" in options_text(pane).lower()
                         and "reading" not in options_text(pane))
        assert "Not a git repository" in options_text(pane) or "not installed" in \
            options_text(pane)

    drive(app, script, size=(120, 36))


@needs_git
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_git_pane_rows_and_diff_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        theme: str) -> None:
    monkeypatch.setenv("VEGA_THEME", theme)
    project = repo(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        pane = app.query_one("#wb-git")
        await wait_until(pilot, lambda: "Start the calculator" in options_text(pane))
        text = options_text(pane)
        lines = text.splitlines()
        assert lines[0].startswith("main ↑1") and "origin/main" in lines[0]
        # project_context (default on) drafted VEGA.md and added Vega's .gitignore block at session
        # start: they are Vega's files, counted apart and listed after the user's changes
        assert lines[1] == "1 staged  1 unstaged  1 new" and lines[2] == "2 by Vega"  # no room
        assert "M  src/calc.py" in text and " M README.md" in text and "?? notes.txt" in text
        assert text.index("M  src/calc.py") < text.index(" M README.md") < text.index("?? notes")
        assert text.index("?? notes") < text.index("?? .gitignore  by Vega") \
            < text.index("?? VEGA.md  by Vega")
        assert "recent commits" in text
        commit = next(line for line in lines if "Raise X" in line)
        assert len(commit) <= pane.content_region.width and commit.endswith("…")
        assert all(len(line) <= pane.content_region.width for line in lines)
        shoot(app, f"git-{theme}-120x36")
        pane.focus()
        await pilot.pause()
        pane.highlighted = index_of(pane, "src/calc.py")
        await pilot.press("enter")
        await wait_until(pilot, lambda: "  2 +     return a + b" in
                         str(app.query_one("#preview-body").render()))
        title = str(app.query_one("#preview").border_title)
        assert "src/calc.py" in title and "staged" in title
        shoot(app, f"git-diff-{theme}-120x36")
        pane.highlighted = index_of(pane, "notes.txt")
        await pilot.press("enter")
        await wait_until(pilot, lambda: "todo" in str(app.query_one("#preview-body").render()))
        pane.highlighted = index_of(pane, "Start the calculator")
        await pilot.press("enter")
        await wait_until(pilot, lambda: "README.md" in str(app.query_one("#preview-body").render())
                         and "Start the calculator" in str(app.query_one("#preview").border_title))
        # a change made outside Vega shows on the next read (r, focus, or after a turn)
        (project / "src" / "util.py").write_text("X = 3\n")
        await pilot.press("r")
        await wait_until(pilot, lambda: " M src/util.py" in options_text(pane))
        assert "2 unstaged" in options_text(pane)

    drive(app, script, size=(120, 36))


@needs_git
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_git_pane_at_80_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                theme: str) -> None:
    monkeypatch.setenv("VEGA_THEME", theme)
    project = repo(tmp_path)
    use_view(("git", "changes"), (), left_width=30)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        assert not app.narrow
        pane = app.query_one("#wb-git")
        await wait_until(pilot, lambda: "Start the calculator" in options_text(pane))
        assert all(len(line) <= pane.content_region.width
                   for line in options_text(pane).splitlines())
        shoot(app, f"git-{theme}-80x24")

    drive(app, script, size=(80, 24))


# -- tests ----------------------------------------------------------------------------------------

def test_parse_pytest_go_cargo_and_jest() -> None:
    run = tui_tests.parse("pytest -q", PYTEST_FAIL, turn=2, wall=1.5)
    assert run.counts == {"failed": 2, "passed": 4, "skipped": 1}
    assert run.duration == pytest.approx(0.31) and run.exit_code == 1 and not run.ok
    assert [(f.test, f.message) for f in run.failures] == [
        ("tests/test_calc.py::test_add", "assert 0 == 4"),
        ("tests/test_calc.py::TestDiv::test_zero", "ZeroDivisionError: division by zero")]
    assert "E       ZeroDivisionError" in tui_tests.section(run, run.failures[1].test)
    assert "assert 0 == 4" not in tui_tests.section(run, run.failures[1].test)
    go = tui_tests.parse("go test ./...", "exit code: 1\n=== RUN   TestA\n--- FAIL: TestA (0.00s)\n"
                         "    a_test.go:9: got 1\n--- PASS: TestB (0.00s)\nFAIL\tex/a\t0.012s\n")
    assert go.counts == {"failed": 1, "passed": 1, "skipped": 0}
    assert [f.test for f in go.failures] == ["TestA"]
    assert "got 1" in tui_tests.section(go, "TestA")
    cargo = tui_tests.parse("cargo test", "exit code: 101\ntest t::a ... FAILED\ntest t::b ... ok\n"
                            "test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; "
                            "0 filtered out; finished in 0.01s\n")
    assert cargo.counts == {"passed": 1, "failed": 1, "skipped": 0}
    assert [f.test for f in cargo.failures] == ["t::a"]
    jest = tui_tests.parse("npm test", "exit code: 1\nTests:       1 failed, 3 passed, 4 total\n")
    assert jest.counts == {"failed": 1, "passed": 3}
    other = tui_tests.parse("make test", "exit code: 2\nmake: *** [test] Error 2\n")
    assert other.counts == {} and not other.ok
    assert tui_tests.verdict_line(other, {"success": "", "error": "", "fg": "", "muted": ""},
                                  glyphs()).plain.endswith("exit code 2")
    for command, expected in (("pytest -q", "pytest"), ("python -m pytest tests", "pytest"),
                              ("npm test", "npm"), ("npm run test", "npm"),
                              ("go test ./...", "go"), ("cargo test", "cargo"),
                              ("FOO=1 pytest", "pytest"), ("ls -la", None),
                              ("python script.py", None), ("npm install", None)):
        assert tui_tests.runner(command) == expected, command


def run_tests_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                  outputs: list[str]) -> tuple:
    project = project_with(tmp_path, src__calc_py="def add(a, b):\n    return a - b\n")

    def fake_check(command: str, project: Path, **_: object) -> str:
        return outputs.pop(0) if outputs else PYTEST_PASS

    monkeypatch.setattr(sandbox, "run_check", fake_check)
    agent, _ = scripted_agent(project, [
        calls(("run_check", {"command": "pytest -q"}), prompt_tokens=3000),
        answer("Two tests fail: `add` subtracts, and `div` divides by zero.", prompt_tokens=3400),
    ])
    return project, agent


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_tests_pane_counts_failures_preview_and_rerun(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch,
                                                      theme: str) -> None:
    monkeypatch.setenv("VEGA_THEME", theme)
    project, agent = run_tests_session(tmp_path, monkeypatch, [PYTEST_FAIL, PYTEST_PASS])
    use_view(("changes", "tests"))
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        pane = app.query_one("#wb-tests")
        assert "No test run yet" in options_text(pane)
        await send(pilot, "run the tests")
        text = options_text(pane)
        assert text.splitlines()[0] == "pytest -q"
        assert "✗ 2 fail 4 pass 1 skip" in text  # 28 cells: the short form
        assert "0.3 s  turn 1" in text
        assert "✗ test_calc.py::test_add" in text and "assert 0 == 4" in text
        assert "TestDiv::test_zero" in text
        assert "2" in str(pane.border_title)
        shoot(app, f"tests-{theme}-120x36")
        pane.focus()
        await pilot.pause()
        pane.highlighted = index_of(pane, "TestDiv::test_zero")
        await pilot.press("enter")
        await pilot.pause()
        body = str(app.query_one("#preview-body").render())
        assert "ZeroDivisionError: division by zero" in body and "def test_zero" in body
        assert "assert add(2, 2)" not in body  # only this test's section
        shoot(app, f"tests-failure-{theme}-120x36")
        await pilot.press("r")
        await wait_until(pilot, lambda: "6 passed" in options_text(pane))
        assert "run from this pane" in options_text(pane)
        assert str(pane.border_title) == "Tests"

    drive(app, script, size=(120, 36))


def test_rerun_waits_for_the_turn_and_esc_stops_it(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    project, agent = run_tests_session(tmp_path, monkeypatch, [PYTEST_FAIL])
    use_view(("tests",))
    started = threading.Event()

    def slow_check(command: str, project: Path, stop: threading.Event | None = None,
                   **_: object) -> str:
        started.set()
        assert stop is not None and stop.wait(10)
        return sandbox.STOPPED_CHECK

    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "run the tests")
        monkeypatch.setattr(sandbox, "run_check", slow_check)
        app.working = True
        app.rerun_tests()
        assert app.tests_running is None  # never during a turn
        app.working = False
        pane = app.query_one("#wb-tests")
        pane.focus()
        await pilot.press("r")
        await wait_until(pilot, started.is_set)
        assert "Esc stops" in options_text(pane)
        await pilot.press("escape")
        await wait_until(pilot, lambda: app.tests_running is None)
        assert "stopped before it finished" in options_text(pane)

    drive(app, script, size=(120, 36))


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_tests_pane_at_80_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                  theme: str) -> None:
    monkeypatch.setenv("VEGA_THEME", theme)
    project, agent = run_tests_session(tmp_path, monkeypatch, [PYTEST_FAIL])
    use_view(("tests",), (), left_width=30)
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "run the tests")
        pane = app.query_one("#wb-tests")
        lines = options_text(pane).splitlines()
        assert all(len(line) <= pane.content_region.width for line in lines), lines
        assert any(line.startswith("✗ 2 fail") for line in lines)
        shoot(app, f"tests-{theme}-80x24")

    drive(app, script, size=(80, 24))


# -- problems -------------------------------------------------------------------------------------

def test_problems_from_write_results() -> None:
    syntax = tui_problems.from_write(
        "src/a.py", "Wrote src/a.py (20 characters).\nWARNING: the file now has a SyntaxError at "
        "line 3: expected ':'. Fix it.")
    assert [(p.line, p.code, p.message, p.severity) for p in syntax] == [
        (3, "syntax", "expected ':'", "error")]
    toml = tui_problems.from_write("a.toml", "Wrote a.toml.\nWARNING: the file is now invalid TOML: "
                                   "Expected '=' (at line 2, column 5). Fix it.")
    assert toml[0].code == "toml" and toml[0].severity == "error"
    lint = tui_problems.from_write(
        "src/a.py", "Edited src/a.py.\nSyntax check: OK.\nLint findings (advisory):\n"
        "1:8 F401 [*] `os` imported but unused\n4:5 F841 Local variable `x` is assigned to but "
        "never used")
    assert [(p.line, p.column, p.code, p.severity) for p in lint] == [
        (1, 8, "F401", "warning"), (4, 5, "F841", "warning")]
    assert tui_problems.from_write("src/a.py", "Edited src/a.py.\nSyntax check: OK.\nLint: OK.") \
        == []


def problems_session(tmp_path: Path) -> tuple:
    """Turn 1 writes src/div.py still broken (it was broken before, so the parse guard lets the
    write through) and src/util.py with an unused import; the model's next call is held until
    `release` is set, then it fixes div.py."""
    project = project_with(tmp_path, src__div_py="def div(a, b)\n    pass\n")  # broken already
    broken = "import os\n\n\ndef div(a, b)\n    return a / b\n"
    fixed = "def div(a, b):\n    return a / b\n"
    release = threading.Event()
    count = [0]

    def gate() -> None:
        count[0] += 1
        if count[0] == 2:
            assert release.wait(20)

    agent, _ = scripted_agent(project, [
        calls(("write_file", {"path": "src/div.py", "content": broken}),
              ("write_file", {"path": "src/util.py", "content": "import sys\n\nX = 1\n"}),
              prompt_tokens=3000),
        calls(("write_file", {"path": "src/div.py", "content": fixed}), prompt_tokens=3500),
        answer("Added `div` in src/div.py.", prompt_tokens=3700),
    ], gate=gate)
    return project, agent, release


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_problems_pane_lists_jumps_and_clears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              theme: str) -> None:
    # a short request naming nothing: clarify_model would ask the scripted model first
    switch_off(monkeypatch, "clarify_first")
    # write_guard: problems_session overwrites src/div.py without reading it
    switch_off(monkeypatch, "write_guard")
    monkeypatch.setenv("VEGA_THEME", theme)
    project, agent, release = problems_session(tmp_path)
    use_view(("changes", "problems"))
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        pane = app.query_one("#wb-problems")
        assert "No problems in files Vega wrote." in options_text(pane)
        await send(pilot, "add div", wait=False)
        await wait_until(pilot, lambda: "syntax" in options_text(pane))
        text = options_text(pane)
        assert "src/div.py" in text and "   4 syntax expected ':'" in text
        assert text.index("src/div.py") < text.index("syntax")
        assert f"]{len(app.problem_rows)}[" in str(pane.border_title)  # one count, error colour
        lines = text.splitlines()
        assert all(len(line) <= pane.content_region.width for line in lines), lines
        shoot(app, f"problems-{theme}-120x36")
        pane.focus()
        await pilot.pause()
        pane.highlighted = index_of(pane, "syntax")
        await pilot.press("enter")
        await pilot.pause()
        body = str(app.query_one("#preview-body").render())
        assert "  4 › def div(a, b)" in body and "expected ':'" in body
        assert "src/div.py:4" in str(app.query_one("#preview").border_title)
        shoot(app, f"problems-line-{theme}-120x36")
        release.set()
        await wait_until(pilot, lambda: not app.working)
        await pilot.pause()
        text = options_text(pane)
        assert "src/div.py" not in text  # fixed: cleared
        assert "syntax" not in text

    drive(app, script, size=(120, 36))


@pytest.mark.skipif(sandbox.find_python_tool(Path.cwd(), "ruff") is None,
                    reason="ruff is not installed")
def test_l_runs_ruff_on_session_files(tmp_path: Path) -> None:
    project = project_with(tmp_path, src__calc_py="def add(a, b):\n    return a + b\n")
    agent, _ = scripted_agent(project, [
        calls(("write_file", {"path": "src/calc.py",
                              "content": "import sys\n\n\ndef add(a, b):\n    return a + b\n"}),
              prompt_tokens=3000),
        answer("Done.", prompt_tokens=3200),
    ])
    use_view(("changes", "problems"))
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "edit")
        pane = app.query_one("#wb-problems")
        pane.focus()
        await pilot.press("l")
        await wait_until(pilot, lambda: not app.lint_running and app.lint_note is not None,
                         seconds=30)
        text = options_text(pane)
        assert "1 found" in text, text
        assert "F401" in text

    drive(app, script, size=(120, 36))


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_problems_pane_at_80_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                     theme: str) -> None:
    # a short request naming nothing: clarify_model would ask the scripted model first
    switch_off(monkeypatch, "clarify_first")
    # write_guard: problems_session overwrites src/div.py without reading it
    switch_off(monkeypatch, "write_guard")
    monkeypatch.setenv("VEGA_THEME", theme)
    project, agent, release = problems_session(tmp_path)
    use_view(("problems",), (), left_width=30)
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "add div", wait=False)
        pane = app.query_one("#wb-problems")
        await wait_until(pilot, lambda: "syntax" in options_text(pane))
        lines = options_text(pane).splitlines()
        assert all(len(line) <= pane.content_region.width for line in lines), lines
        shoot(app, f"problems-{theme}-80x24")
        release.set()
        await wait_until(pilot, lambda: not app.working)

    drive(app, script, size=(80, 24))


# -- context --------------------------------------------------------------------------------------

def test_breakdown_splits_the_messages() -> None:
    messages = [
        {"role": "system", "content": "Rules." * 100 + "\nProject root: /p\n\nProject snapshot"
                                      + "x" * 700},
        {"role": "user", "content": "(Vega harness) Summary of earlier work\n" + "s" * 350},
        {"role": "user", "content": "fix add"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "src/calc.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "1: def add" + "y" * 340},
    ]
    parts = tui_context.breakdown(messages, 500, "")
    fixed = dict(parts.fixed)
    assert fixed["tool definitions"] == 500 and fixed["system prompt"] > 0
    assert fixed["project snapshot"] > 0
    conversation = dict(parts.conversation)
    assert conversation["summary of earlier work"] > 0 and conversation["tool results"] > 0
    assert [path for path, _ in parts.files] == ["src/calc.py"]
    compaction = tui_context.parse_compaction(
        "Context compacted: 14 older messages summarised (~11,200 → ~4,100 tokens; window 16,384).")
    assert compaction is not None and (compaction.dropped, compaction.before,
                                       compaction.after) == (14, 11200, 4100)


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("size", [(120, 36), (80, 24)], ids=["120x36", "80x24"])
def test_context_pane_matches_the_gauge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        theme: str, size: tuple[int, int]) -> None:
    # "why is add wrong" reads as a change request to the rules ("add" is a change verb; see
    # test-what-ships review 2026-10-17): clarify_model would ask the scripted model first
    switch_off(monkeypatch, "clarify_first")
    monkeypatch.setenv("VEGA_THEME", theme)
    project = project_with(tmp_path, src__calc_py="def add(a, b):\n    return a - b\n" * 20)
    agent, _ = scripted_agent(project, [
        calls(("read_file", {"path": "src/calc.py"}), prompt_tokens=3100),
        answer("`add` subtracts.", prompt_tokens=3600),
    ])
    if size[0] < 100:
        use_view(("context",), (), left_width=34)
    else:
        use_view(("changes", "context"))
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await send(pilot, "why is add wrong")
        app.last_compaction = tui_context.Compaction(0.0, 14, 11200, 4100)
        app._refresh_text_panes()
        await pilot.pause()
        body = app.query_one("#wb-context-body")
        text = str(body.render())
        gauge = app.query_one("#gauge", ContextGauge)
        percent = f"{round(100 * gauge.used / gauge.window)}%"
        assert text.splitlines()[0].endswith(percent)
        assert text.startswith(f"{gauge.used / 1000:.1f}K of 16K tokens")
        assert "tool specs" in text or "tool definitions" in text  # short name when narrow
        for part in ("system prompt", "snapshot", "conversation",
                     "tool results", "free", "files in context  1", "src/calc.py",
                     "last compaction", "14 messages summarised", "11.2K → 4.1K tokens"):
            assert part in text, text
        width = app.query_one("#wb-context").scrollable_content_region.width
        assert all(len(line) <= width for line in text.splitlines()), text
        shoot(app, f"context-{theme}-{size[0]}x{size[1]}")

    drive(app, script, size=size)


def test_all_new_panes_are_placeable() -> None:
    for name in ("git", "tests", "problems", "context"):
        assert name in view.PANES and name in view.PANE_TITLES
    assert view.View().left == ("changes", "git", "steps", "loops")
    hidden = [p for p in view.PANES if view.View().column(p) is None]
    assert {"tests", "problems", "context"} <= set(hidden)


@pytest.mark.parametrize("size", [(120, 36), (80, 24)], ids=["120x36", "80x24"])
def test_f2_lists_the_new_panes(tmp_path: Path, size: tuple[int, int]) -> None:
    project = project_with(tmp_path)
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path, layout="workbench", transparent=False)

    async def script(pilot) -> None:
        await pilot.press("f2")
        await pilot.pause()
        body = str(app.screen.query_one("#view-body").render())
        for title in ("Git", "Tests", "Problems", "Context"):
            assert f"  {title}" in body
        shoot(app, f"f2-{size[0]}x{size[1]}")
        await pilot.press("escape")

    drive(app, script, size=size)
