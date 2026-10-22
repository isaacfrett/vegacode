"""Sandboxed check runner: isolation from the real project, network, and command allowlist."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vega import sandbox
from vega.tools import execute


def _sandbox_works() -> bool:
    """sandbox-exec exists and is not already inside a sandbox (nesting is refused).

    Vega's own tests can run inside the sandbox (an eval's regression check), where these must
    skip.
    """
    if not sandbox.available():
        return False
    probe = subprocess.run(["sandbox-exec", "-p", "(version 1)(allow default)", "true"],
                           capture_output=True, check=False)
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _sandbox_works(), reason="working macOS sandbox-exec required")


def check(project: Path, command: str) -> str:
    return execute("run_check", {"command": command}, project)


def test_runs_pytest_and_reports_failures(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text("from calc import add\n\ndef test_add():\n    assert add(2, 2) == 4\n")
    result = check(tmp_path, "python -m pytest -q -p no:cacheprovider")
    assert result.startswith("exit code: 1") and "assert" in result
    (tmp_path / "src" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    assert check(tmp_path, "pytest -q -p no:cacheprovider").startswith("exit code: 0")


def test_cannot_modify_project_or_write_outside(tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    (project / "keep.txt").write_text("original")
    outside = tmp_path / "outside.txt"
    script = (
        "import pathlib; pathlib.Path('keep.txt').write_text('changed'); "
        f"pathlib.Path({str(outside)!r}).write_text('escaped')"
    )
    result = check(project, f'python -c "{script}"')
    assert "Operation not permitted" in result or "PermissionError" in result
    assert (project / "keep.txt").read_text() == "original"
    assert not outside.exists()


def test_network_is_blocked(tmp_path: Path) -> None:
    script = "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)"
    result = check(tmp_path, f'python -c "{script}"')
    assert not result.startswith("exit code: 0")


def test_linked_venv_is_not_writable(tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("ok")
    script = "import pathlib; pathlib.Path('node_modules/lib.js').write_text('tampered')"
    check(tmp_path, f'python -c "{script}"')
    assert (tmp_path / "node_modules" / "lib.js").read_text() == "ok"


def test_clone_is_removed(tmp_path: Path) -> None:
    result = check(tmp_path, "python -c \"import os; print(os.getcwd())\"")
    assert result.startswith("exit code: 0")
    leftovers = [p for p in Path(sandbox.tempfile.gettempdir()).glob("vega-check-*")]
    assert not leftovers


@pytest.mark.parametrize("command", [
    "rm -rf .", "bash -c ls", "sh -c ls", "npm install left-pad", "curl example.com", "go run main.go",
])
def test_rejects_non_check_commands(tmp_path: Path, command: str) -> None:
    result = check(tmp_path, command)
    assert not result.startswith("exit code")


def test_shell_operators_do_not_chain(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("x")
    check(tmp_path, "python -c 'print(1)' && rm keep.txt")
    check(tmp_path, "python -c 'print(1)' ; touch pwned")
    assert (tmp_path / "keep.txt").exists() and not (tmp_path / "pwned").exists()


@pytest.mark.parametrize("command", [
    "ruff check .", "ruff format --check src", "mypy src", "make test", "make check",
])
def test_accepts_checker_commands(
    tmp_path: Path, command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandbox, "find_python_tool", lambda project, name: f"/env/bin/{name}")
    argv = sandbox.parse_command(command, tmp_path)
    assert isinstance(argv, list), argv
    assert Path(argv[0]).name == command.split()[0]


@pytest.mark.parametrize("command", [
    "make install", "make test install", "make -f Evil test", "npx some-random-package",
    "npx tsc", "ruff format src", "ruff clean", "node script.js",
])
def test_rejects_non_check_checker_use(tmp_path: Path, command: str) -> None:
    assert isinstance(sandbox.parse_command(command, tmp_path), str)
    assert not check(tmp_path, command).startswith("exit code")


def test_missing_tool_message_is_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "find_python_tool", lambda project, name: None)
    result = check(tmp_path, "pyright src")
    assert "`pyright` is not installed" in result and "network" in result
    result = check(tmp_path, "npx tsc --noEmit")
    assert "`npx tsc` is not installed" in result and "node_modules" in result


def test_python_tools_prefer_project_venv(tmp_path: Path) -> None:
    tool = tmp_path / ".venv" / "bin" / "mypy"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\necho project-mypy \"$@\"\n")
    tool.chmod(0o755)
    assert sandbox.parse_command("mypy src", tmp_path) == [str(tool), "src"]
    assert check(tmp_path, "mypy src") == "exit code: 0\nproject-mypy src"


def test_npx_runs_local_bin_without_download(tmp_path: Path) -> None:
    tool = tmp_path / "node_modules" / ".bin" / "eslint"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\necho local-eslint \"$@\"\n")
    tool.chmod(0o755)
    assert check(tmp_path, "npx eslint src") == "exit code: 0\nlocal-eslint src"


@pytest.mark.skipif(sandbox.find_python_tool(Path.cwd(), "ruff") is None, reason="ruff missing")
def test_ruff_check_runs_in_sandbox(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("import os\n")
    result = check(tmp_path, "ruff check --no-cache .")
    assert result.startswith("exit code: 1") and "F401" in result
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.py"]


def test_run_check_programs_lists_every_allowed_program() -> None:
    listed = " ".join(sandbox.RUN_CHECK_PROGRAMS)
    assert all(program in listed for program in sandbox.ALLOWED)


def test_stop_kills_a_running_command_at_once(tmp_path: Path) -> None:
    import threading
    import time as _time

    from vega import sandbox

    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    started = _time.monotonic()
    with pytest.raises(sandbox.StoppedRun):
        sandbox.run_stoppable(["python3", "-c", "import time; time.sleep(30)"], cwd=tmp_path,
                              timeout=60, stop=stop)
    assert _time.monotonic() - started < 3  # not the 30 s the command wanted


def test_run_stoppable_times_out_and_returns_output(tmp_path: Path) -> None:
    import subprocess

    from vega import sandbox

    done = sandbox.run_stoppable(["python3", "-c", "print('hi')"], cwd=tmp_path, timeout=10)
    assert done.returncode == 0 and done.stdout.strip() == "hi"
    with pytest.raises(subprocess.TimeoutExpired):
        sandbox.run_stoppable(["python3", "-c", "import time; time.sleep(5)"], cwd=tmp_path, timeout=0.5)


def test_run_check_refuses_shell_syntax_that_would_run_nothing(tmp_path: Path) -> None:
    # Before: `python - <<'PY' ... PY` ran nothing and reported "exit code: 0 (no output)".
    from vega.sandbox import parse_command

    for command in ("python - <<'PY'\nprint(1)\nPY", "python -", "python", "python a.py | head",
                    "pytest -q && echo ok", "python a.py > out.txt"):
        result = parse_command(command, tmp_path)
        assert isinstance(result, str) and ("without a shell" in result or "reads stdin" in result), command
    assert isinstance(parse_command('python -c "print(1)"', tmp_path), list)
    assert isinstance(parse_command("python -m pytest -q", tmp_path), list)
    assert isinstance(parse_command("python check.py", tmp_path), list)


# -- sandbox_toolchains (broad sample 2026-10-17) ----------------------------------------------------

def test_toolchain_programs_are_allowed_with_the_same_rules(tmp_path: Path) -> None:
    from vega.sandbox import parse_command

    (tmp_path / "main.js").write_text("console.log(1)\n")
    (tmp_path / "stats.R").write_text("cat(1)\n")
    for command in ("node main.js", "Rscript stats.R", "javac Main.java", "java -cp . Main",
                    "node --test"):
        argv = parse_command(command, tmp_path)
        if isinstance(argv, str) and "not installed" in argv:
            continue  # the program is missing on this machine, not refused
        assert isinstance(argv, list), (command, argv)
    outside = parse_command("node ../evil.js", tmp_path)
    assert isinstance(outside, str) and "inside the project" in outside
    missing = parse_command("node nope.js", tmp_path)
    assert isinstance(missing, str) and "does not exist" in missing
    assert isinstance(parse_command("node -e 1", tmp_path), str)  # still only --test or a script
    assert isinstance(parse_command("javac A.java | tee x", tmp_path), str)  # no shell, as before


def test_toolchains_switch_off(tmp_path: Path, monkeypatch) -> None:
    from vega import features
    from vega.sandbox import parse_command

    monkeypatch.setenv(features.ENV, "sandbox_toolchains")
    (tmp_path / "main.js").write_text("console.log(1)\n")
    assert parse_command("node main.js", tmp_path) == "`node` is only allowed with: --test."
    assert "only runs test/check programs" in parse_command("javac A.java", tmp_path)


def test_go_caches_are_inside_the_clone(tmp_path: Path) -> None:
    env = sandbox.toolchain_env(tmp_path / ".vega-tmp")
    for name in ("GOCACHE", "GOPATH", "GOMODCACHE"):
        assert env[name].startswith(str(tmp_path / ".vega-tmp")), name
    assert env["GOPROXY"] == "off" and env["GOTOOLCHAIN"] == "local"


@pytest.mark.skipif(sandbox.shutil.which("go") is None, reason="needs Go")
def test_go_test_runs_in_the_sandbox(tmp_path: Path) -> None:
    """Before: 'failed to initialize build cache at ~/Library/Caches/go-build: operation not
    permitted' for every `go test`."""
    (tmp_path / "go.mod").write_text("module example.com/adder\n\ngo 1.21\n")
    (tmp_path / "adder.go").write_text("package adder\n\nfunc Add(a, b int) int { return a + b }\n")
    (tmp_path / "adder_test.go").write_text(
        'package adder\n\nimport "testing"\n\nfunc TestAdd(t *testing.T) {\n'
        '\tif Add(2, 3) != 5 {\n\t\tt.Fatal("bad")\n\t}\n}\n')
    result = sandbox.run_check("go test ./...", tmp_path, timeout=120)
    assert result.startswith("exit code: 0"), result
    assert "ok" in result and "build cache" not in result
    assert not any(p.name.startswith(".vega-tmp") for p in tmp_path.iterdir())


@pytest.mark.skipif(sandbox.shutil.which("node") is None, reason="needs node")
def test_node_script_runs_in_the_sandbox(tmp_path: Path) -> None:
    (tmp_path / "check.js").write_text("console.log('sum', 2 + 3)\n")
    result = sandbox.run_check("node check.js", tmp_path)
    assert result.startswith("exit code: 0") and "sum 5" in result


def test_clone_with_read_only_dirs_is_removed(tmp_path: Path) -> None:
    workdir = tmp_path / "vega-check-x"
    locked = workdir / "gopath" / "pkg" / "mod" / "m@v1"
    locked.mkdir(parents=True)
    (locked / "a.go").write_text("package m\n")
    (locked / "a.go").chmod(0o444)
    locked.chmod(0o555)
    sandbox._remove(workdir)
    assert not workdir.exists()


def test_vega_settings_do_not_reach_the_check(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """An eval's pinned VEGA_SEED must not reach the tests a check runs."""
    (tmp_path / "test_env.py").write_text(
        "import os\n\ndef test_env():\n"
        "    assert 'VEGA_SEED' not in os.environ and os.environ['OTHER_SETTING'] == '1'\n")
    monkeypatch.setenv("VEGA_SEED", "7")
    monkeypatch.setenv("OTHER_SETTING", "1")
    assert check(tmp_path, "python -m pytest -q -p no:cacheprovider").startswith("exit code: 0")
    monkeypatch.setenv("VEGA_FEATURES_OFF", "sandbox_env_scrub")
    assert check(tmp_path, "python -m pytest -q -p no:cacheprovider").startswith("exit code: 1")
