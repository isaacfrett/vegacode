"""Shared test setup: no test may reach a real model server."""

from __future__ import annotations

import faulthandler
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

# Colour switches change Rich and pytest output (FORCE_COLOR forces colour whatever its value,
# even "0"). Cleared before vega is imported: some Rich consoles are created at import time.
for _name in ("FORCE_COLOR", "PY_COLORS", "NO_COLOR"):
    os.environ.pop(_name, None)

# Variables that point git at a repository (`git rev-parse --local-env-vars`). A git hook runs with
# GIT_DIR set to the pushing checkout; inherited by the suite (the pre-push hook's full run,
# 2026-11-03), every `git -C <tmp repo>` in a test acted on the real repository instead: commits on
# the pushing branch, a branch created, core.bare=true and user.name/email rewritten in its config.
# Tests always name their own repository, so none of these may leak in.
GIT_REPO_ENV = ("GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CONFIG", "GIT_CONFIG_PARAMETERS",
                "GIT_CONFIG_COUNT", "GIT_OBJECT_DIRECTORY", "GIT_DIR", "GIT_WORK_TREE",
                "GIT_IMPLICIT_WORK_TREE", "GIT_GRAFT_FILE", "GIT_INDEX_FILE", "GIT_NO_REPLACE_OBJECTS",
                "GIT_REPLACE_REF_BASE", "GIT_PREFIX", "GIT_SHALLOW_FILE", "GIT_COMMON_DIR")
for _name in GIT_REPO_ENV:
    os.environ.pop(_name, None)

from vega import extensions, server

LAB_TESTS = Path(__file__).resolve().parent / "lab"


@pytest.fixture(autouse=True)
def no_server_discovery(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Iterator[None]:
    """Context-window discovery asks Ollama's /api/show; tests get the default unless they patch it."""
    monkeypatch.delenv("VEGA_NUM_CTX", raising=False)
    # Startup code (models.apply_feature_defaults) sets these; restore them after every test.
    for name in ("VEGA_FEATURES_ON", "VEGA_FEATURES_OFF", "VEGA_MODEL", "VEGA_KNOWLEDGE_DIR",
                 "VEGA_DEBUG", "VEGA_TEACH", "VEGA_SEED", "VEGA_TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(server, "context_length", lambda model=None: None)
    # First-run setup checks Ollama and the model when `vega`/`vega run` starts; CLI tests
    # are not about that (tests/test_first_run_setup.py clears this to test it).
    monkeypatch.setenv("VEGA_SKIP_SETUP", "1")
    # CI services set CI=true, and pytest then prints full, untruncated assertion diffs; the
    # pytest runs Vega starts inherit it. Tests describe a developer machine, where it is unset.
    monkeypatch.delenv("CI", raising=False)
    # These tests describe the core: an extension package, if one is installed, is switched off
    # (vega.extensions; tests/test_extensions.py covers the hook points). tests/lab/ describes the
    # core with the private lab loaded, so there it is on. Decided here, from the test's path, and
    # not by an autouse fixture in tests/lab/conftest.py: pytest 9 binds a conftest's fixtures to
    # the first Dir node collected for its folder, so `pytest tests/lab/a.py tests/b.py
    # tests/lab/c.py` gave c.py a second `Dir lab` without them and ran it with the lab off.
    # The loaded package is forgotten before and after each test (extensions.load caches it).
    if Path(request.node.path).resolve().is_relative_to(LAB_TESTS):
        monkeypatch.delenv(extensions.DISABLE_ENV, raising=False)
    else:
        monkeypatch.setenv(extensions.DISABLE_ENV, "1")
    extensions.reset()
    yield
    extensions.reset()
    # Every test runs with the shipped feature defaults ("test what ships"): VEGA_FEATURES_OFF is
    # cleared above and never set here. A test that is not about a default turns that one feature
    # off itself (tests/switches.py), with a comment saying why.
    # tests/test_shipped_defaults.py fails if this fixture turns any default off again.


@pytest.fixture(autouse=True)
def developer_tools_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """The developer tools' tests run the same from a clone or an installed package;
    tests/test_checkout_tools.py checks the refusal outside a clone."""
    monkeypatch.setattr(server, "source_checkout", lambda: True)


@pytest.fixture(autouse=True)
def code_view_for_new_users(monkeypatch: pytest.MonkeyPatch) -> None:
    """New users start in the product view (view.NEW_USER_PRESET); the older TUI tests were written
    against the code workbench, so they start there. tests/test_tui_product.py sets it back."""
    from vega import view

    monkeypatch.setattr(view, "NEW_USER_PRESET", "workbench")


@pytest.fixture(autouse=True)
def no_system_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may overwrite the real clipboard: no copy tool is found unless a test patches it."""
    from vega import clipboard

    monkeypatch.setattr(clipboard, "command", lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """No test may read or write Vega's real state: ~/.vega or the repo's eval transcripts.

    VEGA_HOME points at a scratch folder for every test, so code that reads `$VEGA_HOME or ~/.vega`
    at call time sees an empty home, never the owner's (11-03: a test read the live
    ~/.vega/night/loops.json, failed on this machine only, and the selfbuild fixture guard then
    aborted all 15 cases). The sealed set and the fixture-guard cache follow it. A test that sets its
    own VEGA_HOME (monkeypatch.setenv) runs after this and wins; one that deletes it to test the
    `~/.vega` default must also patch Path.home."""
    from vega import evals

    state = tmp_path_factory.mktemp("vega-state")
    monkeypatch.setenv("VEGA_HOME", str(state))
    monkeypatch.setenv("VEGA_SEALED_DIR", str(state / "sealed"))  # vega_lab.sealed.SEALED_ENV
    monkeypatch.delenv("VEGA_FIXTURE_GUARD_CACHE", raising=False)  # -> $VEGA_HOME/evals/
    monkeypatch.setattr(server, "STATE_DIR", state)
    monkeypatch.setattr(server, "PID_FILE", state / "server.pid")
    monkeypatch.setattr(server, "LOG_FILE", state / "server.log")
    monkeypatch.setattr(evals, "TRANSCRIPTS_DIR", tmp_path_factory.mktemp("transcripts"))


def _loopback_available() -> bool:
    import socket

    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
        return True
    except OSError:
        return False


# Inside Vega's own check sandbox (when a check runs this suite) even 127.0.0.1 cannot be bound, so
# tests that start a local fake server (tests/fake_ollama.py, fake web sites) cannot run there.
# They are reported as skipped instead of erroring, which would read as regressions. Wherever the
# probe can bind, nothing changes.
LOOPBACK = _loopback_available()
NO_LOOPBACK = "needs a loopback socket, which this sandbox denies"


# A test still running after this many seconds gets every thread's stack dumped to the terminal
# (faulthandler, standard library), so a hang names itself in the log instead of ending as a
# cancelled CI job (a hang on ubuntu, 2026-11-03). It dumps once and does not stop the test,
# unless VEGA_TEST_WATCHDOG_EXIT=1 (CI): then the run exits right after the dump, so the job fails
# at once with its log kept (a cancelled job's log can be lost). VEGA_TEST_WATCHDOG_S=0 switches
# it off; a test known to run longer carries @pytest.mark.watchdog(seconds)
# (tests/lab/test_snapshot_shape.py: ~310 s on the M4).
WATCHDOG_ENV = "VEGA_TEST_WATCHDOG_S"
WATCHDOG_EXIT_ENV = "VEGA_TEST_WATCHDOG_EXIT"
WATCHDOG_DEFAULT_S = 240.0
_watchdog: dict[str, int] = {}


def _watchdog_seconds(item: pytest.Item) -> float:
    try:
        seconds = max(0.0, float(os.environ.get(WATCHDOG_ENV, WATCHDOG_DEFAULT_S)))
    except ValueError:
        seconds = WATCHDOG_DEFAULT_S
    mark = item.get_closest_marker("watchdog")
    return max(seconds, float(mark.args[0])) if seconds and mark and mark.args else seconds


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "needs_unsandboxed(what): needs something Vega's check sandbox "
                            "denies where the denial never reaches the test as a PermissionError (a socket "
                            "bound in a child process or server thread, WebKit's services); skipped where "
                            "the loopback probe cannot bind, i.e. inside that sandbox")
    config.addinivalue_line("markers", f"watchdog(seconds): this test may run longer than the "
                            f"{WATCHDOG_DEFAULT_S:.0f} s after which its thread stacks are dumped")
    # Output capture is suspended while plugins configure, so fd 2 is the real terminal here; during
    # a test it points into pytest's capture file, which a hung (then killed) run never prints.
    try:
        _watchdog["fd"] = os.dup(sys.stderr.fileno())
    except (AttributeError, OSError, ValueError):
        pass


def pytest_unconfigure(config: pytest.Config) -> None:
    fd = _watchdog.pop("fd", None)
    if fd is not None:
        faulthandler.cancel_dump_traceback_later()
        os.close(fd)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    seconds, fd = _watchdog_seconds(item), _watchdog.get("fd")
    if not seconds or fd is None:
        yield
        return
    faulthandler.dump_traceback_later(seconds, repeat=False, file=fd,
                                      exit=os.environ.get(WATCHDOG_EXIT_ENV) == "1")
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if LOOPBACK:
        return
    for item in items:
        mark = item.get_closest_marker("needs_unsandboxed")
        if mark:
            what = mark.args[0] if mark.args else "the host"
            item.add_marker(pytest.mark.skip(reason=f"needs {what}, which this sandbox denies"))


def _denied_bind(excinfo: pytest.ExceptionInfo[BaseException]) -> bool:
    """The PermissionError came from binding a socket: in a function named bind / server_bind
    (socketserver), or at a `.bind(` call in the test's own code. socket.bind is C and has no frame
    of its own, so a test binding directly was missed: the selfbuild fixture guard found such tests
    red on the unmodified snapshot at f0b7730."""
    if any(entry.name in {"server_bind", "bind"} for entry in excinfo.traceback):
        return True
    try:
        return ".bind(" in str(excinfo.traceback[-1].statement)
    except Exception:  # no source for the last frame
        return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> object:
    outcome = yield
    report = outcome.get_result()
    excinfo = call.excinfo
    if LOOPBACK or excinfo is None or not excinfo.errisinstance(PermissionError):
        return
    if _denied_bind(excinfo):
        path, line, _ = item.location
        report.outcome = "skipped"
        report.longrepr = (path, (line or 0) + 1, f"Skipped: {NO_LOOPBACK}")


@pytest.fixture(autouse=True)
def no_ollama_readings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The chat status line asks Ollama's /api/ps for the loaded model (vega.live_stats); no test
    may reach a real server, so it answers as if none were running."""
    from vega import live_stats

    monkeypatch.setattr(live_stats, "ollama_ps", lambda: None)


@pytest.fixture(autouse=True)
def no_hardware_sensors(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hardware temperature reader builds a native tool into $VEGA_HOME on first use: no test
    builds or runs it there (its own tests build into tmp_path)."""
    monkeypatch.setenv("VEGA_NO_SENSORS", "1")
