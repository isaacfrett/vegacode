"""Shared test setup: no test may reach a real model server."""

from __future__ import annotations

import pytest

from vega import server


@pytest.fixture(autouse=True)
def no_server_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Context-window discovery asks Ollama's /api/show; tests get the default unless they patch it."""
    monkeypatch.delenv("VEGA_NUM_CTX", raising=False)
    # Startup code (models.apply_feature_defaults) sets these; restore them after every test.
    for name in ("VEGA_FEATURES_ON", "VEGA_FEATURES_OFF", "VEGA_MODEL", "VEGA_KNOWLEDGE_DIR",
                 "VEGA_DEBUG", "VEGA_TEACH", "VEGA_SEED", "VEGA_TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(server, "context_length", lambda model=None: None)
    # CI services set CI=true, and pytest then prints full, untruncated assertion diffs; the
    # pytest runs Vega starts inherit it. Tests describe a developer machine, where it is unset.
    monkeypatch.delenv("CI", raising=False)
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
    """No test may write Vega's real state: ~/.vega or the repo's eval transcripts."""
    from vega import evals

    state = tmp_path_factory.mktemp("vega-state")
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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> object:
    outcome = yield
    report = outcome.get_result()
    excinfo = call.excinfo
    if LOOPBACK or excinfo is None or not excinfo.errisinstance(PermissionError):
        return
    if any(entry.name in {"server_bind", "bind"} for entry in excinfo.traceback):
        path, line, _ = item.location
        report.outcome = "skipped"
        report.longrepr = (path, (line or 0) + 1,
                           "Skipped: needs a loopback socket, which this sandbox denies")
