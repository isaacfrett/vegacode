"""`vega eval` unloads the model when it ends, unless VEGA_EVAL_KEEP_LOADED=1: another eval is
using the same server at the same time (the parallel serving bench, concurrent A/B arms), and an
unload would make it reload the model mid-run."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner


@pytest.mark.parametrize("keep, unloads", [(None, 1), ("1", 0)])
def test_eval_unload_respects_keep_loaded(monkeypatch, keep, unloads) -> None:
    from vega import cli

    calls: list = []
    monkeypatch.setattr(cli, "_require_checkout", lambda command: None)
    monkeypatch.setattr(cli, "_ensure_server", lambda *a: None)
    monkeypatch.setattr(cli, "run_evaluations", lambda **kw: None)
    monkeypatch.setattr(cli.server, "unload", lambda model=None: calls.append(model) or True)
    if keep is None:
        monkeypatch.delenv("VEGA_EVAL_KEEP_LOADED", raising=False)
    else:
        monkeypatch.setenv("VEGA_EVAL_KEEP_LOADED", keep)
    result = CliRunner().invoke(cli.app, ["eval", "--suite", "fixes", "--repeat", "1"])
    assert result.exit_code == 0, result.output
    assert len(calls) == unloads
