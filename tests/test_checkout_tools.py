"""Developer tools that write under the repository stop cleanly in an installed package."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from vega import server
from vega.cli import app

REAL_CHECK = server.source_checkout  # conftest replaces it for every other test


@pytest.mark.parametrize("args", [
    ["eval"], ["latency"], ["model", "queue", "granite-8b"], ["lab", "report"],
    ["train", "add-example", "task", "answer"],
])
def test_developer_tools_refuse_outside_a_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str],
) -> None:
    # an installed package: PROJECT_ROOT is the Python library folder, not a clone
    monkeypatch.setattr(server, "source_checkout", REAL_CHECK)
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    assert "source checkout" in " ".join(result.output.split())
    assert list(tmp_path.iterdir()) == []


def test_a_clone_is_recognised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    assert not REAL_CHECK()
    (tmp_path / "src" / "vega").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    assert REAL_CHECK()
