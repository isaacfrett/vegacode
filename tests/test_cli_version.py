"""`vega --version`: the installed vegacode version, else the checkout's pyproject.toml."""

from __future__ import annotations

import tomllib
from importlib import metadata

from typer.testing import CliRunner

import vega
from vega import cli, server


def test_version_flag() -> None:
    result = CliRunner().invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"vegacode {vega.version()}"
    assert vega.version() != "unknown"


def test_version_falls_back_to_the_checkout(monkeypatch) -> None:
    def not_installed(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", not_installed)
    pyproject = tomllib.loads((server.PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert vega.version() == pyproject["project"]["version"]
