"""Model profiles: parsing, generated Modelfile, --model resolution and score lookup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vega import models, server

PROFILES = """
default = "small"

[profiles.small]
base = "qwen2.5-coder:7b-instruct-q4_K_M"
description = "Default."
num_ctx = 8192
[profiles.small.parameters]
temperature = 0.2

[profiles.thinker]
base = "qwen3:8b"
num_ctx = 16384
"""


@pytest.fixture
def profiles_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "models.toml"
    path.write_text(PROFILES)
    monkeypatch.setattr(models, "PROFILES_FILE", path)
    monkeypatch.setattr(server, "STATE_DIR", tmp_path / "state")
    return path


def test_load_profiles_and_default(profiles_file: Path) -> None:
    default, profiles = models.load()
    assert default == "small"
    assert profiles["thinker"].num_ctx == 16384 and profiles["thinker"].parameters == {}


def test_modelfile_has_base_ctx_and_params(profiles_file: Path) -> None:
    text = models.get("small").modelfile()
    assert text.startswith("FROM qwen2.5-coder:7b-instruct-q4_K_M\n")
    assert "PARAMETER num_ctx 8192" in text and "PARAMETER temperature 0.2" in text
    assert "temperature" not in models.get("thinker").modelfile()


def test_unknown_profile_and_bad_default(profiles_file: Path) -> None:
    with pytest.raises(ValueError, match="Unknown model profile"):
        models.get("huge")
    profiles_file.write_text(PROFILES.replace('default = "small"', 'default = "nope"'))
    with pytest.raises(ValueError, match="default"):
        models.load()


def test_resolve_maps_profiles_and_passes_through_others(profiles_file: Path, monkeypatch) -> None:
    built: list[str] = []
    monkeypatch.setattr(server, "is_running", lambda: True)
    monkeypatch.setattr(server, "has_model", lambda name=None: False)
    monkeypatch.setattr(models, "build", lambda profile, target: built.append(target) or 0)
    assert models.resolve("thinker") == "vega-thinker" and built == ["vega-thinker"]
    assert models.resolve("llama3.1:8b") == "llama3.1:8b"


def test_use_records_active_profile(profiles_file: Path, monkeypatch) -> None:
    monkeypatch.setattr(models, "build", lambda profile, target: 0)
    assert models.use("thinker") == 0 and models.active_profile() == "thinker"


def test_latest_scores_uses_last_fixes_run_per_model(tmp_path: Path) -> None:
    runs = tmp_path / "runs.jsonl"
    rows = [
        {"model": "vega", "passed": 13, "total": 60},
        {"model": "vega", "suite": "goals", "passed": 1, "total": 24},
        {"model": "vega", "suite": "fixes", "passed": 20, "total": 60},
        {"model": "vega-coder-14b", "passed": 40, "total": 60},
    ]
    runs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert models.latest_scores(runs) == {"vega": "20/60", "vega-coder-14b": "40/60"}


def test_repo_profiles_file_is_valid() -> None:
    default, profiles = models.load()
    assert default in profiles
    for profile in profiles.values():
        assert profile.base and profile.num_ctx > 0
