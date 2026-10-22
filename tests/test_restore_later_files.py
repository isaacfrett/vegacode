"""write_loop keeps the best round: files first written in a later round go back too."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from vega import write_loop
from vega.write_loop import Checks, LoopConfig, Produced, Write


class _Host:
    teach = False
    num_ctx = None

    def __init__(self, project: Path) -> None:
        self.project = project

    def emit(self, kind: str, **data: Any) -> None: ...
    def before_write(self, *args: Any, **kwargs: Any) -> Any: return None
    def refresh_written(self) -> None: ...
    def acceptance(self, task: str) -> None: return None
    def stop_flag(self) -> threading.Event | None: return None
    def ask(self, messages: list[dict[str, Any]], **extra: Any) -> str: return ""
    def system(self, prompt: str) -> str: return prompt
    def remember(self, task: str, summary: str) -> None: ...


def _run(project: Path) -> write_loop.LoopResult | None:
    """Round 1 writes app.py (1 failing check); round 2 rewrites app.py and first writes
    schema.py and new.py (3 failing), so round 1 is kept."""
    originals: dict[str, str | None] = {}
    touched: set[str] = set()
    rounds = iter([["app.py"], ["app.py", "schema.py", "new.py"]])
    contents = {1: {"app.py": "A = 1\n"},
                2: {"app.py": "A = 2\n", "schema.py": "S = 2\n", "new.py": "N = 2\n"}}
    state = {"round": 0}

    def targets() -> list[str]:
        state["round"] += 1
        return next(rounds)

    def produce(path: str) -> Produced:
        target = project / path
        originals.setdefault(path, target.read_text() if target.is_file() else None)
        touched.add(path)
        return Produced(writes={path: Write(contents[state["round"]][path])})

    def check(saved: list[str]) -> Checks:
        return Checks(kinds={"tests": 3}, failures=["f"] * (1 if state["round"] == 1 else 3))

    config = LoopConfig(
        name="t", title="T", targets=[], originals=originals, repairs=1, check=check,
        produce_each=produce, round_targets=targets,
        snapshot=lambda: {p: ((project / p).read_text() if (project / p).is_file() else None)
                          for p in touched})
    return write_loop.run_loop(_Host(project), config, [])


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text("A = 0\n")
    (tmp_path / "schema.py").write_text("S = 0\n")
    return tmp_path


def test_later_files_go_back_with_the_kept_round(project: Path) -> None:
    result = _run(project)
    assert result is not None and result.best == 0
    assert (project / "app.py").read_text() == "A = 1\n"
    assert (project / "schema.py").read_text() == "S = 0\n"  # first written in round 2
    assert not (project / "new.py").exists()  # created in round 2


def test_switch_off_leaves_later_files(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "restore_later_files")
    _run(project)
    assert (project / "app.py").read_text() == "A = 1\n"
    assert (project / "schema.py").read_text() == "S = 2\n"
    assert (project / "new.py").read_text() == "N = 2\n"
