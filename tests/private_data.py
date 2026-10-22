"""Tests that read eval data skip cleanly where it is absent.

A test that reads the eval cases (evals/*.jsonl), experiments or the training task pool calls
`require_eval_data()` (or `require_training_tasks()`) first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HAS_EVAL_DATA = (ROOT / "evals" / "cases.jsonl").is_file()
HAS_TRAINING_TASKS = (ROOT / "training" / "tasks" / "tasks.jsonl").is_file()


def require_eval_data() -> None:
    if not HAS_EVAL_DATA:
        pytest.skip("eval data not present")


def require_training_tasks() -> None:
    require_eval_data()
    if not HAS_TRAINING_TASKS:
        pytest.skip("training task data not present")
