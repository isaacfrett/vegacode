"""Per-test feature switches.

conftest runs every test with the shipped defaults. A test that is not about a default feature
(for example one that scripts the agent loop with a question the question pipeline would take)
turns that one feature off with switch_off(monkeypatch, "question_pipeline"), and says why in a
comment next to the call.
"""

from __future__ import annotations

import os

import pytest

from vega import features


def switch_off(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Turn default features off for this test only, keeping any already off."""
    unknown = set(names) - features.FEATURES.keys()
    assert not unknown, f"not default features: {sorted(unknown)}"
    current = [n for n in os.environ.get(features.ENV, "").split(",") if n]
    monkeypatch.setenv(features.ENV, ",".join([*current, *(n for n in names if n not in current)]))
