"""show_on_miss, references_hint, review_diff and the INI parse check (atomic_writes), on the
7B's actual wrong fixes from the 2026-10-13 constrained runs."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_agent_loop import agent_with, call, text

from vega import features
from vega.tools import execute

SETTINGS_INI = "[storage]\npath = /var/data\n\n[http]\nretries = 3\n"


@pytest.fixture
def on(monkeypatch):
    def enable(*names: str) -> None:
        monkeypatch.setenv(features.ENV_ON, ",".join(names))
    return enable


def test_missed_edit_shows_the_current_file(tmp_path: Path, on) -> None:
    on("show_on_miss")
    (tmp_path / "config_store.py").write_text(
        "SETTINGS = {}\n\n\ndef get_setting(name, default=None):\n    return SETTINGS[name]\n")
    result = execute("edit_file", {"path": "config_store.py",
                                   "old_text": "raise KeyError(f'Setting not found: {name}')",
                                   "new_text": "return default"}, tmp_path)
    assert result.startswith("Edit not applied: old_text was not found")
    assert "The file currently reads" in result and "5\t    return SETTINGS[name]" in result


def test_duplicate_ini_section_is_refused_by_atomic_writes(tmp_path: Path, monkeypatch) -> None:
    # the opt-in parse_guard that did this first was retired (2026-10-17); atomic_writes does it
    monkeypatch.setenv(features.ENV, "")
    (tmp_path / "settings.ini").write_text(SETTINGS_INI)
    result = execute("edit_file", {"path": "settings.ini", "old_text": "",
                                   "new_text": "\n[http]\ntimeout = 30\n"}, tmp_path)
    assert "not applied" in result and "invalid INI" in result and "http" in result
    assert (tmp_path / "settings.ini").read_text() == SETTINGS_INI


def test_rename_lists_remaining_references(tmp_path: Path, on) -> None:
    on("references_hint")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "billing.py").write_text("def calc_total(prices):\n    return sum(prices)\n")
    (tmp_path / "src" / "checkout.py").write_text(
        "from billing import calc_total\n\n\ndef checkout(p):\n    return calc_total(p)\n")
    result = execute("edit_file", {"path": "src/billing.py", "old_text": "def calc_total(prices):",
                                   "new_text": "def calculate_total(prices):"}, tmp_path)
    assert "calc_total still appears at src/checkout.py:1, src/checkout.py:5" in result


def test_no_reference_note_when_nothing_remains(tmp_path: Path, on) -> None:
    on("references_hint")
    (tmp_path / "a.py").write_text("def old_name():\n    pass\n")
    result = execute("edit_file", {"path": "a.py", "old_text": "def old_name():",
                                   "new_text": "def new_name():"}, tmp_path)
    assert "Note:" not in result


def test_review_diff_sends_the_diff_back_once(tmp_path: Path, on) -> None:
    on("review_diff")
    (tmp_path / "grading.py").write_text(
        'def letter_grade(score):\n    if score > 90:\n        return "A"\n    return "B"\n')
    agent, seen = agent_with(tmp_path, [
        call("edit_file", path="grading.py", old_text='return "A"', new_text="return 'A'"),
        text("Fixed."),
        call("edit_file", path="grading.py", old_text="if score > 90:", new_text="if score >= 90:"),
        text("Fixed: 90 is now an A."),
    ])
    task = "letter_grade(90) returns 'B'. 90 and up is an A. Fix it."
    assert agent.run(task) == "Fixed: 90 is now an A."
    review = seen[2][-1]["content"]
    assert review.startswith("(Vega harness) Before you finish")
    assert task in review and "+        return 'A'" in review
    assert sum("Before you finish" in str(m.get("content")) for m in agent.messages) == 1
    assert "score >= 90" in (tmp_path / "grading.py").read_text()


def test_review_diff_skips_turns_without_changes(tmp_path: Path, on) -> None:
    on("review_diff")
    (tmp_path / "a.py").write_text("x = 1\n")
    agent, _ = agent_with(tmp_path, [call("read_file", path="a.py"), text("x is 1.")])
    assert agent.run("What is x?") == "x is 1."


def test_append_that_redefines_a_function_is_refused(tmp_path: Path, on) -> None:
    on("append_guard")
    original = "def terrain_value(terrain):\n    return 0\n"
    (tmp_path / "value.py").write_text(original)
    result = execute("edit_file", {"path": "value.py", "old_text": "",
                                   "new_text": "def terrain_value(terrain):\n    return 1\n"}, tmp_path)
    assert result.startswith("Edit not applied: value.py already defines terrain_value")
    assert "(lines 1-2)" in result and "1\tdef terrain_value(terrain):" in result
    assert (tmp_path / "value.py").read_text() == original
    ok = execute("edit_file", {"path": "value.py", "old_text": "",
                               "new_text": "def other():\n    return 2\n"}, tmp_path)
    assert "appended at the end" in ok
