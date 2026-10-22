"""write_guard.

An agent-loop write is held once when write_file replaces an unread existing file and loses lines,
when a dependency/config file loses a package or key, or when a .py file loses a module-level
name. The same write repeated runs; new files, files read this turn with nothing lost, and the
pipelines' own writes are never held.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vega import tools
from vega.agent import CodingAgent
from vega.write_guard import HOLD_PREFIX, MAX_HOLDS_PER_PATH, WriteGuard, lost_entries, lost_names

REQUIREMENTS = "click==8.1.7\nkettlehttp==3.8.3\nrich==13.9.4\n"
CONFIG = ('"""Nightly contact sync settings (Acme CRM)."""\n\n'
          "BATCH_SIZE = 500  # records per bulk-upsert request\n"
          "REQUESTS_PER_SECOND = 20  # client-side throttle, per API token\n")
# A real failure: the docstring is opened and never closed before the
# assignments, so they become docstring text; it parses and lints clean
SWALLOWED = ('"""Nightly contact sync settings (Acme CRM).\n\n'
             "Values from the Acme bulk API docs.\n\n"
             "BATCH_SIZE = 250  # records per bulk-upsert request\n"
             'REQUESTS_PER_SECOND = 8  # client-side throttle, per API token\n"""\n')


# -- the rules (WriteGuard.assess / check) --------------------------------------------------------

def test_unread_existing_file_that_loses_lines_is_held_with_what_is_lost() -> None:
    note = WriteGuard().check("write_file", "requirements.txt", REQUIREMENTS, "kettlehttp==3.8.5\n")
    assert note is not None and note.startswith(HOLD_PREFIX)
    assert "you have not read it this turn" in note
    assert "click==8.1.7" in note and "rich==13.9.4" in note
    assert "1\tclick==8.1.7" in note  # the current content, numbered
    assert "edit_file" in note and "same write_file call again" in note


def test_new_or_empty_file_is_not_held() -> None:
    guard = WriteGuard()
    assert guard.check("write_file", "requirements.txt", None, "kettlehttp==3.8.5\n") is None
    assert guard.check("write_file", "notes.md", "\n", "hello\n") is None


def test_read_file_with_nothing_lost_is_not_held() -> None:
    guard = WriteGuard()
    guard.saw("requirements.txt")
    after = "click==8.1.7\nkettlehttp==3.8.5\nrich==13.9.4\n"  # a version change is not a loss
    assert guard.check("write_file", "requirements.txt", REQUIREMENTS, after) is None
    guard.saw("app.py")
    assert guard.check("write_file", "app.py", "X = 1\n", "X = 2\n\ndef f():\n    return X\n") \
        is None


def test_unread_write_that_keeps_every_line_is_not_held() -> None:
    assert WriteGuard().check("write_file", "README.md", "# App\n", "# App\n\nMore.\n") is None


def test_read_dependency_file_that_loses_a_package_is_held() -> None:
    guard = WriteGuard()
    guard.saw("requirements.txt")  # read, but the write still drops click
    note = guard.check("write_file", "requirements.txt", "click==8.1.7\n", "pdfmaker>=3.0\n")
    assert note is not None and "dependency/config file requirements.txt" in note
    assert "click==8.1.7" in note
    assert "you have not read it" not in note


def test_edit_replacing_a_different_packages_line_is_held() -> None:
    # edit_file old "click==8.1.7" -> "pdftool>=1.4"
    before = "click==8.1.7\n"
    plan_after = "pdftool>=1.4\n"
    note = WriteGuard().check("edit_file", "requirements.txt", before, plan_after)
    assert note is not None and "replaces a different package's line" in note
    assert "Make old_text cover only what should change" in note


def test_edit_changing_a_pin_is_not_held() -> None:
    after = REQUIREMENTS.replace("kettlehttp==3.8.3", "kettlehttp==3.8.5")
    assert WriteGuard().check("edit_file", "requirements.txt", REQUIREMENTS, after) is None


def test_python_names_swallowed_by_an_unclosed_docstring_are_held() -> None:
    guard = WriteGuard()
    guard.saw("sync/config.py")
    note = guard.check("write_file", "sync/config.py", CONFIG, SWALLOWED)
    assert note is not None
    assert "BATCH_SIZE, REQUESTS_PER_SECOND" in note
    assert "no longer module-level" in note


def test_an_edit_that_opens_a_docstring_over_later_code_is_held() -> None:
    before = '"""Settings."""\n\nTIMEOUT = 5\n\nRETRIES = 3\n'
    after = '"""Settings.\n\nTIMEOUT = 10\n\nRETRIES = 3\n"""\n'
    lost, swallowed = lost_names("edit_file", "settings.py", before, after)
    assert lost == ["TIMEOUT", "RETRIES"] and swallowed == ["TIMEOUT", "RETRIES"]
    assert WriteGuard().check("edit_file", "settings.py", before, after) is not None


def test_renaming_the_definition_an_edit_replaced_is_not_held() -> None:
    before = "def old_name():\n    return 1\n\n\ndef other():\n    return 2\n"
    after = before.replace("def old_name", "def new_name")
    assert lost_names("edit_file", "mod.py", before, after) == ([], [])
    assert WriteGuard().check("edit_file", "mod.py", before, after) is None


def test_an_edit_that_deletes_a_definition_it_replaced_is_not_held_but_write_file_is() -> None:
    before = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    after = "def a():\n    return 1\n"  # b removed: its lines changed, so it was touched
    assert WriteGuard().check("edit_file", "mod.py", before, after) is None
    # write_file counts every lost name
    guard = WriteGuard()
    guard.saw("mod.py")
    assert "would no longer define b" in str(guard.check("write_file", "mod.py", before, after))


def test_repeating_the_same_write_runs_and_a_path_is_held_at_most_twice() -> None:
    guard = WriteGuard()
    first = guard.check("write_file", "requirements.txt", REQUIREMENTS, "kettlehttp==3.8.5\n")
    assert first is not None
    # the note showed the file: it now counts as read, and the same write runs
    assert guard.check("write_file", "requirements.txt", REQUIREMENTS, "kettlehttp==3.8.5\n") is None
    assert guard.check("write_file", "requirements.txt", REQUIREMENTS, "rich==13.9.4\n") is not None
    assert MAX_HOLDS_PER_PATH == 2
    assert guard.check("write_file", "requirements.txt", REQUIREMENTS, "click==8.1.7\n") is None


@pytest.mark.parametrize(("path", "before", "after", "lost"), [
    ("pyproject.toml", '[project]\nname = "a"\ndependencies = ["click>=8", "rich"]\n',
     '[project]\nname = "a"\ndependencies = ["rich"]\n', ["project.dependencies: click>=8"]),
    ("pyproject.toml", "[tool.lint]\nmax-line = 100\nignore = []\n",
     "[tool.lint]\nmax-line = 100\n", ["tool.lint.ignore"]),
    ("setup.cfg", "[options]\ninstall_requires =\n    click\n    rich\n",
     "[options]\ninstall_requires =\n    rich\n", ["[options] install_requires: click"]),
    ("package.json", '{"name": "a", "dependencies": {"left-pad": "1.0", "react": "18"}}',
     '{"name": "a", "dependencies": {"react": "19"}}', ["dependencies: left-pad"]),
    ("deps.lock", "click==8.1.7\nrich==13.9.4\n", "rich==13.9.4\n", ["click==8.1.7"]),
])
def test_config_entries_lost(path: str, before: str, after: str, lost: list[str]) -> None:
    assert lost_entries(path, before, after) == lost


def test_config_value_changes_are_not_losses() -> None:
    assert lost_entries("pyproject.toml", '[project]\nversion = "1"\ndependencies = ["a==1"]\n',
                        '[project]\nversion = "2"\ndependencies = ["a==2", "b"]\n') == []
    assert lost_entries("service.toml", "[metrics]\nenabled = false\n",
                        "[metrics]\nenabled = true\n") == []
    assert lost_entries("notes.txt", "a\n", "b\n") == []  # not a dependency/config file


# -- in the agent loop -----------------------------------------------------------------------------

class Msg(SimpleNamespace):
    def model_dump(self, exclude_none: bool = False) -> dict:
        data: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in self.tool_calls]
        return data


def call(name: str, **args: Any) -> Msg:
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    return Msg(content=None, tool_calls=[SimpleNamespace(id=f"c-{name}", function=fn)])


def scripted(project: Path, replies: list[Msg]) -> tuple[CodingAgent, list]:
    agent = CodingAgent(project, allow_commands=False)
    seen: list = []

    def create(**kwargs: Any) -> Any:
        seen.append(list(kwargs["messages"]))
        reply = replies.pop(0) if replies else Msg(content="Done.", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=reply)], usage=None)

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent, seen


def results(seen: list) -> list[str]:
    return [m["content"] for m in seen[-1] if m.get("role") == "tool"]


TASK = "Pin kettlehttp to 3.8.5 in requirements.txt."


def test_agent_loop_holds_the_blind_overwrite_then_runs_the_same_call(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text(REQUIREMENTS)
    write = call("write_file", path="requirements.txt", content="kettlehttp==3.8.5\n")
    agent, seen = scripted(tmp_path, [write, write])
    agent._run(TASK)
    held, ran = results(seen)[:2]
    assert held.startswith(HOLD_PREFIX) and "click==8.1.7" in held
    assert ran.startswith("Wrote requirements.txt")
    assert (tmp_path / "requirements.txt").read_text() == "kettlehttp==3.8.5\n"


def test_agent_loop_follows_the_note_with_an_edit(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text(REQUIREMENTS)
    agent, seen = scripted(tmp_path, [
        call("write_file", path="requirements.txt", content="kettlehttp==3.8.5\n"),
        call("edit_file", path="requirements.txt", old_text="kettlehttp==3.8.3",
             new_text="kettlehttp==3.8.5"),
    ])
    agent._run(TASK)
    assert results(seen)[1].startswith("Edited requirements.txt")
    assert (tmp_path / "requirements.txt").read_text() == \
        "click==8.1.7\nkettlehttp==3.8.5\nrich==13.9.4\n"


def test_agent_loop_does_not_hold_after_read_or_on_new_files(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("X = 1\n")
    agent, seen = scripted(tmp_path, [
        call("read_file", path="app.py"),
        call("write_file", path="app.py", content="X = 2\n"),
        call("write_file", path="new.py", content="Y = 1\n"),
    ])
    agent._run("Set X to 2 in app.py and add new.py with Y = 1.")
    assert results(seen)[1].startswith("Wrote app.py")
    assert results(seen)[2].startswith("Wrote new.py")


def test_switch_off_runs_the_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "write_guard")
    (tmp_path / "requirements.txt").write_text(REQUIREMENTS)
    agent, seen = scripted(tmp_path, [
        call("write_file", path="requirements.txt", content="kettlehttp==3.8.5\n")])
    agent._run(TASK)
    assert results(seen)[0].startswith("Wrote requirements.txt")


def test_pipeline_writes_go_through_before_write_unheld(tmp_path: Path) -> None:
    # write_loop and the project pipeline call execute(..., before_write=agent._before_write)
    (tmp_path / "requirements.txt").write_text(REQUIREMENTS)
    agent = CodingAgent(tmp_path, allow_commands=False)
    result = tools.execute("write_file", {"path": "requirements.txt", "content": "x==1\n"},
                           tmp_path, before_write=agent._before_write, atomic=False)
    assert result.startswith("Wrote requirements.txt")
