"""File-tool behaviour: tolerant edits, line numbers, replace_lines and syntax feedback."""

from __future__ import annotations

from pathlib import Path

from switches import switch_off

from vega.tools import execute


def edit(project: Path, path: str, old: str, new: str) -> str:
    return execute("edit_file", {"path": path, "old_text": old, "new_text": new}, project)


def test_read_file_numbers_lines(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    assert execute("read_file", {"path": "a.py"}, tmp_path) == "   1\tx = 1\n   2\ty = 2"


def test_exact_edit_and_syntax_ok(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")
    result = edit(tmp_path, "a.py", "return 1", "return 2")
    assert "exact match" in result and "Syntax check: OK" in result
    assert (tmp_path / "a.py").read_text() == "def f():\n    return 2\n"



def test_edit_with_identical_texts_is_refused(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")
    result = edit(tmp_path, "a.py", "return 1", "return 1")
    assert result.startswith("Edit not applied: new_text is identical to old_text")
    switch_off(monkeypatch, "noop_edit_refused")
    assert "exact match" in edit(tmp_path, "a.py", "return 1", "return 1")

def test_edit_strips_copied_line_numbers(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")
    edit(tmp_path, "a.py", "   2\t    return 1", "   2\t    return 2")
    assert (tmp_path / "a.py").read_text() == "def f():\n    return 2\n"


def test_edit_tolerates_wrong_indentation_and_reindents(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("class A:\n    def f(self):\n        return 1\n")
    result = edit(tmp_path, "a.py", "def f(self):\n    return 1", "def f(self):\n    return 2")
    assert "ignoring indentation" in result
    assert (tmp_path / "a.py").read_text() == "class A:\n    def f(self):\n        return 2\n"


def test_edit_tolerates_trailing_whitespace(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1   \ny = 2\n")
    result = edit(tmp_path, "a.py", "x = 1\ny = 2", "x = 3\ny = 2")
    assert "trailing whitespace" in result
    assert (tmp_path / "a.py").read_text() == "x = 3\ny = 2\n"


def test_ambiguous_edit_lists_lines(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("    x = 1\ny = 0\n  x = 1\n")
    result = edit(tmp_path, "a.py", "x = 1  ", "x = 2")
    assert result.startswith("Edit not applied") and "lines 1, 3" in result


def test_missing_edit_points_to_closest_line(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def total(values):\n    return sum(values)\n")
    result = edit(tmp_path, "a.py", "def totl(values):", "def total(items):")
    assert "not found" in result and "closest line is 1" in result
    assert (tmp_path / "a.py").read_text().startswith("def total(values)")


def test_syntax_error_is_reported(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    # shipped default (atomic_writes): the write is refused and the file stays as it was
    result = edit(tmp_path, "a.py", "x = 1", "x = (1")
    assert result.startswith("Write not applied: a.py would not parse (SyntaxError")
    assert (tmp_path / "a.py").read_text() == "x = 1\n"
    # with atomic_writes off the edit lands and the result warns
    switch_off(monkeypatch, "atomic_writes")
    result = edit(tmp_path, "a.py", "x = 1", "x = (1")
    assert "WARNING" in result and "SyntaxError" in result


def test_invalid_json_is_reported(tmp_path: Path) -> None:
    result = execute("write_file", {"path": "c.json", "content": '{"a": 1,}'}, tmp_path)
    assert "invalid JSON" in result


def test_replace_lines(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a = 1\nb = 2\nc = 3\n")
    result = execute("replace_lines", {"path": "a.py", "start_line": 2, "end_line": 2, "new_text": "b = 20"}, tmp_path)
    assert result.startswith("Replaced lines 2-2")
    assert (tmp_path / "a.py").read_text() == "a = 1\nb = 20\nc = 3\n"
    bad = execute("replace_lines", {"path": "a.py", "start_line": 3, "end_line": 9, "new_text": ""}, tmp_path)
    assert bad.startswith("Invalid line range")


def test_edits_cannot_escape_project(tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    (tmp_path / "outside.py").write_text("x = 1\n")
    result = execute("replace_lines", {"path": "../outside.py", "start_line": 1, "end_line": 1, "new_text": "x"}, project)
    assert "inside the project" in result
    assert (tmp_path / "outside.py").read_text() == "x = 1\n"


def test_library_folders_are_content_except_at_the_top_of_home(tmp_path, monkeypatch) -> None:
    # a snapshot's Users/<u>/Library (or a project kept under a Library folder) must be visible
    snap = tmp_path / "proj"
    (snap / "root/Users/dana/Library/LaunchAgents").mkdir(parents=True)
    (snap / "root/Users/dana/Library/LaunchAgents/com.x.plist").write_text("<plist/>")
    (snap / "Library").mkdir()
    (snap / "Library/notes.txt").write_text("ledger")
    listed = execute("list_files", {"path": "."}, snap)
    assert "root/Users/dana/Library/LaunchAgents/com.x.plist" in listed
    assert "Library/notes.txt" in listed
    assert "com.x.plist" in execute("find_files", {"query": "com.x"}, snap)
    assert "notes.txt" in execute("search_text", {"query": "ledger"}, snap)
    inside = tmp_path / "Library" / "proj2"
    inside.mkdir(parents=True)
    (inside / "a.py").write_text("x = 1\n")
    assert "a.py" in execute("list_files", {"path": "."}, inside)
    # started in the home folder: ~/Library stays skipped
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: snap))
    assert "Library/notes.txt" not in execute("list_files", {"path": "."}, snap)
    assert "notes.txt" not in execute("find_files", {"query": "notes"}, snap)
