"""Post-write diagnostics: parse checks, ruff lint findings, and their wording."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from vega import diagnostics
from vega.diagnostics import after_write

needs_ruff = pytest.mark.skipif(
    diagnostics._ruff_path(Path.cwd().resolve()) is None, reason="ruff not installed"
)


def write(project: Path, name: str, text: str) -> str:
    path = project / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return after_write(project, path, text)


@needs_ruff
def test_reports_undefined_name_and_unused_import_with_lines(tmp_path: Path) -> None:
    note = write(tmp_path, "src/app.py", "import os\n\n\ndef f():\n    return missing_name\n")
    assert "Syntax check: OK." in note and "Lint findings" in note
    assert "1:8 F401" in note and "`os` imported but unused" in note
    assert "5:12 F821" in note and "missing_name" in note
    assert "WARNING:" not in note


@needs_ruff
def test_clean_file_reports_lint_ok(tmp_path: Path) -> None:
    note = write(tmp_path, "a.py", "import os\n\nprint(os.sep)\n")
    assert note == "\nSyntax check: OK.\nLint: OK."


@needs_ruff
def test_findings_are_capped(tmp_path: Path) -> None:
    text = "".join(f"import mod{i}\n" for i in range(15))
    note = write(tmp_path, "a.py", text)
    assert note.count("F401") == diagnostics.LINT_LIMIT
    assert note.endswith("(5 more)")
    assert "WARNING:" not in note


@needs_ruff
def test_lint_does_not_write_cache_into_project(tmp_path: Path) -> None:
    write(tmp_path, "a.py", "import os\n")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.py"]


@needs_ruff
def test_large_file_is_fast(tmp_path: Path) -> None:
    text = "import os\n\n" + "".join(
        f"def func_{i}(value):\n    return os.path.join(str(value), '{i}')\n\n" for i in range(100)
    )
    assert len(text.splitlines()) >= 300
    write(tmp_path, "warm.py", "x = 1\n")  # first call pays the ruff path lookup
    start = time.perf_counter()
    note = write(tmp_path, "big.py", text)
    assert time.perf_counter() - start < 2
    assert "Lint: OK." in note


def test_syntax_error_wording_unchanged(tmp_path: Path) -> None:
    note = write(tmp_path, "a.py", "def f(:\n    pass\n")
    assert note == "\nWARNING: the file now has a SyntaxError at line 1: invalid syntax. Fix it."


def test_missing_ruff_adds_no_lint_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "_ruff_path", lambda project: None)
    assert write(tmp_path, "a.py", "import os\n") == "\nSyntax check: OK."


def test_ruff_lookup_prefers_project_venv(tmp_path: Path) -> None:
    fake = tmp_path / ".venv" / "bin" / "ruff"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    assert diagnostics._ruff_path(tmp_path.resolve()) == str(fake.resolve())


def test_json_wording_unchanged(tmp_path: Path) -> None:
    assert write(tmp_path, "a.json", '{"a": 1}') == "\nJSON check: OK."
    note = write(tmp_path, "b.json", '{"a": }')
    assert note.startswith("\nWARNING: the file is now invalid JSON at line 1:")


def test_toml_checked(tmp_path: Path) -> None:
    assert write(tmp_path, "a.toml", '[tool]\nname = "x"\n') == "\nTOML check: OK."
    note = write(tmp_path, "b.toml", "[tool]\nname = \n")
    assert note.startswith("\nWARNING: the file is now invalid TOML:") and "line 2" in note


def test_yaml_checked_only_when_pyyaml_installed(tmp_path: Path) -> None:
    try:
        import yaml  # noqa: F401
    except ImportError:
        assert write(tmp_path, "a.yaml", "a: [1, 2\n") == ""
        return
    assert write(tmp_path, "a.yaml", "a: 1\n") == "\nYAML check: OK."
    assert "WARNING: the file is now invalid YAML" in write(tmp_path, "b.yaml", "a: [1, 2\n")


def test_other_files_get_no_note(tmp_path: Path) -> None:
    assert write(tmp_path, "notes.txt", "anything") == ""


# -- the read-only sandbox seam (systems review #6) ---------------------------------------------------

def test_readonly_argv_wraps_only_where_the_sandbox_exists(monkeypatch) -> None:
    from vega import sandbox

    argv = ["ruff", "check", "-"]
    monkeypatch.setattr(sandbox, "available", lambda: True)
    assert sandbox.readonly_argv(argv) == [sandbox.EXECUTABLE, "-p", sandbox.readonly_profile(),
                                           *argv]
    monkeypatch.setattr(sandbox, "available", lambda: False)  # Linux, Windows, nested
    unchanged = sandbox.readonly_argv(argv)
    assert unchanged == argv and unchanged is not argv


def test_readonly_argv_denies_writes_in_the_real_sandbox(tmp_path: Path) -> None:
    import subprocess

    from vega import sandbox

    if not sandbox.available():
        pytest.skip("needs the macOS sandbox")
    target = tmp_path / "written.txt"
    result = subprocess.run(sandbox.readonly_argv(["/usr/bin/touch", str(target)]),
                            capture_output=True, check=False, timeout=10)
    assert result.returncode != 0 and not target.exists()
    readable = subprocess.run(sandbox.readonly_argv(["/bin/echo", "ok"]), capture_output=True,
                              text=True, check=False, timeout=10)
    assert readable.stdout == "ok\n"
