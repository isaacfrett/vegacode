"""route_module_entry: a task that runs a project
package as `python -m pkg` while pkg/__main__.py does not exist also targets pkg/__main__.py and
leaves the direct pipeline, which cannot create it."""

from __future__ import annotations

from pathlib import Path

import pytest

from vega import features, project_pipeline, routing

BASE_OFF = "clarify_first,project_context"


@pytest.fixture(autouse=True)
def defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(features.ENV, BASE_OFF)


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


CLI = ("import sys\n\n\ndef main(argv=None):\n    \"\"\"Run.\"\"\"\n    print(argv)\n    return 0\n")
TEST = "from tally.cli import main\n\n\ndef test_main():\n    assert main([]) == 0\n"


def test_on_by_default() -> None:
    assert features.on("route_module_entry")


def test_a_python_m_task_also_targets_main(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, {"tally/__init__.py": "", "tally/cli.py": CLI, "tests/test_cli.py": TEST})
    task = "Make `python -m tally FILE` print the counts; keep tally/cli.py's main()."
    route = routing.route_task(task, tmp_path)
    assert "tally/__main__.py" in route.targets and route.name != routing.DIRECT
    assert "also a target: tally/__main__.py (python -m needs it)" in route.reasons
    seeds = project_pipeline.route_targets(tmp_path, route.record(), task)
    assert seeds[-1] == {**seeds[-1], "path": "tally/__main__.py", "why": "python -m needs it"}
    (tmp_path / "tally" / "__main__.py").write_text("", encoding="utf-8")
    assert "tally/__main__.py" not in routing.route_task(task, tmp_path).targets
    (tmp_path / "tally" / "__main__.py").unlink()
    monkeypatch.setenv(features.ENV, BASE_OFF + ",route_module_entry")
    assert "tally/__main__.py" not in routing.route_task(task, tmp_path).targets


def test_leaves_direct_without_tests(tmp_path: Path) -> None:
    """No visible tests or acceptance checks: not the project pipeline, but not direct either."""
    _write(tmp_path, {"tally/__init__.py": "", "tally/cli.py": CLI})
    task = "Fix tally/cli.py so that `python -m tally` prints its arguments."
    route = routing.route_task(task, tmp_path)
    assert route.name != routing.DIRECT
    assert "tally/__main__.py" in route.targets


@pytest.mark.parametrize("files, task, expected", [
    # the package under src/
    ({"src/tally/__init__.py": "", "src/tally/cli.py": CLI},
     "Make `python -m tally` work.", ["src/tally/__main__.py"]),
    # a dotted package
    ({"tally/__init__.py": "", "tally/tools/__init__.py": "", "tally/tools/run.py": CLI},
     "Run it as `python3 -m tally.tools`.", ["tally/tools/__main__.py"]),
    # a module runs itself
    ({"tally/__init__.py": "", "tally/cli.py": CLI}, "Make `python -m tally.cli` work.", []),
    # not a project package
    ({"tally/__init__.py": "", "tally/cli.py": CLI}, "Run `python -m pytest -q`.", []),
    # already has one
    ({"tally/__init__.py": "", "tally/__main__.py": "", "tally/cli.py": CLI},
     "Make `python -m tally` work.", []),
])
def test_module_entry_files(files: dict, task: str, expected: list, tmp_path: Path) -> None:
    _write(tmp_path, files)
    assert routing.module_entry_files(tmp_path, task, list(files)) == expected


def test_readme_and_visible_tests_count(tmp_path: Path) -> None:
    files = {"tally/__init__.py": "", "tally/cli.py": CLI}
    _write(tmp_path, {**files, "README.md": "Usage: `python -m tally FILE`\n"})
    assert routing.module_entry_files(tmp_path, "Follow the README.", list(files)) \
        == ["tally/__main__.py"]
    assert routing.module_entry_files(tmp_path, "Fix the counting.", list(files)) == []
    _write(tmp_path, {"tests/test_cli.py": "import subprocess, sys\n"
                      "subprocess.run([sys.executable, '-m', 'tally'])\n" + TEST})
    assert routing.module_entry_files(tmp_path, "Fix the counting.", list(files)) \
        == ["tally/__main__.py"]
