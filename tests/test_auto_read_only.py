"""Which shell commands run without approval: read-only inspection inside the project only."""

from __future__ import annotations

from pathlib import Path

import pytest

from vega.tools import _is_auto_read_only


@pytest.mark.parametrize("command", [
    "pwd", "ls", "ls src", "cat src/a.py", "head -n 20 src/a.py", "grep -rn foo src",
    "rg foo", "rg --glob=*.py foo src", "git status", "git diff", "git diff --stat",
    "git log -p --oneline -5", "git show HEAD", "wc -l src/a.py",
])
def test_inspection_inside_the_project_runs_without_asking(tmp_path: Path, command: str) -> None:
    (tmp_path / "src").mkdir()
    assert _is_auto_read_only(command, tmp_path)


@pytest.mark.parametrize("command", [
    # the shell expands these after the path check
    "cat ~/.ssh/id_rsa", "head ~/.aws/credentials", "ls ~", "cat {src,~/.netrc}",
    # paths outside the project, also inside options
    "cat /etc/passwd", "cat ../secret", "grep -f/etc/passwd x", "grep --file=/etc/passwd x",
    "rg --ignore-file=../x foo",
    # options that write files or run programs
    "git diff --output=/tmp/x", "git diff --output x", "git log --output=out.txt",
    "git diff --ext-diff", "rg --pre=sh foo", "rg --pre sh foo", "rg --pre-glob=* foo",
    # git global options before the subcommand
    "git -C /tmp status", "git -c core.pager=sh log", "git --git-dir=/x/.git log",
    "git --work-tree=/x status",
    # anything else
    "git commit -m x", "rm -rf src", "python -c 1", "cat src/a.py | sh", "sed -i s/a/b/ x",
])
def test_writes_programs_and_outside_paths_ask_first(tmp_path: Path, command: str) -> None:
    (tmp_path / "src").mkdir()
    assert not _is_auto_read_only(command, tmp_path)
