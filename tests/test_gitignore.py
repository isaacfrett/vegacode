"""Vega keeps .vega/ and secret files out of git (vega/gitignore.py, project_context.ensure_gitignore)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vega import gitignore
from vega import project_context as pc


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                           "-c", "commit.gpgsign=false", *args],
                          check=True, capture_output=True, text=True).stdout


def repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(tmp_path, "init", "-q")
    return tmp_path


def test_sensitive_names_and_templates() -> None:
    for path in (".env", "app/.env.local", "certs/server.pem", "deploy/id_rsa", "credentials.json",
                 "config/secrets.yaml", "service-account-prod.json"):
        assert gitignore.is_sensitive(path), path
    for path in (".env.example", ".env.sample", "id_rsa.pub", "README.md", "envelope.py", "keys.py"):
        assert not gitignore.is_sensitive(path), path


def test_untracked_secrets_are_ignored_and_templates_stay_tracked(tmp_path: Path) -> None:
    project = repo(tmp_path, {"app.py": "x = 1\n", ".env": "TOKEN=abc\n", ".env.example": "TOKEN=\n",
                              "certs/dev.pem": "-" * 5 + "BEGIN" + "-" * 5 + "\n"})
    state: dict = {}
    note = pc.ensure_gitignore(project, state)
    block = gitignore.block_text((project / ".gitignore").read_text())
    assert block is not None and ".vega/" in block and ".env" in block and "*.pem" in block
    assert "Added" in (note or "")
    others = git(project, "ls-files", "--others", "--exclude-standard").split()
    assert ".env" not in others and "certs/dev.pem" not in others and ".env.example" in others


def test_committed_secret_is_reported_not_untracked(tmp_path: Path) -> None:
    project = repo(tmp_path, {"app.py": "x = 1\n", "config/secrets.yaml": "key: abc\n"})
    git(project, "add", "-A")
    git(project, "commit", "-qm", "init")
    state: dict = {}
    note = pc.ensure_gitignore(project, state) or ""
    assert "config/secrets.yaml" in note and "git rm --cached" in note
    assert "config/secrets.yaml" in git(project, "ls-files")  # Vega never untracks it
    assert "config/secrets.yaml" not in (pc.ensure_gitignore(project, state) or "")  # warned once


def test_block_grows_in_place_without_touching_user_lines(tmp_path: Path) -> None:
    project = repo(tmp_path, {".gitignore": "node_modules/\n", "app.py": "x = 1\n"})
    state: dict = {}
    pc.ensure_gitignore(project, state)
    (project / ".env").write_text("A=1\n")
    pc.ensure_gitignore(project, state)
    text = (project / ".gitignore").read_text()
    assert text.startswith("node_modules/\n") and text.count(gitignore.GITIGNORE_BEGIN) == 1
    assert ".env" in gitignore.block_text(text)


def test_unstage_sensitive_keeps_the_file_out_of_the_commit(tmp_path: Path) -> None:
    project = repo(tmp_path, {"app.py": "x = 1\n", "deploy/id_ed25519": "PRIVATE\n"})
    git(project, "add", "-A")
    held = gitignore.unstage_sensitive(project)
    assert held == ["deploy/id_ed25519"]
    assert git(project, "diff", "--cached", "--name-only").split() == ["app.py"]
    assert (project / "deploy/id_ed25519").exists()
