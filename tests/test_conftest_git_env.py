"""tests/conftest.py drops every variable that points git at a repository before any test runs: a
git hook (the pre-push hook runs this suite) sets GIT_DIR to the pushing checkout, and inherited,
every test's `git -C <tmp repo>` acted on the real repository (2026-11-03)."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

CONFTEST = Path(__file__).resolve().parent / "conftest.py"

COMMITS = '''import subprocess


def test_commits_in_its_own_repo(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "--allow-empty", "-m", "x")
    git("config", "user.name", "t")
'''


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


def test_the_list_covers_gits_own_local_env_vars():
    spec = importlib.util.spec_from_file_location("root_conftest_copy", CONFTEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    local = subprocess.run(["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True,
                           check=True).stdout.split()
    assert set(local) <= set(module.GIT_REPO_ENV)


def test_a_hooks_git_dir_never_reaches_the_real_repository(tmp_path):
    decoy = tmp_path / "decoy"
    assert git(decoy.parent, "init", "-q", str(decoy)).returncode == 0
    before = (decoy / ".git" / "config").read_text()
    suite = tmp_path / "suite"
    suite.mkdir()
    shutil.copy(CONFTEST, suite / "conftest.py")
    (suite / "test_commits.py").write_text(COMMITS, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env.update(GIT_DIR=str(decoy / ".git"), GIT_WORK_TREE=str(decoy), VEGA_HOME=str(tmp_path / "home"))
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_commits.py"],
                          cwd=suite, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert git(decoy, "rev-parse", "--verify", "-q", "HEAD").returncode != 0  # no commit landed here
    assert (decoy / ".git" / "config").read_text() == before
