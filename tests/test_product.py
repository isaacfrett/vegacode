"""The release checklist and verdict (product.py): scoring, which checks block, warn vs block,
not checked never counting as a pass, and readiness read from temporary git repositories."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from vega import product
from vega.product import FAIL, PASS, UNCHECKED, Check


def git(project: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)


def repo(tmp_path: Path, *, version: str = "0.3.0", changelog: str | None = "## 0.3.0\n",
         tag: str | None = "v0.3.0") -> Path:
    project = tmp_path / "site"
    project.mkdir()
    (project / "index.html").write_text("<h1>hi</h1>\n")
    (project / "pyproject.toml").write_text(f"[project]\nname = 'site'\nversion = '{version}'\n")
    if changelog is not None:
        (project / "CHANGELOG.md").write_text(changelog)
    git(project, "init", "-q")
    git(project, "config", "user.email", "t@example.com")
    git(project, "config", "user.name", "t")
    git(project, "add", "-A")
    git(project, "commit", "-qm", "first")
    if tag:
        git(project, "tag", tag)
    return project


def checks_by_key(report: product.Report) -> dict[str, Check]:
    return {check.key: check for check in report.checks}


GOOD_TESTS = product.Tests("python -m pytest -q", ok=True, passed=12, failed=0)
GOOD_TASK = product.TaskChecks([("import", True, ""), ("call", True, "")])


# -- scoring and the verdict ---------------------------------------------------------------------

def test_score_is_n_of_m_with_warn_and_not_checked_never_a_percentage() -> None:
    checks = [Check("a", "A", PASS, True), Check("b", "B", FAIL, True),
              Check("c", "C", FAIL, False), Check("d", "D", UNCHECKED, False),
              Check("e", "E", PASS, False)]
    s = product.score(checks)
    assert (s.passed, s.total, s.blocking, s.warnings, s.unchecked) == (2, 5, 1, 1, 1)
    assert s.counts() == "2 of 5 pass, 1 warn, 1 not checked"
    assert "%" not in s.counts()
    assert product.verdict(checks) == ("Not ready: 1 check blocks shipping", "blocked")


def test_a_failing_warning_never_blocks_and_ready_needs_every_blocker_run() -> None:
    warn_only = [Check("t", "Tests", PASS, True), Check("v", "Version", FAIL, False)]
    assert product.verdict(warn_only) == ("Ready to ship", "ready")
    assert [c.mark for c in warn_only] == ["pass", "warn"]
    # a blocking check that was not run keeps the verdict off "Ready" and is not a pass
    unrun = [Check("t", "Tests", UNCHECKED, True), Check("v", "Version", PASS, False)]
    assert product.verdict(unrun) == ("Not verified: 1 blocking check not run", "unverified")
    assert product.score(unrun).passed == 1 and product.score(unrun).unchecked == 1
    # a warning that was not run does not hold the verdict back
    assert product.verdict([Check("a", "A", PASS, True), Check("x", "X", UNCHECKED, False)]) \
        == ("Ready to ship", "ready")


def test_blocking_policy_matches_the_users_decision(tmp_path: Path) -> None:
    """Tests, task checks, exposed secrets, preview errors (websites) and a missing changelog
    entry block; version, uncommitted, risky code and advisories warn."""
    project = repo(tmp_path)
    report = product.gather(project, GOOD_TESTS, GOOD_TASK)
    blocks = {c.key: c.blocks for c in report.checks}
    assert blocks == {"tests": True, "task_checks": True, "secrets": True, "risky": False,
                      "preview": True, "changelog": True, "version": False, "committed": False,
                      "advisories": False}


# -- each blocker --------------------------------------------------------------------------------

def test_failing_tests_block_and_a_run_on_older_code_is_not_a_pass(tmp_path: Path) -> None:
    project = repo(tmp_path)
    failing = product.Tests("pytest -q", ok=False, passed=10, failed=2,
                            failures=["test_a", "test_b"])
    check = checks_by_key(product.gather(project, failing, GOOD_TASK))["tests"]
    assert check.mark == "block" and check.detail == "2 of 12 fail"
    assert "test_a" in check.fix and "pytest -q" in check.fix
    stale = product.Tests("pytest -q", ok=True, passed=12, failed=0, current=False)
    check = checks_by_key(product.gather(project, stale, GOOD_TASK))["tests"]
    assert check.status == UNCHECKED and check.detail == "not run on this code"
    check = checks_by_key(product.gather(project, None, GOOD_TASK))["tests"]
    assert check.status == UNCHECKED and check.blocks


def test_tree_fingerprint_changes_when_the_code_does(tmp_path: Path) -> None:
    project = repo(tmp_path)
    before = product.tree_fingerprint(project)
    assert product.tree_fingerprint(project) == before
    time.sleep(0.01)
    (project / "index.html").write_text("<h1>changed</h1>\n")
    assert product.tree_fingerprint(project) != before
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "a.py").write_text("x = 1\n")
    first = product.tree_fingerprint(plain)
    (plain / "a.py").write_text("x = 22\n")
    assert product.tree_fingerprint(plain) != first  # outside git too


def test_task_checks_block_when_one_fails(tmp_path: Path) -> None:
    project = repo(tmp_path)
    task = product.TaskChecks([("menu.html at 375 px", False, "391 px wide"),
                               ("import", True, "")])
    check = checks_by_key(product.gather(project, GOOD_TESTS, task))["task_checks"]
    assert check.mark == "block" and check.detail == "1 of 2 fail"
    assert check.fix == "Fix what the task checks report: menu.html at 375 px: 391 px wide."
    none = checks_by_key(product.gather(project, GOOD_TESTS, None))["task_checks"]
    assert none.status == UNCHECKED


def test_tracked_or_unignored_env_file_blocks(tmp_path: Path) -> None:
    project = repo(tmp_path)
    (project / ".env").write_text("KEY=1\n")
    check = checks_by_key(product.gather(project, GOOD_TESTS, GOOD_TASK))["secrets"]
    assert check.mark == "block" and check.detail == "1 file not ignored"
    git(project, "add", ".env")
    git(project, "commit", "-qm", "oops")
    check = checks_by_key(product.gather(project, GOOD_TESTS, GOOD_TASK))["secrets"]
    assert check.mark == "block" and check.detail == "1 file in git"
    assert "git rm --cached" in check.fix


def test_preview_blocks_only_for_websites_and_is_not_checked_in_phase_1(tmp_path: Path) -> None:
    project = repo(tmp_path)
    check = checks_by_key(product.gather(project, GOOD_TESTS, GOOD_TASK))["preview"]
    assert check.status == UNCHECKED and check.blocks
    (project / "index.html").unlink()
    assert "preview" not in checks_by_key(product.gather(project, GOOD_TESTS, GOOD_TASK))


def test_changelog_untouched_since_the_tag_blocks_and_an_entry_passes(tmp_path: Path) -> None:
    project = repo(tmp_path)
    (project / "pyproject.toml").write_text("[project]\nname = 'site'\nversion = '0.4.0'\n")
    check = checks_by_key(product.gather(project, GOOD_TESTS, GOOD_TASK))["changelog"]
    assert check.mark == "block" and check.detail == "no 0.4.0 entry"
    assert check.fix.startswith("Write the 0.4.0 entry in CHANGELOG.md")
    (project / "CHANGELOG.md").write_text("## 0.4.0\n- pickup times\n\n## 0.3.0\n")
    check = checks_by_key(product.gather(project, GOOD_TESTS, GOOD_TASK))["changelog"]
    assert check.status == PASS and check.detail == "0.4.0 entry"
    # changed, but without the new version: still blocks
    (project / "CHANGELOG.md").write_text("## next\n- pickup times\n\n## 0.3.0\n")
    assert checks_by_key(product.gather(project, GOOD_TESTS, GOOD_TASK))["changelog"].mark \
        == "block"


def test_no_changelog_file_blocks(tmp_path: Path) -> None:
    project = repo(tmp_path, changelog=None)
    check = checks_by_key(product.gather(project, GOOD_TESTS, GOOD_TASK))["changelog"]
    assert check.mark == "block" and check.detail == "no changelog file"


# -- warnings and git ---------------------------------------------------------------------------

def test_version_equal_to_the_tag_warns_and_uncommitted_warns(tmp_path: Path) -> None:
    project = repo(tmp_path)
    by_key = checks_by_key(product.gather(project, GOOD_TESTS, GOOD_TASK))
    assert by_key["version"].mark == "warn" and by_key["version"].detail == "still 0.3.0"
    assert by_key["committed"].status == PASS
    (project / "pyproject.toml").write_text("[project]\nname = 'site'\nversion = '0.4.0'\n")
    by_key = checks_by_key(product.gather(project, GOOD_TESTS, GOOD_TASK))
    assert by_key["version"].status == PASS and by_key["version"].detail == "0.3.0 → 0.4.0"
    assert by_key["committed"].mark == "warn" and by_key["committed"].detail == "1 file"


def test_not_a_git_repository_shows_needs_git(tmp_path: Path) -> None:
    project = tmp_path / "plain"
    project.mkdir()
    (project / "index.html").write_text("<p>x</p>\n")
    by_key = checks_by_key(product.gather(project, GOOD_TESTS, GOOD_TASK))
    for key in ("version", "changelog", "committed", "secrets"):
        assert by_key[key].status == UNCHECKED and by_key[key].detail == "needs git", key
    assert by_key["tests"].status == PASS  # the rest still run


def test_ready_when_every_blocker_passes(tmp_path: Path) -> None:
    project = repo(tmp_path, changelog="## 0.3.0\n")
    (project / "index.html").unlink()  # not a website: no preview check in phase 1
    git(project, "commit", "-qam", "drop page")
    (project / "pyproject.toml").write_text("[project]\nname = 'site'\nversion = '0.4.0'\n")
    (project / "CHANGELOG.md").write_text("## 0.4.0\n- x\n")
    git(project, "commit", "-qam", "0.4.0")
    report = product.gather(project, GOOD_TESTS, GOOD_TASK)
    assert product.verdict(report.checks) == ("Ready to ship", "ready")
    assert product.score(report.checks).counts() == "6 of 8 pass, 2 not checked"


# -- what the product is -------------------------------------------------------------------------

@pytest.mark.parametrize(("files", "kind"), [
    ({"index.html": "<p/>"}, "website"),
    ({"package.json": '{"scripts": {"dev": "vite"}}'}, "website"),
    ({"pyproject.toml": "[project]\nname='t'\n[project.scripts]\nt='t:main'\n"},
     "command-line tool"),
    ({"pyproject.toml": "[project]\nname='t'\n"}, "library"),
    ({"notes.txt": "x"}, "project"),
])
def test_kind_rules(tmp_path: Path, files: dict[str, str], kind: str) -> None:
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    assert product.kind(tmp_path) == kind


def test_facts_from_the_vega_md_auto_block(tmp_path: Path) -> None:
    (tmp_path / "VEGA.md").write_text(
        "# VEGA.md\n\n<!-- vega:auto -->\n## fernhill\nOrder-ahead site\n\n- Languages: Python "
        "(4 files), HTML (2 files)\n- Tests: `python -m pytest -q`\n<!-- /vega:auto -->\n")
    facts = product.facts(tmp_path)
    assert (facts.name, facts.description, facts.languages, facts.tests, facts.known) == (
        "fernhill", "Order-ahead site", ["Python", "HTML"], "python -m pytest -q", True)
    empty = tmp_path / "new"
    empty.mkdir()
    assert product.facts(empty).known is False and product.facts(empty).name == "new"
