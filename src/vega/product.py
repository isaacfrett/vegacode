"""The product view's facts and the release checklist (product workbench, phase 1).

Pure functions over data that exists today: the VEGA.md auto block (what the product is), the
last test run and the last task checks (acceptance) of this session, the secret-file scan
(gitignore.scan), and git (last tag, version, changelog, uncommitted files). No UI imports; the
TUI (tui_product.py) draws what this returns.

The score is "n of m pass" with warnings and not-checked items counted separately, never a
percentage. A check Vega could not run is *not checked* and never counts as a pass. Which checks
block shipping is the user's decision of 2026-10-17 (product workbench plan): tests,
task checks, exposed secrets, preview errors (websites) and a missing changelog entry block;
everything else warns.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PASS, FAIL, UNCHECKED = "pass", "fail", "unchecked"
KINDS = ("website", "command-line tool", "library", "app", "project")
CHANGELOG_NAMES = re.compile(r"^(changelog|changes|history)(\.[a-z]+)?$", re.IGNORECASE)
AUTO_START, AUTO_END = "<!-- vega:auto -->", "<!-- /vega:auto -->"


# -- checks and the verdict --------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    key: str  # tests | task_checks | secrets | risky | preview | changelog | version | ...
    label: str  # in a product owner's words ("Task checks", not "acceptance")
    status: str  # pass | fail | unchecked
    blocks: bool  # a failure blocks shipping; otherwise it warns
    detail: str = ""  # the right-hand figure ("48 of 48", "no 0.4.0 entry")
    evidence: str = ""  # what Enter shows: where the result came from
    fix: str = ""  # the prepared request `f` puts in the message box ("" = nothing to ask)
    next: str = ""  # the blocker as a task, for the Product pane's Next list

    @property
    def mark(self) -> str:
        """pass | block | warn | unchecked: how the row is drawn and counted."""
        if self.status == FAIL:
            return "block" if self.blocks else "warn"
        return self.status


@dataclass(frozen=True)
class Score:
    passed: int
    total: int
    blocking: int
    warnings: int
    unchecked: int
    unchecked_blocking: int  # not checked, and would block if it failed

    def counts(self) -> str:
        """'4 of 9 pass, 2 warn, 1 not checked' (blockers are in the verdict above it)."""
        parts = [f"{self.passed} of {self.total} pass"]
        if self.warnings:
            parts.append(f"{self.warnings} warn")
        if self.unchecked:
            parts.append(f"{self.unchecked} not checked")
        return ", ".join(parts)


def score(checks: list[Check]) -> Score:
    marks = [check.mark for check in checks]
    return Score(passed=marks.count(PASS), total=len(checks), blocking=marks.count("block"),
                 warnings=marks.count("warn"), unchecked=marks.count(UNCHECKED),
                 unchecked_blocking=sum(c.status == UNCHECKED and c.blocks for c in checks))


def verdict(checks: list[Check]) -> tuple[str, str]:
    """(text, tone): tone is blocked, unverified or ready. A blocking check that was not run
    keeps the verdict off "Ready": not checked never counts as a pass."""
    s = score(checks)
    if s.blocking:
        noun = "check blocks" if s.blocking == 1 else "checks block"
        return f"Not ready: {s.blocking} {noun} shipping", "blocked"
    if s.unchecked_blocking:
        noun = "check" if s.unchecked_blocking == 1 else "checks"
        return f"Not verified: {s.unchecked_blocking} blocking {noun} not run", "unverified"
    return "Ready to ship", "ready"


def blockers(checks: list[Check]) -> list[Check]:
    return [check for check in checks if check.mark == "block"]


# -- inputs ------------------------------------------------------------------------------------

@dataclass
class Tests:
    """The last test run this session (the Tests pane's TestRun, reduced)."""
    command: str
    ok: bool
    passed: int | None = None
    failed: int | None = None
    failures: list[str] = field(default_factory=list)
    current: bool = True  # run on the code as it is now (tree fingerprint unchanged)


@dataclass
class TaskChecks:
    """The last acceptance result: (label, ok, detail) per item."""
    items: list[tuple[str, bool, str]]
    current: bool = True


@dataclass
class Git:
    repo: bool = False
    tag: str | None = None  # the last tag reachable from HEAD
    dirty: list[str] = field(default_factory=list)  # uncommitted paths
    changelog: str | None = None  # its file name, when there is one
    changelog_changed: bool = False  # differs from the last tag (or is new since it)
    changelog_text: str = ""


@dataclass
class Facts:
    name: str
    description: str = ""
    kind: str = "project"
    languages: list[str] = field(default_factory=list)
    tests: str = ""  # the test command, from VEGA.md
    known: bool = False  # VEGA.md has its auto block
    version: str | None = None
    version_file: str = ""


@dataclass
class Report:
    facts: Facts
    checks: list[Check]
    git: Git
    secrets: tuple[list[str], list[str]] | None  # (exposed, tracked); None outside git


# -- what the product is -----------------------------------------------------------------------

def _read(path: Path, limit: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(_read(path))
    except tomllib.TOMLDecodeError:
        return {}


def _package(project: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read(project / "package.json") or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def kind(project: Path) -> str:
    """website, command-line tool, library, app or project: rules only, no model.

    index.html (at the root or under public/, static/, src/) or a dev/start script in
    package.json gives a website; a console script, a bin entry or a Run line in VEGA.md gives a
    command-line tool; a package with no entry point gives a library."""
    package = _package(project)
    scripts = package.get("scripts") if isinstance(package.get("scripts"), dict) else {}
    if any((project / where / "index.html").is_file() for where in ("", "public", "static", "src")):
        return "website"
    if "dev" in scripts or "start" in scripts:
        return "website"
    pyproject = _toml(project / "pyproject.toml")
    table = pyproject.get("project") if isinstance(pyproject.get("project"), dict) else {}
    if table.get("scripts") or package.get("bin"):
        return "command-line tool"
    if (project / "src" / "main.rs").is_file() or (project / "main.go").is_file():
        return "command-line tool"
    if re.search(r"^- Run: ", _auto_block(project), re.MULTILINE):
        return "command-line tool"
    if (project / "Package.swift").is_file() or any(project.glob("*.xcodeproj")) \
            or (project / "project.godot").is_file():
        return "app"
    if table or package or (project / "Cargo.toml").is_file() or (project / "setup.py").is_file():
        return "library"
    return "project"


def _auto_block(project: Path) -> str:
    text = _read(project / "VEGA.md", 40_000)
    start, end = text.find(AUTO_START), text.find(AUTO_END)
    if start == -1 or end < start:
        return ""
    return text[start + len(AUTO_START):end].strip("\n")


def version(project: Path) -> tuple[str | None, str]:
    """(the product's version, the file it is in) from pyproject.toml, package.json or
    Cargo.toml; (None, "") when none states one (a dynamic version included)."""
    pyproject = _toml(project / "pyproject.toml")
    table = pyproject.get("project") if isinstance(pyproject.get("project"), dict) else {}
    tool = pyproject.get("tool") if isinstance(pyproject.get("tool"), dict) else {}
    poetry = tool.get("poetry") if isinstance(tool.get("poetry"), dict) else {}
    for value, where in ((table.get("version"), "pyproject.toml"),
                         (poetry.get("version"), "pyproject.toml"),
                         (_package(project).get("version"), "package.json"),
                         (_toml(project / "Cargo.toml").get("package", {}).get("version"),
                          "Cargo.toml")):
        if isinstance(value, str) and value.strip():
            return value.strip(), where
    return None, ""


def facts(project: Path) -> Facts:
    """Name, description, kind and languages: the VEGA.md auto block first, then the project's
    own files for the name. `known` is False when VEGA.md has no auto block yet."""
    block = _auto_block(project)
    found = Facts(name=project.name or "project", kind=kind(project), known=bool(block))
    lines = block.splitlines()
    if lines and lines[0].startswith("## "):
        found.name = lines[0][3:].strip() or found.name
        if len(lines) > 1 and lines[1].strip() and not lines[1].startswith("-"):
            found.description = lines[1].strip()
    for line in lines:
        if line.startswith("- Languages: "):
            found.languages = [re.sub(r"\s*\(.*?\)", "", part).strip()
                               for part in line[len("- Languages: "):].split(",")]
        elif line.startswith("- Tests: "):
            first = re.search(r"`([^`]+)`", line)
            found.tests = first.group(1) if first else ""
    found.version, found.version_file = version(project)
    return found


# -- git -----------------------------------------------------------------------------------------

def _git(project: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(["git", "-C", str(project), *args], capture_output=True, text=True,
                              timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def _out(result: subprocess.CompletedProcess[str] | None) -> str | None:
    return result.stdout if result is not None and result.returncode == 0 else None


def is_repo(project: Path) -> bool:
    return (_out(_git(project, "rev-parse", "--is-inside-work-tree")) or "").strip() == "true"


def changelog_file(project: Path) -> str | None:
    try:
        names = sorted(p.name for p in project.iterdir() if p.is_file())
    except OSError:
        return None
    return next((name for name in names if CHANGELOG_NAMES.match(name)), None)


def git_facts(project: Path) -> Git:
    if not is_repo(project):
        return Git(changelog=changelog_file(project))
    found = Git(repo=True, changelog=changelog_file(project))
    tag = (_out(_git(project, "describe", "--tags", "--abbrev=0")) or "").strip()
    found.tag = tag or None
    status = _out(_git(project, "status", "--porcelain", "--untracked-files=all")) or ""
    found.dirty = [line[3:].strip().strip('"') for line in status.splitlines()
                   if line.strip() and not line[3:].startswith(".vega/")]
    if found.changelog:
        found.changelog_text = _read(project / found.changelog)
        if found.tag:
            tracked = _git(project, "ls-files", "--error-unmatch", found.changelog)
            if tracked is None or tracked.returncode != 0:
                found.changelog_changed = True  # new since the tag (untracked)
            else:
                diff = _git(project, "diff", "--quiet", found.tag, "--", found.changelog)
                found.changelog_changed = diff is not None and diff.returncode == 1
    return found


def tree_fingerprint(project: Path) -> str:
    """A hash of the code as it is now: HEAD, the uncommitted paths and their sizes and mtimes
    (outside git: every project file's). A test run stamped with another fingerprint ran on
    different code, so it is not counted as passing."""
    digest = hashlib.sha1()
    head = _out(_git(project, "rev-parse", "HEAD"))
    status = _out(_git(project, "status", "--porcelain", "--untracked-files=all"))
    if status is not None:
        digest.update((head or "").encode())
        paths = [line[3:].strip().strip('"') for line in status.splitlines() if line.strip()]
    else:
        paths = [str(p.relative_to(project)) for p in sorted(project.rglob("*"))[:5000]
                 if p.is_file() and ".git" not in p.parts and ".vega" not in p.parts]
    for relative in paths:
        if relative.startswith(".vega/"):
            continue
        try:
            stat = (project / relative).stat()
            digest.update(f"{relative}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
        except OSError:
            digest.update(f"{relative}:gone\n".encode())
    return digest.hexdigest()


def _version_key(text: str) -> tuple[int, ...] | None:
    numbers = re.findall(r"\d+", text.lstrip("vV"))
    return tuple(int(n) for n in numbers[:4]) if numbers else None


# -- the checklist -------------------------------------------------------------------------------

def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def _tests_check(tests: Tests | None, command: str) -> Check:
    label = "Tests pass"
    run = f"`{tests.command}`" if tests else (f"`{command}`" if command else "the tests")
    if tests is None:
        return Check("tests", label, UNCHECKED, True, "not run yet",
                     "No test run this session.",
                     f"Run {run} and fix anything that fails.",
                     "Run the tests")
    if not tests.current:
        return Check("tests", label, UNCHECKED, True, "not run on this code",
                     f"The last run ({tests.command}) was on different code; files changed "
                     "since.", f"Run {run} on the current code and fix anything that fails.",
                     "Run the tests on this code")
    total = (tests.passed or 0) + (tests.failed or 0)
    if tests.ok:
        detail = f"{tests.passed} of {total}" if total else "pass"
        return Check("tests", label, PASS, True, detail, f"{tests.command}: {detail} pass.")
    names = ", ".join(tests.failures[:3])
    detail = f"{tests.failed} of {total} fail" if tests.failed and total else "fail"
    return Check("tests", label, FAIL, True, detail,
                 f"{tests.command}: {detail}." + (f" Failing: {names}." if names else ""),
                 f"The tests fail ({names or detail}). Fix the code so {run} passes again.",
                 f"Fix {_plural(tests.failed or 1, 'failing test')}")


def _task_check(task: TaskChecks | None) -> Check:
    label = "Task checks"
    if task is None or not task.items:
        return Check("task_checks", label, UNCHECKED, True, "none this session",
                     "Task checks run after Vega writes code for a task; none ran yet.")
    if not task.current:
        return Check("task_checks", label, UNCHECKED, True, "not run on this code",
                     "The last task checks ran before later changes.")
    failing = [(lbl, detail) for lbl, ok, detail in task.items if not ok]
    total = len(task.items)
    lines = "\n".join(f"{'✗' if not ok else '✓'} {lbl}" + (f": {d}" if d else "")
                      for lbl, ok, d in task.items)
    if not failing:
        return Check("task_checks", label, PASS, True, f"{total} of {total} pass", lines)
    what = "; ".join(f"{lbl}: {d}" if d else lbl for lbl, d in failing[:3])
    return Check("task_checks", label, FAIL, True, f"{len(failing)} of {total} fail", lines,
                 f"Fix what the task checks report: {what}.",
                 f"Fix {_plural(len(failing), 'failing task check')}")


def _secrets_check(secrets: tuple[list[str], list[str]] | None) -> Check:
    label = "No secrets exposed"
    if secrets is None:
        return Check("secrets", label, UNCHECKED, True, "needs git",
                     "Without git Vega cannot tell what would be published.")
    exposed, tracked = secrets
    if tracked:
        names = ", ".join(tracked[:3])
        it = "it" if len(tracked) == 1 else "them"
        return Check("secrets", label, FAIL, True, f"{_plural(len(tracked), 'file')} in git",
                     f"Committed secret files: {', '.join(tracked)}.",
                     f"{names} {'is' if len(tracked) == 1 else 'are'} committed to git. Stop "
                     f"tracking {it} with `git rm --cached` (the file stays on disk), add {it} to "
                     ".gitignore, and tell me which keys need rotating.",
                     f"Take {names} out of git")
    if exposed:
        names = ", ".join(exposed[:3])
        return Check("secrets", label, FAIL, True, f"{_plural(len(exposed), 'file')} not ignored",
                     f"Secret files that nothing ignores: {', '.join(exposed)}.",
                     f"Add {names} to .gitignore so {'it' if len(exposed) == 1 else 'they'} "
                     "can't be committed.", f"Ignore {names}")
    return Check("secrets", label, PASS, True, "",
                 "No .env, key or credential files are tracked or left unignored. (Keys pasted "
                 "inside files are not scanned yet.)")


def _changelog_check(git: Git, product: Facts) -> Check:
    label = "Changelog"
    if not git.repo:
        return Check("changelog", label, UNCHECKED, True, "needs git",
                     "The changelog is compared with the last release tag, which needs git.")
    target = product.version if product.version and (
        not git.tag or _version_key(product.version) != _version_key(git.tag)) else None
    if git.changelog is None:
        return Check("changelog", label, FAIL, True, "no changelog file",
                     "No CHANGELOG.md, CHANGES or HISTORY file at the project root.",
                     f"Create CHANGELOG.md with an entry for {target or 'this release'}: what "
                     "changed, in plain words.", "Write a changelog")
    mentions = bool(target) and re.search(rf"(?<![\w.]){re.escape(target or '')}(?![\w])",
                                          git.changelog_text) is not None
    if git.tag and not git.changelog_changed:
        entry = f"no {target} entry" if target else f"unchanged since {git.tag}"
        return Check("changelog", label, FAIL, True, entry,
                     f"{git.changelog} has not changed since {git.tag}.",
                     f"Write the {target or 'next'} entry in {git.changelog}: what changed since "
                     f"{git.tag}, in plain words.",
                     f"Write the {target + ' ' if target else ''}changelog entry")
    if target and not mentions:
        return Check("changelog", label, FAIL, True, f"no {target} entry",
                     f"{git.changelog} does not mention {target}.",
                     f"Write the {target} entry in {git.changelog}: what changed"
                     f"{' since ' + git.tag if git.tag else ''}, in plain words.",
                     f"Write the {target} changelog entry")
    detail = f"{target} entry" if target else (f"updated since {git.tag}" if git.tag
                                               else "present")
    return Check("changelog", label, PASS, True, detail, f"{git.changelog}: {detail}.")


def _version_check(git: Git, product: Facts) -> Check:
    label = "Version"
    if not git.repo:
        return Check("version", label, UNCHECKED, False, "needs git",
                     "The version is compared with the last release tag, which needs git.")
    if product.version is None:
        return Check("version", label, UNCHECKED, False, "no version found",
                     "pyproject.toml, package.json and Cargo.toml state no version.")
    if git.tag is None:
        return Check("version", label, UNCHECKED, False, f"{product.version}, no tag",
                     "No release tag yet to compare the version with.")
    new, old = _version_key(product.version), _version_key(git.tag)
    shown = f"{git.tag.lstrip('vV')} → {product.version}"
    if new is not None and old is not None and new > old:
        return Check("version", label, PASS, False, shown,
                     f"{product.version_file}: {product.version}; last tag {git.tag}.")
    return Check("version", label, FAIL, False, f"still {product.version}",
                 f"{product.version_file} says {product.version}, the same as (or older than) "
                 f"the last tag {git.tag}.",
                 f"Bump the version in {product.version_file} from {product.version} to the "
                 "next release.", "Bump the version")


def _committed_check(git: Git) -> Check:
    if not git.repo:
        return Check("committed", "Uncommitted", UNCHECKED, False, "needs git",
                     "This folder is not a git repository.")
    if not git.dirty:
        return Check("committed", "Everything committed", PASS, False, "clean",
                     "git status is clean.")
    n = len(git.dirty)
    return Check("committed", "Uncommitted", FAIL, False, _plural(n, "file"),
                 "Not committed yet:\n" + "\n".join(git.dirty[:20]),
                 f"Commit the {_plural(n, 'changed file')} with a message that says what "
                 "changed.", "Commit the changes")


def readiness(product: Facts, git: Git, secrets: tuple[list[str], list[str]] | None,
              tests: Tests | None, task: TaskChecks | None) -> list[Check]:
    """The release checklist, in the order it is shown."""
    checks = [
        _tests_check(tests, product.tests),
        _task_check(task),
        _secrets_check(secrets),
        Check("risky", "Risky code", UNCHECKED, False, "not checked yet",
              "The risky-code scan comes with the Security view (not built yet)."),
    ]
    if product.kind == "website":
        checks.append(Check("preview", "Preview renders", UNCHECKED, True, "not checked yet",
                            "The live preview is not built yet, so page errors are not checked."))
    checks += [
        _changelog_check(git, product),
        _version_check(git, product),
        _committed_check(git),
        Check("advisories", "Dependency advisories", UNCHECKED, False, "no data",
              "No offline advisory data yet (the Security view adds it)."),
    ]
    return checks


def gather(project: Path, tests: Tests | None, task: TaskChecks | None) -> Report:
    """Everything the Product and Release panes show (runs git: call it off the UI thread)."""
    from . import gitignore

    product = facts(project)
    git = git_facts(project)
    secrets = gitignore.scan(project) if git.repo else None
    return Report(product, readiness(product, git, secrets, tests, task), git, secrets)


__all__ = [
    "FAIL", "PASS", "UNCHECKED", "Check", "Facts", "Git", "Report", "Score", "TaskChecks",
    "Tests", "blockers", "facts", "gather", "git_facts", "kind", "readiness", "score",
    "tree_fingerprint", "verdict", "version",
]
