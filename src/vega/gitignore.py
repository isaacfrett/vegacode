"""Keeping secrets and Vega's own files out of git.

Vega owns one marked block in a repo's .gitignore (between GITIGNORE_BEGIN and GITIGNORE_END) and
never edits any other line. The block always holds `.vega/` (session digests, notes, state) plus
patterns for sensitive files found in the work tree that nothing ignores yet (.env files, private
keys, credential files). Templates such as `.env.example` stay tracked. A sensitive file that is
already committed cannot be fixed by an ignore rule: Vega reports it and suggests
`git rm --cached`, but never runs it. Autopilot unstages sensitive files before it commits.
"""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path, PurePosixPath

GITIGNORE_BEGIN = "# >>> vega: managed by Vega (edit outside this block; remove it and Vega won't re-add it)"
GITIGNORE_END = "# <<< vega"
VEGA_ENTRIES = (".vega/",)  # VEGA.md stays tracked: it is meant to be shared and edited

# (pattern matched against a path's file name, ignore rule written for it)
SENSITIVE = (
    (".env", ".env"),
    (".env.*", ".env.*"),
    ("*.pem", "*.pem"),
    ("*.key", "*.key"),
    ("*.p12", "*.p12"),
    ("*.pfx", "*.pfx"),
    ("*.keystore", "*.keystore"),
    ("id_rsa*", "id_rsa*"),
    ("id_ed25519*", "id_ed25519*"),
    ("id_ecdsa*", "id_ecdsa*"),
    ("credentials.json", "credentials.json"),
    ("service-account*.json", "service-account*.json"),
    ("secrets.*", "secrets.*"),
    (".netrc", ".netrc"),
    (".pgpass", ".pgpass"),
)
TEMPLATES = (".env.example", ".env.sample", ".env.template", ".env.dist", "*.pub")


def is_sensitive(path: str) -> bool:
    """A file that usually holds secrets, judged by its name (templates and public keys excluded)."""
    name = PurePosixPath(path).name
    if any(fnmatch.fnmatch(name, pattern) for pattern in TEMPLATES):
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern, _ in SENSITIVE)


def rules_for(paths: list[str]) -> list[str]:
    """Ignore rules covering these sensitive paths, in SENSITIVE order, each once."""
    names = [PurePosixPath(path).name for path in paths]
    return [rule for pattern, rule in SENSITIVE if any(fnmatch.fnmatch(name, pattern) for name in names)]


def _git(project: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(["git", "-C", str(project), *args], capture_output=True, text=True,
                              timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def in_work_tree(project: Path) -> bool:
    result = _git(project, "rev-parse", "--is-inside-work-tree")
    return bool(result and result.returncode == 0 and result.stdout.strip() == "true")


def _lines(result: subprocess.CompletedProcess[str] | None) -> list[str]:
    if not result or result.returncode:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def scan(project: Path) -> tuple[list[str], list[str]]:
    """(sensitive files that are untracked and not ignored, sensitive files already tracked)."""
    exposed = [p for p in _lines(_git(project, "ls-files", "--others", "--exclude-standard")) if is_sensitive(p)]
    tracked = [p for p in _lines(_git(project, "ls-files")) if is_sensitive(p)]
    return exposed, tracked


def vega_dir_ignored(project: Path) -> bool:
    result = _git(project, "check-ignore", "-q", ".vega/sessions.md")
    return bool(result and result.returncode == 0)


def block_text(text: str) -> str | None:
    """The managed block's body (without markers), or None when there is no block."""
    if GITIGNORE_BEGIN not in text or GITIGNORE_END not in text:
        return None
    return text.split(GITIGNORE_BEGIN, 1)[1].split(GITIGNORE_END, 1)[0]


def write_block(project: Path, entries: list[str]) -> None:
    """Create or replace Vega's block; every line outside it stays as it was."""
    path = project / ".gitignore"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    block = "\n".join([GITIGNORE_BEGIN, *entries, GITIGNORE_END]) + "\n"
    if block_text(text) is not None:
        before, rest = text.split(GITIGNORE_BEGIN, 1)
        after = rest.split(GITIGNORE_END, 1)[1].lstrip("\n")
        new = before + block + (("\n" + after) if after else "")
    else:
        separator = "" if not text or text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        new = text + separator + block
    path.write_text(new, encoding="utf-8")


def unstage_sensitive(root: Path) -> list[str]:
    """After `git add`: unstage newly added sensitive files (they stay on disk). Returns them."""
    staged = _lines(_git(root, "diff", "--cached", "--name-only", "--diff-filter=A"))
    risky = [path for path in staged if is_sensitive(path)]
    if risky:
        _git(root, "reset", "-q", "--", *risky)
    return risky
