"""Fail when a workflow uses an action that is not pinned to a full commit SHA (CI "pinned" job).

Every `uses:` in the workflow files must look like

    uses: owner/repo[/path]@<40 hex characters> # vX.Y.Z

so a moved or re-pushed tag cannot change what runs. The trailing comment names the release, which
is what Dependabot reads and rewrites when it proposes an update. Local actions (`./path`) and
Docker images pinned by digest (`docker://image@sha256:<64 hex>`) are accepted too.

Usage: python .github/scripts/check_pinned_actions.py [FILE_OR_DIR ...]   (default .github/workflows)
Exit 0 when every action is pinned, 1 with one line per problem otherwise. Standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

USES = re.compile(r"""^\s*(?:-\s+)?uses:\s*(?P<ref>['"]?[^\s#'"]+['"]?)\s*(?P<comment>#.*)?$""")
PINNED = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[^@\s]+)?@[0-9a-f]{40}$")
DOCKER_DIGEST = re.compile(r"^docker://[^@\s]+@sha256:[0-9a-f]{64}$")
VERSION_COMMENT = re.compile(r"^#\s*v?\d+(?:\.\d+)*\S*")


def check_text(text: str, name: str) -> list[str]:
    """Problems in one workflow file's text, as `name:line: message`."""
    problems = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = USES.match(line)
        if not match:
            continue
        ref = match["ref"].strip("'\"")
        comment = (match["comment"] or "").strip()
        where = f"{name}:{number}"
        if ref.startswith("./") or DOCKER_DIGEST.match(ref):
            continue
        if not PINNED.match(ref):
            problems.append(f"{where}: {ref} is not pinned to a full commit SHA")
        elif not VERSION_COMMENT.match(comment):
            problems.append(f"{where}: {ref} has no '# vX.Y.Z' comment naming the release")
    return problems


def workflow_files(targets: list[Path]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        if target.is_dir():
            files += sorted(p for p in target.rglob("*") if p.suffix in (".yml", ".yaml"))
        else:
            files.append(target)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, default=[Path(".github/workflows")])
    args = parser.parse_args(argv)
    files = workflow_files(args.paths)
    if not files:
        print(f"no workflow files under {', '.join(map(str, args.paths))}")
        return 1
    problems = [p for f in files for p in check_text(f.read_text(encoding="utf-8"), str(f))]
    for problem in problems:
        print(problem)
    if not problems:
        print(f"{len(files)} workflow file(s): every action is pinned to a commit SHA")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
