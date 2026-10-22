"""Keep README.md, pyproject.toml and docs/release/keywords.md consistent (CI "keywords" job).

keywords.md is the source of truth. Its tables say where each phrase must appear ("Required in"
or "Where" column). This script checks the places that live in the repository:

  README H1               -> the README's first `# ` heading
  README first paragraph  -> the README's first prose paragraph after the H1
  description             -> pyproject.toml [project].description (the PyPI summary; the GitHub
                             About description is set by hand in the repository settings)
  topic `name`            -> pyproject.toml [project].keywords must include each topic named
                             anywhere in keywords.md ("local-llm" matches "local LLM")

and that the README carries no personal paths, email addresses or session links.

Matching is case-insensitive and ignores Markdown emphasis, links and code spans. A phrase may
have up to two words inserted ("small models" matches "small open-source models") and plural
endings ("open-weight LLM" matches "open-weight LLMs"). A cell listing alternatives with " / "
passes when any one appears; a cell listing phrases with ", " needs every one.

Usage: python .github/scripts/check_keywords.py [--readme README.md] [--pyproject pyproject.toml]
                                        [--keywords docs/release/keywords.md]
Exit 0 when consistent, 1 with one line per problem otherwise. Standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

PERSONAL = [
    (re.compile(r"/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+"), "a personal path"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "an email address"),
    (re.compile(r"claude\.ai/(?:code/)?session"), "a session link"),
]
ALLOWED_EMAILS = {"noreply@github.com"}


@dataclass
class Rule:
    phrase: str  # the cell as written, e.g. "open-weight LLM / open-source LLM"
    where: str  # the "Required in" / "Where" cell


def parse_rules(text: str) -> list[Rule]:
    """Rows of every table whose second column is "Required in" or "Where"."""
    rules: list[Rule] = []
    in_table = False
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not line.lstrip().startswith("|") or len(cells) < 2:
            in_table = False
            continue
        if cells[0].lower() == "phrase":
            in_table = cells[1].lower() in ("required in", "where")
            continue
        if in_table and not set(cells[0]) <= set("-: "):
            rules.append(Rule(cells[0], cells[1]))
    return rules


def topics(text: str) -> list[str]:
    """Every topic named as "topic `x`" or "topics `x`, `y`" in keywords.md."""
    groups = re.findall(r"\btopics? ((?:`[a-z0-9-]+`(?:,\s*)?)+)", text)
    return sorted({name for group in groups for name in re.findall(r"`([a-z0-9-]+)`", group)})


def plain(markdown: str) -> str:
    """Markdown to the text a reader sees."""
    text = re.sub(r"<!--.*?-->", " ", markdown, flags=re.DOTALL)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def readme_parts(markdown: str) -> tuple[str, str]:
    """(H1 text, first prose paragraph after it)."""
    without_comments = re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL)
    lines = without_comments.splitlines()
    h1_index = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
    if h1_index is None:
        return "", ""
    paragraph: list[str] = []
    for line in lines[h1_index + 1:]:
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith(("#", "[![", "<", "```")) and not paragraph:
            continue
        paragraph.append(stripped)
    return plain(lines[h1_index][2:]), plain(" ".join(paragraph))


def _pattern(phrase: str) -> re.Pattern[str]:
    words = re.findall(r"[\w.+-]+", phrase.lower())
    parts = [re.escape(word) + r"(?:e?s)?" for word in words]
    gap = r"(?:\W+[\w'.+-]+){0,2}?\W+"
    return re.compile(r"(?<![\w-])" + gap.join(parts) + r"(?![\w-])")


def present(cell: str, text: str) -> bool:
    """Every ", " part of the cell is found, each as any one of its " / " alternatives."""
    haystack = text.lower()
    return all(any(_pattern(alt).search(haystack) for alt in part.split(" / "))
               for part in cell.split(", "))


def check(readme_text: str, project: dict, keywords_text: str) -> list[str]:
    problems: list[str] = []
    h1, first = readme_parts(readme_text)
    if not h1:
        problems.append("README: no `# ` heading found")
    if not first:
        problems.append("README: no first paragraph found after the H1")
    description = str(project.get("description", ""))
    rules = parse_rules(keywords_text)
    if not rules:
        problems.append("keywords.md: no phrase tables found (format changed?)")
    for rule in rules:
        where = rule.where.lower()
        for place, text, label in (("readme h1", h1, "README H1"),
                                   ("readme first paragraph", first, "README first paragraph"),
                                   ("description", description, "pyproject description")):
            if place in where and not present(rule.phrase, text):
                problems.append(f"{label}: missing {rule.phrase!r}")
    keywords = {re.sub(r"[\s_]+", "-", str(k).strip().lower()) for k in project.get("keywords", [])}
    for topic in topics(keywords_text):
        if topic not in keywords:
            problems.append(f"pyproject keywords: missing topic {topic!r} (as {topic.replace('-', ' ')!r})")
    for pattern, what in PERSONAL:
        for match in pattern.finditer(readme_text):
            if match.group(0).lower() not in ALLOWED_EMAILS:
                problems.append(f"README: contains {what}: {match.group(0)!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--keywords", type=Path, default=Path("docs/release/keywords.md"))
    args = parser.parse_args(argv)
    try:
        readme = args.readme.read_text(encoding="utf-8")
        project = tomllib.loads(args.pyproject.read_text(encoding="utf-8")).get("project", {})
        keywords = args.keywords.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"check_keywords: cannot read {exc.filename}: {exc.strerror}")
        return 1
    problems = check(readme, project, keywords)
    for problem in problems:
        print(problem)
    if problems:
        print(f"{len(problems)} problem(s); keywords.md is the source of truth.")
        return 1
    print(f"Keywords consistent: {len(parse_rules(keywords))} phrases, "
          f"{len(topics(keywords))} topics checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
