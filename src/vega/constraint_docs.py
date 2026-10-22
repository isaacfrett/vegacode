"""constraint_docs: the repo documents a task names as its constraints, read before the first
write and quoted back at the finish of a decision.

Why (2026-10-19): a decision task can hinge on a rule written in a repo document the task only
points at. Without this, the model sometimes never opened that document, or opened it and still
chose an option the rule excludes. Rules only, no model call:

- named(): a document the task points at with a constraint cue in the words before it
  ("following SECURITY.md", "under docs/RULES.md", "against the guidelines in docs/",
  "It must meet docs/PLATFORM.md", "Using the rules in CONTRIBUTING.md"; not "see NOTICE.md").
  Documents only (.md/.txt/.rst/.adoc, or an extensionless upper-case name like POLICY); a folder
  stands for the documents under it (at most MAX_FOLDER_DOCS, else none).
- hold_note(): the agent loop's first write is not run once while one of them is unread.
- rule_lines() / quote_note(): at the finish of a decision task, the documents' rule sentences
  ("must", "only", "not allowed", "at least"...) with the value chosen, once.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

MAX_FOLDER_DOCS = 6
MAX_RULES = 8
_DOC_SUFFIXES = {".md", ".txt", ".rst", ".adoc"}
# a path or a folder (`docs/`) as the task writes it
_PATH = re.compile(r"(?<![\w/.-])`?((?:[\w.-]+/)+|(?:[\w.-]+/)*[\w-][\w.-]*)`?(?![\w-])")
_CUE = re.compile(
    r"\b(?:follow(?:s|ing)?|under|against|meets?|meeting|per|according to|in line with|"
    r"compl(?:y|ies|ying) with|conform(?:s|ing)? to|satisf(?:y|ies|ying)|respect(?:s|ing)?|"
    r"obey(?:s|ing)?|rules?|constraints?|polic(?:y|ies)|guidelines?|conventions?|"
    r"requirements?)\b", re.IGNORECASE)
_RULE = re.compile(
    r"\b(?:must|should|shall|only|never|always|not|no|at least|at most|oldest|newest|lowest|"
    r"highest|earliest|latest|prefer(?:red)?|required?|allowed|forbidden|rejected|choose|pick|"
    r"stays?|until|minimum|maximum|exactly)\b", re.IGNORECASE)
_DECIDE = re.compile(r"\b(?:choose|chose|choice|pick|decide|decision|select)\b", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _is_doc(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    suffix = os.path.splitext(name)[1].lower()
    return suffix in _DOC_SUFFIXES or (not suffix and name.isupper() and len(name) > 3)


def _folder_docs(project: Path, folder: str) -> list[str]:
    base = project / folder
    found = []
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if len(Path(root).relative_to(base).parts) > 1:
            dirs[:] = []
            continue
        for name in sorted(files):
            relative = (Path(root) / name).relative_to(project).as_posix()
            if _is_doc(relative):
                found.append(relative)
    return found if len(found) <= MAX_FOLDER_DOCS else []


def named(task: str, project: Path) -> list[str]:
    """The project documents the task names as constraints, in task order."""
    project = Path(project)
    found: list[str] = []
    for match in _PATH.finditer(task or ""):
        raw = match.group(1).rstrip(".")
        if ".." in raw or raw.startswith("/") or "/" not in raw and "." not in raw \
                and not raw.isupper():
            continue
        sentence = re.split(r"(?<=[.;!?])\s|\n", task[max(0, match.start() - 80):match.start()])[-1]
        if not _CUE.search(sentence):
            continue
        relative = raw.rstrip("/")
        target = project / relative
        if target.is_dir() and raw.endswith("/"):
            found += _folder_docs(project, relative)
        elif target.is_file() and _is_doc(relative):
            found.append(relative)
    return list(dict.fromkeys(found))


def unread(paths: list[str], read: set[str]) -> list[str]:
    return [path for path in paths if os.path.normpath(path) not in read]


def hold_note(paths: list[str]) -> str:
    return ("(Vega harness) Not written yet: the task says to follow " +
            ", ".join(f"`{path}`" for path in paths) + ", and you have not read "
            + ("it" if len(paths) == 1 else "them") + " this turn. read_file "
            + ("it" if len(paths) == 1 else "each one") + " first: it states the rules your "
            "change and your decision must meet (allowed licences, which version to choose, a "
            "freeze). Then make the change again, adjusted to those rules.")


def _sentences(text: str) -> list[str]:
    """Sentences of the document's prose and bullets (wrapped lines joined, headings, tables and
    code left out)."""
    units: list[str] = []
    current = ""
    fenced = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
            continue
        if fenced or not stripped or stripped.startswith(("#", "|")) or _BULLET.match(line):
            if current:
                units.append(current)
            current = _BULLET.sub("", line).strip() if _BULLET.match(line) and not fenced else ""
            continue
        current = f"{current} {stripped}".strip()
    if current:
        units.append(current)
    out = []
    for unit in units:
        out += [piece.strip() for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z(*`])", unit)
                if piece.strip()]
    return out


def rule_lines(project: Path, paths: list[str]) -> list[tuple[str, str]]:
    """(document, sentence) for each sentence stating a rule, at most MAX_RULES."""
    found: list[tuple[str, str]] = []
    for path in paths:
        try:
            text = (Path(project) / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for sentence in _sentences(text):
            if _RULE.search(sentence):
                found.append((path, sentence[:240]))
    return found[:MAX_RULES]


def is_decision(task: str, template_keys: list[str]) -> bool:
    """A task that asks to choose between options and record it: its JSON template has a
    `choice` key, and it says choose/pick/decide/choice."""
    return "choice" in template_keys and bool(_DECIDE.search(task or ""))


def chosen(project: Path, files: list[str]) -> list[str]:
    """`"choice": value` of each JSON file written this turn that has one."""
    out = []
    for relative in dict.fromkeys(files):
        if not relative.endswith(".json"):
            continue
        try:
            data = json.loads((Path(project) / relative).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if isinstance(data, dict) and "choice" in data:
            out.append(f'{relative} "choice": {json.dumps(data["choice"])}')
    return out


def quote_note(rules: list[tuple[str, str]], choices: list[str]) -> str:
    choice = ("You chose " + "; ".join(choices) + ". ") if choices else ""
    return ("(Vega harness) Check your decision against the rules the task names before you "
            f"finish. {choice}The rules:\n"
            + "\n".join(f"- {path}: {line}" for path, line in rules)
            + "\nGo through them one by one. If your choice (or a value, version or file you "
            "wrote) breaks one, fix the files now. If it meets every rule, give your final answer "
            "again and say which rule decided the choice.")
