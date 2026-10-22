"""Teach mode: Vega says why while it works, from its own actions (the user's request, 2026-10-16).

The harness never writes an explanation for the model. In the agent loop the model is asked to
state its plan, one "why" line before each change or check, what a failing check means, and to end
with "What to take away"; the harness only shows that text, keeps it out of the model's later
context (the plan line stays), and sends a note back once when the text names a file, function or
test that this turn's tool calls and results never showed. The pipelines, which do not stream tool
calls, add a note built from their real stages (planned files, check results, excerpts read); the
direct pipeline's writer also ends its reply with its own takeaway.

Off by default. On with `/teach on` (saved in ~/.vega/view.json), `vega --teach` or VEGA_TEACH=1.
"""

from __future__ import annotations

import builtins
import json
import keyword
import os
import re
import sys
from collections.abc import Iterable
from typing import Any

ENV = "VEGA_TEACH"

_SECTION_BASE = """

Teach mode: the user is learning from how you work. Keep doing the task as well as you can, and:
- Before your first tool call, state your plan and why that approach, in 2-4 sentences.
- Before each edit_file, replace_lines, write_file or run_check call, write one short line starting
  "Why:" that says why this step.
- After a failing check, say in 1-2 sentences what the failure means and why your next change fixes it.
- End your final answer with a section headed "## What to take away": {takeaway}
Name only files, functions and tests that this task's tool results showed you."""
_TAKEAWAY_PLAIN = ("each key decision with the\n  alternative you rejected and the trade-off, the "
                   "concepts involved (named precisely), and\n  optionally one question that "
                   "checks understanding.")
# teach_root_cause (2026-10-17): the writer's takeaways named the fix as the concept
# ("Decimal precision", "default argument binding") and never the mechanism that broke the code.
_TAKEAWAY_ROOT_CAUSE = (
    "for a bug fix, first the root\n  cause: the language or runtime mechanism that made the old "
    "code wrong, named with its\n  precise term (e.g. \"a shallow copy shares the nested lists\"), "
    "not the fix; then the fix and why it\n  works; then the alternative you rejected and the "
    "trade-off. For other work: each key decision\n  with the alternative you rejected and the "
    "trade-off. Then the concepts involved (named precisely),\n  and optionally one question "
    "that checks understanding.")

# The direct pipeline's writer, in teach mode, ends its reply with its own takeaway.
_WRITER_PLAIN = (
    " After the code block, add a short section headed \"## What to take away\" (at most 6 lines): "
    "the key decision with the alternative you rejected and the trade-off, and the concepts "
    "involved, named precisely; optionally one question that checks understanding. Name only "
    "files and functions shown here."
)
_WRITER_ROOT_CAUSE = (
    " After the code block, add a short section headed \"## What to take away\" (at most 6 lines). "
    "If the task fixes a bug, write it in this order: a line starting \"Root cause:\" that names "
    "the language or runtime mechanism that made the old code wrong, with its precise term (e.g. "
    "\"a shallow copy shares the nested lists\"), not the fix; a line starting \"Fix:\" with "
    "what you changed and why it works; a line starting \"Rejected:\" with the alternative you "
    "rejected (one of the task's candidates, if it lists any) and why. Otherwise: the key decision "
    "with the alternative you rejected and the trade-off, and the concepts involved, named "
    "precisely. Optionally end with one question that checks understanding. Name only files and "
    "functions shown here."
)


def __getattr__(name: str) -> str:
    """SECTION and WRITER_LINE follow the teach_root_cause switch, read at each use (the agent
    adds and later removes the same SECTION text, so it stays one value within a session)."""
    from . import features

    if name == "SECTION":
        return _SECTION_BASE.format(takeaway=_TAKEAWAY_ROOT_CAUSE if features.on("teach_root_cause")
                                    else _TAKEAWAY_PLAIN)
    if name == "WRITER_LINE":
        return _WRITER_ROOT_CAUSE if features.on("teach_root_cause") else _WRITER_PLAIN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


TAKEAWAY_HEADING = "What to take away"
STAGES_HEADING = "How this was done"
_HEADING = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*|\*\*)?(?:What to take away|How this was done)\b[^\n]*$",
    re.IGNORECASE | re.MULTILINE)
_TAKEAWAY = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*|\*\*)?What to take away\b[^\n]*$",
                       re.IGNORECASE | re.MULTILINE)


def from_env() -> bool:
    return os.environ.get(ENV, "").strip().lower() in {"1", "on", "true", "yes"}


# -- the answer's teaching sections ---------------------------------------------------------------

def split(answer: str) -> tuple[str, str]:
    """(the answer, its teaching sections) split at the first "How this was done" or "What to
    take away" heading; ('answer', '') when there is none."""
    match = _HEADING.search(answer or "")
    if match is None:
        return answer, ""
    return answer[:match.start()].rstrip(), answer[match.start():].strip()


def takeaway(answer: str) -> str:
    """The "What to take away" section (to the end of the answer), or ''."""
    match = _TAKEAWAY.search(answer or "")
    return answer[match.start():].strip() if match else ""


def strip(answer: str) -> str:
    """The answer without its teaching sections, for the model's later context."""
    body, sections = split(answer)
    return body if sections else answer


def strip_history(messages: list[dict[str, Any]]) -> int:
    """Remove teaching sections from the final answers in the history (in place); returns the
    characters removed. Tool-call messages already had their why lines removed when stored."""
    removed = 0
    for message in messages:
        if message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        content = message.get("content")
        if isinstance(content, str) and _HEADING.search(content):
            kept = strip(content)
            removed += len(content) - len(kept)
            message["content"] = kept
    return removed


# -- the loop's teaching lines ---------------------------------------------------------------------

def role(calls_before: int, failed_check: bool) -> str:
    """What a reply's text alongside tool calls is: the plan (before the turn's first call), an
    explanation of a failing check, or a why line."""
    if calls_before == 0:
        return "plan"
    return "failure" if failed_check else "why"


def clean_line(text: str) -> str:
    """A teaching line as shown: whitespace collapsed, a leading "Why:" label dropped (the view
    labels it)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    return re.sub(r"^(?:\*\*)?why(?:\*\*)?\s*[:\-—]\s*(?:\*\*)?", "", text, flags=re.IGNORECASE)


# -- grounding ------------------------------------------------------------------------------------

_FILE = re.compile(r"(?<![\w/.-])([\w-]+(?:/[\w.-]+)*\.(?:py|pyi|toml|cfg|ini|json|ya?ml|md|txt|"
                   r"csv|js|ts|sh|rs|go))\b")
_TEST = re.compile(r"\btest_\w+")
_BACKTICK = re.compile(r"`([A-Za-z_][\w.]*)(?:\(\))?`")
_CALL = re.compile(r"(?<![\w.`])([A-Za-z_][\w.]*)\(")
_KNOWN = (set(dir(builtins)) | set(keyword.kwlist) | set(sys.stdlib_module_names)
          | {"self", "cls", "pytest", "print", "main"})


def _function_name(token: str) -> str | None:
    """The project name a code token points at, or None for builtins, keywords, the standard
    library, methods of those (`dict.get`, `os.path.join`) and very short names."""
    parts = token.strip(".").split(".")
    if not parts or not parts[-1] or parts[0] in _KNOWN or parts[-1] in _KNOWN:
        return None
    name = parts[-1]
    if len(name) < 3:
        return None
    return name


def named(text: str) -> list[str]:
    """Files, tests and functions a teaching text names (in order, once each)."""
    found: list[str] = []
    for match in _FILE.finditer(text):
        found.append(match.group(1))
    found += _TEST.findall(text)
    for pattern in (_BACKTICK, _CALL):
        for match in pattern.finditer(text):
            token = match.group(1)
            if _FILE.fullmatch(token):
                continue
            name = _function_name(token)
            if name:
                found.append(name)
    return list(dict.fromkeys(found))


def corpus(parts: Iterable[Any]) -> str:
    """What this turn showed: the request, tool call arguments and results, as one text."""
    return "\n".join(part if isinstance(part, str) else json.dumps(part, default=str)
                     for part in parts)


def ungrounded(text: str, seen: str) -> list[str]:
    """Names in `text` that do not appear in `seen` (a file also counts by its base name)."""
    missing = []
    for name in named(text):
        if name in seen or "/" in name and name.rsplit("/", 1)[1] in seen:
            continue
        missing.append(name)
    return missing


def grounding_note(names: list[str], where: str) -> str:
    """Feedback, not an override: the model corrects its own explanation (claim_check style)."""
    shown = ", ".join(f"`{name}`" for name in names[:5])
    if where == "takeaway":
        return (f"(Vega harness) Your \"What to take away\" names {shown}, which this turn's tool "
                "calls and results never showed. Correct the takeaway so it names only what you "
                "actually read, changed or ran, then give your final answer again.")
    return (f"(Vega harness) Your note before this call names {shown}, which this turn's tool "
            "calls and results never showed. Explain only from what the tools showed; correct "
            "it in your next note.")


# -- the pipelines' notes --------------------------------------------------------------------------

def _clip(text: str, limit: int = 110) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def direct_note(relative: str, attempts: list[tuple[int, int, list[str]]], kept: int) -> str:
    """From the direct pipeline's real stages: each attempt's (passed, total, failures) and which
    attempt's code was kept (index)."""
    lines = [f"## {STAGES_HEADING}",
             (f"- Route: a task about one self-contained file, `{relative}`, so Vega had the model "
              "write it whole, ran its docstring examples and tests, and repaired on failures.")]
    for number, (passed, total, failures) in enumerate(attempts, 1):
        result = f"{passed}/{total} checks passed" if total else "no examples or tests to run"
        line = f"- Attempt {number}: {result}"
        if failures:
            line += f"; first failure: {_clip(failures[0])}"
        if len(attempts) > 1 and number - 1 == kept:
            line += " (kept)"
        lines.append(line)
    return "\n".join(lines)


def project_note(plan: list[dict[str, str]], rounds: list[str]) -> str:
    """From the project pipeline's real stages: the files the model planned (with the change it
    stated for each) and each round's test result."""
    lines = [f"## {STAGES_HEADING}",
             ("- Route: a change across the project, so Vega had the model plan the files from the "
              "repo map, wrote each one, ran the project's tests and repaired on failures."),
             "- Planned by the model:"]
    lines += [f"  - `{item['path']}`: {_clip(item.get('change', ''), 90)}" for item in plan]
    lines += [f"- {_clip(line[0].upper() + line[1:], 120)}" for line in rounds]
    return "\n".join(lines)


_EXCERPT_HEAD = re.compile(r"^(\S+) \(?(whole file|lines \d+-\d+)\)?:$", re.MULTILINE)


def question_note(shown: str) -> str:
    """From the question pipeline's real stage: the excerpts the answer was written from."""
    places = [f"`{path}` ({where})" for path, where in _EXCERPT_HEAD.findall(shown)]
    if not places:
        return ""
    return (f"## {STAGES_HEADING}\n- Answered in one model call from these excerpts, found by "
            f"the file names in the question and the code index: {', '.join(places[:8])}"
            + (f" and {len(places) - 8} more" if len(places) > 8 else "") + ".")
