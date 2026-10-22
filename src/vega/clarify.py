"""Ask before building when a request leaves the decisions open (switch `clarify_first`).

Live evidence (2026-10-15, debug log 20261015-163621-027-chat-vega): "Could you build an LLM
workflow, verify it end to end, and setup a loop around watching and measuring it for data
analysis." in the Vega repo. Vega asked nothing; it searched "workflow", "LLM workflow" and "data
analysis loop", read an unrelated scheduler, and after compaction repeated the same searches and
started running the whole test suite. Nothing in the request said what to build, where, or how to
check it.

A gate at the start of a new top-level turn classifies the request with cheap rules:

- `clear`: not a build/change request, or it names something in the code (a path, a backticked
  or code-shaped name, a slash command), or it is long or concrete (several numbers, quoted
  values, "from X to Y"). Nearly every request takes this path, with no model call and no delay.
- `vague`: an open-ended thing to build ("a workflow", "an app", "a pipeline", "a dashboard", ...)
  with nothing that pins it down, or a short request whose only object is "it"/"this" with no
  earlier conversation to resolve it. Vega replies with 2-4 numbered questions whose options come
  from the project (data files, modules, the test command); no tool runs, no file is written.
- `unsure`: a short change request with no anchor ("Add low-stock alerts to the inventory."). The
  project may settle it (a README naming the export format) or not. With the switch
  `clarify_model` (default on since 2026-10-17) one short model call with a constrained JSON reply
  decides, seeing the README, the project context loaded at session start (VEGA.md, session
  digests, notes) and the file list; with it off the turn goes on as before.

Follow-ups are never gated twice: the answer to Vega's questions goes straight to the agent, with
the questions and the original request in the history.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import debuglog, features, verbs

if TYPE_CHECKING:
    from .agent import CodingAgent

CLEAR, VAGUE, UNSURE = "clear", "vague", "unsure"
LONG_TASK = 250  # characters: past this a request carries its own detail (goals tasks are > 410)
SHORT_TASK = 160  # unsure only below this: longer unanchored requests describe behaviour
PRONOUN_TASK = 60

# Something in the code the request points at: these settle "where" and usually "what".
_ANCHOR = re.compile(
    r"`[^`]+`"
    r"|\b[\w./-]+\.(?:py|pyi|toml|json|jsonl|ya?ml|cfg|ini|md|txt|csv|tsv|parquet|js|jsx|ts|tsx|sh"
    r"|sql|go|rs|java|kt|c|h|cc|cpp|hpp|swift|rb|cs|php|html|css|gd|tscn)\b"
    r"|\b[A-Za-z_]\w*\("  # a call: name(
    r"|\b[a-z][a-z0-9]*_[a-z0-9_]+\b"  # snake_case
    r"|\b[a-z]+[A-Z]\w*\b|\b[A-Z][a-z0-9]+[A-Z]\w*\b"  # camelCase, CamelCase
    r"|\b[A-Za-z_]\w*\.[A-Za-z_]\w+\b"  # dotted.name
    r"|\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b"  # CONSTANT_NAME
    r"|(?:^|\s)/[a-z][\w-]+"  # a slash command or absolute path
    r"|\b(?:line|lines)\s+\d+|Traceback|Error:|Exception\b",
)
# Concrete values: each one narrows the request (counted separately, so "from 8080 to 3000" is
# three: the change and its two values).
_SPECIFICS = (
    re.compile(r"\d+(?:[.,:/]\d+)*%?"),
    re.compile(r"\bfrom\s+\S+\s+to\s+\S+", re.IGNORECASE),
    re.compile(r"\"[^\"]+\"|“[^”]+”"),
    re.compile(r"\b(?:e\.g\.|i\.e\.|for example|for instance|such as)\b", re.IGNORECASE),
    re.compile(r"\b[A-Z][A-Z0-9]{3,}\b"),  # a constant or setting name: CURRENCY, README
)


def _specifics(text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in _SPECIFICS)
_CHANGE = re.compile(
    r"\b(?:build|create|make|set\s*up|setup|design|develop|write|implement|add|scaffold|generate"
    r"|put together|spin up|wire up|automate|fix|change|update|remove|delete|clear|refactor"
    r"|improve|optimi[sz]e|speed up|clean up|support|handle|convert|migrate|rewrite|redo)\b",
    re.IGNORECASE,
)
_BUILD = re.compile(
    r"\b(?:build|create|make|set\s*up|setup|design|develop|write|implement|add|scaffold|generate"
    r"|put together|spin up|wire up|automate)\b",
    re.IGNORECASE,
)
_OPEN_NOUNS = (
    r"workflows?|apps?|applications?|systems?|pipelines?|dashboards?|platforms?|frameworks?"
    r"|services?|bots?|websites?|sites?|web\s*apps?|games?|tools?|loops?|agents?|uis?|guis?"
    r"|interfaces?|backends?|frontends?|programs?|solutions?|infrastructure|monitoring|monitors?"
    r"|trackers?|integrations?|engines?|projects?|products?|automations?"
)
# "build an LLM workflow", "set up a loop", "make me a dashboard": a generic determiner, up to three
# words, then an open-ended noun.
_OPEN_BUILD = re.compile(
    r"\b(?:a|an|some|me\s+an?|my|new|the\s+new)\s+(?:[\w'-]+\s+){0,3}?(?:" + _OPEN_NOUNS + r")\b",
    re.IGNORECASE,
)
_DESCRIBED = re.compile(r"\s+(?:that|which|who)\s+(?!is\b|are\b)\w+(?:\s+\w+){2,}", re.IGNORECASE)
_PRONOUN_ONLY = re.compile(
    r"^\W*(?:please\s+)?(?:make|fix|improve|speed\s+up|clean\s+up|tidy\s+up|refactor|optimi[sz]e"
    r"|update|change|redo|finish|rewrite|speed|clean|tidy)\s+(?:it|this|that|things|everything"
    r"|stuff)\b",
    re.IGNORECASE,
)
_CONSTRAINT = re.compile(r"\b(?:Do not|Don't|Only report)[^.!?]*[.!?]", re.IGNORECASE)
_MONITOR = re.compile(r"\b(?:loop|watch\w*|monitor\w*|measur\w*|track\w*|metrics?|schedul\w*)\b",
                      re.IGNORECASE)
_VERIFY = re.compile(r"\b(?:verify|verif\w+|test\w*|check\w*|end to end|e2e|validat\w*)\b",
                     re.IGNORECASE)


@dataclass
class Verdict:
    kind: str  # CLEAR, VAGUE or UNSURE
    reasons: list[str] = field(default_factory=list)


def classify(task: str, *, follow_up: bool = False) -> Verdict:
    """Rules only: is this request clear enough to act on, too vague, or for the model to decide?
    `follow_up`: earlier turns in this conversation can resolve "it" and short references."""
    from .question_pipeline import is_conversation, is_question
    from .routing import strip_code

    text = _CONSTRAINT.sub("", task).strip()
    if not text:
        return Verdict(CLEAR, ["empty"])
    masked = strip_code(text)  # verbs in request position, outside code (see verbs)
    if not verbs.requested(_CHANGE, masked):
        return Verdict(CLEAR, ["no build or change verb"])
    if (is_question(text) or is_conversation(text)) and not verbs.requested(_BUILD, masked):
        return Verdict(CLEAR, ["question"])
    anchor = _ANCHOR.search(text)
    if anchor:
        return Verdict(CLEAR, [f"names {anchor.group(0).strip()!r}"])
    specifics = _specifics(text)
    if specifics >= 2:
        return Verdict(CLEAR, [f"{specifics} concrete values"])
    if len(text) >= LONG_TASK:
        return Verdict(CLEAR, [f"long ({len(text)} characters)"])
    if _PRONOUN_ONLY.search(text) and len(text) < PRONOUN_TASK:
        if follow_up:
            return Verdict(CLEAR, ["refers to the conversation"])
        return Verdict(VAGUE, ["its only object is 'it/this' and there is no earlier conversation"])
    build = _BUILD.search(text)
    thing = _OPEN_BUILD.search(text)
    # "a tool that counts lines in every Python file": the clause says what it does
    described = bool(thing and _DESCRIBED.match(text[thing.end():]))
    if build and thing and specifics == 0 and not described:
        reason = (f"open-ended {thing.group(0).strip()!r}, no file, module, input or output "
                  "named")
        return Verdict(VAGUE, [reason])
    if len(text) < SHORT_TASK and not follow_up:
        return Verdict(UNSURE, ["short change request naming nothing in the code"])
    return Verdict(CLEAR, ["no vague signal"])


# -- suggestions from the project ------------------------------------------------------------------

_DATA_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".xlsx", ".sqlite", ".db"}
_SKIP_PARTS = {"node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".git"}
_WORD = re.compile(r"[a-z]{4,}")
_STOP = {"could", "would", "build", "make", "create", "setup", "around", "with", "that", "this",
         "from", "into", "verify", "data", "watching", "measuring", "please", "some",
         "should", "about", "them", "then", "what", "your", "have", "test", "tests", "file",
         "files", "end", "able", "want", "need", "using", "also", "like", "thing", "things"}


@dataclass
class ProjectFacts:
    data_files: list[str]
    modules: list[str]
    related: list[str]
    test_command: str | None


def project_facts(project: Path, task: str, walk_limit: int = 4000) -> ProjectFacts:
    from .tools import _project_files

    words = {w for w in _WORD.findall(task.lower()) if w not in _STOP}
    words |= {w + "s" for w in words}  # "loop" finds loops/
    data, modules, related = [], [], []
    has_tests = False
    for path in _project_files(project, limit=walk_limit):
        rel = path.relative_to(project)
        if _SKIP_PARTS & set(rel.parts):
            continue
        posix = rel.as_posix()
        if rel.parts[0] in {"tests", "test"} or rel.name.startswith("test_"):
            has_tests = has_tests or path.suffix == ".py"
            continue
        if path.suffix in _DATA_SUFFIXES and len(data) < 40:
            data.append(posix)
        if path.suffix == ".py" and rel.name != "__init__.py":
            modules.append(posix)
            stem_words = set(re.split(r"[\W_]+", posix.lower().removesuffix(".py")))
            if words & stem_words:
                related.append(posix)
    test_command = None
    if has_tests or (project / "pytest.ini").exists() or "pytest" in _read(project / "pyproject.toml"):
        test_command = "python -m pytest -q"
    elif (project / "package.json").exists():
        test_command = "npm test"
    elif (project / "Cargo.toml").exists():
        test_command = "cargo test"
    elif (project / "go.mod").exists():
        test_command = "go test ./..."
    # data files the request's words point at first, then the rest (short paths first)
    data.sort(key=lambda p: (not words & set(re.split(r"[\W_./-]+", p.lower())), len(p)))
    related.sort(key=len)
    return ProjectFacts(data[:3], _top_modules(modules), related[:3], test_command)


def _read(path: Path, limit: int = 20000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _top_modules(modules: list[str]) -> list[str]:
    """Package folders (or top-level modules) by how much code they hold."""
    counts: dict[str, int] = {}
    for module in modules:
        parts = module.split("/")
        key = "/".join(parts[:-1]) + "/" if len(parts) > 1 else module
        counts[key] = counts.get(key, 0) + 1
    return [name for name, _ in sorted(counts.items(), key=lambda item: -item[1])][:3]


def rule_questions(task: str, facts: ProjectFacts) -> list[tuple[str, list[str]]]:
    """2-4 questions with options grounded in the project, for a request the rules call vague."""
    thing = _OPEN_BUILD.search(task)
    noun = re.sub(r"^(?:a|an|some|me\s+an?|my|new|the\s+new)\s+", "", thing.group(0),
                  flags=re.IGNORECASE).strip() if thing else "change"
    inputs = [f"read `{path}`" for path in facts.data_files]
    what = [*inputs[:2], "call a model on prompts I give you and save the replies",
            "something else (describe the input and the output)"]
    if not thing:  # "make it faster", "fix it": the target is the open question
        targets = [f"`{path}`" for path in (facts.related or facts.modules)[:3]]
        questions = [("Which part of the project do you mean?", [*targets, "something else (name it)"]),
                     ("What counts as done?", ["a number to reach (e.g. a time or size)",
                                               "a failing test or example that should pass",
                                               "something else"])]
        if facts.test_command:
            questions.append(("How should I check it?",
                              [f"run the tests (`{facts.test_command}`)", "a before/after timing",
                               "both"]))
        return questions
    where = [f"inside `{path}`" for path in facts.related[:2]]
    where += [f"a new module in `{name}`" for name in facts.modules[:2] if name.endswith("/")]
    where.append("a new standalone script")
    questions = [(f"What should the {noun} take in and produce?", what[:4]),
                 ("Where should it live?", where[:4])]
    if _VERIFY.search(task) or facts.test_command:
        check = []
        if facts.test_command:
            check.append(f"pytest tests run with `{facts.test_command}`")
        check += ["a sample run on a small input, output shown to you", "both"]
        questions.append(("How should I verify it end to end?", check[:3]))
    if _MONITOR.search(task):
        questions.append(("What should the loop measure, and when should it run?",
                          ["latency and success rate of each run",
                           "output quality against a labelled file",
                           "on demand / on a schedule / on every commit"]))
    return questions[:4]


def render(questions: list[tuple[str, list[str]]], reasons: str = "") -> str:
    lead = "Before I build anything, a few decisions only you can make"
    lead += f" ({reasons})" if reasons else ""
    lines = [lead + ". Answer by number and letter, or in your own words:", ""]
    for number, (question, options) in enumerate(questions, 1):
        lines.append(f"{number}. {question}")
        for letter, option in zip("abcd", options):
            lines.append(f"   {letter}) {option}")
    lines += ["", "Which options should I go with?"]
    return "\n".join(lines)


# -- the model check for unsure requests (switch clarify_model) ------------------------------------

CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "specific": {"type": "boolean"},
        "missing": {"type": "string"},
        "questions": {
            "type": "array", "maxItems": 3,
            "items": {"type": "object",
                      "properties": {"question": {"type": "string"},
                                     "options": {"type": "array", "items": {"type": "string"},
                                                 "maxItems": 4}},
                      "required": ["question", "options"]},
        },
    },
    "required": ["specific", "questions"],
}
CHECK_SYSTEM = (
    "You decide whether a coding agent can carry out a request in this project without guessing a "
    "decision that only the user can make: a format, which of several components, delete or keep, "
    "whose data (scope), a threshold value, which of two conflicting conventions, a target that "
    "does not exist in the project, or a goal with nothing to measure. If the README, the project "
    "context from earlier sessions or the files settle the decision, the request is specific. "
    "When in doubt, it is specific: a needless question costs the user as much as a wrong guess. Reply with JSON only: "
    '{"specific": true|false, "missing": "<the decision, if any>", "questions": [{"question": '
    '"...", "options": ["...", "..."]}]} with at most 3 questions, each naming the decision and '
    "offering concrete options drawn from the project."
)
CHECK_MAX_TOKENS = 400
CHECK_TEMPERATURE = 0.0  # the ask/act verdict must not depend on sampling luck
README_CHARS = 1500
FILES_SHOWN = 80
CONTEXT_CHARS = 2000


def model_check(agent: CodingAgent, task: str) -> list[tuple[str, list[str]]] | None:
    """One short constrained call. Questions when the request needs a decision, else None (also on
    any error: the gate never blocks a turn)."""
    from .tools import _project_files

    files = []
    for path in _project_files(agent.project, limit=FILES_SHOWN):
        files.append(path.relative_to(agent.project).as_posix())
    readme = ""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        readme = _read(agent.project / name, README_CHARS)
        if readme:
            break
    if not readme:
        readme = next((_read(agent.project / f, README_CHARS) for f in files
                       if f.lower().endswith("readme.md")), "")
    # what the agent loaded at session start (VEGA.md, session digests, notes): "the bug we found
    # last session" is settled there, not in the README
    loaded = (getattr(agent, "context_section", "") or "").strip()
    if len(loaded) > CONTEXT_CHARS:
        loaded = "…" + loaded[-CONTEXT_CHARS:]  # the newest digests are last
    user = (f"Request: {task}\n\nProject files:\n" + "\n".join(files)
            + (f"\n\nREADME:\n{readme}" if readme else "")
            + (f"\n\nProject context from earlier sessions:\n{loaded}" if loaded else ""))
    # temperature 0 (only this call): one request gets one ask/act verdict; the seed, when an eval
    # pins one, comes with agent.sampling (clarify_model, 2026-10-20)
    sampling = {**agent.sampling,
                **({"temperature": CHECK_TEMPERATURE} if features.on("clarify_greedy") else {}),
                "max_tokens": min(CHECK_MAX_TOKENS, agent.sampling.get("max_tokens", 10**6))}
    try:
        reply = agent.client.chat.completions.create(
            model=agent.model, **sampling,
            messages=[{"role": "system", "content": CHECK_SYSTEM}, {"role": "user", "content": user}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "clarify", "schema": CHECK_SCHEMA}},
        )
        data = json.loads(reply.choices[0].message.content or "")
    except Exception as exc:  # noqa: BLE001 - never block the turn on the check
        debuglog.record("stage", stage="clarify.model_failed", error=str(exc)[:200])
        return None
    if not isinstance(data, dict) or data.get("specific", True) is not False:
        return None
    questions = []
    for item in data.get("questions") or []:
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            continue
        options = [str(o).strip() for o in item.get("options") or [] if str(o).strip()]
        questions.append((str(item["question"]).strip(), options[:4]))
    return questions[:3] or None


# -- the gate -------------------------------------------------------------------------------------

def is_follow_up(agent: CodingAgent) -> bool:
    """Earlier user turns in this conversation (not harness notes or summaries)."""
    return any(m.get("role") == "user" and not str(m.get("content", "")).startswith("(Vega ")
               for m in agent.messages[1:])


def answered(agent: CodingAgent, task: str) -> str:
    """The turn after Vega's questions: the original request with the user's answers, so the goal
    stands on its own (pipelines see only the task, and compaction keeps the goal but may drop the
    questions). Any other turn: the task unchanged."""
    pending = getattr(agent, "_clarify_pending", None)
    agent._clarify_pending = None
    if not pending:
        return task
    return f"{pending}\n\nMy answers to your questions:\n{task}"


def run(agent: CodingAgent, task: str) -> str | None:
    """The clarifying reply for an underspecified request, or None to go on with the turn. Call
    `answered` first: a turn that answers Vega's questions is never gated again."""
    verdict = classify(task, follow_up=is_follow_up(agent))
    debuglog.record("stage", stage="clarify.route", verdict=verdict.kind, reasons=verdict.reasons)
    if verdict.kind == CLEAR:
        return None
    questions: list[tuple[str, list[str]]] | None
    if verdict.kind == VAGUE:
        questions = rule_questions(task, project_facts(agent.project, task))
    elif features.on("clarify_model"):
        agent._emit("notice", message="Checking whether the request needs a decision from you")
        questions = model_check(agent, task)
    else:
        return None
    if not questions:
        return None
    reply = render(questions)
    agent._emit("notice", message="Asking before building: the request leaves decisions open")
    agent.messages += [{"role": "user", "content": task}, {"role": "assistant", "content": reply}]
    agent._clarify_pending = task
    return reply
