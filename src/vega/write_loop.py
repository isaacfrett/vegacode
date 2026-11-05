"""One write → check → repair engine for the writing pipelines (2026-10-17).

The direct, project and deliverable pipelines each grew their own loop: their own input renderer
and character budget, their own code-block regexes, their own best-round rule (the deliverable
pipeline kept the last round) and their own answer wording, so a fix (pipeline_signals, the
honest answer, quoting the failing checks) had to be built once per pipeline. This module holds
the parts they share; a pipeline is a `LoopConfig` plus its prompt:

- `PipelineHost`: what the loop needs from the agent (emit, before_write, refresh_written, the
  acceptance spec, the stop flag, one model reply), so it never reaches into agent internals;
  `AgentHost` adapts a CodingAgent;
- `render_inputs`: one renderer for the read-only inputs, under one budget from the window;
- `pick_block`: the block for a path (a `File: path` label wins, then the file's language, never
  the longest block of any kind);
- `run_loop`: write (merge via codemerge, the atomic check of edits.check) → check stage →
  repair, N rounds, per-round feedback, the round with the fewest failing checks restored;
- `honest_answer`: what was written, N of M checks by kind, the failing lines quoted, the files
  left and why. Nothing passes when nothing ran.

Pipelines on it: the direct pipeline (pipeline.py), the deliverable pipeline, the single-file
non-Python pipeline (nonpython_pipeline.py) and the project pipeline (project_pipeline.py). A
pipeline that writes one reply's files per round gives `produce`; one that writes several targets
per round, each from its own model call (the project pipeline: one call per planned file), gives
`produce_each` and `round_targets`, and `on_write` is told each target's result. `score` ranks
the rounds (default: fewer failing checks), `done` ends them (default: nothing fails), `between`
runs before the next round (the project pipeline replans there) and `snapshot` says what a round's
files are when steps other than the loop's writes change them.
"""

from __future__ import annotations

import contextlib
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from . import data_preview, debuglog, edits, elision, extensions, features
from .codemerge import _parses
from .tools import execute

if TYPE_CHECKING:
    from .acceptance import Spec
    from .agent import CodingAgent

CHARS_PER_TOKEN = 3.5  # context.CHARS_PER_TOKEN
DEFAULT_NUM_CTX = 8192
FEEDBACK_FAILURES = 8  # failing checks shown in one repair prompt
QUOTED_FAILURES = 3  # failing checks quoted in the answer
QUOTED_CHARS = 200
MAX_STALL_ROUNDS = 2  # extensions.stalled_round: rounds it may add to a loop (they ran no checks)


# -- the host -------------------------------------------------------------------------------------

class PipelineHost(Protocol):
    """What a pipeline needs from the agent that runs it."""

    project: Path
    teach: bool
    num_ctx: int | None

    def emit(self, kind: str, **data: Any) -> None: ...
    def before_write(self, *args: Any, **kwargs: Any) -> Any: ...
    def refresh_written(self) -> None: ...
    def acceptance(self, task: str) -> Spec | None: ...
    def stop_flag(self) -> threading.Event | None: ...
    def ask(self, messages: list[dict[str, Any]], **extra: Any) -> str: ...
    def system(self, prompt: str) -> str: ...
    def remember(self, task: str, summary: str) -> None: ...


class AgentHost:
    """A CodingAgent as a PipelineHost (the only place the pipelines touch its internals)."""

    def __init__(self, agent: CodingAgent):
        self.agent = agent
        self.project = agent.project
        self.teach = bool(getattr(agent, "teach", False))
        self.num_ctx = getattr(agent, "num_ctx", None)

    def emit(self, kind: str, **data: Any) -> None:
        self.agent._emit(kind, **data)

    def before_write(self, *args: Any, **kwargs: Any) -> Any:
        return self.agent._before_write(*args, **kwargs)

    def refresh_written(self) -> None:
        self.agent._refresh_written()

    def acceptance(self, task: str) -> Spec | None:
        from . import acceptance

        return acceptance.ensure(self.agent, task)

    def stop_flag(self) -> threading.Event | None:
        return getattr(self.agent, "_stop", None)

    def ask(self, messages: list[dict[str, Any]], **extra: Any) -> str:
        from .pipeline import ask_complete

        return ask_complete(self.agent, messages, **extra)

    def system(self, prompt: str) -> str:
        from .pipeline import with_context

        return with_context(self.agent, prompt)

    def remember(self, task: str, summary: str) -> None:
        self.agent.messages += [{"role": "user", "content": task},
                                {"role": "assistant", "content": summary}]


# -- one input renderer ---------------------------------------------------------------------------

def window_chars(num_ctx: int | None, share: float) -> int:
    """`share` of the context window, in characters."""
    return int((num_ctx or DEFAULT_NUM_CTX) * CHARS_PER_TOKEN * share)


def render_inputs(project: Path, inputs: list[str], *, budget: int,
                  caps: dict[str, int] | None = None, max_files: int | None = None,
                  exclude: set[str] | frozenset[str] = frozenset(), header: str = "",
                  data_header: str = "", label_inside: bool = False, fence: str = "",
                  whole_first: bool = False) -> str:
    """The read-only inputs a task names, for a writer prompt; '' for none.

    Each file is shown whole, or as its data preview (a large data file, a binary database), cut
    so that all of them together stay within `budget` characters: shortest first, each taking at
    most its fair share of what is left (a short file leaves its share to the longer ones) and at
    most its kind's cap (`caps`: "data" / "other"). `header` introduces them (`data_header` when
    every file is data); `label_inside` puts `path:` inside the fence (the direct pipeline's
    layout), else before it. `whole_first`: a readable data file is shown whole while it fits its
    share, and as its preview only when it does not (the project pipeline's rule); otherwise a
    large data file is always shown as its preview (the direct pipeline's)."""
    shown = [p for p in inputs if p not in exclude and (project / p).is_file()]
    if max_files is not None:
        shown = shown[:max_files]
    if not shown or budget <= 0:
        return ""
    texts: dict[str, str] = {}
    for relative in shown:
        path = project / relative
        text = data_preview.preview(path, relative, hint=False) \
            if data_preview.is_data_path(relative) and not whole_first else None
        if text is None and whole_first:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):  # a binary data file (sqlite): its preview
                text = data_preview.preview(path, relative, hint=False)
            if text is None:
                continue
        if text is None:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
        texts[relative] = text
    remaining = budget
    for number, relative in enumerate(sorted(texts, key=lambda p: len(texts[p]))):
        text = texts[relative]
        cap = remaining // (len(texts) - number)
        if caps:
            kind = "data" if data_preview.is_data_path(relative) else "other"
            cap = min(cap, caps.get(kind, cap))
        if len(text) > cap and whole_first and data_preview.is_data_path(relative):
            text = data_preview.preview(project / relative, relative, hint=False) or text
        if len(text) > cap:
            text = (text[:cap].rsplit("\n", 1)[0]
                    + f"\n[... cut: {relative} has {len(text):,} characters]")
        remaining -= len(text)
        texts[relative] = text
    if not texts:
        return ""
    title = data_header if data_header and all(data_preview.is_data_path(p) for p in texts) \
        else header
    if label_inside:
        parts = [(text if text.startswith("Preview of ") else f"{relative}:\n{text}").rstrip("\n")
                 for relative, text in texts.items()]
        mark = fence_for("\n".join(parts))
        return (f"\n\n{title}:\n{mark}{fence}\n" + f"\n{mark}\n{mark}{fence}\n".join(parts)
                + f"\n{mark}")
    parts = [f"{relative}:\n{fence_for(text)}{fence}\n{text.rstrip(chr(10))}\n{fence_for(text)}"
             for relative, text in texts.items()]
    return f"\n\n{title}:\n" + "\n".join(parts)


# -- one block picker -----------------------------------------------------------------------------

_FENCED = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
# fence_lines: a block opens and closes on a line of its own, and closes only with a fence at least
# as long as its opening one, so a ``` inside the code (`line.startswith("```")`) does not end it
# (goals 2026-10-20, docs-table-of-contents: every write of headings.py was cut mid-string).
_FENCE_LINES = re.compile(r"^[ \t]*(`{3,})([^\n`]*)\n(.*?)^[ \t]*\1`*[ \t]*$",
                          re.MULTILINE | re.DOTALL)
_FENCE_LINE = re.compile(r"^[ \t]*```", re.MULTILINE)
_BACKTICK_RUN = re.compile(r"`{3,}")
_ANY_INFO = re.compile(r"[\w.+-]*")
_PYTHON_INFO = {"python", "py", ""}


def fenced(reply: str) -> list[tuple[int, int, str, str]]:
    """Every fenced block of a reply as (start, end, info string, body); the body keeps its last
    newline. With fence_lines the fences are whole lines (falling back to the old inline match
    when no block has line fences, e.g. "Here: ```python ... x```")."""
    inline = [(m.start(), m.end(), m.group(1), m.group(2)) for m in _FENCED.finditer(reply)]
    if features.on("fence_lines"):
        found = [(m.start(), m.end(), m.group(2), m.group(3)) for m in _FENCE_LINES.finditer(reply)]
        if found:
            if [(f[2].strip(), f[3]) for f in found] != [(i[2].strip(), i[3]) for i in inline]:
                features.mark_acted("fence_lines", where="fenced")  # other blocks than inline
            return found
    return inline


def code_blocks(reply: str, any_language: bool = False) -> list[str]:
    """The bodies of a reply's ```python / ```py / bare blocks (any `[\\w.+-]*` language with
    `any_language`), in order."""
    if not features.on("fence_lines"):
        pattern = (r"```[\w.+-]*[ \t]*\n(.*?)```" if any_language
                   else r"```(?:python|py)?[ \t]*\n(.*?)```")
        return re.findall(pattern, reply, re.DOTALL)
    blocks = []
    for _, _, info, body in fenced(reply):
        info = info.strip()
        if (_ANY_INFO.fullmatch(info) is not None if any_language
                else info.lower() in _PYTHON_INFO):
            blocks.append(body)
    pattern = (r"```[\w.+-]*[ \t]*\n(.*?)```" if any_language
               else r"```(?:python|py)?[ \t]*\n(.*?)```")
    if blocks != re.findall(pattern, reply, re.DOTALL):
        features.mark_acted("fence_lines", where="code_blocks")
    return blocks


def fence_for(text: str) -> str:
    """The fence to show `text` in a prompt: ``` unless the text holds a run of backticks that
    long, then one backtick longer (fence_lines), so the shown file does not end mid-way."""
    if not features.on("fence_lines"):
        return "```"
    longest = max((len(run) for run in _BACKTICK_RUN.findall(text)), default=0)
    if longest >= 3:
        features.mark_acted("fence_lines", where="fence_for")
    return "`" * max(3, longest + 1)


def inside_open_block(text: str) -> bool:
    """Whether `text` (a reply cut off at the token limit) stops inside a code block: an odd
    number of fence lines (fence_lines, when it has any), else of ``` anywhere."""
    lines = len(_FENCE_LINE.findall(text)) if features.on("fence_lines") else 0
    inside = lines % 2 == 1 if lines else text.count("```") % 2 == 1
    if lines and inside != (text.count("```") % 2 == 1):
        features.mark_acted("fence_lines", where="inside_open_block")
    return inside


_PATHLIKE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w-][\w.-]*\.[A-Za-z]\w{0,9})(?![\w/])")
LANGUAGES = {  # file extension -> fence languages that hold it
    ".py": {"python", "py", "python3"}, ".pyi": {"python", "py"}, ".toml": {"toml"},
    ".json": {"json"}, ".yaml": {"yaml", "yml"}, ".yml": {"yaml", "yml"},
    ".md": {"markdown", "md"}, ".txt": {"text", "txt", "plaintext", "plain"},
    ".cfg": {"ini", "cfg"}, ".ini": {"ini", "cfg"}, ".csv": {"csv"},
    ".sh": {"bash", "sh", "shell"}, ".html": {"html"}, ".js": {"javascript", "js", "node"},
    ".mjs": {"javascript", "js"}, ".cjs": {"javascript", "js"}, ".ts": {"typescript", "ts"},
    ".go": {"go", "golang"}, ".java": {"java"}, ".rb": {"ruby", "rb"}, ".r": {"r"},
    ".sql": {"sql", "sqlite", "postgresql", "postgres", "mysql"}, ".rs": {"rust", "rs"},
    ".c": {"c"}, ".h": {"c", "cpp", "c++"}, ".cpp": {"cpp", "c++"}, ".cc": {"cpp", "c++"},
    ".hpp": {"cpp", "c++"}, ".cs": {"csharp", "cs", "c#"}, ".swift": {"swift"},
    ".kt": {"kotlin", "kt"},
}
FENCE_LANGUAGE = {".py": "python", ".toml": "toml", ".json": "json", ".yaml": "yaml",
                  ".yml": "yaml", ".md": "markdown", ".txt": "text", ".cfg": "ini", ".ini": "ini",
                  ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
                  ".ts": "typescript", ".go": "go", ".java": "java", ".rb": "ruby", ".r": "r",
                  ".sql": "sql", ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp",
                  ".cc": "cpp", ".hpp": "cpp", ".cs": "csharp", ".swift": "swift",
                  ".kt": "kotlin", ".sh": "bash"}
_EXTENSIONS = {"py", "pyi", "toml", "json", "yaml", "yml", "md", "txt", "cfg", "ini", "csv", "sh",
               "html", "js", "ts", "rst", "in", "lock", "tsv", "xml", "c", "h", "sql", "go",
               "java", "rb", "r", "mjs", "cjs", "rs", "cpp", "cc", "hpp", "cs", "swift", "kt"}
LABEL_WORDS = 6  # a label line before a block: the path plus at most this many other words


def _label_paths(line: str, words: int | None = None) -> list[str]:
    """Distinct file paths named on a label line (`File: a/b.py`, `**a/b.py**`, `# a/b.py`,
    `### Updated \\`a/b.py\\``); [] for a line naming none, or (with `words`) a sentence with more
    than that many other words ("Here is the code, which imports `models.py`, ...")."""
    text = line.strip()
    if not text:
        return []
    found: list[str] = []
    for path in _PATHLIKE.findall(text):
        path = path.strip(".")
        if path.rsplit(".", 1)[-1].lower() in _EXTENSIONS and path not in found:
            found.append(path)
    if found and words is not None:
        rest = text
        for path in found:
            rest = rest.replace(path, " ")
        if len(re.findall(r"[A-Za-z]\w*", rest)) > words:
            return []
    return found


def _label_matches(label: str, path: str) -> bool:
    label = label.strip("`*:#/ ").removeprefix("./")
    return label == path or path.endswith("/" + label) or label.endswith("/" + path)


def reply_blocks(reply: str) -> list[tuple[str, str, str]]:
    """Every fenced block of a writer reply as (language, label path or '', code): the label is a
    path in the fence's info string, else on the last non-empty line before the fence (when that
    line names exactly one file), else a first line inside the block that only names a file."""
    blocks: list[tuple[str, str, str]] = []
    previous_end = 0
    for start, end, info, body in fenced(reply):
        info = info.strip()
        words = info.split()
        language = words[0].lower() if words and not _label_paths(words[0]) else ""
        label = ""
        for word in words:
            paths = _label_paths(word.split("=", 1)[-1].strip("\"'"))
            if paths:
                label = paths[0]
        if not label:
            before = [line for line in reply[previous_end:start].split("\n") if line.strip()]
            paths = _label_paths(before[-1], LABEL_WORDS) if before else []
            if len(paths) == 1:
                label = paths[0]
        if not label:
            first = body.lstrip("\n").split("\n", 1)[0].strip()
            stripped = first
            for prefix in ("#", "//", "--", "File:", "file:", "Path:", "path:"):
                stripped = stripped.removeprefix(prefix).strip()
            paths = _label_paths(stripped)
            if len(paths) == 1 and stripped.strip("`*: ") == paths[0]:
                label = paths[0]
        blocks.append((language, label, body))
        previous_end = end
    return blocks


def pick_block(reply: str, path: str, plan_paths: list[str]) -> tuple[str | None, str]:
    """(the code for `path` from a writer reply, '') or (None, why there is none, to tell the
    writer). A block labelled with the path wins; else an unlabelled block in the file's language
    (or with no language); never a block labelled for another file, never the longest by size."""
    blocks = reply_blocks(reply)
    if not blocks:
        return None, f"Your reply had no code block for {path}."
    mine = [code for _, label, code in blocks if label and _label_matches(label, path)]
    if mine:
        return max(mine, key=len), ""
    suffix = Path(path).suffix.lower()
    wanted = LANGUAGES.get(suffix)
    unlabelled = [(language, code) for language, label, code in blocks if not label]
    if wanted is None:  # an extension without a known fence language: any unlabelled block
        matching = [code for _, code in unlabelled]
    else:
        matching = [code for language, code in unlabelled if language in wanted] or \
                   [code for language, code in unlabelled if not language]
    if matching:
        return max(matching, key=len), ""
    others = []
    for _language, label, _ in blocks:
        if label:
            owner = next((p for p in plan_paths if _label_matches(label, p)), label)
            if owner not in others:
                others.append(owner)
    if others:
        return None, f"You sent code for {', '.join(others)}; this step writes {path}."
    languages = sorted({language for language, _, _ in blocks if language})
    return None, (f"You sent a ```{'/'.join(languages)} block; this step writes {path}, so the "
                  f"block must be {fence_language(path) or suffix.lstrip('.')}.")


def fence_language(path: str) -> str:
    return FENCE_LANGUAGE.get(Path(path).suffix.lower(), "")


# -- the loop -------------------------------------------------------------------------------------

@dataclass
class Write:
    """One file a reply produced: the merged content to write, and the reply's own code (for a
    Python file merged by definitions: checked by itself, since a merge drops what won't parse)."""

    content: str
    code: str | None = None


@dataclass
class Produced:
    """A reply turned into files. `problems`: format problems found in them (a deliverable not
    produced, invalid JSON, unchanged); with no files, `problem` is what to tell the writer."""

    writes: dict[str, Write] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    problem: str = ""
    checked: int = 0  # format checks made (for the count of checks)


@dataclass
class Checks:
    """One round's check stage. `kinds`: number of checks run by kind (in the answer's order)."""

    kinds: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    test_output: str = ""  # the visible tests' failing output, if any (one failing check)
    report: str | None = None  # the run_check result shown, instead of "N/M checks passed ..."
    # each check with the checker that produced it (item()), for extensions.checks_recorded;
    # None: the pipeline does not record them one by one. The answer never reads it.
    items: list[dict[str, Any]] | None = None
    # failing lines an extension added in checks_recorded (add_failure): part of `failures`
    added: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.kinds.values())

    def add_failure(self, source: str, label: str, line: str) -> None:
        """A failing check an extension found (extensions.checks_recorded, before the loop decides
        whether to repair): counted as failing, its line in the repair feedback and its record in
        `items`."""
        self.failures.append(line)
        self.added.append(line)
        if self.items is not None:
            self.items.append(item(source, label, False, line))

    @property
    def failed(self) -> int:
        return len(self.failures) + (1 if self.test_output else 0)


def item(source: str, label: str, ok: bool, detail: str = "", **extra: Any) -> dict[str, Any]:
    """One check for Checks.items: the checker that produced it (extensions.checks_recorded
    lists them), what it checked, whether it passed, and its failing line or detail."""
    return {"source": source, "label": label, "ok": ok, "detail": detail, **extra}


def counted_items(source: str, label: str, total: int, failures: list[str]) -> list[dict[str, Any]]:
    """`total` checks of one checker that reports only a count and its failing lines: one
    failing item per line, then the passing rest."""
    return ([item(source, label, False, line) for line in failures]
            + [item(source, label, True) for _ in range(max(total - len(failures), 0))])


def acceptance_items(result: Any) -> list[dict[str, Any]]:
    """An acceptance.Result's items for Checks.items."""
    return [item("acceptance", i.label, i.ok, i.detail or "", kind=i.kind, output=i.output,
                 note=i.note) for i in (result.items if result is not None else [])]


@dataclass
class LoopConfig:
    """One pipeline's use of the loop."""

    name: str  # debug-log stage prefix, e.g. "direct_pipeline"
    title: str  # for notices, e.g. "Direct pipeline"
    targets: list[str]  # the files the loop may write (project-relative)
    originals: dict[str, str | None]  # their content before the loop (None: new)
    repairs: int
    check: Callable[[list[str]], Checks]
    produce: Callable[[str], Produced] | None = None  # one reply -> its files (or produce_each)
    no_code: str = ""  # the reply had no usable code
    # (refusals, format problems) -> message, when none of a reply's files was saved
    not_saved: Callable[[list[str], list[str]], str] | None = None
    repair: str = ""  # the repair prompt, with {feedback}
    check_call: str = "checks"  # the run_check event's command label
    check_result: str = ""  # its result label (defaults to check_call)
    no_checks_text: str = "no checks to run"
    ask: Callable[[list[dict[str, Any]]], str] | None = None  # instead of host.ask
    stop_early: Callable[[int], bool] | None = None  # after a failed round (its index): stop?
    # several targets per round, each produced by its own call (instead of one reply + produce):
    produce_each: Callable[[str], Produced] | None = None  # target -> its files, or a problem
    round_targets: Callable[[], list[str]] | None = None  # this round's targets, in order
    on_write: Callable[[str, str], None] | None = None  # (target, its save result or problem)
    score: Callable[[Checks], tuple[int, ...]] | None = None  # a round's rank; lower is better
    done: Callable[[Checks, int], bool] | None = None  # (checks, round index): stop here?
    between: Callable[[int], None] | None = None  # before the next round (the index just done)
    snapshot: Callable[[], dict[str, str | None]] | None = None  # a round's files, if not targets
    # unchanged_rounds, several-target loops: gives the next round's writers the note (without
    # it, a round that changed nothing ends the loop: the same prompts would give the same files)
    on_unchanged: Callable[[str], None] | None = None
    task: str = ""  # the request (for extensions.checks_recorded)
    # the request for checks_recorded's evidence providers only (`request`), where `task` stays ""
    # because the task-gated hooks were never measured on that loop (the web pipeline)
    request: str = ""
    # extensions.stalled_round: a several-target loop's way to add files to its rounds
    # ((paths, change text) -> the paths added), and whether a single-file loop's caller can hand
    # a stall over to another pipeline (LoopResult.handover)
    widen: Callable[[list[str], str], list[str]] | None = None
    handover: bool = False


@dataclass
class Attempt:
    passed: int
    total: int
    failures: list[str]  # with the visible tests' last line, if they failed
    kinds: dict[str, int]
    failed: int
    reply: str  # the writer's reply as written (teach mode: with its takeaway)
    checks: Checks | None = None  # the round's check stage


@dataclass
class LoopResult:
    attempts: list[Attempt]
    best: int | None  # index into attempts of the round kept
    snapshot: dict[str, str | None]  # the kept round's files
    messages: list[dict[str, Any]]
    refused: dict[str, str]  # path -> why its last write was not saved
    # extensions.stalled_round: the stall the loop ended on, for the caller to hand over
    # ({"paths", "failure", "reason"}); None: no handover
    handover: dict[str, Any] | None = None

    @property
    def kept(self) -> Attempt | None:
        return self.attempts[self.best] if self.best is not None else None


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else None
    except (OSError, UnicodeDecodeError):
        return None


def save(host: PipelineHost, config: LoopConfig, relative: str, write: Write) -> str:
    """Write one file through the normal write path (review, undo); with pipeline_signals the
    atomic check refuses a write that would not parse or adds new static findings, a Python
    reply that does not parse by itself is refused before the merge silently drops it, and
    (elision_guard) a write whose comment or `...` stands in for code the file had is refused."""
    project = host.project
    args = {"path": relative, "content": write.content}
    host.emit("tool_call", name="write_file", args=args)
    refusal = None
    if features.on("pipeline_signals"):
        features.mark_acted("pipeline_signals", where="save")  # the atomic write check ran
        now = _read(project / relative)
        original = config.originals.get(relative)
        if write.code is not None and original is not None and _parses(original):
            refusal = edits.unparsable_reply(relative, write.code, project)
        if refusal is None and features.on("elision_guard"):
            # a comment or `...` standing in for code the file had (2026-10-27: 3 of the 12
            # saved elisions in the logs were pipeline writes)
            found = elision.elided(relative, now, write.content, write.code)
            if found is not None:
                features.mark_acted("elision_guard", path=relative)
                refusal = elision.refusal("pipeline", relative, found)
        if refusal is None:
            verdict = edits.check("pipeline", relative, now, write.content, project,
                                  pending=[t for t in config.targets if t != relative],
                                  repair=True)
            refusal = verdict.refusal
            if refusal is None and verdict.after is not None:  # repaired (extensions hook)
                args = {"path": relative, "content": verdict.after}
        if refusal is not None:
            debuglog.record("stage", stage=f"{config.name}.refused", path=relative,
                            refusal=refusal[:600])
    result = refusal if refusal is not None else execute(
        "write_file", args, project, before_write=host.before_write, atomic=False)
    host.refresh_written()
    host.emit("tool_result", name="write_file", args=args, result=result)
    return result


UNCHANGED_NOTE = (
    "(Vega) Your rewrite was identical to the current {files} ({paths}): nothing changed, so the "
    "checks were not run again. The failing check is still:\n{failing}\nChange the code that "
    "causes it."
)


def _disk(project: Path, paths: set[str]) -> dict[str, str | None]:
    return {path: _read(project / path) for path in sorted(paths)}


def unchanged_note(saved: list[str], failures: list[str]) -> str:
    """unchanged_rounds: what the writer is told after a round that changed no file."""
    failing = failures[0] if failures else "(the failing checks above)"
    return UNCHANGED_NOTE.format(files="file" if len(saved) == 1 else "files",
                                 paths=", ".join(saved), failing=failing[:QUOTED_CHARS * 2])


def _can_sample(host: PipelineHost) -> bool:
    """Whether one round's requests can carry their own sampling (an AgentHost's agent)."""
    return isinstance(getattr(getattr(host, "agent", None), "sampling", None), dict)


@contextlib.contextmanager
def _sampled(host: PipelineHost, overrides: dict[str, Any]):
    """The agent's request options with `overrides` (e.g. a temperature) for one round's
    calls; the rest (the seed an eval pins) as they are. Restored afterwards."""
    sampling = getattr(getattr(host, "agent", None), "sampling", None)
    if not overrides or not isinstance(sampling, dict):
        yield
        return
    saved = dict(sampling)
    sampling.update(overrides)
    try:
        yield
    finally:
        sampling.clear()
        sampling.update(saved)


def _write_each(host: PipelineHost, config: LoopConfig, state: dict[str, str | None],
                refused: dict[str, str]) -> list[str]:
    """A several-target round: each target produced by its own call, then saved; config.on_write
    gets each target's save result (or the problem when it produced nothing)."""
    assert config.produce_each is not None and config.round_targets is not None
    saved: list[str] = []
    for target in config.round_targets():
        produced = config.produce_each(target)
        result = produced.problem or config.no_code
        for relative, write in produced.writes.items():
            result = save(host, config, relative, write)
            if result.startswith(("Wrote", "Edited")):
                saved.append(relative)
                state[relative] = write.content
                refused.pop(relative, None)
            else:
                refused[relative] = result
        if config.on_write is not None:
            config.on_write(target, result)
    return saved


def run_loop(host: PipelineHost, config: LoopConfig,
             messages: list[dict[str, Any]]) -> LoopResult | None:
    """Write → check → repair, at most config.repairs repairs; the best round (config.score, else
    the fewest failing checks) is restored at the end. None when the model call fails (the agent
    loop runs); with produce_each, its errors are raised to the caller.

    A several-target loop (produce_each) checks every round, saved or not (a refused write is the
    pipeline's failing check to count), keeps no conversation (each call builds its own prompt)
    and runs config.between before each repair round."""
    from . import teaching

    attempts: list[Attempt] = []
    # (rank, attempt, files, the loop's own rank): the rank may come from extensions.round_score
    best: tuple[tuple[int, ...], int, dict[str, str | None], tuple[int, ...]] | None = None
    refused: dict[str, str] = {}
    state = dict(config.originals)  # what each target holds: the last content saved
    each = config.produce_each is not None
    # unchanged_rounds: the files as the last check left them (None: no check yet), and whether
    # the round before this one changed nothing too
    unchanged = features.on("unchanged_rounds")
    checked: dict[str, str | None] | None = None
    checked_problems: list[str] = []
    unchanged_before = False
    # extensions.stalled_round: unchanged rounds in a row, the last action taken on one, the next
    # round's sampling, the rounds it added, and a handover the loop ends on
    stalls = 0
    stall_action: dict[str, Any] | None = None
    sampling: dict[str, Any] = {}
    limit, added_rounds = config.repairs + 1, 0
    handover: dict[str, Any] | None = None
    round_number = -1
    while round_number + 1 < limit:
        round_number += 1
        overrides, sampling = sampling, {}
        produced = Produced()
        written_reply = reply = ""
        if each:
            with _sampled(host, overrides):
                saved = _write_each(host, config, state, refused)
        else:
            try:
                with _sampled(host, overrides):
                    reply = (config.ask(messages) if config.ask is not None
                             else host.ask(messages))
                assert config.produce is not None and config.not_saved is not None
                produced = config.produce(reply)
            except Exception as exc:  # noqa: BLE001 - fall back to the agent loop
                host.emit("notice", message=f"{config.title} stopped ({exc}); using the agent "
                                            "loop.")
                return None
            if not produced.writes:
                debuglog.record("stage", stage=f"{config.name}.round", round=round_number + 1,
                                files=[], problem=(produced.problem or config.no_code)[:600])
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": produced.problem or config.no_code}]
                if config.stop_early is not None and config.stop_early(round_number):
                    break
                continue
            written_reply = reply
            if host.teach:  # the writer's takeaway is shown, not kept for the repair prompts
                reply = teaching.strip(reply)
            saved = []
            refusals: list[str] = []
            for relative, write in produced.writes.items():
                result = save(host, config, relative, write)
                if result.startswith(("Wrote", "Edited")):
                    saved.append(relative)
                    state[relative] = write.content
                    refused.pop(relative, None)
                else:
                    refused[relative] = result
                    refusals.append(result)
            if not saved:
                debuglog.record("stage", stage=f"{config.name}.round", round=round_number + 1,
                                files=[], refused=sorted(refused))
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": config.not_saved(refusals,
                                                                          produced.problems)}]
                if config.stop_early is not None and config.stop_early(round_number):
                    break
                continue
        if (unchanged and checked is not None and attempts and saved
                and produced.problems == checked_problems
                and set(saved) <= set(checked) and _disk(host.project, set(checked)) == checked):
            # 2026-10-21: repair rounds rewrote every file byte-identically for many rounds;
            # at temperature 0 the same feedback gives the same reply
            debuglog.record("stage", stage=f"{config.name}.unchanged", round=round_number + 1,
                            files=saved, again=unchanged_before)
            features.mark_acted("unchanged_rounds", where=config.name)
            previous = attempts[-1].checks
            if config.done is not None and previous is not None \
                    and config.done(previous, round_number):
                break  # the check would repeat the last one, and that ends the loop (stalled)
            stalls += 1
            action = (extensions.stalled_round(
                config.name, project=host.project, targets=sorted(set(saved)), saved=saved,
                failures=list(attempts[-1].failures),
                output="\n".join(text for text in (previous.test_output, previous.report or "")
                                 if text) if previous is not None else "",
                stalls=stalls, previous=stall_action, can_widen=config.widen is not None,
                can_handover=config.handover, can_sample=_can_sample(host), task=config.task,
                round=round_number + 1) if not each or config.on_unchanged is not None else None)
            if action is not None:
                handed = action.get("handover")
                if isinstance(handed, dict) and config.handover:
                    handover = handed
                    debuglog.record("stage", stage=f"{config.name}.stalled",
                                    round=round_number + 1, stalls=stalls,
                                    branch=action.get("branch", ""),
                                    reason=action.get("reason", ""),
                                    handover=handed.get("paths"))
                    host.emit("notice", message=f"{config.title}: the rewrite left the file as it "
                                                "was and the failing check points at another "
                                                "file; handing the task to the project pipeline.")
                    break
                widened = (config.widen(list(action.get("widen") or []),
                                        str(action.get("why") or ""))
                           if action.get("widen") and config.widen is not None else [])
                note_for = action.get("note_for")
                note = (note_for(widened) if callable(note_for) else str(action.get("note") or "")
                        ) or unchanged_note(saved, attempts[-1].failures)
                temperature = action.get("temperature")
                if isinstance(temperature, (int, float)) and _can_sample(host):
                    sampling = {"temperature": float(temperature)}
                if action.get("extra_round") and added_rounds < MAX_STALL_ROUNDS:
                    added_rounds += 1
                    limit += 1
                stall_action = {"note": note, "branch": action.get("branch", "")}
                debuglog.record("stage", stage=f"{config.name}.stalled", round=round_number + 1,
                                stalls=stalls, branch=action.get("branch", ""),
                                reason=action.get("reason", ""), widened=widened,
                                temperature=sampling.get("temperature"), added=added_rounds,
                                note=note[:1500])
                unchanged_before = True
                host.emit("discarded", reason="the rewrite left every file unchanged", retry=True)
                if each:
                    assert config.on_unchanged is not None
                    config.on_unchanged(note)
                else:
                    messages += [{"role": "assistant", "content": reply},
                                 {"role": "user", "content": note}]
                if config.stop_early is not None and config.stop_early(round_number):
                    break
                continue
            if unchanged_before or (each and config.on_unchanged is None):
                host.emit("notice", message=f"{config.title}: the rewrite left every file as it "
                                            "was; keeping the best round.")
                break
            unchanged_before = True
            note = unchanged_note(saved, attempts[-1].failures)
            host.emit("discarded", reason="the rewrite left every file unchanged", retry=True)
            if each:
                assert config.on_unchanged is not None
                config.on_unchanged(note)
            else:
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": note}]
            if config.stop_early is not None and config.stop_early(round_number):
                break
            continue
        unchanged_before = False
        stalls, stall_action = 0, None
        checks = config.check(saved)
        if unchanged:  # every file the loop targets or saved, as this round's check left them
            checked_problems = list(produced.problems)  # format problems are part of the result
            checked = _disk(host.project, set(config.targets) | set(state)
                            | (set(config.snapshot()) if config.snapshot is not None else set()))
        if produced.checked:
            checks.kinds = {"format checks": produced.checked, **checks.kinds}
        refused_now = [f"{path}: not saved — {text.splitlines()[0] if text else ''}"
                       for path, text in refused.items() if path in produced.writes]
        if checks.items is not None:  # the format checks and unsaved writes, as above
            checks.items = (counted_items("format", "", max(produced.checked,
                                                            len(produced.problems)),
                                          produced.problems)
                            + [item("write", line.split(":", 1)[0], False,
                                    line.split(" — ", 1)[-1]) for line in refused_now]
                            + checks.items)
        checks.failures = produced.problems + refused_now + checks.failures
        if checks.report is None:
            summary = (f"{checks.total - checks.failed}/{checks.total} checks passed"
                       if checks.total else config.no_checks_text)
            checks.report = summary + ("\n" + "\n".join(checks.failures) if checks.failures
                                       else "")
        host.emit("tool_call", name="run_check", args={"command": config.check_call})
        host.emit("tool_result", name="run_check",
                  args={"command": config.check_result or config.check_call},
                  result=checks.report)
        failures = checks.failures + ([checks.test_output.splitlines()[-1]]
                                      if checks.test_output else [])
        attempts.append(Attempt(checks.total - checks.failed, checks.total, failures,
                                dict(checks.kinds), checks.failed, written_reply, checks))
        extensions.checks_recorded(config.name, checks=checks, round=round_number + 1,
                                   project=host.project, task=config.task,
                                   request=config.request or config.task,
                                   last=round_number >= config.repairs,
                                   ask=host.ask, originals=config.originals)
        if checks.added:  # an extension's failing checks: in this round's record too
            attempts[-1].failures += checks.added
            attempts[-1].failed = checks.failed
        debuglog.record("stage", stage=f"{config.name}.round", round=round_number + 1,
                        files=saved, checks=checks.total, failing=checks.failed)
        own = config.score(checks) if config.score is not None else (checks.failed,)
        score = extensions.round_score(config.name, checks=checks, default=own,
                                       round=round_number + 1, project=host.project,
                                       task=config.task)
        if score != own:
            debuglog.record("stage", stage=f"{config.name}.rank", round=round_number + 1,
                            score=list(score), own=list(own))
        if best is None or score < best[0]:
            files = (config.snapshot() if config.snapshot is not None
                     else {path: state.get(path) for path in config.targets})
            best = (score, len(attempts) - 1, files, own)
        if config.done(checks, round_number) if config.done is not None else checks.failed == 0:
            break
        if each:
            if round_number < config.repairs and config.between is not None:
                config.between(round_number)
            continue
        feedback = "\n".join(f"- {failure}" for failure in checks.failures[:FEEDBACK_FAILURES])
        if checks.test_output:
            feedback += f"\nTest output:\n{checks.test_output}"
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": config.repair.format(feedback=feedback)}]
        if config.stop_early is not None and config.stop_early(round_number):
            break
    if best is None:
        return LoopResult(attempts, None, {}, messages, refused, handover)
    kept = best[2]
    if config.snapshot is not None and features.on("restore_later_files"):
        # A file first written after the kept round is not in its snapshot: put it back as it
        # was before the loop too (config-env-overrides kept round 1 while round 2's schema.py
        # stayed, and the package failed its own tests).
        later = {path: config.originals[path] for path in config.snapshot()
                 if path not in kept and path in config.originals}
        if later:
            features.mark_acted("restore_later_files", paths=sorted(later))
        kept = {**kept, **later}
    restore(host, config, kept, failing=best[3][0] if config.score is None else best[3])
    return LoopResult(attempts, best[1], kept, messages, refused, handover)


def restore(host: PipelineHost, config: LoopConfig, snapshot: dict[str, str | None], *,
            failing: int | tuple[int, ...]) -> None:
    """Put the kept round's files back where a later round changed them. A file the loop created
    that the kept round did not have is removed again."""
    project = host.project
    changed = False
    for relative, content in snapshot.items():
        path = project / relative
        if _read(path) == content:
            continue
        if content is None:
            if config.originals.get(relative) is None and path.is_file():
                path.unlink()
                changed = True
            continue
        execute("write_file", {"path": relative, "content": content}, project,
                before_write=host.before_write, atomic=False)
        changed = True
    if changed:
        debuglog.record("stage", stage=f"{config.name}.restore_best", failing=failing)
        host.refresh_written()


# -- one honest answer ----------------------------------------------------------------------------

def checks_ran(kinds: dict[str, int]) -> str:
    """The checks by kind, for the answer: "2 docstring examples, the tests, 1 task check",
    "`python check.py`". Only the kinds that ran."""
    parts = []
    for kind, count in kinds.items():
        if not count:
            continue
        if kind == "tests":
            parts.append("the tests")
        elif kind.startswith("`"):  # a named command
            parts.append(kind)
        else:
            parts.append(f"{count} {kind if count != 1 else kind.removesuffix('s')}")
    return ", ".join(parts)


def honest_answer(changed: list[str], failed: int, kinds: dict[str, int], *, repairs: int,
                  kept: list[str] | None = None, done: str = "Implemented the code in",
                  nothing: str = "checks", what: str = "code") -> str:
    """The answer of a pipeline on this loop, worded as the project pipeline's honest answer:
    the files changed (`changed`; `kept`: targets whose kept version is the same as before), how
    many checks of which kind the kept round passed; when no check ran it says nothing was
    verified (direct "Implemented ... pass" answers with no check behind them mostly failed the
    hidden tests, 2026-10-16); no claim of a change when nothing changed."""
    total = sum(kinds.values())
    ran = checks_ran(kinds)
    if not changed:
        answer = (f"No file was changed: the {what} kept for {', '.join(kept or [])} is the same "
                  "as before")
        if total == 0:
            return answer + ", and no check ran, so nothing was verified."
        return answer + f"; {max(total - failed, 0)} of {total} checks pass ({ran})."
    names = ", ".join(changed)
    if total == 0:
        return f"Wrote {names}, but nothing was verified: there were no {nothing} to run."
    if failed == 0:
        return f"{done} {names}; {total} of {total} checks pass ({ran})."
    return (f"Wrote {names}, but only {max(total - failed, 0)} of {total} checks pass ({ran}) "
            f"after {repairs} repair attempts; it may need another look.")


def failing_lines(failures: list[str]) -> str:
    """failing_checks_quoted: the first failing check lines of the kept attempt, quoted, so the
    answer says what fails (2026-10-17: "1 check(s) still
    fail" hid "No module named 'yamlkit'")."""
    lines = []
    for failure in failures:
        line = " ".join(str(failure).split())
        if line and line not in lines:
            lines.append(line)
    if not lines:
        return ""
    shown = [f"- {line if len(line) <= QUOTED_CHARS else line[:QUOTED_CHARS - 1] + '…'}"
             for line in lines[:QUOTED_FAILURES]]
    more = (f"\n- … and {len(lines) - QUOTED_FAILURES} more"
            if len(lines) > QUOTED_FAILURES else "")
    return "\nStill failing:\n" + "\n".join(shown) + more


def left_note(left: dict[str, str]) -> str:
    """Targets not written, with why: "Not written: b.json (not produced)."""
    if not left:
        return ""
    return " Not written: " + "; ".join(f"{path} ({why})" for path, why in left.items()) + "."


def answer_for(result: LoopResult, config: LoopConfig, *, done: str = "Implemented the code in",
               nothing: str = "checks", what: str = "code", agent: Any = None) -> str:
    """honest_answer for a finished loop, with the failing lines quoted and the files left
    (`agent`: the pipeline's agent, passed on to the final_answer hook)."""
    kept = result.kept
    assert kept is not None
    changed = [p for p in config.targets
               if result.snapshot.get(p) is not None
               and result.snapshot.get(p) != config.originals.get(p)]
    unchanged = [p for p in config.targets if p not in changed and result.snapshot.get(p)
                 is not None]
    answer = honest_answer(changed, kept.failed, kept.kinds, repairs=config.repairs,
                           kept=unchanged or config.targets[:1], done=done, nothing=nothing,
                           what=what)
    if kept.failed and features.on("failing_checks_quoted"):
        quoted = failing_lines(kept.failures)
        if quoted:
            features.mark_acted("failing_checks_quoted", where="write_loop")
        answer += quoted
    left = {}
    for path in config.targets:
        if result.snapshot.get(path) is None:
            why = result.refused.get(path, "")
            left[path] = (why.splitlines()[0][:160] if why else "not produced")
    return extensions.final_answer(answer + left_note(left), source=config.name, result=result,
                                   config=config, changed=changed, left=left, what=what,
                                   **({"agent": agent} if agent is not None else {}))


# -- the named check command ----------------------------------------------------------------------

def command_check(command: str, project: Path, *, stop: threading.Event | None = None,
                  timeout: int = 180) -> tuple[bool, str]:
    """Run a check command the task names in the sandbox: (passed, failing output excerpt)."""
    from . import check_feedback, sandbox

    output = sandbox.run_check(check_feedback.short_tracebacks(command), project, timeout=timeout,
                               stop=stop)
    code = check_feedback.exit_code(output)
    if code == 0:
        return True, ""
    if code is None:
        return False, output.strip()[-1500:] or "the command did not run"
    return False, f"`{command}` exit code {code}\n" + check_feedback.excerpt(output, 1500)
