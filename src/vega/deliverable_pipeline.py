"""Deliverable pipeline: tasks whose result is a new file worked out from the project's inputs.

Product baseline (2026-10-14): the agent loop passed almost none of these tasks. Those tasks share
one shape (read a few small inputs, work something out, write a stated JSON/Markdown/CSV file)
and the loop handled it badly: plan churn, step limits, a deliverable never written, an input
file edited, arithmetic done in the model's head. This path:

1. shows every input file in one prompt (whole when small, a head and line count when large);
2. asks for either the deliverable files directly (judgement: labels, notes, stories) or a
   Python script that computes them from the inputs (counting, dates, sums, optimisation);
3. runs a script in the sandbox and collects the files it wrote;
4. checks the files (present, JSON parses) and repairs at most REPAIRS times;
5. writes only the deliverables; inputs are never changed.

Switch: `deliverable_pipeline` (default on since 2026-10-17: the agent loop handled this shape
of task poorly). `deliverable_wide` (default on) widens it: an existing file the task asks to
update or regenerate (`Update requirements.lock`) is a deliverable too (shown with its current
content; an unchanged copy does not count as written), binary inputs are listed by name and size,
and when the task names a command (`python tools/run.py <arguments>`) the writer may first
reply with `Run: <command>` lines: Vega runs them in the sandbox and shows the output (≤2 rounds).

Added 2026-10-20, each behind its own switch: a script and a File: block that give a
JSON field different words get one repair round (`deliverable_disagreement`); the task's own
format rules are checked (`format_rules`, format_rules.py); the script runs under
deliverable_runner.py, which shows the failing line's values (`script_error_values`) and refuses
host reads (`sandbox_read_guard`); a script under a deliverable's label is the script
(`deliverable_code_label`).
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import acceptance, debuglog, features, format_rules, json_template, sandbox, write_loop
from .pipeline import ask_complete, with_context
from .tools import _project_files

if TYPE_CHECKING:
    from .agent import CodingAgent

REPAIRS = 2
SCRIPT_REPAIRS = 3  # up to 4 rounds while the script keeps failing
SCRIPT_NAME = "deliverable_script.py"  # its name in tracebacks, in the project root
DEFAULT_NUM_CTX = 8192
WHOLE_CHARS = 6000  # an input shown whole up to this size
HEAD_LINES = 30  # otherwise its first lines and its length
# deliverable_wide: (whole up to, else head lines) tried in turn while the inputs exceed the budget
TIGHT = [(2500, 10), (1200, 5)]
PROMPT_SHARE = 0.55  # of the window (in characters) for the task and the inputs; the rest replies
PROMPT_OVERHEAD = 2500  # system prompt, project context and the wrapper text
MAX_INPUT_FILES = 40
OUTPUT_LIMIT = 400_000
SCRIPT_DIR = ".vega-deliverable"
MARKER = "<<<VEGA-DELIVERABLES>>>"
_TARGET = re.compile(r"`([\w./-]+\.(?:json|md|csv|txt|ya?ml))`")
_WIDE_TARGET = re.compile(r"`([\w./-]+\.(?:json|jsonl|md|csv|tsv|txt|ya?ml|lock))`")
# an existing file is a deliverable when the words right before it ask to (re)write it:
# "Update `requirements.lock`", "regenerate the `report.json`",
# "write the answer to `answers/x.json`"
_UPDATE_CUE = re.compile(
    r"(?:\b(?:update|regenerate|refresh|recompute|rebuild|rewrite|redo|re-?create|overwrite|"
    r"fill in|correct|write|produce|generate)\s+(?:the\s+|your\s+|a\s+new\s+)?(?:[\w-]+\s+){0,2}"
    r"|\b(?:write|put|save|store|record)\b[^.;\n`]{0,60}\b(?:to|in|into)\s+(?:the\s+)?)$",
    re.IGNORECASE)
_NOT_BEFORE = re.compile(r"(?:\b(?:not|never|without|except)\b|n't\b)[^.;\n]*$", re.IGNORECASE)
# a task that reports line numbers ("lines are 1-based line numbers", `"line": <int>`)
_LINES = re.compile(r"\bline(?:s| numbers?)?\b", re.IGNORECASE)
_CODE_EXT = {"py", "js", "mjs", "cjs", "ts", "tsx", "jsx", "go", "rs", "java", "rb", "php", "c", "h",
             "cc", "cpp", "cs", "sql", "sh", "r", "kt", "swift", "html", "css"}
LOOK_ROUNDS = 2  # deliverable_wide: rounds of `Run:` commands before the deliverables
LOOK_COMMANDS = 3  # commands per round
LOOK_OUTPUT = 6000  # characters of each command's output shown
_RUN_LINE = re.compile(r"^\s*(?:[-*]\s*)?Run:\s*`?([^`\n]+?)`?\s*$", re.MULTILINE)
_NAMED_COMMAND = re.compile(r"`((?:python3?|Rscript|node|sqlite3)\s+[^`\n]+)`")
_SCRIPT = re.compile(r"```python[ \t]*\n(.*?)```", re.DOTALL)
VALUE_FRAMES = 3  # script_error_values: the script's innermost frames whose values are shown
VALUE_WIDTH = 160  # characters of each value's repr
SYSTEM = (
    "You produce the file(s) a task asks for, from the project's input files shown to you. Follow "
    "the task's stated format exactly (paths, field names, ordering, rounding, tie-breaks). "
    "If producing them needs counting, arithmetic, dates, sorting, optimisation or reading more "
    "rows than are shown, reply with ONE Python script in a ```python block that reads the "
    "input files (paths relative to the project root) and writes every deliverable file; use only "
    "the standard library, and work everything out in the script, not by hand. Otherwise reply "
    "with each deliverable as a line `File: <path>` followed by its full content in a fenced "
    "block. Never change the input files."
)
WIDE_SYSTEM = (
    "You produce the file(s) a task asks for, from the project's input files shown to you. Follow "
    "the task's stated format exactly (paths, field names, ordering, rounding, tie-breaks). "
    "If producing them needs counting, arithmetic, dates, hashes, sorting, optimisation or reading "
    "more rows than are shown, reply with ONE Python script in a ```python block that reads the "
    "input files (paths relative to the project root) and writes every deliverable file; use the "
    "standard library (and any library the task says is installed), and work everything out in "
    "the script, not by hand. A script may run the project's own tools with subprocess "
    "(sys.executable for python). Otherwise reply with each deliverable as a line `File: <path>` "
    "followed by its full content in a fenced block. A deliverable that exists already is shown "
    "with its current content: write its whole new version. Never change the input files."
)
LOOK_NOTE = (
    "\n\nThe task names these commands: {commands}. If you need their output before you can write "
    "the deliverables (e.g. to read a diff or a log), reply with ONLY lines `Run: <command>` (at "
    "most {limit}, real arguments, no shell syntax); Vega runs them in a sandbox and shows you the "
    "output. Otherwise reply with the script or the File: blocks now."
)
RUNNER = """
import json, os, sys
from pathlib import Path
targets = json.loads(sys.argv[1])
for target in targets:  # the deliverables' folders exist before the script runs
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
source = Path(sys.argv[2]).read_text(encoding="utf-8")
path = os.path.abspath({name!r})  # as if it lived in the project root (cwd)
namespace = {{"__name__": "__main__", "__file__": path}}
exec(compile(source, path, "exec"), namespace)
print({marker!r})
print(json.dumps({{t: Path(t).read_text(encoding="utf-8") for t in targets if Path(t).is_file()}}))
"""


def _fenced(text: str) -> str:
    """`text` in a fence it cannot close early (a file holding ``` gets ````; fence_lines)."""
    mark = write_loop.fence_for(text)
    return f"{mark}\n{text}\n{mark}"


def targets(project: Path, task: str) -> list[str]:
    """Deliverable paths the task names in backticks that do not exist yet; with
    deliverable_wide also existing ones the task asks to update or regenerate."""
    wide = features.on("deliverable_wide")
    found: list[str] = []
    for match in (_WIDE_TARGET if wide else _TARGET).finditer(task):
        path = match.group(1)
        if path in found or (wide and ".." in path):
            continue
        new = not (project / path).exists()
        if new or (wide and (project / path).is_file() and asks_update(task, match.start())):
            found.append(path)
    return found


def asks_update(task: str, start: int) -> bool:
    """The words right before the backticked path at `start` ask to (re)write it."""
    sentence = re.split(r"(?<=[.;:!?])\s|\n", task[max(0, start - 120):start])[-1]
    return bool(_UPDATE_CUE.search(sentence)) and not _NOT_BEFORE.search(sentence)


def named_commands(project: Path, task: str) -> list[str]:
    """Commands the task names in backticks whose script is in the project
    (`python tools/run.py <arguments>`)."""
    found: list[str] = []
    for command in _NAMED_COMMAND.findall(task):
        words = command.split()
        script = next((w for w in words[1:] if not w.startswith("-")), "")
        if script and (project / script).is_file() and command not in found:
            found.append(command)
    return found


def inputs(project: Path, exclude: set[str], current: set[str] | None = None,
           budget: int | None = None, numbered: bool = False) -> str:
    """Every project file for the prompt (`current`: deliverables that exist, shown as such).
    deliverable_wide: when the whole text is over `budget` characters, files are shown shorter
    (TIGHT tiers) so the prompt and the reply still fit the window; `numbered`: code files get
    line numbers (a review that reports lines)."""
    wide = features.on("deliverable_wide")
    texts: list[tuple[str, str | None, int]] = []  # (path, text or None for binary, size)
    for path in _project_files(project, limit=MAX_INPUT_FILES):
        relative = path.relative_to(project).as_posix()
        if relative in exclude or relative.startswith(SCRIPT_DIR):
            continue
        try:
            texts.append((relative, path.read_text(encoding="utf-8"), 0))
        except (OSError, UnicodeDecodeError):
            if wide:
                try:
                    texts.append((relative, None, path.stat().st_size))
                except OSError:
                    continue
    tiers = [(WHOLE_CHARS, HEAD_LINES), *TIGHT] if wide and budget else [(WHOLE_CHARS, HEAD_LINES)]
    shown = ""
    for whole, head_lines in tiers:
        parts = []
        for relative, text, size in texts:
            if text is None:
                parts.append(f"{relative} (binary, {size} bytes; read it in a script)")
            elif current and relative in current:
                parts.append(f"{relative} (the deliverable's CURRENT version; write its whole new "
                             f"version):\n{_fenced(text)}")
            elif len(text) <= whole:
                if numbered and relative.rsplit(".", 1)[-1].lower() in _CODE_EXT:
                    body = "\n".join(f"{n:>4}| {line}" for n, line in
                                     enumerate(text.splitlines(), 1))
                    parts.append(f"{relative} (line numbers added for you; not in the file):"
                                 f"\n{_fenced(body)}")
                else:
                    parts.append(f"{relative}:\n{_fenced(text)}")
            else:
                lines = text.splitlines()
                head = "\n".join(line[:300] for line in lines[:head_lines])
                parts.append(f"{relative} (first {head_lines} of {len(lines)} lines; read the "
                             f"whole file in a script):\n{_fenced(head)}")
        shown = "\n\n".join(parts)
        if budget is None or len(shown) <= budget:
            break
    return shown


_FIXTURE_ROOT = re.compile(r"`(\w[\w.-]*)/`\s+(?:stands for|is|represents|maps to)\s+`/`")


def fixture_root(task: str) -> str:
    """The project folder a task maps to `/` ("`root/` stands for `/`"), or ''."""
    match = _FIXTURE_ROOT.search(task)
    return match.group(1) if match else ""


def _runner_options(fixture: str) -> dict[str, Any] | None:
    """The runner's options (deliverable_runner.py), or None for the plain RUNNER (both
    script_error_values and sandbox_read_guard off)."""
    values, guard = features.on("script_error_values"), features.on("sandbox_read_guard")
    if not values and not guard:
        return None
    return {"name": SCRIPT_NAME, "marker": MARKER, "values": values, "values_frames": VALUE_FRAMES,
            "values_width": VALUE_WIDTH, "guard": guard, "fixture_root": fixture,
            "real_home": str(Path.home())}


def _run_script(project: Path, script: str, wanted: list[str],
                fixture: str = "") -> tuple[dict[str, str], str]:
    """Run the model's script in the sandbox; (files it wrote, error output or ''). It runs from
    the project root as SCRIPT_NAME there (so `Path(__file__).parent` is the project), with the
    deliverables' folders made first and HOME pointing at the sandbox's copy. `fixture`: the
    folder the task maps to `/` (named in a refused host read)."""
    folder = project / SCRIPT_DIR
    folder.mkdir(exist_ok=True)
    (folder / "make.py").write_text(script, encoding="utf-8")
    options = _runner_options(fixture)
    if options is None:
        (folder / "run.py").write_text(RUNNER.format(marker=MARKER, name=SCRIPT_NAME),
                                       encoding="utf-8")
        extra = ""
    else:
        runner = Path(__file__).with_name("deliverable_runner.py").read_text(encoding="utf-8")
        (folder / "run.py").write_text(runner, encoding="utf-8")
        extra = " " + shlex.quote(json.dumps(options))
    command = (f"python {SCRIPT_DIR}/run.py {shlex.quote(json.dumps(wanted))} "
               f"{SCRIPT_DIR}/make.py{extra}")
    output = sandbox.run_check(command, project, output_limit=OUTPUT_LIMIT, private_home=True)
    if MARKER not in output:
        return {}, output[-3000:]
    try:
        files = json.loads(output.split(MARKER, 1)[1].strip().splitlines()[0])
    except (ValueError, IndexError):
        return {}, "the script's output could not be read"
    return files, ""


# a task that asks for every finding: an empty list is not an answer by itself
_EVERY = re.compile(r"\b(?:report|list|flag|find|include|record)\s+(?:each|every|all)\b|"
                    r"\bfull review\b|\breview (?:the|every|all)\b|\bevery (?:finding|issue|problem|"
                    r"bug|vulnerability|violation)\b", re.IGNORECASE)


def _empty_list(value: Any) -> bool:
    if isinstance(value, list):
        return not value
    if isinstance(value, dict) and value:
        lists = [v for v in value.values() if isinstance(v, list)]
        return bool(lists) and all(not v for v in lists)
    return False


def check(files: dict[str, str], wanted: list[str],
          originals: dict[str, str] | None = None, task: str = "") -> list[str]:
    """The deliverables' format checks: produced, changed (an update), JSON that parses, not
    empty, and no empty list when the task asks to report every finding."""
    problems = [f"{path} was not produced" for path in wanted if path not in files]
    for path, before in (originals or {}).items():
        if path in files and files[path] == before:
            problems.append(f"{path} is unchanged: write its new version")
    for path, text in files.items():
        if path.endswith(".json"):
            try:
                value = json.loads(text)
            except ValueError as exc:
                problems.append(f"{path} is not valid JSON: {exc}")
            else:
                if _EVERY.search(task) and _empty_list(value):
                    problems.append(f"{path} lists nothing, but the task asks to report every "
                                    "finding: look again at the inputs (an empty list passes "
                                    "only when there is truly nothing to report)")
        if not text.strip():
            problems.append(f"{path} is empty")
    return problems


def _look(agent: CodingAgent, messages: list[dict[str, Any]], looks: list[int]) -> str:
    """A reply that is not only `Run:` lines; such lines are run in the sandbox (≤LOOK_ROUNDS
    rounds in all, counted in `looks`) and their output shown before asking again."""
    while True:
        reply = ask_complete(agent, messages)
        commands = _RUN_LINE.findall(reply)
        if (not commands or looks[0] >= LOOK_ROUNDS or _SCRIPT.search(reply)
                or "```" in reply):
            return reply
        looks[0] += 1
        shown = []
        for command in commands[:LOOK_COMMANDS]:
            command = command.strip()
            agent._emit("tool_call", name="run_check", args={"command": command})
            output = sandbox.run_check(command, agent.project, output_limit=OUTPUT_LIMIT,
                                       private_home=True)
            agent._emit("tool_result", name="run_check", args={"command": command}, result=output)
            if len(output) > LOOK_OUTPUT:
                output = output[:LOOK_OUTPUT] + "\n[output cut short]"
            shown.append(f"$ {command}\n{output}")
        debuglog.record("stage", stage="deliverable_pipeline.look",
                        commands=commands[:LOOK_COMMANDS])
        last = looks[0] >= LOOK_ROUNDS
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "\n\n".join(shown) + (
                         "\n\nNow reply with the script or the File: blocks." if last else
                         "\n\nReply with the script or the File: blocks, or with more `Run:` "
                         "lines if you still need output.")}]


def file_blocks(reply: str, wanted: list[str]) -> tuple[dict[str, str], list[str]]:
    """({deliverable: content} from `File: path` blocks, [Python scripts]). A block's path label
    may be on the line before the fence, in its info string or as the first line inside it (that
    line is not content); a lone non-Python block is the deliverable when there is exactly one.
    An unlabelled ```python block is a script."""
    found: dict[str, str] = {}
    scripts: list[str] = []
    loose: list[str] = []
    code_label = features.on("deliverable_code_label")
    for language, label, body in write_loop.reply_blocks(reply):
        owner = next((p for p in wanted if label and write_loop._label_matches(label, p)), None)
        if owner is not None and code_label and _script_under_label(language, owner, body):
            scripts.append(body)  # "Python script to generate `answers/x.json`": a script
        elif owner is not None:
            first, _, rest = body.lstrip("\n").partition("\n")
            if _LABEL_LINE.match(first.strip()) and label in first:
                body = rest
            found.setdefault(owner, body)
        elif language in {"python", "py", "python3"} and not label:
            scripts.append(body)
        elif not label:
            loose.append(body)
    if not found and not scripts and len(wanted) == 1 and len(loose) == 1:
        found[wanted[0]] = loose[0]
    return found, scripts


_PY_START = re.compile(r"^\s*(?:import\s+\w|from\s+[\w.]+\s+import\b|def\s+\w|#!)", re.MULTILINE)


def _script_under_label(language: str, owner: str, body: str) -> bool:
    """deliverable_code_label (2026-10-20): a block labelled with a
    non-Python deliverable that holds a Python script (```python, or a body that is not JSON but
    starts like Python) is the script, not the deliverable's content; refused as invalid JSON, the
    model was asked to resend it "in the same form"."""
    if owner.endswith((".py", ".pyi")):
        return False
    if language in {"python", "py", "python3"}:
        return True
    if language not in {"", "text"} or not _PY_START.search(body[:400]):
        return False
    if owner.endswith(".json"):
        try:
            json.loads(body)
            return False
        except ValueError:
            pass
    try:
        compile(body, "<reply>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def disagreement(script_files: dict[str, str], blocks: dict[str, str]) -> list[str]:
    """deliverable_disagreement: where a script's output and a `File:` block of the same reply
    give a JSON object field different word values (strings, booleans, null, or one leaves the
    field out), one line per such field quoting both. Numbers, lists and objects are not compared:
    those are what a script computes, and in replies that had both (2026-10-17..20) the block
    often held hand-computed numbers that were wrong while the script was right. A looked-up
    word (a config's `no` vs `yes`) is where the block was right and the script wrong."""
    out: list[str] = []
    for path, block in blocks.items():
        if path not in script_files or not path.endswith(".json"):
            continue
        try:
            ours, theirs = json.loads(script_files[path]), json.loads(block)
        except ValueError:
            continue
        if not isinstance(ours, dict) or not isinstance(theirs, dict) or ours == theirs:
            continue
        differences = []
        for key in dict.fromkeys([*ours, *theirs]):
            mine, yours = ours.get(key, _MISSING), theirs.get(key, _MISSING)
            if mine != yours and _word(mine) and _word(yours):
                differences.append(f'"{key}": the script wrote {_quote(ours, key)}, your File: '
                                   f"block has {_quote(theirs, key)}")
        if not differences:
            continue
        shown = differences[:DISAGREEMENT_LINES]
        if len(differences) > len(shown):
            shown.append(f"... and {len(differences) - len(shown)} more difference(s)")
        out.append(f"{path}: your script and your File: block disagree:\n"
                   + "\n".join(f"  - {line}" for line in shown))
    return out


def _word(value: Any) -> bool:
    """A looked-up value rather than a computed one: a string, a boolean, null or absent."""
    return value is _MISSING or value is None or isinstance(value, (str, bool))


_MISSING = object()
DISAGREEMENT_LINES = 8


def _clip(text: str, limit: int = 100) -> str:
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _quote(data: dict[str, Any], key: str) -> str:
    return "nothing" if key not in data else _clip(json.dumps(data[key], ensure_ascii=False))


DISAGREE_NOTE = (
    "Work out from the inputs which one is right. Then either fix the script so it writes the "
    "right values, or reply with File: blocks only (no script); do not send both again unless "
    "they agree.")


_LABEL_LINE = re.compile(r"^(?:#|//|--)?\s*(?:File|file|Path|path):?\s*`?[\w./-]+`?\s*$")
_TRACE_LINE = re.compile(r'File "[^"]*make\.py", line (\d+)|File "[^"]*' + re.escape(
    "deliverable_script.py") + r'", line (\d+)')


class _Producer:
    """Replies → deliverable files, remembering the last script error for the repair feedback."""

    def __init__(self, agent: CodingAgent, task: str, wanted: list[str],
                 originals: dict[str, str]):
        self.agent, self.task, self.wanted, self.originals = agent, task, wanted, originals
        self.last_reply = ""
        self.last_error = ""
        self.script_errors: list[str] = []  # one per round that ended on a script error
        self.rounds = 0
        self.failed_last = False  # the last round ended on a script error
        self.disagreement_asked = False  # deliverable_disagreement: its one repair round is used
        self.fixture = fixture_root(task)

    def __call__(self, reply: str) -> write_loop.Produced:
        self.rounds += 1
        self.failed_last = False
        same_reply = reply.strip() == self.last_reply.strip()
        self.last_reply = reply
        direct, scripts = file_blocks(reply, self.wanted)
        files: dict[str, str] = {}
        disagree: list[str] = []
        if scripts:  # a script is run whenever there is one; File blocks fill only what it left
            script = max(scripts, key=len)
            self.agent._emit("tool_call", name="run_check",
                             args={"command": "python (deliverable script)"})
            files, error = _run_script(self.agent.project, script, self.wanted,
                                       self.fixture)
            self.agent._emit("tool_result", name="run_check",
                             args={"command": "python (deliverable script)"},
                             result=error or f"wrote {', '.join(files) or 'nothing'}")
            if error:
                self.script_errors.append(error)
                self.failed_last = True
                return write_loop.Produced(problem=self._script_feedback(error, script,
                                                                         same_reply))
            if features.on("deliverable_disagreement") and not self.disagreement_asked:
                differ = disagreement({p: t for p, t in files.items() if p in self.wanted},
                                      direct)
                if differ:  # one repair round; the script's files are this round's candidate
                    self.disagreement_asked = True
                    debuglog.record("stage", stage="deliverable_pipeline.disagreement",
                                    paths=[line.split(":", 1)[0] for line in differ])
                    disagree = [*differ[:-1], differ[-1] + "\n" + DISAGREE_NOTE]
            for path, body in direct.items():
                files.setdefault(path, body)
        else:
            files = dict(direct)
        files = {path: content for path, content in files.items() if path in self.wanted}
        problems = disagree + check(files, self.wanted, self.originals, self.task)
        writes = {path: write_loop.Write(content) for path, content in files.items()}
        if not writes:
            return write_loop.Produced(problem="\n".join(problems or [
                "Your reply had no script and no `File: <path>` block."]) + RETRY)
        return write_loop.Produced(writes, problems, checked=len(self.wanted))

    def _script_feedback(self, error: str, script: str, same_reply: bool) -> str:
        last_line = next((line.strip() for line in reversed(_error_core(error).splitlines())
                          if line.strip()), "")
        notes = []
        if same_reply:
            notes.append("Your reply is the same as last round's, which failed the same way: "
                         "change the script.")
        elif last_line and last_line == self.last_error:
            notes.append(f"Same failure as last round ({last_line}).")
        lines = script.split("\n")
        numbers = [int(a or b) for a, b in _TRACE_LINE.findall(error)]
        if numbers and 0 < numbers[-1] <= len(lines):
            notes.append(f"Line {numbers[-1]} of your script is: "
                         f"`{lines[numbers[-1] - 1].strip()}`")
        self.last_error = last_line
        return (f"The script failed:\n{error}" + ("\n\n" + "\n".join(notes) if notes else "")
                + RETRY)


RETRY = "\n\nFix this and reply again in the same way (a script, or File: blocks)."
VALUES_HEADER = "\nValues at the failing line(s) of your script:"  # deliverable_runner.report


def _error_core(error: str) -> str:
    """A script error without the runner's values section (script_error_values): its last line
    is the exception again, for "same failure" and the honest stop's quote."""
    return error.split(VALUES_HEADER, 1)[0].rstrip()
_SAME_FORM = re.compile(r"Reply again with the corrected code for (\S+) in the same form as "
                        r"before\.")


def _same_form(refusal: str) -> str:
    """deliverable_code_label: a refused deliverable that is not code is not asked
    for again "in the same form as before" (that form was the mistake: code under a JSON label)."""
    if not features.on("deliverable_code_label"):
        return refusal

    def instead(match: re.Match[str]) -> str:
        path = match.group(1)
        if path.endswith((".py", ".pyi")):
            return match.group(0)
        return (f"Reply again with {path}'s own content (not code) in a `File: {path}` block, or "
                f"with an unlabelled ```python script that writes {path}.")

    return _SAME_FORM.sub(instead, refusal)


def refusals(project: Path, files: dict[str, str]) -> dict[str, str]:
    """path -> the atomic check's refusal (parse error with its lines) for each produced file
    that would not land; {} when all would."""
    from . import edits

    refused = {}
    for path, content in files.items():
        target = project / path
        before = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else None
        refusal = edits.check("pipeline", path, before, content, project).refusal
        if refusal is not None:
            refused[path] = refusal
    return refused


def run(agent: CodingAgent, task: str) -> str | None:
    """Run a task routing.route_task sent here on write_loop: every round's files are written
    and checked (format checks + acceptance), the round with the fewest failing checks is kept.
    None (no deliverable found, or nothing it produced would parse) falls back to the agent
    loop; when every round ended on a script error the answer says so and quotes the error."""
    project = agent.project
    wanted = targets(project, task)
    if not wanted:
        return None
    wide = features.on("deliverable_wide")
    originals: dict[str, str] = {}
    for path in wanted:
        if (project / path).is_file():
            try:
                originals[path] = (project / path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                originals[path] = ""
    commands = named_commands(project, task) if wide else []
    host = write_loop.AgentHost(agent)
    host.emit("notice", message=f"Deliverable pipeline: {', '.join(wanted)}")
    look = LOOK_NOTE.format(commands=", ".join(f"`{c}`" for c in commands),
                            limit=LOOK_COMMANDS) if commands else ""
    budget = None
    if wide:
        budget = max(4000, write_loop.window_chars(host.num_ctx, PROMPT_SHARE) - len(task)
                     - PROMPT_OVERHEAD)
    shown = inputs(project, set(wanted) - set(originals), set(originals), budget,
                   numbered=wide and bool(_LINES.search(task)))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": with_context(agent, WIDE_SYSTEM if wide else SYSTEM)},
        {"role": "user", "content": f"Task:\n{task}\n\nDeliverables to write: "
                                    f"{', '.join(wanted)}\n\nInput files:\n\n{shown}{look}"},
    ]
    looks = [0] if commands else None
    produce = _Producer(agent, task, wanted, originals)
    accept_spec = host.acceptance(task)

    def check_stage(written: list[str]) -> write_loop.Checks:
        checks = write_loop.Checks()
        templated = [path for path in json_template.stated(task) if path in written]
        if templated and features.on("json_template"):  # the task's `X.json as {...}` keys
            checks.kinds["template keys"] = len(templated)
            checks.failures += json_template.problems(project, task, templated)
        stated_rules = format_rules.rules(task, wanted) if features.on("format_rules") else None
        if stated_rules is not None and stated_rules.count:  # the task's own format rules
            texts = {}
            for path in written:
                try:
                    texts[path] = (project / path).read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
            checks.kinds["format rules"] = stated_rules.count
            checks.failures += format_rules.problems(task, texts, stated_rules)
        if accept_spec is not None:
            accepted = acceptance.run(accept_spec, project, written, tests=False,
                                      stop=host.stop_flag())
            checks.kinds["task checks"] = len(accepted.items)
            checks.failures += acceptance.pipeline_failures(accepted)
        return checks

    def stop_early(round_number: int) -> bool:  # past REPAIRS only while scripts still fail
        return round_number >= REPAIRS and not produce.failed_last

    config = write_loop.LoopConfig(
        name="deliverable_pipeline", title="Deliverable pipeline", targets=wanted,
        originals={path: originals.get(path) for path in wanted}, repairs=SCRIPT_REPAIRS,
        produce=produce, check=check_stage, no_code="Reply with a script or File: blocks." + RETRY,
        not_saved=lambda refused, problems: "\n".join([*problems, *map(_same_form, refused)])
        + RETRY,
        repair="{feedback}" + RETRY, check_call="deliverable checks",
        ask=(lambda convo: _look(agent, convo, looks)) if looks is not None else None,
        stop_early=stop_early)
    try:
        result = write_loop.run_loop(host, config, messages)
    finally:
        import shutil

        shutil.rmtree(project / SCRIPT_DIR, ignore_errors=True)
    if result is None:
        return None
    if result.kept is None:
        if produce.script_errors and len(produce.script_errors) == produce.rounds:
            return _script_failed(host, task, wanted, produce)
        if result.refused:
            host.emit("notice", message="Deliverable pipeline: nothing it produced would parse ("
                                        + ", ".join(result.refused) + "); using the agent loop.")
        return None
    answer = write_loop.answer_for(result, config, done="Wrote", what="content")
    host.remember(task, answer)
    return answer


def _script_failed(host: write_loop.PipelineHost, task: str, wanted: list[str],
                   produce: _Producer) -> str:
    """The honest stop: every round's script failed, nothing was written. Falling back to the
    agent loop here was slow and mostly wrote made-up content."""
    error = produce.script_errors[-1]
    lines = [line for line in _error_core(error).splitlines() if line.strip()]
    quoted = "\n".join(f"    {line}" for line in lines[-6:])
    answer = (f"Not written: {', '.join(wanted)}. The script that computes "
              f"{'it' if len(wanted) == 1 else 'them'} failed in all {produce.rounds} rounds, so "
              f"nothing was produced and nothing was guessed. The last error:\n{quoted}")
    debuglog.record("stage", stage="deliverable_pipeline.script_failed", rounds=produce.rounds)
    host.remember(task, answer)
    return answer
