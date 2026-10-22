"""issue_localize: point the agent loop at the files most likely responsible for an issue.

Why (2026-10-21): on a larger codebase (~100 modules), the model mostly edited the wrong file.
The issue text almost always carries something the right file holds verbatim: a quoted
message ("Total due"), a `backticked` name, a snake_case identifier, a /command or --flag, a file name. This module takes those literals, finds the files holding each
one (a pure Python scan: no git grep, no model call), and ranks files by how rare their literals
are, with extra weight for a file that *defines* a named function/class or registers a named
tool/command as a string. The agent loop adds the top files, with the functions around the
matching lines, to the first prompt as a short "Likely places" block.

Deterministic and cheap: one read of each candidate text file (≤ MAX_FILES, ≤ MAX_FILE_BYTES).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .index import _list_files, _read_text, extract_symbols

MIN_FILES = 25  # projects with fewer candidate files get no block (the model sees them all)
MAX_FILES = 3000
MAX_FILE_BYTES = 400_000
TOP_FILES = 3
MAX_SYMBOLS = 3  # enclosing functions/classes shown per file
MAX_CHARS = 1400  # ~400 tokens at vega.context's 3.5 characters per token
MAX_LITERALS = 24
HEADER = ("Likely places (files holding the task's names and strings, ranked by a text search; "
          "a starting point, not a certainty: read them before deciding where to edit):")

# Not the code an issue is about: data, lock files, logs, generated output, binaries.
_SKIP_SUFFIXES = {
    ".csv", ".tsv", ".jsonl", ".ndjson", ".parquet", ".lock", ".log", ".svg", ".png", ".jpg",
    ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".tar", ".whl", ".pyc", ".map", ".min.js",
    ".ipynb", ".sqlite", ".db",
}
_SKIP_NAMES = {"package-lock.json", "yarn.lock", "poetry.lock", "uv.lock", "Cargo.lock",
               "pnpm-lock.yaml", "CHANGELOG.md"}

# kind -> weight of one literal of that kind (before rarity)
WEIGHTS = {
    "path": 3.0,      # a file the task names (calc.py, src/vega/tools.py)
    "quoted": 2.0,    # "a message the user saw", `backticked text`
    "fragment": 1.5,  # a piece of a quoted message no file holds whole (f-strings, joins)
    "ident": 1.0,     # snake_case / CamelCase / dotted / name()
    "command": 1.5,   # /help, --flag
    "word": 0.2,      # a plain distinctive word (fallback evidence)
}
# A named file the router (vega.routing) reads as one to change, or as an input/context/script
# to run: multipliers of the "path" weight.
NAMED_TARGET = 2.0
NAMED_INPUT = 0.3
DEFINES = 3.0     # multiplier: the file defines the named function/class/constant
REGISTERS = 1.5   # multiplier: the file holds the name as a string ("edit_file": a tool's name)
TEST_FILE = 0.5   # multiplier for test files: the issue is about the code, tests come second
DOC_FILE = 0.5    # multiplier for prose (README, *.md): they mention everything, fixes go in code
_DOC_SUFFIXES = (".md", ".rst", ".txt", ".adoc")
# file suffixes, so "check.py" is not read as the dotted name check.py -> `py`
_FILE_SUFFIXES = {"py", "pyi", "js", "ts", "tsx", "jsx", "c", "h", "cc", "cpp", "hpp", "go", "rs",
                  "rb", "java", "kt", "swift", "md", "txt", "json", "toml", "yaml", "yml", "cfg",
                  "ini", "sql", "sh", "html", "css", "r", "gd", "csv", "lock", "in", "mjs", "cjs"}
# a quoted command the task runs ("python check.py", `pytest -q`): not text in the code
_RUNNER = re.compile(r"^(?:python3?|pytest|py\.test|bash|sh|node|npm|npx|make|go|cargo|uv|pip|"
                     r"ruff|mypy|git|Rscript|clang|gcc|swift)\b")

_STOP_TEXT = """
a an the and or but if then else when while for to of in on at by with from into onto as is are
was were be been being it its this that these those there here i me my we our you your he she
they them their do does did doing done not no nor so too very can could should would will shall
may might must have has had having just only also even still yet than about above below after
before again further once all any both each few more most other some such own same what which who
whom whose why how where get gets got make makes made use uses used using need needs want wants
like work works working way new old file files code line lines function functions method class
please fix bug issue error broken wrong instead actually always never every something anything
thing things show shows shown see seen say says said right left first last one two three
"""
_STOP = set(_STOP_TEXT.split())

_QUOTED = [
    re.compile(r"`([^`\n]{2,120})`"),
    re.compile(r"\"([^\"\n]{3,120})\""),
    re.compile(r"“([^”\n]{3,120})”"),
    re.compile(r"(?<![\w'])'([^'\n]{3,80})'(?![\w'])"),
]
_IDENT = re.compile(r"(?<![\w./-])([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(\(?)")
_COMMAND = re.compile(r"(?<![\w/.~:<-])(/[a-z][a-z0-9_-]{1,24})\b(?![/\w]|\.\w)")
_FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]{1,30})\b")
_PATHISH = re.compile(r"(?<![\w-])((?:[\w.-]+/)*[\w-]+\.[A-Za-z][A-Za-z0-9]{0,5})\b")
_WORD = re.compile(r"\b[a-z][a-z]{4,}\b")


@dataclass
class Literal:
    text: str
    kind: str


@dataclass
class Place:
    path: str
    score: float
    reasons: list[str] = field(default_factory=list)
    lines: dict[int, float] = field(default_factory=dict)  # matching line -> evidence weight


def _is_code_ident(name: str, call: bool) -> bool:
    last = name.rsplit(".", 1)[-1]
    if len(name) < 3:
        return False
    if "." in name:
        return all(part for part in name.split("."))
    if "_" in name.strip("_"):
        return True
    if re.search(r"[a-z][A-Z]", name) or re.fullmatch(r"[A-Z][a-z0-9]+[A-Z]\w*", name):
        return True  # camelCase / PascalCase with an inner capital
    if re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", name) and last.upper() == last and len(name) > 3:
        return True  # CONSTANT
    return call and len(name) >= 3 and name.lower() not in _STOP


def literals(task: str) -> list[Literal]:
    """The literals of an issue text, most specific first, without duplicates."""
    found: list[Literal] = []
    seen: set[str] = set()

    def add(text: str, kind: str) -> None:
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            found.append(Literal(text, kind))

    for pattern in _QUOTED:
        for match in pattern.finditer(task):
            inner = match.group(1).strip()
            if re.fullmatch(r"[\w./-]+\.\w{1,5}", inner) and \
                    inner.rsplit(".", 1)[-1].lower() in _FILE_SUFFIXES:
                continue  # a quoted file name: handled as a path below
            if _RUNNER.match(inner):
                continue  # a command the task runs, not text in the code
            if re.fullmatch(r"[A-Za-z_][\w.]*(?:\(\))?", inner):
                name = inner.removesuffix("()")
                if _is_code_ident(name, True) and all(name.split(".")):
                    add(name, "ident")  # `edit_file`, `Store.export()`
                    if "." in name:
                        add(name.rsplit(".", 1)[-1], "ident")
                continue
            if len(inner) >= 3:
                add(inner, "quoted")
    for match in _PATHISH.finditer(task):
        add(match.group(1), "path")
    for match in _COMMAND.finditer(task):
        add(match.group(1), "command")
    for match in _FLAG.finditer(task):
        add(match.group(1), "command")
    for match in _IDENT.finditer(task):
        name, call = match.group(1), bool(match.group(2))
        if "." in name and name.rsplit(".", 1)[-1].lower() in _FILE_SUFFIXES:
            continue  # a file name (check.py)
        if _is_code_ident(name, call):
            add(name, "ident")
            if "." in name:
                add(name.rsplit(".", 1)[-1], "ident")
    words = [w for w in _WORD.findall(task.lower()) if w not in _STOP]
    for word in dict.fromkeys(words):
        add(word, "word")
    return found


def _fragments(text: str) -> list[str]:
    """The pieces of a quoted message between punctuation, 3+ words or 15+ characters each:
    "export error: Report folder is outside the workspace: q3.csv" ->
    ["Report folder is outside the workspace", "export error"]."""
    parts = [p.strip() for p in re.split(r"[:;,.!?()\[\]{}\"'`|]+|\s[-—]\s", text)]
    keep = [p for p in parts if p != text.strip()
            and (len(p.split()) >= 3 or (len(p) >= 15 and " " in p))]
    return sorted(dict.fromkeys(keep), key=len, reverse=True)[:3]


def _candidates(project: Path) -> list[str]:
    files = []
    for rel in _list_files(project):
        name = rel.rsplit("/", 1)[-1]
        lower = name.lower()
        if name in _SKIP_NAMES or any(lower.endswith(s) for s in _SKIP_SUFFIXES):
            continue
        if lower.startswith(".") and "." not in lower[1:]:
            continue  # .gitignore, .env and the like
        files.append(rel)
        if len(files) >= MAX_FILES:
            break
    return files


def _is_test(path: str) -> bool:
    parts = path.lower().split("/")
    name = parts[-1]
    return (any(p in {"tests", "test", "__tests__", "spec"} for p in parts[:-1])
            or name.startswith("test_") or name.endswith(("_test.py", ".test.js", ".test.ts",
                                                          ".spec.js", ".spec.ts", "_test.go")))


def _matcher(literal: Literal) -> re.Pattern[str]:
    text = re.escape(literal.text)
    if literal.kind in {"ident", "word"}:
        return re.compile(rf"(?<![\w]){text}(?![\w])", 0 if literal.kind == "ident"
                          else re.IGNORECASE)
    if literal.kind == "command":
        return re.compile(rf"(?<![\w/-]){text}(?![\w-])")
    return re.compile(text)


def _defines(name: str, text: str) -> bool:
    base = re.escape(name.rsplit(".", 1)[-1])
    return bool(re.search(
        rf"^\s*(?:async\s+def|def|class|function|func|fn|struct|interface|type|enum)\s+{base}\b"
        rf"|^{base}\s*[:=]|^\s*(?:export\s+)?(?:const|let|var)\s+{base}\s*=", text, re.MULTILINE))


def _registers(name: str, text: str) -> bool:
    return f'"{name}"' in text or f"'{name}'" in text


def _named_role(path: str, route: dict | None) -> float:
    """The "path" weight multiplier for a file the task names, from the router's reading."""
    if not route:
        return 1.0
    if path in (route.get("targets") or []) or path in (route.get("other_targets") or []):
        return NAMED_TARGET
    if any(path in (route.get(key) or []) for key in ("helpers", "inputs", "context")):
        return NAMED_INPUT
    return 1.0


def rank(project: Path, task: str, files: list[str] | None = None,
         texts: dict[str, str] | None = None, route: dict | None = None) -> list[Place]:
    """Files ranked by the task's literals (best first); [] when nothing matches. `route`: the
    router's record for the task (vega.routing.Route.record()), which tells a named file to
    change from one named as an input or a script to run."""
    project = project.resolve()
    if files is None:
        files = _candidates(project)
    if texts is None:
        texts = {}
        for rel in files:
            text = _read_text(project, rel)
            if text is not None and len(text) <= MAX_FILE_BYTES:
                texts[rel] = text
    paths = [rel for rel in files if rel in texts]
    if not paths:
        return []
    n = len(paths)
    places: dict[str, Place] = {}
    found = literals(task)[:MAX_LITERALS * 2]
    used = 0
    while found:
        literal = found.pop(0)
        if literal.kind == "path":
            target = literal.text.lstrip("./")
            hits = [p for p in paths if p == target or p.endswith("/" + target)]
            if not hits:
                continue
            for path in hits:
                place = places.setdefault(path, Place(path, 0.0))
                place.score += WEIGHTS["path"] * _named_role(path, route) / len(hits)
                place.reasons.append("named in the task")
            continue
        if used >= MAX_LITERALS and literal.kind != "word":
            continue
        pattern = _matcher(literal)
        hits = [p for p in paths if pattern.search(texts[p])]
        if not hits:
            if literal.kind == "quoted":  # a message built from parts: look for its pieces
                found[:0] = [Literal(part, "fragment") for part in _fragments(literal.text)]
            continue
        if literal.kind == "word" and len(hits) > n / 4:
            continue  # common words say nothing
        used += literal.kind != "word"
        rarity = math.log(1 + n / len(hits))
        for path in hits:
            text = texts[path]
            weight = WEIGHTS[literal.kind] * rarity
            reason = ""
            if literal.kind in {"ident", "quoted"} and re.fullmatch(r"[\w.]+", literal.text):
                if _defines(literal.text, text):
                    weight *= DEFINES
                    reason = f"defines `{literal.text.rsplit('.', 1)[-1]}`"
                elif _registers(literal.text, text):
                    weight *= REGISTERS
            if _is_test(path):
                weight *= TEST_FILE
            elif path.lower().endswith(_DOC_SUFFIXES) or "." not in path.rsplit("/", 1)[-1]:
                weight *= DOC_FILE
            place = places.setdefault(path, Place(path, 0.0))
            place.score += weight
            if literal.kind != "word":
                quoted = literal.kind in {"quoted", "fragment"}
                label = reason or (f'"{literal.text}"' if quoted else f"`{literal.text}`")
                place.reasons.append(label)
                matched = 0
                for number, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line):
                        place.lines[number] = place.lines.get(number, 0.0) + weight
                        matched += 1
                        if matched >= 40:
                            break
    ranked = sorted(places.values(), key=lambda p: (-p.score, p.path))
    return [p for p in ranked if p.score > 0]


def _enclosing(path: str, text: str, lines: dict[int, float]) -> list[str]:
    """The innermost function/class around the matching lines, the ones holding the strongest
    evidence first (at most MAX_SYMBOLS): 'def name (lines a-b): matches at x, y'."""
    symbols = [s for s in extract_symbols(path, text) if s.kind != "constant"]
    ordered = sorted(symbols, key=lambda s: s.line)
    total = len(text.splitlines())
    spans = []
    for i, sym in enumerate(ordered):
        end = sym.end_line
        if end is None:  # regex languages: up to the next symbol at the same or lower depth
            later = [o.line for o in ordered[i + 1:] if o.depth <= sym.depth]
            end = (later[0] - 1) if later else total
        spans.append((sym, end))
    python = path.endswith((".py", ".pyi"))
    # label -> (first line, matching lines, evidence)
    groups: dict[str, tuple[int, list[int], float]] = {}
    for number in sorted(lines):
        around = [(sym, end) for sym, end in spans if sym.line <= number <= end]
        if around:
            sym, end = max(around, key=lambda item: item[0].line)
            kind = "class" if sym.kind == "class" else "def" if python else sym.kind
            label, start = f"{kind} {sym.name} (lines {sym.line}-{end})", sym.line
        else:
            label, start = "top level", 0
        first, numbers, score = groups.get(label, (start, [], 0.0))
        numbers.append(number)
        groups[label] = (first, numbers, score + lines[number])
    if len(groups) > 1:  # module docstrings and imports: only worth a line when nothing else
        groups.pop("top level", None)
    ranked = sorted(groups.items(), key=lambda kv: (-kv[1][2], kv[1][0]))[:MAX_SYMBOLS]
    out = []
    for label, (_, numbers, _) in sorted(ranked, key=lambda kv: kv[1][0]):
        shown = ", ".join(str(n) for n in numbers[:4]) + (" ..." if len(numbers) > 4 else "")
        out.append(f"{label}: {'line' if len(numbers) == 1 else 'lines'} {shown}")
    return out


TIE_RATIO = 0.85  # localize_ties: a file scoring >= this x the top score is about as likely
TIE_NOTE = ("Files {0} score about the same: read {1} before deciding which to edit "
            "(the top rank alone is not the answer).")


@dataclass
class Located:
    """The block and the ranked files it shows (best first); `tied`: the shown files scoring
    within TIE_RATIO of the top one, the top one included (empty when the top stands alone)."""
    text: str = ""
    paths: list[str] = field(default_factory=list)
    tied: list[str] = field(default_factory=list)


def near_ties(places: list[Place], ratio: float = TIE_RATIO) -> list[str]:
    """localize_ties: the paths scoring >= ratio x the top score, the top included; [] when no
    other file comes that close (selfbuild 2026-10-21: the right file was #2 at 25.33 vs 27.04
    and 10.96 vs 12.09, and the model edited #1)."""
    if len(places) < 2 or places[0].score <= 0:
        return []
    tied = [p.path for p in places if p.score >= ratio * places[0].score]
    return tied if len(tied) > 1 else []


def _numbers(count: int) -> tuple[str, str]:
    words = [str(i) for i in range(1, count + 1)]
    listed = " and ".join(words) if count == 2 else ", ".join(words[:-1]) + " and " + words[-1]
    return listed, "both" if count == 2 else f"all {count}"


def locate(project: Path, task: str, route: dict | None = None, min_files: int = MIN_FILES,
           ties: bool = False) -> Located:
    """The 'Likely places' block for the first prompt and the files it ranks, or an empty
    Located (small project, nothing matched). `ties` (localize_ties): the block says which top
    files score about the same."""
    project = project.resolve()
    files = _candidates(project)
    if len(files) < min_files:
        return Located()
    texts: dict[str, str] = {}
    for rel in files:
        text = _read_text(project, rel)
        if text is not None and len(text) <= MAX_FILE_BYTES:
            texts[rel] = text
    places = rank(project, task, files, texts, route)
    # Only evidence beyond plain words is worth a block: a word-only match is a guess.
    places = [p for p in places if p.reasons][:TOP_FILES]
    if not places:
        return Located()
    tied = near_ties(places) if ties else []
    # the longest the note can be once the block is trimmed (fewer tied files: a shorter note)
    note = TIE_NOTE.format(*_numbers(len(tied))) if tied else ""
    out = [HEADER]
    for i, place in enumerate(places, 1):
        reasons = list(dict.fromkeys(place.reasons))
        why = ", ".join(r if len(r) <= 60 else r[:57] + '..."' for r in reasons[:3])
        mark = " [about as likely as 1]" if tied and i > 1 and place.path in tied else ""
        out.append(f"{i}. {place.path} ({why}){mark}")
        for line in _enclosing(place.path, texts[place.path], place.lines):
            out.append(f"   - {line}")
    budget = MAX_CHARS - (len(note) + 1 if note else 0)
    text = "\n".join(out)
    while len(text) > budget and len(out) > 2:
        out.pop()
        text = "\n".join(out)
    shown = [p.path for p in places[:sum(line[:1].isdigit() for line in out[1:])]]
    tied = [path for path in tied if path in shown]
    if len(tied) > 1:  # trimming the block never leaves the note about a file it dropped
        text += "\n" + TIE_NOTE.format(*_numbers(len(tied)))
    else:
        tied = []
    return Located(text, shown, tied)


def block(project: Path, task: str, route: dict | None = None,
          min_files: int = MIN_FILES, ties: bool = False) -> str:
    """The 'Likely places' block for the first prompt, or "" (small project, nothing matched)."""
    return locate(project, task, route, min_files, ties).text


def is_source(path: str) -> bool:
    """localize_ties: a code file the issue's fix would go in (not a test, not prose)."""
    lower = path.lower()
    return bool(path) and not _is_test(path) and not lower.endswith(_DOC_SUFFIXES) \
        and "." in lower.rsplit("/", 1)[-1]


def tie_unread(target: str | None, tied: list[str], read: set[str]) -> list[str]:
    """localize_ties: the near-tied top files other than `target` not read this turn, when
    `target` is a source file ([] otherwise). `read`: normalised paths read_file returned."""
    if not target or len(tied) < 2 or not is_source(target):
        return []
    norm = os.path.normpath
    return [path for path in tied if norm(path) != norm(target) and norm(path) not in read]


def tie_hold_note(target: str, unread: list[str], tied: list[str]) -> str:
    """localize_ties: the send-back for the first source write while a near-tied file is unread."""
    names = ", ".join(f"`{path}`" for path in unread)
    ranked = ", ".join(f"{i}. `{path}`" for i, path in enumerate(tied, 1))
    return ("(Vega harness) Not written yet: the Likely places ranked these files about equally "
            f"likely ({ranked}), and you have not read {names} this turn. read_file "
            + ("it" if len(unread) == 1 else "each one") + " first and decide which file really "
            f"holds the cause; the top rank alone is not the answer. Then make the change there "
            f"(repeat this edit to `{target}` unchanged if it is the right place).")


def no_edit_places(paths: list[str]) -> str:
    """localize_ties: the line the no-edit finish gate adds when the turn had Likely places."""
    return ("The Likely places for this task were " + ", ".join(f"`{p}`" for p in paths)
            + ": read them and edit the one that holds the cause.")
