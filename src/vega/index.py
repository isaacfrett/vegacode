"""Lightweight code index: symbol extraction, a compact repo map and BM25 keyword search.

Stdlib only. Python files are parsed with `ast`; other languages use simple line regexes. Nothing
is cached to disk — the index is rebuilt in memory on each call, which is fast for typical repos.
"""

from __future__ import annotations

import ast
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from vega.tools import ignored_parts

MAX_FILE_BYTES = 1_000_000
MAX_FILES = 5000
WINDOW_LINES = 40
K1 = 1.5
B = 0.75
NAME_BOOST = 3  # extra term frequency for tokens of a chunk's symbol name
PATH_BOOST = 2  # extra term frequency for tokens of the chunk's file path


@dataclass
class Symbol:
    name: str
    kind: str  # class, function, method, constant, struct, interface, ...
    line: int
    end_line: int | None = None
    signature: str = ""
    depth: int = 0  # 0 = top level, 1 = inside a class


@dataclass
class Hit:
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str


@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    name: str = ""


@dataclass
class FileEntry:
    path: str
    lines: list[str]
    symbols: list[Symbol] = field(default_factory=list)


# --------------------------------------------------------------------------- tokenizing

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercased identifier tokens; compound identifiers also yield their snake/camel parts."""
    tokens: list[str] = []
    for ident in _IDENT.findall(text):
        whole = ident.lower().strip("_")
        if not whole:
            continue
        tokens.append(whole)
        parts = [p.lower() for piece in ident.split("_") if piece for p in _CAMEL.findall(piece)]
        if len(parts) > 1:
            tokens.extend(p for p in parts if len(p) > 1)
    return tokens


# --------------------------------------------------------------------------- file listing

def _list_files(project: Path) -> list[str]:
    result = None
    try:
        result = subprocess.run(
            ["git", "-C", str(project), "ls-files", "--cached", "--others", "--exclude-standard"],
            text=True, capture_output=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    if result is not None and result.returncode == 0:
        names = sorted(set(result.stdout.splitlines()))
    else:
        names = sorted(
            p.relative_to(project).as_posix() for p in project.rglob("*")
            if p.is_file()
            and not ignored_parts(p.relative_to(project).parts, project)
        )
    return [n for n in names if not ignored_parts(n.split("/"), project)]


def _read_text(project: Path, rel: str) -> str | None:
    path = project / rel
    try:
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            return None
        path.resolve().relative_to(project)  # symlinks must not lead outside the project
        data = path.read_bytes()
    except (OSError, ValueError):
        return None
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


# --------------------------------------------------------------------------- symbol extraction

_UPPER = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _python_symbols(source: str) -> list[Symbol]:
    tree = ast.parse(source)
    symbols: list[Symbol] = []

    def visit(body: list[ast.stmt], depth: int) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
                symbols.append(Symbol(node.name, "class", node.lineno, node.end_lineno, sig, depth))
                visit(node.body, depth + 1)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                sig = f"{prefix} {node.name}({ast.unparse(node.args)})"
                if node.returns is not None:
                    sig += f" -> {ast.unparse(node.returns)}"
                kind = "method" if depth else "function"
                sig = sig if len(sig) <= 100 else sig[:97] + "..."
                symbols.append(Symbol(node.name, kind, node.lineno, node.end_lineno, sig, depth))
            elif depth == 0 and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and _UPPER.match(target.id):
                        symbols.append(Symbol(
                            target.id, "constant", node.lineno, node.end_lineno, target.id, 0,
                        ))

    visit(tree.body, 0)
    return symbols


_N = r"(?P<name>[A-Za-z_$][\w$]*)"
_JS_KEYWORDS = {"if", "for", "while", "switch", "catch", "function", "return", "with", "else"}
_C_KEYWORDS = _JS_KEYWORDS | {"sizeof", "do", "case", "defined"}

_JS = [
    ("class", re.compile(rf"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+{_N}")),
    ("function", re.compile(
        rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*{_N}\s*[<(]")),
    ("function", re.compile(
        rf"^\s*(?:export\s+)?(?:const|let|var)\s+{_N}\s*(?::[^=]+)?=\s*(?:async\s+)?"
        r"(?:function\b|(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>)")),
    ("interface", re.compile(rf"^\s*(?:export\s+)?(?:declare\s+)?interface\s+{_N}")),
    ("type", re.compile(rf"^\s*(?:export\s+)?(?:declare\s+)?type\s+{_N}\s*(?:<[^>]*>)?\s*=")),
    ("enum", re.compile(rf"^\s*(?:export\s+)?(?:const\s+)?enum\s+{_N}")),
    ("method", re.compile(
        r"^\s+(?:(?:public|private|protected|static|async|readonly|override|get|set)\s+)*"
        rf"\*?{_N}\s*(?:<[^>]*>)?\([^)]*\)\s*(?::\s*[^{{;]+)?\{{\s*$")),
]
_PY = [
    ("class", re.compile(rf"^\s*class\s+{_N}")),
    ("function", re.compile(rf"^\s*(?:async\s+)?def\s+{_N}")),
]
_GO = [
    ("function", re.compile(rf"^func\s+(?:\([^)]*\)\s*)?{_N}\s*[\[(]")),
    ("type", re.compile(rf"^type\s+{_N}\s+(?:struct|interface)\b")),
]
_RUST_VIS = r"^\s*(?:pub(?:\([^)]*\))?\s+)?"
_RUST = [
    ("function", re.compile(
        _RUST_VIS + r"(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?"
        rf"(?:extern\s+\"[^\"]*\"\s+)?fn\s+{_N}")),
    ("struct", re.compile(_RUST_VIS + rf"struct\s+{_N}")),
    ("enum", re.compile(_RUST_VIS + rf"enum\s+{_N}")),
    ("trait", re.compile(_RUST_VIS + rf"(?:unsafe\s+)?trait\s+{_N}")),
    ("impl", re.compile(rf"^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>, ]+\s+for\s+)?{_N}")),
]
_SWIFT_MODS = (r"^\s*(?:@\w+(?:\([^)]*\))?\s+)*(?:(?:public|private|fileprivate|internal|open|"
               r"final|static|class|override|mutating|nonmutating|required|convenience|"
               r"indirect)\s+)*")
_SWIFT = [
    ("function", re.compile(_SWIFT_MODS + rf"func\s+{_N}")),
    ("class", re.compile(_SWIFT_MODS + rf"(?:class|struct|enum|protocol|actor|extension)\s+{_N}")),
]
_GD = [
    ("class", re.compile(rf"^class_name\s+{_N}")),
    ("class", re.compile(rf"^\s*class\s+{_N}")),
    ("function", re.compile(rf"^\s*(?:static\s+)?func\s+{_N}")),
    ("signal", re.compile(rf"^\s*signal\s+{_N}")),
    ("constant", re.compile(r"^const\s+(?P<name>[A-Z][A-Z0-9_]*)")),
]
_JAVA_MODS = (r"(?:(?:public|private|protected|static|final|abstract|sealed|synchronized|"
              r"native|default)\s+)")
_JAVA = [
    ("class", re.compile(rf"^\s*(?:@\w+\s+)*{_JAVA_MODS}*(?:class|interface|enum|record)\s+{_N}")),
    ("method", re.compile(rf"^\s*{_JAVA_MODS}+(?:<[^>]*>\s*)?[\w<>\[\],.? ]+\s+{_N}\s*\(")),
]
_C = [
    ("struct", re.compile(
        r"^\s*(?:typedef\s+)?(?:struct|class|union|enum)\s+(?P<name>[A-Za-z_]\w*)\s*"
        r"(?::[^{;]*)?\{?\s*$")),
    ("function", re.compile(
        r"^(?:[A-Za-z_][\w:<>]*[\s*&]+)+(?P<name>[A-Za-z_~][\w:~]*)\s*\([^;]*$")),
]
REGEX_LANGUAGES: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    **{ext: _JS for ext in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx")},
    ".py": _PY,
    ".go": _GO,
    ".rs": _RUST,
    ".swift": _SWIFT,
    ".gd": _GD,
    ".java": _JAVA,
    **{ext: _C for ext in (".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".m", ".mm")},
}


def _regex_symbols(lines: list[str], patterns: list[tuple[str, re.Pattern[str]]]) -> list[Symbol]:
    keywords = _C_KEYWORDS if patterns is _C else _JS_KEYWORDS
    symbols: list[Symbol] = []
    for number, line in enumerate(lines, 1):
        for kind, pattern in patterns:
            match = pattern.match(line)
            if not match:
                continue
            name = match.group("name")
            if name in keywords:
                continue
            depth = 1 if kind == "method" or (line[:1].isspace() and kind == "function") else 0
            signature = line.strip().rstrip("{").strip()
            if len(signature) > 100:
                signature = signature[:97] + "..."
            symbols.append(Symbol(name, kind, number, None, signature, depth))
            break
    return symbols


def extract_symbols(path: str, source: str) -> list[Symbol]:
    """Symbols for one file. Python uses `ast`, falling back to regexes on syntax errors."""
    suffix = Path(path).suffix.lower()
    if suffix in {".py", ".pyi"}:
        try:
            return _python_symbols(source)
        except (SyntaxError, ValueError, RecursionError):
            pass
    patterns = REGEX_LANGUAGES.get(".py" if suffix == ".pyi" else suffix)
    return _regex_symbols(source.splitlines(), patterns) if patterns else []


# --------------------------------------------------------------------------- chunking

def _windows(path: str, start: int, end: int, lines: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for lo in range(start, end + 1, WINDOW_LINES):
        hi = min(lo + WINDOW_LINES - 1, end)
        if any(lines[i - 1].strip() for i in range(lo, hi + 1)):
            chunks.append(Chunk(path, lo, hi))
    return chunks


def _chunks(entry: FileEntry) -> list[Chunk]:
    total = len(entry.lines)
    if not total:
        return []
    ranged = [s for s in entry.symbols if s.end_line is not None and s.kind != "constant"]
    if not ranged:
        return _windows(entry.path, 1, total, entry.lines)
    chunks: list[Chunk] = []
    covered = [False] * (total + 2)
    for sym in ranged:
        if sym.kind == "class":
            # Class header up to the first method; methods get their own chunks.
            inner = [s.line for s in ranged if sym.line < s.line <= sym.end_line]
            end = min(inner) - 1 if inner else sym.end_line
        else:
            end = sym.end_line
        if any(covered[sym.line:end + 1]):  # nested def inside a function: already covered
            continue
        for i in range(sym.line, end + 1):
            covered[i] = True
        chunks.append(Chunk(entry.path, sym.line, end, sym.name))
    run_start = None
    for i in range(1, total + 2):
        free = i <= total and not covered[i]
        if free and run_start is None:
            run_start = i
        elif not free and run_start is not None:
            chunks.extend(_windows(entry.path, run_start, i - 1, entry.lines))
            run_start = None
    return sorted(chunks, key=lambda c: c.start_line)


# --------------------------------------------------------------------------- the index

class CodeIndex:
    def __init__(self, project: Path, files: list[FileEntry]) -> None:
        self.project = project
        self.files = files
        self._by_path = {f.path: f for f in files}
        self.chunks: list[Chunk] = [c for f in files for c in _chunks(f)]
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        path_tokens: dict[str, list[str]] = {}
        for idx, chunk in enumerate(self.chunks):
            lines = self._by_path[chunk.path].lines[chunk.start_line - 1:chunk.end_line]
            counts = Counter(tokenize("\n".join(lines)))
            for token in tokenize(chunk.name):
                counts[token] += NAME_BOOST
            if chunk.path not in path_tokens:
                path_tokens[chunk.path] = list(dict.fromkeys(tokenize(chunk.path)))
            for token in path_tokens[chunk.path]:
                counts[token] += PATH_BOOST
            self._lengths.append(sum(counts.values()))
            for token, tf in counts.items():
                self._postings[token].append((idx, tf))
        self._avgdl = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

    # ------------------------------------------------------------------ repo map

    def repo_map(self, max_chars: int = 6000, prefix: str = "") -> str:
        """File paths with indented symbols; trims symbols (never whole files) to fit the budget."""
        prefix = prefix.strip()
        while prefix.startswith("./"):
            prefix = prefix[2:]
        prefix = "" if prefix == "." else prefix
        files = [f for f in self.files if f.path.startswith(prefix)]
        if not files:
            return f"No indexed files under {prefix!r}." if prefix else "(no files)"
        # Top-level symbols first, then nested ones, so trimming keeps the outline.
        ordered = {f.path: sorted(f.symbols, key=lambda s: (s.depth, s.line)) for f in files}
        most = max((len(v) for v in ordered.values()), default=0)

        def render(cap: int) -> str:
            out: list[str] = []
            for f in files:
                keep = sorted(ordered[f.path][:cap], key=lambda s: s.line)
                out.append(f.path)
                for sym in keep:
                    out.append(f"{'  ' * (sym.depth + 1)}{sym.line}: {sym.signature or sym.name}")
                hidden = len(f.symbols) - len(keep)
                if hidden > 0:
                    out.append(f"  ... +{hidden} more symbols")
            return "\n".join(out)

        full = render(most)
        if len(full) <= max_chars:
            return full
        lo, hi = 0, most  # largest per-file cap that fits
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(render(mid)) <= max_chars:
                lo = mid
            else:
                hi = mid - 1
        text = render(lo)
        if len(text) <= max_chars:
            return text
        # Even bare paths overflow: list as many paths as fit and count the rest.
        out: list[str] = []
        used = 0
        for i, f in enumerate(files):
            line = f"{f.path} ({len(f.symbols)} symbols)" if f.symbols else f.path
            if used + len(line) + 40 > max_chars:
                out.append(f"... and {len(files) - i} more files")
                break
            out.append(line)
            used += len(line) + 1
        return "\n".join(out)

    # ------------------------------------------------------------------ search

    def search(self, query: str, limit: int = 20) -> list[Hit]:
        """BM25 over chunks (k1=1.5, b=0.75), with symbol-name and path tokens boosted."""
        terms = list(dict.fromkeys(tokenize(query)))
        n = len(self.chunks)
        if not terms or not n:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for idx, tf in postings:
                norm = K1 * (1 - B + B * self._lengths[idx] / self._avgdl)
                scores[idx] += idf * tf * (K1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:max(limit, 0)]
        return [self._hit(self.chunks[idx], score, terms) for idx, score in ranked]

    def _hit(self, chunk: Chunk, score: float, terms: list[str]) -> Hit:
        lines = self._by_path[chunk.path].lines
        span = range(chunk.start_line, chunk.end_line + 1)
        wanted = set(terms)
        picked = [i for i in span if wanted & set(tokenize(lines[i - 1]))][:3]
        if chunk.start_line not in picked and lines[chunk.start_line - 1].strip():
            picked = [chunk.start_line] + picked[:2]
        if not picked:
            picked = [i for i in span if lines[i - 1].strip()][:3]
        snippet = "\n".join(f"{i}: {lines[i - 1].strip()[:160]}" for i in picked)
        return Hit(chunk.path, chunk.start_line, chunk.end_line, round(score, 2), snippet)


def build_index(project: Path) -> CodeIndex:
    project = project.resolve()
    entries: list[FileEntry] = []
    for rel in _list_files(project)[:MAX_FILES]:
        text = _read_text(project, rel)
        if text is None:
            continue
        entries.append(FileEntry(rel, text.splitlines(), extract_symbols(rel, text)))
    return CodeIndex(project, entries)


def format_hits(hits: list[Hit]) -> str:
    blocks = []
    for hit in hits:
        body = "\n".join(f"  {line}" for line in hit.snippet.splitlines())
        blocks.append(f"{hit.path}:{hit.start_line}-{hit.end_line} ({hit.score:.2f})\n{body}")
    return "\n".join(blocks)
