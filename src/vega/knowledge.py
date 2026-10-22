"""Vega's knowledge library: material it has been told to learn, searchable while it works.

`vega learn add <url | file | folder>` (or /learn in chat) takes in web pages, PDFs (a textbook),
Markdown / text / reStructuredText and HTML files, and folders of them. Text is extracted, split
into sections of about CHUNK_CHARS (keeping headings and PDF page numbers as locations) and stored
under ~/.vega/knowledge/ (VEGA_KNOWLEDGE_DIR overrides; evals point it at a per-case folder). The
agent's search_knowledge tool ranks sections with BM25 and returns them with their source and
location, so answers can cite where a fact came from.

Stdlib only, except PDFs, which need the optional `pypdf` package (`pip install 'vega[learn]'`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, ClassVar

from . import server
from .index import tokenize

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
MAX_FETCH_BYTES = 30_000_000
MAX_FOLDER_FILES = 2000
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".text"}
HTML_SUFFIXES = {".html", ".htm"}
SUPPORTED = TEXT_SUFFIXES | HTML_SUFFIXES | {".pdf"}
USER_AGENT = "Vega/0.1 (local coding agent; learning from a page the user asked for)"
K1, B = 1.5, 0.75
TITLE_BOOST = 2


class LearnError(Exception):
    """A source could not be learned (unreachable, unsupported, empty)."""


@dataclass
class Source:
    id: str
    title: str
    origin: str  # URL or absolute path
    kind: str  # web, pdf, text, html
    added: str  # when it was learned (for a web page: the fetch date)
    chunks: int
    chars: int
    # web research (vega/web.py): the page's own publication date and its trust tier
    # (official: an official-docs domain; low: any other web page). Empty for other sources.
    published: str = ""
    trust: str = ""


@dataclass
class Chunk:
    source: str  # Source.id
    title: str  # source title
    location: str  # "page 12", "Section: Typing > Generics", ...
    text: str


def library_dir() -> Path:
    override = os.environ.get("VEGA_KNOWLEDGE_DIR")
    return Path(override) if override else server.STATE_DIR / "knowledge"


# -- extraction ------------------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Readable text from HTML: skips scripts, navigation and forms; headings become '#' lines so
    sections keep their titles; pre blocks keep their layout."""

    SKIP: ClassVar[set[str]] = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg",
            "button", "template"}
    BLOCK: ClassVar[set[str]] = {"p", "div", "section", "article", "main", "li", "tr", "br", "table", "ul", "ol",
             "blockquote", "dd", "dt", "figcaption", "h1", "h2", "h3", "h4", "h5", "h6", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipping = 0
        self.pre = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self.skipping += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "pre":
            self.pre += 1
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag in {"h1", "h2", "h3", "h4"} and not self.skipping:
            self.parts.append("#" * int(tag[1]) + " ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self.skipping:
            self.skipping -= 1
        elif tag == "title":
            self._in_title = False
        elif tag == "pre" and self.pre:
            self.pre -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self.skipping:
            return
        self.parts.append(data if self.pre else re.sub(r"\s+", " ", data))

    def text(self) -> str:
        joined = "".join(self.parts)
        lines = [line.rstrip() for line in joined.split("\n")]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_text(html: str) -> tuple[str, str]:
    """(title, text) of an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    # Sphinx and many doc sites add a pilcrow link after each heading
    return re.sub(r"\s+", " ", parser.title).strip(), parser.text().replace("\u00b6", "")


def _pdf_pages(data: bytes) -> list[str]:
    try:
        from io import BytesIO

        from pypdf import PdfReader
    except ImportError as exc:
        raise LearnError("Reading PDFs needs the pypdf package: pip install 'vega[learn]' "
                         "(or pip install pypdf).") from exc
    reader = PdfReader(BytesIO(data))
    return [(page.extract_text() or "") for page in reader.pages]


# -- chunking --------------------------------------------------------------------------------------

def _windows(text: str) -> list[str]:
    """Split text into ~CHUNK_CHARS pieces at paragraph or sentence boundaries, overlapping."""
    text = text.strip()
    if len(text) <= CHUNK_CHARS:
        return [text] if text else []
    pieces, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > CHUNK_CHARS // 2:
                end = start + cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return [piece for piece in pieces if piece]


_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")


def sections(text: str) -> list[tuple[str, str]]:
    """(heading path, body) pieces of Markdown-style text; '' when there are no headings."""
    out: list[tuple[str, str]] = []
    trail: list[str] = []
    body: list[str] = []

    def flush() -> None:
        if "".join(body).strip():
            out.append((" > ".join(trail), "\n".join(body)))
        body.clear()

    in_code = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            trail[:] = trail[: level - 1] + [match.group(2).strip()]
            body.append(line)
            continue
        body.append(line)
    flush()
    return out


def chunk_text(source: Source, text: str) -> list[Chunk]:
    chunks = []
    for heading, body in sections(text):
        location = f"section {heading}" if heading else "text"
        for piece in _windows(body):
            chunks.append(Chunk(source.id, source.title, location, piece))
    return chunks


def chunk_pages(source: Source, pages: list[str]) -> list[Chunk]:
    chunks = []
    for number, page in enumerate(pages, 1):
        for piece in _windows(page):
            chunks.append(Chunk(source.id, source.title, f"page {number}", piece))
    return chunks


# -- library ---------------------------------------------------------------------------------------

def _source_id(origin: str) -> str:
    return hashlib.sha1(origin.encode("utf-8")).hexdigest()[:10]


def sources() -> list[Source]:
    path = library_dir() / "sources.json"
    try:
        return [Source(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
    except (OSError, ValueError, TypeError):
        return []


def _save(all_sources: list[Source], all_chunks: list[Chunk]) -> None:
    root = library_dir()
    root.mkdir(parents=True, exist_ok=True)
    (root / "sources.json").write_text(
        json.dumps([asdict(s) for s in all_sources], indent=2) + "\n", encoding="utf-8")
    with (root / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    _INDEX.clear()


def chunks() -> list[Chunk]:
    path = library_dir() / "chunks.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [Chunk(**json.loads(line)) for line in lines if line.strip()]


def _store(source: Source, new_chunks: list[Chunk]) -> Source:
    if not new_chunks:
        raise LearnError(f"No readable text found in {source.origin}.")
    source.chunks = len(new_chunks)
    source.chars = sum(len(chunk.text) for chunk in new_chunks)
    kept_sources = [s for s in sources() if s.id != source.id]  # re-learning replaces
    kept_chunks = [c for c in chunks() if c.source != source.id]
    _save([*kept_sources, source], [*kept_chunks, *new_chunks])
    return source


def forget(source_id: str) -> Source | None:
    found = next((s for s in sources() if s.id == source_id or s.id.startswith(source_id)), None)
    if found is None:
        return None
    _save([s for s in sources() if s.id != found.id], [c for c in chunks() if c.source != found.id])
    return found


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


def learn(target: str, title: str | None = None) -> list[Source]:
    """Learn a URL, a file or every supported file in a folder. Returns the sources added."""
    if re.match(r"https?://", target):
        return [learn_url(target, title)]
    path = Path(target).expanduser().resolve()
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED
                       and not any(part.startswith(".") for part in p.relative_to(path).parts))
        if not files:
            raise LearnError(f"No supported files ({', '.join(sorted(SUPPORTED))}) in {path}.")
        return [learn_file(p) for p in files[:MAX_FOLDER_FILES]]
    if path.is_file():
        return [learn_file(path, title)]
    raise LearnError(f"Not a URL, file or folder: {target}")


def learn_file(path: Path, title: str | None = None) -> Source:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise LearnError(f"Unsupported file type {suffix or '(none)'}; Vega can learn "
                         f"{', '.join(sorted(SUPPORTED))}.")
    origin = str(path)
    if suffix == ".pdf":
        pages = _pdf_pages(path.read_bytes())
        source = Source(_source_id(origin), title or path.stem, origin, "pdf", _now(), 0, 0)
        return _store(source, chunk_pages(source, pages))
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in HTML_SUFFIXES:
        page_title, text = html_to_text(raw)
        source = Source(_source_id(origin), title or page_title or path.stem, origin, "html",
                        _now(), 0, 0)
    else:
        text = raw
        first = next((m.group(2) for line in raw.splitlines()[:20]
                      if (m := _HEADING.match(line))), None)
        source = Source(_source_id(origin), title or first or path.stem, origin, "text", _now(),
                        0, 0)
    return _store(source, chunk_text(source, text))


def fetch(url: str) -> tuple[bytes, str]:
    """(body, content type) of a URL, capped at MAX_FETCH_BYTES."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # the user's URL
            body = response.read(MAX_FETCH_BYTES + 1)
            kind = response.headers.get("Content-Type", "")
    except OSError as exc:
        raise LearnError(f"Could not fetch {url}: {exc}") from exc
    if len(body) > MAX_FETCH_BYTES:
        raise LearnError(f"{url} is larger than {MAX_FETCH_BYTES // 1_000_000} MB.")
    return body, kind


def learn_url(url: str, title: str | None = None) -> Source:
    body, kind = fetch(url)
    path = urllib.parse.urlparse(url).path.lower()
    if "pdf" in kind or path.endswith(".pdf"):
        name = Path(path).stem or url
        source = Source(_source_id(url), title or name, url, "pdf", _now(), 0, 0)
        return _store(source, chunk_pages(source, _pdf_pages(body)))
    charset = re.search(r"charset=([\w-]+)", kind)
    text = body.decode(charset.group(1) if charset else "utf-8", errors="replace")
    if "html" in kind or text.lstrip()[:200].lower().startswith(("<!doctype html", "<html")):
        page_title, text = html_to_text(text)
    else:
        page_title = ""
    source = Source(_source_id(url), title or page_title or url, url, "web", _now(), 0, 0)
    return _store(source, chunk_text(source, text))


def store_web_page(url: str, title: str, published: str, text: str, trust: str) -> Source:
    """Cache a page fetched by web research, with its provenance: URL, title, publication date,
    fetch date (`added`) and trust tier. Fetching the same URL again replaces the cached copy."""
    source = Source(_source_id(url), title or url, url, "web", _now(), 0, 0,
                    published=published, trust=trust)
    return _store(source, chunk_text(source, text))


# -- search ----------------------------------------------------------------------------------------

_INDEX: dict[str, Any] = {}


def _index() -> dict[str, Any]:
    path = library_dir() / "chunks.jsonl"
    try:
        stamp = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return {"chunks": [], "terms": [], "df": Counter(), "avg": 1.0}
    if _INDEX.get("stamp") != stamp:
        all_chunks = chunks()
        terms = [Counter(tokenize(c.text) + tokenize(f"{c.title} {c.location}") * TITLE_BOOST)
                 for c in all_chunks]
        df: Counter[str] = Counter()
        for counts in terms:
            df.update(counts.keys())
        avg = sum(sum(c.values()) for c in terms) / max(1, len(terms))
        _INDEX.update(stamp=stamp, chunks=all_chunks, terms=terms, df=df, avg=avg or 1.0)
    return _INDEX


def search(query: str, limit: int = 5) -> list[tuple[float, Chunk]]:
    index = _index()
    total = len(index["chunks"])
    words = set(tokenize(query))
    if not total or not words:
        return []
    scored = []
    for chunk, counts in zip(index["chunks"], index["terms"], strict=True):
        length = sum(counts.values())
        score = 0.0
        for word in words:
            tf = counts.get(word, 0)
            if not tf:
                continue
            idf = math.log(1 + (total - index["df"][word] + 0.5) / (index["df"][word] + 0.5))
            score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / index["avg"]))
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda item: -item[0])
    return scored[:limit]


RESULT_CHARS = 900


def search_text(query: str, limit: int = 5) -> str:
    """The search_knowledge tool's result."""
    if not query.strip():
        return "Give words describing what to look up."
    hits = search(query, limit)
    if not hits:
        return (f"Nothing in the knowledge library matched {query!r}. Try other words; the "
                f"library holds: {summary()}.")
    blocks = []
    by_id = {source.id: source for source in sources()}
    for score, chunk in hits:
        text = chunk.text if len(chunk.text) <= RESULT_CHARS else chunk.text[:RESULT_CHARS] + " …"
        source = by_id.get(chunk.source)
        provenance = ""
        if source is not None and source.trust:  # a cached web page: where, when, how trusted
            provenance = (f" {source.origin} (published {source.published or 'unknown'}, "
                          f"fetched {source.added[:10]}, {source.trust} trust)")
        blocks.append(f"[{chunk.title} — {chunk.location}]{provenance} (score {score:.1f})\n{text}")
    return "\n\n".join(blocks) + "\n\nCite the source and location you rely on."


def summary(limit: int = 12) -> str:
    """'Title (kind, N sections); ...' for the system prompt and empty searches."""
    found = sources()
    if not found:
        return "nothing yet"
    shown = "; ".join(f"{s.title} ({s.kind}, {s.chunks} sections)" for s in found[:limit])
    more = f"; and {len(found) - limit} more" if len(found) > limit else ""
    return shown + more
