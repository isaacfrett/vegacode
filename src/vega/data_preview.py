"""Data files as inputs: which paths are data, and a structured preview of a large one.

Why (2026-10-15, Granite): read_file on a whole CSV took the prompt from ~2.9k to ~6.3k tokens
of an 8,192 window, so the next step compacted the history and the model re-read the same files
in a loop. A preview gives the shape of
the data (header, column types, first rows, line count) in a bounded size and says how to read
more, so the model decides what else to look at.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
from pathlib import Path, PurePosixPath
from typing import Any

DATA_EXTENSIONS = frozenset({
    ".csv", ".tsv", ".json", ".jsonl", ".parquet", ".txt", ".xml", ".yaml", ".yml", ".sql",
    ".db", ".sqlite",
})
DATA_DIRS = frozenset({"data", "fixtures", "samples"})
CODE_EXTENSIONS = frozenset({".py", ".pyi", ".ipynb"})
PREVIEW_MIN_BYTES = 4096  # smaller files (and those of PREVIEW_MIN_LINES or fewer) are read whole
PREVIEW_MIN_LINES = 60
PREVIEW_ROWS = 20
TYPE_SAMPLE = 200  # rows sampled to infer column types
CELL_CHARS = 120
LINE_CHARS = 300
JSON_ITEMS = 3
JSON_ITEM_CHARS = 600
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")
_BOOLS = {"true", "false"}


def is_data_path(path: str | Path) -> bool:
    """A data/input file: by extension, or any non-code file under data/, fixtures/, samples/."""
    pure = PurePosixPath(str(path).replace("\\", "/"))
    suffix = pure.suffix.lower()
    if suffix in CODE_EXTENSIONS:
        return False
    return suffix in DATA_EXTENSIONS or any(part in DATA_DIRS for part in pure.parts[:-1])


def _cell_type(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    try:
        int(text)
        return "int"
    except ValueError:
        pass
    try:
        float(text)
        return "float"
    except ValueError:
        pass
    if text.lower() in _BOOLS:
        return "bool"
    if _DATE.match(text):
        return "date"
    if _DATETIME.match(text):
        return "datetime"
    return "str"


def _column_type(values: list[str]) -> str:
    kinds = {kind for kind in map(_cell_type, values) if kind}
    blanks = sum(1 for value in values if not value.strip())
    if not kinds:
        kind = "empty"
    elif kinds == {"int"}:
        kind = "int"
    elif kinds <= {"int", "float"}:
        kind = "float"
    elif len(kinds) == 1:
        kind = next(iter(kinds))
    else:
        kind = "mixed (" + ", ".join(sorted(kinds)) + ")"
    return kind + (f", {blanks} blank" if blanks else "")


def _numbered(lines: list[str], first: int = 1) -> str:
    return "\n".join(f"{number:>4}\t{line[:LINE_CHARS]}" + ("…" if len(line) > LINE_CHARS else "")
                     for number, line in enumerate(lines, first))


def _csv_preview(text: str, lines: list[str], delimiter: str) -> list[str]:
    rows = list(csv.reader(io.StringIO("\n".join(lines[:TYPE_SAMPLE + 1])), delimiter=delimiter))
    if not rows:
        return []
    header, sample = rows[0], rows[1:]
    out = [f"Columns ({len(header)}), types inferred from the first {len(sample)} rows:"]
    for index, name in enumerate(header):
        values = [row[index] if index < len(row) else "" for row in sample]
        out.append(f"  {name[:CELL_CHARS]}: {_column_type(values)}")
    ragged = sum(1 for row in sample if len(row) != len(header))
    if ragged:
        out.append(f"  ({ragged} sampled rows have a different number of fields than the header)")
    out.append(f"First {min(PREVIEW_ROWS, len(lines) - 1)} rows (with the header line, numbered "
               "as read_file numbers them):")
    out.append(_numbered(lines[:PREVIEW_ROWS + 1]))
    return out


def _shape(value: Any) -> str:
    if isinstance(value, dict):
        return f"object with {len(value)} keys"
    if isinstance(value, list):
        return f"list of {len(value)} items"
    return type(value).__name__ if value is not None else "null"


def _short_json(value: Any, limit: int = JSON_ITEM_CHARS) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


def _json_preview(text: str) -> list[str] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    out = [f"Top level: {_shape(data)}"]
    if isinstance(data, list):
        keys: dict[str, None] = {}
        for item in data[:TYPE_SAMPLE]:
            if isinstance(item, dict):
                keys.update(dict.fromkeys(item))
        kinds = sorted({_shape(item).split(" ")[0] for item in data[:TYPE_SAMPLE]})
        out.append(f"Item types: {', '.join(kinds) or 'none'}")
        if keys:
            out.append(f"Keys seen in the first {min(len(data), TYPE_SAMPLE)} items: "
                       + ", ".join(list(keys)[:40]))
        out.append(f"First {min(JSON_ITEMS, len(data))} items:")
        out += [f"  [{index}] {_short_json(item)}" for index, item in enumerate(data[:JSON_ITEMS])]
    elif isinstance(data, dict):
        out.append("Keys:")
        for key, value in list(data.items())[:40]:
            out.append(f"  {key}: {_shape(value)} = {_short_json(value, 160)}")
        if len(data) > 40:
            out.append(f"  ({len(data) - 40} more keys)")
    return out


def _jsonl_preview(lines: list[str]) -> list[str]:
    keys: dict[str, None] = {}
    bad = 0
    for line in lines[:TYPE_SAMPLE]:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(item, dict):
            keys.update(dict.fromkeys(item))
    out = []
    if keys:
        out.append(f"Keys seen in the first {min(len(lines), TYPE_SAMPLE)} lines: "
                   + ", ".join(list(keys)[:40]))
    if bad:
        out.append(f"({bad} of the first {min(len(lines), TYPE_SAMPLE)} lines are not valid JSON)")
    out.append(f"First {min(PREVIEW_ROWS, len(lines))} lines:")
    out.append(_numbered(lines[:PREVIEW_ROWS]))
    return out


def _sqlite_preview(path: Path) -> list[str]:
    out = []
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return [f"Could not open as SQLite: {exc}"]
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        out.append(f"SQLite database, {len(tables)} table(s).")
        for table in tables[:10]:
            quoted = '"' + table.replace('"', '""') + '"'
            columns = [f"{row[1]} {row[2]}".strip() for row in
                       connection.execute(f"PRAGMA table_info({quoted})")]
            count = connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            out.append(f"Table {table} ({count} rows): {', '.join(columns)}")
            for row in connection.execute(f"SELECT * FROM {quoted} LIMIT 5"):
                out.append("  " + _short_json(list(row), 200))
    except sqlite3.Error as exc:
        out.append(f"Could not read the database: {exc}")
    finally:
        connection.close()
    return out


def preview(path: Path, shown_path: str, *, hint: bool = True) -> str | None:
    """A structured preview of a large data file, or None when it should be read whole (code
    files, small files, files that are not text). `hint` adds how to read more with the tools."""
    if not is_data_path(path) or path.suffix.lower() not in DATA_EXTENSIONS:
        return None
    size = path.stat().st_size
    suffix = path.suffix.lower()
    if suffix in {".db", ".sqlite"}:
        return "\n".join([f"Preview of {shown_path} ({size:,} bytes; binary, not shown as text).",
                          *_sqlite_preview(path),
                          *(["To query more, run code with run_check (sqlite3)."] if hint else [])])
    if suffix == ".parquet":
        return (f"Preview of {shown_path} ({size:,} bytes): a binary Parquet file, not readable as "
                "text." + (" To look at it, run code with run_check (e.g. pandas.read_parquet)."
                           if hint else ""))
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    lines = text.splitlines()
    if size <= PREVIEW_MIN_BYTES and len(lines) <= PREVIEW_MIN_LINES:
        return None
    head = [(f"Preview of {shown_path} ({size:,} bytes, {len(lines):,} lines); the whole file "
             "is not shown.")]
    shown_to = min(PREVIEW_ROWS, len(lines))  # the last line number the preview shows
    if suffix in {".csv", ".tsv"}:
        body = _csv_preview(text, lines, "\t" if suffix == ".tsv" else ",")
        shown_to = min(PREVIEW_ROWS + 1, len(lines))
    elif suffix == ".json":
        body = _json_preview(text) or ["Not valid JSON. First lines:",
                                       _numbered(lines[:PREVIEW_ROWS])]
        shown_to = 0 if body[0].startswith("Top level") else shown_to
    elif suffix == ".jsonl":
        body = _jsonl_preview(lines)
    else:
        body = [f"First {min(PREVIEW_ROWS, len(lines))} lines:", _numbered(lines[:PREVIEW_ROWS])]
    more = (f"To read more: read_file with start_line/end_line (e.g. start_line={shown_to + 1}, "
            f"end_line={min(shown_to + 60, len(lines))}); to compute over the whole file, run code "
            "with run_check.")
    return "\n".join([*head, *body, *([more] if hint else [])])
