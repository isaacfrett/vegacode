"""The opt-in local usage log: which views and keys get used, so product decisions rest on real
use (product-vision.md, "Real usage as the test").

Off by default. `/usage on` in the app (or `vega usage on`) turns it on; it is the `usage.log`
setting in ~/.vega/settings.json. While on, events are appended to ~/.vega/usage.jsonl: the event
name, the time and a few short fields (a pane or check name, a key, a count). Never code, file
contents, messages or paths. Nothing is ever sent anywhere; `vega usage` summarises the file and
deleting it removes the history.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import server, settings

SETTING = "usage.log"
FIELD_CHARS = 40  # longest value kept per field: names and keys, never text


def path() -> Path:
    return server.STATE_DIR / "usage.jsonl"  # resolved on use: tests move STATE_DIR


def enabled() -> bool:
    return settings.get(SETTING) is True


def set_enabled(on: bool) -> None:
    settings.set_value(SETTING, on)


def _clean(value: Any) -> Any:
    if isinstance(value, bool | int | float) or value is None:
        return value
    return str(value)[:FIELD_CHARS]


def record(event: str, **fields: Any) -> bool:
    """Append one event when the log is on; True if it was written. Never raises."""
    if not enabled():
        return False
    row = {"t": round(time.time(), 1), "event": event[:FIELD_CHARS],
           **{key: _clean(value) for key, value in fields.items()}}
    try:
        target = path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        return False
    return True


def read(days: float | None = None) -> list[dict[str, Any]]:
    try:
        lines = path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    since = time.time() - days * 86400 if days else 0.0
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and float(row.get("t", 0)) >= since:
            rows.append(row)
    return rows


def summary(days: float = 14) -> str:
    """Event counts over the window, with n and the window stated."""
    rows = read(days)
    state = "on" if enabled() else "off"
    head = f"Usage log is {state} ({path()})."
    if not rows:
        return f"{head}\nNo events in the last {days:g} days."
    sessions = len({row.get("session") for row in rows if row.get("session")})
    counts = Counter(str(row.get("event")) for row in rows)
    lines = [head, f"Last {days:g} days: {len(rows)} events from {sessions} sessions."]
    lines += [f"  {name:<22} {count}" for name, count in counts.most_common()]
    fixes = [row for row in rows if row.get("event") == "fix_prefilled"]
    sent = [row for row in rows if row.get("event") == "fix_sent"]
    if fixes:
        lines.append(f"Fix requests: {len(sent)} sent of {len(fixes)} prepared.")
    return "\n".join(lines)


__all__ = ["enabled", "path", "read", "record", "set_enabled", "summary"]
