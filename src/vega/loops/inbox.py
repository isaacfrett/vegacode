"""The inbox: one JSON file per finding in ~/.vega/loops/inbox/, readable without Vega.

A finding's `status` is an append-only history; its current state is the last entry.
States: new → routed → in_progress → proposed → accepted | rejected | failed | duplicate.
A new finding that matches an open (or rejected) one by (loop, where, evidence fingerprint) is
not filed again; the existing one records that it was seen again at the new commit.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .definitions import slugify
from .state import read_json, root, write_json

STATES = ("new", "routed", "in_progress", "proposed", "accepted", "rejected", "failed", "duplicate")
TERMINAL = {"accepted", "rejected", "failed", "duplicate"}
TRANSITIONS = {
    # accepted from new: the user confirms a finding is real (phase 1 has no router, so the user
    # judges findings directly; accepted ÷ all findings is the loop's precision)
    "new": {"routed", "in_progress", "accepted", "rejected", "duplicate"},
    "routed": {"in_progress", "rejected", "duplicate"},
    "in_progress": {"proposed", "failed"},
    "proposed": {"accepted", "rejected", "failed"},
}
# States in which a finding waits for the user: unrouted findings (phase 1 has no router, so the
# user reads them) and proposals awaiting accept / reject. Counted as "needs you" in the panel.
NEEDS_YOU = {"new", "proposed"}
EVIDENCE_TYPES = {"failing_test", "lint", "traceback"}
CONFIDENCE = {"verified", "unverified"}
_SIGNATURE = re.compile(r"^(FAILED|ERROR)\s|^\S+\.py:\d+(:\d+)?:\s")
_NUMBERS = re.compile(r"\bin [\d.]+s\b|:\d+|\[\s*\d+%\]")


def folder() -> Path:
    return root() / "inbox"


def tests_folder() -> Path:
    """Reproducing tests written by model loops, kept as evidence next to the findings."""
    return folder() / "tests"


def fingerprint(evidence: dict[str, Any]) -> str:
    """Stable across commits: the failing test ids or lint codes, with numbers (line numbers,
    timings) removed, plus the evidence type. Falls back to the whole normalised output."""
    output = str(evidence.get("output", ""))
    lines = [line.strip() for line in output.splitlines() if _SIGNATURE.search(line.strip())]
    # A failing test is identified by its id; the message after it varies between runs.
    lines = [" ".join(line.split()[:2]) if line.startswith(("FAILED ", "ERROR ")) else line
             for line in lines]
    basis = lines or [output.strip()]
    normal = sorted({_NUMBERS.sub("#", line) for line in basis})
    text = "\n".join([str(evidence.get("type", "")), *normal])
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _where_key(where: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("path", "")) for item in where})


def dedupe_key(finding: dict[str, Any]) -> str:
    return json.dumps([finding["loop"], _where_key(finding.get("where", [])),
                       finding["evidence"].get("fingerprint") or fingerprint(finding["evidence"])])


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_finding(
    *, loop: str, project: Path | str, commit: str, kind: str, title: str,
    where: list[dict[str, Any]], evidence: dict[str, Any], confidence: str = "verified",
    route: str = "user", debug_log: str | None = None,
) -> dict[str, Any]:
    if evidence.get("type") not in EVIDENCE_TYPES:
        raise ValueError(f"evidence type must be one of {sorted(EVIDENCE_TYPES)}")
    if confidence not in CONFIDENCE:
        raise ValueError(f"confidence must be one of {sorted(CONFIDENCE)}")
    evidence = {**evidence, "fingerprint": evidence.get("fingerprint") or fingerprint(evidence)}
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return {
        "id": f"{stamp}-{slugify(loop)}-{evidence['fingerprint'][:4]}",
        "loop": loop,
        "project": str(project),
        "commit": commit,
        "kind": kind,
        "title": title,
        "where": where,
        "evidence": evidence,
        "confidence": confidence,
        "route": route,
        "status": [{"at": _now(), "state": "new"}],
        "seen_again": [],
        "debug_log": debug_log,
    }


def state_of(finding: dict[str, Any]) -> str:
    return finding["status"][-1]["state"] if finding.get("status") else "new"


def _path(finding_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", finding_id):
        raise ValueError(f"not a finding id: {finding_id!r}")
    return folder() / f"{finding_id}.json"


def save(finding: dict[str, Any]) -> None:
    write_json(_path(finding["id"]), finding)


def get(finding_id: str) -> dict[str, Any]:
    data = read_json(_path(finding_id))
    if not isinstance(data, dict):
        matches = [item for item in items() if item["id"].startswith(finding_id)]
        if len(matches) == 1:
            return matches[0]
        raise LookupError(f"No finding {finding_id!r}" + (" (ambiguous prefix)" if matches else ""))
    return data


def items(loop: str | None = None, *, open_only: bool = False) -> list[dict[str, Any]]:
    """Findings, newest first; optionally one loop's, optionally only those not finished."""
    found = []
    for path in folder().glob("*.json") if folder().is_dir() else []:
        data = read_json(path)
        if not isinstance(data, dict) or "id" not in data:
            continue
        if loop is not None and slugify(data.get("loop", "")) != slugify(loop):
            continue
        if open_only and state_of(data) in TERMINAL:
            continue
        found.append(data)
    return sorted(found, key=lambda item: item["id"], reverse=True)


def add(finding: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """File a finding. Returns (the stored finding, True) or, when it duplicates an open or
    rejected one, (that existing finding with a `seen_again` entry, False)."""
    key = dedupe_key(finding)
    for existing in items(finding["loop"]):
        if state_of(existing) in {"accepted", "failed", "duplicate"}:
            continue  # fixed before (or never fixed): a recurrence is a new finding
        if dedupe_key(existing) == key:
            seen = existing.setdefault("seen_again", [])
            if finding["commit"] not in {entry.get("commit") for entry in seen} | {existing["commit"]}:
                seen.append({"at": _now(), "commit": finding["commit"]})
                save(existing)
            return existing, False
    base, suffix = finding["id"], 1
    while _path(finding["id"]).exists():  # same second, same fingerprint prefix
        suffix += 1
        finding["id"] = f"{base}-{suffix}"
    save(finding)
    return finding, True


def transition(finding_id: str, new_state: str, note: str = "") -> dict[str, Any]:
    """Move a finding to a new state, enforcing the allowed transitions."""
    if new_state not in STATES:
        raise ValueError(f"unknown state {new_state!r} (states: {', '.join(STATES)})")
    finding = get(finding_id)
    current = state_of(finding)
    if new_state not in TRANSITIONS.get(current, set()):
        allowed = ", ".join(sorted(TRANSITIONS.get(current, set()))) or "none (finished)"
        raise ValueError(f"{finding['id']} is {current}; it can move to: {allowed}")
    entry = {"at": _now(), "state": new_state}
    if note:
        entry["note"] = note
    finding["status"].append(entry)
    save(finding)
    return finding


def needs_you(loop: str) -> int:
    return sum(1 for item in items(loop) if state_of(item) in NEEDS_YOU)
