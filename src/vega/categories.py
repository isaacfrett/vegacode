"""Capability areas for eval cases (evals/categories.toml): per-area filtering and breakdowns.

A second, independent axis, topics, groups the engineering suite's cases by subject (system
architecture, performance, ...) from their id prefix."""

from __future__ import annotations

import tomllib
from functools import cache
from pathlib import Path
from typing import Any

CATEGORIES_FILE = Path(__file__).resolve().parents[2] / "evals" / "categories.toml"


@cache
def _raw(path: Path = CATEGORIES_FILE) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _load(path: Path = CATEGORIES_FILE) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    data = _raw(path)
    return data.get("areas", {}), data.get("cases", {})


def areas() -> dict[str, dict[str, str]]:
    """Area id -> {name, description}, in the file's order ({} if the file is absent, as in an
    installed package)."""
    try:
        return _load()[0]
    except FileNotFoundError:
        return {}


def tags(case_id: str) -> list[str]:
    """The case's areas, primary first; [] if untagged (or the file is absent)."""
    try:
        found = _load()[1].get(case_id)
        if found is None:  # generated suites are tagged by id prefix
            found = next((areas for prefix, areas in _raw().get("prefixes", {}).items()
                          if case_id.startswith(prefix)), [])
        return list(found)
    except FileNotFoundError:
        return []


def primary(case_id: str) -> str | None:
    found = tags(case_id)
    return found[0] if found else None


def matches(case_id: str, area: str) -> bool:
    return area in tags(case_id)


def breakdown(cases: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Per primary area: [passed, attempts] over run-record cases ({case, passed, attempts})."""
    result: dict[str, list[int]] = {}
    for case in cases:
        if case.get("error"):
            continue
        area = primary(case["case"]) or "untagged"
        totals = result.setdefault(area, [0, 0])
        totals[0] += int(case.get("passed", 0))
        totals[1] += len(case.get("attempts", []))
    order = list(areas()) + ["untagged"]
    return {area: result[area] for area in order if area in result}


def name(area: str) -> str:
    return areas().get(area, {}).get("name", area)


def topics() -> dict[str, str]:
    """Id prefix -> topic name, in the file's order ({} if the file is absent)."""
    try:
        return dict(_raw().get("topics", {}))
    except FileNotFoundError:
        return {}


def topic(case_id: str) -> str | None:
    return next((prefix for prefix in topics() if case_id.startswith(prefix)), None)


def topic_breakdown(cases: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Per topic name: [passed, attempts], only over cases that have a topic."""
    result: dict[str, list[int]] = {}
    names = topics()
    for case in cases:
        prefix = topic(case["case"])
        if prefix is None or case.get("error"):
            continue
        totals = result.setdefault(names[prefix], [0, 0])
        totals[0] += int(case.get("passed", 0))
        totals[1] += len(case.get("attempts", []))
    return {name: result[name] for name in names.values() if name in result}
