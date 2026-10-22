"""Loops: small, single-purpose jobs that watch a project and report findings with evidence.

A loop is a trigger, a skill (role), permissions, a budget and a status line, defined in one TOML
file (see `definitions`). The daemon (`scheduler`) runs due jobs; deterministic checks run any
time, model work only while the machine is idle (`idle`: screen locked, resources fine, on power).
Findings go to the inbox (`inbox`) as JSON files. Bug Watcher (`bug_watcher`, skill `diff-bug`) is
the first loop. Design: the agent-network plan.

The API a panel renders:
    status.rows() -> (list[LoopRow], definition errors)   one row per loop
    status.header(rows) -> Header                          idle rule and tab badge
    inbox.items(loop=None, open_only=False) -> list[dict]  findings, newest first
    inbox.transition(id, state, note)                      accept / reject / mark duplicate
    state.set_paused(name | None, paused)                  pause / resume
    scheduler.run_now(loop)                                run now (model pass only when idle)
"""

from __future__ import annotations

from .definitions import LoopDef, LoopDefinitionError, Trigger, ensure_default, find, load_all
from .status import Header, LoopRow, header, row, rows, status_text

__all__ = [
    "Header", "LoopDef", "LoopDefinitionError", "LoopRow", "Trigger", "ensure_default", "find",
    "header", "load_all", "row", "rows", "status_text",
]
