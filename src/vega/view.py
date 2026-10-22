"""What the workbench shows and where: pane placement, column widths, step detail and status bar
items. Saved per user in ~/.vega/view.json (plain JSON, safe to edit or delete); presets cover
the common setups. The TUI applies changes live (/view, /pane, /width, F2)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any

from . import server

PANES: dict[str, str] = {
    "changes": "Files changed this session, by turn (select one to preview its diff; u undoes)",
    "steps": "Every tool step this session (select one to see its output)",
    "preview": "The diff or output of the selected file or step",
    "plan": "The current plan checklist",
    "machine": "Context use, model memory, swap and request times",
    "session": "Model, review mode, reply times and the transcript",
    "loops": "Background loops (Bug Watcher, ...) and what they found; Enter shows findings",
    "git": "Branch, ahead/behind, staged/unstaged/untracked files and the last commits "
           "(Enter previews a diff; r refreshes)",
    "tests": "The last test run this session: counts, time and failing tests (Enter shows a "
             "failure's output; r runs it again)",
    "problems": "Syntax and lint problems in files Vega wrote, by file (Enter shows the line; "
                "l runs ruff)",
    "context": "What the model's context holds: prompt, snapshot, conversation, tool results, "
               "files read, last compaction",
    "models": "Local models: size, whether each fits this machine, the running one and scores "
              "(Enter switches, d downloads, x removes; each asks first)",
    "product": "What the product is, whether it can ship, this session's turns and what is next "
               "(f puts the first blocker's fix request in the message box)",
    "release": "The release checklist: n of m pass, warnings and checks not run (Enter shows the "
               "evidence; f prepares a fix request)",
}
PANE_TITLES = {"changes": "Changes", "steps": "Steps", "preview": "Preview", "plan": "Plan", "loops": "Loops",
               "machine": "Machine", "session": "Session", "git": "Git", "tests": "Tests",
               "problems": "Problems", "context": "Context", "models": "Models",
               "product": "Product", "release": "Release"}
COLUMNS = ("left", "right")
STEP_DETAIL = {
    "collapsed": "a one-line summary under each answer; Enter opens the steps",
    "expanded": "every step listed under each answer",
    "hidden": "no step lines in the conversation (the Steps pane still lists them)",
}
STATUS_ITEMS = {
    "gauge": "context gauge",
    "memory": "model memory",
    "reply": "last reply time",
    "keys": "key hints",
}
LEFT_WIDTHS = (24, 64)  # columns
RIGHT_WIDTHS = (30, 100)  # columns; 0 means "share the space equally with the conversation"
MIN_CONVERSATION = 44  # below this the workbench falls back to one column
MODES = ("product", "code")  # F3 swaps them: the product view and the code view
NEW_USER_PRESET = "product"  # with no saved view.json; a saved one is always kept as it is


@dataclass(frozen=True)
class View:
    left: tuple[str, ...] = ("changes", "git", "steps", "loops")
    right: tuple[str, ...] = ("preview",)
    left_width: int = 30
    right_width: int = 0
    steps: str = "collapsed"
    status: tuple[str, ...] = ("gauge", "memory", "reply", "keys")
    preset: str = "workbench"  # the preset it started from ("custom" once changed)
    teach: bool = False  # teach mode (teaching.py); kept across presets, set with /teach
    mode: str = "code"  # product or code view: which way F3 goes (kept when panes move)

    def column(self, pane: str) -> str | None:
        return "left" if pane in self.left else "right" if pane in self.right else None

    def wide_at(self) -> int:
        """Terminal width below which the side columns collapse into the Ctrl+P overlay."""
        width = MIN_CONVERSATION + 2
        if self.left:
            width += self.left_width
        if self.right:
            width += self.right_width or MIN_CONVERSATION
        return width

    def describe(self) -> str:
        def names(panes: tuple[str, ...]) -> str:
            return ", ".join(PANE_TITLES[p] for p in panes) or "nothing"

        right = f"{self.right_width} columns" if self.right_width else "shares with conversation"
        hidden = [PANE_TITLES[p] for p in PANES if self.column(p) is None]
        return (f"left {names(self.left)} ({self.left_width} columns); right {names(self.right)} "
                f"({right}); hidden {', '.join(hidden) or 'nothing'}; steps in the conversation "
                f"{self.steps}; status bar {', '.join(STATUS_ITEMS[s] for s in self.status)}")


PRESETS: dict[str, tuple[View, str]] = {
    "workbench": (View(), "changes, git, steps and loops | conversation | preview"),
    "focus": (View(left=(), right=(), preset="focus", steps="collapsed"),
              "the conversation alone; Ctrl+P opens the details"),
    "review": (View(left=("changes", "git", "problems"), right=("preview",), right_width=60,
                    left_width=30, preset="review"),
               "changed files, git and problems | conversation | a wide diff preview"),
    "monitor": (View(left=("plan", "steps", "loops"), right=("machine", "preview"), right_width=48,
                     preset="monitor", status=("memory", "reply", "keys")),
                "plan, steps and loops | conversation | machine and preview"),
    "steps": (View(left=(), right=("steps", "preview"), right_width=56, steps="hidden",
                   preset="steps"),
              "conversation | steps and their output; no step lines in the conversation"),
    # last, so F2's number keys keep the presets they had
    "product": (View(left=("product", "release"), right=(), left_width=42, preset="product",
                     mode="product"),
                "product and release checklist | conversation; F3 switches to the code view"),
}


def _path():
    return server.STATE_DIR / "view.json"


def validate(raw: dict[str, Any]) -> View:
    """A View from user JSON. Unknown or duplicated panes, bad widths and unknown values are
    dropped or clamped rather than failing start-up; the result is always usable."""
    base = PRESETS.get(str(raw.get("preset", "")), PRESETS["workbench"])[0]
    seen: set[str] = set()

    def panes(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
        out = []
        for name in value if isinstance(value, list) else default:
            if isinstance(name, str) and name in PANES and name not in seen:
                seen.add(name)
                out.append(name)
        return tuple(out)

    def clamp(value: Any, bounds: tuple[int, int], default: int, zero_ok: bool = False) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            return default
        if zero_ok and value == 0:
            return 0
        return max(bounds[0], min(bounds[1], value))

    left = panes(raw.get("left"), base.left)
    right = panes(raw.get("right"), base.right)
    status = raw.get("status")
    status_items = tuple(s for s in STATUS_ITEMS if isinstance(status, list) and s in status) \
        if isinstance(status, list) else base.status
    steps = raw.get("steps") if raw.get("steps") in STEP_DETAIL else base.steps
    return View(
        left=left, right=right,
        left_width=clamp(raw.get("left_width"), LEFT_WIDTHS, base.left_width),
        right_width=clamp(raw.get("right_width"), RIGHT_WIDTHS, base.right_width, zero_ok=True),
        steps=steps, status=status_items, preset=str(raw.get("preset", base.preset)),
        teach=raw.get("teach") is True,
        mode=raw.get("mode") if raw.get("mode") in MODES else base.mode,
    )


def load() -> View:
    """The saved view; a new user (no view.json yet) starts with NEW_USER_PRESET. An unreadable
    file falls back to the code workbench, as before the product view existed."""
    try:
        raw = json.loads(_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return preset(NEW_USER_PRESET)
    except (OSError, ValueError):
        return View()
    return validate(raw) if isinstance(raw, dict) else View()


def save(view: View) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(view)
    for key in ("left", "right", "status"):
        data[key] = list(data[key])
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def preset(name: str) -> View:
    return PRESETS[name][0]


def place(view: View, pane: str, where: str) -> View:
    """Move a pane to the end of a column (`left`/`right`), or hide it (`hide`)."""
    left = tuple(p for p in view.left if p != pane)
    right = tuple(p for p in view.right if p != pane)
    if where == "left":
        left += (pane,)
    elif where == "right":
        right += (pane,)
    return replace(view, left=left, right=right, preset="custom")


def shift(view: View, pane: str, step: int) -> View:
    """Move a pane up (-1) or down (+1) within its column."""
    where = view.column(pane)
    if where is None:
        return view
    panes = list(getattr(view, where))
    index = panes.index(pane)
    target = max(0, min(len(panes) - 1, index + step))
    panes.insert(target, panes.pop(index))
    return replace(view, **{where: tuple(panes)}, preset="custom")


def resize(view: View, column: str, width: int) -> View:
    if column == "left":
        width = max(LEFT_WIDTHS[0], min(LEFT_WIDTHS[1], width))
        return replace(view, left_width=width, preset="custom")
    if width != 0:
        width = max(RIGHT_WIDTHS[0], min(RIGHT_WIDTHS[1], width))
    return replace(view, right_width=width, preset="custom")


def with_steps(view: View, detail: str) -> View:
    return replace(view, steps=detail, preset="custom")


def with_teach(view: View, on: bool) -> View:
    """Teach mode is a setting, not part of the layout: the preset name stays."""
    return replace(view, teach=on)


def toggle_status(view: View, item: str) -> View:
    items = set(view.status) ^ {item}
    return replace(view, status=tuple(s for s in STATUS_ITEMS if s in items), preset="custom")


__all__ = [
    "COLUMNS", "MODES", "NEW_USER_PRESET", "PANES", "PANE_TITLES", "PRESETS", "STATUS_ITEMS", "STEP_DETAIL", "View",
    "load", "place", "preset", "resize", "save", "shift", "toggle_status", "validate",
    "with_steps", "with_teach",
]
