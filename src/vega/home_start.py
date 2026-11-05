"""Home start: `vega` opened in the home folder asks what the user is working on first.

Why: started in the home folder, the user almost certainly wants to work on something, either a
project they have not changed into yet or a new one; that is worth asking first. The home folder
is no project: walking it is slow, and its files say nothing about the request.

What (TUI: tui.VegaApp calls chat_opened and chat_message; routing.route_task calls route_task;
the question pipeline and the agent loop add answer_note's text):

- `chat_opened`: started in the home folder, the filesystem root, a folder above home, or a very
  large folder with no project markers, Vega says so in one line and opens a selector (the
  decision picker's house style: one option recommended, with its reason): open a recent project
  (folders of Vega's own sessions, newest first, then git repositories found one level under ~,
  ~/Developer, ~/Projects and ~/code; at most MAX_PROJECTS), start a new project, or just chat.
  Esc closes it; typing works as well.
- `chat_message`: until a project is chosen, a typed request is read for its intent: a path or a
  known project's name ("fix the tests in vega") offers that project; a build request ("build a
  todo app") offers a new project named from it (~/todo-app, or VEGA_PROJECTS_DIR / ~/Developer /
  ~/Projects / ~/code when one exists); anything else is answered as conversation. A new folder
  is created only after the user confirms it; the request then runs in the chosen project.
- Just chat: `route_task` sends every turn to a one-answer reply with no tools (nothing in the
  folder is read, searched or walked) and the agent cannot write; `answer_context` tells the
  model no project is open (answer_note).
- Switching projects is the core's (VegaApp.switch_project / CodingAgent.change_project): the
  conversation stays, per-project state starts over.

Switch: VEGA_FEATURES_OFF=home_start (on by default). Promoted from the private lab into core on
2026-11-04: the same behaviour, now in the public build.
"""

from __future__ import annotations

import calendar
import json
import os
import re
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import debuglog, features

NAME = "home_start"
PROJECTS_ENV = "VEGA_PROJECTS_DIR"
PROJECT_BASES = ("Developer", "Projects", "code")  # under ~, looked in one level deep
MAX_PROJECTS = 5
MAX_SESSIONS = 200  # newest session files read (their first bytes only)
MAX_CHILDREN = 300  # entries looked at per folder when finding git repositories
LARGE_FOLDERS = 150  # a folder with this many sub-folders and no project marker is "large"
HEAD_BYTES = 4096
NEW = "new"
CHAT = "chat"
LEAD = {
    "home": "You're in your home folder.",
    "root": "You're at the top of the filesystem.",
    "above": "You're in a folder above your home folder.",
    "large": "This folder is large and has no project files.",
}
CHAT_NOTE = ("No project is open: Vega was started outside a project and the user chose to just "
             "chat. Answer from general knowledge; nothing in the folder is read. If they want to "
             "work on code, say they can name a project (or a path) to open, or ask for a new one.")


def enabled() -> bool:
    return features.on(NAME)


# -- where Vega was started ----------------------------------------------------------------------

def _home() -> Path:
    return Path.home().resolve()


def unscoped(project: Any) -> str:
    """"home", "root", "above" or "large" when the folder is no project to work in, else ""."""
    from .tools import PROJECT_MARKERS

    try:
        root = Path(project).resolve()
    except (OSError, TypeError, RuntimeError):
        return ""
    home = _home()
    if root == home:
        return "home"
    if root == Path(root.anchor):
        return "root"
    if home.is_relative_to(root):
        return "above"
    if any((root / marker).exists() for marker in PROJECT_MARKERS):
        return ""
    folders = 0
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                    folders += 1
                    if folders >= LARGE_FOLDERS:
                        return "large"
    except OSError:
        return ""
    return ""


# -- the projects to offer -----------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    path: Path
    source: str  # "session" (Vega worked there) | "git" (a repository found near ~)
    when: float  # epoch seconds: the session's start, or the repository's last change

    @property
    def name(self) -> str:
        return self.path.name


_PROJECT_KEY = re.compile(r'"project"\s*:\s*("(?:[^"\\]|\\.)*")')


def recent_projects(sessions: Path | None, *, limit: int = MAX_PROJECTS) -> list[Candidate]:
    """Folders of Vega's chat sessions (the transcripts in `sessions`), newest first: only the
    newest MAX_SESSIONS files, only their first bytes, only folders that still exist and are
    projects to work in."""
    if sessions is None or not sessions.is_dir():
        return []
    try:
        names = sorted((n for n in os.listdir(sessions) if n.endswith(".json")), reverse=True)
    except OSError:
        return []
    out: list[Candidate] = []
    seen: set[str] = set()
    for name in names[:MAX_SESSIONS]:
        try:
            with open(sessions / name, "rb") as fh:
                head = fh.read(HEAD_BYTES).decode("utf-8", errors="replace")
            match = _PROJECT_KEY.search(head)
            if match is None:
                continue
            path = Path(json.loads(match.group(1)))
        except (OSError, ValueError):
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.is_dir() or unscoped(path):
            continue
        out.append(Candidate(path, "session", _stamp(name)))
        if len(out) >= limit:
            break
    return out


def _stamp(name: str) -> float:
    try:
        return float(calendar.timegm(time.strptime(name[:15], "%Y%m%dT%H%M%S")))  # UTC stamps
    except ValueError:
        return 0.0


def found_repos(home: Path | None = None, *, limit: int = MAX_PROJECTS) -> list[Candidate]:
    """Git repositories one level under ~, ~/Developer, ~/Projects and ~/code (bounded: at most
    MAX_CHILDREN entries per folder, no deeper), most recently changed first."""
    home = home or _home()
    out: list[Candidate] = []
    for base in (home, *(home / name for name in PROJECT_BASES)):
        try:
            with os.scandir(base) as entries:
                for n, entry in enumerate(entries):
                    if n >= MAX_CHILDREN:
                        break
                    if entry.name.startswith(".") or not entry.is_dir(follow_symlinks=False):
                        continue
                    git = Path(entry.path) / ".git"
                    try:
                        when = git.stat().st_mtime
                    except OSError:
                        continue
                    out.append(Candidate(Path(entry.path), "git", when))
        except OSError:
            continue
    out.sort(key=lambda c: c.when, reverse=True)
    return out[:limit]


def candidates(sessions: Path | None, home: Path | None = None) -> list[Candidate]:
    """Vega's own recent projects first, then repositories found near ~; at most MAX_PROJECTS."""
    out = recent_projects(sessions)
    have = {str(c.path) for c in out}
    for repo in found_repos(home):
        if len(out) >= MAX_PROJECTS:
            break
        if str(repo.path) not in have:
            out.append(repo)
    return out[:MAX_PROJECTS]


def _ago(when: float, now: float | None = None) -> str:
    if not when:
        return ""
    seconds = max(0, (now or time.time()) - when)
    for size, unit in ((86400, "d"), (3600, "h"), (60, "min")):
        if seconds >= size:
            return f"{int(seconds // size)} {unit} ago"
    return "just now"


def projects_dir(home: Path | None = None) -> Path:
    """Where a new project goes: VEGA_PROJECTS_DIR, else ~/Developer, ~/Projects or ~/code when
    one exists, else ~."""
    home = home or _home()
    configured = os.environ.get(PROJECTS_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    for name in PROJECT_BASES:
        if (home / name).is_dir():
            return home / name
    return home


# -- what a typed request wants -----------------------------------------------------------------

@dataclass(frozen=True)
class Intent:
    kind: str  # "open" | "new" | "chat"
    path: Path | None = None
    name: str = ""  # the new project's folder name


_BUILD = re.compile(
    r"\b(?:build|create|make|start|scaffold|write|set\s*up|begin|code|develop|prototype)\s+"
    r"(?:me\s+|us\s+)?(?:(?:a|an|the|my|our|new|simple|small|basic|little|quick)\s+)*"
    r"(?P<what>[A-Za-z0-9][\w\s'-]{0,60})", re.IGNORECASE)
_STOP = re.compile(r"\s+(?:that|which|with|for|in|using|to|where|so|on|from|like|and|in)\b.*$",
                   re.IGNORECASE)
_THINGS = re.compile(r"\b(?:app|application|game|site|website|page|tool|cli|api|bot|server|"
                     r"service|script|library|lib|package|project|extension|plugin|dashboard|"
                     r"program|backend|frontend|scraper|tracker|engine|model|agent)s?\b",
                     re.IGNORECASE)
_PATH = re.compile(r"(?:^|\s)((?:~|/)[^\s'\"`]*)")


def slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    words = [w for w in words if w not in {"a", "an", "the", "my", "new", "project"}] or ["project"]
    return "-".join(words[:3])


def new_name(text: str) -> str:
    """The folder name a build request suggests ("build a todo app" -> "todo-app"), or ""."""
    match = _BUILD.search(text)
    if match is None:
        return ""
    raw = match.group("what")
    what = _STOP.sub("", raw).strip(" .,!?")
    if not what or not (_THINGS.search(what) or _THINGS.search(text)):
        return ""
    if slug(what) == "project":  # "a new project for a budget tracker": the part after "for"
        named = re.search(r"\b(?:for|called|named)\s+(?:(?:a|an|the|my)\s+)?(.+)", raw, re.IGNORECASE)
        if named:
            return slug(_STOP.sub("", named.group(1)).strip(" .,!?"))
    return slug(what)


def infer(text: str, offered: list[Candidate], home: Path | None = None) -> Intent:
    """A typed request's intent before a project is open."""
    home = home or _home()
    for raw in _PATH.findall(text):
        path = Path(raw.rstrip(".,;:!?")).expanduser()
        if path.is_dir() and not unscoped(path):
            return Intent("open", path.resolve())
    words = {w.lower() for w in re.findall(r"[\w.-]+", text)}
    for cand in offered:
        if len(cand.name) >= 3 and cand.name.lower() in words:
            return Intent("open", cand.path)
    from .tools import is_project

    for word in sorted(words):
        if len(word) >= 3 and not word.startswith("."):
            folder = home / word
            try:
                if folder.is_dir() and is_project(folder):
                    return Intent("open", folder.resolve())
            except OSError:
                continue
    name = new_name(text)
    if name:
        return Intent("new", name=name)
    return Intent("chat")


# -- the session's state -------------------------------------------------------------------------

@dataclass
class _State:
    kind: str  # unscoped(): "home", "root", ...
    offered: list[Candidate] = field(default_factory=list)
    phase: str = "ask"  # ask | name | chat | done
    request: str = ""  # a typed request waiting for its project


# keyed by the UI object itself (weakly): an id() key could be reused by a later UI and inherit its
# state (seen in tests: a new app started in just-chat mode)
_states: weakref.WeakKeyDictionary[Any, _State] = weakref.WeakKeyDictionary()


def reset() -> None:
    _states.clear()


def state(ui: Any) -> _State | None:
    return _states.get(ui)


def set_chat_only(agent: Any, on: bool) -> None:
    """Just chat: one-answer replies with no tools, no writes (restored when a project opens)."""
    if agent is None:
        return
    if on and not getattr(agent, "_home_chat_only", False):
        agent._home_chat_only, agent._home_can_write = True, getattr(agent, "can_write", True)
        agent.can_write = False
    elif not on and getattr(agent, "_home_chat_only", False):
        agent._home_chat_only = False
        agent.can_write = getattr(agent, "_home_can_write", True)


def chat_only(agent: Any) -> bool:
    return bool(getattr(agent, "_home_chat_only", False))


# -- the selector --------------------------------------------------------------------------------

def _question(kind: str, offered: list[Candidate],
               now: float | None = None) -> tuple[Any, list[tuple[str, str]]]:
    """The selector's question and, per option, (where, when) for its muted columns."""
    from .home_selector import Question

    options, details = [], []
    for cand in offered:
        when = _ago(cand.when, now)
        options.append(f"Open {cand.name}")
        details.append((_tilde(cand.path), when if cand.source == "session"
                        else f"git, {when}" if when else "git"))
    options += ["Start a new project", "Just chat"]
    details += [("", ""), ("no project, nothing read", "")]
    recommended, reason = None, ""
    if offered and offered[0].source == "session":
        recommended, reason = 0, f"Your last session was there, {_ago(offered[0].when, now)}."
    elif offered:
        recommended, reason = 0, "The most recently changed repository near your home folder."
    lead = LEAD.get(kind, LEAD["large"])
    question = Question(f"{lead} What are you working on?", options, recommended, reason)
    return question, details


def _tilde(path: Path) -> str:
    home = str(_home())
    text = str(path)
    return "~" + text[len(home):] if text == home or text.startswith(home + os.sep) else text


def _open_picker(ui: Any, question: Any, title: str, done: Callable[[Any], None],
                 details: list[tuple[str, str]] | None = None) -> bool:
    if not hasattr(ui, "push_screen"):
        return False
    from . import home_selector

    home_selector.open_picker(ui, question, title, done, details)
    return True


# -- entry points (tui, routing, the answer paths) ---------------------------------------------------

def chat_opened(ui: Any, *, project: Any, agent: Any = None, sessions: Path | None = None,
                say: Callable[[str], None] | None = None, switch: Callable[[Path], None] | None = None,
                send: Callable[[str], None] | None = None, **_context: Any) -> bool:
    """The TUI started: outside a project, ask what the user is working on (True: asked)."""
    if not enabled():
        return False
    kind = unscoped(project)
    if not kind:
        return False
    started = time.perf_counter()
    offered = candidates(sessions)
    st = _states[ui] = _State(kind, offered)
    debuglog.record("stage", stage="home_start.opened", where=kind,
                    offered=[str(c.path) for c in offered],
                    seconds=round(time.perf_counter() - started, 3))
    _ask_project(ui, st, agent=agent, say=say, switch=switch, send=send)
    features.mark_acted("home_start", hook="chat_opened", where=kind)
    return True


def _ask_project(ui: Any, st: _State, *, agent: Any, say: Any, switch: Any, send: Any) -> None:
    question, details = _question(st.kind, st.offered)

    def done(pick: Any) -> None:
        if pick is None:  # Esc: typing works as well
            if say:
                say("Type what you're working on, a project's name or a path; or just ask.")
            return
        if isinstance(pick, str):  # their own words
            _typed(ui, st, pick, agent=agent, say=say, switch=switch, send=send, shown=False)
            return
        if pick < len(st.offered):
            _open(st, st.offered[pick].path, agent=agent, say=say, switch=switch, send=send)
        elif pick == len(st.offered):
            st.phase = "name"
            if say:
                say("What should the new project be called? (a name, or a path)")
        else:
            _chat(st, agent, say)

    _open_picker(ui, question, "Before we start", done, details)


def _open(st: _State, path: Path, *, agent: Any, say: Any, switch: Any, send: Any) -> None:
    st.phase = "done"
    set_chat_only(agent, False)
    if switch is not None:
        switch(path)
    debuglog.record("stage", stage="home_start.open", path=str(path), request=bool(st.request))
    request, st.request = st.request, ""
    if request and send is not None:
        send(request)


def _chat(st: _State, agent: Any, say: Any) -> None:
    st.phase = "chat"
    set_chat_only(agent, True)
    debuglog.record("stage", stage="home_start.chat")
    if say:
        say("Just chatting: nothing in this folder is read. Name a project or a path any time to "
            "open it.")


def _confirm_new(ui: Any, st: _State, name: str, *, agent: Any, say: Any, switch: Any,
                 send: Any) -> None:
    """Offer ~/<name> (or the projects folder); created only once the user takes it."""
    from .home_selector import Question

    raw = Path(name).expanduser()
    target = raw if raw.is_absolute() else projects_dir() / slug(name)
    exists = target.is_dir()
    first = (f"Open {_tilde(target)} (it exists)" if exists else f"Create {_tilde(target)}")
    if st.request:
        first += " and do it there"
    options = [first, "Just chat (no project)"]
    question = Question(f"New project: {target.name}", options, 0,
                        "Named from your request." if st.request else "The name you gave.")

    def done(pick: Any) -> None:
        if pick == 0:
            try:
                target.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                if say:
                    say(f"Couldn't create {_tilde(target)}: {exc}")
                return
            debuglog.record("stage", stage="home_start.created", path=str(target),
                            existed=exists)
            _open(st, target, agent=agent, say=say, switch=switch, send=send)
        elif pick == 1:
            st.request = ""
            _chat(st, agent, say)
        elif isinstance(pick, str) and pick.strip():  # another name or path
            _confirm_new(ui, st, pick.strip(), agent=agent, say=say, switch=switch, send=send)
        elif say:
            say("Nothing created. Type a name for the project, or just ask.")
            st.phase = "name"

    if not _open_picker(ui, question, "Start a new project", done) and say:
        say(f"New project: {_tilde(target)} — not created (no selector here).")


def _typed(ui: Any, st: _State, text: str, *, agent: Any, say: Any, switch: Any, send: Any,
           shown: bool = True) -> str | None:
    """A typed message while no project is open: None runs it as a turn now, "" means handled."""
    if st.phase == "name":
        st.phase = "ask"
        _confirm_new(ui, st, new_name(text) or text, agent=agent, say=say, switch=switch, send=send)
        return ""
    intent = infer(text, st.offered)
    debuglog.record("stage", stage="home_start.intent", intent=intent.kind,
                    path=str(intent.path or ""), name=intent.name)
    if intent.kind == "chat":
        if st.phase != "chat":
            _chat(st, agent, None)
        if not shown and send is not None:  # typed into the selector: run it as a turn
            send(text)
            return ""
        return None
    st.request = "" if _only_target(text, intent) else text
    if intent.kind == "open":
        from .home_selector import Question

        target = intent.path
        assert target is not None
        options = [f"Open {_tilde(target)}" + (" and do it there" if st.request else ""),
                   "Stay here and just chat"]
        question = Question(f"That sounds like {target.name}.", options, 0, "You named it.")

        def done(pick: Any) -> None:
            if pick == 0:
                _open(st, target, agent=agent, say=say, switch=switch, send=send)
            elif pick == 1:
                request, st.request = st.request, ""
                _chat(st, agent, say)
                if request and send is not None:
                    send(request)
            elif isinstance(pick, str) and pick.strip():
                _typed(ui, st, pick, agent=agent, say=say, switch=switch, send=send, shown=False)

        _open_picker(ui, question, "Open a project", done)
        return ""
    _confirm_new(ui, st, intent.name, agent=agent, say=say, switch=switch, send=send)
    return ""


_GO = {"open", "go", "to", "cd", "switch", "work", "on", "in", "the", "project", "folder", "let's",
       "lets", "i'm", "im", "working", "please"}


def _only_target(text: str, intent: Intent) -> bool:
    """"vega", "open ~/vega", "cd vega": the message only names the project (no request)."""
    if intent.kind != "open" or intent.path is None:
        return False
    name = intent.path.name.lower()
    return all(w in _GO or w == name or w.startswith(("~", "/"))
               for w in re.findall(r"[\w.~/'-]+", text.lower()))


def chat_message(ui: Any, text: str, *, agent: Any = None, say: Any = None, switch: Any = None,
                 send: Any = None, **_context: Any) -> str | None:
    """A message typed in the TUI. None: run it as usual; "": handled here (a selector opened)."""
    if not enabled():
        return None
    st = _states.get(ui)
    if st is None or st.phase == "done":
        return None
    if st.phase == "chat":
        intent = infer(text, st.offered)
        if intent.kind == "chat":
            return None
    result = _typed(ui, st, text, agent=agent, say=say, switch=switch, send=send)
    if result is not None:
        features.mark_acted("home_start", hook="chat_message")
    return result


def route_task(task: str, *, project: Any, agent: Any = None, **_context: Any) -> Any:
    """Just chat: every turn is one answer with no tools (nothing read, run or written)."""
    if not enabled() or not chat_only(agent):
        return None
    from . import routing

    features.mark_acted("home_start", hook="route_task")
    return routing.Route(routing.AGENT, [f"{NAME}: just chat, no project open"], question=True,
                         mode="reply")


def answer_context(agent: Any, task: str, **_context: Any) -> str:
    """CHAT_NOTE while just chatting, else ""."""
    if not (enabled() and chat_only(agent)):
        return ""
    features.mark_acted("home_start", hook="answer_context")
    return CHAT_NOTE


def answer_note(agent: Any, task: str, extra: str = "") -> str:
    """A turn's answer context: answer_context's note, then `extra` (the extensions' text,
    vega.extensions.answer_context), joined by a blank line; "" when both are empty."""
    return "\n\n".join(part for part in (answer_context(agent, task), extra) if part and part.strip())
