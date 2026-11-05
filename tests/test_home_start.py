"""home_start: `vega` started in the home folder asks what the user is working on first, and
`/project` switches folders (CodingAgent.change_project, VegaApp.switch_project). Also the bounded
file listing that keeps a non-git folder (such as ~) fast: tools._project_files' 5,000-file cap,
used by index._list_files and product.tree_fingerprint.

Everything runs against a fake home folder (Path.home patched) and fake session transcripts; the
model is scripted (tui_helpers) and never reached for the selector itself."""

from __future__ import annotations

import json
import os
from pathlib import Path

from collections.abc import Iterator

import pytest

from vega import home_selector, tools, tui
from vega import home_start as hs


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: root))
    monkeypatch.setenv("HOME", str(root))  # expanduser("~/x") too: never the real home folder
    monkeypatch.delenv(hs.PROJECTS_ENV, raising=False)
    hs.reset()
    yield root
    hs.reset()  # the module caches the fake home: later tests (tests/test_tui_layouts.py) must not see it


def repo(path: Path, mtime: float | None = None) -> Path:
    (path / ".git").mkdir(parents=True)
    (path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    if mtime is not None:
        os.utime(path / ".git", (mtime, mtime))
    return path


def session(folder: Path, stamp: str, project: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stamp}-{project.name}.json").write_text(
        json.dumps({"project": str(project), "turns": [], "messages": ["x" * 9000]}))


# -- detection and the offered projects ------------------------------------------------------------

def test_detects_home_root_above_home_and_large_folders(home: Path, tmp_path: Path) -> None:
    assert hs.unscoped(home) == "home"
    assert hs.unscoped(Path("/")) == "root"
    assert hs.unscoped(tmp_path) == "above"
    assert hs.unscoped(repo(home / "vega")) == ""
    big = home / "Downloads"
    for i in range(hs.LARGE_FOLDERS):
        (big / f"d{i}").mkdir(parents=True)
    assert hs.unscoped(big) == "large"
    (big / "package.json").write_text("{}")
    assert hs.unscoped(big) == ""  # a project marker: a project after all
    assert hs.unscoped(home / "vega" / ".git") == ""


def test_recent_projects_newest_first_bounded_and_deduplicated(home: Path, tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    names = [f"p{i}" for i in range(8)]
    for i, name in enumerate(names):
        session(sessions, f"202610{10 + i:02d}T120000", repo(home / name))
    session(sessions, "20261020T120000", home / "p7")  # p7 again, newer
    session(sessions, "20261021T120000", home)  # the home folder itself: never offered
    session(sessions, "20261022T120000", home / "gone")  # deleted since
    found = hs.recent_projects(sessions)
    assert [c.name for c in found] == ["p7", "p6", "p5", "p4", "p3"]
    assert len(found) == hs.MAX_PROJECTS and all(c.source == "session" for c in found)


def test_found_repos_one_level_under_home_and_project_folders(home: Path) -> None:
    repo(home / "old", 1_000_000)
    repo(home / "Developer" / "newer", 2_000_000)
    repo(home / "deep" / "nested" / "repo")  # two levels down: not looked at
    (home / "plain").mkdir()
    (home / ".hidden").mkdir()
    repo(home / ".hidden" / "x")
    assert [c.name for c in hs.found_repos(home)] == ["newer", "old"]


def test_candidates_put_sessions_first(home: Path, tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    session(sessions, "20261027T120000", repo(home / "vega"))
    repo(home / "idle-colony")
    assert [(c.name, c.source) for c in hs.candidates(sessions)] == [
        ("vega", "session"), ("idle-colony", "git")]


# -- typed requests ------------------------------------------------------------------------------

@pytest.mark.parametrize(("text", "name"), [
    ("build a todo app", "todo-app"),
    ("Make me a simple snake game in python", "snake-game"),
    ("create a CLI that renames photos", "cli"),
    ("start a new project for a budget tracker", "budget-tracker"),
    ("what can you do", ""),
    ("make the tests pass", ""),
])
def test_new_project_name_from_a_build_request(text: str, name: str) -> None:
    assert hs.new_name(text) == name


def test_infer_open_new_or_chat(home: Path, tmp_path: Path) -> None:
    vega = repo(home / "vega")
    other = repo(tmp_path / "elsewhere" / "proj")
    offered = [hs.Candidate(vega, "session", 0.0)]
    assert hs.infer("fix the tests in vega", offered) == hs.Intent("open", vega)
    assert hs.infer(f"look at {other}", offered) == hs.Intent("open", other.resolve())
    repo(home / "starsplit")
    assert hs.infer("starsplit crashes on launch", []).path == (home / "starsplit").resolve()
    assert hs.infer("build a todo app", offered) == hs.Intent("new", name="todo-app")
    assert hs.infer("what can you do", offered).kind == "chat"


def test_projects_dir_prefers_the_configured_one(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert hs.projects_dir() == home
    (home / "Projects").mkdir()
    assert hs.projects_dir() == home / "Projects"
    monkeypatch.setenv(hs.PROJECTS_ENV, str(home / "work"))
    assert hs.projects_dir() == home / "work"


# -- the TUI flow -----------------------------------------------------------------------------------

def app_in(home: Path, tmp_path: Path, replies: list | None = None):
    from tui_helpers import make_app, scripted_agent

    agent, seen = scripted_agent(home, replies or [])
    app = make_app(home, agent, tmp_path, layout="chat")
    return app, seen


def selector(app) -> home_selector.HomeScreen | None:
    screen = app.screen
    return screen if isinstance(screen, home_selector.HomeScreen) else None


def notes(app) -> str:
    return "\n".join(str(n._body) for n in app.query(tui.Note))


def test_home_start_asks_first_with_recent_projects(home: Path, tmp_path: Path) -> None:
    from tui_helpers import drive, wait_until

    vega = repo(home / "vega")
    session(tmp_path / "sessions", "20261027T120000", vega)
    app, _ = app_in(home, tmp_path)

    async def script(pilot) -> None:
        await wait_until(pilot, lambda: selector(app) is not None)
        screen = selector(app)
        assert screen.question.options == ["Open vega", "Start a new project", "Just chat"]
        assert screen.question.recommended == 0
        assert screen.details[0][0] == "~/vega"
        assert tui.NOT_A_PROJECT not in notes(app)  # the selector says it instead
        await pilot.press("enter")  # the recommended project
        await wait_until(pilot, lambda: app.project == vega.resolve())
        assert app.agent.project == vega.resolve()
        assert "Now working in ~/vega" in notes(app)

    drive(app, script)


def test_just_chat_reads_nothing_and_never_writes(home: Path, tmp_path: Path) -> None:
    from tui_helpers import drive, send, wait_until

    (home / "notes.txt").write_text("private")
    app, _ = app_in(home, tmp_path)
    asked: list[dict] = []

    def create(**kwargs):
        asked.append(kwargs)
        from types import SimpleNamespace

        msg = SimpleNamespace(content="I can help you write and fix code.", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    app.agent.client.chat.completions.create = create
    walks: list[str] = []  # tree walks under the home folder (os.walk, Path.rglob) during the turn
    real_walk, real_rglob = os.walk, Path.rglob

    def walk(top, *args, **kwargs):
        if Path(top).resolve().is_relative_to(home.resolve()):
            walks.append(f"os.walk {top}")
        return real_walk(top, *args, **kwargs)

    def rglob(self, *args, **kwargs):
        if self.resolve().is_relative_to(home.resolve()):
            walks.append(f"rglob {self}")
        return real_rglob(self, *args, **kwargs)

    async def script(pilot) -> None:
        await wait_until(pilot, lambda: selector(app) is not None)
        await pilot.press("b")  # Just chat (no recent projects: a new, b chat)
        await wait_until(pilot, lambda: selector(app) is None)
        assert hs.chat_only(app.agent) and not app.agent.can_write
        walks.clear()
        os.walk, Path.rglob = walk, rglob
        try:
            await send(pilot, "what can you do")
        finally:
            os.walk, Path.rglob = real_walk, real_rglob
        assert app.agent.last_route["mode"] == "reply"
        assert len(asked) == 1 and "tools" not in asked[0]  # one answer, no tool on offer
        assert hs.CHAT_NOTE in asked[0]["messages"][-1]["content"]
        assert walks == []  # nothing in the home folder was walked

    drive(app, script)


def test_new_project_is_created_only_after_confirming(home: Path, tmp_path: Path) -> None:
    from tui_helpers import drive, send, wait_until

    app, _ = app_in(home, tmp_path)

    async def script(pilot) -> None:
        await wait_until(pilot, lambda: selector(app) is not None)
        await pilot.press("a")  # Start a new project
        await wait_until(pilot, lambda: selector(app) is None)
        assert "What should the new project be called?" in notes(app)
        await send(pilot, "Recipe box", wait=False)
        await wait_until(pilot, lambda: selector(app) is not None)
        assert selector(app).question.options[0] == "Create ~/recipe-box"
        assert not (home / "recipe-box").exists()  # not before the owner confirms
        await pilot.press("escape")  # declined: nothing created
        await wait_until(pilot, lambda: selector(app) is None)
        assert not (home / "recipe-box").exists()
        await send(pilot, "recipe box", wait=False)
        await wait_until(pilot, lambda: selector(app) is not None)
        await pilot.press("enter")
        await wait_until(pilot, lambda: app.project == (home / "recipe-box").resolve())
        assert (home / "recipe-box").is_dir()

    drive(app, script)


def test_a_typed_build_request_offers_a_new_project_then_runs_there(home: Path,
                                                                    tmp_path: Path) -> None:
    from tui_helpers import answer, drive, send, wait_until

    app, _ = app_in(home, tmp_path, [answer("Plan: a todo app.")])

    async def script(pilot) -> None:
        await wait_until(pilot, lambda: selector(app) is not None)
        await pilot.press("escape")  # typing instead
        await wait_until(pilot, lambda: selector(app) is None)
        await send(pilot, "build a todo app in python", wait=False)
        await wait_until(pilot, lambda: selector(app) is not None)
        assert selector(app).question.options[0] == "Create ~/todo-app and do it there"
        await pilot.press("enter")
        await wait_until(pilot, lambda: app.project == (home / "todo-app").resolve())
        # the request runs as a turn in the new project (in the public core a vague build in an
        # empty folder meets the clarify gate's questions first: no model call needed for that)
        await wait_until(pilot, lambda: not app.working and len(app.agent.messages) > 2)
        assert app.agent.project == (home / "todo-app").resolve()
        assert app.agent.messages[1] == {"role": "user", "content": "build a todo app in python"}
        assert "Now working in ~/todo-app" in notes(app)

    drive(app, script)


def test_a_typed_request_naming_a_project_offers_it(home: Path, tmp_path: Path) -> None:
    from tui_helpers import answer, drive, send, wait_until

    vega = repo(home / "vega")
    session(tmp_path / "sessions", "20261027T120000", vega)
    app, seen = app_in(home, tmp_path, [answer("Looking at the tests.")])

    async def script(pilot) -> None:
        await wait_until(pilot, lambda: selector(app) is not None)
        await pilot.press("escape")
        await wait_until(pilot, lambda: selector(app) is None)
        await send(pilot, "fix the tests in vega", wait=False)
        await wait_until(pilot, lambda: selector(app) is not None)
        assert selector(app).question.options == ["Open ~/vega and do it there",
                                                  "Stay here and just chat"]
        await pilot.press("enter")
        await wait_until(pilot, lambda: app.project == vega.resolve())
        await wait_until(pilot, lambda: bool(seen) and not app.working)

    drive(app, script)


def test_a_project_folder_starts_without_the_selector(home: Path, tmp_path: Path) -> None:
    from tui_helpers import drive, make_app, scripted_agent

    work = repo(home / "vega")
    agent, _ = scripted_agent(work, [])
    app = make_app(work, agent, tmp_path, layout="chat")

    async def script(pilot) -> None:
        await pilot.pause()
        assert selector(app) is None and hs.state(app) is None

    drive(app, script)


def test_off_switch(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tui_helpers import drive

    monkeypatch.setenv("VEGA_FEATURES_OFF", "home_start")
    app, _ = app_in(home, tmp_path)

    async def script(pilot) -> None:
        await pilot.pause()
        assert selector(app) is None
        assert tui.NOT_A_PROJECT in notes(app)

    drive(app, script)


def test_project_command_switches_and_keeps_the_conversation(home: Path, tmp_path: Path) -> None:
    from tui_helpers import answer, drive, send, wait_until

    vega = repo(home / "vega")
    (vega / "notes.py").write_text("x = 1\n")
    app, _ = app_in(home, tmp_path, [answer("Hello."), answer("It sets x.")])

    async def script(pilot) -> None:
        await wait_until(pilot, lambda: selector(app) is not None)
        await pilot.press("escape")
        await wait_until(pilot, lambda: selector(app) is None)
        await send(pilot, "/project ~/vega")
        assert app.project == vega.resolve() and app.agent.project == vega.resolve()
        assert "Project root: " + str(vega.resolve()) in app.agent.messages[0]["content"]
        assert app.session_log.project == vega.resolve()
        assert hs.chat_message(app, "what does notes.py do?") is None  # a project is open now
        assert not hs.chat_only(app.agent) and app.agent.can_write

    drive(app, script)


def test_project_command_names_a_missing_folder_and_stays(home: Path, tmp_path: Path) -> None:
    from tui_helpers import drive, send, wait_until

    app, _ = app_in(home, tmp_path)

    async def script(pilot) -> None:
        await wait_until(pilot, lambda: selector(app) is not None)
        await pilot.press("escape")
        await wait_until(pilot, lambda: selector(app) is None)
        before = app.project
        await send(pilot, "/project ~/nowhere")
        assert app.project == before and app.agent.project == before
        assert "No folder at" in notes(app)

    drive(app, script)


# -- the selector's keys ------------------------------------------------------------------------------

def test_selector_question_drops_an_out_of_range_recommendation() -> None:
    q = home_selector.Question("Q?", ["a", "b"], 5, "why")
    assert q.recommended is None and q.reason == ""
    assert home_selector.Question("Q?", ["a", "b"], 1, "why").reason == "why"


@pytest.mark.parametrize(("keys", "result"), [
    (["enter"], 0),  # the recommended option
    (["r"], 0),  # take the recommended one
    (["down", "enter"], 1),
    (["2"], 1),  # digits pick too
    (["b"], 1),
    (["escape"], None),
])
def test_selector_keys(home: Path, tmp_path: Path, keys: list[str], result: object) -> None:
    from tui_helpers import drive, wait_until

    app, _ = app_in(repo(home / "work"), tmp_path)
    got: list[object] = []

    async def script(pilot) -> None:
        await pilot.pause()
        question = home_selector.Question("Where?", ["Open x", "Just chat"], 0, "Last session.")
        home_selector.open_picker(app, question, "Before we start", got.append)
        await wait_until(pilot, lambda: selector(app) is not None)
        for key in keys:
            await pilot.press(key)
        await wait_until(pilot, lambda: bool(got))

    drive(app, script)
    assert got == [result]


def test_selector_takes_typed_words_from_its_last_row(home: Path, tmp_path: Path) -> None:
    from tui_helpers import drive, wait_until

    app, _ = app_in(repo(home / "work"), tmp_path)
    got: list[object] = []

    async def script(pilot) -> None:
        await pilot.pause()
        question = home_selector.Question("Where?", ["Open x", "Just chat"])  # no recommendation
        home_selector.open_picker(app, question, "Before we start", got.append)
        await wait_until(pilot, lambda: selector(app) is not None)
        await pilot.press("enter")  # no recommendation and not moved: a notice, nothing taken
        assert not got and "No recommendation" in selector(app).notice
        await pilot.press("c")  # Something else…: the typing line
        for char in "fix vega":
            await pilot.press("space" if char == " " else char)
        await pilot.press("enter")
        await wait_until(pilot, lambda: bool(got))

    drive(app, script)
    assert got == ["fix vega"]


# -- routing and the answer note while just chatting ----------------------------------------------

def test_route_and_note_only_while_just_chatting(home: Path) -> None:
    from types import SimpleNamespace

    from vega import features, routing

    agent = SimpleNamespace(can_write=True)
    assert hs.route_task("what can you do", project=home, agent=agent) is None
    assert hs.answer_note(agent, "x", "extra") == "extra"
    hs.set_chat_only(agent, True)
    assert agent.can_write is False
    features.reset_acted()
    route = hs.route_task("what can you do", project=home, agent=agent)
    assert route.name == routing.AGENT and route.mode == "reply"
    assert routing.route_task("fix the bug in app.py", home, agent=agent).mode == "reply"
    assert hs.answer_note(agent, "x", "") == hs.CHAT_NOTE
    assert hs.answer_note(agent, "x", "extra") == hs.CHAT_NOTE + "\n\nextra"
    assert "home_start" in features.acted_names()
    hs.set_chat_only(agent, False)
    assert agent.can_write is True and hs.answer_note(agent, "x") == ""


def test_off_switch_leaves_routing_alone(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    agent = SimpleNamespace(can_write=True)
    hs.set_chat_only(agent, True)
    monkeypatch.setenv("VEGA_FEATURES_OFF", "home_start")
    assert hs.route_task("hi", project=home, agent=agent) is None
    assert hs.answer_note(agent, "hi") == ""
    assert hs.chat_opened(object(), project=home) is False


# -- CodingAgent.change_project --------------------------------------------------------------------

def test_change_project_keeps_the_conversation_and_resets_project_state(tmp_path: Path) -> None:
    from vega.agent import CodingAgent

    first, second = repo(tmp_path / "first"), repo(tmp_path / "second")
    (second / "main.py").write_text("print('hi')\n")
    agent = CodingAgent(first)
    agent.messages.append({"role": "user", "content": "build a todo app"})
    agent.checkpoints = [object()]
    agent.plan = [{"step": "x", "status": "pending"}]
    agent.change_project(second)
    assert agent.project == second.resolve() and agent.web.project == second.resolve()
    assert agent.checkpoints == [] and agent.plan == []
    assert f"Project root: {second.resolve()}" in agent.messages[0]["content"]
    assert "main.py" in agent.messages[0]["content"]  # the new folder's snapshot
    assert agent.messages[-1] == {"role": "user", "content": "build a todo app"}


# -- the bounded listing outside git (the 5,000-file cap) ------------------------------------------

def test_project_files_stop_at_the_cap_and_never_enter_ignored_folders(tmp_path: Path) -> None:
    from vega import index, product

    big = tmp_path / "big"  # no .git: the walk fallback
    for i in range(tools.WALK_LIMIT + 7):
        sub = big / f"d{i // 500:02d}"
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"f{i:05d}.txt").write_text("")
    (big / "node_modules" / "pkg").mkdir(parents=True)
    (big / "node_modules" / "pkg" / "index.js").write_text("")
    walked = list(tools._project_files(big))
    assert tools.WALK_LIMIT == 5000 and len(walked) == 5000
    assert not any("node_modules" in p.parts for p in walked)
    assert len(list(tools._project_files(big, limit=10))) == 10
    listed = index._list_files(big)
    assert len(listed) == 5000 and listed == sorted(listed)
    assert not any(name.startswith("node_modules/") for name in listed)
    before = product.tree_fingerprint(big)
    (big / "d10" / "f99999.txt").write_text("past the cap")  # sorts after d00..d09 at the cap
    assert product.tree_fingerprint(big) == before  # bounded: past the cap is not read


def test_project_files_skip_library_only_at_the_top_of_home(home: Path) -> None:
    (home / "Library" / "Caches").mkdir(parents=True)
    (home / "Library" / "Caches" / "x").write_text("")
    (home / "work" / "Library").mkdir(parents=True)
    (home / "work" / "Library" / "y.py").write_text("")
    names = {p.relative_to(home).as_posix() for p in tools._project_files(home)}
    assert names == {"work/Library/y.py"}
