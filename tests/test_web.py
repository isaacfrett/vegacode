"""Web research (vega/web.py): permission, secret scrubbing, formats, SearXNG, caching, evals.

No test reaches the network: SearXNG is a local http.server on 127.0.0.1. Models are scripted.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from vega import evals, features, knowledge, settings, web
from vega.agent import CodingAgent



def _sandbox_works() -> bool:
    """sandbox-exec exists and is not already inside a sandbox (the hidden tests run in it)."""
    import subprocess

    if not evals.sandbox.available():
        return False
    probe = subprocess.run(["sandbox-exec", "-p", "(version 1)(allow default)", "true"],
                           capture_output=True, check=False)
    return probe.returncode == 0


@pytest.fixture(autouse=True)
def isolated_web(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("VEGA_WEB_BACKEND", "VEGA_SEARXNG_URL", "VEGA_WEB_PERMISSION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VEGA_KNOWLEDGE_DIR", str(tmp_path / "library"))


# -- a scripted model ------------------------------------------------------------------------

class FakeMessage(SimpleNamespace):
    def model_dump(self, exclude_none: bool = False) -> dict:
        data = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in self.tool_calls]
        return data


def text(content: str) -> FakeMessage:
    return FakeMessage(content=content, tool_calls=None)


def call(name: str, **args: Any) -> FakeMessage:
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    return FakeMessage(content=None, tool_calls=[SimpleNamespace(id=f"c-{name}", function=fn)])


def script(agent: CodingAgent, replies: list[FakeMessage], seen: list) -> CodingAgent:
    def create(**kwargs: Any) -> Any:
        seen.append({"messages": list(kwargs["messages"]),
                     "tools": [t["function"]["name"] for t in kwargs.get("tools") or []]})
        reply = replies.pop(0) if replies else text("Done.")
        return SimpleNamespace(choices=[SimpleNamespace(message=reply)], usage=None)

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return agent


class StubBackend:
    """A backend with fixed results (no HTTP)."""

    name = "stub"
    base_url = "http://stub.invalid"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, n: int = 5) -> list[web.Result]:
        self.queries.append(query)
        return [web.Result("Widget 2.0 release notes", "https://docs.widget.example/2.0",
                           "connect() takes timeout_ms now.", "2026-05-01")]

    def fetch(self, url: str) -> web.Page | None:
        if "docs.widget.example/2.0" not in url:
            return None
        return web.Page("https://docs.widget.example/2.0", "Widget 2.0 release notes",
                        "2026-05-01", "# Widget 2.0\nconnect(timeout=) became timeout_ms=.")


# -- formats -----------------------------------------------------------------------------------


def test_result_format_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    session = web.WebSession(Path("."), backend=StubBackend(), permission="allow", cache=False)
    # web_official_hint marks the product's own host (docs.widget.example for "widget")
    assert session.call("web_search", {"query": "widget timeout"}) == (
        'Search results for "widget timeout" (1):\n'
        "1. Widget 2.0 release notes\n"
        "   https://docs.widget.example/2.0  (official site)\n"
        "   published 2026-05-01 - connect() takes timeout_ms now.")
    monkeypatch.setenv("VEGA_FEATURES_OFF", "web_official_hint")
    assert session.call("web_search", {"query": "widget timeout"}) == (
        'Search results for "widget timeout" (1):\n'
        "1. Widget 2.0 release notes\n"
        "   https://docs.widget.example/2.0\n"
        "   published 2026-05-01 - connect() takes timeout_ms now.")
    monkeypatch.delenv("VEGA_FEATURES_OFF")
    assert session.call("fetch_page", {"url": "https://docs.widget.example/2.0"}) == (
        "URL: https://docs.widget.example/2.0\nTitle: Widget 2.0 release notes\n"
        "Published: 2026-05-01\n\n# Widget 2.0\nconnect(timeout=) became timeout_ms=.")
    assert session.call("fetch_page", {"url": "https://docs.widget.example/nope"}) == \
        "Error: 404 Not Found (no such page in this web): https://docs.widget.example/nope"


# -- secrets and code never leave --------------------------------------------------------------

# Token-shaped fakes are joined at runtime so secret scanners (gitleaks, GitGuardian) skip them.
FAKE_AWS_KEY = "".join(["AKIA", "IOSFODNN7", "EXAMPLE"])


@pytest.mark.parametrize("query", [
    f"why does {FAKE_AWS_KEY} get AccessDenied",
    "openai error with key " + "".join(["sk", "-proj-", "abcdefghijklmnopqrstuvwx123"]),
    "".join(["ghp", "_", "abcdefghijklmnopqrstuvwxyz0123456789"]) + " bad credentials",
    "postgres password=hunter2x9 login fails",
    "fetch https://user:s3cretpw@db.example/ fails",
    # a fake JWT
    "token " + ".".join(["eyJhbGciOiJIUzI1NiJ9", "eyJzdWIiOiIxMjM0NTY3ODkwIn0",  # noqa: FLY002
                         "abcdefghijkl"]) + " invalid",
])
def test_queries_with_secrets_are_refused(tmp_path: Path, query: str) -> None:
    backend = StubBackend()
    session = web.WebSession(tmp_path, backend=backend, permission="allow", cache=False)
    result = session.call("web_search", {"query": query})
    assert result.startswith("Search not sent: the query looks like it contains"), result
    assert backend.queries == []  # nothing was sent


def test_values_from_the_projects_secret_files_are_refused(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("DATABASE_URL=postgres://localhost/app\nSTRIPE=plainsecret42\n")
    backend = StubBackend()
    session = web.WebSession(tmp_path, backend=backend, permission="allow", cache=False)
    result = session.call("web_search", {"query": "stripe plainsecret42 rejected"})
    assert "the project's secret files" in result and backend.queries == []
    assert session.call("web_search", {"query": "stripe api key rejected"}).startswith(
        "Search results")


def test_code_and_long_queries_are_refused(tmp_path: Path) -> None:
    backend = StubBackend()
    session = web.WebSession(tmp_path, backend=backend, permission="allow", cache=False)
    assert session.call("web_search", {"query": "def f(x):\n    return x"}).startswith(
        "Search not sent")
    assert session.call("web_search", {"query": "x" * 201}).startswith("Search not sent")
    assert session.call("web_search", {"query": "if (a == b) { return c; }"}).startswith(
        "Search not sent")
    assert backend.queries == []
    ok = "TypeError: connect() got an unexpected keyword argument 'timeout' zinkdb 5.0"
    assert web.check_query(ok) is None
    assert session.call("fetch_page", {"url": f"https://x.example/?api_key={FAKE_AWS_KEY}"}) \
        .startswith("Fetch not sent")


# -- permission --------------------------------------------------------------------------------

def test_permission_is_asked_once_per_session(tmp_path: Path) -> None:
    asked: list[tuple[str, str]] = []
    session = web.WebSession(tmp_path, backend=StubBackend(), cache=False)
    approve = lambda kind, detail: asked.append((kind, detail)) or "session"
    assert session.call("web_search", {"query": "widget timeout"}, approve).startswith("Search")
    assert session.call("fetch_page", {"url": "https://docs.widget.example/2.0"}, approve) \
        .startswith("URL:")
    assert session.call("web_search", {"query": "widget 2"}, approve).startswith("Search")
    assert asked == [("search", "widget timeout")]
    # a new session asks again: "this session" is not stored
    assert settings.get("web.permission", tmp_path) == "ask"
    again = web.WebSession(tmp_path, backend=StubBackend(), cache=False)
    again.call("web_search", {"query": "widget"}, approve)
    assert len(asked) == 2


def test_denied_permission_returns_a_clear_message_and_is_not_asked_again(tmp_path: Path) -> None:
    asked: list[str] = []
    backend = StubBackend()
    session = web.WebSession(tmp_path, backend=backend, cache=False)
    approve = lambda kind, detail: asked.append(detail) or "no"
    first = session.call("web_search", {"query": "widget timeout"}, approve)
    second = session.call("fetch_page", {"url": "https://docs.widget.example/2.0"}, approve)
    assert first == second == web.DECLINED_TEXT
    assert "Continue without them" in first and asked == ["widget timeout"]
    assert backend.queries == [] and not session.available()


def test_always_for_this_project_is_stored_in_the_project(tmp_path: Path) -> None:
    session = web.WebSession(tmp_path, backend=StubBackend(), cache=False)
    session.call("web_search", {"query": "widget"}, lambda kind, detail: "project")
    stored = json.loads((tmp_path / ".vega" / "settings.json").read_text())
    assert stored == {"web": {"permission": "allow"}}
    later = web.WebSession(tmp_path, backend=StubBackend(), cache=False)
    never = lambda kind, detail: pytest.fail("asked again")
    assert later.call("web_search", {"query": "widget"}, never).startswith("Search results")
    other = tmp_path / "other"
    other.mkdir()
    asked: list[str] = []
    web.WebSession(other, backend=StubBackend(), cache=False).call(
        "web_search", {"query": "widget"}, lambda kind, detail: asked.append(detail) or "no")
    assert asked == ["widget"]  # only that project


def test_settings_deny_and_no_backend_make_the_tools_inert(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    session = web.WebSession(tmp_path)  # default settings: backend none
    assert isinstance(session.backend, web.NoBackend) and not session.available()
    result = session.call("web_search", {"query": "widget"})
    assert result.startswith("Web research unavailable") and "vega web set" in result
    settings.set_value("web.backend", "searxng")
    assert isinstance(web.WebSession(tmp_path).backend, web.SearxngBackend)
    assert web.WebSession(tmp_path).available()
    settings.set_value("web.permission", "deny")
    denied = web.WebSession(tmp_path)
    assert not denied.available()
    assert denied.call("web_search", {"query": "w"}, lambda *a: "session") == web.DECLINED_TEXT
    monkeypatch.setenv("VEGA_WEB_PERMISSION", "allow")
    assert web.WebSession(tmp_path).available()
    with pytest.raises(ValueError):
        settings.set_value("web.backend", "google")


def test_console_prompt_in_line_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vega import ui

    printed: list[str] = []
    answers = iter(["p"])
    monkeypatch.setattr(ui.console, "print", lambda *a, **k: printed.append(str(a[0]) if a else ""))
    monkeypatch.setattr(ui.console, "input", lambda prompt: next(answers))
    session = web.WebSession(tmp_path, backend=StubBackend(), cache=False)
    assert session.call("web_search", {"query": "widget timeout"}).startswith("Search results")
    assert any('Vega wants to search the web for: "widget timeout"' in line for line in printed)
    assert settings.get("web.permission", tmp_path) == "allow"

    def closed(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr(ui.console, "input", closed)
    assert web.console_approve("fetch", "https://x.example") == "no"  # no terminal: no


# -- SearXNG over HTTP (a local fake server) ---------------------------------------------------

class _Handler(http.server.BaseHTTPRequestHandler):
    routes: ClassVar[dict[str, tuple[int, str, bytes]]] = {}
    requests: ClassVar[list[str]] = []

    def do_GET(self) -> None:
        self.requests.append(self.path)
        path = self.path.split("?", 1)[0]
        status, kind, body = self.routes.get(path, (404, "text/plain", b"not found"))
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def searxng():
    _Handler.requests = []
    _Handler.routes = {
        "/search": (200, "application/json", json.dumps({"results": [
            {"title": "zinkdb 5.0 migration", "url": "https://zinkdb.example/docs/5.0/migration",
             "content": "connect(timeout=) was replaced by connect_timeout_ms.",
             "publishedDate": "2026-08-02T10:00:00"},
            {"title": "dup", "url": "https://zinkdb.example/docs/5.0/migration/", "content": ""},
            {"title": "Forum", "url": "https://forum.example/t/1", "content": "use timeout_s",
             "publishedDate": None},
        ]}).encode()),
        "/page": (200, "text/html; charset=utf-8", (
            b"<html><head><title>Zinkdb 5.0 migration</title>"
            b'<meta property="article:published_time" content="2026-08-02T10:00:00Z"></head>'
            b"<body><nav>menu</nav><h1>Migration</h1><p>Use <code>connect_timeout_ms</code>.</p>"
            b"<script>track()</script></body></html>")),
        "/big": (200, "text/plain", b"a" * (web.MAX_BYTES + 500)),
        "/pdf": (200, "application/pdf", b"%PDF-1.4"),
        "/forbidden/search": (403, "text/plain", b"no"),
    }
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_searxng_search_and_fetch(searxng: str, tmp_path: Path) -> None:
    backend = web.SearxngBackend(searxng, allow_private=True)
    session = web.WebSession(tmp_path, backend=backend, permission="allow", cache=False)
    result = session.call("web_search", {"query": "zinkdb 5 timeout"})
    assert result == (
        'Search results for "zinkdb 5 timeout" (2):\n'
        "1. zinkdb 5.0 migration\n   https://zinkdb.example/docs/5.0/migration  (official site)\n"
        "   published 2026-08-02 - connect(timeout=) was replaced by connect_timeout_ms.\n"
        "2. Forum\n   https://forum.example/t/1\n   published unknown - use timeout_s")
    assert _Handler.requests[0] == "/search?q=zinkdb+5+timeout&format=json"  # only the query
    page = session.call("fetch_page", {"url": searxng + "/page"})
    assert page.startswith(f"URL: {searxng}/page\nTitle: Zinkdb 5.0 migration\n"
                           "Published: 2026-08-02\n\n")
    assert "connect_timeout_ms" in page and "track()" not in page and "menu" not in page
    assert session.call("fetch_page", {"url": searxng + "/missing"}) == \
        web.NOT_FOUND + searxng + "/missing"
    big = session.call("fetch_page", {"url": searxng + "/big"})
    assert big.endswith("[page cut at 2 MB]") or len(big) < web.MAX_PAGE_CHARS + 200
    assert session.call("fetch_page", {"url": searxng + "/pdf"}).startswith("Fetch failed")


def test_searxng_errors_are_clear(searxng: str, tmp_path: Path) -> None:
    forbidden = web.SearxngBackend(searxng + "/forbidden", allow_private=True)
    session = web.WebSession(tmp_path, backend=forbidden, permission="allow", cache=False)
    assert "formats: [html, json]" in session.call("web_search", {"query": "x"})
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    down = web.WebSession(tmp_path, backend=web.SearxngBackend(f"http://127.0.0.1:{port}"),
                          permission="allow", cache=False)
    result = down.call("web_search", {"query": "x"})
    assert result.startswith("Web search failed: could not reach SearXNG")


def test_fetch_refuses_local_and_private_addresses(tmp_path: Path) -> None:
    session = web.WebSession(tmp_path, backend=web.SearxngBackend("http://localhost:8888"),
                             permission="allow", cache=False)
    for url in ("http://127.0.0.1:8888/", "http://localhost/admin", "http://192.168.1.1/",
                "http://[::1]/", "http://169.254.169.254/latest/meta-data"):
        assert session.call("fetch_page", {"url": url}).startswith("Fetch failed"), url


# -- caching with provenance -------------------------------------------------------------------

def test_fetched_pages_are_cached_with_provenance(searxng: str, tmp_path: Path) -> None:
    session = web.WebSession(tmp_path, backend=web.SearxngBackend(searxng, allow_private=True),
                             permission="allow")
    session.call("fetch_page", {"url": searxng + "/page"})
    [source] = knowledge.sources()
    assert (source.origin, source.kind, source.title) == (searxng + "/page", "web",
                                                          "Zinkdb 5.0 migration")
    assert source.published == "2026-08-02" and source.trust == "low"
    assert source.added  # the fetch date
    found = knowledge.search_text("connect_timeout_ms")
    assert f"{searxng}/page (published 2026-08-02, fetched {source.added[:10]}, low trust)" in found
    session.call("fetch_page", {"url": searxng + "/page"})
    assert len(knowledge.sources()) == 1  # fetched again: replaced, not duplicated


def test_trust_tiers() -> None:
    assert web.trust_tier("https://docs.python.org/3/library/csv.html") == "official"
    assert web.trust_tier("https://www.postgresql.org/docs/17/") == "official"
    assert web.trust_tier("https://github.com/advisories/GHSA-1") == "official"
    assert web.trust_tier("https://github.com/someone/repo") == "low"
    assert web.trust_tier("https://stackoverflow.com/q/1") == "low"
    assert web.trust_tier("https://python.org.evil.example/") == "low"
    assert web.trust_tier("https://zinkdb.example/docs", ["zinkdb.example"]) == "official"


# -- the agent loop ----------------------------------------------------------------------------

def web_agent(tmp_path: Path, replies: list[FakeMessage], seen: list,
              approve: Any = lambda kind, detail: "session") -> CodingAgent:
    agent = CodingAgent(tmp_path, allow_commands=False, web_research=True)
    agent.web = web.WebSession(tmp_path, backend=StubBackend(), cache=False)
    agent.approve_web = approve
    return script(agent, replies, seen)


def test_agent_offers_web_tools_only_when_available(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []
    plain = script(CodingAgent(tmp_path, allow_commands=False), [text("Hi.")], seen)
    plain.run("hi")
    assert "web_search" not in seen[-1]["tools"]  # web_research=False (unattended default)
    script(CodingAgent(tmp_path, allow_commands=False, web_research=True), [text("Hi.")], seen) \
        .run("hi")
    assert "web_search" not in seen[-1]["tools"]  # no backend configured
    web_agent(tmp_path, [text("Hi.")], seen).run("hi")
    assert {"web_search", "fetch_page"} <= set(seen[-1]["tools"])
    monkeypatch.setenv("VEGA_FEATURES_OFF", "web_research")
    web_agent(tmp_path, [text("Hi.")], seen).run("hi")
    assert "web_search" not in seen[-1]["tools"]  # switch off: no tools at all
    agent = web_agent(tmp_path, [], seen)
    assert agent.web.call("web_search", {"query": "x"}).startswith("Web research unavailable")


def test_denied_web_lets_the_model_continue(tmp_path: Path) -> None:
    seen: list = []
    agent = web_agent(tmp_path, [
        call("web_search", query="widget 2.0 timeout"),
        text("I could not check the web; from memory, widget uses timeout_ms (unverified)."),
    ], seen, approve=lambda kind, detail: "no")
    answer = agent.run("Look up how widget 2.0 sets a timeout.")
    assert "unverified" in answer
    tool_results = [m["content"] for m in seen[-1]["messages"] if m.get("role") == "tool"]
    assert tool_results == [web.DECLINED_TEXT]


def test_answer_without_citation_is_sent_back_once(tmp_path: Path) -> None:
    seen: list = []
    agent = web_agent(tmp_path, [
        call("web_search", query="widget 2.0 timeout"),
        call("fetch_page", url="https://docs.widget.example/2.0"),
        text("Use timeout_ms."),
        text("Use timeout_ms (https://docs.widget.example/2.0)."),
    ], seen)
    answer = agent.run("Look up how widget 2.0 sets a timeout.")
    assert answer == "Use timeout_ms (https://docs.widget.example/2.0)."
    notes = [m["content"] for m in seen[-1]["messages"]
             if m.get("role") == "user" and "cites no URL" in m.get("content", "")]
    assert len(notes) == 1 and "- https://docs.widget.example/2.0" in notes[0]
    # cited without the scheme: accepted as it is
    seen.clear()
    agent = web_agent(tmp_path, [
        call("fetch_page", url="https://docs.widget.example/2.0"),
        text("Use timeout_ms, see docs.widget.example/2.0."),
    ], seen)
    assert agent.run("Look up how widget 2.0 sets a timeout.").endswith("2.0.")
    assert len(seen) == 2


def test_citation_note_can_be_switched_off(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "web_citations")
    seen: list = []
    agent = web_agent(tmp_path, [call("fetch_page", url="https://docs.widget.example/2.0"),
                                 text("Use timeout_ms.")], seen)
    assert agent.run("Look up how widget 2.0 sets a timeout.") == "Use timeout_ms."
    assert features.on("web_research")


def test_web_research_tasks_go_to_the_agent_loop(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "")  # every default on
    (tmp_path / "w.py").write_text("def connect():\n    pass\n")
    seen: list = []
    agent = web_agent(tmp_path, [call("web_search", query="widget 2.0 connect timeout"),
                                 text("Nothing to change (https://docs.widget.example/2.0).")],
                      seen)
    agent.run("Look up the widget 2.0 timeout argument and fix connect in w.py. Cite the page.")
    assert agent.last_route["name"] == "agent"
    assert agent.last_route["reasons"] == ["asks for web research: agent loop (web tools)"]
    assert web.wants_web("Research both against the constraints in docs/")
    assert web.wants_web("Which release fixes CVE-2026-41877?")
    assert not web.wants_web("Implement iso_week_label in cal/weeks.py")
    assert not web.wants_web("Fix the off-by-one in pager.py")


def test_steps_show_web_calls() -> None:
    from vega.ui import step_outcome, step_summary

    assert step_summary("web_search", {"query": "widget  timeout"}) == "widget timeout"
    assert step_summary("fetch_page", {"url": "https://docs.widget.example/2.0"}) == \
        "https://docs.widget.example/2.0"
    found = 'Search results for "w" (3):\n1. a\n   u\n   published x - y'
    assert step_outcome("web_search", found) == ("3 results", "")
    assert step_outcome("web_search", 'No results for "w".') == ("no results", "")
    page = "URL: u\nTitle: Widget docs\nPublished: 2026-05-01\n\nbody"
    assert step_outcome("fetch_page", page) == ("Widget docs (2026-05-01)", "")
    assert step_outcome("fetch_page", web.NOT_FOUND + "u")[1] == "warning"
    assert step_outcome("web_search", web.DECLINED_TEXT)[1] == "warning"


# -- eval integration -------------------------------------------------------------------------


def test_non_web_cases_get_no_web(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings.set_value("web.backend", "searxng")  # even with the user's web enabled
    settings.set_value("web.permission", "allow")
    seen: list = []
    case = {"id": "plain", "task": "Say hi.", "files": {"a.txt": "a"}}

    def factory(project: Path, max_steps: int) -> CodingAgent:
        return script(CodingAgent(project, allow_commands=False, web_research=True),
                      [text("Hi.")], seen)

    attempt = evals.run_attempt(case, agent_factory=factory)
    assert "web_calls" not in attempt
    assert all("web_search" not in item["tools"] for item in seen)


# -- the TUI's permission dialog ---------------------------------------------------------------

def test_tui_asks_once_with_three_choices(tmp_path: Path) -> None:
    from tui_helpers import answer as streamed_answer
    from tui_helpers import calls, drive, make_app, scripted_agent, send, wait_until

    from vega.tui import Steps, WebAccessScreen

    project = tmp_path / "proj"
    project.mkdir()
    (project / "README.md").write_text("# demo\n")
    agent, _ = scripted_agent(project, [
        calls(("web_search", {"query": "widget 2.0 timeout"})),
        calls(("fetch_page", {"url": "https://docs.widget.example/2.0"})),
        streamed_answer("Use timeout_ms (https://docs.widget.example/2.0)."),
    ], web_research=True)
    app = make_app(project, agent, tmp_path)
    agent.web = web.WebSession(project, backend=StubBackend(), cache=False)

    async def run(pilot: Any) -> None:
        await send(pilot, "Look up how widget 2.0 sets a timeout.", wait=False)
        await wait_until(pilot, lambda: isinstance(app.screen, WebAccessScreen))
        screen = app.screen
        assert str(screen.query_one("#dialog-title").render()) == "Allow web research?"
        body = str(screen.query_one("#dialog-body").render())
        assert 'search the web for: "widget 2.0 timeout"' in body
        labels = [str(button.label) for button in screen.query("Button")]
        assert labels == ["Allow this session", "Always for this project", "No"]
        await pilot.press("p")
        await wait_until(pilot, lambda: not app.working)
        await pilot.pause()
        steps = [(e.name, e.outcome) for e in app.query_one(Steps).entries if e.kind == "step"]
        assert steps == [("web_search", "1 results"),
                         ("fetch_page", "Widget 2.0 release notes (2026-05-01)")]

    drive(app, run)
    assert settings.get("web.permission", project) == "allow"  # asked once, stored
    assert app.agent.approve_web == app._approve_web


def test_same_day_last_modified_is_not_a_publication_date() -> None:
    from datetime import datetime, timedelta
    from email.utils import format_datetime

    now = format_datetime(datetime.now(UTC))
    old = format_datetime(datetime.now(UTC) - timedelta(days=40))
    assert web._published("<html></html>", now) == "unknown"
    assert web._published("<html></html>", old) != "unknown"
