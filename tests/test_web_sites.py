"""Official site APIs behind fetch_page (vega/web.py site_adapter, switch `web_site_apis`) and the
blocked-site message. No real network: adapters get recorded-style JSON through a fake `api_get`,
the generic fetch and `_api_get` run against a local HTTP server."""

from __future__ import annotations

import gzip
import http.server
import json
import threading
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any, ClassVar

import pytest

from vega import web
from vega.web import ApiResponse, BlockedError, Page, SearxngBackend, WebSession


@pytest.fixture(autouse=True)
def no_dns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Public hostnames in these tests are never resolved (no network) and the cache stays local."""
    monkeypatch.setattr(web, "_is_private_host", lambda host: host in {"127.0.0.1", "localhost"})
    monkeypatch.setenv("VEGA_KNOWLEDGE_DIR", str(tmp_path / "library"))
    web._SE_STATE.update({"wait_until": {}, "quota": None, "quota_day": ""})


class FakeApi:
    """Recorded replies by URL prefix (longest prefix wins); every request is logged."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str] | None = None) -> ApiResponse:
        self.calls.append((url, dict(headers or {})))
        for prefix in sorted(self.routes, key=len, reverse=True):
            if url.startswith(prefix):
                reply = self.routes[prefix]
                if isinstance(reply, ApiResponse):
                    return reply
                if isinstance(reply, str):
                    return ApiResponse(200, reply.encode(), {})
                return ApiResponse(200, json.dumps(reply).encode(), {})
        return ApiResponse(404, b'{"message": "Not Found"}', {})

    @property
    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]


def fetch(url: str, routes: dict[str, Any]) -> tuple[Page | None, FakeApi]:
    api = FakeApi(routes)
    return SearxngBackend(api_get=api).fetch(url), api


def no_generic_fetch(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make the generic HTML fetch fail loudly (and record that it was tried)."""
    tried: list[str] = []

    class Opener:
        def open(self, request: Any, timeout: float = 0) -> Any:
            tried.append(request.full_url)
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", Message(), None)

    monkeypatch.setattr(web.urllib.request, "build_opener", lambda *handlers: Opener())
    return tried


# -- Stack Exchange ------------------------------------------------------------------------------

SE_QUESTION = {
    "items": [{
        "question_id": 19798153, "title": "Difference between map, applymap and apply &amp; more",
        "tags": ["python", "pandas"], "score": 712, "answer_count": 3, "accepted_answer_id": 111,
        "creation_date": 1383674400, "last_activity_date": 1700000000,
        "body": "<p>Can you tell me <code>when</code> to use these?</p>",
    }],
    "has_more": False, "quota_max": 300, "quota_remaining": 297,
}
SE_ANSWERS = {
    "items": [
        {"answer_id": 222, "score": 900, "is_accepted": False, "creation_date": 1383680000,
         "body": "<p>Top voted answer.</p><pre><code>df.apply(f)\n</code></pre>"},
        {"answer_id": 333, "score": 50, "is_accepted": False, "creation_date": 1383690000,
         "last_edit_date": 1600000000, "body": "<p>Another one.</p>"},
    ],
    "quota_max": 300, "quota_remaining": 296,
}
SE_ACCEPTED = {"items": [{"answer_id": 111, "score": 10, "is_accepted": True,
                          "creation_date": 1383675000, "body": "<p>The accepted one.</p>"}],
               "quota_max": 300, "quota_remaining": 295}


def se_routes() -> dict[str, Any]:
    api = web.STACK_API
    return {f"{api}/questions/19798153?": SE_QUESTION,
            f"{api}/questions/19798153/answers?": SE_ANSWERS,
            f"{api}/answers/111?": SE_ACCEPTED,
            f"{api}/answers/222?": {"items": [{"answer_id": 222, "question_id": 19798153}]}}


def test_stack_overflow_question_through_the_api() -> None:
    url = "https://stackoverflow.com/questions/19798153/difference-between-map-applymap"
    page, api = fetch(url, se_routes())
    assert page is not None
    assert page.url == url
    assert page.title == "Difference between map, applymap and apply & more"
    assert page.date == "2013-11-05"
    assert page.text.startswith("Source: Stack Exchange API (official API)\n")
    assert "Can you tell me when to use these?" in page.text
    assert "tags: python, pandas" in page.text
    assert "Answer 1 (score 900, answered 2013-11-05):" in page.text
    assert "df.apply(f)" in page.text
    assert "edited 2020-09-13" in page.text
    # the accepted answer is not among the top-voted page: fetched and marked
    assert "Answer 3 (accepted, score 10" in page.text
    assert all("site=stackoverflow.com" in u for u in api.urls)
    assert "sort=votes" in api.urls[1] and "filter=withbody" in api.urls[1]
    rendered = web.format_page(page)
    assert rendered.startswith(f"URL: {url}\nTitle: Difference between map")


@pytest.mark.parametrize("url,site", [
    ("https://superuser.com/q/19798153", "superuser.com"),
    ("https://unix.stackexchange.com/questions/19798153", "unix.stackexchange.com"),
    ("https://www.serverfault.com/questions/19798153/x", "serverfault.com"),
    ("https://askubuntu.com/a/222", "askubuntu.com"),
])
def test_stack_exchange_sites_and_answer_links(url: str, site: str) -> None:
    page, api = fetch(url, se_routes())
    assert page is not None and page.title.startswith("Difference between")
    assert all(f"site={site}" in u for u in api.urls)
    if "/a/" in url:
        assert api.urls[0].startswith(f"{web.STACK_API}/answers/222?")


def test_stack_exchange_missing_question_is_not_found() -> None:
    page, _ = fetch("https://stackoverflow.com/questions/1",
                    {f"{web.STACK_API}/questions/1?": {"items": [], "quota_remaining": 290}})
    assert page is None


def test_stack_exchange_quota_and_backoff() -> None:
    url = "https://stackoverflow.com/questions/5"
    throttled = ApiResponse(400, json.dumps({
        "error_id": 502, "error_name": "throttle_violation",
        "error_message": "too many requests from this IP, more requests available in 80000 "
                         "seconds"}).encode(), {})
    with pytest.raises(BlockedError) as caught:
        fetch(url, {f"{web.STACK_API}/questions/5?": throttled})
    message = str(caught.value)
    assert message.startswith(web.FETCH_FAILED)
    assert "quota" in message and "300 requests a day" in message and "/open" in message
    assert caught.value.url == url

    # quota_remaining 0 is remembered: the question still shows (answers not sent), and the
    # next page is not requested at all
    web._SE_STATE.update({"wait_until": {}, "quota": None, "quota_day": ""})
    last = dict(SE_QUESTION, quota_remaining=0)
    page, api = fetch("https://stackoverflow.com/questions/19798153",
                      dict(se_routes(), **{f"{web.STACK_API}/questions/19798153?": last}))
    assert page is not None and "Answers not fetched" in page.text and len(api.urls) == 1
    with pytest.raises(BlockedError, match="quota_remaining is 0"):
        fetch("https://stackoverflow.com/questions/19798153", se_routes())

    # backoff: the API asks for a pause on that method; Vega waits instead of sending
    web._SE_STATE.update({"wait_until": {}, "quota": None, "quota_day": ""})
    backing = dict(SE_QUESTION, backoff=10)
    page, _ = fetch("https://stackoverflow.com/questions/19798153",
                    dict(se_routes(), **{f"{web.STACK_API}/questions/19798153?": backing}))
    assert page is not None and "Answer 1" in page.text  # other methods are not held back
    with pytest.raises(BlockedError, match=r"wait \d+ s"):
        fetch("https://stackoverflow.com/questions/42", se_routes())


# -- GitHub -----------------------------------------------------------------------------------------

GH_ISSUE = {"number": 1, "title": "Enable ix with tuples", "state": "closed",
            "user": {"login": "wesm"}, "labels": [{"name": "Bug"}], "comments": 2,
            "created_at": "2010-09-29T00:45:31Z", "updated_at": "2011-01-01T00:00:00Z",
            "closed_at": "2010-10-01T12:00:00Z", "body": "Should work with **tuples**."}
GH_COMMENTS = [{"user": {"login": "a"}, "created_at": "2010-09-30T00:00:00Z", "body": "Agreed."},
               {"user": {"login": "b"}, "created_at": "2010-10-01T00:00:00Z", "body": "Fixed."}]


def test_github_issue_and_pull_request() -> None:
    base = f"{web.GITHUB_API}/repos/pandas-dev/pandas"
    url = "https://github.com/pandas-dev/pandas/issues/1"
    page, api = fetch(url, {f"{base}/issues/1": GH_ISSUE,
                            f"{base}/issues/1/comments": GH_COMMENTS})
    assert page is not None and page.url == url
    assert page.title == "Enable ix with tuples (issue #1, pandas-dev/pandas)"
    assert page.date == "2010-09-29"
    assert page.text.startswith("Source: GitHub REST API (official API)\n")
    assert "State: closed 2010-10-01; opened by wesm on 2010-09-29" in page.text
    assert "labels: Bug" in page.text
    assert "Comment by b on 2010-10-01:\nFixed." in page.text
    assert api.calls[0][1]["Accept"] == "application/vnd.github+json"

    pull = dict(GH_ISSUE, pull_request={"merged_at": "2010-10-02T00:00:00Z"}, comments=0)
    page, api = fetch("https://github.com/pandas-dev/pandas/pull/1/files",
                      {f"{base}/issues/1": pull, f"{base}/issues/1/comments": []})
    assert page is not None and "Pull request #1" in page.text
    assert "State: merged 2010-10-02" in page.text


def test_github_blob_goes_to_raw_with_last_commit_date() -> None:
    url = "https://github.com/psf/requests/blob/main/src/requests/api.py"
    page, api = fetch(url, {
        f"{web.GITHUB_RAW}/psf/requests/main/src/requests/api.py": "def get(url):\n    ...\n",
        f"{web.GITHUB_API}/repos/psf/requests/commits?":
            [{"commit": {"committer": {"date": "2025-03-04T10:00:00Z"}}}]})
    assert page is not None and page.date == "2025-03-04"
    assert page.title == "src/requests/api.py (psf/requests at main)"
    assert "def get(url):" in page.text
    assert page.text.startswith("Source: GitHub raw file service (official API)")
    assert "path=src%2Frequests%2Fapi.py" in api.urls[1] and "sha=main" in api.urls[1]


def test_github_repo_root_reads_the_readme() -> None:
    base = f"{web.GITHUB_API}/repos/psf/requests"
    page, api = fetch("https://github.com/psf/requests", {
        base: {"description": "A simple HTTP library", "default_branch": "main",
               "pushed_at": "2026-10-01T00:00:00Z", "created_at": "2011-02-13T00:00:00Z",
               "stargazers_count": 52000, "license": {"spdx_id": "Apache-2.0"}},
        f"{base}/readme": "# Requests\n\nHTTP for Humans."})
    assert page is not None and page.date == "2026-10-01"
    assert "README:" in page.text and "HTTP for Humans." in page.text
    assert api.calls[1][1]["Accept"] == "application/vnd.github.raw"


def test_github_releases_list_tag_and_latest() -> None:
    base = f"{web.GITHUB_API}/repos/astral-sh/ruff"
    releases = [{"name": "0.7.0", "tag_name": "0.7.0", "published_at": "2026-10-10T00:00:00Z",
                 "body": "Changes"},
                {"name": "0.7.0rc1", "tag_name": "0.7.0rc1", "prerelease": True,
                 "published_at": "2026-10-01T00:00:00Z", "body": ""}]
    page, api = fetch("https://github.com/astral-sh/ruff/releases",
                      {f"{base}/releases?": releases})
    assert page is not None and page.date == "2026-10-10" and "pre-release" in page.text
    page, api = fetch("https://github.com/astral-sh/ruff/releases/tag/0.7.0",
                      {f"{base}/releases/tags/0.7.0": releases[0]})
    assert page is not None and page.title == "0.7.0 (astral-sh/ruff release)"
    page, api = fetch("https://github.com/astral-sh/ruff/releases/latest",
                      {f"{base}/releases/latest": releases[0]})
    assert api.urls == [f"{base}/releases/latest"]


def test_github_rate_limit_message() -> None:
    limited = ApiResponse(403, b'{"message": "API rate limit exceeded"}',
                          {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1800000000"})
    url = "https://github.com/pandas-dev/pandas/issues/1"
    with pytest.raises(BlockedError) as caught:
        fetch(url, {f"{web.GITHUB_API}/repos/": limited})
    assert "GitHub API rate limit" in str(caught.value)
    assert "60 requests an hour" in str(caught.value) and caught.value.url == url


def test_github_pages_without_an_adapter_use_the_generic_fetch(
        monkeypatch: pytest.MonkeyPatch) -> None:
    tried = no_generic_fetch(monkeypatch)
    for url in ("https://github.com/psf/requests/tree/main/docs",
                "https://github.com/orgs/psf/people"):
        with pytest.raises(BlockedError):
            fetch(url, {})
    assert len(tried) == 2


# -- package registries ----------------------------------------------------------------------------

PYPI = {
    "info": {"name": "requests", "version": "2.32.3", "summary": "Python HTTP for Humans.",
             "requires_python": ">=3.8", "license": "Apache-2.0",
             "project_urls": {"Source": "https://github.com/psf/requests"},
             "description": "# Requests\nlong text"},
    "releases": {
        "2.32.3": [{"upload_time_iso_8601": "2024-05-29T15:37:47Z", "yanked": False}],
        "2.32.0": [{"upload_time_iso_8601": "2024-05-20T15:00:00Z", "yanked": True,
                    "yanked_reason": "Yanked due to conflicts with CVE-2024-35195 mitigation"}],
        "2.31.0": [{"upload_time_iso_8601": "2023-05-22T15:12:44Z", "yanked": False}],
        "0.0.1": [],
    },
    "urls": [{"upload_time_iso_8601": "2024-05-29T15:37:47Z"}],
}


def test_pypi_project_and_version() -> None:
    url = "https://pypi.org/project/requests/"
    page, _ = fetch(url, {"https://pypi.org/pypi/requests/json": PYPI})
    assert page is not None and page.date == "2024-05-29" and page.url == url
    assert page.title == "requests 2.32.3 (PyPI)"
    assert page.text.startswith("Source: PyPI JSON API (official API)")
    assert "Requires Python: >=3.8" in page.text
    assert "2.32.0  uploaded 2024-05-20  YANKED (Yanked due to conflicts" in page.text
    assert page.text.index("2.32.3  uploaded") < page.text.index("2.31.0  uploaded")
    assert "0.0.1" not in page.text  # no files, never installable

    version = {"info": dict(PYPI["info"], version="2.31.0"),
               "urls": [{"upload_time_iso_8601": "2023-05-22T15:12:44Z"}]}
    page, _api = fetch("https://pypi.org/project/requests/2.31.0/",
                      {"https://pypi.org/pypi/requests/2.31.0/json": version})
    assert page is not None and page.date == "2023-05-22" and "Version shown: 2.31.0" in page.text


NPM = {"name": "left-pad", "description": "String left pad", "dist-tags": {"latest": "1.3.0"},
       "license": "WTFPL", "homepage": "https://github.com/stevemao/left-pad#readme",
       "repository": {"url": "git+https://github.com/stevemao/left-pad.git"},
       "time": {"modified": "2022-06-19T00:00:00Z", "created": "2014-03-14T00:00:00Z",
                "1.2.0": "2017-11-14T00:00:00Z", "1.3.0": "2018-04-09T01:09:21Z"},
       "versions": {"1.2.0": {}, "1.3.0": {"deprecated": "use String.prototype.padStart()"}},
       "readme": "# left-pad"}


def test_npm_package_and_scoped_package() -> None:
    page, api = fetch("https://www.npmjs.com/package/left-pad",
                      {"https://registry.npmjs.org/left-pad": NPM})
    assert page is not None and page.date == "2018-04-09"
    assert page.title == "left-pad 1.3.0 (npm)"
    assert "DEPRECATED: use String.prototype.padStart()" in page.text
    assert page.text.startswith("Source: npm registry API (official API)")
    page, api = fetch("https://www.npmjs.com/package/@types/node",
                      {"https://registry.npmjs.org/@types%2fnode": dict(NPM, name="@types/node")})
    assert api.urls == ["https://registry.npmjs.org/@types%2fnode"]


def test_npm_oversized_packument_falls_back_to_the_latest_manifest() -> None:
    big = ApiResponse(200, b'{"name": "typescript", "versions": {', {}, truncated=True)
    page, api = fetch("https://www.npmjs.com/package/typescript", {
        "https://registry.npmjs.org/typescript/latest": {"version": "5.6.3",
                                                         "description": "TypeScript"},
        "https://registry.npmjs.org/typescript": big})
    assert page is not None and "latest manifest only" in page.text
    assert api.urls[-1] == "https://registry.npmjs.org/typescript/latest"


def test_crates_io_crate() -> None:
    crate = {"crate": {"name": "serde", "description": "A serialization framework",
                       "max_stable_version": "1.0.210", "newest_version": "1.0.210",
                       "updated_at": "2024-09-06T00:00:00Z", "created_at": "2014-12-05T00:00:00Z",
                       "repository": "https://github.com/serde-rs/serde", "downloads": 1},
             "versions": [{"num": "1.0.210", "created_at": "2024-09-06T03:00:00Z",
                           "license": "MIT OR Apache-2.0", "rust_version": "1.31"},
                          {"num": "1.0.209", "created_at": "2024-08-24T00:00:00Z",
                           "yanked": True}]}
    page, _api = fetch("https://crates.io/crates/serde", {
        "https://crates.io/api/v1/crates/serde": crate})
    assert page is not None and page.date == "2024-09-06"
    assert "minimum Rust: 1.31" in page.text and "1.0.209  published 2024-08-24  YANKED" in page.text
    assert page.text.startswith("Source: crates.io API (official API)")
    assert "Vega/" in web.USER_AGENT  # crates.io requires a descriptive user agent


# -- Wikipedia and MDN ----------------------------------------------------------------------------------

def test_wikipedia_summary_and_plain_content() -> None:
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    page, api = fetch(url, {
        "https://en.wikipedia.org/api/rest_v1/page/summary/": {
            "title": "Python (programming language)", "description": "General-purpose language",
            "extract": "Python is a high-level language.", "timestamp": "2026-10-12T08:00:00Z"},
        "https://en.wikipedia.org/w/api.php?": {
            "query": {"pages": [{"extract": "Python is a high-level...\n\n== History ==\n..."}]}}})
    assert page is not None and page.date == "2026-10-12"
    assert page.title == "Python (programming language) - Wikipedia"
    assert "== History ==" in page.text
    assert "Python_%28programming_language%29" in api.urls[0]
    assert "explaintext=1" in api.urls[1]


def test_mdn_reads_the_mdn_content_markdown() -> None:
    url = "https://developer.mozilla.org/en-US/docs/Web/API/fetch"
    md = ('---\ntitle: "Window: fetch() method"\nslug: Web/API/Window/fetch\n---\n\n'
          '{{APIRef("Fetch API")}}\n\nThe **`fetch()`** method of the {{domxref("Window")}} '
          'interface starts fetching.\n')
    meta = {"doc": {"title": "Window: fetch() method", "modified": "2025-12-16T10:03:04.000Z",
                    "source": {"folder": "en-us/web/api/window/fetch", "filename": "index.md",
                               "github_url": "https://github.com/mdn/content/blob/main/files/"
                                             "en-us/web/api/window/fetch/index.md"}}}
    page, _ = fetch(url, {
        "https://developer.mozilla.org/en-US/docs/Web/API/fetch/index.json": meta,
        f"{web.GITHUB_RAW}/mdn/content/main/files/en-us/web/api/window/fetch/index.md": md})
    assert page is not None and page.url == url and page.title == "Window: fetch() method - MDN"
    assert page.date == "2025-12-16"
    assert "method of the Window interface" in page.text and "{{" not in page.text
    assert "Fetch API" not in page.text  # sidebar/banner macros are dropped, not inlined
    assert page.text.startswith("Source: MDN document API and mdn/content (official API)")
    assert web._mdn_folder("Web/CSS/::before") == "web/css/_doublecolon_before"

    # no document JSON (an outage): the slug's own folder in the repo
    page, _api = fetch("https://developer.mozilla.org/en-US/docs/Web/API/Window/fetch", {
        "https://developer.mozilla.org/": ApiResponse(503, b"", {}),
        f"{web.GITHUB_RAW}/mdn/content/main/files/en-us/web/api/window/fetch/index.md": md})
    assert page is not None and page.date == "unknown" and "starts fetching" in page.text
    assert page is not None and page.title == "Window: fetch() method - MDN"


def test_mdn_page_missing_from_the_repo_falls_back_to_the_page(
        monkeypatch: pytest.MonkeyPatch) -> None:
    tried = no_generic_fetch(monkeypatch)
    with pytest.raises(BlockedError):
        fetch("https://developer.mozilla.org/en-US/docs/Renamed/Page",
              {"https://developer.mozilla.org/": {"doc": {"title": "x"}}})
    assert tried == ["https://developer.mozilla.org/en-US/docs/Renamed/Page"]


# -- matching, switch, generic fetch ---------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://example.com/questions/1", "https://stackoverflow.com/questions/tagged/python",
    "https://pypi.org/search/?q=x", "https://en.wikipedia.org/w/index.php?title=X",
    "https://developer.mozilla.org/en-US/", "ftp://github.com/a/b",
])
def test_urls_without_an_adapter(url: str) -> None:
    assert web.site_adapter(url) is None


def test_switch_off_uses_the_generic_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEGA_FEATURES_OFF", "web_site_apis")
    tried = no_generic_fetch(monkeypatch)
    url = "https://stackoverflow.com/questions/19798153"
    with pytest.raises(BlockedError):
        _page, _api = fetch(url, se_routes())
    assert tried == [url]


class _Site(http.server.BaseHTTPRequestHandler):
    pages: ClassVar[dict[str, tuple[int, dict[str, str], bytes]]] = {
        "/forbidden": (403, {"Content-Type": "text/html"}, b"<html>Forbidden</html>"),
        "/limited": (429, {"Content-Type": "text/html"}, b"slow down"),
        "/challenge": (200, {"Content-Type": "text/html"},
                       (b"<html><head><title>Just a moment...</title></head><body>"
                       b"<script src='/cdn-cgi/challenge-platform/x.js'></script></body></html>")),
        "/cf": (503, {"Content-Type": "text/html", "cf-mitigated": "challenge"}, b"wait"),
        "/broken": (500, {"Content-Type": "text/html"}, b"oops"),
        "/article": (200, {"Content-Type": "text/html"},
                     b"<html><head><title>On CAPTCHAs</title></head><body><p>"
                     + b"A long article about g-recaptcha widgets. " * 60 + b"</p></body></html>"),
        "/api.json": (200, {"Content-Type": "application/json", "Content-Encoding": "gzip"},
                      gzip.compress(b'{"ok": true}')),
    }

    def do_GET(self) -> None:
        status, headers, body = self.pages.get(self.path, (404, {}, b""))
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def site():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_blocked_sites_get_a_clear_message(site: str) -> None:
    backend = SearxngBackend(allow_private=True)
    for path, why in (("/forbidden", "HTTP 403"), ("/limited", "HTTP 429"),
                      ("/challenge", "CAPTCHA or bot-check"), ("/cf", "bot check")):
        with pytest.raises(BlockedError) as caught:
            backend.fetch(site + path)
        message = str(caught.value)
        assert message.startswith(web.FETCH_FAILED), path
        assert "blocks automated reading" in message and why in message, path
        assert "Try another search result" in message and "open" in message
        assert caught.value.url == site + path
    with pytest.raises(web.BackendError) as caught:
        backend.fetch(site + "/broken")
    assert not isinstance(caught.value, BlockedError)
    page = backend.fetch(site + "/article")  # mentions CAPTCHAs, but is content
    assert page is not None and page.title == "On CAPTCHAs"


def test_web_session_remembers_the_blocked_url_for_open(site: str, tmp_path: Path) -> None:
    session = WebSession(tmp_path, backend=SearxngBackend(allow_private=True),
                         permission="allow", cache=False)
    text = session.call("fetch_page", {"url": site + "/forbidden"})
    assert text.startswith(web.FETCH_FAILED) and "ask the user to open" in text
    assert session.blocked_url == site + "/forbidden"
    from vega.ui import _FAILURE_PREFIXES

    assert text.startswith(_FAILURE_PREFIXES)  # a warning step, and the repeat guard knows it


def test_api_get_rules(site: str) -> None:
    reply = web._api_get(site + "/api.json", allow_private=True)
    assert reply.status == 200 and reply.json("x") == {"ok": True}  # gzip undone
    assert web._api_get(site + "/limited", allow_private=True).status == 429  # not raised
    with pytest.raises(web.BackendError, match="local or private"):
        web._api_get(site + "/api.json")


def test_api_get_is_capped(monkeypatch: pytest.MonkeyPatch, site: str) -> None:
    monkeypatch.setattr(web, "MAX_BYTES", 5)
    reply = web._api_get(site + "/article", allow_private=True)
    assert reply.truncated and len(reply.body) == 5


def test_tui_open_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tui_helpers import drive, make_app, scripted_agent, send

    from vega import tui

    opened: list[str] = []
    monkeypatch.setattr(tui.webbrowser, "open", lambda url: opened.append(url) or True)
    project = tmp_path / "proj"
    project.mkdir()
    agent, _ = scripted_agent(project, [])
    app = make_app(project, agent, tmp_path)

    async def run(pilot: Any) -> None:
        await send(pilot, "/open")  # nothing blocked yet: usage, no browser
        assert opened == []
        app.agent.web.blocked_url = "https://stackoverflow.com/questions/tagged/python"
        await send(pilot, "/open")
        await send(pilot, "/open https://Example.com/Page")
        await send(pilot, "/open file:///etc/passwd")
        await pilot.pause()

    drive(app, run)
    assert opened == ["https://stackoverflow.com/questions/tagged/python",
                      "https://Example.com/Page"]
