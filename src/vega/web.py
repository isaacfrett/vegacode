"""Web research: the `web_search` and `fetch_page` tools, their backends and the permission flow.

User decisions (2026-10-16): web access is OFF by default and asked once per session (allow for
this session / always for this project / no); the backend is a self-hosted SearXNG (its JSON API,
`/search?q=...&format=json`, at the configured `web.searxng_url`); a search sends only the query,
never code, file contents or secrets; every search and fetch is a visible step; fetched pages are
cached in the knowledge library with URL, title, publication and fetch date and a trust tier
(`official` for an official-docs domain, else `low`); answers cite the URLs they relied on.

Settings (vega/settings.py): `web.backend` (none | searxng; default none, so the tools are not
offered at all until a backend is configured), `web.searxng_url` (default http://localhost:8888),
`web.permission` (ask | allow | deny; default ask) and `web.official_domains` (extra domains).
The feature switch `web_research` removes the tools entirely.

Result formats are the eval fake web's contract (format_results, format_page, NOT_FOUND),
so a live run and a fake-web eval show the model exactly the same text.
"""

from __future__ import annotations

import contextlib
import functools
import html
import ipaddress
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import date as _date
from pathlib import Path
from typing import Any, Protocol

from . import features, settings, web_facets

TOOL_NAMES = ("web_search", "fetch_page")
MAX_RESULTS = 5
MAX_PAGE_CHARS = 20_000
MAX_BYTES = 2_000_000
SEARCH_TIMEOUT = 15
FETCH_TIMEOUT = 20
MAX_QUERY_CHARS = 200
USER_AGENT = ("Vega/0.1 (local coding agent; one page fetched for a user's question; "
              "+https://github.com/searxng/searxng)")
NOT_FOUND = "Error: 404 Not Found (no such page in this web): "

# Every refusal or failure starts with one of these (ui.py shows them as warnings; the agent's
# repeat guard treats them as failed calls).
DECLINED = "Web access declined"
UNAVAILABLE = "Web research unavailable"
NOT_SENT = "Search not sent"
FETCH_NOT_SENT = "Fetch not sent"
SEARCH_FAILED = "Web search failed"
FETCH_FAILED = "Fetch failed"
FAILURE_PREFIXES = (DECLINED, UNAVAILABLE, NOT_SENT, FETCH_NOT_SENT, SEARCH_FAILED, FETCH_FAILED,
                    NOT_FOUND)

DECLINED_TEXT = (
    f"{DECLINED}: the user did not allow web research in this session. Do not call web_search or "
    "fetch_page again. Continue without them: use the repository, the standard library and what "
    "you know, and say in your answer which facts you could not check on the web."
)
SETUP_HINT = (
    "Web research needs a local SearXNG search server: start one (see the README section "
    "'Web research') and run `vega web set backend searxng`."
)

# Pages from these domains (and their subdomains) are `official` in the knowledge library; every
# other web page is `low` until another source agrees with it (data-quality rules, 2026-10-14).
OFFICIAL_DOMAINS = (
    "docs.python.org", "peps.python.org", "packaging.python.org", "typing.python.org",
    "pip.pypa.io", "devguide.python.org", "developer.mozilla.org", "docs.github.com",
    "git-scm.com", "nodejs.org", "docs.npmjs.com", "doc.rust-lang.org", "docs.rs", "go.dev",
    "pkg.go.dev", "learn.microsoft.com", "docs.oracle.com", "kubernetes.io", "docs.docker.com",
    "www.postgresql.org", "sqlite.org", "numpy.org", "pandas.pydata.org", "docs.scipy.org",
    "scikit-learn.org", "pytorch.org", "docs.pytest.org", "nvd.nist.gov", "www.cve.org",
    "github.com/advisories", "docs.astral.sh", "www.typescriptlang.org", "react.dev",
)


@dataclass
class Result:
    title: str
    url: str
    snippet: str
    date: str  # YYYY-MM-DD, or "unknown"


@dataclass
class Page:
    url: str
    title: str
    date: str  # publication date YYYY-MM-DD, or "unknown"
    text: str


# -- formats (identical to the eval fake web's) -----------------------------------------------

def format_results(query: str, results: list[Result]) -> str:
    if not results:
        return f'No results for "{query}".'
    lines = [f'Search results for "{query}" ({len(results)}):']
    for number, result in enumerate(results, 1):
        lines += [f"{number}. {result.title}", f"   {result.url}",
                  f"   published {result.date} - {result.snippet}"]
    return "\n".join(lines)


def format_page(page: Page) -> str:
    return f"URL: {page.url}\nTitle: {page.title}\nPublished: {page.date}\n\n{page.text}"


def canonical_url(url: str) -> str:
    """Scheme-insensitive, host lower-cased, no fragment, no trailing slash, no 'www.'
    (the eval fake web's rule, so a citation matches however the model writes the URL)."""
    text = url.strip().strip("<>\"'")
    text = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", text).split("#", 1)[0]
    host, slash, rest = text.partition("/")
    host = host.lower().removeprefix("www.")
    return ("https://" + host + slash + rest).rstrip("/")


_FILE_EXTENSIONS = {"py", "md", "txt", "json", "toml", "cfg", "ini", "yaml", "yml", "rst", "csv",
                    "lock", "js", "ts", "go", "rs", "java", "rb", "sh", "sql", "html", "xml", "in"}


def is_web_url(text: str) -> bool:
    """A web page address (`https://host/...`, `host.tld/...`), not a repo path
    (`docs/PLATFORM.md`, `requirements.txt`) or a local server."""
    text = text.strip().strip("<>\"'")
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        try:
            host = urllib.parse.urlparse(text).hostname or ""
        except ValueError:
            return False
    else:
        host = text.split("/", 1)[0].lower()
        if not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", host) or \
                host.rsplit(".", 1)[1] in _FILE_EXTENSIONS:
            return False
    host = host.lower()
    if "." not in host or host.endswith("localhost"):
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return True
    return False


def json_sources(project: Path, paths: list[str]) -> list[str]:
    """The web URLs in the top-level `sources` list of each JSON file in `paths` (files written
    this turn); repo paths and non-strings are left out."""
    found: list[str] = []
    for relative in dict.fromkeys(paths):
        if not relative.endswith(".json"):
            continue
        try:
            data = json.loads((Path(project) / relative).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        listed = data.get("sources") if isinstance(data, dict) else None
        if isinstance(listed, list):
            found += [item for item in listed if isinstance(item, str) and is_web_url(item)]
    return found


# web_official_hint: words of a query that name no product, so a host holding one of them is not
# thereby the product's own site
_GENERIC_WORDS = frozenset((
    "about", "after", "again", "also", "android", "answer", "answers", "apple", "best", "blog",
    "breaking", "change", "changed", "changes", "changelog", "check", "client", "code",
    "compatible", "compatibility", "config", "configuration", "connect", "connection", "default",
    "deprecated", "deprecation", "difference", "docs", "documentation", "download", "driver",
    "error", "errors", "example", "examples", "feature", "features", "field", "file", "files",
    "find", "fixed", "fixes", "format", "forum", "free", "from", "guide", "help", "history", "how",
    "install", "installation", "issue", "issues", "java", "javascript", "latest", "library",
    "limit", "limits", "linux", "list", "maximum", "migrate", "migrating", "migration", "minimum",
    "news", "node", "official", "option", "options", "package", "page", "pages", "plugin", "price",
    "prices", "pricing", "python", "question", "questions", "rate", "reference", "release",
    "released", "releases", "removal", "removed", "request", "requests", "require", "required",
    "requirement", "requirements", "schema", "security", "server", "service", "setting",
    "settings", "should", "signature", "site", "size", "stack", "status", "still", "support",
    "supported", "supports", "their", "there", "tier", "tiers", "timeout", "tutorial", "update",
    "updates", "upgrade", "upgrading", "using", "version", "versions", "what", "when", "which",
    "windows", "with", "without", "works",
))


def official_urls(query: str, urls: list[str], extra: list[str] | tuple[str, ...] = ()) -> set[str]:
    """web_official_hint: the result URLs that look like the queried product's own site (a host
    label, TLD aside, holds a distinctive word of the query: `httpkit` ->
    advisories.httpkit.example) or are on a known official-docs host (trust_tier)."""
    words = {word for word in re.findall(r"[a-z][a-z0-9]{3,}", query.lower())
             if word not in _GENERIC_WORDS}
    found = set()
    for url in urls:
        try:
            host = urllib.parse.urlparse(url if "://" in url else "https://" + url).hostname or ""
        except ValueError:
            continue
        labels = host.lower().removeprefix("www.").split(".")[:-1]
        if trust_tier(url, extra) == "official" or any(
                word in label for word in words for label in labels):
            found.add(url)
    return found


def mark_official(text: str, official: set[str]) -> str:
    """format_results text with "(official site)" after the URL line of each official result."""
    if not official:
        return text
    lines = text.split("\n")
    for number, line in enumerate(lines):
        if line.startswith("   ") and line.strip() in official:
            lines[number] = line + "  (official site)"
    return "\n".join(lines)


def _cited(answer: str) -> set[str]:
    """The URLs an answer names (with or without a scheme), canonical."""
    cited = {canonical_url(match) for match in re.findall(r"https?://[^\s)\]>\"'`,]+", answer)}
    cited |= {canonical_url(match) for match in
              re.findall(r"\b(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+/[^\s)\]>\"'`,]*", answer)}
    return {url.rstrip(".") for url in cited}


# -- secrets and code never leave the machine --------------------------------------------------

SECRET_PATTERNS = (
    ("an AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("a GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("an API key (sk-...)", re.compile(r"\bsk-(?:[A-Za-z0-9_-]{2,}-)?[A-Za-z0-9_-]{20,}")),
    ("a Stripe key", re.compile(r"\b[rsp]k_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("a Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("a Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("a JSON web token", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("credentials in a URL", re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^/\s:@]+:[^/\s@]+@")),
    ("a password or key assignment", re.compile(
        r"(?i)\b(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key)"
        r"\s*[=:]\s*['\"]?(?=[^\s'\"]*[\d_\-+/])[^\s'\"]{6,}")),
    # a long random-looking token: letters and digits mixed, not a plain hex hash
    ("a long random token", re.compile(
        r"\b(?=[A-Za-z0-9_\-+/]*[a-z])(?=[A-Za-z0-9_\-+/]*[A-Z])(?=[A-Za-z0-9_\-+/]*\d)"
        r"[A-Za-z0-9_\-+/]{32,}")),
)
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*[=:]\s*['\"]?([^'\"\s#]+)")


def project_secret_values(project: Path | None) -> list[str]:
    """Values (8+ characters) assigned in the project's secret files (.env, *.key, credentials...
    by vega/gitignore.py's patterns), at most two folders deep, so a query cannot carry them."""
    if project is None:
        return []
    from .gitignore import is_sensitive

    values: list[str] = []
    root = Path(project)
    try:
        candidates = [path for pattern in ("*", "*/*") for path in root.glob(pattern)
                      if path.is_file() and is_sensitive(path.name)]
    except OSError:
        return []
    for path in candidates[:50]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:100_000]
        except OSError:
            continue
        for line in text.splitlines():
            match = _ENV_LINE.match(line)
            if match and len(match.group(1)) >= 8:
                values.append(match.group(1))
    return values


def secret_in(text: str, project: Path | None = None) -> str | None:
    """What kind of secret `text` holds, or None."""
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return label
    if any(value in text for value in project_secret_values(project)):
        return "a value from the project's secret files"
    return None


_CODE_SIGNS = re.compile(r"[{};]|^\s*(?:def|class|import|from|return)\b|=>|\)\s*:|==")


def check_query(query: str, project: Path | None = None) -> str | None:
    """A refusal when the query must not be sent (a secret, code, too long), else None."""
    secret = secret_in(query, project)
    if secret:
        return (f"{NOT_SENT}: the query looks like it contains {secret}, and searches never "
                "send secrets. Search again with a short question in plain words, without keys, "
                "tokens, passwords or values from the project.")
    if "\n" in query.strip() or len(query) > MAX_QUERY_CHARS:
        return (f"{NOT_SENT}: a search is one short line in plain words (at most "
                f"{MAX_QUERY_CHARS} characters); never paste code, logs or file contents into it. "
                "Name the library, version and error code instead.")
    if len(_CODE_SIGNS.findall(query)) >= 2:
        return (f"{NOT_SENT}: the query looks like code, and searches never send code. Describe "
                "what you need in words (library, function name, error message).")
    if not query.strip():
        return f"{NOT_SENT}: give a short question or keywords to search for."
    return None


def check_url(url: str, project: Path | None = None) -> str | None:
    if not re.match(r"https?://[^\s/]+", url.strip()):
        return f"{FETCH_NOT_SENT}: {url!r} is not an http(s) URL. Fetch a URL from search results."
    secret = secret_in(url, project)
    if secret:
        return f"{FETCH_NOT_SENT}: the URL looks like it contains {secret}."
    return None


# -- trust -------------------------------------------------------------------------------------

def trust_tier(url: str, extra: list[str] | tuple[str, ...] = ()) -> str:
    """`official` for a page on an official-docs domain (or under an official path such as
    github.com/advisories), else `low`."""
    parsed = urllib.parse.urlparse(url if "://" in url else "https://" + url)
    host = (parsed.hostname or "").lower()
    bare = host.removeprefix("www.")
    path = parsed.path.rstrip("/")
    for entry in (*OFFICIAL_DOMAINS, *extra):
        domain, _, prefix = entry.lower().partition("/")
        domain = domain.removeprefix("www.")
        if (bare == domain or bare.endswith("." + domain)) and (
                not prefix or path == "/" + prefix or path.startswith("/" + prefix + "/")):
            return "official"
    return "low"


# -- backends ----------------------------------------------------------------------------------

class Backend(Protocol):
    name: str

    def search(self, query: str, n: int = MAX_RESULTS) -> list[Result]: ...

    def fetch(self, url: str) -> Page | None: ...


class BackendError(Exception):
    """The backend could not answer (unreachable, refused, bad reply); the message says why."""


class NoBackend:
    """No search backend configured: every call explains how to enable one."""

    name = "none"

    def search(self, query: str, n: int = MAX_RESULTS) -> list[Result]:
        raise BackendError(f"{UNAVAILABLE}: no search backend is configured. {SETUP_HINT}")

    def fetch(self, url: str) -> Page | None:
        raise BackendError(f"{UNAVAILABLE}: no search backend is configured. {SETUP_HINT}")


def _is_private_host(host: str) -> bool:
    """True for loopback, private, link-local and unresolvable-to-public hosts."""
    host = host.strip("[]").lower()
    if not host or host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return True
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return False  # unresolvable: the fetch itself will fail with a clear error
        addresses = []
        for info in infos:
            try:
                addresses.append(ipaddress.ip_address(info[4][0]))
            except ValueError:
                continue
    return any(a.is_private or a.is_loopback or a.is_link_local or a.is_reserved
               or a.is_unspecified or a.is_multicast for a in addresses)


class _NoPrivateRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_private: bool) -> None:
        self.allow_private = allow_private

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = urllib.parse.urlparse(newurl).hostname or ""
        if not self.allow_private and _is_private_host(host):
            raise BackendError(f"{FETCH_FAILED}: {req.full_url} redirects to a local or private "
                               "address, which Vega does not fetch.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_META_DATE = re.compile(
    r"<meta[^>]+(?:property|name|itemprop)=[\"'](?:article:published_time|og:published_time|"
    r"datePublished|date|dc\.date|DC\.date\.issued|publish-date|pubdate)[\"'][^>]*>",
    re.IGNORECASE)
_CONTENT = re.compile(r"content=[\"']([^\"']+)[\"']", re.IGNORECASE)
_ISO_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _published(html: str, last_modified: str | None) -> str:
    for tag in _META_DATE.findall(html[:200_000]):
        content = _CONTENT.search(tag)
        date = _ISO_DATE.search(content.group(1)) if content else None
        if date:
            return date.group(1)
    if last_modified:
        from email.utils import parsedate_to_datetime

        try:
            modified = parsedate_to_datetime(last_modified)
        except (TypeError, ValueError):
            modified = None
        # Servers that build pages on request send the request time as Last-Modified: that is
        # not a publication date, and calling it one makes any page look current.
        if modified is not None and modified.date() < _date.today():
            return modified.strftime("%Y-%m-%d")
    return "unknown"


def _date_of(value: Any) -> str:
    match = _ISO_DATE.search(str(value or ""))
    return match.group(1) if match else "unknown"


class SearxngBackend:
    """A self-hosted SearXNG instance's JSON API. Only the query goes to SearXNG; pages are
    fetched directly with a polite user agent, a timeout and a size cap."""

    name = "searxng"

    def __init__(self, base_url: str = "http://localhost:8888", *, allow_private: bool = False,
                 timeout: float = SEARCH_TIMEOUT, fetch_timeout: float = FETCH_TIMEOUT,
                 api_get: ApiGet | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.allow_private = allow_private  # fetching local/private addresses (tests only)
        self.timeout = timeout
        self.fetch_timeout = fetch_timeout
        # official site APIs (site_adapter); tests pass recorded replies
        self.api_get: ApiGet = api_get or functools.partial(
            _api_get, allow_private=allow_private, timeout=fetch_timeout)

    def search(self, query: str, n: int = MAX_RESULTS) -> list[Result]:
        url = f"{self.base_url}/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                       "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read(MAX_BYTES + 1)
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                raise BackendError(
                    f"{SEARCH_FAILED}: SearXNG at {self.base_url} refused the JSON API (HTTP 403). "
                    "Enable it in SearXNG's settings.yml: search: formats: [html, json]."
                ) from exc
            raise BackendError(f"{SEARCH_FAILED}: SearXNG at {self.base_url} answered HTTP "
                               f"{exc.code}.") from exc
        except (OSError, ValueError) as exc:
            reason = getattr(exc, "reason", exc)
            raise BackendError(
                f"{SEARCH_FAILED}: could not reach SearXNG at {self.base_url} ({reason}). Is it "
                "running? Continue without web results if it is not."
            ) from exc
        try:
            data = json.loads(body[:MAX_BYTES].decode("utf-8", errors="replace"))
        except ValueError as exc:
            raise BackendError(f"{SEARCH_FAILED}: SearXNG at {self.base_url} did not return "
                               "JSON.") from exc
        results: list[Result] = []
        seen: set[str] = set()
        for item in data.get("results", []) if isinstance(data, dict) else []:
            if not isinstance(item, dict) or not str(item.get("url", "")).startswith("http"):
                continue
            key = canonical_url(item["url"])
            if key in seen:
                continue
            seen.add(key)
            snippet = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
            results.append(Result(
                title=re.sub(r"\s+", " ", str(item.get("title") or item["url"])).strip(),
                url=item["url"], snippet=snippet[:300],
                date=_date_of(item.get("publishedDate") or item.get("pubdate"))))
            if len(results) >= n:
                break
        return results

    def fetch(self, url: str) -> Page | None:
        host = urllib.parse.urlparse(url).hostname or ""
        if not self.allow_private and _is_private_host(host):
            raise BackendError(f"{FETCH_FAILED}: {url} is a local or private address, which Vega "
                               "does not fetch.")
        adapter = site_adapter(url) if features.on("web_site_apis") else None
        if adapter is not None:
            page = adapter(self.api_get)
            if page is not GENERIC:
                return page
        opener = urllib.request.build_opener(_NoPrivateRedirects(self.allow_private))
        request = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT, "Accept": "text/html,text/plain,text/markdown;q=0.9,*/*;q=0.1"})
        try:
            with opener.open(request, timeout=self.fetch_timeout) as response:
                body = response.read(MAX_BYTES + 1)
                kind = response.headers.get("Content-Type", "")
                last_modified = response.headers.get("Last-Modified")
                final_url = response.geturl() or url
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                return None
            if exc.code in (401, 403, 429) or (exc.code == 503 and _challenge_reply(exc)):
                why = {401: "HTTP 401", 403: "HTTP 403",
                       429: "HTTP 429, too many requests"}.get(exc.code, "a bot check, HTTP 503")
                raise BlockedError(blocked_text(url, why), url) from exc
            raise BackendError(f"{FETCH_FAILED}: HTTP {exc.code} for {url}.") from exc
        except BackendError:
            raise
        except (OSError, ValueError) as exc:
            raise BackendError(f"{FETCH_FAILED}: could not fetch {url} "
                               f"({getattr(exc, 'reason', exc)}).") from exc
        truncated = len(body) > MAX_BYTES
        body = body[:MAX_BYTES]
        lowered = kind.lower()
        if "pdf" in lowered:
            raise BackendError(f"{FETCH_FAILED}: {url} is a PDF; ask the user to /learn it.")
        if lowered and not any(word in lowered for word in ("html", "text", "json", "xml",
                                                            "markdown")):
            raise BackendError(f"{FETCH_FAILED}: {url} is {kind.split(';')[0]}, not a text page.")
        charset = re.search(r"charset=([\w-]+)", kind)
        try:
            raw = body.decode(charset.group(1) if charset else "utf-8", errors="replace")
        except LookupError:
            raw = body.decode("utf-8", errors="replace")
        from .knowledge import html_to_text

        if "html" in lowered or raw.lstrip()[:200].lower().startswith(("<!doctype html", "<html")):
            title, text = html_to_text(raw)
            if looks_like_challenge(raw, len(text)):
                raise BlockedError(blocked_text(url, "a CAPTCHA or bot-check page"), url)
            date = _published(raw, last_modified)
        else:
            title, text, date = "", raw.strip(), _published("", last_modified)
        text = text[:MAX_PAGE_CHARS]
        if truncated:
            text += f"\n\n[page cut at {MAX_BYTES // 1_000_000} MB]"
        return Page(url=final_url, title=title or final_url, date=date, text=text)



# -- official site APIs ------------------------------------------------------------------------
# Since 2026-10-17: sites that refuse non-browser clients (stackoverflow.com answers
# 403 to any honest user agent) are read through their OFFICIAL API instead. Vega never
# impersonates a browser and never works around a CAPTCHA or bot check; for a blocked site with
# no API adapter the tool says so and the user can open the page in their own browser (/open).
# Each adapter returns the same Page (url = the URL asked for) so the tool output is unchanged,
# with "Source: <API> (official API)" as the text's first line. Switch: `web_site_apis`.

GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"
STACK_API = "https://api.stackexchange.com/2.3"
LIST_LIMIT = 25  # versions / releases listed per page
COMMENT_LIMIT = 30


class BlockedError(BackendError):
    """The site refused automated reading (403/429/CAPTCHA) or its API's quota is used up; `url`
    is the page the user can open in their own browser."""

    def __init__(self, message: str, url: str) -> None:
        super().__init__(message)
        self.url = url


@dataclass
class ApiResponse:
    status: int
    body: bytes
    headers: dict[str, str]  # lower-cased names
    truncated: bool = False

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self, what: str) -> Any:
        try:
            return json.loads(self.text())
        except ValueError as exc:
            raise BackendError(f"{FETCH_FAILED}: {what} did not return JSON.") from exc


ApiGet = Callable[..., ApiResponse]


def _decompress(body: bytes, encoding: str) -> tuple[bytes, bool]:
    if "gzip" not in encoding.lower() and not body.startswith(b"\x1f\x8b"):
        return body, False
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        data = inflater.decompress(body, MAX_BYTES + 1)
    except zlib.error as exc:
        raise BackendError(f"{FETCH_FAILED}: a compressed reply could not be read.") from exc
    return data, len(data) > MAX_BYTES


def _api_get(url: str, headers: dict[str, str] | None = None, *, allow_private: bool = False,
             timeout: float = FETCH_TIMEOUT) -> ApiResponse:
    """GET one API URL with the same rules as a page fetch (no local/private hosts or redirects
    to them, timeout, 2 MB cap). HTTP errors come back as a status, network errors raise."""
    host = urllib.parse.urlparse(url).hostname or ""
    if not allow_private and _is_private_host(host):
        raise BackendError(f"{FETCH_FAILED}: {url} is a local or private address, which Vega "
                           "does not fetch.")
    opener = urllib.request.build_opener(_NoPrivateRedirects(allow_private))
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "application/json", "Accept-Encoding": "gzip",
        **(headers or {})})
    try:
        with opener.open(request, timeout=timeout) as response:
            status, raw = response.status, response.read(MAX_BYTES + 1)
            found = {key.lower(): value for key, value in response.headers.items()}
    except urllib.error.HTTPError as exc:
        status, found = exc.code, {key.lower(): value for key, value in (exc.headers or {}).items()}
        try:
            raw = exc.read(MAX_BYTES + 1)
        except OSError:
            raw = b""
    except BackendError:
        raise
    except (OSError, ValueError) as exc:
        raise BackendError(f"{FETCH_FAILED}: could not reach {host} "
                           f"({getattr(exc, 'reason', exc)}).") from exc
    truncated = len(raw) > MAX_BYTES
    body, inflated_over = _decompress(raw[:MAX_BYTES], found.get("content-encoding", ""))
    return ApiResponse(status, body[:MAX_BYTES], found, truncated or inflated_over)


GENERIC = object()  # an adapter's answer for "read this one as an ordinary page"


def _day(value: Any) -> str:
    """YYYY-MM-DD of an ISO timestamp or a Unix time, else "unknown"."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return datetime.fromtimestamp(value, tz=UTC).strftime("%Y-%m-%d")
    return _date_of(value)


def _source_page(url: str, title: str, date: str, source: str, body: str) -> Page:
    text = f"Source: {source} (official API)\n\n{body.strip()}"
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n\n[cut at 20,000 characters]"
    return Page(url=url, title=title or url, date=date or "unknown", text=text)


def _html_text(fragment: str) -> str:
    from .knowledge import html_to_text

    return html_to_text(fragment or "")[1]


def _unexpected(api: str, response: ApiResponse, url: str) -> BackendError:
    return BackendError(f"{FETCH_FAILED}: the {api} answered HTTP {response.status} for {url}.")


# Stack Exchange --------------------------------------------------------------------------------

_SE_HOSTS = ("stackoverflow.com", "superuser.com", "serverfault.com", "askubuntu.com",
             "mathoverflow.net", "stackapps.com")
# backoff is per API method ("wait N s before hitting the same method again"); the quota is per day
_SE_STATE: dict[str, Any] = {"wait_until": {}, "quota": None, "quota_day": ""}


def _se_site(host: str) -> str | None:
    host = host.lower().removeprefix("www.")
    if host in _SE_HOSTS or host.endswith(".stackexchange.com"):
        return host
    if host.startswith("meta.") and host[5:] in _SE_HOSTS:
        return host
    return None


def _se_call(get: ApiGet, path: str, params: dict[str, str], url: str) -> list[dict]:
    now = time.monotonic()
    method = re.sub(r"\d+", "{id}", path)
    if _SE_STATE["wait_until"].get(method, 0.0) > now:
        seconds = int(_SE_STATE["wait_until"][method] - now) + 1
        raise BlockedError(
            f"{FETCH_FAILED}: the Stack Exchange API asked Vega to wait {seconds} s before its "
            f"next request (backoff), so {url} was not fetched. Try another search result, or "
            "fetch it again later.", url)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    if _SE_STATE["quota"] == 0 and _SE_STATE["quota_day"] == today:
        raise BlockedError(_se_quota_text(url, "quota_remaining is 0"), url)
    response = get(f"{STACK_API}{path}?" + urllib.parse.urlencode(params))
    data = response.json("the Stack Exchange API") if response.body else {}
    if not isinstance(data, dict):
        raise _unexpected("Stack Exchange API", response, url)
    if isinstance(data.get("backoff"), (int, float)):
        _SE_STATE["wait_until"][method] = time.monotonic() + float(data["backoff"])
    if isinstance(data.get("quota_remaining"), int):
        _SE_STATE["quota"], _SE_STATE["quota_day"] = data["quota_remaining"], today
    if response.status != 200 or "error_id" in data:
        message = str(data.get("error_message") or f"HTTP {response.status}")
        if data.get("error_id") == 502 or data.get("error_name") == "throttle_violation" \
                or response.status == 429:
            raise BlockedError(_se_quota_text(url, message), url)
        raise BackendError(f"{FETCH_FAILED}: the Stack Exchange API refused {url} ({message}).")
    items = data.get("items")
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _se_quota_text(url: str, detail: str) -> str:
    return (f"{FETCH_FAILED}: the Stack Exchange API quota for this machine is used up "
            f"({detail}); without an API key it allows 300 requests a day per address. Try "
            f"another search result, or ask the user to open {url} in their browser (in Vega's "
            "app: /open) and tell you what it says.")


def _stack_exchange(url: str, site: str, question_id: str | None, answer_id: str | None,
                    get: ApiGet) -> Page | None:
    if answer_id:
        answers = _se_call(get, f"/answers/{answer_id}", {"site": site}, url)
        if not answers:
            return None
        question_id = str(answers[0].get("question_id", ""))
    questions = _se_call(get, f"/questions/{question_id}", {"site": site, "filter": "withbody"},
                         url)
    if not questions:
        return None
    question = questions[0]
    missing = ""  # the question arrived but the quota or a backoff stopped the answers
    try:
        answers = _se_call(get, f"/questions/{question_id}/answers",
                           {"site": site, "filter": "withbody", "sort": "votes", "order": "desc",
                            "pagesize": "5"}, url)
    except BlockedError:
        answers, missing = [], "Answers not fetched: the Stack Exchange API quota or backoff."
    accepted = question.get("accepted_answer_id")
    if accepted and not missing and all(a.get("answer_id") != accepted for a in answers):
        with contextlib.suppress(BlockedError):
            answers += _se_call(get, f"/answers/{accepted}",
                                {"site": site, "filter": "withbody"}, url)
    title = html.unescape(str(question.get("title", "")))
    tags = ", ".join(str(tag) for tag in question.get("tags", []))
    lines = [f"Question: {title}",
             f"Score {question.get('score', 0)}, asked {_day(question.get('creation_date'))}, "
             f"last activity {_day(question.get('last_activity_date'))}; tags: {tags or 'none'}; "
             f"{question.get('answer_count', len(answers))} answers"
             + ("; closed" if question.get("closed_date") else ""),
             "", _html_text(str(question.get("body", ""))), ""]
    if missing:
        lines.append(missing)
    elif not answers:
        lines.append("No answers yet.")
    for number, answer in enumerate(answers, 1):
        marks = ["accepted"] if answer.get("is_accepted") or answer.get("answer_id") == accepted \
            else []
        marks.append(f"score {answer.get('score', 0)}")
        marks.append(f"answered {_day(answer.get('creation_date'))}")
        if answer.get("last_edit_date"):
            marks.append(f"edited {_day(answer.get('last_edit_date'))}")
        lines += [f"Answer {number} ({', '.join(marks)}):",
                  _html_text(str(answer.get("body", ""))), ""]
    return _source_page(url, title, _day(question.get("creation_date")),
                        "Stack Exchange API", "\n".join(lines))


# GitHub ------------------------------------------------------------------------------------------

_GH_RESERVED = {"orgs", "settings", "features", "topics", "about", "marketplace", "sponsors",
                "login", "join", "explore", "search", "notifications", "pricing", "collections",
                "trending", "enterprise", "security", "site", "contact", "apps", "advisories"}
_GH_HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}


def _gh_get(get: ApiGet, path: str, url: str, accept: str | None = None) -> ApiResponse:
    headers = dict(_GH_HEADERS)
    if accept:
        headers["Accept"] = accept
    response = get(f"{GITHUB_API}{path}", headers)
    if response.status in (403, 429):
        remaining = response.headers.get("x-ratelimit-remaining")
        detail = response.text()[:300].lower()
        if remaining == "0" or response.status == 429 or "rate limit" in detail:
            reset = response.headers.get("x-ratelimit-reset", "")
            when = ""
            if reset.isdigit():
                when = " until " + time.strftime("%H:%M", time.localtime(int(reset)))
            raise BlockedError(
                f"{FETCH_FAILED}: the GitHub API rate limit for this machine is used up{when} "
                "(60 requests an hour without a token). Try another search result, or ask the "
                f"user to open {url} in their browser (in Vega's app: /open) and tell you what it "
                "says.", url)
    return response


def _gh_json(get: ApiGet, path: str, url: str) -> Any:
    response = _gh_get(get, path, url)
    if response.status == 404:
        return None
    if response.status != 200:
        raise _unexpected("GitHub API", response, url)
    return response.json("the GitHub API")


def _github(url: str, owner: str, repo: str, rest: str, get: ApiGet) -> Any:
    base = f"/repos/{owner}/{repo}"
    name = f"{owner}/{repo}"
    issue = re.match(r"^(issues|pull)/(\d+)(?:/.*)?$", rest)
    if issue:
        return _github_issue(url, base, name, issue.group(2), get)
    blob = re.match(r"^blob/([^/]+)/(.+)$", rest)
    if blob:
        return _github_blob(url, owner, repo, blob.group(1), blob.group(2), get)
    if rest == "":
        return _github_repo(url, base, name, get)
    release = re.match(r"^releases(?:/(latest|tag/(.+)))?$", rest)
    if release:
        return _github_releases(url, base, name, release.group(1), release.group(2), get)
    return GENERIC


def _github_issue(url: str, base: str, name: str, number: str, get: ApiGet) -> Page | None:
    item = _gh_json(get, f"{base}/issues/{number}", url)
    if not isinstance(item, dict):
        return None
    comments = _gh_json(get, f"{base}/issues/{number}/comments?per_page={COMMENT_LIMIT}", url)
    pull = item.get("pull_request")
    kind = "Pull request" if pull else "Issue"
    state = str(item.get("state", "unknown"))
    if isinstance(pull, dict) and pull.get("merged_at"):
        state = f"merged {_day(pull['merged_at'])}"
    elif item.get("closed_at"):
        state += f" {_day(item['closed_at'])}"
    labels = ", ".join(str(label.get("name")) for label in item.get("labels", [])
                       if isinstance(label, dict))
    author = (item.get("user") or {}).get("login", "unknown")
    lines = [f"{kind} #{number} in {name}: {item.get('title', '')}",
             (f"State: {state}; opened by {author} on {_day(item.get('created_at'))}; "
             f"updated {_day(item.get('updated_at'))}; labels: {labels or 'none'}; "
             f"{item.get('comments', 0)} comments"), "",
             str(item.get("body") or "(no description)"),
             ""]
    shown = comments if isinstance(comments, list) else []
    for comment in shown:
        who = (comment.get("user") or {}).get("login", "unknown")
        lines += [f"Comment by {who} on {_day(comment.get('created_at'))}:",
                  str(comment.get("body") or ""), ""]
    if int(item.get("comments") or 0) > len(shown):
        lines.append(f"[first {len(shown)} of {item.get('comments')} comments shown]")
    return _source_page(url, f"{item.get('title', '')} ({kind.lower()} #{number}, {name})",
                        _day(item.get("created_at")), "GitHub REST API", "\n".join(lines))


def _github_blob(url: str, owner: str, repo: str, ref: str, path: str, get: ApiGet) -> Page | None:
    quoted = urllib.parse.quote(path)
    response = get(f"{GITHUB_RAW}/{owner}/{repo}/{urllib.parse.quote(ref)}/{quoted}",
                   {"Accept": "text/plain, */*"})
    if response.status == 404:
        return None
    if response.status != 200:
        raise _unexpected("GitHub raw file service", response, url)
    if b"\x00" in response.body[:8000]:
        raise BackendError(f"{FETCH_FAILED}: {url} is a binary file, not a text page.")
    date = "unknown"
    with contextlib.suppress(BackendError):  # the date is a bonus; the file is the answer
        commits = _gh_json(get, f"/repos/{owner}/{repo}/commits?" + urllib.parse.urlencode(
            {"path": path, "sha": ref, "per_page": "1"}), url)
        if isinstance(commits, list) and commits:
            date = _day(((commits[0].get("commit") or {}).get("committer") or {}).get("date"))
    text = response.text()
    if response.truncated:
        text += "\n\n[file cut at 2 MB]"
    body = (f"File {path} at {ref} in {owner}/{repo} (last commit {date})\n\n{text}")
    return _source_page(url, f"{path} ({owner}/{repo} at {ref})", date,
                        "GitHub raw file service", body)


def _github_repo(url: str, base: str, name: str, get: ApiGet) -> Page | None:
    info = _gh_json(get, base, url)
    if not isinstance(info, dict):
        return None
    readme = _gh_get(get, f"{base}/readme", url, accept="application/vnd.github.raw")
    lines = [f"Repository {name}: {info.get('description') or '(no description)'}",
             f"Default branch {info.get('default_branch', '?')}; last push "
             f"{_day(info.get('pushed_at'))}; created {_day(info.get('created_at'))}; "
             f"{info.get('stargazers_count', 0)} stars; license "
             f"{(info.get('license') or {}).get('spdx_id', 'none')}"
             + ("; ARCHIVED" if info.get("archived") else ""), ""]
    if info.get("homepage"):
        lines.insert(2, f"Homepage: {info['homepage']}")
    if readme.status == 200:
        lines += ["README:", "", readme.text()]
    else:
        lines.append("No README.")
    return _source_page(url, f"{name}: {info.get('description') or 'GitHub repository'}",
                        _day(info.get("pushed_at")), "GitHub REST API", "\n".join(lines))


def _github_releases(url: str, base: str, name: str, which: str | None, tag: str | None,
                     get: ApiGet) -> Page | None:
    if tag:
        found = _gh_json(get, f"{base}/releases/tags/{urllib.parse.quote(tag)}", url)
    elif which == "latest":
        found = _gh_json(get, f"{base}/releases/latest", url)
    else:
        found = _gh_json(get, f"{base}/releases?per_page=10", url)
    if found is None:
        return None
    releases = [found] if isinstance(found, dict) else [r for r in found if isinstance(r, dict)]
    if not releases:
        body = f"{name} has no GitHub releases (it may publish tags or a changelog instead)."
        return _source_page(url, f"Releases of {name}", "unknown", "GitHub REST API", body)
    lines = [f"Releases of {name} ({len(releases)} shown, newest first):", ""]
    for release in releases:
        flags = [f"published {_day(release.get('published_at'))}"]
        if release.get("prerelease"):
            flags.append("pre-release")
        if release.get("draft"):
            flags.append("draft")
        lines += [(f"{release.get('name') or release.get('tag_name')} (tag "
                  f"{release.get('tag_name')}, {', '.join(flags)})"),
                  str(release.get("body") or "")[:4000], ""]
    title = (f"{releases[0].get('name') or releases[0].get('tag_name')} ({name} release)"
             if len(releases) == 1 else f"Releases of {name}")
    return _source_page(url, title, _day(releases[0].get("published_at")), "GitHub REST API",
                        "\n".join(lines))


# package registries ------------------------------------------------------------------------------

def _pypi(url: str, project: str, version: str | None, get: ApiGet) -> Page | None:
    path = f"{project}/{version}" if version else project
    response = get(f"https://pypi.org/pypi/{urllib.parse.quote(path)}/json")
    if response.status == 404:
        return None
    if response.status != 200:
        raise _unexpected("PyPI JSON API", response, url)
    data = response.json("the PyPI JSON API")
    info = data.get("info") or {}
    shown = str(info.get("version", version or "?"))

    def uploaded(files: list) -> str:
        days = sorted(_day(f.get("upload_time_iso_8601") or f.get("upload_time"))
                      for f in files if isinstance(f, dict))
        return days[0] if days else "unknown"

    releases = data.get("releases") or {}
    if not releases and data.get("urls") is not None:
        releases = {shown: data.get("urls") or []}
    rows = []
    for number, files in releases.items():
        if not files:
            continue
        yanked = [f for f in files if isinstance(f, dict) and f.get("yanked")]
        reason = next((f.get("yanked_reason") for f in yanked if f.get("yanked_reason")), "")
        rows.append((uploaded(files), number, bool(yanked), reason))
    rows.sort(reverse=True)
    date = next((row[0] for row in rows if row[1] == shown), uploaded(data.get("urls") or []))
    project_urls = info.get("project_urls") or {}
    lines = [f"{info.get('name', project)} {shown}: {info.get('summary') or '(no summary)'}",
             f"{'Version shown' if version else 'Latest version'}: {shown} (uploaded {date})"
             + (f"; YANKED: {info.get('yanked_reason') or 'no reason given'}"
                if info.get("yanked") else ""),
             f"Requires Python: {info.get('requires_python') or 'not stated'}",
             "License: " + (info.get("license_expression") or (info.get("license") or "")[:100]
                            or "not stated")]
    lines += [f"{key}: {value}" for key, value in list(project_urls.items())[:6]]
    lines += ["", f"Versions ({len(rows)} with files; newest {min(len(rows), LIST_LIMIT)} shown):"]
    for day, number, yanked, reason in rows[:LIST_LIMIT]:
        mark = f"  YANKED ({reason or 'no reason given'})" if yanked else ""
        lines.append(f"  {number}  uploaded {day}{mark}")
    description = str(info.get("description") or "").strip()
    if description:
        lines += ["", "Description:", description[:6000]]
    return _source_page(url, f"{info.get('name', project)} {shown} (PyPI)", date,
                        "PyPI JSON API", "\n".join(lines))


def _npm(url: str, package: str, version: str | None, get: ApiGet) -> Page | None:
    registry = "https://registry.npmjs.org/" + package.replace("/", "%2f")
    response = get(registry)
    if response.status == 404:
        return None
    if response.status != 200:
        raise _unexpected("npm registry API", response, url)
    if response.truncated:  # very large packuments: the latest manifest only
        latest = get(f"{registry.replace('%2f', '/')}/{version or 'latest'}")
        if latest.status == 404:
            return None
        manifest = latest.json("the npm registry")
        lines = [f"{package} {manifest.get('version')}: {manifest.get('description') or ''}",
                 f"License: {manifest.get('license') or 'not stated'}",
                 "(Full version history too large to read; latest manifest only.)"]
        return _source_page(url, f"{package} {manifest.get('version')} (npm)", "unknown",
                            "npm registry API", "\n".join(lines))
    data = response.json("the npm registry")
    times = data.get("time") or {}
    versions = data.get("versions") or {}
    tags = data.get("dist-tags") or {}
    shown = version or str(tags.get("latest", ""))
    manifest = versions.get(shown) or {}
    repository = data.get("repository") or {}
    described = data.get("description") or "(no description)"
    lines = [f"{data.get('name', package)} {shown}: {described}",
             f"{'Version shown' if version else 'Latest version'}: {shown} (published "
             f"{_day(times.get(shown))})"
             + (f"; DEPRECATED: {manifest['deprecated']}" if manifest.get("deprecated") else ""),
             "dist-tags: " + ", ".join(f"{k}={v}" for k, v in tags.items()),
             f"License: {data.get('license') or manifest.get('license') or 'not stated'}",
             f"Homepage: {data.get('homepage') or 'none'}",
             f"Repository: {repository.get('url') if isinstance(repository, dict) else repository}",
             f"Registry entry modified {_day(times.get('modified'))}"]
    if times.get("unpublished"):
        lines.append(f"UNPUBLISHED {_day((times['unpublished'] or {}).get('time'))}")
    rows = sorted(((str(times.get(number, "")), number) for number in versions), reverse=True)
    lines += ["", f"Versions ({len(rows)}; newest {min(len(rows), LIST_LIMIT)} shown):"]
    for stamp, number in rows[:LIST_LIMIT]:
        deprecated = (versions.get(number) or {}).get("deprecated")
        lines.append(f"  {number}  published {_day(stamp)}"
                     + (f"  DEPRECATED ({deprecated})" if deprecated else ""))
    readme = str(data.get("readme") or "").strip()
    if readme:
        lines += ["", "README:", readme[:6000]]
    return _source_page(url, f"{data.get('name', package)} {shown} (npm)", _day(times.get(shown)),
                        "npm registry API", "\n".join(lines))


def _crates(url: str, name: str, version: str | None, get: ApiGet) -> Page | None:
    response = get(f"https://crates.io/api/v1/crates/{urllib.parse.quote(name)}")
    if response.status == 404:
        return None
    if response.status == 429:
        raise BlockedError(f"{FETCH_FAILED}: crates.io is rate-limiting this machine. Try another "
                           f"search result, or ask the user to open {url} in their browser (in "
                           "Vega's app: /open).", url)
    if response.status != 200:
        raise _unexpected("crates.io API", response, url)
    data = response.json("the crates.io API")
    crate = data.get("crate") or {}
    versions = [v for v in data.get("versions") or [] if isinstance(v, dict)]
    shown = version or str(crate.get("max_stable_version") or crate.get("newest_version")
                           or crate.get("max_version") or "")
    entry = next((v for v in versions if v.get("num") == shown), {})
    date = _day(entry.get("created_at")) if entry else "unknown"
    lines = [f"{crate.get('name', name)} {shown}: {crate.get('description') or '(no description)'}",
             f"{'Version shown' if version else 'Latest stable version'}: {shown} (published "
             f"{date})" + ("; YANKED" if entry.get("yanked") else ""),
             (f"Newest version: {crate.get('newest_version') or crate.get('max_version')}; crate "
             f"updated {_day(crate.get('updated_at'))}; created {_day(crate.get('created_at'))}"),
             (f"License: {entry.get('license') or 'not stated'}; minimum Rust: "
             f"{entry.get('rust_version') or 'not stated'}"),
             (f"Repository: {crate.get('repository') or 'none'}; docs: "
             f"{crate.get('documentation') or 'none'}; downloads {crate.get('downloads', '?')}"),
             "", f"Versions ({len(versions)}; newest {min(len(versions), LIST_LIMIT)} shown):"]
    for item in versions[:LIST_LIMIT]:
        lines.append(f"  {item.get('num')}  published {_day(item.get('created_at'))}"
                     + ("  YANKED" if item.get("yanked") else ""))
    return _source_page(url, f"{crate.get('name', name)} {shown} (crates.io)", date,
                        "crates.io API", "\n".join(lines))


# Wikipedia and MDN ------------------------------------------------------------------------------

def _wikipedia(url: str, lang: str, title: str, get: ApiGet) -> Page | None:
    base = f"https://{lang}.wikipedia.org"
    summary = get(f"{base}/api/rest_v1/page/summary/{urllib.parse.quote(title, safe='')}"
                  "?redirect=true")
    if summary.status == 404:
        return None
    if summary.status != 200:
        raise _unexpected("Wikipedia REST API", summary, url)
    head = summary.json("the Wikipedia REST API")
    content = get(f"{base}/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": "1", "redirects": "1",
        "titles": title.replace("_", " "), "format": "json", "formatversion": "2"}))
    extract = ""
    if content.status == 200:
        pages = (content.json("the Wikipedia API").get("query") or {}).get("pages") or []
        extract = str((pages[0] if pages else {}).get("extract") or "")
    page_title = str(head.get("title") or title.replace("_", " "))
    date = _day(head.get("timestamp"))
    lines = [f"{page_title}: {head.get('description') or ''}".rstrip(": "),
             f"Last edited {date}" + ("; disambiguation page" if head.get("type") ==
                                      "disambiguation" else ""), "",
             "Summary:", str(head.get("extract") or ""), ""]
    if extract:
        lines += ["Article:", extract]
    return _source_page(url, f"{page_title} - Wikipedia", date,
                        "Wikipedia REST and Action APIs", "\n".join(lines))


def _mdn_folder(slug: str) -> str:
    # mdn/content's slug-to-folder rule
    return (slug.replace("*", "_star_").replace("::", "_doublecolon_").replace(":", "_colon_")
            .replace("?", "_question_").lower())


# sidebar, banner and compatibility macros carry no text of the page
_MDN_CHROME = re.compile(r"\{\{\s*(?:\w*Sidebar|\w*Ref|SeeCompatTable|\w*_header|\w*Header|"
                         r"Non-standard\w*|Deprecated\w*|Specifications|Compat|"
                         r"AvailableInWorkers|InteractiveExample)\b[^}]*\}\}")
_MDN_MACRO_ARG = re.compile(r"\{\{\s*[\w-]+\(\s*[\"']([^\"']+)[\"'][^}]*\)\s*\}\}")
_MDN_MACRO = re.compile(r"\{\{[^{}]*\}\}")


def _mdn(url: str, lang: str, slug: str, get: ApiGet) -> Any:
    """MDN's document JSON (index.json; it follows MDN's redirects) names the page's source folder
    in mdn/content and its last-modified date; the text is that folder's raw Markdown."""
    locale = lang.lower()
    repo = "mdn/content" if locale == "en-us" else "mdn/translated-content"
    folder, filename, date, title = f"{locale}/{_mdn_folder(slug)}", "index.md", "unknown", ""
    meta = get(f"https://developer.mozilla.org/{lang}/docs/{urllib.parse.quote(slug)}/index.json")
    if meta.status == 404:
        return None
    if meta.status == 200:
        with contextlib.suppress(BackendError, AttributeError, TypeError):
            data = meta.json("MDN")
            doc = data.get("doc") or data
            source = doc.get("source") or {}
            folder = str(source.get("folder") or folder)
            filename = str(source.get("filename") or filename)
            if "translated-content" in str(source.get("github_url", "")):
                repo = "mdn/translated-content"
            date = _day(doc.get("modified"))
            title = str(doc.get("title") or "")
    path = f"files/{folder}/{filename}"
    response = get(f"{GITHUB_RAW}/{repo}/main/{urllib.parse.quote(path)}",
                   {"Accept": "text/plain, */*"})
    if response.status == 404 or not filename.endswith(".md"):
        return GENERIC  # an older HTML source, or a page the repo no longer holds: read the page
    if response.status != 200:
        raise _unexpected("GitHub raw file service", response, url)
    text = response.text()
    front = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if front:
        found = re.search(r"^title:\s*(.+)$", front.group(1), re.MULTILINE)
        title = title or (found.group(1).strip().strip("\"'") if found else "")
        text = text[front.end():]
    text = _MDN_MACRO.sub("", _MDN_MACRO_ARG.sub(r"\1", _MDN_CHROME.sub("", text))).strip()
    body = f"{title}\nLast modified {date} ({repo}, {path})\n\n{text}"
    return _source_page(url, f"{title or slug} - MDN", date, f"MDN document API and {repo}", body)


def site_adapter(url: str) -> Callable[[ApiGet], Any] | None:
    """The official-API reader for this URL, or None when no adapter covers it. The reader
    returns a Page, None (no such page), or GENERIC (read the ordinary page instead). The
    built-in adapters come first, then the ones the user approved (vega/adapter_builder.py)."""
    reader = builtin_adapter(url)
    if reader is None and features.on("web_adapter_builder"):
        from . import adapter_builder

        reader = adapter_builder.stored_reader(url)
    return reader


def builtin_adapter(url: str) -> Callable[[ApiGet], Any] | None:
    """One of the 7 hand-built readers for this URL, or None."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    host = (parsed.hostname or "").lower()
    path = urllib.parse.unquote(parsed.path)
    site = _se_site(host)
    if site:
        question = re.match(r"^/(?:questions|q)/(\d+)(?:/|$)", path)
        answer = re.match(r"^/(?:a|answers)/(\d+)(?:/|$)", path)
        if question or answer:
            return lambda get: _stack_exchange(url, site, question and question.group(1),
                                               answer and answer.group(1), get)
        return None
    if host in {"github.com", "www.github.com"}:
        parts = re.match(r"^/([^/]+)/([^/]+?)(?:\.git)?(?:/(.*?))?/?$", parsed.path)
        if parts and parts.group(1).lower() not in _GH_RESERVED:
            owner, repo, rest = parts.group(1), parts.group(2), parts.group(3) or ""
            return lambda get: _github(url, owner, repo, rest, get)
        return None
    if host in {"pypi.org", "www.pypi.org"}:
        found = re.match(r"^/project/([^/]+)/?(?:([^/]+)/?)?$", path)
        if found:
            return lambda get: _pypi(url, found.group(1), found.group(2), get)
        return None
    if host in {"npmjs.com", "www.npmjs.com"}:
        found = re.match(r"^/package/((?:@[^/]+/)?[^/@]+)(?:/v/([^/]+))?/?$", path)
        if found:
            return lambda get: _npm(url, found.group(1), found.group(2), get)
        return None
    if host == "crates.io":
        found = re.match(r"^/crates/([^/]+)(?:/([^/]+))?/?$", path)
        if found:
            return lambda get: _crates(url, found.group(1), found.group(2), get)
        return None
    wiki = re.match(r"^([a-z][a-z-]*)(?:\.m)?\.wikipedia\.org$", host)
    if wiki:
        found = re.match(r"^/wiki/(.+)$", path)
        if found:
            return lambda get: _wikipedia(url, wiki.group(1), found.group(1), get)
        return None
    if host == "developer.mozilla.org":
        found = re.match(r"^/([A-Za-z]{2}(?:-[A-Za-z]{2,4})?)/docs/(.+?)/?$", path)
        if found:
            return lambda get: _mdn(url, found.group(1), found.group(2), get)
    return None


# -- sites that refuse automated reading ---------------------------------------------------------

_CHALLENGE = re.compile(
    r"(?i)<title>\s*(?:just a moment|attention required|access denied|are you a robot|"
    r"security check|captcha|verify you are human)|cf-challenge|cf_chl_|"
    r"/cdn-cgi/challenge-platform|g-recaptcha|h-captcha|hcaptcha\.com|px-captcha|"
    r"unusual traffic from your computer")


def looks_like_challenge(html_text: str, visible_chars: int) -> bool:
    """A bot check or CAPTCHA page rather than the content (only short pages: an article that
    mentions CAPTCHAs or carries a reCAPTCHA comment form is still content)."""
    return visible_chars < 1500 and bool(_CHALLENGE.search(html_text[:200_000]))


def _challenge_reply(exc: urllib.error.HTTPError) -> bool:
    headers = exc.headers or {}
    if "challenge" in str(headers.get("cf-mitigated", "")).lower():
        return True
    try:
        body = exc.read(200_000).decode("utf-8", errors="replace")
    except OSError:
        return False
    return bool(_CHALLENGE.search(body))


def blocked_text(url: str, why: str) -> str:
    host = urllib.parse.urlparse(url).hostname or url
    return (f"{FETCH_FAILED}: {host} blocks automated reading ({why}). Vega does not get around "
            "such checks. Try another search result for the same fact. If this page matters, "
            f"ask the user to open {url} in their browser (in Vega's app they can type /open) "
            "and tell you what it says.")


class FakeWebBackend:
    """The eval fake web's FakeWeb as a backend: its own recorded tools give the exact eval text."""

    name = "fakeweb"

    def __init__(self, fake: Any) -> None:
        self.fake = fake

    def search(self, query: str, n: int = MAX_RESULTS) -> list[Result]:
        return [Result(**item) for item in self.fake.search(query)[:n]]

    def fetch(self, url: str) -> Page | None:
        page = self.fake.fetch(url)
        return Page(**page) if page else None

    # recorded: the harness writes FakeWeb.calls to .vega/web_log.json after the run
    def web_search(self, query: str) -> str:
        return self.fake.web_search(query)

    def fetch_page(self, url: str) -> str:
        return self.fake.fetch_page(url)


def configured_backend(project: Path | None = None) -> Backend:
    name = str(settings.get("web.backend", project) or "none").lower()
    if name == "searxng":
        return SearxngBackend(str(settings.get("web.searxng_url", project)))
    return NoBackend()


# -- which tasks are web research --------------------------------------------------------------

_WEB_TASK = re.compile(
    r"(?i)\b(?:look\s+up|research(?:es|ed)?\b|search\s+(?:the\s+)?(?:web|online|internet)|"
    r"on\s+the\s+web|online\b|official\s+(?:docs|documentation|site)|cite\b|citing\b|"
    r"urls?\b|sources\b|release\s+notes|changelog|CVE-\d|pricing|status\s+(?:page|history)|"
    r"still\s+supported|newest\s+release|latest\s+(?:release|version|docs))")


def wants_web(task: str) -> bool:
    """A task that asks for facts from the web (look up, research, cite sources, release notes,
    a CVE, pricing...). With web research available these go to the agent loop, the only path
    that has the web tools; without it they route as before."""
    return bool(_WEB_TASK.search(task))


# -- the tools ---------------------------------------------------------------------------------

def tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web. Use it for facts you do not know or that may have changed "
                    "since you were trained: a library's current version or API, an error message "
                    "or code from a tool, release notes, deprecations, security advisories, "
                    "pricing or limits of a service. Do NOT use it for anything in this "
                    "repository (read the files) or in the Python standard library. Send only a "
                    "short question in plain words: never code, file contents or secrets. Then "
                    "fetch_page the most authoritative result (official docs over forums and "
                    "blogs; check the published date) and cite its URL in your answer."
                ),
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                               "required": ["query"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_page",
                "description": (
                    "Read a web page (a URL from web_search results) as text, with its title and "
                    "publication date. Prefer official documentation and the newest relevant "
                    "page; cite the URLs you relied on in your answer."
                ),
                "parameters": {"type": "object", "properties": {"url": {"type": "string"}},
                               "required": ["url"]},
            },
        },
    ]


# approve(kind, detail) -> "session" | "project" | "no"; kind is "search" or "fetch"
Approver = Callable[[str, str], str]
ANSWERS = ("session", "project", "no")


def prompt_text(kind: str, detail: str, backend: str) -> tuple[str, str]:
    """(title, body) of the permission question, shared by the TUI dialog and the terminal."""
    if kind == "search":
        title = "Allow web research?"
        what = f'Vega wants to search the web for: "{detail}"'
    else:
        title = "Allow web research?"
        what = f"Vega wants to read a web page: {detail}"
    where = (f"Only the query goes to your SearXNG ({backend}); "
             if kind == "search" else "The page is fetched directly; ")
    body = (f"{what}\n{where}code, file contents and secrets are never sent. Pages it reads are "
            "kept in Vega's knowledge library with their URL and date.")
    return title, body


def console_approve(kind: str, detail: str, backend: str = "") -> str:
    """The plain terminal's permission question (line mode)."""
    from .ui import console

    title, body = prompt_text(kind, detail, backend)
    console.print(f"\n[bold yellow]{title}[/bold yellow]")
    console.print(body, markup=False, highlight=False)
    try:
        answer = console.input(
            "[yellow]\\[s] allow for this session  \\[p] always for this project  "
            "\\[N] no: [/yellow]").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "no"
    if answer in {"s", "session", "y", "yes"}:
        return "session"
    if answer in {"p", "project", "always", "a"}:
        return "project"
    return "no"


class WebSession:
    """One session's web research: backend, permission state, what was used this turn."""

    def __init__(self, project: Path, *, backend: Backend | None = None,
                 permission: str | None = None, cache: bool = True,
                 extra_official: list[str] | None = None) -> None:
        self.project = Path(project)
        self._backend = backend
        self.permission = permission  # "allow" / "deny" once decided for this session
        self.cache = cache  # store fetched pages in the knowledge library
        self.extra_official = extra_official
        self.turn_urls: list[str] = []  # URLs fetched (found) this turn, in order
        self.turn_results: list[str] = []  # URLs search results showed this turn
        self.turn_official: list[str] = []  # of those, marked "(official site)" (in order)
        self.turn_searches: list[list[str]] = []  # each search's result URLs, in rank order
        # web_facet_fetch (vega/web_facets.py): the task, the repo files read and every result's
        # title/snippet/date this turn, so the notes can name the pages that decide the task
        self.turn_task = ""
        self.turn_context: list[str] = []
        self.turn_meta: dict[str, Result] = {}  # canonical URL -> result (or page) seen this turn
        self.facet_picks: list[web_facets.Pick] = []  # the last facet_note's suggestions
        # cite_named_pages: canonical URL -> the facet a fetch-first or finish note named it for
        self.turn_named: dict[str, str] = {}
        self.named_checked = False  # cite_named_pages: its send-back was made this turn
        self.turn_used = False
        self.blocked_url = ""  # the last page a site refused to Vega (the TUI's /open)
        # the adapter builder (vega/adapter_builder.py, switch web_adapter_builder): the agent
        # sets the model and a progress callback, the TUI the approval dialog (None: terminal)
        self.adapter_model: Callable[[list[dict[str, str]]], str] | None = None
        self.adapter_model_name = "the model"
        self.approve_adapter: Callable[[Any], bool] | None = None
        self.notify: Callable[[str], None] | None = None
        self.result_titles: dict[str, str] = {}  # URL -> title from this session's searches
        self.adapter_attempted: set[str] = set()  # sites built (or tried) this session

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = configured_backend(self.project)
        return self._backend

    def setting(self) -> str:
        return str(settings.get("web.permission", self.project) or "ask").lower()

    def available(self) -> bool:
        """Offer the tools this turn: the switch is on, a backend exists and web use is not
        denied (by this session's answer or the settings)."""
        if not features.on("web_research") or isinstance(self.backend, NoBackend):
            return False
        if self.permission is not None:
            return self.permission == "allow"
        return self.setting() != "deny"

    def start_turn(self, task: str = "") -> None:
        self.turn_urls, self.turn_results, self.turn_used = [], [], False
        self.turn_official, self.turn_searches = [], []
        self.turn_task, self.turn_context, self.turn_meta = task, [], {}
        self.turn_named, self.named_checked = {}, False

    def add_context(self, text: str) -> None:
        """web_facet_fetch: a repository file read this turn (its versions and topic words become
        facets: a platform doc's "Python 3.12", "Licences")."""
        if len(self.turn_context) < 12:
            self.turn_context.append(text[:20_000])

    def _allowed(self, kind: str, detail: str, approve: Approver | None) -> bool:
        if self.permission is not None:
            return self.permission == "allow"
        setting = self.setting()
        if setting in {"allow", "deny"}:
            self.permission = setting
            return setting == "allow"
        backend = getattr(self.backend, "base_url", self.backend.name)
        try:
            answer = (approve(kind, detail) if approve is not None
                      else console_approve(kind, detail, backend))
        except Exception:  # noqa: BLE001 - a question that cannot be asked is a "no"
            answer = "no"
        if answer == "project":
            settings.set_value("web.permission", "allow", project=self.project)
        self.permission = "allow" if answer in {"session", "project"} else "deny"
        return self.permission == "allow"

    def call(self, name: str, args: dict[str, Any], approve: Approver | None = None) -> str:
        """Run web_search or fetch_page; always returns the text for the model."""
        if not features.on("web_research"):
            return f"{UNAVAILABLE}: web research is switched off (feature web_research)."
        backend = self.backend
        if isinstance(backend, NoBackend):
            return f"{UNAVAILABLE}: no search backend is configured. {SETUP_HINT}"
        if name == "web_search":
            query = str(args.get("query", "")).strip()
            refusal = check_query(query, self.project)
            if refusal:
                return refusal
            if not self._allowed("search", query, approve):
                return DECLINED_TEXT
            self.turn_used = True
            try:
                if isinstance(backend, FakeWebBackend):
                    text = backend.web_search(query)
                    results = backend.search(query, MAX_RESULTS)
                    urls = [result.url for result in results]
                else:
                    results = backend.search(query, MAX_RESULTS)
                    urls = [result.url for result in results]
                    self.result_titles.update({result.url: result.title for result in results})
                    text = format_results(query, results)
            except BackendError as exc:
                return str(exc)
            self.turn_results += urls
            self.turn_searches.append(urls)
            for result in results:
                self.turn_meta.setdefault(canonical_url(result.url), result)
            if features.on("web_official_hint"):
                official = official_urls(query, urls, self._extra_official())
                self.turn_official += [url for url in urls
                                       if url in official and url not in self.turn_official]
                text = mark_official(text, official)
            return text
        if name == "fetch_page":
            url = str(args.get("url", "")).strip().strip("<>\"'")
            refusal = check_url(url, self.project)
            if refusal:
                return refusal
            if not self._allowed("fetch", url, approve):
                return DECLINED_TEXT
            self.turn_used = True
            try:
                if isinstance(backend, FakeWebBackend):
                    text = backend.fetch_page(url)
                    page = backend.fetch(url)
                else:
                    page = backend.fetch(url)
                    text = format_page(page) if page else NOT_FOUND + url
            except BlockedError as exc:
                built = self._build_adapter(url, str(exc), blocked=True)
                if built is None or isinstance(built, str):
                    self.blocked_url = exc.url
                    return str(exc) + (f"\n{built}" if built else "")
                page, text = built, format_page(built)
            except BackendError as exc:
                return str(exc)
            if page is not None and not isinstance(backend, FakeWebBackend):
                from . import adapter_builder

                if adapter_builder.poor_page(page):
                    built = self._build_adapter(url, "the page's text is too short to use "
                                                "(likely rendered by scripts)", blocked=False)
                    if isinstance(built, Page):
                        page, text = built, format_page(built)
                    elif built:
                        self.blocked_url = url
                        text += f"\n\n{built}"
            if page is None:
                return text
            self.turn_urls.append(page.url)
            self.turn_meta.setdefault(canonical_url(page.url),
                                      Result(page.title, page.url, "", page.date))
            if self.cache:
                self._cache(page)
            return text
        return f"Unknown tool: {name}"

    def _build_adapter(self, url: str, reason: str, *, blocked: bool) -> Page | str | None:
        """web_adapter_builder: for a page that could not be read, on a site no adapter covers,
        find the site's official API and build an adapter (the user approves it). A Page when
        an approved adapter read it; else a note for the model (None: the builder did not
        run)."""
        from . import adapter_builder

        backend = self.backend
        api_get = getattr(backend, "api_get", None)
        if (not adapter_builder.on() or not features.on("web_site_apis") or api_get is None
                or self.adapter_model is None or site_adapter(url) is not None):
            return None
        site = adapter_builder.site_key(url)
        if site in self.adapter_attempted:
            return None
        self.adapter_attempted.add(site)
        previous = adapter_builder.find(url, include_unusable=True)
        if previous is not None and previous.meta.get("status") != "failing":
            previous = None  # changed by hand or unapproved: build afresh, keep nothing of it

        def search(query: str) -> list[Result]:
            if check_query(query, self.project):
                return []
            results = backend.search(query, MAX_RESULTS)
            self.result_titles.update({result.url: result.title for result in results})
            return results

        result = adapter_builder.build(
            url, reason, get=api_get, model=self.adapter_model, approve=self.approve_adapter,
            search=search, titles=dict(self.result_titles), notify=self.notify,
            model_name=self.adapter_model_name, previous=previous)
        if result.status == "stored" and result.page is not None:
            return result.page
        if result.status == "no_api" and not blocked:
            return (f"(This page's text is short and {result.message} If the page matters, ask "
                    f"the user to open {url} in their browser (in Vega's app: /open).)")
        return f"Adapter builder: {result.message}"

    def _extra_official(self) -> list[str]:
        if self.extra_official is not None:
            return list(self.extra_official)
        return list(settings.get("web.official_domains", self.project) or [])

    def _cache(self, page: Page) -> None:
        from . import knowledge

        extra = self._extra_official()
        with contextlib.suppress(Exception):  # caching never fails the fetch
            knowledge.store_web_page(page.url, page.title, page.date, page.text,
                                     trust_tier(page.url, extra))

    def to_fetch(self, count: int = 2) -> list[str]:
        """The search results to read first, never one already fetched: each search's
        official-site results (web_official_hint) taken in turn (the first of every search, then
        the second...), then all results the same way, so two searches about two products
        suggest one page of each."""
        fetched = {canonical_url(url) for url in self.turn_urls}
        official = set(self.turn_official) if features.on("web_official_hint") else set()
        searches = self.turn_searches or [self.turn_results]
        picked: list[str] = []
        for wanted in ([[u for u in urls if u in official] for urls in searches], searches):
            for rank in range(max((len(urls) for urls in wanted), default=0)):
                for urls in wanted:
                    if rank < len(urls) and urls[rank] not in picked \
                            and canonical_url(urls[rank]) not in fetched:
                        picked.append(urls[rank])
        return picked[:count]

    def _hits(self) -> tuple[list[web_facets.Hit], list[web_facets.Hit]]:
        """web_facet_fetch: this turn's search results (in to_fetch's order: each search's first
        result, then each one's second...) and the pages fetched, as web_facets sees them.
        Official: marked by web_official_hint, on an official-docs host, or the own site of an
        option the task names (web_facets.option_words)."""
        order: list[str] = []
        searches = self.turn_searches or [self.turn_results]
        for rank in range(max((len(urls) for urls in searches), default=0)):
            for urls in searches:
                if rank < len(urls) and canonical_url(urls[rank]) not in order:
                    order.append(canonical_url(urls[rank]))
        fetched = list(dict.fromkeys(canonical_url(url) for url in self.turn_urls))
        extra = self._extra_official()
        official = {canonical_url(url) for url in self.turn_official}

        def hit(url: str, rank: int, mark: bool = False) -> web_facets.Hit:
            meta = self.turn_meta.get(url) or Result("", url, "", "unknown")
            return web_facets.Hit(url, meta.title, meta.snippet, meta.date, rank, mark or (
                url in official or trust_tier(url, extra) == "official"))

        draft = [hit(url, rank) for rank, url in enumerate([*order, *fetched])]
        options = web_facets.option_words(self.turn_task, draft, _GENERIC_WORDS)
        marked = [hit(item.url, item.rank, any(
            label.startswith(word) for word in options
            for label in web_facets.host_labels(item.url))) for item in draft]
        return marked[:len(order)], marked[len(order):]

    def _shown(self, url: str) -> str:
        """A result's URL as the search showed it (web_facets works on canonical URLs)."""
        meta = self.turn_meta.get(url)
        return meta.url if meta is not None else url

    def suggestions(self, count: int = web_facets.MAX_SUGGESTIONS, *, topics: bool = True,
                    stated: str = "", official_only: bool = False) -> list[web_facets.Pick]:
        """web_facet_fetch: the unfetched results that cover the task's facets (and the values
        `stated` names, from the answer and the files written), at most `count`."""
        hits, fetched = self._hits()
        found = web_facets.facets(self.turn_task, self.turn_context, [*hits, *fetched],
                                  _GENERIC_WORDS)
        if stated:
            known = {facet.label for facet in found}
            found += [facet for facet in map(web_facets.value_facet, web_facets.values_in(stated))
                      if facet.label not in known and any(facet.matches(h) for h in hits)]
        return web_facets.suggest(found, hits, fetched, count=count, topics=topics,
                                  official_only=official_only)

    def fetch_list(self) -> list[tuple[str, str]]:
        """The (url, why) pairs a fetch-first note names: with web_facet_fetch, the facet
        suggestions padded with the old top 2 (to_fetch) to at least 2; else the top 2."""
        top = [(url, "") for url in self.to_fetch()]
        if not features.on("web_facet_fetch") or not self.turn_task:
            return top
        listed = [(self._shown(pick.url), pick.why) for pick in self.suggestions()]
        named = {canonical_url(url) for url, _ in listed}
        for url, why in listed:  # cite_named_pages: the pages named for a facet
            if why:
                self.turn_named.setdefault(canonical_url(url), why)
        for url, why in top:
            if len(listed) >= 2:
                break
            if canonical_url(url) not in named:
                listed.append((url, why))
        return listed

    def _listed(self, items: list[str] | list[tuple[str, str]]) -> str:
        official = {canonical_url(url) for url in self.turn_official}
        if features.on("web_facet_fetch"):
            official |= {hit.url for hit in self._hits()[0] if hit.official}
        lines = []
        for item in items:
            url, why = item if isinstance(item, tuple) else (item, "")
            lines.append(f"- {url}" + ("  (official site)" if canonical_url(url) in official
                                       else "") + (f"  [for: {why}]" if why else ""))
        return "\n".join(lines)

    def facet_note(self, answer: str, written: list[str] | None = None) -> str | None:
        """web_facet_fetch: at the finish of a turn that read pages, the options and versions of
        the task (and the versions/dates the answer or the files written this turn state) that
        no page read covers, each with the search result that does (None: nothing open)."""
        if not features.on("web_facet_fetch") or not self.turn_urls or not self.turn_task:
            return None
        picks = self.suggestions(topics=False, stated="\n".join([answer, *(written or [])]),
                                 official_only=True)
        self.facet_picks = picks
        if not picks:
            return None
        for pick in picks:
            self.turn_named.setdefault(canonical_url(pick.url), pick.why)
        return (
            "(Vega harness) Before you finish: parts of the task (or of your answer) are not "
            "covered by any page you read. These search results cover them:\n"
            + self._listed([(self._shown(pick.url), pick.why) for pick in picks])
            + "\nfetch_page each one, check your answer and your changes against what it says "
            "(a newer release, changelog or advisory can change the version or the format), fix "
            "what differs, and cite the pages you relied on (in `sources` too, if the task asks "
            "for it)."
        )

    def fetch_first_note(self) -> str | None:
        """web_fetch_first: what the turn's first write returns instead of running when the web
        was searched but no page read yet (None when that is not the case)."""
        if self.turn_urls or not self.turn_results:
            return None
        return (
            "(Vega harness) Not written yet: you searched the web but have not read any page, so "
            "this change would rest on search-result snippets (often partial or outdated). First "
            "fetch_page the results that settle it:\n" + self._listed(self.fetch_list())
            + "\nCheck what the pages say, then make the change again (adjusted where they say "
            "otherwise)."
        )

    def citation_note(self, answer: str, sources: list[str] | None = None) -> str | None:
        """The note that sends a final answer back about its sources (None when it may stand).

        web_fetch_first: only pages FETCHED this turn count as read. A search result the answer
        names, or any web URL in the `sources` list of a JSON file written this turn (`sources`),
        that was never fetched: "you cite X but never read it". An answer citing no fetched page:
        the pages read (or, when none was, the results to read first). Switched off: the old rule
        (one cited page fetched or seen in search results)."""
        if not self.turn_used:
            return None
        if not features.on("web_fetch_first"):
            return self._old_citation_note(answer)
        fetched = {canonical_url(url) for url in self.turn_urls}
        results = {canonical_url(url) for url in self.turn_results}
        in_answer = _cited(answer)
        listed = [canonical_url(url) for url in sources or [] if is_web_url(url)]
        # an answer's URL that no search showed may be an endpoint or an example, not a citation
        unread = [url for url in dict.fromkeys([*listed, *sorted(in_answer & results)])
                  if url not in fetched]
        if unread:
            where = " (listed in the `sources` of the JSON file you wrote)" \
                if set(unread) & set(listed) else ""
            return (
                f"(Vega harness) You cite pages you never read this turn{where}:\n"
                + "\n".join(f"- {url}" for url in unread[:4])
                + "\nA search snippet is not the page. fetch_page each of them, check your answer "
                "and your change against what the page says, and fix what differs. Cite (and list "
                "in `sources`) only pages you fetched; drop any you did not read."
            )
        left_out = self.named_left_out(listed)
        if left_out:
            self.named_checked = True
            return (
                "(Vega harness) The `sources` of the JSON file you wrote leave out pages you read "
                "this turn that cover parts of the task:\n"
                + "\n".join(f"- {url}" + (f"  [for: {why}]" if why else "") for url, why in left_out)
                + "\nIf your answer or a value you wrote rests on one of them (or it changes "
                "them), add it to `sources` and fix what differs. If a page does not apply or is "
                "outdated, leave it out and say why in your answer."
            )
        if (in_answer | set(listed)) & fetched:
            return None
        if self.turn_urls:
            return (
                "(Vega harness) You used web research this turn, but your answer cites no URL of "
                "a page you read. Give the answer again and include the URL of each page you "
                "relied on (pages fetched this turn):\n"
                + "\n".join(f"- {url}" for url in dict.fromkeys(self.turn_urls))
                + "\nCite only pages you actually read; if the web did not settle something, say "
                "so."
            )
        if not self.turn_results:
            return None
        return (
            "(Vega harness) You searched the web this turn but read no page, so your answer cites "
            "no URL you read and rests on search-result snippets. fetch_page the results that "
            "settle it:\n" + self._listed(self.fetch_list())
            + "\nCheck your answer (and any change you made) against them, fix what differs, and "
            "cite the pages you read."
        )

    def named_left_out(self, listed: list[str]) -> list[tuple[str, str]]:
        """cite_named_pages: (url, facet) for each page a facet note named that was fetched this
        turn and that the written `sources` (canonical URLs, non-empty) leave out; once per turn
        (2026-10-19: an answer read the docs page named for one option and cited only another
        option's news; another kept `sources` at a versions page after the changelogs the finish
        note named were read)."""
        if not listed or self.named_checked or not features.on("cite_named_pages"):
            return []
        fetched = {canonical_url(url) for url in self.turn_urls}
        cited = set(listed)
        return [(self._shown(url), why) for url, why in self.turn_named.items()
                if url in fetched and url not in cited]

    def _old_citation_note(self, answer: str) -> str | None:
        relied = self.turn_urls or self.turn_results
        if not relied:
            return None
        if _cited(answer) & {canonical_url(url) for url in relied}:
            return None
        shown = "\n".join(f"- {url}" for url in dict.fromkeys(relied))
        which = "fetched" if self.turn_urls else "seen in search results"
        return (
            "(Vega harness) You used web research this turn, but your answer cites no URL. Give "
            "the answer again and include the URL of each page you relied on (pages "
            f"{which} this turn):\n{shown}\nCite only pages you actually read; if the web did "
            "not settle something, say so."
        )
