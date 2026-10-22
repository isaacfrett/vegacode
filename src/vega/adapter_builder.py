"""Site-adapter builder: when a page cannot be read and no adapter covers its site, find the site's
OFFICIAL API, have the model write an adapter for it, verify it like a coding task, and store it
only after the user approves it.

Design (2026-10-17): Vega adapts to a site's API the way it solves a coding task. The flow
(switch `web_adapter_builder`, inert without web permission):

1. Trigger: fetch_page of a page is blocked (BlockedError) or returns poor text (< 300 characters
   of real content) on a site with no adapter (built-in or stored).
2. Discovery, official sources only: the site's OpenAPI/Swagger description (/.well-known/...,
   /openapi.json, /swagger.json, api.<domain>), developer-docs links on the site, APIs.guru's
   directory, or docs pages on the site's own domain found by web_search. Nothing found: stop and
   offer the user's browser (/open). Never browser impersonation, scraping workarounds or
   CAPTCHA bypass.
3. Generation: the model writes a module with `MATCH` (URL regex) and `fetch(url, api_get) ->
   Page`, given the prompt, the 7 hand-built adapters of vega/web.py as worked examples, an
   excerpt of the API description and the page URL.
4. Verification: the module runs in a sandboxed subprocess (vega/adapter_runner.py; no network,
   no file writes) whose only way out is the harness's `api_get` proxy, limited to the API's
   domains, against the blocked URL and up to 3 sibling URLs of the same site from search
   results. A page passes when its title matches the search result's title (fuzzy), its text
   has >= 300 characters of real content and its date is a plausible YYYY-MM-DD or "unknown".
   Failures go back to the model as repair feedback (at most 3 repair rounds).
5. Approval: the user approves each new adapter before its first real use (site, API domains,
   what it reads, rate limits, code size; approve / deny / view code). Denied: nothing stored.
6. Storage: ~/.vega/adapters/<site>.py + <site>.json (provenance). web.site_adapter loads them
   after the built-ins; a stored adapter that fails is re-verified, and marked failing (so the
   next blocked fetch rebuilds it, with approval again) when re-verification fails too.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import difflib
import hashlib
import html as html_lib
import inspect
import ipaddress
import json
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from datetime import date as _date
from pathlib import Path
from typing import Any

from . import features, server, web

FEATURE = "web_adapter_builder"
APIS_GURU = "https://api.apis.guru/v2"  # the public API directory (its per-provider JSON)
REPAIR_ROUNDS = 3
MAX_SIBLINGS = 3
MIN_CONTENT = 300  # characters of real content a page must have
RUN_TIMEOUT = 90  # seconds for one sandboxed run (all its URLs)
MAX_CALLS_PER_URL = 20
DEFAULT_INTERVAL = 1.0  # seconds between requests to one API domain when no limit is documented
EXCERPT_CHARS = 6000
CODE_LIMIT = 20_000  # characters of adapter code accepted

# model(messages) -> reply text; approve(candidate) -> True to store and use it
Model = Callable[[list[dict[str, str]]], str]
Approve = Callable[["Candidate"], bool]
Notify = Callable[[str], None]


def on() -> bool:
    return features.on(FEATURE)


def adapters_dir() -> Path:
    return server.STATE_DIR / "adapters"


# -- domains -----------------------------------------------------------------------------------

_TWO_LEVEL = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.jp",
              "co.nz", "com.br", "co.in", "co.kr", "com.cn", "com.mx", "co.za"}


def base_domain(host: str) -> str:
    """The registrable domain (example.com for api.example.com); an IP address is itself."""
    host = host.lower().strip(".").split(":")[0].removeprefix("www.")
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _TWO_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def netloc(url: str) -> str:
    """host[:port] (port only when not the scheme's default), lower-cased, without 'www.'."""
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    port = parsed.port
    default = {"http": 80, "https": 443}.get(parsed.scheme)
    return f"{host}:{port}" if port and port != default else host


def site_key(url_or_site: str) -> str:
    site = netloc(url_or_site) if "://" in url_or_site else url_or_site.lower()
    return re.sub(r"[^a-z0-9.-]", "_", site)


def official(host: str, site: str) -> bool:
    """A host on the site's own registrable domain."""
    return bool(host) and base_domain(host) == base_domain(site)


def domain_allowed(url: str, domains: list[str]) -> bool:
    """True when the URL's host is one of `domains` (or a subdomain of one); an entry with a port
    allows only that port."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    try:
        port = parsed.port or {"http": 80, "https": 443}[parsed.scheme]
    except ValueError:
        return False
    for entry in domains:
        name, _, entry_port = entry.lower().partition(":")
        if entry_port and str(port) != entry_port:
            continue
        if host == name or host.endswith("." + name):
            return True
    return False


# -- page checks -------------------------------------------------------------------------------

def content_chars(text: str) -> int:
    """Characters of real content: lines of prose (3+ words), not the Source line, not JSON."""
    total = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Source:"):
            continue
        if len(re.findall(r"[A-Za-z]{2,}", stripped)) < 3:
            continue
        punctuation = sum(stripped.count(c) for c in '{}[]":')
        if punctuation > len(stripped) * 0.12:
            continue
        total += len(re.sub(r"\s+", " ", stripped))
    return total


def poor_page(page: web.Page) -> bool:
    return content_chars(page.text) < MIN_CONTENT


def _normal_title(title: str) -> str:
    text = html_lib.unescape(title).lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def title_matches(found: str, expected: str) -> bool:
    """Fuzzy: same words in order mostly, or one title contains the other (a site name suffix
    such as ' - Example Blog' or ' | Example' is common on search results)."""
    a, b = _normal_title(found), _normal_title(expected)
    if not a or not b:
        return False
    if a in b or b in a:
        return min(len(a), len(b)) >= 0.5 * max(len(a), len(b)) or len(min(a, b, key=len)) >= 12
    if difflib.SequenceMatcher(None, a, b).ratio() >= 0.75:
        return True
    words_a, words_b = set(a.split()), set(b.split())
    return len(words_a & words_b) >= 0.7 * min(len(words_a), len(words_b))


def plausible_date(value: str) -> bool:
    if value == "unknown":
        return True
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value or ""):
        return False
    try:
        day = _date.fromisoformat(value)
    except ValueError:
        return False
    return _date(1990, 1, 1) <= day <= _date.fromordinal(_date.today().toordinal() + 1)


def check_page(page: dict[str, str] | None, expected_title: str) -> list[str]:
    """What is wrong with one adapter result (empty: it passes)."""
    if page is None:
        return ["no page"]
    problems = []
    title = page.get("title", "")
    if not title.strip():
        problems.append("the title is empty")
    elif expected_title and not title_matches(title, expected_title):
        problems.append(f"the title {title!r} does not match the page's real title "
                        f"{expected_title!r}")
    chars = content_chars(page.get("text", ""))
    if chars < MIN_CONTENT:
        problems.append(f"the text has {chars} characters of readable content (needs at least "
                        f"{MIN_CONTENT}); write the article's text as prose lines, not raw JSON "
                        "or field names")
    if not plausible_date(page.get("date", "")):
        problems.append(f"the date {page.get('date')!r} is not a plausible YYYY-MM-DD "
                        "publication date or \"unknown\"")
    return problems


# -- rate limits -------------------------------------------------------------------------------

_PER = {"second": 1, "sec": 1, "s": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600,
        "day": 86400}
_LIMIT_TEXT = re.compile(
    r"(?i)(\d[\d,]*)\s*(?:api\s+)?(?:requests?|calls?|req)\s*(?:per|/|an?|each|every)\s*"
    r"(second|sec|minute|min|hour|hr|day|s)\b")


def parse_rate_limits(text: str) -> dict[str, Any] | None:
    """The strictest documented limit in `text` ("60 requests per minute"), as requests per
    seconds; None when none is documented."""
    best: dict[str, Any] | None = None
    for match in _LIMIT_TEXT.finditer(text or ""):
        try:
            requests = int(match.group(1).replace(",", ""))
        except ValueError:
            continue
        seconds = _PER[match.group(2).lower()]
        if requests <= 0:
            continue
        found = {"requests": requests, "seconds": seconds, "text": match.group(0).strip()}
        if best is None or seconds / requests > best["seconds"] / best["requests"]:
            best = found
    return best


def describe_limits(limits: dict[str, Any] | None) -> str:
    if not limits:
        return (f"none documented; Vega waits {DEFAULT_INTERVAL:g} s between requests and stops "
                "on HTTP 429 or an exhausted X-RateLimit-Remaining")
    return (f"{limits['text']} (documented); Vega spaces requests to fit and stops on HTTP 429 "
            "or an exhausted X-RateLimit-Remaining")


class RateLimiter:
    """Per-domain spacing between requests, plus server-announced waits (Retry-After,
    X-RateLimit-Remaining: 0 with a reset time). Shared by every adapter run in this process."""

    def __init__(self) -> None:
        self.last: dict[str, float] = {}
        self.blocked_until: dict[str, float] = {}
        self.lock = threading.Lock()

    def interval(self, limits: dict[str, Any] | None) -> float:
        if limits:
            return max(float(limits["seconds"]) / float(limits["requests"]), 0.0)
        return DEFAULT_INTERVAL

    def wait(self, domain: str, limits: dict[str, Any] | None) -> str | None:
        """Sleep until the next request may go; a refusal text when the server asked to wait
        longer than a minute."""
        with self.lock:
            now = time.monotonic()
            until = self.blocked_until.get(domain, 0.0)
            if until - now > 60:
                return (f"the API at {domain} asked Vega to wait {int(until - now)} s (rate "
                        "limit)")
            pause = max(until - now, self.last.get(domain, -1e9) + self.interval(limits) - now)
            if pause > 0:
                time.sleep(min(pause, 60))
            self.last[domain] = time.monotonic()
        return None

    def observe(self, domain: str, response: web.ApiResponse) -> bool:
        """Record what the reply says about limits; True when the server is rate-limiting us."""
        headers = response.headers
        wait = 0.0
        retry = headers.get("retry-after", "")
        if retry.isdigit():
            wait = float(retry)
        remaining = headers.get("x-ratelimit-remaining") or headers.get("ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset") or headers.get("ratelimit-reset") or ""
        if remaining == "0" and reset.isdigit():
            value = int(reset)
            wait = max(wait, value - time.time() if value > 1e9 else value)
        limited = response.status == 429 or remaining == "0"
        if limited:
            with self.lock:
                self.blocked_until[domain] = time.monotonic() + max(wait, 60.0)
        return limited


LIMITER = RateLimiter()

# request headers an adapter may not send: credentials, identity, browser impersonation
_DROPPED_HEADER = re.compile(r"(?i)^(?:user-agent|cookie|referer|origin|proxy-.*|.*auth.*|"
                             r".*token.*|.*api-?key.*|.*secret.*|sec-.*)$")


# -- discovery ---------------------------------------------------------------------------------

@dataclass
class ApiDescription:
    site: str  # the page's site (netloc)
    docs_url: str  # where the description was read
    spec_url: str  # the OpenAPI/Swagger document, or "" for a docs page
    kind: str  # "openapi" | "docs"
    found_via: str  # "well-known" | "site path" | "api subdomain" | "directory" | "docs link"
    # | "search"
    api_domains: list[str]
    excerpt: str
    rate_limits: dict[str, Any] | None = None


def _html(response: web.ApiResponse) -> str:
    return response.text() if response.status == 200 else ""


def _parse_spec(text: str) -> dict[str, Any] | None:
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except ValueError:
            return None
        if isinstance(data, dict) and ("openapi" in data or "swagger" in data) and \
                isinstance(data.get("paths"), dict):
            return data
        return None
    if re.match(r"(?m)^(?:openapi|swagger)\s*:", stripped):
        return {"__yaml__": text}
    return None


def _servers(spec: dict[str, Any], spec_url: str) -> list[str]:
    """Base URLs the spec says its API lives at (relative ones resolved against the spec URL)."""
    found: list[str] = []
    if "__yaml__" in spec:
        text = spec["__yaml__"]
        block = re.search(r"(?ms)^servers\s*:\s*\n((?:[ \t]+.*\n?)+)", text)
        if block:
            found += re.findall(r"url\s*:\s*['\"]?([^'\"\s]+)", block.group(1))
        host = re.search(r"(?m)^host\s*:\s*['\"]?([^'\"\s]+)", text)
        if host:
            base = re.search(r"(?m)^basePath\s*:\s*['\"]?([^'\"\s]+)", text)
            found.append(f"https://{host.group(1)}{base.group(1) if base else ''}")
    else:
        for server_entry in spec.get("servers") or []:
            if isinstance(server_entry, dict) and server_entry.get("url"):
                found.append(str(server_entry["url"]))
        if spec.get("host"):
            schemes = spec.get("schemes") or ["https"]
            found.append(f"{schemes[0]}://{spec['host']}{spec.get('basePath', '')}")
    resolved = [urllib.parse.urljoin(spec_url, url) for url in found] or [
        urllib.parse.urljoin(spec_url, "/")]
    return [url for url in resolved if url.startswith(("http://", "https://"))]


def _resolve_ref(spec: dict[str, Any], node: Any, depth: int = 0) -> Any:
    while isinstance(node, dict) and "$ref" in node and depth < 6:
        path = str(node["$ref"]).lstrip("#/").split("/")
        target: Any = spec
        for part in path:
            target = target.get(part) if isinstance(target, dict) else None
        node, depth = target, depth + 1
    return node


def _schema_text(spec: dict[str, Any], schema: Any, depth: int = 0) -> str:
    schema = _resolve_ref(spec, schema)
    if not isinstance(schema, dict) or depth > 2:
        return ""
    kind = schema.get("type") or ("object" if "properties" in schema else "")
    if kind == "array":
        return "array of " + (_schema_text(spec, schema.get("items"), depth + 1) or "items")
    properties = schema.get("properties")
    if isinstance(properties, dict):
        parts = []
        for name, sub in list(properties.items())[:25]:
            inner = _schema_text(spec, sub, depth + 1)
            sub = _resolve_ref(spec, sub)
            described = (sub or {}).get("description", "") if isinstance(sub, dict) else ""
            typed = (sub or {}).get("type", "") if isinstance(sub, dict) else ""
            label = f"{name}: {inner or typed or 'value'}"
            if described and depth == 0:
                label += f" ({str(described)[:80]})"
            parts.append(label)
        return "object {" + "; ".join(parts) + "}"
    return str(kind)


def _relevance(path: str, page_url: str) -> int:
    words = set(re.findall(r"[a-z]{3,}", urllib.parse.urlparse(page_url).path.lower()))
    path_words = set(re.findall(r"[a-z]{3,}", path.lower()))
    return len(words & path_words) * 3 + (1 if "{" in path else 0) - path.count("/") // 4


def spec_excerpt(spec: dict[str, Any], page_url: str, servers: list[str]) -> str:
    """The parts of an OpenAPI/Swagger document an adapter needs: servers, then the GET
    operations most like the page's URL with parameters and response fields."""
    if "__yaml__" in spec:
        return spec["__yaml__"][:EXCERPT_CHARS]
    info = spec.get("info") or {}
    kind = (f"OpenAPI {spec['openapi']}" if spec.get("openapi")
            else f"Swagger {spec.get('swagger')}")
    lines = [f"{kind}: {info.get('title', '')} (version {info.get('version', '?')})"]
    if info.get("description"):
        lines.append(str(info["description"]).strip()[:800])
    lines.append("Servers: " + ", ".join(servers))
    operations = []
    for path, item in (spec.get("paths") or {}).items():
        if isinstance(item, dict) and isinstance(item.get("get"), dict):
            operations.append((_relevance(path, page_url), path, item["get"], item))
    operations.sort(key=lambda row: -row[0])
    lines.append(f"GET operations ({len(operations)}; most relevant to the page first):")
    for _, path, op, item in operations[:14]:
        lines.append(f"GET {path} - {op.get('summary') or op.get('operationId') or ''}".rstrip(" -"))
        params = [_resolve_ref(spec, p) for p in [*(item.get("parameters") or []),
                                                  *(op.get("parameters") or [])]]
        named = [f"{p.get('name')} ({p.get('in')}{', required' if p.get('required') else ''})"
                 for p in params if isinstance(p, dict) and p.get("name")]
        if named:
            lines.append("  parameters: " + ", ".join(named))
        responses = op.get("responses") or {}
        ok = _resolve_ref(spec, responses.get("200") or responses.get(200) or {})
        schema = None
        if isinstance(ok, dict):
            content = ok.get("content") or {}
            media = content.get("application/json") or next(iter(content.values()), {}) \
                if isinstance(content, dict) else {}
            schema = (media or {}).get("schema") if isinstance(media, dict) else None
            schema = schema or ok.get("schema")
        shape = _schema_text(spec, schema)
        if shape:
            lines.append(f"  200: {shape}"[:900])
    text = "\n".join(lines)
    return text[:EXCERPT_CHARS]


def _limits_in_spec(spec: dict[str, Any], text: str) -> dict[str, Any] | None:
    found = parse_rate_limits(text)
    if found:
        return found
    raw = json.dumps(spec)[:400_000] if "__yaml__" not in spec else spec["__yaml__"]
    for match in re.finditer(r"(?i)\"?x-rate-?limit[\w-]*\"?\s*:\s*(\{[^{}]*\}|[^,\n]+)", raw):
        numbers = re.findall(r"\d+", match.group(1))
        unit = re.search(r"(?i)(second|minute|hour|day)", match.group(1))
        if numbers and unit:
            return {"requests": int(numbers[0]), "seconds": _PER[unit.group(1).lower()],
                    "text": f"{numbers[0]} requests per {unit.group(1).lower()}"}
    return parse_rate_limits(raw[:200_000])


def _describe_spec(spec: dict[str, Any], spec_url: str, page_url: str, found_via: str,
                   need_official_servers: bool) -> ApiDescription | None:
    site = netloc(page_url)
    servers = _servers(spec, spec_url)
    if need_official_servers:
        servers = [url for url in servers if official(urllib.parse.urlparse(url).hostname or "",
                                                      site)]
    if not servers:
        return None
    domains = list(dict.fromkeys(netloc(url) for url in servers))
    excerpt = spec_excerpt(spec, page_url, servers)
    return ApiDescription(site=site, docs_url=spec_url, spec_url=spec_url, kind="openapi",
                          found_via=found_via, api_domains=domains, excerpt=excerpt,
                          rate_limits=_limits_in_spec(spec, excerpt))


_LINK = re.compile(r"<(a|link)\b([^>]*)>(.*?)(?:</a>|)", re.IGNORECASE | re.DOTALL)
_HREF = re.compile(r"href\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
_DOCS_WORDS = re.compile(r"(?i)\b(?:api|apis|developers?|developer\s+docs|rest\s+api)\b")
_DOCS_PATH = re.compile(r"(?i)(?:^|[/.])(?:api|apis|developers?|dev)(?:[/.-]|$)|/docs/api")


def _links(page_html: str, base_url: str) -> tuple[list[str], list[str]]:
    """(spec links, developer-docs links) on a page."""
    specs, docs = [], []
    for tag, attrs, text in _LINK.findall(page_html[:500_000]):
        href = _HREF.search(attrs)
        if not href:
            continue
        url = urllib.parse.urljoin(base_url, html_lib.unescape(href.group(1)))
        if not url.startswith(("http://", "https://")):
            continue
        rel = re.search(r"rel\s*=\s*[\"']([^\"']+)", attrs, re.IGNORECASE)
        lowered = url.lower()
        if (rel and "service-desc" in rel.group(1).lower()) or (
                re.search(r"(openapi|swagger)[^/]*\.(json|ya?ml)$", lowered)):
            specs.append(url)
        elif tag.lower() == "a" and (_DOCS_WORDS.search(re.sub(r"<[^>]+>", " ", text)) or
                                     _DOCS_PATH.search(urllib.parse.urlparse(url).netloc + "/" +
                                                       urllib.parse.urlparse(url).path)):
            docs.append(url)
    return list(dict.fromkeys(specs)), list(dict.fromkeys(docs))


_API_URL = re.compile(r"https?://[^\s\"'<>()`]+")


def _docs_description(docs_url: str, page_html: str, page_url: str, found_via: str
                      ) -> ApiDescription | None:
    """A developer-docs page without a machine-readable spec: its text is the description and
    API URLs it shows (on the site's own domain) are the API's domains."""
    from .knowledge import html_to_text

    site = netloc(page_url)
    text = html_to_text(page_html)[1] if "<" in page_html[:2000] else page_html
    domains = []
    for found in _API_URL.findall(html_lib.unescape(page_html)):
        parsed = urllib.parse.urlparse(found)
        host = parsed.hostname or ""
        api_like = host.startswith("api.") or re.search(r"(?i)/(?:api|v\d+)(?:/|$)", parsed.path)
        if official(host, site) and api_like:
            domains.append(netloc(found))
    domains = list(dict.fromkeys(domains))
    if not domains:
        return None
    return ApiDescription(site=site, docs_url=docs_url, spec_url="", kind="docs",
                          found_via=found_via, api_domains=domains,
                          excerpt=text.strip()[:EXCERPT_CHARS],
                          rate_limits=parse_rate_limits(text))


def discover(page_url: str, get: web.ApiGet, *, search: Callable[[str], list[web.Result]] | None
             = None, directory: str | None = None, notify: Notify | None = None
             ) -> ApiDescription | None:
    """The site's official API description, or None. Sources, in order: the site's own spec
    paths, api.<domain>, APIs.guru, developer-docs links on the site, web_search results on the
    site's own domain."""
    say = notify or (lambda message: None)
    parsed = urllib.parse.urlparse(page_url)
    site = netloc(page_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    base = base_domain(parsed.hostname or "")

    def get_ok(url: str, accept: str = "application/json") -> web.ApiResponse | None:
        try:
            response = get(url, {"Accept": accept})
        except web.BackendError:
            return None
        return response if response.status == 200 and response.body else None

    def try_spec(url: str, found_via: str, need_official: bool) -> ApiDescription | None:
        response = get_ok(url, "application/json, application/yaml, text/yaml, */*")
        spec = _parse_spec(response.text()) if response else None
        if spec is None:
            return None
        return _describe_spec(spec, url, page_url, found_via, need_official)

    # 1. the site's own description: RFC 9727 api-catalog, well-known and common paths
    catalog = get_ok(f"{root}/.well-known/api-catalog", "application/linkset+json, */*")
    if catalog:
        with contextlib.suppress(ValueError, AttributeError, TypeError):
            for entry in json.loads(catalog.text()).get("linkset", []):
                for link in entry.get("service-desc", []):
                    found = try_spec(urllib.parse.urljoin(root, link.get("href", "")),
                                     "well-known", False)
                    if found:
                        say(f"Found {site}'s API description at {found.spec_url}")
                        return found
    for path, via in (("/.well-known/openapi.json", "well-known"),
                      ("/.well-known/openapi.yaml", "well-known"), ("/openapi.json", "site path"),
                      ("/swagger.json", "site path"), ("/api/openapi.json", "site path"),
                      ("/api/swagger.json", "site path")):
        found = try_spec(root + path, via, False)
        if found:
            say(f"Found {site}'s API description at {found.spec_url}")
            return found
    if not re.fullmatch(r"[\d.]+", base) and not (parsed.hostname or "").startswith("api."):
        for path in ("/openapi.json", "/swagger.json"):
            found = try_spec(f"https://api.{base}{path}", "api subdomain", False)
            if found:
                say(f"Found {site}'s API description at {found.spec_url}")
                return found
    # 2. the public API directory (APIs.guru): only specs whose servers are on the site's domain
    listing = get_ok(f"{(directory or APIS_GURU).rstrip('/')}/{base}.json")
    if listing:
        with contextlib.suppress(ValueError, AttributeError, TypeError):
            for api in (json.loads(listing.text()).get("apis") or {}).values():
                version = (api.get("versions") or {}).get(api.get("preferred", ""), {})
                spec_url = version.get("swaggerUrl") or version.get("openapiUrl") or ""
                found = try_spec(spec_url, "directory", True) if spec_url else None
                if found:
                    say(f"Found {site}'s API in the APIs.guru directory: {found.spec_url}")
                    return found

    # 3. developer-docs links on the site, 4. docs pages on the site's domain from web_search
    def from_docs(urls: list[str], via: str) -> ApiDescription | None:
        for docs_url in urls[:3]:
            response = get_ok(docs_url, "text/html, */*")
            if response is None:
                continue
            page_html = response.text()
            spec = _parse_spec(page_html)
            if spec is not None:
                found = _describe_spec(spec, docs_url, page_url, via, False)
                if found:
                    return found
            spec_links, _ = _links(page_html, docs_url)
            for spec_url in spec_links[:3]:
                if official(urllib.parse.urlparse(spec_url).hostname or "", site):
                    found = try_spec(spec_url, via, False)
                    if found:
                        return found
            found = _docs_description(docs_url, page_html, page_url, via)
            if found:
                return found
        return None

    home = get_ok(root + "/", "text/html, */*")
    if home:
        spec_links, docs_links = _links(home.text(), root + "/")
        for spec_url in spec_links[:3]:
            if official(urllib.parse.urlparse(spec_url).hostname or "", site):
                found = try_spec(spec_url, "docs link", False)
                if found:
                    say(f"Found {site}'s API description at {found.spec_url}")
                    return found
        docs_links = [url for url in docs_links
                      if official(urllib.parse.urlparse(url).hostname or "", site)]
        found = from_docs(docs_links, "docs link")
        if found:
            say(f"Found {site}'s developer docs at {found.docs_url}")
            return found
    if search is not None:
        try:
            results = search(f"{base} official API documentation")
        except web.BackendError:
            results = []
        docs_urls = [result.url for result in results
                     if official(urllib.parse.urlparse(result.url).hostname or "", site)]
        found = from_docs(docs_urls, "search")
        if found:
            say(f"Found {site}'s developer docs at {found.docs_url} (web search)")
            return found
    return None


# -- the prompt --------------------------------------------------------------------------------

EXAMPLES = (
    ("stackoverflow.com/questions/<id> (and other Stack Exchange sites)", "Stack Exchange API",
     ("_stack_exchange", "_se_call")),
    ("github.com/<owner>/<repo>/issues/<n> and /pull/<n>", "GitHub REST API", ("_github_issue",)),
    ("pypi.org/project/<name>/[<version>]", "PyPI JSON API", ("_pypi",)),
    ("npmjs.com/package/<name>[/v/<version>]", "npm registry API", ("_npm",)),
    ("crates.io/crates/<name>[/<version>]", "crates.io API", ("_crates",)),
    ("<lang>.wikipedia.org/wiki/<title>", "Wikipedia REST and Action APIs", ("_wikipedia",)),
    ("developer.mozilla.org/<locale>/docs/<slug>", "MDN document API", ("_mdn",)),
)

SYSTEM = (
    "You write site adapters for Vega, a coding agent that reads web pages for its user. An "
    "adapter reads one site's pages through the site's OFFICIAL API instead of its HTML. You "
    "write it like any coding task: one complete Python module, then the harness runs it and "
    "tells you what failed."
)

RULES = """Write one Python module with exactly this interface:

MATCH = r"..."  # a regex that matches this page's URL and the site's other pages of the same kind
def fetch(url, api_get):
    ...  # return a Page, or None when the API says the page does not exist

Rules:
- api_get(url, headers=None) is the ONLY way to reach the network. It returns an ApiResponse with
  .status (int), .headers (dict, lower-case names), .text() and .json(what). HTTP errors come
  back as a status; check it. It only reaches these API domains: {domains}.
- Never send credentials, cookies, API keys or a browser User-Agent; never fetch the HTML page.
- Available without import: Page, ApiResponse, _source_page, _day, _html_text, _unexpected,
  BackendError (as in the examples). You may import only: json, re, html, urllib.parse, datetime,
  contextlib, math, itertools, collections, textwrap, string, typing.
- No files, no os/sys/subprocess/socket, no eval/exec/open, no global state beyond constants.
- Return _source_page(url, title, date, "<API name>", body) like the examples: url is the URL
  asked for; title is the page's title as a reader sees it; date is the publication date as
  YYYY-MM-DD (_day() converts ISO times and Unix times) or "unknown"; body is readable text
  (prose lines, the article's or entry's content, converted from HTML with _html_text when the
  API returns HTML), never a raw JSON dump.
- Put the module in ONE ```python code block. No explanations outside it."""


def _example_sources() -> str:
    blocks = []
    for number, (pattern, api, names) in enumerate(EXAMPLES, 1):
        code = "\n\n".join(inspect.getsource(getattr(web, name)).rstrip() for name in names)
        blocks.append(f"Example {number}: {pattern} through the {api}\n```python\n{code}\n```")
    helpers = "\n\n".join(inspect.getsource(getattr(web, name)).rstrip()
                          for name in ("_source_page", "_day"))
    return ("The 7 adapters Vega already has (each is called with the page URL and api_get; "
            "yours is one module with MATCH and fetch):\n\n" + "\n\n".join(blocks)
            + f"\n\nThe helpers they use:\n```python\n{helpers}\n```")


def generation_messages(page_url: str, reason: str, api: ApiDescription,
                        expected_title: str = "", previous: str = "",
                        previous_failure: str = "") -> list[dict[str, str]]:
    """The first prompt for an adapter (vega's worked examples, the API description excerpt and
    the page URL)."""
    source = (f"{api.spec_url} (OpenAPI/Swagger description)" if api.spec_url
              else f"{api.docs_url} (developer documentation)")
    parts = [
        f"Page to read: {page_url}",
        f"Why: {reason}",
    ]
    if expected_title:
        parts.append(f"Its title in search results: {expected_title}")
    parts += [
        "",
        RULES.format(domains=", ".join(api.api_domains)),
        "",
        f"The site's official API, from {source}:",
        api.excerpt,
        "",
        _example_sources(),
    ]
    if previous:
        parts += ["", "An earlier adapter for this site stopped working:",
                  f"```python\n{previous}\n```", previous_failure,
                  "Write a corrected module."]
    else:
        parts += ["", f"Now write the adapter module for {api.site}."]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(parts)}]


def repair_message(code: str, feedback: str) -> dict[str, str]:
    return {"role": "user", "content": (
        "The harness ran your adapter and it failed verification:\n\n" + feedback
        + "\n\nFix the module and give the whole corrected module in one ```python code block.")}


def extract_code(reply: str) -> str:
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", reply or "", re.DOTALL)
    if blocks:
        return max(blocks, key=lambda block: ("def fetch" in block, len(block))).strip() + "\n"
    return (reply or "").strip() + "\n"


# -- static checks -----------------------------------------------------------------------------

ALLOWED_IMPORTS = {"json", "re", "html", "urllib", "urllib.parse", "datetime", "contextlib",
                   "math", "itertools", "collections", "textwrap", "string", "typing",
                   "__future__"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                   "input", "breakpoint", "memoryview", "setattr", "delattr"}


def static_problems(code: str, page_url: str) -> list[str]:
    """Problems found without running the module (it still only ever runs in the sandbox)."""
    if len(code) > CODE_LIMIT:
        return [f"the module is {len(code)} characters; keep it under {CODE_LIMIT}"]
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"the module does not parse: {exc.msg} (line {exc.lineno})"]
    problems: list[str] = []
    match_value = None
    has_fetch = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MATCH"
                                                for t in node.targets):
            with contextlib.suppress(ValueError):
                match_value = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == "fetch":
            has_fetch = len(node.args.args) >= 2
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else \
                [alias.name for alias in node.names]
            for name in names:
                if name not in ALLOWED_IMPORTS:
                    problems.append(f"import of {name!r} is not allowed (allowed: "
                                    + ", ".join(sorted(ALLOWED_IMPORTS - {"__future__"})) + ")")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"{node.id} is not allowed in an adapter")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and \
                node.attr not in {"__name__"}:
            problems.append(f"attribute {node.attr} is not allowed in an adapter")
    if not isinstance(match_value, str):
        problems.append("MATCH must be a module-level string (a regex for the page URLs)")
    else:
        try:
            if not re.search(match_value, page_url):
                problems.append(f"MATCH {match_value!r} does not match the page URL {page_url}")
        except re.error as exc:
            problems.append(f"MATCH is not a valid regex: {exc}")
    if not has_fetch:
        problems.append("the module must define fetch(url, api_get)")
    return list(dict.fromkeys(problems))


# -- the sandboxed run -------------------------------------------------------------------------

@dataclass
class UrlResult:
    url: str
    page: dict[str, str] | None
    error: str = ""


@dataclass
class RunReport:
    results: list[UrlResult] = field(default_factory=list)
    calls: list[tuple[str, int]] = field(default_factory=list)  # (url, status) the proxy sent
    refused: list[str] = field(default_factory=list)  # URLs outside the allowlist
    rate_limited: str = ""  # why the API stopped answering, when it did
    failure: str = ""  # the run itself failed (timeout, crash)

    def page(self, url: str) -> web.Page | None:
        for result in self.results:
            if result.url == url and result.page is not None:
                return web.Page(url=url, title=result.page["title"], date=result.page["date"],
                                text=result.page["text"])
        return None


_RUNNER = ("import sys; sys.path.insert(0, {src!r}); "
           "from vega.adapter_runner import main; main()")


def _command() -> list[str]:
    src = str(Path(__file__).resolve().parent.parent)
    argv = [sys.executable, "-I", "-B", "-c", _RUNNER.format(src=src)]
    from . import sandbox

    return sandbox.readonly_argv(argv)


def run_adapter(code: str, urls: list[str], get: web.ApiGet, domains: list[str], *,
                limits: dict[str, Any] | None = None, timeout: float = RUN_TIMEOUT) -> RunReport:
    """Run the adapter on `urls` in the sandboxed child; serve its api_get calls through the
    allowlist, the rate limiter and the real `get`."""
    report = RunReport()
    try:
        process = subprocess.Popen(_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                   cwd=str(Path(__file__).resolve().parent.parent.parent),
                                   env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1",
                                        "PYTHONIOENCODING": "utf-8"})
    except OSError as exc:
        report.failure = f"the sandbox could not start ({exc})"
        return report
    timer = threading.Timer(timeout, process.kill)
    timer.start()
    stderr_lines: list[str] = []
    reader = threading.Thread(target=lambda: stderr_lines.extend(process.stderr or []),
                              daemon=True)
    reader.start()
    per_url: dict[str, int] = {}
    current = urls[0] if urls else ""
    try:
        assert process.stdin and process.stdout
        process.stdin.write(json.dumps({"code": code, "urls": urls}) + "\n")
        process.stdin.flush()
        for line in process.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if message.get("op") == "result":
                report.results.append(UrlResult(str(message.get("url")), message.get("page"),
                                                str(message.get("error") or "")))
                done = len(report.results)
                current = urls[done] if done < len(urls) else ""
                continue
            if message.get("op") != "get":
                continue
            reply = _serve(str(message.get("url", "")), dict(message.get("headers") or {}), get,
                           domains, limits, report, per_url, current)
            process.stdin.write(json.dumps(reply) + "\n")
            process.stdin.flush()
    except (BrokenPipeError, OSError) as exc:
        report.failure = f"the sandboxed run stopped ({exc})"
    finally:
        timer.cancel()
        with contextlib.suppress(Exception):
            process.wait(timeout=5)
        if process.poll() is None:
            process.kill()
        reader.join(timeout=2)
    if len(report.results) < len(urls) and not report.failure:
        killed = process.returncode is not None and process.returncode < 0
        report.failure = (f"the run took longer than {timeout:g} s and was stopped" if killed
                          else "the adapter process ended early:\n"
                          + "".join(stderr_lines[-12:]).strip())
    return report


def _serve(url: str, headers: dict[str, str], get: web.ApiGet, domains: list[str],
           limits: dict[str, Any] | None, report: RunReport, per_url: dict[str, int],
           current: str) -> dict[str, Any]:
    if not domain_allowed(url, domains):
        report.refused.append(url)
        host = urllib.parse.urlparse(url).netloc or url
        return {"error": f"refused: {host} is not one of this adapter's API domains "
                         f"({', '.join(domains)}); api_get only reaches those."}
    per_url[current] = per_url.get(current, 0) + 1
    if per_url[current] > MAX_CALLS_PER_URL:
        return {"error": f"refused: more than {MAX_CALLS_PER_URL} API requests for one page"}
    domain = netloc(url)
    refusal = LIMITER.wait(domain, limits)
    if refusal:
        report.rate_limited = refusal
        return {"error": f"refused: {refusal}"}
    clean = {key: value for key, value in headers.items() if not _DROPPED_HEADER.match(key)}
    try:
        response = get(url, clean)
    except web.BackendError as exc:
        report.calls.append((url, 0))
        return {"error": str(exc)}
    report.calls.append((url, response.status))
    if LIMITER.observe(domain, response):
        report.rate_limited = f"the API at {domain} answered HTTP {response.status} (rate limit)"
    return {"status": response.status, "body": base64.b64encode(response.body).decode(),
            "headers": response.headers, "truncated": response.truncated}


@dataclass
class Target:
    url: str
    title: str  # the title search results gave it ("" when unknown)


def verify(code: str, targets: list[Target], get: web.ApiGet, domains: list[str],
           limits: dict[str, Any] | None = None) -> tuple[bool, str, RunReport]:
    """(passed, feedback for the model, report). Feedback names every failed URL and why."""
    report = run_adapter(code, [target.url for target in targets], get, domains, limits=limits)
    lines: list[str] = []
    if report.failure:
        lines.append(f"The run failed: {report.failure}")
    if report.refused:
        lines.append("api_get refused requests outside the API domains ("
                     + ", ".join(domains) + "): " + ", ".join(report.refused[:5]))
    by_url = {result.url: result for result in report.results}
    passed = 0
    for target in targets:
        result = by_url.get(target.url)
        if result is None:
            lines.append(f"- {target.url}: not run")
            continue
        problems = [result.error] if result.error else check_page(result.page, target.title)
        if problems:
            lines.append(f"- {target.url}: " + "; ".join(problems))
        else:
            passed += 1
    ok = passed == len(targets) and not report.refused and not report.failure
    calls = ", ".join(f"{url} -> HTTP {status}" for url, status in report.calls[-8:])
    if not ok and calls:
        lines.append(f"API requests made (last {min(len(report.calls), 8)}): {calls}")
    summary = f"{passed} of {len(targets)} pages passed."
    return ok, summary + ("\n" + "\n".join(lines) if lines else ""), report


# -- approval ----------------------------------------------------------------------------------

@dataclass
class Candidate:
    site: str
    match: str
    code: str
    api: ApiDescription
    targets: list[Target]
    reads: list[str]  # endpoint shapes the adapter requested in verification
    model: str
    rounds: int

    @property
    def code_size(self) -> str:
        lines = self.code.count("\n") + (0 if self.code.endswith("\n") else 1)
        return f"{lines} lines, {len(self.code.encode()) / 1000:.1f} KB"


def endpoint_shapes(calls: list[tuple[str, int]]) -> list[str]:
    """GET host/path templates (numbers and long ids as {id}) the adapter requested."""
    shapes = []
    for url, _ in calls:
        parsed = urllib.parse.urlparse(url)
        path = re.sub(r"/(?:\d+|[0-9a-f]{8,}|[A-Za-z0-9_-]{24,})(?=/|$)", "/{id}", parsed.path)
        shapes.append(f"GET {netloc(url)}{path}" + ("?..." if parsed.query else ""))
    shapes = list(dict.fromkeys(shapes))
    # several pages of one endpoint (/articles/a, /articles/b): one {id} shape
    heads: dict[str, int] = {}
    for shape in shapes:
        head = shape.rsplit("/", 1)[0]
        heads[head] = heads.get(head, 0) + 1
    return list(dict.fromkeys(
        (shape.rsplit("/", 1)[0] + "/{id}" + ("?..." if shape.endswith("?...") else ""))
        if heads[shape.rsplit("/", 1)[0]] > 1 else shape for shape in shapes))


def approval_text(candidate: Candidate) -> tuple[str, str]:
    """(title, body) of the approval question, shared by the TUI dialog and the terminal."""
    api = candidate.api
    where = api.spec_url or api.docs_url
    pages = len(candidate.targets)
    body = "\n".join([
        f"Vega wrote an adapter to read {candidate.site} through its official API. It passed "
        f"verification on {pages} page{'s' if pages != 1 else ''}"
        + (" (no sibling pages found: the blocked page only)" if pages == 1 else "") + ".",
        f"Site:        {candidate.site} (pages matching {candidate.match})",
        f"API domains: {', '.join(api.api_domains)}",
        f"API docs:    {where} (found via {api.found_via})",
        "Reads:       " + ("; ".join(candidate.reads[:6]) or "nothing during verification"),
        f"Rate limits: {describe_limits(api.rate_limits)}",
        (f"Code:        {candidate.code_size}, written by {candidate.model} in "
         f"{candidate.rounds} round{'s' if candidate.rounds != 1 else ''}; it runs sandboxed "
         "with network only to the API domains, no credentials"),
        "Verified on: " + ", ".join(target.url for target in candidate.targets),
        (f"Approve to store it in {adapters_dir() / (site_key(candidate.site) + '.py')} and use "
         "it now. Deny: nothing is stored."),
    ])
    return "Allow a new site adapter?", body


def console_approve(candidate: Candidate) -> bool:
    """The plain terminal's approval (line mode): approve, view the code, or deny."""
    from .ui import console

    title, body = approval_text(candidate)
    while True:
        console.print(f"\n[bold yellow]{title}[/bold yellow]")
        console.print(body, markup=False, highlight=False)
        try:
            answer = console.input("[yellow]\\[a] approve  \\[v] view code  \\[N] deny: "
                                   "[/yellow]").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if answer in {"v", "view"}:
            console.print(candidate.code, markup=False, highlight=False)
            continue
        return answer in {"a", "approve", "y", "yes"}


# -- storage -----------------------------------------------------------------------------------

@dataclass
class Stored:
    site: str
    match: str
    code: str
    meta: dict[str, Any]

    @property
    def domains(self) -> list[str]:
        return list(self.meta.get("api_domains") or [])

    @property
    def limits(self) -> dict[str, Any] | None:
        return self.meta.get("rate_limits")

    @property
    def targets(self) -> list[Target]:
        return [Target(str(item.get("url")), str(item.get("title") or ""))
                for item in self.meta.get("verification") or [] if isinstance(item, dict)]


def _sha(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def store(candidate: Candidate) -> Stored:
    folder = adapters_dir()
    folder.mkdir(parents=True, exist_ok=True)
    key = site_key(candidate.site)
    api = candidate.api
    meta = {
        "site": candidate.site, "match": candidate.match, "api_domains": api.api_domains,
        "api_docs_url": api.docs_url, "api_spec_url": api.spec_url, "found_via": api.found_via,
        "rate_limits": api.rate_limits, "reads": candidate.reads,
        "created": _today(), "verified": _today(), "model": candidate.model,
        "rounds": candidate.rounds,
        "verification": [asdict(target) for target in candidate.targets],
        "approval": {"approved": True, "date": _today(), "by": "user"},
        "code_sha256": _sha(candidate.code), "status": "ok",
    }
    (folder / f"{key}.py").write_text(candidate.code, encoding="utf-8")
    (folder / f"{key}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return Stored(candidate.site, candidate.match, candidate.code, meta)


def load_all(include_unusable: bool = False) -> list[Stored]:
    """Stored adapters. Without include_unusable: only approved ones whose code is unchanged
    since approval and not marked failing."""
    folder = adapters_dir()
    found = []
    for meta_file in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        code_file = meta_file.with_suffix(".py")
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            code = code_file.read_text(encoding="utf-8")
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        usable = ((meta.get("approval") or {}).get("approved") is True
                  and meta.get("code_sha256") == _sha(code) and meta.get("status") == "ok")
        if usable or include_unusable:
            found.append(Stored(str(meta.get("site", meta_file.stem)), str(meta.get("match", "")),
                                code, meta))
    return found


def state_of(adapter: Stored) -> str:
    if (adapter.meta.get("approval") or {}).get("approved") is not True:
        return "not approved"
    if adapter.meta.get("code_sha256") != _sha(adapter.code):
        return "code changed since approval (not used)"
    return str(adapter.meta.get("status", "ok"))


def find(site_or_url: str, include_unusable: bool = False) -> Stored | None:
    key = site_key(site_or_url)
    for adapter in load_all(include_unusable):
        if site_key(adapter.site) == key:
            return adapter
    return None


def _update(adapter: Stored, **changes: Any) -> None:
    adapter.meta.update(changes)
    path = adapters_dir() / f"{site_key(adapter.site)}.json"
    with contextlib.suppress(OSError):
        path.write_text(json.dumps(adapter.meta, indent=2) + "\n", encoding="utf-8")


def remove(site: str) -> bool:
    key = site_key(site)
    removed = False
    for suffix in (".py", ".json"):
        path = adapters_dir() / f"{key}{suffix}"
        if path.exists():
            path.unlink()
            removed = True
    return removed


def matching(url: str) -> Stored | None:
    """The stored, usable adapter for this URL: its site and its MATCH."""
    site = site_key(url)
    for adapter in load_all():
        if site_key(adapter.site) != site:
            continue
        with contextlib.suppress(re.error):
            if re.search(adapter.match, url):
                return adapter
    return None


def reverify(adapter: Stored, get: web.ApiGet) -> tuple[bool, str]:
    """Run the stored adapter on its verification pages again; records the outcome."""
    ok, feedback, report = verify(adapter.code, adapter.targets, get, adapter.domains,
                                  adapter.limits)
    if report.rate_limited:
        return False, f"not re-verified: {report.rate_limited}"
    _update(adapter, status="ok" if ok else "failing",
            **({"verified": _today()} if ok else {"last_failure": feedback[:2000]}))
    return ok, feedback


def stored_reader(url: str) -> Callable[[web.ApiGet], Any] | None:
    """web.site_adapter's hook: a reader for a stored adapter covering `url`, or None."""
    adapter = matching(url)
    if adapter is None:
        return None

    def read(get: web.ApiGet) -> Any:
        report = run_adapter(adapter.code, [url], get, adapter.domains, limits=adapter.limits)
        page = report.page(url)
        result = report.results[0] if report.results else None
        if report.rate_limited:
            raise web.BlockedError(
                f"{web.FETCH_FAILED}: {report.rate_limited}, so {url} was not fetched through "
                f"{adapter.site}'s API. Try another search result, or ask the user to open {url} "
                "in their browser (in Vega's app: /open).", url)
        if page is not None and not check_page(result.page if result else None, ""):
            return _labelled(page, adapter)
        # it failed on this page: is the adapter broken, or is it this page?
        ok, _ = reverify(adapter, get)
        if ok and result is not None and "returned None" in result.error:
            return None
        if ok:
            raise web.BackendError(f"{web.FETCH_FAILED}: {adapter.site}'s API adapter could not "
                                   f"read {url} ({(result.error if result else report.failure)[:200]}).")
        return web.GENERIC  # marked failing: the ordinary fetch runs, and a block rebuilds it

    return read


def _labelled(page: web.Page, adapter: Stored | Candidate) -> web.Page:
    """The text's first line says where it came from (as the built-in adapters do)."""
    text = page.text
    domains = adapter.domains if isinstance(adapter, Stored) else adapter.api.api_domains
    if not text.startswith("Source:"):
        text = f"Source: {', '.join(domains)} (official API)\n\n{text}"
    first, _, rest = text.partition("\n")
    first += " - adapter written by Vega, approved by the user"
    text = (first + "\n" + rest)[: web.MAX_PAGE_CHARS + 200]
    return web.Page(url=page.url, title=page.title, date=page.date, text=text)


# -- the whole build ---------------------------------------------------------------------------

@dataclass
class BuildResult:
    status: str  # stored | denied | no_api | failed | rate_limited | no_model
    message: str
    page: web.Page | None = None
    adapter: Stored | None = None


def find_siblings(page_url: str, search: Callable[[str], list[web.Result]] | None,
                  known: dict[str, str], title: str) -> list[Target]:
    """Up to 3 other pages of the same kind on the same site (same host, same first path
    segment) with their search-result titles."""
    parsed = urllib.parse.urlparse(page_url)
    first = parsed.path.strip("/").split("/")[0]

    def same_kind(url: str) -> bool:
        other = urllib.parse.urlparse(url)
        return (netloc(url) == netloc(page_url) and other.path.strip("/").split("/")[0] == first
                and web.canonical_url(url) != web.canonical_url(page_url))

    found: dict[str, str] = {web.canonical_url(url): (url, name) for url, name in known.items()
                             if same_kind(url)}
    if search is not None and len(found) < MAX_SIBLINGS:
        words = " ".join(re.findall(r"[A-Za-z0-9]+", title)[:6])
        for query in (f"site:{parsed.hostname} {words}".strip(), f"site:{parsed.hostname}"):
            try:
                results = search(query)
            except web.BackendError:
                continue
            for result in results:
                if same_kind(result.url):
                    found.setdefault(web.canonical_url(result.url), (result.url, result.title))
            if len(found) >= MAX_SIBLINGS:
                break
    return [Target(url, name) for url, name in list(found.values())[:MAX_SIBLINGS]]


def build(page_url: str, reason: str, *, get: web.ApiGet, model: Model,
          approve: Approve | None = None, search: Callable[[str], list[web.Result]] | None
          = None, titles: dict[str, str] | None = None, notify: Notify | None = None,
          model_name: str = "the model", directory: str | None = None,
          previous: Stored | None = None) -> BuildResult:
    """Discover the official API, write an adapter, verify it (with repair rounds), ask the user,
    store it. The page comes back only when the user approved."""
    say = notify or (lambda message: None)
    titles = titles or {}
    site = netloc(page_url)
    say(f"No adapter reads {site}: looking for its official API")
    api = discover(page_url, get, search=search, directory=directory, notify=say)
    if api is None:
        return BuildResult("no_api", f"{site} has no official API Vega could find (checked its "
                           "OpenAPI/Swagger paths, the APIs.guru directory and developer-docs "
                           "pages on its own domain).")
    title = next((name for url, name in titles.items()
                  if web.canonical_url(url) == web.canonical_url(page_url)), "")
    targets = [Target(page_url, title), *find_siblings(page_url, search, titles, title)]
    messages = generation_messages(
        page_url, reason, api, title, previous.code if previous else "",
        str(previous.meta.get("last_failure", "")) if previous else "")
    feedback, code, report = "", "", RunReport()
    rounds = 0
    for rounds in range(1, REPAIR_ROUNDS + 2):
        say(f"Writing an adapter for {site} (round {rounds})" if rounds == 1 else
            f"Repairing the {site} adapter (round {rounds}): {feedback.splitlines()[0]}")
        try:
            reply = model(messages)
        except Exception as exc:  # noqa: BLE001 - a model that cannot answer ends the build
            return BuildResult("failed", f"the model could not write the adapter ({exc}).")
        messages.append({"role": "assistant", "content": reply})
        code = extract_code(reply)
        problems = static_problems(code, page_url)
        if problems:
            feedback = "The module was not run:\n- " + "\n- ".join(problems)
            messages.append(repair_message(code, feedback))
            continue
        ok, feedback, report = verify(code, targets, get, api.api_domains, api.rate_limits)
        if report.rate_limited and not ok:
            return BuildResult("rate_limited", f"{site}'s API is rate-limiting this machine "
                               f"({report.rate_limited}); the adapter was not finished.")
        if ok:
            break
        messages.append(repair_message(code, feedback))
    else:
        return BuildResult("failed", f"the adapter for {site} failed verification after "
                           f"{REPAIR_ROUNDS} repair rounds ({feedback.splitlines()[0]}).")
    match = str(ast.literal_eval(next(
        node.value for node in ast.parse(code).body if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "MATCH" for t in node.targets))))
    candidate = Candidate(site=site, match=match, code=code, api=api, targets=targets,
                          reads=endpoint_shapes(report.calls), model=model_name, rounds=rounds)
    say(f"The {site} adapter passed verification on {len(targets)} pages; asking you to approve it")
    try:
        approved = (approve or console_approve)(candidate)
    except Exception:  # noqa: BLE001 - an approval that cannot be asked is a "no"
        approved = False
    if not approved:
        return BuildResult("denied", f"the user did not approve the adapter for {site}; nothing "
                           "was stored.")
    if previous is not None:
        remove(previous.site)
    stored = store(candidate)
    page = report.page(page_url)
    return BuildResult("stored", f"Adapter for {site} stored in {adapters_dir()}.",
                       _labelled(page, stored) if page else None, stored)
