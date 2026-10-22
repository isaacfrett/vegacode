"""web_facet_fetch: point the fetch-first and citation notes at the pages that decide a task.

Why (2026-10-18): the fetch-first notes named the top 2 search results, the model fetched
exactly those, and the decisive page lower in the ranking (an option's release news, a dated
changelog entry) stayed unread. Rules only, no model:

- A task splits into facets: each option it names whose own site is among the results (a host
  label starts with the word and that result's title names it: `alphadb`, `betadb`), each
  version, date or identifier it states (`5.0`, `v2`, `2024-06-01`, `CVE-2024-12345`, `AB1234`)
  that a result mentions, and its topic words that an official result's title or URL path holds
  (`licence`, `history`). The repository files read this turn add values and topic words
  (a platform doc's "Python 3.12", a policy doc's "licensed").
- suggest(): for each facet that no page read covers, the best unfetched result that mentions it:
  official site first, then the newest, then the one covering most still-open facets, then the
  search rank. Options first; then, for each option, its newest changelog/news/release/advisory/
  pricing page when that is at least as new as every page of the option already read or picked
  (by the version in its title/URL when both have one, else by date); then values and topic
  words (official results only, when any result is official). At most MAX_SUGGESTIONS; never fewer than the old top 2 (the
  caller pads with them).
- At the finish (official results only, no topic words): the task's options and values, and the
  versions/dates the answer and the files written this turn state, that no page read covers.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass

from . import features

MAX_SUGGESTIONS = 4


def normalise(text: str) -> str:
    return text.lower().replace("pricing", "price")


def _path(url: str) -> str:
    return urllib.parse.urlparse(url).path if "://" in url else url.partition("/")[2]


@dataclass(frozen=True)
class Hit:
    """A search result (or a fetched page) as the rules see it."""

    url: str
    title: str
    snippet: str
    date: str  # YYYY-MM-DD or "unknown"
    rank: int  # order in this turn's searches (each search's first result, then second...)
    official: bool

    @property
    def text(self) -> str:
        """URL, title and snippet: what option and value facets are matched against."""
        return normalise(f"{self.url} {self.title} {self.snippet}")

    @property
    def head(self) -> str:
        """Title and URL path: what a topic word must appear in (a snippet's "so you can verify
        it" does not make a page about verification)."""
        return normalise(f"{self.title} {_path(self.url)}")

    @property
    def known_date(self) -> str:
        return self.date if re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.date or "") else ""


@dataclass(frozen=True)
class Facet:
    label: str  # what a note names: "alphadb", "2024-06-01", "licensed"
    kind: str  # "option" | "value" | "topic"
    pattern: str  # regex over normalise()d text

    def matches(self, hit: Hit) -> bool:
        return re.search(self.pattern, hit.head if self.kind == "topic" else hit.text) is not None


@dataclass(frozen=True)
class Pick:
    url: str
    why: str  # the facet it covers ("alphadb", "2024-06-01", "alphadb: newer release page")
    official: bool


# words that never make a topic facet (task format words, English function words, URL parts)
_STOP_WORDS = """
about above according across actually after again against also always among another answer
answers anything around asked avoid based because been before being below best between blog
both break bring build built cannot change changes check choice choose chosen cite citing code
come could current currently data decide decides decision doc docs documentation does doing
done each either else ensure even every everything exact exactly example except fail fails
file files find first following forum found from further give given goes going good guide have
having here html http https index into item items json just keep keeps kept know largest last
later latest least like list look lowest made main make makes many mapping maybe meet meets
more most move moving much must name named names need needed needs never newest news next none
note number oldest once only other others ours over page pages part pass path point python
rather really reason reasons record reference relied relies rely reply require required
requirement requirements rest rests return returns right same says second seconds see seen
sentence sentences should show since some source sources step steps still such sure take task
tell than that their them then there these they thing things this those though through today
together tool tools tutorial under until update updated upon used uses using value values very
want wants well were what whatever when where whether which while whole will with within
without work works would write writes written your yours
"""
_STOP = frozenset(_STOP_WORDS.split())

_DATE = r"\b\d{4}-\d{2}-\d{2}\b"
_VERSION = r"(?<![\w.$])v?\d+(?:\.\d+)+(?![\w%]|\.\d)"
_VTAG = r"(?<![\w.])v\d+\b"
_IDENT = r"\b[A-Z][A-Z0-9]*-?\d{2,}(?:-\d+)*\b"
_RELEASE = re.compile(r"(?<![a-z])(?:changelog|change-log|news|releases?|released|advisor(?:y|ies)"
                      r"|security|price[sd]?|whats-?new|announce\w*)(?![a-z])")


def value_facet(value: str) -> Facet:
    """A version/date/identifier facet: the value as a whole token (`5.0` is not `5.0.0`)."""
    return Facet(value.lower(), "value",
                 r"(?<![\w.])" + re.escape(value.lower()) + r"(?![\w]|\.\d)")


def values_in(text: str) -> list[str]:
    """Versions, dates and identifiers stated in `text`, in order, without duplicates."""
    found = [m.group(0) for pattern in (_DATE, _IDENT, _VTAG, _VERSION)
             for m in re.finditer(pattern, text)]
    return list(dict.fromkeys(value.lower() for value in found))


def host_labels(url: str) -> list[str]:
    try:
        host = urllib.parse.urlparse(url if "://" in url else "https://" + url).hostname or ""
    except ValueError:
        return []
    return host.lower().removeprefix("www.").split(".")[:-1]


def _word(word: str) -> str:
    return r"(?<![a-z0-9])" + re.escape(word)


def _stem(word: str) -> str:
    word = normalise(word)
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word[:min(len(word), max(5, len(word) - 3))]


def option_words(task: str, hits: list[Hit], generic: frozenset[str] = frozenset()) -> list[str]:
    """Task words whose own site is among the hits: a host label starts with the word and that
    hit's title names it (`alphadb` -> alphadb.example "alphadb 2.3 released"; not `lint` ->
    lintdocs.example "fastlint pyproject.toml config")."""
    words = []
    for word in dict.fromkeys(re.findall(r"[a-z][a-z0-9]{3,}", task.lower())):
        if word in generic or word in _STOP:
            continue
        if any(any(label.startswith(word) for label in host_labels(hit.url))
               and re.search(_word(word), normalise(hit.title)) for hit in hits):
            words.append(word)
    return words


def facets(task: str, context: Iterable[str], hits: list[Hit],
           generic: frozenset[str] = frozenset()) -> list[Facet]:
    """The task's facets that at least one hit matches (see the module docstring), options
    first, then values, then topic words; a topic word matching exactly the hits an earlier topic
    word matches is dropped."""
    context_text = "\n".join(context)
    options = option_words(task, hits, generic)
    out = [Facet(word, "option", _word(word)) for word in options]
    out += [value_facet(value) for value in values_in(task) + values_in(context_text)]
    words = re.findall(r"[a-z][a-z0-9]{3,}", f"{task}\n{context_text}".lower())
    stems: dict[str, str] = {}  # stem -> the first word that gave it (the label)
    for word in words:
        if word not in _STOP and word not in options:
            stems.setdefault(_stem(word), word)
    out += [Facet(word, "topic", _word(stem)) for stem, word in stems.items()]
    kept: list[Facet] = []
    seen: set[frozenset[int]] = set()
    for facet in out:
        pool = [hit for hit in hits if hit.official] if facet.kind == "topic" else hits
        matched = frozenset(hit.rank for hit in pool if facet.matches(hit))
        if not matched or any(f.label == facet.label for f in kept):
            continue
        if facet.kind == "topic":
            if matched in seen:
                continue
            seen.add(matched)
        kept.append(facet)
    return kept


def _version(hit: Hit) -> tuple[int, ...]:
    """The highest dotted version in the hit's title or URL path (not the snippet: "alphadb 2.2:
    Python 3.12 is not supported yet" is a 2.2 page)."""
    found = [tuple(int(part) for part in m.group(0).lstrip("v").split("."))
             for m in re.finditer(_VERSION, f"{hit.title} {_path(hit.url)}")]
    return max(found, default=())


def release_like(hit: Hit) -> bool:
    """A changelog, news, release, advisory or pricing page (by its title or URL path)."""
    return _RELEASE.search(hit.head) is not None


def option_version(hit: Hit, option: str) -> tuple[int, ...]:
    """facet_same_option: the version of `option` a page is about: on the option's own site
    (a host label starts with it) the page's title/URL version; elsewhere only a version written
    right after the option's name ("widget 4 on gadget 2.6?" is a widget 4 page, not 2.6)."""
    if any(label.startswith(option) for label in host_labels(hit.url)):
        return _version(hit)
    found = [tuple(int(part) for part in m.group(1).split("."))
             for m in re.finditer(_word(option) + r"[\s/_-]*v?(\d+(?:\.\d+)*)(?![\w.]*\d)",
                                  normalise(f"{hit.title} {_path(hit.url)}"))]
    # "alpha-vs-beta-2025" names a year, not beta 2025
    return max((v for v in found if not (len(v) == 1 and v[0] >= 1900)), default=())


def _at_least_as_new(hit: Hit, base: list[Hit], option: str = "") -> bool:
    """Whether `hit` is a newer page than every page in `base`: by version when both have one,
    else by date. facet_same_option (with `option`): versions only of that option (2026-10-19: a
    "widget 3.10 released" blog post beat a forum thread titled with another package's 2.6), and
    never a page dated before the newest page of the option already read or picked."""
    if option and features.on("facet_same_option"):
        dates = [page.known_date for page in base if page.known_date]
        if hit.known_date and dates and hit.known_date < max(dates):
            return False
        mine = option_version(hit, option)
        versions = [option_version(page, option) for page in base]
        versions = [version for version in versions if version]
        if mine and versions:
            return mine > max(versions)
        return bool(hit.known_date) and hit.known_date >= max(dates, default="")
    versions = [_version(page) for page in base if _version(page)]
    if _version(hit) and versions:
        return _version(hit) > max(versions)
    return hit.known_date >= max(page.known_date for page in base) and bool(hit.known_date)


def suggest(found: list[Facet], hits: list[Hit], fetched: list[Hit], *,
            count: int = MAX_SUGGESTIONS, topics: bool = True,
            official_only: bool = False) -> list[Pick]:
    """The unfetched hits to read next, one per open facet (see the module docstring)."""
    fetched_urls = {hit.url for hit in fetched}
    candidates = [hit for hit in hits if hit.url not in fetched_urls
                  and (hit.official or not official_only)]
    covered = {facet for facet in found if any(facet.matches(hit) for hit in fetched)}
    picked: list[tuple[Hit, str]] = []

    def gain(hit: Hit) -> set[Facet]:
        return {facet for facet in found if facet not in covered and facet.matches(hit)}

    def key(hit: Hit) -> tuple:
        return (hit.official, hit.known_date, len(gain(hit)), -hit.rank)

    def pick(hit: Hit, why: str) -> None:
        picked.append((hit, why))
        covered.update(gain(hit))

    def open_pool(facet: Facet, official: bool = False) -> list[Hit]:
        taken = {hit.url for hit, _ in picked}
        return [hit for hit in candidates if hit.url not in taken and facet.matches(hit)
                and (hit.official or not official)]

    def cover(kind: str, official: bool = False) -> None:
        for facet in found:
            if facet.kind == kind and facet not in covered:
                pool = open_pool(facet, official)
                if pool:
                    pick(max(pool, key=key), facet.label)

    def newer_releases() -> None:
        for facet in found:
            if facet.kind != "option":
                continue
            pool = [hit for hit in open_pool(facet) if release_like(hit)]
            base = [hit for hit in [*fetched, *(h for h, _ in picked)] if facet.matches(hit)]
            if not pool or not base:
                continue
            newest = max(pool, key=lambda hit: (hit.known_date, _version(hit), -hit.rank))
            if _at_least_as_new(newest, base, facet.label):
                pick(newest, f"{facet.label}: newer release/news page")

    # a value or topic word that only a forum or an aggregator mentions is not worth a fetch
    # (a "2.6" only in a forum thread's title), unless no result at all is official
    any_official = any(hit.official for hit in candidates)
    cover("option")
    newer_releases()
    cover("value", official=any_official)
    if topics:
        cover("topic", official=True)
    return [Pick(hit.url, why, hit.official) for hit, why in picked[:count]]
