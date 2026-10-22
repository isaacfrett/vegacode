"""The knowledge library: learning files, folders and pages, sections, search, and the agent tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_agent_loop import agent_with, call, text

from vega import features, knowledge
from vega.agent import knowledge_section
from vega.tools import tool_specs

BOOK = """# Zorblax Handbook

## Installing

Install the zorblax package with `zpm get zorblax`. It needs Zorb 4.2 or later.

## Retry policy

Zorblax clients retry a failed request exactly 7 times, waiting 250 milliseconds between tries.
The setting is called `retry_budget` and lives in zorblax.toml.

## Licensing

Zorblax is released under the Quokka Public License.
"""
PAGE = """<!doctype html><html><head><title>Flux capacitor guide</title>
<script>var tracking = 1;</script></head><body><nav>Home | About</nav>
<main><h1>Flux capacitor</h1><p>The capacitor needs   1.21 gigawatts.</p>
<h2>Timing</h2><p>Activation happens at 88 miles per hour.</p><pre>  flux.start(speed=88)
  flux.stop()</pre></main><footer>Copyright</footer></body></html>"""


def test_learn_markdown_file_keeps_sections_and_finds_facts(tmp_path: Path) -> None:
    book = tmp_path / "handbook.md"
    book.write_text(BOOK)
    [source] = knowledge.learn(str(book))
    assert source.title == "Zorblax Handbook" and source.kind == "text" and source.chunks >= 3
    hits = knowledge.search("how many times does zorblax retry")
    assert "7 times" in hits[0][1].text
    assert hits[0][1].location == "section Zorblax Handbook > Retry policy"
    result = knowledge.search_text("retry_budget setting")
    assert result.startswith("[Zorblax Handbook — section Zorblax Handbook > Retry policy]")
    assert "Cite the source" in result


def test_html_extraction_drops_scripts_and_navigation(tmp_path: Path) -> None:
    title, body = knowledge.html_to_text(PAGE)
    assert title == "Flux capacitor guide"
    assert "tracking" not in body and "Home | About" not in body and "Copyright" not in body
    assert "# Flux capacitor" in body and "1.21 gigawatts" in body
    assert "\u00b6" not in knowledge.html_to_text("<h2>Usage<a>\u00b6</a></h2>")[1]
    assert "  flux.start(speed=88)\n  flux.stop()" in body  # pre keeps its layout


def test_learn_folder_relearn_replaces_and_forget(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(BOOK)
    (docs / "page.html").write_text(PAGE)
    (docs / "skip.bin").write_bytes(b"\x00\x01")
    added = knowledge.learn(str(docs))
    assert sorted(s.kind for s in added) == ["html", "text"]
    knowledge.learn(str(docs / "a.md"))  # learning again replaces, never duplicates
    assert len(knowledge.sources()) == 2
    assert len({c.text for c in knowledge.chunks()}) == len(knowledge.chunks())
    removed = knowledge.forget(added[0].id[:6])
    assert removed is not None and len(knowledge.sources()) == 1


def test_long_text_is_windowed_with_overlap() -> None:
    text = " ".join(f"Sentence number {n} talks about topic {n % 7}." for n in range(400))
    pieces = knowledge._windows(text)
    assert len(pieces) > 5 and all(len(p) <= knowledge.CHUNK_CHARS for p in pieces)
    assert all(pieces[i][-40:] in text for i in range(len(pieces)))


def test_errors_are_clear(tmp_path: Path) -> None:
    with pytest.raises(knowledge.LearnError, match="Not a URL, file or folder"):
        knowledge.learn(str(tmp_path / "missing.md"))
    (tmp_path / "x.exe").write_text("x")
    with pytest.raises(knowledge.LearnError, match="Unsupported file type"):
        knowledge.learn(str(tmp_path / "x.exe"))


def test_tool_and_prompt_appear_only_with_material(tmp_path: Path) -> None:
    assert knowledge_section() == ""
    (tmp_path / "h.md").write_text(BOOK)
    knowledge.learn(str(tmp_path / "h.md"))
    assert "Zorblax Handbook (text" in knowledge_section()
    names = [t["function"]["name"] for t in tool_specs(include_knowledge=True)]
    assert "search_knowledge" in names


def test_agent_answers_from_the_library(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "h.md").write_text(BOOK)
    knowledge.learn(str(tmp_path / "h.md"))
    project = tmp_path / "proj"
    project.mkdir()
    agent, seen = agent_with(project, [call("search_knowledge", query="zorblax retry"),
                                       text("7 retries (Zorblax Handbook, Retry policy).")])
    assert "Knowledge library" in agent.messages[0]["content"]
    agent.run("How many times does zorblax retry?")
    result = next(m["content"] for m in agent.messages if m.get("role") == "tool")
    assert "exactly 7 times" in result
    monkeypatch.setenv(features.ENV, "knowledge")  # switched off: no prompt section, no tool
    assert knowledge_section() == ""
