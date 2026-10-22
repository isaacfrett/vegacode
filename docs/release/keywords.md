# Keywords for the public `vegacode` repository

The search phrases the README, the GitHub description, the topics and the PyPI metadata should
carry, grouped by what the searcher wants. CI (`.github/scripts/check_keywords.py`) checks that
each **required** phrase still appears where this file says it should, so the README,
description and topics do not drift apart.

Rules for using them:

- Use each phrase naturally, once or twice. Repeating keywords hurts readability and does nothing
  for GitHub search, which matches the name, description, topics and README text.
- Every phrase must stay true of the code. If a feature is removed, remove its phrase too.
- No competitor product names in topics or the description. In the README body they appear only
  in the factual comparison section.

## 1. Category: "what kind of tool is this?"

People who know they want a coding agent and are browsing options.

| Phrase | Required in | Reasoning |
|---|---|---|
| local coding agent | README H1, README first paragraph, description | The most precise name for what Vega is; combines the category with the differentiator |
| AI coding assistant | README first paragraph, description, topic `ai-coding-assistant` | The most common phrasing for the category |
| coding agent | topic `coding-agent` | Category term used by recent agentic tools |
| AI agent | topics `ai-agent`, `ai-agents` | Broad agent audience |
| developer tools | topic `developer-tools` | Tool lists and awesome-lists filter on it |

## 2. Local and private: "I don't want my code in the cloud"

The strongest intent Vega serves. These searchers have a reason (privacy, policy, cost, no
network) and are likely to try it.

| Phrase | Required in | Reasoning |
|---|---|---|
| offline AI coding assistant | README first paragraph, description | Exact-intent phrase; "offline" is what people type when they need it to work without the cloud |
| runs on your laptop | README H1, description | Plain-language version of "local" that matches how people describe the need |
| code never leaves your machine | README first paragraph, description | The privacy promise in the words people use; true by default (local model, web research off) |
| local LLM | topic `local-llm` | Large topic; the people running models locally |
| local AI | topic `local-ai` | Alternative phrasing of the same intent |
| privacy | README "Why Vega", topic `privacy` | Broad, but it is the reason many of these users look |
| self-hosted | topic `self-hosted` | Self-hosting audience overlaps with local-first users |
| no API key, no account | README first paragraph | Answers the first question of someone comparing with hosted tools |

## 3. Model and runtime: "does it work with what I already run?"

People who already have Ollama or a local model server and want a tool on top.

| Phrase | Required in | Reasoning |
|---|---|---|
| Ollama | README first paragraph, description, topic `ollama` | The runtime most local-model users have installed; a large topic |
| open-weight LLM / open-source LLM | README first paragraph, description, topic `open-source-llm` | "open-weight" is accurate; "open-source LLM" is what people search |
| small models, under 10B, 16 GB | README first paragraph, "Why Vega", description | Matches searches for agents that work on ordinary laptops, and is Vega's design focus |
| Granite, Qwen | README "Default model" | Model-name searches; body text only, since the default may change |
| LM Studio, llama.cpp, llama-server, mlx_lm.server | README "Features" | Long-tail: users of other local servers looking for an agent that supports them |
| LLM | topic `llm` | Very large topic for broad discovery |

## 4. Interface and stack: "how do I use it?"

| Phrase | Required in | Reasoning |
|---|---|---|
| terminal | README first paragraph, description | Many developers search for terminal-based assistants specifically |
| CLI | README first paragraph, description, topic `cli` | Same audience, different word |
| TUI | README first paragraph, description, topic `tui` | Full-screen terminal app; a distinct, growing topic |
| Python | README first paragraph, description, topic `python` | Implementation language |
| Textual | topic `textual` | Textual users browse their framework's topic for showcase apps |

## 5. Capability: "can it do my task?"

Long-tail phrases that match what people want done. Body text only.

| Phrase | Where | Reasoning |
|---|---|---|
| runs your tests in a sandbox | description, README "Safety model" | Distinctive and true (macOS today) |
| write, check, repair | README "Features" | Names the loop |
| fix bugs in existing repositories | README "Features" | Matches "AI fix bug in repo" style searches |
| multi-file project builds | README "Features" | Matches "build a project with a local LLM" |
| edit review, undo | README "Features", "Safety model" | Safety-minded searchers |
| autopilot, Git branch | README "Features" | Matches "autonomous coding agent git branch" |

## 6. Name and spelling

| Phrase | Required in | Reasoning |
|---|---|---|
| vegacode | README first paragraph, repository name, PyPI name | The unique handle; "vega" alone collides with Vega/Vega-Lite (visualisation grammar) and other projects |
| vega | README, CLI command | The product name people will remember |

## Phrases deliberately not used

- "Copilot alternative", "Cursor alternative", "Claude Code alternative": high search volume, but
  they position Vega against named products.
- "Best", "fastest", "state of the art": not supported by a published results table.
- Pass rates or benchmark scores: not published.
- "Cross-platform": Vega is tested on macOS only; Linux is untested beyond CI and Windows has never
  been run.

## What the CI check covers

1. Each **required** phrase above appears (case-insensitive) in each place listed for it.
2. The description is at most 350 characters and starts with "Local coding agent".
3. Topics: 20 at most, each matching `^[a-z0-9][a-z0-9-]{0,49}$`, and each topic named in this file
   is set on the repository.
4. The README has no personal paths (`/Users/`, `/home/`), email addresses or session links.
