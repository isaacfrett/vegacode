# Changelog

Notable changes to vegacode, newest first. Versions follow `pyproject.toml`.

## 0.3.1 (2026-10-23)

- Installs on a stock Mac: `pip3 install vegacode && python3 -m vega`. With Python 3.9 or 3.10
  (macOS's own `pip3` is 3.9) pip installs only a small launcher. It finds a Python 3.11+ (or
  offers to install one with uv or Homebrew, asking first), sets Vega up in its own environment in
  `~/.local/share/vega/venv` after you agree, links `vega` into `~/.local/bin` and starts it. Vega
  itself still needs Python 3.11+.

## 0.3.0 (2026-10-21)

First public release.

Vega is a local coding agent for the terminal. It runs an open-weight model on your own machine
through Ollama (or another local OpenAI-compatible server), reads and edits your project, runs
your tests and checks in a sandbox, and repairs what fails. It is built for small models (under
10B parameters) on a 16 GB machine. The package is `vegacode`; the command is `vega`.

- Full-screen chat (`vega`), line-mode chat (`vega --plain`) and one-shot tasks (`vega run`).
- Task routing: each request goes to a fixed pipeline (a question about the code, a one-file
  Python change, a multi-file change, a generated file, a single non-Python file with a check
  command, a CSS/HTML/JavaScript change) or the general agent loop.
- Write, check, repair: the pipelines run checks derived from the request and the project's
  tests, send failures back to the model, and keep the round with the fewest failing checks.
- Safe edits: writes that would not parse or that add undefined names are refused; `--review`
  shows each edit as a diff first; `/undo` restores Vega's last file changes.
- Autopilot (`vega auto`): works on a new Git branch, commits only rounds whose check passes,
  never pushes.
- Project memory (`VEGA.md`, session digests and notes), a model manager with a RAM-fit estimate,
  and a `.gitignore` block that keeps Vega's files and likely secrets out of Git.
- Default model: IBM Granite 4.1 8B with an 8K context; Qwen2.5-Coder 7B and Qwen3 8B profiles
  in `models.toml`.
- Opt-in or experimental: web research through a self-hosted SearXNG, teach mode (`--teach`),
  and background loops.

Platforms: developed and tested on macOS (sandboxed checks use macOS `sandbox-exec`). Linux is
not yet tested beyond the automated test suite in CI. Windows has never been run or tested.
