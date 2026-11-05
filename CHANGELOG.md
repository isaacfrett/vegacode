# Changelog

Notable changes to vegacode, newest first. Versions follow `pyproject.toml`.

## 0.3.2 (unreleased)

Setup in two steps on a stock Mac, a calmer full-screen chat, and edits that can no longer drop
your code by accident.

### What's new

- **Two-step setup.** `pip3 install vegacode`, then `python3 -m vega`. The first start now also
  sets up the model side: it installs Ollama when it is missing (with Homebrew `brew install
  ollama`, otherwise the Ollama app from ollama.com, its code signature checked, into
  `~/Applications`) and downloads the best model that fits your machine (on a 16 GB Mac, Granite
  8B, a 5.3 GB download), then opens the chat. It shows the plan first and asks yes or no before
  the install and before the download. Later starts repair only what has gone missing; `vega setup`
  runs the same steps again (`vega setup --yes` without questions). Without a terminal nothing is
  installed: `vega run` stops and names the command to run.
- **Chat first.** `vega` opens as just the conversation, in your terminal's own colours: a centred
  reading column with real tables, a boxed input and a live line showing CPU, memory and swap.
  Side panels are opt-in (`/layout`, or the `/` menu). Vega never resizes your terminal window.
- **Models are judged by memory, not parameter count.** A model qualifies when it loads in at most
  10 GB (weights, context and runtime together), which leaves room to work on a 16 GB Mac.
  `vega model recommend`, `vega model list` and `vega model pull` use this rule;
  `VEGA_MODEL_MEMORY_BUDGET_GB` changes the limit for a bigger machine.
- **Better clarifying questions.** On a vague request Vega asks about scope and data (what to
  include, what the input looks like). It no longer asks where your code lives or how to check
  its work: it finds those out itself.
- **`vega update` and `vega uninstall`.** `vega update` checks PyPI, shows both versions, asks, and
  upgrades with the installer you used (the launcher's environment, pipx, uv or pip);
  `vega update --check` only reports. `vega uninstall` lists what it can remove, with sizes, and
  asks for each group: Vega itself, the `PATH` line it added and its background jobs. Your data
  (`~/.vega`) and the models Vega downloaded are kept unless you ask (`--remove-data`,
  `--remove-models`). Ollama, Python and Homebrew are never removed. `--dry-run` changes nothing.
- **Started in the wrong folder?** Started in your home folder, `/` or a large folder that is not
  a project, Vega first asks what you are working on: a recent project, a new one, or just a chat.
  `/project <path>` switches to another folder and keeps the conversation.

### Fixes

- **`vega` works in a new terminal.** The launcher now adds `~/.local/bin` to your `PATH` in your
  shell's startup file (`~/.zshrc`, `~/.bash_profile`, fish or `~/.profile`, listed in the plan you
  agree to). 0.3.1 only printed the line to add, so `vega` was often not found in a new
  terminal.
- **One question when Python 3.11+ is missing.** Installing Python 3.12 (with uv or Homebrew) is
  now the first line of the launcher's plan, and "Set up Vega now?" is the only question. 0.3.1
  asked twice.
- **Esc stops promptly.** Pressing Esc while the conversation was being compacted no longer sends
  one more request to the model, and Esc now also stops sub-agents Vega started.
- **No more silently deleted code.** A write that would replace existing code with a placeholder
  such as `# ... rest unchanged` is refused, and the refusal says which lines would have been lost.
- **Missing submodules are real failures.** Code that imports a submodule an installed package
  does not have now fails Vega's checks instead of being marked "not checkable".
- **Web answers say what was checked.** In HTML, CSS and JavaScript work, checks that only show
  the page loads are reported apart from checks of what you asked for, so a page that merely loads
  is not reported as done.
- **`VEGA_FEATURES_OFF` adds to a model profile's settings.** Your list used to replace the
  profile's own switched-off features (on the default Granite profile, setting it turned the plan
  tool back on). Both now apply.

### Upgrade notes

- The full-screen app opens in the new chat layout. `/layout classic` brings the side panel back
  (kept for next time; `VEGA_LAYOUT=<name>` overrides it for one start).
- Models over the 10 GB memory budget are listed as such and never recommended.
- Coming from 0.3.1 with Ollama and a model already set up: the first start finds them and opens
  the chat; nothing is installed or downloaded.
- Coming from 0.3.1: it has no `vega update`, so upgrade once with the installer you used
  (`pip3 install -U vegacode`, `pipx upgrade vegacode` or `uv tool upgrade vegacode`); from 0.3.2 on,
  `vega update` does it.

Next: 0.4 is a larger release focused on Vega's harness, the code that checks and steers a small
model's work.

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
