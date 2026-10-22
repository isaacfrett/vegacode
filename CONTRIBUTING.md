# Contributing to Vega

Thanks for helping. Vega is a local coding agent built around small open-weight models, so most
improvements are about doing more of the work in code (routing, checks, repairs) rather than in
the prompt. This page covers setup, tests, style, feature switches and how pull requests are
reviewed.

## Contributor License Agreement

Vega is licensed under the GNU AGPL-3.0 (version 3 only). Before your first pull request can be
merged you sign the
[Contributor License Agreement](CLA.md) once. You do not need to do anything in advance: the
**CLA Assistant** check on your first pull request tells you how to sign with your GitHub
account, and turns green once you have. You can also sign in advance at
https://cla-assistant.io/isaacfrett/vegacode. It covers all your later contributions too. No `Signed-off-by` (DCO) line
is needed in commits.

## Development setup

You need Python 3.11, 3.12 or 3.13 and Git. Ollama (or another local OpenAI-compatible server) is only
needed to chat with a model; the test suite never calls a model.

```bash
git clone https://github.com/isaacfrett/vegacode.git
cd vegacode
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
vega --help
```

In a checkout, Vega reads model profiles from `models.toml` at the repository root (the same file
ships inside the package as `vega/default_models.toml`). Set `VEGA_HOME` to a scratch folder
while developing so your experiments do not touch your real `~/.vega`:

```bash
export VEGA_HOME="$PWD/.vega-dev"
```

## Tests

```bash
python -m pytest -q
```

- The suite runs on macOS and Linux in CI (Python 3.11, 3.12 and 3.13), and those jobs must pass.
  Tests that need the macOS sandbox (`sandbox-exec`), `node`, `go` or WebKit skip themselves when
  it is missing, so a skip on Linux is expected; a failure is not.
- **Windows has never been run or tested.** CI runs the suite on Windows for information only; it
  never blocks a pull request. Fixes that make Windows work are welcome.
- No test may reach a real model server or write to `~/.vega`; `tests/conftest.py` enforces this.
- Every behaviour change comes with a test that fails without it.
- A test runs with the shipped feature defaults. If a test is not about a default feature and
  needs it off, use `switch_off(monkeypatch, "name")` from `tests/switches.py` with a comment
  saying why.

## Style

- `ruff check` with the settings in `pyproject.toml` (line length 100). CI enforces the error
  rules (`ruff check --select E9,F63,F7,F82 .`); please keep the files you touch clean under the
  full rule set as well.
- Type hints on public functions; `from __future__ import annotations` at the top of modules.
- Comments say why, not what. Messages shown to users say what happened and what to do next.

## Feature switches

Anything that can change what the model does, and so could make results better or worse, goes
behind a switch in `src/vega/features.py`:

- `FEATURES` holds behaviour that is **on by default**. A user (or an eval) turns entries off with
  `VEGA_FEATURES_OFF=name1,name2`.
- `OPT_IN` holds experimental behaviour that is **off by default**, turned on with
  `VEGA_FEATURES_ON=name`.
- Add a one-line description of what the switch controls; it is shown by `vega features`.
- Check it in code with `features.on("name")`.
- New behaviour starts in `OPT_IN` unless it is a plain bug fix. It moves to `FEATURES` once it
  has been measured to help, and the pull request says how it was measured (cases, model, runs).
- Remove code behind a switch rather than leaving it dead; never delete a switch silently, since
  saved settings may still name it.

## Pull requests

1. Open an issue first for anything larger than a small fix, so the approach can be agreed
   before you write it.
2. Keep a pull request to one change. Refactors go in their own pull request.
3. Fill in the template: what changed, how you checked it (commands and what you looked at), and
   the checklist.
4. CI must pass: ruff, the test matrix, the wheel build and smoke install, the keyword check and
   the secret scan. The CLA check must be green.
5. The maintainer reviews every pull request (CODEOWNERS). Expect questions about how a change
   was measured if it affects what the model is asked to do.
6. Pull requests are squash-merged; the title becomes the commit message.

## Security

Never report a vulnerability in a public issue or pull request. See [SECURITY.md](SECURITY.md).

## Code of conduct

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
