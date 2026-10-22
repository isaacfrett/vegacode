## What this changes

<!-- One or two sentences: the problem and the fix. Link the issue: "Fixes #123". -->

## How it was checked

<!-- The commands you ran and what you looked at. "Tests pass" is a start; say what you checked
     by hand too (the command output, the TUI screen, the file Vega wrote). -->

- [ ] `python -m pytest -q` passes locally (note any test skipped on your OS)
- [ ] `ruff check --select E9,F63,F7,F82 .` is clean, and the files I touched pass `ruff check`
- [ ] New behaviour has a test that fails without the change

## Checklist

- [ ] I have signed the [Contributor License Agreement](https://github.com/isaacfrett/vegacode/blob/main/CLA.md) (the CLA Assistant check on
      this PR asks you once, the first time you contribute)
- [ ] New or changed behaviour that could make results worse is behind a feature switch in
      `src/vega/features.py` (see CONTRIBUTING.md)
- [ ] No secrets, personal paths or private data in code, tests or fixtures
- [ ] README / docs updated if a command, option or default changed
