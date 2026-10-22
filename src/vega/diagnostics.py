"""Cheap, automatic diagnostics appended to the result of every file write.

`after_write` never executes project code: Python is only compiled, data files are only parsed,
and lint runs a fixed `ruff check` argv over the written text (read-only, no network, sandboxed
when the macOS sandbox is available: sandbox.readonly_argv).
"""

from __future__ import annotations

import configparser
import functools
import json
import subprocess
import tomllib
from pathlib import Path

from vega import features, sandbox

LINT_TIMEOUT = 5
LINT_LIMIT = 10
# Pyflakes-level problems only: syntax, undefined names, unused/redefined imports and names.
LINT_SELECT = "E9,F"


def after_write(project: Path, path: Path, text: str) -> str:
    """Return a short note (leading newline, or "") to append to an edit's tool result.

    Parse failures start with "WARNING:" (the agent treats those files as broken); lint findings
    are advisory and never contain it.
    """
    suffix = path.suffix.lower()
    if suffix == ".py":
        try:
            compile(text, str(path), "exec", dont_inherit=True)
        except SyntaxError as exc:
            return (
                f"\nWARNING: the file now has a SyntaxError at line {exc.lineno}: {exc.msg}. Fix it."
            )
        return "\nSyntax check: OK." + _lint_note(project, path, text)
    if suffix == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            return (
                f"\nWARNING: the file is now invalid JSON at line {exc.lineno}: {exc.msg}. Fix it."
            )
        return "\nJSON check: OK."
    if suffix == ".toml":
        try:
            tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            return f"\nWARNING: the file is now invalid TOML: {exc}. Fix it."
        return "\nTOML check: OK."
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # optional: only checked when PyYAML is installed
        except ImportError:
            return ""
        try:
            list(yaml.safe_load_all(text))
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            where = f" at line {mark.line + 1}" if mark is not None else ""
            problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
            return f"\nWARNING: the file is now invalid YAML{where}: {problem}. Fix it."
        return "\nYAML check: OK."
    return ""


def parse_problem(path: Path, text: str) -> tuple[str, int | None] | None:
    """(what is wrong, line number or None) when `text` does not parse as `path`'s kind of file
    (Python, JSON, TOML; YAML when PyYAML is installed); None when it parses or is not checked."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".py":
            compile(text, str(path), "exec", dont_inherit=True)
        elif suffix == ".json":
            json.loads(text)
        elif suffix == ".toml":
            tomllib.loads(text)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError:
                return None
            list(yaml.safe_load_all(text))
        elif suffix in {".ini", ".cfg"}:
            # strict: a duplicated section or option is an error (the 7B appended a second
            # [http] section instead of adding to the existing one, 2026-10-13)
            configparser.ConfigParser(interpolation=None).read_string(text)
    except configparser.Error as exc:
        line = getattr(exc, "lineno", None)
        return f"invalid INI: {str(exc).splitlines()[0]}", line
    except SyntaxError as exc:
        return f"a SyntaxError at line {exc.lineno}: {exc.msg}", exc.lineno
    except json.JSONDecodeError as exc:
        return f"invalid JSON at line {exc.lineno}: {exc.msg}", exc.lineno
    except tomllib.TOMLDecodeError as exc:
        return f"invalid TOML: {exc}", getattr(exc, "lineno", None)
    except Exception as exc:  # noqa: BLE001 - yaml.YAMLError without importing yaml up front
        mark = getattr(exc, "problem_mark", None)
        line = mark.line + 1 if mark is not None else None
        return f"invalid YAML: {getattr(exc, 'problem', None) or exc}", line
    return None


@functools.lru_cache(maxsize=32)
def _ruff_path(project: Path) -> str | None:
    """ruff from the project's .venv, then Vega's own environment, then PATH (cached)."""
    return sandbox.find_python_tool(project, "ruff")


def _lint_note(project: Path, path: Path, text: str) -> str:
    if not features.on("lint"):
        return ""
    ruff = _ruff_path(project.resolve())
    if ruff is None:
        return ""
    findings = _ruff_findings(ruff, project, path, text)
    if findings is None:  # ruff failed (bad config, timeout): stay quiet rather than add noise
        return ""
    if not findings:
        return "\nLint: OK."
    shown = findings[:LINT_LIMIT]
    more = f"\n({len(findings) - LINT_LIMIT} more)" if len(findings) > LINT_LIMIT else ""
    return "\nLint findings (advisory):\n" + "\n".join(shown) + more


def _ruff_findings(ruff: str, project: Path, path: Path, text: str) -> list[str] | None:
    # Fixed argv; the written text goes in on stdin, so ruff reads no other file contents than
    # the project's ruff configuration. --no-cache keeps it from writing .ruff_cache.
    argv = [
        ruff, "check", "--no-cache", "--quiet", "--output-format=json", f"--select={LINT_SELECT}",
        "--stdin-filename", str(path), "-",
    ]
    argv = sandbox.readonly_argv(argv)
    try:
        result = subprocess.run(
            argv, input=text, cwd=project, text=True, capture_output=True, timeout=LINT_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode not in (0, 1):
        return None
    try:
        items = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return None
    findings = []
    for item in sorted(items, key=lambda i: (i["location"]["row"], i["location"]["column"])):
        location = item["location"]
        code = item.get("code") or "syntax"
        message = str(item["message"]).replace("WARNING:", "warning:")  # reserved for breakage
        findings.append(f"{location['row']}:{location['column']} {code} {message}")
    return findings
