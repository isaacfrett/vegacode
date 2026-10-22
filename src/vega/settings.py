"""Vega's persistent settings: dotted keys such as `web.backend`, in JSON files.

Two layers, the later winning: the user's ~/.vega/settings.json (VEGA_HOME moves it) and the
project's .vega/settings.json (a choice made for one project, e.g. "always allow web research
here"). Environment variables named in ENV override both for one process. Unknown keys are kept
as they are, so an older Vega never drops a newer one's settings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import server

DEFAULTS: dict[str, Any] = {
    "web.backend": "none",  # none | searxng
    "web.searxng_url": "http://localhost:8888",
    "web.permission": "ask",  # ask (once per session) | allow | deny
    "web.official_domains": [],  # extra domains whose pages count as official docs
    "usage.log": False,  # the opt-in local usage log (usage.py); never sent anywhere
}
ENV = {
    "web.backend": "VEGA_WEB_BACKEND",
    "web.searxng_url": "VEGA_SEARXNG_URL",
    "web.permission": "VEGA_WEB_PERMISSION",
}
CHOICES = {
    "web.backend": ("none", "searxng"),
    "web.permission": ("ask", "allow", "deny"),
}
PROJECT_FILE = ".vega/settings.json"


def user_file() -> Path:
    return server.STATE_DIR / "settings.json"  # resolved on use: tests move STATE_DIR


def project_file(project: Path) -> Path:
    return Path(project) / PROJECT_FILE


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _lookup(data: dict[str, Any], key: str) -> tuple[bool, Any]:
    node: Any = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def get(key: str, project: Path | None = None) -> Any:
    """The setting's value: environment, then project file, then user file, then the default."""
    variable = ENV.get(key)
    if variable and os.environ.get(variable, "").strip():
        return os.environ[variable].strip()
    layers = [project_file(project)] if project is not None else []
    for path in [*layers, user_file()]:
        found, value = _lookup(_read(path), key)
        if found:
            return value
    return DEFAULTS.get(key)


def validate(key: str, value: Any) -> Any:
    """The value to store, or ValueError for a known key with an invalid value."""
    if key in CHOICES and value not in CHOICES[key]:
        raise ValueError(f"{key} must be one of {', '.join(CHOICES[key])}, not {value!r}")
    if key == "web.searxng_url" and not str(value).startswith(("http://", "https://")):
        raise ValueError(f"{key} must be an http:// or https:// URL, not {value!r}")
    if key == "web.official_domains" and isinstance(value, str):
        return [part.strip().lower() for part in value.split(",") if part.strip()]
    return value


def set_value(key: str, value: Any, project: Path | None = None) -> Path:
    """Store a setting in the user file, or in the project's file when `project` is given."""
    value = validate(key, value)
    path = project_file(project) if project is not None else user_file()
    data = _read(path)
    node = data
    parts = key.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = node[part] = {}
        node = child
    node[parts[-1]] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
