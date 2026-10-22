"""Copy text to the system clipboard with the platform's own tool.

The full-screen app also sends OSC 52 (the terminal's clipboard escape), but some terminals ignore
it, macOS Terminal.app among them. This writes the clipboard directly: pbcopy on macOS, wl-copy,
xclip or xsel on Linux, clip.exe on Windows (and WSL). Each call is bounded by a timeout; callers
run it off the UI thread."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping

TIMEOUT = 2.0  # seconds


def command(platform: str | None = None, env: Mapping[str, str] | None = None,
            which: Callable[[str], str | None] = shutil.which) -> list[str] | None:
    """The command that reads text on stdin into the clipboard, or None when there is none."""
    platform = platform or sys.platform
    env = os.environ if env is None else env
    if platform == "darwin":
        return ["pbcopy"] if which("pbcopy") else None
    if platform.startswith(("win", "cygwin", "msys")):
        found = which("clip.exe") or which("clip")
        return [found] if found else None
    candidates: list[list[str]] = []
    if env.get("WAYLAND_DISPLAY"):
        candidates.append(["wl-copy"])
    candidates += [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
    if not env.get("WAYLAND_DISPLAY"):
        candidates.append(["wl-copy"])
    candidates.append(["clip.exe"])  # WSL: the Windows clipboard
    for candidate in candidates:
        if which(candidate[0]):
            return candidate
    return None


def encode(text: str, argv: list[str]) -> bytes:
    """clip.exe reads UTF-16 with a byte-order mark; the others read UTF-8."""
    if os.path.basename(argv[0]).lower().startswith("clip"):
        return text.encode("utf-16")
    return text.encode("utf-8")


def copy(text: str, *, argv: list[str] | None = None, timeout: float = TIMEOUT) -> bool:
    """Put `text` on the system clipboard; True when the tool accepted it. Never raises."""
    argv = argv or command()
    if not argv:
        return False
    env = dict(os.environ)
    if argv[0] == "pbcopy":
        env.setdefault("LC_CTYPE", "UTF-8")  # pbcopy decodes stdin by locale
    try:
        result = subprocess.run(argv, input=encode(text, argv), stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=timeout, env=env, check=False)
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return result.returncode == 0


__all__ = ["TIMEOUT", "command", "copy", "encode"]
