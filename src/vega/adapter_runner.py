"""The child side of a generated site adapter's sandbox (vega/adapter_builder.py).

Started by the harness as `python -I -c ...` (sandbox.readonly_argv: the macOS sandbox, no network and no
file writes). It reads one job line from stdin ({"code", "urls"}), runs the adapter module's
`fetch(url, api_get)` for each URL and writes JSON lines to its protocol channel (the original
stdout): {"op": "get", ...} for each API request, answered by the harness on stdin, then one
{"op": "result", ...} per URL. The adapter's own prints go to stderr.

The adapter has no network of its own: sockets are disabled here as well as by the OS sandbox,
and `api_get` is the harness's proxy (domain allowlist, size cap, timeouts, rate limits).
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import sys
import traceback
from typing import Any


def _no_network(*args: Any, **kwargs: Any) -> Any:
    raise OSError("network access is not allowed in a site adapter; use api_get(url) only")


def _disable_sockets() -> None:
    for name in ("connect", "connect_ex", "bind", "sendto", "sendmsg"):
        setattr(socket.socket, name, _no_network)
    socket.create_connection = _no_network  # type: ignore[assignment]
    socket.getaddrinfo = _no_network  # type: ignore[assignment]
    socket.socketpair = _no_network  # type: ignore[assignment]


def main() -> None:
    channel = os.fdopen(os.dup(1), "w", encoding="utf-8")
    sys.stdout = sys.stderr  # an adapter's print() must not corrupt the protocol
    job = json.loads(sys.stdin.readline())
    _disable_sockets()

    from vega import web

    class AdapterApiError(web.BackendError):
        """The harness refused or could not complete an api_get."""

    def send(message: dict[str, Any]) -> None:
        channel.write(json.dumps(message) + "\n")
        channel.flush()

    def api_get(url: str, headers: dict[str, str] | None = None) -> web.ApiResponse:
        send({"op": "get", "url": str(url), "headers": {str(k): str(v) for k, v in
                                                         (headers or {}).items()}})
        reply = json.loads(sys.stdin.readline() or "{}")
        if "error" in reply:
            raise AdapterApiError(reply["error"])
        return web.ApiResponse(int(reply.get("status", 0)), base64.b64decode(reply.get("body", "")),
                               dict(reply.get("headers") or {}), bool(reply.get("truncated")))

    namespace: dict[str, Any] = {
        "__name__": "vega_site_adapter", "Page": web.Page, "ApiResponse": web.ApiResponse,
        "BackendError": web.BackendError, "BlockedError": web.BlockedError,
        "_source_page": web._source_page, "_day": web._day, "_html_text": web._html_text,
        "_unexpected": web._unexpected, "GENERIC": None,
    }
    try:
        exec(compile(job["code"], "adapter.py", "exec"), namespace)  # noqa: S102 - sandboxed child
        pattern = re.compile(str(namespace["MATCH"]))
        fetch = namespace["fetch"]
    except Exception:  # noqa: BLE001 - reported to the model as repair feedback
        error = "the module failed to load:\n" + _tail(traceback.format_exc())
        for url in job["urls"]:
            send({"op": "result", "url": url, "page": None, "error": error})
        return
    for url in job["urls"]:
        if not pattern.search(url):
            send({"op": "result", "url": url, "page": None,
                  "error": f"MATCH ({pattern.pattern!r}) does not match this URL"})
            continue
        try:
            page = fetch(url, api_get)
        except Exception:  # noqa: BLE001 - reported to the model as repair feedback
            send({"op": "result", "url": url, "page": None,
                  "error": "fetch raised:\n" + _tail(traceback.format_exc())})
            continue
        if page is None:
            send({"op": "result", "url": url, "page": None,
                  "error": "fetch returned None (page not found)"})
        elif isinstance(page, web.Page):
            send({"op": "result", "url": url, "error": "",
                  "page": {"title": str(page.title), "date": str(page.date),
                           "text": str(page.text)[: web.MAX_PAGE_CHARS + 200]}})
        else:
            send({"op": "result", "url": url, "page": None,
                  "error": f"fetch returned {type(page).__name__}, not a Page"})


def _tail(text: str, lines: int = 12) -> str:
    kept = [line for line in text.splitlines() if 'File "<' not in line or "adapter.py" in line]
    return "\n".join(kept[-lines:])


if __name__ == "__main__":
    main()
