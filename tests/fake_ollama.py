"""A fake Ollama server on localhost for model-management tests: /api/version, /api/tags,
/api/show, a streaming /api/pull and /api/delete, with Ollama's documented shapes (docs/api.md).
Generation endpoints answer 500 so a test can never run a model by accident. Deleting a model
removes only blobs no remaining model references, as Ollama's DeleteHandler does
(manifest.RemoveNamed -> RemoveUnreferencedBlobs)."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

GB = 10**9


def tag(name: str, size: int, parent: str = "", params: str = "8.0B") -> dict[str, Any]:
    return {"name": name, "model": name, "size": size, "digest": f"d-{name}",
            "details": {"parent_model": parent, "format": "gguf", "family": "x",
                        "parameter_size": params, "quantization_level": "Q4_K_M"}}


@dataclass
class FakeOllama:
    models: list[dict[str, Any]] = field(default_factory=list)
    blobs: dict[str, dict[str, int]] = field(default_factory=dict)  # model -> {digest: size}
    pull_sizes: dict[str, int] = field(default_factory=dict)  # tag -> bytes the pull streams
    pull_error: str = ""
    requests: list[tuple[str, str, Any]] = field(default_factory=list)
    url: str = ""
    deleted_blobs: list[str] = field(default_factory=list)
    store: Path | None = None  # manifests on disk (OLLAMA_MODELS), kept in step with deletes

    def start(self) -> FakeOllama:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _body(self) -> Any:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                return json.loads(raw) if raw else None

            def _json(self, code: int, data: Any) -> None:
                body = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                fake.requests.append(("GET", self.path, None))
                if self.path == "/api/version":
                    self._json(200, {"version": "0.0.0-fake"})
                elif self.path == "/api/tags":
                    self._json(200, {"models": fake.models})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self) -> None:
                body = self._body()
                fake.requests.append(("POST", self.path, body))
                if self.path == "/api/show":
                    name = (body or {}).get("model", "")
                    found = next((m for m in fake.models if m["name"] == name), None)
                    if found is None:
                        self._json(404, {"error": f"model '{name}' not found"})
                    else:
                        self._json(200, {"details": found["details"], "parameters": ""})
                elif self.path == "/api/pull":
                    self._pull((body or {}).get("model", ""))
                else:  # /api/generate, /api/chat, /api/create, ...: never in tests
                    self._json(500, {"error": f"fake Ollama refuses {self.path}"})

            def _pull(self, name: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()

                def line(data: dict[str, Any]) -> None:
                    self.wfile.write(json.dumps(data).encode() + b"\n")
                    self.wfile.flush()

                line({"status": "pulling manifest"})
                if fake.pull_error:
                    line({"error": fake.pull_error})
                    return
                total = fake.pull_sizes.get(name, 2 * GB)
                for done in (0, total // 2, total):
                    line({"status": "pulling sha256:aa", "digest": "sha256:aa", "total": total,
                          "completed": done})
                line({"status": "pulling sha256:bb", "digest": "sha256:bb", "total": 100,
                      "completed": 100})
                for status in ("verifying sha256 digest", "writing manifest", "success"):
                    line({"status": status})
                fake.models.append(tag(name, total + 100))
                fake.blobs[name] = {"sha256:aa": total, "sha256:bb": 100}

            def do_DELETE(self) -> None:
                body = self._body() or {}
                fake.requests.append(("DELETE", self.path, body))
                name = body.get("model") or body.get("name") or ""
                found = next((m for m in fake.models if m["name"] == name), None)
                if self.path != "/api/delete" or found is None:
                    self._json(404, {"error": f"model '{name}' not found"})
                    return
                fake.models.remove(found)
                mine = fake.blobs.pop(name, {})
                kept = {d for layers in fake.blobs.values() for d in layers}
                fake.deleted_blobs += [d for d in mine if d not in kept]
                if fake.store is not None:
                    model, _, version = name.partition(":")
                    (fake.store / "manifests" / "registry.ollama.ai" / "library" / model /
                     version).unlink(missing_ok=True)
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02},
                         daemon=True).start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def names(self) -> list[str]:
        return [m["name"] for m in self.models]

    def calls(self, method: str, path: str) -> list[Any]:
        return [body for m, p, body in self.requests if m == method and p == path]
