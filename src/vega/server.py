"""Manage the local Ollama server Vega talks to (model profiles live in models.py)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = Path(os.getenv("VEGA_HOME", Path.home() / ".vega"))


def source_checkout() -> bool:
    """Whether Vega runs from a clone of its repository (not an installed package), so the
    developer tools that read and write evals/, training/ and docs/ there have somewhere to go."""
    return (PROJECT_ROOT / "pyproject.toml").is_file() and (PROJECT_ROOT / "src" / "vega").is_dir()
PID_FILE = STATE_DIR / "server.pid"
LOG_FILE = STATE_DIR / "server.log"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
# Memory savers applied when Vega starts Ollama (unless already set): flash attention lets the
# context (KV) cache be stored at 8 bits, roughly halving its memory at negligible quality cost.
MEMORY_DEFAULTS = {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}
# Keep the model loaded between chats instead of Ollama's 5-minute default, so the next session
# does not wait for a reload. (Applies to servers Vega starts; set OLLAMA_KEEP_ALIVE to override.)
# One loaded model at a time: an A/B between two profiles of the same weights then swaps instead
# of holding two copies (and two context caches) in memory.
# llama.cpp's server (which Ollama runs) keeps past prompt states in RAM for reuse, up to 8 GB by
# default: on a looping 100-step task it reached 53 prompts / 8072 MiB on top of the 4.8 GB model
# (2026-10-14), pushing the 16 GB machine 6 GB into swap. 1 GB keeps a few states for reuse.
SERVER_DEFAULTS = {"OLLAMA_KEEP_ALIVE": "30m", "OLLAMA_MAX_LOADED_MODELS": "1",
                   "LLAMA_ARG_CACHE_RAM": "1024"}


def base_url() -> str:
    return os.getenv("VEGA_BASE_URL", DEFAULT_BASE_URL)


def model_name() -> str:
    return os.getenv("VEGA_MODEL", "vega")


def _ollama_root() -> str:
    """Ollama's native API root, derived from the OpenAI-compatible base URL."""
    url = base_url().rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def _endpoint_profile(model: str | None = None):  # -> models.Profile | None
    """The endpoint profile (a non-Ollama local server) behind `model`, if any. Ollama-only steps
    (starting Ollama, unloading, /api/show) are skipped for those."""
    from . import models  # models imports this module

    try:
        return models.endpoint_profile(model or model_name())
    except Exception:  # noqa: BLE001 - routing questions must never break Ollama use
        return None


def is_local() -> bool:
    return urlparse(base_url()).hostname in {"localhost", "127.0.0.1", "::1"}


def _get_json(path: str) -> dict | None:
    try:
        with urllib.request.urlopen(_ollama_root() + path, timeout=2) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _post_json(path: str, payload: dict) -> dict | None:
    request = urllib.request.Request(
        _ollama_root() + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def context_length(model: str | None = None) -> int | None:
    """The model's configured num_ctx (e.g. `PARAMETER num_ctx` in the Modelfile) from Ollama's
    /api/show, or None when the server is not Ollama, is unreachable, or sets no num_ctx."""
    data = _post_json("/api/show", {"model": model or model_name()})
    parameters = data.get("parameters") if isinstance(data, dict) else None
    if not isinstance(parameters, str):
        return None
    for line in parameters.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "num_ctx":
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def is_running() -> bool:
    return _get_json("/api/version") is not None


def installed_models() -> list[str]:
    data = _get_json("/api/tags") or {}
    return [item.get("name", "") for item in data.get("models", [])]


def has_model(name: str | None = None) -> bool:
    name = name or model_name()
    wanted = {name, f"{name}:latest"} if ":" not in name else {name}
    return any(model in wanted for model in installed_models())


def _owned_pid() -> int | None:
    """PID of the server Vega started, if it is still alive and still Ollama."""
    try:
        pid = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    result = subprocess.run(["ps", "-p", str(pid), "-o", "comm="], text=True, capture_output=True)
    if result.returncode or "ollama" not in result.stdout:
        PID_FILE.unlink(missing_ok=True)
        return None
    return pid


def start(timeout: float = 20.0) -> str:
    if is_running():
        return f"Server already running at {_ollama_root()}."
    if not is_local():
        return f"{base_url()} is not a local address; start that server where it runs."
    executable = shutil.which("ollama")
    if not executable:
        raise RuntimeError("Ollama is not installed. Install it from https://ollama.com/download.")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    port = urlparse(base_url()).port or 11434
    from . import features

    memory = {**SERVER_DEFAULTS, **(MEMORY_DEFAULTS if features.on("kv_q8") else {})}
    if os.environ.get("VEGA_SERVER_PARALLEL"):  # requests the server runs at once (eval shards)
        memory["OLLAMA_NUM_PARALLEL"] = os.environ["VEGA_SERVER_PARALLEL"]
    env = {**memory, **os.environ, "OLLAMA_HOST": f"127.0.0.1:{port}"}
    with LOG_FILE.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [executable, "serve"], stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, env=env,
        )
    PID_FILE.write_text(str(process.pid))
    _settings_file().write_text(json.dumps({key: env[key] for key in sorted(env)
                                            if key.startswith(("OLLAMA_", "LLAMA_ARG_"))}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_running():
            return f"Server started at {_ollama_root()} (pid {process.pid}, log {LOG_FILE})."
        if process.poll() is not None:
            break
        time.sleep(0.25)
    raise RuntimeError(f"Ollama did not become ready within {timeout:.0f}s; see {LOG_FILE}.")


def stop() -> str:
    pid = _owned_pid()
    if pid is None:
        if is_running():
            return "The running server was not started by Vega (e.g. the Ollama app); leaving it alone."
        return "Server is not running."
    os.kill(pid, signal.SIGTERM)
    for _ in range(40):
        if _owned_pid() is None:
            break
        time.sleep(0.25)
    PID_FILE.unlink(missing_ok=True)
    return f"Stopped server (pid {pid})."


def unload(model: str | None = None) -> bool:
    """Ask Ollama to drop the model from memory now instead of after its keep-alive period."""
    if _endpoint_profile(model) is not None:
        return False  # another server's memory is managed by that server
    if not is_running():
        return False
    return _post_json("/api/generate", {"model": model or model_name(), "keep_alive": 0}) is not None


def fingerprint(model: str | None = None) -> dict[str, object]:
    """What the results depend on besides Vega's code: server version, exact model weights and
    server settings. Recorded with every eval run so runs are only compared like for like."""
    name = model or model_name()
    profile = _endpoint_profile(name)
    if profile is not None:
        return {"model": name, "endpoint": profile.endpoint, "api_model": profile.served_name,
                "served_models": served_models(profile.endpoint)}
    version = (_get_json("/api/version") or {}).get("version")
    digest = None
    for item in (_get_json("/api/tags") or {}).get("models", []):
        if item.get("name") in {name, f"{name}:latest"}:
            digest = str(item.get("digest", ""))[:12] or None
    settings = _started_settings()
    return {
        "ollama_version": version, "model": name, "model_digest": digest,
        "started_by_vega": settings is not None,
        "server_env": settings or {},
    }


def _settings_file() -> Path:
    # Resolved on use (not at import) so a changed STATE_DIR, e.g. in tests, is respected.
    return STATE_DIR / "server_settings.json"


def _started_settings() -> dict[str, str] | None:
    """Ollama settings Vega passed when it started the running server (None if not Vega's)."""
    if _owned_pid() is None:
        return None
    try:
        return json.loads(_settings_file().read_text())
    except (OSError, ValueError):
        return {}


def ensure_running(model: str | None = None) -> None:
    """Start the local server on demand so `vega` works without a separate setup step. A model on
    an endpoint profile runs on that server, which Vega does not start: nothing to do."""
    if _endpoint_profile(model) is not None:
        return
    if is_local() and not is_running() and shutil.which("ollama"):
        start()


def served_models(endpoint: str, timeout: float = 2.0) -> list[str] | None:
    """Model ids an OpenAI-compatible server lists at GET <endpoint>/models (None: unreachable).
    A listing only; it never loads or runs a model."""
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=timeout) as response:
            data = json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    items = data.get("data") if isinstance(data, dict) else None
    return [str(item.get("id")) for item in items or [] if isinstance(item, dict)]


def build_model() -> int:
    """Rebuild the default `vega` model from its active profile (models.toml default if unset)."""
    from . import models

    default, _ = models.load()
    return models.use(models.active_profile() or default)


def status_rows() -> list[tuple[str, str]]:
    profile = _endpoint_profile()
    if profile is not None:
        listed = served_models(profile.endpoint)
        return [
            ("Model", f"{model_name()} (profile {profile.name})"),
            ("Endpoint", profile.endpoint),
            ("Endpoint model", profile.served_name),
            ("Endpoint server", "unreachable" if listed is None else "running"),
            ("Model listed", "?" if listed is None else
             ("yes" if profile.served_name in listed else f"no — it lists: {', '.join(listed[:5])}")),
            ("Ollama", "not used by this profile"),
        ]
    running = is_running()
    owner = "vega" if _owned_pid() else ("external" if running else "-")
    rows = [
        ("Base URL", base_url()),
        ("Server", "running" if running else "stopped"),
        ("Started by", owner),
        ("Model", model_name()),
    ]
    if running:
        rows.append(("Model installed", "yes" if has_model() else "no — run `vega server build`"))
    return rows
