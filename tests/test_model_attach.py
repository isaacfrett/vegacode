"""Attaching a model: `vega model add/remove`, endpoint profiles (any local OpenAI-compatible
server) and `vega model check`. Fakes only: a fake OpenAI-compatible HTTP server on 127.0.0.1."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from vega import context, external, latency, model_check, models, server
from vega.agent import CodingAgent, ServedName
from vega.cli import app
from vega.pipeline import WRITER_SYSTEM
from vega.question_pipeline import QA_SYSTEM

# The header and two profiles of the real file, comments and sub-tables included.
ORIGINAL = """# Vega model profiles: the one place to choose the base model and its settings.
#
#   vega model list               profiles, whether built, and any eval scores

default = "coder-7b"

[profiles.coder-7b]
base = "qwen2.5-coder:7b-instruct-q4_K_M"
description = "Default."
num_ctx = 8192
features_on = ["constrained"]  # helped this model on fixes in two runs (2026-10-13)
[profiles.coder-7b.parameters]
temperature = 0.2

[profiles.coder-7b-16k]
base = "qwen2.5-coder:7b-instruct-q4_K_M"
num_ctx = 16384
[profiles.coder-7b-16k.parameters]
temperature = 0.2

# thinking models below
[profiles.qwen3-8b-nothink]
base = "qwen3:8b"
description = "Qwen3 8B with thinking off."
num_ctx = 8192
request = { reasoning_effort = "none" }
"""


@pytest.fixture
def profiles_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "models.toml"
    path.write_text(ORIGINAL, encoding="utf-8")
    monkeypatch.setattr(models, "PROFILES_FILE", path)
    return path


# --- add / remove --------------------------------------------------------------------------------

def test_add_appends_without_touching_the_rest(profiles_file: Path) -> None:
    profile = models.add_profile("granite-8b", base="granite4.1:8b", num_ctx=16384,
                                 description='IBM "Granite" 4.1', request={"seed": 7})
    text = profiles_file.read_text(encoding="utf-8")
    assert text.startswith(ORIGINAL)  # byte for byte
    assert text.endswith('[profiles.granite-8b]\nbase = "granite4.1:8b"\n'
                         'description = "IBM \\"Granite\\" 4.1"\nnum_ctx = 16384\n'
                         "request = { seed = 7 }\n")
    assert profile.base == "granite4.1:8b" and profile.num_ctx == 16384
    assert profile.request == {"seed": 7} and not profile.is_endpoint
    assert models.get("coder-7b").parameters == {"temperature": 0.2}


def test_add_refuses_duplicates_and_bad_names(profiles_file: Path) -> None:
    with pytest.raises(ValueError, match="already exists"):
        models.add_profile("coder-7b", base="x")
    for bad in ("Coder", "a.b", "-x", "has space", ""):
        with pytest.raises(ValueError, match="Profile name"):
            models.add_profile(bad, base="x")
    with pytest.raises(ValueError, match="--api-model only applies"):
        models.add_profile("ok", base="x", api_model="y")
    assert profiles_file.read_text(encoding="utf-8") == ORIGINAL


def test_endpoint_must_be_local_unless_allowed(profiles_file: Path) -> None:
    for remote in ("http://example.com/v1", "http://8.8.8.8:1234/v1"):
        with pytest.raises(ValueError, match="local-first"):
            models.add_profile("far", base="m", endpoint=remote)
    with pytest.raises(ValueError, match="http"):
        models.add_profile("far", base="m", endpoint="127.0.0.1:1234")
    for local in ("http://127.0.0.1:1234/v1", "http://localhost:8080/v1/", "http://[::1]:9/v1",
                  "http://192.168.1.20:1234/v1", "http://10.0.0.5/v1"):
        assert models.is_local_host(models.urlparse(local).hostname or "")
    far = models.add_profile("far", base="m", endpoint="https://example.com/v1", allow_remote=True)
    assert far.allow_remote and "allow_remote = true" in profiles_file.read_text()
    lm = models.add_profile("lm", base="qwen2.5-coder-7b", endpoint="http://localhost:1234/v1/",
                            api_model="qwen2.5-coder-7b-instruct")
    assert lm.endpoint == "http://localhost:1234/v1" and lm.served_name == "qwen2.5-coder-7b-instruct"


def test_request_option_parsing() -> None:
    assert models.parse_request_option("reasoning_effort=none") == ("reasoning_effort", "none")
    assert models.parse_request_option("top_p=0.9") == ("top_p", 0.9)
    assert models.parse_request_option("think=false") == ("think", False)
    assert models.parse_request_option('stop=["</s>"]') == ("stop", ["</s>"])
    with pytest.raises(ValueError):
        models.parse_request_option("no-equals")


def test_remove_round_trips_and_keeps_neighbours(profiles_file: Path) -> None:
    models.add_profile("granite-8b", base="granite4.1:8b")
    models.remove_profile("granite-8b")
    assert profiles_file.read_text(encoding="utf-8") == ORIGINAL
    # a middle profile with a sub-table: its neighbours' comments and tables stay exactly
    removed = models.remove_profile("coder-7b-16k")
    assert removed.num_ctx == 16384
    expected = ORIGINAL.replace(
        '[profiles.coder-7b-16k]\nbase = "qwen2.5-coder:7b-instruct-q4_K_M"\nnum_ctx = 16384\n'
        "[profiles.coder-7b-16k.parameters]\ntemperature = 0.2\n\n", "")
    assert profiles_file.read_text(encoding="utf-8") == expected
    assert set(models.load()[1]) == {"coder-7b", "qwen3-8b-nothink"}


def test_remove_refuses_default_active_and_unknown(profiles_file: Path) -> None:
    with pytest.raises(ValueError, match="default"):
        models.remove_profile("coder-7b")
    with pytest.raises(ValueError, match="Unknown"):
        models.remove_profile("nope")
    server.STATE_DIR.mkdir(parents=True, exist_ok=True)
    (server.STATE_DIR / "active_model.json").write_text(json.dumps({"profile": "coder-7b-16k"}))
    with pytest.raises(ValueError, match="active"):
        models.remove_profile("coder-7b-16k")
    assert profiles_file.read_text(encoding="utf-8") == ORIGINAL


def test_cli_add_and_remove_print_next_steps(profiles_file: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["model", "add", "granite-8b", "--base", "granite4.1:8b",
                                 "--num-ctx", "8192", "--request", "reasoning_effort=none"])
    assert result.exit_code == 0, result.output
    for step in ("ollama pull granite4.1:8b", "vega model check granite-8b",
                 "vega model use granite-8b"):
        assert step in result.output
    again = runner.invoke(app, ["model", "add", "granite-8b", "--base", "x"])
    assert again.exit_code != 0 and "already exists" in again.output
    result = runner.invoke(app, ["model", "remove", "granite-8b"])
    assert result.exit_code == 0, result.output
    assert "ollama rm vega-granite-8b" in result.output
    assert profiles_file.read_text(encoding="utf-8") == ORIGINAL  # nothing else changed
    refused = runner.invoke(app, ["model", "remove", "coder-7b"])
    assert refused.exit_code != 0 and "default" in refused.output


# --- a fake OpenAI-compatible server --------------------------------------------------------------

class FakeServer:
    """Answers /v1/models and /v1/chat/completions like a local model server would, choosing the
    reply from the request (the conformance items each have a recognisable shape)."""

    def __init__(self, *, chat_answer: str = "391", think: bool = False,
                 report_length: bool = True):
        self.requests: list[dict[str, Any]] = []
        self.chat_answer, self.think, self.report_length = chat_answer, think, report_length
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                owner._send(self, {"object": "list", "data": [{"id": "served-model"}]})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(body)
                owner._reply(self, body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    @staticmethod
    def _send(handler: BaseHTTPRequestHandler, data: dict[str, Any]) -> None:
        raw = json.dumps(data).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)

    def _completion(self, body: dict[str, Any], *, content: str | None = None,
                    tool_calls: list[dict[str, Any]] | None = None, finish: str = "stop",
                    tokens: int = 10) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish = "tool_calls"
        return {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": tokens, "total_tokens": 110}}

    def _reply(self, handler: BaseHTTPRequestHandler, body: dict[str, Any]) -> None:
        messages = body["messages"]
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if body.get("stream"):
            answer = (f"<think>17 * 23 = 340 + 51</think>{self.chat_answer}" if self.think
                      else self.chat_answer)
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.end_headers()
            for index, piece in enumerate([answer[:5], answer[5:]]):
                chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0,
                         "model": body["model"], "choices": [{
                             "index": 0, "delta": {"role": "assistant", "content": piece},
                             "finish_reason": "stop" if index else None}]}
                handler.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            handler.wfile.write(b"data: [DONE]\n\n")
            return
        if body.get("max_tokens") == model_check.TRUNCATE_AT:
            parts = ["```python\ndef squares(n):\n", '    """The first n squares."""\n',
                     "    return [i * i for i in range(1, n + 1)]\n```"]
            so_far = "".join(m["content"] for m in messages if m["role"] == "assistant")
            turn = sum(1 for part in parts if part in so_far)  # ask_complete sends the text so far
            last = turn >= len(parts) - 1
            finish = "stop" if last or not self.report_length else "length"
            self._send(handler, self._completion(body, content=parts[min(turn, 2)], finish=finish,
                                                 tokens=model_check.TRUNCATE_AT))
            return
        if system.startswith(QA_SYSTEM[:40]):
            self._send(handler, self._completion(body, content="It returns 7 (shop/cart.py)."))
            return
        if system == WRITER_SYSTEM:
            code = ('```python\n"""Formatting."""\n\n\ndef money(amount):\n'
                    '    """Format an amount with two decimals and a leading $, e.g. money(3) == '
                    '"$3.00".\n\n    >>> money(3)\n    \'$3.00\'\n    """\n'
                    '    return f"${amount:.2f}"\n```')
            self._send(handler, self._completion(body, content=code))
            return
        if body.get("tools") and not any(m["role"] == "tool" for m in messages):
            call = {"id": "c1", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": "shop/inventory.py"})}}
            self._send(handler, self._completion(body, tool_calls=[call]))
            return
        self._send(handler, self._completion(body, content="Inventory is in shop/inventory.py."))


@pytest.fixture
def fake_server() -> Iterator[FakeServer]:
    fake = FakeServer()
    yield fake
    fake.close()


def _endpoint_profile(path: Path, url: str, **extra: Any) -> None:
    models.add_profile("lm", base="qwen-7b", endpoint=url, api_model="served-model",
                       request={"reasoning_effort": "none"}, **extra)


# --- routing --------------------------------------------------------------------------------------

def test_connection_routes_endpoint_profiles(profiles_file: Path, fake_server: FakeServer) -> None:
    _endpoint_profile(profiles_file, fake_server.url)
    route = models.connection("vega-lm")
    assert (route.base_url, route.model, route.profile) == (fake_server.url, "served-model", "lm")
    ollama = models.connection("vega-coder-7b")
    assert ollama.base_url == server.base_url() and ollama.model == "vega-coder-7b"
    assert not ollama.is_endpoint
    # `vega` follows the active profile
    assert not models.connection("vega").is_endpoint
    assert models.use("lm") == 0  # endpoint: recorded, never built
    assert models.active_profile() == "lm" and models.connection("vega").profile == "lm"
    assert models.resolve("lm") == "vega-lm"


def test_agent_requests_reach_the_endpoint_under_its_name(
        profiles_file: Path, fake_server: FakeServer, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    _endpoint_profile(profiles_file, fake_server.url)
    monkeypatch.setenv("VEGA_MODEL", "vega-lm")
    monkeypatch.setenv("VEGA_FEATURES_OFF", "question_pipeline,project_pipeline,direct_pipeline")
    (tmp_path / "shop").mkdir()
    (tmp_path / "shop" / "inventory.py").write_text("class Inventory:\n    pass\n")
    agent = CodingAgent(tmp_path, allow_commands=False, max_steps=4)
    assert agent.base_url == fake_server.url and isinstance(agent.client, ServedName)
    assert agent.num_ctx == 8192 and agent.num_ctx_source == "profile"
    answer = agent.run("Which file defines the Inventory class?")
    assert "shop/inventory.py" in answer
    assert fake_server.requests and all(r["model"] == "served-model" for r in fake_server.requests)
    assert all(r.get("reasoning_effort") == "none" for r in fake_server.requests)


def test_remote_endpoint_written_by_hand_is_refused(profiles_file: Path) -> None:
    profiles_file.write_text(ORIGINAL + '\n[profiles.far]\nbase = "m"\n'
                             'endpoint = "http://example.com/v1"\n')
    with pytest.raises(ValueError, match="local-first"):
        models.connection("vega-far")
    with pytest.raises(ValueError, match="local-first"):
        models.resolve("far")


def test_ollama_only_steps_are_skipped(profiles_file: Path, fake_server: FakeServer,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    _endpoint_profile(profiles_file, fake_server.url)

    def no_ollama(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Ollama must not be touched for an endpoint profile")

    for name in ("start", "_post_json", "is_running", "has_model"):
        monkeypatch.setattr(server, name, no_ollama)
    monkeypatch.setattr(models, "build", no_ollama)
    server.ensure_running("vega-lm")
    assert server.unload("vega-lm") is False
    fingerprint = server.fingerprint("vega-lm")
    assert fingerprint["endpoint"] == fake_server.url
    assert fingerprint["served_models"] == ["served-model"]
    assert context.resolve_num_ctx("vega-lm") == (8192, "profile")
    assert models.resolve("lm") == "vega-lm" and models.use("lm") == 0
    monkeypatch.setenv("VEGA_MODEL", "vega")  # the active profile is now the endpoint one
    rows = dict(server.status_rows())
    assert rows["Endpoint"] == fake_server.url and rows["Model listed"] == "yes"
    with pytest.raises(ValueError, match="Ollama models only"):
        external.model_setup("vega-lm")
    runs: list[bool] = []
    monkeypatch.setattr(latency, "run_once", lambda request, cold: runs.append(cold) or
                        latency.Run(request.name, request.kind, cold))
    monkeypatch.setattr(latency, "_record", lambda *args, **kwargs: {})
    latency.benchmark(1, cold=True)
    assert runs and not any(runs)  # no cold run: Vega cannot unload another server's model


def test_ollama_profiles_behave_as_before(profiles_file: Path, monkeypatch) -> None:
    started: list[str] = []
    monkeypatch.setattr(server, "is_local", lambda: True)
    monkeypatch.setattr(server, "is_running", lambda: False)
    monkeypatch.setattr(server.shutil, "which", lambda name: "/bin/ollama")
    monkeypatch.setattr(server, "start", lambda: started.append("start") or "")
    server.ensure_running("vega-coder-7b")
    assert started == ["start"]
    assert models.uses_ollama("vega-coder-7b") and models.uses_ollama("llama3.1:8b")


def test_model_process_names_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    from vega import resources

    assert resources.is_model_process("/usr/local/bin/ollama")
    assert not resources.is_model_process("LM Studio Helper")
    monkeypatch.setenv("VEGA_MODEL_PROCESSES", "LM Studio, mlx_lm")
    assert resources.is_model_process("LM Studio Helper")


# --- vega model check -----------------------------------------------------------------------------

def _check(model: str, **kwargs: Any) -> model_check.Report:
    return model_check.run(model, measure_memory=False, context_length=lambda name: None, **kwargs)


def test_check_passes_against_a_conforming_server(profiles_file: Path,
                                                  fake_server: FakeServer) -> None:
    _endpoint_profile(profiles_file, fake_server.url)
    report = _check("vega-lm")
    results = {item.name: item for item in report.items}
    assert list(results) == ["chat", "thinking", "question", "direct-edit", "tool-call",
                             "truncation", "options"]
    assert all(item.passed for item in report.items), [(i.name, i.detail) for i in report.items]
    assert report.ok and report.first_reply is not None and not report.ollama
    assert "tools: read_file" in results["tool-call"].detail
    assert "3 replies" in results["truncation"].detail
    assert "reasoning_effort=none sent with all" in results["options"].detail
    assert all(r["model"] == "served-model" for r in fake_server.requests)


def test_check_reports_each_failure(profiles_file: Path) -> None:
    fake = FakeServer(chat_answer="392", report_length=False)
    try:
        _endpoint_profile(profiles_file, fake.url)
        report = _check("vega-lm")
    finally:
        fake.close()
    results = {item.name: item for item in report.items}
    assert not report.ok
    assert results["chat"].passed is False and "392" in results["chat"].detail
    assert results["truncation"].passed is False and "'length'" in results["truncation"].detail
    assert results["question"].passed and results["tool-call"].passed


def test_check_strips_thinking(profiles_file: Path) -> None:
    fake = FakeServer(think=True)
    try:
        _endpoint_profile(profiles_file, fake.url)
        report = _check("vega-lm")
    finally:
        fake.close()
    thinking = next(item for item in report.items if item.name == "thinking")
    assert thinking.passed and "stripped" in thinking.detail


def test_check_stops_when_the_guard_says_so(profiles_file: Path, fake_server: FakeServer) -> None:
    _endpoint_profile(profiles_file, fake_server.url)
    calls: list[int] = []

    def guard() -> str | None:
        calls.append(1)
        return "free memory 5% < 20%" if len(calls) > 1 else None

    report = _check("vega-lm", before_item=guard)
    assert report.stopped and not report.ok
    assert [item.passed for item in report.items[2:6]] == [None] * 4


def test_check_command_exit_codes(profiles_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_check, "_memory", lambda report: "")
    good = FakeServer()
    try:
        _endpoint_profile(profiles_file, good.url)
        result = CliRunner().invoke(app, ["model", "check", "lm", "--no-guard"])
    finally:
        good.close()
    assert result.exit_code == 0, result.output
    assert "Ollama steps skipped" in result.output and "First reply:" in result.output
    for name in ("chat", "thinking", "question", "direct-edit", "tool-call", "truncation",
                 "options"):
        assert name in result.output
    bad = FakeServer(chat_answer="392")
    try:
        models.remove_profile("lm")
        _endpoint_profile(profiles_file, bad.url)
        result = CliRunner().invoke(app, ["model", "check", "lm", "--no-guard"])
    finally:
        bad.close()
    assert result.exit_code == 1 and "FAIL" in result.output and "Failed: chat" in result.output


def test_thinking_item_judges_stripping_not_the_answer() -> None:
    assert model_check._thinking("<think>17*23</think>391").passed
    assert model_check._thinking("392").passed  # a wrong answer is the chat item's failure
    unbalanced = model_check._thinking("17*23 is...</think>391")  # template opened the tag
    assert unbalanced.passed and "stripped" in unbalanced.detail
    assert model_check._thinking("<think>never closed").passed is False


def test_check_on_an_ollama_profile(profiles_file: Path, fake_server: FakeServer,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from vega import resources

    monkeypatch.setenv("VEGA_BASE_URL", fake_server.url)  # stands in for Ollama's /v1
    unloaded: list[str] = []
    monkeypatch.setattr(server, "unload", lambda model=None: unloaded.append(model) or True)
    monkeypatch.setattr(server, "_get_json", lambda path: {"models": [
        {"name": "vega-coder-7b:latest", "size": 5 * 1024**3, "size_vram": 5 * 1024**3}]}
        if path == "/api/ps" and unloaded == [] and len(fake_server.requests) else {})
    monkeypatch.setattr(resources, "snapshot", lambda: resources.Snapshot(
        free_pct=40, swap_used_mb=100.0, disk_free_gb=50.0, model_rss_mb=5200.0))
    report = model_check.run("vega-coder-7b", context_length=lambda name: 4096)
    results = {item.name: item for item in report.items}
    assert report.ollama and not report.ok
    assert results["options"].passed is False and "num_ctx 4096" in results["options"].detail
    assert "Ollama reports 5.0 GB loaded" in report.memory and "RSS 5,200 MB" in report.memory
    assert unloaded == ["vega-coder-7b"]  # not loaded before the check: memory given back
    ok = model_check.run("vega-coder-7b", context_length=lambda name: 8192, measure_memory=False)
    assert ok.ok, [(i.name, i.detail) for i in ok.items]
