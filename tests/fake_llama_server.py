"""A stand-in for llama.cpp's `llama-server` (no model, no GPU): the endpoints and fields Vega's
llama serving layer (vega_lab.llama_serve) uses, over a deterministic toy "model".

    python tests/fake_llama_server.py --host 127.0.0.1 --port N [any llama-server flags]

FakeModel is importable for in-process tests. Its rules mirror what the real server is believed to
do (option B of the inference-serving options plan), so the tests can check Vega's side:

- text: one token per whitespace-separated word ("w12" <-> 12); special tokens are words too;
- the next token is a function of the whole context (prompt + generated so far), so it depends on
  the cached state exactly as a real model does; EOS comes once the context holds `eos_after`
  generated-range tokens (a position, so one call and a chunked continuation end at the same token);
- the repeat penalty looks at the last `repeat_last_n` tokens of the context INCLUDING the prompt
  (`penalty_scope="context"`, what llama-server does: it feeds the prompt through the sampler), or
  only at the generated tokens (`"generated"`, a server that does not: chunked continuation then
  differs from one call, which the tests show the exactness check catches);
- the slot cache keeps every token it evaluated: the prompt plus all generated tokens but the last
  one (sampled, never decoded). `cache_n` = the common prefix with the new prompt, at most
  len(prompt) - 1 (at least one token is always evaluated); with `recurrent=True` a prompt that
  diverges before the end of the cache cannot reuse anything (no checkpoint at that point);
- `checkpoints=True` (a hybrid model as llama-server runs it):
  each call that evaluates a prompt leaves context checkpoints at its end -4 and at end -(ubatch - 4)
  (when inside the evaluated part); a prompt that diverges inside the cache restores the newest
  checkpoint at or below the common prefix (the others above it are erased), or none = full
  re-processing; an exact extension of the slot needs none. With `log_path` it writes the
  verbosity-4 lines the probe and the log parsers read (created / restored / forcing);
- slot files (`/slots/0?action=save|restore|erase`, only with --slot-save-path): save writes the slot's
  tokens to a file in that folder; a restored slot has its tokens and NO checkpoints (the unverified
  case of the plan: an exact extension reuses it, a diverging one cannot).

Started as a script it also: answers /health with 503 for --load-s seconds (loading), then 200;
writes its argv to $FAKE_LLAMA_ARGV when set; ignores SIGTERM when $FAKE_LLAMA_IGNORE_TERM=1 (so
the SIGKILL path is tested); refuses every grammar with HTTP 400 when $FAKE_LLAMA_REJECT_GRAMMAR=1.
Grammar keys are checked as the real server checks them (FakeModel.grammar_error), never applied:
the toy model writes no tool calls.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

EOS = 2


# llama-server's own sampling defaults (build 10969 --help); the final reply reports what it applied as
# generation_settings, as the real server does
SERVER_SAMPLING = {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "typical_p": 1.0,
                   "repeat_penalty": 1.0, "repeat_last_n": 64, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                   "dry_multiplier": 0.0, "xtc_probability": 0.0, "mirostat": 0, "top_n_sigma": -1.0, "seed": -1,
                   "samplers": ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc",
                                "temperature"]}


class FakeRejected(Exception):
    """A request the real server answers with HTTP 400 (its error text in args[0])."""


def encode(text: str) -> list[int]:
    out = []
    for word in text.split():
        out.append(int(word[1:]) if word.startswith("w") and word[1:].isdigit() else 1000 + sum(map(ord, word)) % 997)
    return out


def decode(tokens: list[int]) -> str:
    return "".join(f"w{t} " for t in tokens if t != EOS)


class FakeModel:
    def __init__(self, penalty_scope: str = "context", recurrent: bool = False, eos_after: int = 10_000,
                 reject_grammar: bool = False, checkpoints: bool = False, ubatch: int = 64,
                 log_path: str | None = None, slot_dir: str | None = None) -> None:
        self.penalty_scope, self.recurrent, self.eos_after = penalty_scope, recurrent, eos_after
        self.reject_grammar = reject_grammar
        self.checkpoints, self.ubatch, self.log_path, self.slot_dir = checkpoints, ubatch, log_path, slot_dir
        self.slot: list[int] = []
        self.ckpts: list[int] = []  # tokens of state each checkpoint holds
        self.calls: list[dict] = []
        self.rejected: list[str] = []

    def log(self, line: str) -> None:
        if self.log_path:
            with open(self.log_path, "a") as handle:
                handle.write(line + "\n")

    def next_token(self, context: list[int], generated: list[int], body: dict) -> int:
        if sum(1 for t in context if 100 <= t < 140) >= self.eos_after:  # by position, not per call
            return EOS
        cand = (sum(context[-5:]) * 31 + len(context) * 7) % 40 + 100
        penalty = float(body.get("repeat_penalty", 1.0) or 1.0)
        window = int(body.get("repeat_last_n", 64) or 0)
        if penalty != 1.0 and window:  # the first candidate not in the window (as a penalty would push it)
            seen = set((context if self.penalty_scope == "context" else generated)[-window:])
            cand = next(((cand - 100 + k) % 40 + 100 for k in range(40) if (cand - 100 + k) % 40 + 100 not in seen),
                        cand)
        temp = float(body.get("temperature", 0.8) or 0.0)
        if temp > 0:  # "sampling": the seed moves the choice, deterministically per call
            seed = int(body.get("seed", -1))
            cand = (cand + (seed if seed >= 0 else int(time.time() * 1000)) + len(generated)) % 40 + 100
        return cand

    def cached(self, prompt: list[int]) -> int:
        n = 0
        for a, b in zip(self.slot, prompt):
            if a != b:
                break
            n += 1
        if self.checkpoints and n < len(self.slot):  # roll back to a checkpoint at or below the prefix
            for p in [p for p in self.ckpts if p > n]:
                self.log(f"srv  erased invalidated context checkpoint (pos_min = {p - 1}, n_tokens = {p})")
            self.ckpts = [p for p in self.ckpts if p <= n]
            usable = [p for p in self.ckpts if p <= len(prompt) - 1]
            if not usable:
                self.log("slot update_slots: forcing full prompt re-processing due to lack of cache data "
                         "(likely due to SWA or hybrid/recurrent memory)")
                return 0
            self.log(f"slot update_slots: restored context checkpoint (pos_min = {max(usable) - 1}, "
                     f"n_past = {max(usable)})")
            return max(usable)
        if self.recurrent and n < len(self.slot):
            return 0  # the state cannot be rolled back to a point inside the cache
        return min(n, max(0, len(prompt) - 1))

    def made_checkpoints(self, start: int, end: int) -> None:
        """llama-server's placement: the prompt's end -4 and end -(ubatch - 4), inside [start, end]."""
        if not self.checkpoints:
            return
        for p in sorted({end - 4, end - (self.ubatch - 4)}):
            if start < p <= end and p not in self.ckpts:
                self.ckpts.append(p)
                self.log(f"slot update_slots: created context checkpoint {len(self.ckpts)} of 32 (pos_min = {p - 1}, "
                         f"pos_max = {p - 1}, n_tokens = {p}, size = 55.372 MiB)")

    def slot_action(self, action: str, filename: str) -> tuple[int, dict]:
        """/slots/0?action=...: (status, reply) as llama-server answers them."""
        if not self.slot_dir:
            return 501, {"error": {"code": 501, "message": "This server does not support slots action."}}
        if action == "erase":
            n, self.slot, self.ckpts = len(self.slot), [], []
            return 200, {"id_slot": 0, "n_erased": n}
        if not filename or "/" in filename:
            return 400, {"error": {"code": 400, "message": "Invalid filename"}}
        path = os.path.join(self.slot_dir, filename)
        if action == "save":
            with open(path, "w") as handle:
                json.dump(self.slot, handle)
            return 200, {"id_slot": 0, "filename": filename, "n_saved": len(self.slot),
                         "n_written": os.path.getsize(path), "timings": {"save_ms": 1.5}}
        if action == "restore":
            with open(path) as handle:
                self.slot = [int(t) for t in json.load(handle)]
            self.ckpts = []  # the save format holds no checkpoints (the plan's unverified case)
            return 200, {"id_slot": 0, "filename": filename, "n_restored": len(self.slot),
                         "n_read": os.path.getsize(path), "timings": {"restore_ms": 2.5}}
        return 400, {"error": {"code": 400, "message": "Invalid action"}}

    def grammar_error(self, body: dict) -> str | None:
        """Why the real server would refuse this request's grammar keys (its own error texts, build
        10969's strings), or None. A word trigger that is one special token (`<...>` here) must be
        listed in preserved_tokens; `reject_grammar` refuses every grammar (a build that cannot
        parse ours)."""
        if body.get("grammar") and body.get("json_schema"):
            return "Cannot use both json_schema and grammar"
        if body.get("grammar_lazy") and not body.get("grammar_triggers"):
            return "Error: no triggers set for lazy grammar!"
        preserved = set(body.get("preserved_tokens") or [])
        for trig in body.get("grammar_triggers") or []:
            word = str(trig.get("value", ""))
            if trig.get("type") == 1 and word.startswith("<") and word.endswith(">") and word not in preserved:
                return f"Grammar trigger word should be marked as preserved token: {word}"
        grammar = body.get("grammar")
        if grammar and (self.reject_grammar or "root ::=" not in grammar):
            return "failed to parse grammar"
        return None

    def generate(self, body: dict) -> tuple[list[int], dict]:
        problem = self.grammar_error(body)
        if problem:
            self.rejected.append(problem)
            raise FakeRejected(problem)
        prompt = body["prompt"]
        prompt = encode(prompt) if isinstance(prompt, str) else [int(t) for t in prompt]
        cache_n = self.cached(prompt)
        n_predict = int(body.get("n_predict", -1))
        stops = [s for s in body.get("stop") or [] if s]
        generated: list[int] = []
        stop_type, word = "none", ""
        while n_predict < 0 or len(generated) < n_predict:
            tok = self.next_token(prompt + generated, generated, body)
            if tok == EOS:
                stop_type = "eos"
                break
            generated.append(tok)
            text = decode(generated)
            hit = next((s for s in stops if s in text), None)
            if hit:
                stop_type, word = "word", hit
                break
        else:
            stop_type = "limit"
        self.made_checkpoints(cache_n, len(prompt))
        self.slot = prompt + generated[:-1] if generated else list(prompt)
        timings = {"cache_n": cache_n, "prompt_n": len(prompt) - cache_n, "prompt_ms": (len(prompt) - cache_n) * 0.5,
                   "predicted_n": len(generated), "predicted_ms": len(generated) * 10.0}
        self.calls.append({"prompt_len": len(prompt), "cache_n": cache_n, "n": len(generated), "stop_type": stop_type})
        settings = {**SERVER_SAMPLING, **{k: body[k] for k in SERVER_SAMPLING if k in body}}
        grammar = {"grammar": str(body.get("grammar") or ""), "grammar_lazy": bool(body.get("grammar_lazy")),
                   "grammar_triggers": list(body.get("grammar_triggers") or []),
                   "preserved_tokens": list(body.get("preserved_tokens") or [])}
        return generated, {"stop_type": stop_type, "stopping_word": word, "timings": timings,
                           "generation_settings": {**settings, **grammar, "n_predict": n_predict, "stop": stops},
                           "tokens_cached": cache_n + len(prompt) - cache_n + len(generated),
                           "tokens_evaluated": len(prompt), "tokens_predicted": len(generated)}


def make_handler(model: FakeModel, ready_at: float) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.startswith("/health"):
                if time.monotonic() < ready_at:
                    self._send(503, {"error": {"message": "Loading model", "code": 503}})
                else:
                    self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "no route"})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path.startswith("/slots/"):
                action = self.path.partition("action=")[2]
                self._send(*model.slot_action(action, str(body.get("filename") or "")))
            elif self.path == "/tokenize":
                self._send(200, {"tokens": encode(body.get("content", ""))})
            elif self.path == "/detokenize":
                self._send(200, {"content": decode(body.get("tokens", []))})
            elif self.path == "/completion":
                try:
                    tokens, info = model.generate(body)
                except FakeRejected as exc:
                    self._send(400, {"error": {"code": 400, "message": str(exc), "type": "invalid_request_error"}})
                    return
                if not body.get("stream"):
                    self._send(200, {"content": decode(tokens), "tokens": tokens, "stop": True, **info})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                for tok in tokens:
                    self.wfile.write(b"data: " + json.dumps({"content": decode([tok]), "tokens": [tok],
                                                              "stop": False}).encode() + b"\n\n")
                self.wfile.write(b"data: " + json.dumps({"content": "", "tokens": [], "stop": True,
                                                          **info}).encode() + b"\n\n")
                self.wfile.flush()
            else:
                self._send(404, {"error": "no route"})

    return Handler


def main(argv: list[str]) -> int:
    def opt(name: str, default: str) -> str:
        return argv[argv.index(name) + 1] if name in argv else default

    if os.environ.get("FAKE_LLAMA_ARGV"):
        with open(os.environ["FAKE_LLAMA_ARGV"], "w") as handle:
            json.dump(argv, handle)
    if os.environ.get("FAKE_LLAMA_IGNORE_TERM") == "1":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready_at = time.monotonic() + float(opt("--load-s", "0.3"))
    model = FakeModel(reject_grammar=os.environ.get("FAKE_LLAMA_REJECT_GRAMMAR") == "1",
                      slot_dir=opt("--slot-save-path", "") or None)
    httpd = ThreadingHTTPServer((opt("--host", "127.0.0.1"), int(opt("--port", "8080"))),
                                make_handler(model, ready_at))
    httpd.daemon_threads = True
    print("fake llama-server listening", flush=True)
    httpd.serve_forever(poll_interval=0.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
