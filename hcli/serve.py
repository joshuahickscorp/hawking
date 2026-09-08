"""An OpenAI-compatible endpoint over the PERSISTENT Hawking resident.

WHY THIS EXISTS. Open WebUI (already installed on this machine) speaks
`/v1/models` + `/v1/chat/completions` and nothing else. Hawking has two prior
OpenAI surfaces and neither can drive a browser chat:

  * `crates/hawking-serve` wants `model-*.gravity` shards. Every body on disk
    is .hq30uq4 / .f32v2 / .hgrafv01, so it has nothing to serve.
  * `tools/hcli_resident/serve_sealed.py` is correct and seal-verified, and its
    own docstring says the binary RELOADS THE MODEL PER CALL. That is a
    plumbing surface, not a chat one -- a ten-gigabyte load per message.

`make_backend_for_model` already returns a backend whose `complete(payload)`
takes an OpenAI-shaped dict, and for the native profile that backend holds a
HawkingNativeConnector wrapping a PERSISTENT ResidentProcess: one load, then
every turn is a pipe round trip. So this server is thin on purpose -- HTTP in,
`backend.complete` out. It adds no second runtime, no second model resolution
and no second state universe.

STREAMING IS SHAPE, NOT TIMING, AND IS LABELLED AS SUCH. The connector returns
a completed string; there is no incremental hook to subscribe to yet. With
`stream: true` this emits ONE content delta followed by `[DONE]` -- a valid SSE
conversation that Open WebUI renders correctly -- and stamps
`hawking.streaming: "single-chunk"` on the non-stream path plus `/health` so
nobody reads the SSE frames as evidence of token-by-token decode. Pretending
otherwise would be a false affordance, and those are bugs.

GREEDY PROFILES REFUSE SAMPLERS RATHER THAN IGNORING THEM. sealed-3.14 decodes
argmax. A request asking for temperature 0.8 is answered with a 400 that names
the accepted values, because silently serving a different sampler than the
caller asked for is how a benchmark ends up measuring something else. That rule
is `serve_sealed.py`'s and it is kept here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_PORT = 8011
DEFAULT_HOST = "127.0.0.1"

#: Values that mean "do not sample", per parameter. A greedy profile accepts
#: these and refuses everything else. `True == 1` in Python, so membership is
#: checked with an explicit type guard rather than a bare `in`.
GREEDY_OK: Dict[str, Tuple[Any, ...]] = {
    "temperature": (0, 0.0),
    "top_p": (1, 1.0),
    "top_k": (0, 1),
    "n": (1,),
    "presence_penalty": (0, 0.0),
    "frequency_penalty": (0, 0.0),
}


def _acceptable(key: str, value: Any) -> bool:
    if value is None:
        return True
    for ok in GREEDY_OK[key]:
        if type(value) is bool or type(ok) is bool:
            if type(value) is type(ok) and value == ok:
                return True
        elif value == ok:
            return True
    return False


def profile_is_greedy(model: str) -> bool:
    """True when the artifact's own profile says it does not sample.

    Read from the profile rather than assumed, so an MLX specimen served
    through the same command is not told it must be greedy.
    """
    try:
        path = Path(os.path.expanduser(str(model)))
        if path.suffix != ".json" or not path.is_file():
            return False
        gen = json.loads(path.read_text(encoding="utf-8")).get("generation") or {}
    except Exception:
        return False
    if gen.get("do_sample") is False:
        return True
    return gen.get("temperature") in (0, 0.0) and gen.get("top_k") in (1,)


def sampler_refusal(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    bad = sorted(k for k in GREEDY_OK if k in body and not _acceptable(k, body[k]))
    if not bad:
        return None
    accepted = ", ".join(f"{k}={GREEDY_OK[k][-1]!r}" for k in bad)
    return {"error": {
        "message": (
            f"this resident decodes GREEDY ARGMAX, so {bad} would ask for a sampler "
            f"it does not run. Refused rather than ignored, because silently serving "
            f"a different sampler than you asked for is how a measurement ends up "
            f"describing something else. Accepted here: {accepted} (or leave them "
            f"unset). In Open WebUI these live under Chat Controls > Advanced Params."
        ),
        "type": "unsupported_parameter",
        "param": bad[0],
    }}


def models_payload(identity: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": identity,
        "object": "model",
        # Open WebUI sorts and labels on these two; a missing `created` renders
        # the model as "Invalid Date" in the picker.
        "created": int(time.time()),
        "owned_by": "hawking",
    }
    if extra:
        row.update(extra)
    return {"object": "list", "data": [row]}


def _text_of(result: Any) -> str:
    text = getattr(result, "text", None)
    if isinstance(text, str) and text:
        return text
    raw = getattr(result, "raw", None)
    if isinstance(raw, dict):
        choices = raw.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
    return ""


def chat_payload(result: Any, identity: str, *, request_id: str) -> Dict[str, Any]:
    text = _text_of(result)
    usage = {
        "prompt_tokens": getattr(result, "prompt_tokens", None) or 0,
        "completion_tokens": getattr(result, "completion_tokens", None) or 0,
        "total_tokens": getattr(result, "total_tokens", None) or 0,
    }
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": identity,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": getattr(result, "finish_reason", None) or "stop",
        }],
        "usage": usage,
        # Named so a reader of this response cannot mistake the SSE shape on the
        # other path for evidence of incremental decode.
        "hawking": {
            "streaming": "single-chunk",
            "degraded": list(getattr(result, "degraded", None) or []),
        },
    }


def stream_frames(result: Any, identity: str, *, request_id: str):
    """SSE frames for one completed answer: role, one content delta, stop, DONE."""
    created = int(time.time())

    def frame(delta: Dict[str, Any], finish: Optional[str]) -> bytes:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": identity,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return b"data: " + json.dumps(chunk).encode() + b"\n\n"

    yield frame({"role": "assistant"}, None)
    text = _text_of(result)
    if text:
        yield frame({"content": text}, None)
    yield frame({}, getattr(result, "finish_reason", None) or "stop")
    yield b"data: [DONE]\n\n"


def make_handler(backend: Any, identity: str, *, greedy: bool, health: Dict[str, Any]):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802 - CORS preflight from a browser
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.rstrip("/") or "/"
            if path in ("/v1/models", "/models"):
                return self._send(200, models_payload(identity))
            if path in ("/health", "/"):
                return self._send(200, health)
            self._send(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self._send(404, {"error": {
                    "message": f"no route {self.path}; this server serves "
                               f"/v1/models and /v1/chat/completions"}})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception as exc:
                return self._send(400, {"error": {"message": f"bad json: {exc}"}})
            if not isinstance(body, dict):
                return self._send(400, {"error": {"message": "body must be an object"}})
            if greedy:
                refusal = sampler_refusal(body)
                if refusal is not None:
                    return self._send(400, refusal)
            messages = body.get("messages") or []
            if not messages:
                return self._send(400, {"error": {
                    "message": "no messages: send {'messages':[{'role':'user',"
                               "'content':'...'}]}"}})
            payload = {k: v for k, v in body.items() if k != "stream"}
            payload.setdefault("model", identity)
            request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            try:
                result = backend.complete(payload)
            except Exception as exc:
                return self._send(502, {"error": {
                    "message": f"{type(exc).__name__}: {exc}",
                    "type": "resident_error"}})
            if not body.get("stream"):
                return self._send(200, chat_payload(result, identity, request_id=request_id))
            # An SSE body has no Content-Length, so under HTTP/1.1 keep-alive a
            # client cannot tell where it ends and waits until it times out --
            # which is exactly what a browser chat looks like when it hangs
            # forever on a reply that was in fact delivered. Close the
            # connection to frame the end of the stream.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.close_connection = True
            for chunk in stream_frames(result, identity, request_id=request_id):
                self.wfile.write(chunk)
                self.wfile.flush()

    return Handler


def build_server(model: str, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 ready_timeout: float = 600.0):
    """Spawn the resident, wait for it, and return (httpd, identity, health)."""
    from .runtime_iface import make_backend_for_model

    backend = make_backend_for_model(model)
    backend.spawn()
    if not backend.ready(ready_timeout):
        raise RuntimeError(
            f"the resident did not become ready within {ready_timeout:.0f}s. "
            f"Its log tail is the evidence: "
            f"{getattr(backend, 'log_tail', lambda *_: '(no log)')()!s:.400}"
        )
    ident = backend.identity() if hasattr(backend, "identity") else {}
    identity = str(ident.get("resident_identity") or ident.get("model")
                   or Path(str(model)).stem)
    greedy = profile_is_greedy(model)
    health = {
        "status": "ok",
        "resident": identity,
        "model": str(model),
        "sampling": "greedy-argmax" if greedy else "profile-default",
        "streaming": "single-chunk (the connector returns a completed string; "
                     "SSE shape is provided so browser clients render, not as "
                     "evidence of incremental decode)",
        "endpoints": ["/v1/models", "/v1/chat/completions", "/health"],
        "pid": ident.get("pid"),
    }
    httpd = ThreadingHTTPServer(
        (host, port), make_handler(backend, identity, greedy=greedy, health=health))
    httpd.backend = backend  # type: ignore[attr-defined]
    return httpd, identity, health


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hcli serve",
        description="OpenAI-compatible endpoint over the persistent Hawking resident.")
    ap.add_argument("--model", default=None,
                    help="model artifact or native profile (default: the sealed profile)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--ready-timeout", type=float, default=600.0)
    a = ap.parse_args(list(argv or []))

    model = a.model or str(Path(__file__).resolve().parent / "hawking-native.sealed-3.14.json")
    httpd, identity, health = build_server(
        model, host=a.host, port=a.port, ready_timeout=a.ready_timeout)
    print(json.dumps({"listening": f"http://{a.host}:{a.port}",
                      "openai_base_url": f"http://{a.host}:{a.port}/v1",
                      **health}), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            httpd.backend.stop()  # type: ignore[attr-defined]
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
