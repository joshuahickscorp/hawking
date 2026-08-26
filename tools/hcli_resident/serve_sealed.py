#!/usr/bin/env python3
"""An OpenAI-compatible endpoint in front of the SEALED resident.

WHY THIS EXISTS. S032 §18 asks HCLI to make sealed-3.14 its canonical cheap
resident thinker. HCLI talks to an OpenAI-compatible endpoint
(hcli/delegate.py default_caller). The sealed resident is a Rust greedy example
with no server, and `hawking gravity serve` wants `model-*.gravity` shards that
do not exist -- every Hawking body on disk is .hq30uq4 / .f32v2 / .hgrafv01.
That is the exact gap that has kept the fast body and the OpenAI surface from
talking, and this closes it without inventing a second runtime.

WHAT MAKES IT HONEST. A shim can silently serve a DIFFERENT configuration than
the one the seal binds -- a different artifact, a different binary, the wrong
chat-template arm -- and every number downstream would then describe something
nobody sealed. So THE SERVER REFUSES TO START unless what is on disk matches
receipts/headless/HCLI_RESIDENT_SEAL.json:

  * the runtime binary's sha256 matches the sealed one, byte for byte
  * the tokenizer and chat template hashes match
  * the artifact inventory matches
  * the fusion levers it sets reproduce the sealed 628-dispatch graph
  * the chat-template arm is the sealed one (pre_closed_think)

A mismatch is a REFUSAL naming the field, not a warning. `--allow-unsealed` runs
anyway and stamps every response with `sealed: false` so the divergence travels
with the output rather than being lost at startup.

WHAT IT IS NOT. Greedy decode only -- the sealed path is argmax, so temperature,
top_p and top_k in a request are REFUSED rather than ignored, because silently
serving a different sampler than the caller asked for is how a benchmark ends up
measuring something else. One request at a time: the binary loads the model per
call, so this is a correctness and plumbing surface, NOT a throughput one, and
the seal's own TPS figures are the throughput authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEAL = REPO / "receipts/headless/HCLI_RESIDENT_SEAL.json"

SEALED_LEVERS = {
    "HAWKING_QWEN38_FUSE_ADD_RMSNORM": "1",
    "HAWKING_QWEN38_FUSE_GQA_QKV": "1",
    "HAWKING_QWEN38_FUSE_DN_INPROJ": "1",
    "HAWKING_QWEN38_FUSE_MLP": "swiglu",
}


class Unsealed(RuntimeError):
    """What is on disk is not what the seal binds."""


def _sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def load_seal(path: Path = SEAL) -> dict:
    return json.loads(path.read_text())


def check_seal(seal: dict, *, repo: Path = REPO) -> dict:
    """Every mismatch, not the first. A list of one is still a refusal, and a
    reader who fixes the first of three learns that the hard way otherwise."""
    f = seal["fields"]
    art = Path(f["artifact_root"]["value"])
    binp = repo / f["runtime_binary"]["value"]
    bad = []
    if not art.is_dir():
        bad.append(f"artifact_root {art} does not exist")
    if not binp.is_file():
        bad.append(f"runtime binary {binp} does not exist")
    if binp.is_file():
        got = _sha16(binp)
        want = f["runtime_binary_sha256_16"]["value"]
        if got != want:
            bad.append(f"runtime binary sha {got} != sealed {want}")
    for name, key in (("tokenizer.json", "tokenizer_sha256_16"),
                      ("chat_template.jinja", "chat_template_sha256_16")):
        p = art / name
        if not p.is_file():
            bad.append(f"{name} missing from the artifact")
            continue
        got, want = _sha16(p), f[key]["value"]
        if got != want:
            bad.append(f"{name} sha {got} != sealed {want}")
    if f["chat_template_arm"]["value"] != "pre_closed_think":
        bad.append(f"sealed arm is {f['chat_template_arm']['value']!r}, this server "
                   f"renders pre_closed_think")
    if seal["graph"]["dispatches_per_decode_token"] != 628:
        bad.append("sealed graph is not the 628-dispatch one this server configures")
    for k, v in SEALED_LEVERS.items():
        if k not in seal["graph"]["levers"] and f"{k}={v}" not in seal["graph"]["levers"]:
            bad.append(f"sealed graph does not name lever {k}={v}")
    return {"sealed": not bad, "mismatches": bad}


def render(messages, tokenizer_dir: Path) -> str:
    """The ARTIFACT'S OWN chat template, at the SEALED arm. Not a template this
    file writes -- rendering it here would be a second source of truth for the
    thing the seal hashes."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def generate(prompt: str, *, seal: dict, max_tokens: int, repo: Path = REPO) -> dict:
    f = seal["fields"]
    art = Path(f["artifact_root"]["value"])
    binp = repo / f["runtime_binary"]["value"]
    env = dict(os.environ); env.update(SEALED_LEVERS)
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        out = Path(tf.name)
    try:
        t0 = time.time()
        pr = subprocess.run(
            [str(binp), "--artifact-root", str(art),
             "--tokenizer", str(art / "tokenizer.json"),
             "--prompt", prompt, "--raw-prompt",
             "--max-new-tokens", str(max_tokens),
             "--max-seq-len", str(max_tokens + 512), "--out", str(out)],
            env=env, capture_output=True, text=True, timeout=1800)
        if pr.returncode != 0:
            raise RuntimeError(f"resident exited {pr.returncode}: {pr.stderr[-400:]}")
        d = json.loads(out.read_text())
    finally:
        out.unlink(missing_ok=True)
    raw = d.get("generated_text") or ""
    # The sealed arm pre-closes the think block, so a well-behaved reply has none.
    # If one appears anyway the reply is what FOLLOWS it -- and an UNTERMINATED
    # block means the model never left reasoning and produced no answer, which is
    # an empty reply and not prose to be scored. Same rule as the capability
    # harness, deliberately, so the server and the suite cannot disagree.
    if "</think>" in raw:
        text, leaked = raw.split("</think>", 1)[1], True
    elif raw.lstrip().startswith("<think>"):
        text, leaked = "", True
    else:
        text, leaked = raw, False
    return {"text": text.strip(), "raw": raw, "think_seen": leaked,
            "completion_tokens": len(d.get("new_token_ids") or []),
            "prompt_tokens": d.get("prompt_len"),
            "fallbacks": d.get("fallbacks"),
            "dense_w_materialized": d.get("dense_w_materialized"),
            "wall_s": round(time.time() - t0, 3)}


# key -> the ONLY value this server will accept for it. Anything else is refused.
#
# A first version wrote `body[k] not in (None, 1, False)` and `stream: true`
# SAILED THROUGH, because in Python `True == 1`. The request was then served
# NON-STREAMED with a 200 -- silently giving the caller something other than what
# it asked for, which is the exact failure the refusal exists to prevent. The
# comparison is now TYPE-AWARE.
UNSUPPORTED = {
    "temperature": (0, 0.0),      # greedy argmax is temperature 0
    "top_p": (1, 1.0),
    "top_k": (0, 1),
    "n": (1,),
    "presence_penalty": (0, 0.0),
    "frequency_penalty": (0, 0.0),
    "logit_bias": (None, {}),
    "stream": (False,),
}


def _acceptable(key: str, value) -> bool:
    """`True == 1` in Python, so a bare `in` test is not enough."""
    for ok in UNSUPPORTED[key]:
        if type(value) is bool or type(ok) is bool:
            if type(value) is type(ok) and value == ok:
                return True
        elif value == ok:
            return True
    return value is None


def handle_chat(body: dict, *, seal: dict, sealed: bool) -> tuple:
    """Returns (status, payload)."""
    bad = [k for k in UNSUPPORTED if k in body and not _acceptable(k, body[k])]
    if bad:
        return 400, {"error": {
            "message": f"the sealed decode path is GREEDY ARGMAX; {bad} would ask for a "
                       f"sampler this server does not run. Refused rather than ignored, "
                       f"because silently serving a different sampler than the caller "
                       f"asked for is how a benchmark measures something else.",
            "type": "unsupported_parameter"}}
    msgs = body.get("messages") or []
    if not msgs:
        return 400, {"error": {"message": "no messages", "type": "invalid_request"}}
    art = Path(seal["fields"]["artifact_root"]["value"])
    prompt = render(msgs, art)
    g = generate(prompt, seal=seal, max_tokens=int(body.get("max_tokens") or 512))
    return 200, {
        "id": f"chatcmpl-sealed-{int(time.time()*1000)}",
        "object": "chat.completion", "created": int(time.time()),
        "model": seal["resident"],
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": g["text"]}}],
        "usage": {"prompt_tokens": g["prompt_tokens"],
                  "completion_tokens": g["completion_tokens"],
                  "total_tokens": (g["prompt_tokens"] or 0) + g["completion_tokens"]},
        # Identity travels with every response. A caller that logs only the text
        # cannot later say which body produced it.
        "hawking": {
            "sealed": sealed,
            "resident": seal["resident"],
            "artifact_inventory_sha": seal["fields"]["artifact_inventory_sha"]["value"],
            "runtime_binary_sha256_16": seal["fields"]["runtime_binary_sha256_16"]["value"],
            "chat_template_arm": seal["fields"]["chat_template_arm"]["value"],
            "dispatches_per_decode_token": seal["graph"]["dispatches_per_decode_token"],
            "fallbacks": g["fallbacks"],
            "dense_w_materialized": g["dense_w_materialized"],
            "think_block_seen": g["think_seen"],
            "wall_s": g["wall_s"],
            "throughput_authority": "the seal, not this server -- the binary reloads the "
                                    "model per call, so wall_s here is not a TPS figure",
        },
    }


def make_handler(seal: dict, sealed: bool):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            b = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", "/v1/models"):
                return self._send(200, {"object": "list", "data": [
                    {"id": seal["resident"], "object": "model",
                     "sealed": sealed}]})
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": {"message": "not found"}})
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception as e:
                return self._send(400, {"error": {"message": f"bad json: {e}"}})
            try:
                code, payload = handle_chat(body, seal=seal, sealed=sealed)
            except Exception as e:
                code, payload = 500, {"error": {
                    "message": f"{type(e).__name__}: {e}", "type": "resident_error"}}
            self._send(code, payload)

        def log_message(self, *a):
            pass
    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seal", default=str(SEAL))
    ap.add_argument("--allow-unsealed", action="store_true",
                    help="serve even when disk does not match the seal; every response "
                         "is then stamped sealed:false")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    seal = load_seal(Path(a.seal))
    v = check_seal(seal)
    for m in v["mismatches"]:
        print(f"SEAL MISMATCH: {m}", file=sys.stderr)
    if not v["sealed"] and not a.allow_unsealed:
        print("REFUSED: disk does not match the seal. Pass --allow-unsealed to serve "
              "anyway; every response will be stamped sealed:false.", file=sys.stderr)
        return 2
    print(json.dumps({"sealed": v["sealed"], "resident": seal["resident"],
                      "dispatches_per_decode_token":
                          seal["graph"]["dispatches_per_decode_token"],
                      "arm": seal["fields"]["chat_template_arm"]["value"],
                      "port": a.port}))
    if a.check_only:
        return 0 if v["sealed"] else 1
    HTTPServer(("127.0.0.1", a.port), make_handler(seal, v["sealed"])).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
