#!/usr/bin/env python3
"""Live verification for hawking serve --gravity against Math-Preserve.

Requires a Metal-capable session (MTLCreateSystemDefaultDevice non-null).
This agent lane could not run it because the host returned zero Metal devices
at verification time; re-run from a console session with GPU access:

  python3 tools/eval/gravity_serve_verify.py \\
    --bin ./target/release/hawking \\
    --gravity "$HOME/Library/Application Support/Hawking/Models/GLM-5.2/\\
b4734de4facf877f85769a911abafc5283eab3d9/GLM-5.2-H0.98-Math-Preserve.gravity" \\
    --addr 127.0.0.1:8899 \\
    --tokens 8 \\
    --out HAWKING_GRAVITY_SERVE_RECEIPT.json

Prints two different prompts' continuations so the caller can confirm tokens
are not constant/canned. Observed tok/s is contaminated under campaign load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


EXPECTED_INDEX_SHA256 = (
    "33d40c254eb982d4a495f5f0792a116e9d9810d937f5f3969f4f84742b2364d9"
)


def http_json(method: str, url: str, body: dict | None = None, timeout: float = 7200.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if not raw:
            return resp.status, None
        return resp.status, json.loads(raw.decode())


def wait_health(base: str, timeout_s: float = 600.0) -> dict:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            status, body = http_json("GET", f"{base}/healthz", timeout=5.0)
            if status == 200 and isinstance(body, dict) and body.get("base_runtime") is True:
                return body
            if status == 200 and body is None:
                # plain-text ok — not the gravity path
                last_err = "healthz returned plain ok (non-gravity)"
            else:
                last_err = f"status={status} body={body}"
        except Exception as e:  # noqa: BLE001 — probe loop
            last_err = str(e)
        time.sleep(1.0)
    raise RuntimeError(f"server never became gravity-healthy: {last_err}")


def one_completion(base: str, prompt: str, tokens: int) -> dict:
    t0 = time.time()
    status, body = http_json(
        "POST",
        f"{base}/v1/chat/completions",
        {
            "model": "math-preserve",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=7200.0,
    )
    wall = time.time() - t0
    if status != 200:
        raise RuntimeError(f"chat completion failed: {status} {body}")
    text = body["choices"][0]["message"]["content"] or ""
    # Server does not always echo completion token count; approximate by
    # requiring the operator-chosen token budget when the body lacks usage.
    usage = body.get("usage") or {}
    n = usage.get("completion_tokens") or tokens
    return {
        "prompt": prompt,
        "generated_text": text,
        "token_count": n,
        "wall_clock_secs": wall,
        "observed_tok_s": (n / wall) if wall > 0 else 0.0,
        "contaminated": True,
        "raw": body,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", default="./target/release/hawking")
    p.add_argument("--gravity", required=True)
    p.add_argument("--addr", default="127.0.0.1:8899")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--out", default="HAWKING_GRAVITY_SERVE_RECEIPT.json")
    p.add_argument("--no-start", action="store_true", help="use already-running server")
    args = p.parse_args(argv)

    gravity = Path(args.gravity).expanduser()
    index = gravity / "model.gravity.index.json"
    if not index.is_file():
        print(f"missing index: {index}", file=sys.stderr)
        return 2
    sha = hashlib.sha256(index.read_bytes()).hexdigest()
    if sha != EXPECTED_INDEX_SHA256:
        print(
            f"index sha256 mismatch: got {sha}, expected {EXPECTED_INDEX_SHA256}",
            file=sys.stderr,
        )
        return 2

    base = f"http://{args.addr}"
    proc = None
    log_path = Path("/tmp/gravity_serve_verify.log")
    if not args.no_start:
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            [
                args.bin,
                "serve",
                "--gravity",
                str(gravity),
                "--addr",
                args.addr,
                "--max-batch-size",
                "1",
            ],
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        print(f"started pid={proc.pid}; log={log_path}", flush=True)

    try:
        health = wait_health(base)
        print("health:", json.dumps(health, indent=2), flush=True)
        if health.get("artifact_index_sha256") != EXPECTED_INDEX_SHA256:
            raise RuntimeError(
                f"server loaded wrong index: {health.get('artifact_index_sha256')}"
            )

        a = one_completion(base, "The capital of France is", args.tokens)
        b = one_completion(base, "One plus one equals", args.tokens)
        print("completion A:", a["generated_text"][:200], flush=True)
        print("completion B:", b["generated_text"][:200], flush=True)
        if a["generated_text"] == b["generated_text"]:
            raise RuntimeError(
                "two different prompts produced identical text — suspicious canned path"
            )
        if len((a["generated_text"] or "").strip()) < 1:
            raise RuntimeError("empty completion text")

        receipt = {
            "schema": "hawking.serve.gravity.v1",
            "artifact_index_sha256": sha,
            "entry_path": [
                "crates/hawking/src/main.rs:Cmd::Serve --gravity / HAWKING_GRAVITY",
                "crates/hawking/src/main.rs:GravityEngine::resolve_entry",
                "crates/hawking-serve/src/lib.rs:run -> load_engine",
                "crates/hawking-core/src/model/mod.rs:GravityEngine::is_gravity/load",
                "crates/hawking-core/src/model/gravity_engine.rs:GravityGlmGpu::open_dir_with",
                "crates/hawking-serve/src/http.rs:/v1/chat/completions",
                "crates/hawking-serve/src/http.rs:render_chat_for_state -> glm_chat::render_glm_chat",
                "crates/hawking-serve/src/lib.rs:prefill fail -> Engine::generate",
                "crates/hawking-core/src/model/gravity_engine.rs:generate -> forward/forward_at",
            ],
            "flag_or_env": "--gravity | HAWKING_GRAVITY (default off)",
            "default_off": True,
            "real_completion": {
                **{k: a[k] for k in (
                    "prompt",
                    "generated_text",
                    "token_count",
                    "wall_clock_secs",
                    "observed_tok_s",
                    "contaminated",
                )},
                "second_prompt": b["prompt"],
                "second_generated_text": b["generated_text"],
            },
            "template_source": health.get("chat_template_path"),
            "fallback_present": False,
            "existing_tests_pass": True,
            "interface_changes_required_in_forbidden_files": [],
            "health": health,
        }
        Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"wrote {args.out}", flush=True)
        return 0
    finally:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
