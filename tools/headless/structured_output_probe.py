#!/usr/bin/env python3
"""G003 probe: is HCLI's structured-output failure caused by unconstrained decoding
plus Qwen3 thinking-token burn?

Hypothesis (from .haider/bootstrap-director-v6/runs/*/hcli.log, every epoch):
    "Model did not return a valid structured JSON object"
    ~314 s per failed call, HCLI_MODEL_TOKENS=6500, machine single-decode ~21 tok/s
    => 6500 tok / 314 s = 20.7 tok/s  => the call is exhausting max_tokens, not erroring.
    Qwen3 emits <think> reasoning first; with no grammar constraint it never reaches
    the JSON object before the budget runs out.

Arms (paired, alternating, N reps each so page-cache and thermal drift cancel):
    A  baseline      exactly HCLI's current payload
    B  no_think      + chat_template_kwargs.enable_thinking=false
    C  schema        + response_format json_schema (constrained decoding)
    D  schema+nothink both

Prints per-arm: valid-JSON rate, wall seconds, completion tokens, tok/s.
Writes receipts/headless/STRUCTURED_OUTPUT_PROBE.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = os.path.expanduser(
    "~/models/qwen3.8-27b-abliterated/Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf")

# Verbatim from hcli/engine.py _call_model (build-20260821-155332).
SYSTEM = """You are the native HCLI local engineering worker.

MODELS THINK.
TOOLS KNOW.
DISK STATE IS AUTHORITY.
CONTEXT IS A CACHE.

Return exactly one JSON object and nothing else.

For a read-only request:
{
  "kind": "answer",
  "content": "concise final answer",
  "operations": [],
  "tests": []
}

For a requested code/file change:
{
  "kind": "mutation",
  "content": "concise description of what changed",
  "operations": [
    {
      "op": "replace|create|replace_file|insert_before|insert_after|append",
      "path": "workspace/relative/path",
      "old_text": "exact anchor when required",
      "new_text": "new content"
    }
  ],
  "tests": ["optional safe workspace-relative Python test paths"]
}

Rules:
- maximum 20 operations
- paths are workspace-relative
- never modify .git/**
- never return shell commands as mutation authority
- use exact old_text anchors
- use create only for nonexistent files
- if asked only a question, do not mutate
- do not include reasoning_content, hidden reasoning, chain-of-thought, or <think>
"""

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["answer", "mutation"]},
        "content": {"type": "string"},
        "operations": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string",
                           "enum": ["replace", "create", "replace_file",
                                    "insert_before", "insert_after", "append"]},
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["op", "path"],
                "additionalProperties": False,
            },
        },
        "tests": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["kind", "content", "operations", "tests"],
    "additionalProperties": False,
}


def extract_json(text: str):
    """Mirror engine._extract_json_object's job: find one JSON object in the reply."""
    if not text:
        return None
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    depth = 0; start = -1
    for i, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(t[start:i + 1])
                except Exception:
                    start = -1
    return None


def wait_ready(port: int, timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def call(port: int, arm: str, user: str, max_tokens: int, timeout: float):
    payload = {
        "model": "local",
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
        "temperature": 0.15,
        "max_tokens": max_tokens,
    }
    if arm in ("no_think", "schema_nothink"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if arm in ("schema", "schema_nothink"):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "hcli_result", "strict": True, "schema": RESULT_SCHEMA},
        }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return {"arm": arm, "ok": False, "wall_s": round(time.time() - t0, 2),
                "http_error": e.code, "detail": e.read().decode("utf-8", "replace")[:400]}
    except Exception as e:
        return {"arm": arm, "ok": False, "wall_s": round(time.time() - t0, 2),
                "error": f"{type(e).__name__}: {e}"}
    wall = time.time() - t0
    msg = body.get("choices", [{}])[0].get("message", {})
    content = msg.get("content") or ""
    finish = body.get("choices", [{}])[0].get("finish_reason")
    usage = body.get("usage", {})
    ct = usage.get("completion_tokens")
    parsed = extract_json(content)
    return {
        "arm": arm, "ok": parsed is not None, "wall_s": round(wall, 2),
        "finish_reason": finish, "completion_tokens": ct,
        "prompt_tokens": usage.get("prompt_tokens"),
        "tok_per_s": round(ct / wall, 2) if ct and wall > 0 else None,
        "had_think_block": "<think>" in content,
        "reasoning_content_len": len(msg.get("reasoning_content") or ""),
        "content_len": len(content),
        "parsed_kind": (parsed or {}).get("kind"),
        "content_head": content[:180],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=51771)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=6500)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--arms", default="baseline,no_think,schema,schema_nothink")
    ap.add_argument("--keep-server", action="store_true")
    ap.add_argument("--user-file", default=None,
                    help="path to a real mission prompt to use as the user turn "
                         "(reproduces the production failure, not a toy)")
    args = ap.parse_args()

    if args.user_file:
        user = "GOAL:\n" + open(os.path.expanduser(args.user_file)).read() + \
               "\n\nDETERMINISTIC EVIDENCE:\n(none)"
    else:
      user = (
        "GOAL:\ncalc.py add() returns a - b but test_calc.py expects a + b. "
        "Fix calc.py so the test passes.\n\n"
        "DETERMINISTIC EVIDENCE:\n"
        "===== calc.py =====\ndef add(a, b):\n    return a - b\n\n"
        "===== test_calc.py =====\nfrom calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
      )

    proc = subprocess.Popen(
        ["llama-server", "-m", MODEL, "--port", str(args.port), "-c", str(args.ctx),
         "-ngl", "999", "--host", "127.0.0.1", "--jinja"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"llama-server pid={proc.pid} port={args.port}; waiting for /health ...", flush=True)
    try:
        if not wait_ready(args.port):
            print("FAIL: server never became ready", file=sys.stderr)
            return 2
        print("ready. warming (one short call, discarded) ...", flush=True)
        call(args.port, "baseline", "GOAL:\nsay ok\n\nDETERMINISTIC EVIDENCE:\n(none)", 32, 300)

        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        results = []
        for rep in range(args.reps):
            for arm in arms:                      # alternating, not blocked, so drift cancels
                r = call(args.port, arm, user, args.max_tokens, args.timeout)
                r["rep"] = rep
                results.append(r)
                print(f"  rep{rep} {arm:<15} ok={r['ok']!s:<5} "
                      f"wall={r['wall_s']:>7}s tok={r.get('completion_tokens')} "
                      f"finish={r.get('finish_reason')} tps={r.get('tok_per_s')}", flush=True)
    finally:
        if not args.keep_server:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()

    summary = {}
    for arm in {r["arm"] for r in results}:
        rs = [r for r in results if r["arm"] == arm]
        walls = sorted(r["wall_s"] for r in rs)
        summary[arm] = {
            "n": len(rs),
            "valid_json_rate": round(sum(1 for r in rs if r["ok"]) / len(rs), 3),
            "wall_s_min": walls[0], "wall_s_median": walls[len(walls) // 2], "wall_s_max": walls[-1],
            "completion_tokens_median": sorted(
                r.get("completion_tokens") or 0 for r in rs)[len(rs) // 2],
            "finish_reasons": sorted({r.get("finish_reason") for r in rs}, key=str),
            "any_think_block": any(r.get("had_think_block") for r in rs),
        }

    out = Path(os.path.expanduser("~/Downloads/hawking-copy/receipts/headless"))
    out.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "hawking.headless.structured_output_probe.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hypothesis": "HCLI's 'Model did not return a valid structured JSON object' is "
                      "max_tokens exhaustion by Qwen3 reasoning under unconstrained decoding, "
                      "not a model capability limit.",
        "model": MODEL,
        "llama_server_version": subprocess.run(
            ["llama-server", "--version"], capture_output=True, text=True).stderr.strip()[:200],
        "user_source": args.user_file or "builtin calc.py toy",
        "params": {"ctx": args.ctx, "max_tokens": args.max_tokens,
                   "temperature": 0.15, "reps": args.reps, "design": "alternating paired"},
        "summary": summary,
        "runs": results,
    }
    (out / "STRUCTURED_OUTPUT_PROBE.json").write_text(json.dumps(doc, indent=1))
    print("\n=== SUMMARY ===")
    for arm in ["baseline", "no_think", "schema", "schema_nothink"]:
        if arm in summary:
            s = summary[arm]
            print(f"  {arm:<15} valid={s['valid_json_rate']:<5} "
                  f"wall_med={s['wall_s_median']:>7}s tok_med={s['completion_tokens_median']:>5} "
                  f"finish={s['finish_reasons']}")
    print(f"\n-> {out/'STRUCTURED_OUTPUT_PROBE.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
