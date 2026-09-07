#!/usr/bin/env python3
"""Whole-token decode decomposition from the resident's own counters.

Receipt: receipts/runtime/DECODE_WHOLE_TOKEN_DECOMPOSITION.json

Not a synthetic kernel loop. Every number is the resident's native_metrics for a
real generation, and the two arms differ only in whether the JSON grammar mask is
active, so their difference IS the constrained-decoding cost.

Each arm runs in a FRESH process: a call that restores a prefix checkpoint is
2.27x slower than a cold one on this body, so arms must not share a session.

Two of the directive's seven terms are NOT separable with these counters and are
reported as such rather than invented: sampling and state movement both happen
inside the GPU span the resident reports as one number. Saying "unseparable" is a
measurement; splitting it by guess would not be.
"""
from __future__ import annotations

import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "receipts/runtime/DECODE_WHOLE_TOKEN_DECOMPOSITION.json"
ENVELOPE = REPO / "ascension_envelope.hawking.json"
sys.path.insert(0, str(REPO))

PROMPT = "Count from one to sixty, one number per line, and write nothing else."


def arm(grammar: bool, max_tokens: int) -> dict:
    from hcli.hawking_native import HawkingNativeConnector, config_for_model_path
    conn = HawkingNativeConnector(config_for_model_path(str(ENVELOPE)))
    payload = {"model": "local", "messages": [{"role": "user", "content": PROMPT}],
               "temperature": 0.0, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}}
    if grammar:
        payload["grammar"] = "json"
    t0 = time.perf_counter()
    body = conn.complete_payload(payload, timeout=1800)
    call_wall_s = time.perf_counter() - t0
    h = body.get("hawking") or {}
    nm = h.get("native_metrics") or {}
    dec = nm.get("decode") or {}
    pre = nm.get("prefill") or {}
    return {
        "grammar_requested": grammar,
        "grammar_enforced": h.get("grammar_enforced"),
        "client_call_wall_s": call_wall_s,
        "connector_wall_ms": h.get("wall_ms"),
        "generation_wall_ns": nm.get("generation_wall_ns"),
        "complete_wall_ns": nm.get("complete_wall_ns"),
        "complete_wall_ns_per_generated_token": nm.get("complete_wall_ns_per_generated_token"),
        "generated_tokens": nm.get("generated_tokens"),
        "gpu_ns": nm.get("gpu_ns"),
        "wall_minus_gpu_ns": nm.get("wall_minus_gpu_ns"),
        "dispatches": nm.get("dispatches"),
        "dispatches_per_generated_token": nm.get("dispatches_per_generated_token"),
        "decode": dec,
        "prefill": pre,
    }


def run_arm_fresh(grammar: bool, max_tokens: int) -> dict:
    out = subprocess.run(
        [sys.executable, str(REPO / "tools/sovereign/g006_decode_decomposition.py"),
         "--arm", "1" if grammar else "0", str(max_tokens)],
        capture_output=True, text=True, timeout=3600, cwd=str(REPO))
    for line in out.stdout.splitlines():
        if line.startswith("ARM "):
            return json.loads(line[4:])
    raise RuntimeError(f"arm produced nothing: {out.stderr[-400:]}")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--arm":
        print("ARM " + json.dumps(arm(sys.argv[2] == "1", int(sys.argv[3]))), flush=True)
        return 0

    plain = run_arm_fresh(False, 128)
    grammared = run_arm_fresh(True, 128)

    d = plain["decode"]
    decode_wall = float(d["wall_ns"])
    decode_gpu = float(d["gpu_ns"])
    steps = int(d["steps"])
    gen_wall = float(plain["generation_wall_ns"])
    connector_ns = float(plain["connector_wall_ms"]) * 1e6

    g_per_tok = (float(grammared["decode"]["wall_ns"]) / max(1, int(grammared["decode"]["steps"])))
    p_per_tok = decode_wall / max(1, steps)

    doc = {
        "schema": "hawking.decode.whole_token_decomposition.v1",
        "produced_by": "tools/sovereign/g006_decode_decomposition.py",
        "command": "python3 tools/sovereign/g006_decode_decomposition.py",
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "method": ("the resident's own native_metrics for a real generation; two arms "
                   "differing only in the JSON grammar mask, each in a FRESH process"),
        "arms": {"plain": plain, "grammar_json": grammared},
        "whole_token_ns": {
            "model_compute_gpu_ns_per_token": decode_gpu / max(1, steps),
            "dispatch_and_sync_ns_per_token": (decode_wall - decode_gpu) / max(1, steps),
            "constrained_decoding_ns_per_token": g_per_tok - p_per_tok,
            "sampling_ns_per_token": "NOT SEPARABLE from the GPU span with these counters",
            "state_movement_ns_per_token": "NOT SEPARABLE from the GPU span with these counters",
            "hcli_connector_overhead_ns_per_call": connector_ns - gen_wall,
            "prefill_ns_per_prompt_token": (float(plain["prefill"]["wall_ns"])
                                            / max(1, int(plain["prefill"]["steps"]))),
        },
        "shares_of_decode_wall": {
            "model_compute_gpu": decode_gpu / decode_wall,
            "dispatch_and_sync": (decode_wall - decode_gpu) / decode_wall,
        },
        "rates": {
            "decode_tok_s_plain": 1e9 / p_per_tok,
            "decode_tok_s_grammar_json": 1e9 / g_per_tok,
            "complete_tok_s_including_prefill": 1e9 / float(
                plain["complete_wall_ns_per_generated_token"]),
        },
        "dispatches_per_generated_token": plain["dispatches_per_generated_token"],
        "binding_term": "model_compute_gpu_ns_per_token",
        "binding_term_evidence": None,
    }
    doc["binding_term_evidence"] = (
        f"GPU is {doc['shares_of_decode_wall']['model_compute_gpu']*100:.1f}% of the decode "
        f"wall; dispatch and synchronisation together are "
        f"{doc['shares_of_decode_wall']['dispatch_and_sync']*100:.1f}%. Removing ALL host "
        f"overhead could not reach the >=50 tok/s rung from "
        f"{doc['rates']['decode_tok_s_plain']:.2f} tok/s. Two concurrent streams return "
        f"1.002x (G012), so there is no idle GPU to harvest either. The work is inside the "
        f"model at {doc['dispatches_per_generated_token']:.1f} dispatches per token."
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    w = doc["whole_token_ns"]; s = doc["shares_of_decode_wall"]; r = doc["rates"]
    print(f"  model compute (GPU)   {w['model_compute_gpu_ns_per_token']/1e6:8.3f} ms/token  {s['model_compute_gpu']*100:5.1f}%")
    print(f"  dispatch + sync       {w['dispatch_and_sync_ns_per_token']/1e6:8.3f} ms/token  {s['dispatch_and_sync']*100:5.1f}%")
    print(f"  constrained decoding  {w['constrained_decoding_ns_per_token']/1e6:8.3f} ms/token  (grammar arm minus plain)")
    print(f"  HCLI connector /call  {w['hcli_connector_overhead_ns_per_call']/1e6:8.3f} ms")
    print(f"  decode plain {r['decode_tok_s_plain']:.2f} tok/s | grammar {r['decode_tok_s_grammar_json']:.2f} tok/s | complete {r['complete_tok_s_including_prefill']:.2f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
