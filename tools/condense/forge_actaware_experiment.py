#!/usr/bin/env python3.12
"""Activation-aware vs weight-aware sub-bit packing on the TRUE residual stream (Section 6.2).

The weight-space families collapse and the faithful F2 output divergence is ~0.69 at ~0.75 BPW. The
directive's thesis is that weight error is the wrong objective. This experiment tests the first real
counter-lever: fit the SAME representation (transform_pq) activation-aware - scaling input channels by
their real salience so precision follows the activations - and measure whether the block output
divergence DROPS at matched whole-artifact bytes.

Method (real, bounded): build block-0 residual activations X with the tokenizer + attention forward;
route; for each routed expert pack mlp1/mlp2 both weight-aware and activation-aware (mlp1 salience
from X, mlp2 salience from the real SwiGLU hidden h = swiglu(X @ mlp1)); run the reference MoE with
each and compare output divergence to the original. No capability claim; no escape; bounded experts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gravity_forge as gf          # noqa: E402
import gptoss_moe_runtime as rt     # noqa: E402
import gptoss_block as gb           # noqa: E402

SCHEMA = "hawking.gravity_forge.actaware_experiment.v1"
SRC = "scratch/staging/gpt-oss-120b.partial"
PROMPTS = [
    "The capital of France is Paris and the Seine runs through it.",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "Solve 3x + 7 = 22: 3x = 15, x = 5.",
    "Call the weather API for Tokyo in metric units and summarize.",
]
CFG = dict(dim=16, subspaces=2, k=64)   # ~0.75 bpw body


def _swiglu(h):
    return rt._swiglu(h)


def run(*, seq_per_prompt: int = 12, max_tokens: int = 40, top_k: int = 4) -> dict[str, Any]:
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(f"{SRC}/tokenizer.json")
    reader = rt.ProvenanceReader()
    if not Path(reader.by_name["block.0.mlp.gate.weight"]["shard_path"]).exists():
        return {"schema": SCHEMA, "green": False, "reason": "source absent"}

    emb = reader.bf16("embedding.weight")
    acts_list = []
    for p in PROMPTS:
        ids = [int(i) for i in tk.encode(p).ids[:seq_per_prompt]]
        acts_list.append(gb.block0_moe_inputs(reader, ids, embeddings=emb))
        if sum(a.shape[0] for a in acts_list) >= max_tokens:
            break
    del emb
    X = np.concatenate(acts_list, 0)[:max_tokens].astype(np.float32)     # [n, 2880] real MoE input

    router = rt.load_router(reader, 0)
    routed: set[int] = set()
    for x in X:
        logits = router["weight"] @ x + router["bias"]
        routed.update(int(e) for e in np.argsort(-logits)[:top_k])

    orig, wa, aa = {}, {}, {}
    bpw_wa, bpw_aa, nwt = 0, 0, 0
    sal_in = np.abs(X).mean(0)                                            # shared mlp1 input salience
    for e in routed:
        ex = rt.load_expert(reader, 0, e)
        m1, m2 = ex["mlp1"].astype(np.float32), ex["mlp2"].astype(np.float32)
        h = _swiglu(X @ m1.T + ex["mlp1_bias"])                          # [n, 2880] real mlp2 input
        # weight-aware
        w1 = gf.pack_transform_pq(m1, **CFG); w2 = gf.pack_transform_pq(m2, **CFG)
        # activation-aware (mlp1 salience from X; mlp2 salience from the real hidden h)
        a1 = gf.pack_transform_pq_actaware(m1, X, **CFG)
        a2 = gf.pack_transform_pq_actaware(m2, h, **CFG)
        orig[e] = ex
        wa[e] = {**ex, "mlp1": w1.recon, "mlp2": w2.recon}
        aa[e] = {**ex, "mlp1": a1.recon, "mlp2": a2.recon}
        bpw_wa += w1.physical_bytes + w2.physical_bytes
        bpw_aa += a1.physical_bytes + a2.physical_bytes
        nwt += m1.size + m2.size

    def diverge(packed):
        rel = []
        for x in X:
            y0 = rt.moe_forward_reference(x, router, lambda e: orig[e], top_k=top_k)
            y1 = rt.moe_forward_reference(x, router, lambda e: packed[e], top_k=top_k)
            rel.append(float(np.linalg.norm(y0 - y1) / (np.linalg.norm(y0) + 1e-9)))
        return float(np.mean(rel)), float(np.max(rel))

    wa_mean, wa_max = diverge(wa)
    aa_mean, aa_max = diverge(aa)
    improved = aa_mean < wa_mean
    doc = {
        "schema": SCHEMA, "green": True, "parent": "120B", "activation_source": "true_residual_stream_block0",
        "pack": f"transform_pq {CFG}", "n_tokens": int(X.shape[0]), "n_experts": len(routed),
        "whole_artifact_bpw_weight_aware": round(bpw_wa * 8 / nwt, 4),
        "whole_artifact_bpw_activation_aware": round(bpw_aa * 8 / nwt, 4),
        "output_divergence_weight_aware": {"mean": round(wa_mean, 5), "max": round(wa_max, 5)},
        "output_divergence_activation_aware": {"mean": round(aa_mean, 5), "max": round(aa_max, 5)},
        "activation_aware_reduces_divergence": bool(improved),
        "relative_improvement": round((wa_mean - aa_mean) / (wa_mean + 1e-9), 4),
        "capability_parity": False, "authorizes_escape": False,
        "note": "same representation + matched bytes; only the fit objective differs. Output-space, "
                "real residual activations. A reduction is real progress, NOT yet a capability pass.",
    }
    doc["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in doc.items() if k != "sha256"},
                                              sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Activation-aware vs weight-aware sub-bit packing.")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--out", default="reports/condense/gravity_forge/FORGE_ACTAWARE.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    doc = run(max_tokens=args.max_tokens)
    doc["seconds"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    if doc.get("green"):
        print(f"weight-aware   output divergence: mean {doc['output_divergence_weight_aware']['mean']}")
        print(f"activation-aware output divergence: mean {doc['output_divergence_activation_aware']['mean']}")
        print(f"activation-aware reduces divergence: {doc['activation_aware_reduces_divergence']} "
              f"(relative {doc['relative_improvement']*100:.1f}%) at matched "
              f"~{doc['whole_artifact_bpw_activation_aware']} BPW")
    else:
        print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
