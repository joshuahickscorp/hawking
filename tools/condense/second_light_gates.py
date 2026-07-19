#!/usr/bin/env python3.12
"""Run the Second Light staged gates B-E on the REAL GPT-OSS-120B source (goal Section 18).

Stage A (operator/parity) is produced by the PQ CPU/Metal parity harness. This runs the remaining
progressive gates as APPARATUS: each loads real experts from the source, packs them with the
selected PQ family, and measures functional output divergence through the reference MoE forward
(routed experts actually exercised). Every gate is bounded and labelled; a bounded sample is NEVER
called the full run. Divergence numbers are honest functional PROXIES (reference forward not HF-
parity validated), never a capability claim.

  Stage B  expert gate        - representative experts across early/mid/late layers
  Stage C  full-layer gate    - one complete layer, bounded expert sample (labelled)
  Stage D  multi-layer gate    - early, middle, late layers
  Stage E  short end-to-end    - short synthetic-activation logits/token proxy through the block
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REPO = Path(_HERE).resolve().parents[1]
EV = REPO / "reports" / "condense" / "second_light" / "evidence"

import gptoss_moe_runtime as rt          # noqa: E402
import gravity_forge as gf               # noqa: E402


def _pack_fn(family: str):
    """Return a pack_fn(w)->recon for a per-expert tensor at the program's declared config."""
    if family == "transform_pq":
        return lambda w: gf.pack_transform_pq(w, dim=16, subspaces=2, k=64, seed=0).recon
    # shared grammar is per-cluster; for a single-tensor divergence probe use transform_pq geometry
    return lambda w: gf.pack_transform_pq(w, dim=16, subspaces=2, k=64, seed=0).recon


def _gate_on_layers(reader, layers, *, n_inputs=6, top_k=4, label="") -> dict:
    fam = "transform_pq"
    pf = _pack_fn(fam)
    per_layer = []
    for b in layers:
        try:
            d = gf.output_divergence(reader, b, pf, n_inputs=n_inputs, top_k=top_k, seed=b)
            d["block"] = b
            per_layer.append(d)
        except Exception as e:  # noqa: BLE001
            per_layer.append({"block": b, "error": str(e)})
    divs = [d["mean_output_rel_div"] for d in per_layer if "mean_output_rel_div" in d]
    return {
        "ran": len(divs) > 0,
        "label": label,
        "family": fam,
        "target_bpw": "~0.75",
        "layers": list(layers),
        "per_layer": per_layer,
        "mean_output_rel_div": round(float(np.mean(divs)), 5) if divs else None,
        "worst_output_rel_div": round(float(np.max(divs)), 5) if divs else None,
        "activation_source": "synthetic_gaussian_routed_experts_exercised",
        "capability_parity": False,
        "note": "apparatus gate: real experts packed + reference forward run; functional PROXY, "
                "not a capability pass",
    }


def run() -> dict:
    reader = rt.ProvenanceReader()
    sample = reader.by_name.get("block.0.mlp.gate.weight")
    if sample is None or not Path(sample["shard_path"]).exists():
        return {"ok": False, "reason": "source shards absent"}

    t0 = time.time()
    expert_gate = _gate_on_layers(reader, [0, 18, 35], n_inputs=6, label="Stage B expert gate")
    full_layer_gate = _gate_on_layers(reader, [0], n_inputs=12, label="Stage C full-layer gate "
                                      "(bounded: routed experts of layer 0)")
    multi_layer_gate = _gate_on_layers(reader, [0, 12, 24, 35], n_inputs=6,
                                       label="Stage D multi-layer gate")

    # Stage E: short end-to-end proxy - run a 4-token sequence through block 0's MoE with original
    # vs packed experts, compare the summed block outputs (apparatus token-level proxy).
    router = rt.load_router(reader, 0)
    rng = np.random.default_rng(7)
    seq = [rng.standard_normal(2880).astype(np.float32) * 0.02 for _ in range(4)]
    routed = set()
    for x in seq:
        logits = router["weight"] @ x + router["bias"]
        routed.update(int(e) for e in np.argsort(-logits)[:4])
    pf = _pack_fn("transform_pq")
    orig, packed = {}, {}
    for e in routed:
        ex = rt.load_expert(reader, 0, e)
        orig[e] = ex
        m = dict(ex); m["mlp1"] = pf(ex["mlp1"].astype(np.float32)); m["mlp2"] = pf(ex["mlp2"].astype(np.float32))
        packed[e] = m
    devs = []
    for x in seq:
        y0 = rt.moe_forward_reference(x, router, lambda e: orig[e], top_k=4)
        y1 = rt.moe_forward_reference(x, router, lambda e: packed[e], top_k=4)
        devs.append(float(np.linalg.norm(y0 - y1) / (np.linalg.norm(y0) + 1e-9)))
    short_e2e = {
        "ran": True, "label": "Stage E short end-to-end proxy", "seq_len": len(seq),
        "n_experts_exercised": len(routed),
        "mean_token_output_div": round(float(np.mean(devs)), 5),
        "max_token_output_div": round(float(np.max(devs)), 5),
        "activation_source": "synthetic_gaussian", "capability_parity": False,
        "note": "token-level apparatus proxy; not logit/perplexity capability",
    }

    doc = {
        "schema": "hawking.second_light.staged_gates.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent": "openai/gpt-oss-120b",
        "expert_gate": expert_gate,
        "full_layer_gate": full_layer_gate,
        "multi_layer_gate": multi_layer_gate,
        "short_e2e_gate": short_e2e,
        "seconds": round(time.time() - t0, 1),
        "honesty": {
            "all_apparatus_not_capability": True,
            "subbit_divergence_large": True,
            "note": "these gates prove the apparatus runs on real weights and produce honest "
                    "functional proxies; sub-bit divergence remains large; no capability pass claimed",
        },
    }
    payload = json.dumps(doc, sort_keys=True).encode()
    doc["gates_sha256"] = hashlib.sha256(payload).hexdigest()
    return doc


def main() -> int:
    EV.mkdir(parents=True, exist_ok=True)
    doc = run()
    (EV / "STAGED_GATES.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
    if not doc.get("ok", True):
        print(json.dumps(doc)); return 1
    print(json.dumps({
        "expert_gate_div": doc["expert_gate"]["mean_output_rel_div"],
        "full_layer_gate_div": doc["full_layer_gate"]["mean_output_rel_div"],
        "multi_layer_gate_div": doc["multi_layer_gate"]["mean_output_rel_div"],
        "short_e2e_div": doc["short_e2e_gate"]["mean_token_output_div"],
        "seconds": doc["seconds"], "gates_sha256": doc["gates_sha256"][:16]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
