#!/usr/bin/env python3.12
"""Gate-F G1: larger expert reproduction (General Frontier goal, Part 0.8 G1).

Reproduces the bounded geometry winners on a LARGER real sample: the Gate-F candidate set evaluated
on layer 0 with more real routed experts, a calibration/validation split, and a CPU/Metal parity
spot-check. This is the bridge from the 32 bounded trials toward a parent-quality decision. It stays
honest: synthetic-activation functional-divergence PROXY (capability_parity False), directional only,
NOT a capability pass. Bounded + durable; seals G1_RESULT.json.
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
GF = REPO / "reports" / "condense" / "general_frontier" / "GENERAL_FRONTIER_RESULTS"

import gptoss_moe_runtime as rt          # noqa: E402
import gravity_forge as gf               # noqa: E402

# Gate-F candidate set (Part 0.7). Each is a per-expert-matrix pack_fn(w)->recon.
def _cand_pack_fns():
    tp = lambda w: gf.pack_transform_pq(w, dim=16, subspaces=2, k=64, seed=0).recon
    pq = lambda w: gf.pack_product_quant(w, dim=16, subspaces=2, k=64, seed=0).recon
    rvq = lambda w: gf.pack_naive_rvq(w, dim=16, k=64, stages=2, seed=0).recon
    isl = lambda w: gf.pack_pq_protected_islands(w, dim=16, subspaces=2, k=64,
                                                 strategy="residual_energy", budget_frac=0.01, seed=0).recon
    ident = lambda w: w.astype(np.float32)   # source-native control (equal-byte reference boundary)
    return {
        "mlp1": {"A1_pq_doctor~pq": tp, "A2_product_quant": pq, "A3_pq_islands": isl, "A4_source_native": ident},
        "mlp2": {"B1_pq_islands": isl, "B2_naive_rvq": rvq, "B3_product_quant": pq, "B4_source_native": ident},
    }


def _divergence_split(reader, block, pack_fn, which, n_cal, n_val, seed):
    """Functional output divergence over calibration + validation input sets (Part 19 separation).
    Only the `which` projection is replaced; routed experts actually exercised."""
    def make(nin, s):
        # isolate the tensor class: replace only mlp1 OR only mlp2
        def pf(w):
            return pack_fn(w.astype(np.float32))
        return gf.output_divergence(reader, block,
                                    (pf if which == "both" else _iso(pf, which)),
                                    n_inputs=nin, top_k=4, seed=s)
    cal = make(n_cal, seed)
    val = make(n_val, seed + 999)
    return cal, val


def _iso(pf, which):
    # wrap so output_divergence replaces only the chosen matrix; it packs both mlp1+mlp2 by default,
    # so for class isolation we return identity for the other matrix by tagging via closure.
    return pf  # output_divergence packs both; class isolation handled by the caller's pack_fn choice


def run(block: int = 0, n_cal: int = 16, n_val: int = 12) -> dict:
    reader = rt.ProvenanceReader()
    sample = reader.by_name.get("block.0.mlp.gate.weight")
    if sample is None or not Path(sample["shard_path"]).exists():
        return {"ok": False, "reason": "source absent"}
    cands = _cand_pack_fns()
    t0 = time.time()
    out = {"mlp1": {}, "mlp2": {}}
    # For class isolation we measure by packing ONLY that class. output_divergence packs both mlp1
    # and mlp2 via the same pack_fn; to isolate, we pack the target class and leave the other native.
    for which, group in (("mlp1", cands["mlp1"]), ("mlp2", cands["mlp2"])):
        for name, fn in group.items():
            def pack_only(w, _fn=fn, _rows=(5760 if which == "mlp1" else 2880)):
                # identity for the non-target matrix (by row count), target packer otherwise
                return _fn(w) if w.shape[0] == _rows else w.astype(np.float32)
            cal = gf.output_divergence(reader, block, pack_only, n_inputs=n_cal, top_k=4, seed=block)
            val = gf.output_divergence(reader, block, pack_only, n_inputs=n_val, top_k=4, seed=block + 999)
            out[which][name] = {
                "calibration_div": cal["mean_output_rel_div"],
                "validation_div": val["mean_output_rel_div"],
                "experts_exercised_cal": cal["n_experts_exercised"],
                "capability_parity": False,
            }
    # winners per class (lowest validation divergence)
    winners = {}
    for which in ("mlp1", "mlp2"):
        best = min(out[which].items(), key=lambda kv: kv[1]["validation_div"])
        winners[which] = {"candidate": best[0], "validation_div": best[1]["validation_div"],
                          "calibration_div": best[1]["calibration_div"]}
    # CPU/Metal parity spot-check: one candidate, weight-space rel-error CPU vs MPS via forge parity
    ex = rt.load_expert(reader, block, 0)
    parity = gf.pq_cpu_metal_parity(ex["mlp2"].astype(np.float32), dim=16, k=64, seed=0)
    doc = {
        "schema": "hawking.general_frontier.g1_result.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate": "G1_larger_expert_reproduction", "block": block,
        "n_calibration_inputs": n_cal, "n_validation_inputs": n_val,
        "candidates": out, "winners": winners,
        "cpu_metal_parity": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                             for k, v in (parity.items() if isinstance(parity, dict) else {})},
        "seconds": round(time.time() - t0, 1),
        "honesty": "synthetic-activation functional-divergence PROXY, calibration/validation split, "
                   "capability_parity False. Directional larger reproduction, NOT a capability pass. "
                   "True-residual + real Harmony activations + holdout are the next fidelity (G3/G4).",
    }
    doc["sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    return doc


def main() -> int:
    GF.mkdir(parents=True, exist_ok=True)
    doc = run()
    (GF / "GATE_F_G1_RESULT.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
    if not doc.get("ok", True):
        print(json.dumps(doc)); return 1
    print(json.dumps({"winners": doc["winners"], "seconds": doc["seconds"],
                      "parity_ok": doc["cpu_metal_parity"].get("within_tol", "n/a")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
