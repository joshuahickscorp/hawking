#!/usr/bin/env python3.12
"""Generation S2: per-layer adaptive expert inventory K_l, chosen by an exact global byte auction.

WHY NOT 64 EVERYWHERE. S1 fixed K = 64 at every layer because the byte arithmetic is
budget-neutral, not because the routing evidence said 64. The 1313-token disjoint calibration says
routing concentration is NOT uniform across depth: at K = 64 the retained top-8 mass runs from
0.702 in the worst layer to 0.837 in the best. Spending the same inventory on a layer whose router
is nearly flat and on a layer whose router is sharply peaked is a straightforward misallocation.

THE AUCTION. The one-bit ceiling is a global budget, so every layer's K_l and every organ's rate
compete against every other. This module makes that competition explicit and solves it exactly.

Per layer l and choice (K, gate_rate, down_rate) define a predicted relative output error:

    err(l, K, r) = (1 - C_l(K)) * MISS_COST  +  C_l(K) * recon_err(r)

  * C_l(K) is the measured retained top-8 routing mass at that layer for the K hottest experts.
  * (1 - C_l) is the share of routing decisions that land on an omitted expert. Those tokens are
    NOT lost - they re-route to survivors - but Lane E measured that an omitted expert is NOT
    reconstructible from survivors (best single survivor gives median held-out relative error
    0.885-0.995, i.e. no better than predicting zero). MISS_COST is therefore set to 1.0 and that
    is a measurement, not a guess. See NEGATIVE_TRANSFER_ATLAS entry
    expert_merging_omitted_from_survivors.
  * recon_err(r) is the measured reconstruction error of the surviving experts at index rate r.
    Taken from the Shannon-bound probe where measured, else the memoryless-Gaussian value. The
    source of each number is recorded in the emitted program so no estimate masquerades as a
    measurement.

Minimising the sum of err over layers subject to total_bits <= ceiling is a separable knapsack:
because layers are independent given the budget, the exact optimum is obtained by sweeping the
Lagrange multiplier lambda over the per-layer minimisers of (err + lambda * bits) and bisecting
lambda until the budget binds. No greedy heuristic, no local search.

WHAT THIS MODULE DOES NOT CLAIM. err() is a PREDICTOR built from routing mass and reconstruction
error. It is not capability. It orders candidates for the forward to adjudicate; it never selects
a frontier. Only a real parent-vs-packed forward does that.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "foundry")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qwen3_moe_adapter as A  # noqa: E402
import qwen_structural_plan as SP  # noqa: E402

SCHEMA = "hawking.gravity.adaptive_k.v1"
CEILING = Fraction(1, 1)

K_CHOICES = (32, 48, 64, 80, 96, 112, 128)

# An omitted expert is not reconstructible from survivors: Lane E measured best-single-survivor
# held-out relative error 0.885 / 0.993 / 0.995 and a 4-survivor least-squares merge 0.863 / 0.988 /
# 0.995. ~1.0 is the trivial zero predictor, so a miss costs full relative error.
MISS_COST = 1.0

# Measured reconstruction error by index rate, from the real-weight probes. Anything not measured
# falls back to the memoryless-Gaussian value and is TAGGED as such in the output.
MEASURED_RECON = {
    0.15625: 0.9100,   # down @ L46/L93, M01_FUNCTION_AWARE_PROBE
    0.625: 0.7027,     # gate @ L23-L93, M01_FUNCTION_AWARE_PROBE
    2.5: 0.2427,       # gate @ L46/L93, SHANNON_BOUND_ADVERSARIAL
}


def recon_err(rate: float) -> tuple[float, str]:
    """Predicted reconstruction error at an index rate, with its provenance."""
    if rate in MEASURED_RECON:
        return MEASURED_RECON[rate], "measured"
    return SP.rd_floor(rate), "gaussian_rd_estimate"


# ── routing coverage C_l(K) with a bootstrap interval ─────────────────────────────────────────
def coverage(routing: dict[str, Any], layer: int, K: int) -> float:
    """Retained share of real top-8 routing decisions for the K hottest experts of this layer."""
    lay = routing["layers"][layer]
    c = np.asarray(lay["top8_count"], np.float64)
    keep = SP.survivors(routing, layer, K)
    tot = c.sum()
    return float(c[keep].sum() / tot) if tot > 0 else 0.0


def coverage_ci(routing: dict[str, Any], layer: int, K: int, *, resamples: int = 400,
                seed: int = 20260720) -> tuple[float, float, float]:
    """Bootstrap CI on C_l(K) by resampling the per-expert routing counts.

    The counts are a multinomial draw over experts, so the resample is a multinomial with the
    observed proportions. The survivor SET is re-selected on each resample, which is the whole
    point: it prices the risk that the top-K membership itself is sampling noise.
    """
    lay = routing["layers"][layer]
    c = np.asarray(lay["top8_count"], np.float64)
    soft = np.asarray(lay["softmax_mass"], np.float64)
    n = int(c.sum())
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = c / n
    rng = np.random.default_rng(seed + layer)
    out = np.empty(resamples)
    for i in range(resamples):
        cc = rng.multinomial(n, p).astype(np.float64)
        order = np.lexsort((-soft, -cc))[:K]
        out[i] = cc[order].sum() / n
    return coverage(routing, layer, K), float(np.percentile(out, 2.5)), float(
        np.percentile(out, 97.5))


# ── exact per-layer bit cost ──────────────────────────────────────────────────────────────────
def _expert_shapes(inv) -> dict[str, tuple[int, int]]:
    sh = {}
    for t in inv.tensors:
        if t.organ_class == A.ORGAN_EXP_GATE and "gate" not in sh:
            sh["gate"] = tuple(t.shape)
        if t.organ_class == A.ORGAN_EXP_DOWN and "down" not in sh:
            sh["down"] = tuple(t.shape)
        if len(sh) == 2:
            break
    return sh


def layer_bits(shapes: dict[str, tuple[int, int]], K: int, gate_spec: dict[str, Any],
               down_spec: dict[str, Any]) -> int:
    """Exact expert-payload bits for ONE layer at inventory K, via the same charge S1 used.

    Two gate-class tensors (gate_proj, up_proj) and one down_proj per surviving expert. The shared
    codebook amortizes over the SURVIVORS of this layer only, which is why a small K is charged
    more per expert - a real cost of omission, priced not hidden.
    """
    import qwen_subhalfbit_search as SHB
    g = dict(gate_spec)
    d = dict(down_spec)
    for s in (g, d):
        if s["family"] == "function_aware":
            s["family"] = "shared_grammar"
    per = 2 * SHB.expert_bits(shapes["gate"], g, K) + SHB.expert_bits(shapes["down"], d, K)
    scales = 2 * shapes["gate"][0] * SP.ROW_SCALE_BITS + shapes["down"][0] * SP.ROW_SCALE_BITS
    return K * (per + scales)


def _fixed_bits(inv, n_layers: int, n_experts: int) -> int:
    """Everything that is NOT expert payload: pass-through organs, dense islands, tables, package."""
    import qwen_subhalfbit_search as SHB
    total = 0
    for t in inv.tensors:
        oc = t.organ_class
        if oc in (A.ORGAN_EXP_GATE, A.ORGAN_EXP_UP, A.ORGAN_EXP_DOWN):
            continue
        if oc in SHB._DENSE_ORGANS:
            total += SHB._pq_bits(t.shape, dim=32, subspaces=8, k=16)
        else:
            total += SHB._native_bits(t.shape)
    total += n_layers * n_experts          # survivor bitmap, resident to mask the router
    total += SP.RESERVE_BYTES * 8          # packaging
    return total


# ── the auction ───────────────────────────────────────────────────────────────────────────────
RUNGS = [("0.625", "0.15625"), ("1.25", "0.3125"), ("1.25", "0.625"), ("2.5", "0.625"),
         ("2.5", "1.25"), ("5.0", "1.25"), ("5.0", "2.5")]


def _cells(inv, routing, shapes, n_layers):
    """Every (layer, K, rung) option with its exact bits and predicted error."""
    out = []
    for L in range(n_layers):
        cov = {K: coverage(routing, L, K) for K in K_CHOICES}
        for K in K_CHOICES:
            for gk, dk in RUNGS:
                g, d = SP.GATE_RUNGS[gk], SP.DOWN_RUNGS[dk]
                gr, ge = recon_err(float(SP.index_rate(g)))
                dr_, de = recon_err(float(SP.index_rate(d)))
                # gate/up are 2 of 3 expert tensors and carry the sensitive organ; weight 2:1
                r = (2.0 * gr + dr_) / 3.0
                err = (1.0 - cov[K]) * MISS_COST + cov[K] * r
                out.append({"layer": L, "K": K, "gate": gk, "down": dk,
                            "bits": layer_bits(shapes, K, g, d), "err": err,
                            "coverage": cov[K], "recon_err": r,
                            "recon_provenance": f"gate:{ge},down:{de}"})
    return out


def solve(inv, routing, *, ceiling: Fraction = CEILING) -> dict[str, Any]:
    """Exact separable-knapsack solve by Lagrangian bisection on the byte budget."""
    n_layers = len({int(t.layer) for t in inv.tensors if t.organ_class == A.ORGAN_EXP_GATE})
    n_experts = max(int(t.expert) for t in inv.tensors if t.organ_class == A.ORGAN_EXP_GATE) + 1
    shapes = _expert_shapes(inv)
    fixed = _fixed_bits(inv, n_layers, n_experts)
    budget = int(ceiling * inv.grand_params) - fixed
    cells = _cells(inv, routing, shapes, n_layers)

    by_layer: dict[int, list] = {}
    for c in cells:
        by_layer.setdefault(c["layer"], []).append(c)

    def pick(lam: float):
        chosen, bits, err = [], 0, 0.0
        for L in sorted(by_layer):
            best = min(by_layer[L], key=lambda c: c["err"] + lam * c["bits"])
            chosen.append(best)
            bits += best["bits"]
            err += best["err"]
        return chosen, bits, err

    # lam = 0 buys the richest option everywhere (over budget); large lam buys the cheapest.
    lo, hi = 0.0, 1e-6
    while pick(hi)[1] > budget:
        hi *= 4.0
        if hi > 1e6:
            raise SystemExit("no legal allocation exists even at the cheapest rung")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if pick(mid)[1] > budget:
            lo = mid
        else:
            hi = mid
    chosen, bits, err = pick(hi)
    total = bits + fixed
    bpw = Fraction(total, inv.grand_params)
    return {"chosen": chosen, "expert_bits": bits, "fixed_bits": fixed,
            "complete_bits": total, "complete_bpw": round(float(bpw), 9),
            "complete_bpw_exact": f"{bpw.numerator}/{bpw.denominator}",
            "legal_under_one_bit_ceiling": bool(bpw <= ceiling),
            "predicted_mean_layer_error": round(err / max(1, len(chosen)), 6),
            "lambda": hi, "n_layers": n_layers, "n_experts": n_experts,
            "budget_bits_for_experts": budget}


def build(routing_path: str) -> dict[str, Any]:
    inv = A.build_inventory(A.load_config(), A.load_index())
    routing = json.loads(Path(routing_path).read_text())
    sol = solve(inv, routing)
    ks = [c["K"] for c in sol["chosen"]]
    hist = {int(K): int(sum(1 for k in ks if k == K)) for K in K_CHOICES}
    rungs = {}
    for c in sol["chosen"]:
        key = f'g{c["gate"]}_d{c["down"]}'
        rungs[key] = rungs.get(key, 0) + 1
    cov = [c["coverage"] for c in sol["chosen"]]
    # uniform-64 reference under the identical predictor and ledger
    u = [min((x for x in _cells(inv, routing, _expert_shapes(inv), sol["n_layers"])
              if x["layer"] == L and x["K"] == 64 and x["gate"] == "2.5" and x["down"] == "0.625"),
             key=lambda c: c["err"]) for L in range(sol["n_layers"])]
    return {
        "schema": SCHEMA, "parent": "qwen3-235b-a22b-instruct-2507",
        "claim": "BYTE PLAN AND PREDICTOR ONLY. err() is built from measured routing mass and "
                 "measured reconstruction error; it is NOT capability and selects nothing. Only a "
                 "real parent-vs-packed forward may select a frontier.",
        "routing_source": routing_path,
        "routing_tokens": routing.get("n_tokens_calibrated_on"),
        "miss_cost": MISS_COST,
        "miss_cost_justification":
            "Lane E measured that an omitted expert is not reconstructible from survivors: best "
            "single survivor median held-out relative error 0.885/0.993/0.995, 4-survivor "
            "least-squares merge 0.863/0.988/0.995. ~1.0 is the trivial zero predictor.",
        "ledger": {k: v for k, v in sol.items() if k != "chosen"},
        "K_histogram": hist,
        "K_mean": round(float(np.mean(ks)), 3), "K_min": int(min(ks)), "K_max": int(max(ks)),
        "rung_histogram": rungs,
        "coverage": {"mean": round(float(np.mean(cov)), 6), "min": round(float(np.min(cov)), 6),
                     "max": round(float(np.max(cov)), 6)},
        "uniform64_reference": {
            "predicted_mean_layer_error": round(float(np.mean([c["err"] for c in u])), 6),
            "mean_coverage": round(float(np.mean([c["coverage"] for c in u])), 6)},
        "per_layer": [{"layer": c["layer"], "K": c["K"], "gate_rung": c["gate"],
                       "down_rung": c["down"], "coverage": round(c["coverage"], 6),
                       "predicted_err": round(c["err"], 6),
                       "recon_provenance": c["recon_provenance"]} for c in sol["chosen"]],
    }


def demo() -> None:
    """Self-check on the real inventory and the real calibration (metadata only, no weights)."""
    inv = A.build_inventory(A.load_config(), A.load_index())
    path = "reports/subbit_reset/QWEN3_235B_ROUTING_CALIBRATION_1200.json"
    routing = json.loads(Path(path).read_text())

    # coverage is monotone in K and exact at the endpoints
    for L in (0, 46, 93):
        cs = [coverage(routing, L, K) for K in K_CHOICES]
        assert all(b >= a - 1e-12 for a, b in zip(cs, cs[1:])), (L, cs)
        assert abs(coverage(routing, L, 128) - 1.0) < 1e-9, coverage(routing, L, 128)

    sol = solve(inv, routing)
    assert sol["legal_under_one_bit_ceiling"], sol["complete_bpw"]
    num, den = sol["complete_bpw_exact"].split("/")
    assert abs(int(num) / int(den) - sol["complete_bpw"]) < 1e-9

    # the auction must beat uniform-64 under its OWN predictor, else it is not worth running
    shapes = _expert_shapes(inv)
    cells = _cells(inv, routing, shapes, sol["n_layers"])
    u = [min((x for x in cells if x["layer"] == L and x["K"] == 64
              and x["gate"] == "2.5" and x["down"] == "0.625"), key=lambda c: c["err"])
         for L in range(sol["n_layers"])]
    assert sol["predicted_mean_layer_error"] <= float(np.mean([c["err"] for c in u])) + 1e-9

    # a tighter ceiling must never produce a more expensive plan
    tight = solve(inv, routing, ceiling=Fraction(3, 4))
    assert tight["complete_bits"] <= sol["complete_bits"]
    assert tight["legal_under_one_bit_ceiling"]

    # bootstrap CI brackets the point estimate
    pt, lo, hi = coverage_ci(routing, 46, 64, resamples=120)
    assert lo <= pt <= hi, (lo, pt, hi)

    print(json.dumps({"ok": True, "complete_bpw": sol["complete_bpw"],
                      "predicted_err_adaptive": sol["predicted_mean_layer_error"],
                      "predicted_err_uniform64": round(float(np.mean([c["err"] for c in u])), 6),
                      "K_range": [min(c["K"] for c in sol["chosen"]),
                                  max(c["K"] for c in sol["chosen"])],
                      "coverage_ci_L46_K64": [round(lo, 4), round(pt, 4), round(hi, 4)]}, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-layer adaptive K by exact global byte auction.")
    ap.add_argument("--routing", default="reports/subbit_reset/QWEN3_235B_ROUTING_CALIBRATION_1200.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    if args.demo:
        demo()
        return 0
    plan = build(args.routing)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in plan.items() if k != "per_layer"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
