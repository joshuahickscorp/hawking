#!/usr/bin/env python3.12
"""Structural expert reduction under a HARD complete <= 1.0 BPW ceiling (Lane D: M05 + M09).

THE MOVE. Every fixed-weight arm the F1 campaign ran spent its whole budget coding all 128 experts
of all 94 layers. The measured consequence (reports/subbit_reset/M01_FUNCTION_AWARE_PROBE.json) is
that gate/up sits at 0.625 index BPW where the memoryless-Gaussian rate-distortion floor is
rel_error 0.6484 - and the best function-aware codec measured 0.6435. There is no post-hoc headroom
left to find. The rate itself is the wall.

But the rate is not fixed by the ceiling; the INVENTORY is a free variable. The complete ceiling
constrains total bits over the ORIGINAL weight count:

    complete_bits / original_weight_count <= 1

so halving the expert inventory and doubling the rate on the survivors is BUDGET-NEUTRAL. Under one
identical ceiling:

    keep 128 experts -> gate/up ~1.25 bpw -> RD floor rel_error 0.42   (A1: collapsed 6/6)
    keep  64 experts -> gate/up ~2.50 bpw -> RD floor rel_error 0.177
    keep  32 experts -> gate/up ~5.00 bpw -> RD floor rel_error 0.031  (per-expert near-lossless)

This converts an unsolvable coding problem into a measurable capability question: how much function
survives when the router is restricted to the hottest N experts and the survivors are coded well?
The 88-token sealed calibration puts the top-64 at 92.2 percent of top-8 routing decisions (min
76.9 percent over layers) and the top-32 at 68.8 percent - a real, non-trivial loss that only a
parent-vs-packed forward may adjudicate. Nothing here claims capability.

WHAT OMISSION COSTS IN BYTES, declared: a survivor bitmap of n_experts bits per layer, and nothing
else - the omitted expert tensors are absent from the artifact, not zeroed. The router weight matrix
stays native so the mask can be applied to its logits at run time.

WHAT OMISSION COSTS IN FUNCTION: the router's top-k is taken over survivors only (omitted experts'
logits are masked to -inf BEFORE the top-k, then the k weights are renormalized as usual). That is a
genuine change to the model, which is exactly what the campaign law permits and requires declaring.
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
import qwen_subhalfbit_search as SHB  # noqa: E402

SCHEMA = "hawking.gravity.structural_plan.v1"
CEILING = Fraction(1, 1)

# Shared-grammar rungs, index rate = stages*ceil(log2 k)/dim bits per weight.
GATE_RUNGS = {
    "0.625": {"family": "shared_grammar", "dim": 16, "k": 1024, "stages": 1},
    "1.25":  {"family": "shared_grammar", "dim": 8,  "k": 1024, "stages": 1},
    "2.5":   {"family": "shared_grammar", "dim": 8,  "k": 1024, "stages": 2},
    "5.0":   {"family": "shared_grammar", "dim": 8,  "k": 1024, "stages": 4},
}
DOWN_RUNGS = {
    "0.15625": {"family": "shared_grammar", "dim": 64, "k": 1024, "stages": 1},
    "0.3125":  {"family": "shared_grammar", "dim": 32, "k": 1024, "stages": 1},
    "0.625":   {"family": "shared_grammar", "dim": 16, "k": 1024, "stages": 1},
    "1.25":    {"family": "shared_grammar", "dim": 8,  "k": 1024, "stages": 1},
    "2.5":     {"family": "shared_grammar", "dim": 8,  "k": 1024, "stages": 2},
}

ROW_SCALE_BITS = 16          # M01' per-row bf16 scale, billed on every coded expert tensor
RESERVE_BYTES = 64 * 1024 * 1024


def index_rate(spec: dict[str, Any]) -> Fraction:
    return Fraction(int(spec["stages"]) * math.ceil(math.log2(int(spec["k"]))), int(spec["dim"]))


def rd_floor(rate: float) -> float:
    """Memoryless-Gaussian rate-distortion floor on relative error at this index rate."""
    return math.sqrt(2.0 ** (-2.0 * rate))


# ── survivor selection ────────────────────────────────────────────────────────────────────────
def survivors(routing: dict[str, Any], layer: int, keep: int) -> list[int]:
    """The `keep` hottest experts of a layer: top-8 count first, softmax mass as the tie-break.

    softmax_mass is the lower-variance signal (every token contributes to every expert, no zeros),
    so it is the right tie-break when counts collide - which they do constantly on small samples.
    """
    lay = routing["layers"][layer]
    cnt = np.asarray(lay["top8_count"], dtype=np.int64)
    soft = np.asarray(lay["softmax_mass"], dtype=np.float64)
    order = np.lexsort((-soft, -cnt))
    return sorted(int(e) for e in order[:keep])


def routing_retained(routing: dict[str, Any], keep: int) -> dict[str, float]:
    """Fraction of real top-8 routing decisions and softmax mass the survivor set retains."""
    cnt_keep, cnt_all, mass_keep, mass_all, per_layer = 0.0, 0.0, 0.0, 0.0, []
    for L in range(len(routing["layers"])):
        lay = routing["layers"][L]
        c = np.asarray(lay["top8_count"], np.float64)
        s = np.asarray(lay["softmax_mass"], np.float64)
        keep_ids = survivors(routing, L, keep)
        ck, sk = c[keep_ids].sum(), s[keep_ids].sum()
        cnt_keep += ck; cnt_all += c.sum(); mass_keep += sk; mass_all += s.sum()
        per_layer.append(ck / max(c.sum(), 1e-12))
    return {"top8_count_retained": round(cnt_keep / max(cnt_all, 1e-12), 6),
            "softmax_mass_retained": round(mass_keep / max(mass_all, 1e-12), 6),
            "worst_layer_count_retained": round(float(min(per_layer)), 6),
            "n_layers": len(per_layer)}


# ── exact ledger with omission ────────────────────────────────────────────────────────────────
def ledger(inv, keep: int, gate_spec: dict[str, Any], down_spec: dict[str, Any],
           routing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Complete, itemized bit ledger. Omitted expert tensors contribute ZERO payload bits.

    Cluster amortization of the shared codebook is over the SURVIVING experts only - fewer experts
    to spread it across is a real cost of omission and is charged here, not hidden.
    """
    keep_ids: dict[int, set[int]] = {}
    if routing is not None:
        for L in range(len(routing["layers"])):
            keep_ids[L] = set(survivors(routing, L, keep))

    comp = {c: 0 for c in ("indices", "codebooks", "scales", "metadata", "alignment",
                           "protected_islands", "doctor", "pass_through_tensors",
                           "packaging", "runtime_tables")}
    n_coded_experts = 0
    n_omitted = 0
    for t in inv.tensors:
        oc = t.organ_class
        if oc in (A.ORGAN_EXP_GATE, A.ORGAN_EXP_UP, A.ORGAN_EXP_DOWN):
            layer, expert = int(t.layer), int(t.expert)
            live = (expert in keep_ids[layer]) if keep_ids else (expert < keep)
            if not live:
                n_omitted += 1
                continue
            n_coded_experts += 1
            spec = down_spec if oc == A.ORGAN_EXP_DOWN else gate_spec
            # function_aware ships the SAME index+codebook layout as shared_grammar; its only extra
            # payload is the per-row bf16 scale, charged explicitly under "scales" below.
            billed = dict(spec)
            if billed["family"] == "function_aware":
                billed["family"] = "shared_grammar"
            bits = SHB.expert_bits(t.shape, billed, keep)
            # split the charge back out into the ceiling's declared components
            n = t.shape[0] * t.shape[1]
            idx_bits = (n // int(spec["dim"])) * int(spec["stages"]) * math.ceil(
                math.log2(int(spec["k"])))
            cb_bits = int(spec["stages"]) * int(spec["k"]) * int(spec["dim"]) * 16 // keep
            comp["indices"] += idx_bits
            comp["codebooks"] += cb_bits
            comp["scales"] += t.shape[0] * ROW_SCALE_BITS + 16     # row scales + output gain
            comp["metadata"] += SHB.METADATA_BITS_PER_TENSOR
            comp["alignment"] += max(0, bits - idx_bits - cb_bits - 16 -
                                     SHB.METADATA_BITS_PER_TENSOR)
        elif oc in SHB._DENSE_ORGANS:
            comp["protected_islands"] += SHB._pq_bits(t.shape, dim=32, subspaces=8, k=16)
        else:
            comp["pass_through_tensors"] += SHB._native_bits(t.shape)
    # survivor bitmap: one bit per (layer, expert), needed resident to mask the router
    n_layers = len({int(t.layer) for t in inv.tensors if t.organ_class == A.ORGAN_EXP_GATE})
    n_experts = max(int(t.expert) for t in inv.tensors if t.organ_class == A.ORGAN_EXP_GATE) + 1
    comp["runtime_tables"] += n_layers * n_experts
    comp["packaging"] += RESERVE_BYTES * 8

    total = sum(comp.values())
    bpw = Fraction(total, inv.grand_params)
    return {"components": comp, "complete_bits": total,
            "complete_bytes": math.ceil(total / 8),
            "original_weight_count": inv.grand_params,
            "complete_bpw": round(float(bpw), 9),
            "complete_bpw_exact": f"{bpw.numerator}/{bpw.denominator}",
            "legal_under_one_bit_ceiling": bool(bpw <= CEILING),
            "coded_expert_tensors": n_coded_experts, "omitted_expert_tensors": n_omitted,
            "keep_experts_per_layer": keep, "n_experts": n_experts, "n_layers": n_layers}


def arm(inv, keep: int, gate_key: str, down_key: str,
        routing: dict[str, Any] | None = None) -> dict[str, Any]:
    g, d = GATE_RUNGS[gate_key], DOWN_RUNGS[down_key]
    led = ledger(inv, keep, g, d, routing)
    out = {"id": f"S{keep}_g{gate_key}_d{down_key}", "keep_experts": keep,
           "gate_up_index_bpw": float(index_rate(g)), "down_index_bpw": float(index_rate(d)),
           "gate_up_rd_floor_rel_error": round(rd_floor(float(index_rate(g))), 6),
           "down_rd_floor_rel_error": round(rd_floor(float(index_rate(d))), 6),
           "gate_spec": g, "down_spec": d, "ledger": led}
    if routing is not None:
        out["routing"] = routing_retained(routing, keep)
    return out


def build(routing_path: str | None, keeps: list[int]) -> dict[str, Any]:
    inv = A.build_inventory(A.load_config(), A.load_index())
    routing = json.loads(Path(routing_path).read_text()) if routing_path else None
    # For each inventory size, the richest legal rung pair under the ceiling.
    grid = [("0.625", "0.15625"), ("1.25", "0.3125"), ("1.25", "0.625"),
            ("2.5", "0.625"), ("2.5", "1.25"), ("5.0", "1.25"), ("5.0", "2.5")]
    arms = []
    for keep in keeps:
        best = None
        for gk, dk in grid:
            a = arm(inv, keep, gk, dk, routing)
            if a["ledger"]["legal_under_one_bit_ceiling"]:
                best = a if best is None or (
                    a["ledger"]["complete_bpw"] > best["ledger"]["complete_bpw"]) else best
            arms.append(a)
        if best is not None:
            best["richest_legal_for_this_inventory"] = True
    legal = [a for a in arms if a["ledger"]["legal_under_one_bit_ceiling"]]
    return {"schema": SCHEMA, "parent": "qwen3-235b-a22b-instruct-2507",
            "ceiling": "complete_bits / original_weight_count <= 1/1",
            "claim": "BYTE PLAN ONLY. No capability is claimed; only a real parent-vs-packed "
                     "forward may select a frontier.",
            "routing_source": routing_path,
            "n_arms": len(arms), "n_legal": len(legal), "arms": arms}


def demo() -> None:
    """Runnable check of the ledger's invariants on the real inventory (metadata only, no weights)."""
    inv = A.build_inventory(A.load_config(), A.load_index())
    full = ledger(inv, 128, GATE_RUNGS["0.625"], DOWN_RUNGS["0.15625"])
    half = ledger(inv, 64, GATE_RUNGS["1.25"], DOWN_RUNGS["0.3125"])
    quarter = ledger(inv, 32, GATE_RUNGS["2.5"], DOWN_RUNGS["0.625"])

    # every declared ceiling component is present
    assert set(full["components"]) >= {"indices", "codebooks", "scales", "pass_through_tensors"}
    # omission halves the coded tensor count each step
    assert half["coded_expert_tensors"] * 2 == full["coded_expert_tensors"], (
        half["coded_expert_tensors"], full["coded_expert_tensors"])
    assert quarter["coded_expert_tensors"] * 4 == full["coded_expert_tensors"]
    # budget neutrality: halving inventory while doubling rate keeps index bits within 1 percent
    for a, b in ((full, half), (half, quarter)):
        r = a["components"]["indices"] / b["components"]["indices"]
        assert 0.99 < r < 1.01, (r, a["components"]["indices"], b["components"]["indices"])
    # the ceiling is actually enforced, not decorative
    illegal = ledger(inv, 128, GATE_RUNGS["5.0"], DOWN_RUNGS["2.5"])
    assert not illegal["legal_under_one_bit_ceiling"] and illegal["complete_bpw"] > 1.0
    # exact rational rate agrees with the float
    num, den = full["complete_bpw_exact"].split("/")
    assert abs(int(num) / int(den) - full["complete_bpw"]) < 1e-9
    print(json.dumps({"ok": True,
                      "full_128_bpw": full["complete_bpw"], "half_64_bpw": half["complete_bpw"],
                      "quarter_32_bpw": quarter["complete_bpw"],
                      "illegal_128_at_5bpw": illegal["complete_bpw"]}, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Structural expert reduction plan under <= 1 BPW.")
    ap.add_argument("--routing", default="")
    ap.add_argument("--keeps", default="128,96,64,48,32,16")
    ap.add_argument("--out", default="")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    if args.demo:
        demo()
        return 0
    plan = build(args.routing or None, [int(x) for x in args.keeps.split(",")])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    for a in plan["arms"]:
        if not a["ledger"]["legal_under_one_bit_ceiling"]:
            continue
        r = a.get("routing", {})
        print(f"{a['id']:<26} bpw={a['ledger']['complete_bpw']:.6f} "
              f"gate_floor={a['gate_up_rd_floor_rel_error']:.4f} "
              f"down_floor={a['down_rd_floor_rel_error']:.4f} "
              f"routing_kept={r.get('top8_count_retained', float('nan')):.4f} "
              f"worst_layer={r.get('worst_layer_count_retained', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
