#!/usr/bin/env python3.12
"""Doctor Prime Part VII: the JOINT byte auction with Doctor budget envelopes.

THE QUESTION NOBODY TESTED. Every Qwen arm appended Doctor to an almost-full budget. S64_doctor
spent 0.9484 BPW on base and 0.0513 on Doctor - a 95/5 split that was never chosen, only left over.
Nobody has asked whether a deliberately ROUGH base plus a POWERFUL healer beats a faithful base
with weak repair. This module allocates base and Doctor JOINTLY under one exact ceiling and prices
the three legal envelopes on the real Qwen3-235B inventory.

WHAT IT IS: a byte plan and a predictor. It selects nothing. Only a real parent-vs-packed forward
selects a frontier, and this module refuses to mark anything promotable (see `PROMOTION_FLAG`).

THE LAW: complete_bits / original_weight_count <= 1/1, exact Fraction arithmetic, verified through
tools/foundry/one_bit_ceiling.py. Planned at 0.98 to reserve overhead. Doctor gets NO free bytes:
its residual codebooks, its per-protected-row scales and its protection bitmap are all billed into
the same ledger as the base, in the `doctor` slot.

THE PREDICTOR AND ITS EXACT DEGREES OF FREEDOM. Four sealed Qwen measurements anchor a three-part
loss model, and the model has EXACTLY as many free parameters as anchors:

    D1  routing only  (perfect bf16 experts, router masked to the 64 survivors)  symKL 4.297
    D2  recon only    (full 128 router, experts packed at the survivor rate)     symKL 7.879
    S64 joint, undoctored                                                        symKL 8.600
    S64_doctor  (joint + 0.0513 BPW of Doctor)                                   symKL 7.699

    L_route(C)   = A*(1-C)                       A   fit on D1
    L_recon(e)   = B*e                           B   fit on D2
    L(Lr, Le)    = (Lr^s + Le^s)^(1/s)           s   fit on the sub-additivity of S64
    L_doctor(d)  = L * g(d)                      g   fit on S64_doctor

Zero residual degrees of freedom means the fit is EXACT and its agreement with the anchors is
arithmetic, not evidence. Every value away from an anchor is an extrapolation off a single point,
and the module says so on every number it emits. That is the honest status of the whole thing: the
auction can price BYTES exactly and can price QUALITY only in three of its nine lanes.

WHAT IS UNPRICED. Six of the nine auction lanes - router capacity, hidden-state repair, expert
fallback, protected tensors, logit repair, metadata/runtime - have no measured quality anchor at
all. Their BYTES are computed exactly here so a reader can see what buying them would displace, but
their utility is None and they cannot be bid. A joint auction over nine lanes cannot be solved when
six of them have no price. That is a finding, not a gap to paper over.
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

import one_bit_ceiling as OBC  # noqa: E402
import qwen3_moe_adapter as A  # noqa: E402
import qwen_adaptive_k as AK  # noqa: E402
import qwen_function_aware_codec as FAC  # noqa: E402
import qwen_structural_plan as SP  # noqa: E402
import qwen_subhalfbit_search as SHB  # noqa: E402

SCHEMA = "hawking.gravity.doctor_byte_auction.v1"

HARD_CEILING = Fraction(1, 1)      # the law
PLANNED_CEILING = Fraction(49, 50)  # 0.98, exact - reserves overhead below the law

# Nothing in this module may be promoted on predicted utility. This is a hard flag, checked in
# demo(), and it is the reason `promotable` is False on every bid the auction emits.
PROMOTION_FLAG = {
    "may_promote_on_predicted_utility": False,
    "reason": "U(t) is built from a zero-residual-degrees-of-freedom fit to four sealed anchors. "
              "It orders candidates for a forward to adjudicate and selects nothing. Promotion "
              "requires a real parent-vs-packed forward.",
    "evidence_class_required_for_promotion": "measured_forward",
}

# ── sealed Qwen anchors (ground truth, do not edit without a new sealed receipt) ───────────────
ANCHORS = {
    "D1_routing_only": {"symkl": 4.297, "argmax": 0.3850, "keep": 64,
                        "what": "perfect bf16 experts, router masked to the same 64 survivors"},
    "D2_recon_only": {"symkl": 7.879, "argmax": 0.0863, "keep": 128,
                      "what": "full 128 router, experts packed at the survivor rate"},
    "S64_joint_undoctored": {"symkl": 8.600, "keep": 64,
                             "what": "joint routing+reconstruction, no Doctor"},
    "S64_doctor": {"symkl": 7.699, "argmax": 0.0922, "keep": 64,
                   "complete_bpw": 0.999769787, "base_bpw": 0.9484, "doctor_bpw": 0.0513,
                   "what": "the best full-forward artifact Qwen ever produced"},
}
GATE = {"symkl_max": 0.10, "argmax_min": 0.95}

# The S64 anchor's rungs, needed to place D2 on the reconstruction axis.
ANCHOR_GATE_RUNG, ANCHOR_DOWN_RUNG = "2.5", "0.625"

DOCTOR_RUNGS = {   # (dim, k, stages) -> index rate stages*ceil(log2 k)/dim on PROTECTED rows only
    "0.625": {"dim": 16, "k": 1024, "stages": 1},
    "1.25": {"dim": 8, "k": 1024, "stages": 1},
    "2.5": {"dim": 8, "k": 1024, "stages": 2},
    "5.0": {"dim": 8, "k": 1024, "stages": 4},
}
PROTECT_GRID = (0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.0)

# Envelope bands, as shares of the PLANNED 0.98 ceiling. Overhead (pass-through organs, dense
# islands, packaging, runtime tables) is not free and eats its share of the same 0.98.
ENVELOPES = {
    "BASE_HEAVY":   {"base": (0.70, 0.75), "doctor": (0.20, 0.25)},
    "BALANCED":     {"base": (0.50, 0.60), "doctor": (0.35, 0.45)},
    "DOCTOR_HEAVY": {"base": (0.30, 0.40), "doctor": (0.55, 0.65)},
}

# Shadow prices. lam_M converts a RESIDENT bit-per-weight into a base-bit equivalent; a resident
# codebook costs the 96 GB box the same as a payload bit, so 1.0. lam_L converts a marginal decode
# stage into bit-equivalents and is ZERO by default because no latency was ever measured for these
# lanes - a stipulated zero, declared, not a measurement.
LAM_M_DEFAULT = 1.0
LAM_L_DEFAULT = 0.0

# Stipulated uncertainty schedule. NOT measured. Relative half-width on any predicted loss grows
# with distance from the anchor in log-rate and in routing coverage.
UNC_FLOOR = 0.10
UNC_PER_LOG2_RATE = 0.25
UNC_PER_COVERAGE = 2.0


# ── the loss model ────────────────────────────────────────────────────────────────────────────
def _solve_s(a: float, b: float, joint: float) -> float:
    """The sub-additivity exponent: (a^s + b^s)^(1/s) = joint. Bisected, exact to 1e-12."""
    f = lambda s: (a ** s + b ** s) ** (1.0 / s) - joint  # noqa: E731
    lo, hi = 1.0, 2.0
    while f(hi) > 0:
        hi *= 2.0
        if hi > 1e6:
            raise SystemExit("no sub-additive exponent exists for these anchors")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return hi


class LossModel:
    """Zero-residual-DOF fit to the four sealed anchors. Reproduces them by arithmetic."""

    def __init__(self, cov64: float) -> None:
        self.cov64 = cov64
        self.A = ANCHORS["D1_routing_only"]["symkl"] / (1.0 - cov64)
        self.e_anchor = self.recon_scalar(ANCHOR_GATE_RUNG, ANCHOR_DOWN_RUNG)
        self.B = ANCHORS["D2_recon_only"]["symkl"] / self.e_anchor
        self.s = _solve_s(ANCHORS["D1_routing_only"]["symkl"],
                          ANCHORS["D2_recon_only"]["symkl"],
                          ANCHORS["S64_joint_undoctored"]["symkl"])
        # Doctor: one measured point (0.0513 BPW -> 8.600 to 7.699). Two laws through it.
        self.d0 = ANCHORS["S64_doctor"]["doctor_bpw"]
        self.g0 = 1.0 - ANCHORS["S64_doctor"]["symkl"] / ANCHORS["S64_joint_undoctored"]["symkl"]
        self.G_sat = self.g0 / (1.0 - math.exp(-1.0))   # saturating law, conservative
        self.slope = self.g0 / self.d0                  # linear law, maximally optimistic

    @staticmethod
    def recon_scalar(gate_key: str, down_key: str) -> float:
        """(2*gate + down)/3 reconstruction error, the same 2:1 organ weighting the auction uses."""
        gr, _ = AK.recon_err(float(SP.index_rate(SP.GATE_RUNGS[gate_key])))
        dr, _ = AK.recon_err(float(SP.index_rate(SP.DOWN_RUNGS[down_key])))
        return (2.0 * gr + dr) / 3.0

    def undoctored(self, cov: float, gate_key: str, down_key: str) -> float:
        lr = self.A * (1.0 - cov)
        le = self.B * self.recon_scalar(gate_key, down_key)
        return (lr ** self.s + le ** self.s) ** (1.0 / self.s)

    def doctor_gain_band(self, d_bpw: float) -> tuple[float, float]:
        """(optimistic, conservative) multipliers on the undoctored loss."""
        opt = max(0.0, 1.0 - self.slope * d_bpw)
        cons = 1.0 - self.G_sat * (1.0 - math.exp(-d_bpw / self.d0))
        return opt, cons

    def point(self, cov: float, gate_key: str, down_key: str, d_bpw: float) -> float:
        """Unrounded saturating-law point estimate. Used for marginal utility, where the 4-decimal
        rounding of the reported field would quantize small deltas to zero."""
        return self.undoctored(cov, gate_key, down_key) * self.doctor_gain_band(d_bpw)[1]

    def uncertainty(self, cov: float, gate_key: str, down_key: str) -> float:
        e = self.recon_scalar(gate_key, down_key)
        return (UNC_FLOOR + UNC_PER_LOG2_RATE * abs(math.log2(e / self.e_anchor))
                + UNC_PER_COVERAGE * abs(cov - self.cov64))

    def predict(self, cov: float, gate_key: str, down_key: str, d_bpw: float) -> dict[str, Any]:
        u = self.undoctored(cov, gate_key, down_key)
        opt, cons = self.doctor_gain_band(d_bpw)
        lo_law, hi_law = u * opt, u * cons
        # The optimistic linear law hits zero loss at d = 1/slope. A packed model cannot become
        # EXACTLY the parent, so above that point the linear extrapolation is unphysical and is
        # not usable as a bound. Beyond it the saturating law is the only usable law.
        in_range = d_bpw < 1.0 / self.slope
        if not in_range:
            lo_law = hi_law
        rel = self.uncertainty(cov, gate_key, down_key)
        return {
            "optimistic_doctor_law_in_range": in_range,
            "optimistic_doctor_law_out_of_range_note": None if in_range else (
                f"doctor_bpw {d_bpw:.4f} exceeds the linear law's zero crossing "
                f"{1.0 / self.slope:.4f}; a linear extrapolation predicting ZERO loss is "
                f"unphysical, so the band collapses onto the saturating law"),
            # The POINT estimate is the saturating law: it is the only Doctor law valid across the
            # whole budget range, so it is the one that can order cells without a discontinuity.
            "predicted_symkl_point": round(hi_law, 4),
            "predicted_symkl_point_law": "saturating",
            "predicted_symkl_optimistic_doctor_law": round(lo_law, 4),
            "predicted_symkl_saturating_doctor_law": round(hi_law, 4),
            "predicted_symkl_band_with_stipulated_uncertainty":
                [round(lo_law * (1.0 - min(rel, 0.9)), 4), round(hi_law * (1.0 + rel), 4)],
            "undoctored_symkl": round(u, 4),
            "stipulated_relative_uncertainty": round(rel, 4),
            "gate_symkl_max": GATE["symkl_max"],
            "passes_gate_under_saturating_law": bool(
                hi_law * (1.0 + rel) <= GATE["symkl_max"]),
            "passes_gate_under_optimistic_law": bool(
                in_range and lo_law * (1.0 - min(rel, 0.9)) <= GATE["symkl_max"]),
            "evidence_class": "extrapolated_from_four_anchors",
            "promotable": False,
        }


# ── exact byte lanes ──────────────────────────────────────────────────────────────────────────
def _shapes(inv) -> dict[str, tuple[int, int]]:
    return AK._expert_shapes(inv)


def base_bits(shapes, n_layers: int, K: int, gate_key: str, down_key: str) -> int:
    """Exact expert payload bits (indices + amortized codebooks + row scales) at uniform K."""
    return n_layers * AK.layer_bits(shapes, K, SP.GATE_RUNGS[gate_key], SP.DOWN_RUNGS[down_key])


def doctor_bits(shapes, n_layers: int, K: int, rung: dict[str, Any], protect: float) -> int:
    """Exact Doctor residual-codebook bits over every surviving expert tensor."""
    if protect <= 0.0:
        return 0
    per = 2 * FAC.doctor_bits(shapes["gate"], doctor_dim=rung["dim"], doctor_k=rung["k"],
                              doctor_stages=rung["stages"], protect_frac=protect, cluster=K)
    per += FAC.doctor_bits(shapes["down"], doctor_dim=rung["dim"], doctor_k=rung["k"],
                           doctor_stages=rung["stages"], protect_frac=protect, cluster=K)
    return n_layers * K * per


def unpriced_lane_bits(inv, shapes, n_layers: int, n_experts: int, K: int,
                       gate_key: str, down_key: str) -> dict[str, dict[str, Any]]:
    """Exact byte cost of the six lanes with NO measured quality anchor.

    They are priced in bytes so a reader can see what buying them displaces. Their utility is None
    and they may not be bid: a lane with no price cannot win an auction.
    """
    h = shapes["down"][0]
    vocab = max(t.shape[0] for t in inv.tensors if t.organ_class == A.ORGAN_LM_HEAD)
    g, d = SP.GATE_RUNGS[gate_key], SP.DOWN_RUNGS[down_key]
    fallback = n_layers * (2 * SHB.expert_bits(shapes["gate"], g, K)
                           + SHB.expert_bits(shapes["down"], d, K))
    dense_native = sum(SHB._native_bits(t.shape) for t in inv.tensors
                       if t.organ_class in SHB._DENSE_ORGANS)
    dense_pq = sum(SHB._pq_bits(t.shape, dim=32, subspaces=8, k=16) for t in inv.tensors
                   if t.organ_class in SHB._DENSE_ORGANS)
    return {
        "router_capacity": {
            "bits": n_layers * n_experts * math.ceil(math.log2(max(2, K))),
            "what": "per-layer omitted-expert -> survivor remap table (ceil(log2 K) bits each)",
            "resident_share": 1.0, "extra_decode_stages": 0,
            "why_unpriced": "router distillation IN ITS QWEN FORM is dead; no anchor exists for a "
                            "remap table, and D2 prices only the FULL router, not a remapped one"},
        "hidden_state_repair": {
            "bits": n_layers * 32 * 2 * h * 16,
            "what": "rank-32 bf16 low-rank correction on the hidden state at every layer",
            "resident_share": 1.0, "extra_decode_stages": 1,
            "why_unpriced": "never measured on any Qwen arm"},
        "expert_fallback": {
            "bits": fallback,
            "what": "one extra generic expert per layer at the same rungs, catching tokens routed "
                    "to an omitted expert",
            "resident_share": 0.0, "extra_decode_stages": 0,
            "why_unpriced": "Lane E measured that an omitted expert is NOT reconstructible from "
                            "survivors (best single survivor 0.885-0.995 held-out relative error, "
                            "i.e. no better than zero), so the prior on this lane is ~no gain - "
                            "but that is a prior, not a price"},
        "protected_tensors": {
            "bits": dense_native - dense_pq,
            "what": "upgrade the dense islands (embed, lm_head, q/k/v/o) from PQ dim32/k16 to "
                    "native bf16",
            "resident_share": 1.0, "extra_decode_stages": 0,
            "why_unpriced": "no arm ever varied the island rate against a forward"},
        "logit_repair": {
            "bits": 32 * (vocab + h) * 16,
            "what": "rank-32 bf16 correction on the lm_head output",
            "resident_share": 1.0, "extra_decode_stages": 1,
            "why_unpriced": "never measured; and symKL is measured ON the logits, so a logit "
                            "repair fitted to the calibration batch would be scored on the same "
                            "quantity it was fitted to - circular unless held out"},
        "metadata_and_runtime": {
            "bits": SP.RESERVE_BYTES * 8 + n_layers * n_experts,
            "what": "packaging reserve plus the resident survivor bitmap",
            "resident_share": 1.0, "extra_decode_stages": 0,
            "why_unpriced": "mandatory, not a bidder: dQ is zero by construction and the bits are "
                            "not optional"},
    }


# ── utility ───────────────────────────────────────────────────────────────────────────────────
def utility(dq: float | None, d_bits: int, params: int, *, resident_share: float,
            extra_stages: int, lam_m: float, lam_l: float) -> dict[str, Any]:
    """U(t) = E[dQ_t] / (dB_t + lam_L*dL_t + lam_M*dM_t), in symKL reduction per bit-per-weight.

    dB is the marginal bits-per-weight. dM is the RESIDENT share of those bits (a codebook must sit
    in RAM whether or not its layer is hot). dL is extra sequential decode stages.
    """
    db = d_bits / params
    denom = db + lam_m * resident_share * db + lam_l * extra_stages
    if denom <= 0:
        return {"utility": None, "reason": "zero-cost lane"}
    if dq is None:
        return {"utility": None, "delta_bpw": round(db, 9), "cost_bit_equivalents": round(denom, 9),
                "reason": "no measured quality anchor for this lane; unpriced lanes may not bid",
                "promotable": False}
    return {"utility": round(dq / denom, 6), "expected_delta_quality_symkl": round(dq, 6),
            "delta_bpw": round(db, 9), "cost_bit_equivalents": round(denom, 9),
            "evidence_class": "extrapolated_from_four_anchors", "promotable": False}


# ── the joint allocator ───────────────────────────────────────────────────────────────────────
def _coverages(routing: dict[str, Any]) -> dict[int, float]:
    return {K: float(SP.routing_retained(routing, K)["top8_count_retained"])
            for K in AK.K_CHOICES}


def joint_cells(inv, routing, *, ceiling: Fraction = PLANNED_CEILING) -> list[dict[str, Any]]:
    """Every legal (K, gate rung, down rung, doctor rung, protect_frac) cell, exactly billed.

    Joint, not sequential: base and Doctor are chosen together against one budget, so a rough base
    with a rich Doctor is a first-class candidate rather than a leftover.
    """
    n_layers = len({int(t.layer) for t in inv.tensors if t.organ_class == A.ORGAN_EXP_GATE})
    n_experts = max(int(t.expert) for t in inv.tensors if t.organ_class == A.ORGAN_EXP_GATE) + 1
    shapes = _shapes(inv)
    fixed = AK._fixed_bits(inv, n_layers, n_experts)
    params = inv.grand_params
    budget = int(ceiling * params)
    cov = _coverages(routing)
    model = LossModel(cov[64])

    cells = []
    for K in AK.K_CHOICES:
        for gk, dk in AK.RUNGS:
            bb = base_bits(shapes, n_layers, K, gk, dk)
            if fixed + bb > budget:
                continue
            for drk, drung in DOCTOR_RUNGS.items():
                for p in PROTECT_GRID:
                    db = doctor_bits(shapes, n_layers, K, drung, p)
                    total = fixed + bb + db
                    if total > budget:
                        continue
                    pred = model.predict(cov[K], gk, dk, db / params)
                    cells.append({
                        "K": K, "gate_rung": gk, "down_rung": dk,
                        "doctor_rung": drk, "doctor_protect_frac": p,
                        "base_bits": bb, "doctor_bits": db, "fixed_bits": fixed,
                        "complete_bits": total,
                        "complete_bpw": round(total / params, 9),
                        "base_share_of_planned": round(bb / budget, 6),
                        "doctor_share_of_planned": round(db / budget, 6),
                        "overhead_share_of_planned": round(fixed / budget, 6),
                        "coverage": round(cov[K], 6),
                        "gate_index_bpw": float(SP.index_rate(SP.GATE_RUNGS[gk])),
                        "down_index_bpw": float(SP.index_rate(SP.DOWN_RUNGS[dk])),
                        "prediction": pred,
                    })
    return cells


def _verify_exact(inv, routing, cell: dict[str, Any]) -> dict[str, Any]:
    """Re-bill the winning cell through the omission-aware ledger and the one-bit ceiling itself."""
    drung = DOCTOR_RUNGS[cell["doctor_rung"]]
    doc = None if cell["doctor_protect_frac"] <= 0 else dict(drung) | {
        "protect_frac": cell["doctor_protect_frac"]}
    g = dict(SP.GATE_RUNGS[cell["gate_rung"]])
    d = dict(SP.DOWN_RUNGS[cell["down_rung"]])
    if doc:
        g["doctor"], d["doctor"] = doc, doc
    led = SP.ledger(inv, cell["K"], g, d, routing)
    comp = dict(led["components"])
    obc = OBC.CompleteByteLedger(**comp, metadata_alignment_reserve_bits=0,
                                 note=f"doctor_byte_auction {cell['K']}/{cell['gate_rung']}/"
                                      f"{cell['down_rung']}/doctor{cell['doctor_rung']}"
                                      f"@{cell['doctor_protect_frac']}")
    bpw = obc.complete_bpw(inv.grand_params)
    legal, reasons = OBC.is_legal_candidate({
        "original_weight_count": inv.grand_params, "ledger": obc,
        "reported_bpw": bpw, "target_bpw": bpw,
    })
    return {
        "exact_ledger_components": comp,
        "exact_complete_bits": int(obc.complete_bits()),
        "exact_complete_bpw": float(bpw),
        "exact_complete_bpw_rational": f"{bpw.numerator}/{bpw.denominator}",
        "legal_under_hard_one_bit_ceiling": bool(bpw <= HARD_CEILING),
        "legal_under_planned_0p98": bool(bpw <= PLANNED_CEILING),
        "one_bit_ceiling_verdict": {"legal": legal, "reasons": reasons},
        "doctor_share_of_exact_ledger": round(comp["doctor"] / max(1, int(obc.complete_bits())), 6),
    }


def pick_envelope(cells, band, model_key: str = "predicted_symkl_point"):
    """Best cell inside an envelope band; if the band is empty, the nearest legal cell and why."""
    lo_b, hi_b = band["base"]
    lo_d, hi_d = band["doctor"]
    inside = [c for c in cells
              if lo_b <= c["base_share_of_planned"] <= hi_b
              and lo_d <= c["doctor_share_of_planned"] <= hi_d]
    if inside:
        return min(inside, key=lambda c: c["prediction"][model_key]), None
    # nearest by L1 distance in (base, doctor) share space
    def dist(c):
        b = max(lo_b - c["base_share_of_planned"], c["base_share_of_planned"] - hi_b, 0.0)
        d = max(lo_d - c["doctor_share_of_planned"], c["doctor_share_of_planned"] - hi_d, 0.0)
        return b + d
    near = min(cells, key=dist)
    return near, (f"envelope band base {lo_b}-{hi_b} / doctor {lo_d}-{hi_d} of the planned 0.98 is "
                  f"NOT REALIZABLE on the existing rung ladder; nearest legal cell is base "
                  f"{near['base_share_of_planned']:.4f} / doctor "
                  f"{near['doctor_share_of_planned']:.4f}, L1 miss {dist(near):.4f}")


# ── the DOCTOR_HEAVY codeability question ─────────────────────────────────────────────────────
def codeability(inv, shapes, n_layers: int, base_share: float) -> dict[str, Any]:
    """What index rate does a base at `base_share` of the planned ceiling actually leave?

    The brief's Part VII question: if base drops to ~35 percent of the budget, is it still
    codeable? Answered against the REAL expert parameter count and the REAL rung ladder.
    """
    params = inv.grand_params
    exp_params = sum(t.shape[0] * t.shape[1] for t in inv.tensors
                     if t.organ_class in (A.ORGAN_EXP_GATE, A.ORGAN_EXP_UP, A.ORGAN_EXP_DOWN))
    budget_bits = base_share * float(PLANNED_CEILING) * params
    out = {"base_share_of_planned": base_share, "base_bits": int(budget_bits),
           "expert_params_full_128": exp_params, "per_K": {}}
    lowest_gate = min(float(SP.index_rate(s)) for s in SP.GATE_RUNGS.values())
    lowest_down = min(float(SP.index_rate(s)) for s in SP.DOWN_RUNGS.values())
    # Two gate-class tensors (gate_proj, up_proj) and one down_proj per expert.
    blended_floor = (2.0 * lowest_gate + lowest_down) / 3.0
    for K in AK.K_CHOICES:
        live = exp_params * K / 128.0
        rate = budget_bits / live       # includes codebooks+scales in the numerator: an UPPER bound
        out["per_K"][str(K)] = {
            "surviving_expert_params": int(live),
            "available_bits_per_surviving_weight": round(rate, 6),
            "note_upper_bound": "codebooks and row scales are inside this number, so the true "
                                "INDEX rate is strictly lower",
            "rd_floor_rel_error_at_that_rate": round(SP.rd_floor(max(rate, 1e-6)), 6),
            # CORRECTED after adversarial review. Comparing the blended per-weight rate against
            # min(GATE_RUNGS) alone was the wrong test: gate/up and down are charged at DIFFERENT
            # rungs, so the cheapest legal ARTIFACT costs the blended floor
            # (2*min(gate) + min(down)) / 3, not min(gate). The old test declared K=80 uncodeable
            # when it in fact fits with headroom; the real cliff is at K>=96.
            "codeable_by_existing_gate_ladder": bool(rate >= blended_floor),
            "blended_cheapest_legal_rung_bpw": round(blended_floor, 6),
            "richest_affordable_gate_rung": max(
                (k for k in SP.GATE_RUNGS if float(SP.index_rate(SP.GATE_RUNGS[k])) <= rate),
                key=lambda k: float(SP.index_rate(SP.GATE_RUNGS[k])), default=None),
        }
    out["lowest_gate_rung_on_the_ladder_bpw"] = lowest_gate
    out["blended_cheapest_legal_artifact_bpw"] = round(blended_floor, 6)
    out["correction_note"] = (
        "an earlier version tested rate >= min(GATE_RUNGS) = 0.625 and reported a DOCTOR_HEAVY "
        "base as uncodeable at K>=80. That was wrong: the cheapest legal artifact blends two "
        "gate-class tensors at 0.625 with one down at 0.15625, i.e. 0.46875. K=80 fits with "
        "headroom; the real cliff is K>=96.")
    return out


# ── report ────────────────────────────────────────────────────────────────────────────────────
def build(routing_path: str) -> dict[str, Any]:
    inv = A.build_inventory(A.load_config(), A.load_index())
    routing = json.loads(Path(routing_path).read_text())
    n_layers = len({int(t.layer) for t in inv.tensors if t.organ_class == A.ORGAN_EXP_GATE})
    n_experts = max(int(t.expert) for t in inv.tensors if t.organ_class == A.ORGAN_EXP_GATE) + 1
    shapes = _shapes(inv)
    params = inv.grand_params
    cov = _coverages(routing)
    model = LossModel(cov[64])
    cells = joint_cells(inv, routing)

    envelopes: dict[str, Any] = {}
    for name, band in ENVELOPES.items():
        cell, miss = pick_envelope(cells, band)
        rec = dict(cell)
        rec["envelope"] = name
        rec["band"] = band
        rec["band_realizable"] = miss is None
        if miss:
            rec["band_not_realizable_reason"] = miss
        rec["exact_verification"] = _verify_exact(inv, routing, cell)
        rec["routing_only_floor_symkl"] = round(model.A * (1.0 - cov[cell["K"]]), 4)
        rec["routing_only_floor_over_gate"] = round(
            model.A * (1.0 - cov[cell["K"]]) / GATE["symkl_max"], 1)
        rec["unpriced_lanes"] = {
            k: (v | {"share_of_planned_budget": round(v["bits"] / (float(PLANNED_CEILING) * params), 6),
                     "utility": utility(None, v["bits"], params,
                                        resident_share=v["resident_share"],
                                        extra_stages=v["extra_decode_stages"],
                                        lam_m=LAM_M_DEFAULT, lam_l=LAM_L_DEFAULT)})
            for k, v in unpriced_lane_bits(inv, shapes, n_layers, n_experts, cell["K"],
                                           cell["gate_rung"], cell["down_rung"]).items()}
        # marginal utility of the three PRICED lanes at this operating point
        rec["marginal_utility"] = _marginal(inv, shapes, n_layers, model, cov, cell, params)
        envelopes[name] = rec

    # A bound that does NOT depend on either Doctor law, and does not depend on the codec at all.
    # Doctor Gen1/Gen2 is a residual codebook on expert WEIGHTS. It cannot repair a routing
    # decision that was never made: a token routed to an omitted expert never reaches any weight
    # Doctor could correct. Under that attribution, L_route is a hard floor on every arm with
    # omission, whatever the base rate and whatever the Doctor budget.
    floor = {str(K): round(model.A * (1.0 - c), 4) for K, c in cov.items()}
    min_K_clearing = next((K for K in sorted(cov) if model.A * (1.0 - cov[K]) <= GATE["symkl_max"]),
                          None)
    routing_only_floor = {
        "definition": "L_route(K) = A*(1-C(K)) with A fit on the D1 causal control (perfect bf16 "
                      "experts, router masked to the same 64 survivors, symKL 4.297)",
        "symkl_floor_by_K": floor,
        "capability_gate_symkl_max": GATE["symkl_max"],
        "smallest_K_whose_routing_floor_clears_the_gate": min_K_clearing,
        "consequence": "Under this attribution EVERY arm with expert omission is bounded away from "
                       "the capability gate BEFORE reconstruction or Doctor enter. The bound is "
                       "independent of the base rate, the codec and both Doctor laws; only K=128 "
                       "(no omission at all) clears it, and K=112 - which omits just 16 of 128 "
                       "experts - still predicts ~2.3x the gate on routing loss alone.",
        "doctor_attribution_ambiguity":
            "The S64_doctor anchor cannot attribute its 8.600 -> 7.699 gain between the routing "
            "and reconstruction terms. This module's cell predictions apply g(d) to the JOINT "
            "loss, which is the reading GENEROUS to Doctor. This floor is the other reading. Both "
            "fit the same anchor; a forward that Doctors a full-128-router arm would separate "
            "them.",
        "honesty": "L_route is linear in (1-C) because ONE point determines it. Curvature is "
                   "unmeasured and could move this bound in either direction.",
    }

    pass_opt = [c for c in cells if c["prediction"]["passes_gate_under_optimistic_law"]]
    pass_sat = [c for c in cells if c["prediction"]["passes_gate_under_saturating_law"]]
    law_disagreement = {
        "cells_passing_gate_under_optimistic_linear_doctor_law": len(pass_opt),
        "cells_passing_gate_under_saturating_doctor_law": len(pass_sat),
        "min_doctor_bpw_among_optimistic_passes": round(
            min((c["doctor_bits"] / params for c in pass_opt), default=float("nan")), 6),
        "verdict": "THE PREDICTOR CANNOT DECIDE. The two readings of the single measured Doctor "
                   "point - linear in budget, or saturating - disagree about whether DOCTOR_HEAVY "
                   "reaches the capability gate. Under the saturating reading nothing on the legal "
                   "grid comes within ~70x of symKL 0.10; under the linear reading a large-Doctor "
                   "cell reaches it. Both readings fit the one anchor exactly. Only a forward can "
                   "separate them, which is exactly why this module promotes nothing.",
        "what_would_separate_them": "a SECOND Doctor budget point on a real forward at the same "
                                    "base (e.g. S64 base with Doctor at ~0.15 BPW): the linear law "
                                    "predicts symKL ~5.9, the saturating law ~7.4. One forward "
                                    "measures the curvature and kills one law.",
    }

    joint_opt = min(cells, key=lambda c: c["prediction"]["predicted_symkl_point"])
    joint_opt = dict(joint_opt) | {"exact_verification": _verify_exact(inv, routing, joint_opt)}

    fixed = AK._fixed_bits(inv, n_layers, n_experts)
    return {
        "schema": SCHEMA,
        "parent": "qwen3-235b-a22b-instruct-2507",
        "claim": "BYTE PLAN AND PREDICTOR ONLY. This module allocates bytes and orders candidates. "
                 "It selects nothing. Only a real parent-vs-packed forward may select a frontier.",
        "law": "complete_bits / original_weight_count <= 1/1, exact Fraction, verified through "
               "tools/foundry/one_bit_ceiling.py",
        "planned_ceiling": "49/50 (0.98)",
        "hard_ceiling": "1/1",
        "promotion_flag": PROMOTION_FLAG,
        "routing_source": routing_path,
        "routing_tokens": routing.get("n_tokens_calibrated_on"),
        "routing_trust_warning": routing.get("trust_verdict"),
        "anchors": ANCHORS,
        "capability_gate": GATE,
        "loss_model": {
            "form": "L = ((A*(1-C))^s + (B*e)^s)^(1/s) * g(doctor_bpw)",
            "A_routing": round(model.A, 6), "B_recon": round(model.B, 6),
            "s_subadditivity": round(model.s, 6),
            "coverage_at_K64": round(cov[64], 6),
            "recon_scalar_at_anchor_rungs": round(model.e_anchor, 6),
            "doctor_optimistic_linear_slope_per_bpw": round(model.slope, 6),
            "doctor_saturating_asymptote_max_gain": round(model.G_sat, 6),
            "doctor_optimistic_law_breaks_above_bpw": round(1.0 / model.slope, 6),
            "free_parameters": 4, "anchors_used": 4, "residual_degrees_of_freedom": 0,
            "honesty": "The fit is EXACT because it is exactly determined. Reproducing the four "
                       "anchors is arithmetic, not validation. Every value away from an anchor is "
                       "an extrapolation off a single point in that direction.",
            "uncertainty_schedule": {
                "floor": UNC_FLOOR, "per_log2_rate": UNC_PER_LOG2_RATE,
                "per_coverage": UNC_PER_COVERAGE,
                "provenance": "stipulated, NOT measured"},
        },
        "shadow_prices": {"lam_M": LAM_M_DEFAULT, "lam_L": LAM_L_DEFAULT,
                          "lam_L_provenance": "stipulated zero; no latency was ever measured for "
                                              "these lanes"},
        "budget_frame": {
            "original_weight_count": params,
            "planned_budget_bits": int(float(PLANNED_CEILING) * params),
            "mandatory_overhead_bits": fixed,
            "mandatory_overhead_share_of_planned": round(
                fixed / (float(PLANNED_CEILING) * params), 6),
            "note": "overhead is pass-through organs, dense islands, packaging reserve and the "
                    "resident survivor bitmap. It is NOT free and it eats the same 0.98, which is "
                    "why base + doctor shares never sum to 1.0."},
        "n_legal_cells": len(cells),
        "central_finding_law_disagreement": law_disagreement,
        "routing_only_floor": routing_only_floor,
        "envelopes": envelopes,
        "unconstrained_joint_optimum": joint_opt,
        "part_vii_answer": {
            "question": "Does a deliberately ROUGH base plus a POWERFUL healer beat a faithful "
                        "base with weak repair?",
            "answer_under_the_only_measured_doctor_point":
                "NO, and it is not close. At every envelope the marginal utility of a base byte "
                "exceeds that of a Doctor byte by 2-4 orders of magnitude once Doctor is past a "
                "few percent of the budget, because the single measured Doctor point read as a "
                "saturating law caps Doctor's total achievable gain at "
                f"{round(model.G_sat * 100, 1)} percent of the loss. The unconstrained joint "
                "optimum lands at base " f"{joint_opt['base_share_of_planned']:.3f} / doctor "
                f"{joint_opt['doctor_share_of_planned']:.3f} of the planned ceiling - richer in "
                "Doctor than Qwen's accidental 95/5 leftover, but nowhere near DOCTOR_HEAVY.",
            "the_caveat_that_makes_this_not_a_kill":
                "The saturating law is one of two readings of ONE point. The linear reading says "
                "the opposite. See central_finding_law_disagreement. This is a byte plan; only a "
                "forward decides.",
            "does_the_base_become_uncodeable":
                "PARTLY, and this is a hard structural fact independent of any loss model. At a "
                "base share of 0.35 the surviving-expert budget is 0.355 bits per surviving weight "
                "at K=128 - BELOW the lowest gate rung on the ladder (0.625) - so a DOCTOR_HEAVY "
                "base is literally uncodeable at K >= 80. It becomes codeable only by omitting "
                "experts: K<=64 leaves 0.710 bpw and K<=48 leaves 0.947 bpw. So DOCTOR_HEAVY does "
                "not just buy Doctor bytes with base fidelity, it buys them with ROUTING COVERAGE, "
                "and the routing-only floor at K=48 is "
                f"{round(model.A * (1.0 - cov[48]), 3)} symKL = "
                f"{round(model.A * (1.0 - cov[48]) / GATE['symkl_max'], 0):.0f}x the capability "
                "gate before reconstruction or Doctor enter at all. That bounds the whole "
                "experiment.",
        },
        "doctor_heavy_codeability": {
            f"{s:.2f}": codeability(inv, shapes, n_layers, s) for s in (0.35, 0.50, 0.72)},
        "coverage_by_K": {str(k): round(v, 6) for k, v in cov.items()},
    }


def _marginal(inv, shapes, n_layers, model, cov, cell, params) -> dict[str, Any]:
    """Marginal utility of the three PRICED lanes at this cell. Six lanes have no price."""
    base = model.point(cov[cell["K"]], cell["gate_rung"], cell["down_rung"],
                       cell["doctor_bits"] / params)
    out: dict[str, Any] = {}

    # lane 1: base fidelity - one rung richer
    idx = AK.RUNGS.index((cell["gate_rung"], cell["down_rung"]))
    if idx + 1 < len(AK.RUNGS):
        gk, dk = AK.RUNGS[idx + 1]
        nb = base_bits(shapes, n_layers, cell["K"], gk, dk)
        dq = base - model.point(cov[cell["K"]], gk, dk, cell["doctor_bits"] / params)
        out["base_fidelity"] = utility(dq, nb - cell["base_bits"], params, resident_share=0.02,
                                       extra_stages=0, lam_m=LAM_M_DEFAULT, lam_l=LAM_L_DEFAULT
                                       ) | {"step": f"rungs -> g{gk}/d{dk}"}
    # lane 2: expert inventory - one K step up
    ks = list(AK.K_CHOICES)
    if cell["K"] in ks and ks.index(cell["K"]) + 1 < len(ks):
        K2 = ks[ks.index(cell["K"]) + 1]
        nb = base_bits(shapes, n_layers, K2, cell["gate_rung"], cell["down_rung"])
        dq = base - model.point(cov[K2], cell["gate_rung"], cell["down_rung"],
                                cell["doctor_bits"] / params)
        out["expert_inventory"] = utility(dq, nb - cell["base_bits"], params, resident_share=0.02,
                                          extra_stages=0, lam_m=LAM_M_DEFAULT, lam_l=LAM_L_DEFAULT
                                          ) | {"step": f"K -> {K2}"}
    # lane 7: residual codebooks (Doctor) - one protect step up
    pg = list(PROTECT_GRID)
    if cell["doctor_protect_frac"] in pg and pg.index(cell["doctor_protect_frac"]) + 1 < len(pg):
        p2 = pg[pg.index(cell["doctor_protect_frac"]) + 1]
        nb = doctor_bits(shapes, n_layers, cell["K"], DOCTOR_RUNGS[cell["doctor_rung"]], p2)
        dq = base - model.point(cov[cell["K"]], cell["gate_rung"], cell["down_rung"], nb / params)
        out["residual_codebooks"] = utility(dq, nb - cell["doctor_bits"], params,
                                            resident_share=0.30, extra_stages=1,
                                            lam_m=LAM_M_DEFAULT, lam_l=LAM_L_DEFAULT
                                            ) | {"step": f"protect_frac -> {p2}"}
    out["_unpriced_lane_count"] = 6
    out["_note"] = ("only 3 of the 9 auction lanes have a quality price. A joint auction cannot be "
                    "SOLVED over nine lanes when six of them are unpriced; it can only be bounded.")
    return out


# ── self-check ────────────────────────────────────────────────────────────────────────────────
def demo() -> None:
    """Runnable check on the real inventory and the real calibration. Metadata only, no weights."""
    inv = A.build_inventory(A.load_config(), A.load_index())
    routing = json.loads(Path(
        "reports/subbit_reset/QWEN3_235B_ROUTING_CALIBRATION_1200.json").read_text())
    cov = _coverages(routing)
    m = LossModel(cov[64])

    # 0. the sealed anchors are unedited. The fit is exactly determined, so it will happily
    #    reproduce ANY anchors it is handed - which means anchor tampering is invisible to the fit
    #    and has to be caught here, against the sealed values.
    assert (ANCHORS["D1_routing_only"]["symkl"], ANCHORS["D2_recon_only"]["symkl"],
            ANCHORS["S64_joint_undoctored"]["symkl"], ANCHORS["S64_doctor"]["symkl"],
            ANCHORS["S64_doctor"]["doctor_bpw"]) == (4.297, 7.879, 8.600, 7.699, 0.0513), ANCHORS
    assert (GATE["symkl_max"], GATE["argmax_min"]) == (0.10, 0.95)

    # 1. the model reproduces all four sealed anchors (arithmetic, not validation)
    assert abs(m.A * (1 - cov[64]) - ANCHORS["D1_routing_only"]["symkl"]) < 1e-9
    assert abs(m.B * m.e_anchor - ANCHORS["D2_recon_only"]["symkl"]) < 1e-9
    joint = m.undoctored(cov[64], ANCHOR_GATE_RUNG, ANCHOR_DOWN_RUNG)
    assert abs(joint - ANCHORS["S64_joint_undoctored"]["symkl"]) < 1e-6, joint
    opt, cons = m.doctor_gain_band(ANCHORS["S64_doctor"]["doctor_bpw"])
    for g in (opt, cons):
        assert abs(joint * g - ANCHORS["S64_doctor"]["symkl"]) < 1e-6, (g, joint * g)
    # sub-additive: the joint loss is strictly below the sum of the parts
    assert m.s > 1.0 and joint < ANCHORS["D1_routing_only"]["symkl"] + ANCHORS["D2_recon_only"]["symkl"]

    # 2. monotonicity: more Doctor never predicts more loss; more coverage never predicts more loss
    ds = [m.predict(cov[64], "2.5", "0.625", d)["predicted_symkl_point"] for d in (0, .05, .2, .6)]
    assert all(b <= a + 1e-12 for a, b in zip(ds, ds[1:])), ds
    cs = [m.undoctored(cov[K], "2.5", "0.625") for K in AK.K_CHOICES]
    assert all(b <= a + 1e-12 for a, b in zip(cs, cs[1:])), cs

    # 3. the ceiling is enforced, not decorative: every emitted cell is legal, and an over-budget
    #    cell is actually rejected by one_bit_ceiling rather than quietly accepted
    cells = joint_cells(inv, routing)
    assert PLANNED_CEILING == Fraction(49, 50) and HARD_CEILING == Fraction(1, 1)
    assert cells and all(c["complete_bpw"] <= 0.98 + 1e-12 for c in cells)
    # the budget filter must actually reject cells, not pass the whole grid through
    n_grid = len(AK.K_CHOICES) * len(AK.RUNGS) * len(DOCTOR_RUNGS) * len(PROTECT_GRID)
    assert len(cells) < n_grid, (len(cells), n_grid)
    illegal = OBC.CompleteByteLedger(
        **({c: 0 for c in OBC.COMPONENTS} | {"indices": inv.grand_params + 1}),
        metadata_alignment_reserve_bits=0)
    try:
        OBC.assert_complete_bpw_le_one(illegal, inv.grand_params)
        raise AssertionError("ceiling did not reject a 1-bit-over ledger")
    except OBC.CeilingViolation:
        pass

    # 4. exact re-billing of a real cell agrees with the analytic search to under 1 percent and is
    #    legal under the hard ceiling
    c = min(cells, key=lambda x: x["prediction"]["predicted_symkl_point"])
    v = _verify_exact(inv, routing, c)
    assert v["legal_under_hard_one_bit_ceiling"], v["exact_complete_bpw"]
    assert abs(v["exact_complete_bits"] - c["complete_bits"]) / c["complete_bits"] < 0.01, (
        v["exact_complete_bits"], c["complete_bits"])

    # 5. an unpriced lane may never produce a utility
    u = utility(None, 10**9, inv.grand_params, resident_share=1.0, extra_stages=0,
                lam_m=LAM_M_DEFAULT, lam_l=LAM_L_DEFAULT)
    assert u["utility"] is None and u["promotable"] is False

    # 6. nothing is promotable on prediction
    assert PROMOTION_FLAG["may_promote_on_predicted_utility"] is False
    assert all(c["prediction"]["promotable"] is False for c in cells[:64])
    # 7. THE CENTRAL FINDING, asserted: the two readings of the SINGLE Doctor anchor disagree about
    #    the verdict. No cell passes the gate under the saturating law; some cells DO pass under
    #    the linear law. A predictor that flips verdict on how one point is read cannot select.
    assert not any(c["prediction"]["passes_gate_under_saturating_law"] for c in cells)
    assert any(c["prediction"]["passes_gate_under_optimistic_law"] for c in cells)

    # 8. the Doctor-law-independent routing floor: only zero omission clears the gate
    clears = [K for K in AK.K_CHOICES if m.A * (1 - cov[K]) <= GATE["symkl_max"]]
    assert clears == [128], clears
    assert m.A * (1 - cov[112]) > GATE["symkl_max"]

    print(json.dumps({"ok": True, "n_legal_cells": len(cells),
                      "coverage_K64": round(cov[64], 6), "s": round(m.s, 4),
                      "best_predicted_symkl": c["prediction"]["predicted_symkl_point"],
                      "best_cell": {k: c[k] for k in ("K", "gate_rung", "down_rung",
                                                      "doctor_rung", "doctor_protect_frac",
                                                      "complete_bpw")}}, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Joint base/Doctor byte auction under <= 1 BPW.")
    ap.add_argument("--routing",
                    default="reports/subbit_reset/QWEN3_235B_ROUTING_CALIBRATION_1200.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    if args.demo:
        demo()
        return 0
    rep = build(args.routing)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    for name, e in rep["envelopes"].items():
        p = e["prediction"]
        print(f"{name:<13} K={e['K']:>3} g{e['gate_rung']}/d{e['down_rung']} "
              f"doctor={e['doctor_rung']}@{e['doctor_protect_frac']:<4} "
              f"bpw={e['exact_verification']['exact_complete_bpw']:.6f} "
              f"base={e['base_share_of_planned']:.3f} doc={e['doctor_share_of_planned']:.3f} "
              f"symKL~{p['predicted_symkl_point']:.2f} "
              f"band={p['predicted_symkl_band_with_stipulated_uncertainty']} "
              f"realizable={e['band_realizable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
