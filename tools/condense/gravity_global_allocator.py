#!/usr/bin/env python3.12
"""Gravity global byte-allocation upgrade (Part III of the Gravity campaign).

A uniform bits-per-weight across every tensor is NOT optimal. The whole-model quality of a
sub-bit representation is a function of WHERE the bits go, not just how many there are. The
correct objective is marginal: spend the next byte on the organ whose measured functional
recovery per bit is greatest, and stop protecting an organ once its marginal recovery per bit
falls below some other organ's.

This module turns that principle into a concrete allocator:

  * ORGAN taxonomy       - the model is decomposed into functional organs (embeddings, the
                           output/lm-head projection, per-role attention matrices, the router,
                           the two MoE matmuls, norms, biases/scales, and three expert-frequency
                           tiers). Each organ carries its own param_count, source precision,
                           sensitivity, activation statistic, router/expert frequency, the
                           representation lineages it may use, whether Doctor can reach it, a
                           minimum protection floor, and a per-organ quality-vs-rate curve.

  * MARGINAL UTILITY     - U_o = dQ_o / dB_o, the whole-model quality gain per EXTRA BIT for
                           organ o at its current allocated rate, read straight off the organ's
                           sensitivity curve (an exact secant slope between bracketed rates).

  * GREEDY ALLOCATOR     - given the per-organ curves, the exact-rational candidate rate brackets
                           (fractions.Fraction: 1/4 1/3 2/5 1/2 3/5 2/3 7/10 3/4 4/5 7/8 9/10 1/1),
                           and a total byte budget B_target, repeatedly raise the organ with the
                           highest U_o by one bracket step, skipping any raise that would break the
                           budget, until no affordable improving step remains. This is the classic
                           marginal-allocation (water-filling) algorithm; on CONCAVE separable
                           utilities it weakly dominates any other bracket allocation within the
                           same budget, uniform-BPW included.

Evidence-derived organ priors (from the live 120B Gravity work, gravity_frontier_g2/g3 + Forge),
labelled evidence-derived, NOT proven-optimal:

  * mlp1 / up-gate   is relatively ROBUST to full-rank PQ (pq_doctor_lowrank / product_quant were
                     the G2 layer-0 winners) -> low sensitivity, low protection floor, cheap.
  * mlp2 / down      is SENSITIVE (pq_protected_islands + a fused Doctor residual were needed to
                     hold cosine) -> higher sensitivity, higher floor, Doctor-reserved.
  * router + attn    are held PROTECTED / source-native until parent-bound (end-to-end capability)
                     evidence exists to demote them; encoded here as high protection floors so the
                     allocator does not spend sub-bit budget on them by default.
  * norms + biases   are kept source-native (protection floor at source precision).

None of this authorizes a Gravity Escape Receipt or an Event Horizon seal. The quality curves are
PROJECTED (calibrated priors), explicitly not measured capability parity. The only thing proven
here is the accounting: sum(bytes) <= B_target, exactly, by construction.

House style: plain hyphens and commas only. Byte accounting is exact-rational (fractions.Fraction),
rounded up to whole bytes at the boundary, with a non-zero fixed metadata charge per organ so no
organ can hide structural overhead in "free" metadata.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Optional

ALLOCATOR_SCHEMA = "hawking.gravity.global_allocator.v1"

# Exact-rational candidate rate brackets. A "rate" is the FRACTION of the organ's source precision
# that the packed representation retains (1/1 == source-native, 1/4 == a quarter of the source
# bits). Sub-bit lives in the low brackets when the source is already MXFP4 (4.25 bpw).
CANDIDATE_RATES: tuple[Fraction, ...] = (
    Fraction(1, 4), Fraction(1, 3), Fraction(2, 5), Fraction(1, 2), Fraction(3, 5),
    Fraction(2, 3), Fraction(7, 10), Fraction(3, 4), Fraction(4, 5), Fraction(7, 8),
    Fraction(9, 10), Fraction(1, 1),
)

# Fixed per-organ metadata charge (shape, dtype tags, config, alignment slack). Deliberately
# non-zero and identical to the Forge byte-ledger convention so overhead is never free.
_METADATA_BYTES = 64

# Source precisions expressed as exact bits-per-weight.
_SOURCE_BPW = {
    "mxfp4": Fraction(17, 4),   # 4.25 bpw, the GPT-OSS-120B expert storage precision
    "bf16": Fraction(16),
    "fp16": Fraction(16),
    "int8": Fraction(8),
}

# Representation lineage names, aligned with gravity_forge / gravity_frontier_g3.
REP_SOURCE_NATIVE = "source_native"
REP_PRODUCT_QUANT = "product_quant"
REP_TRANSFORM_PQ = "transform_pq"
REP_PQ_DOCTOR_LOWRANK = "pq_doctor_lowrank"
REP_PQ_PROTECTED_ISLANDS = "pq_protected_islands"
REP_NAIVE_RVQ = "naive_rvq"
REP_SHARED_GRAMMAR = "shared_expert_grammar"


def source_bpw(name: str) -> Fraction:
    key = name.lower()
    if key not in _SOURCE_BPW:
        raise KeyError(f"unknown source precision {name!r}; known: {sorted(_SOURCE_BPW)}")
    return _SOURCE_BPW[key]


def _beta(sensitivity: float) -> float:
    """Curvature of the quality-vs-rate curve as a function of sensitivity.

    Robust organs (low sensitivity) recover quality fast as rate rises -> large beta.
    Sensitive organs (high sensitivity) recover slowly -> small beta. Clamped at >= 1.0 so the
    curve is always weakly CONCAVE in the retained fraction (and therefore concave in bits, since
    bits are linear in the rate). Concavity is what makes the greedy marginal allocator optimal and
    lets it weakly dominate a uniform-BPW baseline.
    """
    return max(1.0, 0.5 + 4.0 * (1.0 - sensitivity))


def _q_floor(sensitivity: float, doctor_engaged: bool) -> float:
    """Projected quality retained at the lowest rate. Robust organs keep more; a reachable Doctor
    lifts the floor (repairability shifts low-rate damage upward). Bounded in [0, 0.98]."""
    base = 0.05 + 0.40 * (1.0 - sensitivity)
    if doctor_engaged:
        base += 0.10
    return max(0.0, min(0.98, base))


def make_quality_curve(sensitivity: float, doctor_engaged: bool) -> Callable[[Fraction], float]:
    """Return a callable rate(Fraction) -> projected functional quality in [0, 1].

    Q(r) = 1 - (1 - q_floor) * (1 - r) ** beta. Monotone non-decreasing, concave in r, Q(1) == 1.
    This is a PROJECTION from calibrated priors, not a measured capability curve.
    """
    beta = _beta(sensitivity)
    q0 = _q_floor(sensitivity, doctor_engaged)

    def curve(rate: Fraction) -> float:
        r = float(rate)
        r = max(0.0, min(1.0, r))
        return 1.0 - (1.0 - q0) * ((1.0 - r) ** beta)

    return curve


@dataclass
class Organ:
    """One functional organ of the model and everything the allocator needs to price it.

    param_count             number of scalar weights in this organ.
    source_precision        name of the source dtype ("mxfp4", "bf16", ...).
    sensitivity             0..1, how fast functional quality degrades as bits are removed.
    activation_stat         0..1, a proxy for how much this organ drives the output on live tokens.
    router_or_expert_frequency  0..1, fraction of tokens that exercise this organ (1.0 for dense
                            organs, the MoE routing frequency for expert tiers).
    representation_options  ordered list of lineage names this organ may be packed with.
    doctor_reachable        whether a fused Doctor residual can repair this organ.
    min_protection_bpw      hard floor on effective bits-per-weight; the allocator never drops below
                            the smallest bracket that meets it.
    quality_curve           callable rate -> projected quality (built from the priors if omitted).
    doctor_bpw              constant Doctor reserve billed across all rates when Doctor is engaged.
    importance_weight       functional weight of this organ in the whole-model quality aggregate.
    """
    name: str
    param_count: int
    source_precision: str
    sensitivity: float
    activation_stat: float
    router_or_expert_frequency: float
    representation_options: list[str]
    doctor_reachable: bool
    min_protection_bpw: Fraction
    quality_curve: Optional[Callable[[Fraction], float]] = None
    doctor_bpw: Fraction = Fraction(0)
    importance_weight: float = 1.0
    curve_points: dict[Fraction, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.min_protection_bpw = Fraction(self.min_protection_bpw)
        self.doctor_bpw = Fraction(self.doctor_bpw)
        if self.quality_curve is None:
            self.quality_curve = make_quality_curve(self.sensitivity, self.doctor_engaged)
        if not self.curve_points:
            self.curve_points = {r: self.quality_curve(r) for r in CANDIDATE_RATES}

    # -- pricing ---------------------------------------------------------------------------------

    @property
    def source_bpw(self) -> Fraction:
        return source_bpw(self.source_precision)

    @property
    def doctor_engaged(self) -> bool:
        return self.doctor_reachable and self.doctor_bpw > 0

    @property
    def min_rate(self) -> Fraction:
        """Smallest candidate bracket whose effective bpw meets min_protection_bpw."""
        need = Fraction(self.min_protection_bpw)
        for r in CANDIDATE_RATES:
            if r * self.source_bpw + self.doctor_bpw >= need:
                return r
        return CANDIDATE_RATES[-1]

    @property
    def pinned_native(self) -> bool:
        """True when the protection floor already forces the top bracket (nothing to allocate)."""
        return self.min_rate >= CANDIDATE_RATES[-1]

    def effective_bpw(self, rate: Fraction) -> Fraction:
        # rate == 1/1 means keep the source precision exactly (true source-native, no Doctor
        # reserve). Below source-native a sub-bit representation carries a constant Doctor reserve
        # when Doctor is reachable. This keeps bytes strictly increasing in rate (the last step's
        # rate gain, source_bpw / bracket, exceeds any doctor_bpw for the default priors).
        rate = Fraction(rate)
        if rate >= CANDIDATE_RATES[-1]:
            return self.source_bpw
        return rate * self.source_bpw + self.doctor_bpw

    def bytes_at(self, rate: Fraction) -> int:
        """Exact whole-artifact bytes for this organ at the given rate: body bits (rounded up to a
        whole byte) plus the fixed metadata charge. Strictly increasing in rate."""
        body_bits = self.effective_bpw(rate) * self.param_count
        body_bytes = math.ceil(body_bits / 8)
        return int(body_bytes) + _METADATA_BYTES

    def quality_at(self, rate: Fraction) -> float:
        return float(self.quality_curve(rate))

    def representation_at(self, rate: Fraction) -> str:
        """Deterministic lineage pick for a chosen rate, from this organ's option list."""
        opts = self.representation_options
        if rate >= CANDIDATE_RATES[-1]:
            # full rate keeps the source precision for every organ, packed or not
            return REP_SOURCE_NATIVE
        for pref in (REP_PQ_DOCTOR_LOWRANK if self.doctor_engaged else None,
                     REP_PQ_PROTECTED_ISLANDS, REP_SHARED_GRAMMAR,
                     REP_TRANSFORM_PQ, REP_PRODUCT_QUANT, REP_NAIVE_RVQ):
            if pref and pref in opts:
                return pref
        return opts[0] if opts else REP_SOURCE_NATIVE


# ------------------------------------------------------------------------------------------------
# Marginal utility
# ------------------------------------------------------------------------------------------------

def marginal_utility(organ: Organ, cur_rate: Fraction, next_rate: Fraction,
                     total_weight: float) -> float:
    """U_o = dQ_o / dB_o: whole-model projected quality gain per EXTRA BIT for organ o when its
    rate is raised from cur_rate to next_rate.

    dQ_o is the organ's contribution to the (weight-normalized) whole-model quality; dB_o is the
    incremental bits. Because the per-organ curve is concave in bits, U_o is non-increasing as the
    rate climbs, so the greedy that always takes the highest available U_o is the marginal
    allocation algorithm.
    """
    dbits = (organ.bytes_at(next_rate) - organ.bytes_at(cur_rate)) * 8
    if dbits <= 0:
        return 0.0
    dq = (organ.quality_at(next_rate) - organ.quality_at(cur_rate))
    dq_whole = (organ.importance_weight / total_weight) * dq
    return dq_whole / dbits


# ------------------------------------------------------------------------------------------------
# Allocation result
# ------------------------------------------------------------------------------------------------

@dataclass
class OrganAllocation:
    name: str
    representation: str
    rate: Fraction
    doctor: bool
    protection_bpw: Fraction
    effective_bpw: Fraction
    bytes: int
    quality: float
    pinned_native: bool


@dataclass
class Allocation:
    organs: list[OrganAllocation]
    total_bytes: int
    budget_bytes: int
    projected_quality: float
    proof: dict
    strategy: str

    def by_name(self, name: str) -> OrganAllocation:
        for o in self.organs:
            if o.name == name:
                return o
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {
            "schema": ALLOCATOR_SCHEMA,
            "strategy": self.strategy,
            "budget_bytes": self.budget_bytes,
            "total_bytes": self.total_bytes,
            "projected_quality": round(self.projected_quality, 6),
            "proof": self.proof,
            "organs": [
                {
                    "name": o.name,
                    "representation": o.representation,
                    "rate": str(o.rate),
                    "doctor": o.doctor,
                    "protection_bpw": str(o.protection_bpw),
                    "effective_bpw": round(float(o.effective_bpw), 4),
                    "bytes": o.bytes,
                    "quality": round(o.quality, 6),
                    "pinned_native": o.pinned_native,
                }
                for o in self.organs
            ],
        }


def _whole_quality(organs: list[Organ], rates: dict[str, Fraction]) -> float:
    total_w = sum(o.importance_weight for o in organs)
    if total_w <= 0:
        return 0.0
    acc = sum(o.importance_weight * o.quality_at(rates[o.name]) for o in organs)
    return acc / total_w


def _build_allocation(organs: list[Organ], rates: dict[str, Fraction], budget: int,
                      strategy: str) -> Allocation:
    rows: list[OrganAllocation] = []
    total = 0
    for o in organs:
        r = rates[o.name]
        b = o.bytes_at(r)
        total += b
        rows.append(OrganAllocation(
            name=o.name,
            representation=o.representation_at(r),
            rate=r,
            doctor=o.doctor_engaged and r < CANDIDATE_RATES[-1],
            protection_bpw=o.min_protection_bpw,
            effective_bpw=o.effective_bpw(r),
            bytes=b,
            quality=o.quality_at(r),
            pinned_native=o.pinned_native,
        ))
    proof = {
        "sum_bytes": total,
        "budget_bytes": budget,
        "within_budget": total <= budget,
        "slack_bytes": budget - total,
        "argument": (
            "each rate-raise is applied only when its incremental bytes keep the running total "
            "at or below B_target; by induction sum(bytes) <= B_target holds at every step and at "
            "return. bytes_at is exact-rational body bits rounded up to whole bytes plus a fixed "
            "metadata charge, so no byte is uncounted."
        ),
    }
    return Allocation(
        organs=rows,
        total_bytes=total,
        budget_bytes=budget,
        projected_quality=_whole_quality(organs, rates),
        proof=proof,
        strategy=strategy,
    )


def floor_rates(organs: list[Organ]) -> dict[str, Fraction]:
    return {o.name: o.min_rate for o in organs}


def floor_bytes(organs: list[Organ]) -> int:
    return sum(o.bytes_at(o.min_rate) for o in organs)


# ------------------------------------------------------------------------------------------------
# Greedy global allocator
# ------------------------------------------------------------------------------------------------

def greedy_allocate(organs: list[Organ], budget_bytes: int,
                    rates: tuple[Fraction, ...] = CANDIDATE_RATES) -> Allocation:
    """Marginal-allocation greedy. Start every organ at its protection floor, then repeatedly raise
    the organ with the highest marginal utility U_o whose next bracket still fits the budget, until
    no affordable improving step remains or all organs are maxed.

    Returns per-organ (representation, rate, doctor, protection, bytes), total bytes, projected
    whole-model quality, and a proof that sum(bytes) <= B_target.
    """
    rates = tuple(rates)
    total_w = sum(o.importance_weight for o in organs) or 1.0

    # index of the current bracket per organ, seeded at the protection floor
    idx: dict[str, int] = {}
    for o in organs:
        floor = o.min_rate
        idx[o.name] = next(i for i, r in enumerate(rates) if r >= floor)

    cur_bytes = sum(o.bytes_at(rates[idx[o.name]]) for o in organs)
    if cur_bytes > budget_bytes:
        raise ValueError(
            f"budget {budget_bytes} bytes is below the protection floor {cur_bytes} bytes; "
            f"cannot satisfy min_protection_bpw for all organs"
        )

    by_name = {o.name: o for o in organs}
    while True:
        best_name: Optional[str] = None
        best_u = 0.0
        best_step_bytes = 0
        for o in organs:
            i = idx[o.name]
            if i >= len(rates) - 1:
                continue  # already at 1/1
            cur_r, nxt_r = rates[i], rates[i + 1]
            step_bytes = o.bytes_at(nxt_r) - o.bytes_at(cur_r)
            if cur_bytes + step_bytes > budget_bytes:
                continue  # unaffordable right now
            u = marginal_utility(o, cur_r, nxt_r, total_w)
            if u > best_u:
                best_u, best_name, best_step_bytes = u, o.name, step_bytes
        if best_name is None:
            break
        idx[best_name] += 1
        cur_bytes += best_step_bytes

    chosen = {name: rates[i] for name, i in idx.items()}
    return _build_allocation([by_name[n] for n in [o.name for o in organs]], chosen,
                             budget_bytes, strategy="greedy_marginal")


def uniform_allocate(organs: list[Organ], budget_bytes: int,
                     rates: tuple[Fraction, ...] = CANDIDATE_RATES) -> Allocation:
    """Uniform-BPW baseline: assign the SAME retained fraction r_u to every organ (clamped up to
    each organ's protection floor), choosing the largest bracket r_u whose total fits the budget.
    This is the allocation the greedy must weakly dominate at equal budget."""
    rates = tuple(rates)
    floor = floor_rates(organs)

    def total_for(r_u: Fraction) -> int:
        return sum(o.bytes_at(max(r_u, floor[o.name])) for o in organs)

    if total_for(rates[0]) > budget_bytes:
        raise ValueError(
            f"budget {budget_bytes} bytes is below the protection floor "
            f"{total_for(rates[0])} bytes; uniform baseline infeasible"
        )
    chosen_r = rates[0]
    for r_u in rates:
        if total_for(r_u) <= budget_bytes:
            chosen_r = r_u
        else:
            break
    chosen = {o.name: max(chosen_r, floor[o.name]) for o in organs}
    return _build_allocation(organs, chosen, budget_bytes, strategy="uniform_bpw")


# ------------------------------------------------------------------------------------------------
# Concentration report (evidence that bytes land on high-utility organs)
# ------------------------------------------------------------------------------------------------

def concentration_report(organs: list[Organ], alloc: Allocation) -> dict:
    """Split organs by their floor-level marginal utility (utility of the FIRST extra bit) and
    measure how the discretionary bytes (bytes above each organ's protection floor) distribute.
    A concentrated allocation puts most discretionary bytes on the high-utility half."""
    total_w = sum(o.importance_weight for o in organs) or 1.0
    by_name = {o.name: o for o in organs}

    util0 = {}
    for o in organs:
        i0 = next(i for i, r in enumerate(CANDIDATE_RATES) if r >= o.min_rate)
        if i0 >= len(CANDIDATE_RATES) - 1:
            util0[o.name] = 0.0
        else:
            util0[o.name] = marginal_utility(o, CANDIDATE_RATES[i0], CANDIDATE_RATES[i0 + 1], total_w)

    disc = {}
    for row in alloc.organs:
        o = by_name[row.name]
        disc[row.name] = max(0, row.bytes - o.bytes_at(o.min_rate))

    ranked = sorted(organs, key=lambda o: util0[o.name], reverse=True)
    half = len(ranked) // 2 or 1
    top = ranked[:half]
    bottom = ranked[half:]
    top_disc = sum(disc[o.name] for o in top)
    bottom_disc = sum(disc[o.name] for o in bottom)
    total_disc = top_disc + bottom_disc
    return {
        "total_discretionary_bytes": total_disc,
        "top_half_organs": [o.name for o in top],
        "top_half_discretionary_bytes": top_disc,
        "bottom_half_discretionary_bytes": bottom_disc,
        "top_half_share": (top_disc / total_disc) if total_disc else 0.0,
        "floor_marginal_utility": {n: util0[n] for n in sorted(util0)},
        "discretionary_bytes": disc,
    }


# ------------------------------------------------------------------------------------------------
# Default organ priors (evidence-derived, GPT-OSS-120B-like), scalable synthetic instance
# ------------------------------------------------------------------------------------------------

def default_gptoss_organs(scale: int = 1) -> list[Organ]:
    """A small synthetic GPT-OSS-like organ set carrying the DEFAULT evidence-derived priors.

    param_counts are deliberately tiny (scaled by `scale`) so the demo and tests run in
    milliseconds. The RELATIVE priors (sensitivity, floors, frequencies, Doctor reach) mirror the
    live 120B evidence; the absolute magnitudes do not. This is not the real model and never reads
    real weights.
    """
    s = max(1, int(scale))

    def mk(name, params, prec, sens, act, freq, opts, doctor, floor_bpw, dbpw, weight):
        return Organ(
            name=name,
            param_count=params * s,
            source_precision=prec,
            sensitivity=sens,
            activation_stat=act,
            router_or_expert_frequency=freq,
            representation_options=opts,
            doctor_reachable=doctor,
            min_protection_bpw=Fraction(floor_bpw),
            doctor_bpw=Fraction(dbpw),
            importance_weight=weight,
        )

    dense = [REP_SOURCE_NATIVE, REP_TRANSFORM_PQ, REP_PRODUCT_QUANT]
    return [
        # -- dense, mostly non-expert organs (bf16 source) ---------------------------------------
        mk("embeddings", 40000, "bf16", 0.55, 0.60, 1.0,
           [REP_PRODUCT_QUANT, REP_TRANSFORM_PQ, REP_SOURCE_NATIVE], True,
           "3", "0", weight=1.2),
        mk("lm_head_output_proj", 40000, "bf16", 0.75, 0.85, 1.0,
           [REP_TRANSFORM_PQ, REP_PQ_DOCTOR_LOWRANK, REP_SOURCE_NATIVE], True,
           "4", Fraction(1, 5), weight=2.4),
        # attention: PROTECTED / source-native until parent-bound evidence -> high floors
        mk("attn_q", 12000, "bf16", 0.70, 0.55, 1.0, dense, False, "16", "0", weight=1.4),
        mk("attn_k", 12000, "bf16", 0.80, 0.55, 1.0, dense, False, "16", "0", weight=1.6),
        mk("attn_v", 12000, "bf16", 0.65, 0.60, 1.0, dense, False, "16", "0", weight=1.3),
        mk("attn_o", 12000, "bf16", 0.70, 0.60, 1.0, dense, False, "16", "0", weight=1.4),
        # router: MOST sensitive, protected source-native until end-to-end evidence
        mk("router", 4000, "bf16", 0.95, 0.90, 1.0, [REP_SOURCE_NATIVE], False,
           "16", "0", weight=3.0),
        # norms and biases/scales: kept source-native
        mk("norms", 2000, "bf16", 0.90, 0.70, 1.0, [REP_SOURCE_NATIVE], False,
           "16", "0", weight=1.0),
        mk("biases_scales", 2000, "bf16", 0.85, 0.50, 1.0, [REP_SOURCE_NATIVE], False,
           "16", "0", weight=0.6),
        # -- MoE matmuls (mxfp4 source) ----------------------------------------------------------
        # mlp1 up-gate: ROBUST to full-rank PQ (G2 layer-0 winner pq_doctor_lowrank/product_quant)
        mk("mlp1_up_gate", 120000, "mxfp4", 0.30, 0.75, 0.55,
           [REP_PQ_DOCTOR_LOWRANK, REP_PRODUCT_QUANT, REP_TRANSFORM_PQ], True,
           Fraction(1, 2), Fraction(1, 10), weight=1.8),
        # mlp2 down: SENSITIVE (needs pq_protected_islands + fused Doctor)
        mk("mlp2_down", 120000, "mxfp4", 0.62, 0.80, 0.55,
           [REP_PQ_PROTECTED_ISLANDS, REP_PQ_DOCTOR_LOWRANK], True,
           "1", Fraction(3, 20), weight=2.2),
        # -- expert-frequency tiers (mxfp4 source) ----------------------------------------------
        mk("shared_experts", 30000, "mxfp4", 0.50, 0.85, 1.0,
           [REP_SHARED_GRAMMAR, REP_PQ_DOCTOR_LOWRANK, REP_PRODUCT_QUANT], True,
           Fraction(3, 2), Fraction(1, 10), weight=1.7),
        mk("frequent_experts", 200000, "mxfp4", 0.40, 0.70, 0.60,
           [REP_PQ_DOCTOR_LOWRANK, REP_PRODUCT_QUANT, REP_TRANSFORM_PQ], True,
           Fraction(3, 4), Fraction(1, 10), weight=1.5),
        mk("rare_experts", 400000, "mxfp4", 0.35, 0.30, 0.05,
           [REP_PRODUCT_QUANT, REP_TRANSFORM_PQ], True,
           Fraction(1, 4), "0", weight=0.4),
        # -- fixed runtime metadata (pinned, never allocated) ------------------------------------
        mk("runtime_metadata", 256, "int8", 1.0, 0.10, 1.0, [REP_SOURCE_NATIVE], False,
           "8", "0", weight=0.2),
    ]


# ------------------------------------------------------------------------------------------------
# CLI / demo
# ------------------------------------------------------------------------------------------------

def _fmt_alloc_table(alloc: Allocation) -> str:
    lines = []
    header = f"{'organ':<20}{'rep':<22}{'rate':>6}{'eff_bpw':>9}{'doctor':>8}{'bytes':>10}{'quality':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for o in sorted(alloc.organs, key=lambda r: -r.bytes):
        lines.append(
            f"{o.name:<20}{o.representation:<22}{str(o.rate):>6}"
            f"{float(o.effective_bpw):>9.3f}{('yes' if o.doctor else 'no'):>8}"
            f"{o.bytes:>10}{o.quality:>9.4f}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<20}{'':<22}{'':>6}{'':>9}{'':>8}"
        f"{alloc.total_bytes:>10}{alloc.projected_quality:>9.4f}"
    )
    return "\n".join(lines)


def run_demo() -> int:
    organs = default_gptoss_organs(scale=1)
    fb = floor_bytes(organs)
    # source-native reference size (every organ at 1/1)
    native_bytes = sum(o.bytes_at(CANDIDATE_RATES[-1]) for o in organs)
    print(f"# Gravity global byte-allocation demo  ({ALLOCATOR_SCHEMA})")
    print(f"# synthetic GPT-OSS-like organ set: {len(organs)} organs")
    print(f"# protection-floor bytes = {fb}   source-native bytes = {native_bytes}")
    print()

    # A few budgets between the floor and native.
    span = native_bytes - fb
    budgets = [fb + int(span * f) for f in (0.15, 0.35, 0.60)]

    all_ok = True
    for budget in budgets:
        greedy = greedy_allocate(organs, budget)
        uniform = uniform_allocate(organs, budget)
        conc = concentration_report(organs, greedy)

        print(f"=== budget = {budget} bytes "
              f"({100.0 * (budget - fb) / span:.0f}% of floor->native span) ===")
        print(_fmt_alloc_table(greedy))
        print()
        print(f"  greedy quality  = {greedy.projected_quality:.5f}  "
              f"(total {greedy.total_bytes} <= {budget})")
        print(f"  uniform quality = {uniform.projected_quality:.5f}  "
              f"(uniform total {uniform.total_bytes})")
        print(f"  concentration: top-half organs {conc['top_half_organs']}")
        print(f"                 hold {conc['top_half_share'] * 100:.1f}% of "
              f"{conc['total_discretionary_bytes']} discretionary bytes")

        # invariant (a): budget respected
        a = greedy.total_bytes <= budget
        # invariant (b): bytes concentrate on high-utility organs
        b = conc["total_discretionary_bytes"] == 0 or conc["top_half_share"] >= 0.5
        # invariant (c): greedy weakly dominates uniform at equal budget
        c = greedy.projected_quality >= uniform.projected_quality - 1e-9
        # doctrine spot-checks derived from the evidence priors
        mlp1 = greedy.by_name("mlp1_up_gate").rate
        mlp2 = greedy.by_name("mlp2_down").rate
        rare = greedy.by_name("rare_experts").rate
        freq = greedy.by_name("frequent_experts").rate
        d = mlp2 >= mlp1 and freq >= rare
        print(f"  invariants: budget<=B {a}  concentrated {b}  "
              f"greedy>=uniform {c}  (mlp2>=mlp1 & freq>=rare) {d}")
        assert a, "budget invariant violated"
        assert b, "concentration invariant violated"
        assert c, "greedy did not weakly dominate uniform"
        all_ok = all_ok and a and b and c and d
        print()

    print("DEMO_OK" if all_ok else "DEMO_INCOMPLETE")
    return 0 if all_ok else 1


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Gravity global byte-allocation upgrade")
    ap.add_argument("--demo", action="store_true",
                    help="print greedy allocations for a synthetic GPT-OSS-like organ set")
    ap.add_argument("--json", action="store_true",
                    help="emit the greedy allocation at the mid budget as JSON")
    args = ap.parse_args(argv)

    if args.json:
        organs = default_gptoss_organs(scale=1)
        fb = floor_bytes(organs)
        native = sum(o.bytes_at(CANDIDATE_RATES[-1]) for o in organs)
        budget = fb + (native - fb) // 2
        alloc = greedy_allocate(organs, budget)
        print(json.dumps(alloc.to_dict(), indent=2))
        return 0

    if args.demo:
        return run_demo()

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
