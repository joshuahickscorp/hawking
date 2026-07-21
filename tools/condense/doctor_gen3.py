#!/usr/bin/env python3.12
"""Doctor Generation 3: the closed-loop degradation-healing system.

WHAT CHANGED FROM GEN 1/2. Gen1 was one mechanism - a sparse residual over the rows with the worst
relative residual energy. It worked (on the S1 full forward it beat the untreated arm on all four
metrics at +0.0513 complete BPW), but "a residual" is not a medical system. Gen3 is the controller
that decides WHICH degradation is happening, WHICH treatments can reach it, what each costs in
exact bits, and whether any of them is worth its slot against every competing use of the budget.

    OBSERVE -> LOCALIZE -> ATTRIBUTE -> PROPOSE -> PRICE -> TREAT -> REPACK -> RE-EVALUATE
            -> RETAIN / ROLLBACK / ESCALATE

THE ONE RULE THAT DOES NOT BEND. Doctor gets no free bytes. Its capability may grow without limit
but every installed treatment competes inside complete_bits / original_weight_count <= 1/1, and a
conditional treatment is billed on INSTALLED bytes even if it fires on one token in a thousand.
`installed_bits` and `active_bits_per_token` are separate fields precisely so a dynamic treatment
cannot quietly bill itself at its average.

WHAT MAKES THIS HONEST RATHER THAN A PLANNING DOCUMENT. Every prior in TREATMENTS carries the
measurement it came from and a provenance tag. Three of them are NEGATIVE and are here so the
controller cannot re-propose them:

  * expert_merge          DEAD. Omitted experts are not reconstructible from survivors (best single
                          survivor median held-out rel error 0.885/0.993/0.995; 4-survivor
                          least-squares 0.863/0.988/0.995 - the trivial zero predictor).
  * router_distill        NEGATIVE. No trained student beats plain masking on held-out ids
                          (masked 0.0784 vs bias 0.0993, low-rank 0.1176, full retune 0.0927), and
                          the KL term is provably inert. An earlier "57.3 pct reachable headroom"
                          figure is WITHDRAWN: the oracle was scored on a biased-easy 16-token
                          head subset against a baseline measured over all 96.
  * low_rank_residual     DEAD as a default (docs/dead_levers.md): quantization error is high rank.
                          Retained ONLY as a diagnosis-gated option when the residual is measured
                          low rank, which it has not been on this parent.

A treatment with a negative prior may only be proposed again under the no-repetition law in
`may_propose`: a materially different diagnosis, an architecture that changes reachability, a
different rate regime, or an implementation that removes the prior blocker.

NOTHING HERE IS A CAPABILITY CLAIM. Predicted utility orders candidates; it never promotes one.
Promotion requires a real parent-vs-packed forward and the frozen quality contract.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "foundry")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "hawking.doctor.gen3.v1"
CEILING = Fraction(1, 1)


# ── diagnosis ontology ────────────────────────────────────────────────────────────────────────
# Every cause a compression/pruning/routing intervention can produce. A diagnosis is a set of
# these with measured weights, never a single guess.
DIAGNOSES: dict[str, str] = {
    "quantization_reconstruction": "the coded weights do not reconstruct the parent tensor",
    "scale_magnitude_distortion": "row/column magnitude is wrong even where direction is right",
    "direction_distortion": "the unit direction of rows is wrong",
    "routing_omission": "the token should have been routed to an expert the artifact does not have",
    "router_redistribution": "the surviving router assigns the wrong weights among survivors",
    "expert_substitution": "a surviving expert stands in for an omitted one and does not match",
    "activation_shift": "the organ input distribution moved, so the fit is off-distribution",
    "hidden_state_drift": "post-layer hidden state diverges and accumulates down the stack",
    "residual_accumulation": "small per-layer errors compound in the residual stream",
    "normalization_mismatch": "RMSNorm statistics differ between parent and student",
    "logit_ranking_collapse": "logit ordering is lost even where the vector is correlated",
    "rare_token_collapse": "low-frequency tokens are destroyed while common ones survive",
    "domain_collapse": "one capability domain fails while others hold",
    "runtime_numerical": "the failure is a kernel/dtype/precision bug, not a representation loss",
    "mixed_interaction": "two or more causes interact and neither alone explains the loss",
}

# The escalation ladder. Cheapest diagnosis-matched treatment first; escalate only when the lower
# rung cannot reach the gate at competitive utility.
LADDER = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]
LADDER_NAMES = {
    "E0": "scale / bias / routing calibration",
    "E1": "protected rows, columns, channels, islands",
    "E2": "sparse or structured residual",
    "E3": "residual / additive codebook",
    "E4": "router / expert function-space distillation",
    "E5": "layerwise QAT",
    "E6": "structural expert or topology change",
    "E7": "generated / tied parameter studentization",
}


@dataclass
class Treatment:
    """One reachable intervention, with the measurement it is priced from."""
    id: str
    rung: str
    space: str                       # parameter | function | routing | structural | dynamic
    treats: tuple[str, ...]          # diagnosis ids it can reach
    installed_bits_per_tensor: str   # exact formula, in words; priced by bits_fn at call time
    prior_recovery: float | None     # measured relative output-error reduction, or None if untested
    prior_bpw_cost: float | None     # measured complete-BPW cost of that recovery
    provenance: str
    status: str                      # alive | negative | dead | untested
    reopen: str = ""
    conditional: bool = False        # if True, active_bits_per_token < installed_bits

    def utility(self, lam_latency: float = 0.0, lam_memory: float = 0.0,
                d_latency: float = 0.0, d_memory: float = 0.0) -> float | None:
        """U = E[dQ] / (dB + lam_L*d_latency + lam_M*d_memory). None when untested."""
        if self.prior_recovery is None or self.prior_bpw_cost is None:
            return None
        denom = self.prior_bpw_cost + lam_latency * d_latency + lam_memory * d_memory
        if denom <= 0:
            # a zero-byte treatment is infinitely efficient per bit; rank it by recovery alone and
            # say so rather than dividing by zero
            return float("inf") if self.prior_recovery > 0 else 0.0
        return self.prior_recovery / denom


# The library. Every prior is a measurement from this campaign or a sealed dead lever.
TREATMENTS: tuple[Treatment, ...] = (
    Treatment(
        id="gamma_weighted_coding", rung="E0", space="function",
        treats=("quantization_reconstruction", "activation_shift", "direction_distortion"),
        installed_bits_per_tensor="0 - gamma already ships as a native pass-through tensor",
        prior_recovery=0.60, prior_bpw_cost=0.0,
        provenance="Lane C adversarial: data-free h = post_attention_layernorm.weight^2 recovers "
                   "83 pct of a 60 pct layer-0 output-error cut; log-corr(h, gamma^2) = 0.9918. "
                   "Gain tracks gamma anisotropy, which is 1.843 at L0 and 0.062 at L93, so the "
                   "0.60 prior is a LAYER-0 number and must be scaled by measured anisotropy.",
        status="alive"),
    Treatment(
        id="row_scale_refit", rung="E0", space="parameter",
        treats=("scale_magnitude_distortion",),
        installed_bits_per_tensor="0 - the bf16 row scale already ships, it just carries a better value",
        prior_recovery=0.028, prior_bpw_cost=0.0,
        provenance="M01_FUNCTION_AWARE_PROBE: scale-invariant VQ mean rel_error 0.7746 -> 0.7526 "
                   "over 15 real cells; largest where row-norm span is largest (L0 gate 0.7214 -> "
                   "0.6435), vanishing where flat (L70 0.7041 -> 0.7029)",
        status="alive"),
    Treatment(
        id="protected_islands", rung="E1", space="parameter",
        treats=("quantization_reconstruction", "rare_token_collapse", "domain_collapse"),
        installed_bits_per_tensor="native_bits(selected rows) + 1 bit per row selector bitmap",
        prior_recovery=None, prior_bpw_cost=None,
        provenance="untested on this parent at the S64 rates",
        status="untested"),
    Treatment(
        id="sparse_residual_doctor_gen1", rung="E2", space="parameter",
        treats=("quantization_reconstruction", "direction_distortion"),
        installed_bits_per_tensor="correction indices + amortized correction codebook + one bf16 "
                                  "scale per protected row + 1 bit per row protection bitmap",
        prior_recovery=0.105, prior_bpw_cost=0.0513,
        provenance="S1 FULL FORWARD, the only capability-fidelity Doctor measurement in the "
                   "campaign: S64_structural -> S64_doctor at +0.0513 complete BPW moved symKL "
                   "8.599 -> 7.699 (10.5 pct), argmax 0.0735 -> 0.0922, cosine 0.434 -> 0.468, "
                   "top5 0.080 -> 0.143. Paired, same six prompts, same direction on all four. "
                   "MECHANISM CORRECTED by the S3C adversary: the ROW SELECTOR is not the "
                   "mechanism. Gen1's 'worst RELATIVE residual energy' rows beat RANDOM row "
                   "selection by only +1.8 and +0.9 pct, inside the seed spread, and random rows "
                   "recover 96-99 pct of the entire gain. What earns the bytes is the SECOND "
                   "CODEBOOK STAGE (added residual capacity), not where it is pointed. Do not "
                   "transfer the selector as if it were the active ingredient.",
        status="alive"),
    Treatment(
        id="low_rank_residual", rung="E2", space="parameter",
        treats=("quantization_reconstruction",),
        installed_bits_per_tensor="rank * (rows + cols) * 16",
        prior_recovery=None, prior_bpw_cost=None,
        provenance="docs/dead_levers.md: quantization error is HIGH rank; LoRA plateaus ~step 25 "
                   "and rank-64 SVD healed TQ3 only 0.114 -> 0.104 while costing 4.6 bpw. Also "
                   "data-free low-rank residual NO-GO (top-64 SVD captures 3-9 pct of energy).",
        status="dead",
        reopen="only when the residual is MEASURED low rank on the parent in question"),
    Treatment(
        id="residual_additive_codebook", rung="E3", space="parameter",
        treats=("quantization_reconstruction", "direction_distortion"),
        installed_bits_per_tensor="extra RVQ stage: (n/dim) * ceil(log2 k) + amortized codebook",
        prior_recovery=0.137, prior_bpw_cost=0.319,
        provenance="Doctor Gen1 on real down_proj L46: rel_error 0.7038 -> 0.6074 for +0.319 bpw "
                   "on that organ. Weight-space, not capability.",
        status="alive"),
    Treatment(
        id="kronecker_repair", rung="E3", space="parameter",
        treats=("quantization_reconstruction",),
        installed_bits_per_tensor="Kronecker factors A (m1 x n1) and B (m2 x n2) at rank r, bf16",
        prior_recovery=0.866, prior_bpw_cost=-0.013,
        provenance="Lane F adversarial: on L0 gate_proj Kronecker beats the incumbent codec "
                   "0.0301 vs 0.2252 rel_error at a CHEAPER complete rate (2.487061 vs 2.500735 "
                   "BPW) - a negative byte cost. Dies with depth: L0 0.0301, L1 0.6611, L2 0.7567, "
                   "L46 0.85+. ADMISSION IS PER-CELL, never global.",
        status="alive",
        reopen="admitted per cell only when the Van Loan spectrum is concentrated AND it beats the "
               "incumbent at equal-or-fewer bits AND it wins in OUTPUT space"),
    Treatment(
        id="router_distill", rung="E4", space="routing",
        treats=("router_redistribution",),
        installed_bits_per_tensor="per-survivor logit bias (K floats) or a low-rank router correction",
        prior_recovery=0.0, prior_bpw_cost=0.0,
        provenance="S2C: no trained student beat plain masking on held-out ids. Masked 0.0784; "
                   "bias 0.0993, low-rank r8 0.1176, full survivor-row retune 0.0927 - all worse. "
                   "The KL term is provably INERT: softmax restricted to survivors IS the teacher "
                   "renormalized onto survivors, so KL = 0 exactly and any correction only raises "
                   "it. TWO CORRECTIONS from the adversary: (a) the '57.3 pct reachable headroom' "
                   "figure is WITHDRAWN - the oracle was scored on only the first 16 holdout "
                   "tokens, which are a biased-easy head subset (retained routing mass 0.969 vs "
                   "0.939 for the rest), against a masked baseline measured over all 96; (b) the "
                   "'students overfit' diagnosis is unproven - the zero-free-parameter masked arm "
                   "itself shows 0.0521 train vs 0.0784 holdout, so the split is not exchangeable "
                   "and the train/holdout delta is not a clean generalization measurement. The "
                   "NEGATIVE verdict stands: every arm was scored on the same holdout.",
        status="negative",
        reopen="a parent where the survivor-restricted oracle gap is large AND a student "
               "generalizes to held-out ids"),
    Treatment(
        id="expert_merge", rung="E4", space="routing",
        treats=("routing_omission", "expert_substitution"),
        installed_bits_per_tensor="per-(layer, omitted expert) coefficient vector over survivors",
        prior_recovery=0.0, prior_bpw_cost=None,
        provenance="Lane E, adversarially UPHELD: best single surviving expert reconstructs an "
                   "omitted expert with median held-out rel error 0.885/0.993/0.995 at L0/L3/L7; a "
                   "4-survivor least-squares merge reaches 0.863/0.988/0.995. ~1.0 is the trivial "
                   "zero predictor. The adversary's found contamination ran IN THE MERGE'S FAVOUR.",
        status="dead",
        reopen="a parent measuring best-single-survivor reconstruction error <= 0.5 held out"),
    Treatment(
        id="layerwise_qat", rung="E5", space="function",
        treats=("quantization_reconstruction", "direction_distortion", "activation_shift"),
        installed_bits_per_tensor="0 - training moves the latent weights; the artifact schema and "
                                  "byte count are unchanged (asserted byte-identical)",
        prior_recovery=None, prior_bpw_cost=0.0,
        provenance="S3A claimed 14.1/11.3/2.45 pct held-out output-error gains at layer 0 at "
                   "byte-identical cost. The adversary REFUTED the 'held-out' label: the fraction "
                   "of held-out activation energy lying inside the FIT split's row space is "
                   "1.000 / 0.991 / 0.424 / 1.000 at layer 0, and the per-expert gain is MONOTONE "
                   "in that overlap (19.8 / 11.6 / 4.0 / 18.0 pct). That is fit-span descent, not "
                   "generalization. The layer-0-wins / layer-46-loses contrast is CONFOUNDED: "
                   "layer 46's proxy has overlap 0.007-0.023, i.e. a genuinely disjoint probe, so "
                   "the contrast measures probe rank and not depth. The ZERO-BIT property is the "
                   "one solid claim and survives: both arms serialize byte-identically.",
        status="untested",
        reopen="score on a split whose held-out energy capture inside the fit span is below ~0.1 "
               "(at layer 0 that needs far more than 1400 calibration tokens, or scoring against "
               "the full corpus rather than each expert's own routed tokens), and vary the split "
               "seed so 'wins under every seed' actually varies the evaluation set"),
    Treatment(
        id="structural_adaptive_k", rung="E6", space="structural",
        treats=("routing_omission", "quantization_reconstruction"),
        installed_bits_per_tensor="survivor bitmap (n_experts bits per layer) + the rate the freed "
                                  "budget buys on survivors",
        prior_recovery=0.115, prior_bpw_cost=0.0,
        provenance="qwen_adaptive_k exact byte auction: K spans 48..96 (mean 65.4) and 84 of 94 "
                   "layers buy a richer down rung than uniform-64 gave them, at 0.996853694 "
                   "complete BPW. Predicted mean layer error 0.46213 vs 0.521819 for uniform-64 "
                   "under the identical predictor. PREDICTED, not yet forward-measured.",
        status="alive"),
    Treatment(
        id="generated_tied_params", rung="E7", space="structural",
        treats=("quantization_reconstruction",),
        installed_bits_per_tensor="template bank + per-expert delta codes",
        prior_recovery=0.0, prior_bpw_cost=None,
        provenance="Lane F: expert templates dead (raw and row-normalized off-diagonal cosine "
                   "agree to 3 s.f., 0.00166 vs 0.00168 - experts are orthogonal in BOTH spaces); "
                   "cross-layer same-index tying indistinguishable from a different-index control "
                   "at 1e-7.",
        status="dead",
        reopen="a parent measuring row-normalized mean off-diagonal expert cosine >= 0.10"),
    Treatment(
        id="conditional_residual", rung="E3", space="dynamic",
        treats=("rare_token_collapse", "domain_collapse", "logit_ranking_collapse"),
        installed_bits_per_tensor="a residual stage that fires on a trigger; INSTALLED bits are "
                                  "billed in full even though active bits per token are lower",
        prior_recovery=None, prior_bpw_cost=None,
        provenance="untested on this parent",
        status="untested",
        conditional=True),
)

BY_ID = {t.id: t for t in TREATMENTS}


# ── reachability + the no-repetition law ──────────────────────────────────────────────────────
def reachable(diagnosis: dict[str, float], *, min_weight: float = 0.05) -> list[Treatment]:
    """Treatments that can reach any diagnosed cause carrying at least min_weight."""
    active = {d for d, w in diagnosis.items() if w >= min_weight}
    if not active:
        return []
    return [t for t in TREATMENTS if active & set(t.treats)]


def may_propose(t: Treatment, *, diagnosis_differs: bool = False, architecture_differs: bool = False,
                rate_regime_differs: bool = False, blocker_removed: bool = False) -> tuple[bool, str]:
    """The no-repetition law. A dead or negative treatment needs a named reason to come back."""
    if t.status in ("alive", "untested"):
        return True, "status permits proposal"
    reasons = [n for n, v in (("diagnosis differs materially", diagnosis_differs),
                              ("architecture changes reachability", architecture_differs),
                              ("rate regime differs", rate_regime_differs),
                              ("implementation removes the prior blocker", blocker_removed)) if v]
    if reasons:
        return True, "re-proposed under the no-repetition law: " + "; ".join(reasons)
    return False, f"{t.id} is {t.status} and no reopening condition is met: {t.reopen or 'n/a'}"


def escalate(diagnosis: dict[str, float], *, budget_bpw: float,
             **repeat_flags: bool) -> list[dict[str, Any]]:
    """Diagnosis-matched proposals, cheapest rung first, priced, with blocked ones explained.

    Returns EVERY candidate including the refused ones - a controller that silently drops a
    treatment is indistinguishable from one that never considered it.
    """
    out = []
    for t in sorted(reachable(diagnosis), key=lambda x: (LADDER.index(x.rung), -(x.utility() or 0))):
        ok, why = may_propose(t, **repeat_flags)
        u = t.utility()
        affordable = t.prior_bpw_cost is None or t.prior_bpw_cost <= budget_bpw
        out.append({
            "treatment": t.id, "rung": t.rung, "rung_name": LADDER_NAMES[t.rung],
            "space": t.space, "status": t.status,
            "treats": [d for d in t.treats if diagnosis.get(d, 0) >= 0.05],
            "prior_recovery": t.prior_recovery, "prior_bpw_cost": t.prior_bpw_cost,
            "utility_per_bpw": (None if u is None else (u if math.isfinite(u) else "inf")),
            "proposable": bool(ok and affordable), "reason": why if not ok else (
                "" if affordable else f"costs {t.prior_bpw_cost} bpw, budget is {budget_bpw}"),
            "conditional": t.conditional,
        })
    return out


# ── the joint byte auction ────────────────────────────────────────────────────────────────────
def auction(slots: list[dict[str, Any]], budget_bpw: float) -> dict[str, Any]:
    """Rank competing uses of a marginal budget by recovery per bit and fill greedily.

    Greedy is EXACT here only because the slots are independent and divisible-at-the-margin; when
    they are not, this is an ordering heuristic and the caller must say so. Zero-cost treatments
    are taken first and unconditionally - they cannot lose an auction they do not spend in.
    """
    priced = []
    for s in slots:
        cost = s.get("bpw_cost")
        rec = s.get("recovery")
        if rec is None or cost is None:
            priced.append({**s, "utility": None, "taken": False,
                           "note": "untested: cannot be priced, so cannot be promoted"})
            continue
        u = float("inf") if cost <= 0 else rec / cost
        priced.append({**s, "utility": u, "taken": False})
    priced.sort(key=lambda x: (x["utility"] is None, -(x["utility"] if x["utility"] is not None
                                                       else -1)))
    spent, taken = 0.0, []
    for p in priced:
        if p["utility"] is None:
            continue
        c = max(0.0, float(p["bpw_cost"]))
        if spent + c <= budget_bpw + 1e-12:
            p["taken"] = True
            spent += c
            taken.append(p["id"])
    return {"budget_bpw": budget_bpw, "spent_bpw": round(spent, 9),
            "remaining_bpw": round(budget_bpw - spent, 9), "taken": taken, "slots": priced}


# ── controller ────────────────────────────────────────────────────────────────────────────────
@dataclass
class Cycle:
    """One OBSERVE..RETAIN pass. Checkpointable and reproducible."""
    cycle: int
    diagnosis: dict[str, float]
    proposals: list[dict[str, Any]] = field(default_factory=list)
    chosen: str | None = None
    quality_before: dict[str, float] | None = None
    quality_after: dict[str, float] | None = None
    bits_before: int | None = None
    bits_after: int | None = None
    outcome: str = "pending"          # retain | rollback | escalate | pending
    note: str = ""


def decide(c: Cycle, *, gate_metric: str = "mean_sym_kl", lower_is_better: bool = True,
           min_relative_gain: float = 0.02) -> Cycle:
    """RETAIN / ROLLBACK / ESCALATE from measured before/after quality and exact bits.

    A treatment that does not move the gate metric by at least min_relative_gain is ROLLED BACK
    even if it looks harmless: bytes it holds are bytes another treatment cannot use.
    """
    if not c.quality_before or not c.quality_after:
        c.outcome, c.note = "pending", "no measured before/after quality"
        return c
    a, b = c.quality_before.get(gate_metric), c.quality_after.get(gate_metric)
    if a is None or b is None:
        c.outcome, c.note = "pending", f"metric {gate_metric} missing"
        return c
    rel = (a - b) / abs(a) if lower_is_better else (b - a) / abs(a)
    if c.bits_after is not None and c.bits_before is not None and c.bits_after > c.bits_before:
        spent = c.bits_after - c.bits_before
    else:
        spent = 0
    if rel >= min_relative_gain:
        c.outcome = "retain"
        c.note = f"{gate_metric} improved {rel*100:.2f} pct for {spent} extra bits"
    elif rel <= 0:
        c.outcome = "rollback"
        c.note = f"{gate_metric} did not improve ({rel*100:.2f} pct); treatment removed"
    else:
        c.outcome = "escalate"
        c.note = (f"{gate_metric} improved only {rel*100:.2f} pct, below the {min_relative_gain*100:.0f} "
                  f"pct bar; escalate to the next rung")
    return c


def registry() -> dict[str, Any]:
    """The promotable Doctor Gen3 registry, with every prior's provenance attached."""
    return {
        "schema": SCHEMA,
        "generation": "DOCTOR_GENERATION_3",
        "supersedes": "sparse-residual Doctor (Gen1/Gen2)",
        "law": {"ceiling": "complete_bits / original_weight_count <= 1/1",
                "doctor_free_bytes": False,
                "conditional_billing": "installed bits are billed in full even when a stage fires "
                                       "on a small fraction of tokens; active_bits_per_token is "
                                       "reported separately and never substituted for it"},
        "pipeline": ["OBSERVE", "LOCALIZE", "ATTRIBUTE", "PROPOSE", "PRICE", "TREAT", "REPACK",
                     "RE-EVALUATE", "RETAIN/ROLLBACK/ESCALATE"],
        "diagnoses": DIAGNOSES,
        "ladder": {r: LADDER_NAMES[r] for r in LADDER},
        "treatments": [asdict(t) | {"utility_per_bpw": (
            None if t.utility() is None else (t.utility() if math.isfinite(t.utility()) else "inf"))}
            for t in TREATMENTS],
        "counts": {"alive": sum(1 for t in TREATMENTS if t.status == "alive"),
                   "negative": sum(1 for t in TREATMENTS if t.status == "negative"),
                   "dead": sum(1 for t in TREATMENTS if t.status == "dead"),
                   "untested": sum(1 for t in TREATMENTS if t.status == "untested")},
        "claim": "Priors order candidates. No treatment is promoted on predicted utility; "
                 "promotion requires a real parent-vs-packed forward under the frozen quality "
                 "contract, and no protected domain may catastrophically regress.",
    }


def demo() -> None:
    """Self-check: reachability, the no-repetition law, auction ordering, and the decide gate."""
    # a routing-bound diagnosis must reach routing treatments and must NOT propose the dead ones
    routing = {"routing_omission": 0.8, "router_redistribution": 0.2}
    props = escalate(routing, budget_bpw=0.05)
    ids = {p["treatment"]: p for p in props}
    assert "expert_merge" in ids and not ids["expert_merge"]["proposable"], ids.get("expert_merge")
    assert "router_distill" in ids and not ids["router_distill"]["proposable"]
    assert ids["structural_adaptive_k"]["proposable"], ids["structural_adaptive_k"]

    # the no-repetition law lets a dead treatment back ONLY with a named reason
    ok, why = may_propose(BY_ID["expert_merge"])
    assert not ok and "dead" in why
    ok2, why2 = may_propose(BY_ID["expert_merge"], architecture_differs=True)
    assert ok2 and "no-repetition law" in why2

    # a reconstruction diagnosis reaches QAT and the codec treatments, cheapest rung first
    recon = {"quantization_reconstruction": 0.9}
    order = [p["rung"] for p in escalate(recon, budget_bpw=1.0)]
    assert order == sorted(order, key=LADDER.index), order

    # zero-cost treatments are taken first and never lose an auction they do not spend in
    res = auction([{"id": "free", "recovery": 0.05, "bpw_cost": 0.0},
                   {"id": "pricey", "recovery": 0.20, "bpw_cost": 0.30},
                   {"id": "cheap", "recovery": 0.10, "bpw_cost": 0.05},
                   {"id": "untested", "recovery": None, "bpw_cost": 0.01}], budget_bpw=0.06)
    assert res["taken"][0] == "free", res["taken"]
    assert "cheap" in res["taken"] and "pricey" not in res["taken"], res["taken"]
    assert "untested" not in res["taken"], "an unpriced treatment must never be promoted"
    assert res["spent_bpw"] <= 0.06 + 1e-12

    # decide(): retain / rollback / escalate all reachable, and a no-op is rolled back not kept
    c = decide(Cycle(1, recon, quality_before={"mean_sym_kl": 8.599},
                     quality_after={"mean_sym_kl": 7.699}, bits_before=100, bits_after=110))
    assert c.outcome == "retain", c
    c2 = decide(Cycle(2, recon, quality_before={"mean_sym_kl": 8.0},
                      quality_after={"mean_sym_kl": 8.1}))
    assert c2.outcome == "rollback", c2
    c3 = decide(Cycle(3, recon, quality_before={"mean_sym_kl": 8.0},
                      quality_after={"mean_sym_kl": 7.95}))
    assert c3.outcome == "escalate", c3

    # the S1 measurement must reproduce the Gen1 prior it is quoted from
    g1 = BY_ID["sparse_residual_doctor_gen1"]
    assert abs((8.599 - 7.699) / 8.599 - g1.prior_recovery) < 0.005, g1.prior_recovery
    assert abs((0.999769787 - 0.948410027) - g1.prior_bpw_cost) < 1e-4, g1.prior_bpw_cost

    r = registry()
    assert r["counts"]["dead"] >= 3 and r["counts"]["alive"] >= 5, r["counts"]
    print(json.dumps({"ok": True, "counts": r["counts"],
                      "routing_proposable": [p["treatment"] for p in props if p["proposable"]],
                      "auction_taken": res["taken"]}, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Doctor Generation 3 controller.")
    ap.add_argument("--registry", default="")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    if args.demo:
        demo()
        return 0
    r = registry()
    if args.registry:
        Path(args.registry).parent.mkdir(parents=True, exist_ok=True)
        Path(args.registry).write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "treatments"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
