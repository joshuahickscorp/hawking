#!/usr/bin/env python3.12
"""Adaptive EXTREME planner: one recommended candidate per parent, from evidence.

This replaces the fixed model x rate x branch matrix with an evidence-driven planner
(the directive's "Replace the fixed bit ladder"). It is a PLANNER: it reads the frozen
prior ledger and emits a plan artifact. It never launches a bake, never mutates the
campaign, and never promotes a quality claim the campaign did not seal.

Per parent it:
  1. binds exact parameter count, parent baseline, byte ceiling, protected capability
     vector;
  2. builds an uncertainty-aware rate frontier from imported priors;
  3. runs the F0..F4 feasibility ladder (proxies prioritize; only F4 proves);
  4. diagnoses each rate (no_material_damage / signal_degradation /
     computation_collapse / mixed_failure / undetermined);
  5. generates a Doctor program from the diagnosis (retaining the four controls,
     promoting only high-value causal treatments);
  6. brackets the Event Horizon and descends adaptively, skipping resolved rates;
  7. reports the lowest passing rate, the next lower failing/unproven rate, the reason
     each lower rate fails or is unproven, and the boundary probes still needed;
  8. recommends the lowest passing rate as EXTREME (BALANCED/FIDELITY secondary).

Evidence discipline: the whole Pass-B campaign runs with quality_claims_permitted=false,
so a passing verdict is PROVISIONAL (physical bytes sealed, quality engineering-grade,
F4_quality pending). The planner never launders that into a sealed win.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, SCHEMA_ADAPTIVE_PLAN, seal_field, sealed, now_iso,
)

# The adaptive descent ladder (directive step 5). 4.0 is the top density anchor.
TOP_ANCHOR = 4.0
DESCENT_RATES: tuple[float, ...] = (3.0, 2.0, 1.0, 0.8, 0.55, 0.5, 0.33, 0.25, 0.1)
EVAL_ORDER: tuple[float, ...] = (TOP_ANCHOR, *DESCENT_RATES)  # high -> low

# The four controls that are always retained (directive step 8).
CONTROLS = ("zero_treatment", "equal_byte_codec", "smaller_higher_bit", "public_same_byte")

DIAGNOSES = ("no_material_damage", "signal_degradation", "computation_collapse",
             "mixed_failure", "undetermined")

VERDICTS = ("PASS", "FAIL", "INCONCLUSIVE", "UNPROVEN")


@dataclasses.dataclass(frozen=True)
class CapabilityContract:
    """The frozen standalone production-quality gate (protected capability vector)."""
    ppl_rel_delta_max: float = 0.08          # campaign Pareto GOOD_PPL_DELTA_MAX
    capability_abs_delta_min: float = -0.05  # campaign Pareto GOOD_CAPABILITY_DELTA_MIN
    # Absolute collapse floors for the diagnosis.
    collapse_ppl_rel: float = 1.0            # ppl doubled = collapse
    collapse_cap_abs: float = -0.30

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DeviceEnvelope:
    name: str
    weight_budget_gb: float
    ram_gb: float
    ram_gbps: float

    def byte_ceiling_bpw(self, params_b: float) -> float:
        """The highest physical BPW whose artifact still fits the resident weight budget."""
        if params_b <= 0:
            return 0.0
        return self.weight_budget_gb * 8.0 / params_b


def default_device() -> DeviceEnvelope:
    from studio_manifest import DEFAULT_HARDWARE as hw
    return DeviceEnvelope(name=hw.name, weight_budget_gb=hw.weight_budget_gb,
                          ram_gb=hw.ram_gb, ram_gbps=hw.ram_gbps)


def diagnose(ppl_rel: float | None, cap_abs: float | None, c: CapabilityContract) -> str:
    if ppl_rel is None or cap_abs is None:
        return "undetermined"
    ppl_ok = ppl_rel <= c.ppl_rel_delta_max
    cap_ok = cap_abs >= c.capability_abs_delta_min
    ppl_collapse = ppl_rel > c.collapse_ppl_rel
    cap_collapse = cap_abs <= c.collapse_cap_abs
    if ppl_ok and cap_ok:
        return "no_material_damage"
    if ppl_collapse and cap_collapse:
        return "computation_collapse"
    if ppl_collapse or cap_collapse:
        # one axis has collapsed while the other has not -> mixed failure
        return "mixed_failure"
    return "signal_degradation"


def doctor_program(diagnosis: str) -> dict[str, Any]:
    """Generate a Doctor program from the diagnosis (not from fixed branches)."""
    program = {"controls": list(CONTROLS), "promote": [], "rationale": ""}
    if diagnosis == "no_material_damage":
        program["rationale"] = "codec control already within contract; descend, no treatment promoted"
    elif diagnosis == "signal_degradation":
        program["promote"] = ["doctor_static", "doctor_conditional"]
        program["rationale"] = "graceful degradation is treatable; promote causal recovery treatments"
    elif diagnosis == "mixed_failure":
        program["promote"] = ["doctor_conditional", "capability_targeted_agr"]
        program["rationale"] = "one capability axis collapsed; promote capability-targeted rehabilitation"
    elif diagnosis == "computation_collapse":
        program["rationale"] = "likely below the Event Horizon; confirm with a control, do not spend heavy treatment"
    else:  # undetermined
        program["promote"] = ["codec_control_probe"]
        program["rationale"] = "no full-model quality evidence yet; probe the control first, then diagnose"
    return program


def _feasibility_ladder(branch_rows: list[dict[str, Any]], byte_ceiling_bpw: float,
                        c: CapabilityContract) -> dict[str, Any]:
    """F0..F4 for one rate, across whatever branch evidence exists.

    F0 byte feasibility (deterministic), F1 tensor/block proxy, F2 shard proxy,
    F3 full-model quality (provisional), F4 replicated/sealed proof.
    """
    control = next((r for r in branch_rows if r.get("branch") == "codec_control"), None)
    any_complete = [r for r in branch_rows if r.get("status") == "complete"]
    best_phys = None
    for r in branch_rows:
        bpw = (r.get("physical") or {}).get("all_in_model_payload_bpw")
        if isinstance(bpw, (int, float)):
            best_phys = bpw if best_phys is None else min(best_phys, bpw)

    # F0: does an achievable physical BPW fit the byte ceiling?
    if best_phys is not None:
        f0 = "feasible" if best_phys <= byte_ceiling_bpw else "infeasible"
    else:
        f0 = "uncertain"

    # F1: tensor/block realization proxy from the packed vs passthrough split.
    f1 = "unavailable"
    realization_gap = None
    if control and control.get("physical"):
        phys = control["physical"]
        target = phys.get("target_physical_bpw")
        actual = phys.get("all_in_model_payload_bpw")
        if isinstance(target, (int, float)) and isinstance(actual, (int, float)) and target:
            realization_gap = round(actual / target, 4)
            f1 = "proxy_available"

    # F2: shard proxy is subsumed once a full-model result exists.
    f2 = "subsumed_by_full_model" if any_complete else "unavailable"

    # F3: full-model quality (provisional, never proof).
    f3 = "provisional" if any_complete else "unavailable"

    # F4: physical bytes seal via receipt; quality F4 is unproven under Pass B.
    f4_physical = "sealed" if any_complete else "unproven"
    f4_quality = "unproven"

    return {
        "F0_byte_feasibility": f0,
        "F1_tensor_block": f1,
        "F1_realization_gap": realization_gap,
        "F2_shard": f2,
        "F3_full_model_quality": f3,
        "F4_physical": f4_physical,
        "F4_quality": f4_quality,
        "best_physical_bpw": best_phys,
    }


def _rate_verdict(branch_rows: list[dict[str, Any]], byte_ceiling_bpw: float,
                  c: CapabilityContract) -> dict[str, Any]:
    """Assess one (parent, rate) across its branches."""
    # Choose the best branch: prefer one that passes the contract, then lowest ppl.
    graded = []
    for r in branch_rows:
        if r.get("status") in ("unsupported", "negative"):
            graded.append({"branch": r.get("branch"), "kind": "disposition",
                           "reason_code": r.get("reason_code"), "row": r})
            continue
        q = r.get("quality_provisional") or {}
        ppl = q.get("ppl_relative_delta")
        cap = q.get("capability_absolute_delta")
        phys = (r.get("physical") or {}).get("all_in_model_payload_bpw")
        graded.append({"branch": r.get("branch"), "kind": "complete", "ppl": ppl,
                       "cap": cap, "phys": phys, "row": r})

    complete = [g for g in graded if g["kind"] == "complete" and g.get("ppl") is not None]
    dispositions = [g for g in graded if g["kind"] == "disposition"]

    ladder = _feasibility_ladder(branch_rows, byte_ceiling_bpw, c)

    def _fits(g) -> bool:
        return g.get("phys") is None or g["phys"] <= byte_ceiling_bpw

    def _is_doctor(g) -> bool:
        return isinstance(g.get("branch"), str) and g["branch"].startswith("doctor")

    passing = [g for g in complete
               if diagnose(g["ppl"], g["cap"], c) == "no_material_damage" and _fits(g)]

    fail_kind = None
    if passing:
        # A passing rate: prefer the treatment-free control, else the cheapest passing branch.
        control_pass = [g for g in passing if g.get("branch") == "codec_control"]
        best = (control_pass or sorted(passing, key=lambda g: (g.get("phys") or math.inf)))[0]
        verdict, resolved, diag = "PASS", True, "no_material_damage"
        reason = ("full-model quality within the frozen contract (provisional; physical sealed, "
                  "F4_quality pending)")
        program = doctor_program("no_material_damage")
        best_branch = best["branch"]
    elif complete:
        # Something was measured but nothing passes. Classify by the best diagnosis and
        # decide whether the rate is RESOLVED (Doctor cannot / did not heal) or UNRESOLVED
        # (a treatable control failure whose Doctor recovery is not yet terminal).
        best = min(complete, key=lambda g: g["ppl"])
        diag = diagnose(best["ppl"], best["cap"], c)
        program = doctor_program(diag)
        doctor_tried = any(_is_doctor(g) for g in complete)
        best_branch = best["branch"]
        if diag == "computation_collapse":
            verdict, resolved, fail_kind = "FAIL", True, "measured_collapse"
            reason = "diagnosis computation_collapse; below the Event Horizon, Doctor cannot recover"
        elif doctor_tried:
            verdict, resolved, fail_kind = "FAIL", True, "measured_out_of_contract"
            reason = "Doctor recovery reached a terminal cell yet stays out of the frozen contract"
        else:
            verdict, resolved = "INCONCLUSIVE", False
            reason = (f"codec control fails but the diagnosis ({diag}) is treatable; Doctor "
                      "recovery is not yet terminal at this rate")
    elif dispositions:
        # The campaign ruled this out with an adaptive-defer disposition. That RESOLVES the
        # rate as a FAIL, but a scheduling deferral is NOT a measured collapse, so it must
        # not set the proven Event-Horizon collapse boundary. Keep the true reason_code and
        # leave the diagnosis undetermined rather than asserting computation_collapse.
        verdict, resolved, diag = "FAIL", True, "undetermined"
        fail_kind = "campaign_deferral"
        reason = f"campaign disposition (not a measured collapse): {dispositions[0].get('reason_code')}"
        program = doctor_program("undetermined")
        best_branch = dispositions[0]["branch"]
    elif any(g["kind"] == "complete" for g in graded):
        # physical sealed but no quality number -> inconclusive on the contract
        verdict, resolved, diag = "INCONCLUSIVE", False, "undetermined"
        reason = "physical bytes sealed but no full-model quality evidence at this rate"
        program = doctor_program("undetermined")
        best_branch = graded[0]["branch"]
    else:
        verdict, resolved, diag = "UNPROVEN", False, "undetermined"
        reason = "no terminal evidence at this rate"
        program = doctor_program("undetermined")
        best_branch = None

    return {
        "verdict": verdict,
        "resolved": resolved,
        "diagnosis": diag,
        "fail_kind": fail_kind,
        "reason": reason,
        "doctor_program": program,
        "feasibility": ladder,
        "best_branch": best_branch,
        "branch_evidence": [
            {"branch": g["branch"], "kind": g["kind"],
             "ppl_relative_delta": g.get("ppl"), "capability_absolute_delta": g.get("cap"),
             "physical_bpw": g.get("phys"), "reason_code": g.get("reason_code")}
            for g in graded
        ],
    }


def _group_by_parent(ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    parents: dict[str, dict[str, Any]] = {}
    for cell in ledger.get("cells", []):
        label = cell.get("model_label")
        if label is None:
            continue
        p = parents.setdefault(label, {"binding": {}, "rates": {}})
        p["binding"] = {
            "model_label": label,
            "model_family": cell.get("model_family"),
            "hf_id": cell.get("hf_id"),
            "exact_stored_parameter_count": cell.get("exact_stored_parameter_count"),
            "nominal_params_b": cell.get("nominal_params_b"),
        }
        bpw = cell.get("rate_bpw")
        p["rates"].setdefault(bpw, []).append(cell)
    return parents


def assess_parent(binding: dict[str, Any], rates: dict[float, list[dict[str, Any]]],
                  device: DeviceEnvelope, contract: CapabilityContract) -> dict[str, Any]:
    params_b = None
    if isinstance(binding.get("exact_stored_parameter_count"), (int, float)):
        params_b = binding["exact_stored_parameter_count"] / 1e9
    if not params_b:
        params_b = binding.get("nominal_params_b") or 0.0
    byte_ceiling_bpw = device.byte_ceiling_bpw(params_b) if params_b else 0.0

    frontier: list[dict[str, Any]] = []
    for rate in EVAL_ORDER:
        rows = rates.get(rate, [])
        assessment = _rate_verdict(rows, byte_ceiling_bpw, contract)
        frontier.append({"rate_bpw": rate, **assessment})

    passing = [f for f in frontier if f["verdict"] == "PASS"]
    lowest_pass = min((f["rate_bpw"] for f in passing), default=None)
    # the highest MEASURED-collapse rate strictly below the lowest pass = the proven
    # Event-Horizon collapse boundary. A campaign adaptive-defer disposition resolves the
    # rate but is a scheduling deferral, not a measured collapse, so it is excluded here.
    collapses_below = [f for f in frontier
                       if f["verdict"] == "FAIL" and f.get("fail_kind") == "measured_collapse"
                       and lowest_pass is not None and f["rate_bpw"] < lowest_pass]
    highest_fail_below = max((f["rate_bpw"] for f in collapses_below), default=None)

    # boundary probes: unresolved rates between the lowest pass and the first resolved fail
    boundary_probes: list[dict[str, Any]] = []
    for f in frontier:
        if f["verdict"] in ("UNPROVEN", "INCONCLUSIVE"):
            if lowest_pass is None or f["rate_bpw"] < lowest_pass:
                boundary_probes.append({
                    "rate_bpw": f["rate_bpw"], "current_verdict": f["verdict"],
                    "next_feasibility_tier": _next_tier(f["feasibility"]),
                    "doctor_program": f["doctor_program"],
                })

    # stop condition: lowest pass known AND the next lower evaluated rate is a resolved
    # FAIL or a disposition (evidence-closed on both sides of the boundary).
    stop_met = False
    if lowest_pass is not None:
        below = [f for f in frontier if f["rate_bpw"] < lowest_pass]
        below.sort(key=lambda f: -f["rate_bpw"])
        stop_met = bool(below) and below[0]["verdict"] == "FAIL"
        if not below:  # lowest pass is already the floor of the ladder
            stop_met = lowest_pass == DESCENT_RATES[-1]

    # provisional floor proxy: the lowest rate still treatably close (not collapsed, not
    # unproven). A cross-scale signal even when nothing passes the strict contract.
    treatable_diag = {"no_material_damage", "signal_degradation", "mixed_failure"}
    floor_proxy = min((f["rate_bpw"] for f in frontier if f["diagnosis"] in treatable_diag),
                      default=None)

    # anomaly: quality should be monotone in bpw, so a MEASURED collapse ABOVE a pass is
    # suspicious (noise or a mislabelled cell). Flag it rather than silently trusting the pass.
    anomaly = None
    if lowest_pass is not None:
        collapses_above = sorted(f["rate_bpw"] for f in frontier
                                 if f["verdict"] == "FAIL" and f.get("fail_kind") == "measured_collapse"
                                 and f["rate_bpw"] > lowest_pass)
        if collapses_above:
            anomaly = {"kind": "measured_collapse_above_pass", "rates_bpw": collapses_above,
                       "note": "non-monotone; the lower-rate pass may be noise, verify before trusting"}

    # lower-rate reasons (why each rate below the candidate fails or is unproven). When no
    # rate passes yet, report the reason for the whole frontier so the picture is complete.
    lower_reasons = []
    for f in frontier:
        if lowest_pass is None or f["rate_bpw"] < lowest_pass:
            lower_reasons.append({"rate_bpw": f["rate_bpw"], "verdict": f["verdict"],
                                  "diagnosis": f["diagnosis"], "reason": f["reason"]})

    if lowest_pass is not None:
        cand = next(f for f in frontier if f["rate_bpw"] == lowest_pass)
        phys = cand["feasibility"].get("best_physical_bpw")
        extreme = {
            "status": "PROVISIONAL_PASS",
            "rate_bpw": lowest_pass,
            "branch": cand["best_branch"],
            "physical_bpw": phys,
            "fits_byte_ceiling": (phys is not None and phys <= byte_ceiling_bpw),
            "evidence_grade": "provisional_pass_physical_sealed_quality_unproven",
            "reason": cand["reason"],
        }
    else:
        # no passing rate yet: recommend the highest treatable INCONCLUSIVE rate to probe next
        treatable = [f for f in frontier if f["verdict"] == "INCONCLUSIVE"]
        next_rate = max((f["rate_bpw"] for f in treatable), default=None)
        extreme = {
            "status": "UNPROVEN",
            "rate_bpw": None,
            "recommended_next_probe_bpw": next_rate,
            "evidence_grade": "unproven_pending_doctor_recovery",
            "reason": ("no rate passes the frozen contract with terminal evidence yet; the measured "
                       "codec-control cells are out of contract and Doctor recovery is not terminal"),
        }

    return {
        "binding": binding,
        "params_b": round(params_b, 4) if params_b else None,
        "byte_ceiling_bpw": round(byte_ceiling_bpw, 4) if byte_ceiling_bpw else None,
        "contract": contract.as_dict(),
        "rate_frontier": frontier,
        "extreme_candidate": extreme,
        "provisional_floor_proxy_bpw": floor_proxy,
        "event_horizon_bracket": {"lowest_pass_bpw": lowest_pass,
                                  "highest_fail_below_bpw": highest_fail_below},
        "lower_rate_reasons": lower_reasons,
        "stop_condition_met": stop_met,
        "boundary_probes_needed": boundary_probes,
        "frontier_anomaly": anomaly,
        "uncertainty": _uncertainty(frontier),
    }


def _next_tier(feasibility: dict[str, Any]) -> str:
    if feasibility.get("F0_byte_feasibility") == "uncertain":
        return "F0_byte_feasibility"
    if feasibility.get("F3_full_model_quality") != "provisional":
        return "F3_full_model_quality"
    return "F4_quality"


def _uncertainty(frontier: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = sum(1 for f in frontier if f["resolved"])
    total = len(frontier)
    return {
        "rates_total": total,
        "rates_resolved": resolved,
        "rates_unresolved": total - resolved,
        "resolution_fraction": round(resolved / total, 3) if total else 0.0,
    }


def scaling_prior(parent_assessments: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-parent scheduling prior: provisional passing floor vs log10(params).

    Used to bracket a parent that has no evidence yet (e.g. 72B). SCHEDULING ONLY,
    never proof (directive: "Use 72B effects only as scheduling priors ... never as proof").
    """
    points = []
    for a in parent_assessments:
        # prefer a real passing floor; fall back to the collapse-boundary proxy so the
        # trend has signal even before any rate passes the strict contract.
        floor = (a.get("extreme_candidate") or {}).get("rate_bpw")
        kind = "passing_floor"
        if floor is None:
            floor = a.get("provisional_floor_proxy_bpw")
            kind = "collapse_boundary_proxy"
        p = a.get("params_b")
        if floor is not None and p:
            points.append({"model_label": a["binding"]["model_label"],
                           "params_b": p, "provisional_floor_bpw": floor, "floor_kind": kind,
                           "log10_params": round(math.log10(p * 1e9), 4)})
    trend = "insufficient_points"
    slope = None
    if len(points) >= 2:
        xs = [q["log10_params"] for q in points]
        ys = [q["provisional_floor_bpw"] for q in points]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        if denom:
            slope = round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom, 4)
            trend = "floor_descends_with_scale" if slope < 0 else "floor_rises_or_flat"
    return {"points": sorted(points, key=lambda q: q["params_b"]),
            "log10_slope_bpw_per_decade": slope, "trend": trend,
            "note": "scheduling prior only; never a proof for an evidence-less parent"}


def build_plan(ledger: dict[str, Any], *, device: DeviceEnvelope | None = None,
               contract: CapabilityContract | None = None) -> dict[str, Any]:
    device = device or default_device()
    contract = contract or CapabilityContract()
    parents = _group_by_parent(ledger)
    assessments = [
        assess_parent(p["binding"], p["rates"], device, contract)
        for p in parents.values()
    ]
    assessments.sort(key=lambda a: (a.get("params_b") or 0.0))

    # cohort parents with no terminal cell yet (e.g. 72B running, 120B pending) are visible
    # as awaiting-evidence, bracketed only by the scaling prior (never proof).
    present = {a["binding"]["model_label"] for a in assessments}
    prior = scaling_prior(assessments)
    awaiting = []
    for m in ledger.get("cohort", []) or []:
        label = m.get("label")
        if label in present or label is None:
            continue
        p_b = m.get("nominal_params_b")
        awaiting.append({
            "model_label": label,
            "family": m.get("family"),
            "hf_id": m.get("hf_id"),
            "nominal_params_b": p_b,
            "status": "awaiting_evidence",
            "predicted_bracket": _predict_bracket(prior, p_b),
            "reason": ("no terminal cell yet; codec-control still running or dependency-blocked. "
                       "EXTREME is unproven until that seal; the scaling prior is scheduling only"),
        })
    awaiting.sort(key=lambda a: (a.get("nominal_params_b") or 0.0))

    plan = {
        "schema": SCHEMA_ADAPTIVE_PLAN,
        "generated_at": now_iso(),
        "campaign_plan_sha256": ledger.get("campaign_plan_sha256"),
        "prior_ledger_sha256": ledger.get("ledger_sha256"),
        "device": dataclasses.asdict(device),
        "contract": contract.as_dict(),
        "descent_rates": list(DESCENT_RATES),
        "parents": assessments,
        "parents_awaiting_evidence": awaiting,
        "scaling_prior": prior,
        "recommended_profile": "EXTREME",
        "secondary_profiles": ["BALANCED", "FIDELITY"],
    }
    return seal_field(plan, "plan_sha256")


def _predict_bracket(prior: dict[str, Any], params_b: float | None) -> dict[str, Any]:
    """Extrapolate a floor bracket for an evidence-less parent from the scaling prior.
    Scheduling only; never proof."""
    if not params_b or prior.get("log10_slope_bpw_per_decade") is None or not prior.get("points"):
        return {"predicted_floor_bpw": None, "basis": "insufficient_prior"}
    slope = prior["log10_slope_bpw_per_decade"]
    anchor = max(prior["points"], key=lambda q: q["params_b"])
    predicted = anchor["provisional_floor_bpw"] + slope * (math.log10(params_b * 1e9) - anchor["log10_params"])
    predicted = max(DESCENT_RATES[-1], min(TOP_ANCHOR, predicted))
    return {"predicted_floor_bpw": round(predicted, 3),
            "basis": f"extrapolated from {anchor['model_label']} at slope {slope}/decade",
            "note": "scheduling prior only; the parent needs its own sealed evidence"}


def selftest() -> dict[str, Any]:
    """Synthetic ledger -> plan, exercising PASS/FAIL/UNPROVEN and the bracket."""
    def cell(label, params, rate, branch, status, ppl=None, cap=None, phys=None, reason=None):
        return {"model_label": label, "model_family": "qwen2.5-dense", "hf_id": "toy/x",
                "exact_stored_parameter_count": int(params * 1e9), "nominal_params_b": params,
                "rate_id": str(rate), "rate_bpw": rate, "branch": branch, "status": status,
                "physical": {"all_in_model_payload_bpw": phys, "target_physical_bpw": rate},
                "quality_provisional": {"ppl_relative_delta": ppl, "capability_absolute_delta": cap,
                                        "quality_claims_permitted": False},
                "reason_code": reason}
    ledger = {
        "schema": "hawking.eco.prior_ledger.v1", "campaign_plan_sha256": "0" * 64,
        "ledger_sha256": "0" * 64,
        "cells": [
            # 7B: 2bpw passes, 1bpw fails -> bracket (2.0, 1.0), EXTREME = 2.0
            cell("7B", 7.6, 2.0, "codec_control", "complete", ppl=0.05, cap=-0.02, phys=2.3),
            cell("7B", 7.6, 1.0, "codec_control", "complete", ppl=1.4, cap=-0.4, phys=1.2),
            # 14B: 1bpw passes, 0.8 unsupported disposition -> bracket (1.0, 0.8)
            cell("14B", 14.8, 1.0, "codec_control", "complete", ppl=0.06, cap=-0.03, phys=1.25),
            cell("14B", 14.8, 0.8, "doctor_full", "unsupported",
                 reason="empirical-quality-cliff-adaptive-defer"),
            # 72B: nothing terminal -> UNPROVEN everywhere (mirrors the live state)
        ],
    }
    plan = build_plan(ledger)
    if not sealed(plan, "plan_sha256"):
        raise EcoError("selftest plan not sealed")
    by_label = {a["binding"]["model_label"]: a for a in plan["parents"]}
    if by_label["7B"]["extreme_candidate"]["rate_bpw"] != 2.0:
        raise EcoError(f"7B extreme wrong: {by_label['7B']['extreme_candidate']}")
    if by_label["7B"]["event_horizon_bracket"] != {"lowest_pass_bpw": 2.0, "highest_fail_below_bpw": 1.0}:
        raise EcoError(f"7B bracket wrong: {by_label['7B']['event_horizon_bracket']}")
    if not by_label["7B"]["stop_condition_met"]:
        raise EcoError("7B stop condition should be met (2.0 pass, 1.0 fail)")
    if by_label["14B"]["extreme_candidate"]["rate_bpw"] != 1.0:
        raise EcoError("14B extreme wrong")
    return {"ok": True, "seven_b_extreme": 2.0, "fourteen_b_extreme": 1.0,
            "scaling_trend": plan["scaling_prior"]["trend"], "plan_sha256": plan["plan_sha256"]}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Adaptive EXTREME planner (plan-only).")
    ap.add_argument("--campaign-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ppl-max", type=float, default=None, help="override contract ppl_rel_delta_max")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
    else:
        import eco_import
        cfg = eco_import.default_config(args.campaign_root)
        ledger = eco_import.build_ledger(cfg)
        contract = CapabilityContract(ppl_rel_delta_max=args.ppl_max) if args.ppl_max else None
        plan = build_plan(ledger, contract=contract)
        if args.out:
            from eco_common import atomic_write_json
            atomic_write_json(args.out, plan)
        summary = {
            "schema": plan["schema"], "plan_sha256": plan["plan_sha256"],
            "parents": [
                {"model_label": a["binding"]["model_label"], "params_b": a["params_b"],
                 "extreme_candidate": a["extreme_candidate"],
                 "bracket": a["event_horizon_bracket"],
                 "stop_condition_met": a["stop_condition_met"],
                 "boundary_probes": len(a["boundary_probes_needed"]),
                 "resolution": a["uncertainty"]["resolution_fraction"]}
                for a in plan["parents"]
            ],
            "scaling_prior": plan["scaling_prior"]["trend"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
