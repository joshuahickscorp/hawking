#!/usr/bin/env python3.12
"""Evidence-closed retirement from the sealed empirical harvest (master goal section 13.1).

Consumes the succ_harvest dataset and generates explicit, sealed retirement receipts for
FUTURE successor experiments that the measured evidence has made scientifically unnecessary,
because they are strictly dominated or lie below a REPLICATED measured collapse boundary.
It never mutates a completed legacy cell; it shrinks the successor's future work. Every
receipt preserves the evidence, states the dominance/impossibility reason and the affected
frontier decision, and records exactly what new evidence would justify reopening.

The successor selector (succ_engine) consults these receipts and skips retired
(model, rate, branch) combinations. This is how the measured findings wire into scheduling.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import EcoError, seal_field, sealed, now_iso  # noqa: E402

RETIREMENT_LEDGER_SCHEMA = "hawking.successor.retirement_ledger.v1"
BRANCHES = ("codec_control", "doctor_static", "doctor_conditional", "doctor_full")
DESCENT_RATES = (3.0, 2.0, 1.0, 0.8, 0.55, 0.5, 0.33, 0.25, 0.1)


class RetireError(EcoError):
    """Fail-closed retirement error."""


def replicated_collapse_boundaries(harvest: dict[str, Any]) -> dict[str, float]:
    """Per model, the HIGHEST rate at which >= 2 branches show measured_computation_collapse
    (replication). Everything strictly below that rate is below the collapse boundary."""
    per: dict[str, dict[float, int]] = {}
    for r in harvest.get("rows", []):
        if r.get("failure_class") != "measured_computation_collapse":
            continue
        model = r.get("model_label")
        rate = r.get("nominal_target_bpw")
        if model is None or not isinstance(rate, (int, float)):
            continue
        per.setdefault(model, {}).setdefault(rate, 0)
        per[model][rate] += 1
    boundaries: dict[str, float] = {}
    for model, rates in per.items():
        replicated = [rate for rate, n in rates.items() if n >= 2]
        if replicated:
            boundaries[model] = max(replicated)
    return boundaries


def dominated_targets(harvest: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for r in harvest.get("rows", []):
        if r.get("dominated") and r.get("dominated_by"):
            out.append({"cell_id": r.get("cell_id"), "model_label": r.get("model_label"),
                        "rate_bpw": r.get("nominal_target_bpw"), "branch": r.get("branch"),
                        "dominated_by": r.get("dominated_by"),
                        "physical_bpw": r.get("physical_bpw"),
                        "result_sha256": r.get("result_sha256")})
    return out


def _retire(plan_identity: str, ground: str, *, rationale: str, observations: list[str],
            affected_region: str, reopening: str, extra_evidence: dict[str, Any]) -> dict[str, Any]:
    import succ_gc
    exp = {
        "plan_identity": plan_identity,
        "rationale": rationale,
        "supporting_observations": observations,
        "uncertainty": {"basis": "measured_terminal_evidence", "quality_claims_permitted": False},
        "affected_region": affected_region,
        "reopening_criterion": reopening,
    }
    evidence = {"ground": ground, **extra_evidence}
    return succ_gc.retire_experiment(exp, evidence)


def build_retirement_ledger(harvest: dict[str, Any]) -> dict[str, Any]:
    """Generate sealed retirement receipts for future successor experiments made unnecessary
    by the measured evidence. Applied to the successor queue, never the legacy campaign."""
    if harvest.get("schema") != "hawking.successor.empirical_harvest.v1":
        raise RetireError("input is not an empirical harvest dataset")
    if not sealed(harvest, "harvest_sha256"):
        raise RetireError("harvest is not self-sealed; refusing to retire from it")

    boundaries = replicated_collapse_boundaries(harvest)
    receipts: list[dict[str, Any]] = []
    retired_keys: list[str] = []

    # (1) below a replicated collapse boundary: retire the successor's would-be probes at
    #     every lower rate x every branch (they cannot pass what already collapsed above).
    for model, boundary in boundaries.items():
        obs = [r.get("result_sha256") for r in harvest["rows"]
               if r.get("model_label") == model and r.get("failure_class") == "measured_computation_collapse"
               and isinstance(r.get("result_sha256"), str)][:4]
        for rate in DESCENT_RATES:
            if rate < boundary:
                for branch in BRANCHES:
                    key = f"{model}__{rate}bpw__{branch}"
                    rec = _retire(
                        key, "cannot_change_frontier_conservative",
                        rationale=(f"{model} shows a REPLICATED measured computation collapse at "
                                   f"{boundary} bpw; under the sealed harvest as a conservative "
                                   f"lower bound, a successor probe at {rate} bpw (< boundary) "
                                   f"cannot change the selected EXTREME frontier."),
                        observations=obs, affected_region=f"{model} sub-{boundary}bpw",
                        reopening=("a new mechanism that restores computation below the collapse "
                                   "boundary with a measured full-model pass, or a corrected "
                                   "collapse boundary from new evidence"),
                        extra_evidence={"conservative_bound_sha256": harvest.get("harvest_sha256"),
                                        "bound_direction": "lower",
                                        "replicated_collapse_boundary_bpw": boundary,
                                        "target_rate_bpw": rate})
                    receipts.append(rec)
                    retired_keys.append(key)

    # (2) dominated branch/rate: retire the successor's would-be re-run of a dominated combo.
    # Single-seed Pass-B evidence cannot claim >=2 replications, so this uses the conservative
    # ground (the sealed harvest bounds the frontier on both axes) rather than overclaiming.
    for tgt in dominated_targets(harvest):
        key = f"{tgt['model_label']}__{tgt['rate_bpw']}bpw__{tgt['branch']}"
        rec = _retire(
            key, "cannot_change_frontier_conservative",
            rationale=(f"{key} is dominated by {tgt['dominated_by']} (<= physical bytes AND "
                       f"<= ppl delta) under the sealed harvest; a successor re-run cannot change "
                       f"the frontier."),
            observations=[tgt["result_sha256"], f"dominating:{tgt['dominated_by']}"],
            affected_region=f"{tgt['model_label']} @ {tgt['rate_bpw']}bpw {tgt['branch']}",
            reopening="a new treatment at this rate/branch that beats the dominating program on "
                      "physical bytes or measured quality",
            extra_evidence={"conservative_bound_sha256": harvest.get("harvest_sha256"),
                            "bound_direction": "both",
                            "dominating_program": tgt["dominated_by"]})
        receipts.append(rec)
        retired_keys.append(key)

    ledger = {
        "schema": RETIREMENT_LEDGER_SCHEMA,
        "generated_at": now_iso(),
        "harvest_sha256": harvest.get("harvest_sha256"),
        "campaign_plan_sha256": harvest.get("campaign_plan_sha256"),
        "applied_to": "successor_queue_only",
        "non_interference": "no completed legacy cell is mutated; these retire FUTURE successor work",
        "replicated_collapse_boundaries": boundaries,
        "retired_count": len(receipts),
        "retired_keys": sorted(set(retired_keys)),
        "receipts": receipts,
    }
    return seal_field(ledger, "ledger_sha256")


def is_retired(ledger: dict[str, Any], model_label: str, rate_bpw: float, branch: str) -> bool:
    return f"{model_label}__{rate_bpw}bpw__{branch}" in set(ledger.get("retired_keys", []))


def selftest() -> dict[str, Any]:
    from eco_common import hash_value
    # a harvest where 7B collapses (replicated: 2 branches) at 2.0 bpw, and a dominated 3bpw cell
    def row(cid, model, rate, branch, fc, phys=None, ppl=None, dom=False, domby=None):
        return {"cell_id": cid, "model_label": model, "nominal_target_bpw": rate, "branch": branch,
                "failure_class": fc, "physical_bpw": phys, "ppl_relative_delta": ppl,
                "result_sha256": "a" * 64, "dominated": dom, "dominated_by": domby, "status": "complete"}
    harvest = seal_field({
        "schema": "hawking.successor.empirical_harvest.v1", "harvest_sha256": "x",
        "campaign_plan_sha256": "c" * 64, "rows": [
            row("7b_2_cc", "7B", 2.0, "codec_control", "measured_computation_collapse", phys=2.4, ppl=5.0),
            row("7b_2_ds", "7B", 2.0, "doctor_static", "measured_computation_collapse", phys=2.5, ppl=6.0),
            row("7b_3_ds", "7B", 3.0, "doctor_static", "measured_quality_failure", phys=3.6, ppl=0.2,
                dom=True, domby="7b_3_cc"),
        ]}, "harvest_sha256")
    ledger = build_retirement_ledger(harvest)
    if not sealed(ledger, "ledger_sha256"):
        raise RetireError("ledger not sealed")
    if ledger["replicated_collapse_boundaries"].get("7B") != 2.0:
        raise RetireError(f"collapse boundary wrong: {ledger['replicated_collapse_boundaries']}")
    # below 2.0 bpw: 7 descent rates (1.0..0.1) x 4 branches = 28 retired, plus 1 dominated
    if not is_retired(ledger, "7B", 1.0, "codec_control"):
        raise RetireError("sub-boundary probe should be retired")
    if is_retired(ledger, "7B", 3.0, "codec_control"):
        raise RetireError("above-boundary control should NOT be blanket-retired")
    if not is_retired(ledger, "7B", 3.0, "doctor_static"):
        raise RetireError("dominated 3bpw doctor_static should be retired")
    # every receipt preserves identity + reopening + a real ground
    for rec in ledger["receipts"]:
        if not sealed(rec, "retirement_sha256"):
            raise RetireError("a retirement receipt is not sealed")
        if not rec.get("preserved", {}).get("reopening_criterion"):
            raise RetireError("retirement must preserve a reopening criterion")
    return {"ok": True, "collapse_boundary_7B": 2.0, "retired_count": ledger["retired_count"],
            "preserves_reopening": True, "additive_only": True, "ledger_sha256": ledger["ledger_sha256"]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evidence-closed retirement from the empirical harvest.")
    ap.add_argument("--harvest", default=None, help="path to a sealed harvest.json")
    ap.add_argument("--campaign-root", default="/Users/scammermike/Downloads/hawking/reports/condense/doctor_v5_ultra")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True)); sys.exit(0)
    if args.harvest:
        from eco_common import read_json_safe
        harvest = read_json_safe(args.harvest)
    else:
        import succ_harvest
        harvest = succ_harvest.harvest(args.campaign_root)
    ledger = build_retirement_ledger(harvest)
    if args.out:
        from eco_common import atomic_write_json
        atomic_write_json(args.out, ledger)
    print(json.dumps({"schema": ledger["schema"], "ledger_sha256": ledger["ledger_sha256"],
                      "replicated_collapse_boundaries": ledger["replicated_collapse_boundaries"],
                      "retired_count": ledger["retired_count"]}, indent=2, sort_keys=True))
