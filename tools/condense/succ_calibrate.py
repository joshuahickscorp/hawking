#!/usr/bin/env python3.12
"""Precompile a per-parent post-release calibration program (master goal section 9).

Once a parent has terminal evidence (e.g. the 72B codec_control cell has sealed), this
reconstructs its untreated frontier, brackets the nearest unresolved boundary, and
materializes the SMALLEST set of source-bound experiments that could change the selected
EXTREME artifact. It launches NOTHING: the program is bound to the legacy release boundary
and executes only after the successor transition fires. It is the State-B deliverable
"72B post-release calibration precompiled from available evidence".

Honesty: the campaign runs with quality_claims_permitted=false, and for 72B the resident
quality eval is disk/RAM-gated (deferred), so the sealed evidence is PHYSICAL bytes plus a
deferred-quality note. The program records that the first required experiment is the
deferred full-model quality evaluation, then the lower-rate boundary probes.
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

CALIBRATION_SCHEMA = "hawking.successor.calibration_program.v1"

# family per model label (qwen2.5-dense spine; gpt-oss-moe for 120B)
_FAMILY = {"0.5B": "qwen2.5-dense", "1.5B": "qwen2.5-dense", "3B": "qwen2.5-dense",
           "7B": "qwen2.5-dense", "14B": "qwen2.5-dense", "32B": "qwen2.5-dense",
           "72B": "qwen2.5-dense", "120B": "gpt-oss-moe"}


class CalibrationError(EcoError):
    """Fail-closed calibration error."""


def _parent_of(plan: dict[str, Any], label: str) -> dict[str, Any] | None:
    for a in plan.get("parents", []):
        if a["binding"]["model_label"] == label:
            return a
    return None


def _untreated_frontier(ledger: dict[str, Any], label: str) -> list[dict[str, Any]]:
    """The codec_control (zero-treatment) sealed physical bytes per rate for this parent."""
    rows = []
    for cell in ledger.get("cells", []):
        if cell.get("model_label") != label or cell.get("branch") != "codec_control":
            continue
        if cell.get("status") != "complete":
            continue
        phys = cell.get("physical") or {}
        qual = cell.get("quality_provisional") or {}
        rows.append({
            "rate_bpw": cell.get("rate_bpw"),
            "all_in_model_payload_bpw": phys.get("all_in_model_payload_bpw"),
            "target_met": phys.get("target_met"),
            "ppl_relative_delta": qual.get("ppl_relative_delta"),
            "quality_status": "measured" if qual.get("ppl_relative_delta") is not None
                              else "deferred_disk_ram_gated",
            "result_sha256": cell.get("result_sha256"),
            "cell_identity_sha256": cell.get("cell_identity_sha256"),
        })
    rows.sort(key=lambda r: -(r["rate_bpw"] or 0))
    return rows


def build_calibration(label: str, *, campaign_root: str | None = None,
                      admission: dict[str, Any] | None = None) -> dict[str, Any]:
    import eco_import, eco_planner, succ_engine

    ledger = eco_import.build_ledger(eco_import.default_config(campaign_root))
    plan = eco_planner.build_plan(ledger)
    parent = _parent_of(plan, label)
    if parent is None:
        raise CalibrationError(f"{label} has no terminal evidence yet; cannot calibrate "
                               f"(awaiting: {[a['model_label'] for a in plan['parents_awaiting_evidence']]})")

    family = _FAMILY.get(label, "unknown")
    if admission is None:
        try:
            import succ_admission
            admission = succ_admission.admit(family, label)
        except Exception as exc:  # noqa: BLE001 - degrade honestly
            admission = {"adapter_id": None, "ready_for_execution": False,
                         "blockers": [f"probe_error:{type(exc).__name__}"]}

    untreated = _untreated_frontier(ledger, label)
    source_anchor = untreated[0]["result_sha256"] if untreated else None

    # ordered experiments: first the deferred full-model quality eval on the sealed physical
    # cells (F3), then the boundary probes the planner needs, materialized source-bound.
    experiments: list[dict[str, Any]] = []
    for row in untreated:
        if row["quality_status"].startswith("deferred"):
            experiments.append({
                "kind": "deferred_full_model_quality_eval",
                "rate_bpw": row["rate_bpw"], "fidelity_tier": "F3",
                "on_sealed_physical": row["result_sha256"],
                "note": "resident BF16 reconstruction + capability eval; disk/RAM-gated, "
                        "runs post-release when the box is free of the legacy campaign",
            })
    for probe in parent.get("boundary_probes_needed", []):
        experiments.append({
            "kind": "boundary_probe", "rate_bpw": probe["rate_bpw"],
            "current_verdict": probe["current_verdict"],
            "fidelity_tier": probe.get("next_feasibility_tier"),
            "doctor_program": probe.get("doctor_program"),
        })

    # materialize the single highest-value next experiment as a source-bound program
    pick = succ_engine.next_experiment({"parents": [parent]})
    materialized = None
    if pick is not None:
        materialized = succ_engine.materialize_program(
            pick, admission, source_manifest_sha256=source_anchor,
            controls=["zero_treatment", "equal_byte_codec", "smaller_higher_bit", "public_same_byte"])

    program = {
        "schema": CALIBRATION_SCHEMA,
        "model_label": label,
        "family": family,
        "generated_at": now_iso(),
        "campaign_plan_sha256": ledger.get("campaign_plan_sha256"),
        "prior_ledger_sha256": ledger.get("ledger_sha256"),
        "params_b": parent.get("params_b"),
        "byte_ceiling_bpw": parent.get("byte_ceiling_bpw"),
        "contract": parent.get("contract"),
        "untreated_frontier": untreated,
        "event_horizon_bracket": parent.get("event_horizon_bracket"),
        "scaling_prior": plan.get("scaling_prior", {}).get("trend"),
        "extreme_status": parent.get("extreme_candidate"),
        "lower_rate_reasons": parent.get("lower_rate_reasons"),
        "ordered_experiments": experiments,
        "next_experiment_materialized": materialized,
        "adapter": {"adapter_id": admission.get("adapter_id"),
                    "ready_for_execution": admission.get("ready_for_execution"),
                    "execution_capable": admission.get("execution_capable"),
                    "blockers": admission.get("blockers")},
        "release_binding": {
            "executes": "post_release_only",
            "gate": "the successor transition must fire (all legacy cells terminal, reports "
                    "sealed, quiescent, operator-signed) before any experiment launches",
            "non_interference": "this program launches nothing while the legacy campaign runs",
        },
    }
    return seal_field(program, "program_sha256")


def selftest() -> dict[str, Any]:
    """Synthetic ledger with a sealed 72B codec_control (deferred quality) -> a program."""
    import eco_planner

    def _cell(label, rate, branch, status, ppl=None, phys=None):
        return {"model_label": label, "model_family": "qwen2.5-dense", "hf_id": "toy/72b",
                "exact_stored_parameter_count": int(72.7 * 1e9), "nominal_params_b": 72.7,
                "rate_id": str(rate), "rate_bpw": rate, "branch": branch, "status": status,
                "result_sha256": "a" * 64, "cell_identity_sha256": "b" * 64,
                "physical": {"all_in_model_payload_bpw": phys, "target_physical_bpw": rate,
                             "target_met": False},
                "quality_provisional": {"ppl_relative_delta": ppl,
                                        "capability_absolute_delta": None,
                                        "quality_claims_permitted": False}}
    ledger = {"schema": "hawking.eco.prior_ledger.v1", "campaign_plan_sha256": "c" * 64,
              "ledger_sha256": "d" * 64, "cohort": [],
              "cells": [_cell("72B", 4.0, "codec_control", "complete", ppl=None, phys=5.1)]}

    # monkeypatch eco_import.build_ledger to return our synthetic ledger
    import eco_import
    real = eco_import.build_ledger
    eco_import.build_ledger = lambda cfg: ledger  # type: ignore[assignment]
    try:
        prog = build_calibration("72B", admission={"adapter_id": "doctor-v5-strand-ladder-qwen25-dense",
                                                   "ready_for_execution": False,
                                                   "execution_capable": True,
                                                   "blockers": ["review_flag_absent"]})
    finally:
        eco_import.build_ledger = real  # type: ignore[assignment]

    if not sealed(prog, "program_sha256"):
        raise CalibrationError("program not sealed")
    if prog["model_label"] != "72B":
        raise CalibrationError("wrong model")
    # the deferred quality eval must be the first experiment for the sealed physical cell
    kinds = [e["kind"] for e in prog["ordered_experiments"]]
    if "deferred_full_model_quality_eval" not in kinds:
        raise CalibrationError(f"deferred eval missing: {kinds}")
    if prog["release_binding"]["executes"] != "post_release_only":
        raise CalibrationError("not release-bound")
    if prog["untreated_frontier"][0]["quality_status"] != "deferred_disk_ram_gated":
        raise CalibrationError("untreated frontier should mark 72B quality as deferred")
    return {"ok": True, "model": "72B", "experiments": len(prog["ordered_experiments"]),
            "untreated_rows": len(prog["untreated_frontier"]),
            "release_bound": True, "program_sha256": prog["program_sha256"]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Precompile a post-release calibration program.")
    ap.add_argument("--model", default="72B")
    ap.add_argument("--campaign-root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True)); sys.exit(0)
    prog = build_calibration(args.model, campaign_root=args.campaign_root)
    if args.out:
        from eco_common import atomic_write_json
        atomic_write_json(args.out, prog)
    summary = {k: prog[k] for k in ("schema", "model_label", "params_b", "byte_ceiling_bpw",
                                    "event_horizon_bracket", "extreme_status", "program_sha256")}
    summary["experiments"] = len(prog["ordered_experiments"])
    summary["untreated_frontier"] = prog["untreated_frontier"]
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
