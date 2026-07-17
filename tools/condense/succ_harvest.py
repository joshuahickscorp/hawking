#!/usr/bin/env python3.12
"""Seal the empirical evidence: harvest every terminal legacy cell into one immutable,
hash-bound dataset with the full measured field set (master goal "First: recover and seal
the empirical evidence").

Read-only. For every TERMINAL cell it joins result.json + execution_receipt.json (and the
plan binding) and records: exact parent identity + parameter/tensor geometry, nominal
target BPW, exact physical whole-artifact BPW, branch/treatment identity, wall time (from
the receipt resource_observations timestamps), disk delta, memory-pressure/thermal/power,
lifecycle, perplexity + capability deltas, quality-evidence level, a failure/deferral
CLASSIFICATION that never relabels a scheduling disposition as scientific collapse,
treatment-vs-equal-byte-control comparison, dominance, and transfer validity.

The dataset is self-sealed and every negative result stays queryable. This is the rich
empirical layer the ETA model, retirement receipts, and the adaptive selector consume.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, seal_field, sealed, now_iso, read_json_safe, CAMPAIGN_PLAN_SHA256,
)

HARVEST_SCHEMA = "hawking.successor.empirical_harvest.v1"
RESULT_SCHEMA = "hawking.doctor_v5_adapter_result.v1"
DISPOSITION_SCHEMA = "hawking.doctor_v5_ultra_disposition.v1"
TERMINAL = frozenset({"complete", "negative", "unsupported"})

# Diagnosis / classification vocabulary (never conflate a deferral with collapse).
FAILURE_CLASSES = (
    "measured_no_material_damage", "measured_quality_failure", "measured_computation_collapse",
    "treatment_failure", "physical_rate_failure", "runtime_failure", "scheduling_deferral",
    "missing_evaluation", "unsupported_adapter", "inconclusive",
)

# gates (the campaign's own promotion gate)
PPL_GOOD = 0.08
CAP_GOOD = -0.05
COLLAPSE_PPL = 1.0
COLLAPSE_CAP = -0.30


def _parse_iso(ts: Any) -> _dt.datetime | None:
    if not isinstance(ts, str):
        return None
    try:
        return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _wall_seconds(receipt: dict[str, Any]) -> float | None:
    ro = receipt.get("resource_observations", {}) if isinstance(receipt, dict) else {}
    before = _parse_iso((ro.get("before") or {}).get("sampled_at"))
    after = _parse_iso((ro.get("after") or {}).get("sampled_at"))
    if before and after:
        return round((after - before).total_seconds(), 3)
    # fall back to completed_at only (no start) -> unknown
    return None


def _classify_complete(ppl: float | None, cap: float | None, physical: dict[str, Any]) -> str:
    if ppl is None:
        return "missing_evaluation"  # physical sealed, quality deferred/unmeasured
    ppl_ok = ppl <= PPL_GOOD
    cap_ok = cap is None or cap >= CAP_GOOD
    if ppl_ok and cap_ok:
        return "measured_no_material_damage"
    if ppl > COLLAPSE_PPL or (cap is not None and cap <= COLLAPSE_CAP):
        return "measured_computation_collapse"
    return "measured_quality_failure"


def _classify_disposition(reason_code: str | None) -> str:
    rc = (reason_code or "").lower()
    if "adaptive-defer" in rc or "scheduling" in rc or "defer" in rc:
        return "scheduling_deferral"
    if "collapse" in rc or "cliff" in rc:
        # a quality-cliff disposition is a MEASURED boundary, but the disposition itself is a
        # deferral of the cell; record as scheduling_deferral with the cliff evidence preserved.
        return "scheduling_deferral"
    if "unsupported" in rc or "adapter" in rc:
        return "unsupported_adapter"
    if "runtime" in rc or "exec" in rc:
        return "runtime_failure"
    return "inconclusive"


def harvest(campaign_root: str, *, expected_plan: str = CAMPAIGN_PLAN_SHA256) -> dict[str, Any]:
    root = Path(campaign_root)
    plan = read_json_safe(root / "campaign_plan.json")
    if plan.get("plan_sha256") != expected_plan:
        raise EcoError(f"plan_sha256 {plan.get('plan_sha256')} != pinned {expected_plan}")
    index = {c.get("cell_id"): c for c in plan.get("cells", []) if isinstance(c, dict)}
    queue = read_json_safe(root / "queue_state.json")
    if queue.get("plan_sha256") != plan.get("plan_sha256"):
        raise EcoError("queue_state plan_sha256 mismatch")
    cells = queue.get("cells", {})

    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for cid, qrow in cells.items():
        status = qrow.get("status")
        if status not in TERMINAL:
            skipped[str(status)] = skipped.get(str(status), 0) + 1
            continue
        meta = index.get(cid, {})
        try:
            if status == "complete":
                rows.append(_row_complete(cid, meta, root))
            else:
                rows.append(_row_disposition(cid, meta, status, root))
        except EcoError as exc:
            rows.append({"cell_id": cid, "status": status, "harvest_error": str(exc)})

    # dominance + treatment-vs-control are cross-row, computed after all rows exist
    _annotate_dominance_and_treatment(rows)

    dataset = {
        "schema": HARVEST_SCHEMA,
        "campaign_plan_sha256": plan.get("plan_sha256"),
        "harvested_at": now_iso(),
        "source_campaign_root": str(root),
        "terminal_rows": len(rows),
        "skipped_nonterminal": skipped,
        "classification_counts": _counts(rows, "failure_class"),
        "rows": rows,
    }
    return seal_field(dataset, "harvest_sha256")


def _row_complete(cid: str, meta: dict[str, Any], root: Path) -> dict[str, Any]:
    result = read_json_safe(root / "results" / cid / "result.json")
    if result.get("schema") != RESULT_SCHEMA or result.get("status") != "complete":
        raise EcoError("bad result schema/status")
    seal_ok = sealed(result, "result_sha256")
    m = result.get("metrics", {})
    phys = m.get("physical_accounting", {}) or {}
    qual = m.get("quality_observation", {}) or {}
    ppl = (qual.get("ppl") or {}).get("relative_delta")
    cap = (qual.get("capability") or {}).get("absolute_delta")
    pa = m.get("parameter_accounting", {}) or {}
    lc = m.get("lifecycle", {}) or {}
    target = phys.get("target_physical_bpw")
    actual = phys.get("all_in_model_payload_bpw")

    receipt = {}
    rp = root / "results" / cid / "execution_receipt.json"
    if rp.exists():
        try:
            receipt = read_json_safe(rp)
        except EcoError:
            receipt = {}
    ro = receipt.get("resource_observations", {}) if isinstance(receipt, dict) else {}
    before = ro.get("before", {}) or {}
    after = ro.get("after", {}) or {}

    quality_level = "measured" if ppl is not None else "deferred_disk_ram_gated"
    return {
        "cell_id": cid, "status": "complete",
        "model_label": meta.get("model_label"), "family": meta.get("model_family"),
        "hf_id": meta.get("hf_id"),
        "exact_stored_parameter_count": pa.get("stored_parameters") or meta.get("exact_stored_parameter_count"),
        "nominal_params_b": meta.get("nominal_params_b"),
        "rate_id": meta.get("rate_id"), "nominal_target_bpw": target, "branch": meta.get("branch"),
        "physical_bpw": actual,
        "packed_2d_tensor_bpw": phys.get("packed_2d_tensor_bpw"),
        "passthrough_bytes": phys.get("lossless_non_2d_passthrough_bytes"),
        "metadata_bytes": phys.get("metadata_bytes_excluded_from_model_bpw"),
        "model_payload_bytes": phys.get("model_payload_bytes"),
        "target_met": phys.get("target_met"),
        "realization_gap": round(actual / target, 4) if (isinstance(actual, (int, float))
                                                         and isinstance(target, (int, float)) and target) else None,
        "geometry": {"stored_parameters": pa.get("stored_parameters"),
                     "quantized_parameters": pa.get("quantized_parameters"),
                     "quantized_tensors": pa.get("quantized_tensors"),
                     "passthrough_tensors": pa.get("passthrough_tensors"),
                     "packed_shard_count": lc.get("packed_shard_count")},
        "ppl_relative_delta": ppl, "capability_absolute_delta": cap,
        "quality_evidence_level": quality_level,
        "wall_seconds": _wall_seconds(receipt),
        "resources": {
            "disk_free_before_bytes": before.get("disk_free_bytes"),
            "disk_free_after_bytes": after.get("disk_free_bytes"),
            "memory_pressure_before": before.get("memory_pressure_level"),
            "memory_pressure_after": after.get("memory_pressure_level"),
            "swap_used_after_bytes": after.get("swap_used_bytes"),
            "thermal_nominal": after.get("thermal_nominal", before.get("thermal_nominal")),
            "ac_power": before.get("ac_power"),
            "note": "peak RSS and per-phase timing are not in the receipt; derivable from "
                    "child_resources.jsonl / phase logs (not harvested here to stay light)",
        },
        "phase_unit_order": (receipt.get("phase_resume") or {}).get("unit_order"),
        "failure_class": _classify_complete(ppl, cap, phys),
        "quality_claims_permitted": qual.get("quality_claims_permitted", False),
        "transfer_valid_as_scheduling_prior": True,
        "transfer_valid_as_quality_proof": False,  # Pass B forbids quality claims
        "seal_ok": seal_ok, "result_sha256": result.get("result_sha256"),
        "cell_identity_sha256": meta.get("cell_identity_sha256"),
    }


def _row_disposition(cid: str, meta: dict[str, Any], status: str, root: Path) -> dict[str, Any]:
    dp = root / "dispositions" / f"{cid}.json"
    disp = read_json_safe(dp) if dp.exists() else {}
    seal_ok = sealed(disp, "disposition_sha256") if disp else False
    reason = disp.get("reason_code")
    return {
        "cell_id": cid, "status": status,
        "model_label": meta.get("model_label"), "family": meta.get("model_family"),
        "rate_id": meta.get("rate_id"), "nominal_target_bpw": meta.get("rate_bpw"),
        "branch": meta.get("branch"),
        "reason_code": reason, "detail": disp.get("detail"),
        "failure_class": _classify_disposition(reason),
        "physical_bpw": None, "ppl_relative_delta": None,
        "quality_evidence_level": "none_disposition",
        "transfer_valid_as_scheduling_prior": True,
        "transfer_valid_as_quality_proof": False,
        "evidence_artifact_count": len(disp.get("evidence_artifacts") or []),
        "seal_ok": seal_ok, "disposition_sha256": disp.get("disposition_sha256"),
        "cell_identity_sha256": meta.get("cell_identity_sha256"),
    }


def _annotate_dominance_and_treatment(rows: list[dict[str, Any]]) -> None:
    # group complete rows by model
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") == "complete" and isinstance(r.get("physical_bpw"), (int, float)):
            by_model.setdefault(r.get("model_label"), []).append(r)
    for r in rows:
        r["dominated"] = False
        r["dominated_by"] = None
        r["treatment_vs_control"] = None
    for model, group in by_model.items():
        # treatment vs equal-rate codec_control (did doctor improve ppl over control?)
        controls = {(g.get("rate_id")): g for g in group if g.get("branch") == "codec_control"}
        for r in group:
            if r.get("branch") != "codec_control":
                ctrl = controls.get(r.get("rate_id"))
                if ctrl and isinstance(r.get("ppl_relative_delta"), (int, float)) \
                        and isinstance(ctrl.get("ppl_relative_delta"), (int, float)):
                    delta = round(ctrl["ppl_relative_delta"] - r["ppl_relative_delta"], 6)
                    r["treatment_vs_control"] = {
                        "control_ppl": ctrl["ppl_relative_delta"], "treated_ppl": r["ppl_relative_delta"],
                        "ppl_improvement": delta, "improved": delta > 0}
            # dominance: another row same model with <= physical AND <= ppl (both known)
            for other in group:
                if other is r:
                    continue
                op, rp_ = other.get("physical_bpw"), r.get("physical_bpw")
                oq, rq = other.get("ppl_relative_delta"), r.get("ppl_relative_delta")
                if all(isinstance(x, (int, float)) for x in (op, rp_, oq, rq)):
                    if op <= rp_ and oq <= rq and (op < rp_ or oq < rq):
                        r["dominated"] = True
                        r["dominated_by"] = other.get("cell_id")
                        break


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        v = str(r.get(key))
        out[v] = out.get(v, 0) + 1
    return out


def eta_observations(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Map sealed harvest rows to succ_eta runtime observations (real measured wall times).
    phase is the whole cell (per-phase timing is not in the receipt), so the ETA model fits
    per-(branch, full_cell) seconds-per-billion, never one global constant."""
    obs = []
    for r in dataset.get("rows", []):
        if r.get("status") != "complete":
            continue
        wall = r.get("wall_seconds")
        params = (r.get("geometry") or {}).get("stored_parameters") or r.get("exact_stored_parameter_count")
        if not isinstance(wall, (int, float)) or not isinstance(params, (int, float)) or params <= 0:
            continue
        obs.append({"branch": r.get("branch"), "phase": "full_cell",
                    "rate": r.get("nominal_target_bpw"), "residency": "resident",
                    "stored_params_b": params / 1e9, "wall_seconds": wall})
    return obs


def selftest() -> dict[str, Any]:
    import tempfile
    from eco_common import atomic_write_json, hash_value
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "doctor_v5_ultra"
        (root / "results" / "m-7b__4bpw__codec-control").mkdir(parents=True)
        (root / "results" / "m-7b__4bpw__doctor-static").mkdir(parents=True)
        idc = hash_value({"c": "ctrl"}); idt = hash_value({"c": "treat"})
        plan = seal_field({"schema": "hawking.doctor_v5_ultra_campaign_plan.v1", "cells": [
            {"cell_id": "m-7b__4bpw__codec-control", "model_label": "7B", "model_family": "qwen2.5-dense",
             "rate_id": "4", "rate_bpw": 4.0, "branch": "codec_control", "cell_identity_sha256": idc,
             "nominal_params_b": 7.6},
            {"cell_id": "m-7b__4bpw__doctor-static", "model_label": "7B", "model_family": "qwen2.5-dense",
             "rate_id": "4", "rate_bpw": 4.0, "branch": "doctor_static", "cell_identity_sha256": idt,
             "nominal_params_b": 7.6}]}, "plan_sha256")
        atomic_write_json(root / "campaign_plan.json", plan)
        atomic_write_json(root / "queue_state.json", {"plan_sha256": plan["plan_sha256"], "cells": {
            "m-7b__4bpw__codec-control": {"status": "complete"},
            "m-7b__4bpw__doctor-static": {"status": "complete"}}})

        def _res(cid, ppl, phys, idh, wall_before, wall_after):
            r = seal_field({"schema": RESULT_SCHEMA, "status": "complete", "metrics": {
                "campaign_cell": {"cell_id": cid}, "parameter_accounting": {"stored_parameters": 7_600_000_000,
                    "quantized_parameters": 7_500_000_000, "quantized_tensors": 200, "passthrough_tensors": 50},
                "physical_accounting": {"all_in_model_payload_bpw": phys, "target_physical_bpw": 4.0,
                    "target_met": phys <= 4.0, "packed_2d_tensor_bpw": phys - 0.01},
                "quality_observation": {"ppl": {"relative_delta": ppl}, "capability": {"absolute_delta": -0.02},
                    "quality_claims_permitted": False}, "lifecycle": {"packed_shard_count": 4}}}, "result_sha256")
            atomic_write_json(root / "results" / cid / "result.json", r)
            er = seal_field({"schema": "x", "resource_observations": {
                "before": {"sampled_at": wall_before, "disk_free_bytes": 200, "memory_pressure_level": "normal",
                           "thermal_nominal": True, "ac_power": True},
                "after": {"sampled_at": wall_after, "disk_free_bytes": 150, "memory_pressure_level": "normal",
                          "swap_used_bytes": 0, "thermal_nominal": True}},
                "phase_resume": {"unit_order": ["preflight", "encode", "receipt"]}}, "receipt_sha256")
            atomic_write_json(root / "results" / cid / "execution_receipt.json", er)

        _res("m-7b__4bpw__codec-control", 0.20, 5.1, idc, "2026-07-17T00:00:00+00:00", "2026-07-17T01:00:00+00:00")
        _res("m-7b__4bpw__doctor-static", 0.06, 5.2, idt, "2026-07-17T00:00:00+00:00", "2026-07-17T02:30:00+00:00")

        ds = harvest(str(root), expected_plan=plan["plan_sha256"])
        if not sealed(ds, "harvest_sha256") or ds["terminal_rows"] != 2:
            raise EcoError("harvest not sealed / wrong count")
        rows = {r["cell_id"]: r for r in ds["rows"]}
        ctrl = rows["m-7b__4bpw__codec-control"]; treat = rows["m-7b__4bpw__doctor-static"]
        if ctrl["wall_seconds"] != 3600.0 or treat["wall_seconds"] != 9000.0:
            raise EcoError(f"wall time wrong: {ctrl['wall_seconds']} {treat['wall_seconds']}")
        if treat["treatment_vs_control"]["improved"] is not True:
            raise EcoError("treatment should improve ppl over control")
        if ctrl["failure_class"] != "measured_quality_failure":
            raise EcoError(f"control class wrong: {ctrl['failure_class']}")
        if treat["failure_class"] != "measured_quality_failure":  # 0.06 <= 0.08 ppl but cap -0.02 ok -> no_material?
            pass  # treat ppl 0.06 <= 0.08 and cap -0.02 >= -0.05 -> measured_no_material_damage
        if treat["failure_class"] != "measured_no_material_damage":
            raise EcoError(f"treated class wrong: {treat['failure_class']}")
        # dominance: control (phys 5.1, ppl 0.20) is dominated by treat (phys 5.2? no, 5.2>5.1)
        # neither strictly dominates (control cheaper physically, treat better quality) -> both undominated
        if ctrl["dominated"] or treat["dominated"]:
            raise EcoError("neither should be dominated (pareto tradeoff)")
    return {"ok": True, "terminal_rows": 2, "wall_time_from_receipt": True,
            "treatment_vs_control": True, "classification": True, "dominance": True,
            "harvest_sha256": ds["harvest_sha256"]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Seal the empirical evidence (rich harvest).")
    ap.add_argument("--campaign-root", default="/Users/scammermike/Downloads/hawking/reports/condense/doctor_v5_ultra")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True)); sys.exit(0)
    ds = harvest(args.campaign_root)
    if args.out:
        from eco_common import atomic_write_json
        atomic_write_json(args.out, ds)
    print(json.dumps({"schema": ds["schema"], "harvest_sha256": ds["harvest_sha256"],
                      "terminal_rows": ds["terminal_rows"],
                      "classification_counts": ds["classification_counts"],
                      "skipped_nonterminal": ds["skipped_nonterminal"]}, indent=2, sort_keys=True))
