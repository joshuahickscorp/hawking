#!/usr/bin/env python3.12
"""Receiptized ETA envelope for the active Doctor V5 accelerated generation."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
from typing import Any

import doctor_v5_block_parallel_real_canary as canary
import doctor_v5_production_eta as eta_contract
import doctor_v5_stacked_admission as stacked


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (ROOT / "reports/condense/doctor_v5_ultra/staged_acceleration/"
          "accelerated_eta.json")
SCHEMA = "hawking.doctor_v5_accelerated_eta.v2"
APPENDIX_SECONDS = (86_400, 259_200)
CLAIM_LIMITS = {
    "unavailable-live-schedule": [
        "single-tensor encode evidence is non-transferable",
        "120B and Appendix remain unavailable without segment-specific receipts",
        "mechanical sensitivity is not an ETA and emits no calendar date",
    ],
    "provisional-canary-calibrated-envelope": [
        "single-tensor encode evidence is non-transferable",
        "120B and Appendix remain unavailable without segment-specific receipts",
        "mechanical sensitivity uses speedup 1.0 for both segments and has no dates",
    ],
}
CONFIDENCE = {
    "unavailable-live-schedule": "blocked; no date emitted",
    "provisional-canary-calibrated-envelope": (
        "diagnostic only; single-tensor evidence is non-transferable"
    ),
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _date(now: dt.datetime, seconds: float) -> str:
    return (now + dt.timedelta(seconds=seconds)).isoformat(timespec="seconds")


def _simulation_blockers(simulations: dict[str, dict[str, Any]]) -> list[str]:
    """Return deterministic blockers instead of indexing a refused simulation.

    The scheduler deliberately returns ``{"ok": false, ...}`` while the live
    campaign has a blocked-execution cell.  ETA tooling must preserve that
    refusal rather than crash or manufacture a date from missing fields.
    """
    blockers: list[str] = []
    for label, simulation in sorted(simulations.items()):
        if not isinstance(simulation, dict):
            blockers.append(f"{label}: simulator returned a non-object")
            continue
        if simulation.get("ok") is not True:
            reason = simulation.get("blocker")
            blockers.append(
                f"{label}: {reason if isinstance(reason, str) and reason else 'schedule unavailable'}"
            )
    return blockers


def _empty_diagnostic_sub_120b() -> dict[str, Any]:
    return {
        "available": False,
        "diagnostic_available": False,
        "not_an_eta": True,
        "calendar_dates_emitted": False,
        "seconds_range": None,
        "days_range": None,
        "date_range": None,
        "lower_semantics": None,
        "upper_semantics": None,
    }


def build() -> dict[str, Any]:
    canary_raw = canary.RECEIPT.read_bytes()
    canary_doc = json.loads(canary_raw)
    errors = canary.validate(canary_doc, verify_files=True)
    if errors:
        raise ValueError("invalid real-tensor canary: " + "; ".join(errors))
    speedup = float(canary_doc["speedup"])
    plan, campaign, _observer, bindings = eta_contract._read_inputs()
    evidence = stacked._observed_tier_rss(plan)
    one_lane = stacked.simulate(
        plan, campaign, margin_bytes=stacked.DYNAMIC_BASE_MARGIN_BYTES,
        speedup=speedup, include_unready_120b=False, max_lanes=1, evidence=evidence,
    )
    ram_packed = stacked.simulate(
        plan, campaign, margin_bytes=stacked.DYNAMIC_BASE_MARGIN_BYTES,
        speedup=speedup, include_unready_120b=False, max_lanes=2, evidence=evidence,
    )
    representative = next(cell for cell in plan["cells"]
                          if cell["model_label"] == "120B")
    hypothetical = dict(evidence)
    hypothetical["120B"] = {
        "peak_bytes": stacked._projected_residency(
            representative, stacked.DYNAMIC_BASE_MARGIN_BYTES
        ),
        "samples": 3,
    }
    through_one = stacked.simulate(
        plan, campaign, margin_bytes=stacked.DYNAMIC_BASE_MARGIN_BYTES,
        speedup=1.0, include_unready_120b=True, max_lanes=1,
        evidence=hypothetical,
    )
    through_two = stacked.simulate(
        plan, campaign, margin_bytes=stacked.DYNAMIC_BASE_MARGIN_BYTES,
        speedup=1.0, include_unready_120b=True, max_lanes=2,
        evidence=hypothetical,
    )
    now = dt.datetime.now(dt.timezone.utc)
    speedup_evidence = {
        "receipt_path": str(canary.RECEIPT.resolve()),
        "receipt_sha256": canary_doc["receipt_sha256"],
        "receipt_file_sha256": hashlib.sha256(canary_raw).hexdigest(),
        "single_tensor_encode_speedup": speedup,
        "measurement_scope": (
            "one real Qwen-3B 4bpw tensor encode; one serial run followed by "
            "one parallel run"
        ),
        "transferable_to_campaign_eta": False,
        "transferable_to_gpt_oss_120b": False,
        "exact_output": canary_doc["exact_output"],
        "parallel_peak_rss_bytes": canary_doc["parallel"]["peak_rss_bytes"],
    }
    simulations = {
        "sub120_one_lane": one_lane,
        "sub120_ram_packed": ram_packed,
        "through120_one_lane_scenario": through_one,
        "through120_two_lane_scenario": through_two,
    }
    blockers = _simulation_blockers(simulations)
    if blockers:
        document = {
            "schema": SCHEMA,
            "created_at": now.isoformat(timespec="seconds"),
            "status": "unavailable-live-schedule",
            "eta_scope": "sub-120b-diagnostic-only",
            "calibration_available": True,
            "eta_blocked": True,
            "input_bindings": bindings,
            "blockers": blockers,
            "speedup_evidence": speedup_evidence,
            "sub_120b": _empty_diagnostic_sub_120b(),
            "through_120b": eta_contract._gated_segment(appendix=False),
            "through_120b_plus_appendix": eta_contract._gated_segment(appendix=True),
            "mechanical_sensitivity": eta_contract._mechanical_sensitivity(
                observed_sub_120b_speedup=speedup, through_range=None,
                blockers=blockers,
            ),
            "confidence": CONFIDENCE["unavailable-live-schedule"],
            "claim_limits": CLAIM_LIMITS["unavailable-live-schedule"],
            "quality_or_rigor_discount_applied": False,
            "source_deletion_permitted": False,
        }
        document["eta_sha256"] = _hash_value(document)
        return document
    sub_low, sub_high = sorted((
        float(ram_packed["sub_120b_seconds"]),
        float(one_lane["sub_120b_seconds"]),
    ))
    through_low, through_high = sorted((
        float(through_two["through_120b_seconds"]),
        float(through_one["through_120b_seconds"]),
    ))
    document = {
        "schema": SCHEMA, "created_at": now.isoformat(timespec="seconds"),
        "status": "provisional-canary-calibrated-envelope",
        "eta_scope": "sub-120b-diagnostic-only",
        "calibration_available": True,
        "eta_blocked": False,
        "input_bindings": bindings,
        "speedup_evidence": speedup_evidence,
        "sub_120b": {
            "available": False,
            "diagnostic_available": True,
            "not_an_eta": True,
            "calendar_dates_emitted": False,
            "seconds_range": [sub_low, sub_high],
            "days_range": [sub_low / 86_400, sub_high / 86_400],
            "date_range": None,
            "lower_semantics": "RAM-packed two-lane model; CPU contention omitted",
            "upper_semantics": "one 20-core encoder lane; current safe operating mode",
        },
        "through_120b": eta_contract._gated_segment(appendix=False),
        "through_120b_plus_appendix": eta_contract._gated_segment(appendix=True),
        "mechanical_sensitivity": eta_contract._mechanical_sensitivity(
            observed_sub_120b_speedup=speedup,
            through_range=[through_low, through_high],
        ),
        "confidence": CONFIDENCE["provisional-canary-calibrated-envelope"],
        "claim_limits": CLAIM_LIMITS["provisional-canary-calibrated-envelope"],
        "quality_or_rigor_discount_applied": False,
        "source_deletion_permitted": False,
    }
    document["eta_sha256"] = _hash_value(document)
    return document


def validate(document: Any, *, verify_freshness: bool = True) -> list[str]:
    if not isinstance(document, dict):
        return ["accelerated ETA must be an object"]
    errors: list[str] = []
    if document.get("schema") != SCHEMA:
        errors.append("accelerated ETA schema differs")
    payload = {key: value for key, value in document.items() if key != "eta_sha256"}
    try:
        sealed = document.get("eta_sha256") == _hash_value(payload)
    except (TypeError, ValueError):
        sealed = False
    if not sealed:
        errors.append("accelerated ETA hash differs")
    status = document.get("status")
    common = {
        "schema", "created_at", "status", "eta_scope",
        "calibration_available", "eta_blocked", "input_bindings",
        "speedup_evidence", "sub_120b", "through_120b",
        "through_120b_plus_appendix", "mechanical_sensitivity",
        "confidence", "claim_limits", "quality_or_rigor_discount_applied",
        "source_deletion_permitted", "eta_sha256",
    }
    expected = {
        "unavailable-live-schedule": common | {"blockers"},
        "provisional-canary-calibrated-envelope": common,
    }.get(status)
    if expected is None:
        errors.append("accelerated ETA status differs")
    elif set(document) != expected:
        errors.append("accelerated ETA top-level shape is not closed-world")
    try:
        created = dt.datetime.fromisoformat(document.get("created_at", ""))
    except (TypeError, ValueError):
        created = None
    if created is None or created.tzinfo is None:
        errors.append("accelerated ETA creation time is invalid")
    if document.get("eta_scope") != "sub-120b-diagnostic-only" \
            or document.get("calibration_available") is not True \
            or document.get("quality_or_rigor_discount_applied") is not False \
            or document.get("source_deletion_permitted") is not False:
        errors.append("accelerated ETA claim boundary differs")
    if status in CLAIM_LIMITS and (
            document.get("claim_limits") != CLAIM_LIMITS[status]
            or document.get("confidence") != CONFIDENCE[status]
    ):
        errors.append("accelerated ETA claim limits differ")
    errors.extend(eta_contract._input_binding_errors(
        document.get("input_bindings"), verify_freshness=verify_freshness
    ))
    evidence = document.get("speedup_evidence")
    evidence_keys = {
        "receipt_path", "receipt_sha256", "receipt_file_sha256",
        "single_tensor_encode_speedup",
        "measurement_scope", "transferable_to_campaign_eta",
        "transferable_to_gpt_oss_120b", "exact_output",
        "parallel_peak_rss_bytes",
    }
    speedup = evidence.get("single_tensor_encode_speedup") \
        if isinstance(evidence, dict) else None
    if not isinstance(evidence, dict) or set(evidence) != evidence_keys \
            or not eta_contract._valid_sha256(evidence.get("receipt_sha256")) \
            or not eta_contract._valid_sha256(evidence.get("receipt_file_sha256")) \
            or not isinstance(evidence.get("receipt_path"), str) \
            or not evidence.get("receipt_path") \
            or not isinstance(evidence.get("measurement_scope"), str) \
            or not evidence.get("measurement_scope") \
            or evidence.get("transferable_to_campaign_eta") is not False \
            or evidence.get("transferable_to_gpt_oss_120b") is not False \
            or evidence.get("exact_output") is not True \
            or isinstance(speedup, bool) or not isinstance(speedup, (int, float)) \
            or not math.isfinite(float(speedup)) or float(speedup) <= 1 \
            or isinstance(evidence.get("parallel_peak_rss_bytes"), bool) \
            or not isinstance(evidence.get("parallel_peak_rss_bytes"), int) \
            or evidence.get("parallel_peak_rss_bytes") <= 0:
        errors.append("single-tensor encode evidence boundary is invalid")
    if verify_freshness and isinstance(evidence, dict):
        if evidence.get("receipt_path") != str(canary.RECEIPT.resolve()):
            errors.append("single-tensor receipt path is non-canonical")
        else:
            try:
                current, reference = eta_contract._read_bound_json(canary.RECEIPT)
                current_errors = canary.validate(current, verify_files=False)
            except (OSError, ValueError, eta_contract.ProductionEtaError) as exc:
                errors.append(f"single-tensor receipt cannot be verified: {exc}")
            else:
                if current_errors \
                        or reference["sha256"] != evidence.get("receipt_file_sha256") \
                        or current.get("receipt_sha256") != evidence.get("receipt_sha256") \
                        or current.get("speedup") != evidence.get(
                            "single_tensor_encode_speedup"
                        ) \
                        or current.get("exact_output") is not True \
                        or current.get("parallel", {}).get("peak_rss_bytes") \
                        != evidence.get("parallel_peak_rss_bytes"):
                    errors.append("single-tensor receipt binding is stale or invalid")
    errors.extend(eta_contract._gated_segment_errors(
        document.get("through_120b"), appendix=False
    ))
    errors.extend(eta_contract._gated_segment_errors(
        document.get("through_120b_plus_appendix"), appendix=True
    ))
    errors.extend(eta_contract._mechanical_errors(
        document.get("mechanical_sensitivity"), calibration_available=True
    ))
    sub = document.get("sub_120b")
    if status == "unavailable-live-schedule":
        blockers = document.get("blockers")
        if document.get("eta_blocked") is not True \
                or not isinstance(blockers, list) or not blockers \
                or any(not isinstance(row, str) or not row for row in blockers) \
                or blockers != sorted(set(blockers)) \
                or not isinstance(sub, dict) \
                or set(sub) != {
                    "available", "diagnostic_available", "not_an_eta",
                    "calendar_dates_emitted", "seconds_range", "days_range",
                    "date_range", "lower_semantics", "upper_semantics",
                } \
                or sub.get("available") is not False \
                or sub.get("diagnostic_available") is not False \
                or sub.get("not_an_eta") is not True \
                or sub.get("calendar_dates_emitted") is not False \
                or any(sub.get(field) is not None for field in (
                    "seconds_range", "days_range", "date_range",
                    "lower_semantics", "upper_semantics",
                )):
            errors.append("unavailable accelerated ETA contract is invalid")
    elif status == "provisional-canary-calibrated-envelope":
        sub_keys = {
            "available", "diagnostic_available", "not_an_eta",
            "calendar_dates_emitted", "seconds_range", "days_range",
            "date_range", "lower_semantics", "upper_semantics",
        }
        seconds = sub.get("seconds_range") if isinstance(sub, dict) else None
        days = sub.get("days_range") if isinstance(sub, dict) else None
        if document.get("eta_blocked") is not False \
                or not isinstance(sub, dict) or set(sub) != sub_keys \
                or sub.get("available") is not False \
                or sub.get("diagnostic_available") is not True \
                or sub.get("not_an_eta") is not True \
                or sub.get("calendar_dates_emitted") is not False \
                or not eta_contract._valid_pair(seconds) \
                or not eta_contract._valid_pair(days) \
                or not all(math.isclose(float(day), float(second) / 86_400,
                                        rel_tol=1e-12, abs_tol=1e-12)
                           for day, second in zip(days or [], seconds or [], strict=True)) \
                or sub.get("date_range") is not None \
                or not isinstance(sub.get("lower_semantics"), str) \
                or not isinstance(sub.get("upper_semantics"), str):
            errors.append("diagnostic sub-120B range is invalid")
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--status", action="store_true",
        help="print a fresh read-only ETA document without writing it",
    )
    mode.add_argument(
        "--write", type=Path, metavar="PATH",
        help="write the fresh document atomically to PATH",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = build()
    errors = validate(document, verify_freshness=True)
    if errors:
        raise ValueError("invalid accelerated ETA: " + "; ".join(errors))
    if not args.status:
        _atomic_json(args.write or OUTPUT, document)
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
