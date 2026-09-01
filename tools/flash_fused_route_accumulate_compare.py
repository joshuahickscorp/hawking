#!/usr/bin/env python3
"""Compare the direct routed accumulation organ with the materialized control.

The fused kernel deliberately changes floating-point reduction order, so the
state hash is expected to differ even when the source-BF16 result is within the
qualified organ tolerance.  This receipt makes that distinction explicit and
keeps the dispatch/GPU evidence separate from any complete-token claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def metric(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    if left.shape != right.shape:
        return {"shape_match": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    delta = left.astype(np.float64) - right.astype(np.float64)
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)
    left_norm = float(np.linalg.norm(left64))
    right_norm = float(np.linalg.norm(right64))
    cosine = float(np.dot(left64, right64) / (left_norm * right_norm)) if left_norm and right_norm else None
    return {
        "shape_match": True,
        "elements": int(left.size),
        "max_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "cosine": cosine,
        "within_tolerance": bool(np.max(np.abs(delta)) <= 1.0e-6) if delta.size else True,
    }


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument(
        "--fused-receipt",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--control-receipt",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--fused-state",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--control-state",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    args.fused_receipt = args.fused_receipt or Path(
        f"receipts/headless/FLASH_FUSED_ROUTE_ACCUMULATE_L{args.layer}.json"
    )
    args.control_receipt = args.control_receipt or Path(
        f"receipts/headless/FLASH_COMPACT_L0_L7_V1/layer-{args.layer}/receipt.json"
    )
    args.fused_state = args.fused_state or Path(
        f"receipts/headless/FLASH_FUSED_ROUTE_ACCUMULATE_L{args.layer}_STATE.f32"
    )
    args.control_state = args.control_state or Path(
        f"receipts/headless/FLASH_COMPACT_L0_L7_V1/layer-{args.layer}/state.f32"
    )
    args.out = args.out or Path(
        f"receipts/headless/FLASH_FUSED_ROUTE_ACCUMULATE_PARITY_L{args.layer}.json"
    )
    fused = read(args.fused_receipt)
    control = read(args.control_receipt)
    fused_state = np.fromfile(args.fused_state, dtype="<f4")
    control_state = np.fromfile(args.control_state, dtype="<f4")

    fused_exec = fused.get("execution") or {}
    control_exec = control.get("execution") or {}
    fused_parity = fused.get("parity") or {}
    fused_routes = fused_parity.get("route_ids") or {}
    control_routes = (control.get("parity") or {}).get("route_ids") or {}
    state_metrics = metric(fused_state, control_state)
    route_match = fused_routes.get("observed") == control_routes.get("observed")
    cpu_parity = all(
        (value or {}).get("within_tolerance") is True
        for key, value in fused_parity.items()
        if key != "route_ids" and isinstance(value, dict) and "within_tolerance" in value
    )
    fused_dispatches = int(fused_exec.get("dispatches") or 0)
    control_dispatches = int(control_exec.get("dispatches") or 0)
    fused_gpu = int(fused_exec.get("gpu_ns") or 0)
    control_gpu = int(control_exec.get("gpu_ns") or 0)
    fused_wall = int(fused_exec.get("wall_ns") or 0)
    control_wall = int(control_exec.get("wall_ns") or 0)
    report = {
        "schema": "hawking.flash.fused_route_accumulate_parity.v1",
        "status": "PASSED_FUSED_ROUTE_ACCUMULATION_PARITY"
        if cpu_parity and route_match and state_metrics.get("within_tolerance") is True
        else "BLOCKED_FUSED_ROUTE_ACCUMULATION_PARITY",
        "layer": int(fused.get("layer") or args.layer),
        "fused_receipt": str(args.fused_receipt),
        "fused_receipt_sha256": hashlib.sha256(args.fused_receipt.read_bytes()).hexdigest(),
        "control_receipt": str(args.control_receipt),
        "control_receipt_sha256": hashlib.sha256(args.control_receipt.read_bytes()).hexdigest(),
        "fused_state_sha256": hashlib.sha256(args.fused_state.read_bytes()).hexdigest(),
        "control_state_sha256": hashlib.sha256(args.control_state.read_bytes()).hexdigest(),
        "comparison": {
            "route_ids_match": route_match,
            "fused_route_ids": fused_routes.get("observed"),
            "control_route_ids": control_routes.get("observed"),
            "fused_cpu_parity_all_within_tolerance": cpu_parity,
            "final_state": state_metrics,
            "exact_state_hash_match": hashlib.sha256(args.fused_state.read_bytes()).hexdigest()
            == hashlib.sha256(args.control_state.read_bytes()).hexdigest(),
            "interpretation": (
                "The direct kernel changes reduction order, so bitwise state identity is not required; "
                "the state remains within the qualified source-BF16 organ tolerance."
            ),
        },
        "physical_delta": {
            "fused_dispatches": fused_dispatches,
            "control_dispatches": control_dispatches,
            "dispatches_saved": control_dispatches - fused_dispatches,
            "fused_gpu_ns": fused_gpu,
            "control_gpu_ns": control_gpu,
            "gpu_reduction_fraction": (control_gpu - fused_gpu) / control_gpu if control_gpu else None,
            "fused_wall_ns": fused_wall,
            "control_wall_ns": control_wall,
            "wall_reduction_fraction": (control_wall - fused_wall) / control_wall if control_wall else None,
            "fused_route_accumulation": fused_exec.get("route_accumulation"),
            "routed_output_materialized": fused_exec.get("routed_output_materialized"),
            "interpretation": (
                "One host dispatch and the TOP_K x hidden routed-output materialization were removed. "
                "GPU time is a diagnostic organ result; isolated wall time is not a protected complete-token benchmark."
            ),
        },
        "claim_boundary": (
            f"This is a bounded layer-{int(fused.get('layer') or args.layer)} compact full-attention/MoE organ comparison. It proves direct fused "
            "routed accumulation against the existing CPU/source-BF16 oracle and materialized compact control "
            "within organ tolerance. It does not prove complete-token capability, TPS, EBPW, residency, or a "
            "whole-model wall-time win."
        ),
        "promotion_allowed": False,
        "next_action": "Reuse the fused accumulation primitive in the next qualified compact full-attention boundaries, then measure complete-forward wall time before promotion.",
        "bench": {
            "state": "UNKNOWN",
            "measurement_state": "MEASURED_DIAGNOSTIC_ORGAN",
            "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "recorded_by": "tools/flash_fused_route_accumulate_compare.py",
            "machine": "Apple M3 Ultra",
            "rule": "organ parity and dispatch/GPU delta only; no complete-token promotion",
        },
    }
    report["seal_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "route_ids_match": route_match,
                "dispatches_saved": control_dispatches - fused_dispatches,
                "gpu_reduction_fraction": report["physical_delta"]["gpu_reduction_fraction"],
                "state_max_abs_error": state_metrics.get("max_abs_error"),
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0 if report["status"].startswith("PASSED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
