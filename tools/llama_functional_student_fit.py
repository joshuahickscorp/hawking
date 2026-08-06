#!/usr/bin/env python3
"""Fit and capability-gate a compact Llama hidden-state student.

This is the smallest learned block worth measuring after source-derived
formats failed: a low-rank map from an actual teacher hidden surface to its
actual output surface.  It is intentionally offline only.  Even a passing
held-out surface score is *not* a model-quality or TPS result; runtime
integration must separately pass generated-token capability and a matched
same-model decode benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "hawking.tg.llama_functional_student_fit.v1"
CAPTURE_SCHEMA = "hawking.tg.llama_functional_student_capture.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_rmse(prediction: np.ndarray, target: np.ndarray, baseline: np.ndarray) -> float:
    error = float(np.square(prediction - target, dtype=np.float64).sum())
    reference = float(np.square(baseline - target, dtype=np.float64).sum())
    return float(np.sqrt(error / max(reference, np.finfo(np.float64).tiny)))


def load_dataset(path: Path, receipt_path: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    if not path.is_file():
        raise ValueError(f"dataset not found: {path}")
    data = np.load(path)
    try:
        x = np.asarray(data["inputs"], dtype=np.float32)
        y = np.asarray(data["targets"], dtype=np.float32)
        heldout = np.asarray(data["heldout"], dtype=np.bool_)
    finally:
        data.close()
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or heldout.shape != (x.shape[0],):
        raise ValueError("dataset arrays must be paired rank-2 inputs/targets plus one heldout flag per row")
    receipt = None
    if receipt_path is not None:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("schema") != CAPTURE_SCHEMA:
            raise ValueError("unexpected capture receipt schema")
        if receipt.get("dataset", {}).get("sha256") != sha256(path):
            raise ValueError("dataset hash does not match capture receipt")
    return x, y, heldout, receipt


def ridge_inverse_cross(x: np.ndarray, cross: np.ndarray, ridge: float = 1e-4, iterations: int = 96) -> np.ndarray:
    """Solve (XᵀX + λI) B = cross without forming a 4096² Gram matrix."""
    estimate = np.zeros_like(cross)
    residual = cross.copy()
    direction = residual.copy()
    residual_norm = (residual * residual).sum(axis=0)
    for _ in range(iterations):
        applied = x.T @ (x @ direction) + np.float32(ridge) * direction
        denominator = (direction * applied).sum(axis=0)
        alpha = residual_norm / np.maximum(denominator, np.finfo(np.float32).tiny)
        estimate += direction * alpha
        residual -= applied * alpha
        next_norm = (residual * residual).sum(axis=0)
        if float(next_norm.max(initial=0.0)) < 1e-12:
            break
        beta = next_norm / np.maximum(residual_norm, np.finfo(np.float32).tiny)
        direction = residual + direction * beta
        residual_norm = next_norm
    return estimate


def fit_low_rank(x: np.ndarray, y: np.ndarray, heldout: np.ndarray, rank: int, *, cg_iterations: int = 24) -> dict[str, Any]:
    fit = ~heldout
    if rank < 1 or rank > min(int(fit.sum()), x.shape[1]):
        raise ValueError("rank must be between 1 and min(fit rows, input width)")
    if cg_iterations < 1:
        raise ValueError("cg_iterations must be positive")
    x_fit, y_fit = x[fit], y[fit]
    x_mean = x_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_mean = y_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered_x = x_fit - x_mean
    # The shared state must be supervised: PCA retains input variance that can
    # be irrelevant to the teacher output.  We whiten XᵀY with a small-ridge
    # CG solve, then take its deterministic randomized range.  This is a
    # low-rank regression direction, rather than covariance's variance-biased
    # direction, and avoids materialising a 4096×4096 Gram matrix.  No heldout
    # row reaches this operation or the following least-squares fit.
    oversample = min(8, max(0, min(centered_x.shape[1], y.shape[1]) - rank))
    probe = np.random.default_rng(17).standard_normal((y.shape[1], rank + oversample), dtype=np.float32)
    cross_range = ridge_inverse_cross(centered_x, centered_x.T @ ((y_fit - y_mean) @ probe), iterations=cg_iterations)
    basis, _ = np.linalg.qr(cross_range, mode="reduced")
    basis = basis[:, :rank].astype(np.float32, copy=False)
    feature_fit = centered_x @ basis
    readout, _, _, _ = np.linalg.lstsq(feature_fit, y_fit - y_mean, rcond=None)
    readout = readout.astype(np.float32, copy=False)

    def predict(values: np.ndarray) -> np.ndarray:
        return (values - x_mean) @ basis @ readout + y_mean

    fit_prediction, heldout_prediction = predict(x_fit), predict(x[heldout])
    fit_score = normalized_rmse(fit_prediction, y_fit, np.broadcast_to(y_mean, y_fit.shape))
    heldout_score = normalized_rmse(heldout_prediction, y[heldout], np.broadcast_to(y_mean, y[heldout].shape))
    # fp16 is a prospective artifact storage bill; fitting and scoring stay
    # f32 so numerical error cannot masquerade as student capacity.
    stored_parameters = int(basis.size + readout.size + x_mean.size + y_mean.size)
    executed_macs = int(x.shape[1] * rank + rank * y.shape[1])
    return {
        "basis": basis, "readout": readout, "x_mean": x_mean, "y_mean": y_mean,
        "fit_normalized_rmse": fit_score, "heldout_normalized_rmse": heldout_score,
        "physical": {"stored_parameters": stored_parameters, "prospective_fp16_artifact_bytes": stored_parameters * 2, "executed_macs_per_token": executed_macs, "sequential_matvecs_per_token": 2},
    }


def fit_rff_silu(x: np.ndarray, y: np.ndarray, heldout: np.ndarray, width: int, *, seed: int = 17) -> dict[str, Any]:
    """Fit a predeclared nonlinear shared feature map and linear readout."""
    fit = ~heldout
    if width < 1:
        raise ValueError("width must be positive")
    x_fit, y_fit = x[fit], y[fit]
    x_mean = x_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_mean = y_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    basis = (np.random.default_rng(seed).standard_normal((x.shape[1], width), dtype=np.float32) / np.sqrt(np.float32(x.shape[1]))).astype(np.float32)

    def features(values: np.ndarray) -> np.ndarray:
        preactivation = (values - x_mean) @ basis
        return preactivation / (1.0 + np.exp(-preactivation, dtype=np.float32))

    feature_fit = features(x_fit)
    feature_mean = feature_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    readout, _, _, _ = np.linalg.lstsq(feature_fit - feature_mean, y_fit - y_mean, rcond=None)
    readout = readout.astype(np.float32, copy=False)
    def predict(values: np.ndarray) -> np.ndarray:
        return (features(values) - feature_mean) @ readout + y_mean
    fit_prediction, heldout_prediction = predict(x_fit), predict(x[heldout])
    fit_score = normalized_rmse(fit_prediction, y_fit, np.broadcast_to(y_mean, y_fit.shape))
    heldout_score = normalized_rmse(heldout_prediction, y[heldout], np.broadcast_to(y_mean, y[heldout].shape))
    stored_parameters = int(basis.size + readout.size + x_mean.size + feature_mean.size + y_mean.size)
    return {
        "basis": basis, "readout": readout, "x_mean": x_mean, "feature_mean": feature_mean, "y_mean": y_mean,
        "fit_normalized_rmse": fit_score, "heldout_normalized_rmse": heldout_score,
        "physical": {"stored_parameters": stored_parameters, "prospective_fp16_artifact_bytes": stored_parameters * 2, "executed_macs_per_token": int(x.shape[1] * width + width * y.shape[1]), "sequential_matvecs_per_token": 2, "activation": "silu"},
    }


def make_receipt(dataset: Path, receipt: dict[str, Any] | None, x: np.ndarray, y: np.ndarray, heldout: np.ndarray, *, rank: int, max_normalized_rmse: float, min_fit_rows: int, min_heldout_rows: int, unsafe_small_probe: bool, cg_iterations: int = 24, family: str = "low_rank") -> dict[str, Any]:
    counts = {"fit_rows": int((~heldout).sum()), "heldout_rows": int(heldout.sum())}
    base: dict[str, Any] = {
        "schema": SCHEMA, "dataset_path": str(dataset), "dataset_sha256": sha256(dataset),
        "capture_receipt": receipt, "geometry": {"input_width": int(x.shape[1]), "target_width": int(y.shape[1]), **counts},
        "rank": rank, "gate": {"max_heldout_normalized_rmse": max_normalized_rmse, "min_fit_rows": min_fit_rows, "min_heldout_rows": min_heldout_rows},
        "tps_claim": None, "runtime_eligibility": "NO: requires generated-token capability, artifact/runtime parity, and matched decode evidence",
    }
    if counts["fit_rows"] < min_fit_rows or counts["heldout_rows"] < min_heldout_rows:
        if not unsafe_small_probe:
            return {**base, "status": "REFUSED_INSUFFICIENT_TEACHER_EVIDENCE", "reason": "capture is below the declared fit/heldout minimum; pass --unsafe-small-probe only to measure a non-promotable diagnostic"}
    if family == "low_rank":
        fitted = fit_low_rank(x, y, heldout, rank, cg_iterations=cg_iterations)
    elif family == "rff_silu":
        fitted = fit_rff_silu(x, y, heldout, rank)
    else:
        raise ValueError(f"unsupported student family {family!r}")
    passed = fitted["heldout_normalized_rmse"] <= max_normalized_rmse
    status = "OFFLINE_SURFACE_GATE_PASS_RUNTIME_REQUIRED" if passed and not unsafe_small_probe else "OFFLINE_SURFACE_GATE_FAILED"
    if unsafe_small_probe:
        status = "UNSAFE_SMALL_PROBE_NOT_PROMOTABLE"
    solver = {"cg_iterations": cg_iterations, "basis": "ridge_whitened_cross_covariance"} if family == "low_rank" else {"seed": 17, "basis": "fixed_gaussian_rff_silu"}
    return {**base, "family": family, "solver": solver, "status": status, "score": {key: value for key, value in fitted.items() if key not in {"basis", "readout", "x_mean", "feature_mean", "y_mean"}}, "student_format": f"llama.{family}_hidden_student.v1"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--capture-receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="JSON receipt; no runtime artifact is emitted")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--family", choices=("low_rank", "rff_silu"), default="low_rank")
    parser.add_argument("--cg-iterations", type=int, default=24)
    parser.add_argument("--max-heldout-normalized-rmse", type=float, default=0.10)
    parser.add_argument("--min-fit-rows", type=int, default=8192)
    parser.add_argument("--min-heldout-rows", type=int, default=2048)
    parser.add_argument("--unsafe-small-probe", action="store_true")
    args = parser.parse_args()
    x, y, heldout, capture_receipt = load_dataset(args.dataset, args.capture_receipt)
    receipt = make_receipt(args.dataset, capture_receipt, x, y, heldout, rank=args.rank, max_normalized_rmse=args.max_heldout_normalized_rmse, min_fit_rows=args.min_fit_rows, min_heldout_rows=args.min_heldout_rows, unsafe_small_probe=args.unsafe_small_probe, cg_iterations=args.cg_iterations, family=args.family)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "out": str(args.out)}, indent=2))
    return 0 if receipt["status"] != "REFUSED_INSUFFICIENT_TEACHER_EVIDENCE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
