#!/usr/bin/env python3.12
"""Auction Kimi F1 Doctor bits between weight repair and hidden-state recovery.

Consumes the immutable F1 teacher capture and serialized base representations.
It performs no parent forward, routing, calibration, or teacher recapture.
Every evaluated recovery Doctor is serialized at its quantized physical dtype and
charged against the original candidate's complete BPW ceiling.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import ml_dtypes
import numpy as np


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_f1_bracket as f1  # noqa: E402


ARCHITECTURES = {
    "BASE_OUTPUT_RECOVERY_R31": (("base_output", 31),),
    "DUAL_PATH_RECOVERY_R16X2": (("expert_input", 16), ("base_output", 16)),
}


def quantized_array(value: np.ndarray, dtype: str) -> tuple[np.ndarray, bytes]:
    if dtype == "bfloat16":
        stored = np.asarray(value, dtype=ml_dtypes.bfloat16)
    elif dtype == "float16":
        stored = np.asarray(value, dtype=np.float16)
    else:
        raise f1.F1Error(f"unsupported Doctor dtype: {dtype}")
    return np.asarray(stored, dtype=np.float32), stored.tobytes(order="C")


def component(name: str, value: np.ndarray, dtype: str) -> tuple[np.ndarray, dict[str, Any]]:
    quantized, data = quantized_array(value, dtype)
    return quantized, {"name": name, "role": "doctor", "encoding": dtype,
                       "shape": list(value.shape), "data": data}


def decode_component(component_value: dict[str, Any]) -> np.ndarray:
    dtype = (ml_dtypes.bfloat16 if component_value["encoding"] == "bfloat16"
             else np.float16)
    return np.frombuffer(component_value["data"], dtype=dtype).astype(np.float32).reshape(
        tuple(int(value) for value in component_value["shape"])
    )


def apply_hidden_doctor(
    payload_path: Path, expert_input: np.ndarray, base_output: np.ndarray,
) -> np.ndarray:
    """Execute a serialized recovery Doctor using only its installed physical components."""
    header, components = f1.read_payload(payload_path)
    architecture = str(header["architecture"])
    by_name = {component_value["name"]: component_value for component_value in components}
    features = []
    for path_name, _ in ARCHITECTURES[architecture]:
        source = expert_input if path_name == "expert_input" else base_output
        mean = decode_component(by_name[f"doctor.{path_name}.mean"])
        basis = decode_component(by_name[f"doctor.{path_name}.basis"])
        features.append((source - mean[None, :]) @ basis.T)
    design = np.concatenate(features, axis=1)
    residual_mean = decode_component(by_name["doctor.output.residual_mean"])
    output_map = decode_component(by_name["doctor.output.low_rank_map"])
    diagonal = decode_component(by_name["doctor.output.diagonal_gain"])
    base_mean = decode_component(by_name["doctor.base_output.mean"])
    correction = residual_mean[None, :] + design @ output_map
    correction += (base_output - base_mean[None, :]) * diagonal[None, :]
    return np.asarray(base_output + correction, dtype=np.float32)


def fit_hidden_doctor(
    architecture: str,
    expert_input_fit: np.ndarray,
    expert_input_score: np.ndarray,
    base_fit: np.ndarray,
    base_score: np.ndarray,
    teacher_fit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Fit a quantized low-rank residual map using only the disjoint fit split."""
    paths = ARCHITECTURES[architecture]
    source = {"expert_input": (expert_input_fit, expert_input_score),
              "base_output": (base_fit, base_score)}
    fit_features = []
    score_features = []
    components: list[dict[str, Any]] = []
    path_records = []
    for path_name, requested_rank in paths:
        fit_value, score_value = source[path_name]
        mean, mean_component = component(
            f"doctor.{path_name}.mean", np.mean(fit_value, axis=0), "bfloat16",
        )
        components.append(mean_component)
        fit_centered = np.asarray(fit_value - mean[None, :], dtype=np.float32)
        score_centered = np.asarray(score_value - mean[None, :], dtype=np.float32)
        _, singular, right = np.linalg.svd(fit_centered, full_matrices=False)
        rank = min(requested_rank, fit_centered.shape[0] - 1, right.shape[0])
        basis, basis_component = component(
            f"doctor.{path_name}.basis", right[:rank], "float16",
        )
        components.append(basis_component)
        fit_features.append(fit_centered @ basis.T)
        score_features.append(score_centered @ basis.T)
        path_records.append({
            "path": path_name, "rank": rank,
            "captured_feature_energy": float(
                np.sum(singular[:rank] ** 2) / (np.sum(singular ** 2) + 1e-30)),
        })

    fit_design = np.concatenate(fit_features, axis=1)
    score_design = np.concatenate(score_features, axis=1)
    residual = np.asarray(teacher_fit - base_fit, dtype=np.float32)
    residual_mean, residual_mean_component = component(
        "doctor.output.residual_mean", np.mean(residual, axis=0), "bfloat16",
    )
    components.append(residual_mean_component)
    target = residual - residual_mean[None, :]
    gram = fit_design.T @ fit_design
    ridge = max(float(np.trace(gram) / max(1, gram.shape[0])) * 0.05, 1e-6)
    output_map = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0], dtype=np.float32),
        fit_design.T @ target,
    )
    output_map, output_map_component = component(
        "doctor.output.low_rank_map", output_map, "float16",
    )
    components.append(output_map_component)
    fit_correction = residual_mean[None, :] + fit_design @ output_map
    score_correction = residual_mean[None, :] + score_design @ output_map

    # A cheap coordinate-local path can correct systematic attenuation that a
    # sample-limited low-rank map cannot span. It is fitted after the low-rank path.
    base_mean = np.mean(base_fit, axis=0, keepdims=True)
    centered_base_fit = base_fit - base_mean
    diagonal_target = residual - fit_correction
    denominator = np.sum(centered_base_fit ** 2, axis=0)
    diagonal_ridge = max(float(np.mean(denominator)) * 0.05, 1e-6)
    diagonal = np.sum(centered_base_fit * diagonal_target, axis=0) / (
        denominator + diagonal_ridge
    )
    diagonal, diagonal_component = component(
        "doctor.output.diagonal_gain", diagonal, "float16",
    )
    components.append(diagonal_component)
    fit_correction += centered_base_fit * diagonal[None, :]
    score_correction += (base_score - base_mean) * diagonal[None, :]
    record = {
        "architecture": architecture,
        "paths": path_records,
        "ridge": ridge,
        "diagonal_ridge": diagonal_ridge,
        "doctor_component_bytes": f1.role_bytes(components, "doctor"),
        "fit_uses_score_data": False,
    }
    return base_fit + fit_correction, base_score + score_correction, components, record


def routed_metrics(
    capture: dict[str, np.ndarray], sentinel: int,
    teacher: np.ndarray, candidate: np.ndarray,
) -> dict[str, Any]:
    routes = capture["score_routes"].astype(np.int32)
    route_weights = capture["score_route_weights"].astype(np.float32)
    mask = routes == sentinel
    tokens = np.any(mask, axis=1)
    if not np.any(tokens):
        return {"tokens": 0, "slots": 0, "status": "NO_SCORE_ROUTE_SLOTS"}
    weight = np.sum(np.where(mask, route_weights, 0.0), axis=1)[tokens, None]
    teacher_weighted = teacher[tokens] * weight
    candidate_weighted = candidate[tokens] * weight
    residual = capture["score_post_attention"].astype(np.float32)[tokens]
    return {
        "tokens": int(np.sum(tokens)), "slots": int(np.sum(mask)),
        "weighted_expert_output": f1.quality(teacher_weighted, candidate_weighted),
        "modeled_first_residual_add": f1.quality(
            residual + teacher_weighted, residual + candidate_weighted,
        ),
    }


def run_variant(
    candidate: str,
    architecture: str,
    capture: dict[str, np.ndarray],
    source_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    original_payload = source_dir / f"{candidate}_sentinel_expert.k26f1"
    original_result = f1.read_json(source_dir / f"{candidate}_F1_RESULT.json")
    base_weights, base_components = f1.decode_base_weights(original_payload)
    fit_x = capture["fit_x"].astype(np.float32)
    score_x = capture["score_x"].astype(np.float32)
    teacher_fit = capture["fit_teacher_output"].astype(np.float32)
    teacher_score = capture["score_teacher_output"].astype(np.float32)
    base_fit, _ = f1.expert_forward(fit_x, base_weights)
    base_score, _ = f1.expert_forward(score_x, base_weights)
    doctored_fit, doctored_score, doctor_components, doctor_record = fit_hidden_doctor(
        architecture, fit_x, score_x, base_fit, base_score, teacher_fit,
    )

    budget = original_result["physical_budget"]
    if f1.role_bytes(base_components, "base") > int(budget["base_ceiling_bytes"]):
        raise f1.F1Error("reused base components exceed sealed base ceiling")
    if f1.role_bytes(doctor_components, "doctor") > int(budget["doctor_ceiling_bytes"]):
        raise f1.F1Error("hidden-state Doctor exceeds sealed Doctor ceiling")
    payload_path = output_dir / f"{candidate}_{architecture}.k26f1"
    payload = f1.write_payload(payload_path, {
        "schema": "hawking.kimi_k26.f1_hidden_doctor_payload.v1",
        "candidate": candidate, "architecture": architecture,
        "revision": f1.REVISION, "layer": f1.LAYER,
        "sentinel_expert": int(capture["sentinel_expert"][0]),
        "base_payload_sha256": f1.sha256_file(original_payload),
        "base_representation_reused": True,
    }, base_components + doctor_components)
    if payload["header_overhead_bytes"] > int(budget["overhead_ceiling_bytes"]):
        raise f1.F1Error("hidden-state Doctor header exceeds overhead ceiling")
    if payload["bytes"] > int(budget["complete_ceiling_bytes"]):
        raise f1.F1Error("hidden-state Doctor payload exceeds complete ceiling")

    fit_metric = f1.quality(teacher_fit, doctored_fit)
    score_metric = f1.quality(teacher_score, doctored_score)
    base_metric = f1.quality(teacher_score, base_score)
    verdict = f1.fidelity_verdict(score_metric)
    sentinel = int(capture["sentinel_expert"][0])
    result = f1.seal({
        "schema": "hawking.kimi_k26.f1_hidden_doctor_result.v1", "status": "PASS",
        "sealed_at": f1.now(), "candidate": candidate, "architecture": architecture,
        "source": {"repo": f1.REPO, "revision": f1.REVISION},
        "layer": f1.LAYER, "sentinel_expert": sentinel,
        "claim_boundary": "F1 ONE_REAL_LAYER ONE_SENTINEL EXPERT; not end-to-end capability",
        "reuse": {"parent_forwards": 0, "teacher_captures": 0, "routing_runs": 0,
                  "base_refits": 0, "serialized_base_payload_reused": True},
        "physical_budget": {
            **budget,
            "actual_complete_bpw": payload["bytes"] * 8 /
                                   int(budget["logical_weights_represented"]),
            "unused_ceiling_bytes": int(budget["complete_ceiling_bytes"]) - payload["bytes"],
        },
        "payload": payload, "doctor": doctor_record,
        "metrics": {
            "fit": fit_metric, "score_base": base_metric, "score_doctored": score_metric,
            "routed_score_subset": routed_metrics(
                capture, sentinel, teacher_score, doctored_score,
            ),
            "doctor_cosine_gain": score_metric["cosine_mean"] - base_metric["cosine_mean"],
            "doctor_recovery_fraction_of_output_relative_l2": (
                1 - score_metric["relative_l2"] / (base_metric["relative_l2"] + 1e-30)
            ),
        },
        "base_verdict": f1.fidelity_verdict(base_metric),
        "candidate_verdict": verdict,
        "doctor_prevented_collapse": (
            f1.fidelity_verdict(base_metric) == "COLLAPSE_F1" and verdict != "COLLAPSE_F1"
        ),
    })
    f1.atomic_json(output_dir / f"{candidate}_{architecture}_RESULT.json", result)
    del base_weights, base_components, doctor_components
    gc.collect()
    return result


def decide(results: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    survivors = [result for result in results if result["candidate_verdict"] == "SURVIVES_F1"]
    all_options = results + [
        {"candidate": candidate, "architecture": "WEIGHT_RESIDUAL_PQ",
         "candidate_verdict": value["candidate_verdict"],
         "physical_budget": value["physical_budget"],
         "metrics": {"score_doctored": value["metrics"]["score"]["doctored"]}}
        for candidate, value in baseline.items()
    ]
    best = max(all_options, key=lambda result:
               result["metrics"]["score_doctored"]["cosine_mean"])
    if survivors:
        best_survivor = max(survivors, key=lambda result:
                            result["metrics"]["score_doctored"]["cosine_mean"])
        p5_survivors = [result for result in survivors if result["candidate"] == "P5"]
        if p5_survivors:
            best_p5 = max(p5_survivors, key=lambda result:
                          result["metrics"]["score_doctored"]["cosine_mean"])
            if (best_p5["metrics"]["score_doctored"]["cosine_mean"] >=
                    best_survivor["metrics"]["score_doctored"]["cosine_mean"] - 0.02):
                best_survivor = best_p5
        return {
            "decision": f"PROMOTE_{best_survivor['candidate']}_HIDDEN_DOCTOR_TO_F2",
            "current_best_candidate": best_survivor["candidate"],
            "current_best_architecture": best_survivor["architecture"],
            "current_best_bpw": best_survivor["physical_budget"]["actual_complete_bpw"],
            "current_best_capability": best_survivor["metrics"]["score_doctored"],
            "current_doctor_allocation": best_survivor["doctor"],
            "current_failure_mode": "NONE_AT_F1_SENTINEL",
            "current_dominant_bottleneck": "CROSS_LAYER_GENERALIZATION_UNTESTED",
            "current_scientific_hypothesis": (
                "Kimi compact experts prefer functional hidden-state recovery over independent "
                "weight-residual reconstruction at the tested envelope."
            ),
            "current_next_experiment": (
                f"{best_survivor['candidate']}_F2_HIDDEN_DOCTOR_EARLY_MIDDLE_LATE_REPLICATION"
            ),
        }
    return {
        "decision": "NO_PROMOTION",
        "current_best_candidate": best["candidate"],
        "current_best_architecture": best["architecture"],
        "current_best_bpw": best["physical_budget"]["actual_complete_bpw"],
        "current_best_capability": best["metrics"]["score_doctored"],
        "current_doctor_allocation": (
            best.get("doctor") or {"architecture": "WEIGHT_RESIDUAL_PQ"}
        ),
        "current_failure_mode": "EXPERT_OUTPUT_DEGRADATION_AFTER_NATIVE_ROUTING",
        "current_dominant_bottleneck": "MULTIPLICATIVE_EXPERT_REPRESENTATION",
        "current_scientific_hypothesis": (
            "Neither independent residual-weight PQ nor sample-limited hidden-state recovery can "
            "restore the missing expert-output direction; test shared grammar against a larger "
            "protected-island allocation using the same teacher seam."
        ),
        "current_next_experiment": "F1_SHARED_GRAMMAR_VS_PROTECTED_ISLANDS_CACHED_SEAM",
    }


def run(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_file = source_dir / "teacher_capture.npz"
    capture_receipt = f1.read_json(source_dir / "teacher_capture.json")
    if f1.sha256_file(capture_file) != capture_receipt["capture_sha256"]:
        raise f1.F1Error("cached teacher capture hash mismatch")
    with np.load(capture_file, allow_pickle=False) as loaded:
        capture = {key: loaded[key] for key in loaded.files}
    baseline = {candidate: f1.read_json(source_dir / f"{candidate}_F1_RESULT.json")
                for candidate in ("P1", "P5")}
    results = []
    for candidate in ("P1", "P5"):
        for architecture in ARCHITECTURES:
            results.append(run_variant(
                candidate, architecture, capture, source_dir, output_dir,
            ))
    decision = decide(results, baseline)
    artifact = f1.seal({
        "schema": "hawking.kimi_k26.f1_doctor_auction.v1", "status": "PASS",
        "sealed_at": f1.now(), "runtime_seconds": time.time() - started,
        "source": {"repo": f1.REPO, "revision": f1.REVISION},
        "experiment": "F1_DOCTOR_WEIGHT_REPAIR_VS_HIDDEN_STATE_RECOVERY",
        "teacher_capture_seal_sha256": capture_receipt["seal_sha256"],
        "reuse": {"parent_forwards": 0, "teacher_captures": 0, "routing_runs": 0,
                  "base_refits": 0},
        "variant_results": [{
            "candidate": result["candidate"], "architecture": result["architecture"],
            "actual_bpw": result["physical_budget"]["actual_complete_bpw"],
            "verdict": result["candidate_verdict"],
            "score_cosine": result["metrics"]["score_doctored"]["cosine_mean"],
            "seal_sha256": result["seal_sha256"],
        } for result in results],
        "diagnosis_metrics_reason_next_decision": {
            "diagnosis": decision["current_failure_mode"],
            "metrics": decision["current_best_capability"],
            "reason": decision["current_scientific_hypothesis"],
            "next_decision": decision["decision"],
        },
        **decision,
    })
    f1.atomic_json(output_dir / "KIMI_K26_F1_DOCTOR_AUCTION.json", artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.source_dir.resolve(strict=True), args.output_dir.resolve())
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
