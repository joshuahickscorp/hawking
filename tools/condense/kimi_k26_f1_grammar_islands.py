#!/usr/bin/env python3.12
"""Kimi F1 shared-grammar versus protected-island representation bracket.

Uses eight experts selected only by fit-route frequency and the immutable F1
activation seam. It compares two complete physical P1-envelope payloads:
independent activation-aware PQ with Doctor bytes auctioned entirely to salient
BF16 rows, and an amortized two-stage additive grammar shared across experts.
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

import gravity_forge as forge  # noqa: E402
import kimi_k26_f1_bracket as f1  # noqa: E402
import kimi_k26_reference as reference  # noqa: E402


TARGET = f1.CANDIDATES["P1"]["target"]
BASE_SHARE = f1.CANDIDATES["P1"]["base_share"]
DOCTOR_SHARE = f1.CANDIDATES["P1"]["doctor_share"]
PROTECTED_GEOMETRY = {"dim": 256, "subspaces": 16, "k": 256, "iters": 3}
SHARED_DIM = 32
SHARED_K = 256
SHARED_STAGES = 2
SHARED_ITERS = 3
SHARED_SAMPLES_PER_EXPERT = 32768


def select_experts(capture: dict[str, np.ndarray], count: int = 8) -> list[int]:
    routes = capture["fit_routes"].astype(np.int32)
    counts = np.bincount(routes.reshape(-1), minlength=384)
    order = np.argsort(-counts, kind="stable")
    selected = [int(expert) for expert in order if counts[expert] > 0][:count]
    if len(selected) != count:
        raise f1.F1Error("insufficient fit-routed experts for grammar bracket")
    return selected


def candidate_caps(total_weights: int) -> dict[str, int]:
    total = total_weights * TARGET.numerator // TARGET.denominator // 8
    base = total * BASE_SHARE.numerator // BASE_SHARE.denominator
    doctor = total * DOCTOR_SHARE.numerator // DOCTOR_SHARE.denominator
    return {"total": total, "base": base, "doctor": doctor,
            "overhead": total - base - doctor}


def base_only_protected_matrix(
    weight: np.ndarray,
    activations: np.ndarray,
    base_cap: int,
    doctor_cap: int,
    name: str,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    scale = f1.salience_scale(activations)
    reconstruction, components = f1.pq_payload(
        weight, scale, PROTECTED_GEOMETRY, seed, name, "base",
    )
    components.append({
        "name": f"{name}.base.salience_scale", "role": "base",
        "data": np.asarray(scale, dtype=ml_dtypes.bfloat16).tobytes(),
        "encoding": "bfloat16", "shape": list(scale.shape),
    })
    used = f1.role_bytes(components, "base")
    if used > base_cap:
        raise f1.F1Error("protected-island base geometry exceeds allocation")
    reconstruction, protected, base_rows = f1.add_protected_rows(
        weight, reconstruction, activations, base_cap - used, name, "base", residual=False,
    )
    components.extend(protected)
    reconstruction, protected, doctor_rows = f1.add_protected_rows(
        weight, reconstruction, activations, doctor_cap,
        name, "doctor", residual=True,
    )
    components.extend(protected)
    return reconstruction, components, {
        "base_protected_rows": base_rows, "doctor_protected_rows": doctor_rows,
        "base_bytes": f1.role_bytes(components, "base"),
        "doctor_bytes": f1.role_bytes(components, "doctor"),
        "weight_relative_l2": float(
            np.linalg.norm(weight - reconstruction) / (np.linalg.norm(weight) + 1e-30)
        ),
    }


def run_protected_islands(
    experts: list[int],
    weights: dict[int, dict[str, np.ndarray]],
    internals: dict[int, dict[str, np.ndarray]],
    capture: dict[str, np.ndarray],
    caps: dict[str, int],
    output_dir: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any], list[dict[str, Any]]]:
    represented = [weights[expert][matrix].size for expert in experts for matrix in f1.MATRICES]
    base_caps = f1.distribute(caps["base"], represented)
    doctor_caps = f1.distribute(caps["doctor"], represented)
    fit_x = capture["fit_x"].astype(np.float32)
    reconstructed: dict[int, dict[str, np.ndarray]] = {expert: {} for expert in experts}
    components: list[dict[str, Any]] = []
    records = {}
    cursor = 0
    for expert in experts:
        records[str(expert)] = {}
        for matrix_index, matrix in enumerate(f1.MATRICES):
            activations = (fit_x if matrix != "down_proj" else
                           internals[expert]["hidden"][:fit_x.shape[0]])
            value, packed, record = base_only_protected_matrix(
                weights[expert][matrix], activations,
                base_caps[cursor], doctor_caps[cursor],
                f"expert.{expert}.{matrix}",
                261100 + expert * 101 + matrix_index * 17,
            )
            cursor += 1
            reconstructed[expert][matrix] = value
            components.extend(packed)
            records[str(expert)][matrix] = record
    payload = f1.write_payload(output_dir / "P1_PROTECTED_ISLANDS_8E.k26f1", {
        "schema": "hawking.kimi_k26.f1_protected_islands_cluster.v1",
        "candidate": "P1", "family": "PROTECTED_ISLANDS",
        "revision": f1.REVISION, "layer": f1.LAYER, "experts": experts,
    }, components)
    return reconstructed, {"matrix_records": records, "payload": payload}, components


def shared_codebooks(
    scaled_weights: list[np.ndarray], seed: int,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], list[np.ndarray]]:
    """Fit sampled shared codebooks, then assign every expert separately."""
    torch = forge._torch()  # noqa: SLF001
    device = forge._device()  # noqa: SLF001
    rng = np.random.default_rng(seed)
    vectors = [np.ascontiguousarray(weight.reshape(-1, SHARED_DIM), dtype=np.float32)
               for weight in scaled_weights]
    sample_indices = [
        rng.choice(value.shape[0], size=min(SHARED_SAMPLES_PER_EXPERT, value.shape[0]),
                   replace=False)
        for value in vectors
    ]
    samples = np.concatenate([value[index] for value, index in zip(
        vectors, sample_indices, strict=True
    )])
    sample_residual = torch.from_numpy(samples).to(device)
    codebooks: list[np.ndarray] = []
    for stage in range(SHARED_STAGES):
        codebook = forge._kmeans(  # noqa: SLF001
            sample_residual, SHARED_K, iters=SHARED_ITERS, seed=seed + stage,
        )
        quantized = np.asarray(codebook.detach().cpu().numpy(), dtype=np.float16)
        codebook = torch.from_numpy(quantized.astype(np.float32)).to(device)
        sample_residual = sample_residual - codebook[forge._assign(  # noqa: SLF001
            sample_residual, codebook,
        )]
        codebooks.append(quantized)

    all_indices: list[list[np.ndarray]] = [[] for _ in vectors]
    reconstructions = []
    for expert_index, value in enumerate(vectors):
        source = torch.from_numpy(value).to(device)
        residual = source.clone()
        reconstruction = torch.zeros_like(source)
        for codebook_np in codebooks:
            codebook = torch.from_numpy(codebook_np.astype(np.float32)).to(device)
            indices = forge._assign(residual, codebook)  # noqa: SLF001
            residual = residual - codebook[indices]
            reconstruction = reconstruction + codebook[indices]
            all_indices[expert_index].append(
                indices.detach().cpu().numpy().astype(np.uint32)
            )
        reconstructions.append(reconstruction.detach().cpu().numpy())
    del samples, sample_residual
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return codebooks, all_indices, reconstructions


def run_shared_grammar(
    experts: list[int],
    weights: dict[int, dict[str, np.ndarray]],
    internals: dict[int, dict[str, np.ndarray]],
    capture: dict[str, np.ndarray],
    caps: dict[str, int],
    output_dir: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any], list[dict[str, Any]]]:
    fit_x = capture["fit_x"].astype(np.float32)
    reconstructed: dict[int, dict[str, np.ndarray]] = {expert: {} for expert in experts}
    components: list[dict[str, Any]] = []
    records: dict[str, Any] = {}
    matrix_base_caps = f1.distribute(caps["base"], [1] * len(f1.MATRICES))
    matrix_doctor_caps = f1.distribute(caps["doctor"], [1] * len(f1.MATRICES))
    for matrix_index, matrix in enumerate(f1.MATRICES):
        scales = []
        scaled = []
        for expert in experts:
            activations = (fit_x if matrix != "down_proj" else
                           internals[expert]["hidden"][:fit_x.shape[0]])
            scale = f1.salience_scale(activations)
            scales.append(scale)
            scaled.append(np.ascontiguousarray(
                weights[expert][matrix] * scale[None, :], dtype=np.float32,
            ))
        codebooks, indices, decoded = shared_codebooks(
            scaled, seed=262600 + matrix_index * 1000,
        )
        for stage, codebook in enumerate(codebooks):
            components.append({
                "name": f"shared.{matrix}.base.codebook.{stage}", "role": "base",
                "data": codebook.tobytes(order="C"), "encoding": "float16",
                "shape": list(codebook.shape),
            })
        matrix_reconstructions = []
        for expert_index, expert in enumerate(experts):
            scale = scales[expert_index]
            components.append({
                "name": f"expert.{expert}.{matrix}.base.salience_scale", "role": "base",
                "data": np.asarray(scale, dtype=ml_dtypes.bfloat16).tobytes(),
                "encoding": "bfloat16", "shape": list(scale.shape),
            })
            for stage, index_values in enumerate(indices[expert_index]):
                packed = f1.pack_unsigned(index_values, 8)
                components.append({
                    "name": f"expert.{expert}.{matrix}.base.indices.{stage}", "role": "base",
                    "data": packed, "encoding": "dense_unsigned_lsb",
                    "shape": list(index_values.shape), "bits": 8,
                })
            value = decoded[expert_index].reshape(weights[expert][matrix].shape) / scale[None, :]
            matrix_reconstructions.append(value.astype(np.float32))

        base_remaining = matrix_base_caps[matrix_index] - f1.role_bytes(components, "base")
        # Prior matrices already consumed bytes, so restore their cap when calculating this matrix.
        base_remaining += sum(matrix_base_caps[:matrix_index])
        doctor_remaining = matrix_doctor_caps[matrix_index]
        base_per_expert = f1.distribute(max(0, base_remaining), [1] * len(experts))
        doctor_per_expert = f1.distribute(doctor_remaining, [1] * len(experts))
        records[matrix] = {}
        for expert_index, expert in enumerate(experts):
            activations = (fit_x if matrix != "down_proj" else
                           internals[expert]["hidden"][:fit_x.shape[0]])
            value, protected, base_rows = f1.add_protected_rows(
                weights[expert][matrix], matrix_reconstructions[expert_index], activations,
                base_per_expert[expert_index], f"expert.{expert}.{matrix}", "base",
                residual=False,
            )
            components.extend(protected)
            value, protected, doctor_rows = f1.add_protected_rows(
                weights[expert][matrix], value, activations,
                doctor_per_expert[expert_index], f"expert.{expert}.{matrix}", "doctor",
                residual=True,
            )
            components.extend(protected)
            reconstructed[expert][matrix] = value
            records[matrix][str(expert)] = {
                "base_protected_rows": base_rows, "doctor_protected_rows": doctor_rows,
                "weight_relative_l2": float(
                    np.linalg.norm(weights[expert][matrix] - value) /
                    (np.linalg.norm(weights[expert][matrix]) + 1e-30)
                ),
            }
        del scaled, decoded, matrix_reconstructions
        gc.collect()
    payload = f1.write_payload(output_dir / "P1_SHARED_GRAMMAR_8E.k26f1", {
        "schema": "hawking.kimi_k26.f1_shared_grammar_cluster.v1",
        "candidate": "P1", "family": "SHARED_ADDITIVE_GRAMMAR",
        "revision": f1.REVISION, "layer": f1.LAYER, "experts": experts,
        "dim": SHARED_DIM, "k": SHARED_K, "stages": SHARED_STAGES,
    }, components)
    return reconstructed, {"matrix_records": records, "payload": payload}, components


def evaluate(
    family: str,
    experts: list[int],
    teacher: dict[int, np.ndarray],
    reconstructed: dict[int, dict[str, np.ndarray]],
    capture: dict[str, np.ndarray],
    details: dict[str, Any],
    components: list[dict[str, Any]],
    caps: dict[str, int],
) -> dict[str, Any]:
    fit_x = capture["fit_x"].astype(np.float32)
    score_x = capture["score_x"].astype(np.float32)
    score_routes = capture["score_routes"].astype(np.int32)
    score_weights = capture["score_route_weights"].astype(np.float32)
    all_teacher = []
    all_candidate = []
    routed_teacher = []
    routed_candidate = []
    expert_metrics = {}
    for expert in experts:
        candidate_fit, _ = f1.expert_forward(fit_x, reconstructed[expert])
        candidate_score, _ = f1.expert_forward(score_x, reconstructed[expert])
        teacher_fit = teacher[expert][:fit_x.shape[0]]
        teacher_score = teacher[expert][fit_x.shape[0]:]
        all_teacher.append(teacher_score)
        all_candidate.append(candidate_score)
        mask = score_routes == expert
        tokens = np.any(mask, axis=1)
        route_weight = np.sum(np.where(mask, score_weights, 0.0), axis=1)
        if np.any(tokens):
            routed_teacher.append(teacher_score[tokens] * route_weight[tokens, None])
            routed_candidate.append(candidate_score[tokens] * route_weight[tokens, None])
        expert_metrics[str(expert)] = {
            "fit": f1.quality(teacher_fit, candidate_fit),
            "score": f1.quality(teacher_score, candidate_score),
            "score_route_slots": int(np.sum(mask)),
        }
    aggregate = f1.quality(np.concatenate(all_teacher), np.concatenate(all_candidate))
    routed = f1.quality(np.concatenate(routed_teacher), np.concatenate(routed_candidate))
    worst = min(value["score"]["cosine_mean"] for value in expert_metrics.values())
    verdict = ("SURVIVES_F1" if aggregate["cosine_mean"] >= 0.90 and
               aggregate["cosine_p10"] >= 0.80 and aggregate["relative_l2"] <= 0.50 and
               routed["cosine_mean"] >= 0.90 and worst >= 0.80 else
               "DEGRADED_F1" if aggregate["cosine_mean"] >= 0.75 and
               aggregate["relative_l2"] <= 0.85 else "COLLAPSE_F1")
    payload = details["payload"]
    if payload["base_component_bytes"] > caps["base"]:
        raise f1.F1Error(f"{family} base bytes exceed ceiling")
    if payload["doctor_component_bytes"] > caps["doctor"]:
        raise f1.F1Error(f"{family} Doctor bytes exceed ceiling")
    if payload["header_overhead_bytes"] > caps["overhead"] or payload["bytes"] > caps["total"]:
        raise f1.F1Error(f"{family} complete payload exceeds ceiling")
    return f1.seal({
        "schema": "hawking.kimi_k26.f1_family_cluster_result.v1", "status": "PASS",
        "sealed_at": f1.now(), "family": family, "candidate": "P1",
        "source": {"repo": f1.REPO, "revision": f1.REVISION},
        "layer": f1.LAYER, "experts": experts,
        "selection_uses_fit_routes_only": True,
        "claim_boundary": "F1 EIGHT-EXPERT OUTPUT RECONSTRUCTION; not full shard capability",
        "physical_budget": {
            "logical_weights_represented": sum(
                reconstructed[expert][matrix].size for expert in experts for matrix in f1.MATRICES
            ),
            "target_complete_bpw": str(TARGET),
            "actual_complete_bpw": payload["bytes"] * 8 / sum(
                reconstructed[expert][matrix].size for expert in experts for matrix in f1.MATRICES
            ),
            "complete_ceiling_bytes": caps["total"], "base_ceiling_bytes": caps["base"],
            "doctor_ceiling_bytes": caps["doctor"], "overhead_ceiling_bytes": caps["overhead"],
            "all_payload_bytes_counted": True,
        },
        "payload": payload, "matrix_records": details["matrix_records"],
        "metrics": {"aggregate_score": aggregate, "routed_score": routed,
                    "worst_expert_score_cosine": worst, "experts": expert_metrics},
        "candidate_verdict": verdict,
    })


def run(source: Path, f1_dir: Path, output_dir: Path) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = f1_dir / "teacher_capture.npz"
    receipt = f1.read_json(f1_dir / "teacher_capture.json")
    if f1.sha256_file(capture_path) != receipt["capture_sha256"]:
        raise f1.F1Error("cached teacher seam hash mismatch")
    with np.load(capture_path, allow_pickle=False) as loaded:
        capture = {key: loaded[key] for key in loaded.files}
    experts = select_experts(capture)
    x = np.concatenate((capture["fit_x"], capture["score_x"])).astype(np.float32)
    shard = reference.TensorShard(reference.shard_path(source, 2))
    weights = {}
    teacher = {}
    internals = {}
    for expert in experts:
        weights[expert] = f1.dequantized_expert(shard, expert)
        teacher[expert], internals[expert] = f1.expert_forward(x, weights[expert])
    del shard
    total_weights = sum(weights[expert][matrix].size
                        for expert in experts for matrix in f1.MATRICES)
    caps = candidate_caps(total_weights)

    protected, protected_details, protected_components = run_protected_islands(
        experts, weights, internals, capture, caps, output_dir,
    )
    protected_result = evaluate(
        "PROTECTED_ISLANDS", experts, teacher, protected, capture,
        protected_details, protected_components, caps,
    )
    f1.atomic_json(output_dir / "P1_PROTECTED_ISLANDS_8E_RESULT.json", protected_result)
    del protected, protected_components
    gc.collect()

    grammar, grammar_details, grammar_components = run_shared_grammar(
        experts, weights, internals, capture, caps, output_dir,
    )
    grammar_result = evaluate(
        "SHARED_ADDITIVE_GRAMMAR", experts, teacher, grammar, capture,
        grammar_details, grammar_components, caps,
    )
    f1.atomic_json(output_dir / "P1_SHARED_GRAMMAR_8E_RESULT.json", grammar_result)
    results = {"PROTECTED_ISLANDS": protected_result,
               "SHARED_ADDITIVE_GRAMMAR": grammar_result}
    survivors = [name for name, value in results.items()
                 if value["candidate_verdict"] == "SURVIVES_F1"]
    best = max(results, key=lambda name:
               results[name]["metrics"]["routed_score"]["cosine_mean"])
    if survivors:
        best = max(survivors, key=lambda name:
                   results[name]["metrics"]["routed_score"]["cosine_mean"])
        decision = f"PROMOTE_P1_{best}_TO_F2"
        next_experiment = f"P1_{best}_F2_SPARSE_PROPAGATION"
        hypothesis = f"{best} preserves the fit-selected Kimi expert cluster at F1."
    else:
        decision = "NO_PROMOTION_RETIRE_BOTH_FAMILIES_AT_P1_INSTANCE"
        next_experiment = "P1_FUNCTIONAL_CODEBOOK_ROUTE_STABILITY_OBJECTIVE_F1"
        hypothesis = (
            "At 0.98 BPW, neither shared additive weight grammar nor salience-protected independent "
            "PQ preserves routed expert output; the remaining reachable axis is functional "
            "codebook fitting against output and routing stability rather than weight residual."
        )
    artifact = f1.seal({
        "schema": "hawking.kimi_k26.f1_grammar_islands_bracket.v1", "status": "PASS",
        "sealed_at": f1.now(), "runtime_seconds": time.time() - started,
        "source": {"repo": f1.REPO, "revision": f1.REVISION},
        "experiment": "P1_F1_SHARED_GRAMMAR_VS_PROTECTED_ISLANDS",
        "teacher_capture_seal_sha256": receipt["seal_sha256"],
        "reuse": {"parent_forwards": 0, "teacher_captures": 0,
                  "attention_runs": 0, "routing_runs": 0},
        "experts": experts,
        "family_results": {name: {
            "verdict": value["candidate_verdict"],
            "actual_bpw": value["physical_budget"]["actual_complete_bpw"],
            "aggregate_score_cosine": value["metrics"]["aggregate_score"]["cosine_mean"],
            "routed_score_cosine": value["metrics"]["routed_score"]["cosine_mean"],
            "seal_sha256": value["seal_sha256"],
        } for name, value in results.items()},
        "current_best_candidate": f"P1_{best}",
        "current_best_bpw": results[best]["physical_budget"]["actual_complete_bpw"],
        "current_best_capability": results[best]["metrics"]["routed_score"],
        "current_failure_mode": ("NONE_AT_F1_CLUSTER" if survivors else
                                 "ROUTED_EXPERT_OUTPUT_REPRESENTATION_LOSS"),
        "current_dominant_bottleneck": ("DOWNSTREAM_PROPAGATION_UNTESTED" if survivors else
                                        "WEIGHT_SPACE_OBJECTIVE_MISMATCH"),
        "current_scientific_hypothesis": hypothesis,
        "current_next_experiment": next_experiment,
        "decision": decision,
        "diagnosis_metrics_reason_next_decision": {
            "diagnosis": ("F1_CLUSTER_SURVIVAL" if survivors else
                          "BOTH_WEIGHT_SPACE_FAMILIES_FAIL_F1_CLUSTER"),
            "metrics": results[best]["metrics"]["routed_score"],
            "reason": hypothesis, "next_decision": decision,
        },
    })
    f1.atomic_json(output_dir / "KIMI_K26_P1_F1_GRAMMAR_ISLANDS_BRACKET.json", artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--f1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.source.resolve(strict=True), args.f1_dir.resolve(strict=True),
                     args.output_dir.resolve())
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
