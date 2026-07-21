#!/usr/bin/env python3.12
"""Doctor Prime causal-harness, treatment-library, and exact-byte auction for Kimi K2.6.

The synthetic causal run validates intervention direction and classification logic only.
It cannot diagnose an unbuilt compact candidate.  The auction is an F0 installed-byte
admission plan, not a claim that any compact payload already exists or preserves quality.
"""
from __future__ import annotations

import argparse
import datetime as dt
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np


REVISION = "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"
REPO = "moonshotai/Kimi-K2.6"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def seal(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    return {**unsigned, "seal_sha256": hashlib.sha256(canonical(unsigned)).hexdigest()}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"required valid JSON absent: {path}") from exc


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    value = np.exp(shifted)
    return value / value.sum(axis=-1, keepdims=True)


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    p = softmax(reference.astype(np.float64))
    q = softmax(candidate.astype(np.float64))
    kl_pq = np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30)), axis=-1)
    kl_qp = np.sum(q * (np.log(q + 1e-30) - np.log(p + 1e-30)), axis=-1)
    cosine = np.sum(reference * candidate, axis=-1) / (
        np.linalg.norm(reference, axis=-1) * np.linalg.norm(candidate, axis=-1) + 1e-30
    )
    return {
        "symmetric_kl": float(np.mean((kl_pq + kl_qp) / 2)),
        "logit_cosine": float(np.mean(cosine)),
        "argmax_agreement": float(np.mean(reference.argmax(-1) == candidate.argmax(-1))),
    }


class Twin:
    """Small deterministic Kimi-shaped seam model used to prove causal intervention wiring."""

    def __init__(self, seed: int, *, student: bool):
        rng = np.random.default_rng(seed)
        layers, hidden, experts, expert_hidden, vocab = 6, 32, 8, 48, 71
        self.layers, self.hidden, self.experts = layers, hidden, experts
        self.attention = rng.normal(0, 0.055, (layers, hidden, hidden)).astype(np.float32)
        self.router = rng.normal(0, 0.16, (layers, hidden, experts)).astype(np.float32)
        self.shared_up = rng.normal(0, 0.08, (layers, hidden, expert_hidden)).astype(np.float32)
        self.shared_down = rng.normal(0, 0.08, (layers, expert_hidden, hidden)).astype(np.float32)
        self.expert_up = rng.normal(
            0, 0.08, (layers, experts, hidden, expert_hidden)).astype(np.float32)
        self.expert_down = rng.normal(
            0, 0.08, (layers, experts, expert_hidden, hidden)).astype(np.float32)
        self.norm = rng.normal(1, 0.02, hidden).astype(np.float32)
        self.head = rng.normal(0, 0.09, (hidden, vocab)).astype(np.float32)
        if student:
            # The harness has a known ground truth: routed-expert output damage dominates,
            # with smaller attention/router/shared/head changes still observable.
            self.attention += rng.normal(0, 0.004, self.attention.shape).astype(np.float32)
            self.router += rng.normal(0, 0.025, self.router.shape).astype(np.float32)
            self.shared_down += rng.normal(0, 0.006, self.shared_down.shape).astype(np.float32)
            self.expert_down += rng.normal(0, 0.075, self.expert_down.shape).astype(np.float32)
            self.head += rng.normal(0, 0.004, self.head.shape).astype(np.float32)


def rms(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def twin_forward(x: np.ndarray, teacher: Twin, student: Twin, spec: dict[str, Any],
                 teacher_hidden: list[np.ndarray] | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    hidden = x.astype(np.float32)
    states = [hidden.copy()]
    for layer in range(teacher.layers):
        attn_model = teacher if spec["attention"] == "teacher" else student
        routed_model = teacher if spec["routed_experts"] == "teacher" else student
        shared_model = teacher if spec["shared_expert"] == "teacher" else student
        router_model = teacher if spec["router"] == "teacher" else student
        normalized = rms(hidden)
        # A bounded causal surrogate for MLA: token mixing plus a learned projection.
        mixed = np.cumsum(normalized, axis=1) / np.arange(1, normalized.shape[1] + 1)[None, :, None]
        hidden = hidden + np.tanh(mixed @ attn_model.attention[layer])
        normalized = rms(hidden)
        router_score = 1 / (1 + np.exp(-(normalized @ router_model.router[layer])))
        route = np.argpartition(router_score, -2, axis=-1)[..., -2:]
        weights = np.take_along_axis(router_score, route, axis=-1)
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-20
        routed = np.zeros_like(hidden)
        for expert in range(teacher.experts):
            mask = route == expert
            token_rows, token_cols, slots = np.where(mask)
            if not len(token_rows):
                continue
            selected = normalized[token_rows, token_cols]
            output = np.tanh(selected @ routed_model.expert_up[layer, expert]) @ \
                routed_model.expert_down[layer, expert]
            routed[token_rows, token_cols] += output * weights[token_rows, token_cols, slots, None]
        shared = np.tanh(normalized @ shared_model.shared_up[layer]) @ shared_model.shared_down[layer]
        hidden = hidden + routed + shared
        if teacher_hidden is not None and layer in spec.get("reset_after", ()):
            hidden = teacher_hidden[layer + 1].copy()
        states.append(hidden.copy())
    final_model = teacher if spec["final"] == "teacher" else student
    normalized = rms(hidden) * final_model.norm
    return normalized @ final_model.head, states


def causal_harness(corpus_path: Path, parent_path: Path, output: Path) -> dict[str, Any]:
    corpus = read_json(corpus_path)
    parent = read_json(parent_path)
    if corpus.get("status") != "PASS":
        raise RuntimeError("clean-corpus gate is not PASS")
    if parent.get("status") != "PASS":
        raise RuntimeError("coherent parent-forward validation is not PASS")
    teacher, student = Twin(260621, student=False), Twin(260621, student=True)
    rng = np.random.default_rng(66102)
    inputs = rng.normal(0, 1, (5, 11, teacher.hidden)).astype(np.float32)
    teacher_spec = {key: "teacher" for key in
                    ("attention", "router", "routed_experts", "shared_expert", "final")}
    student_spec = {key: "student" for key in teacher_spec}
    reference, teacher_states = twin_forward(inputs, teacher, student, teacher_spec)
    replay, _ = twin_forward(inputs, teacher, student, teacher_spec)
    if not np.array_equal(reference, replay):
        raise RuntimeError("synthetic teacher replay is not bit-deterministic")
    full_student, _ = twin_forward(inputs, teacher, student, student_spec)
    full_metric = metrics(reference, full_student)
    if full_metric["symmetric_kl"] <= 0:
        raise RuntimeError("corruption control failed to move logits")

    controls = {
        "teacher_attention_mla__student_moe": {**student_spec, "attention": "teacher"},
        "student_attention_mla__teacher_moe": {
            **student_spec, "router": "teacher", "routed_experts": "teacher",
            "shared_expert": "teacher",
        },
        "teacher_router__student_experts": {**student_spec, "router": "teacher"},
        "student_router__teacher_expert_outputs": {**student_spec, "routed_experts": "teacher"},
        "teacher_shared_expert__student_routed_experts": {
            **student_spec, "shared_expert": "teacher"},
        "student_shared_expert__teacher_routed_experts": {
            **student_spec, "routed_experts": "teacher"},
        "teacher_final_norm_unembed": {**student_spec, "final": "teacher"},
        "hidden_reset_early": {**student_spec, "reset_after": (1,)},
        "hidden_reset_middle": {**student_spec, "reset_after": (3,)},
        "hidden_reset_late": {**student_spec, "reset_after": (5,)},
    }
    rows = []
    for control_id, spec in controls.items():
        logits, _ = twin_forward(inputs, teacher, student, spec, teacher_states)
        score = metrics(reference, logits)
        score["recovery_fraction"] = float(
            1 - score["symmetric_kl"] / full_metric["symmetric_kl"]
        )
        rows.append({"control": control_id, "metrics": score})

    by_id = {row["control"]: row["metrics"]["recovery_fraction"] for row in rows}
    axes = {
        "ATTENTION_MLA_BOUND": by_id["teacher_attention_mla__student_moe"],
        "ROUTING_BOUND": by_id["teacher_router__student_experts"],
        "SHARED_EXPERT_BOUND": by_id["teacher_shared_expert__student_routed_experts"],
        "ROUTED_EXPERT_OUTPUT_BOUND":
            by_id["student_router__teacher_expert_outputs"],
        # A late reset trivially replaces almost the complete computation and therefore is
        # not a localization score.  The early reset measures whether early damage persists
        # through otherwise-student middle/late blocks; all three resets remain reported.
        "RESIDUAL_PROPAGATION_BOUND": by_id["hidden_reset_early"],
        "LOGIT_HEAD_BOUND": by_id["teacher_final_norm_unembed"],
    }
    classification = max(axes, key=axes.get)
    passed = (
        classification == "ROUTED_EXPERT_OUTPUT_BOUND" and
        len(rows) == len(controls) and
        all(math.isfinite(value) for row in rows for value in row["metrics"].values())
    )
    artifact = seal({
        "schema": "hawking.kimi_k26.causal_atlas.v1",
        "status": "PASS" if passed else "FAIL", "sealed_at": now(),
        "source": {"repo": REPO, "revision": REVISION},
        "corpus_seal_sha256": corpus["seal_sha256"],
        "parent_validation_seal_sha256": parent["seal_sha256"],
        "evidence_level": "F0_SYNTHETIC_ARCHITECTURE_PRESERVING_TWIN",
        "claim_boundary": (
            "INTERVENTION_WIRING_AND_CLASSIFIER_VALIDATED; no real compact candidate "
            "diagnosis is claimed"
        ),
        "known_injected_ground_truth": "ROUTED_EXPERT_OUTPUT_BOUND",
        "classified_ground_truth": classification,
        "real_parent_candidate_classification": "PENDING_FIRST_F1_COMPACT_CANDIDATE",
        "teacher_replay_exact": True, "full_student_metrics": full_metric,
        "axis_recovery_fraction": axes, "controls": rows,
        "required_real_controls": list(controls),
        "classes_supported": [
            "ATTENTION_MLA_BOUND", "ROUTING_BOUND", "SHARED_EXPERT_BOUND",
            "ROUTED_EXPERT_OUTPUT_BOUND", "RESIDUAL_PROPAGATION_BOUND",
            "LOGIT_HEAD_BOUND", "MIXED_BOUND",
        ],
    })
    atomic_json(output, artifact)
    return artifact


TREATMENTS = [
    {"id": "organ_specific_base", "status": "OPEN", "targets": ["attention_mla", "routed_expert"]},
    {"id": "protected_tensors_directions", "status": "OPEN", "targets": ["norm", "router", "shared_expert"]},
    {"id": "residual_additive_codebooks", "status": "OPEN", "targets": ["routed_expert_output"]},
    {"id": "hidden_state_repair", "status": "OPEN", "targets": ["residual_propagation"]},
    {"id": "router_logit_correction", "status": "GATED_BY_ROUTING_DIAGNOSIS", "targets": ["router"]},
    {"id": "expert_output_fallback", "status": "OPEN", "targets": ["weighted_moe_output"]},
    {"id": "shared_expert_protection", "status": "OPEN", "targets": ["shared_expert"]},
    {"id": "weighted_moe_output_repair", "status": "OPEN", "targets": ["routed_expert_output"]},
    {"id": "normalization_residual_correction", "status": "OPEN", "targets": ["norm", "residual"]},
    {"id": "logit_ranking_repair", "status": "GATED_BY_LOGIT_HEAD_DIAGNOSIS", "targets": ["lm_head"]},
    {"id": "trainable_codebooks", "status": "OPEN", "targets": ["compressible_weights"]},
    {"id": "clean_data_distillation", "status": "OPEN", "targets": ["all_text_core"]},
    {"id": "conditional_token_triggered_repair", "status": "BILLS_INSTALLED_BITS", "targets": ["activation"]},
    {"id": "structural_redesign", "status": "GATED_BY_CAUSAL_JUSTIFICATION", "targets": ["architecture"]},
]


CANDIDATES = [
    ("P1", Fraction(49, 50), Fraction(75, 100), Fraction(23, 100), "BASE_HEAVY"),
    ("P2", Fraction(49, 50), Fraction(57, 100), Fraction(40, 100), "BALANCED_DOCTOR"),
    ("P3", Fraction(49, 50), Fraction(37, 100), Fraction(60, 100), "DOCTOR_HEAVY"),
    ("P4", Fraction(3, 4), Fraction(62, 100), Fraction(35, 100), "ARCHITECTURE_SPECIFIC_075"),
    ("P5", Fraction(1, 2), Fraction(52, 100), Fraction(45, 100), "AGGRESSIVE_050"),
]


def doctor_auction(ledger_path: Path, causal_path: Path, output: Path) -> dict[str, Any]:
    ledger, causal = read_json(ledger_path), read_json(causal_path)
    if causal.get("status") != "PASS":
        raise RuntimeError("causal intervention harness is not PASS")
    denominators = {
        "all_kimi_logical_weights": int(ledger["all_logical_original_weights"]),
        "kimi_text_core_logical_weights": int(ledger["text_core_logical_weights"]),
        "compressible_logical_weights": int(ledger["compressible_logical_weights"]),
        "active_logical_weights": int(ledger["active_text_core_logical_weights_per_token"]),
    }
    rows = []
    for candidate_id, target, base_share, doctor_share, envelope in CANDIDATES:
        total_bytes = (denominators["all_kimi_logical_weights"] * target.numerator //
                       target.denominator) // 8
        base_bytes = total_bytes * base_share.numerator // base_share.denominator
        doctor_bytes = total_bytes * doctor_share.numerator // doctor_share.denominator
        overhead_bytes = total_bytes - base_bytes - doctor_bytes
        installed_bits = total_bytes * 8
        rates = {
            name: {"numerator_bits": installed_bits, "denominator_weights": count,
                   "decimal_bpw": installed_bits / count}
            for name, count in denominators.items()
        }
        legal = installed_bits * target.denominator <= (
            denominators["all_kimi_logical_weights"] * target.numerator)
        rows.append({
            "candidate": candidate_id, "envelope": envelope,
            "target_complete_bpw": f"{target.numerator}/{target.denominator}",
            "target_complete_bpw_decimal": float(target),
            "installed_byte_ceiling": total_bytes,
            "allocation_bytes": {"compact_base": base_bytes, "doctor": doctor_bytes,
                                 "serialization_runtime_overhead": overhead_bytes},
            "allocation_fraction_of_total": {
                "compact_base": base_bytes / total_bytes, "doctor": doctor_bytes / total_bytes,
                "serialization_runtime_overhead": overhead_bytes / total_bytes,
            },
            "complete_bpw_by_denominator": rates, "hard_rate_law_pass": legal,
            "artifact_status": "F0_BYTE_BUDGET_ONLY_NOT_PACKED",
        })
    passed = all(row["hard_rate_law_pass"] for row in rows)
    artifact = seal({
        "schema": "hawking.kimi_k26.doctor_byte_auction.v1",
        "status": "PASS" if passed else "FAIL", "sealed_at": now(),
        "source": {"repo": REPO, "revision": REVISION},
        "ledger_seal_sha256": ledger["seal_sha256"],
        "causal_harness_seal_sha256": causal["seal_sha256"],
        "claim_boundary": (
            "F0 exact installed-byte ceilings pass; no compact payload, realized BPW, "
            "or compact capability result is claimed"
        ),
        "hard_rate_law": "installed_bits / all Kimi logical weights <= candidate target <= 1",
        "denominators": denominators, "doctor_gets_no_free_bytes": True,
        "conditional_treatments_bill_installed_bits": True,
        "blocked_qwen_defaults": [
            "uniform_frozen_weight_pq_vq", "fixed_expert_omission",
            "weight_cosine_expert_merging", "weight_similarity_shared_bases",
            "entropy_coding_primary_lever", "post_hoc_scalar_gain",
            "qwen_form_router_distillation", "qwen_form_layerwise_qat",
            "global_gamma_weighted_coding",
        ],
        "treatment_library": TREATMENTS, "rows": rows,
    })
    atomic_json(output, artifact)
    return artifact


def tournament(auction_path: Path, parent_path: Path, output: Path,
               checkpoint_output: Path) -> dict[str, Any]:
    auction, parent = read_json(auction_path), read_json(parent_path)
    if auction.get("status") != "PASS" or parent.get("status") != "PASS":
        raise RuntimeError("auction and parent validation must both pass")
    candidates = [{
        "candidate": "P0", "role": "official parent text-core reference",
        "fidelity": "F5_PARENT_REFERENCE", "status": "SEALED",
        "quality": {"coherent_probe_count": parent["coherent_probe_count"],
                    "finite_probe_count": parent["finite_probe_count"],
                    "deterministic_replay": parent["deterministic_replay"]["status"]},
        "complete_physical_bpw": None,
    }]
    for row in auction["rows"]:
        candidates.append({
            "candidate": row["candidate"], "role": row["envelope"],
            "fidelity": "F0_EXACT_BYTE_ADMISSION", "status": "ADMITTED_PENDING_F1",
            "target_complete_bpw": row["target_complete_bpw"],
            "complete_physical_bpw": None,
            "next_gate": "F1 single-layer output-space representation probe on disjoint corpus",
        })
    artifact = seal({
        "schema": "hawking.kimi_k26.tournament.v1", "status": "PASS",
        "tournament_state": "ACTIVE", "sealed_at": now(),
        "source": {"repo": REPO, "revision": REVISION},
        "claim_boundary": "P0 is real parent evidence; P1-P5 are F0 admissions, not packed results",
        "max_full_fidelity_candidates": 6, "candidate_count": 6,
        "fidelity_funnel": ["F0", "F1", "F2", "F3", "F4", "F5"],
        "retirement_rule": "fail closed at each gate; never infer family death from one mismatched configuration",
        "current_best": "P0_OFFICIAL_PARENT_REFERENCE",
        "complete_physical_bpw_best_compact": None,
        "next_experiment": "P1_AND_P5_F1_REPRESENTATION_BRACKET",
        "candidates": candidates,
    })
    atomic_json(output, artifact)
    checkpoint = seal({
        "schema": "hawking.kimi_k26.first_scientific_checkpoint.v1", "status": "PASS",
        "sealed_at": now(), "checkpoint_id": "P0_PARENT_AND_F0_DOCTOR_ADMISSION",
        "source": {"repo": REPO, "revision": REVISION},
        "parent_validation_seal_sha256": parent["seal_sha256"],
        "auction_seal_sha256": auction["seal_sha256"],
        "tournament_seal_sha256": artifact["seal_sha256"],
        "sealed_evidence": [
            "coherent deterministic full text-core parent forward",
            "clean corpus gate", "causal intervention harness",
            "exact F0 installed-byte auction",
        ],
        "next_experiment": "P1_AND_P5_F1_REPRESENTATION_BRACKET",
        "next_experiment_status": "ADVANCING_UNDER_DETACHED_CONTROLLER",
        "compact_capability_claim": "NONE_YET",
    })
    atomic_json(checkpoint_output, checkpoint)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    causal = sub.add_parser("causal")
    causal.add_argument("--corpus", type=Path, required=True)
    causal.add_argument("--parent", type=Path, required=True)
    causal.add_argument("--output", type=Path, required=True)
    auction = sub.add_parser("auction")
    auction.add_argument("--ledger", type=Path, required=True)
    auction.add_argument("--causal", type=Path, required=True)
    auction.add_argument("--output", type=Path, required=True)
    tour = sub.add_parser("tournament")
    tour.add_argument("--auction", type=Path, required=True)
    tour.add_argument("--parent", type=Path, required=True)
    tour.add_argument("--output", type=Path, required=True)
    tour.add_argument("--checkpoint-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "causal":
            result = causal_harness(args.corpus, args.parent, args.output)
        elif args.command == "auction":
            result = doctor_auction(args.ledger, args.causal, args.output)
        else:
            result = tournament(args.auction, args.parent, args.output,
                                args.checkpoint_output)
        print(json.dumps({"status": result["status"], "seal_sha256": result["seal_sha256"]},
                         sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
