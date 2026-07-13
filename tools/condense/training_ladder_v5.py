#!/usr/bin/env python3.12
"""Compile and validate the capability-first Hawking training ladder v5.

This module is a deterministic, stdlib-only planner.  It snapshots every model
in :mod:`ladder`, expands the complete model/rate/claim-track research matrix,
and binds that matrix to the fail-closed contracts in
:mod:`doctor_v5_contract`.  It does not load a model, inspect live Studio
state, acquire the heavy-work lease, import a training framework, or launch a
process.

Commands::

    python training_ladder_v5.py compile
    python training_ladder_v5.py validate [path]
    python training_ladder_v5.py selftest

``compile`` is the only mutating command.  It atomically writes the single
declared planning artifact ``reports/condense/training_ladder_v5.json``.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable

import doctor_v5_contract as doctor_contract
import quality_battery_v5
from ladder import MODELS, WEIGHT_BUDGET


SCHEMA = "hawking.training_ladder.v5"
LADDER_VERSION = "training-ladder-v5.0"
EXPECTED_MODEL_COUNT = 32
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "reports" / "condense" / "training_ladder_v5.json"

RATE_POINTS = (4.0, 3.0, 2.0, 1.0, 0.8, 0.55, 0.5, 0.33, 0.25, 0.1)
STAGE_IDS = tuple(f"L{index}" for index in range(11))
CLAIM_TRACKS = tuple(doctor_contract.CLAIM_SCOPES)

SOURCE_CLASS_IDS = (
    "resident_parent_source",
    "streamed_parent_source",
    "frontier_sharded_parent_source",
)
RESOURCE_CLASS_IDS = (
    "resident_single_host_research",
    "streamed_single_host_research",
    "frontier_out_of_core_research",
)

CONTROL_IDS = (
    "full_precision_parent",
    "untreated_same_rate",
    "zero_correction",
    "scalar_ptq",
    "codec_native_qat",
    "representation_reset",
    "progressive_inherited",
    "same_physical_byte_best_known",
)
def _canonical_hash(value: Any) -> str:
    return doctor_contract.hash_value(value)


def _identity_payload(document: dict[str, Any], field: str) -> dict[str, Any]:
    payload = copy.deepcopy(document)
    payload.pop(field, None)
    return payload


def _stamp(document: dict[str, Any], field: str) -> dict[str, Any]:
    output = copy.deepcopy(document)
    output[field] = _canonical_hash(_identity_payload(output, field))
    return output


def _finite(value: Any, *, positive: bool = False, nonnegative: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if not math.isfinite(number):
        return False
    if positive and number <= 0:
        return False
    if nonnegative and number < 0:
        return False
    return True


def _normalized_model(model: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON-stable planning snapshot imported from ``ladder.MODELS``.

    The legacy ``params_b`` field is a rounded catalogue estimate.  It is
    useful for coarse scheduling but cannot support an exact physical-bpw
    claim.  V5 therefore carries explicit unresolved fields until a hashed
    source tensor manifest supplies an integer parameter count.
    """
    return {
        "family": model["family"],
        "name": model["name"],
        "hf_id": model["hf_id"],
        "params_b": float(model["params_b"]),
        "params_b_semantics": "rounded_catalogue_estimate_not_evidence",
        "exact_parameter_count": None,
        "parameter_manifest_sha256": None,
        "parameter_count_status": "source_tensor_manifest_required",
        "active_b": None if model.get("active_b") is None else float(model["active_b"]),
        "priority": int(model["priority"]),
        "keep_f16": model.get("keep_f16"),
        "note": model.get("note", ""),
    }


def _source_models() -> list[dict[str, Any]]:
    return [_normalized_model(model) for model in MODELS]


def _parameter_count_policy() -> dict[str, Any]:
    return {
        "legacy_params_b_is_rounded_estimate": True,
        "exact_integer_tensor_parameter_count_required_for_evidence": True,
        "source_tensor_manifest_sha256_required": True,
        "physical_bpw_formula": "whole_artifact_bytes*8/exact_integer_parameter_count",
    }


def _source_class(params_b: float) -> str:
    if params_b <= 16.0:
        return "resident_parent_source"
    if params_b <= 235.0:
        return "streamed_parent_source"
    return "frontier_sharded_parent_source"


def _resource_class(params_b: float) -> str:
    if params_b <= 16.0:
        return "resident_single_host_research"
    if params_b <= 235.0:
        return "streamed_single_host_research"
    return "frontier_out_of_core_research"


def _rate_token(rate: float) -> str:
    return f"{rate:g}".replace(".", "p")


def _rate_role(rate: float) -> str:
    return {
        4.0: "high_precision_anchor",
        3.0: "strong_compression_control",
        2.0: "representation_transition",
        1.0: "binary_boundary",
        0.8: "subbit_primary",
        0.55: "progressive_bridge",
        0.5: "half_bit_frontier",
        0.33: "one_third_bit_frontier",
        0.25: "terminal_resident_candidate",
        0.1: "destructive_stress_control",
    }[rate]


def _treatment_branches(rate: float) -> list[str]:
    if rate >= 3.0:
        return [
            "zero_treatment",
            "scalar_ptq",
            "codec_native_qat",
            "same_byte_public_champion",
        ]
    if rate == 2.0:
        return [
            "zero_treatment",
            "scalar_equal_byte",
            "vector_or_additive_representation",
            "codec_native_qat",
            "representation_reset",
        ]
    if rate == 1.0:
        return [
            "zero_treatment",
            "progressive_inherited",
            "representation_reset",
            "binary_factor_or_pattern",
            "sensitive_high_precision_branch",
        ]
    return [
        "zero_treatment",
        "progressive_inherited",
        "representation_reset",
        "binary_factor_or_pattern",
        "shared_parameter_grammar",
        "sensitive_high_precision_branch",
    ]


def _source_classes() -> list[dict[str, Any]]:
    return [
        {
            "id": "resident_parent_source",
            "params_b_range": {"minimum_exclusive": 0.0, "maximum_inclusive": 16.0},
            "source_mode": "verified_local_or_downloaded_parent",
            "required_access": "parent tensors, tokenizer, config, revision, chat template",
            "whole_parent_residency_assumed": True,
            "streaming_required": False,
        },
        {
            "id": "streamed_parent_source",
            "params_b_range": {"minimum_exclusive": 16.0, "maximum_inclusive": 235.0},
            "source_mode": "verified_sharded_parent",
            "required_access": "immutable shard map plus byte offsets and hashes",
            "whole_parent_residency_assumed": False,
            "streaming_required": True,
        },
        {
            "id": "frontier_sharded_parent_source",
            "params_b_range": {"minimum_exclusive": 235.0, "maximum_inclusive": None},
            "source_mode": "remote_or_local_verified_frontier_shards",
            "required_access": "immutable manifest, transactional shard fetch, global merge state",
            "whole_parent_residency_assumed": False,
            "streaming_required": True,
        },
    ]


def _resource_classes() -> list[dict[str, Any]]:
    common = {
        "heavy_lease_required_before_future_execution": True,
        "zero_swap_required": True,
        "normal_memory_pressure_required": True,
        "exact_resume_before_training_required": True,
        "concurrency_requires_measured_peak_wave_fit": True,
        "concurrency_requires_independent_atomic_checkpoints": True,
        "launch_permitted_by_this_ladder": False,
    }
    return [
        {
            "id": "resident_single_host_research",
            "params_b_range": {"minimum_exclusive": 0.0, "maximum_inclusive": 16.0},
            "execution_shape_if_later_authorized": "one resident parent/candidate treatment",
            "future_parallel_cap": 3,
            **common,
        },
        {
            "id": "streamed_single_host_research",
            "params_b_range": {"minimum_exclusive": 16.0, "maximum_inclusive": 235.0},
            "execution_shape_if_later_authorized": "bounded layer/block/shard windows",
            "future_parallel_cap": 1,
            **common,
        },
        {
            "id": "frontier_out_of_core_research",
            "params_b_range": {"minimum_exclusive": 235.0, "maximum_inclusive": None},
            "execution_shape_if_later_authorized": "multi-pass transactional out-of-core shards",
            "future_parallel_cap": 1,
            **common,
        },
    ]


def _track_definitions() -> list[dict[str, Any]]:
    return [
        {
            "id": "codec_fidelity",
            "claim": "codec/representation fidelity without attached Doctor repair or external inference",
            "training_teachers": ["exact_parent_for_reconstruction_only"],
            "doctor_repair_allowed": False,
            "stronger_teacher_allowed": False,
            "external_runtime_allowed": False,
            "external_runtime_bytes_ceiling": 0,
            "l8_action": "prove_zero_doctor_repair_and_zero_external_runtime_dependency",
        },
        {
            "id": "restorative_training",
            "claim": "standalone condensed artifact restores damage against its exact identity teacher",
            "training_teachers": ["exact_parent", "truth_oracle"],
            "doctor_repair_allowed": True,
            "stronger_teacher_allowed": False,
            "external_runtime_allowed": False,
            "external_runtime_bytes_ceiling": 0,
            "l8_action": "prove_identity_teacher_only_and_zero_external_runtime",
        },
        {
            "id": "capability_elevation",
            "claim": "standalone condensed artifact gains verified capability from stronger teachers",
            "training_teachers": ["exact_parent", "provenance_bound_stronger_teacher", "truth_oracle"],
            "doctor_repair_allowed": True,
            "stronger_teacher_allowed": True,
            "external_runtime_allowed": False,
            "external_runtime_bytes_ceiling": 0,
            "l8_action": "prove_training_only_teacher_dependency_and_zero_external_runtime",
        },
        {
            "id": "augmented_system",
            "claim": "fully billed system adds capability through retrieval, tools, or verifiers",
            "training_teachers": ["exact_parent", "truth_oracle_for_external_system_outputs"],
            "doctor_repair_allowed": True,
            "stronger_teacher_allowed": False,
            "external_runtime_allowed": True,
            "external_runtime_bytes_ceiling": "measured_not_assumed",
            "l8_action": "build_and_bill_retrieval_tool_verifier_plane",
        },
    ]


def _stage(
    stage_id: str,
    name: str,
    dependencies: Iterable[str],
    purpose: str,
    evidence: Iterable[str],
    gates: Iterable[str],
    failure_route: Iterable[str],
    scope_actions: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "id": stage_id,
        "name": name,
        "depends_on": list(dependencies),
        "purpose": purpose,
        "claim_tracks": list(CLAIM_TRACKS),
        "evidence_required": list(evidence),
        "promotion_gates": list(gates),
        "failure_route": list(failure_route),
        "scope_actions": scope_actions or {
            scope: "execute_stage_semantics_within_claim_scope" for scope in CLAIM_TRACKS
        },
        "source_classes": list(SOURCE_CLASS_IDS),
        "resource_classes": list(RESOURCE_CLASS_IDS),
        "executor_wired": False,
        "launch_permitted": False,
    }


def _stages() -> list[dict[str, Any]]:
    return [
        _stage(
            "L0",
            "evidence_quarantine",
            [],
            "Freeze identity, evidence, claim scope, and mutually isolated data vaults.",
            [
                "parent/config/tokenizer/chat-template hashes",
                "data split manifests and contamination scans",
                "preregistered capability and noninferiority contract",
            ],
            [
                "sealed final remains optimizer-inaccessible",
                "teacher and retrieval provenance is complete",
                "no benchmark item entered calibration or repair",
            ],
            ["invalidate_lineage"],
        ),
        _stage(
            "L1",
            "mechanistic_disease_atlas",
            ["L0"],
            "Classify preserved-but-noisy signal separately from computation collapse.",
            [
                "parent/condensed paired failure traces",
                "internal geometry and early-signal survival",
                "verified activation and weight-patch recovery",
            ],
            [
                "failure class and confidence recorded",
                "collapse forbids compensation-only treatment",
                "capability absence and evaluator artifacts separated",
            ],
            ["L0", "L2_representation_reset_required"],
        ),
        _stage(
            "L2",
            "equal_byte_representation_tournament",
            ["L1"],
            "Run every mandatory representation and zero-treatment control at equal physical bytes.",
            [
                "scalar, vector/additive, progressive, reset, and sub-bit structural arms",
                "actual metadata-inclusive byte projections",
                "local reconstruction plus causal-signal survival",
            ],
            [
                "all mandatory controls retained",
                "both inherited and representation-reset branches below two bits",
                "no dense parent fallback",
            ],
            ["L1", "close_rate_as_documented_negative"],
        ),
        _stage(
            "L3",
            "codec_native_reconstruction",
            ["L2"],
            "Train and reconstruct through the exact packed representation and decoder semantics.",
            [
                "block/shard reconstruction",
                "packed round-trip identity",
                "codec-native QAT and gradient-stability ablations",
                "exact-resume checkpoint receipt",
            ],
            [
                "actual file bytes at or below target",
                "training decoder equals packed decoder",
                "resume replay is bit-identical",
            ],
            ["L2", "reject_non_native_proxy"],
        ),
        _stage(
            "L4",
            "identity_and_uplift_restoration",
            ["L3"],
            "Restore verified behavior, distributions, internal geometry, and capability vectors.",
            [
                "parent distribution and task loss",
                "causally weighted CKA/geometry alignment",
                "claim-track-specific teacher tribunal receipts",
            ],
            [
                "core fidelity uses exact parent only",
                "uplift is not mislabeled restoration",
                "worst-domain lower bound is reported",
            ],
            ["L2", "L3", "reduce_objective_conflict"],
            {
                "codec_fidelity": "audit_codec_behavior_without_doctor_repair",
                "restorative_training": "train_only_against_exact_identity_teacher",
                "capability_elevation": "train_with_provenance_bound_stronger_teacher",
                "augmented_system": "restore_or_elevate_core_before_external_augmentation",
            },
        ),
        _stage(
            "L5",
            "active_failure_foundry",
            ["L4"],
            "Mine verified teacher/student gaps and oracle-preserving adversarial mutations.",
            [
                "parent-correct candidate-wrong failures",
                "failure taxonomy and causal signature",
                "held-out mutation families and replay set",
            ],
            [
                "truth oracle or quarantined disagreement",
                "selection/sealed prompts invisible to generator",
                "diversity and forgetting are measured",
            ],
            ["L4", "quarantine_unverified_teacher_data"],
            {
                "codec_fidelity": "mine_failures_for_diagnosis_only_no_repair_training",
                "restorative_training": "mine_parent_candidate_restoration_failures",
                "capability_elevation": "mine_verified_teacher_student_capability_gaps",
                "augmented_system": "mine_core_and_external_system_failure_boundaries",
            },
        ),
        _stage(
            "L6",
            "targeted_doctor_synthesis",
            ["L5"],
            "Synthesize the smallest causal static, capability-bank, or conditional repair.",
            [
                "capability/component/treatment topology",
                "treatment-on/off counterfactual repair labels",
                "severity-weighted gate errors and collateral regressions",
            ],
            [
                "zero treatment remains eligible",
                "every targeted repair has intervention and ablation evidence",
                "all repair/router bytes are billed",
            ],
            ["L2", "L5", "replace_gate_with_static_protection"],
            {
                "codec_fidelity": "prove_no_doctor_repair_is_attached",
                "restorative_training": "synthesize_parent_restorative_treatment",
                "capability_elevation": "synthesize_standalone_capability_treatment",
                "augmented_system": "synthesize_core_treatment_before_external_plane",
            },
        ),
        _stage(
            "L7",
            "verified_reasoning_restoration",
            ["L6"],
            "Restore reasoning using verified solutions, mistakes, backtracking, and student rollouts.",
            [
                "executable or formal answer verification",
                "verified correction/backtracking traces",
                "on-policy selective-imitation and no-RL controls",
            ],
            [
                "answer-correct but invalid traces are rejected",
                "reasoning, knowledge, and tool dependence are separated",
                "parent-good replay prevents forgetting",
            ],
            ["L4", "L5", "disable_unverified_reasoning_source"],
            {
                "codec_fidelity": "audit_reasoning_damage_without_reasoning_treatment",
                "restorative_training": "restore_verified_parent_reasoning",
                "capability_elevation": "elevate_verified_reasoning_with_stronger_teacher",
                "augmented_system": "train_core_reasoning_before_tool_or_verifier_use",
            },
        ),
        _stage(
            "L8",
            "augmentation_plane_or_scope_firewall",
            ["L7"],
            "Build fully billed external capability only for the augmented track; prove absence for core tracks.",
            [
                "track-specific L8 action",
                "closed-book and augmented paired evaluation",
                "retrieval/tool/verifier provenance and byte ledger",
            ],
            [
                "core tracks use zero external runtime bytes",
                "augmented gains never support a core claim",
                "retrieval indexes pass the data firewall",
            ],
            ["L7", "invalidate_claim_scope"],
            {
                "codec_fidelity": "prove_zero_doctor_and_external_runtime_dependency",
                "restorative_training": "prove_zero_external_runtime_dependency",
                "capability_elevation": "prove_stronger_teacher_is_training_only",
                "augmented_system": "build_and_bill_retrieval_tool_verifier_plane",
            },
        ),
        _stage(
            "L9",
            "robustness_alignment_calibration_hardening",
            ["L8"],
            "Harden metamorphic robustness, instruction alignment, safety, and uncertainty.",
            [
                "counterfactual and logically invariant test families",
                "safety/helpfulness/instruction tripwires",
                "calibration and selective-risk curves",
            ],
            [
                "no protected capability regresses beyond preregistered margin",
                "metamorphic violations and prompt variance are bounded",
                "low perplexity cannot substitute for alignment evidence",
            ],
            ["L4", "L5", "L6", "invalidate_unsafe_candidate"],
            {
                "codec_fidelity": "audit_robustness_alignment_and_calibration_without_hardening",
                "restorative_training": "harden_without_exceeding_parent_information_claim",
                "capability_elevation": "harden_elevated_standalone_artifact",
                "augmented_system": "harden_core_and_external_system_boundaries",
            },
        ),
        _stage(
            "L10",
            "sealed_champion_audit",
            ["L9"],
            "Run sealed post-cutoff evaluation and independent same-budget reproduction.",
            [
                "paired per-item outputs on every capability domain",
                "five or more independent training/calibration seeds and generation-seed clusters",
                "Holm-corrected confidence intervals",
                "same-budget competitor reproductions",
            ],
            [
                "parent noninferior in every protected domain",
                "every required same-budget competitor is dominated",
                "independent reproduction reaches the same verdict",
            ],
            ["L2", "L4", "L5", "L9", "retain_complete_negative"],
        ),
    ]


def _data_firewall() -> dict[str, Any]:
    contract = doctor_contract.planned_data_contract()
    return {
        "data_contract": contract,
        "split_names": list(doctor_contract.DATA_SPLITS),
        "gates": {
            "content_hash_distinct": True,
            "exact_duplicate_scan": True,
            "near_duplicate_scan": True,
            "semantic_contamination_scan": True,
            "mutation_family_split_before_generation": True,
            "teacher_cache_split_bound": True,
            "retrieval_index_excludes_evaluation": True,
            "selection_hidden_from_failure_generator": True,
            "sealed_hidden_until_training_frozen": True,
            "violation_invalidates_lineage": True,
        },
    }


def _exact_resume() -> dict[str, Any]:
    return {
        "required_before_any_mutating_stage": True,
        "mutating_stages": [f"L{index}" for index in range(2, 10)],
        "required_state": list(doctor_contract.EXACT_RESUME_STATE),
        "atomic_replace": True,
        "fsync_file": True,
        "fsync_parent_directory": True,
        "validate_before_resume": True,
        "source_shard_and_byte_offset_required": True,
        "resume_replay_identity_required": True,
    }


def _controls() -> list[dict[str, Any]]:
    descriptions = {
        "full_precision_parent": "Exact parent upper identity and capability reference.",
        "untreated_same_rate": "Same representation/rate before Doctor treatment.",
        "zero_correction": "Candidate with no repair bytes or runtime correction.",
        "scalar_ptq": "Conventional scalar PTQ at the same declared physical budget.",
        "codec_native_qat": "QAT through the actual representation and decoder.",
        "representation_reset": "Fresh structural reconstruction at or below two bits.",
        "progressive_inherited": "Parent-to-rate progressive continuation arm.",
        "same_physical_byte_best_known": "Strongest reproduced method at no greater physical bytes.",
    }
    applicable_rates = {
        "full_precision_parent": list(RATE_POINTS),
        "untreated_same_rate": list(RATE_POINTS),
        "zero_correction": list(RATE_POINTS),
        "scalar_ptq": [4.0, 3.0, 2.0, 1.0],
        "codec_native_qat": list(RATE_POINTS),
        "representation_reset": [rate for rate in RATE_POINTS if rate <= 2.0],
        "progressive_inherited": list(RATE_POINTS),
        "same_physical_byte_best_known": list(RATE_POINTS),
    }
    return [
        {
            "id": control_id,
            "description": descriptions[control_id],
            "applicable_rates": applicable_rates[control_id],
            "applies_to_all_models": True,
            "applies_to_claim_tracks": list(CLAIM_TRACKS),
            "mandatory": True,
            "negative_result_retained": True,
            "same_parent": True,
            "same_prompt_and_scorer": True,
        }
        for control_id in CONTROL_IDS
    ]


def _competitors() -> dict[str, Any]:
    # The quality battery owns the named, source- and rate-bound registry.  The
    # ladder embeds that canonical object verbatim so a broad family label can
    # never silently replace a required direct implementation.
    coverage = copy.deepcopy(quality_battery_v5._competitor_coverage())
    return {
        "quality_battery_schema": quality_battery_v5.SCHEMA,
        "quality_battery_competitor_coverage_sha256": _canonical_hash(coverage),
        "coverage_policy": coverage["coverage_policy"],
        "required_families": coverage["required_families"],
        "required_direct_implementations": coverage["direct_implementations"],
        "required_implementation_ids_by_rate_bpw": coverage[
            "required_implementation_ids_by_rate_bpw"
        ],
        "ladder_rates_bpw": coverage["ladder_rates_bpw"],
        "fairness_contract": {
            "same_parent": True,
            "same_or_lower_physical_bytes": True,
            "same_data_access": True,
            "same_teacher_budget_within_claim_track": True,
            "same_prompt_protocol": True,
            "same_scorer": True,
            "same_test_time_compute_budget": True,
            "same_output_token_budget": True,
            "same_samples_retries_and_calls": True,
            "same_augmentation_scope": True,
            "packed_artifact_required": True,
            "independent_reproduction_required_for_headline": True,
            "family_representative_substitution_forbidden": True,
            "signed_incompatibility_and_narrowed_claim_required_when_unavailable": True,
        },
    }


def _matched_compute_policy() -> dict[str, Any]:
    """Embed the battery's complete canonical matched-compute contract."""
    return copy.deepcopy(
        quality_battery_v5.compile_manifest()["matched_test_time_compute"]
    )


def _quality_policy() -> dict[str, Any]:
    return {
        "quality_first": True,
        "speed_deferred": True,
        "speed_may_break_quality_ties": False,
        "speed_claim_before_separate_runtime_ladder_forbidden": True,
        "wall_clock_campaign_cutoff": None,
        "checkpointing_required_despite_unbounded_wall_clock": True,
        "objective_order": [
            "all_domain_parent_noninferiority",
            "worst_domain_quality",
            "same_budget_competitor_dominance",
            "physical_capability_density",
        ],
        "headline_unbeatable_language_before_independent_reproduction_forbidden": True,
        "average_score_cannot_mask_protected_domain_regression": True,
        "perplexity_cannot_substitute_for_capability": True,
        "codec_restoration_elevation_augmented_scores_separate": True,
        "matched_test_time_compute_required": True,
        "underpowered_result_is_inconclusive": True,
        "minimum_independent_training_seeds": 5,
        "exact_integer_parameter_count_required_for_physical_bpw": True,
        "rounded_params_b_for_evidence_forbidden": True,
    }


def _rate_profiles() -> list[dict[str, Any]]:
    profiles = []
    for rate in RATE_POINTS:
        profiles.append(
            {
                "physical_bpw_ceiling": rate,
                "role": _rate_role(rate),
                "applies_to_all_models": True,
                "destructive_control_only": rate == 0.1,
                "below_two_requires_representation_reset_arm": rate < 2.0,
                "collapse_forbids_compensation_only": True,
                "required_treatment_branches": _treatment_branches(rate),
            }
        )
    return profiles


def _model_rows(source_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model in source_models:
        params_b = model["params_b"]
        normalized_sha = _canonical_hash(model)
        rows.append(
            {
                **model,
                "source_model_sha256": normalized_sha,
                "source_class": _source_class(params_b),
                "resource_class": _resource_class(params_b),
                "rate_points": list(RATE_POINTS),
                "claim_tracks": list(CLAIM_TRACKS),
                "stage_path": list(STAGE_IDS),
            }
        )
    return rows


def _lane(model: dict[str, Any], rate: float, claim_track: str) -> dict[str, Any]:
    nominal_bytes = max(1, int(model["params_b"] * 1_000_000_000 * rate / 8.0))
    lane_id = f"{model['name']}::{_rate_token(rate)}bpw::{claim_track}"
    payload = {
        "lane_id": lane_id,
        "model": model["name"],
        "model_source_sha256": model["source_model_sha256"],
        "physical_bpw_ceiling": rate,
        "rate_role": _rate_role(rate),
        "claim_track": claim_track,
        "source_class": model["source_class"],
        "resource_class": model["resource_class"],
        "nominal_weight_payload_bytes": nominal_bytes,
        "nominal_weight_payload_gb": round(nominal_bytes / 1_000_000_000, 6),
        "nominal_projection_semantics": "rounded_params_b_planning_estimate_not_evidence",
        "exact_parameter_count": None,
        "parameter_manifest_sha256": None,
        "exact_physical_bpw_computable": False,
        "nominal_resident_weight_fit": nominal_bytes <= int(WEIGHT_BUDGET * 1_000_000_000),
        "fit_disposition": (
            "resident_candidate_subject_to_actual_bytes"
            if nominal_bytes <= int(WEIGHT_BUDGET * 1_000_000_000)
            else "out_of_core_scientific_control"
        ),
        "required_treatment_branches": _treatment_branches(rate),
        "stage_path": list(STAGE_IDS),
        "status": "planned",
        "admission_blockers": [
            "exact_source_tensor_parameter_manifest_missing",
            "root_v5_manifest_missing",
            "user_greenlight_not_recorded",
            "executor_not_wired",
        ],
        "executor_wired": False,
        "launch_permitted": False,
        "speed_deferred": True,
    }
    return _stamp(payload, "lane_sha256")


def compile_ladder() -> dict[str, Any]:
    source_models = _source_models()
    if len(source_models) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"training ladder v5 is pinned to {EXPECTED_MODEL_COUNT} current models; "
            f"ladder.MODELS has {len(source_models)}"
        )
    names = [model["name"] for model in source_models]
    if len(names) != len(set(names)):
        raise ValueError("ladder.MODELS contains duplicate model names")

    models = _model_rows(source_models)
    lanes = [
        _lane(model, rate, claim_track)
        for model in models
        for rate in RATE_POINTS
        for claim_track in CLAIM_TRACKS
    ]
    document = {
        "schema": SCHEMA,
        "ladder_version": LADDER_VERSION,
        "doctor_policy_version": doctor_contract.POLICY_VERSION,
        "compiled_from": {
            "model_source": "ladder.MODELS",
            "model_count": len(source_models),
            "model_snapshot_sha256": _canonical_hash(source_models),
            "parameter_count_policy": _parameter_count_policy(),
            "doctor_program_schema": doctor_contract.PROGRAM_SCHEMA,
            "doctor_artifact_schema": doctor_contract.ARTIFACT_SCHEMA,
            "doctor_observation_schema": doctor_contract.OBSERVATION_SCHEMA,
            "doctor_dominance_schema": doctor_contract.DOMINANCE_SCHEMA,
        },
        "execution_policy": {
            "planner_only": True,
            "launches_processes": False,
            "loads_models": False,
            "reads_live_state": False,
            "touches_live_state": False,
            "only_declared_output": str(DEFAULT_REPORT.relative_to(ROOT)),
        },
        "quality_policy": _quality_policy(),
        "matched_test_time_compute": _matched_compute_policy(),
        "claim_tracks": _track_definitions(),
        "source_classes": _source_classes(),
        "resource_classes": _resource_classes(),
        "rate_profiles": _rate_profiles(),
        "stages": _stages(),
        "data_firewall": _data_firewall(),
        "exact_resume_gate": _exact_resume(),
        "controls": _controls(),
        "competitor_requirements": _competitors(),
        "capability_domains": list(doctor_contract.CAPABILITY_DOMAINS),
        "models": models,
        "lanes": lanes,
        "counts": {
            "models": len(models),
            "rates": len(RATE_POINTS),
            "claim_tracks": len(CLAIM_TRACKS),
            "stages": len(STAGE_IDS),
            "research_lanes": len(lanes),
            "referenced_stage_cells": len(lanes) * len(STAGE_IDS),
        },
    }
    document = _stamp(document, "ladder_sha256")
    errors = validate_ladder(document)
    if errors:
        raise AssertionError("compiler emitted invalid ladder:\n- " + "\n- ".join(errors))
    return document


def _cycle(stages: dict[str, dict[str, Any]]) -> list[str] | None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(stage_id: str, trail: list[str]) -> list[str] | None:
        if stage_id in visiting:
            start = trail.index(stage_id) if stage_id in trail else 0
            return trail[start:] + [stage_id]
        if stage_id in visited:
            return None
        visiting.add(stage_id)
        for dependency in stages[stage_id].get("depends_on", []):
            if dependency in stages:
                found = visit(dependency, trail + [stage_id])
                if found:
                    return found
        visiting.remove(stage_id)
        visited.add(stage_id)
        return None

    for stage_id in stages:
        found = visit(stage_id, [])
        if found:
            return found
    return None


def _validate_execution_policy(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    policy = document.get("execution_policy")
    expected = {
        "planner_only": True,
        "launches_processes": False,
        "loads_models": False,
        "reads_live_state": False,
        "touches_live_state": False,
        "only_declared_output": str(DEFAULT_REPORT.relative_to(ROOT)),
    }
    if policy != expected:
        errors.append("execution_policy must be the exact execution-free v5 policy")
    return errors


def _validate_quality_policy(document: dict[str, Any]) -> list[str]:
    policy = document.get("quality_policy")
    if not isinstance(policy, dict):
        return ["quality_policy must be an object"]
    errors: list[str] = []
    required_true = (
        "quality_first",
        "speed_deferred",
        "speed_claim_before_separate_runtime_ladder_forbidden",
        "checkpointing_required_despite_unbounded_wall_clock",
        "headline_unbeatable_language_before_independent_reproduction_forbidden",
        "average_score_cannot_mask_protected_domain_regression",
        "perplexity_cannot_substitute_for_capability",
        "codec_restoration_elevation_augmented_scores_separate",
        "matched_test_time_compute_required",
        "underpowered_result_is_inconclusive",
    )
    for field in required_true:
        if policy.get(field) is not True:
            errors.append(f"quality_policy.{field} must be true")
    if policy.get("speed_may_break_quality_ties") is not False:
        errors.append("speed cannot break quality ties in this ladder")
    if policy.get("wall_clock_campaign_cutoff") is not None:
        errors.append("quality campaign must not have a wall-clock cutoff")
    if policy.get("minimum_independent_training_seeds") != 5:
        errors.append("quality campaign requires five independent training seeds")
    expected_order = [
        "all_domain_parent_noninferiority",
        "worst_domain_quality",
        "same_budget_competitor_dominance",
        "physical_capability_density",
    ]
    if policy.get("objective_order") != expected_order:
        errors.append("quality objective order changed")
    if policy != _quality_policy():
        errors.append("quality policy differs from the canonical v5 policy")
    return errors


def _validate_tracks(document: dict[str, Any]) -> list[str]:
    tracks = document.get("claim_tracks")
    if not isinstance(tracks, list):
        return ["claim_tracks must be a list"]
    errors: list[str] = []
    by_id = {row.get("id"): row for row in tracks if isinstance(row, dict)}
    if set(by_id) != set(CLAIM_TRACKS) or len(tracks) != len(CLAIM_TRACKS):
        errors.append("claim tracks must exactly match Doctor-v5 claim scopes")
        return errors
    codec = by_id["codec_fidelity"]
    restorative = by_id["restorative_training"]
    elevation = by_id["capability_elevation"]
    augmented = by_id["augmented_system"]
    if codec.get("doctor_repair_allowed") is not False \
            or codec.get("stronger_teacher_allowed") is not False \
            or codec.get("external_runtime_allowed") is not False:
        errors.append("codec_fidelity repair/teacher/runtime firewall weakened")
    if restorative.get("doctor_repair_allowed") is not True \
            or restorative.get("stronger_teacher_allowed") is not False \
            or restorative.get("external_runtime_allowed") is not False:
        errors.append("restorative_training must use identity-teacher standalone treatment")
    if elevation.get("doctor_repair_allowed") is not True \
            or elevation.get("stronger_teacher_allowed") is not True \
            or elevation.get("external_runtime_allowed") is not False:
        errors.append("capability_elevation must allow stronger training teachers but no external runtime")
    if augmented.get("doctor_repair_allowed") is not True \
            or augmented.get("stronger_teacher_allowed") is not False \
            or augmented.get("external_runtime_allowed") is not True:
        errors.append("augmented_system must bill external runtime without adding a stronger training teacher")
    for scope in (codec, restorative, elevation):
        if scope.get("external_runtime_bytes_ceiling") != 0:
            errors.append(f"{scope.get('id')} must have zero external runtime bytes")
    if tracks != _track_definitions():
        errors.append("claim-track definitions differ from the canonical v5 tracks")
    return errors


def _validate_classes(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source_rows = document.get("source_classes")
    resource_rows = document.get("resource_classes")
    source_ids = {
        row.get("id") for row in source_rows if isinstance(row, dict)
    } if isinstance(source_rows, list) else set()
    resource_ids = {
        row.get("id") for row in resource_rows if isinstance(row, dict)
    } if isinstance(resource_rows, list) else set()
    if source_ids != set(SOURCE_CLASS_IDS) or len(source_rows or []) != len(SOURCE_CLASS_IDS):
        errors.append("source classes are incomplete or duplicated")
    if resource_ids != set(RESOURCE_CLASS_IDS) or len(resource_rows or []) != len(RESOURCE_CLASS_IDS):
        errors.append("resource classes are incomplete or duplicated")
    if isinstance(resource_rows, list):
        for row in resource_rows:
            if not isinstance(row, dict):
                continue
            for field in (
                "heavy_lease_required_before_future_execution",
                "zero_swap_required",
                "normal_memory_pressure_required",
                "exact_resume_before_training_required",
            ):
                if row.get(field) is not True:
                    errors.append(f"resource class {row.get('id')} weakens {field}")
            if row.get("launch_permitted_by_this_ladder") is not False:
                errors.append(f"resource class {row.get('id')} permits ladder launch")
            if row.get("concurrency_requires_measured_peak_wave_fit") is not True \
                    or row.get("concurrency_requires_independent_atomic_checkpoints") is not True:
                errors.append(f"resource class {row.get('id')} weakens concurrency safety")
            expected_cap = 3 if row.get("id") == "resident_single_host_research" else 1
            if row.get("future_parallel_cap") != expected_cap:
                errors.append(f"resource class {row.get('id')} future parallel cap changed")
    if source_rows != _source_classes():
        errors.append("source class definitions differ from the canonical v5 classes")
    if resource_rows != _resource_classes():
        errors.append("resource class definitions differ from the canonical v5 classes")
    return errors


def _validate_stages(document: dict[str, Any]) -> list[str]:
    stages_raw = document.get("stages")
    if not isinstance(stages_raw, list):
        return ["stages must be a list"]
    errors: list[str] = []
    stages: dict[str, dict[str, Any]] = {}
    for index, stage in enumerate(stages_raw):
        if not isinstance(stage, dict):
            errors.append(f"stages[{index}] must be an object")
            continue
        stage_id = stage.get("id")
        if stage_id in stages:
            errors.append(f"duplicate stage {stage_id}")
        if not isinstance(stage_id, str):
            errors.append(f"stages[{index}].id missing")
            continue
        stages[stage_id] = stage
        if not isinstance(stage.get("depends_on"), list):
            errors.append(f"stage {stage_id} dependencies must be a list")
        if set(stage.get("claim_tracks", [])) != set(CLAIM_TRACKS):
            errors.append(f"stage {stage_id} claim-track coverage incomplete")
        if set(stage.get("source_classes", [])) != set(SOURCE_CLASS_IDS):
            errors.append(f"stage {stage_id} source-class coverage incomplete")
        if set(stage.get("resource_classes", [])) != set(RESOURCE_CLASS_IDS):
            errors.append(f"stage {stage_id} resource-class coverage incomplete")
        actions = stage.get("scope_actions")
        if not isinstance(actions, dict) or set(actions) != set(CLAIM_TRACKS) \
                or any(not isinstance(action, str) or not action for action in actions.values()):
            errors.append(f"stage {stage_id} scope actions are incomplete")
        if stage.get("executor_wired") is not False or stage.get("launch_permitted") is not False:
            errors.append(f"stage {stage_id} is not execution-free")
        for field in ("purpose", "evidence_required", "promotion_gates", "failure_route"):
            if not stage.get(field):
                errors.append(f"stage {stage_id}.{field} is empty")
    if tuple(stage.get("id") for stage in stages_raw if isinstance(stage, dict)) != STAGE_IDS:
        errors.append("stage order must be exactly L0 through L10")
    if set(stages) != set(STAGE_IDS) or len(stages_raw) != len(STAGE_IDS):
        errors.append("stage set must be exactly L0 through L10")
    for stage_id, stage in stages.items():
        for dependency in stage.get("depends_on", []):
            if dependency not in stages:
                errors.append(f"stage {stage_id} has unknown dependency {dependency}")
    cycle = _cycle(stages) if stages else None
    if cycle:
        errors.append("stage DAG cycle: " + " -> ".join(cycle))
    expected_dependencies = {"L0": []} | {f"L{index}": [f"L{index - 1}"] for index in range(1, 11)}
    for stage_id, expected in expected_dependencies.items():
        if stages.get(stage_id, {}).get("depends_on") != expected:
            errors.append(f"stage {stage_id} dependency must be {expected}")
    collapse_gates = " ".join(stages.get("L1", {}).get("promotion_gates", [])).lower()
    if "collapse forbids compensation-only" not in collapse_gates:
        errors.append("L1 must hard-route computation collapse away from compensation-only treatment")
    if stages_raw != _stages():
        errors.append("stage definitions differ from the canonical L0-L10 ladder")
    return errors


def _validate_firewall(document: dict[str, Any]) -> list[str]:
    firewall = document.get("data_firewall")
    if not isinstance(firewall, dict):
        return ["data_firewall must be an object"]
    errors = doctor_contract.validate_data_contract(firewall.get("data_contract"), "planned")
    if firewall.get("split_names") != list(doctor_contract.DATA_SPLITS):
        errors.append("data firewall split names differ from Doctor-v5")
    gates = firewall.get("gates")
    expected_gates = {
        "content_hash_distinct",
        "exact_duplicate_scan",
        "near_duplicate_scan",
        "semantic_contamination_scan",
        "mutation_family_split_before_generation",
        "teacher_cache_split_bound",
        "retrieval_index_excludes_evaluation",
        "selection_hidden_from_failure_generator",
        "sealed_hidden_until_training_frozen",
        "violation_invalidates_lineage",
    }
    if not isinstance(gates, dict) or set(gates) != expected_gates \
            or any(value is not True for value in gates.values()):
        errors.append("data firewall gates are incomplete or weakened")
    if firewall != _data_firewall():
        errors.append("data firewall differs from the canonical v5 firewall")
    return errors


def _validate_resume(document: dict[str, Any]) -> list[str]:
    resume = document.get("exact_resume_gate")
    if not isinstance(resume, dict):
        return ["exact_resume_gate must be an object"]
    errors: list[str] = []
    if set(resume.get("required_state", [])) != set(doctor_contract.EXACT_RESUME_STATE):
        errors.append("exact resume state differs from Doctor-v5")
    if resume.get("mutating_stages") != [f"L{index}" for index in range(2, 10)]:
        errors.append("exact resume mutating-stage coverage is incomplete")
    for field in (
        "required_before_any_mutating_stage",
        "atomic_replace",
        "fsync_file",
        "fsync_parent_directory",
        "validate_before_resume",
        "source_shard_and_byte_offset_required",
        "resume_replay_identity_required",
    ):
        if resume.get(field) is not True:
            errors.append(f"exact_resume_gate.{field} must be true")
    if resume != _exact_resume():
        errors.append("exact resume gate differs from the canonical v5 gate")
    return errors


def _validate_controls_and_competitors(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    controls = document.get("controls")
    control_ids = {
        row.get("id") for row in controls if isinstance(row, dict)
    } if isinstance(controls, list) else set()
    if control_ids != set(CONTROL_IDS) or len(controls or []) != len(CONTROL_IDS):
        errors.append("mandatory control set is incomplete or duplicated")
    if isinstance(controls, list):
        for row in controls:
            if not isinstance(row, dict):
                continue
            if any(row.get(field) is not True for field in (
                "mandatory", "negative_result_retained", "same_parent", "same_prompt_and_scorer"
            )):
                errors.append(f"control {row.get('id')} weakens comparison requirements")
    competitors = document.get("competitor_requirements")
    if not isinstance(competitors, dict):
        return errors + ["competitor_requirements must be an object"]
    canonical_coverage = quality_battery_v5._competitor_coverage()
    families = competitors.get("required_families")
    family_ids = {
        row.get("family_id") for row in families if isinstance(row, dict)
    } if isinstance(families, list) else set()
    expected_family_ids = {
        row["family_id"] for row in canonical_coverage["required_families"]
    }
    if family_ids != expected_family_ids or len(families or []) != len(expected_family_ids):
        errors.append("competitor family set is incomplete or duplicated")
    implementations = competitors.get("required_direct_implementations")
    implementation_ids = {
        row.get("implementation_id") for row in implementations if isinstance(row, dict)
    } if isinstance(implementations, list) else set()
    expected_implementation_ids = {
        row["implementation_id"] for row in canonical_coverage["direct_implementations"]
    }
    if implementation_ids != expected_implementation_ids \
            or len(implementations or []) != len(expected_implementation_ids):
        errors.append("direct competitor implementation set is incomplete or duplicated")
    if competitors.get("quality_battery_competitor_coverage_sha256") != _canonical_hash(
        canonical_coverage
    ):
        errors.append("quality-battery competitor coverage hash mismatched")
    fairness = competitors.get("fairness_contract")
    expected_fairness = {
        "same_parent",
        "same_or_lower_physical_bytes",
        "same_data_access",
        "same_teacher_budget_within_claim_track",
        "same_prompt_protocol",
        "same_scorer",
        "same_test_time_compute_budget",
        "same_output_token_budget",
        "same_samples_retries_and_calls",
        "same_augmentation_scope",
        "packed_artifact_required",
        "independent_reproduction_required_for_headline",
        "family_representative_substitution_forbidden",
        "signed_incompatibility_and_narrowed_claim_required_when_unavailable",
    }
    if not isinstance(fairness, dict) or set(fairness) != expected_fairness \
            or any(value is not True for value in fairness.values()):
        errors.append("same-budget competitor fairness contract is incomplete or weakened")
    if controls != _controls():
        errors.append("control definitions differ from the canonical v5 controls")
    if competitors != _competitors():
        errors.append("competitor requirements differ from the canonical v5 requirements")
    return errors


def _validate_rates(document: dict[str, Any]) -> list[str]:
    profiles = document.get("rate_profiles")
    if not isinstance(profiles, list):
        return ["rate_profiles must be a list"]
    errors: list[str] = []
    rates = [row.get("physical_bpw_ceiling") for row in profiles if isinstance(row, dict)]
    if rates != list(RATE_POINTS) or len(profiles) != len(RATE_POINTS):
        errors.append("rate profile order/coverage must match the complete v5 ladder")
    for row in profiles:
        if not isinstance(row, dict):
            errors.append("rate profile must be an object")
            continue
        rate = row.get("physical_bpw_ceiling")
        if rate not in RATE_POINTS:
            continue
        if row.get("role") != _rate_role(float(rate)):
            errors.append(f"rate {rate} role changed")
        if row.get("applies_to_all_models") is not True:
            errors.append(f"rate {rate} must cover all models, including negative controls")
        if row.get("required_treatment_branches") != _treatment_branches(float(rate)):
            errors.append(f"rate {rate} treatment branches changed")
        if row.get("below_two_requires_representation_reset_arm") is not (float(rate) < 2.0):
            errors.append(f"rate {rate} representation-reset policy wrong")
        if row.get("collapse_forbids_compensation_only") is not True:
            errors.append(f"rate {rate} permits compensation-only collapse treatment")
    if profiles != _rate_profiles():
        errors.append("rate profiles differ from the canonical v5 profiles")
    return errors


def _validate_models_and_lanes(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source_models = _source_models()
    models = document.get("models")
    if not isinstance(models, list):
        return ["models must be a list"]
    if len(source_models) != EXPECTED_MODEL_COUNT or len(models) != EXPECTED_MODEL_COUNT:
        errors.append(f"model coverage must contain all {EXPECTED_MODEL_COUNT} current models")
    imported_by_name = {model["name"]: model for model in source_models}
    model_by_name: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(models):
        if not isinstance(row, dict):
            errors.append(f"models[{index}] must be an object")
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"models[{index}].name missing")
            continue
        if name in model_by_name:
            errors.append(f"duplicate model {name}")
        model_by_name[name] = row
        source = imported_by_name.get(name)
        if source is None:
            errors.append(f"model {name} is not in ladder.MODELS")
            continue
        for field, expected in source.items():
            if row.get(field) != expected:
                errors.append(f"model {name}.{field} differs from ladder.MODELS")
        if row.get("source_model_sha256") != _canonical_hash(source):
            errors.append(f"model {name} source hash mismatched")
        if row.get("source_class") != _source_class(source["params_b"]):
            errors.append(f"model {name} source class mismatched")
        if row.get("resource_class") != _resource_class(source["params_b"]):
            errors.append(f"model {name} resource class mismatched")
        if row.get("rate_points") != list(RATE_POINTS):
            errors.append(f"model {name} rate coverage incomplete")
        if row.get("claim_tracks") != list(CLAIM_TRACKS):
            errors.append(f"model {name} claim-track coverage incomplete")
        if row.get("stage_path") != list(STAGE_IDS):
            errors.append(f"model {name} stage path incomplete")
    if set(model_by_name) != set(imported_by_name):
        errors.append("model names do not exactly snapshot ladder.MODELS")

    compiled = document.get("compiled_from")
    if not isinstance(compiled, dict):
        errors.append("compiled_from must be an object")
    else:
        if compiled.get("model_source") != "ladder.MODELS":
            errors.append("compiled_from model source changed")
        if compiled.get("model_count") != EXPECTED_MODEL_COUNT:
            errors.append("compiled_from model count wrong")
        if compiled.get("model_snapshot_sha256") != _canonical_hash(source_models):
            errors.append("compiled_from model snapshot hash mismatched")
        if compiled.get("parameter_count_policy") != _parameter_count_policy():
            errors.append("compiled_from parameter-count policy changed")
        expected_contracts = {
            "doctor_program_schema": doctor_contract.PROGRAM_SCHEMA,
            "doctor_artifact_schema": doctor_contract.ARTIFACT_SCHEMA,
            "doctor_observation_schema": doctor_contract.OBSERVATION_SCHEMA,
            "doctor_dominance_schema": doctor_contract.DOMINANCE_SCHEMA,
        }
        for field, expected in expected_contracts.items():
            if compiled.get(field) != expected:
                errors.append(f"compiled_from.{field} differs from Doctor-v5")

    lanes = document.get("lanes")
    if not isinstance(lanes, list):
        return errors + ["lanes must be a list"]
    expected_keys = {
        (name, rate, track)
        for name in imported_by_name
        for rate in RATE_POINTS
        for track in CLAIM_TRACKS
    }
    observed_keys: set[tuple[str, float, str]] = set()
    expected_lane_count = EXPECTED_MODEL_COUNT * len(RATE_POINTS) * len(CLAIM_TRACKS)
    if len(lanes) != expected_lane_count:
        errors.append(f"research lane count must be {expected_lane_count}")
    for index, lane in enumerate(lanes):
        prefix = f"lanes[{index}]"
        if not isinstance(lane, dict):
            errors.append(f"{prefix} must be an object")
            continue
        name = lane.get("model")
        rate = lane.get("physical_bpw_ceiling")
        track = lane.get("claim_track")
        if name not in model_by_name or rate not in RATE_POINTS or track not in CLAIM_TRACKS:
            errors.append(f"{prefix} has unknown model/rate/track")
            continue
        key = (name, float(rate), track)
        if key in observed_keys:
            errors.append(f"duplicate lane {key}")
        observed_keys.add(key)
        model = model_by_name[name]
        expected_id = f"{name}::{_rate_token(float(rate))}bpw::{track}"
        if lane.get("lane_id") != expected_id:
            errors.append(f"{prefix}.lane_id mismatched")
        if lane.get("model_source_sha256") != model.get("source_model_sha256"):
            errors.append(f"{prefix} model source binding mismatched")
        if lane.get("rate_role") != _rate_role(float(rate)):
            errors.append(f"{prefix} rate role mismatched")
        if lane.get("source_class") != model.get("source_class") \
                or lane.get("resource_class") != model.get("resource_class"):
            errors.append(f"{prefix} source/resource class mismatched")
        nominal = max(1, int(model["params_b"] * 1_000_000_000 * float(rate) / 8.0))
        if lane.get("nominal_weight_payload_bytes") != nominal:
            errors.append(f"{prefix} nominal byte projection mismatched")
        if lane.get("nominal_weight_payload_gb") != round(nominal / 1_000_000_000, 6):
            errors.append(f"{prefix} nominal GB projection mismatched")
        fit = nominal <= int(WEIGHT_BUDGET * 1_000_000_000)
        if lane.get("nominal_resident_weight_fit") is not fit:
            errors.append(f"{prefix} nominal fit projection mismatched")
        expected_disposition = (
            "resident_candidate_subject_to_actual_bytes" if fit else "out_of_core_scientific_control"
        )
        if lane.get("fit_disposition") != expected_disposition:
            errors.append(f"{prefix} fit disposition mismatched")
        if lane.get("required_treatment_branches") != _treatment_branches(float(rate)):
            errors.append(f"{prefix} treatment branch coverage mismatched")
        if lane.get("stage_path") != list(STAGE_IDS):
            errors.append(f"{prefix} stage path incomplete")
        if lane.get("nominal_projection_semantics") \
                != "rounded_params_b_planning_estimate_not_evidence":
            errors.append(f"{prefix} misrepresents rounded parameter planning as evidence")
        if lane.get("exact_parameter_count") is not None \
                or lane.get("parameter_manifest_sha256") is not None \
                or lane.get("exact_physical_bpw_computable") is not False:
            errors.append(f"{prefix} fabricates exact source parameter evidence")
        required_blockers = {
            "exact_source_tensor_parameter_manifest_missing",
            "root_v5_manifest_missing",
            "user_greenlight_not_recorded",
            "executor_not_wired",
        }
        if set(lane.get("admission_blockers", [])) != required_blockers:
            errors.append(f"{prefix} exact-accounting or execution blockers changed")
        if lane.get("status") != "planned":
            errors.append(f"{prefix} must remain planned")
        if lane.get("executor_wired") is not False or lane.get("launch_permitted") is not False:
            errors.append(f"{prefix} violates execution-free contract")
        if lane.get("speed_deferred") is not True:
            errors.append(f"{prefix} does not defer speed")
        expected_sha = lane.get("lane_sha256")
        if not doctor_contract.is_sha256(expected_sha) \
                or expected_sha != _canonical_hash(_identity_payload(lane, "lane_sha256")):
            errors.append(f"{prefix}.lane_sha256 missing or mismatched")
        expected_lane = _lane(model, float(rate), track)
        if lane != expected_lane:
            errors.append(f"{prefix} differs from the canonical compiled lane")
    if observed_keys != expected_keys:
        errors.append("lane matrix is not the exact model x rate x claim-track cross product")
    return errors


def _validate_counts(document: dict[str, Any]) -> list[str]:
    expected_lanes = EXPECTED_MODEL_COUNT * len(RATE_POINTS) * len(CLAIM_TRACKS)
    expected = {
        "models": EXPECTED_MODEL_COUNT,
        "rates": len(RATE_POINTS),
        "claim_tracks": len(CLAIM_TRACKS),
        "stages": len(STAGE_IDS),
        "research_lanes": expected_lanes,
        "referenced_stage_cells": expected_lanes * len(STAGE_IDS),
    }
    return [] if document.get("counts") == expected else ["counts do not match the compiled matrix"]


def validate_ladder(document: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["training ladder must be an object"]
    if document.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if document.get("ladder_version") != LADDER_VERSION:
        errors.append(f"ladder_version must be {LADDER_VERSION}")
    if document.get("doctor_policy_version") != doctor_contract.POLICY_VERSION:
        errors.append("doctor policy version differs from Doctor-v5")
    if document.get("capability_domains") != list(doctor_contract.CAPABILITY_DOMAINS):
        errors.append("capability domains differ from Doctor-v5")
    errors.extend(_validate_execution_policy(document))
    errors.extend(_validate_quality_policy(document))
    if document.get("matched_test_time_compute") != _matched_compute_policy():
        errors.append("matched test-time-compute policy differs from the complete quality-battery contract")
    errors.extend(_validate_tracks(document))
    errors.extend(_validate_classes(document))
    errors.extend(_validate_stages(document))
    errors.extend(_validate_firewall(document))
    errors.extend(_validate_resume(document))
    errors.extend(_validate_controls_and_competitors(document))
    errors.extend(_validate_rates(document))
    errors.extend(_validate_models_and_lanes(document))
    errors.extend(_validate_counts(document))
    expected_sha = document.get("ladder_sha256")
    if not doctor_contract.is_sha256(expected_sha) \
            or expected_sha != _canonical_hash(_identity_payload(document, "ladder_sha256")):
        errors.append("ladder_sha256 missing or mismatched")
    return errors


def _restamp_lane(lane: dict[str, Any]) -> dict[str, Any]:
    return _stamp(lane, "lane_sha256")


def _restamp_ladder(document: dict[str, Any]) -> dict[str, Any]:
    return _stamp(document, "ladder_sha256")


def _assert_invalid(document: dict[str, Any], phrase: str) -> None:
    errors = validate_ladder(document)
    assert errors, "tampered ladder unexpectedly validated"
    assert any(phrase in error for error in errors), (phrase, errors)


def selftest() -> int:
    source_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden_modules = {"subprocess", "multiprocessing", "torch", "mlx", "jax", "tensorflow"}
    imported_roots: set[str] = set()
    forbidden_calls: list[str] = []
    for node in ast.walk(source_tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval"}:
                forbidden_calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute) \
                    and node.func.attr in {"Popen", "spawn", "system", "execv", "execve"}:
                forbidden_calls.append(node.func.attr)
    assert not (imported_roots & forbidden_modules), imported_roots & forbidden_modules
    assert not forbidden_calls, forbidden_calls

    document = compile_ladder()
    assert validate_ladder(document) == []
    second = compile_ladder()
    assert document == second
    assert document["ladder_sha256"] == second["ladder_sha256"]
    assert len(document["models"]) == 32
    assert len(document["stages"]) == 11
    assert len(document["lanes"]) == 1280
    assert document["counts"]["referenced_stage_cells"] == 14_080

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "training_ladder_v5.json"
        doctor_contract.atomic_json(path, document)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert validate_ladder(loaded) == []

    damaged = copy.deepcopy(document)
    damaged["rate_profiles"] = [
        row for row in damaged["rate_profiles"] if row["physical_bpw_ceiling"] != 0.33
    ]
    _assert_invalid(_restamp_ladder(damaged), "rate profile order/coverage")

    damaged = copy.deepcopy(document)
    damaged["stages"][0]["depends_on"] = ["L10"]
    _assert_invalid(_restamp_ladder(damaged), "stage DAG cycle")

    damaged = copy.deepcopy(document)
    damaged["models"][0]["params_b"] = 0.6
    _assert_invalid(_restamp_ladder(damaged), "differs from ladder.MODELS")

    damaged = copy.deepcopy(document)
    damaged["compiled_from"]["parameter_count_policy"][
        "exact_integer_tensor_parameter_count_required_for_evidence"
    ] = False
    _assert_invalid(_restamp_ladder(damaged), "parameter-count policy")

    damaged = copy.deepcopy(document)
    damaged["lanes"][0]["exact_parameter_count"] = 500_000_000
    damaged["lanes"][0]["exact_physical_bpw_computable"] = True
    damaged["lanes"][0] = _restamp_lane(damaged["lanes"][0])
    _assert_invalid(_restamp_ladder(damaged), "fabricates exact source parameter evidence")

    damaged = copy.deepcopy(document)
    damaged["lanes"][0]["launch_permitted"] = True
    damaged["lanes"][0] = _restamp_lane(damaged["lanes"][0])
    _assert_invalid(_restamp_ladder(damaged), "execution-free contract")

    damaged = copy.deepcopy(document)
    damaged["claim_tracks"][0]["stronger_teacher_allowed"] = True
    _assert_invalid(_restamp_ladder(damaged), "codec_fidelity repair/teacher/runtime firewall")

    damaged = copy.deepcopy(document)
    damaged["data_firewall"]["gates"]["sealed_hidden_until_training_frozen"] = False
    _assert_invalid(_restamp_ladder(damaged), "data firewall gates")

    damaged = copy.deepcopy(document)
    damaged["exact_resume_gate"]["required_state"].remove("rng_state_all_backends")
    _assert_invalid(_restamp_ladder(damaged), "exact resume state")

    damaged = copy.deepcopy(document)
    damaged["controls"] = [row for row in damaged["controls"] if row["id"] != "zero_correction"]
    _assert_invalid(_restamp_ladder(damaged), "mandatory control set")

    damaged = copy.deepcopy(document)
    damaged["competitor_requirements"]["fairness_contract"]["same_parent"] = False
    _assert_invalid(_restamp_ladder(damaged), "fairness contract")

    damaged = copy.deepcopy(document)
    damaged["quality_policy"]["speed_deferred"] = False
    _assert_invalid(_restamp_ladder(damaged), "speed_deferred")

    damaged = copy.deepcopy(document)
    damaged["matched_test_time_compute"]["required_fields"]["input"].remove(
        "max_input_tokens"
    )
    _assert_invalid(_restamp_ladder(damaged), "matched test-time-compute policy")

    damaged = copy.deepcopy(document)
    damaged["lanes"].pop()
    _assert_invalid(_restamp_ladder(damaged), "research lane count")

    damaged = copy.deepcopy(document)
    damaged["models"][0]["note"] = "tampered-without-restamp"
    _assert_invalid(damaged, "ladder_sha256")

    print("training_ladder_v5.py selftest OK")
    return 0


def _compile_command() -> int:
    document = compile_ladder()
    doctor_contract.atomic_json(DEFAULT_REPORT, document)
    result = {
        "ok": True,
        "schema": document["schema"],
        "ladder_sha256": document["ladder_sha256"],
        "output": str(DEFAULT_REPORT),
        "counts": document["counts"],
        "execution_free": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _validate_command(path: Path) -> int:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "path": str(path), "errors": [str(error)]}, indent=2))
        return 1
    errors = validate_ladder(document)
    print(json.dumps({"ok": not errors, "path": str(path), "errors": errors}, indent=2))
    return 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("compile", help="atomically write the default execution-free report")
    validate = commands.add_parser("validate", help="validate a compiled ladder")
    validate.add_argument("path", nargs="?", type=Path, default=DEFAULT_REPORT)
    commands.add_parser("selftest", help="run deterministic and tamper-resistance tests")
    arguments = parser.parse_args()
    if arguments.command == "compile":
        return _compile_command()
    if arguments.command == "validate":
        return _validate_command(arguments.path)
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
