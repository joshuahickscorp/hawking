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
            "id": source_id,
            "params_b_range": {"minimum_exclusive": minimum, "maximum_inclusive": maximum},
            "source_mode": mode,
            "required_access": access,
            "whole_parent_residency_assumed": resident,
            "streaming_required": not resident,
        }
        for source_id, minimum, maximum, mode, access, resident in (
            ("resident_parent_source", 0.0, 16.0, "verified_local_or_downloaded_parent",
             "parent tensors, tokenizer, config, revision, chat template", True),
            ("streamed_parent_source", 16.0, 235.0, "verified_sharded_parent",
             "immutable shard map plus byte offsets and hashes", False),
            ("frontier_sharded_parent_source", 235.0, None,
             "remote_or_local_verified_frontier_shards",
             "immutable manifest, transactional shard fetch, global merge state", False),
        )
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
            "id": resource_id,
            "params_b_range": {"minimum_exclusive": minimum, "maximum_inclusive": maximum},
            "execution_shape_if_later_authorized": shape,
            "future_parallel_cap": parallel_cap,
            **common,
        }
        for resource_id, minimum, maximum, shape, parallel_cap in (
            ("resident_single_host_research", 0.0, 16.0,
             "one resident parent/candidate treatment", 3),
            ("streamed_single_host_research", 16.0, 235.0,
             "bounded layer/block/shard windows", 1),
            ("frontier_out_of_core_research", 235.0, None,
             "multi-pass transactional out-of-core shards", 1),
        )
    ]


def _track_definitions() -> list[dict[str, Any]]:
    return [
        {
            "id": track_id,
            "claim": claim,
            "training_teachers": list(teachers),
            "doctor_repair_allowed": repair,
            "stronger_teacher_allowed": stronger,
            "external_runtime_allowed": external,
            "external_runtime_bytes_ceiling": runtime_bytes,
            "l8_action": action,
        }
        for track_id, claim, teachers, repair, stronger, external, runtime_bytes, action in (
            (
                "codec_fidelity",
                "codec/representation fidelity without attached Doctor repair or external inference",
                ("exact_parent_for_reconstruction_only",), False, False, False, 0,
                "prove_zero_doctor_repair_and_zero_external_runtime_dependency",
            ),
            (
                "restorative_training",
                "standalone condensed artifact restores damage against its exact identity teacher",
                ("exact_parent", "truth_oracle"), True, False, False, 0,
                "prove_identity_teacher_only_and_zero_external_runtime",
            ),
            (
                "capability_elevation",
                "standalone condensed artifact gains verified capability from stronger teachers",
                ("exact_parent", "provenance_bound_stronger_teacher", "truth_oracle"),
                True, True, False, 0,
                "prove_training_only_teacher_dependency_and_zero_external_runtime",
            ),
            (
                "augmented_system",
                "fully billed system adds capability through retrieval, tools, or verifiers",
                ("exact_parent", "truth_oracle_for_external_system_outputs"),
                True, False, True, "measured_not_assumed",
                "build_and_bill_retrieval_tool_verifier_plane",
            ),
        )
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


def _validate_static_sections(document: dict[str, Any]) -> list[str]:
    """Compare immutable policy sections with their canonical constructors.

    Each former section validator ended by performing this same exact
    comparison after repeating a partial schema by hand.  The constructors are
    already the authoritative schema, so centralizing the comparisons keeps the
    validator fail-closed while removing the duplicate policy implementation.
    """

    execution_policy = {
        "planner_only": True,
        "launches_processes": False,
        "loads_models": False,
        "reads_live_state": False,
        "touches_live_state": False,
        "only_declared_output": str(DEFAULT_REPORT.relative_to(ROOT)),
    }
    expected = (
        ("execution_policy", execution_policy,
         "execution_policy must be the exact execution-free v5 policy"),
        ("quality_policy", _quality_policy(), "quality policy differs from canonical v5"),
        ("matched_test_time_compute", _matched_compute_policy(),
         "matched test-time-compute policy differs from the complete quality-battery contract"),
        ("claim_tracks", _track_definitions(),
         "claim tracks differ; codec_fidelity repair/teacher/runtime firewall changed"),
        ("source_classes", _source_classes(), "source class definitions differ from canonical v5"),
        ("resource_classes", _resource_classes(),
         "resource class definitions differ from canonical v5"),
        ("rate_profiles", _rate_profiles(),
         "rate profile order/coverage must match the complete v5 ladder"),
        ("stages", _stages(), "stage definitions differ from the canonical L0-L10 ladder"),
        ("data_firewall", _data_firewall(), "data firewall gates are incomplete or weakened"),
        ("exact_resume_gate", _exact_resume(), "exact resume state differs from Doctor-v5"),
        ("controls", _controls(), "mandatory control set is incomplete or duplicated"),
        ("competitor_requirements", _competitors(),
         "competitor fairness contract or direct requirements differ from canonical v5"),
    )
    errors: list[str] = []
    for field, canonical, message in expected:
        actual = document.get(field)
        if actual == canonical:
            continue
        if field == "quality_policy" and isinstance(actual, dict) \
                and actual.get("speed_deferred") is not True:
            message = "quality_policy.speed_deferred must be true"
        elif field == "stages" and isinstance(actual, list):
            try:
                stages = {
                    row["id"]: row
                    for row in actual
                    if isinstance(row, dict) and isinstance(row.get("id"), str)
                }
                cycle = _cycle(stages) if len(stages) == len(actual) else None
            except (KeyError, TypeError, ValueError):
                cycle = None
            if cycle:
                message = "stage DAG cycle: " + " -> ".join(cycle)
        errors.append(message)
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
    errors.extend(_validate_static_sections(document))
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

    def reject(phrase: str, edit, *, lane: bool = False, ladder: bool = True) -> None:
        damaged = copy.deepcopy(document)
        edit(damaged)
        if lane:
            damaged["lanes"][0] = _restamp_lane(damaged["lanes"][0])
        _assert_invalid(_restamp_ladder(damaged) if ladder else damaged, phrase)

    def assign(path, replacement):
        def edit(value) -> None:
            target = value
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = replacement
        return edit

    mutations = (
        ("rate profile order/coverage", lambda value: value.__setitem__(
            "rate_profiles",
            [row for row in value["rate_profiles"] if row["physical_bpw_ceiling"] != 0.33],
        ), False, True),
        ("stage DAG cycle", assign(("stages", 0, "depends_on"), ["L10"]), False, True),
        ("differs from ladder.MODELS", assign(("models", 0, "params_b"), 0.6), False, True),
        ("parameter-count policy", assign((
            "compiled_from", "parameter_count_policy",
            "exact_integer_tensor_parameter_count_required_for_evidence",
        ), False), False, True),
        ("fabricates exact source parameter evidence", lambda value: (
            value["lanes"][0].__setitem__("exact_parameter_count", 500_000_000),
            value["lanes"][0].__setitem__("exact_physical_bpw_computable", True),
        ), True, True),
        ("execution-free contract", assign(("lanes", 0, "launch_permitted"), True), True, True),
        ("codec_fidelity repair/teacher/runtime firewall",
         assign(("claim_tracks", 0, "stronger_teacher_allowed"), True), False, True),
        ("data firewall gates", assign((
            "data_firewall", "gates", "sealed_hidden_until_training_frozen",
        ), False), False, True),
        ("exact resume state", lambda value: value["exact_resume_gate"]["required_state"].remove(
            "rng_state_all_backends"
        ), False, True),
        ("mandatory control set", lambda value: value.__setitem__(
            "controls", [row for row in value["controls"] if row["id"] != "zero_correction"]
        ), False, True),
        ("fairness contract", assign((
            "competitor_requirements", "fairness_contract", "same_parent",
        ), False), False, True),
        ("speed_deferred", assign(("quality_policy", "speed_deferred"), False), False, True),
        ("matched test-time-compute policy",
         lambda value: value["matched_test_time_compute"]["required_fields"]["input"].remove(
             "max_input_tokens"
         ), False, True),
        ("research lane count", lambda value: value["lanes"].pop(), False, True),
        ("ladder_sha256", assign(
            ("models", 0, "note"), "tampered-without-restamp"
        ), False, False),
    )
    for phrase, edit, lane, ladder in mutations:
        reject(phrase, edit, lane=lane, ladder=ladder)

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
