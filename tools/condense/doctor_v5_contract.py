#!/usr/bin/env python3.12
"""Fail-closed contracts for the capability-first Condensation Doctor v5.

This module is deliberately execution-free and stdlib-only.  It defines the
identity, data-firewall, quality, evidence, and statistical contracts used by
``doctor_v5.py`` and ``training_ladder_v5.py``.  It does not import a framework,
load a model, run an evaluator, or launch a subprocess.

The core distinction in v5 is claim scope:

* ``codec_fidelity``: representation/codec optimization only, with no attached
  Doctor repair or external inference mechanism.
* ``restorative_training``: a standalone condensed artifact is treated against
  its exact identity teacher; gains are restoration only up to parent parity.
* ``capability_elevation``: a standalone artifact may learn from a stronger
  teacher, but beyond-parent gains are enhancement, never restored information.
* ``augmented_system``: retrieval/tools/verifiers are allowed, fully billed,
  and never allowed to support a core-model quality claim.

All hashes are SHA-256 over actual bytes or canonical JSON.  Planned values use
the literal ``"required"``; executable/result documents require real hashes.
Self-stamps prove consistency, not authority: executable programs and proven
observations additionally require exact receipt identities in a caller-owned,
role-separated trust context.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


POLICY_VERSION = "doctor-v5.0"
PACKAGE_ROOT_SCHEMA = "hawking.doctor_v5_root.v5"
PROGRAM_SCHEMA = "hawking.doctor_v5_program.v5"
PROGRAM_SPEC_SCHEMA = "hawking.doctor_v5_program_spec.v5"
CAMPAIGN_METADATA_SCHEMA = "hawking.doctor_v5_campaign_metadata.v5"
ARTIFACT_SCHEMA = "hawking.doctor_v5_artifact.v5"
OBSERVATION_SCHEMA = "hawking.doctor_v5_observation.v5"
DOMINANCE_SCHEMA = "hawking.doctor_v5_dominance.v5"

CLAIM_SCOPES = (
    "codec_fidelity",
    "restorative_training",
    "capability_elevation",
    "augmented_system",
)
DIAGNOSES = (
    "undetermined",
    "signal_degradation",
    "computation_collapse",
    "mixed_failure",
    "no_material_damage",
)
DIAGNOSTIC_PROBES = {
    "early_signal_survival", "internal_geometry", "capability_failures",
    "hidden_logit_divergence", "adversarial_quantization_trigger_scan",
}
PROGRAM_MODES = ("planned", "executable")
IMPLEMENTATION_STATES = ("control", "measured", "prototype", "research", "unimplemented")
OPERATOR_KINDS = (
    "diagnose",
    "transform",
    "represent",
    "reconstruct",
    "repair_static",
    "repair_conditional",
    "train",
    "harden",
    "augment_external",
    "package",
    "evaluate",
)
PROOF_STATES = (
    "planned",
    "feasibility",
    "tensor_oracle",
    "block_oracle",
    "shard_oracle",
    "full_model_quality",
    "replicated_quality",
    "sealed_final",
    "independent_reproduction",
)
EVIDENCE_STATES = (
    "PLANNED",
    "RUNNING",
    "PROVISIONAL",
    "PROVEN",
    "INVALID",
    "UNREPRODUCED",
    "REVOKED",
)
CAPABILITY_DOMAINS = (
    "language_modeling",
    "knowledge",
    "reasoning",
    "mathematics",
    "science",
    "coding",
    "instruction_following",
    "long_context",
    "multilingual",
    "tool_use",
    "calibration",
    "safety_security",
)
EXTERNAL_KINDS = {"augment_external"}

ARTIFACT_COMPONENTS = (
    "base",
    "pass_through",
    "scales",
    "codebooks",
    "indices",
    "corrections",
    "routers",
    "state",
    "metadata",
    "alignment",
    "tokenizer",
    "retrieval_index",
    "auxiliary_models",
    "persistent_external_state",
    "decoder_runtime",
    "runtime_dependencies",
    "context_state",
)

TRAINING_OPERATOR_KINDS = {
    "reconstruct", "repair_static", "repair_conditional", "train", "harden",
}
MUTATING_OPERATOR_KINDS = {
    "transform", "represent", "reconstruct", "repair_static",
    "repair_conditional", "train", "harden", "augment_external",
}
TREATMENT_ROLES = (
    "diagnostic_only",
    "representation_search",
    "structural_reconstruction",
    "signal_compensation",
    "zero_treatment",
    "external_augmentation",
    "packaging",
    "evaluation",
)
MATCHED_COMPUTE_FIELDS = {
    "max_input_tokens",
    "max_output_tokens",
    "max_reasoning_tokens",
    "samples_per_item",
    "temperature",
    "timeout_ms",
    "verifier_retries",
    "retrieval_calls",
    "tool_calls",
    "external_model_calls",
}
OBSERVATION_STATUSES = (
    "planned",
    "running",
    "succeeded",
    "complete_negative",
    "failed_retryable",
    "failed_terminal",
    "invalidated",
)
TRUST_CONTEXT_ROLES = (
    "package_root_manifest",
    "root_manifest",
    "greenlight_receipt",
    "adapter_allowlist",
    "resource_admission_receipt",
    "parameter_manifest_receipt",
    "parent_teacher_authorization",
    "elevation_teacher_authorization",
    "artifact_validation_receipt",
    "evidence_verifier_receipt",
    "sealed_service_attestation",
    "independent_owner_attestation",
)

MODEL_FIELDS = {
    "id", "params_b", "exact_parameter_count", "active_b",
    "parent_revision_sha256", "config_sha256", "tokenizer_sha256",
    "chat_template_sha256", "parameter_count_binding_sha256",
    "exact_parameter_count_source_sha256", "parameter_manifest_receipt",
}
TRAINING_FIELDS = {
    "parent_teacher_required", "teacher_outputs_split_bound",
    "early_stop_on_frozen_selection_only", "retain_zero_treatment",
    "gradient_conflict_measurement", "bf16_same_treatment_control_required",
    "elevation_teacher_allowed", "external_teacher_at_inference",
    "minimum_independent_training_seeds", "minimum_calibration_draws",
    "training_seeds", "calibration_draw_seeds", "training_seed_manifest_sha256",
    "calibration_draw_manifest_sha256", "teacher_contract",
}
TEACHER_CONTRACT_FIELDS = {
    "parent_teacher", "elevation_teacher", "no_teacher_at_inference",
}
TEACHER_FIELDS = {
    "identity_sha256", "revision_sha256", "role", "output_protocol_sha256",
    "cache_manifest_sha256", "split_manifest_sha256", "provenance_manifest_sha256",
    "training_only", "authorization_receipt",
}
EXECUTION_FIELDS = {
    "state", "root_manifest", "greenlight_receipt", "adapter_allowlist",
    "resource_admission", "campaign_owner_id",
}
PROGRAM_FIELDS = {
    "schema", "policy_version", "program_spec_schema", "program_spec_sha256",
    "package_root_schema", "package_root_manifest_sha256", "mode",
    "experiment_binding", "claim_scope", "model", "target", "diagnostic_contract",
    "operators", "data_contract", "training_contract", "evaluation_contract",
    "execution_contract", "exact_resume_contract", "output_contract",
    "campaign_metadata", "program_sha256",
}
EXPERIMENT_BINDING_FIELDS = {"experiment_id", "candidate_identity_sha256"}
TARGET_FIELDS = {
    "physical_bpw_ceiling", "physical_artifact_bytes_ceiling", "resident_bytes_ceiling",
    "speed_deferred", "bpw_is_all_in_physical", "physical_byte_ceiling_basis",
    "unresolved_exact_parameter_count_blocks_evidence_and_launch",
}
DIAGNOSTIC_FIELDS = {
    "failure_class", "route", "required_probes", "collapse_forbids_compensation_only",
    "represent_alone_never_proves_reconstruction", "probe_receipt_sha256",
}
PLANNED_OPERATOR_FIELDS = {
    "id", "kind", "mechanism", "source", "implementation_status", "depends_on", "executor",
    "treatment_role",
}
EXECUTABLE_OPERATOR_FIELDS = PLANNED_OPERATOR_FIELDS | {"treatment_role"}
EXECUTOR_FIELDS = {"wired", "adapter_id", "source_sha256"}
DATA_CONTRACT_FIELDS = {
    "splits", "exact_duplicate_scan_required", "near_duplicate_scan_required",
    "semantic_contamination_scan_required", "sealed_final_reveal_after_training",
    "teacher_output_cache_split_bound",
}
DATA_SPLIT_FIELDS = {"manifest_sha256", "examples", "tokens", "accessible_to_optimizer"}
EVALUATION_FIELDS = {
    "capability_domains", "required_competitor_ids", "required_competitor_ids_sha256",
    "per_item_outputs_required", "blind_candidate_labels", "paired_parent_prompts",
    "judge_free_where_possible", "test_time_compute_matched", "test_time_compute_budget",
    "suite_manifest_sha256", "prompt_protocol_sha256", "scorer_source_sha256",
    "competitor_registry_sha256", "test_time_compute_protocol_sha256",
}
EXACT_RESUME_FIELDS = {
    "required_state", "atomic_replace", "fsync_file", "fsync_parent_directory", "validate_before_resume",
}
OUTPUT_FIELDS = {
    "actual_file_bytes_authoritative", "all_tensor_ownership_required",
    "dense_parent_fallback_forbidden", "billed_components",
}
CAMPAIGN_METADATA_FIELDS = {
    "schema", "campaign_sha256", "campaign_input_binding_sha256",
    "canonical_policy_bundle_sha256", "direct_competitor_registry_sha256",
    "parameter_count_binding_sha256", "teacher_authority_contract_sha256",
    "test_time_compute_budget_sha256", "research_contracts_sha256", "explicit_ordinal",
    "evidence_stage", "evidence_state", "launch_permitted", "greenlight_recorded",
    "blockers", "metadata_sha256",
}
ROOT_RECEIPT_FIELDS = {
    "manifest_sha256", "package_root_schema", "program_spec_sha256",
    "parameter_manifest_receipt_sha256", "teacher_authorization_receipts_sha256",
    "experiment_id", "candidate_identity_sha256", "policy_version",
    "parent_revision_sha256", "binding_sha256",
}
GREENLIGHT_RECEIPT_FIELDS = {
    "granted", "root_manifest_sha256", "root_binding_sha256", "program_spec_sha256",
    "parameter_manifest_receipt_sha256", "teacher_authorization_receipts_sha256",
    "candidate_identity_sha256", "claim_scope", "issued_by", "issued_at",
    "signer_key_sha256", "signature_sha256", "receipt_sha256",
}
ALLOWLIST_FIELDS = {"program_spec_sha256", "operators_adapters_sha256", "entries", "allowlist_sha256"}
ALLOWLIST_ENTRY_FIELDS = {"adapter_id", "source_sha256"}
RESOURCE_ADMISSION_FIELDS = {
    "admitted", "root_manifest_sha256", "program_spec_sha256",
    "parameter_manifest_receipt_sha256", "exact_parameter_count", "target_ceilings_sha256",
    "compute_budget_sha256", "operators_adapters_sha256", "host_identity_sha256",
    "verifier_source_sha256", "observed_at", "available_memory_bytes", "available_disk_bytes",
    "resident_bytes_limit", "swap_bytes", "requested_peak_resident_bytes",
    "requested_scratch_disk_bytes", "memory_pressure", "thermal_state", "ac_power", "receipt_sha256",
}
PARAMETER_RECEIPT_FIELDS = {
    "program_spec_sha256", "exact_parameter_count",
    "tensor_ownership_manifest_sha256", "tensor_classification_aggregate_sha256",
    "parent_revision_sha256", "config_sha256", "source_shard_manifest_sha256",
    "source_file_manifest_sha256", "counting_code_sha256", "all_tensors_classified",
    "receipt_sha256",
}
TEACHER_AUTH_FIELDS = {
    "authorized", "program_spec_sha256", "claim_scope", "teacher_identity_sha256",
    "teacher_revision_sha256", "teacher_role", "output_protocol_sha256",
    "cache_manifest_sha256", "split_manifest_sha256", "provenance_manifest_sha256",
    "training_only", "receipt_sha256",
}
ARTIFACT_FIELDS = {
    "schema", "program_sha256", "claim_scope", "parent_revision_sha256",
    "packed_semantics_sha256", "decoder_semantics_sha256", "tensor_ownership_sha256",
    "runtime_abi_sha256", "exact_parameter_count", "dense_parent_fallback",
    "all_tensor_ownership_complete", "files", "byte_ledger", "component_manifests",
    "physical_accounting", "all_in_physical_bytes", "all_in_bpw", "payload_bpw",
    "decoded_resident_bytes", "resident_peak_bytes", "expected_active_bytes",
    "worst_active_bytes", "peak_bytes_by_context", "runtime_accounting", "artifact_sha256",
}
ARTIFACT_FILE_FIELDS = {"path", "component", "sha256", "bytes"}
COMPONENT_MANIFEST_FIELDS = {"bytes", "file_count", "files_sha256"}
PHYSICAL_ACCOUNTING_FIELDS = {
    "stored_artifact_bytes", "payload_bytes", "decoder_runtime_bytes",
    "runtime_dependency_bytes", "metadata_bytes", "context_state_bytes", "external_system_bytes",
}
RUNTIME_ACCOUNTING_FIELDS = {
    "decoder_resident_bytes", "runtime_dependency_resident_bytes", "metadata_resident_bytes",
    "persistent_state_resident_bytes", "context_state_bytes_by_context",
}
MOE_ACCOUNTING_FIELDS = {
    "expert_count", "routed_experts_per_token_expected", "routed_experts_per_token_max",
    "total_parameter_count", "expected_active_parameter_count", "worst_active_parameter_count",
    "total_installed_expert_bytes", "expected_active_expert_bytes", "worst_active_expert_bytes",
}
OBSERVATION_FIELDS = {
    "schema", "program_sha256", "claim_scope", "status", "proof_state", "evidence_state",
    "negative_result_retained", "speed_claimed", "artifact_manifest_sha256",
    "physical_artifact_bytes", "capability_metrics", "statistical_contract",
    "competitor_comparisons", "data_firewall_pass", "parent_parity_protocol_pass",
    "test_time_compute_match_pass", "comparator_set_frozen_before_final",
    "sealed_final_consumed_once", "test_time_compute_receipt", "claim_snapshot",
    "external_runtime_bytes", "evidence_bundle", "observation_sha256",
}
METRIC_FIELDS = {
    "candidate", "parent", "delta", "delta_lcb", "delta_ucb", "noninferiority_margin",
    "n", "raw_p_value", "holm_adjusted_p_value", "ci_method", "evidence_slice_sha256",
    "hypothesis_id", "hypothesis_direction", "null_boundary", "noninferiority_pass",
}
STATISTICAL_FIELDS = {
    "paired", "cluster_resampling", "multiple_testing_correction", "familywise_alpha",
    "confidence", "preregistered", "preregistration_sha256",
}
COMPARISON_FIELDS = {
    "competitor_id", "same_parent", "same_or_lower_physical_bytes", "same_prompt_and_scorer",
    "competitor_artifact_sha256", "evidence_slice_sha256", "competitor_domain_scores",
    "domain_deltas", "domain_delta_lcbs", "domain_raw_p_values",
    "domain_holm_adjusted_p_values", "domain_hypothesis_ids", "domain_superiority_passes",
    "superiority_direction", "macro_delta", "macro_delta_lcb", "macro_raw_p_value",
    "macro_holm_adjusted_p_value", "macro_hypothesis_id", "macro_superiority_pass",
    "superiority_delta",
}
COMPUTE_RECEIPT_FIELDS = {
    "passed", "budget_sha256", "participant_manifest_sha256", "measurement_log_sha256", "receipt_sha256",
}
CLAIM_SNAPSHOT_FIELDS = {
    "as_of_date", "competitor_registry_sha256", "quality_battery_sha256", "artifact_sha256",
    "expires_on_registry_benchmark_artifact_or_runtime_change",
}
EVIDENCE_BUNDLE_FIELDS = {
    "artifact_validation_receipt", "per_item_outputs_sha256", "cluster_assignments_sha256",
    "cluster_outputs_sha256", "data_firewall_receipt_sha256", "parent_parity_receipt_sha256",
    "calibration_draw_results_sha256", "training_seed_results_sha256", "training_seed_count",
    "calibration_draw_count", "raw_evidence_index_sha256", "corrected_test_receipt",
    "evidence_verifier_receipt", "sealed_service_attestation", "independent_owner_attestation",
}
ARTIFACT_VALIDATION_RECEIPT_FIELDS = {
    "program_sha256", "artifact_sha256", "exact_parameter_count", "all_in_physical_bytes",
    "all_in_bpw", "aggregate_file_manifest_sha256", "aggregate_component_manifest_sha256",
    "validator_source_sha256", "verify_files", "receipt_sha256",
}
CORRECTED_TEST_RECEIPT_FIELDS = {
    "procedure", "familywise_alpha", "family_size", "raw_p_values_sha256",
    "corrected_p_values_sha256", "summary_sha256", "all_hypotheses_corrected", "receipt_sha256",
}
EVIDENCE_VERIFIER_FIELDS = {
    "raw_evidence_index_sha256", "summary_sha256", "corrected_test_receipt_sha256",
    "verifier_source_sha256", "owner_id", "recomputed_deltas_cis_pvalues_from_raw",
    "passed", "receipt_sha256",
}
SEALED_ATTESTATION_FIELDS = {
    "service_id", "owner_id", "sealed_final_manifest_sha256", "program_sha256",
    "artifact_sha256", "raw_evidence_index_sha256", "execution_receipt_sha256",
    "one_time_nonce_sha256", "consumed_once", "attestation_sha256",
}
INDEPENDENT_ATTESTATION_FIELDS = {
    "owner_id", "program_sha256", "artifact_sha256", "raw_evidence_index_sha256",
    "summary_sha256", "sealed_attestation_sha256", "replication_receipt_sha256",
    "independently_executed", "no_shared_runtime_owner", "attestation_sha256",
}
DOMINANCE_FIELDS = {
    "schema", "program_sha256", "observation_sha256", "claim_scope", "passed",
    "frontier_champion", "uniform_quality_dominance", "parent_noninferior_all_domains",
    "same_budget_competitors_dominated", "independent_reproduction", "parent_failure_domains",
    "competitor_failures", "uniform_quality_failures", "validation_errors", "claim", "dominance_sha256",
}

EXACT_RESUME_STATE = (
    "program_identity",
    "operator_cursor",
    "operator_state",
    "optimizer_state",
    "scheduler_state",
    "gradient_accumulation_phase",
    "microstep",
    "rng_state_all_backends",
    "sampler_cursor",
    "curriculum_state",
    "source_shard_offset",
    "teacher_cache_identity",
    "failure_replay_identity",
    "partial_output_hashes",
    "best_checkpoint_identity",
    "resume_command_identity",
)

DATA_SPLITS = (
    "calibration",
    "reconstruction_train",
    "repair_train",
    "treatment_search",
    "selection",
    "public_validation",
    "shadow",
    "frozen_final",
    "sealed_final",
    "independent_replication",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


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


def _identity_payload(document: dict[str, Any], identity_field: str) -> dict[str, Any]:
    payload = copy.deepcopy(document)
    payload.pop(identity_field, None)
    payload.pop("generated_at", None)
    return payload


def stamp(document: dict[str, Any], identity_field: str) -> dict[str, Any]:
    out = copy.deepcopy(document)
    out[identity_field] = hash_value(_identity_payload(out, identity_field))
    return out


def _required_hash(value: Any, mode: str) -> bool:
    if mode == "planned":
        return value == "required" or is_sha256(value)
    return is_sha256(value)


def _same_number(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    return _finite(left) and _finite(right) and math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
    )


def _valid_probability(value: Any) -> bool:
    return _finite(value, nonnegative=True) and float(value) <= 1.0


def _stamped(document: Any, identity_field: str) -> bool:
    return isinstance(document, dict) and is_sha256(document.get(identity_field)) \
        and document[identity_field] == hash_value(_identity_payload(document, identity_field))


def _exact_keys(value: Any, expected: set[str], path: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{path} must be an object"]
    actual = set(value)
    if actual == expected:
        return []
    return [
        f"{path} field set mismatch: missing={sorted(expected - actual)}, "
        f"unknown={sorted(actual - expected)}"
    ]


def _is_trusted(trust_context: Any, role: str, identity_sha256: Any) -> bool:
    """Check an identity against an external, role-separated caller trust root.

    Accepted forms are ``{role: {hashes...}}`` (lists/tuples also work, which
    keeps JSON-loaded contexts convenient) or a set of ``(role, hash)`` pairs.
    A flat set of hashes is intentionally rejected to prevent receipt-type
    confusion.  The contract never manufactures trust from a self-stamp.
    """
    if role not in TRUST_CONTEXT_ROLES or not is_sha256(identity_sha256):
        return False
    if isinstance(trust_context, dict):
        identities = trust_context.get(role)
        return isinstance(identities, (set, frozenset, list, tuple)) \
            and identity_sha256 in identities
    if isinstance(trust_context, (set, frozenset)):
        return (role, identity_sha256) in trust_context
    return False


def _program_spec_payload(program: dict[str, Any]) -> dict[str, Any]:
    """Return all semantic program content with only circular receipt wrappers removed."""
    payload = copy.deepcopy(program)
    payload.pop("program_sha256", None)
    payload.pop("program_spec_sha256", None)
    model = payload.get("model")
    if isinstance(model, dict):
        model.pop("parameter_manifest_receipt", None)
    training = payload.get("training_contract")
    if isinstance(training, dict):
        teachers = training.get("teacher_contract")
        if isinstance(teachers, dict):
            for field in ("parent_teacher", "elevation_teacher"):
                teacher = teachers.get(field)
                if isinstance(teacher, dict):
                    teacher.pop("authorization_receipt", None)
    execution = payload.get("execution_contract")
    if isinstance(execution, dict):
        for field in ("root_manifest", "greenlight_receipt", "adapter_allowlist", "resource_admission"):
            execution.pop(field, None)
    return payload


def compute_program_spec_sha256(program: dict[str, Any]) -> str:
    """Identity of final semantic executable content, independent of receipt wrappers."""
    return hash_value(_program_spec_payload(program))


def _operators_adapters_sha256(program: dict[str, Any]) -> str:
    return hash_value(program.get("operators"))


def _target_ceilings_sha256(program: dict[str, Any]) -> str:
    target = program.get("target", {})
    return hash_value({
        field: target.get(field)
        for field in ("physical_bpw_ceiling", "physical_artifact_bytes_ceiling", "resident_bytes_ceiling")
    })


def _teacher_receipt_set_sha256(parent: Any, elevation: Any) -> str:
    return hash_value({
        "parent_teacher_authorization": (
            parent.get("receipt_sha256") if isinstance(parent, dict) else parent
        ),
        "elevation_teacher_authorization": (
            elevation.get("receipt_sha256") if isinstance(elevation, dict) else elevation
        ),
    })


def _validate_teacher(
    teacher: Any,
    *,
    mode: str,
    role: str,
    claim_scope: str,
    program_spec_sha256: Any,
    expected_revision_sha256: Any | None,
    expected_split_manifest_sha256: Any,
    trust_context: Any,
    trust_role: str,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    key_errors = _exact_keys(teacher, TEACHER_FIELDS, f"teacher_contract.{role}")
    if not isinstance(teacher, dict):
        return key_errors, {}
    errors.extend(key_errors)
    if teacher.get("role") != role or teacher.get("training_only") is not True:
        errors.append(f"{role} teacher role/training-only boundary is invalid")
    if expected_revision_sha256 is not None and teacher.get("revision_sha256") != expected_revision_sha256:
        errors.append(f"{role} teacher revision differs from the required parent revision")
    for field in (
        "identity_sha256", "revision_sha256", "output_protocol_sha256", "cache_manifest_sha256",
        "split_manifest_sha256", "provenance_manifest_sha256",
    ):
        if not _required_hash(teacher.get(field), mode):
            errors.append(f"{role} teacher {field} invalid")
    if mode == "executable" and teacher.get("split_manifest_sha256") != expected_split_manifest_sha256:
        errors.append(f"{role} teacher is not bound to the executable data-split manifest")
    receipt = teacher.get("authorization_receipt")
    if mode == "planned" and receipt == "required":
        return errors, {}
    errors.extend(_exact_keys(
        receipt, TEACHER_AUTH_FIELDS, f"teacher_contract.{role}.authorization_receipt"
    ))
    if not _stamped(receipt, "receipt_sha256"):
        errors.append(f"{role} teacher authorization receipt missing or self-inconsistent")
        return errors, {}
    if set(receipt) != TEACHER_AUTH_FIELDS \
            or receipt.get("authorized") is not True \
            or receipt.get("program_spec_sha256") != program_spec_sha256 \
            or receipt.get("claim_scope") != claim_scope \
            or receipt.get("teacher_identity_sha256") != teacher.get("identity_sha256") \
            or receipt.get("teacher_revision_sha256") != teacher.get("revision_sha256") \
            or receipt.get("teacher_role") != role \
            or receipt.get("output_protocol_sha256") != teacher.get("output_protocol_sha256") \
            or receipt.get("cache_manifest_sha256") != teacher.get("cache_manifest_sha256") \
            or receipt.get("split_manifest_sha256") != teacher.get("split_manifest_sha256") \
            or receipt.get("provenance_manifest_sha256") != teacher.get("provenance_manifest_sha256") \
            or receipt.get("training_only") is not True:
        errors.append(f"{role} teacher authorization is not bound to exact identity/output/cache/split/provenance")
    if mode == "executable" and not _is_trusted(
        trust_context, trust_role, receipt.get("receipt_sha256")
    ):
        errors.append(f"external trust context does not authorize {role} teacher")
    return errors, receipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _component_file_payload(files: Iterable[dict[str, Any]], component: str) -> list[dict[str, Any]]:
    return sorted(
        (
            {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
            for row in files
            if row.get("component") == component
            and isinstance(row.get("path"), str)
            and isinstance(row.get("bytes"), int)
            and is_sha256(row.get("sha256"))
        ),
        key=lambda row: row["path"],
    )


def _summary_payload(observation: dict[str, Any]) -> dict[str, Any]:
    """The deterministic summaries that the raw-evidence verifier must bind."""
    return {
        "artifact_manifest_sha256": observation.get("artifact_manifest_sha256"),
        "physical_artifact_bytes": observation.get("physical_artifact_bytes"),
        "capability_metrics": observation.get("capability_metrics"),
        "competitor_comparisons": observation.get("competitor_comparisons"),
        "statistical_contract": observation.get("statistical_contract"),
    }


def _cycle(nodes: dict[str, dict[str, Any]]) -> list[str] | None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str, trail: list[str]) -> list[str] | None:
        if node_id in visiting:
            start = trail.index(node_id) if node_id in trail else 0
            return trail[start:] + [node_id]
        if node_id in visited:
            return None
        visiting.add(node_id)
        for dependency in nodes[node_id].get("depends_on", []):
            if dependency in nodes:
                found = visit(dependency, trail + [node_id])
                if found:
                    return found
        visiting.remove(node_id)
        visited.add(node_id)
        return None

    for node_id in nodes:
        found = visit(node_id, [])
        if found:
            return found
    return None


def validate_data_contract(contract: Any, mode: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(contract, dict):
        return ["data_contract must be an object"]
    errors.extend(_exact_keys(contract, DATA_CONTRACT_FIELDS, "data_contract"))
    splits = contract.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(DATA_SPLITS):
        errors.append(f"data_contract.splits must name exactly {list(DATA_SPLITS)}")
        splits = {}
    observed_hashes: list[str] = []
    for split in DATA_SPLITS:
        row = splits.get(split)
        if not isinstance(row, dict):
            errors.append(f"data split {split} must be an object")
            continue
        errors.extend(_exact_keys(row, DATA_SPLIT_FIELDS, f"data_contract.splits.{split}"))
        digest = row.get("manifest_sha256")
        if not _required_hash(digest, mode):
            errors.append(f"data split {split} manifest hash invalid")
        if is_sha256(digest):
            observed_hashes.append(digest)
        if not isinstance(row.get("examples"), int) or row["examples"] < 0:
            errors.append(f"data split {split} examples must be a nonnegative integer")
        if not isinstance(row.get("tokens"), int) or row["tokens"] < 0:
            errors.append(f"data split {split} tokens must be a nonnegative integer")
        if row.get("accessible_to_optimizer") is not (split not in {
            "shadow", "frozen_final", "sealed_final", "independent_replication"
        }):
            errors.append(f"data split {split} optimizer-access policy is wrong")
    if len(observed_hashes) != len(set(observed_hashes)):
        errors.append("every concrete data split manifest must be hash-distinct")
    if contract.get("exact_duplicate_scan_required") is not True:
        errors.append("exact duplicate scan is required")
    if contract.get("near_duplicate_scan_required") is not True:
        errors.append("near-duplicate scan is required")
    if contract.get("semantic_contamination_scan_required") is not True:
        errors.append("semantic contamination scan is required")
    if contract.get("sealed_final_reveal_after_training") is not True:
        errors.append("sealed final set must be revealed only after training is frozen")
    if contract.get("teacher_output_cache_split_bound") is not True:
        errors.append("teacher cache must be split-bound")
    return errors


def validate_program(
    program: Any,
    *,
    allow_planned: bool = True,
    trust_context: Any = None,
    expected_package_root_manifest_sha256: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(program, dict):
        return ["program must be an object"]
    errors.extend(_exact_keys(program, PROGRAM_FIELDS, "program"))
    if program.get("schema") != PROGRAM_SCHEMA:
        errors.append(f"schema must be {PROGRAM_SCHEMA}")
    if program.get("policy_version") != POLICY_VERSION:
        errors.append(f"policy_version must be {POLICY_VERSION}")
    mode = program.get("mode")
    if mode not in PROGRAM_MODES:
        errors.append("mode must be planned or executable")
        mode = "planned"
    if not allow_planned and mode != "executable":
        errors.append("planned program is not executable")
    package_root_schema = program.get("package_root_schema")
    package_root_sha256 = program.get("package_root_manifest_sha256")
    if mode == "planned" and package_root_schema == "required" and package_root_sha256 == "required":
        pass
    elif package_root_schema != PACKAGE_ROOT_SCHEMA or not is_sha256(package_root_sha256):
        errors.append("package root schema/hash binding is invalid")
    else:
        expected_matches = expected_package_root_manifest_sha256 == package_root_sha256 \
            if expected_package_root_manifest_sha256 is not None else False
        if expected_package_root_manifest_sha256 is not None \
                and not is_sha256(expected_package_root_manifest_sha256):
            errors.append("caller expected package root must be a SHA-256 identity")
        if not expected_matches and not _is_trusted(
            trust_context, "package_root_manifest", package_root_sha256
        ):
            errors.append("caller expected/trusted package root does not match program")
    if program.get("program_spec_schema") != PROGRAM_SPEC_SCHEMA:
        errors.append(f"program_spec_schema must be {PROGRAM_SPEC_SCHEMA}")
    declared_spec = program.get("program_spec_sha256")
    computed_spec = compute_program_spec_sha256(program)
    if mode == "planned" and declared_spec == "required":
        pass
    elif not is_sha256(declared_spec) or declared_spec != computed_spec:
        errors.append("program_spec_sha256 missing or mismatched semantic executable content")
    scope = program.get("claim_scope")
    if scope not in CLAIM_SCOPES:
        errors.append("claim_scope invalid")

    binding = program.get("experiment_binding")
    errors.extend(_exact_keys(binding, EXPERIMENT_BINDING_FIELDS, "experiment_binding"))
    if not isinstance(binding, dict) or not isinstance(binding.get("experiment_id"), str) \
            or not is_sha256(binding.get("candidate_identity_sha256")):
        errors.append("experiment binding is incomplete")
        binding = binding if isinstance(binding, dict) else {}

    parameter_receipt: Any = {}
    exact_parameters: Any = None
    model = program.get("model")
    if not isinstance(model, dict):
        errors.append("model binding must be an object")
        model = {}
    else:
        errors.extend(_exact_keys(model, MODEL_FIELDS, "model"))
        if not isinstance(model.get("id"), str) or not model["id"]:
            errors.append("model.id missing")
        if not _finite(model.get("params_b"), positive=True):
            errors.append("model.params_b must be positive finite")
        exact_parameters = model.get("exact_parameter_count")
        if mode == "planned":
            if exact_parameters != "required" and (
                isinstance(exact_parameters, bool)
                or not isinstance(exact_parameters, int)
                or exact_parameters <= 0
            ):
                errors.append("model.exact_parameter_count must be required or a positive integer")
        elif isinstance(exact_parameters, bool) or not isinstance(exact_parameters, int) \
                or exact_parameters <= 0:
            errors.append("executable model requires an exact positive integer parameter count")
        active = model.get("active_b")
        if active is not None and not _finite(active, positive=True):
            errors.append("model.active_b must be null or positive finite")
        if _finite(active, positive=True) and _finite(model.get("params_b"), positive=True) \
                and float(active) > float(model["params_b"]):
            errors.append("model.active_b cannot exceed params_b")
        for field in ("parent_revision_sha256", "config_sha256", "tokenizer_sha256", "chat_template_sha256"):
            if not _required_hash(model.get(field), mode):
                errors.append(f"model.{field} invalid for {mode} mode")
        for field in ("parameter_count_binding_sha256", "exact_parameter_count_source_sha256"):
            if not _required_hash(model.get(field), mode):
                errors.append(f"model.{field} invalid for {mode} mode")
        parameter_receipt = model.get("parameter_manifest_receipt")
        if mode == "planned" and parameter_receipt == "required":
            pass
        elif not _stamped(parameter_receipt, "receipt_sha256"):
            errors.append("parameter-manifest receipt is missing or self-inconsistent")
            parameter_receipt = {}
        elif set(parameter_receipt) != PARAMETER_RECEIPT_FIELDS \
                or parameter_receipt.get("program_spec_sha256") != declared_spec \
                or parameter_receipt.get("exact_parameter_count") != exact_parameters \
                or parameter_receipt.get("parent_revision_sha256") != model.get("parent_revision_sha256") \
                or parameter_receipt.get("config_sha256") != model.get("config_sha256") \
                or parameter_receipt.get("all_tensors_classified") is not True \
                or any(not is_sha256(parameter_receipt.get(field)) for field in (
                    "tensor_ownership_manifest_sha256", "tensor_classification_aggregate_sha256",
                    "source_shard_manifest_sha256", "source_file_manifest_sha256", "counting_code_sha256",
                )):
            errors.append("parameter-manifest receipt is not exactly bound to source, count, and tensor classification")
        if isinstance(parameter_receipt, dict):
            errors.extend(_exact_keys(
                parameter_receipt, PARAMETER_RECEIPT_FIELDS, "model.parameter_manifest_receipt"
            ))
        if mode == "executable" and not _is_trusted(
            trust_context, "parameter_manifest_receipt",
            parameter_receipt.get("receipt_sha256") if isinstance(parameter_receipt, dict) else None,
        ):
            errors.append("external trust context does not authorize parameter-manifest receipt")

    target = program.get("target")
    errors.extend(_exact_keys(target, TARGET_FIELDS, "target"))
    if not isinstance(target, dict):
        errors.append("target must be an object")
        target = {}
    else:
        for field in ("physical_bpw_ceiling", "physical_artifact_bytes_ceiling", "resident_bytes_ceiling"):
            if not _finite(target.get(field), positive=True):
                errors.append(f"target.{field} must be positive finite")
        if target.get("speed_deferred") is not True:
            errors.append("v5 quality program must mark speed_deferred=true")

    diagnostic = program.get("diagnostic_contract")
    errors.extend(_exact_keys(diagnostic, DIAGNOSTIC_FIELDS, "diagnostic_contract"))
    if not isinstance(diagnostic, dict):
        errors.append("diagnostic_contract missing")
    else:
        diagnosis = diagnostic.get("failure_class")
        if diagnosis not in DIAGNOSES:
            errors.append("diagnostic failure_class invalid")
        required = set(diagnostic.get("required_probes", []))
        if required != DIAGNOSTIC_PROBES or not isinstance(diagnostic.get("required_probes"), list) \
                or len(diagnostic["required_probes"]) != len(DIAGNOSTIC_PROBES):
            errors.append("diagnostic probes must exactly match the frozen v5 probe set")
        if diagnostic.get("collapse_forbids_compensation_only") is not True:
            errors.append("computation collapse must forbid compensation-only treatment")
        if diagnostic.get("represent_alone_never_proves_reconstruction") is not True:
            errors.append("generic represent operators must not bypass structural reconstruction")
        route = diagnostic.get("route")
        expected_routes = {
            "undetermined": "diagnostic_hold",
            "signal_degradation": "compensation_or_reconstruction",
            "computation_collapse": "structural_reconstruction",
            "mixed_failure": "reconstruction_then_compensation",
            "no_material_damage": "zero_treatment_control",
        }
        if route != expected_routes.get(diagnosis):
            errors.append("diagnostic route does not match the failure class")
        if mode == "executable" and not is_sha256(diagnostic.get("probe_receipt_sha256")):
            errors.append("executable diagnostic route requires a probe receipt")

    nodes_raw = program.get("operators")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        errors.append("operators must be a non-empty list")
        nodes_raw = []
    nodes: dict[str, dict[str, Any]] = {}
    external = False
    structural_reconstruction = False
    compensation = False
    for index, node in enumerate(nodes_raw):
        prefix = f"operators[{index}]"
        if not isinstance(node, dict):
            errors.append(f"{prefix} must be an object")
            continue
        expected_operator_fields = PLANNED_OPERATOR_FIELDS
        if mode == "executable" and node.get("kind") == "reconstruct":
            expected_operator_fields = EXECUTABLE_OPERATOR_FIELDS | {"representation_schema_sha256"}
        errors.extend(_exact_keys(node, expected_operator_fields, prefix))
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            errors.append(f"{prefix}.id missing")
            continue
        if node_id in nodes:
            errors.append(f"duplicate operator id {node_id}")
        nodes[node_id] = node
        kind = node.get("kind")
        if kind not in OPERATOR_KINDS:
            errors.append(f"{node_id}: invalid kind")
        if kind in EXTERNAL_KINDS:
            external = True
        if kind == "reconstruct" and node.get("treatment_role") == "structural_reconstruction":
            structural_reconstruction = True
        if kind in {"repair_static", "repair_conditional"} \
                and node.get("treatment_role") == "signal_compensation":
            compensation = True
        if mode == "executable":
            if node.get("treatment_role") not in TREATMENT_ROLES:
                errors.append(f"{node_id}: executable operator requires a recognized treatment_role")
            if kind == "reconstruct" and not is_sha256(node.get("representation_schema_sha256")):
                errors.append(f"{node_id}: structural reconstruction requires a representation schema hash")
        if node.get("implementation_status") not in IMPLEMENTATION_STATES:
            errors.append(f"{node_id}: invalid implementation_status")
        if not isinstance(node.get("mechanism"), str) or not node["mechanism"]:
            errors.append(f"{node_id}: mechanism missing")
        if not isinstance(node.get("source"), str) or not node["source"]:
            errors.append(f"{node_id}: source missing")
        dependencies = node.get("depends_on")
        if not isinstance(dependencies, list) or any(not isinstance(value, str) for value in dependencies):
            errors.append(f"{node_id}: depends_on must be a string list")
        executor = node.get("executor")
        errors.extend(_exact_keys(executor, EXECUTOR_FIELDS, f"{prefix}.executor"))
        if not isinstance(executor, dict) or not isinstance(executor.get("wired"), bool):
            errors.append(f"{node_id}: executor.wired Boolean required")
        elif executor["wired"]:
            if mode != "executable":
                errors.append(f"{node_id}: wired executor requires executable mode")
            if node.get("implementation_status") not in {"control", "measured", "prototype"}:
                errors.append(f"{node_id}: research/unimplemented executor cannot be wired")
            if not is_sha256(executor.get("source_sha256")):
                errors.append(f"{node_id}: wired executor requires source_sha256")
            if not isinstance(executor.get("adapter_id"), str) or not executor["adapter_id"]:
                errors.append(f"{node_id}: wired executor requires adapter_id")
        if mode == "executable" and isinstance(executor, dict) and executor.get("wired") is not True:
            errors.append(f"{node_id}: every executable operator must be wired")
    for node_id, node in nodes.items():
        for dependency in node.get("depends_on", []):
            if dependency not in nodes:
                errors.append(f"{node_id}: unknown dependency {dependency}")
    found = _cycle(nodes) if nodes else None
    if found:
        errors.append("operator DAG cycle: " + " -> ".join(found))
    if scope != "augmented_system" and external:
        errors.append("core claim scopes forbid external augmentation operators")
    if scope == "augmented_system" and not external:
        errors.append("augmented_system must explicitly represent its external mechanism")
    if scope == "codec_fidelity" and any(
        node.get("kind") in {"repair_static", "repair_conditional", "harden", "augment_external"}
        for node in nodes.values()
    ):
        errors.append("codec_fidelity forbids Doctor repair, hardening, and augmentation operators")
    failure_class = diagnostic.get("failure_class") if isinstance(diagnostic, dict) else None
    if mode == "executable":
        mutating = [node_id for node_id, node in nodes.items() if node.get("kind") in MUTATING_OPERATOR_KINDS]
        if failure_class == "computation_collapse" and not structural_reconstruction:
            errors.append("computation collapse requires an explicit structural reconstruction operator")
        elif failure_class == "mixed_failure" and not (structural_reconstruction and compensation):
            errors.append("mixed failure requires structural reconstruction followed by signal compensation")
        elif failure_class == "signal_degradation" and not compensation:
            errors.append("signal degradation requires an explicit signal-compensation operator")
        elif failure_class == "undetermined" and mutating:
            errors.append("undetermined diagnosis is a diagnostic hold and forbids treatment execution")
        elif failure_class == "no_material_damage" and mutating:
            errors.append("no-material-damage route must retain the zero-treatment control")

    errors.extend(validate_data_contract(program.get("data_contract"), mode))

    parent_teacher_receipt: Any = {}
    elevation_teacher_receipt: Any = None
    training = program.get("training_contract")
    if not isinstance(training, dict):
        errors.append("training_contract missing")
    else:
        errors.extend(_exact_keys(training, TRAINING_FIELDS, "training_contract"))
        if training.get("parent_teacher_required") is not True:
            errors.append("parent teacher is mandatory")
        if training.get("teacher_outputs_split_bound") is not True:
            errors.append("teacher outputs must be split-bound")
        if training.get("early_stop_on_frozen_selection_only") is not True:
            errors.append("early stopping must use frozen selection only")
        if training.get("retain_zero_treatment") is not True:
            errors.append("zero-treatment fallback must be retained")
        if training.get("gradient_conflict_measurement") is not True:
            errors.append("multi-capability training must measure gradient conflict")
        if training.get("bf16_same_treatment_control_required") is not True:
            errors.append("BF16 plus the same treatment/data/optimization control is required")
        expected_elevation = scope == "capability_elevation"
        if training.get("elevation_teacher_allowed") is not expected_elevation:
            errors.append("elevation-teacher permission does not match claim scope")
        if training.get("external_teacher_at_inference") is not False:
            errors.append("a core training teacher cannot remain resident at inference")
        if training.get("minimum_independent_training_seeds") != 5:
            errors.append("training minimum must be exactly five independent seeds")
        if training.get("minimum_calibration_draws") != 5:
            errors.append("calibration minimum must be exactly five independent draws")
        training_seeds = training.get("training_seeds")
        calibration_draws = training.get("calibration_draw_seeds")
        if mode == "planned":
            if training_seeds not in ("required_if_training", []) and not isinstance(training_seeds, list):
                errors.append("planned training seeds declaration invalid")
            if calibration_draws not in ("required", []) and not isinstance(calibration_draws, list):
                errors.append("planned calibration draws declaration invalid")
        else:
            training_applies = any(
                node.get("kind") in TRAINING_OPERATOR_KINDS for node in nodes.values()
            )
            if training_applies and (
                not isinstance(training_seeds, list)
                or len(training_seeds) < 5
                or len(set(training_seeds)) != len(training_seeds)
                or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in training_seeds)
            ):
                errors.append("training treatment requires at least five distinct integer seeds")
            if not training_applies and training_seeds not in ([], None):
                errors.append("training seeds are forbidden when the program has no training treatment")
            if not isinstance(calibration_draws, list) or len(calibration_draws) < 5 \
                    or len(set(calibration_draws)) != len(calibration_draws) \
                    or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in calibration_draws):
                errors.append("executable program requires at least five distinct calibration draws")
            for field, values in (
                ("training_seed_manifest_sha256", training_seeds or []),
                ("calibration_draw_manifest_sha256", calibration_draws or []),
            ):
                if training.get(field) != hash_value(values):
                    errors.append(f"training_contract.{field} is not bound to its declared seeds")
        teachers = training.get("teacher_contract")
        if not isinstance(teachers, dict) or set(teachers) != TEACHER_CONTRACT_FIELDS:
            errors.append("teacher_contract must exactly declare parent/elevation/inference authority")
            teachers = {}
        if teachers.get("no_teacher_at_inference") is not True:
            errors.append("all training teachers must be absent at inference")
        split_rows = program.get("data_contract", {}).get("splits", {})
        expected_teacher_splits = hash_value({
            split: split_rows.get(split, {}).get("manifest_sha256")
            for split in DATA_SPLITS
        })
        teacher_errors, parent_teacher_receipt = _validate_teacher(
            teachers.get("parent_teacher"), mode=mode, role="exact_identity_parent",
            claim_scope=scope, program_spec_sha256=declared_spec,
            expected_revision_sha256=model.get("parent_revision_sha256"),
            expected_split_manifest_sha256=expected_teacher_splits,
            trust_context=trust_context, trust_role="parent_teacher_authorization",
        )
        errors.extend(teacher_errors)
        elevation_teacher = teachers.get("elevation_teacher")
        if scope == "capability_elevation":
            teacher_errors, elevation_teacher_receipt = _validate_teacher(
                elevation_teacher, mode=mode, role="stronger_training_teacher",
                claim_scope=scope, program_spec_sha256=declared_spec,
                expected_revision_sha256=None,
                expected_split_manifest_sha256=expected_teacher_splits,
                trust_context=trust_context, trust_role="elevation_teacher_authorization",
            )
            errors.extend(teacher_errors)
            if isinstance(elevation_teacher, dict) \
                    and elevation_teacher.get("identity_sha256") == teachers.get(
                        "parent_teacher", {}
                    ).get("identity_sha256"):
                errors.append("elevation teacher must be concretely distinct from the exact parent")
        elif elevation_teacher is not None:
            errors.append("only capability_elevation may declare stronger-teacher fields or authority")

    evaluation = program.get("evaluation_contract")
    errors.extend(_exact_keys(evaluation, EVALUATION_FIELDS, "evaluation_contract"))
    if not isinstance(evaluation, dict):
        errors.append("evaluation_contract missing")
    else:
        if not isinstance(evaluation.get("capability_domains"), list) \
                or set(evaluation["capability_domains"]) != set(CAPABILITY_DOMAINS) \
                or len(evaluation["capability_domains"]) != len(CAPABILITY_DOMAINS):
            errors.append("evaluation must cover every v5 capability domain")
        required_competitors = evaluation.get("required_competitor_ids")
        if not isinstance(required_competitors, list) or not required_competitors \
                or any(not isinstance(value, str) or not value for value in required_competitors) \
                or len(required_competitors) != len(set(required_competitors)):
            errors.append("evaluation requires a frozen, unique, nonempty competitor-id list")
            required_competitors = []
        if mode == "executable" and "required" in required_competitors:
            errors.append("executable competitor list cannot contain a planned placeholder")
        competitor_set_hash = evaluation.get("required_competitor_ids_sha256")
        if mode == "planned" and competitor_set_hash == "required":
            pass
        elif competitor_set_hash != hash_value(sorted(required_competitors)):
            errors.append("required competitor-id set hash is not bound to the frozen list")
        if evaluation.get("per_item_outputs_required") is not True:
            errors.append("per-item evaluation outputs are required")
        if evaluation.get("blind_candidate_labels") is not True:
            errors.append("candidate labels must be blinded")
        if evaluation.get("paired_parent_prompts") is not True:
            errors.append("candidate and parent must receive paired prompts")
        if evaluation.get("judge_free_where_possible") is not True:
            errors.append("objective/judge-free scoring must be preferred")
        if evaluation.get("test_time_compute_matched") is not True:
            errors.append("test-time compute must be matched across parent, candidate, and competitors")
        budget = evaluation.get("test_time_compute_budget")
        errors.extend(_exact_keys(budget, MATCHED_COMPUTE_FIELDS, "evaluation_contract.test_time_compute_budget"))
        if not isinstance(budget, dict) or set(budget) != MATCHED_COMPUTE_FIELDS:
            errors.append("test_time_compute_budget must declare the complete v5 matched-compute tuple")
        elif any(
            field != "temperature" and (
                isinstance(budget[field], bool) or not isinstance(budget[field], int) or budget[field] < 0
            )
            for field in MATCHED_COMPUTE_FIELDS
        ) or not _finite(budget["temperature"], nonnegative=True):
            errors.append("matched-compute limits must be nonnegative integers except finite temperature")
        elif any(budget[field] <= 0 for field in (
            "max_input_tokens", "max_output_tokens", "samples_per_item", "timeout_ms"
        )):
            errors.append("input, output, sample, and timeout budgets must be positive")
        elif budget["max_reasoning_tokens"] > budget["max_output_tokens"]:
            errors.append("reasoning-token budget cannot exceed total output-token budget")
        elif scope != "augmented_system" and any(
            budget[field] != 0
            for field in ("retrieval_calls", "tool_calls", "external_model_calls")
        ):
            errors.append("non-augmented claims cannot spend retrieval/tool/external-model calls")
        for field in (
            "suite_manifest_sha256", "prompt_protocol_sha256", "scorer_source_sha256",
            "competitor_registry_sha256", "test_time_compute_protocol_sha256",
        ):
            if not _required_hash(evaluation.get(field), mode):
                errors.append(f"evaluation_contract.{field} invalid")

    execution = program.get("execution_contract")
    if mode == "planned":
        if not isinstance(execution, dict) or execution.get("state") != "not_authorized":
            errors.append("planned programs require an explicit not-authorized execution contract")
        elif set(execution) != EXECUTION_FIELDS:
            errors.append("planned execution contract must use the exact v5 field set")
    elif not isinstance(execution, dict):
        errors.append("executable program requires an execution contract")
    else:
        errors.extend(_exact_keys(execution, EXECUTION_FIELDS, "execution_contract"))
        if execution.get("state") != "authorized":
            errors.append("executable program is not explicitly authorized")
        campaign_owner = execution.get("campaign_owner_id")
        if not isinstance(campaign_owner, str) or not campaign_owner:
            errors.append("execution contract requires a campaign owner")
        root = execution.get("root_manifest")
        errors.extend(_exact_keys(root, ROOT_RECEIPT_FIELDS, "execution_contract.root_manifest"))
        parameter_receipt_sha256 = (
            parameter_receipt.get("receipt_sha256") if isinstance(parameter_receipt, dict) else None
        )
        teacher_receipts_sha256 = _teacher_receipt_set_sha256(
            parent_teacher_receipt, elevation_teacher_receipt
        )
        operators_adapters_sha256 = _operators_adapters_sha256(program)
        target_ceilings_sha256 = _target_ceilings_sha256(program)
        compute_budget_sha256 = hash_value(evaluation.get("test_time_compute_budget")) \
            if isinstance(evaluation, dict) else None
        if not _stamped(root, "binding_sha256"):
            errors.append("root manifest binding is missing or self-inconsistent")
            root = {}
        else:
            if root.get("experiment_id") != binding.get("experiment_id") \
                    or root.get("candidate_identity_sha256") != binding.get("candidate_identity_sha256"):
                errors.append("root manifest does not bind the exact experiment candidate")
            if root.get("policy_version") != POLICY_VERSION or not is_sha256(root.get("manifest_sha256")):
                errors.append("root manifest policy/hash binding invalid")
            if root.get("parent_revision_sha256") != model.get("parent_revision_sha256"):
                errors.append("root manifest does not bind the exact parent revision")
            if root.get("manifest_sha256") != package_root_sha256 \
                    or root.get("package_root_schema") != package_root_schema \
                    or root.get("program_spec_sha256") != declared_spec \
                    or root.get("parameter_manifest_receipt_sha256") != parameter_receipt_sha256 \
                    or root.get("teacher_authorization_receipts_sha256") != teacher_receipts_sha256:
                errors.append("root authorization does not bind final program spec/root/parameter/teacher authority")
        greenlight = execution.get("greenlight_receipt")
        errors.extend(_exact_keys(
            greenlight, GREENLIGHT_RECEIPT_FIELDS, "execution_contract.greenlight_receipt"
        ))
        if not _stamped(greenlight, "receipt_sha256"):
            errors.append("greenlight receipt is missing or self-inconsistent")
        elif greenlight.get("granted") is not True \
                or greenlight.get("root_manifest_sha256") != root.get("manifest_sha256") \
                or greenlight.get("candidate_identity_sha256") != binding.get("candidate_identity_sha256") \
                or greenlight.get("claim_scope") != scope \
                or greenlight.get("program_spec_sha256") != declared_spec \
                or greenlight.get("root_binding_sha256") != root.get("binding_sha256") \
                or greenlight.get("parameter_manifest_receipt_sha256") != parameter_receipt_sha256 \
                or greenlight.get("teacher_authorization_receipts_sha256") != teacher_receipts_sha256 \
                or not isinstance(greenlight.get("issued_by"), str) \
                or not greenlight.get("issued_by") \
                or not is_sha256(greenlight.get("signer_key_sha256")) \
                or not is_sha256(greenlight.get("signature_sha256")):
            errors.append("greenlight receipt is not bound to this root/candidate/scope/signer")
        allowlist = execution.get("adapter_allowlist")
        errors.extend(_exact_keys(allowlist, ALLOWLIST_FIELDS, "execution_contract.adapter_allowlist"))
        allow_entries = allowlist.get("entries", []) if isinstance(allowlist, dict) else []
        if not _stamped(allowlist, "allowlist_sha256"):
            errors.append("adapter allowlist is missing or self-inconsistent")
        else:
            if allowlist.get("program_spec_sha256") != declared_spec \
                    or allowlist.get("operators_adapters_sha256") != operators_adapters_sha256:
                errors.append("adapter allowlist does not bind final program spec/operators")
            if not isinstance(allow_entries, list) or not allow_entries:
                errors.append("adapter allowlist must be non-empty")
        if not isinstance(allow_entries, list):
            allow_entries = []
        allowed_pairs: set[tuple[str, str]] = set()
        for row in allow_entries:
            errors.extend(_exact_keys(row, ALLOWLIST_ENTRY_FIELDS, "execution_contract.adapter_allowlist.entries[]"))
            if not isinstance(row, dict) or not isinstance(row.get("adapter_id"), str) \
                    or not row.get("adapter_id") or not is_sha256(row.get("source_sha256")):
                errors.append("adapter allowlist contains an invalid entry")
                continue
            pair = (row["adapter_id"], row["source_sha256"])
            if pair in allowed_pairs:
                errors.append("adapter allowlist entries must be unique")
            allowed_pairs.add(pair)
        for node_id, node in nodes.items():
            executor = node.get("executor", {})
            if (executor.get("adapter_id"), executor.get("source_sha256")) not in allowed_pairs:
                errors.append(f"{node_id}: executor is absent from the exact adapter allowlist")
        admission = execution.get("resource_admission")
        errors.extend(_exact_keys(
            admission, RESOURCE_ADMISSION_FIELDS, "execution_contract.resource_admission"
        ))
        if not _stamped(admission, "receipt_sha256"):
            errors.append("resource-admission receipt is missing or self-inconsistent")
        else:
            integer_fields = (
                "available_memory_bytes", "available_disk_bytes", "resident_bytes_limit",
                "swap_bytes", "requested_peak_resident_bytes", "requested_scratch_disk_bytes",
            )
            if admission.get("admitted") is not True \
                    or admission.get("root_manifest_sha256") != root.get("manifest_sha256") \
                    or admission.get("program_spec_sha256") != declared_spec \
                    or admission.get("parameter_manifest_receipt_sha256") != parameter_receipt_sha256 \
                    or admission.get("exact_parameter_count") != exact_parameters \
                    or admission.get("target_ceilings_sha256") != target_ceilings_sha256 \
                    or admission.get("compute_budget_sha256") != compute_budget_sha256 \
                    or admission.get("operators_adapters_sha256") != operators_adapters_sha256 \
                    or not is_sha256(admission.get("host_identity_sha256")) \
                    or not is_sha256(admission.get("verifier_source_sha256")) \
                    or admission.get("memory_pressure") != "normal" \
                    or admission.get("thermal_state") not in {"nominal", "fair"} \
                    or admission.get("ac_power") is not True \
                    or any(isinstance(admission.get(field), bool) or not isinstance(admission.get(field), int)
                           or admission[field] < 0 for field in integer_fields):
                errors.append("resource-admission receipt is incomplete or unhealthy")
            else:
                if admission["requested_peak_resident_bytes"] > admission["resident_bytes_limit"] \
                        or admission["resident_bytes_limit"] > admission["available_memory_bytes"]:
                    errors.append("resource admission exceeds the admitted memory envelope")
                if admission["requested_scratch_disk_bytes"] > admission["available_disk_bytes"]:
                    errors.append("resource admission exceeds available disk")
                target_resident = target.get("resident_bytes_ceiling") if isinstance(target, dict) else None
                if _finite(target_resident, positive=True) \
                        and admission["requested_peak_resident_bytes"] > target_resident:
                    errors.append("resource admission exceeds the program resident-byte ceiling")
        executable_trust = {
            "root_manifest": root.get("binding_sha256") if isinstance(root, dict) else None,
            "greenlight_receipt": (
                greenlight.get("receipt_sha256") if isinstance(greenlight, dict) else None
            ),
            "adapter_allowlist": (
                allowlist.get("allowlist_sha256") if isinstance(allowlist, dict) else None
            ),
            "resource_admission_receipt": (
                admission.get("receipt_sha256") if isinstance(admission, dict) else None
            ),
        }
        for role, identity_sha256 in executable_trust.items():
            if not _is_trusted(trust_context, role, identity_sha256):
                errors.append(f"external trust context does not authorize {role}")

    resume = program.get("exact_resume_contract")
    errors.extend(_exact_keys(resume, EXACT_RESUME_FIELDS, "exact_resume_contract"))
    if not isinstance(resume, dict) or not isinstance(resume.get("required_state"), list) \
            or set(resume["required_state"]) != set(EXACT_RESUME_STATE) \
            or len(resume["required_state"]) != len(EXACT_RESUME_STATE):
        errors.append("exact_resume_contract state is incomplete")
    elif not all(resume.get(field) is True for field in (
        "atomic_replace", "fsync_file", "fsync_parent_directory", "validate_before_resume"
    )):
        errors.append("exact resume requires atomic replace, fsync, and validation")

    output = program.get("output_contract")
    errors.extend(_exact_keys(output, OUTPUT_FIELDS, "output_contract"))
    if not isinstance(output, dict):
        errors.append("output_contract missing")
    else:
        if output.get("actual_file_bytes_authoritative") is not True:
            errors.append("actual output file bytes must be authoritative")
        if output.get("all_tensor_ownership_required") is not True:
            errors.append("all tensor ownership is required")
        if output.get("dense_parent_fallback_forbidden") is not True:
            errors.append("dense parent fallback must be forbidden")
        billed_rows = output.get("billed_components")
        if not isinstance(billed_rows, list):
            errors.append("output_contract.billed_components must be a list")
            billed_rows = []
        billed = set(billed_rows)
        mandatory = {"base", "pass_through", "codebooks", "indices", "corrections", "routers", "metadata", "alignment"}
        if not mandatory <= billed:
            errors.append("output bill omits mandatory artifact components")
        if billed != set(ARTIFACT_COMPONENTS) or len(billed_rows) != len(ARTIFACT_COMPONENTS):
            errors.append("output bill must enumerate every physical component category exactly once")

    metadata = program.get("campaign_metadata")
    errors.extend(_exact_keys(metadata, CAMPAIGN_METADATA_FIELDS, "campaign_metadata"))
    if isinstance(metadata, dict):
        if metadata.get("schema") != CAMPAIGN_METADATA_SCHEMA:
            errors.append(f"campaign_metadata.schema must be {CAMPAIGN_METADATA_SCHEMA}")
        if not _stamped(metadata, "metadata_sha256"):
            errors.append("campaign_metadata identity is missing or mismatched")
        for field in (
            "campaign_sha256", "campaign_input_binding_sha256", "canonical_policy_bundle_sha256",
            "direct_competitor_registry_sha256", "parameter_count_binding_sha256",
            "teacher_authority_contract_sha256", "test_time_compute_budget_sha256",
            "research_contracts_sha256",
        ):
            if not _required_hash(metadata.get(field), mode):
                errors.append(f"campaign_metadata.{field} invalid")
        if not isinstance(metadata.get("explicit_ordinal"), int) or isinstance(
            metadata.get("explicit_ordinal"), bool
        ) or metadata["explicit_ordinal"] < 0:
            errors.append("campaign_metadata.explicit_ordinal must be a nonnegative integer")
        if not isinstance(metadata.get("evidence_stage"), str) or not metadata["evidence_stage"] \
                or not isinstance(metadata.get("evidence_state"), str) or not metadata["evidence_state"]:
            errors.append("campaign_metadata evidence stage/state missing")
        if not isinstance(metadata.get("launch_permitted"), bool) \
                or not isinstance(metadata.get("greenlight_recorded"), bool):
            errors.append("campaign_metadata launch/greenlight fields must be Boolean")
        elif mode == "planned" and (
            metadata["launch_permitted"] is not False or metadata["greenlight_recorded"] is not False
        ):
            errors.append("planned campaign metadata cannot authorize execution")
        blockers = metadata.get("blockers")
        if not isinstance(blockers, list) or not blockers \
                or any(not isinstance(value, str) or not value for value in blockers) \
                or len(blockers) != len(set(blockers)):
            errors.append("campaign_metadata.blockers must be a unique nonempty string list")

    expected = program.get("program_sha256")
    if not is_sha256(expected) or expected != hash_value(_identity_payload(program, "program_sha256")):
        errors.append("program_sha256 missing or mismatched")
    return errors


def _metric_row_errors(domain: str, row: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(row, dict):
        return [f"metric {domain} must be an object"]
    errors.extend(_exact_keys(row, METRIC_FIELDS, f"capability_metrics.{domain}"))
    for field in ("candidate", "parent", "delta", "delta_lcb", "delta_ucb", "noninferiority_margin"):
        if not _finite(row.get(field)):
            errors.append(f"metric {domain}.{field} must be finite")
    if not isinstance(row.get("n"), int) or isinstance(row.get("n"), bool) or row["n"] < 5:
        errors.append(f"metric {domain}.n must contain at least five independent clusters")
    if all(_finite(row.get(field)) for field in ("delta_lcb", "delta", "delta_ucb")) \
            and not (row["delta_lcb"] <= row["delta"] <= row["delta_ucb"]):
        errors.append(f"metric {domain} confidence interval is not ordered")
    if _finite(row.get("noninferiority_margin")) and row["noninferiority_margin"] < 0:
        errors.append(f"metric {domain} noninferiority margin must be nonnegative")
    if _finite(row.get("candidate")) and _finite(row.get("parent")) and _finite(row.get("delta")) \
            and not _same_number(row["delta"], row["candidate"] - row["parent"]):
        errors.append(f"metric {domain}.delta must equal candidate minus parent")
    for field in ("raw_p_value", "holm_adjusted_p_value"):
        if not _valid_probability(row.get(field)):
            errors.append(f"metric {domain}.{field} must be in [0, 1]")
    if _valid_probability(row.get("raw_p_value")) \
            and _valid_probability(row.get("holm_adjusted_p_value")) \
            and row["holm_adjusted_p_value"] < row["raw_p_value"]:
        errors.append(f"metric {domain} Holm-adjusted p-value cannot be below its raw p-value")
    if not isinstance(row.get("ci_method"), str) or not row["ci_method"]:
        errors.append(f"metric {domain}.ci_method missing")
    if not is_sha256(row.get("evidence_slice_sha256")):
        errors.append(f"metric {domain}.evidence_slice_sha256 missing")
    if row.get("hypothesis_id") != f"parent_noninferiority:{domain}" \
            or row.get("hypothesis_direction") != "candidate_minus_parent_greater_than_negative_margin":
        errors.append(f"metric {domain} parent noninferiority hypothesis/direction invalid")
    if _finite(row.get("noninferiority_margin")) \
            and not _same_number(row.get("null_boundary"), -row["noninferiority_margin"]):
        errors.append(f"metric {domain} null boundary must equal negative noninferiority margin")
    if not isinstance(row.get("noninferiority_pass"), bool):
        errors.append(f"metric {domain}.noninferiority_pass Boolean required")
    return errors


def validate_artifact(
    artifact: Any,
    program: dict[str, Any],
    *,
    verify_files: bool = False,
    base_dir: Path | None = None,
    trust_context: Any = None,
    expected_package_root_manifest_sha256: str | None = None,
) -> list[str]:
    """Validate one immutable artifact; optionally stat and hash every real file.

    Structural validation never claims that a path exists.  Callers producing a
    receipt for execution or evidence must use ``verify_files=True`` and provide
    the manifest's root directory as ``base_dir``.
    """
    errors: list[str] = []
    program_errors = validate_program(
        program, trust_context=trust_context,
        expected_package_root_manifest_sha256=expected_package_root_manifest_sha256,
    )
    if program_errors:
        return ["bound program invalid: " + error for error in program_errors]
    if not isinstance(artifact, dict):
        return ["artifact must be an object"]
    expected_artifact_fields = set(ARTIFACT_FIELDS)
    if program.get("model", {}).get("active_b") is not None:
        expected_artifact_fields.add("moe_accounting")
    errors.extend(_exact_keys(artifact, expected_artifact_fields, "artifact"))
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        errors.append(f"schema must be {ARTIFACT_SCHEMA}")
    if artifact.get("program_sha256") != program.get("program_sha256"):
        errors.append("artifact is not bound to the exact program")
    if artifact.get("claim_scope") != program.get("claim_scope"):
        errors.append("artifact claim scope differs from program")
    if artifact.get("dense_parent_fallback") is not False:
        errors.append("dense parent fallback is forbidden")
    if artifact.get("all_tensor_ownership_complete") is not True:
        errors.append("all tensor ownership must be complete")
    for field in (
        "parent_revision_sha256", "packed_semantics_sha256", "decoder_semantics_sha256",
        "tensor_ownership_sha256", "runtime_abi_sha256",
    ):
        if not is_sha256(artifact.get(field)):
            errors.append(f"artifact {field} missing")

    exact_parameters = artifact.get("exact_parameter_count")
    if isinstance(exact_parameters, bool) or not isinstance(exact_parameters, int) \
            or exact_parameters <= 0:
        errors.append("artifact requires an exact positive integer parameter count")
    program_parameters = program.get("model", {}).get("exact_parameter_count")
    if isinstance(program_parameters, int) and exact_parameters != program_parameters:
        errors.append("artifact parameter count differs from the executable program")

    files = artifact.get("files")
    file_sum = 0
    if not isinstance(files, list) or not files:
        errors.append("artifact files must be a non-empty list")
        files = []
    paths: set[str] = set()
    root: Path | None = None
    if verify_files:
        if base_dir is None:
            errors.append("verify_files=True requires base_dir")
        else:
            root = Path(base_dir).resolve()
            if not root.is_dir():
                errors.append("artifact base_dir is not a directory")
    for index, row in enumerate(files):
        prefix = f"files[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix} must be an object")
            continue
        errors.extend(_exact_keys(row, ARTIFACT_FILE_FIELDS, prefix))
        path = row.get("path")
        valid_path = isinstance(path, str) and bool(path) and path not in paths
        if not valid_path:
            errors.append(f"{prefix}.path missing or duplicate")
        else:
            paths.add(path)
            relative = Path(path)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"{prefix}.path must be relative and traversal-free")
                valid_path = False
        if row.get("component") not in ARTIFACT_COMPONENTS:
            errors.append(f"{prefix}.component invalid")
        if not is_sha256(row.get("sha256")):
            errors.append(f"{prefix}.sha256 invalid")
        if isinstance(row.get("bytes"), bool) or not isinstance(row.get("bytes"), int) \
                or row["bytes"] < 0:
            errors.append(f"{prefix}.bytes invalid")
        else:
            file_sum += row["bytes"]
        if verify_files and root is not None and valid_path:
            candidate = root / Path(path)
            try:
                resolved = candidate.resolve(strict=True)
                if resolved.parent != root and root not in resolved.parents:
                    errors.append(f"{prefix}.path escapes base_dir")
                elif candidate.is_symlink() or not resolved.is_file():
                    errors.append(f"{prefix}.path must be a regular non-symlink file")
                else:
                    actual_bytes = resolved.stat().st_size
                    if actual_bytes != row.get("bytes"):
                        errors.append(f"{prefix}.bytes differs from stat result")
                    if _sha256_file(resolved) != row.get("sha256"):
                        errors.append(f"{prefix}.sha256 differs from actual file bytes")
            except (FileNotFoundError, OSError):
                errors.append(f"{prefix}.path cannot be opened and verified")

    ledger = artifact.get("byte_ledger")
    errors.extend(_exact_keys(ledger, set(ARTIFACT_COMPONENTS), "byte_ledger"))
    if not isinstance(ledger, dict) or set(ledger) != set(ARTIFACT_COMPONENTS):
        errors.append("byte_ledger must exactly name every v5 artifact component")
        ledger = {}
    elif any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ledger.values()):
        errors.append("byte_ledger values must be nonnegative integers")
    component_manifests = artifact.get("component_manifests")
    errors.extend(_exact_keys(
        component_manifests, set(ARTIFACT_COMPONENTS), "component_manifests"
    ))
    if not isinstance(component_manifests, dict) \
            or set(component_manifests) != set(ARTIFACT_COMPONENTS):
        errors.append("component_manifests must exactly name every component")
        component_manifests = {}
    for component in ARTIFACT_COMPONENTS:
        payload = _component_file_payload(files, component)
        component_bytes = sum(row["bytes"] for row in payload)
        if ledger.get(component) != component_bytes:
            errors.append(f"byte_ledger.{component} does not equal its file-category sum")
        manifest = component_manifests.get(component)
        errors.extend(_exact_keys(
            manifest, COMPONENT_MANIFEST_FIELDS, f"component_manifests.{component}"
        ))
        if not isinstance(manifest, dict) or set(manifest) != {"bytes", "file_count", "files_sha256"}:
            errors.append(f"component_manifests.{component} is incomplete")
        elif manifest["bytes"] != component_bytes \
                or manifest["file_count"] != len(payload) \
                or manifest["files_sha256"] != hash_value(payload):
            errors.append(f"component_manifests.{component} is not cross-bound to its files")

    ledger_sum = sum(value for value in ledger.values() if isinstance(value, int) and value >= 0)
    all_in = artifact.get("all_in_physical_bytes")
    if isinstance(all_in, bool) or not isinstance(all_in, int) or all_in <= 0:
        errors.append("all_in_physical_bytes must be positive")
    elif all_in != file_sum or all_in != ledger_sum:
        errors.append("all-in bytes must equal both file sum and every component-ledger sum")
    ceiling = program.get("target", {}).get("physical_artifact_bytes_ceiling")
    if isinstance(all_in, int) and _finite(ceiling, positive=True) and all_in > ceiling:
        errors.append("artifact exceeds program physical-byte ceiling")

    payload_components = {
        "base", "pass_through", "scales", "codebooks", "indices", "corrections",
        "routers", "state", "alignment", "tokenizer",
    }
    payload_bytes = sum(ledger.get(component, 0) for component in payload_components)
    external_bytes = sum(ledger.get(component, 0) for component in (
        "retrieval_index", "auxiliary_models", "persistent_external_state"
    ))
    accounting = artifact.get("physical_accounting")
    errors.extend(_exact_keys(accounting, PHYSICAL_ACCOUNTING_FIELDS, "physical_accounting"))
    expected_accounting = {
        "stored_artifact_bytes": all_in,
        "payload_bytes": payload_bytes,
        "decoder_runtime_bytes": ledger.get("decoder_runtime", 0),
        "runtime_dependency_bytes": ledger.get("runtime_dependencies", 0),
        "metadata_bytes": ledger.get("metadata", 0),
        "context_state_bytes": ledger.get("context_state", 0),
        "external_system_bytes": external_bytes,
    }
    if accounting != expected_accounting:
        errors.append("physical_accounting must exactly partition payload/runtime/metadata/context/external bytes")

    expected_bpw = (8.0 * all_in / exact_parameters) \
        if isinstance(all_in, int) and isinstance(exact_parameters, int) and exact_parameters > 0 else None
    all_in_bpw = artifact.get("all_in_bpw")
    if not _finite(all_in_bpw, positive=True) or expected_bpw is None \
            or not _same_number(all_in_bpw, expected_bpw):
        errors.append("all_in_bpw must derive from all-in bytes and exact integer parameters")
    expected_payload_bpw = (8.0 * payload_bytes / exact_parameters) \
        if isinstance(exact_parameters, int) and exact_parameters > 0 else None
    if expected_payload_bpw is None or not _finite(artifact.get("payload_bpw"), nonnegative=True) \
            or not _same_number(artifact["payload_bpw"], expected_payload_bpw):
        errors.append("payload_bpw must derive from billed payload bytes and exact parameters")

    integer_resident_fields = (
        "decoded_resident_bytes", "resident_peak_bytes", "expected_active_bytes", "worst_active_bytes",
    )
    for field in integer_resident_fields:
        if isinstance(artifact.get(field), bool) or not isinstance(artifact.get(field), int) \
                or artifact[field] < 0:
            errors.append(f"artifact {field} must be a nonnegative integer")
    if all(isinstance(artifact.get(field), int) for field in ("expected_active_bytes", "worst_active_bytes")) \
            and artifact["expected_active_bytes"] > artifact["worst_active_bytes"]:
        errors.append("expected active bytes cannot exceed worst active bytes")
    peaks = artifact.get("peak_bytes_by_context")
    valid_peaks = isinstance(peaks, dict) and bool(peaks) and all(
        isinstance(context, str) and context.isdigit() and int(context) > 0
        and isinstance(value, int) and not isinstance(value, bool) and value > 0
        for context, value in peaks.items()
    )
    if not valid_peaks:
        errors.append("peak_bytes_by_context must be a non-empty positive token-context map")
        peaks = {}
    runtime = artifact.get("runtime_accounting")
    errors.extend(_exact_keys(runtime, RUNTIME_ACCOUNTING_FIELDS, "runtime_accounting"))
    runtime_integer_fields = (
        "decoder_resident_bytes", "runtime_dependency_resident_bytes",
        "metadata_resident_bytes", "persistent_state_resident_bytes",
    )
    if not isinstance(runtime, dict) or any(
        isinstance(runtime.get(field), bool) or not isinstance(runtime.get(field), int) or runtime[field] < 0
        for field in runtime_integer_fields
    ):
        errors.append("runtime_accounting fixed resident categories are incomplete")
        runtime = {}
    context_resident = runtime.get("context_state_bytes_by_context")
    if not isinstance(context_resident, dict) or set(context_resident) != set(peaks) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in context_resident.values()
    ):
        errors.append("runtime context-state accounting must exactly match declared contexts")
        context_resident = {}
    fixed_resident = sum(
        runtime.get(field, 0) for field in runtime_integer_fields if isinstance(runtime.get(field), int)
    )
    if isinstance(artifact.get("decoded_resident_bytes"), int) \
            and runtime.get("decoder_resident_bytes") != artifact["decoded_resident_bytes"]:
        errors.append("decoded resident bytes must match runtime accounting")
    for context, peak in peaks.items():
        if peak < fixed_resident + context_resident.get(context, 0):
            errors.append(f"context {context} peak omits fixed or context resident bytes")
    if peaks and artifact.get("resident_peak_bytes") != max(peaks.values()):
        errors.append("resident_peak_bytes must equal the maximum context peak")
    resident_ceiling = program.get("target", {}).get("resident_bytes_ceiling")
    if peaks and _finite(resident_ceiling, positive=True) \
            and max(peaks.values()) > resident_ceiling:
        errors.append("artifact exceeds program resident-byte ceiling")

    if program.get("model", {}).get("active_b") is not None:
        moe = artifact.get("moe_accounting")
        errors.extend(_exact_keys(moe, MOE_ACCOUNTING_FIELDS, "moe_accounting"))
        required_moe = (
            "expert_count", "routed_experts_per_token_expected", "routed_experts_per_token_max",
            "total_parameter_count", "expected_active_parameter_count", "worst_active_parameter_count",
            "total_installed_expert_bytes", "expected_active_expert_bytes", "worst_active_expert_bytes",
        )
        if not isinstance(moe, dict) or any(
            isinstance(moe.get(field), bool) or not isinstance(moe.get(field), int) or moe[field] < 0
            for field in required_moe
        ):
            errors.append("MoE artifact requires exact installed, routed, active-parameter, and byte accounting")
        elif moe["expert_count"] <= 0 \
                or moe["routed_experts_per_token_expected"] > moe["routed_experts_per_token_max"] \
                or moe["routed_experts_per_token_max"] > moe["expert_count"] \
                or moe["total_parameter_count"] != exact_parameters \
                or moe["expected_active_parameter_count"] > moe["worst_active_parameter_count"] \
                or moe["worst_active_parameter_count"] > moe["total_parameter_count"] \
                or moe["expected_active_expert_bytes"] > moe["worst_active_expert_bytes"] \
                or moe["worst_active_expert_bytes"] > moe["total_installed_expert_bytes"]:
            errors.append("MoE active/installed accounting is internally inconsistent")
    if program.get("claim_scope") != "augmented_system" and external_bytes != 0:
        errors.append("non-augmented artifact contains external-system bytes")
    expected = artifact.get("artifact_sha256")
    if not is_sha256(expected) or expected != hash_value(_identity_payload(artifact, "artifact_sha256")):
        errors.append("artifact_sha256 missing or mismatched")
    return errors


def validate_observation(
    observation: Any,
    program: dict[str, Any],
    *,
    trust_context: Any = None,
    expected_package_root_manifest_sha256: str | None = None,
) -> list[str]:
    """Validate result evidence; ``PROVEN`` is unreachable from summaries alone."""
    errors: list[str] = []
    program_errors = validate_program(
        program, trust_context=trust_context,
        expected_package_root_manifest_sha256=expected_package_root_manifest_sha256,
    )
    if program_errors:
        return ["bound program invalid: " + error for error in program_errors]
    if not isinstance(observation, dict):
        return ["observation must be an object"]
    errors.extend(_exact_keys(observation, OBSERVATION_FIELDS, "observation"))
    if observation.get("schema") != OBSERVATION_SCHEMA:
        errors.append(f"schema must be {OBSERVATION_SCHEMA}")
    if observation.get("program_sha256") != program.get("program_sha256"):
        errors.append("observation is not bound to the exact program")
    if observation.get("claim_scope") != program.get("claim_scope"):
        errors.append("observation claim scope differs from program")
    status = observation.get("status")
    proof_state = observation.get("proof_state")
    evidence_state = observation.get("evidence_state")
    if status not in OBSERVATION_STATUSES:
        errors.append("observation status invalid")
    if proof_state not in PROOF_STATES:
        errors.append("observation proof_state invalid")
    if evidence_state not in EVIDENCE_STATES:
        errors.append("observation evidence_state invalid")
    allowed_states = {
        "planned": {("planned", "PLANNED")},
        "running": {
            (state, "RUNNING") for state in PROOF_STATES[:5]
        },
        "succeeded": {
            *((state, "PROVISIONAL") for state in PROOF_STATES[1:7]),
            ("sealed_final", "UNREPRODUCED"),
            ("independent_reproduction", "PROVEN"),
        },
        "complete_negative": {
            *((state, "PROVISIONAL") for state in PROOF_STATES[1:7]),
            ("sealed_final", "UNREPRODUCED"),
            ("independent_reproduction", "PROVEN"),
        },
        "failed_retryable": {
            *((state, "INVALID") for state in PROOF_STATES[:-2]),
        },
        "failed_terminal": {
            *((state, "INVALID") for state in PROOF_STATES),
        },
        "invalidated": {
            *((state, evidence) for state in PROOF_STATES for evidence in ("INVALID", "REVOKED")),
        },
    }
    if status in allowed_states and (proof_state, evidence_state) not in allowed_states[status]:
        errors.append("observation status/proof/evidence transition is forbidden")
    if status not in {"planned", "running"} and program.get("mode") != "executable":
        errors.append("completed observations must bind a fail-closed executable program")
    if evidence_state == "PROVEN" and (status not in {"succeeded", "complete_negative"}
                                         or proof_state != "independent_reproduction"):
        errors.append("PROVEN requires a completed independent reproduction")
    if observation.get("negative_result_retained") is not True:
        errors.append("negative results must be retained")
    if observation.get("speed_claimed") is not False:
        errors.append("Doctor v5 quality observation cannot claim speed")
    artifact_manifest = observation.get("artifact_manifest_sha256")
    if not is_sha256(artifact_manifest):
        errors.append("artifact manifest hash missing")
    if isinstance(observation.get("physical_artifact_bytes"), bool) \
            or not isinstance(observation.get("physical_artifact_bytes"), int) \
            or observation["physical_artifact_bytes"] <= 0:
        errors.append("physical_artifact_bytes must be positive")

    metrics = observation.get("capability_metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(CAPABILITY_DOMAINS):
        errors.append("capability_metrics must cover every v5 domain")
        metrics = {}
    for domain in CAPABILITY_DOMAINS:
        errors.extend(_metric_row_errors(domain, metrics.get(domain)))

    statistics = observation.get("statistical_contract")
    errors.extend(_exact_keys(statistics, STATISTICAL_FIELDS, "statistical_contract"))
    if not isinstance(statistics, dict):
        errors.append("statistical_contract missing")
        statistics = {}
    else:
        if statistics.get("paired") is not True:
            errors.append("statistical comparison must be paired")
        if statistics.get("cluster_resampling") is not True:
            errors.append("statistics must resample at the preregistered independence cluster")
        if statistics.get("multiple_testing_correction") != "holm":
            errors.append("Holm familywise correction is required")
        if not _finite(statistics.get("familywise_alpha"), positive=True) \
                or statistics["familywise_alpha"] > 0.05:
            errors.append("familywise_alpha must be in (0, 0.05]")
        if statistics.get("confidence") != 0.95:
            errors.append("95% confidence contract is required")
        if statistics.get("preregistered") is not True:
            errors.append("margins and tests must be preregistered")
        if not is_sha256(statistics.get("preregistration_sha256")):
            errors.append("statistical preregistration hash missing")

    alpha = statistics.get("familywise_alpha")
    nominal_alpha = 1.0 - statistics.get("confidence", 0.0) \
        if _finite(statistics.get("confidence")) else None
    for domain in CAPABILITY_DOMAINS:
        row = metrics.get(domain, {})
        ci_supports_direction = bool(
            _finite(row.get("delta_lcb")) and _finite(row.get("null_boundary"))
            and row["delta_lcb"] > row["null_boundary"]
        )
        raw_p_supports_direction = bool(
            _valid_probability(row.get("raw_p_value")) and _finite(nominal_alpha, positive=True)
            and row["raw_p_value"] <= nominal_alpha
        )
        if ci_supports_direction is not raw_p_supports_direction:
            errors.append(f"metric {domain} CI direction and raw p-value are inconsistent")
        expected_noninferiority = bool(
            _finite(row.get("delta_lcb")) and _finite(row.get("null_boundary"))
            and _valid_probability(row.get("holm_adjusted_p_value"))
            and _finite(alpha, positive=True)
            and row["delta_lcb"] > row["null_boundary"]
            and row["holm_adjusted_p_value"] <= alpha
        )
        if row.get("noninferiority_pass") is not expected_noninferiority:
            errors.append(
                f"metric {domain} noninferiority pass must derive from CI direction and Holm-adjusted alpha"
            )

    comparisons = observation.get("competitor_comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        errors.append("at least one same-budget competitor comparison is required")
        comparisons = []
    competitor_ids: set[str] = set()
    for index, comparison in enumerate(comparisons):
        prefix = f"competitor_comparisons[{index}]"
        if not isinstance(comparison, dict):
            errors.append(f"{prefix} must be an object")
            continue
        errors.extend(_exact_keys(comparison, COMPARISON_FIELDS, prefix))
        competitor_id = comparison.get("competitor_id")
        if not isinstance(competitor_id, str) or not competitor_id or competitor_id in competitor_ids:
            errors.append(f"{prefix}.competitor_id missing or duplicate")
        else:
            competitor_ids.add(competitor_id)
        if comparison.get("same_parent") is not True:
            errors.append(f"{prefix} must use the same parent")
        if comparison.get("same_or_lower_physical_bytes") is not True:
            errors.append(f"{prefix} must be same-budget")
        if comparison.get("same_prompt_and_scorer") is not True:
            errors.append(f"{prefix} must use identical prompts and scorers")
        if not is_sha256(comparison.get("competitor_artifact_sha256")) \
                or not is_sha256(comparison.get("evidence_slice_sha256")):
            errors.append(f"{prefix} artifact/evidence binding missing")
        competitor_scores = comparison.get("competitor_domain_scores")
        domain_deltas = comparison.get("domain_deltas")
        delta_lcbs = comparison.get("domain_delta_lcbs")
        raw_ps = comparison.get("domain_raw_p_values")
        corrected_ps = comparison.get("domain_holm_adjusted_p_values")
        domain_passes = comparison.get("domain_superiority_passes")
        domain_hypotheses = comparison.get("domain_hypothesis_ids")
        maps = (competitor_scores, domain_deltas, delta_lcbs, raw_ps, corrected_ps)
        if any(not isinstance(value, dict) or set(value) != set(CAPABILITY_DOMAINS) for value in maps):
            errors.append(f"{prefix} domain score/delta/CI/p-value maps are incomplete")
            continue
        if not isinstance(domain_passes, dict) or set(domain_passes) != set(CAPABILITY_DOMAINS) \
                or any(not isinstance(value, bool) for value in domain_passes.values()) \
                or not isinstance(domain_hypotheses, dict) \
                or set(domain_hypotheses) != set(CAPABILITY_DOMAINS):
            errors.append(f"{prefix} domain superiority hypotheses/pass map incomplete")
            continue
        if comparison.get("superiority_direction") != "candidate_greater_than_competitor":
            errors.append(f"{prefix} superiority direction invalid")
        if any(not _finite(value) for mapping in (competitor_scores, domain_deltas, delta_lcbs)
               for value in mapping.values()) \
                or any(not _valid_probability(value) for mapping in (raw_ps, corrected_ps)
                       for value in mapping.values()):
            errors.append(f"{prefix} domain summaries contain invalid numbers")
        for domain in CAPABILITY_DOMAINS:
            candidate_score = metrics.get(domain, {}).get("candidate")
            if _finite(candidate_score) and _finite(competitor_scores.get(domain)) \
                    and _finite(domain_deltas.get(domain)) \
                    and not _same_number(
                        domain_deltas[domain], candidate_score - competitor_scores[domain]
                    ):
                errors.append(f"{prefix}.{domain} delta does not equal candidate minus competitor")
            if _valid_probability(raw_ps.get(domain)) and _valid_probability(corrected_ps.get(domain)) \
                    and corrected_ps[domain] < raw_ps[domain]:
                errors.append(f"{prefix}.{domain} corrected p-value is below raw p-value")
            expected_domain_pass = bool(
                _finite(delta_lcbs.get(domain))
                and _finite(comparison.get("superiority_delta"), nonnegative=True)
                and _valid_probability(corrected_ps.get(domain))
                and _finite(alpha, positive=True)
                and delta_lcbs[domain] > comparison["superiority_delta"]
                and corrected_ps[domain] <= alpha
            )
            ci_supports_direction = bool(
                _finite(delta_lcbs.get(domain))
                and _finite(comparison.get("superiority_delta"), nonnegative=True)
                and delta_lcbs[domain] > comparison["superiority_delta"]
            )
            raw_p_supports_direction = bool(
                _valid_probability(raw_ps.get(domain)) and _finite(nominal_alpha, positive=True)
                and raw_ps[domain] <= nominal_alpha
            )
            if ci_supports_direction is not raw_p_supports_direction:
                errors.append(f"{prefix}.{domain} CI direction and raw p-value are inconsistent")
            if domain_hypotheses.get(domain) != f"competitor_superiority:{competitor_id}:{domain}" \
                    or domain_passes.get(domain) is not expected_domain_pass:
                errors.append(
                    f"{prefix}.{domain} superiority pass must derive from hypothesis/CI/Holm alpha"
                )
        if all(_finite(value) for value in domain_deltas.values()) and not _same_number(
            comparison.get("macro_delta"), sum(domain_deltas.values()) / len(CAPABILITY_DOMAINS)
        ):
            errors.append(f"{prefix}.macro_delta is not the deterministic domain mean")
        if not _finite(comparison.get("macro_delta_lcb")):
            errors.append(f"{prefix}.macro_delta_lcb missing")
        if not _valid_probability(comparison.get("macro_raw_p_value")) \
                or not _valid_probability(comparison.get("macro_holm_adjusted_p_value")):
            errors.append(f"{prefix} macro p-values invalid")
        elif comparison["macro_holm_adjusted_p_value"] < comparison["macro_raw_p_value"]:
            errors.append(f"{prefix} macro corrected p-value is below raw p-value")
        if not _finite(comparison.get("superiority_delta"), nonnegative=True):
            errors.append(f"{prefix}.superiority_delta must be nonnegative finite")
        expected_macro_pass = bool(
            _finite(comparison.get("macro_delta_lcb"))
            and _finite(comparison.get("superiority_delta"), nonnegative=True)
            and _valid_probability(comparison.get("macro_holm_adjusted_p_value"))
            and _finite(alpha, positive=True)
            and comparison["macro_delta_lcb"] > comparison["superiority_delta"]
            and comparison["macro_holm_adjusted_p_value"] <= alpha
        )
        macro_ci_supports = bool(
            _finite(comparison.get("macro_delta_lcb"))
            and _finite(comparison.get("superiority_delta"), nonnegative=True)
            and comparison["macro_delta_lcb"] > comparison["superiority_delta"]
        )
        macro_raw_p_supports = bool(
            _valid_probability(comparison.get("macro_raw_p_value"))
            and _finite(nominal_alpha, positive=True)
            and comparison["macro_raw_p_value"] <= nominal_alpha
        )
        if macro_ci_supports is not macro_raw_p_supports:
            errors.append(f"{prefix} macro CI direction and raw p-value are inconsistent")
        if comparison.get("macro_hypothesis_id") != f"competitor_macro_superiority:{competitor_id}" \
                or comparison.get("macro_superiority_pass") is not expected_macro_pass:
            errors.append(f"{prefix} macro superiority pass must derive from CI direction and Holm alpha")

    required_competitor_ids = set(
        program.get("evaluation_contract", {}).get("required_competitor_ids", [])
    )
    if competitor_ids != required_competitor_ids:
        missing = sorted(required_competitor_ids - competitor_ids)
        extras = sorted(competitor_ids - required_competitor_ids)
        errors.append(
            "completed observation competitor set differs from the frozen registry: "
            f"missing={missing}, extras={extras}"
        )

    for field, message in (
        ("data_firewall_pass", "data firewall must pass"),
        ("parent_parity_protocol_pass", "parent/candidate protocol parity must pass"),
        ("test_time_compute_match_pass", "matched test-time compute contract must pass"),
        ("comparator_set_frozen_before_final", "comparator set must be frozen before final evaluation"),
        ("sealed_final_consumed_once", "sealed final set may be consumed only once by this lineage"),
    ):
        if observation.get(field) is not True:
            errors.append(message)
    compute_receipt = observation.get("test_time_compute_receipt")
    errors.extend(_exact_keys(compute_receipt, COMPUTE_RECEIPT_FIELDS, "test_time_compute_receipt"))
    compute_budget = program.get("evaluation_contract", {}).get("test_time_compute_budget")
    if not _stamped(compute_receipt, "receipt_sha256"):
        errors.append("matched-compute receipt is missing or self-inconsistent")
    elif compute_receipt.get("passed") is not True \
            or compute_receipt.get("budget_sha256") != hash_value(compute_budget) \
            or not is_sha256(compute_receipt.get("participant_manifest_sha256")) \
            or not is_sha256(compute_receipt.get("measurement_log_sha256")):
        errors.append("matched-compute receipt is not bound to the exact budget and participants")

    snapshot = observation.get("claim_snapshot")
    errors.extend(_exact_keys(snapshot, CLAIM_SNAPSHOT_FIELDS, "claim_snapshot"))
    if not isinstance(snapshot, dict):
        errors.append("claim_snapshot missing")
        snapshot = {}
    else:
        if not isinstance(snapshot.get("as_of_date"), str) or not snapshot["as_of_date"]:
            errors.append("claim_snapshot.as_of_date missing")
        for field in ("competitor_registry_sha256", "quality_battery_sha256", "artifact_sha256"):
            if not is_sha256(snapshot.get(field)):
                errors.append(f"claim_snapshot.{field} missing")
        if snapshot.get("competitor_registry_sha256") != program.get(
            "evaluation_contract", {}
        ).get("competitor_registry_sha256"):
            errors.append("claim snapshot is not bound to the frozen competitor registry")
        if snapshot.get("artifact_sha256") != artifact_manifest:
            errors.append("claim snapshot artifact differs from the observed artifact manifest")
        if snapshot.get("expires_on_registry_benchmark_artifact_or_runtime_change") is not True:
            errors.append("claim snapshot must expire on evidence-scope changes")
    if program.get("claim_scope") != "augmented_system" \
            and observation.get("external_runtime_bytes", 0) != 0:
        errors.append("core observation cannot use external runtime bytes")

    evidence = observation.get("evidence_bundle")
    errors.extend(_exact_keys(evidence, EVIDENCE_BUNDLE_FIELDS, "evidence_bundle"))
    if not isinstance(evidence, dict):
        errors.append("raw evidence bundle missing")
        evidence = {}
    artifact_receipt = evidence.get("artifact_validation_receipt")
    errors.extend(_exact_keys(
        artifact_receipt, ARTIFACT_VALIDATION_RECEIPT_FIELDS,
        "evidence_bundle.artifact_validation_receipt",
    ))
    exact_parameters = program.get("model", {}).get("exact_parameter_count")
    physical_bytes = observation.get("physical_artifact_bytes")
    expected_all_in_bpw = (
        8.0 * physical_bytes / exact_parameters
        if isinstance(physical_bytes, int) and isinstance(exact_parameters, int)
        and exact_parameters > 0 else None
    )
    if not _stamped(artifact_receipt, "receipt_sha256"):
        errors.append("artifact-validation receipt is missing or self-inconsistent")
        artifact_receipt = {}
    elif artifact_receipt.get("program_sha256") != program.get("program_sha256") \
            or artifact_receipt.get("artifact_sha256") != artifact_manifest \
            or artifact_receipt.get("exact_parameter_count") != exact_parameters \
            or artifact_receipt.get("all_in_physical_bytes") != physical_bytes \
            or expected_all_in_bpw is None \
            or not _same_number(artifact_receipt.get("all_in_bpw"), expected_all_in_bpw) \
            or not is_sha256(artifact_receipt.get("aggregate_file_manifest_sha256")) \
            or not is_sha256(artifact_receipt.get("aggregate_component_manifest_sha256")) \
            or not is_sha256(artifact_receipt.get("validator_source_sha256")) \
            or artifact_receipt.get("verify_files") is not True:
        errors.append("artifact-validation receipt does not prove the bound strict file validation")
    if status in {"succeeded", "complete_negative"} and not _is_trusted(
        trust_context, "artifact_validation_receipt", artifact_receipt.get("receipt_sha256")
    ):
        errors.append("external trust context does not authorize the artifact-validation receipt")
    raw_fields = (
        "per_item_outputs_sha256", "cluster_assignments_sha256", "cluster_outputs_sha256",
        "data_firewall_receipt_sha256", "parent_parity_receipt_sha256",
        "calibration_draw_results_sha256",
    )
    for field in raw_fields:
        if not is_sha256(evidence.get(field)):
            errors.append(f"evidence_bundle.{field} missing")
    training_applies = any(
        node.get("kind") in TRAINING_OPERATOR_KINDS for node in program.get("operators", [])
    )
    if training_applies and not is_sha256(evidence.get("training_seed_results_sha256")):
        errors.append("training treatment requires raw per-seed result evidence")
    training_contract = program.get("training_contract", {})
    expected_training_count = len(training_contract.get("training_seeds", [])) \
        if isinstance(training_contract.get("training_seeds"), list) else 0
    expected_calibration_count = len(training_contract.get("calibration_draw_seeds", [])) \
        if isinstance(training_contract.get("calibration_draw_seeds"), list) else 0
    if evidence.get("training_seed_count") != expected_training_count:
        errors.append("evidence training-seed count differs from the executable program")
    if evidence.get("calibration_draw_count") != expected_calibration_count \
            or expected_calibration_count < 5:
        errors.append("evidence must cover every one of at least five calibration draws")
    raw_index_payload = {
        field: evidence.get(field) for field in raw_fields
    } | {
        "artifact_validation_receipt_sha256": artifact_receipt.get("receipt_sha256"),
        "training_seed_results_sha256": evidence.get("training_seed_results_sha256"),
        "training_seed_count": evidence.get("training_seed_count"),
        "calibration_draw_count": evidence.get("calibration_draw_count"),
    }
    raw_index_sha256 = hash_value(raw_index_payload)
    if evidence.get("raw_evidence_index_sha256") != raw_index_sha256:
        errors.append("raw evidence index is not cross-bound to per-item/cluster/seed evidence")

    summary_sha256 = hash_value(_summary_payload(observation))
    raw_p_payload = {
        "parent": {domain: metrics.get(domain, {}).get("raw_p_value") for domain in CAPABILITY_DOMAINS},
        "competitors": {
            row.get("competitor_id"): {
                "domains": row.get("domain_raw_p_values"),
                "macro": row.get("macro_raw_p_value"),
            }
            for row in comparisons if isinstance(row, dict)
        },
    }
    corrected_p_payload = {
        "parent": {
            domain: metrics.get(domain, {}).get("holm_adjusted_p_value")
            for domain in CAPABILITY_DOMAINS
        },
        "competitors": {
            row.get("competitor_id"): {
                "domains": row.get("domain_holm_adjusted_p_values"),
                "macro": row.get("macro_holm_adjusted_p_value"),
            }
            for row in comparisons if isinstance(row, dict)
        },
    }
    corrected = evidence.get("corrected_test_receipt")
    errors.extend(_exact_keys(
        corrected, CORRECTED_TEST_RECEIPT_FIELDS, "evidence_bundle.corrected_test_receipt"
    ))
    family_size = len(CAPABILITY_DOMAINS) + len(comparisons) * (len(CAPABILITY_DOMAINS) + 1)
    if not _stamped(corrected, "receipt_sha256"):
        errors.append("corrected-test receipt is missing or self-inconsistent")
        corrected = {}
    elif corrected.get("procedure") != "holm" \
            or not _same_number(corrected.get("familywise_alpha"), statistics.get("familywise_alpha")) \
            or corrected.get("family_size") != family_size \
            or corrected.get("raw_p_values_sha256") != hash_value(raw_p_payload) \
            or corrected.get("corrected_p_values_sha256") != hash_value(corrected_p_payload) \
            or corrected.get("summary_sha256") != summary_sha256 \
            or corrected.get("all_hypotheses_corrected") is not True:
        errors.append("corrected-test receipt does not bind every declared test and summary")
    verifier = evidence.get("evidence_verifier_receipt")
    errors.extend(_exact_keys(
        verifier, EVIDENCE_VERIFIER_FIELDS, "evidence_bundle.evidence_verifier_receipt"
    ))
    if not _stamped(verifier, "receipt_sha256"):
        errors.append("evidence-verifier receipt is missing or self-inconsistent")
        verifier = {}
    elif verifier.get("raw_evidence_index_sha256") != raw_index_sha256 \
            or verifier.get("summary_sha256") != summary_sha256 \
            or verifier.get("corrected_test_receipt_sha256") != corrected.get("receipt_sha256") \
            or not is_sha256(verifier.get("verifier_source_sha256")) \
            or not isinstance(verifier.get("owner_id"), str) or not verifier.get("owner_id") \
            or verifier.get("recomputed_deltas_cis_pvalues_from_raw") is not True \
            or verifier.get("passed") is not True:
        errors.append("evidence verifier did not independently recompute the bound summaries")
    if evidence_state == "PROVEN" and not _is_trusted(
        trust_context, "evidence_verifier_receipt", verifier.get("receipt_sha256")
    ):
        errors.append("external trust context does not authorize the evidence-verifier receipt")

    if proof_state in {"sealed_final", "independent_reproduction"}:
        sealed = evidence.get("sealed_service_attestation")
        errors.extend(_exact_keys(
            sealed, SEALED_ATTESTATION_FIELDS, "evidence_bundle.sealed_service_attestation"
        ))
        sealed_manifest = program.get("data_contract", {}).get("splits", {}).get(
            "sealed_final", {}
        ).get("manifest_sha256")
        if not _stamped(sealed, "attestation_sha256"):
            errors.append("sealed-service attestation is missing or self-inconsistent")
            sealed = {}
        elif not isinstance(sealed.get("service_id"), str) or not sealed.get("service_id") \
                or not isinstance(sealed.get("owner_id"), str) or not sealed.get("owner_id") \
                or sealed.get("sealed_final_manifest_sha256") != sealed_manifest \
                or sealed.get("program_sha256") != program.get("program_sha256") \
                or sealed.get("artifact_sha256") != artifact_manifest \
                or sealed.get("raw_evidence_index_sha256") != raw_index_sha256 \
                or not is_sha256(sealed.get("execution_receipt_sha256")) \
                or not is_sha256(sealed.get("one_time_nonce_sha256")) \
                or sealed.get("consumed_once") is not True:
            errors.append("sealed-service attestation does not bind the one-time sealed execution")
        if not _is_trusted(
            trust_context, "sealed_service_attestation", sealed.get("attestation_sha256")
        ):
            errors.append("external trust context does not authorize the sealed-service attestation")
    else:
        sealed = {}
    if proof_state == "independent_reproduction":
        independent = evidence.get("independent_owner_attestation")
        errors.extend(_exact_keys(
            independent, INDEPENDENT_ATTESTATION_FIELDS,
            "evidence_bundle.independent_owner_attestation",
        ))
        campaign_owner = program.get("execution_contract", {}).get("campaign_owner_id")
        if not _stamped(independent, "attestation_sha256"):
            errors.append("independent-owner attestation is missing or self-inconsistent")
            independent = {}
        elif not isinstance(independent.get("owner_id"), str) or not independent.get("owner_id") \
                or independent.get("owner_id") in {campaign_owner, sealed.get("owner_id")} \
                or independent.get("program_sha256") != program.get("program_sha256") \
                or independent.get("artifact_sha256") != artifact_manifest \
                or independent.get("raw_evidence_index_sha256") != raw_index_sha256 \
                or independent.get("summary_sha256") != summary_sha256 \
                or independent.get("sealed_attestation_sha256") != sealed.get("attestation_sha256") \
                or not is_sha256(independent.get("replication_receipt_sha256")) \
                or independent.get("independently_executed") is not True \
                or independent.get("no_shared_runtime_owner") is not True:
            errors.append("independent-owner attestation is not owner-separated and evidence-bound")
        campaign_owner = program.get("execution_contract", {}).get("campaign_owner_id")
        if sealed.get("owner_id") == campaign_owner:
            errors.append("sealed service owner must differ from the campaign owner")
        if not _is_trusted(
            trust_context, "independent_owner_attestation", independent.get("attestation_sha256")
        ):
            errors.append("external trust context does not authorize the independent-owner attestation")

    expected = observation.get("observation_sha256")
    if not is_sha256(expected) or expected != hash_value(_identity_payload(observation, "observation_sha256")):
        errors.append("observation_sha256 missing or mismatched")
    return errors


def dominance_decision(
    observation: dict[str, Any],
    program: dict[str, Any],
    *,
    trust_context: Any = None,
    expected_package_root_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Apply the predeclared v5 quality-dominance rule.

    A winner must be non-inferior to its parent in every capability domain and
    strictly dominate every same-budget competitor in the macro lower bound,
    without a domain-specific lower bound falling below that domain's declared
    non-inferiority margin.  This is intentionally an intersection-union rule:
    one weak domain blocks the headline.
    """
    errors = validate_observation(
        observation, program, trust_context=trust_context,
        expected_package_root_manifest_sha256=expected_package_root_manifest_sha256,
    )
    parent_failures: list[str] = []
    competitor_failures: list[str] = []
    uniform_failures: list[str] = []
    if not errors:
        for domain, row in observation["capability_metrics"].items():
            if row.get("noninferiority_pass") is not True:
                parent_failures.append(domain)
        for comparison in observation["competitor_comparisons"]:
            delta = comparison["superiority_delta"]
            weak = [
                domain
                for domain, lower_bound in comparison["domain_delta_lcbs"].items()
                if lower_bound < -observation["capability_metrics"][domain]["noninferiority_margin"]
            ]
            strictly_better = [
                domain for domain, passed in comparison["domain_superiority_passes"].items()
                if passed is True
            ]
            uniform_weak = [
                domain for domain, passed in comparison["domain_superiority_passes"].items()
                if passed is not True
            ]
            if comparison.get("macro_superiority_pass") is not True or weak or not strictly_better:
                competitor_failures.append(
                    f"{comparison['competitor_id']}: macro_lcb={comparison['macro_delta_lcb']}, "
                    f"weak={weak}, strict={strictly_better}"
                )
            if uniform_weak:
                uniform_failures.append(f"{comparison['competitor_id']}: {uniform_weak}")
    proof_state = observation.get("proof_state")
    independent = observation.get("status") == "succeeded" \
        and proof_state == "independent_reproduction" \
        and observation.get("evidence_state") == "PROVEN" \
        and program.get("mode") == "executable"
    frontier_champion = not errors and not parent_failures and not competitor_failures and independent
    uniform_quality_dominance = frontier_champion and not uniform_failures
    passed = uniform_quality_dominance
    payload = {
        "schema": DOMINANCE_SCHEMA,
        "program_sha256": program.get("program_sha256"),
        "observation_sha256": observation.get("observation_sha256"),
        "claim_scope": program.get("claim_scope"),
        "passed": passed,
        "frontier_champion": frontier_champion,
        "uniform_quality_dominance": uniform_quality_dominance,
        "parent_noninferior_all_domains": not parent_failures,
        "same_budget_competitors_dominated": not competitor_failures,
        "independent_reproduction": independent,
        "parent_failure_domains": parent_failures,
        "competitor_failures": competitor_failures,
        "uniform_quality_failures": uniform_failures,
        "validation_errors": errors,
        "claim": (
            "uniformly quality-dominant under Doctor-v5 preregistered scope"
            if passed else (
                "frontier champion but not uniform quality dominance"
                if frontier_champion else "unproven; no unbeatable/dominant claim permitted"
            )
        ),
    }
    payload["dominance_sha256"] = hash_value(payload)
    return payload


def validate_dominance(
    dominance: Any,
    observation: dict[str, Any],
    program: dict[str, Any],
    *,
    trust_context: Any = None,
    expected_package_root_manifest_sha256: str | None = None,
) -> list[str]:
    errors = _exact_keys(dominance, DOMINANCE_FIELDS, "dominance")
    if not isinstance(dominance, dict):
        return errors
    expected = dominance_decision(
        observation, program, trust_context=trust_context,
        expected_package_root_manifest_sha256=expected_package_root_manifest_sha256,
    )
    if dominance != expected:
        errors.append("dominance document differs from the deterministic contract decision")
    return errors


def planned_data_contract() -> dict[str, Any]:
    splits = {}
    for split in DATA_SPLITS:
        splits[split] = {
            "manifest_sha256": "required",
            "examples": 0,
            "tokens": 0,
            "accessible_to_optimizer": split not in {
                "shadow", "frozen_final", "sealed_final", "independent_replication"
            },
        }
    return {
        "splits": splits,
        "exact_duplicate_scan_required": True,
        "near_duplicate_scan_required": True,
        "semantic_contamination_scan_required": True,
        "sealed_final_reveal_after_training": True,
        "teacher_output_cache_split_bound": True,
    }


def planned_program(
    *,
    experiment_id: str,
    candidate_identity_sha256: str,
    model_id: str,
    params_b: float,
    active_b: float | None,
    target_bpw: float,
    claim_scope: str,
    failure_class: str,
    operators: list[dict[str, Any]],
) -> dict[str, Any]:
    program = {
        "schema": PROGRAM_SCHEMA,
        "policy_version": POLICY_VERSION,
        "program_spec_schema": PROGRAM_SPEC_SCHEMA,
        "program_spec_sha256": "required",
        "package_root_schema": "required",
        "package_root_manifest_sha256": "required",
        "mode": "planned",
        "experiment_binding": {
            "experiment_id": experiment_id,
            "candidate_identity_sha256": candidate_identity_sha256,
        },
        "claim_scope": claim_scope,
        "model": {
            "id": model_id,
            "params_b": params_b,
            "exact_parameter_count": "required",
            "parameter_count_binding_sha256": "required",
            "exact_parameter_count_source_sha256": "required",
            "parameter_manifest_receipt": "required",
            "active_b": active_b,
            "parent_revision_sha256": "required",
            "config_sha256": "required",
            "tokenizer_sha256": "required",
            "chat_template_sha256": "required",
        },
        "target": {
            "physical_bpw_ceiling": target_bpw,
            "physical_artifact_bytes_ceiling": max(1, int(params_b * 1_000_000_000 * target_bpw / 8)),
            "resident_bytes_ceiling": 64_000_000_000,
            "speed_deferred": True,
            "bpw_is_all_in_physical": False,
            "physical_byte_ceiling_basis": "nominal_params_b_projection_not_evidence",
            "unresolved_exact_parameter_count_blocks_evidence_and_launch": True,
        },
        "diagnostic_contract": {
            "failure_class": failure_class,
            "route": {
                "undetermined": "diagnostic_hold",
                "signal_degradation": "compensation_or_reconstruction",
                "computation_collapse": "structural_reconstruction",
                "mixed_failure": "reconstruction_then_compensation",
                "no_material_damage": "zero_treatment_control",
            }[failure_class],
            "required_probes": [
                "early_signal_survival", "internal_geometry", "capability_failures",
                "hidden_logit_divergence", "adversarial_quantization_trigger_scan",
            ],
            "collapse_forbids_compensation_only": True,
            "represent_alone_never_proves_reconstruction": True,
            "probe_receipt_sha256": "required",
        },
        "operators": operators,
        "data_contract": planned_data_contract(),
        "training_contract": {
            "parent_teacher_required": True,
            "teacher_outputs_split_bound": True,
            "early_stop_on_frozen_selection_only": True,
            "retain_zero_treatment": True,
            "gradient_conflict_measurement": True,
            "bf16_same_treatment_control_required": True,
            "elevation_teacher_allowed": claim_scope == "capability_elevation",
            "external_teacher_at_inference": False,
            "minimum_independent_training_seeds": 5,
            "minimum_calibration_draws": 5,
            "training_seeds": "required_if_training",
            "calibration_draw_seeds": "required",
            "training_seed_manifest_sha256": "required",
            "calibration_draw_manifest_sha256": "required",
            "teacher_contract": {
                "parent_teacher": {
                    "identity_sha256": "required",
                    "revision_sha256": "required",
                    "role": "exact_identity_parent",
                    "output_protocol_sha256": "required",
                    "cache_manifest_sha256": "required",
                    "split_manifest_sha256": "required",
                    "provenance_manifest_sha256": "required",
                    "training_only": True,
                    "authorization_receipt": "required",
                },
                "elevation_teacher": ({
                    "identity_sha256": "required",
                    "revision_sha256": "required",
                    "role": "stronger_training_teacher",
                    "output_protocol_sha256": "required",
                    "cache_manifest_sha256": "required",
                    "split_manifest_sha256": "required",
                    "provenance_manifest_sha256": "required",
                    "training_only": True,
                    "authorization_receipt": "required",
                } if claim_scope == "capability_elevation" else None),
                "no_teacher_at_inference": True,
            },
        },
        "evaluation_contract": {
            "capability_domains": list(CAPABILITY_DOMAINS),
            "required_competitor_ids": ["required"],
            "required_competitor_ids_sha256": "required",
            "per_item_outputs_required": True,
            "blind_candidate_labels": True,
            "paired_parent_prompts": True,
            "judge_free_where_possible": True,
            "test_time_compute_matched": True,
            "test_time_compute_budget": {
                "max_input_tokens": 131072,
                "max_output_tokens": 32768,
                "max_reasoning_tokens": 32768,
                "samples_per_item": 1,
                "temperature": 0.0,
                "timeout_ms": 1_800_000,
                "verifier_retries": 2 if claim_scope == "augmented_system" else 0,
                "retrieval_calls": 4 if claim_scope == "augmented_system" else 0,
                "tool_calls": 8 if claim_scope == "augmented_system" else 0,
                "external_model_calls": 0,
            },
            "suite_manifest_sha256": "required",
            "prompt_protocol_sha256": "required",
            "scorer_source_sha256": "required",
            "competitor_registry_sha256": "required",
            "test_time_compute_protocol_sha256": "required",
        },
        "execution_contract": {
            "state": "not_authorized",
            "root_manifest": "required",
            "greenlight_receipt": "required",
            "adapter_allowlist": "required",
            "resource_admission": "required",
            "campaign_owner_id": "required",
        },
        "exact_resume_contract": {
            "required_state": list(EXACT_RESUME_STATE),
            "atomic_replace": True,
            "fsync_file": True,
            "fsync_parent_directory": True,
            "validate_before_resume": True,
        },
        "output_contract": {
            "actual_file_bytes_authoritative": True,
            "all_tensor_ownership_required": True,
            "dense_parent_fallback_forbidden": True,
            "billed_components": list(ARTIFACT_COMPONENTS),
        },
        "campaign_metadata": stamp({
            "schema": CAMPAIGN_METADATA_SCHEMA,
            "campaign_sha256": "required",
            "campaign_input_binding_sha256": "required",
            "canonical_policy_bundle_sha256": "required",
            "direct_competitor_registry_sha256": "required",
            "parameter_count_binding_sha256": "required",
            "teacher_authority_contract_sha256": "required",
            "test_time_compute_budget_sha256": "required",
            "research_contracts_sha256": "required",
            "explicit_ordinal": 0,
            "evidence_stage": "F0",
            "evidence_state": "PLANNED",
            "launch_permitted": False,
            "greenlight_recorded": False,
            "blockers": ["required"],
        }, "metadata_sha256"),
    }
    program["program_spec_sha256"] = compute_program_spec_sha256(program)
    return stamp(program, "program_sha256")


def _sample_operator(
    node_id: str,
    kind: str,
    dependencies: Iterable[str],
    *,
    mechanism: str | None = None,
    status: str = "research",
) -> dict[str, Any]:
    roles = {
        "diagnose": "diagnostic_only",
        "transform": "representation_search",
        "represent": "representation_search",
        "reconstruct": "structural_reconstruction",
        "repair_static": "signal_compensation",
        "repair_conditional": "signal_compensation",
        "train": "signal_compensation",
        "harden": "signal_compensation",
        "augment_external": "external_augmentation",
        "package": "packaging",
        "evaluate": "evaluation",
    }
    return {
        "id": node_id,
        "kind": kind,
        "mechanism": mechanism or node_id,
        "source": "local:selftest",
        "implementation_status": status,
        "treatment_role": roles[kind],
        "depends_on": list(dependencies),
        "executor": {"wired": False, "adapter_id": None, "source_sha256": None},
    }


def _sample_executable(planned: dict[str, Any]) -> dict[str, Any]:
    program = copy.deepcopy(planned)
    program["mode"] = "executable"
    metadata = copy.deepcopy(program["campaign_metadata"])
    for index, field in enumerate((
        "campaign_sha256", "campaign_input_binding_sha256", "canonical_policy_bundle_sha256",
        "direct_competitor_registry_sha256", "parameter_count_binding_sha256",
        "teacher_authority_contract_sha256", "test_time_compute_budget_sha256",
        "research_contracts_sha256",
    ), start=1):
        metadata[field] = f"{index:x}" * 64
    program["campaign_metadata"] = stamp(metadata, "metadata_sha256")
    program["package_root_schema"] = PACKAGE_ROOT_SCHEMA
    program["package_root_manifest_sha256"] = "c" * 64
    program["model"]["exact_parameter_count"] = 7_000_000_000
    program["model"]["parameter_count_binding_sha256"] = "5" * 64
    program["model"]["exact_parameter_count_source_sha256"] = "6" * 64
    program["target"]["bpw_is_all_in_physical"] = True
    program["target"]["physical_byte_ceiling_basis"] = "source_bound_exact_parameter_count"
    program["target"]["unresolved_exact_parameter_count_blocks_evidence_and_launch"] = False
    for index, field in enumerate((
        "parent_revision_sha256", "config_sha256", "tokenizer_sha256", "chat_template_sha256"
    ), start=1):
        program["model"][field] = f"{index:x}" * 64
    program["diagnostic_contract"]["probe_receipt_sha256"] = "5" * 64
    allow_entries = []
    for node in program["operators"]:
        source_sha256 = hash_value({"selftest_executor": node["id"]})
        adapter_id = f"selftest:{node['id']}"
        node["implementation_status"] = "prototype"
        node["executor"] = {
            "wired": True,
            "adapter_id": adapter_id,
            "source_sha256": source_sha256,
        }
        if node["kind"] == "reconstruct":
            node["representation_schema_sha256"] = "6" * 64
        allow_entries.append({"adapter_id": adapter_id, "source_sha256": source_sha256})
    for index, split in enumerate(DATA_SPLITS):
        program["data_contract"]["splits"][split]["manifest_sha256"] = hash_value(
            {"selftest_split": split, "ordinal": index}
        )
        program["data_contract"]["splits"][split]["examples"] = 100
        program["data_contract"]["splits"][split]["tokens"] = 1000
    training = program["training_contract"]
    training["training_seeds"] = [11, 12, 13, 14, 15]
    training["calibration_draw_seeds"] = [21, 22, 23, 24, 25]
    training["training_seed_manifest_sha256"] = hash_value(training["training_seeds"])
    training["calibration_draw_manifest_sha256"] = hash_value(training["calibration_draw_seeds"])
    teacher_splits_sha256 = hash_value({
        split: program["data_contract"]["splits"][split]["manifest_sha256"]
        for split in DATA_SPLITS
    })
    parent_teacher = {
        "identity_sha256": "a" * 64,
        "revision_sha256": program["model"]["parent_revision_sha256"],
        "role": "exact_identity_parent",
        "output_protocol_sha256": "b" * 64,
        "cache_manifest_sha256": "c" * 64,
        "split_manifest_sha256": teacher_splits_sha256,
        "provenance_manifest_sha256": "d" * 64,
        "training_only": True,
        "authorization_receipt": "required",
    }
    elevation_teacher = None
    if program["claim_scope"] == "capability_elevation":
        elevation_teacher = {
            "identity_sha256": "e" * 64,
            "revision_sha256": "f" * 64,
            "role": "stronger_training_teacher",
            "output_protocol_sha256": "1" * 64,
            "cache_manifest_sha256": "2" * 64,
            "split_manifest_sha256": teacher_splits_sha256,
            "provenance_manifest_sha256": "3" * 64,
            "training_only": True,
            "authorization_receipt": "required",
        }
    training["teacher_contract"] = {
        "parent_teacher": parent_teacher,
        "elevation_teacher": elevation_teacher,
        "no_teacher_at_inference": True,
    }
    required_competitors = [
        "same-parent-same-bytes-control",
        "same-parent-second-frozen-control",
    ]
    program["evaluation_contract"]["required_competitor_ids"] = required_competitors
    program["evaluation_contract"]["required_competitor_ids_sha256"] = hash_value(
        sorted(required_competitors)
    )
    for index, field in enumerate((
        "suite_manifest_sha256", "prompt_protocol_sha256", "scorer_source_sha256",
        "competitor_registry_sha256", "test_time_compute_protocol_sha256",
    ), start=7):
        program["evaluation_contract"][field] = f"{index:x}"[-1] * 64
    program["execution_contract"] = {
        "state": "authorized",
        "campaign_owner_id": "selftest-campaign-owner",
        "root_manifest": "required",
        "greenlight_receipt": "required",
        "adapter_allowlist": "required",
        "resource_admission": "required",
    }
    program["program_spec_sha256"] = compute_program_spec_sha256(program)
    spec_sha256 = program["program_spec_sha256"]
    parameter_receipt = stamp({
        "program_spec_sha256": spec_sha256,
        "exact_parameter_count": program["model"]["exact_parameter_count"],
        "tensor_ownership_manifest_sha256": "4" * 64,
        "tensor_classification_aggregate_sha256": "5" * 64,
        "parent_revision_sha256": program["model"]["parent_revision_sha256"],
        "config_sha256": program["model"]["config_sha256"],
        "source_shard_manifest_sha256": "6" * 64,
        "source_file_manifest_sha256": "7" * 64,
        "counting_code_sha256": "8" * 64,
        "all_tensors_classified": True,
    }, "receipt_sha256")
    program["model"]["parameter_manifest_receipt"] = parameter_receipt

    def authorize_teacher(teacher: dict[str, Any]) -> dict[str, Any]:
        return stamp({
            "authorized": True,
            "program_spec_sha256": spec_sha256,
            "claim_scope": program["claim_scope"],
            "teacher_identity_sha256": teacher["identity_sha256"],
            "teacher_revision_sha256": teacher["revision_sha256"],
            "teacher_role": teacher["role"],
            "output_protocol_sha256": teacher["output_protocol_sha256"],
            "cache_manifest_sha256": teacher["cache_manifest_sha256"],
            "split_manifest_sha256": teacher["split_manifest_sha256"],
            "provenance_manifest_sha256": teacher["provenance_manifest_sha256"],
            "training_only": True,
        }, "receipt_sha256")

    parent_teacher["authorization_receipt"] = authorize_teacher(parent_teacher)
    if elevation_teacher is not None:
        elevation_teacher["authorization_receipt"] = authorize_teacher(elevation_teacher)
    teacher_receipts_sha256 = _teacher_receipt_set_sha256(
        parent_teacher["authorization_receipt"],
        elevation_teacher["authorization_receipt"] if elevation_teacher else None,
    )
    root = stamp({
        "manifest_sha256": program["package_root_manifest_sha256"],
        "package_root_schema": program["package_root_schema"],
        "program_spec_sha256": spec_sha256,
        "parameter_manifest_receipt_sha256": parameter_receipt["receipt_sha256"],
        "teacher_authorization_receipts_sha256": teacher_receipts_sha256,
        "experiment_id": program["experiment_binding"]["experiment_id"],
        "candidate_identity_sha256": program["experiment_binding"]["candidate_identity_sha256"],
        "policy_version": POLICY_VERSION,
        "parent_revision_sha256": program["model"]["parent_revision_sha256"],
    }, "binding_sha256")
    greenlight = stamp({
        "granted": True,
        "root_manifest_sha256": root["manifest_sha256"],
        "root_binding_sha256": root["binding_sha256"],
        "program_spec_sha256": spec_sha256,
        "parameter_manifest_receipt_sha256": parameter_receipt["receipt_sha256"],
        "teacher_authorization_receipts_sha256": teacher_receipts_sha256,
        "candidate_identity_sha256": program["experiment_binding"]["candidate_identity_sha256"],
        "claim_scope": program["claim_scope"],
        "issued_by": "selftest-authorizer",
        "issued_at": "2026-07-12T00:00:00Z",
        "signer_key_sha256": "d" * 64,
        "signature_sha256": "e" * 64,
    }, "receipt_sha256")
    allowlist = stamp({
        "program_spec_sha256": spec_sha256,
        "operators_adapters_sha256": _operators_adapters_sha256(program),
        "entries": allow_entries,
    }, "allowlist_sha256")
    admission = stamp({
        "admitted": True,
        "root_manifest_sha256": root["manifest_sha256"],
        "program_spec_sha256": spec_sha256,
        "parameter_manifest_receipt_sha256": parameter_receipt["receipt_sha256"],
        "exact_parameter_count": program["model"]["exact_parameter_count"],
        "target_ceilings_sha256": _target_ceilings_sha256(program),
        "compute_budget_sha256": hash_value(program["evaluation_contract"]["test_time_compute_budget"]),
        "operators_adapters_sha256": _operators_adapters_sha256(program),
        "host_identity_sha256": "f" * 64,
        "verifier_source_sha256": "1" * 64,
        "observed_at": "2026-07-12T00:00:00Z",
        "available_memory_bytes": 80_000_000_000,
        "available_disk_bytes": 250_000_000_000,
        "resident_bytes_limit": 60_000_000_000,
        "swap_bytes": 0,
        "requested_peak_resident_bytes": 1_000_000_000,
        "requested_scratch_disk_bytes": 1_000_000_000,
        "memory_pressure": "normal",
        "thermal_state": "nominal",
        "ac_power": True,
    }, "receipt_sha256")
    program["execution_contract"].update({
        "root_manifest": root, "greenlight_receipt": greenlight,
        "adapter_allowlist": allowlist, "resource_admission": admission,
    })
    return stamp(program, "program_sha256")


def _sample_observation(program: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        domain: {
            "candidate": 0.80,
            "parent": 0.80,
            "delta": 0.0,
            "delta_lcb": -0.002,
            "delta_ucb": 0.002,
            "noninferiority_margin": 0.01,
            "n": 1000,
            "raw_p_value": 0.01,
            "holm_adjusted_p_value": 0.02,
            "ci_method": "paired_cluster_bootstrap",
            "evidence_slice_sha256": hash_value({"domain": domain}),
            "hypothesis_id": f"parent_noninferiority:{domain}",
            "hypothesis_direction": "candidate_minus_parent_greater_than_negative_margin",
            "null_boundary": -0.01,
            "noninferiority_pass": True,
        }
        for domain in CAPABILITY_DOMAINS
    }
    comparison = {
        "competitor_id": "same-parent-same-bytes-control",
        "same_parent": True,
        "same_or_lower_physical_bytes": True,
        "same_prompt_and_scorer": True,
        "competitor_artifact_sha256": "2" * 64,
        "evidence_slice_sha256": "3" * 64,
        "competitor_domain_scores": {domain: 0.79 for domain in CAPABILITY_DOMAINS},
        "domain_deltas": {domain: 0.01 for domain in CAPABILITY_DOMAINS},
        "domain_delta_lcbs": {domain: 0.001 for domain in CAPABILITY_DOMAINS},
        "domain_raw_p_values": {domain: 0.01 for domain in CAPABILITY_DOMAINS},
        "domain_holm_adjusted_p_values": {domain: 0.02 for domain in CAPABILITY_DOMAINS},
        "domain_hypothesis_ids": {
            domain: f"competitor_superiority:same-parent-same-bytes-control:{domain}"
            for domain in CAPABILITY_DOMAINS
        },
        "domain_superiority_passes": {domain: True for domain in CAPABILITY_DOMAINS},
        "superiority_direction": "candidate_greater_than_competitor",
        "macro_delta": 0.01,
        "macro_delta_lcb": 0.001,
        "macro_raw_p_value": 0.01,
        "macro_holm_adjusted_p_value": 0.02,
        "macro_hypothesis_id": "competitor_macro_superiority:same-parent-same-bytes-control",
        "macro_superiority_pass": True,
        "superiority_delta": 0.0,
    }
    comparison_two = copy.deepcopy(comparison)
    comparison_two.update({
        "competitor_id": "same-parent-second-frozen-control",
        "competitor_artifact_sha256": "4" * 64,
        "evidence_slice_sha256": "5" * 64,
        "competitor_domain_scores": {domain: 0.785 for domain in CAPABILITY_DOMAINS},
        "domain_deltas": {domain: 0.015 for domain in CAPABILITY_DOMAINS},
        "domain_delta_lcbs": {domain: 0.0005 for domain in CAPABILITY_DOMAINS},
        "macro_delta": 0.015,
        "macro_delta_lcb": 0.0005,
        "domain_hypothesis_ids": {
            domain: f"competitor_superiority:same-parent-second-frozen-control:{domain}"
            for domain in CAPABILITY_DOMAINS
        },
        "macro_hypothesis_id": "competitor_macro_superiority:same-parent-second-frozen-control",
    })
    comparisons = [comparison, comparison_two]
    observation = {
        "schema": OBSERVATION_SCHEMA,
        "program_sha256": program["program_sha256"],
        "claim_scope": program["claim_scope"],
        "status": "succeeded",
        "proof_state": "independent_reproduction",
        "evidence_state": "PROVEN",
        "negative_result_retained": True,
        "speed_claimed": False,
        "artifact_manifest_sha256": "a" * 64,
        "physical_artifact_bytes": 1,
        "capability_metrics": metrics,
        "statistical_contract": {
            "paired": True,
            "cluster_resampling": True,
            "multiple_testing_correction": "holm",
            "familywise_alpha": 0.05,
            "confidence": 0.95,
            "preregistered": True,
            "preregistration_sha256": "4" * 64,
        },
        "competitor_comparisons": comparisons,
        "data_firewall_pass": True,
        "parent_parity_protocol_pass": True,
        "test_time_compute_match_pass": True,
        "comparator_set_frozen_before_final": True,
        "sealed_final_consumed_once": True,
        "claim_snapshot": {
            "as_of_date": "2026-07-12",
            "competitor_registry_sha256": program["evaluation_contract"]["competitor_registry_sha256"],
            "quality_battery_sha256": "7" * 64,
            "artifact_sha256": "a" * 64,
            "expires_on_registry_benchmark_artifact_or_runtime_change": True,
        },
        "external_runtime_bytes": 0,
    }
    observation["test_time_compute_receipt"] = stamp({
        "passed": True,
        "budget_sha256": hash_value(program["evaluation_contract"]["test_time_compute_budget"]),
        "participant_manifest_sha256": "5" * 64,
        "measurement_log_sha256": "6" * 64,
    }, "receipt_sha256")
    artifact_receipt = stamp({
        "program_sha256": program["program_sha256"],
        "artifact_sha256": observation["artifact_manifest_sha256"],
        "exact_parameter_count": program["model"]["exact_parameter_count"],
        "all_in_physical_bytes": observation["physical_artifact_bytes"],
        "all_in_bpw": (
            8.0 * observation["physical_artifact_bytes"]
            / program["model"]["exact_parameter_count"]
        ),
        "aggregate_file_manifest_sha256": "7" * 64,
        "aggregate_component_manifest_sha256": "8" * 64,
        "validator_source_sha256": "9" * 64,
        "verify_files": True,
    }, "receipt_sha256")
    raw = {
        "artifact_validation_receipt": artifact_receipt,
        "per_item_outputs_sha256": "7" * 64,
        "cluster_assignments_sha256": "8" * 64,
        "cluster_outputs_sha256": "9" * 64,
        "data_firewall_receipt_sha256": "a" * 64,
        "parent_parity_receipt_sha256": "b" * 64,
        "calibration_draw_results_sha256": "c" * 64,
        "training_seed_results_sha256": "d" * 64,
        "training_seed_count": 5,
        "calibration_draw_count": 5,
    }
    raw_index_payload = {
        field: raw[field] for field in (
            "per_item_outputs_sha256", "cluster_assignments_sha256", "cluster_outputs_sha256",
            "data_firewall_receipt_sha256", "parent_parity_receipt_sha256",
            "calibration_draw_results_sha256",
        )
    } | {
        "artifact_validation_receipt_sha256": artifact_receipt["receipt_sha256"],
        "training_seed_results_sha256": raw["training_seed_results_sha256"],
        "training_seed_count": raw["training_seed_count"],
        "calibration_draw_count": raw["calibration_draw_count"],
    }
    raw["raw_evidence_index_sha256"] = hash_value(raw_index_payload)
    summary_sha256 = hash_value(_summary_payload(observation))
    raw_p_payload = {
        "parent": {domain: metrics[domain]["raw_p_value"] for domain in CAPABILITY_DOMAINS},
        "competitors": {
            row["competitor_id"]: {
                "domains": row["domain_raw_p_values"],
                "macro": row["macro_raw_p_value"],
            }
            for row in comparisons
        },
    }
    corrected_p_payload = {
        "parent": {domain: metrics[domain]["holm_adjusted_p_value"] for domain in CAPABILITY_DOMAINS},
        "competitors": {
            row["competitor_id"]: {
                "domains": row["domain_holm_adjusted_p_values"],
                "macro": row["macro_holm_adjusted_p_value"],
            }
            for row in comparisons
        },
    }
    corrected = stamp({
        "procedure": "holm",
        "familywise_alpha": 0.05,
        "family_size": len(CAPABILITY_DOMAINS) + len(comparisons) * (len(CAPABILITY_DOMAINS) + 1),
        "raw_p_values_sha256": hash_value(raw_p_payload),
        "corrected_p_values_sha256": hash_value(corrected_p_payload),
        "summary_sha256": summary_sha256,
        "all_hypotheses_corrected": True,
    }, "receipt_sha256")
    verifier = stamp({
        "raw_evidence_index_sha256": raw["raw_evidence_index_sha256"],
        "summary_sha256": summary_sha256,
        "corrected_test_receipt_sha256": corrected["receipt_sha256"],
        "verifier_source_sha256": "e" * 64,
        "owner_id": "selftest-independent-owner",
        "recomputed_deltas_cis_pvalues_from_raw": True,
        "passed": True,
    }, "receipt_sha256")
    sealed = stamp({
        "service_id": "selftest-sealed-service",
        "owner_id": "selftest-sealed-owner",
        "sealed_final_manifest_sha256": program["data_contract"]["splits"]["sealed_final"]["manifest_sha256"],
        "program_sha256": program["program_sha256"],
        "artifact_sha256": observation["artifact_manifest_sha256"],
        "raw_evidence_index_sha256": raw["raw_evidence_index_sha256"],
        "execution_receipt_sha256": "f" * 64,
        "one_time_nonce_sha256": "1" * 64,
        "consumed_once": True,
    }, "attestation_sha256")
    independent = stamp({
        "owner_id": "selftest-independent-owner",
        "program_sha256": program["program_sha256"],
        "artifact_sha256": observation["artifact_manifest_sha256"],
        "raw_evidence_index_sha256": raw["raw_evidence_index_sha256"],
        "summary_sha256": summary_sha256,
        "sealed_attestation_sha256": sealed["attestation_sha256"],
        "replication_receipt_sha256": "2" * 64,
        "independently_executed": True,
        "no_shared_runtime_owner": True,
    }, "attestation_sha256")
    raw["corrected_test_receipt"] = corrected
    raw["evidence_verifier_receipt"] = verifier
    raw["sealed_service_attestation"] = sealed
    raw["independent_owner_attestation"] = independent
    observation["evidence_bundle"] = raw
    return stamp(observation, "observation_sha256")


def _sample_trust_context(
    program: dict[str, Any], observation: dict[str, Any] | None = None
) -> dict[str, set[str]]:
    execution = program["execution_contract"]
    context = {role: set() for role in TRUST_CONTEXT_ROLES}
    context["package_root_manifest"].add(program["package_root_manifest_sha256"])
    context["root_manifest"].add(execution["root_manifest"]["binding_sha256"])
    context["greenlight_receipt"].add(execution["greenlight_receipt"]["receipt_sha256"])
    context["adapter_allowlist"].add(execution["adapter_allowlist"]["allowlist_sha256"])
    context["resource_admission_receipt"].add(
        execution["resource_admission"]["receipt_sha256"]
    )
    context["parameter_manifest_receipt"].add(
        program["model"]["parameter_manifest_receipt"]["receipt_sha256"]
    )
    teachers = program["training_contract"]["teacher_contract"]
    context["parent_teacher_authorization"].add(
        teachers["parent_teacher"]["authorization_receipt"]["receipt_sha256"]
    )
    if teachers["elevation_teacher"] is not None:
        context["elevation_teacher_authorization"].add(
            teachers["elevation_teacher"]["authorization_receipt"]["receipt_sha256"]
        )
    if observation is not None:
        evidence = observation["evidence_bundle"]
        context["artifact_validation_receipt"].add(
            evidence["artifact_validation_receipt"]["receipt_sha256"]
        )
        context["evidence_verifier_receipt"].add(
            evidence["evidence_verifier_receipt"]["receipt_sha256"]
        )
        context["sealed_service_attestation"].add(
            evidence["sealed_service_attestation"]["attestation_sha256"]
        )
        context["independent_owner_attestation"].add(
            evidence["independent_owner_attestation"]["attestation_sha256"]
        )
    return context


def _sample_artifact(program: dict[str, Any]) -> dict[str, Any]:
    ledger = {component: 0 for component in ARTIFACT_COMPONENTS}
    ledger["base"] = 1
    exact_parameters = program["model"].get("exact_parameter_count")
    if not isinstance(exact_parameters, int):
        exact_parameters = int(float(program["model"]["params_b"]) * 1_000_000_000)
    files = [{"path": "artifact.bin", "component": "base", "sha256": "5" * 64, "bytes": 1}]
    component_manifests = {
        component: {
            "bytes": sum(row["bytes"] for row in _component_file_payload(files, component)),
            "file_count": len(_component_file_payload(files, component)),
            "files_sha256": hash_value(_component_file_payload(files, component)),
        }
        for component in ARTIFACT_COMPONENTS
    }
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "program_sha256": program["program_sha256"],
        "claim_scope": program["claim_scope"],
        "parent_revision_sha256": "1" * 64,
        "packed_semantics_sha256": "2" * 64,
        "decoder_semantics_sha256": "3" * 64,
        "tensor_ownership_sha256": "4" * 64,
        "runtime_abi_sha256": "6" * 64,
        "exact_parameter_count": exact_parameters,
        "dense_parent_fallback": False,
        "all_tensor_ownership_complete": True,
        "files": files,
        "byte_ledger": ledger,
        "component_manifests": component_manifests,
        "physical_accounting": {
            "stored_artifact_bytes": 1,
            "payload_bytes": 1,
            "decoder_runtime_bytes": 0,
            "runtime_dependency_bytes": 0,
            "metadata_bytes": 0,
            "context_state_bytes": 0,
            "external_system_bytes": 0,
        },
        "all_in_physical_bytes": 1,
        "all_in_bpw": 8.0 / exact_parameters,
        "payload_bpw": 8.0 / exact_parameters,
        "decoded_resident_bytes": 1,
        "resident_peak_bytes": 1,
        "expected_active_bytes": 1,
        "worst_active_bytes": 1,
        "peak_bytes_by_context": {"4096": 1},
        "runtime_accounting": {
            "decoder_resident_bytes": 1,
            "runtime_dependency_resident_bytes": 0,
            "metadata_resident_bytes": 0,
            "persistent_state_resident_bytes": 0,
            "context_state_bytes_by_context": {"4096": 0},
        },
    }
    if program["model"].get("active_b") is not None:
        active_parameters = int(float(program["model"]["active_b"]) * 1_000_000_000)
        artifact["moe_accounting"] = {
            "expert_count": 2,
            "routed_experts_per_token_expected": 1,
            "routed_experts_per_token_max": 2,
            "total_parameter_count": exact_parameters,
            "expected_active_parameter_count": active_parameters,
            "worst_active_parameter_count": min(exact_parameters, active_parameters * 2),
            "total_installed_expert_bytes": 1,
            "expected_active_expert_bytes": 1,
            "worst_active_expert_bytes": 1,
        }
    return stamp(artifact, "artifact_sha256")


def selftest() -> int:
    identity = "b" * 64
    operators = [
        _sample_operator("diagnose", "diagnose", []),
        _sample_operator("represent", "represent", ["diagnose"]),
        _sample_operator("reconstruct", "reconstruct", ["represent"]),
        _sample_operator("repair", "repair_static", ["reconstruct"]),
        _sample_operator("package", "package", ["repair"]),
        _sample_operator("evaluate", "evaluate", ["package"]),
    ]
    program = planned_program(
        experiment_id="dv5-selftest",
        candidate_identity_sha256=identity,
        model_id="selftest-7b",
        params_b=7.0,
        active_b=None,
        target_bpw=0.5,
        claim_scope="restorative_training",
        failure_class="computation_collapse",
        operators=operators,
    )
    assert validate_program(program) == []
    rooted_planned = copy.deepcopy(program)
    rooted_planned["package_root_schema"] = PACKAGE_ROOT_SCHEMA
    rooted_planned["package_root_manifest_sha256"] = "c" * 64
    rooted_planned["program_spec_sha256"] = compute_program_spec_sha256(rooted_planned)
    rooted_planned = stamp(rooted_planned, "program_sha256")
    assert validate_program(
        rooted_planned, expected_package_root_manifest_sha256="c" * 64
    ) == []
    assert any("expected/trusted package root" in error for error in validate_program(
        rooted_planned, expected_package_root_manifest_sha256="d" * 64
    ))
    artifact = _sample_artifact(program)
    assert validate_artifact(artifact, program) == []
    executable = _sample_executable(program)
    program_trust = _sample_trust_context(executable)
    assert any("external trust context" in error for error in validate_program(
        executable, allow_planned=False
    ))
    mismatched_trust = {role: {"0" * 64} for role in TRUST_CONTEXT_ROLES}
    assert any("external trust context" in error for error in validate_program(
        executable, allow_planned=False, trust_context=mismatched_trust
    ))
    assert validate_program(
        executable, allow_planned=False, trust_context=program_trust
    ) == []

    program_unknown_injections = [
        ("program", lambda value: value.__setitem__("unsafe_override", True)),
        ("experiment", lambda value: value["experiment_binding"].__setitem__("unsafe_override", True)),
        ("model", lambda value: value["model"].__setitem__("unsafe_override", True)),
        ("target", lambda value: value["target"].__setitem__("unsafe_override", True)),
        ("diagnostic", lambda value: value["diagnostic_contract"].__setitem__("unsafe_override", True)),
        ("operator", lambda value: value["operators"][0].__setitem__("unsafe_override", True)),
        ("executor", lambda value: value["operators"][0]["executor"].__setitem__("unsafe_override", True)),
        ("data", lambda value: value["data_contract"].__setitem__("unsafe_override", True)),
        ("split", lambda value: value["data_contract"]["splits"]["calibration"].__setitem__("unsafe_override", True)),
        ("training", lambda value: value["training_contract"].__setitem__("unsafe_override", True)),
        ("teacher", lambda value: value["training_contract"]["teacher_contract"]["parent_teacher"].__setitem__("unsafe_override", True)),
        ("teacher_receipt", lambda value: value["training_contract"]["teacher_contract"]["parent_teacher"]["authorization_receipt"].__setitem__("unsafe_override", True)),
        ("evaluation", lambda value: value["evaluation_contract"].__setitem__("unsafe_override", True)),
        ("compute", lambda value: value["evaluation_contract"]["test_time_compute_budget"].__setitem__("unsafe_override", True)),
        ("execution", lambda value: value["execution_contract"].__setitem__("unsafe_override", True)),
        ("root_receipt", lambda value: value["execution_contract"]["root_manifest"].__setitem__("unsafe_override", True)),
        ("greenlight", lambda value: value["execution_contract"]["greenlight_receipt"].__setitem__("unsafe_override", True)),
        ("allowlist", lambda value: value["execution_contract"]["adapter_allowlist"].__setitem__("unsafe_override", True)),
        ("allowlist_entry", lambda value: value["execution_contract"]["adapter_allowlist"]["entries"][0].__setitem__("unsafe_override", True)),
        ("resource", lambda value: value["execution_contract"]["resource_admission"].__setitem__("unsafe_override", True)),
        ("resume", lambda value: value["exact_resume_contract"].__setitem__("unsafe_override", True)),
        ("output", lambda value: value["output_contract"].__setitem__("unsafe_override", True)),
        ("campaign_metadata", lambda value: value["campaign_metadata"].__setitem__("unsafe_override", True)),
    ]
    for level, inject in program_unknown_injections:
        hostile = copy.deepcopy(executable)
        inject(hostile)
        hostile["program_spec_sha256"] = compute_program_spec_sha256(hostile)
        hostile = stamp(hostile, "program_sha256")
        hostile_errors = validate_program(hostile, trust_context=program_trust)
        assert any("unsafe_override" in error for error in hostile_errors), level
    semantic_mutation = copy.deepcopy(executable)
    semantic_mutation["target"]["resident_bytes_ceiling"] -= 1
    semantic_mutation["program_spec_sha256"] = compute_program_spec_sha256(semantic_mutation)
    semantic_mutation = stamp(semantic_mutation, "program_sha256")
    assert any("final program spec" in error or "program_spec_sha256" in error for error in validate_program(
        semantic_mutation, allow_planned=False, trust_context=program_trust
    ))
    count_without_provenance = copy.deepcopy(executable)
    count_without_provenance["model"]["parameter_manifest_receipt"] = "required"
    count_without_provenance = stamp(count_without_provenance, "program_sha256")
    assert any("parameter-manifest receipt" in error for error in validate_program(
        count_without_provenance, allow_planned=False, trust_context=program_trust
    ))
    undeclared_teacher = copy.deepcopy(executable)
    undeclared_teacher["training_contract"]["stronger_teacher"] = {"authorized": True}
    undeclared_teacher["program_spec_sha256"] = compute_program_spec_sha256(undeclared_teacher)
    undeclared_teacher = stamp(undeclared_teacher, "program_sha256")
    assert any("stronger_teacher" in error for error in validate_program(
        undeclared_teacher, trust_context=program_trust
    ))
    executable_artifact = _sample_artifact(executable)
    assert validate_artifact(executable_artifact, executable)
    assert validate_artifact(
        executable_artifact, executable, trust_context=program_trust
    ) == []
    artifact_unknown_injections = [
        ("artifact", lambda value: value.__setitem__("unsafe_override", True)),
        ("artifact_file", lambda value: value["files"][0].__setitem__("unsafe_override", True)),
        ("byte_ledger", lambda value: value["byte_ledger"].__setitem__("unsafe_override", 0)),
        ("component_manifest", lambda value: value["component_manifests"]["base"].__setitem__("unsafe_override", True)),
        ("physical_accounting", lambda value: value["physical_accounting"].__setitem__("unsafe_override", True)),
        ("runtime_accounting", lambda value: value["runtime_accounting"].__setitem__("unsafe_override", True)),
    ]
    for level, inject in artifact_unknown_injections:
        hostile = copy.deepcopy(executable_artifact)
        inject(hostile)
        hostile = stamp(hostile, "artifact_sha256")
        hostile_errors = validate_artifact(hostile, executable, trust_context=program_trust)
        assert any("unsafe_override" in error for error in hostile_errors), level
    moe_planned = planned_program(
        experiment_id="dv5-moe-schema-selftest", candidate_identity_sha256="8" * 64,
        model_id="selftest-moe", params_b=7.0, active_b=1.0, target_bpw=0.5,
        claim_scope="restorative_training", failure_class="computation_collapse",
        operators=operators,
    )
    moe_executable = _sample_executable(moe_planned)
    moe_trust = _sample_trust_context(moe_executable)
    moe_artifact = _sample_artifact(moe_executable)
    assert validate_artifact(moe_artifact, moe_executable, trust_context=moe_trust) == []
    hostile_moe = copy.deepcopy(moe_artifact)
    hostile_moe["moe_accounting"]["unsafe_override"] = True
    hostile_moe = stamp(hostile_moe, "artifact_sha256")
    assert any("unsafe_override" in error for error in validate_artifact(
        hostile_moe, moe_executable, trust_context=moe_trust
    ))
    observation = _sample_observation(executable)
    full_trust = _sample_trust_context(executable, observation)
    assert any("external trust context" in error for error in validate_observation(
        observation, executable
    ))
    assert any("external trust context" in error for error in validate_observation(
        observation, executable, trust_context=mismatched_trust
    ))
    assert validate_observation(observation, executable, trust_context=full_trust) == []
    observation_unknown_injections = [
        ("observation", lambda value: value.__setitem__("unsafe_override", True)),
        ("metric", lambda value: value["capability_metrics"]["coding"].__setitem__("unsafe_override", True)),
        ("statistics", lambda value: value["statistical_contract"].__setitem__("unsafe_override", True)),
        ("comparison", lambda value: value["competitor_comparisons"][0].__setitem__("unsafe_override", True)),
        ("compute_receipt", lambda value: value["test_time_compute_receipt"].__setitem__("unsafe_override", True)),
        ("snapshot", lambda value: value["claim_snapshot"].__setitem__("unsafe_override", True)),
        ("evidence", lambda value: value["evidence_bundle"].__setitem__("unsafe_override", True)),
        ("artifact_receipt", lambda value: value["evidence_bundle"]["artifact_validation_receipt"].__setitem__("unsafe_override", True)),
        ("corrected_receipt", lambda value: value["evidence_bundle"]["corrected_test_receipt"].__setitem__("unsafe_override", True)),
        ("verifier_receipt", lambda value: value["evidence_bundle"]["evidence_verifier_receipt"].__setitem__("unsafe_override", True)),
        ("sealed_attestation", lambda value: value["evidence_bundle"]["sealed_service_attestation"].__setitem__("unsafe_override", True)),
        ("independent_attestation", lambda value: value["evidence_bundle"]["independent_owner_attestation"].__setitem__("unsafe_override", True)),
    ]
    for level, inject in observation_unknown_injections:
        hostile = copy.deepcopy(observation)
        inject(hostile)
        hostile = stamp(hostile, "observation_sha256")
        hostile_errors = validate_observation(hostile, executable, trust_context=full_trust)
        assert any("unsafe_override" in error for error in hostile_errors), level
    dominance = dominance_decision(observation, executable, trust_context=full_trust)
    assert validate_dominance(dominance, observation, executable, trust_context=full_trust) == []
    hostile_dominance = copy.deepcopy(dominance)
    hostile_dominance["unsafe_override"] = True
    hostile_dominance["dominance_sha256"] = hash_value(
        _identity_payload(hostile_dominance, "dominance_sha256")
    )
    assert any("unsafe_override" in error for error in validate_dominance(
        hostile_dominance, observation, executable, trust_context=full_trust
    ))
    high_parent_p = copy.deepcopy(observation)
    high_parent_p["capability_metrics"]["coding"]["raw_p_value"] = 0.8
    high_parent_p["capability_metrics"]["coding"]["holm_adjusted_p_value"] = 0.9
    high_parent_p = stamp(high_parent_p, "observation_sha256")
    assert any("CI direction and raw p-value" in error or "noninferiority pass" in error
               for error in validate_observation(high_parent_p, executable, trust_context=full_trust))
    assert dominance_decision(high_parent_p, executable, trust_context=full_trust)["passed"] is False
    high_competitor_p = copy.deepcopy(observation)
    comparison_with_bad_p = high_competitor_p["competitor_comparisons"][0]
    comparison_with_bad_p["domain_raw_p_values"]["coding"] = 0.8
    comparison_with_bad_p["domain_holm_adjusted_p_values"]["coding"] = 0.9
    high_competitor_p = stamp(high_competitor_p, "observation_sha256")
    assert any("CI direction and raw p-value" in error or "superiority pass" in error
               for error in validate_observation(high_competitor_p, executable, trust_context=full_trust))
    assert dominance_decision(high_competitor_p, executable, trust_context=full_trust)["passed"] is False
    artifact_untrusted = {role: set(values) for role, values in full_trust.items()}
    artifact_untrusted["artifact_validation_receipt"].clear()
    assert any("artifact-validation receipt" in error for error in validate_observation(
        observation, executable, trust_context=artifact_untrusted
    ))
    assert dominance_decision(observation, executable)["passed"] is False
    assert dominance_decision(
        observation, executable, trust_context=full_trust
    )["passed"] is True

    missing_competitor = copy.deepcopy(observation)
    missing_competitor["competitor_comparisons"].pop()
    missing_competitor = stamp(missing_competitor, "observation_sha256")
    assert any("competitor set differs" in error for error in validate_observation(
        missing_competitor, executable, trust_context=full_trust
    ))

    underpowered = copy.deepcopy(observation)
    for metric in underpowered["capability_metrics"].values():
        metric["n"] = 1
    underpowered = stamp(underpowered, "observation_sha256")
    assert any("five independent clusters" in error for error in validate_observation(
        underpowered, executable, trust_context=full_trust
    ))

    damaged = copy.deepcopy(executable)
    damaged["operators"] = [node for node in damaged["operators"] if node["kind"] != "reconstruct"]
    next(node for node in damaged["operators"] if node["id"] == "repair")["depends_on"] = ["represent"]
    damaged = stamp(damaged, "program_sha256")
    assert any("collapse requires" in error for error in validate_program(
        damaged, trust_context=program_trust
    ))

    augmented = copy.deepcopy(program)
    augmented["operators"].insert(-2, _sample_operator("retrieval", "augment_external", ["repair"]))
    augmented["operators"][-2]["depends_on"] = ["retrieval"]
    augmented = stamp(augmented, "program_sha256")
    assert any("core claim" in error for error in validate_program(augmented))

    elevation_planned = planned_program(
        experiment_id="dv5-elevation-selftest", candidate_identity_sha256="9" * 64,
        model_id="selftest-7b", params_b=7.0, active_b=None, target_bpw=0.5,
        claim_scope="capability_elevation", failure_class="computation_collapse",
        operators=operators,
    )
    elevation = _sample_executable(elevation_planned)
    elevation_trust = _sample_trust_context(elevation)
    assert validate_program(elevation, trust_context=elevation_trust) == []
    missing_elevation_authority = copy.deepcopy(elevation)
    missing_elevation_authority["training_contract"]["teacher_contract"][
        "elevation_teacher"
    ]["authorization_receipt"] = "required"
    missing_elevation_authority = stamp(missing_elevation_authority, "program_sha256")
    assert any("stronger_training_teacher teacher authorization" in error
               for error in validate_program(missing_elevation_authority, trust_context=elevation_trust))

    weak = copy.deepcopy(observation)
    weak["capability_metrics"]["coding"]["delta_lcb"] = -0.02
    weak = stamp(weak, "observation_sha256")
    assert dominance_decision(weak, executable, trust_context=full_trust)["passed"] is False

    forged = copy.deepcopy(observation)
    forged["evidence_bundle"].pop("independent_owner_attestation")
    forged = stamp(forged, "observation_sha256")
    assert validate_observation(forged, executable, trust_context=full_trust)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "artifact.bin").write_bytes(b"x")
        strict_artifact = _sample_artifact(executable)
        strict_artifact["files"][0]["sha256"] = hashlib.sha256(b"x").hexdigest()
        for component in ARTIFACT_COMPONENTS:
            payload = _component_file_payload(strict_artifact["files"], component)
            strict_artifact["component_manifests"][component] = {
                "bytes": sum(row["bytes"] for row in payload),
                "file_count": len(payload),
                "files_sha256": hash_value(payload),
            }
        strict_artifact = stamp(strict_artifact, "artifact_sha256")
        assert validate_artifact(
            strict_artifact, executable, verify_files=True, base_dir=root,
            trust_context=program_trust,
        ) == []
        (root / "artifact.bin").write_bytes(b"y")
        assert any("actual file bytes" in error for error in validate_artifact(
            strict_artifact, executable, verify_files=True, base_dir=root,
            trust_context=program_trust,
        ))

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "program.json"
        atomic_json(path, program)
        assert validate_program(json.loads(path.read_text())) == []
    print("doctor_v5_contract.py selftest OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    validate = sub.add_parser("validate-program")
    validate.add_argument("path", type=Path)
    validate.add_argument(
        "--trust-context", type=Path,
        help="JSON role-to-hash-list trust context; required for executable programs",
    )
    validate.add_argument(
        "--expected-package-root-manifest-sha256",
        help="exact caller-expected Doctor-v5 package-root identity",
    )
    args = parser.parse_args()
    if args.command == "selftest":
        return selftest()
    value = json.loads(args.path.read_text())
    trust_context = json.loads(args.trust_context.read_text()) if args.trust_context else None
    errors = validate_program(
        value, trust_context=trust_context,
        expected_package_root_manifest_sha256=args.expected_package_root_manifest_sha256,
    )
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
