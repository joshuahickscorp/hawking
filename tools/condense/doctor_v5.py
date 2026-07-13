#!/usr/bin/env python3.12
"""Compile Doctor-v5's capability-first, fail-closed research campaign.

This module is a deterministic research-space compiler.  It is deliberately
stdlib-only and execution-free: it does not load a model, inspect Studio state,
acquire a lease, import a training framework, or launch a subprocess.  Every
mechanism executor and every candidate remain unwired and unlaunchable until a
separate, future greenlight changes the v5 contract and supplies independently
audited adapters.

Commands::

    python doctor_v5.py compile [--output PATH]
    python doctor_v5.py validate [PATH]
    python doctor_v5.py select [PATH] [--limit N]
    python doctor_v5.py materialize CANDIDATE_ID [--campaign PATH] [--output PATH]
    python doctor_v5.py selftest

Only ``compile`` and ``materialize --output`` write files, using atomic JSON
replacement.  ``materialize`` emits a planned program which must pass
``doctor_v5_contract.validate_program``; it never makes that program executable.
"""
from __future__ import annotations

import argparse
import copy
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

import doctor_v5_contract as contract
from ladder import MODELS, WEIGHT_BUDGET


CAMPAIGN_SCHEMA = "hawking.doctor_v5_campaign.v5"
SELECTION_SCHEMA = "hawking.doctor_v5_selection.v5"
CAMPAIGN_VERSION = "doctor-v5.0"
EXPECTED_MODEL_COUNT = 32
DEFAULT_EXPLICIT_COUNT = 32_768
MANDATORY_CONTROLS_PER_LANE = 4
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = ROOT / "reports" / "condense" / "doctor_v5_campaign.json"
QUALITY_BATTERY_REPORT = ROOT / "reports" / "condense" / "quality_battery_v5.json"
TRAINING_LADDER_REPORT = ROOT / "reports" / "condense" / "training_ladder_v5.json"
CAMPAIGN_INPUT_SCHEMA = "hawking.doctor_v5_campaign_inputs.v5"
PROGRAM_SPEC_SCHEMA = contract.PROGRAM_SPEC_SCHEMA
TEACHER_AUTHORITY_SCHEMA = "hawking.doctor_v5_teacher_authority.v5"
PACKAGE_ROOT_SCHEMA = contract.PACKAGE_ROOT_SCHEMA
PARAMETER_MANIFEST_TRUST_ROLE = "exact_parameter_manifest_receipt"
CAMPAIGN_METADATA_SCHEMA = contract.CAMPAIGN_METADATA_SCHEMA

# Kept identical to the canonical v5 training ladder.  Physical all-in bytes,
# not these labels, are authoritative in every comparison.
RATE_POINTS = (4.0, 3.0, 2.0, 1.0, 0.8, 0.55, 0.5, 0.33, 0.25, 0.1)
FAILURE_ROUTES = ("signal_degradation", "computation_collapse")
FAIL_CLOSED_DIAGNOSES = ("mixed_failure", "undetermined", "no_material_damage")
CLAIM_SCOPES = tuple(contract.CLAIM_SCOPES)
EVIDENCE_STAGES = tuple(
    {
        "id": f"F{index}",
        "proof_state": proof_state,
        "initial_evidence_state": "PLANNED",
        "launch_permitted": False,
        "executor_wired": False,
    }
    for index, proof_state in enumerate(contract.PROOF_STATES)
)

CONTROL_TYPES = (
    "untreated_same_rate",
    "scalar_equal_byte",
    "smaller_higher_bit_equal_byte",
    "best_public_same_byte",
)


def _identity_payload(document: dict[str, Any], identity_field: str) -> dict[str, Any]:
    payload = copy.deepcopy(document)
    payload.pop(identity_field, None)
    return payload


def _stamp(document: dict[str, Any], identity_field: str) -> dict[str, Any]:
    output = copy.deepcopy(document)
    output[identity_field] = contract.hash_value(_identity_payload(output, identity_field))
    return output


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": str(model["family"]),
        "name": str(model["name"]),
        "hf_id": str(model["hf_id"]),
        "params_b": float(model["params_b"]),
        "params_b_semantics": "rounded_catalogue_estimate_not_evidence",
        "exact_parameter_count": None,
        "parameter_manifest_sha256": None,
        "parameter_count_status": "source_tensor_manifest_required",
        "active_b": None if model.get("active_b") is None else float(model["active_b"]),
        "priority": int(model["priority"]),
        "keep_f16": model.get("keep_f16"),
        "note": str(model.get("note", "")),
    }


def _source_models() -> list[dict[str, Any]]:
    models = [_normalized_model(model) for model in MODELS]
    if len(models) != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"Doctor v5 requires {EXPECTED_MODEL_COUNT} ladder models; found {len(models)}"
        )
    names = [model["name"] for model in models]
    if len(names) != len(set(names)):
        raise ValueError("ladder.MODELS contains duplicate model names")
    return models


def _parameter_count_binding(model: dict[str, Any]) -> dict[str, Any]:
    """Bind an exact integer parameter count only when the source explicitly proves it.

    ``params_b`` is a planning label, never an exact denominator.  The current
    ladder rows do not carry exact tensor counts, so they deliberately compile
    to an unresolved binding and add launch blockers.  A future source row may
    supply ``exact_parameter_count`` only together with both a SHA-256 source
    manifest that enumerates tensor ownership/shapes and an independently
    trusted receipt attesting that exact manifest and count.
    """
    exact = model.get("exact_parameter_count")
    source_sha256 = model.get(
        "parameter_manifest_sha256", model.get("exact_parameter_count_source_sha256")
    )
    receipt_sha256 = model.get("parameter_manifest_receipt_sha256")
    if exact is None and source_sha256 is None and receipt_sha256 is None:
        payload = {
            "model": str(model["name"]),
            "nominal_params_b": float(model["params_b"]),
            "exact_parameter_count": None,
            "source_manifest_sha256": "required",
            "parameter_manifest_receipt_sha256": "required",
            "parameter_manifest_trust_role": PARAMETER_MANIFEST_TRUST_ROLE,
            "status": "source_manifest_and_trusted_receipt_required",
            "usable_as_bpw_denominator": False,
        }
    else:
        if isinstance(exact, bool) or not isinstance(exact, int) or exact <= 0:
            raise ValueError(
                f"{model['name']}: exact_parameter_count must be a positive integer when supplied"
            )
        if not contract.is_sha256(source_sha256):
            raise ValueError(
                f"{model['name']}: exact parameter count requires a concrete source-manifest SHA-256"
            )
        if not contract.is_sha256(receipt_sha256):
            raise ValueError(
                f"{model['name']}: exact parameter count requires a trusted parameter-manifest receipt"
            )
        payload = {
            "model": str(model["name"]),
            "nominal_params_b": float(model["params_b"]),
            "exact_parameter_count": exact,
            "source_manifest_sha256": source_sha256,
            "parameter_manifest_receipt_sha256": receipt_sha256,
            "parameter_manifest_trust_role": PARAMETER_MANIFEST_TRUST_ROLE,
            "status": "verified_exact",
            "usable_as_bpw_denominator": True,
        }
    return _stamp(payload, "parameter_count_binding_sha256")


def _source_parameter_count_bindings() -> list[dict[str, Any]]:
    bindings = [_parameter_count_binding(model) for model in MODELS]
    names = [row["model"] for row in bindings]
    if len(bindings) != EXPECTED_MODEL_COUNT or len(names) != len(set(names)):
        raise ValueError("parameter-count bindings must exactly cover the canonical model set")
    return bindings


def _physical_byte_ceiling(
    model: dict[str, Any],
    rate: float,
    parameter_binding: dict[str, Any],
) -> tuple[int, str]:
    rate_decimal = Decimal(str(rate))
    if parameter_binding["status"] == "verified_exact":
        parameters = Decimal(parameter_binding["exact_parameter_count"])
        basis = "verified_exact_parameter_count"
    else:
        parameters = Decimal(str(model["params_b"])) * Decimal(1_000_000_000)
        basis = "nominal_params_b_projection_not_evidence"
    ceiling = (parameters * rate_decimal / Decimal(8)).to_integral_value(
        rounding=ROUND_CEILING
    )
    return max(1, int(ceiling)), basis


def _bound_input_report(
    path: Path,
    *,
    expected_schema: str,
    identity_field: str,
    exclude_generated_at: bool,
) -> dict[str, Any]:
    """Return an immutable byte and document-identity binding for one v5 input."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"required v5 input report is unavailable: {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != expected_schema:
        raise ValueError(f"{path}: expected schema {expected_schema}")
    identity = value.get(identity_field)
    payload = copy.deepcopy(value)
    payload.pop(identity_field, None)
    if exclude_generated_at:
        payload.pop("generated_at", None)
    if not contract.is_sha256(identity) or identity != contract.hash_value(payload):
        raise ValueError(f"{path}: {identity_field} is missing or mismatched")
    return {
        "path": str(path.relative_to(ROOT)),
        "schema": expected_schema,
        "identity_field": identity_field,
        "document_identity_sha256": identity,
        "file_sha256": _file_sha256(path),
        "immutable": True,
    }


def _input_report_bindings() -> dict[str, dict[str, Any]]:
    return {
        "quality_battery": _bound_input_report(
            QUALITY_BATTERY_REPORT,
            expected_schema="hawking.quality_battery.v5",
            identity_field="manifest_sha256",
            exclude_generated_at=True,
        ),
        "training_ladder": _bound_input_report(
            TRAINING_LADDER_REPORT,
            expected_schema="hawking.training_ladder.v5",
            identity_field="ladder_sha256",
            exclude_generated_at=False,
        ),
    }


def _mechanism(
    mechanism_id: str,
    operator_kind: str,
    family: str,
    source: str,
    *,
    status: str = "research",
    routes: Sequence[str] = FAILURE_ROUTES,
    scopes: Sequence[str] = CLAIM_SCOPES,
    min_bpw: float = 0.1,
    max_bpw: float = 4.0,
    control: bool = False,
    moe_only: bool = False,
    hypothesis: bool = False,
    note: str = "",
) -> dict[str, Any]:
    """Build one sourced, explicitly unwired mechanism registry row."""
    horizon = {
        "control": "immediate_implementation",
        "measured": "immediate_implementation",
        "prototype": "immediate_implementation",
        "research": "medium_term_research",
        "unimplemented": "long_term_paradigm_shift",
    }[status]
    return {
        "id": mechanism_id,
        "operator_kind": operator_kind,
        "family": family,
        "source": source,
        "implementation_status": status,
        "failure_routes": list(routes),
        "claim_scopes": list(scopes),
        "minimum_bpw": float(min_bpw),
        "maximum_bpw": float(max_bpw),
        "mandatory_control_eligible": bool(control),
        "moe_only": bool(moe_only),
        "hawking_hypothesis": bool(hypothesis),
        "research_horizon": horizon,
        "executor": {"wired": False, "adapter_id": None, "source_sha256": None},
        "cost_contract": {
            "shipped_bytes": "must_measure_actual",
            "resident_bytes": "must_measure_peak_and_steady_state",
            "training_compute": "must_measure_tokens_steps_and_hardware",
            "test_time_compute": "must_match_preregistered_scope_budget",
            "external_bytes_and_calls": "must_bill_or_be_zero",
        },
        "proposal_evaluation": {
            "theoretical_computational_complexity": "derive_before_F1_promotion",
            "expected_bandwidth_reduction": "unmeasured_no_claim_permitted",
            "expected_latency_reduction": "speed_deferred_unmeasured",
            "implementation_difficulty": f"{horizon}_unscored_until_design_review",
            "compatibility": {
                "existing_gpus": "unwired_requires_reference_and_native_receipts",
                "apple_silicon": "unwired_requires_metal_or_cpu_receipt",
                "cuda": "unwired_requires_separate_cuda_receipt",
                "future_specialized_hardware": "backend_neutral_contract_required",
            },
            "interactions": {
                "quantization": "must_ablate_representation_and_treatment_separately",
                "speculative_decoding": "lossless_parity_or_new_quality_identity",
                "distributed_inference": "communication_and_shard_state_must_be_billed",
                "future_architectures": "no_scale_or_architecture_extrapolation",
            },
            "efficiency_objectives": [
                "capability_per_joule",
                "capability_per_byte_transferred",
                "capability_per_parameter",
                "capability_per_wall_clock_second",
            ],
        },
        "note": note,
    }


def _rows(
    operator_kind: str,
    family: str,
    specs: Sequence[tuple[str, str]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mechanism_id, source in specs:
        row_kwargs = dict(kwargs)
        if source.startswith("local:control/"):
            row_kwargs["status"] = "control"
        rows.append(_mechanism(
            mechanism_id,
            operator_kind,
            family,
            source,
            control=source.startswith("local:control/"),
            hypothesis=source.startswith("local:hawking-v5/"),
            **row_kwargs,
        ))
    return rows


def mechanism_registry() -> list[dict[str, Any]]:
    """Return the canonical v5 mechanism, competitor, and control registry.

    External sources are research/control targets, not imported Hawking evidence.
    ``local:`` rows are either explicit controls or falsifiable Hawking hypotheses.
    No row contains a runnable adapter.
    """
    registry: list[dict[str, Any]] = []

    registry += _rows("diagnose", "diagnostic", (
        ("diag_weight_error_spectrum", "local:hawking-v5/weight-error-spectrum"),
        ("diag_activation_hessian", "https://arxiv.org/abs/2210.17323"),
        ("diag_early_signal_survival", "local:hawking-v5/early-signal-survival"),
        ("diag_internal_geometry_cka", "https://arxiv.org/abs/2606.11244"),
        ("diag_hidden_logit_tail_kl", "local:hawking-v5/hidden-logit-tail-kl"),
        ("diag_capability_failure_clusters", "https://arxiv.org/abs/2501.03035"),
        ("diag_router_expert_drift", "https://arxiv.org/abs/2310.16795"),
        ("diag_long_context_accumulation", "https://arxiv.org/abs/2404.06654"),
        ("diag_calibration_distribution_shift", "https://arxiv.org/abs/2406.12928"),
        ("diag_margin_conditioned_token_flips", "local:hawking-v5/margin-flip-atlas"),
        ("diag_token_entropy_risk", "https://arxiv.org/abs/2606.11244"),
        ("diag_causal_activation_patching", "local:hawking-v5/causal-activation-patching"),
        ("diag_spectral_energy_gain", "https://arxiv.org/abs/2603.00042"),
        ("diag_outlier_topology", "https://arxiv.org/abs/2306.03078"),
        ("diag_prompt_sensitivity", "https://arxiv.org/abs/2502.06065"),
        ("diag_semantic_contamination", "https://arxiv.org/abs/2311.04850"),
        ("diag_teacher_regurgitation_probe", "local:hawking-v5/teacher-regurgitation"),
        ("diag_all_in_byte_audit", "local:hawking-v5/all-in-byte-auditor"),
        ("diag_moe_load_balance", "https://arxiv.org/abs/2505.03804"),
        ("diag_gradient_conflict", "local:hawking-v5/gradient-conflict-tribunal"),
    ), status="prototype")

    registry += _rows("transform", "preconditioner", (
        ("transform_identity", "local:control/identity-transform"),
        ("transform_awq_activation_scaling", "https://arxiv.org/abs/2306.00978"),
        ("transform_smoothquant_migration", "https://arxiv.org/abs/2211.10438"),
        ("transform_hadamard_incoherence", "https://arxiv.org/abs/2402.04396"),
        ("transform_learned_rotation", "https://arxiv.org/abs/2405.16406"),
        ("transform_orthogonal_kronecker", "https://arxiv.org/abs/2605.00422"),
        ("transform_latent_geometry_alignment", "https://arxiv.org/abs/2603.00042"),
        ("transform_bidirectional_channel_reorder", "https://arxiv.org/abs/2602.17698"),
        ("transform_outlier_channel_split", "local:hawking-v5/outlier-channel-split"),
        ("transform_block_whitening", "local:hawking-v5/block-whitening"),
        ("transform_householder_chain", "local:hawking-v5/householder-chain"),
        ("transform_residual_subspace_rotation", "local:hawking-v5/residual-subspace-rotation"),
        ("transform_expert_basis_alignment", "local:hawking-v5/expert-basis-alignment"),
        ("transform_router_preserving_permutation", "local:hawking-v5/router-preserving-permutation"),
        ("transform_heavy_tail_gaussianization", "local:hawking-v5/heavy-tail-gaussianization"),
        ("transform_token_conditioned_calibration_mix", "local:hawking-v5/token-conditioned-calibration"),
    ), status="research")
    identity_transform = next(row for row in registry if row["id"] == "transform_identity")
    identity_transform["implementation_status"] = "control"
    identity_transform["research_horizon"] = "immediate_implementation"
    identity_transform["proposal_evaluation"]["implementation_difficulty"] = (
        "immediate_implementation_unscored_until_design_review"
    )

    # Representation competitors and original hypotheses.  Meta-controls are
    # deliberately valid at every rate because they name a comparison arm, not
    # an assertion that a particular codec natively supports that rate.
    for mechanism_id, source, minimum, maximum, status in (
        ("repr_control_untreated_same_rate", "local:control/untreated-same-rate", 0.1, 4.0, "control"),
        ("repr_control_scalar_equal_byte", "local:control/scalar-equal-byte", 0.1, 4.0, "control"),
        ("repr_control_smaller_higher_bit", "local:control/smaller-higher-bit", 0.1, 4.0, "control"),
        ("repr_control_public_frontier", "local:control/best-public-same-byte", 0.1, 4.0, "control"),
        ("repr_fp16_parent", "local:control/full-precision-parent", 4.0, 4.0, "control"),
        ("repr_rtn_scalar", "local:control/round-to-nearest", 2.0, 4.0, "control"),
        ("repr_gptq", "https://arxiv.org/abs/2210.17323", 2.0, 4.0, "research"),
        ("repr_awq", "https://arxiv.org/abs/2306.00978", 2.0, 4.0, "research"),
        ("repr_omniquant", "https://arxiv.org/abs/2308.13137", 2.0, 4.0, "research"),
        ("repr_spqr", "https://arxiv.org/abs/2306.03078", 2.0, 4.0, "research"),
        ("repr_hqq", "https://arxiv.org/abs/2309.15531", 2.0, 4.0, "research"),
        ("repr_squeezellm", "https://arxiv.org/abs/2306.07629", 2.0, 4.0, "research"),
        ("repr_quip_lattice", "https://arxiv.org/abs/2402.04396", 1.5, 4.0, "research"),
        ("repr_aqlm_additive", "https://arxiv.org/abs/2401.06118", 1.5, 4.0, "research"),
        ("repr_vptq_vector", "https://arxiv.org/abs/2409.17066", 1.5, 4.0, "research"),
        ("repr_tesseraq", "https://arxiv.org/abs/2410.19103", 1.5, 4.0, "research"),
        ("repr_paretoq", "https://arxiv.org/abs/2502.02631", 1.0, 4.0, "research"),
        ("repr_lc_qat", "https://arxiv.org/abs/2606.10531", 1.5, 2.0, "research"),
        ("repr_matgptq", "https://arxiv.org/abs/2602.03537", 0.25, 4.0, "research"),
        ("repr_scalebits", "https://arxiv.org/abs/2602.17698", 0.8, 4.0, "research"),
        ("repr_ccq", "https://arxiv.org/abs/2507.07145", 2.0, 3.0, "research"),
        ("repr_onebit", "https://arxiv.org/abs/2402.11295", 1.0, 1.0, "research"),
        ("repr_bitnet_b158", "https://arxiv.org/abs/2402.17764", 1.0, 2.0, "research"),
        ("repr_billm", "https://arxiv.org/abs/2402.04291", 1.0, 1.34, "research"),
        ("repr_pbllm", "https://arxiv.org/abs/2310.00034", 1.0, 2.0, "research"),
        ("repr_bitdistiller", "https://aclanthology.org/2024.acl-long.7/", 1.0, 3.0, "research"),
        ("repr_stbllm", "https://arxiv.org/abs/2408.01803", 0.25, 1.0, "research"),
        ("repr_btc_llm", "https://arxiv.org/abs/2506.12040", 0.55, 1.34, "research"),
        ("repr_littlebit", "https://arxiv.org/abs/2506.13771", 0.1, 1.0, "research"),
        ("repr_littlebit2", "https://arxiv.org/abs/2603.00042", 0.1, 1.0, "research"),
        ("repr_nanoquant", "https://arxiv.org/abs/2602.06694", 0.1, 1.0, "research"),
        ("repr_dbf", "https://arxiv.org/abs/2505.11076", 1.0, 2.0, "research"),
        ("repr_multi_envelope_dbf", "https://arxiv.org/abs/2512.24545", 1.0, 1.5, "research"),
        ("repr_bwla", "https://arxiv.org/abs/2605.00422", 0.8, 1.34, "research"),
        ("repr_bitstack", "https://arxiv.org/abs/2410.23918", 0.25, 4.0, "research"),
        ("repr_matquant", "https://arxiv.org/abs/2502.06786", 1.5, 4.0, "research"),
        ("repr_qbb_binary_bases", "https://proceedings.neurips.cc/paper_files/paper/2024/file/05b69cc4c8ff6e24c5de1ecd27223d37-Paper-Conference.pdf", 1.0, 4.0, "research"),
        ("repr_shared_parameter_grammar", "local:hawking-v5/shared-parameter-grammar", 0.1, 2.0, "unimplemented"),
        ("repr_cross_layer_dictionary", "local:hawking-v5/cross-layer-dictionary", 0.1, 2.0, "unimplemented"),
        ("repr_rate_distortion_trellis", "local:hawking-v5/rate-distortion-trellis", 0.1, 4.0, "unimplemented"),
        ("repr_binary_latent_program", "local:hawking-v5/binary-latent-program", 0.1, 1.0, "unimplemented"),
        ("repr_on_demand_weight_synthesis", "local:hawking-v5/on-demand-weight-synthesis", 0.1, 1.0, "unimplemented"),
        ("repr_entropy_coded_tiles", "local:hawking-v5/entropy-coded-compute-tiles", 0.1, 3.0, "unimplemented"),
    ):
        registry.append(_mechanism(
            mechanism_id,
            "represent",
            "representation",
            source,
            status=status,
            min_bpw=minimum,
            max_bpw=maximum,
            control=status == "control",
            hypothesis=source.startswith("local:hawking"),
        ))
    registry.append(_mechanism(
        "repr_qmoe_subbit", "represent", "representation", "https://arxiv.org/abs/2310.16795",
        status="research", min_bpw=0.5, max_bpw=1.34, moe_only=True,
    ))
    registry.append(_mechanism(
        "repr_cross_expert_grammar", "represent", "representation",
        "local:hawking-v5/cross-expert-parameter-grammar", status="unimplemented",
        min_bpw=0.1, max_bpw=1.5, moe_only=True, hypothesis=True,
    ))

    registry += _rows("reconstruct", "structural_reconstruction", (
        ("recon_representation_reset", "local:control/representation-reset"),
        ("recon_block_output_match", "local:hawking-v5/block-output-match"),
        ("recon_shard_transactional", "local:hawking-v5/shard-transactional-reconstruction"),
        ("recon_gptq_intrinsic_lora", "https://arxiv.org/abs/2606.01412"),
        ("recon_preserve_then_quantize", "https://arxiv.org/abs/2602.02001"),
        ("recon_module_adaptive_residual", "https://arxiv.org/abs/2605.17997"),
        ("recon_residual_svd", "local:hawking-v5/residual-svd"),
        ("recon_sparse_plus_low_rank", "https://arxiv.org/abs/2306.03078"),
        ("recon_additive_residual_codebook", "https://arxiv.org/abs/2401.06118"),
        ("recon_alternating_quant_lowrank", "local:hawking-v5/alternating-quant-lowrank"),
        ("recon_progressive_binary_factors", "https://arxiv.org/abs/2602.06694"),
        ("recon_causal_capability_islands", "local:hawking-v5/causal-capability-islands"),
        ("recon_cross_expert_dictionary", "local:hawking-v5/cross-expert-dictionary"),
        ("recon_quant_error_syndrome", "local:hawking-v5/quant-error-syndrome"),
        ("recon_oracle_full_residual", "local:control/oracle-full-residual"),
        ("recon_random_equal_rank", "local:control/random-equal-rank"),
    ), status="research")

    repair_scopes = ("restorative_training", "capability_elevation", "augmented_system")
    registry += _rows("repair_static", "static_repair", (
        ("repair_zero", "local:control/zero-repair"),
        ("repair_output_bias", "local:hawking-v5/output-bias"),
        ("repair_low_rank_error", "https://arxiv.org/abs/2606.01412"),
        ("repair_module_adaptive_residual", "https://arxiv.org/abs/2605.17997"),
        ("repair_exact_parent_distill", "https://aclanthology.org/2024.acl-long.7/"),
        ("repair_targeted_reasoning", "https://arxiv.org/abs/2501.03035"),
        ("repair_codebook_residual", "https://arxiv.org/abs/2401.06118"),
        ("repair_capability_island_static", "local:hawking-v5/capability-island-static"),
        ("repair_cross_expert_static", "local:hawking-v5/cross-expert-static"),
        ("repair_reasoning_skill_bank", "local:hawking-v5/reasoning-skill-bank"),
        ("repair_multilingual_skill_bank", "local:hawking-v5/multilingual-skill-bank"),
        ("repair_long_context_skill_bank", "local:hawking-v5/long-context-skill-bank"),
        ("repair_calibration_temperature", "local:hawking-v5/calibration-temperature"),
        ("repair_random_equal_bytes", "local:control/random-equal-byte-repair"),
    ), status="research", scopes=repair_scopes)
    registry += _rows("repair_conditional", "conditional_repair", (
        ("conditional_spear_token_gate", "https://arxiv.org/abs/2606.11244"),
        ("conditional_quant_syndrome_router", "local:hawking-v5/quant-syndrome-router"),
        ("conditional_capability_adapter_bank", "local:hawking-v5/capability-adapter-bank"),
        ("conditional_uncertainty_precision", "local:hawking-v5/uncertainty-precision-router"),
        ("conditional_progressive_slices", "https://arxiv.org/abs/2602.03537"),
        ("conditional_on_demand_weight_synthesis", "local:hawking-v5/on-demand-weight-synthesis"),
        ("conditional_expert_hot_cold_repair", "local:hawking-v5/expert-hot-cold-repair"),
        ("conditional_early_exit_correction", "local:hawking-v5/early-exit-correction"),
        ("conditional_selective_high_precision", "local:hawking-v5/selective-high-precision"),
        ("conditional_causal_island_activation", "local:hawking-v5/causal-island-activation"),
        ("conditional_zero_gate_control", "local:control/zero-conditional-gate"),
        ("conditional_random_gate_control", "local:control/random-conditional-gate"),
    ), status="research", scopes=repair_scopes)
    spear = next(row for row in registry if row["id"] == "conditional_spear_token_gate")
    spear["minimum_bpw"] = 4.0
    spear["maximum_bpw"] = 4.0
    spear["note"] = "Four-bit precedent only; no sub-bit applicability without new evidence."

    registry += _rows("train", "training", (
        ("train_none_control", "local:control/no-training"),
        ("train_codec_native_qat", "local:hawking-v5/codec-native-qat"),
        ("train_paretoq_progressive", "https://arxiv.org/abs/2502.02631"),
        ("train_lcqat_vector", "https://arxiv.org/abs/2606.10531"),
        ("train_onebit", "https://arxiv.org/abs/2402.11295"),
        ("train_littlebit", "https://arxiv.org/abs/2506.13771"),
        ("train_self_distillation", "https://aclanthology.org/2024.acl-long.7/"),
        ("train_exact_parent_distribution", "local:hawking-v5/exact-parent-distribution"),
        ("train_active_failure_mining", "local:hawking-v5/active-failure-mining"),
        ("train_multi_capability_minimax", "local:hawking-v5/multi-capability-minimax"),
        ("train_gradient_conflict_projection", "local:hawking-v5/gradient-conflict-projection"),
        ("train_parent_good_replay", "local:hawking-v5/parent-good-replay"),
        ("train_verified_reasoning_trajectory", "local:hawking-v5/verified-reasoning-trajectory"),
        ("train_rate_curriculum", "https://arxiv.org/abs/2502.06786"),
        ("train_lbllm_three_stage", "https://arxiv.org/abs/2604.19167"),
    ), status="research")
    registry.append(_mechanism(
        "train_stronger_teacher_elevation", "train", "training",
        "local:hawking-v5/provenance-bound-stronger-teacher", status="unimplemented",
        scopes=("capability_elevation",), hypothesis=True,
    ))

    harden_scopes = repair_scopes
    registry += _rows("harden", "hardening", (
        ("harden_none_control", "local:control/no-hardening"),
        ("harden_prompt_metamorphic", "local:hawking-v5/prompt-metamorphic"),
        ("harden_long_context", "https://arxiv.org/abs/2502.05167"),
        ("harden_multilingual", "https://arxiv.org/abs/2503.10497"),
        ("harden_safety_security", "local:hawking-v5/safety-security-tripwires"),
        ("harden_calibration", "local:hawking-v5/selective-risk-calibration"),
        ("harden_rare_token_tail", "local:hawking-v5/rare-token-tail"),
        ("harden_adversarial_quant_trigger", "local:hawking-v5/adversarial-quant-trigger"),
        ("harden_moe_routing", "local:hawking-v5/moe-routing-hardening"),
        ("harden_structured_output", "https://arxiv.org/abs/2311.07911"),
        ("harden_tool_choice", "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-184.html"),
        ("harden_tail_cvar", "local:hawking-v5/tail-cvar"),
        ("harden_forgetting_replay", "local:hawking-v5/forgetting-replay"),
    ), status="research", scopes=harden_scopes)

    registry += _rows("augment_external", "augmentation", (
        ("augment_external_plane_zero_effect", "local:control/external-plane-zero-effect"),
        ("augment_retrieval_closed_corpus", "https://arxiv.org/abs/2112.04426"),
        ("augment_tool_sandbox", "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-184.html"),
        ("augment_symbolic_math_verifier", "local:hawking-v5/symbolic-math-verifier"),
        ("augment_code_execution_verifier", "https://github.com/livecodebench/livecodebench"),
        ("augment_test_time_search", "local:hawking-v5/test-time-search"),
        ("augment_persistent_external_memory", "local:hawking-v5/persistent-external-memory"),
        ("augment_independent_judge_tribunal", "https://arxiv.org/abs/2406.07791"),
        ("augment_provenance_fact_store", "local:hawking-v5/provenance-fact-store"),
        ("augment_formal_proof_checker", "local:hawking-v5/formal-proof-checker"),
    ), status="research", scopes=("augmented_system",))

    registry += _rows("package", "packaging", (
        ("package_actual_bytes_manifest", "local:control/actual-bytes-manifest"),
        ("package_entropy_archive", "local:hawking-v5/entropy-archive"),
        ("package_tile_aligned_stream", "local:hawking-v5/tile-aligned-stream"),
        ("package_mmap_random_access", "local:hawking-v5/mmap-random-access"),
        ("package_progressive_slices", "https://arxiv.org/abs/2602.03537"),
        ("package_shannon_ans", "https://arxiv.org/abs/2606.15789"),
        ("package_expert_shards", "https://arxiv.org/abs/2310.16795"),
        ("package_apple_layout", "local:hawking-v5/apple-layout"),
        ("package_portable_reference", "local:control/portable-reference-package"),
    ), status="prototype")

    registry += _rows("evaluate", "evaluation", (
        ("eval_paired_document_nll", "local:hawking-v5/paired-document-nll"),
        ("eval_teacher_kl_calibration", "local:hawking-v5/teacher-kl-calibration"),
        ("eval_livebench_frozen", "https://arxiv.org/abs/2406.19314"),
        ("eval_livecodebench_postcutoff", "https://arxiv.org/abs/2403.07974"),
        ("eval_arc_agi2", "https://arxiv.org/abs/2505.11831"),
        ("eval_ifeval", "https://arxiv.org/abs/2311.07911"),
        ("eval_bfcl", "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-184.html"),
        ("eval_ruler_curve", "https://arxiv.org/abs/2404.06654"),
        ("eval_nolima_curve", "https://arxiv.org/abs/2502.05167"),
        ("eval_longbench2", "https://arxiv.org/abs/2412.15204"),
        ("eval_bigcodebench", "https://arxiv.org/abs/2406.15877"),
        ("eval_evalplus", "https://arxiv.org/abs/2305.01210"),
        ("eval_mmluprox", "https://arxiv.org/abs/2503.10497"),
        ("eval_hle_secondary", "https://arxiv.org/abs/2501.14249"),
        ("eval_gsm1k_contamination", "https://arxiv.org/abs/2405.00332"),
        ("eval_mmlu_redux", "https://arxiv.org/abs/2406.04127"),
        ("eval_private_postfreeze", "local:hawking-v5/private-postfreeze-battery"),
        ("eval_paired_cluster_statistics", "https://arxiv.org/abs/2103.03098"),
        ("eval_judge_bias_audit", "https://arxiv.org/abs/2406.07791"),
        ("eval_same_budget_dominance", "local:hawking-v5/same-budget-dominance"),
    ), status="prototype")

    # Deterministic ordering is part of campaign identity.
    registry.sort(key=lambda row: row["id"])
    return registry


def _all_in_byte_contract() -> dict[str, Any]:
    return {
        "authoritative_measure": "actual_standalone_shipped_file_bytes",
        "physical_bpw_formula": "8 * actual_standalone_shipped_file_bytes / exact_parent_parameter_count",
        "nominal_payload_bpw_is_secondary": True,
        "no_amortization_across_deployments": True,
        "no_dense_parent_fallback": True,
        "no_remote_or_post_install_unbilled_download": True,
        "required_billed_components": [
            "quantized_payload",
            "pass_through_tensors",
            "embeddings",
            "lm_head",
            "scales_and_zero_points",
            "codebooks_and_dictionaries",
            "indices_masks_and_outliers",
            "transforms",
            "residuals_adapters_and_healers",
            "routers_drafters_and_verifiers",
            "tokenizer_chat_template_and_metadata",
            "retrieval_indexes_and_persistent_state",
            "decoder_required_alignment_and_padding",
        ],
        "required_reported_measures": [
            "actual_archive_bytes",
            "actual_installed_bytes",
            "decoded_steady_state_bytes",
            "peak_resident_bytes_by_context",
            "bytes_read_per_token_mean_p95_worst",
            "active_bytes_mean_p95_worst",
            "total_and_active_moe_bytes",
            "external_runtime_bytes",
        ],
        "oom_timeout_and_unloadable_artifact_count_as_failures": True,
        "same_byte_comparator_uses_no_greater_actual_bytes": True,
        "smaller_higher_bit_control_required": True,
    }


def _matched_compute_budget(scope: str) -> dict[str, Any]:
    """Return the exact lane budget that candidate identity must bind."""
    if scope not in CLAIM_SCOPES:
        raise ValueError(f"unknown claim scope for compute budget: {scope}")
    core_budget = {
        "max_input_tokens": 131_072,
        "max_output_tokens": 32_768,
        "max_reasoning_tokens": 32_768,
        "samples_per_item": 1,
        "temperature": 0.0,
        "timeout_ms": 1_800_000,
        "verifier_retries": 0,
        "retrieval_calls": 0,
        "tool_calls": 0,
        "external_model_calls": 0,
    }
    augmented_budget = dict(core_budget)
    augmented_budget.update({
        "verifier_retries": 2,
        "retrieval_calls": 4,
        "tool_calls": 8,
    })
    return augmented_budget if scope == "augmented_system" else core_budget


def _test_time_compute_contract() -> dict[str, Any]:
    return {
        "exact_dimensions": sorted(contract.MATCHED_COMPUTE_FIELDS),
        "matched_within_claim_scope": True,
        "same_prompt_tokenizer_stopping_and_sampling": True,
        "quality_vs_compute_curve_required": True,
        "unlimited_reasoning_or_retries_forbidden": True,
        "core_budget": _matched_compute_budget("restorative_training"),
        "augmented_budget": _matched_compute_budget("augmented_system"),
        "speed_deferred_but_compute_budget_not_deferred": True,
    }


def _teacher_authority_contract(
    scope: str,
    *,
    parent_teacher_identity_sha256: str = "required",
) -> dict[str, Any]:
    """Return the immutable role-specific teacher authority for one claim scope.

    A training teacher is never inferred from arbitrary extra fields.  The
    identity parent is the only permitted restoration teacher.  A stronger
    teacher is a separate, explicitly authorized capability-elevation role and
    cannot be relabeled as restoration or augmentation.
    """
    if scope not in CLAIM_SCOPES:
        raise ValueError(f"unknown claim scope for teacher authority: {scope}")
    if parent_teacher_identity_sha256 != "required" \
            and not contract.is_sha256(parent_teacher_identity_sha256):
        raise ValueError("parent teacher identity must be required or a SHA-256 identity")
    elevation = scope == "capability_elevation"
    planned_teacher = {
        "identity_sha256": parent_teacher_identity_sha256,
        "revision_sha256": "required",
        "role": "exact_identity_parent",
        "output_protocol_sha256": "required",
        "cache_manifest_sha256": "required",
        "split_manifest_sha256": "required",
        "provenance_manifest_sha256": "required",
        "training_only": True,
        "authorization_receipt": "required",
    }
    elevation_teacher = None
    if elevation:
        elevation_teacher = copy.deepcopy(planned_teacher)
        elevation_teacher["identity_sha256"] = "required"
        elevation_teacher["role"] = "stronger_training_teacher"
    authority = {
        "schema": TEACHER_AUTHORITY_SCHEMA,
        "claim_scope": scope,
        "teacher_contract": {
            "parent_teacher": planned_teacher,
            "elevation_teacher": elevation_teacher,
            "no_teacher_at_inference": True,
        },
        "stronger_teacher_status": (
            "required_before_execution" if elevation else "forbidden"
        ),
        "authorization_trust_roles": {
            "parent_teacher": "parent_teacher_authorization",
            "elevation_teacher": (
                "elevation_teacher_authorization" if elevation else None
            ),
        },
        "external_augmentation_is_not_teacher_authority": scope == "augmented_system",
        "unrecognized_teacher_fields_forbidden": True,
    }
    return _stamp(authority, "teacher_authority_sha256")


def _teacher_authority_contracts() -> list[dict[str, Any]]:
    return [_teacher_authority_contract(scope) for scope in CLAIM_SCOPES]


def _data_firewall_contract() -> dict[str, Any]:
    splits: dict[str, dict[str, Any]] = {}
    optimizer_forbidden = {"shadow", "frozen_final", "sealed_final", "independent_replication"}
    for split in contract.DATA_SPLITS:
        splits[split] = {
            "manifest_sha256": "required",
            "optimizer_access": split not in optimizer_forbidden,
            "exact_near_and_semantic_dedup": "required",
            "teacher_cache_split_bound": True,
        }
    return {
        "splits": splits,
        "public_benchmarks_are_development_only": True,
        "sealed_final_created_or_revealed_after_candidate_freeze": True,
        "sealed_final_consumed_once_per_candidate_lineage": True,
        "doctor_receives_no_item_level_final_feedback": True,
        "pass_fail_retries_on_final_forbidden": True,
        "retrieval_index_excludes_all_evaluation_splits": True,
        "semantic_contamination_scan_required": True,
        "paraphrase_and_translation_overlap_scan_required": True,
        "contamination_or_split_violation_invalidates_lineage": True,
    }


def _statistical_contract() -> dict[str, Any]:
    return {
        "preregistered": True,
        "paired": True,
        "confidence": 0.95,
        "familywise_alpha": 0.05,
        "multiple_testing_correction": "holm",
        "minimum_independent_quantization_or_training_seeds": 5,
        "minimum_calibration_draws": 5,
        "binary_test": "exact_mcnemar",
        "continuous_test": "paired_stratified_cluster_bootstrap",
        "language_modeling_cluster_unit": "document",
        "code_cluster_unit": "task",
        "generation_cluster_unit": "item_and_seed",
        "model_family_is_transfer_unit": True,
        "power_analysis_from_paired_discordance_required": True,
        "inconclusive_if_margin_is_underpowered": True,
        "worst_domain_and_cvar_primary": True,
        "macro_average_secondary_only": True,
        "same_budget_upper_competitor_envelope_required": True,
        "no_unmeasured_interpolation_or_scale_extrapolation": True,
        "independent_reproduction_required_for_dominance": True,
    }


def _scope_contracts() -> list[dict[str, Any]]:
    return [
        {
            "id": "codec_fidelity",
            "claim": "representation fidelity only; no Doctor repair, hardening, or external runtime",
            "allowed_operator_kinds": [
                "diagnose", "transform", "represent", "reconstruct", "train", "package", "evaluate",
            ],
            "stronger_teacher_allowed": False,
            "external_runtime_allowed": False,
            "beyond_parent_gain_label": "not_applicable_to_codec_claim",
        },
        {
            "id": "restorative_training",
            "claim": "standalone artifact restoration against its exact identity parent",
            "allowed_operator_kinds": [
                "diagnose", "transform", "represent", "reconstruct", "repair_static",
                "repair_conditional", "train", "harden", "package", "evaluate",
            ],
            "stronger_teacher_allowed": False,
            "external_runtime_allowed": False,
            "beyond_parent_gain_label": "cannot_be_called_restoration",
        },
        {
            "id": "capability_elevation",
            "claim": "standalone artifact enhancement with a provenance-bound stronger training teacher",
            "allowed_operator_kinds": [
                "diagnose", "transform", "represent", "reconstruct", "repair_static",
                "repair_conditional", "train", "harden", "package", "evaluate",
            ],
            "stronger_teacher_allowed": True,
            "external_runtime_allowed": False,
            "beyond_parent_gain_label": "enhancement_not_restoration",
        },
        {
            "id": "augmented_system",
            "claim": "fully billed model plus retrieval, tools, verifier, or external state",
            "allowed_operator_kinds": list(contract.OPERATOR_KINDS),
            "stronger_teacher_allowed": False,
            "external_runtime_allowed": True,
            "beyond_parent_gain_label": "augmented_system_gain_only",
        },
    ]


def _failure_routing_contract() -> list[dict[str, Any]]:
    return [
        {
            "id": "signal_degradation",
            "materialization_permitted": True,
            "required_next_class": "signal_degradation",
            "definition": "parent capability remains causally recoverable from noisy or distorted internal signal",
            "required_probes": [
                "early_signal_survival", "internal_geometry", "capability_failures",
                "hidden_logit_divergence", "activation_or_weight_patch_recovery",
            ],
            "admissible_treatments": [
                "preconditioning", "representation_refinement", "static_repair",
                "conditional_repair", "distillation", "hardening",
            ],
            "compensation_only_may_be_tested": True,
            "structural_reconstruction_required": False,
        },
        {
            "id": "computation_collapse",
            "materialization_permitted": True,
            "required_next_class": "computation_collapse",
            "definition": "the condensed forward computation no longer implements the required circuit or representation",
            "required_probes": [
                "early_signal_survival", "internal_geometry", "capability_failures",
                "causal_patch_failure", "representation_phase_transition",
            ],
            "admissible_treatments": [
                "representation_reset", "structural_reconstruction", "codec_native_qat",
                "binary_or_codebook_relearning", "then_optional_repair",
            ],
            "compensation_only_may_be_tested": False,
            "structural_reconstruction_required": True,
        },
        {
            "id": "mixed_failure",
            "materialization_permitted": False,
            "required_next_class": "computation_collapse",
            "definition": "both structural collapse and residual signal degradation are present",
            "required_probes": [
                "early_signal_survival", "internal_geometry", "capability_failures",
                "causal_patch_failure", "representation_phase_transition",
            ],
            "admissible_treatments": [],
            "compensation_only_may_be_tested": False,
            "structural_reconstruction_required": True,
            "fail_closed_action": "materialize_computation_collapse_first_then_rediagnose",
        },
        {
            "id": "undetermined",
            "materialization_permitted": False,
            "required_next_class": None,
            "definition": "available probes do not support a treatment-bearing diagnosis",
            "required_probes": [
                "early_signal_survival", "internal_geometry", "capability_failures",
                "additional_causal_probe_or_quarantine",
            ],
            "admissible_treatments": [],
            "compensation_only_may_be_tested": False,
            "structural_reconstruction_required": False,
            "fail_closed_action": "diagnosis_only_no_treatment_materialization",
        },
        {
            "id": "no_material_damage",
            "materialization_permitted": False,
            "required_next_class": None,
            "definition": "candidate is within preregistered damage margins on diagnostic evidence",
            "required_probes": [
                "early_signal_survival", "internal_geometry", "capability_failures",
                "zero_treatment_confirmation",
            ],
            "admissible_treatments": [],
            "compensation_only_may_be_tested": False,
            "structural_reconstruction_required": False,
            "fail_closed_action": "retain_zero_treatment_control_no_doctor_treatment",
        },
    ]


def _rate_profiles() -> list[dict[str, Any]]:
    roles = {
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
    }
    return [
        {
            "physical_bpw_ceiling": rate,
            "role": roles[rate],
            "all_in_not_payload_only": True,
            "representation_reset_arm_required": rate <= 2.0,
            "both_failure_routes_required": True,
        }
        for rate in RATE_POINTS
    ]


def _quality_policy() -> dict[str, Any]:
    return {
        "quality_first": True,
        "speed_deferred": True,
        "wall_clock_cutoff": None,
        "perplexity_cannot_substitute_for_capability": True,
        "average_cannot_mask_domain_regression": True,
        "unbeatable_claim_before_independent_reproduction_forbidden": True,
        "negative_results_retained": True,
    }


def _mandatory_control_policy() -> dict[str, Any]:
    return {
        "control_types": list(CONTROL_TYPES),
        "controls_per_model_rate_scope_failure_lane": MANDATORY_CONTROLS_PER_LANE,
        "exactly_one_each_control_type_per_lane": True,
        "exact_control_mechanism_topology_required": True,
        "control_mechanisms_must_be_rate_applicable": True,
        "same_parent": True,
        "same_or_lower_actual_physical_bytes": True,
        "same_prompt_scorer_and_test_time_compute": True,
        "official_reproduction_or_unreproduced_label": True,
        "smaller_higher_bit_required": True,
        "bf16_same_treatment_required_outside_codec_scope": True,
    }


def _direct_competitor_requirements(
    registry: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Bind the exact named battery registry, never a family proxy."""
    quality = json.loads(QUALITY_BATTERY_REPORT.read_text(encoding="utf-8"))
    coverage = quality.get("competitor_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("quality battery has no named competitor coverage registry")
    direct_rows = coverage.get("direct_implementations")
    if not isinstance(direct_rows, list) or not direct_rows:
        raise ValueError("quality battery direct competitor registry is empty")
    implementation_ids = [row.get("implementation_id") for row in direct_rows]
    if any(not isinstance(value, str) or not value for value in implementation_ids) \
            or len(implementation_ids) != len(set(implementation_ids)):
        raise ValueError("quality battery competitor implementation IDs are missing or duplicated")
    if set(coverage.get("ladder_rates_bpw", [])) != set(RATE_POINTS):
        raise ValueError("quality battery competitor registry does not cover the Doctor rate ladder")
    if coverage.get("registry_sha256") != "required":
        raise ValueError("planned quality battery must leave its evidence registry receipt unresolved")
    coverage_sha256 = contract.hash_value(coverage)

    crosswalk: list[dict[str, Any]] = []
    for comparator in direct_rows:
        source = comparator["primary_source"]
        matches = [row["id"] for row in registry if row["source"] == source]
        if comparator["implementation_id"] == "hawking_prior_source_bound_champion":
            matches = ["repr_control_public_frontier"]
        if not matches:
            raise ValueError(
                f"named competitor {comparator['implementation_id']} has no campaign mechanism"
            )
        crosswalk.append({
            "implementation_id": comparator["implementation_id"],
            "primary_source": source,
            "mechanism_ids": sorted(matches),
            "required_ladder_rates_bpw": list(comparator["required_ladder_rates_bpw"]),
            "applicable_claim_scopes": list(comparator["applicable_claim_scopes"]),
            "applicable_model_classes": list(comparator["applicable_model_classes"]),
        })
    return {
        "registry_mode": "quality_battery_named_direct_implementations",
        "quality_battery_schema": quality.get("schema"),
        "quality_battery_manifest_sha256": quality.get("manifest_sha256"),
        "quality_battery_competitor_coverage_sha256": coverage_sha256,
        "competitor_coverage": copy.deepcopy(coverage),
        "mechanism_crosswalk": crosswalk,
        "required_methods": copy.deepcopy(direct_rows),
    }


def _execution_policy(parameter_bindings: Sequence[dict[str, Any]]) -> dict[str, Any]:
    resolved = sum(row["status"] == "verified_exact" for row in parameter_bindings)
    return {
        "planner_only": True,
        "loads_models": False,
        "launches_processes": False,
        "imports_training_framework": False,
        "reads_or_mutates_live_state": False,
        "acquires_heavy_lease": False,
        "greenlight_recorded": False,
        "all_candidates_launchable": False,
        "all_executors_wired": False,
        "exact_parameter_counts_resolved": resolved,
        "exact_parameter_counts_required": len(parameter_bindings),
        "parameter_manifest_receipts_resolved": resolved,
        "parameter_manifest_receipts_required": len(parameter_bindings),
        "unresolved_parameter_counts_block_launch": True,
        "immutable_program_spec_required": True,
        "package_root_binding_required": True,
        "role_specific_teacher_authorization_required": True,
        "teacher_authorization_receipts_resolved": 0,
    }


def _canonical_policy_bundle(
    parameter_bindings: Sequence[dict[str, Any]],
    direct_competitors: dict[str, Any],
) -> dict[str, Any]:
    return {
        "quality_policy": _quality_policy(),
        "mandatory_control_policy": _mandatory_control_policy(),
        "failure_routing": _failure_routing_contract(),
        "claim_scopes": _scope_contracts(),
        "teacher_authority_contracts": _teacher_authority_contracts(),
        "all_in_byte_contract": _all_in_byte_contract(),
        "test_time_compute_contract": _test_time_compute_contract(),
        "data_firewall_contract": _data_firewall_contract(),
        "statistical_contract": _statistical_contract(),
        "execution_policy": _execution_policy(parameter_bindings),
        "direct_competitor_requirements": direct_competitors,
    }


def _campaign_input_binding(
    models: Sequence[dict[str, Any]],
    registry: Sequence[dict[str, Any]],
    parameter_bindings: Sequence[dict[str, Any]],
    direct_competitors: dict[str, Any],
) -> dict[str, Any]:
    inputs = _input_report_bindings()
    ladder_value = json.loads(TRAINING_LADDER_REPORT.read_text(encoding="utf-8"))
    quality_value = json.loads(QUALITY_BATTERY_REPORT.read_text(encoding="utf-8"))
    ladder_model_hash = ladder_value.get("compiled_from", {}).get("model_snapshot_sha256")
    expected_model_hash = contract.hash_value(list(models))
    if ladder_model_hash != expected_model_hash:
        raise ValueError("training-ladder input does not bind the current Doctor model snapshot")
    quality_compute_policy = quality_value.get("matched_test_time_compute")
    if not isinstance(quality_compute_policy, dict) or not quality_compute_policy:
        raise ValueError("quality-battery matched-test-time-compute policy is missing")
    if ladder_value.get("matched_test_time_compute") != quality_compute_policy:
        raise ValueError("training ladder and quality battery disagree on matched test-time compute")
    policies = _canonical_policy_bundle(parameter_bindings, direct_competitors)
    payload = {
        "schema": CAMPAIGN_INPUT_SCHEMA,
        "campaign_version": CAMPAIGN_VERSION,
        "compiler_source_sha256": _file_sha256(Path(__file__)),
        "contract_source_sha256": _file_sha256(Path(contract.__file__)),
        "contract_program_spec_schema": contract.PROGRAM_SPEC_SCHEMA,
        "contract_package_root_schema": contract.PACKAGE_ROOT_SCHEMA,
        "model_snapshot_sha256": expected_model_hash,
        "mechanism_registry_sha256": contract.hash_value(list(registry)),
        "parameter_count_bindings_sha256": contract.hash_value(list(parameter_bindings)),
        "canonical_policy_bundle_sha256": contract.hash_value(policies),
        "direct_competitor_registry_sha256": contract.hash_value(direct_competitors),
        "quality_battery_matched_test_time_compute": copy.deepcopy(quality_compute_policy),
        "quality_battery_matched_test_time_compute_sha256": contract.hash_value(
            quality_compute_policy
        ),
        "input_reports": inputs,
    }
    return _stamp(payload, "campaign_input_binding_sha256")


def _candidate_launch_blockers(
    parameter_binding: dict[str, Any],
    *,
    scope: str,
) -> list[str]:
    blockers = [
        "user_greenlight_not_recorded",
        "all_mechanism_adapters_unwired",
        "program_not_source_bound",
        "package_root_manifest_missing",
        "parent_teacher_authorization_receipt_missing",
        "diagnostic_gate_receipt_missing",
        "data_firewall_receipts_missing",
        "resource_and_lease_admission_not_part_of_this_compiler",
    ]
    if parameter_binding["status"] != "verified_exact":
        blockers.extend((
            "exact_parameter_count_source_manifest_missing",
            "exact_parameter_count_manifest_receipt_missing",
        ))
    if scope == "capability_elevation":
        blockers.append("stronger_teacher_authorization_receipt_missing")
    return blockers


def _applicable(
    row: dict[str, Any],
    *,
    model: dict[str, Any],
    rate: float,
    scope: str,
    failure_class: str,
) -> bool:
    return (
        scope in row["claim_scopes"]
        and failure_class in row["failure_routes"]
        and row["minimum_bpw"] <= rate <= row["maximum_bpw"]
        and (not row["moe_only"] or model.get("active_b") is not None)
    )


def _pools_for_lane(
    registry: Sequence[dict[str, Any]],
    model: dict[str, Any],
    rate: float,
    scope: str,
    failure_class: str,
) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {}
    for row in registry:
        if _applicable(row, model=model, rate=rate, scope=scope, failure_class=failure_class):
            pools.setdefault(row["operator_kind"], []).append(row["id"])
    for values in pools.values():
        values.sort()
    required = ["diagnose", "transform", "represent", "train", "package", "evaluate"]
    if failure_class == "computation_collapse":
        required.append("reconstruct")
    if scope != "codec_fidelity":
        required.extend(("repair_static", "repair_conditional", "harden"))
    if scope == "augmented_system":
        required.append("augment_external")
    missing = [kind for kind in required if not pools.get(kind)]
    if missing:
        raise ValueError(
            f"mechanism registry has no applicable {missing} for "
            f"{model['name']} {rate} {scope} {failure_class}"
        )
    return {kind: pools[kind] for kind in required}


def _projected_count(models: Sequence[dict[str, Any]], registry: Sequence[dict[str, Any]]) -> int:
    total = 0
    for model in models:
        for rate in RATE_POINTS:
            for scope in CLAIM_SCOPES:
                for failure_class in FAILURE_ROUTES:
                    pools = _pools_for_lane(registry, model, rate, scope, failure_class)
                    combinations = 1
                    for values in pools.values():
                        combinations *= len(values)
                    total += combinations
    return total


def _required_competitor_ids(
    direct_competitors: dict[str, Any],
    *,
    model: dict[str, Any],
    rate: float,
    scope: str,
) -> list[str]:
    model_class = "mixture_of_experts" if model.get("active_b") is not None else "dense"
    ids = [
        row["implementation_id"]
        for row in direct_competitors["required_methods"]
        if rate in row["required_ladder_rates_bpw"]
        and scope in row["applicable_claim_scopes"]
        and model_class in row["applicable_model_classes"]
    ]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError(
            f"direct competitor coverage is empty or duplicated for "
            f"{model['name']} {rate} {scope}"
        )
    return sorted(ids)


def _candidate_program_spec_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    """Canonical execution semantics behind a candidate identity.

    This compact payload is deliberately independent of queue ordinal and
    evidence state.  Any executable-semantic change—including teacher role,
    exact-count provenance, matched compute, or comparator set—must therefore
    produce a new program-spec hash and candidate identity.
    """
    return {
        "schema": PROGRAM_SPEC_SCHEMA,
        "campaign_version": CAMPAIGN_VERSION,
        "model": {
            "name": candidate["model"],
            "source_sha256": candidate["model_source_sha256"],
            "priority": candidate["model_priority"],
            "parameter_count_binding_sha256": candidate[
                "parameter_count_binding_sha256"
            ],
            "parameter_count_status": candidate["parameter_count_status"],
            "exact_parameter_count": candidate["exact_parameter_count"],
            "parameter_source_manifest_sha256": candidate[
                "parameter_source_manifest_sha256"
            ],
            "parameter_manifest_receipt_sha256": candidate[
                "parameter_manifest_receipt_sha256"
            ],
        },
        "target": {
            "physical_bpw_ceiling": candidate["target_bpw"],
            "physical_artifact_bytes_ceiling": candidate[
                "physical_artifact_bytes_ceiling"
            ],
            "physical_byte_ceiling_basis": candidate["physical_byte_ceiling_basis"],
            "speed_deferred": candidate["speed_deferred"],
        },
        "claim_scope": candidate["claim_scope"],
        "failure_class": candidate["failure_class"],
        "diagnostic_gate_action": candidate["diagnostic_gate_action"],
        "mechanism_ids": candidate["mechanism_ids"],
        "mandatory_control": candidate["mandatory_control"],
        "control_type": candidate["control_type"],
        "teacher_authority_contract_sha256": candidate[
            "teacher_authority_contract_sha256"
        ],
        "stronger_teacher_status": candidate["stronger_teacher_status"],
        "test_time_compute_budget_sha256": candidate[
            "test_time_compute_budget_sha256"
        ],
        "required_competitor_ids": candidate["required_competitor_ids"],
        "required_competitor_ids_sha256": candidate["required_competitor_ids_sha256"],
        "campaign_input_binding_sha256": candidate["campaign_input_binding_sha256"],
        "canonical_policy_bundle_sha256": candidate["canonical_policy_bundle_sha256"],
        "direct_competitor_registry_sha256": candidate[
            "direct_competitor_registry_sha256"
        ],
        "package_root_schema": candidate["package_root_schema"],
        "package_root_manifest_sha256": candidate["package_root_manifest_sha256"],
    }


def _candidate_identity_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign_version": CAMPAIGN_VERSION,
        "model": candidate["model"],
        "model_source_sha256": candidate["model_source_sha256"],
        "model_priority": candidate["model_priority"],
        "parameter_count_binding_sha256": candidate["parameter_count_binding_sha256"],
        "parameter_count_status": candidate["parameter_count_status"],
        "exact_parameter_count": candidate["exact_parameter_count"],
        "parameter_source_manifest_sha256": candidate[
            "parameter_source_manifest_sha256"
        ],
        "parameter_manifest_receipt_sha256": candidate[
            "parameter_manifest_receipt_sha256"
        ],
        "target_bpw": candidate["target_bpw"],
        "physical_artifact_bytes_ceiling": candidate["physical_artifact_bytes_ceiling"],
        "physical_byte_ceiling_basis": candidate["physical_byte_ceiling_basis"],
        "claim_scope": candidate["claim_scope"],
        "failure_class": candidate["failure_class"],
        "diagnostic_gate_action": candidate["diagnostic_gate_action"],
        "mechanism_ids": candidate["mechanism_ids"],
        "mandatory_control": candidate["mandatory_control"],
        "control_type": candidate["control_type"],
        "campaign_input_binding_sha256": candidate["campaign_input_binding_sha256"],
        "canonical_policy_bundle_sha256": candidate["canonical_policy_bundle_sha256"],
        "direct_competitor_registry_sha256": candidate["direct_competitor_registry_sha256"],
        "required_competitor_ids": candidate["required_competitor_ids"],
        "required_competitor_ids_sha256": candidate["required_competitor_ids_sha256"],
        "teacher_authority_contract_sha256": candidate[
            "teacher_authority_contract_sha256"
        ],
        "stronger_teacher_status": candidate["stronger_teacher_status"],
        "test_time_compute_budget_sha256": candidate[
            "test_time_compute_budget_sha256"
        ],
        "package_root_schema": candidate["package_root_schema"],
        "package_root_manifest_sha256": candidate["package_root_manifest_sha256"],
        "program_spec_schema": candidate["program_spec_schema"],
        "program_spec_status": candidate["program_spec_status"],
        "program_spec_sha256": candidate["program_spec_sha256"],
        "program_spec_template_sha256": candidate["program_spec_template_sha256"],
        "speed_deferred": candidate["speed_deferred"],
        "launch_blockers": candidate["launch_blockers"],
    }


def _candidate(
    *,
    model: dict[str, Any],
    model_sha256: str,
    parameter_binding: dict[str, Any],
    rate: float,
    scope: str,
    failure_class: str,
    mechanism_ids: Sequence[str],
    mandatory_control: bool,
    control_type: str | None,
    ordinal: int,
    campaign_input_binding: dict[str, Any],
    canonical_policy_bundle_sha256: str,
    direct_competitor_registry_sha256: str,
    direct_competitors: dict[str, Any],
) -> dict[str, Any]:
    physical_bytes, byte_basis = _physical_byte_ceiling(model, rate, parameter_binding)
    blockers = _candidate_launch_blockers(parameter_binding, scope=scope)
    required_competitors = _required_competitor_ids(
        direct_competitors, model=model, rate=rate, scope=scope,
    )
    teacher_authority = _teacher_authority_contract(
        scope, parent_teacher_identity_sha256=model_sha256,
    )
    compute_budget = _matched_compute_budget(scope)
    row = {
        "model": model["name"],
        "model_source_sha256": model_sha256,
        "model_priority": model["priority"],
        "parameter_count_binding_sha256": parameter_binding["parameter_count_binding_sha256"],
        "parameter_count_status": parameter_binding["status"],
        "exact_parameter_count": parameter_binding["exact_parameter_count"],
        "parameter_source_manifest_sha256": parameter_binding["source_manifest_sha256"],
        "parameter_manifest_receipt_sha256": parameter_binding[
            "parameter_manifest_receipt_sha256"
        ],
        "target_bpw": rate,
        "physical_artifact_bytes_ceiling": physical_bytes,
        "physical_byte_ceiling_basis": byte_basis,
        "claim_scope": scope,
        "failure_class": failure_class,
        "diagnostic_gate_action": "materialize_after_matching_diagnosis_receipt",
        "mechanism_ids": list(mechanism_ids),
        "mandatory_control": mandatory_control,
        "control_type": control_type,
        "explicit_ordinal": ordinal,
        "evidence_stage": "F0",
        "proof_state": "planned",
        "evidence_state": "PLANNED",
        "executor_wired": False,
        "launchable": False,
        "launch_blockers": blockers,
        "campaign_input_binding_sha256": campaign_input_binding[
            "campaign_input_binding_sha256"
        ],
        "canonical_policy_bundle_sha256": canonical_policy_bundle_sha256,
        "direct_competitor_registry_sha256": direct_competitor_registry_sha256,
        "required_competitor_ids": required_competitors,
        "required_competitor_ids_sha256": contract.hash_value(required_competitors),
        "teacher_authority_contract_sha256": teacher_authority[
            "teacher_authority_sha256"
        ],
        "stronger_teacher_status": teacher_authority["stronger_teacher_status"],
        "test_time_compute_budget_sha256": contract.hash_value(compute_budget),
        "package_root_schema": PACKAGE_ROOT_SCHEMA,
        "package_root_manifest_sha256": "required",
        "program_spec_schema": PROGRAM_SPEC_SCHEMA,
        "program_spec_status": "final_semantic_executable_spec_required",
        "program_spec_sha256": "required",
        "speed_deferred": True,
    }
    row["program_spec_template_sha256"] = contract.hash_value(
        _candidate_program_spec_payload(row)
    )
    identity = contract.hash_value(_candidate_identity_payload(row))
    row["candidate_identity_sha256"] = identity
    row["experiment_id"] = f"dv5-{identity[:20]}"
    return row


def _control_mechanisms(
    control_type: str,
    scope: str,
    failure_class: str,
) -> list[str]:
    representation = {
        "untreated_same_rate": "repr_control_untreated_same_rate",
        "scalar_equal_byte": "repr_control_scalar_equal_byte",
        "smaller_higher_bit_equal_byte": "repr_control_smaller_higher_bit",
        "best_public_same_byte": "repr_control_public_frontier",
    }[control_type]
    mechanisms = [
        "diag_early_signal_survival" if failure_class == "computation_collapse"
        else "diag_hidden_logit_tail_kl",
        "transform_identity",
        representation,
    ]
    if failure_class == "computation_collapse":
        mechanisms.append("recon_representation_reset")
    mechanisms.append("train_none_control")
    if scope != "codec_fidelity":
        mechanisms.extend(("repair_zero", "conditional_zero_gate_control", "harden_none_control"))
    if scope == "augmented_system":
        mechanisms.append("augment_external_plane_zero_effect")
    mechanisms.extend(("package_actual_bytes_manifest", "eval_same_budget_dominance"))
    return mechanisms


def _digest_choice(
    pool: Sequence[str],
    *,
    ordinal: int,
    axis: str,
    salt: int,
) -> str:
    digest = hashlib.sha256(
        f"{CAMPAIGN_VERSION}:{ordinal}:{axis}:{salt}".encode("utf-8")
    ).digest()
    return pool[int.from_bytes(digest[:8], "big") % len(pool)]


def _exploration_mechanisms(
    pools: dict[str, list[str]],
    *,
    ordinal: int,
    salt: int,
) -> list[str]:
    ordered_kinds = [
        "diagnose", "transform", "represent", "reconstruct", "repair_static",
        "repair_conditional", "train", "harden", "augment_external", "package", "evaluate",
    ]
    return [
        _digest_choice(pools[kind], ordinal=ordinal, axis=kind, salt=salt)
        for kind in ordered_kinds
        if kind in pools
    ]


def _compile_candidates(
    models: Sequence[dict[str, Any]],
    registry: Sequence[dict[str, Any]],
    parameter_bindings: Sequence[dict[str, Any]],
    explicit_count: int,
    campaign_input_binding: dict[str, Any],
    canonical_policy_bundle_sha256: str,
    direct_competitor_registry_sha256: str,
    direct_competitors: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    lane_count = len(models) * len(RATE_POINTS) * len(CLAIM_SCOPES) * len(FAILURE_ROUTES)
    mandatory_count = lane_count * MANDATORY_CONTROLS_PER_LANE
    if explicit_count < max(DEFAULT_EXPLICIT_COUNT, mandatory_count):
        raise ValueError(
            f"explicit_count must be at least {max(DEFAULT_EXPLICIT_COUNT, mandatory_count)}"
        )
    model_hashes = {model["name"]: contract.hash_value(model) for model in models}
    count_by_model = {row["model"]: row for row in parameter_bindings}
    candidates: list[dict[str, Any]] = []
    identities: set[str] = set()
    ordinal = 0

    for model in models:
        for rate in RATE_POINTS:
            for scope in CLAIM_SCOPES:
                for failure_class in FAILURE_ROUTES:
                    for control_type in CONTROL_TYPES:
                        row = _candidate(
                            model=model,
                            model_sha256=model_hashes[model["name"]],
                            parameter_binding=count_by_model[model["name"]],
                            rate=rate,
                            scope=scope,
                            failure_class=failure_class,
                            mechanism_ids=_control_mechanisms(control_type, scope, failure_class),
                            mandatory_control=True,
                            control_type=control_type,
                            ordinal=ordinal,
                            campaign_input_binding=campaign_input_binding,
                            canonical_policy_bundle_sha256=canonical_policy_bundle_sha256,
                            direct_competitor_registry_sha256=direct_competitor_registry_sha256,
                            direct_competitors=direct_competitors,
                        )
                        if row["candidate_identity_sha256"] in identities:
                            raise AssertionError("mandatory control identity collision")
                        identities.add(row["candidate_identity_sha256"])
                        candidates.append(row)
                        ordinal += 1

    # The explicit frontier is a deterministic hash-stratified sample of an
    # analytically projected space.  Controls already guarantee complete lane
    # coverage; exploration cycles lanes before hashing mechanism axes.
    lanes = [
        (model, rate, scope, failure_class)
        for model in models
        for rate in RATE_POINTS
        for scope in CLAIM_SCOPES
        for failure_class in FAILURE_ROUTES
    ]
    exploration_ordinal = 0
    while len(candidates) < explicit_count:
        model, rate, scope, failure_class = lanes[exploration_ordinal % len(lanes)]
        pools = _pools_for_lane(registry, model, rate, scope, failure_class)
        salt = exploration_ordinal // len(lanes)
        mechanisms = _exploration_mechanisms(
            pools, ordinal=exploration_ordinal, salt=salt,
        )
        row = _candidate(
            model=model,
            model_sha256=model_hashes[model["name"]],
            parameter_binding=count_by_model[model["name"]],
            rate=rate,
            scope=scope,
            failure_class=failure_class,
            mechanism_ids=mechanisms,
            mandatory_control=False,
            control_type=None,
            ordinal=ordinal,
            campaign_input_binding=campaign_input_binding,
            canonical_policy_bundle_sha256=canonical_policy_bundle_sha256,
            direct_competitor_registry_sha256=direct_competitor_registry_sha256,
            direct_competitors=direct_competitors,
        )
        exploration_ordinal += 1
        if row["candidate_identity_sha256"] in identities:
            continue
        identities.add(row["candidate_identity_sha256"])
        candidates.append(row)
        ordinal += 1
        if exploration_ordinal > explicit_count * 100:
            raise AssertionError("deterministic exploration sampler failed to find enough unique candidates")
    return candidates, mandatory_count


def compile_campaign(*, explicit_count: int = DEFAULT_EXPLICIT_COUNT) -> dict[str, Any]:
    models = _source_models()
    registry = mechanism_registry()
    parameter_bindings = _source_parameter_count_bindings()
    direct_competitors = _direct_competitor_requirements(registry)
    canonical_policy_bundle_sha256 = contract.hash_value(
        _canonical_policy_bundle(parameter_bindings, direct_competitors)
    )
    direct_competitor_registry_sha256 = contract.hash_value(direct_competitors)
    campaign_input_binding = _campaign_input_binding(
        models, registry, parameter_bindings, direct_competitors,
    )
    candidates, mandatory_count = _compile_candidates(
        models,
        registry,
        parameter_bindings,
        explicit_count,
        campaign_input_binding,
        canonical_policy_bundle_sha256,
        direct_competitor_registry_sha256,
        direct_competitors,
    )
    projected = _projected_count(models, registry)
    document = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_version": CAMPAIGN_VERSION,
        "doctor_policy_version": contract.POLICY_VERSION,
        "compiler_binding": {
            "compiler_source_sha256": _file_sha256(Path(__file__)),
            "contract_source_sha256": _file_sha256(Path(contract.__file__)),
            "contract_program_schema": contract.PROGRAM_SCHEMA,
            "contract_program_spec_schema": contract.PROGRAM_SPEC_SCHEMA,
            "contract_package_root_schema": contract.PACKAGE_ROOT_SCHEMA,
            "contract_artifact_schema": contract.ARTIFACT_SCHEMA,
            "contract_observation_schema": contract.OBSERVATION_SCHEMA,
            "contract_dominance_schema": contract.DOMINANCE_SCHEMA,
            "model_source": "ladder.MODELS",
            "model_snapshot_sha256": contract.hash_value(models),
            "campaign_input_binding_sha256": campaign_input_binding[
                "campaign_input_binding_sha256"
            ],
            "canonical_policy_bundle_sha256": canonical_policy_bundle_sha256,
            "direct_competitor_registry_sha256": direct_competitor_registry_sha256,
        },
        "campaign_input_binding": campaign_input_binding,
        "execution_policy": _execution_policy(parameter_bindings),
        "quality_policy": _quality_policy(),
        "claim_scopes": _scope_contracts(),
        "teacher_authority_contracts": _teacher_authority_contracts(),
        "failure_routing": _failure_routing_contract(),
        "rate_profiles": _rate_profiles(),
        "evidence_stages": [dict(stage) for stage in EVIDENCE_STAGES],
        "evidence_state_machine": {
            "states": list(contract.EVIDENCE_STATES),
            "headline_claim_requires": "PROVEN",
            "invalid_unreproduced_and_revoked_are_terminal_for_current_identity": True,
            "changed_artifact_runtime_benchmark_or_registry_requires_new_identity": True,
        },
        "all_in_byte_contract": _all_in_byte_contract(),
        "test_time_compute_contract": _test_time_compute_contract(),
        "data_firewall_contract": _data_firewall_contract(),
        "statistical_contract": _statistical_contract(),
        "capability_domains": list(contract.CAPABILITY_DOMAINS),
        "models": models,
        "parameter_count_bindings": parameter_bindings,
        "mechanism_registry": registry,
        "direct_competitor_requirements": direct_competitors,
        "mandatory_control_policy": _mandatory_control_policy(),
        "candidates": candidates,
        "counts": {
            "models": len(models),
            "rates": len(RATE_POINTS),
            "claim_scopes": len(CLAIM_SCOPES),
            "failure_routes": len(FAILURE_ROUTES),
            "fail_closed_diagnoses": len(FAIL_CLOSED_DIAGNOSES),
            "mechanisms": len(registry),
            "direct_competitors": len(direct_competitors["required_methods"]),
            "exact_parameter_counts_resolved": sum(
                row["status"] == "verified_exact" for row in parameter_bindings
            ),
            "exact_parameter_counts_required": len(parameter_bindings),
            "parameter_manifest_receipts_resolved": sum(
                row["status"] == "verified_exact" for row in parameter_bindings
            ),
            "parameter_manifest_receipts_required": len(parameter_bindings),
            "program_specs_bound": len(candidates),
            "launchable_candidates": sum(bool(row["launchable"]) for row in candidates),
            "projected_candidates": projected,
            "explicit_candidates": len(candidates),
            "mandatory_controls": mandatory_count,
            "evidence_stages": len(EVIDENCE_STAGES),
        },
    }
    document = _stamp(document, "campaign_sha256")
    errors = validate_campaign(document)
    if errors:
        raise AssertionError("compiler emitted invalid campaign:\n- " + "\n- ".join(errors[:100]))
    return document


def validate_campaign(campaign: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(campaign, dict):
        return ["campaign must be an object"]
    if campaign.get("schema") != CAMPAIGN_SCHEMA:
        errors.append(f"schema must be {CAMPAIGN_SCHEMA}")
    if campaign.get("campaign_version") != CAMPAIGN_VERSION:
        errors.append(f"campaign_version must be {CAMPAIGN_VERSION}")
    if campaign.get("doctor_policy_version") != contract.POLICY_VERSION:
        errors.append("doctor policy version mismatch")

    expected_hash = campaign.get("campaign_sha256")
    if not contract.is_sha256(expected_hash) or expected_hash != contract.hash_value(
        _identity_payload(campaign, "campaign_sha256")
    ):
        errors.append("campaign_sha256 missing or mismatched")

    expected_models = _source_models()
    expected_parameter_bindings = _source_parameter_count_bindings()
    expected_registry_rows = mechanism_registry()
    expected_direct_competitors = _direct_competitor_requirements(expected_registry_rows)
    expected_policy_bundle_sha256 = contract.hash_value(
        _canonical_policy_bundle(expected_parameter_bindings, expected_direct_competitors)
    )
    expected_direct_sha256 = contract.hash_value(expected_direct_competitors)
    try:
        expected_campaign_input = _campaign_input_binding(
            expected_models,
            expected_registry_rows,
            expected_parameter_bindings,
            expected_direct_competitors,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"campaign-input binding invalid: {exc}")
        expected_campaign_input = {}

    execution = campaign.get("execution_policy")
    required_execution = _execution_policy(expected_parameter_bindings)
    if execution != required_execution:
        errors.append("execution_policy is not the exact fail-closed v5 policy")
    if campaign.get("quality_policy") != _quality_policy():
        errors.append("quality_policy is not the exact canonical v5 quality policy")
    if campaign.get("mandatory_control_policy") != _mandatory_control_policy():
        errors.append("mandatory_control_policy is not the exact canonical per-lane control policy")
    if campaign.get("direct_competitor_requirements") != expected_direct_competitors:
        errors.append("direct competitor requirements differ from the canonical direct-method registry")
    if campaign.get("campaign_input_binding") != expected_campaign_input:
        errors.append(
            "campaign input binding differs from current immutable input reports and policies"
        )

    binding = campaign.get("compiler_binding")
    if not isinstance(binding, dict):
        errors.append("compiler_binding missing")
    else:
        expected_bindings = {
            "compiler_source_sha256": _file_sha256(Path(__file__)),
            "contract_source_sha256": _file_sha256(Path(contract.__file__)),
            "contract_program_schema": contract.PROGRAM_SCHEMA,
            "contract_program_spec_schema": contract.PROGRAM_SPEC_SCHEMA,
            "contract_package_root_schema": contract.PACKAGE_ROOT_SCHEMA,
            "contract_artifact_schema": contract.ARTIFACT_SCHEMA,
            "contract_observation_schema": contract.OBSERVATION_SCHEMA,
            "contract_dominance_schema": contract.DOMINANCE_SCHEMA,
            "model_source": "ladder.MODELS",
            "campaign_input_binding_sha256": expected_campaign_input.get(
                "campaign_input_binding_sha256"
            ),
            "canonical_policy_bundle_sha256": expected_policy_bundle_sha256,
            "direct_competitor_registry_sha256": expected_direct_sha256,
        }
        for field, expected in expected_bindings.items():
            if binding.get(field) != expected:
                errors.append(f"compiler_binding.{field} mismatch")

    models = campaign.get("models")
    if models != expected_models:
        errors.append("models must be the exact canonical snapshot of all ladder.MODELS rows")
        model_rows: list[dict[str, Any]] = []
    else:
        model_rows = models
    if len(model_rows) != EXPECTED_MODEL_COUNT:
        errors.append(f"campaign must contain exactly {EXPECTED_MODEL_COUNT} models")
    if isinstance(binding, dict) and binding.get("model_snapshot_sha256") != contract.hash_value(expected_models):
        errors.append("model snapshot hash mismatch")

    parameter_bindings_raw = campaign.get("parameter_count_bindings")
    if parameter_bindings_raw != expected_parameter_bindings:
        errors.append("parameter-count bindings differ from exact source-manifest policy")
        parameter_bindings: list[dict[str, Any]] = []
    else:
        parameter_bindings = parameter_bindings_raw
    parameter_by_model = {
        row["model"]: row for row in parameter_bindings if isinstance(row, dict)
    }

    scopes = campaign.get("claim_scopes")
    if scopes != _scope_contracts():
        errors.append("claim scope contracts must be exact and non-combinable")
    if [row.get("id") for row in scopes or [] if isinstance(row, dict)] != list(CLAIM_SCOPES):
        errors.append("claim scope order/names differ from doctor_v5_contract")
    if campaign.get("teacher_authority_contracts") != _teacher_authority_contracts():
        errors.append("teacher authority contracts must be exact and claim-scope-specific")
    if campaign.get("failure_routing") != _failure_routing_contract():
        errors.append("signal-degradation/computation-collapse routing contract mismatch")
    if campaign.get("rate_profiles") != _rate_profiles():
        errors.append("rate ladder must be the canonical 4.0-to-0.1 ladder")
    if campaign.get("evidence_stages") != [dict(stage) for stage in EVIDENCE_STAGES]:
        errors.append("F0-F8 evidence stages are incomplete or mismatched")
    if campaign.get("evidence_state_machine") != {
        "states": list(contract.EVIDENCE_STATES),
        "headline_claim_requires": "PROVEN",
        "invalid_unreproduced_and_revoked_are_terminal_for_current_identity": True,
        "changed_artifact_runtime_benchmark_or_registry_requires_new_identity": True,
    }:
        errors.append("evidence state machine mismatch")
    if campaign.get("all_in_byte_contract") != _all_in_byte_contract():
        errors.append("all-in byte contract mismatch")
    if campaign.get("test_time_compute_contract") != _test_time_compute_contract():
        errors.append("test-time compute contract mismatch")
    if campaign.get("data_firewall_contract") != _data_firewall_contract():
        errors.append("data firewall contract mismatch")
    if campaign.get("statistical_contract") != _statistical_contract():
        errors.append("statistical contract mismatch")
    if campaign.get("capability_domains") != list(contract.CAPABILITY_DOMAINS):
        errors.append("capability domain set mismatch")

    registry_raw = campaign.get("mechanism_registry")
    if not isinstance(registry_raw, list):
        errors.append("mechanism_registry must be a list")
        registry_raw = []
    if len(registry_raw) < 100:
        errors.append("mechanism registry must contain at least 100 sourced entries")
    registry: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(registry_raw):
        prefix = f"mechanism_registry[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix} must be an object")
            continue
        mechanism_id = row.get("id")
        if not isinstance(mechanism_id, str) or not mechanism_id:
            errors.append(f"{prefix}.id missing")
            continue
        if mechanism_id in registry:
            errors.append(f"duplicate mechanism id {mechanism_id}")
        registry[mechanism_id] = row
        if row.get("operator_kind") not in contract.OPERATOR_KINDS:
            errors.append(f"{mechanism_id}: invalid operator_kind")
        if row.get("implementation_status") not in contract.IMPLEMENTATION_STATES:
            errors.append(f"{mechanism_id}: invalid implementation status")
        if not isinstance(row.get("source"), str) or not row["source"]:
            errors.append(f"{mechanism_id}: source missing")
        routes = row.get("failure_routes")
        if not isinstance(routes, list) or not routes or not set(routes) <= set(FAILURE_ROUTES):
            errors.append(f"{mechanism_id}: invalid failure routes")
        claim_scopes = row.get("claim_scopes")
        if not isinstance(claim_scopes, list) or not claim_scopes or not set(claim_scopes) <= set(CLAIM_SCOPES):
            errors.append(f"{mechanism_id}: invalid claim scopes")
        minimum = row.get("minimum_bpw")
        maximum = row.get("maximum_bpw")
        if (
            isinstance(minimum, bool) or not isinstance(minimum, (int, float))
            or isinstance(maximum, bool) or not isinstance(maximum, (int, float))
            or not math.isfinite(float(minimum)) or not math.isfinite(float(maximum))
            or minimum <= 0 or minimum > maximum
        ):
            errors.append(f"{mechanism_id}: invalid BPW interval")
        executor = row.get("executor")
        if not isinstance(executor, dict) or executor.get("wired") is not False:
            errors.append(f"{mechanism_id}: every executor must be explicitly unwired")
        if row.get("research_horizon") not in {
            "immediate_implementation", "medium_term_research", "long_term_paradigm_shift"
        }:
            errors.append(f"{mechanism_id}: invalid research horizon")
        proposal = row.get("proposal_evaluation")
        if not isinstance(proposal, dict) or set(proposal) != {
            "theoretical_computational_complexity",
            "expected_bandwidth_reduction",
            "expected_latency_reduction",
            "implementation_difficulty",
            "compatibility",
            "interactions",
            "efficiency_objectives",
        }:
            errors.append(f"{mechanism_id}: proposal evaluation contract incomplete")
        else:
            compatibility = proposal.get("compatibility")
            if not isinstance(compatibility, dict) or set(compatibility) != {
                "existing_gpus", "apple_silicon", "cuda", "future_specialized_hardware"
            }:
                errors.append(f"{mechanism_id}: compatibility contract incomplete")
            interactions = proposal.get("interactions")
            if not isinstance(interactions, dict) or set(interactions) != {
                "quantization", "speculative_decoding", "distributed_inference", "future_architectures"
            }:
                errors.append(f"{mechanism_id}: interaction contract incomplete")
        if row.get("operator_kind") == "augment_external" and claim_scopes != ["augmented_system"]:
            errors.append(f"{mechanism_id}: external augmentation must be augmented-system-only")

    if registry_raw != sorted(registry_raw, key=lambda row: row.get("id", "") if isinstance(row, dict) else ""):
        errors.append("mechanism registry must be canonically sorted")
    if registry_raw != expected_registry_rows:
        errors.append("mechanism registry differs from the exact canonical sourced registry")

    candidates_raw = campaign.get("candidates")
    if not isinstance(candidates_raw, list):
        errors.append("candidates must be a list")
        candidates_raw = []
    if len(candidates_raw) < DEFAULT_EXPLICIT_COUNT:
        errors.append(f"explicit campaign must contain at least {DEFAULT_EXPLICIT_COUNT} candidates")
    model_by_name = {model["name"]: model for model in model_rows}
    seen_identities: set[str] = set()
    lane_coverage: set[tuple[str, float, str, str]] = set()
    mandatory_coverage: dict[tuple[str, float, str, str, str], int] = {}
    mandatory_count = 0
    for index, candidate in enumerate(candidates_raw):
        prefix = f"candidates[{index}]"
        if not isinstance(candidate, dict):
            errors.append(f"{prefix} must be an object")
            continue
        identity = candidate.get("candidate_identity_sha256")
        if not contract.is_sha256(identity):
            errors.append(f"{prefix}: candidate identity missing")
        else:
            try:
                identity_matches = identity == contract.hash_value(
                    _candidate_identity_payload(candidate)
                )
            except (KeyError, TypeError, ValueError) as exc:
                identity_matches = False
                errors.append(f"{prefix}: candidate identity payload incomplete: {exc}")
            if not identity_matches:
                errors.append(f"{prefix}: candidate identity mismatched")
            elif identity in seen_identities:
                errors.append(f"duplicate candidate identity {identity}")
            else:
                seen_identities.add(identity)
        if candidate.get("experiment_id") != f"dv5-{str(identity)[:20]}":
            errors.append(f"{prefix}: experiment id is not identity-derived")
        if candidate.get("launchable") is not False:
            errors.append(f"{prefix}: candidate must be unlaunchable")
        if candidate.get("executor_wired") is not False:
            errors.append(f"{prefix}: candidate executor must be unwired")
        if (
            candidate.get("evidence_stage") != "F0"
            or candidate.get("proof_state") != "planned"
            or candidate.get("evidence_state") != "PLANNED"
        ):
            errors.append(f"{prefix}: fresh candidate must be F0/planned/PLANNED")
        model = model_by_name.get(candidate.get("model"))
        if model is None:
            errors.append(f"{prefix}: unknown model")
            continue
        if candidate.get("model_source_sha256") != contract.hash_value(model):
            errors.append(f"{prefix}: model source hash mismatch")
        if candidate.get("model_priority") != model.get("priority"):
            errors.append(f"{prefix}: model priority differs from source snapshot")
        parameter_binding = parameter_by_model.get(model["name"])
        if parameter_binding is None:
            errors.append(f"{prefix}: parameter-count binding missing")
            continue
        if candidate.get("parameter_count_binding_sha256") != parameter_binding.get(
            "parameter_count_binding_sha256"
        ):
            errors.append(f"{prefix}: parameter-count binding hash mismatch")
        for field, expected in (
            ("parameter_count_status", parameter_binding.get("status")),
            ("exact_parameter_count", parameter_binding.get("exact_parameter_count")),
            ("parameter_source_manifest_sha256", parameter_binding.get("source_manifest_sha256")),
            (
                "parameter_manifest_receipt_sha256",
                parameter_binding.get("parameter_manifest_receipt_sha256"),
            ),
        ):
            if candidate.get(field) != expected:
                errors.append(f"{prefix}: {field} differs from parameter-count binding")
        expected_blockers = _candidate_launch_blockers(
            parameter_binding, scope=candidate.get("claim_scope")
        )
        if candidate.get("launch_blockers") != expected_blockers:
            errors.append(f"{prefix}: launch blockers differ from exact fail-closed policy")
        if candidate.get("campaign_input_binding_sha256") != expected_campaign_input.get(
            "campaign_input_binding_sha256"
        ):
            errors.append(f"{prefix}: campaign-input binding mismatch")
        if candidate.get("canonical_policy_bundle_sha256") != expected_policy_bundle_sha256:
            errors.append(f"{prefix}: canonical-policy binding mismatch")
        if candidate.get("direct_competitor_registry_sha256") != expected_direct_sha256:
            errors.append(f"{prefix}: direct-competitor binding mismatch")
        expected_required_competitors = _required_competitor_ids(
            expected_direct_competitors,
            model=model,
            rate=candidate.get("target_bpw"),
            scope=candidate.get("claim_scope"),
        ) if candidate.get("target_bpw") in RATE_POINTS \
            and candidate.get("claim_scope") in CLAIM_SCOPES else []
        if candidate.get("required_competitor_ids") != expected_required_competitors:
            errors.append(f"{prefix}: required named competitor set mismatched")
        if candidate.get("required_competitor_ids_sha256") != contract.hash_value(
            expected_required_competitors
        ):
            errors.append(f"{prefix}: required named competitor hash mismatched")
        expected_teacher_authority = (
            _teacher_authority_contract(
                candidate["claim_scope"],
                parent_teacher_identity_sha256=candidate["model_source_sha256"],
            )
            if candidate.get("claim_scope") in CLAIM_SCOPES else {}
        )
        if candidate.get("teacher_authority_contract_sha256") != (
            expected_teacher_authority.get("teacher_authority_sha256")
        ):
            errors.append(f"{prefix}: teacher-authority binding mismatched")
        if candidate.get("stronger_teacher_status") != (
            expected_teacher_authority.get("stronger_teacher_status")
        ):
            errors.append(f"{prefix}: stronger-teacher status mismatched")
        expected_compute_sha256 = (
            contract.hash_value(_matched_compute_budget(candidate["claim_scope"]))
            if candidate.get("claim_scope") in CLAIM_SCOPES else None
        )
        if candidate.get("test_time_compute_budget_sha256") != expected_compute_sha256:
            errors.append(f"{prefix}: matched-compute budget binding mismatched")
        if candidate.get("package_root_schema") != PACKAGE_ROOT_SCHEMA:
            errors.append(f"{prefix}: package-root schema mismatched")
        if candidate.get("package_root_manifest_sha256") != "required":
            errors.append(f"{prefix}: fresh candidate must require future package-root binding")
        if candidate.get("program_spec_schema") != PROGRAM_SPEC_SCHEMA:
            errors.append(f"{prefix}: program-spec schema mismatched")
        if candidate.get("program_spec_status") != (
            "final_semantic_executable_spec_required"
        ) or candidate.get("program_spec_sha256") != "required":
            errors.append(f"{prefix}: final executable program-spec must remain unresolved")
        try:
            expected_program_spec_template_sha256 = contract.hash_value(
                _candidate_program_spec_payload(candidate)
            )
        except (KeyError, TypeError, ValueError) as exc:
            expected_program_spec_template_sha256 = None
            errors.append(f"{prefix}: program-spec payload incomplete: {exc}")
        if candidate.get("program_spec_template_sha256") != (
            expected_program_spec_template_sha256
        ):
            errors.append(f"{prefix}: immutable program-spec template binding mismatched")
        if candidate.get("diagnostic_gate_action") != "materialize_after_matching_diagnosis_receipt":
            errors.append(f"{prefix}: diagnostic gate action is not fail-closed")
        if candidate.get("speed_deferred") is not True:
            errors.append(f"{prefix}: speed must remain deferred")
        rate = candidate.get("target_bpw")
        scope = candidate.get("claim_scope")
        failure_class = candidate.get("failure_class")
        if rate not in RATE_POINTS:
            errors.append(f"{prefix}: target rate not in canonical ladder")
            continue
        if scope not in CLAIM_SCOPES:
            errors.append(f"{prefix}: invalid claim scope")
            continue
        if failure_class not in FAILURE_ROUTES:
            errors.append(f"{prefix}: diagnosis is fail-closed and cannot materialize a treatment")
            continue
        expected_bytes, expected_basis = _physical_byte_ceiling(model, rate, parameter_binding)
        if candidate.get("physical_artifact_bytes_ceiling") != expected_bytes:
            errors.append(f"{prefix}: physical artifact byte ceiling mismatched")
        if candidate.get("physical_byte_ceiling_basis") != expected_basis:
            errors.append(f"{prefix}: physical byte ceiling basis mismatched")
        lane_coverage.add((model["name"], rate, scope, failure_class))
        mechanisms = candidate.get("mechanism_ids")
        if not isinstance(mechanisms, list) or not mechanisms or len(mechanisms) != len(set(mechanisms)):
            errors.append(f"{prefix}: mechanism_ids must be a nonempty unique list")
            continue
        kinds: set[str] = set()
        for mechanism_id in mechanisms:
            row = registry.get(mechanism_id)
            if row is None:
                errors.append(f"{prefix}: unknown mechanism {mechanism_id}")
                continue
            kinds.add(row["operator_kind"])
            if not _applicable(row, model=model, rate=rate, scope=scope, failure_class=failure_class):
                errors.append(f"{prefix}: mechanism {mechanism_id} is not applicable")
        required_kinds = {"diagnose", "transform", "represent", "train", "package", "evaluate"}
        if not required_kinds <= kinds:
            errors.append(f"{prefix}: missing core mechanism kinds {sorted(required_kinds - kinds)}")
        if failure_class == "computation_collapse" and "reconstruct" not in kinds:
            errors.append(f"{prefix}: computation collapse requires explicit reconstruction")
        if scope == "codec_fidelity" and kinds & {
            "repair_static", "repair_conditional", "harden", "augment_external"
        }:
            errors.append(f"{prefix}: codec fidelity contains forbidden Doctor mechanisms")
        if scope != "codec_fidelity" and not {"repair_static", "repair_conditional", "harden"} <= kinds:
            errors.append(f"{prefix}: treatment scope lacks full repair/hardening topology")
        if scope == "augmented_system" and "augment_external" not in kinds:
            errors.append(f"{prefix}: augmented system lacks explicit external mechanism")
        if scope != "augmented_system" and "augment_external" in kinds:
            errors.append(f"{prefix}: non-augmented scope contains external mechanism")
        mandatory = candidate.get("mandatory_control")
        control_type = candidate.get("control_type")
        if mandatory is True:
            mandatory_count += 1
            if control_type not in CONTROL_TYPES:
                errors.append(f"{prefix}: mandatory control type invalid")
            else:
                control_key = (
                    model["name"], rate, scope, failure_class, control_type,
                )
                mandatory_coverage[control_key] = mandatory_coverage.get(control_key, 0) + 1
                expected_control_mechanisms = _control_mechanisms(
                    control_type, scope, failure_class,
                )
                if mechanisms != expected_control_mechanisms:
                    errors.append(f"{prefix}: mandatory control mechanism topology mismatched")
                for mechanism_id in expected_control_mechanisms:
                    mechanism = registry.get(mechanism_id)
                    if mechanism is None or not _applicable(
                        mechanism,
                        model=model,
                        rate=rate,
                        scope=scope,
                        failure_class=failure_class,
                    ):
                        errors.append(
                            f"{prefix}: mandatory control {mechanism_id} is not rate/lane applicable"
                        )
        elif mandatory is False:
            if control_type is not None:
                errors.append(f"{prefix}: exploration candidate cannot have control_type")
        else:
            errors.append(f"{prefix}: mandatory_control must be Boolean")

    expected_lanes = (
        len(model_rows) * len(RATE_POINTS) * len(CLAIM_SCOPES) * len(FAILURE_ROUTES)
    )
    if len(lane_coverage) != expected_lanes:
        errors.append(f"candidate coverage has {len(lane_coverage)} of {expected_lanes} required lanes")
    expected_mandatory = expected_lanes * MANDATORY_CONTROLS_PER_LANE
    if mandatory_count != expected_mandatory:
        errors.append(f"mandatory controls must be exactly {expected_mandatory}, found {mandatory_count}")
    expected_mandatory_keys = {
        (model["name"], rate, scope, failure_class, control_type)
        for model in model_rows
        for rate in RATE_POINTS
        for scope in CLAIM_SCOPES
        for failure_class in FAILURE_ROUTES
        for control_type in CONTROL_TYPES
    }
    observed_mandatory_keys = set(mandatory_coverage)
    missing_controls = expected_mandatory_keys - observed_mandatory_keys
    extra_controls = observed_mandatory_keys - expected_mandatory_keys
    duplicated_controls = {
        key: count for key, count in mandatory_coverage.items() if count != 1
    }
    if missing_controls or extra_controls or duplicated_controls:
        errors.append(
            "mandatory controls do not form the exact lane x control-type cross product: "
            f"missing={len(missing_controls)} extra={len(extra_controls)} "
            f"nonunit={len(duplicated_controls)}"
        )

    counts = campaign.get("counts")
    if not isinstance(counts, dict):
        errors.append("counts missing")
    else:
        expected_counts = {
            "models": len(model_rows),
            "rates": len(RATE_POINTS),
            "claim_scopes": len(CLAIM_SCOPES),
            "failure_routes": len(FAILURE_ROUTES),
            "fail_closed_diagnoses": len(FAIL_CLOSED_DIAGNOSES),
            "mechanisms": len(registry_raw),
            "direct_competitors": len(expected_direct_competitors["required_methods"]),
            "exact_parameter_counts_resolved": sum(
                row["status"] == "verified_exact" for row in expected_parameter_bindings
            ),
            "exact_parameter_counts_required": len(expected_parameter_bindings),
            "parameter_manifest_receipts_resolved": sum(
                row["status"] == "verified_exact" for row in expected_parameter_bindings
            ),
            "parameter_manifest_receipts_required": len(expected_parameter_bindings),
            "program_specs_bound": len(candidates_raw),
            "launchable_candidates": sum(
                isinstance(row, dict) and bool(row.get("launchable"))
                for row in candidates_raw
            ),
            "explicit_candidates": len(candidates_raw),
            "mandatory_controls": expected_mandatory,
            "evidence_stages": len(EVIDENCE_STAGES),
        }
        for field, expected in expected_counts.items():
            if counts.get(field) != expected:
                errors.append(f"counts.{field} must be {expected}")
        if set(counts) != set(expected_counts) | {"projected_candidates"}:
            errors.append("counts must contain exactly the canonical v5 count fields")
        projected = counts.get("projected_candidates")
        if not isinstance(projected, int) or projected < len(candidates_raw):
            errors.append("projected candidate count must be an integer no smaller than explicit count")
        elif model_rows and registry_raw:
            try:
                recomputed = _projected_count(model_rows, registry_raw)
                if projected != recomputed:
                    errors.append("projected candidate count mismatch")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"could not recompute projected count: {exc}")
    return errors


def _campaign_registry(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in campaign["mechanism_registry"]}


def _operator_treatment_role(kind: str) -> str:
    return {
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
    }[kind]


def _program_campaign_metadata(
    campaign: dict[str, Any],
    candidate: dict[str, Any],
    parameter_binding: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "schema": CAMPAIGN_METADATA_SCHEMA,
        "campaign_sha256": campaign["campaign_sha256"],
        "campaign_input_binding_sha256": campaign["campaign_input_binding"][
            "campaign_input_binding_sha256"
        ],
        "canonical_policy_bundle_sha256": campaign["compiler_binding"][
            "canonical_policy_bundle_sha256"
        ],
        "direct_competitor_registry_sha256": campaign["compiler_binding"][
            "direct_competitor_registry_sha256"
        ],
        "parameter_count_binding_sha256": parameter_binding[
            "parameter_count_binding_sha256"
        ],
        "teacher_authority_contract_sha256": candidate[
            "teacher_authority_contract_sha256"
        ],
        "test_time_compute_budget_sha256": candidate[
            "test_time_compute_budget_sha256"
        ],
        "research_contracts_sha256": contract.hash_value(
            _program_research_contracts(campaign, candidate)
        ),
        "explicit_ordinal": candidate["explicit_ordinal"],
        "evidence_stage": "F0",
        "evidence_state": "PLANNED",
        "launch_permitted": False,
        "greenlight_recorded": False,
        "blockers": list(candidate["launch_blockers"]),
    }
    return _stamp(metadata, "metadata_sha256")


def _program_research_contracts(
    campaign: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "all_in_bytes": campaign["all_in_byte_contract"],
        "test_time_compute": campaign["test_time_compute_contract"],
        "quality_battery_matched_test_time_compute": copy.deepcopy(
            campaign["campaign_input_binding"]["quality_battery_matched_test_time_compute"]
        ),
        "quality_battery_matched_test_time_compute_sha256": campaign[
            "campaign_input_binding"
        ]["quality_battery_matched_test_time_compute_sha256"],
        "data_firewall": campaign["data_firewall_contract"],
        "statistics": campaign["statistical_contract"],
        "teacher_authority": _teacher_authority_contract(
            candidate["claim_scope"],
            parent_teacher_identity_sha256=candidate["model_source_sha256"],
        ),
        "direct_competitors": campaign["direct_competitor_requirements"],
        "diagnostic_gate": campaign["failure_routing"],
    }


def materialize_program(campaign: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    errors = validate_campaign(campaign)
    if errors:
        raise ValueError("campaign invalid:\n- " + "\n- ".join(errors[:50]))
    candidate = next(
        (
            row for row in campaign["candidates"]
            if row["candidate_identity_sha256"] == candidate_id or row["experiment_id"] == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise KeyError(f"unknown candidate: {candidate_id}")
    models = {model["name"]: model for model in campaign["models"]}
    model = models[candidate["model"]]
    parameter_by_model = {
        row["model"]: row for row in campaign["parameter_count_bindings"]
    }
    parameter_binding = parameter_by_model[candidate["model"]]
    registry = _campaign_registry(campaign)
    operators: list[dict[str, Any]] = []
    previous: str | None = None
    for index, mechanism_id in enumerate(candidate["mechanism_ids"]):
        mechanism = registry[mechanism_id]
        node_id = f"op{index:02d}_{mechanism_id}"
        operators.append(
            {
                "id": node_id,
                "kind": mechanism["operator_kind"],
                "mechanism": mechanism_id,
                "source": mechanism["source"],
                "implementation_status": mechanism["implementation_status"],
                "treatment_role": _operator_treatment_role(mechanism["operator_kind"]),
                "depends_on": [] if previous is None else [previous],
                "executor": {"wired": False, "adapter_id": None, "source_sha256": None},
            }
        )
        previous = node_id

    program = contract.planned_program(
        experiment_id=candidate["experiment_id"],
        candidate_identity_sha256=candidate["candidate_identity_sha256"],
        model_id=model["hf_id"],
        params_b=model["params_b"],
        active_b=model["active_b"],
        target_bpw=candidate["target_bpw"],
        claim_scope=candidate["claim_scope"],
        failure_class=candidate["failure_class"],
        operators=operators,
    )
    program["target"]["physical_artifact_bytes_ceiling"] = candidate[
        "physical_artifact_bytes_ceiling"
    ]
    program["target"]["resident_bytes_ceiling"] = int(WEIGHT_BUDGET * 1_000_000_000)
    program["model"]["exact_parameter_count"] = (
        parameter_binding["exact_parameter_count"]
        if parameter_binding["status"] == "verified_exact" else "required"
    )
    program["model"]["exact_parameter_count_source_sha256"] = parameter_binding[
        "source_manifest_sha256"
    ]
    program["model"]["parameter_count_binding_sha256"] = parameter_binding[
        "parameter_count_binding_sha256"
    ]
    program["model"]["parameter_manifest_receipt"] = "required"
    program["target"]["bpw_is_all_in_physical"] = (
        parameter_binding["status"] == "verified_exact"
    )
    program["target"]["physical_byte_ceiling_basis"] = candidate[
        "physical_byte_ceiling_basis"
    ]
    program["target"]["unresolved_exact_parameter_count_blocks_evidence_and_launch"] = (
        parameter_binding["status"] != "verified_exact"
    )
    program["evaluation_contract"]["suite_manifest_sha256"] = campaign[
        "campaign_input_binding"
    ]["input_reports"]["quality_battery"]["document_identity_sha256"]
    program["evaluation_contract"]["competitor_registry_sha256"] = candidate[
        "direct_competitor_registry_sha256"
    ]
    program["evaluation_contract"]["required_competitor_ids"] = list(
        candidate["required_competitor_ids"]
    )
    program["evaluation_contract"]["required_competitor_ids_sha256"] = candidate[
        "required_competitor_ids_sha256"
    ]
    program["evaluation_contract"]["test_time_compute_budget"] = _matched_compute_budget(
        candidate["claim_scope"]
    )
    teacher_authority = _teacher_authority_contract(
        candidate["claim_scope"],
        parent_teacher_identity_sha256=candidate["model_source_sha256"],
    )
    program["training_contract"]["teacher_contract"] = copy.deepcopy(
        teacher_authority["teacher_contract"]
    )
    program["package_root_schema"] = "required"
    program["package_root_manifest_sha256"] = "required"
    program["program_spec_schema"] = PROGRAM_SPEC_SCHEMA
    program["program_spec_sha256"] = "required"
    program["campaign_metadata"] = _program_campaign_metadata(
        campaign, candidate, parameter_binding,
    )
    program["program_spec_sha256"] = contract.compute_program_spec_sha256(program)
    program = contract.stamp(program, "program_sha256")
    program_errors = validate_materialized_program(
        program, campaign, _campaign_already_valid=True,
    )
    if program_errors:
        raise AssertionError("materialized program invalid:\n- " + "\n- ".join(program_errors))
    if any(node["executor"]["wired"] for node in program["operators"]):
        raise AssertionError("materializer wired an executor")
    return program


def validate_materialized_program(
    program: Any,
    campaign: dict[str, Any],
    *,
    expected_package_root_manifest_sha256: str | None = None,
    _campaign_already_valid: bool = False,
) -> list[str]:
    """Validate a planned materialization against its immutable campaign candidate.

    The generic contract validates program structure and final receipt binding;
    this campaign-level validator additionally proves that a restamped planned
    program has not drifted from the exact candidate semantics selected by the
    compiler.
    """
    errors: list[str] = []
    if not _campaign_already_valid:
        campaign_errors = validate_campaign(campaign)
        if campaign_errors:
            return ["campaign invalid: " + error for error in campaign_errors[:50]]
    if not isinstance(program, dict):
        return ["materialized program must be an object"]
    errors.extend(contract.validate_program(
        program,
        expected_package_root_manifest_sha256=expected_package_root_manifest_sha256,
    ))
    experiment = program.get("experiment_binding")
    if not isinstance(experiment, dict):
        return errors + ["materialized program has no experiment binding"]
    candidate = next((
        row for row in campaign["candidates"]
        if row["candidate_identity_sha256"] == experiment.get("candidate_identity_sha256")
        and row["experiment_id"] == experiment.get("experiment_id")
    ), None)
    if candidate is None:
        return errors + ["materialized program does not bind an exact campaign candidate"]
    model = next(
        row for row in campaign["models"] if row["name"] == candidate["model"]
    )
    parameter_binding = next(
        row for row in campaign["parameter_count_bindings"]
        if row["model"] == candidate["model"]
    )
    expected_root = expected_package_root_manifest_sha256 or "required"
    expected_root_schema = PACKAGE_ROOT_SCHEMA if expected_package_root_manifest_sha256 else "required"
    if program.get("package_root_schema") != expected_root_schema \
            or program.get("package_root_manifest_sha256") != expected_root:
        errors.append("materialized program package-root binding drifted from caller expectation")
    if not contract.is_sha256(program.get("program_spec_sha256")) \
            or program.get("program_spec_sha256") != contract.compute_program_spec_sha256(program):
        errors.append("materialized program immutable program-spec hash is missing or drifted")

    expected_exact = (
        parameter_binding["exact_parameter_count"]
        if parameter_binding["status"] == "verified_exact" else "required"
    )
    model_binding = program.get("model", {})
    expected_model_fields = {
        "id": model["hf_id"],
        "params_b": model["params_b"],
        "active_b": model["active_b"],
        "exact_parameter_count": expected_exact,
        "exact_parameter_count_source_sha256": parameter_binding["source_manifest_sha256"],
        "parameter_count_binding_sha256": parameter_binding[
            "parameter_count_binding_sha256"
        ],
        "parameter_manifest_receipt": "required",
    }
    for field, expected in expected_model_fields.items():
        if not isinstance(model_binding, dict) or model_binding.get(field) != expected:
            errors.append(f"materialized program model.{field} drifted from candidate")

    target = program.get("target", {})
    for field, expected in (
        ("physical_bpw_ceiling", candidate["target_bpw"]),
        ("physical_artifact_bytes_ceiling", candidate["physical_artifact_bytes_ceiling"]),
        ("resident_bytes_ceiling", int(WEIGHT_BUDGET * 1_000_000_000)),
        ("bpw_is_all_in_physical", parameter_binding["status"] == "verified_exact"),
        ("physical_byte_ceiling_basis", candidate["physical_byte_ceiling_basis"]),
        (
            "unresolved_exact_parameter_count_blocks_evidence_and_launch",
            parameter_binding["status"] != "verified_exact",
        ),
        ("speed_deferred", True),
    ):
        if not isinstance(target, dict) or target.get(field) != expected:
            errors.append(f"materialized program target.{field} drifted from candidate")
    if program.get("claim_scope") != candidate["claim_scope"]:
        errors.append("materialized program claim scope drifted from candidate")
    diagnostic = program.get("diagnostic_contract", {})
    if not isinstance(diagnostic, dict) \
            or diagnostic.get("failure_class") != candidate["failure_class"]:
        errors.append("materialized program failure class drifted from candidate")

    registry = _campaign_registry(campaign)
    operators = program.get("operators")
    if not isinstance(operators, list) or len(operators) != len(candidate["mechanism_ids"]):
        errors.append("materialized program operator count drifted from candidate")
    else:
        previous: str | None = None
        for index, (node, mechanism_id) in enumerate(zip(
            operators, candidate["mechanism_ids"], strict=True,
        )):
            mechanism = registry[mechanism_id]
            node_id = f"op{index:02d}_{mechanism_id}"
            expected_node = {
                "id": node_id,
                "kind": mechanism["operator_kind"],
                "mechanism": mechanism_id,
                "source": mechanism["source"],
                "implementation_status": mechanism["implementation_status"],
                "treatment_role": _operator_treatment_role(mechanism["operator_kind"]),
                "depends_on": [] if previous is None else [previous],
                "executor": {"wired": False, "adapter_id": None, "source_sha256": None},
            }
            if node != expected_node:
                errors.append(f"materialized program operator {index} drifted from candidate")
            previous = node_id

    training = program.get("training_contract", {})
    expected_teacher_contract = _teacher_authority_contract(
        candidate["claim_scope"],
        parent_teacher_identity_sha256=candidate["model_source_sha256"],
    )["teacher_contract"]
    if not isinstance(training, dict) \
            or training.get("teacher_contract") != expected_teacher_contract:
        errors.append("materialized program teacher identity/authority drifted from claim scope")
    evaluation = program.get("evaluation_contract", {})
    for field, expected in (
        ("required_competitor_ids", candidate["required_competitor_ids"]),
        ("required_competitor_ids_sha256", candidate["required_competitor_ids_sha256"]),
        ("test_time_compute_budget", _matched_compute_budget(candidate["claim_scope"])),
        (
            "competitor_registry_sha256",
            candidate["direct_competitor_registry_sha256"],
        ),
    ):
        if not isinstance(evaluation, dict) or evaluation.get(field) != expected:
            errors.append(f"materialized program evaluation_contract.{field} drifted")

    expected_campaign_metadata = _program_campaign_metadata(
        campaign, candidate, parameter_binding,
    )
    if program.get("campaign_metadata") != expected_campaign_metadata:
        errors.append("materialized program campaign metadata/program-spec binding drifted")
    return errors


def _load_completed(path: Path | None) -> set[str]:
    if path is None:
        return set()
    value = json.loads(path.read_text())
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = value.get("completed_candidate_ids", value.get("candidate_ids", []))
    else:
        raise ValueError("completed file must be a list or object")
    if not isinstance(rows, list) or any(not isinstance(row, str) for row in rows):
        raise ValueError("completed candidate IDs must be a string list")
    return set(rows)


def select_candidates(
    campaign: dict[str, Any],
    *,
    limit: int = 32,
    completed: Iterable[str] = (),
    model: str | None = None,
    scope: str | None = None,
    failure_class: str | None = None,
    maximum_bpw: float | None = None,
    include_controls: bool = True,
) -> dict[str, Any]:
    errors = validate_campaign(campaign)
    if errors:
        raise ValueError("campaign invalid:\n- " + "\n- ".join(errors[:50]))
    if limit <= 0:
        raise ValueError("limit must be positive")
    if scope is not None and scope not in CLAIM_SCOPES:
        raise ValueError(f"scope must be one of {CLAIM_SCOPES}")
    if failure_class is not None and failure_class not in FAILURE_ROUTES:
        raise ValueError(f"failure_class must be one of {FAILURE_ROUTES}")
    completed_ids = set(completed)
    remaining = [
        candidate
        for candidate in campaign["candidates"]
        if candidate["candidate_identity_sha256"] not in completed_ids
        and candidate["experiment_id"] not in completed_ids
        and (model is None or candidate["model"] == model)
        and (scope is None or candidate["claim_scope"] == scope)
        and (failure_class is None or candidate["failure_class"] == failure_class)
        and (maximum_bpw is None or candidate["target_bpw"] <= maximum_bpw)
        and (include_controls or not candidate["mandatory_control"])
    ]
    selected: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    seen_rates: set[float] = set()
    seen_scopes: set[str] = set()
    seen_failures: set[str] = set()
    seen_mechanisms: set[str] = set()

    # Greedy coverage/VOI proxy without pretending to possess measurements.
    # Mandatory controls win first; diversity then prevents a single scale or
    # method family from consuming the whole initial tranche.
    while remaining and len(selected) < limit:
        best_index = 0
        best_key: tuple[int, str] | None = None
        for index, candidate in enumerate(remaining):
            new_mechanisms = len(set(candidate["mechanism_ids"]) - seen_mechanisms)
            rate_frontier = RATE_POINTS.index(candidate["target_bpw"])
            score = (
                (10_000 if candidate["mandatory_control"] else 0)
                + (2_000 if candidate["model"] not in seen_models else 0)
                + (1_000 if candidate["target_bpw"] not in seen_rates else 0)
                + (600 if candidate["claim_scope"] not in seen_scopes else 0)
                + (400 if candidate["failure_class"] not in seen_failures else 0)
                + new_mechanisms * 20
                + rate_frontier * 5
                - candidate["model_priority"] * 3
            )
            key = (score, "".join(chr(255 - ord(ch)) for ch in candidate["candidate_identity_sha256"]))
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        seen_models.add(chosen["model"])
        seen_rates.add(chosen["target_bpw"])
        seen_scopes.add(chosen["claim_scope"])
        seen_failures.add(chosen["failure_class"])
        seen_mechanisms.update(chosen["mechanism_ids"])

    selection = {
        "schema": SELECTION_SCHEMA,
        "campaign_sha256": campaign["campaign_sha256"],
        "planning_only": True,
        "launch_permitted": False,
        "filters": {
            "model": model,
            "claim_scope": scope,
            "failure_class": failure_class,
            "maximum_bpw": maximum_bpw,
            "include_controls": include_controls,
            "completed_count": len(completed_ids),
        },
        "requested": limit,
        "selected_count": len(selected),
        "selected": selected,
    }
    return _stamp(selection, "selection_sha256")


def validate_selection(selection: Any, campaign: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(selection, dict):
        return ["selection must be an object"]
    if selection.get("schema") != SELECTION_SCHEMA:
        errors.append(f"schema must be {SELECTION_SCHEMA}")
    if selection.get("campaign_sha256") != campaign.get("campaign_sha256"):
        errors.append("selection campaign binding mismatch")
    if selection.get("planning_only") is not True or selection.get("launch_permitted") is not False:
        errors.append("selection must remain planning-only and unlaunchable")
    selected = selection.get("selected")
    if not isinstance(selected, list):
        errors.append("selected must be a list")
        selected = []
    if selection.get("selected_count") != len(selected):
        errors.append("selected_count mismatch")
    campaign_ids = {row["candidate_identity_sha256"] for row in campaign.get("candidates", [])}
    selected_ids: set[str] = set()
    for row in selected:
        if not isinstance(row, dict) or row.get("candidate_identity_sha256") not in campaign_ids:
            errors.append("selection contains unknown candidate")
            continue
        identity = row["candidate_identity_sha256"]
        if identity in selected_ids:
            errors.append("selection contains duplicate candidate")
        selected_ids.add(identity)
        if row.get("launchable") is not False or row.get("executor_wired") is not False:
            errors.append("selection contains launchable or wired candidate")
    expected = selection.get("selection_sha256")
    if not contract.is_sha256(expected) or expected != contract.hash_value(
        _identity_payload(selection, "selection_sha256")
    ):
        errors.append("selection_sha256 missing or mismatched")
    return errors


def selftest() -> int:
    campaign = compile_campaign()
    errors = validate_campaign(campaign)
    assert errors == [], errors[:20]
    assert campaign["counts"]["models"] == EXPECTED_MODEL_COUNT
    assert campaign["counts"]["mechanisms"] >= 100
    assert campaign["counts"]["explicit_candidates"] >= DEFAULT_EXPLICIT_COUNT
    assert campaign["counts"]["mandatory_controls"] >= 1_000
    assert campaign["counts"]["exact_parameter_counts_resolved"] == 0
    assert campaign["counts"]["parameter_manifest_receipts_resolved"] == 0
    assert campaign["counts"]["program_specs_bound"] == len(campaign["candidates"])
    assert campaign["counts"]["launchable_candidates"] == 0
    assert all(candidate["launchable"] is False for candidate in campaign["candidates"])
    assert all(row["executor"]["wired"] is False for row in campaign["mechanism_registry"])

    programs_by_scope: dict[str, dict[str, Any]] = {}
    for scope in CLAIM_SCOPES:
        for failure_class in FAILURE_ROUTES:
            candidate = next(
                row for row in campaign["candidates"]
                if row["claim_scope"] == scope and row["failure_class"] == failure_class
            )
            program = materialize_program(campaign, candidate["candidate_identity_sha256"])
            programs_by_scope.setdefault(scope, program)
            assert contract.validate_program(program) == []
            assert validate_materialized_program(program, campaign) == []
            kinds = {node["kind"] for node in program["operators"]}
            assert all(node["executor"]["wired"] is False for node in program["operators"])
            assert program["model"]["exact_parameter_count"] == "required"
            assert program["model"]["parameter_manifest_receipt"] == "required"
            assert program["program_spec_schema"] == PROGRAM_SPEC_SCHEMA
            assert program["program_spec_sha256"] == contract.compute_program_spec_sha256(
                program
            )
            assert program["package_root_schema"] == "required"
            assert program["package_root_manifest_sha256"] == "required"
            assert program["campaign_metadata"]["campaign_input_binding_sha256"] == campaign[
                "campaign_input_binding"
            ]["campaign_input_binding_sha256"]
            assert program["campaign_metadata"]["launch_permitted"] is False
            assert program["campaign_metadata"]["greenlight_recorded"] is False
            assert program["campaign_metadata"]["blockers"] == candidate[
                "launch_blockers"
            ]
            assert program["campaign_metadata"]["metadata_sha256"] == contract.hash_value(
                _identity_payload(program["campaign_metadata"], "metadata_sha256")
            )
            assert program["evaluation_contract"]["required_competitor_ids"] == candidate[
                "required_competitor_ids"
            ]
            assert program["evaluation_contract"]["required_competitor_ids_sha256"] == (
                contract.hash_value(candidate["required_competitor_ids"])
            )
            assert program["evaluation_contract"]["test_time_compute_budget"] == (
                _matched_compute_budget(scope)
            )
            authority = _teacher_authority_contract(
                scope,
                parent_teacher_identity_sha256=candidate["model_source_sha256"],
            )
            assert program["training_contract"]["teacher_contract"] == authority[
                "teacher_contract"
            ]
            assert candidate["teacher_authority_contract_sha256"] == authority[
                "teacher_authority_sha256"
            ]
            assert candidate["program_spec_template_sha256"] == contract.hash_value(
                _candidate_program_spec_payload(candidate)
            )
            if scope == "codec_fidelity":
                assert not kinds & {"repair_static", "repair_conditional", "harden", "augment_external"}
            if scope == "augmented_system":
                assert "augment_external" in kinds
            if failure_class == "computation_collapse":
                assert "reconstruct" in kinds

    def restamp_materialized(value: dict[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(value)
        value["program_spec_sha256"] = contract.compute_program_spec_sha256(value)
        return contract.stamp(value, "program_sha256")

    # Pre-signing extension smuggling must fail recursively.  Each fixture
    # recomputes both semantic and document identities, proving rejection comes
    # from the exact schema rather than a stale hash.
    unknown_key_fixtures: list[tuple[str, Any]] = [
        ("top", lambda value: value.__setitem__("unsafe_override", True)),
        ("target", lambda value: value["target"].__setitem__("unsafe_override", True)),
        (
            "campaign_metadata",
            lambda value: value["campaign_metadata"].__setitem__(
                "unsafe_override", True
            ),
        ),
        (
            "diagnostic",
            lambda value: value["diagnostic_contract"].__setitem__(
                "unsafe_override", True
            ),
        ),
        (
            "operator",
            lambda value: value["operators"][0].__setitem__("unsafe_override", True),
        ),
        (
            "executor",
            lambda value: value["operators"][0]["executor"].__setitem__(
                "unsafe_override", True
            ),
        ),
        (
            "evaluation",
            lambda value: value["evaluation_contract"].__setitem__(
                "unsafe_override", True
            ),
        ),
        (
            "compute_budget",
            lambda value: value["evaluation_contract"][
                "test_time_compute_budget"
            ].__setitem__("unsafe_override", True),
        ),
        (
            "data",
            lambda value: value["data_contract"].__setitem__("unsafe_override", True),
        ),
        (
            "data_split",
            lambda value: next(iter(value["data_contract"]["splits"].values())).__setitem__(
                "unsafe_override", True
            ),
        ),
        (
            "resume",
            lambda value: value["exact_resume_contract"].__setitem__(
                "unsafe_override", True
            ),
        ),
        (
            "output",
            lambda value: value["output_contract"].__setitem__(
                "unsafe_override", True
            ),
        ),
    ]
    for level, inject in unknown_key_fixtures:
        smuggled = copy.deepcopy(programs_by_scope["restorative_training"])
        inject(smuggled)
        smuggled = restamp_materialized(smuggled)
        schema_errors = contract.validate_program(smuggled)
        assert schema_errors, f"unknown {level} program key was accepted"
        assert any(
            "exact" in error or "field" in error or "unknown" in error
            for error in schema_errors
        ), (level, schema_errors)

    # A concrete root must agree with the direct program contract and the
    # caller-owned expected root.  Restamping cannot turn root drift into the
    # selected campaign program.
    root_sha256 = "a" * 64
    root_bound = copy.deepcopy(programs_by_scope["restorative_training"])
    root_bound["package_root_schema"] = PACKAGE_ROOT_SCHEMA
    root_bound["package_root_manifest_sha256"] = root_sha256
    root_bound = restamp_materialized(root_bound)
    assert validate_materialized_program(
        root_bound,
        campaign,
        expected_package_root_manifest_sha256=root_sha256,
    ) == []
    root_drift = copy.deepcopy(root_bound)
    root_drift["package_root_manifest_sha256"] = "b" * 64
    root_drift = restamp_materialized(root_drift)
    assert any(
        "package-root binding drifted" in error or "package root" in error
        for error in validate_materialized_program(
            root_drift,
            campaign,
            expected_package_root_manifest_sha256=root_sha256,
        )
    )

    # Stronger-teacher authority is capability-elevation-only.  Both injecting
    # it into restorative/augmented scopes and deleting it from elevation must
    # fail even after recomputing every self-stamp.
    elevation_teacher = copy.deepcopy(
        programs_by_scope["capability_elevation"]["training_contract"][
            "teacher_contract"
        ]["elevation_teacher"]
    )
    for forbidden_scope in ("restorative_training", "augmented_system"):
        teacher_misuse = copy.deepcopy(programs_by_scope[forbidden_scope])
        teacher_misuse["training_contract"]["teacher_contract"][
            "elevation_teacher"
        ] = copy.deepcopy(elevation_teacher)
        teacher_misuse = restamp_materialized(teacher_misuse)
        misuse_errors = validate_materialized_program(teacher_misuse, campaign)
        assert any("capability_elevation" in error for error in misuse_errors)
        assert any("teacher identity/authority drifted" in error for error in misuse_errors)
    missing_elevation = copy.deepcopy(programs_by_scope["capability_elevation"])
    missing_elevation["training_contract"]["teacher_contract"][
        "elevation_teacher"
    ] = None
    missing_elevation = restamp_materialized(missing_elevation)
    assert any(
        "stronger_training_teacher" in error or "teacher identity/authority drifted" in error
        for error in validate_materialized_program(missing_elevation, campaign)
    )

    # Exact parameter count and its source are candidate semantics.  A fully
    # restamped planned program cannot replace the unresolved trusted manifest
    # with a self-asserted count/source.
    parameter_mutation = copy.deepcopy(programs_by_scope["restorative_training"])
    parameter_mutation["model"]["exact_parameter_count"] = 70_000_000_000
    parameter_mutation["model"]["exact_parameter_count_source_sha256"] = "c" * 64
    parameter_mutation = restamp_materialized(parameter_mutation)
    parameter_errors = validate_materialized_program(parameter_mutation, campaign)
    assert any("model.exact_parameter_count drifted" in error for error in parameter_errors)
    assert any(
        "model.exact_parameter_count_source_sha256 drifted" in error
        for error in parameter_errors
    )

    selection = select_candidates(campaign, limit=32)
    assert validate_selection(selection, campaign) == []
    assert selection["selected_count"] == 32
    assert selection["launch_permitted"] is False

    damaged = copy.deepcopy(campaign)
    damaged["candidates"][0]["launchable"] = True
    damaged = _stamp(damaged, "campaign_sha256")
    assert any("unlaunchable" in error for error in validate_campaign(damaged))

    # Restamping must not make a weakened canonical policy valid.
    damaged = copy.deepcopy(campaign)
    damaged["quality_policy"][
        "unbeatable_claim_before_independent_reproduction_forbidden"
    ] = False
    damaged = _stamp(damaged, "campaign_sha256")
    assert any("quality_policy" in error for error in validate_campaign(damaged))

    damaged = copy.deepcopy(campaign)
    damaged["test_time_compute_contract"]["core_budget"].pop("max_input_tokens")
    damaged = _stamp(damaged, "campaign_sha256")
    assert any("test-time compute contract" in error for error in validate_campaign(damaged))

    damaged = copy.deepcopy(campaign)
    damaged["campaign_input_binding"]["quality_battery_matched_test_time_compute"][
        "required_fields"
    ].pop("reasoning")
    damaged = _stamp(damaged, "campaign_sha256")
    assert any("campaign input binding" in error for error in validate_campaign(damaged))

    # The physical ceiling is semantic candidate identity, not mutable metadata.
    damaged = copy.deepcopy(campaign)
    original_identity = damaged["candidates"][0]["candidate_identity_sha256"]
    damaged["candidates"][0]["physical_artifact_bytes_ceiling"] = 1
    damaged = _stamp(damaged, "campaign_sha256")
    byte_errors = validate_campaign(damaged)
    assert any("byte ceiling" in error for error in byte_errors)
    assert any("identity mismatched" in error for error in byte_errors)
    assert damaged["candidates"][0]["candidate_identity_sha256"] == original_identity

    # Moving a global control count between lanes cannot fake complete coverage.
    damaged = copy.deepcopy(campaign)
    removed = next(row for row in damaged["candidates"] if row["mandatory_control"])
    removed_lane = (
        removed["model"], removed["target_bpw"], removed["claim_scope"], removed["failure_class"],
    )
    added = next(
        row for row in damaged["candidates"]
        if not row["mandatory_control"] and (
            row["model"], row["target_bpw"], row["claim_scope"], row["failure_class"],
        ) != removed_lane
    )
    removed["mandatory_control"] = False
    removed["control_type"] = None
    removed["candidate_identity_sha256"] = contract.hash_value(
        _candidate_identity_payload(removed)
    )
    removed["experiment_id"] = f"dv5-{removed['candidate_identity_sha256'][:20]}"
    added["mandatory_control"] = True
    added["control_type"] = "untreated_same_rate"
    added["candidate_identity_sha256"] = contract.hash_value(
        _candidate_identity_payload(added)
    )
    added["experiment_id"] = f"dv5-{added['candidate_identity_sha256'][:20]}"
    damaged = _stamp(damaged, "campaign_sha256")
    control_errors = validate_campaign(damaged)
    assert any("control mechanism topology" in error for error in control_errors)
    assert any("exact lane x control-type" in error for error in control_errors)

    # Direct-method competitors, blocked diagnoses, parameter sources, and root
    # input identities are canonical and cannot be weakened by restamping.
    damaged = copy.deepcopy(campaign)
    damaged["direct_competitor_requirements"]["required_methods"] = [
        row for row in damaged["direct_competitor_requirements"]["required_methods"]
        if row["implementation_id"] != "littlebit_reference"
    ]
    damaged = _stamp(damaged, "campaign_sha256")
    assert any("direct competitor" in error for error in validate_campaign(damaged))

    damaged = copy.deepcopy(campaign)
    competitor_candidate = next(
        row for row in damaged["candidates"] if len(row["required_competitor_ids"]) > 1
    )
    competitor_candidate["required_competitor_ids"].pop()
    competitor_candidate["required_competitor_ids_sha256"] = contract.hash_value(
        competitor_candidate["required_competitor_ids"]
    )
    competitor_candidate["candidate_identity_sha256"] = contract.hash_value(
        _candidate_identity_payload(competitor_candidate)
    )
    competitor_candidate["experiment_id"] = (
        f"dv5-{competitor_candidate['candidate_identity_sha256'][:20]}"
    )
    damaged = _stamp(damaged, "campaign_sha256")
    assert any("required named competitor" in error for error in validate_campaign(damaged))

    damaged = copy.deepcopy(campaign)
    mixed = next(row for row in damaged["failure_routing"] if row["id"] == "mixed_failure")
    mixed["materialization_permitted"] = True
    damaged = _stamp(damaged, "campaign_sha256")
    assert any("routing contract" in error for error in validate_campaign(damaged))

    damaged = copy.deepcopy(campaign)
    injected = damaged["candidates"][0]
    injected["failure_class"] = "mixed_failure"
    injected["candidate_identity_sha256"] = contract.hash_value(
        _candidate_identity_payload(injected)
    )
    injected["experiment_id"] = f"dv5-{injected['candidate_identity_sha256'][:20]}"
    damaged = _stamp(damaged, "campaign_sha256")
    assert any(
        "fail-closed and cannot materialize" in error for error in validate_campaign(damaged)
    )

    damaged = copy.deepcopy(campaign)
    damaged["parameter_count_bindings"][0]["status"] = "verified_exact"
    damaged = _stamp(damaged, "campaign_sha256")
    assert any("parameter-count bindings" in error for error in validate_campaign(damaged))

    def restamp_candidate_campaign(
        source: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        candidate["program_spec_template_sha256"] = contract.hash_value(
            _candidate_program_spec_payload(candidate)
        )
        candidate["candidate_identity_sha256"] = contract.hash_value(
            _candidate_identity_payload(candidate)
        )
        candidate["experiment_id"] = f"dv5-{candidate['candidate_identity_sha256'][:20]}"
        return _stamp(source, "campaign_sha256")

    # Even a candidate attacker who recomputes the template, candidate, and
    # campaign self-hashes cannot alter the canonical count, teacher, compute,
    # or package-root semantics.
    damaged = copy.deepcopy(campaign)
    injected = damaged["candidates"][0]
    injected["parameter_count_status"] = "verified_exact"
    injected["exact_parameter_count"] = 70_000_000_000
    injected["parameter_source_manifest_sha256"] = "d" * 64
    injected["parameter_manifest_receipt_sha256"] = "e" * 64
    damaged = restamp_candidate_campaign(damaged, injected)
    assert any(
        "differs from parameter-count binding" in error
        for error in validate_campaign(damaged)
    )

    damaged = copy.deepcopy(campaign)
    injected = next(
        row for row in damaged["candidates"]
        if row["claim_scope"] == "restorative_training"
    )
    injected["stronger_teacher_status"] = "required_before_execution"
    injected["teacher_authority_contract_sha256"] = "f" * 64
    damaged = restamp_candidate_campaign(damaged, injected)
    assert any("teacher-authority" in error for error in validate_campaign(damaged))
    assert any("stronger-teacher status" in error for error in validate_campaign(damaged))

    damaged = copy.deepcopy(campaign)
    injected = damaged["candidates"][0]
    injected["test_time_compute_budget_sha256"] = "1" * 64
    damaged = restamp_candidate_campaign(damaged, injected)
    assert any("matched-compute" in error for error in validate_campaign(damaged))

    damaged = copy.deepcopy(campaign)
    injected = damaged["candidates"][0]
    injected["package_root_schema"] = "hawking.wrong_root.v0"
    damaged = restamp_candidate_campaign(damaged, injected)
    assert any("package-root schema" in error for error in validate_campaign(damaged))

    damaged = copy.deepcopy(campaign)
    damaged["campaign_input_binding"]["input_reports"]["quality_battery"][
        "file_sha256"
    ] = "0" * 64
    damaged = _stamp(damaged, "campaign_sha256")
    assert any("campaign input binding" in error for error in validate_campaign(damaged))

    assert campaign["execution_policy"]["exact_parameter_counts_resolved"] == 0
    assert all(
        "exact_parameter_count_source_manifest_missing" in row["launch_blockers"]
        for row in campaign["candidates"]
    )
    assert all(
        "exact_parameter_count_manifest_receipt_missing" in row["launch_blockers"]
        and "parent_teacher_authorization_receipt_missing" in row["launch_blockers"]
        for row in campaign["candidates"]
    )
    assert all(
        ("stronger_teacher_authorization_receipt_missing" in row["launch_blockers"])
        == (row["claim_scope"] == "capability_elevation")
        for row in campaign["candidates"]
    )

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "campaign.json"
        contract.atomic_json(path, campaign)
        loaded = json.loads(path.read_text())
        assert validate_campaign(loaded) == []
    print(
        "doctor_v5.py selftest OK: "
        f"{campaign['counts']['models']} models, "
        f"{campaign['counts']['mechanisms']} mechanisms, "
        f"{campaign['counts']['explicit_candidates']} explicit, "
        f"{campaign['counts']['mandatory_controls']} controls"
    )
    return 0


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    compile_parser.add_argument("--explicit-count", type=int, default=DEFAULT_EXPLICIT_COUNT)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_REPORT)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_REPORT)
    select_parser.add_argument("--limit", type=int, default=32)
    select_parser.add_argument("--completed", type=Path)
    select_parser.add_argument("--model")
    select_parser.add_argument("--scope", choices=CLAIM_SCOPES)
    select_parser.add_argument("--failure-class", choices=FAILURE_ROUTES)
    select_parser.add_argument("--maximum-bpw", type=float)
    select_parser.add_argument("--exclude-controls", action="store_true")

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("candidate_id")
    materialize_parser.add_argument("--campaign", type=Path, default=DEFAULT_REPORT)
    materialize_parser.add_argument("--output", type=Path)

    subparsers.add_parser("selftest")
    args = parser.parse_args()

    if args.command == "compile":
        campaign = compile_campaign(explicit_count=args.explicit_count)
        contract.atomic_json(args.output, campaign)
        print(json.dumps({
            "ok": True,
            "output": str(args.output),
            "campaign_sha256": campaign["campaign_sha256"],
            "counts": campaign["counts"],
            "launchable": False,
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        errors = validate_campaign(_read_json(args.path))
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 1
    if args.command == "select":
        campaign = _read_json(args.path)
        selection = select_candidates(
            campaign,
            limit=args.limit,
            completed=_load_completed(args.completed),
            model=args.model,
            scope=args.scope,
            failure_class=args.failure_class,
            maximum_bpw=args.maximum_bpw,
            include_controls=not args.exclude_controls,
        )
        print(json.dumps(selection, indent=2, sort_keys=True))
        return 0
    if args.command == "materialize":
        campaign = _read_json(args.campaign)
        program = materialize_program(campaign, args.candidate_id)
        if args.output is None:
            print(json.dumps(program, indent=2, sort_keys=True))
        else:
            contract.atomic_json(args.output, program)
            print(json.dumps({
                "ok": True,
                "output": str(args.output),
                "program_sha256": program["program_sha256"],
                "launch_permitted": False,
            }, indent=2, sort_keys=True))
        return 0
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
