#!/usr/bin/env python3.12
"""Compile Doctor-v5's capability-first evaluation and contamination firewall.

This is a manifest compiler, not an evaluator.  It freezes what must be tested,
how the statistical unit is defined, which public suites are development-only,
and which private/post-cutoff suites must be consumed once.  No benchmark data,
model output, private prompt, network call, or process launch occurs here.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import tempfile
from typing import Any

import doctor_v5_contract as contract


SCHEMA = "hawking.quality_battery.v5"
VERSION = "2026-07-12.v5"
DEFAULT_OUTPUT = Path("reports/condense/quality_battery_v5.json")
RATE_POINTS = (4.0, 3.0, 2.0, 1.0, 0.8, 0.55, 0.5, 0.33, 0.25, 0.1)
RECEIPT_PLACEHOLDER = "required"
CENTRAL_DIRECT_IMPLEMENTATIONS = (
    "btc_llm_binary_codebook",
    "bwla_w1a6_ptq",
    "qmoe_switch_subbit",
    "spear_token_gated_compensation",
    "dbf_double_binary_factorization",
    "multi_envelope_dbf",
    "marr_module_adaptive_residual",
    "lbllm_three_stage_distillation",
    "scalebits_mixed_precision",
    "shannon_ans_lossless_wrapper",
)


def _suite(
    suite_id: str,
    domains: tuple[str, ...],
    role: str,
    source: str,
    scoring: str,
    cluster_unit: str,
    *,
    objective: bool = True,
    temporal_freeze: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": suite_id,
        "domains": list(domains),
        "role": role,
        "primary_source": source,
        "scoring": scoring,
        "cluster_unit": cluster_unit,
        "objective_scoring": objective,
        "temporal_freeze_required": temporal_freeze,
        "manifest_sha256": "required",
        "prompt_protocol_sha256": "required",
        "scorer_source_sha256": "required",
        "notes": notes,
    }


SUITES: tuple[dict[str, Any], ...] = (
    _suite(
        "fresh_document_nll",
        ("language_modeling", "calibration", "multilingual"),
        "sealed_primary",
        "local:post-cutoff-document-sampler",
        "document NLL, teacher KL/JS, Brier/ECE, rare-token/tail loss",
        "document",
        temporal_freeze=True,
        notes="Bootstrap documents, never correlated tokens; stratify language/domain/length.",
    ),
    _suite(
        "livebench_frozen_snapshot",
        ("knowledge", "reasoning", "mathematics", "coding", "instruction_following"),
        "public_development",
        "https://arxiv.org/abs/2406.19314",
        "official deterministic ground-truth scorers",
        "item",
        temporal_freeze=True,
        notes="Contamination-limited public development evidence, never the sealed proof alone.",
    ),
    _suite(
        "livecodebench_post_cutoff",
        ("coding", "reasoning"),
        "frozen_primary",
        "https://arxiv.org/abs/2403.07974",
        "identical execution sandbox, tests, sampling n/k, and token budget",
        "task",
        temporal_freeze=True,
    ),
    _suite(
        "arc_agi_2",
        ("reasoning",),
        "frozen_primary",
        "https://arxiv.org/abs/2505.11831",
        "exact grid output under matched attempts/compute",
        "task",
    ),
    _suite(
        "ifeval",
        ("instruction_following",),
        "public_development",
        "https://arxiv.org/abs/2311.07911",
        "official verifiable instruction checks",
        "item",
    ),
    _suite(
        "bfcl_v4_frozen",
        ("tool_use", "instruction_following"),
        "frozen_primary",
        "https://gorilla.cs.berkeley.edu/leaderboard",
        "official AST/live/multiturn/hallucination checks",
        "interaction",
        temporal_freeze=True,
    ),
    _suite(
        "ruler_curve",
        ("long_context", "reasoning"),
        "frozen_primary",
        "https://arxiv.org/abs/2404.06654",
        "official exact/objective tasks at 4k,8k,16k,32k,max context",
        "item_by_context",
    ),
    _suite(
        "nolima_curve",
        ("long_context", "knowledge", "reasoning"),
        "frozen_primary",
        "https://arxiv.org/abs/2502.05167",
        "official long-context latent-association scoring",
        "item_by_context",
    ),
    _suite(
        "longbench_v2",
        ("long_context", "reasoning", "coding"),
        "secondary_diagnostic",
        "https://arxiv.org/abs/2412.15204",
        "official multiple-choice scoring by context/task",
        "item_by_task",
        notes="Public and exposed; useful transfer diagnostic, not a one-shot private proof.",
    ),
    _suite(
        "bigcodebench",
        ("coding", "instruction_following"),
        "frozen_primary",
        "https://arxiv.org/abs/2406.15877",
        "identical execution environment and pass@k budget",
        "task",
    ),
    _suite(
        "evalplus",
        ("coding",),
        "public_development",
        "https://arxiv.org/abs/2305.01210",
        "official augmented tests under matched samples",
        "task",
    ),
    _suite(
        "mmlu_prox",
        ("knowledge", "reasoning", "science", "multilingual"),
        "frozen_primary",
        "https://arxiv.org/abs/2503.10497",
        "official per-language/per-domain exact choice scoring",
        "item_by_language_domain",
    ),
    _suite(
        "mmlu_redux_audited",
        ("knowledge", "reasoning", "science"),
        "secondary_diagnostic",
        "https://arxiv.org/abs/2406.04127",
        "audited-item exact choice scoring",
        "item_by_domain",
        notes="Prefer audited items; original MMLU is not primary v5 proof evidence.",
    ),
    _suite(
        "gpqa_diamond",
        ("science", "reasoning"),
        "secondary_diagnostic",
        "https://arxiv.org/abs/2311.12022",
        "exact choice, prompt variants reported",
        "item_by_domain",
        notes="Exposed static benchmark; secondary diagnostic only.",
    ),
    _suite(
        "math_rob",
        ("mathematics", "reasoning"),
        "transfer_primary",
        "https://arxiv.org/abs/2503.04550",
        "official robustness transformations and exact answers",
        "base_item_and_mutation_family",
    ),
    _suite(
        "lgmt_metamorphic",
        ("reasoning", "instruction_following", "safety_security"),
        "transfer_primary",
        "https://arxiv.org/abs/2605.23965",
        "logic-grounded metamorphic consistency",
        "base_item_and_relation",
    ),
    _suite(
        "fresh_procedural_reasoning",
        ("reasoning", "mathematics", "science", "instruction_following"),
        "sealed_primary",
        "local:seed-committed-procedural-generator",
        "programmatic oracle plus expert audit sample",
        "generator_family_and_seed",
        temporal_freeze=True,
        notes="Seeds committed before generation; mutation families held out from Doctor.",
    ),
    _suite(
        "fresh_multilingual_constraints",
        ("multilingual", "instruction_following", "reasoning"),
        "sealed_primary",
        "local:post-cutoff-human-and-procedural-vault",
        "objective constraints plus bilingual human adjudication sample",
        "language_family_and_item",
        temporal_freeze=True,
    ),
    _suite(
        "quantization_security_metamorphics",
        ("safety_security", "calibration", "instruction_following"),
        "sealed_primary",
        "https://arxiv.org/abs/2605.15152",
        "harmful-compliance, over-refusal, injected-outlier and confidence-shift checks",
        "attack_family_model_seed_item",
        temporal_freeze=True,
        notes="Includes benign-parent/quantized-child differential behavior and clean controls.",
    ),
    _suite(
        "sealed_capability_tail",
        tuple(contract.CAPABILITY_DOMAINS),
        "sealed_primary",
        "local:post-freeze-private-capability-vault",
        "objective oracle where possible; blind multi-judge plus human adjudication otherwise",
        "capability_family_item_prompt_seed",
        temporal_freeze=True,
        notes="One-shot lineage consumption; promotion service returns aggregates only.",
    ),
    _suite(
        "independent_replication_battery",
        tuple(contract.CAPABILITY_DOMAINS),
        "independent_primary",
        "local:independent-owner-post-freeze-vault",
        "same preregistered metrics under an independently held prompt set",
        "capability_family_item_prompt_seed",
        temporal_freeze=True,
    ),
)


FORBIDDEN_PRIMARY = {
    "swe_bench_verified": "OpenAI's 2026 audit found contamination and material task/test defects",
    "original_mmlu": "known item errors and saturation; audited Redux/ProX only",
    "wikitext_perplexity_only": "proxy cannot establish capability preservation",
}


def _competitor(
    implementation_id: str,
    family_id: str,
    source: str,
    required_rates: tuple[float, ...],
    source_rate_evidence: dict[str, Any],
    *,
    claim_scopes: tuple[str, ...] = tuple(contract.CLAIM_SCOPES),
    model_classes: tuple[str, ...] = ("dense", "mixture_of_experts"),
    extension_rates: tuple[float, ...] = (),
    exclusions: dict[str, str] | None = None,
    control_kind: str = "representation",
    source_model_evidence: dict[str, Any] | None = None,
    source_release_status: str = "verify_source_and_code_receipt_before_run",
    notes: str = "",
) -> dict[str, Any]:
    """Describe one implementation obligation, never a fungible family label."""

    return {
        "implementation_id": implementation_id,
        "family_id": family_id,
        "control_kind": control_kind,
        "primary_source": source,
        "source_release_status": source_release_status,
        "source_model_evidence": copy.deepcopy(source_model_evidence or {
            "qualification": "reproduce only on a source-compatible parent or issue incompatibility receipt"
        }),
        "applicable_model_classes": list(model_classes),
        "applicable_claim_scopes": list(claim_scopes),
        "required_ladder_rates_bpw": list(required_rates),
        "source_rate_evidence": copy.deepcopy(source_rate_evidence),
        "research_extension_rates_bpw": list(extension_rates),
        "explicitly_excluded_ladder_rates_bpw": copy.deepcopy(exclusions or {}),
        "mandatory_when_parent_architecture_and_source_are_reproducible": True,
        "direct_same_parent_reproduction_required": True,
        "packed_artifact_required": True,
        "same_or_lower_all_in_physical_bytes_required": True,
        "reported_table_scores_are_not_evidence": True,
        "resolution_contract": {
            "compatible_path": (
                "direct same-parent packed reproduction plus artifact and evaluation receipts"
            ),
            "incompatible_path": (
                "signed rate/model/scope incompatibility receipt plus machine-readable narrowed claim"
            ),
            "silent_omission_forbidden": True,
            "incompatibility_is_not_a_competitor_win": True,
            "missing_or_unverified_source_release_blocks_broad_headline": True,
        },
        "method_source_sha256": RECEIPT_PLACEHOLDER,
        "applicability_receipt_sha256": RECEIPT_PLACEHOLDER,
        "packed_artifact_receipt_sha256": RECEIPT_PLACEHOLDER,
        "reproduction_receipt_sha256": RECEIPT_PLACEHOLDER,
        "incompatibility_receipt_sha256": RECEIPT_PLACEHOLDER,
        "narrowed_claim_receipt_sha256": RECEIPT_PLACEHOLDER,
        "notes": notes,
    }


def _competitor_coverage() -> dict[str, Any]:
    """Return direct, rate-bound obligations for every v5 precision region.

    ``source_rate_evidence`` records what the cited work actually demonstrates.
    A ``research_extension_rates_bpw`` entry is deliberately *not* imported
    evidence: it is an obligation to run and measure that implementation at the
    requested Hawking rate.  Failure to reproduce narrows the claim.
    """

    trained_scopes = ("restorative_training", "capability_elevation", "augmented_system")
    implementations = [
        _competitor(
            "gptq_reference",
            "scalar_ptq",
            "https://arxiv.org/abs/2210.17323",
            (4.0, 3.0, 2.0),
            {
                "kind": "closed_integer_range",
                "minimum_bpw": 2.0,
                "maximum_bpw": 4.0,
                "qualification": "reproduce the released/reference method; physical metadata is billed",
            },
        ),
        _competitor(
            "awq_reference",
            "scalar_ptq",
            "https://arxiv.org/abs/2306.00978",
            (4.0, 3.0),
            {
                "kind": "reported_integer_rates",
                "rates_bpw": [4.0, 3.0],
                "qualification": "no 2-bit obligation is inferred from a <=4-bit family label",
            },
        ),
        _competitor(
            "quip_sharp_e8",
            "lattice_vector_ptq",
            "https://arxiv.org/abs/2402.04396",
            (4.0, 3.0, 2.0),
            {
                "kind": "closed_integer_range",
                "minimum_bpw": 2.0,
                "maximum_bpw": 4.0,
                "qualification": "extreme-compression <=4-bit method; reproduce exact applicable configurations",
            },
        ),
        _competitor(
            "aqlm_additive",
            "additive_codebook",
            "https://arxiv.org/abs/2401.06118",
            (3.0, 2.0),
            {
                "kind": "closed_range",
                "minimum_bpw": 2.0,
                "maximum_bpw": 3.0,
                "qualification": "paper defines its extreme regime as 2--3 bits per parameter",
            },
        ),
        _competitor(
            "lc_qat_2bit",
            "vector_qat",
            "https://arxiv.org/abs/2606.10531",
            (2.0,),
            {
                "kind": "exact_weight_only_rate",
                "rates_bpw": [2.0],
                "qualification": "2-bit weight-only VQ-QAT; all non-weight bytes remain in the all-in bill",
            },
            claim_scopes=trained_scopes,
        ),
        _competitor(
            "matquant_qat",
            "nested_precision",
            "https://arxiv.org/abs/2502.06786",
            (4.0, 3.0, 2.0),
            {
                "kind": "trained_and_interpolated_integer_rates",
                "trained_rates_bpw": [8.0, 4.0, 2.0],
                "interpolated_rates_bpw": [6.0, 3.0],
                "qualification": "3-bit is explicitly interpolation; it must be measured, never assumed",
            },
            claim_scopes=trained_scopes,
        ),
        _competitor(
            "matgptq_ptq",
            "nested_precision",
            "https://arxiv.org/abs/2602.03537",
            (4.0, 3.0),
            {
                "kind": "implementation_support_vs_direct_quality_evidence",
                "implementation_supported_integer_rates_bpw": [8.0, 6.0, 4.0, 3.0, 2.0],
                "uniform_direct_quality_rates_bpw": [8.0, 6.0, 4.0, 3.0],
                "uniform_optimized_rates_bpw": [8.0, 4.0, 3.0],
                "heterogeneous_minimum_reported_average_bpw": 2.5,
                "qualification": "2-bit kernel/support is not direct 2-bit quality evidence",
            },
            exclusions={
                "2.0": (
                    "not mandatory as a sourced MatGPTQ quality control: direct uniform PTQ quality "
                    "evidence begins at 3 bits and the reported heterogeneous floor is 2.5 average bpw"
                )
            },
            notes="A separately measured 2-bit MatGPTQ extension may be diagnostic, never cited as sourced proof.",
        ),
        _competitor(
            "btc_llm_binary_codebook",
            "binary_pattern_codebook",
            "https://arxiv.org/abs/2506.12040",
            (1.0, 0.8),
            {
                "kind": "reported_closed_extreme_compression_range",
                "minimum_bpw": 0.7,
                "maximum_bpw": 1.11,
                "explicit_reported_rate_bpw": 0.8,
                "qualification": "codebook, transform, scales, and indices remain in the all-in bill",
            },
            source_model_evidence={
                "families": ["LLaMA", "Qwen", "FBI-LLM"],
                "maximum_reported_parameters_b": 65.0,
                "qualification": "other families/scales require a signed compatibility determination",
            },
            exclusions={
                "0.55": "below the paper's reported 0.7-bpw floor",
                "0.5": "below the paper's reported 0.7-bpw floor",
                "0.33": "below the paper's reported 0.7-bpw floor",
                "0.25": "below the paper's reported 0.7-bpw floor",
                "0.1": "below the paper's reported 0.7-bpw floor",
            },
        ),
        _competitor(
            "dbf_double_binary_factorization",
            "binary_factorization",
            "https://arxiv.org/abs/2505.11076",
            (2.0, 1.0),
            {
                "kind": "fine_grained_reported_range",
                "minimum_bpw": 1.0,
                "maximum_bpw": 2.0,
                "qualification": "intermediate rank controls rate; both binary factors and scales are billed",
            },
            source_model_evidence={
                "full_model_parameter_range_b": [7.0, 8.0],
                "qualification": "larger-matrix error studies are not full-model evidence",
            },
            source_release_status="paper_reports_public_code_verify_commit_before_run",
        ),
        _competitor(
            "multi_envelope_dbf",
            "binary_factorization",
            "https://arxiv.org/abs/2512.24545",
            (2.0, 1.0),
            {
                "kind": "reported_weight_rate_range",
                "minimum_bpw": 1.0,
                "maximum_bpw": 1.5,
                "qualification": "magnitude envelopes and protected layers are included in physical bytes",
            },
            source_model_evidence={
                "full_model_parameter_range_b": [0.6, 8.0],
                "qualification": "outside-scale use requires direct reproduction or incompatibility receipt",
            },
        ),
        _competitor(
            "bwla_w1a6_ptq",
            "binary_transform_low_rank",
            "https://arxiv.org/abs/2605.00422",
            (2.0, 1.0),
            {
                "kind": "one_bit_weight_class_with_measured_overhead",
                "nominal_weight_bits": 1.0,
                "activation_bits_example": 6,
                "reported_actual_weight_bpw_range": [1.15, 1.19],
                "qualification": "OKT/PSP low-rank and metadata prevent treating W1 as 1.0 all-in bpw",
            },
            source_model_evidence={
                "maximum_reported_parameters_b": 70.0,
                "explicit_model": "Qwen3-32B",
                "qualification": "reasoning and all-in whole-artifact quality must be remeasured",
            },
            extension_rates=(1.0,),
            exclusions={
                "0.8": "below reported actual weight rate; requires a new directly measured extension",
                "0.55": "below reported actual weight rate",
                "0.5": "below reported actual weight rate",
                "0.33": "below reported actual weight rate",
                "0.25": "below reported actual weight rate",
                "0.1": "below reported actual weight rate",
            },
        ),
        _competitor(
            "qmoe_switch_subbit",
            "moe_compression",
            "https://arxiv.org/abs/2310.16795",
            (1.0, 0.8),
            {
                "kind": "exact_historical_moe_point",
                "rates_bpw": [0.8],
                "qualification": "0.8-bpw QMoE may compare at 1.0 or 0.8, never at a smaller byte budget",
            },
            model_classes=("mixture_of_experts",),
            source_model_evidence={
                "architecture": "SwitchTransformer-c2048",
                "parameters_b": 1600.0,
                "artifact_bytes_upper_bound": 160_000_000_000,
                "qualification": "historical SwitchTransformer evidence is not modern generative-MoE proof",
            },
            source_release_status="paper_reports_public_source_and_compressed_models_verify_commit_before_run",
            exclusions={
                "0.55": "the source-supported artifact is approximately 0.8 bpw",
                "0.5": "the source-supported artifact is approximately 0.8 bpw",
                "0.33": "the source-supported artifact is approximately 0.8 bpw",
                "0.25": "the source-supported artifact is approximately 0.8 bpw",
                "0.1": "the source-supported artifact is approximately 0.8 bpw",
            },
        ),
        _competitor(
            "spear_token_gated_compensation",
            "conditional_error_compensation",
            "https://arxiv.org/abs/2606.11244",
            (4.0,),
            {
                "kind": "exact_base_quantizer_rate",
                "base_weight_bits": 4.0,
                "reported_additional_model_memory_fraction_max": 0.01,
                "qualification": "gates, compensators, scheduler state, and synchronization are billed",
            },
            claim_scopes=trained_scopes,
            control_kind="conditional_runtime_repair",
            source_model_evidence={
                "qualification": "W4 recovery precedent only; no sub-bit computation-collapse evidence",
            },
            notes="If the all-in W4+EC artifact exceeds the 4-bpw ceiling, issue incompatibility and narrow the claim.",
        ),
        _competitor(
            "marr_module_adaptive_residual",
            "residual_reconstruction",
            "https://arxiv.org/abs/2605.17997",
            (4.0, 3.0, 2.0),
            {
                "kind": "closed_low_bit_range",
                "minimum_bpw": 2.0,
                "maximum_bpw": 4.0,
                "qualification": "module coefficients and residual state count toward all-in bytes",
            },
            claim_scopes=trained_scopes,
            control_kind="static_residual_repair",
            source_model_evidence={
                "model_classes": ["large_language_model", "vision_transformer"],
                "qualification": "source reports <=4-bit gains, not sub-bit collapse reversal",
            },
            source_release_status="paper_promises_code_upon_acceptance_require_availability_receipt",
        ),
        _competitor(
            "lbllm_three_stage_distillation",
            "binary_distillation",
            "https://arxiv.org/abs/2604.19167",
            (3.0, 2.0),
            {
                "kind": "declared_weight_activation_format",
                "weight_format": "W(1+1)",
                "activation_bits": 4,
                "comparison_label": "W2A4",
                "qualification": "bitmaps, quantization parameters, and activation factors are billed",
            },
            claim_scopes=trained_scopes,
            control_kind="distilled_representation",
            source_model_evidence={
                "training_tokens_b": 0.016,
                "qualification": "teacher and data authority must match the claim scope",
            },
            extension_rates=(2.0,),
            notes="The 3-bpw lane admits a verified <=3 all-in artifact; 2-bpw requires an actual <=2 receipt.",
        ),
        _competitor(
            "scalebits_mixed_precision",
            "mixed_precision_allocation",
            "https://arxiv.org/abs/2602.17698",
            (4.0, 3.0, 2.0),
            {
                "kind": "reported_effective_rates_plus_budgeted_extension",
                "reported_effective_rates_bpw": [3.1, 2.3, 2.1],
                "search_bitwidth_set": [1, 2, 3, 4, 5, 6, 7, 8],
                "qualification": "2.0 is a new budgeted run; 2.1 tables are not 2.0 evidence",
            },
            extension_rates=(2.0,),
            source_model_evidence={
                "families": ["Llama-2", "Llama-3", "Gemma-2"],
                "maximum_reported_parameters_b": 70.0,
                "qualification": "mixed-precision allocation and RTN backend are reproduced together",
            },
        ),
        _competitor(
            "shannon_ans_lossless_wrapper",
            "lossless_entropy_packaging",
            "https://arxiv.org/abs/2606.15789",
            RATE_POINTS,
            {
                "kind": "input_format_conditional_entropy_rate",
                "source_formats": ["bf16", "int4", "AWQ", "SQ8"],
                "shannon_gap_bits_per_symbol_range": [0.01, 0.1],
                "qualification": "achieved rate is measured per wrapped artifact; entropy is not capability",
            },
            control_kind="lossless_packaging_wrapper",
            source_model_evidence={
                "parameter_range_b": [1.5, 405.0],
                "explicit_systems": ["Qwen-14B", "Mixtral-176B"],
                "qualification": "wrapped weights must decode bit-exactly and meet the candidate byte ceiling",
            },
            notes=(
                "Run on every compatible baseline format; if measured entropy cannot meet a lane budget, "
                "record incompatibility. Never credit lossless packing as capability restoration."
            ),
        ),
        _competitor(
            "nanoquant_reference",
            "binary_factorization",
            "https://arxiv.org/abs/2602.06694",
            (1.0, 0.8, 0.55, 0.5, 0.33, 0.25, 0.1),
            {
                "kind": "reported_plus_configurable_subbit",
                "reported_rates_bpw": [1.0, 0.8, 0.55],
                "qualification": "rates below 0.55 are campaign extensions and require direct evidence",
            },
            claim_scopes=trained_scopes,
            extension_rates=(0.5, 0.33, 0.25, 0.1),
        ),
        _competitor(
            "littlebit_reference",
            "latent_binary_factorization",
            "https://arxiv.org/abs/2506.13771",
            (1.0, 0.8, 0.55, 0.5, 0.33, 0.25, 0.1),
            {
                "kind": "subbit_with_extreme_point",
                "minimum_bpw": 0.1,
                "maximum_bpw": 1.0,
                "explicit_extreme_rate_bpw": 0.1,
                "qualification": "each Hawking rate still requires a direct same-parent packed artifact",
            },
            claim_scopes=trained_scopes,
        ),
        _competitor(
            "littlebit2_reference",
            "latent_binary_factorization",
            "https://arxiv.org/abs/2603.00042",
            (1.0, 0.8, 0.55, 0.5, 0.33, 0.25, 0.1),
            {
                "kind": "closed_subbit_range",
                "minimum_bpw": 0.1,
                "maximum_bpw": 1.0,
                "qualification": "paper claims the 1--0.1 bpp regime; every exact v5 rate is remeasured",
            },
            claim_scopes=trained_scopes,
        ),
        _competitor(
            "hawking_prior_source_bound_champion",
            "internal_prior_champion",
            "local:best-prior-source-bound-artifact",
            RATE_POINTS,
            {
                "kind": "campaign_receipt_bound",
                "qualification": "only an existing same-parent receipt at no greater all-in bytes is applicable",
            },
            notes="This is additive to public methods and cannot satisfy a public implementation obligation.",
        ),
    ]
    families = [
        {
            "family_id": family_id,
            "implementation_ids": [
                row["implementation_id"]
                for row in implementations
                if row["family_id"] == family_id
            ],
            "coverage_rule": "run_every_applicable_direct_implementation_not_one_family_representative",
        }
        for family_id in dict.fromkeys(row["family_id"] for row in implementations)
    ]
    rate_coverage = {
        str(rate): [
            row["implementation_id"]
            for row in implementations
            if rate in row["required_ladder_rates_bpw"]
        ]
        for rate in RATE_POINTS
    }
    return {
        "ladder_rates_bpw": list(RATE_POINTS),
        "direct_implementations": implementations,
        "required_families": families,
        "required_implementation_ids_by_rate_bpw": rate_coverage,
        "coverage_policy": {
            "all_applicable_direct_implementations_required": True,
            "one_arbitrary_competitor_never_satisfies_coverage": True,
            "family_representative_substitution_forbidden": True,
            "dynamic_strongest_same_parent_same_byte_artifact_additional": True,
            "dynamic_artifact_does_not_replace_named_implementations": True,
            "external_reported_scores_cannot_replace_reproduction": True,
            "unavailable_source_or_parent_requires_narrowed_claim_and_open_gap": True,
            "no_unmeasured_rate_interpolation": True,
            "same_parent": True,
            "same_or_lower_all_in_physical_bytes": True,
            "same_data_and_teacher_authority_within_claim_scope": True,
            "same_prompt_scorer_and_test_time_compute": True,
            "same_augmentation_scope": True,
            "every_method_requires_applicability_resolution_receipt": True,
            "incompatibility_requires_signed_receipt_and_narrowed_claim": True,
            "silent_incompatible_or_unavailable_omission_forbidden": True,
        },
        "registry_sha256": RECEIPT_PLACEHOLDER,
    }


def _vaults() -> list[dict[str, Any]]:
    owner_roles = {
        "shadow": "firewalled_shadow_owner",
        "frozen_final": "frozen_evaluation_owner",
        "sealed_final": "sealed_final_owner",
        "independent_replication": "independent_replication_owner",
    }
    rows = []
    for split in contract.DATA_SPLITS:
        optimizer_access = split not in {
            "shadow", "frozen_final", "sealed_final", "independent_replication"
        }
        rows.append(
            {
                "id": split,
                "manifest_sha256": RECEIPT_PLACEHOLDER,
                "optimizer_access": optimizer_access,
                "candidate_selector_access": optimizer_access,
                "owner_role": owner_roles.get(split, "campaign_data_owner"),
                "owner_receipt_sha256": RECEIPT_PLACEHOLDER,
                "access_log_receipt_sha256": RECEIPT_PLACEHOLDER,
            }
        )
    return rows


def compile_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "mode": "manifest_only_no_evaluation",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "quality_priority": "pure capability and quality; speed claims deferred",
        "claim_scopes": list(contract.CLAIM_SCOPES),
        "capability_domains": list(contract.CAPABILITY_DOMAINS),
        "suites": copy.deepcopy(list(SUITES)),
        "forbidden_primary_suites": copy.deepcopy(FORBIDDEN_PRIMARY),
        "vaults": _vaults(),
        "contamination_firewall": {
            "exact_hash_exclusion": True,
            "normalized_ngram_minhash_exclusion": True,
            "semantic_neighbor_exclusion": True,
            "paraphrase_translation_mutation_exclusion": True,
            "teacher_query_log_exclusion": True,
            "retrieval_index_exclusion": True,
            "tool_trace_and_verifier_cache_exclusion": True,
            "public_benchmark_membership_scan": True,
            "train_calibration_repair_and_selection_union_scanned": True,
            "all_derived_mutations_inherit_source_split": True,
            "contamination_scan_code_and_thresholds_frozen_before_final": True,
            "positive_and_negative_scan_receipts_required": True,
            "suspected_overlap_quarantined_not_adjudicated_by_optimizer": True,
            "sealed_final_created_or_revealed_only_after_candidate_freeze": True,
            "lineage_consumes_sealed_final_once": True,
            "promotion_service_returns_item_labels": False,
            "promotion_service_returns_item_outputs": False,
            "promotion_service_returns_aggregate_and_gate_only": True,
            "independent_replication_uses_distinct_owner_and_prompt_vault": True,
            "firewall_receipt_sha256": RECEIPT_PLACEHOLDER,
        },
        "matched_test_time_compute": {
            "protocol_sha256": RECEIPT_PLACEHOLDER,
            "freeze_before_final": True,
            "match_parent_candidate_and_every_required_competitor": True,
            "match_within_claim_scope_and_rate": True,
            "required_fields": {
                "identity_and_runtime": [
                    "artifact_sha256",
                    "tokenizer_sha256",
                    "chat_template_sha256",
                    "system_prompt_sha256",
                    "runtime_build_sha256",
                    "numeric_semantics",
                    "context_window_limit",
                ],
                "input": [
                    "input_text_sha256",
                    "input_token_count",
                    "max_input_tokens",
                    "truncation_policy",
                    "context_packing_policy",
                    "attachment_and_modality_budget",
                ],
                "output": [
                    "max_output_tokens",
                    "minimum_output_tokens",
                    "stop_sequence_sha256",
                    "response_format_sha256",
                    "invalid_output_policy",
                ],
                "sampling": [
                    "samples_per_item",
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "typical_p",
                    "repetition_penalty",
                    "presence_penalty",
                    "frequency_penalty",
                    "sampling_seed_schedule_sha256",
                ],
                "reasoning": [
                    "reasoning_mode",
                    "reasoning_effort",
                    "max_reasoning_tokens",
                    "hidden_reasoning_tokens_count_toward_budget",
                    "self_consistency_samples",
                ],
                "verification": [
                    "verifier_identity_sha256",
                    "verifier_retries",
                    "verifier_token_budget",
                    "verifier_timeout_seconds",
                    "answer_revision_rounds",
                ],
                "retrieval": [
                    "retrieval_enabled",
                    "retrieval_calls",
                    "retrieval_corpus_snapshot_sha256",
                    "retrieval_index_sha256",
                    "embedder_sha256",
                    "reranker_sha256",
                    "retrieval_top_k",
                    "retrieved_tokens",
                    "retrieved_bytes",
                ],
                "tools": [
                    "tools_enabled",
                    "tool_schema_sha256",
                    "tool_calls",
                    "tool_result_tokens",
                    "tool_result_bytes",
                    "tool_timeout_seconds",
                    "tool_failure_policy",
                ],
                "external_models": [
                    "external_models_enabled",
                    "external_model_identity_sha256",
                    "external_model_calls",
                    "external_model_input_tokens",
                    "external_model_output_tokens",
                    "external_model_samples",
                ],
                "state_and_speculation": [
                    "persistent_state_snapshot_sha256",
                    "prompt_cache_policy",
                    "kv_cache_precision",
                    "speculative_decoding_enabled",
                    "draft_model_sha256",
                    "speculative_verifier_sha256",
                    "accepted_output_semantics",
                ],
                "resource_and_failure": [
                    "timeout_seconds",
                    "memory_limit_bytes",
                    "concurrency",
                    "network_access_policy",
                    "oom_policy",
                    "timeout_policy",
                ],
            },
            "oom_timeout_invalid_output": "count_as_failure_never_drop",
            "missing_or_unmatched_field_invalidates_comparison": True,
            "augmented_system_bills_every_external_byte_token_call_and_retry": True,
            "standalone_scopes_require_zero_retrieval_tools_external_models_and_persistent_external_state": True,
            "quality_curves": [
                "quality_vs_input_tokens",
                "quality_vs_output_tokens",
                "quality_vs_reasoning_tokens",
                "quality_vs_samples",
                "quality_vs_tool_calls",
                "quality_vs_retrieval_calls",
                "quality_vs_external_model_calls",
                "quality_vs_timeout",
            ],
        },
        "statistics": {
            "confidence": 0.95,
            "familywise_alpha_max": 0.05,
            "multiple_testing": "holm_or_closed_testing",
            "multiple_testing_family": (
                "all protected domains x primary endpoints x required direct competitor "
                "implementations x candidate lineages claimed from the sealed evaluation"
            ),
            "intersection_union_required_for_uniform_dominance": True,
            "selection_aware_holdout_required": True,
            "sealed_results_cannot_select_or_retune_candidate": True,
            "preregistration_receipt_sha256": RECEIPT_PLACEHOLDER,
            "binary_accuracy": "exact_paired_mcnemar",
            "continuous_generation": "paired_hierarchical_cluster_bootstrap",
            "nll": "document_cluster_bootstrap_not_token_bootstrap",
            "code": "task_bootstrap_with_identical_n_k",
            "generative": "item_by_generation_seed_cluster",
            "training_generalization": "model_family_by_independent_training_seed",
            "minimum_independent_training_seeds": 5,
            "minimum_independent_calibration_draws": 5,
            "minimum_generation_seeds_per_stochastic_item": 5,
            "deterministic_codec_replication_unit": (
                "independent calibration draw, ordering, initialization, and packing receipt"
            ),
            "seed_and_calibration_draw_crossing_reported_not_pooled_as_independent": True,
            "pre_power_each_primary_domain": True,
            "inconclusive_when_underpowered": True,
            "failed_runs_and_nan_scores_retained": True,
        },
        "competitor_coverage": _competitor_coverage(),
        "dominance": {
            "primary_summary": "worst_domain_normalized_retention_and_capability_CVaR",
            "parent_noninferiority": "simultaneous LCB(delta_domain) >= -preregistered_margin for all domains",
            "frontier_champion": (
                "noninferior all domains and LCB > delta in >=1 primary domain versus every "
                "applicable named implementation and dynamic matched-budget control"
            ),
            "uniform_quality_dominance": (
                "simultaneous multiplicity-corrected LCB > 0 in every protected domain versus "
                "every applicable named implementation and dynamic matched-budget control"
            ),
            "weighted_average_secondary_only": True,
            "no_scale_extrapolation": True,
            "no_missing_competitor_treated_as_a_win": True,
            "sealed_final_receipt_required": True,
            "independent_replication_receipt_required": True,
            "headline_blocked_until_both_receipts_verify": True,
            "expiry_triggers": [
                "new_competitor", "benchmark_revision", "artifact_hash_change",
                "prompt_or_scorer_change", "runtime_semantics_change", "data_leakage_discovery",
            ],
        },
        "judge_policy": {
            "objective_first": True,
            "blind_candidate_identity": True,
            "both_presentation_orders": True,
            "distillation_teacher_cannot_judge": True,
            "multiple_unrelated_judges": True,
            "human_adjudicate_disagreements": True,
            "judge_results_never_override_objective_oracle": True,
        },
        "sealed_and_independent_receipts": {
            "sealed_final": {
                "required": True,
                "owner_independent_of_optimizer_and_candidate_selector": True,
                "vault_created_or_revealed_after_candidate_and_protocol_freeze": True,
                "one_time_consumption_per_candidate_lineage": True,
                "candidate_receives_no_item_labels_outputs_or_gradients": True,
                "owner_attests_no_training_calibration_repair_or_selection_access": True,
                "required_receipt_fields": [
                    "owner_identity_sha256",
                    "owner_signature_sha256",
                    "candidate_and_protocol_freeze_receipt_sha256",
                    "vault_manifest_sha256",
                    "access_log_sha256",
                    "one_time_consumption_nonce_sha256",
                    "raw_outputs_sha256",
                    "scorer_source_sha256",
                    "aggregate_report_sha256",
                ],
                "owner_receipt_sha256": RECEIPT_PLACEHOLDER,
            },
            "independent_replication": {
                "required": True,
                "owner_distinct_from_campaign_and_sealed_final_owners": True,
                "no_candidate_selection_or_optimizer_access": True,
                "distinct_post_freeze_prompt_vault": True,
                "independent_environment_and_runtime_build": True,
                "rebuild_artifact_from_source_bound_recipe": True,
                "reproduce_every_applicable_direct_competitor": True,
                "recompute_all_in_bytes_scores_intervals_and_dominance": True,
                "failed_reproduction_blocks_headline": True,
                "required_receipt_fields": [
                    "owner_identity_sha256",
                    "owner_signature_sha256",
                    "source_checkout_sha256",
                    "environment_sha256",
                    "data_vault_manifest_sha256",
                    "rebuilt_artifact_sha256",
                    "competitor_registry_sha256",
                    "raw_outputs_sha256",
                    "scorer_source_sha256",
                    "recomputed_dominance_report_sha256",
                ],
                "owner_receipt_sha256": RECEIPT_PLACEHOLDER,
            },
            "receipt_verifier_distinct_from_candidate_selector": True,
            "placeholder_is_planning_only_and_never_proof": True,
        },
    }
    payload = {key: value for key, value in manifest.items() if key not in {"manifest_sha256", "generated_at"}}
    manifest["manifest_sha256"] = contract.hash_value(payload)
    return manifest


def _is_receipt(value: Any) -> bool:
    return value == RECEIPT_PLACEHOLDER or contract.is_sha256(value)


def _receipt_slots(document: dict[str, Any]):
    """Yield only the evidence slots allowed to change after policy compilation."""

    suites = document.get("suites")
    if isinstance(suites, list):
        for index, suite in enumerate(suites):
            if not isinstance(suite, dict):
                continue
            for field in ("manifest_sha256", "prompt_protocol_sha256", "scorer_source_sha256"):
                yield f"suites[{index}].{field}", suite, field
    vaults = document.get("vaults")
    if isinstance(vaults, list):
        for index, vault in enumerate(vaults):
            if not isinstance(vault, dict):
                continue
            for field in ("manifest_sha256", "owner_receipt_sha256", "access_log_receipt_sha256"):
                yield f"vaults[{index}].{field}", vault, field
    singleton_slots = (
        ("contamination_firewall", "firewall_receipt_sha256"),
        ("matched_test_time_compute", "protocol_sha256"),
        ("statistics", "preregistration_receipt_sha256"),
        ("competitor_coverage", "registry_sha256"),
    )
    for section, field in singleton_slots:
        container = document.get(section)
        if isinstance(container, dict):
            yield f"{section}.{field}", container, field
    coverage = document.get("competitor_coverage")
    implementations = coverage.get("direct_implementations") if isinstance(coverage, dict) else None
    if isinstance(implementations, list):
        for index, implementation in enumerate(implementations):
            if not isinstance(implementation, dict):
                continue
            for field in (
                "method_source_sha256",
                "applicability_receipt_sha256",
                "packed_artifact_receipt_sha256",
                "reproduction_receipt_sha256",
                "incompatibility_receipt_sha256",
                "narrowed_claim_receipt_sha256",
            ):
                yield f"competitor_coverage.direct_implementations[{index}].{field}", implementation, field
    receipt_contract = document.get("sealed_and_independent_receipts")
    if isinstance(receipt_contract, dict):
        for owner_id in ("sealed_final", "independent_replication"):
            owner = receipt_contract.get(owner_id)
            if isinstance(owner, dict):
                yield f"sealed_and_independent_receipts.{owner_id}.owner_receipt_sha256", owner, (
                    "owner_receipt_sha256"
                )


def _policy_projection(document: dict[str, Any]) -> dict[str, Any]:
    """Normalize only timestamps and cryptographic evidence receipts.

    Every other byte is immutable policy.  In particular, a valid re-stamped
    manifest cannot change suites, vault access, rate coverage, matched compute,
    statistics, contamination controls, or proof-owner requirements.
    """

    projected = copy.deepcopy(document)
    projected.pop("generated_at", None)
    projected.pop("manifest_sha256", None)
    for _label, container, field in _receipt_slots(projected):
        if field in container and _is_receipt(container[field]):
            container[field] = RECEIPT_PLACEHOLDER
    return projected


def _first_difference(actual: Any, canonical: Any, path: str = "$") -> str:
    if type(actual) is not type(canonical):
        return f"{path} type {type(actual).__name__} != {type(canonical).__name__}"
    if isinstance(actual, dict):
        missing = sorted(set(canonical) - set(actual))
        if missing:
            return f"{path} missing key {missing[0]!r}"
        extra = sorted(set(actual) - set(canonical))
        if extra:
            return f"{path} has non-canonical key {extra[0]!r}"
        for key in canonical:
            if actual[key] != canonical[key]:
                return _first_difference(actual[key], canonical[key], f"{path}.{key}")
    elif isinstance(actual, list):
        if len(actual) != len(canonical):
            return f"{path} length {len(actual)} != {len(canonical)}"
        for index, (left, right) in enumerate(zip(actual, canonical, strict=True)):
            if left != right:
                return _first_difference(left, right, f"{path}[{index}]")
    elif actual != canonical:
        return f"{path} value {actual!r} != canonical {canonical!r}"
    return f"{path} differs"


def _valid_generated_at(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _restamp(document: dict[str, Any]) -> None:
    payload = {
        key: value
        for key, value in document.items()
        if key not in {"manifest_sha256", "generated_at"}
    }
    document["manifest_sha256"] = contract.hash_value(payload)


def validate_manifest(manifest: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    if manifest.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if manifest.get("version") != VERSION:
        errors.append(f"version must be {VERSION}")
    if manifest.get("mode") != "manifest_only_no_evaluation":
        errors.append("manifest mode must remain execution-free")
    if not _valid_generated_at(manifest.get("generated_at")):
        errors.append("generated_at must be a timezone-aware ISO-8601 timestamp")
    if manifest.get("claim_scopes") != list(contract.CLAIM_SCOPES):
        errors.append("claim-scope order mismatch")
    if manifest.get("capability_domains") != list(contract.CAPABILITY_DOMAINS):
        errors.append("capability-domain order mismatch")

    suites = manifest.get("suites")
    if not isinstance(suites, list) or not suites:
        errors.append("suites must be a non-empty array")
        suites = []
    seen: set[str] = set()
    primary_coverage: set[str] = set()
    for index, suite in enumerate(suites):
        prefix = f"suites[{index}]"
        if not isinstance(suite, dict):
            errors.append(f"{prefix} must be an object")
            continue
        suite_id = suite.get("id")
        if not isinstance(suite_id, str) or not suite_id or suite_id in seen:
            errors.append(f"{prefix}.id missing or duplicate")
            continue
        seen.add(suite_id)
        domains = suite.get("domains")
        if not isinstance(domains, list) or not domains or any(
            domain not in contract.CAPABILITY_DOMAINS for domain in domains
        ):
            errors.append(f"{suite_id}: invalid domains")
        if suite.get("role") in {"sealed_primary", "frozen_primary", "independent_primary"}:
            primary_coverage.update(domains if isinstance(domains, list) else [])
        if not isinstance(suite.get("primary_source"), str) or not suite["primary_source"]:
            errors.append(f"{suite_id}: primary source missing")
        if not isinstance(suite.get("cluster_unit"), str) or not suite["cluster_unit"]:
            errors.append(f"{suite_id}: cluster unit missing")
    missing = set(contract.CAPABILITY_DOMAINS) - primary_coverage
    if missing:
        errors.append("primary suite coverage missing domains: " + ", ".join(sorted(missing)))

    for label, container, field in _receipt_slots(manifest):
        if field not in container or not _is_receipt(container.get(field)):
            errors.append(f"{label} must be 'required' or a SHA-256 receipt")

    vaults = manifest.get("vaults")
    vault_ids = [row.get("id") for row in vaults if isinstance(row, dict)] \
        if isinstance(vaults, list) else []
    if vault_ids != list(contract.DATA_SPLITS):
        errors.append("vault order and coverage must exactly match the v5 data splits")

    compute = manifest.get("matched_test_time_compute")
    required_compute_fields = compute.get("required_fields") if isinstance(compute, dict) else None
    if not isinstance(required_compute_fields, dict):
        errors.append("matched test-time-compute field registry is missing")
    else:
        flattened = [
            field
            for group in required_compute_fields.values()
            if isinstance(group, list)
            for field in group
        ]
        must_include = {
            "input_token_count", "max_input_tokens", "max_output_tokens", "temperature",
            "max_reasoning_tokens", "timeout_seconds", "samples_per_item", "tool_calls",
            "retrieval_calls", "external_model_calls", "verifier_retries",
            "sampling_seed_schedule_sha256", "retrieval_index_sha256", "tool_schema_sha256",
        }
        if len(flattened) != len(set(flattened)) or not must_include <= set(flattened):
            errors.append("matched test-time-compute fields are duplicated or incomplete")

    statistics = manifest.get("statistics")
    if not isinstance(statistics, dict) or any((
        statistics.get("minimum_independent_training_seeds") != 5,
        statistics.get("minimum_independent_calibration_draws") != 5,
        statistics.get("minimum_generation_seeds_per_stochastic_item") != 5,
        statistics.get("multiple_testing") != "holm_or_closed_testing",
        statistics.get("intersection_union_required_for_uniform_dominance") is not True,
        statistics.get("pre_power_each_primary_domain") is not True,
    )):
        errors.append("statistical replication, calibration, power, or multiplicity contract incomplete")

    coverage = manifest.get("competitor_coverage")
    implementations = coverage.get("direct_implementations") if isinstance(coverage, dict) else None
    implementation_ids = [
        row.get("implementation_id") for row in implementations if isinstance(row, dict)
    ] if isinstance(implementations, list) else []
    if not implementation_ids or len(implementation_ids) != len(set(implementation_ids)):
        errors.append("direct competitor implementation registry is empty or duplicated")
    missing_central = set(CENTRAL_DIRECT_IMPLEMENTATIONS) - set(implementation_ids)
    if missing_central:
        errors.append(
            "central direct competitor obligations missing: " + ", ".join(sorted(missing_central))
        )
    rate_coverage = coverage.get("required_implementation_ids_by_rate_bpw") \
        if isinstance(coverage, dict) else None
    if not isinstance(rate_coverage, dict):
        errors.append("direct competitor rate coverage is missing")
    else:
        for rate in RATE_POINTS:
            ids = rate_coverage.get(str(rate))
            public_ids = [value for value in ids or [] if value != "hawking_prior_source_bound_champion"]
            if not isinstance(ids, list) or len(public_ids) < 2 or any(
                value not in implementation_ids for value in ids
            ):
                errors.append(f"rate {rate} lacks at least two named direct public implementations")
    matgptq = next(
        (
            row for row in implementations or []
            if isinstance(row, dict) and row.get("implementation_id") == "matgptq_ptq"
        ),
        None,
    )
    if not isinstance(matgptq, dict) \
            or matgptq.get("required_ladder_rates_bpw") != [4.0, 3.0] \
            or "2.0" not in matgptq.get("explicitly_excluded_ladder_rates_bpw", {}):
        errors.append("MatGPTQ coverage must distinguish 4/3-bit quality evidence from 2-bit support")
    qmoe = next(
        (row for row in implementations or [] if isinstance(row, dict)
         and row.get("implementation_id") == "qmoe_switch_subbit"),
        None,
    )
    spear = next(
        (row for row in implementations or [] if isinstance(row, dict)
         and row.get("implementation_id") == "spear_token_gated_compensation"),
        None,
    )
    if not isinstance(qmoe, dict) or qmoe.get("applicable_model_classes") != [
        "mixture_of_experts"
    ] or qmoe.get("required_ladder_rates_bpw") != [1.0, 0.8]:
        errors.append("QMoE must remain MoE-only and bound to its 0.8-bpw-compatible lanes")
    if not isinstance(spear, dict) or spear.get("required_ladder_rates_bpw") != [4.0] \
            or spear.get("control_kind") != "conditional_runtime_repair":
        errors.append("SPEAR must remain a four-bit conditional-repair control")

    owner_contract = manifest.get("sealed_and_independent_receipts")
    sealed_owner = owner_contract.get("sealed_final") if isinstance(owner_contract, dict) else None
    independent_owner = owner_contract.get("independent_replication") \
        if isinstance(owner_contract, dict) else None
    if not isinstance(sealed_owner, dict) or not isinstance(independent_owner, dict) or any((
        sealed_owner.get("required") is not True,
        sealed_owner.get("owner_independent_of_optimizer_and_candidate_selector") is not True,
        independent_owner.get("required") is not True,
        independent_owner.get("owner_distinct_from_campaign_and_sealed_final_owners") is not True,
        independent_owner.get("reproduce_every_applicable_direct_competitor") is not True,
    )):
        errors.append("sealed-final or independent-owner receipt contract is incomplete")

    # The schema checks above make failures legible; this exact projection is
    # the security boundary.  A malicious edit followed by a valid restamp must
    # still differ from the fresh, code-owned canonical policy.
    canonical = compile_manifest()
    projected = _policy_projection(manifest)
    canonical_projected = _policy_projection(canonical)
    if projected != canonical_projected:
        errors.append(
            "immutable canonical v5 policy mismatch: "
            + _first_difference(projected, canonical_projected)
        )

    expected = manifest.get("manifest_sha256")
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_sha256", "generated_at"}
    }
    if not contract.is_sha256(expected) or expected != contract.hash_value(payload):
        errors.append("manifest_sha256 missing or mismatched")
    return sorted(set(errors))


def selftest() -> int:
    manifest = compile_manifest()
    assert validate_manifest(manifest) == []
    assert set(contract.CAPABILITY_DOMAINS) <= {
        domain
        for suite in manifest["suites"]
        if suite["role"] in {"sealed_primary", "frozen_primary", "independent_primary"}
        for domain in suite["domains"]
    }
    assert set(CENTRAL_DIRECT_IMPLEMENTATIONS) <= {
        row["implementation_id"]
        for row in manifest["competitor_coverage"]["direct_implementations"]
    }

    # An un-restamped policy edit fails the ordinary identity check.
    damaged = copy.deepcopy(manifest)
    damaged["suites"][0]["cluster_unit"] = "token"
    assert "manifest_sha256 missing or mismatched" in validate_manifest(damaged)

    # Red-team the stronger boundary: every mutation is re-stamped with a valid
    # manifest hash and must *still* fail canonical-policy validation.
    def assert_restamped_rejected(label: str, edit) -> None:
        candidate = copy.deepcopy(manifest)
        edit(candidate)
        _restamp(candidate)
        failures = validate_manifest(candidate)
        assert any("immutable canonical v5 policy mismatch" in failure for failure in failures), (
            label,
            failures,
        )

    assert_restamped_rejected("delete suite", lambda value: value["suites"].pop())
    assert_restamped_rejected(
        "change suite scoring",
        lambda value: value["suites"][0].__setitem__("scoring", "token average only"),
    )
    assert_restamped_rejected(
        "open sealed vault",
        lambda value: next(
            row for row in value["vaults"] if row["id"] == "sealed_final"
        ).__setitem__("optimizer_access", True),
    )
    assert_restamped_rejected(
        "delete temperature matching",
        lambda value: value["matched_test_time_compute"]["required_fields"]["sampling"].remove(
            "temperature"
        ),
    )
    assert_restamped_rejected(
        "lower training seeds",
        lambda value: value["statistics"].__setitem__("minimum_independent_training_seeds", 1),
    )
    assert_restamped_rejected(
        "lower calibration draws",
        lambda value: value["statistics"].__setitem__("minimum_independent_calibration_draws", 1),
    )
    assert_restamped_rejected(
        "disable multiple testing",
        lambda value: value["statistics"].__setitem__("multiple_testing", "none"),
    )
    assert_restamped_rejected(
        "weaken contamination",
        lambda value: value["contamination_firewall"].__setitem__(
            "semantic_neighbor_exclusion", False
        ),
    )
    assert_restamped_rejected(
        "remove contamination control",
        lambda value: value["contamination_firewall"].pop("teacher_query_log_exclusion"),
    )
    assert_restamped_rejected(
        "weaken independent owner",
        lambda value: value["sealed_and_independent_receipts"]["independent_replication"].__setitem__(
            "owner_distinct_from_campaign_and_sealed_final_owners", False
        ),
    )
    assert_restamped_rejected(
        "remove sealed receipt field",
        lambda value: value["sealed_and_independent_receipts"]["sealed_final"][
            "required_receipt_fields"
        ].pop(),
    )
    assert_restamped_rejected(
        "delete named competitor",
        lambda value: value["competitor_coverage"]["direct_implementations"].pop(0),
    )
    assert_restamped_rejected(
        "substitute one competitor",
        lambda value: value["competitor_coverage"].__setitem__(
            "direct_implementations",
            [value["competitor_coverage"]["direct_implementations"][0]],
        ),
    )
    assert_restamped_rejected(
        "misstate MatGPTQ as sourced 2-bit quality evidence",
        lambda value: next(
            row
            for row in value["competitor_coverage"]["direct_implementations"]
            if row["implementation_id"] == "matgptq_ptq"
        )["required_ladder_rates_bpw"].append(2.0),
    )
    assert_restamped_rejected(
        "silently omit BTC-LLM",
        lambda value: value["competitor_coverage"]["direct_implementations"].remove(next(
            row
            for row in value["competitor_coverage"]["direct_implementations"]
            if row["implementation_id"] == "btc_llm_binary_codebook"
        )),
    )
    assert_restamped_rejected(
        "broaden QMoE to dense without evidence",
        lambda value: next(
            row
            for row in value["competitor_coverage"]["direct_implementations"]
            if row["implementation_id"] == "qmoe_switch_subbit"
        )["applicable_model_classes"].append("dense"),
    )
    assert_restamped_rejected(
        "silently omit SPEAR incompatibility path",
        lambda value: next(
            row
            for row in value["competitor_coverage"]["direct_implementations"]
            if row["implementation_id"] == "spear_token_gated_compensation"
        )["resolution_contract"].__setitem__("silent_omission_forbidden", False),
    )
    assert_restamped_rejected(
        "misclassify Shannon ANS as capability repair",
        lambda value: next(
            row
            for row in value["competitor_coverage"]["direct_implementations"]
            if row["implementation_id"] == "shannon_ans_lossless_wrapper"
        ).__setitem__("control_kind", "capability_repair"),
    )
    assert_restamped_rejected(
        "add unknown policy escape hatch",
        lambda value: value.__setitem__("allow_policy_override", True),
    )

    # Concrete cryptographic receipts are the sole mutable policy projection.
    receiptized = copy.deepcopy(manifest)
    receiptized["suites"][0]["manifest_sha256"] = "a" * 64
    receiptized["vaults"][0]["owner_receipt_sha256"] = "b" * 64
    receiptized["matched_test_time_compute"]["protocol_sha256"] = "c" * 64
    receiptized["competitor_coverage"]["direct_implementations"][0][
        "reproduction_receipt_sha256"
    ] = "d" * 64
    receiptized["sealed_and_independent_receipts"]["sealed_final"][
        "owner_receipt_sha256"
    ] = "e" * 64
    _restamp(receiptized)
    assert validate_manifest(receiptized) == []

    invalid_receipt = copy.deepcopy(manifest)
    invalid_receipt["matched_test_time_compute"]["protocol_sha256"] = "trust-me"
    _restamp(invalid_receipt)
    assert any("protocol_sha256" in error for error in validate_manifest(invalid_receipt))

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "quality.json"
        contract.atomic_json(path, manifest)
        assert validate_manifest(json.loads(path.read_text())) == []
    print("quality_battery_v5.py selftest OK (19 restamped policy tamper cases rejected)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser("compile")
    compile_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("path", type=Path)
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "selftest":
        return selftest()
    if args.command == "compile":
        manifest = compile_manifest()
        contract.atomic_json(args.output, manifest)
        print(json.dumps({
            "output": str(args.output),
            "manifest_sha256": manifest["manifest_sha256"],
            "suite_count": len(manifest["suites"]),
            "capability_domain_count": len(manifest["capability_domains"]),
        }, indent=2))
        return 0
    manifest = json.loads(args.path.read_text())
    errors = validate_manifest(manifest)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
