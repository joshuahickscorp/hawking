#!/usr/bin/env python3.12
"""Concrete fusion operation for the ceremony-free base Frankenstein harness.

This module states the *only* admissible cross-architecture operation for
Kimi-K3 / GLM-5.2 inheritance into a DeepSeek-V4-Flash body.  It is deliberately
honest about what is impossible:

  weight-average / direct splice of mismatched tensors  →  IMPOSSIBLE
  direct donor weight transplant into student slots     →  IMPOSSIBLE (contract)
  loss-fitted residual adapter (SGD/Adam / backprop)    →  EXCLUDED (training)

The admissible op is **training-free weight-space transfer** (see
``frankenstein_transfer``): SVD/PCA math-subspace extraction from GLM weights,
closed-form GLM→DeepSeek projection, fixed residual steering + router bias,
sealed as a reversible residual module.  No two donor bodies are ever resident
together; the DeepSeek body is read-only.

The historical ``loss_target`` / fitted-adapter path is retained only as a
deprecated reference describing what *would* require the student forward; it is
not executed by the harness.
"""
from __future__ import annotations

from typing import Any, Mapping


# --- pinned geometries (source-bound; not inferred from secondary claims) ---

DEEPSEEK_V4_FLASH = {
    "family": "student_body",
    "repository": "deepseek-ai/DeepSeek-V4-Flash",
    "revision": "60d8d70770c6776ff598c94bb586a859a38244f1",
    "model_type": "deepseek_v4",
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "n_shared_experts": 1,
    "vocab_size": 129280,
    "source_torch_dtype": "bfloat16",
    "moe_intermediate_size": 2048,
    "q_lora_rank": 1024,
    "o_lora_rank": 1024,
    "rms_norm_eps": 1e-6,
}

KIMI_K3 = {
    "family": "strategic_donor",
    "repository": "moonshotai/Kimi-K3",
    "revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
    "model_type": "kimi_k3",
    "text_model_type": "kimi_linear",
    "hidden_size": 7168,
    "num_hidden_layers": 93,
    # MoE expert counts are null in the admitted metadata; do not invent them.
    "n_routed_experts": None,
    "num_experts_per_tok": None,
    "source_torch_dtype": "bfloat16",
    "weight_shard_count": 96,
    "weight_shard_bytes": 1_560_936_091_448,
}

GLM_5_2 = {
    "family": "mathematical_donor",
    "repository": "zai-org/GLM-5.2",
    "revision": "b4734de4facf877f85769a911abafc5283eab3d9",
    "model_type": "glm_moe_dsa",
    "architecture": "GlmMoeDsaForCausalLM",
    "hidden_size": 6144,
    "num_hidden_layers": 78,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "vocab_size": 154_880,
    "source_torch_dtype": "bfloat16",
    "moe_intermediate_size": 2048,
    "q_lora_rank": 2048,
}

# Bridge / transplant contracts (v3 child baseline).
BRIDGE_INPUT_SHAPE = ["batch", "sequence", 4096]  # student residual stream
BRIDGE_OUTPUT_SHAPE = ["batch", "sequence", 4096]
BRIDGE_DTYPE = "bfloat16"
BRIDGE_RMS_NORM_EPS = 1e-6

TRANSPLANT_POINT_NAMES: tuple[str, ...] = (
    "pre_norm_hidden_state",
    "post_attention_hidden_state",
    "pre_router_hidden_state",
    "router_logits",
    "selected_expert_ids",
    "route_probabilities_and_margins",
    "post_moe_hidden_state",
    "mhc_state",
    "attention_index_state",
    "final_hidden_state",
    "lm_head_logits",
    "hcli_tool_action_decision",
)

BRIDGES: tuple[str, ...] = (
    "KIMI_STRATEGIC_BRIDGE",
    "GLM_MATH_BRIDGE",
)

# DEPRECATED: loss weights for the residual-adapter *fit* path.
# Training-free transfer does not use these.  Kept so historical receipts and
# unit tests that document the excluded path remain readable.
DEFAULT_LOSS_WEIGHTS = {
    "mse": 1.0,
    "cosine": 0.1,
    "route_kl": 0.05,  # only when transplant point is route-related
}

LOSS_FIT_PATH_STATUS = "DEPRECATED_EXCLUDED_TRAINING_PATH"
TRAINING_FREE_METHOD = "weight_space_gram_pca_closed_form_projection_steering"

ROUTE_RELATED_POINTS = frozenset(
    {
        "router_logits",
        "selected_expert_ids",
        "route_probabilities_and_margins",
    }
)

FORWARD_GATE = "DEEPSEEK_FORWARD_PENDING"


def shape_mismatches() -> list[dict[str, Any]]:
    """Exact geometry conflicts that forbid tensor-mean / weight splice."""

    rows: list[dict[str, Any]] = []
    fields = (
        "hidden_size",
        "num_hidden_layers",
        "n_routed_experts",
        "num_experts_per_tok",
        "vocab_size",
        "q_lora_rank",
        "model_type",
    )
    models = {
        "deepseek_v4_flash": DEEPSEEK_V4_FLASH,
        "kimi_k3": KIMI_K3,
        "glm_5_2": GLM_5_2,
    }
    for field in fields:
        values = {name: model.get(field) for name, model in models.items()}
        distinct = {v for v in values.values() if v is not None}
        if len(distinct) > 1 or any(v is None for v in values.values()):
            rows.append(
                {
                    "field": field,
                    "values": values,
                    "compatible_for_elementwise_mean": False,
                    "reason": (
                        "missing or unequal across bodies; no shared tensor layout"
                        if any(v is None for v in values.values())
                        else "unequal; elementwise mean / splice is undefined"
                    ),
                }
            )
    return rows


def impossible_operations() -> list[dict[str, Any]]:
    """Operations a caller might expect that are invalid as stated."""

    mismatches = shape_mismatches()
    return [
        {
            "name": "stream_portion_and_average_weights",
            "verdict": "IMPOSSIBLE_AS_STATED",
            "reason": (
                "Kimi-K3 (H=7168, L=93), GLM-5.2 (H=6144, L=78), and DeepSeek-V4-Flash "
                "(H=4096, L=43) do not share tensor shapes, layer counts, tokenizers, "
                "or MoE layouts. Elementwise mean of misaligned tensors is undefined."
            ),
            "mismatches": mismatches,
        },
        {
            "name": "direct_weight_transplant",
            "verdict": "IMPOSSIBLE_AS_STATED",
            "reason": (
                "DSV4F latent-bridge and transplant-point contracts set "
                "direct_weight_transplant=false; donor weights may not replace "
                "student parameters."
            ),
            "contract_fields": {
                "direct_weight_transplant": False,
                "future_adapter_requirement": (
                    "separately sealed reversible adapter or policy head; "
                    "no direct donor weight replacement"
                ),
            },
        },
        {
            "name": "hold_two_donors_resident",
            "verdict": "PROHIBITED_BY_DISK_CONTRACT",
            "reason": (
                "Working-set invariant: DeepSeek body (read-only) + at most ONE donor "
                "bounded window + current output block. Kimi-K3 alone is ~1.56 TB."
            ),
        },
    ]


def projection_shape(*, donor: str) -> dict[str, Any]:
    """Linear projection from donor hidden width into student residual stream."""

    if donor == "kimi_k3":
        in_features = KIMI_K3["hidden_size"]
        donor_repo = KIMI_K3["repository"]
    elif donor == "glm_5_2":
        in_features = GLM_5_2["hidden_size"]
        donor_repo = GLM_5_2["repository"]
    else:
        raise ValueError(f"unknown donor family: {donor!r}")
    out_features = DEEPSEEK_V4_FLASH["hidden_size"]
    return {
        "name": f"project_{donor}_to_student",
        "weight_shape": [in_features, out_features],
        "bias_shape": [out_features],
        "dtype": BRIDGE_DTYPE,
        "parameter_count": in_features * out_features + out_features,
        "bytes_bf16": 2 * (in_features * out_features + out_features),
        "donor_repository": donor_repo,
        "student_repository": DEEPSEEK_V4_FLASH["repository"],
        "math": "a_d_prime = a_d @ W_proj + b_proj  # a_d: [B,S,H_d] → [B,S,4096]",
    }


def residual_adapter_shape() -> dict[str, Any]:
    """Per-(bridge, transplant_point, student_layer) residual adapter."""

    h = DEEPSEEK_V4_FLASH["hidden_size"]
    return {
        "name": "reversible_residual_adapter",
        "weight_shape": [h, h],
        "bias_shape": [h],
        "dtype": BRIDGE_DTYPE,
        "parameter_count": h * h + h,
        "bytes_bf16": 2 * (h * h + h),
        "apply": "a_out = a_s + A(a_s)  # residual; A is zero-init at start",
        "bridge_io": {
            "input_tensor_state": {
                "name": "per_token_hidden_state",
                "shape_contract": list(BRIDGE_INPUT_SHAPE),
                "source_dtype": BRIDGE_DTYPE,
            },
            "output_tensor_state": {
                "name": "reversible_residual_adapter_output",
                "shape_contract": list(BRIDGE_OUTPUT_SHAPE),
                "source_dtype": BRIDGE_DTYPE,
            },
        },
    }


def layer_map(*, donor: str, student_layer: int) -> dict[str, Any]:
    """Monotone map from a student layer index onto a donor layer index.

    This is a *schedule* map only (which donor window to stream).  It is not a
    claim that the layers compute the same function.
    """

    if student_layer < 0 or student_layer >= DEEPSEEK_V4_FLASH["num_hidden_layers"]:
        raise ValueError(f"student_layer out of range: {student_layer}")
    if donor == "kimi_k3":
        donor_layers = KIMI_K3["num_hidden_layers"]
    elif donor == "glm_5_2":
        donor_layers = GLM_5_2["num_hidden_layers"]
    else:
        raise ValueError(f"unknown donor family: {donor!r}")
    # Inclusive last-layer targeting: student L-1 maps to donor L_d-1.
    denom = DEEPSEEK_V4_FLASH["num_hidden_layers"] - 1
    mapped = round(student_layer * (donor_layers - 1) / denom) if denom else 0
    return {
        "student_layer": student_layer,
        "donor": donor,
        "donor_layer": int(mapped),
        "student_layers": DEEPSEEK_V4_FLASH["num_hidden_layers"],
        "donor_layers": donor_layers,
        "map": "round(student_layer * (donor_L-1) / (student_L-1))",
    }


def loss_target(
    *,
    transplant_point: str,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """DEPRECATED training-path description (not executed by the harness).

    Retained for contract documentation and tests that assert the historical
    shape of a loss that *would* require student activations.  The live path is
    ``frankenstein_transfer`` (closed-form only).
    """

    if transplant_point not in TRANSPLANT_POINT_NAMES:
        raise ValueError(f"unknown transplant point: {transplant_point!r}")
    w = dict(DEFAULT_LOSS_WEIGHTS)
    if weights is not None:
        w.update({k: float(v) for k, v in weights.items()})
    route = transplant_point in ROUTE_RELATED_POINTS
    terms = [
        {
            "name": "mse_projected_donor",
            "weight": w["mse"],
            "formula": "|| (a_s + A(a_s)) - a_d_prime ||_2^2 / (B*S*H)",
            "shapes": {
                "a_s": BRIDGE_INPUT_SHAPE,
                "a_d_prime": BRIDGE_INPUT_SHAPE,
                "A_out": BRIDGE_OUTPUT_SHAPE,
            },
        },
        {
            "name": "cosine_alignment",
            "weight": w["cosine"],
            "formula": "1 - cos(a_s + A(a_s), a_d_prime)  # mean over tokens",
        },
    ]
    if route:
        terms.append(
            {
                "name": "route_kl",
                "weight": w["route_kl"],
                "formula": (
                    "KL(softmax(route_s) || stopgrad(mapped_donor_route)) "
                    "when both student and donor expose comparable top-k routes; "
                    "else term is zeroed and recorded as UNAVAILABLE"
                ),
                "note": (
                    "Kimi expert counts are null in admission metadata; route KL "
                    "for Kimi is unavailable until architecture facts are bound."
                ),
            }
        )
    return {
        "transplant_point": transplant_point,
        "status": "DEFINED_BUT_GATED_ON_STUDENT_FORWARD",
        "deprecated": True,
        "path_status": LOSS_FIT_PATH_STATUS,
        "executed_by_harness": False,
        "forward_gate": FORWARD_GATE,
        "normalization": {
            "source_config_rms_norm_eps": BRIDGE_RMS_NORM_EPS,
            "runtime_calibration": "NOT_MEASURED_NO_43_LAYER_RUNTIME",
        },
        "token_alignment": (
            "one source-token position to one adapter position after a shared "
            "prompt corpus is tokenized with each model's own tokenizer; "
            "cross-tokenizer positions are matched by prompt-id + char-span, "
            "not by raw token id"
        ),
        "terms": terms,
        "direct_weight_transplant": False,
        "replacement": TRAINING_FREE_METHOD,
    }


def training_free_operation_spec() -> dict[str, Any]:
    """Executable description of the training-free transfer (live path)."""

    h_g = int(GLM_5_2["hidden_size"])
    h_s = int(DEEPSEEK_V4_FLASH["hidden_size"])
    return {
        "name": TRAINING_FREE_METHOD,
        "verdict": "REAL_AND_MINIMAL_TRAINING_FREE",
        "summary": (
            "Extract a low-rank math subspace from GLM weight matrices via Gram "
            "PCA/SVD (weight-only, no GLM runtime).  Map GLM→DeepSeek with a "
            "closed-form isometric embedding of that subspace into H=4096.  "
            "Seal a reversible rank-1 residual steering module + router bias at "
            "v3 transplant points.  No gradient descent, no loss fit."
        ),
        "excluded": [
            "gradient_descent",
            "backprop",
            "optimizer_steps",
            "loss_minimization_loops",
            "loss_fitted_residual_adapter",
        ],
        "steps": [
            "stream_glm_math_weight_tensors",
            "accumulate_hidden_gram",
            "top_r_eigenspace_svd_pca",
            "closed_form_projection_B_E_T",
            "steering_vector_from_top_singular_direction",
            "router_bias_from_expert_frobenius_energy",
            "seal_reversible_residual_module",
            "structural_apply_reference_body_read_only",
            "evict_working_windows",
        ],
        "shapes": {
            "glm_hidden": h_g,
            "deepseek_hidden": h_s,
            "projection_weight": [h_g, h_s],
            "residual_A": [h_s, h_s],
            "router_bias": [int(DEEPSEEK_V4_FLASH["n_routed_experts"])],
        },
        "forward_gate": FORWARD_GATE,
        "forward_gate_blocks": [
            "math_bench_measurement",
            "paired_activation_procrustes_refinement",
        ],
        "forward_gate_does_not_block": [
            "weight_subspace_extraction",
            "closed_form_projection",
            "module_seal",
            "structural_apply",
        ],
        "capability_status_default": "UNVALIDATED_WEIGHT_ONLY_DERIVED",
        "implementation": "lab.operators.frankenstein_transfer",
    }


def block_id(*, bridge: str, transplant_point: str, student_layer: int) -> str:
    if bridge not in BRIDGES:
        raise ValueError(f"unknown bridge: {bridge!r}")
    if transplant_point not in TRANSPLANT_POINT_NAMES:
        raise ValueError(f"unknown transplant point: {transplant_point!r}")
    if student_layer < 0 or student_layer >= DEEPSEEK_V4_FLASH["num_hidden_layers"]:
        raise ValueError(f"student_layer out of range: {student_layer}")
    return f"{bridge}__{transplant_point}__L{student_layer:02d}"


def donor_for_bridge(bridge: str) -> str:
    if bridge == "KIMI_STRATEGIC_BRIDGE":
        return "kimi_k3"
    if bridge == "GLM_MATH_BRIDGE":
        return "glm_5_2"
    raise ValueError(f"unknown bridge: {bridge!r}")


def fusion_operation_spec() -> dict[str, Any]:
    """Full executable description of the minimal correct fusion op.

    Primary path is training-free (``training_free``).  The historical
    distillation name is retained for schedule/harness compatibility; the
    ``fit_residual_adapter`` step is bypassed in favour of
    ``frankenstein_transfer``.
    """

    adapter = residual_adapter_shape()
    training_free = training_free_operation_spec()
    return {
        "name": "block_wise_streaming_distillation_via_latent_bridge",
        "verdict": "REAL_AND_MINIMAL",
        "summary": (
            "Keep the DeepSeek-V4-Flash body as the executable student.  Live "
            "path: training-free GLM math-subspace extraction + closed-form "
            "projection + reversible residual steering (no loss fit).  For each "
            "schedule block: stream one donor window (or reuse weight-only "
            "module), seal raw residual module (no gravity), evict donor window."
        ),
        "primary_method": training_free["name"],
        "training_free": training_free,
        "loss_fit_path": {
            "status": LOSS_FIT_PATH_STATUS,
            "executed_by_harness": False,
            "note": (
                "α‖·‖² + β cos + γ KL residual-adapter fit is excluded.  "
                "loss_target() remains as documentation only."
            ),
        },
        "impossible": impossible_operations(),
        "geometries": {
            "deepseek_v4_flash": DEEPSEEK_V4_FLASH,
            "kimi_k3": KIMI_K3,
            "glm_5_2": GLM_5_2,
        },
        "shape_mismatches": shape_mismatches(),
        "bridge_contract_citation": {
            "path_hint": (
                "workspace/campaign/records/runs/deepseek-v4/child-baseline-v3/"
                "DSV4F_LATENT_BRIDGE_CONTRACT.json"
            ),
            "schema": "hawking.gravity.deepseek_v4.latent_bridge_contract.v3",
            "status": "DSV4F_FUTURE_BRIDGE_INTERFACES_V3_DECLARED_NO_DONOR_INHERITANCE",
            "input_shape": list(BRIDGE_INPUT_SHAPE),
            "output_shape": list(BRIDGE_OUTPUT_SHAPE),
            "dtype": BRIDGE_DTYPE,
            "direct_weight_transplant": False,
            "loss_target_in_contract": "NOT_DEFINED_NO_DONOR_TRAINING_IN_THIS_LANE",
            "loss_target_defined_here": True,
            "loss_target_executed": False,
        },
        "transplant_points_citation": {
            "path_hint": (
                "workspace/campaign/records/runs/deepseek-v4/child-baseline-v3/"
                "DSV4F_TRANSPLANT_POINTS.json"
            ),
            "schema": "hawking.gravity.deepseek_v4.transplant_points.v3",
            "status": "DSV4F_TRANSPLANT_POINTS_V3_FROZEN_SOURCE_BOUND_NO_WEIGHT_GRAFT",
            "points": list(TRANSPLANT_POINT_NAMES),
            "point_shape_contract": list(BRIDGE_INPUT_SHAPE),
        },
        "projections": {
            "kimi_k3": projection_shape(donor="kimi_k3"),
            "glm_5_2": projection_shape(donor="glm_5_2"),
        },
        "residual_adapter": adapter,
        "per_block_lifecycle": [
            "disk_floor_check",
            "stream_one_donor_window_or_weight_only_module",
            "verify_range_identity_and_provenance",
            "training_free_subspace_project_steer_or_record_PENDING",
            "student_forward_validation_or_DEEPSEEK_FORWARD_PENDING",
            "seal_output_block_raw_no_gravity",
            "evict_donor_window_and_scratch",
            "append_progress_cursor",
        ],
        "forward_gate": FORWARD_GATE,
        "output_artifact": {
            "kind": "raw_base_frankenstein_adapter_archive",
            "gravity_compressed": False,
            "contains_merged_donor_weights": False,
            "contains_student_body": False,
            "student_body_reference": "read-only DeepSeek-V4 full-43-layer-stream.gravity",
            "blocks": (
                "training-free reversible residual module + per-schedule raw blocks"
            ),
        },
        "example_loss": loss_target(transplant_point="post_moe_hidden_state"),
        "example_layer_map": {
            "kimi_k3_L0": layer_map(donor="kimi_k3", student_layer=0),
            "kimi_k3_L42": layer_map(donor="kimi_k3", student_layer=42),
            "glm_5_2_L0": layer_map(donor="glm_5_2", student_layer=0),
            "glm_5_2_L42": layer_map(donor="glm_5_2", student_layer=42),
        },
    }


def estimated_adapter_archive_bytes(
    *,
    bridges: tuple[str, ...] = BRIDGES,
    points: tuple[str, ...] = TRANSPLANT_POINT_NAMES,
    layers: int | None = None,
) -> dict[str, Any]:
    """Upper bound on raw adapter archive size (bf16 dense residuals + projections)."""

    layer_count = DEEPSEEK_V4_FLASH["num_hidden_layers"] if layers is None else layers
    adapter = residual_adapter_shape()
    per_adapter = adapter["bytes_bf16"]
    block_count = len(bridges) * len(points) * layer_count
    adapters_total = per_adapter * block_count
    projections_total = sum(
        projection_shape(donor=d)["bytes_bf16"] for d in ("kimi_k3", "glm_5_2")
    )
    # Metadata / receipts budget: 64 MiB envelope.
    metadata = 64 * 1024 * 1024
    return {
        "block_count": block_count,
        "per_adapter_bytes_bf16": per_adapter,
        "adapters_total_bytes": adapters_total,
        "projections_total_bytes": projections_total,
        "metadata_budget_bytes": metadata,
        "archive_upper_bound_bytes": adapters_total + projections_total + metadata,
        "note": (
            "Upper bound assumes a full dense H×H residual per (bridge, point, layer). "
            "Production may sparsify or rank-reduce; schedule must still budget the bound."
        ),
    }
