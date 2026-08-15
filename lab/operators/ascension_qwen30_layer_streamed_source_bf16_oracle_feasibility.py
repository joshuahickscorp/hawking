"""Fail-closed working-set feasibility gate for a streamed Qwen30 source oracle.

The existing source-BF16 oracle contract correctly refuses to resident-load
the entire 56.9 GiB source model on the measured machine.  This preparer asks
a narrower, deliberately non-executing question: could a future *layer-
streamed* implementation keep only declared BF16 row windows plus its exact
369-token-prefix/forced-token state in memory instead?  It derives the
required windows from the sealed source contract and measured memory snapshot.

This module never opens a source tensor payload (including a safetensors
shard), loads a model, uses an accelerator, starts a server, or invokes HCLI.
It cannot prove that a future range reader has source-equivalent numerical
semantics.  Consequently it only reports a prepared feasibility model when a
separate, sealed exact-semantics attestation is supplied, and it *never*
authorizes an oracle execution, reports source quality, or acts as a
benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.receipts import SealIntegrityError, seal, verify


SCHEMA = "hawking.ascension.qwen30_layer_streamed_source_bf16_final_logit_oracle_feasibility.v1"
PREPARED_STATUS = "PREPARED_QWEN30_LAYER_STREAMED_SOURCE_BF16_ORACLE_FEASIBILITY_NOT_EXECUTED"
REFUSED_STATUS = "REFUSED_QWEN30_LAYER_STREAMED_SOURCE_BF16_ORACLE_FEASIBILITY_UNSAFE_OR_UNPROVEN"

SOURCE_CONTRACT_SCHEMA = "hawking.ascension.qwen30_hq30gr2_source_bf16_three_way_final_logit_contract.v1"
SOURCE_CONTRACT_STATUS = "PREPARED_SOURCE_BF16_THREE_WAY_FINAL_LOGIT_DISTANCE_CONTRACT_NOT_RUN"
WHOLE_MODEL_PREFLIGHT_SCHEMA = "hawking.ascension.qwen30_hq30gr2_source_bf16_memory_lease_preflight.v1"
WHOLE_MODEL_PREFLIGHT_READY_STATUS = "PREPARED_STRICT_SOURCE_BF16_MEMORY_LEASE_PREFLIGHT_NO_LEASE_GRANTED"
WHOLE_MODEL_PREFLIGHT_BLOCKED_STATUS = (
    "BLOCKED_STRICT_SOURCE_BF16_MEMORY_LEASE_PREFLIGHT_INSUFFICIENT_RECLAIMABLE_HEADROOM"
)
SEMANTICS_SCHEMA = "hawking.ascension.qwen30_layer_streamed_source_bf16_exact_semantics_attestation.v1"
SEMANTICS_STATUS = "EARNED_QWEN30_LAYER_STREAMED_SOURCE_BF16_EXACT_SEMANTICS_ATTESTED"

PREFIX_TOKENS = 369
FORCED_TOKEN_ID = 949
TRACE_TOKENS = PREFIX_TOKENS + 1
LAYER_COUNT = 48
VOCAB_ROWS = 151_936
HIDDEN_SIZE = 2_048
ATTENTION_HEADS = 32
KV_HEADS = 4
HEAD_DIM = 128
EXPERT_COUNT = 128
TOP_K = 8
MOE_INTERMEDIATE = 768
SOURCE_TENSOR_COUNT = 18_867
BF16_BYTES = 2
F32_BYTES = 4
U32_BYTES = 4
MIN_ROW_TILE_ROWS = 1
MAX_ROW_TILE_ROWS = 128
MIN_BACKEND_ALLOCATOR_RESERVE_BYTES = 64 * 1024**2
MIN_SAFETY_MARGIN_BYTES = 1024**3

DEFAULT_POLICY = {
    "row_tile_rows": 128,
    "max_simultaneous_expert_bodies": 1,
    "activation_element_bytes": F32_BYTES,
    "kv_cache_element_bytes": BF16_BYTES,
    "attention_score_element_bytes": F32_BYTES,
    "backend_allocator_reserve_bytes": 128 * 1024**2,
    "minimum_unallocated_safety_margin_bytes": MIN_SAFETY_MARGIN_BYTES,
}


class LayerStreamedOracleFeasibilityError(RuntimeError):
    """Inputs cannot support a truthful source-BF16 streamed feasibility report."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sealed(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        checked = verify(raw, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise LayerStreamedOracleFeasibilityError(f"{label} is absent or invalid: {exc}") from exc
    if not isinstance(checked, Mapping):
        raise LayerStreamedOracleFeasibilityError(f"{label} must be an object")
    return dict(checked)


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LayerStreamedOracleFeasibilityError(f"{label} must be an object")
    return dict(value)


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise LayerStreamedOracleFeasibilityError(f"{label} must be a non-empty string")
    if sha256 and (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
        raise LayerStreamedOracleFeasibilityError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LayerStreamedOracleFeasibilityError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise LayerStreamedOracleFeasibilityError(f"{label} must be boolean")
    return value


def _require(value: object, *, label: str) -> None:
    if value is not True:
        raise LayerStreamedOracleFeasibilityError(f"{label} must be true")


def _source_contract(path: Path) -> dict[str, Any]:
    contract = _sealed(path, label="source-BF16 three-way contract")
    if contract.get("schema") != SOURCE_CONTRACT_SCHEMA or contract.get("status") != SOURCE_CONTRACT_STATUS:
        raise LayerStreamedOracleFeasibilityError("source-BF16 three-way contract schema/status drifted")
    exact = _object(contract.get("exact_input"), label="source-BF16 exact input")
    if _integer(exact.get("source_template_token_count"), label="source prefix token count") != PREFIX_TOKENS:
        raise LayerStreamedOracleFeasibilityError("source contract does not bind the required 369-token prefix")
    if _integer(exact.get("forced_identical_continuation_token_id"), label="forced token") != FORCED_TOKEN_ID:
        raise LayerStreamedOracleFeasibilityError("source contract does not bind forced token 949")
    _require(
        exact.get("source_must_execute_the_same_369_token_prefix_then_the_forced_token"),
        label="source exact prefix/forced ordering",
    )
    _require(exact.get("sampling_or_autoregressive_feedback_is_forbidden"), label="source no-sampling rule")
    token_hash = _text(exact.get("source_template_token_ids_u32le_sha256"), label="source token hash", sha256=True)
    evidence = _object(contract.get("evidence"), label="source-BF16 evidence")
    config = _object(evidence.get("source_config"), label="source config evidence")
    index = _object(evidence.get("source_index"), label="source tensor index evidence")
    source_weight_bytes = _integer(
        evidence.get("source_weight_bytes_exact"), label="source exact weight bytes", minimum=1
    )
    resource = _object(contract.get("resource_and_capture_requirements"), label="source resource requirements")
    if _integer(
        resource.get("source_weights_static_lower_bound_bytes"),
        label="source static lower bound",
        minimum=1,
    ) != source_weight_bytes:
        raise LayerStreamedOracleFeasibilityError("source exact bytes and static lower bound differ")
    _require(
        resource.get("source_model_has_not_been_loaded_by_this_preflight"),
        label="source contract no-model-load boundary",
    )
    return {
        "path": str(path.resolve()),
        "document_sha256": _sha256_file(path),
        "seal_sha256": _text(contract.get("seal_sha256"), label="source contract seal", sha256=True),
        "source_config_sha256": _text(config.get("sha256"), label="source config SHA", sha256=True),
        "source_index_sha256": _text(index.get("sha256"), label="source tensor index SHA", sha256=True),
        "source_weight_bytes": source_weight_bytes,
        "source_template_token_ids_u32le_sha256": token_hash,
    }


def _whole_model_snapshot(path: Path, *, source: Mapping[str, Any]) -> dict[str, Any]:
    preflight = _sealed(path, label="whole-model source-BF16 memory preflight")
    if preflight.get("schema") != WHOLE_MODEL_PREFLIGHT_SCHEMA:
        raise LayerStreamedOracleFeasibilityError("whole-model memory preflight schema drifted")
    if preflight.get("status") not in (
        WHOLE_MODEL_PREFLIGHT_READY_STATUS,
        WHOLE_MODEL_PREFLIGHT_BLOCKED_STATUS,
    ):
        raise LayerStreamedOracleFeasibilityError("whole-model memory preflight status is not recognized")
    contract_ref = _object(
        preflight.get("source_bf16_three_way_contract"), label="whole-model source contract reference"
    )
    preflight_source_seal = _text(
        contract_ref.get("seal_sha256"), label="whole-model source contract seal", sha256=True
    )
    if preflight_source_seal != source["seal_sha256"]:
        raise LayerStreamedOracleFeasibilityError("whole-model preflight is bound to a different source contract")
    snapshot = _object(preflight.get("measured_system_snapshot"), label="whole-model measured system snapshot")
    vm = _object(snapshot.get("vm_stat"), label="whole-model vm snapshot")
    swap = _object(snapshot.get("swap"), label="whole-model swap snapshot")
    assessment = _object(preflight.get("headroom_assessment"), label="whole-model headroom assessment")
    reclaimable = _integer(vm.get("reclaimable_bytes"), label="measured reclaimable bytes")
    swap_used = _integer(swap.get("used_bytes"), label="measured swap bytes")
    preflight_source_bytes = _integer(
        assessment.get("source_weight_bytes"), label="preflight source weight bytes", minimum=1
    )
    if preflight_source_bytes != source["source_weight_bytes"]:
        raise LayerStreamedOracleFeasibilityError("whole-model preflight source-byte binding differs")
    whole_model_required = _integer(
        assessment.get("minimum_reclaimable_bytes_required_before_source_load"),
        label="whole-model required reclaimable bytes",
        minimum=source["source_weight_bytes"],
    )
    if whole_model_required < source["source_weight_bytes"]:
        raise LayerStreamedOracleFeasibilityError("whole-model requirement is below the source weights")
    return {
        "path": str(path.resolve()),
        "document_sha256": _sha256_file(path),
        "seal_sha256": _text(preflight.get("seal_sha256"), label="whole-model preflight seal", sha256=True),
        "status": _text(preflight.get("status"), label="whole-model preflight status"),
        "reclaimable_bytes": reclaimable,
        "swap_used_bytes": swap_used,
        "physical_memory_bytes": _integer(
            snapshot.get("physical_memory_bytes"), label="measured physical memory", minimum=1
        ),
        "whole_model_required_bytes": whole_model_required,
        "whole_model_deficit_bytes": max(0, whole_model_required - reclaimable),
    }


def _policy(raw: Mapping[str, Any]) -> dict[str, int]:
    row_tile_rows = _integer(raw.get("row_tile_rows"), label="streamed row tile rows", minimum=MIN_ROW_TILE_ROWS)
    if row_tile_rows > MAX_ROW_TILE_ROWS:
        raise LayerStreamedOracleFeasibilityError(
            f"streamed row tile rows must be <= {MAX_ROW_TILE_ROWS} to retain the declared working-set bound"
        )
    max_experts = _integer(
        raw.get("max_simultaneous_expert_bodies"), label="maximum simultaneous expert bodies", minimum=1
    )
    if max_experts != 1:
        raise LayerStreamedOracleFeasibilityError(
            "only one simultaneous expert body is modeled; a wider expert residency needs a new bounded plan"
        )
    activation_bytes = _integer(raw.get("activation_element_bytes"), label="activation element bytes", minimum=1)
    kv_bytes = _integer(raw.get("kv_cache_element_bytes"), label="KV cache element bytes", minimum=1)
    score_bytes = _integer(raw.get("attention_score_element_bytes"), label="attention score element bytes", minimum=1)
    if activation_bytes != F32_BYTES or score_bytes != F32_BYTES or kv_bytes != BF16_BYTES:
        raise LayerStreamedOracleFeasibilityError(
            "the modeled exact source-BF16 plan requires F32 activation/score scratch and BF16 KV cache"
        )
    allocator_reserve = _integer(
        raw.get("backend_allocator_reserve_bytes"),
        label="backend allocator reserve",
        minimum=MIN_BACKEND_ALLOCATOR_RESERVE_BYTES,
    )
    safety_margin = _integer(
        raw.get("minimum_unallocated_safety_margin_bytes"),
        label="minimum unallocated safety margin",
        minimum=MIN_SAFETY_MARGIN_BYTES,
    )
    return {
        "row_tile_rows": row_tile_rows,
        "max_simultaneous_expert_bodies": max_experts,
        "activation_element_bytes": activation_bytes,
        "kv_cache_element_bytes": kv_bytes,
        "attention_score_element_bytes": score_bytes,
        "backend_allocator_reserve_bytes": allocator_reserve,
        "minimum_unallocated_safety_margin_bytes": safety_margin,
    }


def _semantics(
    path: Path | None,
    *,
    source: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, Any], bool]:
    if path is None:
        return (
            dict(DEFAULT_POLICY),
            {
                "present": False,
                "semantic_equivalence_proven": False,
                "missing_requirements": [
                    "sealed exact source-BF16 range-reader semantics attestation",
                    "bounded no-whole-shard-cache evidence",
                    "exact 369-token-prefix then forced-949 cache replay evidence",
                    "exact source operator and accumulation-order evidence",
                ],
            },
            False,
        )
    attestation = _sealed(path, label="layer-streamed source-BF16 semantics attestation")
    if attestation.get("schema") != SEMANTICS_SCHEMA or attestation.get("status") != SEMANTICS_STATUS:
        raise LayerStreamedOracleFeasibilityError("semantics attestation schema/status drifted")
    binding = _object(attestation.get("source_binding"), label="semantics source binding")
    required_bindings = {
        "source_config_sha256": source["source_config_sha256"],
        "source_index_sha256": source["source_index_sha256"],
        "source_template_token_ids_u32le_sha256": source["source_template_token_ids_u32le_sha256"],
    }
    for field, expected in required_bindings.items():
        if _text(binding.get(field), label=f"semantics {field}", sha256=True) != expected:
            raise LayerStreamedOracleFeasibilityError(f"semantics attestation {field} differs from source contract")
    semantics_source_bytes = _integer(
        binding.get("source_weight_bytes"), label="semantics source weight bytes", minimum=1
    )
    if semantics_source_bytes != source["source_weight_bytes"]:
        raise LayerStreamedOracleFeasibilityError("semantics attestation source bytes differ from source contract")
    semantics_tensor_count = _integer(
        binding.get("source_tensor_count"), label="semantics source tensor count", minimum=1
    )
    if semantics_tensor_count != SOURCE_TENSOR_COUNT:
        raise LayerStreamedOracleFeasibilityError("semantics attestation source tensor count differs from Qwen30")
    trace = _object(attestation.get("exact_trace"), label="semantics exact trace")
    if _integer(trace.get("prefix_token_count"), label="semantics prefix token count") != PREFIX_TOKENS:
        raise LayerStreamedOracleFeasibilityError("semantics attestation prefix length differs")
    if _integer(trace.get("forced_token_id"), label="semantics forced token") != FORCED_TOKEN_ID:
        raise LayerStreamedOracleFeasibilityError("semantics attestation forced token differs")
    _require(trace.get("prefill_then_forced_cache_order"), label="semantics prefill/forced cache order")
    _require(trace.get("sampling_or_autoregressive_feedback_forbidden"), label="semantics no-sampling rule")
    checks = _object(attestation.get("exact_semantics"), label="semantics exactness checks")
    for field in (
        "source_bf16_tensor_row_order_and_offsets_verified",
        "range_reader_never_maps_or_caches_a_complete_source_shard",
        "source_rmsnorm_rope_attention_router_topk_moe_operator_order_verified",
        "source_accumulation_and_expert_combine_order_verified",
        "all_48_layers_and_final_norm_head_verified",
        "full_f32_final_logits_at_prefix_and_forced_endpoints_verified",
    ):
        _require(checks.get(field), label=f"semantics {field}")
    policy = _policy(_object(attestation.get("working_set_policy"), label="semantics working-set policy"))
    return (
        policy,
        {
            "present": True,
            "path": str(path.resolve()),
            "document_sha256": _sha256_file(path),
            "seal_sha256": _text(attestation.get("seal_sha256"), label="semantics seal", sha256=True),
            "semantic_equivalence_proven": True,
            "missing_requirements": [],
        },
        True,
    )


def _tensor_window(
    *,
    role: str,
    tensor: str,
    shape: tuple[int, ...],
    rows: int | None = None,
    selected_expert: bool = False,
) -> dict[str, Any]:
    if rows is None:
        rows = shape[0]
    trailing = math.prod(shape[1:]) if len(shape) > 1 else 1
    return {
        "role": role,
        "source_tensor": tensor,
        "full_shape": list(shape),
        "source_dtype": "bf16",
        "row_window_shape": [rows, *shape[1:]],
        "source_bf16_window_bytes": rows * trailing * BF16_BYTES,
        "row_range_access_only": True,
        "selected_expert_only": selected_expert,
    }


def _per_layer_windows(policy: Mapping[str, int]) -> list[dict[str, Any]]:
    tile = policy["row_tile_rows"]
    result: list[dict[str, Any]] = []
    for layer in range(LAYER_COUNT):
        prefix = f"model.layers.{layer}"
        tensors = [
            _tensor_window(
                role="input_rmsnorm", tensor=f"{prefix}.input_layernorm.weight", shape=(HIDDEN_SIZE,)
            ),
            _tensor_window(
                role="q_rmsnorm", tensor=f"{prefix}.self_attn.q_norm.weight", shape=(HEAD_DIM,)
            ),
            _tensor_window(
                role="k_rmsnorm", tensor=f"{prefix}.self_attn.k_norm.weight", shape=(HEAD_DIM,)
            ),
            _tensor_window(
                role="q_projection",
                tensor=f"{prefix}.self_attn.q_proj.weight",
                shape=(ATTENTION_HEADS * HEAD_DIM, HIDDEN_SIZE),
                rows=tile,
            ),
            _tensor_window(
                role="k_projection",
                tensor=f"{prefix}.self_attn.k_proj.weight",
                shape=(KV_HEADS * HEAD_DIM, HIDDEN_SIZE),
                rows=tile,
            ),
            _tensor_window(
                role="v_projection",
                tensor=f"{prefix}.self_attn.v_proj.weight",
                shape=(KV_HEADS * HEAD_DIM, HIDDEN_SIZE),
                rows=tile,
            ),
            _tensor_window(
                role="o_projection",
                tensor=f"{prefix}.self_attn.o_proj.weight",
                shape=(HIDDEN_SIZE, ATTENTION_HEADS * HEAD_DIM),
                rows=tile,
            ),
            _tensor_window(
                role="post_attention_rmsnorm",
                tensor=f"{prefix}.post_attention_layernorm.weight",
                shape=(HIDDEN_SIZE,),
            ),
            _tensor_window(
                role="router",
                tensor=f"{prefix}.mlp.gate.weight",
                shape=(EXPERT_COUNT, HIDDEN_SIZE),
                rows=EXPERT_COUNT,
            ),
            _tensor_window(
                role="selected_expert_gate_projection",
                tensor=f"{prefix}.mlp.experts.{{selected_expert_id_0_to_127}}.gate_proj.weight",
                shape=(MOE_INTERMEDIATE, HIDDEN_SIZE),
                rows=tile,
                selected_expert=True,
            ),
            _tensor_window(
                role="selected_expert_up_projection",
                tensor=f"{prefix}.mlp.experts.{{selected_expert_id_0_to_127}}.up_proj.weight",
                shape=(MOE_INTERMEDIATE, HIDDEN_SIZE),
                rows=tile,
                selected_expert=True,
            ),
            _tensor_window(
                role="selected_expert_down_projection",
                tensor=f"{prefix}.mlp.experts.{{selected_expert_id_0_to_127}}.down_proj.weight",
                shape=(HIDDEN_SIZE, MOE_INTERMEDIATE),
                rows=tile,
                selected_expert=True,
            ),
        ]
        result.append(
            {
                "source_layer_index": layer,
                "strict_operator_order": [
                    "input_rmsnorm",
                    "q_rmsnorm_then_q_projection",
                    "k_rmsnorm_then_k_projection",
                    "v_projection",
                    "rope_on_q_and_k",
                    "causal_attention_against_layer_kv_cache",
                    "o_projection_then_first_residual",
                    "post_attention_rmsnorm_then_router_top8",
                    "one_selected_expert_gate_up_activation_down_at_a_time",
                    "source_ordered_route_weighted_combine_then_second_residual",
                ],
                "top_k_routes_per_token": TOP_K,
                "max_simultaneous_expert_bodies": policy["max_simultaneous_expert_bodies"],
                "tensors": tensors,
                "kv_cache_for_forced_token": {
                    "shape_per_key_or_value": [TRACE_TOKENS, KV_HEADS, HEAD_DIM],
                    "key_and_value": True,
                    "dtype": "bf16",
                    "bytes": TRACE_TOKENS * KV_HEADS * HEAD_DIM * 2 * policy["kv_cache_element_bytes"],
                },
            }
        )
    return result


def _head_windows(policy: Mapping[str, int]) -> list[dict[str, Any]]:
    tile = policy["row_tile_rows"]
    return [
        _tensor_window(
            role="embedding_row", tensor="model.embed_tokens.weight", shape=(VOCAB_ROWS, HIDDEN_SIZE), rows=1
        ),
        _tensor_window(role="final_rmsnorm", tensor="model.norm.weight", shape=(HIDDEN_SIZE,)),
        _tensor_window(role="lm_head_all_rows", tensor="lm_head.weight", shape=(VOCAB_ROWS, HIDDEN_SIZE), rows=tile),
    ]


def _working_set(policy: Mapping[str, int], *, source_weight_bytes: int) -> dict[str, Any]:
    activation_bytes = policy["activation_element_bytes"]
    score_bytes = policy["attention_score_element_bytes"]
    kv_bytes = policy["kv_cache_element_bytes"]
    tile = policy["row_tile_rows"]
    per_layer = _per_layer_windows(policy)
    head = _head_windows(policy)
    max_source_window = max(
        item["source_bf16_window_bytes"]
        for layer in per_layer
        for item in layer["tensors"]
    )
    max_source_window = max(max_source_window, *(item["source_bf16_window_bytes"] for item in head))

    kv_cache = LAYER_COUNT * TRACE_TOKENS * KV_HEADS * HEAD_DIM * 2 * kv_bytes
    residual_buffers = 4 * TRACE_TOKENS * HIDDEN_SIZE * activation_bytes
    q_projection = TRACE_TOKENS * ATTENTION_HEADS * HEAD_DIM * activation_bytes
    kv_projection = 2 * TRACE_TOKENS * KV_HEADS * HEAD_DIM * activation_bytes
    attention_scores = ATTENTION_HEADS * TRACE_TOKENS * TRACE_TOKENS * score_bytes
    attention_context = TRACE_TOKENS * ATTENTION_HEADS * HEAD_DIM * activation_bytes
    moe_gate_up = 2 * TRACE_TOKENS * MOE_INTERMEDIATE * activation_bytes
    moe_accumulator = TRACE_TOKENS * HIDDEN_SIZE * activation_bytes
    router_logits = TRACE_TOKENS * EXPERT_COUNT * activation_bytes
    route_table = TRACE_TOKENS * TOP_K * (U32_BYTES + activation_bytes)
    retained_endpoint_logits = 2 * VOCAB_ROWS * F32_BYTES
    input_token_ids = TRACE_TOKENS * U32_BYTES
    embedding_output = TRACE_TOKENS * HIDDEN_SIZE * BF16_BYTES
    transformer_transient = (
        residual_buffers
        + q_projection
        + kv_projection
        + attention_scores
        + attention_context
        + moe_gate_up
        + moe_accumulator
        + router_logits
        + route_table
    )
    transient_peak = max(transformer_transient, embedding_output, tile * F32_BYTES)
    modeled_working_set = kv_cache + max_source_window + retained_endpoint_logits + input_token_ids + transient_peak
    with_allocator_reserve = modeled_working_set + policy["backend_allocator_reserve_bytes"]
    required_with_safety = with_allocator_reserve + policy["minimum_unallocated_safety_margin_bytes"]
    return {
        "trace_tokens": TRACE_TOKENS,
        "source_tensor_count": SOURCE_TENSOR_COUNT,
        "source_weight_bytes_whole_model_not_resident_in_this_plan": source_weight_bytes,
        "source_weight_bytes_excluded_from_streamed_working_set_only_if_range_reader_bound_holds": source_weight_bytes,
        "per_layer_windows": per_layer,
        "terminal_windows": head,
        "source_tensor_window_peak_bytes": max_source_window,
        "persistent_kv_cache_bytes": kv_cache,
        "retained_prefix_and_forced_full_f32_logit_bytes": retained_endpoint_logits,
        "input_token_id_bytes": input_token_ids,
        "transient_phase_bytes": {
            "embedding_output_bf16": embedding_output,
            "residual_buffers_f32": residual_buffers,
            "q_projection_f32": q_projection,
            "k_and_v_projection_f32": kv_projection,
            "causal_attention_score_scratch_f32": attention_scores,
            "attention_context_f32": attention_context,
            "one_expert_gate_and_up_activation_f32": moe_gate_up,
            "one_expert_output_accumulator_f32": moe_accumulator,
            "router_logits_f32": router_logits,
            "top8_route_table": route_table,
            "transformer_phase_peak": transformer_transient,
        },
        "transient_peak_bytes": transient_peak,
        "modeled_working_set_before_allocator_reserve_bytes": modeled_working_set,
        "backend_allocator_reserve_bytes": policy["backend_allocator_reserve_bytes"],
        "modeled_working_set_with_allocator_reserve_bytes": with_allocator_reserve,
        "minimum_unallocated_safety_margin_bytes": policy["minimum_unallocated_safety_margin_bytes"],
        "minimum_reclaimable_bytes_required_for_streamed_plan": required_with_safety,
        "no_whole_model_residency_is_only_a_future_range_reader_requirement": True,
    }


def build_feasibility(
    *,
    source_contract_path: Path,
    whole_model_preflight_path: Path,
    semantics_attestation_path: Path | None = None,
) -> dict[str, Any]:
    """Build a sealed metadata result; do not load source tensors or a model."""
    source = _source_contract(source_contract_path)
    snapshot = _whole_model_snapshot(whole_model_preflight_path, source=source)
    policy, semantics, semantic_equivalence_proven = _semantics(semantics_attestation_path, source=source)
    working_set = _working_set(policy, source_weight_bytes=source["source_weight_bytes"])
    available = snapshot["reclaimable_bytes"]
    required = working_set["minimum_reclaimable_bytes_required_for_streamed_plan"]
    margin_after_allocator = available - working_set["modeled_working_set_with_allocator_reserve_bytes"]
    margin_after_safety = available - required
    memory_fit = margin_after_safety >= 0
    zero_swap = snapshot["swap_used_bytes"] == 0
    safe = memory_fit and zero_swap and semantic_equivalence_proven
    refusal_reasons: list[str] = []
    if not semantic_equivalence_proven:
        refusal_reasons.extend(semantics["missing_requirements"])
    if not zero_swap:
        refusal_reasons.append("measured swap usage is non-zero")
    if not memory_fit:
        refusal_reasons.append("measured reclaimable memory is below the streamed working-set safety floor")
    return seal(
        {
            "schema": SCHEMA,
            "status": PREPARED_STATUS if safe else REFUSED_STATUS,
            "recorded_at": _utc_now(),
            "source_bf16_three_way_contract": source,
            "whole_model_memory_preflight": snapshot,
            "exact_trace": {
                "prefix_token_count": PREFIX_TOKENS,
                "forced_token_id": FORCED_TOKEN_ID,
                "total_tokens_with_forced_continuation": TRACE_TOKENS,
                "source_template_token_ids_u32le_sha256": source["source_template_token_ids_u32le_sha256"],
                "sampling_or_autoregressive_feedback_forbidden": True,
            },
            "source_geometry": {
                "layers": LAYER_COUNT,
                "hidden_size": HIDDEN_SIZE,
                "attention_heads": ATTENTION_HEADS,
                "key_value_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
                "experts": EXPERT_COUNT,
                "top_k": TOP_K,
                "moe_intermediate_size": MOE_INTERMEDIATE,
                "vocab_rows": VOCAB_ROWS,
                "source_tensor_count": SOURCE_TENSOR_COUNT,
                "source_dtype": "bf16",
            },
            "streaming_semantics_attestation": semantics,
            "working_set_policy": policy,
            "working_set": working_set,
            "memory_assessment": {
                "available_reclaimable_bytes_from_sealed_preflight": available,
                "measured_swap_used_bytes": snapshot["swap_used_bytes"],
                "available_margin_after_modeled_working_set_and_allocator_reserve_bytes": margin_after_allocator,
                "available_margin_after_required_safety_floor_bytes": margin_after_safety,
                "streamed_memory_arithmetic_fits": memory_fit,
                "zero_swap_condition_met": zero_swap,
                "whole_model_source_residency_required_bytes": snapshot["whole_model_required_bytes"],
                "whole_model_source_residency_deficit_bytes": snapshot["whole_model_deficit_bytes"],
                "whole_model_residency_block_remains_authoritative": True,
                "streaming_model_does_not_reclassify_or_override_whole_model_preflight": True,
            },
            "feasibility": {
                "semantic_equivalence_proven_by_external_sealed_attestation": semantic_equivalence_proven,
                "safe_streamed_plan_prepared_not_executed": safe,
                "oracle_execution_authorized": False,
                "requires_fresh_memory_and_swap_observation_immediately_before_any_future_execution": True,
                "requires_a_separate_execution_lease_and_range_reader_implementation": True,
                "refusal_reasons": refusal_reasons,
            },
            "claim_boundary": {
                "metadata_and_memory_arithmetic_only": True,
                "does_not_open_or_map_source_tensor_payloads": True,
                "does_not_load_a_source_model": True,
                "does_not_use_gpu_metal_mps_or_other_accelerator": True,
                "does_not_start_or_contact_a_server": True,
                "does_not_invoke_hcli": True,
                "does_not_execute_the_source_oracle": True,
                "does_not_claim_source_quality_candidate_quality_or_a_benchmark_result": True,
                "does_not_grant_a_memory_or_execution_lease": True,
            },
        }
    )


def _write_new_json(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise LayerStreamedOracleFeasibilityError("--out must be an absolute path")
    if not path.parent.is_dir():
        raise LayerStreamedOracleFeasibilityError("--out parent directory must already exist")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    except FileExistsError as exc:
        raise LayerStreamedOracleFeasibilityError("--out must name a new file") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-contract", type=Path, required=True, help="sealed three-way source-BF16 contract")
    parser.add_argument(
        "--whole-model-memory-preflight", type=Path, required=True, help="sealed no-load whole-model memory preflight"
    )
    parser.add_argument(
        "--semantics-attestation",
        type=Path,
        help="future sealed exact-semantics attestation; omit to emit the truthful current refusal",
    )
    parser.add_argument("--out", type=Path, required=True, help="new absolute JSON feasibility report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_feasibility(
            source_contract_path=args.source_contract,
            whole_model_preflight_path=args.whole_model_memory_preflight,
            semantics_attestation_path=args.semantics_attestation,
        )
        _write_new_json(args.out, result)
    except LayerStreamedOracleFeasibilityError as exc:
        print(f"Q30 layer-streamed source-BF16 oracle feasibility refused: {exc}")
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "status": result["status"],
                "seal_sha256": result["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
