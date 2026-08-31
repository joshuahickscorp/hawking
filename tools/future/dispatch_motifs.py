"""DISPATCH MOTIFS — why 628, and what could be one PersistentPhysicalRegion.

Production issues 628 dispatches per decoded token with the sealed fusion
env, 964 without. This sidecar clusters those launches into semantic motifs
(operation sequence, layer repetition, representation decode, producer ->
consumer boundary, state update, norm, residual, routing) and judges which
motifs could become a PersistentPhysicalRegion.

It is not a fusion plan. The target is not 628 -> 500. A motif is what the
encode path already repeats; a region is a judgment about whether that
repetition could stay resident.

    python3 tools/future/dispatch_motifs.py --record
    python3 -m pytest tools/future/test_dispatch_motifs.py -q

evidence_class STATIC_ONLY. No GPU. No bench lock. Does not touch crates/.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import time
from typing import Any, Mapping, Sequence

from tools.future._common import (
    HARDWARE_FIELDS,
    REPO,
    HardwareClaimError,
    _assert_no_hardware_claims,
    load_json,
    write_receipt,
)
from tools.future.tps_budget import (
    DECODE_SRC,
    Fusion,
    _read,
    count_dispatches_per_decoded_token,
    decode_path_markers,
    load_geometry,
    mixer_kind,
)

RECEIPT = "DISPATCH_MOTIFS.json"
SCHEMA = "hawking.future.dispatch_motifs.v1"
VERSION = 1
RECORDED_BY = "tools/future.dispatch_motifs.py"

EVIDENCE_CLASS = "STATIC_ONLY"
RESIDENT_BUDGET_REL = "receipts/future/RESIDENT_TOKEN_BUDGET.json"
PHYSICAL_PRIMITIVES_REL = "tools/future/physical_primitives.py"
NEGATIVE_INDEX_REL = "receipts/future/NEGATIVE_SCIENCE_INDEX.json"
DISPATCH_CEREMONY_REL = "receipts/future/DISPATCH_CEREMONY.json"

# Established production figures. Do not re-derive; the walk must match them
# or refuse. Cited from RESIDENT_TOKEN_BUDGET / the live probe, independently
# reproduced by tools/future/tps_budget.py's coarse encode-path walk.
ESTABLISHED_SEALED = 628
ESTABLISHED_UNFUSED = 964
ESTABLISHED_FUSION_REMOVED = 336
CITED_MARGINAL_US = 6.25
CITED_FUSION_SAVED_MS = 2.10
CITED_MARGINAL_SOURCE = (
    f"{RESIDENT_BUDGET_REL} derived.measured_marginal_dispatch_us"
)

MARGINAL_CAVEAT = (
    "6.25 us is the paired A/B of the 336 dispatches fusion removed "
    f"({ESTABLISHED_FUSION_REMOVED} launches, {CITED_FUSION_SAVED_MS} ms). "
    "Do not assume that figure extrapolates linearly to zero, and do not "
    "assume the remaining 628 cost the same as the 336 that fusion removed. "
    "Fusion also changed the surviving kernels' work (gate+up+SwiGLU in one "
    "launch still does the GEMVs), so the 2.10 ms is not a pure launch tax. "
    "A work-free removed dispatch was priced at ~1 us on a different class "
    "(DISPATCH_CEREMONY / ACCELERATOR_DISPATCH_IS_NOT_THE_PRICE). Product "
    "count x 6.25 us is CITED_MARGINAL_NOT_EXTRAPOLATED, not a measurement "
    "of that motif."
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. Motif counts are "
    "re-derived from encode/dispatch call sites in "
    "crates/hawking-core/src/model/qwen38_hybrid_decode.rs (one "
    "dispatch_threads = one launch) and must reconcile to the established "
    "628/964 totals or this module refuses to emit. The 6.25 us marginal is "
    f"cited from {RESIDENT_BUDGET_REL}, not re-measured. gpu_authority is "
    "false. evidence_class is STATIC_ONLY."
)

# Mixer kind is a static function of layer index. Qwen3.8 hybrid on this
# path is dense MLP; there is no expert routing.
HELPER_MARKERS = {
    "encode_full_token": "fn encode_full_token",
    "encode_embed": "fn encode_embed(",
    "encode_layers": "fn encode_layers(",
    "encode_terminal": "fn encode_terminal(",
    "encode_deltanet": "fn encode_deltanet(",
    "encode_gqa": "fn encode_gqa(",
    "encode_dense_mlp": "fn encode_dense_mlp(",
    "encode_fused_pair_concat": "fn encode_fused_pair_concat(",
    "encode_fused_qkv": "fn encode_fused_qkv(",
    "encode_fused_gate_up": "fn encode_fused_gate_up(",
    "encode_dn_ba_and_delta": "fn encode_dn_ba_and_delta(",
    "encode_ba_to_decay": "fn encode_ba_to_decay(",
    "encode_gated_delta": "fn encode_gated_delta(",
    "encode_gated_delta_fused_ba": "fn encode_gated_delta_fused_ba(",
    "encode_rmsnorm": "fn encode_rmsnorm(",
    "encode_add_residual_rmsnorm": "fn encode_add_residual_rmsnorm(",
    "encode_rearrange": "fn encode_rearrange(",
    "encode_sigmoid_gate": "fn encode_sigmoid_gate(",
    "mha_decode_f32_tcb": "mha_decode_f32_tcb(",
    "qwen_next_add_residual_tcb": "qwen_next_add_residual_tcb(",
    "sample_argmax_f32_tcb": "sample_argmax_f32_tcb(",
    "fuse_ba_delta_off_default": "Default Off — production stays 756/628",
}


class MotifRefuse(ValueError):
    """Census does not reconcile, or the encode path is no longer the graph."""


# ---------------------------------------------------------------------------
# Motif catalog. Fine grain: one id per distinct launch kind. A launch is
# assigned exactly one id, so sealed counts and unfused counts each partition
# their graph. Families also partition (elementwise is 0 under sealed fusion).
# ---------------------------------------------------------------------------

# family is the primary partition axis. tags are extra grouping axes.
MOTIF_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "embed_lookup",
        "family": "representation_decode",
        "tags": ("representation_decode",),
        "repetition": 1,
        "encode": "encode_embed / encode_embed_mixed",
        "kernel": "qwen_uniform_q4_embedding_lookup | hgravu | hgravf",
        "dynamic_slots": ("token_id",),
        "what": (
            "token-indexed packed-weight row lookup into hidden. One launch. "
            "Not a GEMV; the token id is a set_bytes slot."
        ),
    },
    {
        "id": "mixer_input_rmsnorm",
        "family": "norm",
        "tags": ("norm",),
        "repetition": "layers_unless_folded",
        "encode": "encode_rmsnorm",
        "kernel": "qwen80_residual_rmsnorm_tg",
        "dynamic_slots": (),
        "what": (
            "mixer input RMSNorm. Layer 0 always runs it. fuse_add_rmsnorm "
            "folds layer>0 into the previous MLP residual."
        ),
    },
    {
        "id": "dn_inproj_qkvz",
        "family": "representation_decode",
        "tags": ("representation_decode", "layer_repetition"),
        "repetition": "QWEN38_DELTANET_LAYERS",
        "encode": "encode_named_matvec / encode_q4_matvec (qkvz)",
        "kernel": "geo_tpr64 matvec",
        "dynamic_slots": (),
        "what": "DeltaNet in_proj_qkvz GEMV. Unfused only; sealed uses pair-concat.",
    },
    {
        "id": "dn_inproj_ba",
        "family": "representation_decode",
        "tags": ("representation_decode", "layer_repetition"),
        "repetition": "QWEN38_DELTANET_LAYERS",
        "encode": "encode_named_matvec / encode_q4_matvec (ba)",
        "kernel": "geo_tpr64 matvec",
        "dynamic_slots": (),
        "what": "DeltaNet in_proj_ba GEMV. Unfused only; sealed uses pair-concat.",
    },
    {
        "id": "dn_inproj_pair_concat",
        "family": "representation_decode",
        "tags": ("representation_decode", "layer_repetition", "operation_sequence"),
        "repetition": "QWEN38_DELTANET_LAYERS",
        "encode": "encode_fused_pair_concat",
        "kernel": "qwen_uniform_q4_group64_matvec_pair_concat_geo_tpr64_tg128",
        "dynamic_slots": (),
        "what": (
            "fused qkvz+ba pair-concat GEMV. Sealed FUSE_DN_INPROJ. One launch "
            "replaces two independent matvecs."
        ),
    },
    {
        "id": "dn_rearrange_conv",
        "family": "state_update",
        "tags": ("state_update", "layer_repetition"),
        "repetition": "QWEN38_DELTANET_LAYERS",
        "encode": "encode_deltanet inline / encode_rearrange",
        "kernel": "qwen38_qkvz_rearrange_conv_l2_f32",
        "dynamic_slots": (),
        "what": (
            "rearrange qkvz, depthwise conv1d, L2. Writes conv_state at a "
            "per-layer slot. O(1) in sequence length."
        ),
    },
    {
        "id": "dn_ba_to_decay",
        "family": "producer_consumer",
        "tags": ("producer_consumer", "layer_repetition"),
        "repetition": "QWEN38_DELTANET_LAYERS",
        "encode": "encode_ba_to_decay",
        "kernel": "qwen80_ba_to_decay_beta_f32",
        "dynamic_slots": (),
        "what": (
            "ba activations + A_log + dt_bias -> decay, beta. Producer for "
            "gated-delta. Folded by FUSE_BA_DELTA (default Off on the 628 graph)."
        ),
    },
    {
        "id": "dn_gated_delta",
        "family": "state_update",
        "tags": ("state_update", "layer_repetition"),
        "repetition": "QWEN38_DELTANET_LAYERS",
        "encode": "encode_gated_delta",
        "kernel": "qwen38_gated_delta_decode_vi_simd",
        "dynamic_slots": (),
        "what": (
            "recurrent state machine step. Reads/writes rec_state at a "
            "per-layer slot. O(1) in sequence length."
        ),
    },
    {
        "id": "dn_gated_delta_fused_ba",
        "family": "state_update",
        "tags": ("state_update", "layer_repetition", "operation_sequence"),
        "repetition": "QWEN38_DELTANET_LAYERS",
        "encode": "encode_gated_delta_fused_ba",
        "kernel": "qwen38_gated_delta_decode_vi_simd_ba",
        "dynamic_slots": (),
        "what": "ba_to_decay folded into gated-delta. Not on the 628 graph.",
    },
    {
        "id": "dn_gated_rmsnorm",
        "family": "norm",
        "tags": ("norm", "layer_repetition"),
        "repetition": "QWEN38_DELTANET_LAYERS",
        "encode": "encode_gated_rmsnorm / encode_deltanet inline",
        "kernel": "qwen80_deltanet_gated_rmsnorm_tg",
        "dynamic_slots": (),
        "what": "per-head gated RMSNorm on rec_out * z, before out_proj.",
    },
    {
        "id": "dn_out_proj",
        "family": "representation_decode",
        "tags": ("representation_decode", "layer_repetition"),
        "repetition": "QWEN38_DELTANET_LAYERS",
        "encode": "encode_named_matvec / encode_q4_matvec (out_proj)",
        "kernel": "geo_tpr64 matvec",
        "dynamic_slots": (),
        "what": "DeltaNet out_proj GEMV. Gated vector -> hidden.",
    },
    {
        "id": "gqa_q_proj",
        "family": "representation_decode",
        "tags": ("representation_decode", "layer_repetition"),
        "repetition": "QWEN38_GQA_LAYERS",
        "encode": "encode_q4_matvec (q_proj)",
        "kernel": "geo_tpr64 matvec",
        "dynamic_slots": (),
        "what": "GQA q_proj GEMV. Unfused only.",
    },
    {
        "id": "gqa_k_proj",
        "family": "representation_decode",
        "tags": ("representation_decode", "layer_repetition"),
        "repetition": "QWEN38_GQA_LAYERS",
        "encode": "encode_q4_matvec (k_proj)",
        "kernel": "geo_tpr64 matvec",
        "dynamic_slots": (),
        "what": "GQA k_proj GEMV. Unfused only.",
    },
    {
        "id": "gqa_v_proj",
        "family": "representation_decode",
        "tags": ("representation_decode", "layer_repetition"),
        "repetition": "QWEN38_GQA_LAYERS",
        "encode": "encode_q4_matvec (v_proj)",
        "kernel": "geo_tpr64 matvec",
        "dynamic_slots": (),
        "what": "GQA v_proj GEMV. Unfused only.",
    },
    {
        "id": "gqa_fused_qkv",
        "family": "representation_decode",
        "tags": ("representation_decode", "layer_repetition", "operation_sequence"),
        "repetition": "QWEN38_GQA_LAYERS",
        "encode": "encode_fused_qkv",
        "kernel": "qwen_uniform_q4_group64_matvec_qkv_geo_tpr64_tg128",
        "dynamic_slots": (),
        "what": "fused QKV GEMV. Sealed FUSE_GQA_QKV. One launch replaces three.",
    },
    {
        "id": "gqa_qk_norm_rope_cache",
        "family": "state_update",
        "tags": ("state_update", "layer_repetition"),
        "repetition": "QWEN38_GQA_LAYERS",
        "encode": "encode_gqa inline / encode_rope_cache",
        "kernel": "qwen38_gqa_qk_norm_rope_cache_tg",
        "dynamic_slots": ("position",),
        "what": (
            "Q/K RMSNorm, RoPE, write K/V cache at position. Grid is static "
            "(heads x tg). Position is set_bytes."
        ),
    },
    {
        "id": "gqa_mha_decode",
        "family": "producer_consumer",
        "tags": ("producer_consumer", "state_update", "layer_repetition"),
        "repetition": "QWEN38_GQA_LAYERS",
        "encode": "mha_decode_f32_tcb",
        "kernel": "mha_decode_f32",
        "dynamic_slots": ("seq_len", "threadgroup_memory"),
        "what": (
            "attention over the growing KV cache. Grid is static "
            "(n_heads * tg). Threadgroup memory is (seq_len + tg) floats, so "
            "it grows every token. seq_len already lives in a KernelArgBuffer, "
            "not set_bytes — the only launch on this graph that already uses "
            "argument buffers for scalars."
        ),
    },
    {
        "id": "gqa_sigmoid_gate",
        "family": "routing",
        "tags": ("routing", "layer_repetition"),
        "repetition": "QWEN38_GQA_LAYERS",
        "encode": "encode_sigmoid_gate",
        "kernel": "qwen38_attention_apply_sigmoid_gate",
        "dynamic_slots": (),
        "what": (
            "elementwise sigmoid gate of attention. Value-dependent arithmetic, "
            "not expert routing: it does not change dispatch topology. This "
            "model's MLP is dense; mixer kind is a static function of layer."
        ),
    },
    {
        "id": "gqa_o_proj",
        "family": "representation_decode",
        "tags": ("representation_decode", "layer_repetition"),
        "repetition": "QWEN38_GQA_LAYERS",
        "encode": "encode_named_matvec / encode_q4_matvec (o_proj)",
        "kernel": "geo_tpr64 matvec",
        "dynamic_slots": (),
        "what": "GQA o_proj GEMV.",
    },
    {
        "id": "mixer_add_residual",
        "family": "residual",
        "tags": ("residual", "producer_consumer"),
        "repetition": "QWEN38_LAYERS",
        "encode": "qwen_next_add_residual_tcb",
        "kernel": "qwen_next_add_residual",
        "dynamic_slots": (),
        "what": "plain mixer residual add. Unfused only.",
    },
    {
        "id": "mixer_add_residual_rmsnorm",
        "family": "residual",
        "tags": ("residual", "norm", "producer_consumer", "operation_sequence"),
        "repetition": "QWEN38_LAYERS",
        "encode": "encode_add_residual_rmsnorm",
        "kernel": "qwen38 add_residual_rmsnorm",
        "dynamic_slots": (),
        "what": (
            "mixer residual + post-attention RMSNorm in one launch. Sealed "
            "FUSE_ADD_RMSNORM. Counted once as residual; tagged as norm."
        ),
    },
    {
        "id": "mlp_post_attn_rmsnorm",
        "family": "norm",
        "tags": ("norm", "layer_repetition"),
        "repetition": "QWEN38_LAYERS",
        "encode": "encode_rmsnorm (post_attention_layernorm)",
        "kernel": "qwen80_residual_rmsnorm_tg",
        "dynamic_slots": (),
        "what": "MLP input RMSNorm. Folded into mixer residual under sealed fusion.",
    },
    {
        "id": "mlp_gate_proj",
        "family": "representation_decode",
        "tags": ("representation_decode", "layer_repetition"),
        "repetition": "QWEN38_LAYERS",
        "encode": "encode_named_matvec (gate_proj)",
        "kernel": "geo_tpr64 matvec",
        "dynamic_slots": (),
        "what": "MLP gate GEMV. Unfused only.",
    },
    {
        "id": "mlp_up_proj",
        "family": "representation_decode",
        "tags": ("representation_decode", "layer_repetition"),
        "repetition": "QWEN38_LAYERS",
        "encode": "encode_named_matvec (up_proj)",
        "kernel": "geo_tpr64 matvec",
        "dynamic_slots": (),
        "what": "MLP up GEMV. Unfused only.",
    },
    {
        "id": "mlp_swiglu",
        "family": "elementwise",
        "tags": ("elementwise", "producer_consumer", "layer_repetition"),
        "repetition": "QWEN38_LAYERS",
        "encode": "encode_dense_mlp inline swiglu_f32",
        "kernel": "swiglu_f32",
        "dynamic_slots": (),
        "what": "SwiGLU on gate x up. Folded into the fused GEMV under sealed swiglu.",
    },
    {
        "id": "mlp_fused_gate_up",
        "family": "representation_decode",
        "tags": ("representation_decode", "operation_sequence"),
        "repetition": "QWEN38_LAYERS",
        "encode": "encode_fused_gate_up(with_swiglu=false)",
        "kernel": "qwen_uniform_q4_group64_matvec_gate_up_geo_tpr64_tg128",
        "dynamic_slots": (),
        "what": "fused gate+up GEMV, SwiGLU still separate. Not the sealed graph.",
    },
    {
        "id": "mlp_fused_gate_up_swiglu",
        "family": "representation_decode",
        "tags": ("representation_decode", "operation_sequence", "layer_repetition"),
        "repetition": "QWEN38_LAYERS",
        "encode": "encode_fused_gate_up(with_swiglu=true)",
        "kernel": "qwen_uniform_q4_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
        "dynamic_slots": (),
        "what": "fused gate+up+SwiGLU. Sealed FUSE_MLP=swiglu. One launch replaces three.",
    },
    {
        "id": "mlp_down_proj",
        "family": "representation_decode",
        "tags": ("representation_decode", "layer_repetition"),
        "repetition": "QWEN38_LAYERS",
        "encode": "encode_named_matvec / encode_q4_matvec (down_proj)",
        "kernel": "geo_tpr64 matvec",
        "dynamic_slots": (),
        "what": "MLP down_proj GEMV. Intermediate -> hidden. Survives every fusion set.",
    },
    {
        "id": "mlp_add_residual",
        "family": "residual",
        "tags": ("residual", "producer_consumer"),
        "repetition": "QWEN38_LAYERS",
        "encode": "qwen_next_add_residual_tcb",
        "kernel": "qwen_next_add_residual",
        "dynamic_slots": (),
        "what": "plain MLP residual add. Unfused only.",
    },
    {
        "id": "mlp_add_residual_rmsnorm",
        "family": "residual",
        "tags": ("residual", "norm", "producer_consumer", "operation_sequence"),
        "repetition": "QWEN38_LAYERS",
        "encode": "encode_add_residual_rmsnorm",
        "kernel": "qwen38 add_residual_rmsnorm",
        "dynamic_slots": (),
        "what": (
            "MLP residual + next mixer input RMSNorm (last layer: final norm) "
            "in one launch. Sealed FUSE_ADD_RMSNORM."
        ),
    },
    {
        "id": "final_rmsnorm",
        "family": "norm",
        "tags": ("norm",),
        "repetition": 1,
        "encode": "encode_terminal encode_rmsnorm",
        "kernel": "qwen80_residual_rmsnorm_tg",
        "dynamic_slots": (),
        "what": "standalone final RMSNorm. Folded into last MLP residual under sealed fusion.",
    },
    {
        "id": "lm_head",
        "family": "representation_decode",
        "tags": ("representation_decode",),
        "repetition": 1,
        "encode": "encode_named_matvec / encode_q4_matvec (lm_head)",
        "kernel": "geo_tpr64 matvec",
        "dynamic_slots": (),
        "what": "lm_head GEMV, hidden -> vocab. Always runs on encode_full_token.",
    },
    {
        "id": "argmax",
        "family": "producer_consumer",
        "tags": ("producer_consumer",),
        "repetition": 1,
        "encode": "encode_argmax / sample_argmax_f32_tcb",
        "kernel": "sample_argmax_f32",
        "dynamic_slots": (),
        "what": (
            "greedy argmax over logits. Output token is the next step's embed "
            "slot. Two-pass default Off (one launch)."
        ),
    },
)

CATALOG_BY_ID: dict[str, dict[str, Any]] = {row["id"]: row for row in MOTIF_CATALOG}
FAMILIES: tuple[str, ...] = (
    "representation_decode",
    "state_update",
    "producer_consumer",
    "norm",
    "residual",
    "routing",
    "elementwise",
)

HIGH_FREQUENCY_MIN = 16

# Per-motif PersistentPhysicalRegion judgment. High-frequency sequences are
# the region candidates; a single launch kind is usually not a region by
# itself. Missing keys refuse at cluster time.
_LAYER_DN = ("dn_layer_state_machine", "hybrid_4_layer_tile", "token_graph_persistent_executor")
_LAYER_GQA = ("gqa_layer_static_skeleton", "hybrid_4_layer_tile", "token_graph_persistent_executor")
_LAYER_MLP = ("mlp_suffix_representation_region", "hybrid_4_layer_tile", "token_graph_persistent_executor")
_GEMV_Q = ("representation_decode_persistent_queue", "token_graph_persistent_executor")

MOTIF_REGION: dict[str, dict[str, Any]] = {
    "embed_lookup": {
        "standalone": False,
        "form": "static_skeleton_with_dynamic_token_or_route_slots",
        "absorbed_by": ("token_graph_persistent_executor", "representation_decode_persistent_queue"),
        "why": "one launch; token_id is a slot of the token-graph skeleton, not a region",
    },
    "mixer_input_rmsnorm": {
        "standalone": False,
        "form": None,
        "absorbed_by": _LAYER_DN,
        "why": "sealed: layer-0 only. Later layers already folded into the previous MLP residual",
    },
    "dn_inproj_qkvz": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + _LAYER_DN,
        "why": "unfused GEMV; sealed replaces it with pair-concat",
    },
    "dn_inproj_ba": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + _LAYER_DN,
        "why": "unfused GEMV; sealed replaces it with pair-concat",
    },
    "dn_inproj_pair_concat": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + _LAYER_DN,
        "why": "representation decode of two packed weights; first stage of the DN state machine",
    },
    "dn_rearrange_conv": {
        "standalone": False,
        "form": "long_lived_state_machine",
        "absorbed_by": _LAYER_DN,
        "why": "writes conv_state; O(1) in seq; not a region without the rest of the DN sequence",
    },
    "dn_ba_to_decay": {
        "standalone": False,
        "form": "long_lived_state_machine",
        "absorbed_by": ("dn_ba_delta_existing_lever",) + _LAYER_DN,
        "why": "producer for gated-delta; existing FUSE_BA_DELTA folds it",
    },
    "dn_gated_delta": {
        "standalone": False,
        "form": "long_lived_state_machine",
        "absorbed_by": ("dn_ba_delta_existing_lever",) + _LAYER_DN,
        "why": "the recurrent step; the DN region is this plus its producers and epilogue",
    },
    "dn_gated_delta_fused_ba": {
        "standalone": False,
        "form": "long_lived_state_machine",
        "absorbed_by": _LAYER_DN,
        "why": "not on the 628 graph; the inner cut of the DN state machine",
    },
    "dn_gated_rmsnorm": {
        "standalone": False,
        "form": "long_lived_state_machine",
        "absorbed_by": _LAYER_DN,
        "why": "norm on rec_out; epilogue of the recurrent step, before out_proj",
    },
    "dn_out_proj": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + _LAYER_DN,
        "why": "GEMV epilogue of the DN sequence; residual follows it",
    },
    "gqa_q_proj": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + _LAYER_GQA,
        "why": "unfused GEMV; sealed uses fused QKV",
    },
    "gqa_k_proj": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + _LAYER_GQA,
        "why": "unfused GEMV; sealed uses fused QKV",
    },
    "gqa_v_proj": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + _LAYER_GQA,
        "why": "unfused GEMV; sealed uses fused QKV",
    },
    "gqa_fused_qkv": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + _LAYER_GQA,
        "why": "first stage of the GQA skeleton",
    },
    "gqa_qk_norm_rope_cache": {
        "standalone": False,
        "form": "static_skeleton_with_dynamic_token_or_route_slots",
        "absorbed_by": _LAYER_GQA,
        "why": "position is a scalar slot; writes KV. Not a region without MHA",
    },
    "gqa_mha_decode": {
        "standalone": False,
        "form": "static_skeleton_with_dynamic_token_or_route_slots",
        "absorbed_by": _LAYER_GQA,
        "why": "the dynamic-shape launch (TG memory grows with seq). Pad to max_seq and the GQA skeleton is static",
    },
    "gqa_sigmoid_gate": {
        "standalone": False,
        "form": None,
        "absorbed_by": _LAYER_GQA,
        "why": "value-dependent elementwise, not expert routing; does not change topology",
    },
    "gqa_o_proj": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + _LAYER_GQA,
        "why": "GEMV epilogue of GQA; residual follows it",
    },
    "mixer_add_residual": {
        "standalone": False,
        "form": None,
        "absorbed_by": ("residual_rmsnorm_is_a_boundary_not_a_region",) + _LAYER_DN + _LAYER_GQA,
        "why": "producer-consumer boundary, unfused",
    },
    "mixer_add_residual_rmsnorm": {
        "standalone": False,
        "form": None,
        "absorbed_by": ("residual_rmsnorm_is_a_boundary_not_a_region",) + _LAYER_DN + _LAYER_GQA + _LAYER_MLP,
        "why": "producer-consumer boundary. 128 sealed looks like a region and is not",
    },
    "mlp_post_attn_rmsnorm": {
        "standalone": False,
        "form": None,
        "absorbed_by": _LAYER_MLP,
        "why": "folded into mixer residual under sealed fusion",
    },
    "mlp_gate_proj": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + _LAYER_MLP,
        "why": "unfused GEMV; sealed uses fused gate_up_swiglu",
    },
    "mlp_up_proj": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + _LAYER_MLP,
        "why": "unfused GEMV; sealed uses fused gate_up_swiglu",
    },
    "mlp_swiglu": {
        "standalone": False,
        "form": None,
        "absorbed_by": _LAYER_MLP,
        "why": "elementwise; sealed fusion already folded it into the GEMV",
    },
    "mlp_fused_gate_up": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + _LAYER_MLP,
        "why": "not the sealed graph (pair, not swiglu)",
    },
    "mlp_fused_gate_up_swiglu": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + _LAYER_MLP,
        "why": "first stage of the MLP suffix region",
    },
    "mlp_down_proj": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + _LAYER_MLP,
        "why": "survives every fusion set; epilogue-fuses with residual_rmsnorm in the MLP suffix region",
    },
    "mlp_add_residual": {
        "standalone": False,
        "form": None,
        "absorbed_by": ("residual_rmsnorm_is_a_boundary_not_a_region",) + _LAYER_MLP,
        "why": "producer-consumer boundary, unfused",
    },
    "mlp_add_residual_rmsnorm": {
        "standalone": False,
        "form": None,
        "absorbed_by": ("residual_rmsnorm_is_a_boundary_not_a_region",) + _LAYER_MLP,
        "why": "producer-consumer boundary and the MLP suffix epilogue",
    },
    "final_rmsnorm": {
        "standalone": False,
        "form": None,
        "absorbed_by": _LAYER_MLP + ("token_graph_persistent_executor",),
        "why": "folded into last MLP residual under sealed fusion",
    },
    "lm_head": {
        "standalone": False,
        "form": "representation_native_region",
        "absorbed_by": _GEMV_Q + ("token_graph_persistent_executor",),
        "why": "already one launch; a region only as the terminal of the token graph",
    },
    "argmax": {
        "standalone": False,
        "form": "static_skeleton_with_dynamic_token_or_route_slots",
        "absorbed_by": ("token_graph_persistent_executor",),
        "why": "produces the next embed token_id slot; not a region",
    },
}

_missing_region = [row["id"] for row in MOTIF_CATALOG if row["id"] not in MOTIF_REGION]
if _missing_region:
    raise MotifRefuse(f"MOTIF_REGION missing { _missing_region }")
_extra_region = [k for k in MOTIF_REGION if k not in CATALOG_BY_ID]
if _extra_region:
    raise MotifRefuse(f"MOTIF_REGION has unknown ids { _extra_region }")


# ---------------------------------------------------------------------------
# Encode-path walk. Independent of tps_budget's coarse site fold. One atom
# per dispatch_threads. Fusion changes which atoms fire, not their meaning.
# ---------------------------------------------------------------------------


def _require_catalog(motif_id: str) -> dict[str, Any]:
    spec = CATALOG_BY_ID.get(motif_id)
    if spec is None:
        raise MotifRefuse(f"walk emitted unknown motif id {motif_id!r}")
    return spec


def walk_launches(geo: Mapping[str, int], fusion: Fusion) -> list[dict[str, Any]]:
    """Walk encode_full_token the way generate_greedy.step does, at launch grain.

    generate_greedy -> session.step -> encode_full_token:
        encode_embed (1) + encode_layers (64 mixers + 64 MLP) + encode_terminal.

    Mixed catalog vs uniform-Q4 does not change launch count on this graph.
    Split DeltaNet projections (qkv/z/a/b + fuse kernels) are a missing-weight
    fallback, not the production QKVZ/BA tensors; they are not walked.
    """
    n_layers = int(geo["QWEN38_LAYERS"])
    interval = int(geo["QWEN38_FULL_ATTENTION_INTERVAL"])
    launches: list[dict[str, Any]] = []

    def emit(motif_id: str, *, layer: int | None, mixer: str | None) -> None:
        spec = _require_catalog(motif_id)
        launches.append(
            {
                "motif_id": motif_id,
                "family": spec["family"],
                "layer": layer,
                "mixer": mixer,
                "encode": spec["encode"],
            }
        )

    emit("embed_lookup", layer=None, mixer=None)

    for layer in range(n_layers):
        kind = mixer_kind(layer, interval)
        if not (fusion.add_rmsnorm and layer > 0):
            emit("mixer_input_rmsnorm", layer=layer, mixer=kind)

        if kind == "dn":
            if fusion.dn_inproj:
                emit("dn_inproj_pair_concat", layer=layer, mixer=kind)
            else:
                emit("dn_inproj_qkvz", layer=layer, mixer=kind)
                emit("dn_inproj_ba", layer=layer, mixer=kind)
            emit("dn_rearrange_conv", layer=layer, mixer=kind)
            if fusion.ba_delta:
                emit("dn_gated_delta_fused_ba", layer=layer, mixer=kind)
            else:
                emit("dn_ba_to_decay", layer=layer, mixer=kind)
                emit("dn_gated_delta", layer=layer, mixer=kind)
            emit("dn_gated_rmsnorm", layer=layer, mixer=kind)
            emit("dn_out_proj", layer=layer, mixer=kind)
        else:
            if fusion.gqa_qkv:
                emit("gqa_fused_qkv", layer=layer, mixer=kind)
            else:
                emit("gqa_q_proj", layer=layer, mixer=kind)
                emit("gqa_k_proj", layer=layer, mixer=kind)
                emit("gqa_v_proj", layer=layer, mixer=kind)
            emit("gqa_qk_norm_rope_cache", layer=layer, mixer=kind)
            emit("gqa_mha_decode", layer=layer, mixer=kind)
            emit("gqa_sigmoid_gate", layer=layer, mixer=kind)
            emit("gqa_o_proj", layer=layer, mixer=kind)

        if fusion.add_rmsnorm:
            emit("mixer_add_residual_rmsnorm", layer=layer, mixer=kind)
        else:
            emit("mixer_add_residual", layer=layer, mixer=kind)

        if not fusion.add_rmsnorm:
            emit("mlp_post_attn_rmsnorm", layer=layer, mixer=kind)

        if fusion.mlp == "off":
            emit("mlp_gate_proj", layer=layer, mixer=kind)
            emit("mlp_up_proj", layer=layer, mixer=kind)
            emit("mlp_swiglu", layer=layer, mixer=kind)
        elif fusion.mlp == "pair":
            emit("mlp_fused_gate_up", layer=layer, mixer=kind)
            emit("mlp_swiglu", layer=layer, mixer=kind)
        elif fusion.mlp == "swiglu":
            emit("mlp_fused_gate_up_swiglu", layer=layer, mixer=kind)
        else:
            raise MotifRefuse(f"unrecognised mlp fusion {fusion.mlp!r}")

        emit("mlp_down_proj", layer=layer, mixer=kind)
        if fusion.add_rmsnorm:
            emit("mlp_add_residual_rmsnorm", layer=layer, mixer=kind)
        else:
            emit("mlp_add_residual", layer=layer, mixer=kind)

    if not fusion.add_rmsnorm:
        emit("final_rmsnorm", layer=None, mixer=None)
    emit("lm_head", layer=None, mixer=None)
    if fusion.argmax_two_pass:
        raise MotifRefuse(
            "argmax two-pass is not the production graph; this walk refuses "
            "to count it as 628/964"
        )
    emit("argmax", layer=None, mixer=None)
    return launches


def cluster_launches(launches: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {row["id"]: 0 for row in MOTIF_CATALOG}
    for atom in launches:
        motif_id = str(atom["motif_id"])
        if motif_id not in counts:
            raise MotifRefuse(f"cluster saw unknown motif {motif_id!r}")
        counts[motif_id] += 1
    return counts


def family_counts(motif_counts: Mapping[str, int]) -> dict[str, int]:
    out = {name: 0 for name in FAMILIES}
    for motif_id, n in motif_counts.items():
        family = CATALOG_BY_ID[motif_id]["family"]
        if family not in out:
            raise MotifRefuse(f"motif {motif_id} has unknown family {family!r}")
        out[family] += int(n)
    return out


def cited_marginal_product_ms(count: int) -> dict[str, Any]:
    """Arithmetic on the cited 6.25 us. Not a measurement of this motif."""
    product_us = float(count) * CITED_MARGINAL_US
    return {
        "count": int(count),
        "cited_marginal_us": CITED_MARGINAL_US,
        "cited_from": CITED_MARGINAL_SOURCE,
        "product_us": product_us,
        "product_ms": product_us / 1000.0,
        "status": "CITED_MARGINAL_NOT_EXTRAPOLATED",
        "caveat": MARGINAL_CAVEAT,
        "not_a_hardware_measurement": True,
    }


def motif_row(
    motif_id: str, sealed: int, unfused: int
) -> dict[str, Any]:
    spec = CATALOG_BY_ID[motif_id]
    return {
        "id": motif_id,
        "family": spec["family"],
        "tags": list(spec["tags"]),
        "encode": spec["encode"],
        "kernel": spec["kernel"],
        "dynamic_slots": list(spec["dynamic_slots"]),
        "what": spec["what"],
        "sealed_count": int(sealed),
        "unfused_count": int(unfused),
        "high_frequency": int(sealed) >= HIGH_FREQUENCY_MIN,
        "cited_marginal_if_this_class": cited_marginal_product_ms(sealed),
        "region": {
            **MOTIF_REGION[motif_id],
            "absorbed_by": list(MOTIF_REGION[motif_id]["absorbed_by"]),
        },
    }


def reconcile_census(
    sealed_counts: Mapping[str, int],
    unfused_counts: Mapping[str, int],
    *,
    sealed_expected: int = ESTABLISHED_SEALED,
    unfused_expected: int = ESTABLISHED_UNFUSED,
) -> dict[str, Any]:
    """Refuse rather than emit an unreconciled census.

    Both graphs must partition: every catalog id is present, sums match the
    established 628/964 totals, and families partition the same sums.
    """
    missing = [row["id"] for row in MOTIF_CATALOG if row["id"] not in sealed_counts]
    missing += [row["id"] for row in MOTIF_CATALOG if row["id"] not in unfused_counts]
    if missing:
        raise MotifRefuse(f"unreconciled census: catalog ids missing {missing}")

    extra = [k for k in sealed_counts if k not in CATALOG_BY_ID]
    extra += [k for k in unfused_counts if k not in CATALOG_BY_ID]
    if extra:
        raise MotifRefuse(f"unreconciled census: unknown motif ids {extra}")

    sealed_sum = sum(int(sealed_counts[row["id"]]) for row in MOTIF_CATALOG)
    unfused_sum = sum(int(unfused_counts[row["id"]]) for row in MOTIF_CATALOG)
    if sealed_sum != int(sealed_expected) or unfused_sum != int(unfused_expected):
        raise MotifRefuse(
            f"unreconciled census: motif counts sum to sealed={sealed_sum} "
            f"unfused={unfused_sum}, expected {sealed_expected}/{unfused_expected}; "
            "refusing to emit"
        )

    sealed_fam = family_counts(sealed_counts)
    unfused_fam = family_counts(unfused_counts)
    if sum(sealed_fam.values()) != sealed_sum:
        raise MotifRefuse("unreconciled census: sealed family sum drifted")
    if sum(unfused_fam.values()) != unfused_sum:
        raise MotifRefuse("unreconciled census: unfused family sum drifted")

    fusion_removed = unfused_sum - sealed_sum
    if (
        sealed_expected == ESTABLISHED_SEALED
        and unfused_expected == ESTABLISHED_UNFUSED
        and fusion_removed != ESTABLISHED_FUSION_REMOVED
    ):
        raise MotifRefuse(
            f"unreconciled census: unfused-sealed={fusion_removed}, "
            f"established fusion removed {ESTABLISHED_FUSION_REMOVED}"
        )

    return {
        "ok": True,
        "sealed_sum": sealed_sum,
        "unfused_sum": unfused_sum,
        "fusion_removed": fusion_removed,
        "sealed_families": sealed_fam,
        "unfused_families": unfused_fam,
        "treated_unknown_as_zero": False,
    }


def helper_markers(decode_text: str | None = None) -> dict[str, Any]:
    text = decode_text if decode_text is not None else _read(DECODE_SRC)
    present = {key: needle in text for key, needle in HELPER_MARKERS.items()}
    missing = [key for key, ok in present.items() if not ok]
    return {
        "source": DECODE_SRC,
        "required_present": present,
        "missing": missing,
        "ok": not missing,
    }


# ---------------------------------------------------------------------------
# PersistentPhysicalRegion judgments.
#
# Textbooks: CUDA Graphs (capture/replay of a static launch DAG), persistent
# kernels (one grid stays occupied and pulls work), TPU static execution
# (compiled program, dynamic slots), FPGA spatial pipelines (one region
# wired in space). Metal is the physical authority. ICB is Metal's graph-
# replay analogue and is the wrong textbook for the remaining 6.25 us class:
# that cost is mostly on-device launch/teardown (host gap is 0.8 ms of 28.44),
# and ICB was Type-1 killed for host encode (NEGATIVE_SCIENCE L6: after
# residency batching, host encode is ~0.4% of wall; ICB cannot capture
# set_bytes without argument-buffer scalars).
# ---------------------------------------------------------------------------

RISK_EXISTING_LEVER = 1
RISK_EPILOGUE_FUSION = 2
RISK_LAYER_STATE_MACHINE = 3
RISK_LAYER_WITH_DYNAMIC_SEQ = 4
RISK_NEW_EXECUTION_MODEL = 5

METAL_BLOCKERS = (
    "argument_buffers",
    "icb",
    "resource_residency",
    "dynamic_shapes",
    "routing_dependence",
)


def _candidate(
    *,
    id: str,
    form: str,
    judgment: str,
    motifs: Sequence[str],
    sealed_dispatches: int,
    launches_after: int,
    risk: int,
    metal_blockers: Mapping[str, str],
    blocked_today: bool,
    cheapest_falsifier: str,
    why: str,
    not_a_fusion_plan: str,
) -> dict[str, Any]:
    removed = int(sealed_dispatches) - int(launches_after)
    if removed < 0:
        raise MotifRefuse(f"candidate {id} launches_after exceeds sealed_dispatches")
    missing_blockers = [k for k in METAL_BLOCKERS if k not in metal_blockers]
    if missing_blockers:
        raise MotifRefuse(
            f"candidate {id} missing Metal blocker keys {missing_blockers}"
        )
    return {
        "id": id,
        "form": form,
        "judgment": judgment,
        "motifs": list(motifs),
        "sealed_dispatches": int(sealed_dispatches),
        "launches_after": int(launches_after),
        "dispatches_removed": removed,
        "not_628_to_500": True,
        "why_not_628_to_500": (
            "the goalpost is a PersistentPhysicalRegion, not shaving 128 "
            "launches. A candidate that happens to remove 128 is still a "
            "region judgment, not a 628 -> 500 plan"
        ),
        "risk": int(risk),
        "removed_per_risk_unit": (float(removed) / float(risk)) if risk else None,
        "blocked_today": bool(blocked_today),
        "metal_blockers": dict(metal_blockers),
        "cited_marginal_if_this_class": cited_marginal_product_ms(removed),
        "cheapest_falsifier": cheapest_falsifier,
        "why": why,
        "not_a_fusion_plan": not_a_fusion_plan,
        "icb_is_the_wrong_textbook_for_the_6_25us_class": True,
        "host_encode_already_falsified_as_the_token": (
            "NEGATIVE_SCIENCE L6 Type-1 kill: ICB removes host encode; after "
            "residency batching that is ~0.4% of wall. THE_TOKEN_IS_GPU_BOUND: "
            "host gap 0.8 ms of 28.44. The 6.25 us class is on-device launch "
            "and teardown, which kernel fusion / a persistent occupancy region "
            "/ an FPGA spatial pipeline can remove and ICB replay cannot be "
            "assumed to."
        ),
    }


def region_candidates(sealed_counts: Mapping[str, int]) -> list[dict[str, Any]]:
    """Judgments, not a plan. Each candidate names how many of the 628 it
    would remove and what blocks it on Metal today."""
    c = sealed_counts
    # rearrange / mha / down_proj fire once per DN / GQA / MLP layer on both graphs.
    n_dn = int(c["dn_rearrange_conv"])
    n_gqa = int(c["gqa_mha_decode"])
    n_mlp = int(c["mlp_down_proj"])
    if n_dn + n_gqa != n_mlp:
        raise MotifRefuse(
            f"mixer layers dn={n_dn} + gqa={n_gqa} != mlp {n_mlp}"
        )
    dn_owned = (
        int(c["mixer_input_rmsnorm"])  # layer 0 is DN under the source mixer rule
        + int(c["dn_inproj_pair_concat"])
        + int(c["dn_rearrange_conv"])
        + int(c["dn_ba_to_decay"])
        + int(c["dn_gated_delta"])
        + int(c["dn_gated_rmsnorm"])
        + int(c["dn_out_proj"])
        + n_dn  # mixer residual after each DN layer
    )
    gqa_owned = (
        int(c["gqa_fused_qkv"])
        + int(c["gqa_qk_norm_rope_cache"])
        + int(c["gqa_mha_decode"])
        + int(c["gqa_sigmoid_gate"])
        + int(c["gqa_o_proj"])
        + n_gqa  # mixer residual after each GQA layer
    )
    mlp_owned = (
        int(c["mlp_fused_gate_up_swiglu"])
        + int(c["mlp_down_proj"])
        + int(c["mlp_add_residual_rmsnorm"])
    )
    gemv_owned = (
        int(c["embed_lookup"])
        + int(c["dn_inproj_pair_concat"])
        + int(c["dn_out_proj"])
        + int(c["gqa_fused_qkv"])
        + int(c["gqa_o_proj"])
        + int(c["mlp_fused_gate_up_swiglu"])
        + int(c["mlp_down_proj"])
        + int(c["lm_head"])
    )
    residual_owned = (
        int(c["mixer_add_residual_rmsnorm"]) + int(c["mlp_add_residual_rmsnorm"])
    )
    if dn_owned + gqa_owned + mlp_owned + int(c["embed_lookup"]) + int(c["lm_head"]) + int(
        c["argmax"]
    ) != ESTABLISHED_SEALED:
        raise MotifRefuse(
            f"region ownership drifted: dn={dn_owned} gqa={gqa_owned} "
            f"mlp={mlp_owned} terminals="
            f"{int(c['embed_lookup'])+int(c['lm_head'])+int(c['argmax'])}"
        )

    rows = [
        _candidate(
            id="dn_ba_delta_existing_lever",
            form="long_lived_state_machine",
            judgment="EXISTING_FUSION_LEVER_NOT_A_NEW_REGION",
            motifs=("dn_ba_to_decay", "dn_gated_delta"),
            sealed_dispatches=int(c["dn_ba_to_decay"]) + int(c["dn_gated_delta"]),
            launches_after=int(c["dn_gated_delta"]),  # folded into one
            risk=RISK_EXISTING_LEVER,
            metal_blockers={
                "argument_buffers": (
                    "the fused sibling already exists "
                    "(qwen38_gated_delta_decode_vi_simd_ba); it still uses "
                    "set_bytes for heads/kd/vd, same as the split pair"
                ),
                "icb": "not required; this is one kernel, not graph replay",
                "resource_residency": (
                    "rec_state is already UMA-resident at a per-layer offset"
                ),
                "dynamic_shapes": "none; DeltaNet is O(1) in sequence length",
                "routing_dependence": "none; mixer kind is static in layer index",
            },
            blocked_today=False,
            cheapest_falsifier=(
                "Re-walk with Fusion.ba_delta=True must emit 580 and reconcile. "
                "A later GPU timestamp A/B of HAWKING_QWEN38_FUSE_BA_DELTA=1 vs "
                "unset, token-identical, zero fallback, answers whether those "
                "48 launches are this 6.25 us class. Named, not run: this "
                "module is STATIC_ONLY."
            ),
            why=(
                "encode_dn_ba_and_delta already folds ba_to_decay into gated-"
                "delta when FUSE_BA_DELTA=1. Source comment: '48 launches on "
                "the 628 graph. Default Off'. This is the inner cut of the DN "
                "state machine, not a new region, and 628 -> 580 is exactly "
                "the 628 -> 500-class increment the task said not to target. "
                "Recorded so the census does not pretend the lever is absent."
            ),
            not_a_fusion_plan=(
                "listed as a lever that already exists, ranked separately from "
                "region candidates that would absorb a whole operation sequence"
            ),
        ),
        _candidate(
            id="mlp_suffix_representation_region",
            form="representation_native_region",
            judgment="YES_REPRESENTATION_NATIVE",
            motifs=(
                "mlp_fused_gate_up_swiglu",
                "mlp_down_proj",
                "mlp_add_residual_rmsnorm",
            ),
            sealed_dispatches=mlp_owned,
            launches_after=n_mlp,  # one launch per layer; cannot cross the mixer
            risk=RISK_EPILOGUE_FUSION,
            metal_blockers={
                "argument_buffers": (
                    "a fused kernel can keep layer weights as buffer bindings; "
                    "ICB replay of the current three would need argument-buffer "
                    "scalars because all three use set_bytes for rows/cols"
                ),
                "icb": (
                    "wrong textbook for the 6.25 us class. Kernel fusion of "
                    "down_proj epilogue + residual RMSNorm is the Metal-native "
                    "move, the same class as FUSE_MLP=swiglu which already landed"
                ),
                "resource_residency": (
                    "weights already resident; activations are the workspace "
                    "buffers the three launches already share"
                ),
                "dynamic_shapes": "none; hidden and intermediate are compile-time constants",
                "routing_dependence": "none; MLP is dense on this model",
            },
            blocked_today=False,
            cheapest_falsifier=(
                "GPU timestamps on isolated 64-layer MLP (already "
                "measure_isolated_dense_mlp) vs a fused down+add_rms sibling, "
                "token-identical. If 128 fewer launches do not save on the "
                "order of 128 x 6.25 us = 0.80 ms, this suffix is not that tax "
                "class. STATIC_ONLY: named, not run."
            ),
            why=(
                "The sealed MLP suffix is already a 3-launch operation "
                "sequence, identical 64 times, static grids, no recurrent "
                "state, no growing KV. Producer-consumer is closed inside the "
                "layer: fused gate_up_swiglu writes act, down_proj reads act, "
                "residual_rmsnorm reads down. It cannot loop all 64 MLPs in "
                "one launch because a mixer sits between layer i and i+1. "
                "One PersistentPhysicalRegion per layer with a layer-index "
                "weight slot is the honest form. 192 -> 64 removes 128."
            ),
            not_a_fusion_plan=(
                "the motif is the 64-wide repetition of a 3-launch suffix; "
                "the region judgment is whether that suffix can stay resident, "
                "not a list of further FUSE_* env bits"
            ),
        ),
        _candidate(
            id="dn_layer_state_machine",
            form="long_lived_state_machine",
            judgment="YES_STATE_MACHINE",
            motifs=(
                "mixer_input_rmsnorm",
                "dn_inproj_pair_concat",
                "dn_rearrange_conv",
                "dn_ba_to_decay",
                "dn_gated_delta",
                "dn_gated_rmsnorm",
                "dn_out_proj",
                "mixer_add_residual_rmsnorm",
            ),
            sealed_dispatches=dn_owned,
            launches_after=n_dn,  # one DN-layer region, replayed per DN layer
            risk=RISK_LAYER_WITH_DYNAMIC_SEQ,  # mega-kernel; seq is static, kernel is not small
            metal_blockers={
                "argument_buffers": (
                    "seven kernels, most using set_bytes for layout constants "
                    "(heads, kd, vd, conv kernel, rms eps). A persistent layer "
                    "kernel can bake those; ICB replay cannot capture set_bytes "
                    "without promoting them to argument-buffer scalars"
                ),
                "icb": (
                    "graph replay of 7 serial launches does not keep conv_state "
                    "and rec_state in registers/threadgroup across tokens. A "
                    "state machine does. ICB is the wrong textbook"
                ),
                "resource_residency": (
                    "conv_state and rec_state are already sequence-scoped UMA "
                    "buffers with per-layer offsets. The region is occupancy "
                    "and binding lifetime, not a copy-in"
                ),
                "dynamic_shapes": (
                    "none. DeltaNet rearrange/delta are O(1) in sequence "
                    "length — this is why DN is the cheaper state machine"
                ),
                "routing_dependence": (
                    "mixer kind is static: GQA every 4th layer. A 48-layer DN "
                    "pipeline cannot run without yielding to GQA. The honest "
                    "region is one DN layer with a layer slot, replayed 48 "
                    "times, not one region covering all 48"
                ),
            },
            blocked_today=False,
            cheapest_falsifier=(
                "The inner cut already in tree: FUSE_BA_DELTA removes 48 of "
                "these 337. A GPU timestamp A/B of that lever, token-identical, "
                "zero fallback, is the cheapest probe of whether DN launches "
                "are the 6.25 us class. If 48 fewer launches do not save on "
                "the order of 0.30 ms, a 7-kernel DN region is the wrong bet. "
                "STATIC_ONLY: named, not run."
            ),
            why=(
                "Sealed DeltaNet is a 7-launch sequence (8 on layer 0 because "
                "input RMSNorm still runs): inproj -> conv/rearrange -> "
                "ba_to_decay -> gated_delta -> gated_rmsnorm -> out_proj -> "
                "residual_rmsnorm. conv_state and rec_state are the LocalState "
                "the PersistentPhysicalRegion contract asks to keep valid. "
                "FPGA spatial pipeline is the textbook; CUDA persistent kernel "
                "is the occupancy textbook. 337 -> 48 removes 289. GQA "
                "interrupts every 4th layer, so this is not 'one region for "
                "all 48 DN layers'."
            ),
            not_a_fusion_plan=(
                "the motif is the repeated DN operation sequence plus its "
                "resident state; fusing ba_to_decay is one inner cut of it, "
                "not the region"
            ),
        ),
        _candidate(
            id="gqa_layer_static_skeleton",
            form="static_skeleton_with_dynamic_token_or_route_slots",
            judgment="YES_STATIC_SKELETON",
            motifs=(
                "gqa_fused_qkv",
                "gqa_qk_norm_rope_cache",
                "gqa_mha_decode",
                "gqa_sigmoid_gate",
                "gqa_o_proj",
                "mixer_add_residual_rmsnorm",
            ),
            sealed_dispatches=gqa_owned,
            launches_after=n_gqa,
            risk=RISK_LAYER_WITH_DYNAMIC_SEQ,
            metal_blockers={
                "argument_buffers": (
                    "mha_decode_f32_tcb already packs seq_len into a "
                    "KernelArgBuffer. RoPE and sigmoid still use set_bytes "
                    "for position / dims. ICB capture of the 6-launch sequence "
                    "needs those scalars promoted"
                ),
                "icb": (
                    "CUDA-graph recapture problem: MHA threadgroup memory is "
                    "(seq_len + tg) * 4 bytes and grows every token. ICB "
                    "replay is legal only if TG memory is allocated at "
                    "max_seq_len. Host encode savings would not be the 6.25 us "
                    "class"
                ),
                "resource_residency": (
                    "KV cache is already sequence-scoped UMA at a per-layer "
                    "slot; residency is not the blocker"
                ),
                "dynamic_shapes": (
                    "seq_len = position+1 is the only growing quantity on the "
                    "628 graph. Grid is static (n_heads * tg). The dynamic "
                    "slot is threadgroup memory length, not dispatch topology. "
                    "TPU static execution would pad to max_seq; Metal must "
                    "either pad or recapture"
                ),
                "routing_dependence": (
                    "sigmoid gate is value-dependent arithmetic, not topology. "
                    "No expert routing"
                ),
            },
            blocked_today=True,
            cheapest_falsifier=(
                "Pad mha_decode_f32 threadgroup memory to max_seq_len and "
                "count whether the encode path is then a static DAG (it is, "
                "if position stays a scalar slot). That is a source reading "
                "plus a kernel change, not a GPU lease. A later timestamp A/B "
                "against the growing-TG form tells whether padding is free."
            ),
            why=(
                "Sealed GQA is a 6-launch sequence: fused QKV -> QK-norm/RoPE/"
                "cache-write -> MHA -> sigmoid -> o_proj -> residual_rmsnorm. "
                "16 times. The skeleton is static except MHA's TG memory. "
                "96 -> 16 removes 80. Blocked today by that growing TG "
                "allocation, which is a dynamic shape even though the grid "
                "is not."
            ),
            not_a_fusion_plan=(
                "the motif is the GQA operation sequence with a position slot; "
                "QKV fusion already landed and is not the remaining question"
            ),
        ),
        _candidate(
            id="residual_rmsnorm_is_a_boundary_not_a_region",
            form="producer_consumer_boundary",
            judgment="NO_PRODUCER_CONSUMER_BOUNDARY",
            motifs=("mixer_add_residual_rmsnorm", "mlp_add_residual_rmsnorm"),
            sealed_dispatches=residual_owned,
            launches_after=residual_owned,  # cannot collapse without producers
            risk=RISK_EPILOGUE_FUSION,
            metal_blockers={
                "argument_buffers": "n/a as a standalone region",
                "icb": (
                    "replaying 128 residual launches without the GEMVs that "
                    "produce them is not a legal graph"
                ),
                "resource_residency": "n/a",
                "dynamic_shapes": "none",
                "routing_dependence": "none",
            },
            blocked_today=True,
            cheapest_falsifier=(
                "Data dependence is already in source: mixer residual consumes "
                "mixer out_proj/o_proj, MLP residual consumes down_proj, and "
                "the next mixer consumes the MLP residual's RMSNorm output. "
                "A walk that tried to group all 128 residual launches into one "
                "region would reorder across those edges. This module refuses "
                "that grouping rather than emit it."
            ),
            why=(
                "128 sealed residual+RMSNorm launches look like a high-"
                "frequency motif, and they are. They are not a region. They "
                "sit on the producer-consumer boundary between mixer and MLP "
                "and between layer i and layer i+1. Collapsing them requires "
                "absorbing the preceding GEMV (mlp_suffix / dn_layer / "
                "gqa_layer). Standalone: 128 -> 128, zero removed."
            ),
            not_a_fusion_plan=(
                "this is a negative: FUSE_ADD_RMSNORM already folded the "
                "adjacent RMSNorm in; further folding belongs to the producer "
                "regions, not to residual as a thing"
            ),
        ),
        _candidate(
            id="representation_decode_persistent_queue",
            form="representation_native_region",
            judgment="YES_REPRESENTATION_NATIVE",
            motifs=(
                "embed_lookup",
                "dn_inproj_pair_concat",
                "dn_out_proj",
                "gqa_fused_qkv",
                "gqa_o_proj",
                "mlp_fused_gate_up_swiglu",
                "mlp_down_proj",
                "lm_head",
            ),
            sealed_dispatches=gemv_owned,
            launches_after=1,
            risk=RISK_NEW_EXECUTION_MODEL,
            metal_blockers={
                "argument_buffers": (
                    "a device-side worklist of (codes, scales, in, out, rows) "
                    "is exactly an argument-buffer table. Today's GEMV helpers "
                    "bind those per launch with set_buffer + set_bytes"
                ),
                "icb": (
                    "GPU-encoded MTLIndirectCommandBuffer is Metal's persistent-"
                    "kernel analogue. Host-encoded ICB is Type-1 killed. Metal "
                    "compute cannot dispatch other kernels except by encoding "
                    "into an ICB. Blocked today: no such interpreter is in the "
                    "decode path, and most GEMVs still use set_bytes"
                ),
                "resource_residency": (
                    "weights are already resident. The queue does not move them"
                ),
                "dynamic_shapes": (
                    "GEMV grids differ by organ (hidden, intermediate, vocab, "
                    "concat rows). A persistent kernel can loop variable rows; "
                    "a captured ICB needs one command per shape class"
                ),
                "routing_dependence": (
                    "GEMVs are interleaved with DN/GQA consumers. You cannot "
                    "run all 258 representation-decode launches first. A "
                    "persistent kernel has to stay alive across the whole "
                    "token and accept a sequenced worklist"
                ),
            },
            blocked_today=True,
            cheapest_falsifier=(
                "Count distinct GEMV launch geometries on the sealed graph "
                "(already in this census: pair-concat, fused QKV, fused "
                "gate_up_swiglu, vanilla geo_tpr64, embed lookup, lm_head). "
                "If they do not share a dispatch ABI, a single persistent "
                "queue is several kernels wearing a trench coat. That is a "
                "source reading."
            ),
            why=(
                "258 of 628 launches are representation decode of packed "
                "weights. That is the high-frequency family. It is not one "
                "region today because the GEMVs are sequenced through mixer "
                "and MLP residuals. CUDA persistent kernels / FPGA spatial "
                "GEMV pipelines are the textbooks. 258 -> 1 would remove 257 "
                "and is blocked on Metal by the absence of a device-side "
                "enqueue that keeps occupancy between organs."
            ),
            not_a_fusion_plan=(
                "grouping by representation decode, not by which FUSE_* bit "
                "would merge two of these GEMVs"
            ),
        ),
        _candidate(
            id="hybrid_4_layer_tile",
            form="static_skeleton_with_dynamic_token_or_route_slots",
            judgment="YES_STATIC_SKELETON",
            motifs=tuple(
                m
                for m in CATALOG_BY_ID
                if m
                not in {
                    "embed_lookup",
                    "lm_head",
                    "argmax",
                    "final_rmsnorm",
                }
            ),
            sealed_dispatches=ESTABLISHED_SEALED
            - int(c["embed_lookup"])
            - int(c["lm_head"])
            - int(c["argmax"]),
            launches_after=n_gqa,  # one tile per GQA, i.e. one per attention interval
            risk=RISK_NEW_EXECUTION_MODEL,
            metal_blockers={
                "argument_buffers": (
                    "a tile kernel / ICB would have to rebind every organ's "
                    "weights; argument buffers are the slot mechanism"
                ),
                "icb": (
                    "the 4-layer tile is the CUDA-graph unit one would "
                    "capture if MHA TG memory were padded to max_seq. Host "
                    "replay still would not be the 6.25 us class"
                ),
                "resource_residency": "weights and state already resident",
                "dynamic_shapes": (
                    "inherits GQA MHA's growing threadgroup memory; the DN "
                    "half of the tile is static"
                ),
                "routing_dependence": (
                    "none. (layer+1) % 4 == 0 is compile-time. The tile is "
                    "the static mixer schedule, not a route"
                ),
            },
            blocked_today=True,
            cheapest_falsifier=(
                "Walk layers 4..7 and 8..11 and assert identical motif "
                "sequences (this module already does: 39 launches per tile "
                "after layer 0). That confirms the skeleton. The remaining "
                "blocker is GQA TG memory, same as gqa_layer_static_skeleton."
            ),
            why=(
                "After layer 0's extra RMSNorm, the stack is 15 identical "
                "39-dispatch tiles of DN,DN,DN,GQA plus four MLP suffixes, "
                "plus one 40-dispatch first tile. Mixer kind is static, so "
                "the tile is a static skeleton with a layer-base slot. 625 "
                "layer launches -> 16 tile launches removes 609. Blocked by "
                "the GQA dynamic TG memory inside the tile and by the lack of "
                "a Metal persistent occupancy story for a 39-kernel DAG."
            ),
            not_a_fusion_plan=(
                "the motif is layer repetition of the hybrid schedule, which "
                "is already in qwen38_mixer_kind; grouping tiles is not a new "
                "FUSE_* bit"
            ),
        ),
        _candidate(
            id="token_graph_persistent_executor",
            form="graph_replay_equivalent",
            judgment="YES_GRAPH_REPLAY",
            motifs=tuple(row["id"] for row in MOTIF_CATALOG),
            sealed_dispatches=ESTABLISHED_SEALED,
            launches_after=1,
            risk=RISK_NEW_EXECUTION_MODEL,
            metal_blockers={
                "argument_buffers": (
                    "almost every launch on this graph uses set_bytes for "
                    "shapes, eps, position, or token id. L6: ICB cannot "
                    "capture set_bytes without argument-buffer scalars. "
                    "Embed's token id and GQA's position are the actual "
                    "per-token slots; the rest are constants"
                ),
                "icb": (
                    "MTLIndirectCommandBuffer is Metal's CUDA-graph analogue. "
                    "Type-1 killed for this decode token graph: the cost ICB "
                    "removes (host encode) had already been removed by "
                    "residency batching (~0.4% of wall). serial_token_encoder "
                    "already covers the token with one compute encoder and "
                    "did not separate complete-wall. ICB replay is not the "
                    "PersistentPhysicalRegion that removes on-device launch"
                ),
                "resource_residency": (
                    "weights and sequence state are already resident on UMA; "
                    "the contract's residency preconditions hold"
                ),
                "dynamic_shapes": (
                    "embed token_id (scalar), RoPE position (scalar), MHA "
                    "seq_len in an argument buffer, MHA threadgroup memory "
                    "growing with seq. Across tokens the only topology change "
                    "is that TG length. Pad it and the DAG is static"
                ),
                "routing_dependence": (
                    "none on this model. Argmax -> next embed token is a "
                    "slot, not a route. Mixer kind is static"
                ),
            },
            blocked_today=True,
            cheapest_falsifier=(
                "Already run: serial_token_encoder (one compute encoder, "
                "still 628 dispatches inside) did not separate wall "
                "(DISPATCH_CEREMONY / GPU_IDLE_GAP). ICB pre-encode was "
                "Type-1 killed (NEGATIVE_SCIENCE L6). A new falsifier is "
                "only needed if someone claims a *device-side* persistent "
                "occupancy region, which this tree does not have. Per-kernel "
                "GPU timestamps for one decode step remain the cheapest probe "
                "of whether the 628 survivors are the 6.25 us class at all."
            ),
            why=(
                "The PersistentPhysicalRegion invariant: when state and "
                "bindings remain valid, repeated host entry is not "
                "semantically required. TPU static execution / CUDA Graphs / "
                "FPGA spatial pipelines are the textbooks. On Metal today "
                "the graph-replay equivalent (ICB) does not remove the cost "
                "class that fusion's 336 actually moved. A long-lived "
                "on-device state machine would. 628 -> 1 removes 627 host "
                "launches and an unknown fraction of on-device launch tax. "
                "That unknown is why this is not 'zero the token'."
            ),
            not_a_fusion_plan=(
                "this is the whole token as one region, which is the question "
                "the task asked; it is not 628 -> 500"
            ),
        ),
    ]
    return rows


def rank_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def key(row: Mapping[str, Any]) -> tuple[float, int]:
        ratio = row.get("removed_per_risk_unit")
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
            ratio = -1.0
        return (float(ratio), int(row.get("dispatches_removed") or 0))

    ranked = sorted(candidates, key=key, reverse=True)
    unblocked = [row for row in ranked if not row.get("blocked_today") and int(row["dispatches_removed"]) > 0]
    regions = [
        row
        for row in ranked
        if row.get("judgment") not in {
            "EXISTING_FUSION_LEVER_NOT_A_NEW_REGION",
            "NO_PRODUCER_CONSUMER_BOUNDARY",
        }
        and int(row["dispatches_removed"]) > 0
    ]
    unblocked_regions = [
        row for row in regions if not row.get("blocked_today")
    ]
    top = unblocked_regions[0] if unblocked_regions else (unblocked[0] if unblocked else ranked[0])
    return {
        "order": [row["id"] for row in ranked],
        "order_unblocked": [row["id"] for row in unblocked],
        "order_region_candidates": [row["id"] for row in regions],
        "order_unblocked_regions": [row["id"] for row in unblocked_regions],
        "top_unblocked_region": top["id"],
        "top_removed_per_risk_unit": top.get("removed_per_risk_unit"),
        "top_dispatches_removed": top.get("dispatches_removed"),
        "cheapest_falsifier_for_top": top.get("cheapest_falsifier"),
        "risk_legend": {
            str(RISK_EXISTING_LEVER): "existing lever already in tree",
            str(RISK_EPILOGUE_FUSION): "adjacent-kernel epilogue fusion, static grids",
            str(RISK_LAYER_STATE_MACHINE): "multi-kernel layer state machine, static seq",
            str(RISK_LAYER_WITH_DYNAMIC_SEQ): (
                "layer state machine or mega-kernel; dynamic seq or large kernel"
            ),
            str(RISK_NEW_EXECUTION_MODEL): (
                "new execution model (persistent kernel / GPU ICB interpreter / "
                "whole-token region)"
            ),
        },
        "ranking_rule": (
            "dispatches_removed / risk, then dispatches_removed. Residual-as-"
            "region (0 removed) and existing levers are listed but the named "
            "top is the highest-ratio unblocked region candidate. The target "
            "is not 628 -> 500; an existing 48-launch lever is recorded so it "
            "is not rediscovered as a region."
        ),
    }


def four_layer_tile(
    launches: Sequence[Mapping[str, Any]], interval: int
) -> dict[str, Any]:
    """Layer-repetition structure: GQA every `interval`th layer, first tile has +1 RMS."""
    if interval <= 0:
        raise MotifRefuse(f"attention interval {interval} is not a tile size")
    by_layer: dict[int, list[str]] = {}
    for atom in launches:
        layer = atom.get("layer")
        if layer is None:
            continue
        by_layer.setdefault(int(layer), []).append(str(atom["motif_id"]))
    if not by_layer:
        raise MotifRefuse("no per-layer launches")
    tiles = []
    layers = sorted(by_layer)
    for start in range(0, len(layers), interval):
        group = layers[start : start + interval]
        seq = [by_layer[i] for i in group]
        tiles.append(
            {
                "layers": group,
                "kinds": [mixer_kind(i, interval) for i in group],
                "dispatches": sum(len(s) for s in seq),
                "per_layer": [len(s) for s in seq],
            }
        )
    first = tiles[0]["dispatches"] if tiles else None
    rest = {t["dispatches"] for t in tiles[1:]}
    return {
        "n_tiles": len(tiles),
        "first_tile_dispatches": first,
        "later_tile_dispatches": sorted(rest),
        "later_tiles_identical": len(rest) == 1,
        "tiles": tiles,
        "reading": (
            "mixer_kind is (layer+1) % 4 == 0 -> GQA. After layer 0's extra "
            "input RMSNorm, each 4-layer tile is an identical dispatch "
            "sequence. That is the layer-repetition motif."
        ),
    }


def fusion_delta(sealed: Mapping[str, int], unfused: Mapping[str, int]) -> dict[str, Any]:
    """What the 336 are, as motif multiplicity changes. Not a next plan."""
    rows = []
    for spec in MOTIF_CATALOG:
        sid = spec["id"]
        delta = int(unfused[sid]) - int(sealed[sid])
        if delta:
            rows.append(
                {
                    "id": sid,
                    "unfused": int(unfused[sid]),
                    "sealed": int(sealed[sid]),
                    "removed_or_added": delta,
                }
            )
    removed = sum(max(r["removed_or_added"], 0) for r in rows)
    added = sum(max(-r["removed_or_added"], 0) for r in rows)
    net = removed - added
    return {
        "net_removed": net,
        "launches_that_vanished": removed,
        "launches_that_appeared": added,
        "rows": rows,
        "reading": (
            "Fusion does not delete work; it replaces launch kinds. "
            f"{removed} unfused launches vanished, {added} sealed launches "
            f"appeared (the fused kernels), net {net}. Established net is "
            f"{ESTABLISHED_FUSION_REMOVED}."
        ),
    }


# ---------------------------------------------------------------------------
# Build / record
# ---------------------------------------------------------------------------


def analyze() -> dict[str, Any]:
    started = time.perf_counter()
    geo = load_geometry()
    markers = decode_path_markers()
    helpers = helper_markers()
    if not markers["ok"]:
        raise MotifRefuse(
            f"decode path markers missing {markers['missing']}; refusing to "
            "invent a motif census"
        )
    if not helpers["ok"]:
        raise MotifRefuse(
            f"encode helpers missing {helpers['missing']}; refusing to "
            "cluster a graph we cannot see"
        )

    sealed_fusion = Fusion.sealed_resident()
    unfused_fusion = Fusion.env_unset_default()
    if sealed_fusion.ba_delta or unfused_fusion.ba_delta:
        raise MotifRefuse("production walks must have ba_delta Off")
    if sealed_fusion.argmax_two_pass or unfused_fusion.argmax_two_pass:
        raise MotifRefuse("production walks must have argmax two-pass Off")

    sealed_launches = walk_launches(geo, sealed_fusion)
    unfused_launches = walk_launches(geo, unfused_fusion)
    sealed_counts = cluster_launches(sealed_launches)
    unfused_counts = cluster_launches(unfused_launches)

    # Independent oracle: tps_budget's coarse walk. Two methods, one integer.
    coarse_sealed = count_dispatches_per_decoded_token(geo, sealed_fusion)
    coarse_unfused = count_dispatches_per_decoded_token(geo, unfused_fusion)
    if int(coarse_sealed["total"]) != ESTABLISHED_SEALED:
        raise MotifRefuse(
            f"tps_budget sealed walk is {coarse_sealed['total']}, established "
            f"{ESTABLISHED_SEALED}; refusing"
        )
    if int(coarse_unfused["total"]) != ESTABLISHED_UNFUSED:
        raise MotifRefuse(
            f"tps_budget unfused walk is {coarse_unfused['total']}, established "
            f"{ESTABLISHED_UNFUSED}; refusing"
        )
    if len(sealed_launches) != ESTABLISHED_SEALED:
        raise MotifRefuse(
            f"fine sealed walk is {len(sealed_launches)}, established "
            f"{ESTABLISHED_SEALED}; refusing"
        )
    if len(unfused_launches) != ESTABLISHED_UNFUSED:
        raise MotifRefuse(
            f"fine unfused walk is {len(unfused_launches)}, established "
            f"{ESTABLISHED_UNFUSED}; refusing"
        )

    reconciliation = reconcile_census(sealed_counts, unfused_counts)

    ba_delta_fusion = Fusion(
        mlp="swiglu",
        gqa_qkv=True,
        dn_inproj=True,
        add_rmsnorm=True,
        ba_delta=True,
        argmax_two_pass=False,
    )
    ba_delta_launches = walk_launches(geo, ba_delta_fusion)
    if len(ba_delta_launches) != ESTABLISHED_SEALED - 48:
        raise MotifRefuse(
            f"sealed+ba_delta walk is {len(ba_delta_launches)}, expected "
            f"{ESTABLISHED_SEALED - 48}"
        )

    motifs = [
        motif_row(spec["id"], sealed_counts[spec["id"]], unfused_counts[spec["id"]])
        for spec in MOTIF_CATALOG
    ]
    high_frequency = [row for row in motifs if row["high_frequency"]]
    high_frequency.sort(key=lambda r: (-r["sealed_count"], r["id"]))
    high_frequency_sealed = sum(int(row["sealed_count"]) for row in high_frequency)

    candidates = region_candidates(sealed_counts)
    ranking = rank_candidates(candidates)
    tiles = four_layer_tile(
        sealed_launches, int(geo["QWEN38_FULL_ATTENTION_INTERVAL"])
    )
    if not tiles["later_tiles_identical"]:
        raise MotifRefuse(
            f"later 4-layer tiles are not identical: {tiles['later_tile_dispatches']}"
        )

    budget_rel = None
    try:
        budget_rel = load_json(REPO / RESIDENT_BUDGET_REL)
        cited_628 = budget_rel.get("derived", {}).get("production_dispatches_per_token")
        cited_964 = (
            budget_rel.get("ab", {})
            .get("unfused", {})
            .get("dispatches_per_decode_step")
        )
        if cited_628 not in (ESTABLISHED_SEALED, float(ESTABLISHED_SEALED)):
            raise MotifRefuse(
                f"{RESIDENT_BUDGET_REL} production_dispatches_per_token is "
                f"{cited_628}, not {ESTABLISHED_SEALED}"
            )
        if cited_964 not in (ESTABLISHED_UNFUSED, float(ESTABLISHED_UNFUSED)):
            raise MotifRefuse(
                f"{RESIDENT_BUDGET_REL} unfused dispatches are {cited_964}, "
                f"not {ESTABLISHED_UNFUSED}"
            )
    except MotifRefuse:
        raise
    except Exception as exc:
        raise MotifRefuse(
            f"{RESIDENT_BUDGET_REL} unreadable; refusing to cite 6.25 us "
            f"without the receipt ({exc})"
        ) from exc

    cpu_s = time.perf_counter() - started
    sealed_families = reconciliation["sealed_families"]
    unfused_families = reconciliation["unfused_families"]

    return {
        "purpose": (
            "Cluster the 628 (and 964) dispatches per decoded token into "
            "semantic motifs, judge which could become a "
            "PersistentPhysicalRegion, and refuse an unreconciled census."
        ),
        "schema": SCHEMA,
        "version": VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "established": {
            "sealed_dispatches_per_decoded_token": ESTABLISHED_SEALED,
            "unfused_dispatches_per_decoded_token": ESTABLISHED_UNFUSED,
            "fusion_removed": ESTABLISHED_FUSION_REMOVED,
            "cited_marginal_us": CITED_MARGINAL_US,
            "cited_fusion_saved_ms": CITED_FUSION_SAVED_MS,
            "cited_from": RESIDENT_BUDGET_REL,
            "do_not_extrapolate_linearly_to_zero": True,
            "do_not_assume_remaining_628_cost_the_same": True,
            "marginal_caveat": MARGINAL_CAVEAT,
            "layer_structure": {
                "deltanet_layers": 48,
                "gqa_layers": 16,
                "mlp_layers": 64,
                "embed": 1,
                "terminal": "lm_head + argmax (final RMSNorm folded under sealed)",
                "mixer_rule": "(layer + 1) % QWEN38_FULL_ATTENTION_INTERVAL == 0 -> GQA",
            },
        },
        "geometry_from_source": {
            "source": "crates/hawking-core/src/model/qwen38_geometry.rs",
            "constants": dict(geo),
        },
        "decode_path_markers": markers,
        "helper_markers": helpers,
        "walk": {
            "path": (
                f"{DECODE_SRC} generate_greedy -> "
                "Qwen38HybridDecodeSession::step -> encode_full_token "
                "(encode_embed + encode_layers + encode_terminal)"
            ),
            "grain": "one dispatch_threads helper = one launch",
            "not_copied_from_a_receipt": True,
            "gpu_counter": False,
            "sealed_launches": len(sealed_launches),
            "unfused_launches": len(unfused_launches),
            "agrees_with_tps_budget_coarse_walk": True,
            "tps_budget_sealed": int(coarse_sealed["total"]),
            "tps_budget_unfused": int(coarse_unfused["total"]),
            "ba_delta_on_sealed_would_be": len(ba_delta_launches),
            "split_deltanet_projections_not_walked": True,
            "mixed_vs_q4_does_not_change_count": True,
        },
        "reconciliation": reconciliation,
        "motifs": motifs,
        "high_frequency_motifs": high_frequency,
        "high_frequency_threshold": HIGH_FREQUENCY_MIN,
        "high_frequency_sealed_sum": high_frequency_sealed,
        "high_frequency_reading": (
            f"{high_frequency_sealed} of {ESTABLISHED_SEALED} sealed launches "
            f"belong to motifs that fire at least {HIGH_FREQUENCY_MIN} times "
            "(the 64-wide MLP suffix and fused residuals, the 48-wide "
            "DeltaNet sequence, the 16-wide GQA sequence). The four leftover "
            "singletons are embed, layer-0 mixer RMSNorm, lm_head, and argmax."
        ),
        "families": {
            "axis": (
                "representation_decode | state_update | producer_consumer | "
                "norm | residual | routing | elementwise"
            ),
            "sealed": sealed_families,
            "unfused": unfused_families,
            "reading": (
                "Each launch has one primary family, so families partition. "
                "Fused add_residual_rmsnorm is counted as residual and tagged "
                "as norm. elementwise is SwiGLU, which sealed fusion folded "
                "into the GEMV (count 0). routing is the GQA sigmoid gate: "
                "value-dependent arithmetic, not expert routing. This model "
                "has no MoE on the decode path; mixer kind is static."
            ),
        },
        "fusion_delta_is_not_a_plan": fusion_delta(sealed_counts, unfused_counts),
        "layer_repetition": tiles,
        "persistent_physical_region": {
            "primitive": "PersistentPhysicalRegion",
            "contract_source": PHYSICAL_PRIMITIVES_REL,
            "invariant": (
                "when state and bindings remain valid, repeated host entry "
                "is not semantically required"
            ),
            "textbooks": {
                "cuda_graphs": "capture/replay of a static launch DAG (host encode)",
                "persistent_kernels": (
                    "one grid stays occupied and pulls work (on-device occupancy)"
                ),
                "tpu_static_execution": "compiled program with dynamic slots",
                "fpga_spatial_pipelines": "one region wired in space",
                "metal_authority": (
                    "MTLIndirectCommandBuffer is the graph-replay analogue and "
                    "is the wrong textbook for the remaining 6.25 us class. "
                    "A persistent occupancy region or a fused kernel is the "
                    "right one. Metal has no CUDA-style persistent kernel API; "
                    "GPU-encoded ICB plus argument buffers is the closest, "
                    "and host-encoded ICB was Type-1 killed."
                ),
            },
            "forms": (
                "static_skeleton_with_dynamic_token_or_route_slots",
                "graph_replay_equivalent",
                "representation_native_region",
                "long_lived_state_machine",
            ),
        },
        "candidates": candidates,
        "ranking": ranking,
        "target_is_not_628_to_500": {
            "what_628_to_500_would_be": (
                "removing 128 launches. That number is coincidentally the MLP "
                "suffix (192 -> 64) or add_rmsnorm's already-landed 128, and "
                "is the wrong goalpost. A region either absorbs a repeating "
                "operation sequence (hundreds) or it is an inner cut of one."
            ),
            "existing_inner_cut_still_off": {
                "lever": "HAWKING_QWEN38_FUSE_BA_DELTA",
                "would_remove": 48,
                "would_leave": ESTABLISHED_SEALED - 48,
                "why_listed": "so it is not rediscovered as the region answer",
            },
        },
        "loads": {
            "resident_token_budget": {
                "rel": RESIDENT_BUDGET_REL,
                "status": "LOADED",
            },
            "tps_budget_coarse_walk": {
                "sealed": int(coarse_sealed["total"]),
                "unfused": int(coarse_unfused["total"]),
            },
        },
        "self_timing": {
            "class": "SELF_MEASURED_DIRTY",
            "cpu_parse_s": cpu_s,
            "not": (
                "a GPU measurement, a lease, a qualified TPS, a roof, or "
                "evidence the cited 6.25 us still holds on this host"
            ),
            "numbers_decide_nothing": True,
        },
        "resident_callable": {
            "entry_point": "tools.future.dispatch_motifs.build() / record()",
            "fails_closed": (
                "motif counts that do not sum to 628 and 964 raise MotifRefuse "
                "rather than emit; a truncated walk raises; missing encode "
                "helpers raise; HardwareClaimError on tps/wall_ns/gpu_ns/"
                "dispatch_ns keys; this module emits no GPU measurement it "
                "did not derive"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "gpu_authority": False,
            "evidence_class": EVIDENCE_CLASS,
        },
        "recovered_implementation": [
            f"{DECODE_SRC} encode_full_token / encode_embed / encode_layers / "
            "encode_deltanet / encode_gqa / encode_dense_mlp / encode_terminal",
            f"{DECODE_SRC} encode_fused_pair_concat / encode_fused_qkv / "
            "encode_fused_gate_up / encode_dn_ba_and_delta / "
            "encode_add_residual_rmsnorm",
            "crates/hawking-core/src/kernels/mod.rs mha_decode_f32_tcb "
            "(KernelArgBuffer seq_len; TG memory grows with seq)",
            "crates/hawking-core/src/kernels/mod.rs qwen_next_add_residual_tcb / "
            "sample_argmax_f32_tcb",
            "crates/hawking-core/src/model/qwen38_geometry.rs layer census / mixer_kind",
            "tools/future/tps_budget.py coarse encode-path walk (oracle, not the motif grain)",
            f"{RESIDENT_BUDGET_REL} 628/964/336 and the 6.25 us paired A/B",
            f"{PHYSICAL_PRIMITIVES_REL} PersistentPhysicalRegion contract",
            f"{NEGATIVE_INDEX_REL} L6 ICB Type-1 kill / set_bytes without argbuf scalars",
            f"{DISPATCH_CEREMONY_REL} serial encoder did not separate; host ceremony is not the token",
        ],
        "gaps_closed": [
            "628 and 964 partitioned into semantic motifs at launch grain, not copied as a headline",
            "each motif carries a PersistentPhysicalRegion judgment or an explicit negative",
            "unreconciled census refuses rather than emitting",
        ],
        "negative_findings": [
            "this module did not run a GPU benchmark and did not take a bench lock",
            "6.25 us is cited from the 336 fusion removed; products over the remaining 628 are not measurements of those motifs",
            "ICB / CUDA Graphs remove host encode, which is already not the token; they are the wrong textbook for the 6.25 us class",
            "serial_token_encoder already one compute encoder around 628 launches and did not separate wall",
            "the residual+RMSNorm motif (128) is a producer-consumer boundary and removes zero as a standalone region",
            "Qwen3.8 hybrid decode has no MoE routing; mixer kind is static; sigmoid is not a route",
            "the target is not 628 -> 500",
            "FUSE_BA_DELTA would remove 48 and is default Off; that is an inner cut, not the region",
        ],
    }


def build() -> dict[str, Any]:
    """In-memory document. Does not write. record() seals the receipt."""
    doc = analyze()
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        if key in doc and isinstance(doc[key], (int, float)):
            raise HardwareClaimError(f"{key} leaked into the motifs document")
    return doc


def record() -> Any:
    doc = build()
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    if not (args.record or args.build):
        ap.error("pass --record (or --build)")
    out = record()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
