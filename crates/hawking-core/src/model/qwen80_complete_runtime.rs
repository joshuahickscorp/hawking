//! Strict Qwen3-Coder-Next complete-binary catalog and native-state binding.
//!
//! Qwen3-Coder-Next is not Qwen30 with a larger expert table.  Its 48-token
//! mixer schedule is three Gated DeltaNet layers followed by one gated GQA
//! layer, repeated twelve times.  This module owns the exact source geometry,
//! all 74,391 direct-binary tensor names/shapes, and the native state layout
//! needed by a future complete decoder.
//!
//! It deliberately stops at structural artifact admission plus native state
//! allocation/direct packed-tensor upload.  It does not make a full token,
//! generate text, expose HCLI, use a BF16/MPS fallback, or emit TPS evidence.
//! Those remain impossible until the complete hybrid token graph is composed.

use super::qwen_complete_binary::{
    admit_complete_binary_artifact, decode_complete_binary_f32, parse_complete_binary_header,
    CompleteBinaryAdmission, CompleteBinaryArtifact, CompleteBinaryHeader, QwenCompleteBinaryModel,
};
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use half::f16;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(target_os = "macos")]
use crate::kernels::{
    moe_topk_gate_tcb, qwen_binary_sign_scale_matvec_component_tcb,
    qwen_complete_normalize_route_weights_tcb, qwen_next_add_residual_tcb,
    qwen_next_ba_to_decay_beta_tcb, qwen_next_deltanet_gated_rmsnorm_tcb,
    qwen_next_direct_packed_input_rmsnorm_tcb,
    qwen_next_gated_delta_decode_single_at_state_offset_tcb,
    qwen_next_gated_delta_decode_single_tcb, qwen_next_qkvz_rearrange_conv_l2_tcb,
};
#[cfg(target_os = "macos")]
use crate::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};

pub const QWEN80_COMPLETE_NATIVE_MAX_CONTEXT: usize = 4096;

const QWEN80_MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const QWEN80_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const QWEN80_ARCHITECTURE: &str = "Qwen3NextForCausalLM";
const QWEN80_MODEL_TYPE: &str = "qwen3_next";
const QWEN80_LAYERS: usize = 48;
const QWEN80_HIDDEN: usize = 2048;
const QWEN80_FULL_ATTN_HEADS: usize = 16;
const QWEN80_FULL_ATTN_KV_HEADS: usize = 2;
const QWEN80_FULL_ATTN_HEAD_DIM: usize = 256;
const QWEN80_LINEAR_KEY_HEADS: usize = 16;
const QWEN80_LINEAR_VALUE_HEADS: usize = 32;
const QWEN80_LINEAR_KEY_HEAD_DIM: usize = 128;
const QWEN80_LINEAR_VALUE_HEAD_DIM: usize = 128;
const QWEN80_LINEAR_CONV_KERNEL: usize = 4;
const QWEN80_EXPERTS: usize = 512;
const QWEN80_TOP_K: usize = 10;
const QWEN80_COMPLETE_BINARY_TENSORS: usize = 74_391;
const QWEN80_MOE_INTERMEDIATE: usize = 512;
const QWEN80_SHARED_EXPERT_INTERMEDIATE: usize = 512;
const QWEN80_VOCAB: usize = 151_936;
// The source tokenizer has 151,643 base entries plus 26 added entries.  The
// final 267 lm_head rows are source-reserved/unmapped ids; a future device
// sampler must mask them rather than decode a non-token id.
const QWEN80_TOKENIZER_VOCAB: usize = 151_669;
const QWEN80_GROUP_SIZE: usize = 128;
const QWEN80_FULL_ATTENTION_INTERVAL: usize = 4;
const QWEN80_DECODER_SPARSE_STEP: usize = 1;
const QWEN80_ROPE_THETA: f32 = 5_000_000.0;
const QWEN80_PARTIAL_ROTARY_FACTOR: f32 = 0.25;
const QWEN80_RMS_EPS: f32 = 1.0e-6;

// These labels are capture-only structural evidence, not a performance trace.
// A future source-token L0→L1 continuation is valid only when one fresh,
// non-timed TokenCommandBuffer owns exactly this 23+9 sequence before its
// single fence.  Keeping the lists here (next to the runtime encoders) makes
// a numeric dispatch count insufficient to authorize a continuation.
#[cfg(target_os = "macos")]
const QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS: [&str; 23] = [
    "qwen_next_direct_packed_input_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_qkvz_rearrange_conv_l2",
    "qwen_next_ba_to_decay_beta",
    "qwen_next_gated_delta_decode_single",
    "qwen_next_deltanet_gated_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_add_residual",
    "qwen80_postnorm_router_top10_rmsnorm",
    "qwen80_postnorm_router_top10_matvec",
    "qwen80_postnorm_router_top10_select",
    "qwen80_all_ten_routed_wave_route_guard",
    "qwen80_all_ten_routed_wave_gate_up",
    "qwen80_all_ten_routed_wave_swiglu",
    "qwen80_all_ten_routed_wave_down_weighted",
    "qwen80_shared_expert_wave_gate_up",
    "qwen80_shared_expert_wave_swiglu",
    "qwen80_shared_expert_wave_down",
    "qwen80_shared_expert_wave_scalar_gate",
    "qwen80_shared_expert_wave_apply_sigmoid_gate",
    "qwen80_moe_wave_aggregate_second_residual_route_sum",
    "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
];

#[cfg(target_os = "macos")]
const QWEN80_SOURCE_TOKEN_L1_DELTA_PREFIX_KERNELS: [&str; 9] = [
    "qwen_next_direct_packed_input_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_qkvz_rearrange_conv_l2",
    "qwen_next_ba_to_decay_beta",
    "qwen_next_gated_delta_decode_single",
    "qwen_next_deltanet_gated_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_add_residual",
];

// The only permitted Layer-1 completion suffix.  It is intentionally kept
// beside the L0 and L1-prefix trace lists so a later host cannot promote a
// raw dispatch count into a same-runtime component claim.  The full physical
// source-token boundary is exactly L0(23) + L1-prefix(9) + this suffix(14).
#[cfg(target_os = "macos")]
const QWEN80_SOURCE_TOKEN_L1_MOE_SUFFIX_KERNELS: [&str; 14] = [
    "qwen80_postnorm_router_top10_rmsnorm",
    "qwen80_postnorm_router_top10_matvec",
    "qwen80_postnorm_router_top10_select",
    "qwen80_all_ten_routed_wave_route_guard",
    "qwen80_all_ten_routed_wave_gate_up",
    "qwen80_all_ten_routed_wave_swiglu",
    "qwen80_all_ten_routed_wave_down_weighted",
    "qwen80_shared_expert_wave_gate_up",
    "qwen80_shared_expert_wave_swiglu",
    "qwen80_shared_expert_wave_down",
    "qwen80_shared_expert_wave_scalar_gate",
    "qwen80_shared_expert_wave_apply_sigmoid_gate",
    "qwen80_moe_wave_aggregate_second_residual_route_sum",
    "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
];

/// Multi-layer same-runtime DeltaNet chain encode (L0..L2+).
#[cfg(target_os = "macos")]
#[path = "qwen80_multi_layer_same_runtime_encode.rs"]
mod multi_layer_same_runtime_encode;
#[cfg(target_os = "macos")]
pub use multi_layer_same_runtime_encode::{
    Qwen80MultiLayerSuffixWitness, Qwen80SameRuntimeDeltaNetLayerPrefixEncoder,
    Qwen80SameRuntimeMultiLayerChainParity,
};

#[cfg(target_os = "macos")]
fn qwen80_require_exact_structural_kernel_trace(
    observed: Option<&[String]>,
    expected: &[&str],
    label: &str,
) -> Result<()> {
    let observed = observed.ok_or_else(|| {
        model_error(format!(
            "{label} requires a structural kernel trace on a fresh non-timed command buffer"
        ))
    })?;
    if observed.len() != expected.len()
        || observed
            .iter()
            .map(String::as_str)
            .zip(expected.iter().copied())
            .any(|(actual, expected)| actual != expected)
    {
        return Err(model_error(format!(
            "{label} structural kernel trace differs from the exact required command graph"
        )));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn qwen80_require_same_live_pinned_allocation(
    left: &PinnedBuffer,
    right: &PinnedBuffer,
    label: &str,
) -> Result<()> {
    if left.length() == 0
        || right.length() == 0
        || left.length() != right.length()
        || left.contents() != right.contents()
    {
        return Err(model_error(format!(
            "{label} does not retain one identical live Metal allocation"
        )));
    }
    Ok(())
}

fn model_error(message: impl Into<String>) -> Error {
    Error::Model(format!(
        "qwen80 complete native runtime: {}",
        message.into()
    ))
}

fn usize_field(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| model_error(format!("config missing unsigned {field:?}")))
}

fn finite_f32_field(value: &Value, field: &str) -> Result<f32> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .map(|number| number as f32)
        .ok_or_else(|| model_error(format!("config missing finite numeric {field:?}")))
}

fn string_field<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| model_error(format!("config missing non-empty string {field:?}")))
}

fn required_bool(value: &Value, field: &str, expected: bool) -> Result<()> {
    if value.get(field).and_then(Value::as_bool) != Some(expected) {
        return Err(model_error(format!(
            "config {field:?} must be {expected:?} for the admitted Qwen80 runtime"
        )));
    }
    Ok(())
}

fn checked_product(values: &[usize], label: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| model_error(format!("{label} overflows usize")))
    })
}

fn bytes_for_f32(elements: usize, label: &str) -> Result<usize> {
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| model_error(format!("{label} byte count overflows usize")))
}

/// Exact source-derived mixer selection for one Qwen3-Next decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum Qwen80LayerKind {
    LinearAttention,
    FullAttention,
}

impl Qwen80LayerKind {
    pub const fn as_source_name(self) -> &'static str {
        match self {
            Self::LinearAttention => "linear_attention",
            Self::FullAttention => "full_attention",
        }
    }
}

/// Exact source configuration required by the Qwen80 direct artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen80CompleteRuntimeConfig {
    pub model_id: String,
    pub source_repository: String,
    pub source_revision: String,
    pub layers: usize,
    pub hidden: usize,
    pub attention_heads: usize,
    pub key_value_heads: usize,
    pub attention_head_dim: usize,
    pub linear_key_heads: usize,
    pub linear_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub experts: usize,
    pub experts_per_token: usize,
    pub moe_intermediate: usize,
    pub shared_expert_intermediate: usize,
    pub vocab_size: usize,
    pub source_max_position_embeddings: usize,
    pub rope_theta_bits: u32,
    pub partial_rotary_factor_bits: u32,
    pub rms_norm_eps_bits: u32,
}

impl Qwen80CompleteRuntimeConfig {
    /// Parse an exact Qwen3-Coder-Next source config.  Nearby Qwen configs are
    /// refused before a direct artifact can be addressed.
    pub fn from_source_config(
        document: &Value,
        source_repository: &str,
        source_revision: &str,
    ) -> Result<Self> {
        let architectures = document
            .get("architectures")
            .and_then(Value::as_array)
            .ok_or_else(|| model_error("config missing architectures array"))?;
        if !architectures
            .iter()
            .any(|value| value.as_str() == Some(QWEN80_ARCHITECTURE))
        {
            return Err(model_error(format!(
                "config architectures does not contain {QWEN80_ARCHITECTURE}"
            )));
        }
        if string_field(document, "model_type")? != QWEN80_MODEL_TYPE {
            return Err(model_error("config model_type is not qwen3_next"));
        }
        if source_repository != QWEN80_REPOSITORY {
            return Err(model_error(format!(
                "artifact repository {source_repository:?} is not {QWEN80_REPOSITORY:?}"
            )));
        }
        if source_revision.is_empty() {
            return Err(model_error("artifact source revision is empty"));
        }
        for (field, expected) in [
            ("num_hidden_layers", QWEN80_LAYERS),
            ("hidden_size", QWEN80_HIDDEN),
            ("num_attention_heads", QWEN80_FULL_ATTN_HEADS),
            ("num_key_value_heads", QWEN80_FULL_ATTN_KV_HEADS),
            ("head_dim", QWEN80_FULL_ATTN_HEAD_DIM),
            ("linear_num_key_heads", QWEN80_LINEAR_KEY_HEADS),
            ("linear_num_value_heads", QWEN80_LINEAR_VALUE_HEADS),
            ("linear_key_head_dim", QWEN80_LINEAR_KEY_HEAD_DIM),
            ("linear_value_head_dim", QWEN80_LINEAR_VALUE_HEAD_DIM),
            ("linear_conv_kernel_dim", QWEN80_LINEAR_CONV_KERNEL),
            ("num_experts", QWEN80_EXPERTS),
            ("num_experts_per_tok", QWEN80_TOP_K),
            ("moe_intermediate_size", QWEN80_MOE_INTERMEDIATE),
            (
                "shared_expert_intermediate_size",
                QWEN80_SHARED_EXPERT_INTERMEDIATE,
            ),
            ("vocab_size", QWEN80_VOCAB),
            ("decoder_sparse_step", QWEN80_DECODER_SPARSE_STEP),
            ("full_attention_interval", QWEN80_FULL_ATTENTION_INTERVAL),
            ("intermediate_size", 5120),
        ] {
            let observed = usize_field(document, field)?;
            if observed != expected {
                return Err(model_error(format!(
                    "config {field}={observed}, expected exact Qwen80 value {expected}"
                )));
            }
        }
        if string_field(document, "hidden_act")? != "silu" {
            return Err(model_error(
                "Qwen80 runtime requires source SiLU activation",
            ));
        }
        required_bool(document, "norm_topk_prob", true)?;
        required_bool(document, "tie_word_embeddings", false)?;
        required_bool(document, "attention_bias", false)?;
        required_bool(document, "use_sliding_window", false)?;
        required_bool(document, "use_cache", true)?;
        if document.get("attention_dropout").and_then(Value::as_f64) != Some(0.0) {
            return Err(model_error("Qwen80 runtime requires attention_dropout=0"));
        }
        if document.get("rope_scaling") != Some(&Value::Null) {
            return Err(model_error(
                "Qwen80 runtime only admits the source unscaled RoPE configuration",
            ));
        }
        let mlp_only_layers = document
            .get("mlp_only_layers")
            .and_then(Value::as_array)
            .ok_or_else(|| model_error("config missing mlp_only_layers"))?;
        if !mlp_only_layers.is_empty() {
            return Err(model_error(
                "Qwen80 source must use sparse MoE in every decoder layer",
            ));
        }
        if let Some(layer_types) = document.get("layer_types") {
            if !layer_types.is_null() {
                let layer_types = layer_types
                    .as_array()
                    .ok_or_else(|| model_error("config layer_types must be null or an array"))?;
                if layer_types.len() != QWEN80_LAYERS {
                    return Err(model_error("config layer_types length is not 48"));
                }
                for (layer, layer_type) in layer_types.iter().enumerate() {
                    if layer_type.as_str() != Some(qwen80_layer_kind(layer)?.as_source_name()) {
                        return Err(model_error(format!(
                            "config layer_types[{layer}] disagrees with the exact 3-linear/1-full hybrid schedule"
                        )));
                    }
                }
            }
        }
        let rope_theta = finite_f32_field(document, "rope_theta")?;
        let partial_rotary_factor = finite_f32_field(document, "partial_rotary_factor")?;
        let rms_norm_eps = finite_f32_field(document, "rms_norm_eps")?;
        if rope_theta.to_bits() != QWEN80_ROPE_THETA.to_bits()
            || partial_rotary_factor.to_bits() != QWEN80_PARTIAL_ROTARY_FACTOR.to_bits()
            || rms_norm_eps.to_bits() != QWEN80_RMS_EPS.to_bits()
        {
            return Err(model_error(
                "config RoPE/RMS values differ from the exact Qwen80 source geometry",
            ));
        }
        let source_max_position_embeddings = usize_field(document, "max_position_embeddings")?;
        if source_max_position_embeddings == 0 {
            return Err(model_error(
                "config max_position_embeddings must be non-zero",
            ));
        }
        Ok(Self {
            model_id: QWEN80_MODEL_ID.to_owned(),
            source_repository: source_repository.to_owned(),
            source_revision: source_revision.to_owned(),
            layers: QWEN80_LAYERS,
            hidden: QWEN80_HIDDEN,
            attention_heads: QWEN80_FULL_ATTN_HEADS,
            key_value_heads: QWEN80_FULL_ATTN_KV_HEADS,
            attention_head_dim: QWEN80_FULL_ATTN_HEAD_DIM,
            linear_key_heads: QWEN80_LINEAR_KEY_HEADS,
            linear_value_heads: QWEN80_LINEAR_VALUE_HEADS,
            linear_key_head_dim: QWEN80_LINEAR_KEY_HEAD_DIM,
            linear_value_head_dim: QWEN80_LINEAR_VALUE_HEAD_DIM,
            linear_conv_kernel_dim: QWEN80_LINEAR_CONV_KERNEL,
            experts: QWEN80_EXPERTS,
            experts_per_token: QWEN80_TOP_K,
            moe_intermediate: QWEN80_MOE_INTERMEDIATE,
            shared_expert_intermediate: QWEN80_SHARED_EXPERT_INTERMEDIATE,
            vocab_size: QWEN80_VOCAB,
            source_max_position_embeddings,
            rope_theta_bits: rope_theta.to_bits(),
            partial_rotary_factor_bits: partial_rotary_factor.to_bits(),
            rms_norm_eps_bits: rms_norm_eps.to_bits(),
        })
    }

    pub fn layer_kind(&self, layer: usize) -> Result<Qwen80LayerKind> {
        qwen80_layer_kind(layer)
    }

    pub fn rope_theta(&self) -> f32 {
        f32::from_bits(self.rope_theta_bits)
    }

    pub fn partial_rotary_factor(&self) -> f32 {
        f32::from_bits(self.partial_rotary_factor_bits)
    }

    pub fn rms_norm_eps(&self) -> f32 {
        f32::from_bits(self.rms_norm_eps_bits)
    }

    pub fn attention_query_dim(&self) -> usize {
        self.attention_heads * self.attention_head_dim
    }

    pub fn attention_kv_dim(&self) -> usize {
        self.key_value_heads * self.attention_head_dim
    }

    pub fn linear_key_dim(&self) -> usize {
        self.linear_key_heads * self.linear_key_head_dim
    }

    pub fn linear_value_dim(&self) -> usize {
        self.linear_value_heads * self.linear_value_head_dim
    }

    pub fn linear_conv_dim(&self) -> usize {
        self.linear_key_dim() * 2 + self.linear_value_dim()
    }
}

/// Return the immutable 3x DeltaNet then 1x full-attention source schedule.
pub fn qwen80_layer_kind(layer: usize) -> Result<Qwen80LayerKind> {
    if layer >= QWEN80_LAYERS {
        return Err(Error::Model(
            "qwen80 complete native runtime: layer index is outside 0..48".into(),
        ));
    }
    if (layer + 1) % QWEN80_FULL_ATTENTION_INTERVAL == 0 {
        Ok(Qwen80LayerKind::FullAttention)
    } else {
        Ok(Qwen80LayerKind::LinearAttention)
    }
}

fn tensor_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut expected = BTreeMap::new();
    expected.insert(
        "model.embed_tokens.weight".into(),
        vec![QWEN80_VOCAB, QWEN80_HIDDEN],
    );
    for layer in 0..QWEN80_LAYERS {
        let prefix = format!("model.layers.{layer}");
        expected.insert(
            format!("{prefix}.input_layernorm.weight"),
            vec![QWEN80_HIDDEN],
        );
        expected.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![QWEN80_HIDDEN],
        );
        match qwen80_layer_kind(layer).expect("loop bounds are exact") {
            Qwen80LayerKind::LinearAttention => {
                let key_dim = QWEN80_LINEAR_KEY_HEADS * QWEN80_LINEAR_KEY_HEAD_DIM;
                let value_dim = QWEN80_LINEAR_VALUE_HEADS * QWEN80_LINEAR_VALUE_HEAD_DIM;
                let conv_dim = key_dim * 2 + value_dim;
                expected.insert(
                    format!("{prefix}.linear_attn.A_log"),
                    vec![QWEN80_LINEAR_VALUE_HEADS],
                );
                expected.insert(
                    format!("{prefix}.linear_attn.conv1d.weight"),
                    vec![conv_dim, 1, QWEN80_LINEAR_CONV_KERNEL],
                );
                expected.insert(
                    format!("{prefix}.linear_attn.dt_bias"),
                    vec![QWEN80_LINEAR_VALUE_HEADS],
                );
                expected.insert(
                    format!("{prefix}.linear_attn.in_proj_ba.weight"),
                    vec![QWEN80_LINEAR_VALUE_HEADS * 2, QWEN80_HIDDEN],
                );
                expected.insert(
                    format!("{prefix}.linear_attn.in_proj_qkvz.weight"),
                    vec![key_dim * 2 + value_dim * 2, QWEN80_HIDDEN],
                );
                expected.insert(
                    format!("{prefix}.linear_attn.norm.weight"),
                    vec![QWEN80_LINEAR_VALUE_HEAD_DIM],
                );
                expected.insert(
                    format!("{prefix}.linear_attn.out_proj.weight"),
                    vec![QWEN80_HIDDEN, value_dim],
                );
            }
            Qwen80LayerKind::FullAttention => {
                let query_dim = QWEN80_FULL_ATTN_HEADS * QWEN80_FULL_ATTN_HEAD_DIM;
                let kv_dim = QWEN80_FULL_ATTN_KV_HEADS * QWEN80_FULL_ATTN_HEAD_DIM;
                expected.insert(
                    format!("{prefix}.self_attn.q_norm.weight"),
                    vec![QWEN80_FULL_ATTN_HEAD_DIM],
                );
                expected.insert(
                    format!("{prefix}.self_attn.k_norm.weight"),
                    vec![QWEN80_FULL_ATTN_HEAD_DIM],
                );
                expected.insert(
                    format!("{prefix}.self_attn.q_proj.weight"),
                    vec![query_dim * 2, QWEN80_HIDDEN],
                );
                expected.insert(
                    format!("{prefix}.self_attn.k_proj.weight"),
                    vec![kv_dim, QWEN80_HIDDEN],
                );
                expected.insert(
                    format!("{prefix}.self_attn.v_proj.weight"),
                    vec![kv_dim, QWEN80_HIDDEN],
                );
                expected.insert(
                    format!("{prefix}.self_attn.o_proj.weight"),
                    vec![QWEN80_HIDDEN, query_dim],
                );
            }
        }
        expected.insert(
            format!("{prefix}.mlp.gate.weight"),
            vec![QWEN80_EXPERTS, QWEN80_HIDDEN],
        );
        for expert in 0..QWEN80_EXPERTS {
            let expert_prefix = format!("{prefix}.mlp.experts.{expert}");
            expected.insert(
                format!("{expert_prefix}.gate_proj.weight"),
                vec![QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
            );
            expected.insert(
                format!("{expert_prefix}.up_proj.weight"),
                vec![QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
            );
            expected.insert(
                format!("{expert_prefix}.down_proj.weight"),
                vec![QWEN80_HIDDEN, QWEN80_MOE_INTERMEDIATE],
            );
        }
        let shared = format!("{prefix}.mlp.shared_expert");
        expected.insert(
            format!("{shared}.gate_proj.weight"),
            vec![QWEN80_SHARED_EXPERT_INTERMEDIATE, QWEN80_HIDDEN],
        );
        expected.insert(
            format!("{shared}.up_proj.weight"),
            vec![QWEN80_SHARED_EXPERT_INTERMEDIATE, QWEN80_HIDDEN],
        );
        expected.insert(
            format!("{shared}.down_proj.weight"),
            vec![QWEN80_HIDDEN, QWEN80_SHARED_EXPERT_INTERMEDIATE],
        );
        expected.insert(
            format!("{prefix}.mlp.shared_expert_gate.weight"),
            vec![1, QWEN80_HIDDEN],
        );
    }
    expected.insert("model.norm.weight".into(), vec![QWEN80_HIDDEN]);
    expected.insert("lm_head.weight".into(), vec![QWEN80_VOCAB, QWEN80_HIDDEN]);
    expected
}

fn validate_complete_catalog(
    artifact: &CompleteBinaryArtifact,
    config: &Qwen80CompleteRuntimeConfig,
) -> Result<()> {
    if artifact.model != QwenCompleteBinaryModel::Qwen80CoderNext {
        return Err(model_error(
            "complete artifact is not the Qwen80 model family",
        ));
    }
    if artifact.source_revision != config.source_revision
        || artifact.source_revision.is_empty()
        || config.source_repository != QWEN80_REPOSITORY
    {
        return Err(model_error(
            "artifact and source configuration revision binding disagrees",
        ));
    }
    let expected = tensor_shapes();
    if artifact.tensors.len() != expected.len() {
        return Err(model_error(format!(
            "admitted artifact tensor count {} does not equal required Qwen80 count {}",
            artifact.tensors.len(),
            expected.len()
        )));
    }
    let actual: HashSet<&str> = artifact.tensors.keys().map(String::as_str).collect();
    let required: HashSet<&str> = expected.keys().map(String::as_str).collect();
    if actual != required {
        let missing = expected
            .keys()
            .find(|name| !actual.contains(name.as_str()))
            .cloned();
        let unexpected = artifact
            .tensors
            .keys()
            .find(|name| !required.contains(name.as_str()))
            .cloned();
        return Err(model_error(format!(
            "admitted Qwen80 catalog tensor set mismatch; missing={missing:?} unexpected={unexpected:?}"
        )));
    }
    for (name, shape) in expected {
        let tensor = artifact.tensors.get(&name).ok_or_else(|| {
            model_error(format!("required tensor {name:?} vanished after set check"))
        })?;
        if tensor.header.shape != shape || tensor.header.group_size != QWEN80_GROUP_SIZE {
            return Err(model_error(format!(
                "tensor {name:?} has shape {:?}/group {} but requires {:?}/{}",
                tensor.header.shape, tensor.header.group_size, shape, QWEN80_GROUP_SIZE
            )));
        }
    }
    Ok(())
}

fn regular_bytes(path: &Path, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| model_error(format!("cannot stat {label} {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(model_error(format!(
            "{label} must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    let raw = fs::read(path)
        .map_err(|error| model_error(format!("cannot read {label} {}: {error}", path.display())))?;
    if raw.len() as u64 != metadata.len() {
        return Err(model_error(format!("{label} changed while being read")));
    }
    Ok(raw)
}

fn source_sidecar_path(artifact: &CompleteBinaryArtifact, filename: &str) -> Result<PathBuf> {
    let root = artifact
        .source_index_path
        .parent()
        .ok_or_else(|| model_error("admitted source index has no parent directory"))?;
    let source_root = fs::canonicalize(root).map_err(|error| {
        model_error(format!(
            "cannot canonicalize admitted source directory {}: {error}",
            root.display()
        ))
    })?;
    let candidate = fs::canonicalize(source_root.join(filename))
        .map_err(|error| model_error(format!("cannot canonicalize source {filename}: {error}")))?;
    if candidate.parent() != Some(source_root.as_path()) {
        return Err(model_error(format!(
            "source {filename} is not a direct admitted-source child"
        )));
    }
    Ok(candidate)
}

fn parse_source_config(artifact: &CompleteBinaryArtifact) -> Result<(PathBuf, String, Value)> {
    let path = source_sidecar_path(artifact, "config.json")?;
    let raw = regular_bytes(&path, "source config")?;
    let document: Value = serde_json::from_slice(&raw)
        .map_err(|error| model_error(format!("source config is invalid JSON: {error}")))?;
    Ok((path, format!("{:x}", Sha256::digest(&raw)), document))
}

fn tokenizer_from_source(
    artifact: &CompleteBinaryArtifact,
) -> Result<(PathBuf, String, Tokenizer)> {
    let path = source_sidecar_path(artifact, "tokenizer.json")?;
    let raw = regular_bytes(&path, "source tokenizer")?;
    let tokenizer = Tokenizer::from_file(&path)?;
    if tokenizer.vocab_size() != QWEN80_TOKENIZER_VOCAB {
        return Err(model_error(format!(
            "source tokenizer vocab {} does not equal exact Qwen80 tokenizer vocabulary {QWEN80_TOKENIZER_VOCAB}",
            tokenizer.vocab_size()
        )));
    }
    Ok((path, format!("{:x}", Sha256::digest(&raw)), tokenizer))
}

/// Exact device-state accounting for one native Qwen80 session.
///
/// The recurrent and causal-convolution state is only allocated for the 36
/// linear-attention layers.  The K/V cache is only allocated for the 12 full
/// attention layers.  Keeping those domains separate prevents a future
/// scheduler from silently treating Qwen3-Next as a conventional all-GQA
/// model.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen80NativeStateGeometry {
    pub max_seq_len: usize,
    pub linear_layers: usize,
    pub full_attention_layers: usize,
    pub linear_conv_state_elements: usize,
    pub linear_recurrent_state_elements: usize,
    pub full_attention_key_cache_elements: usize,
    pub full_attention_value_cache_elements: usize,
}

impl Qwen80NativeStateGeometry {
    pub fn from_config(config: &Qwen80CompleteRuntimeConfig, max_seq_len: usize) -> Result<Self> {
        if max_seq_len == 0 || max_seq_len > QWEN80_COMPLETE_NATIVE_MAX_CONTEXT {
            return Err(model_error(format!(
                "requested max_seq_len={max_seq_len} is outside native hybrid support 1..={QWEN80_COMPLETE_NATIVE_MAX_CONTEXT}"
            )));
        }
        if max_seq_len > config.source_max_position_embeddings {
            return Err(model_error(format!(
                "requested max_seq_len={max_seq_len} exceeds source maximum {}",
                config.source_max_position_embeddings
            )));
        }
        let linear_layers = (0..config.layers)
            .filter(|layer| {
                matches!(
                    config.layer_kind(*layer),
                    Ok(Qwen80LayerKind::LinearAttention)
                )
            })
            .count();
        let full_attention_layers = config.layers - linear_layers;
        let linear_conv_state_elements = checked_product(
            &[
                linear_layers,
                config.linear_conv_dim(),
                config.linear_conv_kernel_dim - 1,
            ],
            "linear convolution state",
        )?;
        let linear_recurrent_state_elements = checked_product(
            &[
                linear_layers,
                config.linear_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            ],
            "linear recurrent state",
        )?;
        let full_attention_cache_elements = checked_product(
            &[
                full_attention_layers,
                max_seq_len,
                config.key_value_heads,
                config.attention_head_dim,
            ],
            "full attention cache",
        )?;
        Ok(Self {
            max_seq_len,
            linear_layers,
            full_attention_layers,
            linear_conv_state_elements,
            linear_recurrent_state_elements,
            full_attention_key_cache_elements: full_attention_cache_elements,
            full_attention_value_cache_elements: full_attention_cache_elements,
        })
    }

    pub fn total_elements(&self) -> Result<usize> {
        [
            self.linear_conv_state_elements,
            self.linear_recurrent_state_elements,
            self.full_attention_key_cache_elements,
            self.full_attention_value_cache_elements,
        ]
        .iter()
        .try_fold(0usize, |sum, value| {
            sum.checked_add(*value)
                .ok_or_else(|| model_error("native Qwen80 state element count overflows usize"))
        })
    }

    pub fn total_f32_bytes(&self) -> Result<usize> {
        bytes_for_f32(self.total_elements()?, "native Qwen80 state")
    }
}

/// A strict direct-binary catalog.  It remains bound to the admitted artifact,
/// so any future decoder can ask it for a packed tensor without reopening BF16
/// source shards or inventing a second tensor name mapping.
pub struct Qwen80CompleteArtifactCatalog {
    artifact: CompleteBinaryArtifact,
    pub config: Qwen80CompleteRuntimeConfig,
}

impl Qwen80CompleteArtifactCatalog {
    /// Perform the one full strict admission scan required before a Qwen80
    /// process is allowed to touch a direct packed tensor.
    ///
    /// Callers that need both a structural preflight and a native state/stage
    /// must retain the returned catalog and hand it forward.  Re-admitting the
    /// same sealed 74,391-payload artifact inside one process adds latency but
    /// no integrity: `from_admitted` below deliberately consumes this exact
    /// catalog result without reopening a payload.
    pub fn load(
        manifest_path: impl AsRef<Path>,
        admission: &CompleteBinaryAdmission,
    ) -> Result<Self> {
        let artifact = admit_complete_binary_artifact(manifest_path, admission)?;
        Self::from_admitted(artifact)
    }

    /// Finish Qwen3-Next-specific catalog validation from an artifact that
    /// has already passed the full source, payload, header, and ledger scan.
    ///
    /// This is intentionally private: external callers cannot construct a
    /// bypass around `load`, while this module can transfer one strict
    /// admission result into native state/stage construction in-process.
    fn from_admitted(artifact: CompleteBinaryArtifact) -> Result<Self> {
        let (_config_path, _config_sha256, source_config) = parse_source_config(&artifact)?;
        let config = Qwen80CompleteRuntimeConfig::from_source_config(
            &source_config,
            QWEN80_REPOSITORY,
            &artifact.source_revision,
        )?;
        validate_complete_catalog(&artifact, &config)?;
        Ok(Self { artifact, config })
    }

    pub fn manifest_path(&self) -> &Path {
        &self.artifact.manifest_path
    }

    pub fn manifest_seal(&self) -> &str {
        &self.artifact.manifest_seal_sha256
    }

    pub fn tensor_count(&self) -> usize {
        self.artifact.tensors.len()
    }

    pub fn tensor_payload_bytes(&self) -> u64 {
        self.artifact.tensor_payload_bytes
    }

    pub fn source_weight_elements(&self) -> u64 {
        self.artifact.source_weight_elements
    }

    pub fn direct_tensor_header(&self, name: &str) -> Result<&CompleteBinaryHeader> {
        self.artifact
            .tensors
            .get(name)
            .map(|tensor| &tensor.header)
            .ok_or_else(|| model_error(format!("unknown exact Qwen80 tensor {name:?}")))
    }

    /// Return the manifest-sealed digest for one exact compact tensor.  This
    /// exposes identity only; callers that execute it must still request the
    /// immutable in-process snapshot through [`Self::verified_direct_tensor_payload`].
    pub fn direct_tensor_artifact_sha256(&self, name: &str) -> Result<&str> {
        self.artifact
            .tensors
            .get(name)
            .map(|tensor| tensor.artifact_sha256.as_str())
            .ok_or_else(|| model_error(format!("unknown exact Qwen80 tensor {name:?}")))
    }

    /// Return one immutable direct complete-binary payload retained by the
    /// complete-artifact admission scan.  Production/native paths must use
    /// this snapshot rather than reopening an artifact path after admission.
    pub fn verified_direct_tensor_payload(&self, name: &str) -> Result<Arc<[u8]>> {
        let expected = self.direct_tensor_header(name)?;
        let payload = self.artifact.verified_tensor_payload(name)?;
        let observed = parse_complete_binary_header(&payload)?;
        if &observed != expected {
            return Err(model_error(format!(
                "admission-verified direct payload header for {name:?} differs from sealed catalog"
            )));
        }
        if observed.group_size != QWEN80_GROUP_SIZE {
            return Err(model_error(format!(
                "direct payload for {name:?} has unsupported group size {}",
                observed.group_size
            )));
        }
        Ok(payload)
    }

    /// Compatibility helper for older bounded component code.  It returns a
    /// copy of the immutable admission snapshot rather than re-reading a
    /// file.  New direct-packed paths should prefer
    /// [`Self::verified_direct_tensor_payload`] so payload authority and
    /// residency remain explicit.
    pub fn read_direct_tensor_payload(&self, name: &str) -> Result<Vec<u8>> {
        Ok(self.verified_direct_tensor_payload(name)?.to_vec())
    }
}

/// Structural native-runtime proof.  This deliberately has no generation,
/// capability, HCLI, TPS, or TG meaning.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen80CompleteRuntimePreflight {
    pub manifest_path: PathBuf,
    pub manifest_seal_sha256: String,
    pub source_revision: String,
    pub config_path: PathBuf,
    pub config_sha256: String,
    pub tokenizer_path: PathBuf,
    pub tokenizer_sha256: String,
    pub tokenizer_vocab_size: usize,
    pub reserved_lm_head_tail_rows: usize,
    pub tensor_count: usize,
    pub tensor_payload_bytes: u64,
    pub source_weight_elements: u64,
    pub direct_layout_group_size: usize,
    pub default_native_state: Qwen80NativeStateGeometry,
}

/// Admit an exact artifact, verify every hybrid tensor name/shape, bind the
/// tokenizer/config, and derive the state layout.  This intentionally does
/// not allocate a full model body or execute a token.
pub fn preflight_complete_runtime(
    manifest_path: impl AsRef<Path>,
    admission: &CompleteBinaryAdmission,
) -> Result<Qwen80CompleteRuntimePreflight> {
    let catalog = Qwen80CompleteArtifactCatalog::load(manifest_path, admission)?;
    preflight_from_admitted_catalog(&catalog)
}

/// Derive the complete-runtime structural receipt from one already-admitted
/// catalog.  It checks source sidecars/config/tokenizer but deliberately does
/// not reopen or rehash the complete packed payload set; `load` performed that
/// one full strict scan immediately before this handoff.
pub fn preflight_from_admitted_catalog(
    catalog: &Qwen80CompleteArtifactCatalog,
) -> Result<Qwen80CompleteRuntimePreflight> {
    let (config_path, config_sha256, _source_config) = parse_source_config(&catalog.artifact)?;
    let (tokenizer_path, tokenizer_sha256, tokenizer) = tokenizer_from_source(&catalog.artifact)?;
    let default_native_state = Qwen80NativeStateGeometry::from_config(
        &catalog.config,
        QWEN80_COMPLETE_NATIVE_MAX_CONTEXT,
    )?;
    Ok(Qwen80CompleteRuntimePreflight {
        manifest_path: catalog.artifact.manifest_path.clone(),
        manifest_seal_sha256: catalog.artifact.manifest_seal_sha256.clone(),
        source_revision: catalog.artifact.source_revision.clone(),
        config_path,
        config_sha256,
        tokenizer_path,
        tokenizer_sha256,
        tokenizer_vocab_size: tokenizer.vocab_size(),
        reserved_lm_head_tail_rows: QWEN80_VOCAB - tokenizer.vocab_size(),
        tensor_count: catalog.artifact.tensors.len(),
        tensor_payload_bytes: catalog.artifact.tensor_payload_bytes,
        source_weight_elements: catalog.artifact.source_weight_elements,
        direct_layout_group_size: QWEN80_GROUP_SIZE,
        default_native_state,
    })
}

/// Runtime choices that do not alter Qwen80 source geometry or artifact
/// binding.  The context cap is a current implementation cap, not a source
/// long-context qualification.
#[derive(Clone, Debug)]
pub struct Qwen80CompleteRuntimeOptions {
    pub max_seq_len: usize,
    pub trace_dispatch: bool,
}

impl Default for Qwen80CompleteRuntimeOptions {
    fn default() -> Self {
        Self {
            max_seq_len: 256,
            trace_dispatch: false,
        }
    }
}

/// A direct packed tensor as addressed by the complete Qwen3-Next artifact.
///
/// This is deliberately a catalog address, not a decoded weight buffer.  A
/// backend which consumes this binding must use the admitted compact payload;
/// it may not substitute a BF16/MPS shadow tensor.  Keeping the expected
/// header geometry on every address makes an accidental Qwen2/Qwen3 tensor
/// mapping fail before execution begins.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen80PackedTensorBinding {
    pub name: String,
    pub shape: Vec<usize>,
    pub group_size: usize,
}

/// The exact direct-packed tensors used by one Gated DeltaNet token mixer.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen80LinearDeltaNetLayerBindings {
    pub in_proj_qkvz: Qwen80PackedTensorBinding,
    pub in_proj_ba: Qwen80PackedTensorBinding,
    pub causal_conv1d: Qwen80PackedTensorBinding,
    pub a_log: Qwen80PackedTensorBinding,
    pub dt_bias: Qwen80PackedTensorBinding,
    pub gated_rms_norm: Qwen80PackedTensorBinding,
    pub out_proj: Qwen80PackedTensorBinding,
}

/// The exact direct-packed tensors used by one gated GQA token mixer.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen80FullAttentionLayerBindings {
    pub q_proj: Qwen80PackedTensorBinding,
    pub k_proj: Qwen80PackedTensorBinding,
    pub v_proj: Qwen80PackedTensorBinding,
    pub q_norm: Qwen80PackedTensorBinding,
    pub k_norm: Qwen80PackedTensorBinding,
    pub o_proj: Qwen80PackedTensorBinding,
}

/// The mixer body selected by the immutable Qwen3-Next 3:1 schedule.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub enum Qwen80HybridMixerBindings {
    LinearDeltaNet(Qwen80LinearDeltaNetLayerBindings),
    FullAttention(Qwen80FullAttentionLayerBindings),
}

/// Tensor address template for one source expert.  The concrete tensor names
/// are derived only after a device router returns the source top-10 ids.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen80ExpertBindings {
    pub expert: usize,
    pub gate_proj: Qwen80PackedTensorBinding,
    pub up_proj: Qwen80PackedTensorBinding,
    pub down_proj: Qwen80PackedTensorBinding,
}

/// Exact routed + shared MoE bindings for one decoder layer.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen80MoeLayerBindings {
    pub router: Qwen80PackedTensorBinding,
    pub shared_gate_proj: Qwen80PackedTensorBinding,
    pub shared_up_proj: Qwen80PackedTensorBinding,
    pub shared_down_proj: Qwen80PackedTensorBinding,
    pub shared_expert_gate: Qwen80PackedTensorBinding,
}

/// One fully-addressed Qwen3-Next decoder layer.  Linear and full-attention
/// state slots are separate by construction: an implementation cannot
/// silently apply a conventional KV cache to DeltaNet layers or vice versa.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen80HybridDecoderLayerPlan {
    pub layer: usize,
    pub kind: Qwen80LayerKind,
    pub linear_state_slot: Option<usize>,
    pub full_attention_state_slot: Option<usize>,
    pub input_layernorm: Qwen80PackedTensorBinding,
    pub mixer: Qwen80HybridMixerBindings,
    pub post_attention_layernorm: Qwen80PackedTensorBinding,
    pub moe: Qwen80MoeLayerBindings,
}

/// Explicit native operator groups still required to turn the complete graph
/// scheduler into a production Qwen80 token backend.  These are not a list of
/// optional optimizations: all entries are mandatory before a generated token
/// or TPS value can be claimed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum Qwen80HybridNativeOperatorGap {
    DirectPackedEmbeddingGather,
    DirectPackedRmsNorm,
    LinearDeltaNetConvRearrangeAndGatedNorm,
    LinearDeltaNetOutProjectionAndResidual,
    FullAttentionGqaKvRopeGateAndResidual,
    RoutedTopTenGateUpDownAndWeightedCombine,
    SharedExpertGateUpDownAndCombine,
    DirectPackedFinalNormLmHeadReservedTailMaskAndSampler,
    DeviceResidentAutoregressiveStateAndFeedback,
}

impl Qwen80HybridNativeOperatorGap {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DirectPackedEmbeddingGather => "direct_packed_embedding_gather",
            Self::DirectPackedRmsNorm => "direct_packed_rms_norm",
            Self::LinearDeltaNetConvRearrangeAndGatedNorm => {
                "linear_deltanet_conv_rearrange_and_gated_norm"
            }
            Self::LinearDeltaNetOutProjectionAndResidual => {
                "linear_deltanet_out_projection_and_residual"
            }
            Self::FullAttentionGqaKvRopeGateAndResidual => {
                "full_attention_gqa_kv_rope_gate_and_residual"
            }
            Self::RoutedTopTenGateUpDownAndWeightedCombine => {
                "routed_top10_gate_up_down_and_weighted_combine"
            }
            Self::SharedExpertGateUpDownAndCombine => "shared_expert_gate_up_down_and_combine",
            Self::DirectPackedFinalNormLmHeadReservedTailMaskAndSampler => {
                "direct_packed_final_norm_lm_head_reserved_tail_mask_and_sampler"
            }
            Self::DeviceResidentAutoregressiveStateAndFeedback => {
                "device_resident_autoregressive_state_and_feedback"
            }
        }
    }
}

const QWEN80_HYBRID_NATIVE_OPERATOR_GAPS: [Qwen80HybridNativeOperatorGap; 9] = [
    Qwen80HybridNativeOperatorGap::DirectPackedEmbeddingGather,
    Qwen80HybridNativeOperatorGap::DirectPackedRmsNorm,
    Qwen80HybridNativeOperatorGap::LinearDeltaNetConvRearrangeAndGatedNorm,
    Qwen80HybridNativeOperatorGap::LinearDeltaNetOutProjectionAndResidual,
    Qwen80HybridNativeOperatorGap::FullAttentionGqaKvRopeGateAndResidual,
    Qwen80HybridNativeOperatorGap::RoutedTopTenGateUpDownAndWeightedCombine,
    Qwen80HybridNativeOperatorGap::SharedExpertGateUpDownAndCombine,
    Qwen80HybridNativeOperatorGap::DirectPackedFinalNormLmHeadReservedTailMaskAndSampler,
    Qwen80HybridNativeOperatorGap::DeviceResidentAutoregressiveStateAndFeedback,
];

/// A catalog-bound complete Qwen3-Next token graph.  Constructing this plan
/// proves that every official direct-packed tensor is addressable in the
/// correct hybrid graph; it does not execute any payload or make a runtime
/// capability claim.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen80CompleteHybridDecoderPlan {
    pub manifest_seal_sha256: String,
    pub source_revision: String,
    pub max_seq_len: usize,
    pub state: Qwen80NativeStateGeometry,
    pub embedding: Qwen80PackedTensorBinding,
    pub layers: Vec<Qwen80HybridDecoderLayerPlan>,
    pub final_norm: Qwen80PackedTensorBinding,
    pub lm_head: Qwen80PackedTensorBinding,
    pub tokenizer_vocab_size: usize,
    pub reserved_lm_head_tail_rows: usize,
    pub required_native_operator_gaps: Vec<Qwen80HybridNativeOperatorGap>,
}

impl Qwen80CompleteHybridDecoderPlan {
    fn binding(
        catalog: &Qwen80CompleteArtifactCatalog,
        name: impl Into<String>,
        expected_shape: &[usize],
    ) -> Result<Qwen80PackedTensorBinding> {
        let name = name.into();
        let header = catalog.direct_tensor_header(&name)?;
        if header.shape != expected_shape || header.group_size != QWEN80_GROUP_SIZE {
            return Err(model_error(format!(
                "hybrid decoder binding {name:?} has shape {:?}/group {}, expected {:?}/{}",
                header.shape, header.group_size, expected_shape, QWEN80_GROUP_SIZE
            )));
        }
        Ok(Qwen80PackedTensorBinding {
            name,
            shape: header.shape.clone(),
            group_size: header.group_size,
        })
    }

    fn expert_bindings(
        &self,
        catalog: &Qwen80CompleteArtifactCatalog,
        layer: usize,
        expert: usize,
    ) -> Result<Qwen80ExpertBindings> {
        if layer >= self.layers.len() || expert >= QWEN80_EXPERTS {
            return Err(model_error(format!(
                "hybrid decoder expert address layer={layer}, expert={expert} is outside source bounds"
            )));
        }
        let prefix = format!("model.layers.{layer}.mlp.experts.{expert}");
        Ok(Qwen80ExpertBindings {
            expert,
            gate_proj: Self::binding(
                catalog,
                format!("{prefix}.gate_proj.weight"),
                &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
            )?,
            up_proj: Self::binding(
                catalog,
                format!("{prefix}.up_proj.weight"),
                &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
            )?,
            down_proj: Self::binding(
                catalog,
                format!("{prefix}.down_proj.weight"),
                &[QWEN80_HIDDEN, QWEN80_MOE_INTERMEDIATE],
            )?,
        })
    }

    /// Return exact direct-packed expert addresses only after the caller has
    /// validated a source top-10 route.  This keeps filename selection bound
    /// to router output instead of letting a scheduler invent an expert body.
    pub fn routed_expert_bindings(
        &self,
        catalog: &Qwen80CompleteArtifactCatalog,
        layer: usize,
        route: &Qwen80RouteSelection,
    ) -> Result<Vec<Qwen80ExpertBindings>> {
        if layer >= self.layers.len() {
            return Err(model_error(format!(
                "hybrid decoder route layer {layer} is outside 0..{}",
                self.layers.len()
            )));
        }
        route.validate()?;
        route
            .ids
            .iter()
            .map(|&expert| self.expert_bindings(catalog, layer, expert as usize))
            .collect()
    }

    /// The complete Qwen80 graph is intentionally marked incomplete until a
    /// direct-packed native backend implements every listed operation group.
    /// The scheduler itself never converts this list into a false pass.
    pub fn has_complete_native_operator_backend(&self) -> bool {
        self.required_native_operator_gaps.is_empty()
    }

    /// Parse and bind the descriptor-only all-ten route plan to this exact
    /// hybrid graph.  The returned plan is still non-executing; callers must
    /// use the direct-packed all-ten CPU/device wave executor and preserve
    /// its immutable document SHA in every witness.
    pub fn bind_all_ten_routed_expert_plan(
        &self,
        layer_index: usize,
        authority: &Qwen80AllTenRoutedExpertPlanAuthority<'_>,
        descriptor: &Value,
    ) -> Result<Qwen80AllTenRoutedExpertPlan> {
        let layer = self.layers.get(layer_index).ok_or_else(|| {
            model_error(format!(
                "all-ten routed-expert plan layer {layer_index} is outside hybrid plan"
            ))
        })?;
        require_all_ten_route_plan(self, layer, authority, descriptor)
    }

    /// Bind the separately sealed source-token route authority.  Unlike the
    /// historical plan parser above, this path does not accept a standalone
    /// postnorm/router fixture receipt: its route is tied to source token 1,
    /// zeroed L0 DeltaNet state, and the strict-Metal prefix output seal.
    pub fn bind_source_token_all_ten_routed_expert_plan(
        &self,
        layer_index: usize,
        authority: &Qwen80SourceTokenAllTenRoutedExpertPlanAuthority<'_>,
        descriptor: &Value,
    ) -> Result<Qwen80AllTenRoutedExpertPlan> {
        let layer = self.layers.get(layer_index).ok_or_else(|| {
            model_error(format!(
                "source-token all-ten routed-expert plan layer {layer_index} is outside hybrid plan"
            ))
        })?;
        require_source_token_all_ten_route_plan(self, layer, authority, descriptor)
    }
}

impl Qwen80CompleteArtifactCatalog {
    /// Build an all-48-layer source/artifact-bound hybrid decoder graph without
    /// opening source BF16 tensors, decoding a compact payload, allocating
    /// Metal state, or executing a token.  The catalog passed here was already
    /// admitted by the sole strict full-artifact scan.
    pub fn complete_hybrid_decoder_plan(
        &self,
        max_seq_len: usize,
    ) -> Result<Qwen80CompleteHybridDecoderPlan> {
        let state = Qwen80NativeStateGeometry::from_config(&self.config, max_seq_len)?;
        let embedding = Qwen80CompleteHybridDecoderPlan::binding(
            self,
            "model.embed_tokens.weight",
            &[QWEN80_VOCAB, QWEN80_HIDDEN],
        )?;
        let mut layers = Vec::with_capacity(QWEN80_LAYERS);
        let mut linear_state_slot = 0usize;
        let mut full_attention_state_slot = 0usize;
        for layer in 0..QWEN80_LAYERS {
            let prefix = format!("model.layers.{layer}");
            let kind = self.config.layer_kind(layer)?;
            let input_layernorm = Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.input_layernorm.weight"),
                &[QWEN80_HIDDEN],
            )?;
            let post_attention_layernorm = Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.post_attention_layernorm.weight"),
                &[QWEN80_HIDDEN],
            )?;
            let (mixer, linear_slot, attention_slot) = match kind {
                Qwen80LayerKind::LinearAttention => {
                    let key_dim = self.config.linear_key_dim();
                    let value_dim = self.config.linear_value_dim();
                    let conv_dim = self.config.linear_conv_dim();
                    let binding = Qwen80LinearDeltaNetLayerBindings {
                        in_proj_qkvz: Qwen80CompleteHybridDecoderPlan::binding(
                            self,
                            format!("{prefix}.linear_attn.in_proj_qkvz.weight"),
                            &[key_dim * 2 + value_dim * 2, QWEN80_HIDDEN],
                        )?,
                        in_proj_ba: Qwen80CompleteHybridDecoderPlan::binding(
                            self,
                            format!("{prefix}.linear_attn.in_proj_ba.weight"),
                            &[QWEN80_LINEAR_VALUE_HEADS * 2, QWEN80_HIDDEN],
                        )?,
                        causal_conv1d: Qwen80CompleteHybridDecoderPlan::binding(
                            self,
                            format!("{prefix}.linear_attn.conv1d.weight"),
                            &[conv_dim, 1, QWEN80_LINEAR_CONV_KERNEL],
                        )?,
                        a_log: Qwen80CompleteHybridDecoderPlan::binding(
                            self,
                            format!("{prefix}.linear_attn.A_log"),
                            &[QWEN80_LINEAR_VALUE_HEADS],
                        )?,
                        dt_bias: Qwen80CompleteHybridDecoderPlan::binding(
                            self,
                            format!("{prefix}.linear_attn.dt_bias"),
                            &[QWEN80_LINEAR_VALUE_HEADS],
                        )?,
                        gated_rms_norm: Qwen80CompleteHybridDecoderPlan::binding(
                            self,
                            format!("{prefix}.linear_attn.norm.weight"),
                            &[QWEN80_LINEAR_VALUE_HEAD_DIM],
                        )?,
                        out_proj: Qwen80CompleteHybridDecoderPlan::binding(
                            self,
                            format!("{prefix}.linear_attn.out_proj.weight"),
                            &[QWEN80_HIDDEN, value_dim],
                        )?,
                    };
                    let slot = linear_state_slot;
                    linear_state_slot = linear_state_slot
                        .checked_add(1)
                        .ok_or_else(|| model_error("linear state slot counter overflowed"))?;
                    (
                        Qwen80HybridMixerBindings::LinearDeltaNet(binding),
                        Some(slot),
                        None,
                    )
                }
                Qwen80LayerKind::FullAttention => {
                    let query_dim = self.config.attention_query_dim();
                    let kv_dim = self.config.attention_kv_dim();
                    let binding = Qwen80FullAttentionLayerBindings {
                        q_proj: Qwen80CompleteHybridDecoderPlan::binding(
                            self,
                            format!("{prefix}.self_attn.q_proj.weight"),
                            &[query_dim * 2, QWEN80_HIDDEN],
                        )?,
                        k_proj: Qwen80CompleteHybridDecoderPlan::binding(
                            self,
                            format!("{prefix}.self_attn.k_proj.weight"),
                            &[kv_dim, QWEN80_HIDDEN],
                        )?,
                        v_proj: Qwen80CompleteHybridDecoderPlan::binding(
                            self,
                            format!("{prefix}.self_attn.v_proj.weight"),
                            &[kv_dim, QWEN80_HIDDEN],
                        )?,
                        q_norm: Qwen80CompleteHybridDecoderPlan::binding(
                            self,
                            format!("{prefix}.self_attn.q_norm.weight"),
                            &[QWEN80_FULL_ATTN_HEAD_DIM],
                        )?,
                        k_norm: Qwen80CompleteHybridDecoderPlan::binding(
                            self,
                            format!("{prefix}.self_attn.k_norm.weight"),
                            &[QWEN80_FULL_ATTN_HEAD_DIM],
                        )?,
                        o_proj: Qwen80CompleteHybridDecoderPlan::binding(
                            self,
                            format!("{prefix}.self_attn.o_proj.weight"),
                            &[QWEN80_HIDDEN, query_dim],
                        )?,
                    };
                    let slot = full_attention_state_slot;
                    full_attention_state_slot =
                        full_attention_state_slot.checked_add(1).ok_or_else(|| {
                            model_error("full-attention state slot counter overflowed")
                        })?;
                    (
                        Qwen80HybridMixerBindings::FullAttention(binding),
                        None,
                        Some(slot),
                    )
                }
            };
            let moe = Qwen80MoeLayerBindings {
                router: Qwen80CompleteHybridDecoderPlan::binding(
                    self,
                    format!("{prefix}.mlp.gate.weight"),
                    &[QWEN80_EXPERTS, QWEN80_HIDDEN],
                )?,
                shared_gate_proj: Qwen80CompleteHybridDecoderPlan::binding(
                    self,
                    format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                    &[QWEN80_SHARED_EXPERT_INTERMEDIATE, QWEN80_HIDDEN],
                )?,
                shared_up_proj: Qwen80CompleteHybridDecoderPlan::binding(
                    self,
                    format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                    &[QWEN80_SHARED_EXPERT_INTERMEDIATE, QWEN80_HIDDEN],
                )?,
                shared_down_proj: Qwen80CompleteHybridDecoderPlan::binding(
                    self,
                    format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                    &[QWEN80_HIDDEN, QWEN80_SHARED_EXPERT_INTERMEDIATE],
                )?,
                shared_expert_gate: Qwen80CompleteHybridDecoderPlan::binding(
                    self,
                    format!("{prefix}.mlp.shared_expert_gate.weight"),
                    &[1, QWEN80_HIDDEN],
                )?,
            };
            layers.push(Qwen80HybridDecoderLayerPlan {
                layer,
                kind,
                linear_state_slot: linear_slot,
                full_attention_state_slot: attention_slot,
                input_layernorm,
                mixer,
                post_attention_layernorm,
                moe,
            });
        }
        if linear_state_slot != state.linear_layers
            || full_attention_state_slot != state.full_attention_layers
        {
            return Err(model_error(format!(
                "hybrid decoder state slots linear={linear_state_slot}/{} attention={full_attention_state_slot}/{} disagree with state geometry",
                state.linear_layers, state.full_attention_layers
            )));
        }
        Ok(Qwen80CompleteHybridDecoderPlan {
            manifest_seal_sha256: self.manifest_seal().to_owned(),
            source_revision: self.config.source_revision.clone(),
            max_seq_len,
            state,
            embedding,
            layers,
            final_norm: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                "model.norm.weight",
                &[QWEN80_HIDDEN],
            )?,
            lm_head: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                "lm_head.weight",
                &[QWEN80_VOCAB, QWEN80_HIDDEN],
            )?,
            tokenizer_vocab_size: QWEN80_TOKENIZER_VOCAB,
            reserved_lm_head_tail_rows: QWEN80_VOCAB - QWEN80_TOKENIZER_VOCAB,
            required_native_operator_gaps: QWEN80_HYBRID_NATIVE_OPERATOR_GAPS.to_vec(),
        })
    }
}

/// A validated source top-10 routing decision.  The route is a backend result
/// rather than a scheduler heuristic; validation rejects duplicates, invalid
/// expert ids, non-finite weights, and unnormalised source probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80RouteSelection {
    pub ids: [u16; QWEN80_TOP_K],
    pub weights: [f32; QWEN80_TOP_K],
}

impl Qwen80RouteSelection {
    pub fn validate(&self) -> Result<()> {
        let mut seen = HashSet::with_capacity(QWEN80_TOP_K);
        let mut total = 0.0f32;
        for (&id, &weight) in self.ids.iter().zip(&self.weights) {
            if id as usize >= QWEN80_EXPERTS || !seen.insert(id) {
                return Err(model_error(format!(
                    "source top-10 route has invalid or duplicate expert id {id}"
                )));
            }
            if !weight.is_finite() || weight < 0.0 {
                return Err(model_error(format!(
                    "source top-10 route has invalid weight {weight} for expert {id}"
                )));
            }
            total += weight;
        }
        if !total.is_finite() || (total - 1.0).abs() > 1.0e-4 {
            return Err(model_error(format!(
                "source norm_topk_prob route weights sum to {total}, expected 1"
            )));
        }
        Ok(())
    }
}

/// The direct-packed backend contract consumed by the complete hybrid
/// scheduler.  It deliberately has no default/CPU-shadow implementation.
/// Production implementations must retain compact payload authority and keep
/// Qwen80 state on the native backend; a CPU implementation is only suitable
/// for bounded parity tests and must never be registered as a production
/// runtime.
pub trait Qwen80PackedHybridDecodeBackend {
    fn begin_token(
        &mut self,
        token_id: u32,
        position: usize,
        embedding: &Qwen80PackedTensorBinding,
    ) -> Result<()>;
    fn input_rms_norm(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()>;
    fn linear_deltanet_mixer(
        &mut self,
        layer: &Qwen80HybridDecoderLayerPlan,
        state_slot: usize,
        mixer: &Qwen80LinearDeltaNetLayerBindings,
    ) -> Result<()>;
    fn full_attention_mixer(
        &mut self,
        layer: &Qwen80HybridDecoderLayerPlan,
        state_slot: usize,
        mixer: &Qwen80FullAttentionLayerBindings,
        position: usize,
    ) -> Result<()>;
    fn add_mixer_residual(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()>;
    fn post_attention_rms_norm(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()>;
    fn route_top10(&mut self, layer: &Qwen80HybridDecoderLayerPlan)
        -> Result<Qwen80RouteSelection>;
    fn routed_expert(
        &mut self,
        layer: &Qwen80HybridDecoderLayerPlan,
        route_index: usize,
        route_weight: f32,
        expert: &Qwen80ExpertBindings,
    ) -> Result<()>;
    fn shared_expert(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()>;
    fn combine_moe_and_add_residual(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()>;
    fn final_rms_norm(&mut self, final_norm: &Qwen80PackedTensorBinding) -> Result<()>;
    fn lm_head(&mut self, lm_head: &Qwen80PackedTensorBinding) -> Result<()>;
    fn mask_reserved_lm_head_tail(&mut self, first_reserved_id: u32) -> Result<()>;
    fn sample_token(&mut self, tokenizer_vocab_size: usize) -> Result<u32>;
}

/// Lower-level executor seam for a single Qwen3-Next layer.  The bridge below
/// owns source order, source tensor identity, top-10 routing validation, and
/// residual boundaries; an implementation of this trait owns only the actual
/// direct-packed operator work.  This is deliberately shaped so the existing
/// layer-0 DeltaNet component and layer-3 GQA component can be connected
/// without teaching either component to schedule all 48 layers itself.
pub trait Qwen80HybridPerLayerComponentExecutor {
    fn begin_token(
        &mut self,
        token_id: u32,
        position: usize,
        embedding: &Qwen80PackedTensorBinding,
    ) -> Result<()>;
    fn input_rms_norm(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()>;
    fn linear_deltanet_component(
        &mut self,
        layer: &Qwen80HybridDecoderLayerPlan,
        state_slot: usize,
        mixer: &Qwen80LinearDeltaNetLayerBindings,
    ) -> Result<()>;
    fn full_attention_component(
        &mut self,
        layer: &Qwen80HybridDecoderLayerPlan,
        state_slot: usize,
        mixer: &Qwen80FullAttentionLayerBindings,
        position: usize,
    ) -> Result<()>;
    fn add_mixer_residual(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()>;
    fn post_attention_rms_norm(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()>;
    fn route_top10(&mut self, layer: &Qwen80HybridDecoderLayerPlan)
        -> Result<Qwen80RouteSelection>;
    fn routed_expert(
        &mut self,
        layer: &Qwen80HybridDecoderLayerPlan,
        route_index: usize,
        route_weight: f32,
        expert: &Qwen80ExpertBindings,
    ) -> Result<()>;
    fn shared_expert(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()>;
    fn combine_moe_and_add_residual(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()>;
    fn final_rms_norm(&mut self, final_norm: &Qwen80PackedTensorBinding) -> Result<()>;
    fn lm_head(&mut self, lm_head: &Qwen80PackedTensorBinding) -> Result<()>;
    fn mask_reserved_lm_head_tail(&mut self, first_reserved_id: u32) -> Result<()>;
    fn sample_token(&mut self, tokenizer_vocab_size: usize) -> Result<u32>;
}

#[derive(Clone, Debug)]
struct Qwen80ActiveRoute {
    layer: usize,
    selection: Qwen80RouteSelection,
    next_route_index: usize,
}

/// Artifact-bound adapter from individual direct-packed components to the
/// complete hybrid scheduler.  It refuses cross-artifact bindings, an invalid
/// DeltaNet/GQA state slot, route/expert ordering drift, or an incomplete
/// top-10 wave before delegating to the component executor.
///
/// This adapter is not itself a production backend.  A Metal executor passed
/// to it must still earn full all-layer numerical/capability evidence.  The
/// CPU test executor exercises this adapter only as deterministic control-flow
/// parity and can never be promoted as a native runtime fallback.
pub struct Qwen80ArtifactBoundPerLayerBackendBridge<E> {
    manifest_seal_sha256: String,
    source_revision: String,
    embedding: Qwen80PackedTensorBinding,
    layers: Vec<Qwen80HybridDecoderLayerPlan>,
    final_norm: Qwen80PackedTensorBinding,
    lm_head: Qwen80PackedTensorBinding,
    tokenizer_vocab_size: usize,
    executor: E,
    active_route: Option<Qwen80ActiveRoute>,
}

impl<E> Qwen80ArtifactBoundPerLayerBackendBridge<E> {
    pub fn new(plan: &Qwen80CompleteHybridDecoderPlan, executor: E) -> Result<Self> {
        if plan.layers.len() != QWEN80_LAYERS
            || plan.manifest_seal_sha256.len() != 64
            || plan.source_revision.is_empty()
            || plan.tokenizer_vocab_size != QWEN80_TOKENIZER_VOCAB
        {
            return Err(model_error(
                "cannot build per-layer bridge from a non-exact Qwen80 hybrid plan",
            ));
        }
        Ok(Self {
            manifest_seal_sha256: plan.manifest_seal_sha256.clone(),
            source_revision: plan.source_revision.clone(),
            embedding: plan.embedding.clone(),
            layers: plan.layers.clone(),
            final_norm: plan.final_norm.clone(),
            lm_head: plan.lm_head.clone(),
            tokenizer_vocab_size: plan.tokenizer_vocab_size,
            executor,
            active_route: None,
        })
    }

    pub fn manifest_seal_sha256(&self) -> &str {
        &self.manifest_seal_sha256
    }

    pub fn source_revision(&self) -> &str {
        &self.source_revision
    }

    pub fn into_inner(self) -> E {
        self.executor
    }

    fn expected_layer(
        &self,
        layer: &Qwen80HybridDecoderLayerPlan,
    ) -> Result<&Qwen80HybridDecoderLayerPlan> {
        let expected = self.layers.get(layer.layer).ok_or_else(|| {
            model_error(format!(
                "per-layer bridge received out-of-range layer {}",
                layer.layer
            ))
        })?;
        if expected != layer {
            return Err(model_error(format!(
                "per-layer bridge layer {} does not match its admitted artifact plan",
                layer.layer
            )));
        }
        Ok(expected)
    }

    fn require_no_active_route(&self, stage: &str) -> Result<()> {
        if let Some(active) = &self.active_route {
            return Err(model_error(format!(
                "per-layer bridge entered {stage} while layer {} route wave is incomplete at index {}",
                active.layer, active.next_route_index
            )));
        }
        Ok(())
    }

    fn active_route_for_layer_mut(&mut self, layer: usize) -> Result<&mut Qwen80ActiveRoute> {
        let active = self.active_route.as_mut().ok_or_else(|| {
            model_error(format!(
                "layer {layer} routed expert operation has no active top-10 route"
            ))
        })?;
        if active.layer != layer {
            return Err(model_error(format!(
                "layer {layer} routed expert operation crossed active route from layer {}",
                active.layer
            )));
        }
        Ok(active)
    }
}

impl<E: Qwen80HybridPerLayerComponentExecutor> Qwen80PackedHybridDecodeBackend
    for Qwen80ArtifactBoundPerLayerBackendBridge<E>
{
    fn begin_token(
        &mut self,
        token_id: u32,
        position: usize,
        embedding: &Qwen80PackedTensorBinding,
    ) -> Result<()> {
        self.require_no_active_route("begin_token")?;
        if embedding != &self.embedding {
            return Err(model_error(
                "per-layer bridge begin_token received an embedding outside its admitted artifact plan",
            ));
        }
        self.executor.begin_token(token_id, position, embedding)
    }

    fn input_rms_norm(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
        self.require_no_active_route("input_rms_norm")?;
        self.expected_layer(layer)?;
        self.executor.input_rms_norm(layer)
    }

    fn linear_deltanet_mixer(
        &mut self,
        layer: &Qwen80HybridDecoderLayerPlan,
        state_slot: usize,
        mixer: &Qwen80LinearDeltaNetLayerBindings,
    ) -> Result<()> {
        let expected = self.expected_layer(layer)?;
        match (
            &expected.mixer,
            expected.linear_state_slot,
            expected.full_attention_state_slot,
        ) {
            (
                Qwen80HybridMixerBindings::LinearDeltaNet(expected_mixer),
                Some(expected_slot),
                None,
            ) if expected_mixer == mixer && expected_slot == state_slot => {}
            _ => {
                return Err(model_error(format!(
                    "layer {} DeltaNet component binding/state slot is not source-exact",
                    layer.layer
                )));
            }
        }
        self.executor
            .linear_deltanet_component(layer, state_slot, mixer)
    }

    fn full_attention_mixer(
        &mut self,
        layer: &Qwen80HybridDecoderLayerPlan,
        state_slot: usize,
        mixer: &Qwen80FullAttentionLayerBindings,
        position: usize,
    ) -> Result<()> {
        let expected = self.expected_layer(layer)?;
        match (
            &expected.mixer,
            expected.linear_state_slot,
            expected.full_attention_state_slot,
        ) {
            (
                Qwen80HybridMixerBindings::FullAttention(expected_mixer),
                None,
                Some(expected_slot),
            ) if expected_mixer == mixer && expected_slot == state_slot => {}
            _ => {
                return Err(model_error(format!(
                    "layer {} GQA component binding/state slot is not source-exact",
                    layer.layer
                )));
            }
        }
        self.executor
            .full_attention_component(layer, state_slot, mixer, position)
    }

    fn add_mixer_residual(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
        self.expected_layer(layer)?;
        self.executor.add_mixer_residual(layer)
    }

    fn post_attention_rms_norm(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
        self.expected_layer(layer)?;
        self.executor.post_attention_rms_norm(layer)
    }

    fn route_top10(
        &mut self,
        layer: &Qwen80HybridDecoderLayerPlan,
    ) -> Result<Qwen80RouteSelection> {
        self.require_no_active_route("route_top10")?;
        self.expected_layer(layer)?;
        let route = self.executor.route_top10(layer)?;
        route.validate()?;
        self.active_route = Some(Qwen80ActiveRoute {
            layer: layer.layer,
            selection: route.clone(),
            next_route_index: 0,
        });
        Ok(route)
    }

    fn routed_expert(
        &mut self,
        layer: &Qwen80HybridDecoderLayerPlan,
        route_index: usize,
        route_weight: f32,
        expert: &Qwen80ExpertBindings,
    ) -> Result<()> {
        self.expected_layer(layer)?;
        let active = self.active_route_for_layer_mut(layer.layer)?;
        if route_index != active.next_route_index || route_index >= QWEN80_TOP_K {
            return Err(model_error(format!(
                "layer {} routed expert index {route_index} does not follow expected {}",
                layer.layer, active.next_route_index
            )));
        }
        if expert.expert != active.selection.ids[route_index] as usize
            || (route_weight - active.selection.weights[route_index]).abs() > 1.0e-7
        {
            return Err(model_error(format!(
                "layer {} routed expert binding/weight drifted from device top-10 route at index {route_index}",
                layer.layer
            )));
        }
        active.next_route_index = active.next_route_index.saturating_add(1);
        self.executor
            .routed_expert(layer, route_index, route_weight, expert)
    }

    fn shared_expert(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
        self.expected_layer(layer)?;
        let active = self.active_route_for_layer_mut(layer.layer)?;
        if active.next_route_index != QWEN80_TOP_K {
            return Err(model_error(format!(
                "layer {} shared expert started before all {QWEN80_TOP_K} routed experts completed",
                layer.layer
            )));
        }
        self.executor.shared_expert(layer)
    }

    fn combine_moe_and_add_residual(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
        self.expected_layer(layer)?;
        let active = self.active_route_for_layer_mut(layer.layer)?;
        if active.next_route_index != QWEN80_TOP_K {
            return Err(model_error(format!(
                "layer {} MoE residual combine started before its routed wave completed",
                layer.layer
            )));
        }
        self.executor.combine_moe_and_add_residual(layer)?;
        self.active_route = None;
        Ok(())
    }

    fn final_rms_norm(&mut self, final_norm: &Qwen80PackedTensorBinding) -> Result<()> {
        self.require_no_active_route("final_rms_norm")?;
        if final_norm != &self.final_norm {
            return Err(model_error(
                "per-layer bridge final RMSNorm is outside its artifact plan",
            ));
        }
        self.executor.final_rms_norm(final_norm)
    }

    fn lm_head(&mut self, lm_head: &Qwen80PackedTensorBinding) -> Result<()> {
        if lm_head != &self.lm_head {
            return Err(model_error(
                "per-layer bridge lm_head is outside its artifact plan",
            ));
        }
        self.executor.lm_head(lm_head)
    }

    fn mask_reserved_lm_head_tail(&mut self, first_reserved_id: u32) -> Result<()> {
        if first_reserved_id as usize != self.tokenizer_vocab_size {
            return Err(model_error(format!(
                "per-layer bridge tail mask starts at {first_reserved_id}, expected {}",
                self.tokenizer_vocab_size
            )));
        }
        self.executor.mask_reserved_lm_head_tail(first_reserved_id)
    }

    fn sample_token(&mut self, tokenizer_vocab_size: usize) -> Result<u32> {
        if tokenizer_vocab_size != self.tokenizer_vocab_size {
            return Err(model_error(format!(
                "per-layer bridge sample namespace {tokenizer_vocab_size} differs from admitted {}",
                self.tokenizer_vocab_size
            )));
        }
        self.executor.sample_token(tokenizer_vocab_size)
    }
}

const QWEN80_DIRECT_PACKED_LINEAR_COMPONENT_STATUS: &str =
    "EARNED_QWEN80_ADMITTED_DIRECT_PACKED_FIRST_LINEAR_DELTANET_ROUTER_EXPERT_STAGE_NOT_FULL_LAYER_OR_TOKEN";
const QWEN80_DIRECT_PACKED_LAYER3_GQA_COMPONENT_STATUS: &str =
    "EARNED_QWEN80_ADMITTED_DIRECT_PACKED_LAYER3_GQA_TWO_TOKEN_COMPONENT_STAGE_NOT_COMPLETE_LAYER_OR_TOKEN";

fn ledger_object<'a>(value: &'a Value, field: &str) -> Result<&'a Value> {
    value
        .get(field)
        .filter(|value| value.is_object())
        .ok_or_else(|| model_error(format!("component ledger missing object {field:?}")))
}

fn ledger_array<'a>(value: &'a Value, field: &str) -> Result<&'a Vec<Value>> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| model_error(format!("component ledger missing array {field:?}")))
}

fn ledger_string<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| model_error(format!("component ledger missing non-empty {field:?}")))
}

fn ledger_usize(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| model_error(format!("component ledger missing unsigned {field:?}")))
}

fn require_ledger_string(value: &Value, field: &str, expected: &str) -> Result<()> {
    let observed = ledger_string(value, field)?;
    if observed != expected {
        return Err(model_error(format!(
            "component ledger {field:?}={observed:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn ledger_bool(value: &Value, field: &str) -> Result<bool> {
    value
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| model_error(format!("component ledger missing boolean {field:?}")))
}

fn require_canonical_sha256(value: &str, label: &str) -> Result<()> {
    let is_lowercase_hex = value.bytes().all(|byte| {
        byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
    });
    if value.len() != 64 || !is_lowercase_hex {
        return Err(model_error(format!(
            "{label} is not a canonical lowercase SHA-256 digest"
        )));
    }
    Ok(())
}

fn ledger_sha256<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    let digest = ledger_string(value, field)?;
    require_canonical_sha256(digest, &format!("component ledger {field:?}"))?;
    Ok(digest)
}

fn require_ledger_sha256(value: &Value, field: &str, expected: &str) -> Result<()> {
    let observed = ledger_sha256(value, field)?;
    if observed != expected {
        return Err(model_error(format!(
            "component ledger {field:?} SHA-256 {observed:?} differs from expected {expected:?}"
        )));
    }
    Ok(())
}

fn ledger_f32(value: &Value, field: &str) -> Result<f32> {
    let number = value
        .get(field)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .ok_or_else(|| model_error(format!("component ledger missing finite number {field:?}")))?;
    let number = number as f32;
    if !number.is_finite() {
        return Err(model_error(format!(
            "component ledger {field:?} is outside finite f32 range"
        )));
    }
    Ok(number)
}

fn ledger_vector_sha256(
    value: &Value,
    field: &str,
    expected_elements: usize,
    label: &str,
) -> Result<String> {
    let vector = ledger_object(value, field)?;
    if ledger_usize(vector, "elements")? != expected_elements {
        return Err(model_error(format!(
            "{label} vector {field:?} has wrong element count"
        )));
    }
    if !ledger_bool(vector, "all_finite")? {
        return Err(model_error(format!(
            "{label} vector {field:?} is not sealed as finite"
        )));
    }
    Ok(ledger_sha256(vector, "sha256")?.to_owned())
}

fn route_from_ledger_arrays(
    value: &Value,
    ids_field: &str,
    weights_field: &str,
    label: &str,
) -> Result<Qwen80RouteSelection> {
    let ids_value = ledger_array(value, ids_field)?;
    let weights_value = ledger_array(value, weights_field)?;
    if ids_value.len() != QWEN80_TOP_K || weights_value.len() != QWEN80_TOP_K {
        return Err(model_error(format!(
            "{label} has ids={} weights={}, expected {QWEN80_TOP_K}",
            ids_value.len(),
            weights_value.len()
        )));
    }
    let mut ids = [0u16; QWEN80_TOP_K];
    let mut weights = [0.0f32; QWEN80_TOP_K];
    for index in 0..QWEN80_TOP_K {
        ids[index] = ids_value[index]
            .as_u64()
            .and_then(|id| u16::try_from(id).ok())
            .ok_or_else(|| model_error(format!("{label} route id {index} is outside u16")))?;
        let weight = weights_value[index]
            .as_f64()
            .filter(|weight| weight.is_finite())
            .map(|weight| weight as f32)
            .filter(|weight| weight.is_finite())
            .ok_or_else(|| model_error(format!("{label} route weight {index} is non-finite")))?;
        weights[index] = weight;
    }
    let route = Qwen80RouteSelection { ids, weights };
    route.validate()?;
    Ok(route)
}

fn require_same_route(
    observed: &Qwen80RouteSelection,
    expected: &Qwen80RouteSelection,
    label: &str,
) -> Result<()> {
    if observed.ids != expected.ids {
        return Err(model_error(format!(
            "{label} expert ids {:?} differ from the exact router top-10 {:?}",
            observed.ids, expected.ids
        )));
    }
    for (index, (&observed_weight, &expected_weight)) in
        observed.weights.iter().zip(&expected.weights).enumerate()
    {
        if (observed_weight - expected_weight).abs() > 1.0e-6 {
            return Err(model_error(format!(
                "{label} route weight {index}={observed_weight} differs from router {expected_weight}"
            )));
        }
    }
    Ok(())
}

fn require_qwen80_moe_component_metal<'a>(receipt: &'a Value, label: &str) -> Result<&'a Value> {
    require_ledger_string(receipt, "mode", "metal")?;
    if !ledger_bool(receipt, "metal_device_or_dispatch_performed")? {
        return Err(model_error(format!(
            "{label} did not record a real Metal device/dispatch path"
        )));
    }
    let metal = ledger_object(receipt, "metal_intermediate_error_ledger")?;
    if !ledger_bool(metal, "performed")?
        || !ledger_bool(metal, "strict_math")?
        || ledger_bool(metal, "timing_or_benchmarking_performed")?
        || ledger_usize(metal, "command_buffers")? == 0
        || ledger_usize(metal, "compute_dispatches")? == 0
    {
        return Err(model_error(format!(
            "{label} lacks a non-timed strict-Math Metal intermediate ledger"
        )));
    }
    Ok(metal)
}

fn require_qwen80_moe_component_authority<'a>(
    receipt: &'a Value,
    plan: &Qwen80CompleteHybridDecoderPlan,
    layer: &Qwen80HybridDecoderLayerPlan,
    admission_receipt_seal_sha256: &str,
    manifest_document_sha256: &str,
    label: &str,
) -> Result<&'a Value> {
    require_canonical_sha256(
        admission_receipt_seal_sha256,
        "requested Qwen80 admission receipt seal",
    )?;
    require_canonical_sha256(
        manifest_document_sha256,
        "requested Qwen80 complete-manifest document SHA-256",
    )?;
    let binding = ledger_object(receipt, "artifact_binding")?;
    require_ledger_string(binding, "manifest_seal_sha256", &plan.manifest_seal_sha256)?;
    require_ledger_sha256(
        binding,
        "manifest_document_sha256",
        manifest_document_sha256,
    )?;
    require_canonical_sha256(&plan.manifest_seal_sha256, "hybrid plan manifest seal")?;
    require_ledger_string(binding, "source_revision", &plan.source_revision)?;
    require_ledger_sha256(
        binding,
        "admission_receipt_seal_sha256",
        admission_receipt_seal_sha256,
    )?;
    if ledger_usize(binding, "layer")? != layer.layer
        || ledger_string(binding, "layer_kind")? != layer.kind.as_source_name()
        || ledger_usize(binding, "hidden")? != QWEN80_HIDDEN
        || ledger_usize(binding, "experts")? != QWEN80_EXPERTS
        || ledger_usize(binding, "experts_per_token")? != QWEN80_TOP_K
    {
        return Err(model_error(format!(
            "{label} artifact binding does not describe exact Qwen80 layer {} MoE geometry",
            layer.layer
        )));
    }
    Ok(binding)
}

/// Schema required for a future routed-expert witness that is one member of a
/// complete source top-10 wave.  The existing expert-65 receipt intentionally
/// does not use this status and therefore cannot be promoted by this contract.
pub const QWEN80_REAL_ROUTED_EXPERT_COMPONENT_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_routed_expert_wave.v1";
pub const QWEN80_REAL_ROUTED_EXPERT_COMPONENT_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_ROUTED_EXPERT_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_MOE_OR_LAYER";

/// Schema required for the device-produced hidden vector after a real Qwen80
/// mixer residual.  It is deliberately a component-level fact, not a layer or
/// decoder receipt.
pub const QWEN80_FIRST_RESIDUAL_COMPONENT_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_mixer_first_residual.v1";
pub const QWEN80_FIRST_RESIDUAL_COMPONENT_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_MIXER_FIRST_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";

/// Schema/status required before a source top-10 router receipt may seed a
/// real MoE boundary.  A synthetic router component may retain the same
/// schema/status, but it cannot pass this contract without a hash binding to
/// a real first-residual vector.
pub const QWEN80_REAL_MOE_ROUTER_COMPONENT_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_postnorm_router_top10.v1";
pub const QWEN80_REAL_MOE_ROUTER_COMPONENT_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";

/// The shared expert remains a component even after it participates in a
/// real MoE boundary.  Its output must be linked to the same first-residual
/// and post-norm vector as every routed expert.
pub const QWEN80_REAL_SHARED_EXPERT_COMPONENT_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_shared_expert_wave.v1";
pub const QWEN80_REAL_SHARED_EXPERT_COMPONENT_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_SHARED_EXPERT_STRICT_MATH_METAL_COMPONENT_NOT_ROUTED_MOE_OR_LAYER";

/// The future device combine witness must consume ten physical expert vectors
/// plus one shared vector and one first residual.  The current materialized
/// route-shaped CPU fixture uses a different status and is explicitly refused.
pub const QWEN80_REAL_MOE_COMBINE_COMPONENT_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_moe_combine.v1";
pub const QWEN80_REAL_MOE_COMBINE_COMPONENT_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_MOE_COMBINE_SHARED_ADD_SECOND_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";

/// A descriptor-only all-ten routed-expert plan is required before a future
/// physical MoE boundary can accept ten device witnesses.  The plan itself is
/// deliberately non-executing; it contributes source identity and exact
/// tensor bindings, never a layer/token result.
pub const QWEN80_ALL_TEN_ROUTE_PLAN_SCHEMA: &str =
    "hawking.ascension.qwen80_all_ten_routed_expert_binding_plan.v1";
pub const QWEN80_ALL_TEN_ROUTE_PLAN_STATUS: &str =
    "SOURCE_BOUND_ALL_TEN_ROUTED_EXPERT_PLAN_READY_NOT_EXECUTED_NOT_COMBINED";
pub const QWEN80_ALL_TEN_ROUTE_PLAN_GATE_SCHEMA: &str =
    "hawking.ascension.qwen80_real_all_ten_routed_expert_provenance_gate_input.v1";

/// A source-token route plan is deliberately distinct from the historical
/// fixture-derived all-ten descriptor.  It is emitted only after replaying a
/// retained source token through the admitted direct-packed L0 CPU oracle and
/// binding that result to a sealed strict-Metal first-residual prefix.
pub const QWEN80_SOURCE_TOKEN_ALL_TEN_ROUTE_PLAN_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_all_ten_routed_expert_binding_plan.v1";
pub const QWEN80_SOURCE_TOKEN_ALL_TEN_ROUTE_PLAN_STATUS: &str =
    "SOURCE_TOKEN_BOUND_ALL_TEN_ROUTED_EXPERT_PLAN_READY_NOT_EXECUTED_NOT_COMBINED";
pub const QWEN80_SOURCE_TOKEN_ALL_TEN_ROUTE_PLAN_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_all_ten_route_plan_authority.v1";
pub const QWEN80_SOURCE_TOKEN_ALL_TEN_ROUTE_PLAN_AUTHORITY_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_ALL_TEN_ROUTE_PLAN_READY_FOR_NEW_TYPED_BRIDGE";

/// Inputs for a CPU-only provenance check over device-component receipts.  A
/// caller must obtain the two router receipt digests from the durable capture
/// chain; this type never treats a mutable current-pointer file as authority.
pub struct Qwen80RealMoeBoundaryInputs<'a> {
    pub admission_receipt_seal_sha256: &'a str,
    /// Raw SHA-256 of the immutable complete-binary manifest document, not a
    /// mutable current pointer or its path.
    pub manifest_document_sha256: &'a str,
    pub router_receipt_sha256: &'a str,
    pub router_outer_receipt_sha256: &'a str,
    pub router_outer_receipt_seal_sha256: &'a str,
    /// Raw SHA-256 of the immutable descriptor-only all-ten plan.  The caller
    /// must calculate it while reading the plan file; a parsed JSON `Value`
    /// alone cannot prove its original bytes.
    pub all_ten_route_plan_document_sha256: &'a str,
    pub all_ten_route_plan: &'a Value,
    pub first_residual_receipt: &'a Value,
    pub router_receipt: &'a Value,
    pub routed_expert_receipts: &'a [Value],
    pub shared_expert_receipt: &'a Value,
    pub combine_receipt: &'a Value,
}

/// A verified MoE boundary proof.  It proves only the receipt relationship
/// for one layer: it neither executes a layer nor discharges any remaining
/// Qwen80 decoder, generation, HCLI, TPS, or tournament gate.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80VerifiedRealMoeBoundary {
    pub layer: usize,
    /// Immutable authority retained with the proof so a later all-layer
    /// harness cannot detach the accepted boundary from its exact artifact.
    pub manifest_document_sha256: String,
    /// Immutable descriptor authority for all ten routed waves.
    pub all_ten_route_plan_document_sha256: String,
    pub route: Qwen80RouteSelection,
    pub first_residual_sha256: String,
    pub post_attention_normalized_hidden_sha256: String,
    pub routed_weighted_delta_sha256: Vec<String>,
    pub gated_shared_sha256: String,
    pub second_residual_sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
struct Qwen80AllTenRoutePlanProjection {
    tensor_name: String,
    artifact_sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
struct Qwen80AllTenRoutePlanWave {
    expert: u16,
    normalized_weight: f32,
    gate: Qwen80AllTenRoutePlanProjection,
    up: Qwen80AllTenRoutePlanProjection,
    down: Qwen80AllTenRoutePlanProjection,
}

/// Immutable, descriptor-only authority for a single layer's source-selected
/// routed top-10.  It contains no decoded weights or execution result: a
/// caller must feed it to the direct-packed CPU/device executor and retain
/// the plan document SHA in every resulting witness.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80AllTenRoutedExpertPlan {
    manifest_document_sha256: String,
    plan_document_sha256: String,
    manifest_seal_sha256: String,
    source_revision: String,
    layer: usize,
    route: Qwen80RouteSelection,
    waves: Vec<Qwen80AllTenRoutePlanWave>,
}

/// One exact compact tensor section inside the concatenated all-ten device
/// payload.  The offsets address *only* the scale/sign body copied from an
/// immutable admission snapshot; they never address a decoded weight matrix
/// or a file reopened after admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen80AllTenPackedRouteProjectionSection {
    pub tensor_name: String,
    pub artifact_sha256: String,
    pub shape: Vec<usize>,
    pub group_size: usize,
    pub scale_offset_bytes: usize,
    pub scale_bytes: usize,
    pub sign_offset_bytes: usize,
    pub sign_bytes: usize,
}

/// The immutable source identity and offsets for one of the ten route waves.
/// It is a payload-layout witness, not an execution receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80AllTenPackedRouteWave {
    pub wave_index: usize,
    pub expert: u16,
    pub normalized_weight: f32,
    pub gate: Qwen80AllTenPackedRouteProjectionSection,
    pub up: Qwen80AllTenPackedRouteProjectionSection,
    pub down: Qwen80AllTenPackedRouteProjectionSection,
}

/// Direct-packed source-selected all-ten route bodies laid out for a single
/// generic device graph.  The six byte arrays contain exactly ten compact
/// scale/sign sections in source router order.  They intentionally exclude
/// headers, decoded weights, the shared expert, residual vectors, and any
/// device execution result.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80AllTenPackedRoutePayloadBundle {
    manifest_document_sha256: String,
    manifest_seal_sha256: String,
    source_revision: String,
    plan_document_sha256: String,
    layer: usize,
    route: Qwen80RouteSelection,
    waves: Vec<Qwen80AllTenPackedRouteWave>,
    gate_scales: Arc<[u8]>,
    gate_signs: Arc<[u8]>,
    up_scales: Arc<[u8]>,
    up_signs: Arc<[u8]>,
    down_scales: Arc<[u8]>,
    down_signs: Arc<[u8]>,
    gate_scales_sha256: String,
    gate_signs_sha256: String,
    up_scales_sha256: String,
    up_signs_sha256: String,
    down_scales_sha256: String,
    down_signs_sha256: String,
}

impl Qwen80AllTenPackedRoutePayloadBundle {
    pub fn manifest_document_sha256(&self) -> &str {
        &self.manifest_document_sha256
    }

    pub fn manifest_seal_sha256(&self) -> &str {
        &self.manifest_seal_sha256
    }

    pub fn source_revision(&self) -> &str {
        &self.source_revision
    }

    pub fn plan_document_sha256(&self) -> &str {
        &self.plan_document_sha256
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn route(&self) -> &Qwen80RouteSelection {
        &self.route
    }

    pub fn waves(&self) -> &[Qwen80AllTenPackedRouteWave] {
        &self.waves
    }

    pub fn gate_scales(&self) -> &[u8] {
        &self.gate_scales
    }

    pub fn gate_signs(&self) -> &[u8] {
        &self.gate_signs
    }

    pub fn up_scales(&self) -> &[u8] {
        &self.up_scales
    }

    pub fn up_signs(&self) -> &[u8] {
        &self.up_signs
    }

    pub fn down_scales(&self) -> &[u8] {
        &self.down_scales
    }

    pub fn down_signs(&self) -> &[u8] {
        &self.down_signs
    }

    pub fn gate_scales_sha256(&self) -> &str {
        &self.gate_scales_sha256
    }

    pub fn gate_signs_sha256(&self) -> &str {
        &self.gate_signs_sha256
    }

    pub fn up_scales_sha256(&self) -> &str {
        &self.up_scales_sha256
    }

    pub fn up_signs_sha256(&self) -> &str {
        &self.up_signs_sha256
    }

    pub fn down_scales_sha256(&self) -> &str {
        &self.down_scales_sha256
    }

    pub fn down_signs_sha256(&self) -> &str {
        &self.down_signs_sha256
    }

    /// Reject a router result unless it is the exact source-selected route
    /// authority for this compact body bundle.  Device parity uses a bounded
    /// numerical tolerance for normalized probabilities, while expert order
    /// remains byte-for-byte exact.  This method is a refusal gate only; it
    /// does not execute any projection or promote a component into a layer.
    pub fn require_router_route(
        &self,
        observed: &Qwen80RouteSelection,
        weight_tolerance: f32,
    ) -> Result<()> {
        self.validate()?;
        observed.validate()?;
        if !weight_tolerance.is_finite() || weight_tolerance < 0.0 {
            return Err(model_error(
                "all-ten packed route router weight tolerance is invalid",
            ));
        }
        for index in 0..QWEN80_TOP_K {
            if observed.ids[index] != self.route.ids[index] {
                return Err(model_error(format!(
                    "all-ten packed route router expert at index {index} is {}, expected {}",
                    observed.ids[index], self.route.ids[index]
                )));
            }
            let error = (observed.weights[index] - self.route.weights[index]).abs();
            if error > weight_tolerance {
                return Err(model_error(format!(
                    "all-ten packed route router weight at index {index} differs by {error}, tolerance {weight_tolerance}",
                )));
            }
        }
        Ok(())
    }
}

/// The expected physical boundary between a source-scheduled DeltaNet mixer
/// and its post-attention MoE suffix.  This proves only that a future pinned
/// buffer has the required layer/state/shape authority.  It cannot prove the
/// buffer's contents were produced by a mixer dispatch; the future strict
/// same-command-buffer parity capture must retain that separate witness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen80FirstResidualDeviceBinding {
    manifest_seal_sha256: String,
    source_revision: String,
    layer: usize,
    linear_state_slot: usize,
    elements: usize,
    same_command_graph_required: bool,
}

impl Qwen80FirstResidualDeviceBinding {
    pub fn manifest_seal_sha256(&self) -> &str {
        &self.manifest_seal_sha256
    }

    pub fn source_revision(&self) -> &str {
        &self.source_revision
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn linear_state_slot(&self) -> usize {
        self.linear_state_slot
    }

    pub fn elements(&self) -> usize {
        self.elements
    }

    pub fn same_command_graph_required(&self) -> bool {
        self.same_command_graph_required
    }
}

/// CPU/build-time bridge between one already-admitted Qwen80 catalog, exact
/// source-selected compact route bodies, and the expected DeltaNet first
/// residual.  It does not allocate a Metal buffer or claim the residual has
/// been computed.  On macOS a later explicitly leased caller may upload these
/// immutable compact sections and attach the actual `PinnedBuffer` through
/// [`Self::upload_with_first_residual`].
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80AllTenTrueMoeSourceBridge {
    route_payloads: Qwen80AllTenPackedRoutePayloadBundle,
    first_residual: Qwen80FirstResidualDeviceBinding,
}

impl Qwen80AllTenTrueMoeSourceBridge {
    pub fn route_payloads(&self) -> &Qwen80AllTenPackedRoutePayloadBundle {
        &self.route_payloads
    }

    pub fn first_residual(&self) -> &Qwen80FirstResidualDeviceBinding {
        &self.first_residual
    }
}

/// Immutable documents which tie an all-ten descriptor to the already-sealed
/// router capture.  These values must be hashes captured while reading the
/// regular files; paths/current pointers alone are intentionally absent.
#[derive(Clone, Debug)]
pub struct Qwen80AllTenRoutedExpertPlanAuthority<'a> {
    pub manifest_document_sha256: &'a str,
    pub plan_document_sha256: &'a str,
    pub router_receipt_sha256: &'a str,
    pub router_outer_receipt_sha256: &'a str,
    pub router_outer_receipt_seal_sha256: &'a str,
}

/// Immutable identities captured while reading the sealed source-token route
/// authority.  This is intentionally a different type from
/// [`Qwen80AllTenRoutedExpertPlanAuthority`]: a source-token plan must never
/// be redirected through the legacy synthetic-router receipt chain.
#[derive(Clone, Debug)]
pub struct Qwen80SourceTokenAllTenRoutedExpertPlanAuthority<'a> {
    pub manifest_document_sha256: &'a str,
    pub plan_authority_document_sha256: &'a str,
    pub admission_receipt_seal_sha256: &'a str,
    pub first_residual_outer_receipt_seal_sha256: &'a str,
}

fn require_exact_moe_tensor_name(
    binding: &Value,
    field: &str,
    expected_name: &str,
    label: &str,
) -> Result<()> {
    let tensor = ledger_object(binding, field)?;
    require_ledger_string(tensor, "name", expected_name).map_err(|error| {
        model_error(format!(
            "{label} has wrong tensor binding for {field:?}: {error}"
        ))
    })
}

fn require_exact_moe_tensor_binding(
    binding: &Value,
    field: &str,
    expected: &Qwen80AllTenRoutePlanProjection,
    label: &str,
) -> Result<()> {
    let tensor = ledger_object(binding, field)?;
    require_ledger_string(tensor, "name", &expected.tensor_name).map_err(|error| {
        model_error(format!(
            "{label} has wrong tensor binding for {field:?}: {error}"
        ))
    })?;
    require_ledger_sha256(tensor, "artifact_sha256", &expected.artifact_sha256).map_err(|error| {
        model_error(format!(
            "{label} has wrong direct-packed artifact binding for {field:?}: {error}"
        ))
    })
}

fn require_plan_shape(value: &Value, expected: &[usize], label: &str) -> Result<()> {
    let observed = ledger_array(value, "shape")?;
    if observed.len() != expected.len() {
        return Err(model_error(format!(
            "{label} shape rank {} differs from expected {}",
            observed.len(),
            expected.len()
        )));
    }
    for (index, (&expected, observed)) in expected.iter().zip(observed).enumerate() {
        let observed = observed
            .as_u64()
            .and_then(|value| usize::try_from(value).ok());
        if observed != Some(expected) {
            return Err(model_error(format!(
                "{label} shape element {index} differs from expected {expected}"
            )));
        }
    }
    Ok(())
}

fn require_all_ten_plan_projection(
    wave: &Value,
    field: &str,
    expected_name: &str,
    expected_shape: &[usize],
    unique_artifacts: &mut HashSet<String>,
) -> Result<Qwen80AllTenRoutePlanProjection> {
    let projection = ledger_object(wave, field)?;
    require_ledger_string(projection, "tensor_name", expected_name)?;
    require_plan_shape(
        projection,
        expected_shape,
        &format!("all-ten route plan {field} projection"),
    )?;
    if ledger_usize(projection, "elements")? != QWEN80_HIDDEN * QWEN80_MOE_INTERMEDIATE
        || ledger_usize(projection, "artifact_bytes")? == 0
        || ledger_bool(projection, "payload_opened_by_this_plan")?
    {
        return Err(model_error(format!(
            "all-ten route plan {field} projection has invalid direct-packed descriptor boundary"
        )));
    }
    let artifact_path = ledger_string(projection, "artifact_path")?;
    if !Path::new(artifact_path).is_absolute() {
        return Err(model_error(format!(
            "all-ten route plan {field} projection artifact path is not absolute"
        )));
    }
    require_ledger_string(projection, "source_dtype", "BF16")?;
    let source_shard = ledger_string(projection, "source_shard")?;
    if !source_shard.ends_with(".safetensors") {
        return Err(model_error(format!(
            "all-ten route plan {field} projection source shard is not safetensors"
        )));
    }
    ledger_sha256(projection, "source_shard_sha256")?;
    let artifact_sha256 = ledger_sha256(projection, "artifact_sha256")?.to_owned();
    if !unique_artifacts.insert(artifact_sha256.clone()) {
        return Err(model_error(format!(
            "all-ten route plan reuses direct-packed artifact {artifact_sha256} across projections"
        )));
    }
    let layout = ledger_object(projection, "layout")?;
    require_ledger_string(layout, "magic", "HQ30G1B1")?;
    if ledger_usize(layout, "group_size")? != QWEN80_GROUP_SIZE
        || ledger_usize(layout, "version")? != 1
    {
        return Err(model_error(format!(
            "all-ten route plan {field} projection has incompatible packed layout"
        )));
    }
    require_ledger_string(layout, "scale_dtype", "float16")?;
    require_ledger_string(layout, "sign_bit_order", "little")?;
    Ok(Qwen80AllTenRoutePlanProjection {
        tensor_name: expected_name.to_owned(),
        artifact_sha256,
    })
}

fn require_all_ten_route_plan(
    plan: &Qwen80CompleteHybridDecoderPlan,
    layer: &Qwen80HybridDecoderLayerPlan,
    authority: &Qwen80AllTenRoutedExpertPlanAuthority<'_>,
    descriptor: &Value,
) -> Result<Qwen80AllTenRoutedExpertPlan> {
    require_canonical_sha256(
        authority.plan_document_sha256,
        "all-ten routed-expert plan document SHA-256",
    )?;
    require_canonical_sha256(
        authority.router_outer_receipt_sha256,
        "router outer receipt document SHA-256",
    )?;
    require_ledger_string(descriptor, "schema", QWEN80_ALL_TEN_ROUTE_PLAN_SCHEMA)?;
    require_ledger_string(descriptor, "status", QWEN80_ALL_TEN_ROUTE_PLAN_STATUS)?;
    require_ledger_string(descriptor, "model_id", QWEN80_MODEL_ID)?;
    require_ledger_string(descriptor, "model_key", "qwen80")?;
    require_ledger_string(descriptor, "source_repository", QWEN80_REPOSITORY)?;
    require_ledger_string(descriptor, "source_revision", &plan.source_revision)?;
    if ledger_usize(descriptor, "layer")? != layer.layer {
        return Err(model_error(format!(
            "all-ten route plan belongs to layer {}, not requested layer {}",
            ledger_usize(descriptor, "layer")?,
            layer.layer
        )));
    }
    for field in [
        "route_execution_performed",
        "route_combine_performed",
        "shared_expert_performed",
        "residual_combine_performed",
        "metal_device_or_dispatch_performed",
        "model_execution_performed",
        "hcli_execution_performed",
        "tps_or_tg_measurement_performed",
        "complete_layer_or_decoder_claim_earned",
    ] {
        if ledger_bool(descriptor, field)? {
            return Err(model_error(format!(
                "all-ten route plan must remain descriptor-only; {field:?} was true"
            )));
        }
    }

    let manifest = ledger_object(descriptor, "manifest_descriptor_inventory")?;
    require_ledger_string(
        manifest,
        "manifest_schema",
        "hawking.ascension.qwen80_complete_binary_gravity.v1",
    )?;
    require_ledger_string(
        manifest,
        "manifest_status",
        "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED",
    )?;
    require_ledger_sha256(
        manifest,
        "inventory_document_sha256",
        authority.manifest_document_sha256,
    )?;
    require_ledger_sha256(manifest, "manifest_seal_sha256", &plan.manifest_seal_sha256)?;
    require_ledger_string(manifest, "source_repository", QWEN80_REPOSITORY)?;
    if ledger_usize(manifest, "declared_tensor_count")? != QWEN80_COMPLETE_BINARY_TENSORS
        || ledger_usize(manifest, "received_descriptor_count")? != QWEN80_COMPLETE_BINARY_TENSORS
        || ledger_usize(manifest, "resolved_route_tensor_count")? != QWEN80_TOP_K * 3
        || ledger_bool(manifest, "payload_opened_by_this_plan")?
    {
        return Err(model_error(
            "all-ten route plan manifest inventory is incomplete or opens payloads",
        ));
    }

    let router = ledger_object(descriptor, "router_evidence")?;
    require_ledger_sha256(
        router,
        "outer_receipt_document_sha256",
        authority.router_outer_receipt_sha256,
    )?;
    require_ledger_sha256(
        router,
        "outer_receipt_seal_sha256",
        authority.router_outer_receipt_seal_sha256,
    )?;
    require_ledger_sha256(
        router,
        "inner_receipt_document_sha256",
        authority.router_receipt_sha256,
    )?;
    if !ledger_bool(router, "source_router_component_only")? {
        return Err(model_error(
            "all-ten route plan must explicitly preserve router component-only scope",
        ));
    }
    let route = route_from_ledger_arrays(
        router,
        "source_stable_route_ids",
        "source_stable_normalized_weights",
        "all-ten route plan source route",
    )?;

    let rawls = ledger_object(descriptor, "rawls_real_all_ten_provenance_gate")?;
    require_ledger_string(rawls, "schema", QWEN80_ALL_TEN_ROUTE_PLAN_GATE_SCHEMA)?;
    if !ledger_bool(rawls, "all_ten_source_bindings_complete")?
        || ledger_usize(rawls, "expected_layer")? != layer.layer
    {
        return Err(model_error(
            "all-ten route plan Rawls provenance gate is incomplete or for another layer",
        ));
    }
    let rawls_route = route_from_ledger_arrays(
        rawls,
        "route_order",
        "normalized_weights",
        "all-ten route plan Rawls route",
    )?;
    require_same_route(&rawls_route, &route, "all-ten route plan Rawls route")?;
    let indices = ledger_array(rawls, "deterministic_wave_indices")?;
    if indices.len() != QWEN80_TOP_K
        || indices
            .iter()
            .enumerate()
            .any(|(index, value)| value.as_u64() != Some(index as u64))
    {
        return Err(model_error(
            "all-ten route plan Rawls wave indices are not the exact ordered 0..9 schedule",
        ));
    }
    for field in [
        "execution_receipt_required_for_each_wave",
        "direct_packed_execution_required_for_each_wave",
        "source_bound_input_required_for_each_wave",
        "route_combine_receipt_required_separately",
        "shared_expert_receipt_required_separately",
        "first_and_second_residual_receipts_required_separately",
        "rejects_tensor_substitution",
        "rejects_route_reorder",
        "rejects_duplicate_experts",
        "rejects_missing_tensor_or_weight",
    ] {
        if !ledger_bool(rawls, field)? {
            return Err(model_error(format!(
                "all-ten route plan Rawls gate omitted required guard {field:?}"
            )));
        }
    }

    let waves = ledger_array(descriptor, "deterministic_waves")?;
    if waves.len() != QWEN80_TOP_K {
        return Err(model_error(format!(
            "all-ten route plan has {} waves, expected {QWEN80_TOP_K}",
            waves.len()
        )));
    }
    let mut unique_artifacts = HashSet::with_capacity(QWEN80_TOP_K * 3);
    let mut parsed_waves = Vec::with_capacity(QWEN80_TOP_K);
    for (index, wave) in waves.iter().enumerate() {
        let expert = route.ids[index];
        if ledger_usize(wave, "wave_index")? != index
            || ledger_usize(wave, "layer")? != layer.layer
            || ledger_usize(wave, "expert_id")? != usize::from(expert)
        {
            return Err(model_error(format!(
                "all-ten route plan wave {index} does not bind its exact source route"
            )));
        }
        let normalized_weight = ledger_f32(wave, "normalized_weight")?;
        if (normalized_weight - route.weights[index]).abs() > 1.0e-6 {
            return Err(model_error(format!(
                "all-ten route plan wave {index} normalized weight differs from source route"
            )));
        }
        let weight_bits = ledger_string(wave, "normalized_weight_bits_hex")?;
        if weight_bits.len() != 18
            || !weight_bits.starts_with("0x")
            || !weight_bits[2..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(model_error(format!(
                "all-ten route plan wave {index} lacks a canonical f64 source-weight bit record"
            )));
        }
        let operation_order = ledger_array(wave, "fixed_operation_order")?;
        const EXPECTED_ROUTE_OPERATION_ORDER: [&str; 5] = [
            "gate_proj [512,2048]",
            "up_proj [512,2048]",
            "SiLU(gate) * up [512]",
            "down_proj [2048,512]",
            "apply this route's source-normalized weight [2048]",
        ];
        if operation_order.len() != EXPECTED_ROUTE_OPERATION_ORDER.len()
            || operation_order
                .iter()
                .zip(EXPECTED_ROUTE_OPERATION_ORDER)
                .any(|(observed, expected)| observed.as_str() != Some(expected))
        {
            return Err(model_error(format!(
                "all-ten route plan wave {index} has a non-source-exact expert operation order"
            )));
        }
        require_ledger_string(
            wave,
            "route_execution_status",
            "NOT_EXECUTED_SOURCE_BOUND_PLAN_ONLY",
        )?;
        if ledger_bool(wave, "route_delta_materialized")?
            || ledger_bool(wave, "route_weight_applied")?
        {
            return Err(model_error(format!(
                "all-ten route plan wave {index} is not descriptor-only"
            )));
        }
        let prefix = format!("model.layers.{}.mlp.experts.{expert}", layer.layer);
        let gate = require_all_ten_plan_projection(
            wave,
            "gate",
            &format!("{prefix}.gate_proj.weight"),
            &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
            &mut unique_artifacts,
        )?;
        let up = require_all_ten_plan_projection(
            wave,
            "up",
            &format!("{prefix}.up_proj.weight"),
            &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
            &mut unique_artifacts,
        )?;
        let down = require_all_ten_plan_projection(
            wave,
            "down",
            &format!("{prefix}.down_proj.weight"),
            &[QWEN80_HIDDEN, QWEN80_MOE_INTERMEDIATE],
            &mut unique_artifacts,
        )?;
        parsed_waves.push(Qwen80AllTenRoutePlanWave {
            expert,
            normalized_weight,
            gate,
            up,
            down,
        });
    }
    Ok(Qwen80AllTenRoutedExpertPlan {
        manifest_document_sha256: authority.manifest_document_sha256.to_owned(),
        plan_document_sha256: authority.plan_document_sha256.to_owned(),
        manifest_seal_sha256: plan.manifest_seal_sha256.clone(),
        source_revision: plan.source_revision.clone(),
        layer: layer.layer,
        route,
        waves: parsed_waves,
    })
}

/// Parse the source-token-specific descriptor nested in a separately sealed
/// authority.  This deliberately shares only the exact direct-packed wave
/// geometry with the historical parser; its route provenance is token-1 plus
/// zero L0 state and a strict-Metal prefix seal, never the fixture router.
fn require_source_token_all_ten_route_plan(
    plan: &Qwen80CompleteHybridDecoderPlan,
    layer: &Qwen80HybridDecoderLayerPlan,
    authority: &Qwen80SourceTokenAllTenRoutedExpertPlanAuthority<'_>,
    descriptor: &Value,
) -> Result<Qwen80AllTenRoutedExpertPlan> {
    for (value, label) in [
        (
            authority.manifest_document_sha256,
            "source-token all-ten manifest document SHA-256",
        ),
        (
            authority.plan_authority_document_sha256,
            "source-token all-ten authority document SHA-256",
        ),
        (
            authority.admission_receipt_seal_sha256,
            "source-token all-ten admission receipt seal",
        ),
        (
            authority.first_residual_outer_receipt_seal_sha256,
            "source-token all-ten first-residual outer seal",
        ),
    ] {
        require_canonical_sha256(value, label)?;
    }
    require_ledger_string(
        descriptor,
        "schema",
        QWEN80_SOURCE_TOKEN_ALL_TEN_ROUTE_PLAN_SCHEMA,
    )?;
    require_ledger_string(
        descriptor,
        "status",
        QWEN80_SOURCE_TOKEN_ALL_TEN_ROUTE_PLAN_STATUS,
    )?;
    require_ledger_string(descriptor, "model_id", QWEN80_MODEL_ID)?;
    require_ledger_string(descriptor, "model_key", "qwen80")?;
    require_ledger_string(descriptor, "source_repository", QWEN80_REPOSITORY)?;
    require_ledger_string(descriptor, "source_revision", &plan.source_revision)?;
    if ledger_usize(descriptor, "layer")? != layer.layer {
        return Err(model_error(format!(
            "source-token all-ten route plan belongs to layer {}, not requested layer {}",
            ledger_usize(descriptor, "layer")?,
            layer.layer
        )));
    }
    for field in [
        "route_execution_performed",
        "route_combine_performed",
        "shared_expert_performed",
        "residual_combine_performed",
        "metal_device_or_dispatch_performed",
        "model_execution_performed",
        "hcli_execution_performed",
        "tps_or_tg_measurement_performed",
        "complete_layer_or_decoder_claim_earned",
    ] {
        if ledger_bool(descriptor, field)? {
            return Err(model_error(format!(
                "source-token all-ten route plan must remain descriptor-only; {field:?} was true"
            )));
        }
    }

    let input = ledger_object(descriptor, "source_input_provenance")?;
    if ledger_usize(input, "source_token_id")? != 1
        || !ledger_bool(input, "same_input_state_identity_required")?
    {
        return Err(model_error(
            "source-token all-ten route plan lost token-1/zero-state identity",
        ));
    }
    require_ledger_sha256(
        input,
        "prefix_outer_receipt_seal_sha256",
        authority.first_residual_outer_receipt_seal_sha256,
    )?;
    for field in [
        "input_hidden_f32le_sha256",
        "cpu_first_residual_f32le_sha256",
        "strict_metal_prefix_first_residual_sha256",
        "zero_conv_state_f32le_sha256",
        "zero_recurrent_state_f32le_sha256",
    ] {
        ledger_sha256(input, field)?;
    }
    let prefix_evidence = ledger_object(input, "prefix_outer_receipt")?;
    ledger_sha256(prefix_evidence, "sha256")?;
    let baseline_evidence = ledger_object(input, "cpu_baseline_receipt")?;
    ledger_sha256(baseline_evidence, "sha256")?;

    let manifest = ledger_object(descriptor, "manifest_descriptor_inventory")?;
    require_ledger_string(
        manifest,
        "manifest_schema",
        "hawking.ascension.qwen80_complete_binary_gravity.v1",
    )?;
    require_ledger_sha256(
        manifest,
        "inventory_document_sha256",
        authority.manifest_document_sha256,
    )?;
    require_ledger_sha256(manifest, "manifest_seal_sha256", &plan.manifest_seal_sha256)?;
    if ledger_usize(manifest, "declared_tensor_count")? != QWEN80_COMPLETE_BINARY_TENSORS
        || ledger_usize(manifest, "resolved_route_tensor_count")? != QWEN80_TOP_K * 3
        || ledger_bool(manifest, "payload_opened_by_this_plan")?
    {
        return Err(model_error(
            "source-token all-ten route plan manifest inventory is incomplete or opens payloads",
        ));
    }

    let router = ledger_object(descriptor, "source_token_router_evidence")?;
    if !ledger_bool(
        router,
        "derived_from_direct_packed_source_token_l0_cpu_oracle",
    )? || !ledger_bool(router, "router_component_only")?
    {
        return Err(model_error(
            "source-token all-ten route plan lacks the direct-packed CPU router provenance",
        ));
    }
    ledger_sha256(router, "post_attention_normalized_hidden_f32le_sha256")?;
    ledger_sha256(router, "router_logits_f32le_sha256")?;
    let route = route_from_ledger_arrays(
        router,
        "source_stable_route_ids",
        "source_stable_normalized_weights",
        "source-token all-ten route plan source route",
    )?;

    let rawls = ledger_object(descriptor, "rawls_real_all_ten_provenance_gate")?;
    require_ledger_string(rawls, "schema", QWEN80_ALL_TEN_ROUTE_PLAN_GATE_SCHEMA)?;
    if !ledger_bool(rawls, "all_ten_source_bindings_complete")?
        || ledger_usize(rawls, "expected_layer")? != layer.layer
    {
        return Err(model_error(
            "source-token all-ten route plan Rawls gate is incomplete or for another layer",
        ));
    }
    let rawls_route = route_from_ledger_arrays(
        rawls,
        "route_order",
        "normalized_weights",
        "source-token all-ten route plan Rawls route",
    )?;
    require_same_route(
        &rawls_route,
        &route,
        "source-token all-ten route plan Rawls route",
    )?;
    let indices = ledger_array(rawls, "deterministic_wave_indices")?;
    if indices.len() != QWEN80_TOP_K
        || indices
            .iter()
            .enumerate()
            .any(|(index, value)| value.as_u64() != Some(index as u64))
    {
        return Err(model_error(
            "source-token all-ten route plan has non-deterministic wave indices",
        ));
    }
    for field in [
        "execution_receipt_required_for_each_wave",
        "direct_packed_execution_required_for_each_wave",
        "source_bound_input_required_for_each_wave",
        "route_combine_receipt_required_separately",
        "shared_expert_receipt_required_separately",
        "first_and_second_residual_receipts_required_separately",
        "rejects_tensor_substitution",
        "rejects_route_reorder",
        "rejects_duplicate_experts",
        "rejects_missing_tensor_or_weight",
    ] {
        if !ledger_bool(rawls, field)? {
            return Err(model_error(format!(
                "source-token all-ten Rawls gate omitted required guard {field:?}"
            )));
        }
    }

    let waves = ledger_array(descriptor, "deterministic_waves")?;
    if waves.len() != QWEN80_TOP_K {
        return Err(model_error(format!(
            "source-token all-ten plan has {} waves, expected {QWEN80_TOP_K}",
            waves.len()
        )));
    }
    let mut unique_artifacts = HashSet::with_capacity(QWEN80_TOP_K * 3);
    let mut parsed_waves = Vec::with_capacity(QWEN80_TOP_K);
    for (index, wave) in waves.iter().enumerate() {
        let expert = route.ids[index];
        if ledger_usize(wave, "wave_index")? != index
            || ledger_usize(wave, "layer")? != layer.layer
            || ledger_usize(wave, "expert_id")? != usize::from(expert)
        {
            return Err(model_error(format!(
                "source-token all-ten route plan wave {index} does not bind its exact source route"
            )));
        }
        let normalized_weight = ledger_f32(wave, "normalized_weight")?;
        if (normalized_weight - route.weights[index]).abs() > 1.0e-6 {
            return Err(model_error(format!(
                "source-token all-ten route plan wave {index} normalized weight differs from source route"
            )));
        }
        let weight_bits = ledger_string(wave, "normalized_weight_bits_hex")?;
        if weight_bits.len() != 18
            || !weight_bits.starts_with("0x")
            || !weight_bits[2..]
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(model_error(format!(
                "source-token all-ten route plan wave {index} lacks source weight bits"
            )));
        }
        let operation_order = ledger_array(wave, "fixed_operation_order")?;
        const EXPECTED_ROUTE_OPERATION_ORDER: [&str; 5] = [
            "gate_proj [512,2048]",
            "up_proj [512,2048]",
            "SiLU(gate) * up [512]",
            "down_proj [2048,512]",
            "apply this route's source-normalized weight [2048]",
        ];
        if operation_order.len() != EXPECTED_ROUTE_OPERATION_ORDER.len()
            || operation_order
                .iter()
                .zip(EXPECTED_ROUTE_OPERATION_ORDER)
                .any(|(observed, expected)| observed.as_str() != Some(expected))
        {
            return Err(model_error(format!(
                "source-token all-ten route plan wave {index} has non-source operation order"
            )));
        }
        require_ledger_string(
            wave,
            "route_execution_status",
            "NOT_EXECUTED_SOURCE_TOKEN_BOUND_PLAN_ONLY",
        )?;
        if ledger_bool(wave, "route_delta_materialized")?
            || ledger_bool(wave, "route_weight_applied")?
        {
            return Err(model_error(format!(
                "source-token all-ten route plan wave {index} is not descriptor-only"
            )));
        }
        let prefix = format!("model.layers.{}.mlp.experts.{expert}", layer.layer);
        let gate = require_all_ten_plan_projection(
            wave,
            "gate",
            &format!("{prefix}.gate_proj.weight"),
            &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
            &mut unique_artifacts,
        )?;
        let up = require_all_ten_plan_projection(
            wave,
            "up",
            &format!("{prefix}.up_proj.weight"),
            &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
            &mut unique_artifacts,
        )?;
        let down = require_all_ten_plan_projection(
            wave,
            "down",
            &format!("{prefix}.down_proj.weight"),
            &[QWEN80_HIDDEN, QWEN80_MOE_INTERMEDIATE],
            &mut unique_artifacts,
        )?;
        parsed_waves.push(Qwen80AllTenRoutePlanWave {
            expert,
            normalized_weight,
            gate,
            up,
            down,
        });
    }
    if unique_artifacts.len() != QWEN80_TOP_K * 3 {
        return Err(model_error(
            "source-token all-ten route plan lacks thirty unique direct-packed bodies",
        ));
    }
    Ok(Qwen80AllTenRoutedExpertPlan {
        manifest_document_sha256: authority.manifest_document_sha256.to_owned(),
        plan_document_sha256: authority.plan_authority_document_sha256.to_owned(),
        manifest_seal_sha256: plan.manifest_seal_sha256.clone(),
        source_revision: plan.source_revision.clone(),
        layer: layer.layer,
        route,
        waves: parsed_waves,
    })
}

fn require_first_residual_component(
    plan: &Qwen80CompleteHybridDecoderPlan,
    layer: &Qwen80HybridDecoderLayerPlan,
    admission_receipt_seal_sha256: &str,
    manifest_document_sha256: &str,
    receipt: &Value,
) -> Result<String> {
    require_ledger_string(receipt, "schema", QWEN80_FIRST_RESIDUAL_COMPONENT_SCHEMA)?;
    require_ledger_string(receipt, "status", QWEN80_FIRST_RESIDUAL_COMPONENT_STATUS)?;
    require_qwen80_moe_component_authority(
        receipt,
        plan,
        layer,
        admission_receipt_seal_sha256,
        manifest_document_sha256,
        "first-residual component",
    )?;
    let metal = require_qwen80_moe_component_metal(receipt, "first-residual component")?;
    if !ledger_bool(receipt, "first_residual_performed")? {
        return Err(model_error(
            "first-residual component did not record the residual add",
        ));
    }
    if receipt.get("first_residual_output").is_some() {
        ledger_vector_sha256(
            receipt,
            "first_residual_output",
            QWEN80_HIDDEN,
            "first-residual",
        )
    } else {
        let device = ledger_object(metal, "device_intermediates")?;
        ledger_vector_sha256(device, "first_residual", QWEN80_HIDDEN, "first-residual")
    }
}

fn require_router_component(
    plan: &Qwen80CompleteHybridDecoderPlan,
    layer: &Qwen80HybridDecoderLayerPlan,
    inputs: &Qwen80RealMoeBoundaryInputs<'_>,
    first_residual_sha256: &str,
) -> Result<(Qwen80RouteSelection, String)> {
    let receipt = inputs.router_receipt;
    require_ledger_string(receipt, "schema", QWEN80_REAL_MOE_ROUTER_COMPONENT_SCHEMA)?;
    require_ledger_string(receipt, "status", QWEN80_REAL_MOE_ROUTER_COMPONENT_STATUS)?;
    let binding = require_qwen80_moe_component_authority(
        receipt,
        plan,
        layer,
        inputs.admission_receipt_seal_sha256,
        inputs.manifest_document_sha256,
        "postnorm/router component",
    )?;
    require_exact_moe_tensor_name(
        binding,
        "post_attention_norm",
        &layer.post_attention_layernorm.name,
        "postnorm/router component",
    )?;
    require_exact_moe_tensor_name(
        binding,
        "router_gate",
        &layer.moe.router.name,
        "postnorm/router component",
    )?;
    let input = ledger_object(receipt, "input_provenance")?;
    require_ledger_sha256(input, "first_residual_output_sha256", first_residual_sha256)?;
    let metal = require_qwen80_moe_component_metal(receipt, "postnorm/router component")?;
    let device = ledger_object(metal, "device_intermediates")?;
    let route = route_from_ledger_arrays(
        device,
        "route_ids",
        "renormalized_route_weights",
        "postnorm/router device route",
    )?;
    let normalized = ledger_vector_sha256(
        device,
        "normalized_hidden",
        QWEN80_HIDDEN,
        "postnorm/router",
    )?;
    Ok((route, normalized))
}

fn require_routed_expert_component(
    plan: &Qwen80CompleteHybridDecoderPlan,
    layer: &Qwen80HybridDecoderLayerPlan,
    inputs: &Qwen80RealMoeBoundaryInputs<'_>,
    route: &Qwen80RouteSelection,
    all_ten_route_plan: &Qwen80AllTenRoutedExpertPlan,
    route_index: usize,
    first_residual_sha256: &str,
    normalized_hidden_sha256: &str,
    receipt: &Value,
) -> Result<String> {
    require_ledger_string(
        receipt,
        "schema",
        QWEN80_REAL_ROUTED_EXPERT_COMPONENT_SCHEMA,
    )?;
    require_ledger_string(
        receipt,
        "status",
        QWEN80_REAL_ROUTED_EXPERT_COMPONENT_STATUS,
    )?;
    let binding = require_qwen80_moe_component_authority(
        receipt,
        plan,
        layer,
        inputs.admission_receipt_seal_sha256,
        inputs.manifest_document_sha256,
        "routed-expert component",
    )?;
    let planned_wave = all_ten_route_plan.waves.get(route_index).ok_or_else(|| {
        model_error(format!(
            "all-ten route plan omitted routed-expert wave {route_index}"
        ))
    })?;
    let expert = route.ids[route_index] as usize;
    if planned_wave.expert as usize != expert
        || (planned_wave.normalized_weight - route.weights[route_index]).abs() > 1.0e-6
    {
        return Err(model_error(format!(
            "all-ten route plan wave {route_index} differs from exact router source route"
        )));
    }
    if ledger_usize(binding, "selected_expert")? != expert
        || ledger_usize(binding, "moe_intermediate")? != QWEN80_MOE_INTERMEDIATE
    {
        return Err(model_error(format!(
            "routed-expert component does not bind source route {route_index} expert {expert}"
        )));
    }
    require_exact_moe_tensor_name(
        binding,
        "post_attention_norm",
        &layer.post_attention_layernorm.name,
        "routed-expert component",
    )?;
    require_exact_moe_tensor_binding(
        binding,
        "expert_gate_proj",
        &planned_wave.gate,
        "routed-expert component",
    )?;
    require_exact_moe_tensor_binding(
        binding,
        "expert_up_proj",
        &planned_wave.up,
        "routed-expert component",
    )?;
    require_exact_moe_tensor_binding(
        binding,
        "expert_down_proj",
        &planned_wave.down,
        "routed-expert component",
    )?;
    let evidence = ledger_object(receipt, "route_evidence")?;
    if ledger_usize(evidence, "selected_route_index")? != route_index
        || ledger_usize(evidence, "selected_expert")? != expert
    {
        return Err(model_error(format!(
            "routed-expert component route index {route_index} does not match its source-selected expert"
        )));
    }
    let selected_weight = ledger_f32(evidence, "selected_normalized_weight_f32")?;
    if (selected_weight - route.weights[route_index]).abs() > 1.0e-6 {
        return Err(model_error(format!(
            "routed-expert component route {route_index} weight {selected_weight} differs from router {}",
            route.weights[route_index]
        )));
    }
    let echoed_route = route_from_ledger_arrays(
        evidence,
        "source_top10_ids",
        "source_top10_renormalized_weights",
        "routed-expert echoed source route",
    )?;
    require_same_route(&echoed_route, route, "routed-expert echoed source route")?;
    require_ledger_sha256(
        evidence,
        "router_receipt_sha256",
        inputs.router_receipt_sha256,
    )?;
    require_ledger_sha256(
        evidence,
        "router_outer_receipt_seal_sha256",
        inputs.router_outer_receipt_seal_sha256,
    )?;
    require_ledger_sha256(
        evidence,
        "all_ten_route_plan_document_sha256",
        inputs.all_ten_route_plan_document_sha256,
    )?;
    require_ledger_sha256(
        evidence,
        "first_residual_output_sha256",
        first_residual_sha256,
    )?;
    require_ledger_sha256(
        evidence,
        "router_normalized_hidden_sha256",
        normalized_hidden_sha256,
    )?;
    let metal = require_qwen80_moe_component_metal(receipt, "routed-expert component")?;
    let device = ledger_object(metal, "device_intermediates")?;
    ledger_vector_sha256(
        device,
        "weighted_one_route_delta",
        QWEN80_HIDDEN,
        "routed-expert",
    )
}

fn require_shared_expert_component(
    plan: &Qwen80CompleteHybridDecoderPlan,
    layer: &Qwen80HybridDecoderLayerPlan,
    inputs: &Qwen80RealMoeBoundaryInputs<'_>,
    first_residual_sha256: &str,
    normalized_hidden_sha256: &str,
) -> Result<String> {
    let receipt = inputs.shared_expert_receipt;
    require_ledger_string(
        receipt,
        "schema",
        QWEN80_REAL_SHARED_EXPERT_COMPONENT_SCHEMA,
    )?;
    require_ledger_string(
        receipt,
        "status",
        QWEN80_REAL_SHARED_EXPERT_COMPONENT_STATUS,
    )?;
    if !ledger_bool(receipt, "shared_expert_only")? {
        return Err(model_error(
            "shared-expert component is not explicitly scoped to the shared expert",
        ));
    }
    let binding = require_qwen80_moe_component_authority(
        receipt,
        plan,
        layer,
        inputs.admission_receipt_seal_sha256,
        inputs.manifest_document_sha256,
        "shared-expert component",
    )?;
    if ledger_usize(binding, "shared_expert_intermediate")? != QWEN80_SHARED_EXPERT_INTERMEDIATE {
        return Err(model_error(
            "shared-expert component has wrong source intermediate geometry",
        ));
    }
    require_exact_moe_tensor_name(
        binding,
        "post_attention_norm",
        &layer.post_attention_layernorm.name,
        "shared-expert component",
    )?;
    for (field, expected) in [
        ("shared_gate_proj", &layer.moe.shared_gate_proj.name),
        ("shared_up_proj", &layer.moe.shared_up_proj.name),
        ("shared_down_proj", &layer.moe.shared_down_proj.name),
        ("shared_expert_gate", &layer.moe.shared_expert_gate.name),
    ] {
        require_exact_moe_tensor_name(binding, field, expected, "shared-expert component")?;
    }
    let input = ledger_object(receipt, "input_provenance")?;
    require_ledger_sha256(input, "first_residual_output_sha256", first_residual_sha256)?;
    require_ledger_sha256(
        input,
        "router_normalized_hidden_sha256",
        normalized_hidden_sha256,
    )?;
    let metal = require_qwen80_moe_component_metal(receipt, "shared-expert component")?;
    let device = ledger_object(metal, "device_intermediates")?;
    ledger_vector_sha256(device, "gated_shared", QWEN80_HIDDEN, "shared-expert")
}

fn require_combine_component(
    plan: &Qwen80CompleteHybridDecoderPlan,
    layer: &Qwen80HybridDecoderLayerPlan,
    inputs: &Qwen80RealMoeBoundaryInputs<'_>,
    route: &Qwen80RouteSelection,
    first_residual_sha256: &str,
    normalized_hidden_sha256: &str,
    routed_delta_sha256: &[String],
    gated_shared_sha256: &str,
) -> Result<String> {
    let receipt = inputs.combine_receipt;
    require_ledger_string(receipt, "schema", QWEN80_REAL_MOE_COMBINE_COMPONENT_SCHEMA)?;
    require_ledger_string(receipt, "status", QWEN80_REAL_MOE_COMBINE_COMPONENT_STATUS)?;
    if ledger_bool(receipt, "materialized_source_route_shaped_fixture_only")?
        || !ledger_bool(receipt, "routed_expert_aggregation_performed")?
        || !ledger_bool(receipt, "shared_expert_add_performed")?
        || !ledger_bool(receipt, "second_residual_performed")?
    {
        return Err(model_error(
            "MoE combine receipt is a fixture or lacks all physical aggregate/shared/residual stages",
        ));
    }
    require_qwen80_moe_component_authority(
        receipt,
        plan,
        layer,
        inputs.admission_receipt_seal_sha256,
        inputs.manifest_document_sha256,
        "MoE combine component",
    )?;
    let combine_inputs = ledger_object(receipt, "combine_inputs")?;
    require_ledger_sha256(
        combine_inputs,
        "all_ten_route_plan_document_sha256",
        inputs.all_ten_route_plan_document_sha256,
    )?;
    let echoed_route = route_from_ledger_arrays(
        combine_inputs,
        "source_top10_ids",
        "source_top10_renormalized_weights",
        "MoE combine source route",
    )?;
    require_same_route(&echoed_route, route, "MoE combine source route")?;
    require_ledger_sha256(
        combine_inputs,
        "first_residual_output_sha256",
        first_residual_sha256,
    )?;
    require_ledger_sha256(
        combine_inputs,
        "router_normalized_hidden_sha256",
        normalized_hidden_sha256,
    )?;
    require_ledger_sha256(combine_inputs, "gated_shared_sha256", gated_shared_sha256)?;
    let routed = ledger_array(combine_inputs, "routed_weighted_delta_sha256")?;
    if routed.len() != QWEN80_TOP_K || routed_delta_sha256.len() != QWEN80_TOP_K {
        return Err(model_error(format!(
            "MoE combine requires exactly {QWEN80_TOP_K} routed output hashes"
        )));
    }
    for (index, (observed, expected)) in routed.iter().zip(routed_delta_sha256).enumerate() {
        let observed = observed.as_str().ok_or_else(|| {
            model_error(format!(
                "MoE combine routed output hash {index} is not a string"
            ))
        })?;
        require_canonical_sha256(observed, "MoE combine routed output hash")?;
        if observed != expected {
            return Err(model_error(format!(
                "MoE combine routed output hash {index} does not match its route witness"
            )));
        }
    }
    let metal = require_qwen80_moe_component_metal(receipt, "MoE combine component")?;
    let device = ledger_object(metal, "device_intermediates")?;
    ledger_vector_sha256(device, "second_residual", QWEN80_HIDDEN, "MoE combine")
}

impl Qwen80CompleteHybridDecoderPlan {
    /// Validate a future, physically complete layer-MoE provenance boundary.
    ///
    /// This is CPU-only receipt validation.  It deliberately rejects the
    /// current expert-65 witness and materialized combine fixture, and it
    /// cannot make the scheduler, a component executor, or the 48-layer graph
    /// claim native runtime completion.
    pub fn require_real_moe_boundary(
        &self,
        layer_index: usize,
        inputs: &Qwen80RealMoeBoundaryInputs<'_>,
    ) -> Result<Qwen80VerifiedRealMoeBoundary> {
        if inputs.routed_expert_receipts.len() != QWEN80_TOP_K {
            return Err(model_error(format!(
                "real Qwen80 MoE boundary requires exactly {QWEN80_TOP_K} routed expert receipts, found {}",
                inputs.routed_expert_receipts.len()
            )));
        }
        require_canonical_sha256(inputs.router_receipt_sha256, "router inner receipt SHA-256")?;
        require_canonical_sha256(
            inputs.manifest_document_sha256,
            "complete-manifest document SHA-256",
        )?;
        require_canonical_sha256(
            inputs.router_outer_receipt_sha256,
            "router outer receipt document SHA-256",
        )?;
        require_canonical_sha256(
            inputs.router_outer_receipt_seal_sha256,
            "router outer receipt seal",
        )?;
        let layer = self.layers.get(layer_index).ok_or_else(|| {
            model_error(format!(
                "real MoE boundary layer {layer_index} is outside hybrid plan"
            ))
        })?;
        let all_ten_authority = Qwen80AllTenRoutedExpertPlanAuthority {
            manifest_document_sha256: inputs.manifest_document_sha256,
            plan_document_sha256: inputs.all_ten_route_plan_document_sha256,
            router_receipt_sha256: inputs.router_receipt_sha256,
            router_outer_receipt_sha256: inputs.router_outer_receipt_sha256,
            router_outer_receipt_seal_sha256: inputs.router_outer_receipt_seal_sha256,
        };
        let all_ten_route_plan = self.bind_all_ten_routed_expert_plan(
            layer_index,
            &all_ten_authority,
            inputs.all_ten_route_plan,
        )?;
        let first_residual_sha256 = require_first_residual_component(
            self,
            layer,
            inputs.admission_receipt_seal_sha256,
            inputs.manifest_document_sha256,
            inputs.first_residual_receipt,
        )?;
        let (route, post_attention_normalized_hidden_sha256) =
            require_router_component(self, layer, inputs, &first_residual_sha256)?;
        require_same_route(
            &route,
            &all_ten_route_plan.route,
            "postnorm/router route versus all-ten descriptor",
        )?;
        let mut routed_weighted_delta_sha256 = Vec::with_capacity(QWEN80_TOP_K);
        for (route_index, receipt) in inputs.routed_expert_receipts.iter().enumerate() {
            routed_weighted_delta_sha256.push(require_routed_expert_component(
                self,
                layer,
                inputs,
                &route,
                &all_ten_route_plan,
                route_index,
                &first_residual_sha256,
                &post_attention_normalized_hidden_sha256,
                receipt,
            )?);
        }
        let gated_shared_sha256 = require_shared_expert_component(
            self,
            layer,
            inputs,
            &first_residual_sha256,
            &post_attention_normalized_hidden_sha256,
        )?;
        let second_residual_sha256 = require_combine_component(
            self,
            layer,
            inputs,
            &route,
            &first_residual_sha256,
            &post_attention_normalized_hidden_sha256,
            &routed_weighted_delta_sha256,
            &gated_shared_sha256,
        )?;
        Ok(Qwen80VerifiedRealMoeBoundary {
            layer: layer.layer,
            manifest_document_sha256: inputs.manifest_document_sha256.to_owned(),
            all_ten_route_plan_document_sha256: inputs
                .all_ten_route_plan_document_sha256
                .to_owned(),
            route,
            first_residual_sha256,
            post_attention_normalized_hidden_sha256,
            routed_weighted_delta_sha256,
            gated_shared_sha256,
            second_residual_sha256,
        })
    }
}

/// Existing component receipts that may seed a bounded hybrid stage runner.
/// Both component receipts are necessary but deliberately insufficient: the
/// type records their exact artifact/layer boundary so a caller cannot turn a
/// layer-0 DeltaNet fixture plus layer-3 GQA fixture into a false full-runtime
/// pass.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80BoundedHybridComponentEvidence {
    pub manifest_seal_sha256: String,
    pub source_revision: String,
    pub deltanet_layer: usize,
    pub deltanet_route: Qwen80RouteSelection,
    pub deltanet_selected_expert: usize,
    pub attention_layer: usize,
    pub attention_fixture_positions: Vec<usize>,
    pub attention_metal_dispatches: usize,
}

impl Qwen80BoundedHybridComponentEvidence {
    /// Parse and cross-bind the existing direct-packed layer-0 DeltaNet and
    /// layer-3 two-token GQA component ledgers.  The first argument is the
    /// `result` object of the strict Qwen80 direct-linear-stage handoff; the
    /// second is the durable inner GQA `receipt.json` object.  No provenance
    /// is relaxed and no component is promoted beyond its ledger boundary.
    pub fn from_component_ledgers(
        plan: &Qwen80CompleteHybridDecoderPlan,
        direct_linear_result: &Value,
        layer3_gqa_receipt: &Value,
    ) -> Result<Self> {
        require_ledger_string(
            direct_linear_result,
            "status",
            QWEN80_DIRECT_PACKED_LINEAR_COMPONENT_STATUS,
        )?;
        require_ledger_string(
            direct_linear_result,
            "manifest_seal_sha256",
            &plan.manifest_seal_sha256,
        )?;
        let model = ledger_object(direct_linear_result, "model")?;
        require_ledger_string(model, "revision", &plan.source_revision)?;
        let native_state = ledger_object(direct_linear_result, "native_state")?;
        require_ledger_string(
            native_state,
            "status",
            QWEN80_DIRECT_PACKED_LINEAR_COMPONENT_STATUS,
        )?;
        let delta = ledger_object(native_state, "stage")?;
        if ledger_usize(delta, "layer")? != 0
            || ledger_string(delta, "layer_kind")?
                != Qwen80LayerKind::LinearAttention.as_source_name()
        {
            return Err(model_error(
                "direct DeltaNet component ledger is not source-bound to Qwen80 layer 0",
            ));
        }
        let layer_zero = plan.layers.first().ok_or_else(|| {
            model_error("hybrid plan has no layer 0 for direct DeltaNet component binding")
        })?;
        let delta_mixer = match &layer_zero.mixer {
            Qwen80HybridMixerBindings::LinearDeltaNet(mixer) => mixer,
            Qwen80HybridMixerBindings::FullAttention(_) => {
                return Err(model_error("hybrid plan layer 0 is not DeltaNet"));
            }
        };
        require_ledger_string(
            delta,
            "direct_packed_input_projection_tensor",
            &delta_mixer.in_proj_qkvz.name,
        )?;
        require_ledger_string(
            delta,
            "direct_packed_ba_projection_tensor",
            &delta_mixer.in_proj_ba.name,
        )?;
        require_ledger_string(
            delta,
            "direct_packed_router_tensor",
            &layer_zero.moe.router.name,
        )?;
        let route_ids = ledger_array(delta, "route_ids")?;
        let route_weights = ledger_array(delta, "route_weights")?;
        if route_ids.len() != QWEN80_TOP_K || route_weights.len() != QWEN80_TOP_K {
            return Err(model_error(format!(
                "direct DeltaNet component route has ids={} weights={}, expected {QWEN80_TOP_K}",
                route_ids.len(),
                route_weights.len()
            )));
        }
        let mut ids = [0u16; QWEN80_TOP_K];
        let mut weights = [0.0f32; QWEN80_TOP_K];
        for index in 0..QWEN80_TOP_K {
            let id = route_ids[index]
                .as_u64()
                .and_then(|value| u16::try_from(value).ok())
                .ok_or_else(|| {
                    model_error(format!("direct DeltaNet route id {index} is outside u16"))
                })?;
            let weight = route_weights[index]
                .as_f64()
                .filter(|value| value.is_finite())
                .map(|value| value as f32)
                .ok_or_else(|| {
                    model_error(format!(
                        "direct DeltaNet route weight {index} is non-finite"
                    ))
                })?;
            ids[index] = id;
            weights[index] = weight;
        }
        let deltanet_route = Qwen80RouteSelection { ids, weights };
        deltanet_route.validate()?;
        let deltanet_selected_expert = ledger_usize(delta, "selected_expert")?;
        if deltanet_selected_expert != deltanet_route.ids[0] as usize {
            return Err(model_error(
                "direct DeltaNet component selected expert is not route top-1",
            ));
        }
        let expected_selected =
            format!("model.layers.0.mlp.experts.{deltanet_selected_expert}.gate_proj.weight");
        require_ledger_string(
            delta,
            "direct_packed_selected_expert_gate_tensor",
            &expected_selected,
        )?;

        require_ledger_string(
            layer3_gqa_receipt,
            "status",
            QWEN80_DIRECT_PACKED_LAYER3_GQA_COMPONENT_STATUS,
        )?;
        let attention_artifact = ledger_object(layer3_gqa_receipt, "artifact")?;
        require_ledger_string(
            attention_artifact,
            "manifest_seal_sha256",
            &plan.manifest_seal_sha256,
        )?;
        require_ledger_string(attention_artifact, "source_revision", &plan.source_revision)?;
        let attention = ledger_object(layer3_gqa_receipt, "source_bound_layer")?;
        if ledger_usize(attention, "layer")? != 3
            || ledger_string(attention, "kind")? != Qwen80LayerKind::FullAttention.as_source_name()
        {
            return Err(model_error(
                "GQA component ledger is not source-bound to Qwen80 full-attention layer 3",
            ));
        }
        let layer_three = plan
            .layers
            .get(3)
            .ok_or_else(|| model_error("hybrid plan has no layer 3 for GQA component binding"))?;
        let attention_mixer = match &layer_three.mixer {
            Qwen80HybridMixerBindings::FullAttention(mixer) => mixer,
            Qwen80HybridMixerBindings::LinearDeltaNet(_) => {
                return Err(model_error("hybrid plan layer 3 is not full attention"));
            }
        };
        let attention_tensors = ledger_object(attention, "tensors")?;
        for (field, expected) in [
            ("q_proj", &attention_mixer.q_proj.name),
            ("k_proj", &attention_mixer.k_proj.name),
            ("v_proj", &attention_mixer.v_proj.name),
            ("o_proj", &attention_mixer.o_proj.name),
            ("q_norm", &attention_mixer.q_norm.name),
            ("k_norm", &attention_mixer.k_norm.name),
        ] {
            require_ledger_string(attention_tensors, field, expected)?;
        }
        let geometry = ledger_object(attention, "geometry")?;
        let positions = ledger_array(geometry, "fixture_positions")?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        model_error(format!(
                            "GQA fixture position {index} is not an unsigned integer"
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        if positions.as_slice() != [0, 1] {
            return Err(model_error(format!(
                "GQA component fixture positions {positions:?} are not the required two-token [0, 1]"
            )));
        }
        let metal = ledger_object(layer3_gqa_receipt, "metal_execution")?;
        if metal.get("performed").and_then(Value::as_bool) != Some(true) {
            return Err(model_error(
                "GQA component ledger did not record real Metal execution",
            ));
        }
        let attention_metal_dispatches = ledger_usize(metal, "compute_dispatches")?;
        if attention_metal_dispatches == 0 {
            return Err(model_error(
                "GQA component ledger has zero Metal dispatches",
            ));
        }
        Ok(Self {
            manifest_seal_sha256: plan.manifest_seal_sha256.clone(),
            source_revision: plan.source_revision.clone(),
            deltanet_layer: 0,
            deltanet_route,
            deltanet_selected_expert,
            attention_layer: 3,
            attention_fixture_positions: positions,
            attention_metal_dispatches,
        })
    }
}

/// A bounded stage runner that wires the two existing component ledgers into
/// the all-layer bridge.  Its executor can be a deterministic CPU oracle for
/// test coverage, or a future direct-packed Metal executor.  It deliberately
/// does not infer that either choice makes the final 48-layer token native.
pub struct Qwen80BoundedHybridStageRunner<E> {
    decoder: Qwen80CompleteHybridDecoder,
    bridge: Qwen80ArtifactBoundPerLayerBackendBridge<E>,
    component_evidence: Qwen80BoundedHybridComponentEvidence,
}

impl<E: Qwen80HybridPerLayerComponentExecutor> Qwen80BoundedHybridStageRunner<E> {
    pub fn from_admitted_catalog_with_component_ledgers(
        catalog: Qwen80CompleteArtifactCatalog,
        max_seq_len: usize,
        direct_linear_result: &Value,
        layer3_gqa_receipt: &Value,
        executor: E,
    ) -> Result<Self> {
        let plan = catalog.complete_hybrid_decoder_plan(max_seq_len)?;
        let component_evidence = Qwen80BoundedHybridComponentEvidence::from_component_ledgers(
            &plan,
            direct_linear_result,
            layer3_gqa_receipt,
        )?;
        if max_seq_len > component_evidence.attention_fixture_positions.len() {
            return Err(model_error(format!(
                "bounded hybrid stage runner max_seq_len={max_seq_len} exceeds the admitted layer-3 GQA fixture coverage of {} positions",
                component_evidence.attention_fixture_positions.len()
            )));
        }
        let bridge = Qwen80ArtifactBoundPerLayerBackendBridge::new(&plan, executor)?;
        let decoder = Qwen80CompleteHybridDecoder::new(catalog, plan)?;
        Ok(Self {
            decoder,
            bridge,
            component_evidence,
        })
    }

    pub fn component_evidence(&self) -> &Qwen80BoundedHybridComponentEvidence {
        &self.component_evidence
    }

    pub fn next_position(&self) -> usize {
        self.decoder.next_position()
    }

    /// Execute a scheduler/control invocation through the bridge.  This name
    /// intentionally says `bounded_control`: callers must separately prove a
    /// direct-packed production executor and all-layer numerical behavior
    /// before treating this as a native token execution.
    pub fn execute_bounded_control_token(
        &mut self,
        input_token_id: u32,
    ) -> Result<Qwen80HybridScheduledToken> {
        self.decoder.execute_one(&mut self.bridge, input_token_id)
    }

    pub fn into_executor(self) -> E {
        self.bridge.into_inner()
    }
}

/// One source-visible segment inside a per-key-head QKVZ projection block.
/// Qwen3-Next stores Q, K, V, and Z interleaved per *key* head; V and Z each
/// contain two value heads.  A device executor must use this ordering rather
/// than treating the 12,288 projection rows as four global contiguous blocks.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum Qwen80LinearQkvzSegment {
    Query,
    Key,
    Value,
    Z,
}

/// One source-visible segment inside a per-key-head BA projection block.
/// The two beta logits precede the two decay inputs: `[b0, b1, a0, a1]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum Qwen80LinearBaSegment {
    BetaLogit,
    DecayInput,
}

/// Exact CPU/device indexing geometry for one Qwen3-Next Gated DeltaNet
/// decode layer.  The methods below are deliberately small enough to become
/// generated constants in a future Metal dispatch, while retaining checks in
/// the CPU oracle and launch-time device contract.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen80CanonicalLinearDeltaNetLayout {
    pub hidden_elements: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub value_heads_per_key_head: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub qkvz_rows_per_key_head: usize,
    pub qkvz_query_offset_rows: usize,
    pub qkvz_key_offset_rows: usize,
    pub qkvz_value_offset_rows: usize,
    pub qkvz_z_offset_rows: usize,
    pub ba_rows_per_key_head: usize,
    pub ba_beta_offset_rows: usize,
    pub ba_decay_offset_rows: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
    pub conv_state_tokens: usize,
    pub recurrent_state_head_stride: usize,
    pub recurrent_state_key_stride: usize,
}

impl Qwen80CanonicalLinearDeltaNetLayout {
    fn source_exact() -> Self {
        let value_heads_per_key_head = QWEN80_LINEAR_VALUE_HEADS / QWEN80_LINEAR_KEY_HEADS;
        let value_rows_per_key_head = value_heads_per_key_head * QWEN80_LINEAR_VALUE_HEAD_DIM;
        Self {
            hidden_elements: QWEN80_HIDDEN,
            key_heads: QWEN80_LINEAR_KEY_HEADS,
            value_heads: QWEN80_LINEAR_VALUE_HEADS,
            value_heads_per_key_head,
            key_head_dim: QWEN80_LINEAR_KEY_HEAD_DIM,
            value_head_dim: QWEN80_LINEAR_VALUE_HEAD_DIM,
            qkvz_rows_per_key_head: QWEN80_LINEAR_KEY_HEAD_DIM * 2 + value_rows_per_key_head * 2,
            qkvz_query_offset_rows: 0,
            qkvz_key_offset_rows: QWEN80_LINEAR_KEY_HEAD_DIM,
            qkvz_value_offset_rows: QWEN80_LINEAR_KEY_HEAD_DIM * 2,
            qkvz_z_offset_rows: QWEN80_LINEAR_KEY_HEAD_DIM * 2 + value_rows_per_key_head,
            ba_rows_per_key_head: value_heads_per_key_head * 2,
            ba_beta_offset_rows: 0,
            ba_decay_offset_rows: value_heads_per_key_head,
            conv_channels: QWEN80_LINEAR_KEY_HEADS * QWEN80_LINEAR_KEY_HEAD_DIM * 2
                + QWEN80_LINEAR_VALUE_HEADS * QWEN80_LINEAR_VALUE_HEAD_DIM,
            conv_kernel: QWEN80_LINEAR_CONV_KERNEL,
            conv_state_tokens: QWEN80_LINEAR_CONV_KERNEL - 1,
            recurrent_state_head_stride: QWEN80_LINEAR_KEY_HEAD_DIM * QWEN80_LINEAR_VALUE_HEAD_DIM,
            recurrent_state_key_stride: QWEN80_LINEAR_VALUE_HEAD_DIM,
        }
    }

    pub fn qkvz_projection_elements(&self) -> Result<usize> {
        self.key_heads
            .checked_mul(self.qkvz_rows_per_key_head)
            .ok_or_else(|| model_error("Qwen80 QKVZ projection geometry overflowed"))
    }

    pub fn ba_projection_elements(&self) -> Result<usize> {
        self.key_heads
            .checked_mul(self.ba_rows_per_key_head)
            .ok_or_else(|| model_error("Qwen80 BA projection geometry overflowed"))
    }

    pub fn key_elements(&self) -> Result<usize> {
        self.key_heads
            .checked_mul(self.key_head_dim)
            .ok_or_else(|| model_error("Qwen80 DeltaNet key geometry overflowed"))
    }

    pub fn value_elements(&self) -> Result<usize> {
        self.value_heads
            .checked_mul(self.value_head_dim)
            .ok_or_else(|| model_error("Qwen80 DeltaNet value geometry overflowed"))
    }

    pub fn conv_state_elements(&self) -> Result<usize> {
        self.conv_channels
            .checked_mul(self.conv_state_tokens)
            .ok_or_else(|| model_error("Qwen80 DeltaNet convolution state geometry overflowed"))
    }

    pub fn recurrent_state_elements(&self) -> Result<usize> {
        self.value_heads
            .checked_mul(self.recurrent_state_head_stride)
            .ok_or_else(|| model_error("Qwen80 DeltaNet recurrent state geometry overflowed"))
    }

    pub fn qkvz_row_offset(
        &self,
        key_head: usize,
        segment: Qwen80LinearQkvzSegment,
    ) -> Result<usize> {
        if key_head >= self.key_heads {
            return Err(model_error(format!(
                "Qwen80 QKVZ key head {key_head} is outside {}",
                self.key_heads
            )));
        }
        let segment_offset = match segment {
            Qwen80LinearQkvzSegment::Query => self.qkvz_query_offset_rows,
            Qwen80LinearQkvzSegment::Key => self.qkvz_key_offset_rows,
            Qwen80LinearQkvzSegment::Value => self.qkvz_value_offset_rows,
            Qwen80LinearQkvzSegment::Z => self.qkvz_z_offset_rows,
        };
        key_head
            .checked_mul(self.qkvz_rows_per_key_head)
            .and_then(|base| base.checked_add(segment_offset))
            .ok_or_else(|| model_error("Qwen80 QKVZ row offset overflowed"))
    }

    pub fn ba_row_offset(&self, key_head: usize, segment: Qwen80LinearBaSegment) -> Result<usize> {
        if key_head >= self.key_heads {
            return Err(model_error(format!(
                "Qwen80 BA key head {key_head} is outside {}",
                self.key_heads
            )));
        }
        let segment_offset = match segment {
            Qwen80LinearBaSegment::BetaLogit => self.ba_beta_offset_rows,
            Qwen80LinearBaSegment::DecayInput => self.ba_decay_offset_rows,
        };
        key_head
            .checked_mul(self.ba_rows_per_key_head)
            .and_then(|base| base.checked_add(segment_offset))
            .ok_or_else(|| model_error("Qwen80 BA row offset overflowed"))
    }

    pub fn validate(&self) -> Result<()> {
        let expected = Self::source_exact();
        if self != &expected
            || self.qkvz_projection_elements()? != 12_288
            || self.ba_projection_elements()? != 64
            || self.key_elements()? != 2_048
            || self.value_elements()? != 4_096
            || self.conv_state_elements()? != 24_576
            || self.recurrent_state_elements()? != 524_288
        {
            return Err(model_error(
                "Qwen80 canonical DeltaNet layout differs from the pinned source geometry",
            ));
        }
        let final_qkvz_row = self
            .qkvz_row_offset(self.key_heads - 1, Qwen80LinearQkvzSegment::Z)?
            + self.value_heads_per_key_head * self.value_head_dim;
        let final_ba_row = self
            .ba_row_offset(self.key_heads - 1, Qwen80LinearBaSegment::DecayInput)?
            + self.value_heads_per_key_head;
        if final_qkvz_row != 12_288 || final_ba_row != 64 {
            return Err(model_error(
                "Qwen80 canonical DeltaNet source segment offsets do not close their projections",
            ));
        }
        Ok(())
    }
}

/// Byte/element requirements that a later Metal implementation must satisfy
/// before binding the canonical DeltaNet operator.  This type is metadata
/// only: it neither allocates a Metal buffer nor asserts that a dispatch ran.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen80CanonicalLinearDeltaNetDeviceResources {
    pub hidden_input_elements: usize,
    pub normalized_hidden_elements: usize,
    pub qkvz_projection_elements: usize,
    pub ba_projection_elements: usize,
    pub mixed_qkv_elements: usize,
    pub convolved_qkv_elements: usize,
    pub repeated_query_elements: usize,
    pub repeated_key_elements: usize,
    pub value_elements: usize,
    pub z_elements: usize,
    pub gated_output_elements: usize,
    pub mixer_output_elements: usize,
    pub conv_state_offset_elements: usize,
    pub conv_state_capacity_elements: usize,
    pub recurrent_state_offset_elements: usize,
    pub recurrent_state_capacity_elements: usize,
    pub direct_packed_payload_bytes: BTreeMap<String, usize>,
}

impl Qwen80CanonicalLinearDeltaNetDeviceResources {
    fn minimum(
        layout: &Qwen80CanonicalLinearDeltaNetLayout,
        linear_state_slot: usize,
        direct_packed_payload_bytes: BTreeMap<String, usize>,
    ) -> Result<Self> {
        layout.validate()?;
        let conv_state_offset_elements = linear_state_slot
            .checked_mul(layout.conv_state_elements()?)
            .ok_or_else(|| model_error("Qwen80 DeltaNet convolution state offset overflowed"))?;
        let recurrent_state_offset_elements = linear_state_slot
            .checked_mul(layout.recurrent_state_elements()?)
            .ok_or_else(|| model_error("Qwen80 DeltaNet recurrent state offset overflowed"))?;
        let conv_state_capacity_elements = conv_state_offset_elements
            .checked_add(layout.conv_state_elements()?)
            .ok_or_else(|| model_error("Qwen80 DeltaNet convolution capacity overflowed"))?;
        let recurrent_state_capacity_elements = recurrent_state_offset_elements
            .checked_add(layout.recurrent_state_elements()?)
            .ok_or_else(|| model_error("Qwen80 DeltaNet recurrent capacity overflowed"))?;
        Ok(Self {
            hidden_input_elements: layout.hidden_elements,
            normalized_hidden_elements: layout.hidden_elements,
            qkvz_projection_elements: layout.qkvz_projection_elements()?,
            ba_projection_elements: layout.ba_projection_elements()?,
            mixed_qkv_elements: layout.conv_channels,
            convolved_qkv_elements: layout.conv_channels,
            repeated_query_elements: layout.value_elements()?,
            repeated_key_elements: layout.value_elements()?,
            value_elements: layout.value_elements()?,
            z_elements: layout.value_elements()?,
            gated_output_elements: layout.value_elements()?,
            mixer_output_elements: layout.hidden_elements,
            conv_state_offset_elements,
            conv_state_capacity_elements,
            recurrent_state_offset_elements,
            recurrent_state_capacity_elements,
            direct_packed_payload_bytes,
        })
    }
}

/// Strict source/artifact contract for a single Gated DeltaNet decode layer.
/// It is the handoff boundary between the proven compact-payload CPU oracle
/// and a future device-resident Metal executor.  Constructing or validating
/// this contract is not a Metal dispatch or a native-runtime receipt.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen80CanonicalLinearDeltaNetOperatorContract {
    pub manifest_seal_sha256: String,
    pub source_revision: String,
    pub layer: usize,
    pub linear_state_slot: usize,
    pub input_layernorm: Qwen80PackedTensorBinding,
    pub mixer: Qwen80LinearDeltaNetLayerBindings,
    pub layout: Qwen80CanonicalLinearDeltaNetLayout,
    pub minimum_device_resources: Qwen80CanonicalLinearDeltaNetDeviceResources,
}

impl Qwen80CanonicalLinearDeltaNetOperatorContract {
    fn required_bindings(&self) -> [&Qwen80PackedTensorBinding; 8] {
        [
            &self.input_layernorm,
            &self.mixer.in_proj_qkvz,
            &self.mixer.in_proj_ba,
            &self.mixer.causal_conv1d,
            &self.mixer.a_log,
            &self.mixer.dt_bias,
            &self.mixer.gated_rms_norm,
            &self.mixer.out_proj,
        ]
    }

    pub fn validate(&self) -> Result<()> {
        self.layout.validate()?;
        if self.manifest_seal_sha256.len() != 64 || self.source_revision.is_empty() {
            return Err(model_error(
                "Qwen80 canonical DeltaNet contract has no complete artifact identity",
            ));
        }
        if !matches!(
            qwen80_layer_kind(self.layer)?,
            Qwen80LayerKind::LinearAttention
        ) || self.linear_state_slot != self.layer - self.layer / QWEN80_FULL_ATTENTION_INTERVAL
        {
            return Err(model_error(format!(
                "Qwen80 canonical DeltaNet contract layer {} / state slot {} is not on the 3:1 source schedule",
                self.layer, self.linear_state_slot
            )));
        }
        let expected_prefix = format!("model.layers.{}", self.layer);
        let expected = [
            (
                "input_layernorm",
                &self.input_layernorm,
                vec![QWEN80_HIDDEN],
                "input_layernorm.weight",
            ),
            (
                "in_proj_qkvz",
                &self.mixer.in_proj_qkvz,
                vec![self.layout.qkvz_projection_elements()?, QWEN80_HIDDEN],
                "linear_attn.in_proj_qkvz.weight",
            ),
            (
                "in_proj_ba",
                &self.mixer.in_proj_ba,
                vec![self.layout.ba_projection_elements()?, QWEN80_HIDDEN],
                "linear_attn.in_proj_ba.weight",
            ),
            (
                "conv1d",
                &self.mixer.causal_conv1d,
                vec![self.layout.conv_channels, 1, self.layout.conv_kernel],
                "linear_attn.conv1d.weight",
            ),
            (
                "A_log",
                &self.mixer.a_log,
                vec![self.layout.value_heads],
                "linear_attn.A_log",
            ),
            (
                "dt_bias",
                &self.mixer.dt_bias,
                vec![self.layout.value_heads],
                "linear_attn.dt_bias",
            ),
            (
                "gated_rms_norm",
                &self.mixer.gated_rms_norm,
                vec![self.layout.value_head_dim],
                "linear_attn.norm.weight",
            ),
            (
                "out_proj",
                &self.mixer.out_proj,
                vec![QWEN80_HIDDEN, self.layout.value_elements()?],
                "linear_attn.out_proj.weight",
            ),
        ];
        for (label, binding, shape, suffix) in expected {
            if binding.group_size != QWEN80_GROUP_SIZE || binding.shape != shape {
                return Err(model_error(format!(
                    "Qwen80 canonical DeltaNet {label} binding geometry differs from source"
                )));
            }
            if binding.name != format!("{expected_prefix}.{suffix}") {
                return Err(model_error(format!(
                    "Qwen80 canonical DeltaNet {label} binding is outside its exact layer {}",
                    self.layer
                )));
            }
        }
        self.validate_device_resources(&self.minimum_device_resources)
    }

    /// Check a prospective device allocation plan. This intentionally checks
    /// only offsets, capacities, and immutable compact payload lengths; the
    /// caller must separately create and execute a real Metal command graph.
    pub fn validate_device_resources(
        &self,
        resources: &Qwen80CanonicalLinearDeltaNetDeviceResources,
    ) -> Result<()> {
        self.layout.validate()?;
        let required = &self.minimum_device_resources;
        let exact_scalars = [
            (
                "hidden input",
                resources.hidden_input_elements,
                required.hidden_input_elements,
            ),
            (
                "normalized hidden",
                resources.normalized_hidden_elements,
                required.normalized_hidden_elements,
            ),
            (
                "QKVZ projection",
                resources.qkvz_projection_elements,
                required.qkvz_projection_elements,
            ),
            (
                "BA projection",
                resources.ba_projection_elements,
                required.ba_projection_elements,
            ),
            (
                "mixed QKV",
                resources.mixed_qkv_elements,
                required.mixed_qkv_elements,
            ),
            (
                "convolved QKV",
                resources.convolved_qkv_elements,
                required.convolved_qkv_elements,
            ),
            (
                "repeated query",
                resources.repeated_query_elements,
                required.repeated_query_elements,
            ),
            (
                "repeated key",
                resources.repeated_key_elements,
                required.repeated_key_elements,
            ),
            ("value", resources.value_elements, required.value_elements),
            ("Z", resources.z_elements, required.z_elements),
            (
                "gated output",
                resources.gated_output_elements,
                required.gated_output_elements,
            ),
            (
                "mixer output",
                resources.mixer_output_elements,
                required.mixer_output_elements,
            ),
        ];
        for (label, observed, expected) in exact_scalars {
            if observed != expected {
                return Err(model_error(format!(
                    "Qwen80 canonical DeltaNet {label} resource has {observed} elements; expected {expected}"
                )));
            }
        }
        if resources.conv_state_offset_elements != required.conv_state_offset_elements
            || resources.recurrent_state_offset_elements != required.recurrent_state_offset_elements
            || resources.conv_state_capacity_elements < required.conv_state_capacity_elements
            || resources.recurrent_state_capacity_elements
                < required.recurrent_state_capacity_elements
        {
            return Err(model_error(
                "Qwen80 canonical DeltaNet state resource offsets/capacities do not cover its exact slot",
            ));
        }
        if resources.direct_packed_payload_bytes != required.direct_packed_payload_bytes {
            return Err(model_error(
                "Qwen80 canonical DeltaNet compact payload byte map differs from its admitted contract",
            ));
        }
        Ok(())
    }

    fn validate_against_catalog(&self, catalog: &Qwen80CompleteArtifactCatalog) -> Result<()> {
        self.validate()?;
        if catalog.manifest_seal() != self.manifest_seal_sha256
            || catalog.config.source_revision != self.source_revision
        {
            return Err(model_error(
                "Qwen80 canonical DeltaNet contract does not match the admitted catalog identity",
            ));
        }
        for binding in self.required_bindings() {
            let header = catalog.direct_tensor_header(&binding.name)?;
            if header.shape != binding.shape || header.group_size != binding.group_size {
                return Err(model_error(format!(
                    "Qwen80 canonical DeltaNet admitted header drifted for {:?}",
                    binding.name
                )));
            }
            let payload = catalog.verified_direct_tensor_payload(&binding.name)?;
            let required_bytes = self
                .minimum_device_resources
                .direct_packed_payload_bytes
                .get(&binding.name)
                .copied()
                .ok_or_else(|| {
                    model_error(format!(
                        "Qwen80 canonical DeltaNet resource map omits {:?}",
                        binding.name
                    ))
                })?;
            if payload.len() != required_bytes || header.payload_bytes != required_bytes {
                return Err(model_error(format!(
                    "Qwen80 canonical DeltaNet compact payload length drifted for {:?}",
                    binding.name
                )));
            }
        }
        Ok(())
    }
}

/// Strict source/artifact contract for the post-DeltaNet MoE half of one
/// source-scheduled linear decoder layer.  The first mixer contract remains
/// embedded rather than reconstructed, so a router/expert step cannot attach
/// to an unrelated manifest, source revision, or DeltaNet state slot.  It
/// specifies only compact-payload CPU/device boundaries; it is not a runtime
/// receipt and does not create a fallback execution path.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen80CanonicalLinearMoEOperatorContract {
    pub manifest_seal_sha256: String,
    pub source_revision: String,
    pub mixer: Qwen80CanonicalLinearDeltaNetOperatorContract,
    pub post_attention_layernorm: Qwen80PackedTensorBinding,
    pub router: Qwen80PackedTensorBinding,
    pub shared_gate_proj: Qwen80PackedTensorBinding,
    pub shared_up_proj: Qwen80PackedTensorBinding,
    pub shared_down_proj: Qwen80PackedTensorBinding,
    pub shared_expert_gate: Qwen80PackedTensorBinding,
}

impl Qwen80CanonicalLinearMoEOperatorContract {
    fn fixed_moe_bindings(&self) -> [&Qwen80PackedTensorBinding; 6] {
        [
            &self.post_attention_layernorm,
            &self.router,
            &self.shared_gate_proj,
            &self.shared_up_proj,
            &self.shared_down_proj,
            &self.shared_expert_gate,
        ]
    }

    pub fn validate(&self) -> Result<()> {
        self.mixer.validate()?;
        if self.manifest_seal_sha256 != self.mixer.manifest_seal_sha256
            || self.source_revision != self.mixer.source_revision
        {
            return Err(model_error(
                "Qwen80 canonical linear MoE contract drifted from its DeltaNet artifact identity",
            ));
        }
        let prefix = format!("model.layers.{}", self.mixer.layer);
        let expected = [
            (
                "post_attention_layernorm",
                &self.post_attention_layernorm,
                vec![QWEN80_HIDDEN],
                "post_attention_layernorm.weight",
            ),
            (
                "router",
                &self.router,
                vec![QWEN80_EXPERTS, QWEN80_HIDDEN],
                "mlp.gate.weight",
            ),
            (
                "shared_gate_proj",
                &self.shared_gate_proj,
                vec![QWEN80_SHARED_EXPERT_INTERMEDIATE, QWEN80_HIDDEN],
                "mlp.shared_expert.gate_proj.weight",
            ),
            (
                "shared_up_proj",
                &self.shared_up_proj,
                vec![QWEN80_SHARED_EXPERT_INTERMEDIATE, QWEN80_HIDDEN],
                "mlp.shared_expert.up_proj.weight",
            ),
            (
                "shared_down_proj",
                &self.shared_down_proj,
                vec![QWEN80_HIDDEN, QWEN80_SHARED_EXPERT_INTERMEDIATE],
                "mlp.shared_expert.down_proj.weight",
            ),
            (
                "shared_expert_gate",
                &self.shared_expert_gate,
                vec![1, QWEN80_HIDDEN],
                "mlp.shared_expert_gate.weight",
            ),
        ];
        for (label, binding, shape, suffix) in expected {
            if binding.group_size != QWEN80_GROUP_SIZE || binding.shape != shape {
                return Err(model_error(format!(
                    "Qwen80 canonical linear MoE {label} binding geometry differs from source"
                )));
            }
            if binding.name != format!("{prefix}.{suffix}") {
                return Err(model_error(format!(
                    "Qwen80 canonical linear MoE {label} binding is outside source layer {}",
                    self.mixer.layer
                )));
            }
        }
        Ok(())
    }

    fn validate_against_catalog(&self, catalog: &Qwen80CompleteArtifactCatalog) -> Result<()> {
        self.validate()?;
        self.mixer.validate_against_catalog(catalog)?;
        if catalog.manifest_seal() != self.manifest_seal_sha256
            || catalog.config.source_revision != self.source_revision
        {
            return Err(model_error(
                "Qwen80 canonical linear MoE contract does not match the admitted catalog identity",
            ));
        }
        for binding in self.fixed_moe_bindings() {
            let header = catalog.direct_tensor_header(&binding.name)?;
            if header.shape != binding.shape || header.group_size != binding.group_size {
                return Err(model_error(format!(
                    "Qwen80 canonical linear MoE admitted header drifted for {:?}",
                    binding.name
                )));
            }
            let payload = catalog.verified_direct_tensor_payload(&binding.name)?;
            if payload.len() != header.payload_bytes {
                return Err(model_error(format!(
                    "Qwen80 canonical linear MoE compact payload length drifted for {:?}",
                    binding.name
                )));
            }
        }
        Ok(())
    }

    /// Materialize exact *bindings* for the device- or CPU-selected top-10
    /// route, after the source router has emitted it.  This has no filename
    /// heuristic: every expert body is derived from a validated route and is
    /// itself required to be present in the immutable admission payload cache.
    fn routed_expert_bindings(
        &self,
        catalog: &Qwen80CompleteArtifactCatalog,
        route: &Qwen80RouteSelection,
    ) -> Result<Vec<Qwen80ExpertBindings>> {
        self.validate_against_catalog(catalog)?;
        route.validate()?;
        let mut bindings = Vec::with_capacity(QWEN80_TOP_K);
        for &expert_id in &route.ids {
            let expert = expert_id as usize;
            let prefix = format!("model.layers.{}.mlp.experts.{expert}", self.mixer.layer);
            let expert_bindings = Qwen80ExpertBindings {
                expert,
                gate_proj: Qwen80CompleteHybridDecoderPlan::binding(
                    catalog,
                    format!("{prefix}.gate_proj.weight"),
                    &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
                )?,
                up_proj: Qwen80CompleteHybridDecoderPlan::binding(
                    catalog,
                    format!("{prefix}.up_proj.weight"),
                    &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
                )?,
                down_proj: Qwen80CompleteHybridDecoderPlan::binding(
                    catalog,
                    format!("{prefix}.down_proj.weight"),
                    &[QWEN80_HIDDEN, QWEN80_MOE_INTERMEDIATE],
                )?,
            };
            for binding in [
                &expert_bindings.gate_proj,
                &expert_bindings.up_proj,
                &expert_bindings.down_proj,
            ] {
                let header = catalog.direct_tensor_header(&binding.name)?;
                let payload = catalog.verified_direct_tensor_payload(&binding.name)?;
                if header.shape != binding.shape
                    || header.group_size != binding.group_size
                    || payload.len() != header.payload_bytes
                {
                    return Err(model_error(format!(
                        "Qwen80 canonical routed expert {expert} compact payload/header drifted for {:?}",
                        binding.name
                    )));
                }
            }
            bindings.push(expert_bindings);
        }
        if bindings.len() != QWEN80_TOP_K {
            return Err(model_error(
                "Qwen80 canonical routed-expert binding count is not source top-10",
            ));
        }
        Ok(bindings)
    }
}

impl Qwen80CompleteArtifactCatalog {
    /// Produce an artifact-bound contract for one source-scheduled Gated
    /// DeltaNet layer.  This validates immutable direct payload snapshots but
    /// deliberately does not allocate or dispatch Metal work.
    pub fn canonical_linear_deltanet_operator_contract(
        &self,
        layer: usize,
    ) -> Result<Qwen80CanonicalLinearDeltaNetOperatorContract> {
        if !matches!(
            self.config.layer_kind(layer)?,
            Qwen80LayerKind::LinearAttention
        ) {
            return Err(model_error(format!(
                "Qwen80 layer {layer} is not a canonical Gated DeltaNet layer"
            )));
        }
        let layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
        layout.validate()?;
        let linear_state_slot = layer - layer / QWEN80_FULL_ATTENTION_INTERVAL;
        let prefix = format!("model.layers.{layer}");
        let input_layernorm = Qwen80CompleteHybridDecoderPlan::binding(
            self,
            format!("{prefix}.input_layernorm.weight"),
            &[layout.hidden_elements],
        )?;
        let mixer = Qwen80LinearDeltaNetLayerBindings {
            in_proj_qkvz: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.linear_attn.in_proj_qkvz.weight"),
                &[layout.qkvz_projection_elements()?, layout.hidden_elements],
            )?,
            in_proj_ba: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.linear_attn.in_proj_ba.weight"),
                &[layout.ba_projection_elements()?, layout.hidden_elements],
            )?,
            causal_conv1d: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.linear_attn.conv1d.weight"),
                &[layout.conv_channels, 1, layout.conv_kernel],
            )?,
            a_log: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.linear_attn.A_log"),
                &[layout.value_heads],
            )?,
            dt_bias: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.linear_attn.dt_bias"),
                &[layout.value_heads],
            )?,
            gated_rms_norm: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.linear_attn.norm.weight"),
                &[layout.value_head_dim],
            )?,
            out_proj: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.linear_attn.out_proj.weight"),
                &[layout.hidden_elements, layout.value_elements()?],
            )?,
        };
        let bindings = [
            &input_layernorm,
            &mixer.in_proj_qkvz,
            &mixer.in_proj_ba,
            &mixer.causal_conv1d,
            &mixer.a_log,
            &mixer.dt_bias,
            &mixer.gated_rms_norm,
            &mixer.out_proj,
        ];
        let mut direct_packed_payload_bytes = BTreeMap::new();
        for binding in bindings {
            let header = self.direct_tensor_header(&binding.name)?;
            direct_packed_payload_bytes.insert(binding.name.clone(), header.payload_bytes);
        }
        let contract = Qwen80CanonicalLinearDeltaNetOperatorContract {
            manifest_seal_sha256: self.manifest_seal().to_owned(),
            source_revision: self.config.source_revision.clone(),
            layer,
            linear_state_slot,
            input_layernorm,
            mixer,
            layout: layout.clone(),
            minimum_device_resources: Qwen80CanonicalLinearDeltaNetDeviceResources::minimum(
                &layout,
                linear_state_slot,
                direct_packed_payload_bytes,
            )?,
        };
        contract.validate_against_catalog(self)?;
        Ok(contract)
    }

    /// Produce the artifact-bound post-DeltaNet MoE contract for one
    /// source-scheduled linear layer.  All fixed bodies are verified now;
    /// routed expert bodies remain deferred until an actual source top-10
    /// decision exists, at which point `routed_expert_bindings` validates their
    /// compact payloads one by one.
    pub fn canonical_linear_moe_operator_contract(
        &self,
        layer: usize,
    ) -> Result<Qwen80CanonicalLinearMoEOperatorContract> {
        let mixer = self.canonical_linear_deltanet_operator_contract(layer)?;
        let prefix = format!("model.layers.{layer}");
        let contract = Qwen80CanonicalLinearMoEOperatorContract {
            manifest_seal_sha256: self.manifest_seal().to_owned(),
            source_revision: self.config.source_revision.clone(),
            mixer,
            post_attention_layernorm: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.post_attention_layernorm.weight"),
                &[QWEN80_HIDDEN],
            )?,
            router: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.mlp.gate.weight"),
                &[QWEN80_EXPERTS, QWEN80_HIDDEN],
            )?,
            shared_gate_proj: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                &[QWEN80_SHARED_EXPERT_INTERMEDIATE, QWEN80_HIDDEN],
            )?,
            shared_up_proj: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                &[QWEN80_SHARED_EXPERT_INTERMEDIATE, QWEN80_HIDDEN],
            )?,
            shared_down_proj: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                &[QWEN80_HIDDEN, QWEN80_SHARED_EXPERT_INTERMEDIATE],
            )?,
            shared_expert_gate: Qwen80CompleteHybridDecoderPlan::binding(
                self,
                format!("{prefix}.mlp.shared_expert_gate.weight"),
                &[1, QWEN80_HIDDEN],
            )?,
        };
        contract.validate_against_catalog(self)?;
        Ok(contract)
    }
}

/// State for one canonical source-scheduled Qwen3-Next Gated DeltaNet mixer
/// decode step. The convolution state is `[8192 channels][3 prior tokens]`;
/// the recurrent state is `[32 value heads][128 key][128 value]`. It is a
/// layer-local state slice, not the entire 36-layer runtime state body.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80CanonicalLinearLayerCpuState {
    pub conv_state: Vec<f32>,
    pub recurrent_state: Vec<f32>,
}

impl Qwen80CanonicalLinearLayerCpuState {
    pub fn zeroed() -> Self {
        let conv_channels = QWEN80_LINEAR_KEY_HEADS * QWEN80_LINEAR_KEY_HEAD_DIM * 2
            + QWEN80_LINEAR_VALUE_HEADS * QWEN80_LINEAR_VALUE_HEAD_DIM;
        Self {
            conv_state: vec![0.0; conv_channels * (QWEN80_LINEAR_CONV_KERNEL - 1)],
            recurrent_state: vec![
                0.0;
                QWEN80_LINEAR_VALUE_HEADS
                    * QWEN80_LINEAR_KEY_HEAD_DIM
                    * QWEN80_LINEAR_VALUE_HEAD_DIM
            ],
        }
    }

    fn validate(&self) -> Result<()> {
        let expected_conv = QWEN80_LINEAR_KEY_HEADS
            .checked_mul(QWEN80_LINEAR_KEY_HEAD_DIM)
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| {
                value.checked_add(QWEN80_LINEAR_VALUE_HEADS * QWEN80_LINEAR_VALUE_HEAD_DIM)
            })
            .and_then(|value| value.checked_mul(QWEN80_LINEAR_CONV_KERNEL - 1))
            .ok_or_else(|| model_error("canonical linear convolution state geometry overflowed"))?;
        let expected_recurrent = checked_product(
            &[
                QWEN80_LINEAR_VALUE_HEADS,
                QWEN80_LINEAR_KEY_HEAD_DIM,
                QWEN80_LINEAR_VALUE_HEAD_DIM,
            ],
            "canonical linear recurrent state",
        )?;
        if self.conv_state.len() != expected_conv
            || self.recurrent_state.len() != expected_recurrent
        {
            return Err(model_error(format!(
                "canonical linear CPU state has conv={} / recurrent={}, expected {expected_conv} / {expected_recurrent}",
                self.conv_state.len(),
                self.recurrent_state.len(),
            )));
        }
        if self
            .conv_state
            .iter()
            .chain(self.recurrent_state.iter())
            .any(|value| !value.is_finite())
        {
            return Err(model_error(
                "canonical linear CPU state contains a non-finite value",
            ));
        }
        Ok(())
    }
}

/// Input for the one-layer direct-packed CPU reference.  `hidden` is supplied
/// by an upstream layer/test harness; embedding lookup and the MoE portion of
/// this layer are deliberately outside this bounded mixer oracle.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80CanonicalLinearLayerCpuInput {
    pub hidden: Vec<f32>,
    pub state: Qwen80CanonicalLinearLayerCpuState,
}

impl Qwen80CanonicalLinearLayerCpuInput {
    pub fn with_zero_state(hidden: Vec<f32>) -> Self {
        Self {
            hidden,
            state: Qwen80CanonicalLinearLayerCpuState::zeroed(),
        }
    }

    fn validate(&self) -> Result<()> {
        if self.hidden.len() != QWEN80_HIDDEN || self.hidden.iter().any(|value| !value.is_finite())
        {
            return Err(model_error(format!(
                "canonical linear CPU input has {} hidden values; expected {QWEN80_HIDDEN} finite values",
                self.hidden.len()
            )));
        }
        self.state.validate()
    }
}

/// One source-token embedding resolved directly from the admitted compact
/// artifact. This is a CPU/reference parity result only: it establishes the
/// exact input boundary a future device-resident embedding gather must match,
/// and is never a production CPU fallback or a generated-token result.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80DirectPackedEmbeddingCpuOracleResult {
    pub token_id: u32,
    pub direct_packed_embedding_tensor: String,
    pub hidden: Vec<f32>,
    pub source_algorithm_boundary: String,
}

/// Result from one exact-source-shape canonical DeltaNet mixer plus its first
/// residual add. It is intentionally bridge-compatible (layer/state-slot,
/// compact tensor identities, mixer output, residual output, next state) but
/// is not a complete decoder layer: post-attention RMSNorm, routed/shared
/// MoE, second residual, later layers, final norm, lm_head, sampler, and
/// feedback remain absent.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80CanonicalLinearLayerCpuOracleResult {
    pub layer: usize,
    pub linear_state_slot: usize,
    pub direct_packed_input_layernorm_tensor: String,
    pub direct_packed_qkvz_tensor: String,
    pub direct_packed_ba_tensor: String,
    pub direct_packed_conv_tensor: String,
    pub direct_packed_gated_norm_tensor: String,
    pub direct_packed_out_proj_tensor: String,
    pub input_rms_norm_output: Vec<f32>,
    pub projected_qkvz: Vec<f32>,
    pub projected_ba: Vec<f32>,
    pub repeated_query_l2_scaled: Vec<f32>,
    pub repeated_key_l2: Vec<f32>,
    pub convolved_value: Vec<f32>,
    pub z: Vec<f32>,
    pub decay: Vec<f32>,
    pub beta: Vec<f32>,
    pub recurrent_output: Vec<f32>,
    pub gated_rms_norm_output: Vec<f32>,
    pub mixer_output: Vec<f32>,
    pub mixer_residual_output: Vec<f32>,
    pub next_state: Qwen80CanonicalLinearLayerCpuState,
    pub source_algorithm_boundary: String,
}

/// One routed source expert evaluated through immutable direct-packed payloads
/// in the bounded CPU oracle.  This mirrors `silu(gate_proj(x)) * up_proj(x)`
/// followed by `down_proj`, then applies the already normalized source route
/// weight.  It is a parity record, not a CPU production execution path.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80RoutedExpertCpuOracleResult {
    pub expert: usize,
    pub route_weight: f32,
    pub direct_packed_gate_proj_tensor: String,
    pub direct_packed_up_proj_tensor: String,
    pub direct_packed_down_proj_tensor: String,
    pub gate_projection: Vec<f32>,
    pub up_projection: Vec<f32>,
    pub gated_up_product: Vec<f32>,
    pub output: Vec<f32>,
    pub weighted_output: Vec<f32>,
}

/// CPU/reference witness for one member of a source-selected all-ten routed
/// expert wave.  The vectors remain available for a later same-capture Metal
/// parity check; this type is never a production CPU fallback.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80AllTenRoutedExpertCpuWitness {
    pub wave_index: usize,
    pub expert: usize,
    pub normalized_weight: f32,
    pub direct_packed_gate_artifact_sha256: String,
    pub direct_packed_up_artifact_sha256: String,
    pub direct_packed_down_artifact_sha256: String,
    pub weighted_output_sha256: String,
    pub oracle: Qwen80RoutedExpertCpuOracleResult,
}

/// One source-bound direct-packed all-ten routed-expert CPU reference pass.
/// It begins with an explicit already-normalized hidden `[2048]` input and
/// ends before shared expert, route combine, residual, token feedback, or any
/// production model claim.  It exists to prepare a single device graph whose
/// per-route inputs and outputs have exact immutable identities.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80AllTenRoutedExpertCpuOracleResult {
    pub layer: usize,
    pub manifest_document_sha256: String,
    pub plan_document_sha256: String,
    pub normalized_hidden_sha256: String,
    pub route: Qwen80RouteSelection,
    pub witnesses: Vec<Qwen80AllTenRoutedExpertCpuWitness>,
    pub routed_expert_sum: Vec<f32>,
    pub routed_expert_sum_sha256: String,
    pub source_algorithm_boundary: String,
}

/// A complete source-shaped layer-0 DeltaNet + sparse-MoE CPU parity result.
/// The oracle starts from a supplied hidden vector and canonical DeltaNet
/// state, directly consumes only admission-verified compact payloads, and
/// ends at the second residual.  It is deliberately not a native runtime,
/// token, decoder, HCLI, or throughput result; it exists to make the next
/// device stage prove all intervening source boundaries.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen80CanonicalLinearMoECpuOracleResult {
    pub mixer: Qwen80CanonicalLinearLayerCpuOracleResult,
    pub direct_packed_post_attention_layernorm_tensor: String,
    pub direct_packed_router_tensor: String,
    pub direct_packed_shared_gate_proj_tensor: String,
    pub direct_packed_shared_up_proj_tensor: String,
    pub direct_packed_shared_down_proj_tensor: String,
    pub direct_packed_shared_expert_gate_tensor: String,
    pub post_attention_rms_norm_output: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub route: Qwen80RouteSelection,
    pub routed_experts: Vec<Qwen80RoutedExpertCpuOracleResult>,
    pub routed_expert_sum: Vec<f32>,
    pub shared_gate_projection: Vec<f32>,
    pub shared_up_projection: Vec<f32>,
    pub shared_gated_up_product: Vec<f32>,
    pub shared_expert_output: Vec<f32>,
    pub shared_expert_gate_logit: f32,
    pub shared_expert_gate_value: f32,
    pub shared_gated_output: Vec<f32>,
    pub moe_output: Vec<f32>,
    pub layer_output: Vec<f32>,
    pub source_algorithm_boundary: String,
}

#[derive(Clone)]
struct Qwen80CpuPackedTensor {
    name: String,
    header: CompleteBinaryHeader,
    payload: Arc<[u8]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum Qwen80CpuPackedReadMode {
    StreamingDirect,
    MaterializedReference,
}

impl Qwen80CpuPackedTensor {
    fn from_binding(
        catalog: &Qwen80CompleteArtifactCatalog,
        binding: &Qwen80PackedTensorBinding,
    ) -> Result<Self> {
        let header = catalog.direct_tensor_header(&binding.name)?;
        if header.shape != binding.shape || header.group_size != binding.group_size {
            return Err(model_error(format!(
                "canonical linear direct tensor {:?} has shape {:?}/group {}, expected {:?}/{}",
                binding.name, header.shape, header.group_size, binding.shape, binding.group_size
            )));
        }
        let payload = catalog.verified_direct_tensor_payload(&binding.name)?;
        Ok(Self {
            name: binding.name.clone(),
            header: header.clone(),
            payload,
        })
    }

    fn value(&self, index: usize) -> Result<f32> {
        if index >= self.header.elements {
            return Err(model_error(format!(
                "canonical linear direct tensor {:?} element {index} is outside {}",
                self.name, self.header.elements
            )));
        }
        let group = index / self.header.group_size;
        let local = index % self.header.group_size;
        let scale_offset =
            self.header
                .scale_offset
                .checked_add(group.checked_mul(2).ok_or_else(|| {
                    model_error("canonical linear direct scale offset overflowed")
                })?)
                .ok_or_else(|| model_error("canonical linear direct scale offset overflowed"))?;
        let scale_bytes = self
            .payload
            .get(scale_offset..scale_offset + 2)
            .ok_or_else(|| model_error("canonical linear direct scale is truncated"))?;
        let scale = f16::from_bits(u16::from_le_bytes([scale_bytes[0], scale_bytes[1]])).to_f32();
        let sign_group_offset = self
            .header
            .sign_offset
            .checked_add(
                group
                    .checked_mul(self.header.group_size / 8)
                    .ok_or_else(|| model_error("canonical linear direct sign offset overflowed"))?,
            )
            .ok_or_else(|| model_error("canonical linear direct sign offset overflowed"))?;
        let byte = *self
            .payload
            .get(sign_group_offset + local / 8)
            .ok_or_else(|| model_error("canonical linear direct sign is truncated"))?;
        let value = if (byte >> (local % 8)) & 1 == 1 {
            scale
        } else {
            -scale
        };
        if !value.is_finite() {
            return Err(model_error(
                "canonical linear direct packed value is non-finite",
            ));
        }
        Ok(value)
    }

    fn vector(&self, elements: usize, mode: Qwen80CpuPackedReadMode) -> Result<Vec<f32>> {
        if self.header.shape != [elements] {
            return Err(model_error(format!(
                "canonical linear direct vector {:?} has shape {:?}, expected [{elements}]",
                self.name, self.header.shape
            )));
        }
        match mode {
            Qwen80CpuPackedReadMode::StreamingDirect => {
                (0..elements).map(|index| self.value(index)).collect()
            }
            Qwen80CpuPackedReadMode::MaterializedReference => {
                let (header, values) = decode_complete_binary_f32(&self.payload)?;
                if header != self.header || values.len() != elements {
                    return Err(model_error(format!(
                        "canonical linear materialized vector {:?} drifted from admission header",
                        self.name
                    )));
                }
                Ok(values)
            }
        }
    }

    /// Read one direct-packed matrix row without materializing any other row.
    /// This is the CPU reference counterpart to a future device embedding
    /// gather and deliberately retains the compact sign/scale representation.
    fn row(
        &self,
        row: usize,
        rows: usize,
        cols: usize,
        mode: Qwen80CpuPackedReadMode,
    ) -> Result<Vec<f32>> {
        if self.header.shape != [rows, cols] || row >= rows {
            return Err(model_error(format!(
                "canonical linear direct matrix {:?} cannot read row {row} from shape {:?}; expected [{rows}, {cols}]",
                self.name, self.header.shape,
            )));
        }
        let start = row
            .checked_mul(cols)
            .ok_or_else(|| model_error("canonical linear direct row start overflowed"))?;
        let end = start
            .checked_add(cols)
            .ok_or_else(|| model_error("canonical linear direct row end overflowed"))?;
        match mode {
            Qwen80CpuPackedReadMode::StreamingDirect => {
                (start..end).map(|index| self.value(index)).collect()
            }
            Qwen80CpuPackedReadMode::MaterializedReference => {
                let (header, values) = decode_complete_binary_f32(&self.payload)?;
                if header != self.header || values.len() != rows * cols {
                    return Err(model_error(format!(
                        "canonical linear materialized matrix {:?} drifted from admission header",
                        self.name
                    )));
                }
                Ok(values[start..end].to_vec())
            }
        }
    }

    fn matvec(
        &self,
        input: &[f32],
        rows: usize,
        cols: usize,
        mode: Qwen80CpuPackedReadMode,
    ) -> Result<Vec<f32>> {
        if self.header.shape != [rows, cols] || input.len() != cols {
            return Err(model_error(format!(
                "canonical linear direct matrix {:?} shape {:?} / input {} differs from [{rows}, {cols}]",
                self.name,
                self.header.shape,
                input.len(),
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(model_error(
                "canonical linear direct matvec input is non-finite",
            ));
        }
        let mut output = vec![0.0f32; rows];
        match mode {
            Qwen80CpuPackedReadMode::StreamingDirect => {
                for row in 0..rows {
                    let row_start = row
                        .checked_mul(cols)
                        .ok_or_else(|| model_error("canonical linear row offset overflowed"))?;
                    let row_end = row_start
                        .checked_add(cols)
                        .ok_or_else(|| model_error("canonical linear row end overflowed"))?;
                    let mut element = row_start;
                    let mut sum = 0.0f32;
                    while element < row_end {
                        let group = element / self.header.group_size;
                        let group_end = ((group + 1) * self.header.group_size).min(row_end);
                        let scale_offset = self.header.scale_offset + group * 2;
                        let scale_bytes = self
                            .payload
                            .get(scale_offset..scale_offset + 2)
                            .ok_or_else(|| {
                                model_error("canonical linear matvec scale is truncated")
                            })?;
                        let scale =
                            f16::from_bits(u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]))
                                .to_f32();
                        let sign_group_offset =
                            self.header.sign_offset + group * (self.header.group_size / 8);
                        for index in element..group_end {
                            let local = index % self.header.group_size;
                            let byte =
                                *self.payload.get(sign_group_offset + local / 8).ok_or_else(
                                    || model_error("canonical linear matvec sign is truncated"),
                                )?;
                            let weight = if (byte >> (local % 8)) & 1 == 1 {
                                scale
                            } else {
                                -scale
                            };
                            sum += weight * input[index - row_start];
                        }
                        element = group_end;
                    }
                    if !sum.is_finite() {
                        return Err(model_error(format!(
                            "canonical linear direct matvec {:?} produced non-finite row {row}",
                            self.name
                        )));
                    }
                    output[row] = sum;
                }
            }
            Qwen80CpuPackedReadMode::MaterializedReference => {
                let (header, values) = decode_complete_binary_f32(&self.payload)?;
                if header != self.header || values.len() != rows * cols {
                    return Err(model_error(format!(
                        "canonical linear materialized matrix {:?} drifted from admission header",
                        self.name
                    )));
                }
                for row in 0..rows {
                    let weights = &values[row * cols..(row + 1) * cols];
                    let sum = weights
                        .iter()
                        .zip(input)
                        .fold(0.0f32, |sum, (&weight, &value)| sum + weight * value);
                    if !sum.is_finite() {
                        return Err(model_error(format!(
                            "canonical linear materialized matvec {:?} produced non-finite row {row}",
                            self.name
                        )));
                    }
                    output[row] = sum;
                }
            }
        }
        Ok(output)
    }
}

fn source_qwen80_topk_router(logits: &[f32]) -> Result<Qwen80RouteSelection> {
    if logits.len() != QWEN80_EXPERTS || logits.iter().any(|value| !value.is_finite()) {
        return Err(model_error(
            "Qwen80 source router requires 512 finite direct-packed logits",
        ));
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = logits
        .iter()
        .map(|value| (*value - maximum).exp())
        .collect::<Vec<_>>();
    let sum = probabilities.iter().sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(model_error("Qwen80 source router softmax sum is invalid"));
    }
    for probability in &mut probabilities {
        *probability /= sum;
    }
    let tie_epsilon = crate::moe::route_tie_epsilon();
    let mut ids = [0u16; QWEN80_TOP_K];
    let mut weights = [0.0f32; QWEN80_TOP_K];
    for route_index in 0..QWEN80_TOP_K {
        let mut best_index = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, &value) in probabilities.iter().enumerate() {
            let finite_pair = best_value.is_finite() && value.is_finite();
            let tied =
                tie_epsilon > 0.0 && finite_pair && (value - best_value).abs() <= tie_epsilon;
            if (value > best_value && !tied) || (tied && index < best_index) {
                best_index = index;
                best_value = value;
            }
        }
        if !best_value.is_finite() || best_value < 0.0 || best_index >= QWEN80_EXPERTS {
            return Err(model_error(
                "Qwen80 source router top-k selected an invalid probability",
            ));
        }
        ids[route_index] = u16::try_from(best_index)
            .map_err(|_| model_error("Qwen80 source router expert id overflows u16"))?;
        weights[route_index] = best_value;
        probabilities[best_index] = f32::NEG_INFINITY;
    }
    let selected_sum = weights.iter().sum::<f32>();
    if !selected_sum.is_finite() || selected_sum <= 0.0 {
        return Err(model_error(
            "Qwen80 source router selected-weight sum is invalid",
        ));
    }
    for weight in &mut weights {
        *weight /= selected_sum;
    }
    let route = Qwen80RouteSelection { ids, weights };
    route.validate()?;
    Ok(route)
}

#[derive(Clone, Debug, PartialEq)]
struct Qwen80CpuMlpOracleResult {
    gate_projection: Vec<f32>,
    up_projection: Vec<f32>,
    gated_up_product: Vec<f32>,
    output: Vec<f32>,
}

fn qwen80_f32_vector_sha256(values: &[f32], label: &str) -> Result<String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(model_error(format!(
            "Qwen80 {label} vector is empty or non-finite"
        )));
    }
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Return an opaque, process-local identity for a live Metal allocation.
///
/// The identity deliberately hashes the shared-storage address rather than
/// serializing it into a receipt. It is meaningful only while the `PinnedBuffer`
/// remains owned by the same runtime/command-graph; it is not an artifact
/// identity and must never be used to compare buffers across processes.
#[cfg(target_os = "macos")]
fn qwen80_pinned_buffer_identity_sha256(buffer: &PinnedBuffer, label: &str) -> Result<String> {
    let bytes = buffer.length();
    let contents = buffer.contents() as usize;
    if bytes == 0 || contents == 0 {
        return Err(model_error(format!(
            "Qwen80 {label} has no live shared Metal allocation"
        )));
    }
    Ok(format!(
        "{:x}",
        Sha256::digest(
            format!("qwen80-live-pinned-buffer-v1:{label}:{contents:x}:{bytes}").as_bytes()
        )
    ))
}

/// Exact `Qwen3NextMLP.forward`: down(silu(gate(x)) * up(x)).  It uses a
/// compact-payload CPU oracle only for source/reference parity and never
/// becomes a production model fallback.
fn source_qwen80_mlp(
    gate_proj: &Qwen80CpuPackedTensor,
    up_proj: &Qwen80CpuPackedTensor,
    down_proj: &Qwen80CpuPackedTensor,
    input: &[f32],
    intermediate: usize,
    mode: Qwen80CpuPackedReadMode,
) -> Result<Qwen80CpuMlpOracleResult> {
    if input.len() != QWEN80_HIDDEN || intermediate == 0 {
        return Err(model_error(
            "Qwen80 source MLP input/intermediate geometry is invalid",
        ));
    }
    let gate_projection = gate_proj.matvec(input, intermediate, QWEN80_HIDDEN, mode)?;
    let up_projection = up_proj.matvec(input, intermediate, QWEN80_HIDDEN, mode)?;
    let gated_up_product = gate_projection
        .iter()
        .zip(&up_projection)
        .map(|(&gate, &up)| (gate / (1.0 + (-gate).exp())) * up)
        .collect::<Vec<_>>();
    if gated_up_product.iter().any(|value| !value.is_finite()) {
        return Err(model_error(
            "Qwen80 source MLP gate/up product is non-finite",
        ));
    }
    let output = down_proj.matvec(&gated_up_product, QWEN80_HIDDEN, intermediate, mode)?;
    Ok(Qwen80CpuMlpOracleResult {
        gate_projection,
        up_projection,
        gated_up_product,
        output,
    })
}

impl Qwen80AllTenRoutedExpertPlan {
    /// Recheck every descriptor-selected packed body against the immutable
    /// admitted catalog before a CPU reference or future Metal graph may use
    /// it.  The descriptor itself never opens payloads; this method requires
    /// the catalog's admission-verified snapshots to exist for all 30 bodies.
    fn validate_against_catalog(&self, catalog: &Qwen80CompleteArtifactCatalog) -> Result<()> {
        require_canonical_sha256(
            &self.manifest_document_sha256,
            "all-ten routed-expert manifest document SHA-256",
        )?;
        require_canonical_sha256(
            &self.plan_document_sha256,
            "all-ten routed-expert plan document SHA-256",
        )?;
        if catalog.manifest_seal() != self.manifest_seal_sha256
            || catalog.config.source_revision != self.source_revision
            || self.layer >= QWEN80_LAYERS
            || self.waves.len() != QWEN80_TOP_K
        {
            return Err(model_error(
                "all-ten routed-expert plan does not match the admitted catalog identity/geometry",
            ));
        }
        self.route.validate()?;
        for (wave_index, wave) in self.waves.iter().enumerate() {
            if wave.expert != self.route.ids[wave_index]
                || (wave.normalized_weight - self.route.weights[wave_index]).abs() > 1.0e-6
            {
                return Err(model_error(format!(
                    "all-ten routed-expert plan wave {wave_index} drifted from its source route"
                )));
            }
            let prefix = format!("model.layers.{}.mlp.experts.{}", self.layer, wave.expert);
            for (projection, expected_name, expected_shape) in [
                (
                    &wave.gate,
                    format!("{prefix}.gate_proj.weight"),
                    vec![QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
                ),
                (
                    &wave.up,
                    format!("{prefix}.up_proj.weight"),
                    vec![QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
                ),
                (
                    &wave.down,
                    format!("{prefix}.down_proj.weight"),
                    vec![QWEN80_HIDDEN, QWEN80_MOE_INTERMEDIATE],
                ),
            ] {
                if projection.tensor_name != expected_name
                    || catalog.direct_tensor_artifact_sha256(&projection.tensor_name)?
                        != projection.artifact_sha256
                {
                    return Err(model_error(format!(
                        "all-ten routed-expert plan wave {wave_index} direct-packed projection identity drifted"
                    )));
                }
                let header = catalog.direct_tensor_header(&projection.tensor_name)?;
                if header.shape != expected_shape || header.group_size != QWEN80_GROUP_SIZE {
                    return Err(model_error(format!(
                        "all-ten routed-expert plan wave {wave_index} compact projection geometry drifted"
                    )));
                }
                // Require the immutable snapshot itself rather than allowing
                // a later executor to reopen the artifact path.
                let payload = catalog.verified_direct_tensor_payload(&projection.tensor_name)?;
                if payload.len() != header.payload_bytes {
                    return Err(model_error(format!(
                        "all-ten routed-expert plan wave {wave_index} snapshot/header byte count drifted"
                    )));
                }
            }
        }
        Ok(())
    }
}

fn qwen80_sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn append_all_ten_packed_projection_section(
    catalog: &Qwen80CompleteArtifactCatalog,
    projection: &Qwen80AllTenRoutePlanProjection,
    expected_shape: &[usize],
    scales: &mut Vec<u8>,
    signs: &mut Vec<u8>,
) -> Result<Qwen80AllTenPackedRouteProjectionSection> {
    if catalog.direct_tensor_artifact_sha256(&projection.tensor_name)? != projection.artifact_sha256
    {
        return Err(model_error(format!(
            "all-ten compact projection {:?} drifted from its descriptor artifact SHA-256",
            projection.tensor_name
        )));
    }
    let expected_header = catalog.direct_tensor_header(&projection.tensor_name)?;
    if expected_header.shape != expected_shape || expected_header.group_size != QWEN80_GROUP_SIZE {
        return Err(model_error(format!(
            "all-ten compact projection {:?} has wrong source geometry",
            projection.tensor_name
        )));
    }
    let payload = catalog.verified_direct_tensor_payload(&projection.tensor_name)?;
    let observed_header = parse_complete_binary_header(&payload)?;
    if &observed_header != expected_header || observed_header.payload_bytes != payload.len() {
        return Err(model_error(format!(
            "all-ten compact projection {:?} immutable payload/header drifted after admission",
            projection.tensor_name
        )));
    }
    let expected_elements = checked_product(expected_shape, "all-ten compact projection")?;
    let expected_groups = expected_elements
        .checked_add(QWEN80_GROUP_SIZE - 1)
        .ok_or_else(|| model_error("all-ten compact group count overflowed"))?
        / QWEN80_GROUP_SIZE;
    let expected_scale_bytes = expected_groups
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| model_error("all-ten compact scale byte count overflowed"))?;
    let expected_sign_bytes = expected_groups
        .checked_mul(QWEN80_GROUP_SIZE / 8)
        .ok_or_else(|| model_error("all-ten compact sign byte count overflowed"))?;
    if observed_header.elements != expected_elements
        || observed_header.groups != expected_groups
        || observed_header
            .sign_offset
            .checked_sub(observed_header.scale_offset)
            != Some(expected_scale_bytes)
        || observed_header
            .payload_bytes
            .checked_sub(observed_header.sign_offset)
            != Some(expected_sign_bytes)
    {
        return Err(model_error(format!(
            "all-ten compact projection {:?} fixed HQ30G1B1 body geometry drifted",
            projection.tensor_name
        )));
    }
    let source_scales = payload
        .get(observed_header.scale_offset..observed_header.sign_offset)
        .ok_or_else(|| {
            model_error(format!(
                "all-ten compact projection {:?} lacks scale section",
                projection.tensor_name
            ))
        })?;
    let source_signs = payload
        .get(observed_header.sign_offset..observed_header.payload_bytes)
        .ok_or_else(|| {
            model_error(format!(
                "all-ten compact projection {:?} lacks sign section",
                projection.tensor_name
            ))
        })?;
    if source_scales.len() != expected_scale_bytes || source_signs.len() != expected_sign_bytes {
        return Err(model_error(format!(
            "all-ten compact projection {:?} section lengths drifted",
            projection.tensor_name
        )));
    }
    let scale_offset_bytes = scales.len();
    scales.extend_from_slice(source_scales);
    let sign_offset_bytes = signs.len();
    signs.extend_from_slice(source_signs);
    Ok(Qwen80AllTenPackedRouteProjectionSection {
        tensor_name: projection.tensor_name.clone(),
        artifact_sha256: projection.artifact_sha256.clone(),
        shape: observed_header.shape,
        group_size: observed_header.group_size,
        scale_offset_bytes,
        scale_bytes: source_scales.len(),
        sign_offset_bytes,
        sign_bytes: source_signs.len(),
    })
}

impl Qwen80AllTenPackedRoutePayloadBundle {
    fn validate(&self) -> Result<()> {
        require_canonical_sha256(
            &self.manifest_document_sha256,
            "all-ten packed route manifest document SHA-256",
        )?;
        require_canonical_sha256(
            &self.manifest_seal_sha256,
            "all-ten packed route manifest seal",
        )?;
        require_canonical_sha256(
            &self.plan_document_sha256,
            "all-ten packed route plan document SHA-256",
        )?;
        self.route.validate()?;
        if self.layer >= QWEN80_LAYERS || self.waves.len() != QWEN80_TOP_K {
            return Err(model_error(
                "all-ten packed route bundle has invalid layer or wave count",
            ));
        }
        let scales_per_projection = QWEN80_MOE_INTERMEDIATE
            .checked_mul(QWEN80_HIDDEN)
            .and_then(|elements| elements.checked_div(QWEN80_GROUP_SIZE))
            .and_then(|groups| groups.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| model_error("all-ten packed route scale geometry overflowed"))?;
        let signs_per_projection = QWEN80_MOE_INTERMEDIATE
            .checked_mul(QWEN80_HIDDEN)
            .and_then(|elements| elements.checked_div(8))
            .ok_or_else(|| model_error("all-ten packed route sign geometry overflowed"))?;
        let expected_total_scales = scales_per_projection
            .checked_mul(QWEN80_TOP_K)
            .ok_or_else(|| model_error("all-ten packed route scale aggregate overflowed"))?;
        let expected_total_signs = signs_per_projection
            .checked_mul(QWEN80_TOP_K)
            .ok_or_else(|| model_error("all-ten packed route sign aggregate overflowed"))?;
        for (label, bytes, expected, observed_sha256) in [
            (
                "gate scales",
                self.gate_scales.as_ref(),
                expected_total_scales,
                self.gate_scales_sha256.as_str(),
            ),
            (
                "gate signs",
                self.gate_signs.as_ref(),
                expected_total_signs,
                self.gate_signs_sha256.as_str(),
            ),
            (
                "up scales",
                self.up_scales.as_ref(),
                expected_total_scales,
                self.up_scales_sha256.as_str(),
            ),
            (
                "up signs",
                self.up_signs.as_ref(),
                expected_total_signs,
                self.up_signs_sha256.as_str(),
            ),
            (
                "down scales",
                self.down_scales.as_ref(),
                expected_total_scales,
                self.down_scales_sha256.as_str(),
            ),
            (
                "down signs",
                self.down_signs.as_ref(),
                expected_total_signs,
                self.down_signs_sha256.as_str(),
            ),
        ] {
            if bytes.len() != expected || qwen80_sha256_hex(bytes) != observed_sha256 {
                return Err(model_error(format!(
                    "all-ten packed route {label} aggregate length or digest drifted",
                )));
            }
        }
        for (wave_index, wave) in self.waves.iter().enumerate() {
            if wave.wave_index != wave_index
                || wave.expert != self.route.ids[wave_index]
                || wave.normalized_weight.to_bits() != self.route.weights[wave_index].to_bits()
            {
                return Err(model_error(format!(
                    "all-ten packed route wave {wave_index} drifted from source router order/weight",
                )));
            }
            for (label, section, expected_shape) in [
                (
                    "gate",
                    &wave.gate,
                    &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN][..],
                ),
                (
                    "up",
                    &wave.up,
                    &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN][..],
                ),
                (
                    "down",
                    &wave.down,
                    &[QWEN80_HIDDEN, QWEN80_MOE_INTERMEDIATE][..],
                ),
            ] {
                if section.shape != expected_shape
                    || section.group_size != QWEN80_GROUP_SIZE
                    || section.scale_offset_bytes != wave_index * scales_per_projection
                    || section.sign_offset_bytes != wave_index * signs_per_projection
                    || section.scale_bytes != scales_per_projection
                    || section.sign_bytes != signs_per_projection
                {
                    return Err(model_error(format!(
                        "all-ten packed route wave {wave_index} {label} section layout drifted",
                    )));
                }
                require_canonical_sha256(
                    &section.artifact_sha256,
                    "all-ten packed route projection artifact SHA-256",
                )?;
            }
        }
        Ok(())
    }
}

impl Qwen80FirstResidualDeviceBinding {
    fn validate_for(&self, payloads: &Qwen80AllTenPackedRoutePayloadBundle) -> Result<()> {
        if self.manifest_seal_sha256 != payloads.manifest_seal_sha256
            || self.source_revision != payloads.source_revision
            || self.layer != payloads.layer
            || self.elements != QWEN80_HIDDEN
            || !self.same_command_graph_required
        {
            return Err(model_error(
                "DeltaNet first-residual device binding does not match the all-ten route payload authority",
            ));
        }
        Ok(())
    }
}

impl Qwen80CompleteArtifactCatalog {
    /// Materialize only the admitted compact scale/sign sections for the
    /// descriptor-selected ten routed experts.  This does not allocate Metal
    /// memory and does not decode weights; it makes a later device upload
    /// deterministic and impossible to redirect by filename lookup.
    pub fn build_all_ten_route_payload_bundle(
        &self,
        plan: &Qwen80AllTenRoutedExpertPlan,
    ) -> Result<Qwen80AllTenPackedRoutePayloadBundle> {
        plan.validate_against_catalog(self)?;
        let mut gate_scales = Vec::new();
        let mut gate_signs = Vec::new();
        let mut up_scales = Vec::new();
        let mut up_signs = Vec::new();
        let mut down_scales = Vec::new();
        let mut down_signs = Vec::new();
        let mut waves = Vec::with_capacity(QWEN80_TOP_K);
        for (wave_index, plan_wave) in plan.waves.iter().enumerate() {
            let gate = append_all_ten_packed_projection_section(
                self,
                &plan_wave.gate,
                &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
                &mut gate_scales,
                &mut gate_signs,
            )?;
            let up = append_all_ten_packed_projection_section(
                self,
                &plan_wave.up,
                &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
                &mut up_scales,
                &mut up_signs,
            )?;
            let down = append_all_ten_packed_projection_section(
                self,
                &plan_wave.down,
                &[QWEN80_HIDDEN, QWEN80_MOE_INTERMEDIATE],
                &mut down_scales,
                &mut down_signs,
            )?;
            waves.push(Qwen80AllTenPackedRouteWave {
                wave_index,
                expert: plan_wave.expert,
                normalized_weight: plan_wave.normalized_weight,
                gate,
                up,
                down,
            });
        }
        let bundle = Qwen80AllTenPackedRoutePayloadBundle {
            manifest_document_sha256: plan.manifest_document_sha256.clone(),
            manifest_seal_sha256: plan.manifest_seal_sha256.clone(),
            source_revision: plan.source_revision.clone(),
            plan_document_sha256: plan.plan_document_sha256.clone(),
            layer: plan.layer,
            route: plan.route.clone(),
            waves,
            gate_scales: Arc::from(gate_scales),
            gate_signs: Arc::from(gate_signs),
            up_scales: Arc::from(up_scales),
            up_signs: Arc::from(up_signs),
            down_scales: Arc::from(down_scales),
            down_signs: Arc::from(down_signs),
            gate_scales_sha256: String::new(),
            gate_signs_sha256: String::new(),
            up_scales_sha256: String::new(),
            up_signs_sha256: String::new(),
            down_scales_sha256: String::new(),
            down_signs_sha256: String::new(),
        };
        let bundle = Qwen80AllTenPackedRoutePayloadBundle {
            gate_scales_sha256: qwen80_sha256_hex(&bundle.gate_scales),
            gate_signs_sha256: qwen80_sha256_hex(&bundle.gate_signs),
            up_scales_sha256: qwen80_sha256_hex(&bundle.up_scales),
            up_signs_sha256: qwen80_sha256_hex(&bundle.up_signs),
            down_scales_sha256: qwen80_sha256_hex(&bundle.down_scales),
            down_signs_sha256: qwen80_sha256_hex(&bundle.down_signs),
            ..bundle
        };
        bundle.validate()?;
        Ok(bundle)
    }

    /// Return the only legal source/state boundary for a DeltaNet
    /// first-residual buffer.  The returned contract is intentionally not an
    /// execution receipt and cannot substitute for future buffer-content
    /// parity evidence.
    pub fn first_residual_device_binding(
        &self,
        layer_index: usize,
    ) -> Result<Qwen80FirstResidualDeviceBinding> {
        let plan = self.complete_hybrid_decoder_plan(1)?;
        let layer = plan.layers.get(layer_index).ok_or_else(|| {
            model_error(format!(
                "first-residual device binding layer {layer_index} is outside hybrid plan"
            ))
        })?;
        let linear_state_slot = match (&layer.kind, &layer.mixer, layer.linear_state_slot) {
            (
                Qwen80LayerKind::LinearAttention,
                Qwen80HybridMixerBindings::LinearDeltaNet(_),
                Some(slot),
            ) => slot,
            _ => {
                return Err(model_error(format!(
                    "first-residual device binding layer {layer_index} is not a DeltaNet mixer"
                )))
            }
        };
        Ok(Qwen80FirstResidualDeviceBinding {
            manifest_seal_sha256: self.manifest_seal().to_owned(),
            source_revision: self.config.source_revision.clone(),
            layer: layer.layer,
            linear_state_slot,
            elements: QWEN80_HIDDEN,
            same_command_graph_required: true,
        })
    }

    /// Join exact compact routed-expert bodies with the expected source
    /// DeltaNet first-residual boundary.  The result remains CPU/build-only
    /// until a later caller explicitly invokes the macOS upload method with
    /// a real `PinnedBuffer` from that mixer command graph.
    pub fn build_all_ten_true_moe_source_bridge(
        &self,
        plan: &Qwen80AllTenRoutedExpertPlan,
        first_residual: Qwen80FirstResidualDeviceBinding,
    ) -> Result<Qwen80AllTenTrueMoeSourceBridge> {
        let route_payloads = self.build_all_ten_route_payload_bundle(plan)?;
        first_residual.validate_for(&route_payloads)?;
        Ok(Qwen80AllTenTrueMoeSourceBridge {
            route_payloads,
            first_residual,
        })
    }

    /// Build the Layer-1 all-ten bridge only after an external caller has
    /// validated a sealed source-token Layer-1 route authority.  This narrow
    /// adapter exists because the historical L0 route-plan grammar deliberately
    /// cannot be repurposed for Layer 1.  It derives all thirty bodies from
    /// the admitted catalog and the supplied source route, then rechecks every
    /// compact section while building the bridge; it never accepts a filename
    /// list or a caller-provided payload buffer.
    ///
    /// `route_authority_document_sha256` is retained as the plan identity in
    /// the resulting bridge.  The caller must have checked the authority's
    /// schema, source-token lineage, six fixed payload identities, and all
    /// thirty route descriptors before reaching this method.  In particular,
    /// no prior process-local PinnedBuffer or historical component receipt is
    /// accepted here.
    pub fn build_source_token_l1_all_ten_true_moe_source_bridge_from_validated_authority(
        &self,
        manifest_document_sha256: &str,
        route_authority_document_sha256: &str,
        route: Qwen80RouteSelection,
    ) -> Result<Qwen80AllTenTrueMoeSourceBridge> {
        require_canonical_sha256(
            manifest_document_sha256,
            "source-token Layer-1 manifest document SHA-256",
        )?;
        require_canonical_sha256(
            route_authority_document_sha256,
            "source-token Layer-1 route-authority document SHA-256",
        )?;
        route.validate()?;
        let layer = 1usize;
        let contract = self.canonical_linear_moe_operator_contract(layer)?;
        contract.validate_against_catalog(self)?;
        if contract.mixer.layer != layer || contract.mixer.linear_state_slot != 1 {
            return Err(model_error(
                "source-token Layer-1 bridge did not bind DeltaNet layer/state slot one",
            ));
        }
        let hybrid = self.complete_hybrid_decoder_plan(1)?;
        let bindings = hybrid.routed_expert_bindings(self, layer, &route)?;
        if bindings.len() != QWEN80_TOP_K {
            return Err(model_error(
                "source-token Layer-1 bridge did not resolve exactly ten routed experts",
            ));
        }
        let mut waves = Vec::with_capacity(QWEN80_TOP_K);
        for (index, (expected_id, bindings)) in route.ids.iter().zip(bindings).enumerate() {
            if bindings.expert != usize::from(*expected_id) {
                return Err(model_error(format!(
                    "source-token Layer-1 bridge expert {index} does not match the validated route"
                )));
            }
            let projection =
                |binding: &Qwen80PackedTensorBinding| -> Result<Qwen80AllTenRoutePlanProjection> {
                    Ok(Qwen80AllTenRoutePlanProjection {
                        tensor_name: binding.name.clone(),
                        artifact_sha256: self
                            .direct_tensor_artifact_sha256(&binding.name)?
                            .to_owned(),
                    })
                };
            waves.push(Qwen80AllTenRoutePlanWave {
                expert: *expected_id,
                normalized_weight: route.weights[index],
                gate: projection(&bindings.gate_proj)?,
                up: projection(&bindings.up_proj)?,
                down: projection(&bindings.down_proj)?,
            });
        }
        let plan = Qwen80AllTenRoutedExpertPlan {
            manifest_document_sha256: manifest_document_sha256.to_owned(),
            plan_document_sha256: route_authority_document_sha256.to_owned(),
            manifest_seal_sha256: self.manifest_seal().to_owned(),
            source_revision: self.config.source_revision.clone(),
            layer,
            route,
            waves,
        };
        self.build_all_ten_true_moe_source_bridge(&plan, self.first_residual_device_binding(layer)?)
    }
}

/// Resident compact route bodies and an externally produced DeltaNet
/// first-residual buffer for one future true-input L0 MoE graph.  Constructing
/// this type uploads only immutable scale/sign sections and route controls;
/// it does not encode a command buffer, evaluate a layer, or create a token.
#[cfg(target_os = "macos")]
#[derive(Clone)]
pub struct Qwen80AllTenTrueMoeDeviceBridge {
    source_bridge: Qwen80AllTenTrueMoeSourceBridge,
    first_residual: PinnedBuffer,
    route_gate_scales: PinnedBuffer,
    route_gate_signs: PinnedBuffer,
    route_up_scales: PinnedBuffer,
    route_up_signs: PinnedBuffer,
    route_down_scales: PinnedBuffer,
    route_down_signs: PinnedBuffer,
    expected_route_ids: PinnedBuffer,
    expected_route_weights: PinnedBuffer,
}

#[cfg(target_os = "macos")]
impl Qwen80AllTenTrueMoeDeviceBridge {
    pub fn source_bridge(&self) -> &Qwen80AllTenTrueMoeSourceBridge {
        &self.source_bridge
    }

    pub fn first_residual(&self) -> &PinnedBuffer {
        &self.first_residual
    }

    pub fn route_gate_scales(&self) -> &PinnedBuffer {
        &self.route_gate_scales
    }

    pub fn route_gate_signs(&self) -> &PinnedBuffer {
        &self.route_gate_signs
    }

    pub fn route_up_scales(&self) -> &PinnedBuffer {
        &self.route_up_scales
    }

    pub fn route_up_signs(&self) -> &PinnedBuffer {
        &self.route_up_signs
    }

    pub fn route_down_scales(&self) -> &PinnedBuffer {
        &self.route_down_scales
    }

    pub fn route_down_signs(&self) -> &PinnedBuffer {
        &self.route_down_signs
    }

    pub fn expected_route_ids(&self) -> &PinnedBuffer {
        &self.expected_route_ids
    }

    pub fn expected_route_weights(&self) -> &PinnedBuffer {
        &self.expected_route_weights
    }
}

#[cfg(target_os = "macos")]
impl Qwen80AllTenTrueMoeSourceBridge {
    /// Upload the exact already-admitted compact all-ten route body sections
    /// and attach the DeltaNet output buffer by handle, not by a copied host
    /// vector.  A future caller must pass the `PinnedBuffer` produced in the
    /// same command graph as the source layer's DeltaNet first residual, then
    /// use the returned buffers with the staged graph host.  This method has
    /// no dispatch/commit path and cannot create a runtime claim on its own.
    pub fn upload_with_first_residual(
        &self,
        context: &MetalContext,
        first_residual: PinnedBuffer,
    ) -> Result<Qwen80AllTenTrueMoeDeviceBridge> {
        self.route_payloads.validate()?;
        self.first_residual.validate_for(&self.route_payloads)?;
        let expected_first_residual_bytes = bytes_for_f32(
            self.first_residual.elements,
            "Qwen80 DeltaNet first-residual device bridge",
        )?;
        if first_residual.length() as usize != expected_first_residual_bytes {
            return Err(model_error(format!(
                "Qwen80 DeltaNet first-residual device buffer has {} bytes, expected {expected_first_residual_bytes}",
                first_residual.length()
            )));
        }
        let expected_route_ids = self
            .route_payloads
            .route
            .ids
            .iter()
            .map(|&id| u32::from(id))
            .collect::<Vec<_>>();
        let expected_route_weights = self.route_payloads.route.weights;
        Ok(Qwen80AllTenTrueMoeDeviceBridge {
            source_bridge: self.clone(),
            first_residual,
            route_gate_scales: context
                .new_buffer_with_bytes_checked(self.route_payloads.gate_scales())?,
            route_gate_signs: context
                .new_buffer_with_bytes_checked(self.route_payloads.gate_signs())?,
            route_up_scales: context
                .new_buffer_with_bytes_checked(self.route_payloads.up_scales())?,
            route_up_signs: context
                .new_buffer_with_bytes_checked(self.route_payloads.up_signs())?,
            route_down_scales: context
                .new_buffer_with_bytes_checked(self.route_payloads.down_scales())?,
            route_down_signs: context
                .new_buffer_with_bytes_checked(self.route_payloads.down_signs())?,
            expected_route_ids: context
                .new_buffer_with_bytes_checked(bytemuck::cast_slice(&expected_route_ids))?,
            expected_route_weights: context
                .new_buffer_with_bytes_checked(bytemuck::cast_slice(&expected_route_weights))?,
        })
    }
}

fn source_qwen80_residual_rms_norm(input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
    if input.len() != weight.len() || input.is_empty() {
        return Err(model_error(
            "Qwen80 residual RMSNorm input/weight geometry is invalid",
        ));
    }
    let variance = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inverse_rms = (variance + QWEN80_RMS_EPS).sqrt().recip();
    let output = input
        .iter()
        .zip(weight)
        .map(|(&value, &weight)| value * inverse_rms * (1.0 + weight))
        .collect::<Vec<_>>();
    if output.iter().any(|value| !value.is_finite()) {
        return Err(model_error(
            "Qwen80 residual RMSNorm produced non-finite output",
        ));
    }
    Ok(output)
}

fn source_qwen80_gated_rms_norm(
    input: &[f32],
    gate: &[f32],
    repeated_weight: &[f32],
    heads: usize,
    value_head_dim: usize,
) -> Result<Vec<f32>> {
    let expected_elements = heads
        .checked_mul(value_head_dim)
        .ok_or_else(|| model_error("Qwen80 gated RMSNorm source head geometry overflows"))?;
    if heads == 0
        || value_head_dim == 0
        || input.len() != expected_elements
        || input.len() != gate.len()
        || input.len() != repeated_weight.len()
    {
        return Err(model_error(
            "Qwen80 gated RMSNorm input/gate/weight geometry is invalid",
        ));
    }
    // `Qwen3NextGatedDeltaNet.forward` reshapes both tensors to
    // `[-1, head_v_dim]` before `Qwen3NextRMSNormGated`, whose `.mean(-1)`
    // is therefore per value head, never across the flattened 4096-wide
    // DeltaNet body. Keep this source-layout rule independent of the Metal
    // implementation so a shared-oracle error cannot disguise a shader bug.
    let mut output = vec![0.0f32; expected_elements];
    for head in 0..heads {
        let base = head * value_head_dim;
        let end = base + value_head_dim;
        let values = &input[base..end];
        let variance =
            values.iter().map(|value| value * value).sum::<f32>() / value_head_dim as f32;
        let inverse_rms = (variance + QWEN80_RMS_EPS).sqrt().recip();
        for index in base..end {
            let gate = gate[index];
            let silu = gate / (1.0 + (-gate).exp());
            output[index] = input[index] * inverse_rms * repeated_weight[index] * silu;
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(model_error(
            "Qwen80 gated RMSNorm produced non-finite output",
        ));
    }
    Ok(output)
}

fn source_qwen80_ba_to_decay_beta(
    ba: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    layout: &Qwen80CanonicalLinearDeltaNetLayout,
) -> Result<(Vec<f32>, Vec<f32>)> {
    layout.validate()?;
    if ba.len() != layout.ba_projection_elements()?
        || a_log.len() != layout.value_heads
        || dt_bias.len() != layout.value_heads
    {
        return Err(model_error("Qwen80 BA source control geometry is invalid"));
    }
    let mut decay = vec![0.0f32; layout.value_heads];
    let mut beta = vec![0.0f32; layout.value_heads];
    for value_head in 0..layout.value_heads {
        let key_head = value_head / layout.value_heads_per_key_head;
        let within_key_head = value_head % layout.value_heads_per_key_head;
        let b =
            ba[layout.ba_row_offset(key_head, Qwen80LinearBaSegment::BetaLogit)? + within_key_head];
        let a = ba
            [layout.ba_row_offset(key_head, Qwen80LinearBaSegment::DecayInput)? + within_key_head];
        let x = a + dt_bias[value_head];
        let softplus = x.max(0.0) + (-x.abs()).exp().ln_1p();
        let g = -a_log[value_head].exp() * softplus;
        decay[value_head] = g.exp();
        beta[value_head] = 1.0 / (1.0 + (-b).exp());
        if !decay[value_head].is_finite()
            || !beta[value_head].is_finite()
            || decay[value_head] <= 0.0
            || decay[value_head] > 1.0
        {
            return Err(model_error(format!(
                "Qwen80 BA source control is invalid at value head {value_head}"
            )));
        }
    }
    Ok((decay, beta))
}

fn source_qwen80_l2_normalize(values: &mut [f32], scale: f32) -> Result<()> {
    let norm = values.iter().map(|value| value * value).sum::<f32>();
    let inverse_norm = (norm + QWEN80_RMS_EPS).sqrt().recip() * scale;
    for value in values.iter_mut() {
        *value *= inverse_norm;
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(model_error(
            "Qwen80 DeltaNet L2 normalization became non-finite",
        ));
    }
    Ok(())
}

fn source_qwen80_recurrent_deltanet(
    state: &mut [f32],
    query: &[f32],
    key: &[f32],
    value: &[f32],
    decay: &[f32],
    beta: &[f32],
    layout: &Qwen80CanonicalLinearDeltaNetLayout,
) -> Result<Vec<f32>> {
    layout.validate()?;
    if state.len() != layout.recurrent_state_elements()?
        || query.len() != layout.value_heads * layout.key_head_dim
        || key.len() != layout.value_heads * layout.key_head_dim
        || value.len() != layout.value_elements()?
        || decay.len() != layout.value_heads
        || beta.len() != layout.value_heads
    {
        return Err(model_error("Qwen80 recurrent DeltaNet geometry is invalid"));
    }
    let mut output = vec![0.0f32; layout.value_elements()?];
    for head in 0..layout.value_heads {
        let state_base = head * layout.recurrent_state_head_stride;
        let key_base = head * layout.key_head_dim;
        let value_base = head * layout.value_head_dim;
        for value_index in 0..layout.value_head_dim {
            let mut kv_memory = 0.0f32;
            for key_index in 0..layout.key_head_dim {
                let index =
                    state_base + key_index * layout.recurrent_state_key_stride + value_index;
                state[index] *= decay[head];
                kv_memory += state[index] * key[key_base + key_index];
            }
            let delta = (value[value_base + value_index] - kv_memory) * beta[head];
            for key_index in 0..layout.key_head_dim {
                let index =
                    state_base + key_index * layout.recurrent_state_key_stride + value_index;
                state[index] += key[key_base + key_index] * delta;
            }
        }
        for value_index in 0..layout.value_head_dim {
            let mut sum = 0.0f32;
            for key_index in 0..layout.key_head_dim {
                sum += state
                    [state_base + key_index * layout.recurrent_state_key_stride + value_index]
                    * query[key_base + key_index];
            }
            output[value_base + value_index] = sum;
        }
    }
    if output.iter().any(|value| !value.is_finite()) || state.iter().any(|value| !value.is_finite())
    {
        return Err(model_error(
            "Qwen80 recurrent DeltaNet produced a non-finite value",
        ));
    }
    Ok(output)
}

fn source_qwen80_split_linear_qkvz(
    projection: &[f32],
    layout: &Qwen80CanonicalLinearDeltaNetLayout,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    layout.validate()?;
    let value_dim_per_key_head = layout.value_heads_per_key_head * layout.value_head_dim;
    if projection.len() != layout.qkvz_projection_elements()? {
        return Err(model_error(
            "Qwen80 QKVZ source projection geometry is invalid",
        ));
    }
    let mut query = vec![0.0f32; layout.key_elements()?];
    let mut key = vec![0.0f32; layout.key_elements()?];
    let mut value = vec![0.0f32; layout.value_elements()?];
    let mut z = vec![0.0f32; layout.value_elements()?];
    for key_head in 0..layout.key_heads {
        let query_source = layout.qkvz_row_offset(key_head, Qwen80LinearQkvzSegment::Query)?;
        let key_source = layout.qkvz_row_offset(key_head, Qwen80LinearQkvzSegment::Key)?;
        let value_source = layout.qkvz_row_offset(key_head, Qwen80LinearQkvzSegment::Value)?;
        let z_source = layout.qkvz_row_offset(key_head, Qwen80LinearQkvzSegment::Z)?;
        query[key_head * layout.key_head_dim..(key_head + 1) * layout.key_head_dim]
            .copy_from_slice(&projection[query_source..query_source + layout.key_head_dim]);
        key[key_head * layout.key_head_dim..(key_head + 1) * layout.key_head_dim]
            .copy_from_slice(&projection[key_source..key_source + layout.key_head_dim]);
        let value_destination = key_head * value_dim_per_key_head;
        value[value_destination..value_destination + value_dim_per_key_head]
            .copy_from_slice(&projection[value_source..value_source + value_dim_per_key_head]);
        z[value_destination..value_destination + value_dim_per_key_head]
            .copy_from_slice(&projection[z_source..z_source + value_dim_per_key_head]);
    }
    Ok((query, key, value, z))
}

fn source_qwen80_causal_conv_step(
    mixed_qkv: &[f32],
    prior_state: &[f32],
    conv: &Qwen80CpuPackedTensor,
    mode: Qwen80CpuPackedReadMode,
    layout: &Qwen80CanonicalLinearDeltaNetLayout,
) -> Result<(Vec<f32>, Vec<f32>)> {
    layout.validate()?;
    if mixed_qkv.len() != layout.conv_channels
        || prior_state.len() != layout.conv_state_elements()?
        || conv.header.shape != [layout.conv_channels, 1, layout.conv_kernel]
    {
        return Err(model_error("Qwen80 causal convolution geometry is invalid"));
    }
    let conv_weights = match mode {
        Qwen80CpuPackedReadMode::StreamingDirect => None,
        Qwen80CpuPackedReadMode::MaterializedReference => {
            let (header, values) = decode_complete_binary_f32(&conv.payload)?;
            if header != conv.header {
                return Err(model_error(
                    "Qwen80 materialized causal convolution header drifted",
                ));
            }
            Some(values)
        }
    };
    let mut output = vec![0.0f32; layout.conv_channels];
    let mut next_state = vec![0.0f32; prior_state.len()];
    for channel in 0..layout.conv_channels {
        let state_base = channel * layout.conv_state_tokens;
        let mut sum = 0.0f32;
        for tap in 0..layout.conv_state_tokens {
            let weight_index = channel * layout.conv_kernel + tap;
            let weight = match &conv_weights {
                Some(values) => values[weight_index],
                None => conv.value(weight_index)?,
            };
            sum += prior_state[state_base + tap] * weight;
            if tap + 1 < layout.conv_state_tokens {
                next_state[state_base + tap] = prior_state[state_base + tap + 1];
            } else {
                next_state[state_base + tap] = mixed_qkv[channel];
            }
        }
        let newest_weight_index = channel * layout.conv_kernel + layout.conv_state_tokens;
        let newest_weight = match &conv_weights {
            Some(values) => values[newest_weight_index],
            None => conv.value(newest_weight_index)?,
        };
        sum += mixed_qkv[channel] * newest_weight;
        output[channel] = sum / (1.0 + (-sum).exp());
    }
    if output.iter().any(|value| !value.is_finite())
        || next_state.iter().any(|value| !value.is_finite())
    {
        return Err(model_error(
            "Qwen80 causal convolution produced a non-finite value",
        ));
    }
    Ok((output, next_state))
}

#[cfg(test)]
fn max_abs_error_cpu(expected: &[f32], observed: &[f32], label: &str) -> Result<f32> {
    if expected.len() != observed.len() {
        return Err(model_error(format!(
            "{label} reference length {} differs from observed {}",
            expected.len(),
            observed.len()
        )));
    }
    let mut error = 0.0f32;
    for (index, (&expected, &observed)) in expected.iter().zip(observed).enumerate() {
        if !expected.is_finite() || !observed.is_finite() {
            return Err(model_error(format!(
                "{label} reference is non-finite at {index}"
            )));
        }
        error = error.max((expected - observed).abs());
    }
    Ok(error)
}

impl Qwen80CompleteArtifactCatalog {
    /// Resolve one source-addressable token embedding through the immutable
    /// direct-packed catalog. The tokenizer namespace is intentionally
    /// smaller than the 151,936-row embedding/lm-head body: the final 267
    /// rows are source-reserved and must be rejected before any state or
    /// native backend can observe them.
    ///
    /// This is deliberately a compact CPU/reference oracle. It supplies a
    /// parity target for a future native embedding gather but cannot serve as
    /// a production decoder fallback.
    pub fn execute_embedding_lookup_cpu_oracle(
        &self,
        token_id: u32,
    ) -> Result<Qwen80DirectPackedEmbeddingCpuOracleResult> {
        if token_id as usize >= QWEN80_TOKENIZER_VOCAB {
            return Err(model_error(format!(
                "Qwen80 direct-packed embedding rejects token {token_id}; source tokenizer namespace is 0..{} and rows {}..{} are reserved",
                QWEN80_TOKENIZER_VOCAB.saturating_sub(1),
                QWEN80_TOKENIZER_VOCAB,
                QWEN80_VOCAB.saturating_sub(1),
            )));
        }
        let binding = Qwen80CompleteHybridDecoderPlan::binding(
            self,
            "model.embed_tokens.weight",
            &[QWEN80_VOCAB, QWEN80_HIDDEN],
        )?;
        let embedding = Qwen80CpuPackedTensor::from_binding(self, &binding)?;
        let hidden = embedding.row(
            token_id as usize,
            QWEN80_VOCAB,
            QWEN80_HIDDEN,
            Qwen80CpuPackedReadMode::StreamingDirect,
        )?;
        if hidden.len() != QWEN80_HIDDEN || hidden.iter().any(|value| !value.is_finite()) {
            return Err(model_error(
                "Qwen80 direct-packed embedding lookup did not yield 2048 finite hidden values",
            ));
        }
        Ok(Qwen80DirectPackedEmbeddingCpuOracleResult {
            token_id,
            direct_packed_embedding_tensor: binding.name,
            hidden,
            source_algorithm_boundary: "source-token id below the exact 151669-token namespace -> one direct-packed model.embed_tokens.weight row [2048]; no native gather, layer, decoder, generation, HCLI, or TPS execution".to_owned(),
        })
    }

    /// Execute one canonical source-shaped layer-0 Gated DeltaNet mixer on a
    /// CPU reference path that reads only immutable admission-verified compact
    /// payloads.  It is a bridge-development/parity oracle, never a
    /// production model fallback: it stops immediately after the mixer output
    /// projection and first residual add, before post norm or any MoE work.
    pub fn execute_first_linear_layer_cpu_oracle(
        &self,
        input: &Qwen80CanonicalLinearLayerCpuInput,
    ) -> Result<Qwen80CanonicalLinearLayerCpuOracleResult> {
        let contract = self.canonical_linear_deltanet_operator_contract(0)?;
        self.execute_canonical_linear_deltanet_cpu_oracle(&contract, input)
    }

    /// Execute the bounded CPU source oracle through an explicit artifact and
    /// resource contract. This accepts only a source-scheduled DeltaNet
    /// contract built by this catalog; it is not a device fallback and stops
    /// before post-norm or MoE work.
    pub fn execute_canonical_linear_deltanet_cpu_oracle(
        &self,
        contract: &Qwen80CanonicalLinearDeltaNetOperatorContract,
        input: &Qwen80CanonicalLinearLayerCpuInput,
    ) -> Result<Qwen80CanonicalLinearLayerCpuOracleResult> {
        self.execute_canonical_linear_deltanet_cpu_impl(
            contract,
            input,
            Qwen80CpuPackedReadMode::StreamingDirect,
        )
    }

    fn execute_canonical_linear_deltanet_cpu_impl(
        &self,
        contract: &Qwen80CanonicalLinearDeltaNetOperatorContract,
        input: &Qwen80CanonicalLinearLayerCpuInput,
        mode: Qwen80CpuPackedReadMode,
    ) -> Result<Qwen80CanonicalLinearLayerCpuOracleResult> {
        input.validate()?;
        contract.validate_against_catalog(self)?;
        let layout = &contract.layout;
        let input_layernorm = Qwen80CpuPackedTensor::from_binding(self, &contract.input_layernorm)?;
        let qkvz = Qwen80CpuPackedTensor::from_binding(self, &contract.mixer.in_proj_qkvz)?;
        let ba = Qwen80CpuPackedTensor::from_binding(self, &contract.mixer.in_proj_ba)?;
        let conv = Qwen80CpuPackedTensor::from_binding(self, &contract.mixer.causal_conv1d)?;
        let a_log = Qwen80CpuPackedTensor::from_binding(self, &contract.mixer.a_log)?;
        let dt_bias = Qwen80CpuPackedTensor::from_binding(self, &contract.mixer.dt_bias)?;
        let gated_norm = Qwen80CpuPackedTensor::from_binding(self, &contract.mixer.gated_rms_norm)?;
        let out_proj = Qwen80CpuPackedTensor::from_binding(self, &contract.mixer.out_proj)?;

        let input_weight = input_layernorm.vector(layout.hidden_elements, mode)?;
        let input_rms_norm_output = source_qwen80_residual_rms_norm(&input.hidden, &input_weight)?;
        let projected_qkvz = qkvz.matvec(
            &input_rms_norm_output,
            layout.qkvz_projection_elements()?,
            layout.hidden_elements,
            mode,
        )?;
        let projected_ba = ba.matvec(
            &input_rms_norm_output,
            layout.ba_projection_elements()?,
            layout.hidden_elements,
            mode,
        )?;
        let (raw_query, raw_key, raw_value, z) =
            source_qwen80_split_linear_qkvz(&projected_qkvz, layout)?;
        let mut mixed_qkv = Vec::with_capacity(layout.conv_channels);
        mixed_qkv.extend_from_slice(&raw_query);
        mixed_qkv.extend_from_slice(&raw_key);
        mixed_qkv.extend_from_slice(&raw_value);
        let (convolved_qkv, next_conv_state) = source_qwen80_causal_conv_step(
            &mixed_qkv,
            &input.state.conv_state,
            &conv,
            mode,
            layout,
        )?;
        let raw_query_len = layout.key_elements()?;
        let raw_key_len = raw_query_len;
        let raw_value_len = layout.value_elements()?;
        let convolved_query = &convolved_qkv[..raw_query_len];
        let convolved_key = &convolved_qkv[raw_query_len..raw_query_len + raw_key_len];
        let convolved_value = &convolved_qkv[raw_query_len + raw_key_len..];
        if convolved_value.len() != raw_value_len {
            return Err(model_error(
                "Qwen80 convolution did not preserve value geometry",
            ));
        }
        let convolved_value = convolved_value.to_vec();
        let mut repeated_query = vec![0.0f32; raw_value_len];
        let mut repeated_key = vec![0.0f32; raw_value_len];
        for value_head in 0..layout.value_heads {
            let key_head = value_head / layout.value_heads_per_key_head;
            let mut query_head = convolved_query
                [key_head * layout.key_head_dim..(key_head + 1) * layout.key_head_dim]
                .to_vec();
            let mut key_head_values = convolved_key
                [key_head * layout.key_head_dim..(key_head + 1) * layout.key_head_dim]
                .to_vec();
            source_qwen80_l2_normalize(
                &mut query_head,
                (layout.key_head_dim as f32).sqrt().recip(),
            )?;
            source_qwen80_l2_normalize(&mut key_head_values, 1.0)?;
            let destination = value_head * layout.key_head_dim;
            repeated_query[destination..destination + layout.key_head_dim]
                .copy_from_slice(&query_head);
            repeated_key[destination..destination + layout.key_head_dim]
                .copy_from_slice(&key_head_values);
        }
        let a_log_values = a_log.vector(layout.value_heads, mode)?;
        let dt_bias_values = dt_bias.vector(layout.value_heads, mode)?;
        let (decay, beta) =
            source_qwen80_ba_to_decay_beta(&projected_ba, &a_log_values, &dt_bias_values, layout)?;
        let mut next_recurrent_state = input.state.recurrent_state.clone();
        let recurrent_output = source_qwen80_recurrent_deltanet(
            &mut next_recurrent_state,
            &repeated_query,
            &repeated_key,
            &convolved_value,
            &decay,
            &beta,
            layout,
        )?;
        let gated_norm_weight = gated_norm.vector(layout.value_head_dim, mode)?;
        let repeated_gated_norm_weight = (0..layout.value_heads)
            .flat_map(|_| gated_norm_weight.iter().copied())
            .collect::<Vec<_>>();
        let gated_output = source_qwen80_gated_rms_norm(
            &recurrent_output,
            &z,
            &repeated_gated_norm_weight,
            layout.value_heads,
            layout.value_head_dim,
        )?;
        let mixer_output =
            out_proj.matvec(&gated_output, layout.hidden_elements, raw_value_len, mode)?;
        let mixer_residual_output = input
            .hidden
            .iter()
            .zip(&mixer_output)
            .map(|(&residual, &mixer)| residual + mixer)
            .collect::<Vec<_>>();
        if mixer_residual_output.iter().any(|value| !value.is_finite()) {
            return Err(model_error(
                "Qwen80 canonical linear mixer residual produced non-finite output",
            ));
        }
        Ok(Qwen80CanonicalLinearLayerCpuOracleResult {
            layer: contract.layer,
            linear_state_slot: contract.linear_state_slot,
            direct_packed_input_layernorm_tensor: input_layernorm.name,
            direct_packed_qkvz_tensor: qkvz.name,
            direct_packed_ba_tensor: ba.name,
            direct_packed_conv_tensor: conv.name,
            direct_packed_gated_norm_tensor: gated_norm.name,
            direct_packed_out_proj_tensor: out_proj.name,
            input_rms_norm_output,
            projected_qkvz,
            projected_ba,
            repeated_query_l2_scaled: repeated_query,
            repeated_key_l2: repeated_key,
            convolved_value,
            z,
            decay,
            beta,
            recurrent_output,
            gated_rms_norm_output: gated_output,
            mixer_output,
            mixer_residual_output,
            next_state: Qwen80CanonicalLinearLayerCpuState {
                conv_state: next_conv_state,
                recurrent_state: next_recurrent_state,
            },
            source_algorithm_boundary: format!("layer-{} input RMSNorm -> direct packed QKVZ/BA -> source [key-head][Q,K,V,Z] rearrange -> causal depthwise SiLU convolution -> repeated Q/K L2 normalisation -> BA/A_log/dt_bias DeltaNet recurrence -> gated RMSNorm with Z -> direct packed out projection -> first residual only; post-attention norm and routed/shared MoE are intentionally absent", contract.layer),
        })
    }

    /// Execute one full source-shaped layer-0 DeltaNet + sparse-MoE CPU
    /// oracle. It reads only immutable admission-verified compact payloads,
    /// including exactly the experts selected by the source top-10 router.
    /// This is a bounded parity/development oracle, never a BF16/MPS or CPU
    /// production fallback; it stops after layer zero's second residual.
    pub fn execute_first_linear_layer_cpu_moe_oracle(
        &self,
        input: &Qwen80CanonicalLinearLayerCpuInput,
    ) -> Result<Qwen80CanonicalLinearMoECpuOracleResult> {
        let contract = self.canonical_linear_moe_operator_contract(0)?;
        self.execute_canonical_linear_moe_cpu_oracle(&contract, input)
    }

    /// Execute the post-mixer source MoE path through the explicit artifact
    /// contract. A caller cannot swap in a different manifest/revision or an
    /// arbitrary expert filename: validation derives expert bodies only after
    /// the exact compact router result exists.
    pub fn execute_canonical_linear_moe_cpu_oracle(
        &self,
        contract: &Qwen80CanonicalLinearMoEOperatorContract,
        input: &Qwen80CanonicalLinearLayerCpuInput,
    ) -> Result<Qwen80CanonicalLinearMoECpuOracleResult> {
        self.execute_canonical_linear_moe_cpu_impl(
            contract,
            input,
            Qwen80CpuPackedReadMode::StreamingDirect,
        )
    }

    fn execute_canonical_linear_moe_cpu_impl(
        &self,
        contract: &Qwen80CanonicalLinearMoEOperatorContract,
        input: &Qwen80CanonicalLinearLayerCpuInput,
        mode: Qwen80CpuPackedReadMode,
    ) -> Result<Qwen80CanonicalLinearMoECpuOracleResult> {
        input.validate()?;
        contract.validate_against_catalog(self)?;
        let mixer =
            self.execute_canonical_linear_deltanet_cpu_impl(&contract.mixer, input, mode)?;

        let post_attention_layernorm =
            Qwen80CpuPackedTensor::from_binding(self, &contract.post_attention_layernorm)?;
        let router = Qwen80CpuPackedTensor::from_binding(self, &contract.router)?;
        let shared_gate_proj =
            Qwen80CpuPackedTensor::from_binding(self, &contract.shared_gate_proj)?;
        let shared_up_proj = Qwen80CpuPackedTensor::from_binding(self, &contract.shared_up_proj)?;
        let shared_down_proj =
            Qwen80CpuPackedTensor::from_binding(self, &contract.shared_down_proj)?;
        let shared_expert_gate =
            Qwen80CpuPackedTensor::from_binding(self, &contract.shared_expert_gate)?;

        let post_norm_weight = post_attention_layernorm.vector(QWEN80_HIDDEN, mode)?;
        let post_attention_rms_norm_output =
            source_qwen80_residual_rms_norm(&mixer.mixer_residual_output, &post_norm_weight)?;

        // Preserve the source SparseMoeBlock ordering: shared MLP first,
        // router and all routed experts next, then shared gate and combine.
        let shared_mlp = source_qwen80_mlp(
            &shared_gate_proj,
            &shared_up_proj,
            &shared_down_proj,
            &post_attention_rms_norm_output,
            QWEN80_SHARED_EXPERT_INTERMEDIATE,
            mode,
        )?;
        let router_logits = router.matvec(
            &post_attention_rms_norm_output,
            QWEN80_EXPERTS,
            QWEN80_HIDDEN,
            mode,
        )?;
        let route = source_qwen80_topk_router(&router_logits)?;
        let expert_bindings = contract.routed_expert_bindings(self, &route)?;
        let mut routed_expert_sum = vec![0.0f32; QWEN80_HIDDEN];
        let mut routed_experts = Vec::with_capacity(QWEN80_TOP_K);
        for ((expert, &route_weight), &route_id) in expert_bindings
            .iter()
            .zip(route.weights.iter())
            .zip(route.ids.iter())
        {
            if expert.expert != route_id as usize {
                return Err(model_error(
                    "Qwen80 routed expert binding no longer matches its source router id",
                ));
            }
            let gate_proj = Qwen80CpuPackedTensor::from_binding(self, &expert.gate_proj)?;
            let up_proj = Qwen80CpuPackedTensor::from_binding(self, &expert.up_proj)?;
            let down_proj = Qwen80CpuPackedTensor::from_binding(self, &expert.down_proj)?;
            let mlp = source_qwen80_mlp(
                &gate_proj,
                &up_proj,
                &down_proj,
                &post_attention_rms_norm_output,
                QWEN80_MOE_INTERMEDIATE,
                mode,
            )?;
            let weighted_output = mlp
                .output
                .iter()
                .map(|&value| value * route_weight)
                .collect::<Vec<_>>();
            if weighted_output.iter().any(|value| !value.is_finite()) {
                return Err(model_error(format!(
                    "Qwen80 routed expert {} produced a non-finite weighted output",
                    expert.expert
                )));
            }
            for (sum, value) in routed_expert_sum.iter_mut().zip(&weighted_output) {
                *sum += value;
            }
            routed_experts.push(Qwen80RoutedExpertCpuOracleResult {
                expert: expert.expert,
                route_weight,
                direct_packed_gate_proj_tensor: gate_proj.name,
                direct_packed_up_proj_tensor: up_proj.name,
                direct_packed_down_proj_tensor: down_proj.name,
                gate_projection: mlp.gate_projection,
                up_projection: mlp.up_projection,
                gated_up_product: mlp.gated_up_product,
                output: mlp.output,
                weighted_output,
            });
        }
        if routed_experts.len() != QWEN80_TOP_K
            || routed_expert_sum.iter().any(|value| !value.is_finite())
        {
            return Err(model_error(
                "Qwen80 routed top-10 wave did not produce a finite complete source sum",
            ));
        }

        let shared_expert_gate_logit = shared_expert_gate
            .matvec(&post_attention_rms_norm_output, 1, QWEN80_HIDDEN, mode)?
            .into_iter()
            .next()
            .ok_or_else(|| model_error("Qwen80 shared expert gate returned no logit"))?;
        let shared_expert_gate_value = 1.0 / (1.0 + (-shared_expert_gate_logit).exp());
        if !shared_expert_gate_value.is_finite() || !(0.0..=1.0).contains(&shared_expert_gate_value)
        {
            return Err(model_error("Qwen80 shared expert gate sigmoid is invalid"));
        }
        let shared_gated_output = shared_mlp
            .output
            .iter()
            .map(|&value| value * shared_expert_gate_value)
            .collect::<Vec<_>>();
        let moe_output = routed_expert_sum
            .iter()
            .zip(&shared_gated_output)
            .map(|(&routed, &shared)| routed + shared)
            .collect::<Vec<_>>();
        let layer_output = mixer
            .mixer_residual_output
            .iter()
            .zip(&moe_output)
            .map(|(&residual, &moe)| residual + moe)
            .collect::<Vec<_>>();
        if shared_gated_output.iter().any(|value| !value.is_finite())
            || moe_output.iter().any(|value| !value.is_finite())
            || layer_output.iter().any(|value| !value.is_finite())
        {
            return Err(model_error(
                "Qwen80 canonical linear MoE/residual produced a non-finite value",
            ));
        }

        Ok(Qwen80CanonicalLinearMoECpuOracleResult {
            direct_packed_post_attention_layernorm_tensor: post_attention_layernorm.name,
            direct_packed_router_tensor: router.name,
            direct_packed_shared_gate_proj_tensor: shared_gate_proj.name,
            direct_packed_shared_up_proj_tensor: shared_up_proj.name,
            direct_packed_shared_down_proj_tensor: shared_down_proj.name,
            direct_packed_shared_expert_gate_tensor: shared_expert_gate.name,
            mixer,
            post_attention_rms_norm_output,
            router_logits,
            route,
            routed_experts,
            routed_expert_sum,
            shared_gate_projection: shared_mlp.gate_projection,
            shared_up_projection: shared_mlp.up_projection,
            shared_gated_up_product: shared_mlp.gated_up_product,
            shared_expert_output: shared_mlp.output,
            shared_expert_gate_logit,
            shared_expert_gate_value,
            shared_gated_output,
            moe_output,
            layer_output,
            source_algorithm_boundary: format!(
                "layer-{} direct-packed input RMSNorm -> Gated DeltaNet mixer -> first residual -> direct-packed post-attention RMSNorm -> source top-10 router -> all ten routed gate/up/down waves weighted by normalized router probabilities -> source shared gate/up/down MLP -> sigmoid shared gate -> MoE combine -> second residual only; later layers, final norm, lm_head, sampler, feedback, HCLI, and TPS are absent",
                contract.mixer.layer
            ),
        })
    }

    /// Execute exactly the descriptor-selected ten routed expert bodies from
    /// an already-normalized hidden vector.  This is the generic CPU parity
    /// half of the future one-graph Metal capture: it performs no router,
    /// shared expert, residual, token, or model-loop work.
    ///
    /// The input must be the exact post-attention normalized hidden vector
    /// bound by a separate strict router receipt.  It is intentionally not a
    /// convenience production fallback for arbitrary text/model input.
    pub fn execute_all_ten_routed_expert_cpu_oracle(
        &self,
        plan: &Qwen80AllTenRoutedExpertPlan,
        normalized_hidden: &[f32],
    ) -> Result<Qwen80AllTenRoutedExpertCpuOracleResult> {
        if normalized_hidden.len() != QWEN80_HIDDEN
            || normalized_hidden.iter().any(|value| !value.is_finite())
        {
            return Err(model_error(format!(
                "all-ten routed-expert CPU oracle requires {QWEN80_HIDDEN} finite normalized-hidden values"
            )));
        }
        plan.validate_against_catalog(self)?;
        let normalized_hidden_sha256 =
            qwen80_f32_vector_sha256(normalized_hidden, "all-ten normalized hidden")?;
        let mut routed_expert_sum = vec![0.0f32; QWEN80_HIDDEN];
        let mut witnesses = Vec::with_capacity(QWEN80_TOP_K);
        for (wave_index, wave) in plan.waves.iter().enumerate() {
            let gate_binding = Qwen80CompleteHybridDecoderPlan::binding(
                self,
                wave.gate.tensor_name.clone(),
                &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
            )?;
            let up_binding = Qwen80CompleteHybridDecoderPlan::binding(
                self,
                wave.up.tensor_name.clone(),
                &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN],
            )?;
            let down_binding = Qwen80CompleteHybridDecoderPlan::binding(
                self,
                wave.down.tensor_name.clone(),
                &[QWEN80_HIDDEN, QWEN80_MOE_INTERMEDIATE],
            )?;
            let gate_proj = Qwen80CpuPackedTensor::from_binding(self, &gate_binding)?;
            let up_proj = Qwen80CpuPackedTensor::from_binding(self, &up_binding)?;
            let down_proj = Qwen80CpuPackedTensor::from_binding(self, &down_binding)?;
            let mlp = source_qwen80_mlp(
                &gate_proj,
                &up_proj,
                &down_proj,
                normalized_hidden,
                QWEN80_MOE_INTERMEDIATE,
                Qwen80CpuPackedReadMode::StreamingDirect,
            )?;
            let weighted_output = mlp
                .output
                .iter()
                .map(|&value| value * wave.normalized_weight)
                .collect::<Vec<_>>();
            if weighted_output.iter().any(|value| !value.is_finite()) {
                return Err(model_error(format!(
                    "all-ten routed-expert wave {wave_index} generated non-finite weighted output"
                )));
            }
            for (sum, value) in routed_expert_sum.iter_mut().zip(&weighted_output) {
                *sum += value;
            }
            witnesses.push(Qwen80AllTenRoutedExpertCpuWitness {
                wave_index,
                expert: wave.expert as usize,
                normalized_weight: wave.normalized_weight,
                direct_packed_gate_artifact_sha256: wave.gate.artifact_sha256.clone(),
                direct_packed_up_artifact_sha256: wave.up.artifact_sha256.clone(),
                direct_packed_down_artifact_sha256: wave.down.artifact_sha256.clone(),
                weighted_output_sha256: qwen80_f32_vector_sha256(
                    &weighted_output,
                    "all-ten weighted routed-expert output",
                )?,
                oracle: Qwen80RoutedExpertCpuOracleResult {
                    expert: wave.expert as usize,
                    route_weight: wave.normalized_weight,
                    direct_packed_gate_proj_tensor: gate_proj.name,
                    direct_packed_up_proj_tensor: up_proj.name,
                    direct_packed_down_proj_tensor: down_proj.name,
                    gate_projection: mlp.gate_projection,
                    up_projection: mlp.up_projection,
                    gated_up_product: mlp.gated_up_product,
                    output: mlp.output,
                    weighted_output,
                },
            });
        }
        if witnesses.len() != QWEN80_TOP_K
            || routed_expert_sum.iter().any(|value| !value.is_finite())
        {
            return Err(model_error(
                "all-ten routed-expert CPU oracle did not retain ten finite route witnesses",
            ));
        }
        let routed_expert_sum_sha256 =
            qwen80_f32_vector_sha256(&routed_expert_sum, "all-ten routed-expert sum")?;
        Ok(Qwen80AllTenRoutedExpertCpuOracleResult {
            layer: plan.layer,
            manifest_document_sha256: plan.manifest_document_sha256.clone(),
            plan_document_sha256: plan.plan_document_sha256.clone(),
            normalized_hidden_sha256,
            route: plan.route.clone(),
            witnesses,
            routed_expert_sum,
            routed_expert_sum_sha256,
            source_algorithm_boundary: format!(
                "layer-{} source-selected all-ten routed direct-packed gate/up -> SiLU(gate)*up -> down -> source-normalized weight waves only; postnorm/router evidence is consumed externally and shared expert, route combine, second residual, later layers, token feedback, HCLI, and TPS remain absent",
                plan.layer
            ),
        })
    }

    #[cfg(test)]
    fn execute_first_linear_layer_cpu_materialized_reference(
        &self,
        input: &Qwen80CanonicalLinearLayerCpuInput,
    ) -> Result<Qwen80CanonicalLinearLayerCpuOracleResult> {
        let contract = self.canonical_linear_deltanet_operator_contract(0)?;
        self.execute_canonical_linear_deltanet_cpu_impl(
            &contract,
            input,
            Qwen80CpuPackedReadMode::MaterializedReference,
        )
    }

    #[cfg(test)]
    fn execute_first_linear_layer_cpu_moe_materialized_reference(
        &self,
        input: &Qwen80CanonicalLinearLayerCpuInput,
    ) -> Result<Qwen80CanonicalLinearMoECpuOracleResult> {
        let contract = self.canonical_linear_moe_operator_contract(0)?;
        self.execute_canonical_linear_moe_cpu_impl(
            &contract,
            input,
            Qwen80CpuPackedReadMode::MaterializedReference,
        )
    }
}

/// Completed control-flow facts from one backend-executed scheduled token.
/// This is intentionally not a capability/TPS receipt: the scheduler cannot
/// tell whether a backend used native Metal, a test recorder, or an invalid
/// shadow unless the caller separately supplies backend evidence.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen80HybridScheduledToken {
    pub position: usize,
    pub input_token_id: u32,
    pub sampled_token_id: u32,
    pub linear_layers: usize,
    pub full_attention_layers: usize,
    pub routed_expert_calls: usize,
    pub shared_expert_calls: usize,
    pub direct_packed_execution_is_not_proven_by_scheduler: bool,
}

/// The all-layer Qwen3-Next scheduler.  It is usable with an actual native
/// direct-packed backend once all required operators exist, and is separately
/// exercised in unit tests with a recorder only to prove exact source order.
/// It does not allocate a model body, open raw BF16 source, or create a
/// fallback execution path.
pub struct Qwen80CompleteHybridDecoder {
    catalog: Qwen80CompleteArtifactCatalog,
    plan: Qwen80CompleteHybridDecoderPlan,
    next_position: usize,
}

impl Qwen80CompleteHybridDecoder {
    /// Consume one already-admitted catalog into an all-layer scheduler.  This
    /// preserves the strict complete-artifact authority for dynamic routed
    /// expert names without reopening the 74,391 payloads.
    pub fn from_admitted_catalog(
        catalog: Qwen80CompleteArtifactCatalog,
        max_seq_len: usize,
    ) -> Result<Self> {
        let plan = catalog.complete_hybrid_decoder_plan(max_seq_len)?;
        Self::new(catalog, plan)
    }

    /// Construct from a catalog and its matching plan.  Keeping both inputs
    /// explicit makes a manifest/revision mismatch fail before a backend can
    /// bind a single projection or expert.
    pub fn new(
        catalog: Qwen80CompleteArtifactCatalog,
        plan: Qwen80CompleteHybridDecoderPlan,
    ) -> Result<Self> {
        if plan.layers.len() != QWEN80_LAYERS
            || plan.state.linear_layers != 36
            || plan.state.full_attention_layers != 12
            || plan.tokenizer_vocab_size != QWEN80_TOKENIZER_VOCAB
            || plan.reserved_lm_head_tail_rows != QWEN80_VOCAB - QWEN80_TOKENIZER_VOCAB
            || plan.manifest_seal_sha256 != catalog.manifest_seal()
            || plan.source_revision != catalog.config.source_revision
        {
            return Err(model_error(
                "hybrid decoder plan/catalog does not retain the exact Qwen80 architecture contract",
            ));
        }
        Ok(Self {
            catalog,
            plan,
            next_position: 0,
        })
    }

    pub fn catalog(&self) -> &Qwen80CompleteArtifactCatalog {
        &self.catalog
    }

    pub fn plan(&self) -> &Qwen80CompleteHybridDecoderPlan {
        &self.plan
    }

    pub fn next_position(&self) -> usize {
        self.next_position
    }

    pub fn reset_position(&mut self) {
        self.next_position = 0;
    }

    /// Run one exact all-layer schedule against an explicitly supplied
    /// backend.  This method is deliberately unable to fall back to a raw
    /// source model: every operation receives catalog-bound compact tensor
    /// addresses, and any missing backend operation aborts the token.
    pub fn execute_one<B: Qwen80PackedHybridDecodeBackend>(
        &mut self,
        backend: &mut B,
        input_token_id: u32,
    ) -> Result<Qwen80HybridScheduledToken> {
        if input_token_id as usize >= self.plan.tokenizer_vocab_size {
            return Err(model_error(format!(
                "input token {input_token_id} is outside the source tokenizer namespace 0..{}",
                self.plan.tokenizer_vocab_size.saturating_sub(1)
            )));
        }
        if self.next_position >= self.plan.max_seq_len {
            return Err(model_error(format!(
                "hybrid decoder position {} exceeds planned native context {}",
                self.next_position, self.plan.max_seq_len
            )));
        }
        let position = self.next_position;
        backend.begin_token(input_token_id, position, &self.plan.embedding)?;
        let mut linear_layers = 0usize;
        let mut full_attention_layers = 0usize;
        let mut routed_expert_calls = 0usize;
        let mut shared_expert_calls = 0usize;
        for layer in &self.plan.layers {
            backend.input_rms_norm(layer)?;
            match (
                &layer.mixer,
                layer.linear_state_slot,
                layer.full_attention_state_slot,
            ) {
                (Qwen80HybridMixerBindings::LinearDeltaNet(mixer), Some(slot), None) => {
                    backend.linear_deltanet_mixer(layer, slot, mixer)?;
                    linear_layers = linear_layers.saturating_add(1);
                }
                (Qwen80HybridMixerBindings::FullAttention(mixer), None, Some(slot)) => {
                    backend.full_attention_mixer(layer, slot, mixer, position)?;
                    full_attention_layers = full_attention_layers.saturating_add(1);
                }
                _ => {
                    return Err(model_error(format!(
                        "layer {} has an invalid mixed DeltaNet/GQA state binding",
                        layer.layer
                    )));
                }
            }
            backend.add_mixer_residual(layer)?;
            backend.post_attention_rms_norm(layer)?;
            let route = backend.route_top10(layer)?;
            let experts = self
                .plan
                .routed_expert_bindings(&self.catalog, layer.layer, &route)?;
            if experts.len() != QWEN80_TOP_K {
                return Err(model_error(format!(
                    "layer {} produced {} expert bindings, expected {QWEN80_TOP_K}",
                    layer.layer,
                    experts.len()
                )));
            }
            for (route_index, (expert, &route_weight)) in
                experts.iter().zip(route.weights.iter()).enumerate()
            {
                backend.routed_expert(layer, route_index, route_weight, expert)?;
                routed_expert_calls = routed_expert_calls.saturating_add(1);
            }
            backend.shared_expert(layer)?;
            shared_expert_calls = shared_expert_calls.saturating_add(1);
            backend.combine_moe_and_add_residual(layer)?;
        }
        if linear_layers != self.plan.state.linear_layers
            || full_attention_layers != self.plan.state.full_attention_layers
            || routed_expert_calls != self.plan.layers.len() * QWEN80_TOP_K
            || shared_expert_calls != self.plan.layers.len()
        {
            return Err(model_error(format!(
                "hybrid decoder dispatch counts linear={linear_layers}/{} attention={full_attention_layers}/{} routed={routed_expert_calls}/{} shared={shared_expert_calls}/{} are incomplete",
                self.plan.state.linear_layers,
                self.plan.state.full_attention_layers,
                self.plan.layers.len() * QWEN80_TOP_K,
                self.plan.layers.len(),
            )));
        }
        backend.final_rms_norm(&self.plan.final_norm)?;
        backend.lm_head(&self.plan.lm_head)?;
        backend.mask_reserved_lm_head_tail(self.plan.tokenizer_vocab_size as u32)?;
        let sampled_token_id = backend.sample_token(self.plan.tokenizer_vocab_size)?;
        if sampled_token_id as usize >= self.plan.tokenizer_vocab_size {
            return Err(model_error(format!(
                "sampler emitted token {sampled_token_id} in the source-reserved lm_head tail"
            )));
        }
        self.next_position = self.next_position.saturating_add(1);
        Ok(Qwen80HybridScheduledToken {
            position,
            input_token_id,
            sampled_token_id,
            linear_layers,
            full_attention_layers,
            routed_expert_calls,
            shared_expert_calls,
            direct_packed_execution_is_not_proven_by_scheduler: true,
        })
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
pub struct Qwen80GpuBinaryTensor {
    pub signs: PinnedBuffer,
    pub scales: PinnedBuffer,
    pub header: CompleteBinaryHeader,
}

/// Runtime-owned fixed compact payloads and scratch buffers for the L0
/// post-DeltaNet true-MoE suffix.  The ten routed expert compact sections
/// deliberately remain outside this type: they must arrive through the
/// source-selected [`Qwen80AllTenTrueMoeDeviceBridge`] so callers cannot
/// substitute a filename-selected route after strict admission.
///
/// Allocating this holder does not encode, commit, fence, or read back a
/// command buffer.  It is a narrow resource boundary for a future explicitly
/// leased source-input L0 capture, not a layer or token executor.
#[cfg(target_os = "macos")]
pub struct Qwen80L0TrueMoeFixedDeviceBuffers {
    pub contract: Qwen80CanonicalLinearMoEOperatorContract,
    pub postnorm: Qwen80GpuBinaryTensor,
    pub router: Qwen80GpuBinaryTensor,
    pub shared_gate_proj: Qwen80GpuBinaryTensor,
    pub shared_up_proj: Qwen80GpuBinaryTensor,
    pub shared_down_proj: Qwen80GpuBinaryTensor,
    pub shared_expert_gate: Qwen80GpuBinaryTensor,
    pub postnorm_hidden: PinnedBuffer,
    pub router_logits: PinnedBuffer,
    pub router_probabilities: PinnedBuffer,
    pub router_route_ids: PinnedBuffer,
    pub router_route_weights: PinnedBuffer,
    pub route_guard: PinnedBuffer,
    pub route_gate: PinnedBuffer,
    pub route_up: PinnedBuffer,
    pub route_activated: PinnedBuffer,
    pub route_weighted: PinnedBuffer,
    pub shared_gate: PinnedBuffer,
    pub shared_up: PinnedBuffer,
    pub shared_activated: PinnedBuffer,
    pub shared_output: PinnedBuffer,
    pub shared_scalar_logit: PinnedBuffer,
    pub gated_shared: PinnedBuffer,
    pub routed_sum: PinnedBuffer,
    pub second_residual: PinnedBuffer,
}

/// A parity result for the exact Qwen3-Next recurrent-state component at the
/// first linear layer's real 32x128x128 state geometry.  This is explicitly
/// not a model layer or full-token result: its inputs are deterministic
/// operator fixtures, while the artifact is separately bound by the compact
/// tensor upload path.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub struct Qwen80NativeDeltaNetComponentStep {
    pub max_abs_state_error: f32,
    pub max_abs_output_error: f32,
    pub metal_dispatches: usize,
}

/// Evidence from one deliberately bounded first-Qwen3-Next-linear-layer
/// execution stage.  It is source- and artifact-bound native Metal work, but
/// it is not a complete decoder layer: the causal convolution, Q/K/V/Z
/// rearrangement, gated RMSNorm, output projection, full top-10 expert wave,
/// shared expert, residual path, later layers, final norm, lm_head, sampler,
/// and token loop remain separate obligations.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80NativeLinearDeltaNetRouterExpertStage {
    pub layer: usize,
    pub layer_kind: String,
    pub direct_packed_input_projection_tensor: String,
    pub direct_packed_ba_projection_tensor: String,
    pub direct_packed_router_tensor: String,
    pub direct_packed_selected_expert_gate_tensor: String,
    pub qkvz_projection_max_abs_error: f32,
    pub ba_projection_max_abs_error: f32,
    pub ba_decay_max_abs_error: f32,
    pub ba_beta_max_abs_error: f32,
    pub deltanet_state_max_abs_error: f32,
    pub deltanet_output_max_abs_error: f32,
    pub router_logits_max_abs_error: f32,
    pub route_ids: Vec<u32>,
    pub route_weights: Vec<f32>,
    pub selected_expert: u32,
    pub selected_expert_gate_projection_max_abs_error: f32,
    pub first_command_buffer_dispatches: usize,
    pub selected_expert_command_buffer_dispatches: usize,
    pub source_ba_layout: String,
    pub source_router_policy: String,
}

/// One source-bound complete *DeltaNet mixer* parity stage for Qwen80 layer
/// zero. This is intentionally narrower than a decoder layer: it ends after
/// the first mixer residual and does not include post-attention RMSNorm, the
/// routed/shared MoE, a later layer, logits, sampling, feedback, HCLI, or
/// throughput measurement.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80NativeFirstLinearDeltaNetMixerStage {
    pub layer: usize,
    pub linear_state_slot: usize,
    pub direct_packed_input_layernorm_tensor: String,
    pub direct_packed_qkvz_tensor: String,
    pub direct_packed_ba_tensor: String,
    pub direct_packed_conv_tensor: String,
    pub direct_packed_a_log_tensor: String,
    pub direct_packed_dt_bias_tensor: String,
    pub direct_packed_gated_norm_tensor: String,
    pub direct_packed_out_proj_tensor: String,
    pub input_rms_norm_max_abs_error: f32,
    pub qkvz_projection_max_abs_error: f32,
    pub ba_projection_max_abs_error: f32,
    pub conv_state_max_abs_error: f32,
    pub repeated_query_max_abs_error: f32,
    pub repeated_key_max_abs_error: f32,
    pub convolved_value_max_abs_error: f32,
    pub z_max_abs_error: f32,
    pub decay_max_abs_error: f32,
    pub beta_max_abs_error: f32,
    pub recurrent_state_max_abs_error: f32,
    pub recurrent_output_max_abs_error: f32,
    pub gated_rms_norm_max_abs_error: f32,
    pub mixer_output_max_abs_error: f32,
    pub mixer_residual_max_abs_error: f32,
    pub metal_dispatches: usize,
    pub source_algorithm_boundary: String,
}

/// One source-token L0 DeltaNet encoder retained across an already-open Metal
/// command buffer.  Unlike the older bounded mixer stage, this object starts
/// from a direct-packed source embedding row and zeroed source-shaped state;
/// it does not use the historical deterministic fixture.  It owns every
/// temporary buffer needed until the caller fences the *same*
/// [`TokenCommandBuffer`] after appending the true-MoE suffix.
///
/// It is intentionally component-scoped: a source embedding is still a CPU
/// reference upload here, not a production native embedding gather, and no
/// post-norm/router/expert/second-residual work is implied by constructing it.
#[cfg(target_os = "macos")]
pub struct Qwen80SourceInputFirstResidualEncoder {
    source_token_id: u32,
    source_embedding_tensor: String,
    layer: usize,
    linear_state_slot: usize,
    input_f32le_sha256: String,
    initial_conv_state_f32le_sha256: String,
    initial_recurrent_state_f32le_sha256: String,
    expected_first_residual: Vec<f32>,
    expected_next_conv_state: Vec<f32>,
    expected_next_recurrent_state: Vec<f32>,
    // Source-token L0 begins from a known zero state. Keep an independent
    // device-resident checkpoint of those exact bytes so a future handoff
    // child can report an honest L0 rollback witness without treating the
    // CPU reference vector as a device checkpoint.
    rollback_conv_state: PinnedBuffer,
    rollback_recurrent_state: PinnedBuffer,
    _input: PinnedBuffer,
    _normalized: PinnedBuffer,
    _qkvz_output: PinnedBuffer,
    _ba_output: PinnedBuffer,
    _repeated_query: PinnedBuffer,
    _repeated_key: PinnedBuffer,
    _convolved_value: PinnedBuffer,
    _z: PinnedBuffer,
    _decay: PinnedBuffer,
    _beta: PinnedBuffer,
    _recurrent_output: PinnedBuffer,
    _gated_output: PinnedBuffer,
    _mixer_output: PinnedBuffer,
    first_residual: PinnedBuffer,
    _input_layernorm: Qwen80GpuBinaryTensor,
    _qkvz: Qwen80GpuBinaryTensor,
    _ba: Qwen80GpuBinaryTensor,
    _conv: Qwen80GpuBinaryTensor,
    _a_log: Qwen80GpuBinaryTensor,
    _dt_bias: Qwen80GpuBinaryTensor,
    _gated_norm: Qwen80GpuBinaryTensor,
    _out_proj: Qwen80GpuBinaryTensor,
    dispatches_encoded: usize,
}

/// Fence/readback evidence for the source-input first-residual component.
/// This record is deliberately insufficient for a complete Qwen80 layer or
/// token; it proves only the retained L0 source-input boundary.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80SourceInputFirstResidualParity {
    pub source_token_id: u32,
    pub source_embedding_tensor: String,
    pub layer: usize,
    pub linear_state_slot: usize,
    pub input_f32le_sha256: String,
    pub initial_conv_state_f32le_sha256: String,
    pub initial_recurrent_state_f32le_sha256: String,
    pub cpu_first_residual_f32le_sha256: String,
    pub device_first_residual_f32le_sha256: String,
    pub device_post_conv_state_f32le_sha256: String,
    pub device_post_recurrent_state_f32le_sha256: String,
    pub rollback_conv_state_f32le_sha256: String,
    pub rollback_recurrent_state_f32le_sha256: String,
    pub active_conv_state_buffer_identity_sha256: String,
    pub active_recurrent_state_buffer_identity_sha256: String,
    pub rollback_conv_state_buffer_identity_sha256: String,
    pub rollback_recurrent_state_buffer_identity_sha256: String,
    pub first_residual_max_abs_error: f32,
    pub conv_state_max_abs_error: f32,
    pub recurrent_state_max_abs_error: f32,
    pub first_residual_elements: usize,
    pub first_residual_bytes: usize,
    pub linear_conv_state_elements: usize,
    pub linear_conv_state_bytes: usize,
    pub linear_recurrent_state_elements: usize,
    pub linear_recurrent_state_bytes: usize,
    pub same_command_graph_required: bool,
    pub dispatches_encoded_before_suffix: usize,
}

/// Opaque custody capability for exactly one freshly encoded source-token L0
/// 9+14 DeltaNet/true-MoE graph.  It is intentionally non-serializable and
/// has no public constructor: the prior component receipt is provenance only,
/// while this value retains the live source runtime allocations needed to
/// append Layer 1 before the same command-buffer fence.
///
/// In particular, this is not a generic `[2048]` buffer wrapper.  Its creator
/// verifies the canonical L0 structural trace, source token, route body,
/// fixed suffix, CPU shadow, and live first-residual handoff before it can be
/// consumed by the Layer-1 prefix encoder.
#[cfg(target_os = "macos")]
pub struct Qwen80CanonicalSourceTokenL0TrueMoeContinuation {
    first_residual: Qwen80SourceInputFirstResidualEncoder,
    _route_bridge: Qwen80AllTenTrueMoeDeviceBridge,
    fixed: Qwen80L0TrueMoeFixedDeviceBuffers,
    cpu_l0: Qwen80CanonicalLinearMoECpuOracleResult,
    l0_structural_kernel_names: Vec<String>,
    runtime_state_arena_owner: Qwen80SameRuntimeStateArenaOwnerIdentity,
}

/// Process-local custody identity for the two shared DeltaNet state arenas.
///
/// The labels are deliberately centralized here: the opaque L0 continuation,
/// the Layer-1 prefix encoder, and every consuming finalizer must calculate
/// the identical identity for the same runtime allocation.  They are not
/// receipt/artifact hashes and must never be accepted as cross-process input.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct Qwen80SameRuntimeStateArenaOwnerIdentity {
    conv_state_buffer_identity_sha256: String,
    recurrent_state_buffer_identity_sha256: String,
}

#[cfg(target_os = "macos")]
impl Qwen80SameRuntimeStateArenaOwnerIdentity {
    const CONV_LABEL: &'static str = "canonical source-token same-runtime convolution state arena";
    const RECURRENT_LABEL: &'static str =
        "canonical source-token same-runtime recurrent state arena";

    fn from_runtime(runtime: &Qwen80CompleteNativeRuntime) -> Result<Self> {
        Ok(Self {
            conv_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &runtime.linear_conv_state,
                Self::CONV_LABEL,
            )?,
            recurrent_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &runtime.linear_recurrent_state,
                Self::RECURRENT_LABEL,
            )?,
        })
    }

    fn require_matches(&self, observed: &Self, mismatch: &str) -> Result<()> {
        if self != observed {
            return Err(model_error(mismatch));
        }
        Ok(())
    }

    fn require_runtime_owner(
        &self,
        runtime: &Qwen80CompleteNativeRuntime,
        mismatch: &str,
    ) -> Result<()> {
        self.require_matches(&Self::from_runtime(runtime)?, mismatch)
    }
}

#[cfg(target_os = "macos")]
impl Qwen80CanonicalSourceTokenL0TrueMoeContinuation {
    fn l0_second_residual(&self) -> &PinnedBuffer {
        &self.fixed.second_residual
    }

    fn cpu_l0_layer_output(&self) -> &[f32] {
        &self.cpu_l0.layer_output
    }

    fn require_l0_trace(&self) -> Result<()> {
        qwen80_require_exact_structural_kernel_trace(
            Some(&self.l0_structural_kernel_names),
            &QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS,
            "source-token canonical L0 true-MoE continuation",
        )
    }
}

#[cfg(target_os = "macos")]
impl Qwen80SourceInputFirstResidualEncoder {
    /// The actual device output handle.  A later true-MoE graph must retain a
    /// clone of this exact Metal buffer in the same command buffer; it may not
    /// reconstruct the vector from the CPU reference bytes.
    pub fn first_residual(&self) -> &PinnedBuffer {
        &self.first_residual
    }

    /// The source-zero DeltaNet convolution checkpoint retained on the same
    /// device until the caller has fenced and recorded its component receipt.
    pub fn rollback_conv_state(&self) -> &PinnedBuffer {
        &self.rollback_conv_state
    }

    /// The source-zero DeltaNet recurrent checkpoint retained on the same
    /// device until the caller has fenced and recorded its component receipt.
    pub fn rollback_recurrent_state(&self) -> &PinnedBuffer {
        &self.rollback_recurrent_state
    }

    pub fn source_token_id(&self) -> u32 {
        self.source_token_id
    }

    pub fn input_f32le_sha256(&self) -> &str {
        &self.input_f32le_sha256
    }

    pub fn expected_first_residual_f32le_sha256(&self) -> Result<String> {
        qwen80_f32_vector_sha256(&self.expected_first_residual, "source-input first residual")
    }

    /// Read only after the caller has committed and waited on the same token
    /// command buffer that encoded this mixer and any appended true-MoE work.
    pub fn verify_after_fence(
        &self,
        runtime: &Qwen80CompleteNativeRuntime,
    ) -> Result<Qwen80SourceInputFirstResidualParity> {
        const TOLERANCE: f32 = 1.0e-3;
        let device_first_residual = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.first_residual,
            QWEN80_HIDDEN,
            "Qwen80 source-input first residual",
        )?;
        let device_conv_state = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &runtime.linear_conv_state,
            self.expected_next_conv_state.len(),
            "Qwen80 source-input DeltaNet conv state",
        )?;
        let device_recurrent_state = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &runtime.linear_recurrent_state,
            self.expected_next_recurrent_state.len(),
            "Qwen80 source-input DeltaNet recurrent state",
        )?;
        let rollback_conv_state = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.rollback_conv_state,
            self.expected_next_conv_state.len(),
            "Qwen80 source-input DeltaNet rollback convolution state",
        )?;
        let rollback_recurrent_state = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.rollback_recurrent_state,
            self.expected_next_recurrent_state.len(),
            "Qwen80 source-input DeltaNet rollback recurrent state",
        )?;
        let first_residual_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &self.expected_first_residual,
            &device_first_residual,
            "Qwen80 source-input first residual",
            TOLERANCE,
        )?;
        let conv_state_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &self.expected_next_conv_state,
            &device_conv_state,
            "Qwen80 source-input DeltaNet conv state",
            TOLERANCE,
        )?;
        let recurrent_state_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &self.expected_next_recurrent_state,
            &device_recurrent_state,
            "Qwen80 source-input DeltaNet recurrent state",
            TOLERANCE,
        )?;
        let rollback_conv_state_f32le_sha256 = qwen80_f32_vector_sha256(
            &rollback_conv_state,
            "source-input DeltaNet rollback convolution state",
        )?;
        let rollback_recurrent_state_f32le_sha256 = qwen80_f32_vector_sha256(
            &rollback_recurrent_state,
            "source-input DeltaNet rollback recurrent state",
        )?;
        if rollback_conv_state_f32le_sha256 != self.initial_conv_state_f32le_sha256
            || rollback_recurrent_state_f32le_sha256 != self.initial_recurrent_state_f32le_sha256
        {
            return Err(model_error(
                "Qwen80 source-input DeltaNet rollback checkpoint differs from its source-zero pre-state",
            ));
        }
        let linear_conv_state_elements = self.expected_next_conv_state.len();
        let linear_recurrent_state_elements = self.expected_next_recurrent_state.len();
        Ok(Qwen80SourceInputFirstResidualParity {
            source_token_id: self.source_token_id,
            source_embedding_tensor: self.source_embedding_tensor.clone(),
            layer: self.layer,
            linear_state_slot: self.linear_state_slot,
            input_f32le_sha256: self.input_f32le_sha256.clone(),
            initial_conv_state_f32le_sha256: self.initial_conv_state_f32le_sha256.clone(),
            initial_recurrent_state_f32le_sha256: self.initial_recurrent_state_f32le_sha256.clone(),
            cpu_first_residual_f32le_sha256: self.expected_first_residual_f32le_sha256()?,
            device_first_residual_f32le_sha256: qwen80_f32_vector_sha256(
                &device_first_residual,
                "source-input device first residual",
            )?,
            device_post_conv_state_f32le_sha256: qwen80_f32_vector_sha256(
                &device_conv_state,
                "source-input device post convolution state",
            )?,
            device_post_recurrent_state_f32le_sha256: qwen80_f32_vector_sha256(
                &device_recurrent_state,
                "source-input device post recurrent state",
            )?,
            rollback_conv_state_f32le_sha256,
            rollback_recurrent_state_f32le_sha256,
            active_conv_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &runtime.linear_conv_state,
                "source-input DeltaNet active convolution state",
            )?,
            active_recurrent_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &runtime.linear_recurrent_state,
                "source-input DeltaNet active recurrent state",
            )?,
            rollback_conv_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.rollback_conv_state,
                "source-input DeltaNet rollback convolution state",
            )?,
            rollback_recurrent_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.rollback_recurrent_state,
                "source-input DeltaNet rollback recurrent state",
            )?,
            first_residual_max_abs_error,
            conv_state_max_abs_error,
            recurrent_state_max_abs_error,
            first_residual_elements: QWEN80_HIDDEN,
            first_residual_bytes: bytes_for_f32(
                QWEN80_HIDDEN,
                "Qwen80 source-input first residual bytes",
            )?,
            linear_conv_state_elements,
            linear_conv_state_bytes: bytes_for_f32(
                linear_conv_state_elements,
                "Qwen80 source-input DeltaNet convolution state bytes",
            )?,
            linear_recurrent_state_elements,
            linear_recurrent_state_bytes: bytes_for_f32(
                linear_recurrent_state_elements,
                "Qwen80 source-input DeltaNet recurrent state bytes",
            )?,
            same_command_graph_required: true,
            dispatches_encoded_before_suffix: self.dispatches_encoded,
        })
    }
}

/// One retained-input Layer-1 DeltaNet prefix encoded only inside the same
/// native runtime and token command buffer that still owns its Layer-0
/// second-residual allocation.  The caller supplies the CPU/reference shadow
/// of that exact L0 output for post-fence parity; this type owns the live
/// Metal clone and never reuploads a receipt-derived vector.
///
/// It is deliberately narrower than a decoder continuation.  It encodes only
/// Layer 1's nine DeltaNet prefix dispatches and stops before Layer 1
/// post-attention norm, MoE, second residual, Layer 2, sampling, or feedback.
#[cfg(target_os = "macos")]
pub struct Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder {
    source_token_id: u32,
    layer: usize,
    linear_state_slot: usize,
    input_f32le_sha256: String,
    input_buffer_identity_sha256: String,
    expected_input: Vec<f32>,
    initial_conv_state_f32le_sha256: String,
    initial_recurrent_state_f32le_sha256: String,
    expected_first_residual: Vec<f32>,
    expected_next_conv_state: Vec<f32>,
    expected_next_recurrent_state: Vec<f32>,
    conv_state_offset_elements: usize,
    conv_state_capacity_elements: usize,
    recurrent_state_offset_elements: usize,
    recurrent_state_capacity_elements: usize,
    rollback_conv_state: PinnedBuffer,
    rollback_recurrent_state: PinnedBuffer,
    input: PinnedBuffer,
    _normalized: PinnedBuffer,
    _qkvz_output: PinnedBuffer,
    _ba_output: PinnedBuffer,
    _repeated_query: PinnedBuffer,
    _repeated_key: PinnedBuffer,
    _convolved_value: PinnedBuffer,
    _z: PinnedBuffer,
    _decay: PinnedBuffer,
    _beta: PinnedBuffer,
    _recurrent_output: PinnedBuffer,
    _gated_output: PinnedBuffer,
    _mixer_output: PinnedBuffer,
    first_residual: PinnedBuffer,
    _input_layernorm: Qwen80GpuBinaryTensor,
    _qkvz: Qwen80GpuBinaryTensor,
    _ba: Qwen80GpuBinaryTensor,
    _conv: Qwen80GpuBinaryTensor,
    _a_log: Qwen80GpuBinaryTensor,
    _dt_bias: Qwen80GpuBinaryTensor,
    _gated_norm: Qwen80GpuBinaryTensor,
    _out_proj: Qwen80GpuBinaryTensor,
    // Retain the exact L0 source graph, including the L0 output allocation
    // and source-selected route/fixed suffix holders, until the consuming
    // joint finalizer has fenced and validated it.
    l0_continuation: Qwen80CanonicalSourceTokenL0TrueMoeContinuation,
    dispatches_encoded: usize,
}

/// Post-fence evidence for the same-runtime Layer-1 prefix.  Every buffer
/// identity is process-local custody evidence, not a serializable authority
/// that a later process may use to recreate this continuation.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80SameRuntimeLayer1DeltaNetPrefixParity {
    pub source_token_id: u32,
    pub layer: usize,
    pub linear_state_slot: usize,
    pub input_f32le_sha256: String,
    pub device_input_f32le_sha256: String,
    pub input_buffer_identity_sha256: String,
    pub cpu_first_residual_f32le_sha256: String,
    pub device_first_residual_f32le_sha256: String,
    pub first_residual_buffer_identity_sha256: String,
    pub device_post_conv_state_f32le_sha256: String,
    pub device_post_recurrent_state_f32le_sha256: String,
    pub rollback_conv_state_f32le_sha256: String,
    pub rollback_recurrent_state_f32le_sha256: String,
    pub active_conv_state_buffer_identity_sha256: String,
    pub active_recurrent_state_buffer_identity_sha256: String,
    pub rollback_conv_state_buffer_identity_sha256: String,
    pub rollback_recurrent_state_buffer_identity_sha256: String,
    pub input_max_abs_error: f32,
    pub first_residual_max_abs_error: f32,
    pub conv_state_max_abs_error: f32,
    pub recurrent_state_max_abs_error: f32,
    pub first_residual_elements: usize,
    pub first_residual_bytes: usize,
    pub conv_state_offset_elements: usize,
    pub conv_state_capacity_elements: usize,
    pub recurrent_state_offset_elements: usize,
    pub recurrent_state_capacity_elements: usize,
    pub required_l0_dispatches_before_prefix: usize,
    pub total_dispatches_after_prefix: usize,
    pub same_runtime_same_command_buffer_required: bool,
    pub dispatches_encoded: usize,
}

/// One bounded all-ten source-selected routed-wave readback from a fresh L0
/// graph.  It remains component evidence: each witness proves the exact
/// weighted body feeding the L0 second residual, not a reusable expert result
/// for another layer/process.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80SameRuntimeL0RoutedWaveParity {
    pub wave_index: usize,
    pub expert_id: usize,
    pub normalized_weight: f32,
    /// The fresh same-runtime CPU-oracle hash for this weighted expert body.
    /// Keeping it alongside the device hash prevents a later receipt writer
    /// from inventing CPU provenance from the device readback alone.
    pub cpu_output_f32le_sha256: String,
    /// The fresh post-fence device hash for this weighted expert body.
    pub device_output_f32le_sha256: String,
    /// Compatibility alias for earlier component ledgers; it is exactly the
    /// device hash above and must not be used as a CPU witness.
    pub output_f32le_sha256: String,
    pub max_abs_error: f32,
}

/// Fresh post-fence L0 suffix evidence retained by the same-runtime joint
/// finalizer.  The route guard and all ten weighted bodies prevent an L1
/// continuation from silently treating only the L0 second-residual vector as
/// sufficient provenance.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80SameRuntimeL0TrueMoeSuffixParity {
    pub route_guard: u32,
    pub observed_route_ids: Vec<u32>,
    pub expected_route_ids: Vec<u16>,
    pub observed_route_weights: Vec<f32>,
    pub expected_route_weights: Vec<f32>,
    pub route_weights_max_abs_error: f32,
    pub postnorm_cpu_f32le_sha256: String,
    pub postnorm_output_f32le_sha256: String,
    pub postnorm_max_abs_error: f32,
    pub router_logits_cpu_f32le_sha256: String,
    pub router_logits_output_f32le_sha256: String,
    pub router_logits_max_abs_error: f32,
    pub all_ten_route_witnesses: Vec<Qwen80SameRuntimeL0RoutedWaveParity>,
    pub shared_cpu_f32le_sha256: String,
    pub shared_output_f32le_sha256: String,
    pub shared_max_abs_error: f32,
    pub routed_sum_cpu_f32le_sha256: String,
    pub routed_sum_output_f32le_sha256: String,
    pub routed_sum_max_abs_error: f32,
    pub second_residual_cpu_f32le_sha256: String,
    pub second_residual_output_f32le_sha256: String,
    pub second_residual_max_abs_error: f32,
}

/// One post-fence weighted routed-expert witness for the fresh Layer-1 MoE
/// suffix.  CPU and device content hashes deliberately remain separate: the
/// bounded numeric parity result, not a bitwise hash match, proves the
/// strict-Math component result.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80SameRuntimeL1RoutedWaveParity {
    pub wave_index: usize,
    pub expert_id: usize,
    pub normalized_weight: f32,
    pub cpu_output_f32le_sha256: String,
    pub device_output_f32le_sha256: String,
    pub max_abs_error: f32,
}

/// Post-fence parity for the fourteen Layer-1 postnorm/router/all-ten/shared
/// suffix dispatches.  This is intentionally a component witness; it is not
/// a transferable route result or a full decoder/token claim.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80SameRuntimeL1TrueMoeSuffixParity {
    pub layer: usize,
    pub linear_state_slot: usize,
    pub route_guard: u32,
    pub observed_route_ids: Vec<u32>,
    pub expected_route_ids: Vec<u16>,
    pub observed_route_weights: Vec<f32>,
    pub expected_route_weights: Vec<f32>,
    pub route_weights_max_abs_error: f32,
    pub postnorm_cpu_f32le_sha256: String,
    pub postnorm_output_f32le_sha256: String,
    pub postnorm_max_abs_error: f32,
    pub router_logits_cpu_f32le_sha256: String,
    pub router_logits_output_f32le_sha256: String,
    pub router_logits_max_abs_error: f32,
    pub all_ten_route_witnesses: Vec<Qwen80SameRuntimeL1RoutedWaveParity>,
    pub shared_cpu_f32le_sha256: String,
    pub shared_output_f32le_sha256: String,
    pub shared_max_abs_error: f32,
    pub routed_sum_cpu_f32le_sha256: String,
    pub routed_sum_output_f32le_sha256: String,
    pub routed_sum_max_abs_error: f32,
    pub second_residual_cpu_f32le_sha256: String,
    pub second_residual_output_f32le_sha256: String,
    pub second_residual_max_abs_error: f32,
}

/// Fresh source-token L0 re-encode evidence retained by the 46-dispatch
/// finalizer.  The L0 first-residual state/rollback witness and the L0 second
/// residual parity are deliberately separate from the historical L0 receipt:
/// this capture has re-encoded them in the same process/runtime/TCB as L1.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80SameRuntimeFreshL0Parity {
    pub first_residual: Qwen80SourceInputFirstResidualParity,
    pub second_residual_cpu_f32le_sha256: String,
    pub second_residual_device_f32le_sha256: String,
    pub second_residual_buffer_identity_sha256: String,
    pub second_residual_max_abs_error: f32,
}

/// Receipt-ready evidence emitted only by the consuming 23+9+14 finalizer.
/// The command buffer is already committed when this value is constructed;
/// callers cannot append a later kernel and reuse its parity/readbacks.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80SameRuntimeL0L1FullLayerParity {
    pub fresh_l0: Qwen80SameRuntimeFreshL0Parity,
    pub l1_prefix: Qwen80SameRuntimeLayer1DeltaNetPrefixParity,
    pub l1_true_moe_suffix: Qwen80SameRuntimeL1TrueMoeSuffixParity,
    pub l0_dispatches: usize,
    pub l1_prefix_dispatches: usize,
    pub l1_moe_suffix_dispatches: usize,
    pub total_dispatches: usize,
    pub structural_kernel_names: Vec<String>,
    pub same_runtime_same_command_buffer_required: bool,
    pub single_fence_after_all_dispatches_required: bool,
}

/// Receipt-ready parity for the only legal same-process continuation boundary
/// from the source-token L0 true-MoE graph into the Layer-1 DeltaNet prefix.
/// It stops after Layer 1's first residual.  Neither a Layer-1 suffix nor a
/// complete layer/token is represented here.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80SameRuntimeL0L1PrefixParity {
    pub l0_first_residual: Qwen80SourceInputFirstResidualParity,
    pub l0_true_moe_suffix: Qwen80SameRuntimeL0TrueMoeSuffixParity,
    pub l0_second_residual_cpu_f32le_sha256: String,
    pub l0_second_residual_device_f32le_sha256: String,
    pub l0_second_residual_buffer_identity_sha256: String,
    pub l0_second_residual_max_abs_error: f32,
    pub l0_route_ids: Vec<u16>,
    pub l0_route_weights: Vec<f32>,
    pub l1_prefix: Qwen80SameRuntimeLayer1DeltaNetPrefixParity,
    pub l0_dispatches: usize,
    pub l1_prefix_dispatches: usize,
    pub total_dispatches: usize,
    pub structural_kernel_names: Vec<String>,
    pub same_runtime_same_command_buffer_required: bool,
    pub single_fence_after_l0_and_l1_prefix_required: bool,
    pub l1_suffix_or_moe_executed: bool,
}

#[cfg(target_os = "macos")]
impl Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder {
    /// The live Layer-1 output allocation.  It remains valid only while this
    /// owner and its source-runtime command graph remain alive.
    pub fn first_residual(&self) -> &PinnedBuffer {
        &self.first_residual
    }

    /// Read back and validate only after the caller has fenced the exact
    /// command buffer that first produced the retained L0 input and then
    /// appended this Layer-1 prefix.
    fn verify_after_fence(
        &self,
        runtime: &Qwen80CompleteNativeRuntime,
    ) -> Result<Qwen80SameRuntimeLayer1DeltaNetPrefixParity> {
        const TOLERANCE: f32 = 1.0e-3;
        let device_input = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.input,
            QWEN80_HIDDEN,
            "same-runtime Layer-1 retained L0 input",
        )?;
        let device_first_residual = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.first_residual,
            QWEN80_HIDDEN,
            "same-runtime Layer-1 first residual",
        )?;
        let device_conv_state = Qwen80CompleteNativeRuntime::device_f32_snapshot_at_offset(
            &runtime.linear_conv_state,
            self.conv_state_offset_elements,
            self.expected_next_conv_state.len(),
            "same-runtime Layer-1 active convolution state",
        )?;
        let device_recurrent_state = Qwen80CompleteNativeRuntime::device_f32_snapshot_at_offset(
            &runtime.linear_recurrent_state,
            self.recurrent_state_offset_elements,
            self.expected_next_recurrent_state.len(),
            "same-runtime Layer-1 active recurrent state",
        )?;
        let rollback_conv_state = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.rollback_conv_state,
            self.expected_next_conv_state.len(),
            "same-runtime Layer-1 rollback convolution state",
        )?;
        let rollback_recurrent_state = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.rollback_recurrent_state,
            self.expected_next_recurrent_state.len(),
            "same-runtime Layer-1 rollback recurrent state",
        )?;
        // The opaque continuation proves that this is the exact live L0
        // allocation retained by the same runtime/TCB.  CPU and Metal are
        // allowed their normal bounded floating-point divergence, so their
        // content hashes are evidence to retain separately—not a bitwise
        // equality predicate.  Requiring both hashes to match after the
        // tolerance check would reject a valid same-allocation handoff.
        let (input_max_abs_error, device_input_f32le_sha256) =
            Qwen80CompleteNativeRuntime::validate_same_runtime_l1_input_parity(
                &self.expected_input,
                &device_input,
            )?;
        let first_residual_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &self.expected_first_residual,
            &device_first_residual,
            "same-runtime Layer-1 first residual",
            TOLERANCE,
        )?;
        let conv_state_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &self.expected_next_conv_state,
            &device_conv_state,
            "same-runtime Layer-1 convolution state",
            TOLERANCE,
        )?;
        let recurrent_state_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &self.expected_next_recurrent_state,
            &device_recurrent_state,
            "same-runtime Layer-1 recurrent state",
            TOLERANCE,
        )?;
        let rollback_conv_state_f32le_sha256 = qwen80_f32_vector_sha256(
            &rollback_conv_state,
            "same-runtime Layer-1 rollback convolution state",
        )?;
        let rollback_recurrent_state_f32le_sha256 = qwen80_f32_vector_sha256(
            &rollback_recurrent_state,
            "same-runtime Layer-1 rollback recurrent state",
        )?;
        if rollback_conv_state_f32le_sha256 != self.initial_conv_state_f32le_sha256
            || rollback_recurrent_state_f32le_sha256 != self.initial_recurrent_state_f32le_sha256
        {
            return Err(model_error(
                "same-runtime Layer-1 rollback checkpoint differs from its source-zero pre-state",
            ));
        }
        Ok(Qwen80SameRuntimeLayer1DeltaNetPrefixParity {
            source_token_id: self.source_token_id,
            layer: self.layer,
            linear_state_slot: self.linear_state_slot,
            input_f32le_sha256: self.input_f32le_sha256.clone(),
            device_input_f32le_sha256,
            input_buffer_identity_sha256: self.input_buffer_identity_sha256.clone(),
            cpu_first_residual_f32le_sha256: qwen80_f32_vector_sha256(
                &self.expected_first_residual,
                "same-runtime Layer-1 CPU first residual",
            )?,
            device_first_residual_f32le_sha256: qwen80_f32_vector_sha256(
                &device_first_residual,
                "same-runtime Layer-1 device first residual",
            )?,
            first_residual_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.first_residual,
                "same-runtime Layer-1 first residual",
            )?,
            device_post_conv_state_f32le_sha256: qwen80_f32_vector_sha256(
                &device_conv_state,
                "same-runtime Layer-1 active convolution state",
            )?,
            device_post_recurrent_state_f32le_sha256: qwen80_f32_vector_sha256(
                &device_recurrent_state,
                "same-runtime Layer-1 active recurrent state",
            )?,
            rollback_conv_state_f32le_sha256,
            rollback_recurrent_state_f32le_sha256,
            active_conv_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &runtime.linear_conv_state,
                "same-runtime Layer-1 active convolution arena",
            )?,
            active_recurrent_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &runtime.linear_recurrent_state,
                "same-runtime Layer-1 active recurrent arena",
            )?,
            rollback_conv_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.rollback_conv_state,
                "same-runtime Layer-1 rollback convolution state",
            )?,
            rollback_recurrent_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.rollback_recurrent_state,
                "same-runtime Layer-1 rollback recurrent state",
            )?,
            input_max_abs_error,
            first_residual_max_abs_error,
            conv_state_max_abs_error,
            recurrent_state_max_abs_error,
            first_residual_elements: QWEN80_HIDDEN,
            first_residual_bytes: bytes_for_f32(
                QWEN80_HIDDEN,
                "same-runtime Layer-1 first residual bytes",
            )?,
            conv_state_offset_elements: self.conv_state_offset_elements,
            conv_state_capacity_elements: self.conv_state_capacity_elements,
            recurrent_state_offset_elements: self.recurrent_state_offset_elements,
            recurrent_state_capacity_elements: self.recurrent_state_capacity_elements,
            required_l0_dispatches_before_prefix: 23,
            total_dispatches_after_prefix: 32,
            same_runtime_same_command_buffer_required: true,
            dispatches_encoded: self.dispatches_encoded,
        })
    }

    /// Recompute the full Layer-1 CPU/reference oracle from the opaque
    /// continuation's retained L0 shadow.  This is deliberately a method on
    /// the opaque prefix owner rather than an API accepting a caller vector:
    /// a future suffix cannot mix a CPU oracle derived from another L0
    /// capture with this live first-residual buffer.
    pub fn derive_fresh_l1_full_cpu_oracle(
        &self,
        runtime: &Qwen80CompleteNativeRuntime,
    ) -> Result<Qwen80CanonicalLinearMoECpuOracleResult> {
        let contract = runtime.catalog.canonical_linear_moe_operator_contract(1)?;
        if contract.mixer.layer != self.layer
            || contract.mixer.linear_state_slot != self.linear_state_slot
            || self.layer != 1
            || self.linear_state_slot != 1
        {
            return Err(model_error(
                "same-runtime Layer-1 full CPU oracle did not bind layer/state slot one",
            ));
        }
        let input =
            Qwen80CanonicalLinearLayerCpuInput::with_zero_state(self.expected_input.clone());
        let cpu = runtime
            .catalog
            .execute_canonical_linear_moe_cpu_oracle(&contract, &input)?;
        if cpu.mixer.layer != self.layer || cpu.mixer.linear_state_slot != self.linear_state_slot {
            return Err(model_error(
                "same-runtime Layer-1 full CPU oracle layer/state slot drifted",
            ));
        }
        Qwen80CompleteNativeRuntime::require_parity(
            &cpu.mixer.mixer_residual_output,
            &self.expected_first_residual,
            "same-runtime Layer-1 full CPU oracle first residual",
            0.0,
        )?;
        Qwen80CompleteNativeRuntime::require_parity(
            &cpu.mixer.next_state.conv_state,
            &self.expected_next_conv_state,
            "same-runtime Layer-1 full CPU oracle convolution state",
            0.0,
        )?;
        Qwen80CompleteNativeRuntime::require_parity(
            &cpu.mixer.next_state.recurrent_state,
            &self.expected_next_recurrent_state,
            "same-runtime Layer-1 full CPU oracle recurrent state",
            0.0,
        )?;
        if cpu.route.ids.len() != QWEN80_TOP_K || cpu.route.weights.len() != QWEN80_TOP_K {
            return Err(model_error(
                "same-runtime Layer-1 full CPU oracle did not retain exactly ten source routes",
            ));
        }
        if cpu.routed_experts.len() != QWEN80_TOP_K {
            return Err(model_error(
                "same-runtime Layer-1 full CPU oracle did not retain ten routed bodies",
            ));
        }
        Ok(cpu)
    }

    fn collect_l1_true_moe_suffix_parity_after_fence(
        fixed: &Qwen80L0TrueMoeFixedDeviceBuffers,
        cpu: &Qwen80CanonicalLinearMoECpuOracleResult,
    ) -> Result<Qwen80SameRuntimeL1TrueMoeSuffixParity> {
        if fixed.contract.mixer.layer != 1
            || fixed.contract.mixer.linear_state_slot != 1
            || cpu.mixer.layer != 1
            || cpu.mixer.linear_state_slot != 1
            || cpu.route.ids.len() != QWEN80_TOP_K
            || cpu.route.weights.len() != QWEN80_TOP_K
            || cpu.routed_experts.len() != QWEN80_TOP_K
        {
            return Err(model_error(
                "same-runtime Layer-1 suffix readback lost its exact layer-one/top-ten contract",
            ));
        }
        let postnorm = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &fixed.postnorm_hidden,
            QWEN80_HIDDEN,
            "same-runtime Layer-1 postnorm",
        )?;
        let router_logits = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &fixed.router_logits,
            QWEN80_EXPERTS,
            "same-runtime Layer-1 router logits",
        )?;
        let observed_route_ids = Qwen80CompleteNativeRuntime::device_u32_snapshot(
            &fixed.router_route_ids,
            QWEN80_TOP_K,
            "same-runtime Layer-1 router IDs",
        )?;
        let observed_route_weights = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &fixed.router_route_weights,
            QWEN80_TOP_K,
            "same-runtime Layer-1 router weights",
        )?;
        let route_guard = Qwen80CompleteNativeRuntime::device_u32_snapshot(
            &fixed.route_guard,
            1,
            "same-runtime Layer-1 route guard",
        )?[0];
        let expected_route_ids = cpu
            .route
            .ids
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        if route_guard != 1 || observed_route_ids != expected_route_ids {
            return Err(model_error(
                "same-runtime Layer-1 route guard/readback differs from the source-selected top-10 route",
            ));
        }
        let postnorm_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &cpu.post_attention_rms_norm_output,
            &postnorm,
            "same-runtime Layer-1 postnorm",
            2.0e-4,
        )?;
        let router_logits_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &cpu.router_logits,
            &router_logits,
            "same-runtime Layer-1 router logits",
            5.0e-4,
        )?;
        let route_weights_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &cpu.route.weights,
            &observed_route_weights,
            "same-runtime Layer-1 router weights",
            2.0e-5,
        )?;
        let route_weighted = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &fixed.route_weighted,
            QWEN80_TOP_K.checked_mul(QWEN80_HIDDEN).ok_or_else(|| {
                model_error("same-runtime Layer-1 route output geometry overflowed")
            })?,
            "same-runtime Layer-1 weighted route outputs",
        )?;
        let mut all_ten_route_witnesses = Vec::with_capacity(QWEN80_TOP_K);
        for (wave_index, expected) in cpu.routed_experts.iter().enumerate() {
            let start = wave_index
                .checked_mul(QWEN80_HIDDEN)
                .ok_or_else(|| model_error("same-runtime Layer-1 route offset overflowed"))?;
            let end = start
                .checked_add(QWEN80_HIDDEN)
                .ok_or_else(|| model_error("same-runtime Layer-1 route range overflowed"))?;
            let observed = route_weighted
                .get(start..end)
                .ok_or_else(|| model_error("same-runtime Layer-1 route output is truncated"))?;
            let max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
                &expected.weighted_output,
                observed,
                &format!("same-runtime Layer-1 weighted route {wave_index}"),
                3.0e-4,
            )?;
            all_ten_route_witnesses.push(Qwen80SameRuntimeL1RoutedWaveParity {
                wave_index,
                expert_id: expected.expert,
                normalized_weight: expected.route_weight,
                cpu_output_f32le_sha256: qwen80_f32_vector_sha256(
                    &expected.weighted_output,
                    &format!("same-runtime Layer-1 CPU weighted route {wave_index}"),
                )?,
                device_output_f32le_sha256: qwen80_f32_vector_sha256(
                    observed,
                    &format!("same-runtime Layer-1 weighted route {wave_index}"),
                )?,
                max_abs_error,
            });
        }
        let shared = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &fixed.gated_shared,
            QWEN80_HIDDEN,
            "same-runtime Layer-1 shared output",
        )?;
        let routed_sum = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &fixed.routed_sum,
            QWEN80_HIDDEN,
            "same-runtime Layer-1 routed sum",
        )?;
        let second_residual = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &fixed.second_residual,
            QWEN80_HIDDEN,
            "same-runtime Layer-1 second residual",
        )?;
        let shared_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &cpu.shared_gated_output,
            &shared,
            "same-runtime Layer-1 shared output",
            3.0e-4,
        )?;
        let routed_sum_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &cpu.routed_expert_sum,
            &routed_sum,
            "same-runtime Layer-1 routed sum",
            3.0e-5,
        )?;
        let second_residual_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &cpu.layer_output,
            &second_residual,
            "same-runtime Layer-1 second residual",
            1.0e-3,
        )?;
        Ok(Qwen80SameRuntimeL1TrueMoeSuffixParity {
            layer: 1,
            linear_state_slot: 1,
            route_guard,
            observed_route_ids,
            expected_route_ids: cpu.route.ids.to_vec(),
            observed_route_weights,
            expected_route_weights: cpu.route.weights.to_vec(),
            route_weights_max_abs_error,
            postnorm_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                &cpu.post_attention_rms_norm_output,
                "same-runtime Layer-1 CPU postnorm",
            )?,
            postnorm_output_f32le_sha256: qwen80_f32_vector_sha256(
                &postnorm,
                "same-runtime Layer-1 postnorm",
            )?,
            postnorm_max_abs_error,
            router_logits_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                &cpu.router_logits,
                "same-runtime Layer-1 CPU router logits",
            )?,
            router_logits_output_f32le_sha256: qwen80_f32_vector_sha256(
                &router_logits,
                "same-runtime Layer-1 router logits",
            )?,
            router_logits_max_abs_error,
            all_ten_route_witnesses,
            shared_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                &cpu.shared_gated_output,
                "same-runtime Layer-1 CPU shared output",
            )?,
            shared_output_f32le_sha256: qwen80_f32_vector_sha256(
                &shared,
                "same-runtime Layer-1 shared output",
            )?,
            shared_max_abs_error,
            routed_sum_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                &cpu.routed_expert_sum,
                "same-runtime Layer-1 CPU routed sum",
            )?,
            routed_sum_output_f32le_sha256: qwen80_f32_vector_sha256(
                &routed_sum,
                "same-runtime Layer-1 routed sum",
            )?,
            routed_sum_max_abs_error,
            second_residual_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                &cpu.layer_output,
                "same-runtime Layer-1 CPU second residual",
            )?,
            second_residual_output_f32le_sha256: qwen80_f32_vector_sha256(
                &second_residual,
                "same-runtime Layer-1 second residual",
            )?,
            second_residual_max_abs_error,
        })
    }

    /// Consume the same-runtime L0→L1 prefix owner, require the complete
    /// canonical 23+9 structural graph, submit the one common fence, and
    /// produce the only receipt-ready parity result.  There is deliberately
    /// no public post-encode verification hook: callers cannot append a
    /// hidden suffix after the nine Layer-1 dispatches and still claim this
    /// component boundary.
    pub fn finalize_after_exact_joint_fence(
        self,
        runtime: &Qwen80CompleteNativeRuntime,
        command: TokenCommandBuffer<'_>,
    ) -> Result<Qwen80SameRuntimeL0L1PrefixParity> {
        const L0_DISPATCHES: usize = QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS.len();
        const L1_PREFIX_DISPATCHES: usize = QWEN80_SOURCE_TOKEN_L1_DELTA_PREFIX_KERNELS.len();
        const TOTAL_DISPATCHES: usize = L0_DISPATCHES + L1_PREFIX_DISPATCHES;
        if self.dispatches_encoded != L1_PREFIX_DISPATCHES
            || command.dispatch_count() != TOTAL_DISPATCHES
        {
            return Err(model_error(format!(
                "same-runtime L0→L1 finalizer requires exactly {L0_DISPATCHES}+{L1_PREFIX_DISPATCHES} dispatches, observed encoded={} total={}",
                self.dispatches_encoded,
                command.dispatch_count(),
            )));
        }
        self.l0_continuation
            .runtime_state_arena_owner
            .require_runtime_owner(
                runtime,
                "same-runtime L0→L1 finalizer refuses a runtime other than the continuation owner",
            )?;
        self.l0_continuation.require_l0_trace()?;
        let expected_kernels = QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS
            .iter()
            .chain(QWEN80_SOURCE_TOKEN_L1_DELTA_PREFIX_KERNELS.iter())
            .copied()
            .collect::<Vec<_>>();
        qwen80_require_exact_structural_kernel_trace(
            command.structural_kernel_names(),
            &expected_kernels,
            "same-runtime source-token L0→L1 finalizer",
        )?;
        let structural_kernel_names = command
            .structural_kernel_names()
            .expect("checked exact structural trace")
            .to_vec();

        command.commit_and_wait().map_err(|error| {
            model_error(format!("same-runtime L0→L1 common fence failed: {error}"))
        })?;

        const TOLERANCE: f32 = 1.0e-3;
        let l0_second_residual = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            self.l0_continuation.l0_second_residual(),
            QWEN80_HIDDEN,
            "same-runtime canonical L0 second residual",
        )?;
        let l0_second_residual_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            self.l0_continuation.cpu_l0_layer_output(),
            &l0_second_residual,
            "same-runtime canonical L0 second residual",
            TOLERANCE,
        )?;
        let l0_cpu = &self.l0_continuation.cpu_l0;
        let postnorm = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.l0_continuation.fixed.postnorm_hidden,
            QWEN80_HIDDEN,
            "same-runtime canonical L0 postnorm",
        )?;
        let router_logits = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.l0_continuation.fixed.router_logits,
            QWEN80_EXPERTS,
            "same-runtime canonical L0 router logits",
        )?;
        let observed_route_ids = Qwen80CompleteNativeRuntime::device_u32_snapshot(
            &self.l0_continuation.fixed.router_route_ids,
            QWEN80_TOP_K,
            "same-runtime canonical L0 router IDs",
        )?;
        let observed_route_weights = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.l0_continuation.fixed.router_route_weights,
            QWEN80_TOP_K,
            "same-runtime canonical L0 router weights",
        )?;
        let route_guard = Qwen80CompleteNativeRuntime::device_u32_snapshot(
            &self.l0_continuation.fixed.route_guard,
            1,
            "same-runtime canonical L0 route guard",
        )?[0];
        let expected_route_ids = l0_cpu
            .route
            .ids
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        if route_guard != 1 || observed_route_ids != expected_route_ids {
            return Err(model_error(
                "same-runtime canonical L0 route guard/readback differs from the source-selected top-10 route",
            ));
        }
        let postnorm_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &l0_cpu.post_attention_rms_norm_output,
            &postnorm,
            "same-runtime canonical L0 postnorm",
            2.0e-4,
        )?;
        let router_logits_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &l0_cpu.router_logits,
            &router_logits,
            "same-runtime canonical L0 router logits",
            5.0e-4,
        )?;
        let route_weights_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &l0_cpu.route.weights,
            &observed_route_weights,
            "same-runtime canonical L0 router weights",
            2.0e-5,
        )?;
        let route_weighted = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.l0_continuation.fixed.route_weighted,
            QWEN80_TOP_K.checked_mul(QWEN80_HIDDEN).ok_or_else(|| {
                model_error("same-runtime canonical L0 route output geometry overflowed")
            })?,
            "same-runtime canonical L0 weighted route outputs",
        )?;
        if l0_cpu.routed_experts.len() != QWEN80_TOP_K {
            return Err(model_error(
                "same-runtime canonical L0 CPU route witness count is not ten",
            ));
        }
        let mut all_ten_route_witnesses = Vec::with_capacity(QWEN80_TOP_K);
        for (wave_index, expected) in l0_cpu.routed_experts.iter().enumerate() {
            let start = wave_index
                .checked_mul(QWEN80_HIDDEN)
                .ok_or_else(|| model_error("same-runtime canonical L0 route offset overflowed"))?;
            let end = start
                .checked_add(QWEN80_HIDDEN)
                .ok_or_else(|| model_error("same-runtime canonical L0 route range overflowed"))?;
            let observed = route_weighted.get(start..end).ok_or_else(|| {
                model_error("same-runtime canonical L0 route output is truncated")
            })?;
            let max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
                &expected.weighted_output,
                observed,
                &format!("same-runtime canonical L0 weighted route {wave_index}"),
                3.0e-4,
            )?;
            all_ten_route_witnesses.push(Qwen80SameRuntimeL0RoutedWaveParity {
                wave_index,
                expert_id: expected.expert,
                normalized_weight: expected.route_weight,
                cpu_output_f32le_sha256: qwen80_f32_vector_sha256(
                    &expected.weighted_output,
                    &format!("same-runtime canonical L0 CPU weighted route {wave_index}"),
                )?,
                device_output_f32le_sha256: qwen80_f32_vector_sha256(
                    observed,
                    &format!("same-runtime canonical L0 weighted route {wave_index}"),
                )?,
                output_f32le_sha256: qwen80_f32_vector_sha256(
                    observed,
                    &format!("same-runtime canonical L0 weighted route {wave_index}"),
                )?,
                max_abs_error,
            });
        }
        let shared = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.l0_continuation.fixed.gated_shared,
            QWEN80_HIDDEN,
            "same-runtime canonical L0 shared output",
        )?;
        let routed_sum = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.l0_continuation.fixed.routed_sum,
            QWEN80_HIDDEN,
            "same-runtime canonical L0 routed sum",
        )?;
        let shared_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &l0_cpu.shared_gated_output,
            &shared,
            "same-runtime canonical L0 shared output",
            3.0e-4,
        )?;
        let routed_sum_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &l0_cpu.routed_expert_sum,
            &routed_sum,
            "same-runtime canonical L0 routed sum",
            3.0e-5,
        )?;
        let l0_true_moe_suffix = Qwen80SameRuntimeL0TrueMoeSuffixParity {
            route_guard,
            observed_route_ids,
            expected_route_ids: l0_cpu.route.ids.to_vec(),
            observed_route_weights,
            expected_route_weights: l0_cpu.route.weights.to_vec(),
            route_weights_max_abs_error,
            postnorm_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                &l0_cpu.post_attention_rms_norm_output,
                "same-runtime canonical L0 CPU postnorm",
            )?,
            postnorm_output_f32le_sha256: qwen80_f32_vector_sha256(
                &postnorm,
                "same-runtime canonical L0 postnorm",
            )?,
            postnorm_max_abs_error,
            router_logits_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                &l0_cpu.router_logits,
                "same-runtime canonical L0 CPU router logits",
            )?,
            router_logits_output_f32le_sha256: qwen80_f32_vector_sha256(
                &router_logits,
                "same-runtime canonical L0 router logits",
            )?,
            router_logits_max_abs_error,
            all_ten_route_witnesses,
            shared_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                &l0_cpu.shared_gated_output,
                "same-runtime canonical L0 CPU shared output",
            )?,
            shared_output_f32le_sha256: qwen80_f32_vector_sha256(
                &shared,
                "same-runtime canonical L0 shared output",
            )?,
            shared_max_abs_error,
            routed_sum_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                &l0_cpu.routed_expert_sum,
                "same-runtime canonical L0 CPU routed sum",
            )?,
            routed_sum_output_f32le_sha256: qwen80_f32_vector_sha256(
                &routed_sum,
                "same-runtime canonical L0 routed sum",
            )?,
            routed_sum_max_abs_error,
            second_residual_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                self.l0_continuation.cpu_l0_layer_output(),
                "same-runtime canonical L0 CPU second residual",
            )?,
            second_residual_output_f32le_sha256: qwen80_f32_vector_sha256(
                &l0_second_residual,
                "same-runtime canonical L0 second residual",
            )?,
            second_residual_max_abs_error: l0_second_residual_max_abs_error,
        };
        let l0_first_residual = self
            .l0_continuation
            .first_residual
            .verify_after_fence(runtime)?;
        let l1_prefix = self.verify_after_fence(runtime)?;
        Ok(Qwen80SameRuntimeL0L1PrefixParity {
            l0_first_residual,
            l0_true_moe_suffix,
            l0_second_residual_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                self.l0_continuation.cpu_l0_layer_output(),
                "same-runtime canonical L0 CPU second residual",
            )?,
            l0_second_residual_device_f32le_sha256: qwen80_f32_vector_sha256(
                &l0_second_residual,
                "same-runtime canonical L0 device second residual",
            )?,
            l0_second_residual_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                self.l0_continuation.l0_second_residual(),
                "same-runtime canonical L0 second residual",
            )?,
            l0_second_residual_max_abs_error,
            l0_route_ids: l0_cpu.route.ids.to_vec(),
            l0_route_weights: l0_cpu.route.weights.to_vec(),
            l1_prefix,
            l0_dispatches: L0_DISPATCHES,
            l1_prefix_dispatches: L1_PREFIX_DISPATCHES,
            total_dispatches: TOTAL_DISPATCHES,
            structural_kernel_names,
            same_runtime_same_command_buffer_required: true,
            single_fence_after_l0_and_l1_prefix_required: true,
            l1_suffix_or_moe_executed: false,
        })
    }

    /// Consume the opaque L0→L1 prefix owner only after a caller has appended
    /// the canonical fourteen-dispatch Layer-1 MoE suffix.  This is a sibling
    /// of [`Self::finalize_after_exact_joint_fence`], not a widening of it:
    /// the older finalizer remains a strict 23+9 custody boundary and still
    /// refuses an appended suffix.
    ///
    /// The caller retains the Layer-1 source-selected bridge and fixed suffix
    /// buffers so it can perform the required post-fence route/shared/residual
    /// readbacks.  This consuming method owns the sole command-buffer fence
    /// and proves that those readbacks follow exactly 23+9+14 non-timed
    /// dispatches in the same runtime and command buffer.
    pub fn finalize_after_exact_l1_moe_completion_fence(
        self,
        runtime: &Qwen80CompleteNativeRuntime,
        command: TokenCommandBuffer<'_>,
    ) -> Result<Qwen80SameRuntimeLayer1DeltaNetPrefixParity> {
        const L0_DISPATCHES: usize = QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS.len();
        const L1_PREFIX_DISPATCHES: usize = QWEN80_SOURCE_TOKEN_L1_DELTA_PREFIX_KERNELS.len();
        const L1_SUFFIX_DISPATCHES: usize = QWEN80_SOURCE_TOKEN_L1_MOE_SUFFIX_KERNELS.len();
        const TOTAL_DISPATCHES: usize = L0_DISPATCHES + L1_PREFIX_DISPATCHES + L1_SUFFIX_DISPATCHES;
        if self.dispatches_encoded != L1_PREFIX_DISPATCHES
            || command.dispatch_count() != TOTAL_DISPATCHES
        {
            return Err(model_error(format!(
                "same-runtime L1 completion finalizer requires exactly {L0_DISPATCHES}+{L1_PREFIX_DISPATCHES}+{L1_SUFFIX_DISPATCHES} dispatches, observed prefix={} total={}",
                self.dispatches_encoded,
                command.dispatch_count(),
            )));
        }
        self.l0_continuation.runtime_state_arena_owner.require_runtime_owner(
            runtime,
            "same-runtime L1 completion finalizer refuses a runtime other than the opaque continuation owner",
        )?;
        self.l0_continuation.require_l0_trace()?;
        let expected_kernels = QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS
            .iter()
            .chain(QWEN80_SOURCE_TOKEN_L1_DELTA_PREFIX_KERNELS.iter())
            .chain(QWEN80_SOURCE_TOKEN_L1_MOE_SUFFIX_KERNELS.iter())
            .copied()
            .collect::<Vec<_>>();
        qwen80_require_exact_structural_kernel_trace(
            command.structural_kernel_names(),
            &expected_kernels,
            "same-runtime source-token L0→L1 complete-MoE finalizer",
        )?;
        command.commit_and_wait().map_err(|error| {
            model_error(format!(
                "same-runtime L0→L1 complete-MoE common fence failed: {error}"
            ))
        })?;
        self.verify_after_fence(runtime)
    }

    /// Consume a freshly encoded L0(23)+L1-prefix(9)+L1-MoE-suffix(14)
    /// graph and retain every required post-fence parity witness.  This is
    /// the only 46-dispatch finalizer intended for the complete-L1 outer
    /// receipt: it derives the CPU suffix oracle from this opaque prefix,
    /// validates the structural trace, performs one fence, and then reads
    /// L0/L1 state, route, shared, routed-sum, and second-residual evidence.
    ///
    /// `fixed` must be the layer-one holder allocated by
    /// [`Qwen80CompleteNativeRuntime::upload_canonical_linear_moe_fixed_device_buffers`]
    /// in the same runtime.  It has no caller-provided buffer/import surface.
    pub fn finalize_after_exact_l1_moe_completion_fence_with_readbacks(
        self,
        runtime: &Qwen80CompleteNativeRuntime,
        command: TokenCommandBuffer<'_>,
        fixed: &Qwen80L0TrueMoeFixedDeviceBuffers,
    ) -> Result<Qwen80SameRuntimeL0L1FullLayerParity> {
        const L0_DISPATCHES: usize = QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS.len();
        const L1_PREFIX_DISPATCHES: usize = QWEN80_SOURCE_TOKEN_L1_DELTA_PREFIX_KERNELS.len();
        const L1_SUFFIX_DISPATCHES: usize = QWEN80_SOURCE_TOKEN_L1_MOE_SUFFIX_KERNELS.len();
        const TOTAL_DISPATCHES: usize = L0_DISPATCHES + L1_PREFIX_DISPATCHES + L1_SUFFIX_DISPATCHES;
        if self.dispatches_encoded != L1_PREFIX_DISPATCHES
            || command.dispatch_count() != TOTAL_DISPATCHES
        {
            return Err(model_error(format!(
                "same-runtime full-L1 finalizer requires exactly {L0_DISPATCHES}+{L1_PREFIX_DISPATCHES}+{L1_SUFFIX_DISPATCHES} dispatches, observed prefix={} total={}",
                self.dispatches_encoded,
                command.dispatch_count(),
            )));
        }
        if fixed.contract.mixer.layer != 1 || fixed.contract.mixer.linear_state_slot != 1 {
            return Err(model_error(
                "same-runtime full-L1 finalizer refuses a non-Layer-1 fixed suffix holder",
            ));
        }
        self.l0_continuation.runtime_state_arena_owner.require_runtime_owner(
            runtime,
            "same-runtime full-L1 finalizer refuses a runtime other than the opaque continuation owner",
        )?;
        self.l0_continuation.require_l0_trace()?;
        let expected_kernels = QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS
            .iter()
            .chain(QWEN80_SOURCE_TOKEN_L1_DELTA_PREFIX_KERNELS.iter())
            .chain(QWEN80_SOURCE_TOKEN_L1_MOE_SUFFIX_KERNELS.iter())
            .copied()
            .collect::<Vec<_>>();
        qwen80_require_exact_structural_kernel_trace(
            command.structural_kernel_names(),
            &expected_kernels,
            "same-runtime source-token full-L1 readback finalizer",
        )?;
        let structural_kernel_names = command
            .structural_kernel_names()
            .expect("checked exact structural trace")
            .to_vec();
        let l1_cpu = self.derive_fresh_l1_full_cpu_oracle(runtime)?;
        if fixed.contract.post_attention_layernorm.name
            != l1_cpu.direct_packed_post_attention_layernorm_tensor
            || fixed.contract.router.name != l1_cpu.direct_packed_router_tensor
            || fixed.contract.shared_gate_proj.name != l1_cpu.direct_packed_shared_gate_proj_tensor
            || fixed.contract.shared_up_proj.name != l1_cpu.direct_packed_shared_up_proj_tensor
            || fixed.contract.shared_down_proj.name != l1_cpu.direct_packed_shared_down_proj_tensor
            || fixed.contract.shared_expert_gate.name
                != l1_cpu.direct_packed_shared_expert_gate_tensor
        {
            return Err(model_error(
                "same-runtime full-L1 fixed suffix tensors drifted from the opaque-prefix CPU oracle",
            ));
        }
        command.commit_and_wait().map_err(|error| {
            model_error(format!(
                "same-runtime full-L1 one-command-buffer fence failed: {error}"
            ))
        })?;

        let l0_second_residual = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            self.l0_continuation.l0_second_residual(),
            QWEN80_HIDDEN,
            "same-runtime full-L1 fresh L0 second residual",
        )?;
        let l0_second_residual_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            self.l0_continuation.cpu_l0_layer_output(),
            &l0_second_residual,
            "same-runtime full-L1 fresh L0 second residual",
            1.0e-3,
        )?;
        let fresh_l0 = Qwen80SameRuntimeFreshL0Parity {
            first_residual: self
                .l0_continuation
                .first_residual
                .verify_after_fence(runtime)?,
            second_residual_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                self.l0_continuation.cpu_l0_layer_output(),
                "same-runtime full-L1 fresh L0 CPU second residual",
            )?,
            second_residual_device_f32le_sha256: qwen80_f32_vector_sha256(
                &l0_second_residual,
                "same-runtime full-L1 fresh L0 device second residual",
            )?,
            second_residual_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                self.l0_continuation.l0_second_residual(),
                "same-runtime full-L1 fresh L0 second residual",
            )?,
            second_residual_max_abs_error: l0_second_residual_max_abs_error,
        };
        let l1_prefix = self.verify_after_fence(runtime)?;
        let l1_true_moe_suffix =
            Self::collect_l1_true_moe_suffix_parity_after_fence(fixed, &l1_cpu)?;
        Ok(Qwen80SameRuntimeL0L1FullLayerParity {
            fresh_l0,
            l1_prefix,
            l1_true_moe_suffix,
            l0_dispatches: L0_DISPATCHES,
            l1_prefix_dispatches: L1_PREFIX_DISPATCHES,
            l1_moe_suffix_dispatches: L1_SUFFIX_DISPATCHES,
            total_dispatches: TOTAL_DISPATCHES,
            structural_kernel_names,
            same_runtime_same_command_buffer_required: true,
            single_fence_after_all_dispatches_required: true,
        })
    }
}

/// Metal-side cache/state bootstrap for Qwen80.  It allocates only exact
/// hybrid state and uploads only a caller-selected direct packed tensor.  No
/// BF16 tensor is opened, and no function in this type can generate a token.
#[cfg(target_os = "macos")]
pub struct Qwen80CompleteNativeRuntime {
    catalog: Qwen80CompleteArtifactCatalog,
    context: MetalContext,
    state: Qwen80NativeStateGeometry,
    linear_conv_state: PinnedBuffer,
    linear_recurrent_state: PinnedBuffer,
    full_attention_key_cache: PinnedBuffer,
    full_attention_value_cache: PinnedBuffer,
}

/// Process-local, non-pointer identity for one active DeltaNet state slice.
///
/// The direct-packed component path uses this only to bind an already
/// allocated runtime state arena to a sealed same-command-buffer witness. It
/// neither allocates state nor permits a caller to mutate or snapshot an
/// unrelated slot. Offsets/capacities are source-scheduled and the identity
/// is a SHA-256 value rather than a raw Metal pointer.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80LinearDeltaNetStateSlotDeviceBinding {
    pub layer: usize,
    pub linear_state_slot: usize,
    pub conv_state_offset_elements: usize,
    pub conv_state_capacity_elements: usize,
    pub recurrent_state_offset_elements: usize,
    pub recurrent_state_capacity_elements: usize,
    pub active_conv_state_buffer_identity_sha256: String,
    pub active_recurrent_state_buffer_identity_sha256: String,
}

#[cfg(target_os = "macos")]
impl Qwen80CompleteNativeRuntime {
    /// Admit one complete direct artifact, then move that one catalog into
    /// native Qwen-Next state construction.
    pub fn load(
        manifest_path: impl AsRef<Path>,
        admission: &CompleteBinaryAdmission,
        options: Qwen80CompleteRuntimeOptions,
    ) -> Result<Self> {
        let catalog = Qwen80CompleteArtifactCatalog::load(manifest_path, admission)?;
        Self::from_admitted_catalog(catalog, options)
    }

    /// Construct native Metal state from the one strict catalog admission
    /// already completed by this process.  No raw source tensor is opened and
    /// no packed payload is re-admitted here; per-tensor access still verifies
    /// its hash/header at the point it is actually uploaded or decoded.
    pub fn from_admitted_catalog(
        catalog: Qwen80CompleteArtifactCatalog,
        options: Qwen80CompleteRuntimeOptions,
    ) -> Result<Self> {
        Self::from_admitted_catalog_with_math(catalog, options, false)
    }

    /// Construct one diagnostic-only strict-Math native state body from an
    /// already admitted catalog.  This does not change the ordinary runtime
    /// policy: callers must opt in explicitly, and it still has no decoder,
    /// token, or benchmark meaning.  It exists so a bounded source-input
    /// component can make its fast-math-disabled receipt truthful without
    /// reopening or re-admitting the artifact.
    pub fn from_admitted_catalog_strict_math(
        catalog: Qwen80CompleteArtifactCatalog,
        options: Qwen80CompleteRuntimeOptions,
    ) -> Result<Self> {
        Self::from_admitted_catalog_with_math(catalog, options, true)
    }

    fn from_admitted_catalog_with_math(
        catalog: Qwen80CompleteArtifactCatalog,
        options: Qwen80CompleteRuntimeOptions,
        strict_math: bool,
    ) -> Result<Self> {
        // Keep the native-state entrypoint tied to the same source tokenizer
        // contract as the catalog preflight. A caller may not bypass this
        // check merely by asking to allocate state instead of a full token.
        let _tokenizer = tokenizer_from_source(&catalog.artifact)?;
        let state = Qwen80NativeStateGeometry::from_config(&catalog.config, options.max_seq_len)?;
        let context = if strict_math {
            MetalContext::new_with_trace_strict_math(options.trace_dispatch)?
        } else {
            MetalContext::new_with_trace(options.trace_dispatch)?
        };
        let linear_conv_state = context.new_buffer_checked(bytes_for_f32(
            state.linear_conv_state_elements,
            "Qwen80 linear convolution state",
        )?)?;
        let linear_recurrent_state = context.new_buffer_checked(bytes_for_f32(
            state.linear_recurrent_state_elements,
            "Qwen80 linear recurrent state",
        )?)?;
        let full_attention_key_cache = context.new_buffer_checked(bytes_for_f32(
            state.full_attention_key_cache_elements,
            "Qwen80 full-attention key cache",
        )?)?;
        let full_attention_value_cache = context.new_buffer_checked(bytes_for_f32(
            state.full_attention_value_cache_elements,
            "Qwen80 full-attention value cache",
        )?)?;
        let mut runtime = Self {
            catalog,
            context,
            state,
            linear_conv_state,
            linear_recurrent_state,
            full_attention_key_cache,
            full_attention_value_cache,
        };
        runtime.reset_state();
        Ok(runtime)
    }

    pub fn catalog(&self) -> &Qwen80CompleteArtifactCatalog {
        &self.catalog
    }

    pub fn state_geometry(&self) -> &Qwen80NativeStateGeometry {
        &self.state
    }

    /// Return the exact active state-arena slice that a source-scheduled
    /// DeltaNet layer would use. This is metadata-only: it does not create a
    /// command buffer, dispatch work, or expose raw pointer values.
    pub fn linear_deltanet_state_slot_device_binding(
        &self,
        layer: usize,
    ) -> Result<Qwen80LinearDeltaNetStateSlotDeviceBinding> {
        let contract = self
            .catalog
            .canonical_linear_deltanet_operator_contract(layer)?;
        contract.validate_device_resources(&contract.minimum_device_resources)?;
        let resources = &contract.minimum_device_resources;
        if contract.linear_state_slot >= self.state.linear_layers
            || resources.conv_state_capacity_elements > self.state.linear_conv_state_elements
            || resources.recurrent_state_capacity_elements
                > self.state.linear_recurrent_state_elements
        {
            return Err(model_error(
                "Qwen80 DeltaNet state-slot binding exceeds the allocated native arena",
            ));
        }
        Ok(Qwen80LinearDeltaNetStateSlotDeviceBinding {
            layer: contract.layer,
            linear_state_slot: contract.linear_state_slot,
            conv_state_offset_elements: resources.conv_state_offset_elements,
            conv_state_capacity_elements: resources.conv_state_capacity_elements,
            recurrent_state_offset_elements: resources.recurrent_state_offset_elements,
            recurrent_state_capacity_elements: resources.recurrent_state_capacity_elements,
            active_conv_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.linear_conv_state,
                "Qwen80 active DeltaNet convolution state arena",
            )?,
            active_recurrent_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.linear_recurrent_state,
                "Qwen80 active DeltaNet recurrent state arena",
            )?,
        })
    }

    pub fn device_name(&self) -> String {
        self.context.device_name()
    }

    /// Clear only native session state; the sealed packed catalog remains
    /// immutable and shared.  A scheduler may use one runtime body for many
    /// sessions only after it adds explicit per-session state slicing.
    pub fn reset_state(&mut self) {
        for buffer in [
            &self.linear_conv_state,
            &self.linear_recurrent_state,
            &self.full_attention_key_cache,
            &self.full_attention_value_cache,
        ] {
            let zeroes = vec![0u8; buffer.length() as usize];
            MetalContext::write_buffer_bytes(buffer, &zeroes);
        }
    }

    /// Upload one direct binary tensor's sign/scales to shared Metal memory.
    /// The payload remains compact; this function never reconstructs BF16 or
    /// f32 model weights.
    pub fn upload_direct_tensor(&self, name: &str) -> Result<Qwen80GpuBinaryTensor> {
        let payload = self.catalog.read_direct_tensor_payload(name)?;
        let header = parse_complete_binary_header(&payload)?;
        let scales = payload
            .get(header.scale_offset..header.sign_offset)
            .ok_or_else(|| model_error(format!("tensor {name:?} has truncated scale bytes")))?;
        let signs = payload
            .get(header.sign_offset..header.payload_bytes)
            .ok_or_else(|| model_error(format!("tensor {name:?} has truncated sign bytes")))?;
        if scales.len() != header.groups * 2
            || signs.len()
                != header
                    .groups
                    .checked_mul(header.group_size / 8)
                    .ok_or_else(|| model_error("packed sign byte count overflows usize"))?
        {
            return Err(model_error(format!(
                "tensor {name:?} compact payload sections disagree with its admitted header"
            )));
        }
        Ok(Qwen80GpuBinaryTensor {
            signs: self.context.new_buffer_with_bytes_checked(signs)?,
            scales: self.context.new_buffer_with_bytes_checked(scales)?,
            header,
        })
    }

    /// Upload the six exact layer-zero fixed-MoE compact payloads and reserve
    /// only the suffix work buffers prescribed by the source/static ABI.
    ///
    /// The caller still must obtain source-selected all-ten route bodies via
    /// [`Self::upload_all_ten_true_moe_device_bridge`] and keep both holders
    /// alive through one common token-command-buffer fence.  This method has
    /// no dispatch/commit/readback path and must not be treated as device
    /// execution evidence.
    pub fn upload_l0_true_moe_fixed_device_buffers(
        &self,
    ) -> Result<Qwen80L0TrueMoeFixedDeviceBuffers> {
        self.upload_canonical_linear_moe_fixed_device_buffers(0)
    }

    /// Upload the six fixed compact MoE bodies and the exact scratch layout
    /// for one canonical DeltaNet layer.  The returned holder deliberately
    /// retains the historical L0 type name for source compatibility, but its
    /// embedded contract is authoritative and may describe Layer 0, Layer 1,
    /// or another source-scheduled DeltaNet layer.  Routed bodies remain
    /// caller-supplied through a separately validated all-ten bridge.
    ///
    /// This is allocation/upload only: it neither creates a command buffer nor
    /// encodes, commits, fences, or reads back a device graph.
    pub fn upload_canonical_linear_moe_fixed_device_buffers(
        &self,
        layer: usize,
    ) -> Result<Qwen80L0TrueMoeFixedDeviceBuffers> {
        let contract = self.catalog.canonical_linear_moe_operator_contract(layer)?;
        contract.validate_against_catalog(&self.catalog)?;
        if contract.mixer.layer != layer
            || contract.post_attention_layernorm.shape != [QWEN80_HIDDEN]
            || contract.router.shape != [QWEN80_EXPERTS, QWEN80_HIDDEN]
            || contract.shared_gate_proj.shape != [QWEN80_SHARED_EXPERT_INTERMEDIATE, QWEN80_HIDDEN]
            || contract.shared_up_proj.shape != [QWEN80_SHARED_EXPERT_INTERMEDIATE, QWEN80_HIDDEN]
            || contract.shared_down_proj.shape != [QWEN80_HIDDEN, QWEN80_SHARED_EXPERT_INTERMEDIATE]
            || contract.shared_expert_gate.shape != [1, QWEN80_HIDDEN]
        {
            return Err(model_error(
                "Qwen80 canonical true-MoE fixed resource contract has unexpected geometry",
            ));
        }

        let postnorm = self.upload_direct_tensor(&contract.post_attention_layernorm.name)?;
        let router = self.upload_direct_tensor(&contract.router.name)?;
        let shared_gate_proj = self.upload_direct_tensor(&contract.shared_gate_proj.name)?;
        let shared_up_proj = self.upload_direct_tensor(&contract.shared_up_proj.name)?;
        let shared_down_proj = self.upload_direct_tensor(&contract.shared_down_proj.name)?;
        let shared_expert_gate = self.upload_direct_tensor(&contract.shared_expert_gate.name)?;

        let hidden = QWEN80_HIDDEN;
        let intermediate = QWEN80_SHARED_EXPERT_INTERMEDIATE;
        let routes = QWEN80_TOP_K;
        Ok(Qwen80L0TrueMoeFixedDeviceBuffers {
            contract,
            postnorm,
            router,
            shared_gate_proj,
            shared_up_proj,
            shared_down_proj,
            shared_expert_gate,
            postnorm_hidden: self
                .context
                .new_buffer_checked(bytes_for_f32(hidden, "Qwen80 L0 true-MoE postnorm hidden")?)?,
            router_logits: self.context.new_buffer_checked(bytes_for_f32(
                QWEN80_EXPERTS,
                "Qwen80 L0 true-MoE router logits",
            )?)?,
            router_probabilities: self.context.new_buffer_checked(bytes_for_f32(
                QWEN80_EXPERTS,
                "Qwen80 L0 true-MoE router probability scratch",
            )?)?,
            router_route_ids: self
                .context
                .new_buffer_checked(bytes_for_f32(routes, "Qwen80 L0 true-MoE router IDs")?)?,
            router_route_weights: self
                .context
                .new_buffer_checked(bytes_for_f32(routes, "Qwen80 L0 true-MoE router weights")?)?,
            route_guard: self
                .context
                .new_buffer_checked(bytes_for_f32(1, "Qwen80 L0 true-MoE route guard")?)?,
            route_gate: self.context.new_buffer_checked(bytes_for_f32(
                routes.checked_mul(QWEN80_MOE_INTERMEDIATE).ok_or_else(|| {
                    model_error("Qwen80 L0 true-MoE route gate element count overflowed")
                })?,
                "Qwen80 L0 true-MoE route gate",
            )?)?,
            route_up: self.context.new_buffer_checked(bytes_for_f32(
                routes.checked_mul(QWEN80_MOE_INTERMEDIATE).ok_or_else(|| {
                    model_error("Qwen80 L0 true-MoE route up element count overflowed")
                })?,
                "Qwen80 L0 true-MoE route up",
            )?)?,
            route_activated: self.context.new_buffer_checked(bytes_for_f32(
                routes.checked_mul(QWEN80_MOE_INTERMEDIATE).ok_or_else(|| {
                    model_error("Qwen80 L0 true-MoE route activation element count overflowed")
                })?,
                "Qwen80 L0 true-MoE route activation",
            )?)?,
            route_weighted: self.context.new_buffer_checked(bytes_for_f32(
                routes.checked_mul(hidden).ok_or_else(|| {
                    model_error("Qwen80 L0 true-MoE weighted route element count overflowed")
                })?,
                "Qwen80 L0 true-MoE weighted route outputs",
            )?)?,
            shared_gate: self.context.new_buffer_checked(bytes_for_f32(
                intermediate,
                "Qwen80 L0 true-MoE shared gate",
            )?)?,
            shared_up: self
                .context
                .new_buffer_checked(bytes_for_f32(intermediate, "Qwen80 L0 true-MoE shared up")?)?,
            shared_activated: self.context.new_buffer_checked(bytes_for_f32(
                intermediate,
                "Qwen80 L0 true-MoE shared activation",
            )?)?,
            shared_output: self
                .context
                .new_buffer_checked(bytes_for_f32(hidden, "Qwen80 L0 true-MoE shared output")?)?,
            shared_scalar_logit: self
                .context
                .new_buffer_checked(bytes_for_f32(1, "Qwen80 L0 true-MoE shared scalar logit")?)?,
            gated_shared: self
                .context
                .new_buffer_checked(bytes_for_f32(hidden, "Qwen80 L0 true-MoE gated shared")?)?,
            routed_sum: self
                .context
                .new_buffer_checked(bytes_for_f32(hidden, "Qwen80 L0 true-MoE routed sum")?)?,
            second_residual: self
                .context
                .new_buffer_checked(bytes_for_f32(hidden, "Qwen80 L0 true-MoE second residual")?)?,
        })
    }

    /// Begin a non-timed component command buffer for a caller that will
    /// encode the source-input DeltaNet prefix and its explicitly staged MoE
    /// suffix before one fence.  Constructing this buffer does not dispatch
    /// anything or relax the caller's quiet-lease requirement.
    pub fn begin_component_token_command_buffer(&self) -> TokenCommandBuffer<'_> {
        TokenCommandBuffer::new(&self.context)
    }

    /// Upload one already-admitted all-ten bridge while preserving a clone of
    /// the exact source-input DeltaNet first-residual device buffer.  The
    /// caller must keep the returned bridge and original encoder alive until
    /// the common command buffer has fenced.
    pub fn upload_all_ten_true_moe_device_bridge(
        &self,
        source_bridge: &Qwen80AllTenTrueMoeSourceBridge,
        first_residual: PinnedBuffer,
    ) -> Result<Qwen80AllTenTrueMoeDeviceBridge> {
        source_bridge.upload_with_first_residual(&self.context, first_residual)
    }

    /// Type-narrowed version of [`Self::upload_all_ten_true_moe_device_bridge`]
    /// for the source-input L0 path.  The all-ten bridge receives a retained
    /// clone of the exact allocation produced by
    /// [`Self::encode_source_token_first_linear_deltanet_into`], rather than
    /// an arbitrary equal-length buffer supplied by a caller.
    pub fn bind_source_input_first_residual_to_all_ten(
        &self,
        source_bridge: &Qwen80AllTenTrueMoeSourceBridge,
        first_residual: &Qwen80SourceInputFirstResidualEncoder,
    ) -> Result<Qwen80AllTenTrueMoeDeviceBridge> {
        self.upload_all_ten_true_moe_device_bridge(
            source_bridge,
            first_residual.first_residual().to_owned(),
        )
    }

    /// Encode a non-synthetic L0 DeltaNet prefix from one source-token
    /// embedding into an already-open command buffer.  It retains all input,
    /// compact-weight, temporary, and first-residual handles for a future
    /// same-buffer true-MoE suffix; it deliberately does *not* commit, fence,
    /// read back, or call a downstream router/expert kernel itself.
    ///
    /// The input embedding is resolved from the admitted compact artifact on
    /// the CPU only as a source/reference gather, then uploaded as the exact
    /// `[2048]` L0 input.  This is not a native embedding-gather claim and
    /// cannot become a full decoder input without its own device proof.
    pub fn encode_source_token_first_linear_deltanet_into(
        &self,
        command: &mut TokenCommandBuffer<'_>,
        token_id: u32,
    ) -> Result<Qwen80SourceInputFirstResidualEncoder> {
        let contract = self
            .catalog
            .canonical_linear_deltanet_operator_contract(0)?;
        contract.validate_device_resources(&contract.minimum_device_resources)?;
        if contract.layer != 0
            || contract.linear_state_slot != 0
            || contract.minimum_device_resources.conv_state_offset_elements != 0
            || contract
                .minimum_device_resources
                .recurrent_state_offset_elements
                != 0
        {
            return Err(model_error(
                "source-input Qwen80 DeltaNet encoder did not bind layer/state slot zero",
            ));
        }
        let embedding = self.catalog.execute_embedding_lookup_cpu_oracle(token_id)?;
        let cpu_input = Qwen80CanonicalLinearLayerCpuInput::with_zero_state(embedding.hidden);
        let expected = self
            .catalog
            .execute_canonical_linear_deltanet_cpu_oracle(&contract, &cpu_input)?;
        let layout = &contract.layout;
        let input_f32le_sha256 =
            qwen80_f32_vector_sha256(&cpu_input.hidden, "source-input DeltaNet embedding hidden")?;
        let initial_conv_state_f32le_sha256 = qwen80_f32_vector_sha256(
            &cpu_input.state.conv_state,
            "source-input DeltaNet initial convolution state",
        )?;
        let initial_recurrent_state_f32le_sha256 = qwen80_f32_vector_sha256(
            &cpu_input.state.recurrent_state,
            "source-input DeltaNet initial recurrent state",
        )?;

        let input_layernorm = self.upload_direct_tensor(&contract.input_layernorm.name)?;
        let qkvz = self.upload_direct_tensor(&contract.mixer.in_proj_qkvz.name)?;
        let ba = self.upload_direct_tensor(&contract.mixer.in_proj_ba.name)?;
        let conv = self.upload_direct_tensor(&contract.mixer.causal_conv1d.name)?;
        let a_log = self.upload_direct_tensor(&contract.mixer.a_log.name)?;
        let dt_bias = self.upload_direct_tensor(&contract.mixer.dt_bias.name)?;
        let gated_norm = self.upload_direct_tensor(&contract.mixer.gated_rms_norm.name)?;
        let out_proj = self.upload_direct_tensor(&contract.mixer.out_proj.name)?;
        for binding in contract.required_bindings() {
            let expected_bytes = contract
                .minimum_device_resources
                .direct_packed_payload_bytes
                .get(&binding.name)
                .copied()
                .ok_or_else(|| {
                    model_error(format!(
                        "source-input Qwen80 DeltaNet encoder has no compact-byte requirement for {:?}",
                        binding.name
                    ))
                })?;
            if self
                .catalog
                .direct_tensor_header(&binding.name)?
                .payload_bytes
                != expected_bytes
            {
                return Err(model_error(format!(
                    "source-input Qwen80 DeltaNet compact-byte requirement drifted for {:?}",
                    binding.name
                )));
            }
        }

        let input = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&cpu_input.hidden))?;
        let rollback_conv_state = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&cpu_input.state.conv_state))?;
        let rollback_recurrent_state =
            self.context
                .new_buffer_with_bytes_checked(bytemuck::cast_slice(
                    &cpu_input.state.recurrent_state,
                ))?;
        let normalized = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            "Qwen80 source-input DeltaNet input RMSNorm output",
        )?)?;
        let qkvz_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.qkvz_projection_elements()?,
            "Qwen80 source-input DeltaNet QKVZ projection",
        )?)?;
        let ba_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.ba_projection_elements()?,
            "Qwen80 source-input DeltaNet BA projection",
        )?)?;
        let repeated_query = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "Qwen80 source-input DeltaNet repeated query",
        )?)?;
        let repeated_key = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "Qwen80 source-input DeltaNet repeated key",
        )?)?;
        let convolved_value = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "Qwen80 source-input DeltaNet convolved value",
        )?)?;
        let z = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "Qwen80 source-input DeltaNet Z gate",
        )?)?;
        let decay = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_heads,
            "Qwen80 source-input DeltaNet decay",
        )?)?;
        let beta = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_heads,
            "Qwen80 source-input DeltaNet beta",
        )?)?;
        let recurrent_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "Qwen80 source-input DeltaNet recurrent output",
        )?)?;
        let gated_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "Qwen80 source-input DeltaNet gated RMSNorm output",
        )?)?;
        let mixer_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            "Qwen80 source-input DeltaNet mixer output",
        )?)?;
        let first_residual = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            "Qwen80 source-input DeltaNet first residual",
        )?)?;

        MetalContext::write_buffer_bytes(
            &self.linear_conv_state,
            bytemuck::cast_slice(&cpu_input.state.conv_state),
        );
        MetalContext::write_buffer_bytes(
            &self.linear_recurrent_state,
            bytemuck::cast_slice(&cpu_input.state.recurrent_state),
        );
        let dispatches_before = command.dispatch_count();
        qwen_next_direct_packed_input_rmsnorm_tcb(
            command,
            &input,
            &input_layernorm.signs,
            &input_layernorm.scales,
            &normalized,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
            QWEN80_RMS_EPS,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            command,
            &qkvz.signs,
            &qkvz.scales,
            &normalized,
            &qkvz_output,
            layout.qkvz_projection_elements()?,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            command,
            &ba.signs,
            &ba.scales,
            &normalized,
            &ba_output,
            layout.ba_projection_elements()?,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_qkvz_rearrange_conv_l2_tcb(
            command,
            &qkvz_output,
            &conv.signs,
            &conv.scales,
            &self.linear_conv_state,
            contract.minimum_device_resources.conv_state_offset_elements,
            &repeated_query,
            &repeated_key,
            &convolved_value,
            &z,
            layout.key_heads,
            layout.value_heads_per_key_head,
            layout.key_head_dim,
            layout.value_head_dim,
            layout.conv_kernel,
            QWEN80_GROUP_SIZE,
            QWEN80_RMS_EPS,
        )?;
        qwen_next_ba_to_decay_beta_tcb(
            command,
            &ba_output,
            &a_log.signs,
            &a_log.scales,
            &dt_bias.signs,
            &dt_bias.scales,
            &decay,
            &beta,
            layout.key_heads,
            layout.value_heads_per_key_head,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_gated_delta_decode_single_tcb(
            command,
            &self.linear_recurrent_state,
            &repeated_query,
            &repeated_key,
            &convolved_value,
            &decay,
            &beta,
            &recurrent_output,
            layout.value_heads,
            layout.key_head_dim,
            layout.value_head_dim,
        )?;
        qwen_next_deltanet_gated_rmsnorm_tcb(
            command,
            &recurrent_output,
            &z,
            &gated_norm.signs,
            &gated_norm.scales,
            &gated_output,
            layout.value_heads,
            layout.value_head_dim,
            QWEN80_GROUP_SIZE,
            QWEN80_RMS_EPS,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            command,
            &out_proj.signs,
            &out_proj.scales,
            &gated_output,
            &mixer_output,
            layout.hidden_elements,
            layout.value_elements()?,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_add_residual_tcb(
            command,
            &input,
            &mixer_output,
            &first_residual,
            layout.hidden_elements,
        )?;
        let dispatches_encoded = command
            .dispatch_count()
            .checked_sub(dispatches_before)
            .ok_or_else(|| model_error("source-input DeltaNet dispatch count underflowed"))?;
        if dispatches_encoded != 9 {
            return Err(model_error(format!(
                "source-input DeltaNet encoded {dispatches_encoded} dispatches, expected 9"
            )));
        }
        Ok(Qwen80SourceInputFirstResidualEncoder {
            source_token_id: token_id,
            source_embedding_tensor: embedding.direct_packed_embedding_tensor,
            layer: contract.layer,
            linear_state_slot: contract.linear_state_slot,
            input_f32le_sha256,
            initial_conv_state_f32le_sha256,
            initial_recurrent_state_f32le_sha256,
            expected_first_residual: expected.mixer_residual_output,
            expected_next_conv_state: expected.next_state.conv_state,
            expected_next_recurrent_state: expected.next_state.recurrent_state,
            rollback_conv_state,
            rollback_recurrent_state,
            _input: input,
            _normalized: normalized,
            _qkvz_output: qkvz_output,
            _ba_output: ba_output,
            _repeated_query: repeated_query,
            _repeated_key: repeated_key,
            _convolved_value: convolved_value,
            _z: z,
            _decay: decay,
            _beta: beta,
            _recurrent_output: recurrent_output,
            _gated_output: gated_output,
            _mixer_output: mixer_output,
            first_residual,
            _input_layernorm: input_layernorm,
            _qkvz: qkvz,
            _ba: ba,
            _conv: conv,
            _a_log: a_log,
            _dt_bias: dt_bias,
            _gated_norm: gated_norm,
            _out_proj: out_proj,
            dispatches_encoded,
        })
    }

    /// Consume the exact owners produced by the canonical source-token L0
    /// 9+14 encoder and turn them into an opaque, same-runtime continuation
    /// capability.  This is intentionally stricter than checking a buffer
    /// length or a dispatch count: it refuses a caller-provided equal-sized
    /// allocation, an untraced command buffer, a reordered L0 graph, route
    /// drift, and a different fixed-MoE suffix.
    #[allow(clippy::too_many_lines)]
    pub fn certify_source_token_l0_true_moe_continuation(
        &self,
        command: &TokenCommandBuffer<'_>,
        first_residual: Qwen80SourceInputFirstResidualEncoder,
        route_bridge: Qwen80AllTenTrueMoeDeviceBridge,
        fixed: Qwen80L0TrueMoeFixedDeviceBuffers,
    ) -> Result<Qwen80CanonicalSourceTokenL0TrueMoeContinuation> {
        const SOURCE_TOKEN_ID: u32 = 1;
        if command.dispatch_count() != QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS.len() {
            return Err(model_error(format!(
                "canonical source-token L0 continuation requires exactly {} preceding dispatches, observed {}",
                QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS.len(),
                command.dispatch_count(),
            )));
        }
        qwen80_require_exact_structural_kernel_trace(
            command.structural_kernel_names(),
            &QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS,
            "canonical source-token L0 continuation",
        )?;
        if first_residual.source_token_id != SOURCE_TOKEN_ID
            || first_residual.layer != 0
            || first_residual.linear_state_slot != 0
            || first_residual.dispatches_encoded != 9
        {
            return Err(model_error(
                "canonical source-token L0 continuation did not receive the exact L0 DeltaNet prefix owner",
            ));
        }
        fixed.contract.validate_against_catalog(&self.catalog)?;
        if fixed.contract.mixer.layer != 0 || fixed.contract.mixer.linear_state_slot != 0 {
            return Err(model_error(
                "canonical source-token L0 continuation received a non-L0 fixed-MoE suffix",
            ));
        }
        let route_payloads = route_bridge.source_bridge().route_payloads();
        if route_payloads.layer() != 0
            || route_payloads.manifest_seal_sha256() != self.catalog.manifest_seal()
            || route_payloads.source_revision() != self.catalog.config.source_revision
        {
            return Err(model_error(
                "canonical source-token L0 continuation route payloads drifted from the admitted L0 catalog",
            ));
        }
        qwen80_require_same_live_pinned_allocation(
            first_residual.first_residual(),
            route_bridge.first_residual(),
            "canonical source-token L0 first-residual route handoff",
        )?;
        if fixed.second_residual.length() as usize
            != bytes_for_f32(QWEN80_HIDDEN, "canonical source-token L0 second residual")?
        {
            return Err(model_error(
                "canonical source-token L0 fixed suffix did not retain a 2048-f32 second residual",
            ));
        }

        // Recompute the non-synthetic source input from this admitted catalog
        // inside the same runtime owner.  No prior receipt vector is accepted
        // as a substitute for either the L0 input or its CPU parity shadow.
        let embedding = self
            .catalog
            .execute_embedding_lookup_cpu_oracle(SOURCE_TOKEN_ID)?;
        let cpu_l0_input = Qwen80CanonicalLinearLayerCpuInput::with_zero_state(embedding.hidden);
        let cpu_l0 = self
            .catalog
            .execute_first_linear_layer_cpu_moe_oracle(&cpu_l0_input)?;
        if first_residual.input_f32le_sha256
            != qwen80_f32_vector_sha256(&cpu_l0_input.hidden, "canonical source-token L0 input")?
            || first_residual.initial_conv_state_f32le_sha256
                != qwen80_f32_vector_sha256(
                    &cpu_l0_input.state.conv_state,
                    "canonical source-token L0 convolution pre-state",
                )?
            || first_residual.initial_recurrent_state_f32le_sha256
                != qwen80_f32_vector_sha256(
                    &cpu_l0_input.state.recurrent_state,
                    "canonical source-token L0 recurrent pre-state",
                )?
        {
            return Err(model_error(
                "canonical source-token L0 prefix owner does not match the freshly derived source input/state",
            ));
        }
        Self::require_parity(
            &cpu_l0.mixer.mixer_residual_output,
            &first_residual.expected_first_residual,
            "canonical source-token L0 first-residual CPU shadow",
            0.0,
        )?;
        route_payloads.require_router_route(&cpu_l0.route, 0.0)?;
        Ok(Qwen80CanonicalSourceTokenL0TrueMoeContinuation {
            first_residual,
            _route_bridge: route_bridge,
            fixed,
            cpu_l0,
            l0_structural_kernel_names: command
                .structural_kernel_names()
                .expect("checked exact structural trace")
                .to_vec(),
            runtime_state_arena_owner: Qwen80SameRuntimeStateArenaOwnerIdentity::from_runtime(
                self,
            )?,
        })
    }

    /// Append the exact nine-dispatch source Layer-1 DeltaNet prefix to an
    /// already-open command buffer which still owns the live Layer-0 second
    /// residual allocation.  This is intentionally a same-runtime API: it
    /// accepts only the opaque canonical L0 capability, which retains the
    /// live allocation and freshly derived CPU shadow together.  It never
    /// accepts a receipt JSON, caller-supplied `PinnedBuffer`, dispatch count,
    /// or reuploaded f32 bytes.
    ///
    /// The method is deliberately locked to source token 1, Layer 1, and
    /// DeltaNet state slot 1.  It stops after the Layer-1 first residual; no
    /// post-attention norm, MoE, second residual, later layer, token, or
    /// decoder transition is implied.
    #[allow(clippy::too_many_lines)]
    pub fn encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into(
        &self,
        command: &mut TokenCommandBuffer<'_>,
        continuation: Qwen80CanonicalSourceTokenL0TrueMoeContinuation,
    ) -> Result<Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder> {
        const SOURCE_TOKEN_ID: u32 = 1;
        const L0_DISPATCHES: usize = 23;
        if command.dispatch_count() != L0_DISPATCHES {
            return Err(model_error(format!(
                "same-runtime Layer-1 prefix requires exactly {L0_DISPATCHES} preceding L0 dispatches in the open command buffer, observed {}",
                command.dispatch_count()
            )));
        }
        continuation.require_l0_trace()?;
        qwen80_require_exact_structural_kernel_trace(
            command.structural_kernel_names(),
            &QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS,
            "same-runtime Layer-1 prefix",
        )?;
        let source_token_id = continuation.first_residual.source_token_id;
        if source_token_id != SOURCE_TOKEN_ID {
            return Err(model_error(
                "same-runtime Layer-1 prefix continuation did not retain source token one",
            ));
        }
        continuation.runtime_state_arena_owner.require_runtime_owner(
            self,
            "same-runtime Layer-1 prefix refuses an opaque L0 capability from another runtime state arena",
        )?;
        let retained_l0_output = continuation.l0_second_residual().to_owned();
        let cpu_l1_input = Qwen80CanonicalLinearLayerCpuInput::with_zero_state(
            continuation.cpu_l0_layer_output().to_vec(),
        );
        let contract = self
            .catalog
            .canonical_linear_deltanet_operator_contract(1)?;
        contract.validate_device_resources(&contract.minimum_device_resources)?;
        let resources = &contract.minimum_device_resources;
        if contract.layer != 1
            || contract.linear_state_slot != 1
            || resources.conv_state_offset_elements == 0
            || resources.recurrent_state_offset_elements == 0
        {
            return Err(model_error(
                "same-runtime Layer-1 prefix did not bind source layer/state slot one",
            ));
        }
        let live_state = self.linear_deltanet_state_slot_device_binding(1)?;
        if live_state.layer != contract.layer
            || live_state.linear_state_slot != contract.linear_state_slot
            || live_state.conv_state_offset_elements != resources.conv_state_offset_elements
            || live_state.conv_state_capacity_elements != resources.conv_state_capacity_elements
            || live_state.recurrent_state_offset_elements
                != resources.recurrent_state_offset_elements
            || live_state.recurrent_state_capacity_elements
                != resources.recurrent_state_capacity_elements
        {
            return Err(model_error(
                "same-runtime Layer-1 active state arena drifted from its source-scheduled slot-one contract",
            ));
        }
        if retained_l0_output.length() as usize
            != bytes_for_f32(QWEN80_HIDDEN, "same-runtime Layer-1 retained L0 output")?
        {
            return Err(model_error(
                "same-runtime Layer-1 input is not the exact retained 2048-f32 L0 output",
            ));
        }
        cpu_l1_input.validate()?;
        if cpu_l1_input
            .state
            .conv_state
            .iter()
            .chain(cpu_l1_input.state.recurrent_state.iter())
            .any(|value| value.to_bits() != 0)
        {
            return Err(model_error(
                "same-runtime source-token Layer-1 prefix requires a zeroed slot-one DeltaNet state",
            ));
        }
        let layout = &contract.layout;
        let expected = self
            .catalog
            .execute_canonical_linear_deltanet_cpu_oracle(&contract, &cpu_l1_input)?;
        if expected.layer != 1 || expected.linear_state_slot != 1 {
            return Err(model_error(
                "same-runtime Layer-1 CPU oracle did not retain source layer/state slot one",
            ));
        }
        let input_f32le_sha256 = qwen80_f32_vector_sha256(
            &cpu_l1_input.hidden,
            "same-runtime Layer-1 retained L0 CPU shadow",
        )?;
        let input_buffer_identity_sha256 = qwen80_pinned_buffer_identity_sha256(
            &retained_l0_output,
            "same-runtime retained L0 second residual passed to Layer-1",
        )?;

        // A rollback witness must snapshot the live slot-one state before
        // mutation.  Never overwrite this arena with CPU zeroes merely to
        // make a later receipt look like a fresh runtime: a reused runtime
        // must fail closed unless the actual source slot is still zero.
        let live_initial_conv_state = Self::device_f32_snapshot_at_offset(
            &self.linear_conv_state,
            resources.conv_state_offset_elements,
            cpu_l1_input.state.conv_state.len(),
            "same-runtime Layer-1 live initial convolution state",
        )?;
        let live_initial_recurrent_state = Self::device_f32_snapshot_at_offset(
            &self.linear_recurrent_state,
            resources.recurrent_state_offset_elements,
            cpu_l1_input.state.recurrent_state.len(),
            "same-runtime Layer-1 live initial recurrent state",
        )?;
        if live_initial_conv_state
            .iter()
            .chain(live_initial_recurrent_state.iter())
            .any(|value| value.to_bits() != 0)
        {
            return Err(model_error(
                "same-runtime Layer-1 prefix refuses a reused non-zero slot-one state instead of forging a rollback checkpoint",
            ));
        }
        Self::require_parity(
            &cpu_l1_input.state.conv_state,
            &live_initial_conv_state,
            "same-runtime Layer-1 live initial convolution state",
            0.0,
        )?;
        Self::require_parity(
            &cpu_l1_input.state.recurrent_state,
            &live_initial_recurrent_state,
            "same-runtime Layer-1 live initial recurrent state",
            0.0,
        )?;
        let initial_conv_state_f32le_sha256 = qwen80_f32_vector_sha256(
            &live_initial_conv_state,
            "same-runtime Layer-1 snapshotted initial convolution state",
        )?;
        let initial_recurrent_state_f32le_sha256 = qwen80_f32_vector_sha256(
            &live_initial_recurrent_state,
            "same-runtime Layer-1 snapshotted initial recurrent state",
        )?;

        let input_layernorm = self.upload_direct_tensor(&contract.input_layernorm.name)?;
        let qkvz = self.upload_direct_tensor(&contract.mixer.in_proj_qkvz.name)?;
        let ba = self.upload_direct_tensor(&contract.mixer.in_proj_ba.name)?;
        let conv = self.upload_direct_tensor(&contract.mixer.causal_conv1d.name)?;
        let a_log = self.upload_direct_tensor(&contract.mixer.a_log.name)?;
        let dt_bias = self.upload_direct_tensor(&contract.mixer.dt_bias.name)?;
        let gated_norm = self.upload_direct_tensor(&contract.mixer.gated_rms_norm.name)?;
        let out_proj = self.upload_direct_tensor(&contract.mixer.out_proj.name)?;
        for binding in contract.required_bindings() {
            let expected_bytes = resources
                .direct_packed_payload_bytes
                .get(&binding.name)
                .copied()
                .ok_or_else(|| {
                    model_error(format!(
                        "same-runtime Layer-1 prefix has no compact-byte requirement for {:?}",
                        binding.name
                    ))
                })?;
            if self
                .catalog
                .direct_tensor_header(&binding.name)?
                .payload_bytes
                != expected_bytes
            {
                return Err(model_error(format!(
                    "same-runtime Layer-1 compact-byte requirement drifted for {:?}",
                    binding.name
                )));
            }
        }

        let rollback_conv_state = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&live_initial_conv_state))?;
        let rollback_recurrent_state = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&live_initial_recurrent_state))?;

        let normalized = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            "same-runtime Layer-1 input RMSNorm output",
        )?)?;
        let qkvz_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.qkvz_projection_elements()?,
            "same-runtime Layer-1 QKVZ projection",
        )?)?;
        let ba_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.ba_projection_elements()?,
            "same-runtime Layer-1 BA projection",
        )?)?;
        let repeated_query = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "same-runtime Layer-1 repeated query",
        )?)?;
        let repeated_key = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "same-runtime Layer-1 repeated key",
        )?)?;
        let convolved_value = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "same-runtime Layer-1 convolved value",
        )?)?;
        let z = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "same-runtime Layer-1 Z gate",
        )?)?;
        let decay = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_heads,
            "same-runtime Layer-1 decay",
        )?)?;
        let beta = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_heads,
            "same-runtime Layer-1 beta",
        )?)?;
        let recurrent_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "same-runtime Layer-1 recurrent output",
        )?)?;
        let gated_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "same-runtime Layer-1 gated RMSNorm output",
        )?)?;
        let mixer_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            "same-runtime Layer-1 mixer output",
        )?)?;
        let first_residual = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            "same-runtime Layer-1 first residual",
        )?)?;

        let dispatches_before = command.dispatch_count();
        qwen_next_direct_packed_input_rmsnorm_tcb(
            command,
            &retained_l0_output,
            &input_layernorm.signs,
            &input_layernorm.scales,
            &normalized,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
            QWEN80_RMS_EPS,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            command,
            &qkvz.signs,
            &qkvz.scales,
            &normalized,
            &qkvz_output,
            layout.qkvz_projection_elements()?,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            command,
            &ba.signs,
            &ba.scales,
            &normalized,
            &ba_output,
            layout.ba_projection_elements()?,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_qkvz_rearrange_conv_l2_tcb(
            command,
            &qkvz_output,
            &conv.signs,
            &conv.scales,
            &self.linear_conv_state,
            resources.conv_state_offset_elements,
            &repeated_query,
            &repeated_key,
            &convolved_value,
            &z,
            layout.key_heads,
            layout.value_heads_per_key_head,
            layout.key_head_dim,
            layout.value_head_dim,
            layout.conv_kernel,
            QWEN80_GROUP_SIZE,
            QWEN80_RMS_EPS,
        )?;
        qwen_next_ba_to_decay_beta_tcb(
            command,
            &ba_output,
            &a_log.signs,
            &a_log.scales,
            &dt_bias.signs,
            &dt_bias.scales,
            &decay,
            &beta,
            layout.key_heads,
            layout.value_heads_per_key_head,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_gated_delta_decode_single_at_state_offset_tcb(
            command,
            &self.linear_recurrent_state,
            resources.recurrent_state_offset_elements,
            &repeated_query,
            &repeated_key,
            &convolved_value,
            &decay,
            &beta,
            &recurrent_output,
            layout.value_heads,
            layout.key_head_dim,
            layout.value_head_dim,
        )?;
        qwen_next_deltanet_gated_rmsnorm_tcb(
            command,
            &recurrent_output,
            &z,
            &gated_norm.signs,
            &gated_norm.scales,
            &gated_output,
            layout.value_heads,
            layout.value_head_dim,
            QWEN80_GROUP_SIZE,
            QWEN80_RMS_EPS,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            command,
            &out_proj.signs,
            &out_proj.scales,
            &gated_output,
            &mixer_output,
            layout.hidden_elements,
            layout.value_elements()?,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_add_residual_tcb(
            command,
            &retained_l0_output,
            &mixer_output,
            &first_residual,
            layout.hidden_elements,
        )?;
        let dispatches_encoded = command
            .dispatch_count()
            .checked_sub(dispatches_before)
            .ok_or_else(|| model_error("same-runtime Layer-1 dispatch count underflowed"))?;
        if dispatches_encoded != 9 {
            return Err(model_error(format!(
                "same-runtime Layer-1 prefix encoded {dispatches_encoded} dispatches, expected 9"
            )));
        }
        if command.dispatch_count() != 32 {
            return Err(model_error(format!(
                "same-runtime Layer-1 prefix produced {} total dispatches, expected 32",
                command.dispatch_count()
            )));
        }
        Ok(Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder {
            source_token_id,
            layer: contract.layer,
            linear_state_slot: contract.linear_state_slot,
            input_f32le_sha256,
            input_buffer_identity_sha256,
            expected_input: cpu_l1_input.hidden,
            initial_conv_state_f32le_sha256,
            initial_recurrent_state_f32le_sha256,
            expected_first_residual: expected.mixer_residual_output,
            expected_next_conv_state: expected.next_state.conv_state,
            expected_next_recurrent_state: expected.next_state.recurrent_state,
            conv_state_offset_elements: resources.conv_state_offset_elements,
            conv_state_capacity_elements: resources.conv_state_capacity_elements,
            recurrent_state_offset_elements: resources.recurrent_state_offset_elements,
            recurrent_state_capacity_elements: resources.recurrent_state_capacity_elements,
            rollback_conv_state,
            rollback_recurrent_state,
            input: retained_l0_output,
            _normalized: normalized,
            _qkvz_output: qkvz_output,
            _ba_output: ba_output,
            _repeated_query: repeated_query,
            _repeated_key: repeated_key,
            _convolved_value: convolved_value,
            _z: z,
            _decay: decay,
            _beta: beta,
            _recurrent_output: recurrent_output,
            _gated_output: gated_output,
            _mixer_output: mixer_output,
            first_residual,
            _input_layernorm: input_layernorm,
            _qkvz: qkvz,
            _ba: ba,
            _conv: conv,
            _a_log: a_log,
            _dt_bias: dt_bias,
            _gated_norm: gated_norm,
            _out_proj: out_proj,
            l0_continuation: continuation,
            dispatches_encoded,
        })
    }

    fn direct_cpu_matvec_reference(
        &self,
        name: &str,
        input: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        if input.len() != cols {
            return Err(model_error(format!(
                "CPU parity input for {name:?} has {} elements, expected {cols}",
                input.len()
            )));
        }
        let payload = self.catalog.read_direct_tensor_payload(name)?;
        let (header, weights) = decode_complete_binary_f32(&payload)?;
        if header.shape != [rows, cols] {
            return Err(model_error(format!(
                "CPU parity matrix {name:?} has {:?}, expected [{rows}, {cols}]",
                header.shape
            )));
        }
        let mut output = vec![0.0f32; rows];
        for row in 0..rows {
            let row_weights = &weights[row * cols..(row + 1) * cols];
            let mut sum = 0.0f32;
            for (&weight, &value) in row_weights.iter().zip(input) {
                sum += weight * value;
            }
            if !sum.is_finite() {
                return Err(model_error(format!(
                    "CPU direct packed parity for {name:?} produced a non-finite result at row {row}"
                )));
            }
            output[row] = sum;
        }
        Ok(output)
    }

    fn direct_cpu_vector_reference(&self, name: &str, elements: usize) -> Result<Vec<f32>> {
        let payload = self.catalog.read_direct_tensor_payload(name)?;
        let (header, values) = decode_complete_binary_f32(&payload)?;
        if header.shape != [elements] || values.len() != elements {
            return Err(model_error(format!(
                "CPU parity vector {name:?} has shape {:?}/{} elements, expected [{elements}]",
                header.shape,
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(model_error(format!(
                "CPU direct packed parity vector {name:?} contains a non-finite value"
            )));
        }
        Ok(values)
    }

    fn max_abs_error(expected: &[f32], observed: &[f32], label: &str) -> Result<f32> {
        if expected.len() != observed.len() {
            return Err(model_error(format!(
                "{label} parity length mismatch: expected {}, observed {}",
                expected.len(),
                observed.len()
            )));
        }
        let mut max_error = 0.0f32;
        for (index, (&expected, &observed)) in expected.iter().zip(observed).enumerate() {
            if !expected.is_finite() || !observed.is_finite() {
                return Err(model_error(format!(
                    "{label} parity is non-finite at element {index}: expected={expected}, observed={observed}"
                )));
            }
            max_error = max_error.max((expected - observed).abs());
        }
        Ok(max_error)
    }

    fn require_parity(
        expected: &[f32],
        observed: &[f32],
        label: &str,
        tolerance: f32,
    ) -> Result<f32> {
        let error = Self::max_abs_error(expected, observed, label)?;
        if error > tolerance {
            return Err(model_error(format!(
                "{label} direct-packed Metal parity failed: max_abs_error={error}, tolerance={tolerance}"
            )));
        }
        Ok(error)
    }

    /// Validate the CPU oracle shadow against the live retained Metal input
    /// for the L0→L1 continuation.  The opaque continuation and pinned-buffer
    /// identity prove custody of the same allocation; the two f32 snapshots
    /// need only meet the component parity tolerance and must retain separate
    /// hashes for audit.  A bitwise hash match would incorrectly reject
    /// ordinary bounded CPU/Metal floating-point divergence.
    fn validate_same_runtime_l1_input_parity(
        cpu_shadow: &[f32],
        device_snapshot: &[f32],
    ) -> Result<(f32, String)> {
        const TOLERANCE: f32 = 1.0e-3;
        let max_abs_error = Self::require_parity(
            cpu_shadow,
            device_snapshot,
            "same-runtime Layer-1 retained L0 input",
            TOLERANCE,
        )?;
        let device_f32le_sha256 = qwen80_f32_vector_sha256(
            device_snapshot,
            "same-runtime Layer-1 retained L0 device input",
        )?;
        Ok((max_abs_error, device_f32le_sha256))
    }

    fn device_f32_snapshot(
        buffer: &PinnedBuffer,
        elements: usize,
        label: &str,
    ) -> Result<Vec<f32>> {
        let required_bytes = bytes_for_f32(elements, label)?;
        if buffer.length() < required_bytes as u64 {
            return Err(model_error(format!(
                "{label} device snapshot needs {required_bytes} bytes but buffer has {}",
                buffer.length()
            )));
        }
        Ok(unsafe {
            std::slice::from_raw_parts(buffer.contents() as *const f32, elements).to_vec()
        })
    }

    /// Snapshot one explicit f32 slice of a source-scheduled shared state
    /// arena.  This remains private so callers cannot mistake a copied host
    /// vector for a transferable continuation resource; it exists only for
    /// post-fence parity of the live same-runtime owner.
    fn device_f32_snapshot_at_offset(
        buffer: &PinnedBuffer,
        offset_elements: usize,
        elements: usize,
        label: &str,
    ) -> Result<Vec<f32>> {
        let offset_bytes = bytes_for_f32(offset_elements, label)?;
        let required_bytes = bytes_for_f32(elements, label)?;
        let end = offset_bytes
            .checked_add(required_bytes)
            .ok_or_else(|| model_error(format!("{label} snapshot byte range overflows usize")))?;
        if end > buffer.length() as usize {
            return Err(model_error(format!(
                "{label} device snapshot needs bytes [{offset_bytes}, {end}) but buffer has {}",
                buffer.length()
            )));
        }
        Ok(unsafe {
            std::slice::from_raw_parts(
                (buffer.contents() as *const u8).add(offset_bytes) as *const f32,
                elements,
            )
            .to_vec()
        })
    }

    fn device_u32_snapshot(
        buffer: &PinnedBuffer,
        elements: usize,
        label: &str,
    ) -> Result<Vec<u32>> {
        let required_bytes = elements
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| model_error(format!("{label} u32 snapshot byte count overflows")))?;
        if buffer.length() < required_bytes as u64 {
            return Err(model_error(format!(
                "{label} device snapshot needs {required_bytes} bytes but buffer has {}",
                buffer.length()
            )));
        }
        Ok(unsafe {
            std::slice::from_raw_parts(buffer.contents() as *const u32, elements).to_vec()
        })
    }

    fn fixture_row(width: usize, seed: usize) -> Vec<f32> {
        (0..width)
            .map(|index| {
                let numerator = ((index * 73 + seed * 41 + 19) % 4093) as f32 - 2046.0;
                numerator / 4093.0
            })
            .collect()
    }

    fn source_ba_to_decay_beta(
        ba: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if ba.len() != QWEN80_LINEAR_VALUE_HEADS * 2
            || a_log.len() != QWEN80_LINEAR_VALUE_HEADS
            || dt_bias.len() != QWEN80_LINEAR_VALUE_HEADS
        {
            return Err(model_error(
                "Qwen80 BA control reference received an invalid exact source geometry",
            ));
        }
        let values_per_key_head = QWEN80_LINEAR_VALUE_HEADS / QWEN80_LINEAR_KEY_HEADS;
        if values_per_key_head != 2 {
            return Err(model_error(
                "Qwen80 BA control reference requires two value heads per key head",
            ));
        }
        let mut decay = vec![0.0f32; QWEN80_LINEAR_VALUE_HEADS];
        let mut beta = vec![0.0f32; QWEN80_LINEAR_VALUE_HEADS];
        for value_head in 0..QWEN80_LINEAR_VALUE_HEADS {
            let key_head = value_head / values_per_key_head;
            let within_key_head = value_head % values_per_key_head;
            let base = key_head * 2 * values_per_key_head;
            let b = ba[base + within_key_head];
            let a = ba[base + values_per_key_head + within_key_head];
            let x = a + dt_bias[value_head];
            let softplus = x.max(0.0) + (-x.abs()).exp().ln_1p();
            let g = -a_log[value_head].exp() * softplus;
            let decay_value = g.exp();
            let beta_value = 1.0 / (1.0 + (-b).exp());
            if !decay_value.is_finite()
                || !beta_value.is_finite()
                || decay_value <= 0.0
                || decay_value > 1.0
            {
                return Err(model_error(format!(
                    "Qwen80 BA control reference produced invalid recurrence controls at value head {value_head}"
                )));
            }
            decay[value_head] = decay_value;
            beta[value_head] = beta_value;
        }
        Ok((decay, beta))
    }

    fn source_topk_router(logits: &[f32]) -> Result<(Vec<u32>, Vec<f32>)> {
        let route = source_qwen80_topk_router(logits)?;
        Ok((
            route.ids.iter().map(|&id| u32::from(id)).collect(),
            route.weights.to_vec(),
        ))
    }

    fn direct_matrix_shape(
        tensor: &Qwen80GpuBinaryTensor,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if tensor.header.shape != [rows, cols] || tensor.header.group_size != QWEN80_GROUP_SIZE {
            return Err(model_error(format!(
                "direct tensor {name:?} has shape {:?}/group {}, expected [{rows}, {cols}]/{}",
                tensor.header.shape, tensor.header.group_size, QWEN80_GROUP_SIZE
            )));
        }
        Ok(())
    }

    /// Exercise a bounded, source-mapped Qwen3-Next linear-attention + MoE
    /// stage directly from the admitted compact binary artifact.
    ///
    /// The mixer fixture is explicitly the input *after* the source
    /// input-layer norm, and the MoE fixture is explicitly the input *after*
    /// post-attention norm.  Those surrounding norms and the intervening
    /// mixer/residual graph are not yet composed, so this method must never be
    /// promoted as a complete layer or token.  Within that boundary it does
    /// execute real source tensors on Metal: QKVZ/BA direct projections,
    /// direct-packed A_log/dt_bias control conversion, the exact first
    /// DeltaNet recurrent-state geometry, the 512-way router/top-10 policy,
    /// and one device-selected expert gate projection.  CPU evaluates only
    /// the same compact representation as a parity oracle.
    pub fn execute_first_linear_deltanet_router_expert_stage(
        &mut self,
    ) -> Result<Qwen80NativeLinearDeltaNetRouterExpertStage> {
        if !matches!(
            self.catalog.config.layer_kind(0),
            Ok(Qwen80LayerKind::LinearAttention)
        ) {
            return Err(model_error(
                "layer zero is not the expected first Qwen80 linear-attention layer",
            ));
        }
        let hidden = self.catalog.config.hidden;
        let key_heads = self.catalog.config.linear_key_heads;
        let value_heads = self.catalog.config.linear_value_heads;
        let key_dim = self.catalog.config.linear_key_head_dim;
        let value_dim = self.catalog.config.linear_value_head_dim;
        let values_per_key_head = value_heads
            .checked_div(key_heads)
            .filter(|value| *value > 0)
            .ok_or_else(|| model_error("Qwen80 value/key head ratio is invalid"))?;
        if hidden != QWEN80_HIDDEN
            || key_heads != QWEN80_LINEAR_KEY_HEADS
            || value_heads != QWEN80_LINEAR_VALUE_HEADS
            || key_dim != QWEN80_LINEAR_KEY_HEAD_DIM
            || value_dim != QWEN80_LINEAR_VALUE_HEAD_DIM
            || values_per_key_head != 2
        {
            return Err(model_error(
                "catalog geometry drifted from the exact Qwen3-Next first-linear-stage contract",
            ));
        }

        const QKVZ_NAME: &str = "model.layers.0.linear_attn.in_proj_qkvz.weight";
        const BA_NAME: &str = "model.layers.0.linear_attn.in_proj_ba.weight";
        const A_LOG_NAME: &str = "model.layers.0.linear_attn.A_log";
        const DT_BIAS_NAME: &str = "model.layers.0.linear_attn.dt_bias";
        const ROUTER_NAME: &str = "model.layers.0.mlp.gate.weight";

        // These bounded fixture rows are not embeddings or generated tokens.
        // They name the exact subgraph input boundary so host-side parity never
        // masquerades as a source model fallback.
        let mixer_fixture = Self::fixture_row(hidden, 17);
        let moe_fixture = Self::fixture_row(hidden, 101);
        let qkvz_rows = self
            .catalog
            .config
            .linear_key_dim()
            .checked_mul(2)
            .and_then(|value| value.checked_add(self.catalog.config.linear_value_dim() * 2))
            .ok_or_else(|| model_error("Qwen80 QKVZ row count overflows"))?;
        let qkvz_expected =
            self.direct_cpu_matvec_reference(QKVZ_NAME, &mixer_fixture, qkvz_rows, hidden)?;
        let ba_expected =
            self.direct_cpu_matvec_reference(BA_NAME, &mixer_fixture, value_heads * 2, hidden)?;
        let a_log_expected = self.direct_cpu_vector_reference(A_LOG_NAME, value_heads)?;
        let dt_bias_expected = self.direct_cpu_vector_reference(DT_BIAS_NAME, value_heads)?;
        let (decay_expected, beta_expected) =
            Self::source_ba_to_decay_beta(&ba_expected, &a_log_expected, &dt_bias_expected)?;
        let router_expected =
            self.direct_cpu_matvec_reference(ROUTER_NAME, &moe_fixture, QWEN80_EXPERTS, hidden)?;
        let (route_ids_expected, route_weights_expected) =
            Self::source_topk_router(&router_expected)?;

        let qkvz = self.upload_direct_tensor(QKVZ_NAME)?;
        let ba = self.upload_direct_tensor(BA_NAME)?;
        let a_log = self.upload_direct_tensor(A_LOG_NAME)?;
        let dt_bias = self.upload_direct_tensor(DT_BIAS_NAME)?;
        let router = self.upload_direct_tensor(ROUTER_NAME)?;
        Self::direct_matrix_shape(&qkvz, QKVZ_NAME, qkvz_rows, hidden)?;
        Self::direct_matrix_shape(&ba, BA_NAME, value_heads * 2, hidden)?;
        Self::direct_matrix_shape(&router, ROUTER_NAME, QWEN80_EXPERTS, hidden)?;
        if a_log.header.shape != [value_heads]
            || dt_bias.header.shape != [value_heads]
            || a_log.header.group_size != QWEN80_GROUP_SIZE
            || dt_bias.header.group_size != QWEN80_GROUP_SIZE
        {
            return Err(model_error(
                "Qwen80 DeltaNet A_log/dt_bias direct-packed control shapes drifted",
            ));
        }

        let mixer_input = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&mixer_fixture))?;
        let moe_input = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&moe_fixture))?;
        let qkvz_output = self.context.new_buffer_checked(bytes_for_f32(
            qkvz_rows,
            "Qwen80 direct QKVZ projection output",
        )?)?;
        let ba_output = self.context.new_buffer_checked(bytes_for_f32(
            value_heads * 2,
            "Qwen80 direct BA projection output",
        )?)?;
        let decay_output = self
            .context
            .new_buffer_checked(bytes_for_f32(value_heads, "Qwen80 DeltaNet decay output")?)?;
        let beta_output = self
            .context
            .new_buffer_checked(bytes_for_f32(value_heads, "Qwen80 DeltaNet beta output")?)?;
        let router_logits = self.context.new_buffer_checked(bytes_for_f32(
            QWEN80_EXPERTS,
            "Qwen80 direct router logits",
        )?)?;
        let route_ids = self.context.new_buffer_checked(
            QWEN80_TOP_K
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or_else(|| model_error("Qwen80 route id byte count overflows"))?,
        )?;
        let route_weights = self.context.new_buffer_checked(bytes_for_f32(
            QWEN80_TOP_K,
            "Qwen80 normalized top-k route weights",
        )?)?;

        let state_elements = checked_product(&[value_heads, key_dim, value_dim], "stage state")?;
        let initial_state = (0..state_elements)
            .map(|index| ((index * 13 % 101) as f32 - 50.0) * 0.0005)
            .collect::<Vec<_>>();
        let mut query = Vec::with_capacity(value_heads * key_dim);
        let mut key = Vec::with_capacity(value_heads * key_dim);
        let mut value = Vec::with_capacity(value_heads * value_dim);
        for head in 0..value_heads {
            query.extend(Self::component_l2_row(head, 1.0 / (key_dim as f32).sqrt()));
            key.extend(Self::component_l2_row(head + 47, 1.0));
            value.extend(
                (0..value_dim).map(|index| ((head * 17 + index * 11) % 127) as f32 / 127.0 - 0.5),
            );
        }
        let (expected_state, expected_deltanet_output) = Self::component_cpu_oracle(
            initial_state.clone(),
            &query,
            &key,
            &value,
            &decay_expected,
            &beta_expected,
        );
        self.reset_state();
        MetalContext::write_buffer_bytes(
            &self.linear_recurrent_state,
            bytemuck::cast_slice(&initial_state),
        );
        let query_buffer = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&query))?;
        let key_buffer = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&key))?;
        let value_buffer = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&value))?;
        let deltanet_output = self.context.new_buffer_checked(bytes_for_f32(
            value_heads * value_dim,
            "Qwen80 bounded DeltaNet stage output",
        )?)?;

        let mut first_tcb = TokenCommandBuffer::new(&self.context);
        // QKVZ/BA both consume a documented post-input-norm fixture.  They
        // are independent of router work, but keeping this ordered command
        // buffer makes the BA -> control -> recurrence dependency explicit.
        qwen_binary_sign_scale_matvec_component_tcb(
            &mut first_tcb,
            &qkvz.signs,
            &qkvz.scales,
            &mixer_input,
            &qkvz_output,
            qkvz_rows,
            hidden,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            &mut first_tcb,
            &ba.signs,
            &ba.scales,
            &mixer_input,
            &ba_output,
            value_heads * 2,
            hidden,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_ba_to_decay_beta_tcb(
            &mut first_tcb,
            &ba_output,
            &a_log.signs,
            &a_log.scales,
            &dt_bias.signs,
            &dt_bias.scales,
            &decay_output,
            &beta_output,
            key_heads,
            values_per_key_head,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_gated_delta_decode_single_tcb(
            &mut first_tcb,
            &self.linear_recurrent_state,
            &query_buffer,
            &key_buffer,
            &value_buffer,
            &decay_output,
            &beta_output,
            &deltanet_output,
            value_heads,
            key_dim,
            value_dim,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            &mut first_tcb,
            &router.signs,
            &router.scales,
            &moe_input,
            &router_logits,
            QWEN80_EXPERTS,
            hidden,
            QWEN80_GROUP_SIZE,
        )?;
        moe_topk_gate_tcb(
            &mut first_tcb,
            &router_logits,
            &route_ids,
            &route_weights,
            QWEN80_EXPERTS,
            QWEN80_TOP_K,
        )?;
        qwen_complete_normalize_route_weights_tcb(&mut first_tcb, &route_weights, QWEN80_TOP_K)?;
        let first_command_buffer_dispatches = first_tcb.dispatch_count();
        first_tcb.commit_and_wait()?;

        let qkvz_observed =
            Self::device_f32_snapshot(&qkvz_output, qkvz_rows, "Qwen80 QKVZ projection")?;
        let qkvz_projection_max_abs_error = Self::require_parity(
            &qkvz_expected,
            &qkvz_observed,
            "Qwen80 direct QKVZ projection",
            5.0e-4,
        )?;
        let ba_observed =
            Self::device_f32_snapshot(&ba_output, value_heads * 2, "Qwen80 BA projection")?;
        let ba_projection_max_abs_error = Self::require_parity(
            &ba_expected,
            &ba_observed,
            "Qwen80 direct BA projection",
            5.0e-4,
        )?;
        let decay_observed =
            Self::device_f32_snapshot(&decay_output, value_heads, "Qwen80 BA-derived decay")?;
        let beta_observed =
            Self::device_f32_snapshot(&beta_output, value_heads, "Qwen80 BA-derived beta")?;
        let ba_decay_max_abs_error = Self::require_parity(
            &decay_expected,
            &decay_observed,
            "Qwen80 direct BA-to-decay control",
            2.0e-5,
        )?;
        let ba_beta_max_abs_error = Self::require_parity(
            &beta_expected,
            &beta_observed,
            "Qwen80 direct BA-to-beta control",
            2.0e-5,
        )?;
        let observed_state = Self::device_f32_snapshot(
            &self.linear_recurrent_state,
            expected_state.len(),
            "Qwen80 first linear recurrent state",
        )?;
        let deltanet_state_max_abs_error = Self::require_parity(
            &expected_state,
            &observed_state,
            "Qwen80 BA-parameterized DeltaNet state",
            3.0e-5,
        )?;
        let observed_deltanet_output = Self::device_f32_snapshot(
            &deltanet_output,
            expected_deltanet_output.len(),
            "Qwen80 BA-parameterized DeltaNet output",
        )?;
        let deltanet_output_max_abs_error = Self::require_parity(
            &expected_deltanet_output,
            &observed_deltanet_output,
            "Qwen80 BA-parameterized DeltaNet output",
            3.0e-5,
        )?;
        let router_observed =
            Self::device_f32_snapshot(&router_logits, QWEN80_EXPERTS, "Qwen80 router logits")?;
        let router_logits_max_abs_error = Self::require_parity(
            &router_expected,
            &router_observed,
            "Qwen80 direct 512-way router logits",
            5.0e-4,
        )?;
        let route_ids_observed =
            Self::device_u32_snapshot(&route_ids, QWEN80_TOP_K, "Qwen80 route ids")?;
        if route_ids_observed != route_ids_expected {
            return Err(model_error(format!(
                "Qwen80 device router ids {:?} disagree with compact CPU parity ids {:?}",
                route_ids_observed, route_ids_expected
            )));
        }
        let route_weights_observed = Self::device_f32_snapshot(
            &route_weights,
            QWEN80_TOP_K,
            "Qwen80 normalized route weights",
        )?;
        Self::require_parity(
            &route_weights_expected,
            &route_weights_observed,
            "Qwen80 direct device normalized top-k weights",
            2.0e-5,
        )?;
        let selected_expert = *route_ids_observed
            .first()
            .ok_or_else(|| model_error("Qwen80 device router returned no top-1 expert"))?;
        if selected_expert as usize >= QWEN80_EXPERTS {
            return Err(model_error(format!(
                "Qwen80 device router selected out-of-range expert {selected_expert}"
            )));
        }
        let selected_expert_gate_tensor =
            format!("model.layers.0.mlp.experts.{selected_expert}.gate_proj.weight");
        let selected_expert_gate = self.upload_direct_tensor(&selected_expert_gate_tensor)?;
        Self::direct_matrix_shape(
            &selected_expert_gate,
            &selected_expert_gate_tensor,
            self.catalog.config.moe_intermediate,
            hidden,
        )?;
        let selected_expert_expected = self.direct_cpu_matvec_reference(
            &selected_expert_gate_tensor,
            &moe_fixture,
            self.catalog.config.moe_intermediate,
            hidden,
        )?;
        let selected_expert_output = self.context.new_buffer_checked(bytes_for_f32(
            self.catalog.config.moe_intermediate,
            "Qwen80 selected expert gate projection",
        )?)?;
        let mut selected_expert_tcb = TokenCommandBuffer::new(&self.context);
        qwen_binary_sign_scale_matvec_component_tcb(
            &mut selected_expert_tcb,
            &selected_expert_gate.signs,
            &selected_expert_gate.scales,
            &moe_input,
            &selected_expert_output,
            self.catalog.config.moe_intermediate,
            hidden,
            QWEN80_GROUP_SIZE,
        )?;
        let selected_expert_command_buffer_dispatches = selected_expert_tcb.dispatch_count();
        selected_expert_tcb.commit_and_wait()?;
        let selected_expert_observed = Self::device_f32_snapshot(
            &selected_expert_output,
            self.catalog.config.moe_intermediate,
            "Qwen80 selected expert gate projection",
        )?;
        let selected_expert_gate_projection_max_abs_error = Self::require_parity(
            &selected_expert_expected,
            &selected_expert_observed,
            "Qwen80 selected device-routed expert gate projection",
            5.0e-4,
        )?;

        Ok(Qwen80NativeLinearDeltaNetRouterExpertStage {
            layer: 0,
            layer_kind: Qwen80LayerKind::LinearAttention.as_source_name().to_owned(),
            direct_packed_input_projection_tensor: QKVZ_NAME.to_owned(),
            direct_packed_ba_projection_tensor: BA_NAME.to_owned(),
            direct_packed_router_tensor: ROUTER_NAME.to_owned(),
            direct_packed_selected_expert_gate_tensor: selected_expert_gate_tensor,
            qkvz_projection_max_abs_error,
            ba_projection_max_abs_error,
            ba_decay_max_abs_error,
            ba_beta_max_abs_error,
            deltanet_state_max_abs_error,
            deltanet_output_max_abs_error,
            router_logits_max_abs_error,
            route_ids: route_ids_observed,
            route_weights: route_weights_observed,
            selected_expert,
            selected_expert_gate_projection_max_abs_error,
            first_command_buffer_dispatches,
            selected_expert_command_buffer_dispatches,
            source_ba_layout: "in_proj_ba rows [16 key-heads][b0,b1,a0,a1]; values are flattened to 32 heads after source split".to_owned(),
            source_router_policy: "512-way softmax -> deterministic top-10 -> device-side norm_topk_prob=true normalization".to_owned(),
        })
    }

    /// Execute a bounded, source-shaped layer-0 Gated DeltaNet mixer through
    /// direct-packed Metal kernels and require CPU/Metal parity at every
    /// operator boundary. The input and initial states are deterministic
    /// fixtures, not embeddings or generated tokens; this API therefore
    /// cannot establish a complete layer, decoder, generation, HCLI, TPS, or
    /// tournament gate.
    pub fn execute_first_linear_deltanet_mixer_stage(
        &mut self,
    ) -> Result<Qwen80NativeFirstLinearDeltaNetMixerStage> {
        const DIRECT_PROJECTION_TOLERANCE: f32 = 1.0e-3;
        const SOURCE_OPERATOR_TOLERANCE: f32 = 1.0e-3;

        let contract = self
            .catalog
            .canonical_linear_deltanet_operator_contract(0)?;
        contract.validate_device_resources(&contract.minimum_device_resources)?;
        if contract.layer != 0
            || contract.linear_state_slot != 0
            || contract.minimum_device_resources.conv_state_offset_elements != 0
            || contract
                .minimum_device_resources
                .recurrent_state_offset_elements
                != 0
        {
            return Err(model_error(
                "first Qwen80 DeltaNet Metal stage did not bind source layer/state slot zero",
            ));
        }
        let layout = &contract.layout;
        let hidden = Self::fixture_row(layout.hidden_elements, 211);
        let initial_conv_state = (0..layout.conv_state_elements()?)
            .map(|index| ((index * 29 % 137) as f32 - 68.0) * 0.0002)
            .collect::<Vec<_>>();
        let initial_recurrent_state = (0..layout.recurrent_state_elements()?)
            .map(|index| ((index * 13 % 101) as f32 - 50.0) * 0.0005)
            .collect::<Vec<_>>();
        let cpu_input = Qwen80CanonicalLinearLayerCpuInput {
            hidden: hidden.clone(),
            state: Qwen80CanonicalLinearLayerCpuState {
                conv_state: initial_conv_state.clone(),
                recurrent_state: initial_recurrent_state.clone(),
            },
        };
        let expected = self
            .catalog
            .execute_canonical_linear_deltanet_cpu_oracle(&contract, &cpu_input)?;

        let input_layernorm = self.upload_direct_tensor(&contract.input_layernorm.name)?;
        let qkvz = self.upload_direct_tensor(&contract.mixer.in_proj_qkvz.name)?;
        let ba = self.upload_direct_tensor(&contract.mixer.in_proj_ba.name)?;
        let conv = self.upload_direct_tensor(&contract.mixer.causal_conv1d.name)?;
        let a_log = self.upload_direct_tensor(&contract.mixer.a_log.name)?;
        let dt_bias = self.upload_direct_tensor(&contract.mixer.dt_bias.name)?;
        let gated_norm = self.upload_direct_tensor(&contract.mixer.gated_rms_norm.name)?;
        let out_proj = self.upload_direct_tensor(&contract.mixer.out_proj.name)?;
        for binding in contract.required_bindings() {
            let expected_bytes = contract
                .minimum_device_resources
                .direct_packed_payload_bytes
                .get(&binding.name)
                .copied()
                .ok_or_else(|| {
                    model_error(format!(
                        "first Qwen80 DeltaNet Metal stage has no compact-byte requirement for {:?}",
                        binding.name
                    ))
                })?;
            if self
                .catalog
                .direct_tensor_header(&binding.name)?
                .payload_bytes
                != expected_bytes
            {
                return Err(model_error(format!(
                    "first Qwen80 DeltaNet Metal stage compact-byte requirement drifted for {:?}",
                    binding.name
                )));
            }
        }

        let input = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&hidden))?;
        let normalized = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            "Qwen80 first DeltaNet input RMSNorm output",
        )?)?;
        let qkvz_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.qkvz_projection_elements()?,
            "Qwen80 first DeltaNet QKVZ projection",
        )?)?;
        let ba_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.ba_projection_elements()?,
            "Qwen80 first DeltaNet BA projection",
        )?)?;
        let repeated_query = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "Qwen80 first DeltaNet repeated query",
        )?)?;
        let repeated_key = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "Qwen80 first DeltaNet repeated key",
        )?)?;
        let convolved_value = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "Qwen80 first DeltaNet convolved value",
        )?)?;
        let z = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "Qwen80 first DeltaNet Z gate",
        )?)?;
        let decay = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_heads,
            "Qwen80 first DeltaNet decay",
        )?)?;
        let beta = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_heads,
            "Qwen80 first DeltaNet beta",
        )?)?;
        let recurrent_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "Qwen80 first DeltaNet recurrent output",
        )?)?;
        let gated_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            "Qwen80 first DeltaNet gated RMSNorm output",
        )?)?;
        let mixer_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            "Qwen80 first DeltaNet mixer output",
        )?)?;
        let mixer_residual = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            "Qwen80 first DeltaNet mixer residual",
        )?)?;

        MetalContext::write_buffer_bytes(
            &self.linear_conv_state,
            bytemuck::cast_slice(&initial_conv_state),
        );
        MetalContext::write_buffer_bytes(
            &self.linear_recurrent_state,
            bytemuck::cast_slice(&initial_recurrent_state),
        );

        let mut tcb = TokenCommandBuffer::new(&self.context);
        qwen_next_direct_packed_input_rmsnorm_tcb(
            &mut tcb,
            &input,
            &input_layernorm.signs,
            &input_layernorm.scales,
            &normalized,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
            QWEN80_RMS_EPS,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            &mut tcb,
            &qkvz.signs,
            &qkvz.scales,
            &normalized,
            &qkvz_output,
            layout.qkvz_projection_elements()?,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            &mut tcb,
            &ba.signs,
            &ba.scales,
            &normalized,
            &ba_output,
            layout.ba_projection_elements()?,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_qkvz_rearrange_conv_l2_tcb(
            &mut tcb,
            &qkvz_output,
            &conv.signs,
            &conv.scales,
            &self.linear_conv_state,
            contract.minimum_device_resources.conv_state_offset_elements,
            &repeated_query,
            &repeated_key,
            &convolved_value,
            &z,
            layout.key_heads,
            layout.value_heads_per_key_head,
            layout.key_head_dim,
            layout.value_head_dim,
            layout.conv_kernel,
            QWEN80_GROUP_SIZE,
            QWEN80_RMS_EPS,
        )?;
        qwen_next_ba_to_decay_beta_tcb(
            &mut tcb,
            &ba_output,
            &a_log.signs,
            &a_log.scales,
            &dt_bias.signs,
            &dt_bias.scales,
            &decay,
            &beta,
            layout.key_heads,
            layout.value_heads_per_key_head,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_gated_delta_decode_single_tcb(
            &mut tcb,
            &self.linear_recurrent_state,
            &repeated_query,
            &repeated_key,
            &convolved_value,
            &decay,
            &beta,
            &recurrent_output,
            layout.value_heads,
            layout.key_head_dim,
            layout.value_head_dim,
        )?;
        qwen_next_deltanet_gated_rmsnorm_tcb(
            &mut tcb,
            &recurrent_output,
            &z,
            &gated_norm.signs,
            &gated_norm.scales,
            &gated_output,
            layout.value_heads,
            layout.value_head_dim,
            QWEN80_GROUP_SIZE,
            QWEN80_RMS_EPS,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            &mut tcb,
            &out_proj.signs,
            &out_proj.scales,
            &gated_output,
            &mixer_output,
            layout.hidden_elements,
            layout.value_elements()?,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_add_residual_tcb(
            &mut tcb,
            &input,
            &mixer_output,
            &mixer_residual,
            layout.hidden_elements,
        )?;
        let metal_dispatches = tcb.dispatch_count();
        tcb.commit_and_wait()?;

        let input_rms_norm_max_abs_error = Self::require_parity(
            &expected.input_rms_norm_output,
            &Self::device_f32_snapshot(
                &normalized,
                layout.hidden_elements,
                "Qwen80 first DeltaNet input RMSNorm",
            )?,
            "Qwen80 first DeltaNet direct input RMSNorm",
            SOURCE_OPERATOR_TOLERANCE,
        )?;
        let qkvz_projection_max_abs_error = Self::require_parity(
            &expected.projected_qkvz,
            &Self::device_f32_snapshot(
                &qkvz_output,
                layout.qkvz_projection_elements()?,
                "Qwen80 first DeltaNet QKVZ projection",
            )?,
            "Qwen80 first DeltaNet direct QKVZ projection",
            DIRECT_PROJECTION_TOLERANCE,
        )?;
        let ba_projection_max_abs_error = Self::require_parity(
            &expected.projected_ba,
            &Self::device_f32_snapshot(
                &ba_output,
                layout.ba_projection_elements()?,
                "Qwen80 first DeltaNet BA projection",
            )?,
            "Qwen80 first DeltaNet direct BA projection",
            DIRECT_PROJECTION_TOLERANCE,
        )?;
        let conv_state_max_abs_error = Self::require_parity(
            &expected.next_state.conv_state,
            &Self::device_f32_snapshot(
                &self.linear_conv_state,
                layout.conv_state_elements()?,
                "Qwen80 first DeltaNet convolution state",
            )?,
            "Qwen80 first DeltaNet direct convolution state",
            SOURCE_OPERATOR_TOLERANCE,
        )?;
        let repeated_query_max_abs_error = Self::require_parity(
            &expected.repeated_query_l2_scaled,
            &Self::device_f32_snapshot(
                &repeated_query,
                layout.value_elements()?,
                "Qwen80 first DeltaNet repeated query",
            )?,
            "Qwen80 first DeltaNet Q L2 layout",
            SOURCE_OPERATOR_TOLERANCE,
        )?;
        let repeated_key_max_abs_error = Self::require_parity(
            &expected.repeated_key_l2,
            &Self::device_f32_snapshot(
                &repeated_key,
                layout.value_elements()?,
                "Qwen80 first DeltaNet repeated key",
            )?,
            "Qwen80 first DeltaNet K L2 layout",
            SOURCE_OPERATOR_TOLERANCE,
        )?;
        let convolved_value_max_abs_error = Self::require_parity(
            &expected.convolved_value,
            &Self::device_f32_snapshot(
                &convolved_value,
                layout.value_elements()?,
                "Qwen80 first DeltaNet convolved value",
            )?,
            "Qwen80 first DeltaNet convolved value",
            SOURCE_OPERATOR_TOLERANCE,
        )?;
        let z_max_abs_error = Self::require_parity(
            &expected.z,
            &Self::device_f32_snapshot(&z, layout.value_elements()?, "Qwen80 first DeltaNet Z")?,
            "Qwen80 first DeltaNet Z layout",
            SOURCE_OPERATOR_TOLERANCE,
        )?;
        let decay_max_abs_error = Self::require_parity(
            &expected.decay,
            &Self::device_f32_snapshot(&decay, layout.value_heads, "Qwen80 first DeltaNet decay")?,
            "Qwen80 first DeltaNet BA decay",
            SOURCE_OPERATOR_TOLERANCE,
        )?;
        let beta_max_abs_error = Self::require_parity(
            &expected.beta,
            &Self::device_f32_snapshot(&beta, layout.value_heads, "Qwen80 first DeltaNet beta")?,
            "Qwen80 first DeltaNet BA beta",
            SOURCE_OPERATOR_TOLERANCE,
        )?;
        let recurrent_state_max_abs_error = Self::require_parity(
            &expected.next_state.recurrent_state,
            &Self::device_f32_snapshot(
                &self.linear_recurrent_state,
                layout.recurrent_state_elements()?,
                "Qwen80 first DeltaNet recurrent state",
            )?,
            "Qwen80 first DeltaNet recurrent state",
            SOURCE_OPERATOR_TOLERANCE,
        )?;
        let recurrent_output_max_abs_error = Self::require_parity(
            &expected.recurrent_output,
            &Self::device_f32_snapshot(
                &recurrent_output,
                layout.value_elements()?,
                "Qwen80 first DeltaNet recurrent output",
            )?,
            "Qwen80 first DeltaNet recurrent output",
            SOURCE_OPERATOR_TOLERANCE,
        )?;
        let gated_rms_norm_max_abs_error = Self::require_parity(
            &expected.gated_rms_norm_output,
            &Self::device_f32_snapshot(
                &gated_output,
                layout.value_elements()?,
                "Qwen80 first DeltaNet gated RMSNorm output",
            )?,
            "Qwen80 first DeltaNet gated RMSNorm",
            SOURCE_OPERATOR_TOLERANCE,
        )?;
        let mixer_output_max_abs_error = Self::require_parity(
            &expected.mixer_output,
            &Self::device_f32_snapshot(
                &mixer_output,
                layout.hidden_elements,
                "Qwen80 first DeltaNet mixer output",
            )?,
            "Qwen80 first DeltaNet direct out projection",
            DIRECT_PROJECTION_TOLERANCE,
        )?;
        let mixer_residual_max_abs_error = Self::require_parity(
            &expected.mixer_residual_output,
            &Self::device_f32_snapshot(
                &mixer_residual,
                layout.hidden_elements,
                "Qwen80 first DeltaNet mixer residual",
            )?,
            "Qwen80 first DeltaNet first residual",
            DIRECT_PROJECTION_TOLERANCE,
        )?;

        Ok(Qwen80NativeFirstLinearDeltaNetMixerStage {
            layer: contract.layer,
            linear_state_slot: contract.linear_state_slot,
            direct_packed_input_layernorm_tensor: contract.input_layernorm.name,
            direct_packed_qkvz_tensor: contract.mixer.in_proj_qkvz.name,
            direct_packed_ba_tensor: contract.mixer.in_proj_ba.name,
            direct_packed_conv_tensor: contract.mixer.causal_conv1d.name,
            direct_packed_a_log_tensor: contract.mixer.a_log.name,
            direct_packed_dt_bias_tensor: contract.mixer.dt_bias.name,
            direct_packed_gated_norm_tensor: contract.mixer.gated_rms_norm.name,
            direct_packed_out_proj_tensor: contract.mixer.out_proj.name,
            input_rms_norm_max_abs_error,
            qkvz_projection_max_abs_error,
            ba_projection_max_abs_error,
            conv_state_max_abs_error,
            repeated_query_max_abs_error,
            repeated_key_max_abs_error,
            convolved_value_max_abs_error,
            z_max_abs_error,
            decay_max_abs_error,
            beta_max_abs_error,
            recurrent_state_max_abs_error,
            recurrent_output_max_abs_error,
            gated_rms_norm_max_abs_error,
            mixer_output_max_abs_error,
            mixer_residual_max_abs_error,
            metal_dispatches,
            source_algorithm_boundary: "direct-packed source layer-0 input RMSNorm -> QKVZ/BA projections -> [key-head][Q,K,V,Z] rearrange -> causal depthwise SiLU convolution -> repeated Q/K L2 -> BA/A_log/dt_bias recurrence controls -> DeltaNet recurrence -> Z-gated RMSNorm -> out projection -> first residual; post-attention RMSNorm and routed/shared MoE intentionally absent".to_owned(),
        })
    }

    fn component_l2_row(seed: usize, scale: f32) -> Vec<f32> {
        let mut values = (0..QWEN80_LINEAR_KEY_HEAD_DIM)
            .map(|index| (((seed * 97 + index * 31) % 4093) as f32 / 2048.0) - 1.0)
            .collect::<Vec<_>>();
        let norm = values
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1.0e-6);
        for value in &mut values {
            *value = *value / norm * scale;
        }
        values
    }

    fn component_cpu_oracle(
        mut state: Vec<f32>,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        decay: &[f32],
        beta: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut output = vec![0.0f32; QWEN80_LINEAR_VALUE_HEADS * QWEN80_LINEAR_VALUE_HEAD_DIM];
        for head in 0..QWEN80_LINEAR_VALUE_HEADS {
            let state_base = head * QWEN80_LINEAR_KEY_HEAD_DIM * QWEN80_LINEAR_VALUE_HEAD_DIM;
            let key_base = head * QWEN80_LINEAR_KEY_HEAD_DIM;
            let value_base = head * QWEN80_LINEAR_VALUE_HEAD_DIM;
            for value_index in 0..QWEN80_LINEAR_VALUE_HEAD_DIM {
                let mut kv_memory = 0.0f32;
                for key_index in 0..QWEN80_LINEAR_KEY_HEAD_DIM {
                    let index = state_base + key_index * QWEN80_LINEAR_VALUE_HEAD_DIM + value_index;
                    state[index] *= decay[head];
                    kv_memory += state[index] * key[key_base + key_index];
                }
                let delta = (value[value_base + value_index] - kv_memory) * beta[head];
                for key_index in 0..QWEN80_LINEAR_KEY_HEAD_DIM {
                    let index = state_base + key_index * QWEN80_LINEAR_VALUE_HEAD_DIM + value_index;
                    state[index] += key[key_base + key_index] * delta;
                }
            }
            for value_index in 0..QWEN80_LINEAR_VALUE_HEAD_DIM {
                let mut sum = 0.0f32;
                for key_index in 0..QWEN80_LINEAR_KEY_HEAD_DIM {
                    sum += state
                        [state_base + key_index * QWEN80_LINEAR_VALUE_HEAD_DIM + value_index]
                        * query[key_base + key_index];
                }
                output[value_base + value_index] = sum;
            }
        }
        (state, output)
    }

    /// Run one strict native Metal recurrence on the first real linear-layer
    /// state slice and compare it with a CPU reference.  The CPU computation
    /// is only an operator oracle; it is not a model fallback and its result
    /// never reaches an output token.
    pub fn verify_first_linear_deltanet_component(
        &mut self,
    ) -> Result<Qwen80NativeDeltaNetComponentStep> {
        let heads = self.catalog.config.linear_value_heads;
        let key_dim = self.catalog.config.linear_key_head_dim;
        let value_dim = self.catalog.config.linear_value_head_dim;
        if heads != QWEN80_LINEAR_VALUE_HEADS
            || key_dim != QWEN80_LINEAR_KEY_HEAD_DIM
            || value_dim != QWEN80_LINEAR_VALUE_HEAD_DIM
        {
            return Err(model_error(
                "catalog geometry does not match the exact Qwen80 DeltaNet Metal component",
            ));
        }
        let state_elements = checked_product(&[heads, key_dim, value_dim], "component state")?;
        let state = (0..state_elements)
            .map(|index| ((index * 13 % 101) as f32 - 50.0) * 0.0005)
            .collect::<Vec<_>>();
        let mut query = Vec::with_capacity(heads * key_dim);
        let mut key = Vec::with_capacity(heads * key_dim);
        let mut value = Vec::with_capacity(heads * value_dim);
        let mut decay = Vec::with_capacity(heads);
        let mut beta = Vec::with_capacity(heads);
        for head in 0..heads {
            query.extend(Self::component_l2_row(head, 1.0 / (key_dim as f32).sqrt()));
            key.extend(Self::component_l2_row(head + 47, 1.0));
            value.extend(
                (0..value_dim).map(|index| ((head * 17 + index * 11) % 127) as f32 / 127.0 - 0.5),
            );
            decay.push(0.65 + (head % 7) as f32 * 0.03);
            beta.push(0.20 + (head % 5) as f32 * 0.07);
        }
        let (expected_state, expected_output) =
            Self::component_cpu_oracle(state.clone(), &query, &key, &value, &decay, &beta);
        MetalContext::write_buffer_bytes(
            &self.linear_recurrent_state,
            bytemuck::cast_slice(&state),
        );
        let query_buffer = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&query))?;
        let key_buffer = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&key))?;
        let value_buffer = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&value))?;
        let decay_buffer = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&decay))?;
        let beta_buffer = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&beta))?;
        let output_buffer = self.context.new_buffer_checked(bytes_for_f32(
            heads * value_dim,
            "Qwen80 DeltaNet component output",
        )?)?;
        let mut tcb = TokenCommandBuffer::new(&self.context);
        qwen_next_gated_delta_decode_single_tcb(
            &mut tcb,
            &self.linear_recurrent_state,
            &query_buffer,
            &key_buffer,
            &value_buffer,
            &decay_buffer,
            &beta_buffer,
            &output_buffer,
            heads,
            key_dim,
            value_dim,
        )?;
        let metal_dispatches = tcb.dispatch_count();
        tcb.commit_and_wait()?;
        let observed_state = unsafe {
            std::slice::from_raw_parts(
                self.linear_recurrent_state.contents() as *const f32,
                expected_state.len(),
            )
        };
        let observed_output = unsafe {
            std::slice::from_raw_parts(
                output_buffer.contents() as *const f32,
                expected_output.len(),
            )
        };
        let max_abs_state_error = expected_state
            .iter()
            .zip(observed_state)
            .map(|(expected, observed)| (expected - observed).abs())
            .fold(0.0f32, f32::max);
        let max_abs_output_error = expected_output
            .iter()
            .zip(observed_output)
            .map(|(expected, observed)| (expected - observed).abs())
            .fold(0.0f32, f32::max);
        if max_abs_state_error > 2.0e-5 || max_abs_output_error > 2.0e-5 {
            return Err(model_error(format!(
                "native Qwen80 DeltaNet component parity failed: state={max_abs_state_error}, output={max_abs_output_error}"
            )));
        }
        Ok(Qwen80NativeDeltaNetComponentStep {
            max_abs_state_error,
            max_abs_output_error,
            metal_dispatches,
        })
    }
}

#[cfg(not(target_os = "macos"))]
pub struct Qwen80CompleteNativeRuntime;

#[cfg(not(target_os = "macos"))]
impl Qwen80CompleteNativeRuntime {
    pub fn load(
        _manifest_path: impl AsRef<Path>,
        _admission: &CompleteBinaryAdmission,
        _options: Qwen80CompleteRuntimeOptions,
    ) -> Result<Self> {
        Err(Error::Metal(
            "Qwen80 complete native runtime is Metal-only and requires macOS".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::qwen_complete_binary::{
        CompleteBinaryTensor, COMPLETE_BINARY_MAGIC, COMPLETE_BINARY_VERSION,
    };
    use super::*;
    use half::f16;
    use serde_json::json;
    use std::sync::Arc;

    fn source_config() -> Value {
        json!({
            "architectures": ["Qwen3NextForCausalLM"],
            "model_type": "qwen3_next",
            "num_hidden_layers": 48,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "vocab_size": 151936,
            "decoder_sparse_step": 1,
            "full_attention_interval": 4,
            "intermediate_size": 5120,
            "hidden_act": "silu",
            "norm_topk_prob": true,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "attention_dropout": 0,
            "use_sliding_window": false,
            "use_cache": true,
            "rope_scaling": null,
            "mlp_only_layers": [],
            "layer_types": null,
            "rope_theta": 5000000.0,
            "partial_rotary_factor": 0.25,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144,
        })
    }

    /// Build all exact Qwen80 catalog rows without placing any payload file on
    /// disk.  A successful `from_admitted` call therefore proves that a
    /// previously completed strict admission can be handed forward without a
    /// second 74,391-payload scan; the public `load` entrypoint remains the
    /// only path that performs that full scan.
    fn already_admitted_catalog_fixture(temp: &tempfile::TempDir) -> CompleteBinaryArtifact {
        let source_dir = temp.path().join("source");
        fs::create_dir_all(&source_dir).unwrap();
        fs::write(
            source_dir.join("config.json"),
            serde_json::to_vec(&source_config()).unwrap(),
        )
        .unwrap();
        let source_index_path = source_dir.join("model.safetensors.index.json");
        fs::write(&source_index_path, b"{}\n").unwrap();

        let mut source_weight_elements = 0u64;
        let tensors = tensor_shapes()
            .into_iter()
            .map(|(tensor_name, shape)| {
                let elements = shape
                    .iter()
                    .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
                    .expect("exact Qwen80 tensor shape product must fit usize");
                source_weight_elements += u64::try_from(elements).unwrap();
                let groups = (elements + QWEN80_GROUP_SIZE - 1) / QWEN80_GROUP_SIZE;
                let artifact_path = source_dir
                    .join("unread-payloads")
                    .join(format!("{tensor_name}.hq30g"));
                let tensor = CompleteBinaryTensor {
                    tensor_name: tensor_name.clone(),
                    source_shard: "unit.safetensors".into(),
                    source_shard_sha256: "0".repeat(64),
                    source_dtype: "BF16".into(),
                    artifact_path,
                    artifact_sha256: "0".repeat(64),
                    header: CompleteBinaryHeader {
                        version: 1,
                        group_size: QWEN80_GROUP_SIZE,
                        shape,
                        elements,
                        groups,
                        scale_offset: 0,
                        sign_offset: 0,
                        payload_bytes: 0,
                    },
                };
                (tensor_name, tensor)
            })
            .collect::<BTreeMap<_, _>>();
        CompleteBinaryArtifact {
            model: QwenCompleteBinaryModel::Qwen80CoderNext,
            manifest_path: temp.path().join("admitted-manifest.json"),
            manifest_seal_sha256: "a".repeat(64),
            source_audit_path: temp.path().join("source-audit.json"),
            source_audit_seal_sha256: "b".repeat(64),
            source_revision: "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb".into(),
            source_index_path,
            source_weight_elements,
            tensor_payload_bytes: 0,
            tensors,
            verified_payloads: BTreeMap::new(),
        }
    }

    /// Build a valid fixed-group direct payload for a source-shaped tensor.
    /// The bytes deliberately stay in the in-process admission cache; no
    /// fixture payload is written to disk, so the CPU stage test exercises
    /// the same immutable-payload authority a native runtime must consume.
    fn direct_packed_fixture_payload(shape: &[usize], seed: u64) -> Arc<[u8]> {
        let elements = shape
            .iter()
            .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
            .expect("source-shaped fixture dimensions must fit usize");
        let groups = (elements + QWEN80_GROUP_SIZE - 1) / QWEN80_GROUP_SIZE;
        let sign_bytes = groups * (QWEN80_GROUP_SIZE / 8);
        let mut payload = Vec::with_capacity(32 + shape.len() * 4 + groups * 2 + sign_bytes);
        payload.extend_from_slice(&COMPLETE_BINARY_MAGIC);
        payload.extend_from_slice(&COMPLETE_BINARY_VERSION.to_le_bytes());
        payload.extend_from_slice(&(QWEN80_GROUP_SIZE as u32).to_le_bytes());
        payload.extend_from_slice(&(shape.len() as u16).to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&(elements as u64).to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        for dimension in shape {
            payload.extend_from_slice(&(*dimension as u32).to_le_bytes());
        }
        for group in 0..groups {
            let scale = 0.0015 + ((group as u64 + seed * 3) % 7) as f32 * 0.0002;
            payload.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
        }
        for byte_index in 0..sign_bytes {
            let mixed = (byte_index as u64)
                .wrapping_mul(0x9e37_79b9)
                .wrapping_add(seed.wrapping_mul(0x85eb_ca6b));
            payload.push((mixed ^ (mixed >> 13) ^ (mixed >> 29)) as u8);
        }
        let parsed = parse_complete_binary_header(&payload)
            .expect("source-shaped compact fixture must satisfy the fixed binary format");
        assert_eq!(parsed.shape, shape);
        Arc::from(payload)
    }

    fn install_verified_direct_fixture_payload(
        artifact: &mut CompleteBinaryArtifact,
        tensor_name: &str,
        seed: u64,
    ) {
        let shape = artifact
            .tensors
            .get(tensor_name)
            .expect("canonical fixture tensor must exist")
            .header
            .shape
            .clone();
        let payload = direct_packed_fixture_payload(&shape, seed);
        let payload_sha256 = format!("{:x}", Sha256::digest(&payload));
        let header = parse_complete_binary_header(&payload)
            .expect("fixture payload must parse before it enters admission cache");
        let tensor = artifact
            .tensors
            .get_mut(tensor_name)
            .expect("canonical fixture tensor must still exist");
        tensor.header = header;
        tensor.artifact_sha256 = payload_sha256;
        artifact
            .verified_payloads
            .insert(tensor_name.to_owned(), payload);
    }

    fn canonical_linear_cpu_catalog_fixture(
        temp: &tempfile::TempDir,
    ) -> Qwen80CompleteArtifactCatalog {
        let mut artifact = already_admitted_catalog_fixture(temp);
        for (seed, tensor_name) in [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.linear_attn.in_proj_qkvz.weight",
            "model.layers.0.linear_attn.in_proj_ba.weight",
            "model.layers.0.linear_attn.conv1d.weight",
            "model.layers.0.linear_attn.A_log",
            "model.layers.0.linear_attn.dt_bias",
            "model.layers.0.linear_attn.norm.weight",
            "model.layers.0.linear_attn.out_proj.weight",
        ]
        .iter()
        .enumerate()
        {
            install_verified_direct_fixture_payload(&mut artifact, tensor_name, seed as u64 + 1);
        }
        Qwen80CompleteArtifactCatalog::from_admitted(artifact)
            .expect("source-shaped in-process direct payload cache must bind to the exact catalog")
    }

    fn canonical_layer_one_linear_cpu_catalog_fixture(
        temp: &tempfile::TempDir,
    ) -> Qwen80CompleteArtifactCatalog {
        let mut artifact = already_admitted_catalog_fixture(temp);
        for (seed, tensor_name) in [
            "model.layers.1.input_layernorm.weight",
            "model.layers.1.linear_attn.in_proj_qkvz.weight",
            "model.layers.1.linear_attn.in_proj_ba.weight",
            "model.layers.1.linear_attn.conv1d.weight",
            "model.layers.1.linear_attn.A_log",
            "model.layers.1.linear_attn.dt_bias",
            "model.layers.1.linear_attn.norm.weight",
            "model.layers.1.linear_attn.out_proj.weight",
        ]
        .iter()
        .enumerate()
        {
            install_verified_direct_fixture_payload(&mut artifact, tensor_name, seed as u64 + 101);
        }
        Qwen80CompleteArtifactCatalog::from_admitted(artifact)
            .expect("source-shaped layer-one direct payload cache must bind to the exact catalog")
    }

    fn canonical_linear_cpu_input_fixture() -> Qwen80CanonicalLinearLayerCpuInput {
        let mut state = Qwen80CanonicalLinearLayerCpuState::zeroed();
        for (index, value) in state.conv_state.iter_mut().enumerate() {
            *value = (index % 7) as f32 * 0.0003 - 0.0009;
        }
        for (index, value) in state.recurrent_state.iter_mut().enumerate() {
            *value = (index % 11) as f32 * 0.00002 - 0.0001;
        }
        Qwen80CanonicalLinearLayerCpuInput {
            hidden: (0..QWEN80_HIDDEN)
                .map(|index| (index % 53) as f32 * 0.002 - 0.052)
                .collect(),
            state,
        }
    }

    /// Build a layer-0 source-shaped fixture that holds the fixed DeltaNet,
    /// post-norm/router/shared bodies and only the ten expert bodies selected
    /// by that fixture's *actual* compact router result.  This exercises the
    /// same dynamic expert binding rule as the production contract without
    /// constructing a 512-expert in-memory test body.
    fn canonical_linear_moe_cpu_catalog_fixture(
        temp: &tempfile::TempDir,
        input: &Qwen80CanonicalLinearLayerCpuInput,
    ) -> Qwen80CompleteArtifactCatalog {
        let mut artifact = already_admitted_catalog_fixture(temp);
        for (seed, tensor_name) in [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.linear_attn.in_proj_qkvz.weight",
            "model.layers.0.linear_attn.in_proj_ba.weight",
            "model.layers.0.linear_attn.conv1d.weight",
            "model.layers.0.linear_attn.A_log",
            "model.layers.0.linear_attn.dt_bias",
            "model.layers.0.linear_attn.norm.weight",
            "model.layers.0.linear_attn.out_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.layers.0.mlp.shared_expert_gate.weight",
        ]
        .iter()
        .enumerate()
        {
            install_verified_direct_fixture_payload(&mut artifact, tensor_name, seed as u64 + 1);
        }

        let routing_catalog = Qwen80CompleteArtifactCatalog::from_admitted(artifact.clone())
            .expect("fixed MoE bodies must admit before dynamic expert routing");
        let contract = routing_catalog
            .canonical_linear_moe_operator_contract(0)
            .expect("fixed source MoE contract must bind");
        let mixer = routing_catalog
            .execute_canonical_linear_deltanet_cpu_oracle(&contract.mixer, input)
            .expect("fixture DeltaNet mixer must execute before source route selection");
        let post_norm = Qwen80CpuPackedTensor::from_binding(
            &routing_catalog,
            &contract.post_attention_layernorm,
        )
        .expect("fixture post-attention norm must be direct-packed");
        let post_weight = post_norm
            .vector(QWEN80_HIDDEN, Qwen80CpuPackedReadMode::StreamingDirect)
            .expect("fixture post-attention norm vector must decode directly");
        let post_hidden =
            source_qwen80_residual_rms_norm(&mixer.mixer_residual_output, &post_weight)
                .expect("fixture post-attention norm must execute");
        let router = Qwen80CpuPackedTensor::from_binding(&routing_catalog, &contract.router)
            .expect("fixture router must be direct-packed");
        let logits = router
            .matvec(
                &post_hidden,
                QWEN80_EXPERTS,
                QWEN80_HIDDEN,
                Qwen80CpuPackedReadMode::StreamingDirect,
            )
            .expect("fixture direct router must execute");
        let route = source_qwen80_topk_router(&logits)
            .expect("fixture source router must emit a normalized top-10 route");
        for (route_index, expert) in route.ids.iter().enumerate() {
            let prefix = format!("model.layers.0.mlp.experts.{expert}");
            for (projection_index, suffix) in
                ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                    .iter()
                    .enumerate()
            {
                install_verified_direct_fixture_payload(
                    &mut artifact,
                    &format!("{prefix}.{suffix}"),
                    100 + route_index as u64 * 3 + projection_index as u64,
                );
            }
        }
        Qwen80CompleteArtifactCatalog::from_admitted(artifact)
            .expect("source-routed expert payload cache must bind to the exact catalog")
    }

    fn all_ten_route_descriptor_fixture(
        catalog: &Qwen80CompleteArtifactCatalog,
        plan: &Qwen80CompleteHybridDecoderPlan,
        route: &Qwen80RouteSelection,
    ) -> Value {
        let projection = |expert: u16, suffix: &str, shape: &[usize]| {
            let tensor_name = format!("model.layers.0.mlp.experts.{expert}.{suffix}.weight");
            json!({
                "tensor_name": tensor_name,
                "shape": shape,
                "elements": QWEN80_HIDDEN * QWEN80_MOE_INTERMEDIATE,
                "artifact_path": format!("/fixture/{expert}-{suffix}.hq30g"),
                "artifact_bytes": catalog
                    .verified_direct_tensor_payload(&tensor_name)
                    .expect("fixture route payload must exist")
                    .len(),
                "artifact_sha256": catalog
                    .direct_tensor_artifact_sha256(&tensor_name)
                    .expect("fixture route artifact digest must exist"),
                "source_dtype": "BF16",
                "source_shard": "unit.safetensors",
                "source_shard_sha256": "0".repeat(64),
                "layout": {
                    "magic": "HQ30G1B1",
                    "group_size": QWEN80_GROUP_SIZE,
                    "scale_dtype": "float16",
                    "sign_bit_order": "little",
                    "version": 1,
                },
                "payload_opened_by_this_plan": false,
            })
        };
        let waves = route
            .ids
            .iter()
            .zip(route.weights)
            .enumerate()
            .map(|(wave_index, (&expert, normalized_weight))| {
                json!({
                    "wave_index": wave_index,
                    "layer": 0,
                    "expert_id": usize::from(expert),
                    "normalized_weight": normalized_weight,
                    "normalized_weight_bits_hex": format!("0x{:016x}", (normalized_weight as f64).to_bits()),
                    "gate": projection(expert, "gate_proj", &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN]),
                    "up": projection(expert, "up_proj", &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN]),
                    "down": projection(expert, "down_proj", &[QWEN80_HIDDEN, QWEN80_MOE_INTERMEDIATE]),
                    "fixed_operation_order": [
                        "gate_proj [512,2048]",
                        "up_proj [512,2048]",
                        "SiLU(gate) * up [512]",
                        "down_proj [2048,512]",
                        "apply this route's source-normalized weight [2048]",
                    ],
                    "route_execution_status": "NOT_EXECUTED_SOURCE_BOUND_PLAN_ONLY",
                    "route_delta_materialized": false,
                    "route_weight_applied": false,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "schema": QWEN80_ALL_TEN_ROUTE_PLAN_SCHEMA,
            "status": QWEN80_ALL_TEN_ROUTE_PLAN_STATUS,
            "model_id": QWEN80_MODEL_ID,
            "model_key": "qwen80",
            "source_repository": QWEN80_REPOSITORY,
            "source_revision": plan.source_revision,
            "layer": 0,
            "router_evidence": {
                "outer_receipt_document_sha256": "c".repeat(64),
                "outer_receipt_seal_sha256": "d".repeat(64),
                "inner_receipt_document_sha256": "e".repeat(64),
                "source_stable_route_ids": route.ids.to_vec(),
                "source_stable_normalized_weights": route.weights.to_vec(),
                "source_router_component_only": true,
            },
            "manifest_descriptor_inventory": {
                "inventory_document_sha256": "b".repeat(64),
                "manifest_schema": "hawking.ascension.qwen80_complete_binary_gravity.v1",
                "manifest_status": "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED",
                "manifest_seal_sha256": plan.manifest_seal_sha256,
                "source_repository": QWEN80_REPOSITORY,
                "declared_tensor_count": QWEN80_COMPLETE_BINARY_TENSORS,
                "received_descriptor_count": QWEN80_COMPLETE_BINARY_TENSORS,
                "resolved_route_tensor_count": QWEN80_TOP_K * 3,
                "payload_opened_by_this_plan": false,
            },
            "deterministic_waves": waves,
            "rawls_real_all_ten_provenance_gate": {
                "schema": QWEN80_ALL_TEN_ROUTE_PLAN_GATE_SCHEMA,
                "all_ten_source_bindings_complete": true,
                "expected_layer": 0,
                "deterministic_wave_indices": (0..QWEN80_TOP_K).collect::<Vec<_>>(),
                "route_order": route.ids.to_vec(),
                "normalized_weights": route.weights.to_vec(),
                "execution_receipt_required_for_each_wave": true,
                "direct_packed_execution_required_for_each_wave": true,
                "source_bound_input_required_for_each_wave": true,
                "route_combine_receipt_required_separately": true,
                "shared_expert_receipt_required_separately": true,
                "first_and_second_residual_receipts_required_separately": true,
                "rejects_tensor_substitution": true,
                "rejects_route_reorder": true,
                "rejects_duplicate_experts": true,
                "rejects_missing_tensor_or_weight": true,
            },
            "route_execution_performed": false,
            "route_combine_performed": false,
            "shared_expert_performed": false,
            "residual_combine_performed": false,
            "metal_device_or_dispatch_performed": false,
            "model_execution_performed": false,
            "hcli_execution_performed": false,
            "tps_or_tg_measurement_performed": false,
            "complete_layer_or_decoder_claim_earned": false,
        })
    }

    #[test]
    fn exact_qwen80_source_config_and_hybrid_schedule_are_required() {
        let config = Qwen80CompleteRuntimeConfig::from_source_config(
            &source_config(),
            QWEN80_REPOSITORY,
            "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
        )
        .expect("exact source config must pass");
        assert_eq!(
            config.layer_kind(0).unwrap(),
            Qwen80LayerKind::LinearAttention
        );
        assert_eq!(
            config.layer_kind(2).unwrap(),
            Qwen80LayerKind::LinearAttention
        );
        assert_eq!(
            config.layer_kind(3).unwrap(),
            Qwen80LayerKind::FullAttention
        );
        assert_eq!(
            config.layer_kind(47).unwrap(),
            Qwen80LayerKind::FullAttention
        );
        assert!(config.layer_kind(48).is_err());

        let mut wrong = source_config();
        wrong["full_attention_interval"] = json!(2);
        assert!(Qwen80CompleteRuntimeConfig::from_source_config(
            &wrong,
            QWEN80_REPOSITORY,
            "revision"
        )
        .is_err());
    }

    #[test]
    fn gated_deltanet_rmsnorm_uses_the_source_per_value_head_reduction() {
        // The two value heads deliberately have a 10x amplitude difference.
        // A flattened 4-wide reduction would preserve that difference after
        // normalization; Qwen3Next reshapes to [-1, head_v_dim], so each
        // two-wide head must instead normalize to the same magnitude.
        let input = [1.0f32, 1.0, 10.0, 10.0];
        let gate = [1.0f32; 4];
        let weight = [1.0f32; 4];
        let output = source_qwen80_gated_rms_norm(&input, &gate, &weight, 2, 2)
            .expect("two source-shaped value heads must normalize");
        let silu_one = 1.0f32 / (1.0 + (-1.0f32).exp());
        assert!((output[0] - silu_one).abs() < 2.0e-6);
        assert!((output[2] - silu_one).abs() < 2.0e-6);
        assert!(
            (output[0] - output[2]).abs() < 2.0e-6,
            "per-head source RMSNorm must not use a flattened DeltaNet variance"
        );

        let flattened_inverse = ((1.0 + 1.0 + 100.0 + 100.0) / 4.0 + QWEN80_RMS_EPS)
            .sqrt()
            .recip();
        let flattened_head_one = 10.0 * flattened_inverse * silu_one;
        assert!(
            (output[2] - flattened_head_one).abs() > 0.2,
            "regression fixture must distinguish the incorrect flattened reduction"
        );
    }

    #[test]
    fn compact_deltanet_conv_state_matches_independent_four_tap_source_window() {
        // Hugging Face's cache stores a four-token convolution window, while
        // the native single-token contract retains the three *prior* tokens
        // and appends the current projection at execution.  Prove the latter
        // is source-equivalent across several transitions with a reference
        // formulated from full causal history, rather than reusing the
        // compact-state shift logic under test.
        let layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
        let payload =
            direct_packed_fixture_payload(&[layout.conv_channels, 1, layout.conv_kernel], 73);
        let conv = Qwen80CpuPackedTensor {
            name: "fixture.linear_attn.conv1d.weight".to_owned(),
            header: parse_complete_binary_header(&payload)
                .expect("source-shaped direct convolution fixture must parse"),
            payload,
        };
        let (_, materialized_weights) = decode_complete_binary_f32(&conv.payload)
            .expect("independent full-window convolution reference must decode its fixture");
        let mut compact_prior_state = vec![0.0f32; layout.conv_state_elements().unwrap()];
        let mut full_history = Vec::<Vec<f32>>::new();

        for step in 0..5usize {
            let current = (0..layout.conv_channels)
                .map(|channel| {
                    let seed = (channel * 17 + step * 31 + 11) % 101;
                    seed as f32 * 0.0025 - 0.125
                })
                .collect::<Vec<_>>();
            let (observed, next_state) = source_qwen80_causal_conv_step(
                &current,
                &compact_prior_state,
                &conv,
                Qwen80CpuPackedReadMode::StreamingDirect,
                &layout,
            )
            .expect("compact direct convolution must execute");

            // This intentionally reconstructs every source causal window
            // from token history. It neither reads nor shifts the compact
            // state representation, so it catches a tap/order regression.
            let mut expected = vec![0.0f32; layout.conv_channels];
            for channel in 0..layout.conv_channels {
                let mut convolution = 0.0f32;
                for tap in 0..layout.conv_kernel {
                    let source_position = step as isize - (layout.conv_kernel - 1 - tap) as isize;
                    let source_value = if source_position == step as isize {
                        current[channel]
                    } else if source_position >= 0 {
                        full_history[source_position as usize][channel]
                    } else {
                        0.0
                    };
                    convolution +=
                        materialized_weights[channel * layout.conv_kernel + tap] * source_value;
                }
                expected[channel] = convolution / (1.0 + (-convolution).exp());
            }
            assert!(
                max_abs_error_cpu(&expected, &observed, "independent source causal convolution")
                    .unwrap()
                    <= 1.0e-6,
                "compact prior-token state must preserve source 4-tap causal ordering at step {step}"
            );

            // After token `step`, the compact state contains exactly the
            // three preceding values for the next token, in source tap order.
            for channel in 0..layout.conv_channels {
                let state_base = channel * layout.conv_state_tokens;
                for state_index in 0..layout.conv_state_tokens {
                    let source_position =
                        step as isize - (layout.conv_state_tokens - 1 - state_index) as isize;
                    let expected_state = if source_position == step as isize {
                        current[channel]
                    } else if source_position >= 0 {
                        full_history[source_position as usize][channel]
                    } else {
                        0.0
                    };
                    assert!(
                        (next_state[state_base + state_index] - expected_state).abs() <= 1.0e-7,
                        "compact causal state drifted at step {step}, channel {channel}, state index {state_index}"
                    );
                }
            }
            compact_prior_state = next_state;
            full_history.push(current);
        }
    }

    #[test]
    fn direct_packed_embedding_row_reader_matches_materialized_reference() {
        // Keep this fixture tiny while exercising the exact fixed-128 compact
        // row reader used for the real [151936, 2048] embedding body.
        let payload = direct_packed_fixture_payload(&[2, QWEN80_GROUP_SIZE], 41);
        let header = parse_complete_binary_header(&payload).unwrap();
        let embedding = Qwen80CpuPackedTensor {
            name: "fixture.embed_tokens.weight".to_owned(),
            header,
            payload,
        };
        let direct = embedding
            .row(
                1,
                2,
                QWEN80_GROUP_SIZE,
                Qwen80CpuPackedReadMode::StreamingDirect,
            )
            .expect("direct compact embedding row must decode");
        let materialized = embedding
            .row(
                1,
                2,
                QWEN80_GROUP_SIZE,
                Qwen80CpuPackedReadMode::MaterializedReference,
            )
            .expect("materialized embedding row must decode");
        assert_eq!(direct.len(), QWEN80_GROUP_SIZE);
        assert!(max_abs_error_cpu(&direct, &materialized, "embedding row").unwrap() <= 1.0e-6);
        assert!(embedding
            .row(
                2,
                2,
                QWEN80_GROUP_SIZE,
                Qwen80CpuPackedReadMode::StreamingDirect
            )
            .is_err());
    }

    #[test]
    fn embedding_oracle_refuses_source_reserved_tail_before_payload_access() {
        let temp = tempfile::tempdir().unwrap();
        let catalog =
            Qwen80CompleteArtifactCatalog::from_admitted(already_admitted_catalog_fixture(&temp))
                .expect("admitted exact catalog fixture must bind without payload reads");
        let error = catalog
            .execute_embedding_lookup_cpu_oracle(QWEN80_TOKENIZER_VOCAB as u32)
            .expect_err("the first source-reserved row must not be embeddable")
            .to_string();
        assert!(error.contains("reserved"));
        assert!(error.contains("151669"));
    }

    #[test]
    fn complete_tensor_catalog_has_exact_qwen80_body_and_hybrid_operators() {
        let tensors = tensor_shapes();
        assert_eq!(tensors.len(), 74_391);
        assert_eq!(
            tensors["model.layers.0.linear_attn.in_proj_qkvz.weight"],
            vec![12_288, 2_048]
        );
        assert_eq!(
            tensors["model.layers.3.self_attn.q_proj.weight"],
            vec![8_192, 2_048]
        );
        assert!(tensors.contains_key("model.layers.47.self_attn.o_proj.weight"));
        assert!(!tensors.contains_key("model.layers.47.linear_attn.out_proj.weight"));
        assert_eq!(
            tensors["model.layers.0.mlp.shared_expert_gate.weight"],
            vec![1, 2_048]
        );
    }

    #[test]
    fn admitted_catalog_handoff_does_not_reopen_payloads_after_one_strict_scan() {
        let temp = tempfile::tempdir().unwrap();
        let catalog =
            Qwen80CompleteArtifactCatalog::from_admitted(already_admitted_catalog_fixture(&temp))
                .expect("an already-admitted exact catalog must hand off without payload reads");
        assert_eq!(catalog.tensor_count(), 74_391);
        assert_eq!(catalog.manifest_seal(), "a".repeat(64));

        // No fixture payload exists. `from_admitted` succeeded above only
        // because it reused the completed admission/catalog; a direct payload
        // read remains separately guarded and correctly refuses now.
        assert!(catalog
            .read_direct_tensor_payload("model.layers.0.linear_attn.in_proj_qkvz.weight")
            .is_err());
    }

    #[test]
    fn canonical_linear_deltanet_cpu_stage_reads_verified_packed_payloads_and_matches_reference() {
        let temp = tempfile::tempdir().unwrap();
        let catalog = canonical_linear_cpu_catalog_fixture(&temp);
        let contract = catalog
            .canonical_linear_deltanet_operator_contract(0)
            .expect("layer 0 must bind an artifact-backed DeltaNet operator contract");
        let qkvz_name = "model.layers.0.linear_attn.in_proj_qkvz.weight";
        assert!(
            catalog
                .verified_direct_tensor_payload(qkvz_name)
                .expect("the CPU stage must read the admission snapshot")
                .len()
                > 32
        );

        let mut state = Qwen80CanonicalLinearLayerCpuState::zeroed();
        assert_eq!(state.conv_state.len(), 8_192 * 3);
        assert_eq!(state.recurrent_state.len(), 32 * 128 * 128);
        for (index, value) in state.conv_state.iter_mut().enumerate() {
            *value = (index % 7) as f32 * 0.0003 - 0.0009;
        }
        for (index, value) in state.recurrent_state.iter_mut().enumerate() {
            *value = (index % 11) as f32 * 0.00002 - 0.0001;
        }
        let input = Qwen80CanonicalLinearLayerCpuInput {
            hidden: (0..QWEN80_HIDDEN)
                .map(|index| (index % 53) as f32 * 0.002 - 0.052)
                .collect(),
            state,
        };

        let direct = catalog
            .execute_canonical_linear_deltanet_cpu_oracle(&contract, &input)
            .expect("direct compact DeltaNet CPU stage must execute source geometry");
        let reference = catalog
            .execute_first_linear_layer_cpu_materialized_reference(&input)
            .expect("materialized CPU reference must execute the identical source stage");

        assert_eq!(direct.layer, 0);
        assert_eq!(direct.linear_state_slot, 0);
        assert_eq!(direct.direct_packed_qkvz_tensor, qkvz_name);
        assert_eq!(
            direct.direct_packed_out_proj_tensor,
            "model.layers.0.linear_attn.out_proj.weight"
        );
        assert!(direct
            .source_algorithm_boundary
            .contains("post-attention norm and routed/shared MoE are intentionally absent"));
        assert!(
            max_abs_error_cpu(
                &reference.input_rms_norm_output,
                &direct.input_rms_norm_output,
                "input RMSNorm",
            )
            .unwrap()
                <= 1.0e-6
        );
        for (label, reference_values, direct_values) in [
            (
                "QKVZ projection",
                reference.projected_qkvz.as_slice(),
                direct.projected_qkvz.as_slice(),
            ),
            (
                "BA projection",
                reference.projected_ba.as_slice(),
                direct.projected_ba.as_slice(),
            ),
            (
                "repeated Q L2",
                reference.repeated_query_l2_scaled.as_slice(),
                direct.repeated_query_l2_scaled.as_slice(),
            ),
            (
                "repeated K L2",
                reference.repeated_key_l2.as_slice(),
                direct.repeated_key_l2.as_slice(),
            ),
            (
                "causal convolved V",
                reference.convolved_value.as_slice(),
                direct.convolved_value.as_slice(),
            ),
            ("Z gate", reference.z.as_slice(), direct.z.as_slice()),
            ("decay", reference.decay.as_slice(), direct.decay.as_slice()),
            ("beta", reference.beta.as_slice(), direct.beta.as_slice()),
            (
                "recurrent output",
                reference.recurrent_output.as_slice(),
                direct.recurrent_output.as_slice(),
            ),
            (
                "gated RMSNorm output",
                reference.gated_rms_norm_output.as_slice(),
                direct.gated_rms_norm_output.as_slice(),
            ),
        ] {
            assert!(
                max_abs_error_cpu(reference_values, direct_values, label).unwrap() <= 1.0e-6,
                "{label} direct-packed/materialized source parity drifted"
            );
        }
        assert!(
            max_abs_error_cpu(
                &reference.mixer_output,
                &direct.mixer_output,
                "mixer output"
            )
            .unwrap()
                <= 1.0e-6
        );
        assert!(
            max_abs_error_cpu(
                &reference.mixer_residual_output,
                &direct.mixer_residual_output,
                "mixer residual",
            )
            .unwrap()
                <= 1.0e-6
        );
        assert!(
            max_abs_error_cpu(
                &reference.next_state.conv_state,
                &direct.next_state.conv_state,
                "convolution state",
            )
            .unwrap()
                <= 1.0e-6
        );
        assert!(
            max_abs_error_cpu(
                &reference.next_state.recurrent_state,
                &direct.next_state.recurrent_state,
                "recurrent state",
            )
            .unwrap()
                <= 1.0e-6
        );
        assert_ne!(direct.next_state.conv_state, input.state.conv_state);
        assert_ne!(
            direct.next_state.recurrent_state,
            input.state.recurrent_state
        );
    }

    #[test]
    fn canonical_linear_moe_cpu_stage_routes_all_top10_experts_and_matches_materialized_reference()
    {
        let temp = tempfile::tempdir().unwrap();
        let input = canonical_linear_cpu_input_fixture();
        let catalog = canonical_linear_moe_cpu_catalog_fixture(&temp, &input);
        let contract = catalog
            .canonical_linear_moe_operator_contract(0)
            .expect("layer 0 fixed post-DeltaNet MoE contract must bind admitted compact tensors");
        assert_eq!(contract.mixer.layer, 0);
        assert_eq!(
            contract.post_attention_layernorm.name,
            "model.layers.0.post_attention_layernorm.weight"
        );
        assert_eq!(contract.router.shape, vec![QWEN80_EXPERTS, QWEN80_HIDDEN]);

        let direct = catalog
            .execute_canonical_linear_moe_cpu_oracle(&contract, &input)
            .expect("direct compact layer-0 DeltaNet + source top-10 MoE must execute");
        let reference = catalog
            .execute_first_linear_layer_cpu_moe_materialized_reference(&input)
            .expect("materialized reference must execute the identical layer-0 source path");

        assert_eq!(direct.mixer.layer, 0);
        assert_eq!(direct.route, reference.route);
        assert_eq!(direct.route.ids.len(), QWEN80_TOP_K);
        assert_eq!(direct.routed_experts.len(), QWEN80_TOP_K);
        assert!(direct
            .source_algorithm_boundary
            .contains("all ten routed gate/up/down waves"));
        assert_eq!(
            direct
                .routed_experts
                .iter()
                .map(|expert| expert.expert as u16)
                .collect::<Vec<_>>(),
            direct.route.ids,
            "every loaded expert must be selected by this exact compact router result"
        );
        assert!(
            (direct.route.weights.iter().sum::<f32>() - 1.0).abs() <= 1.0e-4,
            "source norm_topk_prob weights must remain normalized"
        );

        for (label, reference_values, direct_values) in [
            (
                "post-attention RMSNorm",
                reference.post_attention_rms_norm_output.as_slice(),
                direct.post_attention_rms_norm_output.as_slice(),
            ),
            (
                "router logits",
                reference.router_logits.as_slice(),
                direct.router_logits.as_slice(),
            ),
            (
                "routed expert sum",
                reference.routed_expert_sum.as_slice(),
                direct.routed_expert_sum.as_slice(),
            ),
            (
                "shared gate projection",
                reference.shared_gate_projection.as_slice(),
                direct.shared_gate_projection.as_slice(),
            ),
            (
                "shared up projection",
                reference.shared_up_projection.as_slice(),
                direct.shared_up_projection.as_slice(),
            ),
            (
                "shared gate/up product",
                reference.shared_gated_up_product.as_slice(),
                direct.shared_gated_up_product.as_slice(),
            ),
            (
                "shared expert output",
                reference.shared_expert_output.as_slice(),
                direct.shared_expert_output.as_slice(),
            ),
            (
                "shared gated output",
                reference.shared_gated_output.as_slice(),
                direct.shared_gated_output.as_slice(),
            ),
            (
                "MoE output",
                reference.moe_output.as_slice(),
                direct.moe_output.as_slice(),
            ),
            (
                "second residual",
                reference.layer_output.as_slice(),
                direct.layer_output.as_slice(),
            ),
        ] {
            assert!(
                max_abs_error_cpu(reference_values, direct_values, label).unwrap() <= 1.0e-6,
                "{label} direct-packed/materialized parity drifted"
            );
        }
        assert!(
            (reference.shared_expert_gate_logit - direct.shared_expert_gate_logit).abs() <= 1.0e-6
        );
        assert!(
            (reference.shared_expert_gate_value - direct.shared_expert_gate_value).abs() <= 1.0e-6
        );
        for (reference_expert, direct_expert) in
            reference.routed_experts.iter().zip(&direct.routed_experts)
        {
            assert_eq!(reference_expert.expert, direct_expert.expert);
            assert!((reference_expert.route_weight - direct_expert.route_weight).abs() <= 1.0e-6);
            for (label, reference_values, direct_values) in [
                (
                    "routed gate projection",
                    reference_expert.gate_projection.as_slice(),
                    direct_expert.gate_projection.as_slice(),
                ),
                (
                    "routed up projection",
                    reference_expert.up_projection.as_slice(),
                    direct_expert.up_projection.as_slice(),
                ),
                (
                    "routed gate/up product",
                    reference_expert.gated_up_product.as_slice(),
                    direct_expert.gated_up_product.as_slice(),
                ),
                (
                    "routed expert output",
                    reference_expert.output.as_slice(),
                    direct_expert.output.as_slice(),
                ),
                (
                    "weighted routed expert output",
                    reference_expert.weighted_output.as_slice(),
                    direct_expert.weighted_output.as_slice(),
                ),
            ] {
                assert!(
                    max_abs_error_cpu(reference_values, direct_values, label).unwrap() <= 1.0e-6,
                    "expert {} {label} direct-packed/materialized parity drifted",
                    direct_expert.expert
                );
            }
        }
    }

    #[test]
    fn all_ten_routed_expert_cpu_oracle_executes_only_the_bound_source_wave() {
        let temp = tempfile::tempdir().unwrap();
        let input = canonical_linear_cpu_input_fixture();
        let catalog = canonical_linear_moe_cpu_catalog_fixture(&temp, &input);
        let complete_plan = catalog.complete_hybrid_decoder_plan(2).unwrap();
        let full_layer = catalog
            .execute_first_linear_layer_cpu_moe_oracle(&input)
            .expect("fixture must establish its exact compact router route");
        let descriptor =
            all_ten_route_descriptor_fixture(&catalog, &complete_plan, &full_layer.route);
        let manifest_document_sha256 = "b".repeat(64);
        let plan_document_sha256 = "a".repeat(64);
        let router_receipt_sha256 = "e".repeat(64);
        let router_outer_receipt_sha256 = "c".repeat(64);
        let router_outer_receipt_seal_sha256 = "d".repeat(64);
        let authority = Qwen80AllTenRoutedExpertPlanAuthority {
            manifest_document_sha256: &manifest_document_sha256,
            plan_document_sha256: &plan_document_sha256,
            router_receipt_sha256: &router_receipt_sha256,
            router_outer_receipt_sha256: &router_outer_receipt_sha256,
            router_outer_receipt_seal_sha256: &router_outer_receipt_seal_sha256,
        };
        let bound = complete_plan
            .bind_all_ten_routed_expert_plan(0, &authority, &descriptor)
            .expect("all ten descriptor bindings must match the admitted fixture catalog");
        let wave = catalog
            .execute_all_ten_routed_expert_cpu_oracle(
                &bound,
                &full_layer.post_attention_rms_norm_output,
            )
            .expect("every descriptor-selected direct-packed wave must execute on CPU reference");

        assert_eq!(wave.layer, 0);
        assert_eq!(wave.route, full_layer.route);
        assert_eq!(wave.witnesses.len(), QWEN80_TOP_K);
        assert_eq!(wave.routed_expert_sum, full_layer.routed_expert_sum);
        assert!(wave
            .source_algorithm_boundary
            .contains("shared expert, route combine, second residual"));
        for (index, (witness, full)) in wave
            .witnesses
            .iter()
            .zip(&full_layer.routed_experts)
            .enumerate()
        {
            assert_eq!(witness.wave_index, index);
            assert_eq!(witness.expert, full.expert);
            assert!((witness.normalized_weight - full.route_weight).abs() <= 1.0e-6);
            assert_eq!(witness.oracle.weighted_output, full.weighted_output);
            assert_eq!(
                witness.weighted_output_sha256,
                qwen80_f32_vector_sha256(&full.weighted_output, "fixture routed output").unwrap()
            );
        }
    }

    #[test]
    fn all_ten_routed_expert_cpu_oracle_refuses_descriptor_substitution_and_missing_payload() {
        let temp = tempfile::tempdir().unwrap();
        let input = canonical_linear_cpu_input_fixture();
        let mut catalog = canonical_linear_moe_cpu_catalog_fixture(&temp, &input);
        let complete_plan = catalog.complete_hybrid_decoder_plan(2).unwrap();
        let full_layer = catalog
            .execute_first_linear_layer_cpu_moe_oracle(&input)
            .expect("fixture must establish its exact compact router route");
        let manifest_document_sha256 = "b".repeat(64);
        let plan_document_sha256 = "a".repeat(64);
        let router_receipt_sha256 = "e".repeat(64);
        let router_outer_receipt_sha256 = "c".repeat(64);
        let router_outer_receipt_seal_sha256 = "d".repeat(64);
        let authority = Qwen80AllTenRoutedExpertPlanAuthority {
            manifest_document_sha256: &manifest_document_sha256,
            plan_document_sha256: &plan_document_sha256,
            router_receipt_sha256: &router_receipt_sha256,
            router_outer_receipt_sha256: &router_outer_receipt_sha256,
            router_outer_receipt_seal_sha256: &router_outer_receipt_seal_sha256,
        };
        let mut substituted =
            all_ten_route_descriptor_fixture(&catalog, &complete_plan, &full_layer.route);
        substituted["deterministic_waves"][3]["up"]["artifact_sha256"] = json!("f".repeat(64));
        let substituted = complete_plan
            .bind_all_ten_routed_expert_plan(0, &authority, &substituted)
            .expect("descriptor syntax alone cannot claim payload identity");
        assert!(catalog
            .execute_all_ten_routed_expert_cpu_oracle(
                &substituted,
                &full_layer.post_attention_rms_norm_output,
            )
            .is_err());

        let descriptor =
            all_ten_route_descriptor_fixture(&catalog, &complete_plan, &full_layer.route);
        let bound = complete_plan
            .bind_all_ten_routed_expert_plan(0, &authority, &descriptor)
            .unwrap();
        let missing_name = format!(
            "model.layers.0.mlp.experts.{}.down_proj.weight",
            full_layer.route.ids[6]
        );
        assert!(catalog
            .artifact
            .verified_payloads
            .remove(&missing_name)
            .is_some());
        assert!(catalog
            .execute_all_ten_routed_expert_cpu_oracle(
                &bound,
                &full_layer.post_attention_rms_norm_output,
            )
            .is_err());
    }

    #[test]
    fn all_ten_device_bridge_source_bundle_is_exactly_admission_bound_and_route_ordered() {
        let temp = tempfile::tempdir().unwrap();
        let input = canonical_linear_cpu_input_fixture();
        let catalog = canonical_linear_moe_cpu_catalog_fixture(&temp, &input);
        let complete_plan = catalog.complete_hybrid_decoder_plan(2).unwrap();
        let full_layer = catalog
            .execute_first_linear_layer_cpu_moe_oracle(&input)
            .expect("fixture must establish the compact source top-10 route");
        let descriptor =
            all_ten_route_descriptor_fixture(&catalog, &complete_plan, &full_layer.route);
        let manifest_document_sha256 = "b".repeat(64);
        let plan_document_sha256 = "a".repeat(64);
        let router_receipt_sha256 = "e".repeat(64);
        let router_outer_receipt_sha256 = "c".repeat(64);
        let router_outer_receipt_seal_sha256 = "d".repeat(64);
        let authority = Qwen80AllTenRoutedExpertPlanAuthority {
            manifest_document_sha256: &manifest_document_sha256,
            plan_document_sha256: &plan_document_sha256,
            router_receipt_sha256: &router_receipt_sha256,
            router_outer_receipt_sha256: &router_outer_receipt_sha256,
            router_outer_receipt_seal_sha256: &router_outer_receipt_seal_sha256,
        };
        let route_plan = complete_plan
            .bind_all_ten_routed_expert_plan(0, &authority, &descriptor)
            .unwrap();
        let bundle = catalog
            .build_all_ten_route_payload_bundle(&route_plan)
            .unwrap();
        let first_residual = catalog.first_residual_device_binding(0).unwrap();
        let bridge = catalog
            .build_all_ten_true_moe_source_bridge(&route_plan, first_residual.clone())
            .unwrap();

        let scales_per_projection = QWEN80_HIDDEN * QWEN80_MOE_INTERMEDIATE / QWEN80_GROUP_SIZE
            * std::mem::size_of::<u16>();
        let signs_per_projection = QWEN80_HIDDEN * QWEN80_MOE_INTERMEDIATE / 8;
        assert_eq!(bundle.layer(), 0);
        assert_eq!(bundle.route(), &full_layer.route);
        assert_eq!(bundle.waves().len(), QWEN80_TOP_K);
        assert_eq!(
            bundle.gate_scales().len(),
            scales_per_projection * QWEN80_TOP_K
        );
        assert_eq!(
            bundle.gate_signs().len(),
            signs_per_projection * QWEN80_TOP_K
        );
        assert_eq!(
            bundle.up_scales().len(),
            scales_per_projection * QWEN80_TOP_K
        );
        assert_eq!(
            bundle.down_signs().len(),
            signs_per_projection * QWEN80_TOP_K
        );
        assert_eq!(
            bundle.gate_scales_sha256(),
            qwen80_sha256_hex(bundle.gate_scales())
        );
        assert_eq!(first_residual.layer(), 0);
        assert_eq!(first_residual.linear_state_slot(), 0);
        assert_eq!(first_residual.elements(), QWEN80_HIDDEN);
        assert!(first_residual.same_command_graph_required());
        assert_eq!(bridge.route_payloads().route(), &full_layer.route);
        assert_eq!(bridge.first_residual(), &first_residual);
        bundle
            .require_router_route(&full_layer.route, 0.0)
            .expect("exact source route must be accepted by its compact body bundle");
        let mut reordered_route = full_layer.route.clone();
        reordered_route.ids.swap(0, 1);
        reordered_route.weights.swap(0, 1);
        assert!(bundle
            .require_router_route(&reordered_route, 2.0e-5)
            .is_err());
        let mut wrong_weight_route = full_layer.route.clone();
        let donor = wrong_weight_route
            .weights
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .unwrap();
        let recipient = (donor + 1) % QWEN80_TOP_K;
        wrong_weight_route.weights[donor] -= 1.0e-3;
        wrong_weight_route.weights[recipient] += 1.0e-3;
        assert!(wrong_weight_route.validate().is_ok());
        assert!(bundle
            .require_router_route(&wrong_weight_route, 0.0)
            .is_err());

        for (wave_index, wave) in bundle.waves().iter().enumerate() {
            assert_eq!(wave.wave_index, wave_index);
            assert_eq!(wave.expert, full_layer.route.ids[wave_index]);
            assert_eq!(
                wave.normalized_weight.to_bits(),
                full_layer.route.weights[wave_index].to_bits()
            );
            for (section, compact_sections, projection) in [
                (
                    &wave.gate,
                    (bundle.gate_scales(), bundle.gate_signs()),
                    "gate_proj",
                ),
                (&wave.up, (bundle.up_scales(), bundle.up_signs()), "up_proj"),
                (
                    &wave.down,
                    (bundle.down_scales(), bundle.down_signs()),
                    "down_proj",
                ),
            ] {
                assert_eq!(
                    section.scale_offset_bytes,
                    wave_index * scales_per_projection
                );
                assert_eq!(section.sign_offset_bytes, wave_index * signs_per_projection);
                let payload = catalog
                    .verified_direct_tensor_payload(&section.tensor_name)
                    .unwrap();
                let header = parse_complete_binary_header(&payload).unwrap();
                assert_eq!(
                    &compact_sections.0[section.scale_offset_bytes
                        ..section.scale_offset_bytes + section.scale_bytes],
                    &payload[header.scale_offset..header.sign_offset],
                    "route {wave_index} {projection} scale body must be copied in exact source order"
                );
                assert_eq!(
                    &compact_sections.1[section.sign_offset_bytes
                        ..section.sign_offset_bytes + section.sign_bytes],
                    &payload[header.sign_offset..header.payload_bytes],
                    "route {wave_index} {projection} sign body must be copied in exact source order"
                );
            }
        }
    }

    #[test]
    fn all_ten_device_bridge_refuses_missing_or_wrong_layer_first_residual_authority() {
        let temp = tempfile::tempdir().unwrap();
        let input = canonical_linear_cpu_input_fixture();
        let mut catalog = canonical_linear_moe_cpu_catalog_fixture(&temp, &input);
        let complete_plan = catalog.complete_hybrid_decoder_plan(2).unwrap();
        let full_layer = catalog
            .execute_first_linear_layer_cpu_moe_oracle(&input)
            .expect("fixture must establish the compact source top-10 route");
        let manifest_document_sha256 = "b".repeat(64);
        let plan_document_sha256 = "a".repeat(64);
        let router_receipt_sha256 = "e".repeat(64);
        let router_outer_receipt_sha256 = "c".repeat(64);
        let router_outer_receipt_seal_sha256 = "d".repeat(64);
        let authority = Qwen80AllTenRoutedExpertPlanAuthority {
            manifest_document_sha256: &manifest_document_sha256,
            plan_document_sha256: &plan_document_sha256,
            router_receipt_sha256: &router_receipt_sha256,
            router_outer_receipt_sha256: &router_outer_receipt_sha256,
            router_outer_receipt_seal_sha256: &router_outer_receipt_seal_sha256,
        };
        let descriptor =
            all_ten_route_descriptor_fixture(&catalog, &complete_plan, &full_layer.route);
        let route_plan = complete_plan
            .bind_all_ten_routed_expert_plan(0, &authority, &descriptor)
            .unwrap();
        let wrong_layer = catalog.first_residual_device_binding(1).unwrap();
        assert!(catalog
            .build_all_ten_true_moe_source_bridge(&route_plan, wrong_layer)
            .is_err());

        let missing_name = format!(
            "model.layers.0.mlp.experts.{}.up_proj.weight",
            full_layer.route.ids[4]
        );
        assert!(catalog
            .artifact
            .verified_payloads
            .remove(&missing_name)
            .is_some());
        assert!(catalog
            .build_all_ten_route_payload_bundle(&route_plan)
            .is_err());
    }

    #[test]
    fn canonical_linear_moe_cpu_stage_refuses_a_missing_router_selected_expert_payload() {
        let temp = tempfile::tempdir().unwrap();
        let input = canonical_linear_cpu_input_fixture();
        let mut catalog = canonical_linear_moe_cpu_catalog_fixture(&temp, &input);
        let contract = catalog
            .canonical_linear_moe_operator_contract(0)
            .expect("the fixed layer-0 contract must bind before an expert is selected");
        let initial = catalog
            .execute_canonical_linear_moe_cpu_oracle(&contract, &input)
            .expect("fixture must first establish an actual source top-10 route");
        let selected = initial.route.ids[0] as usize;
        let missing = format!("model.layers.0.mlp.experts.{selected}.down_proj.weight");
        assert!(catalog
            .artifact
            .verified_payloads
            .remove(&missing)
            .is_some());

        // The fixed contract remains structurally addressable, but execution
        // must fail closed before it can substitute an arbitrary expert body.
        assert!(catalog
            .execute_canonical_linear_moe_cpu_oracle(&contract, &input)
            .is_err());
    }

    #[test]
    fn canonical_linear_deltanet_contract_locks_source_offsets_and_device_resources() {
        let temp = tempfile::tempdir().unwrap();
        let catalog = canonical_linear_cpu_catalog_fixture(&temp);
        let contract = catalog
            .canonical_linear_deltanet_operator_contract(0)
            .expect("the eight admitted layer-0 payloads must produce a strict contract");

        assert_eq!(contract.layer, 0);
        assert_eq!(contract.linear_state_slot, 0);
        assert_eq!(contract.layout.qkvz_rows_per_key_head, 768);
        assert_eq!(
            contract
                .layout
                .qkvz_row_offset(15, Qwen80LinearQkvzSegment::Query)
                .unwrap(),
            11_520
        );
        assert_eq!(
            contract
                .layout
                .qkvz_row_offset(15, Qwen80LinearQkvzSegment::Key)
                .unwrap(),
            11_648
        );
        assert_eq!(
            contract
                .layout
                .qkvz_row_offset(15, Qwen80LinearQkvzSegment::Value)
                .unwrap(),
            11_776
        );
        assert_eq!(
            contract
                .layout
                .qkvz_row_offset(15, Qwen80LinearQkvzSegment::Z)
                .unwrap(),
            12_032
        );
        assert_eq!(
            contract
                .layout
                .ba_row_offset(15, Qwen80LinearBaSegment::BetaLogit)
                .unwrap(),
            60
        );
        assert_eq!(
            contract
                .layout
                .ba_row_offset(15, Qwen80LinearBaSegment::DecayInput)
                .unwrap(),
            62
        );
        assert_eq!(
            contract
                .minimum_device_resources
                .conv_state_capacity_elements,
            24_576
        );
        assert_eq!(
            contract
                .minimum_device_resources
                .recurrent_state_capacity_elements,
            524_288
        );
        assert_eq!(
            contract
                .minimum_device_resources
                .direct_packed_payload_bytes
                .len(),
            8
        );
        contract
            .validate_device_resources(&contract.minimum_device_resources)
            .expect("the contract's own source-shaped resource floor must validate");

        let mut wrong_state_slot = contract.minimum_device_resources.clone();
        wrong_state_slot.recurrent_state_offset_elements += 1;
        assert!(contract
            .validate_device_resources(&wrong_state_slot)
            .is_err());
        let mut wrong_payload = contract.minimum_device_resources.clone();
        *wrong_payload
            .direct_packed_payload_bytes
            .get_mut("model.layers.0.linear_attn.in_proj_qkvz.weight")
            .unwrap() += 1;
        assert!(contract.validate_device_resources(&wrong_payload).is_err());
        assert!(catalog
            .canonical_linear_deltanet_operator_contract(3)
            .is_err());
    }

    #[test]
    fn layer_one_deltanet_contract_uses_slot_one_offsets_and_zero_state_cpu_oracle() {
        let temp = tempfile::tempdir().unwrap();
        let catalog = canonical_layer_one_linear_cpu_catalog_fixture(&temp);
        let contract = catalog
            .canonical_linear_deltanet_operator_contract(1)
            .expect("layer 1 must bind the second source-scheduled DeltaNet slot");
        assert_eq!(contract.layer, 1);
        assert_eq!(contract.linear_state_slot, 1);
        assert_eq!(
            contract.minimum_device_resources.conv_state_offset_elements,
            24_576
        );
        assert_eq!(
            contract
                .minimum_device_resources
                .conv_state_capacity_elements,
            49_152
        );
        assert_eq!(
            contract
                .minimum_device_resources
                .recurrent_state_offset_elements,
            524_288
        );
        assert_eq!(
            contract
                .minimum_device_resources
                .recurrent_state_capacity_elements,
            1_048_576
        );
        let input = Qwen80CanonicalLinearLayerCpuInput::with_zero_state(
            (0..QWEN80_HIDDEN)
                .map(|index| (index % 29) as f32 * 0.003 - 0.041)
                .collect(),
        );
        let oracle = catalog
            .execute_canonical_linear_deltanet_cpu_oracle(&contract, &input)
            .expect("the layer-one compact CPU oracle must accept a source-zero slot-one state");
        assert_eq!(oracle.layer, 1);
        assert_eq!(oracle.linear_state_slot, 1);
        assert_eq!(oracle.mixer_residual_output.len(), QWEN80_HIDDEN);
        assert!(oracle
            .mixer_residual_output
            .iter()
            .all(|value| value.is_finite()));
        assert_eq!(oracle.next_state.conv_state.len(), 24_576);
        assert_eq!(oracle.next_state.recurrent_state.len(), 524_288);
    }

    #[test]
    fn state_geometry_keeps_deltanet_and_attention_domains_distinct() {
        let config = Qwen80CompleteRuntimeConfig::from_source_config(
            &source_config(),
            QWEN80_REPOSITORY,
            "revision",
        )
        .unwrap();
        let state = Qwen80NativeStateGeometry::from_config(&config, 4096).unwrap();
        assert_eq!(state.linear_layers, 36);
        assert_eq!(state.full_attention_layers, 12);
        assert_eq!(state.linear_conv_state_elements, 884_736);
        assert_eq!(state.linear_recurrent_state_elements, 18_874_368);
        assert_eq!(state.full_attention_key_cache_elements, 25_165_824);
        assert_eq!(state.full_attention_value_cache_elements, 25_165_824);
        assert_eq!(state.total_f32_bytes().unwrap(), 280_363_008);
    }

    #[test]
    fn tokenizer_namespace_is_strictly_smaller_than_lm_head_namespace() {
        assert_eq!(QWEN80_VOCAB - QWEN80_TOKENIZER_VOCAB, 267);
        assert!(QWEN80_TOKENIZER_VOCAB < QWEN80_VOCAB);
    }

    #[derive(Default)]
    struct RecordingHybridBackend {
        events: Vec<String>,
        linear_slots: Vec<(usize, usize)>,
        attention_slots: Vec<(usize, usize, usize)>,
        routed: Vec<(usize, usize, usize)>,
        sampled: u32,
        invalid_route: bool,
    }

    impl RecordingHybridBackend {
        fn event(&mut self, value: impl Into<String>) {
            self.events.push(value.into());
        }
    }

    impl Qwen80PackedHybridDecodeBackend for RecordingHybridBackend {
        fn begin_token(
            &mut self,
            token_id: u32,
            position: usize,
            embedding: &Qwen80PackedTensorBinding,
        ) -> Result<()> {
            assert_eq!(embedding.name, "model.embed_tokens.weight");
            self.event(format!("begin:{token_id}:{position}"));
            Ok(())
        }

        fn input_rms_norm(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
            self.event(format!("input_norm:{}", layer.layer));
            Ok(())
        }

        fn linear_deltanet_mixer(
            &mut self,
            layer: &Qwen80HybridDecoderLayerPlan,
            state_slot: usize,
            mixer: &Qwen80LinearDeltaNetLayerBindings,
        ) -> Result<()> {
            assert!(mixer.in_proj_qkvz.name.ends_with("in_proj_qkvz.weight"));
            self.linear_slots.push((layer.layer, state_slot));
            self.event(format!("linear:{}:{state_slot}", layer.layer));
            Ok(())
        }

        fn full_attention_mixer(
            &mut self,
            layer: &Qwen80HybridDecoderLayerPlan,
            state_slot: usize,
            mixer: &Qwen80FullAttentionLayerBindings,
            position: usize,
        ) -> Result<()> {
            assert!(mixer.q_proj.name.ends_with("self_attn.q_proj.weight"));
            self.attention_slots
                .push((layer.layer, state_slot, position));
            self.event(format!("attention:{}:{state_slot}:{position}", layer.layer));
            Ok(())
        }

        fn add_mixer_residual(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
            self.event(format!("mixer_residual:{}", layer.layer));
            Ok(())
        }

        fn post_attention_rms_norm(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
            self.event(format!("post_norm:{}", layer.layer));
            Ok(())
        }

        fn route_top10(
            &mut self,
            layer: &Qwen80HybridDecoderLayerPlan,
        ) -> Result<Qwen80RouteSelection> {
            self.event(format!("route:{}", layer.layer));
            let mut ids = [0u16; QWEN80_TOP_K];
            for (index, id) in ids.iter_mut().enumerate() {
                *id = ((layer.layer * QWEN80_TOP_K + index) % QWEN80_EXPERTS) as u16;
            }
            if self.invalid_route {
                ids[QWEN80_TOP_K - 1] = ids[0];
            }
            Ok(Qwen80RouteSelection {
                ids,
                weights: [0.1; QWEN80_TOP_K],
            })
        }

        fn routed_expert(
            &mut self,
            layer: &Qwen80HybridDecoderLayerPlan,
            route_index: usize,
            route_weight: f32,
            expert: &Qwen80ExpertBindings,
        ) -> Result<()> {
            assert!((route_weight - 0.1).abs() < 1.0e-6);
            assert!(expert.gate_proj.name.contains(&format!(
                "model.layers.{}.mlp.experts.{}",
                layer.layer, expert.expert
            )));
            self.routed.push((layer.layer, route_index, expert.expert));
            self.event(format!(
                "expert:{}:{route_index}:{}",
                layer.layer, expert.expert
            ));
            Ok(())
        }

        fn shared_expert(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
            assert!(layer
                .moe
                .shared_expert_gate
                .name
                .ends_with("mlp.shared_expert_gate.weight"));
            self.event(format!("shared:{}", layer.layer));
            Ok(())
        }

        fn combine_moe_and_add_residual(
            &mut self,
            layer: &Qwen80HybridDecoderLayerPlan,
        ) -> Result<()> {
            self.event(format!("moe_residual:{}", layer.layer));
            Ok(())
        }

        fn final_rms_norm(&mut self, final_norm: &Qwen80PackedTensorBinding) -> Result<()> {
            assert_eq!(final_norm.name, "model.norm.weight");
            self.event("final_norm");
            Ok(())
        }

        fn lm_head(&mut self, lm_head: &Qwen80PackedTensorBinding) -> Result<()> {
            assert_eq!(lm_head.name, "lm_head.weight");
            self.event("lm_head");
            Ok(())
        }

        fn mask_reserved_lm_head_tail(&mut self, first_reserved_id: u32) -> Result<()> {
            assert_eq!(first_reserved_id as usize, QWEN80_TOKENIZER_VOCAB);
            self.event(format!("mask_tail:{first_reserved_id}"));
            Ok(())
        }

        fn sample_token(&mut self, tokenizer_vocab_size: usize) -> Result<u32> {
            assert!((self.sampled as usize) < tokenizer_vocab_size);
            self.event(format!("sample:{}", self.sampled));
            Ok(self.sampled)
        }
    }

    // Reuse the same deterministic recorder through the per-layer bridge.
    // This is control-flow parity only: it does not decode a real payload or
    // stand in for the native Metal components.
    impl Qwen80HybridPerLayerComponentExecutor for RecordingHybridBackend {
        fn begin_token(
            &mut self,
            token_id: u32,
            position: usize,
            embedding: &Qwen80PackedTensorBinding,
        ) -> Result<()> {
            <Self as Qwen80PackedHybridDecodeBackend>::begin_token(
                self, token_id, position, embedding,
            )
        }

        fn input_rms_norm(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
            <Self as Qwen80PackedHybridDecodeBackend>::input_rms_norm(self, layer)
        }

        fn linear_deltanet_component(
            &mut self,
            layer: &Qwen80HybridDecoderLayerPlan,
            state_slot: usize,
            mixer: &Qwen80LinearDeltaNetLayerBindings,
        ) -> Result<()> {
            <Self as Qwen80PackedHybridDecodeBackend>::linear_deltanet_mixer(
                self, layer, state_slot, mixer,
            )
        }

        fn full_attention_component(
            &mut self,
            layer: &Qwen80HybridDecoderLayerPlan,
            state_slot: usize,
            mixer: &Qwen80FullAttentionLayerBindings,
            position: usize,
        ) -> Result<()> {
            <Self as Qwen80PackedHybridDecodeBackend>::full_attention_mixer(
                self, layer, state_slot, mixer, position,
            )
        }

        fn add_mixer_residual(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
            <Self as Qwen80PackedHybridDecodeBackend>::add_mixer_residual(self, layer)
        }

        fn post_attention_rms_norm(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
            <Self as Qwen80PackedHybridDecodeBackend>::post_attention_rms_norm(self, layer)
        }

        fn route_top10(
            &mut self,
            layer: &Qwen80HybridDecoderLayerPlan,
        ) -> Result<Qwen80RouteSelection> {
            <Self as Qwen80PackedHybridDecodeBackend>::route_top10(self, layer)
        }

        fn routed_expert(
            &mut self,
            layer: &Qwen80HybridDecoderLayerPlan,
            route_index: usize,
            route_weight: f32,
            expert: &Qwen80ExpertBindings,
        ) -> Result<()> {
            <Self as Qwen80PackedHybridDecodeBackend>::routed_expert(
                self,
                layer,
                route_index,
                route_weight,
                expert,
            )
        }

        fn shared_expert(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
            <Self as Qwen80PackedHybridDecodeBackend>::shared_expert(self, layer)
        }

        fn combine_moe_and_add_residual(
            &mut self,
            layer: &Qwen80HybridDecoderLayerPlan,
        ) -> Result<()> {
            <Self as Qwen80PackedHybridDecodeBackend>::combine_moe_and_add_residual(self, layer)
        }

        fn final_rms_norm(&mut self, final_norm: &Qwen80PackedTensorBinding) -> Result<()> {
            <Self as Qwen80PackedHybridDecodeBackend>::final_rms_norm(self, final_norm)
        }

        fn lm_head(&mut self, lm_head: &Qwen80PackedTensorBinding) -> Result<()> {
            <Self as Qwen80PackedHybridDecodeBackend>::lm_head(self, lm_head)
        }

        fn mask_reserved_lm_head_tail(&mut self, first_reserved_id: u32) -> Result<()> {
            <Self as Qwen80PackedHybridDecodeBackend>::mask_reserved_lm_head_tail(
                self,
                first_reserved_id,
            )
        }

        fn sample_token(&mut self, tokenizer_vocab_size: usize) -> Result<u32> {
            <Self as Qwen80PackedHybridDecodeBackend>::sample_token(self, tokenizer_vocab_size)
        }
    }

    fn expected_hybrid_control_events(token_id: u32, position: usize, sampled: u32) -> Vec<String> {
        let mut events = vec![format!("begin:{token_id}:{position}")];
        for layer in 0..QWEN80_LAYERS {
            events.push(format!("input_norm:{layer}"));
            if layer % QWEN80_FULL_ATTENTION_INTERVAL == 3 {
                events.push(format!("attention:{layer}:{}:{position}", layer / 4));
            } else {
                events.push(format!("linear:{layer}:{}", layer - layer / 4));
            }
            events.push(format!("mixer_residual:{layer}"));
            events.push(format!("post_norm:{layer}"));
            events.push(format!("route:{layer}"));
            for route_index in 0..QWEN80_TOP_K {
                let expert = (layer * QWEN80_TOP_K + route_index) % QWEN80_EXPERTS;
                events.push(format!("expert:{layer}:{route_index}:{expert}"));
            }
            events.push(format!("shared:{layer}"));
            events.push(format!("moe_residual:{layer}"));
        }
        events.extend([
            "final_norm".to_owned(),
            "lm_head".to_owned(),
            format!("mask_tail:{QWEN80_TOKENIZER_VOCAB}"),
            format!("sample:{sampled}"),
        ]);
        events
    }

    fn admitted_component_route() -> Qwen80RouteSelection {
        // This is the source-bound layer-0 route recorded by the admitted
        // direct DeltaNet component receipt.  It is consumed here only as a
        // bounded bridge input, never as a claim that all ten experts ran.
        Qwen80RouteSelection {
            ids: [132, 108, 12, 28, 321, 509, 490, 309, 193, 262],
            weights: [
                0.128_725_96,
                0.103_463_776,
                0.102_862_81,
                0.100_973_636,
                0.098_895_6,
                0.094_906_315,
                0.094_085_16,
                0.093_955_584,
                0.091_349_64,
                0.090_781_51,
            ],
        }
    }

    fn direct_linear_component_ledger(plan: &Qwen80CompleteHybridDecoderPlan) -> Value {
        let layer_zero = &plan.layers[0];
        let mixer = match &layer_zero.mixer {
            Qwen80HybridMixerBindings::LinearDeltaNet(mixer) => mixer,
            Qwen80HybridMixerBindings::FullAttention(_) => panic!("layer 0 must be DeltaNet"),
        };
        let route = admitted_component_route();
        json!({
            "status": QWEN80_DIRECT_PACKED_LINEAR_COMPONENT_STATUS,
            "manifest_seal_sha256": plan.manifest_seal_sha256,
            "model": { "revision": plan.source_revision },
            "native_state": {
                "status": QWEN80_DIRECT_PACKED_LINEAR_COMPONENT_STATUS,
                "stage": {
                    "layer": 0,
                    "layer_kind": "linear_attention",
                    "direct_packed_input_projection_tensor": mixer.in_proj_qkvz.name,
                    "direct_packed_ba_projection_tensor": mixer.in_proj_ba.name,
                    "direct_packed_router_tensor": layer_zero.moe.router.name,
                    "route_ids": route.ids.to_vec(),
                    "route_weights": route.weights.to_vec(),
                    "selected_expert": 132,
                    "direct_packed_selected_expert_gate_tensor": "model.layers.0.mlp.experts.132.gate_proj.weight",
                }
            }
        })
    }

    fn layer3_gqa_component_ledger(plan: &Qwen80CompleteHybridDecoderPlan) -> Value {
        let layer_three = &plan.layers[3];
        let mixer = match &layer_three.mixer {
            Qwen80HybridMixerBindings::FullAttention(mixer) => mixer,
            Qwen80HybridMixerBindings::LinearDeltaNet(_) => {
                panic!("layer 3 must be full attention")
            }
        };
        json!({
            "status": QWEN80_DIRECT_PACKED_LAYER3_GQA_COMPONENT_STATUS,
            "artifact": {
                "manifest_seal_sha256": plan.manifest_seal_sha256,
                "source_revision": plan.source_revision,
            },
            "source_bound_layer": {
                "layer": 3,
                "kind": "full_attention",
                "tensors": {
                    "q_proj": mixer.q_proj.name,
                    "k_proj": mixer.k_proj.name,
                    "v_proj": mixer.v_proj.name,
                    "o_proj": mixer.o_proj.name,
                    "q_norm": mixer.q_norm.name,
                    "k_norm": mixer.k_norm.name,
                },
                "geometry": { "fixture_positions": [0, 1] },
            },
            "metal_execution": {
                "performed": true,
                "compute_dispatches": 14,
            }
        })
    }

    struct RealMoeBoundaryFixture {
        admission_receipt_seal_sha256: String,
        manifest_document_sha256: String,
        router_receipt_sha256: String,
        router_outer_receipt_sha256: String,
        router_outer_receipt_seal_sha256: String,
        all_ten_route_plan_document_sha256: String,
        all_ten_route_plan: Value,
        first_residual: Value,
        router: Value,
        routes: Vec<Value>,
        shared: Value,
        combine: Value,
    }

    impl RealMoeBoundaryFixture {
        fn inputs(&self) -> Qwen80RealMoeBoundaryInputs<'_> {
            Qwen80RealMoeBoundaryInputs {
                admission_receipt_seal_sha256: &self.admission_receipt_seal_sha256,
                manifest_document_sha256: &self.manifest_document_sha256,
                router_receipt_sha256: &self.router_receipt_sha256,
                router_outer_receipt_sha256: &self.router_outer_receipt_sha256,
                router_outer_receipt_seal_sha256: &self.router_outer_receipt_seal_sha256,
                all_ten_route_plan_document_sha256: &self.all_ten_route_plan_document_sha256,
                all_ten_route_plan: &self.all_ten_route_plan,
                first_residual_receipt: &self.first_residual,
                router_receipt: &self.router,
                routed_expert_receipts: &self.routes,
                shared_expert_receipt: &self.shared,
                combine_receipt: &self.combine,
            }
        }
    }

    fn real_moe_vector(sha256: &str) -> Value {
        json!({
            "elements": QWEN80_HIDDEN,
            "sha256": sha256,
            "all_finite": true,
        })
    }

    fn real_moe_metal(intermediates: Value) -> Value {
        json!({
            "performed": true,
            "strict_math": true,
            "timing_or_benchmarking_performed": false,
            "command_buffers": 1,
            "compute_dispatches": 3,
            "device_intermediates": intermediates,
        })
    }

    fn real_moe_artifact_binding(
        plan: &Qwen80CompleteHybridDecoderPlan,
        admission_receipt_seal_sha256: &str,
        manifest_document_sha256: &str,
    ) -> Value {
        let layer = &plan.layers[0];
        json!({
            "manifest_seal_sha256": plan.manifest_seal_sha256,
            "manifest_document_sha256": manifest_document_sha256,
            "source_revision": plan.source_revision,
            "admission_receipt_seal_sha256": admission_receipt_seal_sha256,
            "layer": layer.layer,
            "layer_kind": layer.kind.as_source_name(),
            "hidden": QWEN80_HIDDEN,
            "experts": QWEN80_EXPERTS,
            "experts_per_token": QWEN80_TOP_K,
        })
    }

    fn real_moe_boundary_fixture(plan: &Qwen80CompleteHybridDecoderPlan) -> RealMoeBoundaryFixture {
        let layer = &plan.layers[0];
        let admission_receipt_seal_sha256 = "f".repeat(64);
        let manifest_document_sha256 = "a".repeat(64);
        let router_receipt_sha256 = "e".repeat(64);
        let router_outer_receipt_sha256 = "c".repeat(64);
        let router_outer_receipt_seal_sha256 = "d".repeat(64);
        let all_ten_route_plan_document_sha256 = "b".repeat(64);
        let first_residual_sha256 = "1".repeat(64);
        let normalized_hidden_sha256 = "2".repeat(64);
        let shared_sha256 = "3".repeat(64);
        let second_residual_sha256 = "4".repeat(64);
        let route = Qwen80RouteSelection {
            ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            weights: [0.1; QWEN80_TOP_K],
        };
        let route_hashes = (0..QWEN80_TOP_K)
            .map(|index| format!("{:064x}", index + 16))
            .collect::<Vec<_>>();
        let route_plan_waves = route
            .ids
            .iter()
            .zip(route.weights)
            .enumerate()
            .map(|(wave_index, (&expert, normalized_weight))| {
                let prefix = format!("model.layers.0.mlp.experts.{expert}");
                let projection = |suffix: &str, shape: &[usize], offset: usize| {
                    json!({
                        "tensor_name": format!("{prefix}.{suffix}.weight"),
                        "shape": shape,
                        "elements": QWEN80_HIDDEN * QWEN80_MOE_INTERMEDIATE,
                        "artifact_path": format!("/fixture/{wave_index}-{suffix}.hq30g"),
                        "artifact_bytes": 1,
                        "artifact_sha256": format!("{:064x}", 1_000 + wave_index * 3 + offset),
                        "source_dtype": "BF16",
                        "source_shard": "model-00001-of-00040.safetensors",
                        "source_shard_sha256": "9".repeat(64),
                        "layout": {
                            "magic": "HQ30G1B1",
                            "group_size": QWEN80_GROUP_SIZE,
                            "scale_dtype": "float16",
                            "sign_bit_order": "little",
                            "version": 1,
                        },
                        "payload_opened_by_this_plan": false,
                    })
                };
                json!({
                    "wave_index": wave_index,
                    "layer": 0,
                    "expert_id": usize::from(expert),
                    "normalized_weight": normalized_weight,
                    "normalized_weight_bits_hex": format!("0x{:016x}", (normalized_weight as f64).to_bits()),
                    "gate": projection("gate_proj", &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN], 0),
                    "up": projection("up_proj", &[QWEN80_MOE_INTERMEDIATE, QWEN80_HIDDEN], 1),
                    "down": projection("down_proj", &[QWEN80_HIDDEN, QWEN80_MOE_INTERMEDIATE], 2),
                    "fixed_operation_order": [
                        "gate_proj [512,2048]",
                        "up_proj [512,2048]",
                        "SiLU(gate) * up [512]",
                        "down_proj [2048,512]",
                        "apply this route's source-normalized weight [2048]",
                    ],
                    "route_execution_status": "NOT_EXECUTED_SOURCE_BOUND_PLAN_ONLY",
                    "route_delta_materialized": false,
                    "route_weight_applied": false,
                })
            })
            .collect::<Vec<_>>();
        let all_ten_route_plan = json!({
            "schema": QWEN80_ALL_TEN_ROUTE_PLAN_SCHEMA,
            "status": QWEN80_ALL_TEN_ROUTE_PLAN_STATUS,
            "model_id": QWEN80_MODEL_ID,
            "model_key": "qwen80",
            "source_repository": QWEN80_REPOSITORY,
            "source_revision": plan.source_revision,
            "layer": 0,
            "router_evidence": {
                "outer_receipt_document_sha256": router_outer_receipt_sha256,
                "outer_receipt_seal_sha256": router_outer_receipt_seal_sha256,
                "inner_receipt_document_sha256": router_receipt_sha256,
                "source_stable_route_ids": route.ids.to_vec(),
                "source_stable_normalized_weights": route.weights.to_vec(),
                "source_router_component_only": true,
            },
            "manifest_descriptor_inventory": {
                "inventory_document_sha256": manifest_document_sha256,
                "manifest_schema": "hawking.ascension.qwen80_complete_binary_gravity.v1",
                "manifest_status": "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED",
                "manifest_seal_sha256": plan.manifest_seal_sha256,
                "source_repository": QWEN80_REPOSITORY,
                "declared_tensor_count": QWEN80_COMPLETE_BINARY_TENSORS,
                "received_descriptor_count": QWEN80_COMPLETE_BINARY_TENSORS,
                "resolved_route_tensor_count": QWEN80_TOP_K * 3,
                "payload_opened_by_this_plan": false,
            },
            "deterministic_waves": route_plan_waves,
            "rawls_real_all_ten_provenance_gate": {
                "schema": QWEN80_ALL_TEN_ROUTE_PLAN_GATE_SCHEMA,
                "all_ten_source_bindings_complete": true,
                "expected_layer": 0,
                "deterministic_wave_indices": (0..QWEN80_TOP_K).collect::<Vec<_>>(),
                "route_order": route.ids.to_vec(),
                "normalized_weights": route.weights.to_vec(),
                "execution_receipt_required_for_each_wave": true,
                "direct_packed_execution_required_for_each_wave": true,
                "source_bound_input_required_for_each_wave": true,
                "route_combine_receipt_required_separately": true,
                "shared_expert_receipt_required_separately": true,
                "first_and_second_residual_receipts_required_separately": true,
                "rejects_tensor_substitution": true,
                "rejects_route_reorder": true,
                "rejects_duplicate_experts": true,
                "rejects_missing_tensor_or_weight": true,
            },
            "route_execution_performed": false,
            "route_combine_performed": false,
            "shared_expert_performed": false,
            "residual_combine_performed": false,
            "metal_device_or_dispatch_performed": false,
            "model_execution_performed": false,
            "hcli_execution_performed": false,
            "tps_or_tg_measurement_performed": false,
            "complete_layer_or_decoder_claim_earned": false,
        });

        let first_residual = json!({
            "schema": QWEN80_FIRST_RESIDUAL_COMPONENT_SCHEMA,
            "status": QWEN80_FIRST_RESIDUAL_COMPONENT_STATUS,
            "mode": "metal",
            "metal_device_or_dispatch_performed": true,
            "first_residual_performed": true,
            "artifact_binding": real_moe_artifact_binding(plan, &admission_receipt_seal_sha256, &manifest_document_sha256),
            "first_residual_output": real_moe_vector(&first_residual_sha256),
            "metal_intermediate_error_ledger": real_moe_metal(json!({})),
        });

        let mut router_binding = real_moe_artifact_binding(
            plan,
            &admission_receipt_seal_sha256,
            &manifest_document_sha256,
        );
        router_binding["post_attention_norm"] =
            json!({"name": layer.post_attention_layernorm.name});
        router_binding["router_gate"] = json!({"name": layer.moe.router.name});
        let router = json!({
            "schema": QWEN80_REAL_MOE_ROUTER_COMPONENT_SCHEMA,
            "status": QWEN80_REAL_MOE_ROUTER_COMPONENT_STATUS,
            "mode": "metal",
            "metal_device_or_dispatch_performed": true,
            "artifact_binding": router_binding,
            "input_provenance": {
                "first_residual_output_sha256": first_residual_sha256,
            },
            "metal_intermediate_error_ledger": real_moe_metal(json!({
                "normalized_hidden": real_moe_vector(&normalized_hidden_sha256),
                "route_ids": route.ids.to_vec(),
                "renormalized_route_weights": route.weights.to_vec(),
            })),
        });

        let routes = route
            .ids
            .iter()
            .zip(route.weights)
            .enumerate()
            .map(|(route_index, (&expert, weight))| {
                let mut binding = real_moe_artifact_binding(
                    plan,
                    &admission_receipt_seal_sha256,
                    &manifest_document_sha256,
                );
                binding["selected_expert"] = json!(usize::from(expert));
                binding["moe_intermediate"] = json!(QWEN80_MOE_INTERMEDIATE);
                binding["post_attention_norm"] =
                    json!({"name": layer.post_attention_layernorm.name});
                let planned_wave = &route_plan_waves[route_index];
                let plan_projection = |field: &str| {
                    let projection = planned_wave
                        .get(field)
                        .and_then(Value::as_object)
                        .expect("fixture all-ten projection exists");
                    json!({
                        "name": projection
                            .get("tensor_name")
                            .and_then(Value::as_str)
                            .expect("fixture tensor name exists"),
                        "artifact_sha256": projection
                            .get("artifact_sha256")
                            .and_then(Value::as_str)
                            .expect("fixture artifact SHA exists"),
                    })
                };
                binding["expert_gate_proj"] = plan_projection("gate");
                binding["expert_up_proj"] = plan_projection("up");
                binding["expert_down_proj"] = plan_projection("down");
                json!({
                    "schema": QWEN80_REAL_ROUTED_EXPERT_COMPONENT_SCHEMA,
                    "status": QWEN80_REAL_ROUTED_EXPERT_COMPONENT_STATUS,
                    "mode": "metal",
                    "metal_device_or_dispatch_performed": true,
                    "artifact_binding": binding,
                    "route_evidence": {
                        "selected_route_index": route_index,
                        "selected_expert": usize::from(expert),
                        "selected_normalized_weight_f32": weight,
                        "source_top10_ids": route.ids.to_vec(),
                        "source_top10_renormalized_weights": route.weights.to_vec(),
                        "router_receipt_sha256": router_receipt_sha256,
                        "router_outer_receipt_seal_sha256": router_outer_receipt_seal_sha256,
                        "all_ten_route_plan_document_sha256": all_ten_route_plan_document_sha256,
                        "first_residual_output_sha256": first_residual_sha256,
                        "router_normalized_hidden_sha256": normalized_hidden_sha256,
                    },
                    "metal_intermediate_error_ledger": real_moe_metal(json!({
                        "weighted_one_route_delta": real_moe_vector(&route_hashes[route_index]),
                    })),
                })
            })
            .collect::<Vec<_>>();

        let mut shared_binding = real_moe_artifact_binding(
            plan,
            &admission_receipt_seal_sha256,
            &manifest_document_sha256,
        );
        shared_binding["shared_expert_intermediate"] = json!(QWEN80_SHARED_EXPERT_INTERMEDIATE);
        shared_binding["post_attention_norm"] =
            json!({"name": layer.post_attention_layernorm.name});
        shared_binding["shared_gate_proj"] = json!({"name": layer.moe.shared_gate_proj.name});
        shared_binding["shared_up_proj"] = json!({"name": layer.moe.shared_up_proj.name});
        shared_binding["shared_down_proj"] = json!({"name": layer.moe.shared_down_proj.name});
        shared_binding["shared_expert_gate"] = json!({"name": layer.moe.shared_expert_gate.name});
        let shared = json!({
            "schema": QWEN80_REAL_SHARED_EXPERT_COMPONENT_SCHEMA,
            "status": QWEN80_REAL_SHARED_EXPERT_COMPONENT_STATUS,
            "mode": "metal",
            "metal_device_or_dispatch_performed": true,
            "shared_expert_only": true,
            "artifact_binding": shared_binding,
            "input_provenance": {
                "first_residual_output_sha256": first_residual_sha256,
                "router_normalized_hidden_sha256": normalized_hidden_sha256,
            },
            "metal_intermediate_error_ledger": real_moe_metal(json!({
                "gated_shared": real_moe_vector(&shared_sha256),
            })),
        });

        let combine = json!({
            "schema": QWEN80_REAL_MOE_COMBINE_COMPONENT_SCHEMA,
            "status": QWEN80_REAL_MOE_COMBINE_COMPONENT_STATUS,
            "mode": "metal",
            "metal_device_or_dispatch_performed": true,
            "materialized_source_route_shaped_fixture_only": false,
            "routed_expert_aggregation_performed": true,
            "shared_expert_add_performed": true,
            "second_residual_performed": true,
            "artifact_binding": real_moe_artifact_binding(plan, &admission_receipt_seal_sha256, &manifest_document_sha256),
            "combine_inputs": {
                "source_top10_ids": route.ids.to_vec(),
                "source_top10_renormalized_weights": route.weights.to_vec(),
                "first_residual_output_sha256": first_residual_sha256,
                "router_normalized_hidden_sha256": normalized_hidden_sha256,
                "routed_weighted_delta_sha256": route_hashes,
                "gated_shared_sha256": shared_sha256,
                "all_ten_route_plan_document_sha256": all_ten_route_plan_document_sha256,
            },
            "metal_intermediate_error_ledger": real_moe_metal(json!({
                "second_residual": real_moe_vector(&second_residual_sha256),
            })),
        });

        RealMoeBoundaryFixture {
            admission_receipt_seal_sha256,
            manifest_document_sha256,
            router_receipt_sha256,
            router_outer_receipt_sha256,
            router_outer_receipt_seal_sha256,
            all_ten_route_plan_document_sha256,
            all_ten_route_plan,
            first_residual,
            router,
            routes,
            shared,
            combine,
        }
    }

    /// Deterministic scalar oracle for exercising the bridge/runner control
    /// contract. It deliberately derives synthetic coefficients from admitted
    /// tensor *addresses* rather than reading payloads, so it cannot be
    /// mistaken for a Qwen80 source-model fallback or numerical parity result.
    struct DeterministicCpuOracleExecutor {
        layer_zero_route: Qwen80RouteSelection,
        max_attention_positions: usize,
        hidden: f64,
        mixer: f64,
        routed_moe: f64,
        shared_moe: f64,
        final_logit: f64,
        linear_state: Vec<f64>,
        attention_history: Vec<[f64; 2]>,
        delta_components: usize,
        attention_components: usize,
        routed_experts: usize,
        shared_experts: usize,
        tail_mask_start: Option<u32>,
    }

    impl DeterministicCpuOracleExecutor {
        fn new(layer_zero_route: Qwen80RouteSelection, max_attention_positions: usize) -> Self {
            Self {
                layer_zero_route,
                max_attention_positions,
                hidden: 0.0,
                mixer: 0.0,
                routed_moe: 0.0,
                shared_moe: 0.0,
                final_logit: 0.0,
                linear_state: vec![0.0; 36],
                attention_history: vec![[0.0; 2]; 12],
                delta_components: 0,
                attention_components: 0,
                routed_experts: 0,
                shared_experts: 0,
                tail_mask_start: None,
            }
        }

        fn binding_seed(binding: &Qwen80PackedTensorBinding) -> f64 {
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            for byte in binding.name.bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x1000_0000_01b3);
            }
            for dimension in &binding.shape {
                hash ^= *dimension as u64;
                hash = hash.wrapping_mul(0x1000_0000_01b3);
            }
            (hash % 1009) as f64 / 1009.0
        }

        fn residual_rms(value: f64, binding: &Qwen80PackedTensorBinding) -> f64 {
            let residual_scale = 1.0 + (Self::binding_seed(binding) - 0.5) * 0.1;
            value / (value * value + f64::from(QWEN80_RMS_EPS)).sqrt() * residual_scale
        }

        fn sigmoid(value: f64) -> f64 {
            1.0 / (1.0 + (-value).exp())
        }

        fn silu(value: f64) -> f64 {
            value * Self::sigmoid(value)
        }

        fn require_finite(&self, stage: &str) -> Result<()> {
            if !self.hidden.is_finite()
                || !self.mixer.is_finite()
                || !self.routed_moe.is_finite()
                || !self.shared_moe.is_finite()
            {
                return Err(model_error(format!(
                    "deterministic hybrid CPU control oracle became non-finite at {stage}"
                )));
            }
            Ok(())
        }

        fn fingerprint(&self) -> (u64, u64, u64) {
            let state_sum = self
                .linear_state
                .iter()
                .chain(
                    self.attention_history
                        .iter()
                        .flat_map(|values| values.iter()),
                )
                .sum::<f64>();
            (
                self.hidden.to_bits(),
                self.final_logit.to_bits(),
                state_sum.to_bits(),
            )
        }
    }

    impl Qwen80HybridPerLayerComponentExecutor for DeterministicCpuOracleExecutor {
        fn begin_token(
            &mut self,
            token_id: u32,
            position: usize,
            embedding: &Qwen80PackedTensorBinding,
        ) -> Result<()> {
            self.hidden = f64::from(token_id) * 0.000_001
                + position as f64 * 0.000_01
                + Self::binding_seed(embedding) * 0.01;
            self.mixer = 0.0;
            self.routed_moe = 0.0;
            self.shared_moe = 0.0;
            self.require_finite("begin_token")
        }

        fn input_rms_norm(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
            self.hidden = Self::residual_rms(self.hidden, &layer.input_layernorm);
            self.require_finite("input_rms_norm")
        }

        fn linear_deltanet_component(
            &mut self,
            layer: &Qwen80HybridDecoderLayerPlan,
            state_slot: usize,
            mixer: &Qwen80LinearDeltaNetLayerBindings,
        ) -> Result<()> {
            if state_slot >= self.linear_state.len()
                || mixer.in_proj_qkvz.shape != [12_288, 2_048]
                || mixer.causal_conv1d.shape != [8_192, 1, 4]
            {
                return Err(model_error(
                    "deterministic DeltaNet bridge geometry drifted",
                ));
            }
            let controls = Self::binding_seed(&mixer.in_proj_qkvz)
                + Self::binding_seed(&mixer.in_proj_ba)
                + Self::binding_seed(&mixer.causal_conv1d)
                + Self::binding_seed(&mixer.a_log)
                + Self::binding_seed(&mixer.dt_bias)
                + Self::binding_seed(&mixer.gated_rms_norm);
            let state = &mut self.linear_state[state_slot];
            *state = *state * (0.70 + controls * 0.01) + self.hidden * (0.15 + controls * 0.02);
            self.mixer = Self::silu(*state + self.hidden * 0.5)
                * (0.9 + Self::binding_seed(&mixer.out_proj) * 0.1);
            self.delta_components = self.delta_components.saturating_add(1);
            if layer.layer == 0 && state_slot != 0 {
                return Err(model_error(
                    "layer-0 DeltaNet component did not use state slot 0",
                ));
            }
            self.require_finite("linear_deltanet_component")
        }

        fn full_attention_component(
            &mut self,
            layer: &Qwen80HybridDecoderLayerPlan,
            state_slot: usize,
            mixer: &Qwen80FullAttentionLayerBindings,
            position: usize,
        ) -> Result<()> {
            if state_slot >= self.attention_history.len()
                || position >= self.max_attention_positions
                || mixer.q_proj.shape != [8_192, 2_048]
                || mixer.k_proj.shape != [512, 2_048]
                || mixer.v_proj.shape != [512, 2_048]
                || mixer.o_proj.shape != [2_048, 4_096]
                || mixer.q_norm.shape != [256]
                || mixer.k_norm.shape != [256]
            {
                return Err(model_error(
                    "deterministic GQA bridge geometry/cache drifted",
                ));
            }
            let query = Self::residual_rms(self.hidden, &mixer.q_norm);
            let key = Self::residual_rms(
                self.hidden * (0.8 + Self::binding_seed(&mixer.k_proj) * 0.1),
                &mixer.k_norm,
            );
            let value = self.hidden * (0.7 + Self::binding_seed(&mixer.v_proj) * 0.2);
            self.attention_history[state_slot][position] = key + value;
            let causal_sum = self.attention_history[state_slot][..=position]
                .iter()
                .sum::<f64>();
            let causal_mean = causal_sum / (position + 1) as f64;
            let gate = Self::sigmoid(query + Self::binding_seed(&mixer.q_proj));
            self.mixer =
                query * causal_mean * gate * (0.6 + Self::binding_seed(&mixer.o_proj) * 0.2);
            self.attention_components = self.attention_components.saturating_add(1);
            if layer.layer == 3 && state_slot != 0 {
                return Err(model_error(
                    "layer-3 GQA component did not use KV state slot 0",
                ));
            }
            self.require_finite("full_attention_component")
        }

        fn add_mixer_residual(&mut self, _layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
            self.hidden += self.mixer;
            self.require_finite("add_mixer_residual")
        }

        fn post_attention_rms_norm(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
            self.hidden = Self::residual_rms(self.hidden, &layer.post_attention_layernorm);
            self.require_finite("post_attention_rms_norm")
        }

        fn route_top10(
            &mut self,
            layer: &Qwen80HybridDecoderLayerPlan,
        ) -> Result<Qwen80RouteSelection> {
            if layer.layer == 0 {
                return Ok(self.layer_zero_route.clone());
            }
            let mut ids = [0u16; QWEN80_TOP_K];
            for (index, id) in ids.iter_mut().enumerate() {
                *id = ((layer.layer * QWEN80_TOP_K + index) % QWEN80_EXPERTS) as u16;
            }
            Ok(Qwen80RouteSelection {
                ids,
                weights: [0.1; QWEN80_TOP_K],
            })
        }

        fn routed_expert(
            &mut self,
            _layer: &Qwen80HybridDecoderLayerPlan,
            _route_index: usize,
            route_weight: f32,
            expert: &Qwen80ExpertBindings,
        ) -> Result<()> {
            let activation =
                Self::silu(self.hidden * (0.9 + Self::binding_seed(&expert.gate_proj) * 0.1))
                    * (0.8 + Self::binding_seed(&expert.up_proj) * 0.2);
            self.routed_moe += f64::from(route_weight)
                * activation
                * (0.9 + Self::binding_seed(&expert.down_proj) * 0.1);
            self.routed_experts = self.routed_experts.saturating_add(1);
            self.require_finite("routed_expert")
        }

        fn shared_expert(&mut self, layer: &Qwen80HybridDecoderLayerPlan) -> Result<()> {
            let gate = Self::sigmoid(
                self.hidden * (0.8 + Self::binding_seed(&layer.moe.shared_expert_gate) * 0.2),
            );
            let activation = Self::silu(
                self.hidden * (0.9 + Self::binding_seed(&layer.moe.shared_gate_proj) * 0.1),
            ) * (0.8 + Self::binding_seed(&layer.moe.shared_up_proj) * 0.2);
            self.shared_moe =
                gate * activation * (0.9 + Self::binding_seed(&layer.moe.shared_down_proj) * 0.1);
            self.shared_experts = self.shared_experts.saturating_add(1);
            self.require_finite("shared_expert")
        }

        fn combine_moe_and_add_residual(
            &mut self,
            _layer: &Qwen80HybridDecoderLayerPlan,
        ) -> Result<()> {
            self.hidden += self.routed_moe + self.shared_moe;
            self.routed_moe = 0.0;
            self.shared_moe = 0.0;
            self.require_finite("combine_moe_and_add_residual")
        }

        fn final_rms_norm(&mut self, final_norm: &Qwen80PackedTensorBinding) -> Result<()> {
            self.hidden = Self::residual_rms(self.hidden, final_norm);
            self.require_finite("final_rms_norm")
        }

        fn lm_head(&mut self, lm_head: &Qwen80PackedTensorBinding) -> Result<()> {
            self.final_logit = self.hidden * (0.8 + Self::binding_seed(lm_head) * 0.2);
            self.require_finite("lm_head")
        }

        fn mask_reserved_lm_head_tail(&mut self, first_reserved_id: u32) -> Result<()> {
            self.tail_mask_start = Some(first_reserved_id);
            Ok(())
        }

        fn sample_token(&mut self, tokenizer_vocab_size: usize) -> Result<u32> {
            if self.tail_mask_start != Some(tokenizer_vocab_size as u32) {
                return Err(model_error(
                    "deterministic oracle sampled before the reserved lm_head tail was masked",
                ));
            }
            let scaled = (self.final_logit.abs() * 1_000_000.0).round();
            if !scaled.is_finite() || scaled < 0.0 {
                return Err(model_error(
                    "deterministic oracle final logit cannot be sampled",
                ));
            }
            Ok((scaled as u64 % tokenizer_vocab_size as u64) as u32)
        }
    }

    #[test]
    fn real_moe_boundary_contract_accepts_only_a_complete_vector_bound_top10_wave() {
        let temp = tempfile::tempdir().unwrap();
        let catalog =
            Qwen80CompleteArtifactCatalog::from_admitted(already_admitted_catalog_fixture(&temp))
                .unwrap();
        let plan = catalog.complete_hybrid_decoder_plan(2).unwrap();
        let fixture = real_moe_boundary_fixture(&plan);

        let proof = plan
            .require_real_moe_boundary(0, &fixture.inputs())
            .unwrap();
        assert_eq!(proof.layer, 0);
        assert_eq!(proof.manifest_document_sha256, "a".repeat(64));
        assert_eq!(proof.all_ten_route_plan_document_sha256, "b".repeat(64));
        assert_eq!(proof.route.ids, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(proof.routed_weighted_delta_sha256.len(), QWEN80_TOP_K);
        assert_eq!(proof.first_residual_sha256, "1".repeat(64));
        assert_eq!(
            proof.post_attention_normalized_hidden_sha256,
            "2".repeat(64)
        );
        assert_eq!(proof.gated_shared_sha256, "3".repeat(64));
        assert_eq!(proof.second_residual_sha256, "4".repeat(64));
        assert!(
            !plan.has_complete_native_operator_backend(),
            "a single source-bound MoE boundary must not promote the incomplete 48-layer backend"
        );
    }

    #[test]
    fn real_moe_boundary_contract_refuses_one_route_and_materialized_fixture_evidence() {
        let temp = tempfile::tempdir().unwrap();
        let catalog =
            Qwen80CompleteArtifactCatalog::from_admitted(already_admitted_catalog_fixture(&temp))
                .unwrap();
        let plan = catalog.complete_hybrid_decoder_plan(2).unwrap();

        let mut one_route = real_moe_boundary_fixture(&plan);
        one_route.routes.truncate(1);
        let error = plan
            .require_real_moe_boundary(0, &one_route.inputs())
            .expect_err("the existing expert-65-only shape must not form a MoE boundary")
            .to_string();
        assert!(error.contains("exactly 10 routed expert receipts"));

        let mut old_one_route_status = real_moe_boundary_fixture(&plan);
        old_one_route_status.routes[0]["status"] = json!(
            "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_ONE_ROUTED_EXPERT_65_STRICT_MATH_METAL_COMPONENT_NOT_TEN_ROUTE_OR_LAYER"
        );
        let error = plan
            .require_real_moe_boundary(0, &old_one_route_status.inputs())
            .expect_err("the current one-route component status must remain unpromotable")
            .to_string();
        assert!(error.contains("ONE_ROUTED_EXPERT_65"));

        let mut materialized_fixture = real_moe_boundary_fixture(&plan);
        materialized_fixture.combine["materialized_source_route_shaped_fixture_only"] = json!(true);
        let error = plan
            .require_real_moe_boundary(0, &materialized_fixture.inputs())
            .expect_err("a materialized source-shaped fixture is not a physical combine")
            .to_string();
        assert!(error.contains("fixture"));
    }

    #[test]
    fn real_moe_boundary_contract_refuses_reordered_routes_and_cross_vector_combine() {
        let temp = tempfile::tempdir().unwrap();
        let catalog =
            Qwen80CompleteArtifactCatalog::from_admitted(already_admitted_catalog_fixture(&temp))
                .unwrap();
        let plan = catalog.complete_hybrid_decoder_plan(2).unwrap();

        let mut duplicate_route = real_moe_boundary_fixture(&plan);
        duplicate_route.routes[1]["route_evidence"]["selected_route_index"] = json!(0);
        let error = plan
            .require_real_moe_boundary(0, &duplicate_route.inputs())
            .expect_err("duplicate route index must be refused before combine")
            .to_string();
        assert!(error.contains("route index 1"));

        let mut wrong_shared_vector = real_moe_boundary_fixture(&plan);
        wrong_shared_vector.combine["combine_inputs"]["gated_shared_sha256"] =
            json!("0".repeat(64));
        let error = plan
            .require_real_moe_boundary(0, &wrong_shared_vector.inputs())
            .expect_err("combine cannot consume a shared vector from another component")
            .to_string();
        assert!(error.contains("gated_shared_sha256"));
    }

    #[test]
    fn real_moe_boundary_contract_requires_the_immutable_all_ten_plan_and_exact_bindings() {
        let temp = tempfile::tempdir().unwrap();
        let catalog =
            Qwen80CompleteArtifactCatalog::from_admitted(already_admitted_catalog_fixture(&temp))
                .unwrap();
        let plan = catalog.complete_hybrid_decoder_plan(2).unwrap();

        let mut substituted_tensor = real_moe_boundary_fixture(&plan);
        substituted_tensor.all_ten_route_plan["deterministic_waves"][0]["gate"]
            ["artifact_sha256"] = json!("0".repeat(64));
        let error = plan
            .require_real_moe_boundary(0, &substituted_tensor.inputs())
            .expect_err("a route plan cannot substitute a direct-packed tensor")
            .to_string();
        assert!(error.contains("direct-packed artifact binding"));

        let mut disconnected_plan = real_moe_boundary_fixture(&plan);
        disconnected_plan.routes[4]["route_evidence"]["all_ten_route_plan_document_sha256"] =
            json!("0".repeat(64));
        let error = plan
            .require_real_moe_boundary(0, &disconnected_plan.inputs())
            .expect_err("each executed route must echo the immutable all-ten plan SHA")
            .to_string();
        assert!(error.contains("all_ten_route_plan_document_sha256"));

        let mut non_descriptor_plan = real_moe_boundary_fixture(&plan);
        non_descriptor_plan.all_ten_route_plan["route_execution_performed"] = json!(true);
        let error = plan
            .require_real_moe_boundary(0, &non_descriptor_plan.inputs())
            .expect_err("a descriptor-only plan cannot claim route execution")
            .to_string();
        assert!(error.contains("descriptor-only"));

        let mut reordered_operations = real_moe_boundary_fixture(&plan);
        reordered_operations.all_ten_route_plan["deterministic_waves"][0]
            ["fixed_operation_order"][0] = json!("down_proj [2048,512]");
        let error = plan
            .require_real_moe_boundary(0, &reordered_operations.inputs())
            .expect_err("the all-ten descriptor must preserve the source expert operation order")
            .to_string();
        assert!(error.contains("operation order"));
    }

    #[test]
    fn artifact_bound_hybrid_plan_covers_every_layer_state_domain_and_terminal_head() {
        let temp = tempfile::tempdir().unwrap();
        let catalog =
            Qwen80CompleteArtifactCatalog::from_admitted(already_admitted_catalog_fixture(&temp))
                .unwrap();
        let plan = catalog.complete_hybrid_decoder_plan(8).unwrap();
        assert_eq!(plan.layers.len(), QWEN80_LAYERS);
        assert_eq!(plan.state.linear_layers, 36);
        assert_eq!(plan.state.full_attention_layers, 12);
        assert_eq!(plan.embedding.name, "model.embed_tokens.weight");
        assert_eq!(plan.final_norm.name, "model.norm.weight");
        assert_eq!(plan.lm_head.name, "lm_head.weight");
        assert_eq!(plan.tokenizer_vocab_size, QWEN80_TOKENIZER_VOCAB);
        assert_eq!(plan.reserved_lm_head_tail_rows, 267);
        assert!(!plan.has_complete_native_operator_backend());
        assert_eq!(plan.required_native_operator_gaps.len(), 9);
        assert_eq!(
            plan.required_native_operator_gaps[0].as_str(),
            "direct_packed_embedding_gather"
        );
        assert_eq!(
            plan.required_native_operator_gaps.last().unwrap().as_str(),
            "device_resident_autoregressive_state_and_feedback"
        );

        for layer in &plan.layers {
            match (layer.layer % 4, &layer.mixer) {
                (0 | 1 | 2, Qwen80HybridMixerBindings::LinearDeltaNet(mixer)) => {
                    assert_eq!(layer.linear_state_slot, Some(layer.layer - layer.layer / 4));
                    assert_eq!(layer.full_attention_state_slot, None);
                    assert_eq!(mixer.causal_conv1d.shape, vec![8192, 1, 4]);
                }
                (3, Qwen80HybridMixerBindings::FullAttention(mixer)) => {
                    assert_eq!(layer.linear_state_slot, None);
                    assert_eq!(layer.full_attention_state_slot, Some(layer.layer / 4));
                    assert_eq!(mixer.q_proj.shape, vec![8192, 2048]);
                }
                _ => panic!(
                    "layer {} was assigned the wrong Qwen3-Next mixer",
                    layer.layer
                ),
            }
            assert_eq!(layer.moe.router.shape, vec![512, 2048]);
            assert_eq!(layer.moe.shared_expert_gate.shape, vec![1, 2048]);
        }

        let route = Qwen80RouteSelection {
            ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            weights: [0.1; QWEN80_TOP_K],
        };
        let experts = plan.routed_expert_bindings(&catalog, 47, &route).unwrap();
        assert_eq!(experts.len(), QWEN80_TOP_K);
        assert_eq!(
            experts[9].down_proj.name,
            "model.layers.47.mlp.experts.9.down_proj.weight"
        );
    }

    #[test]
    fn hybrid_scheduler_runs_exact_all_layer_control_order_with_test_backend_only() {
        let temp = tempfile::tempdir().unwrap();
        let catalog =
            Qwen80CompleteArtifactCatalog::from_admitted(already_admitted_catalog_fixture(&temp))
                .unwrap();
        let mut decoder = Qwen80CompleteHybridDecoder::from_admitted_catalog(catalog, 3).unwrap();
        let mut backend = RecordingHybridBackend {
            sampled: 42,
            ..Default::default()
        };

        let first = decoder.execute_one(&mut backend, 7).unwrap();
        assert_eq!(first.position, 0);
        assert_eq!(first.sampled_token_id, 42);
        assert_eq!(first.linear_layers, 36);
        assert_eq!(first.full_attention_layers, 12);
        assert_eq!(first.routed_expert_calls, 480);
        assert_eq!(first.shared_expert_calls, 48);
        assert!(first.direct_packed_execution_is_not_proven_by_scheduler);
        assert_eq!(decoder.next_position(), 1);
        assert_eq!(backend.linear_slots.first(), Some(&(0, 0)));
        assert_eq!(backend.linear_slots.last(), Some(&(46, 35)));
        assert_eq!(backend.attention_slots.first(), Some(&(3, 0, 0)));
        assert_eq!(backend.attention_slots.last(), Some(&(47, 11, 0)));
        assert_eq!(backend.routed.len(), 480);
        assert_eq!(backend.routed[0], (0, 0, 0));
        assert_eq!(backend.routed[479], (47, 9, 479));
        assert_eq!(
            &backend.events[..18],
            [
                "begin:7:0",
                "input_norm:0",
                "linear:0:0",
                "mixer_residual:0",
                "post_norm:0",
                "route:0",
                "expert:0:0:0",
                "expert:0:1:1",
                "expert:0:2:2",
                "expert:0:3:3",
                "expert:0:4:4",
                "expert:0:5:5",
                "expert:0:6:6",
                "expert:0:7:7",
                "expert:0:8:8",
                "expert:0:9:9",
                "shared:0",
                "moe_residual:0",
            ]
        );
        assert_eq!(
            &backend.events[backend.events.len() - 4..],
            ["final_norm", "lm_head", "mask_tail:151669", "sample:42"]
        );

        let second = decoder
            .execute_one(&mut backend, first.sampled_token_id)
            .unwrap();
        assert_eq!(second.position, 1);
        assert_eq!(decoder.next_position(), 2);
        assert_eq!(backend.attention_slots[12], (3, 0, 1));
    }

    #[test]
    fn artifact_bound_per_layer_bridge_matches_the_complete_hybrid_control_reference() {
        let temp = tempfile::tempdir().unwrap();
        let catalog =
            Qwen80CompleteArtifactCatalog::from_admitted(already_admitted_catalog_fixture(&temp))
                .unwrap();
        let mut decoder = Qwen80CompleteHybridDecoder::from_admitted_catalog(catalog, 2).unwrap();
        let plan = decoder.plan().clone();
        let mut bridge = Qwen80ArtifactBoundPerLayerBackendBridge::new(
            &plan,
            RecordingHybridBackend {
                sampled: 42,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(bridge.manifest_seal_sha256(), "a".repeat(64));
        assert_eq!(
            bridge.source_revision(),
            "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb"
        );

        // A plausible but wrong DeltaNet state slot must be refused before it
        // reaches the component executor.
        let layer_zero = plan.layers[0].clone();
        let mixer = match &layer_zero.mixer {
            Qwen80HybridMixerBindings::LinearDeltaNet(mixer) => mixer.clone(),
            Qwen80HybridMixerBindings::FullAttention(_) => unreachable!(),
        };
        assert!(<Qwen80ArtifactBoundPerLayerBackendBridge<RecordingHybridBackend> as Qwen80PackedHybridDecodeBackend>::linear_deltanet_mixer(
            &mut bridge,
            &layer_zero,
            1,
            &mixer,
        )
        .is_err());
        assert!(bridge.executor.events.is_empty());

        let step = decoder.execute_one(&mut bridge, 7).unwrap();
        assert_eq!(step.linear_layers, 36);
        assert_eq!(step.full_attention_layers, 12);
        assert_eq!(step.routed_expert_calls, 480);
        assert_eq!(step.shared_expert_calls, 48);
        let recorded = bridge.into_inner();
        assert_eq!(
            recorded.events,
            expected_hybrid_control_events(7, 0, 42),
            "artifact-bound bridge must preserve the independently built source schedule/control trace"
        );
        assert_eq!(recorded.linear_slots.first(), Some(&(0, 0)));
        assert_eq!(recorded.attention_slots.first(), Some(&(3, 0, 0)));
        assert_eq!(recorded.routed.len(), 480);
    }

    #[test]
    fn bounded_stage_runner_consumes_existing_component_ledgers_with_cpu_oracle_only() {
        let temp_a = tempfile::tempdir().unwrap();
        let catalog_a =
            Qwen80CompleteArtifactCatalog::from_admitted(already_admitted_catalog_fixture(&temp_a))
                .unwrap();
        let plan_a = catalog_a.complete_hybrid_decoder_plan(2).unwrap();
        let linear_ledger_a = direct_linear_component_ledger(&plan_a);
        let attention_ledger_a = layer3_gqa_component_ledger(&plan_a);
        let evidence_a = Qwen80BoundedHybridComponentEvidence::from_component_ledgers(
            &plan_a,
            &linear_ledger_a,
            &attention_ledger_a,
        )
        .unwrap();
        assert_eq!(evidence_a.deltanet_layer, 0);
        assert_eq!(evidence_a.deltanet_selected_expert, 132);
        assert_eq!(evidence_a.deltanet_route.ids[0], 132);
        assert_eq!(evidence_a.attention_layer, 3);
        assert_eq!(evidence_a.attention_fixture_positions, vec![0, 1]);
        assert_eq!(evidence_a.attention_metal_dispatches, 14);

        let mut bad_attention = attention_ledger_a.clone();
        bad_attention["source_bound_layer"]["tensors"]["q_proj"] = json!("wrong.tensor");
        assert!(
            Qwen80BoundedHybridComponentEvidence::from_component_ledgers(
                &plan_a,
                &linear_ledger_a,
                &bad_attention,
            )
            .is_err()
        );

        let mut runner_a =
            Qwen80BoundedHybridStageRunner::from_admitted_catalog_with_component_ledgers(
                catalog_a,
                2,
                &linear_ledger_a,
                &attention_ledger_a,
                DeterministicCpuOracleExecutor::new(evidence_a.deltanet_route.clone(), 2),
            )
            .unwrap();
        let first_a = runner_a.execute_bounded_control_token(7).unwrap();
        let second_a = runner_a
            .execute_bounded_control_token(first_a.sampled_token_id)
            .unwrap();
        assert_eq!(first_a.position, 0);
        assert_eq!(second_a.position, 1);
        assert_eq!(first_a.linear_layers, 36);
        assert_eq!(first_a.full_attention_layers, 12);
        assert_eq!(first_a.routed_expert_calls, 480);
        assert_eq!(first_a.shared_expert_calls, 48);
        assert_eq!(runner_a.next_position(), 2);
        let oracle_a = runner_a.into_executor();
        assert_eq!(oracle_a.delta_components, 72);
        assert_eq!(oracle_a.attention_components, 24);
        assert_eq!(oracle_a.routed_experts, 960);
        assert_eq!(oracle_a.shared_experts, 96);
        assert_eq!(
            oracle_a.tail_mask_start,
            Some(QWEN80_TOKENIZER_VOCAB as u32)
        );

        // A second independent artifact-bound runner must reproduce the exact
        // deterministic CPU-oracle result, including the layer-0 receipt route
        // and two-token layer-3 cache control. This is not source-model or
        // Metal parity; it is regression parity for the bridge wiring.
        let temp_b = tempfile::tempdir().unwrap();
        let catalog_b =
            Qwen80CompleteArtifactCatalog::from_admitted(already_admitted_catalog_fixture(&temp_b))
                .unwrap();
        let plan_b = catalog_b.complete_hybrid_decoder_plan(2).unwrap();
        let linear_ledger_b = direct_linear_component_ledger(&plan_b);
        let attention_ledger_b = layer3_gqa_component_ledger(&plan_b);
        let evidence_b = Qwen80BoundedHybridComponentEvidence::from_component_ledgers(
            &plan_b,
            &linear_ledger_b,
            &attention_ledger_b,
        )
        .unwrap();
        let mut runner_b =
            Qwen80BoundedHybridStageRunner::from_admitted_catalog_with_component_ledgers(
                catalog_b,
                2,
                &linear_ledger_b,
                &attention_ledger_b,
                DeterministicCpuOracleExecutor::new(evidence_b.deltanet_route.clone(), 2),
            )
            .unwrap();
        let first_b = runner_b.execute_bounded_control_token(7).unwrap();
        let second_b = runner_b
            .execute_bounded_control_token(first_b.sampled_token_id)
            .unwrap();
        let oracle_b = runner_b.into_executor();
        assert_eq!(first_a, first_b);
        assert_eq!(second_a, second_b);
        assert_eq!(oracle_a.fingerprint(), oracle_b.fingerprint());
        assert_eq!(oracle_a.linear_state, oracle_b.linear_state);
        assert_eq!(oracle_a.attention_history, oracle_b.attention_history);
    }

    #[test]
    fn hybrid_scheduler_refuses_reserved_input_tail_and_invalid_routes_without_advancing_state() {
        let temp = tempfile::tempdir().unwrap();
        let catalog =
            Qwen80CompleteArtifactCatalog::from_admitted(already_admitted_catalog_fixture(&temp))
                .unwrap();
        let mut decoder = Qwen80CompleteHybridDecoder::from_admitted_catalog(catalog, 2).unwrap();
        let mut backend = RecordingHybridBackend {
            sampled: 42,
            ..Default::default()
        };
        assert!(decoder
            .execute_one(&mut backend, QWEN80_TOKENIZER_VOCAB as u32)
            .is_err());
        assert!(backend.events.is_empty());
        assert_eq!(decoder.next_position(), 0);

        backend.invalid_route = true;
        assert!(decoder.execute_one(&mut backend, 1).is_err());
        assert_eq!(decoder.next_position(), 0);
        assert!(backend.routed.is_empty());
        assert_eq!(backend.events[0], "begin:1:0");
        assert!(backend.events.iter().any(|event| event == "route:0"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn source_token_l0_l1_joint_trace_refuses_count_only_and_appended_kernels() {
        let l0 = QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS
            .iter()
            .map(|kernel| (*kernel).to_owned())
            .collect::<Vec<_>>();
        qwen80_require_exact_structural_kernel_trace(
            Some(&l0),
            &QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS,
            "test canonical L0 trace",
        )
        .expect("the canonical 23-kernel L0 trace must be accepted");

        let mut count_only = l0.clone();
        count_only[0] = "arbitrary_kernel_with_the_same_count".to_owned();
        assert!(qwen80_require_exact_structural_kernel_trace(
            Some(&count_only),
            &QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS,
            "test count-only L0 trace",
        )
        .is_err());

        let expected_joint = QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS
            .iter()
            .chain(QWEN80_SOURCE_TOKEN_L1_DELTA_PREFIX_KERNELS.iter())
            .copied()
            .collect::<Vec<_>>();
        let mut appended = expected_joint
            .iter()
            .map(|kernel| (*kernel).to_owned())
            .collect::<Vec<_>>();
        appended.push("qwen80_illegal_l1_suffix".to_owned());
        assert!(qwen80_require_exact_structural_kernel_trace(
            Some(&appended),
            &expected_joint,
            "test appended L1 suffix trace",
        )
        .is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn source_token_l1_public_encoder_accepts_only_the_opaque_l0_capability() {
        // This compile-checked signature guard intentionally has no Metal
        // context or invocation.  It prevents reintroducing a public API
        // which accepts a raw PinnedBuffer or caller-selected CPU shadow in
        // place of the canonical L0 custody capability.
        let _ = Qwen80CompleteNativeRuntime::encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into
            as fn(
                &Qwen80CompleteNativeRuntime,
                &mut TokenCommandBuffer<'_>,
                Qwen80CanonicalSourceTokenL0TrueMoeContinuation,
            ) -> Result<Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder>;
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn full_l1_finalizer_owner_identity_accepts_the_continuation_owner_and_refuses_another_runtime()
    {
        // This is the exact pure custody check used by the 23+9+14 full-L1
        // finalizer.  Keeping it CPU-only prevents a label-only identity
        // drift from reaching a Metal fence again.
        let host_owner = Qwen80SameRuntimeStateArenaOwnerIdentity {
            conv_state_buffer_identity_sha256: "host-runtime-conv".into(),
            recurrent_state_buffer_identity_sha256: "host-runtime-recurrent".into(),
        };
        host_owner
            .require_matches(
                &host_owner,
                "same-runtime full-L1 finalizer refuses a runtime other than the opaque continuation owner",
            )
            .expect("the continuation owner must reach the full-L1 finalizer guard");

        let other_runtime_owner = Qwen80SameRuntimeStateArenaOwnerIdentity {
            conv_state_buffer_identity_sha256: "other-runtime-conv".into(),
            recurrent_state_buffer_identity_sha256: "other-runtime-recurrent".into(),
        };
        let error = host_owner
            .require_matches(
                &other_runtime_owner,
                "same-runtime full-L1 finalizer refuses a runtime other than the opaque continuation owner",
            )
            .expect_err("a continuation may not be consumed by another runtime");
        assert!(error
            .to_string()
            .contains("same-runtime full-L1 finalizer refuses a runtime other than the opaque continuation owner"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn same_runtime_l1_input_parity_accepts_tolerated_distinct_cpu_device_hashes() {
        let cpu_shadow = vec![1.0f32, -2.0, 0.25];
        let tolerated_device_snapshot = vec![1.0005f32, -2.0, 0.25];
        let cpu_hash =
            qwen80_f32_vector_sha256(&cpu_shadow, "test same-runtime Layer-1 CPU input").unwrap();
        let (max_abs_error, device_hash) =
            Qwen80CompleteNativeRuntime::validate_same_runtime_l1_input_parity(
                &cpu_shadow,
                &tolerated_device_snapshot,
            )
            .expect("bounded CPU/Metal input divergence must remain valid");
        assert!(max_abs_error > 0.0 && max_abs_error <= 1.0e-3);
        assert_ne!(cpu_hash, device_hash);

        let outside_tolerance = vec![1.01f32, -2.0, 0.25];
        assert!(
            Qwen80CompleteNativeRuntime::validate_same_runtime_l1_input_parity(
                &cpu_shadow,
                &outside_tolerance,
            )
            .is_err()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ba_controls_keep_the_exact_key_head_b_then_a_source_order() {
        let mut ba = vec![0.0f32; QWEN80_LINEAR_VALUE_HEADS * 2];
        for key_head in 0..QWEN80_LINEAR_KEY_HEADS {
            let base = key_head * 4;
            // Source order is [b0, b1, a0, a1] per key head, not an
            // all-B/all-A flat vector. Use deliberately distinct values so a
            // superficially plausible reshape cannot pass this parity guard.
            ba[base] = -0.5 - key_head as f32 * 0.01;
            ba[base + 1] = 0.25 + key_head as f32 * 0.02;
            ba[base + 2] = -0.75 + key_head as f32 * 0.03;
            ba[base + 3] = 0.5 - key_head as f32 * 0.04;
        }
        let a_log = vec![0.0f32; QWEN80_LINEAR_VALUE_HEADS];
        let dt_bias = vec![0.0f32; QWEN80_LINEAR_VALUE_HEADS];
        let (decay, beta) =
            Qwen80CompleteNativeRuntime::source_ba_to_decay_beta(&ba, &a_log, &dt_bias)
                .expect("exact Qwen3-Next BA geometry must map");

        for key_head in 0..QWEN80_LINEAR_KEY_HEADS {
            let base = key_head * 4;
            for within_key_head in 0..2 {
                let value_head = key_head * 2 + within_key_head;
                let expected_beta = 1.0 / (1.0 + (-ba[base + within_key_head]).exp());
                let source_a = ba[base + 2 + within_key_head];
                let expected_decay = (-source_a.max(0.0) - (-source_a.abs()).exp().ln_1p()).exp();
                assert!((beta[value_head] - expected_beta).abs() < 1.0e-6);
                assert!((decay[value_head] - expected_decay).abs() < 1.0e-6);
            }
        }
    }
}
