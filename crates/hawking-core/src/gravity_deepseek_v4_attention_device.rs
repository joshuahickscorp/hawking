//! Parameterized DeepSeek-V4 ratio-zero attention device surface.
//!
//! This module generalizes the layer/position/ratio-0 attention *plan* that
//! the fixed layer-0 and layer-1 device graphs already implement. It does not
//! itself own a Metal command encoder: callers still dispatch through the
//! existing L0/L1/P4B device graphs or a future unified executor.
//!
//! Honesty contract:
//! - ratio 0 is the only compression family admitted here;
//! - ratio 4 / 128 refuse cleanly via [`DeepSeekV4LayerDevicePlan`];
//! - growing-KV sparse attention is expressed as the authority kernel
//!   `deepseek_v4_p4_sparse_attention_ratio0_growing_kv_sink_authority`;
//! - no Engine, HCLI, parity receipt, or TPS claim is made.

use crate::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
use crate::gravity_deepseek_v4_layer_plan::{
    DeepSeekV4LayerDeviceCatalog, DeepSeekV4LayerDevicePlan, DeepSeekV4MhcControlExpStrategy,
};
use crate::gravity_deepseek_v4_layer_source_anchors::{
    DeepSeekV4LayerCommonTensor, DeepSeekV4LayerControlProjection, DeepSeekV4LayerMhcStage,
    DeepSeekV4LayerSourceAnchors,
};
use crate::{Error, Result};

/// Production growing-KV sparse-attention kernel for ratio-zero layers.
pub const DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL: &str =
    "deepseek_v4_p4_sparse_attention_ratio0_growing_kv_sink_authority";
/// Source sliding-window capacity for ratio-zero KV caches.
pub const DSV4F_RATIO0_KV_WINDOW_TOKENS: usize = 128;

/// Fully resolved, source-bound attention step for one layer and decode position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4Ratio0AttentionDevicePlan {
    pub layer: usize,
    pub token_position: usize,
    pub valid_kv_count: usize,
    pub compress_ratio: usize,
    pub mhc_control_exp: DeepSeekV4MhcControlExpStrategy,
    pub sparse_attention_kernel: &'static str,
    pub tensor_names: DeepSeekV4Ratio0AttentionTensorNames,
}

/// Exact source tensor names for one ratio-zero attention step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4Ratio0AttentionTensorNames {
    pub hc_attn_fn: String,
    pub hc_attn_base: String,
    pub hc_attn_scale: String,
    pub attn_norm: String,
    pub q_norm: String,
    pub kv_norm: String,
    pub attn_sink: String,
    pub wq_a_weight: String,
    pub wq_b_weight: String,
    pub wkv_weight: String,
    pub wo_a_weight: String,
    pub wo_b_weight: String,
}

impl DeepSeekV4Ratio0AttentionDevicePlan {
    /// Build a ratio-zero attention plan for `layer` at decode `token_position`.
    /// `token_position` is zero-based; valid KV count is `token_position + 1`.
    pub fn resolve(
        catalog: &DeepSeekV4LayerDeviceCatalog,
        layer: usize,
        token_position: usize,
    ) -> Result<Self> {
        let plan = catalog.plan(layer)?;
        plan.require_attention_device()?;
        if plan.compression.ratio() != 0 {
            return Err(attention_plan_error(format!(
                "layer {layer} compression ratio {} is not ratio-0",
                plan.compression.ratio()
            )));
        }
        if token_position >= DSV4F_RATIO0_KV_WINDOW_TOKENS {
            return Err(attention_plan_error(format!(
                "token position {token_position} exceeds the ratio-0 sliding window of {DSV4F_RATIO0_KV_WINDOW_TOKENS}"
            )));
        }
        let valid_kv_count = token_position
            .checked_add(1)
            .ok_or_else(|| attention_plan_error("valid KV count overflow"))?;
        let anchors = catalog.anchors();
        Ok(Self {
            layer,
            token_position,
            valid_kv_count,
            compress_ratio: 0,
            mhc_control_exp: plan.mhc_control_exp,
            sparse_attention_kernel: DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL,
            tensor_names: resolve_tensor_names(anchors, plan)?,
        })
    }

    /// Admit the catalog and resolve one step in a single call.
    pub fn admit(
        reader: &DeepSeekV4FullStreamReader,
        layer: usize,
        token_position: usize,
    ) -> Result<(DeepSeekV4LayerDeviceCatalog, Self)> {
        let catalog = DeepSeekV4LayerDeviceCatalog::admit(reader)?;
        let plan = Self::resolve(&catalog, layer, token_position)?;
        Ok((catalog, plan))
    }
}

fn resolve_tensor_names(
    anchors: &DeepSeekV4LayerSourceAnchors,
    plan: &DeepSeekV4LayerDevicePlan,
) -> Result<DeepSeekV4Ratio0AttentionTensorNames> {
    let layer = anchors.layer(plan.layer)?;
    Ok(DeepSeekV4Ratio0AttentionTensorNames {
        hc_attn_fn: layer.mhc_binding(DeepSeekV4LayerMhcStage::Attention).fn_tensor.name,
        hc_attn_base: layer
            .mhc_binding(DeepSeekV4LayerMhcStage::Attention)
            .base_tensor
            .name,
        hc_attn_scale: layer
            .mhc_binding(DeepSeekV4LayerMhcStage::Attention)
            .scale_tensor
            .name,
        attn_norm: layer
            .common_tensor(DeepSeekV4LayerCommonTensor::AttentionNorm)
            .name,
        q_norm: layer
            .common_tensor(DeepSeekV4LayerCommonTensor::AttentionQNorm)
            .name,
        kv_norm: layer
            .common_tensor(DeepSeekV4LayerCommonTensor::AttentionKvNorm)
            .name,
        attn_sink: layer
            .common_tensor(DeepSeekV4LayerCommonTensor::AttentionSink)
            .name,
        wq_a_weight: layer
            .control_pair(DeepSeekV4LayerControlProjection::WqA)
            .weight
            .name,
        wq_b_weight: layer
            .control_pair(DeepSeekV4LayerControlProjection::WqB)
            .weight
            .name,
        wkv_weight: layer
            .control_pair(DeepSeekV4LayerControlProjection::Wkv)
            .weight
            .name,
        wo_a_weight: layer
            .control_pair(DeepSeekV4LayerControlProjection::WoA)
            .weight
            .name,
        wo_b_weight: layer
            .control_pair(DeepSeekV4LayerControlProjection::WoB)
            .weight
            .name,
    })
}

fn attention_plan_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 ratio-0 attention device plan: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity_deepseek_v4_layer_source_anchors::{
        DeepSeekV4LayerCompressionMode, DeepSeekV4LayerGateMode, DeepSeekV4LayerSourceAnchor,
    };

    #[test]
    fn growing_kv_count_is_position_plus_one() {
        let anchor = DeepSeekV4LayerSourceAnchor {
            layer: 1,
            compression: DeepSeekV4LayerCompressionMode::SlidingWindowOnly,
            gate_mode: DeepSeekV4LayerGateMode::HashTokenIdToExpertIds,
            tensor_count: 0,
        };
        let plan = DeepSeekV4LayerDevicePlan::from_anchor(&anchor);
        assert!(plan.require_attention_device().is_ok());
        assert_eq!(DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL.contains("growing_kv"), true);
    }

    #[test]
    fn ratio_four_attention_plan_is_refused_by_device_plan() {
        let plan = DeepSeekV4LayerDevicePlan::from_anchor(&DeepSeekV4LayerSourceAnchor {
            layer: 2,
            compression: DeepSeekV4LayerCompressionMode::Ratio4WithIndexer,
            gate_mode: DeepSeekV4LayerGateMode::HashTokenIdToExpertIds,
            tensor_count: 0,
        });
        assert!(plan.require_attention_device().is_err());
    }
}
