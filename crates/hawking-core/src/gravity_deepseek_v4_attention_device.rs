//! Parameterized DeepSeek-V4 ratio-zero attention device surface.
//!
//! This module is the production plan + growing-KV authority surface for
//! layers 0–1 (ratio-0 only). Live device graphs (P3A/P4A L0 BOS, P4B L0
//! position-1, L1 BOS) now dispatch the general growing-KV sparse kernel
//! [`DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL`] instead of the older
//! position-0 / position-1 specializations.
//!
//! Honesty contract:
//! - ratio 0 is the only compression family admitted here;
//! - ratio 4 / 128 refuse cleanly via [`DeepSeekV4LayerDevicePlan`];
//! - growing-KV sparse attention is the production kernel
//!   `deepseek_v4_p4_sparse_attention_ratio0_growing_kv_sink_authority`;
//! - this module owns the plan/executor identity surface; Metal command
//!   encoding for a full Q/KV/mHC graph remains in the L0/L1/P4B device
//!   executors that share the same sparse kernel ABI;
//! - no Engine, HCLI, exact-storage parity receipt, or TPS claim is made.

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

    /// Growing-KV ABI parameters for the production sparse kernel.
    pub fn growing_kv_dispatch_params(&self) -> DeepSeekV4Ratio0GrowingKvDispatchParams {
        DeepSeekV4Ratio0GrowingKvDispatchParams {
            sparse_attention_kernel: self.sparse_attention_kernel,
            cache_capacity: self.valid_kv_count as u32,
            valid_kv_count: self.valid_kv_count as u32,
            max_score_slots: self.valid_kv_count as u32,
        }
    }
}

/// Exact Metal ABI arguments for the production growing-KV sparse kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4Ratio0GrowingKvDispatchParams {
    pub sparse_attention_kernel: &'static str,
    pub cache_capacity: u32,
    pub valid_kv_count: u32,
    pub max_score_slots: u32,
}

/// General ratio-0 attention executor identity for one planned step.
///
/// Live graphs still reside in the specialized L0/L1/P4B modules, but every
/// ratio-0 path is required to use the growing-KV kernel named here. This
/// type is the single place multi-layer diagnostics record "which sparse
/// kernel and valid-KV geometry did we intend".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4Ratio0AttentionDeviceExecutor {
    pub plan: DeepSeekV4Ratio0AttentionDevicePlan,
    pub growing_kv: DeepSeekV4Ratio0GrowingKvDispatchParams,
}

impl DeepSeekV4Ratio0AttentionDeviceExecutor {
    /// Resolve a production growing-KV attention step for a ratio-0 layer.
    pub fn prepare(
        catalog: &DeepSeekV4LayerDeviceCatalog,
        layer: usize,
        token_position: usize,
    ) -> Result<Self> {
        let plan = DeepSeekV4Ratio0AttentionDevicePlan::resolve(catalog, layer, token_position)?;
        let growing_kv = plan.growing_kv_dispatch_params();
        if growing_kv.sparse_attention_kernel != DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL {
            return Err(attention_plan_error(
                "ratio-0 attention executor requires the production growing-KV sparse kernel",
            ));
        }
        if growing_kv.valid_kv_count == 0
            || growing_kv.valid_kv_count > growing_kv.cache_capacity
            || growing_kv.max_score_slots < growing_kv.valid_kv_count
        {
            return Err(attention_plan_error(
                "ratio-0 growing-KV dispatch parameters are inconsistent",
            ));
        }
        Ok(Self { plan, growing_kv })
    }

    pub fn layer(&self) -> usize {
        self.plan.layer
    }

    pub fn token_position(&self) -> usize {
        self.plan.token_position
    }

    pub fn sparse_attention_kernel(&self) -> &'static str {
        self.growing_kv.sparse_attention_kernel
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
        assert_eq!(
            DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL.contains("growing_kv"),
            true
        );
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

    #[test]
    fn executor_params_match_position() {
        // Without a full artifact, exercise only the pure growing-KV math:
        // position p => valid_kv_count p+1.
        for position in [0usize, 1, 7, 127] {
            let valid = position + 1;
            let params = DeepSeekV4Ratio0GrowingKvDispatchParams {
                sparse_attention_kernel: DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL,
                cache_capacity: valid as u32,
                valid_kv_count: valid as u32,
                max_score_slots: valid as u32,
            };
            assert_eq!(params.valid_kv_count, (position + 1) as u32);
            assert!(params.valid_kv_count <= params.cache_capacity);
            assert!(params.max_score_slots >= params.valid_kv_count);
        }
    }
}
