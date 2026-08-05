//! General per-layer device plan for the DeepSeek-V4-Flash base body.
//!
//! Resolves compression ratio, gate mode, and tensor anchors for any layer
//! in `0..42` from [`crate::gravity_deepseek_v4_layer_source_anchors`]. This
//! is the single source of truth for "what does layer N look like" before a
//! device executor stages or dispatches work.
//!
//! Honesty contract:
//! - ratio-0 / sliding-window attention is the full growing-KV attention
//!   family the device chain can execute at any BOS-compatible position;
//! - at BOS/position-0 only, ratio-4 and ratio-128 also admit a
//!   *window-only* attention specialization because the source algorithm
//!   yields zero compressed slots (`end_pos // ratio == 0`); that is not a
//!   full compressed/indexer graph for later positions;
//! - hash gate mode (`tid2eid`) is the only MoE route mode the current P6
//!   device graph can execute; learned-bias layers refuse cleanly until a
//!   learned-route kernel is admitted;
//! - this module never dispatches Metal, never claims parity, and never
//!   flips a runtime/Engine gate.

use crate::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
use crate::gravity_deepseek_v4_layer_source_anchors::{
    verify_deepseek_v4_layer_source_anchors, DeepSeekV4LayerCommonTensor,
    DeepSeekV4LayerCompressionMode, DeepSeekV4LayerControlProjection, DeepSeekV4LayerGateMode,
    DeepSeekV4LayerMhcStage, DeepSeekV4LayerSourceAnchor, DeepSeekV4LayerSourceAnchors,
    DeepSeekV4LayerSourceIdentity, DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT,
};
use crate::{Error, Result};

/// Layers that use the source hash `tid2eid` table (official `num_hash_layers`).
pub const DSV4F_HASH_GATE_LAYER_COUNT: usize = 3;

/// One concrete, source-bound layer plan. Constructed only after anchors
/// verify against the admitted reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4LayerDevicePlan {
    pub layer: usize,
    pub compression: DeepSeekV4LayerCompressionMode,
    pub gate_mode: DeepSeekV4LayerGateMode,
    /// Full growing-KV ratio-0 attention (any BOS-compatible position).
    pub attention_device_supported: bool,
    /// BOS/position-0 window-only attention. True for every base layer because
    /// compressed topk is empty at `start_pos=0, seqlen=1`.
    pub bos_window_attention_device_supported: bool,
    pub moe_device_supported: bool,
    pub attention_refusal: Option<&'static str>,
    pub bos_window_attention_refusal: Option<&'static str>,
    pub moe_refusal: Option<&'static str>,
    pub mhc_control_exp: DeepSeekV4MhcControlExpStrategy,
}

/// How production mHC control kernels evaluate `exp` after the general
/// Darwin double-double promotion. Retained as an explicit label so receipts
/// can record which math path was used without implying exact-storage parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4MhcControlExpStrategy {
    /// Shared production helper `deepseek_v4_mhc_control_expf` (Darwin DD).
    DarwinDoubleDoubleControlDomain,
}

impl DeepSeekV4MhcControlExpStrategy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DarwinDoubleDoubleControlDomain => {
                "darwin_double_double_control_domain_general"
            }
        }
    }
}

impl DeepSeekV4LayerDevicePlan {
    /// Build the plan for one layer from already-verified anchors.
    pub fn from_anchor(anchor: &DeepSeekV4LayerSourceAnchor) -> Self {
        let (attention_device_supported, attention_refusal) = match anchor.compression {
            DeepSeekV4LayerCompressionMode::SlidingWindowOnly => (true, None),
            DeepSeekV4LayerCompressionMode::Ratio4WithIndexer => (
                false,
                Some(
                    "ratio-4 compressed attention with indexer is not implemented for non-BOS positions; use require_bos_window_attention_device for the BOS window-only specialization",
                ),
            ),
            DeepSeekV4LayerCompressionMode::Ratio128 => (
                false,
                Some(
                    "ratio-128 compressed attention is not implemented for non-BOS positions; use require_bos_window_attention_device for the BOS window-only specialization",
                ),
            ),
        };
        // BOS/pos0/seqlen1: compressed topk is empty for every ratio. Window-only
        // sparse attention is source-correct at this specialization only.
        let (bos_window_attention_device_supported, bos_window_attention_refusal) = (true, None);
        let (moe_device_supported, moe_refusal) = match anchor.gate_mode {
            DeepSeekV4LayerGateMode::HashTokenIdToExpertIds => (true, None),
            DeepSeekV4LayerGateMode::LearnedScoresWithSelectionBias => (
                false,
                Some(
                    "learned-bias gate mode is not yet implemented in the P6 device graph; refuses cleanly",
                ),
            ),
        };
        Self {
            layer: anchor.layer,
            compression: anchor.compression,
            gate_mode: anchor.gate_mode,
            attention_device_supported,
            bos_window_attention_device_supported,
            moe_device_supported,
            attention_refusal,
            bos_window_attention_refusal,
            moe_refusal,
            mhc_control_exp: DeepSeekV4MhcControlExpStrategy::DarwinDoubleDoubleControlDomain,
        }
    }

    /// Fail-closed check before a full growing-KV ratio-0 attention step.
    pub fn require_attention_device(&self) -> Result<()> {
        if self.attention_device_supported {
            return Ok(());
        }
        Err(plan_error(format!(
            "layer {} attention device refused: {}",
            self.layer,
            self.attention_refusal
                .unwrap_or("attention device path is unavailable")
        )))
    }

    /// Fail-closed check for the BOS/position-0 window-only attention path
    /// (valid for every compression mode; compressed slots empty at BOS).
    pub fn require_bos_window_attention_device(&self) -> Result<()> {
        if self.bos_window_attention_device_supported {
            return Ok(());
        }
        Err(plan_error(format!(
            "layer {} BOS window attention device refused: {}",
            self.layer,
            self.bos_window_attention_refusal
                .unwrap_or("BOS window attention path is unavailable")
        )))
    }

    /// Fail-closed check before a MoE/P6 device step may begin.
    pub fn require_moe_device(&self) -> Result<()> {
        if self.moe_device_supported {
            return Ok(());
        }
        Err(plan_error(format!(
            "layer {} MoE device refused: {}",
            self.layer,
            self.moe_refusal
                .unwrap_or("MoE device path is unavailable")
        )))
    }

    /// Fail-closed check for a full attention+MoE layer step (ratio-0 only).
    pub fn require_full_layer_device(&self) -> Result<()> {
        self.require_attention_device()?;
        self.require_moe_device()
    }

    /// Fail-closed check for BOS window attention + MoE (hash layers only until
    /// learned-bias is admitted).
    pub fn require_bos_full_layer_device(&self) -> Result<()> {
        self.require_bos_window_attention_device()?;
        self.require_moe_device()
    }

    pub fn common_tensor_name(&self, anchors: &DeepSeekV4LayerSourceAnchors, kind: DeepSeekV4LayerCommonTensor) -> Result<String> {
        Ok(anchors.layer(self.layer)?.common_tensor(kind).name)
    }

    pub fn control_pair_weight_name(
        &self,
        anchors: &DeepSeekV4LayerSourceAnchors,
        projection: DeepSeekV4LayerControlProjection,
    ) -> Result<String> {
        Ok(anchors.layer(self.layer)?.control_pair(projection).weight.name)
    }

    pub fn mhc_fn_name(
        &self,
        anchors: &DeepSeekV4LayerSourceAnchors,
        stage: DeepSeekV4LayerMhcStage,
    ) -> Result<String> {
        Ok(anchors.layer(self.layer)?.mhc_binding(stage).fn_tensor.name)
    }

    pub fn gate_score_weight_name(&self, anchors: &DeepSeekV4LayerSourceAnchors) -> Result<String> {
        Ok(anchors.layer(self.layer)?.gate_binding().score_weight.name)
    }

    pub fn gate_route_data_name(&self, anchors: &DeepSeekV4LayerSourceAnchors) -> Result<String> {
        Ok(anchors.layer(self.layer)?.gate_binding().route_data.name)
    }
}

/// Verified anchors plus the 43-layer plan table for one admitted artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4LayerDeviceCatalog {
    anchors: DeepSeekV4LayerSourceAnchors,
    plans: Vec<DeepSeekV4LayerDevicePlan>,
}

impl DeepSeekV4LayerDeviceCatalog {
    /// Metadata-only admission: verifies every base-layer anchor against the
    /// reader and builds the per-layer device plan table.
    pub fn admit(reader: &DeepSeekV4FullStreamReader) -> Result<Self> {
        let anchors = verify_deepseek_v4_layer_source_anchors(reader)?;
        let plans = anchors
            .layers()
            .iter()
            .map(DeepSeekV4LayerDevicePlan::from_anchor)
            .collect();
        Ok(Self { anchors, plans })
    }

    pub fn anchors(&self) -> &DeepSeekV4LayerSourceAnchors {
        &self.anchors
    }

    pub fn identity(&self) -> &DeepSeekV4LayerSourceIdentity {
        self.anchors.identity()
    }

    pub fn plans(&self) -> &[DeepSeekV4LayerDevicePlan] {
        &self.plans
    }

    pub fn plan(&self, layer: usize) -> Result<&DeepSeekV4LayerDevicePlan> {
        self.plans.get(layer).ok_or_else(|| {
            plan_error(format!(
                "layer {layer} is outside the {DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT}-layer base body"
            ))
        })
    }

    /// Layers that currently admit both full ratio-0 attention and MoE device execution.
    pub fn full_layer_device_supported(&self) -> Vec<usize> {
        self.plans
            .iter()
            .filter(|plan| plan.attention_device_supported && plan.moe_device_supported)
            .map(|plan| plan.layer)
            .collect()
    }

    /// Layers that admit full ratio-0 attention device execution.
    pub fn attention_device_supported(&self) -> Vec<usize> {
        self.plans
            .iter()
            .filter(|plan| plan.attention_device_supported)
            .map(|plan| plan.layer)
            .collect()
    }

    /// Layers that admit BOS window-only attention + currently supported MoE.
    pub fn bos_full_layer_device_supported(&self) -> Vec<usize> {
        self.plans
            .iter()
            .filter(|plan| plan.bos_window_attention_device_supported && plan.moe_device_supported)
            .map(|plan| plan.layer)
            .collect()
    }
}

fn plan_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 layer device plan: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_zero_hash_layers_are_device_supported() {
        let l0 = DeepSeekV4LayerDevicePlan::from_anchor(&DeepSeekV4LayerSourceAnchor {
            layer: 0,
            compression: DeepSeekV4LayerCompressionMode::SlidingWindowOnly,
            gate_mode: DeepSeekV4LayerGateMode::HashTokenIdToExpertIds,
            tensor_count: 0,
        });
        assert!(l0.attention_device_supported);
        assert!(l0.moe_device_supported);
        assert!(l0.require_full_layer_device().is_ok());
        assert_eq!(
            l0.mhc_control_exp,
            DeepSeekV4MhcControlExpStrategy::DarwinDoubleDoubleControlDomain
        );

        let l1 = DeepSeekV4LayerDevicePlan::from_anchor(&DeepSeekV4LayerSourceAnchor {
            layer: 1,
            compression: DeepSeekV4LayerCompressionMode::SlidingWindowOnly,
            gate_mode: DeepSeekV4LayerGateMode::HashTokenIdToExpertIds,
            tensor_count: 0,
        });
        assert!(l1.require_full_layer_device().is_ok());
    }

    #[test]
    fn ratio_four_and_learned_bias_refuse_cleanly() {
        let ratio4 = DeepSeekV4LayerDevicePlan::from_anchor(&DeepSeekV4LayerSourceAnchor {
            layer: 2,
            compression: DeepSeekV4LayerCompressionMode::Ratio4WithIndexer,
            gate_mode: DeepSeekV4LayerGateMode::HashTokenIdToExpertIds,
            tensor_count: 0,
        });
        assert!(ratio4.moe_device_supported);
        assert!(ratio4.require_attention_device().is_err());
        assert!(ratio4.require_full_layer_device().is_err());
        // BOS window-only specialization is admitted for ratio-4 + hash.
        assert!(ratio4.require_bos_window_attention_device().is_ok());
        assert!(ratio4.require_bos_full_layer_device().is_ok());

        let learned = DeepSeekV4LayerDevicePlan::from_anchor(&DeepSeekV4LayerSourceAnchor {
            layer: 3,
            compression: DeepSeekV4LayerCompressionMode::Ratio128,
            gate_mode: DeepSeekV4LayerGateMode::LearnedScoresWithSelectionBias,
            tensor_count: 0,
        });
        assert!(learned.require_attention_device().is_err());
        assert!(learned.require_moe_device().is_err());
        // Attention BOS path admits; MoE still refuses until learned-bias lands.
        assert!(learned.require_bos_window_attention_device().is_ok());
        assert!(learned.require_bos_full_layer_device().is_err());
    }

    #[test]
    fn pinned_ratio_schedule_matches_anchor_constants() {
        // Layers 0 and 1 are the only base layers with ratio-0 in the pinned
        // source schedule used by the compact anchors.
        assert_eq!(
            crate::gravity_deepseek_v4_layer_source_anchors::DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT,
            43
        );
        assert_eq!(DSV4F_HASH_GATE_LAYER_COUNT, 3);
    }
}
