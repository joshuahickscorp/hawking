//! Executable, bounded source-preparation scheduler for one DeepSeek-V4 layer.
//!
//! This is deliberately below a causal runtime. Each call performs one real,
//! bounded source-stage operation through [`DeepSeekV4ExecutionContext`] and
//! immediately exposes the source-native payload to a caller-supplied sink.
//! The sink is the seam where a future Metal command encoder may bind/upload
//! buffers. No default sink, command buffer, GPU dispatch, forward, router,
//! KV write, sampling, Engine, HCLI, parity, or TPS claim exists here.

use crate::gravity_deepseek_v4_execution_context::{
    DeepSeekV4ControlLease, DeepSeekV4ControlPayload, DeepSeekV4ExecutionContext,
    DeepSeekV4MhcBranch, DeepSeekV4SelectedRouteSet,
};
use crate::gravity_deepseek_v4_expert_cache::{
    resolve_expert_bundle, ExpertBundleKey, ExpertCacheAccess,
};
use crate::gravity_deepseek_v4_runtime_spine::{
    DeepSeekV4ControlProjection, DeepSeekV4ExpertProjection, DSV4F_BASE_LAYER_COUNT,
};
use crate::{Error, Result};

/// Exactly eleven source-staging steps feed a single generic base-layer
/// preparation. They are not the same thing as the nine logical operations in
/// the causal token graph: several native pairs feed one logical operation.
pub const DSV4F_LAYER_PREPARATION_STAGE_COUNT: usize = 11;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4LayerPreparationStage {
    MhcAttentionControl,
    AttentionControl(DeepSeekV4ControlProjection),
    MhcFfnControl,
    RoutedExpertWave,
    SharedExpertControl(DeepSeekV4ExpertProjection),
}

impl DeepSeekV4LayerPreparationStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MhcAttentionControl => "mhc_attention_control",
            Self::AttentionControl(DeepSeekV4ControlProjection::WqA) => "attention_wq_a",
            Self::AttentionControl(DeepSeekV4ControlProjection::WqB) => "attention_wq_b",
            Self::AttentionControl(DeepSeekV4ControlProjection::Wkv) => "attention_wkv",
            Self::AttentionControl(DeepSeekV4ControlProjection::WoA) => "attention_wo_a",
            Self::AttentionControl(DeepSeekV4ControlProjection::WoB) => "attention_wo_b",
            Self::MhcFfnControl => "mhc_ffn_control",
            Self::RoutedExpertWave => "routed_expert_wave",
            Self::SharedExpertControl(DeepSeekV4ExpertProjection::W1) => "shared_expert_w1",
            Self::SharedExpertControl(DeepSeekV4ExpertProjection::W2) => "shared_expert_w2",
            Self::SharedExpertControl(DeepSeekV4ExpertProjection::W3) => "shared_expert_w3",
        }
    }

    /// Index into the token graph's nine logical operations for this layer.
    /// The scheduler deliberately leaves computation-only graph nodes to a
    /// future native causal executor rather than synthesizing fake work.
    pub const fn logical_node_offset(self) -> usize {
        match self {
            Self::MhcAttentionControl => 0,
            Self::AttentionControl(DeepSeekV4ControlProjection::WqA)
            | Self::AttentionControl(DeepSeekV4ControlProjection::WqB) => 1,
            Self::AttentionControl(DeepSeekV4ControlProjection::Wkv) => 2,
            Self::AttentionControl(DeepSeekV4ControlProjection::WoA)
            | Self::AttentionControl(DeepSeekV4ControlProjection::WoB) => 3,
            Self::MhcFfnControl => 4,
            Self::RoutedExpertWave => 6,
            Self::SharedExpertControl(_) => 7,
        }
    }

    pub const fn is_control_stage(self) -> bool {
        !matches!(self, Self::RoutedExpertWave)
    }
}

/// Bounded source-load result of a single scheduler step. A control lease must
/// be consumed by a native sink before another control stage is requested,
/// because the context's FIFO arena may evict it at the next stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeepSeekV4LayerPreparationResult {
    ControlLease(DeepSeekV4ControlLease),
    RoutedExpertAccesses(Vec<ExpertCacheAccess>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4LayerPreparationStep {
    pub sequence: usize,
    pub token_position: usize,
    pub layer: usize,
    pub stage: DeepSeekV4LayerPreparationStage,
    pub logical_graph_node_ordinal: usize,
    pub result: DeepSeekV4LayerPreparationResult,
    /// The scheduler itself cannot submit GPU work. These remain zero until a
    /// concrete sink returns its completed device-consumption receipt.
    pub actual_command_buffers: usize,
    pub actual_compute_encoders: usize,
    pub actual_gpu_dispatches: usize,
    pub actual_cpu_visible_waits: usize,
}

impl DeepSeekV4LayerPreparationStep {
    pub const fn control_lease(&self) -> Option<DeepSeekV4ControlLease> {
        match self.result {
            DeepSeekV4LayerPreparationResult::ControlLease(lease) => Some(lease),
            DeepSeekV4LayerPreparationResult::RoutedExpertAccesses(_) => None,
        }
    }
}

/// Borrowed source-native staging input, valid only during the sink callback.
/// A Metal adapter can consume this immediately to allocate/bind its bounded
/// resources without retaining an unbounded reference to the context.
pub enum DeepSeekV4NativeStage<'a> {
    Control {
        step: &'a DeepSeekV4LayerPreparationStep,
        payload: &'a DeepSeekV4ControlPayload,
    },
    RoutedExpertWave {
        step: &'a DeepSeekV4LayerPreparationStep,
        route_set: DeepSeekV4SelectedRouteSet,
        context: &'a DeepSeekV4ExecutionContext,
        accesses: &'a [ExpertCacheAccess],
    },
}

/// Completed work reported by a synchronous native stage sink. A sink must
/// report only command buffers/dispatches it actually committed and waited on;
/// `Default` is the explicit source-preparation/no-device-work receipt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4NativeStageConsumption {
    pub actual_command_buffers: usize,
    pub actual_compute_encoders: usize,
    pub actual_gpu_dispatches: usize,
    pub actual_cpu_visible_waits: usize,
    pub host_intermediate_handoff_bytes: usize,
}

impl<'a> DeepSeekV4NativeStage<'a> {
    pub fn step(&self) -> &DeepSeekV4LayerPreparationStep {
        match self {
            Self::Control { step, .. } | Self::RoutedExpertWave { step, .. } => step,
        }
    }
}

/// Consumer seam for an eventual native/Metal stage encoder. It receives
/// source-native data synchronously, before the FIFO control arena can advance.
/// Implementing this trait alone does not authorize a full causal runtime.
pub trait DeepSeekV4NativeStageSink {
    fn consume_native_stage(
        &mut self,
        stage: DeepSeekV4NativeStage<'_>,
    ) -> Result<DeepSeekV4NativeStageConsumption>;
}

/// Incremental, deterministic layer program. It accepts a route set from a
/// future source-faithful router; it never invents router logits or experts.
#[derive(Debug, Clone)]
pub struct DeepSeekV4LayerPreparationScheduler {
    token_position: usize,
    layer: usize,
    route_set: DeepSeekV4SelectedRouteSet,
    route_wave_hot_bytes_required: u64,
    next_stage_index: usize,
    completed: Vec<DeepSeekV4LayerPreparationStep>,
}

impl DeepSeekV4LayerPreparationScheduler {
    /// Start a layer only after `prepare_decode_input` has seeded mHC from an
    /// authenticated embedding row. This checks source topology but performs
    /// no source read until [`Self::execute_next`] is called.
    pub fn new(
        context: &DeepSeekV4ExecutionContext,
        layer: usize,
        route_set: DeepSeekV4SelectedRouteSet,
    ) -> Result<Self> {
        context.spine().topology().layer(layer)?;
        if !context.decode_state().m_hc.initialized {
            return Err(scheduler_error(
                "prepare_decode_input must seed mHC before scheduling a layer",
            ));
        }
        let route_wave_hot_bytes_required = route_wave_hot_bytes(context, layer, route_set)?;
        let hot_capacity_bytes = context.expert_cache_state().hot_capacity_bytes;
        if hot_capacity_bytes < route_wave_hot_bytes_required {
            return Err(scheduler_error(format!(
                "top-6 route wave at layer {layer} needs {route_wave_hot_bytes_required} hot bytes, but the context has {hot_capacity_bytes}; a source-stage sink must be able to borrow every selected expert concurrently"
            )));
        }
        Ok(Self {
            token_position: context.decode_state().position,
            layer,
            route_set,
            route_wave_hot_bytes_required,
            next_stage_index: 0,
            completed: Vec::with_capacity(DSV4F_LAYER_PREPARATION_STAGE_COUNT),
        })
    }

    pub const fn token_position(&self) -> usize {
        self.token_position
    }

    pub const fn layer(&self) -> usize {
        self.layer
    }

    pub const fn route_set(&self) -> DeepSeekV4SelectedRouteSet {
        self.route_set
    }

    /// Exact source-native payload bytes which must remain hot for this
    /// selected six-expert wave. This is a concurrent-borrow contract, not an
    /// allocation or measurement.
    pub const fn route_wave_hot_bytes_required(&self) -> u64 {
        self.route_wave_hot_bytes_required
    }

    pub fn completed(&self) -> &[DeepSeekV4LayerPreparationStep] {
        &self.completed
    }

    pub const fn is_complete(&self) -> bool {
        self.next_stage_index == DSV4F_LAYER_PREPARATION_STAGE_COUNT
    }

    pub fn next_stage(&self) -> Option<DeepSeekV4LayerPreparationStage> {
        layer_stage_program().get(self.next_stage_index).copied()
    }

    /// Perform one bounded source-stage operation. It has no GPU side effects;
    /// callers that attach a native encoder should prefer
    /// [`Self::execute_next_with_sink`] so a control lease is consumed before
    /// a subsequent FIFO-arena mutation.
    pub fn execute_next(
        &mut self,
        context: &mut DeepSeekV4ExecutionContext,
    ) -> Result<Option<DeepSeekV4LayerPreparationStep>> {
        let Some(step) = self.stage_next(context)? else {
            return Ok(None);
        };
        self.commit_step(step.clone())?;
        Ok(Some(step))
    }

    /// Performs one bounded source-stage operation and synchronously gives it
    /// to the caller's native sink. This is the intended hand-off point for a
    /// later Metal encoder; it still has no default device implementation.
    ///
    /// A sink failure deliberately leaves the scheduler at the same program
    /// index. The source stage may have filled a bounded cache/arena entry,
    /// but the logical stage is never marked complete unless its consumer
    /// returns successfully.
    pub fn execute_next_with_sink<S: DeepSeekV4NativeStageSink>(
        &mut self,
        context: &mut DeepSeekV4ExecutionContext,
        sink: &mut S,
    ) -> Result<Option<DeepSeekV4LayerPreparationStep>> {
        let Some(mut step) = self.stage_next(context)? else {
            return Ok(None);
        };
        match &step.result {
            DeepSeekV4LayerPreparationResult::ControlLease(lease) => {
                let payload = context.control_arena().get(*lease)?;
                let consumption = sink.consume_native_stage(DeepSeekV4NativeStage::Control {
                    step: &step,
                    payload,
                })?;
                apply_consumption(&mut step, consumption)?;
            }
            DeepSeekV4LayerPreparationResult::RoutedExpertAccesses(accesses) => {
                let consumption =
                    sink.consume_native_stage(DeepSeekV4NativeStage::RoutedExpertWave {
                        step: &step,
                        route_set: self.route_set,
                        context,
                        accesses,
                    })?;
                apply_consumption(&mut step, consumption)?;
            }
        }
        self.commit_step(step.clone())?;
        Ok(Some(step))
    }

    fn stage_next(
        &self,
        context: &mut DeepSeekV4ExecutionContext,
    ) -> Result<Option<DeepSeekV4LayerPreparationStep>> {
        self.ensure_context(context)?;
        let Some(stage) = self.next_stage() else {
            return Ok(None);
        };
        let result = match stage {
            DeepSeekV4LayerPreparationStage::MhcAttentionControl => {
                DeepSeekV4LayerPreparationResult::ControlLease(
                    context.stage_mhc_control(self.layer, DeepSeekV4MhcBranch::Attention)?,
                )
            }
            DeepSeekV4LayerPreparationStage::AttentionControl(projection) => {
                DeepSeekV4LayerPreparationResult::ControlLease(
                    context.stage_attention_control(self.layer, projection)?,
                )
            }
            DeepSeekV4LayerPreparationStage::MhcFfnControl => {
                DeepSeekV4LayerPreparationResult::ControlLease(
                    context.stage_mhc_control(self.layer, DeepSeekV4MhcBranch::Ffn)?,
                )
            }
            DeepSeekV4LayerPreparationStage::RoutedExpertWave => {
                DeepSeekV4LayerPreparationResult::RoutedExpertAccesses(
                    context.acquire_selected_route_set(self.layer, self.route_set)?,
                )
            }
            DeepSeekV4LayerPreparationStage::SharedExpertControl(projection) => {
                DeepSeekV4LayerPreparationResult::ControlLease(
                    context.stage_shared_expert_control(self.layer, projection)?,
                )
            }
        };
        let step = DeepSeekV4LayerPreparationStep {
            sequence: self.next_stage_index,
            token_position: self.token_position,
            layer: self.layer,
            stage,
            logical_graph_node_ordinal: layer_graph_node_ordinal(self.layer, stage)?,
            result,
            actual_command_buffers: 0,
            actual_compute_encoders: 0,
            actual_gpu_dispatches: 0,
            actual_cpu_visible_waits: 0,
        };
        Ok(Some(step))
    }

    fn commit_step(&mut self, step: DeepSeekV4LayerPreparationStep) -> Result<()> {
        if step.sequence != self.next_stage_index {
            return Err(scheduler_error(
                "attempted to commit a stage that is not the current scheduler index",
            ));
        }
        self.next_stage_index = self
            .next_stage_index
            .checked_add(1)
            .ok_or_else(|| scheduler_error("scheduler stage index overflow"))?;
        self.completed.push(step);
        Ok(())
    }

    fn ensure_context(&self, context: &DeepSeekV4ExecutionContext) -> Result<()> {
        if context.decode_state().position != self.token_position {
            return Err(scheduler_error(
                "context token position changed after scheduler construction",
            ));
        }
        if !context.decode_state().m_hc.initialized {
            return Err(scheduler_error(
                "mHC state was reset during layer scheduling",
            ));
        }
        Ok(())
    }
}

/// The fixed source-native staging program for every one of the 43 base
/// layers. It intentionally excludes the MTP auxiliary layer.
pub const fn layer_stage_program(
) -> [DeepSeekV4LayerPreparationStage; DSV4F_LAYER_PREPARATION_STAGE_COUNT] {
    [
        DeepSeekV4LayerPreparationStage::MhcAttentionControl,
        DeepSeekV4LayerPreparationStage::AttentionControl(DeepSeekV4ControlProjection::WqA),
        DeepSeekV4LayerPreparationStage::AttentionControl(DeepSeekV4ControlProjection::WqB),
        DeepSeekV4LayerPreparationStage::AttentionControl(DeepSeekV4ControlProjection::Wkv),
        DeepSeekV4LayerPreparationStage::AttentionControl(DeepSeekV4ControlProjection::WoA),
        DeepSeekV4LayerPreparationStage::AttentionControl(DeepSeekV4ControlProjection::WoB),
        DeepSeekV4LayerPreparationStage::MhcFfnControl,
        DeepSeekV4LayerPreparationStage::RoutedExpertWave,
        DeepSeekV4LayerPreparationStage::SharedExpertControl(DeepSeekV4ExpertProjection::W1),
        DeepSeekV4LayerPreparationStage::SharedExpertControl(DeepSeekV4ExpertProjection::W2),
        DeepSeekV4LayerPreparationStage::SharedExpertControl(DeepSeekV4ExpertProjection::W3),
    ]
}

/// Ordinal in the canonical 391-node token graph, not a command-buffer ID.
pub fn layer_graph_node_ordinal(
    layer: usize,
    stage: DeepSeekV4LayerPreparationStage,
) -> Result<usize> {
    if layer >= DSV4F_BASE_LAYER_COUNT {
        return Err(scheduler_error(format!(
            "layer {layer} is outside the {DSV4F_BASE_LAYER_COUNT}-layer base body"
        )));
    }
    1usize
        .checked_add(
            layer
                .checked_mul(9)
                .ok_or_else(|| scheduler_error("layer graph ordinal multiplication overflow"))?,
        )
        .and_then(|base| base.checked_add(stage.logical_node_offset()))
        .ok_or_else(|| scheduler_error("layer graph ordinal overflow"))
}

fn scheduler_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 layer preparation scheduler: {}",
        message.into()
    ))
}

fn apply_consumption(
    step: &mut DeepSeekV4LayerPreparationStep,
    consumption: DeepSeekV4NativeStageConsumption,
) -> Result<()> {
    if consumption.host_intermediate_handoff_bytes != 0 {
        return Err(scheduler_error(
            "native stage sink reported a prohibited host intermediate handoff",
        ));
    }
    step.actual_command_buffers = consumption.actual_command_buffers;
    step.actual_compute_encoders = consumption.actual_compute_encoders;
    step.actual_gpu_dispatches = consumption.actual_gpu_dispatches;
    step.actual_cpu_visible_waits = consumption.actual_cpu_visible_waits;
    Ok(())
}

fn route_wave_hot_bytes(
    context: &DeepSeekV4ExecutionContext,
    layer: usize,
    route_set: DeepSeekV4SelectedRouteSet,
) -> Result<u64> {
    let layer_key = u16::try_from(layer)
        .map_err(|_| scheduler_error("layer does not fit a routed-expert cache key"))?;
    route_set.experts.iter().try_fold(0u64, |total, expert| {
        let descriptor = resolve_expert_bundle(
            context.spine().reader(),
            ExpertBundleKey::new(layer_key, *expert),
        )?;
        total
            .checked_add(descriptor.payload_bytes)
            .ok_or_else(|| scheduler_error("top-6 route-wave byte total overflow"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_program_covers_all_bounded_native_staging_families() {
        let program = layer_stage_program();
        assert_eq!(program.len(), DSV4F_LAYER_PREPARATION_STAGE_COUNT);
        assert_eq!(
            program[0],
            DeepSeekV4LayerPreparationStage::MhcAttentionControl
        );
        assert_eq!(
            program[7],
            DeepSeekV4LayerPreparationStage::RoutedExpertWave
        );
        assert_eq!(
            program[10],
            DeepSeekV4LayerPreparationStage::SharedExpertControl(DeepSeekV4ExpertProjection::W3)
        );
    }

    #[test]
    fn graph_ordinals_are_layer_local_and_exclude_mtp() {
        let first =
            layer_graph_node_ordinal(0, DeepSeekV4LayerPreparationStage::MhcAttentionControl)
                .unwrap();
        let last = layer_graph_node_ordinal(
            42,
            DeepSeekV4LayerPreparationStage::SharedExpertControl(DeepSeekV4ExpertProjection::W3),
        )
        .unwrap();
        assert_eq!(first, 1);
        assert_eq!(last, 1 + 42 * 9 + 7);
        assert!(layer_graph_node_ordinal(43, layer_stage_program()[0]).is_err());
    }
}
