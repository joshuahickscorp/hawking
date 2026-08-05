//! Source-lease and device-handoff contract for the DeepSeek-V4 P7 child body.
//!
//! The macOS-only `gravity_deepseek_v4_p7_device` module connects an all-device
//! P4B attention/KV continuation to the reusable six-expert P6 wave. This
//! preparation module deliberately implements neither side itself. It does
//! three safe, necessary things:
//!
//! 1. stages the real, hash-bound `hc_ffn_*` lease and `ffn_norm` control;
//! 2. validates the exact device-state shapes P4B must hand to FFN/MoE; and
//! 3. defines a P6 entry point whose input is a caller-owned Metal context and
//!    BF16 device buffer, never a readback/copy through host activations.
//!
//! This preparation module has no kernel dispatch, numerical-parity result,
//! Engine registration, causal loop, HCLI endpoint, or TPS claim. The separate
//! bounded P7 executor is likewise not a decoder-runtime or parity receipt.

use sha2::{Digest, Sha256};

use crate::gravity_deepseek_v4_execution_context::{
    DeepSeekV4ControlPayload, DeepSeekV4ExecutionContext, DeepSeekV4MhcBranch,
    DeepSeekV4PreparedDecodeInput,
};
use crate::gravity_deepseek_v4_layer0_prefix::{HC_FLAT_WIDTH, HC_MIX_WIDTH, HC_MULT, HIDDEN_SIZE};
use crate::gravity_deepseek_v4_layer_scheduler::{
    DeepSeekV4LayerPreparationStage, DeepSeekV4LayerPreparationStep, DeepSeekV4NativeStage,
    DeepSeekV4NativeStageConsumption, DeepSeekV4NativeStageSink,
};
use crate::gravity_deepseek_v4_runtime_spine::DeepSeekV4StagedTensor;
use crate::{Error, Result};

/// The static FFn RMSNorm tensor follows the same width for every base layer.
pub const DSV4F_P7_FFN_NORM_ELEMENTS: usize = HIDDEN_SIZE;
pub const DSV4F_P7_FFN_NORM_BF16_BYTES: usize = HIDDEN_SIZE * std::mem::size_of::<u16>();
pub const DSV4F_P7_MHC_STATE_BF16_BYTES: usize = HC_MULT * HIDDEN_SIZE * std::mem::size_of::<u16>();
pub const DSV4F_P7_MHC_POST_F32_BYTES: usize = HC_MULT * std::mem::size_of::<f32>();
pub const DSV4F_P7_MHC_COMB_F32_BYTES: usize = HC_MULT * HC_MULT * std::mem::size_of::<f32>();
pub const DSV4F_P7_POSITION1_KV_ROWS: usize = 2;
pub const DSV4F_P7_KV_HEAD_DIM: usize = 512;
pub const DSV4F_P7_POSITION1_KV_BF16_BYTES: usize =
    DSV4F_P7_POSITION1_KV_ROWS * DSV4F_P7_KV_HEAD_DIM * std::mem::size_of::<u16>();
pub const DSV4F_P7_ROUTE_COUNT: usize = 6;
pub const DSV4F_P7_ROUTED_EXPERT_COUNT: usize = 256;
pub const DSV4F_P7_GATE_LOGITS_F32_BYTES: usize =
    DSV4F_P7_ROUTED_EXPERT_COUNT * std::mem::size_of::<f32>();
pub const DSV4F_P7_ROUTE_VALID_U32_BYTES: usize = std::mem::size_of::<u32>();

/// Preparation phase only.  `SourceControlsBound` says the raw artifact
/// controls are available for a direct device upload; it does *not*
/// say that any mHC, norm, routing, expert, or residual computation ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4P7CompositionPhase {
    AwaitMhcFfnControl,
    SourceControlsBound,
}

impl DeepSeekV4P7CompositionPhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AwaitMhcFfnControl => "await_mhc_ffn_control",
            Self::SourceControlsBound => "source_controls_bound_not_executed",
        }
    }
}

/// Hash-bound description of one bounded source tensor.  Contracts carry this
/// metadata rather than persisting a duplicate source file or raw activation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4P7SourceTensorBinding {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: usize,
    pub sha256: String,
}

/// Exact source controls a P7 mHC-FFN pre/norm stage must upload from
/// the sealed Gravity artifact.  The payloads remain private to the staging
/// object and may be borrowed only for a direct device upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4P7FfnSourceContract {
    pub layer: usize,
    pub token_id: u32,
    pub token_position: usize,
    pub ffn_norm: DeepSeekV4P7SourceTensorBinding,
    pub hc_ffn_fn: DeepSeekV4P7SourceTensorBinding,
    pub hc_ffn_base: DeepSeekV4P7SourceTensorBinding,
    pub hc_ffn_scale: DeepSeekV4P7SourceTensorBinding,
    pub source_parent_retained: bool,
    pub source_upload_required_before_execution: bool,
    pub host_activation_handoff_permitted: bool,
    pub runtime_boundary: &'static str,
}

/// Bounded source staging for the P7 mHC-FFN pre/norm boundary.  This is a
/// native-stage sink solely so it consumes the scheduler lease while valid.
/// It returns zero command buffers because this source-preparation object
/// stages controls only; the separate P7 device executor owns dispatch.
pub struct DeepSeekV4P7SourceLeasePreparation {
    layer: usize,
    token_id: u32,
    token_position: usize,
    phase: DeepSeekV4P7CompositionPhase,
    ffn_norm: DeepSeekV4StagedTensor,
    mhc_ffn: Option<[DeepSeekV4StagedTensor; 3]>,
}

impl DeepSeekV4P7SourceLeasePreparation {
    /// Read only the bounded static FFn norm control.  The mHC controls must
    /// arrive later via the real scheduler's `MhcFfnControl` lease.
    pub fn new(
        context: &DeepSeekV4ExecutionContext,
        prepared: &DeepSeekV4PreparedDecodeInput,
        layer: usize,
    ) -> Result<Self> {
        context.spine().topology().layer(layer)?;
        if prepared.position != context.decode_state().position {
            return Err(p7_error(
                "prepared decode input position differs from the execution context",
            ));
        }
        let ffn_norm_name = format!("layers.{layer}.ffn_norm.weight");
        let metadata = context.spine().reader().tensor_metadata(&ffn_norm_name)?;
        if metadata.dtype != "BF16"
            || metadata.shape.as_slice() != [DSV4F_P7_FFN_NORM_ELEMENTS as u64]
            || metadata.bytes != DSV4F_P7_FFN_NORM_BF16_BYTES as u64
        {
            return Err(p7_error(format!(
                "{ffn_norm_name} geometry differs from the P7 BF16[4096] contract",
            )));
        }
        let ffn_norm = context.spine().stage_base_tensor_range(
            &ffn_norm_name,
            0..metadata.bytes,
            DSV4F_P7_FFN_NORM_BF16_BYTES,
        )?;
        Ok(Self {
            layer,
            token_id: prepared.token_id,
            token_position: prepared.position,
            phase: DeepSeekV4P7CompositionPhase::AwaitMhcFfnControl,
            ffn_norm,
            mhc_ffn: None,
        })
    }

    pub const fn phase(&self) -> DeepSeekV4P7CompositionPhase {
        self.phase
    }

    pub const fn layer(&self) -> usize {
        self.layer
    }

    pub const fn token_id(&self) -> u32 {
        self.token_id
    }

    pub const fn token_position(&self) -> usize {
        self.token_position
    }

    /// The exact raw BF16 bytes that an eventual device mHC-FFN/norm encoder
    /// must bind.  This borrows the bounded source control; it is not a host
    /// activation and callers must not turn it into an intermediate handoff.
    pub fn ffn_norm_source(&self) -> &DeepSeekV4StagedTensor {
        &self.ffn_norm
    }

    /// The exact raw mHC FFn controls once the source lease has been consumed.
    /// Returning `None` means no scheduler lease has yet been accepted.
    pub fn mhc_ffn_sources(&self) -> Option<&[DeepSeekV4StagedTensor; 3]> {
        self.mhc_ffn.as_ref()
    }

    /// Consume a real `MhcFfnControl` source payload synchronously.  No
    /// device work is performed here; the cloned bytes are bounded source
    /// controls needed for a direct upload before the FIFO arena moves.
    pub fn bind_mhc_ffn_control(
        &mut self,
        step: &DeepSeekV4LayerPreparationStep,
        payload: &DeepSeekV4ControlPayload,
    ) -> Result<DeepSeekV4NativeStageConsumption> {
        if self.phase != DeepSeekV4P7CompositionPhase::AwaitMhcFfnControl {
            return Err(p7_error("mHC FFn control was bound more than once"));
        }
        if step.layer != self.layer
            || step.token_position != self.token_position
            || step.stage != DeepSeekV4LayerPreparationStage::MhcFfnControl
        {
            return Err(p7_error(
                "scheduler step does not match the prepared P7 mHC-FFN boundary",
            ));
        }
        let tensors = match payload {
            DeepSeekV4ControlPayload::MhcControl {
                layer,
                branch: DeepSeekV4MhcBranch::Ffn,
                tensors,
            } if *layer == self.layer => tensors,
            DeepSeekV4ControlPayload::MhcControl { .. } => {
                return Err(p7_error(
                    "P7 requires the mHC FFn control branch for its prepared layer",
                ));
            }
            _ => return Err(p7_error("P7 mHC-FFN boundary received a non-mHC payload")),
        };
        validate_mhc_tensor(
            &tensors[0],
            &format!("layers.{}.hc_ffn_fn", self.layer),
            "F32",
            &[HC_MIX_WIDTH as u64, HC_FLAT_WIDTH as u64],
        )?;
        validate_mhc_tensor(
            &tensors[1],
            &format!("layers.{}.hc_ffn_base", self.layer),
            "F32",
            &[HC_MIX_WIDTH as u64],
        )?;
        validate_mhc_tensor(
            &tensors[2],
            &format!("layers.{}.hc_ffn_scale", self.layer),
            "F32",
            &[3],
        )?;
        self.mhc_ffn = Some(tensors.clone());
        self.phase = DeepSeekV4P7CompositionPhase::SourceControlsBound;
        Ok(DeepSeekV4NativeStageConsumption::default())
    }

    /// Return the complete source contract only after the live mHC source
    /// lease is safely bound.  This has no success status: it is an input
    /// specification for a separately parity-gated device executor.
    pub fn source_contract(&self) -> Result<DeepSeekV4P7FfnSourceContract> {
        let [hc_fn, hc_base, hc_scale] = self
            .mhc_ffn
            .as_ref()
            .ok_or_else(|| p7_error("P7 source contract requested before mHC FFn lease binding"))?;
        Ok(DeepSeekV4P7FfnSourceContract {
            layer: self.layer,
            token_id: self.token_id,
            token_position: self.token_position,
            ffn_norm: source_binding(&self.ffn_norm),
            hc_ffn_fn: source_binding(hc_fn),
            hc_ffn_base: source_binding(hc_base),
            hc_ffn_scale: source_binding(hc_scale),
            source_parent_retained: false,
            source_upload_required_before_execution: true,
            host_activation_handoff_permitted: false,
            runtime_boundary: "this P7 source-lease preparation stages static controls only; it performs no P4B handoff, mHC-FFN, norm, routing, MoE, residual, command submission, parity result, Engine, HCLI, or TPS execution; a separately parity-gated device executor must consume the controls",
        })
    }
}

impl DeepSeekV4NativeStageSink for DeepSeekV4P7SourceLeasePreparation {
    fn consume_native_stage(
        &mut self,
        stage: DeepSeekV4NativeStage<'_>,
    ) -> Result<DeepSeekV4NativeStageConsumption> {
        match stage {
            DeepSeekV4NativeStage::Control { step, payload } => {
                self.bind_mhc_ffn_control(step, payload)
            }
            DeepSeekV4NativeStage::RoutedExpertWave { step, .. } => Err(p7_error(format!(
                "P7 source preparation only consumes mHC-FFN controls, not routed-expert stage {}",
                step.stage.as_str()
            ))),
        }
    }
}

fn validate_mhc_tensor(
    tensor: &DeepSeekV4StagedTensor,
    expected_name: &str,
    expected_dtype: &str,
    expected_shape: &[u64],
) -> Result<()> {
    let expected_bytes = match expected_dtype {
        "F32" => expected_shape
            .iter()
            .try_fold(1usize, |total, dimension| {
                total
                    .checked_mul(*dimension as usize)
                    .ok_or_else(|| p7_error("P7 mHC tensor element count overflow"))
            })?
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| p7_error("P7 mHC tensor byte count overflow"))?,
        _ => return Err(p7_error("unsupported P7 mHC tensor dtype contract")),
    };
    if tensor.name != expected_name
        || tensor.dtype != expected_dtype
        || tensor.shape.as_slice() != expected_shape
        || tensor.bytes.len() != expected_bytes
    {
        return Err(p7_error(format!(
            "{} source geometry differs from P7 mHC-FFN contract",
            expected_name
        )));
    }
    Ok(())
}

fn source_binding(tensor: &DeepSeekV4StagedTensor) -> DeepSeekV4P7SourceTensorBinding {
    DeepSeekV4P7SourceTensorBinding {
        name: tensor.name.clone(),
        dtype: tensor.dtype.clone(),
        shape: tensor.shape.clone(),
        bytes: tensor.bytes.len(),
        sha256: format!("{:x}", Sha256::digest(&tensor.bytes)),
    }
}

fn p7_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 P7 composition contract: {}",
        message.into()
    ))
}

/// The all-device P7 handoff types are macOS-only because they borrow actual
/// Metal buffers.  They carry no host slices and deliberately offer no
/// readback API.
#[cfg(target_os = "macos")]
mod device {
    use super::*;
    use crate::metal::MetalContext;

    /// Caller-owned bounded-attention output/state. The predecessor must produce
    /// the complete attention residual `BF16[4,4096]`; a sparse-only output is
    /// intentionally insufficient to construct this type.  Position zero from
    /// the exact P3A->P4A continuation has no reusable causal-KV buffer because
    /// that bounded attention proof does not claim a decode-cache contract.
    pub struct DeepSeekV4P7AttentionDeviceState<'a> {
        pub metal: &'a MetalContext,
        pub attention_hc_post_bf16: &'a metal::Buffer,
        pub causal_kv_cache_bf16: Option<&'a metal::Buffer>,
        pub layer: usize,
        pub token_id: u32,
        pub token_position: usize,
        pub kv_rows: usize,
    }

    impl<'a> DeepSeekV4P7AttentionDeviceState<'a> {
        /// Bind the complete, exact P3A->P4A BOS attention residual without
        /// inventing or uploading a causal KV cache.  The FFN continuation
        /// consumes only this mHC state; no cache is required or implied here.
        pub fn position0(
            metal: &'a MetalContext,
            attention_hc_post_bf16: &'a metal::Buffer,
            layer: usize,
            token_id: u32,
        ) -> Result<Self> {
            if attention_hc_post_bf16.length() < DSV4F_P7_MHC_STATE_BF16_BYTES as u64 {
                return Err(p7_error(
                    "P4A attention residual buffer is smaller than BF16[4,4096]",
                ));
            }
            Ok(Self {
                metal,
                attention_hc_post_bf16,
                causal_kv_cache_bf16: None,
                layer,
                token_id,
                token_position: 0,
                kv_rows: 0,
            })
        }

        /// Bind P4B output without creating a new Metal context or copying a
        /// hidden state through host memory.  Position-one currently requires
        /// exactly two cache rows, which is the narrow P4B proof scope.
        pub fn position1(
            metal: &'a MetalContext,
            attention_hc_post_bf16: &'a metal::Buffer,
            causal_kv_cache_bf16: &'a metal::Buffer,
            layer: usize,
            token_id: u32,
        ) -> Result<Self> {
            if attention_hc_post_bf16.length() < DSV4F_P7_MHC_STATE_BF16_BYTES as u64 {
                return Err(p7_error(
                    "P4B attention residual buffer is smaller than BF16[4,4096]",
                ));
            }
            if causal_kv_cache_bf16.length() < DSV4F_P7_POSITION1_KV_BF16_BYTES as u64 {
                return Err(p7_error("P4B causal KV buffer is smaller than BF16[2,512]"));
            }
            Ok(Self {
                metal,
                attention_hc_post_bf16,
                causal_kv_cache_bf16: Some(causal_kv_cache_bf16),
                layer,
                token_id,
                token_position: 1,
                kv_rows: DSV4F_P7_POSITION1_KV_ROWS,
            })
        }

        /// Bind a fullseq multi-token attention residual with a growing KV
        /// cache. `kv_rows` must equal `token_position + 1` (causal fill).
        pub fn fullseq(
            metal: &'a MetalContext,
            attention_hc_post_bf16: &'a metal::Buffer,
            causal_kv_cache_bf16: &'a metal::Buffer,
            layer: usize,
            token_id: u32,
            token_position: usize,
            kv_rows: usize,
        ) -> Result<Self> {
            if token_position >= 128 {
                return Err(p7_error(
                    "fullseq P7 attention state refuses positions beyond the 128-token window",
                ));
            }
            if kv_rows != token_position.saturating_add(1) {
                return Err(p7_error(
                    "fullseq P7 attention state requires kv_rows == token_position + 1",
                ));
            }
            if attention_hc_post_bf16.length() < DSV4F_P7_MHC_STATE_BF16_BYTES as u64 {
                return Err(p7_error(
                    "fullseq attention residual buffer is smaller than BF16[4,4096]",
                ));
            }
            let min_kv = kv_rows
                .checked_mul(DSV4F_P7_KV_HEAD_DIM)
                .and_then(|n| n.checked_mul(std::mem::size_of::<u16>()))
                .ok_or_else(|| p7_error("fullseq KV byte geometry overflow"))?;
            if causal_kv_cache_bf16.length() < min_kv as u64 {
                return Err(p7_error(
                    "fullseq causal KV buffer is smaller than the declared growing-KV geometry",
                ));
            }
            Ok(Self {
                metal,
                attention_hc_post_bf16,
                causal_kv_cache_bf16: Some(causal_kv_cache_bf16),
                layer,
                token_id,
                token_position,
                kv_rows,
            })
        }
    }

    /// Device-only result of mHC-FFN-pre plus FFn RMSNorm. The source-lease
    /// preparation object cannot construct it; the bounded P7 device executor
    /// supplies it in the same context as [`DeepSeekV4P7AttentionDeviceState`].
    pub struct DeepSeekV4P7FfnDeviceState<'a> {
        pub attention: &'a DeepSeekV4P7AttentionDeviceState<'a>,
        pub ffn_norm_bf16: &'a metal::Buffer,
        pub ffn_post_f32: &'a metal::Buffer,
        pub ffn_comb_f32: &'a metal::Buffer,
    }

    impl<'a> DeepSeekV4P7FfnDeviceState<'a> {
        pub fn new(
            attention: &'a DeepSeekV4P7AttentionDeviceState<'a>,
            ffn_norm_bf16: &'a metal::Buffer,
            ffn_post_f32: &'a metal::Buffer,
            ffn_comb_f32: &'a metal::Buffer,
        ) -> Result<Self> {
            if ffn_norm_bf16.length() < DSV4F_P7_FFN_NORM_BF16_BYTES as u64
                || ffn_post_f32.length() < DSV4F_P7_MHC_POST_F32_BYTES as u64
                || ffn_comb_f32.length() < DSV4F_P7_MHC_COMB_F32_BYTES as u64
            {
                return Err(p7_error(
                    "P7 FFn device state has an invalid norm/post/comb buffer geometry",
                ));
            }
            Ok(Self {
                attention,
                ffn_norm_bf16,
                ffn_post_f32,
                ffn_comb_f32,
            })
        }

        /// Build the only valid reusable P6 entry: the P6 implementation gets
        /// the caller's original Metal context and a BF16 device buffer.  No
        /// `Vec`, host pointer, readback, or independently-created context is
        /// present in this API.
        pub fn p6_input(
            &'a self,
            source: &'a DeepSeekV4P7FfnSourceContract,
        ) -> Result<DeepSeekV4P7P6DeviceInput<'a>> {
            if source.layer != self.attention.layer
                || source.token_id != self.attention.token_id
                || source.token_position != self.attention.token_position
            {
                return Err(p7_error(
                    "P7 source contract does not match the caller-owned device state",
                ));
            }
            Ok(DeepSeekV4P7P6DeviceInput {
                metal: self.attention.metal,
                ffn_norm_bf16: self.ffn_norm_bf16,
                layer: self.attention.layer,
                token_id: self.attention.token_id,
                token_position: self.attention.token_position,
            })
        }
    }

    /// No-host P6 entry contract.  The later P6 implementation must perform
    /// Gate/tid2eid/device routing and source-bound expert execution directly
    /// from this buffer; route weights must never arrive from host code.
    pub struct DeepSeekV4P7P6DeviceInput<'a> {
        pub metal: &'a MetalContext,
        pub ffn_norm_bf16: &'a metal::Buffer,
        pub layer: usize,
        pub token_id: u32,
        pub token_position: usize,
    }

    /// Device-only P6 output. It preserves the routed result plus read-only
    /// route-control diagnostics on device so a bounded post-completion
    /// parity probe can inspect them without placing a host activation bridge
    /// inside the P7 graph.
    pub struct DeepSeekV4P7P6DeviceOutput {
        pub moe_output_bf16: metal::Buffer,
        pub route_ids_u32: metal::Buffer,
        pub route_weights_f32: metal::Buffer,
        pub gate_logits_f32: metal::Buffer,
        pub original_scores_f32: metal::Buffer,
        pub route_valid_u32: metal::Buffer,
    }

    impl DeepSeekV4P7P6DeviceOutput {
        pub fn validate(&self) -> Result<()> {
            if self.moe_output_bf16.length() < DSV4F_P7_FFN_NORM_BF16_BYTES as u64
                || self.route_ids_u32.length()
                    < (DSV4F_P7_ROUTE_COUNT * std::mem::size_of::<u32>()) as u64
                || self.route_weights_f32.length()
                    < (DSV4F_P7_ROUTE_COUNT * std::mem::size_of::<f32>()) as u64
                || self.gate_logits_f32.length() < DSV4F_P7_GATE_LOGITS_F32_BYTES as u64
                || self.original_scores_f32.length() < DSV4F_P7_GATE_LOGITS_F32_BYTES as u64
                || self.route_valid_u32.length() < DSV4F_P7_ROUTE_VALID_U32_BYTES as u64
            {
                return Err(p7_error(
                    "P6 device output does not expose the required MoE/route buffers",
                ));
            }
            Ok(())
        }
    }

    /// P6 reuse point. Implementors receive the caller-owned context
    /// and BF16 device buffer above.  A P6 implementation that copies the
    /// hidden state to CPU, accepts host route weights, or opens a separate
    /// context does not satisfy this trait's contract.
    pub trait DeepSeekV4P7P6DeviceExecutor {
        fn execute_p6_on_device(
            &mut self,
            input: DeepSeekV4P7P6DeviceInput<'_>,
        ) -> Result<DeepSeekV4P7P6DeviceOutput>;
    }

    /// Exact input needed by the mHC-FFN-post device stage. The output becomes
    /// the next layer's BF16[4,4096] child state; this preparation module does
    /// not provide a kernel, while the bounded P7 executor does.
    pub struct DeepSeekV4P7FfnPostDeviceInput<'a> {
        pub metal: &'a MetalContext,
        pub moe_output_bf16: &'a metal::Buffer,
        pub attention_hc_post_bf16: &'a metal::Buffer,
        pub ffn_post_f32: &'a metal::Buffer,
        pub ffn_comb_f32: &'a metal::Buffer,
    }

    impl<'a> DeepSeekV4P7FfnPostDeviceInput<'a> {
        pub fn new(
            ffn: &'a DeepSeekV4P7FfnDeviceState<'a>,
            p6: &'a DeepSeekV4P7P6DeviceOutput,
        ) -> Result<Self> {
            p6.validate()?;
            Ok(Self {
                metal: ffn.attention.metal,
                moe_output_bf16: &p6.moe_output_bf16,
                attention_hc_post_bf16: ffn.attention.attention_hc_post_bf16,
                ffn_post_f32: ffn.ffn_post_f32,
                ffn_comb_f32: ffn.ffn_comb_f32,
            })
        }
    }
}

#[cfg(target_os = "macos")]
pub use device::{
    DeepSeekV4P7AttentionDeviceState, DeepSeekV4P7FfnDeviceState, DeepSeekV4P7FfnPostDeviceInput,
    DeepSeekV4P7P6DeviceExecutor, DeepSeekV4P7P6DeviceInput, DeepSeekV4P7P6DeviceOutput,
};
