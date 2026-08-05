//! Bounded all-device DeepSeek-V4 P7 composition shell.
//!
//! This is deliberately narrow: layer 0 at either tokenizer-bound BOS /
//! position zero from the exact P3A->P4A continuation, or token 19923 /
//! position one from P4B. Given a caller-owned bounded-attention device state,
//! it keeps the same Metal context through the following real device graph:
//!
//! ```text
//! P4B attention HC state [4,4096]
//!   -> P7 hc_ffn_pre / Sinkhorn -> P7 FFn RMSNorm
//!   -> reusable P6 Gate/router/six-expert/shared-expert graph
//!   -> P7 hc_ffn_post -> child HC state [4,4096]
//! ```
//!
//! Static controls are hash-checked before their one-time device upload.  No
//! activation, route weight, route ID, KV state, or child state is read back
//! to CPU here.  This module is not an Engine, causal loop, receipt producer,
//! HCLI surface, generation path, or TPS claim.  In particular, P4B's current
//! V2.1-only predecessor status remains explicit in the returned metadata and
//! cannot be represented as exact-storage P7 evidence.

use std::mem::size_of;

use sha2::{Digest, Sha256};

use crate::gravity_deepseek_v4_layer0_continuation::{
    POSITION1, POSITION1_KV_ROWS, POSITION1_TOKEN_ID,
};
use crate::gravity_deepseek_v4_layer0_prefix::{
    HC_EPS, HC_FLAT_WIDTH, HC_MIX_WIDTH, HC_MULT, HC_SINKHORN_ITERS, HIDDEN_SIZE, PREFIX_TOKEN_ID,
    RMS_NORM_EPS,
};
use crate::gravity_deepseek_v4_p4b_device::{
    DeepSeekV4Layer0P4bDeviceExecutor, DeepSeekV4P4bParityClassification,
};
use crate::gravity_deepseek_v4_p7_composition::{
    DeepSeekV4P7AttentionDeviceState, DeepSeekV4P7CompositionPhase, DeepSeekV4P7FfnDeviceState,
    DeepSeekV4P7FfnSourceContract, DeepSeekV4P7P6DeviceExecutor, DeepSeekV4P7P6DeviceOutput,
    DeepSeekV4P7SourceLeasePreparation, DeepSeekV4P7SourceTensorBinding,
    DSV4F_P7_FFN_NORM_BF16_BYTES, DSV4F_P7_MHC_COMB_F32_BYTES, DSV4F_P7_MHC_POST_F32_BYTES,
    DSV4F_P7_MHC_STATE_BF16_BYTES, DSV4F_P7_POSITION1_KV_BF16_BYTES,
};
use crate::gravity_deepseek_v4_runtime_spine::DeepSeekV4StagedTensor;
use crate::metal::MetalContext;
use crate::{Error, Result};

const P7_MHC_PRE_KERNEL: &str = "deepseek_v4_p7_mhc_ffn_pre_authority";
const P7_FFN_NORM_KERNEL: &str = "deepseek_v4_p7_ffn_rmsnorm_bf16_authority";
const P7_MHC_POST_KERNEL: &str = "deepseek_v4_p7_mhc_ffn_post_authority";

const P7_LAYER: usize = 0;
const P7_POSITION0_TOKEN_ID: u32 = PREFIX_TOKEN_ID as u32;
const P7_POSITION0: usize = 0;
const P7_POSITION1_TOKEN_ID: u32 = POSITION1_TOKEN_ID as u32;
const P7_POSITION1: usize = POSITION1;
const P7_PRE_NORM_THREADS: u32 = 1;
const P7_POST_THREADS: u32 = 256;

/// Structural counts owned by `DeepSeekV4P7BoundedDeviceExecutor` after its
/// P4B predecessor is ready: pre/norm in one batch and mHC-post in a second.
/// These deliberately exclude the caller-owned P4B graph and the reusable P6
/// graph. They are not a runtime or TPS measurement.
pub const DSV4F_P7_OWNED_COMMAND_BUFFERS: usize = 2;
pub const DSV4F_P7_OWNED_CPU_VISIBLE_WAITS: usize = 2;
pub const DSV4F_P7_OWNED_DEVICE_DISPATCHES: usize = 3;
pub const DSV4F_P7_OWNED_COMPUTE_ENCODERS: usize = 3;

/// The bounded-attention predecessor's classification is intentionally carried
/// through the device output.  It distinguishes the exact P3A->P4A BOS
/// attention boundary from P4B's Numeric Parity V2.1-only position-one
/// boundary; neither label promotes P7's own math or child state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4P7P4BPredecessorParity {
    NumericParityV21Only,
    ExactP4aAttentionOnly,
}

impl DeepSeekV4P7P4BPredecessorParity {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NumericParityV21Only => "P4B_NUMERIC_PARITY_V2_1_ONLY",
            Self::ExactP4aAttentionOnly => "P4A_EXACT_ATTENTION_ONLY",
        }
    }

    /// Whether the *attention predecessor* is exact. This says nothing about
    /// P7 mHC/FFN/MoE parity or about the child state.
    pub const fn predecessor_attention_is_exact(self) -> bool {
        matches!(self, Self::ExactP4aAttentionOnly)
    }

    /// Retained for the existing P4B caller. Do not use this to promote the
    /// P7 child: P7 currently carries no end-to-end exact-storage claim.
    pub const fn is_exact_storage(self) -> bool {
        false
    }
}

/// Device-only output of the bounded P7 graph.  P6's output owns the device
/// route IDs/weights and MoE BF16 vector; `child_hc_state_bf16` is the next
/// layer-shaped mHC state. The mHC pre/norm intermediates are retained as
/// device-only post-completion diagnostic handles so a later sidecar can
/// assess the exact bounded-attention input against independent FP64 authority. There is
/// intentionally no host readback helper in this library surface.
pub struct DeepSeekV4P7DeviceOutput {
    pub ffn_reduced_bf16: metal::Buffer,
    pub ffn_norm_bf16: metal::Buffer,
    pub mhc_flat_rsqrt_f32: metal::Buffer,
    pub mhc_mixes_f32: metal::Buffer,
    pub mhc_pre_f32: metal::Buffer,
    pub mhc_post_f32: metal::Buffer,
    pub mhc_comb_f32: metal::Buffer,
    pub child_hc_state_bf16: metal::Buffer,
    pub p6: DeepSeekV4P7P6DeviceOutput,
    pub p4b_predecessor_parity: DeepSeekV4P7P4BPredecessorParity,
    pub layer: usize,
    pub token_id: u32,
    pub token_position: usize,
}

impl DeepSeekV4P7DeviceOutput {
    pub fn validate(&self) -> Result<()> {
        if self.ffn_reduced_bf16.length() < DSV4F_P7_FFN_NORM_BF16_BYTES as u64
            || self.ffn_norm_bf16.length() < DSV4F_P7_FFN_NORM_BF16_BYTES as u64
            || self.mhc_flat_rsqrt_f32.length() < size_of::<f32>() as u64
            || self.mhc_mixes_f32.length() < (HC_MIX_WIDTH * size_of::<f32>()) as u64
            || self.mhc_pre_f32.length() < (HC_MULT * size_of::<f32>()) as u64
            || self.mhc_post_f32.length() < DSV4F_P7_MHC_POST_F32_BYTES as u64
            || self.mhc_comb_f32.length() < DSV4F_P7_MHC_COMB_F32_BYTES as u64
            || self.child_hc_state_bf16.length() < DSV4F_P7_MHC_STATE_BF16_BYTES as u64
            || !is_supported_p7_identity(self.layer, self.token_id, self.token_position)
        {
            return Err(p7_device_error(
                "P7 device output has invalid child-state geometry or identity",
            ));
        }
        self.p6.validate()
    }
}

/// Prepared static P7 device graph.  It owns only uploaded source controls and
/// P7 scratch/output buffers.  The caller owns P4B state and the Metal context
/// for each execution; the queue identity is checked before any dispatch.
pub struct DeepSeekV4P7BoundedDeviceExecutor {
    source: DeepSeekV4P7FfnSourceContract,
    context_queue_identity: usize,
    hc_fn_f32: metal::Buffer,
    hc_base_f32: metal::Buffer,
    hc_scale_f32: metal::Buffer,
    ffn_norm_weight_bf16: metal::Buffer,
    reduced_bf16: metal::Buffer,
    flat_rsqrt_f32: metal::Buffer,
    mixes_f32: metal::Buffer,
    pre_f32: metal::Buffer,
    post_f32: metal::Buffer,
    comb_f32: metal::Buffer,
    ffn_norm_bf16: metal::Buffer,
    post_threads: u32,
    p6: Box<dyn DeepSeekV4P7P6DeviceExecutor>,
}

impl DeepSeekV4P7BoundedDeviceExecutor {
    /// Prepare static P7 controls from the caller's already-consumed P7
    /// source lease.  Source bytes are uploaded once; no activation is copied
    /// or computed on the host.
    pub fn prepare_from_source_lease(
        metal: &MetalContext,
        preparation: &DeepSeekV4P7SourceLeasePreparation,
        p6: Box<dyn DeepSeekV4P7P6DeviceExecutor>,
    ) -> Result<Self> {
        if preparation.phase() != DeepSeekV4P7CompositionPhase::SourceControlsBound {
            return Err(p7_device_error(
                "P7 source lease is not bound to MhcFfnControl",
            ));
        }
        let source = preparation.source_contract()?;
        let mhc_ffn = preparation
            .mhc_ffn_sources()
            .ok_or_else(|| p7_device_error("P7 source lease lacks mHC-FFN payloads"))?;
        Self::prepare(metal, source, preparation.ffn_norm_source(), mhc_ffn, p6)
    }

    /// Prepare against a caller-provided, source-hash-bound P7 contract.
    /// This accepts an abstract P6 trait object so the executor does not
    /// smuggle a separate context or host routing implementation into P7.
    pub fn prepare(
        metal: &MetalContext,
        source: DeepSeekV4P7FfnSourceContract,
        ffn_norm_source: &DeepSeekV4StagedTensor,
        mhc_ffn_sources: &[DeepSeekV4StagedTensor; 3],
        p6: Box<dyn DeepSeekV4P7P6DeviceExecutor>,
    ) -> Result<Self> {
        validate_source_contract(&source)?;
        validate_staged_tensor(
            ffn_norm_source,
            &source.ffn_norm,
            "BF16",
            &[HIDDEN_SIZE as u64],
            DSV4F_P7_FFN_NORM_BF16_BYTES,
        )?;
        validate_staged_tensor(
            &mhc_ffn_sources[0],
            &source.hc_ffn_fn,
            "F32",
            &[HC_MIX_WIDTH as u64, HC_FLAT_WIDTH as u64],
            HC_MIX_WIDTH * HC_FLAT_WIDTH * size_of::<f32>(),
        )?;
        validate_staged_tensor(
            &mhc_ffn_sources[1],
            &source.hc_ffn_base,
            "F32",
            &[HC_MIX_WIDTH as u64],
            HC_MIX_WIDTH * size_of::<f32>(),
        )?;
        validate_staged_tensor(
            &mhc_ffn_sources[2],
            &source.hc_ffn_scale,
            "F32",
            &[3],
            3 * size_of::<f32>(),
        )?;

        let max_post_threads = metal
            .pipeline(P7_MHC_POST_KERNEL)?
            .max_total_threads_per_threadgroup() as u32;
        if max_post_threads < P7_POST_THREADS {
            return Err(p7_device_error(format!(
                "P7 mHC-FFN-post kernel supports only {max_post_threads} threads"
            )));
        }
        for kernel in [P7_MHC_PRE_KERNEL, P7_FFN_NORM_KERNEL] {
            let maximum = metal.pipeline(kernel)?.max_total_threads_per_threadgroup() as u32;
            if maximum < P7_PRE_NORM_THREADS {
                return Err(p7_device_error(format!(
                    "P7 {kernel} does not support one authority thread"
                )));
            }
        }

        Ok(Self {
            source,
            context_queue_identity: context_queue_identity(metal),
            hc_fn_f32: metal.new_buffer_with_bytes_checked(&mhc_ffn_sources[0].bytes)?,
            hc_base_f32: metal.new_buffer_with_bytes_checked(&mhc_ffn_sources[1].bytes)?,
            hc_scale_f32: metal.new_buffer_with_bytes_checked(&mhc_ffn_sources[2].bytes)?,
            ffn_norm_weight_bf16: metal.new_buffer_with_bytes_checked(&ffn_norm_source.bytes)?,
            reduced_bf16: metal.new_buffer_checked(DSV4F_P7_FFN_NORM_BF16_BYTES)?,
            flat_rsqrt_f32: metal.new_buffer_checked(size_of::<f32>())?,
            mixes_f32: metal.new_buffer_checked(HC_MIX_WIDTH * size_of::<f32>())?,
            pre_f32: metal.new_buffer_checked(HC_MULT * size_of::<f32>())?,
            post_f32: metal.new_buffer_checked(DSV4F_P7_MHC_POST_F32_BYTES)?,
            comb_f32: metal.new_buffer_checked(DSV4F_P7_MHC_COMB_F32_BYTES)?,
            ffn_norm_bf16: metal.new_buffer_checked(DSV4F_P7_FFN_NORM_BF16_BYTES)?,
            post_threads: P7_POST_THREADS,
            p6,
        })
    }

    pub fn source_contract(&self) -> &DeepSeekV4P7FfnSourceContract {
        &self.source
    }

    /// Execute directly from the reusable P4B executor's retained device
    /// state. This is the only convenience handoff provided by P7: it borrows
    /// P4B's same-context buffers without mapping, copying, or reading them
    /// on the host, and it derives the honest predecessor label from P4B
    /// rather than accepting a caller-supplied stronger classification.
    pub fn execute_from_p4b(
        &mut self,
        p4b: &DeepSeekV4Layer0P4bDeviceExecutor,
        metal: &MetalContext,
    ) -> Result<DeepSeekV4P7DeviceOutput> {
        let p4b_predecessor_parity = match p4b.parity_classification() {
            DeepSeekV4P4bParityClassification::NumericParityV21Only => {
                DeepSeekV4P7P4BPredecessorParity::NumericParityV21Only
            }
        };
        let attention = p4b.p7_attention_state(metal)?;
        self.execute_position1(attention, p4b_predecessor_parity)
    }

    /// Execute the exact P3A->P4A BOS attention continuation without reading
    /// an intermediate activation on the host. The P4A predecessor label is
    /// preserved, but this does not claim P7 or its child state is exact.
    pub fn execute_position0(
        &mut self,
        attention: DeepSeekV4P7AttentionDeviceState<'_>,
    ) -> Result<DeepSeekV4P7DeviceOutput> {
        if attention.layer != P7_LAYER
            || attention.token_id != P7_POSITION0_TOKEN_ID
            || attention.token_position != P7_POSITION0
        {
            return Err(p7_device_error(
                "position-zero P4A continuation requires layer-0/BOS/position-0 state",
            ));
        }
        self.execute_bounded(
            attention,
            DeepSeekV4P7P4BPredecessorParity::ExactP4aAttentionOnly,
        )
    }

    /// Execute the bounded P7 graph without reading any intermediate device
    /// state on the host. The caller must supply the live P4B buffers from the
    /// same context that prepared this executor. A V2.1-only P4B input is
    /// preserved as such in the output and cannot become exact-storage P7
    /// evidence.
    pub fn execute_position1(
        &mut self,
        attention: DeepSeekV4P7AttentionDeviceState<'_>,
        p4b_predecessor_parity: DeepSeekV4P7P4BPredecessorParity,
    ) -> Result<DeepSeekV4P7DeviceOutput> {
        if p4b_predecessor_parity != DeepSeekV4P7P4BPredecessorParity::NumericParityV21Only {
            return Err(p7_device_error(
                "position-one P4B continuation requires its Numeric Parity V2.1-only predecessor label",
            ));
        }
        if attention.layer != P7_LAYER
            || attention.token_id != P7_POSITION1_TOKEN_ID
            || attention.token_position != P7_POSITION1
        {
            return Err(p7_device_error(
                "position-one P4B continuation requires layer-0/token-19923/position-1 state",
            ));
        }
        self.execute_bounded(attention, p4b_predecessor_parity)
    }

    fn execute_bounded(
        &mut self,
        attention: DeepSeekV4P7AttentionDeviceState<'_>,
        p4b_predecessor_parity: DeepSeekV4P7P4BPredecessorParity,
    ) -> Result<DeepSeekV4P7DeviceOutput> {
        self.validate_attention(&attention)?;
        self.dispatch_pre_and_norm(&attention)?;

        let ffn = DeepSeekV4P7FfnDeviceState::new(
            &attention,
            &self.ffn_norm_bf16,
            &self.post_f32,
            &self.comb_f32,
        )?;
        let p6_input = ffn.p6_input(&self.source)?;
        let p6 = self.p6.execute_p6_on_device(p6_input)?;
        p6.validate()?;

        let child_hc_state_bf16 = attention
            .metal
            .new_buffer_checked(DSV4F_P7_MHC_STATE_BF16_BYTES)?;
        self.dispatch_post(&attention, &p6, &child_hc_state_bf16)?;
        let output = DeepSeekV4P7DeviceOutput {
            ffn_reduced_bf16: self.reduced_bf16.to_owned(),
            ffn_norm_bf16: self.ffn_norm_bf16.to_owned(),
            mhc_flat_rsqrt_f32: self.flat_rsqrt_f32.to_owned(),
            mhc_mixes_f32: self.mixes_f32.to_owned(),
            mhc_pre_f32: self.pre_f32.to_owned(),
            mhc_post_f32: self.post_f32.to_owned(),
            mhc_comb_f32: self.comb_f32.to_owned(),
            child_hc_state_bf16,
            p6,
            p4b_predecessor_parity,
            layer: attention.layer,
            token_id: attention.token_id,
            token_position: attention.token_position,
        };
        output.validate()?;
        Ok(output)
    }

    fn validate_attention(&self, attention: &DeepSeekV4P7AttentionDeviceState<'_>) -> Result<()> {
        if context_queue_identity(attention.metal) != self.context_queue_identity {
            return Err(p7_device_error(
                "P7 attention state belongs to a different caller-owned MetalContext/queue",
            ));
        }
        let position0 = attention.layer == P7_LAYER
            && attention.token_id == P7_POSITION0_TOKEN_ID
            && attention.token_position == P7_POSITION0
            && attention.kv_rows == 0
            && attention.causal_kv_cache_bf16.is_none();
        let position1 = attention.layer == P7_LAYER
            && attention.token_id == P7_POSITION1_TOKEN_ID
            && attention.token_position == P7_POSITION1
            && attention.kv_rows == POSITION1_KV_ROWS
            && attention
                .causal_kv_cache_bf16
                .is_some_and(|cache| cache.length() >= DSV4F_P7_POSITION1_KV_BF16_BYTES as u64);
        if !position0 && !position1
            || attention.attention_hc_post_bf16.length() < DSV4F_P7_MHC_STATE_BF16_BYTES as u64
        {
            return Err(p7_device_error(
                "P7 accepts only same-context P4A layer-0/BOS/position-0 or P4B layer-0/token-19923/position-1 BF16 state",
            ));
        }
        Ok(())
    }

    fn dispatch_pre_and_norm(
        &self,
        attention: &DeepSeekV4P7AttentionDeviceState<'_>,
    ) -> Result<()> {
        let hidden = HIDDEN_SIZE as u32;
        let hc_mult = HC_MULT as u32;
        let mix_width = HC_MIX_WIDTH as u32;
        let sinkhorn_iters = HC_SINKHORN_ITERS as u32;
        attention.metal.dispatch_batch(|batch| {
            batch.dispatch_threads(
                P7_MHC_PRE_KERNEL,
                (1, 1, 1),
                (P7_PRE_NORM_THREADS, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(attention.attention_hc_post_bf16), 0);
                    encoder.set_buffer(1, Some(&self.hc_fn_f32), 0);
                    encoder.set_buffer(2, Some(&self.hc_scale_f32), 0);
                    encoder.set_buffer(3, Some(&self.hc_base_f32), 0);
                    encoder.set_buffer(4, Some(&self.reduced_bf16), 0);
                    encoder.set_buffer(5, Some(&self.flat_rsqrt_f32), 0);
                    encoder.set_buffer(6, Some(&self.mixes_f32), 0);
                    encoder.set_buffer(7, Some(&self.pre_f32), 0);
                    encoder.set_buffer(8, Some(&self.post_f32), 0);
                    encoder.set_buffer(9, Some(&self.comb_f32), 0);
                    set_u32(encoder, 10, &hidden);
                    set_u32(encoder, 11, &hc_mult);
                    set_u32(encoder, 12, &mix_width);
                    set_u32(encoder, 13, &sinkhorn_iters);
                    set_f32(encoder, 14, &RMS_NORM_EPS);
                    set_f32(encoder, 15, &HC_EPS);
                },
            )?;
            batch.dispatch_threads(
                P7_FFN_NORM_KERNEL,
                (1, 1, 1),
                (P7_PRE_NORM_THREADS, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.reduced_bf16), 0);
                    encoder.set_buffer(1, Some(&self.ffn_norm_weight_bf16), 0);
                    encoder.set_buffer(2, Some(&self.ffn_norm_bf16), 0);
                    set_u32(encoder, 3, &hidden);
                    set_f32(encoder, 4, &RMS_NORM_EPS);
                },
            )?;
            Ok(())
        })
    }

    fn dispatch_post(
        &self,
        attention: &DeepSeekV4P7AttentionDeviceState<'_>,
        p6: &DeepSeekV4P7P6DeviceOutput,
        child_hc_state_bf16: &metal::Buffer,
    ) -> Result<()> {
        let hidden = HIDDEN_SIZE as u32;
        let hc_mult = HC_MULT as u32;
        attention.metal.dispatch_batch(|batch| {
            batch.dispatch_threads(
                P7_MHC_POST_KERNEL,
                (HC_FLAT_WIDTH as u32, 1, 1),
                (self.post_threads, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&p6.moe_output_bf16), 0);
                    encoder.set_buffer(1, Some(attention.attention_hc_post_bf16), 0);
                    encoder.set_buffer(2, Some(&self.post_f32), 0);
                    encoder.set_buffer(3, Some(&self.comb_f32), 0);
                    encoder.set_buffer(4, Some(child_hc_state_bf16), 0);
                    set_u32(encoder, 5, &hidden);
                    set_u32(encoder, 6, &hc_mult);
                },
            )
        })
    }
}

fn validate_source_contract(source: &DeepSeekV4P7FfnSourceContract) -> Result<()> {
    if !is_supported_p7_identity(source.layer, source.token_id, source.token_position)
        || source.source_parent_retained
        || !source.source_upload_required_before_execution
        || source.host_activation_handoff_permitted
    {
        return Err(p7_device_error(
            "P7 device graph requires a bounded layer-0/BOS/position-0 or layer-0/token-19923/position-1 no-host source contract",
        ));
    }
    Ok(())
}

const fn is_supported_p7_identity(layer: usize, token_id: u32, token_position: usize) -> bool {
    layer == P7_LAYER
        && ((token_id == P7_POSITION0_TOKEN_ID && token_position == P7_POSITION0)
            || (token_id == P7_POSITION1_TOKEN_ID && token_position == P7_POSITION1))
}

fn validate_staged_tensor(
    staged: &DeepSeekV4StagedTensor,
    binding: &DeepSeekV4P7SourceTensorBinding,
    expected_dtype: &str,
    expected_shape: &[u64],
    expected_bytes: usize,
) -> Result<()> {
    if staged.name != binding.name
        || staged.dtype != expected_dtype
        || staged.dtype != binding.dtype
        || staged.shape.as_slice() != expected_shape
        || staged.shape != binding.shape
        || staged.bytes.len() != expected_bytes
        || binding.bytes != expected_bytes
        || sha256(&staged.bytes) != binding.sha256
    {
        return Err(p7_device_error(format!(
            "P7 static source control {} does not match its hash-bound geometry",
            binding.name
        )));
    }
    Ok(())
}

fn context_queue_identity(context: &MetalContext) -> usize {
    context.queue() as *const _ as usize
}

fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &u32) {
    encoder.set_bytes(
        index,
        size_of::<u32>() as u64,
        value as *const u32 as *const _,
    );
}

fn set_f32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &f32) {
    encoder.set_bytes(
        index,
        size_of::<f32>() as u64,
        value as *const f32 as *const _,
    );
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn p7_device_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 bounded P7 device graph: {}",
        message.into()
    ))
}
