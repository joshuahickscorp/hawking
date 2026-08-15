//! Same-runtime multi-layer full-layer chain encode (L0..Ln).
//!
//! Extends the proven L0+L1 46-dispatch path by appending subsequent full
//! layers (DeltaNet or GQA mixer prefix + MoE suffix) before the single shared
//! fence.  L0..L2 is three DeltaNet layers (69 dispatches).  L0..L3 crosses the
//! first GQA layer (layer 3) for 92 dispatches with one fence.
//!
//! Privacy: this submodule lives under `qwen80_complete_runtime` so it can
//! reach private helpers and encoder fields without widening the public API
//! surface of the L1 finalizer.

use super::*;
use crate::kernels::{
    mha_decode_f32_tcb, qwen_binary_sign_scale_matvec_component_tcb, qwen_next_add_residual_tcb,
    qwen_next_ba_to_decay_beta_tcb, qwen_next_deltanet_gated_rmsnorm_tcb,
    qwen_next_direct_packed_input_rmsnorm_tcb,
    qwen_next_gated_delta_decode_single_at_state_offset_tcb,
    qwen_next_qkvz_rearrange_conv_l2_tcb,
};
use crate::metal::{PinnedBuffer, TokenCommandBuffer};
use crate::model::qwen80_48_layer_execution_schedule::{
    qwen80_multi_layer_structural_kernel_trace, QWEN80_GQA_FULL_LAYER_DISPATCHES,
    QWEN80_GQA_MIXER_PREFIX_KERNELS, QWEN80_MIXER_PREFIX_DISPATCHES,
};

/// Local scalar-binding helper for GQA attention kernels (matches the
/// direct-packed attention stage / all-ten MoE graph pattern).
trait StageSetScalar {
    fn stage_set_u32(&self, index: u64, value: u32);
    fn stage_set_f32(&self, index: u64, value: f32);
}

impl StageSetScalar for ::metal::ComputeCommandEncoderRef {
    #[inline(always)]
    fn stage_set_u32(&self, index: u64, value: u32) {
        self.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }
    #[inline(always)]
    fn stage_set_f32(&self, index: u64, value: f32) {
        self.set_bytes(
            index,
            std::mem::size_of::<f32>() as u64,
            &value as *const f32 as *const _,
        );
    }
}

/// One subsequent (layer ≥ 2) DeltaNet mixer prefix living on the open TCB.
pub struct Qwen80SameRuntimeDeltaNetLayerPrefixEncoder {
    pub source_token_id: u32,
    pub layer: usize,
    pub linear_state_slot: usize,
    expected_input: Vec<f32>,
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
    dispatches_encoded: usize,
    expected_preceding_dispatches: usize,
}

impl Qwen80SameRuntimeDeltaNetLayerPrefixEncoder {
    pub fn first_residual(&self) -> &PinnedBuffer {
        &self.first_residual
    }

    pub fn expected_input(&self) -> &[f32] {
        &self.expected_input
    }

    pub fn dispatches_encoded(&self) -> usize {
        self.dispatches_encoded
    }

    pub fn derive_full_cpu_oracle(
        &self,
        runtime: &Qwen80CompleteNativeRuntime,
    ) -> Result<Qwen80CanonicalLinearMoECpuOracleResult> {
        let contract = runtime
            .catalog
            .canonical_linear_moe_operator_contract(self.layer)?;
        if contract.mixer.layer != self.layer
            || contract.mixer.linear_state_slot != self.linear_state_slot
        {
            return Err(model_error(format!(
                "same-runtime layer {} full CPU oracle did not bind expected layer/slot (observed layer={}, slot={}; expected layer={}, slot={})",
                self.layer,
                contract.mixer.layer,
                contract.mixer.linear_state_slot,
                self.layer,
                self.linear_state_slot
            )));
        }
        let input =
            Qwen80CanonicalLinearLayerCpuInput::with_zero_state(self.expected_input.clone());
        let cpu = runtime
            .catalog
            .execute_canonical_linear_moe_cpu_oracle(&contract, &input)?;
        Qwen80CompleteNativeRuntime::require_parity(
            &cpu.mixer.mixer_residual_output,
            &self.expected_first_residual,
            &format!("same-runtime layer {} full CPU oracle first residual", self.layer),
            0.0,
        )?;
        Ok(cpu)
    }

    fn verify_after_fence(
        &self,
        runtime: &Qwen80CompleteNativeRuntime,
    ) -> Result<Qwen80SameRuntimeLayer1DeltaNetPrefixParity> {
        let device_input = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.input,
            QWEN80_HIDDEN,
            &format!("same-runtime layer {} input", self.layer),
        )?;
        let input_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &self.expected_input,
            &device_input,
            &format!("same-runtime layer {} retained input", self.layer),
            1.0e-3,
        )?;
        let device_first_residual = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.first_residual,
            QWEN80_HIDDEN,
            &format!("same-runtime layer {} first residual", self.layer),
        )?;
        let first_residual_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &self.expected_first_residual,
            &device_first_residual,
            &format!("same-runtime layer {} first residual", self.layer),
            1.0e-3,
        )?;
        let device_conv_state = Qwen80CompleteNativeRuntime::device_f32_snapshot_at_offset(
            &runtime.linear_conv_state,
            self.conv_state_offset_elements,
            self.expected_next_conv_state.len(),
            &format!("same-runtime layer {} active convolution state", self.layer),
        )?;
        let device_recurrent_state = Qwen80CompleteNativeRuntime::device_f32_snapshot_at_offset(
            &runtime.linear_recurrent_state,
            self.recurrent_state_offset_elements,
            self.expected_next_recurrent_state.len(),
            &format!("same-runtime layer {} active recurrent state", self.layer),
        )?;
        let conv_state_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &self.expected_next_conv_state,
            &device_conv_state,
            &format!("same-runtime layer {} convolution state", self.layer),
            1.0e-3,
        )?;
        let recurrent_state_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &self.expected_next_recurrent_state,
            &device_recurrent_state,
            &format!("same-runtime layer {} recurrent state", self.layer),
            1.0e-3,
        )?;
        let rollback_conv = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.rollback_conv_state,
            self.expected_next_conv_state.len(),
            &format!("same-runtime layer {} rollback convolution", self.layer),
        )?;
        let rollback_recurrent = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.rollback_recurrent_state,
            self.expected_next_recurrent_state.len(),
            &format!("same-runtime layer {} rollback recurrent", self.layer),
        )?;
        Ok(Qwen80SameRuntimeLayer1DeltaNetPrefixParity {
            source_token_id: self.source_token_id,
            layer: self.layer,
            linear_state_slot: self.linear_state_slot,
            input_f32le_sha256: qwen80_f32_vector_sha256(
                &self.expected_input,
                &format!("same-runtime layer {} CPU input", self.layer),
            )?,
            device_input_f32le_sha256: qwen80_f32_vector_sha256(
                &device_input,
                &format!("same-runtime layer {} device input", self.layer),
            )?,
            input_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.input,
                &format!("same-runtime layer {} input buffer", self.layer),
            )?,
            cpu_first_residual_f32le_sha256: qwen80_f32_vector_sha256(
                &self.expected_first_residual,
                &format!("same-runtime layer {} CPU first residual", self.layer),
            )?,
            device_first_residual_f32le_sha256: qwen80_f32_vector_sha256(
                &device_first_residual,
                &format!("same-runtime layer {} device first residual", self.layer),
            )?,
            first_residual_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.first_residual,
                &format!("same-runtime layer {} first residual buffer", self.layer),
            )?,
            device_post_conv_state_f32le_sha256: qwen80_f32_vector_sha256(
                &device_conv_state,
                &format!("same-runtime layer {} active convolution state", self.layer),
            )?,
            device_post_recurrent_state_f32le_sha256: qwen80_f32_vector_sha256(
                &device_recurrent_state,
                &format!("same-runtime layer {} active recurrent state", self.layer),
            )?,
            rollback_conv_state_f32le_sha256: qwen80_f32_vector_sha256(
                &rollback_conv,
                &format!("same-runtime layer {} rollback convolution", self.layer),
            )?,
            rollback_recurrent_state_f32le_sha256: qwen80_f32_vector_sha256(
                &rollback_recurrent,
                &format!("same-runtime layer {} rollback recurrent", self.layer),
            )?,
            active_conv_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &runtime.linear_conv_state,
                &format!("same-runtime layer {} active convolution arena", self.layer),
            )?,
            active_recurrent_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &runtime.linear_recurrent_state,
                &format!("same-runtime layer {} active recurrent arena", self.layer),
            )?,
            rollback_conv_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.rollback_conv_state,
                &format!("same-runtime layer {} rollback convolution buffer", self.layer),
            )?,
            rollback_recurrent_state_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.rollback_recurrent_state,
                &format!("same-runtime layer {} rollback recurrent buffer", self.layer),
            )?,
            input_max_abs_error,
            first_residual_max_abs_error,
            conv_state_max_abs_error,
            recurrent_state_max_abs_error,
            first_residual_elements: QWEN80_HIDDEN,
            first_residual_bytes: bytes_for_f32(
                QWEN80_HIDDEN,
                &format!("same-runtime layer {} first residual bytes", self.layer),
            )?,
            conv_state_offset_elements: self.conv_state_offset_elements,
            conv_state_capacity_elements: self.conv_state_capacity_elements,
            recurrent_state_offset_elements: self.recurrent_state_offset_elements,
            recurrent_state_capacity_elements: self.recurrent_state_capacity_elements,
            required_l0_dispatches_before_prefix: self.expected_preceding_dispatches,
            total_dispatches_after_prefix: self
                .expected_preceding_dispatches
                .saturating_add(self.dispatches_encoded),
            same_runtime_same_command_buffer_required: true,
            dispatches_encoded: self.dispatches_encoded,
        })
    }
}

/// One layer's MoE suffix holder kept alive through the multi-layer fence.
pub struct Qwen80MultiLayerSuffixWitness {
    pub layer: usize,
    pub fixed: Qwen80L0TrueMoeFixedDeviceBuffers,
    pub cpu: Qwen80MultiLayerSuffixCpuOracle,
}

/// CPU oracle shadow for one full-layer MoE suffix (DeltaNet or GQA mixer).
#[derive(Clone, Debug)]
pub enum Qwen80MultiLayerSuffixCpuOracle {
    Linear(Qwen80CanonicalLinearMoECpuOracleResult),
    Gqa(Qwen80CanonicalGqaMoECpuOracleResult),
}

impl Qwen80MultiLayerSuffixCpuOracle {
    pub fn layer(&self) -> usize {
        match self {
            Self::Linear(cpu) => cpu.mixer.layer,
            Self::Gqa(cpu) => cpu.mixer.layer,
        }
    }

    pub fn route(&self) -> &Qwen80RouteSelection {
        match self {
            Self::Linear(cpu) => &cpu.route,
            Self::Gqa(cpu) => &cpu.route,
        }
    }

    pub fn routed_experts(&self) -> &[Qwen80RoutedExpertCpuOracleResult] {
        match self {
            Self::Linear(cpu) => &cpu.routed_experts,
            Self::Gqa(cpu) => &cpu.routed_experts,
        }
    }

    pub fn post_attention_rms_norm_output(&self) -> &[f32] {
        match self {
            Self::Linear(cpu) => &cpu.post_attention_rms_norm_output,
            Self::Gqa(cpu) => &cpu.post_attention_rms_norm_output,
        }
    }

    pub fn router_logits(&self) -> &[f32] {
        match self {
            Self::Linear(cpu) => &cpu.router_logits,
            Self::Gqa(cpu) => &cpu.router_logits,
        }
    }

    pub fn shared_gated_output(&self) -> &[f32] {
        match self {
            Self::Linear(cpu) => &cpu.shared_gated_output,
            Self::Gqa(cpu) => &cpu.shared_gated_output,
        }
    }

    pub fn routed_expert_sum(&self) -> &[f32] {
        match self {
            Self::Linear(cpu) => &cpu.routed_expert_sum,
            Self::Gqa(cpu) => &cpu.routed_expert_sum,
        }
    }

    pub fn layer_output(&self) -> &[f32] {
        match self {
            Self::Linear(cpu) => &cpu.layer_output,
            Self::Gqa(cpu) => &cpu.layer_output,
        }
    }
}

/// Subsequent (layer ≥ 2) mixer prefix on the open TCB: DeltaNet or GQA.
pub enum Qwen80SameRuntimeSubsequentLayerPrefix {
    DeltaNet(Qwen80SameRuntimeDeltaNetLayerPrefixEncoder),
    Gqa(Qwen80SameRuntimeGqaLayerPrefixEncoder),
}

impl Qwen80SameRuntimeSubsequentLayerPrefix {
    pub fn layer(&self) -> usize {
        match self {
            Self::DeltaNet(prefix) => prefix.layer,
            Self::Gqa(prefix) => prefix.layer,
        }
    }

    pub fn first_residual(&self) -> &PinnedBuffer {
        match self {
            Self::DeltaNet(prefix) => prefix.first_residual(),
            Self::Gqa(prefix) => prefix.first_residual(),
        }
    }

    pub fn expected_first_residual(&self) -> &[f32] {
        match self {
            Self::DeltaNet(prefix) => &prefix.expected_first_residual,
            Self::Gqa(prefix) => &prefix.expected_first_residual,
        }
    }

    fn verify_after_fence(
        &self,
        runtime: &Qwen80CompleteNativeRuntime,
    ) -> Result<Qwen80SameRuntimeLayerPrefixParity> {
        match self {
            Self::DeltaNet(prefix) => Ok(Qwen80SameRuntimeLayerPrefixParity::DeltaNet(
                prefix.verify_after_fence(runtime)?,
            )),
            Self::Gqa(prefix) => Ok(Qwen80SameRuntimeLayerPrefixParity::Gqa(
                prefix.verify_after_fence(runtime)?,
            )),
        }
    }
}

/// One subsequent GQA mixer prefix living on the open TCB.
pub struct Qwen80SameRuntimeGqaLayerPrefixEncoder {
    pub source_token_id: u32,
    pub layer: usize,
    pub full_attention_state_slot: usize,
    pub position: usize,
    expected_input: Vec<f32>,
    expected_first_residual: Vec<f32>,
    expected_key_row: Vec<f32>,
    expected_value_row: Vec<f32>,
    key_cache_offset_elements: usize,
    value_cache_offset_elements: usize,
    key_cache_capacity_elements: usize,
    value_cache_capacity_elements: usize,
    rollback_key_cache: PinnedBuffer,
    rollback_value_cache: PinnedBuffer,
    input: PinnedBuffer,
    _normalized: PinnedBuffer,
    _q_projection: PinnedBuffer,
    _k_projection: PinnedBuffer,
    _v_projection: PinnedBuffer,
    _query: PinnedBuffer,
    _attention: PinnedBuffer,
    _gated: PinnedBuffer,
    _mixer_output: PinnedBuffer,
    first_residual: PinnedBuffer,
    _input_layernorm: Qwen80GpuBinaryTensor,
    _q_proj: Qwen80GpuBinaryTensor,
    _k_proj: Qwen80GpuBinaryTensor,
    _v_proj: Qwen80GpuBinaryTensor,
    _q_norm: Qwen80GpuBinaryTensor,
    _k_norm: Qwen80GpuBinaryTensor,
    _o_proj: Qwen80GpuBinaryTensor,
    dispatches_encoded: usize,
    expected_preceding_dispatches: usize,
}

impl Qwen80SameRuntimeGqaLayerPrefixEncoder {
    pub fn first_residual(&self) -> &PinnedBuffer {
        &self.first_residual
    }

    pub fn expected_input(&self) -> &[f32] {
        &self.expected_input
    }

    pub fn dispatches_encoded(&self) -> usize {
        self.dispatches_encoded
    }

    pub fn derive_full_cpu_oracle(
        &self,
        runtime: &Qwen80CompleteNativeRuntime,
    ) -> Result<Qwen80CanonicalGqaMoECpuOracleResult> {
        let contract = runtime.catalog.canonical_gqa_moe_operator_contract(self.layer)?;
        if contract.mixer.layer != self.layer
            || contract.mixer.full_attention_state_slot != self.full_attention_state_slot
        {
            return Err(model_error(format!(
                "same-runtime GQA layer {} full CPU oracle did not bind expected layer/slot (observed layer={}, slot={}; expected layer={}, slot={})",
                self.layer,
                contract.mixer.layer,
                contract.mixer.full_attention_state_slot,
                self.layer,
                self.full_attention_state_slot
            )));
        }
        let input = Qwen80CanonicalGqaLayerCpuInput::with_zero_state_at_position(
            self.expected_input.clone(),
            self.position,
        )?;
        let cpu = runtime
            .catalog
            .execute_canonical_gqa_moe_cpu_oracle(&contract, &input)?;
        Qwen80CompleteNativeRuntime::require_parity(
            &cpu.mixer.mixer_residual_output,
            &self.expected_first_residual,
            &format!("same-runtime GQA layer {} full CPU oracle first residual", self.layer),
            0.0,
        )?;
        Ok(cpu)
    }

    fn verify_after_fence(
        &self,
        runtime: &Qwen80CompleteNativeRuntime,
    ) -> Result<Qwen80SameRuntimeGqaLayerPrefixParity> {
        let device_input = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.input,
            QWEN80_HIDDEN,
            &format!("same-runtime GQA layer {} input", self.layer),
        )?;
        let input_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &self.expected_input,
            &device_input,
            &format!("same-runtime GQA layer {} retained input", self.layer),
            1.0e-3,
        )?;
        let device_first_residual = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.first_residual,
            QWEN80_HIDDEN,
            &format!("same-runtime GQA layer {} first residual", self.layer),
        )?;
        let first_residual_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &self.expected_first_residual,
            &device_first_residual,
            &format!("same-runtime GQA layer {} first residual", self.layer),
            1.0e-3,
        )?;
        let device_key_row = Qwen80CompleteNativeRuntime::device_f32_snapshot_at_offset(
            &runtime.full_attention_key_cache,
            self.key_cache_offset_elements
                .checked_add(
                    self.position
                        .checked_mul(self.expected_key_row.len())
                        .ok_or_else(|| {
                            model_error(format!(
                                "same-runtime GQA layer {} key row offset overflowed",
                                self.layer
                            ))
                        })?,
                )
                .ok_or_else(|| {
                    model_error(format!(
                        "same-runtime GQA layer {} key row absolute offset overflowed",
                        self.layer
                    ))
                })?,
            self.expected_key_row.len(),
            &format!("same-runtime GQA layer {} active key row", self.layer),
        )?;
        let device_value_row = Qwen80CompleteNativeRuntime::device_f32_snapshot_at_offset(
            &runtime.full_attention_value_cache,
            self.value_cache_offset_elements
                .checked_add(
                    self.position
                        .checked_mul(self.expected_value_row.len())
                        .ok_or_else(|| {
                            model_error(format!(
                                "same-runtime GQA layer {} value row offset overflowed",
                                self.layer
                            ))
                        })?,
                )
                .ok_or_else(|| {
                    model_error(format!(
                        "same-runtime GQA layer {} value row absolute offset overflowed",
                        self.layer
                    ))
                })?,
            self.expected_value_row.len(),
            &format!("same-runtime GQA layer {} active value row", self.layer),
        )?;
        let key_cache_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &self.expected_key_row,
            &device_key_row,
            &format!("same-runtime GQA layer {} key cache row", self.layer),
            1.0e-3,
        )?;
        let value_cache_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &self.expected_value_row,
            &device_value_row,
            &format!("same-runtime GQA layer {} value cache row", self.layer),
            1.0e-3,
        )?;
        let rollback_key = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.rollback_key_cache,
            self.expected_key_row.len(),
            &format!("same-runtime GQA layer {} rollback key", self.layer),
        )?;
        let rollback_value = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &self.rollback_value_cache,
            self.expected_value_row.len(),
            &format!("same-runtime GQA layer {} rollback value", self.layer),
        )?;
        Ok(Qwen80SameRuntimeGqaLayerPrefixParity {
            source_token_id: self.source_token_id,
            layer: self.layer,
            full_attention_state_slot: self.full_attention_state_slot,
            position: self.position,
            input_f32le_sha256: qwen80_f32_vector_sha256(
                &self.expected_input,
                &format!("same-runtime GQA layer {} CPU input", self.layer),
            )?,
            device_input_f32le_sha256: qwen80_f32_vector_sha256(
                &device_input,
                &format!("same-runtime GQA layer {} device input", self.layer),
            )?,
            input_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.input,
                &format!("same-runtime GQA layer {} input buffer", self.layer),
            )?,
            cpu_first_residual_f32le_sha256: qwen80_f32_vector_sha256(
                &self.expected_first_residual,
                &format!("same-runtime GQA layer {} CPU first residual", self.layer),
            )?,
            device_first_residual_f32le_sha256: qwen80_f32_vector_sha256(
                &device_first_residual,
                &format!("same-runtime GQA layer {} device first residual", self.layer),
            )?,
            first_residual_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.first_residual,
                &format!("same-runtime GQA layer {} first residual buffer", self.layer),
            )?,
            device_key_row_f32le_sha256: qwen80_f32_vector_sha256(
                &device_key_row,
                &format!("same-runtime GQA layer {} active key row", self.layer),
            )?,
            device_value_row_f32le_sha256: qwen80_f32_vector_sha256(
                &device_value_row,
                &format!("same-runtime GQA layer {} active value row", self.layer),
            )?,
            rollback_key_cache_f32le_sha256: qwen80_f32_vector_sha256(
                &rollback_key,
                &format!("same-runtime GQA layer {} rollback key", self.layer),
            )?,
            rollback_value_cache_f32le_sha256: qwen80_f32_vector_sha256(
                &rollback_value,
                &format!("same-runtime GQA layer {} rollback value", self.layer),
            )?,
            active_key_cache_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &runtime.full_attention_key_cache,
                &format!("same-runtime GQA layer {} active key arena", self.layer),
            )?,
            active_value_cache_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &runtime.full_attention_value_cache,
                &format!("same-runtime GQA layer {} active value arena", self.layer),
            )?,
            rollback_key_cache_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.rollback_key_cache,
                &format!("same-runtime GQA layer {} rollback key buffer", self.layer),
            )?,
            rollback_value_cache_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                &self.rollback_value_cache,
                &format!("same-runtime GQA layer {} rollback value buffer", self.layer),
            )?,
            input_max_abs_error,
            first_residual_max_abs_error,
            key_cache_max_abs_error,
            value_cache_max_abs_error,
            first_residual_elements: QWEN80_HIDDEN,
            first_residual_bytes: bytes_for_f32(
                QWEN80_HIDDEN,
                &format!("same-runtime GQA layer {} first residual bytes", self.layer),
            )?,
            key_cache_offset_elements: self.key_cache_offset_elements,
            key_cache_capacity_elements: self.key_cache_capacity_elements,
            value_cache_offset_elements: self.value_cache_offset_elements,
            value_cache_capacity_elements: self.value_cache_capacity_elements,
            required_preceding_dispatches_before_prefix: self.expected_preceding_dispatches,
            total_dispatches_after_prefix: self
                .expected_preceding_dispatches
                .saturating_add(self.dispatches_encoded),
            same_runtime_same_command_buffer_required: true,
            dispatches_encoded: self.dispatches_encoded,
            frozen_mixer_prefix_kernel_names: QWEN80_GQA_MIXER_PREFIX_KERNELS
                .iter()
                .map(|name| (*name).to_owned())
                .collect(),
        })
    }
}

/// GQA mixer-prefix parity after the single multi-layer fence.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80SameRuntimeGqaLayerPrefixParity {
    pub source_token_id: u32,
    pub layer: usize,
    pub full_attention_state_slot: usize,
    pub position: usize,
    pub input_f32le_sha256: String,
    pub device_input_f32le_sha256: String,
    pub input_buffer_identity_sha256: String,
    pub cpu_first_residual_f32le_sha256: String,
    pub device_first_residual_f32le_sha256: String,
    pub first_residual_buffer_identity_sha256: String,
    pub device_key_row_f32le_sha256: String,
    pub device_value_row_f32le_sha256: String,
    pub rollback_key_cache_f32le_sha256: String,
    pub rollback_value_cache_f32le_sha256: String,
    pub active_key_cache_buffer_identity_sha256: String,
    pub active_value_cache_buffer_identity_sha256: String,
    pub rollback_key_cache_buffer_identity_sha256: String,
    pub rollback_value_cache_buffer_identity_sha256: String,
    pub input_max_abs_error: f32,
    pub first_residual_max_abs_error: f32,
    pub key_cache_max_abs_error: f32,
    pub value_cache_max_abs_error: f32,
    pub first_residual_elements: usize,
    pub first_residual_bytes: usize,
    pub key_cache_offset_elements: usize,
    pub key_cache_capacity_elements: usize,
    pub value_cache_offset_elements: usize,
    pub value_cache_capacity_elements: usize,
    pub required_preceding_dispatches_before_prefix: usize,
    pub total_dispatches_after_prefix: usize,
    pub same_runtime_same_command_buffer_required: bool,
    pub dispatches_encoded: usize,
    pub frozen_mixer_prefix_kernel_names: Vec<String>,
}

/// Per-layer mixer-prefix parity (DeltaNet or GQA) for the multi-layer receipt.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "mixer", rename_all = "snake_case")]
pub enum Qwen80SameRuntimeLayerPrefixParity {
    DeltaNet(Qwen80SameRuntimeLayer1DeltaNetPrefixParity),
    Gqa(Qwen80SameRuntimeGqaLayerPrefixParity),
}

impl Qwen80SameRuntimeLayerPrefixParity {
    pub fn retained_max_abs_error(&self) -> f32 {
        match self {
            Self::DeltaNet(p) => p
                .first_residual_max_abs_error
                .max(p.input_max_abs_error)
                .max(p.conv_state_max_abs_error)
                .max(p.recurrent_state_max_abs_error),
            Self::Gqa(p) => p
                .first_residual_max_abs_error
                .max(p.input_max_abs_error)
                .max(p.key_cache_max_abs_error)
                .max(p.value_cache_max_abs_error),
        }
    }
}

/// Receipt-ready multi-layer chain evidence after the single fence.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80SameRuntimeMultiLayerChainParity {
    pub layer_count: usize,
    pub total_dispatches: usize,
    pub structural_kernel_names: Vec<String>,
    pub fresh_l0: Qwen80SameRuntimeFreshL0Parity,
    pub per_layer_prefix: Vec<Qwen80SameRuntimeLayerPrefixParity>,
    pub per_layer_suffix: Vec<Qwen80SameRuntimeL1TrueMoeSuffixParity>,
    pub retained_max_abs_error: f32,
    pub same_runtime_same_command_buffer_required: bool,
    pub single_fence_after_all_dispatches_required: bool,
}

impl Qwen80CompleteNativeRuntime {
    /// Build an all-ten MoE source bridge for any DeltaNet layer from a
    /// caller-validated top-10 route (CPU oracle, not device).
    pub fn build_source_token_layer_all_ten_true_moe_source_bridge_from_route(
        &self,
        layer: usize,
        manifest_document_sha256: &str,
        plan_document_sha256: &str,
        route: Qwen80RouteSelection,
    ) -> Result<Qwen80AllTenTrueMoeSourceBridge> {
        require_canonical_sha256(
            manifest_document_sha256,
            &format!("source-token layer {layer} manifest document SHA-256"),
        )?;
        require_canonical_sha256(
            plan_document_sha256,
            &format!("source-token layer {layer} route/plan document SHA-256"),
        )?;
        route.validate()?;
        let catalog = &self.catalog;
        // DeltaNet or GQA: both full-layer same-runtime encodes are wired.
        // Do NOT hoist canonical_linear_moe_operator_contract here — it refuses
        // any non-LinearAttention layer and would make the GQA arm below dead.
        let hybrid = catalog.complete_hybrid_decoder_plan(1)?;
        let layer_plan = hybrid.layers.get(layer).ok_or_else(|| {
            model_error(format!(
                "source-token layer {layer} is outside hybrid plan (observed plan_len={})",
                hybrid.layers.len()
            ))
        })?;
        match &layer_plan.kind {
            Qwen80LayerKind::LinearAttention => {
                let contract = catalog.canonical_linear_moe_operator_contract(layer)?;
                contract.validate_against_catalog(catalog)?;
                if contract.mixer.layer != layer {
                    return Err(model_error(format!(
                        "source-token layer {layer} DeltaNet MoE contract layer drifted: observed={}, expected={layer}",
                        contract.mixer.layer
                    )));
                }
            }
            Qwen80LayerKind::FullAttention => {
                let contract = catalog.canonical_gqa_moe_operator_contract(layer)?;
                contract.validate_against_catalog(catalog)?;
                if contract.mixer.layer != layer {
                    return Err(model_error(format!(
                        "source-token layer {layer} GQA MoE contract layer drifted: observed={}, expected={layer}",
                        contract.mixer.layer
                    )));
                }
            }
        }
        let bindings = hybrid.routed_expert_bindings(catalog, layer, &route)?;
        if bindings.len() != QWEN80_TOP_K {
            return Err(model_error(format!(
                "source-token layer {layer} bridge did not resolve exactly ten routed experts (observed={}, expected={QWEN80_TOP_K})",
                bindings.len()
            )));
        }
        let mut waves = Vec::with_capacity(QWEN80_TOP_K);
        for (index, (expected_id, bindings)) in route.ids.iter().zip(bindings).enumerate() {
            if bindings.expert != usize::from(*expected_id) {
                return Err(model_error(format!(
                    "source-token layer {layer} bridge expert {index} observed={}, expected={}",
                    bindings.expert, expected_id
                )));
            }
            let projection =
                |binding: &Qwen80PackedTensorBinding| -> Result<Qwen80AllTenRoutePlanProjection> {
                    Ok(Qwen80AllTenRoutePlanProjection {
                        tensor_name: binding.name.clone(),
                        artifact_sha256: catalog
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
            plan_document_sha256: plan_document_sha256.to_owned(),
            manifest_seal_sha256: catalog.manifest_seal().to_owned(),
            source_revision: catalog.config.source_revision.clone(),
            layer,
            route,
            waves,
        };
        catalog.build_all_ten_true_moe_source_bridge(
            &plan,
            catalog.first_residual_device_binding(layer)?,
        )
    }

    /// Encode the nine-dispatch DeltaNet mixer prefix for layer ≥ 1 from a
    /// live previous-layer second residual (device allocation + CPU shadow).
    ///
    /// Does not hardcode total dispatch counts beyond `expected_preceding + 9`,
    /// so L2+ may follow a completed L1 suffix on the same open TCB.
    #[allow(clippy::too_many_lines)]
    pub fn encode_source_token_deltanet_prefix_from_previous_second_residual_into(
        &self,
        command: &mut TokenCommandBuffer<'_>,
        layer: usize,
        previous_second_residual: PinnedBuffer,
        previous_second_residual_cpu: &[f32],
        expected_preceding_dispatches: usize,
    ) -> Result<Qwen80SameRuntimeDeltaNetLayerPrefixEncoder> {
        const SOURCE_TOKEN_ID: u32 = 1;
        if layer == 0 {
            return Err(model_error(
                "deltanet prefix-from-previous refuses layer 0 (use the source-token L0 encoder); observed layer=0, expected layer>=1",
            ));
        }
        if command.dispatch_count() != expected_preceding_dispatches {
            return Err(model_error(format!(
                "same-runtime layer {layer} prefix requires exactly {expected_preceding_dispatches} preceding dispatches, observed {}",
                command.dispatch_count()
            )));
        }
        if previous_second_residual_cpu.len() != QWEN80_HIDDEN {
            return Err(model_error(format!(
                "same-runtime layer {layer} previous CPU second residual len observed={}, expected={QWEN80_HIDDEN}",
                previous_second_residual_cpu.len()
            )));
        }
        if previous_second_residual.length() as usize
            != bytes_for_f32(QWEN80_HIDDEN, "previous second residual bytes")?
        {
            return Err(model_error(format!(
                "same-runtime layer {layer} previous device second residual byte length observed={}, expected={}",
                previous_second_residual.length(),
                bytes_for_f32(QWEN80_HIDDEN, "previous second residual bytes")?
            )));
        }
        let contract = self
            .catalog
            .canonical_linear_deltanet_operator_contract(layer)?;
        contract.validate_device_resources(&contract.minimum_device_resources)?;
        let resources = &contract.minimum_device_resources;
        if contract.layer != layer {
            return Err(model_error(format!(
                "same-runtime layer {layer} prefix contract layer drifted: observed={}",
                contract.layer
            )));
        }
        let live_state = self.linear_deltanet_state_slot_device_binding(layer)?;
        if live_state.layer != contract.layer
            || live_state.linear_state_slot != contract.linear_state_slot
            || live_state.conv_state_offset_elements != resources.conv_state_offset_elements
            || live_state.recurrent_state_offset_elements
                != resources.recurrent_state_offset_elements
        {
            return Err(model_error(format!(
                "same-runtime layer {layer} active state arena drifted from its source-scheduled slot contract (observed slot={}, expected={})",
                live_state.linear_state_slot, contract.linear_state_slot
            )));
        }
        let cpu_input =
            Qwen80CanonicalLinearLayerCpuInput::with_zero_state(previous_second_residual_cpu.to_vec());
        cpu_input.validate()?;
        if cpu_input
            .state
            .conv_state
            .iter()
            .chain(cpu_input.state.recurrent_state.iter())
            .any(|value| value.to_bits() != 0)
        {
            return Err(model_error(format!(
                "same-runtime layer {layer} prefix requires a zeroed exclusive state slot (observed non-zero CPU pre-state)"
            )));
        }
        let layout = &contract.layout;
        let expected = self
            .catalog
            .execute_canonical_linear_deltanet_cpu_oracle(&contract, &cpu_input)?;
        if expected.layer != layer || expected.linear_state_slot != contract.linear_state_slot {
            return Err(model_error(format!(
                "same-runtime layer {layer} CPU oracle did not retain source layer/state slot (observed layer={}, slot={})",
                expected.layer, expected.linear_state_slot
            )));
        }

        let live_initial_conv_state = Self::device_f32_snapshot_at_offset(
            &self.linear_conv_state,
            resources.conv_state_offset_elements,
            cpu_input.state.conv_state.len(),
            &format!("same-runtime layer {layer} live initial convolution state"),
        )?;
        let live_initial_recurrent_state = Self::device_f32_snapshot_at_offset(
            &self.linear_recurrent_state,
            resources.recurrent_state_offset_elements,
            cpu_input.state.recurrent_state.len(),
            &format!("same-runtime layer {layer} live initial recurrent state"),
        )?;
        if live_initial_conv_state
            .iter()
            .chain(live_initial_recurrent_state.iter())
            .any(|value| value.to_bits() != 0)
        {
            return Err(model_error(format!(
                "same-runtime layer {layer} prefix refuses a reused non-zero state slot instead of a zeroed exclusive slot (honour rollback buffers)"
            )));
        }
        Self::require_parity(
            &cpu_input.state.conv_state,
            &live_initial_conv_state,
            &format!("same-runtime layer {layer} live initial convolution state"),
            0.0,
        )?;
        Self::require_parity(
            &cpu_input.state.recurrent_state,
            &live_initial_recurrent_state,
            &format!("same-runtime layer {layer} live initial recurrent state"),
            0.0,
        )?;

        let input_layernorm = self.upload_direct_tensor(&contract.input_layernorm.name)?;
        let qkvz = self.upload_direct_tensor(&contract.mixer.in_proj_qkvz.name)?;
        let ba = self.upload_direct_tensor(&contract.mixer.in_proj_ba.name)?;
        let conv = self.upload_direct_tensor(&contract.mixer.causal_conv1d.name)?;
        let a_log = self.upload_direct_tensor(&contract.mixer.a_log.name)?;
        let dt_bias = self.upload_direct_tensor(&contract.mixer.dt_bias.name)?;
        let gated_norm = self.upload_direct_tensor(&contract.mixer.gated_rms_norm.name)?;
        let out_proj = self.upload_direct_tensor(&contract.mixer.out_proj.name)?;

        let rollback_conv_state = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&live_initial_conv_state))?;
        let rollback_recurrent_state = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&live_initial_recurrent_state))?;

        let normalized = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            &format!("same-runtime layer {layer} input RMSNorm output"),
        )?)?;
        let qkvz_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.qkvz_projection_elements()?,
            &format!("same-runtime layer {layer} QKVZ projection"),
        )?)?;
        let ba_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.ba_projection_elements()?,
            &format!("same-runtime layer {layer} BA projection"),
        )?)?;
        let repeated_query = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            &format!("same-runtime layer {layer} repeated query"),
        )?)?;
        let repeated_key = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            &format!("same-runtime layer {layer} repeated key"),
        )?)?;
        let convolved_value = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            &format!("same-runtime layer {layer} convolved value"),
        )?)?;
        let z = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            &format!("same-runtime layer {layer} Z gate"),
        )?)?;
        let decay = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_heads,
            &format!("same-runtime layer {layer} decay"),
        )?)?;
        let beta = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_heads,
            &format!("same-runtime layer {layer} beta"),
        )?)?;
        let recurrent_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            &format!("same-runtime layer {layer} recurrent output"),
        )?)?;
        let gated_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.value_elements()?,
            &format!("same-runtime layer {layer} gated RMSNorm output"),
        )?)?;
        let mixer_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            &format!("same-runtime layer {layer} mixer output"),
        )?)?;
        let first_residual = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            &format!("same-runtime layer {layer} first residual"),
        )?)?;

        let dispatches_before = command.dispatch_count();
        qwen_next_direct_packed_input_rmsnorm_tcb(
            command,
            &previous_second_residual,
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
            &previous_second_residual,
            &mixer_output,
            &first_residual,
            layout.hidden_elements,
        )?;
        let dispatches_encoded = command
            .dispatch_count()
            .checked_sub(dispatches_before)
            .ok_or_else(|| {
                model_error(format!(
                    "same-runtime layer {layer} dispatch count underflowed"
                ))
            })?;
        if dispatches_encoded != 9 {
            return Err(model_error(format!(
                "same-runtime layer {layer} prefix encoded {dispatches_encoded} dispatches, expected=9"
            )));
        }
        let expected_total = expected_preceding_dispatches
            .checked_add(9)
            .ok_or_else(|| model_error("same-runtime multi-layer dispatch total overflowed"))?;
        if command.dispatch_count() != expected_total {
            return Err(model_error(format!(
                "same-runtime layer {layer} prefix produced {} total dispatches, expected={expected_total}",
                command.dispatch_count()
            )));
        }
        Ok(Qwen80SameRuntimeDeltaNetLayerPrefixEncoder {
            source_token_id: SOURCE_TOKEN_ID,
            layer: contract.layer,
            linear_state_slot: contract.linear_state_slot,
            expected_input: cpu_input.hidden,
            expected_first_residual: expected.mixer_residual_output,
            expected_next_conv_state: expected.next_state.conv_state,
            expected_next_recurrent_state: expected.next_state.recurrent_state,
            conv_state_offset_elements: resources.conv_state_offset_elements,
            conv_state_capacity_elements: resources.conv_state_capacity_elements,
            recurrent_state_offset_elements: resources.recurrent_state_offset_elements,
            recurrent_state_capacity_elements: resources.recurrent_state_capacity_elements,
            rollback_conv_state,
            rollback_recurrent_state,
            input: previous_second_residual,
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
            expected_preceding_dispatches,
        })
    }
}

impl Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder {
    /// Collect MoE suffix parity for any full layer (DeltaNet or GQA).
    pub(super) fn collect_layer_true_moe_suffix_parity_after_fence(
        fixed: &Qwen80L0TrueMoeFixedDeviceBuffers,
        cpu: &Qwen80MultiLayerSuffixCpuOracle,
        layer: usize,
    ) -> Result<Qwen80SameRuntimeL1TrueMoeSuffixParity> {
        if fixed.contract.layer() != layer
            || cpu.layer() != layer
            || cpu.route().ids.len() != QWEN80_TOP_K
            || cpu.route().weights.len() != QWEN80_TOP_K
            || cpu.routed_experts().len() != QWEN80_TOP_K
        {
            return Err(model_error(format!(
                "same-runtime layer {layer} suffix readback lost its exact layer/top-ten contract (fixed.layer={}, cpu.layer={}, route_ids={}, route_weights={}, experts={})",
                fixed.contract.layer(),
                cpu.layer(),
                cpu.route().ids.len(),
                cpu.route().weights.len(),
                cpu.routed_experts().len()
            )));
        }
        let postnorm = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &fixed.postnorm_hidden,
            QWEN80_HIDDEN,
            &format!("same-runtime layer {layer} postnorm"),
        )?;
        let router_logits = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &fixed.router_logits,
            QWEN80_EXPERTS,
            &format!("same-runtime layer {layer} router logits"),
        )?;
        let observed_route_ids = Qwen80CompleteNativeRuntime::device_u32_snapshot(
            &fixed.router_route_ids,
            QWEN80_TOP_K,
            &format!("same-runtime layer {layer} router IDs"),
        )?;
        let observed_route_weights = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &fixed.router_route_weights,
            QWEN80_TOP_K,
            &format!("same-runtime layer {layer} router weights"),
        )?;
        let route_guard = Qwen80CompleteNativeRuntime::device_u32_snapshot(
            &fixed.route_guard,
            1,
            &format!("same-runtime layer {layer} route guard"),
        )?[0];
        let expected_route_ids = cpu
            .route()
            .ids
            .iter()
            .copied()
            .map(u32::from)
            .collect::<Vec<_>>();
        if route_guard != 1 || observed_route_ids != expected_route_ids {
            return Err(model_error(format!(
                "same-runtime layer {layer} route guard/readback differs from the CPU-oracle top-10 route (route_guard observed={route_guard}, expected=1; route_ids observed={observed_route_ids:?}, expected={expected_route_ids:?})"
            )));
        }
        let postnorm_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            cpu.post_attention_rms_norm_output(),
            &postnorm,
            &format!("same-runtime layer {layer} postnorm"),
            2.0e-4,
        )?;
        let router_logits_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            cpu.router_logits(),
            &router_logits,
            &format!("same-runtime layer {layer} router logits"),
            5.0e-4,
        )?;
        let route_weights_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            &cpu.route().weights,
            &observed_route_weights,
            &format!("same-runtime layer {layer} router weights"),
            2.0e-5,
        )?;
        let route_weighted = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &fixed.route_weighted,
            QWEN80_TOP_K.checked_mul(QWEN80_HIDDEN).ok_or_else(|| {
                model_error(format!(
                    "same-runtime layer {layer} route output geometry overflowed"
                ))
            })?,
            &format!("same-runtime layer {layer} weighted route outputs"),
        )?;
        let mut all_ten_route_witnesses = Vec::with_capacity(QWEN80_TOP_K);
        for (wave_index, expected) in cpu.routed_experts().iter().enumerate() {
            let start = wave_index.checked_mul(QWEN80_HIDDEN).ok_or_else(|| {
                model_error(format!(
                    "same-runtime layer {layer} route offset overflowed"
                ))
            })?;
            let end = start.checked_add(QWEN80_HIDDEN).ok_or_else(|| {
                model_error(format!(
                    "same-runtime layer {layer} route range overflowed"
                ))
            })?;
            let observed = route_weighted.get(start..end).ok_or_else(|| {
                model_error(format!(
                    "same-runtime layer {layer} route output is truncated"
                ))
            })?;
            let max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
                &expected.weighted_output,
                observed,
                &format!("same-runtime layer {layer} weighted route {wave_index}"),
                3.0e-4,
            )?;
            all_ten_route_witnesses.push(Qwen80SameRuntimeL1RoutedWaveParity {
                wave_index,
                expert_id: expected.expert,
                normalized_weight: expected.route_weight,
                cpu_output_f32le_sha256: qwen80_f32_vector_sha256(
                    &expected.weighted_output,
                    &format!("same-runtime layer {layer} CPU weighted route {wave_index}"),
                )?,
                device_output_f32le_sha256: qwen80_f32_vector_sha256(
                    observed,
                    &format!("same-runtime layer {layer} weighted route {wave_index}"),
                )?,
                max_abs_error,
            });
        }
        let shared = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &fixed.gated_shared,
            QWEN80_HIDDEN,
            &format!("same-runtime layer {layer} shared output"),
        )?;
        let routed_sum = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &fixed.routed_sum,
            QWEN80_HIDDEN,
            &format!("same-runtime layer {layer} routed sum"),
        )?;
        let second_residual = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            &fixed.second_residual,
            QWEN80_HIDDEN,
            &format!("same-runtime layer {layer} second residual"),
        )?;
        let shared_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            cpu.shared_gated_output(),
            &shared,
            &format!("same-runtime layer {layer} shared output"),
            3.0e-4,
        )?;
        let routed_sum_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            cpu.routed_expert_sum(),
            &routed_sum,
            &format!("same-runtime layer {layer} routed sum"),
            3.0e-5,
        )?;
        let second_residual_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            cpu.layer_output(),
            &second_residual,
            &format!("same-runtime layer {layer} second residual"),
            1.0e-3,
        )?;
        Ok(Qwen80SameRuntimeL1TrueMoeSuffixParity {
            layer,
            linear_state_slot: fixed.contract.state_slot(),
            route_guard,
            observed_route_ids,
            expected_route_ids: cpu.route().ids.to_vec(),
            observed_route_weights,
            expected_route_weights: cpu.route().weights.to_vec(),
            route_weights_max_abs_error,
            postnorm_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                cpu.post_attention_rms_norm_output(),
                &format!("same-runtime layer {layer} CPU postnorm"),
            )?,
            postnorm_output_f32le_sha256: qwen80_f32_vector_sha256(
                &postnorm,
                &format!("same-runtime layer {layer} postnorm"),
            )?,
            postnorm_max_abs_error,
            router_logits_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                cpu.router_logits(),
                &format!("same-runtime layer {layer} CPU router logits"),
            )?,
            router_logits_output_f32le_sha256: qwen80_f32_vector_sha256(
                &router_logits,
                &format!("same-runtime layer {layer} router logits"),
            )?,
            router_logits_max_abs_error,
            all_ten_route_witnesses,
            shared_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                cpu.shared_gated_output(),
                &format!("same-runtime layer {layer} CPU shared output"),
            )?,
            shared_output_f32le_sha256: qwen80_f32_vector_sha256(
                &shared,
                &format!("same-runtime layer {layer} shared output"),
            )?,
            shared_max_abs_error,
            routed_sum_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                cpu.routed_expert_sum(),
                &format!("same-runtime layer {layer} CPU routed sum"),
            )?,
            routed_sum_output_f32le_sha256: qwen80_f32_vector_sha256(
                &routed_sum,
                &format!("same-runtime layer {layer} routed sum"),
            )?,
            routed_sum_max_abs_error,
            second_residual_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                cpu.layer_output(),
                &format!("same-runtime layer {layer} CPU second residual"),
            )?,
            second_residual_output_f32le_sha256: qwen80_f32_vector_sha256(
                &second_residual,
                &format!("same-runtime layer {layer} second residual"),
            )?,
            second_residual_max_abs_error,
        })
    }

    /// Consume the L0→L1 prefix owner after L1 MoE + any subsequent full
    /// layers (DeltaNet or GQA) have been appended.  One fence; parity against
    /// per-layer CPU oracles (never device-vs-device).
    ///
    /// `layer_suffixes` must start with the Layer-1 witness and then L2.. in
    /// order.  `subsequent_prefixes` holds the layer≥2 mixer-prefix encoders
    /// in the same order as suffixes[1..].
    pub fn finalize_after_exact_multi_layer_deltanet_chain_fence_with_readbacks(
        self,
        runtime: &Qwen80CompleteNativeRuntime,
        command: TokenCommandBuffer<'_>,
        layer_suffixes: &[Qwen80MultiLayerSuffixWitness],
        subsequent_prefixes: &[Qwen80SameRuntimeSubsequentLayerPrefix],
    ) -> Result<Qwen80SameRuntimeMultiLayerChainParity> {
        const L0_DISPATCHES: usize = QWEN80_SOURCE_TOKEN_L0_TRUE_MOE_KERNELS.len();
        const PER_LAYER: usize = 23;
        if layer_suffixes.is_empty() {
            return Err(model_error(
                "multi-layer finalizer requires at least the Layer-1 suffix witness (observed layer_suffixes len=0, expected>=1)",
            ));
        }
        // layer_count = L0 + len(suffixes) where suffixes cover L1..L(n-1)
        let layer_count = layer_suffixes
            .len()
            .checked_add(1)
            .ok_or_else(|| model_error("multi-layer layer_count overflowed"))?;
        let expected_total = layer_count
            .checked_mul(PER_LAYER)
            .ok_or_else(|| model_error("multi-layer total dispatch overflowed"))?;
        if subsequent_prefixes.len() != layer_suffixes.len().saturating_sub(1) {
            return Err(model_error(format!(
                "multi-layer subsequent_prefixes len observed={}, expected={} (one per layer>=2)",
                subsequent_prefixes.len(),
                layer_suffixes.len().saturating_sub(1)
            )));
        }
        if self.dispatches_encoded != QWEN80_SOURCE_TOKEN_L1_DELTA_PREFIX_KERNELS.len()
            || command.dispatch_count() != expected_total
        {
            return Err(model_error(format!(
                "multi-layer finalizer requires L1 prefix dispatches={} and total={expected_total} for layer_count={layer_count}, observed prefix={} total={}",
                QWEN80_SOURCE_TOKEN_L1_DELTA_PREFIX_KERNELS.len(),
                self.dispatches_encoded,
                command.dispatch_count()
            )));
        }
        for (index, witness) in layer_suffixes.iter().enumerate() {
            let expected_layer = index + 1;
            if witness.layer != expected_layer {
                return Err(model_error(format!(
                    "multi-layer suffix[{index}].layer observed={}, expected={expected_layer}",
                    witness.layer
                )));
            }
            if witness.fixed.contract.layer() != expected_layer {
                return Err(model_error(format!(
                    "multi-layer suffix[{index}] fixed contract layer observed={}, expected={expected_layer}",
                    witness.fixed.contract.layer()
                )));
            }
        }
        for (index, prefix) in subsequent_prefixes.iter().enumerate() {
            let expected_layer = index + 2;
            if prefix.layer() != expected_layer {
                return Err(model_error(format!(
                    "multi-layer subsequent_prefix[{index}].layer observed={}, expected={expected_layer}",
                    prefix.layer()
                )));
            }
        }
        self.l0_continuation.runtime_state_arena_owner.require_runtime_owner(
            runtime,
            "multi-layer finalizer refuses a runtime other than the opaque continuation owner",
        )?;
        self.l0_continuation.require_l0_trace()?;

        // Structural kernel trace from the 48-layer execution schedule authority
        // (DeltaNet and GQA full-layer tables concatenated in layer order).
        let expected_kernels = qwen80_multi_layer_structural_kernel_trace(layer_count, false)
            .map_err(|error| {
                model_error(format!(
                    "multi-layer finalizer schedule kernel trace refused: {error}"
                ))
            })?;
        if expected_kernels.len() != expected_total {
            return Err(model_error(format!(
                "multi-layer finalizer schedule kernel count drifted: observed={}, expected={expected_total}",
                expected_kernels.len()
            )));
        }
        qwen80_require_exact_structural_kernel_trace(
            command.structural_kernel_names(),
            &expected_kernels,
            &format!("same-runtime multi-layer L0..L{} finalizer", layer_count - 1),
        )?;
        let structural_kernel_names = command
            .structural_kernel_names()
            .expect("checked exact structural trace")
            .to_vec();

        command.commit_and_wait().map_err(|error| {
            model_error(format!(
                "same-runtime multi-layer one-command-buffer fence failed: {error}"
            ))
        })?;

        // L0 second residual vs CPU oracle.
        let l0_second_residual = Qwen80CompleteNativeRuntime::device_f32_snapshot(
            self.l0_continuation.l0_second_residual(),
            QWEN80_HIDDEN,
            "same-runtime multi-layer fresh L0 second residual",
        )?;
        let l0_second_residual_max_abs_error = Qwen80CompleteNativeRuntime::require_parity(
            self.l0_continuation.cpu_l0_layer_output(),
            &l0_second_residual,
            "same-runtime multi-layer fresh L0 second residual",
            1.0e-3,
        )?;
        let fresh_l0 = Qwen80SameRuntimeFreshL0Parity {
            first_residual: self
                .l0_continuation
                .first_residual
                .verify_after_fence(runtime)?,
            second_residual_cpu_f32le_sha256: qwen80_f32_vector_sha256(
                self.l0_continuation.cpu_l0_layer_output(),
                "same-runtime multi-layer fresh L0 CPU second residual",
            )?,
            second_residual_device_f32le_sha256: qwen80_f32_vector_sha256(
                &l0_second_residual,
                "same-runtime multi-layer fresh L0 device second residual",
            )?,
            second_residual_buffer_identity_sha256: qwen80_pinned_buffer_identity_sha256(
                self.l0_continuation.l0_second_residual(),
                "same-runtime multi-layer fresh L0 second residual",
            )?,
            second_residual_max_abs_error: l0_second_residual_max_abs_error,
        };

        let mut per_layer_prefix = Vec::with_capacity(layer_count - 1);
        let mut per_layer_suffix = Vec::with_capacity(layer_count - 1);
        let mut retained_max = l0_second_residual_max_abs_error;

        // Layer 1 prefix + suffix
        let l1_prefix = self.verify_after_fence(runtime)?;
        retained_max = retained_max
            .max(l1_prefix.first_residual_max_abs_error)
            .max(l1_prefix.input_max_abs_error);
        let l1_suffix = Self::collect_layer_true_moe_suffix_parity_after_fence(
            &layer_suffixes[0].fixed,
            &layer_suffixes[0].cpu,
            1,
        )?;
        retained_max = retained_max.max(l1_suffix.second_residual_max_abs_error);
        per_layer_prefix.push(Qwen80SameRuntimeLayerPrefixParity::DeltaNet(l1_prefix));
        per_layer_suffix.push(l1_suffix);

        // Layers ≥ 2 (DeltaNet or GQA)
        for (index, prefix) in subsequent_prefixes.iter().enumerate() {
            let prefix_parity = prefix.verify_after_fence(runtime)?;
            retained_max = retained_max.max(prefix_parity.retained_max_abs_error());
            let suffix = Self::collect_layer_true_moe_suffix_parity_after_fence(
                &layer_suffixes[index + 1].fixed,
                &layer_suffixes[index + 1].cpu,
                prefix.layer(),
            )?;
            retained_max = retained_max.max(suffix.second_residual_max_abs_error);
            per_layer_prefix.push(prefix_parity);
            per_layer_suffix.push(suffix);
        }

        let _ = L0_DISPATCHES; // documented constant retained for readers
        let _ = QWEN80_GQA_FULL_LAYER_DISPATCHES;
        Ok(Qwen80SameRuntimeMultiLayerChainParity {
            layer_count,
            total_dispatches: expected_total,
            structural_kernel_names,
            fresh_l0,
            per_layer_prefix,
            per_layer_suffix,
            retained_max_abs_error: retained_max,
            same_runtime_same_command_buffer_required: true,
            single_fence_after_all_dispatches_required: true,
        })
    }
}

impl Qwen80CompleteNativeRuntime {
    /// Encode the nine-dispatch GQA mixer prefix for a full-attention layer
    /// from a live previous-layer second residual (device allocation + CPU shadow).
    ///
    /// Honour the exclusive caller-owned GQA KV state slot: refuse non-zero
    /// active cache at the slot offset, snapshot into rollback buffers, and
    /// write only into this layer's arena slice.  Position is the source-token
    /// decode slot (0 for the first multi-layer chain position).
    #[allow(clippy::too_many_lines)]
    pub fn encode_source_token_gqa_prefix_from_previous_second_residual_into(
        &self,
        command: &mut TokenCommandBuffer<'_>,
        layer: usize,
        previous_second_residual: PinnedBuffer,
        previous_second_residual_cpu: &[f32],
        expected_preceding_dispatches: usize,
        position: usize,
    ) -> Result<Qwen80SameRuntimeGqaLayerPrefixEncoder> {
        const SOURCE_TOKEN_ID: u32 = 1;
        if command.dispatch_count() != expected_preceding_dispatches {
            return Err(model_error(format!(
                "same-runtime GQA layer {layer} prefix requires exactly {expected_preceding_dispatches} preceding dispatches, observed {}",
                command.dispatch_count()
            )));
        }
        if previous_second_residual_cpu.len() != QWEN80_HIDDEN {
            return Err(model_error(format!(
                "same-runtime GQA layer {layer} previous CPU second residual len observed={}, expected={QWEN80_HIDDEN}",
                previous_second_residual_cpu.len()
            )));
        }
        if previous_second_residual.length() as usize
            != bytes_for_f32(QWEN80_HIDDEN, "previous second residual bytes")?
        {
            return Err(model_error(format!(
                "same-runtime GQA layer {layer} previous device second residual byte length observed={}, expected={}",
                previous_second_residual.length(),
                bytes_for_f32(QWEN80_HIDDEN, "previous second residual bytes")?
            )));
        }
        let contract = self.catalog.canonical_gqa_operator_contract(layer)?;
        contract.validate_against_catalog(&self.catalog)?;
        if contract.layer != layer {
            return Err(model_error(format!(
                "same-runtime GQA layer {layer} prefix contract layer drifted: observed={}",
                contract.layer
            )));
        }
        let layout = &contract.layout;
        let resources = Qwen80CanonicalGqaDeviceResources::for_max_seq_len(
            layout,
            contract.full_attention_state_slot,
            self.state.max_seq_len,
            contract.minimum_device_resources.direct_packed_payload_bytes.clone(),
        )?;
        if position >= resources.max_seq_len {
            return Err(model_error(format!(
                "same-runtime GQA layer {layer} position observed={position}, expected in 0..{} (max_seq_len)",
                resources.max_seq_len
            )));
        }
        let cpu_input = Qwen80CanonicalGqaLayerCpuInput::with_zero_state_at_position(
            previous_second_residual_cpu.to_vec(),
            position,
        )?;
        cpu_input.validate()?;
        if cpu_input
            .state
            .key_cache
            .iter()
            .chain(cpu_input.state.value_cache.iter())
            .any(|value| value.to_bits() != 0)
        {
            return Err(model_error(format!(
                "same-runtime GQA layer {layer} prefix requires a zeroed exclusive KV slot (observed non-zero CPU pre-state)"
            )));
        }
        let expected = self
            .catalog
            .execute_canonical_gqa_cpu_oracle(&contract, &cpu_input)?;
        if expected.layer != layer
            || expected.full_attention_state_slot != contract.full_attention_state_slot
        {
            return Err(model_error(format!(
                "same-runtime GQA layer {layer} CPU oracle did not retain source layer/state slot (observed layer={}, slot={})",
                expected.layer, expected.full_attention_state_slot
            )));
        }

        // Active slot must be zeroed before encode (exclusive caller-owned slot).
        let live_initial_key = Self::device_f32_snapshot_at_offset(
            &self.full_attention_key_cache,
            resources.key_cache_offset_elements,
            resources.key_cache_capacity_elements,
            &format!("same-runtime GQA layer {layer} live initial key cache slot"),
        )?;
        let live_initial_value = Self::device_f32_snapshot_at_offset(
            &self.full_attention_value_cache,
            resources.value_cache_offset_elements,
            resources.value_cache_capacity_elements,
            &format!("same-runtime GQA layer {layer} live initial value cache slot"),
        )?;
        if live_initial_key
            .iter()
            .chain(live_initial_value.iter())
            .any(|value| value.to_bits() != 0)
        {
            return Err(model_error(format!(
                "same-runtime GQA layer {layer} prefix refuses a reused non-zero KV slot instead of a zeroed exclusive slot (honour gqa_key_cache_rollback / gqa_value_cache_rollback; observed non-zero active slot)"
            )));
        }

        let input_layernorm = self.upload_direct_tensor(&contract.input_layernorm.name)?;
        let q_proj = self.upload_direct_tensor(&contract.mixer.q_proj.name)?;
        let k_proj = self.upload_direct_tensor(&contract.mixer.k_proj.name)?;
        let v_proj = self.upload_direct_tensor(&contract.mixer.v_proj.name)?;
        let q_norm = self.upload_direct_tensor(&contract.mixer.q_norm.name)?;
        let k_norm = self.upload_direct_tensor(&contract.mixer.k_norm.name)?;
        let o_proj = self.upload_direct_tensor(&contract.mixer.o_proj.name)?;

        // Rollback is the pre-encode (zero) key/value *row* for this position.
        let rollback_key_cache = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&expected.key_row.iter().map(|_| 0.0f32).collect::<Vec<_>>()))?;
        let rollback_value_cache = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&expected.value_row.iter().map(|_| 0.0f32).collect::<Vec<_>>()))?;

        let normalized = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            &format!("same-runtime GQA layer {layer} input RMSNorm output"),
        )?)?;
        let q_projection = self.context.new_buffer_checked(bytes_for_f32(
            layout.q_proj_rows,
            &format!("same-runtime GQA layer {layer} q projection"),
        )?)?;
        let k_projection = self.context.new_buffer_checked(bytes_for_f32(
            layout.kv_dim,
            &format!("same-runtime GQA layer {layer} k projection"),
        )?)?;
        let v_projection = self.context.new_buffer_checked(bytes_for_f32(
            layout.kv_dim,
            &format!("same-runtime GQA layer {layer} v projection"),
        )?)?;
        let query = self.context.new_buffer_checked(bytes_for_f32(
            layout.query_dim,
            &format!("same-runtime GQA layer {layer} normalized query"),
        )?)?;
        let attention = self.context.new_buffer_checked(bytes_for_f32(
            layout.query_dim,
            &format!("same-runtime GQA layer {layer} causal GQA output"),
        )?)?;
        let gated = self.context.new_buffer_checked(bytes_for_f32(
            layout.query_dim,
            &format!("same-runtime GQA layer {layer} gated attention"),
        )?)?;
        let mixer_output = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            &format!("same-runtime GQA layer {layer} mixer output"),
        )?)?;
        let first_residual = self.context.new_buffer_checked(bytes_for_f32(
            layout.hidden_elements,
            &format!("same-runtime GQA layer {layer} first residual"),
        )?)?;

        let key_slot_byte_offset = resources
            .key_cache_offset_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| model_error("GQA key slot byte offset overflowed"))?;
        let value_slot_byte_offset = resources
            .value_cache_offset_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| model_error("GQA value slot byte offset overflowed"))?;

        let dispatches_before = command.dispatch_count();
        qwen_next_direct_packed_input_rmsnorm_tcb(
            command,
            &previous_second_residual,
            &input_layernorm.signs,
            &input_layernorm.scales,
            &normalized,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
            QWEN80_RMS_EPS,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            command,
            &q_proj.signs,
            &q_proj.scales,
            &normalized,
            &q_projection,
            layout.q_proj_rows,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            command,
            &k_proj.signs,
            &k_proj.scales,
            &normalized,
            &k_projection,
            layout.kv_dim,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            command,
            &v_proj.signs,
            &v_proj.scales,
            &normalized,
            &v_projection,
            layout.kv_dim,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
        )?;
        command.dispatch_threads(
            "qwen80_attention_qk_norm_rope_cache",
            (layout.query_heads as u32, 1, 1),
            (layout.query_heads as u32, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&q_projection), 0);
                encoder.set_buffer(1, Some(&k_projection), 0);
                encoder.set_buffer(2, Some(&v_projection), 0);
                encoder.set_buffer(3, Some(&q_norm.signs), 0);
                encoder.set_buffer(4, Some(&q_norm.scales), 0);
                encoder.set_buffer(5, Some(&k_norm.signs), 0);
                encoder.set_buffer(6, Some(&k_norm.scales), 0);
                encoder.set_buffer(7, Some(&query), 0);
                encoder.set_buffer(
                    8,
                    Some(&self.full_attention_key_cache),
                    key_slot_byte_offset as u64,
                );
                encoder.set_buffer(
                    9,
                    Some(&self.full_attention_value_cache),
                    value_slot_byte_offset as u64,
                );
                encoder.stage_set_u32(10, position as u32);
                encoder.stage_set_u32(11, layout.query_heads as u32);
                encoder.stage_set_u32(12, layout.key_value_heads as u32);
                encoder.stage_set_u32(13, layout.head_dim as u32);
                encoder.stage_set_u32(14, layout.rotary_dim as u32);
                encoder.stage_set_u32(15, layout.group_size as u32);
                encoder.stage_set_f32(16, f32::from_bits(layout.rope_theta_bits));
                encoder.stage_set_f32(17, f32::from_bits(layout.rms_eps_bits));
            },
        )?;
        mha_decode_f32_tcb(
            command,
            &query,
            &self.full_attention_key_cache,
            key_slot_byte_offset,
            &self.full_attention_value_cache,
            value_slot_byte_offset,
            &attention,
            position + 1,
            layout.head_dim,
            layout.query_heads,
            layout.key_value_heads,
        )?;
        command.dispatch_threads(
            "qwen80_attention_apply_sigmoid_gate",
            (layout.query_dim as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&attention), 0);
                encoder.set_buffer(1, Some(&q_projection), 0);
                encoder.set_buffer(2, Some(&gated), 0);
                encoder.stage_set_u32(3, layout.query_dim as u32);
                encoder.stage_set_u32(4, layout.head_dim as u32);
            },
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            command,
            &o_proj.signs,
            &o_proj.scales,
            &gated,
            &mixer_output,
            layout.hidden_elements,
            layout.query_dim,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_add_residual_tcb(
            command,
            &previous_second_residual,
            &mixer_output,
            &first_residual,
            layout.hidden_elements,
        )?;
        let dispatches_encoded = command
            .dispatch_count()
            .checked_sub(dispatches_before)
            .ok_or_else(|| {
                model_error(format!(
                    "same-runtime GQA layer {layer} dispatch count underflowed"
                ))
            })?;
        if dispatches_encoded != QWEN80_MIXER_PREFIX_DISPATCHES {
            return Err(model_error(format!(
                "same-runtime GQA layer {layer} prefix encoded {dispatches_encoded} dispatches, expected={QWEN80_MIXER_PREFIX_DISPATCHES}"
            )));
        }
        let expected_total = expected_preceding_dispatches
            .checked_add(QWEN80_MIXER_PREFIX_DISPATCHES)
            .ok_or_else(|| model_error("same-runtime multi-layer GQA dispatch total overflowed"))?;
        if command.dispatch_count() != expected_total {
            return Err(model_error(format!(
                "same-runtime GQA layer {layer} prefix produced {} total dispatches, expected={expected_total}",
                command.dispatch_count()
            )));
        }
        Ok(Qwen80SameRuntimeGqaLayerPrefixEncoder {
            source_token_id: SOURCE_TOKEN_ID,
            layer: contract.layer,
            full_attention_state_slot: contract.full_attention_state_slot,
            position,
            expected_input: cpu_input.hidden,
            expected_first_residual: expected.mixer_residual_output,
            expected_key_row: expected.key_row,
            expected_value_row: expected.value_row,
            key_cache_offset_elements: resources.key_cache_offset_elements,
            value_cache_offset_elements: resources.value_cache_offset_elements,
            key_cache_capacity_elements: resources.key_cache_capacity_elements,
            value_cache_capacity_elements: resources.value_cache_capacity_elements,
            rollback_key_cache,
            rollback_value_cache,
            input: previous_second_residual,
            _normalized: normalized,
            _q_projection: q_projection,
            _k_projection: k_projection,
            _v_projection: v_projection,
            _query: query,
            _attention: attention,
            _gated: gated,
            _mixer_output: mixer_output,
            first_residual,
            _input_layernorm: input_layernorm,
            _q_proj: q_proj,
            _k_proj: k_proj,
            _v_proj: v_proj,
            _q_norm: q_norm,
            _k_norm: k_norm,
            _o_proj: o_proj,
            dispatches_encoded,
            expected_preceding_dispatches,
        })
    }
}
