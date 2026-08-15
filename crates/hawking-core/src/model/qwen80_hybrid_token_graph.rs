//! Composed Qwen3-Next native token graph.
//!
//! This module wires the nine hybrid operator groups onto the already-admitted
//! Qwen80 shaders.  It follows the Q30 device-expert-table token-graph shape
//! (one command buffer per token, per-layer wave, device-resident state and
//! AR feedback) adapted to the 36× DeltaNet / 12× gated-GQA schedule, top-10
//! routed experts, and the shared expert.
//!
//! It does not invent new math, change weights/seals/ceilings, first-generate
//! on the 0.796 body, load the 74,391-tensor catalog, or claim a functional
//! generate.  Those remain a later wave.

use super::{Qwen80HybridNativeOperatorGap, QWEN80_HYBRID_NATIVE_OPERATOR_GAPS};
use crate::model::qwen80_48_layer_execution_schedule::{
    qwen80_layer_execution_schedule, QWEN80_DELTANET_FULL_LAYER_KERNELS,
    QWEN80_GQA_FULL_LAYER_KERNELS, QWEN80_LAYERS,
};

/// One closed native-operator group: historical gap name, the on-disk shaders
/// that implement it, and the Rust dispatch site that encodes them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen80NativeOperatorWiring {
    pub gap: Qwen80HybridNativeOperatorGap,
    pub shaders: &'static [&'static str],
    pub rust_dispatch_site: &'static str,
}

/// Host token id vs the previous step's device-resident argmax.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen80EmbedSource {
    HostToken(u32),
    DeviceSampledToken,
}

/// The nine historical operator groups, each bound to existing shaders and a
/// token-graph dispatch site.  A gap remains required only when this table
/// cannot name both a shader and a dispatch function.
pub const QWEN80_NATIVE_OPERATOR_WIRING: [Qwen80NativeOperatorWiring; 9] = [
    Qwen80NativeOperatorWiring {
        gap: Qwen80HybridNativeOperatorGap::DirectPackedEmbeddingGather,
        shaders: &["qwen_complete_binary_embedding_lookup"],
        rust_dispatch_site: "dispatch_qwen80_embedding_gather_tcb",
    },
    Qwen80NativeOperatorWiring {
        gap: Qwen80HybridNativeOperatorGap::DirectPackedRmsNorm,
        shaders: &["qwen_next_direct_packed_input_rmsnorm"],
        rust_dispatch_site: "dispatch_qwen80_input_rmsnorm_tcb",
    },
    Qwen80NativeOperatorWiring {
        gap: Qwen80HybridNativeOperatorGap::LinearDeltaNetConvRearrangeAndGatedNorm,
        shaders: &[
            "qwen_next_qkvz_rearrange_conv_l2",
            "qwen_next_ba_to_decay_beta",
            "qwen_next_gated_delta_decode_single",
            "qwen_next_deltanet_gated_rmsnorm",
        ],
        rust_dispatch_site: "dispatch_qwen80_deltanet_mixer_prefix_tcb",
    },
    Qwen80NativeOperatorWiring {
        gap: Qwen80HybridNativeOperatorGap::LinearDeltaNetOutProjectionAndResidual,
        shaders: &["qwen_binary_sign_scale_matvec", "qwen_next_add_residual"],
        rust_dispatch_site: "dispatch_qwen80_deltanet_mixer_prefix_tcb",
    },
    Qwen80NativeOperatorWiring {
        gap: Qwen80HybridNativeOperatorGap::FullAttentionGqaKvRopeGateAndResidual,
        shaders: &[
            "qwen80_attention_qk_norm_rope_cache",
            "mha_decode_f32",
            "qwen80_attention_apply_sigmoid_gate",
            "qwen_binary_sign_scale_matvec",
            "qwen_next_add_residual",
        ],
        rust_dispatch_site: "dispatch_qwen80_gqa_mixer_prefix_tcb",
    },
    Qwen80NativeOperatorWiring {
        gap: Qwen80HybridNativeOperatorGap::RoutedTopTenGateUpDownAndWeightedCombine,
        shaders: &[
            "qwen80_postnorm_router_top10_rmsnorm",
            "qwen80_postnorm_router_top10_matvec",
            "qwen80_postnorm_router_top10_select",
            "qwen80_all_ten_routed_wave_route_guard",
            "qwen80_all_ten_routed_wave_gate_up",
            "qwen80_all_ten_routed_wave_swiglu",
            "qwen80_all_ten_routed_wave_down_weighted",
            "qwen80_moe_wave_aggregate_second_residual_route_sum",
        ],
        rust_dispatch_site: "dispatch_qwen80_moe_suffix_tcb",
    },
    Qwen80NativeOperatorWiring {
        gap: Qwen80HybridNativeOperatorGap::SharedExpertGateUpDownAndCombine,
        shaders: &[
            "qwen80_shared_expert_wave_gate_up",
            "qwen80_shared_expert_wave_swiglu",
            "qwen80_shared_expert_wave_down",
            "qwen80_shared_expert_wave_scalar_gate",
            "qwen80_shared_expert_wave_apply_sigmoid_gate",
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
        ],
        rust_dispatch_site: "dispatch_qwen80_moe_suffix_tcb",
    },
    Qwen80NativeOperatorWiring {
        gap: Qwen80HybridNativeOperatorGap::DirectPackedFinalNormLmHeadReservedTailMaskAndSampler,
        shaders: &[
            "qwen80_terminal_head_final_rmsnorm_direct_packed",
            "qwen80_terminal_head_all_row_direct_packed",
            "qwen80_terminal_head_mask_reserved_tail",
            "qwen80_terminal_head_greedy_sample_lowest_id",
        ],
        rust_dispatch_site: "dispatch_qwen80_terminal_head_tcb",
    },
    Qwen80NativeOperatorWiring {
        gap: Qwen80HybridNativeOperatorGap::DeviceResidentAutoregressiveStateAndFeedback,
        shaders: &[
            "qwen_complete_binary_embedding_lookup_device_token",
            "qwen80_terminal_head_feedback_guard",
        ],
        rust_dispatch_site: "dispatch_qwen80_embedding_from_device_token_tcb",
    },
];

/// Gaps that still have no shader or no dispatch site.  Empty once every
/// historical group in [`QWEN80_NATIVE_OPERATOR_WIRING`] is composed.
pub fn qwen80_composed_required_native_operator_gaps() -> Vec<Qwen80HybridNativeOperatorGap> {
    QWEN80_NATIVE_OPERATOR_WIRING
        .iter()
        .filter(|wiring| wiring.shaders.is_empty() || wiring.rust_dispatch_site.is_empty())
        .map(|wiring| wiring.gap)
        .collect()
}

pub fn qwen80_native_operator_wiring() -> &'static [Qwen80NativeOperatorWiring; 9] {
    &QWEN80_NATIVE_OPERATOR_WIRING
}

/// Embedding kernel plus the frozen 48-layer schedule plus the terminal head.
/// This is the structural 1-CB/token order; it is not an execution claim.
pub fn qwen80_hybrid_token_graph_kernel_trace(
    embed: Qwen80EmbedSource,
) -> Result<Vec<&'static str>, String> {
    let embed_kernel = match embed {
        Qwen80EmbedSource::HostToken(_) => "qwen_complete_binary_embedding_lookup",
        Qwen80EmbedSource::DeviceSampledToken => {
            "qwen_complete_binary_embedding_lookup_device_token"
        }
    };
    let mut kernels = Vec::with_capacity(1 + QWEN80_LAYERS * 23 + 5);
    kernels.push(embed_kernel);
    for layer in 0..QWEN80_LAYERS {
        let schedule = qwen80_layer_execution_schedule(layer)?;
        kernels.extend(schedule.full_layer_kernel_names.iter().copied());
    }
    kernels.extend_from_slice(&QWEN80_TERMINAL_HEAD_KERNELS);
    Ok(kernels)
}

pub const QWEN80_TERMINAL_HEAD_KERNELS: [&str; 5] = [
    "qwen80_terminal_head_final_rmsnorm_direct_packed",
    "qwen80_terminal_head_all_row_direct_packed",
    "qwen80_terminal_head_mask_reserved_tail",
    "qwen80_terminal_head_greedy_sample_lowest_id",
    "qwen80_terminal_head_feedback_guard",
];

pub const QWEN80_HYBRID_TOKEN_GRAPH_HOST_EMBED_DISPATCHES: usize = 1 + QWEN80_LAYERS * 23 + 5;

/// Prove the wiring table covers every historical gap exactly once and that
/// the 1-CB token order is embed + 48 scheduled layers + terminal head.
pub fn qwen80_assert_native_operator_composition_complete() -> Result<(), String> {
    if QWEN80_NATIVE_OPERATOR_WIRING.len() != QWEN80_HYBRID_NATIVE_OPERATOR_GAPS.len() {
        return Err(format!(
            "native operator wiring has {} entries, historical gap list has {}",
            QWEN80_NATIVE_OPERATOR_WIRING.len(),
            QWEN80_HYBRID_NATIVE_OPERATOR_GAPS.len()
        ));
    }
    for (index, (wiring, expected)) in QWEN80_NATIVE_OPERATOR_WIRING
        .iter()
        .zip(QWEN80_HYBRID_NATIVE_OPERATOR_GAPS.iter())
        .enumerate()
    {
        if wiring.gap != *expected {
            return Err(format!(
                "wiring[{index}] gap {:?}, expected {:?}",
                wiring.gap.as_str(),
                expected.as_str()
            ));
        }
        if wiring.shaders.is_empty() {
            return Err(format!("wiring {} has no shader", wiring.gap.as_str()));
        }
        if wiring.rust_dispatch_site.is_empty() {
            return Err(format!(
                "wiring {} has no Rust dispatch site",
                wiring.gap.as_str()
            ));
        }
    }
    let remaining = qwen80_composed_required_native_operator_gaps();
    if !remaining.is_empty() {
        return Err(format!(
            "composed required gaps is not empty: {:?}",
            remaining.iter().map(|gap| gap.as_str()).collect::<Vec<_>>()
        ));
    }
    let host_trace = qwen80_hybrid_token_graph_kernel_trace(Qwen80EmbedSource::HostToken(1))?;
    let device_trace =
        qwen80_hybrid_token_graph_kernel_trace(Qwen80EmbedSource::DeviceSampledToken)?;
    if host_trace.len() != QWEN80_HYBRID_TOKEN_GRAPH_HOST_EMBED_DISPATCHES
        || device_trace.len() != QWEN80_HYBRID_TOKEN_GRAPH_HOST_EMBED_DISPATCHES
    {
        return Err(format!(
            "token-graph trace length host={}, device={}, expected={}",
            host_trace.len(),
            device_trace.len(),
            QWEN80_HYBRID_TOKEN_GRAPH_HOST_EMBED_DISPATCHES
        ));
    }
    if host_trace[0] != "qwen_complete_binary_embedding_lookup"
        || device_trace[0] != "qwen_complete_binary_embedding_lookup_device_token"
    {
        return Err("token-graph embedding kernel drifted".into());
    }
    if host_trace[1..24] != QWEN80_DELTANET_FULL_LAYER_KERNELS
        || host_trace[70..93] != QWEN80_GQA_FULL_LAYER_KERNELS
    {
        return Err(
            "token-graph layer-0/layer-3 kernel order drifted from the frozen schedule".into(),
        );
    }
    if host_trace[host_trace.len() - 5..] != QWEN80_TERMINAL_HEAD_KERNELS {
        return Err("token-graph terminal-head kernel order drifted".into());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
mod device {
    use super::super::{
        bytes_for_f32, model_error, Qwen80CanonicalGqaDeviceResources, Qwen80CanonicalGqaLayout,
        Qwen80CanonicalLinearDeltaNetLayout, Qwen80CompleteNativeRuntime, Qwen80GpuBinaryTensor,
        QWEN80_EXPERTS, QWEN80_GROUP_SIZE, QWEN80_HIDDEN, QWEN80_LAYERS, QWEN80_MOE_INTERMEDIATE,
        QWEN80_RMS_EPS, QWEN80_TOKENIZER_VOCAB, QWEN80_TOP_K, QWEN80_VOCAB,
    };
    use super::Qwen80EmbedSource;
    use crate::kernels::{
        mha_decode_f32_tcb, qwen_binary_sign_scale_matvec_component_tcb,
        qwen_next_add_residual_tcb, qwen_next_ba_to_decay_beta_tcb,
        qwen_next_deltanet_gated_rmsnorm_tcb, qwen_next_direct_packed_input_rmsnorm_tcb,
        qwen_next_gated_delta_decode_single_at_state_offset_tcb,
        qwen_next_qkvz_rearrange_conv_l2_tcb,
    };
    use crate::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use crate::Result;

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

    /// Reused scratch for one composed token.  Mixer, MoE, and terminal stages
    /// ping-pong the residual through `hidden` so the graph stays one CB.
    pub struct Qwen80HybridTokenGraphWorkspace {
        pub hidden: PinnedBuffer,
        pub normalized: PinnedBuffer,
        pub qkvz_output: PinnedBuffer,
        pub ba_output: PinnedBuffer,
        pub repeated_query: PinnedBuffer,
        pub repeated_key: PinnedBuffer,
        pub convolved_value: PinnedBuffer,
        pub z: PinnedBuffer,
        pub decay: PinnedBuffer,
        pub beta: PinnedBuffer,
        pub recurrent_output: PinnedBuffer,
        pub gated_output: PinnedBuffer,
        pub mixer_output: PinnedBuffer,
        pub first_residual: PinnedBuffer,
        pub q_projection: PinnedBuffer,
        pub k_projection: PinnedBuffer,
        pub v_projection: PinnedBuffer,
        pub query: PinnedBuffer,
        pub attention: PinnedBuffer,
        pub gated_attention: PinnedBuffer,
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
        pub final_normalized: PinnedBuffer,
        pub logits: PinnedBuffer,
        pub sampled_token: PinnedBuffer,
        pub feedback_guard: PinnedBuffer,
    }

    impl Qwen80HybridTokenGraphWorkspace {
        pub fn allocate(context: &MetalContext) -> Result<Self> {
            let layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
            let gqa = Qwen80CanonicalGqaLayout::source_exact();
            let hidden = bytes_for_f32(QWEN80_HIDDEN, "token-graph hidden")?;
            let value = bytes_for_f32(layout.value_elements()?, "token-graph DeltaNet value")?;
            let qkvz = bytes_for_f32(
                layout.qkvz_projection_elements()?,
                "token-graph QKVZ projection",
            )?;
            let ba = bytes_for_f32(
                layout.ba_projection_elements()?,
                "token-graph BA projection",
            )?;
            let query = bytes_for_f32(gqa.query_dim, "token-graph GQA query")?;
            let kv = bytes_for_f32(gqa.kv_dim, "token-graph GQA KV")?;
            let q_proj = bytes_for_f32(gqa.q_proj_rows, "token-graph GQA q/gate")?;
            let route_mid = bytes_for_f32(
                QWEN80_TOP_K
                    .checked_mul(QWEN80_MOE_INTERMEDIATE)
                    .ok_or_else(|| model_error("token-graph route mid overflowed"))?,
                "token-graph routed intermediate",
            )?;
            let route_out = bytes_for_f32(
                QWEN80_TOP_K
                    .checked_mul(QWEN80_HIDDEN)
                    .ok_or_else(|| model_error("token-graph route out overflowed"))?,
                "token-graph routed hidden",
            )?;
            Ok(Self {
                hidden: context.new_buffer_checked(hidden)?,
                normalized: context.new_buffer_checked(hidden)?,
                qkvz_output: context.new_buffer_checked(qkvz)?,
                ba_output: context.new_buffer_checked(ba)?,
                repeated_query: context.new_buffer_checked(value)?,
                repeated_key: context.new_buffer_checked(value)?,
                convolved_value: context.new_buffer_checked(value)?,
                z: context.new_buffer_checked(value)?,
                decay: context
                    .new_buffer_checked(bytes_for_f32(layout.value_heads, "token-graph decay")?)?,
                beta: context
                    .new_buffer_checked(bytes_for_f32(layout.value_heads, "token-graph beta")?)?,
                recurrent_output: context.new_buffer_checked(value)?,
                gated_output: context.new_buffer_checked(value)?,
                mixer_output: context.new_buffer_checked(hidden)?,
                first_residual: context.new_buffer_checked(hidden)?,
                q_projection: context.new_buffer_checked(q_proj)?,
                k_projection: context.new_buffer_checked(kv)?,
                v_projection: context.new_buffer_checked(kv)?,
                query: context.new_buffer_checked(query)?,
                attention: context.new_buffer_checked(query)?,
                gated_attention: context.new_buffer_checked(query)?,
                postnorm_hidden: context.new_buffer_checked(hidden)?,
                router_logits: context.new_buffer_checked(bytes_for_f32(
                    QWEN80_EXPERTS,
                    "token-graph router logits",
                )?)?,
                router_probabilities: context.new_buffer_checked(bytes_for_f32(
                    QWEN80_EXPERTS,
                    "token-graph router probabilities",
                )?)?,
                router_route_ids: context.new_buffer_checked(
                    QWEN80_TOP_K
                        .checked_mul(std::mem::size_of::<u32>())
                        .ok_or_else(|| model_error("token-graph route id bytes overflowed"))?,
                )?,
                router_route_weights: context.new_buffer_checked(bytes_for_f32(
                    QWEN80_TOP_K,
                    "token-graph route weights",
                )?)?,
                route_guard: context.new_buffer_checked(std::mem::size_of::<u32>())?,
                route_gate: context.new_buffer_checked(route_mid)?,
                route_up: context.new_buffer_checked(route_mid)?,
                route_activated: context.new_buffer_checked(route_mid)?,
                route_weighted: context.new_buffer_checked(route_out)?,
                shared_gate: context.new_buffer_checked(bytes_for_f32(
                    QWEN80_MOE_INTERMEDIATE,
                    "token-graph shared gate",
                )?)?,
                shared_up: context.new_buffer_checked(bytes_for_f32(
                    QWEN80_MOE_INTERMEDIATE,
                    "token-graph shared up",
                )?)?,
                shared_activated: context.new_buffer_checked(bytes_for_f32(
                    QWEN80_MOE_INTERMEDIATE,
                    "token-graph shared activated",
                )?)?,
                shared_output: context.new_buffer_checked(hidden)?,
                shared_scalar_logit: context
                    .new_buffer_checked(bytes_for_f32(1, "token-graph shared scalar")?)?,
                gated_shared: context.new_buffer_checked(hidden)?,
                routed_sum: context.new_buffer_checked(hidden)?,
                final_normalized: context.new_buffer_checked(hidden)?,
                logits: context
                    .new_buffer_checked(bytes_for_f32(QWEN80_VOCAB, "token-graph logits")?)?,
                sampled_token: context.new_buffer_checked(std::mem::size_of::<u32>())?,
                feedback_guard: context.new_buffer_checked(std::mem::size_of::<u32>())?,
            })
        }
    }

    /// Device-resident mixer + fixed MoE + already-selected all-ten payloads
    /// for one layer.  The all-ten sign/scale sections are the ten source
    /// routes concatenated in router order — the same binding the existing
    /// true-MoE encoder uses.  This composition does not invent a 512-way
    /// indirect gather.
    pub struct Qwen80TokenGraphLayerResident {
        pub layer: usize,
        pub linear_state_slot: Option<usize>,
        pub full_attention_state_slot: Option<usize>,
        pub input_layernorm: Qwen80GpuBinaryTensor,
        pub deltanet: Option<Qwen80DeltaNetResident>,
        pub gqa: Option<Qwen80GqaResident>,
        pub postnorm: Qwen80GpuBinaryTensor,
        pub router: Qwen80GpuBinaryTensor,
        pub all_ten_gate_signs: PinnedBuffer,
        pub all_ten_gate_scales: PinnedBuffer,
        pub all_ten_up_signs: PinnedBuffer,
        pub all_ten_up_scales: PinnedBuffer,
        pub all_ten_down_signs: PinnedBuffer,
        pub all_ten_down_scales: PinnedBuffer,
        pub shared_gate: Qwen80GpuBinaryTensor,
        pub shared_up: Qwen80GpuBinaryTensor,
        pub shared_down: Qwen80GpuBinaryTensor,
        pub shared_scalar_gate: Qwen80GpuBinaryTensor,
    }

    pub struct Qwen80DeltaNetResident {
        pub qkvz: Qwen80GpuBinaryTensor,
        pub ba: Qwen80GpuBinaryTensor,
        pub conv: Qwen80GpuBinaryTensor,
        pub a_log: Qwen80GpuBinaryTensor,
        pub dt_bias: Qwen80GpuBinaryTensor,
        pub gated_norm: Qwen80GpuBinaryTensor,
        pub out_proj: Qwen80GpuBinaryTensor,
    }

    pub struct Qwen80GqaResident {
        pub q_proj: Qwen80GpuBinaryTensor,
        pub k_proj: Qwen80GpuBinaryTensor,
        pub v_proj: Qwen80GpuBinaryTensor,
        pub q_norm: Qwen80GpuBinaryTensor,
        pub k_norm: Qwen80GpuBinaryTensor,
        pub o_proj: Qwen80GpuBinaryTensor,
    }

    pub struct Qwen80HybridTokenGraphBindings {
        pub embedding: Qwen80GpuBinaryTensor,
        pub layers: Vec<Qwen80TokenGraphLayerResident>,
        pub final_norm: Qwen80GpuBinaryTensor,
        pub lm_head: Qwen80GpuBinaryTensor,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct Qwen80HybridTokenGraphEncode {
        pub dispatches: usize,
        pub position: usize,
        pub device_resident_feedback: bool,
    }

    pub fn dispatch_qwen80_embedding_gather_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        embedding: &Qwen80GpuBinaryTensor,
        hidden: &PinnedBuffer,
        token: u32,
    ) -> Result<()> {
        if embedding.header.shape != [QWEN80_VOCAB, QWEN80_HIDDEN] {
            return Err(model_error(
                "token-graph embedding shape drifted from [151936, 2048]",
            ));
        }
        tcb.dispatch_threads(
            "qwen_complete_binary_embedding_lookup",
            (QWEN80_HIDDEN as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&embedding.signs), 0);
                encoder.set_buffer(1, Some(&embedding.scales), 0);
                encoder.set_buffer(2, Some(hidden), 0);
                encoder.stage_set_u32(3, token);
                encoder.stage_set_u32(4, QWEN80_HIDDEN as u32);
                encoder.stage_set_u32(5, QWEN80_VOCAB as u32);
                encoder.stage_set_u32(6, QWEN80_GROUP_SIZE as u32);
            },
        )
    }

    pub fn dispatch_qwen80_embedding_from_device_token_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        embedding: &Qwen80GpuBinaryTensor,
        hidden: &PinnedBuffer,
        sampled_token: &PinnedBuffer,
    ) -> Result<()> {
        if embedding.header.shape != [QWEN80_VOCAB, QWEN80_HIDDEN] {
            return Err(model_error(
                "token-graph device embedding shape drifted from [151936, 2048]",
            ));
        }
        tcb.dispatch_threads(
            "qwen_complete_binary_embedding_lookup_device_token",
            (QWEN80_HIDDEN as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&embedding.signs), 0);
                encoder.set_buffer(1, Some(&embedding.scales), 0);
                encoder.set_buffer(2, Some(hidden), 0);
                encoder.set_buffer(3, Some(sampled_token), 0);
                encoder.stage_set_u32(4, QWEN80_HIDDEN as u32);
                encoder.stage_set_u32(5, QWEN80_VOCAB as u32);
                encoder.stage_set_u32(6, QWEN80_GROUP_SIZE as u32);
            },
        )
    }

    pub fn dispatch_qwen80_input_rmsnorm_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        input: &PinnedBuffer,
        weight: &Qwen80GpuBinaryTensor,
        output: &PinnedBuffer,
    ) -> Result<()> {
        qwen_next_direct_packed_input_rmsnorm_tcb(
            tcb,
            input,
            &weight.signs,
            &weight.scales,
            output,
            QWEN80_HIDDEN,
            QWEN80_GROUP_SIZE,
            QWEN80_RMS_EPS,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_qwen80_deltanet_mixer_prefix_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        workspace: &Qwen80HybridTokenGraphWorkspace,
        input: &PinnedBuffer,
        layer: &Qwen80TokenGraphLayerResident,
        conv_state: &PinnedBuffer,
        recurrent_state: &PinnedBuffer,
        linear_state_slot: usize,
    ) -> Result<()> {
        let mixer = layer.deltanet.as_ref().ok_or_else(|| {
            model_error(format!(
                "token-graph layer {} has no DeltaNet resident mixer",
                layer.layer
            ))
        })?;
        let layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
        let conv_offset = linear_state_slot
            .checked_mul(layout.conv_state_elements()?)
            .ok_or_else(|| model_error("token-graph DeltaNet conv offset overflowed"))?;
        let recurrent_offset = linear_state_slot
            .checked_mul(layout.recurrent_state_elements()?)
            .ok_or_else(|| model_error("token-graph DeltaNet recurrent offset overflowed"))?;
        dispatch_qwen80_input_rmsnorm_tcb(
            tcb,
            input,
            &layer.input_layernorm,
            &workspace.normalized,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            tcb,
            &mixer.qkvz.signs,
            &mixer.qkvz.scales,
            &workspace.normalized,
            &workspace.qkvz_output,
            layout.qkvz_projection_elements()?,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            tcb,
            &mixer.ba.signs,
            &mixer.ba.scales,
            &workspace.normalized,
            &workspace.ba_output,
            layout.ba_projection_elements()?,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_qkvz_rearrange_conv_l2_tcb(
            tcb,
            &workspace.qkvz_output,
            &mixer.conv.signs,
            &mixer.conv.scales,
            conv_state,
            conv_offset,
            &workspace.repeated_query,
            &workspace.repeated_key,
            &workspace.convolved_value,
            &workspace.z,
            layout.key_heads,
            layout.value_heads_per_key_head,
            layout.key_head_dim,
            layout.value_head_dim,
            layout.conv_kernel,
            QWEN80_GROUP_SIZE,
            QWEN80_RMS_EPS,
        )?;
        qwen_next_ba_to_decay_beta_tcb(
            tcb,
            &workspace.ba_output,
            &mixer.a_log.signs,
            &mixer.a_log.scales,
            &mixer.dt_bias.signs,
            &mixer.dt_bias.scales,
            &workspace.decay,
            &workspace.beta,
            layout.key_heads,
            layout.value_heads_per_key_head,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_gated_delta_decode_single_at_state_offset_tcb(
            tcb,
            recurrent_state,
            recurrent_offset,
            &workspace.repeated_query,
            &workspace.repeated_key,
            &workspace.convolved_value,
            &workspace.decay,
            &workspace.beta,
            &workspace.recurrent_output,
            layout.value_heads,
            layout.key_head_dim,
            layout.value_head_dim,
        )?;
        qwen_next_deltanet_gated_rmsnorm_tcb(
            tcb,
            &workspace.recurrent_output,
            &workspace.z,
            &mixer.gated_norm.signs,
            &mixer.gated_norm.scales,
            &workspace.gated_output,
            layout.value_heads,
            layout.value_head_dim,
            QWEN80_GROUP_SIZE,
            QWEN80_RMS_EPS,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            tcb,
            &mixer.out_proj.signs,
            &mixer.out_proj.scales,
            &workspace.gated_output,
            &workspace.mixer_output,
            layout.hidden_elements,
            layout.value_elements()?,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_add_residual_tcb(
            tcb,
            input,
            &workspace.mixer_output,
            &workspace.first_residual,
            layout.hidden_elements,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_qwen80_gqa_mixer_prefix_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        workspace: &Qwen80HybridTokenGraphWorkspace,
        input: &PinnedBuffer,
        layer: &Qwen80TokenGraphLayerResident,
        key_cache: &PinnedBuffer,
        value_cache: &PinnedBuffer,
        full_attention_state_slot: usize,
        position: usize,
        max_seq_len: usize,
    ) -> Result<()> {
        let mixer = layer.gqa.as_ref().ok_or_else(|| {
            model_error(format!(
                "token-graph layer {} has no GQA resident mixer",
                layer.layer
            ))
        })?;
        let layout = Qwen80CanonicalGqaLayout::source_exact();
        let resources = Qwen80CanonicalGqaDeviceResources::for_max_seq_len(
            &layout,
            full_attention_state_slot,
            max_seq_len,
            Default::default(),
        )?;
        if position >= resources.max_seq_len {
            return Err(model_error(format!(
                "token-graph GQA position {position} is outside 0..{}",
                resources.max_seq_len
            )));
        }
        let key_slot_byte_offset = resources
            .key_cache_offset_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| model_error("token-graph GQA key slot byte offset overflowed"))?;
        let value_slot_byte_offset = resources
            .value_cache_offset_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| model_error("token-graph GQA value slot byte offset overflowed"))?;
        dispatch_qwen80_input_rmsnorm_tcb(
            tcb,
            input,
            &layer.input_layernorm,
            &workspace.normalized,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            tcb,
            &mixer.q_proj.signs,
            &mixer.q_proj.scales,
            &workspace.normalized,
            &workspace.q_projection,
            layout.q_proj_rows,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            tcb,
            &mixer.k_proj.signs,
            &mixer.k_proj.scales,
            &workspace.normalized,
            &workspace.k_projection,
            layout.kv_dim,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            tcb,
            &mixer.v_proj.signs,
            &mixer.v_proj.scales,
            &workspace.normalized,
            &workspace.v_projection,
            layout.kv_dim,
            layout.hidden_elements,
            QWEN80_GROUP_SIZE,
        )?;
        tcb.dispatch_threads(
            "qwen80_attention_qk_norm_rope_cache",
            (layout.query_heads as u32, 1, 1),
            (layout.query_heads as u32, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&workspace.q_projection), 0);
                encoder.set_buffer(1, Some(&workspace.k_projection), 0);
                encoder.set_buffer(2, Some(&workspace.v_projection), 0);
                encoder.set_buffer(3, Some(&mixer.q_norm.signs), 0);
                encoder.set_buffer(4, Some(&mixer.q_norm.scales), 0);
                encoder.set_buffer(5, Some(&mixer.k_norm.signs), 0);
                encoder.set_buffer(6, Some(&mixer.k_norm.scales), 0);
                encoder.set_buffer(7, Some(&workspace.query), 0);
                encoder.set_buffer(8, Some(key_cache), key_slot_byte_offset as u64);
                encoder.set_buffer(9, Some(value_cache), value_slot_byte_offset as u64);
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
            tcb,
            &workspace.query,
            key_cache,
            key_slot_byte_offset,
            value_cache,
            value_slot_byte_offset,
            &workspace.attention,
            position + 1,
            layout.head_dim,
            layout.query_heads,
            layout.key_value_heads,
        )?;
        tcb.dispatch_threads(
            "qwen80_attention_apply_sigmoid_gate",
            (layout.query_dim as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&workspace.attention), 0);
                encoder.set_buffer(1, Some(&workspace.q_projection), 0);
                encoder.set_buffer(2, Some(&workspace.gated_attention), 0);
                encoder.stage_set_u32(3, layout.query_dim as u32);
                encoder.stage_set_u32(4, layout.head_dim as u32);
            },
        )?;
        qwen_binary_sign_scale_matvec_component_tcb(
            tcb,
            &mixer.o_proj.signs,
            &mixer.o_proj.scales,
            &workspace.gated_attention,
            &workspace.mixer_output,
            layout.hidden_elements,
            layout.query_dim,
            QWEN80_GROUP_SIZE,
        )?;
        qwen_next_add_residual_tcb(
            tcb,
            input,
            &workspace.mixer_output,
            &workspace.first_residual,
            layout.hidden_elements,
        )
    }

    pub fn dispatch_qwen80_moe_suffix_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        workspace: &Qwen80HybridTokenGraphWorkspace,
        layer: &Qwen80TokenGraphLayerResident,
        second_residual: &PinnedBuffer,
    ) -> Result<()> {
        tcb.dispatch_threads(
            "qwen80_postnorm_router_top10_rmsnorm",
            (256, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&workspace.first_residual), 0);
                encoder.set_buffer(1, Some(&layer.postnorm.signs), 0);
                encoder.set_buffer(2, Some(&layer.postnorm.scales), 0);
                encoder.set_buffer(3, Some(&workspace.postnorm_hidden), 0);
                encoder.stage_set_u32(4, QWEN80_HIDDEN as u32);
                encoder.stage_set_u32(5, QWEN80_GROUP_SIZE as u32);
                encoder.stage_set_f32(6, QWEN80_RMS_EPS);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_postnorm_router_top10_matvec",
            (256, QWEN80_EXPERTS as u32, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&layer.router.signs), 0);
                encoder.set_buffer(1, Some(&layer.router.scales), 0);
                encoder.set_buffer(2, Some(&workspace.postnorm_hidden), 0);
                encoder.set_buffer(3, Some(&workspace.router_logits), 0);
                encoder.stage_set_u32(4, QWEN80_EXPERTS as u32);
                encoder.stage_set_u32(5, QWEN80_HIDDEN as u32);
                encoder.stage_set_u32(6, QWEN80_GROUP_SIZE as u32);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_postnorm_router_top10_select",
            (1, 1, 1),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&workspace.router_logits), 0);
                encoder.set_buffer(1, Some(&workspace.router_probabilities), 0);
                encoder.set_buffer(2, Some(&workspace.router_route_ids), 0);
                encoder.set_buffer(3, Some(&workspace.router_route_weights), 0);
                encoder.stage_set_u32(4, QWEN80_EXPERTS as u32);
                encoder.stage_set_u32(5, QWEN80_TOP_K as u32);
                encoder.stage_set_f32(6, 0.0);
            },
        )?;
        // Generate-shaped graph: the live router output is the authority, so
        // expected IDs/weights alias the observed buffers and the frozen
        // route_guard dispatch stays in the 14-kernel suffix.
        tcb.dispatch_threads(
            "qwen80_all_ten_routed_wave_route_guard",
            (1, 1, 1),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&workspace.router_route_ids), 0);
                encoder.set_buffer(1, Some(&workspace.router_route_weights), 0);
                encoder.set_buffer(2, Some(&workspace.router_route_ids), 0);
                encoder.set_buffer(3, Some(&workspace.router_route_weights), 0);
                encoder.set_buffer(4, Some(&workspace.route_guard), 0);
                encoder.stage_set_u32(5, QWEN80_TOP_K as u32);
                encoder.stage_set_f32(6, 0.0);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_all_ten_routed_wave_gate_up",
            (256, QWEN80_MOE_INTERMEDIATE as u32, QWEN80_TOP_K as u32),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&layer.all_ten_gate_signs), 0);
                encoder.set_buffer(1, Some(&layer.all_ten_gate_scales), 0);
                encoder.set_buffer(2, Some(&layer.all_ten_up_signs), 0);
                encoder.set_buffer(3, Some(&layer.all_ten_up_scales), 0);
                encoder.set_buffer(4, Some(&workspace.postnorm_hidden), 0);
                encoder.set_buffer(5, Some(&workspace.route_gate), 0);
                encoder.set_buffer(6, Some(&workspace.route_up), 0);
                encoder.stage_set_u32(7, QWEN80_TOP_K as u32);
                encoder.stage_set_u32(8, QWEN80_MOE_INTERMEDIATE as u32);
                encoder.stage_set_u32(9, QWEN80_HIDDEN as u32);
                encoder.stage_set_u32(10, QWEN80_GROUP_SIZE as u32);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_all_ten_routed_wave_swiglu",
            (QWEN80_MOE_INTERMEDIATE as u32, QWEN80_TOP_K as u32, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&workspace.route_gate), 0);
                encoder.set_buffer(1, Some(&workspace.route_up), 0);
                encoder.set_buffer(2, Some(&workspace.route_activated), 0);
                encoder.stage_set_u32(3, QWEN80_TOP_K as u32);
                encoder.stage_set_u32(4, QWEN80_MOE_INTERMEDIATE as u32);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_all_ten_routed_wave_down_weighted",
            (256, QWEN80_HIDDEN as u32, QWEN80_TOP_K as u32),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&layer.all_ten_down_signs), 0);
                encoder.set_buffer(1, Some(&layer.all_ten_down_scales), 0);
                encoder.set_buffer(2, Some(&workspace.route_activated), 0);
                encoder.set_buffer(3, Some(&workspace.router_route_weights), 0);
                encoder.set_buffer(4, Some(&workspace.route_weighted), 0);
                encoder.stage_set_u32(5, QWEN80_TOP_K as u32);
                encoder.stage_set_u32(6, QWEN80_HIDDEN as u32);
                encoder.stage_set_u32(7, QWEN80_MOE_INTERMEDIATE as u32);
                encoder.stage_set_u32(8, QWEN80_GROUP_SIZE as u32);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_shared_expert_wave_gate_up",
            (256, QWEN80_MOE_INTERMEDIATE as u32, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&layer.shared_gate.signs), 0);
                encoder.set_buffer(1, Some(&layer.shared_gate.scales), 0);
                encoder.set_buffer(2, Some(&layer.shared_up.signs), 0);
                encoder.set_buffer(3, Some(&layer.shared_up.scales), 0);
                encoder.set_buffer(4, Some(&workspace.postnorm_hidden), 0);
                encoder.set_buffer(5, Some(&workspace.shared_gate), 0);
                encoder.set_buffer(6, Some(&workspace.shared_up), 0);
                encoder.stage_set_u32(7, QWEN80_MOE_INTERMEDIATE as u32);
                encoder.stage_set_u32(8, QWEN80_HIDDEN as u32);
                encoder.stage_set_u32(9, QWEN80_GROUP_SIZE as u32);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_shared_expert_wave_swiglu",
            (QWEN80_MOE_INTERMEDIATE as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&workspace.shared_gate), 0);
                encoder.set_buffer(1, Some(&workspace.shared_up), 0);
                encoder.set_buffer(2, Some(&workspace.shared_activated), 0);
                encoder.stage_set_u32(3, QWEN80_MOE_INTERMEDIATE as u32);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_shared_expert_wave_down",
            (256, QWEN80_HIDDEN as u32, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&layer.shared_down.signs), 0);
                encoder.set_buffer(1, Some(&layer.shared_down.scales), 0);
                encoder.set_buffer(2, Some(&workspace.shared_activated), 0);
                encoder.set_buffer(3, Some(&workspace.shared_output), 0);
                encoder.stage_set_u32(4, QWEN80_HIDDEN as u32);
                encoder.stage_set_u32(5, QWEN80_MOE_INTERMEDIATE as u32);
                encoder.stage_set_u32(6, QWEN80_GROUP_SIZE as u32);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_shared_expert_wave_scalar_gate",
            (256, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&layer.shared_scalar_gate.signs), 0);
                encoder.set_buffer(1, Some(&layer.shared_scalar_gate.scales), 0);
                encoder.set_buffer(2, Some(&workspace.postnorm_hidden), 0);
                encoder.set_buffer(3, Some(&workspace.shared_scalar_logit), 0);
                encoder.stage_set_u32(4, QWEN80_HIDDEN as u32);
                encoder.stage_set_u32(5, QWEN80_GROUP_SIZE as u32);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_shared_expert_wave_apply_sigmoid_gate",
            (QWEN80_HIDDEN as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&workspace.shared_output), 0);
                encoder.set_buffer(1, Some(&workspace.shared_scalar_logit), 0);
                encoder.set_buffer(2, Some(&workspace.gated_shared), 0);
                encoder.stage_set_u32(3, QWEN80_HIDDEN as u32);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_moe_wave_aggregate_second_residual_route_sum",
            (QWEN80_HIDDEN as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&workspace.route_weighted), 0);
                encoder.set_buffer(1, Some(&workspace.routed_sum), 0);
                encoder.stage_set_u32(2, QWEN80_TOP_K as u32);
                encoder.stage_set_u32(3, QWEN80_HIDDEN as u32);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
            (QWEN80_HIDDEN as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&workspace.routed_sum), 0);
                encoder.set_buffer(1, Some(&workspace.gated_shared), 0);
                encoder.set_buffer(2, Some(&workspace.first_residual), 0);
                encoder.set_buffer(3, Some(second_residual), 0);
                encoder.stage_set_u32(4, QWEN80_HIDDEN as u32);
            },
        )
    }

    pub fn dispatch_qwen80_terminal_head_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        workspace: &Qwen80HybridTokenGraphWorkspace,
        hidden: &PinnedBuffer,
        final_norm: &Qwen80GpuBinaryTensor,
        lm_head: &Qwen80GpuBinaryTensor,
    ) -> Result<()> {
        tcb.dispatch_threads(
            "qwen80_terminal_head_final_rmsnorm_direct_packed",
            (256, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(hidden), 0);
                encoder.set_buffer(1, Some(&final_norm.signs), 0);
                encoder.set_buffer(2, Some(&final_norm.scales), 0);
                encoder.set_buffer(3, Some(&workspace.final_normalized), 0);
                encoder.stage_set_u32(4, QWEN80_HIDDEN as u32);
                encoder.stage_set_u32(5, QWEN80_GROUP_SIZE as u32);
                encoder.stage_set_f32(6, QWEN80_RMS_EPS);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_terminal_head_all_row_direct_packed",
            (256, QWEN80_VOCAB as u32, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&lm_head.signs), 0);
                encoder.set_buffer(1, Some(&lm_head.scales), 0);
                encoder.set_buffer(2, Some(&workspace.final_normalized), 0);
                encoder.set_buffer(3, Some(&workspace.logits), 0);
                encoder.stage_set_u32(4, QWEN80_VOCAB as u32);
                encoder.stage_set_u32(5, QWEN80_HIDDEN as u32);
                encoder.stage_set_u32(6, QWEN80_GROUP_SIZE as u32);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_terminal_head_mask_reserved_tail",
            (256, (QWEN80_VOCAB - QWEN80_TOKENIZER_VOCAB) as u32, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&workspace.logits), 0);
                encoder.stage_set_u32(1, QWEN80_TOKENIZER_VOCAB as u32);
                encoder.stage_set_u32(2, (QWEN80_VOCAB - QWEN80_TOKENIZER_VOCAB) as u32);
                encoder.stage_set_u32(3, QWEN80_VOCAB as u32);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_terminal_head_greedy_sample_lowest_id",
            (1, 1, 1),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&workspace.logits), 0);
                encoder.set_buffer(1, Some(&workspace.sampled_token), 0);
                encoder.stage_set_u32(2, QWEN80_TOKENIZER_VOCAB as u32);
                encoder.stage_set_u32(3, QWEN80_VOCAB as u32);
            },
        )?;
        tcb.dispatch_threads(
            "qwen80_terminal_head_feedback_guard",
            (1, 1, 1),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&workspace.sampled_token), 0);
                encoder.set_buffer(1, Some(&workspace.feedback_guard), 0);
                encoder.stage_set_u32(2, QWEN80_TOKENIZER_VOCAB as u32);
            },
        )
    }

    /// Encode one token: embedding + 48 hybrid layers + terminal head, one CB.
    /// Does not commit, wait, or read back a sample.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_qwen80_hybrid_token_graph_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        workspace: &Qwen80HybridTokenGraphWorkspace,
        bindings: &Qwen80HybridTokenGraphBindings,
        embed: Qwen80EmbedSource,
        position: usize,
        conv_state: &PinnedBuffer,
        recurrent_state: &PinnedBuffer,
        key_cache: &PinnedBuffer,
        value_cache: &PinnedBuffer,
        max_seq_len: usize,
    ) -> Result<Qwen80HybridTokenGraphEncode> {
        if bindings.layers.len() != QWEN80_LAYERS {
            return Err(model_error(format!(
                "token-graph bindings have {} layers, expected {QWEN80_LAYERS}",
                bindings.layers.len()
            )));
        }
        let before = tcb.dispatch_count();
        match embed {
            Qwen80EmbedSource::HostToken(token) => {
                dispatch_qwen80_embedding_gather_tcb(
                    tcb,
                    &bindings.embedding,
                    &workspace.hidden,
                    token,
                )?;
            }
            Qwen80EmbedSource::DeviceSampledToken => {
                dispatch_qwen80_embedding_from_device_token_tcb(
                    tcb,
                    &bindings.embedding,
                    &workspace.hidden,
                    &workspace.sampled_token,
                )?;
            }
        }
        let mut stream = &workspace.hidden;
        for layer in &bindings.layers {
            match (
                layer.deltanet.is_some(),
                layer.gqa.is_some(),
                layer.linear_state_slot,
                layer.full_attention_state_slot,
            ) {
                (true, false, Some(slot), None) => {
                    dispatch_qwen80_deltanet_mixer_prefix_tcb(
                        tcb,
                        workspace,
                        stream,
                        layer,
                        conv_state,
                        recurrent_state,
                        slot,
                    )?;
                }
                (false, true, None, Some(slot)) => {
                    dispatch_qwen80_gqa_mixer_prefix_tcb(
                        tcb,
                        workspace,
                        stream,
                        layer,
                        key_cache,
                        value_cache,
                        slot,
                        position,
                        max_seq_len,
                    )?;
                }
                _ => {
                    return Err(model_error(format!(
                        "token-graph layer {} has an invalid DeltaNet/GQA resident split",
                        layer.layer
                    )));
                }
            }
            dispatch_qwen80_moe_suffix_tcb(tcb, workspace, layer, &workspace.hidden)?;
            stream = &workspace.hidden;
        }
        dispatch_qwen80_terminal_head_tcb(
            tcb,
            workspace,
            stream,
            &bindings.final_norm,
            &bindings.lm_head,
        )?;
        let dispatches = tcb
            .dispatch_count()
            .checked_sub(before)
            .ok_or_else(|| model_error("token-graph dispatch count underflowed"))?;
        if dispatches != super::QWEN80_HYBRID_TOKEN_GRAPH_HOST_EMBED_DISPATCHES {
            return Err(model_error(format!(
                "token-graph encoded {dispatches} dispatches, expected {}",
                super::QWEN80_HYBRID_TOKEN_GRAPH_HOST_EMBED_DISPATCHES
            )));
        }
        Ok(Qwen80HybridTokenGraphEncode {
            dispatches,
            position,
            device_resident_feedback: matches!(embed, Qwen80EmbedSource::DeviceSampledToken),
        })
    }

    impl Qwen80CompleteNativeRuntime {
        /// Encode the composed hybrid token graph onto an already-open command
        /// buffer using this runtime's device-resident state arenas.  Does not
        /// commit, generate, or open the 74,391-tensor catalog.
        pub fn encode_composed_hybrid_token_graph(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            workspace: &Qwen80HybridTokenGraphWorkspace,
            bindings: &Qwen80HybridTokenGraphBindings,
            embed: Qwen80EmbedSource,
            position: usize,
        ) -> Result<Qwen80HybridTokenGraphEncode> {
            if position >= self.state.max_seq_len {
                return Err(model_error(format!(
                    "token-graph position {position} exceeds native context {}",
                    self.state.max_seq_len
                )));
            }
            encode_qwen80_hybrid_token_graph_tcb(
                tcb,
                workspace,
                bindings,
                embed,
                position,
                &self.linear_conv_state,
                &self.linear_recurrent_state,
                &self.full_attention_key_cache,
                &self.full_attention_value_cache,
                self.state.max_seq_len,
            )
        }
    }

    /// Force the nine dispatch sites to stay in the crate so a later generate
    /// wave can call them without re-deriving the graph.
    #[allow(dead_code)]
    fn qwen80_native_operator_dispatch_sites_are_linked() {
        let _: fn(
            &mut TokenCommandBuffer<'_>,
            &Qwen80GpuBinaryTensor,
            &PinnedBuffer,
            u32,
        ) -> Result<()> = dispatch_qwen80_embedding_gather_tcb;
        let _: fn(
            &mut TokenCommandBuffer<'_>,
            &Qwen80GpuBinaryTensor,
            &PinnedBuffer,
            &PinnedBuffer,
        ) -> Result<()> = dispatch_qwen80_embedding_from_device_token_tcb;
        let _: fn(
            &mut TokenCommandBuffer<'_>,
            &PinnedBuffer,
            &Qwen80GpuBinaryTensor,
            &PinnedBuffer,
        ) -> Result<()> = dispatch_qwen80_input_rmsnorm_tcb;
        let _ = dispatch_qwen80_deltanet_mixer_prefix_tcb;
        let _ = dispatch_qwen80_gqa_mixer_prefix_tcb;
        let _ = dispatch_qwen80_moe_suffix_tcb;
        let _ = dispatch_qwen80_terminal_head_tcb;
        let _ = encode_qwen80_hybrid_token_graph_tcb;
        let _ = crate::model::qwen80_device_expert_table::dispatch_qwen80_device_expert_table_tcb;
        let _ = super::device_activations::dispatch_qwen80_residual_rmsnorm_f32_tcb;
        let _ = super::device_activations::dispatch_qwen80_silu_mul_f32_tcb;
        let _ = super::device_activations::dispatch_qwen80_qkvz_rearrange_conv_l2_f32_tcb;
        let _ = super::device_activations::dispatch_qwen80_ba_to_decay_beta_f32_tcb;
        let _ = super::device_activations::dispatch_qwen80_deltanet_gated_rmsnorm_f32_tcb;
        let _ = super::device_activations::dispatch_qwen80_gqa_qk_norm_rope_cache_f32_tcb;
        let _ = super::device_activations::dispatch_qwen80_gated_delta_decode_tg_tcb;
        let _ = super::device_activations::QWEN80_DEVICE_ACTIVATION_KERNELS;
    }
}

#[cfg(target_os = "macos")]
#[allow(unused_imports)]
pub use device::{
    dispatch_qwen80_deltanet_mixer_prefix_tcb, dispatch_qwen80_embedding_from_device_token_tcb,
    dispatch_qwen80_embedding_gather_tcb, dispatch_qwen80_gqa_mixer_prefix_tcb,
    dispatch_qwen80_input_rmsnorm_tcb, dispatch_qwen80_moe_suffix_tcb,
    dispatch_qwen80_terminal_head_tcb, encode_qwen80_hybrid_token_graph_tcb,
    Qwen80DeltaNetResident, Qwen80GqaResident, Qwen80HybridTokenGraphBindings,
    Qwen80HybridTokenGraphEncode, Qwen80HybridTokenGraphWorkspace, Qwen80TokenGraphLayerResident,
};

/// Additive f32 activation dispatch sites for the uniform-Q4 hybrid decode.
/// These keep residuals device-resident between already-native Q4 matvecs.
#[cfg(target_os = "macos")]
mod device_activations {
    use crate::metal::{PinnedBuffer, TokenCommandBuffer};
    use crate::Result;

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

    pub const QWEN80_DEVICE_ACTIVATION_KERNELS: [&str; 7] = [
        "qwen80_residual_rmsnorm_f32",
        "qwen80_silu_mul_f32",
        "qwen80_qkvz_rearrange_conv_l2_f32",
        "qwen80_ba_to_decay_beta_f32",
        "qwen80_deltanet_gated_rmsnorm_f32",
        "qwen80_gated_delta_decode_tg",
        "qwen80_gqa_qk_norm_rope_cache_f32",
    ];

    pub fn dispatch_qwen80_residual_rmsnorm_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        input: &PinnedBuffer,
        weight: &PinnedBuffer,
        output: &PinnedBuffer,
        hidden: u32,
        eps: f32,
    ) -> Result<()> {
        tcb.dispatch_threads(
            "qwen80_residual_rmsnorm_f32",
            (256, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(weight), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.stage_set_u32(3, hidden);
                encoder.stage_set_f32(4, eps);
                encoder.set_threadgroup_memory_length(0, 256 * 4);
            },
        )
    }

    pub fn dispatch_qwen80_silu_mul_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        gate: &PinnedBuffer,
        up: &PinnedBuffer,
        output: &PinnedBuffer,
        n: u32,
    ) -> Result<()> {
        tcb.dispatch_threads(
            "qwen80_silu_mul_f32",
            (n, 1, 1),
            (n.min(256).max(1), 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(gate), 0);
                encoder.set_buffer(1, Some(up), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.stage_set_u32(3, n);
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_qwen80_qkvz_rearrange_conv_l2_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        projected_qkvz: &PinnedBuffer,
        conv_weight: &PinnedBuffer,
        conv_state: &PinnedBuffer,
        conv_state_offset_bytes: u64,
        repeated_query: &PinnedBuffer,
        repeated_key: &PinnedBuffer,
        convolved_value: &PinnedBuffer,
        z: &PinnedBuffer,
        key_heads: u32,
        values_per_key_head: u32,
        key_head_dim: u32,
        value_head_dim: u32,
        conv_kernel: u32,
        eps: f32,
    ) -> Result<()> {
        tcb.dispatch_threads(
            "qwen80_qkvz_rearrange_conv_l2_f32",
            (256, key_heads, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(projected_qkvz), 0);
                encoder.set_buffer(1, Some(conv_weight), 0);
                encoder.set_buffer(2, Some(conv_state), conv_state_offset_bytes);
                encoder.set_buffer(3, Some(repeated_query), 0);
                encoder.set_buffer(4, Some(repeated_key), 0);
                encoder.set_buffer(5, Some(convolved_value), 0);
                encoder.set_buffer(6, Some(z), 0);
                encoder.stage_set_u32(7, key_heads);
                encoder.stage_set_u32(8, values_per_key_head);
                encoder.stage_set_u32(9, key_head_dim);
                encoder.stage_set_u32(10, value_head_dim);
                encoder.stage_set_u32(11, conv_kernel);
                encoder.stage_set_f32(12, eps);
                encoder.set_threadgroup_memory_length(0, 4 * 256 * 4);
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_qwen80_ba_to_decay_beta_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        projected_ba: &PinnedBuffer,
        a_log: &PinnedBuffer,
        dt_bias: &PinnedBuffer,
        decay: &PinnedBuffer,
        beta: &PinnedBuffer,
        key_heads: u32,
        values_per_key_head: u32,
    ) -> Result<()> {
        let value_heads = key_heads.saturating_mul(values_per_key_head);
        tcb.dispatch_threads(
            "qwen80_ba_to_decay_beta_f32",
            (value_heads, 1, 1),
            (value_heads.min(32).max(1), 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(projected_ba), 0);
                encoder.set_buffer(1, Some(a_log), 0);
                encoder.set_buffer(2, Some(dt_bias), 0);
                encoder.set_buffer(3, Some(decay), 0);
                encoder.set_buffer(4, Some(beta), 0);
                encoder.stage_set_u32(5, key_heads);
                encoder.stage_set_u32(6, values_per_key_head);
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_qwen80_gated_delta_decode_tg_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        state: &PinnedBuffer,
        state_offset_bytes: u64,
        query: &PinnedBuffer,
        key: &PinnedBuffer,
        value: &PinnedBuffer,
        decay: &PinnedBuffer,
        beta: &PinnedBuffer,
        output: &PinnedBuffer,
        heads: u32,
        key_dim: u32,
        value_dim: u32,
    ) -> Result<()> {
        tcb.dispatch_threads(
            "qwen80_gated_delta_decode_tg",
            (key_dim, heads, 1),
            (key_dim.max(1), 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(state), state_offset_bytes);
                encoder.set_buffer(1, Some(query), 0);
                encoder.set_buffer(2, Some(key), 0);
                encoder.set_buffer(3, Some(value), 0);
                encoder.set_buffer(4, Some(decay), 0);
                encoder.set_buffer(5, Some(beta), 0);
                encoder.set_buffer(6, Some(output), 0);
                encoder.stage_set_u32(7, heads);
                encoder.stage_set_u32(8, key_dim);
                encoder.stage_set_u32(9, value_dim);
                encoder.set_threadgroup_memory_length(0, 128 * 4);
            },
        )
    }

    pub fn dispatch_qwen80_deltanet_gated_rmsnorm_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        input: &PinnedBuffer,
        gate: &PinnedBuffer,
        weight: &PinnedBuffer,
        output: &PinnedBuffer,
        heads: u32,
        value_head_dim: u32,
        eps: f32,
    ) -> Result<()> {
        tcb.dispatch_threads(
            "qwen80_deltanet_gated_rmsnorm_f32",
            (heads, 1, 1),
            (heads.min(32).max(1), 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(gate), 0);
                encoder.set_buffer(2, Some(weight), 0);
                encoder.set_buffer(3, Some(output), 0);
                encoder.stage_set_u32(4, heads);
                encoder.stage_set_u32(5, value_head_dim);
                encoder.stage_set_f32(6, eps);
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_qwen80_gqa_qk_norm_rope_cache_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q_proj: &PinnedBuffer,
        k_proj: &PinnedBuffer,
        v_proj: &PinnedBuffer,
        q_norm: &PinnedBuffer,
        k_norm: &PinnedBuffer,
        query: &PinnedBuffer,
        key_cache: &PinnedBuffer,
        key_offset_bytes: u64,
        value_cache: &PinnedBuffer,
        value_offset_bytes: u64,
        sequence_slot: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        rope_theta: f32,
        rms_eps: f32,
    ) -> Result<()> {
        tcb.dispatch_threads(
            "qwen80_gqa_qk_norm_rope_cache_f32",
            (n_heads, 1, 1),
            (n_heads.min(16).max(1), 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(q_proj), 0);
                encoder.set_buffer(1, Some(k_proj), 0);
                encoder.set_buffer(2, Some(v_proj), 0);
                encoder.set_buffer(3, Some(q_norm), 0);
                encoder.set_buffer(4, Some(k_norm), 0);
                encoder.set_buffer(5, Some(query), 0);
                encoder.set_buffer(6, Some(key_cache), key_offset_bytes);
                encoder.set_buffer(7, Some(value_cache), value_offset_bytes);
                encoder.stage_set_u32(8, sequence_slot);
                encoder.stage_set_u32(9, n_heads);
                encoder.stage_set_u32(10, n_kv_heads);
                encoder.stage_set_u32(11, head_dim);
                encoder.stage_set_u32(12, rotary_dim);
                encoder.stage_set_f32(13, rope_theta);
                encoder.stage_set_f32(14, rms_eps);
            },
        )
    }
}

#[cfg(target_os = "macos")]
#[allow(unused_imports)]
pub use device_activations::{
    dispatch_qwen80_ba_to_decay_beta_f32_tcb, dispatch_qwen80_deltanet_gated_rmsnorm_f32_tcb,
    dispatch_qwen80_gated_delta_decode_tg_tcb, dispatch_qwen80_gqa_qk_norm_rope_cache_f32_tcb,
    dispatch_qwen80_qkvz_rearrange_conv_l2_f32_tcb, dispatch_qwen80_residual_rmsnorm_f32_tcb,
    dispatch_qwen80_silu_mul_f32_tcb, QWEN80_DEVICE_ACTIVATION_KERNELS,
};

#[cfg(target_os = "macos")]
#[allow(unused_imports)]
pub use crate::model::qwen80_device_expert_table::dispatch_qwen80_device_expert_table_tcb;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_operator_composition_closes_all_nine_historical_gaps() {
        qwen80_assert_native_operator_composition_complete().unwrap();
        assert!(qwen80_composed_required_native_operator_gaps().is_empty());
        assert_eq!(
            qwen80_native_operator_wiring()[0].gap.as_str(),
            "direct_packed_embedding_gather"
        );
        assert_eq!(
            qwen80_native_operator_wiring().last().unwrap().gap.as_str(),
            "device_resident_autoregressive_state_and_feedback"
        );
    }

    #[test]
    fn composed_shaders_exist_in_the_on_disk_metal_sources() {
        let sources = [
            include_str!("../../shaders/qwen_complete_runtime.metal"),
            include_str!("../../shaders/qwen_next.metal"),
            include_str!("../../shaders/qwen_binary.metal"),
            include_str!("../../shaders/qwen80_direct_packed_attention_stage.metal"),
            include_str!("../../shaders/mha.metal"),
            include_str!("../../shaders/qwen80_postnorm_router_top10.metal"),
            include_str!("../../shaders/qwen80_all_ten_routed_expert_wave.metal"),
            include_str!("../../shaders/qwen80_shared_expert_wave.metal"),
            include_str!("../../shaders/qwen80_moe_wave_aggregate_second_residual.metal"),
            include_str!("../../shaders/qwen80_direct_packed_terminal_head_preflight.metal"),
        ]
        .join("\n");
        for wiring in qwen80_native_operator_wiring() {
            for shader in wiring.shaders {
                assert!(
                    sources.contains(&format!("kernel void {shader}")),
                    "composed shader {shader} for {} is missing from the on-disk Metal sources",
                    wiring.gap.as_str()
                );
            }
        }
    }
}
