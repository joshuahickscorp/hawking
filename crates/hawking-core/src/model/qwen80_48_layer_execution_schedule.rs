//! Exact Qwen3-Coder-Next 48-layer *execution* schedule authority.
//!
//! This is the per-layer runtime schedule that sits on top of the sealed
//! payload schedule authority (`hawking.ascension.qwen80_48_layer_payload_schedule_authority.v1`).
//! It does not open artifacts, create Metal contexts, or execute kernels.  It
//! freezes:
//!
//! - the 36× DeltaNet / 12× GQA mixer assignment (source rule: every 4th layer);
//! - per-domain state-slot indices (36 linear slots + 12 GQA slots);
//! - the exact structural kernel sequence and dispatch count for each full layer;
//! - payload / residency requirements for a future same-runtime multi-layer host.
//!
//! Source identity is bound to revision `a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb`
//! and gravity manifest seal
//! `14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b`.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Schema for the sealed 48-layer execution schedule authority document.
pub const QWEN80_48_LAYER_EXECUTION_SCHEDULE_SCHEMA: &str =
    "hawking.ascension.qwen80_48_layer_execution_schedule_authority.v1";

/// Status of a prepared (not executed) execution schedule authority.
pub const QWEN80_48_LAYER_EXECUTION_SCHEDULE_STATUS: &str =
    "PREPARED_QWEN80_48_LAYER_EXECUTION_SCHEDULE_AUTHORITY_NOT_EXECUTED";

pub const QWEN80_MODEL_ID: &str = "Qwen3-Coder-Next-80B";
pub const QWEN80_MODEL_KEY: &str = "qwen80";
pub const QWEN80_SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
pub const QWEN80_SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
pub const QWEN80_SOURCE_CONFIG_SHA256: &str =
    "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8";
/// Gravity complete-binary manifest seal (descriptor inventory seal_sha256).
pub const QWEN80_GRAVITY_MANIFEST_SEAL_SHA256: &str =
    "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
/// Document SHA of the sealed payload schedule authority that this execution
/// schedule respects (not re-derives).
pub const QWEN80_PAYLOAD_SCHEDULE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen80_48_layer_payload_schedule_authority.v1";

pub const QWEN80_LAYERS: usize = 48;
pub const QWEN80_DELTANET_LAYERS: usize = 36;
pub const QWEN80_GQA_LAYERS: usize = 12;
pub const QWEN80_FULL_ATTENTION_INTERVAL: usize = 4;
pub const QWEN80_HIDDEN: usize = 2_048;
pub const QWEN80_TOP_K: usize = 10;
pub const QWEN80_EXPERTS: usize = 512;

/// Full DeltaNet+MoE layer: 9 mixer prefix + 14 MoE suffix.
pub const QWEN80_DELTANET_FULL_LAYER_DISPATCHES: usize = 23;
/// Full GQA+MoE layer: 9 mixer prefix + 14 MoE suffix (same total as DeltaNet).
pub const QWEN80_GQA_FULL_LAYER_DISPATCHES: usize = 23;
/// MoE suffix shared by every decoder layer after its mixer residual.
pub const QWEN80_MOE_SUFFIX_DISPATCHES: usize = 14;
/// DeltaNet / GQA mixer-prefix dispatch count (through first residual).
pub const QWEN80_MIXER_PREFIX_DISPATCHES: usize = 9;

/// Proven L0 true-MoE structural kernel order (23 dispatches).
pub const QWEN80_DELTANET_FULL_LAYER_KERNELS: [&str; QWEN80_DELTANET_FULL_LAYER_DISPATCHES] = [
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

/// DeltaNet mixer prefix only (through first residual).
pub const QWEN80_DELTANET_MIXER_PREFIX_KERNELS: [&str; QWEN80_MIXER_PREFIX_DISPATCHES] = [
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

/// MoE suffix after any mixer first residual (14 dispatches).
pub const QWEN80_MOE_SUFFIX_KERNELS: [&str; QWEN80_MOE_SUFFIX_DISPATCHES] = [
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

/// Exact GQA mixer-prefix structural order for one source-token position.
/// Derived from the layer-3 direct-packed attention stage plus input RMSNorm
/// and residual add that close the first residual boundary on a full layer.
///
/// Kernel names:
/// 1. input RMSNorm (direct packed)
/// 2–4. q/k/v projections
/// 5. qk-norm + RoPE + KV cache append
/// 6. causal GQA (`mha_decode_f32`)
/// 7. attention sigmoid gate
/// 8. o projection
/// 9. residual add
pub const QWEN80_GQA_MIXER_PREFIX_KERNELS: [&str; QWEN80_MIXER_PREFIX_DISPATCHES] = [
    "qwen_next_direct_packed_input_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_binary_sign_scale_matvec",
    "qwen_binary_sign_scale_matvec",
    "qwen80_attention_qk_norm_rope_cache",
    "mha_decode_f32",
    "qwen80_attention_apply_sigmoid_gate",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_add_residual",
];

/// Full GQA+MoE structural kernel order (9 prefix + 14 MoE).
pub const QWEN80_GQA_FULL_LAYER_KERNELS: [&str; QWEN80_GQA_FULL_LAYER_DISPATCHES] = [
    "qwen_next_direct_packed_input_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_binary_sign_scale_matvec",
    "qwen_binary_sign_scale_matvec",
    "qwen80_attention_qk_norm_rope_cache",
    "mha_decode_f32",
    "qwen80_attention_apply_sigmoid_gate",
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

/// Mixer family selected by the immutable 3× DeltaNet / 1× GQA source rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen80ExecutionMixerKind {
    DeltaNet,
    Gqa,
}

impl Qwen80ExecutionMixerKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DeltaNet => "delta_net",
            Self::Gqa => "gqa",
        }
    }

    pub const fn as_source_layer_type(self) -> &'static str {
        match self {
            Self::DeltaNet => "linear_attention",
            Self::Gqa => "full_attention",
        }
    }
}

impl fmt::Display for Qwen80ExecutionMixerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// State domain owned by one mixer layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen80ExecutionStateDomain {
    DeltaNetConvAndRecurrent,
    GqaKv,
}

impl Qwen80ExecutionStateDomain {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DeltaNetConvAndRecurrent => "delta_net_conv_and_recurrent",
            Self::GqaKv => "gqa_kv",
        }
    }
}

/// Exact source-derived mixer selection for one Qwen3-Next decoder layer.
///
/// Rule (matches `qwen80_layer_kind` and the payload schedule authority):
/// layer is GQA iff `(layer + 1) % 4 == 0` (i.e. layers 3, 7, 11, …, 47).
pub fn qwen80_execution_mixer_kind(layer: usize) -> Result<Qwen80ExecutionMixerKind, String> {
    if layer >= QWEN80_LAYERS {
        return Err(format!(
            "layer index {layer} is outside 0..{QWEN80_LAYERS} (observed layer={layer}, expected in 0..{})",
            QWEN80_LAYERS
        ));
    }
    if (layer + 1) % QWEN80_FULL_ATTENTION_INTERVAL == 0 {
        Ok(Qwen80ExecutionMixerKind::Gqa)
    } else {
        Ok(Qwen80ExecutionMixerKind::DeltaNet)
    }
}

/// DeltaNet linear state-slot index for a DeltaNet layer, or error if GQA.
///
/// Formula: `slot = layer - layer / 4` (matches runtime `linear_state_slot`).
pub fn qwen80_deltanet_state_slot(layer: usize) -> Result<usize, String> {
    match qwen80_execution_mixer_kind(layer)? {
        Qwen80ExecutionMixerKind::DeltaNet => Ok(layer - layer / QWEN80_FULL_ATTENTION_INTERVAL),
        Qwen80ExecutionMixerKind::Gqa => Err(format!(
            "layer {layer} is GQA; DeltaNet state slot is undefined (observed mixer=gqa, expected delta_net)"
        )),
    }
}

/// GQA KV state-slot index for a GQA layer, or error if DeltaNet.
///
/// Formula: `slot = layer / 4` (layers 3,7,…,47 → slots 0..11).
pub fn qwen80_gqa_state_slot(layer: usize) -> Result<usize, String> {
    match qwen80_execution_mixer_kind(layer)? {
        Qwen80ExecutionMixerKind::Gqa => Ok(layer / QWEN80_FULL_ATTENTION_INTERVAL),
        Qwen80ExecutionMixerKind::DeltaNet => Err(format!(
            "layer {layer} is DeltaNet; GQA state slot is undefined (observed mixer=delta_net, expected gqa)"
        )),
    }
}

/// Caller-owned state slot identity for one layer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Qwen80LayerStateSlot {
    pub layer: usize,
    pub slot: usize,
    pub domain: Qwen80ExecutionStateDomain,
    pub device_buffers_required_before_execution: Vec<&'static str>,
    pub rollback_buffers_required_before_execution: Vec<&'static str>,
    /// Multi-layer hosts must never share mutable state across slots.
    pub exclusive_caller_owned_slot: bool,
}

fn deltanet_state_slot(layer: usize, slot: usize) -> Qwen80LayerStateSlot {
    Qwen80LayerStateSlot {
        layer,
        slot,
        domain: Qwen80ExecutionStateDomain::DeltaNetConvAndRecurrent,
        device_buffers_required_before_execution: vec![
            "deltanet_conv_history",
            "deltanet_recurrent_state",
        ],
        rollback_buffers_required_before_execution: vec![
            "deltanet_conv_history_rollback",
            "deltanet_recurrent_state_rollback",
        ],
        exclusive_caller_owned_slot: true,
    }
}

fn gqa_state_slot(layer: usize, slot: usize) -> Qwen80LayerStateSlot {
    Qwen80LayerStateSlot {
        layer,
        slot,
        domain: Qwen80ExecutionStateDomain::GqaKv,
        device_buffers_required_before_execution: vec!["gqa_key_cache", "gqa_value_cache"],
        rollback_buffers_required_before_execution: vec![
            "gqa_key_cache_rollback",
            "gqa_value_cache_rollback",
        ],
        exclusive_caller_owned_slot: true,
    }
}

/// Payload / residency requirements for one full layer in a same-runtime graph.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Qwen80LayerResidencyRequirements {
    pub input_hidden_elements: usize,
    pub output_hidden_elements: usize,
    pub mixer_compact_payloads_required: bool,
    pub moe_fixed_compact_payloads_required: bool,
    pub moe_routed_top10_compact_payloads_required: bool,
    pub shared_expert_compact_payloads_required: bool,
    pub state_slot_zeroed_or_caller_restored_before_encode: bool,
    pub second_residual_is_next_layer_input: bool,
}

/// Exact per-layer execution schedule entry.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Qwen80LayerExecutionSchedule {
    pub layer: usize,
    pub mixer: Qwen80ExecutionMixerKind,
    pub source_layer_type: &'static str,
    pub state_slot: Qwen80LayerStateSlot,
    pub mixer_prefix_dispatch_count: usize,
    pub moe_suffix_dispatch_count: usize,
    pub full_layer_dispatch_count: usize,
    pub mixer_prefix_kernel_names: Vec<&'static str>,
    pub moe_suffix_kernel_names: Vec<&'static str>,
    pub full_layer_kernel_names: Vec<&'static str>,
    pub residency: Qwen80LayerResidencyRequirements,
    /// Same-runtime multi-layer host readiness for this layer's encode path.
    /// True for every source-scheduled mixer once its full-layer same-runtime
    /// encode (prefix + MoE suffix) is wired into the multi-layer host.
    /// Physical capture still proves device parity; readiness here only means
    /// the host will not refuse at CPU preflight for that mixer family.
    pub same_runtime_full_layer_encode_ready: bool,
}

/// Build the exact execution schedule for one layer in `0..48`.
pub fn qwen80_layer_execution_schedule(layer: usize) -> Result<Qwen80LayerExecutionSchedule, String> {
    let mixer = qwen80_execution_mixer_kind(layer)?;
    let (state_slot, prefix, full) = match mixer {
        Qwen80ExecutionMixerKind::DeltaNet => {
            let slot = qwen80_deltanet_state_slot(layer)?;
            (
                deltanet_state_slot(layer, slot),
                QWEN80_DELTANET_MIXER_PREFIX_KERNELS.as_slice(),
                QWEN80_DELTANET_FULL_LAYER_KERNELS.as_slice(),
            )
        }
        Qwen80ExecutionMixerKind::Gqa => {
            let slot = qwen80_gqa_state_slot(layer)?;
            (
                gqa_state_slot(layer, slot),
                QWEN80_GQA_MIXER_PREFIX_KERNELS.as_slice(),
                QWEN80_GQA_FULL_LAYER_KERNELS.as_slice(),
            )
        }
    };
    if prefix.len() != QWEN80_MIXER_PREFIX_DISPATCHES
        || full.len() != QWEN80_DELTANET_FULL_LAYER_DISPATCHES
    {
        return Err(format!(
            "layer {layer} kernel table length drifted: prefix={} full={} (expected prefix={} full={})",
            prefix.len(),
            full.len(),
            QWEN80_MIXER_PREFIX_DISPATCHES,
            QWEN80_DELTANET_FULL_LAYER_DISPATCHES
        ));
    }
    Ok(Qwen80LayerExecutionSchedule {
        layer,
        mixer,
        source_layer_type: mixer.as_source_layer_type(),
        state_slot,
        mixer_prefix_dispatch_count: QWEN80_MIXER_PREFIX_DISPATCHES,
        moe_suffix_dispatch_count: QWEN80_MOE_SUFFIX_DISPATCHES,
        full_layer_dispatch_count: full.len(),
        mixer_prefix_kernel_names: prefix.to_vec(),
        moe_suffix_kernel_names: QWEN80_MOE_SUFFIX_KERNELS.to_vec(),
        full_layer_kernel_names: full.to_vec(),
        residency: Qwen80LayerResidencyRequirements {
            input_hidden_elements: QWEN80_HIDDEN,
            output_hidden_elements: QWEN80_HIDDEN,
            mixer_compact_payloads_required: true,
            moe_fixed_compact_payloads_required: true,
            moe_routed_top10_compact_payloads_required: true,
            shared_expert_compact_payloads_required: true,
            state_slot_zeroed_or_caller_restored_before_encode: true,
            second_residual_is_next_layer_input: true,
        },
        // Both DeltaNet and GQA full-layer same-runtime encodes are wired.
        same_runtime_full_layer_encode_ready: true,
    })
}

/// Frozen schedule for every decoder layer `0..47`.
pub fn qwen80_all_48_layer_execution_schedules() -> Result<Vec<Qwen80LayerExecutionSchedule>, String>
{
    (0..QWEN80_LAYERS)
        .map(qwen80_layer_execution_schedule)
        .collect()
}

/// Cumulative structural kernel trace for layers `0..layer_count` (exclusive end).
///
/// `layer_count` is the number of sequential layers starting at 0
/// (e.g. `3` means L0+L1+L2; `4` means L0..L3 and crosses the first GQA layer).
///
/// `allow_scheduled_gqa` is retained for authority/document callers that used
/// the pre-encode-ready flag; it is a no-op once every mixer family is
/// encode-ready (both DeltaNet and GQA).  Still refuses unknown layer counts
/// with observed vs expected values.
pub fn qwen80_multi_layer_structural_kernel_trace(
    layer_count: usize,
    allow_scheduled_gqa: bool,
) -> Result<Vec<&'static str>, String> {
    let _ = allow_scheduled_gqa;
    if layer_count == 0 || layer_count > QWEN80_LAYERS {
        return Err(format!(
            "layer_count={layer_count} is outside 1..={QWEN80_LAYERS} (observed layer_count={layer_count}, expected in 1..={QWEN80_LAYERS})"
        ));
    }
    let mut kernels = Vec::with_capacity(layer_count * QWEN80_DELTANET_FULL_LAYER_DISPATCHES);
    for layer in 0..layer_count {
        let schedule = qwen80_layer_execution_schedule(layer)?;
        if !schedule.same_runtime_full_layer_encode_ready {
            return Err(format!(
                "layer {layer} mixer={} is not same-runtime full-layer encode ready (observed ready=false, expected true)",
                schedule.mixer
            ));
        }
        kernels.extend(schedule.full_layer_kernel_names.iter().copied());
    }
    Ok(kernels)
}

/// Total dispatch count for layers `0..layer_count`.
pub fn qwen80_multi_layer_total_dispatches(
    layer_count: usize,
    allow_scheduled_gqa: bool,
) -> Result<usize, String> {
    Ok(qwen80_multi_layer_structural_kernel_trace(layer_count, allow_scheduled_gqa)?.len())
}

/// Source identity binding required by every multi-layer authority document.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Qwen80ExecutionScheduleSourceBinding {
    pub model_id: &'static str,
    pub model_key: &'static str,
    pub source_repository: &'static str,
    pub source_revision: &'static str,
    pub source_config_sha256: &'static str,
    pub gravity_manifest_seal_sha256: &'static str,
    pub payload_schedule_authority_schema: &'static str,
    pub full_attention_interval: usize,
    pub layer_count: usize,
    pub deltanet_layers: usize,
    pub gqa_layers: usize,
}

impl Qwen80ExecutionScheduleSourceBinding {
    pub fn exact() -> Self {
        Self {
            model_id: QWEN80_MODEL_ID,
            model_key: QWEN80_MODEL_KEY,
            source_repository: QWEN80_SOURCE_REPOSITORY,
            source_revision: QWEN80_SOURCE_REVISION,
            source_config_sha256: QWEN80_SOURCE_CONFIG_SHA256,
            gravity_manifest_seal_sha256: QWEN80_GRAVITY_MANIFEST_SEAL_SHA256,
            payload_schedule_authority_schema: QWEN80_PAYLOAD_SCHEDULE_AUTHORITY_SCHEMA,
            full_attention_interval: QWEN80_FULL_ATTENTION_INTERVAL,
            layer_count: QWEN80_LAYERS,
            deltanet_layers: QWEN80_DELTANET_LAYERS,
            gqa_layers: QWEN80_GQA_LAYERS,
        }
    }

    pub fn validate_exact(&self) -> Result<(), String> {
        let expected = Self::exact();
        if self != &expected {
            return Err(format!(
                "source binding drifted: observed model_id={} revision={} gravity_seal={} deltanet={} gqa={}, expected model_id={} revision={} gravity_seal={} deltanet={} gqa={}",
                self.model_id,
                self.source_revision,
                self.gravity_manifest_seal_sha256,
                self.deltanet_layers,
                self.gqa_layers,
                expected.model_id,
                expected.source_revision,
                expected.gravity_manifest_seal_sha256,
                expected.deltanet_layers,
                expected.gqa_layers
            ));
        }
        Ok(())
    }
}

/// Validate the global 36/12 counts and slot uniqueness.
pub fn validate_full_48_layer_schedule(
    layers: &[Qwen80LayerExecutionSchedule],
) -> Result<(), String> {
    if layers.len() != QWEN80_LAYERS {
        return Err(format!(
            "schedule length={} observed, expected exactly {QWEN80_LAYERS}",
            layers.len()
        ));
    }
    let mut deltanet = 0usize;
    let mut gqa = 0usize;
    let mut deltanet_slots = Vec::new();
    let mut gqa_slots = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        if layer.layer != index {
            return Err(format!(
                "schedule[{index}].layer={} drifted (expected {index})",
                layer.layer
            ));
        }
        let expected = qwen80_layer_execution_schedule(index)?;
        if layer.mixer != expected.mixer {
            return Err(format!(
                "layer {index} mixer observed={}, expected={}",
                layer.mixer, expected.mixer
            ));
        }
        if layer.full_layer_dispatch_count != expected.full_layer_dispatch_count {
            return Err(format!(
                "layer {index} dispatch_count observed={}, expected={}",
                layer.full_layer_dispatch_count, expected.full_layer_dispatch_count
            ));
        }
        if layer.full_layer_kernel_names != expected.full_layer_kernel_names {
            return Err(format!(
                "layer {index} kernel sequence drifted (observed len={}, expected len={})",
                layer.full_layer_kernel_names.len(),
                expected.full_layer_kernel_names.len()
            ));
        }
        match layer.mixer {
            Qwen80ExecutionMixerKind::DeltaNet => {
                deltanet += 1;
                deltanet_slots.push(layer.state_slot.slot);
            }
            Qwen80ExecutionMixerKind::Gqa => {
                gqa += 1;
                gqa_slots.push(layer.state_slot.slot);
            }
        }
    }
    if deltanet != QWEN80_DELTANET_LAYERS || gqa != QWEN80_GQA_LAYERS {
        return Err(format!(
            "mixer counts drifted: deltanet={deltanet} gqa={gqa}, expected deltanet={QWEN80_DELTANET_LAYERS} gqa={QWEN80_GQA_LAYERS}"
        ));
    }
    let mut sorted_dn = deltanet_slots.clone();
    sorted_dn.sort_unstable();
    let expected_dn: Vec<usize> = (0..QWEN80_DELTANET_LAYERS).collect();
    if sorted_dn != expected_dn {
        return Err(format!(
            "DeltaNet state slots are not exactly 0..{QWEN80_DELTANET_LAYERS}: observed={sorted_dn:?}"
        ));
    }
    let mut sorted_gqa = gqa_slots.clone();
    sorted_gqa.sort_unstable();
    let expected_gqa: Vec<usize> = (0..QWEN80_GQA_LAYERS).collect();
    if sorted_gqa != expected_gqa {
        return Err(format!(
            "GQA state slots are not exactly 0..{QWEN80_GQA_LAYERS}: observed={sorted_gqa:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixer_assignment_is_exact_36_deltanet_12_gqa() {
        let mut dn = 0;
        let mut gqa = 0;
        for layer in 0..QWEN80_LAYERS {
            match qwen80_execution_mixer_kind(layer).unwrap() {
                Qwen80ExecutionMixerKind::DeltaNet => dn += 1,
                Qwen80ExecutionMixerKind::Gqa => {
                    gqa += 1;
                    assert_eq!((layer + 1) % 4, 0, "GQA layer {layer}");
                }
            }
        }
        assert_eq!(dn, 36);
        assert_eq!(gqa, 12);
        assert_eq!(qwen80_execution_mixer_kind(0).unwrap(), Qwen80ExecutionMixerKind::DeltaNet);
        assert_eq!(qwen80_execution_mixer_kind(1).unwrap(), Qwen80ExecutionMixerKind::DeltaNet);
        assert_eq!(qwen80_execution_mixer_kind(2).unwrap(), Qwen80ExecutionMixerKind::DeltaNet);
        assert_eq!(qwen80_execution_mixer_kind(3).unwrap(), Qwen80ExecutionMixerKind::Gqa);
        assert_eq!(qwen80_execution_mixer_kind(47).unwrap(), Qwen80ExecutionMixerKind::Gqa);
    }

    #[test]
    fn state_slots_are_exclusive_and_contiguous() {
        assert_eq!(qwen80_deltanet_state_slot(0).unwrap(), 0);
        assert_eq!(qwen80_deltanet_state_slot(1).unwrap(), 1);
        assert_eq!(qwen80_deltanet_state_slot(2).unwrap(), 2);
        assert_eq!(qwen80_deltanet_state_slot(4).unwrap(), 3);
        assert!(qwen80_deltanet_state_slot(3).is_err());
        assert_eq!(qwen80_gqa_state_slot(3).unwrap(), 0);
        assert_eq!(qwen80_gqa_state_slot(7).unwrap(), 1);
        assert_eq!(qwen80_gqa_state_slot(47).unwrap(), 11);
        assert!(qwen80_gqa_state_slot(0).is_err());
    }

    #[test]
    fn full_48_layer_schedule_validates() {
        let layers = qwen80_all_48_layer_execution_schedules().unwrap();
        validate_full_48_layer_schedule(&layers).unwrap();
        assert!(layers[0].same_runtime_full_layer_encode_ready);
        assert!(layers[1].same_runtime_full_layer_encode_ready);
        assert!(layers[2].same_runtime_full_layer_encode_ready);
        assert!(layers[3].same_runtime_full_layer_encode_ready);
        assert!(layers[7].same_runtime_full_layer_encode_ready);
        assert!(layers[47].same_runtime_full_layer_encode_ready);
        assert_eq!(layers[0].full_layer_dispatch_count, 23);
        assert_eq!(layers[3].full_layer_dispatch_count, 23);
        assert_eq!(
            layers[0].full_layer_kernel_names,
            QWEN80_DELTANET_FULL_LAYER_KERNELS
        );
        assert_eq!(layers[3].full_layer_kernel_names, QWEN80_GQA_FULL_LAYER_KERNELS);
        assert_eq!(
            layers[3].mixer_prefix_kernel_names,
            QWEN80_GQA_MIXER_PREFIX_KERNELS
        );
        assert_eq!(layers[3].mixer_prefix_dispatch_count, QWEN80_MIXER_PREFIX_DISPATCHES);
        assert_eq!(layers[3].full_layer_dispatch_count, QWEN80_GQA_FULL_LAYER_DISPATCHES);
        assert_eq!(
            layers[3].state_slot.device_buffers_required_before_execution,
            vec!["gqa_key_cache", "gqa_value_cache"]
        );
        assert_eq!(
            layers[3].state_slot.rollback_buffers_required_before_execution,
            vec!["gqa_key_cache_rollback", "gqa_value_cache_rollback"]
        );
        assert!(layers[3].state_slot.exclusive_caller_owned_slot);
    }

    #[test]
    fn multi_layer_trace_l0_l2_is_69_deltanet_dispatches() {
        let trace = qwen80_multi_layer_structural_kernel_trace(3, false).unwrap();
        assert_eq!(trace.len(), 69);
        assert_eq!(&trace[..23], &QWEN80_DELTANET_FULL_LAYER_KERNELS);
        assert_eq!(&trace[23..46], &QWEN80_DELTANET_FULL_LAYER_KERNELS);
        assert_eq!(&trace[46..69], &QWEN80_DELTANET_FULL_LAYER_KERNELS);
        // L0..L3 crosses GQA L3: 3×DeltaNet + 1×GQA = 92 dispatches.
        let with_gqa = qwen80_multi_layer_structural_kernel_trace(4, false).unwrap();
        assert_eq!(with_gqa.len(), 92);
        assert_eq!(&with_gqa[69..92], &QWEN80_GQA_FULL_LAYER_KERNELS);
        assert_eq!(
            qwen80_multi_layer_total_dispatches(4, false).unwrap(),
            92
        );
        let err = qwen80_multi_layer_structural_kernel_trace(0, false).unwrap_err();
        assert!(err.contains("layer_count=0"), "{err}");
        assert!(err.contains("1..=48"), "{err}");
    }

    #[test]
    fn source_binding_exact() {
        Qwen80ExecutionScheduleSourceBinding::exact()
            .validate_exact()
            .unwrap();
        let mut bad = Qwen80ExecutionScheduleSourceBinding::exact();
        bad.source_revision = "deadbeef";
        assert!(bad.validate_exact().is_err());
    }

    #[test]
    fn layer_out_of_range_refuses_with_values() {
        let err = qwen80_execution_mixer_kind(48).unwrap_err();
        assert!(err.contains("48"));
        assert!(err.contains("0..48"));
    }

    #[test]
    fn gqa_full_layer_dispatch_count_and_frozen_kernel_order_are_declared() {
        assert_eq!(QWEN80_GQA_FULL_LAYER_DISPATCHES, 23);
        assert_eq!(QWEN80_MIXER_PREFIX_DISPATCHES, 9);
        assert_eq!(QWEN80_GQA_MIXER_PREFIX_KERNELS.len(), 9);
        assert_eq!(QWEN80_GQA_FULL_LAYER_KERNELS.len(), 23);
        assert_eq!(
            QWEN80_GQA_MIXER_PREFIX_KERNELS,
            [
                "qwen_next_direct_packed_input_rmsnorm",
                "qwen_binary_sign_scale_matvec",
                "qwen_binary_sign_scale_matvec",
                "qwen_binary_sign_scale_matvec",
                "qwen80_attention_qk_norm_rope_cache",
                "mha_decode_f32",
                "qwen80_attention_apply_sigmoid_gate",
                "qwen_binary_sign_scale_matvec",
                "qwen_next_add_residual",
            ]
        );
        // MoE suffix is shared with DeltaNet and closes every full layer.
        assert_eq!(
            &QWEN80_GQA_FULL_LAYER_KERNELS[9..],
            QWEN80_MOE_SUFFIX_KERNELS.as_slice()
        );
        // All twelve GQA layers share the same structural table and exclusive KV slots.
        for (index, layer) in [3usize, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
            .into_iter()
            .enumerate()
        {
            let schedule = qwen80_layer_execution_schedule(layer).unwrap();
            assert!(schedule.same_runtime_full_layer_encode_ready);
            assert_eq!(schedule.mixer, Qwen80ExecutionMixerKind::Gqa);
            assert_eq!(schedule.state_slot.slot, index);
            assert_eq!(
                schedule.state_slot.device_buffers_required_before_execution,
                vec!["gqa_key_cache", "gqa_value_cache"]
            );
            assert_eq!(
                schedule.state_slot.rollback_buffers_required_before_execution,
                vec!["gqa_key_cache_rollback", "gqa_value_cache_rollback"]
            );
            assert!(schedule.state_slot.exclusive_caller_owned_slot);
            assert_eq!(schedule.full_layer_kernel_names, QWEN80_GQA_FULL_LAYER_KERNELS);
        }
    }

    #[test]
    fn multi_layer_trace_refuses_zero_and_oversized_layer_count_with_values() {
        let err0 = qwen80_multi_layer_structural_kernel_trace(0, false).unwrap_err();
        assert!(err0.contains("layer_count=0"), "{err0}");
        assert!(err0.contains("1..=48"), "{err0}");
        let err49 = qwen80_multi_layer_structural_kernel_trace(49, false).unwrap_err();
        assert!(err49.contains("layer_count=49"), "{err49}");
        assert!(err49.contains("1..=48"), "{err49}");
    }
}
