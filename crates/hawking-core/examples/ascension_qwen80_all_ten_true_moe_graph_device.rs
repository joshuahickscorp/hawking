//! Staged source-bound all-ten Qwen80 layer-0 true-MoE device graph.
//!
//! This is an isolated host encoder, not a live runtime entrypoint.  It starts
//! from one already-produced, source-bound DeltaNet `first_residual` device
//! buffer and encodes the exact L0 MoE suffix in one command buffer:
//!
//! `postnorm -> router top-10 -> ten direct-packed waves -> shared expert
//! -> fixed route[0..9]/shared/first-residual combine`.
//!
//! The generic ten-route shader is staged beside this file but intentionally
//! unregistered.  `main` only prints the CPU/build-time ABI plan; it creates
//! no Metal context and cannot dispatch.  A future separately authorized
//! integration must append the shader to `metal/mod.rs`, create a narrow
//! catalog-to-buffer bridge, and provide a strict outer-reaped component lease.

#[cfg(target_os = "macos")]
use hawking_core::metal::{PinnedBuffer, TokenCommandBuffer};
#[cfg(target_os = "macos")]
use hawking_core::model::qwen80_complete_runtime::{
    Qwen80AllTenRoutedExpertPlanAuthority, Qwen80AllTenTrueMoeDeviceBridge,
    Qwen80CanonicalLinearLayerCpuInput, Qwen80CompleteArtifactCatalog, Qwen80CompleteNativeRuntime,
    Qwen80CompleteRuntimeOptions, Qwen80L0TrueMoeFixedDeviceBuffers,
};
#[cfg(target_os = "macos")]
use hawking_core::model::qwen_complete_binary::{CompleteBinaryAdmission, QwenCompleteBinaryModel};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

// Keep scalar ABI binding local to this staged component graph.  This is the
// same `set_bytes` primitive used by the already sealed Q80 component probes;
// it neither creates a Metal context nor turns this build-only encoder into a
// live runtime entrypoint.
#[cfg(target_os = "macos")]
trait StageSetScalar {
    fn stage_set_u32(&self, index: u64, value: u32);
    fn stage_set_f32(&self, index: u64, value: f32);
}

#[cfg(target_os = "macos")]
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

const HIDDEN: u32 = 2_048;
const INTERMEDIATE: u32 = 512;
const ROUTES: u32 = 10;
const EXPERTS: u32 = 512;
const GROUP: u32 = 128;
const ROUTE_SIGN_BYTES: u32 = INTERMEDIATE * HIDDEN / 8;
const ROUTE_SCALE_BYTES: u32 = INTERMEDIATE * HIDDEN / GROUP * 2;

const CHILD_SCHEMA: &str = "hawking.ascension.qwen80_all_ten_true_moe_graph_device.v1";
const CHILD_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_LAYER0_TRUE_INPUT_ALL_TEN_ROUTE_SHARED_SECOND_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const ADMISSION_POINTER_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1";
const ADMISSION_POINTER_STATUS: &str = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED";
const ADMISSION_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1";
const ADMISSION_RECEIPT_STATUS: &str =
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";
const TYPED_BRIDGE_SCHEMA: &str = "hawking.ascension.qwen80_all_ten_true_moe_source_bridge.v1";
const TYPED_BRIDGE_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_ALL_TEN_TRUE_MOE_SOURCE_BRIDGE_READY_FOR_DEVICE_LEASE";
const FIXED_ABI_SCHEMA: &str = "hawking.ascension.qwen80_l0_true_moe_fixed_payload_contract.v1";
const FIXED_ABI_STATUS: &str = "PREPARED_QWEN80_L0_TRUE_MOE_FIXED_SUFFIX_PAYLOAD_PLAN_NOT_EXECUTED";
const LEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_true_input_all_ten_moe_graph_quiet_metal_lease.v1";
const LEASE_STATUS: &str =
    "GRANTED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_NON_TIMED_DEVICE_PARITY_LEASE";
const STRICT_METAL_MODE: &str = "metal";
const SOURCE_TOKEN_ID: u32 = 1;
const PREFIX_DISPATCHES: usize = 9;
const SUFFIX_DISPATCHES: usize = 14;
const FULL_COMPONENT_DISPATCHES: usize = PREFIX_DISPATCHES + SUFFIX_DISPATCHES;

#[allow(dead_code)] // consumed by the static source/ABI regression below.
const STAGED_SHADER: &str = include_str!("../shaders/qwen80_all_ten_routed_expert_wave.metal");

#[derive(Clone, Debug, Eq, PartialEq)]
struct DispatchSpec {
    kernel: &'static str,
    grid: (u32, u32, u32),
    threadgroup: (u32, u32, u32),
    purpose: &'static str,
}

fn staged_dispatch_plan() -> Vec<DispatchSpec> {
    vec![
        DispatchSpec {
            kernel: "qwen80_postnorm_router_top10_rmsnorm",
            grid: (256, 1, 1),
            threadgroup: (256, 1, 1),
            purpose: "post-attention RMSNorm(first_residual) -> postnorm hidden",
        },
        DispatchSpec {
            kernel: "qwen80_postnorm_router_top10_matvec",
            grid: (256, EXPERTS, 1),
            threadgroup: (256, 1, 1),
            purpose: "direct-packed router -> all 512 logits",
        },
        DispatchSpec {
            kernel: "qwen80_postnorm_router_top10_select",
            grid: (1, 1, 1),
            threadgroup: (1, 1, 1),
            purpose: "source tie-policy top-10 IDs and normalized weights",
        },
        DispatchSpec {
            kernel: "qwen80_all_ten_routed_wave_route_guard",
            grid: (1, 1, 1),
            threadgroup: (1, 1, 1),
            purpose: "reject router IDs/weights that differ from the retained all-ten plan",
        },
        DispatchSpec {
            kernel: "qwen80_all_ten_routed_wave_gate_up",
            grid: (256, INTERMEDIATE, ROUTES),
            threadgroup: (256, 1, 1),
            purpose: "all ten direct-packed gate/up projections, route in Z",
        },
        DispatchSpec {
            kernel: "qwen80_all_ten_routed_wave_swiglu",
            grid: (INTERMEDIATE, ROUTES, 1),
            threadgroup: (256, 1, 1),
            purpose: "all ten SiLU(gate) * up activations",
        },
        DispatchSpec {
            kernel: "qwen80_all_ten_routed_wave_down_weighted",
            grid: (256, HIDDEN, ROUTES),
            threadgroup: (256, 1, 1),
            purpose: "all ten direct-packed down projections and source weights",
        },
        DispatchSpec {
            kernel: "qwen80_shared_expert_wave_gate_up",
            grid: (256, INTERMEDIATE, 1),
            threadgroup: (256, 1, 1),
            purpose: "direct-packed shared gate/up",
        },
        DispatchSpec {
            kernel: "qwen80_shared_expert_wave_swiglu",
            grid: (INTERMEDIATE, 1, 1),
            threadgroup: (256, 1, 1),
            purpose: "shared SiLU(gate) * up",
        },
        DispatchSpec {
            kernel: "qwen80_shared_expert_wave_down",
            grid: (256, HIDDEN, 1),
            threadgroup: (256, 1, 1),
            purpose: "direct-packed shared down",
        },
        DispatchSpec {
            kernel: "qwen80_shared_expert_wave_scalar_gate",
            grid: (256, 1, 1),
            threadgroup: (256, 1, 1),
            purpose: "source shared-expert scalar gate",
        },
        DispatchSpec {
            kernel: "qwen80_shared_expert_wave_apply_sigmoid_gate",
            grid: (HIDDEN, 1, 1),
            threadgroup: (256, 1, 1),
            purpose: "gated shared output",
        },
        DispatchSpec {
            kernel: "qwen80_moe_wave_aggregate_second_residual_route_sum",
            grid: (HIDDEN, 1, 1),
            threadgroup: (256, 1, 1),
            purpose: "fixed f32 route[0] through route[9] sum",
        },
        DispatchSpec {
            kernel: "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
            grid: (HIDDEN, 1, 1),
            threadgroup: (256, 1, 1),
            purpose: "routed sum + gated shared + exact first residual",
        },
    ]
}

fn validate_staged_plan(plan: &[DispatchSpec]) -> Result<(), String> {
    if plan.len() != 14 {
        return Err(format!(
            "L0 true-MoE graph has {} dispatches, expected 14",
            plan.len()
        ));
    }
    let expected = [
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
    if plan.iter().map(|stage| stage.kernel).ne(expected) {
        return Err("L0 true-MoE graph changed source operation order".into());
    }
    let route_gate = &plan[4];
    let route_down = &plan[6];
    if route_gate.grid != (256, INTERMEDIATE, ROUTES)
        || route_down.grid != (256, HIDDEN, ROUTES)
        || route_gate.threadgroup != (256, 1, 1)
        || route_down.threadgroup != (256, 1, 1)
    {
        return Err("all-ten route kernels lost their exact route-in-Z geometry".into());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub struct Qwen80AllTenTrueMoeGraphBuffers<'a> {
    /// The only externally produced input. The caller must keep it in the
    /// same command graph as the DeltaNet mixer and retain its output hash.
    pub first_residual: &'a PinnedBuffer,
    pub postnorm_signs: &'a PinnedBuffer,
    pub postnorm_scales: &'a PinnedBuffer,
    pub postnorm_hidden: &'a PinnedBuffer,
    pub router_signs: &'a PinnedBuffer,
    pub router_scales: &'a PinnedBuffer,
    pub router_logits: &'a PinnedBuffer,
    pub router_probabilities: &'a PinnedBuffer,
    /// Device router output. It is distinct from the immutable all-ten plan
    /// buffers and must pass the guard before a capture can be sealed.
    pub router_route_ids: &'a PinnedBuffer,
    pub router_route_weights: &'a PinnedBuffer,
    pub expected_route_ids: &'a PinnedBuffer,
    pub expected_route_weights: &'a PinnedBuffer,
    /// One `u32` written by `qwen80_all_ten_routed_wave_route_guard`.
    /// A terminal capture must reject any value other than one.
    pub route_guard: &'a PinnedBuffer,
    /// Each direct-packed route projection is concatenated in exact source
    /// route order 0..9. No name lookup happens after admission.
    pub route_gate_signs: &'a PinnedBuffer,
    pub route_gate_scales: &'a PinnedBuffer,
    pub route_up_signs: &'a PinnedBuffer,
    pub route_up_scales: &'a PinnedBuffer,
    pub route_down_signs: &'a PinnedBuffer,
    pub route_down_scales: &'a PinnedBuffer,
    pub route_gate: &'a PinnedBuffer,
    pub route_up: &'a PinnedBuffer,
    pub route_activated: &'a PinnedBuffer,
    pub route_weighted: &'a PinnedBuffer,
    pub shared_gate_signs: &'a PinnedBuffer,
    pub shared_gate_scales: &'a PinnedBuffer,
    pub shared_up_signs: &'a PinnedBuffer,
    pub shared_up_scales: &'a PinnedBuffer,
    pub shared_down_signs: &'a PinnedBuffer,
    pub shared_down_scales: &'a PinnedBuffer,
    pub shared_scalar_signs: &'a PinnedBuffer,
    pub shared_scalar_scales: &'a PinnedBuffer,
    pub shared_gate: &'a PinnedBuffer,
    pub shared_up: &'a PinnedBuffer,
    pub shared_activated: &'a PinnedBuffer,
    pub shared_output: &'a PinnedBuffer,
    pub shared_scalar_logit: &'a PinnedBuffer,
    pub gated_shared: &'a PinnedBuffer,
    pub routed_sum: &'a PinnedBuffer,
    pub second_residual: &'a PinnedBuffer,
}

/// Buffers that are not supplied by the strict admitted all-ten source
/// bridge.  The route compact bodies and expected IDs/weights must come from
/// [`Qwen80AllTenTrueMoeDeviceBridge`], preventing an integration caller from
/// substituting a filename-selected route tensor after catalog admission.
#[cfg(target_os = "macos")]
pub struct Qwen80AllTenTrueMoeGraphFixedBuffers<'a> {
    pub postnorm_signs: &'a PinnedBuffer,
    pub postnorm_scales: &'a PinnedBuffer,
    pub postnorm_hidden: &'a PinnedBuffer,
    pub router_signs: &'a PinnedBuffer,
    pub router_scales: &'a PinnedBuffer,
    pub router_logits: &'a PinnedBuffer,
    pub router_probabilities: &'a PinnedBuffer,
    pub router_route_ids: &'a PinnedBuffer,
    pub router_route_weights: &'a PinnedBuffer,
    pub route_guard: &'a PinnedBuffer,
    pub route_gate: &'a PinnedBuffer,
    pub route_up: &'a PinnedBuffer,
    pub route_activated: &'a PinnedBuffer,
    pub route_weighted: &'a PinnedBuffer,
    pub shared_gate_signs: &'a PinnedBuffer,
    pub shared_gate_scales: &'a PinnedBuffer,
    pub shared_up_signs: &'a PinnedBuffer,
    pub shared_up_scales: &'a PinnedBuffer,
    pub shared_down_signs: &'a PinnedBuffer,
    pub shared_down_scales: &'a PinnedBuffer,
    pub shared_scalar_signs: &'a PinnedBuffer,
    pub shared_scalar_scales: &'a PinnedBuffer,
    pub shared_gate: &'a PinnedBuffer,
    pub shared_up: &'a PinnedBuffer,
    pub shared_activated: &'a PinnedBuffer,
    pub shared_output: &'a PinnedBuffer,
    pub shared_scalar_logit: &'a PinnedBuffer,
    pub gated_shared: &'a PinnedBuffer,
    pub routed_sum: &'a PinnedBuffer,
    pub second_residual: &'a PinnedBuffer,
}

#[cfg(target_os = "macos")]
impl<'a> Qwen80AllTenTrueMoeGraphBuffers<'a> {
    /// Join the only legal admitted all-ten compact-body upload with the
    /// caller's already allocated non-route work buffers. This constructor
    /// preserves the original DeltaNet `first_residual` handle; it does not
    /// copy it through CPU memory or create a context/dispatch.
    pub fn from_admitted_route_bridge(
        bridge: &'a Qwen80AllTenTrueMoeDeviceBridge,
        fixed: Qwen80AllTenTrueMoeGraphFixedBuffers<'a>,
    ) -> Self {
        Self {
            first_residual: bridge.first_residual(),
            postnorm_signs: fixed.postnorm_signs,
            postnorm_scales: fixed.postnorm_scales,
            postnorm_hidden: fixed.postnorm_hidden,
            router_signs: fixed.router_signs,
            router_scales: fixed.router_scales,
            router_logits: fixed.router_logits,
            router_probabilities: fixed.router_probabilities,
            router_route_ids: fixed.router_route_ids,
            router_route_weights: fixed.router_route_weights,
            expected_route_ids: bridge.expected_route_ids(),
            expected_route_weights: bridge.expected_route_weights(),
            route_guard: fixed.route_guard,
            route_gate_signs: bridge.route_gate_signs(),
            route_gate_scales: bridge.route_gate_scales(),
            route_up_signs: bridge.route_up_signs(),
            route_up_scales: bridge.route_up_scales(),
            route_down_signs: bridge.route_down_signs(),
            route_down_scales: bridge.route_down_scales(),
            route_gate: fixed.route_gate,
            route_up: fixed.route_up,
            route_activated: fixed.route_activated,
            route_weighted: fixed.route_weighted,
            shared_gate_signs: fixed.shared_gate_signs,
            shared_gate_scales: fixed.shared_gate_scales,
            shared_up_signs: fixed.shared_up_signs,
            shared_up_scales: fixed.shared_up_scales,
            shared_down_signs: fixed.shared_down_signs,
            shared_down_scales: fixed.shared_down_scales,
            shared_scalar_signs: fixed.shared_scalar_signs,
            shared_scalar_scales: fixed.shared_scalar_scales,
            shared_gate: fixed.shared_gate,
            shared_up: fixed.shared_up,
            shared_activated: fixed.shared_activated,
            shared_output: fixed.shared_output,
            shared_scalar_logit: fixed.shared_scalar_logit,
            gated_shared: fixed.gated_shared,
            routed_sum: fixed.routed_sum,
            second_residual: fixed.second_residual,
        }
    }
}

/// Encode the L0 true-MoE suffix into an already-open command buffer.
///
/// The caller MUST only invoke this after a strict source/artifact admission,
/// must retain the very same `first_residual` from the DeltaNet mixer, and
/// must fence/read back every source-order route witness after this encoder.
/// It intentionally does not create a context, allocate buffers, reload a
/// manifest, or commit a command buffer.
#[cfg(target_os = "macos")]
pub fn encode_all_ten_true_moe_from_first_residual(
    command: &mut TokenCommandBuffer<'_>,
    buffers: &Qwen80AllTenTrueMoeGraphBuffers<'_>,
) -> Result<usize, String> {
    let before = command.dispatch_count();
    command
        .dispatch_threads(
            "qwen80_postnorm_router_top10_rmsnorm",
            (256, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(buffers.first_residual), 0);
                encoder.set_buffer(1, Some(buffers.postnorm_signs), 0);
                encoder.set_buffer(2, Some(buffers.postnorm_scales), 0);
                encoder.set_buffer(3, Some(buffers.postnorm_hidden), 0);
                encoder.stage_set_u32(4, HIDDEN);
                encoder.stage_set_u32(5, GROUP);
                encoder.stage_set_f32(6, 1.0e-6);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_postnorm_router_top10_matvec",
            (256, EXPERTS, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(buffers.router_signs), 0);
                encoder.set_buffer(1, Some(buffers.router_scales), 0);
                encoder.set_buffer(2, Some(buffers.postnorm_hidden), 0);
                encoder.set_buffer(3, Some(buffers.router_logits), 0);
                encoder.stage_set_u32(4, EXPERTS);
                encoder.stage_set_u32(5, HIDDEN);
                encoder.stage_set_u32(6, GROUP);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_postnorm_router_top10_select",
            (1, 1, 1),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(buffers.router_logits), 0);
                encoder.set_buffer(1, Some(buffers.router_probabilities), 0);
                encoder.set_buffer(2, Some(buffers.router_route_ids), 0);
                encoder.set_buffer(3, Some(buffers.router_route_weights), 0);
                encoder.stage_set_u32(4, EXPERTS);
                encoder.stage_set_u32(5, ROUTES);
                encoder.stage_set_f32(6, 0.0);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_all_ten_routed_wave_route_guard",
            (1, 1, 1),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(buffers.router_route_ids), 0);
                encoder.set_buffer(1, Some(buffers.router_route_weights), 0);
                encoder.set_buffer(2, Some(buffers.expected_route_ids), 0);
                encoder.set_buffer(3, Some(buffers.expected_route_weights), 0);
                encoder.set_buffer(4, Some(buffers.route_guard), 0);
                encoder.stage_set_u32(5, ROUTES);
                // This matches the bounded source/device probability parity
                // gate. Expert order itself remains exact u32 equality.
                encoder.stage_set_f32(6, 2.0e-5);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_all_ten_routed_wave_gate_up",
            (256, INTERMEDIATE, ROUTES),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(buffers.route_gate_signs), 0);
                encoder.set_buffer(1, Some(buffers.route_gate_scales), 0);
                encoder.set_buffer(2, Some(buffers.route_up_signs), 0);
                encoder.set_buffer(3, Some(buffers.route_up_scales), 0);
                encoder.set_buffer(4, Some(buffers.postnorm_hidden), 0);
                encoder.set_buffer(5, Some(buffers.route_gate), 0);
                encoder.set_buffer(6, Some(buffers.route_up), 0);
                encoder.stage_set_u32(7, ROUTES);
                encoder.stage_set_u32(8, INTERMEDIATE);
                encoder.stage_set_u32(9, HIDDEN);
                encoder.stage_set_u32(10, GROUP);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_all_ten_routed_wave_swiglu",
            (INTERMEDIATE, ROUTES, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(buffers.route_gate), 0);
                encoder.set_buffer(1, Some(buffers.route_up), 0);
                encoder.set_buffer(2, Some(buffers.route_activated), 0);
                encoder.stage_set_u32(3, ROUTES);
                encoder.stage_set_u32(4, INTERMEDIATE);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_all_ten_routed_wave_down_weighted",
            (256, HIDDEN, ROUTES),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(buffers.route_down_signs), 0);
                encoder.set_buffer(1, Some(buffers.route_down_scales), 0);
                encoder.set_buffer(2, Some(buffers.route_activated), 0);
                encoder.set_buffer(3, Some(buffers.router_route_weights), 0);
                encoder.set_buffer(4, Some(buffers.route_weighted), 0);
                encoder.stage_set_u32(5, ROUTES);
                encoder.stage_set_u32(6, HIDDEN);
                encoder.stage_set_u32(7, INTERMEDIATE);
                encoder.stage_set_u32(8, GROUP);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_shared_expert_wave_gate_up",
            (256, INTERMEDIATE, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(buffers.shared_gate_signs), 0);
                encoder.set_buffer(1, Some(buffers.shared_gate_scales), 0);
                encoder.set_buffer(2, Some(buffers.shared_up_signs), 0);
                encoder.set_buffer(3, Some(buffers.shared_up_scales), 0);
                encoder.set_buffer(4, Some(buffers.postnorm_hidden), 0);
                encoder.set_buffer(5, Some(buffers.shared_gate), 0);
                encoder.set_buffer(6, Some(buffers.shared_up), 0);
                encoder.stage_set_u32(7, INTERMEDIATE);
                encoder.stage_set_u32(8, HIDDEN);
                encoder.stage_set_u32(9, GROUP);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_shared_expert_wave_swiglu",
            (INTERMEDIATE, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(buffers.shared_gate), 0);
                encoder.set_buffer(1, Some(buffers.shared_up), 0);
                encoder.set_buffer(2, Some(buffers.shared_activated), 0);
                encoder.stage_set_u32(3, INTERMEDIATE);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_shared_expert_wave_down",
            (256, HIDDEN, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(buffers.shared_down_signs), 0);
                encoder.set_buffer(1, Some(buffers.shared_down_scales), 0);
                encoder.set_buffer(2, Some(buffers.shared_activated), 0);
                encoder.set_buffer(3, Some(buffers.shared_output), 0);
                encoder.stage_set_u32(4, HIDDEN);
                encoder.stage_set_u32(5, INTERMEDIATE);
                encoder.stage_set_u32(6, GROUP);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_shared_expert_wave_scalar_gate",
            (256, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(buffers.shared_scalar_signs), 0);
                encoder.set_buffer(1, Some(buffers.shared_scalar_scales), 0);
                encoder.set_buffer(2, Some(buffers.postnorm_hidden), 0);
                encoder.set_buffer(3, Some(buffers.shared_scalar_logit), 0);
                encoder.stage_set_u32(4, HIDDEN);
                encoder.stage_set_u32(5, GROUP);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_shared_expert_wave_apply_sigmoid_gate",
            (HIDDEN, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(buffers.shared_output), 0);
                encoder.set_buffer(1, Some(buffers.shared_scalar_logit), 0);
                encoder.set_buffer(2, Some(buffers.gated_shared), 0);
                encoder.stage_set_u32(3, HIDDEN);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_moe_wave_aggregate_second_residual_route_sum",
            (HIDDEN, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(buffers.route_weighted), 0);
                encoder.set_buffer(1, Some(buffers.routed_sum), 0);
                encoder.stage_set_u32(2, ROUTES);
                encoder.stage_set_u32(3, HIDDEN);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
            (HIDDEN, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(buffers.routed_sum), 0);
                encoder.set_buffer(1, Some(buffers.gated_shared), 0);
                encoder.set_buffer(2, Some(buffers.first_residual), 0);
                encoder.set_buffer(3, Some(buffers.second_residual), 0);
                encoder.stage_set_u32(4, HIDDEN);
            },
        )
        .map_err(|error| error.to_string())?;
    let encoded = command.dispatch_count().saturating_sub(before);
    if encoded != 14 {
        return Err(format!(
            "true-MoE encoder emitted {encoded} dispatches, expected 14"
        ));
    }
    Ok(encoded)
}

fn printable_plan() -> serde_json::Value {
    let plan = staged_dispatch_plan();
    json!({
        "schema": "hawking.ascension.qwen80_all_ten_true_moe_graph_device_plan.v1",
        "status": "STAGED_CPU_BUILD_ONLY_QWEN80_ALL_TEN_TRUE_MOE_DEVICE_GRAPH_NOT_LEASED_OR_EXECUTED",
        "layer": 0,
        "exact_geometry": {
            "hidden": HIDDEN,
            "experts": EXPERTS,
            "selected_routes": ROUTES,
            "intermediate": INTERMEDIATE,
            "group_size": GROUP,
            "per_route_sign_bytes": ROUTE_SIGN_BYTES,
            "per_route_scale_bytes": ROUTE_SCALE_BYTES,
        },
        "dispatches": plan.iter().map(|stage| json!({
            "kernel": stage.kernel,
            "grid": [stage.grid.0, stage.grid.1, stage.grid.2],
            "threadgroup": [stage.threadgroup.0, stage.threadgroup.1, stage.threadgroup.2],
            "purpose": stage.purpose,
        })).collect::<Vec<_>>(),
        "future_integration_requirements": [
            "append-only registration of qwen80_all_ten_routed_expert_wave.metal in metal/mod.rs and all_shader_sources",
            "narrow native-runtime bridge that keeps one admitted catalog and the DeltaNet first_residual PinnedBuffer in the same command graph",
            "concatenate exact plan-selected route 0..9 HQ30G1B1 sign/scale sections in source order only after strict admission",
            "outer-reaped strict-Math non-timed component lease with source-bound CPU/device parity for every route/shared/second-residual buffer",
            "read back and require route_guard=1; reject any router ID/weight divergence from the retained all-ten plan before a receipt is sealed",
        ],
        "claim_boundary": {
            "device_context_or_dispatch_performed": false,
            "no_artifact_scan_or_payload_open": true,
            "no_watcher_server_hcli_tps_or_token_claim": true,
            "historical_materialized_combine_fixture_remains_refused": true,
        },
    })
}

/// The outer reaper supplies every immutable authority explicitly.  This
/// child performs no implicit discovery: a future capture is bound to the
/// exact current admission, router pair, source route plan, prefix witness,
/// typed bridge, fixed ABI plan, and one fresh component-only lease.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct TrueMoeChildArgs {
    manifest: PathBuf,
    admission_current: PathBuf,
    router_receipt: PathBuf,
    router_outer_receipt: PathBuf,
    route_plan: PathBuf,
    first_residual_receipt: PathBuf,
    typed_bridge_receipt: PathBuf,
    fixed_abi_contract: PathBuf,
    lease_receipt: PathBuf,
    outer_capture_dir: PathBuf,
    capture_dir: PathBuf,
    workers: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct TrueMoeChildAuthority {
    args: TrueMoeChildArgs,
    manifest_document_sha256: String,
    manifest_seal_sha256: String,
    admission_pointer_seal_sha256: String,
    admission_receipt_seal_sha256: String,
    source_audit_seal_sha256: String,
    source_revision: String,
    router_receipt_sha256: String,
    router_outer_receipt_sha256: String,
    router_outer_receipt_seal_sha256: String,
    route_plan_sha256: String,
    first_residual_receipt_sha256: String,
    first_residual_receipt_seal_sha256: String,
    first_residual_output_sha256: String,
    typed_bridge_sha256: String,
    typed_bridge_seal_sha256: String,
    fixed_abi_sha256: String,
    lease_sha256: String,
    lease_seal_sha256: String,
}

#[cfg(target_os = "macos")]
fn true_moe_usage() -> &'static str {
    "usage: ascension_qwen80_all_ten_true_moe_graph_device \\\n+--manifest ABSOLUTE_PATH --admission-current ABSOLUTE_PATH \\\n+--router-receipt ABSOLUTE_PATH --router-outer-receipt ABSOLUTE_PATH \\\n+--route-plan ABSOLUTE_PATH --first-residual-receipt ABSOLUTE_PATH \\\n+--typed-bridge-receipt ABSOLUTE_PATH --fixed-abi-contract ABSOLUTE_PATH \\\n+--lease-receipt ABSOLUTE_PATH --outer-capture-dir ABSOLUTE_DIRECTORY \\\n+--capture-dir NEW_ABSOLUTE_DIRECTORY --mode metal --workers 1..4"
}

#[cfg(target_os = "macos")]
fn parse_true_moe_child_args<I>(arguments: I) -> Result<TrueMoeChildArgs, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = BTreeMap::<String, String>::new();
    let mut arguments = arguments.into_iter();
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--manifest"
            | "--admission-current"
            | "--router-receipt"
            | "--router-outer-receipt"
            | "--route-plan"
            | "--first-residual-receipt"
            | "--typed-bridge-receipt"
            | "--fixed-abi-contract"
            | "--lease-receipt"
            | "--outer-capture-dir"
            | "--capture-dir"
            | "--mode"
            | "--workers" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| format!("{flag} requires a value; {}", true_moe_usage()))?;
                if values.insert(flag.clone(), value).is_some() {
                    return Err(format!("{flag} repeated; {}", true_moe_usage()));
                }
            }
            "--help" | "-h" => return Err(true_moe_usage().to_owned()),
            _ => {
                return Err(format!(
                    "unsupported argument {flag:?}; {}",
                    true_moe_usage()
                ))
            }
        }
    }
    let mut required = |flag: &str| -> Result<PathBuf, String> {
        let value = values
            .remove(flag)
            .ok_or_else(|| format!("missing {flag}; {}", true_moe_usage()))?;
        let path = PathBuf::from(value);
        if !path.is_absolute() {
            return Err(format!("{flag} must be absolute"));
        }
        Ok(path)
    };
    let args = TrueMoeChildArgs {
        manifest: required("--manifest")?,
        admission_current: required("--admission-current")?,
        router_receipt: required("--router-receipt")?,
        router_outer_receipt: required("--router-outer-receipt")?,
        route_plan: required("--route-plan")?,
        first_residual_receipt: required("--first-residual-receipt")?,
        typed_bridge_receipt: required("--typed-bridge-receipt")?,
        fixed_abi_contract: required("--fixed-abi-contract")?,
        lease_receipt: required("--lease-receipt")?,
        outer_capture_dir: required("--outer-capture-dir")?,
        capture_dir: required("--capture-dir")?,
        workers: values
            .remove("--workers")
            .ok_or_else(|| format!("missing --workers; {}", true_moe_usage()))?
            .parse::<usize>()
            .map_err(|_| "--workers must be an unsigned integer".to_owned())?,
    };
    let mode = values
        .remove("--mode")
        .ok_or_else(|| format!("missing --mode; {}", true_moe_usage()))?;
    if mode != STRICT_METAL_MODE {
        return Err(format!("--mode must be {STRICT_METAL_MODE:?}"));
    }
    if !(1..=4).contains(&args.workers) {
        return Err("--workers must be 1..4".to_owned());
    }
    if !values.is_empty() {
        return Err(format!("unconsumed arguments: {values:?}"));
    }
    Ok(args)
}

#[cfg(target_os = "macos")]
fn canonical_regular(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    fs::canonicalize(path).map_err(|error| format!("cannot canonicalize {label}: {error}"))
}

#[cfg(target_os = "macos")]
fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(target_os = "macos")]
fn read_json_authority(path: &Path, label: &str) -> Result<(PathBuf, Vec<u8>, Value), String> {
    let path = canonical_regular(path, label)?;
    let raw = fs::read(&path).map_err(|error| format!("cannot read {label}: {error}"))?;
    let document: Value =
        serde_json::from_slice(&raw).map_err(|error| format!("cannot parse {label}: {error}"))?;
    if !document.is_object() {
        return Err(format!("{label} must be a JSON object"));
    }
    Ok((path, raw, document))
}

#[cfg(target_os = "macos")]
fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

#[cfg(target_os = "macos")]
fn field_object<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    object
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label}.{field} must be an object"))
}

#[cfg(target_os = "macos")]
fn field_string(object: &Map<String, Value>, field: &str, label: &str) -> Result<String, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("{label}.{field} must be a string"))
}

#[cfg(target_os = "macos")]
fn require_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value.bytes().all(|byte| {
            byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
        })
    {
        return Err(format!("{label} must be a lowercase SHA-256"));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn evidence_sha(object: &Map<String, Value>, field: &str, label: &str) -> Result<String, String> {
    let evidence = field_object(object, field, label)?;
    if evidence.get("present") != Some(&Value::Bool(true)) {
        return Err(format!("{label}.{field} must attest a present file"));
    }
    let sha = field_string(evidence, "sha256", &format!("{label}.{field}"))?;
    require_sha256(&sha, &format!("{label}.{field}.sha256"))?;
    Ok(sha)
}

#[cfg(target_os = "macos")]
fn validate_child_paths(args: &TrueMoeChildArgs) -> Result<TrueMoeChildArgs, String> {
    let outer_metadata = fs::symlink_metadata(&args.outer_capture_dir).map_err(|error| {
        format!(
            "cannot stat --outer-capture-dir {}: {error}",
            args.outer_capture_dir.display()
        )
    })?;
    if outer_metadata.file_type().is_symlink() || !outer_metadata.is_dir() {
        return Err("--outer-capture-dir must be an existing non-symlink directory".to_owned());
    }
    let outer_capture_dir = fs::canonicalize(&args.outer_capture_dir).map_err(|error| {
        format!(
            "cannot canonicalize --outer-capture-dir {}: {error}",
            args.outer_capture_dir.display()
        )
    })?;
    if args.capture_dir.exists() {
        return Err("--capture-dir must not exist before this one-shot child starts".to_owned());
    }
    let capture_parent = args
        .capture_dir
        .parent()
        .ok_or("--capture-dir has no parent")?;
    if fs::canonicalize(capture_parent)
        .map_err(|error| format!("cannot canonicalize --capture-dir parent: {error}"))?
        != outer_capture_dir
    {
        return Err("--capture-dir must be a direct child of --outer-capture-dir".to_owned());
    }
    Ok(TrueMoeChildArgs {
        manifest: canonical_regular(&args.manifest, "--manifest")?,
        admission_current: canonical_regular(&args.admission_current, "--admission-current")?,
        router_receipt: canonical_regular(&args.router_receipt, "--router-receipt")?,
        router_outer_receipt: canonical_regular(
            &args.router_outer_receipt,
            "--router-outer-receipt",
        )?,
        route_plan: canonical_regular(&args.route_plan, "--route-plan")?,
        first_residual_receipt: canonical_regular(
            &args.first_residual_receipt,
            "--first-residual-receipt",
        )?,
        typed_bridge_receipt: canonical_regular(
            &args.typed_bridge_receipt,
            "--typed-bridge-receipt",
        )?,
        fixed_abi_contract: canonical_regular(&args.fixed_abi_contract, "--fixed-abi-contract")?,
        lease_receipt: canonical_regular(&args.lease_receipt, "--lease-receipt")?,
        outer_capture_dir,
        capture_dir: args.capture_dir.clone(),
        workers: args.workers,
    })
}

#[cfg(target_os = "macos")]
fn validate_true_moe_child_authority(
    args: &TrueMoeChildArgs,
) -> Result<TrueMoeChildAuthority, String> {
    let args = validate_child_paths(args)?;
    let (_, manifest_raw, manifest) = read_json_authority(&args.manifest, "manifest")?;
    let manifest_object = object(&manifest, "manifest")?;
    if field_string(manifest_object, "schema", "manifest")? != MANIFEST_SCHEMA {
        return Err("manifest schema drifted".to_owned());
    }
    let manifest_document_sha256 = sha256_hex(&manifest_raw);
    let manifest_seal_sha256 = field_string(manifest_object, "seal_sha256", "manifest")?;
    require_sha256(&manifest_seal_sha256, "manifest seal")?;

    let (_, admission_raw, admission) =
        read_json_authority(&args.admission_current, "admission current")?;
    let admission_object = object(&admission, "admission current")?;
    if field_string(admission_object, "schema", "admission current")? != ADMISSION_POINTER_SCHEMA
        || field_string(admission_object, "status", "admission current")?
            != ADMISSION_POINTER_STATUS
    {
        return Err("admission-current schema/status drifted".to_owned());
    }
    let admission_pointer_seal_sha256 =
        field_string(admission_object, "seal_sha256", "admission current")?;
    require_sha256(&admission_pointer_seal_sha256, "admission-current seal")?;
    let selected_manifest =
        field_object(admission_object, "complete_manifest", "admission current")?;
    if field_string(
        selected_manifest,
        "document_sha256",
        "admission current manifest",
    )? != manifest_document_sha256
        || field_string(
            selected_manifest,
            "seal_sha256",
            "admission current manifest",
        )? != manifest_seal_sha256
    {
        return Err("admission-current manifest identity drifted".to_owned());
    }
    let selected_receipt =
        field_object(admission_object, "admission_receipt", "admission current")?;
    let receipt_path = canonical_regular(
        Path::new(&field_string(
            selected_receipt,
            "path",
            "admission receipt",
        )?),
        "admission receipt",
    )?;
    let (_, _receipt_raw, receipt) = read_json_authority(&receipt_path, "admission receipt")?;
    let receipt_object = object(&receipt, "admission receipt")?;
    if field_string(receipt_object, "schema", "admission receipt")? != ADMISSION_RECEIPT_SCHEMA
        || field_string(receipt_object, "status", "admission receipt")? != ADMISSION_RECEIPT_STATUS
    {
        return Err("immutable admission receipt schema/status drifted".to_owned());
    }
    let admission_receipt_seal_sha256 =
        field_string(receipt_object, "seal_sha256", "admission receipt")?;
    if field_string(selected_receipt, "seal_sha256", "admission current receipt")?
        != admission_receipt_seal_sha256
    {
        return Err("admission-current selected receipt seal drifted".to_owned());
    }
    require_sha256(&admission_receipt_seal_sha256, "admission receipt seal")?;
    let revalidation = field_object(
        receipt_object,
        "current_source_revalidation",
        "admission receipt",
    )?;
    let source_audit_seal_sha256 = field_string(
        revalidation,
        "source_audit_seal_sha256",
        "source revalidation",
    )?;
    let source_revision = field_string(revalidation, "revision", "source revalidation")?;
    require_sha256(&source_audit_seal_sha256, "source audit seal")?;

    let (_, router_raw, _router) = read_json_authority(&args.router_receipt, "router receipt")?;
    let (_, router_outer_raw, router_outer) =
        read_json_authority(&args.router_outer_receipt, "router outer receipt")?;
    let router_outer_object = object(&router_outer, "router outer receipt")?;
    let router_outer_receipt_seal_sha256 =
        field_string(router_outer_object, "seal_sha256", "router outer receipt")?;
    require_sha256(&router_outer_receipt_seal_sha256, "router outer seal")?;
    let (_, route_plan_raw, _route_plan) = read_json_authority(&args.route_plan, "route plan")?;
    let (_, first_raw, first) =
        read_json_authority(&args.first_residual_receipt, "first-residual outer receipt")?;
    let first_object = object(&first, "first-residual outer receipt")?;
    let first_residual_receipt_seal_sha256 =
        field_string(first_object, "seal_sha256", "first-residual outer receipt")?;
    let first_output = field_object(
        first_object,
        "first_residual_output",
        "first-residual outer receipt",
    )?;
    let first_residual_output_sha256 =
        field_string(first_output, "sha256", "first-residual output")?;
    require_sha256(&first_residual_output_sha256, "first-residual output SHA")?;

    let (_, typed_raw, typed) = read_json_authority(&args.typed_bridge_receipt, "typed bridge")?;
    let typed_object = object(&typed, "typed bridge")?;
    if field_string(typed_object, "schema", "typed bridge")? != TYPED_BRIDGE_SCHEMA
        || field_string(typed_object, "status", "typed bridge")? != TYPED_BRIDGE_STATUS
    {
        return Err("typed bridge schema/status drifted".to_owned());
    }
    let typed_bridge_seal_sha256 = field_string(typed_object, "seal_sha256", "typed bridge")?;
    let typed_source = field_object(typed_object, "source_binding", "typed bridge")?;
    if evidence_sha(typed_source, "manifest", "typed bridge")? != manifest_document_sha256
        || evidence_sha(typed_source, "route_plan", "typed bridge")? != sha256_hex(&route_plan_raw)
        || evidence_sha(typed_source, "first_residual_receipt", "typed bridge")?
            != sha256_hex(&first_raw)
        || field_string(typed_source, "manifest_seal_sha256", "typed bridge")?
            != manifest_seal_sha256
        || field_string(
            typed_source,
            "admission_receipt_seal_sha256",
            "typed bridge",
        )? != admission_receipt_seal_sha256
    {
        return Err("typed bridge immutable source authority drifted".to_owned());
    }
    let typed_payload = field_object(typed_object, "typed_bridge", "typed bridge")?;
    if field_string(
        typed_payload,
        "first_residual_output_sha256",
        "typed bridge",
    )? != first_residual_output_sha256
        || field_string(
            typed_payload,
            "first_residual_receipt_seal_sha256",
            "typed bridge",
        )? != first_residual_receipt_seal_sha256
    {
        return Err("typed bridge first-residual authority drifted".to_owned());
    }

    let (_, fixed_raw, fixed) = read_json_authority(&args.fixed_abi_contract, "fixed ABI")?;
    let fixed_object = object(&fixed, "fixed ABI")?;
    if field_string(fixed_object, "schema", "fixed ABI")? != FIXED_ABI_SCHEMA
        || field_string(fixed_object, "status", "fixed ABI")? != FIXED_ABI_STATUS
        || fixed_object.get("seal_sha256").is_some()
    {
        return Err("fixed ABI must be the exact unsealed static plan".to_owned());
    }

    let (_, lease_raw, lease) = read_json_authority(&args.lease_receipt, "quiet component lease")?;
    let lease_object = object(&lease, "quiet component lease")?;
    if field_string(lease_object, "schema", "quiet component lease")? != LEASE_SCHEMA
        || field_string(lease_object, "status", "quiet component lease")? != LEASE_STATUS
    {
        return Err("quiet component lease schema/status drifted".to_owned());
    }
    let lease_seal_sha256 = field_string(lease_object, "seal_sha256", "quiet component lease")?;
    let lease_artifact = field_object(lease_object, "artifact_binding", "quiet component lease")?;
    if field_string(
        lease_artifact,
        "manifest_document_sha256",
        "quiet component lease",
    )? != manifest_document_sha256
        || field_string(
            lease_artifact,
            "manifest_seal_sha256",
            "quiet component lease",
        )? != manifest_seal_sha256
        || field_string(
            lease_artifact,
            "admission_receipt_seal_sha256",
            "quiet component lease",
        )? != admission_receipt_seal_sha256
    {
        return Err("quiet component lease artifact authority drifted".to_owned());
    }
    let lease_bridge = field_object(
        lease_object,
        "typed_bridge_binding",
        "quiet component lease",
    )?;
    if field_string(
        lease_bridge,
        "document_sha256",
        "quiet component lease typed bridge",
    )? != sha256_hex(&typed_raw)
        || field_string(
            lease_bridge,
            "seal_sha256",
            "quiet component lease typed bridge",
        )? != typed_bridge_seal_sha256
    {
        return Err("quiet component lease typed bridge identity drifted".to_owned());
    }
    let lease_fixed = field_object(
        lease_object,
        "fixed_abi_contract_binding",
        "quiet component lease",
    )?;
    if field_string(
        lease_fixed,
        "document_sha256",
        "quiet component lease fixed ABI",
    )? != sha256_hex(&fixed_raw)
    {
        return Err("quiet component lease fixed ABI identity drifted".to_owned());
    }

    let _ = admission_raw;
    Ok(TrueMoeChildAuthority {
        args,
        manifest_document_sha256,
        manifest_seal_sha256,
        admission_pointer_seal_sha256,
        admission_receipt_seal_sha256,
        source_audit_seal_sha256,
        source_revision,
        router_receipt_sha256: sha256_hex(&router_raw),
        router_outer_receipt_sha256: sha256_hex(&router_outer_raw),
        router_outer_receipt_seal_sha256,
        route_plan_sha256: sha256_hex(&route_plan_raw),
        first_residual_receipt_sha256: sha256_hex(&first_raw),
        first_residual_receipt_seal_sha256,
        first_residual_output_sha256,
        typed_bridge_sha256: sha256_hex(&typed_raw),
        typed_bridge_seal_sha256,
        fixed_abi_sha256: sha256_hex(&fixed_raw),
        lease_sha256: sha256_hex(&lease_raw),
        lease_seal_sha256,
    })
}

#[cfg(target_os = "macos")]
fn write_new_atomic(capture_dir: &Path, name: &str, contents: &[u8]) -> Result<(), String> {
    let target = capture_dir.join(name);
    if target.exists() {
        return Err(format!(
            "refusing to overwrite capture artifact {}",
            target.display()
        ));
    }
    let temporary = capture_dir.join(format!(".{name}.{}.tmp", process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("cannot create {}: {error}", temporary.display()))?;
    if let Err(error) = file.write_all(contents).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(format!("cannot write {}: {error}", temporary.display()));
    }
    drop(file);
    fs::hard_link(&temporary, &target).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        format!("cannot publish {}: {error}", target.display())
    })?;
    fs::remove_file(&temporary)
        .map_err(|error| format!("cannot retire {}: {error}", temporary.display()))
}

#[cfg(target_os = "macos")]
fn now_unix_millis() -> Result<u128, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock before Unix epoch: {error}"))
        .map(|duration| duration.as_millis())
}

#[cfg(target_os = "macos")]
fn begin_capture(authority: &TrueMoeChildAuthority) -> Result<(), String> {
    fs::create_dir(&authority.args.capture_dir).map_err(|error| {
        format!(
            "refusing non-exclusive --capture-dir {}: {error}",
            authority.args.capture_dir.display()
        )
    })?;
    let invocation = json!({
        "schema": CHILD_SCHEMA,
        "status": "STARTED_QWEN80_SOURCE_INPUT_L0_TRUE_MOE_STRICT_MATH_COMPONENT",
        "started_unix_millis": now_unix_millis()?,
        "mode": STRICT_METAL_MODE,
        "manifest": authority.args.manifest,
        "admission_current": authority.args.admission_current,
        "typed_bridge_receipt": authority.args.typed_bridge_receipt,
        "fixed_abi_contract": authority.args.fixed_abi_contract,
        "lease_receipt": authority.args.lease_receipt,
        "workers": authority.args.workers,
        "execution_policy": {
            "strict_math": true,
            "timing_or_benchmarking_allowed": false,
            "complete_layer_or_token_allowed": false,
            "tps_or_tg_claim_allowed": false,
            "source_token_id": SOURCE_TOKEN_ID,
            "prefix_dispatches_expected": PREFIX_DISPATCHES,
            "suffix_dispatches_expected": SUFFIX_DISPATCHES,
            "total_dispatches_expected": FULL_COMPONENT_DISPATCHES,
        },
    });
    write_new_atomic(
        &authority.args.capture_dir,
        "invocation.json",
        &serde_json::to_vec_pretty(&invocation).map_err(|error| error.to_string())?,
    )
}

#[cfg(target_os = "macos")]
fn snapshot_f32(buffer: &PinnedBuffer, elements: usize, label: &str) -> Result<Vec<f32>, String> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} byte count overflows"))?;
    if buffer.length() < bytes as u64 {
        return Err(format!(
            "{label} needs {bytes} bytes but device buffer has {}",
            buffer.length()
        ));
    }
    Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, elements).to_vec() })
}

#[cfg(target_os = "macos")]
fn snapshot_u32(buffer: &PinnedBuffer, elements: usize, label: &str) -> Result<Vec<u32>, String> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| format!("{label} byte count overflows"))?;
    if buffer.length() < bytes as u64 {
        return Err(format!(
            "{label} needs {bytes} bytes but device buffer has {}",
            buffer.length()
        ));
    }
    Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const u32, elements).to_vec() })
}

#[cfg(target_os = "macos")]
fn f32_sha256(values: &[f32], label: &str) -> Result<String, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} is empty or non-finite"));
    }
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(target_os = "macos")]
fn require_parity(
    expected: &[f32],
    observed: &[f32],
    label: &str,
    tolerance: f32,
) -> Result<f32, String> {
    if expected.len() != observed.len() {
        return Err(format!(
            "{label} length mismatch: expected {}, observed {}",
            expected.len(),
            observed.len()
        ));
    }
    let mut maximum = 0.0f32;
    for (index, (&expected, &observed)) in expected.iter().zip(observed).enumerate() {
        if !expected.is_finite() || !observed.is_finite() {
            return Err(format!("{label} is non-finite at {index}"));
        }
        maximum = maximum.max((expected - observed).abs());
    }
    if maximum > tolerance {
        return Err(format!(
            "{label} strict-Metal parity failed: max_abs_error={maximum}, tolerance={tolerance}"
        ));
    }
    Ok(maximum)
}

#[cfg(target_os = "macos")]
fn fixed_graph_buffers(
    fixed: &Qwen80L0TrueMoeFixedDeviceBuffers,
) -> Qwen80AllTenTrueMoeGraphFixedBuffers<'_> {
    Qwen80AllTenTrueMoeGraphFixedBuffers {
        postnorm_signs: &fixed.postnorm.signs,
        postnorm_scales: &fixed.postnorm.scales,
        postnorm_hidden: &fixed.postnorm_hidden,
        router_signs: &fixed.router.signs,
        router_scales: &fixed.router.scales,
        router_logits: &fixed.router_logits,
        router_probabilities: &fixed.router_probabilities,
        router_route_ids: &fixed.router_route_ids,
        router_route_weights: &fixed.router_route_weights,
        route_guard: &fixed.route_guard,
        route_gate: &fixed.route_gate,
        route_up: &fixed.route_up,
        route_activated: &fixed.route_activated,
        route_weighted: &fixed.route_weighted,
        shared_gate_signs: &fixed.shared_gate_proj.signs,
        shared_gate_scales: &fixed.shared_gate_proj.scales,
        shared_up_signs: &fixed.shared_up_proj.signs,
        shared_up_scales: &fixed.shared_up_proj.scales,
        shared_down_signs: &fixed.shared_down_proj.signs,
        shared_down_scales: &fixed.shared_down_proj.scales,
        shared_scalar_signs: &fixed.shared_expert_gate.signs,
        shared_scalar_scales: &fixed.shared_expert_gate.scales,
        shared_gate: &fixed.shared_gate,
        shared_up: &fixed.shared_up,
        shared_activated: &fixed.shared_activated,
        shared_output: &fixed.shared_output,
        shared_scalar_logit: &fixed.shared_scalar_logit,
        gated_shared: &fixed.gated_shared,
        routed_sum: &fixed.routed_sum,
        second_residual: &fixed.second_residual,
    }
}

#[cfg(target_os = "macos")]
fn run_true_moe_component(authority: &TrueMoeChildAuthority) -> Result<Value, String> {
    let (_, _route_raw, route_descriptor) =
        read_json_authority(&authority.args.route_plan, "route plan")?;
    let admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        expected_manifest_seal_sha256: authority.manifest_seal_sha256.clone(),
        expected_source_audit_seal_sha256: authority.source_audit_seal_sha256.clone(),
        expected_source_revision: authority.source_revision.clone(),
    };
    // This is the sole full compact-artifact admission scan in this child.
    // Everything after it uses the retained catalog snapshots only.
    let catalog = Qwen80CompleteArtifactCatalog::load(&authority.args.manifest, &admission)
        .map_err(|error| format!("strict Qwen80 artifact admission failed: {error}"))?;
    let hybrid_plan = catalog
        .complete_hybrid_decoder_plan(1)
        .map_err(|error| format!("hybrid schedule binding failed: {error}"))?;
    let route_authority = Qwen80AllTenRoutedExpertPlanAuthority {
        manifest_document_sha256: &authority.manifest_document_sha256,
        plan_document_sha256: &authority.route_plan_sha256,
        router_receipt_sha256: &authority.router_receipt_sha256,
        router_outer_receipt_sha256: &authority.router_outer_receipt_sha256,
        router_outer_receipt_seal_sha256: &authority.router_outer_receipt_seal_sha256,
    };
    let route_plan = hybrid_plan
        .bind_all_ten_routed_expert_plan(0, &route_authority, &route_descriptor)
        .map_err(|error| format!("source-selected all-ten route plan refused: {error}"))?;
    let source_bridge = catalog
        .build_all_ten_true_moe_source_bridge(
            &route_plan,
            catalog
                .first_residual_device_binding(0)
                .map_err(|error| format!("first-residual binding refused: {error}"))?,
        )
        .map_err(|error| format!("all-ten source bridge refused: {error}"))?;
    let embedding = catalog
        .execute_embedding_lookup_cpu_oracle(SOURCE_TOKEN_ID)
        .map_err(|error| {
            format!("source-token direct-packed embedding reference refused: {error}")
        })?;
    let cpu_input = Qwen80CanonicalLinearLayerCpuInput::with_zero_state(embedding.hidden);
    let cpu = catalog
        .execute_first_linear_layer_cpu_moe_oracle(&cpu_input)
        .map_err(|error| format!("source-token L0 true-MoE CPU reference refused: {error}"))?;
    if source_bridge.route_payloads().route().ids != cpu.route.ids
        || source_bridge.route_payloads().route().weights != cpu.route.weights
    {
        return Err(
            "all-ten plan route differs from same-input direct-packed CPU router".to_owned(),
        );
    }
    let cpu_first_residual_sha =
        f32_sha256(&cpu.mixer.mixer_residual_output, "CPU first residual")?;
    let runtime = Qwen80CompleteNativeRuntime::from_admitted_catalog_strict_math(
        catalog,
        Qwen80CompleteRuntimeOptions {
            max_seq_len: 1,
            trace_dispatch: false,
        },
    )
    .map_err(|error| format!("strict-Math Qwen80 native runtime construction failed: {error}"))?;
    let mut command = runtime.begin_component_token_command_buffer();
    let prefix = runtime
        .encode_source_token_first_linear_deltanet_into(&mut command, SOURCE_TOKEN_ID)
        .map_err(|error| format!("source-token L0 DeltaNet prefix encode failed: {error}"))?;
    let prefix_dispatches = command.dispatch_count();
    if prefix_dispatches != PREFIX_DISPATCHES {
        return Err(format!(
            "L0 prefix encoded {prefix_dispatches} dispatches, expected {PREFIX_DISPATCHES}"
        ));
    }
    let fixed = runtime
        .upload_l0_true_moe_fixed_device_buffers()
        .map_err(|error| format!("fixed true-MoE device buffers refused: {error}"))?;
    let route_bridge = runtime
        .bind_source_input_first_residual_to_all_ten(&source_bridge, &prefix)
        .map_err(|error| format!("same-TCB all-ten route bridge refused: {error}"))?;
    let buffers = Qwen80AllTenTrueMoeGraphBuffers::from_admitted_route_bridge(
        &route_bridge,
        fixed_graph_buffers(&fixed),
    );
    let suffix_dispatches = encode_all_ten_true_moe_from_first_residual(&mut command, &buffers)?;
    if suffix_dispatches != SUFFIX_DISPATCHES
        || command.dispatch_count() != FULL_COMPONENT_DISPATCHES
    {
        return Err("L0 true-MoE command graph dispatch count drifted".to_owned());
    }
    command
        .commit_and_wait()
        .map_err(|error| format!("L0 true-MoE common command-buffer fence failed: {error}"))?;
    let prefix_parity = prefix.verify_after_fence(&runtime).map_err(|error| {
        format!("source-token L0 prefix parity failed after common fence: {error}")
    })?;
    if prefix_parity.cpu_first_residual_f32le_sha256 != cpu_first_residual_sha
        || prefix_parity.device_first_residual_f32le_sha256
            != authority.first_residual_output_sha256
        || prefix_parity.dispatches_encoded_before_suffix != PREFIX_DISPATCHES
        || !prefix_parity.same_command_graph_required
    {
        return Err(
            "same-input first-residual lineage does not match sealed antecedent".to_owned(),
        );
    }

    let postnorm = snapshot_f32(&fixed.postnorm_hidden, HIDDEN as usize, "postnorm hidden")?;
    let logits = snapshot_f32(&fixed.router_logits, EXPERTS as usize, "router logits")?;
    let observed_ids = snapshot_u32(&fixed.router_route_ids, ROUTES as usize, "router route IDs")?;
    let observed_weights = snapshot_f32(
        &fixed.router_route_weights,
        ROUTES as usize,
        "router route weights",
    )?;
    let guard = snapshot_u32(&fixed.route_guard, 1, "route guard")?[0];
    let route_weighted = snapshot_f32(
        &fixed.route_weighted,
        (ROUTES * HIDDEN) as usize,
        "weighted all-ten route outputs",
    )?;
    let gated_shared = snapshot_f32(&fixed.gated_shared, HIDDEN as usize, "gated shared output")?;
    let routed_sum = snapshot_f32(&fixed.routed_sum, HIDDEN as usize, "routed sum")?;
    let second_residual = snapshot_f32(&fixed.second_residual, HIDDEN as usize, "second residual")?;

    let postnorm_error = require_parity(
        &cpu.post_attention_rms_norm_output,
        &postnorm,
        "postnorm",
        2.0e-4,
    )?;
    let logits_error = require_parity(&cpu.router_logits, &logits, "router logits", 5.0e-4)?;
    let expected_ids = cpu.route.ids.map(u32::from);
    let expected_weights = cpu.route.weights;
    if observed_ids.as_slice() != expected_ids || guard != 1 {
        return Err(
            "device router top-10 or route guard differs from exact source authority".to_owned(),
        );
    }
    let weights_error = require_parity(
        &expected_weights,
        &observed_weights,
        "router weights",
        2.0e-5,
    )?;
    let mut route_witnesses = Vec::with_capacity(ROUTES as usize);
    for (index, cpu_route) in cpu.routed_experts.iter().enumerate() {
        let start = index * HIDDEN as usize;
        let end = start + HIDDEN as usize;
        let observed = route_weighted
            .get(start..end)
            .ok_or_else(|| format!("route {index} output range drifted"))?;
        let error = require_parity(
            &cpu_route.weighted_output,
            observed,
            &format!("route[{index}] weighted output"),
            3.0e-4,
        )?;
        route_witnesses.push(json!({
            "wave_index": index,
            "expert_id": cpu_route.expert,
            "normalized_weight": cpu_route.route_weight,
            "elements": HIDDEN,
            "cpu_device_parity_passed": true,
            "max_abs_error": error,
            "tolerance": 3.0e-4,
            "output_sha256": f32_sha256(observed, &format!("route[{index}] output"))?,
        }));
    }
    if route_witnesses.len() != ROUTES as usize {
        return Err("all-ten route witness count drifted".to_owned());
    }
    let shared_error = require_parity(
        &cpu.shared_gated_output,
        &gated_shared,
        "gated shared output",
        3.0e-4,
    )?;
    let routed_sum_error =
        require_parity(&cpu.routed_expert_sum, &routed_sum, "routed sum", 3.0e-5)?;
    let second_residual_error = require_parity(
        &cpu.layer_output,
        &second_residual,
        "second residual",
        3.0e-5,
    )?;
    let command_buffer_identity = sha256_hex(
        format!(
            "{}:{}:{}:{}:{}",
            prefix_parity.input_f32le_sha256,
            prefix_parity.device_first_residual_f32le_sha256,
            authority.route_plan_sha256,
            authority.typed_bridge_sha256,
            process::id(),
        )
        .as_bytes(),
    );
    Ok(json!({
        "schema": CHILD_SCHEMA,
        "status": CHILD_STATUS,
        "mode": STRICT_METAL_MODE,
        "metal_device_or_dispatch_performed": true,
        "component_only": true,
        "complete_layer_or_token_performed": false,
        "complete_artifact_scan_performed_once": true,
        "raw_bf16_or_safetensors_opened": false,
        "artifact_binding": {
            "manifest_document_sha256": authority.manifest_document_sha256,
            "manifest_seal_sha256": authority.manifest_seal_sha256,
            "admission_pointer_seal_sha256": authority.admission_pointer_seal_sha256,
            "admission_receipt_seal_sha256": authority.admission_receipt_seal_sha256,
            "source_audit_seal_sha256": authority.source_audit_seal_sha256,
            "source_revision": authority.source_revision,
            "native_device": runtime.device_name(),
            "layer": 0,
            "linear_state_slot": 0,
        },
        "typed_bridge_binding": {
            "receipt_path": authority.args.typed_bridge_receipt,
            "receipt_document_sha256": authority.typed_bridge_sha256,
            "seal_sha256": authority.typed_bridge_seal_sha256,
            "schema": TYPED_BRIDGE_SCHEMA,
            "status": TYPED_BRIDGE_STATUS,
            "first_residual_output_sha256": authority.first_residual_output_sha256,
        },
        "first_residual_antecedent": {
            "receipt_path": authority.args.first_residual_receipt,
            "receipt_document_sha256": authority.first_residual_receipt_sha256,
            "seal_sha256": authority.first_residual_receipt_seal_sha256,
            "output_sha256": authority.first_residual_output_sha256,
        },
        "fixed_abi_contract_binding": {
            "path": authority.args.fixed_abi_contract,
            "document_sha256": authority.fixed_abi_sha256,
            "schema": FIXED_ABI_SCHEMA,
            "status": FIXED_ABI_STATUS,
        },
        "route_plan_binding": {
            "path": authority.args.route_plan,
            "sha256": authority.route_plan_sha256,
        },
        "same_command_graph": {
            "source_token_id": SOURCE_TOKEN_ID,
            "same_command_graph_required": true,
            "same_command_graph_retained": true,
            "command_buffer_identity_sha256": command_buffer_identity,
            "prefix_dispatches": prefix_dispatches,
            "suffix_dispatches": suffix_dispatches,
            "total_dispatches": FULL_COMPONENT_DISPATCHES,
            "first_residual_device_buffer_sha256": prefix_parity.device_first_residual_f32le_sha256,
            "first_residual_matches_sealed_prefix_antecedent": true,
            "command_buffer_fenced_once_after_prefix_and_suffix": true,
        },
        "prefix_parity": prefix_parity,
        "route_guard_readback": {
            "value": guard,
            "passed": true,
            "observed_ids": observed_ids,
            "expected_ids": expected_ids,
            "observed_weights": observed_weights,
            "expected_weights": expected_weights,
            "weights_max_abs_error": weights_error,
            "weights_tolerance": 2.0e-5,
        },
        "readback_parity": {
            "postnorm_cpu_device_parity_passed": true,
            "postnorm_max_abs_error": postnorm_error,
            "router_logits_cpu_device_parity_passed": true,
            "router_logits_max_abs_error": logits_error,
            "all_ten_route_witnesses": route_witnesses.len(),
            "all_ten_route_cpu_device_parity_passed": true,
            "route_witnesses": route_witnesses,
            "shared_expert_cpu_device_parity_passed": true,
            "shared_expert_max_abs_error": shared_error,
            "shared_expert_output_sha256": f32_sha256(&gated_shared, "gated shared output")?,
            "routed_sum_cpu_device_parity_passed": true,
            "routed_sum_max_abs_error": routed_sum_error,
            "routed_sum_output_sha256": f32_sha256(&routed_sum, "routed sum")?,
            "second_residual_cpu_device_parity_passed": true,
            "second_residual_max_abs_error": second_residual_error,
            "second_residual_output_sha256": f32_sha256(&second_residual, "second residual")?,
        },
        "metal_execution_policy": {
            "strict_math_required": true,
            "timing_or_benchmarking_allowed": false,
            "complete_layer_or_token_allowed": false,
            "tps_or_tg_claim_allowed": false,
            "lease_binding": {
                "receipt_path": authority.args.lease_receipt,
                "receipt_document_sha256": authority.lease_sha256,
                "seal_sha256": authority.lease_seal_sha256,
                "schema": LEASE_SCHEMA,
                "status": LEASE_STATUS,
            },
        },
        "durable_capture": {
            "receipt_written_last_is_completion_marker": true,
            "outer_reaped_capture_required": true,
            "replay_guarded": true,
            "capture_directory": authority.args.capture_dir,
        },
        "claim_boundary": {
            "strict_math_source_input_layer0_component_only": true,
            "not_a_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_result": true,
            "no_watcher_or_server_started": true,
        },
    }))
}

#[cfg(target_os = "macos")]
fn finalize_capture(
    authority: &TrueMoeChildAuthority,
    outcome: Result<Value, String>,
) -> Result<(Value, Option<String>), String> {
    let (receipt, failure) = match outcome {
        Ok(receipt) => (receipt, None),
        Err(error) => (
            json!({
                "schema": CHILD_SCHEMA,
                "status": "REFUSED_QWEN80_SOURCE_INPUT_L0_TRUE_MOE_STRICT_MATH_COMPONENT",
                "mode": STRICT_METAL_MODE,
                "metal_device_or_dispatch_performed": false,
                "component_only": true,
                "complete_layer_or_token_performed": false,
                "error": error,
                "claim_boundary": {
                    "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": true,
                },
                "durable_capture": {
                    "receipt_written_last_is_completion_marker": true,
                    "outer_reaped_capture_required": true,
                    "replay_guarded": true,
                },
            }),
            Some(error),
        ),
    };
    let rendered = serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?;
    let mut stdout = rendered.clone();
    stdout.push(b'\n');
    write_new_atomic(&authority.args.capture_dir, "stdout.jsonl", &stdout)?;
    let stderr = failure
        .as_ref()
        .map_or_else(|| b"\n".to_vec(), |error| format!("{error}\n").into_bytes());
    write_new_atomic(&authority.args.capture_dir, "stderr.log", &stderr)?;
    write_new_atomic(&authority.args.capture_dir, "receipt.json", &rendered)?;
    Ok((receipt, failure))
}

fn main() {
    if let Err(error) = validate_staged_plan(&staged_dispatch_plan()) {
        eprintln!("Qwen80 staged all-ten true-MoE graph refused: {error}");
        std::process::exit(2);
    }
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.is_empty() || arguments == ["--print-plan"] {
        println!(
            "{}",
            serde_json::to_string_pretty(&printable_plan()).unwrap()
        );
        return;
    }

    #[cfg(target_os = "macos")]
    {
        let args = match parse_true_moe_child_args(arguments) {
            Ok(args) => args,
            Err(error) => {
                eprintln!("Qwen80 true-input all-ten child argument refusal: {error}");
                process::exit(2);
            }
        };
        let authority = match validate_true_moe_child_authority(&args) {
            Ok(authority) => authority,
            Err(error) => {
                eprintln!("Qwen80 true-input all-ten child authority refusal: {error}");
                process::exit(2);
            }
        };
        if let Err(error) = begin_capture(&authority) {
            eprintln!("Qwen80 true-input all-ten child capture refusal: {error}");
            process::exit(2);
        }
        match finalize_capture(&authority, run_true_moe_component(&authority)) {
            Ok((receipt, None)) => match serde_json::to_string_pretty(&receipt) {
                Ok(rendered) => println!("{rendered}"),
                Err(error) => {
                    eprintln!("Qwen80 true-input all-ten child receipt rendering failed: {error}");
                    process::exit(2);
                }
            },
            Ok((_receipt, Some(error))) => {
                eprintln!("Qwen80 true-input all-ten child terminal refusal: {error}");
                process::exit(2);
            }
            Err(error) => {
                eprintln!("Qwen80 true-input all-ten child durable-capture failure: {error}");
                process::exit(2);
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = arguments;
        eprintln!("Qwen80 true-input all-ten Metal child requires macOS");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_plan_has_exact_router_all_ten_shared_and_combine_order() {
        let plan = staged_dispatch_plan();
        validate_staged_plan(&plan).unwrap();
        assert_eq!(plan.len(), 14);
        assert_eq!(plan[4].grid, (256, INTERMEDIATE, ROUTES));
        assert_eq!(plan[6].grid, (256, HIDDEN, ROUTES));
        assert!(plan[4].purpose.contains("route in Z"));
    }

    #[test]
    fn staged_shader_declares_route_dimension_and_exact_direct_packed_kernels() {
        for kernel in [
            "qwen80_all_ten_routed_wave_gate_up",
            "qwen80_all_ten_routed_wave_swiglu",
            "qwen80_all_ten_routed_wave_down_weighted",
            "qwen80_all_ten_routed_wave_route_guard",
        ] {
            assert!(STAGED_SHADER.contains(&format!("kernel void {kernel}(")));
        }
        assert!(STAGED_SHADER.contains("group_position.z"));
        assert!(STAGED_SHADER.contains("route_count != qwen80_all_ten_route_count"));
        assert!(STAGED_SHADER.contains("route * rows + row"));
        assert!(STAGED_SHADER.contains("observed_ids[route] == expected_ids[route]"));
    }

    #[test]
    fn all_ten_shader_is_registered_without_selecting_a_runtime_or_server_path() {
        let registered = hawking_core::metal::all_shader_sources();
        for kernel in [
            "qwen80_all_ten_routed_wave_route_guard",
            "qwen80_all_ten_routed_wave_gate_up",
            "qwen80_all_ten_routed_wave_swiglu",
            "qwen80_all_ten_routed_wave_down_weighted",
        ] {
            assert!(
                registered.contains(kernel),
                "missing registered all-ten kernel {kernel}"
            );
        }
        assert!(registered.contains("route_count != qwen80_all_ten_route_count"));
        assert!(
            !printable_plan()["claim_boundary"]["device_context_or_dispatch_performed"]
                .as_bool()
                .unwrap()
        );
    }

    #[test]
    fn printable_plan_refuses_fixture_promotion_and_marks_no_device_work() {
        let value = printable_plan();
        assert_eq!(
            value["claim_boundary"]["historical_materialized_combine_fixture_remains_refused"],
            true
        );
        assert_eq!(
            value["claim_boundary"]["device_context_or_dispatch_performed"],
            false
        );
        assert_eq!(value["dispatches"].as_array().unwrap().len(), 14);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn child_parser_requires_exact_metal_mode_and_all_immutable_authorities() {
        let root = Path::new("/tmp/outer");
        let base = vec![
            "--manifest",
            "/tmp/manifest.json",
            "--admission-current",
            "/tmp/admission.json",
            "--router-receipt",
            "/tmp/router.json",
            "--router-outer-receipt",
            "/tmp/router-outer.json",
            "--route-plan",
            "/tmp/routes.json",
            "--first-residual-receipt",
            "/tmp/first-residual.json",
            "--typed-bridge-receipt",
            "/tmp/typed-bridge.json",
            "--fixed-abi-contract",
            "/tmp/fixed-abi.json",
            "--lease-receipt",
            "/tmp/lease.json",
            "--outer-capture-dir",
            "/tmp/outer",
            "--capture-dir",
            "/tmp/outer/inner",
            "--mode",
            "metal",
            "--workers",
            "2",
        ];
        let parsed = parse_true_moe_child_args(base.iter().map(|value| value.to_string())).unwrap();
        assert_eq!(parsed.workers, 2);
        assert_eq!(parsed.capture_dir, root.join("inner"));

        let mut wrong_mode = base.clone();
        let mode_index = wrong_mode
            .iter()
            .position(|value| *value == "metal")
            .unwrap();
        wrong_mode[mode_index] = "cpu";
        assert!(
            parse_true_moe_child_args(wrong_mode.iter().map(|value| value.to_string())).is_err()
        );

        let truncated = &base[..base.len() - 2];
        assert!(
            parse_true_moe_child_args(truncated.iter().map(|value| value.to_string())).is_err()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn static_component_contract_has_exact_prefix_suffix_boundary() {
        assert_eq!(PREFIX_DISPATCHES, 9);
        assert_eq!(SUFFIX_DISPATCHES, staged_dispatch_plan().len());
        assert_eq!(FULL_COMPONENT_DISPATCHES, 23);
        assert_eq!(SOURCE_TOKEN_ID, 1);
        assert_eq!(
            CHILD_SCHEMA,
            "hawking.ascension.qwen80_all_ten_true_moe_graph_device.v1"
        );
    }
}
