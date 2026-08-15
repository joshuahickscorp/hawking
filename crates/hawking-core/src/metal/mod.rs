use crate::{Error, Result};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};

// ── Physical-evidence attribution scope ────────────────────────────────────

/// Immutable identity shared by the probe's OS signpost and every Metal
/// command-buffer/encoder label created during one measured phase.
///
/// The interval id is deliberately computed by the probe *before* work starts;
/// the final interval hash cannot exist until the ending clocks are sampled.
/// Keeping both ids in the emitted JSON binds that predeclared identity to the
/// completed interval without relying on timestamp enclosure for attribution.
#[derive(Debug, Clone)]
pub struct PhysicalTraceIdentity {
    pub interval_id: String,
    pub run_nonce: String,
    pub phase: String,
    pub role: String,
    pub batch: Option<usize>,
    pub iteration: usize,
}

impl PhysicalTraceIdentity {
    pub fn new(
        interval_id: String,
        run_nonce: String,
        phase: String,
        role: String,
        batch: Option<usize>,
        iteration: usize,
    ) -> Result<Self> {
        fn is_sha256(value: &str) -> bool {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        }
        fn safe(value: &str) -> bool {
            !value.is_empty()
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        }
        if !is_sha256(&interval_id) || !is_sha256(&run_nonce) {
            return Err(Error::Metal(
                "physical trace interval id and run nonce must be lowercase SHA-256".into(),
            ));
        }
        if !safe(&phase) || !safe(&role) {
            return Err(Error::Metal(
                "physical trace phase and role contain unsafe label bytes".into(),
            ));
        }
        Ok(Self {
            interval_id,
            run_nonce,
            phase,
            role,
            batch,
            iteration,
        })
    }

    fn base_label(&self) -> String {
        format!(
            "hawking.physical.v1|interval_id={}|run_nonce={}|phase={}|role={}|batch={}|iteration={}",
            self.interval_id,
            self.run_nonce,
            self.phase,
            self.role,
            self.batch
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_owned()),
            self.iteration,
        )
    }
}

struct PhysicalTraceState {
    identity: PhysicalTraceIdentity,
    next_command: AtomicU64,
    next_encoder: AtomicU64,
}

static ACTIVE_PHYSICAL_TRACE: OnceLock<StdMutex<Option<Arc<PhysicalTraceState>>>> = OnceLock::new();

fn active_physical_trace() -> &'static StdMutex<Option<Arc<PhysicalTraceState>>> {
    ACTIVE_PHYSICAL_TRACE.get_or_init(|| StdMutex::new(None))
}

#[derive(Clone)]
struct PhysicalCommandIdentity {
    state: Arc<PhysicalTraceState>,
    command_index: u64,
}

fn physical_command_label(kind: &str) -> Option<(PhysicalCommandIdentity, String)> {
    let state = active_physical_trace().lock().ok()?.clone()?;
    let command_index = state.next_command.fetch_add(1, AtomicOrdering::Relaxed);
    let label = format!(
        "{}|kind={kind}|command_index={command_index}",
        state.identity.base_label(),
    );
    Some((
        PhysicalCommandIdentity {
            state,
            command_index,
        },
        label,
    ))
}

fn physical_encoder_label(command: &PhysicalCommandIdentity, kind: &str, kernel: &str) -> String {
    let encoder_index = command
        .state
        .next_encoder
        .fetch_add(1, AtomicOrdering::Relaxed);
    let safe_kernel: String = kernel
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    format!(
        "{}|kind={kind}|command_index={}|encoder_index={encoder_index}|kernel={safe_kernel}",
        command.state.identity.base_label(),
        command.command_index,
    )
}

#[cfg(target_os = "macos")]
mod physical_signpost {
    use std::ffi::{c_char, CString};

    extern "C" {
        fn hawking_physical_signpost_begin(interval_id: u64, identity: *const c_char);
        fn hawking_physical_signpost_end(interval_id: u64, identity: *const c_char);
    }

    pub struct Interval {
        name: CString,
        id: u64,
    }

    impl Interval {
        pub fn begin(name: String, interval_id: &str) -> std::result::Result<Self, String> {
            let name = CString::new(name).map_err(|_| "signpost name contains NUL".to_owned())?;
            let mut id = u64::from_str_radix(&interval_id[..16], 16)
                .map_err(|_| "invalid signpost interval id".to_owned())?;
            // 0 and u64::MAX are reserved by os_signpost.
            if id == 0 || id == u64::MAX {
                id = 1;
            }
            unsafe {
                hawking_physical_signpost_begin(id, name.as_ptr());
            }
            Ok(Self { name, id })
        }
    }

    impl Drop for Interval {
        fn drop(&mut self) {
            unsafe {
                hawking_physical_signpost_end(self.id, self.name.as_ptr());
            }
        }
    }
}

/// RAII scope for exact physical attribution. Only one scope may be active in
/// a process: a second scope fails closed instead of mixing Metal events.
pub struct PhysicalTraceGuard {
    state: Arc<PhysicalTraceState>,
    #[cfg(target_os = "macos")]
    _signpost: physical_signpost::Interval,
}

/// Exact number of physical Metal command buffers and encoders created while
/// a [`PhysicalTraceGuard`] was active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct PhysicalTraceCounts {
    pub command_count: u64,
    pub encoder_count: u64,
}

impl PhysicalTraceGuard {
    pub fn begin(identity: PhysicalTraceIdentity) -> Result<Self> {
        let state = Arc::new(PhysicalTraceState {
            identity,
            next_command: AtomicU64::new(0),
            next_encoder: AtomicU64::new(0),
        });
        let mut active = active_physical_trace()
            .lock()
            .map_err(|_| Error::Metal("physical trace mutex poisoned".into()))?;
        if active.is_some() {
            return Err(Error::Metal(
                "a physical trace interval is already active".into(),
            ));
        }
        #[cfg(target_os = "macos")]
        let signpost = physical_signpost::Interval::begin(
            state.identity.base_label(),
            &state.identity.interval_id,
        )
        .map_err(Error::Metal)?;
        *active = Some(state.clone());
        drop(active);
        Ok(Self {
            state,
            #[cfg(target_os = "macos")]
            _signpost: signpost,
        })
    }

    /// Read the exact physical command/encoder counters without ending the
    /// trace interval. Call after the measured GPU work has completed.
    pub fn counts(&self) -> PhysicalTraceCounts {
        PhysicalTraceCounts {
            command_count: self.state.next_command.load(AtomicOrdering::Relaxed),
            encoder_count: self.state.next_encoder.load(AtomicOrdering::Relaxed),
        }
    }
}

impl Drop for PhysicalTraceGuard {
    fn drop(&mut self) {
        if let Ok(mut active) = active_physical_trace().lock() {
            if active
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, &self.state))
            {
                *active = None;
            }
        }
        // `_signpost` drops after this method and emits the exact matching end.
    }
}

#[cfg(test)]
mod physical_trace_tests {
    use super::*;
    #[test]
    fn guard_counts_exact_physical_commands_and_encoders() {
        let identity = PhysicalTraceIdentity::new(
            "a".repeat(64),
            "b".repeat(64),
            "unit".into(),
            "counts".into(),
            None,
            0,
        )
        .unwrap();
        let guard = PhysicalTraceGuard::begin(identity).unwrap();
        let (command, _) = physical_command_label("command_buffer").unwrap();
        let _ = physical_encoder_label(&command, "compute_encoder", "kernel_a");
        let _ = physical_encoder_label(&command, "compute_encoder", "kernel_b");
        assert_eq!(
            guard.counts(),
            PhysicalTraceCounts {
                command_count: 1,
                encoder_count: 2,
            }
        );
    }
}

/// Embedded shader sources. Compiled at runtime via
/// `MTLDevice::newLibraryWithSource:` -- shipping a single binary with
/// no `metallib` artifact in tree means contributors don't need
/// xcrun to build.
pub const SHADER_COMMON: &str = include_str!("../../shaders/common.metal");
pub const SHADER_QUANT: &str = include_str!("../../shaders/quant.metal");
pub const SHADER_MOE: &str = include_str!("../../shaders/moe.metal");
pub const SHADER_ATTN: &str = include_str!("../../shaders/attn.metal");
pub const SHADER_SAMPLE: &str = include_str!("../../shaders/sample.metal");
pub const SHADER_MATMUL: &str = include_str!("../../shaders/matmul.metal");
pub const SHADER_MHA: &str = include_str!("../../shaders/mha.metal");
pub const SHADER_MEGAKERNEL: &str = include_str!("../../shaders/megakernel_qwen3b.metal");
/// Exact single-token Gated DeltaNet recurrence for Qwen3-Next.  The initial
/// implementation is a device-resident parity baseline; it is not a complete
/// Qwen decoder or a throughput claim.
pub const SHADER_QWEN_NEXT: &str = include_str!("../../shaders/qwen_next.metal");
/// Bounded Qwen3-Coder-Next layer-3 direct-packed GQA parity kernels. They
/// are compiled only for an isolated attention-stage probe and do not select
/// a generic runtime or serving path.
pub const SHADER_QWEN80_DIRECT_PACKED_ATTENTION_STAGE: &str =
    include_str!("../../shaders/qwen80_direct_packed_attention_stage.metal");
/// Isolated Qwen3-Coder-Next layer-0 post-attention RMSNorm/router/top-10
/// component kernels.  They are compiled only for an explicitly leased,
/// strict-math, non-timed parity capture; no generic runtime or serving path
/// selects them.
pub const SHADER_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10: &str =
    include_str!("../../shaders/qwen80_postnorm_router_top10.metal");
/// Isolated Qwen3-Coder-Next layer-0 route-0/expert-65 gate/up/SwiGLU/down
/// component kernels.  They are available only to the explicit strict-math,
/// non-timed parity probe; no generic runtime or serving path selects them.
pub const SHADER_QWEN80_DIRECT_PACKED_ROUTED_EXPERT_WAVE: &str =
    include_str!("../../shaders/qwen80_routed_expert_wave.metal");
/// Isolated Qwen3-Coder-Next layer-0 shared-expert body kernels. They are
/// compiled only for an explicitly leased strict-math, non-timed component
/// parity capture; no generic runtime or serving path selects them.
pub const SHADER_QWEN80_DIRECT_PACKED_SHARED_EXPERT_WAVE: &str =
    include_str!("../../shaders/qwen80_shared_expert_wave.metal");
/// Isolated Qwen3-Coder-Next layer-0 MoE aggregation/shared-add/second
/// residual kernels. They are compiled only for a separately leased,
/// non-timed materialized-vector component parity capture; no generic runtime
/// or serving path selects them.
pub const SHADER_QWEN80_DIRECT_PACKED_MOE_COMBINE: &str =
    include_str!("../../shaders/qwen80_moe_wave_aggregate_second_residual.metal");
/// Isolated Qwen3-Coder-Next layer-0 source-selected all-ten routed-expert
/// body kernels.  They are only reachable from the explicitly leased,
/// strict-math, non-timed true-input component capture; no generic runtime,
/// watcher, or serving path selects them.
pub const SHADER_QWEN80_ALL_TEN_ROUTED_EXPERT_WAVE: &str =
    include_str!("../../shaders/qwen80_all_ten_routed_expert_wave.metal");
/// Composed Qwen3-Next terminal head (final RMSNorm, all-row lm_head,
/// reserved-tail mask, greedy sample, feedback guard).  Registered so the
/// hybrid token graph can encode it; this is not a generate or TPS claim.
pub const SHADER_QWEN80_DIRECT_PACKED_TERMINAL_HEAD: &str =
    include_str!("../../shaders/qwen80_direct_packed_terminal_head_preflight.metal");
/// Packed binary sign + FP16 group-scale Qwen component matvec. This is a
/// bounded operator primitive, not a complete decoder or model TPS surface.
pub const SHADER_QWEN_BINARY: &str = include_str!("../../shaders/qwen_binary.metal");
/// Device-side glue for the admitted Qwen30 complete-binary runtime.  It
/// retains the packed sign/scale body through embedding, Q/K RMSNorm, and
/// routed-expert control operations; it is not a generic BF16 fallback.
pub const SHADER_QWEN_COMPLETE_RUNTIME: &str =
    include_str!("../../shaders/qwen_complete_runtime.metal");
/// Isolated direct-packed Qwen30 routed-expert gate/up/SwiGLU fusion
/// candidate.  It is compiled into the shared library so a separately named
/// diagnostic runtime path can establish all-layer parity; no generic engine
/// or serving endpoint selects it by default.
pub const SHADER_QWEN_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED: &str =
    include_str!("../../shaders/qwen_direct_packed_gate_up_swiglu_fused.metal");
/// Distinct Qwen30 gate/up paired-topology diagnostic that preserves the
/// scalar control's non-FMA accumulation order.  It is not selected by a
/// generic runtime or endpoint.
pub const SHADER_QWEN_DIRECT_PACKED_GATE_UP_SWIGLU_PAIRED_SCALAR_ORDER: &str =
    include_str!("../../shaders/qwen_direct_packed_gate_up_swiglu_paired_scalar_order.metal");
/// Isolated HQ30GR2 direct-base-plus-sparse-residual Qwen30 gate/up kernel.
/// It is available only to a typed, non-serving candidate diagnostic and is
/// not a generic engine or endpoint selection.
pub const SHADER_QWEN30_QUALITY_REPACK_SPARSE_GATE_UP: &str =
    include_str!("../../shaders/qwen30_quality_repack_sparse_gate_up.metal");
/// Qwen30 device-indexed expert address table (route-id on device; no host
/// expert bind). Mirrors the GLM device expert table idiom for HQ30G1B1 /
/// HGRAVS01 organs.
pub const SHADER_QWEN30_DEVICE_EXPERT_TABLE: &str =
    include_str!("../../shaders/qwen30_device_expert_table.metal");
/// Exact packed uniform-Q4 + FP16 group-scale Qwen component matvec. The
/// fixed group-64 layout is a bounded operator primitive, not a complete
/// decoder or model TPS surface.
pub const SHADER_QWEN_UNIFORM_Q4: &str = include_str!("../../shaders/qwen_uniform_q4.metal");
pub const SHADER_QWEN_UNIFORM_QN: &str = include_str!("../../shaders/qwen_uniform_qn.metal");
/// RWKV-7 WKV-7 single-step decode recurrence (`rwkv7_wkv_decode`). The novel,
/// tps-critical kernel of the RWKV-7 GPU decode path — threadgroup-per-head with
/// the fixed `head_size×head_size` recurrent state in a persistent GPU buffer
/// (no growing KV cache). Self-contained (its own `ArgbufRwkv7Wkv` struct, no
/// shared symbols); appended after the megakernel so it cannot perturb any
/// existing kernel's codegen.
pub const SHADER_RWKV7: &str = include_str!("../../shaders/rwkv7.metal");
/// Product-quantized matvec for `.gravity` `gravity-pq` tensors. Graded against the
/// frozen fixtures in `tests/fixtures/gravity_pq`, whose authority is the Python
/// `gravity_forge.pq_execute`.
pub const SHADER_GRAVITY_PQ: &str = include_str!("../../shaders/gravity_pq.metal");
/// Shared DeepSeek-V4 mHC control-path exp (Darwin double-double reconstruction).
/// Compiled before the P4B/P7 mHC kernels that call into it.
pub const SHADER_DEEPSEEK_V4_MHC_CONTROL_EXP: &str =
    include_str!("../../shaders/deepseek_v4_mhc_control_exp.metal");
/// Isolated DeepSeek-V4 P7 mHC-FFN pre/norm/post kernels.  These are compiled
/// for traceable future device composition only; no Engine or HCLI path selects
/// them until P4B/P6 composition has its own parity admission.
pub const SHADER_DEEPSEEK_V4_P7: &str = include_str!("../../shaders/deepseek_v4_p7.metal");
/// TQ G4 bitslice decode→GEMV kernel family (`tq` feature). Ported verbatim from
/// `vendor/strand-decode-kernel/shaders/strand_bitslice.metal`. Compiled into the
/// runtime library so `pipeline("strand_bitslice_decode")` etc. resolve by name.
/// Self-contained (its own `bs_*` helpers, no shared symbols) — appended last so
/// it cannot perturb any existing kernel's codegen.
#[cfg(feature = "tq")]
pub const SHADER_STRAND_BITSLICE: &str = include_str!("../../shaders/strand_bitslice.metal");

/// Concatenation of all shader sources for a single library compile.
/// Cheaper than separate compile units; lets common helpers be shared.
pub fn all_shader_sources() -> String {
    // `mut` is only exercised on the `tq` path below; the default build pushes
    // nothing, so suppress the unused-mut lint there (and keep the golden shader
    // set — and the profile.rs shader-hash cache key — byte-for-byte unchanged).
    #[cfg_attr(not(feature = "tq"), allow(unused_mut))]
    let mut srcs = vec![
        SHADER_COMMON,
        SHADER_QUANT,
        SHADER_MOE,
        SHADER_ATTN,
        SHADER_SAMPLE,
        // mHC control exp must precede matmul/P7 kernels that call it.
        SHADER_DEEPSEEK_V4_MHC_CONTROL_EXP,
        SHADER_MATMUL,
        SHADER_MHA,
        SHADER_MEGAKERNEL,
        SHADER_QWEN_NEXT,
        SHADER_QWEN80_DIRECT_PACKED_ATTENTION_STAGE,
        SHADER_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10,
        SHADER_QWEN80_DIRECT_PACKED_ROUTED_EXPERT_WAVE,
        SHADER_QWEN80_DIRECT_PACKED_SHARED_EXPERT_WAVE,
        SHADER_QWEN80_DIRECT_PACKED_MOE_COMBINE,
        SHADER_QWEN80_ALL_TEN_ROUTED_EXPERT_WAVE,
        SHADER_QWEN80_DIRECT_PACKED_TERMINAL_HEAD,
        SHADER_QWEN_BINARY,
        SHADER_QWEN_COMPLETE_RUNTIME,
        SHADER_QWEN_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED,
        SHADER_QWEN_DIRECT_PACKED_GATE_UP_SWIGLU_PAIRED_SCALAR_ORDER,
        SHADER_QWEN30_QUALITY_REPACK_SPARSE_GATE_UP,
        SHADER_QWEN30_DEVICE_EXPERT_TABLE,
        SHADER_QWEN_UNIFORM_Q4,
        SHADER_QWEN_UNIFORM_QN,
        SHADER_RWKV7,
        SHADER_GRAVITY_PQ,
        SHADER_DEEPSEEK_V4_P7,
    ];
    // The TQ bitslice family is feature-gated: only compiled into the library
    // when `tq` is on.
    #[cfg(feature = "tq")]
    srcs.push(SHADER_STRAND_BITSLICE);
    srcs.join("\n\n")
}

pub fn current_device_name() -> Option<String> {
    MetalContext::new().ok().map(|ctx| ctx.device_name())
}

// ── Per-dispatch trace types (public so bench can drain them) ──────────────

/// One timed GPU dispatch. `kernel_name` is a `&'static str` to avoid
/// per-dispatch allocation; `layer_hint` comes from the thread-local
/// set by `forward_token_final_norm`.
///
/// `wall_us` is CPU encoding wall time (pipeline lookup + command
/// encoding, not GPU execution). `gpu_us` is populated only by
/// `HAWKING_TCB_TRACE=gpu` mode where each dispatch lands in its own
/// command buffer so `MTLCommandBuffer::gpuStartTime/gpuEndTime` can be
/// read directly. In the default and `HAWKING_TCB_TRACE=cpu` modes
/// `gpu_us` is `None`.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct DispatchSample {
    pub kernel_name: &'static str,
    pub wall_us: u64,
    pub layer_hint: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_us: Option<u64>,
    /// Raw GPU-clock start/end timestamps (ns) for this dispatch, from the
    /// timestamp counter set. Populated ONLY by `HAWKING_TCB_TRACE=gpu_prod`
    /// (single-CB production path); `None` everywhere else. Carrying the raw
    /// endpoints — not just their difference — lets an offline parser compute
    /// the PRODUCTION inter-dispatch gap (start[i+1] - end[i]) without
    /// Instruments. Off by default ⇒ parity-neutral (skipped when `None`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_start_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_end_ns: Option<u64>,
}

/// Timing from one explicitly committed Metal compute dispatch.
///
/// This is deliberately a diagnostic/probe surface rather than a decode
/// scheduler primitive.  It keeps the timing authority honest: the GPU fields
/// are populated only from the completed command buffer's
/// `GPUStartTime`/`GPUEndTime`; they are never a CPU-wall proxy.  A caller must
/// treat `None` as unavailable rather than substituting another clock.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct MetalDispatchTiming {
    /// CPU time spent resolving the pipeline. This can include first-use
    /// compilation and is intentionally separated from command execution.
    pub pipeline_lookup_us: u64,
    /// CPU time from command-buffer allocation through encoder completion.
    pub encode_us: u64,
    /// CPU duration of `commit`.
    pub submit_us: u64,
    /// CPU duration of the completed-command-buffer wait.
    pub wait_us: u64,
    /// End-to-end CPU wall time for the diagnostic dispatch.
    pub host_wall_us: u64,
    /// GPU execution time from `GPUEndTime - GPUStartTime`, if the driver
    /// exposed a valid timestamp pair after completion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_duration_us: Option<u64>,
    /// Raw GPU timeline endpoints in nanoseconds, if the driver exposed them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_start_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_end_ns: Option<u64>,
    /// This surface always owns one command buffer, one compute encoder, and
    /// one `dispatch_threads` invocation. They are explicit so probe receipts
    /// cannot imply a larger topology.
    pub command_buffers: u64,
    pub compute_encoders: u64,
    pub compute_dispatches: u64,
}

/// Timing from one explicitly committed Metal command buffer containing one or
/// more compute encoders/dispatches.
///
/// This is a diagnostic topology surface for bounded component probes.  GPU
/// timing is the completed command buffer's `GPUStartTime`/`GPUEndTime`, so it
/// measures the entire ordered GPU chain rather than adding CPU wall clocks.
/// Per-encoder GPU timestamps are intentionally not implied by this type.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct MetalBatchTiming {
    /// Sum of CPU time resolving every pipeline used by the batch.
    pub pipeline_lookup_us: u64,
    /// CPU time from command-buffer allocation through final encoder close.
    pub encode_us: u64,
    /// CPU duration of the single `commit`.
    pub submit_us: u64,
    /// CPU duration of the completed-command-buffer wait.
    pub wait_us: u64,
    /// End-to-end CPU wall time for the diagnostic batch.
    pub host_wall_us: u64,
    /// GPU execution time from `GPUEndTime - GPUStartTime`, if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_duration_us: Option<u64>,
    /// Raw command-buffer GPU timeline endpoints in nanoseconds, if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_start_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_end_ns: Option<u64>,
    /// Explicit command topology; unlike [`MetalDispatchTiming`], the
    /// encoder/dispatch counts may exceed one.
    pub command_buffers: u64,
    pub compute_encoders: u64,
    pub compute_dispatches: u64,
}

/// Thread-local current-layer index. Set/cleared by the forward pass
/// around each transformer layer so dispatch timing can be attributed
/// to a layer without touching the kernel API.
///
/// Exposed as free functions rather than on MetalContext because the
/// caller (`deepseek_v2::forward_token_final_norm`) runs on whatever
/// thread calls `generate` -- not on the GPU thread.
mod layer_hint {
    use std::cell::Cell;
    thread_local! {
        static CURRENT_LAYER: Cell<Option<u32>> = Cell::new(None);
    }
    pub fn set(v: Option<u32>) {
        CURRENT_LAYER.with(|c| c.set(v));
    }
    pub fn get() -> Option<u32> {
        CURRENT_LAYER.with(|c| c.get())
    }
}

/// Set the current transformer layer index for dispatch attribution.
/// Call before each layer's kernels; call with `None` after the loop.
pub fn set_current_layer(v: Option<u32>) {
    layer_hint::set(v);
}

/// Read the current transformer layer index (used inside dispatch_threads).
pub fn current_layer() -> Option<u32> {
    layer_hint::get()
}

#[cfg(target_os = "macos")]
mod imp {
    use super::*;
    use metal::objc::{class, msg_send, sel, sel_impl};
    use metal::{
        Buffer, CommandBufferRef, CommandQueue, ComputeCommandEncoder, ComputePipelineDescriptor,
        ComputePipelineState, Device, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
        Library, MTLDispatchType, MTLIndirectCommandType, MTLResourceOptions, MTLResourceUsage,
        MTLSize, NSRange,
    };

    /// Read `GPUStartTime` / `GPUEndTime` on an MTLCommandBuffer via raw
    /// objc msg_send. The `metal` 0.29 crate doesn't wrap these selectors,
    /// so we go direct. Returns the GPU compute duration in microseconds,
    /// clamped to 0 if the times come back inverted or zero (driver
    /// quirks; callers shouldn't have to defend).
    ///
    /// SAFETY: caller must guarantee the command buffer has finished
    /// (`wait_until_completed`) before reading; otherwise the values are
    /// undefined.
    unsafe fn cb_gpu_duration_us(cb: &metal::CommandBufferRef) -> u64 {
        match cb_gpu_start_end_s(cb) {
            (Some(start), Some(end)) if end > start => ((end - start) * 1_000_000.0) as u64,
            _ => 0,
        }
    }

    /// Raw GPU start/end CFTimeInterval seconds, or `None` when the driver
    /// returns non-positive / inverted values. Used by the Temporal Gravity
    /// cost ledger so queue-wait derivation never silently substitutes a
    /// CPU proxy for a missing GPU timestamp.
    ///
    /// SAFETY: command buffer must have finished (`wait_until_completed`).
    unsafe fn cb_gpu_start_end_s(cb: &metal::CommandBufferRef) -> (Option<f64>, Option<f64>) {
        let start: f64 = msg_send![cb, GPUStartTime];
        let end: f64 = msg_send![cb, GPUEndTime];
        if start > 0.0 && end > start {
            (Some(start), Some(end))
        } else {
            (None, None)
        }
    }

    /// After commit+wait, push host + GPU times into the cost ledger when
    /// recording. No-op when the ledger is off.
    fn ledger_record_cb_gpu(
        cb: &metal::CommandBufferRef,
        host_commit_us: u64,
        host_wait_us: u64,
        dispatches: u64,
        stage_dispatches: &[(crate::cost_ledger::GpuStage, u64)],
    ) {
        use crate::cost_ledger;
        if !cost_ledger::is_recording() {
            return;
        }
        let (start, end) = unsafe { cb_gpu_start_end_s(cb) };
        cost_ledger::record_gpu_command_buffer_staged(
            host_commit_us,
            host_wait_us,
            start,
            end,
            dispatches,
            stage_dispatches,
        );
    }

    /// Probe whether the device exposes the Metal `timestamp` common counter
    /// set. Does **not** encode sample markers (that would perturb the CB);
    /// only reports capability so the ledger can say "supported but not
    /// recorded" vs "unsupported" vs "unprobed".
    fn ledger_probe_counter_capability(device: &Device) {
        use crate::cost_ledger;
        if !cost_ledger::is_recording() {
            return;
        }
        let supported = find_timestamp_counter_set(device).is_some();
        cost_ledger::record_counter_sample_capability(true, Some(supported));
    }
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    /// Wave-6 residency lever (`HAWKING_QWEN_RESIDENCY=1`, DEFAULT-OFF).
    ///
    /// Create one `MTLResidencySet` on `device`, add every buffer in
    /// `allocations` to it, `commit`, attach it to `queue`, and
    /// `requestResidency`. This pins the decode working set (the
    /// ~1.6 GB no-copy weights mmap + the persistent arena/model
    /// buffers) resident for the queue's lifetime, so the driver stops
    /// implicitly re-validating residency per command buffer -- the
    /// runtime/command-buffer-layer overhead the Wave-6 research flagged
    /// as the home of the llama.cpp tps gap.
    ///
    /// macOS 15+ only (all selectors are `API_AVAILABLE(macos(15.0))`);
    /// the runtime here is macOS 26.5. The `metal` 0.29 crate wraps none
    /// of `MTLResidencySet`, so -- exactly like `cb_gpu_duration_us`
    /// above -- we go through raw objc `msg_send`. A `&XxxRef` encodes as
    /// its object pointer (foreign_obj_type! impls `objc::Message` for
    /// every Ref; the crate itself passes `&BufferRef` to `setBuffer:`),
    /// so we hand `&DeviceRef`/`&CommandQueueRef`/`&BufferRef` straight
    /// to the selectors.
    ///
    /// The created set is INTENTIONALLY not retained by us: `-addResidencySet:`
    /// makes the command queue keep it resident (and retained) for the
    /// whole process, so the `+1` from `newResidencySetWithDescriptor:`
    /// is handed to the queue. We never store it on a `Send + Sync`
    /// engine struct (a raw `*mut Object` is `!Send`), and we never call
    /// it more than once per process.
    ///
    /// SAFETY: every buffer in `allocations` must outlive the command
    /// queue. In hawking they all do -- the weights buffer is backed by
    /// the engine-lifetime GGUF mmap and the arena/model buffers live on
    /// the engine for its whole lifetime.
    unsafe fn install_residency_set(
        device: &metal::DeviceRef,
        queue: &metal::CommandQueueRef,
        allocations: &[&metal::BufferRef],
    ) -> Result<()> {
        use metal::objc::runtime::Object;
        // 1. Descriptor: class!(...) + `new` (alloc+init), like every
        //    MTL*Descriptor in the metal crate (e.g. counters.rs).
        let desc_cls = class!(MTLResidencySetDescriptor);
        let desc: *mut Object = msg_send![desc_cls, new];
        if desc.is_null() {
            return Err(Error::Metal(
                "MTLResidencySetDescriptor alloc failed".into(),
            ));
        }
        let cap: u64 = allocations.len() as u64;
        let _: () = msg_send![desc, setInitialCapacity: cap];
        // 2. Create the set off the device. ObjC selector is
        //    `newResidencySetWithDescriptor:error:` (the Swift name
        //    `makeResidencySetWithDescriptor:` is NOT a selector). Error
        //    out-param idiom copied from `new_library_with_source`.
        let mut err: *mut Object = std::ptr::null_mut();
        let set: *mut Object =
            msg_send![device, newResidencySetWithDescriptor: desc error: &mut err];
        // Balance the +1 from `new` on the descriptor now that the set
        // owns its copy of the parameters.
        let _: () = msg_send![desc, release];
        if set.is_null() {
            let msg = if err.is_null() {
                "newResidencySetWithDescriptor: returned nil".to_string()
            } else {
                let d: *mut Object = msg_send![err, localizedDescription];
                let c: *const std::os::raw::c_char = msg_send![d, UTF8String];
                std::ffi::CStr::from_ptr(c).to_string_lossy().into_owned()
            };
            return Err(Error::Metal(format!(
                "newResidencySetWithDescriptor:error: {msg}"
            )));
        }
        // 3. Add each allocation (uncommitted), then commit in bulk.
        for buf in allocations {
            let _: () = msg_send![set, addAllocation: *buf];
        }
        let _: () = msg_send![set, commit];
        // 4. Attach to the queue (queue now retains it for its lifetime)
        //    and request immediate residency.
        let _: () = msg_send![queue, addResidencySet: set];
        let _: () = msg_send![set, requestResidency];
        // Intentional: do NOT release `set`. The queue holds it for the
        // process lifetime; the `+1` is handed off deliberately.
        Ok(())
    }

    /// v2.2.0-L7: lookup the `timestamp` common counter set on the device,
    /// returning `Some(CounterSet)` if available. Apple silicon (M1/M2/M3)
    /// always supports the timestamp counter set; intel macs may not.
    ///
    /// Counter-set names are reported as NSString. We compare against the
    /// well-known constant `MTLCommonCounterSetTimestamp` (which is itself
    /// an NSString with value "timestamp"); the simplest match is by name.
    fn find_timestamp_counter_set(device: &Device) -> Option<::metal::CounterSet> {
        let sets = device.counter_sets();
        sets.into_iter().find(|s| s.name() == "timestamp")
    }

    /// v2.2.0-L7: counter-sample tracer used by `ProdCbGpu` mode.
    ///
    /// A recyclable sample buffer leased to one `TokenCommandBuffer` at a
    /// time; `sample_count` is sized for 2 samples per dispatch ×
    /// MAX_DISPATCHES. Each dispatch occupies indices `[2*n, 2*n+1]`. After
    /// that CB completes, the sample buffer holds raw GPU timestamps (ns);
    /// `gpu_us = (ts[2n+1] - ts[2n]) / 1000`. Recycling avoids exhausting
    /// Metal's small per-device CounterSampleBuffer limit during a complete
    /// multi-command-buffer token profile.
    struct ProdCbTracer {
        sample_buf: ::metal::CounterSampleBuffer,
        /// Index of the next pair (so the start of the next dispatch's
        /// samples is `2 * next_pair`). One pair per dispatch.
        next_pair: AtomicUsize,
        capacity_pairs: usize,
        /// Pending samples, populated in dispatch order with the pair index
        /// they were stamped at; resolved into `tcb_samples` post-wait.
        pending: Mutex<Vec<ProdCbPending>>,
    }

    struct ProdCbPending {
        kernel_name: &'static str,
        cpu_us: u64,
        pair_index: usize,
        layer_hint: Option<u32>,
    }

    impl ProdCbTracer {
        /// Sample slot capacity per TCB. Apple caps the sample buffer at
        /// 32 KiB (= 4096 u64 samples = 2048 pairs). One TCB = one token =
        /// ~270 dispatches for V2-Lite (27 layers × ~10 kernels + LM head),
        /// so 1024 pairs (= 16 KiB) is comfortably above the worst-case
        /// per-token dispatch count.
        const CAPACITY_PAIRS: usize = 1024;

        fn try_new(device: &Device) -> Option<Self> {
            let cset = find_timestamp_counter_set(device)?;
            let desc = ::metal::CounterSampleBufferDescriptor::new();
            desc.set_counter_set(&cset);
            desc.set_sample_count((Self::CAPACITY_PAIRS * 2) as u64);
            // Shared storage so we can read raw via resolveCounterRange:
            // without a separate blit-encoder resolve pass.
            desc.set_storage_mode(::metal::MTLStorageMode::Shared);
            let sample_buf = device
                .new_counter_sample_buffer_with_descriptor(&desc)
                .ok()?;
            Some(Self {
                sample_buf,
                next_pair: AtomicUsize::new(0),
                capacity_pairs: Self::CAPACITY_PAIRS,
                pending: Mutex::new(Vec::with_capacity(Self::CAPACITY_PAIRS)),
            })
        }

        /// Reserve a pair index for the next dispatch. Returns `None` if
        /// capacity is exhausted (in which case the caller falls back to
        /// recording the sample without `gpu_us`).
        fn reserve_pair(&self) -> Option<usize> {
            let i = self.next_pair.fetch_add(1, Ordering::Relaxed);
            if i < self.capacity_pairs {
                Some(i)
            } else {
                None
            }
        }

        /// Record one dispatch's metadata; gpu_us is populated post-wait.
        fn record_pending(
            &self,
            kernel_name: &'static str,
            cpu_us: u64,
            pair_index: usize,
            layer_hint: Option<u32>,
        ) {
            self.pending.lock().push(ProdCbPending {
                kernel_name,
                cpu_us,
                pair_index,
                layer_hint,
            });
        }

        /// After commit+wait, walk pending and emit one `DispatchSample`
        /// per recorded dispatch, with `gpu_us` filled from the resolved
        /// counter sample buffer.
        ///
        /// `resolveCounterRange:` returns NSData of `2 * sample_count`
        /// `u64` words; we read pairs and subtract. The values are in
        /// nanoseconds for the timestamp counter set (per Apple docs).
        fn drain(&self) -> Vec<super::DispatchSample> {
            let pending = std::mem::take(&mut *self.pending.lock());
            let pair_count = self
                .next_pair
                .load(Ordering::Relaxed)
                .min(self.capacity_pairs);
            if pair_count == 0 {
                let samples = pending
                    .into_iter()
                    .map(|p| super::DispatchSample {
                        kernel_name: p.kernel_name,
                        wall_us: p.cpu_us,
                        layer_hint: p.layer_hint,
                        gpu_us: None,
                        gpu_start_ns: None,
                        gpu_end_ns: None,
                    })
                    .collect();
                self.next_pair.store(0, Ordering::Relaxed);
                return samples;
            }
            // Resolve the [0, 2*pair_count) sample range. Returns NSData.
            // SAFETY: CB has committed + waited before this is called;
            // resolveCounterRange: is a synchronous read on shared storage.
            let timestamps = unsafe {
                let ns_range = ::metal::NSRange {
                    location: 0,
                    length: (pair_count * 2) as u64,
                };
                let nsdata: *mut metal::objc::runtime::Object =
                    msg_send![&*self.sample_buf, resolveCounterRange: ns_range];
                if nsdata.is_null() {
                    Vec::new()
                } else {
                    let bytes: *const u8 = msg_send![nsdata, bytes];
                    let len: usize = msg_send![nsdata, length];
                    let n_u64 = len / 8;
                    let slice = std::slice::from_raw_parts(bytes as *const u64, n_u64);
                    slice.to_vec()
                }
            };
            // Per Apple, an "absent" sample is encoded as MTLCounterErrorValue
            // (0xFFFFFFFFFFFFFFFF). If we see one we leave gpu_us=None.
            const ERR: u64 = u64::MAX;
            let samples = pending
                .into_iter()
                .map(|p| {
                    let i0 = p.pair_index * 2;
                    let i1 = i0 + 1;
                    let valid = i1 < timestamps.len()
                        && timestamps[i0] != ERR
                        && timestamps[i1] != ERR
                        && timestamps[i1] >= timestamps[i0];
                    let (gpu_us, gpu_start_ns, gpu_end_ns) = if valid {
                        (
                            Some((timestamps[i1] - timestamps[i0]) / 1000),
                            Some(timestamps[i0]),
                            Some(timestamps[i1]),
                        )
                    } else {
                        (None, None, None)
                    };
                    super::DispatchSample {
                        kernel_name: p.kernel_name,
                        wall_us: p.cpu_us,
                        layer_hint: p.layer_hint,
                        gpu_us,
                        gpu_start_ns,
                        gpu_end_ns,
                    }
                })
                .collect();
            // This tracer is returned to a context-local pool only after its
            // command buffer has completed and timestamps have been resolved.
            // The next lease overwrites the same sample indices safely.
            self.next_pair.store(0, Ordering::Relaxed);
            samples
        }
    }

    // Re-export Metal's Buffer type so callers can hold pinned-weight
    // handles without depending on the upstream `metal` crate directly.
    pub use metal::Buffer as PinnedBuffer;

    /// Accumulated per-dispatch timing samples. Gate-able via
    /// `HAWKING_TRACE_DISPATCH` env var; when the var is absent the
    /// samples vec is never populated and overhead is zero.
    pub struct DispatchTrace {
        pub samples: Mutex<Vec<super::DispatchSample>>,
    }

    impl DispatchTrace {
        fn new() -> Self {
            Self {
                // Pre-allocate for a 64-token decode to avoid Vec growth
                // perturbing measurements (≈60 dispatches/token × 64 tokens).
                samples: Mutex::new(Vec::with_capacity(10_000)),
            }
        }

        fn record(&self, kernel_name: &'static str, wall_us: u64, layer_hint: Option<u32>) {
            self.samples.lock().push(super::DispatchSample {
                kernel_name,
                wall_us,
                layer_hint,
                gpu_us: None,
                gpu_start_ns: None,
                gpu_end_ns: None,
            });
        }

        /// Drain all collected samples (called by bench after a run).
        pub fn drain(&self) -> Vec<super::DispatchSample> {
            std::mem::take(&mut *self.samples.lock())
        }
    }

    /// Structural counters for buffer allocations and command-buffer commits.
    /// Only incremented when `MetalContext::trace_dispatch` is true; zero-cost
    /// when off (no atomic ops on the hot path).
    pub struct MetalContextStats {
        pub buffers_created: AtomicUsize,
        pub bytes_allocated: AtomicUsize,
        pub commits: AtomicUsize,
    }

    impl MetalContextStats {
        fn new() -> Self {
            Self {
                buffers_created: AtomicUsize::new(0),
                bytes_allocated: AtomicUsize::new(0),
                commits: AtomicUsize::new(0),
            }
        }

        /// Drain counters, returning (buffers_created, bytes_allocated, commits).
        pub fn drain(&self) -> (usize, usize, usize) {
            (
                self.buffers_created.swap(0, Ordering::Relaxed),
                self.bytes_allocated.swap(0, Ordering::Relaxed),
                self.commits.swap(0, Ordering::Relaxed),
            )
        }
    }

    /// The owned device handle. Cheap to clone via `Arc`.
    #[derive(Clone)]
    pub struct MetalContext {
        inner: Arc<Inner>,
        /// Shared trace accumulator; `Arc` so `Clone` works without copying.
        pub trace: Arc<DispatchTrace>,
        /// Shared structural counters; `Arc` so `Clone` works without copying.
        pub stats: Arc<MetalContextStats>,
        /// Whether to collect dispatch trace and structural counters.
        /// Mirrors `EngineConfig::trace_dispatch`; env var `HAWKING_TRACE_DISPATCH`
        /// acts as a fallback when this is false.
        pub trace_dispatch: bool,
        /// Reusable diagnostic counter-sample buffers. A complete token may
        /// commit dozens of serial command buffers; leasing and returning a
        /// tracer prevents the hardware sample-buffer quota from silently
        /// truncating a gpu_prod profile after its first few stages.
        prod_cb_tracer_pool: Arc<Mutex<Vec<ProdCbTracer>>>,
    }

    /// One command buffer that can encode several compute kernels before
    /// a single commit/wait. This is the stepping stone between the
    /// current per-kernel dispatch path and the future strict single
    /// FlashMoE kernel.
    pub struct CommandBatch<'a> {
        ctx: &'a MetalContext,
        cmd: &'a CommandBufferRef,
        physical_trace: Option<PhysicalCommandIdentity>,
        // An opt-in `MTLDispatchTypeConcurrent` encoder for a caller-proved
        // independent wave.  It is intentionally separate from the ordinary
        // per-dispatch and ordered-pair paths: callers must close it before a
        // dependent dispatch can be encoded.
        concurrent_encoder: Option<ComputeCommandEncoder>,
        pipeline_lookup_us: u64,
        compute_encoders: u64,
        compute_dispatches: u64,
    }

    struct Inner {
        device: Device,
        queue: CommandQueue,
        library: Library,
        pipelines: Mutex<HashMap<String, ComputePipelineState>>,
        icb_pipelines: Mutex<HashMap<String, ComputePipelineState>>,
    }

    /// Resolve a runtime kernel name to a `&'static str` for zero-alloc
    /// trace recording. Covers all kernel names used by hawking; anything
    /// unknown falls through to `"other"`.
    fn static_kernel_name(name: &str) -> &'static str {
        match name {
            "rmsnorm" => "rmsnorm",
            "gemv_f16" => "gemv_f16",
            "gemv_f32_attn" => "gemv_f32_attn",
            "mla_decode_kernel" => "mla_decode_kernel",
            "moe_topk_gate" => "moe_topk_gate",
            "moe_gather_combine" => "moe_gather_combine",
            "moe_batched_silu_mul" => "moe_batched_silu_mul",
            "moe_route_accumulate" => "moe_route_accumulate",
            "moe_route_accumulate_add" => "moe_route_accumulate_add",
            "sample_argmax_f32" => "sample_argmax_f32",
            "sample_argmax_f32_with_finite" => "sample_argmax_f32_with_finite",
            // Admitted Qwen30 complete-binary runtime.  These labels remain
            // per-stage so a future complete-token profile can distinguish
            // packed decode, state, and routed-expert time rather than fold
            // it into an opaque "other" bucket.
            "qwen_binary_sign_scale_matvec" => "qwen_binary_sign_scale_matvec",
            "qwen_binary_sign_scale_matvec_serial" => "qwen_binary_sign_scale_matvec_serial",
            "qwen_binary_sign_scale_matvec_tiled" => "qwen_binary_sign_scale_matvec_tiled",
            "qwen_binary_sign_scale_matvec_simdgroup_candidate" => {
                "qwen_binary_sign_scale_matvec_simdgroup_candidate"
            }
            "qwen_binary_sign_scale_matvec_qkv" => "qwen_binary_sign_scale_matvec_qkv",
            "qwen_binary_sign_scale_matvec_qkv_rowblock4" => {
                "qwen_binary_sign_scale_matvec_qkv_rowblock4"
            }
            "qwen_binary_postnorm_router_matvec" => "qwen_binary_postnorm_router_matvec",
            "qwen_binary_sign_scale_matvec_rowblock2" => {
                "qwen_binary_sign_scale_matvec_rowblock2"
            }
            "qwen_binary_sign_scale_matvec_rowblock4" => {
                "qwen_binary_sign_scale_matvec_rowblock4"
            }
            "qwen_binary_sign_scale_matvec_rowblock8" => {
                "qwen_binary_sign_scale_matvec_rowblock8"
            }
            "qwen_complete_binary_decode_vector" => "qwen_complete_binary_decode_vector",
            "qwen_complete_binary_embedding_lookup" => "qwen_complete_binary_embedding_lookup",
            "qwen_uniform_q4_group64_matvec" => "qwen_uniform_q4_group64_matvec",
            "qwen_uniform_q4_group64_matvec_rowblock" => {
                "qwen_uniform_q4_group64_matvec_rowblock"
            }
            "qwen_uniform_q4_group64_matvec_simdgroup" => {
                "qwen_uniform_q4_group64_matvec_simdgroup"
            }
            "qwen_uniform_q4_group64_matvec_simdgroup_rowblock4" => {
                "qwen_uniform_q4_group64_matvec_simdgroup_rowblock4"
            }
            "qwen_uniform_q4_group64_matvec_simdgroup_rowblock8" => {
                "qwen_uniform_q4_group64_matvec_simdgroup_rowblock8"
            }
            "qwen_uniform_q4_group64_matvec_simdgroup_x64" => {
                "qwen_uniform_q4_group64_matvec_simdgroup_x64"
            }
            "qwen_uniform_q4_group64_matvec_qkv" => "qwen_uniform_q4_group64_matvec_qkv",
            "qwen_uniform_q4_group64_matvec_qkv_simdgroup" => {
                "qwen_uniform_q4_group64_matvec_qkv_simdgroup"
            }
            "qwen_uniform_q4_decode_vector" => "qwen_uniform_q4_decode_vector",
            "qwen_uniform_q4_embedding_lookup" => "qwen_uniform_q4_embedding_lookup",
            "qwen_uniform_q4_embedding_lookup_device_token" => {
                "qwen_uniform_q4_embedding_lookup_device_token"
            }
            "qwen_uniform_q4_group64_final_norm_lm_head_simdgroup8" => {
                "qwen_uniform_q4_group64_final_norm_lm_head_simdgroup8"
            },
            "qwen_uniform_qn_matvec" => "qwen_uniform_qn_matvec",
            "qwen_uniform_qn_decode_vector" => "qwen_uniform_qn_decode_vector",
            "qwen_uniform_qn_embedding_lookup" => "qwen_uniform_qn_embedding_lookup",
            "qwen_complete_binary_embedding_lookup_device_token" => {
                "qwen_complete_binary_embedding_lookup_device_token"
            }
            "qwen_complete_rmsnorm_rows_f32" => "qwen_complete_rmsnorm_rows_f32",
            "qwen_complete_qk_rmsnorm_rope_kv_append_f32" => {
                "qwen_complete_qk_rmsnorm_rope_kv_append_f32"
            }
            "qwen_complete_normalize_route_weights" => "qwen_complete_normalize_route_weights",
            "qwen_complete_silu_mul_offset" => "qwen_complete_silu_mul_offset",
            "qwen_direct_packed_gate_up_swiglu_fused_candidate" => {
                "qwen_direct_packed_gate_up_swiglu_fused_candidate"
            }
            "qwen_direct_packed_gate_up_swiglu_paired_scalar_order_candidate" => {
                "qwen_direct_packed_gate_up_swiglu_paired_scalar_order_candidate"
            }
            "qwen30_expert_table_binary_matvec" => "qwen30_expert_table_binary_matvec",
            "qwen30_expert_table_binary_matvec_serial" => "qwen30_expert_table_binary_matvec_serial",
            "qwen30_expert_table_uniform_q4_matvec_serial" => {
                "qwen30_expert_table_uniform_q4_matvec_serial"
            }
            "qwen30_expert_table_uniform_q4_matvec_rowblock" => {
                "qwen30_expert_table_uniform_q4_matvec_rowblock"
            }
            "qwen30_expert_table_uniform_q4_matvec_simdgroup" => {
                "qwen30_expert_table_uniform_q4_matvec_simdgroup"
            }
            "qwen30_expert_table_uniform_q4_matvec_simdgroup_rowblock4" => {
                "qwen30_expert_table_uniform_q4_matvec_simdgroup_rowblock4"
            }
            "qwen30_expert_table_uniform_q4_matvec_simdgroup_rowblock8" => {
                "qwen30_expert_table_uniform_q4_matvec_simdgroup_rowblock8"
            },
            "qwen30_expert_table_binary_matvec_simdgroup" => {
                "qwen30_expert_table_binary_matvec_simdgroup"
            }
            "qwen30_expert_table_binary_matvec_rowblock2" => {
                "qwen30_expert_table_binary_matvec_rowblock2"
            }
            "qwen30_expert_table_binary_matvec_rowblock4" => {
                "qwen30_expert_table_binary_matvec_rowblock4"
            }
            "qwen30_expert_table_binary_matvec_rowblock8" => {
                "qwen30_expert_table_binary_matvec_rowblock8"
            }
            "qwen30_expert_table_paired_gate_up_swiglu" => {
                "qwen30_expert_table_paired_gate_up_swiglu"
            }
            "qwen30_expert_table_uniform_q4_paired_gate_up_swiglu" => {
                "qwen30_expert_table_uniform_q4_paired_gate_up_swiglu"
            }
            "qwen30_expert_table_uniform_q4_paired_gate_up_swiglu_simdgroup8" => {
                "qwen30_expert_table_uniform_q4_paired_gate_up_swiglu_simdgroup8"
            }
            "qwen30_expert_table_hgravs_gemv" => "qwen30_expert_table_hgravs_gemv",
            "qwen30_expert_table_hgravs_gemv_rowblock2" => {
                "qwen30_expert_table_hgravs_gemv_rowblock2"
            }
            "qwen30_expert_table_hgravs_gemv_rowblock4" => {
                "qwen30_expert_table_hgravs_gemv_rowblock4"
            }
            "qwen30_expert_table_hgravs_gemv_rowblock8" => {
                "qwen30_expert_table_hgravs_gemv_rowblock8"
            }
            "qwen_complete_weighted_expert_add" => "qwen_complete_weighted_expert_add",
            "qwen_complete_any_nonfinite_f32" => "qwen_complete_any_nonfinite_f32",
            "qwen_next_gated_delta_decode_single" => "qwen_next_gated_delta_decode_single",
            "qwen_next_ba_to_decay_beta" => "qwen_next_ba_to_decay_beta",
            "qwen_next_direct_packed_input_rmsnorm" => "qwen_next_direct_packed_input_rmsnorm",
            "qwen_next_qkvz_rearrange_conv_l2" => "qwen_next_qkvz_rearrange_conv_l2",
            "qwen_next_deltanet_gated_rmsnorm" => "qwen_next_deltanet_gated_rmsnorm",
            "qwen_next_add_residual" => "qwen_next_add_residual",
            "qwen80_attention_qk_norm_rope_cache" => "qwen80_attention_qk_norm_rope_cache",
            "qwen80_attention_apply_sigmoid_gate" => "qwen80_attention_apply_sigmoid_gate",
            "qwen80_postnorm_router_top10_rmsnorm" => "qwen80_postnorm_router_top10_rmsnorm",
            "qwen80_postnorm_router_top10_matvec" => "qwen80_postnorm_router_top10_matvec",
            "qwen80_postnorm_router_top10_select" => "qwen80_postnorm_router_top10_select",
            "qwen80_all_ten_routed_wave_route_guard" => "qwen80_all_ten_routed_wave_route_guard",
            "qwen80_all_ten_routed_wave_gate_up" => "qwen80_all_ten_routed_wave_gate_up",
            "qwen80_all_ten_routed_wave_swiglu" => "qwen80_all_ten_routed_wave_swiglu",
            "qwen80_all_ten_routed_wave_down_weighted" => {
                "qwen80_all_ten_routed_wave_down_weighted"
            }
            "qwen80_shared_expert_wave_gate_up" => "qwen80_shared_expert_wave_gate_up",
            "qwen80_shared_expert_wave_swiglu" => "qwen80_shared_expert_wave_swiglu",
            "qwen80_shared_expert_wave_down" => "qwen80_shared_expert_wave_down",
            "qwen80_shared_expert_wave_scalar_gate" => "qwen80_shared_expert_wave_scalar_gate",
            "qwen80_shared_expert_wave_apply_sigmoid_gate" => {
                "qwen80_shared_expert_wave_apply_sigmoid_gate"
            }
            "qwen80_moe_wave_aggregate_second_residual_route_sum" => {
                "qwen80_moe_wave_aggregate_second_residual_route_sum"
            }
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual" => {
                "qwen80_moe_wave_aggregate_second_residual_add_shared_residual"
            }
            "qwen80_terminal_head_final_rmsnorm_direct_packed" => {
                "qwen80_terminal_head_final_rmsnorm_direct_packed"
            }
            "qwen80_terminal_head_all_row_direct_packed" => {
                "qwen80_terminal_head_all_row_direct_packed"
            }
            "qwen80_terminal_head_mask_reserved_tail" => "qwen80_terminal_head_mask_reserved_tail",
            "qwen80_terminal_head_greedy_sample_lowest_id" => {
                "qwen80_terminal_head_greedy_sample_lowest_id"
            }
            "qwen80_terminal_head_feedback_guard" => "qwen80_terminal_head_feedback_guard",
            // attn / rope / embed kernels
            "rope_inplace" => "rope_inplace",
            "rope_norm_llama_b9430" => "rope_norm_llama_b9430",
            "rope_norm_llama_b9430_cache_kv_f16" => "rope_norm_llama_b9430_cache_kv_f16",
            "rope_norm_llama_b9430_qkv_cache_f16" => "rope_norm_llama_b9430_qkv_cache_f16",
            "rmsnorm_llama_b9430" => "rmsnorm_llama_b9430",
            "add_rmsnorm_llama_b9430" => "add_rmsnorm_llama_b9430",
            "swiglu_llama_b9430" => "swiglu_llama_b9430",
            "round_f16_llama_b9430" => "round_f16_llama_b9430",
            "mha_decode_llama_b9430_short" => "mha_decode_llama_b9430_short",
            "mha_decode_llama_b9430_fattn_main" => "mha_decode_llama_b9430_fattn_main",
            "mha_decode_llama_b9430_fattn_reduce" => "mha_decode_llama_b9430_fattn_reduce",
            "mha_decode_llama_b9430_fattn_prefill_main" => {
                "mha_decode_llama_b9430_fattn_prefill_main"
            }
            "mha_decode_llama_b9430_fattn_prefill_reduce" => {
                "mha_decode_llama_b9430_fattn_prefill_reduce"
            }
            "llama_b9430_cache_append_kv_f16" => "llama_b9430_cache_append_kv_f16",
            "llama_b9430_cache_append_kv_f16_off" => "llama_b9430_cache_append_kv_f16_off",
            // dequant / gemm variants
            "dequant_q8_0" => "dequant_q8_0",
            "gemm_q4_k_m_fused" => "gemm_q4_k_m_fused",
            "gemm_q4_k_m_llama_b9430" => "gemm_q4_k_m_llama_b9430",
            "gemm_q4_k_m_llama_b9430_batched" => "gemm_q4_k_m_llama_b9430_batched",
            "gemm_q4_k_m_llama_b9430_pair" => "gemm_q4_k_m_llama_b9430_pair",
            "gemm_q5_k_serial_authority" => "gemm_q5_k_serial_authority",
            "gemm_q6_k_llama_b9430" => "gemm_q6_k_llama_b9430",
            "gemm_q4_k_m_fused_simd" => "gemm_q4_k_m_fused_simd",
            "gemm_q4_k_m_fused_v2" => "gemm_q4_k_m_fused_v2",
            // Bounded DeepSeek-V4 FP8 component authority probe. This is a
            // source-native E4M3FN/E8M0 matvec, not a registered V4 runtime.
            "deepseek_v4_fp8_e4m3fn_e8m0_matvec_authority" => {
                "deepseek_v4_fp8_e4m3fn_e8m0_matvec_authority"
            }
            // Optional component-only split-K SIMDgroup candidate. It is not
            // wired into any V4 engine/runtime path; the dedicated sweep
            // records whether it earns promotion against the authority probe.
            "deepseek_v4_fp8_e4m3fn_e8m0_matvec_simdgroup_v4_splitk_candidate" => {
                "deepseek_v4_fp8_e4m3fn_e8m0_matvec_simdgroup_v4_splitk_candidate"
            }
            // Bounded source `Linear` checkpoint: GPU BF16 act_quant into
            // native E4M3FN/E8M0, followed by a separate source-native FP8
            // projection. These remain component probes, not a V4 runtime.
            "deepseek_v4_act_quant_bf16_ue8m0_authority" => {
                "deepseek_v4_act_quant_bf16_ue8m0_authority"
            }
            // Optional block-parallel SIMDgroup act-quant candidate.  It is
            // intentionally component-only and can be selected only by its
            // byte-exact CPU-oracle sweep receipt, never by a V4 runtime.
            "deepseek_v4_act_quant_bf16_ue8m0_simdgroup_block_candidate" => {
                "deepseek_v4_act_quant_bf16_ue8m0_simdgroup_block_candidate"
            }
            "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority" => {
                "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority"
            }
            "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_simdgroup_v4_splitk_candidate" => {
                "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_simdgroup_v4_splitk_candidate"
            }
            // Bounded P3A layer-0 pre-attention authority probes.  These
            // symbols are intentionally traceable but are not registered in
            // an Engine, token loop, or HCLI runtime path.
            "deepseek_v4_p3a_layer0_hc_attn_pre_bos_authority" => {
                "deepseek_v4_p3a_layer0_hc_attn_pre_bos_authority"
            }
            "deepseek_v4_p3a_rmsnorm_bf16_authority" => "deepseek_v4_p3a_rmsnorm_bf16_authority",
            "deepseek_v4_p3a_fp32_to_bf16_authority" => "deepseek_v4_p3a_fp32_to_bf16_authority",
            "deepseek_v4_p3a_per_head_rmsnorm_bf16_authority" => {
                "deepseek_v4_p3a_per_head_rmsnorm_bf16_authority"
            }
            // Bounded P4A continuation: complete layer-0 attention at the
            // fixed BOS/position-zero specialization. These remain separate
            // source-authority probes, not Engine/HCLI runtime kernels.
            "deepseek_v4_p4a_kv_nonrope_qat_inplace_authority" => {
                "deepseek_v4_p4a_kv_nonrope_qat_inplace_authority"
            }
            "deepseek_v4_p4a_sparse_attention_position0_sink_authority" => {
                "deepseek_v4_p4a_sparse_attention_position0_sink_authority"
            }
            "deepseek_v4_p4a_wo_a_convert_bf16_einsum_authority" => {
                "deepseek_v4_p4a_wo_a_convert_bf16_einsum_authority"
            }
            "deepseek_v4_p4a_hc_attn_post_authority" => "deepseek_v4_p4a_hc_attn_post_authority",
            // Bounded P4B continuation: real position-one, ratio-zero RoPE
            // and causal KV cache read/write authority probes. They are not
            // a persistent decode/runtime registration.
            "deepseek_v4_p4b_rope_position1_bf16_authority" => {
                "deepseek_v4_p4b_rope_position1_bf16_authority"
            }
            "deepseek_v4_p4b_kv_cache_write_bf16_authority" => {
                "deepseek_v4_p4b_kv_cache_write_bf16_authority"
            }
            "deepseek_v4_p4b_sparse_attention_position1_two_kv_sink_authority" => {
                "deepseek_v4_p4b_sparse_attention_position1_two_kv_sink_authority"
            }
            // Ratio-0 growing-KV sibling of the position1 kernel above. It is
            // compiled into matmul.metal and dispatched, but was never given a
            // trace name, so every dispatch of it was attributed to "other" and
            // vanished from per-kernel profiling.
            "deepseek_v4_p4_sparse_attention_ratio0_growing_kv_sink_authority" => {
                "deepseek_v4_p4_sparse_attention_ratio0_growing_kv_sink_authority"
            }
            // Isolated P4B mHC-control precision experiment. This is never a
            // baseline/runtime selection: it may be invoked only to test
            // whether precise exponent evaluation restores control and final
            // storage equality against the independent CPU oracle.
            "deepseek_v4_p4b_hc_post_comb_precise_exp_candidate" => {
                "deepseek_v4_p4b_hc_post_comb_precise_exp_candidate"
            }
            // Bounded DeepSeek-V4 FP4 routed-expert component authority
            // probe. This is deliberately not a registered V4 runtime.
            "deepseek_v4_fp4_e2m1fn_x2_e8m0_matvec_authority" => {
                "deepseek_v4_fp4_e2m1fn_x2_e8m0_matvec_authority"
            }
            // Optional component-only packed FP4 split-K candidate. This
            // trace name intentionally makes it impossible to conflate with
            // the serial source-native authority probe.
            "deepseek_v4_fp4_e2m1fn_x2_e8m0_matvec_simdgroup_v4_splitk_candidate" => {
                "deepseek_v4_fp4_e2m1fn_x2_e8m0_matvec_simdgroup_v4_splitk_candidate"
            }
            // Bounded P5B layer-0 MoE source-storage authority kernels.  They
            // are traceable component probes only; no runtime path selects
            // them until a separately proved causal adapter exists.
            "deepseek_v4_p5b_swiglu_route_bf16_authority" => {
                "deepseek_v4_p5b_swiglu_route_bf16_authority"
            }
            "deepseek_v4_p5b_fp4_act_quant_e2m1fn_x2_e8m0_matvec_authority" => {
                "deepseek_v4_p5b_fp4_act_quant_e2m1fn_x2_e8m0_matvec_authority"
            }
            "deepseek_v4_p5b_route_shared_combine_bf16_authority" => {
                "deepseek_v4_p5b_route_shared_combine_bf16_authority"
            }
            // Bounded P6A layer-0 full hash-route / six-expert wave. These
            // remain explicitly traceable component probes; no runtime path
            // selects them until the causal layer adapter is separately proved.
            "deepseek_v4_p6a_gate_bf16_matvec_authority" => {
                "deepseek_v4_p6a_gate_bf16_matvec_authority"
            }
            // The isolated P0 C4 Gate reduction was admitted solely for the
            // bounded reusable P6 layer-0 seam. Keep it trace-named so it
            // cannot disappear into the generic `other` bucket.
            "deepseek_v4_p0_gate_reduction_c4_simd32_fma_candidate" => {
                "deepseek_v4_p0_gate_reduction_c4_simd32_fma_candidate"
            }
            "deepseek_v4_p6a_hash_route_sqrtsoftplus_authority" => {
                "deepseek_v4_p6a_hash_route_sqrtsoftplus_authority"
            }
            "deepseek_v4_p6a_learned_bias_route_sqrtsoftplus_authority" => {
                "deepseek_v4_p6a_learned_bias_route_sqrtsoftplus_authority"
            }
            "deepseek_v4_p6a_swiglu_route_weight_buffer_bf16_authority" => {
                "deepseek_v4_p6a_swiglu_route_weight_buffer_bf16_authority"
            }
            "deepseek_v4_p6a_route6_shared_combine_bf16_authority" => {
                "deepseek_v4_p6a_route6_shared_combine_bf16_authority"
            }
            // Isolated P7 mHC-FFN composition kernels.  These are traceable
            // library residents only, not a registered causal-runtime path.
            "deepseek_v4_p7_mhc_ffn_pre_authority" => "deepseek_v4_p7_mhc_ffn_pre_authority",
            "deepseek_v4_p7_ffn_rmsnorm_bf16_authority" => {
                "deepseek_v4_p7_ffn_rmsnorm_bf16_authority"
            }
            "deepseek_v4_p7_mhc_ffn_post_authority" => "deepseek_v4_p7_mhc_ffn_post_authority",
            "gemv_f32_moe" => "gemv_f32_moe",
            "moe_grouped_gemm_q4" => "moe_grouped_gemm_q4",
            // indexed moe batched gemm variants
            "moe_batched_gemm_q4_indexed" => "moe_batched_gemm_q4_indexed",
            "moe_batched_gemm_q4_indexed_v2" => "moe_batched_gemm_q4_indexed_v2",
            "moe_batched_gemm_q4_indexed_v2s" => "moe_batched_gemm_q4_indexed_v2s",
            "moe_batched_gemm_q4_indexed_v2t" => "moe_batched_gemm_q4_indexed_v2t",
            "moe_batched_gemm_q5_0_indexed" => "moe_batched_gemm_q5_0_indexed",
            "moe_batched_gemm_q6_k_indexed" => "moe_batched_gemm_q6_k_indexed",
            "moe_batched_gemm_q8_0_indexed" => "moe_batched_gemm_q8_0_indexed",
            // silu / activation
            "silu_mul" => "silu_mul",
            // residual / element-wise kernels
            "add_inplace" => "add_inplace",
            "add_inplace_off" => "add_inplace_off",
            // Phase 7 fp16 kernels
            "rmsnorm_f16" => "rmsnorm_f16",
            "silu_mul_f16" => "silu_mul_f16",
            // sampling kernels
            "sample_repetition" => "sample_repetition",
            "sample_temperature" => "sample_temperature",
            // v0.5.7 sampling kernels
            "sample_topk" => "sample_topk",
            "sample_topp" => "sample_topp",
            "sample_multinomial" => "sample_multinomial",
            // v0.5.8 fused rmsnorm+gemv kernels
            "rmsnorm_gemv_f32_attn_pinned" => "rmsnorm_gemv_f32_attn_pinned",
            "rmsnorm_gemv_q4k_pair" => "rmsnorm_gemv_q4k_pair",
            // v0.5.9 fp16 activation kernels
            "gemv_f32_attn_f16" => "gemv_f32_attn_f16",
            "gemv_f32_moe_f16" => "gemv_f32_moe_f16",
            "softmax_f16" => "softmax_f16",
            "layer_norm_f16" => "layer_norm_f16",
            // v1.1.0-X simdgroup LM-head
            "gemv_f16_simdmat" => "gemv_f16_simdmat",
            "gemv_simdgroup_f32" => "gemv_simdgroup_f32",
            // GLM native.bf16 lm_head (sequential accumulate, host parity)
            "gemv_native_bf16_seq" => "gemv_native_bf16_seq",
            // GLM activation-aware factorized GEMV.
            "activation_aware_project_f16" => "activation_aware_project_f16",
            "activation_aware_expand_f16" => "activation_aware_expand_f16",
            // Additive/default-off native.bf16 accuracy candidates.
            "gemv_native_bf16_neumaier" => "gemv_native_bf16_neumaier",
            "gemv_native_bf16_neumaier_compensated_product" => {
                "gemv_native_bf16_neumaier_compensated_product"
            }
            // Gravity .gravity elementwise / PQ (expert-wave, resident)
            "gravity_silu_mul_f32" => "gravity_silu_mul_f32",
            "gravity_axpy_f32" => "gravity_axpy_f32",
            "gravity_add_inplace_f32" => "gravity_add_inplace_f32",
            "gravity_rmsnorm_f32" => "gravity_rmsnorm_f32",
            "gravity_rope_interleaved_f32" => "gravity_rope_interleaved_f32",
            "gravity_rope_prefix_tail_f32" => "gravity_rope_prefix_tail_f32",
            "gravity_rope_prefix_tail_positioned_f32" => "gravity_rope_prefix_tail_positioned_f32",
            "gravity_glm_mla_append_kv" => "gravity_glm_mla_append_kv",
            "gravity_glm_mla_append_compact" => "gravity_glm_mla_append_compact",
            "gravity_pq_k_transpose_heads" => "gravity_pq_k_transpose_heads",
            "gravity_glm_compact_ranked_attn" => "gravity_glm_compact_ranked_attn",
            "gravity_pq_v_rows_heads" => "gravity_pq_v_rows_heads",
            "gravity_glm_build_queries" => "gravity_glm_build_queries",
            "gravity_copy_head_prefix_f32" => "gravity_copy_head_prefix_f32",
            "gravity_layernorm_affine_f32" => "gravity_layernorm_affine_f32",
            "gravity_glm_append_index_key" => "gravity_glm_append_index_key",
            "gravity_copy_tail_f32" => "gravity_copy_tail_f32",
            "gravity_glm_dsa_scores" => "gravity_glm_dsa_scores",
            "gravity_glm_stable_topk_f32" => "gravity_glm_stable_topk_f32",
            "gravity_glm_radix_topk_f32" => "gravity_glm_radix_topk_f32",
            "gravity_glm_sort_u32_ascending" => "gravity_glm_sort_u32_ascending",
            "gravity_glm_sparse_attn" => "gravity_glm_sparse_attn",
            "gravity_glm_router_correct" => "gravity_glm_router_correct",
            "gravity_glm_router_select_noaux_f32" => "gravity_glm_router_select_noaux_f32",
            "gravity_glm_expert_trace_copy" => "gravity_glm_expert_trace_copy",
            "gravity_glm_expert_table_validate" => "gravity_glm_expert_table_validate",
            "gravity_glm_expert_table_pq_matvec" => "gravity_glm_expert_table_pq_matvec",
            "gravity_glm_expert_table_native_bf16_matvec" => {
                "gravity_glm_expert_table_native_bf16_matvec"
            }
            "gravity_glm_expert_table_zero_f32" => "gravity_glm_expert_table_zero_f32",
            "gravity_glm_expert_table_silu_mul_f32" => "gravity_glm_expert_table_silu_mul_f32",
            "gravity_glm_expert_table_axpy_f32" => "gravity_glm_expert_table_axpy_f32",
            "gravity_glm_expert_table_residual_add_f32" => {
                "gravity_glm_expert_table_residual_add_f32"
            }
            "gravity_zero_f32" => "gravity_zero_f32",
            "replayable_compute_graph" => "replayable_compute_graph",
            "gravity_pq_matvec" => "gravity_pq_matvec",
            "gravity_raw_q5_0_matvec" => "gravity_raw_q5_0_matvec",
            "gravity_raw_q5_0_pair_matvec" => "gravity_raw_q5_0_pair_matvec",
            "gravity_raw_q8_0_matvec" => "gravity_raw_q8_0_matvec",
            "gravity_raw_q5q5qv_rope_append" => "gravity_raw_q5q5qv_rope_append",
            "gravity_pq_matvec_bits8_direct" => "gravity_pq_matvec_bits8_direct",
            "gravity_pq_matvec_bits8_vec4" => "gravity_pq_matvec_bits8_vec4",
            "gravity_pq_matvec_bits8_double_single" => "gravity_pq_matvec_bits8_double_single",
            "gravity_pq_matvec_bits8_2d" => "gravity_pq_matvec_bits8_2d",
            "gravity_pq_reduce_2d" => "gravity_pq_reduce_2d",
            // v0.5.10 fp16 Q-format kernels
            "gemm_q4_k_m_fused_f16" => "gemm_q4_k_m_fused_f16",
            "moe_grouped_gemm_q4_f16" => "moe_grouped_gemm_q4_f16",
            "dequant_q6_k_f16" => "dequant_q6_k_f16",
            // v1.1.1 / v2.1.0 -- T1.1 audit closed 22 names previously
            // bucketed as "other" (incl. the post-T2.1 default MoE Q4_K
            // v2t_gu_v2 kernel itself -- biggest attribution miss).
            "moe_batched_gemm_q4_indexed_v2t_gu" => "moe_batched_gemm_q4_indexed_v2t_gu",
            "moe_batched_gemm_q4_indexed_v2t_gu_v2" => "moe_batched_gemm_q4_indexed_v2t_gu_v2",
            "moe_batched_gemm_q4_indexed_v2t_gu_v3" => "moe_batched_gemm_q4_indexed_v2t_gu_v3",
            "moe_batched_gemm_q8_0_indexed_v2t" => "moe_batched_gemm_q8_0_indexed_v2t",
            "moe_batched_gemm_q5_0_indexed_v2t" => "moe_batched_gemm_q5_0_indexed_v2t",
            "moe_batched_gemm_q6_k_indexed_v2t" => "moe_batched_gemm_q6_k_indexed_v2t",
            "gemm_q3_k_fused_v2" => "gemm_q3_k_fused_v2",
            "gemm_q3_k_fused_2r" => "gemm_q3_k_fused_2r",
            "gemm_q3_k_v4_predec" => "gemm_q3_k_v4_predec",
            "gemm_q6_k_fused_v2" => "gemm_q6_k_fused_v2",
            "gemm_q6_k_fused_v2_swiglu" => "gemm_q6_k_fused_v2_swiglu",
            "gemm_q6_k_fused_v2_swiglu_2r" => "gemm_q6_k_fused_v2_swiglu_2r",
            "gemm_q6_k_fused_v2_swiglu_4r" => "gemm_q6_k_fused_v2_swiglu_4r",
            "gemm_q4_k_m_simdmat" => "gemm_q4_k_m_simdmat",
            "gemm_q4_k_m_v3_8r" => "gemm_q4_k_m_v3_8r",
            "gemm_q4_k_v4_predec" => "gemm_q4_k_v4_predec",
            "gemm_q4_k_v4_predec_swiglu" => "gemm_q4_k_v4_predec_swiglu",
            "gemm_q4_k_v4_predec_2r" => "gemm_q4_k_v4_predec_2r",
            "gemm_q4_k_v4_predec_2r_add" => "gemm_q4_k_v4_predec_2r_add",
            "gemm_q4_k_v4_predec_2r_add_f16s" => "gemm_q4_k_v4_predec_2r_add_f16s",
            "gemm_q4_k_v4_predec_2r_swiglu" => "gemm_q4_k_v4_predec_2r_swiglu",
            "gemm_q4_k_v4_predec_2r_f16s" => "gemm_q4_k_v4_predec_2r_f16s",
            "gemm_q4_k_v4_predec_4r" => "gemm_q4_k_v4_predec_4r",
            "gemm_q4_k_v4_predec_4r_add" => "gemm_q4_k_v4_predec_4r_add",
            "gemm_q4_k_v4_predec_4r_add_f16s" => "gemm_q4_k_v4_predec_4r_add_f16s",
            "gemm_q4_k_v4_predec_4r_f16s" => "gemm_q4_k_v4_predec_4r_f16s",
            "gemm_q4_k_v4_predec_4r_swiglu" => "gemm_q4_k_v4_predec_4r_swiglu",
            "gemm_q4_k_v4_predec_f16s_4r_swiglu" => "gemm_q4_k_v4_predec_f16s_4r_swiglu",
            "gemm_q4_k_v4_predec_pair" => "gemm_q4_k_v4_predec_pair",
            "gemm_q4_k_v4_predec_pair_2r" => "gemm_q4_k_v4_predec_pair_2r",
            "gemm_q4_k_v4_predec_pair_2r_inline" => "gemm_q4_k_v4_predec_pair_2r_inline",
            "gemm_q4_k_v4_predec_pair_2r_inline_nox" => "gemm_q4_k_v4_predec_pair_2r_inline_nox",
            "gemm_q4_k_v4_predec_pair_2r_inline_f16s" => "gemm_q4_k_v4_predec_pair_2r_inline_f16s",
            "gemm_q4_k_v4_predec_pair_f16s_nox" => "gemm_q4_k_v4_predec_pair_f16s_nox",
            "gemm_q4_k_v4_predec_pair_f16s_halfreg" => "gemm_q4_k_v4_predec_pair_f16s_halfreg",
            "gemm_q4_k_v4_predec_pair_3r" => "gemm_q4_k_v4_predec_pair_3r",
            "gemm_q4_k_v4_predec_pair_4r" => "gemm_q4_k_v4_predec_pair_4r",
            "gemm_q4_k_v4_predec_pair_4r_f16s" => "gemm_q4_k_v4_predec_pair_4r_f16s",
            "gemm_q4_k_v4_predec_pair_8r" => "gemm_q4_k_v4_predec_pair_8r",
            "gemm_q4_k_v4_predec_pair_f16s" => "gemm_q4_k_v4_predec_pair_f16s",
            "gemm_q4_k_m_v3_dual" => "gemm_q4_k_m_v3_dual",
            "gemm_q4k_predec_qkv_triple" => "gemm_q4k_predec_qkv_triple",
            "gemm_q4k_q4k_q6k_triple" => "gemm_q4k_q4k_q6k_triple",
            "gemm_q4k_predec_qkv_rope_append" => "gemm_q4k_predec_qkv_rope_append",
            "gemm_q4k_predec_qkv_rope_append_f16s" => "gemm_q4k_predec_qkv_rope_append_f16s",
            "gemm_q4k_predec_qkv_rope_append_4r" => "gemm_q4k_predec_qkv_rope_append_4r",
            "gemm_q4k_predec_qkv_rope_append_4r_f16s" => "gemm_q4k_predec_qkv_rope_append_4r_f16s",
            "gemm_q4k_q4k_q6k_rope_append" => "gemm_q4k_q4k_q6k_rope_append",
            "gemm_q4_k_m_v3_llama" => "gemm_q4_k_m_v3_llama",
            "gemv_f16_f16in" => "gemv_f16_f16in",
            "kv_append_f32" => "kv_append_f32",
            "rmsnorm_f32" => "rmsnorm_f32",
            "rmsnorm_f32_to_f16" => "rmsnorm_f32_to_f16",
            "rmsnorm_gemv_f16w_attn_pinned" => "rmsnorm_gemv_f16w_attn_pinned",
            "rmsnorm_gemv_f16w_attn_pinned_v2t" => "rmsnorm_gemv_f16w_attn_pinned_v2t",
            "rope_q_f32_inplace" => "rope_q_f32_inplace",
            "rope_q_interleaved_concat" => "rope_q_interleaved_concat",
            "kv_append_vbias_f32" => "kv_append_vbias_f32",
            "rope_qk_f32_b1_bias" => "rope_qk_f32_b1_bias",
            "rope_qk_kv_append_vbias_f32" => "rope_qk_kv_append_vbias_f32",
            "rope_slice_f32_inplace" => "rope_slice_f32_inplace",
            "rope_slice_interleaved_concat" => "rope_slice_interleaved_concat",
            "embed_lookup_f32" => "embed_lookup_f32",
            "embed_lookup_q4_k_m" => "embed_lookup_q4_k_m",
            "flash_attn_decode_kernel" => "flash_attn_decode_kernel",
            // Session F (sketch) -- fused add_inplace + rmsnorm_f32
            "add_rmsnorm_fused" => "add_rmsnorm_fused",
            // W4A8 production wire-up (2026-05-24)
            "quantize_f32_to_int8_per_block" => "quantize_f32_to_int8_per_block",
            "quantize_f32_to_int8_per_block_scaled" => "quantize_f32_to_int8_per_block_scaled",
            "gemm_q4_k_a8_v3_8r" => "gemm_q4_k_a8_v3_8r",
            "add_rmsnorm_fused_q8" => "add_rmsnorm_fused_q8",
            "add_rmsnorm_fused_q8_scaled" => "add_rmsnorm_fused_q8_scaled",
            // 0.4 (2026-05-30): close the remaining unmapped 'other' bucket so
            // every dispatched kernel is attributed and traces pass the §1 gate
            // (INV2: other-share must be < 5%). Covers decode-path attention +
            // residual/util kernels plus the prefill/batched, W4A8-per-channel,
            // Q4K_FAST, and megakernel/POC kernels (named now so they attribute
            // correctly the moment they're wired in, never silently as 'other').
            "mha_decode_f32" => "mha_decode_f32",
            "mha_decode_f32_batched" => "mha_decode_f32_batched",
            "mha_decode_f16kv" => "mha_decode_f16kv",
            "mha_decode_f16kv_batched" => "mha_decode_f16kv_batched",
            "mha_decode_flash_f32" => "mha_decode_flash_f32",
            "mha_decode_flash_f16kv" => "mha_decode_flash_f16kv",
            "kv_quant_int4_append" => "kv_quant_int4_append",
            "mha_decode_flash_int4kv" => "mha_decode_flash_int4kv",
            "kv_int4_calib_max" => "kv_int4_calib_max",
            "kv_quant_int4_append_pc" => "kv_quant_int4_append_pc",
            "mha_decode_flash_int4kv_pc" => "mha_decode_flash_int4kv_pc",
            "mla_decode_kernel_q8kv" => "mla_decode_kernel_q8kv",
            "kv_append_q8_0_f32" => "kv_append_q8_0_f32",
            "add_inplace_broadcast" => "add_inplace_broadcast",
            "memcpy_f32_off" => "memcpy_f32_off",
            "memcpy_f32_to_f16_off" => "memcpy_f32_to_f16_off",
            "llama_b9430_cache_append_f32_f16" => "llama_b9430_cache_append_f32_f16",
            "add_rmsnorm_fused_batched" => "add_rmsnorm_fused_batched",
            "gemm_q4_k_m_batched_v2" => "gemm_q4_k_m_batched_v2",
            "gemm_q4_k_m_batched_v3" => "gemm_q4_k_m_batched_v3",
            "gemm_q4_k_m_batched_v3w" => "gemm_q4_k_m_batched_v3w",
            "gemm_q4_k_m_batched_v3w_predec" => "gemm_q4_k_m_batched_v3w_predec",
            "gemm_q4_k_m_batched_v4r_predec" => "gemm_q4_k_m_batched_v4r_predec",
            "gemm_q4_k_m_batched_v4r_predec_swiglu" => "gemm_q4_k_m_batched_v4r_predec_swiglu",
            "gemm_q4k_fast_v1" => "gemm_q4k_fast_v1",
            "gemm_q4_k_a8_v3_8r_per_channel" => "gemm_q4_k_a8_v3_8r_per_channel",
            "moe_batched_gemm_q8_0_indexed_v2t_w4" => "moe_batched_gemm_q8_0_indexed_v2t_w4",
            "moe_batched_gemm_q8_0_indexed_v2t_w2" => "moe_batched_gemm_q8_0_indexed_v2t_w2",
            "quantize_f32_to_int8_per_channel" => "quantize_f32_to_int8_per_channel",
            "embed_lookup_rmsnorm_f32" => "embed_lookup_rmsnorm_f32",
            "sample_argmax_f32_batched" => "sample_argmax_f32_batched",
            "embed_lookup_f32_batched" => "embed_lookup_f32_batched",
            "rmsnorm_f32_batched" => "rmsnorm_f32_batched",
            "rope_f32_batched_multiseq" => "rope_f32_batched_multiseq",
            "rope_qk_f32_batched_multiseq" => "rope_qk_f32_batched_multiseq",
            "mha_decode_f32_batched_multiseq" => "mha_decode_f32_batched_multiseq",
            "kv_scatter_append_multiseq" => "kv_scatter_append_multiseq",
            "qwen3b_megakernel_2layer" => "qwen3b_megakernel_2layer",
            "qwen3b_megakernel_nlayer" => "qwen3b_megakernel_nlayer",
            "gpu_address_probe" => "gpu_address_probe",
            "use_resource_poc_add" => "use_resource_poc_add",
            // RWKV-7 WKV-7 single-step decode recurrence (state-space model)
            // plus the elementwise glue kernels for its time-mix/channel-mix.
            "rwkv7_wkv_decode" => "rwkv7_wkv_decode",
            "rwkv7_token_shift_lerp" => "rwkv7_token_shift_lerp",
            "rwkv7_channel_mix_shift" => "rwkv7_channel_mix_shift",
            "rwkv7_tanh_inplace" => "rwkv7_tanh_inplace",
            "rwkv7_decay_act" => "rwkv7_decay_act",
            "rwkv7_sigmoid_bias" => "rwkv7_sigmoid_bias",
            "rwkv7_sigmoid_inplace" => "rwkv7_sigmoid_inplace",
            "rwkv7_value_residual_mix" => "rwkv7_value_residual_mix",
            "rwkv7_kk_kmix" => "rwkv7_kk_kmix",
            "rwkv7_relu_sq_inplace" => "rwkv7_relu_sq_inplace",
            "rwkv7_add_into" => "rwkv7_add_into",
            "rwkv7_layernorm" => "rwkv7_layernorm",
            "rwkv7_copy" => "rwkv7_copy",
            "rwkv7_mul_inplace" => "rwkv7_mul_inplace",
            // RWKV-7 continuous-batch (multi-seq) decode kernels — B streams,
            // one weight read across the B activation columns.
            "rwkv7_wkv_decode_multiseq" => "rwkv7_wkv_decode_multiseq",
            "rwkv7_kk_kmix_multiseq" => "rwkv7_kk_kmix_multiseq",
            "rwkv7_token_shift_lerp_multiseq" => "rwkv7_token_shift_lerp_multiseq",
            "rwkv7_channel_mix_shift_multiseq" => "rwkv7_channel_mix_shift_multiseq",
            "rwkv7_shift_writeback_multiseq" => "rwkv7_shift_writeback_multiseq",
            "rwkv7_layernorm_multiseq" => "rwkv7_layernorm_multiseq",
            "rwkv7_decay_act_multiseq" => "rwkv7_decay_act_multiseq",
            "rwkv7_sigmoid_bias_multiseq" => "rwkv7_sigmoid_bias_multiseq",
            "rwkv7_value_residual_mix_multiseq" => "rwkv7_value_residual_mix_multiseq",
            "rwkv7_add_into_flat" => "rwkv7_add_into_flat",
            _ => "other",
        }
    }

    #[cfg(test)]
    mod static_kernel_name_tests {
        use super::static_kernel_name;
        use crate::metal::{
            SHADER_DEEPSEEK_V4_MHC_CONTROL_EXP, SHADER_DEEPSEEK_V4_P7, SHADER_GRAVITY_PQ,
            SHADER_MATMUL, SHADER_MOE,
        };
        #[test]
        fn compiled_dormant_resident_kernels_have_static_trace_names() {
            const DORMANT_RESIDENT_KERNELS: &[&str] = &[
                "gravity_rmsnorm_f32",
                "gravity_rope_interleaved_f32",
                "gravity_rope_prefix_tail_f32",
                "gravity_rope_prefix_tail_positioned_f32",
                "gravity_glm_mla_append_kv",
                "gravity_glm_mla_append_compact",
                "gravity_pq_k_transpose_heads",
                "gravity_glm_compact_ranked_attn",
                "gravity_pq_v_rows_heads",
                "gravity_glm_build_queries",
                "gravity_copy_head_prefix_f32",
                "gravity_layernorm_affine_f32",
                "gravity_glm_append_index_key",
                "gravity_copy_tail_f32",
                "gravity_glm_dsa_scores",
                "gravity_glm_stable_topk_f32",
                "gravity_glm_radix_topk_f32",
                "gravity_glm_sparse_attn",
                "gravity_glm_router_correct",
                "gravity_glm_router_select_noaux_f32",
                "gravity_glm_expert_trace_copy",
                "gravity_glm_expert_table_validate",
                "gravity_glm_expert_table_pq_matvec",
                "gravity_glm_expert_table_native_bf16_matvec",
                "gravity_glm_expert_table_zero_f32",
                "gravity_glm_expert_table_silu_mul_f32",
                "gravity_glm_expert_table_axpy_f32",
                "gravity_glm_expert_table_residual_add_f32",
                "gravity_zero_f32",
            ];
            for &name in DORMANT_RESIDENT_KERNELS {
                assert_eq!(
                    static_kernel_name(name),
                    name,
                    "{name} would be attributed to the generic trace bucket"
                );
                assert!(
                    SHADER_GRAVITY_PQ.contains(&format!("kernel void {name}(")),
                    "{name} is registered but not compiled from gravity_pq.metal"
                );
            }
            assert_eq!(static_kernel_name("not_a_hawking_kernel"), "other");
        }

        #[test]
        fn deepseek_v4_fp8_authority_probe_is_trace_named_and_compiled() {
            const KERNEL: &str = "deepseek_v4_fp8_e4m3fn_e8m0_matvec_authority";
            assert_eq!(static_kernel_name(KERNEL), KERNEL);
            assert!(
                SHADER_MATMUL.contains(&format!("kernel void {KERNEL}(")),
                "DeepSeek-V4 FP8 authority kernel must remain in the runtime Metal library"
            );
        }

        #[test]
        fn deepseek_v4_fp4_authority_probe_is_trace_named_and_compiled() {
            const KERNEL: &str = "deepseek_v4_fp4_e2m1fn_x2_e8m0_matvec_authority";
            assert_eq!(static_kernel_name(KERNEL), KERNEL);
            assert!(
                SHADER_MATMUL.contains(&format!("kernel void {KERNEL}(")),
                "DeepSeek-V4 FP4 authority kernel must remain in the runtime Metal library"
            );
        }

        #[test]
        fn deepseek_v4_p5b_moe_authority_kernels_are_trace_named_and_compiled() {
            const KERNELS: &[&str] = &[
                "deepseek_v4_p5b_swiglu_route_bf16_authority",
                "deepseek_v4_p5b_fp4_act_quant_e2m1fn_x2_e8m0_matvec_authority",
                "deepseek_v4_p5b_route_shared_combine_bf16_authority",
            ];
            for &kernel in KERNELS {
                assert_eq!(static_kernel_name(kernel), kernel);
                assert!(
                    SHADER_MOE.contains(&format!("kernel void {kernel}(")),
                    "DeepSeek-V4 P5B authority kernel must remain in moe.metal"
                );
            }
        }

        #[test]
        fn deepseek_v4_p6a_full_route_wave_kernels_are_trace_named_and_compiled() {
            const KERNELS: &[&str] = &[
                "deepseek_v4_p6a_gate_bf16_matvec_authority",
                "deepseek_v4_p6a_hash_route_sqrtsoftplus_authority",
                "deepseek_v4_p6a_learned_bias_route_sqrtsoftplus_authority",
                "deepseek_v4_p6a_swiglu_route_weight_buffer_bf16_authority",
                "deepseek_v4_p6a_route6_shared_combine_bf16_authority",
            ];
            for &kernel in KERNELS {
                assert_eq!(static_kernel_name(kernel), kernel);
                assert!(
                    SHADER_MOE.contains(&format!("kernel void {kernel}(")),
                    "DeepSeek-V4 P6A authority kernel must remain in moe.metal"
                );
            }
        }

        #[test]
        fn deepseek_v4_p6_c4_gate_kernel_is_trace_named_and_compiled() {
            const KERNEL: &str = "deepseek_v4_p0_gate_reduction_c4_simd32_fma_candidate";
            assert_eq!(static_kernel_name(KERNEL), KERNEL);
            assert!(
                SHADER_MOE.contains(&format!("kernel void {KERNEL}(")),
                "DeepSeek-V4 P6 C4 Gate kernel must remain in moe.metal"
            );
        }

        #[test]
        fn deepseek_v4_optional_simdgroup_candidates_are_trace_named_and_compiled() {
            const KERNELS: &[&str] = &[
                "deepseek_v4_fp8_e4m3fn_e8m0_matvec_simdgroup_v4_splitk_candidate",
                "deepseek_v4_fp4_e2m1fn_x2_e8m0_matvec_simdgroup_v4_splitk_candidate",
            ];
            for &kernel in KERNELS {
                assert_eq!(static_kernel_name(kernel), kernel);
                assert!(
                    SHADER_MATMUL.contains(&format!("kernel void {kernel}(")),
                    "optional DeepSeek-V4 SIMDgroup candidate must remain in the runtime Metal library"
                );
            }
        }

        #[test]
        fn deepseek_v4_source_linear_checkpoint_kernels_are_trace_named_and_compiled() {
            const KERNELS: &[&str] = &[
                "deepseek_v4_act_quant_bf16_ue8m0_authority",
                "deepseek_v4_act_quant_bf16_ue8m0_simdgroup_block_candidate",
                "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority",
                "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_simdgroup_v4_splitk_candidate",
            ];
            for &kernel in KERNELS {
                assert_eq!(static_kernel_name(kernel), kernel);
                assert!(
                    SHADER_MATMUL.contains(&format!("kernel void {kernel}(")),
                    "DeepSeek-V4 source-linear checkpoint kernel must remain in the runtime Metal library"
                );
            }
        }

        #[test]
        fn deepseek_v4_p3a_layer0_preattention_kernels_are_trace_named_and_compiled() {
            const KERNELS: &[&str] = &[
                "deepseek_v4_p3a_layer0_hc_attn_pre_bos_authority",
                "deepseek_v4_p3a_rmsnorm_bf16_authority",
                "deepseek_v4_p3a_fp32_to_bf16_authority",
                "deepseek_v4_p3a_per_head_rmsnorm_bf16_authority",
            ];
            for &kernel in KERNELS {
                assert_eq!(static_kernel_name(kernel), kernel);
                assert!(
                    SHADER_MATMUL.contains(&format!("kernel void {kernel}(")),
                    "DeepSeek-V4 P3A kernel must remain in the runtime Metal library"
                );
            }
        }

        #[test]
        fn deepseek_v4_p4a_layer0_attention_kernels_are_trace_named_and_compiled() {
            const KERNELS: &[&str] = &[
                "deepseek_v4_p4a_kv_nonrope_qat_inplace_authority",
                "deepseek_v4_p4a_sparse_attention_position0_sink_authority",
                "deepseek_v4_p4a_wo_a_convert_bf16_einsum_authority",
                "deepseek_v4_p4a_hc_attn_post_authority",
            ];
            for &kernel in KERNELS {
                assert_eq!(static_kernel_name(kernel), kernel);
                assert!(
                    SHADER_MATMUL.contains(&format!("kernel void {kernel}(")),
                    "DeepSeek-V4 P4A kernel must remain in the runtime Metal library"
                );
            }
        }

        #[test]
        fn deepseek_v4_p4b_position1_causal_kv_kernels_are_trace_named_and_compiled() {
            const KERNELS: &[&str] = &[
                "deepseek_v4_p4b_rope_position1_bf16_authority",
                "deepseek_v4_p4b_kv_cache_write_bf16_authority",
                "deepseek_v4_p4b_sparse_attention_position1_two_kv_sink_authority",
                "deepseek_v4_p4_sparse_attention_ratio0_growing_kv_sink_authority",
                "deepseek_v4_p4b_hc_post_comb_precise_exp_candidate",
            ];
            for &kernel in KERNELS {
                assert_eq!(static_kernel_name(kernel), kernel);
                assert!(
                    SHADER_MATMUL.contains(&format!("kernel void {kernel}(")),
                    "DeepSeek-V4 P4B kernel must remain in the runtime Metal library"
                );
            }
            assert!(
                SHADER_DEEPSEEK_V4_MHC_CONTROL_EXP.contains("deepseek_v4_mhc_control_expf"),
                "mHC control exp helper must compile into the shared Metal library"
            );
        }

        #[test]
        fn deepseek_v4_p7_mhc_ffn_kernels_are_trace_named_and_compiled() {
            const KERNELS: &[&str] = &[
                "deepseek_v4_p7_mhc_ffn_pre_authority",
                "deepseek_v4_p7_ffn_rmsnorm_bf16_authority",
                "deepseek_v4_p7_mhc_ffn_post_authority",
            ];
            for &kernel in KERNELS {
                assert_eq!(static_kernel_name(kernel), kernel);
                assert!(
                    SHADER_DEEPSEEK_V4_P7.contains(&format!("kernel void {kernel}(")),
                    "DeepSeek-V4 P7 kernel must remain in deepseek_v4_p7.metal"
                );
            }
        }
    }

    /// Metallib disk cache keyed by (device name, shader source hash, math mode).
    /// Disable with `HAWKING_METALLIB_CACHE=0`. When a matching `.metallib` is
    /// present it is loaded; otherwise sources are compiled at runtime. Writing
    /// a new `.metallib` requires the Xcode `metal`/`metallib` tools; when they
    /// are absent the warm path still benefits from any pre-seeded cache file
    /// and from the OS Metal shader cache for repeated `newLibraryWithSource`.
    fn metallib_cache_enabled() -> bool {
        match std::env::var("HAWKING_METALLIB_CACHE") {
            Ok(v) if matches!(v.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF") => {
                false
            }
            _ => true,
        }
    }

    fn metallib_cache_root() -> std::path::PathBuf {
        if let Ok(dir) = std::env::var("HAWKING_METALLIB_CACHE_DIR") {
            return std::path::PathBuf::from(dir);
        }
        if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
            return std::path::PathBuf::from(xdg).join("hawking").join("metallib");
        }
        if let Ok(home) = std::env::var("HOME") {
            return std::path::PathBuf::from(home)
                .join(".cache")
                .join("hawking")
                .join("metallib");
        }
        std::path::PathBuf::from("/tmp/hawking-cache/metallib")
    }

    fn shader_source_sha256(src: &str) -> String {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(src.as_bytes()))
    }

    fn sanitize_device_name(name: &str) -> String {
        name.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }

    fn metallib_cache_path(device: &Device, source_sha: &str, strict_math: bool) -> std::path::PathBuf {
        let math = if strict_math {
            "strict_math"
        } else {
            "fast_math_default"
        };
        metallib_cache_root()
            .join(sanitize_device_name(&device.name().to_string()))
            .join(format!("{source_sha}_{math}.metallib"))
    }

    /// Opt-in offline metallib build via Xcode tools (`HAWKING_METALLIB_BUILD=1`).
    /// Default off so normal experiment starts never shell out.
    fn try_build_metallib_with_xcrun(
        src: &str,
        out_path: &std::path::Path,
        strict_math: bool,
    ) -> Option<()> {
        use std::io::Write;
        use std::process::Command;
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).ok()?;
        }
        let tmp_dir = out_path
            .parent()?
            .join(format!(".build-{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir).ok()?;
        let metal_src = tmp_dir.join("all_shaders.metal");
        let air = tmp_dir.join("all_shaders.air");
        let metallib_tmp = tmp_dir.join("out.metallib");
        {
            let mut f = std::fs::File::create(&metal_src).ok()?;
            f.write_all(src.as_bytes()).ok()?;
        }
        let metal_bin = Command::new("xcrun")
            .args(["-sdk", "macosx", "-f", "metal"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|p| !p.is_empty())?;
        let metallib_bin = Command::new("xcrun")
            .args(["-sdk", "macosx", "-f", "metallib"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|p| !p.is_empty())?;
        let mut compile = Command::new(&metal_bin);
        compile.arg("-c");
        if strict_math {
            compile.arg("-fno-fast-math");
        }
        let compile = compile
            .arg(metal_src.as_os_str())
            .arg("-o")
            .arg(air.as_os_str())
            .output()
            .ok()?;
        if !compile.status.success() {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return None;
        }
        let link = Command::new(&metallib_bin)
            .arg(air.as_os_str())
            .arg("-o")
            .arg(metallib_tmp.as_os_str())
            .output()
            .ok()?;
        if !link.status.success() {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return None;
        }
        std::fs::rename(&metallib_tmp, out_path).ok()?;
        let _ = std::fs::remove_dir_all(&tmp_dir);
        Some(())
    }

    fn load_or_compile_shader_library(device: &Device, strict_math: bool) -> Result<Library> {
        let src = super::all_shader_sources();
        let source_sha = shader_source_sha256(&src);

        if metallib_cache_enabled() {
            let path = metallib_cache_path(device, &source_sha, strict_math);
            if path.is_file() {
                let load_start = std::time::Instant::now();
                match device.new_library_with_file(&path) {
                    Ok(library) => {
                        crate::startup_timing::record_ms(
                            "metal_shader_library_load_metallib_cache_hit",
                            crate::startup_timing::duration_ms(load_start.elapsed()),
                        );
                        return Ok(library);
                    }
                    Err(_err) => {
                        // Corrupt or wrong-GPU metallib: fall through to source.
                        crate::startup_timing::record_ms(
                            "metal_shader_library_metallib_load_failed_fallback_source",
                            0,
                        );
                    }
                }
            } else if matches!(
                std::env::var("HAWKING_METALLIB_BUILD").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE")
            ) {
                let build_start = std::time::Instant::now();
                if try_build_metallib_with_xcrun(&src, &path, strict_math).is_some() {
                    crate::startup_timing::record_ms(
                        "metal_shader_library_xcrun_metallib_build",
                        crate::startup_timing::duration_ms(build_start.elapsed()),
                    );
                    if let Ok(library) = device.new_library_with_file(&path) {
                        return Ok(library);
                    }
                }
            }
        }

        let compile_start = std::time::Instant::now();
        let opts = metal::CompileOptions::new();
        if strict_math {
            opts.set_fast_math_enabled(false);
        }
        let library = device.new_library_with_source(&src, &opts).map_err(|e| {
            Error::Metal(format!(
                "{}shader compile: {e}",
                if strict_math { "strict-math " } else { "" }
            ))
        })?;
        crate::startup_timing::record_ms(
            "metal_shader_library_compile_from_source",
            crate::startup_timing::duration_ms(compile_start.elapsed()),
        );
        Ok(library)
    }

    impl MetalContext {
        pub fn new() -> Result<Self> {
            Self::new_with_trace(false)
        }

        pub fn new_with_trace(trace_dispatch: bool) -> Result<Self> {
            crate::startup_timing::time_ms_result("metal_context_new_with_trace", || {
                let device = Device::system_default()
                    .ok_or_else(|| Error::Metal("no Metal-capable GPU".into()))?;
                let queue = device.new_command_queue();
                let library = load_or_compile_shader_library(&device, /*strict_math=*/ false)?;
                // Resolve at construction so hot-path checks are a single bool load.
                let effective =
                    trace_dispatch || std::env::var_os("HAWKING_TRACE_DISPATCH").is_some();
                Ok(Self {
                    inner: Arc::new(Inner {
                        device,
                        queue,
                        library,
                        pipelines: Mutex::new(HashMap::new()),
                        icb_pipelines: Mutex::new(HashMap::new()),
                    }),
                    trace: Arc::new(DispatchTrace::new()),
                    stats: Arc::new(MetalContextStats::new()),
                    trace_dispatch: effective,
                    prod_cb_tracer_pool: Arc::new(Mutex::new(Vec::new())),
                })
            })
        }

        /// Compile a separate diagnostic-only Metal library with fast-math
        /// disabled.  The normal constructors intentionally retain their
        /// default compile options; callers must opt in explicitly and must
        /// not treat this as a runtime-wide arithmetic policy.
        pub fn new_with_trace_strict_math(trace_dispatch: bool) -> Result<Self> {
            crate::startup_timing::time_ms_result("metal_context_new_with_trace_strict_math", || {
                let device = Device::system_default()
                    .ok_or_else(|| Error::Metal("no Metal-capable GPU".into()))?;
                let queue = device.new_command_queue();
                let library = load_or_compile_shader_library(&device, /*strict_math=*/ true)?;
                let effective =
                    trace_dispatch || std::env::var_os("HAWKING_TRACE_DISPATCH").is_some();
                Ok(Self {
                    inner: Arc::new(Inner {
                        device,
                        queue,
                        library,
                        pipelines: Mutex::new(HashMap::new()),
                        icb_pipelines: Mutex::new(HashMap::new()),
                    }),
                    trace: Arc::new(DispatchTrace::new()),
                    stats: Arc::new(MetalContextStats::new()),
                    trace_dispatch: effective,
                    prod_cb_tracer_pool: Arc::new(Mutex::new(Vec::new())),
                })
            })
        }

        pub fn device(&self) -> &Device {
            &self.inner.device
        }
        pub fn queue(&self) -> &CommandQueue {
            &self.inner.queue
        }

        /// Wave-6 (`HAWKING_QWEN_RESIDENCY=1`, DEFAULT-OFF): pin
        /// `allocations` (the decode working set: weights mmap + arena +
        /// model buffers) into one process-lifetime `MTLResidencySet`
        /// attached to this context's command queue. Call ONCE, after
        /// load / on first decode. No-op (returns `Ok`) when the flag is
        /// unset -- so the golden path issues zero residency objc traffic
        /// and stays byte-for-byte unchanged.
        ///
        /// SAFETY contract is `install_residency_set`'s: each buffer must
        /// outlive the command queue (true for all hawking decode
        /// buffers -- weights are mmap-backed for the engine lifetime,
        /// arena/model buffers live on the engine).
        pub fn request_residency(&self, allocations: &[&Buffer]) -> Result<()> {
            if !crate::env_on("HAWKING_QWEN_RESIDENCY") {
                return Ok(());
            }
            if allocations.is_empty() {
                return Ok(());
            }
            // &Buffer derefs to &BufferRef (foreign_obj_type! Deref).
            let refs: Vec<&metal::BufferRef> = allocations.iter().map(|b| &***b).collect();
            unsafe { install_residency_set(&self.inner.device, &self.inner.queue, &refs) }
        }
        pub fn library(&self) -> &Library {
            &self.inner.library
        }
        pub fn device_name(&self) -> String {
            self.inner.device.name().to_string()
        }

        /// Look up -- or create + cache -- a compute pipeline for a
        /// kernel function.
        pub fn pipeline(&self, fn_name: &str) -> Result<ComputePipelineState> {
            let mut pipes = self.inner.pipelines.lock();
            if let Some(p) = pipes.get(fn_name) {
                return Ok(p.clone());
            }
            let start = std::time::Instant::now();
            let f = self
                .inner
                .library
                .get_function(fn_name, None)
                .map_err(|e| Error::Metal(format!("kernel `{fn_name}` not found: {e}")))?;
            let p = self
                .inner
                .device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| Error::Metal(format!("pipeline `{fn_name}`: {e}")))?;
            let ms = crate::startup_timing::duration_ms(start.elapsed());
            // Aggregate first-create cost; hot path hits cache above.
            crate::startup_timing::record_ms(
                format!("metal_pipeline_create:{fn_name}"),
                ms,
            );
            pipes.insert(fn_name.to_string(), p.clone());
            Ok(p)
        }

        /// Resolve an ICB-capable pipeline. Metal requires
        /// `supportIndirectCommandBuffers=true` at pipeline construction;
        /// ordinary cached pipelines cannot be retrofitted after creation.
        fn icb_pipeline(&self, fn_name: &str) -> Result<ComputePipelineState> {
            let mut pipes = self.inner.icb_pipelines.lock();
            if let Some(pipeline) = pipes.get(fn_name) {
                return Ok(pipeline.clone());
            }
            let function = self
                .inner
                .library
                .get_function(fn_name, None)
                .map_err(|e| Error::Metal(format!("kernel `{fn_name}` not found: {e}")))?;
            let descriptor = ComputePipelineDescriptor::new();
            descriptor.set_compute_function(Some(&function));
            descriptor.set_support_indirect_command_buffers(true);
            let pipeline = self
                .inner
                .device
                .new_compute_pipeline_state(&descriptor)
                .map_err(|e| {
                    Error::Metal(format!(
                        "ICB-capable pipeline `{fn_name}` could not be created: {e}"
                    ))
                })?;
            pipes.insert(fn_name.to_owned(), pipeline.clone());
            Ok(pipeline)
        }

        /// Shared (CPU+GPU readable) buffer of the given byte size.
        pub fn new_buffer(&self, len: usize) -> Buffer {
            if self.trace_dispatch {
                self.stats.buffers_created.fetch_add(1, Ordering::Relaxed);
                self.stats.bytes_allocated.fetch_add(len, Ordering::Relaxed);
            }
            self.inner
                .device
                .new_buffer(len as u64, MTLResourceOptions::StorageModeShared)
        }

        /// Buffer initialized from a CPU byte slice.
        pub fn new_buffer_with_bytes(&self, bytes: &[u8]) -> Buffer {
            if self.trace_dispatch {
                self.stats.buffers_created.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .bytes_allocated
                    .fetch_add(bytes.len(), Ordering::Relaxed);
            }
            self.inner.device.new_buffer_with_data(
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }

        /// P1-E: largest buffer this device should be asked to allocate. We
        /// CANNOT post-check a nil MTLBuffer: metal-rs 0.29's `new_buffer`
        /// asserts non-null (panics/aborts) the instant `newBufferWithLength:`
        /// returns nil, before any length check runs. So the checked allocators
        /// reject over-sized requests UP FRONT. `max_buffer_length()` is not
        /// usable (metal 0.29 hardcodes it to 1 GB on macOS≥12, which would
        /// reject valid >1 GB allocs); `recommendedMaxWorkingSetSize` is the
        /// real per-device unified-memory ceiling (falls back to 1 TB if the
        /// device reports 0). This guards the dominant OOM cause — a model too
        /// large for the device. A transient OOM *within* the ceiling would
        /// still abort inside metal-rs; a fully fallible alloc needs the
        /// objc2-metal binding (followup), not in scope here.
        fn alloc_ceiling(&self) -> u64 {
            let ws = self.inner.device.recommended_max_working_set_size();
            if ws > 0 {
                ws
            } else {
                1u64 << 40 // 1 TB fallback if the device reports no working set
            }
        }

        /// P1-E: fallible buffer allocation — returns `Err` (instead of a panic
        /// or a nil buffer that crashes a later dispatch) when `len` exceeds the
        /// device ceiling. Use at input/driver-facing load-path sites where the
        /// size is user/model-driven (embed / lm-head / large weights).
        pub fn new_buffer_checked(&self, len: usize) -> Result<Buffer> {
            let ceiling = self.alloc_ceiling();
            if len as u64 > ceiling {
                return Err(Error::Metal(format!(
                    "MTLBuffer allocation of {len} B exceeds device working-set ceiling {ceiling} B"
                )));
            }
            if self.trace_dispatch {
                self.stats.buffers_created.fetch_add(1, Ordering::Relaxed);
                self.stats.bytes_allocated.fetch_add(len, Ordering::Relaxed);
            }
            Ok(self
                .inner
                .device
                .new_buffer(len as u64, MTLResourceOptions::StorageModeShared))
        }

        /// P1-E: fallible counterpart of [`Self::new_buffer_with_bytes`] — same
        /// over-size guard as [`Self::new_buffer_checked`].
        pub fn new_buffer_with_bytes_checked(&self, bytes: &[u8]) -> Result<Buffer> {
            let ceiling = self.alloc_ceiling();
            if bytes.len() as u64 > ceiling {
                return Err(Error::Metal(format!(
                    "MTLBuffer allocation of {} B exceeds device working-set ceiling {ceiling} B",
                    bytes.len()
                )));
            }
            if self.trace_dispatch {
                self.stats.buffers_created.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .bytes_allocated
                    .fetch_add(bytes.len(), Ordering::Relaxed);
            }
            Ok(self.inner.device.new_buffer_with_data(
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            ))
        }

        /// Write `bytes` into an existing shared buffer. The buffer must
        /// have been allocated with `new_buffer` and have capacity ≥ `bytes.len()`.
        /// On unified-memory Apple Silicon this is a plain `memcpy` -- no GPU
        /// round-trip; the data is visible to subsequent GPU dispatches immediately.
        pub fn write_buffer_bytes(buf: &Buffer, bytes: &[u8]) {
            let ptr = buf.contents() as *mut u8;
            unsafe { ptr.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len()) };
        }

        /// **Zero-copy** buffer view over a borrowed mmap region.
        ///
        /// SAFETY: caller guarantees `bytes` outlives any Metal command
        /// buffer that uses the returned buffer. Used by the GGUF
        /// loader to pin tensor weights without a copy. The mmap is
        /// kept alive by the engine for its entire lifetime, so this
        /// is sound in practice.
        ///
        /// # Safety
        ///
        /// `bytes` must outlive every Metal command buffer that
        /// references the returned buffer. The GGUF mmap pins the
        /// underlying memory for the engine's lifetime, which is
        /// where this is currently called from.
        pub unsafe fn new_buffer_no_copy(&self, bytes: &[u8]) -> Buffer {
            self.inner.device.new_buffer_with_bytes_no_copy(
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }

        /// Drain all trace samples accumulated since the last drain.
        /// Returns an empty vec when trace is disabled.
        pub fn drain_trace(&self) -> Vec<super::DispatchSample> {
            self.trace.drain()
        }

        /// Drain structural counters, returning (buffers_created, bytes_allocated, commits).
        /// Returns (0, 0, 0) when trace_dispatch is false.
        pub fn drain_stats(&self) -> (usize, usize, usize) {
            self.stats.drain()
        }

        pub fn dispatch_threads(
            &self,
            fn_name: &str,
            grid: (u32, u32, u32),
            tg: (u32, u32, u32),
            encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) -> Result<()> {
            use crate::cost_ledger::{self, Bucket};

            let trace_enabled = self.trace_dispatch;
            let ledger = cost_ledger::is_recording();
            let gpu_stage = if ledger {
                Some(cost_ledger::current_gpu_stage().unwrap_or(cost_ledger::GpuStage::Untagged))
            } else {
                None
            };
            let t0 = if trace_enabled {
                Some(Instant::now())
            } else {
                None
            };

            let pipe = self.pipeline(fn_name)?;
            let t_encode = if ledger { Some(Instant::now()) } else { None };
            let cmd = self.inner.queue.new_command_buffer();
            let physical_trace = physical_command_label("command_buffer");
            if let Some((_, label)) = physical_trace.as_ref() {
                cmd.set_label(label);
            }
            let enc = cmd.new_compute_command_encoder();
            if let Some((command, _)) = physical_trace.as_ref() {
                enc.set_label(&physical_encoder_label(command, "compute_encoder", fn_name));
            } else {
                enc.set_label(fn_name);
            }
            enc.set_compute_pipeline_state(&pipe);
            encode(enc);
            enc.dispatch_threads(
                MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
                MTLSize::new(tg.0 as u64, tg.1 as u64, tg.2 as u64),
            );
            enc.end_encoding();
            if let Some(t) = t_encode {
                cost_ledger::add_duration(Bucket::MetalEncode, t.elapsed());
                cost_ledger::record_dispatches(1);
            }

            let t_submit = if ledger { Some(Instant::now()) } else { None };
            cmd.commit();
            let commit_us = t_submit
                .map(|t| {
                    let d = t.elapsed();
                    cost_ledger::add_duration(Bucket::MetalSubmit, d);
                    cost_ledger::record_command_buffer();
                    d.as_micros() as u64
                })
                .unwrap_or(0);
            if trace_enabled {
                self.stats.commits.fetch_add(1, Ordering::Relaxed);
            }

            let t_sync = if ledger { Some(Instant::now()) } else { None };
            cmd.wait_until_completed();
            let wait_us = t_sync
                .map(|t| {
                    let d = t.elapsed();
                    cost_ledger::add_duration(Bucket::MetalSynchronize, d);
                    cost_ledger::record_sync_point();
                    d.as_micros() as u64
                })
                .unwrap_or(0);
            if ledger {
                // GPU timestamps — never invent when the driver returns zeros.
                let stage_dispatches = gpu_stage.map(|stage| [(stage, 1)]);
                ledger_record_cb_gpu(
                    &cmd,
                    commit_us,
                    wait_us,
                    1,
                    stage_dispatches
                        .as_ref()
                        .map(|s| s.as_slice())
                        .unwrap_or(&[]),
                );
                ledger_probe_counter_capability(&self.inner.device);
            }

            if let Some(t0) = t0 {
                let wall_us = t0.elapsed().as_micros() as u64;
                self.trace
                    .record(static_kernel_name(fn_name), wall_us, super::current_layer());
            }
            Ok(())
        }

        /// Submit one compute dispatch and return its completed-command-buffer
        /// timing without treating a CPU wall clock as GPU execution time.
        ///
        /// This intentionally uses the same pipeline cache, queue, labels,
        /// physical-trace attribution, shared-buffer assumptions, and
        /// `dispatch_threads` geometry as [`Self::dispatch_threads`].  It is
        /// suited to bounded source-native parity probes where a missing GPU
        /// timestamp must be visible in the receipt instead of silently
        /// replaced by `wait_us`.
        pub fn dispatch_threads_timed(
            &self,
            fn_name: &str,
            grid: (u32, u32, u32),
            tg: (u32, u32, u32),
            encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) -> Result<super::MetalDispatchTiming> {
            let total_started = Instant::now();
            let pipeline_started = Instant::now();
            let pipe = self.pipeline(fn_name)?;
            let pipeline_lookup_us = pipeline_started.elapsed().as_micros() as u64;

            let encode_started = Instant::now();
            let cmd = self.inner.queue.new_command_buffer();
            let physical_trace = physical_command_label("command_buffer");
            if let Some((_, label)) = physical_trace.as_ref() {
                cmd.set_label(label);
            }
            let enc = cmd.new_compute_command_encoder();
            if let Some((command, _)) = physical_trace.as_ref() {
                enc.set_label(&physical_encoder_label(command, "compute_encoder", fn_name));
            } else {
                enc.set_label(fn_name);
            }
            enc.set_compute_pipeline_state(&pipe);
            encode(enc);
            enc.dispatch_threads(
                MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
                MTLSize::new(tg.0 as u64, tg.1 as u64, tg.2 as u64),
            );
            enc.end_encoding();
            let encode_us = encode_started.elapsed().as_micros() as u64;

            let submit_started = Instant::now();
            cmd.commit();
            let submit_us = submit_started.elapsed().as_micros() as u64;
            if self.trace_dispatch {
                self.stats.commits.fetch_add(1, Ordering::Relaxed);
            }

            let wait_started = Instant::now();
            cmd.wait_until_completed();
            let wait_us = wait_started.elapsed().as_micros() as u64;
            let (gpu_start_s, gpu_end_s) = unsafe { cb_gpu_start_end_s(&cmd) };
            let (gpu_duration_us, gpu_start_ns, gpu_end_ns) = match (gpu_start_s, gpu_end_s) {
                (Some(start), Some(end)) if end > start => (
                    Some(((end - start) * 1_000_000.0) as u64),
                    Some((start * 1_000_000_000.0) as u64),
                    Some((end * 1_000_000_000.0) as u64),
                ),
                _ => (None, None, None),
            };
            let host_wall_us = total_started.elapsed().as_micros() as u64;

            if self.trace_dispatch {
                self.trace.samples.lock().push(super::DispatchSample {
                    kernel_name: static_kernel_name(fn_name),
                    wall_us: host_wall_us,
                    layer_hint: super::current_layer(),
                    gpu_us: gpu_duration_us,
                    gpu_start_ns,
                    gpu_end_ns,
                });
            }

            Ok(super::MetalDispatchTiming {
                pipeline_lookup_us,
                encode_us,
                submit_us,
                wait_us,
                host_wall_us,
                gpu_duration_us,
                gpu_start_ns,
                gpu_end_ns,
                command_buffers: 1,
                compute_encoders: 1,
                compute_dispatches: 1,
            })
        }

        pub fn dispatch_batch(
            &self,
            encode: impl FnOnce(&mut CommandBatch<'_>) -> Result<()>,
        ) -> Result<()> {
            let trace_enabled = self.trace_dispatch;
            let t0 = if trace_enabled {
                Some(Instant::now())
            } else {
                None
            };

            let cmd = self.inner.queue.new_command_buffer();
            let physical_trace = physical_command_label("command_buffer");
            if let Some((_, label)) = physical_trace.as_ref() {
                cmd.set_label(label);
            }
            let mut batch = CommandBatch {
                ctx: self,
                cmd,
                physical_trace: physical_trace.map(|(identity, _)| identity),
                concurrent_encoder: None,
                pipeline_lookup_us: 0,
                compute_encoders: 0,
                compute_dispatches: 0,
            };
            encode(&mut batch)?;
            if batch.concurrent_encoder.is_some() {
                return Err(Error::Metal(
                    "dispatch_batch returned with an unclosed concurrent group".into(),
                ));
            }
            let CommandBatch { cmd, .. } = batch;
            cmd.commit();
            if trace_enabled {
                self.stats.commits.fetch_add(1, Ordering::Relaxed);
            }
            cmd.wait_until_completed();

            if let Some(t0) = t0 {
                let wall_us = t0.elapsed().as_micros() as u64;
                // dispatch_batch is used for mla_decode_and_o_proj_metal
                self.trace
                    .record("dispatch_batch", wall_us, super::current_layer());
            }
            Ok(())
        }

        /// Encode one or more dependent GPU kernels into one command buffer
        /// and return completed-command-buffer GPU timestamps plus exact
        /// command topology.  This is intentionally a probe-only surface:
        /// it does not expose per-encoder timestamps or claim that a batch is
        /// a production decode graph.
        pub fn dispatch_batch_timed(
            &self,
            encode: impl FnOnce(&mut CommandBatch<'_>) -> Result<()>,
        ) -> Result<super::MetalBatchTiming> {
            let total_started = Instant::now();
            let encode_started = Instant::now();
            let cmd = self.inner.queue.new_command_buffer();
            let physical_trace = physical_command_label("command_buffer");
            if let Some((_, label)) = physical_trace.as_ref() {
                cmd.set_label(label);
            }
            let mut batch = CommandBatch {
                ctx: self,
                cmd,
                physical_trace: physical_trace.map(|(identity, _)| identity),
                concurrent_encoder: None,
                pipeline_lookup_us: 0,
                compute_encoders: 0,
                compute_dispatches: 0,
            };
            encode(&mut batch)?;
            if batch.concurrent_encoder.is_some() {
                return Err(Error::Metal(
                    "dispatch_batch_timed returned with an unclosed concurrent group".into(),
                ));
            }
            let encode_us = encode_started.elapsed().as_micros() as u64;
            let CommandBatch {
                cmd,
                pipeline_lookup_us,
                compute_encoders,
                compute_dispatches,
                ..
            } = batch;

            let submit_started = Instant::now();
            cmd.commit();
            let submit_us = submit_started.elapsed().as_micros() as u64;
            if self.trace_dispatch {
                self.stats.commits.fetch_add(1, Ordering::Relaxed);
            }
            let wait_started = Instant::now();
            cmd.wait_until_completed();
            let wait_us = wait_started.elapsed().as_micros() as u64;
            let (gpu_start_s, gpu_end_s) = unsafe { cb_gpu_start_end_s(&cmd) };
            let (gpu_duration_us, gpu_start_ns, gpu_end_ns) = match (gpu_start_s, gpu_end_s) {
                (Some(start), Some(end)) if end > start => (
                    Some(((end - start) * 1_000_000.0) as u64),
                    Some((start * 1_000_000_000.0) as u64),
                    Some((end * 1_000_000_000.0) as u64),
                ),
                _ => (None, None, None),
            };
            let host_wall_us = total_started.elapsed().as_micros() as u64;
            if self.trace_dispatch {
                self.trace.samples.lock().push(super::DispatchSample {
                    kernel_name: "dispatch_batch",
                    wall_us: host_wall_us,
                    layer_hint: super::current_layer(),
                    gpu_us: gpu_duration_us,
                    gpu_start_ns,
                    gpu_end_ns,
                });
            }
            Ok(super::MetalBatchTiming {
                pipeline_lookup_us,
                encode_us,
                submit_us,
                wait_us,
                host_wall_us,
                gpu_duration_us,
                gpu_start_ns,
                gpu_end_ns,
                command_buffers: 1,
                compute_encoders,
                compute_dispatches,
            })
        }
    }

    impl CommandBatch<'_> {
        /// Begin one explicit `MTLDispatchTypeConcurrent` wave.
        ///
        /// This is a narrow probe surface for independently writable work
        /// items. The caller is responsible for proving that every dispatch
        /// in the wave has disjoint writes and no intra-wave producer/
        /// consumer dependency. `end_concurrent_group` is mandatory before
        /// returning from the batch closure or issuing an ordered dispatch.
        pub fn begin_concurrent_group(&mut self) -> Result<()> {
            if self.concurrent_encoder.is_some() {
                return Err(Error::Metal(
                    "begin_concurrent_group called while a group is already active".into(),
                ));
            }
            let enc = self
                .cmd
                .compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
            if let Some(command) = self.physical_trace.as_ref() {
                enc.set_label(&physical_encoder_label(
                    command,
                    "compute_encoder",
                    "concurrent_group",
                ));
            } else {
                enc.set_label("concurrent_group");
            }
            self.concurrent_encoder = Some(enc.to_owned());
            Ok(())
        }

        /// Encode one caller-proved independent dispatch into the active
        /// concurrent group. It consumes no additional command encoder.
        pub fn dispatch_threads_in_concurrent_group(
            &mut self,
            fn_name: &str,
            grid: (u32, u32, u32),
            tg: (u32, u32, u32),
            encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) -> Result<()> {
            let pipeline_started = Instant::now();
            let pipe = self.ctx.pipeline(fn_name)?;
            self.pipeline_lookup_us += pipeline_started.elapsed().as_micros() as u64;
            let enc = self.concurrent_encoder.as_ref().ok_or_else(|| {
                Error::Metal(
                    "dispatch_threads_in_concurrent_group called without an active group".into(),
                )
            })?;
            enc.set_compute_pipeline_state(&pipe);
            encode(enc);
            enc.dispatch_threads(
                MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
                MTLSize::new(tg.0 as u64, tg.1 as u64, tg.2 as u64),
            );
            self.compute_dispatches += 1;
            Ok(())
        }

        /// Close the active concurrent wave. Closing is explicit so the next
        /// command encoder establishes the required dependency boundary.
        pub fn end_concurrent_group(&mut self) -> Result<()> {
            let enc = self.concurrent_encoder.take().ok_or_else(|| {
                Error::Metal("end_concurrent_group called with no active group".into())
            })?;
            enc.end_encoding();
            self.compute_encoders += 1;
            Ok(())
        }

        pub fn dispatch_threads(
            &mut self,
            fn_name: &str,
            grid: (u32, u32, u32),
            tg: (u32, u32, u32),
            encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) -> Result<()> {
            if self.concurrent_encoder.is_some() {
                return Err(Error::Metal(
                    "dispatch_threads cannot run while a concurrent group is active".into(),
                ));
            }
            let pipeline_started = Instant::now();
            let pipe = self.ctx.pipeline(fn_name)?;
            self.pipeline_lookup_us += pipeline_started.elapsed().as_micros() as u64;
            let enc = self.cmd.new_compute_command_encoder();
            if let Some(command) = self.physical_trace.as_ref() {
                enc.set_label(&physical_encoder_label(command, "compute_encoder", fn_name));
            } else {
                enc.set_label(fn_name);
            }
            enc.set_compute_pipeline_state(&pipe);
            encode(enc);
            enc.dispatch_threads(
                MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
                MTLSize::new(tg.0 as u64, tg.1 as u64, tg.2 as u64),
            );
            enc.end_encoding();
            self.compute_encoders += 1;
            self.compute_dispatches += 1;
            Ok(())
        }

        /// Encode two dependency-ordered compute dispatches into one compute
        /// encoder. A resource-scoped Metal memory barrier makes writes from
        /// the first dispatch visible to the second without adding an encoder
        /// or command-buffer boundary. This is a diagnostic building block;
        /// callers must still prove output parity for their particular chain.
        #[allow(clippy::too_many_arguments)]
        pub fn dispatch_threads_pair_in_one_encoder(
            &mut self,
            first_name: &str,
            first_grid: (u32, u32, u32),
            first_tg: (u32, u32, u32),
            first_encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
            barrier_resources: &[&metal::ResourceRef],
            second_name: &str,
            second_grid: (u32, u32, u32),
            second_tg: (u32, u32, u32),
            second_encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) -> Result<()> {
            if self.concurrent_encoder.is_some() {
                return Err(Error::Metal(
                    "dispatch_threads_pair_in_one_encoder cannot run while a concurrent group is active".into(),
                ));
            }
            let first_pipeline_started = Instant::now();
            let first_pipe = self.ctx.pipeline(first_name)?;
            self.pipeline_lookup_us += first_pipeline_started.elapsed().as_micros() as u64;
            let second_pipeline_started = Instant::now();
            let second_pipe = self.ctx.pipeline(second_name)?;
            self.pipeline_lookup_us += second_pipeline_started.elapsed().as_micros() as u64;
            let enc = self.cmd.new_compute_command_encoder();
            if let Some(command) = self.physical_trace.as_ref() {
                enc.set_label(&physical_encoder_label(
                    command,
                    "compute_encoder",
                    "ordered_pair_with_resource_barrier",
                ));
            } else {
                enc.set_label("ordered_pair_with_resource_barrier");
            }
            enc.set_compute_pipeline_state(&first_pipe);
            first_encode(enc);
            enc.dispatch_threads(
                MTLSize::new(
                    first_grid.0 as u64,
                    first_grid.1 as u64,
                    first_grid.2 as u64,
                ),
                MTLSize::new(first_tg.0 as u64, first_tg.1 as u64, first_tg.2 as u64),
            );
            enc.memory_barrier_with_resources(barrier_resources);
            enc.set_compute_pipeline_state(&second_pipe);
            second_encode(enc);
            enc.dispatch_threads(
                MTLSize::new(
                    second_grid.0 as u64,
                    second_grid.1 as u64,
                    second_grid.2 as u64,
                ),
                MTLSize::new(second_tg.0 as u64, second_tg.1 as u64, second_tg.2 as u64),
            );
            enc.end_encoding();
            self.compute_encoders += 1;
            self.compute_dispatches += 2;
            Ok(())
        }
    }

    // ── Temporal Gravity replay substrate (default-off) ─────────────────────

    const MAX_REPLAY_KERNEL_BUFFERS: usize = 31;

    /// One persistent `MTLBuffer` binding captured by a replayable compute
    /// graph. Scalar arguments must also live in buffers (for example a
    /// [`KernelArgBuffer`](super::KernelArgBuffer)); ICB commands cannot
    /// capture `set_bytes` payloads.
    #[derive(Clone)]
    pub struct ReplayBufferBinding {
        index: usize,
        buffer: Buffer,
        offset: usize,
        usage: MTLResourceUsage,
    }

    impl ReplayBufferBinding {
        pub fn read(index: usize, buffer: &Buffer, offset: usize) -> Self {
            Self {
                index,
                buffer: buffer.clone(),
                offset,
                usage: MTLResourceUsage::Read,
            }
        }

        pub fn write(index: usize, buffer: &Buffer, offset: usize) -> Self {
            Self {
                index,
                buffer: buffer.clone(),
                offset,
                usage: MTLResourceUsage::Write,
            }
        }

        pub fn read_write(index: usize, buffer: &Buffer, offset: usize) -> Self {
            Self {
                index,
                buffer: buffer.clone(),
                offset,
                usage: MTLResourceUsage::Read | MTLResourceUsage::Write,
            }
        }
    }

    /// A captured resource referenced indirectly (for example by a GPU address
    /// stored in a descriptor table) rather than through a kernel buffer slot.
    #[derive(Clone)]
    pub struct ReplayResourceDeclaration {
        buffer: Buffer,
        usage: MTLResourceUsage,
    }

    impl ReplayResourceDeclaration {
        pub fn read(buffer: &Buffer) -> Self {
            Self {
                buffer: buffer.clone(),
                usage: MTLResourceUsage::Read,
            }
        }

        pub fn write(buffer: &Buffer) -> Self {
            Self {
                buffer: buffer.clone(),
                usage: MTLResourceUsage::Write,
            }
        }

        pub fn read_write(buffer: &Buffer) -> Self {
            Self {
                buffer: buffer.clone(),
                usage: MTLResourceUsage::Read | MTLResourceUsage::Write,
            }
        }
    }

    /// One pre-encoded compute dispatch in a [`ReplayableComputeGraph`].
    ///
    /// `barrier_before` makes this command wait for all preceding commands in
    /// the ICB. It is required when a prior stage writes a buffer this stage
    /// reads or writes. Independent stages may leave it false.
    pub struct ReplayComputeStage {
        kernel: String,
        grid: (u32, u32, u32),
        threadgroup: (u32, u32, u32),
        bindings: Vec<ReplayBufferBinding>,
        threadgroup_memory: Vec<(usize, usize)>,
        ledger_stage: Option<crate::cost_ledger::GpuStage>,
        barrier_before: bool,
    }

    impl ReplayComputeStage {
        pub fn new(
            kernel: impl Into<String>,
            grid: (u32, u32, u32),
            threadgroup: (u32, u32, u32),
            bindings: Vec<ReplayBufferBinding>,
        ) -> Self {
            Self {
                kernel: kernel.into(),
                grid,
                threadgroup,
                bindings,
                threadgroup_memory: Vec::new(),
                ledger_stage: None,
                barrier_before: false,
            }
        }

        pub fn with_threadgroup_memory_length(mut self, index: usize, length: usize) -> Self {
            self.threadgroup_memory.push((index, length));
            self
        }

        pub fn with_ledger_stage(mut self, stage: crate::cost_ledger::GpuStage) -> Self {
            self.ledger_stage = Some(stage);
            self
        }

        pub fn with_barrier_before(mut self) -> Self {
            self.barrier_before = true;
            self
        }
    }

    struct ReplayResource {
        buffer: Buffer,
        usage: MTLResourceUsage,
    }

    /// An immutable sequence of Metal compute commands encoded once into an
    /// indirect command buffer and replayed against stable buffer addresses.
    ///
    /// This is intentionally not wired into decode selection yet. It is a
    /// correctness/replayability substrate, not a throughput claim: Hawking's
    /// measured CPU encoding share remains below the ICB ship gate.
    pub struct ReplayableComputeGraph {
        context: Arc<Inner>,
        icb: IndirectCommandBuffer,
        resources: Vec<ReplayResource>,
        // Retain pipelines explicitly for the lifetime of the encoded ICB.
        _pipelines: Vec<ComputePipelineState>,
        command_count: usize,
        explicit_ledger_stages: bool,
        ledger_stage_dispatches: [u64; crate::cost_ledger::GpuStage::ALL.len()],
    }

    impl ReplayableComputeGraph {
        /// Validate and pre-encode `stages`. All validation and pipeline
        /// resolution happens before the ICB is allocated, so invalid graphs
        /// fail closed without creating a partially encoded replay object.
        pub fn new(ctx: &MetalContext, stages: Vec<ReplayComputeStage>) -> Result<Self> {
            Self::new_with_resources(ctx, stages, Vec::new())
        }

        /// As [`Self::new`], plus resources reached through captured GPU
        /// addresses rather than direct kernel buffer bindings.
        pub fn new_with_resources(
            ctx: &MetalContext,
            stages: Vec<ReplayComputeStage>,
            indirect_resources: Vec<ReplayResourceDeclaration>,
        ) -> Result<Self> {
            if stages.is_empty() {
                return Err(Error::Metal(
                    "replayable compute graph requires at least one stage".into(),
                ));
            }

            let mut pipelines = Vec::new();
            let mut pipeline_slots = HashMap::<String, usize>::new();
            let mut stage_pipeline_slots = Vec::with_capacity(stages.len());
            let mut max_bind_count = 0usize;
            let mut tagged_stage_count = 0usize;
            let mut ledger_stage_dispatches = [0u64; crate::cost_ledger::GpuStage::ALL.len()];
            for (stage_index, stage) in stages.iter().enumerate() {
                if stage.kernel.is_empty() {
                    return Err(Error::Metal(format!(
                        "replayable compute stage {stage_index} has an empty kernel name"
                    )));
                }
                if [stage.grid.0, stage.grid.1, stage.grid.2]
                    .into_iter()
                    .any(|dim| dim == 0)
                {
                    return Err(Error::Metal(format!(
                        "replayable compute stage {} (`{}`) has a zero grid dimension",
                        stage_index, stage.kernel
                    )));
                }
                if [
                    stage.threadgroup.0,
                    stage.threadgroup.1,
                    stage.threadgroup.2,
                ]
                .into_iter()
                .any(|dim| dim == 0)
                {
                    return Err(Error::Metal(format!(
                        "replayable compute stage {} (`{}`) has a zero threadgroup dimension",
                        stage_index, stage.kernel
                    )));
                }
                if stage.bindings.is_empty() {
                    return Err(Error::Metal(format!(
                        "replayable compute stage {} (`{}`) has no persistent buffer bindings",
                        stage_index, stage.kernel
                    )));
                }

                let mut occupied = [false; MAX_REPLAY_KERNEL_BUFFERS];
                for binding in &stage.bindings {
                    if binding.index >= MAX_REPLAY_KERNEL_BUFFERS {
                        return Err(Error::Metal(format!(
                            "replayable compute stage {} (`{}`) buffer index {} exceeds Metal limit {}",
                            stage_index,
                            stage.kernel,
                            binding.index,
                            MAX_REPLAY_KERNEL_BUFFERS - 1
                        )));
                    }
                    if occupied[binding.index] {
                        return Err(Error::Metal(format!(
                            "replayable compute stage {} (`{}`) binds buffer index {} twice",
                            stage_index, stage.kernel, binding.index
                        )));
                    }
                    occupied[binding.index] = true;
                    if binding.buffer.length() == 0
                        || binding.offset as u64 >= binding.buffer.length()
                    {
                        return Err(Error::Metal(format!(
                            "replayable compute stage {} (`{}`) buffer index {} offset {} is outside {} bytes",
                            stage_index,
                            stage.kernel,
                            binding.index,
                            binding.offset,
                            binding.buffer.length()
                        )));
                    }
                    max_bind_count = max_bind_count.max(binding.index + 1);
                }
                let mut occupied_threadgroup = [false; MAX_REPLAY_KERNEL_BUFFERS];
                for &(index, length) in &stage.threadgroup_memory {
                    if index >= MAX_REPLAY_KERNEL_BUFFERS || length == 0 {
                        return Err(Error::Metal(format!(
                            "replayable compute stage {} (`{}`) has invalid threadgroup memory binding {index}/{length}",
                            stage_index, stage.kernel
                        )));
                    }
                    if occupied_threadgroup[index] {
                        return Err(Error::Metal(format!(
                            "replayable compute stage {} (`{}`) binds threadgroup memory index {index} twice",
                            stage_index, stage.kernel
                        )));
                    }
                    occupied_threadgroup[index] = true;
                }
                if let Some(ledger_stage) = stage.ledger_stage {
                    tagged_stage_count = tagged_stage_count.saturating_add(1);
                    ledger_stage_dispatches[ledger_stage.index()] =
                        ledger_stage_dispatches[ledger_stage.index()].saturating_add(1);
                }

                let pipeline_slot = if let Some(&slot) = pipeline_slots.get(&stage.kernel) {
                    slot
                } else {
                    let pipeline = ctx.icb_pipeline(&stage.kernel)?;
                    let slot = pipelines.len();
                    pipeline_slots.insert(stage.kernel.clone(), slot);
                    pipelines.push(pipeline);
                    slot
                };
                let pipeline = &pipelines[pipeline_slot];
                let threads = (stage.threadgroup.0 as usize)
                    .checked_mul(stage.threadgroup.1 as usize)
                    .and_then(|n| n.checked_mul(stage.threadgroup.2 as usize))
                    .ok_or_else(|| {
                        Error::Metal(format!(
                            "replayable compute stage {} (`{}`) threadgroup size overflows",
                            stage_index, stage.kernel
                        ))
                    })?;
                if threads > pipeline.max_total_threads_per_threadgroup() as usize {
                    return Err(Error::Metal(format!(
                        "replayable compute stage {} (`{}`) requests {} threads per group; pipeline maximum is {}",
                        stage_index,
                        stage.kernel,
                        threads,
                        pipeline.max_total_threads_per_threadgroup()
                    )));
                }
                stage_pipeline_slots.push(pipeline_slot);
            }
            if tagged_stage_count != 0 && tagged_stage_count != stages.len() {
                return Err(Error::Metal(format!(
                    "replayable compute graph must ledger-tag every stage or none: tagged {tagged_stage_count}/{}",
                    stages.len()
                )));
            }

            let descriptor = IndirectCommandBufferDescriptor::new();
            descriptor.set_command_types(MTLIndirectCommandType::ConcurrentDispatchThreads);
            descriptor.set_inherit_buffers(false);
            descriptor.set_inherit_pipeline_state(false);
            descriptor.set_max_kernel_buffer_bind_count(max_bind_count as u64);
            let icb = ctx
                .inner
                .device
                .new_indirect_command_buffer_with_descriptor(
                    &descriptor,
                    stages.len() as u64,
                    MTLResourceOptions::StorageModePrivate,
                );
            icb.set_label("hawking.replayable_compute_graph");

            let mut resources: Vec<ReplayResource> = Vec::new();
            let mut resource_slots = HashMap::<u64, usize>::new();
            let declare_resource =
                |buffer: &Buffer,
                 usage: MTLResourceUsage,
                 resources: &mut Vec<ReplayResource>,
                 resource_slots: &mut HashMap<u64, usize>| {
                    let address = buffer.gpu_address();
                    if let Some(&resource_index) = resource_slots.get(&address) {
                        resources[resource_index].usage |= usage;
                    } else {
                        resource_slots.insert(address, resources.len());
                        resources.push(ReplayResource {
                            buffer: buffer.clone(),
                            usage,
                        });
                    }
                };
            for declaration in &indirect_resources {
                declare_resource(
                    &declaration.buffer,
                    declaration.usage,
                    &mut resources,
                    &mut resource_slots,
                );
            }
            for (command_index, stage) in stages.iter().enumerate() {
                let pipeline = &pipelines[stage_pipeline_slots[command_index]];
                let command = icb.indirect_compute_command_at_index(command_index as u64);
                command.set_compute_pipeline_state(pipeline);
                for binding in &stage.bindings {
                    command.set_kernel_buffer(
                        binding.index as u64,
                        Some(&binding.buffer),
                        binding.offset as u64,
                    );
                    declare_resource(
                        &binding.buffer,
                        binding.usage,
                        &mut resources,
                        &mut resource_slots,
                    );
                }
                for &(index, length) in &stage.threadgroup_memory {
                    command.set_threadgroup_memory_length(index as u64, length as u64);
                }
                if stage.barrier_before {
                    command.set_barrier();
                }
                command.concurrent_dispatch_threads(
                    MTLSize::new(
                        stage.grid.0 as u64,
                        stage.grid.1 as u64,
                        stage.grid.2 as u64,
                    ),
                    MTLSize::new(
                        stage.threadgroup.0 as u64,
                        stage.threadgroup.1 as u64,
                        stage.threadgroup.2 as u64,
                    ),
                );
            }

            Ok(Self {
                context: ctx.inner.clone(),
                icb,
                resources,
                _pipelines: pipelines,
                command_count: stages.len(),
                explicit_ledger_stages: tagged_stage_count == stages.len(),
                ledger_stage_dispatches,
            })
        }

        pub fn command_count(&self) -> usize {
            self.command_count
        }
    }

    #[cfg(test)]
    mod replayable_compute_graph_tests {
        use super::*;
        fn write_f32(buffer: &Buffer, values: &[f32]) {
            assert!(buffer.length() >= std::mem::size_of_val(values) as u64);
            unsafe {
                (buffer.contents() as *mut f32)
                    .copy_from_nonoverlapping(values.as_ptr(), values.len());
            }
        }
        fn read_f32(buffer: &Buffer, count: usize) -> Vec<f32> {
            assert!(buffer.length() >= (count * std::mem::size_of::<f32>()) as u64);
            unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, count).to_vec() }
        }
        fn zero_replay_stages(
            output: &Buffer,
            n_buffer: &Buffer,
            n: u32,
            count: usize,
        ) -> Vec<ReplayComputeStage> {
            (0..count)
                .map(|index| {
                    let stage = ReplayComputeStage::new(
                        "gravity_zero_f32",
                        (n, 1, 1),
                        (64, 1, 1),
                        vec![
                            ReplayBufferBinding::write(0, output, 0),
                            ReplayBufferBinding::read(1, n_buffer, 0),
                        ],
                    );
                    if index == 0 {
                        stage
                    } else {
                        stage.with_barrier_before()
                    }
                })
                .collect()
        }
        fn percentile_ns(samples: &[u128], percentile: usize) -> u128 {
            let mut sorted = samples.to_vec();
            sorted.sort_unstable();
            let rank = percentile
                .saturating_mul(sorted.len().saturating_sub(1))
                .saturating_add(99)
                / 100;
            sorted[rank.min(sorted.len().saturating_sub(1))]
        }
        #[test]
        #[ignore = "requires a Metal device with compute indirect-command-buffer support"]
        fn replayable_icb_reuses_addresses_and_one_submit() {
            let ctx = MetalContext::new_with_trace(true).unwrap();
            let n = 257u32;
            let bytes = n as usize * std::mem::size_of::<f32>();
            let output = ctx.new_buffer(bytes);
            let input = ctx.new_buffer(bytes);
            let n_buffer = ctx.new_buffer_with_bytes(&n.to_ne_bytes());
            let invalid = ReplayableComputeGraph::new(
                &ctx,
                vec![ReplayComputeStage::new(
                    "gravity_zero_f32",
                    (0, 1, 1),
                    (64, 1, 1),
                    vec![
                        ReplayBufferBinding::write(0, &output, 0),
                        ReplayBufferBinding::read(1, &n_buffer, 0),
                    ],
                )],
            );
            let invalid_error = match invalid {
                Ok(_) => panic!("zero grid dimension unexpectedly produced a replay graph"),
                Err(error) => error,
            };
            assert!(invalid_error.to_string().contains("zero grid dimension"));
            let output_address = output.gpu_address();
            let input_address = input.gpu_address();
            let n_address = n_buffer.gpu_address();
            let graph = ReplayableComputeGraph::new(
                &ctx,
                vec![
                    ReplayComputeStage::new(
                        "gravity_zero_f32",
                        (n, 1, 1),
                        (64, 1, 1),
                        vec![
                            ReplayBufferBinding::write(0, &output, 0),
                            ReplayBufferBinding::read(1, &n_buffer, 0),
                        ],
                    )
                    .with_ledger_stage(crate::cost_ledger::GpuStage::KvAndNorm),
                    ReplayComputeStage::new(
                        "gravity_add_inplace_f32",
                        (n, 1, 1),
                        (64, 1, 1),
                        vec![
                            ReplayBufferBinding::read_write(0, &output, 0),
                            ReplayBufferBinding::read(1, &input, 0),
                            ReplayBufferBinding::read(2, &n_buffer, 0),
                        ],
                    )
                    .with_ledger_stage(crate::cost_ledger::GpuStage::FinalHead)
                    .with_barrier_before(),
                ],
            )
            .unwrap();
            assert_eq!(graph.command_count(), 2);
            assert!(graph.explicit_ledger_stages);
            assert_eq!(
                graph.ledger_stage_dispatches[crate::cost_ledger::GpuStage::KvAndNorm.index()],
                1
            );
            assert_eq!(
                graph.ledger_stage_dispatches[crate::cost_ledger::GpuStage::FinalHead.index()],
                1
            );
            assert_eq!(output.gpu_address(), output_address);
            assert_eq!(input.gpu_address(), input_address);
            assert_eq!(n_buffer.gpu_address(), n_address);
            let _ = ctx.drain_stats();
            let first: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 - 17.0).collect();
            write_f32(&input, &first);
            write_f32(&output, &vec![f32::NAN; n as usize]);
            let identity = PhysicalTraceIdentity::new(
                "a".repeat(64),
                "b".repeat(64),
                "icb".into(),
                "replay".into(),
                None,
                0,
            )
            .unwrap();
            let guard = PhysicalTraceGuard::begin(identity).unwrap();
            let mut tcb = TokenCommandBuffer::new(&ctx);
            tcb.execute_replayable_graph(&graph).unwrap();
            assert_eq!(tcb.dispatch_count(), 2);
            tcb.commit_and_wait().unwrap();
            assert_eq!(
                guard.counts(),
                PhysicalTraceCounts {
                    command_count: 1,
                    encoder_count: 1,
                }
            );
            drop(guard);
            assert_eq!(read_f32(&output, n as usize), first);
            let second: Vec<f32> = (0..n).map(|i| 100.0 - i as f32 * 0.5).collect();
            write_f32(&input, &second);
            write_f32(&output, &vec![-1234.0; n as usize]);
            let mut tcb = TokenCommandBuffer::new(&ctx);
            tcb.execute_replayable_graph(&graph).unwrap();
            assert_eq!(tcb.dispatch_count(), 2);
            tcb.commit_and_wait().unwrap();
            assert_eq!(read_f32(&output, n as usize), second);
            // The graph itself allocates no hot buffers; the two complete
            // replays above each submit exactly one token command buffer.
            assert_eq!(ctx.drain_stats(), (0, 0, 2));
            assert_eq!(output.gpu_address(), output_address);
            assert_eq!(input.gpu_address(), input_address);
            assert_eq!(n_buffer.gpu_address(), n_address);
        }

        #[test]
        #[ignore = "requires a Metal device with compute indirect-command-buffer support"]
        fn replayable_icb_binds_scalar_buffers_for_b9430_rmsnorm() {
            let ctx = MetalContext::new().unwrap();
            let x = ctx.new_buffer_with_bytes(bytemuck::cast_slice(&[1.0_f32; 8]));
            let weight = ctx.new_buffer_with_bytes(bytemuck::cast_slice(&[1.0_f32; 8]));
            let out = ctx.new_buffer(8 * std::mem::size_of::<f32>());
            let hidden = ctx.new_buffer_with_bytes(&8_u32.to_ne_bytes());
            let eps = ctx.new_buffer_with_bytes(&1e-5_f32.to_ne_bytes());
            let graph = ReplayableComputeGraph::new(
                &ctx,
                vec![ReplayComputeStage::new(
                    "rmsnorm_llama_b9430",
                    (1024, 1, 1),
                    (1024, 1, 1),
                    vec![
                        ReplayBufferBinding::read(0, &x, 0),
                        ReplayBufferBinding::read(1, &weight, 0),
                        ReplayBufferBinding::write(2, &out, 0),
                        ReplayBufferBinding::read(3, &hidden, 0),
                        ReplayBufferBinding::read(4, &eps, 0),
                    ],
                )
                .with_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>())],
            )
            .unwrap();
            let mut tcb = TokenCommandBuffer::new(&ctx);
            tcb.execute_replayable_graph(&graph).unwrap();
            tcb.commit_and_wait().unwrap();
            let values = read_f32(&out, 8);
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(values.iter().all(|value| (*value - 0.999_995).abs() < 1e-5));
        }

        #[test]
        fn replayable_icb_group_orders_cross_graph_dependencies() {
            const N: u32 = 257;
            let Ok(ctx) = MetalContext::new_with_trace(true) else {
                return;
            };
            let output = ctx.new_buffer(N as usize * std::mem::size_of::<f32>());
            let input: Vec<f32> = (0..N).map(|index| index as f32 * 0.25 - 9.0).collect();
            let input_buffer = ctx.new_buffer_with_bytes(bytemuck::cast_slice(&input));
            let n_buffer = ctx.new_buffer_with_bytes(&N.to_ne_bytes());
            let zero_graph =
                ReplayableComputeGraph::new(&ctx, zero_replay_stages(&output, &n_buffer, N, 1))
                    .unwrap();
            let add_graph = ReplayableComputeGraph::new(
                &ctx,
                vec![ReplayComputeStage::new(
                    "gravity_add_inplace_f32",
                    (N, 1, 1),
                    (64, 1, 1),
                    vec![
                        ReplayBufferBinding::read_write(0, &output, 0),
                        ReplayBufferBinding::read(1, &input_buffer, 0),
                        ReplayBufferBinding::read(2, &n_buffer, 0),
                    ],
                )],
            )
            .unwrap();
            write_f32(&output, &vec![f32::NAN; N as usize]);
            let mut tcb = TokenCommandBuffer::new(&ctx);
            tcb.execute_replayable_graphs(&[&zero_graph, &add_graph])
                .unwrap();
            assert_eq!(tcb.dispatch_count(), 2);
            tcb.commit_and_wait().unwrap();
            assert_eq!(read_f32(&output, N as usize), input);
        }
        #[test]
        #[ignore = "explicit bounded Metal ICB host-encoding benchmark"]
        fn replayable_icb_fifteen_command_fusion_encode_benchmark() {
            use std::time::Instant;
            const N: u32 = 257;
            const SAMPLES: usize = 257;
            let ctx = MetalContext::new_with_trace(true).unwrap();
            let output = ctx.new_buffer(N as usize * std::mem::size_of::<f32>());
            let n_buffer = ctx.new_buffer_with_bytes(&N.to_ne_bytes());
            let split_9 =
                ReplayableComputeGraph::new(&ctx, zero_replay_stages(&output, &n_buffer, N, 9))
                    .unwrap();
            let split_6 =
                ReplayableComputeGraph::new(&ctx, zero_replay_stages(&output, &n_buffer, N, 6))
                    .unwrap();
            let fused_15 =
                ReplayableComputeGraph::new(&ctx, zero_replay_stages(&output, &n_buffer, N, 15))
                    .unwrap();
            let identity = |run: &str| {
                PhysicalTraceIdentity::new(
                    "a".repeat(64),
                    "b".repeat(64),
                    "icb".into(),
                    run.into(),
                    None,
                    0,
                )
                .unwrap()
            };
            let split_guard = PhysicalTraceGuard::begin(identity("split")).unwrap();
            let mut split_tcb = TokenCommandBuffer::new(&ctx);
            split_tcb.execute_replayable_graph(&split_9).unwrap();
            split_tcb.execute_replayable_graph(&split_6).unwrap();
            assert_eq!(split_tcb.dispatch_count(), 15);
            split_tcb.commit_and_wait().unwrap();
            assert_eq!(
                split_guard.counts(),
                PhysicalTraceCounts {
                    command_count: 1,
                    encoder_count: 2,
                }
            );
            drop(split_guard);
            let fused_guard = PhysicalTraceGuard::begin(identity("fused")).unwrap();
            let mut fused_tcb = TokenCommandBuffer::new(&ctx);
            fused_tcb.execute_replayable_graph(&fused_15).unwrap();
            assert_eq!(fused_tcb.dispatch_count(), 15);
            fused_tcb.commit_and_wait().unwrap();
            assert_eq!(
                fused_guard.counts(),
                PhysicalTraceCounts {
                    command_count: 1,
                    encoder_count: 1,
                }
            );
            drop(fused_guard);
            let input: Vec<f32> = (0..N).map(|index| index as f32 * 0.25 - 9.0).collect();
            let input_buffer = ctx.new_buffer_with_bytes(bytemuck::cast_slice(&input));
            let zero_graph =
                ReplayableComputeGraph::new(&ctx, zero_replay_stages(&output, &n_buffer, N, 1))
                    .unwrap();
            let add_graph = ReplayableComputeGraph::new(
                &ctx,
                vec![ReplayComputeStage::new(
                    "gravity_add_inplace_f32",
                    (N, 1, 1),
                    (64, 1, 1),
                    vec![
                        ReplayBufferBinding::read_write(0, &output, 0),
                        ReplayBufferBinding::read(1, &input_buffer, 0),
                        ReplayBufferBinding::read(2, &n_buffer, 0),
                    ],
                )],
            )
            .unwrap();
            write_f32(&output, &vec![f32::NAN; N as usize]);
            let dependency_guard = PhysicalTraceGuard::begin(identity("dependency")).unwrap();
            let mut dependency_tcb = TokenCommandBuffer::new(&ctx);
            dependency_tcb
                .execute_replayable_graphs(&[&zero_graph, &add_graph])
                .unwrap();
            dependency_tcb.commit_and_wait().unwrap();
            assert_eq!(
                dependency_guard.counts(),
                PhysicalTraceCounts {
                    command_count: 1,
                    encoder_count: 1,
                }
            );
            drop(dependency_guard);
            assert_eq!(read_f32(&output, N as usize), input);
            let mut direct_ns = Vec::with_capacity(SAMPLES);
            let mut split_ns = Vec::with_capacity(SAMPLES);
            let mut grouped_ns = Vec::with_capacity(SAMPLES);
            let mut fused_ns = Vec::with_capacity(SAMPLES);
            for iteration in 0..SAMPLES {
                let measure_direct = || {
                    let mut tcb = TokenCommandBuffer::new(&ctx);
                    let started = Instant::now();
                    for _ in 0..15 {
                        let output = output.clone();
                        tcb.dispatch_threads(
                            "gravity_zero_f32",
                            (N, 1, 1),
                            (64, 1, 1),
                            move |encoder| {
                                encoder.set_buffer(0, Some(&output), 0);
                                encoder.set_bytes(1, 4, &N as *const u32 as *const _);
                            },
                        )
                        .unwrap();
                    }
                    let elapsed = started.elapsed().as_nanos();
                    tcb.commit_and_wait().unwrap();
                    elapsed
                };
                let measure_split = || {
                    let mut tcb = TokenCommandBuffer::new(&ctx);
                    let started = Instant::now();
                    tcb.execute_replayable_graph(&split_9).unwrap();
                    tcb.execute_replayable_graph(&split_6).unwrap();
                    let elapsed = started.elapsed().as_nanos();
                    tcb.commit_and_wait().unwrap();
                    elapsed
                };
                let measure_grouped = || {
                    let mut tcb = TokenCommandBuffer::new(&ctx);
                    let started = Instant::now();
                    tcb.execute_replayable_graphs(&[&split_9, &split_6])
                        .unwrap();
                    let elapsed = started.elapsed().as_nanos();
                    tcb.commit_and_wait().unwrap();
                    elapsed
                };
                let measure_fused = || {
                    let mut tcb = TokenCommandBuffer::new(&ctx);
                    let started = Instant::now();
                    tcb.execute_replayable_graph(&fused_15).unwrap();
                    let elapsed = started.elapsed().as_nanos();
                    tcb.commit_and_wait().unwrap();
                    elapsed
                };
                if iteration % 2 == 0 {
                    direct_ns.push(measure_direct());
                    split_ns.push(measure_split());
                    grouped_ns.push(measure_grouped());
                    fused_ns.push(measure_fused());
                } else {
                    fused_ns.push(measure_fused());
                    grouped_ns.push(measure_grouped());
                    split_ns.push(measure_split());
                    direct_ns.push(measure_direct());
                }
            }
            assert_eq!(read_f32(&output, N as usize), vec![0.0; N as usize]);
            eprintln!(
                "ICB_FUSION_ENCODE_BENCHMARK samples={SAMPLES} \
                 direct_15_p50_ns={} direct_15_p95_ns={} \
                 split_9_6_p50_ns={} split_9_6_p95_ns={} \
                 grouped_9_6_p50_ns={} grouped_9_6_p95_ns={} \
                 fused_15_p50_ns={} fused_15_p95_ns={}",
                percentile_ns(&direct_ns, 50),
                percentile_ns(&direct_ns, 95),
                percentile_ns(&split_ns, 50),
                percentile_ns(&split_ns, 95),
                percentile_ns(&grouped_ns, 50),
                percentile_ns(&grouped_ns, 95),
                percentile_ns(&fused_ns, 50),
                percentile_ns(&fused_ns, 95),
            );
        }
    }

    // ── v0.5.12: TokenCommandBuffer ───────────────────────────────────────────

    /// Owns one MTLCommandBuffer; multiple kernels can be dispatched into it
    /// before a single `commit_and_wait`. If dropped without committing,
    /// the Drop impl commits automatically so GPU work is never silently lost.
    ///
    /// Usage:
    /// ```ignore
    /// let mut tcb = TokenCommandBuffer::new(&ctx);
    /// tcb.dispatch_threads("rmsnorm", ...)?;
    /// tcb.dispatch_threads("add_inplace", ...)?;
    /// tcb.commit_and_wait()?;
    /// ```
    ///
    /// TCB-internal trace mode (parsed once from `HAWKING_TCB_TRACE` at
    /// construction so the hot path is a single enum compare).
    ///
    /// - **Off** (env var unset or `=0`): zero-overhead default.
    /// - **CpuEncode** (`=1` or `=cpu`): per-dispatch CPU encoding wall time.
    /// - **SplitCbGpu** (`=gpu`): per-dispatch GPU time via dedicated CBs --
    ///   `MTLCommandBuffer::gpuStartTime`/`gpuEndTime` after wait. Inflates
    ///   absolute percentages because each dispatch pays a commit/wait sync.
    /// - **ProdCbGpu** (`=gpu_prod`, v2.2.0-L7): per-dispatch GPU time via
    ///   `MTLCounterSampleBuffer` inside the SAME command buffer. No split,
    ///   so the production TCB pipelining is preserved. Inserts two
    ///   `sampleCountersInBuffer:atSampleIndex:withBarrier:true` calls per
    ///   dispatch (one before, one after); after `commit_and_wait` we read
    ///   the sample buffer and populate `DispatchSample::gpu_us` with the
    ///   real production GPU duration of each kernel. Use this whenever
    ///   tier-2/3 perf decisions need accurate per-kernel attribution
    ///   without the split-CB skew.
    #[derive(Copy, Clone, PartialEq, Eq)]
    pub enum TcbTraceMode {
        Off,
        CpuEncode,
        SplitCbGpu,
        ProdCbGpu,
    }

    impl TcbTraceMode {
        fn from_env() -> Self {
            let raw = std::env::var("HAWKING_TCB_TRACE");
            let mode = match raw.as_deref() {
                Err(_) => Self::Off,
                Ok("") | Ok("0") => Self::Off,
                Ok(s) if s.eq_ignore_ascii_case("gpu_prod") => Self::ProdCbGpu,
                Ok(s) if s.eq_ignore_ascii_case("gpu") => Self::SplitCbGpu,
                Ok(_) => Self::CpuEncode,
            };
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "[hawking] HAWKING_TCB_TRACE={:?} → mode={}",
                    raw.as_deref().unwrap_or("(unset)"),
                    match mode {
                        Self::Off => "Off",
                        Self::CpuEncode => "CpuEncode",
                        Self::SplitCbGpu => "SplitCbGpu",
                        Self::ProdCbGpu => "ProdCbGpu",
                    }
                );
            });
            mode
        }
    }

    /// Set `HAWKING_TCB_TRACE=cpu` for per-kernel CPU encoding timing
    /// (records pipeline-lookup + encode wall time per call, plus a
    /// `tcb_commit` total for the whole CB).
    ///
    /// Set `HAWKING_TCB_TRACE=gpu` for per-kernel GPU-side timing
    /// (each dispatch in its own CB so `gpu_start_time/gpu_end_time` can
    /// be read directly). Populates `DispatchSample::gpu_us`. Slower --
    /// diagnostic mode only. See `TcbTraceMode` for details.
    pub struct TokenCommandBuffer<'ctx> {
        pub ctx: &'ctx MetalContext,
        /// `None` after `commit_and_wait` so the Drop impl knows not to re-commit.
        cmd: Option<metal::CommandBuffer>,
        /// TCB-internal trace mode; resolved once at construction.
        mode: TcbTraceMode,
        /// Accumulated per-dispatch samples; only populated when `mode` is on.
        tcb_samples: Vec<super::DispatchSample>,
        /// An opt-in, non-timing structural trace of the exact kernel labels
        /// accepted by `dispatch_threads` in this command buffer.  Component
        /// parity captures use this to seal dispatch order without enabling
        /// CPU/GPU timing or changing the single-command-buffer execution
        /// shape.  It stays `None` on ordinary runtime paths.
        structural_kernel_names: Option<Vec<String>>,
        /// Live in `ProdCbGpu` mode. `None` in other modes or when the device
        /// doesn't support the timestamp counter set. It is leased from the
        /// context-local pool and returned only after commit/wait + resolve.
        prod_cb_tracer: Option<ProdCbTracer>,
        /// Pool paired with `prod_cb_tracer`; kept optional so off/CPU modes
        /// retain their historical zero-allocation trace path.
        prod_cb_tracer_pool: Option<Arc<Mutex<Vec<ProdCbTracer>>>>,
        /// Physical-evidence identity for the pending Metal command buffer.
        physical_trace: Option<PhysicalCommandIdentity>,
        /// P0.1 spike: active concurrent encoder. When `Some`, dispatches
        /// route into this shared encoder instead of creating per-dispatch
        /// encoders. Only set under `Off` / `CpuEncode` trace modes by
        /// `begin_concurrent_group`; cleared by `end_concurrent_group`.
        /// Caller is responsible for the independence claim of the group's
        /// dispatches (no overlapping read-write or write-write buffer
        /// ranges between any two dispatches in the group).
        concurrent_encoder: Option<ComputeCommandEncoder>,
        /// Track 3.1 / Track 5.1: running count of Metal compute dispatches
        /// encoded into this TCB. Incremented unconditionally (no trace_dispatch
        /// guard) since it's a plain usize add on the hot path — zero cost
        /// compared to the GPU work being encoded. Read back via
        /// `dispatch_count()` after encoding, before `commit_and_wait`.
        pub dispatch_count: usize,
        /// True once any encoder work (compute dispatch, blit, replay) has
        /// been recorded into `cmd`. Drop auto-commits only when set — an
        /// empty temporary TCB must not pay submit/wait.
        has_encoded_work: bool,
        /// Host nanos spent encoding dispatches while a cost-ledger token is
        /// active. Folded into [`crate::cost_ledger::Bucket::MetalEncode`] at
        /// commit. Zero when the ledger is off (no `Instant` on the hot path).
        ledger_encode_ns: u128,
        /// Exact semantic dispatch composition for whole-CB GPU timestamp
        /// attribution. Indexed by [`crate::cost_ledger::GpuStage`].
        ledger_stage_dispatches: [u64; crate::cost_ledger::GpuStage::ALL.len()],
    }

    impl<'ctx> TokenCommandBuffer<'ctx> {
        pub fn new(ctx: &'ctx MetalContext) -> Self {
            let cmd = ctx.inner.queue.new_command_buffer().to_owned();
            let physical_trace = physical_command_label("command_buffer");
            if let Some((_, label)) = physical_trace.as_ref() {
                cmd.set_label(label);
            }
            let mode = TcbTraceMode::from_env();
            let prod_cb_tracer_pool = if mode == TcbTraceMode::ProdCbGpu {
                Some(ctx.prod_cb_tracer_pool.clone())
            } else {
                None
            };
            let pooled_tracer = prod_cb_tracer_pool
                .as_ref()
                .and_then(|pool| pool.lock().pop());
            let prod_cb_tracer = if mode == TcbTraceMode::ProdCbGpu {
                pooled_tracer.or_else(|| ProdCbTracer::try_new(&ctx.inner.device))
            } else {
                None
            };
            Self {
                ctx,
                cmd: Some(cmd),
                mode,
                tcb_samples: Vec::new(),
                structural_kernel_names: None,
                prod_cb_tracer,
                prod_cb_tracer_pool,
                physical_trace: physical_trace.map(|(identity, _)| identity),
                concurrent_encoder: None,
                dispatch_count: 0,
                has_encoded_work: false,
                ledger_encode_ns: 0,
                ledger_stage_dispatches: [0; crate::cost_ledger::GpuStage::ALL.len()],
            }
        }

        /// Return the number of compute dispatches encoded so far.
        /// Valid both before and after `commit_and_wait`.
        pub fn dispatch_count(&self) -> usize {
            self.dispatch_count
        }

        /// Enable a capture-only structural trace for this fresh command
        /// buffer.  The trace records only successful kernel encodes in
        /// order; it never records timestamps or submits additional command
        /// buffers.  Diagnostic callers must opt in before their first
        /// dispatch so a partial historical trace cannot be mistaken for a
        /// full graph.
        pub fn enable_structural_kernel_trace(&mut self) -> Result<()> {
            if self.dispatch_count != 0 {
                return Err(Error::Metal(
                    "structural kernel trace must be enabled before the first dispatch".into(),
                ));
            }
            if self.structural_kernel_names.is_some() {
                return Err(Error::Metal(
                    "structural kernel trace was already enabled for this command buffer".into(),
                ));
            }
            if self.mode != TcbTraceMode::Off {
                return Err(Error::Metal(
                    "structural kernel trace requires non-timed TokenCommandBuffer mode".into(),
                ));
            }
            self.structural_kernel_names = Some(Vec::new());
            Ok(())
        }

        /// Return the exact successful dispatch labels recorded by
        /// [`Self::enable_structural_kernel_trace`].  The slice remains
        /// available before and after the common fence so a failure receipt
        /// can distinguish "encoded" from "committed" without inventing a
        /// timing result.
        pub fn structural_kernel_names(&self) -> Option<&[String]> {
            self.structural_kernel_names.as_deref()
        }

        /// P0.1 spike (Q/K/V concurrent-encoder).
        ///
        /// Open a single `MTLDispatchTypeConcurrent` compute encoder. While
        /// the group is active, subsequent `dispatch_threads` calls record
        /// into this shared encoder (no per-dispatch encoder creation, no
        /// `end_encoding` between them) — the driver may then overlap
        /// SIMD-group-independent dispatches on the GPU.
        ///
        /// **Caller-asserted independence**: the caller MUST guarantee that
        /// no two dispatches in the group share an overlapping read-write
        /// or write-write buffer range. Violation produces undefined
        /// results. The general declarative-range-tracker API is a
        /// follow-on (P0.5); this is a hard-coded sibling for known-
        /// independent triples (e.g., the Q/K/V projection triple that
        /// reads `x_norm_buf` and writes disjoint outputs).
        ///
        /// Under `SplitCbGpu` / `ProdCbGpu` trace modes this method is a
        /// no-op — those modes require per-dispatch encoders for per-
        /// kernel timing, so dispatches in the "group" continue with the
        /// existing per-dispatch encoder pattern. Production paired-bench
        /// runs in `Off` mode, so the spike's ship decision is unaffected.
        ///
        /// Calling twice without an intervening `end_concurrent_group` is
        /// an error.
        pub fn begin_concurrent_group(&mut self) -> Result<()> {
            if self.concurrent_encoder.is_some() {
                return Err(Error::Metal(
                    "begin_concurrent_group called while a group is already active".into(),
                ));
            }
            if !matches!(self.mode, TcbTraceMode::Off | TcbTraceMode::CpuEncode) {
                return Ok(());
            }
            let cmd = self
                .cmd
                .as_ref()
                .ok_or_else(|| Error::Metal("TokenCommandBuffer already committed".into()))?;
            let enc = cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
            if let Some(command) = self.physical_trace.as_ref() {
                enc.set_label(&physical_encoder_label(
                    command,
                    "compute_encoder",
                    "concurrent_group",
                ));
            } else {
                enc.set_label("concurrent_group");
            }
            self.concurrent_encoder = Some(enc.to_owned());
            Ok(())
        }

        /// Open one ordinary (ordered) compute encoder for a sequence of
        /// dependent dispatches. Unlike [`Self::begin_concurrent_group`], this
        /// uses Metal's serial dispatch type: write-after-read and
        /// write-after-write dependencies are preserved by command order. It
        /// is intended for a fully device-resident token lane that contains no
        /// intervening blit encoder, and is opt-in at the caller until its
        /// complete-token parity receipt is green.
        pub fn begin_serial_group(&mut self) -> Result<()> {
            if self.concurrent_encoder.is_some() {
                return Err(Error::Metal(
                    "begin_serial_group called while a group is already active".into(),
                ));
            }
            // SplitCbGpu still no-ops: it needs one CB per dispatch for
            // gpuStart/gpuEnd. ProdCbGpu used to no-op too for per-dispatch
            // counter samples, but that forced the multi-encoder topology and
            // changed Q30 greedy tokens (lane-tracebug). Prefer numeric
            // identity with production: honour serial groups under ProdCbGpu
            // and Off/CpuEncode. Per-dispatch gpu_us is unavailable while a
            // group is open (shared encoder has no per-kernel sample pair).
            if matches!(self.mode, TcbTraceMode::SplitCbGpu) {
                return Ok(());
            }
            let cmd = self
                .cmd
                .as_ref()
                .ok_or_else(|| Error::Metal("TokenCommandBuffer already committed".into()))?;
            // Explicit Serial: never rely on the platform default of
            // `computeCommandEncoder` alone. Dependent dispatches in a Q30
            // token wave (rope→mha, R@x→L@mid, topk→expert) must not run
            // concurrently or residual/router state becomes non-deterministic.
            let enc =
                cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
            if let Some(command) = self.physical_trace.as_ref() {
                enc.set_label(&physical_encoder_label(
                    command,
                    "compute_encoder",
                    "serial_group",
                ));
            } else {
                enc.set_label("serial_group");
            }
            self.concurrent_encoder = Some(enc.to_owned());
            Ok(())
        }

        /// Close the active concurrent group. No-op if none is active.
        pub fn end_concurrent_group(&mut self) -> Result<()> {
            if let Some(enc) = self.concurrent_encoder.take() {
                enc.end_encoding();
            }
            Ok(())
        }

        /// True while a concurrent or serial compute encoder group is open.
        ///
        /// When a group is active, Metal resource residency declared via
        /// [`Self::use_resources_read_on_group`] (or `use_resources` on the
        /// shared encoder) persists for the rest of the group. Callers that
        /// re-declare on every dispatch pay host time for no GPU gain.
        pub fn has_active_group(&self) -> bool {
            self.concurrent_encoder.is_some()
        }

        /// Declare read residency for `resources` on the open serial/concurrent
        /// group encoder. Fail closed if no group is active — per-dispatch
        /// encoders must call `use_resources` inside their own encode closure.
        pub fn use_resources_read_on_group(&mut self, resources: &[PinnedBuffer]) -> Result<()> {
            let enc = self.concurrent_encoder.as_ref().ok_or_else(|| {
                Error::Metal(
                    "use_resources_read_on_group requires an open serial/concurrent group".into(),
                )
            })?;
            if resources.is_empty() {
                return Ok(());
            }
            let mut refs: Vec<&metal::ResourceRef> = Vec::with_capacity(resources.len());
            for resource in resources {
                refs.push(resource);
            }
            enc.use_resources(&refs, MTLResourceUsage::Read);
            Ok(())
        }

        /// Insert a resource-scoped memory barrier on the open group encoder.
        ///
        /// Required under [`MTLDispatchType::Concurrent`] at every real
        /// producer→consumer edge. No-op when no group is active (per-dispatch
        /// encoder modes already serialize across encoder boundaries). Under a
        /// serial group the barrier is redundant but safe.
        pub fn memory_barrier_with_resources(
            &mut self,
            resources: &[&metal::ResourceRef],
        ) -> Result<()> {
            if resources.is_empty() {
                return Ok(());
            }
            if let Some(enc) = self.concurrent_encoder.as_ref() {
                enc.memory_barrier_with_resources(resources);
            }
            Ok(())
        }

        /// Convenience wrapper: barrier on a list of `PinnedBuffer` resources.
        pub fn memory_barrier_with_buffers(&mut self, buffers: &[&PinnedBuffer]) -> Result<()> {
            if buffers.is_empty() || self.concurrent_encoder.is_none() {
                return Ok(());
            }
            let mut refs: Vec<&metal::ResourceRef> = Vec::with_capacity(buffers.len());
            for buffer in buffers {
                refs.push(buffer);
            }
            self.memory_barrier_with_resources(&refs)
        }

        /// Append one execution of a pre-encoded compute graph to this token
        /// command buffer. The replay itself creates one direct compute
        /// encoder and performs no pipeline lookup or buffer-address rebinding.
        ///
        /// Split/per-dispatch GPU trace modes cannot attribute timestamps
        /// inside an ICB and therefore fail closed. Off and aggregate
        /// CPU-encode modes preserve one command-buffer submit/wait.
        pub fn execute_replayable_graph(&mut self, graph: &ReplayableComputeGraph) -> Result<()> {
            self.execute_replayable_graph_group(&[graph], "replayable_compute_graph")
        }

        /// Append several dependency-ordered pre-encoded graphs through one
        /// direct compute encoder. A resource-scoped Metal memory barrier is
        /// inserted between adjacent ICB executions, so a later graph can
        /// consume buffers written by the preceding graph without paying an
        /// additional direct-encoder boundary.
        pub fn execute_replayable_graphs(
            &mut self,
            graphs: &[&ReplayableComputeGraph],
        ) -> Result<()> {
            self.execute_replayable_graph_group(graphs, "replayable_compute_graph_group")
        }

        fn execute_replayable_graph_group(
            &mut self,
            graphs: &[&ReplayableComputeGraph],
            label: &'static str,
        ) -> Result<()> {
            if graphs.is_empty() {
                return Err(Error::Metal(
                    "replayable compute graph group cannot be empty".into(),
                ));
            }
            if graphs
                .iter()
                .any(|graph| !Arc::ptr_eq(&self.ctx.inner, &graph.context))
            {
                return Err(Error::Metal(
                    "replayable compute graph belongs to a different Metal context".into(),
                ));
            }
            if self.concurrent_encoder.is_some() {
                return Err(Error::Metal(
                    "cannot execute a replayable compute graph inside a concurrent group".into(),
                ));
            }
            if !matches!(self.mode, TcbTraceMode::Off | TcbTraceMode::CpuEncode) {
                return Err(Error::Metal(
                    "replayable compute graph does not support split or production per-dispatch tracing"
                        .into(),
                ));
            }

            let ledger_t0 = if crate::cost_ledger::is_recording() {
                Some(Instant::now())
            } else {
                None
            };
            if ledger_t0.is_some() {
                for graph in graphs {
                    if graph.explicit_ledger_stages {
                        for stage in crate::cost_ledger::GpuStage::ALL {
                            let graph_dispatches = graph.ledger_stage_dispatches[stage.index()];
                            let slot = &mut self.ledger_stage_dispatches[stage.index()];
                            *slot = slot.saturating_add(graph_dispatches);
                        }
                    } else {
                        let stage = crate::cost_ledger::current_gpu_stage()
                            .unwrap_or(crate::cost_ledger::GpuStage::Untagged);
                        let slot = &mut self.ledger_stage_dispatches[stage.index()];
                        *slot = slot.saturating_add(graph.command_count as u64);
                    }
                }
            }
            let cpu_t0 = if self.mode == TcbTraceMode::CpuEncode {
                Some(Instant::now())
            } else {
                None
            };
            let cmd = self
                .cmd
                .as_ref()
                .ok_or_else(|| Error::Metal("TokenCommandBuffer already committed".into()))?;
            let enc = cmd.new_compute_command_encoder();
            if let Some(command) = self.physical_trace.as_ref() {
                enc.set_label(&physical_encoder_label(command, "compute_encoder", label));
            } else {
                enc.set_label(label);
            }

            let mut resources = Vec::<ReplayResource>::new();
            let mut resource_slots = HashMap::<u64, usize>::new();
            for graph in graphs {
                for resource in &graph.resources {
                    let address = resource.buffer.gpu_address();
                    if let Some(&slot) = resource_slots.get(&address) {
                        resources[slot].usage |= resource.usage;
                    } else {
                        resource_slots.insert(address, resources.len());
                        resources.push(ReplayResource {
                            buffer: resource.buffer.clone(),
                            usage: resource.usage,
                        });
                    }
                }
            }
            for resource in &resources {
                enc.use_resource(&resource.buffer, resource.usage);
            }
            let barrier_resources: Vec<&metal::ResourceRef> = resources
                .iter()
                .map(|resource| &**resource.buffer)
                .collect();
            let mut command_count = 0usize;
            for (index, graph) in graphs.iter().enumerate() {
                if index != 0 {
                    enc.memory_barrier_with_resources(&barrier_resources);
                }
                let range = NSRange {
                    location: 0,
                    length: graph.command_count as u64,
                };
                unsafe {
                    let _: () = msg_send![
                        enc,
                        executeCommandsInBuffer: &*graph.icb
                        withRange: range
                    ];
                }
                command_count = command_count.saturating_add(graph.command_count);
            }
            enc.end_encoding();

            self.dispatch_count = self.dispatch_count.saturating_add(command_count);
            self.has_encoded_work = true;
            if let Some(t0) = ledger_t0 {
                self.ledger_encode_ns = self
                    .ledger_encode_ns
                    .saturating_add(t0.elapsed().as_nanos());
            }
            if let Some(t0) = cpu_t0 {
                self.tcb_samples.push(super::DispatchSample {
                    kernel_name: static_kernel_name(label),
                    wall_us: t0.elapsed().as_micros() as u64,
                    layer_hint: super::current_layer(),
                    gpu_us: None,
                    gpu_start_ns: None,
                    gpu_end_ns: None,
                });
            }
            Ok(())
        }

        /// Encode one kernel dispatch.
        ///
        /// In **Off** and **CpuEncode** modes the dispatch is appended to the
        /// pending TCB and committed in bulk at `commit_and_wait`. CpuEncode
        /// additionally records pipeline-lookup + encoding wall time.
        ///
        /// In **SplitCbGpu** mode the dispatch is encoded into a fresh
        /// dedicated command buffer that is committed and waited
        /// synchronously. The CB's `gpu_start_time/gpu_end_time` are read
        /// and recorded as `DispatchSample::gpu_us`. The pending TCB is
        /// left empty (`commit_and_wait` will commit an empty CB, which is
        /// a fast no-op on Apple Silicon).
        pub fn dispatch_threads(
            &mut self,
            fn_name: &str,
            grid: (u32, u32, u32),
            tg: (u32, u32, u32),
            encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) -> Result<()> {
            // Track 3.1 / 5.1: count every kernel dispatch unconditionally.
            self.dispatch_count += 1;
            self.has_encoded_work = true;
            // Cost-ledger encode wall: Instant only while a token is active.
            // Folded into MetalEncode at commit_and_wait_split. Default-off
            // path pays one atomic load via is_recording().
            let ledger_t0 = if crate::cost_ledger::is_recording() {
                Some(Instant::now())
            } else {
                None
            };
            if ledger_t0.is_some() {
                let stage = crate::cost_ledger::current_gpu_stage()
                    .unwrap_or(crate::cost_ledger::GpuStage::Untagged);
                let slot = &mut self.ledger_stage_dispatches[stage.index()];
                *slot = slot.saturating_add(1);
            }
            let result = self.dispatch_threads_inner(fn_name, grid, tg, encode);
            if result.is_ok() {
                if let Some(names) = self.structural_kernel_names.as_mut() {
                    names.push(fn_name.to_owned());
                }
            }
            if let Some(t0) = ledger_t0 {
                self.ledger_encode_ns = self
                    .ledger_encode_ns
                    .saturating_add(t0.elapsed().as_nanos());
            }
            result
        }

        fn dispatch_threads_inner(
            &mut self,
            fn_name: &str,
            grid: (u32, u32, u32),
            tg: (u32, u32, u32),
            encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) -> Result<()> {
            // Shared serial/concurrent group encoder. Opened by
            // `begin_serial_group` / `begin_concurrent_group` under Off,
            // CpuEncode, and ProdCbGpu (ProdCbGpu keeps serial groups for
            // Q30 bit-identity). SplitCbGpu never opens a group.
            if self.concurrent_encoder.is_some() {
                let want_wall = matches!(
                    self.mode,
                    TcbTraceMode::CpuEncode | TcbTraceMode::ProdCbGpu
                );
                let t0 = if want_wall {
                    Some(Instant::now())
                } else {
                    None
                };
                let pipe = self.ctx.pipeline(fn_name)?;
                let enc = self
                    .concurrent_encoder
                    .as_ref()
                    .expect("checked is_some above");
                enc.set_compute_pipeline_state(&pipe);
                encode(enc);
                enc.dispatch_threads(
                    MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
                    MTLSize::new(tg.0 as u64, tg.1 as u64, tg.2 as u64),
                );
                if let Some(t0) = t0 {
                    // Encode-wall only: a shared encoder cannot attach
                    // per-dispatch counter-sample pairs. gpu_us stays None.
                    self.tcb_samples.push(super::DispatchSample {
                        kernel_name: static_kernel_name(fn_name),
                        wall_us: t0.elapsed().as_micros() as u64,
                        layer_hint: super::current_layer(),
                        gpu_us: None,
                        gpu_start_ns: None,
                        gpu_end_ns: None,
                    });
                }
                return Ok(());
            }
            if self.mode == TcbTraceMode::SplitCbGpu {
                return self.dispatch_threads_split_cb(fn_name, grid, tg, encode);
            }
            if self.mode == TcbTraceMode::ProdCbGpu && self.prod_cb_tracer.is_some() {
                return self.dispatch_threads_prod_cb(fn_name, grid, tg, encode);
            }
            let t0 = if self.mode == TcbTraceMode::CpuEncode {
                Some(Instant::now())
            } else {
                None
            };
            let cmd = self
                .cmd
                .as_ref()
                .ok_or_else(|| Error::Metal("TokenCommandBuffer already committed".into()))?;
            let pipe = self.ctx.pipeline(fn_name)?;
            let enc = cmd.new_compute_command_encoder();
            if let Some(command) = self.physical_trace.as_ref() {
                enc.set_label(&physical_encoder_label(command, "compute_encoder", fn_name));
            } else {
                enc.set_label(fn_name);
            }
            enc.set_compute_pipeline_state(&pipe);
            encode(enc);
            enc.dispatch_threads(
                MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
                MTLSize::new(tg.0 as u64, tg.1 as u64, tg.2 as u64),
            );
            enc.end_encoding();
            if let Some(t0) = t0 {
                self.tcb_samples.push(super::DispatchSample {
                    kernel_name: static_kernel_name(fn_name),
                    wall_us: t0.elapsed().as_micros() as u64,
                    layer_hint: super::current_layer(),
                    gpu_us: None,
                    gpu_start_ns: None,
                    gpu_end_ns: None,
                });
            }
            Ok(())
        }

        /// v2.2.0-L7: ProdCbGpu path -- same TCB pipelining as Off mode,
        /// but each per-dispatch compute encoder is created via
        /// `MTLComputePassDescriptor` with a sample-buffer attachment
        /// that records GPU timestamps at the encoder's start and end
        /// boundary. After CB wait, those timestamps give the real
        /// production GPU duration per kernel without splitting the
        /// command buffer.
        ///
        /// We use BOUNDARY sampling rather than mid-pass
        /// `sampleCountersInBuffer:atSampleIndex:withBarrier:` because
        /// the M-series GPU family (AGXG15X on M3 Pro) does NOT support
        /// mid-pass compute counter sampling. Boundary-mode IS supported
        /// -- see Apple's
        /// `MTLCommandEncoder::startOfEncoderSampleIndex`/`endOfEncoderSampleIndex`.
        fn dispatch_threads_prod_cb(
            &mut self,
            fn_name: &str,
            grid: (u32, u32, u32),
            tg: (u32, u32, u32),
            encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) -> Result<()> {
            let t0_cpu = Instant::now();
            let cmd = self
                .cmd
                .as_ref()
                .ok_or_else(|| Error::Metal("TokenCommandBuffer already committed".into()))?;
            let tracer = self
                .prod_cb_tracer
                .as_ref()
                .expect("ProdCbGpu path requires a tracer; constructor ensures this");
            let pipe = self.ctx.pipeline(fn_name)?;
            let pair_index = tracer.reserve_pair();

            // Build a per-encoder ComputePassDescriptor with one sample
            // buffer attachment at slot 0, pointing at our shared sample
            // buffer with start/end indices = (2p, 2p+1).
            // Correctness: `compute_command_encoder_with_descriptor` with
            // counter-sample boundary attachments changed Q30 greedy token
            // ids vs plain `new_compute_command_encoder` under the same
            // multi-encoder topology (deterministic; lane-tracebug). Encode
            // compute with a plain encoder. Pair indices are still reserved
            // so post-wait resolve stays structurally valid; without boundary
            // samples `gpu_us` is not a trustworthy per-kernel duration
            // (correctness outranks attribution). Prefer serial groups under
            // ProdCbGpu for production bit-identity (see begin_serial_group).
            let enc = cmd.new_compute_command_encoder();
            let _ = pair_index;
            if let Some(command) = self.physical_trace.as_ref() {
                enc.set_label(&physical_encoder_label(command, "compute_encoder", fn_name));
            } else {
                enc.set_label(fn_name);
            }
            enc.set_compute_pipeline_state(&pipe);
            encode(enc);
            enc.dispatch_threads(
                MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
                MTLSize::new(tg.0 as u64, tg.1 as u64, tg.2 as u64),
            );
            enc.end_encoding();
            let cpu_us = t0_cpu.elapsed().as_micros() as u64;
            let kn = static_kernel_name(fn_name);
            if let Some(p) = pair_index {
                tracer.record_pending(kn, cpu_us, p, super::current_layer());
            } else {
                // Out of capacity -- emit the sample now with gpu_us=None.
                self.tcb_samples.push(super::DispatchSample {
                    kernel_name: kn,
                    wall_us: cpu_us,
                    layer_hint: super::current_layer(),
                    gpu_us: None,
                    gpu_start_ns: None,
                    gpu_end_ns: None,
                });
            }
            Ok(())
        }

        /// SplitCbGpu path: each dispatch in its own CB, gpu times read
        /// directly from `gpu_start_time/gpu_end_time` after wait.
        fn dispatch_threads_split_cb(
            &mut self,
            fn_name: &str,
            grid: (u32, u32, u32),
            tg: (u32, u32, u32),
            encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) -> Result<()> {
            let t0_cpu = Instant::now();
            let dedicated = self.ctx.inner.queue.new_command_buffer();
            let physical_trace = physical_command_label("command_buffer");
            if let Some((_, label)) = physical_trace.as_ref() {
                dedicated.set_label(label);
            }
            let pipe = self.ctx.pipeline(fn_name)?;
            let enc = dedicated.new_compute_command_encoder();
            if let Some((command, _)) = physical_trace.as_ref() {
                enc.set_label(&physical_encoder_label(command, "compute_encoder", fn_name));
            } else {
                enc.set_label(fn_name);
            }
            enc.set_compute_pipeline_state(&pipe);
            encode(enc);
            enc.dispatch_threads(
                MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
                MTLSize::new(tg.0 as u64, tg.1 as u64, tg.2 as u64),
            );
            enc.end_encoding();
            let cpu_us = t0_cpu.elapsed().as_micros() as u64;
            dedicated.commit();
            if self.ctx.trace_dispatch {
                self.ctx.stats.commits.fetch_add(1, Ordering::Relaxed);
            }
            dedicated.wait_until_completed();
            // GPUStartTime / GPUEndTime are not wrapped by metal 0.29 -- go
            // direct via objc msg_send. Both return CFTimeInterval (f64
            // seconds since an absolute reference); their difference is the
            // GPU compute duration. Safe because we just waited.
            let gpu_us = unsafe { cb_gpu_duration_us(&dedicated) };
            self.tcb_samples.push(super::DispatchSample {
                kernel_name: static_kernel_name(fn_name),
                wall_us: cpu_us,
                layer_hint: super::current_layer(),
                gpu_us: Some(gpu_us),
                gpu_start_ns: None,
                gpu_end_ns: None,
            });
            Ok(())
        }

        /// Encode a GPU-side buffer copy into the pending command buffer.
        ///
        /// Uses a `MTLBlitCommandEncoder` -- very cheap (~100 ns; a plain GPU
        /// memcpy). Call once per MoE layer to snapshot route_ids into the
        /// per-token route history buffer without breaking the single-CB design.
        ///
        /// In `SplitCbGpu` mode the blit is committed in its own CB so the
        /// next compute dispatch starts cleanly; no GPU time is recorded for
        /// blits (they're not the audit target).
        pub fn copy_buffer_bytes(
            &mut self,
            src: &metal::Buffer,
            src_offset: u64,
            dst: &metal::Buffer,
            dst_offset: u64,
            size: u64,
        ) -> Result<()> {
            if size == 0 {
                return Ok(());
            }
            if self.mode == TcbTraceMode::SplitCbGpu {
                let dedicated = self.ctx.inner.queue.new_command_buffer();
                let physical_trace = physical_command_label("command_buffer");
                if let Some((_, label)) = physical_trace.as_ref() {
                    dedicated.set_label(label);
                }
                let blit = dedicated.new_blit_command_encoder();
                if let Some((command, _)) = physical_trace.as_ref() {
                    blit.set_label(&physical_encoder_label(command, "blit_encoder", "copy"));
                }
                blit.copy_from_buffer(src, src_offset, dst, dst_offset, size);
                blit.end_encoding();
                dedicated.commit();
                if self.ctx.trace_dispatch {
                    self.ctx.stats.commits.fetch_add(1, Ordering::Relaxed);
                }
                dedicated.wait_until_completed();
                return Ok(());
            }
            let cmd = self
                .cmd
                .as_ref()
                .ok_or_else(|| Error::Metal("TokenCommandBuffer already committed".into()))?;
            let blit = cmd.new_blit_command_encoder();
            if let Some(command) = self.physical_trace.as_ref() {
                blit.set_label(&physical_encoder_label(command, "blit_encoder", "copy"));
            }
            blit.copy_from_buffer(src, src_offset, dst, dst_offset, size);
            blit.end_encoding();
            self.has_encoded_work = true;
            Ok(())
        }

        /// Encode a GPU-side byte fill into the pending command buffer.
        /// Used to clear device flags (e.g. finite-check) without a host write
        /// that would race a prior in-flight command buffer on the same queue.
        pub fn fill_buffer_bytes(
            &mut self,
            dst: &metal::Buffer,
            dst_offset: u64,
            size: u64,
            value: u8,
        ) -> Result<()> {
            if size == 0 {
                return Ok(());
            }
            if self.mode == TcbTraceMode::SplitCbGpu {
                let dedicated = self.ctx.inner.queue.new_command_buffer();
                let physical_trace = physical_command_label("command_buffer");
                if let Some((_, label)) = physical_trace.as_ref() {
                    dedicated.set_label(label);
                }
                let blit = dedicated.new_blit_command_encoder();
                if let Some((command, _)) = physical_trace.as_ref() {
                    blit.set_label(&physical_encoder_label(command, "blit_encoder", "fill"));
                }
                blit.fill_buffer(
                    dst,
                    NSRange {
                        location: dst_offset,
                        length: size,
                    },
                    value,
                );
                blit.end_encoding();
                dedicated.commit();
                if self.ctx.trace_dispatch {
                    self.ctx.stats.commits.fetch_add(1, Ordering::Relaxed);
                }
                dedicated.wait_until_completed();
                return Ok(());
            }
            let cmd = self
                .cmd
                .as_ref()
                .ok_or_else(|| Error::Metal("TokenCommandBuffer already committed".into()))?;
            let blit = cmd.new_blit_command_encoder();
            if let Some(command) = self.physical_trace.as_ref() {
                blit.set_label(&physical_encoder_label(command, "blit_encoder", "fill"));
            }
            blit.fill_buffer(
                dst,
                NSRange {
                    location: dst_offset,
                    length: size,
                },
                value,
            );
            blit.end_encoding();
            self.has_encoded_work = true;
            Ok(())
        }

        /// Resolve a completed production-CB tracer, publish its ordered
        /// samples, then return the now-reset buffer to the context pool.
        /// This must run only after `wait_until_completed`; callers below
        /// satisfy that fence before invoking it.
        fn flush_prod_cb_trace_and_recycle(&mut self) {
            if let Some(tracer) = self.prod_cb_tracer.take() {
                for sample in tracer.drain() {
                    self.ctx.trace.samples.lock().push(sample);
                }
                if let Some(pool) = self.prod_cb_tracer_pool.as_ref() {
                    pool.lock().push(tracer);
                }
            }
            // Out-of-capacity samples (if a single CB exceeds the fixed
            // 1024-pair buffer) retain their explicit gpu_us=None boundary.
            for sample in self.tcb_samples.drain(..) {
                self.ctx.trace.samples.lock().push(sample);
            }
        }

        /// Return an unused diagnostic tracer without resolving timestamps.
        /// No command buffer has been committed in this branch, so its pair
        /// counter is necessarily zero and the buffer is safe to reuse.
        fn recycle_unused_prod_cb_tracer(&mut self) {
            if let Some(tracer) = self.prod_cb_tracer.take() {
                if let Some(pool) = self.prod_cb_tracer_pool.as_ref() {
                    pool.lock().push(tracer);
                }
            }
        }

        /// Commit the command buffer and block until the GPU finishes.
        /// Consumes self; subsequent dispatch calls would fail.
        ///
        /// When the per-token cost ledger is **recording**, this is identical
        /// to [`commit_and_wait_split`] (metal submit / synchronize + GPU
        /// timestamps). When the ledger is off, behaviour is unchanged —
        /// a single atomic load then the historical flush path.
        pub fn commit_and_wait(self) -> Result<()> {
            self.commit_and_wait_split()
        }

        /// Commit without waiting. The Metal command queue still serialises
        /// buffers, so a later `commit_and_wait` on the same queue drains
        /// every earlier committed buffer. Used by device-resident
        /// autoregressive feedback so token N+1 can be enqueued while token N
        /// runs — the host no longer blocks on the sampled id between steps.
        ///
        /// Consumes self. Trace modes that need a completed CB for GPU
        /// timestamps (SplitCbGpu / ProdCbGpu) fall back to
        /// [`commit_and_wait`] so diagnostic attribution stays valid.
        pub fn commit_no_wait(mut self) -> Result<()> {
            use crate::cost_ledger::{self, Bucket};
            use std::time::Instant;

            // Diagnostic GPU-timestamp modes require a completed CB to resolve
            // counters; keep their historical wait behaviour.
            if matches!(
                self.mode,
                TcbTraceMode::SplitCbGpu | TcbTraceMode::ProdCbGpu
            ) {
                return self.commit_and_wait();
            }

            if let Some(cmd) = self.cmd.take() {
                if let Some(enc) = self.concurrent_encoder.take() {
                    enc.end_encoding();
                }
                if cost_ledger::is_recording() {
                    if self.ledger_encode_ns > 0 {
                        cost_ledger::add_duration(
                            Bucket::MetalEncode,
                            std::time::Duration::from_nanos(self.ledger_encode_ns as u64),
                        );
                        self.ledger_encode_ns = 0;
                    }
                    cost_ledger::record_dispatches(self.dispatch_count as u64);
                    let t_submit = Instant::now();
                    cmd.commit();
                    if self.ctx.trace_dispatch {
                        self.ctx.stats.commits.fetch_add(1, Ordering::Relaxed);
                    }
                    let commit_d = t_submit.elapsed();
                    cost_ledger::add_duration(Bucket::MetalSubmit, commit_d);
                    cost_ledger::record_command_buffer();
                    // No wait: GPU timestamps and host_wait are recorded when
                    // a later terminal drain waits on the queue.
                } else {
                    cmd.commit();
                    if self.ctx.trace_dispatch {
                        self.ctx.stats.commits.fetch_add(1, Ordering::Relaxed);
                    }
                }
                match self.mode {
                    TcbTraceMode::Off => {}
                    TcbTraceMode::CpuEncode => {
                        let layer = super::current_layer();
                        for s in self.tcb_samples.drain(..) {
                            self.ctx
                                .trace
                                .record(s.kernel_name, s.wall_us, s.layer_hint);
                        }
                        self.ctx.trace.record("tcb_commit_no_wait", 0, layer);
                    }
                    TcbTraceMode::SplitCbGpu | TcbTraceMode::ProdCbGpu => unreachable!(),
                }
                // Dropping `cmd` is safe: after commit the queue retains the
                // buffer until completion. We must not wait here.
                drop(cmd);
            }
            // Prevent Drop from re-committing: cmd is already None.
            self.has_encoded_work = false;
            self.recycle_unused_prod_cb_tracer();
            Ok(())
        }

        /// Like [`commit_and_wait`], but when the per-token cost ledger is
        /// recording, attributes CPU wall time of `commit` and
        /// `wait_until_completed` to separate buckets and counts one
        /// command buffer + one sync point. Behaviour on the GPU is
        /// identical; decode output is unchanged. When the ledger is not
        /// recording this is the same as the historical uninstrumented commit.
        pub fn commit_and_wait_split(mut self) -> Result<()> {
            use crate::cost_ledger::{self, Bucket};
            use std::time::Instant;

            if let Some(cmd) = self.cmd.take() {
                // Close any still-open concurrent group before committing.
                if let Some(enc) = self.concurrent_encoder.take() {
                    enc.end_encoding();
                }
                if cost_ledger::is_recording() {
                    // Charge encode wall that was accumulated while
                    // `dispatch_threads` ran under an active token (default-off).
                    if self.ledger_encode_ns > 0 {
                        cost_ledger::add_duration(
                            Bucket::MetalEncode,
                            std::time::Duration::from_nanos(self.ledger_encode_ns as u64),
                        );
                        self.ledger_encode_ns = 0;
                    }
                    cost_ledger::record_dispatches(self.dispatch_count as u64);

                    let t_submit = Instant::now();
                    cmd.commit();
                    if self.ctx.trace_dispatch {
                        self.ctx.stats.commits.fetch_add(1, Ordering::Relaxed);
                    }
                    let commit_d = t_submit.elapsed();
                    cost_ledger::add_duration(Bucket::MetalSubmit, commit_d);
                    cost_ledger::record_command_buffer();

                    let t_sync = Instant::now();
                    cmd.wait_until_completed();
                    let wait_d = t_sync.elapsed();
                    cost_ledger::add_duration(Bucket::MetalSynchronize, wait_d);
                    cost_ledger::record_sync_point();
                    let status = cmd.status();
                    if status != metal::MTLCommandBufferStatus::Completed {
                        eprintln!(
                            "[hawking] Metal command buffer did not complete cleanly: status={status:?}"
                        );
                    }

                    // Device timeline: GPUStartTime/GPUEndTime after wait.
                    // Counter-sample markers are not encoded on this path
                    // (would change the CB); capability is probed only.
                    let n_disp = self.dispatch_count() as u64;
                    let stage_dispatches: Vec<_> = crate::cost_ledger::GpuStage::ALL
                        .iter()
                        .copied()
                        .filter_map(|stage| {
                            let n = self.ledger_stage_dispatches[stage.index()];
                            (n > 0).then_some((stage, n))
                        })
                        .collect();
                    ledger_record_cb_gpu(
                        &cmd,
                        commit_d.as_micros() as u64,
                        wait_d.as_micros() as u64,
                        n_disp,
                        &stage_dispatches,
                    );
                    ledger_probe_counter_capability(&self.ctx.inner.device);

                    // Drain TCB-internal trace samples the same way
                    // flush_and_commit does in Off mode (nothing to do).
                    match self.mode {
                        TcbTraceMode::Off => {}
                        TcbTraceMode::CpuEncode => {
                            let layer = super::current_layer();
                            for s in self.tcb_samples.drain(..) {
                                self.ctx
                                    .trace
                                    .record(s.kernel_name, s.wall_us, s.layer_hint);
                            }
                            self.ctx.trace.record("tcb_commit", 0, layer);
                        }
                        TcbTraceMode::SplitCbGpu => {
                            for s in self.tcb_samples.drain(..) {
                                self.ctx.trace.samples.lock().push(s);
                            }
                        }
                        TcbTraceMode::ProdCbGpu => {
                            self.flush_prod_cb_trace_and_recycle();
                        }
                    }
                } else {
                    self.flush_and_commit(cmd);
                }
            }
            Ok(())
        }

        /// Internal: commit `cmd`, wait for GPU completion, then flush TCB trace
        /// samples to `ctx.trace`. In SplitCbGpu mode `cmd` is the trailing
        /// empty CB (each dispatch already self-committed); we still commit
        /// it for symmetry and flush the per-dispatch samples without adding
        /// a tcb_commit record (it would be meaningless in split mode).
        fn flush_and_commit(&mut self, cmd: metal::CommandBuffer) {
            // Close any still-open concurrent group before committing the
            // CB — Metal requires all encoders be ended before commit.
            if let Some(enc) = self.concurrent_encoder.take() {
                enc.end_encoding();
            }
            let t0 = if self.mode == TcbTraceMode::CpuEncode {
                Some(Instant::now())
            } else {
                None
            };
            cmd.commit();
            if self.ctx.trace_dispatch {
                self.ctx.stats.commits.fetch_add(1, Ordering::Relaxed);
            }
            cmd.wait_until_completed();
            // Fail closed on GPU errors: an errored CB can leave shared buffers
            // half-written, which surfaces as non-deterministic greedy decode
            // (varying route ids / dispatch counts) rather than a hard error
            // if we only wait. Drop path cannot return Result — log loudly.
            let status = cmd.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                eprintln!(
                    "[hawking] Metal command buffer did not complete cleanly: status={status:?}"
                );
            }
            match self.mode {
                TcbTraceMode::Off => {}
                TcbTraceMode::CpuEncode => {
                    let layer = super::current_layer();
                    let total_us = t0.unwrap().elapsed().as_micros() as u64;
                    for s in self.tcb_samples.drain(..) {
                        self.ctx
                            .trace
                            .record(s.kernel_name, s.wall_us, s.layer_hint);
                    }
                    self.ctx.trace.record("tcb_commit", total_us, layer);
                }
                TcbTraceMode::SplitCbGpu => {
                    // Flush per-dispatch GPU-timed samples directly. There is
                    // no aggregate `tcb_commit` in split mode -- the GPU times
                    // already sum to the decoded total.
                    for s in self.tcb_samples.drain(..) {
                        self.ctx.trace.samples.lock().push(s);
                    }
                }
                TcbTraceMode::ProdCbGpu => {
                    // v2.2.0-L7: resolve the counter sample buffer now that
                    // the CB has completed. `drain()` reads the raw
                    // timestamps and pairs them with the recorded
                    // dispatch metadata to populate `gpu_us`.
                    self.flush_prod_cb_trace_and_recycle();
                }
            }
        }
    }

    impl Drop for TokenCommandBuffer<'_> {
        fn drop(&mut self) {
            // Only auto-commit when work was encoded. An empty Drop used to
            // `commit` + `wait_until_completed` on a fresh CB (full submit
            // latency, no useful GPU work). That topology bug made
            // per-encode temporary TCBs look free in the ledger while still
            // burning wall clock — the C1/C2 / device-SiLU failure shape.
            // Explicit `commit_and_wait` still flushes empty buffers if a
            // caller wants a fence with no dispatches.
            if !self.has_encoded_work {
                let _ = self.cmd.take();
                let _ = self.concurrent_encoder.take();
                self.recycle_unused_prod_cb_tracer();
                return;
            }
            if let Some(cmd) = self.cmd.take() {
                self.flush_and_commit(cmd);
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::*;

    /// Platform-stub. Constructors error so non-macOS targets can
    /// still compile hawking-core (running it requires a Mac).
    #[derive(Clone)]
    pub struct MetalContext {
        _priv: Arc<()>,
    }

    impl MetalContext {
        pub fn new() -> Result<Self> {
            Err(Error::Metal("metal unavailable on this platform".into()))
        }

        pub fn new_with_trace(_trace_dispatch: bool) -> Result<Self> {
            Err(Error::Metal("metal unavailable on this platform".into()))
        }

        pub fn new_with_trace_strict_math(_trace_dispatch: bool) -> Result<Self> {
            Err(Error::Metal("metal unavailable on this platform".into()))
        }

        pub fn device_name(&self) -> String {
            "metal-unavailable".into()
        }

        pub fn drain_trace(&self) -> Vec<super::DispatchSample> {
            Vec::new()
        }

        pub fn drain_stats(&self) -> (usize, usize, usize) {
            (0, 0, 0)
        }
    }

    /// Non-macOS stub for the macOS pinned-buffer handle. Never
    /// constructed (DeepSeekV2 always holds Option<PinnedBuffer> = None
    /// off-macOS); exists so downstream struct fields type-check.
    #[derive(Clone)]
    pub struct PinnedBuffer {
        _priv: Arc<()>,
    }

    /// Non-macOS stub for TokenCommandBuffer. Never constructed off-macOS.
    pub struct TokenCommandBuffer<'ctx> {
        _ctx: std::marker::PhantomData<&'ctx ()>,
        pub dispatch_count: usize,
    }

    impl<'ctx> TokenCommandBuffer<'ctx> {
        pub fn new(_ctx: &'ctx MetalContext) -> Self {
            panic!("TokenCommandBuffer: Metal unavailable on this platform")
        }

        pub fn dispatch_threads(
            &mut self,
            _fn_name: &str,
            _grid: (u32, u32, u32),
            _tg: (u32, u32, u32),
            _encode: impl FnOnce(()),
        ) -> Result<()> {
            Err(Error::Metal("metal unavailable on this platform".into()))
        }

        pub fn dispatch_count(&self) -> usize {
            0
        }

        pub fn commit_and_wait(self) -> Result<()> {
            Err(Error::Metal("metal unavailable on this platform".into()))
        }
    }
}

pub use imp::{MetalContext, PinnedBuffer, TokenCommandBuffer};

#[cfg(target_os = "macos")]
pub use imp::{
    CommandBatch, ReplayBufferBinding, ReplayComputeStage, ReplayResourceDeclaration,
    ReplayableComputeGraph,
};

pub mod argbuf;
pub use argbuf::{ArgLayout, KernelArgBuffer};

pub mod decode_arena;
pub use decode_arena::DecodeArena;

pub mod dense_decode_arena;
pub use dense_decode_arena::DenseDecodeArena;

pub mod rwkv_decode_arena;
#[cfg(target_os = "macos")]
pub use rwkv_decode_arena::RwkvDecodeArena;
