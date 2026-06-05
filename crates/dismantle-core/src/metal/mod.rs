use crate::{Error, Result};
use std::sync::Arc;

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

/// Concatenation of all shader sources for a single library compile.
/// Cheaper than separate compile units; lets common helpers be shared.
pub fn all_shader_sources() -> String {
    [
        SHADER_COMMON,
        SHADER_QUANT,
        SHADER_MOE,
        SHADER_ATTN,
        SHADER_SAMPLE,
        SHADER_MATMUL,
        SHADER_MHA,
        SHADER_MEGAKERNEL,
    ]
    .join("\n\n")
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
/// `DISMANTLE_TCB_TRACE=gpu` mode where each dispatch lands in its own
/// command buffer so `MTLCommandBuffer::gpuStartTime/gpuEndTime` can be
/// read directly. In the default and `DISMANTLE_TCB_TRACE=cpu` modes
/// `gpu_us` is `None`.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct DispatchSample {
    pub kernel_name: &'static str,
    pub wall_us: u64,
    pub layer_hint: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_us: Option<u64>,
    /// Raw GPU-clock start/end timestamps (ns) for this dispatch, from the
    /// timestamp counter set. Populated ONLY by `DISMANTLE_TCB_TRACE=gpu_prod`
    /// (single-CB production path); `None` everywhere else. Carrying the raw
    /// endpoints — not just their difference — lets an offline parser compute
    /// the PRODUCTION inter-dispatch gap (start[i+1] - end[i]) without
    /// Instruments. Off by default ⇒ parity-neutral (skipped when `None`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_start_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_end_ns: Option<u64>,
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
    use metal::{
        Buffer, CommandBufferRef, CommandQueue, ComputeCommandEncoder, ComputePipelineState,
        Device, Library, MTLDispatchType, MTLResourceOptions, MTLSize,
    };
    use metal::objc::{class, msg_send, sel, sel_impl};

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
        // CFTimeInterval is `double` (f64) -- seconds since absolute reference.
        let start: f64 = msg_send![cb, GPUStartTime];
        let end: f64 = msg_send![cb, GPUEndTime];
        let dt = end - start;
        if dt > 0.0 {
            (dt * 1_000_000.0) as u64
        } else {
            0
        }
    }
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    /// Wave-6 residency lever (`DISMANTLE_QWEN_RESIDENCY=1`, DEFAULT-OFF).
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
    /// queue. In dismantle they all do -- the weights buffer is backed by
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
            return Err(Error::Metal("MTLResidencySetDescriptor alloc failed".into()));
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
            return Err(Error::Metal(format!("newResidencySetWithDescriptor:error: {msg}")));
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
    /// One sample buffer per `TokenCommandBuffer`; `sample_count` is sized
    /// for 2 samples per dispatch × MAX_DISPATCHES. Each dispatch occupies
    /// indices `[2*n, 2*n+1]`. After CB completes, the sample buffer holds
    /// raw GPU timestamps (ns); `gpu_us = (ts[2n+1] - ts[2n]) / 1000`.
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
            let sample_buf = device.new_counter_sample_buffer_with_descriptor(&desc).ok()?;
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
            let pair_count = self.next_pair.load(Ordering::Relaxed).min(self.capacity_pairs);
            if pair_count == 0 {
                return pending
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
            pending
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
                .collect()
        }
    }

    // Re-export Metal's Buffer type so callers can hold pinned-weight
    // handles without depending on the upstream `metal` crate directly.
    pub use ::metal::Buffer as PinnedBuffer;

    /// Accumulated per-dispatch timing samples. Gate-able via
    /// `DISMANTLE_TRACE_DISPATCH` env var; when the var is absent the
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
        /// Mirrors `EngineConfig::trace_dispatch`; env var `DISMANTLE_TRACE_DISPATCH`
        /// acts as a fallback when this is false.
        pub trace_dispatch: bool,
    }

    /// One command buffer that can encode several compute kernels before
    /// a single commit/wait. This is the stepping stone between the
    /// current per-kernel dispatch path and the future strict single
    /// FlashMoE kernel.
    pub struct CommandBatch<'a> {
        ctx: &'a MetalContext,
        cmd: &'a CommandBufferRef,
    }

    struct Inner {
        device: Device,
        queue: CommandQueue,
        library: Library,
        pipelines: Mutex<HashMap<String, ComputePipelineState>>,
    }

    /// Resolve a runtime kernel name to a `&'static str` for zero-alloc
    /// trace recording. Covers all kernel names used by dismantle; anything
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
            "sample_argmax_f32" => "sample_argmax_f32",
            // attn / rope / embed kernels
            "rope_inplace" => "rope_inplace",
            // dequant / gemm variants
            "dequant_q8_0" => "dequant_q8_0",
            "gemm_q4_k_m_fused" => "gemm_q4_k_m_fused",
            "gemm_q4_k_m_fused_simd" => "gemm_q4_k_m_fused_simd",
            "gemm_q4_k_m_fused_v2" => "gemm_q4_k_m_fused_v2",
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
            // v0.5.10 fp16 Q-format kernels
            "gemm_q4_k_m_fused_f16" => "gemm_q4_k_m_fused_f16",
            "moe_grouped_gemm_q4_f16" => "moe_grouped_gemm_q4_f16",
            "dequant_q6_k_f16" => "dequant_q6_k_f16",
            // v1.1.1 / v2.1.0 -- T1.1 audit closed 22 names previously
            // bucketed as "other" (incl. the post-T2.1 default MoE Q4_K
            // v2t_gu_v2 kernel itself -- biggest attribution miss).
            "moe_batched_gemm_q4_indexed_v2t_gu" => "moe_batched_gemm_q4_indexed_v2t_gu",
            "moe_batched_gemm_q4_indexed_v2t_gu_v2" => "moe_batched_gemm_q4_indexed_v2t_gu_v2",
            "moe_batched_gemm_q8_0_indexed_v2t" => "moe_batched_gemm_q8_0_indexed_v2t",
            "moe_batched_gemm_q5_0_indexed_v2t" => "moe_batched_gemm_q5_0_indexed_v2t",
            "moe_batched_gemm_q6_k_indexed_v2t" => "moe_batched_gemm_q6_k_indexed_v2t",
            "gemm_q3_k_fused_v2" => "gemm_q3_k_fused_v2",
            "gemm_q3_k_fused_2r" => "gemm_q3_k_fused_2r",
            "gemm_q3_k_v4_predec" => "gemm_q3_k_v4_predec",
            "gemm_q6_k_fused_v2" => "gemm_q6_k_fused_v2",
            "gemm_q6_k_fused_v2_swiglu" => "gemm_q6_k_fused_v2_swiglu",
            "gemm_q4_k_m_simdmat" => "gemm_q4_k_m_simdmat",
            "gemm_q4_k_m_v3_8r" => "gemm_q4_k_m_v3_8r",
            "gemm_q4_k_v4_predec" => "gemm_q4_k_v4_predec",
            "gemm_q4_k_v4_predec_swiglu" => "gemm_q4_k_v4_predec_swiglu",
            "gemm_q4_k_v4_predec_2r" => "gemm_q4_k_v4_predec_2r",
            "gemm_q4_k_v4_predec_2r_swiglu" => "gemm_q4_k_v4_predec_2r_swiglu",
            "gemm_q4_k_v4_predec_2r_f16s" => "gemm_q4_k_v4_predec_2r_f16s",
            "gemm_q4_k_v4_predec_4r" => "gemm_q4_k_v4_predec_4r",
            "gemm_q4_k_v4_predec_4r_swiglu" => "gemm_q4_k_v4_predec_4r_swiglu",
            "gemm_q4_k_v4_predec_pair" => "gemm_q4_k_v4_predec_pair",
            "gemm_q4_k_v4_predec_pair_f16s" => "gemm_q4_k_v4_predec_pair_f16s",
            "gemm_q4_k_m_v3_dual" => "gemm_q4_k_m_v3_dual",
            "gemm_q4_k_m_v3_llama" => "gemm_q4_k_m_v3_llama",
            "gemv_f16_f16in" => "gemv_f16_f16in",
            "kv_append_f32" => "kv_append_f32",
            "rmsnorm_f32" => "rmsnorm_f32",
            "rmsnorm_f32_to_f16" => "rmsnorm_f32_to_f16",
            "rmsnorm_gemv_f16w_attn_pinned" => "rmsnorm_gemv_f16w_attn_pinned",
            "rmsnorm_gemv_f16w_attn_pinned_v2t" => "rmsnorm_gemv_f16w_attn_pinned_v2t",
            "rope_q_f32_inplace" => "rope_q_f32_inplace",
            "rope_slice_f32_inplace" => "rope_slice_f32_inplace",
            "embed_lookup_f32" => "embed_lookup_f32",
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
            "add_inplace_broadcast" => "add_inplace_broadcast",
            "memcpy_f32_off" => "memcpy_f32_off",
            "memcpy_f32_to_f16_off" => "memcpy_f32_to_f16_off",
            "add_rmsnorm_fused_batched" => "add_rmsnorm_fused_batched",
            "gemm_q4_k_m_batched_v2" => "gemm_q4_k_m_batched_v2",
            "gemm_q4_k_m_batched_v3" => "gemm_q4_k_m_batched_v3",
            "gemm_q4_k_m_batched_v3w" => "gemm_q4_k_m_batched_v3w",
            "gemm_q4_k_m_batched_v3w_predec" => "gemm_q4_k_m_batched_v3w_predec",
            "gemm_q4k_fast_v1" => "gemm_q4k_fast_v1",
            "gemm_q4_k_a8_v3_8r_per_channel" => "gemm_q4_k_a8_v3_8r_per_channel",
            "quantize_f32_to_int8_per_channel" => "quantize_f32_to_int8_per_channel",
            "qwen3b_megakernel_2layer" => "qwen3b_megakernel_2layer",
            "qwen3b_megakernel_nlayer" => "qwen3b_megakernel_nlayer",
            "gpu_address_probe" => "gpu_address_probe",
            "use_resource_poc_add" => "use_resource_poc_add",
            _ => "other",
        }
    }

    impl MetalContext {
        pub fn new() -> Result<Self> {
            Self::new_with_trace(false)
        }

        pub fn new_with_trace(trace_dispatch: bool) -> Result<Self> {
            let device = Device::system_default()
                .ok_or_else(|| Error::Metal("no Metal-capable GPU".into()))?;
            let queue = device.new_command_queue();
            let opts = metal::CompileOptions::new();
            let src = super::all_shader_sources();
            let library = device
                .new_library_with_source(&src, &opts)
                .map_err(|e| Error::Metal(format!("shader compile: {e}")))?;
            // Resolve at construction so hot-path checks are a single bool load.
            let effective = trace_dispatch || std::env::var_os("DISMANTLE_TRACE_DISPATCH").is_some();
            Ok(Self {
                inner: Arc::new(Inner {
                    device,
                    queue,
                    library,
                    pipelines: Mutex::new(HashMap::new()),
                }),
                trace: Arc::new(DispatchTrace::new()),
                stats: Arc::new(MetalContextStats::new()),
                trace_dispatch: effective,
            })
        }

        pub fn device(&self) -> &Device {
            &self.inner.device
        }
        pub fn queue(&self) -> &CommandQueue {
            &self.inner.queue
        }

        /// Wave-6 (`DISMANTLE_QWEN_RESIDENCY=1`, DEFAULT-OFF): pin
        /// `allocations` (the decode working set: weights mmap + arena +
        /// model buffers) into one process-lifetime `MTLResidencySet`
        /// attached to this context's command queue. Call ONCE, after
        /// load / on first decode. No-op (returns `Ok`) when the flag is
        /// unset -- so the golden path issues zero residency objc traffic
        /// and stays byte-for-byte unchanged.
        ///
        /// SAFETY contract is `install_residency_set`'s: each buffer must
        /// outlive the command queue (true for all dismantle decode
        /// buffers -- weights are mmap-backed for the engine lifetime,
        /// arena/model buffers live on the engine).
        pub fn request_residency(&self, allocations: &[&Buffer]) -> Result<()> {
            if !crate::env_on("DISMANTLE_QWEN_RESIDENCY") {
                return Ok(());
            }
            if allocations.is_empty() {
                return Ok(());
            }
            // &Buffer derefs to &BufferRef (foreign_obj_type! Deref).
            let refs: Vec<&metal::BufferRef> =
                allocations.iter().map(|b| &***b).collect();
            unsafe {
                install_residency_set(&self.inner.device, &self.inner.queue, &refs)
            }
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
            pipes.insert(fn_name.to_string(), p.clone());
            Ok(p)
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
                self.stats.bytes_allocated.fetch_add(bytes.len(), Ordering::Relaxed);
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
                self.stats.bytes_allocated.fetch_add(bytes.len(), Ordering::Relaxed);
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
            let trace_enabled = self.trace_dispatch;
            let t0 = if trace_enabled { Some(Instant::now()) } else { None };

            let pipe = self.pipeline(fn_name)?;
            let cmd = self.inner.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_label(fn_name);
            enc.set_compute_pipeline_state(&pipe);
            encode(enc);
            enc.dispatch_threads(
                MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
                MTLSize::new(tg.0 as u64, tg.1 as u64, tg.2 as u64),
            );
            enc.end_encoding();
            cmd.commit();
            if trace_enabled {
                self.stats.commits.fetch_add(1, Ordering::Relaxed);
            }
            cmd.wait_until_completed();

            if let Some(t0) = t0 {
                let wall_us = t0.elapsed().as_micros() as u64;
                self.trace.record(
                    static_kernel_name(fn_name),
                    wall_us,
                    super::current_layer(),
                );
            }
            Ok(())
        }

        pub fn dispatch_batch(
            &self,
            encode: impl FnOnce(&mut CommandBatch<'_>) -> Result<()>,
        ) -> Result<()> {
            let trace_enabled = self.trace_dispatch;
            let t0 = if trace_enabled { Some(Instant::now()) } else { None };

            let cmd = self.inner.queue.new_command_buffer();
            let mut batch = CommandBatch { ctx: self, cmd };
            encode(&mut batch)?;
            let CommandBatch { cmd, .. } = batch;
            cmd.commit();
            if trace_enabled {
                self.stats.commits.fetch_add(1, Ordering::Relaxed);
            }
            cmd.wait_until_completed();

            if let Some(t0) = t0 {
                let wall_us = t0.elapsed().as_micros() as u64;
                // dispatch_batch is used for mla_decode_and_o_proj_metal
                self.trace.record("dispatch_batch", wall_us, super::current_layer());
            }
            Ok(())
        }
    }

    impl CommandBatch<'_> {
        pub fn dispatch_threads(
            &mut self,
            fn_name: &str,
            grid: (u32, u32, u32),
            tg: (u32, u32, u32),
            encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
        ) -> Result<()> {
            let pipe = self.ctx.pipeline(fn_name)?;
            let enc = self.cmd.new_compute_command_encoder();
            enc.set_label(fn_name);
            enc.set_compute_pipeline_state(&pipe);
            encode(enc);
            enc.dispatch_threads(
                MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
                MTLSize::new(tg.0 as u64, tg.1 as u64, tg.2 as u64),
            );
            enc.end_encoding();
            Ok(())
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
    /// TCB-internal trace mode (parsed once from `DISMANTLE_TCB_TRACE` at
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
            let raw = std::env::var("DISMANTLE_TCB_TRACE");
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
                    "[dismantle] DISMANTLE_TCB_TRACE={:?} → mode={}",
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

    /// Set `DISMANTLE_TCB_TRACE=cpu` for per-kernel CPU encoding timing
    /// (records pipeline-lookup + encode wall time per call, plus a
    /// `tcb_commit` total for the whole CB).
    ///
    /// Set `DISMANTLE_TCB_TRACE=gpu` for per-kernel GPU-side timing
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
        /// v2.2.0-L7: live in `ProdCbGpu` mode. `None` in other modes or
        /// when the device doesn't support the timestamp counter set.
        prod_cb_tracer: Option<ProdCbTracer>,
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
    }

    impl<'ctx> TokenCommandBuffer<'ctx> {
        pub fn new(ctx: &'ctx MetalContext) -> Self {
            let cmd = ctx.inner.queue.new_command_buffer().to_owned();
            let mode = TcbTraceMode::from_env();
            let prod_cb_tracer = if mode == TcbTraceMode::ProdCbGpu {
                ProdCbTracer::try_new(&ctx.inner.device)
            } else {
                None
            };
            Self {
                ctx,
                cmd: Some(cmd),
                mode,
                tcb_samples: Vec::new(),
                prod_cb_tracer,
                concurrent_encoder: None,
                dispatch_count: 0,
            }
        }

        /// Return the number of compute dispatches encoded so far.
        /// Valid both before and after `commit_and_wait`.
        pub fn dispatch_count(&self) -> usize {
            self.dispatch_count
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
            let enc =
                cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
            enc.set_label("concurrent_group");
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
            // P0.1: if a concurrent group is active, record into its shared
            // encoder. Only set under Off/CpuEncode modes by
            // `begin_concurrent_group`, so the Split/Prod branches below
            // remain reachable for normal (non-grouped) dispatches.
            if self.concurrent_encoder.is_some() {
                let t0 = if self.mode == TcbTraceMode::CpuEncode {
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
            enc.set_label(fn_name);
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
            let enc = if let Some(p) = pair_index {
                let pass = ::metal::ComputePassDescriptor::new();
                let attachments = pass.sample_buffer_attachments();
                let att = ::metal::ComputePassSampleBufferAttachmentDescriptor::new();
                att.set_sample_buffer(&tracer.sample_buf);
                att.set_start_of_encoder_sample_index((p * 2) as u64);
                att.set_end_of_encoder_sample_index((p * 2 + 1) as u64);
                attachments.set_object_at(0, Some(&att));
                cmd.compute_command_encoder_with_descriptor(pass)
            } else {
                cmd.new_compute_command_encoder()
            };
            enc.set_label(fn_name);
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
            let pipe = self.ctx.pipeline(fn_name)?;
            let enc = dedicated.new_compute_command_encoder();
            enc.set_label(fn_name);
            enc.set_compute_pipeline_state(&pipe);
            encode(enc);
            enc.dispatch_threads(
                MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
                MTLSize::new(tg.0 as u64, tg.1 as u64, tg.2 as u64),
            );
            enc.end_encoding();
            let cpu_us = t0_cpu.elapsed().as_micros() as u64;
            dedicated.commit();
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
                let blit = dedicated.new_blit_command_encoder();
                blit.copy_from_buffer(src, src_offset, dst, dst_offset, size);
                blit.end_encoding();
                dedicated.commit();
                dedicated.wait_until_completed();
                return Ok(());
            }
            let cmd = self
                .cmd
                .as_ref()
                .ok_or_else(|| Error::Metal("TokenCommandBuffer already committed".into()))?;
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_buffer(src, src_offset, dst, dst_offset, size);
            blit.end_encoding();
            Ok(())
        }

        /// Commit the command buffer and block until the GPU finishes.
        /// Consumes self; subsequent dispatch calls would fail.
        pub fn commit_and_wait(mut self) -> Result<()> {
            if let Some(cmd) = self.cmd.take() {
                self.flush_and_commit(cmd);
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
            cmd.wait_until_completed();
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
                    if let Some(tracer) = self.prod_cb_tracer.as_ref() {
                        for s in tracer.drain() {
                            self.ctx.trace.samples.lock().push(s);
                        }
                    }
                    // Any out-of-capacity dispatches were pushed straight
                    // to `tcb_samples` with gpu_us=None.
                    for s in self.tcb_samples.drain(..) {
                        self.ctx.trace.samples.lock().push(s);
                    }
                }
            }
        }
    }

    impl Drop for TokenCommandBuffer<'_> {
        fn drop(&mut self) {
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
    /// still compile dismantle-core (running it requires a Mac).
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
pub use imp::CommandBatch;

pub mod argbuf;
pub use argbuf::{ArgLayout, KernelArgBuffer};

pub mod decode_arena;
pub use decode_arena::DecodeArena;

pub mod dense_decode_arena;
pub use dense_decode_arena::DenseDecodeArena;
