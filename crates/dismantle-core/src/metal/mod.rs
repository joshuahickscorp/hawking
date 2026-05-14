//! Metal device, command queues, shader cache.
//!
//! Pure runtime layer — no model knowledge. Owns the
//! `MTLDevice`, holds compiled shader pipelines, and exposes a
//! command-buffer abstraction the rest of the engine talks to.
//!
//! On non-macOS targets every constructor returns
//! `Error::Metal("metal unavailable on this platform")`; the engine
//! still compiles so dev tooling (gguf-cli, tests, schema-only checks)
//! works on Linux CI.

use crate::{Error, Result};
use std::sync::Arc;

/// Embedded shader sources. Compiled at runtime via
/// `MTLDevice::newLibraryWithSource:` — shipping a single binary with
/// no `metallib` artifact in tree means contributors don't need
/// xcrun to build.
pub const SHADER_COMMON: &str = include_str!("../../shaders/common.metal");
pub const SHADER_QUANT: &str = include_str!("../../shaders/quant.metal");
pub const SHADER_MOE: &str = include_str!("../../shaders/moe.metal");
pub const SHADER_ATTN: &str = include_str!("../../shaders/attn.metal");
pub const SHADER_SAMPLE: &str = include_str!("../../shaders/sample.metal");
pub const SHADER_MATMUL: &str = include_str!("../../shaders/matmul.metal");

/// Concatenation of all shader sources for a single library compile.
/// Cheaper than five compile units; lets common helpers be shared.
pub fn all_shader_sources() -> String {
    [
        SHADER_COMMON,
        SHADER_QUANT,
        SHADER_MOE,
        SHADER_ATTN,
        SHADER_SAMPLE,
        SHADER_MATMUL,
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
}

/// Thread-local current-layer index. Set/cleared by the forward pass
/// around each transformer layer so dispatch timing can be attributed
/// to a layer without touching the kernel API.
///
/// Exposed as free functions rather than on MetalContext because the
/// caller (`deepseek_v2::forward_token_final_norm`) runs on whatever
/// thread calls `generate` — not on the GPU thread.
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
        Buffer, CommandBufferRef, CommandQueue, ComputePipelineState, Device, Library,
        MTLResourceOptions, MTLSize,
    };
    use metal::objc::{msg_send, sel, sel_impl};

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
        // CFTimeInterval is `double` (f64) — seconds since absolute reference.
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
            "mla_decode_kernel_batched" => "mla_decode_kernel_batched",
            "mla_decode_kernel_batched_slots" => "mla_decode_kernel_batched_slots",
            "moe_topk_gate" => "moe_topk_gate",
            "moe_gather_combine" => "moe_gather_combine",
            "moe_batched_gemm_q4" => "moe_batched_gemm_q4",
            "moe_batched_gemm_q6_k" => "moe_batched_gemm_q6_k",
            "moe_batched_gemm_q8_0" => "moe_batched_gemm_q8_0",
            "moe_batched_silu_mul" => "moe_batched_silu_mul",
            "moe_block_fused_q4_one" => "moe_block_fused_q4_one",
            "moe_block_fused_q4_topk" => "moe_block_fused_q4_topk",
            "moe_block_fused_v2lite" => "moe_block_fused_v2lite",
            "moe_block_fused_v2lite_indexed" => "moe_block_fused_v2lite_indexed",
            "moe_block_two_stage_intermediate" => "moe_block_two_stage_intermediate",
            "moe_block_two_stage_output" => "moe_block_two_stage_output",
            "moe_route_accumulate" => "moe_route_accumulate",
            "sample_argmax_f32" => "sample_argmax_f32",
            // attn / rope / embed kernels
            "rope_inplace" => "rope_inplace",
            "embed_lookup" => "embed_lookup",
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
            // v0.8.1-v0.8.2 Phase 7 f16 bridge kernels
            "rmsnorm_gemv_f16_attn_pinned" => "rmsnorm_gemv_f16_attn_pinned",
            "rmsnorm_gemv_q4k_pair_f16" => "rmsnorm_gemv_q4k_pair_f16",
            // v0.5.9 fp16 activation kernels
            "gemv_f32_attn_f16" => "gemv_f32_attn_f16",
            "gemv_f32_moe_f16" => "gemv_f32_moe_f16",
            "add_inplace_f16" => "add_inplace_f16",
            "softmax_f16" => "softmax_f16",
            "layer_norm_f16" => "layer_norm_f16",
            // v1.1.0-X simdgroup LM-head
            "gemv_f16_simdmat" => "gemv_f16_simdmat",
            "gemv_simdgroup_f32" => "gemv_simdgroup_f32",
            // v0.5.10 fp16 Q-format kernels
            "gemm_q4_k_m_fused_f16" => "gemm_q4_k_m_fused_f16",
            "moe_grouped_gemm_q4_f16" => "moe_grouped_gemm_q4_f16",
            "dequant_q6_k_f16" => "dequant_q6_k_f16",
            // v1.1.1 / v2.1.0 — T1.1 audit closed 22 names previously
            // bucketed as "other" (incl. the post-T2.1 default MoE Q4_K
            // v2t_gu_v2 kernel itself — biggest attribution miss).
            "moe_batched_gemm_q4_indexed_v2t_gu" => "moe_batched_gemm_q4_indexed_v2t_gu",
            "moe_batched_gemm_q4_indexed_v2t_gu_v2" => "moe_batched_gemm_q4_indexed_v2t_gu_v2",
            "moe_batched_gemm_q8_0_indexed_v2t" => "moe_batched_gemm_q8_0_indexed_v2t",
            "moe_batched_gemm_q5_0_indexed_v2t" => "moe_batched_gemm_q5_0_indexed_v2t",
            "moe_batched_gemm_q6_k_indexed_v2t" => "moe_batched_gemm_q6_k_indexed_v2t",
            "gemm_q3_k_fused_v2" => "gemm_q3_k_fused_v2",
            "gemm_q4_k_m_simdmat" => "gemm_q4_k_m_simdmat",
            "gemm_q4_k_m_v3_8r" => "gemm_q4_k_m_v3_8r",
            "gemm_q4_k_m_v3_dual" => "gemm_q4_k_m_v3_dual",
            "gemm_q4_k_m_v3_llama" => "gemm_q4_k_m_v3_llama",
            "gemv_f16_f16in" => "gemv_f16_f16in",
            "kv_append_f16" => "kv_append_f16",
            "kv_append_f32" => "kv_append_f32",
            "mla_decode_kernel_batched_slots_f16" => "mla_decode_kernel_batched_slots_f16",
            "mla_decode_kernel_f16" => "mla_decode_kernel_f16",
            "rmsnorm_f16_to_f32" => "rmsnorm_f16_to_f32",
            "rmsnorm_f32" => "rmsnorm_f32",
            "rmsnorm_f32_to_f16" => "rmsnorm_f32_to_f16",
            "rmsnorm_gemv_f16w_attn_pinned" => "rmsnorm_gemv_f16w_attn_pinned",
            "rope_q_f32_inplace" => "rope_q_f32_inplace",
            "rope_slice_f32_inplace" => "rope_slice_f32_inplace",
            "cast_f32_to_f16" => "cast_f32_to_f16",
            "embed_lookup_f32" => "embed_lookup_f32",
            "flash_attn_decode_kernel" => "flash_attn_decode_kernel",
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
        pub fn library(&self) -> &Library {
            &self.inner.library
        }
        pub fn device_name(&self) -> String {
            self.inner.device.name().to_string()
        }

        /// Look up — or create + cache — a compute pipeline for a
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

        /// Write `bytes` into an existing shared buffer. The buffer must
        /// have been allocated with `new_buffer` and have capacity ≥ `bytes.len()`.
        /// On unified-memory Apple Silicon this is a plain `memcpy` — no GPU
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
    /// - **Off** (env var unset or `=0`): zero-overhead default; no per-kernel
    ///   samples, no tcb_commit record.
    /// - **CpuEncode** (env var `=1` or `=cpu` or any non-recognized value):
    ///   the historical behavior — records CPU-side encoding wall time per
    ///   dispatch + a `tcb_commit` total. Tells you nothing about GPU compute
    ///   distribution inside the TCB; only useful for dispatch-overhead audits.
    /// - **SplitCbGpu** (env var `=gpu`): each dispatch lands in its OWN
    ///   command buffer that is committed + waited immediately, so
    ///   `MTLCommandBuffer::gpu_start_time`/`gpu_end_time` can be read
    ///   directly. Populates `DispatchSample::gpu_us` with the per-kernel
    ///   GPU compute time. **Diagnostic mode**: kills the TCB amortization
    ///   benefit (each dispatch pays a full commit/sync), so dec_tps drops
    ///   substantially. Use only to attribute where decode time goes.
    ///   Numerical output is bit-identical to Off mode (same kernels, same
    ///   args, same order — only commit granularity differs).
    #[derive(Copy, Clone, PartialEq, Eq)]
    pub enum TcbTraceMode {
        Off,
        CpuEncode,
        SplitCbGpu,
    }

    impl TcbTraceMode {
        fn from_env() -> Self {
            let raw = std::env::var("DISMANTLE_TCB_TRACE");
            let mode = match raw.as_deref() {
                Err(_) => Self::Off,
                Ok("") | Ok("0") => Self::Off,
                Ok(s) if s.eq_ignore_ascii_case("gpu") => Self::SplitCbGpu,
                Ok(_) => Self::CpuEncode,
            };
            // One-shot diagnostic on the first TCB construction, gated so the
            // hot path stays clean. Removed after T1.1 validation lands.
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "[dismantle] DISMANTLE_TCB_TRACE={:?} → mode={}",
                    raw.as_deref().unwrap_or("(unset)"),
                    match mode {
                        Self::Off => "Off",
                        Self::CpuEncode => "CpuEncode",
                        Self::SplitCbGpu => "SplitCbGpu",
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
    /// be read directly). Populates `DispatchSample::gpu_us`. Slower —
    /// diagnostic mode only. See `TcbTraceMode` for details.
    pub struct TokenCommandBuffer<'ctx> {
        pub ctx: &'ctx MetalContext,
        /// `None` after `commit_and_wait` so the Drop impl knows not to re-commit.
        cmd: Option<metal::CommandBuffer>,
        /// TCB-internal trace mode; resolved once at construction.
        mode: TcbTraceMode,
        /// Accumulated per-dispatch samples; only populated when `mode` is on.
        tcb_samples: Vec<super::DispatchSample>,
    }

    impl<'ctx> TokenCommandBuffer<'ctx> {
        pub fn new(ctx: &'ctx MetalContext) -> Self {
            let cmd = ctx.inner.queue.new_command_buffer().to_owned();
            let mode = TcbTraceMode::from_env();
            Self { ctx, cmd: Some(cmd), mode, tcb_samples: Vec::new() }
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
            if self.mode == TcbTraceMode::SplitCbGpu {
                return self.dispatch_threads_split_cb(fn_name, grid, tg, encode);
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
            // GPUStartTime / GPUEndTime are not wrapped by metal 0.29 — go
            // direct via objc msg_send. Both return CFTimeInterval (f64
            // seconds since an absolute reference); their difference is the
            // GPU compute duration. Safe because we just waited.
            let gpu_us = unsafe { cb_gpu_duration_us(&dedicated) };
            self.tcb_samples.push(super::DispatchSample {
                kernel_name: static_kernel_name(fn_name),
                wall_us: cpu_us,
                layer_hint: super::current_layer(),
                gpu_us: Some(gpu_us),
            });
            Ok(())
        }

        /// Encode a GPU-side buffer copy into the pending command buffer.
        ///
        /// Uses a `MTLBlitCommandEncoder` — very cheap (~100 ns; a plain GPU
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
                    // no aggregate `tcb_commit` in split mode — the GPU times
                    // already sum to the decoded total.
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
