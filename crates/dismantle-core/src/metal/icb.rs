//! Indirect Command Buffer (ICB) wrapper — production-scale infrastructure
//! for the named "wire forward-pass into ICB" swing (see
//! `memory/icb_poc_2026_05_24.md`).
//!
//! Scope: this module is INTERNAL infrastructure, not yet wired into any
//! forward path. The next attended session reviews the production-scale
//! measurement (see `crates/dismantle-core/tests/icb_production_scale.rs`)
//! and decides whether to wire `forward_token_greedy_tcb`'s dispatches into
//! this wrapper.
//!
//! API parallels `TokenCommandBuffer::dispatch_threads`, except:
//! - The wrapper pre-allocates an ICB of `capacity` commands and records
//!   into slot `cmd_idx`, which advances per call.
//! - The bind closure receives an `&IndirectComputeCommandRef` (not an
//!   `&ComputeCommandEncoderRef`). The relevant methods (`set_kernel_buffer`,
//!   `set_threadgroup_memory_length`) are named slightly differently and
//!   listed in metal-rs 0.29's `indirect_encoder.rs`.
//! - PSOs MUST be created with `set_support_indirect_command_buffers(true)`;
//!   `MetalContext::pipeline()` does NOT set this flag, so this wrapper
//!   maintains its own parallel PSO cache (`icb_pipelines`).
//! - `execute_and_wait` submits a one-encoder, one-CB execution that fires
//!   all recorded commands via `executeCommandsInBuffer:withRange:`. The
//!   selector is unbound on `ComputeCommandEncoderRef` in metal-rs 0.29,
//!   so a `msg_send!` wrapper is used (same pattern as the POC).
//!
//! Caveats:
//! - The wrapper does NOT support per-resource `use_resource` declarations
//!   from outside; the caller passes resource sets via
//!   `mark_resources_used` before `execute_and_wait`. ICB requires the
//!   encoder to mark every buffer the ICB will touch.
//! - No tracing hooks (DispatchSample) — measurement is the test's job.
//! - No commit-and-wait safety net via Drop: the caller MUST call
//!   `execute_and_wait` or `discard` explicitly.

#![cfg(target_os = "macos")]

use crate::{Error, Result};
use std::collections::HashMap;

use metal::{
    Buffer, ComputePipelineDescriptor, ComputePipelineState, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, MTLIndirectCommandType, MTLResourceOptions,
    MTLResourceUsage, MTLSize, NSRange,
};
use metal::objc::{msg_send, sel, sel_impl};

use super::MetalContext;

/// metal-rs 0.29 exposes `executeCommandsInBuffer:withRange:` only on
/// `RenderCommandEncoderRef`. The Apple selector exists on compute too
/// (`MTLComputeCommandEncoder::executeCommandsInBuffer:withRange:`).
unsafe fn compute_execute_commands_in_buffer(
    encoder: &metal::ComputeCommandEncoderRef,
    icb: &metal::IndirectCommandBufferRef,
    range: NSRange,
) {
    let _: () = msg_send![encoder, executeCommandsInBuffer:icb withRange:range];
}

/// One resource the ICB will touch, with its declared usage. Captured
/// at `mark_resource_used` time and replayed via `enc.use_resource()` at
/// `execute_and_wait` time.
struct ResourceUse {
    buf: Buffer,
    usage: MTLResourceUsage,
}

pub struct IndirectTokenCommandBuffer<'ctx> {
    ctx: &'ctx MetalContext,
    icb: IndirectCommandBuffer,
    /// Index of the next dispatch slot to record into. Advances on each
    /// `dispatch_threads`. Must not exceed `capacity`.
    cmd_idx: usize,
    capacity: usize,
    /// ICB-capable PSOs, keyed by kernel name. Distinct from
    /// `MetalContext::pipeline()`'s cache because those PSOs lack the
    /// `support_indirect_command_buffers` flag and crash on bind.
    icb_pipelines: HashMap<String, ComputePipelineState>,
    /// Resources to `use_resource()` at execute time. The caller declares
    /// them once via `mark_resource_used`.
    resources: Vec<ResourceUse>,
}

impl<'ctx> IndirectTokenCommandBuffer<'ctx> {
    /// Allocate an ICB sized for `capacity` compute commands.
    /// `max_kernel_buffer_bind_count` is set generously (8) to cover the
    /// fattest production kernels (e.g. fused W4A8 ops with weights +
    /// scales + activations + activation-scales + output + n).
    pub fn new(ctx: &'ctx MetalContext, capacity: usize) -> Result<Self> {
        let desc = IndirectCommandBufferDescriptor::new();
        desc.set_command_types(MTLIndirectCommandType::ConcurrentDispatchThreads);
        desc.set_inherit_buffers(false);
        desc.set_inherit_pipeline_state(false);
        desc.set_max_kernel_buffer_bind_count(8);

        let icb = ctx.device().new_indirect_command_buffer_with_descriptor(
            &desc,
            capacity as u64,
            MTLResourceOptions::StorageModeShared,
        );
        if icb.size() == 0 {
            return Err(Error::Metal(
                "IndirectCommandBuffer allocation returned 0 size; device may not support ICB"
                    .into(),
            ));
        }
        Ok(Self {
            ctx,
            icb,
            cmd_idx: 0,
            capacity,
            icb_pipelines: HashMap::new(),
            resources: Vec::new(),
        })
    }

    /// Build (or retrieve cached) ICB-capable PSO for `fn_name`.
    fn ensure_icb_pipeline(&mut self, fn_name: &str) -> Result<ComputePipelineState> {
        if let Some(p) = self.icb_pipelines.get(fn_name) {
            return Ok(p.clone());
        }
        let f = self
            .ctx
            .library()
            .get_function(fn_name, None)
            .map_err(|e| Error::Metal(format!("function lookup {}: {}", fn_name, e)))?;
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&f));
        desc.set_support_indirect_command_buffers(true);
        let pso = self
            .ctx
            .device()
            .new_compute_pipeline_state(&desc)
            .map_err(|e| Error::Metal(format!("PSO build {} (ICB): {}", fn_name, e)))?;
        self.icb_pipelines.insert(fn_name.to_string(), pso.clone());
        Ok(pso)
    }

    /// Record one dispatch into the next ICB slot. Mirrors
    /// `TokenCommandBuffer::dispatch_threads` except the bind closure
    /// operates on `&IndirectComputeCommandRef`.
    pub fn dispatch_threads(
        &mut self,
        fn_name: &str,
        grid: (u32, u32, u32),
        tg: (u32, u32, u32),
        bind: impl FnOnce(&metal::IndirectComputeCommandRef),
    ) -> Result<()> {
        if self.cmd_idx >= self.capacity {
            return Err(Error::Metal(format!(
                "IndirectTokenCommandBuffer capacity exhausted ({} slots used, capacity {})",
                self.cmd_idx, self.capacity
            )));
        }
        let pipe = self.ensure_icb_pipeline(fn_name)?;
        let cmd = self
            .icb
            .indirect_compute_command_at_index(self.cmd_idx as u64);
        cmd.set_compute_pipeline_state(&pipe);
        bind(cmd);
        cmd.concurrent_dispatch_threads(
            MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
            MTLSize::new(tg.0 as u64, tg.1 as u64, tg.2 as u64),
        );
        self.cmd_idx += 1;
        Ok(())
    }

    /// Declare a buffer that the ICB will touch. Required because the ICB
    /// references buffers directly (the wrapping compute encoder must call
    /// `use_resource` so the driver knows the buffers are live).
    pub fn mark_resource_used(&mut self, buf: &Buffer, usage: MTLResourceUsage) {
        self.resources.push(ResourceUse {
            buf: buf.clone(),
            usage,
        });
    }

    /// Number of dispatches recorded so far.
    pub fn len(&self) -> usize {
        self.cmd_idx
    }

    pub fn is_empty(&self) -> bool {
        self.cmd_idx == 0
    }

    /// Submit the ICB for execution and block until complete.
    pub fn execute_and_wait(self) -> Result<()> {
        if self.cmd_idx == 0 {
            return Ok(());
        }
        let cb = self.ctx.queue().new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for r in &self.resources {
            enc.use_resource(&r.buf, r.usage);
        }
        unsafe {
            compute_execute_commands_in_buffer(
                enc,
                &self.icb,
                NSRange {
                    location: 0,
                    length: self.cmd_idx as u64,
                },
            );
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }
}
