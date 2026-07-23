//! Minimal Metal backend: one descriptor-light kernel for the measured bottleneck — the tied
//! vocabulary projection, executed DIRECTLY on the Q8_0 embedding blocks (dequant + f16 round happen
//! in-shader; the weight is never expanded to a dense buffer). One thread computes one vocab logit.
//! Falls back to CPU (`cpu::logits_tied`) when no Metal device is present. The kernel mirrors the CPU
//! op's per-element order, so its logits match the CPU reference.

use crate::{Error, Result};
use metal::{
    ComputePipelineState, Device, MTLResourceOptions, MTLSize, CommandQueue,
};
use std::ffi::c_void;

const KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Tied LM head directly on Q8_0 blocks: out[v] = sum_c f16(d * qs[c]) * x[c].
kernel void q8_0_logits(
    device const uchar* embd    [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      hidden  [[buffer(3)]],
    uint v [[thread_position_in_grid]])
{
    uint blocks_per_row = hidden / 32u;
    uint row_bytes = blocks_per_row * 34u;
    uint base = v * row_bytes;
    float acc = 0.0f;
    uint c = 0u;
    for (uint b = 0u; b < blocks_per_row; ++b) {
        uint bo = base + b * 34u;
        ushort dbits = (ushort)embd[bo] | ((ushort)embd[bo + 1u] << 8);
        half d = as_type<half>(dbits);
        for (uint i = 0u; i < 32u; ++i) {
            char q = (char)embd[bo + 2u + i];
            half hv = half(float(d) * float(q));
            acc += float(hv) * x[c];
            c += 1u;
        }
    }
    out[v] = acc;
}
"#;

pub struct MetalGemv {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    pub device_name: String,
}

impl MetalGemv {
    /// Returns None if no Metal device is available (CPU fallback path is used instead).
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        let opts = metal::CompileOptions::new();
        let lib = device.new_library_with_source(KERNEL_SRC, &opts).ok()?;
        let func = lib.get_function("q8_0_logits", None).ok()?;
        let pipeline = device.new_compute_pipeline_state_with_function(&func).ok()?;
        let device_name = device.name().to_string();
        Some(MetalGemv { device, queue, pipeline, device_name })
    }

    /// Compute the tied vocab logits on the GPU from the raw Q8_0 embedding bytes.
    pub fn logits_q8_0(&self, embd: &[u8], hidden: usize, vocab: usize, x: &[f32]) -> Result<Vec<f32>> {
        let dev = &self.device;
        let opt = MTLResourceOptions::StorageModeShared;
        let embd_buf = dev.new_buffer_with_data(embd.as_ptr() as *const c_void, embd.len() as u64, opt);
        let x_buf = dev.new_buffer_with_data(x.as_ptr() as *const c_void, (x.len() * 4) as u64, opt);
        let out_buf = dev.new_buffer((vocab * 4) as u64, opt);
        let hidden32 = hidden as u32;

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&embd_buf), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_bytes(3, 4, &hidden32 as *const u32 as *const c_void);
        let tg = self.pipeline.max_total_threads_per_threadgroup().min(256) as u64;
        enc.dispatch_threads(MTLSize::new(vocab as u64, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if cb.status() != metal::MTLCommandBufferStatus::Completed {
            return Err(Error::Metal(format!("command buffer status {:?}", cb.status())));
        }

        let ptr = out_buf.contents() as *const f32;
        let out = unsafe { std::slice::from_raw_parts(ptr, vocab) }.to_vec();
        Ok(out)
    }
}
