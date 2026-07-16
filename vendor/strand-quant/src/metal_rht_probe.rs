//! Experimental Metal RHT probe. This module is never selected by an encoder.
//! Runtime activation remains false until a strict device receipt is promoted.

#![allow(unsafe_code)]

use metal::{Buffer, CommandQueue, CompileOptions, Device, MTLResourceOptions, MTLSize};

pub const METAL_RHT_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct RhtParams {
    ulong seed;
    uint in_features;
    uint axis;
};

inline ulong splitmix64(thread ulong &state) {
    state += 0x9E3779B97F4A7C15ul;
    ulong z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    return z ^ (z >> 31);
}

inline float rht_sign(ulong seed, ulong index) {
    ulong state = seed ^ (index * 0x9E3779B97F4A7C15ul);
    return ((splitmix64(state) >> 63) & 1ul) == 0ul ? 1.0f : -1.0f;
}

kernel void rht_forward_256(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant RhtParams &params [[buffer(2)]],
    threadgroup float *values [[threadgroup(0)]],
    uint lane [[thread_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
    uint index = block * 256u + lane;
    ulong sign_index = params.axis == 0u
        ? ulong(index)
        : ulong(index % params.in_features);
    values[lane] = input[index] * rht_sign(params.seed, sign_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1u; stride < 256u; stride <<= 1u) {
        uint pair_base = (lane / (stride << 1u)) * (stride << 1u);
        uint offset = lane & (stride - 1u);
        float lo = values[pair_base + offset];
        float hi = values[pair_base + offset + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[lane] = (lane & stride) == 0u ? lo + hi : lo - hi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    output[index] = values[lane] * 0.0625f;
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RhtAxis {
    Rows,
    Cols,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RhtParams {
    seed: u64,
    in_features: u32,
    axis: u32,
}

pub struct MetalRhtProbe {
    device: Device,
    queue: CommandQueue,
    pipeline: metal::ComputePipelineState,
}

impl MetalRhtProbe {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no system Metal device")?;
        let library = device
            .new_library_with_source(METAL_RHT_SOURCE, &CompileOptions::new())
            .map_err(|error| format!("compile Metal RHT probe: {error}"))?;
        let function = library
            .get_function("rht_forward_256", None)
            .map_err(|error| format!("load Metal RHT probe kernel: {error}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| format!("create Metal RHT probe pipeline: {error}"))?;
        if pipeline.max_total_threads_per_threadgroup() < 256 {
            return Err("Metal device cannot dispatch a 256-thread RHT group".into());
        }
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pipeline,
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn forward(
        &self,
        input: &[f32],
        seed: u64,
        in_features: usize,
        axis: RhtAxis,
    ) -> Result<Vec<f32>, String> {
        if input.is_empty()
            || input.len() % 256 != 0
            || in_features == 0
            || in_features % 256 != 0
            || input.len() % in_features != 0
        {
            return Err(
                "Metal RHT probe requires non-empty 256-aligned rows and tensor length".into(),
            );
        }
        let input_buffer = self.upload(input);
        let output_buffer = self.device.new_buffer(
            (input.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = [RhtParams {
            seed,
            in_features: u32::try_from(in_features).map_err(|_| "in_features does not fit u32")?,
            axis: match axis {
                RhtAxis::Rows => 0,
                RhtAxis::Cols => 1,
            },
        }];
        let params_buffer = self.upload(&params);
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&params_buffer), 0);
        encoder.set_threadgroup_memory_length(0, (256 * std::mem::size_of::<f32>()) as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (input.len() / 256) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        let pointer = output_buffer.contents() as *const f32;
        if pointer.is_null() {
            return Err("Metal RHT output buffer is null".into());
        }
        Ok(unsafe { std::slice::from_raw_parts(pointer, input.len()) }.to_vec())
    }

    fn upload<T: Copy>(&self, values: &[T]) -> Buffer {
        let byte_len = values.len() * std::mem::size_of::<T>();
        let buffer = self.device.new_buffer(
            byte_len.max(4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                buffer.contents().cast::<u8>(),
                byte_len,
            );
        }
        buffer
    }
}
