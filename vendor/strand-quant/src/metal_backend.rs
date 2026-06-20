
#![allow(unsafe_code)]

#![allow(clippy::upper_case_acronyms)]

use metal::{
    Buffer, CommandQueue, CompileOptions, Device, MTLResourceOptions, MTLSize, NSUInteger,
};

use crate::encode::SUB_BLOCK;
use crate::gpu_types::{BlockParams, GpuViterbiResult};

const VITERBI_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BlockParams {
    uint block_len;
    uint num_states;
    uint k_bits;
    uint n_sub;
    uint sub_size;
    uint max_block_len;   // padded stride for weights[] and back_buf[] rows
    uint _pad0;
    uint _pad1;
};

// Viterbi forward sweep with threadgroup-local cost vectors.
// Buffers:
//   0: weights     [n_blocks * max_block_len]               f32
//   1: levels_flat [n_blocks * n_sub * num_states]           f32
//   2: back_buf    [n_blocks * max_block_len * num_states]   u32  (out)
//   3: params_buf  [n_blocks]                               BlockParams
//   4: final_cost  [n_blocks * num_states]                   f32  (out)
// Threadgroup:
//   0: tg_cost_cur [num_states] f32
//   1: tg_cost_nxt [num_states] f32
kernel void viterbi_tg(
    device const float*      weights     [[ buffer(0) ]],
    device const float*      levels_flat [[ buffer(1) ]],
    device uint*             back_buf    [[ buffer(2) ]],
    constant BlockParams*    params_buf  [[ buffer(3) ]],
    device float*            final_cost  [[ buffer(4) ]],
    threadgroup float*       tg_cur      [[ threadgroup(0) ]],
    threadgroup float*       tg_nxt      [[ threadgroup(1) ]],
    uint ns       [[ thread_index_in_threadgroup ]],
    uint block_id [[ threadgroup_position_in_grid ]]
) {
    constant BlockParams& p = params_buf[block_id];

    const uint num_states    = p.num_states;
    const uint k_bits        = p.k_bits;
    const uint block_len     = p.block_len;
    const uint n_sub         = p.n_sub;
    const uint sub_size      = p.sub_size;
    const uint max_block_len = p.max_block_len;
    const uint num_inputs    = 1u << k_bits;

    // num_states is always a power of two (2^L).
    const uint l_bits = 31u - clz(num_states);
    const uint lk     = l_bits - k_bits;

    const uint w_off    = block_id * max_block_len;
    const uint lv_off   = block_id * n_sub * num_states;
    const uint back_off = block_id * max_block_len * num_states;
    const uint fc_off   = block_id * num_states;

    // Out-of-range threads (threadgroup size > num_states): mark as INF.
    if (ns >= num_states) {
        tg_cur[ns] = INFINITY;
        tg_nxt[ns] = INFINITY;
        return;
    }

    // Free start: all states begin at cost 0.
    tg_cur[ns] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread handles exactly one destination state `ns`.
    // Predecessors s satisfy ns = ((s << k) | inp) & state_mask
    // => s_lo = ns >> k, s = s_lo | (s_hi << lk) for s_hi in [0, 2^k).
    const uint s_lo = ns >> k_bits;

    for (uint step = 0u; step < block_len; step++) {
        float target  = weights[w_off + step];
        uint  sub     = step / sub_size;
        uint  lv_base = lv_off + sub * num_states;
        float lvl     = levels_flat[lv_base + ns];

        float best_cost = INFINITY;
        uint  best_s    = 0u;

        for (uint s_hi = 0u; s_hi < num_inputs; s_hi++) {
            uint  s = s_lo | (s_hi << lk);
            float c = tg_cur[s];
            if (c < INFINITY) {
                float d  = target - lvl;
                float nc = c + d * d;
                if (nc < best_cost) {
                    best_cost = nc;
                    best_s    = s;
                }
            }
        }

        tg_nxt[ns] = best_cost;
        back_buf[back_off + step * num_states + ns] = best_s;

        // All threads must finish reading tg_cur and writing tg_nxt before swap.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        tg_cur[ns] = tg_nxt[ns];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    final_cost[fc_off + ns] = tg_cur[ns];
}
"#;

pub struct MetalViterbi {
    device: Device,
    queue: CommandQueue,
    pipeline: metal::ComputePipelineState,
    
    max_threads: usize,
}

impl MetalViterbi {
    
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;

        let opts = CompileOptions::new();
        let lib = match device.new_library_with_source(VITERBI_MSL, &opts) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[strand-quant] Metal shader compile error: {e}");
                return None;
            }
        };

        let func = match lib.get_function("viterbi_tg", None) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[strand-quant] Metal function lookup error: {e}");
                return None;
            }
        };

        let pipeline = match device.new_compute_pipeline_state_with_function(&func) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[strand-quant] Metal pipeline error: {e}");
                return None;
            }
        };

        let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
        let queue = device.new_command_queue();

        eprintln!(
            "[strand-quant] Metal GPU ready: {} (max_threads_per_tg={})",
            device.name(),
            max_threads
        );

        Some(Self { device, queue, pipeline, max_threads })
    }

    pub fn run_blocks(
        &self,
        all_weights: &[f32],
        sub_levels_all: &[f32],
        block_lens: &[usize],
        max_block_len: usize,
        num_states: usize,
        k_bits: u32,
    ) -> Option<GpuViterbiResult> {
        if all_weights.is_empty() {
            return Some(GpuViterbiResult {
                back_flat: Vec::new(),
                final_cost: Vec::new(),
                max_block_len,
            });
        }
        if num_states > self.max_threads {
            return None;
        }

        let n_blocks = block_lens.len();
        let n_sub = max_block_len.div_ceil(SUB_BLOCK);

        let weights_padded_len = n_blocks * max_block_len;
        let mut weights_padded = vec![0.0f32; weights_padded_len];
        {
            let mut src_off = 0;
            for (bi, &blen) in block_lens.iter().enumerate() {
                let dst = bi * max_block_len;
                weights_padded[dst..dst + blen]
                    .copy_from_slice(&all_weights[src_off..src_off + blen]);
                src_off += blen;
            }
        }
        let w_buf = self.upload(&weights_padded);
        let lv_buf = self.upload(sub_levels_all);

        let back_len = n_blocks * max_block_len * num_states;
        let back_buf = self.alloc_shared(back_len * std::mem::size_of::<u32>());

        let params: Vec<BlockParams> = block_lens
            .iter()
            .map(|&blen| BlockParams {
                block_len: blen as u32,
                num_states: num_states as u32,
                k_bits,
                n_sub: n_sub as u32,
                sub_size: SUB_BLOCK as u32,
                max_block_len: max_block_len as u32,
                _pad0: 0,
                _pad1: 0,
            })
            .collect();
        let params_buf = self.upload(&params);

        let fc_len = n_blocks * num_states;
        let fc_buf = self.alloc_shared(fc_len * std::mem::size_of::<f32>());

        let cmd_buf = self.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&lv_buf), 0);
        enc.set_buffer(2, Some(&back_buf), 0);
        enc.set_buffer(3, Some(&params_buf), 0);
        enc.set_buffer(4, Some(&fc_buf), 0);

        let tg_floats = num_states * std::mem::size_of::<f32>();
        enc.set_threadgroup_memory_length(0, tg_floats as NSUInteger);
        enc.set_threadgroup_memory_length(1, tg_floats as NSUInteger);

        let tpg = MTLSize { width: num_states as NSUInteger, height: 1, depth: 1 };
        let groups = MTLSize { width: n_blocks as NSUInteger, height: 1, depth: 1 };
        enc.dispatch_thread_groups(groups, tpg);
        enc.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let back_flat = self.read_u32(&back_buf, back_len)?;
        let final_cost = self.read_f32(&fc_buf, fc_len)?;

        Some(GpuViterbiResult { back_flat, final_cost, max_block_len })
    }

    fn upload<T: Copy>(&self, data: &[T]) -> Buffer {
        let byte_len = data.len() * std::mem::size_of::<T>();
        let buf = self
            .device
            .new_buffer(byte_len.max(4) as NSUInteger, MTLResourceOptions::StorageModeShared);
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buf.contents() as *mut u8,
                byte_len,
            );
        }
        buf
    }

    fn alloc_shared(&self, byte_len: usize) -> Buffer {
        self.device
            .new_buffer(byte_len.max(4) as NSUInteger, MTLResourceOptions::StorageModeShared)
    }

    fn read_u32(&self, buf: &Buffer, len: usize) -> Option<Vec<u32>> {
        let ptr = buf.contents() as *const u32;
        if ptr.is_null() { return None; }
        
        Some(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
    }

    fn read_f32(&self, buf: &Buffer, len: usize) -> Option<Vec<f32>> {
        let ptr = buf.contents() as *const f32;
        if ptr.is_null() { return None; }
        
        Some(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
    }
}
