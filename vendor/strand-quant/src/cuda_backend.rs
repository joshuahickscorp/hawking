
#![allow(unsafe_code)]

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

use crate::encode::SUB_BLOCK;
use crate::gpu_types::{BlockParams, GpuViterbiResult};

const VITERBI_CUDA_SRC: &str = r#"
// (no <float.h>: nvrtc has no system include path by default, and the only
// sentinel we need is +inf via __int_as_float below — FLT_MAX is unused.)

struct BlockParams {
    unsigned block_len;
    unsigned num_states;
    unsigned k_bits;
    unsigned n_sub;
    unsigned sub_size;
    unsigned max_block_len;
    unsigned _pad0;
    unsigned _pad1;
};

// +inf sentinel — avoids FLT_MAX accumulation overflow. Defined as a macro (not a
// __device__ static const): nvrtc forbids dynamic init of device globals, and
// __int_as_float() is treated as a runtime call. The macro expands at each use site,
// all of which are inside __device__/__global__ functions where the intrinsic is legal.
#define COST_INF (__int_as_float(0x7f800000u))

// Viterbi forward sweep.
// Grid:  (n_blocks, 1, 1)  — one block per trellis block.
// Block: (num_states, 1, 1) — one thread per destination state.
// Shared: 2 * num_states * sizeof(float) — tg_cur and tg_nxt interleaved.
extern "C" __global__ void viterbi_tg(
    const float*         weights,       // [n_blocks * max_block_len]
    const float*         levels_flat,   // [n_blocks * n_sub * num_states]
    unsigned*            back_buf,      // [n_blocks * max_block_len * num_states]
    const BlockParams*   params_buf,    // [n_blocks]
    float*               final_cost     // [n_blocks * num_states]
) {
    extern __shared__ float shared[];
    float* tg_cur = shared;
    float* tg_nxt = shared + blockDim.x;  // blockDim.x == num_states

    const unsigned ns       = threadIdx.x;
    const unsigned block_id = blockIdx.x;
    const BlockParams p     = params_buf[block_id];

    const unsigned num_states    = p.num_states;
    const unsigned k_bits        = p.k_bits;
    const unsigned block_len     = p.block_len;
    const unsigned n_sub         = p.n_sub;
    const unsigned sub_size      = p.sub_size;
    const unsigned max_block_len = p.max_block_len;
    const unsigned num_inputs    = 1u << k_bits;

    // log2(num_states): num_states is always a power of two.
    const unsigned l_bits = 31u - __clz(num_states);
    const unsigned lk     = l_bits - k_bits;  // L - k

    const unsigned w_off    = block_id * max_block_len;
    const unsigned lv_off   = block_id * n_sub * num_states;
    const unsigned back_off = block_id * max_block_len * num_states;
    const unsigned fc_off   = block_id * num_states;

    // Guard: extra threads beyond num_states (shouldn't happen with our dispatch,
    // but be safe in case num_states < warp size and warp overruns).
    if (ns >= num_states) {
        return;
    }

    // Free start: all states at cost 0.
    tg_cur[ns] = 0.0f;
    __syncthreads();

    // Each thread owns exactly one destination state `ns`.
    // Predecessor set: for ns, predecessors s satisfy
    //   ns = ((s << k) | inp) & state_mask
    // => lower (L-k) bits of s = ns >> k
    const unsigned s_lo = ns >> k_bits;

    for (unsigned step = 0u; step < block_len; step++) {
        float target  = weights[w_off + step];
        unsigned sub  = step / sub_size;
        float lvl     = levels_flat[lv_off + sub * num_states + ns];

        float best_cost = COST_INF;
        unsigned best_s = 0u;

        for (unsigned s_hi = 0u; s_hi < num_inputs; s_hi++) {
            unsigned s = s_lo | (s_hi << lk);
            float c = tg_cur[s];
            if (c < COST_INF) {
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

        // Swap cost vectors: all threads must finish writing tg_nxt before anyone
        // reads it in the next step.
        __syncthreads();
        tg_cur[ns] = tg_nxt[ns];
        __syncthreads();
    }

    final_cost[fc_off + ns] = tg_cur[ns];
}
"#;

const MODULE: &str = "strand_viterbi";
const KERNEL: &str = "viterbi_tg";

unsafe impl cudarc::driver::DeviceRepr for BlockParams {}

pub struct CudaViterbi {
    device: Arc<CudaDevice>,
}

impl CudaViterbi {
    
    pub fn new() -> Option<Self> {
        let device = CudaDevice::new(0)
            .map_err(|e| eprintln!("[strand-quant] CUDA device error: {e}"))
            .ok()?;

        let ptx = compile_ptx(VITERBI_CUDA_SRC)
            .map_err(|e| eprintln!("[strand-quant] CUDA compile error: {e}"))
            .ok()?;

        device
            .load_ptx(ptx, MODULE, &[KERNEL])
            .map_err(|e| eprintln!("[strand-quant] CUDA load error: {e}"))
            .ok()?;

        eprintln!("[strand-quant] CUDA GPU ready: device 0");

        Some(Self { device })
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
        
        if num_states > 1024 {
            return None;
        }

        let n_blocks = block_lens.len();
        let n_sub = max_block_len.div_ceil(SUB_BLOCK);

        let mut weights_padded = vec![0.0f32; n_blocks * max_block_len];
        let mut src_off = 0usize;
        for (bi, &blen) in block_lens.iter().enumerate() {
            let dst = bi * max_block_len;
            weights_padded[dst..dst + blen]
                .copy_from_slice(&all_weights[src_off..src_off + blen]);
            src_off += blen;
        }

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

        let d_weights = self.device.htod_sync_copy(&weights_padded).ok()?;
        let d_levels = self.device.htod_sync_copy(sub_levels_all).ok()?;
        let d_params = self.device.htod_sync_copy(&params).ok()?;
        let mut d_back: CudaSlice<u32> =
            self.device.alloc_zeros(n_blocks * max_block_len * num_states).ok()?;
        let mut d_final: CudaSlice<f32> =
            self.device.alloc_zeros(n_blocks * num_states).ok()?;

        let f = self.device.get_func(MODULE, KERNEL)?;
        let shared_bytes = (2 * num_states * std::mem::size_of::<f32>()) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks as u32, 1, 1),
            block_dim: (num_states as u32, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        unsafe {
            f.launch(cfg, (&d_weights, &d_levels, &mut d_back, &d_params, &mut d_final))
        }
        .map_err(|e| eprintln!("[strand-quant] CUDA launch error: {e}"))
        .ok()?;

        let back_flat = self.device.dtoh_sync_copy(&d_back).ok()?;
        let final_cost = self.device.dtoh_sync_copy(&d_final).ok()?;

        Some(GpuViterbiResult { back_flat, final_cost, max_block_len })
    }
}
