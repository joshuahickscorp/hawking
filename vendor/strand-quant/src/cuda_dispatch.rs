//! Memory-bounded GPU (CUDA) encode dispatch.
//!
//! WHY THIS MODULE EXISTS
//! ----------------------
//! The original `encode::encode_tensor_with_cuda` precomputes a `Vec<BlockPrep>`
//! for *every* block of the whole tensor up front, and each `BlockPrep` carries a
//! `levels_f32: Vec<f32>` of `n_sub_per_block * num_states` floats. That vector is
//! resident for the entire function (it is built before the batch loop and freed
//! only at return), so for a 235M-param tensor at L=10 it costs ~30 GB of HOST
//! memory PER TENSOR — and `quantize-model` runs ~27 such jobs concurrently in a
//! `std::thread::scope` worker pool, so a handful of large tensors blow the 125 GB
//! cgroup and the process is OOM-killed.
//!
//! The 2026-06-05 fix (commit d446f7b, "chunk back_flat to 512 MB") capped the
//! per-batch device/host *back-buffer* staging, which is correct and preserved
//! here — but it never touched the `preps.levels_f32` term, which is the actual
//! dominant host allocation at scale. This module fixes that by computing the
//! per-block levels (and the staged `weights`/`back_flat` buffers) INSIDE the
//! batch loop, bounded to `batch_size` blocks at a time. Resident `levels`
//! footprint drops from `n_blocks * levels_bytes` (~30 GB) to
//! `batch_size * levels_bytes` (~16 MB).
//!
//! OUTPUT IS BIT-IDENTICAL to the all-at-once path: the Viterbi recursion is
//! per-block (no cross-block state on the GPU — `tg_cur` is reset to 0 at the
//! start of each block in the kernel), the levels for block `b` depend only on
//! block `b`'s `(scale_q, mults)`, and the bit-packing order is unchanged. Only
//! the *staging granularity* changes, never the values. DECODE determinism is
//! unaffected (this is encode-time only).
//!
//! INTEGRATION: this is a free function; the operator replaces the body of
//! `encode::encode_tensor_with_cuda` with a delegate to
//! `crate::cuda_dispatch::encode_tensor_with_cuda` (see the integration plan in
//! the task return). It lives in-crate so it can call the `pub(crate)` scale
//! search helpers and the `cuda_backend`.

#![cfg(feature = "cuda")]

use crate::codebook::{codebook_lut, QUANTILE_SHIFT};
use crate::decode::{eff_scale_q, reconstruct_q};
use crate::encode::{
    choose_scale_q, choose_sub_scales, n_sub_blocks, pack_sub_scales, BlockMeta, EncodeOpts,
    EncodedTensor, SUB_BLOCK, SUB_SCALE_UNITY,
};
use crate::trellis::{push_bits, TrellisConfig};

/// Upper bound on the per-batch back-buffer staging (device `d_back` AND the
/// host `back_flat` copy each cost this). Mirrors the original
/// `MAX_BACK_BYTES_CUDA`. 512 MB sits far under the 24 GB GPU and under the
/// 125 GB host cgroup even with ~27 concurrent worker threads
/// (27 * 512 MB ≈ 13.5 GB of transient back-buffer staging across all threads).
const MAX_BACK_BYTES_CUDA: usize = 512 * 1024 * 1024;

/// Per-batch budget for the staged input `weights` (host `weights_padded` AND
/// device `d_weights`). `batch_size * max_block_len * 4` bytes. Kept well under
/// the back-buffer budget; for L>=8 the back-buffer dominates anyway
/// (back is `num_states`x the weights), so this only bites at tiny L.
const MAX_WEIGHTS_BYTES_CUDA: usize = 128 * 1024 * 1024;

/// Compute the bounded batch size (number of blocks staged per GPU launch).
///
/// Pulled out as a pure function so it is unit-testable without a GPU. Returns
/// the number of blocks such that BOTH the back-buffer and the input-weights
/// staging stay within budget, clamped to `[1, n_blocks]`.
///
/// NOTE: unlike the original (`.max(64)`), the floor here is `1`, not `64`. The
/// `.max(64)` floor in the old code could *raise* a budget-derived batch above
/// the byte cap when a single block was already large — at L=10, k=2,
/// block_len=256 the back-buffer budget gives 512 blocks so the floor never bit,
/// but the floor is an unbounded-memory footgun at higher L (e.g. L=14 ->
/// num_states=16384 -> one block's back-buffer is 16 MB, 64 of them = 1 GB
/// device, which the old `.max(64)` would have forced past a tighter cap). A
/// floor of 1 guarantees the cap is never violated; throughput at the large-L
/// end is bounded by per-block work, not launch count.
pub(crate) fn cuda_batch_size(
    block_len: usize,
    num_states: usize,
    n_blocks: usize,
) -> usize {
    if n_blocks == 0 {
        return 0;
    }
    let back_bytes_per_block = block_len
        .saturating_mul(num_states)
        .saturating_mul(std::mem::size_of::<u32>())
        .max(1);
    let weight_bytes_per_block = block_len
        .saturating_mul(std::mem::size_of::<f32>())
        .max(1);

    let by_back = MAX_BACK_BYTES_CUDA / back_bytes_per_block;
    let by_weights = MAX_WEIGHTS_BYTES_CUDA / weight_bytes_per_block;

    by_back.min(by_weights).max(1).min(n_blocks)
}

/// Per-block scale-search metadata. Cheap to hold for the whole tensor: just an
/// `i32` and a `Vec<u8>` of `n_sub_blocks` (<= block_len/SUB_BLOCK, e.g. 8 u8 at
/// block_len=256). This is ~24-32 bytes/block vs the ~32 KB/block `levels_f32`
/// the old code held — i.e. the part that was actually huge is NO LONGER kept.
struct BlockPlan {
    chunk_offset: usize,
    chunk_len: usize,
    scale_q: i32,
    mults: Vec<u8>,
}

/// Build the `[n_sub_per_block * num_states]` f32 reconstruction levels for ONE
/// block. Identical to the closure in the original `encode_tensor_with_cuda`.
#[inline]
fn build_levels(
    scale_q: i32,
    mults: &[u8],
    lut: &[i32],
    num_states: usize,
    n_sub_per_block: usize,
    q_to_real: f32,
) -> Vec<f32> {
    let mut lv = vec![0.0f32; n_sub_per_block * num_states];
    for (j, &mult) in mults.iter().enumerate() {
        let es = eff_scale_q(scale_q, mult);
        let base = j * num_states;
        for s in 0..num_states {
            lv[base + s] = (reconstruct_q(es, lut[s]) as f32) * q_to_real;
        }
    }
    lv
}

/// Memory-bounded CUDA encode. Drop-in replacement for
/// `encode::encode_tensor_with_cuda` — same signature, same output, bounded host
/// memory. Returns `None` (so the caller falls back to CPU) iff the GPU is
/// unavailable or the kernel rejects the config (e.g. num_states > 1024).
pub(crate) fn encode_tensor_with_cuda(
    weights: &[f32],
    cfg: &TrellisConfig,
    opts: &EncodeOpts,
) -> Option<EncodedTensor> {
    let cu = crate::encode::cuda_viterbi()?;
    let lut = codebook_lut(cfg.l_bits);
    let num_states = cfg.num_states();
    let n_sub_per_block = cfg.block_len.div_ceil(SUB_BLOCK);
    let q_to_real = 1.0f32 / (1u32 << QUANTILE_SHIFT) as f32;

    // --- Plan pass: cheap per-block metadata only (NO levels_f32 held). -------
    let n_blocks = weights.len().div_ceil(cfg.block_len);
    let mut plans: Vec<BlockPlan> = Vec::with_capacity(n_blocks);
    let mut offset = 0usize;
    for chunk in weights.chunks(cfg.block_len) {
        let scale_q = choose_scale_q(chunk, lut, cfg);
        let mults = if opts.adaptive {
            choose_sub_scales(chunk, scale_q, lut, cfg)
        } else {
            vec![SUB_SCALE_UNITY; n_sub_blocks(chunk.len())]
        };
        plans.push(BlockPlan {
            chunk_offset: offset,
            chunk_len: chunk.len(),
            scale_q,
            mults,
        });
        offset += chunk.len();
    }

    let batch_size = cuda_batch_size(cfg.block_len, num_states, n_blocks);

    let input_mask = (1usize << cfg.k_bits) - 1;
    let mut all_paths: Vec<Vec<u32>> = vec![Vec::new(); n_blocks];
    let mut all_init_states: Vec<usize> = vec![0usize; n_blocks];

    // --- Batched GPU launches: levels built per-batch, bounded staging. -------
    let mut bi_base = 0;
    while bi_base < n_blocks {
        let bi_end = (bi_base + batch_size).min(n_blocks);
        let w_start = plans[bi_base].chunk_offset;
        let w_end = if bi_end < n_blocks {
            plans[bi_end].chunk_offset
        } else {
            weights.len()
        };
        let batch_weights = &weights[w_start..w_end];
        let batch_lens: Vec<usize> =
            plans[bi_base..bi_end].iter().map(|p| p.chunk_len).collect();

        // Levels for ONLY this batch's blocks. Peak resident = batch_size *
        // (n_sub_per_block * num_states) f32. At L=10/block_len=256/batch=512:
        // 512 * 8 * 1024 * 4 B ≈ 16 MB (vs ~30 GB for the whole tensor).
        let mut sub_levels: Vec<f32> =
            Vec::with_capacity((bi_end - bi_base) * n_sub_per_block * num_states);
        for p in &plans[bi_base..bi_end] {
            let lv = build_levels(
                p.scale_q,
                &p.mults,
                lut,
                num_states,
                n_sub_per_block,
                q_to_real,
            );
            sub_levels.extend_from_slice(&lv);
        }

        let gpu = cu.run_blocks(
            batch_weights,
            &sub_levels,
            &batch_lens,
            cfg.block_len,
            num_states,
            cfg.k_bits as u32,
        )?;
        let mbl = gpu.max_block_len;

        for (i, plan) in plans[bi_base..bi_end].iter().enumerate() {
            let blen = plan.chunk_len;
            let fc_base = i * num_states;
            let back_base = i * mbl * num_states;
            let terminal = (0..num_states)
                .min_by(|&a, &b| {
                    gpu.final_cost[fc_base + a]
                        .partial_cmp(&gpu.final_cost[fc_base + b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            let mut path = vec![0u32; blen];
            let mut state = terminal;
            for step in (0..blen).rev() {
                path[step] = (state & input_mask) as u32;
                state = gpu.back_flat[back_base + step * num_states + state] as usize;
            }
            all_paths[bi_base + i] = path;
            all_init_states[bi_base + i] = state;
        }
        bi_base = bi_end;
    }

    // --- Emit bits + block metadata (order identical to the original). --------
    let mut bits = Vec::new();
    let mut bit_cursor = 0usize;
    let mut blocks = Vec::new();
    for (bi, plan) in plans.iter().enumerate() {
        for &sym in &all_paths[bi] {
            push_bits(&mut bits, &mut bit_cursor, sym as usize, cfg.k_bits);
        }
        blocks.push(BlockMeta {
            scale_q: plan.scale_q,
            sub_scales: pack_sub_scales(&plan.mults),
            min_base_q: 0,
            mins: Vec::new(),
            init_state: all_init_states[bi] as u32,
            n: plan.chunk_len as u32,
        });
    }

    Some(EncodedTensor {
        bits,
        blocks,
        total: weights.len(),
        has_rht_seed: false,
        tail_biting: false,
        has_affine_min: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- batch-size math: the OOM-prevention invariant -----------------------

    #[test]
    fn batch_never_exceeds_back_buffer_budget() {
        // For a sweep of (block_len, L), the staged back-buffer for one batch
        // must never exceed MAX_BACK_BYTES_CUDA.
        for &block_len in &[64usize, 128, 256, 512] {
            for l in 4u32..=14 {
                let num_states = 1usize << l;
                let n_blocks = 1_000_000; // huge tensor
                let bs = cuda_batch_size(block_len, num_states, n_blocks);
                let back_bytes = bs * block_len * num_states * std::mem::size_of::<u32>();
                assert!(
                    back_bytes <= MAX_BACK_BYTES_CUDA,
                    "L={l} block_len={block_len}: batch={bs} -> back={back_bytes} > cap",
                );
                assert!(bs >= 1, "batch size must be at least 1");
            }
        }
    }

    #[test]
    fn batch_clamped_to_n_blocks_when_small() {
        // A tiny tensor must not over-batch.
        assert_eq!(cuda_batch_size(256, 1024, 3), 3);
        assert_eq!(cuda_batch_size(256, 1024, 0), 0);
        assert_eq!(cuda_batch_size(256, 1024, 1), 1);
    }

    #[test]
    fn batch_matches_original_at_l10_default() {
        // Regression guard: at the standard GPU config (block_len=256, L=10) the
        // original code derived batch=512 (512MB / (256*1024*4)). Ours must match
        // so behaviour is unchanged where the old floor never bit.
        let bs = cuda_batch_size(256, 1024, 1_000_000);
        assert_eq!(bs, 512);
    }

    #[test]
    fn resident_levels_footprint_is_bounded_not_whole_tensor() {
        // The whole point: per-batch resident levels must be ~MB, not ~GB, for a
        // 235M-param tensor. We assert the *math*, no allocation.
        let block_len = 256usize;
        let num_states = 1024usize; // L=10
        let n_sub_per_block = block_len.div_ceil(SUB_BLOCK); // 8
        let total_weights = 235_000_000usize;
        let n_blocks = total_weights.div_ceil(block_len);

        let levels_bytes_per_block = n_sub_per_block * num_states * std::mem::size_of::<f32>();
        let whole_tensor = n_blocks * levels_bytes_per_block; // OLD resident
        let bs = cuda_batch_size(block_len, num_states, n_blocks);
        let per_batch = bs * levels_bytes_per_block; // NEW resident

        // Old path held tens of GB; new path holds tens of MB.
        assert!(
            whole_tensor > 20 * 1024 * 1024 * 1024,
            "sanity: old resident should be >20GB, got {whole_tensor}"
        );
        assert!(
            per_batch < 64 * 1024 * 1024,
            "new per-batch resident must be <64MB, got {per_batch}"
        );
    }

    // ---- per-batch level construction == all-at-once construction -------------

    #[test]
    fn per_batch_levels_equal_all_at_once() {
        // Prove the batching is value-preserving: concatenating per-block levels
        // built one-at-a-time (the new path) equals the single flat buffer the
        // old path built. Uses real LUT + real scale search, CPU-only, no GPU.
        let cfg = TrellisConfig::new(8, 2, 256); // L=8 so test is light
        let lut = codebook_lut(cfg.l_bits);
        let num_states = cfg.num_states();
        let n_sub_per_block = cfg.block_len.div_ceil(SUB_BLOCK);
        let q_to_real = 1.0f32 / (1u32 << QUANTILE_SHIFT) as f32;

        // 5 blocks worth of structured weights.
        let total = cfg.block_len * 5 + 37; // last block ragged on purpose
        let weights: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.011).sin() * 0.3 + ((i % 7) as f32) * 0.02)
            .collect();

        // Build the plan (same as both code paths).
        let mut plans: Vec<BlockPlan> = Vec::new();
        let mut offset = 0usize;
        for chunk in weights.chunks(cfg.block_len) {
            let scale_q = choose_scale_q(chunk, lut, &cfg);
            let mults = choose_sub_scales(chunk, scale_q, lut, &cfg);
            plans.push(BlockPlan {
                chunk_offset: offset,
                chunk_len: chunk.len(),
                scale_q,
                mults,
            });
            offset += chunk.len();
        }

        // OLD: one flat buffer for all blocks at once.
        let mut all_at_once: Vec<f32> = Vec::new();
        for p in &plans {
            let lv = build_levels(
                p.scale_q, &p.mults, lut, num_states, n_sub_per_block, q_to_real,
            );
            all_at_once.extend_from_slice(&lv);
        }

        // NEW: per-batch concatenation with a deliberately tiny batch_size to
        // force multiple batches (worst case for ordering bugs).
        let batch_size = 2usize;
        let mut batched: Vec<f32> = Vec::new();
        let mut bi_base = 0;
        while bi_base < plans.len() {
            let bi_end = (bi_base + batch_size).min(plans.len());
            for p in &plans[bi_base..bi_end] {
                let lv = build_levels(
                    p.scale_q, &p.mults, lut, num_states, n_sub_per_block, q_to_real,
                );
                batched.extend_from_slice(&lv);
            }
            bi_base = bi_end;
        }

        assert_eq!(all_at_once.len(), batched.len());
        assert_eq!(all_at_once, batched, "per-batch levels must equal all-at-once");
    }
}
