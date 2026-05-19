//! Path B — parallel-K verify kernels.
//!
//! See `reports/path_to_90/path_b/design.md` for the full design.
//!
//! Goal: drop spec-decode verify cost from `K × single-forward` to
//! `~1.5 × single-forward` by rewriting the three heaviest kernels
//! (MLA decode, lm_head GEMV, MoE block) to process K queries in one
//! dispatch, sharing the weight read across the K.
//!
//! Status: design-only skeleton. The function signatures are stable
//! (they're what `forward_tokens_batched_parallel_k` will call once it
//! lands), but the bodies return `Err(Unimplemented)`. The parity
//! tests in `tests/path_b_parity.rs` exercise these signatures and
//! are marked `#[ignore]` until the kernels are real.
//!
//! Implementation order (per design doc §"Per-kernel implementation"):
//!   1. `gemv_q6_k_v3_kbatch`           — simplest; lm_head is the largest single read
//!   2. `mla_decode_kernel_fc_kbatch`   — heaviest; KV-cache sharing is the real win
//!   3. `moe_block_batched_indexed_kbatch` — most algorithmically novel
//!   4. Engine wire-up in `model/deepseek_v2.rs::forward_tokens_batched_parallel_k`
//!   5. Autotune sweep on M3 Pro 18 GB per context-length bucket
//!
//! Stage 2.1 design refresh (see `path_b/design.md`): the V2-Lite
//! production lm_head is fp16, not Q6_K, so Stage 2.2 lands
//! `gemv_f16_lmhead_kbatch_tcb` first against the actual production
//! surface. `gemv_q6_k_v3_kbatch` (below, byte-slice stub) remains a
//! deferred deliverable for quantized-lm_head model variants.

use crate::{Error, Result};
#[cfg(target_os = "macos")]
use crate::metal::{PinnedBuffer, TokenCommandBuffer};

/// Parallel-K MLA decode kernel.
///
/// Processes K queries against the SAME KV cache in one dispatch.
/// Output: `(K, n_heads * v_head_dim)` flat layout.
///
/// Per design doc, the KV-cache read amortizes across the K columns;
/// expected wall-clock at K=4 is ~1.5× single-query (vs 4× sequential).
pub fn mla_decode_kernel_fc_kbatch(
    // ... full signature TBD; placeholder shapes match the single-token kernel
    _c_kv: &[u8],
    _k_pe: &[u8],
    _q_nope_batched: &[u8],  // shape (K, n_heads, qk_nope_head_dim) packed
    _q_rope_batched: &[u8],  // shape (K, n_heads, qk_rope_head_dim) packed
    _kv_b_proj: &[u8],
    _output_batched: &mut [u8],  // shape (K, n_heads * v_head_dim) packed
    _seq_len: usize,
    _k: usize,
) -> Result<()> {
    Err(Error::Unimplemented(
        "parallel_k::mla_decode_kernel_fc_kbatch (Path B; see reports/path_to_90/path_b/design.md)",
    ))
}

/// Parallel-K lm_head GEMV (Q6_K weights).
///
/// `(vocab, hidden) @ (K, hidden).T → (K, vocab)`. The weights are read
/// once and dot-producted against all K queries inside the same threadgroup,
/// saving ~4× on the largest single weight read in the model.
pub fn gemv_q6_k_v3_kbatch(
    _w: &[u8],  // Q6_K-packed weight blocks
    _x_batched: &[u8],  // (K, hidden) packed f16
    _y_batched: &mut [u8],  // (K, vocab) packed f32
    _vocab: usize,
    _hidden: usize,
    _k: usize,
) -> Result<()> {
    Err(Error::Unimplemented(
        "parallel_k::gemv_q6_k_v3_kbatch (Path B; see reports/path_to_90/path_b/design.md)",
    ))
}

/// Parallel-K MoE block (gate + routed-expert FFN).
///
/// Each K-th query has its own top-k=6 route. The kernel:
///   1. Builds a SHARED expert-batch (union of distinct experts across K queries).
///   2. For each shared expert, GEMMs against all K queries that selected it,
///      weighted by per-query per-route weight.
///   3. Sums per-query outputs.
///
/// Typical K=4 overlap is 50-70% → ~2.5× saving on expert-weight reads.
pub fn moe_block_batched_indexed_kbatch(
    _gate_w: &[u8],
    _up_w: &[u8],
    _down_w: &[u8],
    _x_batched: &[u8],  // (K, hidden)
    _distinct_experts: &[u32],  // (n_distinct,)
    _per_k_route_idx: &[u32],   // (K, top_k)
    _per_k_route_weight: &[f32],  // (K, top_k)
    _y_batched: &mut [u8],  // (K, hidden)
    _hidden: usize,
    _intermediate: usize,
    _k: usize,
) -> Result<()> {
    Err(Error::Unimplemented(
        "parallel_k::moe_block_batched_indexed_kbatch (Path B; see reports/path_to_90/path_b/design.md)",
    ))
}

/// Plan for tree-decoding mask support (extends Path B per
/// `reports/path_to_90/tree_decode/design.md`).
///
/// The MLA kernel above will eventually accept an optional `(K, K)` attention
/// mask. For linear K-spec the mask is causal; for tree spec it encodes
/// the parent-of-relation. Adding the mask is a signature extension once
/// the base parallel-K kernel is correctness-validated; not in this
/// skeleton.
pub fn mla_decode_kernel_fc_kbatch_masked(
    _c_kv: &[u8],
    _k_pe: &[u8],
    _q_nope_batched: &[u8],
    _q_rope_batched: &[u8],
    _kv_b_proj: &[u8],
    _output_batched: &mut [u8],
    _attention_mask: &[f32],  // (K, K) — 0 for attend, -inf for block
    _seq_len: usize,
    _k: usize,
) -> Result<()> {
    Err(Error::Unimplemented(
        "parallel_k::mla_decode_kernel_fc_kbatch_masked (tree decode extension; \
         see reports/path_to_90/tree_decode/design.md)",
    ))
}

// ── Stage 2.2 — K-batched f16 lm_head GEMV ──────────────────────────────────

/// Path B Stage 2.2 — K-batched fp16 lm_head GEMV.
///
/// Computes `y_kbatch[k, r] = Σ_c W[r, c] * x_kbatch[k, c]` for
/// `k ∈ [0, k_batch)`, `r ∈ [0, rows)`. Mirrors the dispatch geometry of
/// `gemv_f16_simdmat_tcb` (matmul.metal:97); the K-batching is folded
/// into the same 8×8 simdgroup matmul tile that the K=1 kernel already
/// uses, so 1 ≤ K ≤ 8 has identical TG memory (192 floats) and
/// near-identical wall-clock to the K=1 dispatch.
///
/// Requires `cols % 8 == 0` and `k_batch ∈ [1, 8]`. At K=1 the kernel
/// is bit-equivalent to `gemv_f16_simdmat` by construction (out-of-range
/// X columns get zero-padded).
///
/// Buffer layout:
/// * `w_buf` — fp16 row-major `(rows × cols)`.
/// * `x_buf` — fp32 row-major `(k_batch × cols)`.
/// * `y_buf` — fp32 row-major `(k_batch × rows)`; caller pre-allocates.
#[cfg(target_os = "macos")]
pub fn gemv_f16_lmhead_kbatch_tcb(
    tcb: &mut TokenCommandBuffer<'_>,
    w_buf: &PinnedBuffer,
    rows: usize,
    cols: usize,
    x_buf: &PinnedBuffer,
    y_buf: &PinnedBuffer,
    k_batch: usize,
) -> Result<()> {
    if cols % 8 != 0 {
        return Err(Error::Kernel(format!(
            "gemv_f16_lmhead_kbatch requires cols % 8 == 0; cols={cols}"
        )));
    }
    if !(1..=8).contains(&k_batch) {
        return Err(Error::Kernel(format!(
            "gemv_f16_lmhead_kbatch requires k_batch in 1..=8; k_batch={k_batch}"
        )));
    }
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    let k_u32 = k_batch as u32;
    let n_groups = rows.div_ceil(8) as u32;
    let shmem_bytes: u64 = 192 * std::mem::size_of::<f32>() as u64;
    tcb.dispatch_threads(
        "gemv_f16_lmhead_kbatch",
        (n_groups * 32, 1, 1),
        (32, 1, 1),
        |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(y_buf), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &rows_u32 as *const u32 as *const _,
            );
            enc.set_bytes(
                4,
                std::mem::size_of::<u32>() as u64,
                &cols_u32 as *const u32 as *const _,
            );
            enc.set_bytes(
                5,
                std::mem::size_of::<u32>() as u64,
                &k_u32 as *const u32 as *const _,
            );
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stubs_return_unimplemented_with_pointer() {
        // The error message must include a pointer to the design doc so
        // anyone tripping these in a stack trace knows where to read up.
        let dummy = vec![0u8; 16];
        let mut out = vec![0u8; 16];
        let err = mla_decode_kernel_fc_kbatch(
            &dummy, &dummy, &dummy, &dummy, &dummy, &mut out, 1, 1,
        )
        .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("Path B"), "error must reference Path B: {s}");
        assert!(s.contains("design.md"), "error must reference design doc: {s}");
    }
}
