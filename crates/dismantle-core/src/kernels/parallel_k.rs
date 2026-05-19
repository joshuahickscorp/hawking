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
use crate::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};

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

// ── Stage 2.3 — K-batched Q4_K_M GEMV ───────────────────────────────────────

/// Path B Stage 2.3 — K-batched Q4_K_M GEMV.
///
/// Computes `y_kbatch[k, r] = Σ_c W_q4k[r, c] * x_kbatch[k, c]` for
/// `k ∈ [0, k_batch)`, `r ∈ [0, rows)`. Mirrors the dispatch geometry of
/// `gemv_q4_k_m_v2_pinned_tcb` (mod.rs:394); the per-block Q4_K_M decode
/// is unchanged, and each lane accumulates K f32 partials in registers
/// (zero TG memory; matches the K=1 kernel's no-TG-mem property).
///
/// Requires `cols % 256 == 0` and `k_batch ∈ [1, 8]`. At K=1 the kernel
/// is bit-equivalent to `gemm_q4_k_m_fused_v2`.
///
/// `w_buf` is a pinned model buffer; `w_offset` and `w_byte_size` slice
/// out the Q4_K_M weight tensor. `x_buf` is `(k_batch × cols)` f32
/// row-major; `y_buf` is `(k_batch × rows)` f32 row-major (caller
/// pre-allocates).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn gemv_q4_k_m_v2_kbatch_pinned_tcb(
    tcb: &mut TokenCommandBuffer<'_>,
    model_buf: &PinnedBuffer,
    w_offset: usize,
    w_byte_size: usize,
    rows: usize,
    cols: usize,
    x_buf: &PinnedBuffer,
    y_buf: &PinnedBuffer,
    k_batch: usize,
) -> Result<()> {
    const KERNEL: &str = "gemm_q4_k_m_fused_v2_kbatch";
    if cols % 256 != 0 {
        return Err(Error::Kernel(format!(
            "{KERNEL} requires cols % 256 == 0; got cols={cols}"
        )));
    }
    if !(1..=8).contains(&k_batch) {
        return Err(Error::Kernel(format!(
            "{KERNEL} requires k_batch in 1..=8; k_batch={k_batch}"
        )));
    }
    let blocks_per_row = cols / 256;
    let expected_bytes = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(144))
        .ok_or_else(|| Error::Kernel(format!("{KERNEL} byte-size overflow")))?;
    if w_byte_size != expected_bytes {
        return Err(Error::Kernel(format!(
            "{KERNEL} weight bytes: got {w_byte_size} expected {expected_bytes}"
        )));
    }
    let end = w_offset
        .checked_add(w_byte_size)
        .ok_or_else(|| Error::Kernel(format!("{KERNEL} offset overflow")))?;
    if end > model_buf.length() as usize {
        return Err(Error::Kernel(format!(
            "{KERNEL} offset out of bounds: {w_offset}+{w_byte_size} > {}",
            model_buf.length()
        )));
    }
    let x_bytes = k_batch * cols * std::mem::size_of::<f32>();
    let y_bytes = k_batch * rows * std::mem::size_of::<f32>();
    if x_buf.length() < x_bytes as u64 || y_buf.length() < y_bytes as u64 {
        return Err(Error::Kernel(format!(
            "{KERNEL} buffer sizes: x={} expected>={x_bytes} y={} expected>={y_bytes}",
            x_buf.length(),
            y_buf.length()
        )));
    }

    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    let k_u32 = k_batch as u32;
    const V2_TG: u32 = 256;
    let n_tg = rows_u32.div_ceil(8);
    tcb.dispatch_threads(
        KERNEL,
        (n_tg * V2_TG, 1, 1),
        (V2_TG, 1, 1),
        |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
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
        },
    )
}

// ── Stage 2.4 — K-batched MLA decode ────────────────────────────────────────

/// Path B Stage 2.4 — K-batched MLA decode.
///
/// Mirrors `mla_decode_metal` (kernels/mod.rs:1560) but processes K
/// queries against the SAME (c_kv, k_pe) KV cache in one dispatch.
/// Weight reads (kv_b_proj) and KV-cache reads amortize across the K
/// queries inside one threadgroup.
///
/// At K ≥ 2 the naive extension of the K=1 TG-mem scores buffer busts
/// the 32 KB / core budget (scores[K × seq_len] = 64 KB at K=4,
/// seq_len=4096). This kernel moves scores to a device-scratch buffer
/// (n_heads × K × seq_len f32) allocated here. TG memory stays at
/// 2 × K × kv_lora_rank f32 = 16 KB at K=4 kv_lora_rank=512.
///
/// At K=1 the kernel is bit-equivalent to `mla_decode_kernel` by
/// construction (the K-fold loops reduce to single iterations).
///
/// One-shot helper for tests/benches (allocates Metal buffers from
/// CPU slices). Production wire-up will follow the
/// `mla_decode_and_o_proj_arena_*_tcb` pattern (Stage 2.6).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_metal_kbatch(
    ctx: &MetalContext,
    q_kbatch: &[f32],            // (K × n_heads × (qk_nope + qk_rope)) f32
    c_kv: &[f32],                // (seq_len × kv_lora_rank) f32
    k_pe: &[f32],                // (seq_len × qk_rope_head_dim) f32
    kv_b_proj: &PinnedBuffer,
    n_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    seq_len: usize,
    scale: f32,
    out_kbatch: &mut [f32],      // (K × n_heads × v_head_dim) f32
    k_batch: usize,
) -> Result<()> {
    const KERNEL: &str = "mla_decode_kernel_fc_kbatch";
    const TG_SIZE: u32 = 256;
    if !(1..=8).contains(&k_batch) {
        return Err(Error::Kernel(format!(
            "{KERNEL} requires k_batch in 1..=8; k_batch={k_batch}"
        )));
    }
    if seq_len == 0 {
        return Err(Error::Kernel(format!("{KERNEL}: seq_len must be >= 1")));
    }
    let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    let expected_q = k_batch * n_heads * q_head_dim;
    if q_kbatch.len() != expected_q {
        return Err(Error::Kernel(format!(
            "{KERNEL}: q_kbatch.len={} expected {}",
            q_kbatch.len(),
            expected_q
        )));
    }
    if c_kv.len() != seq_len * kv_lora_rank {
        return Err(Error::Kernel(format!(
            "{KERNEL}: c_kv.len={} expected {}",
            c_kv.len(),
            seq_len * kv_lora_rank
        )));
    }
    if k_pe.len() != seq_len * qk_rope_head_dim {
        return Err(Error::Kernel(format!(
            "{KERNEL}: k_pe.len={} expected {}",
            k_pe.len(),
            seq_len * qk_rope_head_dim
        )));
    }
    let expected_kv_b =
        (n_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank * std::mem::size_of::<f32>())
            as u64;
    if kv_b_proj.length() < expected_kv_b {
        return Err(Error::Kernel(format!(
            "{KERNEL}: kv_b_proj buffer too small: got {} expected {}",
            kv_b_proj.length(),
            expected_kv_b
        )));
    }
    let expected_out = k_batch * n_heads * v_head_dim;
    if out_kbatch.len() != expected_out {
        return Err(Error::Kernel(format!(
            "{KERNEL}: out_kbatch.len={} expected {}",
            out_kbatch.len(),
            expected_out
        )));
    }

    let q_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(q_kbatch));
    let c_kv_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(c_kv));
    let k_pe_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(k_pe));
    let out_buf = ctx.new_buffer(expected_out * std::mem::size_of::<f32>());
    let scores_scratch =
        ctx.new_buffer(n_heads * k_batch * seq_len * std::mem::size_of::<f32>());

    let n_heads_u32 = n_heads as u32;
    let qk_nope_u32 = qk_nope_head_dim as u32;
    let qk_rope_u32 = qk_rope_head_dim as u32;
    let v_head_u32 = v_head_dim as u32;
    let kv_lora_u32 = kv_lora_rank as u32;
    let seq_len_u32 = seq_len as u32;
    let k_batch_u32 = k_batch as u32;

    let qp_bytes = (k_batch * kv_lora_rank * std::mem::size_of::<f32>()) as u64;
    let cwt_bytes = (k_batch * kv_lora_rank * std::mem::size_of::<f32>()) as u64;

    ctx.dispatch_threads(
        KERNEL,
        (n_heads_u32 * TG_SIZE, 1, 1),
        (TG_SIZE, 1, 1),
        |enc| {
            enc.set_buffer(0, Some(&q_buf), 0);
            enc.set_buffer(1, Some(&c_kv_buf), 0);
            enc.set_buffer(2, Some(&k_pe_buf), 0);
            enc.set_buffer(3, Some(kv_b_proj), 0);
            enc.set_buffer(4, Some(&out_buf), 0);
            enc.set_buffer(5, Some(&scores_scratch), 0);
            enc.set_bytes(6, std::mem::size_of::<u32>() as u64, &n_heads_u32 as *const u32 as *const _);
            enc.set_bytes(7, std::mem::size_of::<u32>() as u64, &qk_nope_u32 as *const u32 as *const _);
            enc.set_bytes(8, std::mem::size_of::<u32>() as u64, &qk_rope_u32 as *const u32 as *const _);
            enc.set_bytes(9, std::mem::size_of::<u32>() as u64, &v_head_u32 as *const u32 as *const _);
            enc.set_bytes(10, std::mem::size_of::<u32>() as u64, &kv_lora_u32 as *const u32 as *const _);
            enc.set_bytes(11, std::mem::size_of::<u32>() as u64, &seq_len_u32 as *const u32 as *const _);
            enc.set_bytes(12, std::mem::size_of::<f32>() as u64, &scale as *const f32 as *const _);
            enc.set_bytes(13, std::mem::size_of::<u32>() as u64, &k_batch_u32 as *const u32 as *const _);
            enc.set_threadgroup_memory_length(0, qp_bytes);
            enc.set_threadgroup_memory_length(1, cwt_bytes);
        },
    )?;

    let out_ptr = out_buf.contents() as *const f32;
    let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, expected_out) };
    out_kbatch.copy_from_slice(out_slice);
    Ok(())
}

// ── Stage 2.5 — K-batched MoE block (no-overlap baseline) ───────────────────

/// Path B Stage 2.5 — K-batched routed-MoE block, no-overlap baseline.
///
/// Per the production roadmap (Stage 2.5: "ship no-overlap K=4 first to
/// validate parity; the masked-prefetch variant is Stage 3 task 3.2"),
/// this dispatcher bundles K independent MoE forwards into a SINGLE
/// `TokenCommandBuffer`. Each of the K queries runs its own per-route
/// gate/up/down + silu_mul + accumulate sequence; weight reads are NOT
/// amortized across the K (that's Stage 3.2's union-routing kernel).
///
/// The value of the no-overlap variant is twofold:
///   1. Provides the Path B API surface that the Stage 2.6 wire-up
///      (`forward_tokens_batched_parallel_k`) calls into.
///   2. Validates that K dispatches inside one TCB produce output
///      bit-identical to K independent TCB commits (no shared-scratch
///      ordering bugs).
///
/// Per-K-query inputs: `(x_buf, out_buf, route_ids, route_weights,
/// shared_route_ids)` — typically the K speculative query slots from the
/// arena. Per-query buffers are caller-owned PinnedBuffers; this fn
/// only encodes the TCB and does NOT commit (caller decides when to
/// commit, allowing further dispatches to be batched after).
///
/// Returns the union of temporary scratch buffers (gate/up outputs etc.)
/// across the K calls. Caller must hold them alive until the TCB
/// commits.
///
/// At K=1 the result is bit-identical to a single
/// `encode_moe_block_batched_indexed_tcb` call.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn moe_block_batched_indexed_kbatch_tcb<'a>(
    tcb: &mut TokenCommandBuffer<'_>,
    ctx: &MetalContext,
    model_buf: &PinnedBuffer,
    routed_gate_offset: usize,
    routed_up_offset: usize,
    routed_down_offset: usize,
    per_k_route_ids: &[&'a PinnedBuffer],     // K buffers of (top_k,) u32
    per_k_route_weights: &[&'a PinnedBuffer], // K buffers of (top_k,) f32
    per_k_routes: usize,                       // top_k (constant per K)
    shared_route_ids_buf: &PinnedBuffer,
    shared_gate_offset: Option<usize>,
    shared_up_offset: Option<usize>,
    shared_down_offset: Option<usize>,
    hidden: usize,
    routed_mid: usize,
    shared_mid: usize,
    q4k_schedule: &str,
    routed_down_kernel: &str,
    shared_down_kernel: &str,
    per_k_x: &[&'a PinnedBuffer],   // K input buffers of (hidden,) f32
    per_k_out: &[&'a PinnedBuffer], // K output buffers of (hidden,) f32
) -> Result<Vec<PinnedBuffer>> {
    let k = per_k_x.len();
    if k == 0 {
        return Err(Error::Kernel(
            "moe_block_batched_indexed_kbatch_tcb: k_batch must be >= 1".into(),
        ));
    }
    if !(1..=8).contains(&k) {
        return Err(Error::Kernel(format!(
            "moe_block_batched_indexed_kbatch_tcb: k_batch in 1..=8; k={k}"
        )));
    }
    if per_k_out.len() != k
        || per_k_route_ids.len() != k
        || per_k_route_weights.len() != k
    {
        return Err(Error::Kernel(format!(
            "moe_block_batched_indexed_kbatch_tcb: ragged per-K slots: \
             x={} out={} ids={} weights={} (all must equal k_batch={k})",
            per_k_x.len(),
            per_k_out.len(),
            per_k_route_ids.len(),
            per_k_route_weights.len(),
        )));
    }

    let mut scratch: Vec<PinnedBuffer> = Vec::with_capacity(k * 8);
    for kk in 0..k {
        let part = crate::kernels::encode_moe_block_batched_indexed_tcb(
            tcb,
            ctx,
            model_buf,
            routed_gate_offset,
            routed_up_offset,
            routed_down_offset,
            per_k_route_ids[kk],
            per_k_route_weights[kk],
            per_k_routes,
            shared_route_ids_buf,
            shared_gate_offset,
            shared_up_offset,
            shared_down_offset,
            hidden,
            routed_mid,
            shared_mid,
            q4k_schedule,
            routed_down_kernel,
            shared_down_kernel,
            per_k_x[kk],
            per_k_out[kk],
        )?;
        scratch.extend(part);
    }
    Ok(scratch)
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
