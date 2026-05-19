//! Path B parity tests — every parallel-K kernel must match K sequential
//! single-token runs of the corresponding existing kernel (within atol=1e-3
//! fp16, matching the existing phase1 kernel parity gate).
//!
//! These tests are scaffolded as #[ignore] until the kernel bodies land.
//! Once a kernel is implemented, remove the #[ignore] for its test.

use dismantle_core::kernels::parallel_k;
#[cfg(target_os = "macos")]
use dismantle_core::{
    kernels,
    metal::{MetalContext, PinnedBuffer, TokenCommandBuffer},
};
#[cfg(target_os = "macos")]
use half::f16;
#[cfg(target_os = "macos")]
use once_cell::sync::Lazy;
#[cfg(target_os = "macos")]
use rand::Rng;
#[cfg(target_os = "macos")]
use rand_pcg::Pcg64Mcg;

#[test]
#[ignore = "Path B kernel not yet implemented; see reports/path_to_90/path_b/design.md"]
fn mla_decode_kbatch_matches_sequential_k4() {
    // Plan: construct K=4 random q_nope/q_rope queries, a fixed KV cache,
    // run mla_decode_kernel_fc K times sequentially to get the reference,
    // run mla_decode_kernel_fc_kbatch once to get the parallel result,
    // assert each (k, out_dim) element matches within 1e-3 fp16 atol.

    let dummy = vec![0u8; 16];
    let mut out = vec![0u8; 16];
    // Currently returns Unimplemented; once the kernel lands, remove
    // #[ignore] and replace with the real parity comparison.
    let result = parallel_k::mla_decode_kernel_fc_kbatch(
        &dummy, &dummy, &dummy, &dummy, &dummy, &mut out, 1, 4,
    );
    assert!(
        result.is_err(),
        "skeleton phase: kernel should still be Unimplemented",
    );
}

#[test]
#[ignore = "Path B kernel not yet implemented; see reports/path_to_90/path_b/design.md"]
fn gemv_q6_k_v3_kbatch_matches_sequential_k4() {
    let dummy = vec![0u8; 16];
    let mut out = vec![0u8; 16];
    let result = parallel_k::gemv_q6_k_v3_kbatch(
        &dummy, &dummy, &mut out, 1, 1, 4,
    );
    assert!(result.is_err());
}

#[test]
#[ignore = "Path B kernel not yet implemented; see reports/path_to_90/path_b/design.md"]
fn moe_block_kbatch_matches_sequential_k4() {
    let dummy = vec![0u8; 16];
    let mut out = vec![0u8; 16];
    let result = parallel_k::moe_block_batched_indexed_kbatch(
        &dummy, &dummy, &dummy, &dummy,
        &[0u32], &[0u32], &[0.0f32],
        &mut out, 1, 1, 4,
    );
    assert!(result.is_err());
}

// ── Stage 2.2 — K-batched fp16 lm_head GEMV parity ─────────────────────────

#[cfg(target_os = "macos")]
fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

#[cfg(target_os = "macos")]
fn fixed_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

#[cfg(target_os = "macos")]
fn fixed_f16(n: usize, seed: u64) -> Vec<f16> {
    fixed_f32(n, seed).iter().map(|&v| f16::from_f32(v)).collect()
}

#[cfg(target_os = "macos")]
fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

#[cfg(target_os = "macos")]
fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

#[cfg(target_os = "macos")]
fn read_f32_buf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

#[cfg(target_os = "macos")]
fn run_kbatch(
    ctx: &MetalContext,
    w_buf: &PinnedBuffer,
    rows: usize,
    cols: usize,
    x_kbatch: &[f32],
    k_batch: usize,
) -> Vec<f32> {
    let x_buf = new_f32_buf(ctx, x_kbatch);
    let y_buf = ctx.new_buffer(k_batch * rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    parallel_k::gemv_f16_lmhead_kbatch_tcb(&mut tcb, w_buf, rows, cols, &x_buf, &y_buf, k_batch)
        .expect("kbatch dispatch");
    tcb.commit_and_wait().expect("commit");
    read_f32_buf(&y_buf, k_batch * rows)
}

#[cfg(target_os = "macos")]
fn run_sequential(
    ctx: &MetalContext,
    w_buf: &PinnedBuffer,
    rows: usize,
    cols: usize,
    x_kbatch: &[f32],
    k_batch: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; k_batch * rows];
    for k in 0..k_batch {
        let x_buf = new_f32_buf(ctx, &x_kbatch[k * cols..(k + 1) * cols]);
        let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_f16_simdmat_tcb(&mut tcb, w_buf, rows, cols, &x_buf, &y_buf)
            .expect("sequential dispatch");
        tcb.commit_and_wait().expect("commit");
        let y = read_f32_buf(&y_buf, rows);
        out[k * rows..(k + 1) * rows].copy_from_slice(&y);
    }
    out
}

#[cfg(target_os = "macos")]
fn assert_kbatch_matches_sequential(rows: usize, cols: usize, k_batch: usize, seed: u64) {
    let ctx = ctx();
    let w = fixed_f16(rows * cols, seed ^ 0xA5A5_5A5A);
    let x_kbatch = fixed_f32(k_batch * cols, seed ^ 0x1234_5678);
    let w_buf = new_f16_buf(ctx, &w);

    let kbatch_out = run_kbatch(ctx, &w_buf, rows, cols, &x_kbatch, k_batch);
    let seq_out = run_sequential(ctx, &w_buf, rows, cols, &x_kbatch, k_batch);

    let diff = kbatch_out
        .iter()
        .zip(seq_out.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff < 1e-3,
        "K={k_batch} rows={rows} cols={cols}: max_abs_diff={diff:.3e} >= 1e-3",
    );
}

#[cfg(target_os = "macos")]
#[test]
fn gemv_f16_lmhead_kbatch_matches_sequential_basic_shapes() {
    for &(rows, cols) in &[(8usize, 8usize), (16, 8), (32, 64), (64, 128)] {
        for &k in &[1usize, 2, 4, 8] {
            assert_kbatch_matches_sequential(rows, cols, k, 0xDEAD_BEEF ^ (rows * cols) as u64);
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn gemv_f16_lmhead_kbatch_matches_sequential_lmhead_shape_k4() {
    // V2-Lite lm_head analogue: rows=vocab=102400 is too large for a unit test,
    // but rows=512 cols=2048 exercises the same (rows % 8 == 0, cols % 8 == 0)
    // dispatch geometry and the simdgroup_matrix tile shape, mirroring
    // v1x_lm_head_simdmat_parity::phase_x_simdmat_matches_cpu_lm_head_shape.
    assert_kbatch_matches_sequential(512, 2048, 4, 0xBEEF_1234);
}

#[cfg(target_os = "macos")]
#[test]
fn gemv_f16_lmhead_kbatch_k1_matches_simdmat_bitwise() {
    // At K=1 the kbatch kernel reduces to gemv_f16_simdmat exactly (the
    // K-fold X tile collapses to the original broadcast). Verify the
    // outputs match bit-for-bit, not just within atol — the only difference
    // would be rounding-order variation from out-of-range X cols being
    // zero-padded, which contributes literal zero terms.
    let ctx = ctx();
    let rows = 256usize;
    let cols = 256usize;
    let w = fixed_f16(rows * cols, 0xCAFE_BABE);
    let x = fixed_f32(cols, 0xF00D_FEED);
    let w_buf = new_f16_buf(ctx, &w);

    let kbatch = run_kbatch(ctx, &w_buf, rows, cols, &x, 1);
    let seq = run_sequential(ctx, &w_buf, rows, cols, &x, 1);

    assert_eq!(
        kbatch, seq,
        "K=1 kbatch must match gemv_f16_simdmat bitwise (zero-padded X cols contribute literal zero terms)",
    );
}

// ── Stage 2.3 — K-batched Q4_K_M GEMV parity ───────────────────────────────

#[cfg(target_os = "macos")]
fn synthetic_q4_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        let d = 0.01 + rng.gen::<f32>() * 0.01;
        let d_bits = f16::from_f32(d).to_bits();
        bytes[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
        let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
        let dmin_bits = f16::from_f32(dmin).to_bits();
        bytes[off + 2..off + 4].copy_from_slice(&dmin_bits.to_le_bytes());
        for i in 4..16 {
            bytes[off + i] = rng.gen::<u8>() & 0x3F;
        }
        for i in 16..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}

#[cfg(target_os = "macos")]
fn assert_q4k_kbatch_matches_sequential(rows: usize, cols: usize, k_batch: usize, seed: u64) {
    assert_eq!(cols % 256, 0, "Q4_K requires cols % 256 == 0");
    let ctx = ctx();
    let n_blocks = rows * (cols / 256);
    let w_bytes = synthetic_q4_k_bytes(n_blocks, seed ^ 0xA5A5_5A5A);
    let x_kbatch = fixed_f32(k_batch * cols, seed ^ 0x1234_5678);

    // CPU reference: dequant Q4_K_M → f32, then K f32 GEMVs. This bypasses
    // the broken non-pinned `dispatch_q4_k_m_gemv_v2` dispatcher whose
    // `set_bytes(3, 4)` + `set_bytes(4, 4)` doesn't match the v2 shader's
    // 8-byte `ArgbufRowsCols` struct at slot 3 (silent UB on synthetic
    // data; production is unaffected because it uses the pinned-tcb
    // dispatcher with proper ArgBuffer binding). The CPU dequant+gemv
    // matches `gemm_q4_k_m_fused` (v1, slot-correct) per phase 1 parity.
    use dismantle_core::gguf::GgmlType;
    use dismantle_core::quant::dequant_into;
    let mut w_f32 = vec![0.0f32; rows * cols];
    dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32)
        .expect("Q4_K dequant succeeds for valid synthetic bytes");
    let mut seq_out = vec![0.0f32; k_batch * rows];
    for k in 0..k_batch {
        let x_k = &x_kbatch[k * cols..(k + 1) * cols];
        let mut y_k = vec![0.0f32; rows];
        dismantle_core::kernels::gemv_f32(&w_f32, rows, cols, x_k, &mut y_k);
        seq_out[k * rows..(k + 1) * rows].copy_from_slice(&y_k);
    }

    // K-batched: pin weights, dispatch one kbatch kernel.
    let w_buf = ctx.new_buffer_with_bytes(&w_bytes);
    let x_buf = new_f32_buf(ctx, &x_kbatch);
    let y_buf = ctx.new_buffer(k_batch * rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    parallel_k::gemv_q4_k_m_v2_kbatch_pinned_tcb(
        &mut tcb,
        &w_buf,
        0,
        w_bytes.len(),
        rows,
        cols,
        &x_buf,
        &y_buf,
        k_batch,
    )
    .expect("kbatch dispatch");
    tcb.commit_and_wait().expect("commit");
    let kbatch_out = read_f32_buf(&y_buf, k_batch * rows);

    let diff = kbatch_out
        .iter()
        .zip(seq_out.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff < 1e-3,
        "K={k_batch} rows={rows} cols={cols}: max_abs_diff={diff:.3e} >= 1e-3 (Q4_K noise floor)",
    );
}

#[cfg(target_os = "macos")]
#[test]
fn gemv_q4_k_m_v2_kbatch_matches_sequential_basic_shapes() {
    // cols must be multiple of 256 for Q4_K super-blocks.
    for &(rows, cols) in &[(8usize, 256usize), (16, 256), (32, 256), (64, 512)] {
        for &k in &[1usize, 2, 4, 8] {
            assert_q4k_kbatch_matches_sequential(rows, cols, k, 0xDEAD_BEEF ^ (rows * cols) as u64);
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn gemv_q4_k_m_v2_kbatch_matches_sequential_v2lite_attn_shape_k4() {
    // V2-Lite attn-projection analogue (e.g., q_b_proj: rows=2048, cols=1536
    // → cols rounded up to next 256 = 1536 is already aligned).
    // Use 2048 × 1536 K=4 as the representative production shape.
    assert_q4k_kbatch_matches_sequential(2048, 1536, 4, 0xBEEF_1234);
}

#[cfg(target_os = "macos")]
#[test]
fn gemv_q4_k_m_v2_kbatch_k1_matches_v2_within_atol() {
    // At K=1 the kbatch kernel reduces to gemm_q4_k_m_fused_v2 exactly
    // (per-block decode + partial accumulation logic is identical).
    // Confirm a sample shape matches within Q4_K noise (1e-3).
    assert_q4k_kbatch_matches_sequential(64, 512, 1, 0xCAFE_BABE);
}

// ── Stage 2.4 — K-batched MLA decode parity ────────────────────────────────

#[cfg(target_os = "macos")]
fn assert_mla_kbatch_matches_sequential(
    n_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    seq_len: usize,
    k_batch: usize,
    seed: u64,
) {
    let ctx = ctx();
    let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    let q_kbatch = fixed_f32(k_batch * n_heads * q_head_dim, seed ^ 0x0101_2020);
    let c_kv = fixed_f32(seq_len * kv_lora_rank, seed ^ 0x0303_4040);
    let k_pe = fixed_f32(seq_len * qk_rope_head_dim, seed ^ 0x0505_6060);
    let kv_b_proj_data =
        fixed_f32(n_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank, seed ^ 0x0707_8080);
    let scale = 1.0f32 / (q_head_dim as f32).sqrt();

    let kv_b_buf = new_f32_buf(ctx, &kv_b_proj_data);

    // Causal-mask convention for the K-batched kernel (A1.1):
    //   seq_len = seq_len_base + k_batch (caller appends K new KVs first)
    //   query kk attends to positions [0, seq_len_base + kk]
    //                              = [0, seq_len - k_batch + kk]
    // So the sequential reference computes mla_decode at per-K seq_lens:
    //   query 0:        seq_len - k_batch + 1
    //   query k_batch-1: seq_len
    // At k_batch=1 this reduces to seq_len (bit-identical to the K=1 path).
    let mut seq_out = vec![0.0f32; k_batch * n_heads * v_head_dim];
    for k in 0..k_batch {
        let q_k = &q_kbatch[k * n_heads * q_head_dim..(k + 1) * n_heads * q_head_dim];
        let mut y_k = vec![0.0f32; n_heads * v_head_dim];
        let per_k_seq_len = seq_len - k_batch + k + 1;
        let c_kv_slice = &c_kv[..per_k_seq_len * kv_lora_rank];
        let k_pe_slice = &k_pe[..per_k_seq_len * qk_rope_head_dim];
        dismantle_core::kernels::mla_decode_metal(
            ctx,
            q_k,
            c_kv_slice,
            k_pe_slice,
            &kv_b_buf,
            n_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            kv_lora_rank,
            per_k_seq_len,
            scale,
            &mut y_k,
        )
        .expect("sequential mla_decode_metal");
        seq_out[k * n_heads * v_head_dim..(k + 1) * n_heads * v_head_dim].copy_from_slice(&y_k);
    }

    // K-batched
    let mut kbatch_out = vec![0.0f32; k_batch * n_heads * v_head_dim];
    parallel_k::mla_decode_metal_kbatch(
        ctx,
        &q_kbatch,
        &c_kv,
        &k_pe,
        &kv_b_buf,
        n_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        kv_lora_rank,
        seq_len,
        scale,
        &mut kbatch_out,
        k_batch,
    )
    .expect("mla_decode_metal_kbatch");

    let diff = kbatch_out
        .iter()
        .zip(seq_out.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff < 1e-3,
        "K={k_batch} seq_len={seq_len} heads={n_heads}: max_abs_diff={diff:.3e} >= 1e-3",
    );
}

#[cfg(target_os = "macos")]
#[test]
fn mla_decode_kbatch_matches_sequential_v2lite_shape() {
    // V2-Lite MLA shape: n_heads=16, qk_nope=128, qk_rope=64,
    // v_head=128, kv_lora=512. seq_len=4 covers Phase 0..4 with
    // multiple timesteps; small enough to be a unit test.
    for &k in &[1usize, 2, 4] {
        assert_mla_kbatch_matches_sequential(16, 128, 64, 128, 512, 4, k, 0xCAFE_BABE);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn mla_decode_kbatch_matches_sequential_realistic_seq() {
    // Longer seq_len = 64 (still unit-test-fast) at K=4.
    assert_mla_kbatch_matches_sequential(16, 128, 64, 128, 512, 64, 4, 0xBEEF_1234);
}

#[cfg(target_os = "macos")]
#[test]
fn mla_decode_kbatch_k1_matches_sequential_at_atol() {
    // At K=1 the K-batched kernel should reduce to the K=1 reference.
    assert_mla_kbatch_matches_sequential(16, 128, 64, 128, 512, 8, 1, 0xDEAD_BEEF);
}

#[test]
#[ignore = "Tree-decode extension not yet implemented; see reports/path_to_90/tree_decode/design.md"]
fn mla_decode_kbatch_tree_mask_matches_unmasked_when_all_zero() {
    // Once landed: a zero-bias mask (all 0s) should produce identical output
    // to the unmasked version. Validates the mask-arg surface is wired
    // without changing the math when mask is trivial.
    let dummy = vec![0u8; 16];
    let mut out = vec![0u8; 16];
    let zero_mask = vec![0.0f32; 16];
    let result = parallel_k::mla_decode_kernel_fc_kbatch_masked(
        &dummy, &dummy, &dummy, &dummy, &dummy, &mut out, &zero_mask, 1, 4,
    );
    assert!(result.is_err());
}
