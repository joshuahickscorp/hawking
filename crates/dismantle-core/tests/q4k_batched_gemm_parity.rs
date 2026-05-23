#![cfg(target_os = "macos")]
//! P3 — parity test for `gemm_q4_k_m_batched_v2_pinned_tcb`.
//!
//! The batched kernel must produce the same outputs as B back-to-back
//! single-matrix GEMVs (modulo fp32 reduction order tolerance). We
//! verify this against the existing `gemv_q4_k_m_v2_pinned_tcb` which
//! is the row-wise scalar reference shipped in the qwen_dense pipeline.

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use dismantle_core::quant;
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

fn fixed_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

fn read_f32_buf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Run B single-vector GEMVs against the same weight and concatenate
/// the outputs into a (B, rows) row-major matrix.
fn reference_b_gemvs(
    ctx: &MetalContext,
    w_buf: &PinnedBuffer,
    w_bytes_len: usize,
    rows: usize,
    cols: usize,
    x_batch: &[f32],
    batch: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * rows];
    for b in 0..batch {
        let x_slice = &x_batch[b * cols..(b + 1) * cols];
        let x_buf = new_f32_buf(ctx, x_slice);
        let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_m_v2_pinned_tcb(
                &mut tcb,
                w_buf,
                0,
                w_bytes_len,
                rows,
                cols,
                &x_buf,
                &y_buf,
            )
            .expect("v2 gemv");
            tcb.commit_and_wait().expect("commit");
        }
        let y = read_f32_buf(&y_buf, rows);
        out[b * rows..(b + 1) * rows].copy_from_slice(&y);
    }
    out
}

#[test]
fn batched_q4k_matches_per_token_gemv() {
    // Mirrors a Qwen-3B FFN gate shape (intermediate × hidden).
    let rows = 1024usize;
    let cols = 2048usize;

    // Build a Q4_K weight from random f32.
    let w_f32 = fixed_f32(rows * cols, 0xAA55_AA55);
    let blocks = (rows * cols) / 256;
    let mut w_q4 = vec![0u8; blocks * quant::Q4_K_BLOCK_BYTES];
    quant::quantize_q4_k(&w_f32, &mut w_q4).expect("Q4_K quantize");

    let ctx = ctx();
    let model_buf = ctx.new_buffer_with_bytes(&w_q4);

    for &batch in &[1usize, 2, 3, 4] {
        let x_batch = fixed_f32(batch * cols, 0x1234_5678 ^ (batch as u64));
        let expected =
            reference_b_gemvs(ctx, &model_buf, w_q4.len(), rows, cols, &x_batch, batch);

        let x_buf = new_f32_buf(ctx, &x_batch);
        let y_buf = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_m_batched_v2_pinned_tcb(
                &mut tcb,
                &model_buf,
                0,
                w_q4.len(),
                rows,
                cols,
                batch,
                &x_buf,
                &y_buf,
            )
            .expect("batched gemm encode");
            tcb.commit_and_wait().expect("commit");
        }
        let actual = read_f32_buf(&y_buf, batch * rows);
        let diff = max_abs_diff(&expected, &actual);
        assert!(
            diff < 1e-3,
            "batched Q4_K vs per-token v2 (batch={batch}): max_abs_diff = {diff} (limit 1e-3)"
        );

        // v3 parity: same expected output via the shmem-staged variant.
        let y_buf_v3 = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_m_batched_v3_pinned_tcb(
                &mut tcb, &model_buf, 0, w_q4.len(),
                rows, cols, batch, &x_buf, &y_buf_v3,
            ).expect("batched gemm v3 encode");
            tcb.commit_and_wait().expect("commit v3");
        }
        let actual_v3 = read_f32_buf(&y_buf_v3, batch * rows);
        let diff_v3 = max_abs_diff(&expected, &actual_v3);
        assert!(
            diff_v3 < 1e-3,
            "batched Q4_K v3 vs per-token v2 (batch={batch}): max_abs_diff = {diff_v3} (limit 1e-3)"
        );
    }
}
