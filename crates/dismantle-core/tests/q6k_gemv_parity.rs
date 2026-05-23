#![cfg(target_os = "macos")]
//! P2 — parity test for `gemv_q6_k_pinned_tcb` against CPU reference.
//!
//! Builds a Q6_K weight via `quantize_q6_k`, computes the CPU reference
//! GEMV (dequant-then-multiply), runs the new Metal kernel through a
//! pinned buffer, and compares.
//!
//! Tolerance: 5e-2 absolute. Q6_K quant introduces noticeable error
//! (round-trip dequant→requant is *not* exact for Q6_K; see the comment
//! on `quantize_q6_k` in src/quant/mod.rs). The downstream tolerance in
//! the real model is dominated by the per-block round-off; a few units
//! of last-place error per accumulated element is expected. Test data
//! is chosen small enough that accumulated errors stay below 5e-2.

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

#[test]
fn q6k_gemv_matches_cpu_reference() {
    // Shape mirrors a Qwen-3B Q6_K projection (kv_dim × hidden).
    let rows = 256usize;
    let cols = 2048usize;

    // Build random weight as f32, quantize to Q6_K, then dequant back —
    // gives us a CPU reference that's bit-identical to what the GPU
    // kernel decodes from the same Q6_K bytes.
    let w_f32 = fixed_f32(rows * cols, 0xC0DEC0DE);
    let blocks = (rows * cols) / 256;
    let mut w_q6 = vec![0u8; blocks * quant::Q6_K_BLOCK_BYTES];
    quant::quantize_q6_k(&w_f32, &mut w_q6).expect("Q6_K quant");

    // Reconstruct CPU view of the matrix from the Q6_K bytes (so we
    // compare GPU GEMV vs a CPU GEMV that uses the *same* dequant).
    let mut w_recon = vec![0.0f32; rows * cols];
    quant::dequant_into(
        dismantle_core::gguf::GgmlType::Q6_K,
        &w_q6,
        &mut w_recon,
    )
    .expect("Q6_K dequant");

    let x = fixed_f32(cols, 0xBEEFBEEF);
    let mut expected = vec![0.0f32; rows];
    // y = w_recon @ x  (row-major)
    for r in 0..rows {
        let mut acc = 0.0f32;
        let row = &w_recon[r * cols..(r + 1) * cols];
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        expected[r] = acc;
    }

    // GPU path.
    let ctx = ctx();
    let model_buf = ctx.new_buffer_with_bytes(&w_q6);
    let x_buf = new_f32_buf(ctx, &x);
    let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q6_k_pinned_tcb(
            &mut tcb,
            &model_buf,
            0,
            w_q6.len(),
            rows,
            cols,
            &x_buf,
            &out_buf,
        )
        .expect("gemv_q6_k encode");
        tcb.commit_and_wait().expect("commit");
    }

    let actual = read_f32_buf(&out_buf, rows);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 5e-2, "q6_k gemv max_abs_diff = {diff} (limit 5e-2)");
}
