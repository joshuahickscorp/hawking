//! q4k_predec_f16s — relative parity between the f32-scales predec GEMV
//! (gemv_q4_k_v4_predec_pinned_tcb) and the f16-scales variant
//! (gemv_q4_k_v4_predec_2r_f16s_pinned_tcb, Stage-2 bandwidth lever 1.2).
//!
//! Unlike q4k_predec_parity (which asserts BIT-identity between the inline and
//! f32-predec paths), this is NOT bit-identical: storing the pre-decoded
//! `(ds, dm)` pairs as f16 rounds each by ~half-mantissa (≈5e-4 relative). So
//! this is a QUALITY gate: the f16-scales output must track the f32-scales
//! output within the f16 precision budget. The f32 reference is dispatched via
//! the production wrapper; all f32 predec row-variants (1r/2r/4r) are
//! bit-identical to each other, so the only delta measured here is the f16
//! scale rounding, regardless of any DISMANTLE_QWEN_PREDEC_* env state.
//!
//! Gate = relative L2 norm of the difference (robust to individual near-zero
//! outputs from cancellation). f16 scale rounding keeps this well under 1e-2.
//! GPU-gated (needs a Metal device).

#![cfg(target_os = "macos")]

use dismantle_core::kernels::{self, predecode_q4_k_scale_table_f16};
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use half::f16;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

mod common;
use common::*;

/// Realistic Q4_K weights (144 B/block): small fp16 d/dmin, random sub-block
/// 6-bit indices and 4-bit quants. Same generator as q4k_predec_parity.rs.
fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n_blocks = rows * (cols / 256);
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        let d = 0.01_f32 + rng.gen::<f32>() * 0.01;
        let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}

fn make_x(cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..cols).map(|_| rng.gen_range(-3.0_f32..3.0_f32)).collect()
}

/// Pin a Vec<f16> as raw little-endian bytes (no bytemuck Pod dependency on
/// half::f16); the f16s kernel reads buffer(1) as `device const half*`.
fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
    let bytes: Vec<u8> = data.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect();
    ctx.new_buffer_with_bytes(&bytes)
}

#[test]
fn q4k_v4_predec_f16s_relative_parity() {
    let rows = 2048_usize;
    let cols = 2048_usize;
    let ctx = ctx();

    let w_bytes = make_q4k_bytes(rows, cols, 0xF165_8E1E);
    let model_buf = ctx.new_buffer_with_bytes(&w_bytes);

    let x = make_x(cols, 0xCAFE_F00D);
    let x_buf = new_f32_buf(ctx, &x);

    // f32-scales reference (production predec wrapper). f32 table = 16 f32/block.
    let scales_f32 = kernels::predecode_q4_k_scale_table(&w_bytes);
    let scales_f32_buf = new_f32_buf(ctx, &scales_f32);
    let y_ref_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pinned_tcb(
            &mut tcb, &model_buf, 0, w_bytes.len(),
            &scales_f32_buf, 0,
            rows, cols, &x_buf, &y_ref_buf,
        ).expect("f32 predec encode");
        tcb.commit_and_wait().expect("f32 predec commit");
    }
    let y_ref = read_f32_buf(&y_ref_buf, rows);

    // f16-scales variant. f16 table = 16 halfs/block.
    let scales_f16 = predecode_q4_k_scale_table_f16(&w_bytes);
    assert_eq!(scales_f16.len(), rows * (cols / 256) * 16,
        "predecode_q4_k_scale_table_f16 length mismatch");
    let scales_f16_buf = new_f16_buf(ctx, &scales_f16);
    let y_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_2r_f16s_pinned_tcb(
            &mut tcb, &model_buf, 0, w_bytes.len(),
            &scales_f16_buf, 0,
            rows, cols, &x_buf, &y_f16_buf,
        ).expect("f16s predec encode");
        tcb.commit_and_wait().expect("f16s predec commit");
    }
    let y_f16 = read_f32_buf(&y_f16_buf, rows);

    // Relative L2 norm of the difference — the right metric for a lossy kernel
    // (robust to individual near-zero outputs from cancellation).
    let mut num = 0.0_f64; // ||ref - f16||^2
    let mut den = 0.0_f64; // ||ref||^2
    let mut max_abs = 0.0_f32;
    for i in 0..rows {
        let d = (y_ref[i] - y_f16[i]) as f64;
        num += d * d;
        den += (y_ref[i] as f64) * (y_ref[i] as f64);
        max_abs = max_abs.max((y_ref[i] - y_f16[i]).abs());
    }
    let rel_l2 = (num / den.max(1e-30)).sqrt();
    eprintln!(
        "[q4k_v4_predec_f16s parity] rel_L2={rel_l2:.3e} max_abs={max_abs:.3e} \
         (||ref||={:.3e})", den.sqrt()
    );
    // f16 scale rounding (~5e-4 relative per scale) keeps the whole-vector
    // relative error well under 1%. A failure here means the f16 table or the
    // shader widening is wrong, not just rounding.
    assert!(
        rel_l2 < 1e-2,
        "f16-scales predec rel_L2 {rel_l2:.3e} exceeds the 1e-2 f16 precision budget"
    );
}
