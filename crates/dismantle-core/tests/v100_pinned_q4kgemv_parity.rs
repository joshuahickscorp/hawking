//! Wedge A parity: gemv_q4_k_m_v2_pinned matches gemv_q4_k_m_v2 at atol=1e-5.
//! Both paths dispatch the same `gemm_q4_k_m_fused_v2` kernel; the only
//! difference is whether weights are memcpy'd into a fresh buffer or read
//! from a pre-pinned buffer via byte offset. Outputs should be bit-identical.
#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer};
use rand::Rng;
use rand_pcg::Pcg64Mcg;

mod common;
use common::*;

fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

fn synthetic_q4_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    use half::f16;
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
        for i in 4..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}

fn pinned_from_bytes(ctx: &MetalContext, bytes: &[u8]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytes)
}

#[test]
fn pinned_q4kgemv_small() {
    let rows = 64;
    let cols = 256;
    let n_blocks = rows * (cols / 256);
    let w_bytes = synthetic_q4_k_bytes(n_blocks, 42);
    let x = fixed_input(cols, 0xDEAD_BEEF);

    let ctx = ctx();

    let mut copy_out = vec![0.0f32; rows];
    kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut copy_out)
        .expect("copy path should succeed");

    let model_buf = pinned_from_bytes(ctx, &w_bytes);
    let mut pinned_out = vec![0.0f32; rows];
    kernels::gemv_q4_k_m_v2_pinned(
        ctx,
        &model_buf,
        0,
        w_bytes.len(),
        rows,
        cols,
        &x,
        &mut pinned_out,
    )
    .expect("pinned path should succeed");

    let diff = max_abs_diff(&copy_out, &pinned_out);
    println!("[WedgeA] pinned vs copy small (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
    assert!(
        diff < 1e-5,
        "pinned vs copy diff {diff:.2e} >= 1e-5 (should be bit-identical)"
    );
}

#[test]
fn pinned_q4kgemv_realistic() {
    let rows = 512;
    let cols = 2048;
    let n_blocks = rows * (cols / 256);
    let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xCAFE_BABE);
    let x = fixed_input(cols, 0x1234_5678);

    let ctx = ctx();

    let mut copy_out = vec![0.0f32; rows];
    kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut copy_out)
        .expect("copy path should succeed");

    let model_buf = pinned_from_bytes(ctx, &w_bytes);
    let mut pinned_out = vec![0.0f32; rows];
    kernels::gemv_q4_k_m_v2_pinned(
        ctx,
        &model_buf,
        0,
        w_bytes.len(),
        rows,
        cols,
        &x,
        &mut pinned_out,
    )
    .expect("pinned path should succeed");

    let diff = max_abs_diff(&copy_out, &pinned_out);
    println!(
        "[WedgeA] pinned vs copy realistic (rows={rows} cols={cols}) max abs diff = {diff:.2e}"
    );
    assert!(
        diff < 1e-5,
        "pinned vs copy diff {diff:.2e} >= 1e-5 (should be bit-identical)"
    );
}

#[test]
fn pinned_q4kgemv_nonzero_offset() {
    let rows = 128;
    let cols = 512;
    let n_blocks = rows * (cols / 256);
    let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xBEEF_CAFE);
    let x = fixed_input(cols, 0xABCD_1234);

    let pad = 1024usize;
    let mut padded = vec![0xFFu8; pad];
    padded.extend_from_slice(&w_bytes);

    let ctx = ctx();

    let mut copy_out = vec![0.0f32; rows];
    kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut copy_out).expect("copy path");

    let model_buf = pinned_from_bytes(ctx, &padded);
    let mut pinned_out = vec![0.0f32; rows];
    kernels::gemv_q4_k_m_v2_pinned(
        ctx,
        &model_buf,
        pad,
        w_bytes.len(),
        rows,
        cols,
        &x,
        &mut pinned_out,
    )
    .expect("pinned path with offset");

    let diff = max_abs_diff(&copy_out, &pinned_out);
    println!("[WedgeA] pinned vs copy nonzero-offset (pad={pad}) max abs diff = {diff:.2e}");
    assert!(diff < 1e-5, "offset pinned vs copy diff {diff:.2e} >= 1e-5");
}
