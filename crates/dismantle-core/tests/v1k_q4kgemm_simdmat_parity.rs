//! Wedge K parity: gemv_q4_k_m_simdmat_pinned vs gemv_q4_k_m_v2 at atol=1e-3.
//! Different summation order (paired nibble reads) may cause rounding differences
//! at Q4_K noise level; atol=1e-3 matches the fp16 quantization floor.
#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer};
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
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
        // scales: 12 bytes at off+4..off+15 (legal values: s_byte/m_byte ≤ 63)
        for i in 4..16 {
            bytes[off + i] = rng.gen::<u8>() & 0x3F;
        }
        // nibbles: 128 bytes at off+16..off+143
        for i in 16..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}

fn pinned_from_bytes(ctx: &MetalContext, bytes: &[u8]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytes)
}

#[test]
fn v1k_simdmat_vs_v2_small() {
    let rows = 64;
    let cols = 256;
    let n_blocks = rows * (cols / 256);
    let w_bytes = synthetic_q4_k_bytes(n_blocks, 42);
    let x = fixed_input(cols, 0xDEAD_BEEF);

    let ctx = ctx();

    let mut v2_out = vec![0.0f32; rows];
    kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut v2_out)
        .expect("v2 path should succeed");

    let model_buf = pinned_from_bytes(ctx, &w_bytes);
    let mut sm_out = vec![0.0f32; rows];
    kernels::gemv_q4_k_m_simdmat_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut sm_out)
        .expect("simdmat path should succeed");

    let diff = max_abs_diff(&v2_out, &sm_out);
    println!("[WedgeK] simdmat vs v2 small (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
    assert!(
        diff < 1e-3,
        "simdmat vs v2 diff {diff:.2e} >= 1e-3 (Q4_K noise floor)"
    );
}

#[test]
fn v1k_simdmat_vs_v2_realistic() {
    let rows = 512;
    let cols = 2048;
    let n_blocks = rows * (cols / 256);
    let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xCAFE_BABE);
    let x = fixed_input(cols, 0x1234_5678);

    let ctx = ctx();

    let mut v2_out = vec![0.0f32; rows];
    kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut v2_out)
        .expect("v2 path");

    let model_buf = pinned_from_bytes(ctx, &w_bytes);
    let mut sm_out = vec![0.0f32; rows];
    kernels::gemv_q4_k_m_simdmat_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut sm_out)
        .expect("simdmat path");

    let diff = max_abs_diff(&v2_out, &sm_out);
    println!("[WedgeK] simdmat vs v2 realistic (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
    assert!(
        diff < 1e-3,
        "simdmat vs v2 diff {diff:.2e} >= 1e-3"
    );
}

#[test]
fn v1k_simdmat_argmax_agrees() {
    // Argmax of simdmat output must match v2 output on a DeepSeek-V2-like shape.
    let rows = 128;
    let cols = 7168;
    let n_blocks = rows * (cols / 256);
    let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xBEEF_1234);
    let x = fixed_input(cols, 0xABCD_5678);

    let ctx = ctx();

    let mut v2_out = vec![0.0f32; rows];
    kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut v2_out)
        .expect("v2 path");

    let model_buf = pinned_from_bytes(ctx, &w_bytes);
    let mut sm_out = vec![0.0f32; rows];
    kernels::gemv_q4_k_m_simdmat_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut sm_out)
        .expect("simdmat path");

    let diff = max_abs_diff(&v2_out, &sm_out);
    println!("[WedgeK] simdmat vs v2 argmax shape (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
    assert!(
        diff < 1e-3,
        "simdmat vs v2 diff {diff:.2e} >= 1e-3 on argmax shape"
    );

    let v2_argmax = v2_out.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
    let sm_argmax = sm_out.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
    println!("[WedgeK] argmax: v2={v2_argmax} simdmat={sm_argmax}");
    assert_eq!(v2_argmax, sm_argmax, "argmax must match between v2 and simdmat");
}
