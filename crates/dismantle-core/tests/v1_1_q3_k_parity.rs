//! v1.1.0 Phase 1A — Q3_K Metal GEMV parity against scalar dequant.

#![cfg(target_os = "macos")]

use dismantle_core::gguf::GgmlType;
use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer};
use dismantle_core::quant::dequant_into;
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

const ATOL: f32 = 1e-2;

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

fn pin(ctx: &MetalContext, bytes: &[u8]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytes)
}

fn pack_q3_scale(block: &mut [u8], scale_idx: usize, signed_scale: i8) {
    let l = (signed_scale + 32) as u8;
    if scale_idx < 8 {
        block[96 + scale_idx] |= l & 0x0f;
    } else {
        block[96 + scale_idx - 8] |= (l & 0x0f) << 4;
    }
    block[104 + scale_idx % 4] |= (l >> 4) << (2 * (scale_idx / 4));
}

fn synthetic_q3_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    use half::f16;
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 110];
    for b in 0..n_blocks {
        let off = b * 110;
        for i in 0..108 {
            bytes[off + i] = rng.gen::<u8>();
        }
        let d = 0.004 + rng.gen::<f32>() * 0.004;
        bytes[off + 108..off + 110].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    }
    bytes
}

fn zero_q3_k_bytes(n_blocks: usize) -> Vec<u8> {
    vec![0u8; n_blocks * 110]
}

fn ones_q3_k_bytes(n_blocks: usize) -> Vec<u8> {
    use half::f16;
    let mut bytes = vec![0xffu8; n_blocks * 110];
    for b in 0..n_blocks {
        let off = b * 110;
        bytes[off + 108..off + 110].copy_from_slice(&f16::from_f32(0.002).to_bits().to_le_bytes());
    }
    bytes
}

fn known_q3_k_bytes(n_blocks: usize) -> Vec<u8> {
    use half::f16;
    let mut bytes = vec![0u8; n_blocks * 110];
    for b in 0..n_blocks {
        let off = b * 110;
        let block = &mut bytes[off..off + 110];
        for scale_idx in 0..16 {
            pack_q3_scale(block, scale_idx, 1);
        }
        block[108..110].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
        for i in 0..32 {
            block[i] = if i % 2 == 0 { 0xff } else { 0x00 };
        }
        for i in 0..64 {
            block[32 + i] = match i % 4 {
                0 => 0b1110_0100,
                1 => 0b0001_1011,
                2 => 0b0101_0101,
                _ => 0b1010_1010,
            };
        }
    }
    bytes
}

fn assert_q3_gemv_matches_scalar(rows: usize, cols: usize, w_bytes: &[u8], seed: u64, label: &str) {
    let x = fixed_input(cols, seed);

    let mut w_f32 = vec![0.0_f32; rows * cols];
    dequant_into(GgmlType::Q3_K, w_bytes, &mut w_f32).expect("Q3_K scalar dequant");
    let mut scalar_out = vec![0.0_f32; rows];
    kernels::gemv_f32(&w_f32, rows, cols, &x, &mut scalar_out);

    let ctx = ctx();
    let model_buf = pin(ctx, w_bytes);
    let mut metal_out = vec![0.0_f32; rows];
    kernels::gemv_q3_k_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut metal_out)
        .expect("Q3_K Metal GEMV");

    let diff = max_abs_diff(&scalar_out, &metal_out);
    println!("[v1.1.0] Q3_K {label} rows={rows} cols={cols} max abs diff = {diff:.6e}");
    assert!(
        diff < ATOL,
        "Q3_K {label} diff {diff:.6e} >= atol {ATOL}"
    );
}

#[test]
fn q3_k_metal_matches_scalar_known_patterns() {
    let rows = 16;
    let cols = 256;
    let n_blocks = rows * (cols / 256);
    assert_q3_gemv_matches_scalar(rows, cols, &zero_q3_k_bytes(n_blocks), 0xA, "zero");
    assert_q3_gemv_matches_scalar(rows, cols, &ones_q3_k_bytes(n_blocks), 0xB, "ones");
    assert_q3_gemv_matches_scalar(rows, cols, &known_q3_k_bytes(n_blocks), 0xC, "known");
}

#[test]
fn q3_k_metal_matches_scalar_random_small() {
    let rows = 64;
    let cols = 256;
    let n_blocks = rows * (cols / 256);
    let w_bytes = synthetic_q3_k_bytes(n_blocks, 42);
    assert_q3_gemv_matches_scalar(rows, cols, &w_bytes, 0xDEAD_BEEF, "random-small");
}

#[test]
fn q3_k_metal_matches_scalar_random_realistic() {
    let rows = 256;
    let cols = 2048;
    let n_blocks = rows * (cols / 256);
    let w_bytes = synthetic_q3_k_bytes(n_blocks, 0xCAFE_BABE);
    assert_q3_gemv_matches_scalar(rows, cols, &w_bytes, 0x1234_5678, "random-realistic");
}
