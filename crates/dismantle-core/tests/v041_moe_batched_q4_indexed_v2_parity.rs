//! v0.4.1 — Parity test: moe_batched_gemm_q4_indexed_v2 vs scalar reference.
//!
//! Test 1: routes=2, rows=64,  cols=256  (sub-TG sanity)
//! Test 2: routes=4, rows=256, cols=2048 (realistic shape)
//! Asserts max |scalar - v2| < 1e-3 (fp16 quant noise tolerance).

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use half::f16;

mod common;
use common::*;

fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

fn synthetic_q4_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        let d = 0.001 + rng.gen::<f32>() * 0.001;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        let dmin = (rng.gen::<f32>() - 0.5) * 0.001;
        bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}

fn run_parity(routes: usize, rows: usize, cols: usize, seed_base: u64) {
    let n_experts = routes + 3;
    let blocks_per_expert = rows * (cols / 256);
    let bytes_per_expert = blocks_per_expert * 144;

    // Build full fused tensor: n_experts consecutive matrices.
    let fused = synthetic_q4_k_bytes(n_experts * blocks_per_expert, seed_base);

    // Prepend 64 bytes of padding to exercise base_offset != 0.
    let mut model_bytes = vec![0xA5u8; 64];
    let base_offset = model_bytes.len();
    model_bytes.extend_from_slice(&fused);

    // Select experts (spread across available ids).
    let route_ids: Vec<u32> = (0..routes)
        .map(|i| ((i * 2 + 1) % n_experts) as u32)
        .collect();

    let x = fixed_input(cols, seed_base ^ 0xDEAD_BEEF);

    let mut scalar_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_raw(
        ctx(),
        false,
        &model_bytes,
        base_offset,
        &route_ids,
        &x,
        routes,
        rows,
        cols,
        &mut scalar_out,
    )
    .expect("scalar dispatch should succeed");

    let mut v2_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_raw(
        ctx(),
        true,
        &model_bytes,
        base_offset,
        &route_ids,
        &x,
        routes,
        rows,
        cols,
        &mut v2_out,
    )
    .expect("v2 dispatch should succeed");

    let diff = max_abs_diff(&scalar_out, &v2_out);
    println!(
        "[v0.4.1] indexed q4k parity (routes={routes} rows={rows} cols={cols}) max abs diff = {diff:.6e}"
    );
    // Verify bytes_per_expert is used (suppress unused warning).
    let _ = bytes_per_expert;
    assert!(
        diff < ATOL,
        "moe_batched_gemm_q4_indexed_v2 vs scalar diff {diff:.6e} >= atol {ATOL}"
    );
}

#[test]
fn test_indexed_q4k_v2_small() {
    run_parity(2, 64, 256, 0x4100_0001);
}

#[test]
fn test_indexed_q4k_v2_realistic() {
    run_parity(4, 256, 2048, 0x4100_0002);
}
