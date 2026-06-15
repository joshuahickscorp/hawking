//! Parity test: moe_batched_gemm_q4_indexed_v2s vs v2 reference.
//!
//! Test 1: routes=2, rows=64,  cols=256  (sub-TG sanity)
//! Test 2: routes=4, rows=256, cols=2048 (realistic gate/up shape)
//! Test 3: rows=70,  cols=256            (partial-TG boundary)

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use half::f16;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

mod common;
use common::*;

const ATOL: f32 = 2e-5;

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
    let blocks_per_row = cols / 256;
    let blocks_per_expert = rows * blocks_per_row;
    let fused = synthetic_q4_k_bytes(n_experts * blocks_per_expert, seed_base);
    let mut model_bytes = vec![0xA5u8; 64];
    let base_offset = model_bytes.len();
    model_bytes.extend_from_slice(&fused);
    let route_ids: Vec<u32> = (0..routes)
        .map(|i| ((i * 2 + 1) % n_experts) as u32)
        .collect();
    let x = fixed_input(cols, seed_base ^ 0xDEAD_BEEF);

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
    .expect("v2 dispatch");

    let mut v2s_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_v2s_raw(
        ctx(),
        &model_bytes,
        base_offset,
        &route_ids,
        &x,
        routes,
        rows,
        cols,
        &mut v2s_out,
    )
    .expect("v2s dispatch");

    let mut v2t_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_v2t_raw(
        ctx(),
        &model_bytes,
        base_offset,
        &route_ids,
        &x,
        routes,
        rows,
        cols,
        &mut v2t_out,
    )
    .expect("v2t dispatch");

    let diff_s = max_abs_diff(&v2_out, &v2s_out);
    let diff_t = max_abs_diff(&v2_out, &v2t_out);
    println!("[v2s parity] routes={routes} rows={rows} cols={cols} v2s_diff={diff_s:.6e} v2t_diff={diff_t:.6e}");
    assert!(diff_s < ATOL, "v2s vs v2 diff {diff_s:.6e} >= atol {ATOL}");
    assert!(diff_t < ATOL, "v2t vs v2 diff {diff_t:.6e} >= atol {ATOL}");
}

#[test]
fn test_v2s_small() {
    run_parity(2, 64, 256, 0xBEEF_0001);
}

#[test]
fn test_v2s_realistic() {
    run_parity(4, 256, 2048, 0xBEEF_0002);
}

#[test]
fn test_v2s_partial_tg() {
    run_parity(2, 70, 256, 0xBEEF_0003);
}

// ── v2t_gu parity ────────────────────────────────────────────────────────────

const ATOL_GU: f32 = 2e-4;

fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn run_gu_parity(routes: usize, rows: usize, cols: usize, seed_base: u64) {
    let n_experts = routes + 3;
    let blocks_per_row = cols / 256;
    let blocks_per_expert = rows * blocks_per_row;
    let _n_bytes = n_experts * blocks_per_expert * 144;

    let gate_bytes = synthetic_q4_k_bytes(n_experts * blocks_per_expert, seed_base);
    let up_bytes = synthetic_q4_k_bytes(n_experts * blocks_per_expert, seed_base ^ 0xCAFE_BABE);

    let mut model_bytes = vec![0xA5u8; 64];
    let gate_offset = model_bytes.len();
    model_bytes.extend_from_slice(&gate_bytes);
    let up_offset = model_bytes.len();
    model_bytes.extend_from_slice(&up_bytes);

    let route_ids: Vec<u32> = (0..routes)
        .map(|i| ((i * 2 + 1) % n_experts) as u32)
        .collect();
    let x = fixed_input(cols, seed_base ^ 0xDEAD_BEEF);

    // Reference: v2t gate + v2t up + CPU silu_mul
    let mut gate_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_v2t_raw(
        ctx(),
        &model_bytes,
        gate_offset,
        &route_ids,
        &x,
        routes,
        rows,
        cols,
        &mut gate_out,
    )
    .expect("v2t gate");

    let mut up_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_v2t_raw(
        ctx(),
        &model_bytes,
        up_offset,
        &route_ids,
        &x,
        routes,
        rows,
        cols,
        &mut up_out,
    )
    .expect("v2t up");

    let ref_act: Vec<f32> = gate_out
        .iter()
        .zip(up_out.iter())
        .map(|(&g, &u)| silu_f32(g) * u)
        .collect();

    // Fused kernel
    let mut gu_act = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_v2t_gu_raw(
        ctx(),
        &model_bytes,
        gate_offset,
        up_offset,
        &route_ids,
        &x,
        routes,
        rows,
        cols,
        &mut gu_act,
    )
    .expect("v2t_gu");

    let diff = max_abs_diff(&ref_act, &gu_act);
    println!("[v2t_gu parity] routes={routes} rows={rows} cols={cols} diff={diff:.6e}");
    assert!(diff < ATOL_GU, "v2t_gu diff {diff:.6e} >= atol {ATOL_GU}");
}

#[test]
fn test_v2t_gu_small() {
    run_gu_parity(2, 64, 256, 0xFACE_0001);
}

#[test]
fn test_v2t_gu_realistic() {
    run_gu_parity(4, 256, 2048, 0xFACE_0002);
}

#[test]
fn test_v2t_gu_partial_tg() {
    run_gu_parity(2, 70, 256, 0xFACE_0003);
}

// ── Q8_0 v2t parity ──────────────────────────────────────────────────────────
// Reference: CPU dequant of Q8_0 weights × route-major x.
// Kernel under test: moe_batched_gemm_q8_0_indexed_v2t.

const ATOL_Q8: f32 = 1e-4;

fn synthetic_q8_0_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 34];
    for b in 0..n_blocks {
        let off = b * 34;
        let d = 0.001 + rng.gen::<f32>() * 0.001;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        for i in 2..34 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}

fn cpu_q8_0_matvec(
    w_bytes: &[u8],
    base_offset: usize,
    route_ids: &[u32],
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    let blocks_per_row = cols / 32;
    let per_matrix_bytes = rows * blocks_per_row * 34;
    for (ri, &expert) in route_ids.iter().enumerate() {
        for row in 0..rows {
            let row_off =
                base_offset + expert as usize * per_matrix_bytes + row * blocks_per_row * 34;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let bo = row_off + b * 34;
                let d_bits = u16::from_le_bytes([w_bytes[bo], w_bytes[bo + 1]]);
                let d = f16::from_bits(d_bits).to_f32();
                for i in 0..32usize {
                    let qi = (w_bytes[bo + 2 + i] as i8) as f32;
                    let xi = x[ri * cols + b * 32 + i];
                    acc += d * qi * xi;
                }
            }
            out[ri * rows + row] = acc;
        }
    }
}

fn run_q8_parity(routes: usize, rows: usize, cols: usize, seed_base: u64) {
    let n_experts = routes + 3;
    let blocks_per_row = cols / 32;
    let blocks_per_expert = rows * blocks_per_row;
    let w_bytes = synthetic_q8_0_bytes(n_experts * blocks_per_expert, seed_base);
    let mut model_bytes = vec![0xA5u8; 64];
    let base_offset = model_bytes.len();
    model_bytes.extend_from_slice(&w_bytes);
    let route_ids: Vec<u32> = (0..routes)
        .map(|i| ((i * 2 + 1) % n_experts) as u32)
        .collect();
    // x is route-major: each route has its own cols-element slice
    let x = fixed_input(routes * cols, seed_base ^ 0xDEAD_BEEF);

    let mut ref_out = vec![0.0_f32; routes * rows];
    cpu_q8_0_matvec(
        &model_bytes,
        base_offset,
        &route_ids,
        &x,
        rows,
        cols,
        &mut ref_out,
    );

    let mut gpu_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q8_0_indexed_v2t_raw(
        ctx(),
        &model_bytes,
        base_offset,
        &route_ids,
        &x,
        routes,
        rows,
        cols,
        &mut gpu_out,
    )
    .expect("q8_0 v2t dispatch");

    let diff = max_abs_diff(&ref_out, &gpu_out);
    println!("[q8_0 v2t parity] routes={routes} rows={rows} cols={cols} diff={diff:.6e}");
    assert!(diff < ATOL_Q8, "q8_0 v2t diff {diff:.6e} >= atol {ATOL_Q8}");
}

#[test]
fn test_q8_0_v2t_small() {
    run_q8_parity(2, 64, 64, 0xB00B_0001);
}

#[test]
fn test_q8_0_v2t_realistic() {
    run_q8_parity(6, 2048, 1408, 0xB00B_0002);
}

#[test]
fn test_q8_0_v2t_partial_tg() {
    run_q8_parity(2, 70, 64, 0xB00B_0003);
}
