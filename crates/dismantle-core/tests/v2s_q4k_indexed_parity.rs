//! Parity test: moe_batched_gemm_q4_indexed_v2s vs v2 reference.
//!
//! Test 1: routes=2, rows=64,  cols=256  (sub-TG sanity)
//! Test 2: routes=4, rows=256, cols=2048 (realistic gate/up shape)
//! Test 3: rows=70,  cols=256            (partial-TG boundary)

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::MetalContext;
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use half::f16;

const ATOL: f32 = 2e-5;

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
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).fold(0.0_f32, f32::max)
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
    let route_ids: Vec<u32> = (0..routes).map(|i| ((i * 2 + 1) % n_experts) as u32).collect();
    let x = fixed_input(cols, seed_base ^ 0xDEAD_BEEF);

    let mut v2_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_raw(
        ctx(), true, &model_bytes, base_offset, &route_ids, &x, routes, rows, cols, &mut v2_out,
    ).expect("v2 dispatch");

    let mut v2s_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_v2s_raw(
        ctx(), &model_bytes, base_offset, &route_ids, &x, routes, rows, cols, &mut v2s_out,
    ).expect("v2s dispatch");

    let mut v2t_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_v2t_raw(
        ctx(), &model_bytes, base_offset, &route_ids, &x, routes, rows, cols, &mut v2t_out,
    ).expect("v2t dispatch");

    let diff_s = max_abs_diff(&v2_out, &v2s_out);
    let diff_t = max_abs_diff(&v2_out, &v2t_out);
    println!("[v2s parity] routes={routes} rows={rows} cols={cols} v2s_diff={diff_s:.6e} v2t_diff={diff_t:.6e}");
    assert!(diff_s < ATOL, "v2s vs v2 diff {diff_s:.6e} >= atol {ATOL}");
    assert!(diff_t < ATOL, "v2t vs v2 diff {diff_t:.6e} >= atol {ATOL}");
}

#[test]
fn test_v2s_small() { run_parity(2, 64, 256, 0xBEEF_0001); }

#[test]
fn test_v2s_realistic() { run_parity(4, 256, 2048, 0xBEEF_0002); }

#[test]
fn test_v2s_partial_tg() { run_parity(2, 70, 256, 0xBEEF_0003); }

// ── v2t_gu parity ────────────────────────────────────────────────────────────

const ATOL_GU: f32 = 2e-4;

fn silu_f32(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

fn run_gu_parity(routes: usize, rows: usize, cols: usize, seed_base: u64) {
    let n_experts = routes + 3;
    let blocks_per_row = cols / 256;
    let blocks_per_expert = rows * blocks_per_row;
    let _n_bytes = n_experts * blocks_per_expert * 144;

    let gate_bytes = synthetic_q4_k_bytes(n_experts * blocks_per_expert, seed_base);
    let up_bytes   = synthetic_q4_k_bytes(n_experts * blocks_per_expert, seed_base ^ 0xCAFE_BABE);

    let mut model_bytes = vec![0xA5u8; 64];
    let gate_offset = model_bytes.len();
    model_bytes.extend_from_slice(&gate_bytes);
    let up_offset = model_bytes.len();
    model_bytes.extend_from_slice(&up_bytes);

    let route_ids: Vec<u32> = (0..routes).map(|i| ((i * 2 + 1) % n_experts) as u32).collect();
    let x = fixed_input(cols, seed_base ^ 0xDEAD_BEEF);

    // Reference: v2t gate + v2t up + CPU silu_mul
    let mut gate_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_v2t_raw(
        ctx(), &model_bytes, gate_offset, &route_ids, &x, routes, rows, cols, &mut gate_out,
    ).expect("v2t gate");

    let mut up_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_v2t_raw(
        ctx(), &model_bytes, up_offset, &route_ids, &x, routes, rows, cols, &mut up_out,
    ).expect("v2t up");

    let ref_act: Vec<f32> = gate_out.iter().zip(up_out.iter())
        .map(|(&g, &u)| silu_f32(g) * u)
        .collect();

    // Fused kernel
    let mut gu_act = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_v2t_gu_raw(
        ctx(), &model_bytes, gate_offset, up_offset,
        &route_ids, &x, routes, rows, cols, &mut gu_act,
    ).expect("v2t_gu");

    let diff = max_abs_diff(&ref_act, &gu_act);
    println!("[v2t_gu parity] routes={routes} rows={rows} cols={cols} diff={diff:.6e}");
    assert!(diff < ATOL_GU, "v2t_gu diff {diff:.6e} >= atol {ATOL_GU}");
}

#[test]
fn test_v2t_gu_small() { run_gu_parity(2, 64, 256, 0xFACE_0001); }

#[test]
fn test_v2t_gu_realistic() { run_gu_parity(4, 256, 2048, 0xFACE_0002); }

#[test]
fn test_v2t_gu_partial_tg() { run_gu_parity(2, 70, 256, 0xFACE_0003); }
