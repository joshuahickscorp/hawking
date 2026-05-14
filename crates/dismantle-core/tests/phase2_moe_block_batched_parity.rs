//! Phase 2 — batched MoE expert GEMV parity.
//!
//! These tests cover the practical FlashDMoE precursor: selected
//! expert matrices are packed route-major and dispatched as batched
//! GEMVs. The full block test compares the new Metal path against the
//! existing CPU route loop on small fixed inputs.

#![cfg(target_os = "macos")]

use dismantle_core::gguf::GgmlType;
use dismantle_core::kernels;
use dismantle_core::metal::MetalContext;
use dismantle_core::quant::dequant_into;
use half::f16;
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

const ATOL: f32 = 1e-3;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> = Lazy::new(|| MetalContext::new().expect("Metal device"));
    &CTX
}

fn fixed_input(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
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

fn synthetic_q8_0_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 34];
    for b in 0..n_blocks {
        let off = b * 34;
        let d = 0.001 + rng.gen::<f32>() * 0.001;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        for i in 0..32 {
            let q = rng.gen_range(-16i8..=16i8);
            bytes[off + 2 + i] = q as u8;
        }
    }
    bytes
}

fn synthetic_q6_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 210];
    for b in 0..n_blocks {
        let off = b * 210;
        for i in 0..192 {
            bytes[off + i] = rng.gen::<u8>();
        }
        for i in 0..16 {
            let s = rng.gen_range(-4i8..=4i8);
            bytes[off + 192 + i] = s as u8;
        }
        let d = 0.0005 + rng.gen::<f32>() * 0.0005;
        bytes[off + 208..off + 210].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    }
    bytes
}

fn cpu_batched_gemv(
    dtype: GgmlType,
    bytes: &[u8],
    routes: usize,
    rows: usize,
    cols: usize,
    x: &[f32],
    shared_x: bool,
) -> Vec<f32> {
    let elems_per_matrix = rows * cols;
    let bytes_per_matrix = bytes.len() / routes;
    let mut out = vec![0.0f32; routes * rows];
    for r in 0..routes {
        let mut w = vec![0.0f32; elems_per_matrix];
        let wb = &bytes[r * bytes_per_matrix..(r + 1) * bytes_per_matrix];
        dequant_into(dtype, wb, &mut w).expect("synthetic dequant");
        let xs = if shared_x {
            x
        } else {
            &x[r * cols..(r + 1) * cols]
        };
        kernels::gemv_f32(&w, rows, cols, xs, &mut out[r * rows..(r + 1) * rows]);
    }
    out
}

#[test]
fn test_batched_q4_gemv_matches_cpu() {
    let routes = 3;
    let rows = 64;
    let cols = 256;
    let bytes = synthetic_q4_k_bytes(routes * rows * (cols / 256), 0xA401);
    let x = fixed_input(cols, 0xA402, 0.05);

    let cpu = cpu_batched_gemv(GgmlType::Q4_K, &bytes, routes, rows, cols, &x, true);
    let mut metal = vec![0.0f32; routes * rows];
    kernels::moe_batched_gemm_q4_metal(ctx(), &bytes, routes, rows, cols, &x, &mut metal)
        .expect("batched q4 metal");

    let diff = max_abs_diff(&cpu, &metal);
    println!("[P2] batched q4 max abs diff = {diff:.6}");
    assert!(diff < ATOL, "batched q4 diff {diff} >= {ATOL}");
}

#[test]
fn test_batched_q8_gemv_matches_cpu() {
    let routes = 3;
    let rows = 64;
    let cols = 256;
    let bytes = synthetic_q8_0_bytes(routes * rows * (cols / 32), 0xB801);
    let x = fixed_input(routes * cols, 0xB802, 0.05);

    let cpu = cpu_batched_gemv(GgmlType::Q8_0, &bytes, routes, rows, cols, &x, false);
    let mut metal = vec![0.0f32; routes * rows];
    kernels::moe_batched_gemm_q8_0_metal(ctx(), &bytes, routes, rows, cols, &x, &mut metal)
        .expect("batched q8 metal");

    let diff = max_abs_diff(&cpu, &metal);
    println!("[P2] batched q8_0 max abs diff = {diff:.6}");
    assert!(diff < ATOL, "batched q8_0 diff {diff} >= {ATOL}");
}

#[test]
fn test_batched_q6_gemv_matches_cpu() {
    let routes = 3;
    let rows = 64;
    let cols = 256;
    let bytes = synthetic_q6_k_bytes(routes * rows * (cols / 256), 0xC601);
    let x = fixed_input(routes * cols, 0xC602, 0.05);

    let cpu = cpu_batched_gemv(GgmlType::Q6_K, &bytes, routes, rows, cols, &x, false);
    let mut metal = vec![0.0f32; routes * rows];
    kernels::moe_batched_gemm_q6_k_metal(ctx(), &bytes, routes, rows, cols, &x, &mut metal)
        .expect("batched q6 metal");

    let diff = max_abs_diff(&cpu, &metal);
    println!("[P2] batched q6_k max abs diff = {diff:.6}");
    assert!(diff < ATOL, "batched q6_k diff {diff} >= {ATOL}");
}

