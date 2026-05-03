//! Phase 2 / W1B — MLA / Q-LoRA gemv parity tests.
//!
//! W1B routes the four MLA fp32 gemv call sites in
//! `model::deepseek_v2::attention` (q_a_proj, q_b_proj,
//! kv_a_proj_with_mqa, kv_b_proj) through `gemv_f32_attn_dispatch`,
//! which lands them on `gemv_f32_attn_metal` under
//! `cfg(target_os = "macos")` + `Some(metal_ctx)`.
//!
//! `gemv_f32_attn_metal` is already attested at atol=1e-3 fp16 by the
//! G1.3 parity test in `phase1_kernel_parity.rs` for one shape
//! (2048×2048). This test exercises the kernel on the four
//! MLA-specific shapes from DeepSeek-V2-Lite to catch any
//! shape-edge bugs the production gemv would expose.
//!
//! Shapes (rows × cols, where rows = output dim, cols = input dim):
//! - q_a_proj            : 1536 × 2048
//! - q_b_proj            : 3072 × 1536
//! - kv_a_proj_with_mqa  :  576 × 2048
//! - kv_b_proj           : 2048 ×  512

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::MetalContext;
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

const ATOL: f32 = 1e-3;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device on M3 Pro"));
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

fn parity_check(name: &'static str, rows: usize, cols: usize, seed_x: u64, seed_w: u64) {
    let x = fixed_input(cols, seed_x);
    let w = fixed_input(rows * cols, seed_w);

    let mut cpu_out = vec![0.0_f32; rows];
    kernels::gemv_f32(&w, rows, cols, &x, &mut cpu_out);

    let ctx = ctx().clone();
    let mut metal_out = vec![0.0_f32; rows];
    kernels::gemv_f32_attn_metal(&ctx, &w, rows, cols, &x, &mut metal_out)
        .expect("gemv_f32_attn_metal should succeed");

    let diff = max_abs_diff(&cpu_out, &metal_out);
    println!("[W1B] {name} ({rows}x{cols}) parity max abs diff = {diff:.6}");
    assert!(diff < ATOL, "{name} CPU/Metal diff {diff} >= atol {ATOL}");
}

#[test]
fn test_q_a_proj_shape_matches_cpu() {
    parity_check("q_a_proj", 1536, 2048, 0x1A1A_1A1A, 0x1B1B_1B1B);
}

#[test]
fn test_q_b_proj_shape_matches_cpu() {
    parity_check("q_b_proj", 3072, 1536, 0x2A2A_2A2A, 0x2B2B_2B2B);
}

#[test]
fn test_kv_a_proj_shape_matches_cpu() {
    parity_check("kv_a_proj_with_mqa", 576, 2048, 0x3A3A_3A3A, 0x3B3B_3B3B);
}

#[test]
fn test_kv_b_proj_shape_matches_cpu() {
    parity_check("kv_b_proj", 2048, 512, 0x4A4A_4A4A, 0x4B4B_4B4B);
}
