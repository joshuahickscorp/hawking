#![cfg(target_os = "macos")]
use hawking_core::kernels;
use rand::Rng;
use rand_pcg::Pcg64Mcg;
mod common;
use common::*;
pub const ATOL: f32 = 1e-3;
fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}
#[test]
fn test_rmsnorm_matches_cpu() {
    let hidden = 4096;
    let x = fixed_input(hidden, 0xCAFEBABE);
    let w = fixed_input(hidden, 0xDEADBEEF);
    let eps = 1e-6_f32;
    let mut cpu_out = vec![0.0_f32; hidden];
    kernels::rmsnorm(&x, &w, eps, &mut cpu_out);
    let ctx = ctx().clone();
    let mut metal_out = vec![0.0_f32; hidden];
    kernels::rmsnorm_metal(&ctx, &x, &w, eps, &mut metal_out)
        .expect("rmsnorm_metal should succeed once G1.1 lands");
    let diff = max_abs_diff(&cpu_out, &metal_out);
    assert!(diff < ATOL, "rmsnorm CPU/Metal diff {diff} >= atol {ATOL}");
}

#[test]
fn test_llama_b9430_rmsnorm_matches_cpu() {
    let hidden = 4096;
    let x = fixed_input(hidden, 0xA11A_B943);
    let w = fixed_input(hidden, 0xF32A_B943);
    let eps = 1e-6_f32;
    let mut cpu_out = vec![0.0_f32; hidden];
    kernels::rmsnorm(&x, &w, eps, &mut cpu_out);

    let ctx = ctx().clone();
    let mut metal_out = vec![0.0_f32; hidden];
    kernels::rmsnorm_llama_b9430(&ctx, &x, &w, eps, &mut metal_out)
        .expect("llama b9430 RMSNorm Metal dispatch should succeed");

    let diff = max_abs_diff(&cpu_out, &metal_out);
    assert!(
        diff < ATOL,
        "llama b9430 RMSNorm CPU/Metal diff {diff} >= atol {ATOL}"
    );
}
#[test]
fn test_gemv_f16_matches_cpu() {
    use half::f16;
    let rows = 4096;
    let cols = 2048;
    let x = fixed_input(cols, 0xA1A1A1A1);
    let w_f32 = fixed_input(rows * cols, 0xB2B2B2B2);
    let w_f16: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let mut cpu_out = vec![0.0_f32; rows];
    kernels::gemv_f16(&w_f16, rows, cols, &x, &mut cpu_out);
    let ctx = ctx().clone();
    let w_bytes: &[u8] = bytemuck::cast_slice(&w_f16);
    let mut metal_out = vec![0.0_f32; rows];
    kernels::gemv_f16_metal(&ctx, w_bytes, rows, cols, &x, &mut metal_out)
        .expect("gemv_f16_metal should succeed once G1.2 lands");
    let diff = max_abs_diff(&cpu_out, &metal_out);
    assert!(diff < ATOL, "gemv_f16 CPU/Metal diff {diff} >= atol {ATOL}");
}
#[test]
fn test_gemv_f16_argmax_pinned_matches_cpu() {
    use half::f16;
    let rows = 1024;
    let cols = 512;
    let x = fixed_input(cols, 0x1234ABCD);
    let w_f32 = fixed_input(rows * cols, 0x4567DCBA);
    let w_f16: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let mut cpu_logits = vec![0.0_f32; rows];
    kernels::gemv_f16(&w_f16, rows, cols, &x, &mut cpu_logits);
    let cpu = kernels::argmax_f32(&cpu_logits);
    let ctx = ctx().clone();
    let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&w_f16));
    let metal = kernels::gemv_f16_argmax_metal_pinned(&ctx, &w_buf, rows, cols, &x)
        .expect("gemv_f16_argmax_metal_pinned should return token id");
    assert_eq!(cpu, metal);
}
#[test]
fn test_gemv_f32_attn_matches_cpu() {
    let rows = 2048;
    let cols = 2048;
    let x = fixed_input(cols, 0xC3C3C3C3);
    let w = fixed_input(rows * cols, 0xD4D4D4D4);
    let mut cpu_out = vec![0.0_f32; rows];
    kernels::gemv_f32(&w, rows, cols, &x, &mut cpu_out);
    let ctx = ctx().clone();
    let mut metal_out = vec![0.0_f32; rows];
    kernels::gemv_f32_attn_metal(&ctx, &w, rows, cols, &x, &mut metal_out)
        .expect("gemv_f32_attn_metal should succeed once G1.3 lands");
    let diff = max_abs_diff(&cpu_out, &metal_out);
    assert!(
        diff < ATOL,
        "gemv_f32_attn CPU/Metal diff {diff} >= atol {ATOL}"
    );
}
#[test]
fn test_gemv_f32_moe_matches_cpu() {
    let rows = 64;
    let cols = 2048;
    let x = fixed_input(cols, 0xE5E5E5E5);
    let w = fixed_input(rows * cols, 0xF6F6F6F6);
    let mut cpu_out = vec![0.0_f32; rows];
    kernels::gemv_f32(&w, rows, cols, &x, &mut cpu_out);
    let ctx = ctx().clone();
    let mut metal_out = vec![0.0_f32; rows];
    kernels::gemv_f32_moe_metal(&ctx, &w, rows, cols, &x, &mut metal_out)
        .expect("gemv_f32_moe_metal should succeed once G1.4 lands");
    let diff = max_abs_diff(&cpu_out, &metal_out);
    assert!(
        diff < ATOL,
        "gemv_f32_moe CPU/Metal diff {diff} >= atol {ATOL}"
    );
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
#[test]
fn test_gemm_q4_k_m_fused_matches_cpu() {
    use hawking_core::gguf::GgmlType;
    use hawking_core::quant::dequant_into;
    let rows = 64;
    let cols = 256;
    let blocks = rows * (cols / 256);
    let w_bytes = synthetic_q4_k_bytes(blocks, 0xE6E6E6E6);
    let x = fixed_input(cols, 0xF7F7F7F7);
    let mut w_f32 = vec![0.0_f32; rows * cols];
    dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32)
        .expect("Q4_K dequant should succeed for valid synthetic bytes");
    let mut cpu_out = vec![0.0_f32; rows];
    kernels::gemv_f32(&w_f32, rows, cols, &x, &mut cpu_out);
    let ctx = ctx().clone();
    let mut metal_out = vec![0.0_f32; rows];
    kernels::gemv_q4_k_m(&ctx, &w_bytes, rows, cols, &x, &mut metal_out)
        .expect("gemv_q4_k_m should succeed once H2.4 lands");
    let diff = max_abs_diff(&cpu_out, &metal_out);
    assert!(
        diff < ATOL,
        "gemm_q4_k_m_fused CPU/Metal diff {diff} >= atol {ATOL}"
    );
}
