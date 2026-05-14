//! Phase 1 / Haul 1 — Numerical parity tests between CPU reference
//! kernels and Metal-dispatched kernels.
//!
//! **Status: SCAFFOLDING.** Each test below is `#[ignore]` until its
//! corresponding gate's haul item lands the implementation. The haul
//! removes the `#[ignore]` attribute when filling in the body.
//!
//! Every test must:
//!   1. Generate a fixed-seed input (so baselines are reproducible).
//!   2. Run the CPU reference kernel from `dismantle_core::kernels`.
//!   3. Run the Metal-dispatched kernel from
//!      `dismantle_core::kernels::metal_dispatch::*`.
//!   4. Assert max abs diff < `ATOL` (1e-3 fp16 quant noise).
//!
//! Common test plumbing is provided below so each gate's body is
//! short and obvious.

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::MetalContext;
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

/// fp16 absolute tolerance — about 1 part in 1024 (fp16 mantissa is
/// 10 bits + 1 implicit). Allows reduction-order sensitivity.
pub const ATOL: f32 = 1e-3;

/// Single shared Metal context across all parity tests in this file.
/// Avoids re-running device lookup + library compile + pipeline cache
/// init on every test — those are ~50-200ms each. Cargo runs the
/// 4 parity tests in the same binary, so they share this Lazy.
/// `MetalContext` is Clone (Arc-backed) so individual test bodies can
/// hold a `&'static MetalContext` directly.
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

// ---------------------------------------------------------------------
// G1.1 — Metal scaffold + rmsnorm round-trip
// ---------------------------------------------------------------------

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
    println!("[G1.1] rmsnorm parity max abs diff = {diff:.6}");
    assert!(diff < ATOL, "rmsnorm CPU/Metal diff {diff} >= atol {ATOL}");
}

// ---------------------------------------------------------------------
// G1.2 — LM-head GEMV (fp16 weights × fp32 vec → fp32 logits)
// ---------------------------------------------------------------------

#[test]
fn test_gemv_f16_matches_cpu() {
    use half::f16;

    // Smaller than the real LM head (vocab=102400 × hidden=2048) so
    // the parity test is fast; the size still exceeds one threadgroup
    // tile and exercises reduction logic.
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
    println!("[G1.2] gemv_f16 parity max abs diff = {diff:.6}");
    assert!(diff < ATOL, "gemv_f16 CPU/Metal diff {diff} >= atol {ATOL}");
}

#[test]
fn test_sample_argmax_f32_matches_cpu() {
    let mut logits = fixed_input(4097, 0x5A5A5A5A);
    logits[1024] = 7.0;
    logits[3072] = 7.0; // tie must keep the lower index, like CPU argmax.

    let cpu = kernels::argmax_f32(&logits);
    let ctx = ctx().clone();
    let metal = kernels::sample_argmax_f32_metal(&ctx, &logits)
        .expect("sample_argmax_f32_metal should return a token id");

    println!("[SAMPLE] argmax parity cpu={cpu} metal={metal}");
    assert_eq!(cpu, metal);
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

    println!("[SAMPLE] gemv_f16+argmax parity cpu={cpu} metal={metal}");
    assert_eq!(cpu, metal);
}

// ---------------------------------------------------------------------
// G1.3 — Attention o_proj GEMV (fp32 weights × fp32 vec → fp32 vec)
// ---------------------------------------------------------------------

#[test]
fn test_gemv_f32_attn_matches_cpu() {
    // o_proj: hidden × (n_heads × v_head_dim) = 2048 × 2048
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
    println!("[G1.3] gemv_f32 (attn) parity max abs diff = {diff:.6}");
    assert!(
        diff < ATOL,
        "gemv_f32_attn CPU/Metal diff {diff} >= atol {ATOL}"
    );
}

// ---------------------------------------------------------------------
// G1.4 — MoE gate-logit GEMV (fp32 weights × fp32 vec → fp32 logits)
// ---------------------------------------------------------------------

#[test]
fn test_gemv_f32_moe_matches_cpu() {
    // ffn_gate_inp: n_routed_experts × hidden = 64 × 2048
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
    println!("[G1.4] gemv_f32 (moe) parity max abs diff = {diff:.6}");
    assert!(
        diff < ATOL,
        "gemv_f32_moe CPU/Metal diff {diff} >= atol {ATOL}"
    );
}

// ---------------------------------------------------------------------
// H2.1 — top-K softmax gate (Wedge 2: MoE block, gate stage)
// ---------------------------------------------------------------------

#[test]
fn test_moe_topk_gate_matches_cpu() {
    let n_tokens = 16;
    let n_experts = 64;
    let top_k = 6;

    let logits = fixed_input(n_tokens * n_experts, 0xF7F7F7F7);

    let mut cpu_ids = vec![0u32; n_tokens * top_k];
    let mut cpu_weights = vec![0.0_f32; n_tokens * top_k];
    kernels::topk_softmax_batch(
        &logits,
        n_tokens,
        n_experts,
        top_k,
        &mut cpu_ids,
        &mut cpu_weights,
    );

    let ctx = ctx().clone();
    let mut metal_ids = vec![0u32; n_tokens * top_k];
    let mut metal_weights = vec![0.0_f32; n_tokens * top_k];
    kernels::moe_topk_gate_metal(
        &ctx,
        &logits,
        n_tokens,
        n_experts,
        top_k,
        &mut metal_ids,
        &mut metal_weights,
    )
    .expect("moe_topk_gate_metal should succeed once H2.1 lands");

    assert_eq!(cpu_ids, metal_ids, "expert ids must match exactly");

    let diff = max_abs_diff(&cpu_weights, &metal_weights);
    println!("[H2.1] moe_topk_gate parity max abs diff = {diff:.6}");
    assert!(
        diff < ATOL,
        "moe_topk_gate CPU/Metal diff {diff} >= atol {ATOL}"
    );
}

// ---------------------------------------------------------------------
// H2.3 — moe gather-combine (Wedge 2: MoE block, gather stage)
// ---------------------------------------------------------------------

#[test]
fn test_moe_gather_combine_matches_cpu() {
    let n_tokens = 16;
    let top_k = 6;
    let hidden = 256;

    let expert_out = fixed_input(n_tokens * top_k * hidden, 0xA8A8A8A8);
    let weights = fixed_input(n_tokens * top_k, 0xB9B9B9B9);

    let mut cpu_out = vec![0.0_f32; n_tokens * hidden];
    kernels::gather_combine(&expert_out, &weights, n_tokens, top_k, hidden, &mut cpu_out);

    let ctx = ctx().clone();
    let mut metal_out = vec![0.0_f32; n_tokens * hidden];
    kernels::moe_gather_combine_metal(
        &ctx,
        &expert_out,
        &weights,
        n_tokens,
        top_k,
        hidden,
        &mut metal_out,
    )
    .expect("moe_gather_combine_metal should succeed once H2.3 lands");

    let diff = max_abs_diff(&cpu_out, &metal_out);
    println!("[H2.3] moe_gather_combine parity max abs diff = {diff:.6}");
    assert!(
        diff < ATOL,
        "moe_gather_combine CPU/Metal diff {diff} >= atol {ATOL}"
    );
}

// ---------------------------------------------------------------------
// H2.2 — moe grouped GEMM with fused Q4_K_M dequant (Wedge 2: the moat)
// ---------------------------------------------------------------------

/// Construct synthetic Q4_K_M weight bytes for parity testing.
///
/// `d` and `dmin` are deliberately small (~1e-2) so the per-element
/// dequant values stay in a tight range. With 6-bit scales (max 63),
/// nibbles (max 15), and `d≈0.015`, max element magnitude is
/// `0.015 × 63 × 15 ≈ 14`, sum of 256 such terms is O(32). At that
/// magnitude the sequential-vs-tree reduction-order divergence is
/// well below the 1e-3 parity tolerance — without this clamp, large
/// random outputs cross atol from accumulation order alone.
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

// ---------------------------------------------------------------------
// H2.4 — gemm_q4_k_m_fused (Wedge 2: dense-path Q4_K_M GEMV)
// ---------------------------------------------------------------------

#[test]
fn test_gemm_q4_k_m_fused_matches_cpu() {
    use dismantle_core::gguf::GgmlType;
    use dismantle_core::quant::dequant_into;

    let rows = 64;
    let cols = 256;
    let blocks = rows * (cols / 256);

    // Different seeds from H2.2 so this test exercises distinct bytes.
    let w_bytes = synthetic_q4_k_bytes(blocks, 0xE6E6E6E6);
    let x = fixed_input(cols, 0xF7F7F7F7);

    let mut w_f32 = vec![0.0_f32; rows * cols];
    dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32)
        .expect("Q4_K dequant should succeed for valid synthetic bytes");
    let mut cpu_out = vec![0.0_f32; rows];
    kernels::gemv_f32(&w_f32, rows, cols, &x, &mut cpu_out);

    // Metal dense path: gemv_q4_k_m dispatches gemm_q4_k_m_fused.
    let ctx = ctx().clone();
    let mut metal_out = vec![0.0_f32; rows];
    kernels::gemv_q4_k_m(&ctx, &w_bytes, rows, cols, &x, &mut metal_out)
        .expect("gemv_q4_k_m should succeed once H2.4 lands");

    let diff = max_abs_diff(&cpu_out, &metal_out);
    println!("[H2.4] gemm_q4_k_m_fused parity max abs diff = {diff:.6}");
    assert!(
        diff < ATOL,
        "gemm_q4_k_m_fused CPU/Metal diff {diff} >= atol {ATOL}"
    );
}
