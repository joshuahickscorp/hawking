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

// ── mla_decode_kernel parity tests ─────────────────────────────────────────
//
// CPU reference mirrors the four-phase algorithm in `mla_decode_kernel`
// (shaders/attn.metal): w_uk^T @ q_nope, scores, softmax, c_kv_weighted,
// w_uv @ c_kv_weighted.
//
// kv_b_proj layout: (n_heads, qk_nope + v_head_dim, kv_lora_rank) row-major.

#[allow(clippy::too_many_arguments)]
fn mla_decode_cpu_reference(
    q: &[f32],
    c_kv: &[f32],
    k_pe: &[f32],
    kv_b_proj: &[f32],
    n_heads: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    seq_len: usize,
    scale: f32,
    out: &mut [f32],
) {
    let q_head_dim = qk_nope + qk_rope;
    let kv_b_per_head = (qk_nope + v_head_dim) * kv_lora_rank;

    let mut q_nope_proj = vec![0.0f32; kv_lora_rank];
    let mut scores = vec![0.0f32; seq_len];
    let mut c_kv_wt = vec![0.0f32; kv_lora_rank];

    for head in 0..n_heads {
        let q_nope = &q[head * q_head_dim..head * q_head_dim + qk_nope];
        let q_rope = &q[head * q_head_dim + qk_nope..(head + 1) * q_head_dim];

        let w_uk_base = head * kv_b_per_head;
        let w_uk = &kv_b_proj[w_uk_base..w_uk_base + qk_nope * kv_lora_rank];
        let w_uv_base = w_uk_base + qk_nope * kv_lora_rank;
        let w_uv = &kv_b_proj[w_uv_base..w_uv_base + v_head_dim * kv_lora_rank];

        // Phase 0: q_nope_proj[r] = Σ_i w_uk[i,r] * q_nope[i]
        for r in 0..kv_lora_rank {
            let mut acc = 0.0f32;
            for i in 0..qk_nope {
                acc += w_uk[i * kv_lora_rank + r] * q_nope[i];
            }
            q_nope_proj[r] = acc;
        }

        // Phase 1: scores[t] = (q_nope_proj · c_kv[t] + q_rope · k_pe[t]) * scale
        for t in 0..seq_len {
            let c_kv_t = &c_kv[t * kv_lora_rank..(t + 1) * kv_lora_rank];
            let k_pe_t = &k_pe[t * qk_rope..(t + 1) * qk_rope];
            let mut s = 0.0f32;
            for r in 0..kv_lora_rank {
                s += q_nope_proj[r] * c_kv_t[r];
            }
            for r in 0..qk_rope {
                s += q_rope[r] * k_pe_t[r];
            }
            scores[t] = s * scale;
        }

        // Phase 2: softmax
        let mx = scores[..seq_len]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for t in 0..seq_len {
            scores[t] = (scores[t] - mx).exp();
            sum += scores[t];
        }
        for t in 0..seq_len {
            scores[t] /= sum;
        }

        // Phase 3: c_kv_wt[r] = Σ_t scores[t] * c_kv[t,r]
        c_kv_wt.fill(0.0);
        for r in 0..kv_lora_rank {
            let mut acc = 0.0f32;
            for t in 0..seq_len {
                acc += scores[t] * c_kv[t * kv_lora_rank + r];
            }
            c_kv_wt[r] = acc;
        }

        // Phase 4: out[head,vi] = w_uv[vi,:] · c_kv_wt
        for vi in 0..v_head_dim {
            let w_uv_row = &w_uv[vi * kv_lora_rank..(vi + 1) * kv_lora_rank];
            let mut acc = 0.0f32;
            for r in 0..kv_lora_rank {
                acc += w_uv_row[r] * c_kv_wt[r];
            }
            out[head * v_head_dim + vi] = acc;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mla_decode_parity(
    name: &'static str,
    n_heads: usize,
    qk_nope: usize,
    qk_rope: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    seq_len: usize,
    seed: u64,
) {
    let q_head_dim = qk_nope + qk_rope;
    let scale = 1.0f32 / (q_head_dim as f32).sqrt();

    let q = fixed_input(n_heads * q_head_dim, seed);
    let c_kv = fixed_input(seq_len * kv_lora_rank, seed + 1);
    let k_pe = fixed_input(seq_len * qk_rope, seed + 2);
    let kv_b_proj = fixed_input(n_heads * (qk_nope + v_head_dim) * kv_lora_rank, seed + 3);

    let mut cpu_out = vec![0.0f32; n_heads * v_head_dim];
    mla_decode_cpu_reference(
        &q,
        &c_kv,
        &k_pe,
        &kv_b_proj,
        n_heads,
        qk_nope,
        qk_rope,
        v_head_dim,
        kv_lora_rank,
        seq_len,
        scale,
        &mut cpu_out,
    );

    let ctx = ctx();
    let kv_b_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&kv_b_proj));
    let mut metal_out = vec![0.0f32; n_heads * v_head_dim];
    dismantle_core::kernels::mla_decode_metal(
        ctx,
        &q,
        &c_kv,
        &k_pe,
        &kv_b_buf,
        n_heads,
        qk_nope,
        qk_rope,
        v_head_dim,
        kv_lora_rank,
        seq_len,
        scale,
        &mut metal_out,
    )
    .expect("mla_decode_metal should succeed");

    let diff = max_abs_diff(&cpu_out, &metal_out);
    println!("[W1] {name} parity max abs diff = {diff:.6}");
    assert!(diff < ATOL, "{name} CPU/Metal diff {diff} >= atol {ATOL}");
}

#[test]
fn test_mla_decode_smoke_matches_cpu() {
    mla_decode_parity(
        "mla_decode_smoke",
        /*n_heads=*/ 2,
        /*qk_nope=*/ 8,
        /*qk_rope=*/ 4,
        /*v_head_dim=*/ 8,
        /*kv_lora_rank=*/ 16,
        /*seq_len=*/ 4,
        0xDEAD_BEEF,
    );
}

#[test]
fn test_mla_decode_production_shape_matches_cpu() {
    // DeepSeek-V2-Lite shapes: n_heads=16, qk_nope=128, qk_rope=64,
    // v_head_dim=128, kv_lora_rank=512. Use a shorter seq_len to keep
    // the test fast; the kernel scales linearly in seq_len.
    mla_decode_parity(
        "mla_decode_production",
        /*n_heads=*/ 4,
        /*qk_nope=*/ 128,
        /*qk_rope=*/ 64,
        /*v_head_dim=*/ 128,
        /*kv_lora_rank=*/ 64,
        /*seq_len=*/ 8,
        0xCAFE_BABE,
    );
}
