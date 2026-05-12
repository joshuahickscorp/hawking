//! Phase X parity: gemv_f16_simdmat (simdgroup_matrix LM-head) matches CPU reference.
//!
//! Tests:
//!   1. `phase_x_simdmat_matches_cpu_basic` — small shapes; atol 1e-3 (fp16 quant noise).
//!   2. `phase_x_simdmat_matches_cpu_lm_head_shape` — (rows=512, cols=2048) LM-head analogue.
//!   3. `phase_x_simdmat_argmax_matches_cpu` — token id (argmax) from simdmat matches CPU.
#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use half::f16;
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

fn fixed_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

fn fixed_f16(n: usize, seed: u64) -> Vec<f16> {
    fixed_f32(n, seed).iter().map(|&v| f16::from_f32(v)).collect()
}

fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

fn read_f32_buf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn cpu_gemv_f16(w: &[f16], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    kernels::gemv_f16(w, rows, cols, x, &mut out);
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn cpu_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

// ─────────────────────────────────────────────────────────────────────────────

/// Basic shapes — output matches CPU gemv_f16 at atol 1e-3 (fp16 quant noise).
#[test]
fn phase_x_simdmat_matches_cpu_basic() {
    let ctx = ctx();

    for &(rows, cols) in &[(8usize, 8usize), (16, 8), (8, 16), (32, 64), (64, 128)] {
        let w = fixed_f16(rows * cols, 0xA1B2_C3D4 ^ rows as u64);
        let x = fixed_f32(cols, 0xE5F6_0718 ^ cols as u64);
        let cpu = cpu_gemv_f16(&w, rows, cols, &x);

        let w_buf = new_f16_buf(ctx, &w);
        let x_buf = new_f32_buf(ctx, &x);
        let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_f16_simdmat_tcb(&mut tcb, &w_buf, rows, cols, &x_buf, &y_buf)
            .unwrap_or_else(|e| panic!("gemv_f16_simdmat_tcb rows={rows} cols={cols}: {e}"));
        tcb.commit_and_wait().expect("commit");

        let gpu = read_f32_buf(&y_buf, rows);
        let diff = max_abs_diff(&cpu, &gpu);
        assert!(
            diff < 1e-3,
            "rows={rows} cols={cols}: max_abs_diff={diff:.2e} > 1e-3"
        );
    }
}

/// LM-head analogue shape: rows=512, cols=2048.
/// Validates the actual hidden_dim used by DeepSeek-V2-Lite (cols=2048 % 8 == 0 ✓).
#[test]
fn phase_x_simdmat_matches_cpu_lm_head_shape() {
    let ctx = ctx();
    let rows = 512usize;
    let cols = 2048usize;

    let w = fixed_f16(rows * cols, 0xBEEF_1234);
    let x = fixed_f32(cols, 0xDEAD_5678);
    let cpu = cpu_gemv_f16(&w, rows, cols, &x);

    let w_buf = new_f16_buf(ctx, &w);
    let x_buf = new_f32_buf(ctx, &x);
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_f16_simdmat_tcb(&mut tcb, &w_buf, rows, cols, &x_buf, &y_buf)
        .expect("gemv_f16_simdmat_tcb");
    tcb.commit_and_wait().expect("commit");

    let gpu = read_f32_buf(&y_buf, rows);
    let diff = max_abs_diff(&cpu, &gpu);
    assert!(
        diff < 1e-3,
        "lm_head_shape rows={rows} cols={cols}: max_abs_diff={diff:.2e} > 1e-3"
    );
}

/// Token parity (temp=0 greedy): argmax of simdmat output matches CPU.
/// This is the exact gate used by forward_token_greedy.
#[test]
fn phase_x_simdmat_argmax_matches_cpu() {
    let ctx = ctx();
    let rows = 256usize;
    let cols = 128usize;

    let w = fixed_f16(rows * cols, 0xCAFE_BABE);
    let x = fixed_f32(cols, 0xF00D_FEED);
    let cpu_logits = cpu_gemv_f16(&w, rows, cols, &x);
    let cpu_token = cpu_argmax(&cpu_logits);

    let w_buf = new_f16_buf(ctx, &w);
    let x_buf = new_f32_buf(ctx, &x);
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_f16_simdmat_tcb(&mut tcb, &w_buf, rows, cols, &x_buf, &y_buf)
        .expect("gemv_f16_simdmat_tcb");
    tcb.commit_and_wait().expect("commit");

    let gpu_logits = read_f32_buf(&y_buf, rows);
    let gpu_token = cpu_argmax(&gpu_logits);
    assert_eq!(
        gpu_token, cpu_token,
        "argmax: gpu={gpu_token} cpu={cpu_token}"
    );

    let diff = max_abs_diff(&cpu_logits, &gpu_logits);
    assert!(diff < 1e-3, "logits diff={diff:.2e} > 1e-3");
}
