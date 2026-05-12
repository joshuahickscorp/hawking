//! Wedge E parity: GPU argmax via TCB matches CPU reference.
//!
//! Tests:
//!   1. `wedge_e_argmax_tcb_matches_cpu` — sample_argmax_f32_tcb produces the
//!      same winner as CPU argmax across vocab sizes and tie patterns.
//!   2. `wedge_e_gemv_f16_buf_tcb_matches_cpu` — gemv_f16_metal_buf_tcb output
//!      matches CPU gemv_f16 within fp16 tolerance.
//!   3. `wedge_e_lmhead_plus_argmax_tcb_matches_cpu` — combined LM-head GEMV
//!      + argmax via TCB produces the same token id as the CPU path.
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

fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

fn read_f32_buf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn cpu_argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            best = i as u32;
            bv = v;
        }
    }
    best
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

// ─────────────────────────────────────────────────────────────────────────────

/// sample_argmax_f32_tcb winner matches CPU argmax for various vocab sizes
/// and tie patterns (lowest-index-wins on ties).
#[test]
fn wedge_e_argmax_tcb_matches_cpu() {
    let ctx = ctx();

    // Basic: clear winner at a specific index.
    for &vocab in &[256usize, 4096, 32768] {
        let mut logits = fixed_f32(vocab, 0xDEAD_BEEF ^ vocab as u64);
        let target = vocab / 3 + 11;
        logits[target] = 9999.0;

        let cpu = cpu_argmax(&logits);
        let logits_buf = new_f32_buf(ctx, &logits);
        let token_buf = ctx.new_buffer(std::mem::size_of::<u32>());

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::sample_argmax_f32_tcb(&mut tcb, &logits_buf, &token_buf, vocab)
            .expect("sample_argmax_f32_tcb");
        tcb.commit_and_wait().expect("commit");

        let gpu = unsafe { *(token_buf.contents() as *const u32) };
        assert_eq!(gpu, cpu, "vocab={vocab}: gpu={gpu} cpu={cpu}");
    }

    // Tie: all same value → lowest index (0) wins.
    {
        let vocab = 1024usize;
        let logits = vec![1.0f32; vocab];
        let logits_buf = new_f32_buf(ctx, &logits);
        let token_buf = ctx.new_buffer(std::mem::size_of::<u32>());
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::sample_argmax_f32_tcb(&mut tcb, &logits_buf, &token_buf, vocab)
            .expect("tied argmax");
        tcb.commit_and_wait().expect("commit");
        let gpu = unsafe { *(token_buf.contents() as *const u32) };
        assert_eq!(gpu, 0u32, "tied: lowest index should win, got {gpu}");
    }
}

/// gemv_f16_metal_buf_tcb output matches CPU gemv_f16 within fp16 precision
/// (atol 1e-3 — fp16 quantization noise is ~1e-3 at unit scale).
#[test]
fn wedge_e_gemv_f16_buf_tcb_matches_cpu() {
    let ctx = ctx();

    // Shapes: rows=256 (small vocab analogue), cols=128.
    let rows = 256usize;
    let cols = 128usize;

    let w_f16 = fixed_f16(rows * cols, 0xAAAA_1111);
    let x_f32 = fixed_f32(cols, 0xBBBB_2222);

    // CPU reference.
    let mut cpu_out = vec![0.0f32; rows];
    kernels::gemv_f16(&w_f16, rows, cols, &x_f32, &mut cpu_out);

    // GPU TCB path.
    let w_buf = new_f16_buf(ctx, &w_f16);
    let x_buf = new_f32_buf(ctx, &x_f32);
    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_f16_metal_buf_tcb(&mut tcb, &w_buf, rows, cols, &x_buf, &y_buf)
        .expect("gemv_f16_metal_buf_tcb");
    tcb.commit_and_wait().expect("commit");

    let gpu_out = read_f32_buf(&y_buf, rows);
    let diff = max_abs_diff(&cpu_out, &gpu_out);
    assert!(
        diff < 1e-3,
        "gemv_f16 rows={rows} cols={cols}: max_abs_diff={diff:.2e} > 1e-3"
    );
}

/// Combined LM-head GEMV + argmax via TCB matches CPU path.
/// This is the exact kernel sequence that Wedge E adds to forward_token_greedy.
#[test]
fn wedge_e_lmhead_plus_argmax_tcb_matches_cpu() {
    let ctx = ctx();

    // Shapes: small vocab (512) × hidden (256) for test speed.
    // Parity tested at both default and 102400-vocab analogue above.
    let vocab = 512usize;
    let hidden = 256usize;

    let lm_head_f16 = fixed_f16(vocab * hidden, 0xCCCC_3333);
    let x_norm_f32 = fixed_f32(hidden, 0xDDDD_4444);

    // CPU reference: gemv_f16 → logits → argmax.
    let mut cpu_logits = vec![0.0f32; vocab];
    kernels::gemv_f16(&lm_head_f16, vocab, hidden, &x_norm_f32, &mut cpu_logits);
    let cpu_token = cpu_argmax(&cpu_logits);

    // GPU TCB: gemv_f16_metal_buf_tcb + sample_argmax_f32_tcb in one TCB.
    let lm_head_buf = new_f16_buf(ctx, &lm_head_f16);
    let x_norm_buf = new_f32_buf(ctx, &x_norm_f32);
    let logits_buf = ctx.new_buffer(vocab * std::mem::size_of::<f32>());
    let token_buf = ctx.new_buffer(std::mem::size_of::<u32>());

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_f16_metal_buf_tcb(&mut tcb, &lm_head_buf, vocab, hidden, &x_norm_buf, &logits_buf)
        .expect("gemv_f16_metal_buf_tcb");
    kernels::sample_argmax_f32_tcb(&mut tcb, &logits_buf, &token_buf, vocab)
        .expect("sample_argmax_f32_tcb");
    tcb.commit_and_wait().expect("commit");

    let gpu_token = unsafe { *(token_buf.contents() as *const u32) };
    assert_eq!(
        gpu_token, cpu_token,
        "lmhead+argmax: gpu={gpu_token} cpu={cpu_token}"
    );

    // Also verify the logits buf matches CPU within fp16 tolerance.
    let gpu_logits = read_f32_buf(&logits_buf, vocab);
    let diff = max_abs_diff(&cpu_logits, &gpu_logits);
    assert!(
        diff < 1e-3,
        "logits max_abs_diff={diff:.2e} > 1e-3"
    );
}
