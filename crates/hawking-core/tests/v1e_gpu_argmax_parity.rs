#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
mod common;
use common::*;
fn fixed_f16(n: usize, seed: u64) -> Vec<f16> {
    fixed_f32(n, seed)
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect()
}
fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
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
#[test]
fn wedge_e_argmax_tcb_matches_cpu() {
    let ctx = ctx();
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
#[test]
fn wedge_e_gemv_f16_buf_tcb_matches_cpu() {
    let ctx = ctx();
    let rows = 256usize;
    let cols = 128usize;
    let w_f16 = fixed_f16(rows * cols, 0xAAAA_1111);
    let x_f32 = fixed_f32(cols, 0xBBBB_2222);
    let mut cpu_out = vec![0.0f32; rows];
    kernels::gemv_f16(&w_f16, rows, cols, &x_f32, &mut cpu_out);
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
#[test]
fn wedge_e_lmhead_plus_argmax_tcb_matches_cpu() {
    let ctx = ctx();
    let vocab = 512usize;
    let hidden = 256usize;
    let lm_head_f16 = fixed_f16(vocab * hidden, 0xCCCC_3333);
    let x_norm_f32 = fixed_f32(hidden, 0xDDDD_4444);
    let mut cpu_logits = vec![0.0f32; vocab];
    kernels::gemv_f16(&lm_head_f16, vocab, hidden, &x_norm_f32, &mut cpu_logits);
    let cpu_token = cpu_argmax(&cpu_logits);
    let lm_head_buf = new_f16_buf(ctx, &lm_head_f16);
    let x_norm_buf = new_f32_buf(ctx, &x_norm_f32);
    let logits_buf = ctx.new_buffer(vocab * std::mem::size_of::<f32>());
    let token_buf = ctx.new_buffer(std::mem::size_of::<u32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_f16_metal_buf_tcb(
        &mut tcb,
        &lm_head_buf,
        vocab,
        hidden,
        &x_norm_buf,
        &logits_buf,
    )
    .expect("gemv_f16_metal_buf_tcb");
    kernels::sample_argmax_f32_tcb(&mut tcb, &logits_buf, &token_buf, vocab)
        .expect("sample_argmax_f32_tcb");
    tcb.commit_and_wait().expect("commit");
    let gpu_token = unsafe { *(token_buf.contents() as *const u32) };
    assert_eq!(
        gpu_token, cpu_token,
        "lmhead+argmax: gpu={gpu_token} cpu={cpu_token}"
    );
    let gpu_logits = read_f32_buf(&logits_buf, vocab);
    let diff = max_abs_diff(&cpu_logits, &gpu_logits);
    assert!(diff < 1e-3, "logits max_abs_diff={diff:.2e} > 1e-3");
}
