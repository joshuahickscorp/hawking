#![cfg(target_os = "macos")]
use hawking_core::kernels;
mod common;
use common::*;
fn attn_pair_check(rows_a: usize, rows_b: usize, cols: usize) {
    let x = fixed_f32(cols, 0xA1B2C3D4);
    let w_a = fixed_f32(rows_a * cols, 0xDEAD_BEEF);
    let w_b = fixed_f32(rows_b * cols, 0xCAFE_F00D);
    let ctx = ctx().clone();
    let mut ref_a = vec![0.0_f32; rows_a];
    let mut ref_b = vec![0.0_f32; rows_b];
    kernels::gemv_f32_attn_metal(&ctx, &w_a, rows_a, cols, &x, &mut ref_a)
        .expect("gemv_f32_attn_metal A");
    kernels::gemv_f32_attn_metal(&ctx, &w_b, rows_b, cols, &x, &mut ref_b)
        .expect("gemv_f32_attn_metal B");
    let w_a_bytes: &[u8] = bytemuck::cast_slice(&w_a);
    let w_b_bytes: &[u8] = bytemuck::cast_slice(&w_b);
    let w_a_buf = ctx.new_buffer_with_bytes(w_a_bytes);
    let w_b_buf = ctx.new_buffer_with_bytes(w_b_bytes);
    let mut out_a = vec![0.0_f32; rows_a];
    let mut out_b = vec![0.0_f32; rows_b];
    kernels::dispatch_gemv_f32_attn_pinned_pair_batched(
        &ctx, &w_a_buf, rows_a, &w_b_buf, rows_b, cols, &x, &mut out_a, &mut out_b,
    )
    .expect("dispatch_gemv_f32_attn_pinned_pair_batched");
    let max_diff_a = ref_a
        .iter()
        .zip(out_a.iter())
        .map(|(r, g)| (r - g).abs())
        .fold(0.0_f32, f32::max);
    let max_diff_b = ref_b
        .iter()
        .zip(out_b.iter())
        .map(|(r, g)| (r - g).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff_a <= ATOL,
        "shape ({rows_a},{rows_b},{cols}) out_a diff={max_diff_a:e} > ATOL={ATOL:e}"
    );
    assert!(
        max_diff_b <= ATOL,
        "shape ({rows_a},{rows_b},{cols}) out_b diff={max_diff_b:e} > ATOL={ATOL:e}"
    );
}
#[test]
fn test_attn_pair_q_lora_proxy() {
    attn_pair_check(512, 256, 2048);
}
#[test]
fn test_attn_pair_non_q_lora_proxy() {
    attn_pair_check(2048, 256, 2048);
}
