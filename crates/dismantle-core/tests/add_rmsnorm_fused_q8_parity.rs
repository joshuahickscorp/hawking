//! Fusion gate for `add_rmsnorm_fused_q8`: the single-dispatch kernel must
//! produce bit-identical outputs to the back-to-back pair
//! `add_rmsnorm_fused` + `quantize_f32_to_int8_per_block`. This is the
//! non-negotiable correctness invariant — the fusion is purely a dispatch-
//! reorganization, not a numerical approximation.
//!
//! Checks all four output buffers: x (post add), x_norm (f32), x_norm_int8,
//! and x_norm_scales.

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{PinnedBuffer, TokenCommandBuffer};
use rand::Rng;
use rand_pcg::Pcg64Mcg;

mod common;
use common::*;

fn read_f32(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn read_i8(buf: &PinnedBuffer, n: usize) -> Vec<i8> {
    let ptr = buf.contents() as *const i8;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn run_one(hidden: usize, seed: u64) {
    let ctx = ctx();
    let mut rng = Pcg64Mcg::new(seed as u128);

    // Inputs.
    let x: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let attn_out: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-0.5f32..0.5)).collect();
    let weight: Vec<f32> = (0..hidden).map(|_| rng.gen_range(0.5f32..1.5)).collect();
    let eps = 1e-6f32;
    let blocks = hidden / 256;

    // Reference path: separate add_rmsnorm_fused + quantize.
    let ref_x_buf = new_f32_buf(ctx, &x);
    let ref_attn_buf = new_f32_buf(ctx, &attn_out);
    let weight_buf = new_f32_buf(ctx, &weight);
    let ref_xnorm_buf = ctx.new_buffer(hidden * 4);
    let ref_int8_buf = ctx.new_buffer(hidden);
    let ref_scales_buf = ctx.new_buffer(blocks * 4);
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::add_rmsnorm_fused_tcb(
            &mut tcb,
            &ref_x_buf,
            &ref_attn_buf,
            &weight_buf,
            &ref_xnorm_buf,
            eps,
            hidden,
        )
        .expect("ref add_rmsnorm_fused encode");
        kernels::quantize_f32_to_int8_per_block_tcb(
            &mut tcb,
            &ref_xnorm_buf,
            &ref_int8_buf,
            &ref_scales_buf,
            hidden,
        )
        .expect("ref quantize encode");
        tcb.commit_and_wait().expect("ref commit");
    }
    let ref_x = read_f32(&ref_x_buf, hidden);
    let ref_xnorm = read_f32(&ref_xnorm_buf, hidden);
    let ref_int8 = read_i8(&ref_int8_buf, hidden);
    let ref_scales = read_f32(&ref_scales_buf, blocks);

    // Fused path.
    let f_x_buf = new_f32_buf(ctx, &x);
    let f_attn_buf = new_f32_buf(ctx, &attn_out);
    let f_xnorm_buf = ctx.new_buffer(hidden * 4);
    let f_int8_buf = ctx.new_buffer(hidden);
    let f_scales_buf = ctx.new_buffer(blocks * 4);
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::add_rmsnorm_fused_q8_tcb(
            &mut tcb,
            &f_x_buf,
            &f_attn_buf,
            &weight_buf,
            &f_xnorm_buf,
            &f_int8_buf,
            &f_scales_buf,
            eps,
            hidden,
        )
        .expect("fused encode");
        tcb.commit_and_wait().expect("fused commit");
    }
    let f_x = read_f32(&f_x_buf, hidden);
    let f_xnorm = read_f32(&f_xnorm_buf, hidden);
    let f_int8 = read_i8(&f_int8_buf, hidden);
    let f_scales = read_f32(&f_scales_buf, blocks);

    assert_eq!(
        ref_x, f_x,
        "x (post-add) mismatch at hidden={hidden} seed={seed}"
    );
    assert_eq!(
        ref_xnorm, f_xnorm,
        "x_norm mismatch at hidden={hidden} seed={seed}"
    );
    assert_eq!(
        ref_scales, f_scales,
        "x_norm_scales mismatch at hidden={hidden} seed={seed}"
    );
    let mut diffs = 0usize;
    let mut first_bad = None;
    for i in 0..hidden {
        if ref_int8[i] != f_int8[i] {
            diffs += 1;
            if first_bad.is_none() {
                first_bad = Some((i, ref_int8[i], f_int8[i], f_xnorm[i], f_scales[i / 256]));
            }
        }
    }
    assert_eq!(
        diffs, 0,
        "x_norm_int8 mismatch at hidden={hidden} seed={seed}: {diffs} elems; first {:?}",
        first_bad
    );
}

#[test]
fn add_rmsnorm_fused_q8_parity_hidden_256() {
    run_one(256, 0xA11CE);
}

#[test]
fn add_rmsnorm_fused_q8_parity_hidden_2048() {
    // Qwen-3B hidden.
    run_one(2048, 0xBEEF);
}

#[test]
fn add_rmsnorm_fused_q8_parity_hidden_2048_alt_seed() {
    run_one(2048, 0xC0FFEE);
}
