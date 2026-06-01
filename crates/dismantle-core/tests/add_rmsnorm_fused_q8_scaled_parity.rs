//! Fusion gate for `add_rmsnorm_fused_q8_scaled` (AWQ Option B): the single-
//! dispatch kernel must produce bit-identical outputs to the back-to-back
//! pair `add_rmsnorm_fused` + `quantize_f32_to_int8_per_block_scaled`.
//!
//! Checks four buffers: x (post add), x_norm (f32, NOT divided by s), the
//! scaled int8 quant, and the scaled per-block scales.

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

fn make_smoothing(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128 ^ 0xA5BAEu128);
    (0..n)
        .map(|i| {
            if i % 20 == 0 {
                rng.gen_range(2.0..5.0)
            } else {
                rng.gen_range(0.3..1.6)
            }
        })
        .collect()
}

fn run_one(hidden: usize, seed: u64) {
    let ctx = ctx();
    let mut rng = Pcg64Mcg::new(seed as u128);

    let x: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let attn_out: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-0.5f32..0.5)).collect();
    let weight: Vec<f32> = (0..hidden).map(|_| rng.gen_range(0.5f32..1.5)).collect();
    let s = make_smoothing(hidden, seed);
    let eps = 1e-6f32;
    let blocks = hidden / 256;

    // Reference: unfused add_rmsnorm_fused then scaled per-block quantize.
    let ref_x_buf = new_f32_buf(ctx, &x);
    let ref_attn_buf = new_f32_buf(ctx, &attn_out);
    let weight_buf = new_f32_buf(ctx, &weight);
    let s_buf = new_f32_buf(ctx, &s);
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
        kernels::quantize_f32_to_int8_per_block_scaled_tcb(
            &mut tcb,
            &ref_xnorm_buf,
            &s_buf,
            &ref_int8_buf,
            &ref_scales_buf,
            hidden,
        )
        .expect("ref scaled quantize encode");
        tcb.commit_and_wait().expect("ref commit");
    }
    let ref_x = read_f32(&ref_x_buf, hidden);
    let ref_xnorm = read_f32(&ref_xnorm_buf, hidden);
    let ref_int8 = read_i8(&ref_int8_buf, hidden);
    let ref_scales = read_f32(&ref_scales_buf, blocks);

    // Fused-scaled path.
    let f_x_buf = new_f32_buf(ctx, &x);
    let f_attn_buf = new_f32_buf(ctx, &attn_out);
    let f_xnorm_buf = ctx.new_buffer(hidden * 4);
    let f_int8_buf = ctx.new_buffer(hidden);
    let f_scales_buf = ctx.new_buffer(blocks * 4);
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::add_rmsnorm_fused_q8_scaled_tcb(
            &mut tcb,
            &f_x_buf,
            &f_attn_buf,
            &weight_buf,
            &f_xnorm_buf,
            &f_int8_buf,
            &f_scales_buf,
            &s_buf,
            eps,
            hidden,
        )
        .expect("fused-scaled encode");
        tcb.commit_and_wait().expect("fused-scaled commit");
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
        "x_norm mismatch at hidden={hidden} seed={seed} \
         (Option B keeps x_norm unscaled — only the int8 sees s)"
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
                first_bad = Some((
                    i,
                    ref_int8[i],
                    f_int8[i],
                    f_xnorm[i],
                    s[i],
                    f_scales[i / 256],
                ));
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
fn add_rmsnorm_fused_q8_scaled_parity_hidden_256() {
    run_one(256, 0xA11CE);
}

#[test]
fn add_rmsnorm_fused_q8_scaled_parity_hidden_2048() {
    run_one(2048, 0xBEEF);
}

#[test]
fn add_rmsnorm_fused_q8_scaled_parity_hidden_2048_alt_seed() {
    run_one(2048, 0xC0FFEE);
}

// NOTE: this fused shader's phase-3 quantize uses one simdgroup per 256-block
// across the 8 simdgroups in a 256-thread TG, so it caps at hidden=2048 (8
// blocks). The existing un-scaled `add_rmsnorm_fused_q8_parity.rs` reflects
// the same constraint (no >2048 case). At runtime the only callers of the
// fused-scaled kernel are the two `hidden`-sized norm boundaries in
// `qwen_dense.rs`; the 11008-sized `ffn_act` quantize uses the standalone
// `quantize_f32_to_int8_per_block_scaled` kernel instead, which has no such
// cap and is covered by `quantize_int8_scaled_parity::*_11008`.
