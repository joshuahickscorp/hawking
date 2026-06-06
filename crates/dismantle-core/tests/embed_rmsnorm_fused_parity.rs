#![cfg(target_os = "macos")]
//! Track B7 parity: `embed_lookup_rmsnorm_f32` must produce bit-identical
//! results to the two-dispatch sequence:
//!   1. `embed_lookup_metal_f32_tcb` (embed[token] → x)
//!   2. `rmsnorm_metal_buf_tcb`       (x → x_norm)
//!
//! Verifies both `x` (written in-place by the embed lookup) and `x_norm`
//! (the normalized output). Tests several (hidden, token) pairs including
//! the Qwen-3B production shape (hidden=2048).

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, TokenCommandBuffer};

mod common;
use common::*;

fn make_embed(vocab: usize, hidden: usize, seed: u32) -> Vec<u16> {
    // Build a fp16 embedding table (vocab × hidden).
    (0..vocab * hidden)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761u32).wrapping_add(seed);
            // Map to [-1, 1] and convert to fp16 bits.
            let f = (x as f32 / u32::MAX as f32) * 2.0 - 1.0;
            half::f16::from_f32(f).to_bits()
        })
        .collect()
}

fn make_weight(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(1_664_525u32).wrapping_add(seed);
            0.5 + (x as f32 / u32::MAX as f32) // positive weights in [0.5, 1.5]
        })
        .collect()
}

/// Reference: embed_lookup_metal_f32_tcb + rmsnorm_metal_buf_tcb (2 dispatches).
fn run_ref(
    ctx: &MetalContext,
    embed_f16: &[u16],
    weight: &[f32],
    token: u32,
    hidden: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let embed_bytes: Vec<u8> = embed_f16.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let embed_buf = ctx.new_buffer_with_bytes(&embed_bytes);
    let weight_buf = new_f32_buf(ctx, weight);
    let x_buf = ctx.new_buffer(hidden * 4);
    let x_norm_buf = ctx.new_buffer(hidden * 4);

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::embed_lookup_metal_f32_tcb(&mut tcb, &embed_buf, token, hidden, &x_buf)
        .expect("embed_lookup");
    kernels::rmsnorm_metal_buf_tcb(&mut tcb, &x_buf, &weight_buf, eps, hidden, &x_norm_buf)
        .expect("rmsnorm");
    tcb.commit_and_wait().expect("ref commit");

    (read_f32_buf(&x_buf, hidden), read_f32_buf(&x_norm_buf, hidden))
}

/// Fused: embed_lookup_rmsnorm_f32_tcb (1 dispatch).
fn run_fused(
    ctx: &MetalContext,
    embed_f16: &[u16],
    weight: &[f32],
    token: u32,
    hidden: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let embed_bytes: Vec<u8> = embed_f16.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let embed_buf = ctx.new_buffer_with_bytes(&embed_bytes);
    let weight_buf = new_f32_buf(ctx, weight);
    let x_buf = ctx.new_buffer(hidden * 4);
    let x_norm_buf = ctx.new_buffer(hidden * 4);

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::embed_lookup_rmsnorm_f32_tcb(
        &mut tcb, &embed_buf, &weight_buf, token, hidden, eps, &x_buf, &x_norm_buf,
    )
    .expect("fused dispatch");
    tcb.commit_and_wait().expect("fused commit");

    (read_f32_buf(&x_buf, hidden), read_f32_buf(&x_norm_buf, hidden))
}

#[test]
fn embed_lookup_rmsnorm_fused_matches_two_dispatch() {
    let ctx = ctx();
    let eps = 1e-6_f32;
    let vocab = 256; // small vocab for fast test; only token offset matters

    // (hidden, token, seed) — Qwen-3B uses hidden=2048. Also test boundary shapes.
    let cases: &[(usize, u32, u32)] = &[
        (256,  0, 0xE001),  // minimal hidden = tg_size
        (512,  3, 0xE002),
        (1024, 7, 0xE003),
        (2048, 0, 0xE004),  // Qwen-3B production shape
        (2048, 5, 0xE005),  // different token
        (4096, 1, 0xE006),  // max supported hidden = tg_size * 16
    ];

    for &(hidden, token, seed) in cases {
        let embed = make_embed(vocab, hidden, seed);
        let weight = make_weight(hidden, seed ^ 0xFF);

        let (ref_x, ref_norm) = run_ref(ctx, &embed, &weight, token, hidden, eps);
        let (got_x, got_norm) = run_fused(ctx, &embed, &weight, token, hidden, eps);

        let dx = max_abs_diff(&ref_x, &got_x);
        let dn = max_abs_diff(&ref_norm, &got_norm);
        assert_eq!(dx, 0.0, "hidden={hidden} token={token}: x max_diff={dx:.2e}");
        assert_eq!(dn, 0.0, "hidden={hidden} token={token}: x_norm max_diff={dn:.2e}");
        eprintln!("B7 hidden={hidden} token={token}: x={dx:.0e} norm={dn:.0e} OK");
    }
}
