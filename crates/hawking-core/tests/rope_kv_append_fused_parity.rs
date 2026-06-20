#![cfg(target_os = "macos")]
//! Track B6 parity: `rope_qk_kv_append_vbias_f32` must produce bit-identical
//! results to the two-dispatch sequence:
//!   1. `rope_qk_f32_b1_bias_tcb`   (Q+K bias + rope, in-place)
//!   2. `kv_append_vbias_f32_tcb`   (V-bias + K+V cache append)
//!
//! The fused kernel differs in one intentional way: k_token_buf is left in its
//! pre-rope state (k is rotated directly into k_cache). This is correct since
//! nothing reads k_token_buf after the KV append. The parity check verifies
//! q_buf (in-place rope), k_cache[kv_off..], and v_cache[kv_off..].

use hawking_core::kernels;
use hawking_core::metal::{MetalContext, TokenCommandBuffer};

mod common;
use common::*;

fn rnd(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761u32).wrapping_add(seed);
            (x as f32 / u32::MAX as f32) * 4.0 - 2.0
        })
        .collect()
}

/// Two-dispatch reference: rope_qk_f32_b1_bias + kv_append_vbias_f32.
/// Returns (q_out, k_cache_slice, v_cache_slice) after commit.
#[allow(clippy::too_many_arguments)]
fn run_ref(
    ctx: &MetalContext,
    q: &[f32],
    k_tok: &[f32],
    v_tok: &[f32],
    q_bias: Option<&[f32]>,
    k_bias: Option<&[f32]>,
    v_bias: Option<&[f32]>,
    kv_off: usize,
    cache_size: usize, // total elements in each k_cache / v_cache buffer
    n_q: usize,
    n_k: usize,
    head_dim: usize,
    pos: u32,
    base: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let kv_dim = k_tok.len();
    let q_buf = new_f32_buf(ctx, q);
    let k_buf = new_f32_buf(ctx, k_tok);
    let v_buf = new_f32_buf(ctx, v_tok);
    let k_cache = ctx.new_buffer(cache_size * 4);
    let v_cache = ctx.new_buffer(cache_size * 4);

    let qb_buf = q_bias.map(|b| new_f32_buf(ctx, b));
    let kb_buf = k_bias.map(|b| new_f32_buf(ctx, b));
    let vb_buf = v_bias.map(|b| new_f32_buf(ctx, b));

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::rope_qk_f32_b1_bias_tcb(
        &mut tcb,
        &q_buf,
        &k_buf,
        qb_buf.as_ref(),
        kb_buf.as_ref(),
        n_q,
        n_k,
        head_dim,
        pos,
        base,
    )
    .expect("ref rope_qk");
    kernels::kv_append_vbias_f32_tcb(
        &mut tcb,
        &k_buf,
        &v_buf,
        vb_buf.as_ref(),
        &k_cache,
        &v_cache,
        kv_dim,
        kv_off,
    )
    .expect("ref kv_append");
    tcb.commit_and_wait().expect("ref commit");

    (
        read_f32_buf(&q_buf, q.len()),
        read_f32_buf(&k_cache, cache_size),
        read_f32_buf(&v_cache, cache_size),
    )
}

/// One-dispatch fused: rope_qk_kv_append_vbias_f32.
/// Returns (q_out, k_cache_slice, v_cache_slice).
#[allow(clippy::too_many_arguments)]
fn run_fused(
    ctx: &MetalContext,
    q: &[f32],
    k_tok: &[f32],
    v_tok: &[f32],
    q_bias: Option<&[f32]>,
    k_bias: Option<&[f32]>,
    v_bias: Option<&[f32]>,
    kv_off: usize,
    cache_size: usize,
    n_q: usize,
    n_k: usize,
    head_dim: usize,
    pos: u32,
    base: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let kv_dim = k_tok.len();
    let q_buf = new_f32_buf(ctx, q);
    let k_buf = new_f32_buf(ctx, k_tok);
    let v_buf = new_f32_buf(ctx, v_tok);
    let k_cache = ctx.new_buffer(cache_size * 4);
    let v_cache = ctx.new_buffer(cache_size * 4);

    let qb_buf = q_bias.map(|b| new_f32_buf(ctx, b));
    let kb_buf = k_bias.map(|b| new_f32_buf(ctx, b));
    let vb_buf = v_bias.map(|b| new_f32_buf(ctx, b));

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::rope_qk_kv_append_vbias_f32_tcb(
        &mut tcb,
        &q_buf,
        &k_buf,
        &v_buf,
        qb_buf.as_ref(),
        kb_buf.as_ref(),
        vb_buf.as_ref(),
        &k_cache,
        &v_cache,
        n_q,
        n_k,
        head_dim,
        pos,
        base,
        kv_dim,
        kv_off,
    )
    .expect("fused dispatch");
    tcb.commit_and_wait().expect("fused commit");

    (
        read_f32_buf(&q_buf, q.len()),
        read_f32_buf(&k_cache, cache_size),
        read_f32_buf(&v_cache, cache_size),
    )
}

#[test]
fn rope_qk_kv_append_vbias_fused_matches_two_dispatch() {
    let ctx = ctx();
    let base = 10_000.0f32;

    // (n_q_heads, n_k_heads, head_dim, pos, kv_off_heads, with_biases)
    // Qwen-3B shape: n_q=16, n_k=8, hd=128; also test GQA 4:1 and 1:1.
    let cases: &[(usize, usize, usize, u32, usize, bool)] = &[
        (16, 8, 128, 1, 0, false),  // Qwen-3B shape, pos=1, no bias, append at 0
        (16, 8, 128, 5, 8, true),   // with biases, kv_off=8 kv_heads
        (4, 4, 64, 0, 0, true),     // GQA 1:1, pos=0
        (8, 2, 64, 10, 3, false),   // GQA 4:1 variant
        (1, 1, 128, 100, 50, true), // single-head sanity
    ];

    for &(n_q, n_k, head_dim, pos, kv_off_heads, with_biases) in cases {
        let q_dim = n_q * head_dim;
        let kv_dim = n_k * head_dim;
        let kv_off = kv_off_heads * head_dim; // element offset into cache
        let cache_size = kv_off + kv_dim + 32; // extra padding to detect OOB writes

        let seed = (n_q + n_k + head_dim + pos as usize) as u32;
        let q = rnd(q_dim, seed);
        let k_tok = rnd(kv_dim, seed ^ 0x1000);
        let v_tok = rnd(kv_dim, seed ^ 0x2000);
        let q_bias_data = rnd(q_dim, seed ^ 0x3000);
        let k_bias_data = rnd(kv_dim, seed ^ 0x4000);
        let v_bias_data = rnd(kv_dim, seed ^ 0x5000);
        let q_bias = if with_biases {
            Some(q_bias_data.as_slice())
        } else {
            None
        };
        let k_bias = if with_biases {
            Some(k_bias_data.as_slice())
        } else {
            None
        };
        let v_bias = if with_biases {
            Some(v_bias_data.as_slice())
        } else {
            None
        };

        let (ref_q, ref_kc, ref_vc) = run_ref(
            ctx, &q, &k_tok, &v_tok, q_bias, k_bias, v_bias, kv_off, cache_size, n_q, n_k,
            head_dim, pos, base,
        );
        let (fused_q, fused_kc, fused_vc) = run_fused(
            ctx, &q, &k_tok, &v_tok, q_bias, k_bias, v_bias, kv_off, cache_size, n_q, n_k,
            head_dim, pos, base,
        );

        let dq = max_abs_diff(&ref_q, &fused_q);
        let dkc = max_abs_diff(&ref_kc, &fused_kc);
        let dvc = max_abs_diff(&ref_vc, &fused_vc);

        assert_eq!(
            dq, 0.0,
            "n_q={n_q} n_k={n_k} hd={head_dim} pos={pos}: q max_diff={dq:.2e}"
        );
        assert_eq!(
            dkc, 0.0,
            "n_q={n_q} n_k={n_k} hd={head_dim} pos={pos}: k_cache max_diff={dkc:.2e}"
        );
        assert_eq!(
            dvc, 0.0,
            "n_q={n_q} n_k={n_k} hd={head_dim} pos={pos}: v_cache max_diff={dvc:.2e}"
        );
        eprintln!(
            "B6 n_q={n_q} n_k={n_k} hd={head_dim} pos={pos} bias={with_biases}: q={dq:.0e} kc={dkc:.0e} vc={dvc:.0e} OK"
        );
    }
}
