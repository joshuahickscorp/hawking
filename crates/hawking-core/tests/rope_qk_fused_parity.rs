#![cfg(target_os = "macos")]
//! Track 3.4 parity: `rope_qk_f32_batched_multiseq` (fused Q+K) must be
//! BIT-IDENTICAL to two separate `rope_f32_batched_multiseq` calls.
//!
//! Saves 1 dispatch/layer × 28 layers = 28 dispatches on Qwen-3B.

use hawking_core::kernels;

mod common;
use common::*;

/// Run the fused Q+K kernel and return (q_out, k_out).
fn run_fused(
    q: &[f32],
    k: &[f32],
    n_q_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    positions: &[u32],
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let ctx = ctx();
    let b = positions.len();
    let q_dim = n_q_heads * head_dim;
    let kv_dim = n_k_heads * head_dim;
    let q_buf = new_f32_buf(ctx, q);
    let k_buf = new_f32_buf(ctx, k);
    let pos_bytes: Vec<u8> = positions.iter().flat_map(|&p| p.to_le_bytes()).collect();
    let pos_buf = ctx.new_buffer_with_bytes(&pos_bytes);
    let mut tcb = hawking_core::metal::TokenCommandBuffer::new(ctx);
    kernels::rope_qk_f32_batched_multiseq_tcb(
        &mut tcb, &q_buf, &k_buf, &pos_buf, n_q_heads, n_k_heads, head_dim, q_dim, kv_dim, b, theta,
    )
    .expect("rope_qk fused");
    tcb.commit_and_wait().expect("commit");
    let q_out = read_f32_buf(&q_buf, b * q_dim);
    let k_out = read_f32_buf(&k_buf, b * kv_dim);
    (q_out, k_out)
}

/// Run two separate rope calls and return (q_out, k_out).
fn run_separate(
    q: &[f32],
    k: &[f32],
    n_q_heads: usize,
    n_k_heads: usize,
    head_dim: usize,
    positions: &[u32],
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let ctx = ctx();
    let b = positions.len();
    let q_dim = n_q_heads * head_dim;
    let kv_dim = n_k_heads * head_dim;
    let q_buf = new_f32_buf(ctx, q);
    let k_buf = new_f32_buf(ctx, k);
    let pos_bytes: Vec<u8> = positions.iter().flat_map(|&p| p.to_le_bytes()).collect();
    let pos_buf = ctx.new_buffer_with_bytes(&pos_bytes);
    let mut tcb = hawking_core::metal::TokenCommandBuffer::new(ctx);
    kernels::rope_f32_batched_multiseq_tcb(
        &mut tcb, &q_buf, &pos_buf, n_q_heads, head_dim, q_dim, b, theta,
    )
    .expect("rope Q");
    kernels::rope_f32_batched_multiseq_tcb(
        &mut tcb, &k_buf, &pos_buf, n_k_heads, head_dim, kv_dim, b, theta,
    )
    .expect("rope K");
    tcb.commit_and_wait().expect("commit");
    let q_out = read_f32_buf(&q_buf, b * q_dim);
    let k_out = read_f32_buf(&k_buf, b * kv_dim);
    (q_out, k_out)
}

fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
            (x / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn rope_qk_fused_matches_separate_bit_identical() {
    // Qwen-3B-like dimensions: n_heads=16, n_kv_heads=8, head_dim=128, B=1..8
    let configs: &[(usize, usize, usize, &[u32])] = &[
        (16, 8, 128, &[0]),             // B=1, pos=0 (identity rotation)
        (16, 8, 128, &[1, 7]),          // B=2
        (16, 8, 128, &[3, 17, 42, 99]), // B=4
        (16, 8, 128, &[0, 1, 511, 1023, 2047, 3, 77, 200]), // B=8
        (32, 32, 128, &[5, 15, 100]),   // square GQA (n_q=n_kv)
    ];
    let theta = 1000000.0f32; // Qwen rope base

    for &(n_q, n_k, hd, positions) in configs {
        let b = positions.len();
        let q = rand_vec(b * n_q * hd, 0xDEAD);
        let k = rand_vec(b * n_k * hd, 0xBEEF);

        let (fq, fk) = run_fused(&q, &k, n_q, n_k, hd, positions, theta);
        let (sq, sk) = run_separate(&q, &k, n_q, n_k, hd, positions, theta);

        let q_diff = fq
            .iter()
            .zip(&sq)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let k_diff = fk
            .iter()
            .zip(&sk)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert_eq!(
            q_diff, 0.0,
            "B={b} n_q={n_q} n_k={n_k} hd={hd}: Q max_diff={q_diff} (expected 0)"
        );
        assert_eq!(
            k_diff, 0.0,
            "B={b} n_q={n_q} n_k={n_k} hd={hd}: K max_diff={k_diff} (expected 0)"
        );
        eprintln!("B={b} n_q_heads={n_q} n_kv_heads={n_k} head_dim={hd}: bit-identical OK");
    }
}
