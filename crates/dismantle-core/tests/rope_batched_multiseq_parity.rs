#![cfg(target_os = "macos")]
//! R2 parity: `rope_f32_batched_multiseq` == the per-slot `rope_q_f32_inplace_off`
//! loop it replaces, BIT-IDENTICAL.
//!
//! The multi-seq decode stack used to RoPE each slot with its own dispatch
//! (`rope_q_f32_inplace_off_tcb` × B, for Q and K). R2 batches that into ONE
//! dispatch per tensor reading a per-slot `positions[]` buffer. RoPE is purely
//! elementwise (no cross-element reduction), so the batched kernel must produce
//! byte-identical output to the per-slot loop — not merely atol-close. This test
//! runs BOTH on the GPU over B slots at divergent positions (incl. pos 0 = the
//! identity rotation and a long-context pos 2047) and asserts max_abs_diff == 0,
//! for both the Q width (n_heads) and the K width (n_kv_heads).

use dismantle_core::kernels;
use dismantle_core::metal::TokenCommandBuffer;

mod common;
use common::*;

fn run_per_slot(
    x: &[f32],
    n_heads: usize,
    head_dim: usize,
    slot_dim: usize,
    positions: &[u32],
    theta: f32,
) -> Vec<f32> {
    let ctx = ctx();
    let buf = new_f32_buf(ctx, x);
    let b = positions.len();
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        for bi in 0..b {
            kernels::rope_q_f32_inplace_off_tcb(
                &mut tcb,
                &buf,
                bi * slot_dim * std::mem::size_of::<f32>(),
                n_heads,
                head_dim,
                0,
                head_dim,
                positions[bi],
                theta,
            )
            .expect("per-slot rope encode");
        }
        tcb.commit_and_wait().expect("per-slot rope commit");
    }
    read_f32_buf(&buf, b * slot_dim)
}

fn run_batched(
    x: &[f32],
    n_heads: usize,
    head_dim: usize,
    slot_dim: usize,
    positions: &[u32],
    theta: f32,
) -> Vec<f32> {
    let ctx = ctx();
    let buf = new_f32_buf(ctx, x);
    let b = positions.len();
    let pos_bytes: Vec<u8> = positions.iter().flat_map(|&p| p.to_le_bytes()).collect();
    let pos_buf = ctx.new_buffer_with_bytes(&pos_bytes);
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::rope_f32_batched_multiseq_tcb(
            &mut tcb, &buf, &pos_buf, n_heads, head_dim, slot_dim, b, theta,
        )
        .expect("batched rope encode");
        tcb.commit_and_wait().expect("batched rope commit");
    }
    read_f32_buf(&buf, b * slot_dim)
}

#[test]
fn rope_batched_multiseq_matches_per_slot() {
    let n_heads = 16usize; // Qwen2.5-3B Q heads
    let n_kv_heads = 2usize; // GQA KV heads
    let head_dim = 128usize;
    let theta = 1_000_000.0f32; // Qwen rope_theta
    let positions: [u32; 5] = [2047, 13, 500, 0, 1024];
    let b = positions.len();

    // Q width.
    let q_dim = n_heads * head_dim;
    let xq = fixed_f32(b * q_dim, 0x5151_5151_5151_5151);
    let expected_q = run_per_slot(&xq, n_heads, head_dim, q_dim, &positions, theta);
    let actual_q = run_batched(&xq, n_heads, head_dim, q_dim, &positions, theta);
    let diff_q = max_abs_diff(&expected_q, &actual_q);
    assert_eq!(diff_q, 0.0, "Q rope: batched != per-slot (max_abs_diff {diff_q})");

    // K width (GQA: fewer heads, narrower slot stride).
    let kv_dim = n_kv_heads * head_dim;
    let xk = fixed_f32(b * kv_dim, 0x6262_6262_6262_6262);
    let expected_k = run_per_slot(&xk, n_kv_heads, head_dim, kv_dim, &positions, theta);
    let actual_k = run_batched(&xk, n_kv_heads, head_dim, kv_dim, &positions, theta);
    let diff_k = max_abs_diff(&expected_k, &actual_k);
    assert_eq!(diff_k, 0.0, "K rope: batched != per-slot (max_abs_diff {diff_k})");

    // B=1 degenerate case must also match.
    let one = [777u32];
    let x1 = fixed_f32(q_dim, 0x7373_7373_7373_7373);
    let e1 = run_per_slot(&x1, n_heads, head_dim, q_dim, &one, theta);
    let a1 = run_batched(&x1, n_heads, head_dim, q_dim, &one, theta);
    assert_eq!(max_abs_diff(&e1, &a1), 0.0, "B=1 rope: batched != per-slot");

    println!("[rope-batched-multiseq] Q+K+B=1 bit-identical over positions {positions:?}");
}
