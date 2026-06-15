#![cfg(target_os = "macos")]
//! Continuous-batching multi-seq decode MHA parity (build task #1).
//!
//! Kernel under test: `mha_decode_f32_batched_multiseq` — B INDEPENDENT
//! sequences in one dispatch, each with its OWN position (`positions[bi]`) and
//! its OWN slot-strided K/V region (`bi * kv_slot_stride` elements). This is the
//! one genuinely-new kernel the continuous-batching decode path needs; the
//! existing `mha_decode_f32_batched` shares a single K/V window across the batch
//! (B tokens of ONE sequence), which is wrong for multi-stream serving.
//!
//! Verified vs the CPU `attn::mha_decode_step` run per slot over that slot's own
//! causal prefix `[0..positions[bi]+1)` at atol=1e-3 (fp16 floor, never loosened).
//! A degenerate B=1 case catches indexing bugs; a MANDATORY long-context case
//! runs at position 2047.

use dismantle_core::attn::mha_decode_step;
use dismantle_core::kernels;
use dismantle_core::metal::TokenCommandBuffer;

mod common;
use common::*;

const N_HEADS: usize = 16;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;

fn u32_buf(
    ctx: &dismantle_core::metal::MetalContext,
    data: &[u32],
) -> dismantle_core::metal::PinnedBuffer {
    let mut bytes = vec![0u8; data.len() * std::mem::size_of::<u32>()];
    for (i, &x) in data.iter().enumerate() {
        bytes[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes());
    }
    ctx.new_buffer_with_bytes(&bytes)
}

fn run_multiseq(label: &str, positions: &[u32], max_seq: usize, atol: f32) {
    let ctx = ctx();
    let b = positions.len();
    let q_dim = N_HEADS * HEAD_DIM;
    let kv_dim = N_KV_HEADS * HEAD_DIM;
    let stride = max_seq * kv_dim; // elements per slot's K (and V) region

    // B query rows (B, n_heads, head_dim) + B slot-strided K/V regions.
    let q = fixed_f32(b * q_dim, 0x5EED_0001 ^ b as u64);
    let k = fixed_f32(b * stride, 0x0B2B_2B2B ^ b as u64);
    let v = fixed_f32(b * stride, 0x0C3C_3C3C ^ b as u64);

    // CPU reference: each slot attends ONLY over its own region [0..seq_bi).
    let mut ref_cpu = vec![0.0f32; b * q_dim];
    for bi in 0..b {
        let seq_bi = positions[bi] as usize + 1;
        assert!(
            seq_bi <= max_seq,
            "seq_bi {seq_bi} exceeds max_seq {max_seq}"
        );
        let q_bi = &q[bi * q_dim..(bi + 1) * q_dim];
        let k_bi = &k[bi * stride..bi * stride + seq_bi * kv_dim];
        let v_bi = &v[bi * stride..bi * stride + seq_bi * kv_dim];
        let out_bi = &mut ref_cpu[bi * q_dim..(bi + 1) * q_dim];
        mha_decode_step(
            q_bi, k_bi, v_bi, N_HEADS, N_KV_HEADS, HEAD_DIM, seq_bi, out_bi,
        )
        .expect("cpu mha_decode_step (slot)");
    }

    // GPU under test.
    let q_buf = new_f32_buf(ctx, &q);
    let k_buf = new_f32_buf(ctx, &k);
    let v_buf = new_f32_buf(ctx, &v);
    let pos_buf = u32_buf(ctx, positions);
    // region == batch index here (each slot's KV is at its own bi*stride region).
    let region_ids: Vec<u32> = (0..b as u32).collect();
    let region_buf = u32_buf(ctx, &region_ids);
    let out_buf = ctx.new_buffer(b * q_dim * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::mha_decode_f32_batched_multiseq_tcb(
            &mut tcb,
            &q_buf,
            &k_buf,
            0,
            &v_buf,
            0,
            &out_buf,
            &pos_buf,
            &region_buf,
            max_seq,
            stride,
            b,
            HEAD_DIM,
            N_HEADS,
            N_KV_HEADS,
        )
        .expect("multiseq encode");
        tcb.commit_and_wait().expect("multiseq commit");
    }
    let actual = read_f32_buf(&out_buf, b * q_dim);

    let diff = max_abs_diff(&ref_cpu, &actual);
    println!(
        "[multiseq] {label}: b={b} positions={positions:?} diff_vs_cpu={diff:.3e} atol={atol:.0e}"
    );
    assert!(
        diff < atol,
        "{label}: multiseq vs CPU diff {diff:.3e} >= {atol:.0e}"
    );
}

#[test]
fn multiseq_divergent_positions() {
    // The core case: B independent sequences at DISTINCT positions.
    run_multiseq("divergent", &[5, 12, 2, 0], 16, ATOL);
}

#[test]
fn multiseq_b1_matches_single() {
    // Degenerate B=1 — catches per-slot indexing bugs.
    run_multiseq("b=1 pos=7", &[7], 16, ATOL);
}

#[test]
fn multiseq_b8_mixed() {
    run_multiseq("b=8 mixed", &[0, 1, 3, 7, 15, 4, 9, 2], 16, ATOL);
}

// MANDATORY long-context case (a slot near 2K positions).
#[test]
fn multiseq_long_context() {
    run_multiseq("long-ctx", &[2047, 1024, 512, 100], 2048, ATOL);
}
