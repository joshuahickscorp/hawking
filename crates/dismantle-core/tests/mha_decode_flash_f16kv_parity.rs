#![cfg(target_os = "macos")]
//! Wave-R6 — parity for `mha_decode_flash_f16kv_tcb` (GQA flash online-softmax
//! decode reading a HALF K/V cache). It is validated against the CPU reference
//! `crate::attn::mha_decode_step` computed on the SAME f16-roundtripped K/V:
//! flash widens each cached `half` to float exactly as `f16::to_f32`, so the
//! f16 rounding is identical on both sides and the only residual difference is
//! the tile-wise online-softmax reduction reorder.
//!
//! Tolerance: atol = 1e-3 AND rtol = 1e-4 (same contract as
//! `mha_decode_flash_parity.rs` — the atol floor is the kernel parity floor and
//! is never loosened; the rtol covers the reorder). seq ∈ {1,128,129,384,4096}
//! exercises the partial-tile tail (`t_len = min(FLASH_TG, seq - t_base)`) and
//! the long-context (4096 = 32 tiles) regime this kernel exists to enable — a
//! length the O(seq)-shmem `mha_decode_f16kv` cannot reach near the 32 KB cap.

use dismantle_core::attn::mha_decode_step;
use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use half::f16;

mod common;
use common::*;

/// |a - b| <= atol + rtol*|b|; returns the worst excess (0.0 if all pass) + idx.
fn worst_violation(actual: &[f32], reference: &[f32], atol: f32, rtol: f32) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for (i, (&a, &r)) in actual.iter().zip(reference.iter()).enumerate() {
        let excess = (a - r).abs() - (atol + rtol * r.abs());
        if excess > worst {
            worst = excess;
            worst_i = i;
        }
    }
    (worst, worst_i)
}

const ATOL: f32 = 1e-3;
const RTOL: f32 = 1e-4;

/// Pin an f32 slice as a little-endian f16 buffer (the f16 K/V cache layout).
fn new_f16_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes())
        .collect();
    ctx.new_buffer_with_bytes(&bytes)
}

/// f16 round-trip a slice (what the kernel's `(float)half` widening yields).
fn f16_round_trip(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
}

/// Run the flash-f16kv kernel for one decode step. k/v are passed as f32 and
/// stored into the half cache here.
fn run_flash_f16kv(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    let q_dim = n_heads * head_dim;
    let ctx = ctx();
    let q_buf = new_f32_buf(ctx, q);
    let k_buf = new_f16_buf(ctx, k);
    let v_buf = new_f16_buf(ctx, v);
    let out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::mha_decode_flash_f16kv_tcb(
            &mut tcb, &q_buf, &k_buf, 0, &v_buf, 0, &out_buf, seq_len, head_dim, n_heads,
            n_kv_heads,
        )
        .expect("mha_decode_flash_f16kv_tcb encode");
        tcb.commit_and_wait()
            .expect("mha_decode_flash_f16kv_tcb commit");
    }
    read_f32_buf(&out_buf, q_dim)
}

/// flash-f16kv vs CPU ref on the SAME f16-roundtripped K/V, atol 1e-3 + rtol 1e-4.
fn check_geometry(n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) {
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let seed = (seq_len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ ((n_heads as u64) << 17)
        ^ ((head_dim as u64) << 33);
    let q = fixed_f32(q_dim, seed ^ 0xA1);
    let k = fixed_f32(seq_len * kv_dim, seed ^ 0xB2);
    let v = fixed_f32(seq_len * kv_dim, seed ^ 0xC3);

    // CPU reference on the f16-roundtripped cache (kernel widens half→float the
    // same way), isolating the difference to the online-softmax reorder.
    let k_rt = f16_round_trip(&k);
    let v_rt = f16_round_trip(&v);
    let mut cpu = vec![0.0f32; q_dim];
    mha_decode_step(
        &q, &k_rt, &v_rt, n_heads, n_kv_heads, head_dim, seq_len, &mut cpu,
    )
    .expect("cpu mha_decode_step");

    let flash = run_flash_f16kv(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);

    let (vf, i) = worst_violation(&flash, &cpu, ATOL, RTOL);
    assert!(
        vf <= 0.0,
        "flash_f16kv vs CPU(f16-rt): seq={seq_len} h={n_heads} kvh={n_kv_heads} hd={head_dim}: \
         violation {vf} beyond atol={ATOL}+rtol={RTOL} at i={i} (flash={} cpu={})",
        flash[i],
        cpu[i]
    );
    eprintln!("flash_f16kv seq={seq_len} h={n_heads} kvh={n_kv_heads} hd={head_dim}: OK");
}

/// Production Qwen2.5-3B geometry (head_dim=128, GQA 16/2) across the tile
/// boundaries that break flash kernels: 1, 128, 129, 384, and 4096 (long ctx).
#[test]
fn flash_f16kv_matches_ref_qwen_geometry_multi_tile() {
    let (n_heads, n_kv_heads, head_dim) = (16usize, 2usize, 128usize);
    for &seq_len in &[1usize, 128, 129, 384, 4096] {
        check_geometry(n_heads, n_kv_heads, head_dim, seq_len);
    }
}

/// MHA (non-grouped) at a partial-tile boundary, group_size == 1 path.
#[test]
fn flash_f16kv_full_mha_tile_boundary() {
    check_geometry(4, 4, 128, 129);
}

/// Long-context headline: 4096 standalone so a failure names the regime directly.
#[test]
fn flash_f16kv_long_context_4k() {
    check_geometry(16, 2, 128, 4096);
}
