#![cfg(target_os = "macos")]
//! Phase 2.3 — parity test for `mha_decode_flash_f32_tcb` (GQA online-softmax
//! flash decode) against BOTH the CPU reference `crate::attn::mha_decode_step`
//! AND the existing GPU `mha_decode_f32_tcb` (the materialize-all-scores path
//! it replaces).
//!
//! Tolerance: atol = 1e-3 AND rtol = 1e-4. The flash kernel recomputes the
//! softmax tile-wise with a running max/sum, so the reduction order differs
//! from both the per-thread CPU loop and the single-pass GPU kernel — a
//! reduction reorder, NOT a bug. The atol floor (1e-3) is the kernel parity
//! floor and is never loosened; the rtol (1e-4) covers the reorder. (The
//! sibling `mha_decode_metal_parity.rs` asserts the materialize kernel at the
//! tighter 1e-4; flash cannot in general hit that, hence the spec'd
//! 1e-3 + rtol.)
//!
//! Multi-tile boundaries are tested explicitly (seq = 1, 128, 129, 384, 4096):
//! the partial-tile `t_len = min(FLASH_TG, seq - t_base)` path is where flash
//! kernels break (the v1l MLA flash test learned the same lesson). FLASH_TG is
//! 128, so 129 and 384 exercise non-tile-aligned tails and 4096 exercises the
//! long-context regime this kernel exists to enable.

use hawking_core::attn::mha_decode_step;
use hawking_core::kernels;
use hawking_core::metal::TokenCommandBuffer;

mod common;
use common::*;

/// atol/rtol combined check: |a - b| <= atol + rtol * |b|, with `b` the
/// reference. Returns the worst (signed-into-abs) violation magnitude beyond
/// the allowed band (0.0 if all pass) and the index for diagnostics.
fn worst_violation(actual: &[f32], reference: &[f32], atol: f32, rtol: f32) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for (i, (&a, &r)) in actual.iter().zip(reference.iter()).enumerate() {
        let allowed = atol + rtol * r.abs();
        let excess = (a - r).abs() - allowed;
        if excess > worst {
            worst = excess;
            worst_i = i;
        }
    }
    (worst, worst_i)
}

const ATOL: f32 = 1e-3;
const RTOL: f32 = 1e-4;

/// Run the flash kernel for one decode step and return its output.
fn run_flash(
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
    let k_buf = new_f32_buf(ctx, k);
    let v_buf = new_f32_buf(ctx, v);
    let out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::mha_decode_flash_f32_tcb(
            &mut tcb, &q_buf, &k_buf, 0, &v_buf, 0, &out_buf, seq_len, head_dim, n_heads,
            n_kv_heads,
        )
        .expect("mha_decode_flash_f32_tcb encode");
        tcb.commit_and_wait()
            .expect("mha_decode_flash_f32_tcb commit");
    }
    read_f32_buf(&out_buf, q_dim)
}

/// Run the existing materialize kernel for one decode step (the path flash
/// replaces) so we can assert flash matches it too, not just the CPU ref.
fn run_materialize(
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
    let k_buf = new_f32_buf(ctx, k);
    let v_buf = new_f32_buf(ctx, v);
    let out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::mha_decode_f32_tcb(
            &mut tcb, &q_buf, &k_buf, 0, &v_buf, 0, &out_buf, seq_len, head_dim, n_heads,
            n_kv_heads,
        )
        .expect("mha_decode_f32_tcb encode");
        tcb.commit_and_wait().expect("mha_decode_f32_tcb commit");
    }
    read_f32_buf(&out_buf, q_dim)
}

/// Core parity check at one geometry: flash vs CPU ref AND flash vs the
/// materialize GPU kernel, both at atol=1e-3 + rtol=1e-4.
fn check_geometry(n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) {
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    // Distinct seeds per geometry so cases don't accidentally share inputs.
    let seed = (seq_len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ ((n_heads as u64) << 17)
        ^ ((head_dim as u64) << 33);
    let q = fixed_f32(q_dim, seed ^ 0xA1);
    let k = fixed_f32(seq_len * kv_dim, seed ^ 0xB2);
    let v = fixed_f32(seq_len * kv_dim, seed ^ 0xC3);

    // CPU reference.
    let mut cpu = vec![0.0f32; q_dim];
    mha_decode_step(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, &mut cpu)
        .expect("cpu mha_decode_step");

    // Flash GPU output.
    let flash = run_flash(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);

    // Materialize GPU output (the path flash replaces).
    let materialize = run_materialize(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);

    let (vf_cpu, i_cpu) = worst_violation(&flash, &cpu, ATOL, RTOL);
    assert!(
        vf_cpu <= 0.0,
        "flash vs CPU: seq={seq_len} h={n_heads} kvh={n_kv_heads} hd={head_dim}: \
         violation {vf_cpu} beyond atol={ATOL}+rtol={RTOL} at i={i_cpu} \
         (flash={} cpu={})",
        flash[i_cpu],
        cpu[i_cpu]
    );

    let (vf_mat, i_mat) = worst_violation(&flash, &materialize, ATOL, RTOL);
    assert!(
        vf_mat <= 0.0,
        "flash vs materialize: seq={seq_len} h={n_heads} kvh={n_kv_heads} hd={head_dim}: \
         violation {vf_mat} beyond atol={ATOL}+rtol={RTOL} at i={i_mat} \
         (flash={} materialize={})",
        flash[i_mat],
        materialize[i_mat]
    );
}

/// Production Qwen2.5-3B geometry (head_dim=128, GQA 16/2) across the tile
/// boundaries that break flash kernels: 1, 128 (exactly one full tile), 129
/// (one full tile + a 1-token partial tail), 384 (3 full tiles), and 4096
/// (the long-context regime this kernel exists to enable — 32 tiles).
#[test]
fn flash_matches_refs_qwen_geometry_multi_tile() {
    let (n_heads, n_kv_heads, head_dim) = (16usize, 2usize, 128usize);
    for &seq_len in &[1usize, 128, 129, 384, 4096] {
        check_geometry(n_heads, n_kv_heads, head_dim, seq_len);
    }
}

/// seq_len = 1 (first decode token) at a minimal geometry: the softmax is a
/// single element (weight 1.0) so flash must reproduce V[0] exactly within
/// tolerance, and the n_heads < 32-simdgroup-count path is still correct
/// because FLASH_TG fixes the simdgroup count, not n_heads.
#[test]
fn flash_seq_len_one() {
    check_geometry(2, 1, 128, 1);
}

/// MHA (non-grouped: n_kv_heads == n_heads) at a tile boundary, to cover the
/// group_size == 1 path distinctly from GQA.
#[test]
fn flash_full_mha_tile_boundary() {
    check_geometry(4, 4, 128, 129);
}

/// Long-context correctness is the headline of this spike: assert the 4096
/// case standalone so a failure names the long-context regime directly. This
/// length is impossible on the materialize kernel near the 32 KB cap at large
/// n_heads, which is the whole reason flash exists; here both kernels run so
/// we get a direct A/B at 4K.
#[test]
fn flash_long_context_4k() {
    check_geometry(16, 2, 128, 4096);
}
