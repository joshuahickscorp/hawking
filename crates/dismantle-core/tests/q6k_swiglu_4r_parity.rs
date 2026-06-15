#![cfg(target_os = "macos")]
//! Track D8 parity: `gemm_q6_k_fused_v2_swiglu_4r` must produce outputs
//! matching `gemm_q6_k_fused_v2_swiglu_2r` (Track D7, default-on) within 1e-4.
//!
//! The 4r kernel handles 4 rows per simdgroup (32 rows/TG) vs 2 rows for 2r
//! (16 rows/TG), halving TG count for the Qwen-3B ffn_down shape
//! (2048 rows → 64 TGs vs 128). Each row reads a separate Q6K weight stream;
//! FMA interleaving across 4 rows may differ by ~2 ULPs vs the 2r reference.
//! Gate 1e-4 is well below Q6K quantization error (~1e-2 rel scale).

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, TokenCommandBuffer};
use dismantle_core::quant;

mod common;
use common::*;

fn make_q6k(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let w_f32 = fixed_f32(rows * cols, seed);
    let blocks = (rows * cols) / 256;
    let mut w_q6 = vec![0u8; blocks * quant::Q6_K_BLOCK_BYTES];
    quant::quantize_q6_k(&w_f32, &mut w_q6).expect("Q6_K quant");
    w_q6
}

/// Run Q6K swiglu 2r (Track D7 reference, direct dispatch).
fn run_2r(
    ctx: &MetalContext,
    w_q6: &[u8],
    gate: &[f32],
    up: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let model_buf = ctx.new_buffer_with_bytes(w_q6);
    let gate_buf = new_f32_buf(ctx, gate);
    let up_buf = new_f32_buf(ctx, up);
    let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q6_k_swiglu_2r_direct_tcb(
        &mut tcb,
        &model_buf,
        0,
        w_q6.len(),
        rows,
        cols,
        &gate_buf,
        &up_buf,
        &out_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&out_buf, rows)
}

/// Run Q6K swiglu 4r (Track D8 candidate, direct dispatch).
fn run_4r(
    ctx: &MetalContext,
    w_q6: &[u8],
    gate: &[f32],
    up: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let model_buf = ctx.new_buffer_with_bytes(w_q6);
    let gate_buf = new_f32_buf(ctx, gate);
    let up_buf = new_f32_buf(ctx, up);
    let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q6_k_swiglu_4r_direct_tcb(
        &mut tcb,
        &model_buf,
        0,
        w_q6.len(),
        rows,
        cols,
        &gate_buf,
        &up_buf,
        &out_buf,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    read_f32_buf(&out_buf, rows)
}

/// D8 quality gate: 4r must agree with 2r within 1e-4.
/// FMA interleaving across 4 independent row accumulators may differ by ~2 ULPs
/// vs the 2r reference. 1e-4 is far below Q6K quantization error and catches any
/// structural bug (wrong block pointer, scale misread, missing has guard).
///
/// Key boundary cases (32 rows/TG for 4r):
///  - rows=32:  exactly 1 TG (4 simdgroups × 8 rows)
///  - rows=33:  2 TGs; second TG has 1 row (has1/has2/has3 all false for simd_id>0)
///  - rows=40:  2 TGs; second TG has 8 rows
///  - rows=64:  exactly 2 TGs
///  - rows=256, cols=256: larger shapes
#[test]
fn d8_q6k_swiglu_4r_matches_2r() {
    let ctx = ctx();
    const MAX_DIFF: f32 = 1e-4;

    let cases: &[(usize, usize, u64)] = &[
        (32, 256, 0xD800), // exactly 1 TG for 4r (32 rows/TG)
        (33, 256, 0xD801), // 2 TGs, 2nd has 1 row: tests has1/has2/has3 guards
        (40, 256, 0xD802), // 2 TGs, 2nd has 8 rows
        (64, 512, 0xD803), // exactly 2 TGs
        (128, 512, 0xD804),
        (256, 512, 0xD805),
        // Non-multiples of 32 to stress the has-guard logic.
        (17, 256, 0xD806),
        (25, 256, 0xD807),
        (48, 512, 0xD808),
    ];

    for &(rows, cols, seed) in cases {
        let w_q6 = make_q6k(rows, cols, seed);
        let gate = fixed_f32(cols, seed ^ 0x1000);
        let up = fixed_f32(cols, seed ^ 0x2000);

        let ref_out = run_2r(ctx, &w_q6, &gate, &up, rows, cols);
        let got_out = run_4r(ctx, &w_q6, &gate, &up, rows, cols);

        let diff = max_abs_diff(&ref_out, &got_out);
        assert!(
            diff <= MAX_DIFF,
            "D8 rows={rows} cols={cols}: 4r vs 2r diff={diff:.2e} > {MAX_DIFF:.2e}"
        );
        eprintln!("D8 q6k_swiglu_4r rows={rows} cols={cols}: diff={diff:.2e} OK");
    }
}
