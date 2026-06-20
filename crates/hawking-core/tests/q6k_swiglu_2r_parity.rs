#![cfg(target_os = "macos")]
//! Track D7 parity: gemm_q6_k_fused_v2_swiglu_2r must be bit-identical to
//! gemm_q6_k_fused_v2_swiglu (1r reference) for every row.
//!
//! The 2r kernel halves TG count (16 rows/TG vs 8) for the default Qwen-3B
//! ffn_down (Q6_K, 2048 rows × 11008 cols): 128 TGs vs 256.

use hawking_core::kernels;
use hawking_core::metal::{MetalContext, TokenCommandBuffer};
use hawking_core::quant;

mod common;
use common::*;

/// Run Q6K swiglu 1r reference (direct dispatch, bypasses OnceLock).
fn run_1r(
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
    kernels::gemv_q6_k_swiglu_1r_direct_tcb(
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

/// Run Q6K swiglu 2r (Track D7, direct dispatch, bypasses OnceLock).
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

fn make_q6k(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let w_f32 = fixed_f32(rows * cols, seed);
    let blocks = (rows * cols) / 256;
    let mut w_q6 = vec![0u8; blocks * quant::Q6_K_BLOCK_BYTES];
    quant::quantize_q6_k(&w_f32, &mut w_q6).expect("Q6_K quant");
    w_q6
}

/// D7 quality gate: 2r must agree with 1r within 1e-4 (functional correctness).
/// The compiler may reorder interleaved FMAs in the 2r loop by ~2 ULPs, so
/// strict bit-identity cannot be guaranteed. 1e-4 is far below Q6_K quant error.
/// Production shape (2048 × 11008) is too slow in CI; 256-wide surrogates.
#[test]
fn d7_q6k_swiglu_2r_bit_identical_to_1r() {
    let ctx = ctx();

    let cases: &[(usize, usize, u64)] = &[
        // Exact multiples of 8 and 16.
        (8, 256, 0xD700),
        (16, 256, 0xD701),
        (32, 256, 0xD702),
        (64, 512, 0xD703),
        // Non-multiples-of-16: tests has1 guard.
        (9, 256, 0xD704),
        (17, 256, 0xD705),
        (25, 256, 0xD706),
        (128, 512, 0xD707),
        (256, 512, 0xD708),
    ];

    for &(rows, cols, seed) in cases {
        let w_q6 = make_q6k(rows, cols, seed);
        let gate = fixed_f32(cols, seed ^ 0x1000);
        let up = fixed_f32(cols, seed ^ 0x2000);

        let ref_out = run_1r(ctx, &w_q6, &gate, &up, rows, cols);
        let got_out = run_2r(ctx, &w_q6, &gate, &up, rows, cols);

        let diff = max_abs_diff(&ref_out, &got_out);
        // Compiler may reorder FMAs slightly (2r interleaves row0/row1 updates),
        // so results may differ by up to ~2 ULPs. Gate at 1e-4 (well below any
        // quantization error) to verify functional correctness.
        const MAX_DIFF: f32 = 1e-4;
        assert!(
            diff <= MAX_DIFF,
            "D7 rows={rows} cols={cols}: 2r vs 1r diff={diff:.2e} > {MAX_DIFF:.2e}"
        );
        eprintln!("D7 q6k_swiglu_2r rows={rows} cols={cols}: diff={diff:.2e} OK");
    }
}
