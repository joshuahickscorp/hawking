//! q3k_fused_2r — parity between gemv_q3_k_pinned_tcb (gemm_q3_k_fused_v2,
//! 8 rows/TG, one row per simdgroup) and gemv_q3_k_fused_2r_pinned_tcb
//! (gemm_q3_k_fused_2r, 16 rows/TG, two rows per simdgroup with two accumulator
//! chains sharing the `x` load).
//!
//! Both use the SAME inline 6-bit Q3_K scale decode and the same per-element
//! `d*scale*q * xv` FMA in the same order; the 2r kernel only changes the row
//! pairing and shares the activation load. There is NO predec scale-table round
//! involved, so unlike q3k_predec_parity this is expected BIT-IDENTICAL. The
//! test asserts exact equality first and falls back to atol 1e-3 (the project
//! fp16 bar) only if the compiler FMA-recontracts the shared-`x` form
//! differently — that fallback is logged loudly so a real bug can't hide.
//!
//! This is the byte-cut speed-viability validation: gemm_q3_k_fused_2r is the
//! fewest-byte Q3_K GEMV (110 B/block, no scale table) given the 2-row-ILP
//! fast-path structure. SYNTHETIC weights — no model load. GPU-gated.

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use half::f16;
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

/// Synthetic Q3_K weights: 110 bytes/block. Bytes 0..108 (hmask + qs + packed
/// 6-bit scales) are arbitrary; byte 108..110 is a small positive fp16 `d`.
/// Matches the generator in `q3k_predec_parity.rs` / `v1_1_q3_k_parity.rs`.
fn make_q3k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n_blocks = rows * (cols / 256);
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 110];
    for b in 0..n_blocks {
        let off = b * 110;
        for i in 0..108 {
            bytes[off + i] = rng.gen::<u8>();
        }
        let d = 0.004 + rng.gen::<f32>() * 0.004;
        bytes[off + 108..off + 110].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    }
    bytes
}

fn make_x(cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..cols).map(|_| rng.gen_range(-3.0_f32..3.0_f32)).collect()
}

fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

fn read_f32_buf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn run_one(rows: usize, cols: usize, seed: u64) {
    let ctx = ctx();

    let w_bytes = make_q3k_bytes(rows, cols, seed);
    let model_buf = ctx.new_buffer_with_bytes(&w_bytes);

    let x = make_x(cols, 0xCAFE_F00D ^ seed);
    let x_buf = new_f32_buf(ctx, &x);

    // Baseline: gemm_q3_k_fused_v2 (8 rows/TG).
    let y_v2_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q3_k_pinned_tcb(
            &mut tcb, &model_buf, 0, w_bytes.len(),
            rows, cols, &x_buf, &y_v2_buf,
        ).expect("q3_k fused_v2 encode");
        tcb.commit_and_wait().expect("q3_k fused_v2 commit");
    }
    let y_v2 = read_f32_buf(&y_v2_buf, rows);

    // Candidate: gemm_q3_k_fused_2r (16 rows/TG, 2-row ILP).
    let y_2r_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q3_k_fused_2r_pinned_tcb(
            &mut tcb, &model_buf, 0, w_bytes.len(),
            rows, cols, &x_buf, &y_2r_buf,
        ).expect("q3_k fused_2r encode");
        tcb.commit_and_wait().expect("q3_k fused_2r commit");
    }
    let y_2r = read_f32_buf(&y_2r_buf, rows);

    // Bit-identical check first.
    let mut bit_identical = true;
    let mut max_abs = 0.0_f32;
    let mut worst = 0usize;
    for i in 0..rows {
        if y_v2[i].to_bits() != y_2r[i].to_bits() {
            bit_identical = false;
        }
        let d = (y_v2[i] - y_2r[i]).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
    }

    if bit_identical {
        eprintln!(
            "[q3k_fused_2r parity {rows}x{cols}] BIT-IDENTICAL to fused_v2 ({rows} rows)"
        );
    } else {
        // Fall back to the project fp16 bar; log loudly so a real bug surfaces.
        const ATOL: f32 = 1e-3;
        assert!(
            max_abs < ATOL,
            "q3k_fused_2r exceeds fp16 tol vs fused_v2: max_abs={max_abs:e} (atol {ATOL}) \
             at i={worst}  v2={}  2r={}",
            y_v2[worst], y_2r[worst],
        );
        eprintln!(
            "[q3k_fused_2r parity {rows}x{cols}] NOT bit-identical (compiler FMA-recontraction); \
             within fp16 tol max_abs={max_abs:e} (atol {ATOL})"
        );
    }
}

#[test]
fn q3k_fused_2r_matches_fused_v2() {
    // Three representative Qwen2.5-3B decode GEMV shapes. All rows%16==0.
    run_one(2048, 2048, 0x3D15_8E1E);
    run_one(11008, 2048, 0x51C0_0001);
    run_one(2048, 11008, 0x7A11_BEEF);
}

/// Cover rows NOT divisible by 16 (the has1 alias path: the last TG has a
/// row0 whose row1 is past the end). rows=2056 => last TG handles rows
/// 2048..2063, of which only 2048..2055 exist; row 2056's simdgroup writes
/// row0=2056 and aliases row1=2064→2056 (never written).
#[test]
fn q3k_fused_2r_ragged_rows() {
    run_one(2056, 2048, 0x0DD0_1234);
}
