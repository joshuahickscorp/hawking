//! q3k_bytecut_bench — characterizes the Q3_K GEMV kernel options for the
//! byte-cut win (Q3 < Q4 weight bytes, but the byte-cut was "tps-blocked on a
//! fast Q3_K kernel"). A full Q3_K dense path is too big to wire unattended;
//! this microbench instead settles WHICH Q3_K GEMV a future byte-cut should use
//! and whether the byte-cut is speed-viable at the GEMV level.
//!
//! The Q4_K story: predec won +40% because Q4's inline 6-bit scale-decode was
//! expensive, and the +scale-byte cost was paid back by the compute saving.
//! The Q3_K subtlety is the OPPOSITE risk: predec ADDS f32 scale bytes
//! (anti-byte-cut), so on a strictly bandwidth-bound GEMV the fewer-byte FUSED
//! kernel may beat the predec one. This bench measures it.
//!
//! Three kernels, three representative Qwen2.5-3B decode shapes, ITERS iters
//! after WARMUP, each iteration a fresh TCB committed-and-waited (one full GPU
//! dispatch round-trip = the decode-path cost):
//!   1. Q3_K fused  — gemv_q3_k_pinned_tcb (gemm_q3_k_fused_v2): 110 B/block,
//!      inline 6-bit scale decode, NO scale table.
//!   2. Q3_K predec — gemv_q3_k_v4_predec_pinned_tcb (gemm_q3_k_v4_predec):
//!      110 B weights + 16 f32 pre-decoded scales (64 B) = 174 B/block. The
//!      committed Q3_K predec kernel reads f32 scales (predecode_q3_k_scale_table
//!      returns f32; there is NO Q3_K f16-scales variant today).
//!   3. Q4_K predec — gemv_q4_k_v4_predec_pinned_tcb (gemm_q4_k_v4_predec): the
//!      byte-cut comparison baseline. 144 B weights + 16 f32 scales (64 B) =
//!      208 B/block.
//!
//! NOT a correctness gate (parity lives in q3k_predec_parity.rs /
//! q4k_predec_parity.rs / q3k_fused_2r_parity.rs). This ignores numerical
//! output and only measures dispatch wall time + the actual bytes/block read +
//! achieved GB/s. GPU-gated. Marked #[ignore] so it never runs in the default
//! suite — run explicitly with `--ignored --nocapture`.
//!
//! 2026-05-31 — added a 4th kernel, gemm_q3_k_fused_2r (the FUSED Q3_K given
//! the 2-row-ILP / 2-accumulator / shared-`x` structure of
//! gemm_q4_k_v4_predec_2r), to test whether the byte-cut-TRUE (fewest-byte)
//! Q3_K becomes speed-viable. FINDING: it does NOT. Across the three shapes the
//! fused/fused_2r Q3_K run at only 7–21 GB/s while Q4_predec runs at 18–50 GB/s
//! on a ~150 GB/s machine — the Q3_K GEMV is NOT bandwidth-bound, it is
//! compute-bound on the inline 6-bit scale decode (q3_k_scale: branchy
//! low-nibble select + high-2-bit shift/mask + (-32), per element). 2-row ILP
//! attacks DRAM load latency, which is the WRONG bottleneck; it helps only the
//! 2048x2048 shape (+~8%) and REGRESSES the wide ffn shapes (−5% to −30%, from
//! register pressure of the doubled accumulator state). fused_2r stays −32% to
//! −55% slower than Q4_predec. The only competitive Q3_K kernel is predec
//! (which hoists the decode out), but predec ADDS 64 B/block of scales —
//! anti-byte-cut. Conclusion: the Q3_K byte-cut is NOT speed-viable via a
//! row-ILP fused kernel; it would need a cheaper-decode Q3_K layout (out of
//! scope here).

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::TokenCommandBuffer;
use dismantle_core::quant::predecode_q3_k_scale_table;
use half::f16;
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use std::time::Instant;

mod common;
use common::*;

/// Synthetic Q3_K weights, 110 B/block (matches q3k_predec_parity.rs). Bytes
/// 0..108 (hmask + qs + packed 6-bit scales) arbitrary; 108..110 a small +fp16 d.
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

/// Synthetic Q4_K weights, 144 B/block (matches q4k_predec_f16s_bench.rs).
fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n_blocks = rows * (cols / 256);
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        let d = 0.01_f32 + rng.gen::<f32>() * 0.01;
        let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}

fn make_x(cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..cols)
        .map(|_| rng.gen_range(-3.0_f32..3.0_f32))
        .collect()
}

const WARMUP: usize = 30;
const ITERS: usize = 200;

/// Time one dispatch `ITERS` times after `WARMUP`, each iteration a fresh TCB
/// committed-and-waited (wall time = one full GPU dispatch round-trip). Returns
/// mean µs/call.
fn time_dispatch<F>(label: &str, mut encode: F) -> f64
where
    F: FnMut(&mut TokenCommandBuffer<'_>),
{
    let ctx = ctx();
    for _ in 0..WARMUP {
        let mut tcb = TokenCommandBuffer::new(ctx);
        encode(&mut tcb);
        tcb.commit_and_wait().expect("warmup commit");
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let mut tcb = TokenCommandBuffer::new(ctx);
        encode(&mut tcb);
        tcb.commit_and_wait().expect("timed commit");
    }
    let elapsed = t0.elapsed();
    let us_per_call = elapsed.as_secs_f64() * 1e6 / ITERS as f64;
    eprintln!("  [{label:<16}] {us_per_call:.3} µs/call ({ITERS} iters)");
    us_per_call
}

/// GB/s = bytes / (µs * 1e3 ns).
fn gbps(bytes: f64, us: f64) -> f64 {
    bytes / (us * 1e3)
}

fn bench_shape(rows: usize, cols: usize, tag: &str) {
    let ctx = ctx();
    let blocks = rows * (cols / 256); // blocks per output row-vector total

    // --- Q3_K buffers (fused + predec share the same 110 B/block weights) ---
    let q3_w = make_q3k_bytes(rows, cols, 0x3D15_8E1E ^ (rows as u64));
    let q3_buf = ctx.new_buffer_with_bytes(&q3_w);
    let q3_scales = predecode_q3_k_scale_table(&q3_w); // 16 f32/block
    let q3_scales_buf = new_f32_buf(ctx, &q3_scales);

    // --- Q4_K buffers (predec baseline) ---
    let q4_w = make_q4k_bytes(rows, cols, 0xF165_8E1E ^ (rows as u64));
    let q4_buf = ctx.new_buffer_with_bytes(&q4_w);
    let q4_scales = kernels::predecode_q4_k_scale_table(&q4_w); // 16 f32/block
    let q4_scales_buf = new_f32_buf(ctx, &q4_scales);

    let x = make_x(cols, 0xCAFE_F00D);
    let x_buf = new_f32_buf(ctx, &x);

    let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let q3_wlen = q3_w.len();
    let q4_wlen = q4_w.len();

    eprintln!("\n=== shape {tag}: rows={rows} cols={cols} ({blocks} blocks/output) ===");

    // 1. Q3_K fused: 110 B weights, no scale table.
    let us_q3_fused = time_dispatch("Q3_K fused", |tcb| {
        kernels::gemv_q3_k_pinned_tcb(tcb, &q3_buf, 0, q3_wlen, rows, cols, &x_buf, &y_buf)
            .expect("q3_k fused encode");
    });

    // 1b. Q3_K fused 2r: 110 B weights, no scale table, 2-row ILP (16 rows/TG).
    let us_q3_fused_2r = time_dispatch("Q3_K fused 2r", |tcb| {
        kernels::gemv_q3_k_fused_2r_pinned_tcb(
            tcb, &q3_buf, 0, q3_wlen, rows, cols, &x_buf, &y_buf,
        )
        .expect("q3_k fused 2r encode");
    });

    // 2. Q3_K predec: 110 B weights + 16 f32 scales.
    let us_q3_predec = time_dispatch("Q3_K predec", |tcb| {
        kernels::gemv_q3_k_v4_predec_pinned_tcb(
            tcb,
            &q3_buf,
            0,
            q3_wlen,
            &q3_scales_buf,
            0,
            rows,
            cols,
            &x_buf,
            &y_buf,
        )
        .expect("q3_k predec encode");
    });

    // 3. Q4_K predec: 144 B weights + 16 f32 scales.
    let us_q4_predec = time_dispatch("Q4_K predec", |tcb| {
        kernels::gemv_q4_k_v4_predec_pinned_tcb(
            tcb,
            &q4_buf,
            0,
            q4_wlen,
            &q4_scales_buf,
            0,
            rows,
            cols,
            &x_buf,
            &y_buf,
        )
        .expect("q4_k predec encode");
    });

    // --- Explicit bytes/block read (the bandwidth-bound terms) ---
    // Weights + scale table dominate. x (cols*4) and y (rows*4) are constant
    // across all three kernels (same shape), included for honest GB/s.
    let x_bytes = (cols * 4) as f64;
    let y_bytes = (rows * 4) as f64;
    let scale_f32_per_block = 16 * 4; // 16 f32 pre-decoded scales = 64 B

    let bpb_q3_fused = 110.0; // weights only (fused + fused_2r read identical bytes)
    let bpb_q3_predec = 110.0 + scale_f32_per_block as f64; // 174 B
    let bpb_q4_predec = 144.0 + scale_f32_per_block as f64; // 208 B

    let bytes_q3_fused = blocks as f64 * bpb_q3_fused + x_bytes + y_bytes;
    let bytes_q3_predec = blocks as f64 * bpb_q3_predec + x_bytes + y_bytes;
    let bytes_q4_predec = blocks as f64 * bpb_q4_predec + x_bytes + y_bytes;

    eprintln!(
        "  bytes/block (weights+scales): Q3 fused={:.0}  Q3 fused_2r={:.0}  Q3 predec={:.0}  Q4 predec={:.0}",
        bpb_q3_fused, bpb_q3_fused, bpb_q3_predec, bpb_q4_predec
    );
    eprintln!(
        "  total KiB/call:               Q3 fused={:.0}  Q3 fused_2r={:.0}  Q3 predec={:.0}  Q4 predec={:.0}",
        bytes_q3_fused / 1024.0,
        bytes_q3_fused / 1024.0,
        bytes_q3_predec / 1024.0,
        bytes_q4_predec / 1024.0
    );
    eprintln!(
        "  GB/s:                         Q3 fused={:.1}  Q3 fused_2r={:.1}  Q3 predec={:.1}  Q4 predec={:.1}",
        gbps(bytes_q3_fused, us_q3_fused),
        gbps(bytes_q3_fused, us_q3_fused_2r),
        gbps(bytes_q3_predec, us_q3_predec),
        gbps(bytes_q4_predec, us_q4_predec)
    );

    // Verdict (a): fastest Q3_K kernel of the three.
    let mut q3_winner = "fused";
    let mut q3_us = us_q3_fused;
    if us_q3_fused_2r < q3_us {
        q3_winner = "fused_2r";
        q3_us = us_q3_fused_2r;
    }
    if us_q3_predec < q3_us {
        q3_winner = "predec";
        q3_us = us_q3_predec;
    }
    // Intra-Q3: how much faster is fused_2r than the old fused_v2.
    let r2_vs_fused = (us_q3_fused - us_q3_fused_2r) / us_q3_fused * 100.0; // + => 2r faster
                                                                            // Verdict (b): does the fastest Q3_K beat Q4_K predec (the byte-cut premise)?
    let bytecut = (us_q4_predec - q3_us) / us_q4_predec * 100.0; // + => Q3 faster (byte-cut holds)
                                                                 // Verdict (c): does fused_2r alone beat Q4_predec (the byte-cut-TRUE kernel)?
    let r2_vs_q4 = (us_q4_predec - us_q3_fused_2r) / us_q4_predec * 100.0; // + => Q3 fused_2r faster

    eprintln!(
        "  >>> {tag}: Q3_fused={us_q3_fused:.3}  Q3_fused_2r={us_q3_fused_2r:.3}  Q3_predec={us_q3_predec:.3}  Q4_predec={us_q4_predec:.3} µs"
    );
    eprintln!(
        "  >>> (a) fastest Q3_K = {q3_winner} ({q3_us:.3} µs); fused_2r is {r2_vs_fused:+.2}% vs fused_v2 (+ => 2r faster)"
    );
    eprintln!(
        "  >>> (b) byte-cut: fastest Q3_K ({q3_winner}) vs Q4_predec = {bytecut:+.2}% (+ => Q3 faster = byte-cut speed holds)"
    );
    eprintln!(
        "  >>> (c) byte-cut-TRUE kernel: Q3 fused_2r vs Q4_predec = {r2_vs_q4:+.2}% (+ => fewest-byte Q3 wins on speed)"
    );
}

#[test]
#[ignore = "microbench — run with --ignored --nocapture; needs a free GPU"]
fn q3k_bytecut_gemv_bench() {
    eprintln!(
        "[q3k_bytecut_bench] Q3_K fused vs Q3_K predec vs Q4_K predec GEMV, {ITERS} iters/shape after {WARMUP} warmup"
    );
    eprintln!(
        "  Q3_K byte-cut premise: fewer weight bytes (110 vs 144) => faster on BW-bound decode.\n  Subtlety: Q3_K predec ADDS 64 B f32 scales/block (anti-byte-cut); fused reads no scale table."
    );
    // Representative Qwen2.5-3B decode GEMV shapes.
    bench_shape(2048, 2048, "attn-square 2048x2048");
    bench_shape(11008, 2048, "ffn-up 11008x2048");
    bench_shape(2048, 11008, "ffn-down 2048x11008");
}
