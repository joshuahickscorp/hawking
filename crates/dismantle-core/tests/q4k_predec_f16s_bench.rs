//! q4k_predec_f16s_bench — early bandwidth signal for the f16-scales predec
//! GEMV (Stage-2 bandwidth lever 1.2). The f16 scale table shrinks the
//! per-block pre-decoded scales from 16 f32 (64 B) to 16 f16 (32 B); on the
//! bandwidth-bound decode GEMV the predec block footprint drops 192 B → 160 B
//! (144 B Q4_K weights + scales), a ~16.7% read-traffic cut on the scale-heavy
//! path. This times the f32-scales production wrapper
//! (`gemv_q4_k_v4_predec_pinned_tcb`, default 2r) vs the f16-scales variant
//! (`gemv_q4_k_v4_predec_2r_f16s_pinned_tcb`) on representative Qwen2.5-3B
//! decode shapes and reports µs/call + achieved GB/s + the f16s speedup.
//!
//! NOT a correctness gate (parity lives in q4k_predec_f16s_parity.rs). This is
//! a microbench: it ignores numerical output and only measures dispatch wall
//! time. GPU-gated. Marked #[ignore] so it never runs in the default suite —
//! run explicitly with `--ignored --nocapture`.

#![cfg(target_os = "macos")]

use dismantle_core::kernels::{self, predecode_q4_k_scale_table_f16};
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use half::f16;
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use std::time::Instant;

mod common;
use common::*;

/// Realistic Q4_K weights (144 B/block) — identical generator to the parity test.
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
    (0..cols).map(|_| rng.gen_range(-3.0_f32..3.0_f32)).collect()
}

/// Pin a Vec<f16> as raw little-endian bytes — the f16s kernel reads buffer(1)
/// as `device const half*`. Matches new_f16_buf in the parity test.
fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
    let bytes: Vec<u8> = data.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect();
    ctx.new_buffer_with_bytes(&bytes)
}

const WARMUP: usize = 30;
const ITERS: usize = 200;

/// Time one predec dispatch `ITERS` times after `WARMUP`, each iteration a
/// fresh TCB committed-and-waited (so the wall time is one full GPU dispatch
/// round-trip). Returns mean µs/call.
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
    eprintln!("  [{label}] {us_per_call:.3} µs/call ({ITERS} iters)");
    us_per_call
}

fn bench_shape(rows: usize, cols: usize, tag: &str) {
    let ctx = ctx();
    let blocks = rows * (cols / 256);

    let w_bytes = make_q4k_bytes(rows, cols, 0xF165_8E1E ^ (rows as u64));
    let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
    let x = make_x(cols, 0xCAFE_F00D);
    let x_buf = new_f32_buf(ctx, &x);

    let scales_f32 = kernels::predecode_q4_k_scale_table(&w_bytes);
    let scales_f32_buf = new_f32_buf(ctx, &scales_f32);
    let scales_f16 = predecode_q4_k_scale_table_f16(&w_bytes);
    let scales_f16_buf = new_f16_buf(ctx, &scales_f16);

    let y_f32_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let y_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let wlen = w_bytes.len();

    eprintln!("\n=== shape {tag}: rows={rows} cols={cols} ({blocks} blocks/output) ===");

    // f32-scales: 144 B weights + 16 f32 scales (64 B) per block + x (4 B/col) + y.
    let us_f32 = time_dispatch("f32 scales (2r)", |tcb| {
        kernels::gemv_q4_k_v4_predec_pinned_tcb(
            tcb, &model_buf, 0, wlen, &scales_f32_buf, 0, rows, cols, &x_buf, &y_f32_buf,
        )
        .expect("f32 predec encode");
    });

    // f16-scales: 144 B weights + 16 f16 scales (32 B) per block + x + y.
    let us_f16 = time_dispatch("f16 scales (2r)", |tcb| {
        kernels::gemv_q4_k_v4_predec_2r_f16s_pinned_tcb(
            tcb, &model_buf, 0, wlen, &scales_f16_buf, 0, rows, cols, &x_buf, &y_f16_buf,
        )
        .expect("f16s predec encode");
    });

    // Bytes read per call (the bandwidth-bound terms). Weights + scale table
    // dominate; x (cols*4) and y (rows*4) are small but included for honesty.
    let weights = (blocks * 144) as f64;
    let x_bytes = (cols * 4) as f64;
    let y_bytes = (rows * 4) as f64;
    let bytes_f32 = weights + (blocks * 16 * 4) as f64 + x_bytes + y_bytes;
    let bytes_f16 = weights + (blocks * 16 * 2) as f64 + x_bytes + y_bytes;

    let gbps_f32 = bytes_f32 / (us_f32 * 1e3); // bytes / (µs*1e3 ns) -> GB/s
    let gbps_f16 = bytes_f16 / (us_f16 * 1e3);
    let speedup = (us_f32 - us_f16) / us_f32 * 100.0;

    eprintln!(
        "  bytes/call: f32={:.0} KiB  f16={:.0} KiB ({:.1}% less scale traffic)",
        bytes_f32 / 1024.0,
        bytes_f16 / 1024.0,
        (1.0 - bytes_f16 / bytes_f32) * 100.0
    );
    eprintln!("  GB/s:       f32={gbps_f32:.1}  f16={gbps_f16:.1}");
    eprintln!(
        "  >>> {tag}: f32={us_f32:.3} µs  f16={us_f16:.3} µs  speedup={speedup:+.2}%"
    );
}

#[test]
#[ignore = "microbench — run with --ignored --nocapture; needs a free GPU"]
fn q4k_predec_f16s_bandwidth_bench() {
    eprintln!(
        "[q4k_predec_f16s_bench] f32-scales vs f16-scales predec GEMV, {ITERS} iters/shape after {WARMUP} warmup"
    );
    // Representative Qwen2.5-3B decode GEMV shapes.
    bench_shape(2048, 2048, "attn-square 2048x2048");
    bench_shape(11008, 2048, "ffn-up 11008x2048");
    bench_shape(2048, 11008, "ffn-down 2048x11008");
}
