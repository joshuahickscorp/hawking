//! DIAGNOSTIC (path-to-50, 2026-05-29): isolate the in-TCB marginal
//! per-dispatch cost from the command-buffer round-trip latency.
//!
//! `bench-kernel` commits+waits one command buffer per dispatch, so its
//! ~130 us "floor" is the CB round-trip, not the kernel. Production decode
//! batches many dispatches into ONE TokenCommandBuffer with a single
//! commit_and_wait. This test dispatches the SAME Q4_K GEMV K times into one
//! TCB and times the whole commit, for K in {1,2,4,...,256}. The slope of
//! total-vs-K is the true in-TCB per-dispatch cost; the intercept is the CB
//! round-trip. That slope sets the ceiling for any dispatch-fusion lever.
//!
//! Run: cargo test --release -p hawking-core --test tcb_dispatch_cost -- --nocapture

use hawking_core::metal::{MetalContext, TokenCommandBuffer};
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Time M iterations of {new TCB; K dispatches of gemm_q4_k_m_fused_v2; commit_and_wait}.
/// Returns median total microseconds per iteration.
///
/// `distinct` weight buffers are cycled across the K dispatches. With
/// distinct=1 every dispatch re-reads the same matrix (L2-cache-hit, like a
/// microbench). With distinct=32 the working set (32 x matrix bytes) exceeds
/// L2 so every dispatch in a window reads a COLD matrix from DRAM — the
/// production decode regime (each layer's q/k/v/o/gate/up/down differs).
fn bench_k(
    ctx: &MetalContext,
    rows: usize,
    cols: usize,
    k: usize,
    iters: usize,
    distinct: usize,
) -> f64 {
    let blocks_per_row = cols / 256;
    let w_bytes = rows * blocks_per_row * 144;
    let x_bytes = cols * std::mem::size_of::<f32>();
    let out_bytes = rows * std::mem::size_of::<f32>();

    let w_bufs: Vec<_> = (0..distinct.max(1))
        .map(|_| ctx.new_buffer(w_bytes))
        .collect();
    let x_buf = ctx.new_buffer(x_bytes);
    let out_buf = ctx.new_buffer(out_bytes);

    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    const TG: u32 = 256;
    let n_tg = (rows as u32 + 7) / 8;
    let grid = (n_tg * TG, 1, 1);
    let tg = (TG, 1, 1);

    let one_tcb = |ctx: &MetalContext| {
        let mut tcb = TokenCommandBuffer::new(ctx);
        for i in 0..k {
            let w_buf = &w_bufs[i % w_bufs.len()];
            tcb.dispatch_threads("gemm_q4_k_m_fused_v2", grid, tg, |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
            })
            .unwrap();
        }
        tcb.commit_and_wait().unwrap();
    };

    // warmup
    for _ in 0..20 {
        one_tcb(ctx);
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        one_tcb(ctx);
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    median(samples)
}

#[test]
fn tcb_dispatch_cost_curve() {
    let ctx = MetalContext::new().expect("Metal device required");
    let ks = [1usize, 2, 4, 8, 16, 32, 64, 128, 256];
    // 2048x2048 = q/o proj shape; 11008x2048 = ffn gate/up shape.
    for (rows, cols, label) in [
        (2048usize, 2048usize, "q/o 2048x2048"),
        (11008, 2048, "ffn 11008x2048"),
    ] {
        for distinct in [1usize, 32usize] {
            let regime = if distinct == 1 {
                "SAME buf (L2 cache-hit)"
            } else {
                "32 distinct bufs (cold DRAM, = decode)"
            };
            println!("\n=== {label} — {regime} — K dispatches in ONE TCB ===");
            println!(
                "{:>5}  {:>12}  {:>14}  {:>16}",
                "K", "total_us", "us/dispatch", "marginal_us/disp"
            );
            let base = bench_k(&ctx, rows, cols, 1, 300, distinct);
            for &k in &ks {
                let total = bench_k(
                    &ctx,
                    rows,
                    cols,
                    k,
                    if k <= 16 { 300 } else { 150 },
                    distinct,
                );
                let per = total / k as f64;
                let marginal = if k > 1 {
                    (total - base) / (k as f64 - 1.0)
                } else {
                    total
                };
                println!("{k:>5}  {total:>12.1}  {per:>14.1}  {marginal:>16.1}");
            }
        }
    }
}
