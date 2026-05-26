//! PSO-transition microbench — tests whether Pipeline State Object switches
//! between adjacent kernel dispatches are the dominant cause of the 36 ms/token
//! "gap" in `decode_gap_anatomy_2026_05_24.md`.
//!
//! Diagnostic context: the W4A8 production trace shows decode_wall=48.8 ms/token
//! with only 12.6 ms of measurable GPU kernel time. The remaining 36 ms is
//! unaccounted-for ("gap"). ICB POC + concurrent QKV both targeted host-side
//! per-dispatch overhead and delivered <3% e2e — so the gap is GPU-side, not
//! host-side. PSO transitions between dispatches are the next candidate.
//!
//! Test design:
//!   - Path A: 200 dispatches of the SAME kernel, identical args, same TCB.
//!             No PSO transitions at all.
//!   - Path B: 100 alternating pairs (kernel1, kernel2). 199 PSO transitions.
//!   - Path C: 200 dispatches cycling through 4 distinct kernels. 199 transitions
//!             across 4 PSO states (mimics decode's per-layer kernel diversity).
//!
//! All paths use tiny buffers (n=64 floats) so kernel-compute time is
//! negligible — what we measure is the dispatch overhead, dominated by PSO
//! transition cost when present.
//!
//! Decision criteria:
//!   - Path A << Path B/C: PSO transitions ARE the gap. Fix = kernel fusion at
//!     the PSO-count level, or scheduling to cluster same-kernel runs.
//!   - Path A ≈ Path B ≈ Path C: PSO transitions are NOT the gap. The 36 ms
//!     comes from somewhere else (driver scheduler, residency, undocumented
//!     barriers) that's not directly observable without Instruments.

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use std::time::Instant;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

fn new_f32_buf(ctx: &MetalContext, n: usize, seed: u64) -> PinnedBuffer {
    let mut rng = Pcg64Mcg::new(seed as u128);
    let data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(&data))
}

fn pct(arr: &[f64], p: f64) -> f64 {
    let mut sorted: Vec<f64> = arr.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() as f64) * p).min(sorted.len() as f64 - 1.0) as usize;
    sorted[idx]
}

#[test]
fn pso_transition_microbench() {
    let ctx = ctx();
    let n = 64usize;
    let f32b = std::mem::size_of::<f32>();

    // Buffers — all sized n=64 floats to make kernel compute time trivial.
    // add_inplace needs (a, b)
    let a_buf = new_f32_buf(ctx, n, 0x11);
    let b_buf = new_f32_buf(ctx, n, 0x22);

    // rmsnorm_f32 needs (x, weight, out) — but rmsnorm dispatches with a fixed
    // TG_SIZE=256 grid regardless of n, so we use n=256 for this kernel only.
    let n_rms = 256usize;
    let rms_x_buf = new_f32_buf(ctx, n_rms, 0x33);
    let rms_w_buf = new_f32_buf(ctx, n_rms, 0x44);
    let rms_out_buf = ctx.new_buffer(n_rms * f32b);

    // silu_mul needs (gate, up, out)
    let silu_gate_buf = new_f32_buf(ctx, n, 0x55);
    let silu_up_buf = new_f32_buf(ctx, n, 0x66);
    let silu_out_buf = ctx.new_buffer(n * f32b);

    // embed_lookup_f32 needs (embed, x) — embed is (vocab=64, hidden=64), small.
    // Note: shader expects half* embed in common.metal:411.
    let embed_buf_bytes = vec![0u8; n * n * 2]; // n×n half values, zeros are fine
    let embed_buf = ctx.new_buffer_with_bytes(&embed_buf_bytes);
    let embed_out_buf = ctx.new_buffer(n * f32b);

    let warmup = 20;
    let calls = 100;
    const N_DISPATCH: usize = 200;

    // ===== Path A: 200 identical dispatches (no PSO transitions) =====
    let mut path_a_times: Vec<f64> = Vec::with_capacity(calls);
    let run_a = |ctx: &MetalContext| {
        let mut tcb = TokenCommandBuffer::new(ctx);
        for _ in 0..N_DISPATCH {
            kernels::add_inplace_metal_tcb(&mut tcb, &a_buf, &b_buf, n).unwrap();
        }
        tcb.commit_and_wait().unwrap();
    };
    for _ in 0..warmup { run_a(ctx); }
    for _ in 0..calls {
        let t0 = Instant::now();
        run_a(ctx);
        path_a_times.push(t0.elapsed().as_micros() as f64);
    }

    // ===== Path B: 100 alternating pairs (199 PSO transitions, 2 distinct PSOs) =====
    let mut path_b_times: Vec<f64> = Vec::with_capacity(calls);
    let run_b = |ctx: &MetalContext| {
        let mut tcb = TokenCommandBuffer::new(ctx);
        for i in 0..N_DISPATCH {
            if i % 2 == 0 {
                kernels::add_inplace_metal_tcb(&mut tcb, &a_buf, &b_buf, n).unwrap();
            } else {
                // 1e-6 eps for stability; same kernel binding pattern as production.
                kernels::rmsnorm_metal_buf_tcb(
                    &mut tcb, &rms_x_buf, &rms_w_buf, 1e-6, n_rms, &rms_out_buf,
                ).unwrap();
            }
        }
        tcb.commit_and_wait().unwrap();
    };
    for _ in 0..warmup { run_b(ctx); }
    for _ in 0..calls {
        let t0 = Instant::now();
        run_b(ctx);
        path_b_times.push(t0.elapsed().as_micros() as f64);
    }

    // ===== Path C: 200 dispatches cycling 4 distinct kernels (199 transitions) =====
    let mut path_c_times: Vec<f64> = Vec::with_capacity(calls);
    let run_c = |ctx: &MetalContext| {
        let mut tcb = TokenCommandBuffer::new(ctx);
        for i in 0..N_DISPATCH {
            match i % 4 {
                0 => {
                    kernels::add_inplace_metal_tcb(&mut tcb, &a_buf, &b_buf, n).unwrap();
                }
                1 => {
                    kernels::rmsnorm_metal_buf_tcb(
                        &mut tcb, &rms_x_buf, &rms_w_buf, 1e-6, n_rms, &rms_out_buf,
                    ).unwrap();
                }
                2 => {
                    kernels::silu_mul_tcb(
                        &mut tcb, &silu_gate_buf, &silu_up_buf, &silu_out_buf, n,
                    ).unwrap();
                }
                _ => {
                    kernels::embed_lookup_metal_f32_tcb(
                        &mut tcb, &embed_buf, 0u32, n, &embed_out_buf,
                    ).unwrap();
                }
            }
        }
        tcb.commit_and_wait().unwrap();
    };
    for _ in 0..warmup { run_c(ctx); }
    for _ in 0..calls {
        let t0 = Instant::now();
        run_c(ctx);
        path_c_times.push(t0.elapsed().as_micros() as f64);
    }

    // ===== Report =====
    let a_med = pct(&path_a_times, 0.50);
    let a_p99 = pct(&path_a_times, 0.99);
    let a_mean = path_a_times.iter().sum::<f64>() / calls as f64;
    let b_med = pct(&path_b_times, 0.50);
    let b_p99 = pct(&path_b_times, 0.99);
    let b_mean = path_b_times.iter().sum::<f64>() / calls as f64;
    let c_med = pct(&path_c_times, 0.50);
    let c_p99 = pct(&path_c_times, 0.99);
    let c_mean = path_c_times.iter().sum::<f64>() / calls as f64;

    eprintln!("\n=== PSO transition microbench (n={} dispatches per path, {} trials) ===", N_DISPATCH, calls);
    eprintln!("{:<50} {:>10} {:>10} {:>10}", "path", "median_us", "mean_us", "p99_us");
    eprintln!("{:<50} {:>10.1} {:>10.1} {:>10.1}", "A: 200× identical (no PSO transitions)", a_med, a_mean, a_p99);
    eprintln!("{:<50} {:>10.1} {:>10.1} {:>10.1}", "B: 200× alternating (199 transitions, 2 PSOs)", b_med, b_mean, b_p99);
    eprintln!("{:<50} {:>10.1} {:>10.1} {:>10.1}", "C: 200× cycling 4 kernels (199 transitions)",   c_med, c_mean, c_p99);
    eprintln!("");
    eprintln!("Per-dispatch (total / 200):");
    eprintln!("  A: {:>6.2} us/dispatch", a_med / N_DISPATCH as f64);
    eprintln!("  B: {:>6.2} us/dispatch", b_med / N_DISPATCH as f64);
    eprintln!("  C: {:>6.2} us/dispatch", c_med / N_DISPATCH as f64);
    eprintln!("");
    eprintln!("Ratios (B/A, C/A):");
    eprintln!("  B/A = {:.2}× (each dispatch in B costs {:.0} us more than A)", b_med / a_med, (b_med - a_med) / N_DISPATCH as f64);
    eprintln!("  C/A = {:.2}× (each dispatch in C costs {:.0} us more than A)", c_med / a_med, (c_med - a_med) / N_DISPATCH as f64);

    // Decision
    eprintln!("\n=== DECISION ===");
    let ratio_b = b_med / a_med;
    let ratio_c = c_med / a_med;
    let max_ratio: f64 = ratio_b.max(ratio_c);
    if max_ratio >= 2.0 {
        eprintln!("VERDICT: PSO transitions ARE dominant. Path with transitions is {:.2}× slower per dispatch.", max_ratio);
        eprintln!("FIX direction: kernel fusion at the PSO-count level, OR cluster same-kernel runs.");
    } else if max_ratio < 1.3 {
        eprintln!("VERDICT: PSO transitions are NOT the dominant gap. Identical-vs-mixed dispatches differ by < 30%.");
        eprintln!("IMPLICATION: The 36 ms/token gap is from something neither host-side overhead NOR PSO transitions.");
        eprintln!("Remaining candidates: driver scheduler stalls, GPU residency, undocumented barriers.");
        eprintln!("Next step requires Instruments (Metal System Trace timeline).");
    } else {
        eprintln!("VERDICT: PSO transitions are A CONTRIBUTOR but not dominant. Ratio = {:.2}×.", max_ratio);
        eprintln!("Real gap source is a mix; PSO fusion would help partially but not fully close the gap.");
    }

    // Projection to production scale:
    // 36 ms/token gap, ~210 dispatches/token, ~180 distinct PSO transitions/token
    let added_per_dispatch_us = (b_med - a_med) / N_DISPATCH as f64;
    let projected_gap_explained_us = added_per_dispatch_us * 180.0;
    eprintln!("\n=== PROJECTION ===");
    eprintln!("If production decode has ~180 PSO transitions/token, and each costs {:.0} us extra,", added_per_dispatch_us);
    eprintln!("then PSO transitions would explain {:.1} ms of the 36 ms/token gap ({:.1}%).", projected_gap_explained_us / 1000.0, projected_gap_explained_us / 36000.0 * 100.0);

    // ALWAYS pass — informational test only.
}
