//! `gpu_us` measurement-accuracy diagnostic.
//!
//! The hypothesis under test (from
//! `memory/pso_transition_dead_2026_05_24.md`):
//!
//!   The 36 ms/token "gap" in the decode trace is a measurement artifact.
//!   Apple Metal's counter-sample-buffer (CSB) stage-boundary sampling
//!   under-reports per-dispatch GPU duration (per `temperkit_salvage.md`,
//!   coverage is documented at 45–200%). If true, `gpu_us` collected in
//!   `ProdCbGpu` mode is wrong by ~3× and there is no gap to close.
//!
//! How this test answers it:
//!
//!   For each of three per-dispatch workload sizes (tiny / medium / large),
//!   we submit N=200 dispatches of the same `add_inplace` kernel in ONE
//!   `TokenCommandBuffer`, host-time the entire `commit_and_wait`, and
//!   compare against the sum of per-dispatch `gpu_us` values reported via
//!   the `ProdCbGpu` CSB tracer (already enabled in production).
//!
//!   Apple's `GPUEndTime − GPUStartTime` is the ground-truth wall-clock for
//!   GPU activity on that command buffer; we approximate it host-side by
//!   timing around `commit_and_wait` (the difference between true cb-wall
//!   and host-timed cb-wall is sub-100 μs CPU sync overhead, negligible
//!   when sum_gpu_us is in the millisecond range).
//!
//! Decision:
//!   - host_wall ≈ sum_gpu_us (ratio ~1.0): `gpu_us` is accurate. The 36 ms
//!     gap in production decode is real and only Instruments can find it.
//!   - host_wall ≫ sum_gpu_us (ratio ≥ 2): `gpu_us` undercounts. The "gap"
//!     framework collapses; the lever is the Q4_K kernel itself, not
//!     dispatch orchestration.

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use std::time::Instant;

fn make_f32_buf(ctx: &MetalContext, n: usize, seed: u64) -> PinnedBuffer {
    let mut rng = Pcg64Mcg::new(seed as u128);
    let data: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(&data))
}

fn run_n_dispatches(
    ctx: &MetalContext,
    a: &PinnedBuffer,
    b: &PinnedBuffer,
    n_elems: usize,
    n_dispatches: usize,
) -> (u128, u128) {
    // Returns (host_wall_us_around_commit, sum_gpu_us_from_csb).
    let mut tcb = TokenCommandBuffer::new(ctx);
    for _ in 0..n_dispatches {
        kernels::add_inplace_metal_tcb(&mut tcb, a, b, n_elems).unwrap();
    }
    // Drain any prior trace samples so this run is the only one we see.
    let _stale = ctx.drain_trace();
    let t0 = Instant::now();
    tcb.commit_and_wait().unwrap();
    let host_wall_us = t0.elapsed().as_micros();
    // After the CB completes the ProdCbGpu tracer pushes per-dispatch
    // samples into ctx.trace.samples; drain them.
    let samples = ctx.drain_trace();
    let sum_gpu_us: u128 = samples
        .iter()
        .filter(|s| s.kernel_name == "add_inplace")
        .map(|s| s.gpu_us.unwrap_or(0) as u128)
        .sum();
    (host_wall_us, sum_gpu_us)
}

#[test]
fn gpu_us_accuracy() {
    // CRITICAL: must set BEFORE MetalContext::new() so the TCB inherits
    // ProdCbGpu mode at construction.
    std::env::set_var("DISMANTLE_TCB_TRACE", "gpu_prod");

    let ctx = MetalContext::new().expect("Metal device required");

    let workloads = [
        ("tiny  (n=64)", 64usize),
        ("med   (n=2M)", 2_000_000usize),
        ("large (n=8M)", 8_000_000usize),
    ];

    let warmup = 5;
    let trials = 10;
    let n_dispatches = 200;

    eprintln!("\n=== gpu_us measurement accuracy ===");
    eprintln!("N dispatches per command buffer: {}", n_dispatches);
    eprintln!("Each measurement: warmup={}, trials={} (median reported)\n", warmup, trials);
    eprintln!(
        "{:<14} {:>14} {:>14} {:>10}  {}",
        "workload", "host_wall_us", "sum_gpu_us", "ratio", "interpretation"
    );

    for (label, n_elems) in workloads {
        // allocate once per workload size (avoid alloc cost contaminating)
        let a = make_f32_buf(&ctx, n_elems, 0xAA);
        let b = make_f32_buf(&ctx, n_elems, 0xBB);

        // warmup
        for _ in 0..warmup {
            let _ = run_n_dispatches(&ctx, &a, &b, n_elems, n_dispatches);
        }
        // trials
        let mut host_vals: Vec<u128> = Vec::with_capacity(trials);
        let mut gpu_sums: Vec<u128> = Vec::with_capacity(trials);
        for _ in 0..trials {
            let (h, g) = run_n_dispatches(&ctx, &a, &b, n_elems, n_dispatches);
            host_vals.push(h);
            gpu_sums.push(g);
        }
        host_vals.sort();
        gpu_sums.sort();
        let host_med = host_vals[trials / 2];
        let gpu_med = gpu_sums[trials / 2];
        let ratio = host_med as f64 / gpu_med.max(1) as f64;

        let interp = if ratio < 1.3 {
            "gpu_us ACCURATE (≤30% overhead, consistent with sync cost)"
        } else if ratio < 2.0 {
            "gpu_us PARTIAL undercount (30–100% extra time unaccounted)"
        } else if ratio < 4.0 {
            "gpu_us UNDERCOUNTS (~2-4× — matches temperkit-salvage warning)"
        } else {
            "gpu_us SEVERELY UNDERCOUNTS (>4× — definitive)"
        };

        eprintln!(
            "{:<14} {:>14} {:>14} {:>10.2}  {}",
            label, host_med, gpu_med, ratio, interp
        );
    }

    eprintln!("\n=== Interpretation guide ===");
    eprintln!("If ratio ≥ 2.0 at large workload (where sync overhead is < 1%):");
    eprintln!("  → CSB systematically undercounts. The production trace's");
    eprintln!("    71%% 'gap' is partly or wholly measurement artifact.");
    eprintln!("  → Real per-token GPU compute is closer to decode_wall;");
    eprintln!("    the next lever is the Q4_K kernel itself (Lever 0).");
    eprintln!("");
    eprintln!("If ratio ≈ 1.0 at large workload:");
    eprintln!("  → CSB is accurate. The 36 ms/token gap is real, not artifact.");
    eprintln!("  → Code-only investigation is at its limit. Next swing must");
    eprintln!("    be an attended Instruments session.");

    // Always pass; informational test only.
}
