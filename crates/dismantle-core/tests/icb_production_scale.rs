//! ICB at production scale (235 dispatches/token, varied per-dispatch args).
//!
//! Companion to `icb_overhead_microbench.rs` (the POC, which fired 500
//! identical-args dispatches and reported a 76.4% per-dispatch reduction).
//! This test repeats the comparison at production cadence — 235 dispatches
//! per "token" — and uses VARYING per-dispatch arguments (rotating buffer
//! pointers, varying n) so per-dispatch argument-binding work is realistic.
//!
//! The single question this test answers:
//!   Does ICB hold the POC's 76.4% per-dispatch saving when the binding
//!   work per command is non-trivial?
//!
//! Output: a decision-matrix bucket and a projected decode-tps using the
//! formula from `memory/decode_gap_anatomy_2026_05_24.md`:
//!   projected_decode_ms = 14.2 + (1 - reduction) × 34.6
//!   projected_dec_tps   = 21.0 × 48.8 / projected_decode_ms
//!
//! The test always passes (it's a measurement). Parse the printed report
//! into the memo.
//!
//! Run:
//!   cargo test --release -p dismantle-core --test icb_production_scale -- --nocapture

#![cfg(target_os = "macos")]

use std::time::Instant;

use dismantle_core::metal::MetalContext;
use dismantle_core::metal::icb::IndirectTokenCommandBuffer;

use metal::{
    ComputePipelineDescriptor, ComputePipelineState, MTLResourceUsage, MTLSize,
};

const N_DISPATCHES: usize = 235;
const N_BUF_POOL: usize = 8;
const TRIALS_WARMUP: usize = 20;
const TRIALS_MEASURE: usize = 100;

/// Production Qwen-3B decode constants (from
/// `memory/decode_gap_anatomy_2026_05_24.md`).
const DECODE_WALL_MS_BASELINE: f64 = 48.8;
const GPU_MS: f64 = 14.2;
const GAP_MS: f64 = 34.6;
const DEC_TPS_BASELINE: f64 = 21.0;

/// Build an ICB-capable PSO for `add_inplace`. The production PSO from
/// `MetalContext::pipeline()` lacks `support_indirect_command_buffers`
/// and crashes the GPU when bound to an indirect command.
fn icb_pipeline(ctx: &MetalContext, fn_name: &str) -> ComputePipelineState {
    let f = ctx
        .library()
        .get_function(fn_name, None)
        .expect("function lookup");
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&f));
    desc.set_support_indirect_command_buffers(true);
    ctx.device()
        .new_compute_pipeline_state(&desc)
        .expect("ICB-capable PSO")
}

/// Build the per-dispatch parameter table. For each of N_DISPATCHES slots
/// we pick:
///   - which a-buffer (rotating 0..N_BUF_POOL)
///   - which b-buffer (rotating, offset by 1 so the pair is distinct)
///   - which n_buf — n varies per dispatch
///
/// Varying these three values means every dispatch's recorded args are
/// distinct → each `set_kernel_buffer` call hits a different binding,
/// matching the production pattern of "every dispatch binds different
/// activation tensors".
fn build_dispatch_plan() -> Vec<(usize, usize, usize, u32)> {
    // n values mimicking dismantle's mix (hidden=2048, hidden×ffn_mul=11008,
    // kv_dim variants, head*dim variants). Cycled deterministically.
    let n_choices: &[u32] = &[2048, 4096, 11008, 1024, 2048, 8192, 2048, 11008];
    let mut plan = Vec::with_capacity(N_DISPATCHES);
    for i in 0..N_DISPATCHES {
        let a_idx = i % N_BUF_POOL;
        let b_idx = (i + 1) % N_BUF_POOL;
        let n_idx = i % n_choices.len();
        let n = n_choices[n_idx];
        plan.push((a_idx, b_idx, n_idx, n));
    }
    plan
}

struct BufferPool {
    a_bufs: Vec<metal::Buffer>,
    b_bufs: Vec<metal::Buffer>,
    /// One n-buffer per distinct n value (8). Each holds a single u32.
    n_bufs: Vec<metal::Buffer>,
}

impl BufferPool {
    fn new(ctx: &MetalContext, n_choices: &[u32]) -> Self {
        // Each a/b buffer is sized to the maximum n. Real production
        // tensors are this large too.
        let max_n = *n_choices.iter().max().unwrap() as usize;
        let mut a_bufs = Vec::with_capacity(N_BUF_POOL);
        let mut b_bufs = Vec::with_capacity(N_BUF_POOL);
        for _ in 0..N_BUF_POOL {
            a_bufs.push(ctx.new_buffer(max_n * std::mem::size_of::<f32>()));
            let b = ctx.new_buffer(max_n * std::mem::size_of::<f32>());
            let b_init = vec![1.0f32; max_n];
            MetalContext::write_buffer_bytes(&b, bytemuck::cast_slice(&b_init));
            b_bufs.push(b);
        }
        let mut n_bufs = Vec::with_capacity(n_choices.len());
        for &n in n_choices {
            n_bufs.push(ctx.new_buffer_with_bytes(bytemuck::cast_slice(&[n])));
        }
        Self {
            a_bufs,
            b_bufs,
            n_bufs,
        }
    }
}

/// Production-style: one CB, per-dispatch encoder (matches
/// `metal/mod.rs::dispatch_batch`). This is the baseline we compare ICB
/// against.
fn run_tcb_per_dispatch_encoder(
    ctx: &MetalContext,
    pipe: &ComputePipelineState,
    pool: &BufferPool,
    plan: &[(usize, usize, usize, u32)],
) -> u128 {
    let queue = ctx.queue();
    let t0 = Instant::now();
    let cmd = queue.new_command_buffer();
    let tg = MTLSize::new(64, 1, 1);
    for &(a_idx, b_idx, n_idx, n) in plan {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(&pool.a_bufs[a_idx]), 0);
        enc.set_buffer(1, Some(&pool.b_bufs[b_idx]), 0);
        enc.set_buffer(2, Some(&pool.n_bufs[n_idx]), 0);
        let grid = MTLSize::new(n as u64, 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();
    t0.elapsed().as_nanos()
}

/// ICB path: record all 235 dispatches with VARYING args into the ICB,
/// then submit once.
fn run_icb(
    ctx: &MetalContext,
    pool: &BufferPool,
    plan: &[(usize, usize, usize, u32)],
) -> (u128, u128) {
    let t_build = Instant::now();
    let mut icb = IndirectTokenCommandBuffer::new(ctx, N_DISPATCHES).expect("ICB alloc");
    let tg = (64u32, 1u32, 1u32);
    for &(a_idx, b_idx, n_idx, n) in plan {
        let a = pool.a_bufs[a_idx].clone();
        let b = pool.b_bufs[b_idx].clone();
        let n_buf = pool.n_bufs[n_idx].clone();
        let grid = (n, 1u32, 1u32);
        icb.dispatch_threads("add_inplace", grid, tg, move |cmd| {
            cmd.set_kernel_buffer(0, Some(&a), 0);
            cmd.set_kernel_buffer(1, Some(&b), 0);
            cmd.set_kernel_buffer(2, Some(&n_buf), 0);
        })
        .expect("ICB dispatch record");
    }
    // ICB must declare each buffer it touches via `use_resource`.
    for a in &pool.a_bufs {
        icb.mark_resource_used(a, MTLResourceUsage::Read | MTLResourceUsage::Write);
    }
    for b in &pool.b_bufs {
        icb.mark_resource_used(b, MTLResourceUsage::Read);
    }
    for n in &pool.n_bufs {
        icb.mark_resource_used(n, MTLResourceUsage::Read);
    }
    let build_ns = t_build.elapsed().as_nanos();

    let t_exec = Instant::now();
    icb.execute_and_wait().expect("ICB execute");
    let exec_ns = t_exec.elapsed().as_nanos();
    (build_ns, exec_ns)
}

fn percentile(sorted: &[u128], p: f64) -> u128 {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn median(sorted: &[u128]) -> u128 {
    sorted[sorted.len() / 2]
}

fn mean(samples: &[u128]) -> u128 {
    let sum: u128 = samples.iter().sum();
    sum / samples.len() as u128
}

fn fmt_us(ns: u128) -> String {
    format!("{:.2} us", ns as f64 / 1_000.0)
}

fn fmt_ms(ns: u128) -> String {
    format!("{:.3} ms", ns as f64 / 1_000_000.0)
}

#[test]
fn icb_at_production_scale() {
    let ctx = MetalContext::new().expect("metal device");
    let pipe = icb_pipeline(&ctx, "add_inplace");
    let n_choices: Vec<u32> = vec![2048, 4096, 11008, 1024, 2048, 8192, 2048, 11008];
    let pool = BufferPool::new(&ctx, &n_choices);
    let plan = build_dispatch_plan();

    println!("=== ICB production-scale microbench ===");
    println!(
        "dispatches/run: {} (matches Qwen-3B decode)",
        N_DISPATCHES
    );
    println!(
        "varied args: rotating among {} buffer pairs × {} n values",
        N_BUF_POOL,
        n_choices.len()
    );
    println!("warmup runs: {}, measured runs: {} each", TRIALS_WARMUP, TRIALS_MEASURE);
    println!();

    // Warm-up — first dispatches incur PSO compile + driver init + JIT.
    for _ in 0..TRIALS_WARMUP {
        let _ = run_tcb_per_dispatch_encoder(&ctx, &pipe, &pool, &plan);
        let _ = run_icb(&ctx, &pool, &plan);
    }

    let mut tcb_times: Vec<u128> = Vec::with_capacity(TRIALS_MEASURE);
    let mut icb_exec_times: Vec<u128> = Vec::with_capacity(TRIALS_MEASURE);
    let mut icb_build_times: Vec<u128> = Vec::with_capacity(TRIALS_MEASURE);

    for _ in 0..TRIALS_MEASURE {
        let t_tcb = run_tcb_per_dispatch_encoder(&ctx, &pipe, &pool, &plan);
        let (t_build, t_icb) = run_icb(&ctx, &pool, &plan);
        tcb_times.push(t_tcb);
        icb_exec_times.push(t_icb);
        icb_build_times.push(t_build);
    }
    tcb_times.sort();
    icb_exec_times.sort();
    icb_build_times.sort();

    let tcb_med = median(&tcb_times);
    let tcb_p99 = percentile(&tcb_times, 0.99);
    let tcb_mean = mean(&tcb_times);
    let icb_med = median(&icb_exec_times);
    let icb_p99 = percentile(&icb_exec_times, 0.99);
    let icb_mean = mean(&icb_exec_times);
    let build_med = median(&icb_build_times);

    let tcb_per = tcb_med as f64 / N_DISPATCHES as f64;
    let icb_per = icb_med as f64 / N_DISPATCHES as f64;
    let reduction_pct = 100.0 * (1.0 - icb_per / tcb_per);

    println!("--- TCB (per-dispatch encoder, current production path) ---");
    println!(
        "  wall: median={} p99={} mean={}",
        fmt_us(tcb_med),
        fmt_us(tcb_p99),
        fmt_us(tcb_mean)
    );
    println!("  per-dispatch: {:.2} us", tcb_per / 1_000.0);
    println!();
    println!("--- ICB (single execute_commands_in_buffer dispatch) ---");
    println!(
        "  wall: median={} p99={} mean={}",
        fmt_us(icb_med),
        fmt_us(icb_p99),
        fmt_us(icb_mean)
    );
    println!("  per-dispatch: {:.2} us", icb_per / 1_000.0);
    println!("  one-time build cost: {} (amortized at production cadence)", fmt_us(build_med));
    println!();
    println!("--- ICB per-dispatch reduction vs TCB ---");
    println!("  {:+.2}%", reduction_pct);
    println!();

    // Projected decode tps from the matrix formula.
    let r = reduction_pct.max(0.0) / 100.0;
    let projected_decode_ms = GPU_MS + (1.0 - r) * GAP_MS;
    let projected_dec_tps = DEC_TPS_BASELINE * DECODE_WALL_MS_BASELINE / projected_decode_ms;
    println!("--- Projected decode tps (matrix-literal) ---");
    println!(
        "  projected_decode_ms = {:.2} + (1 - {:.4}) × {:.2} = {:.2} ms/token",
        GPU_MS, r, GAP_MS, projected_decode_ms
    );
    println!(
        "  projected_dec_tps = {:.1} × {:.1} / {:.2} = {:.2} dec_tps",
        DEC_TPS_BASELINE, DECODE_WALL_MS_BASELINE, projected_decode_ms, projected_dec_tps
    );
    println!();

    let bucket = if reduction_pct >= 65.0 {
        "GO_FULL_WIRE_UP"
    } else if reduction_pct >= 40.0 {
        "GO_WITH_TWO_STAGE"
    } else if reduction_pct >= 10.0 {
        "DEMOTE_TO_SECONDARY"
    } else {
        "BLOCKED_OR_DEAD"
    };
    println!("--- Decision matrix bucket: {} ---", bucket);
    println!(
        "  (>=65% GO_FULL_WIRE_UP, 40-64% GO_WITH_TWO_STAGE, 10-39% DEMOTE, <10% DEAD)"
    );

    // Comparison to POC (76.4% on N=500 identical args).
    println!();
    println!("--- POC (identical args, N=500, add_inplace n=64) ---");
    println!("  reduction: 76.4%");
    println!("--- Production scale (this test, N=235 varied args) ---");
    println!("  reduction: {:+.2}%", reduction_pct);
    println!("  delta: {:+.2}pp", reduction_pct - 76.4);
    let _ = (tcb_mean, icb_mean, build_med, fmt_ms(tcb_med));
}
