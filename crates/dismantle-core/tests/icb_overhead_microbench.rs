//! ICB per-dispatch overhead microbench (Session 2026-05-24).
//!
//! Quantify the per-dispatch saving available from Indirect Command
//! Buffers (ICB) vs the production Metal dispatch path used by
//! dismantle. The decode-gap-anatomy memo attributes ~96% of the
//! 34.6 ms/token GPU-idle gap to uniform per-dispatch driver
//! overhead (~147 µs/dispatch × 235 dispatches/token). ICB collapses
//! N dispatches into one encoded "execute" command, in principle
//! amortizing the driver per-dispatch cost across the batch.
//!
//! This is a POC — it does NOT touch production paths. It writes
//! to a brand-new test file and uses the existing `add_inplace`
//! kernel on a tiny buffer so the kernel itself contributes
//! negligibly to wall-clock.
//!
//! Run:
//!
//!   cargo test --release -p dismantle-core --test icb_overhead_microbench -- --nocapture
//!
//! The test always passes (it's a measurement, not an assertion).
//! Parse the printed table for the ratio.

#![cfg(target_os = "macos")]

use std::time::Instant;

use dismantle_core::metal::MetalContext;
use metal::{
    ComputePipelineDescriptor, ComputePipelineState, IndirectCommandBufferDescriptor,
    IndirectCommandBufferRef, MTLIndirectCommandType, MTLResourceOptions, MTLSize, NSRange,
};
use metal::objc::{msg_send, sel, sel_impl};

/// Build a compute PSO with `supportIndirectCommandBuffers = true`.
/// PSOs from `MetalContext::pipeline()` do NOT have this flag set, so
/// they crash with SIGSEGV when bound to an indirect compute command.
fn icb_capable_pipeline(
    ctx: &MetalContext,
    fn_name: &str,
) -> ComputePipelineState {
    let f = ctx
        .library()
        .get_function(fn_name, None)
        .expect("function lookup");
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&f));
    desc.set_support_indirect_command_buffers(true);
    ctx.device()
        .new_compute_pipeline_state(&desc)
        .expect("PSO with ICB support")
}

/// metal-rs 0.29 exposes `executeCommandsInBuffer:withRange:` on
/// `RenderCommandEncoderRef` but not on `ComputeCommandEncoderRef`. The
/// Apple API supports it on both; we hand-roll it here via msg_send,
/// matching the pattern used in `crates/dismantle-core/src/metal/mod.rs:107`.
unsafe fn compute_execute_commands_in_buffer(
    encoder: &metal::ComputeCommandEncoderRef,
    icb: &IndirectCommandBufferRef,
    range: NSRange,
) {
    let _: () = msg_send![encoder, executeCommandsInBuffer:icb withRange:range];
}

const N_DISPATCHES: usize = 500;
const KERNEL_N: u32 = 64;
const TRIALS: usize = 5;

fn make_buffers(ctx: &MetalContext) -> (metal::Buffer, metal::Buffer, metal::Buffer) {
    let a = ctx.new_buffer((KERNEL_N as usize) * std::mem::size_of::<f32>());
    let b = ctx.new_buffer((KERNEL_N as usize) * std::mem::size_of::<f32>());
    // Initialize `b` to ones so add_inplace does something visible
    // (but the GPU work is essentially free at n=64).
    let b_init = vec![1.0f32; KERNEL_N as usize];
    MetalContext::write_buffer_bytes(&b, bytemuck::cast_slice(&b_init));
    let n_buf =
        ctx.new_buffer_with_bytes(bytemuck::cast_slice(&[KERNEL_N]));
    (a, b, n_buf)
}

/// Path A1: one CB, ONE compute encoder reused for N dispatches.
/// Best case for the regular path — encoder lifecycle amortized.
fn path_a1_one_encoder(ctx: &MetalContext) -> u128 {
    let pipe = icb_capable_pipeline(ctx, "add_inplace");
    let (a, b, n_buf) = make_buffers(ctx);
    let queue = ctx.queue();

    let t0 = Instant::now();
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(0, Some(&a), 0);
    enc.set_buffer(1, Some(&b), 0);
    enc.set_buffer(2, Some(&n_buf), 0);
    let grid = MTLSize::new(KERNEL_N as u64, 1, 1);
    let tg = MTLSize::new(64, 1, 1);
    for _ in 0..N_DISPATCHES {
        enc.dispatch_threads(grid, tg);
    }
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    t0.elapsed().as_nanos()
}

/// Path A2: one CB, SEPARATE compute encoder per dispatch.
/// Matches the production dismantle pattern (each `dispatch_threads`
/// in `metal/mod.rs:dispatch_batch` calls `new_compute_command_encoder()`).
fn path_a2_per_dispatch_encoder(ctx: &MetalContext) -> u128 {
    let pipe = icb_capable_pipeline(ctx, "add_inplace");
    let (a, b, n_buf) = make_buffers(ctx);
    let queue = ctx.queue();

    let t0 = Instant::now();
    let cmd = queue.new_command_buffer();
    let grid = MTLSize::new(KERNEL_N as u64, 1, 1);
    let tg = MTLSize::new(64, 1, 1);
    for _ in 0..N_DISPATCHES {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&a), 0);
        enc.set_buffer(1, Some(&b), 0);
        enc.set_buffer(2, Some(&n_buf), 0);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();
    t0.elapsed().as_nanos()
}

/// Path B: ICB with N recorded compute commands, executed in one CB.
/// Returns (build_ns, dispatch_ns) — dispatch_ns is the comparable
/// number for the per-dispatch overhead question; build_ns is one-time
/// amortizable cost in any production wire-up.
fn path_b_icb(ctx: &MetalContext) -> Option<(u128, u128)> {
    let pipe = icb_capable_pipeline(ctx, "add_inplace");
    let (a, b, n_buf) = make_buffers(ctx);
    let queue = ctx.queue();
    let device = ctx.device();

    // Build ICB.
    let desc = IndirectCommandBufferDescriptor::new();
    desc.set_command_types(MTLIndirectCommandType::ConcurrentDispatchThreads);
    desc.set_inherit_buffers(false);
    desc.set_inherit_pipeline_state(false);
    desc.set_max_kernel_buffer_bind_count(3);

    let t_build = Instant::now();
    let icb = device.new_indirect_command_buffer_with_descriptor(
        &desc,
        N_DISPATCHES as u64,
        MTLResourceOptions::StorageModeShared,
    );
    if icb.size() == 0 {
        eprintln!("[ICB] new_indirect_command_buffer_with_descriptor returned 0-size — unsupported on this device?");
        return None;
    }
    let grid = MTLSize::new(KERNEL_N as u64, 1, 1);
    let tg = MTLSize::new(64, 1, 1);
    for i in 0..N_DISPATCHES {
        let cmd = icb.indirect_compute_command_at_index(i as u64);
        cmd.set_compute_pipeline_state(&pipe);
        cmd.set_kernel_buffer(0, Some(&a), 0);
        cmd.set_kernel_buffer(1, Some(&b), 0);
        cmd.set_kernel_buffer(2, Some(&n_buf), 0);
        cmd.concurrent_dispatch_threads(grid, tg);
    }
    let build_ns = t_build.elapsed().as_nanos();

    // Execute.
    let t_dispatch = Instant::now();
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    // ICB references buffers directly — encoder must mark them used.
    enc.use_resource(&a, metal::MTLResourceUsage::Write | metal::MTLResourceUsage::Read);
    enc.use_resource(&b, metal::MTLResourceUsage::Read);
    enc.use_resource(&n_buf, metal::MTLResourceUsage::Read);
    unsafe {
        compute_execute_commands_in_buffer(
            enc,
            &icb,
            NSRange {
                location: 0,
                length: N_DISPATCHES as u64,
            },
        );
    }
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let dispatch_ns = t_dispatch.elapsed().as_nanos();

    Some((build_ns, dispatch_ns))
}

fn fmt_ns(ns: u128) -> String {
    let us = ns as f64 / 1_000.0;
    if us > 1_000.0 {
        format!("{:.2} ms", us / 1_000.0)
    } else {
        format!("{:.1} us", us)
    }
}

#[test]
fn icb_per_dispatch_overhead_microbench() {
    let ctx = MetalContext::new().expect("metal device");
    println!("=== ICB per-dispatch overhead microbench ===");
    println!(
        "kernel: add_inplace; n={}; dispatches/run: {}; trials: {}",
        KERNEL_N, N_DISPATCHES, TRIALS
    );
    println!("metal-rs: 0.29.0 (ICB API exposed)");
    println!();

    // Warm-up — first dispatch incurs PSO compile + library load.
    let _ = path_a1_one_encoder(&ctx);
    let _ = path_a2_per_dispatch_encoder(&ctx);
    let _ = path_b_icb(&ctx);

    let mut a1: Vec<u128> = Vec::with_capacity(TRIALS);
    let mut a2: Vec<u128> = Vec::with_capacity(TRIALS);
    let mut b: Vec<u128> = Vec::with_capacity(TRIALS);
    let mut b_build: Vec<u128> = Vec::with_capacity(TRIALS);
    let mut icb_supported = true;

    for trial in 0..TRIALS {
        let t_a1 = path_a1_one_encoder(&ctx);
        let t_a2 = path_a2_per_dispatch_encoder(&ctx);
        let (build_ns, t_b) = match path_b_icb(&ctx) {
            Some(t) => t,
            None => {
                icb_supported = false;
                (0, 0)
            }
        };
        a1.push(t_a1);
        a2.push(t_a2);
        b.push(t_b);
        b_build.push(build_ns);
        println!(
            "trial {}: A1={} A2={} B={} (B build={})",
            trial,
            fmt_ns(t_a1),
            fmt_ns(t_a2),
            fmt_ns(t_b),
            fmt_ns(build_ns)
        );
    }

    fn median(mut v: Vec<u128>) -> u128 {
        v.sort();
        v[v.len() / 2]
    }
    let m_a1 = median(a1);
    let m_a2 = median(a2);
    let m_b = median(b);
    let m_build = median(b_build);

    println!();
    println!("--- median wall (N={} dispatches/run) ---", N_DISPATCHES);
    println!("A1 (one CB, one encoder, many dispatches): {}", fmt_ns(m_a1));
    println!("A2 (one CB, per-dispatch encoder — production pattern): {}", fmt_ns(m_a2));
    if icb_supported {
        println!("B  (ICB execute_commands_in_buffer): {}", fmt_ns(m_b));
        println!("B  (one-time build cost): {}", fmt_ns(m_build));
    } else {
        println!("B  UNSUPPORTED on this device (ICB allocation returned 0-size)");
    }

    println!();
    println!("--- per-dispatch overhead (median wall / N) ---");
    let pd_a1 = (m_a1 as f64) / (N_DISPATCHES as f64);
    let pd_a2 = (m_a2 as f64) / (N_DISPATCHES as f64);
    println!("A1: {:.2} us/dispatch", pd_a1 / 1_000.0);
    println!("A2: {:.2} us/dispatch", pd_a2 / 1_000.0);
    if icb_supported {
        let pd_b = (m_b as f64) / (N_DISPATCHES as f64);
        println!("B : {:.2} us/dispatch (build amortized over many runs)", pd_b / 1_000.0);
        println!();
        let red_vs_a1 = 100.0 * (1.0 - pd_b / pd_a1);
        let red_vs_a2 = 100.0 * (1.0 - pd_b / pd_a2);
        println!("--- ICB per-dispatch reduction ---");
        println!("ICB vs A1 (best-case path A): {:+.1}%", red_vs_a1);
        println!("ICB vs A2 (production pattern): {:+.1}%", red_vs_a2);
        println!();
        println!("(For the decode-gap-anatomy projection, the relevant comparison");
        println!(" is ICB vs A2 — production currently uses per-dispatch encoders");
        println!(" in dispatch_batch.)");
    } else {
        println!();
        println!("--- ICB POC blocked ---");
        println!("metal-rs 0.29 exposes the ICB API but allocation failed at runtime.");
        println!("Likely cause: device feature-set check or descriptor parameter.");
    }
}
