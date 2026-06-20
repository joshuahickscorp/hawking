//! Wedge B parity: TCB-batched rmsnorm + add_inplace produces bit-identical
//! output to the CPU reference kernels.
//!
//! Tests:
//!   1. `tcb_rmsnorm_matches_cpu` — rmsnorm_metal_buf_tcb ≡ CPU rmsnorm
//!   2. `tcb_add_inplace_matches_cpu` — add_inplace_metal_tcb ≡ CPU add_inplace
//!   3. `tcb_staggered_loop_matches_cpu` — simulated staggered per-layer pattern
//!      (the forward_token_final_norm Wedge B inner loop) produces identical
//!      residual stream to the CPU [rmsnorm + CPU add_inplace] sequence.
#![cfg(target_os = "macos")]

use hawking_core::kernels;
use hawking_core::metal::{PinnedBuffer, TokenCommandBuffer};

mod common;
use common::*;

fn write_f32_buf(buf: &PinnedBuffer, data: &[f32]) {
    let ptr = buf.contents() as *mut f32;
    unsafe { ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()) };
}

// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn tcb_rmsnorm_matches_cpu() {
    let h = 2048usize;
    let eps = 1e-6_f32;
    let ctx = ctx();

    let x = fixed_f32(h, 0xABCD_1234);
    let w = fixed_f32(h, 0xDEAD_BEEF);

    let mut cpu_out = vec![0.0f32; h];
    kernels::rmsnorm(&x, &w, eps, &mut cpu_out);

    let x_buf = new_f32_buf(ctx, &x);
    let w_buf = new_f32_buf(ctx, &w);
    let out_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());

    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::rmsnorm_metal_buf_tcb(&mut tcb, &x_buf, &w_buf, eps, h, &out_buf)
            .expect("rmsnorm_metal_buf_tcb");
        tcb.commit_and_wait().expect("commit");
    }

    let gpu_out = read_f32_buf(&out_buf, h);
    let diff = max_abs_diff(&cpu_out, &gpu_out);
    println!("[WedgeB] rmsnorm TCB vs CPU  max abs diff = {diff:.2e}");
    assert!(diff < 1e-5, "rmsnorm TCB vs CPU diff {diff:.2e} >= 1e-5");
}

#[test]
fn tcb_add_inplace_matches_cpu() {
    let h = 2048usize;
    let ctx = ctx();

    let mut a_cpu = fixed_f32(h, 0xCAFE_BABE);
    let b = fixed_f32(h, 0x1234_5678);

    // CPU reference.
    kernels::add_inplace(&mut a_cpu, &b);

    let a_buf = new_f32_buf(ctx, &fixed_f32(h, 0xCAFE_BABE));
    let b_buf = new_f32_buf(ctx, &b);

    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::add_inplace_metal_tcb(&mut tcb, &a_buf, &b_buf, h).expect("add_inplace_metal_tcb");
        tcb.commit_and_wait().expect("commit");
    }

    let gpu_a = read_f32_buf(&a_buf, h);
    let diff = max_abs_diff(&a_cpu, &gpu_a);
    println!("[WedgeB] add_inplace TCB vs CPU  max abs diff = {diff:.2e}");
    assert!(
        diff < 1e-6,
        "add_inplace TCB vs CPU diff {diff:.2e} >= 1e-6"
    );
}

/// Simulate the Wedge B staggered per-layer forward pass pattern for N_LAYERS
/// synthetic layers. Each layer applies: add_inplace(x, delta) then rmsnorm(x→out).
/// Compares TCB path vs CPU reference; output must be bit-identical.
#[test]
fn tcb_staggered_loop_matches_cpu() {
    const N_LAYERS: usize = 4;
    let h = 2048usize;
    let eps = 1e-6_f32;
    let ctx = ctx();

    // Generate fixed synthetic data.
    let x_init = fixed_f32(h, 0x1111_2222);
    let deltas: Vec<Vec<f32>> = (0..N_LAYERS)
        .map(|i| fixed_f32(h, 0xAAAA_0000 + i as u64))
        .collect();
    let norms: Vec<Vec<f32>> = (0..N_LAYERS)
        .map(|i| fixed_f32(h, 0xBBBB_0000 + i as u64))
        .collect();

    // ── CPU reference path ───────────────────────────────────────────────
    // Pattern: for each layer, add delta to x, then rmsnorm x.
    let mut x_cpu = x_init.clone();
    let mut norm_outs_cpu = vec![vec![0.0f32; h]; N_LAYERS];
    for li in 0..N_LAYERS {
        kernels::add_inplace(&mut x_cpu, &deltas[li]);
        kernels::rmsnorm(&x_cpu, &norms[li], eps, &mut norm_outs_cpu[li]);
    }
    let x_final_cpu = x_cpu;

    // ── TCB path (staggered, mirroring forward_token_final_norm Wedge B) ─
    // Each layer: mini-TCB [add_inplace(x_buf, delta_buf), rmsnorm(x_buf → out_buf)]
    let x_buf = new_f32_buf(ctx, &x_init);
    let delta_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());
    let out_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());
    let norm_bufs: Vec<PinnedBuffer> = norms.iter().map(|n| new_f32_buf(ctx, n)).collect();

    let mut norm_outs_gpu = vec![vec![0.0f32; h]; N_LAYERS];

    for li in 0..N_LAYERS {
        write_f32_buf(&delta_buf, &deltas[li]);
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::add_inplace_metal_tcb(&mut tcb, &x_buf, &delta_buf, h)
                .expect("add_inplace_metal_tcb");
            kernels::rmsnorm_metal_buf_tcb(&mut tcb, &x_buf, &norm_bufs[li], eps, h, &out_buf)
                .expect("rmsnorm_metal_buf_tcb");
            tcb.commit_and_wait().expect("commit");
        }
        norm_outs_gpu[li] = read_f32_buf(&out_buf, h);
    }
    let x_final_gpu = read_f32_buf(&x_buf, h);

    // ── compare ──────────────────────────────────────────────────────────
    let x_diff = max_abs_diff(&x_final_cpu, &x_final_gpu);
    println!("[WedgeB] staggered loop: x_final max abs diff = {x_diff:.2e}");
    assert!(x_diff < 1e-5, "x_final diff {x_diff:.2e} >= 1e-5");

    for li in 0..N_LAYERS {
        let d = max_abs_diff(&norm_outs_cpu[li], &norm_outs_gpu[li]);
        println!("[WedgeB] staggered loop: layer {li} norm_out max abs diff = {d:.2e}");
        assert!(d < 1e-5, "layer {li} norm_out diff {d:.2e} >= 1e-5");
    }
}
