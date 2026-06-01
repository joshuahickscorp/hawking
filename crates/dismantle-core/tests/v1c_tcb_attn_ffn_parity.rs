//! Wedge C parity: attention + FFN TCB kernel variants produce the same output
//! as CPU reference kernels.
//!
//! Tests:
//!   1. `wedge_c_pair_gemv_norm_matches_cpu` — gemv_f32_attn_pair_arena_tcb +
//!      rmsnorm_metal_buf_tcb chain matches CPU gemv_f32 + rmsnorm for both outputs.
//!   2. `wedge_c_moe_gate_gemv_matches_cpu` — gemv_f32_moe_pinned_buf_tcb matches
//!      CPU gemv_f32.
//!   3. `wedge_c_full_layer_loop_matches_cpu` — simulated Wedge C per-layer pattern
//!      (add_attn + rmsnorm_attn + add_ffn + rmsnorm_ffn) produces identical
//!      residual stream to the CPU reference.
#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{PinnedBuffer, TokenCommandBuffer};

mod common;
use common::*;

// ─────────────────────────────────────────────────────────────────────────────

/// Tests gemv_f32_attn_pair_arena_tcb (two GEMVs sharing one input) then
/// rmsnorm_metal_buf_tcb on each output. Compares to CPU gemv_f32 + rmsnorm.
#[test]
fn wedge_c_pair_gemv_norm_matches_cpu() {
    // Shapes matching DeepSeek-V2-Lite: hidden=2048, q_lora_rank=1536, kv_lora_rank+rope=576+64=640
    let hidden = 256usize; // smaller for test speed
    let rows_a = 64usize;  // q_lora_rank analogue
    let rows_b = 80usize;  // kv_a_dim analogue (kv_lora_rank + qk_rope)
    let kv_lora = 64usize; // kv_lora_rank (first portion of rows_b)
    let eps = 1e-6_f32;
    let ctx = ctx();

    let x = fixed_f32(hidden, 0x1111_AAAA);
    let w_a = fixed_f32(rows_a * hidden, 0x2222_BBBB);
    let w_b = fixed_f32(rows_b * hidden, 0x3333_CCCC);
    let norm_a_w = fixed_f32(rows_a, 0x4444_DDDD);
    let norm_b_w = fixed_f32(kv_lora, 0x5555_EEEE); // norm only first kv_lora elements

    // CPU reference.
    let mut cpu_out_a = vec![0.0f32; rows_a];
    let mut cpu_out_b = vec![0.0f32; rows_b];
    kernels::gemv_f32(&w_a, rows_a, hidden, &x, &mut cpu_out_a);
    kernels::gemv_f32(&w_b, rows_b, hidden, &x, &mut cpu_out_b);
    let mut cpu_normed_a = vec![0.0f32; rows_a];
    let mut cpu_normed_b = vec![0.0f32; kv_lora];
    kernels::rmsnorm(&cpu_out_a, &norm_a_w, eps, &mut cpu_normed_a);
    kernels::rmsnorm(&cpu_out_b[..kv_lora], &norm_b_w, eps, &mut cpu_normed_b);

    // GPU TCB path.
    let x_buf = new_f32_buf(ctx, &x);
    let w_a_buf = new_f32_buf(ctx, &w_a);
    let w_b_buf = new_f32_buf(ctx, &w_b);
    let norm_a_buf = new_f32_buf(ctx, &norm_a_w);
    let norm_b_buf = new_f32_buf(ctx, &norm_b_w);
    let out_a_buf = ctx.new_buffer(rows_a * std::mem::size_of::<f32>());
    let out_b_buf = ctx.new_buffer(rows_b * std::mem::size_of::<f32>());
    let normed_a_buf = ctx.new_buffer(rows_a * std::mem::size_of::<f32>());
    let normed_b_buf = ctx.new_buffer(kv_lora * std::mem::size_of::<f32>());

    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_f32_attn_pair_arena_tcb(
            &mut tcb, &w_a_buf, rows_a, &w_b_buf, rows_b, hidden, &x_buf, &out_a_buf, &out_b_buf,
        ).expect("pair_gemv_tcb");
        // Norm a (all rows_a elements).
        kernels::rmsnorm_metal_buf_tcb(&mut tcb, &out_a_buf, &norm_a_buf, eps, rows_a, &normed_a_buf)
            .expect("rmsnorm_a_tcb");
        // Norm b (only first kv_lora elements of out_b_buf, same pattern as kv_a_norm).
        kernels::rmsnorm_metal_buf_tcb(&mut tcb, &out_b_buf, &norm_b_buf, eps, kv_lora, &normed_b_buf)
            .expect("rmsnorm_b_tcb");
        tcb.commit_and_wait().expect("commit");
    }

    let gpu_out_a = read_f32_buf(&out_a_buf, rows_a);
    let gpu_out_b = read_f32_buf(&out_b_buf, rows_b);
    let gpu_normed_a = read_f32_buf(&normed_a_buf, rows_a);
    let gpu_normed_b = read_f32_buf(&normed_b_buf, kv_lora);

    let diff_a = max_abs_diff(&cpu_out_a, &gpu_out_a);
    let diff_b = max_abs_diff(&cpu_out_b[..rows_b], &gpu_out_b);
    let diff_na = max_abs_diff(&cpu_normed_a, &gpu_normed_a);
    let diff_nb = max_abs_diff(&cpu_normed_b, &gpu_normed_b);

    println!("[WedgeC] pair_gemv out_a diff = {diff_a:.2e}");
    println!("[WedgeC] pair_gemv out_b diff = {diff_b:.2e}");
    println!("[WedgeC] pair_gemv normed_a diff = {diff_na:.2e}");
    println!("[WedgeC] pair_gemv normed_b diff = {diff_nb:.2e}");

    assert!(diff_a < 1e-4, "gemv out_a diff {diff_a:.2e} >= 1e-4");
    assert!(diff_b < 1e-4, "gemv out_b diff {diff_b:.2e} >= 1e-4");
    assert!(diff_na < 1e-4, "normed_a diff {diff_na:.2e} >= 1e-4");
    assert!(diff_nb < 1e-4, "normed_b diff {diff_nb:.2e} >= 1e-4");
}

/// Tests gemv_f32_moe_pinned_buf_tcb (gate-logit GEMV) against CPU gemv_f32.
#[test]
fn wedge_c_moe_gate_gemv_matches_cpu() {
    let hidden = 256usize;
    let n_experts = 32usize; // n_routed_experts analogue
    let ctx = ctx();

    let x = fixed_f32(hidden, 0xCAFE_0001);
    let w = fixed_f32(n_experts * hidden, 0xDEAD_0002);

    // CPU reference.
    let mut cpu_out = vec![0.0f32; n_experts];
    kernels::gemv_f32(&w, n_experts, hidden, &x, &mut cpu_out);

    // GPU TCB path.
    let x_buf = new_f32_buf(ctx, &x);
    let w_buf = new_f32_buf(ctx, &w);
    let out_buf = ctx.new_buffer(n_experts * std::mem::size_of::<f32>());

    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_f32_moe_pinned_buf_tcb(&mut tcb, &w_buf, n_experts, hidden, &x_buf, &out_buf)
            .expect("gemv_f32_moe_pinned_buf_tcb");
        tcb.commit_and_wait().expect("commit");
    }

    let gpu_out = read_f32_buf(&out_buf, n_experts);
    let diff = max_abs_diff(&cpu_out, &gpu_out);
    println!("[WedgeC] moe_gate_gemv max abs diff = {diff:.2e}");
    assert!(diff < 1e-4, "moe gate gemv diff {diff:.2e} >= 1e-4");
}

/// Simulate the Wedge C per-layer pattern: add_attn + rmsnorm_attn then
/// add_ffn + rmsnorm_ffn for N_LAYERS synthetic layers. Verifies the
/// full residual stream is consistent between TCB path and CPU path.
#[test]
fn wedge_c_full_layer_loop_matches_cpu() {
    const N_LAYERS: usize = 4;
    let h = 256usize;
    let eps = 1e-6_f32;
    let ctx = ctx();

    let x_init = fixed_f32(h, 0xF00D_1111);
    let attn_outs: Vec<Vec<f32>> = (0..N_LAYERS)
        .map(|i| fixed_f32(h, 0xAA00_0000 + i as u64))
        .collect();
    let ffn_outs: Vec<Vec<f32>> = (0..N_LAYERS)
        .map(|i| fixed_f32(h, 0xBB00_0000 + i as u64))
        .collect();
    let attn_norms: Vec<Vec<f32>> = (0..N_LAYERS)
        .map(|i| fixed_f32(h, 0xCC00_0000 + i as u64))
        .collect();
    let ffn_norms: Vec<Vec<f32>> = (0..N_LAYERS)
        .map(|i| fixed_f32(h, 0xDD00_0000 + i as u64))
        .collect();

    // CPU reference: Wedge C loop structure.
    // Per layer:
    //   if li>0: x += ffn_out[li-1]
    //   x_norm_attn = rmsnorm(x, attn_norm)
    //   x += attn_out[li]                        ← add_inplace(x_buf, arena.out) in Wedge C
    //   x_norm_ffn = rmsnorm(x, ffn_norm)
    //   [deferred] ffn_out[li] stored for next iter
    // After loop: x += ffn_out[n_layers-1]
    let mut x_cpu = x_init.clone();
    let mut x_norm_attn_cpu = vec![vec![0.0f32; h]; N_LAYERS];
    let mut x_norm_ffn_cpu = vec![vec![0.0f32; h]; N_LAYERS];

    for li in 0..N_LAYERS {
        if li > 0 {
            kernels::add_inplace(&mut x_cpu, &ffn_outs[li - 1]);
        }
        kernels::rmsnorm(&x_cpu, &attn_norms[li], eps, &mut x_norm_attn_cpu[li]);
        // Wedge C: add attn_out to x (Wedge B added attn_out to ffn_out_buf,
        // Wedge C adds arena.out directly to x_buf in mini-TCB β).
        kernels::add_inplace(&mut x_cpu, &attn_outs[li]);
        kernels::rmsnorm(&x_cpu, &ffn_norms[li], eps, &mut x_norm_ffn_cpu[li]);
    }
    // Final: apply last layer's deferred ffn_out.
    kernels::add_inplace(&mut x_cpu, &ffn_outs[N_LAYERS - 1]);
    let x_final_cpu = x_cpu;

    // GPU TCB path (mirroring forward_token_final_norm Wedge C inner loop).
    let x_buf = new_f32_buf(ctx, &x_init);
    let ffn_out_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());
    let attn_out_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());
    let x_norm_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());
    let attn_norm_bufs: Vec<PinnedBuffer> = attn_norms.iter().map(|n| new_f32_buf(ctx, n)).collect();
    let ffn_norm_bufs: Vec<PinnedBuffer> = ffn_norms.iter().map(|n| new_f32_buf(ctx, n)).collect();

    let mut x_norm_attn_gpu = vec![vec![0.0f32; h]; N_LAYERS];
    let mut x_norm_ffn_gpu = vec![vec![0.0f32; h]; N_LAYERS];

    for li in 0..N_LAYERS {
        // Write synthetic attn_out and ffn_out for this layer.
        {
            let ptr = attn_out_buf.contents() as *mut f32;
            unsafe { ptr.copy_from_nonoverlapping(attn_outs[li].as_ptr(), h) };
        }

        // Mini-TCB α: (add ffn_out_buf if li>0) + rmsnorm_attn → x_norm_buf.
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            if li > 0 {
                kernels::add_inplace_metal_tcb(&mut tcb, &x_buf, &ffn_out_buf, h)
                    .expect("add_inplace α");
            }
            kernels::rmsnorm_metal_buf_tcb(&mut tcb, &x_buf, &attn_norm_bufs[li], eps, h, &x_norm_buf)
                .expect("rmsnorm_attn_tcb");
            tcb.commit_and_wait().expect("commit α");
        }
        x_norm_attn_gpu[li] = read_f32_buf(&x_norm_buf, h);

        // Mini-TCB β: add_inplace(x_buf, attn_out_buf) + rmsnorm_ffn → x_norm_buf.
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::add_inplace_metal_tcb(&mut tcb, &x_buf, &attn_out_buf, h)
                .expect("add_inplace β");
            kernels::rmsnorm_metal_buf_tcb(&mut tcb, &x_buf, &ffn_norm_bufs[li], eps, h, &x_norm_buf)
                .expect("rmsnorm_ffn_tcb");
            tcb.commit_and_wait().expect("commit β");
        }
        x_norm_ffn_gpu[li] = read_f32_buf(&x_norm_buf, h);

        // Write synthetic ffn_out into ffn_out_buf (simulates ffn_tcb_inner output).
        {
            let ptr = ffn_out_buf.contents() as *mut f32;
            unsafe { ptr.copy_from_nonoverlapping(ffn_outs[li].as_ptr(), h) };
        }
    }

    // Final: add last layer's ffn_out.
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::add_inplace_metal_tcb(&mut tcb, &x_buf, &ffn_out_buf, h)
            .expect("add_inplace final");
        tcb.commit_and_wait().expect("commit final");
    }

    let x_final_gpu = read_f32_buf(&x_buf, h);

    // Comparisons.
    let x_diff = max_abs_diff(&x_final_cpu, &x_final_gpu);
    println!("[WedgeC] full_loop x_final max abs diff = {x_diff:.2e}");
    assert!(x_diff < 1e-5, "x_final diff {x_diff:.2e} >= 1e-5");

    for li in 0..N_LAYERS {
        let d_attn = max_abs_diff(&x_norm_attn_cpu[li], &x_norm_attn_gpu[li]);
        let d_ffn = max_abs_diff(&x_norm_ffn_cpu[li], &x_norm_ffn_gpu[li]);
        println!("[WedgeC] layer {li}: x_norm_attn diff = {d_attn:.2e}, x_norm_ffn diff = {d_ffn:.2e}");
        assert!(d_attn < 1e-5, "layer {li} x_norm_attn diff {d_attn:.2e} >= 1e-5");
        assert!(d_ffn < 1e-5, "layer {li} x_norm_ffn diff {d_ffn:.2e} >= 1e-5");
    }
}
