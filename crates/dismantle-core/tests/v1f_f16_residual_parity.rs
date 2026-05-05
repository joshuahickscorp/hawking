//! Wedge F parity: f16 residual stream kernel variants.
//!
//! Tests:
//!   1. `wedge_f_rmsnorm_f16_to_f32_matches_cpu` — rmsnorm_f16_to_f32_tcb output
//!      matches CPU rmsnorm at atol 1e-3 (fp16 precision).
//!   2. `wedge_f_cast_f32_to_f16_round_trips` — cast_f32_to_f16_tcb is
//!      equivalent to element-wise half::from_f32 conversion.
//!   3. `wedge_f_add_inplace_f16_matches_cpu` — add_inplace_f16_tcb matches
//!      CPU element-wise f16 add.
//!   4. `wedge_f_embed_lookup_f16_matches_cpu` — embed_lookup_f16_tcb matches
//!      CPU embed_lookup output at atol 0 (exact f16 copy).
//!   5. `wedge_f_residual_loop_matches_f32_path` — simulated two-layer Wedge F
//!      residual accumulation (f16 path) matches the Wedge C f32 path at atol 5e-3.
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

fn fixed_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

fn fixed_f16(n: usize, seed: u64) -> Vec<f16> {
    fixed_f32(n, seed).iter().map(|&v| f16::from_f32(v)).collect()
}

fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

fn read_f32_buf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn read_f16_buf(buf: &PinnedBuffer, n: usize) -> Vec<f16> {
    let ptr = buf.contents() as *const f16;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

// ─────────────────────────────────────────────────────────────────────────────

/// rmsnorm_f16_to_f32_tcb reads f16 x, writes f32 norm.
/// Compare to CPU rmsnorm with f32 x (after converting f16→f32).
#[test]
fn wedge_f_rmsnorm_f16_to_f32_matches_cpu() {
    let ctx = ctx();
    let hidden = 256usize;
    let eps = 1e-6_f32;

    let x_f16 = fixed_f16(hidden, 0xAAAA_1111);
    let weight = fixed_f32(hidden, 0xBBBB_2222);

    // CPU reference: convert x_f16 → f32 first, then rmsnorm.
    let x_f32: Vec<f32> = x_f16.iter().map(|v| v.to_f32()).collect();
    let mut cpu_out = vec![0.0f32; hidden];
    kernels::rmsnorm(&x_f32, &weight, eps, &mut cpu_out);

    // GPU TCB path.
    let x_buf = new_f16_buf(ctx, &x_f16);
    let w_buf = new_f32_buf(ctx, &weight);
    let out_buf = ctx.new_buffer(hidden * std::mem::size_of::<f32>());

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::rmsnorm_f16_to_f32_tcb(&mut tcb, &x_buf, &w_buf, eps, hidden, &out_buf)
        .expect("rmsnorm_f16_to_f32_tcb");
    tcb.commit_and_wait().expect("commit");

    let gpu_out = read_f32_buf(&out_buf, hidden);
    let diff = max_abs_diff_f32(&cpu_out, &gpu_out);
    assert!(
        diff < 1e-3,
        "rmsnorm_f16_to_f32: max_abs_diff={diff:.2e} > 1e-3"
    );
}

/// cast_f32_to_f16_tcb: dst[i] = (half)src[i]. Compare to half::from_f32.
#[test]
fn wedge_f_cast_f32_to_f16_round_trips() {
    let ctx = ctx();
    let n = 512usize;

    let src = fixed_f32(n, 0xCCCC_3333);
    let expected: Vec<f16> = src.iter().map(|&v| f16::from_f32(v)).collect();

    let src_buf = new_f32_buf(ctx, &src);
    let dst_buf = ctx.new_buffer(n * std::mem::size_of::<f16>());

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::cast_f32_to_f16_tcb(&mut tcb, &src_buf, &dst_buf, n).expect("cast_f32_to_f16_tcb");
    tcb.commit_and_wait().expect("commit");

    let got = read_f16_buf(&dst_buf, n);
    for i in 0..n {
        assert_eq!(
            got[i].to_bits(),
            expected[i].to_bits(),
            "cast_f32_to_f16 mismatch at i={i}: got={} expected={}",
            got[i].to_f32(),
            expected[i].to_f32()
        );
    }
}

/// add_inplace_f16_tcb: a[i] += b[i], both f16. Matches CPU f16 add.
#[test]
fn wedge_f_add_inplace_f16_matches_cpu() {
    let ctx = ctx();
    let n = 256usize;

    let a_f16 = fixed_f16(n, 0xDDDD_4444);
    let b_f16 = fixed_f16(n, 0xEEEE_5555);

    // CPU reference.
    let cpu_out: Vec<f16> = a_f16
        .iter()
        .zip(b_f16.iter())
        .map(|(&a, &b)| f16::from_f32(a.to_f32() + b.to_f32()))
        .collect();

    let a_buf = new_f16_buf(ctx, &a_f16);
    let b_buf = new_f16_buf(ctx, &b_f16);

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::add_inplace_f16_tcb(&mut tcb, &a_buf, &b_buf, n).expect("add_inplace_f16_tcb");
    tcb.commit_and_wait().expect("commit");

    let gpu_out = read_f16_buf(&a_buf, n);
    let diff = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(g, c)| (g.to_f32() - c.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    assert!(diff < 1e-3, "add_inplace_f16: max_abs_diff={diff:.2e} > 1e-3");
}

/// embed_lookup_f16_tcb: reads f16 embed table at row `token`, writes f16 out.
/// Compared to the CPU embed_lookup (which outputs f32), with round-trip through f16.
#[test]
fn wedge_f_embed_lookup_f16_matches_cpu() {
    let ctx = ctx();
    let vocab = 64usize;
    let hidden = 128usize;
    let token = 17u32;

    let embed_f16 = fixed_f16(vocab * hidden, 0xFFFF_6666);

    // CPU reference: extract row `token` directly from f16 table.
    let expected_f16: Vec<f16> = embed_f16[token as usize * hidden..(token as usize + 1) * hidden].to_vec();

    let embed_buf = new_f16_buf(ctx, &embed_f16);
    let out_buf = ctx.new_buffer(hidden * std::mem::size_of::<f16>());

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::embed_lookup_f16_tcb(&mut tcb, &embed_buf, token, hidden, &out_buf)
        .expect("embed_lookup_f16_tcb");
    tcb.commit_and_wait().expect("commit");

    let got = read_f16_buf(&out_buf, hidden);
    for i in 0..hidden {
        assert_eq!(
            got[i].to_bits(),
            expected_f16[i].to_bits(),
            "embed_lookup_f16 mismatch at i={i}: got={} expected={}",
            got[i].to_f32(),
            expected_f16[i].to_f32()
        );
    }
}

/// Simulated 2-layer Wedge F residual loop (f16 path) vs CPU f32 reference.
/// Flow: embed_f16 → x_f16, 2× (rmsnorm_f16_to_f32 + fake_attn_delta + add_f16 + rmsnorm_f16_to_f32 + fake_ffn_delta + add_f16) → rmsnorm_f16_to_f32 → x_norm_f32.
/// Compare final x_norm_f32 to the CPU f32 path. atol 5e-3 (fp16 noise).
#[test]
fn wedge_f_residual_loop_matches_f32_path() {
    let ctx = ctx();
    let hidden = 128usize;
    let n_layers = 2usize;
    let eps = 1e-6_f32;

    // Fixed initial embed row (f16).
    let embed_row_f16 = fixed_f16(hidden, 0x1234_ABCD);
    let embed_row_f32: Vec<f32> = embed_row_f16.iter().map(|v| v.to_f32()).collect();

    // Per-layer norm weights (f32) and fake deltas (f32, representing attn/FFN outputs).
    let attn_norms: Vec<Vec<f32>> = (0..n_layers)
        .map(|li| fixed_f32(hidden, 0x1000_0000 ^ li as u64))
        .collect();
    let ffn_norms: Vec<Vec<f32>> = (0..n_layers)
        .map(|li| fixed_f32(hidden, 0x2000_0000 ^ li as u64))
        .collect();
    let attn_deltas: Vec<Vec<f32>> = (0..n_layers)
        .map(|li| fixed_f32(hidden, 0x3000_0000 ^ li as u64).iter().map(|&v| v * 0.1).collect())
        .collect();
    let ffn_deltas: Vec<Vec<f32>> = (0..n_layers)
        .map(|li| fixed_f32(hidden, 0x4000_0000 ^ li as u64).iter().map(|&v| v * 0.1).collect())
        .collect();

    // ── CPU reference (f32 path) ──────────────────────────────────────────────
    let mut x_f32 = embed_row_f32.clone();
    let mut prev_ffn_delta = vec![0.0f32; hidden];
    for li in 0..n_layers {
        if li > 0 {
            for i in 0..hidden { x_f32[i] += prev_ffn_delta[i]; }
        }
        let mut x_norm = vec![0.0f32; hidden];
        kernels::rmsnorm(&x_f32, &attn_norms[li], eps, &mut x_norm);
        // simulate attention: x += attn_delta
        for i in 0..hidden { x_f32[i] += attn_deltas[li][i]; }
        kernels::rmsnorm(&x_f32, &ffn_norms[li], eps, &mut x_norm);
        // simulate FFN: record ffn_delta for next layer
        prev_ffn_delta = ffn_deltas[li].clone();
    }
    // last layer's FFN delta
    for i in 0..hidden { x_f32[i] += prev_ffn_delta[i]; }
    let mut cpu_x_norm = vec![0.0f32; hidden];
    let final_norm_w = fixed_f32(hidden, 0xF000_0000);
    kernels::rmsnorm(&x_f32, &final_norm_w, eps, &mut cpu_x_norm);

    // ── GPU Wedge F path (f16 residual) ──────────────────────────────────────
    let embed_buf = new_f16_buf(ctx, &embed_row_f16);
    let x_f16_buf = ctx.new_buffer(hidden * std::mem::size_of::<f16>());
    let delta_f16_buf = ctx.new_buffer(hidden * std::mem::size_of::<f16>());
    let x_norm_f32_buf = ctx.new_buffer(hidden * std::mem::size_of::<f32>());
    let final_norm_buf = new_f32_buf(ctx, &final_norm_w);

    // Embed lookup → x_f16_buf.
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::embed_lookup_f16_tcb(&mut tcb, &embed_buf, 0, hidden, &x_f16_buf).unwrap();
        tcb.commit_and_wait().unwrap();
    }

    let mut prev_ffn_delta_f32 = vec![0.0f32; hidden];
    for li in 0..n_layers {
        // (a) if li>0: add FFN delta (f16) to x_f16.
        if li > 0 {
            let delta_f16_v: Vec<f16> = prev_ffn_delta_f32.iter().map(|&v| f16::from_f32(v)).collect();
            let prev_delta_buf = new_f16_buf(ctx, &delta_f16_v);
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::add_inplace_f16_tcb(&mut tcb, &x_f16_buf, &prev_delta_buf, hidden).unwrap();
            tcb.commit_and_wait().unwrap();
        }
        // (b) rmsnorm_f16_to_f32 (attn norm).
        {
            let attn_norm_buf = new_f32_buf(ctx, &attn_norms[li]);
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::rmsnorm_f16_to_f32_tcb(&mut tcb, &x_f16_buf, &attn_norm_buf, eps, hidden, &x_norm_f32_buf).unwrap();
            tcb.commit_and_wait().unwrap();
        }
        // (c) cast attn_delta f32→f16 and add to x_f16.
        {
            let attn_delta_buf = new_f32_buf(ctx, &attn_deltas[li]);
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::cast_f32_to_f16_tcb(&mut tcb, &attn_delta_buf, &delta_f16_buf, hidden).unwrap();
            kernels::add_inplace_f16_tcb(&mut tcb, &x_f16_buf, &delta_f16_buf, hidden).unwrap();
            tcb.commit_and_wait().unwrap();
        }
        // (d) rmsnorm_f16_to_f32 (FFN norm).
        {
            let ffn_norm_buf = new_f32_buf(ctx, &ffn_norms[li]);
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::rmsnorm_f16_to_f32_tcb(&mut tcb, &x_f16_buf, &ffn_norm_buf, eps, hidden, &x_norm_f32_buf).unwrap();
            tcb.commit_and_wait().unwrap();
        }
        prev_ffn_delta_f32 = ffn_deltas[li].clone();
    }
    // Last FFN delta + final norm.
    {
        let ffn_delta_f16: Vec<f16> = prev_ffn_delta_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let prev_delta_buf = new_f16_buf(ctx, &ffn_delta_f16);
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::add_inplace_f16_tcb(&mut tcb, &x_f16_buf, &prev_delta_buf, hidden).unwrap();
        kernels::rmsnorm_f16_to_f32_tcb(&mut tcb, &x_f16_buf, &final_norm_buf, eps, hidden, &x_norm_f32_buf).unwrap();
        tcb.commit_and_wait().unwrap();
    }
    let gpu_x_norm = read_f32_buf(&x_norm_f32_buf, hidden);

    let diff = max_abs_diff_f32(&cpu_x_norm, &gpu_x_norm);
    assert!(
        diff < 5e-3,
        "wedge_f residual loop: max_abs_diff={diff:.2e} > 5e-3 (fp16 noise expected ≤1e-3)"
    );
}
