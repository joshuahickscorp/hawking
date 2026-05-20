//! path-to-100 L5 Lever A — parity gate for the
//! `eagle4_rmsnorm_residual_gate` Metal kernel.
//!
//! The kernel fuses the CPU output stage at
//! `crates/dismantle-core/src/speculate/eagle4_head.rs:771-806`:
//!     baseline      = rmsnorm(h_high, output_norm, eps)
//!     draft_hidden  = baseline + residual_gate · x
//! into a single threadgroup-resident dispatch. This test asserts the
//! GPU dispatch matches a faithful CPU port of those two steps to
//! within fp16-equivalent tolerance, at three operating shapes:
//!
//!   1. V2-Lite HIDDEN=2048 with scalar gate (residual_gate length 1)
//!   2. V2-Lite HIDDEN=2048 with per-dim gate (length HIDDEN)
//!   3. Gate=0 edge case (output should equal rmsnorm(h_high) exactly)
//!
//! Tolerance: 1e-4 relative (fp32 round-trip; the kernel does the same
//! fmadd sequence as the CPU ref, so the diff is sum-order-noise only).

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};

fn fixed_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut x = vec![0.0f32; n];
    let mut s = seed;
    for v in &mut x {
        s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let bits = (s >> 33) as u32;
        *v = ((bits as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.5;
    }
    x
}

fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(data))
}

fn read_f32_buf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

/// CPU reference: replicates eagle4_head.rs:774-782 exactly.
fn cpu_reference(h_high: &[f32], output_norm: &[f32], gate: &[f32], x: &[f32], eps: f32) -> Vec<f32> {
    let h = h_high.len();
    let mut baseline = vec![0.0f32; h];
    kernels::rmsnorm(h_high, output_norm, eps, &mut baseline);
    let gate_last = gate.len() - 1;
    let mut out = vec![0.0f32; h];
    for i in 0..h {
        let alpha = gate[i.min(gate_last)];
        out[i] = baseline[i] + alpha * x[i];
    }
    out
}

fn run_gpu(
    ctx: &MetalContext,
    h_high: &[f32],
    output_norm: &[f32],
    gate: &[f32],
    x: &[f32],
    eps: f32,
) -> Vec<f32> {
    let h = h_high.len();
    let h_high_buf = new_f32_buf(ctx, h_high);
    let weight_buf = new_f32_buf(ctx, output_norm);
    let gate_buf = new_f32_buf(ctx, gate);
    let x_buf = new_f32_buf(ctx, x);
    let out_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::eagle4_rmsnorm_residual_gate_tcb(
        &mut tcb,
        &h_high_buf,
        &weight_buf,
        &gate_buf,
        &x_buf,
        &out_buf,
        h,
        gate.len() > 1,
        eps,
    )
    .expect("eagle4_rmsnorm_residual_gate dispatch");
    tcb.commit_and_wait().expect("commit");

    read_f32_buf(&out_buf, h)
}

fn assert_close(label: &str, ref_v: &[f32], gpu_v: &[f32]) {
    assert_eq!(ref_v.len(), gpu_v.len());
    let max_abs = ref_v
        .iter()
        .zip(gpu_v.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_ref = ref_v.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let rel = max_abs / max_ref.max(1.0);
    assert!(
        rel < 1e-4,
        "{label}: max_abs_diff={max_abs:.3e} max_ref={max_ref:.3e} \
         rel={rel:.3e} (threshold=1e-4)"
    );
}

#[test]
fn rmsnorm_residual_gate_v2lite_scalar_gate() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    let h = 2048usize;
    let eps = 1e-6f32;
    let h_high = fixed_f32(h, 0xDEAD_BEEF);
    let output_norm = fixed_f32(h, 0xCAFE_F00D);
    let gate = vec![0.123_f32];
    let x = fixed_f32(h, 0xABCD_1234);

    let ref_out = cpu_reference(&h_high, &output_norm, &gate, &x, eps);
    let gpu_out = run_gpu(&ctx, &h_high, &output_norm, &gate, &x, eps);
    assert_close("v2lite scalar gate", &ref_out, &gpu_out);
}

#[test]
fn rmsnorm_residual_gate_v2lite_vector_gate() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    let h = 2048usize;
    let eps = 1e-6f32;
    let h_high = fixed_f32(h, 0xBEEF_BABE);
    let output_norm = fixed_f32(h, 0xFEED_FACE);
    let gate = fixed_f32(h, 0x1357_9BDF);
    let x = fixed_f32(h, 0x2468_ACE0);

    let ref_out = cpu_reference(&h_high, &output_norm, &gate, &x, eps);
    let gpu_out = run_gpu(&ctx, &h_high, &output_norm, &gate, &x, eps);
    assert_close("v2lite vector gate", &ref_out, &gpu_out);
}

#[test]
fn rmsnorm_residual_gate_zero_gate_equals_rmsnorm_only() {
    // With gate=0 the kernel's output must equal rmsnorm(h_high) exactly
    // (modulo the per-element add of 0.0 * x which is bit-identical to
    // baseline in IEEE-754 for finite gate values). This isolates the
    // rmsnorm half of the kernel from the residual-gate half.
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    let h = 2048usize;
    let eps = 1e-6f32;
    let h_high = fixed_f32(h, 0x0F0F_F0F0);
    let output_norm = fixed_f32(h, 0xA5A5_5A5A);
    let gate = vec![0.0_f32];
    let x = fixed_f32(h, 0x5555_AAAA);

    let mut ref_out = vec![0.0f32; h];
    kernels::rmsnorm(&h_high, &output_norm, eps, &mut ref_out);
    let gpu_out = run_gpu(&ctx, &h_high, &output_norm, &gate, &x, eps);
    assert_close("zero-gate isolates rmsnorm", &ref_out, &gpu_out);
}

#[test]
fn rmsnorm_residual_gate_small_shape() {
    // 512-wide variant — exercises the non-V2-Lite shape path the
    // kernel must handle without divisibility assumptions.
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    let h = 512usize;
    let eps = 1e-6f32;
    let h_high = fixed_f32(h, 0xAAAA_5555);
    let output_norm = fixed_f32(h, 0x3333_CCCC);
    let gate = fixed_f32(h, 0x9999_6666);
    let x = fixed_f32(h, 0xC0DE_F00D);

    let ref_out = cpu_reference(&h_high, &output_norm, &gate, &x, eps);
    let gpu_out = run_gpu(&ctx, &h_high, &output_norm, &gate, &x, eps);
    assert_close("small shape vector gate", &ref_out, &gpu_out);
}
