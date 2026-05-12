//! Phase 7 wedge 7d-prep parity. silu_mul_f16 vs CPU silu_mul via f16 roundtrip.
#![cfg(target_os = "macos")]

use half::f16;
use dismantle_core::kernels::{silu_mul, silu_mul_f16_metal};
use dismantle_core::metal::MetalContext;

#[test]
fn silu_mul_f16_matches_cpu_roundtrip() {
    let ctx = MetalContext::new().expect("metal ctx");
    let n = 4096usize;
    let gate_f32: Vec<f32> = (0..n).map(|i| ((i as f32) - (n as f32) * 0.5) * 0.01).collect();
    let up_f32:   Vec<f32> = (0..n).map(|i| ((i as f32) * 0.5e-3) - 1.0).collect();

    let gate_rt: Vec<f32> = gate_f32.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let up_rt:   Vec<f32> = up_f32.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let mut cpu_ref = vec![0.0f32; n];
    silu_mul(&gate_rt, &up_rt, &mut cpu_ref);

    let gate_f16: Vec<f16> = gate_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let up_f16:   Vec<f16> = up_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let g_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&gate_f16));
    let u_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&up_f16));
    let out_buf = ctx.new_buffer(n * std::mem::size_of::<f16>());
    silu_mul_f16_metal(&ctx, &g_buf, &u_buf, &out_buf, n).expect("dispatch");

    let out_ptr = out_buf.contents() as *const f16;
    let metal_out: &[f16] = unsafe { std::slice::from_raw_parts(out_ptr, n) };
    let metal_f32: Vec<f32> = metal_out.iter().map(|h| h.to_f32()).collect();

    let mut max_diff = 0.0f32;
    for (m, r) in metal_f32.iter().zip(cpu_ref.iter()) {
        let d = (m - r).abs();
        if d > max_diff { max_diff = d; }
    }
    assert!(max_diff < 1e-2, "silu_mul_f16 max diff {} > atol 1e-2", max_diff);
}
