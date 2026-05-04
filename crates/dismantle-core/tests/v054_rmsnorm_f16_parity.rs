//! Phase 7 wedge 7b parity. rmsnorm_f16 vs CPU rmsnorm via f32->f16->f32 roundtrip.
#![cfg(target_os = "macos")]

use half::f16;
use dismantle_core::kernels::{rmsnorm, rmsnorm_f16_metal};
use dismantle_core::metal::MetalContext;

#[test]
fn rmsnorm_f16_matches_cpu_via_f16_roundtrip() {
    let ctx = MetalContext::new().expect("metal ctx");
    let hidden = 2048usize;
    let eps = 1e-5_f32;

    let x_f32: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.001 - 1.0) * 0.5).collect();
    let weight: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 1e-4).collect();

    let x_f16_then_f32: Vec<f32> = x_f32.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let mut cpu_ref = vec![0.0f32; hidden];
    rmsnorm(&x_f16_then_f32, &weight, eps, &mut cpu_ref);

    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&x_f16));
    let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&weight));
    let out_buf = ctx.new_buffer(hidden * std::mem::size_of::<f16>());
    rmsnorm_f16_metal(&ctx, &x_buf, &w_buf, eps, hidden, &out_buf).expect("dispatch");

    let out_ptr = out_buf.contents() as *const f16;
    let metal_out_f16: &[f16] = unsafe { std::slice::from_raw_parts(out_ptr, hidden) };
    let metal_out_f32: Vec<f32> = metal_out_f16.iter().map(|h| h.to_f32()).collect();

    let mut max_diff = 0.0f32;
    for (m, r) in metal_out_f32.iter().zip(cpu_ref.iter()) {
        let d = (m - r).abs();
        if d > max_diff { max_diff = d; }
    }
    assert!(max_diff < 1e-2, "rmsnorm_f16 max diff {} > atol 1e-2", max_diff);
}

#[test]
fn rmsnorm_f16_no_nan_under_extreme_input() {
    let ctx = MetalContext::new().expect("metal ctx");
    let hidden = 2048usize;
    let eps = 1e-5_f32;
    let max_f16 = 65504.0f32;
    let x_f32: Vec<f32> = vec![max_f16 * 0.5; hidden];
    let weight: Vec<f32> = vec![1.0; hidden];

    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&x_f16));
    let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&weight));
    let out_buf = ctx.new_buffer(hidden * std::mem::size_of::<f16>());
    rmsnorm_f16_metal(&ctx, &x_buf, &w_buf, eps, hidden, &out_buf).expect("dispatch");

    let out_ptr = out_buf.contents() as *const f16;
    let metal_out_f16: &[f16] = unsafe { std::slice::from_raw_parts(out_ptr, hidden) };
    for (i, h) in metal_out_f16.iter().enumerate() {
        let v = h.to_f32();
        assert!(v.is_finite(), "rmsnorm_f16[{i}] = {v} not finite");
    }
}
