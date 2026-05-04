//! v0.8.1 parity: rmsnorm_gemv_f16_attn_pinned_metal vs the f32 sibling
//! via f16 roundtrip on input. Validates the bridge kernel produces
//! matching output within f16 quantization noise (atol=1e-2).
#![cfg(target_os = "macos")]

use dismantle_core::kernels::{
    rmsnorm_gemv_f32_attn_pinned_metal,
    rmsnorm_gemv_f16_attn_pinned_metal,
};
use dismantle_core::metal::MetalContext;
use half::f16;

#[test]
fn rmsnorm_gemv_f16_attn_matches_f32_via_roundtrip() {
    let ctx = MetalContext::new().expect("metal");
    let rows = 256usize;
    let cols = 2048usize;
    let eps = 1e-5_f32;

    let w: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) * 1e-4).sin() * 0.05)
        .collect();
    let weight: Vec<f32> = (0..cols).map(|i| 1.0 + (i as f32) * 1e-5).collect();
    let x_f32: Vec<f32> = (0..cols)
        .map(|i| ((i as f32) * 0.001 - 1.0) * 0.5)
        .collect();

    // f32 reference: roundtrip x through f16 first so the comparison is fair.
    let x_rt: Vec<f32> = x_f32
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect();
    let x_rt_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x_rt));
    let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&w));
    let weight_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&weight));
    let out_ref_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    rmsnorm_gemv_f32_attn_pinned_metal(
        &ctx, &w_buf, &x_rt_buf, &weight_buf, eps, &out_ref_buf, rows, cols,
    )
    .expect("f32 reference");

    let y_ref: Vec<f32> = {
        let ptr = out_ref_buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, rows) }.to_vec()
    };

    // f16 path: upload x as f16 buffer.
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let x_f16_buf =
        ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&x_f16));
    let out_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    rmsnorm_gemv_f16_attn_pinned_metal(
        &ctx, &w_buf, &x_f16_buf, &weight_buf, eps, &out_f16_buf, rows, cols,
    )
    .expect("f16 bridge");

    let y_metal: Vec<f32> = {
        let ptr = out_f16_buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, rows) }.to_vec()
    };

    let max_diff = y_ref
        .iter()
        .zip(y_metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-2,
        "f16 attn bridge max_diff = {} > atol 1e-2",
        max_diff
    );
}
