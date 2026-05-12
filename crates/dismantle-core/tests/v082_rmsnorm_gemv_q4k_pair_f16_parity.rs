//! v0.8.2 parity: rmsnorm_gemv_q4k_pair_f16_metal vs the f32 sibling
//! via f16 roundtrip on input. atol=1e-2 (Q4K quantization noise is larger).
#![cfg(target_os = "macos")]

use dismantle_core::kernels::{
    rmsnorm_gemv_q4k_pair_metal,
    rmsnorm_gemv_q4k_pair_f16_metal,
};
use dismantle_core::metal::MetalContext;
use half::f16;

fn make_q4k_bytes(rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    let blocks_per_row = cols / 256;
    let total = rows * blocks_per_row * 144;
    let mut v = vec![0u8; total];
    let mut s = seed;
    for b in v.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (s >> 16) as u8;
    }
    // Fix d/dmin to small positive f16 values to avoid NaN/Inf.
    for block in 0..(rows * blocks_per_row) {
        let bo = block * 144;
        v[bo]     = 0x66; v[bo + 1] = 0x2E; // d    = 0.1
        v[bo + 2] = 0x66; v[bo + 3] = 0x2A; // dmin = 0.05
    }
    v
}

#[test]
fn rmsnorm_gemv_q4k_pair_f16_matches_f32_via_roundtrip() {
    let ctx = MetalContext::new().expect("metal");
    let rows = 32usize;
    let cols = 256usize;
    let eps = 1e-5_f32;

    let x_f32: Vec<f32> = (0..cols)
        .map(|i| ((i as f32 * 0.2431).sin()) * 1.5)
        .collect();
    let weight_f32: Vec<f32> = (0..cols)
        .map(|i| 1.0 + (i as f32 * 0.0019).cos() * 0.05)
        .collect();
    let weight_f16: Vec<f16> = weight_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let w_gate = make_q4k_bytes(rows, cols, 0xDEAD_BEEF);
    let w_up   = make_q4k_bytes(rows, cols, 0xCAFE_BABE);

    // f32 reference path: roundtrip x through f16 for a fair comparison.
    let x_rt: Vec<f32> = x_f32.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let x_rt_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x_rt));
    let gate_ref_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let up_ref_buf   = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    rmsnorm_gemv_q4k_pair_metal(
        &ctx,
        &weight_f16,
        eps,
        &w_gate,
        &w_up,
        &gate_ref_buf,
        &up_ref_buf,
        &x_rt_buf,
        rows,
        cols,
    )
    .expect("f32 reference");

    let y_gate_ref: Vec<f32> = unsafe {
        std::slice::from_raw_parts(gate_ref_buf.contents() as *const f32, rows).to_vec()
    };
    let y_up_ref: Vec<f32> = unsafe {
        std::slice::from_raw_parts(up_ref_buf.contents() as *const f32, rows).to_vec()
    };

    // f16 bridge path.
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let x_f16_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&x_f16));
    let gate_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let up_f16_buf   = ctx.new_buffer(rows * std::mem::size_of::<f32>());

    rmsnorm_gemv_q4k_pair_f16_metal(
        &ctx,
        &weight_f16,
        eps,
        &w_gate,
        &w_up,
        &gate_f16_buf,
        &up_f16_buf,
        &x_f16_buf,
        rows,
        cols,
    )
    .expect("f16 bridge");

    let y_gate_f16: Vec<f32> = unsafe {
        std::slice::from_raw_parts(gate_f16_buf.contents() as *const f32, rows).to_vec()
    };
    let y_up_f16: Vec<f32> = unsafe {
        std::slice::from_raw_parts(up_f16_buf.contents() as *const f32, rows).to_vec()
    };

    let max_gate = y_gate_ref
        .iter()
        .zip(y_gate_f16.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_up = y_up_ref
        .iter()
        .zip(y_up_f16.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_gate < 1e-2,
        "q4k pair f16 gate max_diff = {} > atol 1e-2",
        max_gate
    );
    assert!(
        max_up < 1e-2,
        "q4k pair f16 up max_diff = {} > atol 1e-2",
        max_up
    );
}
