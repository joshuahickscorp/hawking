//! v1.1.0 Phase 1B — llama_port Q4_K GEMV parity at production shapes.

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer};

mod common;
use common::*;

fn synthetic_input(cols: usize) -> Vec<f32> {
    (0..cols).map(|i| ((i % 97) as f32 - 48.0) / 97.0).collect()
}

fn synthetic_q4_k_bytes(n_blocks: usize) -> Vec<u8> {
    use half::f16;

    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
        bytes[off + 2] = 0x00;
        bytes[off + 3] = 0x00; // f16 0.0 dmin
        for i in 4..144 {
            bytes[off + i] = ((b * 13 + i * 37) & 0xff) as u8;
        }
    }
    bytes
}

fn pin(ctx: &MetalContext, bytes: &[u8]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytes)
}

fn assert_llama_port_matches_v2(rows: usize, cols: usize, label: &str) {
    let ctx = ctx();
    let w_bytes = synthetic_q4_k_bytes(rows * (cols / 256));
    let model_buf = pin(ctx, &w_bytes);
    let x = synthetic_input(cols);
    let mut v2_out = vec![0.0f32; rows];
    let mut llama_out = vec![0.0f32; rows];

    kernels::gemv_q4_k_m_v2_pinned(
        ctx,
        &model_buf,
        0,
        w_bytes.len(),
        rows,
        cols,
        &x,
        &mut v2_out,
    )
    .expect("v2 Q4_K GEMV");
    kernels::gemv_q4_k_m_llama_port_pinned(
        ctx,
        &model_buf,
        0,
        w_bytes.len(),
        rows,
        cols,
        &x,
        &mut llama_out,
    )
    .expect("llama_port Q4_K GEMV");

    let diff = max_abs_diff(&v2_out, &llama_out);
    println!("[v1.1.0] llama_port vs v2 {label} rows={rows} cols={cols} max abs diff = {diff:.6e}");
    assert!(
        diff < ATOL,
        "llama_port {label} diff {diff:.6e} >= atol {ATOL}"
    );
}

#[test]
fn llama_port_gate_up_shape_matches_v2() {
    assert_llama_port_matches_v2(1024, 4096, "gate_up");
}

#[test]
fn llama_port_down_shape_matches_v2() {
    assert_llama_port_matches_v2(4096, 1024, "down");
}

#[test]
fn llama_port_dense_shape_matches_v2() {
    assert_llama_port_matches_v2(4096, 4096, "dense");
}
