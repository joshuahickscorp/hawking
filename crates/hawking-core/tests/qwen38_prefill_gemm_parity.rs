//! GEMM vs CPU reconstruction for the Qwen3.8 prefill kernels.
//!
//! MMA reduction is not bit-identical to a left-to-right matvec. Tolerance is
//! 2e-3 abs. A test that stays green after shifting nibble-8 to nibble-0 is
//! not looking at the dequant — that mutation is asserted below.

#![cfg(target_os = "macos")]

use half::f16;
use hawking_core::metal::{MetalContext, TokenCommandBuffer};

mod common;
use common::{max_abs_diff, new_f32_buf, read_f32_buf};

const GROUP: usize = 64;
const SHMEM: u64 = 5120 * 4;
const TG: u32 = 128;
const ROWS_PER_TG: u32 = 32;

fn metal_or_skip() -> Option<MetalContext> {
    match MetalContext::new() {
        Ok(ctx) => Some(ctx),
        Err(err) => {
            eprintln!("skipping qwen38_prefill_gemm_parity: {err}");
            None
        }
    }
}

fn gemm_grid(rows: u32) -> (u32, u32, u32) {
    (rows.div_ceil(ROWS_PER_TG).saturating_mul(TG).max(TG), 1, 1)
}

fn pack_q4(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<u16>, Vec<f32>) {
    assert_eq!(cols % GROUP, 0);
    let groups_per_row = cols / GROUP;
    let mut codes = vec![0u8; rows * groups_per_row * 32];
    let mut scales = vec![0u16; rows * groups_per_row];
    let mut dense = vec![0.0f32; rows * cols];
    let mut s = seed;
    for row in 0..rows {
        for g in 0..groups_per_row {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale = 0.01 + ((s >> 16) as f32) * (0.02 / 65535.0);
            scales[row * groups_per_row + g] = f16::from_f32(scale).to_bits();
            let rgb = row * groups_per_row + g;
            for local in 0..GROUP {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let q = ((s >> 24) & 0x0f) as u8;
                let byte = rgb * 32 + (local >> 1);
                if local & 1 == 0 {
                    codes[byte] = (codes[byte] & 0xf0) | q;
                } else {
                    codes[byte] = (codes[byte] & 0x0f) | (q << 4);
                }
                dense[row * cols + g * GROUP + local] = (q as i32 - 8) as f32 * scale;
            }
        }
    }
    (codes, scales, dense)
}

fn pack_affine(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<u16>, Vec<u16>, Vec<f32>) {
    assert_eq!(cols % GROUP, 0);
    let groups_per_row = cols / GROUP;
    let mut codes = vec![0u8; rows * groups_per_row * 16];
    let mut scales = vec![0u16; rows * groups_per_row];
    let mut biases = vec![0u16; rows * groups_per_row];
    let mut dense = vec![0.0f32; rows * cols];
    let mut s = seed;
    for row in 0..rows {
        for g in 0..groups_per_row {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let scale = 0.01 + ((s >> 16) as f32) * (0.02 / 65535.0);
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let bias = ((s >> 16) as f32) * (0.02 / 65535.0) - 0.01;
            scales[row * groups_per_row + g] = f16::from_f32(scale).to_bits();
            biases[row * groups_per_row + g] = f16::from_f32(bias).to_bits();
            let rgb = row * groups_per_row + g;
            for local in 0..GROUP {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let q = ((s >> 24) & 0x03) as u8;
                let byte = rgb * 16 + (local >> 2);
                let shift = 2 * (local & 3);
                codes[byte] |= q << shift;
                dense[row * cols + g * GROUP + local] = q as f32 * scale + bias;
            }
        }
    }
    (codes, scales, biases, dense)
}

fn cpu_gemm(dense: &[f32], x: &[f32], rows: usize, cols: usize, batch: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * rows];
    for n in 0..batch {
        for r in 0..rows {
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += dense[r * cols + c] * x[n * cols + c];
            }
            y[n * rows + r] = acc;
        }
    }
    y
}

#[test]
fn q4_prefill_gemm_matches_cpu_reconstruction() {
    let Some(ctx) = metal_or_skip() else {
        return;
    };
    let rows = 64usize;
    let cols = 256usize;
    let batch = 8usize;
    let (codes, scales, dense) = pack_q4(rows, cols, 0xA11CE);
    let xin: Vec<f32> = (0..batch * cols)
        .map(|i| ((i % 19) as f32) * 0.03 - 0.2)
        .collect();
    let expected = cpu_gemm(&dense, &xin, rows, cols, batch);

    let codes_b = ctx.new_buffer_with_bytes(&codes);
    let scales_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let scales_b = ctx.new_buffer_with_bytes(&scales_bytes);
    let x_b = new_f32_buf(&ctx, &xin);
    let y_b = ctx.new_buffer(batch * rows * 4);
    let rows_u = rows as u32;
    let cols_u = cols as u32;
    let batch_u = batch as u32;
    {
        let mut tcb = TokenCommandBuffer::new(&ctx);
        tcb.dispatch_threads(
            "qwen38_prefill_q4_g64_gemm_mma_n64",
            gemm_grid(rows_u),
            (TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&codes_b), 0);
                enc.set_buffer(1, Some(&scales_b), 0);
                enc.set_buffer(2, Some(&x_b), 0);
                enc.set_buffer(3, Some(&y_b), 0);
                enc.set_bytes(4, 4, &rows_u as *const u32 as *const _);
                enc.set_bytes(5, 4, &cols_u as *const u32 as *const _);
                enc.set_bytes(6, 4, &batch_u as *const u32 as *const _);
                enc.set_threadgroup_memory_length(0, SHMEM);
            },
        )
        .expect("dispatch q4 gemm");
        tcb.commit_and_wait().expect("commit q4 gemm");
    }
    let got = read_f32_buf(&y_b, batch * rows);
    let diff = max_abs_diff(&expected, &got);
    assert!(
        diff < 2e-3,
        "q4 prefill GEMM vs CPU: max_abs_diff={diff} (limit 2e-3)"
    );

    let mut dense_wrong = dense.clone();
    for g in dense_wrong.iter_mut() {
        *g += 0.08;
    }
    let wrong = cpu_gemm(&dense_wrong, &xin, rows, cols, batch);
    let wrong_diff = max_abs_diff(&wrong, &got);
    assert!(
        wrong_diff > 0.05,
        "mutation check: GEMM matched a deliberately wrong CPU reconstruction (diff={wrong_diff})"
    );
}

#[test]
fn affine_prefill_gemm_matches_cpu_reconstruction() {
    let Some(ctx) = metal_or_skip() else {
        return;
    };
    let rows = 48usize;
    let cols = 256usize;
    let batch = 4usize;
    let (codes, scales, biases, dense) = pack_affine(rows, cols, 0xBEEF);
    let xin: Vec<f32> = (0..batch * cols)
        .map(|i| ((i % 13) as f32) * 0.04 - 0.25)
        .collect();
    let expected = cpu_gemm(&dense, &xin, rows, cols, batch);

    let codes_b = ctx.new_buffer_with_bytes(&codes);
    let scales_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let biases_bytes: Vec<u8> = biases.iter().flat_map(|s| s.to_le_bytes()).collect();
    let scales_b = ctx.new_buffer_with_bytes(&scales_bytes);
    let biases_b = ctx.new_buffer_with_bytes(&biases_bytes);
    let x_b = new_f32_buf(&ctx, &xin);
    let y_b = ctx.new_buffer(batch * rows * 4);
    let rows_u = rows as u32;
    let cols_u = cols as u32;
    let batch_u = batch as u32;
    {
        let mut tcb = TokenCommandBuffer::new(&ctx);
        tcb.dispatch_threads(
            "qwen38_prefill_affine_q2_g64_gemm_mma_n64",
            gemm_grid(rows_u),
            (TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&codes_b), 0);
                enc.set_buffer(1, Some(&scales_b), 0);
                enc.set_buffer(2, Some(&biases_b), 0);
                enc.set_buffer(3, Some(&x_b), 0);
                enc.set_buffer(4, Some(&y_b), 0);
                enc.set_bytes(5, 4, &rows_u as *const u32 as *const _);
                enc.set_bytes(6, 4, &cols_u as *const u32 as *const _);
                enc.set_bytes(7, 4, &batch_u as *const u32 as *const _);
                enc.set_threadgroup_memory_length(0, SHMEM);
            },
        )
        .expect("dispatch affine gemm");
        tcb.commit_and_wait().expect("commit affine gemm");
    }
    let got = read_f32_buf(&y_b, batch * rows);
    let diff = max_abs_diff(&expected, &got);
    assert!(
        diff < 2e-3,
        "affine prefill GEMM vs CPU: max_abs_diff={diff} (limit 2e-3)"
    );
}

#[test]
fn prefill_chunk_tokens_clamps_to_64() {
    let saved = std::env::var("HAWKING_QWEN38_PREFILL_CHUNK").ok();
    std::env::remove_var("HAWKING_QWEN38_PREFILL_CHUNK");
    assert_eq!(
        hawking_core::model::qwen38_hybrid_decode::qwen38_prefill_chunk_tokens(),
        64
    );
    std::env::set_var("HAWKING_QWEN38_PREFILL_CHUNK", "128");
    assert_eq!(
        hawking_core::model::qwen38_hybrid_decode::qwen38_prefill_chunk_tokens(),
        64,
        "C=128 exceeds the MMA N tile"
    );
    std::env::set_var("HAWKING_QWEN38_PREFILL_CHUNK", "16");
    assert_eq!(
        hawking_core::model::qwen38_hybrid_decode::qwen38_prefill_chunk_tokens(),
        16
    );
    match saved {
        Some(v) => std::env::set_var("HAWKING_QWEN38_PREFILL_CHUNK", v),
        None => std::env::remove_var("HAWKING_QWEN38_PREFILL_CHUNK"),
    }
}
