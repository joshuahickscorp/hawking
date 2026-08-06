#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::kernels;
use rand::Rng;
use rand_pcg::Pcg64Mcg;
mod common;
use common::*;
fn synthetic_q4_k_bytes(n_experts: usize, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks_per_row = cols / 256;
    let bytes_per_expert = rows * blocks_per_row * 144;
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_experts * bytes_per_expert];
    for b in 0..(n_experts * rows * blocks_per_row) {
        let off = b * 144;
        let d = 0.0005 + rng.gen::<f32>() * 0.001;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        let dmin = (rng.gen::<f32>() - 0.5) * 0.001;
        bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}
fn run_parity(routes: usize, rows: usize, cols: usize, seed_base: u64) {
    let n_experts = routes + 4;
    let blocks_per_row = cols / 256;
    let bytes_per_expert = rows * blocks_per_row * 144;
    let gate_bytes = synthetic_q4_k_bytes(n_experts, rows, cols, seed_base);
    let up_bytes = synthetic_q4_k_bytes(n_experts, rows, cols, seed_base ^ 0x1234_5678);
    let pad = 128usize;
    let mut w_all = vec![0xA5u8; pad + gate_bytes.len() + up_bytes.len()];
    let gate_offset = pad;
    let up_offset = pad + n_experts * bytes_per_expert;
    w_all[gate_offset..gate_offset + gate_bytes.len()].copy_from_slice(&gate_bytes);
    w_all[up_offset..up_offset + up_bytes.len()].copy_from_slice(&up_bytes);
    let route_ids: Vec<u32> = (0..routes)
        .map(|i| ((i * 3 + 1) % n_experts) as u32)
        .collect();
    let x = fixed_f32(cols, seed_base ^ 0xDEAD_BEEF);
    let mut ref_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_v2t_gu_raw(
        ctx(),
        &w_all,
        gate_offset,
        up_offset,
        &route_ids,
        &x,
        routes,
        rows,
        cols,
        &mut ref_out,
    )
    .expect("v2t_gu dispatch failed");
    let mut v2_out = vec![0.0_f32; routes * rows];
    kernels::moe_batched_gemm_q4_indexed_v2t_gu_v2_raw(
        ctx(),
        &w_all,
        gate_offset,
        up_offset,
        &route_ids,
        &x,
        routes,
        rows,
        cols,
        &mut v2_out,
    )
    .expect("v2t_gu_v2 dispatch failed");
    let diff = max_abs_diff(&ref_out, &v2_out);
    assert!(
        diff < ATOL,
        "v2t_gu_v2 vs v2t_gu diff {diff:.6e} >= atol {ATOL} \
         (routes={routes} rows={rows} cols={cols})"
    );
}
#[test]
fn test_gu_v2_parity_small() {
    run_parity(2, 16, 256, 0xBEEF_0001);
}
#[test]
fn test_gu_v2_parity_production() {
    run_parity(6, 1408, 2048, 0xBEEF_0002);
}
