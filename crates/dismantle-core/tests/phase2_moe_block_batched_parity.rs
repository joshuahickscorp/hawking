//! Phase 2 — batched MoE expert GEMV parity.
//!
//! These tests cover the practical FlashDMoE precursor: selected
//! expert matrices are packed route-major and dispatched as batched
//! GEMVs. The full block test compares the new Metal path against the
//! existing CPU route loop on small fixed inputs.

#![cfg(target_os = "macos")]

use dismantle_core::gguf::GgmlType;
use dismantle_core::kernels;
use dismantle_core::metal::MetalContext;
use dismantle_core::quant::dequant_into;
use half::f16;
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

const ATOL: f32 = 1e-3;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> = Lazy::new(|| MetalContext::new().expect("Metal device"));
    &CTX
}

fn fixed_input(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn synthetic_q4_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 144];
    for b in 0..n_blocks {
        let off = b * 144;
        let d = 0.001 + rng.gen::<f32>() * 0.001;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        let dmin = (rng.gen::<f32>() - 0.5) * 0.001;
        bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..144 {
            bytes[off + i] = rng.gen::<u8>();
        }
    }
    bytes
}

fn synthetic_q8_0_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 34];
    for b in 0..n_blocks {
        let off = b * 34;
        let d = 0.001 + rng.gen::<f32>() * 0.001;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        for i in 0..32 {
            let q = rng.gen_range(-16i8..=16i8);
            bytes[off + 2 + i] = q as u8;
        }
    }
    bytes
}

fn synthetic_q6_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    let mut bytes = vec![0u8; n_blocks * 210];
    for b in 0..n_blocks {
        let off = b * 210;
        for i in 0..192 {
            bytes[off + i] = rng.gen::<u8>();
        }
        for i in 0..16 {
            let s = rng.gen_range(-4i8..=4i8);
            bytes[off + 192 + i] = s as u8;
        }
        let d = 0.0005 + rng.gen::<f32>() * 0.0005;
        bytes[off + 208..off + 210].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
    }
    bytes
}

fn cpu_batched_gemv(
    dtype: GgmlType,
    bytes: &[u8],
    routes: usize,
    rows: usize,
    cols: usize,
    x: &[f32],
    shared_x: bool,
) -> Vec<f32> {
    let elems_per_matrix = rows * cols;
    let bytes_per_matrix = bytes.len() / routes;
    let mut out = vec![0.0f32; routes * rows];
    for r in 0..routes {
        let mut w = vec![0.0f32; elems_per_matrix];
        let wb = &bytes[r * bytes_per_matrix..(r + 1) * bytes_per_matrix];
        dequant_into(dtype, wb, &mut w).expect("synthetic dequant");
        let xs = if shared_x {
            x
        } else {
            &x[r * cols..(r + 1) * cols]
        };
        kernels::gemv_f32(&w, rows, cols, xs, &mut out[r * rows..(r + 1) * rows]);
    }
    out
}

fn pack_selected(full: &[u8], route_ids: &[u32], bytes_per_matrix: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(route_ids.len() * bytes_per_matrix);
    for &eid in route_ids {
        let start = eid as usize * bytes_per_matrix;
        out.extend_from_slice(&full[start..start + bytes_per_matrix]);
    }
    out
}

#[test]
fn test_batched_q4_gemv_matches_cpu() {
    let routes = 3;
    let rows = 64;
    let cols = 256;
    let bytes = synthetic_q4_k_bytes(routes * rows * (cols / 256), 0xA401);
    let x = fixed_input(cols, 0xA402, 0.05);

    let cpu = cpu_batched_gemv(GgmlType::Q4_K, &bytes, routes, rows, cols, &x, true);
    let mut metal = vec![0.0f32; routes * rows];
    kernels::moe_batched_gemm_q4_metal(ctx(), &bytes, routes, rows, cols, &x, &mut metal)
        .expect("batched q4 metal");

    let diff = max_abs_diff(&cpu, &metal);
    println!("[P2] batched q4 max abs diff = {diff:.6}");
    assert!(diff < ATOL, "batched q4 diff {diff} >= {ATOL}");
}

#[test]
fn test_batched_q8_gemv_matches_cpu() {
    let routes = 3;
    let rows = 64;
    let cols = 256;
    let bytes = synthetic_q8_0_bytes(routes * rows * (cols / 32), 0xB801);
    let x = fixed_input(routes * cols, 0xB802, 0.05);

    let cpu = cpu_batched_gemv(GgmlType::Q8_0, &bytes, routes, rows, cols, &x, false);
    let mut metal = vec![0.0f32; routes * rows];
    kernels::moe_batched_gemm_q8_0_metal(ctx(), &bytes, routes, rows, cols, &x, &mut metal)
        .expect("batched q8 metal");

    let diff = max_abs_diff(&cpu, &metal);
    println!("[P2] batched q8_0 max abs diff = {diff:.6}");
    assert!(diff < ATOL, "batched q8_0 diff {diff} >= {ATOL}");
}

#[test]
fn test_batched_q6_gemv_matches_cpu() {
    let routes = 3;
    let rows = 64;
    let cols = 256;
    let bytes = synthetic_q6_k_bytes(routes * rows * (cols / 256), 0xC601);
    let x = fixed_input(routes * cols, 0xC602, 0.05);

    let cpu = cpu_batched_gemv(GgmlType::Q6_K, &bytes, routes, rows, cols, &x, false);
    let mut metal = vec![0.0f32; routes * rows];
    kernels::moe_batched_gemm_q6_k_metal(ctx(), &bytes, routes, rows, cols, &x, &mut metal)
        .expect("batched q6 metal");

    let diff = max_abs_diff(&cpu, &metal);
    println!("[P2] batched q6_k max abs diff = {diff:.6}");
    assert!(diff < ATOL, "batched q6_k diff {diff} >= {ATOL}");
}

#[test]
fn test_moe_block_batched_matches_cpu() {
    let routes = 3;
    let hidden = 256;
    let routed_mid = 256;
    let shared_mid = 256;

    let x = fixed_input(hidden, 0xD001, 0.04);
    let weights = vec![0.45f32, 0.32, 0.19];

    let routed_gate = synthetic_q4_k_bytes(routes * routed_mid * (hidden / 256), 0xD101);
    let routed_up = synthetic_q4_k_bytes(routes * routed_mid * (hidden / 256), 0xD102);
    let routed_down = synthetic_q8_0_bytes(routes * hidden * (routed_mid / 32), 0xD103);
    let shared_gate = synthetic_q4_k_bytes(shared_mid * (hidden / 256), 0xD201);
    let shared_up = synthetic_q4_k_bytes(shared_mid * (hidden / 256), 0xD202);
    let shared_down = synthetic_q6_k_bytes(hidden * (shared_mid / 256), 0xD203);

    let mut cpu = vec![0.0f32; hidden];
    let gate_cpu = cpu_batched_gemv(
        GgmlType::Q4_K,
        &routed_gate,
        routes,
        routed_mid,
        hidden,
        &x,
        true,
    );
    let up_cpu = cpu_batched_gemv(
        GgmlType::Q4_K,
        &routed_up,
        routes,
        routed_mid,
        hidden,
        &x,
        true,
    );
    let mut act_cpu = vec![0.0f32; routes * routed_mid];
    kernels::silu_mul(&gate_cpu, &up_cpu, &mut act_cpu);
    let routed_out = cpu_batched_gemv(
        GgmlType::Q8_0,
        &routed_down,
        routes,
        hidden,
        routed_mid,
        &act_cpu,
        false,
    );
    for r in 0..routes {
        for h in 0..hidden {
            cpu[h] += weights[r] * routed_out[r * hidden + h];
        }
    }

    let shared_gate_cpu = cpu_batched_gemv(
        GgmlType::Q4_K,
        &shared_gate,
        1,
        shared_mid,
        hidden,
        &x,
        true,
    );
    let shared_up_cpu =
        cpu_batched_gemv(GgmlType::Q4_K, &shared_up, 1, shared_mid, hidden, &x, true);
    let mut shared_act = vec![0.0f32; shared_mid];
    kernels::silu_mul(&shared_gate_cpu, &shared_up_cpu, &mut shared_act);
    let shared_out = cpu_batched_gemv(
        GgmlType::Q6_K,
        &shared_down,
        1,
        hidden,
        shared_mid,
        &shared_act,
        false,
    );
    for h in 0..hidden {
        cpu[h] += shared_out[h];
    }

    let mut metal = vec![0.0f32; hidden];
    kernels::moe_block_batched_metal(
        ctx(),
        &routed_gate,
        &routed_up,
        &routed_down,
        &weights,
        Some(&shared_gate),
        Some(&shared_up),
        Some(&shared_down),
        hidden,
        routed_mid,
        shared_mid,
        &x,
        &mut metal,
    )
    .expect("batched moe block metal");

    let diff = max_abs_diff(&cpu, &metal);
    println!("[P2] batched moe block max abs diff = {diff:.6}");
    assert!(diff < ATOL, "batched moe block diff {diff} >= {ATOL}");
}

#[test]
fn test_moe_block_indexed_no_pack_matches_packed_batched() {
    let n_experts = 5;
    let route_ids = vec![4u32, 1, 3];
    let route_weights = vec![0.41f32, 0.33, 0.18];
    let routes = route_ids.len();
    let hidden = 256;
    let routed_mid = 256;
    let shared_mid = 256;

    let x = fixed_input(hidden, 0xE001, 0.04);

    let routed_gate_full = synthetic_q4_k_bytes(n_experts * routed_mid * (hidden / 256), 0xE101);
    let routed_up_full = synthetic_q4_k_bytes(n_experts * routed_mid * (hidden / 256), 0xE102);
    let routed_down_full = synthetic_q8_0_bytes(n_experts * hidden * (routed_mid / 32), 0xE103);
    let shared_gate = synthetic_q4_k_bytes(shared_mid * (hidden / 256), 0xE201);
    let shared_up = synthetic_q4_k_bytes(shared_mid * (hidden / 256), 0xE202);
    let shared_down = synthetic_q6_k_bytes(hidden * (shared_mid / 256), 0xE203);

    let routed_gate_bytes_per = routed_mid * (hidden / 256) * 144;
    let routed_up_bytes_per = routed_mid * (hidden / 256) * 144;
    let routed_down_bytes_per = hidden * (routed_mid / 32) * 34;
    let routed_gate_packed = pack_selected(&routed_gate_full, &route_ids, routed_gate_bytes_per);
    let routed_up_packed = pack_selected(&routed_up_full, &route_ids, routed_up_bytes_per);
    let routed_down_packed = pack_selected(&routed_down_full, &route_ids, routed_down_bytes_per);

    let mut packed = vec![0.0f32; hidden];
    kernels::moe_block_batched_metal(
        ctx(),
        &routed_gate_packed,
        &routed_up_packed,
        &routed_down_packed,
        &route_weights,
        Some(&shared_gate),
        Some(&shared_up),
        Some(&shared_down),
        hidden,
        routed_mid,
        shared_mid,
        &x,
        &mut packed,
    )
    .expect("packed batched moe block metal");

    let mut model_bytes = vec![0xA5u8; 64];
    let routed_gate_offset = model_bytes.len();
    model_bytes.extend_from_slice(&routed_gate_full);
    let routed_up_offset = model_bytes.len();
    model_bytes.extend_from_slice(&routed_up_full);
    let routed_down_offset = model_bytes.len();
    model_bytes.extend_from_slice(&routed_down_full);
    let shared_gate_offset = model_bytes.len();
    model_bytes.extend_from_slice(&shared_gate);
    let shared_up_offset = model_bytes.len();
    model_bytes.extend_from_slice(&shared_up);
    let shared_down_offset = model_bytes.len();
    model_bytes.extend_from_slice(&shared_down);

    let model_buf = ctx().new_buffer_with_bytes(&model_bytes);
    let mut indexed = vec![0.0f32; hidden];
    kernels::moe_block_batched_indexed_metal(
        ctx(),
        &model_buf,
        routed_gate_offset,
        routed_up_offset,
        routed_down_offset,
        n_experts,
        &route_ids,
        &route_weights,
        Some(shared_gate_offset),
        Some(shared_up_offset),
        Some(shared_down_offset),
        hidden,
        routed_mid,
        shared_mid,
        &x,
        &mut indexed,
    )
    .expect("indexed no-pack moe block metal");

    let diff = max_abs_diff(&packed, &indexed);
    println!("[P2] indexed no-pack moe block max abs diff = {diff:.6}");
    assert!(
        diff < ATOL,
        "indexed no-pack moe block diff {diff} >= {ATOL}"
    );
    assert_eq!(routes, route_weights.len());
}
