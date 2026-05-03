//! Phase 2 — strict single-launch fused MoE block parity.
//!
//! Stage 1a covers the simplest variant: ONE expert, all-Q4_K weights,
//! no top-K, no shared expert. Confirms that the workgroup-per-output-row
//! design with intermediate cached in threadgroup memory is numerically
//! correct vs the per-step reference (gemv ×3 + SwiGLU).
//!
//! Later stages of Stage B extend this file with:
//!   - Stage 1b: top-K (multiple routed experts via expert-id buffer)
//!   - Stage 1c: mixed quants (Q8_0 down) + shared expert (Q6_K down)

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

/// Mirrors `phase2_moe_block_batched_parity.rs::synthetic_q4_k_bytes`.
/// Tiny `d`/`dmin` and 8-bit unsigned `q` nibbles keep the dequanted
/// values bounded so the SwiGLU activation doesn't saturate.
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

fn one_expert_reference(
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    hidden: usize,
    mid: usize,
    x: &[f32],
) -> Vec<f32> {
    let mut gate_f32 = vec![0.0f32; mid * hidden];
    dequant_into(GgmlType::Q4_K, gate, &mut gate_f32).expect("dequant gate");
    let mut up_f32 = vec![0.0f32; mid * hidden];
    dequant_into(GgmlType::Q4_K, up, &mut up_f32).expect("dequant up");
    let mut down_f32 = vec![0.0f32; hidden * mid];
    dequant_into(GgmlType::Q4_K, down, &mut down_f32).expect("dequant down");

    let mut g = vec![0.0f32; mid];
    let mut u = vec![0.0f32; mid];
    let mut a = vec![0.0f32; mid];
    let mut out = vec![0.0f32; hidden];
    kernels::gemv_f32(&gate_f32, mid, hidden, x, &mut g);
    kernels::gemv_f32(&up_f32, mid, hidden, x, &mut u);
    kernels::silu_mul(&g, &u, &mut a);
    kernels::gemv_f32(&down_f32, hidden, mid, &a, &mut out);
    out
}

#[test]
fn test_moe_block_fused_q4_one_matches_cpu_small() {
    // Smallest 256-aligned shape that exercises the workgroup-per-row
    // design: 1 hidden block, 1 mid block.
    let hidden = 256;
    let mid = 256;
    let x = fixed_input(hidden, 0xF1A1, 0.04);
    let gate = synthetic_q4_k_bytes(mid * (hidden / 256), 0xF1B1);
    let up = synthetic_q4_k_bytes(mid * (hidden / 256), 0xF1B2);
    let down = synthetic_q4_k_bytes(hidden * (mid / 256), 0xF1B3);

    let cpu = one_expert_reference(&gate, &up, &down, hidden, mid, &x);

    let mut metal = vec![0.0f32; hidden];
    kernels::moe_block_fused_q4_one_metal(ctx(), &gate, &up, &down, hidden, mid, &x, &mut metal)
        .expect("moe_block_fused_q4_one_metal small");

    let diff = max_abs_diff(&cpu, &metal);
    println!("[stage1a] moe_block_fused_q4_one (small {hidden}x{mid}) max abs diff = {diff:.6}");
    assert!(
        diff < ATOL,
        "moe_block_fused_q4_one small diverged: {diff} >= {ATOL}"
    );
}

/// Top-K reference: full fused-table inputs, expert_ids selects per
/// iteration, accumulate weighted contributions.
#[allow(clippy::too_many_arguments)]
fn topk_reference(
    gate_full: &[u8],
    up_full: &[u8],
    down_full: &[u8],
    expert_ids: &[u32],
    weights: &[f32],
    n_experts: usize,
    hidden: usize,
    mid: usize,
    x: &[f32],
) -> Vec<f32> {
    let gate_per = mid * (hidden / 256) * 144;
    let up_per = mid * (hidden / 256) * 144;
    let down_per = hidden * (mid / 256) * 144;
    assert_eq!(gate_full.len(), n_experts * gate_per);
    assert_eq!(up_full.len(), n_experts * up_per);
    assert_eq!(down_full.len(), n_experts * down_per);

    let mut out = vec![0.0f32; hidden];
    for (k, &eid) in expert_ids.iter().enumerate() {
        let e = eid as usize;
        let gate_slice = &gate_full[e * gate_per..(e + 1) * gate_per];
        let up_slice = &up_full[e * up_per..(e + 1) * up_per];
        let down_slice = &down_full[e * down_per..(e + 1) * down_per];
        let contribution = one_expert_reference(gate_slice, up_slice, down_slice, hidden, mid, x);
        for i in 0..hidden {
            out[i] += weights[k] * contribution[i];
        }
    }
    out
}

#[test]
fn test_moe_block_fused_q4_topk_matches_cpu_small() {
    let n_experts = 4;
    let expert_ids = vec![3u32, 0, 2];
    let weights = vec![0.45f32, 0.32, 0.18];
    let hidden = 256;
    let mid = 256;
    let x = fixed_input(hidden, 0xF3A1, 0.04);

    let gate = synthetic_q4_k_bytes(n_experts * mid * (hidden / 256), 0xF3B1);
    let up = synthetic_q4_k_bytes(n_experts * mid * (hidden / 256), 0xF3B2);
    let down = synthetic_q4_k_bytes(n_experts * hidden * (mid / 256), 0xF3B3);

    let cpu = topk_reference(
        &gate,
        &up,
        &down,
        &expert_ids,
        &weights,
        n_experts,
        hidden,
        mid,
        &x,
    );

    let mut metal = vec![0.0f32; hidden];
    kernels::moe_block_fused_q4_topk_metal(
        ctx(),
        &gate,
        &up,
        &down,
        &expert_ids,
        &weights,
        n_experts,
        hidden,
        mid,
        &x,
        &mut metal,
    )
    .expect("moe_block_fused_q4_topk_metal small");

    let diff = max_abs_diff(&cpu, &metal);
    println!(
        "[stage1b] moe_block_fused_q4_topk (small {hidden}x{mid}, top_k={}) max abs diff = {diff:.6}",
        expert_ids.len()
    );
    assert!(
        diff < ATOL,
        "moe_block_fused_q4_topk small diverged: {diff} >= {ATOL}"
    );
}

#[test]
fn test_moe_block_fused_q4_topk_matches_cpu_rectangular() {
    let n_experts = 6;
    let expert_ids = vec![5u32, 1, 4, 0, 3, 2]; // full top-6 like DeepSeek
    let weights = vec![0.30f32, 0.22, 0.18, 0.12, 0.10, 0.08];
    let hidden = 512;
    let mid = 768;
    let x = fixed_input(hidden, 0xF4A1, 0.03);

    let gate = synthetic_q4_k_bytes(n_experts * mid * (hidden / 256), 0xF4B1);
    let up = synthetic_q4_k_bytes(n_experts * mid * (hidden / 256), 0xF4B2);
    let down = synthetic_q4_k_bytes(n_experts * hidden * (mid / 256), 0xF4B3);

    let cpu = topk_reference(
        &gate,
        &up,
        &down,
        &expert_ids,
        &weights,
        n_experts,
        hidden,
        mid,
        &x,
    );

    let mut metal = vec![0.0f32; hidden];
    kernels::moe_block_fused_q4_topk_metal(
        ctx(),
        &gate,
        &up,
        &down,
        &expert_ids,
        &weights,
        n_experts,
        hidden,
        mid,
        &x,
        &mut metal,
    )
    .expect("moe_block_fused_q4_topk_metal rectangular");

    let diff = max_abs_diff(&cpu, &metal);
    println!(
        "[stage1b] moe_block_fused_q4_topk ({hidden}x{mid}, top_k={}) max abs diff = {diff:.6}",
        expert_ids.len()
    );
    assert!(
        diff < ATOL,
        "moe_block_fused_q4_topk rectangular diverged: {diff} >= {ATOL}"
    );
}

#[test]
fn test_moe_block_fused_q4_one_matches_cpu_rectangular() {
    // mid != hidden — different block counts on the two stages.
    // Picks shapes representative of DeepSeek-V2-Lite's MoE
    // (hidden=2048, moe_intermediate=1408) but smaller for fast test.
    let hidden = 512;
    let mid = 768; // 3 blocks × 256
    let x = fixed_input(hidden, 0xF2A1, 0.03);
    let gate = synthetic_q4_k_bytes(mid * (hidden / 256), 0xF2B1);
    let up = synthetic_q4_k_bytes(mid * (hidden / 256), 0xF2B2);
    let down = synthetic_q4_k_bytes(hidden * (mid / 256), 0xF2B3);

    let cpu = one_expert_reference(&gate, &up, &down, hidden, mid, &x);

    let mut metal = vec![0.0f32; hidden];
    kernels::moe_block_fused_q4_one_metal(ctx(), &gate, &up, &down, hidden, mid, &x, &mut metal)
        .expect("moe_block_fused_q4_one_metal rectangular");

    let diff = max_abs_diff(&cpu, &metal);
    println!("[stage1a] moe_block_fused_q4_one ({hidden}x{mid}) max abs diff = {diff:.6}");
    assert!(
        diff < ATOL,
        "moe_block_fused_q4_one rectangular diverged: {diff} >= {ATOL}"
    );
}

// ---- Stage 1c helpers -------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn v2lite_reference(
    routed_gate: &[u8],
    routed_up: &[u8],
    routed_down: &[u8],
    shared_gate: &[u8],
    shared_up: &[u8],
    shared_down: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    n_experts: usize,
    hidden: usize,
    routed_mid: usize,
    shared_mid: usize,
    x: &[f32],
) -> Vec<f32> {
    let routed_gate_per = routed_mid * (hidden / 256) * 144;
    let routed_up_per = routed_mid * (hidden / 256) * 144;
    let routed_down_per = hidden * (routed_mid / 32) * 34;

    let mut out = vec![0.0f32; hidden];

    // Routed experts: Q4_K gate/up, Q8_0 down.
    for (k, &eid) in expert_ids.iter().enumerate() {
        let e = eid as usize;
        let gate_s = &routed_gate[e * routed_gate_per..(e + 1) * routed_gate_per];
        let up_s = &routed_up[e * routed_up_per..(e + 1) * routed_up_per];
        let down_s = &routed_down[e * routed_down_per..(e + 1) * routed_down_per];

        let mut gate_f32 = vec![0.0f32; routed_mid * hidden];
        dequant_into(GgmlType::Q4_K, gate_s, &mut gate_f32).expect("dequant routed gate");
        let mut up_f32 = vec![0.0f32; routed_mid * hidden];
        dequant_into(GgmlType::Q4_K, up_s, &mut up_f32).expect("dequant routed up");
        let mut down_f32 = vec![0.0f32; hidden * routed_mid];
        dequant_into(GgmlType::Q8_0, down_s, &mut down_f32).expect("dequant routed down");

        let mut g = vec![0.0f32; routed_mid];
        let mut u = vec![0.0f32; routed_mid];
        let mut a = vec![0.0f32; routed_mid];
        let mut contrib = vec![0.0f32; hidden];
        kernels::gemv_f32(&gate_f32, routed_mid, hidden, x, &mut g);
        kernels::gemv_f32(&up_f32, routed_mid, hidden, x, &mut u);
        kernels::silu_mul(&g, &u, &mut a);
        kernels::gemv_f32(&down_f32, hidden, routed_mid, &a, &mut contrib);
        for i in 0..hidden {
            out[i] += route_weights[k] * contrib[i];
        }
    }
    let _ = n_experts; // validated by caller

    // Shared expert: Q4_K gate/up, Q6_K down.
    let mut sg_f32 = vec![0.0f32; shared_mid * hidden];
    dequant_into(GgmlType::Q4_K, shared_gate, &mut sg_f32).expect("dequant shared gate");
    let mut su_f32 = vec![0.0f32; shared_mid * hidden];
    dequant_into(GgmlType::Q4_K, shared_up, &mut su_f32).expect("dequant shared up");
    let mut sd_f32 = vec![0.0f32; hidden * shared_mid];
    dequant_into(GgmlType::Q6_K, shared_down, &mut sd_f32).expect("dequant shared down");

    let mut sg = vec![0.0f32; shared_mid];
    let mut su = vec![0.0f32; shared_mid];
    let mut sa = vec![0.0f32; shared_mid];
    let mut scontrib = vec![0.0f32; hidden];
    kernels::gemv_f32(&sg_f32, shared_mid, hidden, x, &mut sg);
    kernels::gemv_f32(&su_f32, shared_mid, hidden, x, &mut su);
    kernels::silu_mul(&sg, &su, &mut sa);
    kernels::gemv_f32(&sd_f32, hidden, shared_mid, &sa, &mut scontrib);
    for i in 0..hidden {
        out[i] += scontrib[i];
    }

    out
}

#[test]
fn test_moe_block_fused_v2lite_matches_cpu() {
    let n_experts = 4;
    let expert_ids = vec![2u32, 0, 3];
    let weights = vec![0.45f32, 0.32, 0.19];
    let hidden = 256;
    let routed_mid = 256;
    let shared_mid = 256;

    let x = fixed_input(hidden, 0xF5A1, 0.04);

    let routed_gate = synthetic_q4_k_bytes(n_experts * routed_mid * (hidden / 256), 0xF5B1);
    let routed_up = synthetic_q4_k_bytes(n_experts * routed_mid * (hidden / 256), 0xF5B2);
    let routed_down = synthetic_q8_0_bytes(n_experts * hidden * (routed_mid / 32), 0xF5B3);
    let shared_gate = synthetic_q4_k_bytes(shared_mid * (hidden / 256), 0xF5C1);
    let shared_up = synthetic_q4_k_bytes(shared_mid * (hidden / 256), 0xF5C2);
    let shared_down = synthetic_q6_k_bytes(hidden * (shared_mid / 256), 0xF5C3);

    let cpu = v2lite_reference(
        &routed_gate,
        &routed_up,
        &routed_down,
        &shared_gate,
        &shared_up,
        &shared_down,
        &expert_ids,
        &weights,
        n_experts,
        hidden,
        routed_mid,
        shared_mid,
        &x,
    );

    let mut metal = vec![0.0f32; hidden];
    kernels::moe_block_fused_v2lite_metal(
        ctx(),
        &routed_gate,
        &routed_up,
        &routed_down,
        &shared_gate,
        &shared_up,
        &shared_down,
        &expert_ids,
        &weights,
        n_experts,
        hidden,
        routed_mid,
        shared_mid,
        &x,
        &mut metal,
    )
    .expect("moe_block_fused_v2lite_metal");

    let diff = max_abs_diff(&cpu, &metal);
    println!(
        "[stage1c] moe_block_fused_v2lite ({hidden}x{routed_mid}/{shared_mid}, top_k={}) \
         max abs diff = {diff:.6}",
        expert_ids.len()
    );
    assert!(
        diff < ATOL,
        "moe_block_fused_v2lite diverged: {diff} >= {ATOL}"
    );
}

#[test]
fn test_moe_block_fused_v2lite_indexed_matches_slice() {
    let n_experts = 4;
    let expert_ids = vec![2u32, 0, 3];
    let weights = vec![0.45f32, 0.32, 0.19];
    let hidden = 256;
    let routed_mid = 256;
    let shared_mid = 256;

    let x = fixed_input(hidden, 0xF5A1, 0.04);

    let routed_gate = synthetic_q4_k_bytes(n_experts * routed_mid * (hidden / 256), 0xF5B1);
    let routed_up = synthetic_q4_k_bytes(n_experts * routed_mid * (hidden / 256), 0xF5B2);
    let routed_down = synthetic_q8_0_bytes(n_experts * hidden * (routed_mid / 32), 0xF5B3);
    let shared_gate = synthetic_q4_k_bytes(shared_mid * (hidden / 256), 0xF5C1);
    let shared_up = synthetic_q4_k_bytes(shared_mid * (hidden / 256), 0xF5C2);
    let shared_down = synthetic_q6_k_bytes(hidden * (shared_mid / 256), 0xF5C3);

    // Build a flat model buffer with a header sentinel to ensure offsets are respected.
    let mut model_bytes = vec![0xA5u8; 64];
    let routed_gate_off = model_bytes.len();
    model_bytes.extend_from_slice(&routed_gate);
    let routed_up_off = model_bytes.len();
    model_bytes.extend_from_slice(&routed_up);
    let routed_down_off = model_bytes.len();
    model_bytes.extend_from_slice(&routed_down);
    let shared_gate_off = model_bytes.len();
    model_bytes.extend_from_slice(&shared_gate);
    let shared_up_off = model_bytes.len();
    model_bytes.extend_from_slice(&shared_up);
    let shared_down_off = model_bytes.len();
    model_bytes.extend_from_slice(&shared_down);

    let model_buf = ctx().new_buffer_with_bytes(&model_bytes);

    // Slice-based reference (already attested vs CPU above).
    let mut reference = vec![0.0f32; hidden];
    kernels::moe_block_fused_v2lite_metal(
        ctx(),
        &routed_gate,
        &routed_up,
        &routed_down,
        &shared_gate,
        &shared_up,
        &shared_down,
        &expert_ids,
        &weights,
        n_experts,
        hidden,
        routed_mid,
        shared_mid,
        &x,
        &mut reference,
    )
    .expect("moe_block_fused_v2lite_metal");

    let mut indexed = vec![0.0f32; hidden];
    kernels::moe_block_fused_v2lite_indexed_metal(
        ctx(),
        &model_buf,
        routed_gate_off,
        routed_up_off,
        routed_down_off,
        shared_gate_off,
        shared_up_off,
        shared_down_off,
        &expert_ids,
        &weights,
        n_experts,
        hidden,
        routed_mid,
        shared_mid,
        &x,
        &mut indexed,
    )
    .expect("moe_block_fused_v2lite_indexed_metal");

    let diff = max_abs_diff(&reference, &indexed);
    println!("[stage_b4] moe_block_fused_v2lite_indexed vs slice max abs diff = {diff:.6}");
    assert!(
        diff < ATOL,
        "moe_block_fused_v2lite_indexed diverged from slice: {diff} >= {ATOL}"
    );
}

// ---- Wedge 2: two-stage parity -------------------------------------------------

#[test]
fn test_moe_block_two_stage_matches_cpu() {
    let n_experts = 4;
    let n_shared = 1usize;
    let expert_ids = vec![2u32, 0, 3];
    let weights = vec![0.45f32, 0.32, 0.19];
    let hidden = 256;
    let routed_mid = 256;
    let shared_mid = 256;

    let x = fixed_input(hidden, 0xF6A1, 0.04);

    let routed_gate = synthetic_q4_k_bytes(n_experts * routed_mid * (hidden / 256), 0xF6B1);
    let routed_up = synthetic_q4_k_bytes(n_experts * routed_mid * (hidden / 256), 0xF6B2);
    let routed_down = synthetic_q8_0_bytes(n_experts * hidden * (routed_mid / 32), 0xF6B3);
    let shared_gate = synthetic_q4_k_bytes(n_shared * shared_mid * (hidden / 256), 0xF6C1);
    let shared_up = synthetic_q4_k_bytes(n_shared * shared_mid * (hidden / 256), 0xF6C2);
    let shared_down = synthetic_q6_k_bytes(n_shared * hidden * (shared_mid / 256), 0xF6C3);

    let cpu = v2lite_reference(
        &routed_gate,
        &routed_up,
        &routed_down,
        &shared_gate,
        &shared_up,
        &shared_down,
        &expert_ids,
        &weights,
        n_experts,
        hidden,
        routed_mid,
        shared_mid,
        &x,
    );

    let mut metal = vec![0.0f32; hidden];
    kernels::moe_block_two_stage_metal(
        ctx(),
        &routed_gate,
        &routed_up,
        &routed_down,
        &shared_gate,
        &shared_up,
        &shared_down,
        &expert_ids,
        &weights,
        n_experts,
        n_shared,
        hidden,
        routed_mid,
        shared_mid,
        &x,
        &mut metal,
    )
    .expect("moe_block_two_stage_metal");

    let diff = max_abs_diff(&cpu, &metal);
    println!(
        "[wedge2] moe_block_two_stage ({hidden}x{routed_mid}/{shared_mid}, top_k={}, n_shared={n_shared}) \
         max abs diff = {diff:.6}",
        expert_ids.len()
    );
    assert!(
        diff < ATOL,
        "moe_block_two_stage diverged: {diff} >= {ATOL}"
    );
}
