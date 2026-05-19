//! path-to-150 Phase L7.2 — parity gate for the mixed-quant
//! `moe_expert_pair_fused` Metal kernel (Q4_K gate + Q4_K up +
//! silu_mul + Q8_0 down) against a CPU reference.
//!
//! Reference math (per route, one expert):
//!   gate_vec[routed_mid] = W_gate_Q4K @ x[hidden_in]
//!   up_vec  [routed_mid] = W_up_Q4K   @ x[hidden_in]
//!   act     [routed_mid] = silu(gate_vec) * up_vec
//!   y       [hidden_out] = W_down_Q8 @ act
//!
//! Tolerance: 1e-3 relative — slightly looser than the standalone GEMV
//! parity tests because two chained GEMVs + a nonlinearity amplify
//! fp32-reorder noise. The fp16 quant noise floor is ~1e-3 to 5e-3
//! relative; this tolerance is meaningful (real bugs would be orders
//! of magnitude beyond it: one wrong nibble ≈ 6% relative).

#![cfg(target_os = "macos")]

use dismantle_core::metal::MetalContext;
use half::f16;

fn synthetic_q4_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    let mut bytes = vec![0u8; n_blocks * 144];
    let mut s = seed;
    let mut next_u8 = || {
        s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        ((s >> 33) & 0xFF) as u8
    };
    for b in 0..n_blocks {
        let off = b * 144;
        let d = ((next_u8() as f32) / 255.0 - 0.5) * 0.1;
        let dmin = ((next_u8() as f32) / 255.0 - 0.5) * 0.01;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
        for i in 4..16 {
            bytes[off + i] = next_u8() & 0x3F;
        }
        for i in 16..144 {
            bytes[off + i] = next_u8();
        }
    }
    bytes
}

fn synthetic_q8_0_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
    let mut bytes = vec![0u8; n_blocks * 34];
    let mut s = seed;
    let mut next_u8 = || {
        s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        ((s >> 33) & 0xFF) as u8
    };
    for b in 0..n_blocks {
        let off = b * 34;
        let d = ((next_u8() as f32) / 255.0 - 0.5) * 0.05;
        bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        for i in 0..32 {
            // signed int8 in [-128, 127]
            bytes[off + 2 + i] = next_u8();
        }
    }
    bytes
}

fn synthetic_x(n: usize, seed: u64) -> Vec<f32> {
    let mut x = vec![0.0f32; n];
    let mut s = seed;
    for v in &mut x {
        s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let bits = (s >> 33) as u32;
        *v = ((bits as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.5;
    }
    x
}

fn cpu_ref(
    w_gate_f32: &[f32],
    w_up_f32: &[f32],
    w_down_f32: &[f32],
    hidden_in: usize,
    routed_mid: usize,
    hidden_out: usize,
    x: &[f32],
) -> Vec<f32> {
    // gate_vec, up_vec, act: all length routed_mid
    let mut gate_vec = vec![0.0f32; routed_mid];
    let mut up_vec = vec![0.0f32; routed_mid];
    dismantle_core::kernels::gemv_f32(w_gate_f32, routed_mid, hidden_in, x, &mut gate_vec);
    dismantle_core::kernels::gemv_f32(w_up_f32, routed_mid, hidden_in, x, &mut up_vec);
    let act: Vec<f32> = gate_vec
        .iter()
        .zip(up_vec.iter())
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    let mut y = vec![0.0f32; hidden_out];
    dismantle_core::kernels::gemv_f32(w_down_f32, hidden_out, routed_mid, &act, &mut y);
    y
}

#[test]
fn fused_matches_cpu_ref_small_shape() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    parity_at(&ctx, 256, 64, 64, 0xDEAD_BEEF);
}

#[test]
fn fused_matches_cpu_ref_v2lite_shape() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    parity_at(&ctx, 2048, 1408, 2048, 0xCAFE_F00D);
}

fn parity_at(
    ctx: &MetalContext,
    hidden_in: usize,
    routed_mid: usize,
    hidden_out: usize,
    seed: u64,
) {
    assert_eq!(hidden_in % 256, 0);
    assert_eq!(routed_mid % 32, 0);
    assert!(routed_mid % 8 == 0, "routed_mid must be multiple of 8 for stage-A geometry");

    // Per-expert tensor sizes for ONE expert.
    let bpr_q4 = hidden_in / 256;
    let bpr_q8 = routed_mid / 32;
    let gate_block_count = routed_mid * bpr_q4;
    let up_block_count = gate_block_count;
    let down_block_count = hidden_out * bpr_q8;

    let w_gate_bytes = synthetic_q4_k_bytes(gate_block_count, seed ^ 0x1111_2222);
    let w_up_bytes = synthetic_q4_k_bytes(up_block_count, seed ^ 0x3333_4444);
    let w_down_bytes = synthetic_q8_0_bytes(down_block_count, seed ^ 0x5555_6666);
    let x = synthetic_x(hidden_in, seed ^ 0x7777_8888);

    // Pack into one buffer: [ gate | up | down ].
    let gate_offset = 0usize;
    let up_offset = gate_offset + w_gate_bytes.len();
    let down_offset = up_offset + w_up_bytes.len();
    let mut combined = Vec::with_capacity(down_offset + w_down_bytes.len());
    combined.extend_from_slice(&w_gate_bytes);
    combined.extend_from_slice(&w_up_bytes);
    combined.extend_from_slice(&w_down_bytes);
    let w_buf = ctx.new_buffer_with_bytes(&combined);

    let n_experts = 1u32;
    let n_routes = 1u32;
    let route_ids = vec![0u32];
    let route_kk = vec![0u32];
    let per_k_x = x.clone(); // K=1, so per_k_x = x

    let mut out_gpu = vec![0.0f32; hidden_out];
    dismantle_core::kernels::dispatch_moe_expert_pair_fused_pinned(
        ctx,
        &w_buf,
        gate_offset,
        up_offset,
        down_offset,
        n_experts,
        n_routes,
        hidden_in as u32,
        routed_mid as u32,
        hidden_out as u32,
        &route_ids,
        &route_kk,
        &per_k_x,
        &mut out_gpu,
    )
    .expect("fused dispatch");

    // CPU reference.
    use dismantle_core::gguf::GgmlType;
    use dismantle_core::quant::dequant_into;
    let mut w_gate_f32 = vec![0.0f32; routed_mid * hidden_in];
    let mut w_up_f32 = vec![0.0f32; routed_mid * hidden_in];
    let mut w_down_f32 = vec![0.0f32; hidden_out * routed_mid];
    dequant_into(GgmlType::Q4_K, &w_gate_bytes, &mut w_gate_f32).expect("dequant gate");
    dequant_into(GgmlType::Q4_K, &w_up_bytes, &mut w_up_f32).expect("dequant up");
    dequant_into(GgmlType::Q8_0, &w_down_bytes, &mut w_down_f32).expect("dequant down");
    let out_cpu = cpu_ref(
        &w_gate_f32, &w_up_f32, &w_down_f32, hidden_in, routed_mid, hidden_out, &x,
    );

    let max_abs = out_cpu
        .iter()
        .zip(out_gpu.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_ref = out_cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let rel = max_abs / max_ref.max(1.0);
    assert!(
        rel < 1e-3,
        "fused vs CPU-ref: hidden_in={hidden_in} routed_mid={routed_mid} hidden_out={hidden_out} \
         max_abs_diff={max_abs:.3e} max_ref={max_ref:.3e} rel={rel:.3e} (threshold=1e-3 relative)"
    );
}
