//! path-to-150 Phase L7 / Stage 0.5 — parity gate for the
//! `gemm_q4_k_m_v3_xtg_sumy` Metal kernel (v3_xtg + min-correction
//! sumy trick).
//!
//! The sumy variant rearranges the Q4_K_M GEMV math: the
//! per-sub-block `dm[k] * xl[k]` term is hoisted out of the nibble
//! loop and accumulated as `dm[k] * simd_sum(xl[k])` instead. The
//! reordering changes the floating-point accumulation order but
//! preserves the mathematical result to within fp32 ULP noise.
//!
//! Tolerance: same 1e-5 relative bound as `q4_k_v3_xtg_parity.rs`.

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

fn synthetic_x(cols: usize, seed: u64) -> Vec<f32> {
    let mut x = vec![0.0f32; cols];
    let mut s = seed;
    for v in &mut x {
        s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let bits = (s >> 33) as u32;
        *v = ((bits as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.5;
    }
    x
}

#[test]
fn v3_xtg_sumy_matches_cpu_ref_basic_shape() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    parity_at(&ctx, 16, 256, 0xDEAD_BEEF);
    parity_at(&ctx, 64, 512, 0xCAFE_F00D);
}

/// V2-Lite expert-projection shape: rows=10944 cols=2048.
#[test]
fn v3_xtg_sumy_matches_cpu_ref_v2lite_expert_shape() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    parity_at(&ctx, 10944, 2048, 0xABCD_1234);
}

/// LM head shape — the primary L7 target (102400 rows × 2048 cols),
/// sliced to 4096 rows for test runtime.
#[test]
fn v3_xtg_sumy_matches_cpu_ref_lm_head_shape() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    parity_at(&ctx, 4096, 2048, 0xFEED_BEEF);
}

/// Cross-check: v3_xtg_sumy must agree with v3_xtg, since both are
/// re-orderings of the same math. The reordering is exact-equivalent
/// in real arithmetic; in fp32 the result lands within a few ULP.
#[test]
fn v3_xtg_sumy_matches_v3_xtg_v2lite_expert_shape() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };
    cross_check_at(&ctx, 10944, 2048, 0xBEEF_DEAD);
}

fn parity_at(ctx: &MetalContext, rows: usize, cols: usize, seed: u64) {
    assert_eq!(cols % 256, 0);
    let n_blocks = rows * (cols / 256);
    let w_bytes = synthetic_q4_k_bytes(n_blocks, seed ^ 0xA5A5_5A5A);
    let x = synthetic_x(cols, seed ^ 0x1234_5678);

    let w_buf = ctx.new_buffer_with_bytes(&w_bytes);

    let mut out_sumy = vec![0.0f32; rows];
    let mut out_ref = vec![0.0f32; rows];

    dismantle_core::kernels::dispatch_q4_k_m_v3_xtg_sumy_pinned(
        ctx, &w_buf, 0, w_bytes.len(), rows, cols, &x, &mut out_sumy,
    )
    .expect("xtg_sumy dispatch");

    use dismantle_core::gguf::GgmlType;
    use dismantle_core::quant::dequant_into;
    let mut w_f32 = vec![0.0f32; rows * cols];
    dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32).expect("dequant");
    dismantle_core::kernels::gemv_f32(&w_f32, rows, cols, &x, &mut out_ref);

    let max_abs = out_ref
        .iter()
        .zip(out_sumy.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_ref = out_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let rel = max_abs / max_ref.max(1.0);
    assert!(
        rel < 1e-5,
        "v3_xtg_sumy vs CPU-ref: rows={rows} cols={cols} max_abs_diff={max_abs:.3e} \
         max_ref={max_ref:.3e} rel={rel:.3e} (threshold=1e-5 relative)"
    );
}

fn cross_check_at(ctx: &MetalContext, rows: usize, cols: usize, seed: u64) {
    assert_eq!(cols % 256, 0);
    let n_blocks = rows * (cols / 256);
    let w_bytes = synthetic_q4_k_bytes(n_blocks, seed ^ 0xA5A5_5A5A);
    let x = synthetic_x(cols, seed ^ 0x1234_5678);

    let w_buf = ctx.new_buffer_with_bytes(&w_bytes);

    let mut out_sumy = vec![0.0f32; rows];
    let mut out_xtg = vec![0.0f32; rows];

    dismantle_core::kernels::dispatch_q4_k_m_v3_xtg_sumy_pinned(
        ctx, &w_buf, 0, w_bytes.len(), rows, cols, &x, &mut out_sumy,
    )
    .expect("xtg_sumy dispatch");
    dismantle_core::kernels::dispatch_q4_k_m_v3_xtg_pinned(
        ctx, &w_buf, 0, w_bytes.len(), rows, cols, &x, &mut out_xtg,
    )
    .expect("xtg dispatch");

    let max_abs = out_xtg
        .iter()
        .zip(out_sumy.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_ref = out_xtg.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let rel = max_abs / max_ref.max(1.0);
    assert!(
        rel < 1e-5,
        "v3_xtg_sumy vs v3_xtg: rows={rows} cols={cols} max_abs_diff={max_abs:.3e} \
         max_ref={max_ref:.3e} rel={rel:.3e} (threshold=1e-5 relative)"
    );
}
