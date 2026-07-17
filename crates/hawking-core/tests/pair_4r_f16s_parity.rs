#![cfg(target_os = "macos")]
//! Track D4 parity: gemm_q4_k_v4_predec_pair_4r_f16s must produce
//! rel_L2 < 1% vs the pair_4r f32-scales reference kernel.
//!
//! pair_4r_f16s = pair_4r geometry (32 rows/TG) + half* scale reads.
//! For gate+up (11008 rows × 2048 cols): 344 TGs vs 688 (pair_4r_f32)
//! or 1376 (pair_f16s 1r). Both bandwidth and TG scheduling savings.

use half::f16;
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, TokenCommandBuffer};

mod common;
use common::*;

fn make_q4k_predec(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let bpr = cols / 256;
    let w: Vec<u8> = (0..rows * bpr * 144)
        .map(|i| ((i as u32).wrapping_mul(2246822519).wrapping_add(seed)) as u8)
        .collect();
    // Avoid near-zero scales: [0.1, 2.0] so f16 rounding is controlled.
    let s: Vec<f32> = (0..rows * bpr * 16)
        .map(|i| {
            let v = ((i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(seed ^ 0xAB)) as f32
                / u32::MAX as f32;
            0.1 + v * 1.9
        })
        .collect();
    (w, s)
}

fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| f16::from_f32(x).to_le_bytes())
        .collect()
}

fn rel_l2(reference: &[f32], got: &[f32]) -> f64 {
    let num: f64 = reference
        .iter()
        .zip(got)
        .map(|(&r, &g)| ((r - g) as f64).powi(2))
        .sum();
    let den: f64 = reference
        .iter()
        .map(|&r| (r as f64).powi(2))
        .sum::<f64>()
        .max(1e-30);
    (num / den).sqrt()
}

/// Run pair_4r (f32 scales) → (gate_out, up_out).
fn run_f32_4r(
    ctx: &MetalContext,
    model: &[u8],
    g_off: usize,
    g_len: usize,
    u_off: usize,
    u_len: usize,
    g_scales: &[f32],
    u_scales: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>) {
    let model_buf = ctx.new_buffer_with_bytes(model);
    let gs_buf = new_f32_buf(ctx, g_scales);
    let us_buf = new_f32_buf(ctx, u_scales);
    let x_buf = new_f32_buf(ctx, x);
    let g_out = ctx.new_buffer(rows * 4);
    let u_out = ctx.new_buffer(rows * 4);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4_k_v4_predec_pair_4r_pinned_tcb(
        &mut tcb, &model_buf, g_off, g_len, &gs_buf, 0, u_off, u_len, &us_buf, 0, rows, cols,
        &x_buf, &g_out, &u_out,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    (read_f32_buf(&g_out, rows), read_f32_buf(&u_out, rows))
}

/// Run pair_4r_f16s (half scales) → (gate_out, up_out).
fn run_f16s_4r(
    ctx: &MetalContext,
    model: &[u8],
    g_off: usize,
    g_len: usize,
    u_off: usize,
    u_len: usize,
    g_scales_f16: &[u8],
    u_scales_f16: &[u8],
    x: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<f32>, Vec<f32>) {
    let model_buf = ctx.new_buffer_with_bytes(model);
    let gs_buf = ctx.new_buffer_with_bytes(g_scales_f16);
    let us_buf = ctx.new_buffer_with_bytes(u_scales_f16);
    let x_buf = new_f32_buf(ctx, x);
    let g_out = ctx.new_buffer(rows * 4);
    let u_out = ctx.new_buffer(rows * 4);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4_k_v4_predec_pair_4r_f16s_pinned_tcb(
        &mut tcb, &model_buf, g_off, g_len, &gs_buf, 0, u_off, u_len, &us_buf, 0, rows, cols,
        &x_buf, &g_out, &u_out,
    )
    .unwrap();
    tcb.commit_and_wait().unwrap();
    (read_f32_buf(&g_out, rows), read_f32_buf(&u_out, rows))
}

/// Quality gate: pair_4r_f16s rel_L2 < 1% vs pair_4r_f32.
/// Production shapes: gate+up (11008 × 2048) and K/V-like (1024 × 2048).
#[test]
fn pair_4r_f16s_rel_l2_quality_gate() {
    let ctx = ctx();
    const MAX_REL_L2: f64 = 1e-2;

    let cases: &[(usize, usize, u32)] = &[
        // gate+up production shape
        (11008, 2048, 0xD40A),
        (11008, 2048, 0xD40B),
        // KV-pair-like shapes
        (1024, 2048, 0xD40C),
        (2048, 2048, 0xD40D),
    ];

    for &(rows, cols, seed) in cases {
        let (g_w, g_sc) = make_q4k_predec(rows, cols, seed);
        let (u_w, u_sc) = make_q4k_predec(rows, cols, seed ^ 0x10);
        let g_sc_f16 = f32_to_f16_bytes(&g_sc);
        let u_sc_f16 = f32_to_f16_bytes(&u_sc);
        let model = [g_w.as_slice(), u_w.as_slice()].concat();
        let g_off = 0;
        let u_off = g_w.len();
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                ((i as u32).wrapping_mul(1664525).wrapping_add(seed) as f32 / u32::MAX as f32) * 2.0
                    - 1.0
            })
            .collect();

        let (ref_g, ref_u) = run_f32_4r(
            ctx,
            &model,
            g_off,
            g_w.len(),
            u_off,
            u_w.len(),
            &g_sc,
            &u_sc,
            &x,
            rows,
            cols,
        );
        let (got_g, got_u) = run_f16s_4r(
            ctx,
            &model,
            g_off,
            g_w.len(),
            u_off,
            u_w.len(),
            &g_sc_f16,
            &u_sc_f16,
            &x,
            rows,
            cols,
        );

        let rg = rel_l2(&ref_g, &got_g);
        let ru = rel_l2(&ref_u, &got_u);
        assert!(
            rg < MAX_REL_L2,
            "rows={rows} cols={cols}: gate rel_L2={rg:.4e} >= {MAX_REL_L2:.4e}"
        );
        assert!(
            ru < MAX_REL_L2,
            "rows={rows} cols={cols}: up   rel_L2={ru:.4e} >= {MAX_REL_L2:.4e}"
        );
        eprintln!("D4 4r_f16s rows={rows} cols={cols}: gate={rg:.2e} up={ru:.2e} OK");
    }
}
