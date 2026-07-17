#![cfg(target_os = "macos")]
//! Track D3 parity: QKV rope-append f16-scales variants must produce rel_L2
//! < 1% vs the f32-scales reference kernels.
//!
//! Tests cover both the 2r variant (gemm_q4k_predec_qkv_rope_append_f16s)
//! and the 4r variant (gemm_q4k_predec_qkv_rope_append_4r_f16s), across
//! production-like shapes (cols=2048, ≥8 blocks/row).
//!
//! The f16 scale rounding introduces ~5e-4 relative error per multiply;
//! this averages down with sufficient blocks. We gate on rel_L2 < 1e-2
//! (same bar as pair_f16s and swiglu_f16s).

use half::f16;
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, TokenCommandBuffer};

mod common;
use common::*;

/// Build random Q4_K weights and f32 predecoded scale table.
fn make_q4k_predec(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let bpr = cols / 256;
    let total_w = rows * bpr * 144;
    let w: Vec<u8> = (0..total_w)
        .map(|i| ((i as u32).wrapping_mul(2246822519u32).wrapping_add(seed)) as u8)
        .collect();
    let ns = rows * bpr * 16;
    // Avoid near-zero scales: generate in [0.1, 2.0] so f16 rounding is benign.
    let s: Vec<f32> = (0..ns)
        .map(|i| {
            let v = ((i as u32)
                .wrapping_mul(2654435761u32)
                .wrapping_add(seed ^ 0xAB)) as f32
                / u32::MAX as f32;
            0.1 + v * 1.9
        })
        .collect();
    (w, s)
}

/// Convert f32 scale table to packed f16 bytes.
fn f32_to_f16_scales(scales: &[f32]) -> Vec<u8> {
    scales
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect()
}

/// Relative L2 error = ||ref - got|| / ||ref||.
fn rel_l2(reference: &[f32], got: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&r, &g) in reference.iter().zip(got) {
        let d = (r - g) as f64;
        num += d * d;
        den += (r as f64) * (r as f64);
    }
    (num / den.max(1e-30)).sqrt()
}

struct Shape {
    n_q: usize,
    n_k: usize,
    hd: usize,
    cols: usize,
    pos: u32,
    kv_off: usize,
}

struct Q4kWeights {
    q_w: Vec<u8>,
    q_sc_f32: Vec<f32>,
    q_sc_f16: Vec<u8>,
    k_w: Vec<u8>,
    k_sc_f32: Vec<f32>,
    k_sc_f16: Vec<u8>,
    v_w: Vec<u8>,
    v_sc_f32: Vec<f32>,
    v_sc_f16: Vec<u8>,
}

fn make_weights(s: &Shape, seed: u32) -> Q4kWeights {
    let q_rows = s.n_q * s.hd;
    let kv_rows = s.n_k * s.hd;
    let (q_w, q_sc_f32) = make_q4k_predec(q_rows, s.cols, seed);
    let (k_w, k_sc_f32) = make_q4k_predec(kv_rows, s.cols, seed ^ 0x10);
    let (v_w, v_sc_f32) = make_q4k_predec(kv_rows, s.cols, seed ^ 0x20);
    let q_sc_f16 = f32_to_f16_scales(&q_sc_f32);
    let k_sc_f16 = f32_to_f16_scales(&k_sc_f32);
    let v_sc_f16 = f32_to_f16_scales(&v_sc_f32);
    Q4kWeights {
        q_w,
        q_sc_f32,
        q_sc_f16,
        k_w,
        k_sc_f32,
        k_sc_f16,
        v_w,
        v_sc_f32,
        v_sc_f16,
    }
}

/// Run f32-scales 2r kernel → (q_out, k_cache_slice, v_cache_slice).
fn run_f32_2r(
    ctx: &MetalContext,
    shape: &Shape,
    w: &Q4kWeights,
    x: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_rows = shape.n_q * shape.hd;
    let kv_rows = shape.n_k * shape.hd;
    let model_bytes = [&w.q_w[..], &w.k_w[..], &w.v_w[..]].concat();
    let q_off = 0;
    let k_off = w.q_w.len();
    let v_off = w.q_w.len() + w.k_w.len();
    let model = ctx.new_buffer_with_bytes(&model_bytes);
    let q_sc = new_f32_buf(ctx, &w.q_sc_f32);
    let k_sc = new_f32_buf(ctx, &w.k_sc_f32);
    let v_sc = new_f32_buf(ctx, &w.v_sc_f32);
    let x_buf = new_f32_buf(ctx, x);
    let q_buf = ctx.new_buffer(q_rows * 4);
    let cache_len = shape.kv_off + kv_rows + 8;
    let k_cache = new_f32_buf(ctx, &vec![0.0f32; cache_len]);
    let v_cache = new_f32_buf(ctx, &vec![0.0f32; cache_len]);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4k_predec_qkv_rope_append_pinned_tcb(
        &mut tcb,
        &model,
        q_off,
        w.q_w.len(),
        &q_sc,
        k_off,
        w.k_w.len(),
        &k_sc,
        v_off,
        w.v_w.len(),
        &v_sc,
        q_rows,
        kv_rows,
        shape.cols,
        shape.n_q,
        shape.n_k,
        shape.hd,
        shape.pos,
        10000.0,
        shape.kv_off,
        &x_buf,
        &q_buf,
        None,
        None,
        None,
        &k_cache,
        &v_cache,
    )
    .expect("f32 2r");
    tcb.commit_and_wait().expect("f32 2r commit");
    let q = read_f32_buf(&q_buf, q_rows);
    let k = read_f32_buf(&k_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
    let v = read_f32_buf(&v_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
    (q, k, v)
}

/// Run f16-scales 2r kernel → (q_out, k_cache_slice, v_cache_slice).
fn run_f16s_2r(
    ctx: &MetalContext,
    shape: &Shape,
    w: &Q4kWeights,
    x: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_rows = shape.n_q * shape.hd;
    let kv_rows = shape.n_k * shape.hd;
    let model_bytes = [&w.q_w[..], &w.k_w[..], &w.v_w[..]].concat();
    let q_off = 0;
    let k_off = w.q_w.len();
    let v_off = w.q_w.len() + w.k_w.len();
    let model = ctx.new_buffer_with_bytes(&model_bytes);
    let q_sc = ctx.new_buffer_with_bytes(&w.q_sc_f16);
    let k_sc = ctx.new_buffer_with_bytes(&w.k_sc_f16);
    let v_sc = ctx.new_buffer_with_bytes(&w.v_sc_f16);
    let x_buf = new_f32_buf(ctx, x);
    let q_buf = ctx.new_buffer(q_rows * 4);
    let cache_len = shape.kv_off + kv_rows + 8;
    let k_cache = new_f32_buf(ctx, &vec![0.0f32; cache_len]);
    let v_cache = new_f32_buf(ctx, &vec![0.0f32; cache_len]);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4k_predec_qkv_rope_append_f16s_pinned_tcb(
        &mut tcb,
        &model,
        q_off,
        w.q_w.len(),
        &q_sc,
        k_off,
        w.k_w.len(),
        &k_sc,
        v_off,
        w.v_w.len(),
        &v_sc,
        q_rows,
        kv_rows,
        shape.cols,
        shape.n_q,
        shape.n_k,
        shape.hd,
        shape.pos,
        10000.0,
        shape.kv_off,
        &x_buf,
        &q_buf,
        None,
        None,
        None,
        &k_cache,
        &v_cache,
    )
    .expect("f16s 2r");
    tcb.commit_and_wait().expect("f16s 2r commit");
    let q = read_f32_buf(&q_buf, q_rows);
    let k = read_f32_buf(&k_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
    let v = read_f32_buf(&v_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
    (q, k, v)
}

/// Run f32-scales 4r kernel.
fn run_f32_4r(
    ctx: &MetalContext,
    shape: &Shape,
    w: &Q4kWeights,
    x: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_rows = shape.n_q * shape.hd;
    let kv_rows = shape.n_k * shape.hd;
    let model_bytes = [&w.q_w[..], &w.k_w[..], &w.v_w[..]].concat();
    let q_off = 0;
    let k_off = w.q_w.len();
    let v_off = w.q_w.len() + w.k_w.len();
    let model = ctx.new_buffer_with_bytes(&model_bytes);
    let q_sc = new_f32_buf(ctx, &w.q_sc_f32);
    let k_sc = new_f32_buf(ctx, &w.k_sc_f32);
    let v_sc = new_f32_buf(ctx, &w.v_sc_f32);
    let x_buf = new_f32_buf(ctx, x);
    let q_buf = ctx.new_buffer(q_rows * 4);
    let cache_len = shape.kv_off + kv_rows + 8;
    let k_cache = new_f32_buf(ctx, &vec![0.0f32; cache_len]);
    let v_cache = new_f32_buf(ctx, &vec![0.0f32; cache_len]);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4k_predec_qkv_rope_append_4r_pinned_tcb(
        &mut tcb,
        &model,
        q_off,
        w.q_w.len(),
        &q_sc,
        k_off,
        w.k_w.len(),
        &k_sc,
        v_off,
        w.v_w.len(),
        &v_sc,
        q_rows,
        kv_rows,
        shape.cols,
        shape.n_q,
        shape.n_k,
        shape.hd,
        shape.pos,
        10000.0,
        shape.kv_off,
        &x_buf,
        &q_buf,
        None,
        None,
        None,
        &k_cache,
        &v_cache,
    )
    .expect("f32 4r");
    tcb.commit_and_wait().expect("f32 4r commit");
    let q = read_f32_buf(&q_buf, q_rows);
    let k = read_f32_buf(&k_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
    let v = read_f32_buf(&v_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
    (q, k, v)
}

/// Run f16-scales 4r kernel.
fn run_f16s_4r(
    ctx: &MetalContext,
    shape: &Shape,
    w: &Q4kWeights,
    x: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_rows = shape.n_q * shape.hd;
    let kv_rows = shape.n_k * shape.hd;
    let model_bytes = [&w.q_w[..], &w.k_w[..], &w.v_w[..]].concat();
    let q_off = 0;
    let k_off = w.q_w.len();
    let v_off = w.q_w.len() + w.k_w.len();
    let model = ctx.new_buffer_with_bytes(&model_bytes);
    let q_sc = ctx.new_buffer_with_bytes(&w.q_sc_f16);
    let k_sc = ctx.new_buffer_with_bytes(&w.k_sc_f16);
    let v_sc = ctx.new_buffer_with_bytes(&w.v_sc_f16);
    let x_buf = new_f32_buf(ctx, x);
    let q_buf = ctx.new_buffer(q_rows * 4);
    let cache_len = shape.kv_off + kv_rows + 8;
    let k_cache = new_f32_buf(ctx, &vec![0.0f32; cache_len]);
    let v_cache = new_f32_buf(ctx, &vec![0.0f32; cache_len]);
    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4k_predec_qkv_rope_append_4r_f16s_pinned_tcb(
        &mut tcb,
        &model,
        q_off,
        w.q_w.len(),
        &q_sc,
        k_off,
        w.k_w.len(),
        &k_sc,
        v_off,
        w.v_w.len(),
        &v_sc,
        q_rows,
        kv_rows,
        shape.cols,
        shape.n_q,
        shape.n_k,
        shape.hd,
        shape.pos,
        10000.0,
        shape.kv_off,
        &x_buf,
        &q_buf,
        None,
        None,
        None,
        &k_cache,
        &v_cache,
    )
    .expect("f16s 4r");
    tcb.commit_and_wait().expect("f16s 4r commit");
    let q = read_f32_buf(&q_buf, q_rows);
    let k = read_f32_buf(&k_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
    let v = read_f32_buf(&v_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
    (q, k, v)
}

/// Check (q, k, v) rel_L2 against a reference triple.
fn check_rel_l2(
    label: &str,
    ref_q: &[f32],
    ref_k: &[f32],
    ref_v: &[f32],
    got_q: &[f32],
    got_k: &[f32],
    got_v: &[f32],
) {
    const MAX_REL_L2: f64 = 1e-2;
    let rq = rel_l2(ref_q, got_q);
    let rk = rel_l2(ref_k, got_k);
    let rv = rel_l2(ref_v, got_v);
    assert!(
        rq < MAX_REL_L2,
        "{label} Q: rel_L2={rq:.4e} >= {MAX_REL_L2:.4e}"
    );
    assert!(
        rk < MAX_REL_L2,
        "{label} K: rel_L2={rk:.4e} >= {MAX_REL_L2:.4e}"
    );
    assert!(
        rv < MAX_REL_L2,
        "{label} V: rel_L2={rv:.4e} >= {MAX_REL_L2:.4e}"
    );
    eprintln!("{label} Q={rq:.2e} K={rk:.2e} V={rv:.2e} OK");
}

/// Production-like shapes: cols=2048 (8 blocks/row), enough blocks to average
/// down the f16 rounding error to well below the 1% gate.
#[test]
fn qkv_rope_append_f16s_2r_quality_gate() {
    let ctx = ctx();
    // (n_q, n_k, hd, cols, pos, kv_off, seed)
    let cases: &[(usize, usize, usize, usize, u32, usize, u32)] = &[
        // Qwen-3B-like shape (2048 Q rows, 1024 KV rows)
        (16, 8, 128, 2048, 0, 0, 0xD300),
        (16, 8, 128, 2048, 63, 31, 0xD301),
        (16, 8, 128, 2048, 255, 127, 0xD302),
        // Smaller Q (8 heads)
        (8, 4, 128, 2048, 17, 5, 0xD310),
    ];

    for &(n_q, n_k, hd, cols, pos, kv_off, seed) in cases {
        let shape = Shape {
            n_q,
            n_k,
            hd,
            cols,
            pos,
            kv_off,
        };
        let q_rows = n_q * hd;
        let w = make_weights(&shape, seed);
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                ((i as u32).wrapping_mul(1664525).wrapping_add(seed) as f32 / u32::MAX as f32) * 2.0
                    - 1.0
            })
            .collect();

        let (ref_q, ref_k, ref_v) = run_f32_2r(ctx, &shape, &w, &x);
        let (got_q, got_k, got_v) = run_f16s_2r(ctx, &shape, &w, &x);

        let label = format!("2r nq={n_q} nk={n_k} cols={cols} pos={pos} off={kv_off}");
        check_rel_l2(&label, &ref_q, &ref_k, &ref_v, &got_q, &got_k, &got_v);
    }
}

/// Same shapes for the 4r variant. q_rows and kv_rows must be divisible by 4.
#[test]
fn qkv_rope_append_f16s_4r_quality_gate() {
    let ctx = ctx();
    let cases: &[(usize, usize, usize, usize, u32, usize, u32)] = &[
        (16, 8, 128, 2048, 0, 0, 0xD400),
        (16, 8, 128, 2048, 63, 31, 0xD401),
        (16, 8, 128, 2048, 255, 127, 0xD402),
        (8, 4, 128, 2048, 17, 5, 0xD410),
    ];

    for &(n_q, n_k, hd, cols, pos, kv_off, seed) in cases {
        let shape = Shape {
            n_q,
            n_k,
            hd,
            cols,
            pos,
            kv_off,
        };
        let q_rows = n_q * hd;
        let w = make_weights(&shape, seed);
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                ((i as u32).wrapping_mul(1664525).wrapping_add(seed) as f32 / u32::MAX as f32) * 2.0
                    - 1.0
            })
            .collect();

        // Verify divisibility-by-4 for the 4r kernel (should hold for hd=128).
        let kv_rows = n_k * hd;
        assert!(q_rows % 4 == 0 && kv_rows % 4 == 0);

        let (ref_q, ref_k, ref_v) = run_f32_4r(ctx, &shape, &w, &x);
        let (got_q, got_k, got_v) = run_f16s_4r(ctx, &shape, &w, &x);

        let label = format!("4r nq={n_q} nk={n_k} cols={cols} pos={pos} off={kv_off}");
        check_rel_l2(&label, &ref_q, &ref_k, &ref_v, &got_q, &got_k, &got_v);
    }
}

/// Cross-check: 2r and 4r f16s variants agree with each other (not just with f32).
#[test]
fn qkv_rope_append_f16s_2r_vs_4r_agree() {
    let ctx = ctx();
    let shape = Shape {
        n_q: 16,
        n_k: 8,
        hd: 128,
        cols: 2048,
        pos: 42,
        kv_off: 13,
    };
    let q_rows = shape.n_q * shape.hd;
    let w = make_weights(&shape, 0xD500);
    let x: Vec<f32> = (0..shape.cols)
        .map(|i| {
            ((i as u32).wrapping_mul(1664525).wrapping_add(0xD500) as f32 / u32::MAX as f32) * 2.0
                - 1.0
        })
        .collect();

    let (q2, k2, v2) = run_f16s_2r(ctx, &shape, &w, &x);
    let (q4, k4, v4) = run_f16s_4r(ctx, &shape, &w, &x);

    let rq = rel_l2(&q2, &q4);
    let rk = rel_l2(&k2, &k4);
    let rv = rel_l2(&v2, &v4);
    // 2r and 4r use the same SIMD lane arithmetic; difference should be FP
    // non-associativity only — well below 1e-4.
    let _ = q_rows;
    assert!(rq < 1e-4, "2r vs 4r Q: rel_L2={rq:.4e}");
    assert!(rk < 1e-4, "2r vs 4r K: rel_L2={rk:.4e}");
    assert!(rv < 1e-4, "2r vs 4r V: rel_L2={rv:.4e}");
    eprintln!("2r vs 4r Q={rq:.2e} K={rk:.2e} V={rv:.2e} OK");
}
