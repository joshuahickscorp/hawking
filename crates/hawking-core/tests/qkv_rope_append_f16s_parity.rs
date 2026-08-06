#![cfg(target_os = "macos")]
use hawking_core::kernels;
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
mod common;
use common::*;
fn make_q4k_predec_hi(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let bpr = cols / 256;
    let w: Vec<u8> = (0..rows * bpr * 144)
        .map(|i| ((i as u32).wrapping_mul(2246822519u32).wrapping_add(seed)) as u8)
        .collect();
    let s: Vec<f32> = (0..rows * bpr * 16)
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
    let (q_w, q_sc_f32) = make_q4k_predec_hi(s.n_q * s.hd, s.cols, seed);
    let (k_w, k_sc_f32) = make_q4k_predec_hi(s.n_k * s.hd, s.cols, seed ^ 0x10);
    let (v_w, v_sc_f32) = make_q4k_predec_hi(s.n_k * s.hd, s.cols, seed ^ 0x20);
    Q4kWeights {
        q_sc_f16: f32_to_f16_bytes(&q_sc_f32),
        k_sc_f16: f32_to_f16_bytes(&k_sc_f32),
        v_sc_f16: f32_to_f16_bytes(&v_sc_f32),
        q_w,
        q_sc_f32,
        k_w,
        k_sc_f32,
        v_w,
        v_sc_f32,
    }
}
#[derive(Clone, Copy)]
enum Variant {
    F32_2r,
    F16s_2r,
    F32_4r,
    F16s_4r,
}
fn run(
    ctx: &MetalContext,
    shape: &Shape,
    w: &Q4kWeights,
    x: &[f32],
    variant: Variant,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_rows = shape.n_q * shape.hd;
    let kv_rows = shape.n_k * shape.hd;
    let model_bytes = [&w.q_w[..], &w.k_w[..], &w.v_w[..]].concat();
    let q_off = 0usize;
    let k_off = w.q_w.len();
    let v_off = w.q_w.len() + w.k_w.len();
    let model = ctx.new_buffer_with_bytes(&model_bytes);
    let (q_sc, k_sc, v_sc): (PinnedBuffer, PinnedBuffer, PinnedBuffer) = match variant {
        Variant::F32_2r | Variant::F32_4r => (
            new_f32_buf(ctx, &w.q_sc_f32),
            new_f32_buf(ctx, &w.k_sc_f32),
            new_f32_buf(ctx, &w.v_sc_f32),
        ),
        Variant::F16s_2r | Variant::F16s_4r => (
            ctx.new_buffer_with_bytes(&w.q_sc_f16),
            ctx.new_buffer_with_bytes(&w.k_sc_f16),
            ctx.new_buffer_with_bytes(&w.v_sc_f16),
        ),
    };
    let x_buf = new_f32_buf(ctx, x);
    let q_buf = ctx.new_buffer(q_rows * 4);
    let cache_len = shape.kv_off + kv_rows + 8;
    let k_cache = new_f32_buf(ctx, &vec![0.0f32; cache_len]);
    let v_cache = new_f32_buf(ctx, &vec![0.0f32; cache_len]);
    let mut tcb = TokenCommandBuffer::new(ctx);
    let r = match variant {
        Variant::F32_2r => kernels::gemv_q4k_predec_qkv_rope_append_pinned_tcb(
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
        ),
        Variant::F16s_2r => kernels::gemv_q4k_predec_qkv_rope_append_f16s_pinned_tcb(
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
        ),
        Variant::F32_4r => kernels::gemv_q4k_predec_qkv_rope_append_4r_pinned_tcb(
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
        ),
        Variant::F16s_4r => kernels::gemv_q4k_predec_qkv_rope_append_4r_f16s_pinned_tcb(
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
        ),
    };
    r.expect("qkv rope dispatch");
    tcb.commit_and_wait().expect("qkv rope commit");
    let q = read_f32_buf(&q_buf, q_rows);
    let k = read_f32_buf(&k_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
    let v = read_f32_buf(&v_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
    (q, k, v)
}
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
    assert!(rq < MAX_REL_L2, "{label} Q: rel_L2={rq:.4e}");
    assert!(rk < MAX_REL_L2, "{label} K: rel_L2={rk:.4e}");
    assert!(rv < MAX_REL_L2, "{label} V: rel_L2={rv:.4e}");
}
fn shape_x(shape: &Shape, seed: u32) -> (Q4kWeights, Vec<f32>) {
    (
        make_weights(shape, seed),
        lcg_f32(shape.cols, seed, -1.0, 1.0),
    )
}
#[test]
fn qkv_rope_append_f16s_2r_quality_gate() {
    let ctx = ctx();
    let cases: &[(usize, usize, usize, usize, u32, usize, u32)] = &[
        (16, 8, 128, 2048, 0, 0, 0xD300),
        (16, 8, 128, 2048, 63, 31, 0xD301),
        (16, 8, 128, 2048, 255, 127, 0xD302),
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
        let (w, x) = shape_x(&shape, seed);
        let (ref_q, ref_k, ref_v) = run(ctx, &shape, &w, &x, Variant::F32_2r);
        let (got_q, got_k, got_v) = run(ctx, &shape, &w, &x, Variant::F16s_2r);
        check_rel_l2(
            &format!("2r nq={n_q} nk={n_k} cols={cols} pos={pos} off={kv_off}"),
            &ref_q,
            &ref_k,
            &ref_v,
            &got_q,
            &got_k,
            &got_v,
        );
    }
}
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
        assert!((n_q * hd) % 4 == 0 && (n_k * hd) % 4 == 0);
        let (w, x) = shape_x(&shape, seed);
        let (ref_q, ref_k, ref_v) = run(ctx, &shape, &w, &x, Variant::F32_4r);
        let (got_q, got_k, got_v) = run(ctx, &shape, &w, &x, Variant::F16s_4r);
        check_rel_l2(
            &format!("4r nq={n_q} nk={n_k} cols={cols} pos={pos} off={kv_off}"),
            &ref_q,
            &ref_k,
            &ref_v,
            &got_q,
            &got_k,
            &got_v,
        );
    }
}
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
    let (w, x) = shape_x(&shape, 0xD500);
    let (q2, k2, v2) = run(ctx, &shape, &w, &x, Variant::F16s_2r);
    let (q4, k4, v4) = run(ctx, &shape, &w, &x, Variant::F16s_4r);
    assert!(rel_l2(&q2, &q4) < 1e-4, "2r vs 4r Q");
    assert!(rel_l2(&k2, &k4) < 1e-4, "2r vs 4r K");
    assert!(rel_l2(&v2, &v4) < 1e-4, "2r vs 4r V");
}
