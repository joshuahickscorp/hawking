#![cfg(target_os = "macos")]
//! Track 3.12/3.13 parity: QKV triple with inline Q/K bias+RoPE and f32
//! KV-cache append must match the current three-dispatch sequence:
//! QKV triple, `rope_qk_f32_b1_bias`, then `kv_append_vbias_f32`.

use hawking_core::kernels;
use hawking_core::metal::{MetalContext, TokenCommandBuffer};
use hawking_core::quant;

mod common;
use common::*;

fn make_q4k(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let bpr = cols / 256;
    let total = rows * bpr * 144;
    let w: Vec<u8> = (0..total)
        .map(|i| ((i as u32).wrapping_mul(2246822519).wrapping_add(seed)) as u8)
        .collect();
    let ns = rows * bpr * 16;
    let s: Vec<f32> = (0..ns)
        .map(|i| {
            let v =
                ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32 / u32::MAX as f32;
            (v * 2.0 - 1.0) * 0.02
        })
        .collect();
    (w, s)
}

fn make_q6k(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let w = fixed_f32(rows * cols, seed);
    let mut q = vec![0u8; rows * (cols / 256) * quant::Q6_K_BLOCK_BYTES];
    quant::quantize_q6_k(&w, &mut q).expect("Q6_K quant");
    q
}

fn rel_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()).max(1.0))
        .fold(0.0_f32, f32::max)
}

fn assert_close(label: &str, a: &[f32], b: &[f32]) {
    let rel = rel_diff(a, b);
    assert!(rel < 1e-5, "{label}: max_rel={rel:.2e} > 1e-5");
}

struct Shape {
    n_q: usize,
    n_k: usize,
    hd: usize,
    cols: usize,
    pos: u32,
    kv_off: usize,
}

struct Q4Weights {
    q: Vec<u8>,
    q_scales: Vec<f32>,
    k: Vec<u8>,
    k_scales: Vec<f32>,
    v: Vec<u8>,
    v_scales: Vec<f32>,
}

fn run_q4_ref(
    ctx: &MetalContext,
    shape: &Shape,
    w: &Q4Weights,
    x: &[f32],
    q_bias: Option<&[f32]>,
    k_bias: Option<&[f32]>,
    v_bias: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_rows = shape.n_q * shape.hd;
    let kv_rows = shape.n_k * shape.hd;
    let model_bytes = [&w.q[..], &w.k[..], &w.v[..]].concat();
    let q_off = 0;
    let k_off = w.q.len();
    let v_off = w.q.len() + w.k.len();
    let model = ctx.new_buffer_with_bytes(&model_bytes);
    let q_sc = new_f32_buf(ctx, &w.q_scales);
    let k_sc = new_f32_buf(ctx, &w.k_scales);
    let v_sc = new_f32_buf(ctx, &w.v_scales);
    let x_buf = new_f32_buf(ctx, x);
    let q_buf = ctx.new_buffer(q_rows * 4);
    let k_tok = ctx.new_buffer(kv_rows * 4);
    let v_tok = ctx.new_buffer(kv_rows * 4);
    let q_bias_buf = q_bias.map(|b| new_f32_buf(ctx, b));
    let k_bias_buf = k_bias.map(|b| new_f32_buf(ctx, b));
    let v_bias_buf = v_bias.map(|b| new_f32_buf(ctx, b));
    let cache_len = shape.kv_off + kv_rows + 8;
    let k_cache = new_f32_buf(ctx, &vec![-17.0; cache_len]);
    let v_cache = new_f32_buf(ctx, &vec![23.0; cache_len]);

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4k_predec_qkv_triple_pinned_tcb(
        &mut tcb,
        &model,
        q_off,
        w.q.len(),
        &q_sc,
        k_off,
        w.k.len(),
        &k_sc,
        v_off,
        w.v.len(),
        &v_sc,
        q_rows,
        kv_rows,
        shape.cols,
        &x_buf,
        &q_buf,
        &k_tok,
        &v_tok,
    )
    .expect("qkv triple ref");
    kernels::rope_qk_f32_b1_bias_tcb(
        &mut tcb,
        &q_buf,
        &k_tok,
        q_bias_buf.as_ref(),
        k_bias_buf.as_ref(),
        shape.n_q,
        shape.n_k,
        shape.hd,
        shape.pos,
        10000.0,
    )
    .expect("rope ref");
    kernels::kv_append_vbias_f32_tcb(
        &mut tcb,
        &k_tok,
        &v_tok,
        v_bias_buf.as_ref(),
        &k_cache,
        &v_cache,
        kv_rows,
        shape.kv_off,
    )
    .expect("kv append ref");
    tcb.commit_and_wait().expect("ref commit");

    let q = read_f32_buf(&q_buf, q_rows);
    let k = read_f32_buf(&k_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
    let v = read_f32_buf(&v_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
    (q, k, v)
}

fn run_q4_fused(
    ctx: &MetalContext,
    shape: &Shape,
    w: &Q4Weights,
    x: &[f32],
    q_bias: Option<&[f32]>,
    k_bias: Option<&[f32]>,
    v_bias: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_rows = shape.n_q * shape.hd;
    let kv_rows = shape.n_k * shape.hd;
    let model_bytes = [&w.q[..], &w.k[..], &w.v[..]].concat();
    let q_off = 0;
    let k_off = w.q.len();
    let v_off = w.q.len() + w.k.len();
    let model = ctx.new_buffer_with_bytes(&model_bytes);
    let q_sc = new_f32_buf(ctx, &w.q_scales);
    let k_sc = new_f32_buf(ctx, &w.k_scales);
    let v_sc = new_f32_buf(ctx, &w.v_scales);
    let x_buf = new_f32_buf(ctx, x);
    let q_buf = ctx.new_buffer(q_rows * 4);
    let q_bias_buf = q_bias.map(|b| new_f32_buf(ctx, b));
    let k_bias_buf = k_bias.map(|b| new_f32_buf(ctx, b));
    let v_bias_buf = v_bias.map(|b| new_f32_buf(ctx, b));
    let cache_len = shape.kv_off + kv_rows + 8;
    let k_cache = new_f32_buf(ctx, &vec![-17.0; cache_len]);
    let v_cache = new_f32_buf(ctx, &vec![23.0; cache_len]);

    let mut tcb = TokenCommandBuffer::new(ctx);
    kernels::gemv_q4k_predec_qkv_rope_append_pinned_tcb(
        &mut tcb,
        &model,
        q_off,
        w.q.len(),
        &q_sc,
        k_off,
        w.k.len(),
        &k_sc,
        v_off,
        w.v.len(),
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
        q_bias_buf.as_ref(),
        k_bias_buf.as_ref(),
        v_bias_buf.as_ref(),
        &k_cache,
        &v_cache,
    )
    .expect("qkv rope append fused");
    tcb.commit_and_wait().expect("fused commit");

    let q = read_f32_buf(&q_buf, q_rows);
    let k = read_f32_buf(&k_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
    let v = read_f32_buf(&v_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
    (q, k, v)
}

#[test]
fn q4k_qkv_rope_append_matches_ref() {
    let ctx = ctx();
    let shape = Shape {
        n_q: 4,
        n_k: 2,
        hd: 64,
        cols: 256,
        pos: 127,
        kv_off: 19,
    };
    let q_rows = shape.n_q * shape.hd;
    let kv_rows = shape.n_k * shape.hd;
    let (q, q_scales) = make_q4k(q_rows, shape.cols, 0x1001);
    let (k, k_scales) = make_q4k(kv_rows, shape.cols, 0x1002);
    let (v, v_scales) = make_q4k(kv_rows, shape.cols, 0x1003);
    let weights = Q4Weights {
        q,
        q_scales,
        k,
        k_scales,
        v,
        v_scales,
    };
    let x = fixed_f32(shape.cols, 0xBEEF);
    let q_bias = fixed_f32(q_rows, 0xCAFE);
    let k_bias = fixed_f32(kv_rows, 0xF00D);
    let v_bias = fixed_f32(kv_rows, 0xD00D);

    let reference = run_q4_ref(
        ctx,
        &shape,
        &weights,
        &x,
        Some(&q_bias),
        Some(&k_bias),
        Some(&v_bias),
    );
    let fused = run_q4_fused(
        ctx,
        &shape,
        &weights,
        &x,
        Some(&q_bias),
        Some(&k_bias),
        Some(&v_bias),
    );
    assert_close("q4 q", &reference.0, &fused.0);
    assert_close("q4 k_cache", &reference.1, &fused.1);
    assert_close("q4 v_cache", &reference.2, &fused.2);
}

#[test]
fn mixed_q4k_q4k_q6k_rope_append_matches_ref() {
    let ctx = ctx();
    let shape = Shape {
        n_q: 4,
        n_k: 2,
        hd: 64,
        cols: 256,
        pos: 511,
        kv_off: 37,
    };
    let q_rows = shape.n_q * shape.hd;
    let kv_rows = shape.n_k * shape.hd;
    let (q, q_scales) = make_q4k(q_rows, shape.cols, 0x2001);
    let (k, k_scales) = make_q4k(kv_rows, shape.cols, 0x2002);
    let v = make_q6k(kv_rows, shape.cols, 0x2003);
    let x = fixed_f32(shape.cols, 0xFEED);
    let q_bias = fixed_f32(q_rows, 0xABCD);
    let k_bias = fixed_f32(kv_rows, 0x1234);
    let v_bias = fixed_f32(kv_rows, 0x5678);

    let model_bytes = [&q[..], &k[..], &v[..]].concat();
    let q_off = 0;
    let k_off = q.len();
    let v_off = q.len() + k.len();
    let model = ctx.new_buffer_with_bytes(&model_bytes);
    let q_sc = new_f32_buf(ctx, &q_scales);
    let k_sc = new_f32_buf(ctx, &k_scales);
    let x_buf = new_f32_buf(ctx, &x);
    let q_bias_buf = new_f32_buf(ctx, &q_bias);
    let k_bias_buf = new_f32_buf(ctx, &k_bias);
    let v_bias_buf = new_f32_buf(ctx, &v_bias);
    let cache_len = shape.kv_off + kv_rows + 8;

    let run_ref = || {
        let q_buf = ctx.new_buffer(q_rows * 4);
        let k_tok = ctx.new_buffer(kv_rows * 4);
        let v_tok = ctx.new_buffer(kv_rows * 4);
        let k_cache = new_f32_buf(ctx, &vec![-3.0; cache_len]);
        let v_cache = new_f32_buf(ctx, &vec![5.0; cache_len]);
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4k_q4k_q6k_triple_pinned_tcb(
            &mut tcb,
            &model,
            q_off,
            q.len(),
            &q_sc,
            k_off,
            k.len(),
            &k_sc,
            v_off,
            v.len(),
            q_rows,
            kv_rows,
            shape.cols,
            &x_buf,
            &q_buf,
            &k_tok,
            &v_tok,
        )
        .expect("mixed ref triple");
        kernels::rope_qk_f32_b1_bias_tcb(
            &mut tcb,
            &q_buf,
            &k_tok,
            Some(&q_bias_buf),
            Some(&k_bias_buf),
            shape.n_q,
            shape.n_k,
            shape.hd,
            shape.pos,
            10000.0,
        )
        .expect("mixed ref rope");
        kernels::kv_append_vbias_f32_tcb(
            &mut tcb,
            &k_tok,
            &v_tok,
            Some(&v_bias_buf),
            &k_cache,
            &v_cache,
            kv_rows,
            shape.kv_off,
        )
        .expect("mixed ref append");
        tcb.commit_and_wait().expect("mixed ref commit");
        (
            read_f32_buf(&q_buf, q_rows),
            read_f32_buf(&k_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec(),
            read_f32_buf(&v_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec(),
        )
    };

    let run_fused = || {
        let q_buf = ctx.new_buffer(q_rows * 4);
        let k_cache = new_f32_buf(ctx, &vec![-3.0; cache_len]);
        let v_cache = new_f32_buf(ctx, &vec![5.0; cache_len]);
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4k_q4k_q6k_rope_append_pinned_tcb(
            &mut tcb,
            &model,
            q_off,
            q.len(),
            &q_sc,
            k_off,
            k.len(),
            &k_sc,
            v_off,
            v.len(),
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
            Some(&q_bias_buf),
            Some(&k_bias_buf),
            Some(&v_bias_buf),
            &k_cache,
            &v_cache,
        )
        .expect("mixed fused");
        tcb.commit_and_wait().expect("mixed fused commit");
        (
            read_f32_buf(&q_buf, q_rows),
            read_f32_buf(&k_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec(),
            read_f32_buf(&v_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec(),
        )
    };

    let reference = run_ref();
    let fused = run_fused();
    assert_close("mixed q", &reference.0, &fused.0);
    assert_close("mixed k_cache", &reference.1, &fused.1);
    assert_close("mixed v_cache", &reference.2, &fused.2);
}
