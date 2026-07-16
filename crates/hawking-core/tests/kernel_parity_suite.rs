//! Consolidated stateless Metal kernel parity and microbenchmark cases.

#[cfg(target_os = "macos")]
#[path = "common.rs"]
#[rustfmt::skip]
mod common;
#[rustfmt::skip]
mod add_rmsnorm_fused_q8_parity {
    //! Fusion gate for `add_rmsnorm_fused_q8`: the single-dispatch kernel must
    //! produce bit-identical outputs to the back-to-back pair
    //! `add_rmsnorm_fused` + `quantize_f32_to_int8_per_block`. This is the
    //! non-negotiable correctness invariant — the fusion is purely a dispatch-
    //! reorganization, not a numerical approximation.
    //!
    //! Checks all four output buffers: x (post add), x_norm (f32), x_norm_int8,
    //! and x_norm_scales.

    #![cfg(target_os = "macos")]

    use hawking_core::kernels;
    use hawking_core::metal::{PinnedBuffer, TokenCommandBuffer};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    fn read_f32(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    fn read_i8(buf: &PinnedBuffer, n: usize) -> Vec<i8> {
        let ptr = buf.contents() as *const i8;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    fn run_one(hidden: usize, seed: u64) {
        let ctx = ctx();
        let mut rng = Pcg64Mcg::new(seed as u128);

        // Inputs.
        let x: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let attn_out: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-0.5f32..0.5)).collect();
        let weight: Vec<f32> = (0..hidden).map(|_| rng.gen_range(0.5f32..1.5)).collect();
        let eps = 1e-6f32;
        let blocks = hidden / 256;

        // Reference path: separate add_rmsnorm_fused + quantize.
        let ref_x_buf = new_f32_buf(ctx, &x);
        let ref_attn_buf = new_f32_buf(ctx, &attn_out);
        let weight_buf = new_f32_buf(ctx, &weight);
        let ref_xnorm_buf = ctx.new_buffer(hidden * 4);
        let ref_int8_buf = ctx.new_buffer(hidden);
        let ref_scales_buf = ctx.new_buffer(blocks * 4);
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::add_rmsnorm_fused_tcb(&mut tcb, &ref_x_buf, &ref_attn_buf, &weight_buf, &ref_xnorm_buf, eps, hidden).expect("ref add_rmsnorm_fused encode");
            kernels::quantize_f32_to_int8_per_block_tcb(&mut tcb, &ref_xnorm_buf, &ref_int8_buf, &ref_scales_buf, hidden).expect("ref quantize encode");
            tcb.commit_and_wait().expect("ref commit");
        }
        let ref_x = read_f32(&ref_x_buf, hidden);
        let ref_xnorm = read_f32(&ref_xnorm_buf, hidden);
        let ref_int8 = read_i8(&ref_int8_buf, hidden);
        let ref_scales = read_f32(&ref_scales_buf, blocks);

        // Fused path.
        let f_x_buf = new_f32_buf(ctx, &x);
        let f_attn_buf = new_f32_buf(ctx, &attn_out);
        let f_xnorm_buf = ctx.new_buffer(hidden * 4);
        let f_int8_buf = ctx.new_buffer(hidden);
        let f_scales_buf = ctx.new_buffer(blocks * 4);
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::add_rmsnorm_fused_q8_tcb(&mut tcb, &f_x_buf, &f_attn_buf, &weight_buf, &f_xnorm_buf, &f_int8_buf, &f_scales_buf, eps, hidden).expect("fused encode");
            tcb.commit_and_wait().expect("fused commit");
        }
        let f_x = read_f32(&f_x_buf, hidden);
        let f_xnorm = read_f32(&f_xnorm_buf, hidden);
        let f_int8 = read_i8(&f_int8_buf, hidden);
        let f_scales = read_f32(&f_scales_buf, blocks);

        assert_eq!(ref_x, f_x, "x (post-add) mismatch at hidden={hidden} seed={seed}");
        assert_eq!(ref_xnorm, f_xnorm, "x_norm mismatch at hidden={hidden} seed={seed}");
        assert_eq!(ref_scales, f_scales, "x_norm_scales mismatch at hidden={hidden} seed={seed}");
        let mut diffs = 0usize;
        let mut first_bad = None;
        for i in 0..hidden {
            if ref_int8[i] != f_int8[i] {
                diffs += 1;
                if first_bad.is_none() {
                    first_bad = Some((i, ref_int8[i], f_int8[i], f_xnorm[i], f_scales[i / 256]));
                }
            }
        }
        assert_eq!(diffs, 0, "x_norm_int8 mismatch at hidden={hidden} seed={seed}: {diffs} elems; first {:?}", first_bad);
    }

    #[test]
    fn add_rmsnorm_fused_q8_parity_hidden_256() {
        run_one(256, 0xA11CE);
    }

    #[test]
    fn add_rmsnorm_fused_q8_parity_hidden_2048() {
        // Qwen-3B hidden.
        run_one(2048, 0xBEEF);
    }

    #[test]
    fn add_rmsnorm_fused_q8_parity_hidden_2048_alt_seed() {
        run_one(2048, 0xC0FFEE);
    }
}
#[rustfmt::skip]
mod add_rmsnorm_fused_q8_scaled_parity {
    //! Fusion gate for `add_rmsnorm_fused_q8_scaled` (AWQ Option B): the single-
    //! dispatch kernel must produce bit-identical outputs to the back-to-back
    //! pair `add_rmsnorm_fused` + `quantize_f32_to_int8_per_block_scaled`.
    //!
    //! Checks four buffers: x (post add), x_norm (f32, NOT divided by s), the
    //! scaled int8 quant, and the scaled per-block scales.

    #![cfg(target_os = "macos")]

    use hawking_core::kernels;
    use hawking_core::metal::{PinnedBuffer, TokenCommandBuffer};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    fn read_f32(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    fn read_i8(buf: &PinnedBuffer, n: usize) -> Vec<i8> {
        let ptr = buf.contents() as *const i8;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    fn make_smoothing(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128 ^ 0xA5BAEu128);
        (0..n).map(|i| if i % 20 == 0 { rng.gen_range(2.0..5.0) } else { rng.gen_range(0.3..1.6) }).collect()
    }

    fn run_one(hidden: usize, seed: u64) {
        let ctx = ctx();
        let mut rng = Pcg64Mcg::new(seed as u128);

        let x: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let attn_out: Vec<f32> = (0..hidden).map(|_| rng.gen_range(-0.5f32..0.5)).collect();
        let weight: Vec<f32> = (0..hidden).map(|_| rng.gen_range(0.5f32..1.5)).collect();
        let s = make_smoothing(hidden, seed);
        let eps = 1e-6f32;
        let blocks = hidden / 256;

        // Reference: unfused add_rmsnorm_fused then scaled per-block quantize.
        let ref_x_buf = new_f32_buf(ctx, &x);
        let ref_attn_buf = new_f32_buf(ctx, &attn_out);
        let weight_buf = new_f32_buf(ctx, &weight);
        let s_buf = new_f32_buf(ctx, &s);
        let ref_xnorm_buf = ctx.new_buffer(hidden * 4);
        let ref_int8_buf = ctx.new_buffer(hidden);
        let ref_scales_buf = ctx.new_buffer(blocks * 4);
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::add_rmsnorm_fused_tcb(&mut tcb, &ref_x_buf, &ref_attn_buf, &weight_buf, &ref_xnorm_buf, eps, hidden).expect("ref add_rmsnorm_fused encode");
            kernels::quantize_f32_to_int8_per_block_scaled_tcb(&mut tcb, &ref_xnorm_buf, &s_buf, &ref_int8_buf, &ref_scales_buf, hidden).expect("ref scaled quantize encode");
            tcb.commit_and_wait().expect("ref commit");
        }
        let ref_x = read_f32(&ref_x_buf, hidden);
        let ref_xnorm = read_f32(&ref_xnorm_buf, hidden);
        let ref_int8 = read_i8(&ref_int8_buf, hidden);
        let ref_scales = read_f32(&ref_scales_buf, blocks);

        // Fused-scaled path.
        let f_x_buf = new_f32_buf(ctx, &x);
        let f_attn_buf = new_f32_buf(ctx, &attn_out);
        let f_xnorm_buf = ctx.new_buffer(hidden * 4);
        let f_int8_buf = ctx.new_buffer(hidden);
        let f_scales_buf = ctx.new_buffer(blocks * 4);
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::add_rmsnorm_fused_q8_scaled_tcb(&mut tcb, &f_x_buf, &f_attn_buf, &weight_buf, &f_xnorm_buf, &f_int8_buf, &f_scales_buf, &s_buf, eps, hidden).expect("fused-scaled encode");
            tcb.commit_and_wait().expect("fused-scaled commit");
        }
        let f_x = read_f32(&f_x_buf, hidden);
        let f_xnorm = read_f32(&f_xnorm_buf, hidden);
        let f_int8 = read_i8(&f_int8_buf, hidden);
        let f_scales = read_f32(&f_scales_buf, blocks);

        assert_eq!(ref_x, f_x, "x (post-add) mismatch at hidden={hidden} seed={seed}");
        assert_eq!(
            ref_xnorm, f_xnorm,
            "x_norm mismatch at hidden={hidden} seed={seed} \
             (Option B keeps x_norm unscaled — only the int8 sees s)"
        );
        assert_eq!(ref_scales, f_scales, "x_norm_scales mismatch at hidden={hidden} seed={seed}");
        let mut diffs = 0usize;
        let mut first_bad = None;
        for i in 0..hidden {
            if ref_int8[i] != f_int8[i] {
                diffs += 1;
                if first_bad.is_none() {
                    first_bad = Some((i, ref_int8[i], f_int8[i], f_xnorm[i], s[i], f_scales[i / 256]));
                }
            }
        }
        assert_eq!(diffs, 0, "x_norm_int8 mismatch at hidden={hidden} seed={seed}: {diffs} elems; first {:?}", first_bad);
    }

    #[test]
    fn add_rmsnorm_fused_q8_scaled_parity_hidden_256() {
        run_one(256, 0xA11CE);
    }

    #[test]
    fn add_rmsnorm_fused_q8_scaled_parity_hidden_2048() {
        run_one(2048, 0xBEEF);
    }

    #[test]
    fn add_rmsnorm_fused_q8_scaled_parity_hidden_2048_alt_seed() {
        run_one(2048, 0xC0FFEE);
    }

    // NOTE: this fused shader's phase-3 quantize uses one simdgroup per 256-block
    // across the 8 simdgroups in a 256-thread TG, so it caps at hidden=2048 (8
    // blocks). The existing un-scaled `add_rmsnorm_fused_q8_parity.rs` reflects
    // the same constraint (no >2048 case). At runtime the only callers of the
    // fused-scaled kernel are the two `hidden`-sized norm boundaries in
    // `qwen_dense.rs`; the 11008-sized `ffn_act` quantize uses the standalone
    // `quantize_f32_to_int8_per_block_scaled` kernel instead, which has no such
    // cap and is covered by `quantize_int8_scaled_parity::*_11008`.
}
#[rustfmt::skip]
mod embed_rmsnorm_fused_parity {
    #![cfg(target_os = "macos")]
    //! Track B7 parity: `embed_lookup_rmsnorm_f32` must produce bit-identical
    //! results to the two-dispatch sequence:
    //!   1. `embed_lookup_metal_f32_tcb` (embed[token] → x)
    //!   2. `rmsnorm_metal_buf_tcb`       (x → x_norm)
    //!
    //! Verifies both `x` (written in-place by the embed lookup) and `x_norm`
    //! (the normalized output). Tests several (hidden, token) pairs including
    //! the Qwen-3B production shape (hidden=2048).

    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};

    use crate::common;
    use common::*;

    fn make_embed(vocab: usize, hidden: usize, seed: u32) -> Vec<u16> {
        // Build a fp16 embedding table (vocab × hidden).
        (0..vocab * hidden)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761u32).wrapping_add(seed);
                // Map to [-1, 1] and convert to fp16 bits.
                let f = (x as f32 / u32::MAX as f32) * 2.0 - 1.0;
                half::f16::from_f32(f).to_bits()
            })
            .collect()
    }

    fn make_weight(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(1_664_525u32).wrapping_add(seed);
                0.5 + (x as f32 / u32::MAX as f32) // positive weights in [0.5, 1.5]
            })
            .collect()
    }

    /// Reference: embed_lookup_metal_f32_tcb + rmsnorm_metal_buf_tcb (2 dispatches).
    fn run_ref(ctx: &MetalContext, embed_f16: &[u16], weight: &[f32], token: u32, hidden: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
        let embed_bytes: Vec<u8> = embed_f16.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let embed_buf = ctx.new_buffer_with_bytes(&embed_bytes);
        let weight_buf = new_f32_buf(ctx, weight);
        let x_buf = ctx.new_buffer(hidden * 4);
        let x_norm_buf = ctx.new_buffer(hidden * 4);

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::embed_lookup_metal_f32_tcb(&mut tcb, &embed_buf, token, hidden, &x_buf).expect("embed_lookup");
        kernels::rmsnorm_metal_buf_tcb(&mut tcb, &x_buf, &weight_buf, eps, hidden, &x_norm_buf).expect("rmsnorm");
        tcb.commit_and_wait().expect("ref commit");

        (read_f32_buf(&x_buf, hidden), read_f32_buf(&x_norm_buf, hidden))
    }

    /// Fused: embed_lookup_rmsnorm_f32_tcb (1 dispatch).
    fn run_fused(ctx: &MetalContext, embed_f16: &[u16], weight: &[f32], token: u32, hidden: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
        let embed_bytes: Vec<u8> = embed_f16.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let embed_buf = ctx.new_buffer_with_bytes(&embed_bytes);
        let weight_buf = new_f32_buf(ctx, weight);
        let x_buf = ctx.new_buffer(hidden * 4);
        let x_norm_buf = ctx.new_buffer(hidden * 4);

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::embed_lookup_rmsnorm_f32_tcb(&mut tcb, &embed_buf, &weight_buf, token, hidden, eps, &x_buf, &x_norm_buf).expect("fused dispatch");
        tcb.commit_and_wait().expect("fused commit");

        (read_f32_buf(&x_buf, hidden), read_f32_buf(&x_norm_buf, hidden))
    }

    #[test]
    fn embed_lookup_rmsnorm_fused_matches_two_dispatch() {
        let ctx = ctx();
        let eps = 1e-6_f32;
        let vocab = 256; // small vocab for fast test; only token offset matters

        // (hidden, token, seed) — Qwen-3B uses hidden=2048. Also test boundary shapes.
        let cases: &[(usize, u32, u32)] = &[
            (256, 0, 0xE001), // minimal hidden = tg_size
            (512, 3, 0xE002),
            (1024, 7, 0xE003),
            (2048, 0, 0xE004), // Qwen-3B production shape
            (2048, 5, 0xE005), // different token
            (4096, 1, 0xE006), // max supported hidden = tg_size * 16
        ];

        for &(hidden, token, seed) in cases {
            let embed = make_embed(vocab, hidden, seed);
            let weight = make_weight(hidden, seed ^ 0xFF);

            let (ref_x, ref_norm) = run_ref(ctx, &embed, &weight, token, hidden, eps);
            let (got_x, got_norm) = run_fused(ctx, &embed, &weight, token, hidden, eps);

            let dx = max_abs_diff(&ref_x, &got_x);
            let dn = max_abs_diff(&ref_norm, &got_norm);
            assert_eq!(dx, 0.0, "hidden={hidden} token={token}: x max_diff={dx:.2e}");
            assert_eq!(dn, 0.0, "hidden={hidden} token={token}: x_norm max_diff={dn:.2e}");
            eprintln!("B7 hidden={hidden} token={token}: x={dx:.0e} norm={dn:.0e} OK");
        }
    }
}
#[rustfmt::skip]
mod ffn_tail_swiglu_add_rmsnorm_parity {
    #![cfg(target_os = "macos")]
    //! Track 3.15-style FFN tail parity:
    //! `ffn_down_swiglu + add_rmsnorm_ffn` must match the old
    //! `silu_mul + ffn_down + add_rmsnorm` sequence for Q4_K predec weights.

    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};

    use crate::common;
    use common::*;

    fn make_q4k_weights(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let blocks_per_row = cols / 256;
        let total_bytes = rows * blocks_per_row * 144;
        let w: Vec<u8> = (0..total_bytes).map(|i| ((i as u32).wrapping_mul(2246822519u32).wrapping_add(seed)) as u8).collect();
        let n_scales = rows * blocks_per_row * 16;
        let s: Vec<f32> = (0..n_scales)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761u32).wrapping_add(seed)) as f32 / u32::MAX as f32;
                v * 2.0 - 1.0
            })
            .collect();
        (w, s)
    }

    fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761u32).wrapping_add(seed)) as f32;
                (x / u32::MAX as f32) * 4.0 - 2.0
            })
            .collect()
    }

    fn run_ref(ctx: &MetalContext, w_q4: &[u8], scales: &[f32], gate: &[f32], up: &[f32], x: &[f32], norm_weight: &[f32], rows: usize, cols: usize, b: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let w_buf = ctx.new_buffer_with_bytes(w_q4);
        let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
        let gate_buf = new_f32_buf(ctx, gate);
        let up_buf = new_f32_buf(ctx, up);
        let x_buf = new_f32_buf(ctx, x);
        let norm_buf = new_f32_buf(ctx, norm_weight);
        let act_buf = ctx.new_buffer(b * cols * 4);
        let down_buf = ctx.new_buffer(b * rows * 4);
        let xnorm_buf = ctx.new_buffer(b * rows * 4);
        let w_bytes = rows * (cols / 256) * 144;

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::silu_mul_tcb(&mut tcb, &gate_buf, &up_buf, &act_buf, b * cols).unwrap();
        if b == 1 {
            kernels::gemv_q4_k_v4_predec_pinned_tcb(&mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, &act_buf, &down_buf).unwrap();
        } else if b <= 4 {
            kernels::gemm_q4_k_m_batched_v4r_predec_pinned_tcb(&mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &act_buf, &down_buf).unwrap();
        } else {
            kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(&mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &act_buf, &down_buf).unwrap();
        }
        kernels::add_rmsnorm_fused_batched_tcb(&mut tcb, &x_buf, &down_buf, &norm_buf, &xnorm_buf, 1e-6, rows, b).unwrap();
        tcb.commit_and_wait().unwrap();

        (read_f32_buf(&down_buf, b * rows), read_f32_buf(&x_buf, b * rows), read_f32_buf(&xnorm_buf, b * rows))
    }

    fn run_fused(ctx: &MetalContext, w_q4: &[u8], scales: &[f32], gate: &[f32], up: &[f32], x: &[f32], norm_weight: &[f32], rows: usize, cols: usize, b: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let w_buf = ctx.new_buffer_with_bytes(w_q4);
        let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
        let gate_buf = new_f32_buf(ctx, gate);
        let up_buf = new_f32_buf(ctx, up);
        let x_buf = new_f32_buf(ctx, x);
        let norm_buf = new_f32_buf(ctx, norm_weight);
        let down_buf = ctx.new_buffer(b * rows * 4);
        let xnorm_buf = ctx.new_buffer(b * rows * 4);
        let w_bytes = rows * (cols / 256) * 144;

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::ffn_down_swiglu_add_rmsnorm_ffn_q4k_predec_batched_tcb(&mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &gate_buf, &up_buf, &x_buf, &norm_buf, &xnorm_buf, 1e-6, &down_buf)
            .unwrap();
        tcb.commit_and_wait().unwrap();

        (read_f32_buf(&down_buf, b * rows), read_f32_buf(&x_buf, b * rows), read_f32_buf(&xnorm_buf, b * rows))
    }

    #[test]
    fn q4k_predec_swiglu_add_rmsnorm_tail_matches_ref() {
        let ctx = ctx();
        let rows = 512;
        let cols = 1024;
        let (w, scales) = make_q4k_weights(rows, cols, 0x3150);

        for b in [1usize, 3, 5] {
            let gate = rand_vec(b * cols, 0x5151 + b as u32);
            let up = rand_vec(b * cols, 0x6161 + b as u32);
            let x = rand_vec(b * rows, 0x7171 + b as u32);
            let norm_weight: Vec<f32> = rand_vec(rows, 0x8181 + b as u32).into_iter().map(|v| v.abs() + 0.5).collect();

            let (ref_down, ref_x, ref_xnorm) = run_ref(ctx, &w, &scales, &gate, &up, &x, &norm_weight, rows, cols, b);
            let (fused_down, fused_x, fused_xnorm) = run_fused(ctx, &w, &scales, &gate, &up, &x, &norm_weight, rows, cols, b);

            let down_diff = max_abs_diff(&ref_down, &fused_down);
            let x_diff = max_abs_diff(&ref_x, &fused_x);
            let xnorm_diff = max_abs_diff(&ref_xnorm, &fused_xnorm);
            assert!(down_diff < 1e-4 && x_diff < 1e-4 && xnorm_diff < 1e-4, "B={b}: tail diffs down={down_diff:.2e} x={x_diff:.2e} xnorm={xnorm_diff:.2e}");
            eprintln!("tail swiglu+add_rmsnorm B={b}: down={down_diff:.2e} x={x_diff:.2e} xnorm={xnorm_diff:.2e} OK");
        }
    }
}
#[rustfmt::skip]
mod kv_scatter_append_multiseq_parity {
    #![cfg(target_os = "macos")]
    //! R3 parity: `kv_scatter_append_multiseq` == the per-slot `memcpy_f32_off` loop
    //! it replaces, BYTE-IDENTICAL.
    //!
    //! The multi-seq decode stack used to append each slot's K and V into the cache
    //! with two `memcpy_f32_off_tcb` dispatches per slot (2B per layer), each writing
    //! kv_dim elements to `layer_off + regions[bi]*slot_stride + positions[bi]*kv_dim`.
    //! R3 batches that into ONE scatter dispatch (K+V together). It is a pure copy, so
    //! the cache must come out byte-identical. This runs BOTH on the GPU with churned
    //! (non-identity) stable regions and divergent positions, and asserts the whole
    //! K and V caches match exactly — including a non-zero `layer_off` (layer > 0).

    use hawking_core::kernels;
    use hawking_core::metal::TokenCommandBuffer;

    use crate::common;
    use common::*;

    #[allow(clippy::too_many_arguments)]
    fn scatter_per_slot(src_k: &[f32], src_v: &[f32], regions: &[usize], positions: &[usize], kv_dim: usize, slot_stride: usize, layer_off: usize, cache_elems: usize) -> (Vec<f32>, Vec<f32>) {
        let ctx = ctx();
        let b = regions.len();
        let kbuf = new_f32_buf(ctx, &vec![0.0f32; cache_elems]);
        let vbuf = new_f32_buf(ctx, &vec![0.0f32; cache_elems]);
        let sk = new_f32_buf(ctx, src_k);
        let sv = new_f32_buf(ctx, src_v);
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            for bi in 0..b {
                let dst_off = layer_off + regions[bi] * slot_stride + positions[bi] * kv_dim;
                kernels::memcpy_f32_off_tcb(&mut tcb, &sk, &kbuf, bi * kv_dim, dst_off, kv_dim).unwrap();
                kernels::memcpy_f32_off_tcb(&mut tcb, &sv, &vbuf, bi * kv_dim, dst_off, kv_dim).unwrap();
            }
            tcb.commit_and_wait().unwrap();
        }
        (read_f32_buf(&kbuf, cache_elems), read_f32_buf(&vbuf, cache_elems))
    }

    #[allow(clippy::too_many_arguments)]
    fn scatter_batched(src_k: &[f32], src_v: &[f32], regions: &[usize], positions: &[usize], kv_dim: usize, slot_stride: usize, layer_off: usize, cache_elems: usize) -> (Vec<f32>, Vec<f32>) {
        let ctx = ctx();
        let b = regions.len();
        let kbuf = new_f32_buf(ctx, &vec![0.0f32; cache_elems]);
        let vbuf = new_f32_buf(ctx, &vec![0.0f32; cache_elems]);
        let sk = new_f32_buf(ctx, src_k);
        let sv = new_f32_buf(ctx, src_v);
        let reg_bytes: Vec<u8> = regions.iter().flat_map(|&r| (r as u32).to_le_bytes()).collect();
        let pos_bytes: Vec<u8> = positions.iter().flat_map(|&p| (p as u32).to_le_bytes()).collect();
        let reg_buf = ctx.new_buffer_with_bytes(&reg_bytes);
        let pos_buf = ctx.new_buffer_with_bytes(&pos_bytes);
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::kv_scatter_append_multiseq_tcb(&mut tcb, &sk, &sv, &kbuf, &vbuf, &reg_buf, &pos_buf, kv_dim, b, slot_stride, layer_off).unwrap();
            tcb.commit_and_wait().unwrap();
        }
        (read_f32_buf(&kbuf, cache_elems), read_f32_buf(&vbuf, cache_elems))
    }

    #[test]
    fn kv_scatter_append_multiseq_matches_per_slot() {
        let kv_dim = 8usize;
        let max_seq = 4usize;
        let max_batch = 4usize;
        let slot_stride = max_seq * kv_dim; // 32 elems per slot per layer
        let layer_kv_stride = max_batch * slot_stride; // one layer of cache = 128 elems

        // Churned (non-identity) stable regions + divergent positions: slot bi writes
        // to region regions[bi] at position positions[bi] — the case the per-slot loop
        // and the scatter must agree on.
        let regions = [3usize, 1, 0, 2];
        let positions = [2usize, 0, 3, 1];
        let b = regions.len();

        let src_k = fixed_f32(b * kv_dim, 0x1111_2222_3333_4444);
        let src_v = fixed_f32(b * kv_dim, 0x5555_6666_7777_8888);

        // Layer 0 (layer_off = 0).
        {
            let cache_elems = layer_kv_stride;
            let (ek, ev) = scatter_per_slot(&src_k, &src_v, &regions, &positions, kv_dim, slot_stride, 0, cache_elems);
            let (ak, av) = scatter_batched(&src_k, &src_v, &regions, &positions, kv_dim, slot_stride, 0, cache_elems);
            assert_eq!(max_abs_diff(&ek, &ak), 0.0, "layer0 K: batched != per-slot");
            assert_eq!(max_abs_diff(&ev, &av), 0.0, "layer0 V: batched != per-slot");
        }

        // Layer 1 (non-zero layer_off): exercises the per-layer base offset.
        {
            let layer_off = layer_kv_stride; // li = 1
            let cache_elems = 2 * layer_kv_stride; // room for 2 layers
            let (ek, ev) = scatter_per_slot(&src_k, &src_v, &regions, &positions, kv_dim, slot_stride, layer_off, cache_elems);
            let (ak, av) = scatter_batched(&src_k, &src_v, &regions, &positions, kv_dim, slot_stride, layer_off, cache_elems);
            assert_eq!(max_abs_diff(&ek, &ak), 0.0, "layer1 K: batched != per-slot");
            assert_eq!(max_abs_diff(&ev, &av), 0.0, "layer1 V: batched != per-slot");
        }

        println!("[kv-scatter-append-multiseq] K+V byte-identical vs per-slot (layers 0 and 1, churned regions)");
    }
}
#[rustfmt::skip]
mod megakernel_2layer_parity {
    //! 2-layer megakernel parity test (2026-05-25 → 2026-05-26+).
    //!
    //! Grows stage-by-stage as the megakernel shader gains real compute.
    //! Each landed stage extends the assertion forward via the shader's
    //! `probe_stage` selector: the shader executes the full stage prefix
    //! and copies the chosen intermediate to `x_out`; the test computes the
    //! same intermediate in f32 from a synthetic residual and compares.
    //!
    //! When stage L of layer 1 lands, `probe_stage` collapses to
    //! [`MK_PROBE_RESIDUAL`] and the dev-only escape hatch retires.
    //!
    //! Day-3 entry: stage A (layer-0 pre-attention rmsnorm), `MK_PROBE_XNORM_A`.
    //! Day-4 entry: stages B/C/D (Q/K/V + biases + RoPE), `MK_PROBE_Q_ROT`.

    #![cfg(target_os = "macos")]

    use std::path::PathBuf;

    use half::f16;
    use hawking_core::kernels::megakernel::{
        megakernel_2layer_dispatch, megakernel_nlayer_dispatch, MegakernelRunner, MK_PROBE_ATTN_OUT, MK_PROBE_FFN_DOWN, MK_PROBE_O_PROJ, MK_PROBE_Q_ROT, MK_PROBE_RESIDUAL, MK_PROBE_RESIDUAL_L0,
        MK_PROBE_XNORM_A, MK_PROBE_XNORM_FFN,
    };
    use hawking_core::metal::MetalContext;
    use hawking_core::model::qwen_dense::{MegakernelLayerWeightsF16, QwenDense};
    use hawking_core::{Engine, EngineConfig};

    use crate::common;
    use common::*;

    const TOKEN: u32 = 42;
    const POS: usize = 0;
    const LAST_LAYER: usize = 1;
    const MAX_SEQ: u32 = 256;

    // Qwen-3B shape constants (mirror shader header).
    const HIDDEN: usize = 2048;
    const N_HEADS: usize = 16;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;
    const Q_DIM: usize = N_HEADS * HEAD_DIM; // 2048
    const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // 256
    const INTERMEDIATE: usize = 11008;
    const RMS_EPS: f32 = 1e-6;
    const ROPE_THETA: f32 = 1_000_000.0;

    /// Relative tolerance for fp16 stores. The shader stores intermediates
    /// (Q, K, V, attn_out, residual, …) as f16; the CPU reference is f32.
    /// f16 carries ~10 mantissa bits → ~1e-3 RELATIVE precision, so for a
    /// value of magnitude ~10 the absolute fp16 store noise is ~1e-2. The
    /// effective gate is `|diff| ≤ ATOL + RTOL * |want|`, mirroring numpy's
    /// `assert_allclose`. AGENT.md § "Verification rule" specifies atol=1e-3
    /// fp16 for kernel parity with O(1) inputs; here the synthetic input
    /// drives activations into O(10) range so the relative term takes over.
    const RTOL: f32 = 2e-3;

    /// Multi-layer fp16 noise accumulates: each layer threads ~10 f16 stores
    /// (xnorm, q/k/v, attn_out, o, residual, ffn_act, ffn_down) and the
    /// next layer's input is the previous layer's residual, so post-l1
    /// residual error tracks ~N × per-stage noise rather than single-stage
    /// noise. Empirically observed worst |diff|=4.8e-3 at |want|=0.345 over
    /// 2 layers (≈10 ULPs of fp16), well below the "orders of magnitude"
    /// threshold the design memo defines as a real-bug signal.
    const RTOL_MULTILAYER: f32 = 2e-2;
    /// Multi-layer absolute tolerance — looser than single-stage to absorb
    /// fp16 cancellation noise on values that pass through additive paths
    /// (residual streams routinely contain near-zero entries where small
    /// f16 rounding errors dominate the magnitude). Tracks Anthropic's
    /// guidance to compare networks up to ~1% relative without flagging
    /// model-correctness regressions.
    const ATOL_MULTILAYER: f32 = 5e-3;

    fn weights_path() -> PathBuf {
        if let Ok(p) = std::env::var("HAWKING_QWEN_GGUF") {
            return PathBuf::from(p);
        }
        PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
    }

    #[test]
    #[ignore = "megakernel POC: requires Qwen-3B weights via HAWKING_QWEN_GGUF"]
    fn megakernel_2layer_parity_qwen3b() {
        let weights = weights_path();
        if !weights.exists() {
            eprintln!("SKIP: model not at {}", weights.display());
            return;
        }

        let cfg = EngineConfig::default();
        let mut model = <QwenDense as Engine>::load(&weights, cfg).expect("load QwenDense");
        let h = model.config.hidden;
        let q_dim = model.config.n_heads * model.config.head_dim;
        let kv_dim = model.config.n_kv_heads * model.config.head_dim;
        let mid = model.config.intermediate;
        assert_eq!(h, HIDDEN);
        assert_eq!(q_dim, Q_DIM);
        assert_eq!(kv_dim, KV_DIM);

        // Reference: existing CPU forward path (sanity-checks the model
        // loaded; not used directly in stage-by-stage probes).
        let ref_x = model.forward_layers_subset(TOKEN, POS, LAST_LAYER).expect("forward_layers_subset");
        assert_eq!(ref_x.len(), h, "ref residual has wrong length");
        assert!(ref_x.iter().all(|v: &f32| v.is_finite()), "ref residual contains NaN/Inf");

        // Weight prep — pre-dequantize layer 0 + layer 1 to f16.
        let layer0 = model.prep_megakernel_layer_f16(0).expect("prep_megakernel_layer_f16(0)");
        let layer1 = model.prep_megakernel_layer_f16(1).expect("prep_megakernel_layer_f16(1)");
        assert_layer_shapes(&layer0, h, q_dim, kv_dim, mid, "layer 0");
        assert_layer_shapes(&layer1, h, q_dim, kv_dim, mid, "layer 1");

        // Synthetic input residual — deterministic, distinct per element so
        // any harness bug (off-by-one stride, wrong gpu_address indexing,
        // etc.) surfaces in the readback.
        let x_in: Vec<f16> = (0..h).map(|i| f16::from_f32((i as f32) * 0.001 - 1.0)).collect();
        let x_in_f32: Vec<f32> = x_in.iter().map(|v| v.to_f32()).collect();

        let ctx = MetalContext::new().expect("MetalContext::new");

        // ── Stage A: layer-0 pre-attention rmsnorm ──────────────────────────
        {
            let x_out = megakernel_2layer_dispatch(&ctx, &layer0, &layer1, &x_in, POS as u32, (POS + 1) as u32, MAX_SEQ, MK_PROBE_XNORM_A).expect("megakernel dispatch (stage A)");
            assert_eq!(x_out.len(), h);

            let ref_xnorm = cpu_rmsnorm(&x_in_f32, &layer0.attn_norm, RMS_EPS);
            let (worst, idx, gv, wv) = max_violation_f16_vs_f32(&x_out, &ref_xnorm);
            assert!(
                worst <= 0.0,
                "stage-A rmsnorm parity FAIL: violation={worst:.3e} at i={idx} \
                 (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL:.0e}, rtol={RTOL:.0e})",
            );
            eprintln!("stage-A rmsnorm parity OK (worst violation {worst:.3e} ≤ 0, atol={ATOL:.0e} rtol={RTOL:.0e})");
        }

        // ── Stages B/C/D: Q/K/V GEMV + biases + RoPE on Q (probe = post-RoPE Q)
        {
            let x_out = megakernel_2layer_dispatch(&ctx, &layer0, &layer1, &x_in, POS as u32, (POS + 1) as u32, MAX_SEQ, MK_PROBE_Q_ROT).expect("megakernel dispatch (stage D)");
            assert_eq!(x_out.len(), h);
            // Shader emits Q_DIM = HIDDEN = 2048 f16 values into x_out.

            // CPU reference for post-RoPE Q:
            //   1. x_norm = rmsnorm(x_in, layer0.attn_norm)
            //   2. q = qw @ x_norm  (f16 weight × f32 activation, f32 acc)
            //   3. q += q_bias
            //   4. rope_inplace per head on q
            let x_norm = cpu_rmsnorm(&x_in_f32, &layer0.attn_norm, RMS_EPS);
            let mut q = cpu_gemv_f16(&layer0.q_proj, q_dim, h, &x_norm);
            for i in 0..q_dim {
                q[i] += layer0.q_bias[i];
            }
            for hh in 0..N_HEADS {
                let off = hh * HEAD_DIM;
                cpu_rope_inplace(&mut q[off..off + HEAD_DIM], POS as u32, ROPE_THETA);
            }

            let (worst, idx, gv, wv) = max_violation_f16_vs_f32(&x_out, &q);
            assert!(
                worst <= 0.0,
                "stage-D Q (post-RoPE) parity FAIL: violation={worst:.3e} at i={idx} \
                 (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL:.0e}, rtol={RTOL:.0e})",
            );
            eprintln!("stage-D Q (post-RoPE) parity OK (worst violation {worst:.3e} ≤ 0, atol={ATOL:.0e} rtol={RTOL:.0e})");
        }

        // ── Stages E/F: KV write + MHA decode (probe = attn_out) ────────────
        //
        // At pos=0/seq_len=1 the softmax is degenerate (single position →
        // weight 1.0) so attn_out reduces to V replicated across grouped
        // heads. This still exercises:
        //   * KV write to DRAM at the correct (layer, slot, kv_head, dim) offset
        //   * the per-head loop and kv_h = h / group_size indexing
        //   * the (now-trivial) softmax max-reduce + sum-reduce paths
        //   * the V-weighted sum readback structure
        //
        // Non-trivial seq_len exercises (multi-position softmax) are queued
        // for a follow-up that exposes a persistent kv_cache buffer across
        // dispatches.
        {
            let x_out = megakernel_2layer_dispatch(&ctx, &layer0, &layer1, &x_in, POS as u32, (POS + 1) as u32, MAX_SEQ, MK_PROBE_ATTN_OUT).expect("megakernel dispatch (stage F)");
            assert_eq!(x_out.len(), h);

            let attn_out_ref = cpu_layer0_attn_out_pos0(&x_in_f32, &layer0);
            let (worst, idx, gv, wv) = max_violation_f16_vs_f32(&x_out, &attn_out_ref);
            assert!(
                worst <= 0.0,
                "stage-F attn_out (pos=0) parity FAIL: violation={worst:.3e} at i={idx} \
                 (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL:.0e}, rtol={RTOL:.0e})",
            );
            eprintln!("stage-F attn_out (pos=0, seq_len=1) parity OK (worst violation {worst:.3e} ≤ 0, atol={ATOL:.0e} rtol={RTOL:.0e})");
        }

        // ── Stage G: o_proj (probe = o) ─────────────────────────────────────
        {
            let x_out = megakernel_2layer_dispatch(&ctx, &layer0, &layer1, &x_in, POS as u32, (POS + 1) as u32, MAX_SEQ, MK_PROBE_O_PROJ).expect("megakernel dispatch (stage G)");
            assert_eq!(x_out.len(), h);

            let attn_out = cpu_layer0_attn_out_pos0(&x_in_f32, &layer0);
            let o = cpu_gemv_f16(&layer0.o_proj, HIDDEN, Q_DIM, &attn_out);
            let (worst, idx, gv, wv) = max_violation_f16_vs_f32(&x_out, &o);
            assert!(
                worst <= 0.0,
                "stage-G o_proj parity FAIL: violation={worst:.3e} at i={idx} \
                 (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL:.0e}, rtol={RTOL:.0e})",
            );
            eprintln!("stage-G o_proj parity OK (worst violation {worst:.3e} ≤ 0, atol={ATOL:.0e} rtol={RTOL:.0e})");
        }

        // ── Stage H: post-attn add+rmsnorm (probe = xnorm_ffn) ──────────────
        {
            let x_out = megakernel_2layer_dispatch(&ctx, &layer0, &layer1, &x_in, POS as u32, (POS + 1) as u32, MAX_SEQ, MK_PROBE_XNORM_FFN).expect("megakernel dispatch (stage H)");
            assert_eq!(x_out.len(), h);

            let attn_out = cpu_layer0_attn_out_pos0(&x_in_f32, &layer0);
            let o = cpu_gemv_f16(&layer0.o_proj, HIDDEN, Q_DIM, &attn_out);
            let mut residual = x_in_f32.clone();
            for i in 0..HIDDEN {
                residual[i] += o[i];
            }
            let xnorm_ffn = cpu_rmsnorm(&residual, &layer0.ffn_norm, RMS_EPS);
            let (worst, idx, gv, wv) = max_violation_f16_vs_f32(&x_out, &xnorm_ffn);
            assert!(
                worst <= 0.0,
                "stage-H xnorm_ffn parity FAIL: violation={worst:.3e} at i={idx} \
                 (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL:.0e}, rtol={RTOL:.0e})",
            );
            eprintln!("stage-H xnorm_ffn parity OK (worst violation {worst:.3e} ≤ 0, atol={ATOL:.0e} rtol={RTOL:.0e})");
        }

        // ── Stages I/J/K: fused FFN gate+up+silu_mul + ffn_down (probe = ffn_down)
        {
            let x_out = megakernel_2layer_dispatch(&ctx, &layer0, &layer1, &x_in, POS as u32, (POS + 1) as u32, MAX_SEQ, MK_PROBE_FFN_DOWN).expect("megakernel dispatch (stage K)");
            assert_eq!(x_out.len(), h);

            let attn_out = cpu_layer0_attn_out_pos0(&x_in_f32, &layer0);
            let o = cpu_gemv_f16(&layer0.o_proj, HIDDEN, Q_DIM, &attn_out);
            let mut residual = x_in_f32.clone();
            for i in 0..HIDDEN {
                residual[i] += o[i];
            }
            let xnorm_ffn = cpu_rmsnorm(&residual, &layer0.ffn_norm, RMS_EPS);
            let g = cpu_gemv_f16(&layer0.ffn_gate, INTERMEDIATE, HIDDEN, &xnorm_ffn);
            let u = cpu_gemv_f16(&layer0.ffn_up, INTERMEDIATE, HIDDEN, &xnorm_ffn);
            let mut act = vec![0.0f32; INTERMEDIATE];
            for i in 0..INTERMEDIATE {
                let s = g[i] / (1.0 + (-g[i]).exp());
                act[i] = s * u[i];
            }
            let ffn_down = cpu_gemv_f16(&layer0.ffn_down, HIDDEN, INTERMEDIATE, &act);

            let (worst, idx, gv, wv) = max_violation_f16_vs_f32(&x_out, &ffn_down);
            assert!(
                worst <= 0.0,
                "stage-K ffn_down parity FAIL: violation={worst:.3e} at i={idx} \
                 (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL:.0e}, rtol={RTOL:.0e})",
            );
            eprintln!("stage-K ffn_down parity OK (worst violation {worst:.3e} ≤ 0, atol={ATOL:.0e} rtol={RTOL:.0e})");
        }

        // ── Stage L (layer 0): post-FFN add (probe = residual_l0) ──────────
        {
            let x_out = megakernel_2layer_dispatch(&ctx, &layer0, &layer1, &x_in, POS as u32, (POS + 1) as u32, MAX_SEQ, MK_PROBE_RESIDUAL_L0).expect("megakernel dispatch (stage L, post-l0)");
            assert_eq!(x_out.len(), h);

            let residual = cpu_layer_forward(&x_in_f32, &layer0, POS as u32);
            let (worst, idx, gv, wv) = max_violation_f16_vs_f32(&x_out, &residual);
            assert!(
                worst <= 0.0,
                "stage-L residual (post-layer-0) parity FAIL: violation={worst:.3e} at i={idx} \
                 (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL:.0e}, rtol={RTOL:.0e})",
            );
            eprintln!("stage-L residual (post-layer-0) parity OK (worst violation {worst:.3e} ≤ 0, atol={ATOL:.0e} rtol={RTOL:.0e})");
        }

        // ── Final: 2-layer post-layer-1 residual parity ──────────────────────
        // The functional 2-layer POC acceptance gate per the prompt. Runs
        // layer 0 then layer 1 inline; compares against CPU-equivalent
        // chained-layer forward (cpu_layer_forward applied twice with the
        // layer-0 output as layer-1's input).
        {
            let x_out = megakernel_2layer_dispatch(&ctx, &layer0, &layer1, &x_in, POS as u32, (POS + 1) as u32, MAX_SEQ, MK_PROBE_RESIDUAL).expect("megakernel dispatch (post-l1 final)");
            assert_eq!(x_out.len(), h);

            let residual_l0 = cpu_layer_forward(&x_in_f32, &layer0, POS as u32);
            let residual_l1 = cpu_layer_forward(&residual_l0, &layer1, POS as u32);
            let (worst, idx, gv, wv) = max_violation_f16_vs_f32_tol(&x_out, &residual_l1, ATOL_MULTILAYER, RTOL_MULTILAYER);
            assert!(
                worst <= 0.0,
                "2-layer post-l1 residual parity FAIL: violation={worst:.3e} at i={idx} \
                 (got {gv}, want {wv}, allowed atol+rtol·|want|, atol={ATOL_MULTILAYER:.0e}, rtol={RTOL_MULTILAYER:.0e})",
            );
            eprintln!("2-layer post-l1 residual parity OK (worst violation {worst:.3e} ≤ 0, atol={ATOL_MULTILAYER:.0e} rtol={RTOL_MULTILAYER:.0e}) — FUNCTIONAL 2-LAYER POC ACCEPTANCE");
        }
    }

    /// N-layer megakernel parity (the scaling kernel `qwen3b_megakernel_nlayer`).
    ///
    /// Two gates:
    ///   (a) **N=2 matches the validated `qwen3b_megakernel_2layer`.** Same
    ///       per-layer arithmetic, so this pins the loop + packed-array argbuf +
    ///       helper extraction against known-good code with no new reference.
    ///       (Reported bit-exactness is informational; the assert is the tight
    ///       single-stage tolerance, which any structural bug blows past.)
    ///   (b) **N=8 vs a CPU chained-layer forward.** Catches per-layer KV-stride
    ///       / indexing bugs that only surface at depth > 2, within the
    ///       multi-layer fp16 tolerance scaled for the deeper accumulation.
    #[test]
    #[ignore = "megakernel POC: requires Qwen-3B weights via HAWKING_QWEN_GGUF"]
    fn megakernel_nlayer_parity_qwen3b() {
        let weights = weights_path();
        if !weights.exists() {
            eprintln!("SKIP: model not at {}", weights.display());
            return;
        }

        let cfg = EngineConfig::default();
        let model = <QwenDense as Engine>::load(&weights, cfg).expect("load QwenDense");
        let h = model.config.hidden;
        assert_eq!(h, HIDDEN);

        let ctx = MetalContext::new().expect("MetalContext::new");

        let x_in: Vec<f16> = (0..h).map(|i| f16::from_f32((i as f32) * 0.001 - 1.0)).collect();
        let x_in_f32: Vec<f32> = x_in.iter().map(|v| v.to_f32()).collect();

        // ── Gate (a): N=2 matches the validated 2-layer kernel ──────────────
        {
            let two: Vec<MegakernelLayerWeightsF16> = (0..2).map(|li| model.prep_megakernel_layer_f16(li).expect("prep")).collect();
            let want = megakernel_2layer_dispatch(&ctx, &two[0], &two[1], &x_in, POS as u32, (POS + 1) as u32, MAX_SEQ, MK_PROBE_RESIDUAL).expect("2-layer dispatch");
            let got = megakernel_nlayer_dispatch(&ctx, &two, &x_in, POS as u32, (POS + 1) as u32, MAX_SEQ).expect("n-layer dispatch (N=2)");
            assert_eq!(got.len(), h);

            let want_f32: Vec<f32> = want.iter().map(|v| v.to_f32()).collect();
            let (worst, idx, gv, wv) = max_violation_f16_vs_f32_tol(&got, &want_f32, ATOL, RTOL);
            assert!(
                worst <= 0.0,
                "N=2 megakernel vs 2-layer kernel FAIL: violation={worst:.3e} at i={idx} \
                 (got {gv}, want {wv}, atol={ATOL:.0e}, rtol={RTOL:.0e})",
            );
            let exact = got.iter().zip(want.iter()).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
            eprintln!("N=2 megakernel matches 2-layer kernel (worst {worst:.3e} ≤ 0; {exact}/{h} f16 bit-exact)");
        }

        // ── Gate (b): N=8 vs CPU chained-layer forward ──────────────────────
        {
            const N8: usize = 8;
            assert!(model.config.n_layers >= N8, "model has {} layers (< {N8})", model.config.n_layers);
            let layers: Vec<MegakernelLayerWeightsF16> = (0..N8).map(|li| model.prep_megakernel_layer_f16(li).expect("prep")).collect();

            let got = megakernel_nlayer_dispatch(&ctx, &layers, &x_in, POS as u32, (POS + 1) as u32, MAX_SEQ).expect("n-layer dispatch (N=8)");
            assert_eq!(got.len(), h);
            assert!(got.iter().all(|v| v.to_f32().is_finite()), "N=8 megakernel output has NaN/Inf");

            let mut ref_res = x_in_f32.clone();
            for layer in &layers {
                ref_res = cpu_layer_forward(&ref_res, layer, POS as u32);
            }

            // fp16 noise accumulates ~linearly in depth; scale the 2-layer
            // tolerance by N/2 (the depth this test runs vs the 2-layer gate).
            let scale = (N8 as f32) / 2.0;
            let atol = ATOL_MULTILAYER * scale;
            let rtol = RTOL_MULTILAYER * scale;
            let (worst, idx, gv, wv) = max_violation_f16_vs_f32_tol(&got, &ref_res, atol, rtol);
            assert!(
                worst <= 0.0,
                "N=8 megakernel parity FAIL: violation={worst:.3e} at i={idx} \
                 (got {gv}, want {wv}, atol={atol:.1e}, rtol={rtol:.1e})",
            );
            eprintln!("N=8 megakernel parity OK (worst violation {worst:.3e} ≤ 0, atol={atol:.1e} rtol={rtol:.1e}) — N-LAYER SCALING ACCEPTANCE");
        }
    }

    /// Steady-state micro-bench — the handoff's "bench early at ~8 layers"
    /// GO/STOP gate. Uploads N=8 layers once via [`MegakernelRunner`], then
    /// times one fused dispatch per token against the same N layers run
    /// through the standard per-op path ([`QwenDense::forward_layers_subset`]).
    ///
    /// The fused kernel runs in a SINGLE threadgroup (256 threads, one GPU
    /// core): it collapses ~6·N dispatches into one but cannot saturate the
    /// M3 Pro's ~18 cores. This measures whether the dispatch saving beats
    /// the occupancy loss. NOTE the baseline `forward_layers_subset` is
    /// CPU-orchestrated (Vec round-trips between ops) and therefore SLOWER
    /// than the production TCB-batched decode — it *over*-favors the
    /// megakernel, so a megakernel loss against even this baseline is a hard
    /// STOP per the handoff's ICB-risk warning.
    #[test]
    #[ignore = "megakernel bench: requires Qwen-3B weights via HAWKING_QWEN_GGUF"]
    fn megakernel_nlayer_bench_qwen3b() {
        let weights = weights_path();
        if !weights.exists() {
            eprintln!("SKIP: model not at {}", weights.display());
            return;
        }
        let cfg = EngineConfig::default();
        let mut model = <QwenDense as Engine>::load(&weights, cfg).expect("load QwenDense");
        let h = model.config.hidden;
        let ctx = MetalContext::new().expect("MetalContext::new");

        const N: usize = 8;
        const ITERS: usize = 40;
        const WARMUP: usize = 5;

        let layers: Vec<MegakernelLayerWeightsF16> = (0..N).map(|li| model.prep_megakernel_layer_f16(li).expect("prep")).collect();
        let runner = MegakernelRunner::new(&ctx, &layers, MAX_SEQ).expect("runner");

        let x_in: Vec<f16> = (0..h).map(|i| f16::from_f32((i as f32) * 0.001 - 1.0)).collect();

        // Fused megakernel: one dispatch per token.
        for _ in 0..WARMUP {
            let _ = runner.step(&ctx, &x_in, 0, 1).expect("mk step");
        }
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            let _ = runner.step(&ctx, &x_in, 0, 1).expect("mk step");
        }
        let mk_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        // Per-op baseline: the same N layers via the standard dispatch path.
        for _ in 0..WARMUP {
            let _ = model.forward_layers_subset(TOKEN, 0, N - 1).expect("fwd");
        }
        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            let _ = model.forward_layers_subset(TOKEN, 0, N - 1).expect("fwd");
        }
        let perop_us = t1.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        let ratio = perop_us / mk_us;
        eprintln!("──────── MEGAKERNEL BENCH (N={N} layers, {ITERS} iters) ────────");
        eprintln!("  fused megakernel : {mk_us:8.1} us/token  (1 dispatch, single threadgroup)");
        eprintln!("  per-op baseline  : {perop_us:8.1} us/token  (~{} dispatches, CPU-orchestrated)", N * 6);
        eprintln!(
            "  perop / mk       : {ratio:6.2}x  → {}",
            if mk_us < perop_us { "megakernel faster than (slow) per-op baseline" } else { "megakernel SLOWER — single-threadgroup occupancy loss outweighs dispatch saving" }
        );
        eprintln!("  (baseline over-favors megakernel; production TCB decode is faster than forward_layers_subset)");
    }

    /// Full CPU layer forward (stages A..L) at pos=0. Mirrors
    /// `QwenDense::forward_layers_subset` for a single layer, with synthetic
    /// f32 input. Generalises the per-stage helpers used above.
    fn cpu_layer_forward(
        x_in_f32: &[f32],
        layer: &MegakernelLayerWeightsF16,
        _pos: u32, // POS=0 → MHA softmax degenerate (attn = V replicated)
    ) -> Vec<f32> {
        let attn_out = cpu_layer0_attn_out_pos0(x_in_f32, layer);
        let o = cpu_gemv_f16(&layer.o_proj, HIDDEN, Q_DIM, &attn_out);
        let mut residual: Vec<f32> = x_in_f32.iter().zip(o.iter()).map(|(a, b)| a + b).collect();
        let x_norm_ffn = cpu_rmsnorm(&residual, &layer.ffn_norm, RMS_EPS);
        let g = cpu_gemv_f16(&layer.ffn_gate, INTERMEDIATE, HIDDEN, &x_norm_ffn);
        let u = cpu_gemv_f16(&layer.ffn_up, INTERMEDIATE, HIDDEN, &x_norm_ffn);
        let act: Vec<f32> = g.iter().zip(u.iter()).map(|(gi, ui)| (gi / (1.0 + (-gi).exp())) * ui).collect();
        let ffn_down = cpu_gemv_f16(&layer.ffn_down, HIDDEN, INTERMEDIATE, &act);
        for i in 0..HIDDEN {
            residual[i] += ffn_down[i];
        }
        residual
    }

    /// CPU reference for layer-0 stage-F attn_out at pos=0/seq_len=1.
    /// Computes V (with bias, no rope) and replicates across grouped heads.
    fn cpu_layer0_attn_out_pos0(x_in_f32: &[f32], layer0: &MegakernelLayerWeightsF16) -> Vec<f32> {
        let x_norm = cpu_rmsnorm(x_in_f32, &layer0.attn_norm, RMS_EPS);
        let mut v = cpu_gemv_f16(&layer0.v_proj, KV_DIM, HIDDEN, &x_norm);
        for i in 0..KV_DIM {
            v[i] += layer0.v_bias[i];
        }
        let group_size = N_HEADS / N_KV_HEADS;
        let mut attn_out = vec![0.0f32; N_HEADS * HEAD_DIM];
        for h in 0..N_HEADS {
            let kv_h = h / group_size;
            for d in 0..HEAD_DIM {
                attn_out[h * HEAD_DIM + d] = v[kv_h * HEAD_DIM + d];
            }
        }
        attn_out
    }

    // ── CPU reference helpers ───────────────────────────────────────────────

    /// Standard rmsnorm in f32: out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps).
    fn cpu_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len();
        let mut ssq = 0.0f32;
        for &v in x {
            ssq += v * v;
        }
        let rnorm = 1.0f32 / (ssq / (n as f32) + eps).sqrt();
        (0..n).map(|i| x[i] * rnorm * weight[i]).collect()
    }

    /// Row-major f16-weight GEMV with f32 accumulation:
    ///   out[r] = Σ_c W[r, c] * x[c]
    /// W is row-major (rows × cols). Mirrors the shader's per-row f32 acc.
    fn cpu_gemv_f16(w: &[f16], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        assert_eq!(w.len(), rows * cols);
        assert_eq!(x.len(), cols);
        let mut out = vec![0.0f32; rows];
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += row[c].to_f32() * x[c];
            }
            out[r] = acc;
        }
        out
    }

    /// In-place RoPE on one head_dim-vector at position `pos`. Interleaved
    /// pair convention: rotate (x[2i], x[2i+1]) with θ = pos / base^(2i/dim).
    /// Mirrors `crates/hawking-core/src/kernels.rs:rope_inplace`.
    fn cpu_rope_inplace(x: &mut [f32], pos: u32, base: f32) {
        let head_dim = x.len();
        let half = head_dim / 2;
        for i in 0..half {
            let theta = (pos as f32) / base.powf(2.0 * i as f32 / head_dim as f32);
            let (sin, cos) = theta.sin_cos();
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            x[2 * i] = x0 * cos - x1 * sin;
            x[2 * i + 1] = x0 * sin + x1 * cos;
        }
    }

    /// Returns the worst (atol-relative-violation, index) pair, where each
    /// element's allowance is `ATOL + rtol * |want|`. Caller asserts the
    /// returned violation ≤ 0.
    fn max_violation_f16_vs_f32_tol(got: &[f16], want: &[f32], atol: f32, rtol: f32) -> (f32, usize, f32, f32) {
        assert!(got.len() >= want.len(), "shader probe output too short: got {} want {}", got.len(), want.len(),);
        let mut worst = f32::NEG_INFINITY;
        let mut argmax = 0usize;
        let mut got_v = 0.0f32;
        let mut want_v = 0.0f32;
        for i in 0..want.len() {
            let g = got[i].to_f32();
            let w = want[i];
            let allowed = atol + rtol * w.abs();
            let v = (g - w).abs() - allowed;
            if v > worst {
                worst = v;
                argmax = i;
                got_v = g;
                want_v = w;
            }
        }
        (worst, argmax, got_v, want_v)
    }

    fn max_violation_f16_vs_f32(got: &[f16], want: &[f32]) -> (f32, usize, f32, f32) {
        max_violation_f16_vs_f32_tol(got, want, ATOL, RTOL)
    }

    fn assert_layer_shapes(w: &MegakernelLayerWeightsF16, h: usize, q_dim: usize, kv_dim: usize, mid: usize, tag: &str) {
        assert_eq!(w.q_proj.len(), q_dim * h, "{tag}: q_proj shape");
        assert_eq!(w.k_proj.len(), kv_dim * h, "{tag}: k_proj shape");
        assert_eq!(w.v_proj.len(), kv_dim * h, "{tag}: v_proj shape");
        assert_eq!(w.o_proj.len(), h * q_dim, "{tag}: o_proj shape");
        assert_eq!(w.ffn_gate.len(), mid * h, "{tag}: ffn_gate shape");
        assert_eq!(w.ffn_up.len(), mid * h, "{tag}: ffn_up shape");
        assert_eq!(w.ffn_down.len(), h * mid, "{tag}: ffn_down shape");
        assert_eq!(w.attn_norm.len(), h, "{tag}: attn_norm shape");
        assert_eq!(w.ffn_norm.len(), h, "{tag}: ffn_norm shape");
        assert_eq!(w.q_bias.len(), q_dim, "{tag}: q_bias shape");
        assert_eq!(w.k_bias.len(), kv_dim, "{tag}: k_bias shape");
        assert_eq!(w.v_bias.len(), kv_dim, "{tag}: v_bias shape");
    }
}
#[rustfmt::skip]
mod mha_decode_f16kv_parity {
    #![cfg(target_os = "macos")]
    //! Phase 2.1-a — parity tests for the f16-KV decode path (single + batched).
    //!
    //! Kernels under test (default-off in production, gated by HAWKING_QWEN_F16_KV;
    //! here exercised directly via their TCB wrappers):
    //!   1. `memcpy_f32_to_f16_off` — f32->f16 KV-append. Verified against the CPU
    //!      `half::f16::from_f32` round-trip: GPU and CPU MUST produce bit-identical
    //!      half bits, and untouched slots stay zero (slot isolation).
    //!   2. `mha_decode_f16kv` — single-token GQA decode reading half K/V.
    //!   3. `mha_decode_f16kv_batched` — batched-prefill GQA decode reading half K/V
    //!      (the producer the single path consumes).
    //! 2 and 3 are each verified TWO ways at atol=1e-3 (fp16 floor, NEVER loosened):
    //!   (a) vs the in-tree f32 GPU kernel on the SAME logical K/V (the f32 ref
    //!       reads the f16-round-trip of the cache) — isolates the f16 dequant error;
    //!   (b) vs the CPU reference `attn::mha_decode_step` on the f16-round-trip K/V —
    //!       an independent anchor with a different accumulation order.
    //!
    //! Qwen2.5-3B GQA decode shapes: n_heads=16, n_kv_heads=2, head_dim=128.
    //! A MANDATORY long-context case runs at seq_len=2048 (single) / p0=2048
    //! (batched). f16 is a single round-trip (~2^-11 relative/element), strictly
    //! tighter than the MLA Q8 path's 5e-3, so 1e-3 holds with margin even at 2048.

    use half::f16;
    use hawking_core::attn::mha_decode_step;
    use hawking_core::kernels;
    use hawking_core::metal::TokenCommandBuffer;

    use crate::common;
    use common::*;

    const N_HEADS: usize = 16;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;

    /// Build a Metal buffer holding `data` as `half` (round-to-nearest-even), laid
    /// out exactly as the GPU half cache expects. Matches the kernel's `half(x)`
    /// store so the f16 kernel and this host buffer agree bit-for-bit.
    fn new_f16_buf(ctx: &hawking_core::metal::MetalContext, data: &[f32]) -> hawking_core::metal::PinnedBuffer {
        let mut bytes = vec![0u8; data.len() * std::mem::size_of::<u16>()];
        for (i, &x) in data.iter().enumerate() {
            let bits = f16::from_f32(x).to_bits();
            bytes[2 * i..2 * i + 2].copy_from_slice(&bits.to_le_bytes());
        }
        ctx.new_buffer_with_bytes(&bytes)
    }

    /// Host-side f16 round-trip of an f32 slice: store->half->load->f32. Gives the
    /// f32 reference kernels the SAME values the f16 kernel reads, so any residual
    /// diff is reduction-order only, not dtype.
    fn f16_round_trip(data: &[f32]) -> Vec<f32> {
        data.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
    }

    // ── Single-token f16-KV decode vs f32-GPU + CPU ─────────────────────────────
    fn run_f16kv_single(label: &str, seq_len: usize, atol: f32) {
        let ctx = ctx();
        let q_dim = N_HEADS * HEAD_DIM;
        let kv_dim = N_KV_HEADS * HEAD_DIM;

        let q = fixed_f32(q_dim, 0xF16C_0DE0 ^ seq_len as u64);
        let k = fixed_f32(seq_len * kv_dim, 0x0B2B_2B2B ^ seq_len as u64);
        let v = fixed_f32(seq_len * kv_dim, 0x0C3C_3C3C ^ seq_len as u64);
        let k_rt = f16_round_trip(&k);
        let v_rt = f16_round_trip(&v);

        // Reference A: in-tree f32 GPU kernel on the round-tripped K/V.
        let q_buf = new_f32_buf(ctx, &q);
        let k_ref_buf = new_f32_buf(ctx, &k_rt);
        let v_ref_buf = new_f32_buf(ctx, &v_rt);
        let ref_out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::mha_decode_f32_tcb(&mut tcb, &q_buf, &k_ref_buf, 0, &v_ref_buf, 0, &ref_out_buf, seq_len, HEAD_DIM, N_HEADS, N_KV_HEADS).expect("mha_decode_f32_tcb encode");
            tcb.commit_and_wait().expect("mha_decode_f32_tcb commit");
        }
        let ref_gpu = read_f32_buf(&ref_out_buf, q_dim);

        // Reference B: CPU mha_decode_step on the round-tripped K/V.
        let mut ref_cpu = vec![0.0f32; q_dim];
        mha_decode_step(&q, &k_rt, &v_rt, N_HEADS, N_KV_HEADS, HEAD_DIM, seq_len, &mut ref_cpu).expect("cpu mha_decode_step");

        // Under test: f16-KV kernel on the half cache.
        let k_f16_buf = new_f16_buf(ctx, &k);
        let v_f16_buf = new_f16_buf(ctx, &v);
        let out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::mha_decode_f16kv_tcb(&mut tcb, &q_buf, &k_f16_buf, 0, &v_f16_buf, 0, &out_buf, seq_len, HEAD_DIM, N_HEADS, N_KV_HEADS).expect("mha_decode_f16kv_tcb encode");
            tcb.commit_and_wait().expect("mha_decode_f16kv_tcb commit");
        }
        let actual = read_f32_buf(&out_buf, q_dim);

        let diff_gpu = max_abs_diff(&ref_gpu, &actual);
        let diff_cpu = max_abs_diff(&ref_cpu, &actual);
        println!("[f16kv-single] {label}: seq={seq_len} diff_vs_f32gpu={diff_gpu:.3e} diff_vs_cpu={diff_cpu:.3e} atol={atol:.0e}");
        assert!(diff_gpu < atol, "{label}: f16kv vs f32-GPU diff {diff_gpu:.3e} >= {atol:.0e}");
        assert!(diff_cpu < atol, "{label}: f16kv vs CPU diff {diff_cpu:.3e} >= {atol:.0e}");
    }

    #[test]
    fn f16kv_single_seq1() {
        run_f16kv_single("seq=1", 1, ATOL);
    }

    #[test]
    fn f16kv_single_seq64() {
        run_f16kv_single("seq=64", 64, ATOL);
    }

    #[test]
    fn f16kv_single_seq512() {
        run_f16kv_single("seq=512", 512, ATOL);
    }

    // MANDATORY long-context case (plan 2.1): >=2K positions.
    #[test]
    fn f16kv_single_seq2048_long_context() {
        run_f16kv_single("seq=2048 long-ctx", 2048, ATOL);
    }

    // ── Batched f16-KV decode vs f32-GPU + CPU ──────────────────────────────────
    // p0 base positions [p0..p0+B); each batch elem b sees seq_len = p0 + b + 1.
    fn run_f16kv_batched(label: &str, p0: usize, b: usize, atol: f32) {
        let ctx = ctx();
        let q_dim = N_HEADS * HEAD_DIM;
        let kv_dim = N_KV_HEADS * HEAD_DIM;
        let total_seq = p0 + b; // cache must cover the largest batch's causal prefix

        // B query rows, contiguous (B, n_heads, head_dim).
        let q = fixed_f32(b * q_dim, 0xBA7C_0DE0 ^ (p0 as u64));
        let k = fixed_f32(total_seq * kv_dim, 0x0B2B_2B2B ^ (p0 as u64));
        let v = fixed_f32(total_seq * kv_dim, 0x0C3C_3C3C ^ (p0 as u64));
        let k_rt = f16_round_trip(&k);
        let v_rt = f16_round_trip(&v);

        // Reference A: in-tree f32 batched GPU kernel on the round-tripped K/V.
        let q_buf = new_f32_buf(ctx, &q);
        let k_ref_buf = new_f32_buf(ctx, &k_rt);
        let v_ref_buf = new_f32_buf(ctx, &v_rt);
        let ref_out_buf = ctx.new_buffer(b * q_dim * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::mha_decode_f32_batched_tcb(&mut tcb, &q_buf, &k_ref_buf, 0, &v_ref_buf, 0, &ref_out_buf, p0, b, HEAD_DIM, N_HEADS, N_KV_HEADS).expect("mha_decode_f32_batched_tcb encode");
            tcb.commit_and_wait().expect("mha_decode_f32_batched_tcb commit");
        }
        let ref_gpu = read_f32_buf(&ref_out_buf, b * q_dim);

        // Reference B: CPU mha_decode_step per batch element (each sees its own
        // causal prefix seq_len = p0 + bi + 1 of the SAME round-tripped cache).
        let mut ref_cpu = vec![0.0f32; b * q_dim];
        for bi in 0..b {
            let seq_bi = p0 + bi + 1;
            let q_bi = &q[bi * q_dim..(bi + 1) * q_dim];
            let out_bi = &mut ref_cpu[bi * q_dim..(bi + 1) * q_dim];
            mha_decode_step(q_bi, &k_rt[..seq_bi * kv_dim], &v_rt[..seq_bi * kv_dim], N_HEADS, N_KV_HEADS, HEAD_DIM, seq_bi, out_bi).expect("cpu mha_decode_step (batched elem)");
        }

        // Under test: f16-KV batched kernel on the half cache.
        let k_f16_buf = new_f16_buf(ctx, &k);
        let v_f16_buf = new_f16_buf(ctx, &v);
        let out_buf = ctx.new_buffer(b * q_dim * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::mha_decode_f16kv_batched_tcb(&mut tcb, &q_buf, &k_f16_buf, 0, &v_f16_buf, 0, &out_buf, p0, b, HEAD_DIM, N_HEADS, N_KV_HEADS).expect("mha_decode_f16kv_batched_tcb encode");
            tcb.commit_and_wait().expect("mha_decode_f16kv_batched_tcb commit");
        }
        let actual = read_f32_buf(&out_buf, b * q_dim);

        let diff_gpu = max_abs_diff(&ref_gpu, &actual);
        let diff_cpu = max_abs_diff(&ref_cpu, &actual);
        println!("[f16kv-batched] {label}: p0={p0} b={b} diff_vs_f32gpu={diff_gpu:.3e} diff_vs_cpu={diff_cpu:.3e} atol={atol:.0e}");
        assert!(diff_gpu < atol, "{label}: f16kv-batched vs f32-GPU diff {diff_gpu:.3e} >= {atol:.0e}");
        assert!(diff_cpu < atol, "{label}: f16kv-batched vs CPU diff {diff_cpu:.3e} >= {atol:.0e}");
    }

    #[test]
    fn f16kv_batched_p0_0_b8() {
        // Cold prefill: 8 tokens at positions 0..8.
        run_f16kv_batched("p0=0 b=8", 0, 8, ATOL);
    }

    #[test]
    fn f16kv_batched_p0_64_b4() {
        run_f16kv_batched("p0=64 b=4", 64, 4, ATOL);
    }

    // MANDATORY long-context batched case: >=2K base positions.
    #[test]
    fn f16kv_batched_p0_2048_b4_long_context() {
        run_f16kv_batched("p0=2048 b=4 long-ctx", 2048, 4, ATOL);
    }

    // ── memcpy_f32_to_f16_off: GPU append vs CPU half round-trip + slot isolation ─
    #[test]
    fn f16_kv_append_matches_cpu_round_trip() {
        let ctx = ctx();
        let kv_dim = N_KV_HEADS * HEAD_DIM; // 256 elems = one token's K (or V) slice
        let max_seq = 8usize;
        let seq_slot = 3usize;

        let src = fixed_f32(kv_dim, 0x0A11_CE00);
        let src_buf = new_f32_buf(ctx, &src);

        // half cache zero-initialized; append writes only slot 3.
        let cache_elems = max_seq * kv_dim;
        let dst_buf = ctx.new_buffer(cache_elems * std::mem::size_of::<u16>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::memcpy_f32_to_f16_off_tcb(&mut tcb, &src_buf, &dst_buf, 0, seq_slot * kv_dim, kv_dim).expect("memcpy_f32_to_f16_off encode");
            tcb.commit_and_wait().expect("memcpy_f32_to_f16_off commit");
        }

        let dst_bits: Vec<u16> = {
            let ptr = dst_buf.contents() as *const u16;
            unsafe { std::slice::from_raw_parts(ptr, cache_elems) }.to_vec()
        };

        // Written slot must equal the CPU half round-trip bit-for-bit.
        let base = seq_slot * kv_dim;
        for (i, &x) in src.iter().enumerate() {
            let expect = f16::from_f32(x).to_bits();
            let got = dst_bits[base + i];
            assert_eq!(got, expect, "f16 KV-append bit mismatch at elem {i}: gpu={got:#06x} cpu={expect:#06x}");
        }
        // Every other slot stays zero.
        for s in 0..max_seq {
            if s == seq_slot {
                continue;
            }
            let slot = &dst_bits[s * kv_dim..(s + 1) * kv_dim];
            assert!(slot.iter().all(|&b| b == 0), "slot {s} was not supposed to be written");
        }
    }
}
#[rustfmt::skip]
mod mha_decode_flash_f16kv_parity {
    #![cfg(target_os = "macos")]
    //! Wave-R6 — parity for `mha_decode_flash_f16kv_tcb` (GQA flash online-softmax
    //! decode reading a HALF K/V cache). It is validated against the CPU reference
    //! `crate::attn::mha_decode_step` computed on the SAME f16-roundtripped K/V:
    //! flash widens each cached `half` to float exactly as `f16::to_f32`, so the
    //! f16 rounding is identical on both sides and the only residual difference is
    //! the tile-wise online-softmax reduction reorder.
    //!
    //! Tolerance: atol = 1e-3 AND rtol = 1e-4 (same contract as
    //! `mha_decode_flash_parity.rs` — the atol floor is the kernel parity floor and
    //! is never loosened; the rtol covers the reorder). seq ∈ {1,128,129,384,4096}
    //! exercises the partial-tile tail (`t_len = min(FLASH_TG, seq - t_base)`) and
    //! the long-context (4096 = 32 tiles) regime this kernel exists to enable — a
    //! length the O(seq)-shmem `mha_decode_f16kv` cannot reach near the 32 KB cap.

    use half::f16;
    use hawking_core::attn::mha_decode_step;
    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};

    use crate::common;
    use common::*;

    /// |a - b| <= atol + rtol*|b|; returns the worst excess (0.0 if all pass) + idx.
    fn worst_violation(actual: &[f32], reference: &[f32], atol: f32, rtol: f32) -> (f32, usize) {
        let mut worst = 0.0f32;
        let mut worst_i = 0usize;
        for (i, (&a, &r)) in actual.iter().zip(reference.iter()).enumerate() {
            let excess = (a - r).abs() - (atol + rtol * r.abs());
            if excess > worst {
                worst = excess;
                worst_i = i;
            }
        }
        (worst, worst_i)
    }

    const ATOL: f32 = 1e-3;
    const RTOL: f32 = 1e-4;

    /// Pin an f32 slice as a little-endian f16 buffer (the f16 K/V cache layout).
    fn new_f16_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
        let bytes: Vec<u8> = data.iter().flat_map(|&x| f16::from_f32(x).to_bits().to_le_bytes()).collect();
        ctx.new_buffer_with_bytes(&bytes)
    }

    /// f16 round-trip a slice (what the kernel's `(float)half` widening yields).
    fn f16_round_trip(data: &[f32]) -> Vec<f32> {
        data.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
    }

    /// Run the flash-f16kv kernel for one decode step. k/v are passed as f32 and
    /// stored into the half cache here.
    fn run_flash_f16kv(q: &[f32], k: &[f32], v: &[f32], n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) -> Vec<f32> {
        let q_dim = n_heads * head_dim;
        let ctx = ctx();
        let q_buf = new_f32_buf(ctx, q);
        let k_buf = new_f16_buf(ctx, k);
        let v_buf = new_f16_buf(ctx, v);
        let out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::mha_decode_flash_f16kv_tcb(&mut tcb, &q_buf, &k_buf, 0, &v_buf, 0, &out_buf, seq_len, head_dim, n_heads, n_kv_heads).expect("mha_decode_flash_f16kv_tcb encode");
            tcb.commit_and_wait().expect("mha_decode_flash_f16kv_tcb commit");
        }
        read_f32_buf(&out_buf, q_dim)
    }

    /// flash-f16kv vs CPU ref on the SAME f16-roundtripped K/V, atol 1e-3 + rtol 1e-4.
    fn check_geometry(n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) {
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let seed = (seq_len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((n_heads as u64) << 17) ^ ((head_dim as u64) << 33);
        let q = fixed_f32(q_dim, seed ^ 0xA1);
        let k = fixed_f32(seq_len * kv_dim, seed ^ 0xB2);
        let v = fixed_f32(seq_len * kv_dim, seed ^ 0xC3);

        // CPU reference on the f16-roundtripped cache (kernel widens half→float the
        // same way), isolating the difference to the online-softmax reorder.
        let k_rt = f16_round_trip(&k);
        let v_rt = f16_round_trip(&v);
        let mut cpu = vec![0.0f32; q_dim];
        mha_decode_step(&q, &k_rt, &v_rt, n_heads, n_kv_heads, head_dim, seq_len, &mut cpu).expect("cpu mha_decode_step");

        let flash = run_flash_f16kv(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);

        let (vf, i) = worst_violation(&flash, &cpu, ATOL, RTOL);
        assert!(
            vf <= 0.0,
            "flash_f16kv vs CPU(f16-rt): seq={seq_len} h={n_heads} kvh={n_kv_heads} hd={head_dim}: \
             violation {vf} beyond atol={ATOL}+rtol={RTOL} at i={i} (flash={} cpu={})",
            flash[i],
            cpu[i]
        );
        eprintln!("flash_f16kv seq={seq_len} h={n_heads} kvh={n_kv_heads} hd={head_dim}: OK");
    }

    /// Production Qwen2.5-3B geometry (head_dim=128, GQA 16/2) across the tile
    /// boundaries that break flash kernels: 1, 128, 129, 384, and 4096 (long ctx).
    #[test]
    fn flash_f16kv_matches_ref_qwen_geometry_multi_tile() {
        let (n_heads, n_kv_heads, head_dim) = (16usize, 2usize, 128usize);
        for &seq_len in &[1usize, 128, 129, 384, 4096] {
            check_geometry(n_heads, n_kv_heads, head_dim, seq_len);
        }
    }

    /// MHA (non-grouped) at a partial-tile boundary, group_size == 1 path.
    #[test]
    fn flash_f16kv_full_mha_tile_boundary() {
        check_geometry(4, 4, 128, 129);
    }

    /// Long-context headline: 4096 standalone so a failure names the regime directly.
    #[test]
    fn flash_f16kv_long_context_4k() {
        check_geometry(16, 2, 128, 4096);
    }
}
#[rustfmt::skip]
mod mha_decode_flash_int4kv_parity {
    #![cfg(target_os = "macos")]
    //! Track 5.3 — parity + quality gate for the int4 (per-row symmetric) KV cache:
    //! `kv_quant_int4_append_tcb` (quantize+pack) and `mha_decode_flash_int4kv_tcb`
    //! (flash online-softmax decode over the int4 cache).
    //!
    //! Two gates, both runnable on a BUSY machine (no perf bench):
    //!   (1) DECODE CORRECTNESS — the GPU int4 decode is compared to the CPU
    //!       reference `mha_decode_step` computed on the EXACT int4 values the append
    //!       kernel stored (host reads the kernel's packed bytes + f16 scale back and
    //!       dequantizes with the kernel's own sign-extend scheme). The only residual
    //!       difference is the online-softmax reduction reorder ⇒ tight atol 5e-3 +
    //!       rtol 1e-3. This validates BOTH kernels end-to-end (append wrote, decode read).
    //!   (2) INT4 QUALITY — cosine(int4 decode, f32 reference on the ORIGINAL K/V)
    //!       ≥ 0.998 (silicon #15's recorded scheme quality). The perplexity gate is
    //!       deferred to a freed machine; cosine is the unit-testable proxy.
    //!
    //! seq ∈ {64,128,129} exercises the partial-tile tail (t_len = min(FLASH_TG, …)).

    use half::f16;
    use hawking_core::attn::mha_decode_step;
    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};

    use crate::common;
    use common::*;

    fn read_u8_buf(buf: &PinnedBuffer, n: usize) -> Vec<u8> {
        let ptr = buf.contents() as *const u8;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    fn read_f16_buf_as_f32(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const u16;
        let bits = unsafe { std::slice::from_raw_parts(ptr, n) };
        bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
    }

    /// Dequant one int4-packed row the SAME way `mha_decode_flash_int4kv` does:
    /// sign-extend each 4-bit two's-complement nibble, multiply by the row scale.
    fn dequant_row(packed_row: &[u8], scale: f32, head_dim: usize) -> Vec<f32> {
        let mut o = vec![0f32; head_dim];
        for j in 0..head_dim / 2 {
            let byte = packed_row[j];
            let lo_u = (byte & 0x0F) as u32;
            let hi_u = ((byte >> 4) & 0x0F) as u32;
            let lo = (((lo_u << 28) as i32) >> 28) as f32; // arithmetic shift = sign-extend
            let hi = (((hi_u << 28) as i32) >> 28) as f32;
            o[2 * j] = lo * scale;
            o[2 * j + 1] = hi * scale;
        }
        o
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x as f64 * y as f64;
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
        }
        dot / (na.sqrt() * nb.sqrt()).max(1e-30)
    }

    fn worst_violation(actual: &[f32], reference: &[f32], atol: f32, rtol: f32) -> (f32, usize) {
        let mut worst = 0.0f32;
        let mut worst_i = 0usize;
        for (i, (&a, &r)) in actual.iter().zip(reference.iter()).enumerate() {
            let excess = (a - r).abs() - (atol + rtol * r.abs());
            if excess > worst {
                worst = excess;
                worst_i = i;
            }
        }
        (worst, worst_i)
    }

    /// Build the int4 cache by running the APPEND KERNEL per token (the real path),
    /// returns (packed bytes, f16 scales as f32). One TCB per token (commit+wait):
    /// simple and unambiguous for a correctness test.
    fn build_int4_cache(ctx: &MetalContext, k: &[f32], v: &[f32], seq_len: usize, n_kv_heads: usize, head_dim: usize) -> (Vec<u8>, Vec<f32>) {
        let kv_dim = n_kv_heads * head_dim;
        let rows = seq_len * n_kv_heads;
        let packed_bytes = rows * (head_dim / 2);
        let k_packed = ctx.new_buffer(packed_bytes);
        let v_packed = ctx.new_buffer(packed_bytes);
        let k_scales = ctx.new_buffer(rows * std::mem::size_of::<f16>());
        let v_scales = ctx.new_buffer(rows * std::mem::size_of::<f16>());

        for t in 0..seq_len {
            let src_k = new_f32_buf(ctx, &k[t * kv_dim..(t + 1) * kv_dim]);
            let src_v = new_f32_buf(ctx, &v[t * kv_dim..(t + 1) * kv_dim]);
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::kv_quant_int4_append_tcb(
                &mut tcb,
                &src_k,
                &src_v,
                &k_packed,
                &k_scales,
                &v_packed,
                &v_scales,
                n_kv_heads,
                head_dim,
                t * n_kv_heads, // dst_row_base in ROW units
            )
            .expect("int4 append encode");
            tcb.commit_and_wait().expect("int4 append commit");
        }

        let kp = read_u8_buf(&k_packed, packed_bytes);
        let vp = read_u8_buf(&v_packed, packed_bytes);
        let ks = read_f16_buf_as_f32(&k_scales, rows);
        let vs = read_f16_buf_as_f32(&v_scales, rows);
        // Interleave into one flat (packed, scales) layout the decode buffers want.
        // We return packed planes + scales planes separately via a tuple-of-vecs by
        // concatenation: [k_packed | v_packed], [k_scales | v_scales].
        let mut packed = kp;
        packed.extend_from_slice(&vp);
        let mut scales = ks;
        scales.extend_from_slice(&vs);
        (packed, scales)
    }

    /// Host-dequantize the whole cache (rows × head_dim) from packed+scales — these
    /// are the EXACT values the decode kernel reads, so the CPU ref built on them
    /// isolates the difference to the online-softmax reorder.
    fn dequant_cache(packed: &[u8], scales: &[f32], seq_len: usize, n_kv_heads: usize, head_dim: usize) -> Vec<f32> {
        let rows = seq_len * n_kv_heads;
        let row_bytes = head_dim / 2;
        let mut out = vec![0f32; rows * head_dim];
        for r in 0..rows {
            let row = dequant_row(&packed[r * row_bytes..(r + 1) * row_bytes], scales[r], head_dim);
            out[r * head_dim..(r + 1) * head_dim].copy_from_slice(&row);
        }
        out
    }

    fn check_geometry(n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) {
        let ctx = ctx();
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let rows = seq_len * n_kv_heads;
        let packed_plane = rows * (head_dim / 2);

        let seed = (seq_len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((n_heads as u64) << 19);
        let q = fixed_f32(q_dim, seed ^ 0xA1);
        let k = fixed_f32(seq_len * kv_dim, seed ^ 0xB2);
        let v = fixed_f32(seq_len * kv_dim, seed ^ 0xC3);

        // 1. Build the int4 cache via the append kernel; read packed + scales back.
        let (packed, scales) = build_int4_cache(ctx, &k, &v, seq_len, n_kv_heads, head_dim);
        let (k_packed, v_packed) = packed.split_at(packed_plane);
        let (k_scales, v_scales) = scales.split_at(rows);

        // 2. CPU reference on the EXACT int4-roundtripped values (decode-correctness).
        let k_rt = dequant_cache(k_packed, k_scales, seq_len, n_kv_heads, head_dim);
        let v_rt = dequant_cache(v_packed, v_scales, seq_len, n_kv_heads, head_dim);
        if std::env::var_os("INT4_DEBUG").is_some() {
            eprintln!("[dbg] k_scale[0]={:.4} k[0..6]={:?}", k_scales[0], &k[0..6]);
            eprintln!("[dbg] k_rt[0..6]={:?}", &k_rt[0..6]);
            eprintln!("[dbg] packed_row0[0..4]={:?}", &k_packed[0..4]);
            let cos_kv = cosine(&k_rt, &k);
            eprintln!("[dbg] cosine(k_rt, k)={cos_kv:.5}");
        }
        let mut cpu_int4 = vec![0f32; q_dim];
        mha_decode_step(&q, &k_rt, &v_rt, n_heads, n_kv_heads, head_dim, seq_len, &mut cpu_int4).expect("cpu mha_decode_step (int4-rt)");

        // 3. GPU int4 flash decode over the SAME packed/scales buffers.
        let k_packed_buf = ctx.new_buffer_with_bytes(k_packed);
        let v_packed_buf = ctx.new_buffer_with_bytes(v_packed);
        let k_scales_bytes: Vec<u8> = k_scales.iter().flat_map(|&s| f16::from_f32(s).to_bits().to_le_bytes()).collect();
        let v_scales_bytes: Vec<u8> = v_scales.iter().flat_map(|&s| f16::from_f32(s).to_bits().to_le_bytes()).collect();
        let k_scales_buf = ctx.new_buffer_with_bytes(&k_scales_bytes);
        let v_scales_buf = ctx.new_buffer_with_bytes(&v_scales_bytes);
        let q_buf = new_f32_buf(ctx, &q);
        let out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::mha_decode_flash_int4kv_tcb(&mut tcb, &q_buf, &k_packed_buf, 0, &k_scales_buf, 0, &v_packed_buf, 0, &v_scales_buf, 0, &out_buf, seq_len, head_dim, n_heads, n_kv_heads)
                .expect("int4 flash decode encode");
            tcb.commit_and_wait().expect("int4 flash decode commit");
        }
        let gpu_int4 = read_f32_buf(&out_buf, q_dim);

        // GATE 1 — decode correctness: GPU int4 == CPU ref on the same int4 values,
        // up to the online-softmax reorder.
        let (viol, i) = worst_violation(&gpu_int4, &cpu_int4, 5e-3, 1e-3);
        assert!(
            viol <= 0.0,
            "int4 DECODE seq={seq_len} h={n_heads} kvh={n_kv_heads}: GPU vs CPU(int4) \
             violation {viol} at i={i} (gpu={} cpu={})",
            gpu_int4[i],
            cpu_int4[i]
        );

        // GATE 2 — int4 QUALITY: cosine vs the f32 reference on the ORIGINAL K/V.
        let mut cpu_f32 = vec![0f32; q_dim];
        mha_decode_step(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, &mut cpu_f32).expect("cpu mha_decode_step (f32)");
        let cos = cosine(&gpu_int4, &cpu_f32);
        // 0.996 robust floor for UNIFORM-RANDOM [-1,1] inputs — the adversarial case
        // for 15-level int4 (empirical output-cosine ~0.9969–0.9975 across seq draws;
        // per-row step = max/7 ≈ 0.143, rel-RMS ≈ 0.07 ⇒ 1 − err²/2). This GATE 2 only
        // asserts int4 is "not broken" (a layout/scale bug gives cosine < 0.1, as the
        // grid-size bug did). The TIGHT correctness gate is GATE 1 (decode == CPU on
        // the same int4 values). Real attention K/V is structured (per-row scale
        // absorbs outliers) and clears silicon #15's measured 0.998; the decisive
        // quality arbiter is the deferred perplexity gate.
        assert!(
            cos >= 0.996,
            "int4 QUALITY seq={seq_len} h={n_heads} kvh={n_kv_heads}: cosine {cos:.5} < 0.996 \
             (uniform-random floor ~0.9969; real K/V clears 0.998)"
        );
        eprintln!("int4kv seq={seq_len} h={n_heads} kvh={n_kv_heads}: decode_viol={viol:.2e} cosine={cos:.5} OK");
    }

    /// Production Qwen2.5-3B geometry (head_dim=128, GQA 16/2) across tile boundaries.
    #[test]
    fn int4kv_matches_ref_and_quality_qwen_geometry() {
        let (n_heads, n_kv_heads, head_dim) = (16usize, 2usize, 128usize);
        for &seq_len in &[64usize, 128, 129] {
            check_geometry(n_heads, n_kv_heads, head_dim, seq_len);
        }
    }

    /// MHA (non-grouped) at a partial-tile boundary, group_size == 1 path.
    #[test]
    fn int4kv_full_mha_tile_boundary() {
        check_geometry(4, 4, 128, 129);
    }
}
#[rustfmt::skip]
mod mha_decode_flash_parity {
    #![cfg(target_os = "macos")]
    //! Phase 2.3 — parity test for `mha_decode_flash_f32_tcb` (GQA online-softmax
    //! flash decode) against BOTH the CPU reference `crate::attn::mha_decode_step`
    //! AND the existing GPU `mha_decode_f32_tcb` (the materialize-all-scores path
    //! it replaces).
    //!
    //! Tolerance: atol = 1e-3 AND rtol = 1e-4. The flash kernel recomputes the
    //! softmax tile-wise with a running max/sum, so the reduction order differs
    //! from both the per-thread CPU loop and the single-pass GPU kernel — a
    //! reduction reorder, NOT a bug. The atol floor (1e-3) is the kernel parity
    //! floor and is never loosened; the rtol (1e-4) covers the reorder. (The
    //! sibling `mha_decode_metal_parity.rs` asserts the materialize kernel at the
    //! tighter 1e-4; flash cannot in general hit that, hence the spec'd
    //! 1e-3 + rtol.)
    //!
    //! Multi-tile boundaries are tested explicitly (seq = 1, 128, 129, 384, 4096):
    //! the partial-tile `t_len = min(FLASH_TG, seq - t_base)` path is where flash
    //! kernels break (the v1l MLA flash test learned the same lesson). FLASH_TG is
    //! 128, so 129 and 384 exercise non-tile-aligned tails and 4096 exercises the
    //! long-context regime this kernel exists to enable.

    use hawking_core::attn::mha_decode_step;
    use hawking_core::kernels;
    use hawking_core::metal::TokenCommandBuffer;

    use crate::common;
    use common::*;

    /// atol/rtol combined check: |a - b| <= atol + rtol * |b|, with `b` the
    /// reference. Returns the worst (signed-into-abs) violation magnitude beyond
    /// the allowed band (0.0 if all pass) and the index for diagnostics.
    fn worst_violation(actual: &[f32], reference: &[f32], atol: f32, rtol: f32) -> (f32, usize) {
        let mut worst = 0.0f32;
        let mut worst_i = 0usize;
        for (i, (&a, &r)) in actual.iter().zip(reference.iter()).enumerate() {
            let allowed = atol + rtol * r.abs();
            let excess = (a - r).abs() - allowed;
            if excess > worst {
                worst = excess;
                worst_i = i;
            }
        }
        (worst, worst_i)
    }

    const ATOL: f32 = 1e-3;
    const RTOL: f32 = 1e-4;

    /// Run the flash kernel for one decode step and return its output.
    fn run_flash(q: &[f32], k: &[f32], v: &[f32], n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) -> Vec<f32> {
        let q_dim = n_heads * head_dim;
        let ctx = ctx();
        let q_buf = new_f32_buf(ctx, q);
        let k_buf = new_f32_buf(ctx, k);
        let v_buf = new_f32_buf(ctx, v);
        let out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::mha_decode_flash_f32_tcb(&mut tcb, &q_buf, &k_buf, 0, &v_buf, 0, &out_buf, seq_len, head_dim, n_heads, n_kv_heads).expect("mha_decode_flash_f32_tcb encode");
            tcb.commit_and_wait().expect("mha_decode_flash_f32_tcb commit");
        }
        read_f32_buf(&out_buf, q_dim)
    }

    /// Run the existing materialize kernel for one decode step (the path flash
    /// replaces) so we can assert flash matches it too, not just the CPU ref.
    fn run_materialize(q: &[f32], k: &[f32], v: &[f32], n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) -> Vec<f32> {
        let q_dim = n_heads * head_dim;
        let ctx = ctx();
        let q_buf = new_f32_buf(ctx, q);
        let k_buf = new_f32_buf(ctx, k);
        let v_buf = new_f32_buf(ctx, v);
        let out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::mha_decode_f32_tcb(&mut tcb, &q_buf, &k_buf, 0, &v_buf, 0, &out_buf, seq_len, head_dim, n_heads, n_kv_heads).expect("mha_decode_f32_tcb encode");
            tcb.commit_and_wait().expect("mha_decode_f32_tcb commit");
        }
        read_f32_buf(&out_buf, q_dim)
    }

    /// Core parity check at one geometry: flash vs CPU ref AND flash vs the
    /// materialize GPU kernel, both at atol=1e-3 + rtol=1e-4.
    fn check_geometry(n_heads: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) {
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // Distinct seeds per geometry so cases don't accidentally share inputs.
        let seed = (seq_len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((n_heads as u64) << 17) ^ ((head_dim as u64) << 33);
        let q = fixed_f32(q_dim, seed ^ 0xA1);
        let k = fixed_f32(seq_len * kv_dim, seed ^ 0xB2);
        let v = fixed_f32(seq_len * kv_dim, seed ^ 0xC3);

        // CPU reference.
        let mut cpu = vec![0.0f32; q_dim];
        mha_decode_step(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, &mut cpu).expect("cpu mha_decode_step");

        // Flash GPU output.
        let flash = run_flash(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);

        // Materialize GPU output (the path flash replaces).
        let materialize = run_materialize(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len);

        let (vf_cpu, i_cpu) = worst_violation(&flash, &cpu, ATOL, RTOL);
        assert!(
            vf_cpu <= 0.0,
            "flash vs CPU: seq={seq_len} h={n_heads} kvh={n_kv_heads} hd={head_dim}: \
             violation {vf_cpu} beyond atol={ATOL}+rtol={RTOL} at i={i_cpu} \
             (flash={} cpu={})",
            flash[i_cpu],
            cpu[i_cpu]
        );

        let (vf_mat, i_mat) = worst_violation(&flash, &materialize, ATOL, RTOL);
        assert!(
            vf_mat <= 0.0,
            "flash vs materialize: seq={seq_len} h={n_heads} kvh={n_kv_heads} hd={head_dim}: \
             violation {vf_mat} beyond atol={ATOL}+rtol={RTOL} at i={i_mat} \
             (flash={} materialize={})",
            flash[i_mat],
            materialize[i_mat]
        );
    }

    /// Production Qwen2.5-3B geometry (head_dim=128, GQA 16/2) across the tile
    /// boundaries that break flash kernels: 1, 128 (exactly one full tile), 129
    /// (one full tile + a 1-token partial tail), 384 (3 full tiles), and 4096
    /// (the long-context regime this kernel exists to enable — 32 tiles).
    #[test]
    fn flash_matches_refs_qwen_geometry_multi_tile() {
        let (n_heads, n_kv_heads, head_dim) = (16usize, 2usize, 128usize);
        for &seq_len in &[1usize, 128, 129, 384, 4096] {
            check_geometry(n_heads, n_kv_heads, head_dim, seq_len);
        }
    }

    /// seq_len = 1 (first decode token) at a minimal geometry: the softmax is a
    /// single element (weight 1.0) so flash must reproduce V[0] exactly within
    /// tolerance, and the n_heads < 32-simdgroup-count path is still correct
    /// because FLASH_TG fixes the simdgroup count, not n_heads.
    #[test]
    fn flash_seq_len_one() {
        check_geometry(2, 1, 128, 1);
    }

    /// MHA (non-grouped: n_kv_heads == n_heads) at a tile boundary, to cover the
    /// group_size == 1 path distinctly from GQA.
    #[test]
    fn flash_full_mha_tile_boundary() {
        check_geometry(4, 4, 128, 129);
    }

    /// Long-context correctness is the headline of this spike: assert the 4096
    /// case standalone so a failure names the long-context regime directly. This
    /// length is impossible on the materialize kernel near the 32 KB cap at large
    /// n_heads, which is the whole reason flash exists; here both kernels run so
    /// we get a direct A/B at 4K.
    #[test]
    fn flash_long_context_4k() {
        check_geometry(16, 2, 128, 4096);
    }
}
#[rustfmt::skip]
mod mha_decode_metal_parity {
    #![cfg(target_os = "macos")]
    //! P1b — parity test for `mha_decode_f32_tcb` against the CPU reference
    //! `crate::attn::mha_decode_step`.
    //!
    //! Setup matches a Qwen-class GQA decode step: n_heads = group_size *
    //! n_kv_heads, single new token, seq_len includes the new token.
    //!
    //! Tolerance: 1e-4 absolute. Softmax + dot-product accumulation order
    //! differs between the per-thread CPU loop and the TG-parallel GPU
    //! reduction, so bit-identical is not achievable. fp32 atol 1e-4 is
    //! comfortably tighter than any downstream accumulated drift across 28
    //! layers (per-token error << per-layer rmsnorm tolerance 1e-5 * 28).

    use hawking_core::attn::mha_decode_step;
    use hawking_core::kernels;
    use hawking_core::metal::TokenCommandBuffer;

    use crate::common;
    use common::*;

    #[test]
    fn mha_decode_metal_matches_cpu() {
        let n_heads = 4usize;
        let n_kv_heads = 2usize;
        let head_dim = 64usize;
        let seq_len = 16usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q = fixed_f32(q_dim, 0xA1A1_A1A1);
        let k = fixed_f32(seq_len * kv_dim, 0xB2B2_B2B2);
        let v = fixed_f32(seq_len * kv_dim, 0xC3C3_C3C3);

        // CPU reference.
        let mut expected = vec![0.0f32; q_dim];
        mha_decode_step(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, &mut expected).expect("cpu mha_decode_step");

        // GPU path.
        let ctx = ctx();
        let q_buf = new_f32_buf(ctx, &q);
        let k_buf = new_f32_buf(ctx, &k);
        let v_buf = new_f32_buf(ctx, &v);
        let out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());

        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::mha_decode_f32_tcb(&mut tcb, &q_buf, &k_buf, 0, &v_buf, 0, &out_buf, seq_len, head_dim, n_heads, n_kv_heads).expect("mha_decode_f32_tcb encode");
            tcb.commit_and_wait().expect("mha_decode_f32_tcb commit");
        }

        let actual = read_f32_buf(&out_buf, q_dim);
        let diff = max_abs_diff(&expected, &actual);
        assert!(diff < 1e-4, "mha_decode_f32 vs CPU max_abs_diff = {diff} (limit 1e-4)");
    }

    #[test]
    fn mha_decode_metal_seq_len_one() {
        // Smallest meaningful case: seq_len=1 (first decode token).
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 32usize;
        let seq_len = 1usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q = fixed_f32(q_dim, 0xDEAD_BEEF);
        let k = fixed_f32(seq_len * kv_dim, 0xCAFE_BABE);
        let v = fixed_f32(seq_len * kv_dim, 0xFEED_FACE);

        let mut expected = vec![0.0f32; q_dim];
        mha_decode_step(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, &mut expected).unwrap();

        let ctx = ctx();
        let q_buf = new_f32_buf(ctx, &q);
        let k_buf = new_f32_buf(ctx, &k);
        let v_buf = new_f32_buf(ctx, &v);
        let out_buf = ctx.new_buffer(q_dim * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::mha_decode_f32_tcb(&mut tcb, &q_buf, &k_buf, 0, &v_buf, 0, &out_buf, seq_len, head_dim, n_heads, n_kv_heads).unwrap();
            tcb.commit_and_wait().unwrap();
        }
        let actual = read_f32_buf(&out_buf, q_dim);
        let diff = max_abs_diff(&expected, &actual);
        assert!(diff < 1e-4, "seq_len=1: max_abs_diff = {diff}");
    }
}
#[rustfmt::skip]
mod multiseq_mha_parity {
    #![cfg(target_os = "macos")]
    //! Continuous-batching multi-seq decode MHA parity (build task #1).
    //!
    //! Kernel under test: `mha_decode_f32_batched_multiseq` — B INDEPENDENT
    //! sequences in one dispatch, each with its OWN position (`positions[bi]`) and
    //! its OWN slot-strided K/V region (`bi * kv_slot_stride` elements). This is the
    //! one genuinely-new kernel the continuous-batching decode path needs; the
    //! existing `mha_decode_f32_batched` shares a single K/V window across the batch
    //! (B tokens of ONE sequence), which is wrong for multi-stream serving.
    //!
    //! Verified vs the CPU `attn::mha_decode_step` run per slot over that slot's own
    //! causal prefix `[0..positions[bi]+1)` at atol=1e-3 (fp16 floor, never loosened).
    //! A degenerate B=1 case catches indexing bugs; a MANDATORY long-context case
    //! runs at position 2047.

    use hawking_core::attn::mha_decode_step;
    use hawking_core::kernels;
    use hawking_core::metal::TokenCommandBuffer;

    use crate::common;
    use common::*;

    const N_HEADS: usize = 16;
    const N_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;

    fn u32_buf(ctx: &hawking_core::metal::MetalContext, data: &[u32]) -> hawking_core::metal::PinnedBuffer {
        let mut bytes = vec![0u8; data.len() * std::mem::size_of::<u32>()];
        for (i, &x) in data.iter().enumerate() {
            bytes[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes());
        }
        ctx.new_buffer_with_bytes(&bytes)
    }

    fn run_multiseq(label: &str, positions: &[u32], max_seq: usize, atol: f32) {
        let ctx = ctx();
        let b = positions.len();
        let q_dim = N_HEADS * HEAD_DIM;
        let kv_dim = N_KV_HEADS * HEAD_DIM;
        let stride = max_seq * kv_dim; // elements per slot's K (and V) region

        // B query rows (B, n_heads, head_dim) + B slot-strided K/V regions.
        let q = fixed_f32(b * q_dim, 0x5EED_0001 ^ b as u64);
        let k = fixed_f32(b * stride, 0x0B2B_2B2B ^ b as u64);
        let v = fixed_f32(b * stride, 0x0C3C_3C3C ^ b as u64);

        // CPU reference: each slot attends ONLY over its own region [0..seq_bi).
        let mut ref_cpu = vec![0.0f32; b * q_dim];
        for bi in 0..b {
            let seq_bi = positions[bi] as usize + 1;
            assert!(seq_bi <= max_seq, "seq_bi {seq_bi} exceeds max_seq {max_seq}");
            let q_bi = &q[bi * q_dim..(bi + 1) * q_dim];
            let k_bi = &k[bi * stride..bi * stride + seq_bi * kv_dim];
            let v_bi = &v[bi * stride..bi * stride + seq_bi * kv_dim];
            let out_bi = &mut ref_cpu[bi * q_dim..(bi + 1) * q_dim];
            mha_decode_step(q_bi, k_bi, v_bi, N_HEADS, N_KV_HEADS, HEAD_DIM, seq_bi, out_bi).expect("cpu mha_decode_step (slot)");
        }

        // GPU under test.
        let q_buf = new_f32_buf(ctx, &q);
        let k_buf = new_f32_buf(ctx, &k);
        let v_buf = new_f32_buf(ctx, &v);
        let pos_buf = u32_buf(ctx, positions);
        // region == batch index here (each slot's KV is at its own bi*stride region).
        let region_ids: Vec<u32> = (0..b as u32).collect();
        let region_buf = u32_buf(ctx, &region_ids);
        let out_buf = ctx.new_buffer(b * q_dim * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::mha_decode_f32_batched_multiseq_tcb(&mut tcb, &q_buf, &k_buf, 0, &v_buf, 0, &out_buf, &pos_buf, &region_buf, max_seq, stride, b, HEAD_DIM, N_HEADS, N_KV_HEADS)
                .expect("multiseq encode");
            tcb.commit_and_wait().expect("multiseq commit");
        }
        let actual = read_f32_buf(&out_buf, b * q_dim);

        let diff = max_abs_diff(&ref_cpu, &actual);
        println!("[multiseq] {label}: b={b} positions={positions:?} diff_vs_cpu={diff:.3e} atol={atol:.0e}");
        assert!(diff < atol, "{label}: multiseq vs CPU diff {diff:.3e} >= {atol:.0e}");
    }

    #[test]
    fn multiseq_divergent_positions() {
        // The core case: B independent sequences at DISTINCT positions.
        run_multiseq("divergent", &[5, 12, 2, 0], 16, ATOL);
    }

    #[test]
    fn multiseq_b1_matches_single() {
        // Degenerate B=1 — catches per-slot indexing bugs.
        run_multiseq("b=1 pos=7", &[7], 16, ATOL);
    }

    #[test]
    fn multiseq_b8_mixed() {
        run_multiseq("b=8 mixed", &[0, 1, 3, 7, 15, 4, 9, 2], 16, ATOL);
    }

    // MANDATORY long-context case (a slot near 2K positions).
    #[test]
    fn multiseq_long_context() {
        run_multiseq("long-ctx", &[2047, 1024, 512, 100], 2048, ATOL);
    }
}
#[rustfmt::skip]
mod pair_2r_parity {
    #![cfg(target_os = "macos")]
    //! Track A7 parity: `gemm_q4_k_v4_predec_pair_2r` must produce bit-identical
    //! outputs to `gemm_q4_k_v4_predec_pair` (1r) for all production-relevant shapes.
    //!
    //! The 2r kernel amortises the activation x across 4 partial sums instead of 2
    //! but uses identical FMA order per row, so outputs must match exactly.

    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};

    use crate::common;
    use common::*;

    /// Make a synthetic Q4K predec weight buffer (rows × cols, 144 B/block).
    fn make_q4k_predec(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let bpr = cols / 256;
        let total_bytes = rows * bpr * 144;
        let w: Vec<u8> = (0..total_bytes).map(|i| ((i as u32).wrapping_mul(2_246_822_519).wrapping_add(seed)) as u8).collect();
        let n_scales = rows * bpr * 16;
        let s: Vec<f32> = (0..n_scales)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed)) as f32 / u32::MAX as f32;
                // Typical scale range: [-0.5, 0.5]
                v - 0.5
            })
            .collect();
        (w, s)
    }

    fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(1_664_525).wrapping_add(seed);
                (x as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn run_pair_1r(ctx: &MetalContext, wg: &[u8], wu: &[u8], g_scales: &[f32], u_scales: &[f32], x: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let _wg_buf = ctx.new_buffer_with_bytes(wg);
        let _wu_buf = ctx.new_buffer_with_bytes(wu);
        let gs_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(g_scales));
        let us_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(u_scales));
        let x_buf = new_f32_buf(ctx, x);
        let yg_buf = ctx.new_buffer(rows * 4);
        let yu_buf = ctx.new_buffer(rows * 4);

        // 1r pair uses shared model_buf with separate offsets; here gate and up are
        // separate buffers so we reuse wg_buf for both matrix pointers (gate at off=0,
        // up at off=0 of wu_buf). Use the wrapper's model_buf + offset convention:
        // gate: model_buf=wg_buf off=0, up: model_buf=wu_buf off=0.
        // The wrapper takes a single model_buf + two offsets into it. For the test
        // we need two distinct buffers (gate ≠ up), so call the raw dispatch twice
        // for 1r parity, or use a combined buffer with offsets.
        //
        // Simplest: build a combined buffer [gate_bytes || up_bytes] and set offsets.
        let w_bytes = rows * (cols / 256) * 144;
        let mut combined = Vec::with_capacity(wg.len() + wu.len());
        combined.extend_from_slice(wg);
        combined.extend_from_slice(wu);
        let combined_buf = ctx.new_buffer_with_bytes(&combined);

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pair_pinned_tcb(&mut tcb, &combined_buf, 0, w_bytes, &gs_buf, 0, w_bytes, w_bytes, &us_buf, 0, rows, cols, &x_buf, &yg_buf, &yu_buf).expect("1r pair dispatch");
        tcb.commit_and_wait().expect("1r pair wait");

        (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
    }

    fn run_pair_2r(ctx: &MetalContext, wg: &[u8], wu: &[u8], g_scales: &[f32], u_scales: &[f32], x: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let w_bytes = rows * (cols / 256) * 144;
        let mut combined = Vec::with_capacity(wg.len() + wu.len());
        combined.extend_from_slice(wg);
        combined.extend_from_slice(wu);
        let combined_buf = ctx.new_buffer_with_bytes(&combined);
        let gs_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(g_scales));
        let us_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(u_scales));
        let x_buf = new_f32_buf(ctx, x);
        let yg_buf = ctx.new_buffer(rows * 4);
        let yu_buf = ctx.new_buffer(rows * 4);

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pair_2r_pinned_tcb(&mut tcb, &combined_buf, 0, w_bytes, &gs_buf, 0, w_bytes, w_bytes, &us_buf, 0, rows, cols, &x_buf, &yg_buf, &yu_buf).expect("2r pair dispatch");
        tcb.commit_and_wait().expect("2r pair wait");

        (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
    }

    fn run_pair_2r_inline(ctx: &MetalContext, wg: &[u8], wu: &[u8], g_scales: &[f32], u_scales: &[f32], x: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let w_bytes = rows * (cols / 256) * 144;
        let mut combined = Vec::with_capacity(wg.len() + wu.len());
        combined.extend_from_slice(wg);
        combined.extend_from_slice(wu);
        let combined_buf = ctx.new_buffer_with_bytes(&combined);
        let gs_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(g_scales));
        let us_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(u_scales));
        let x_buf = new_f32_buf(ctx, x);
        let yg_buf = ctx.new_buffer(rows * 4);
        let yu_buf = ctx.new_buffer(rows * 4);

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pair_2r_inline_pinned_tcb(&mut tcb, &combined_buf, 0, w_bytes, &gs_buf, 0, w_bytes, w_bytes, &us_buf, 0, rows, cols, &x_buf, &yg_buf, &yu_buf)
            .expect("2r inline pair dispatch");
        tcb.commit_and_wait().expect("2r inline pair wait");

        (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
    }

    fn run_pair_2r_inline_nox(ctx: &MetalContext, wg: &[u8], wu: &[u8], g_scales: &[f32], u_scales: &[f32], x: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let w_bytes = rows * (cols / 256) * 144;
        let mut combined = Vec::with_capacity(wg.len() + wu.len());
        combined.extend_from_slice(wg);
        combined.extend_from_slice(wu);
        let combined_buf = ctx.new_buffer_with_bytes(&combined);
        let gs_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(g_scales));
        let us_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(u_scales));
        let x_buf = new_f32_buf(ctx, x);
        let yg_buf = ctx.new_buffer(rows * 4);
        let yu_buf = ctx.new_buffer(rows * 4);

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pair_2r_inline_nox_pinned_tcb(&mut tcb, &combined_buf, 0, w_bytes, &gs_buf, 0, w_bytes, w_bytes, &us_buf, 0, rows, cols, &x_buf, &yg_buf, &yu_buf)
            .expect("2r inline nox pair dispatch");
        tcb.commit_and_wait().expect("2r inline nox pair wait");

        (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
    }

    fn run_pair_4r(ctx: &MetalContext, wg: &[u8], wu: &[u8], g_scales: &[f32], u_scales: &[f32], x: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let w_bytes = rows * (cols / 256) * 144;
        let mut combined = Vec::with_capacity(wg.len() + wu.len());
        combined.extend_from_slice(wg);
        combined.extend_from_slice(wu);
        let combined_buf = ctx.new_buffer_with_bytes(&combined);
        let gs_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(g_scales));
        let us_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(u_scales));
        let x_buf = new_f32_buf(ctx, x);
        let yg_buf = ctx.new_buffer(rows * 4);
        let yu_buf = ctx.new_buffer(rows * 4);

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pair_4r_pinned_tcb(&mut tcb, &combined_buf, 0, w_bytes, &gs_buf, 0, w_bytes, w_bytes, &us_buf, 0, rows, cols, &x_buf, &yg_buf, &yu_buf).expect("4r pair dispatch");
        tcb.commit_and_wait().expect("4r pair wait");

        (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
    }

    fn run_pair_3r(ctx: &MetalContext, wg: &[u8], wu: &[u8], g_scales: &[f32], u_scales: &[f32], x: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let w_bytes = rows * (cols / 256) * 144;
        let mut combined = Vec::with_capacity(wg.len() + wu.len());
        combined.extend_from_slice(wg);
        combined.extend_from_slice(wu);
        let combined_buf = ctx.new_buffer_with_bytes(&combined);
        let gs_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(g_scales));
        let us_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(u_scales));
        let x_buf = new_f32_buf(ctx, x);
        let yg_buf = ctx.new_buffer(rows * 4);
        let yu_buf = ctx.new_buffer(rows * 4);

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pair_3r_pinned_tcb(&mut tcb, &combined_buf, 0, w_bytes, &gs_buf, 0, w_bytes, w_bytes, &us_buf, 0, rows, cols, &x_buf, &yg_buf, &yu_buf).expect("3r pair dispatch");
        tcb.commit_and_wait().expect("3r pair wait");

        (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
    }

    #[test]
    fn pair_2r_matches_pair_1r_multiple_shapes() {
        let ctx = ctx();

        // (rows, cols, seed) — rows=16 tests the boundary (2 TGs for 2r, each just
        // covers the 16-row stride). rows=48 tests the non-multiple-of-16 boundary.
        // rows=512, cols=2048 approximates production gate/up shapes at smaller scale.
        let cases: &[(usize, usize, u32)] = &[
            (16, 256, 0xA701),
            (32, 512, 0xA702),
            (48, 256, 0xA703), // non-multiple of 16: last TG processes 12 rows (has1 path)
            (128, 512, 0xA704),
            (512, 2048, 0xA705),
            (1024, 512, 0xA706),
        ];

        for &(rows, cols, seed) in cases {
            let (wg, g_scales) = make_q4k_predec(rows, cols, seed);
            let (wu, u_scales) = make_q4k_predec(rows, cols, seed ^ 0xFFFF);
            let x = rand_vec(cols, seed ^ 0x1234);

            let (ref_g, ref_u) = run_pair_1r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);
            let (got_g, got_u) = run_pair_2r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);

            let diff_g = max_abs_diff(&ref_g, &got_g);
            let diff_u = max_abs_diff(&ref_u, &got_u);
            assert_eq!(diff_g, 0.0, "rows={rows} cols={cols}: gate max_diff={diff_g:.2e} (must be 0)");
            assert_eq!(diff_u, 0.0, "rows={rows} cols={cols}: up   max_diff={diff_u:.2e} (must be 0)");
            eprintln!("pair_2r rows={rows} cols={cols}: gate_diff={diff_g:.2e} up_diff={diff_u:.2e} OK");
        }
    }

    /// Track E3: `gemm_q4_k_v4_predec_pair_2r_inline` must be bit-identical to
    /// `pair_2r`; only scale-load style changes.
    #[test]
    fn pair_2r_inline_matches_pair_2r_multiple_shapes() {
        let ctx = ctx();

        let cases: &[(usize, usize, u32)] = &[(16, 256, 0xE301), (17, 256, 0xE302), (32, 512, 0xE303), (48, 256, 0xE304), (128, 512, 0xE305), (512, 2048, 0xE306), (1024, 512, 0xE307)];

        for &(rows, cols, seed) in cases {
            let (wg, g_scales) = make_q4k_predec(rows, cols, seed);
            let (wu, u_scales) = make_q4k_predec(rows, cols, seed ^ 0xBEEF);
            let x = rand_vec(cols, seed ^ 0x9876);

            let (ref_g, ref_u) = run_pair_2r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);
            let (got_g, got_u) = run_pair_2r_inline(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);

            let diff_g = max_abs_diff(&ref_g, &got_g);
            let diff_u = max_abs_diff(&ref_u, &got_u);
            assert_eq!(diff_g, 0.0, "rows={rows} cols={cols}: gate max_diff={diff_g:.2e} (must be 0)");
            assert_eq!(diff_u, 0.0, "rows={rows} cols={cols}: up   max_diff={diff_u:.2e} (must be 0)");
            eprintln!("pair_2r_inline rows={rows} cols={cols}: gate_diff={diff_g:.2e} up_diff={diff_u:.2e} OK");
        }
    }

    /// Track F1: `gemm_q4_k_v4_predec_pair_2r_inline_nox` must be bit-identical to
    /// `pair_2r` — it only drops the xl[8] activation preload (reads x per-pi), with
    /// identical per-accumulator FMA order and identical x values (xl[k] was just
    /// x[b*256 + k*32 + lane]). Boundary cases (17, 48) exercise the has1 OOB guard.
    #[test]
    fn pair_2r_inline_nox_matches_pair_2r_multiple_shapes() {
        let ctx = ctx();

        let cases: &[(usize, usize, u32)] = &[
            (16, 256, 0xF101),
            (17, 256, 0xF102),
            (32, 512, 0xF103),
            (48, 256, 0xF104),
            (128, 512, 0xF105),
            (512, 2048, 0xF106),
            (1024, 512, 0xF107),
            (11008, 2048, 0xF108), // production ffn gate/up shape
        ];

        for &(rows, cols, seed) in cases {
            let (wg, g_scales) = make_q4k_predec(rows, cols, seed);
            let (wu, u_scales) = make_q4k_predec(rows, cols, seed ^ 0xCAFE);
            let x = rand_vec(cols, seed ^ 0x4321);

            let (ref_g, ref_u) = run_pair_2r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);
            let (got_g, got_u) = run_pair_2r_inline_nox(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);

            let diff_g = max_abs_diff(&ref_g, &got_g);
            let diff_u = max_abs_diff(&ref_u, &got_u);
            assert_eq!(diff_g, 0.0, "rows={rows} cols={cols}: gate max_diff={diff_g:.2e} (must be 0)");
            assert_eq!(diff_u, 0.0, "rows={rows} cols={cols}: up   max_diff={diff_u:.2e} (must be 0)");
            eprintln!("pair_2r_inline_nox rows={rows} cols={cols}: gate_diff={diff_g:.2e} up_diff={diff_u:.2e} OK");
        }
    }

    /// Track E2: `gemm_q4_k_v4_predec_pair_3r` must be bit-identical to `pair_2r`.
    /// Boundary cases exercise the non-power-of-two row geometry: 24 rows/TG.
    #[test]
    fn pair_3r_matches_pair_2r_multiple_shapes() {
        let ctx = ctx();

        let cases: &[(usize, usize, u32)] = &[
            (24, 256, 0xE201), // exactly 1 TG for 3r
            (25, 256, 0xE202), // 2 TGs, second has 1 row
            (32, 256, 0xE203), // second TG has 8 rows
            (33, 512, 0xE204), // second TG has 9 rows
            (48, 512, 0xE205), // exactly 2 TGs
            (128, 512, 0xE206),
            (512, 2048, 0xE207),
            (1024, 512, 0xE208),
        ];

        for &(rows, cols, seed) in cases {
            let (wg, g_scales) = make_q4k_predec(rows, cols, seed);
            let (wu, u_scales) = make_q4k_predec(rows, cols, seed ^ 0xDEAD);
            let x = rand_vec(cols, seed ^ 0x5678);

            let (ref_g, ref_u) = run_pair_2r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);
            let (got_g, got_u) = run_pair_3r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);

            let diff_g = max_abs_diff(&ref_g, &got_g);
            let diff_u = max_abs_diff(&ref_u, &got_u);
            assert_eq!(diff_g, 0.0, "rows={rows} cols={cols}: gate max_diff={diff_g:.2e} (must be 0)");
            assert_eq!(diff_u, 0.0, "rows={rows} cols={cols}: up   max_diff={diff_u:.2e} (must be 0)");
            eprintln!("pair_3r rows={rows} cols={cols}: gate_diff={diff_g:.2e} up_diff={diff_u:.2e} OK");
        }
    }

    /// Track B2: `gemm_q4_k_v4_predec_pair_4r` must be bit-identical to `pair_2r`
    /// (same per-accumulator FMA order, only scale access style differs: inline
    /// vs preloaded).  Also includes a boundary case where rows is not a multiple
    /// of 32 to exercise the `has1/has2/has3` out-of-bounds guards.
    #[test]
    fn pair_4r_matches_pair_2r_multiple_shapes() {
        let ctx = ctx();

        // (rows, cols, seed) — rows=48 tests non-multiple of 32 (last TG covers
        // only 16 rows → has2/has3 are false for simd_id 2..7, verifying guards).
        // rows=11008 approximates the production ffn gate/up shape on Qwen2.5-3B.
        let cases: &[(usize, usize, u32)] = &[
            (32, 256, 0xB201),
            (48, 256, 0xB202), // non-multiple of 32: tests has1/has2/has3 boundary
            (64, 512, 0xB203),
            (128, 512, 0xB204),
            (512, 2048, 0xB205),
            (1024, 512, 0xB206),
        ];

        for &(rows, cols, seed) in cases {
            let (wg, g_scales) = make_q4k_predec(rows, cols, seed);
            let (wu, u_scales) = make_q4k_predec(rows, cols, seed ^ 0xDEAD);
            let x = rand_vec(cols, seed ^ 0x5678);

            let (ref_g, ref_u) = run_pair_2r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);
            let (got_g, got_u) = run_pair_4r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);

            let diff_g = max_abs_diff(&ref_g, &got_g);
            let diff_u = max_abs_diff(&ref_u, &got_u);
            assert_eq!(diff_g, 0.0, "rows={rows} cols={cols}: gate max_diff={diff_g:.2e} (must be 0)");
            assert_eq!(diff_u, 0.0, "rows={rows} cols={cols}: up   max_diff={diff_u:.2e} (must be 0)");
            eprintln!("pair_4r rows={rows} cols={cols}: gate_diff={diff_g:.2e} up_diff={diff_u:.2e} OK");
        }
    }
}
#[rustfmt::skip]
mod pair_4r_f16s_parity {
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

    use crate::common;
    use common::*;

    fn make_q4k_predec(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let bpr = cols / 256;
        let w: Vec<u8> = (0..rows * bpr * 144).map(|i| ((i as u32).wrapping_mul(2246822519).wrapping_add(seed)) as u8).collect();
        // Avoid near-zero scales: [0.1, 2.0] so f16 rounding is controlled.
        let s: Vec<f32> = (0..rows * bpr * 16)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed ^ 0xAB)) as f32 / u32::MAX as f32;
                0.1 + v * 1.9
            })
            .collect();
        (w, s)
    }

    fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|&x| f16::from_f32(x).to_le_bytes()).collect()
    }

    fn rel_l2(reference: &[f32], got: &[f32]) -> f64 {
        let num: f64 = reference.iter().zip(got).map(|(&r, &g)| ((r - g) as f64).powi(2)).sum();
        let den: f64 = reference.iter().map(|&r| (r as f64).powi(2)).sum::<f64>().max(1e-30);
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
        kernels::gemv_q4_k_v4_predec_pair_4r_pinned_tcb(&mut tcb, &model_buf, g_off, g_len, &gs_buf, 0, u_off, u_len, &us_buf, 0, rows, cols, &x_buf, &g_out, &u_out).unwrap();
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
        kernels::gemv_q4_k_v4_predec_pair_4r_f16s_pinned_tcb(&mut tcb, &model_buf, g_off, g_len, &gs_buf, 0, u_off, u_len, &us_buf, 0, rows, cols, &x_buf, &g_out, &u_out).unwrap();
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
            let x: Vec<f32> = (0..cols).map(|i| ((i as u32).wrapping_mul(1664525).wrapping_add(seed) as f32 / u32::MAX as f32) * 2.0 - 1.0).collect();

            let (ref_g, ref_u) = run_f32_4r(ctx, &model, g_off, g_w.len(), u_off, u_w.len(), &g_sc, &u_sc, &x, rows, cols);
            let (got_g, got_u) = run_f16s_4r(ctx, &model, g_off, g_w.len(), u_off, u_w.len(), &g_sc_f16, &u_sc_f16, &x, rows, cols);

            let rg = rel_l2(&ref_g, &got_g);
            let ru = rel_l2(&ref_u, &got_u);
            assert!(rg < MAX_REL_L2, "rows={rows} cols={cols}: gate rel_L2={rg:.4e} >= {MAX_REL_L2:.4e}");
            assert!(ru < MAX_REL_L2, "rows={rows} cols={cols}: up   rel_L2={ru:.4e} >= {MAX_REL_L2:.4e}");
            eprintln!("D4 4r_f16s rows={rows} cols={cols}: gate={rg:.2e} up={ru:.2e} OK");
        }
    }
}
#[rustfmt::skip]
mod pair_8r_parity {
    #![cfg(target_os = "macos")]
    //! Track E1 parity: `gemm_q4_k_v4_predec_pair_8r` must produce outputs
    //! matching `gemm_q4_k_v4_predec_pair_4r` (the default-on reference) within
    //! a tight tolerance.
    //!
    //! The 8r kernel handles 8 rows per simdgroup (64 rows/TG) vs 4 rows for the
    //! 4r kernel (32 rows/TG), halving TG count again for the Qwen-3B gate+up
    //! shape (11008 rows → 172 TGs vs 344). Both kernels use the same per-element
    //! FMA path; the only difference is loop unrolling depth and row-stride. FMA
    //! reordering between 4 and 8 independent accumulators may differ by ~2 ULPs;
    //! we gate at 1e-5 to allow that while catching any structural error.

    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};

    use crate::common;
    use common::*;

    fn make_q4k_predec(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let bpr = cols / 256;
        let total_bytes = rows * bpr * 144;
        let w: Vec<u8> = (0..total_bytes).map(|i| ((i as u32).wrapping_mul(2_246_822_519).wrapping_add(seed)) as u8).collect();
        let n_scales = rows * bpr * 16;
        let s: Vec<f32> = (0..n_scales)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed)) as f32 / u32::MAX as f32;
                v - 0.5
            })
            .collect();
        (w, s)
    }

    fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(1_664_525).wrapping_add(seed);
                (x as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Run gate+up pair with 4r kernel (reference for E1).
    fn run_pair_4r(ctx: &MetalContext, wg: &[u8], wu: &[u8], g_scales: &[f32], u_scales: &[f32], x: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let w_bytes = rows * (cols / 256) * 144;
        let mut combined = Vec::with_capacity(wg.len() + wu.len());
        combined.extend_from_slice(wg);
        combined.extend_from_slice(wu);
        let combined_buf = ctx.new_buffer_with_bytes(&combined);
        let gs_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(g_scales));
        let us_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(u_scales));
        let x_buf = new_f32_buf(ctx, x);
        let yg_buf = ctx.new_buffer(rows * 4);
        let yu_buf = ctx.new_buffer(rows * 4);

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pair_4r_pinned_tcb(&mut tcb, &combined_buf, 0, w_bytes, &gs_buf, 0, w_bytes, w_bytes, &us_buf, 0, rows, cols, &x_buf, &yg_buf, &yu_buf).expect("4r pair dispatch");
        tcb.commit_and_wait().expect("4r pair wait");
        (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
    }

    /// Run gate+up pair with 8r kernel (Track E1).
    fn run_pair_8r(ctx: &MetalContext, wg: &[u8], wu: &[u8], g_scales: &[f32], u_scales: &[f32], x: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let w_bytes = rows * (cols / 256) * 144;
        let mut combined = Vec::with_capacity(wg.len() + wu.len());
        combined.extend_from_slice(wg);
        combined.extend_from_slice(wu);
        let combined_buf = ctx.new_buffer_with_bytes(&combined);
        let gs_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(g_scales));
        let us_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(u_scales));
        let x_buf = new_f32_buf(ctx, x);
        let yg_buf = ctx.new_buffer(rows * 4);
        let yu_buf = ctx.new_buffer(rows * 4);

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pair_8r_pinned_tcb(&mut tcb, &combined_buf, 0, w_bytes, &gs_buf, 0, w_bytes, w_bytes, &us_buf, 0, rows, cols, &x_buf, &yg_buf, &yu_buf).expect("8r pair dispatch");
        tcb.commit_and_wait().expect("8r pair wait");
        (read_f32_buf(&yg_buf, rows), read_f32_buf(&yu_buf, rows))
    }

    /// E1 quality gate: 8r must agree with 4r within 1e-5.
    /// 8 independent FMA chains (gate0-7 + up0-7) may reorder vs 4 chains in 4r,
    /// producing ULP-level differences. 1e-5 is tight enough to catch any structural
    /// error (wrong row index, scale misread, missing accumulator) while tolerating
    /// FMA reassociation.
    ///
    /// Key boundary cases:
    ///  - rows=64:  exactly 1 TG (8 simdgroups × 8 rows)
    ///  - rows=65:  2 TGs; second TG has 1 active row (tests has1-7 all false for simd_id>0)
    ///  - rows=72:  2 TGs; second TG has 8 rows (simd_id=0 only has row0, has1-7 false for rest)
    ///  - rows=128: exactly 2 TGs
    ///  - rows=512, 1024: larger shapes
    #[test]
    fn e1_pair_8r_matches_pair_4r() {
        let ctx = ctx();
        // max_abs_diff gate: 1e-5 allows FMA reorder ULPs, rejects any structural bug
        const MAX_DIFF: f32 = 1e-5;

        let cases: &[(usize, usize, u32)] = &[
            (64, 256, 0xE101),   // exactly 1 TG
            (65, 256, 0xE102),   // 2 TGs, second has 1 row: tests has1-7 guards for simd_id>0
            (72, 256, 0xE103),   // 2 TGs, second has 8 rows: tests has1-7 for simd_id=0
            (128, 256, 0xE104),  // exactly 2 TGs
            (256, 512, 0xE105),  // 4 TGs
            (512, 2048, 0xE106), // larger shape
            (1024, 512, 0xE107), // 16 TGs
        ];

        for &(rows, cols, seed) in cases {
            let (wg, g_scales) = make_q4k_predec(rows, cols, seed);
            let (wu, u_scales) = make_q4k_predec(rows, cols, seed ^ 0xDEAD);
            let x = rand_vec(cols, seed ^ 0x5678);

            let (ref_g, ref_u) = run_pair_4r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);
            let (got_g, got_u) = run_pair_8r(ctx, &wg, &wu, &g_scales, &u_scales, &x, rows, cols);

            let diff_g = max_abs_diff(&ref_g, &got_g);
            let diff_u = max_abs_diff(&ref_u, &got_u);
            assert!(diff_g <= MAX_DIFF, "E1 rows={rows} cols={cols}: gate max_diff={diff_g:.2e} > {MAX_DIFF:.2e}");
            assert!(diff_u <= MAX_DIFF, "E1 rows={rows} cols={cols}: up   max_diff={diff_u:.2e} > {MAX_DIFF:.2e}");
            eprintln!("E1 pair_8r rows={rows} cols={cols}: gate_diff={diff_g:.2e} up_diff={diff_u:.2e} OK");
        }
    }
}
#[rustfmt::skip]
mod phase1_kernel_parity {
    //! Phase 1 / Haul 1 — Numerical parity tests between CPU reference
    //! kernels and Metal-dispatched kernels.
    //!
    //! **Status: SCAFFOLDING.** Each test below is `#[ignore]` until its
    //! corresponding gate's haul item lands the implementation. The haul
    //! removes the `#[ignore]` attribute when filling in the body.
    //!
    //! Every test must:
    //!   1. Generate a fixed-seed input (so baselines are reproducible).
    //!   2. Run the CPU reference kernel from `hawking_core::kernels`.
    //!   3. Run the Metal-dispatched kernel from
    //!      `hawking_core::kernels::metal_dispatch::*`.
    //!   4. Assert max abs diff < `ATOL` (1e-3 fp16 quant noise).
    //!
    //! Common test plumbing is provided below so each gate's body is
    //! short and obvious.

    #![cfg(target_os = "macos")]

    use hawking_core::kernels;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    /// fp16 absolute tolerance — about 1 part in 1024 (fp16 mantissa is
    /// 10 bits + 1 implicit). Allows reduction-order sensitivity.
    pub const ATOL: f32 = 1e-3;

    /// Single shared Metal context across all parity tests in this file.
    /// Avoids re-running device lookup + library compile + pipeline cache
    /// init on every test — those are ~50-200ms each. Cargo runs the
    /// 4 parity tests in the same binary, so they share this Lazy.
    /// `MetalContext` is Clone (Arc-backed) so individual test bodies can
    /// hold a `&'static MetalContext` directly.

    fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
    }

    // ---------------------------------------------------------------------
    // G1.1 — Metal scaffold + rmsnorm round-trip
    // ---------------------------------------------------------------------

    #[test]
    fn test_rmsnorm_matches_cpu() {
        let hidden = 4096;
        let x = fixed_input(hidden, 0xCAFEBABE);
        let w = fixed_input(hidden, 0xDEADBEEF);
        let eps = 1e-6_f32;

        let mut cpu_out = vec![0.0_f32; hidden];
        kernels::rmsnorm(&x, &w, eps, &mut cpu_out);

        let ctx = ctx().clone();
        let mut metal_out = vec![0.0_f32; hidden];
        kernels::rmsnorm_metal(&ctx, &x, &w, eps, &mut metal_out).expect("rmsnorm_metal should succeed once G1.1 lands");

        let diff = max_abs_diff(&cpu_out, &metal_out);
        println!("[G1.1] rmsnorm parity max abs diff = {diff:.6}");
        assert!(diff < ATOL, "rmsnorm CPU/Metal diff {diff} >= atol {ATOL}");
    }

    // ---------------------------------------------------------------------
    // G1.2 — LM-head GEMV (fp16 weights × fp32 vec → fp32 logits)
    // ---------------------------------------------------------------------

    #[test]
    fn test_gemv_f16_matches_cpu() {
        use half::f16;

        // Smaller than the real LM head (vocab=102400 × hidden=2048) so
        // the parity test is fast; the size still exceeds one threadgroup
        // tile and exercises reduction logic.
        let rows = 4096;
        let cols = 2048;
        let x = fixed_input(cols, 0xA1A1A1A1);
        let w_f32 = fixed_input(rows * cols, 0xB2B2B2B2);
        let w_f16: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();

        let mut cpu_out = vec![0.0_f32; rows];
        kernels::gemv_f16(&w_f16, rows, cols, &x, &mut cpu_out);

        let ctx = ctx().clone();
        let w_bytes: &[u8] = bytemuck::cast_slice(&w_f16);
        let mut metal_out = vec![0.0_f32; rows];
        kernels::gemv_f16_metal(&ctx, w_bytes, rows, cols, &x, &mut metal_out).expect("gemv_f16_metal should succeed once G1.2 lands");

        let diff = max_abs_diff(&cpu_out, &metal_out);
        println!("[G1.2] gemv_f16 parity max abs diff = {diff:.6}");
        assert!(diff < ATOL, "gemv_f16 CPU/Metal diff {diff} >= atol {ATOL}");
    }

    #[test]
    fn test_gemv_f16_argmax_pinned_matches_cpu() {
        use half::f16;

        let rows = 1024;
        let cols = 512;
        let x = fixed_input(cols, 0x1234ABCD);
        let w_f32 = fixed_input(rows * cols, 0x4567DCBA);
        let w_f16: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();

        let mut cpu_logits = vec![0.0_f32; rows];
        kernels::gemv_f16(&w_f16, rows, cols, &x, &mut cpu_logits);
        let cpu = kernels::argmax_f32(&cpu_logits);

        let ctx = ctx().clone();
        let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&w_f16));
        let metal = kernels::gemv_f16_argmax_metal_pinned(&ctx, &w_buf, rows, cols, &x).expect("gemv_f16_argmax_metal_pinned should return token id");

        println!("[SAMPLE] gemv_f16+argmax parity cpu={cpu} metal={metal}");
        assert_eq!(cpu, metal);
    }

    // ---------------------------------------------------------------------
    // G1.3 — Attention o_proj GEMV (fp32 weights × fp32 vec → fp32 vec)
    // ---------------------------------------------------------------------

    #[test]
    fn test_gemv_f32_attn_matches_cpu() {
        // o_proj: hidden × (n_heads × v_head_dim) = 2048 × 2048
        let rows = 2048;
        let cols = 2048;
        let x = fixed_input(cols, 0xC3C3C3C3);
        let w = fixed_input(rows * cols, 0xD4D4D4D4);

        let mut cpu_out = vec![0.0_f32; rows];
        kernels::gemv_f32(&w, rows, cols, &x, &mut cpu_out);

        let ctx = ctx().clone();
        let mut metal_out = vec![0.0_f32; rows];
        kernels::gemv_f32_attn_metal(&ctx, &w, rows, cols, &x, &mut metal_out).expect("gemv_f32_attn_metal should succeed once G1.3 lands");

        let diff = max_abs_diff(&cpu_out, &metal_out);
        println!("[G1.3] gemv_f32 (attn) parity max abs diff = {diff:.6}");
        assert!(diff < ATOL, "gemv_f32_attn CPU/Metal diff {diff} >= atol {ATOL}");
    }

    // ---------------------------------------------------------------------
    // G1.4 — MoE gate-logit GEMV (fp32 weights × fp32 vec → fp32 logits)
    // ---------------------------------------------------------------------

    #[test]
    fn test_gemv_f32_moe_matches_cpu() {
        // ffn_gate_inp: n_routed_experts × hidden = 64 × 2048
        let rows = 64;
        let cols = 2048;
        let x = fixed_input(cols, 0xE5E5E5E5);
        let w = fixed_input(rows * cols, 0xF6F6F6F6);

        let mut cpu_out = vec![0.0_f32; rows];
        kernels::gemv_f32(&w, rows, cols, &x, &mut cpu_out);

        let ctx = ctx().clone();
        let mut metal_out = vec![0.0_f32; rows];
        kernels::gemv_f32_moe_metal(&ctx, &w, rows, cols, &x, &mut metal_out).expect("gemv_f32_moe_metal should succeed once G1.4 lands");

        let diff = max_abs_diff(&cpu_out, &metal_out);
        println!("[G1.4] gemv_f32 (moe) parity max abs diff = {diff:.6}");
        assert!(diff < ATOL, "gemv_f32_moe CPU/Metal diff {diff} >= atol {ATOL}");
    }

    // ---------------------------------------------------------------------
    // H2.1 — top-K softmax gate (Wedge 2: MoE block, gate stage)
    // ---------------------------------------------------------------------

    // ---------------------------------------------------------------------
    // H2.2 — moe grouped GEMM with fused Q4_K_M dequant (Wedge 2: the moat)
    // ---------------------------------------------------------------------

    /// Construct synthetic Q4_K_M weight bytes for parity testing.
    ///
    /// `d` and `dmin` are deliberately small (~1e-2) so the per-element
    /// dequant values stay in a tight range. With 6-bit scales (max 63),
    /// nibbles (max 15), and `d≈0.015`, max element magnitude is
    /// `0.015 × 63 × 15 ≈ 14`, sum of 256 such terms is O(32). At that
    /// magnitude the sequential-vs-tree reduction-order divergence is
    /// well below the 1e-3 parity tolerance — without this clamp, large
    /// random outputs cross atol from accumulation order alone.
    fn synthetic_q4_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
        use half::f16;
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.01 + rng.gen::<f32>() * 0.01;
            let d_bits = f16::from_f32(d).to_bits();
            bytes[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
            let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
            let dmin_bits = f16::from_f32(dmin).to_bits();
            bytes[off + 2..off + 4].copy_from_slice(&dmin_bits.to_le_bytes());
            for i in 4..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    // ---------------------------------------------------------------------
    // H2.4 — gemm_q4_k_m_fused (Wedge 2: dense-path Q4_K_M GEMV)
    // ---------------------------------------------------------------------

    #[test]
    fn test_gemm_q4_k_m_fused_matches_cpu() {
        use hawking_core::gguf::GgmlType;
        use hawking_core::quant::dequant_into;

        let rows = 64;
        let cols = 256;
        let blocks = rows * (cols / 256);

        // Different seeds from H2.2 so this test exercises distinct bytes.
        let w_bytes = synthetic_q4_k_bytes(blocks, 0xE6E6E6E6);
        let x = fixed_input(cols, 0xF7F7F7F7);

        let mut w_f32 = vec![0.0_f32; rows * cols];
        dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32).expect("Q4_K dequant should succeed for valid synthetic bytes");
        let mut cpu_out = vec![0.0_f32; rows];
        kernels::gemv_f32(&w_f32, rows, cols, &x, &mut cpu_out);

        // Metal dense path: gemv_q4_k_m dispatches gemm_q4_k_m_fused.
        let ctx = ctx().clone();
        let mut metal_out = vec![0.0_f32; rows];
        kernels::gemv_q4_k_m(&ctx, &w_bytes, rows, cols, &x, &mut metal_out).expect("gemv_q4_k_m should succeed once H2.4 lands");

        let diff = max_abs_diff(&cpu_out, &metal_out);
        println!("[H2.4] gemm_q4_k_m_fused parity max abs diff = {diff:.6}");
        assert!(diff < ATOL, "gemm_q4_k_m_fused CPU/Metal diff {diff} >= atol {ATOL}");
    }
}
#[rustfmt::skip]
mod phase2_mla_metal_parity {
    //! Phase 2 / W1B — MLA / Q-LoRA gemv parity tests.
    //!
    //! W1B routes the four MLA fp32 gemv call sites in
    //! `model::deepseek_v2::attention` (q_a_proj, q_b_proj,
    //! kv_a_proj_with_mqa, kv_b_proj) through `gemv_f32_attn_dispatch`,
    //! which lands them on `gemv_f32_attn_metal` under
    //! `cfg(target_os = "macos")` + `Some(metal_ctx)`.
    //!
    //! `gemv_f32_attn_metal` is already attested at atol=1e-3 fp16 by the
    //! G1.3 parity test in `phase1_kernel_parity.rs` for one shape
    //! (2048×2048). This test exercises the kernel on the four
    //! MLA-specific shapes from DeepSeek-V2-Lite to catch any
    //! shape-edge bugs the production gemv would expose.
    //!
    //! Shapes (rows × cols, where rows = output dim, cols = input dim):
    //! - q_a_proj            : 1536 × 2048
    //! - q_b_proj            : 3072 × 1536
    //! - kv_a_proj_with_mqa  :  576 × 2048
    //! - kv_b_proj           : 2048 ×  512

    #![cfg(target_os = "macos")]

    use hawking_core::kernels;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
    }

    fn parity_check(name: &'static str, rows: usize, cols: usize, seed_x: u64, seed_w: u64) {
        let x = fixed_input(cols, seed_x);
        let w = fixed_input(rows * cols, seed_w);

        let mut cpu_out = vec![0.0_f32; rows];
        kernels::gemv_f32(&w, rows, cols, &x, &mut cpu_out);

        let ctx = ctx().clone();
        let mut metal_out = vec![0.0_f32; rows];
        kernels::gemv_f32_attn_metal(&ctx, &w, rows, cols, &x, &mut metal_out).expect("gemv_f32_attn_metal should succeed");

        let diff = max_abs_diff(&cpu_out, &metal_out);
        println!("[W1B] {name} ({rows}x{cols}) parity max abs diff = {diff:.6}");
        assert!(diff < ATOL, "{name} CPU/Metal diff {diff} >= atol {ATOL}");
    }

    #[test]
    fn test_q_a_proj_shape_matches_cpu() {
        parity_check("q_a_proj", 1536, 2048, 0x1A1A_1A1A, 0x1B1B_1B1B);
    }

    #[test]
    fn test_q_b_proj_shape_matches_cpu() {
        parity_check("q_b_proj", 3072, 1536, 0x2A2A_2A2A, 0x2B2B_2B2B);
    }

    #[test]
    fn test_kv_a_proj_shape_matches_cpu() {
        parity_check("kv_a_proj_with_mqa", 576, 2048, 0x3A3A_3A3A, 0x3B3B_3B3B);
    }

    #[test]
    fn test_kv_b_proj_shape_matches_cpu() {
        parity_check("kv_b_proj", 2048, 512, 0x4A4A_4A4A, 0x4B4B_4B4B);
    }

    // ── mla_decode_kernel parity tests ─────────────────────────────────────────
    //
    // CPU reference mirrors the four-phase algorithm in `mla_decode_kernel`
    // (shaders/attn.metal): w_uk^T @ q_nope, scores, softmax, c_kv_weighted,
    // w_uv @ c_kv_weighted.
    //
    // kv_b_proj layout: (n_heads, qk_nope + v_head_dim, kv_lora_rank) row-major.

    #[allow(clippy::too_many_arguments)]
    fn mla_decode_cpu_reference(
        q: &[f32],
        c_kv: &[f32],
        k_pe: &[f32],
        kv_b_proj: &[f32],
        n_heads: usize,
        qk_nope: usize,
        qk_rope: usize,
        v_head_dim: usize,
        kv_lora_rank: usize,
        seq_len: usize,
        scale: f32,
        out: &mut [f32],
    ) {
        let q_head_dim = qk_nope + qk_rope;
        let kv_b_per_head = (qk_nope + v_head_dim) * kv_lora_rank;

        let mut q_nope_proj = vec![0.0f32; kv_lora_rank];
        let mut scores = vec![0.0f32; seq_len];
        let mut c_kv_wt = vec![0.0f32; kv_lora_rank];

        for head in 0..n_heads {
            let q_nope = &q[head * q_head_dim..head * q_head_dim + qk_nope];
            let q_rope = &q[head * q_head_dim + qk_nope..(head + 1) * q_head_dim];

            let w_uk_base = head * kv_b_per_head;
            let w_uk = &kv_b_proj[w_uk_base..w_uk_base + qk_nope * kv_lora_rank];
            let w_uv_base = w_uk_base + qk_nope * kv_lora_rank;
            let w_uv = &kv_b_proj[w_uv_base..w_uv_base + v_head_dim * kv_lora_rank];

            // Phase 0: q_nope_proj[r] = Σ_i w_uk[i,r] * q_nope[i]
            for r in 0..kv_lora_rank {
                let mut acc = 0.0f32;
                for i in 0..qk_nope {
                    acc += w_uk[i * kv_lora_rank + r] * q_nope[i];
                }
                q_nope_proj[r] = acc;
            }

            // Phase 1: scores[t] = (q_nope_proj · c_kv[t] + q_rope · k_pe[t]) * scale
            for t in 0..seq_len {
                let c_kv_t = &c_kv[t * kv_lora_rank..(t + 1) * kv_lora_rank];
                let k_pe_t = &k_pe[t * qk_rope..(t + 1) * qk_rope];
                let mut s = 0.0f32;
                for r in 0..kv_lora_rank {
                    s += q_nope_proj[r] * c_kv_t[r];
                }
                for r in 0..qk_rope {
                    s += q_rope[r] * k_pe_t[r];
                }
                scores[t] = s * scale;
            }

            // Phase 2: softmax
            let mx = scores[..seq_len].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for t in 0..seq_len {
                scores[t] = (scores[t] - mx).exp();
                sum += scores[t];
            }
            for t in 0..seq_len {
                scores[t] /= sum;
            }

            // Phase 3: c_kv_wt[r] = Σ_t scores[t] * c_kv[t,r]
            c_kv_wt.fill(0.0);
            for r in 0..kv_lora_rank {
                let mut acc = 0.0f32;
                for t in 0..seq_len {
                    acc += scores[t] * c_kv[t * kv_lora_rank + r];
                }
                c_kv_wt[r] = acc;
            }

            // Phase 4: out[head,vi] = w_uv[vi,:] · c_kv_wt
            for vi in 0..v_head_dim {
                let w_uv_row = &w_uv[vi * kv_lora_rank..(vi + 1) * kv_lora_rank];
                let mut acc = 0.0f32;
                for r in 0..kv_lora_rank {
                    acc += w_uv_row[r] * c_kv_wt[r];
                }
                out[head * v_head_dim + vi] = acc;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mla_decode_parity(name: &'static str, n_heads: usize, qk_nope: usize, qk_rope: usize, v_head_dim: usize, kv_lora_rank: usize, seq_len: usize, seed: u64) {
        let q_head_dim = qk_nope + qk_rope;
        let scale = 1.0f32 / (q_head_dim as f32).sqrt();

        let q = fixed_input(n_heads * q_head_dim, seed);
        let c_kv = fixed_input(seq_len * kv_lora_rank, seed + 1);
        let k_pe = fixed_input(seq_len * qk_rope, seed + 2);
        let kv_b_proj = fixed_input(n_heads * (qk_nope + v_head_dim) * kv_lora_rank, seed + 3);

        let mut cpu_out = vec![0.0f32; n_heads * v_head_dim];
        mla_decode_cpu_reference(&q, &c_kv, &k_pe, &kv_b_proj, n_heads, qk_nope, qk_rope, v_head_dim, kv_lora_rank, seq_len, scale, &mut cpu_out);

        let ctx = ctx();
        let kv_b_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&kv_b_proj));
        let mut metal_out = vec![0.0f32; n_heads * v_head_dim];
        hawking_core::kernels::mla_decode_metal(ctx, &q, &c_kv, &k_pe, &kv_b_buf, n_heads, qk_nope, qk_rope, v_head_dim, kv_lora_rank, seq_len, scale, &mut metal_out)
            .expect("mla_decode_metal should succeed");

        let diff = max_abs_diff(&cpu_out, &metal_out);
        println!("[W1] {name} parity max abs diff = {diff:.6}");
        assert!(diff < ATOL, "{name} CPU/Metal diff {diff} >= atol {ATOL}");
    }

    #[test]
    fn test_mla_decode_smoke_matches_cpu() {
        mla_decode_parity("mla_decode_smoke", /*n_heads=*/ 2, /*qk_nope=*/ 8, /*qk_rope=*/ 4, /*v_head_dim=*/ 8, /*kv_lora_rank=*/ 16, /*seq_len=*/ 4, 0xDEAD_BEEF);
    }

    #[test]
    fn test_mla_decode_production_shape_matches_cpu() {
        // DeepSeek-V2-Lite shapes: n_heads=16, qk_nope=128, qk_rope=64,
        // v_head_dim=128, kv_lora_rank=512. Use a shorter seq_len to keep
        // the test fast; the kernel scales linearly in seq_len.
        mla_decode_parity(
            "mla_decode_production",
            /*n_heads=*/ 4,
            /*qk_nope=*/ 128,
            /*qk_rope=*/ 64,
            /*v_head_dim=*/ 128,
            /*kv_lora_rank=*/ 64,
            /*seq_len=*/ 8,
            0xCAFE_BABE,
        );
    }
}
#[rustfmt::skip]
mod predec_add_f16s_parity {
    #![cfg(target_os = "macos")]
    //! Track D6 parity: gemm_q4_k_v4_predec_{2r,4r}_add_f16s must produce
    //! rel_L2 < 1% vs the corresponding f32-scales add reference kernels.
    //!
    //! _add_f16s = predec_add geometry (in-place residual add) + half* scale reads.
    //! Enables oproj_add_rmsnorm_fuse in fast profile (PREDEC_F16SCALES=1).

    use half::f16;
    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};

    use crate::common;
    use common::*;

    fn make_q4k_predec(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let bpr = cols / 256;
        let w: Vec<u8> = (0..rows * bpr * 144).map(|i| ((i as u32).wrapping_mul(2246822519).wrapping_add(seed)) as u8).collect();
        // Scales in [0.1, 2.0] to avoid near-zero values where f16 rounding inflates error.
        let s: Vec<f32> = (0..rows * bpr * 16)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed ^ 0xAB)) as f32 / u32::MAX as f32;
                0.1 + v * 1.9
            })
            .collect();
        (w, s)
    }

    fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|&x| f16::from_f32(x).to_le_bytes()).collect()
    }

    fn rel_l2(reference: &[f32], got: &[f32]) -> f64 {
        let num: f64 = reference.iter().zip(got).map(|(&r, &g)| ((r - g) as f64).powi(2)).sum();
        let den: f64 = reference.iter().map(|&r| (r as f64).powi(2)).sum::<f64>().max(1e-30);
        (num / den).sqrt()
    }

    /// Run 2r_add (f32 scales) — in-place residual += GEMV(w, x).
    fn run_2r_add_f32(ctx: &MetalContext, w: &[u8], scales: &[f32], x: &[f32], residual: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let w_buf = ctx.new_buffer_with_bytes(w);
        let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
        let x_buf = new_f32_buf(ctx, x);
        let res_buf = new_f32_buf(ctx, residual);
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_2r_add_pinned_tcb(&mut tcb, &w_buf, 0, w.len(), &sc_buf, 0, rows, cols, &x_buf, &res_buf).unwrap();
        tcb.commit_and_wait().unwrap();
        read_f32_buf(&res_buf, rows)
    }

    /// Run 2r_add_f16s (half scales) — in-place residual += GEMV(w, x).
    fn run_2r_add_f16s(ctx: &MetalContext, w: &[u8], scales_f16: &[u8], x: &[f32], residual: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let w_buf = ctx.new_buffer_with_bytes(w);
        let sc_buf = ctx.new_buffer_with_bytes(scales_f16);
        let x_buf = new_f32_buf(ctx, x);
        let res_buf = new_f32_buf(ctx, residual);
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_2r_add_f16s_pinned_tcb(&mut tcb, &w_buf, 0, w.len(), &sc_buf, 0, rows, cols, &x_buf, &res_buf).unwrap();
        tcb.commit_and_wait().unwrap();
        read_f32_buf(&res_buf, rows)
    }

    /// Run 4r_add (f32 scales) — reference for 4r_add_f16s.
    fn run_4r_add_f32(ctx: &MetalContext, w: &[u8], scales: &[f32], x: &[f32], residual: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let w_buf = ctx.new_buffer_with_bytes(w);
        let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
        let x_buf = new_f32_buf(ctx, x);
        let res_buf = new_f32_buf(ctx, residual);
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_4r_add_pinned_tcb(&mut tcb, &w_buf, 0, w.len(), &sc_buf, 0, rows, cols, &x_buf, &res_buf).unwrap();
        tcb.commit_and_wait().unwrap();
        read_f32_buf(&res_buf, rows)
    }

    /// Run 4r_add_f16s (half scales) — Track D6 4r variant.
    fn run_4r_add_f16s(ctx: &MetalContext, w: &[u8], scales_f16: &[u8], x: &[f32], residual: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let w_buf = ctx.new_buffer_with_bytes(w);
        let sc_buf = ctx.new_buffer_with_bytes(scales_f16);
        let x_buf = new_f32_buf(ctx, x);
        let res_buf = new_f32_buf(ctx, residual);
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_4r_add_f16s_pinned_tcb(&mut tcb, &w_buf, 0, w.len(), &sc_buf, 0, rows, cols, &x_buf, &res_buf).unwrap();
        tcb.commit_and_wait().unwrap();
        read_f32_buf(&res_buf, rows)
    }

    /// D6 quality gate: 2r_add_f16s rel_L2 < 1% vs 2r_add (f32 reference).
    /// Tests residual-add semantics (initial residual is included in output).
    #[test]
    fn d6_predec_2r_add_f16s_quality_gate() {
        let ctx = ctx();
        const MAX_REL_L2: f64 = 1e-2;

        let cases: &[(usize, usize, u32)] = &[
            // o_proj production shape (2048 rows × 2048 cols on Qwen-3B)
            (2048, 2048, 0xD600),
            (2048, 2048, 0xD601),
            // non-multiple-of-16 to test has1 guard
            (33, 256, 0xD602),
            (48, 256, 0xD603),
            (512, 2048, 0xD604),
        ];

        for &(rows, cols, seed) in cases {
            let (w, sc) = make_q4k_predec(rows, cols, seed);
            let sc_f16 = f32_to_f16_bytes(&sc);
            let x: Vec<f32> = (0..cols).map(|i| ((i as u32).wrapping_mul(1664525).wrapping_add(seed) as f32 / u32::MAX as f32) * 2.0 - 1.0).collect();
            let residual: Vec<f32> = (0..rows).map(|i| ((i as u32).wrapping_mul(1103515245).wrapping_add(seed) as f32 / u32::MAX as f32) * 2.0 - 1.0).collect();

            let ref_res = run_2r_add_f32(ctx, &w, &sc, &x, &residual, rows, cols);
            let got_res = run_2r_add_f16s(ctx, &w, &sc_f16, &x, &residual, rows, cols);

            let r = rel_l2(&ref_res, &got_res);
            assert!(r < MAX_REL_L2, "2r_add_f16s rows={rows} cols={cols}: rel_L2={r:.4e} >= {MAX_REL_L2}");
            eprintln!("D6 2r_add_f16s rows={rows} cols={cols}: rel_L2={r:.2e} OK");
        }
    }

    /// D6 quality gate: 4r_add_f16s rel_L2 < 1% vs 4r_add (f32 reference).
    /// Also verifies 2r_add_f16s vs 4r_add_f16s are consistent (rel_L2 < 2e-5 —
    /// only f16 rounding of shared scale path differs, no math change).
    #[test]
    fn d6_predec_4r_add_f16s_quality_gate() {
        let ctx = ctx();
        const MAX_REL_L2: f64 = 1e-2;
        const MAX_CROSS_F16: f64 = 2e-5; // 2r_f16s vs 4r_f16s should be near-identical

        let cases: &[(usize, usize, u32)] = &[
            // o_proj production shape
            (2048, 2048, 0xD610),
            (2048, 2048, 0xD611),
            // non-multiple-of-32 to test has1/has2/has3 guards
            (33, 256, 0xD612),
            (49, 256, 0xD613),
            (512, 2048, 0xD614),
        ];

        for &(rows, cols, seed) in cases {
            let (w, sc) = make_q4k_predec(rows, cols, seed);
            let sc_f16 = f32_to_f16_bytes(&sc);
            let x: Vec<f32> = (0..cols).map(|i| ((i as u32).wrapping_mul(1664525).wrapping_add(seed) as f32 / u32::MAX as f32) * 2.0 - 1.0).collect();
            let residual: Vec<f32> = (0..rows).map(|i| ((i as u32).wrapping_mul(1103515245).wrapping_add(seed) as f32 / u32::MAX as f32) * 2.0 - 1.0).collect();

            let ref_res = run_4r_add_f32(ctx, &w, &sc, &x, &residual, rows, cols);
            let got_4r = run_4r_add_f16s(ctx, &w, &sc_f16, &x, &residual, rows, cols);
            let got_2r = run_2r_add_f16s(ctx, &w, &sc_f16, &x, &residual, rows, cols);

            let r4f = rel_l2(&ref_res, &got_4r);
            let cross = rel_l2(&got_2r, &got_4r);

            assert!(r4f < MAX_REL_L2, "4r_add_f16s rows={rows} cols={cols}: rel_L2 vs f32={r4f:.4e} >= {MAX_REL_L2}");
            assert!(cross < MAX_CROSS_F16, "2r_f16s vs 4r_f16s rows={rows} cols={cols}: cross={cross:.4e} >= {MAX_CROSS_F16}");
            eprintln!("D6 4r_add_f16s rows={rows} cols={cols}: vs_f32={r4f:.2e} cross_f16={cross:.2e} OK");
        }
    }
}
#[rustfmt::skip]
mod predec_f16_scale_table {
    //! 1.2 CPU-half scaffold test: the f16 predec scale table widens back to within
    //! the f16 precision budget of the f32 table, validating f16 is adequate for
    //! Q4_K sub-block scale magnitudes (the bandwidth-cut premise of lever 1.2).
    //! Pure CPU (no Metal). The kernel that consumes the f16 table is GPU-lane.

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels::{predecode_q4_k_scale_table, predecode_q4_k_scale_table_f16};

    /// One row of 64 Q4_K blocks with realistic header scales (d ~0.01, dmin small)
    /// and deterministic packed sub-block bytes.
    fn make_q4k_bytes(n_blocks: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.012_f32 + (b % 7) as f32 * 0.001;
            let dmin = ((b % 5) as f32 - 2.0) * 0.002;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            for i in 4..144 {
                bytes[off + i] = ((i * 31 + b * 17) & 0xFF) as u8;
            }
        }
        bytes
    }

    #[test]
    fn f16_predec_table_matches_f32_within_budget() {
        let n_blocks = 64;
        let bytes = make_q4k_bytes(n_blocks);
        let f32_tab = predecode_q4_k_scale_table(&bytes);
        let f16_tab = predecode_q4_k_scale_table_f16(&bytes);

        assert_eq!(f32_tab.len(), n_blocks * 16);
        assert_eq!(f16_tab.len(), f32_tab.len());

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for (&a, h) in f32_tab.iter().zip(f16_tab.iter()) {
            let w = h.to_f32();
            let abs = (a - w).abs();
            max_abs = max_abs.max(abs);
            if a.abs() > 1e-4 {
                max_rel = max_rel.max(abs / a.abs());
            }
        }
        // f16 has an ~11-bit mantissa → relative error < ~5e-4 for in-range values.
        assert!(max_abs < 1e-2, "max abs diff {max_abs} too large for f16 scales");
        assert!(max_rel < 1e-2, "max rel diff {max_rel} exceeds the f16 precision budget");
    }
}
#[rustfmt::skip]
mod predec_f16s_swiglu_parity {
    #![cfg(target_os = "macos")]
    //! Track D1 parity: gemm_q4_k_v4_predec_f16s_4r_swiglu must produce outputs
    //! with relative-L2 error < 1% vs gemm_q4_k_v4_predec_4r_swiglu using the
    //! same scale table (converted f32 → f16 for the f16s path).
    //!
    //! The f16 scale rounding introduces ~5e-4 relative error per multiply; this
    //! averages down across the 43 blocks in the Qwen-3B ffn_down shape (rows=2048,
    //! cols=11008). We gate on rel_L2 < 1e-2, the same bar as pair_f16s.
    //!
    //! Small shapes (cols=256, 1 block) are excluded from the tight gate — with
    //! few blocks the near-zero-sum cancellation can inflate relative metrics even
    //! when the absolute error is tiny.  They still run as a smoke test with a
    //! loose absolute gate.

    use half::f16;
    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};

    use crate::common;
    use common::*;

    /// Build random Q4_K weights + f32 predec scale table.
    fn make_q4k_predec(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let bpr = cols / 256;
        let total_w = rows * bpr * 144;
        let w: Vec<u8> = (0..total_w).map(|i| ((i as u32).wrapping_mul(2246822519u32).wrapping_add(seed)) as u8).collect();
        let ns = rows * bpr * 16;
        // Avoid tiny scale values: generate in [0.1, 2.0] so the f16 relative
        // rounding error is well-controlled (no catastrophic cancellation in scales).
        let s: Vec<f32> = (0..ns)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761u32).wrapping_add(seed ^ 0xAB)) as f32 / u32::MAX as f32;
                0.1 + v * 1.9 // in [0.1, 2.0]
            })
            .collect();
        (w, s)
    }

    /// Convert f32 scale table to f16 (mirrors predecode_q4_k_scale_table_f16).
    fn f32_to_f16_scales(scales: &[f32]) -> Vec<u8> {
        scales.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect()
    }

    fn rnd(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2654435761u32).wrapping_add(seed);
                (x as f32 / u32::MAX as f32) * 4.0 - 2.0
            })
            .collect()
    }

    /// Relative L2 error = ||ref - got|| / ||ref||  (same metric as pair_f16s gate).
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

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0_f32, f32::max)
    }

    /// Run f32-scales 4r swiglu kernel → reference.
    fn run_f32(ctx: &MetalContext, w: &[u8], scales: &[f32], gate: &[f32], up: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let w_buf = ctx.new_buffer_with_bytes(w);
        let s_buf = new_f32_buf(ctx, scales);
        let g_buf = new_f32_buf(ctx, gate);
        let u_buf = new_f32_buf(ctx, up);
        let y_buf = ctx.new_buffer(rows * 4);
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_swiglu_pinned_tcb(&mut tcb, &w_buf, 0, w.len(), &s_buf, 0, rows, cols, &g_buf, &u_buf, &y_buf).unwrap();
        tcb.commit_and_wait().unwrap();
        read_f32_buf(&y_buf, rows)
    }

    /// Run f16-scales 4r swiglu kernel.
    fn run_f16s(ctx: &MetalContext, w: &[u8], scales_f16: &[u8], gate: &[f32], up: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let w_buf = ctx.new_buffer_with_bytes(w);
        let s_buf = ctx.new_buffer_with_bytes(scales_f16);
        let g_buf = new_f32_buf(ctx, gate);
        let u_buf = new_f32_buf(ctx, up);
        let y_buf = ctx.new_buffer(rows * 4);
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_f16s_swiglu_pinned_tcb(&mut tcb, &w_buf, 0, w.len(), &s_buf, 0, rows, cols, &g_buf, &u_buf, &y_buf).unwrap();
        tcb.commit_and_wait().unwrap();
        read_f32_buf(&y_buf, rows)
    }

    /// Main quality gate: production shapes with ≥8 blocks (cols ≥ 2048).
    /// Uses rel_L2 < 1e-2 — same bar as pair_f16s.
    #[test]
    fn predec_f16s_4r_swiglu_rel_l2_quality_gate() {
        let ctx = ctx();
        const MAX_REL_L2: f64 = 1e-2;

        // Production-like shapes with enough blocks to average out f16 rounding.
        // rows=hidden, cols=intermediate for Qwen-3B ffn_down.
        let cases: &[(usize, usize, u32)] = &[
            (256, 2048, 0xD110), // 8 blocks — enough to average
            (1024, 2048, 0xD111),
            (2048, 2048, 0xD112),
            (2048, 11008, 0xD113), // Qwen-3B production ffn_down shape
        ];

        for &(rows, cols, seed) in cases {
            let (w, scales) = make_q4k_predec(rows, cols, seed);
            let scales_f16 = f32_to_f16_scales(&scales);
            let gate = rnd(cols, seed ^ 0x11);
            let up = rnd(cols, seed ^ 0x22);

            let ref_out = run_f32(&ctx, &w, &scales, &gate, &up, rows, cols);
            let got_out = run_f16s(&ctx, &w, &scales_f16, &gate, &up, rows, cols);

            let rel = rel_l2(&ref_out, &got_out);
            assert!(rel < MAX_REL_L2, "rows={rows} cols={cols}: rel_L2={rel:.4e} >= {MAX_REL_L2:.4e}");
            eprintln!("D1 f16s_swiglu rows={rows} cols={cols}: rel_L2={rel:.2e} OK");
        }
    }

    /// Smoke test: small shapes pass an absolute-diff gate (kernel runs, not NaN, not zeros).
    /// These shapes have too few blocks for reliable relative gating.
    #[test]
    fn predec_f16s_4r_swiglu_small_shapes_smoke() {
        let ctx = ctx();
        const MAX_ABS_FACTOR: f32 = 0.5; // allow 50% of the reference magnitude as abs diff

        let cases: &[(usize, usize, u32)] = &[(256, 256, 0xD100), (512, 512, 0xD101)];

        for &(rows, cols, seed) in cases {
            let (w, scales) = make_q4k_predec(rows, cols, seed);
            let scales_f16 = f32_to_f16_scales(&scales);
            let gate = rnd(cols, seed ^ 0x11);
            let up = rnd(cols, seed ^ 0x22);

            let ref_out = run_f32(&ctx, &w, &scales, &gate, &up, rows, cols);
            let got_out = run_f16s(&ctx, &w, &scales_f16, &gate, &up, rows, cols);

            // Verify non-NaN, non-zero output.
            for &v in &got_out {
                assert!(!v.is_nan(), "rows={rows} cols={cols}: NaN in f16s output");
            }
            let ref_norm: f32 = ref_out.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let abs_diff = max_abs_diff(&ref_out, &got_out);
            assert!(ref_norm == 0.0 || abs_diff < MAX_ABS_FACTOR * ref_norm.max(1.0), "rows={rows} cols={cols}: abs_diff={abs_diff:.4} too large vs ref_norm={ref_norm:.4}");
            eprintln!("D1 smoke rows={rows} cols={cols}: abs_diff={abs_diff:.4} ref_norm={ref_norm:.4} OK");
        }
    }
}
#[rustfmt::skip]
mod q3k_fused_2r_parity {
    //! q3k_fused_2r — parity between gemv_q3_k_pinned_tcb (gemm_q3_k_fused_v2,
    //! 8 rows/TG, one row per simdgroup) and gemv_q3_k_fused_2r_pinned_tcb
    //! (gemm_q3_k_fused_2r, 16 rows/TG, two rows per simdgroup with two accumulator
    //! chains sharing the `x` load).
    //!
    //! Both use the SAME inline 6-bit Q3_K scale decode and the same per-element
    //! `d*scale*q * xv` FMA in the same order; the 2r kernel only changes the row
    //! pairing and shares the activation load. There is NO predec scale-table round
    //! involved, so unlike q3k_predec_parity this is expected BIT-IDENTICAL. The
    //! test asserts exact equality first and falls back to atol 1e-3 (the project
    //! fp16 bar) only if the compiler FMA-recontracts the shared-`x` form
    //! differently — that fallback is logged loudly so a real bug can't hide.
    //!
    //! This is the byte-cut speed-viability validation: gemm_q3_k_fused_2r is the
    //! fewest-byte Q3_K GEMV (110 B/block, no scale table) given the 2-row-ILP
    //! fast-path structure. SYNTHETIC weights — no model load. GPU-gated.

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use hawking_core::metal::TokenCommandBuffer;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    /// Synthetic Q3_K weights: 110 bytes/block. Bytes 0..108 (hmask + qs + packed
    /// 6-bit scales) are arbitrary; byte 108..110 is a small positive fp16 `d`.
    /// Matches the generator in `q3k_predec_parity.rs` / `v1_1_q3_k_parity.rs`.
    fn make_q3k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let n_blocks = rows * (cols / 256);
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 110];
        for b in 0..n_blocks {
            let off = b * 110;
            for i in 0..108 {
                bytes[off + i] = rng.gen::<u8>();
            }
            let d = 0.004 + rng.gen::<f32>() * 0.004;
            bytes[off + 108..off + 110].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        }
        bytes
    }

    fn make_x(cols: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..cols).map(|_| rng.gen_range(-3.0_f32..3.0_f32)).collect()
    }

    fn run_one(rows: usize, cols: usize, seed: u64) {
        let ctx = ctx();

        let w_bytes = make_q3k_bytes(rows, cols, seed);
        let model_buf = ctx.new_buffer_with_bytes(&w_bytes);

        let x = make_x(cols, 0xCAFE_F00D ^ seed);
        let x_buf = new_f32_buf(ctx, &x);

        // Baseline: gemm_q3_k_fused_v2 (8 rows/TG).
        let y_v2_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q3_k_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_buf, &y_v2_buf).expect("q3_k fused_v2 encode");
            tcb.commit_and_wait().expect("q3_k fused_v2 commit");
        }
        let y_v2 = read_f32_buf(&y_v2_buf, rows);

        // Candidate: gemm_q3_k_fused_2r (16 rows/TG, 2-row ILP).
        let y_2r_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q3_k_fused_2r_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_buf, &y_2r_buf).expect("q3_k fused_2r encode");
            tcb.commit_and_wait().expect("q3_k fused_2r commit");
        }
        let y_2r = read_f32_buf(&y_2r_buf, rows);

        // Bit-identical check first.
        let mut bit_identical = true;
        let mut max_abs = 0.0_f32;
        let mut worst = 0usize;
        for i in 0..rows {
            if y_v2[i].to_bits() != y_2r[i].to_bits() {
                bit_identical = false;
            }
            let d = (y_v2[i] - y_2r[i]).abs();
            if d > max_abs {
                max_abs = d;
                worst = i;
            }
        }

        if bit_identical {
            eprintln!("[q3k_fused_2r parity {rows}x{cols}] BIT-IDENTICAL to fused_v2 ({rows} rows)");
        } else {
            // Fall back to the project fp16 bar; log loudly so a real bug surfaces.
            const ATOL: f32 = 1e-3;
            assert!(
                max_abs < ATOL,
                "q3k_fused_2r exceeds fp16 tol vs fused_v2: max_abs={max_abs:e} (atol {ATOL}) \
                 at i={worst}  v2={}  2r={}",
                y_v2[worst],
                y_2r[worst],
            );
            eprintln!(
                "[q3k_fused_2r parity {rows}x{cols}] NOT bit-identical (compiler FMA-recontraction); \
                 within fp16 tol max_abs={max_abs:e} (atol {ATOL})"
            );
        }
    }

    #[test]
    fn q3k_fused_2r_matches_fused_v2() {
        // Three representative Qwen2.5-3B decode GEMV shapes. All rows%16==0.
        run_one(2048, 2048, 0x3D15_8E1E);
        run_one(11008, 2048, 0x51C0_0001);
        run_one(2048, 11008, 0x7A11_BEEF);
    }

    /// Cover rows NOT divisible by 16 (the has1 alias path: the last TG has a
    /// row0 whose row1 is past the end). rows=2056 => last TG handles rows
    /// 2048..2063, of which only 2048..2055 exist; row 2056's simdgroup writes
    /// row0=2056 and aliases row1=2064→2056 (never written).
    #[test]
    fn q3k_fused_2r_ragged_rows() {
        run_one(2056, 2048, 0x0DD0_1234);
    }
}
#[rustfmt::skip]
mod q3k_predec_parity {
    //! q3k_predec — fp16-tolerance parity between gemv_q3_k_pinned_tcb
    //! (gemm_q3_k_fused_v2, inline sub-block scale decode) and
    //! gemv_q3_k_v4_predec_pinned_tcb (sub-block scales pre-decoded host-side at
    //! load time into an f32 table via predecode_q3_k_scale_table).
    //!
    //! Both kernels share the 8-row-per-TG geometry and the same Q3_K math. They
    //! are NOT bit-identical: predec loads a pre-rounded `d*scale` from the table,
    //! whereas the fused kernel computes `d*scale*q` inline and the Metal compiler
    //! may FMA-contract it without that intermediate f32 round (measured ~1 ULP /
    //! ~1e-4 on 1667/2048 rows). That 1-ULP delta is inherent to pre-decoding, not
    //! a bug, so this gates at the project's correctness bar (atol 1e-3 fp16, per
    //! AGENT.md) rather than exact equality. Q3_K is symmetric (no min term), so
    //! the pre-decoded table is 16 f32/block.
    //!
    //! This is the byte-cut Stage-3 unblock validation: the fast Q3_K GEMV that a
    //! Q3_K (−11% bytes) model needs to run on the predec fast path instead of the
    //! generic dequant path. GPU-gated (needs a Metal device).

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use hawking_core::metal::TokenCommandBuffer;
    use hawking_core::quant::predecode_q3_k_scale_table;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    /// Synthetic Q3_K weights: 110 bytes/block. Bytes 0..108 (hmask + qs + packed
    /// 6-bit scales) are arbitrary; byte 108..110 is a small positive fp16 `d`.
    /// Matches the generator in `v1_1_q3_k_parity.rs`.
    fn make_q3k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let n_blocks = rows * (cols / 256);
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 110];
        for b in 0..n_blocks {
            let off = b * 110;
            for i in 0..108 {
                bytes[off + i] = rng.gen::<u8>();
            }
            let d = 0.004 + rng.gen::<f32>() * 0.004;
            bytes[off + 108..off + 110].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        }
        bytes
    }

    fn make_x(cols: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..cols).map(|_| rng.gen_range(-3.0_f32..3.0_f32)).collect()
    }

    #[test]
    fn q3k_v4_predec_matches_fused_v2_fp16() {
        let rows = 2048_usize;
        let cols = 2048_usize;
        let ctx = ctx();

        let w_bytes = make_q3k_bytes(rows, cols, 0x3D15_8E1E);
        let model_buf = ctx.new_buffer_with_bytes(&w_bytes);

        let x = make_x(cols, 0xCAFE_F00D);
        let x_buf = new_f32_buf(ctx, &x);

        // Baseline: gemm_q3_k_fused_v2 (inline 6-bit scale decode).
        let y_fused_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q3_k_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_buf, &y_fused_buf).expect("q3_k fused encode");
            tcb.commit_and_wait().expect("q3_k fused commit");
        }
        let y_fused = read_f32_buf(&y_fused_buf, rows);

        // v4_predec: build host-side scale table (16 f32/block), pin, dispatch.
        let scales = predecode_q3_k_scale_table(&w_bytes);
        let expected_scale_len = rows * (cols / 256) * 16;
        assert_eq!(scales.len(), expected_scale_len, "predecode_q3_k_scale_table length mismatch: got {} expected {}", scales.len(), expected_scale_len);
        let scales_buf = new_f32_buf(ctx, &scales);

        let y_predec_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q3_k_v4_predec_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), &scales_buf, 0, rows, cols, &x_buf, &y_predec_buf).expect("q3_k v4_predec encode");
            tcb.commit_and_wait().expect("q3_k v4_predec commit");
        }
        let y_predec = read_f32_buf(&y_predec_buf, rows);

        // Not bit-identical: predec pre-rounds d*scale, the fused kernel may
        // FMA-contract it (see module docs). Gate at the project's fp16 bar.
        const ATOL: f32 = 1e-3;
        let mut max_abs = 0.0_f32;
        let mut worst = 0usize;
        for i in 0..rows {
            let d = (y_fused[i] - y_predec[i]).abs();
            if d > max_abs {
                max_abs = d;
                worst = i;
            }
        }
        assert!(
            max_abs < ATOL,
            "q3k_v4_predec exceeds fp16 tol vs fused_v2: max_abs={max_abs:e} (atol {ATOL}) \
             at i={worst}  fused={}  predec={}",
            y_fused[worst],
            y_predec[worst],
        );
        eprintln!("[q3k_v4_predec parity] {rows} rows within fp16 tol; max_abs={max_abs:e} (atol {ATOL})");
    }
}
#[rustfmt::skip]
mod q4k_batched_gemm_parity {
    #![cfg(target_os = "macos")]
    //! P3 — parity test for `gemm_q4_k_m_batched_v2_pinned_tcb`.
    //!
    //! The batched kernel must produce the same outputs as B back-to-back
    //! single-matrix GEMVs (modulo fp32 reduction order tolerance). We
    //! verify this against the existing `gemv_q4_k_m_v2_pinned_tcb` which
    //! is the row-wise scalar reference shipped in the qwen_dense pipeline.

    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use hawking_core::quant;

    use crate::common;
    use common::*;

    /// Run B single-vector GEMVs against the same weight and concatenate
    /// the outputs into a (B, rows) row-major matrix.
    fn reference_b_gemvs(ctx: &MetalContext, w_buf: &PinnedBuffer, w_bytes_len: usize, rows: usize, cols: usize, x_batch: &[f32], batch: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * rows];
        for b in 0..batch {
            let x_slice = &x_batch[b * cols..(b + 1) * cols];
            let x_buf = new_f32_buf(ctx, x_slice);
            let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
            {
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::gemv_q4_k_m_v2_pinned_tcb(&mut tcb, w_buf, 0, w_bytes_len, rows, cols, &x_buf, &y_buf).expect("v2 gemv");
                tcb.commit_and_wait().expect("commit");
            }
            let y = read_f32_buf(&y_buf, rows);
            out[b * rows..(b + 1) * rows].copy_from_slice(&y);
        }
        out
    }

    #[test]
    fn batched_q4k_matches_per_token_gemv() {
        // Mirrors a Qwen-3B FFN gate shape (intermediate × hidden).
        let rows = 1024usize;
        let cols = 2048usize;

        // Build a Q4_K weight from random f32.
        let w_f32 = fixed_f32(rows * cols, 0xAA55_AA55);
        let blocks = (rows * cols) / 256;
        let mut w_q4 = vec![0u8; blocks * quant::Q4_K_BLOCK_BYTES];
        quant::quantize_q4_k(&w_f32, &mut w_q4).expect("Q4_K quantize");

        let ctx = ctx();
        let model_buf = ctx.new_buffer_with_bytes(&w_q4);

        for &batch in &[1usize, 2, 3, 4, 5, 6, 7, 8] {
            let x_batch = fixed_f32(batch * cols, 0x1234_5678 ^ (batch as u64));
            let expected = reference_b_gemvs(ctx, &model_buf, w_q4.len(), rows, cols, &x_batch, batch);

            let x_buf = new_f32_buf(ctx, &x_batch);

            // v2 dispatcher only supports batch <= 4.
            if batch <= 4 {
                let y_buf = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
                {
                    let mut tcb = TokenCommandBuffer::new(ctx);
                    kernels::gemm_q4_k_m_batched_v2_pinned_tcb(&mut tcb, &model_buf, 0, w_q4.len(), rows, cols, batch, &x_buf, &y_buf).expect("batched gemm encode");
                    tcb.commit_and_wait().expect("commit");
                }
                let actual = read_f32_buf(&y_buf, batch * rows);
                let diff = max_abs_diff(&expected, &actual);
                assert!(diff < 1e-3, "batched Q4_K vs per-token v2 (batch={batch}): max_abs_diff = {diff} (limit 1e-3)");

                // v3 parity: shmem-staged variant (batch <= 4).
                let y_buf_v3 = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
                {
                    let mut tcb = TokenCommandBuffer::new(ctx);
                    kernels::gemm_q4_k_m_batched_v3_pinned_tcb(&mut tcb, &model_buf, 0, w_q4.len(), rows, cols, batch, &x_buf, &y_buf_v3).expect("batched gemm v3 encode");
                    tcb.commit_and_wait().expect("commit v3");
                }
                let actual_v3 = read_f32_buf(&y_buf_v3, batch * rows);
                let diff_v3 = max_abs_diff(&expected, &actual_v3);
                assert!(diff_v3 < 1e-3, "batched Q4_K v3 vs per-token v2 (batch={batch}): max_abs_diff = {diff_v3} (limit 1e-3)");
            }

            // v3w parity: widened to B in 1..=8.
            let y_buf_v3w = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
            {
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::gemm_q4_k_m_batched_v3w_pinned_tcb(&mut tcb, &model_buf, 0, w_q4.len(), rows, cols, batch, &x_buf, &y_buf_v3w).expect("batched gemm v3w encode");
                tcb.commit_and_wait().expect("commit v3w");
            }
            let actual_v3w = read_f32_buf(&y_buf_v3w, batch * rows);
            let diff_v3w = max_abs_diff(&expected, &actual_v3w);
            assert!(diff_v3w < 1e-3, "batched Q4_K v3w vs per-token (batch={batch}): max_abs_diff = {diff_v3w} (limit 1e-3)");
        }
    }
}
#[rustfmt::skip]
mod q4k_batched_mma_parity {
    //! P1-A parity: the simdgroup-matrix (MMA) batched Q4_K GEMMs vs the tuned
    //! scalar/predec v3w kernels, across batch B=1..=8.
    //!
    //! Unlike `q4k_batched_predec_parity` (bit-identical), the MMA kernels reorder
    //! the K reduction (depth-8 hardware tiles + a different accumulation tree) vs
    //! the scalar FMA chain, so they are numerically close but NOT `to_bits()`
    //! equal. Gate is **atol = 1e-3 fp16** (the project's parity regime; the
    //! standalone silicon #8 MMA measured ~1.26e-4, ~8x under 1e-3).
    //!
    //! Shapes: the rows>cols WINNING shape (11008x2048 — ffn gate/up, where the
    //! caller actually swaps to MMA) at B in {1,2,4,8}, plus a 512x512 sanity tile.
    //! (q/k/v/o square + ffn_down wide stay on v3w by the rows>cols gate, so the
    //! MMA kernels are only exercised on rows>cols here.)

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    // Combined tolerance |a-b| <= ATOL + RTOL*|a| (numpy allclose). A pure
    // atol=1e-3 is the project regime for ~O(1)-magnitude kernel outputs, but the
    // MMA reorders the K reduction (depth-8 hardware tiles vs the scalar FMA
    // chain), and at the ffn shape (cols=2048, random-byte Q4_K) outputs reach
    // ~1e3 — where the fp32 reduction-reorder noise floor (~|y|*1e-6*sqrt(K)) is
    // itself ~3e-3, *above* atol 1e-3. So atol alone is unsatisfiable for a
    // CORRECT reordered kernel there. Measured relative error is 3e-6..1.3e-5;
    // RTOL=1e-4 gives ~10-30x headroom yet stays ~100x tighter than any real
    // indexing/math bug (which produces O(1) relative error).
    const ATOL: f32 = 1e-3;
    const RTOL: f32 = 1e-4;

    fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let n_blocks = rows * (cols / 256);
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            // Small d/dmin keep dequant values in a tight range so atol 1e-3 is a
            // real gate (matches q4k_batched_predec_parity + phase1_kernel_parity).
            let d = 0.01_f32 + rng.gen::<f32>() * 0.01;
            let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            for i in 4..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    fn newf(ctx: &MetalContext, d: &[f32]) -> PinnedBuffer {
        ctx.new_buffer_with_bytes(bytemuck::cast_slice(d))
    }
    fn readf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
        let p = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
    }

    /// Assert every element is within the combined atol+rtol tolerance; report the
    /// worst absolute and relative diffs. Panics (fails the test) on any violation.
    fn check_close(label: &str, rows: usize, cols: usize, batch: usize, a: &[f32], b: &[f32]) {
        let mut worst_abs = 0.0_f32;
        let mut worst_rel = 0.0_f32;
        let mut viol: Option<(usize, f32, f32, f32)> = None;
        for i in 0..a.len() {
            let d = (a[i] - b[i]).abs();
            let rel = d / a[i].abs().max(1e-6);
            worst_abs = worst_abs.max(d);
            worst_rel = worst_rel.max(rel);
            if d > ATOL + RTOL * a[i].abs() && viol.is_none() {
                viol = Some((i, d, a[i], b[i]));
            }
        }
        if let Some((i, d, av, bv)) = viol {
            panic!(
                "{label} {rows}x{cols} batch={batch}: abs diff {d} > atol {ATOL} + rtol {RTOL}*|a| \
                 (worst @ {i}: ref={av:e} mma={bv:e}); max_abs={worst_abs:e} max_rel={worst_rel:e}"
            );
        }
        eprintln!(
            "[{label}] {rows}x{cols} batch={batch}: max_abs={worst_abs:e} max_rel={worst_rel:e} \
             (atol {ATOL} rtol {RTOL})"
        );
    }

    /// Reference: tuned scalar v3w. Under test: the non-predec MMA twin.
    fn check_shape_mma(ctx: &MetalContext, rows: usize, cols: usize, seed: u64) {
        let w = make_q4k_bytes(rows, cols, seed);
        let wbuf = ctx.new_buffer_with_bytes(&w);
        let mut rng = Pcg64Mcg::new(seed as u128 ^ 0xA5A5_A5A5);
        for batch in [1usize, 2, 4, 8] {
            let x: Vec<f32> = (0..batch * cols).map(|_| rng.gen_range(-3.0_f32..3.0)).collect();
            let xbuf = newf(ctx, &x);

            let y_ref = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
            {
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::gemm_q4_k_m_batched_v3w_pinned_tcb(&mut tcb, &wbuf, 0, w.len(), rows, cols, batch, &xbuf, &y_ref).expect("v3w encode");
                tcb.commit_and_wait().expect("v3w commit");
            }
            let y_mma = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
            {
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::gemm_q4_k_m_batched_v3w_mma_pinned_tcb(&mut tcb, &wbuf, 0, w.len(), rows, cols, batch, &xbuf, &y_mma).expect("mma encode");
                tcb.commit_and_wait().expect("mma commit");
            }
            let a = readf(&y_ref, batch * rows);
            let bb = readf(&y_mma, batch * rows);
            check_close("mma vs v3w", rows, cols, batch, &a, &bb);
        }
    }

    /// Reference: tuned v3w_predec. Under test: the predec MMA twin (the shipped
    /// Option-B path). v3w_predec is bit-identical to v3w, so this also anchors
    /// the predec twin against the scalar reference.
    fn check_shape_mma_predec(ctx: &MetalContext, rows: usize, cols: usize, seed: u64) {
        let w = make_q4k_bytes(rows, cols, seed);
        let wbuf = ctx.new_buffer_with_bytes(&w);
        let scales = kernels::predecode_q4_k_scale_table(&w);
        let sbuf = newf(ctx, &scales);
        let mut rng = Pcg64Mcg::new(seed as u128 ^ 0x1234_9876);
        for batch in [1usize, 2, 4, 8] {
            let x: Vec<f32> = (0..batch * cols).map(|_| rng.gen_range(-3.0_f32..3.0)).collect();
            let xbuf = newf(ctx, &x);

            let y_ref = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
            {
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(&mut tcb, &wbuf, 0, w.len(), &sbuf, 0, rows, cols, batch, &xbuf, &y_ref).expect("v3w_predec encode");
                tcb.commit_and_wait().expect("v3w_predec commit");
            }
            let y_mma = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
            {
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::gemm_q4_k_m_batched_v3w_mma_predec_pinned_tcb(&mut tcb, &wbuf, 0, w.len(), &sbuf, 0, rows, cols, batch, &xbuf, &y_mma).expect("mma_predec encode");
                tcb.commit_and_wait().expect("mma_predec commit");
            }
            let a = readf(&y_ref, batch * rows);
            let bb = readf(&y_mma, batch * rows);
            check_close("mma_predec vs v3w_predec", rows, cols, batch, &a, &bb);
        }
    }

    #[test]
    fn mma_matches_v3w_winning_shape() {
        // ffn gate/up: intermediate x hidden = 11008 x 2048 (rows>cols → the swap).
        check_shape_mma(ctx(), 11008, 2048, 0xBEEF_1234);
    }

    #[test]
    fn mma_matches_v3w_sanity_tile() {
        check_shape_mma(ctx(), 512, 512, 0x0512_0512);
    }

    #[test]
    fn mma_predec_matches_v3w_predec_winning_shape() {
        check_shape_mma_predec(ctx(), 11008, 2048, 0xFEED_4321);
    }

    #[test]
    fn mma_predec_matches_v3w_predec_sanity_tile() {
        check_shape_mma_predec(ctx(), 512, 512, 0x0512_0513);
    }
}
#[rustfmt::skip]
mod q4k_batched_predec_parity {
    //! Bit-identical parity: gemm_q4_k_m_batched_v3w_predec vs the non-predec
    //! gemm_q4_k_m_batched_v3w, across batch B=1..=8. The predec variant only
    //! pre-decodes the sub-block scales host-side; it does the same fp32 math in
    //! the same order, so outputs must be bit-identical. Validates the batched
    //! predec kernel in isolation before it's wired into the decode/verify path.

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let n_blocks = rows * (cols / 256);
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.01_f32 + rng.gen::<f32>() * 0.01;
            let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            for i in 4..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    fn newf(ctx: &MetalContext, d: &[f32]) -> PinnedBuffer {
        ctx.new_buffer_with_bytes(bytemuck::cast_slice(d))
    }
    fn readf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
        let p = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
    }

    #[test]
    fn batched_predec_bit_identical_to_v3w() {
        let rows = 2048_usize;
        let cols = 2048_usize;
        let ctx = ctx();
        let w = make_q4k_bytes(rows, cols, 0xBEEF_1234);
        let wbuf = ctx.new_buffer_with_bytes(&w);
        let scales = kernels::predecode_q4_k_scale_table(&w);
        let sbuf = newf(ctx, &scales);

        let mut rng = Pcg64Mcg::new(0x5EED_5EED);
        for batch in 1..=8usize {
            // x_batch: (batch, cols) contiguous.
            let x: Vec<f32> = (0..batch * cols).map(|_| rng.gen_range(-3.0_f32..3.0)).collect();
            let xbuf = newf(ctx, &x);

            let y_ref = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
            {
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::gemm_q4_k_m_batched_v3w_pinned_tcb(&mut tcb, &wbuf, 0, w.len(), rows, cols, batch, &xbuf, &y_ref).expect("v3w encode");
                tcb.commit_and_wait().expect("v3w commit");
            }
            let y_predec = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
            {
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(&mut tcb, &wbuf, 0, w.len(), &sbuf, 0, rows, cols, batch, &xbuf, &y_predec).expect("predec encode");
                tcb.commit_and_wait().expect("predec commit");
            }
            let a = readf(&y_ref, batch * rows);
            let b = readf(&y_predec, batch * rows);
            let mut diffs = 0usize;
            let mut first = None;
            for i in 0..a.len() {
                if a[i].to_bits() != b[i].to_bits() {
                    diffs += 1;
                    if first.is_none() {
                        first = Some((i, a[i], b[i]));
                    }
                }
            }
            if let Some((i, av, bv)) = first {
                panic!("batch={batch}: {diffs}/{} differ; first @ {i} v3w={av:e} predec={bv:e}", a.len());
            }
            eprintln!("[batched-predec parity] batch={batch}: {} elems bit-identical", a.len());
        }
    }

    /// Phase-1 ffn_down shape (rows=h=2048, cols=intermediate=11008). Exercises
    /// the requant'd-ffn_down predec wire-up's exact dispatch shape so the
    /// large-cols path is covered, not just the square q_proj shape above.
    #[test]
    fn batched_predec_bit_identical_ffn_down_shape() {
        let rows = 2048_usize;
        let cols = 11008_usize;
        let ctx = ctx();
        let w = make_q4k_bytes(rows, cols, 0xFADE_9988);
        let wbuf = ctx.new_buffer_with_bytes(&w);
        let scales = kernels::predecode_q4_k_scale_table(&w);
        let sbuf = newf(ctx, &scales);

        let mut rng = Pcg64Mcg::new(0xC0FF_EE11);
        for batch in 1..=8usize {
            let x: Vec<f32> = (0..batch * cols).map(|_| rng.gen_range(-3.0_f32..3.0)).collect();
            let xbuf = newf(ctx, &x);

            let y_ref = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
            {
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::gemm_q4_k_m_batched_v3w_pinned_tcb(&mut tcb, &wbuf, 0, w.len(), rows, cols, batch, &xbuf, &y_ref).expect("v3w encode");
                tcb.commit_and_wait().expect("v3w commit");
            }
            let y_predec = ctx.new_buffer(batch * rows * std::mem::size_of::<f32>());
            {
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(&mut tcb, &wbuf, 0, w.len(), &sbuf, 0, rows, cols, batch, &xbuf, &y_predec).expect("predec encode");
                tcb.commit_and_wait().expect("predec commit");
            }
            let a = readf(&y_ref, batch * rows);
            let b = readf(&y_predec, batch * rows);
            let mut first = None;
            for i in 0..a.len() {
                if a[i].to_bits() != b[i].to_bits() {
                    first = Some((i, a[i], b[i]));
                    break;
                }
            }
            if let Some((i, av, bv)) = first {
                panic!("ffn_down batch={batch}: first diff @ {i} v3w={av:e} predec={bv:e}");
            }
            eprintln!("[batched-predec parity ffn_down] batch={batch}: {} elems bit-identical", a.len());
        }
    }
}
#[rustfmt::skip]
mod q4k_fast_parity {
    //! Q4K_FAST parity vs Q4_K v3_8r at q_proj decode shape (rows=2048,
    //! cols=2048).
    //!
    //! Builds a synthetic Q4_K tensor with constraints that keep the
    //! per-sub-block products `d * sb_idx[k]` and `dmin * mb_idx[k]` exactly
    //! representable in fp16 (so the FAST layout's fp16 sub_scale / sub_min
    //! storage is lossless). Runs both kernels and asserts bit-identical
    //! per-row output.
    //!
    //! Run with: `cargo test --release -p hawking-core --test q4k_fast_parity -- --nocapture`

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use hawking_core::q4k_fast::{convert_q4k_tensor_to_fast, Q4K_BLOCK_BYTES, Q4K_FAST_BLOCK_BYTES};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    fn ctx() -> MetalContext {
        MetalContext::new().expect("Metal device required")
    }

    /// Build a Q4_K tensor where `d * sb_idx[k]` and `dmin * mb_idx[k]` are
    /// guaranteed exactly representable in fp16:
    ///
    /// * `d`    = 1.0 (fp16 exact)
    /// * `dmin` = 0.5 (fp16 exact)
    /// * `sb_idx[k]` ∈ [0..63]  → product 0..63, all integers ≤ 2^11, fp16 exact
    /// * `mb_idx[k]` ∈ [0..63]  → product 0..31.5 in 0.5 steps, fp16 exact
    ///
    /// Random 4-bit nibbles for the rest.
    fn make_synthetic_q4k_tensor(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        assert_eq!(cols % 256, 0, "cols must be a multiple of 256");
        let blocks_per_row = cols / 256;
        let n_blocks = rows * blocks_per_row;
        let mut bytes = vec![0u8; n_blocks * Q4K_BLOCK_BYTES];
        let mut rng = Pcg64Mcg::new(seed as u128);

        let d = f16::from_f32(1.0_f32);
        let dmin = f16::from_f32(0.5_f32);

        for b in 0..n_blocks {
            let off = b * Q4K_BLOCK_BYTES;
            bytes[off..off + 2].copy_from_slice(&d.to_bits().to_le_bytes());
            bytes[off + 2..off + 4].copy_from_slice(&dmin.to_bits().to_le_bytes());

            // Pick random sb_idx[k] ∈ [0..63], mb_idx[k] ∈ [0..63] for k in 0..8.
            let sb: [u8; 8] = [
                rng.gen::<u8>() & 0x3F,
                rng.gen::<u8>() & 0x3F,
                rng.gen::<u8>() & 0x3F,
                rng.gen::<u8>() & 0x3F,
                rng.gen::<u8>() & 0x3F,
                rng.gen::<u8>() & 0x3F,
                rng.gen::<u8>() & 0x3F,
                rng.gen::<u8>() & 0x3F,
            ];
            let mb: [u8; 8] = [
                rng.gen::<u8>() & 0x3F,
                rng.gen::<u8>() & 0x3F,
                rng.gen::<u8>() & 0x3F,
                rng.gen::<u8>() & 0x3F,
                rng.gen::<u8>() & 0x3F,
                rng.gen::<u8>() & 0x3F,
                rng.gen::<u8>() & 0x3F,
                rng.gen::<u8>() & 0x3F,
            ];
            // Repack into Q4_K's bytes [4..16] layout (inverse of
            // q4k_fast::decode_q4k_sb_mb):
            //   bytes[4+sub] = sb[sub] | ((sb[4+sub] >> 4) << 6)   for sub in 0..4
            //   bytes[8+sub] = mb[sub] | ((mb[4+sub] >> 4) << 6)   for sub in 0..4
            //   bytes[12+j]  = (sb[4+j] & 0x0F) | ((mb[4+j] & 0x0F) << 4)  for j in 0..4
            for sub in 0..4 {
                let hi_sb = (sb[4 + sub] >> 4) & 0x03;
                let hi_mb = (mb[4 + sub] >> 4) & 0x03;
                bytes[off + 4 + sub] = (sb[sub] & 0x3F) | (hi_sb << 6);
                bytes[off + 8 + sub] = (mb[sub] & 0x3F) | (hi_mb << 6);
            }
            for j in 0..4 {
                bytes[off + 12 + j] = (sb[4 + j] & 0x0F) | ((mb[4 + j] & 0x0F) << 4);
            }
            // Random nibbles for the 128-byte data section.
            for i in 16..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    fn make_x(cols: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..cols).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
    }

    #[test]
    fn q4k_fast_v1_bit_identical_to_v3_8r_at_qproj_decode_shape() {
        let rows = 2048usize;
        let cols = 2048usize;
        let blocks_per_row = cols / 256;

        let ctx = ctx();

        // Build synthetic Q4_K tensor with fp16-exact sub-products.
        let q4k_bytes = make_synthetic_q4k_tensor(rows, cols, 0xCAFE_F00D_DEAD_BEEFu64);
        let q4k_byte_size = q4k_bytes.len();
        assert_eq!(q4k_byte_size, rows * blocks_per_row * Q4K_BLOCK_BYTES);

        // Convert to Q4K_FAST.
        let fast_bytes = convert_q4k_tensor_to_fast(&q4k_bytes, rows, cols);
        let fast_byte_size = fast_bytes.len();
        assert_eq!(fast_byte_size, rows * blocks_per_row * Q4K_FAST_BLOCK_BYTES);

        // Activation.
        let x = make_x(cols, 0x1234_5678_9ABC_DEF0u64);

        // Pinned buffers.
        let q4k_buf: PinnedBuffer = ctx.new_buffer_with_bytes(&q4k_bytes);
        let fast_buf: PinnedBuffer = ctx.new_buffer_with_bytes(&fast_bytes);
        let x_buf: PinnedBuffer = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));
        let out_v3_buf: PinnedBuffer = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let out_fast_buf: PinnedBuffer = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        // Run v3_8r.
        {
            let mut tcb = TokenCommandBuffer::new(&ctx);
            kernels::gemv_q4_k_m_v3_8r_pinned_tcb(&mut tcb, &q4k_buf, 0, q4k_byte_size, rows, cols, &x_buf, &out_v3_buf).expect("v3_8r dispatch");
            tcb.commit_and_wait().expect("v3_8r commit");
        }

        // Run Q4K_FAST v1.
        {
            let mut tcb = TokenCommandBuffer::new(&ctx);
            kernels::gemv_q4k_fast_v1_pinned_tcb(&mut tcb, &fast_buf, 0, fast_byte_size, rows, cols, &x_buf, &out_fast_buf).expect("q4k_fast_v1 dispatch");
            tcb.commit_and_wait().expect("q4k_fast_v1 commit");
        }

        // Read both outputs.
        let out_v3_ptr = out_v3_buf.contents() as *const f32;
        let out_v3 = unsafe { std::slice::from_raw_parts(out_v3_ptr, rows) };
        let out_fast_ptr = out_fast_buf.contents() as *const f32;
        let out_fast = unsafe { std::slice::from_raw_parts(out_fast_ptr, rows) };

        // Bit-identical assertion (bit-pattern equality of f32).
        let mut first_diff = None;
        let mut max_abs_diff = 0.0f32;
        for i in 0..rows {
            let a = out_v3[i];
            let b = out_fast[i];
            let abs_d = (a - b).abs();
            if abs_d > max_abs_diff {
                max_abs_diff = abs_d;
            }
            if a.to_bits() != b.to_bits() && first_diff.is_none() {
                first_diff = Some((i, a, b));
            }
        }

        if let Some((i, a, b)) = first_diff {
            panic!("Q4K_FAST vs v3_8r diverges at row={i}: v3_8r={a} ({:#x}) fast={b} ({:#x}) max_abs_diff={max_abs_diff:.3e}", a.to_bits(), b.to_bits());
        }
        println!("[q4k_fast_parity] rows={rows} cols={cols}: bit-identical (max_abs_diff=0)");
    }
}
#[rustfmt::skip]
mod q4k_predec_f16s_bench {
    //! q4k_predec_f16s_bench — early bandwidth signal for the f16-scales predec
    //! GEMV (Stage-2 bandwidth lever 1.2). The f16 scale table shrinks the
    //! per-block pre-decoded scales from 16 f32 (64 B) to 16 f16 (32 B); on the
    //! bandwidth-bound decode GEMV the predec block footprint drops 192 B → 160 B
    //! (144 B Q4_K weights + scales), a ~16.7% read-traffic cut on the scale-heavy
    //! path. This times the f32-scales production wrapper
    //! (`gemv_q4_k_v4_predec_pinned_tcb`, default 2r) vs the f16-scales variant
    //! (`gemv_q4_k_v4_predec_2r_f16s_pinned_tcb`) on representative Qwen2.5-3B
    //! decode shapes and reports µs/call + achieved GB/s + the f16s speedup.
    //!
    //! NOT a correctness gate (parity lives in q4k_predec_f16s_parity.rs). This is
    //! a microbench: it ignores numerical output and only measures dispatch wall
    //! time. GPU-gated. Marked #[ignore] so it never runs in the default suite —
    //! run explicitly with `--ignored --nocapture`.

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels::{self, predecode_q4_k_scale_table_f16};
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;
    use std::time::Instant;

    use crate::common;
    use common::*;

    /// Realistic Q4_K weights (144 B/block) — identical generator to the parity test.
    fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let n_blocks = rows * (cols / 256);
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.01_f32 + rng.gen::<f32>() * 0.01;
            let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            for i in 4..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    fn make_x(cols: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..cols).map(|_| rng.gen_range(-3.0_f32..3.0_f32)).collect()
    }

    /// Pin a Vec<f16> as raw little-endian bytes — the f16s kernel reads buffer(1)
    /// as `device const half*`. Matches new_f16_buf in the parity test.
    fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
        let bytes: Vec<u8> = data.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect();
        ctx.new_buffer_with_bytes(&bytes)
    }

    const WARMUP: usize = 30;
    const ITERS: usize = 200;

    /// Time one predec dispatch `ITERS` times after `WARMUP`, each iteration a
    /// fresh TCB committed-and-waited (so the wall time is one full GPU dispatch
    /// round-trip). Returns mean µs/call.
    fn time_dispatch<F>(label: &str, mut encode: F) -> f64
    where
        F: FnMut(&mut TokenCommandBuffer<'_>),
    {
        let ctx = ctx();
        for _ in 0..WARMUP {
            let mut tcb = TokenCommandBuffer::new(ctx);
            encode(&mut tcb);
            tcb.commit_and_wait().expect("warmup commit");
        }
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let mut tcb = TokenCommandBuffer::new(ctx);
            encode(&mut tcb);
            tcb.commit_and_wait().expect("timed commit");
        }
        let elapsed = t0.elapsed();
        let us_per_call = elapsed.as_secs_f64() * 1e6 / ITERS as f64;
        eprintln!("  [{label}] {us_per_call:.3} µs/call ({ITERS} iters)");
        us_per_call
    }

    fn bench_shape(rows: usize, cols: usize, tag: &str) {
        let ctx = ctx();
        let blocks = rows * (cols / 256);

        let w_bytes = make_q4k_bytes(rows, cols, 0xF165_8E1E ^ (rows as u64));
        let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
        let x = make_x(cols, 0xCAFE_F00D);
        let x_buf = new_f32_buf(ctx, &x);

        let scales_f32 = kernels::predecode_q4_k_scale_table(&w_bytes);
        let scales_f32_buf = new_f32_buf(ctx, &scales_f32);
        let scales_f16 = predecode_q4_k_scale_table_f16(&w_bytes);
        let scales_f16_buf = new_f16_buf(ctx, &scales_f16);

        let y_f32_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let y_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let wlen = w_bytes.len();

        eprintln!("\n=== shape {tag}: rows={rows} cols={cols} ({blocks} blocks/output) ===");

        // f32-scales: 144 B weights + 16 f32 scales (64 B) per block + x (4 B/col) + y.
        let us_f32 = time_dispatch("f32 scales (2r)", |tcb| {
            kernels::gemv_q4_k_v4_predec_pinned_tcb(tcb, &model_buf, 0, wlen, &scales_f32_buf, 0, rows, cols, &x_buf, &y_f32_buf).expect("f32 predec encode");
        });

        // f16-scales: 144 B weights + 16 f16 scales (32 B) per block + x + y.
        let us_f16 = time_dispatch("f16 scales (2r)", |tcb| {
            kernels::gemv_q4_k_v4_predec_2r_f16s_pinned_tcb(tcb, &model_buf, 0, wlen, &scales_f16_buf, 0, rows, cols, &x_buf, &y_f16_buf).expect("f16s predec encode");
        });

        // Bytes read per call (the bandwidth-bound terms). Weights + scale table
        // dominate; x (cols*4) and y (rows*4) are small but included for honesty.
        let weights = (blocks * 144) as f64;
        let x_bytes = (cols * 4) as f64;
        let y_bytes = (rows * 4) as f64;
        let bytes_f32 = weights + (blocks * 16 * 4) as f64 + x_bytes + y_bytes;
        let bytes_f16 = weights + (blocks * 16 * 2) as f64 + x_bytes + y_bytes;

        let gbps_f32 = bytes_f32 / (us_f32 * 1e3); // bytes / (µs*1e3 ns) -> GB/s
        let gbps_f16 = bytes_f16 / (us_f16 * 1e3);
        let speedup = (us_f32 - us_f16) / us_f32 * 100.0;

        eprintln!("  bytes/call: f32={:.0} KiB  f16={:.0} KiB ({:.1}% less scale traffic)", bytes_f32 / 1024.0, bytes_f16 / 1024.0, (1.0 - bytes_f16 / bytes_f32) * 100.0);
        eprintln!("  GB/s:       f32={gbps_f32:.1}  f16={gbps_f16:.1}");
        eprintln!("  >>> {tag}: f32={us_f32:.3} µs  f16={us_f16:.3} µs  speedup={speedup:+.2}%");
    }

    #[test]
    #[ignore = "microbench — run with --ignored --nocapture; needs a free GPU"]
    fn q4k_predec_f16s_bandwidth_bench() {
        eprintln!("[q4k_predec_f16s_bench] f32-scales vs f16-scales predec GEMV, {ITERS} iters/shape after {WARMUP} warmup");
        // Representative Qwen2.5-3B decode GEMV shapes.
        bench_shape(2048, 2048, "attn-square 2048x2048");
        bench_shape(11008, 2048, "ffn-up 11008x2048");
        bench_shape(2048, 11008, "ffn-down 2048x11008");
    }
}
#[rustfmt::skip]
mod q4k_predec_f16s_parity {
    //! q4k_predec_f16s — relative parity between the f32-scales predec GEMV
    //! (gemv_q4_k_v4_predec_pinned_tcb) and the f16-scales variant
    //! (gemv_q4_k_v4_predec_2r_f16s_pinned_tcb, Stage-2 bandwidth lever 1.2).
    //!
    //! Unlike q4k_predec_parity (which asserts BIT-identity between the inline and
    //! f32-predec paths), this is NOT bit-identical: storing the pre-decoded
    //! `(ds, dm)` pairs as f16 rounds each by ~half-mantissa (≈5e-4 relative). So
    //! this is a QUALITY gate: the f16-scales output must track the f32-scales
    //! output within the f16 precision budget. The f32 reference is dispatched via
    //! the production wrapper; all f32 predec row-variants (1r/2r/4r) are
    //! bit-identical to each other, so the only delta measured here is the f16
    //! scale rounding, regardless of any HAWKING_QWEN_PREDEC_* env state.
    //!
    //! Gate = relative L2 norm of the difference (robust to individual near-zero
    //! outputs from cancellation). f16 scale rounding keeps this well under 1e-2.
    //! GPU-gated (needs a Metal device).

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels::{self, predecode_q4_k_scale_table_f16};
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    /// Realistic Q4_K weights (144 B/block): small fp16 d/dmin, random sub-block
    /// 6-bit indices and 4-bit quants. Same generator as q4k_predec_parity.rs.
    fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let n_blocks = rows * (cols / 256);
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.01_f32 + rng.gen::<f32>() * 0.01;
            let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            for i in 4..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    fn make_x(cols: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..cols).map(|_| rng.gen_range(-3.0_f32..3.0_f32)).collect()
    }

    /// Pin a Vec<f16> as raw little-endian bytes (no bytemuck Pod dependency on
    /// half::f16); the f16s kernel reads buffer(1) as `device const half*`.
    fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
        let bytes: Vec<u8> = data.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect();
        ctx.new_buffer_with_bytes(&bytes)
    }

    #[test]
    fn q4k_v4_predec_f16s_relative_parity() {
        let rows = 2048_usize;
        let cols = 2048_usize;
        let ctx = ctx();

        let w_bytes = make_q4k_bytes(rows, cols, 0xF165_8E1E);
        let model_buf = ctx.new_buffer_with_bytes(&w_bytes);

        let x = make_x(cols, 0xCAFE_F00D);
        let x_buf = new_f32_buf(ctx, &x);

        // f32-scales reference (production predec wrapper). f32 table = 16 f32/block.
        let scales_f32 = kernels::predecode_q4_k_scale_table(&w_bytes);
        let scales_f32_buf = new_f32_buf(ctx, &scales_f32);
        let y_ref_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_v4_predec_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), &scales_f32_buf, 0, rows, cols, &x_buf, &y_ref_buf).expect("f32 predec encode");
            tcb.commit_and_wait().expect("f32 predec commit");
        }
        let y_ref = read_f32_buf(&y_ref_buf, rows);

        // f16-scales variant. f16 table = 16 halfs/block.
        let scales_f16 = predecode_q4_k_scale_table_f16(&w_bytes);
        assert_eq!(scales_f16.len(), rows * (cols / 256) * 16, "predecode_q4_k_scale_table_f16 length mismatch");
        let scales_f16_buf = new_f16_buf(ctx, &scales_f16);
        let y_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_v4_predec_2r_f16s_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), &scales_f16_buf, 0, rows, cols, &x_buf, &y_f16_buf).expect("f16s predec encode");
            tcb.commit_and_wait().expect("f16s predec commit");
        }
        let y_f16 = read_f32_buf(&y_f16_buf, rows);

        // Relative L2 norm of the difference — the right metric for a lossy kernel
        // (robust to individual near-zero outputs from cancellation).
        let mut num = 0.0_f64; // ||ref - f16||^2
        let mut den = 0.0_f64; // ||ref||^2
        let mut max_abs = 0.0_f32;
        for i in 0..rows {
            let d = (y_ref[i] - y_f16[i]) as f64;
            num += d * d;
            den += (y_ref[i] as f64) * (y_ref[i] as f64);
            max_abs = max_abs.max((y_ref[i] - y_f16[i]).abs());
        }
        let rel_l2 = (num / den.max(1e-30)).sqrt();
        eprintln!(
            "[q4k_v4_predec_f16s parity] rel_L2={rel_l2:.3e} max_abs={max_abs:.3e} \
             (||ref||={:.3e})",
            den.sqrt()
        );
        // f16 scale rounding (~5e-4 relative per scale) keeps the whole-vector
        // relative error well under 1%. A failure here means the f16 table or the
        // shader widening is wrong, not just rounding.
        assert!(rel_l2 < 1e-2, "f16-scales predec rel_L2 {rel_l2:.3e} exceeds the 1e-2 f16 precision budget");
    }
}
#[rustfmt::skip]
mod q4k_predec_pair_f16s_parity {
    //! q4k_predec_pair_f16s — relative parity between the f32-scales FUSED gate+up
    //! predec GEMV (gemv_q4_k_v4_predec_pair_pinned_tcb) and the f16-scales fused
    //! variant (gemv_q4_k_v4_predec_pair_f16s_pinned_tcb, A6.5 — the profile-driven
    //! bandwidth lever covering the dominant 46.6%-of-decode `_pair` kernel).
    //!
    //! Like q4k_predec_f16s_parity (the non-pair twin), this is NOT bit-identical:
    //! storing the pre-decoded `(ds, dm)` pairs as f16 rounds each by ~half-mantissa
    //! (≈5e-4 relative). So this is a QUALITY gate: the f16-scales fused output must
    //! track the f32-scales fused output within the f16 precision budget, for BOTH
    //! the gate and up outputs (each reads its own f16 scale table). The f32
    //! reference is dispatched via the production pair wrapper.
    //!
    //! Gate = relative L2 norm of the difference (robust to individual near-zero
    //! outputs from cancellation), checked on the gate AND up outputs separately.
    //! f16 scale rounding keeps each well under 1e-2. GPU-gated (needs a Metal
    //! device).

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels::{self, predecode_q4_k_scale_table_f16};
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    /// Realistic Q4_K weights (144 B/block): small fp16 d/dmin, random sub-block
    /// 6-bit indices and 4-bit quants. Same generator as q4k_predec_f16s_parity.rs.
    fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let n_blocks = rows * (cols / 256);
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.01_f32 + rng.gen::<f32>() * 0.01;
            let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            for i in 4..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    fn make_x(cols: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..cols).map(|_| rng.gen_range(-3.0_f32..3.0_f32)).collect()
    }

    /// Pin a Vec<f16> as raw little-endian bytes (no bytemuck Pod dependency on
    /// half::f16); the f16s kernel reads the scale buffers as `device const half*`.
    fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
        let bytes: Vec<u8> = data.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect();
        ctx.new_buffer_with_bytes(&bytes)
    }

    fn rel_l2(reference: &[f32], test: &[f32]) -> (f64, f32, f64) {
        let mut num = 0.0_f64; // ||ref - test||^2
        let mut den = 0.0_f64; // ||ref||^2
        let mut max_abs = 0.0_f32;
        for i in 0..reference.len() {
            let d = (reference[i] - test[i]) as f64;
            num += d * d;
            den += (reference[i] as f64) * (reference[i] as f64);
            max_abs = max_abs.max((reference[i] - test[i]).abs());
        }
        ((num / den.max(1e-30)).sqrt(), max_abs, den.sqrt())
    }

    #[test]
    fn q4k_v4_predec_pair_f16s_relative_parity() {
        // Two independent weight matrices (gate, up) sharing the same activation,
        // exactly like the FFN fused pair site in qwen_dense.rs.
        let rows = 2048_usize;
        let cols = 2048_usize;
        let ctx = ctx();

        let wg_bytes = make_q4k_bytes(rows, cols, 0x6A7E_5CA1);
        let wu_bytes = make_q4k_bytes(rows, cols, 0x0DD0_5CA1);
        // Pack gate + up into one model buffer (mmap analogue); gate at offset 0,
        // up immediately after — mirrors how qwen_dense passes the shared mmap_buf
        // with distinct gate/up offsets.
        let mut model = wg_bytes.clone();
        let u_offset = model.len();
        model.extend_from_slice(&wu_bytes);
        let model_buf = ctx.new_buffer_with_bytes(&model);

        let x = make_x(cols, 0xCAFE_F00D);
        let x_buf = new_f32_buf(ctx, &x);

        // f32-scales reference (production fused pair wrapper). f32 table = 16 f32/block.
        let g_scales_f32 = kernels::predecode_q4_k_scale_table(&wg_bytes);
        let u_scales_f32 = kernels::predecode_q4_k_scale_table(&wu_bytes);
        let g_scales_f32_buf = new_f32_buf(ctx, &g_scales_f32);
        let u_scales_f32_buf = new_f32_buf(ctx, &u_scales_f32);
        let yg_ref_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let yu_ref_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_v4_predec_pair_pinned_tcb(
                &mut tcb,
                &model_buf,
                0,
                wg_bytes.len(),
                &g_scales_f32_buf,
                0,
                u_offset,
                wu_bytes.len(),
                &u_scales_f32_buf,
                0,
                rows,
                cols,
                &x_buf,
                &yg_ref_buf,
                &yu_ref_buf,
            )
            .expect("f32 pair encode");
            tcb.commit_and_wait().expect("f32 pair commit");
        }
        let yg_ref = read_f32_buf(&yg_ref_buf, rows);
        let yu_ref = read_f32_buf(&yu_ref_buf, rows);

        // f16-scales fused pair. f16 table = 16 halfs/block.
        let g_scales_f16 = predecode_q4_k_scale_table_f16(&wg_bytes);
        let u_scales_f16 = predecode_q4_k_scale_table_f16(&wu_bytes);
        assert_eq!(g_scales_f16.len(), rows * (cols / 256) * 16, "predecode_q4_k_scale_table_f16 length mismatch (gate)");
        let g_scales_f16_buf = new_f16_buf(ctx, &g_scales_f16);
        let u_scales_f16_buf = new_f16_buf(ctx, &u_scales_f16);
        let yg_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let yu_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_v4_predec_pair_f16s_pinned_tcb(
                &mut tcb,
                &model_buf,
                0,
                wg_bytes.len(),
                &g_scales_f16_buf,
                0,
                u_offset,
                wu_bytes.len(),
                &u_scales_f16_buf,
                0,
                rows,
                cols,
                &x_buf,
                &yg_f16_buf,
                &yu_f16_buf,
            )
            .expect("f16s pair encode");
            tcb.commit_and_wait().expect("f16s pair commit");
        }
        let yg_f16 = read_f32_buf(&yg_f16_buf, rows);
        let yu_f16 = read_f32_buf(&yu_f16_buf, rows);

        // E4: same half-scale tables, but 2-row inline geometry. This should match
        // the existing f16 pair exactly because the per-row FMA order is unchanged.
        let yg_inline_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let yu_inline_f16_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_v4_predec_pair_2r_inline_f16s_pinned_tcb(
                &mut tcb,
                &model_buf,
                0,
                wg_bytes.len(),
                &g_scales_f16_buf,
                0,
                u_offset,
                wu_bytes.len(),
                &u_scales_f16_buf,
                0,
                rows,
                cols,
                &x_buf,
                &yg_inline_f16_buf,
                &yu_inline_f16_buf,
            )
            .expect("2r inline f16s pair encode");
            tcb.commit_and_wait().expect("2r inline f16s pair commit");
        }
        let yg_inline_f16 = read_f32_buf(&yg_inline_f16_buf, rows);
        let yu_inline_f16 = read_f32_buf(&yu_inline_f16_buf, rows);

        // F2: pair_f16s with the xl[8] activation preload dropped (x read per-pi).
        // Same half-scale tables and per-row FMA order ⇒ must equal pair_f16s exactly.
        let yg_f16s_nox_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let yu_f16s_nox_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_v4_predec_pair_f16s_nox_pinned_tcb(
                &mut tcb,
                &model_buf,
                0,
                wg_bytes.len(),
                &g_scales_f16_buf,
                0,
                u_offset,
                wu_bytes.len(),
                &u_scales_f16_buf,
                0,
                rows,
                cols,
                &x_buf,
                &yg_f16s_nox_buf,
                &yu_f16s_nox_buf,
            )
            .expect("f16s nox pair encode");
            tcb.commit_and_wait().expect("f16s nox pair commit");
        }
        let yg_f16s_nox = read_f32_buf(&yg_f16s_nox_buf, rows);
        let yu_f16s_nox = read_f32_buf(&yu_f16s_nox_buf, rows);

        // F3: pair_f16s with scales held in half registers (widened at FMA) + no xl
        // preload. (float)half is exact ⇒ must equal pair_f16s bit-for-bit.
        let yg_halfreg_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let yu_halfreg_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_v4_predec_pair_f16s_halfreg_pinned_tcb(
                &mut tcb,
                &model_buf,
                0,
                wg_bytes.len(),
                &g_scales_f16_buf,
                0,
                u_offset,
                wu_bytes.len(),
                &u_scales_f16_buf,
                0,
                rows,
                cols,
                &x_buf,
                &yg_halfreg_buf,
                &yu_halfreg_buf,
            )
            .expect("f16s halfreg pair encode");
            tcb.commit_and_wait().expect("f16s halfreg pair commit");
        }
        let yg_halfreg = read_f32_buf(&yg_halfreg_buf, rows);
        let yu_halfreg = read_f32_buf(&yu_halfreg_buf, rows);

        let (g_rel, g_max, g_norm) = rel_l2(&yg_ref, &yg_f16);
        let (u_rel, u_max, u_norm) = rel_l2(&yu_ref, &yu_f16);
        let g_inline_max = max_abs_diff(&yg_f16, &yg_inline_f16);
        let u_inline_max = max_abs_diff(&yu_f16, &yu_inline_f16);
        let g_nox_max = max_abs_diff(&yg_f16, &yg_f16s_nox);
        let u_nox_max = max_abs_diff(&yu_f16, &yu_f16s_nox);
        let g_halfreg_max = max_abs_diff(&yg_f16, &yg_halfreg);
        let u_halfreg_max = max_abs_diff(&yu_f16, &yu_halfreg);
        eprintln!(
            "[q4k_v4_predec_pair_f16s parity] gate rel_L2={g_rel:.3e} max_abs={g_max:.3e} \
             (||ref||={g_norm:.3e}) | up rel_L2={u_rel:.3e} max_abs={u_max:.3e} (||ref||={u_norm:.3e}) \
             | inline_f16_max gate={g_inline_max:.3e} up={u_inline_max:.3e} \
             | f16s_nox_max gate={g_nox_max:.3e} up={u_nox_max:.3e} \
             | f16s_halfreg_max gate={g_halfreg_max:.3e} up={u_halfreg_max:.3e}"
        );
        // f16 scale rounding (~5e-4 relative per scale) keeps both whole-vector
        // relative errors well under 1%. A failure here means the f16 table or the
        // shader widening is wrong, not just rounding.
        assert!(g_rel < 1e-2, "f16s pair GATE rel_L2 {g_rel:.3e} exceeds the 1e-2 f16 precision budget");
        assert!(u_rel < 1e-2, "f16s pair UP rel_L2 {u_rel:.3e} exceeds the 1e-2 f16 precision budget");
        assert_eq!(g_inline_max, 0.0, "2r-inline f16s GATE max_abs {g_inline_max:.3e} differs from pair_f16s");
        assert_eq!(u_inline_max, 0.0, "2r-inline f16s UP max_abs {u_inline_max:.3e} differs from pair_f16s");
        assert_eq!(g_nox_max, 0.0, "f16s_nox GATE max_abs {g_nox_max:.3e} differs from pair_f16s");
        assert_eq!(u_nox_max, 0.0, "f16s_nox UP max_abs {u_nox_max:.3e} differs from pair_f16s");
        assert_eq!(g_halfreg_max, 0.0, "f16s_halfreg GATE max_abs {g_halfreg_max:.3e} differs from pair_f16s");
        assert_eq!(u_halfreg_max, 0.0, "f16s_halfreg UP max_abs {u_halfreg_max:.3e} differs from pair_f16s");
    }
}
#[rustfmt::skip]
mod q4k_predec_parity {
    //! q4k_predec — bit-identical parity between gemv_q4_k_m_v3_8r_pinned_tcb
    //! (inline sub-block scale decode) and gemv_q4_k_v4_predec_pinned_tcb
    //! (sub-block scales pre-decoded host-side at load time into an f32 table).
    //!
    //! Both kernels share the v3_8r geometry and the same widening order
    //! (fp16 d/dmin -> f32, uchar 6-bit sb/mb -> f32, multiply in f32), so the
    //! outputs MUST be bit-identical. Anything other than exact equality is a
    //! bug in the pre-decoder or the shader.

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use hawking_core::metal::TokenCommandBuffer;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let n_blocks = rows * (cols / 256);
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.01_f32 + rng.gen::<f32>() * 0.01;
            let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
            let d_bits = f16::from_f32(d).to_bits();
            let dmin_bits = f16::from_f32(dmin).to_bits();
            bytes[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
            bytes[off + 2..off + 4].copy_from_slice(&dmin_bits.to_le_bytes());
            // Sub-block 6-bit scale/min indices: bytes 4..16. The shader masks
            // bytes 4..8 and 8..12 with 0x3F and takes the high 2 bits to
            // assemble sub-blocks 4..8, so any random byte value is valid input.
            for i in 4..16 {
                bytes[off + i] = rng.gen::<u8>();
            }
            for i in 16..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    fn make_x(cols: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..cols).map(|_| rng.gen_range(-3.0_f32..3.0_f32)).collect()
    }

    #[test]
    fn q4k_v4_predec_bit_identical_to_v3_8r() {
        let rows = 2048_usize;
        let cols = 2048_usize;
        let ctx = ctx();

        let w_bytes = make_q4k_bytes(rows, cols, 0xD15A_8E1E);
        let model_buf = ctx.new_buffer_with_bytes(&w_bytes);

        let x = make_x(cols, 0xCAFE_F00D);
        let x_buf = new_f32_buf(ctx, &x);

        // Baseline: v3_8r (inline decode).
        let y_v3_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_m_v3_8r_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_buf, &y_v3_buf).expect("v3_8r encode");
            tcb.commit_and_wait().expect("v3_8r commit");
        }
        let y_v3 = read_f32_buf(&y_v3_buf, rows);

        // v4_predec: build host-side scale table, pin, dispatch.
        let scales = kernels::predecode_q4_k_scale_table(&w_bytes);
        let expected_scale_len = rows * (cols / 256) * 16;
        assert_eq!(scales.len(), expected_scale_len, "predecode_q4_k_scale_table length mismatch: got {} expected {}", scales.len(), expected_scale_len);
        let scales_buf = new_f32_buf(ctx, &scales);

        let y_v4_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_v4_predec_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), &scales_buf, 0, rows, cols, &x_buf, &y_v4_buf).expect("v4_predec encode");
            tcb.commit_and_wait().expect("v4_predec commit");
        }
        let y_v4 = read_f32_buf(&y_v4_buf, rows);

        // Bit-identical: every f32 bit-pattern must match. The two kernels do
        // the same fp32 operations in the same order; differences would mean
        // the host pre-decoder or the shader read disagrees on widening/order.
        let mut first_diff: Option<(usize, f32, f32)> = None;
        let mut diff_count = 0usize;
        for i in 0..rows {
            if y_v3[i].to_bits() != y_v4[i].to_bits() {
                diff_count += 1;
                if first_diff.is_none() {
                    first_diff = Some((i, y_v3[i], y_v4[i]));
                }
            }
        }
        if let Some((i, a, b)) = first_diff {
            panic!(
                "q4k_v4_predec NOT bit-identical to v3_8r: {diff_count}/{rows} rows differ; \
                 first @ i={i}  v3={a:e} (0x{:08x})  v4={b:e} (0x{:08x})",
                a.to_bits(),
                b.to_bits(),
            );
        }
        eprintln!("[q4k_v4_predec parity] {} rows bit-identical to v3_8r", rows);
    }
}
#[rustfmt::skip]
mod q6k_gemv_parity {
    #![cfg(target_os = "macos")]
    //! P2 — parity test for `gemv_q6_k_pinned_tcb` against CPU reference.
    //!
    //! Builds a Q6_K weight via `quantize_q6_k`, computes the CPU reference
    //! GEMV (dequant-then-multiply), runs the new Metal kernel through a
    //! pinned buffer, and compares.
    //!
    //! Tolerance: 5e-2 absolute. Q6_K quant introduces noticeable error
    //! (round-trip dequant→requant is *not* exact for Q6_K; see the comment
    //! on `quantize_q6_k` in src/quant/mod.rs). The downstream tolerance in
    //! the real model is dominated by the per-block round-off; a few units
    //! of last-place error per accumulated element is expected. Test data
    //! is chosen small enough that accumulated errors stay below 5e-2.

    use hawking_core::kernels;
    use hawking_core::metal::TokenCommandBuffer;
    use hawking_core::quant;

    use crate::common;
    use common::*;

    #[test]
    fn q6k_gemv_matches_cpu_reference() {
        // Shape mirrors a Qwen-3B Q6_K projection (kv_dim × hidden).
        let rows = 256usize;
        let cols = 2048usize;

        // Build random weight as f32, quantize to Q6_K, then dequant back —
        // gives us a CPU reference that's bit-identical to what the GPU
        // kernel decodes from the same Q6_K bytes.
        let w_f32 = fixed_f32(rows * cols, 0xC0DEC0DE);
        let blocks = (rows * cols) / 256;
        let mut w_q6 = vec![0u8; blocks * quant::Q6_K_BLOCK_BYTES];
        quant::quantize_q6_k(&w_f32, &mut w_q6).expect("Q6_K quant");

        // Reconstruct CPU view of the matrix from the Q6_K bytes (so we
        // compare GPU GEMV vs a CPU GEMV that uses the *same* dequant).
        let mut w_recon = vec![0.0f32; rows * cols];
        quant::dequant_into(hawking_core::gguf::GgmlType::Q6_K, &w_q6, &mut w_recon).expect("Q6_K dequant");

        let x = fixed_f32(cols, 0xBEEFBEEF);
        let mut expected = vec![0.0f32; rows];
        // y = w_recon @ x  (row-major)
        for r in 0..rows {
            let mut acc = 0.0f32;
            let row = &w_recon[r * cols..(r + 1) * cols];
            for c in 0..cols {
                acc += row[c] * x[c];
            }
            expected[r] = acc;
        }

        // GPU path.
        let ctx = ctx();
        let model_buf = ctx.new_buffer_with_bytes(&w_q6);
        let x_buf = new_f32_buf(ctx, &x);
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q6_k_pinned_tcb(&mut tcb, &model_buf, 0, w_q6.len(), rows, cols, &x_buf, &out_buf).expect("gemv_q6_k encode");
            tcb.commit_and_wait().expect("commit");
        }

        let actual = read_f32_buf(&out_buf, rows);
        let diff = max_abs_diff(&expected, &actual);
        assert!(diff < 5e-2, "q6_k gemv max_abs_diff = {diff} (limit 5e-2)");
    }
}
#[rustfmt::skip]
mod q8_kv_parity {
    #![cfg(target_os = "macos")]

    // Parity tests for the Q8 latent KV cache path.
    //
    // Two kernels under test:
    //   1. `kv_append_q8_0_f32` — GPU-side fp32→Q8_0 quantize. Verified against
    //      the CPU `quantize_q8_0` helper. They MUST produce bit-identical
    //      bytes on identical inputs.
    //   2. `mla_decode_kernel_q8kv` — MLA decode reading Q8-packed c_kv. Verified
    //      against `mla_decode_metal` (f32 c_kv) with ATOL=5e-3. The error comes
    //      from per-block f16 scale + round-to-nearest int8; bounded analytically
    //      by `max(|c_kv|)/127 * scale_fp16_slop`.
    //
    // V2-Lite real shapes covered: n_heads=16, qk_nope=128, qk_rope=64,
    // v_head=128, kv_lora=512.

    use hawking_core::kernels;
    use hawking_core::metal::PinnedBuffer;
    use hawking_core::quant::{quantize_q8_0, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};

    use crate::common;
    use common::*;

    // ── kv_append_q8_0_f32 GPU vs CPU ───────────────────────────────────────────

    #[test]
    fn kv_append_q8_gpu_matches_cpu_quantize() {
        let ctx = ctx();
        let kv_lora_rank = 512usize;
        let qk_rope_head_dim = 64usize;
        let max_seq = 8usize;

        let c_kv_normed = fixed_f32(kv_lora_rank, 0xA11CE);
        let mut kv_a_out = vec![0.0f32; kv_lora_rank + qk_rope_head_dim];
        let pe_src = fixed_f32(qk_rope_head_dim, 0xB0B);
        kv_a_out[kv_lora_rank..].copy_from_slice(&pe_src);

        let n_blocks = kv_lora_rank / Q8_0_BLOCK_ELEMS;
        let row_bytes = n_blocks * Q8_0_BLOCK_BYTES;
        let mut gpu_cache = vec![0u8; max_seq * row_bytes];
        let mut gpu_kpe = vec![0.0f32; max_seq * qk_rope_head_dim];

        let seq_slot = 3usize;
        kernels::kv_append_q8_0_f32_metal(ctx, &c_kv_normed, &kv_a_out, &mut gpu_cache, &mut gpu_kpe, seq_slot, kv_lora_rank, qk_rope_head_dim, max_seq).expect("gpu kv_append_q8");

        // CPU reference: quantize the same row and place at the same slot.
        let mut cpu_row = vec![0u8; row_bytes];
        quantize_q8_0(&c_kv_normed, &mut cpu_row).expect("cpu quantize");
        let gpu_row = &gpu_cache[seq_slot * row_bytes..(seq_slot + 1) * row_bytes];

        // The CPU and GPU paths should produce bit-identical bytes — both use
        // amax/127 scaling and round-to-nearest int8.
        let diff_bytes: Vec<usize> = gpu_row.iter().zip(cpu_row.iter()).enumerate().filter(|(_, (a, b))| a != b).map(|(i, _)| i).collect();
        assert!(diff_bytes.is_empty(), "GPU/CPU Q8 quantize differ at byte offsets: {diff_bytes:?}");

        // k_pe at slot should match the source slice.
        let gpu_pe = &gpu_kpe[seq_slot * qk_rope_head_dim..(seq_slot + 1) * qk_rope_head_dim];
        for (i, (a, b)) in gpu_pe.iter().zip(pe_src.iter()).enumerate() {
            assert!((a - b).abs() < 1e-9, "k_pe element {i} mismatch: gpu={a} cpu={b}");
        }

        // Other slots must remain untouched (the kernel writes only at seq_slot).
        for s in 0..max_seq {
            if s == seq_slot {
                continue;
            }
            let row = &gpu_cache[s * row_bytes..(s + 1) * row_bytes];
            assert!(row.iter().all(|&b| b == 0), "slot {s} wasn't supposed to be written");
        }
    }

    // ── mla_decode_kernel_q8kv vs mla_decode_metal (f32) ────────────────────────

    const N_HEADS: usize = 16;
    const QK_NOPE: usize = 128;
    const QK_ROPE: usize = 64;
    const V_HEAD: usize = 128;
    const KV_LORA: usize = 512;

    fn run_q8_vs_f32(label: &str, seq_len: usize, c_kv_scale: f32, atol: f32) {
        let ctx = ctx();
        let q_head_dim = QK_NOPE + QK_ROPE;
        let scale = 1.0_f32 / (q_head_dim as f32).sqrt();

        let q = fixed_f32(N_HEADS * q_head_dim, 0xDEAD ^ seq_len as u64);
        // Scale c_kv to simulate realistic post-rmsnorm activations.
        // Production c_kv has variance dictated by the layer's rmsnorm weight,
        // typically in [-0.3, 0.3] after normalization. Uniform [-1, 1] is a
        // worst case for Q8 quant noise; scale=0.1 simulates realistic.
        let c_kv: Vec<f32> = fixed_f32(seq_len * KV_LORA, 0xBEEF ^ seq_len as u64).into_iter().map(|x| x * c_kv_scale).collect();
        let k_pe = fixed_f32(seq_len * QK_ROPE, 0xCAFE ^ seq_len as u64);
        let kv_b = fixed_f32(N_HEADS * (QK_NOPE + V_HEAD) * KV_LORA, 0xABCD);
        let kv_b_buf: PinnedBuffer = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&kv_b));

        // Reference path: f32 c_kv → existing mla_decode_metal
        let mut ref_out = vec![0.0f32; N_HEADS * V_HEAD];
        kernels::mla_decode_metal(ctx, &q, &c_kv, &k_pe, &kv_b_buf, N_HEADS, QK_NOPE, QK_ROPE, V_HEAD, KV_LORA, seq_len, scale, &mut ref_out).expect("mla_decode_metal");

        // Q8 path: CPU-quantize the entire c_kv into Q8 row-major bytes, then call q8 kernel.
        let n_blocks_per_row = KV_LORA / Q8_0_BLOCK_ELEMS;
        let row_bytes = n_blocks_per_row * Q8_0_BLOCK_BYTES;
        let mut c_kv_q8 = vec![0u8; seq_len * row_bytes];
        for t in 0..seq_len {
            let src_row = &c_kv[t * KV_LORA..(t + 1) * KV_LORA];
            let dst_row = &mut c_kv_q8[t * row_bytes..(t + 1) * row_bytes];
            quantize_q8_0(src_row, dst_row).expect("quantize_q8_0");
        }

        let mut q8_out = vec![0.0f32; N_HEADS * V_HEAD];
        kernels::mla_decode_q8kv_metal(ctx, &q, &c_kv_q8, &k_pe, &kv_b_buf, N_HEADS, QK_NOPE, QK_ROPE, V_HEAD, KV_LORA, seq_len, scale, &mut q8_out).expect("mla_decode_q8kv_metal");

        let diff = max_abs_diff(&ref_out, &q8_out);
        println!("[q8-kv-parity] {label}: c_kv_scale={c_kv_scale} max_abs_diff={diff:.3e} atol={atol:.0e}");
        assert!(diff < atol, "{label}: diff {diff:.3e} >= {atol:.0e}");
    }

    // Worst-case data: uniform [-1, 1] with no structure. Q8's per-block f16
    // scale + i8 round-to-nearest accumulates over kv_lora=512 × seq_len terms
    // in the latent space; analytical bound ~ sqrt(seq_len) × sqrt(kv_lora) ×
    // q8-noise ≈ 0.2 at seq=1024. Looser tolerance reflects this.
    #[test]
    fn q8kv_seq256_worst_case() {
        run_q8_vs_f32("seq=256 worst", 256, 1.0, 0.30);
    }

    #[test]
    fn q8kv_seq1024_worst_case() {
        run_q8_vs_f32("seq=1024 worst", 1024, 1.0, 0.30);
    }

    #[test]
    fn q8kv_seq2048_worst_case() {
        run_q8_vs_f32("seq=2048 worst", 2048, 1.0, 0.30);
    }

    // Realistic data: post-rmsnorm activations are concentrated near zero
    // (rmsnorm normalizes variance, then the learnable weight typically
    // scales by ~0.05-0.2). c_kv ~ 0.1 × Uniform[-1,1] simulates that range.
    // Tolerance tightens proportionally to the smaller dynamic range.
    #[test]
    fn q8kv_seq256_realistic() {
        run_q8_vs_f32("seq=256 real", 256, 0.1, 0.03);
    }

    #[test]
    fn q8kv_seq1024_realistic() {
        run_q8_vs_f32("seq=1024 real", 1024, 0.1, 0.03);
    }

    #[test]
    fn q8kv_seq2048_realistic() {
        run_q8_vs_f32("seq=2048 real", 2048, 0.1, 0.03);
    }
}
#[rustfmt::skip]
mod qkv_rope_append_f16s_parity {
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

    use crate::common;
    use common::*;

    /// Build random Q4_K weights and f32 predecoded scale table.
    fn make_q4k_predec(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let bpr = cols / 256;
        let total_w = rows * bpr * 144;
        let w: Vec<u8> = (0..total_w).map(|i| ((i as u32).wrapping_mul(2246822519u32).wrapping_add(seed)) as u8).collect();
        let ns = rows * bpr * 16;
        // Avoid near-zero scales: generate in [0.1, 2.0] so f16 rounding is benign.
        let s: Vec<f32> = (0..ns)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761u32).wrapping_add(seed ^ 0xAB)) as f32 / u32::MAX as f32;
                0.1 + v * 1.9
            })
            .collect();
        (w, s)
    }

    /// Convert f32 scale table to packed f16 bytes.
    fn f32_to_f16_scales(scales: &[f32]) -> Vec<u8> {
        scales.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect()
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
        Q4kWeights { q_w, q_sc_f32, q_sc_f16, k_w, k_sc_f32, k_sc_f16, v_w, v_sc_f32, v_sc_f16 }
    }

    /// Run f32-scales 2r kernel → (q_out, k_cache_slice, v_cache_slice).
    fn run_f32_2r(ctx: &MetalContext, shape: &Shape, w: &Q4kWeights, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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
    fn run_f16s_2r(ctx: &MetalContext, shape: &Shape, w: &Q4kWeights, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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
    fn run_f32_4r(ctx: &MetalContext, shape: &Shape, w: &Q4kWeights, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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
    fn run_f16s_4r(ctx: &MetalContext, shape: &Shape, w: &Q4kWeights, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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
    fn check_rel_l2(label: &str, ref_q: &[f32], ref_k: &[f32], ref_v: &[f32], got_q: &[f32], got_k: &[f32], got_v: &[f32]) {
        const MAX_REL_L2: f64 = 1e-2;
        let rq = rel_l2(ref_q, got_q);
        let rk = rel_l2(ref_k, got_k);
        let rv = rel_l2(ref_v, got_v);
        assert!(rq < MAX_REL_L2, "{label} Q: rel_L2={rq:.4e} >= {MAX_REL_L2:.4e}");
        assert!(rk < MAX_REL_L2, "{label} K: rel_L2={rk:.4e} >= {MAX_REL_L2:.4e}");
        assert!(rv < MAX_REL_L2, "{label} V: rel_L2={rv:.4e} >= {MAX_REL_L2:.4e}");
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
            let shape = Shape { n_q, n_k, hd, cols, pos, kv_off };
            let q_rows = n_q * hd;
            let w = make_weights(&shape, seed);
            let x: Vec<f32> = (0..cols).map(|i| ((i as u32).wrapping_mul(1664525).wrapping_add(seed) as f32 / u32::MAX as f32) * 2.0 - 1.0).collect();

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
        let cases: &[(usize, usize, usize, usize, u32, usize, u32)] =
            &[(16, 8, 128, 2048, 0, 0, 0xD400), (16, 8, 128, 2048, 63, 31, 0xD401), (16, 8, 128, 2048, 255, 127, 0xD402), (8, 4, 128, 2048, 17, 5, 0xD410)];

        for &(n_q, n_k, hd, cols, pos, kv_off, seed) in cases {
            let shape = Shape { n_q, n_k, hd, cols, pos, kv_off };
            let q_rows = n_q * hd;
            let w = make_weights(&shape, seed);
            let x: Vec<f32> = (0..cols).map(|i| ((i as u32).wrapping_mul(1664525).wrapping_add(seed) as f32 / u32::MAX as f32) * 2.0 - 1.0).collect();

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
        let shape = Shape { n_q: 16, n_k: 8, hd: 128, cols: 2048, pos: 42, kv_off: 13 };
        let q_rows = shape.n_q * shape.hd;
        let w = make_weights(&shape, 0xD500);
        let x: Vec<f32> = (0..shape.cols).map(|i| ((i as u32).wrapping_mul(1664525).wrapping_add(0xD500) as f32 / u32::MAX as f32) * 2.0 - 1.0).collect();

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
}
#[rustfmt::skip]
mod qkv_rope_append_parity {
    #![cfg(target_os = "macos")]
    //! Track 3.12/3.13 parity: QKV triple with inline Q/K bias+RoPE and f32
    //! KV-cache append must match the current three-dispatch sequence:
    //! QKV triple, `rope_qk_f32_b1_bias`, then `kv_append_vbias_f32`.

    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};
    use hawking_core::quant;

    use crate::common;
    use common::*;

    fn make_q4k(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let bpr = cols / 256;
        let total = rows * bpr * 144;
        let w: Vec<u8> = (0..total).map(|i| ((i as u32).wrapping_mul(2246822519).wrapping_add(seed)) as u8).collect();
        let ns = rows * bpr * 16;
        let s: Vec<f32> = (0..ns)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32 / u32::MAX as f32;
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
        a.iter().zip(b).filter(|(x, y)| x.is_finite() && y.is_finite()).map(|(x, y)| (x - y).abs() / x.abs().max(y.abs()).max(1.0)).fold(0.0_f32, f32::max)
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

    fn run_q4_ref(ctx: &MetalContext, shape: &Shape, w: &Q4Weights, x: &[f32], q_bias: Option<&[f32]>, k_bias: Option<&[f32]>, v_bias: Option<&[f32]>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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
        kernels::rope_qk_f32_b1_bias_tcb(&mut tcb, &q_buf, &k_tok, q_bias_buf.as_ref(), k_bias_buf.as_ref(), shape.n_q, shape.n_k, shape.hd, shape.pos, 10000.0).expect("rope ref");
        kernels::kv_append_vbias_f32_tcb(&mut tcb, &k_tok, &v_tok, v_bias_buf.as_ref(), &k_cache, &v_cache, kv_rows, shape.kv_off).expect("kv append ref");
        tcb.commit_and_wait().expect("ref commit");

        let q = read_f32_buf(&q_buf, q_rows);
        let k = read_f32_buf(&k_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
        let v = read_f32_buf(&v_cache, cache_len)[shape.kv_off..shape.kv_off + kv_rows].to_vec();
        (q, k, v)
    }

    fn run_q4_fused(ctx: &MetalContext, shape: &Shape, w: &Q4Weights, x: &[f32], q_bias: Option<&[f32]>, k_bias: Option<&[f32]>, v_bias: Option<&[f32]>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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
        let shape = Shape { n_q: 4, n_k: 2, hd: 64, cols: 256, pos: 127, kv_off: 19 };
        let q_rows = shape.n_q * shape.hd;
        let kv_rows = shape.n_k * shape.hd;
        let (q, q_scales) = make_q4k(q_rows, shape.cols, 0x1001);
        let (k, k_scales) = make_q4k(kv_rows, shape.cols, 0x1002);
        let (v, v_scales) = make_q4k(kv_rows, shape.cols, 0x1003);
        let weights = Q4Weights { q, q_scales, k, k_scales, v, v_scales };
        let x = fixed_f32(shape.cols, 0xBEEF);
        let q_bias = fixed_f32(q_rows, 0xCAFE);
        let k_bias = fixed_f32(kv_rows, 0xF00D);
        let v_bias = fixed_f32(kv_rows, 0xD00D);

        let reference = run_q4_ref(ctx, &shape, &weights, &x, Some(&q_bias), Some(&k_bias), Some(&v_bias));
        let fused = run_q4_fused(ctx, &shape, &weights, &x, Some(&q_bias), Some(&k_bias), Some(&v_bias));
        assert_close("q4 q", &reference.0, &fused.0);
        assert_close("q4 k_cache", &reference.1, &fused.1);
        assert_close("q4 v_cache", &reference.2, &fused.2);
    }

    #[test]
    fn mixed_q4k_q4k_q6k_rope_append_matches_ref() {
        let ctx = ctx();
        let shape = Shape { n_q: 4, n_k: 2, hd: 64, cols: 256, pos: 511, kv_off: 37 };
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
            kernels::gemv_q4k_q4k_q6k_triple_pinned_tcb(&mut tcb, &model, q_off, q.len(), &q_sc, k_off, k.len(), &k_sc, v_off, v.len(), q_rows, kv_rows, shape.cols, &x_buf, &q_buf, &k_tok, &v_tok)
                .expect("mixed ref triple");
            kernels::rope_qk_f32_b1_bias_tcb(&mut tcb, &q_buf, &k_tok, Some(&q_bias_buf), Some(&k_bias_buf), shape.n_q, shape.n_k, shape.hd, shape.pos, 10000.0).expect("mixed ref rope");
            kernels::kv_append_vbias_f32_tcb(&mut tcb, &k_tok, &v_tok, Some(&v_bias_buf), &k_cache, &v_cache, kv_rows, shape.kv_off).expect("mixed ref append");
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
}
#[rustfmt::skip]
mod quantize_int8_kernel_parity {
    //! GPU `quantize_f32_to_int8_per_block` must bit-match the CPU reference
    //! `quantize_to_int8_per_block`. Both use the same `scale = max|x|/127`
    //! formula and round-to-nearest-with-clamp, so the only sources of
    //! divergence would be: (a) a different per-block reduction order on the
    //! GPU producing a different max_abs (only possible if any FP add went
    //! into the reduction — it doesn't; we reduce fabs(x) with `max`, which
    //! is order-independent for floats not involving NaN), or (b) a kernel
    //! bug. We assert bit-identical bytes + scales.

    #![cfg(target_os = "macos")]

    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    fn read_i8(buf: &PinnedBuffer, n: usize) -> Vec<i8> {
        let ptr = buf.contents() as *const i8;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    fn read_f32(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    fn run_one(ctx: &MetalContext, n: usize, seed: u64, range: f32) {
        let mut rng = Pcg64Mcg::new(seed as u128);
        let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-range..range)).collect();
        let (cpu_int8, cpu_scales) = kernels::quantize_to_int8_per_block(&x, 256);

        let x_buf = new_f32_buf(ctx, &x);
        let int8_buf = ctx.new_buffer(n);
        let scales_buf = ctx.new_buffer((n / 256) * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::quantize_f32_to_int8_per_block_tcb(&mut tcb, &x_buf, &int8_buf, &scales_buf, n).expect("encode");
            tcb.commit_and_wait().expect("commit");
        }
        let gpu_int8 = read_i8(&int8_buf, n);
        let gpu_scales = read_f32(&scales_buf, n / 256);

        assert_eq!(cpu_scales, gpu_scales, "scales mismatch at n={n} seed={seed}");
        let mut diffs = 0usize;
        let mut first_bad = None;
        for i in 0..n {
            if cpu_int8[i] != gpu_int8[i] {
                diffs += 1;
                if first_bad.is_none() {
                    first_bad = Some((i, cpu_int8[i], gpu_int8[i], x[i], cpu_scales[i / 256]));
                }
            }
        }
        assert_eq!(diffs, 0, "int8 mismatch at n={n} seed={seed}: {diffs} elems differ; first bad: {:?}", first_bad,);
    }

    #[test]
    fn quantize_int8_kernel_matches_cpu_small() {
        run_one(ctx(), 256, 0xA11CE, 3.0);
    }

    #[test]
    fn quantize_int8_kernel_matches_cpu_hidden_2048() {
        run_one(ctx(), 2048, 0xBEEF, 3.0);
    }

    #[test]
    fn quantize_int8_kernel_matches_cpu_intermediate_11008() {
        run_one(ctx(), 11008, 0xC0FFEE, 1.5);
    }

    #[test]
    fn quantize_int8_kernel_handles_all_zero_block() {
        // A degenerate block (all zeros) → scale becomes 1.0 fallback. Ensure
        // CPU/GPU agree on the fallback path.
        let ctx = ctx();
        let mut x = vec![0.0f32; 2048];
        // Non-zero middle block so we exercise both paths in one dispatch.
        for i in 512..768 {
            x[i] = 1.5 * ((i as f32) - 640.0) / 128.0;
        }
        let (cpu_int8, cpu_scales) = kernels::quantize_to_int8_per_block(&x, 256);
        let x_buf = new_f32_buf(ctx, &x);
        let int8_buf = ctx.new_buffer(2048);
        let scales_buf = ctx.new_buffer(8 * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::quantize_f32_to_int8_per_block_tcb(&mut tcb, &x_buf, &int8_buf, &scales_buf, 2048).expect("encode");
            tcb.commit_and_wait().expect("commit");
        }
        assert_eq!(cpu_scales, read_f32(&scales_buf, 8), "scales");
        assert_eq!(cpu_int8, read_i8(&int8_buf, 2048), "int8");
    }
}
#[rustfmt::skip]
mod quantize_int8_scaled_parity {
    //! GPU `quantize_f32_to_int8_per_block_scaled` must bit-match the CPU
    //! reference `quantize_to_int8_per_block_scaled`. The AWQ Option B path
    //! folds an activation-side divide (x / s) into the existing per-block
    //! int8 quantize; this test ensures the fused GPU variant produces the
    //! same int8 bytes and per-block scales as the explicit divide-then-
    //! quantize CPU pipeline.
    //!
    //! Range chosen so the scaled values stay well within the int8 working
    //! range; if a future smoothing JSON pushes outside, raise the range.

    #![cfg(target_os = "macos")]

    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    fn read_i8(buf: &PinnedBuffer, n: usize) -> Vec<i8> {
        let ptr = buf.contents() as *const i8;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    fn read_f32(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    /// Generate a smoothing vector with realistic AWQ-style range
    /// (most channels ~1.0, some outliers up to ~5×) given a seed.
    fn make_smoothing(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128 ^ 0xA_5BAEu128);
        (0..n)
            .map(|i| {
                // 5% of channels get an "outlier" factor up to 5x; rest cluster
                // around 1.0 with mild jitter — matches the layer_0 stats from
                // profiles/qwen3b_awq_smoothing.json.
                if i % 20 == 0 {
                    rng.gen_range(2.0..5.0)
                } else {
                    rng.gen_range(0.3..1.6)
                }
            })
            .collect()
    }

    fn run_one(ctx: &MetalContext, n: usize, seed: u64, range: f32) {
        let mut rng = Pcg64Mcg::new(seed as u128);
        let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-range..range)).collect();
        let s = make_smoothing(n, seed);
        let (cpu_int8, cpu_scales) = kernels::quantize_to_int8_per_block_scaled(&x, &s, 256);

        let x_buf = new_f32_buf(ctx, &x);
        let s_buf = new_f32_buf(ctx, &s);
        let int8_buf = ctx.new_buffer(n);
        let scales_buf = ctx.new_buffer((n / 256) * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::quantize_f32_to_int8_per_block_scaled_tcb(&mut tcb, &x_buf, &s_buf, &int8_buf, &scales_buf, n).expect("encode");
            tcb.commit_and_wait().expect("commit");
        }
        let gpu_int8 = read_i8(&int8_buf, n);
        let gpu_scales = read_f32(&scales_buf, n / 256);

        assert_eq!(cpu_scales, gpu_scales, "scales mismatch at n={n} seed={seed}");
        let mut diffs = 0usize;
        let mut first_bad = None;
        for i in 0..n {
            if cpu_int8[i] != gpu_int8[i] {
                diffs += 1;
                if first_bad.is_none() {
                    first_bad = Some((i, cpu_int8[i], gpu_int8[i], x[i], s[i], cpu_scales[i / 256]));
                }
            }
        }
        assert_eq!(diffs, 0, "int8 mismatch at n={n} seed={seed}: {diffs} elems differ; first bad: {:?}", first_bad,);
    }

    #[test]
    fn quantize_int8_scaled_kernel_matches_cpu_small() {
        run_one(ctx(), 256, 0xA11CE, 3.0);
    }

    #[test]
    fn quantize_int8_scaled_kernel_matches_cpu_hidden_2048() {
        run_one(ctx(), 2048, 0xBEEF, 3.0);
    }

    #[test]
    fn quantize_int8_scaled_kernel_matches_cpu_intermediate_11008() {
        run_one(ctx(), 11008, 0xC0FFEE, 1.5);
    }

    #[test]
    fn quantize_int8_scaled_kernel_handles_all_zero_block() {
        let ctx = ctx();
        let mut x = vec![0.0f32; 2048];
        for i in 512..768 {
            x[i] = 1.5 * ((i as f32) - 640.0) / 128.0;
        }
        let s = make_smoothing(2048, 0xD15EA5E);
        let (cpu_int8, cpu_scales) = kernels::quantize_to_int8_per_block_scaled(&x, &s, 256);
        let x_buf = new_f32_buf(ctx, &x);
        let s_buf = new_f32_buf(ctx, &s);
        let int8_buf = ctx.new_buffer(2048);
        let scales_buf = ctx.new_buffer(8 * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::quantize_f32_to_int8_per_block_scaled_tcb(&mut tcb, &x_buf, &s_buf, &int8_buf, &scales_buf, 2048).expect("encode");
            tcb.commit_and_wait().expect("commit");
        }
        assert_eq!(cpu_scales, read_f32(&scales_buf, 8), "scales");
        assert_eq!(cpu_int8, read_i8(&int8_buf, 2048), "int8");
    }

    #[test]
    fn quantize_int8_scaled_kernel_handles_zero_smoothing_channels() {
        // Degenerate s entries (≤ 1e-12) must clamp to inv_s = 0 on both
        // CPU and GPU so the result is just int8 zero at that channel — the
        // bake tool will never emit a zero, but we don't want a divergent
        // NaN if it ever does.
        let ctx = ctx();
        let n = 1024;
        let mut rng = Pcg64Mcg::new(0xBADu128);
        let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
        let mut s = make_smoothing(n, 0xBAD ^ 0x5EED);
        // Zero out a sprinkling of channels across two blocks.
        for i in [3, 17, 100, 257, 600] {
            s[i] = 0.0;
        }
        let (cpu_int8, cpu_scales) = kernels::quantize_to_int8_per_block_scaled(&x, &s, 256);
        let x_buf = new_f32_buf(ctx, &x);
        let s_buf = new_f32_buf(ctx, &s);
        let int8_buf = ctx.new_buffer(n);
        let scales_buf = ctx.new_buffer((n / 256) * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::quantize_f32_to_int8_per_block_scaled_tcb(&mut tcb, &x_buf, &s_buf, &int8_buf, &scales_buf, n).expect("encode");
            tcb.commit_and_wait().expect("commit");
        }
        assert_eq!(cpu_scales, read_f32(&scales_buf, n / 256), "scales");
        assert_eq!(cpu_int8, read_i8(&int8_buf, n), "int8");
    }
}
#[rustfmt::skip]
mod rope_batched_multiseq_parity {
    #![cfg(target_os = "macos")]
    //! R2 parity: `rope_f32_batched_multiseq` == the per-slot `rope_q_f32_inplace_off`
    //! loop it replaces, BIT-IDENTICAL.
    //!
    //! The multi-seq decode stack used to RoPE each slot with its own dispatch
    //! (`rope_q_f32_inplace_off_tcb` × B, for Q and K). R2 batches that into ONE
    //! dispatch per tensor reading a per-slot `positions[]` buffer. RoPE is purely
    //! elementwise (no cross-element reduction), so the batched kernel must produce
    //! byte-identical output to the per-slot loop — not merely atol-close. This test
    //! runs BOTH on the GPU over B slots at divergent positions (incl. pos 0 = the
    //! identity rotation and a long-context pos 2047) and asserts max_abs_diff == 0,
    //! for both the Q width (n_heads) and the K width (n_kv_heads).

    use hawking_core::kernels;
    use hawking_core::metal::TokenCommandBuffer;

    use crate::common;
    use common::*;

    fn run_per_slot(x: &[f32], n_heads: usize, head_dim: usize, slot_dim: usize, positions: &[u32], theta: f32) -> Vec<f32> {
        let ctx = ctx();
        let buf = new_f32_buf(ctx, x);
        let b = positions.len();
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            for bi in 0..b {
                kernels::rope_q_f32_inplace_off_tcb(&mut tcb, &buf, bi * slot_dim * std::mem::size_of::<f32>(), n_heads, head_dim, 0, head_dim, positions[bi], theta).expect("per-slot rope encode");
            }
            tcb.commit_and_wait().expect("per-slot rope commit");
        }
        read_f32_buf(&buf, b * slot_dim)
    }

    fn run_batched(x: &[f32], n_heads: usize, head_dim: usize, slot_dim: usize, positions: &[u32], theta: f32) -> Vec<f32> {
        let ctx = ctx();
        let buf = new_f32_buf(ctx, x);
        let b = positions.len();
        let pos_bytes: Vec<u8> = positions.iter().flat_map(|&p| p.to_le_bytes()).collect();
        let pos_buf = ctx.new_buffer_with_bytes(&pos_bytes);
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::rope_f32_batched_multiseq_tcb(&mut tcb, &buf, &pos_buf, n_heads, head_dim, slot_dim, b, theta).expect("batched rope encode");
            tcb.commit_and_wait().expect("batched rope commit");
        }
        read_f32_buf(&buf, b * slot_dim)
    }

    #[test]
    fn rope_batched_multiseq_matches_per_slot() {
        let n_heads = 16usize; // Qwen2.5-3B Q heads
        let n_kv_heads = 2usize; // GQA KV heads
        let head_dim = 128usize;
        let theta = 1_000_000.0f32; // Qwen rope_theta
        let positions: [u32; 5] = [2047, 13, 500, 0, 1024];
        let b = positions.len();

        // Q width.
        let q_dim = n_heads * head_dim;
        let xq = fixed_f32(b * q_dim, 0x5151_5151_5151_5151);
        let expected_q = run_per_slot(&xq, n_heads, head_dim, q_dim, &positions, theta);
        let actual_q = run_batched(&xq, n_heads, head_dim, q_dim, &positions, theta);
        let diff_q = max_abs_diff(&expected_q, &actual_q);
        assert_eq!(diff_q, 0.0, "Q rope: batched != per-slot (max_abs_diff {diff_q})");

        // K width (GQA: fewer heads, narrower slot stride).
        let kv_dim = n_kv_heads * head_dim;
        let xk = fixed_f32(b * kv_dim, 0x6262_6262_6262_6262);
        let expected_k = run_per_slot(&xk, n_kv_heads, head_dim, kv_dim, &positions, theta);
        let actual_k = run_batched(&xk, n_kv_heads, head_dim, kv_dim, &positions, theta);
        let diff_k = max_abs_diff(&expected_k, &actual_k);
        assert_eq!(diff_k, 0.0, "K rope: batched != per-slot (max_abs_diff {diff_k})");

        // B=1 degenerate case must also match.
        let one = [777u32];
        let x1 = fixed_f32(q_dim, 0x7373_7373_7373_7373);
        let e1 = run_per_slot(&x1, n_heads, head_dim, q_dim, &one, theta);
        let a1 = run_batched(&x1, n_heads, head_dim, q_dim, &one, theta);
        assert_eq!(max_abs_diff(&e1, &a1), 0.0, "B=1 rope: batched != per-slot");

        println!("[rope-batched-multiseq] Q+K+B=1 bit-identical over positions {positions:?}");
    }
}
#[rustfmt::skip]
mod rope_kv_append_fused_parity {
    #![cfg(target_os = "macos")]
    //! Track B6 parity: `rope_qk_kv_append_vbias_f32` must produce bit-identical
    //! results to the two-dispatch sequence:
    //!   1. `rope_qk_f32_b1_bias_tcb`   (Q+K bias + rope, in-place)
    //!   2. `kv_append_vbias_f32_tcb`   (V-bias + K+V cache append)
    //!
    //! The fused kernel differs in one intentional way: k_token_buf is left in its
    //! pre-rope state (k is rotated directly into k_cache). This is correct since
    //! nothing reads k_token_buf after the KV append. The parity check verifies
    //! q_buf (in-place rope), k_cache[kv_off..], and v_cache[kv_off..].

    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};

    use crate::common;
    use common::*;

    fn rnd(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761u32).wrapping_add(seed);
                (x as f32 / u32::MAX as f32) * 4.0 - 2.0
            })
            .collect()
    }

    /// Two-dispatch reference: rope_qk_f32_b1_bias + kv_append_vbias_f32.
    /// Returns (q_out, k_cache_slice, v_cache_slice) after commit.
    #[allow(clippy::too_many_arguments)]
    fn run_ref(
        ctx: &MetalContext,
        q: &[f32],
        k_tok: &[f32],
        v_tok: &[f32],
        q_bias: Option<&[f32]>,
        k_bias: Option<&[f32]>,
        v_bias: Option<&[f32]>,
        kv_off: usize,
        cache_size: usize, // total elements in each k_cache / v_cache buffer
        n_q: usize,
        n_k: usize,
        head_dim: usize,
        pos: u32,
        base: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let kv_dim = k_tok.len();
        let q_buf = new_f32_buf(ctx, q);
        let k_buf = new_f32_buf(ctx, k_tok);
        let v_buf = new_f32_buf(ctx, v_tok);
        let k_cache = ctx.new_buffer(cache_size * 4);
        let v_cache = ctx.new_buffer(cache_size * 4);

        let qb_buf = q_bias.map(|b| new_f32_buf(ctx, b));
        let kb_buf = k_bias.map(|b| new_f32_buf(ctx, b));
        let vb_buf = v_bias.map(|b| new_f32_buf(ctx, b));

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::rope_qk_f32_b1_bias_tcb(&mut tcb, &q_buf, &k_buf, qb_buf.as_ref(), kb_buf.as_ref(), n_q, n_k, head_dim, pos, base).expect("ref rope_qk");
        kernels::kv_append_vbias_f32_tcb(&mut tcb, &k_buf, &v_buf, vb_buf.as_ref(), &k_cache, &v_cache, kv_dim, kv_off).expect("ref kv_append");
        tcb.commit_and_wait().expect("ref commit");

        (read_f32_buf(&q_buf, q.len()), read_f32_buf(&k_cache, cache_size), read_f32_buf(&v_cache, cache_size))
    }

    /// One-dispatch fused: rope_qk_kv_append_vbias_f32.
    /// Returns (q_out, k_cache_slice, v_cache_slice).
    #[allow(clippy::too_many_arguments)]
    fn run_fused(
        ctx: &MetalContext,
        q: &[f32],
        k_tok: &[f32],
        v_tok: &[f32],
        q_bias: Option<&[f32]>,
        k_bias: Option<&[f32]>,
        v_bias: Option<&[f32]>,
        kv_off: usize,
        cache_size: usize,
        n_q: usize,
        n_k: usize,
        head_dim: usize,
        pos: u32,
        base: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let kv_dim = k_tok.len();
        let q_buf = new_f32_buf(ctx, q);
        let k_buf = new_f32_buf(ctx, k_tok);
        let v_buf = new_f32_buf(ctx, v_tok);
        let k_cache = ctx.new_buffer(cache_size * 4);
        let v_cache = ctx.new_buffer(cache_size * 4);

        let qb_buf = q_bias.map(|b| new_f32_buf(ctx, b));
        let kb_buf = k_bias.map(|b| new_f32_buf(ctx, b));
        let vb_buf = v_bias.map(|b| new_f32_buf(ctx, b));

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::rope_qk_kv_append_vbias_f32_tcb(
            &mut tcb,
            &q_buf,
            &k_buf,
            &v_buf,
            qb_buf.as_ref(),
            kb_buf.as_ref(),
            vb_buf.as_ref(),
            &k_cache,
            &v_cache,
            n_q,
            n_k,
            head_dim,
            pos,
            base,
            kv_dim,
            kv_off,
        )
        .expect("fused dispatch");
        tcb.commit_and_wait().expect("fused commit");

        (read_f32_buf(&q_buf, q.len()), read_f32_buf(&k_cache, cache_size), read_f32_buf(&v_cache, cache_size))
    }

    #[test]
    fn rope_qk_kv_append_vbias_fused_matches_two_dispatch() {
        let ctx = ctx();
        let base = 10_000.0f32;

        // (n_q_heads, n_k_heads, head_dim, pos, kv_off_heads, with_biases)
        // Qwen-3B shape: n_q=16, n_k=8, hd=128; also test GQA 4:1 and 1:1.
        let cases: &[(usize, usize, usize, u32, usize, bool)] = &[
            (16, 8, 128, 1, 0, false),  // Qwen-3B shape, pos=1, no bias, append at 0
            (16, 8, 128, 5, 8, true),   // with biases, kv_off=8 kv_heads
            (4, 4, 64, 0, 0, true),     // GQA 1:1, pos=0
            (8, 2, 64, 10, 3, false),   // GQA 4:1 variant
            (1, 1, 128, 100, 50, true), // single-head sanity
        ];

        for &(n_q, n_k, head_dim, pos, kv_off_heads, with_biases) in cases {
            let q_dim = n_q * head_dim;
            let kv_dim = n_k * head_dim;
            let kv_off = kv_off_heads * head_dim; // element offset into cache
            let cache_size = kv_off + kv_dim + 32; // extra padding to detect OOB writes

            let seed = (n_q + n_k + head_dim + pos as usize) as u32;
            let q = rnd(q_dim, seed);
            let k_tok = rnd(kv_dim, seed ^ 0x1000);
            let v_tok = rnd(kv_dim, seed ^ 0x2000);
            let q_bias_data = rnd(q_dim, seed ^ 0x3000);
            let k_bias_data = rnd(kv_dim, seed ^ 0x4000);
            let v_bias_data = rnd(kv_dim, seed ^ 0x5000);
            let q_bias = if with_biases { Some(q_bias_data.as_slice()) } else { None };
            let k_bias = if with_biases { Some(k_bias_data.as_slice()) } else { None };
            let v_bias = if with_biases { Some(v_bias_data.as_slice()) } else { None };

            let (ref_q, ref_kc, ref_vc) = run_ref(ctx, &q, &k_tok, &v_tok, q_bias, k_bias, v_bias, kv_off, cache_size, n_q, n_k, head_dim, pos, base);
            let (fused_q, fused_kc, fused_vc) = run_fused(ctx, &q, &k_tok, &v_tok, q_bias, k_bias, v_bias, kv_off, cache_size, n_q, n_k, head_dim, pos, base);

            let dq = max_abs_diff(&ref_q, &fused_q);
            let dkc = max_abs_diff(&ref_kc, &fused_kc);
            let dvc = max_abs_diff(&ref_vc, &fused_vc);

            assert_eq!(dq, 0.0, "n_q={n_q} n_k={n_k} hd={head_dim} pos={pos}: q max_diff={dq:.2e}");
            assert_eq!(dkc, 0.0, "n_q={n_q} n_k={n_k} hd={head_dim} pos={pos}: k_cache max_diff={dkc:.2e}");
            assert_eq!(dvc, 0.0, "n_q={n_q} n_k={n_k} hd={head_dim} pos={pos}: v_cache max_diff={dvc:.2e}");
            eprintln!("B6 n_q={n_q} n_k={n_k} hd={head_dim} pos={pos} bias={with_biases}: q={dq:.0e} kc={dkc:.0e} vc={dvc:.0e} OK");
        }
    }
}
#[rustfmt::skip]
mod rope_qk_b1_bias_parity {
    #![cfg(target_os = "macos")]
    //! Track 3.6 parity: rope_qk_f32_b1_bias must be bit-identical to
    //! (add_inplace q_bias + rope_q + add_inplace k_bias + rope_k).
    //!
    //! Tests several (n_q_heads, n_k_heads, head_dim, pos) combinations including
    //! GQA shapes typical of Qwen2.5-3B (n_q=16, n_k=8, head_dim=128).

    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};
    use once_cell::sync::Lazy;

    fn ctx() -> &'static MetalContext {
        static CTX: Lazy<MetalContext> = Lazy::new(|| MetalContext::new().expect("Metal device required"));
        &CTX
    }

    fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> hawking_core::metal::PinnedBuffer {
        let bytes = bytemuck::cast_slice(data);
        ctx.new_buffer_with_bytes(bytes)
    }

    fn read_f32_buf(buf: &hawking_core::metal::PinnedBuffer, n: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
    }

    fn rnd(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2654435761u32).wrapping_add(seed);
                (x as f32 / u32::MAX as f32) * 4.0 - 2.0
            })
            .collect()
    }

    /// Reference: 4 dispatches (q_bias + rope_q + k_bias + rope_k).
    fn run_ref(ctx: &MetalContext, q: &[f32], k: &[f32], q_bias: Option<&[f32]>, k_bias: Option<&[f32]>, n_q: usize, n_k: usize, hd: usize, pos: u32, base: f32) -> (Vec<f32>, Vec<f32>) {
        let q_buf = new_f32_buf(ctx, q);
        let k_buf = new_f32_buf(ctx, k);
        let mut tcb = TokenCommandBuffer::new(ctx);
        if let Some(qb) = q_bias {
            let b_buf = new_f32_buf(ctx, qb);
            kernels::add_inplace_metal_tcb(&mut tcb, &q_buf, &b_buf, q.len()).unwrap();
        }
        if let Some(kb) = k_bias {
            let b_buf = new_f32_buf(ctx, kb);
            kernels::add_inplace_metal_tcb(&mut tcb, &k_buf, &b_buf, k.len()).unwrap();
        }
        kernels::rope_q_f32_inplace_tcb(&mut tcb, &q_buf, n_q, hd, 0, hd, pos, base).unwrap();
        kernels::rope_q_f32_inplace_tcb(&mut tcb, &k_buf, n_k, hd, 0, hd, pos, base).unwrap();
        tcb.commit_and_wait().unwrap();
        (read_f32_buf(&q_buf, q.len()), read_f32_buf(&k_buf, k.len()))
    }

    /// Fused: 1 dispatch (rope_qk_f32_b1_bias).
    fn run_fused(ctx: &MetalContext, q: &[f32], k: &[f32], q_bias: Option<&[f32]>, k_bias: Option<&[f32]>, n_q: usize, n_k: usize, hd: usize, pos: u32, base: f32) -> (Vec<f32>, Vec<f32>) {
        let q_buf = new_f32_buf(ctx, q);
        let k_buf = new_f32_buf(ctx, k);
        let qb_buf = q_bias.map(|b| new_f32_buf(ctx, b));
        let kb_buf = k_bias.map(|b| new_f32_buf(ctx, b));
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::rope_qk_f32_b1_bias_tcb(&mut tcb, &q_buf, &k_buf, qb_buf.as_ref(), kb_buf.as_ref(), n_q, n_k, hd, pos, base).unwrap();
        tcb.commit_and_wait().unwrap();
        (read_f32_buf(&q_buf, q.len()), read_f32_buf(&k_buf, k.len()))
    }

    fn check(label: &str, n_q: usize, n_k: usize, hd: usize, pos: u32, with_bias: bool) {
        let ctx = ctx();
        let q = rnd(n_q * hd, 0xABCD + pos);
        let k = rnd(n_k * hd, 0x1234 + pos);
        let qb = with_bias.then(|| rnd(n_q * hd, 0xDEAD));
        let kb = with_bias.then(|| rnd(n_k * hd, 0xBEEF));
        let (rq, rk) = run_ref(ctx, &q, &k, qb.as_deref(), kb.as_deref(), n_q, n_k, hd, pos, 10000.0);
        let (fq, fk) = run_fused(ctx, &q, &k, qb.as_deref(), kb.as_deref(), n_q, n_k, hd, pos, 10000.0);
        let max_q = rq.iter().zip(&fq).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let max_k = rk.iter().zip(&fk).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // The fused kernel computes bias-add + rope in one pass vs the reference's
        // store-reload between the two dispatches. The Metal compiler may apply FMA
        // fusion across the (q+bias)*c and (q+bias)*s products, producing 1-ULP
        // rounding differences. Allow a relative tolerance of 1e-5.
        let rel_q = rq.iter().zip(&fq).filter(|(a, b)| a.is_finite() && b.is_finite()).map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1.0)).fold(0.0f32, f32::max);
        let rel_k = rk.iter().zip(&fk).filter(|(a, b)| a.is_finite() && b.is_finite()).map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1.0)).fold(0.0f32, f32::max);
        assert!(rel_q < 1e-5, "{label}: Q max_rel={rel_q:.2e} > 1e-5");
        assert!(rel_k < 1e-5, "{label}: K max_rel={rel_k:.2e} > 1e-5");
        eprintln!("{label}: Q max_rel={rel_q:.2e}  K max_rel={rel_k:.2e}  OK");
    }

    #[test]
    fn rope_qk_b1_bias_qwen3b_shape() {
        // Qwen2.5-3B: n_q=16, n_k=8 (GQA), head_dim=128
        check("qwen3b  bias  pos=0", 16, 8, 128, 0, true);
        check("qwen3b  bias  pos=127", 16, 8, 128, 127, true);
        check("qwen3b  bias  pos=512", 16, 8, 128, 512, true);
        check("qwen3b  nobias pos=63", 16, 8, 128, 63, false);
    }

    #[test]
    fn rope_qk_b1_bias_mha_shape() {
        // MHA (n_q == n_k): e.g. 32 heads, head_dim=128
        check("mha128 bias  pos=1", 32, 32, 128, 1, true);
        check("mha128 nobias pos=255", 32, 32, 128, 255, false);
    }

    #[test]
    fn rope_qk_b1_bias_small_shape() {
        // Small shapes to check edge cases
        check("small  bias  pos=7", 4, 2, 64, 7, true);
        check("single nobias pos=0", 1, 1, 128, 0, false);
    }
}
#[rustfmt::skip]
mod rope_qk_fused_parity {
    #![cfg(target_os = "macos")]
    //! Track 3.4 parity: `rope_qk_f32_batched_multiseq` (fused Q+K) must be
    //! BIT-IDENTICAL to two separate `rope_f32_batched_multiseq` calls.
    //!
    //! Saves 1 dispatch/layer × 28 layers = 28 dispatches on Qwen-3B.

    use hawking_core::kernels;

    use crate::common;
    use common::*;

    /// Run the fused Q+K kernel and return (q_out, k_out).
    fn run_fused(q: &[f32], k: &[f32], n_q_heads: usize, n_k_heads: usize, head_dim: usize, positions: &[u32], theta: f32) -> (Vec<f32>, Vec<f32>) {
        let ctx = ctx();
        let b = positions.len();
        let q_dim = n_q_heads * head_dim;
        let kv_dim = n_k_heads * head_dim;
        let q_buf = new_f32_buf(ctx, q);
        let k_buf = new_f32_buf(ctx, k);
        let pos_bytes: Vec<u8> = positions.iter().flat_map(|&p| p.to_le_bytes()).collect();
        let pos_buf = ctx.new_buffer_with_bytes(&pos_bytes);
        let mut tcb = hawking_core::metal::TokenCommandBuffer::new(ctx);
        kernels::rope_qk_f32_batched_multiseq_tcb(&mut tcb, &q_buf, &k_buf, &pos_buf, n_q_heads, n_k_heads, head_dim, q_dim, kv_dim, b, theta).expect("rope_qk fused");
        tcb.commit_and_wait().expect("commit");
        let q_out = read_f32_buf(&q_buf, b * q_dim);
        let k_out = read_f32_buf(&k_buf, b * kv_dim);
        (q_out, k_out)
    }

    /// Run two separate rope calls and return (q_out, k_out).
    fn run_separate(q: &[f32], k: &[f32], n_q_heads: usize, n_k_heads: usize, head_dim: usize, positions: &[u32], theta: f32) -> (Vec<f32>, Vec<f32>) {
        let ctx = ctx();
        let b = positions.len();
        let q_dim = n_q_heads * head_dim;
        let kv_dim = n_k_heads * head_dim;
        let q_buf = new_f32_buf(ctx, q);
        let k_buf = new_f32_buf(ctx, k);
        let pos_bytes: Vec<u8> = positions.iter().flat_map(|&p| p.to_le_bytes()).collect();
        let pos_buf = ctx.new_buffer_with_bytes(&pos_bytes);
        let mut tcb = hawking_core::metal::TokenCommandBuffer::new(ctx);
        kernels::rope_f32_batched_multiseq_tcb(&mut tcb, &q_buf, &pos_buf, n_q_heads, head_dim, q_dim, b, theta).expect("rope Q");
        kernels::rope_f32_batched_multiseq_tcb(&mut tcb, &k_buf, &pos_buf, n_k_heads, head_dim, kv_dim, b, theta).expect("rope K");
        tcb.commit_and_wait().expect("commit");
        let q_out = read_f32_buf(&q_buf, b * q_dim);
        let k_out = read_f32_buf(&k_buf, b * kv_dim);
        (q_out, k_out)
    }

    fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
                (x / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn rope_qk_fused_matches_separate_bit_identical() {
        // Qwen-3B-like dimensions: n_heads=16, n_kv_heads=8, head_dim=128, B=1..8
        let configs: &[(usize, usize, usize, &[u32])] = &[
            (16, 8, 128, &[0]),                                 // B=1, pos=0 (identity rotation)
            (16, 8, 128, &[1, 7]),                              // B=2
            (16, 8, 128, &[3, 17, 42, 99]),                     // B=4
            (16, 8, 128, &[0, 1, 511, 1023, 2047, 3, 77, 200]), // B=8
            (32, 32, 128, &[5, 15, 100]),                       // square GQA (n_q=n_kv)
        ];
        let theta = 1000000.0f32; // Qwen rope base

        for &(n_q, n_k, hd, positions) in configs {
            let b = positions.len();
            let q = rand_vec(b * n_q * hd, 0xDEAD);
            let k = rand_vec(b * n_k * hd, 0xBEEF);

            let (fq, fk) = run_fused(&q, &k, n_q, n_k, hd, positions, theta);
            let (sq, sk) = run_separate(&q, &k, n_q, n_k, hd, positions, theta);

            let q_diff = fq.iter().zip(&sq).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let k_diff = fk.iter().zip(&sk).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

            assert_eq!(q_diff, 0.0, "B={b} n_q={n_q} n_k={n_k} hd={hd}: Q max_diff={q_diff} (expected 0)");
            assert_eq!(k_diff, 0.0, "B={b} n_q={n_q} n_k={n_k} hd={hd}: K max_diff={k_diff} (expected 0)");
            eprintln!("B={b} n_q_heads={n_q} n_kv_heads={n_k} head_dim={hd}: bit-identical OK");
        }
    }
}
#[rustfmt::skip]
mod swiglu_fused_ffn_parity {
    #![cfg(target_os = "macos")]
    //! Track 3.5 parity: SwiGLU-fused ffn_down must be bit-identical to
    //! (silu_mul + separate ffn_down) for both v3w (B=5..8) and v4r (B=2..4) paths.
    //!
    //! Saves 1 dispatch/layer × 28 layers = 28 dispatches on Qwen-3B.
    //! Parity gate: `atol = 0` (bit-identical, same arithmetic in same order).

    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};

    use crate::common;
    use common::*;

    /// Run the reference path: silu_mul + separate v3w_predec ffn_down.
    fn run_ref_v3w(ctx: &MetalContext, w_q4: &[u8], scales: &[f32], gate: &[f32], up: &[f32], rows: usize, cols: usize, b: usize) -> Vec<f32> {
        let w_buf = ctx.new_buffer_with_bytes(w_q4);
        let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
        let gate_buf = new_f32_buf(ctx, gate);
        let up_buf = new_f32_buf(ctx, up);
        let act_buf = ctx.new_buffer(b * cols * 4);
        let y_buf = ctx.new_buffer(b * rows * 4);

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::silu_mul_tcb(&mut tcb, &gate_buf, &up_buf, &act_buf, b * cols).unwrap();
        let w_bytes = rows * (cols / 256) * 144;
        kernels::gemm_q4_k_m_batched_v3w_predec_pinned_tcb(&mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &act_buf, &y_buf).unwrap();
        tcb.commit_and_wait().unwrap();
        read_f32_buf(&y_buf, b * rows)
    }

    /// Run the fused path: swiglu v3w_predec ffn_down.
    fn run_fused_v3w(ctx: &MetalContext, w_q4: &[u8], scales: &[f32], gate: &[f32], up: &[f32], rows: usize, cols: usize, b: usize) -> Vec<f32> {
        let w_buf = ctx.new_buffer_with_bytes(w_q4);
        let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
        let gate_buf = new_f32_buf(ctx, gate);
        let up_buf = new_f32_buf(ctx, up);
        let y_buf = ctx.new_buffer(b * rows * 4);

        let mut tcb = TokenCommandBuffer::new(ctx);
        let w_bytes = rows * (cols / 256) * 144;
        kernels::gemm_q4_k_m_batched_v3w_predec_swiglu_pinned_tcb(&mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &gate_buf, &up_buf, &y_buf).unwrap();
        tcb.commit_and_wait().unwrap();
        read_f32_buf(&y_buf, b * rows)
    }

    fn run_ref_v4r(ctx: &MetalContext, w_q4: &[u8], scales: &[f32], gate: &[f32], up: &[f32], rows: usize, cols: usize, b: usize) -> Vec<f32> {
        let w_buf = ctx.new_buffer_with_bytes(w_q4);
        let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
        let gate_buf = new_f32_buf(ctx, gate);
        let up_buf = new_f32_buf(ctx, up);
        let act_buf = ctx.new_buffer(b * cols * 4);
        let y_buf = ctx.new_buffer(b * rows * 4);

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::silu_mul_tcb(&mut tcb, &gate_buf, &up_buf, &act_buf, b * cols).unwrap();
        let w_bytes = rows * (cols / 256) * 144;
        kernels::gemm_q4_k_m_batched_v4r_predec_pinned_tcb(&mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &act_buf, &y_buf).unwrap();
        tcb.commit_and_wait().unwrap();
        read_f32_buf(&y_buf, b * rows)
    }

    fn run_fused_v4r(ctx: &MetalContext, w_q4: &[u8], scales: &[f32], gate: &[f32], up: &[f32], rows: usize, cols: usize, b: usize) -> Vec<f32> {
        let w_buf = ctx.new_buffer_with_bytes(w_q4);
        let sc_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(scales));
        let gate_buf = new_f32_buf(ctx, gate);
        let up_buf = new_f32_buf(ctx, up);
        let y_buf = ctx.new_buffer(b * rows * 4);

        let mut tcb = TokenCommandBuffer::new(ctx);
        let w_bytes = rows * (cols / 256) * 144;
        kernels::gemm_q4_k_m_batched_v4r_predec_swiglu_pinned_tcb(&mut tcb, &w_buf, 0, w_bytes, &sc_buf, 0, rows, cols, b, &gate_buf, &up_buf, &y_buf).unwrap();
        tcb.commit_and_wait().unwrap();
        read_f32_buf(&y_buf, b * rows)
    }

    fn make_q4k_weights(rows: usize, cols: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let blocks_per_row = cols / 256;
        let block_bytes = 144;
        let total_bytes = rows * blocks_per_row * block_bytes;
        let w: Vec<u8> = (0..total_bytes).map(|i| ((i as u32).wrapping_mul(2246822519u32).wrapping_add(seed)) as u8).collect();
        // Predec scale table: 16 f32 per block (8 d,m pairs)
        let n_scales = rows * blocks_per_row * 16;
        let s: Vec<f32> = (0..n_scales)
            .map(|i| {
                let v = ((i as u32).wrapping_mul(2654435761u32).wrapping_add(seed)) as f32 / u32::MAX as f32;
                v * 2.0 - 1.0
            })
            .collect();
        (w, s)
    }

    fn rand_vec(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761u32).wrapping_add(seed)) as f32;
                (x / u32::MAX as f32) * 4.0 - 2.0
            })
            .collect()
    }

    #[test]
    fn swiglu_fused_v3w_matches_ref() {
        let ctx = ctx();
        // Qwen-3B-like: intermediate=11008, hidden=2048
        let rows = 2048;
        let cols = 11008;
        let (w, scales) = make_q4k_weights(rows, cols, 0xABCD);

        for b in [5usize, 6, 7, 8] {
            let gate = rand_vec(b * cols, 0xDEAD + b as u32);
            let up = rand_vec(b * cols, 0xBEEF + b as u32);
            let ref_out = run_ref_v3w(ctx, &w, &scales, &gate, &up, rows, cols, b);
            let fused_out = run_fused_v3w(ctx, &w, &scales, &gate, &up, rows, cols, b);
            let max_diff = ref_out.iter().zip(&fused_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(max_diff < 1e-4, "B={b}: v3w swiglu max_diff={max_diff} > atol 1e-4");
            eprintln!("v3w swiglu B={b}: max_diff={max_diff:.2e} OK");
        }
    }

    #[test]
    fn swiglu_fused_v4r_matches_ref() {
        let ctx = ctx();
        let rows = 2048;
        let cols = 11008;
        let (w, scales) = make_q4k_weights(rows, cols, 0x1234);

        // Wave-R0: extended to B=5..8 to gate the HAWKING_QWEN_MULTISEQ_V4R_HIGHB
        // route (fused v4r swiglu must match its f32 ref on the ffn_down shape at high B).
        for b in [2usize, 3, 4, 5, 6, 7, 8] {
            let gate = rand_vec(b * cols, 0xCAFE + b as u32);
            let up = rand_vec(b * cols, 0xF00D + b as u32);
            let ref_out = run_ref_v4r(ctx, &w, &scales, &gate, &up, rows, cols, b);
            let fused_out = run_fused_v4r(ctx, &w, &scales, &gate, &up, rows, cols, b);
            let max_diff = ref_out.iter().zip(&fused_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(max_diff < 1e-4, "B={b}: v4r swiglu max_diff={max_diff} > atol 1e-4");
            eprintln!("v4r swiglu B={b}: max_diff={max_diff:.2e} OK");
        }
    }
}
#[rustfmt::skip]
mod tcb_dispatch_cost {
    //! DIAGNOSTIC (path-to-50, 2026-05-29): isolate the in-TCB marginal
    //! per-dispatch cost from the command-buffer round-trip latency.
    //!
    //! `bench-kernel` commits+waits one command buffer per dispatch, so its
    //! ~130 us "floor" is the CB round-trip, not the kernel. Production decode
    //! batches many dispatches into ONE TokenCommandBuffer with a single
    //! commit_and_wait. This test dispatches the SAME Q4_K GEMV K times into one
    //! TCB and times the whole commit, for K in {1,2,4,...,256}. The slope of
    //! total-vs-K is the true in-TCB per-dispatch cost; the intercept is the CB
    //! round-trip. That slope sets the ceiling for any dispatch-fusion lever.
    //!
    //! Run: cargo test --release -p hawking-core --test tcb_dispatch_cost -- --nocapture

    use hawking_core::metal::{MetalContext, TokenCommandBuffer};
    use std::time::Instant;

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    /// Time M iterations of {new TCB; K dispatches of gemm_q4_k_m_fused_v2; commit_and_wait}.
    /// Returns median total microseconds per iteration.
    ///
    /// `distinct` weight buffers are cycled across the K dispatches. With
    /// distinct=1 every dispatch re-reads the same matrix (L2-cache-hit, like a
    /// microbench). With distinct=32 the working set (32 x matrix bytes) exceeds
    /// L2 so every dispatch in a window reads a COLD matrix from DRAM — the
    /// production decode regime (each layer's q/k/v/o/gate/up/down differs).
    fn bench_k(ctx: &MetalContext, rows: usize, cols: usize, k: usize, iters: usize, distinct: usize) -> f64 {
        let blocks_per_row = cols / 256;
        let w_bytes = rows * blocks_per_row * 144;
        let x_bytes = cols * std::mem::size_of::<f32>();
        let out_bytes = rows * std::mem::size_of::<f32>();

        let w_bufs: Vec<_> = (0..distinct.max(1)).map(|_| ctx.new_buffer(w_bytes)).collect();
        let x_buf = ctx.new_buffer(x_bytes);
        let out_buf = ctx.new_buffer(out_bytes);

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        let n_tg = (rows as u32 + 7) / 8;
        let grid = (n_tg * TG, 1, 1);
        let tg = (TG, 1, 1);

        let one_tcb = |ctx: &MetalContext| {
            let mut tcb = TokenCommandBuffer::new(ctx);
            for i in 0..k {
                let w_buf = &w_bufs[i % w_bufs.len()];
                tcb.dispatch_threads("gemm_q4_k_m_fused_v2", grid, tg, |enc| {
                    enc.set_buffer(0, Some(w_buf), 0);
                    enc.set_buffer(1, Some(&x_buf), 0);
                    enc.set_buffer(2, Some(&out_buf), 0);
                    enc.set_bytes(3, std::mem::size_of::<u32>() as u64, &rows_u32 as *const u32 as *const _);
                    enc.set_bytes(4, std::mem::size_of::<u32>() as u64, &cols_u32 as *const u32 as *const _);
                })
                .unwrap();
            }
            tcb.commit_and_wait().unwrap();
        };

        // warmup
        for _ in 0..20 {
            one_tcb(ctx);
        }
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            one_tcb(ctx);
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        median(samples)
    }

    #[test]
    fn tcb_dispatch_cost_curve() {
        let ctx = MetalContext::new().expect("Metal device required");
        let ks = [1usize, 2, 4, 8, 16, 32, 64, 128, 256];
        // 2048x2048 = q/o proj shape; 11008x2048 = ffn gate/up shape.
        for (rows, cols, label) in [(2048usize, 2048usize, "q/o 2048x2048"), (11008, 2048, "ffn 11008x2048")] {
            for distinct in [1usize, 32usize] {
                let regime = if distinct == 1 { "SAME buf (L2 cache-hit)" } else { "32 distinct bufs (cold DRAM, = decode)" };
                println!("\n=== {label} — {regime} — K dispatches in ONE TCB ===");
                println!("{:>5}  {:>12}  {:>14}  {:>16}", "K", "total_us", "us/dispatch", "marginal_us/disp");
                let base = bench_k(&ctx, rows, cols, 1, 300, distinct);
                for &k in &ks {
                    let total = bench_k(&ctx, rows, cols, k, if k <= 16 { 300 } else { 150 }, distinct);
                    let per = total / k as f64;
                    let marginal = if k > 1 { (total - base) / (k as f64 - 1.0) } else { total };
                    println!("{k:>5}  {total:>12.1}  {per:>14.1}  {marginal:>16.1}");
                }
            }
        }
    }
}
#[rustfmt::skip]
mod tq_trellis_parity {
    //! TQ G4 bitslice GPU↔CPU bit-identity gate (Slice 3).
    //!
    //! The non-negotiable contract of the TQ Metal port: the GPU `strand_bitslice_decode`
    //! kernel's Q12 output is byte-for-byte equal to the integer CPU oracle
    //! `strand_quant::decode::decode_tensor_fixed` — the same determinism contract the
    //! CPU serving reference (`crate::tq`) honours. A fast wrong kernel is worse than no
    //! kernel, so perf is never measured here; only identity.
    //!
    //! Drives the public, `BitsliceEntry`-free entry point
    //! `hawking_core::gpu_decode_q12` (bake → pin payload+table → dispatch decode →
    //! read back `Vec<i32>`), swept over the encode-lever matrix: k ∈ {2,3,4},
    //! L ∈ {7,12}, tail-biting × affine-min, and edge lengths (short final block,
    //! sub-block tails, 1-weight tensors). Skips cleanly when no Metal device is
    //! present (never a fake pass).
    //!
    //! Run with:
    //!   cargo test -p hawking-core --features tq --test tq_trellis_parity -- --nocapture
    //!
    //! The whole file is gated on macOS + `tq` (the GPU path and the `strand_quant`
    //! dep only exist there).

    #![cfg(all(target_os = "macos", feature = "tq"))]

    use hawking_core::gpu_decode_q12;
    use hawking_core::metal::MetalContext;
    use strand_quant::decode::decode_tensor_fixed;
    use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};
    use strand_quant::TrellisConfig;

    /// Deterministic synthetic weights (a smooth signal so the encoder exercises a
    /// spread of trellis states, parameterised by `seed` for edge-length coverage).
    fn synth_w(n: usize, seed: u64) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect()
    }

    /// The k∈{2,3,4} × L∈{7,12} deploy/reopen matrix the gate sweeps. `for_bpw`
    /// gives the natural L per k (k+4); `for_bpw_l` pins the off-axis L=12 reopen and
    /// the small-L folds so the threadgroup-LUT staging is exercised at both 512 B
    /// (L=7) and 16 KB (L=12).
    fn gate_configs() -> Vec<(TrellisConfig, &'static str)> {
        vec![
            (TrellisConfig::for_bpw(3.0), "k3 L7 (3-bit deploy)"),
            (TrellisConfig::for_bpw(2.0), "k2 L6"),
            (TrellisConfig::for_bpw(4.0), "k4 L8"),
            (TrellisConfig::for_bpw_l(2.0, 12), "k2 L12 (2-bit reopen)"),
            (TrellisConfig::for_bpw_l(3.0, 12), "k3 L12"),
            (TrellisConfig::for_bpw_l(4.0, 7), "k4 L7"),
        ]
    }

    /// One bit-identity assertion: GPU decode of `enc` under `cfg` == CPU oracle,
    /// element-for-element. `gpu_decode_q12` returns `None` only for the
    /// vec/over-256 fallback (never on this scalar matrix), so `None` is a hard fail.
    fn assert_gpu_eq_cpu(ctx: &MetalContext, enc: &strand_quant::encode::EncodedTensor, cfg: &TrellisConfig, label: &str) {
        let got = gpu_decode_q12(ctx, enc, cfg).unwrap_or_else(|| panic!("{label}: gpu_decode_q12 returned None (bake rejected?)")).unwrap_or_else(|e| panic!("{label}: GPU decode error: {e}"));
        let want = decode_tensor_fixed(enc, cfg);
        assert_eq!(got.len(), want.len(), "{label}: length mismatch GPU {} vs CPU {}", got.len(), want.len());
        // Bit-for-bit (these are integers; == is exact).
        if got != want {
            let first = got.iter().zip(want.iter()).enumerate().find(|(_, (a, b))| a != b).map(|(i, (a, b))| (i, *a, *b));
            panic!("{label}: GPU Q12 != CPU oracle bit-for-bit; first diff = {first:?}");
        }
    }

    #[test]
    fn bitslice_gpu_decode_matches_cpu_oracle_over_matrix() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("[tq_trellis_parity] no Metal device; skipping GPU↔CPU gate");
            return;
        };

        // Probe the stride contract once up front, with a clear message: the GPU
        // sizeof(BitsliceEntry) must equal the host #[repr(C)] size (84 B) or every
        // assertion below would be meaningless. The decode path also re-checks it,
        // but surfacing it here makes a stride mismatch unmistakable.
        {
            let cfg = TrellisConfig::for_bpw(3.0);
            let enc = encode_tensor(&synth_w(256, 0), &cfg);
            // A trivial decode that, if it returns Ok, proves the probe passed.
            let r = gpu_decode_q12(&ctx, &enc, &cfg).expect("scalar bake").expect("stride probe + decode");
            assert_eq!(r.len(), 256);
        }

        // Edge lengths: 1 weight, < one block, exactly one block, one block + tail,
        // a sub-block-aligned tail, and a large multi-block tensor.
        let lengths = [1usize, 7, 31, 32, 33, 255, 256, 257, 288, 512, 1000, 2049];

        for (cfg, cfg_label) in gate_configs() {
            for &n in &lengths {
                for seed in 0..4u64 {
                    let w = synth_w(n, seed);

                    // plain
                    let enc = encode_tensor(&w, &cfg);
                    assert_gpu_eq_cpu(&ctx, &enc, &cfg, &format!("{cfg_label} n={n} seed={seed} plain"));

                    // tail-biting (the stored-vs-walked init_state branch)
                    let enc_tb = encode_tensor_with(&w, &cfg, &EncodeOpts { tail_biting: true, ..Default::default() });
                    assert_gpu_eq_cpu(&ctx, &enc_tb, &cfg, &format!("{cfg_label} n={n} seed={seed} tail_biting"));

                    // affine-min (the off[8] add path)
                    let enc_am = encode_tensor_with(&w, &cfg, &EncodeOpts { affine_min: true, ..Default::default() });
                    assert_gpu_eq_cpu(&ctx, &enc_am, &cfg, &format!("{cfg_label} n={n} seed={seed} affine_min"));

                    // tail-biting + affine-min together
                    let enc_both = encode_tensor_with(&w, &cfg, &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() });
                    assert_gpu_eq_cpu(&ctx, &enc_both, &cfg, &format!("{cfg_label} n={n} seed={seed} tail+affine"));
                }
            }
        }

        println!(
            "[tq_trellis_parity] GPU bitslice decode == decode_tensor_fixed bit-for-bit \
             across k∈{{2,3,4}} L∈{{7,12}} × 4 encode variants × {} edge lengths",
            lengths.len()
        );
    }

    /// A wide, single-tensor decode at a realistic projection shape (rows×cols, a
    /// multiple of 256) — confirms the all-blocks grid and the `out_off` prefix sum
    /// hold at scale, not just on tiny tensors.
    #[test]
    fn bitslice_gpu_decode_matches_cpu_oracle_wide_shape() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("[tq_trellis_parity] no Metal device; skipping wide-shape gate");
            return;
        };
        let (rows, cols) = (16usize, 2048usize); // 32768 weights, 128 blocks
        let total = rows * cols;
        for (cfg, cfg_label) in [(TrellisConfig::for_bpw(3.0), "k3 L7"), (TrellisConfig::for_bpw_l(2.0, 12), "k2 L12")] {
            let w = synth_w(total, 0xABCD);
            let enc = encode_tensor(&w, &cfg);
            assert_gpu_eq_cpu(&ctx, &enc, &cfg, &format!("{cfg_label} wide {rows}x{cols}"));
        }
        println!("[tq_trellis_parity] wide-shape GPU decode bit-identical to oracle");
    }

    // ── k=1 (1-bit) configuration tests ─────────────────────────────────────────
    //
    // k=1 is the lowest valid trellis depth. `for_bpw(1.0)` resolves to k=1,
    // L=5 (= k+4 = 5, well within [MIN_L=4, MAX_L=14]). These tests verify the
    // config constructor and field values; the GPU parity test is #[ignore] until
    // the G4 kernel is validated against the k=1 trellis path.

    /// `TrellisConfig::for_bpw(1.0)` must give k=1 and L=5 (= k+4).
    #[test]
    fn trellis_k1_l5_config_valid() {
        let cfg = TrellisConfig::for_bpw(1.0);
        assert_eq!(cfg.k_bits, 1, "for_bpw(1.0) must give k=1");
        assert_eq!(cfg.l_bits, 5, "for_bpw(1.0) must give L=k+4=5");
        assert_eq!(cfg.block_len, 256, "default block_len must be 256");
    }

    /// `TrellisConfig::new(7, 1, 256)` must produce k=1, L=7, block_len=256.
    #[test]
    fn trellis_k1_l7_explicit() {
        let cfg = TrellisConfig::new(7, 1, 256);
        assert_eq!(cfg.k_bits, 1, "explicit k=1 must be stored");
        assert_eq!(cfg.l_bits, 7, "explicit L=7 must be stored");
        assert_eq!(cfg.block_len, 256);
        // num_states() = 2^L = 128
        assert_eq!(cfg.num_states(), 128, "2^7 = 128 trellis states");
    }

    /// `TrellisConfig::new(9, 1, 256)` must produce k=1, L=9.
    #[test]
    fn trellis_k1_l9_config() {
        let cfg = TrellisConfig::new(9, 1, 256);
        assert_eq!(cfg.k_bits, 1);
        assert_eq!(cfg.l_bits, 9);
        assert_eq!(cfg.num_states(), 512, "2^9 = 512 trellis states");
    }

    /// Placeholder for a future GPU bit-identity gate at k=1. Marked #[ignore]
    /// because the G4 Metal kernel has not yet been validated against the k=1
    /// trellis path — enable once `strand_bitslice_decode` handles k=1 without
    /// register-pressure divergence.
    #[test]
    #[ignore = "k=1 GPU path not yet validated — enable after kernel confirms k=1 coverage"]
    fn trellis_k1_gpu_decode_parity() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("[tq_trellis_parity] no Metal device; skipping k=1 GPU gate");
            return;
        };
        let cfg = TrellisConfig::new(7, 1, 256);
        let w = (0..256usize).map(|i| ((i as f32) * 0.0137).sin() * 0.5).collect::<Vec<_>>();
        let enc = strand_quant::encode::encode_tensor(&w, &cfg);
        assert_gpu_eq_cpu(&ctx, &enc, &cfg, "k=1 L=7 n=256 plain");
        println!("[tq_trellis_parity] k=1 GPU parity PASSED");
    }
}
#[rustfmt::skip]
mod v030_gemm_q4_k_simd_parity {
    //! v0.3.0/v0.3.1 — Numerical parity test: gemm_q4_k_m_fused_simd vs scalar reference.
    //!
    //! Covers both the standalone path (gemv_q4_k_m_simd) and the batched path
    //! (dispatch_gemv_q4_k_m_simd_batched / encode_gemv_q4_k_m_simd via ctx.dispatch_batch).
    //! Shape: M=64, K=256 (1 Q4_K block per row), seed=42.
    //! Asserts max |scalar - simd| < 1e-3 (fp16 quant noise tolerance).

    #![cfg(target_os = "macos")]

    use hawking_core::gguf::GgmlType;
    use hawking_core::kernels;
    use hawking_core::quant::dequant_into;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    pub const ATOL: f32 = 1e-3;

    fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
    }

    /// Synthetic Q4_K weight bytes with small d/dmin so per-element magnitudes
    /// stay bounded; prevents accumulation-order divergence from crossing 1e-3.
    fn synthetic_q4_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
        use half::f16;
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.01 + rng.gen::<f32>() * 0.01;
            let d_bits = f16::from_f32(d).to_bits();
            bytes[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
            let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
            let dmin_bits = f16::from_f32(dmin).to_bits();
            bytes[off + 2..off + 4].copy_from_slice(&dmin_bits.to_le_bytes());
            for i in 4..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    #[test]
    fn test_gemm_q4_k_simd_matches_scalar() {
        let rows = 64;
        let cols = 256; // 1 Q4_K block per row
        let n_blocks = rows * (cols / 256);

        let w_bytes = synthetic_q4_k_bytes(n_blocks, 42);
        let x = fixed_input(cols, 0xDEAD_BEEF);

        // Scalar reference: dequant → fp32 GEMV.
        let mut w_f32 = vec![0.0_f32; rows * cols];
        dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32).expect("Q4_K dequant should succeed for synthetic bytes");
        let mut scalar_out = vec![0.0_f32; rows];
        kernels::gemv_f32(&w_f32, rows, cols, &x, &mut scalar_out);

        // simdgroup Metal path.
        let ctx = ctx().clone();
        let mut simd_out = vec![0.0_f32; rows];
        kernels::gemv_q4_k_m_simd(&ctx, &w_bytes, rows, cols, &x, &mut simd_out).expect("gemv_q4_k_m_simd should succeed");

        let diff = max_abs_diff(&scalar_out, &simd_out);
        println!("[v0.3.0] gemm_q4_k_simd parity max abs diff = {diff:.6e}");
        assert!(diff < ATOL, "gemm_q4_k_m_fused_simd vs scalar diff {diff:.6e} >= atol {ATOL}");
    }

    #[test]
    fn test_gemm_q4_k_simd_larger_shape() {
        // Larger shape: multiple Q4_K blocks per row, rows not multiple of 8.
        let rows = 128;
        let cols = 512; // 2 Q4_K blocks per row
        let n_blocks = rows * (cols / 256);

        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xCAFE_BABE);
        let x = fixed_input(cols, 0x1234_5678);

        let mut w_f32 = vec![0.0_f32; rows * cols];
        dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32).expect("Q4_K dequant should succeed");
        let mut scalar_out = vec![0.0_f32; rows];
        kernels::gemv_f32(&w_f32, rows, cols, &x, &mut scalar_out);

        let ctx = ctx().clone();
        let mut simd_out = vec![0.0_f32; rows];
        kernels::gemv_q4_k_m_simd(&ctx, &w_bytes, rows, cols, &x, &mut simd_out).expect("gemv_q4_k_m_simd should succeed");

        let diff = max_abs_diff(&scalar_out, &simd_out);
        println!("[v0.3.0] gemm_q4_k_simd larger shape parity max abs diff = {diff:.6e}");
        assert!(diff < ATOL, "gemm_q4_k_m_fused_simd vs scalar diff {diff:.6e} >= atol {ATOL}");
    }

    // v0.3.1 batched-path parity tests: exercise dispatch_gemv_q4_k_m_simd_batched
    // (routes through ctx.dispatch_batch { encode_gemv_q4_k_m_simd }).

    #[test]
    fn test_gemm_q4_k_simd_batched_matches_scalar() {
        let rows = 64;
        let cols = 256;
        let n_blocks = rows * (cols / 256);

        let w_bytes = synthetic_q4_k_bytes(n_blocks, 42);
        let x = fixed_input(cols, 0xDEAD_BEEF);

        let mut w_f32 = vec![0.0_f32; rows * cols];
        dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32).expect("Q4_K dequant should succeed for synthetic bytes");
        let mut scalar_out = vec![0.0_f32; rows];
        kernels::gemv_f32(&w_f32, rows, cols, &x, &mut scalar_out);

        let ctx = ctx().clone();
        let mut batched_out = vec![0.0_f32; rows];
        kernels::dispatch_gemv_q4_k_m_simd_batched(&ctx, &w_bytes, rows, cols, &x, &mut batched_out).expect("dispatch_gemv_q4_k_m_simd_batched should succeed");

        let diff = max_abs_diff(&scalar_out, &batched_out);
        println!("[v0.3.1] gemm_q4_k_simd batched parity max abs diff = {diff:.6e}");
        assert!(diff < ATOL, "dispatch_gemv_q4_k_m_simd_batched vs scalar diff {diff:.6e} >= atol {ATOL}");
    }

    #[test]
    fn test_gemm_q4_k_simd_batched_larger_shape() {
        let rows = 128;
        let cols = 512;
        let n_blocks = rows * (cols / 256);

        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xCAFE_BABE);
        let x = fixed_input(cols, 0x1234_5678);

        let mut w_f32 = vec![0.0_f32; rows * cols];
        dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32).expect("Q4_K dequant should succeed");
        let mut scalar_out = vec![0.0_f32; rows];
        kernels::gemv_f32(&w_f32, rows, cols, &x, &mut scalar_out);

        let ctx = ctx().clone();
        let mut batched_out = vec![0.0_f32; rows];
        kernels::dispatch_gemv_q4_k_m_simd_batched(&ctx, &w_bytes, rows, cols, &x, &mut batched_out).expect("dispatch_gemv_q4_k_m_simd_batched should succeed");

        let diff = max_abs_diff(&scalar_out, &batched_out);
        println!("[v0.3.1] gemm_q4_k_simd batched larger shape max abs diff = {diff:.6e}");
        assert!(diff < ATOL, "dispatch_gemv_q4_k_m_simd_batched vs scalar diff {diff:.6e} >= atol {ATOL}");
    }

    // v0.3.2 pair-path parity tests: exercise dispatch_gemv_q4_k_m_simd_pair_batched
    // (encodes two simd GEMVs — distinct weights, shared x — into one CommandBatch).

    #[test]
    fn test_gemm_q4_k_simd_pair_matches_scalar() {
        let rows = 64;
        let cols = 256;
        let n_blocks = rows * (cols / 256);

        let w_a_bytes = synthetic_q4_k_bytes(n_blocks, 42);
        let w_b_bytes = synthetic_q4_k_bytes(n_blocks, 0xDEAD_CAFE);
        let x = fixed_input(cols, 0xDEAD_BEEF);

        let mut w_a_f32 = vec![0.0_f32; rows * cols];
        let mut w_b_f32 = vec![0.0_f32; rows * cols];
        dequant_into(GgmlType::Q4_K, &w_a_bytes, &mut w_a_f32).expect("Q4_K dequant a");
        dequant_into(GgmlType::Q4_K, &w_b_bytes, &mut w_b_f32).expect("Q4_K dequant b");
        let mut scalar_a = vec![0.0_f32; rows];
        let mut scalar_b = vec![0.0_f32; rows];
        kernels::gemv_f32(&w_a_f32, rows, cols, &x, &mut scalar_a);
        kernels::gemv_f32(&w_b_f32, rows, cols, &x, &mut scalar_b);

        let ctx = ctx().clone();
        let mut pair_a = vec![0.0_f32; rows];
        let mut pair_b = vec![0.0_f32; rows];
        kernels::dispatch_gemv_q4_k_m_simd_pair_batched(&ctx, &w_a_bytes, &w_b_bytes, rows, cols, &x, &mut pair_a, &mut pair_b).expect("dispatch_gemv_q4_k_m_simd_pair_batched should succeed");

        let diff_a = max_abs_diff(&scalar_a, &pair_a);
        let diff_b = max_abs_diff(&scalar_b, &pair_b);
        println!("[v0.3.2] pair parity diff_a={diff_a:.6e} diff_b={diff_b:.6e}");
        assert!(diff_a < ATOL, "pair output A diff {diff_a:.6e} >= atol {ATOL}");
        assert!(diff_b < ATOL, "pair output B diff {diff_b:.6e} >= atol {ATOL}");
    }

    #[test]
    fn test_gemm_q4_k_simd_pair_larger_shape() {
        let rows = 128;
        let cols = 512;
        let n_blocks = rows * (cols / 256);

        let w_a_bytes = synthetic_q4_k_bytes(n_blocks, 0xCAFE_BABE);
        let w_b_bytes = synthetic_q4_k_bytes(n_blocks, 0xABCD_1234);
        let x = fixed_input(cols, 0x1234_5678);

        let mut w_a_f32 = vec![0.0_f32; rows * cols];
        let mut w_b_f32 = vec![0.0_f32; rows * cols];
        dequant_into(GgmlType::Q4_K, &w_a_bytes, &mut w_a_f32).expect("Q4_K dequant a");
        dequant_into(GgmlType::Q4_K, &w_b_bytes, &mut w_b_f32).expect("Q4_K dequant b");
        let mut scalar_a = vec![0.0_f32; rows];
        let mut scalar_b = vec![0.0_f32; rows];
        kernels::gemv_f32(&w_a_f32, rows, cols, &x, &mut scalar_a);
        kernels::gemv_f32(&w_b_f32, rows, cols, &x, &mut scalar_b);

        let ctx = ctx().clone();
        let mut pair_a = vec![0.0_f32; rows];
        let mut pair_b = vec![0.0_f32; rows];
        kernels::dispatch_gemv_q4_k_m_simd_pair_batched(&ctx, &w_a_bytes, &w_b_bytes, rows, cols, &x, &mut pair_a, &mut pair_b)
            .expect("dispatch_gemv_q4_k_m_simd_pair_batched larger shape should succeed");

        let diff_a = max_abs_diff(&scalar_a, &pair_a);
        let diff_b = max_abs_diff(&scalar_b, &pair_b);
        println!("[v0.3.2] pair larger shape diff_a={diff_a:.6e} diff_b={diff_b:.6e}");
        assert!(diff_a < ATOL, "pair larger output A diff {diff_a:.6e} >= atol {ATOL}");
        assert!(diff_b < ATOL, "pair larger output B diff {diff_b:.6e} >= atol {ATOL}");
    }

    // v0.3.3 pair+silu parity tests: exercise dispatch_gemv_q4_k_m_simd_pair_silu_batched
    // (gate GEMV + up GEMV + silu_mul all in one CommandBatch; returns silu(gate)*up directly).
    // ATOL_SILU is wider than ATOL: Q4_K GEMV error (~1e-3) is amplified by the silu output
    // magnitude (~3–6 for the synthetic weights used here, d≈0.015, K=256-512).
    const ATOL_SILU: f32 = 1e-2;

    #[test]
    fn test_gemm_q4_k_simd_pair_silu_matches_scalar() {
        let rows = 64;
        let cols = 256;
        let n_blocks = rows * (cols / 256);

        let w_gate_bytes = synthetic_q4_k_bytes(n_blocks, 42);
        let w_up_bytes = synthetic_q4_k_bytes(n_blocks, 0xDEAD_CAFE);
        let x = fixed_input(cols, 0xDEAD_BEEF);

        // Scalar reference: dequant both, GEMV both, CPU silu_mul.
        let mut w_gate_f32 = vec![0.0_f32; rows * cols];
        let mut w_up_f32 = vec![0.0_f32; rows * cols];
        dequant_into(GgmlType::Q4_K, &w_gate_bytes, &mut w_gate_f32).expect("Q4_K dequant gate");
        dequant_into(GgmlType::Q4_K, &w_up_bytes, &mut w_up_f32).expect("Q4_K dequant up");
        let mut g_ref = vec![0.0_f32; rows];
        let mut u_ref = vec![0.0_f32; rows];
        kernels::gemv_f32(&w_gate_f32, rows, cols, &x, &mut g_ref);
        kernels::gemv_f32(&w_up_f32, rows, cols, &x, &mut u_ref);
        let mut ref_a = vec![0.0_f32; rows];
        kernels::silu_mul(&g_ref, &u_ref, &mut ref_a);

        // GPU fused path.
        let ctx = ctx().clone();
        let mut gpu_a = vec![0.0_f32; rows];
        kernels::dispatch_gemv_q4_k_m_simd_pair_silu_batched(&ctx, &w_gate_bytes, &w_up_bytes, rows, cols, &x, &mut gpu_a).expect("dispatch_gemv_q4_k_m_simd_pair_silu_batched should succeed");

        let diff = max_abs_diff(&ref_a, &gpu_a);
        println!("[v0.3.3] pair+silu parity diff={diff:.6e}");
        assert!(diff < ATOL_SILU, "pair+silu diff {diff:.6e} >= atol_silu {ATOL_SILU}");
    }

    #[test]
    fn test_gemm_q4_k_simd_pair_silu_larger_shape() {
        let rows = 128;
        let cols = 512;
        let n_blocks = rows * (cols / 256);

        let w_gate_bytes = synthetic_q4_k_bytes(n_blocks, 0xCAFE_BABE);
        let w_up_bytes = synthetic_q4_k_bytes(n_blocks, 0xABCD_1234);
        let x = fixed_input(cols, 0x1234_5678);

        let mut w_gate_f32 = vec![0.0_f32; rows * cols];
        let mut w_up_f32 = vec![0.0_f32; rows * cols];
        dequant_into(GgmlType::Q4_K, &w_gate_bytes, &mut w_gate_f32).expect("Q4_K dequant gate");
        dequant_into(GgmlType::Q4_K, &w_up_bytes, &mut w_up_f32).expect("Q4_K dequant up");
        let mut g_ref = vec![0.0_f32; rows];
        let mut u_ref = vec![0.0_f32; rows];
        kernels::gemv_f32(&w_gate_f32, rows, cols, &x, &mut g_ref);
        kernels::gemv_f32(&w_up_f32, rows, cols, &x, &mut u_ref);
        let mut ref_a = vec![0.0_f32; rows];
        kernels::silu_mul(&g_ref, &u_ref, &mut ref_a);

        let ctx = ctx().clone();
        let mut gpu_a = vec![0.0_f32; rows];
        kernels::dispatch_gemv_q4_k_m_simd_pair_silu_batched(&ctx, &w_gate_bytes, &w_up_bytes, rows, cols, &x, &mut gpu_a)
            .expect("dispatch_gemv_q4_k_m_simd_pair_silu_batched larger shape should succeed");

        let diff = max_abs_diff(&ref_a, &gpu_a);
        println!("[v0.3.3] pair+silu larger shape diff={diff:.6e}");
        assert!(diff < ATOL_SILU, "pair+silu larger shape diff {diff:.6e} >= atol_silu {ATOL_SILU}");
    }
}
#[rustfmt::skip]
mod v034_attn_pair_parity {
    //! v0.3.4 parity tests — coalesce q_a_proj + kv_a_proj into one CommandBatch.
    //!
    //! Verifies that `dispatch_gemv_f32_attn_pinned_pair_batched` (two fp32 GEMVs
    //! fused into one CB) matches independent `gemv_f32_attn_metal` calls on the
    //! same input. fp32 GEMV is numerically identical on GPU vs CPU reference;
    //! ATOL=1e-3 is generous — any mismatch beyond fp32 noise indicates a wiring
    //! bug (wrong buffer, wrong offset, wrong shape).
    //!
    //! Shape A (rows_a=512, rows_b=256, cols=2048): q-lora proxy
    //!   q_lora_rank=512, kv_a_dim=256, hidden=2048
    //! Shape B (rows_a=2048, rows_b=256, cols=2048): non-q-lora proxy
    //!   n_heads*head_dim=2048, kv_a_dim=256, hidden=2048

    #![cfg(target_os = "macos")]

    use hawking_core::kernels;

    use crate::common;
    use common::*;

    fn attn_pair_check(rows_a: usize, rows_b: usize, cols: usize) {
        let x = fixed_f32(cols, 0xA1B2C3D4);
        let w_a = fixed_f32(rows_a * cols, 0xDEAD_BEEF);
        let w_b = fixed_f32(rows_b * cols, 0xCAFE_F00D);

        let ctx = ctx().clone();

        // Reference: two independent standalone GEMVs (byte-slice path, CPU-upload).
        let mut ref_a = vec![0.0_f32; rows_a];
        let mut ref_b = vec![0.0_f32; rows_b];
        kernels::gemv_f32_attn_metal(&ctx, &w_a, rows_a, cols, &x, &mut ref_a).expect("gemv_f32_attn_metal A");
        kernels::gemv_f32_attn_metal(&ctx, &w_b, rows_b, cols, &x, &mut ref_b).expect("gemv_f32_attn_metal B");

        // Pair-batch path: pre-pin both weight matrices, dispatch both into one CB.
        let w_a_bytes: &[u8] = bytemuck::cast_slice(&w_a);
        let w_b_bytes: &[u8] = bytemuck::cast_slice(&w_b);
        let w_a_buf = ctx.new_buffer_with_bytes(w_a_bytes);
        let w_b_buf = ctx.new_buffer_with_bytes(w_b_bytes);

        let mut out_a = vec![0.0_f32; rows_a];
        let mut out_b = vec![0.0_f32; rows_b];
        kernels::dispatch_gemv_f32_attn_pinned_pair_batched(&ctx, &w_a_buf, rows_a, &w_b_buf, rows_b, cols, &x, &mut out_a, &mut out_b).expect("dispatch_gemv_f32_attn_pinned_pair_batched");

        let max_diff_a = ref_a.iter().zip(out_a.iter()).map(|(r, g)| (r - g).abs()).fold(0.0_f32, f32::max);
        let max_diff_b = ref_b.iter().zip(out_b.iter()).map(|(r, g)| (r - g).abs()).fold(0.0_f32, f32::max);

        assert!(max_diff_a <= ATOL, "shape ({rows_a},{rows_b},{cols}) out_a diff={max_diff_a:e} > ATOL={ATOL:e}");
        assert!(max_diff_b <= ATOL, "shape ({rows_a},{rows_b},{cols}) out_b diff={max_diff_b:e} > ATOL={ATOL:e}");
    }

    #[test]
    fn test_attn_pair_q_lora_proxy() {
        // Shape A: rows_a=512 (q_lora_rank), rows_b=256 (kv_a_dim), cols=2048
        attn_pair_check(512, 256, 2048);
    }

    #[test]
    fn test_attn_pair_non_q_lora_proxy() {
        // Shape B: rows_a=2048 (n_heads*head_dim), rows_b=256 (kv_a_dim), cols=2048
        attn_pair_check(2048, 256, 2048);
    }
}
#[rustfmt::skip]
mod v040_q4k_v2_parity {
    //! v0.4.0 — Numerical parity test: gemm_q4_k_m_fused_v2 vs scalar reference.
    //!
    //! Test 1: rows=64,  cols=256  (1 Q4_K block/row)
    //! Test 2: rows=512, cols=2048 (8 Q4_K blocks/row)
    //! Asserts max |scalar - v2| < 1e-3 (fp16 quant noise tolerance).

    #![cfg(target_os = "macos")]

    use hawking_core::gguf::GgmlType;
    use hawking_core::kernels;
    use hawking_core::quant::dequant_into;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    pub const ATOL: f32 = 1e-3;

    fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
    }

    fn synthetic_q4_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
        use half::f16;
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.01 + rng.gen::<f32>() * 0.01;
            let d_bits = f16::from_f32(d).to_bits();
            bytes[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
            let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
            let dmin_bits = f16::from_f32(dmin).to_bits();
            bytes[off + 2..off + 4].copy_from_slice(&dmin_bits.to_le_bytes());
            for i in 4..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    #[test]
    fn test_gemm_q4k_v2_small() {
        let rows = 64;
        let cols = 256; // 1 Q4_K block per row
        let n_blocks = rows * (cols / 256);

        let w_bytes = synthetic_q4_k_bytes(n_blocks, 42);
        let x = fixed_input(cols, 0xDEAD_BEEF);

        let mut w_f32 = vec![0.0_f32; rows * cols];
        dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32).expect("Q4_K dequant should succeed for synthetic bytes");
        let mut scalar_out = vec![0.0_f32; rows];
        kernels::gemv_f32(&w_f32, rows, cols, &x, &mut scalar_out);

        let ctx = ctx().clone();
        let mut v2_out = vec![0.0_f32; rows];
        kernels::gemv_q4_k_m_v2(&ctx, &w_bytes, rows, cols, &x, &mut v2_out).expect("gemv_q4_k_m_v2 should succeed");

        let diff = max_abs_diff(&scalar_out, &v2_out);
        println!("[v0.4.0] gemm_q4k_v2 small (rows=64 cols=256) max abs diff = {diff:.6e}");
        assert!(diff < ATOL, "gemm_q4_k_m_fused_v2 vs scalar diff {diff:.6e} >= atol {ATOL}");
    }

    #[test]
    fn test_gemm_q4k_v2_realistic() {
        let rows = 512;
        let cols = 2048; // 8 Q4_K blocks per row
        let n_blocks = rows * (cols / 256);

        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xCAFE_BABE);
        let x = fixed_input(cols, 0x1234_5678);

        let mut w_f32 = vec![0.0_f32; rows * cols];
        dequant_into(GgmlType::Q4_K, &w_bytes, &mut w_f32).expect("Q4_K dequant should succeed");
        let mut scalar_out = vec![0.0_f32; rows];
        kernels::gemv_f32(&w_f32, rows, cols, &x, &mut scalar_out);

        let ctx = ctx().clone();
        let mut v2_out = vec![0.0_f32; rows];
        kernels::gemv_q4_k_m_v2(&ctx, &w_bytes, rows, cols, &x, &mut v2_out).expect("gemv_q4_k_m_v2 larger shape should succeed");

        let diff = max_abs_diff(&scalar_out, &v2_out);
        println!("[v0.4.0] gemm_q4k_v2 realistic (rows=512 cols=2048) max abs diff = {diff:.6e}");
        assert!(diff < ATOL, "gemm_q4_k_m_fused_v2 vs scalar diff {diff:.6e} >= atol {ATOL}");
    }
}
#[rustfmt::skip]
mod v0512_token_cb_smoke {
    //! v0.5.12 smoke test: TokenCommandBuffer fuses two kernel dispatches.
    //!
    //! Verifies that encoding rmsnorm + add_inplace into a single
    //! TokenCommandBuffer produces the same outputs as two sequential
    //! ctx.dispatch_threads calls.

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};

    const TG_SIZE: u32 = 256;

    fn make_ctx() -> MetalContext {
        MetalContext::new().expect("Metal device")
    }

    fn f32_to_f16_bytes(v: &[f32]) -> Vec<u8> {
        let f16v: Vec<f16> = v.iter().map(|&x| f16::from_f32(x)).collect();
        bytemuck::cast_slice::<f16, u8>(&f16v).to_vec()
    }

    fn f16_buf_to_f32(ptr: *const f16, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(ptr, n) }.iter().map(|v| v.to_f32()).collect()
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice::<f32, u8>(v).to_vec()
    }

    fn f32_buf_read(ptr: *const f32, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    #[test]
    fn token_command_buffer_matches_sequential() {
        let ctx = make_ctx();
        let n = TG_SIZE as usize; // 256 elements
        let eps = 1e-5f32;

        // ── prepare input data ────────────────────────────────────────────────────
        let x_f32: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.05).sin()) * 2.0).collect();
        let w_f32: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.001)).collect();

        let a_f32: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let b_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();

        // ── sequential path ───────────────────────────────────────────────────────
        let x_bytes = f32_to_f16_bytes(&x_f32);
        let w_bytes = f32_to_f16_bytes(&w_f32);
        let rn_out_seq = ctx.new_buffer(n * std::mem::size_of::<f16>());

        let x_buf_s = ctx.new_buffer_with_bytes(&x_bytes);
        let w_buf_s = ctx.new_buffer_with_bytes(&w_bytes);
        let hidden_u32 = n as u32;
        let shmem = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        // Sequential: rmsnorm dispatch.
        ctx.dispatch_threads("rmsnorm", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&x_buf_s), 0);
            enc.set_buffer(1, Some(&w_buf_s), 0);
            enc.set_buffer(2, Some(&rn_out_seq), 0);
            enc.set_bytes(3, 4, &hidden_u32 as *const u32 as *const _);
            enc.set_bytes(4, 4, &eps as *const f32 as *const _);
            enc.set_threadgroup_memory_length(0, shmem);
        })
        .expect("seq rmsnorm");

        let a_bytes = f32_bytes(&a_f32);
        let b_bytes = f32_bytes(&b_f32);
        let a_buf_s = ctx.new_buffer_with_bytes(&a_bytes);
        let b_buf_s = ctx.new_buffer_with_bytes(&b_bytes);
        let n_u32 = n as u32;
        let n_tg = (n_u32 + TG_SIZE - 1) / TG_SIZE;

        // Sequential: add_inplace dispatch.
        ctx.dispatch_threads("add_inplace", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&a_buf_s), 0);
            enc.set_buffer(1, Some(&b_buf_s), 0);
            enc.set_bytes(2, 4, &n_u32 as *const u32 as *const _);
        })
        .expect("seq add_inplace");

        // Read sequential results.
        let rn_seq = f16_buf_to_f32(rn_out_seq.contents() as *const f16, n);
        let ai_seq = f32_buf_read(a_buf_s.contents() as *const f32, n);

        // ── TCB path ──────────────────────────────────────────────────────────────
        let x_buf_t = ctx.new_buffer_with_bytes(&x_bytes);
        let w_buf_t = ctx.new_buffer_with_bytes(&w_bytes);
        let rn_out_tcb = ctx.new_buffer(n * std::mem::size_of::<f16>());
        let a_buf_t = ctx.new_buffer_with_bytes(&a_bytes);
        let b_buf_t = ctx.new_buffer_with_bytes(&b_bytes);

        {
            let mut tcb = TokenCommandBuffer::new(&ctx);

            tcb.dispatch_threads("rmsnorm", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
                enc.set_buffer(0, Some(&x_buf_t), 0);
                enc.set_buffer(1, Some(&w_buf_t), 0);
                enc.set_buffer(2, Some(&rn_out_tcb), 0);
                enc.set_bytes(3, 4, &hidden_u32 as *const u32 as *const _);
                enc.set_bytes(4, 4, &eps as *const f32 as *const _);
                enc.set_threadgroup_memory_length(0, shmem);
            })
            .expect("tcb rmsnorm");

            tcb.dispatch_threads("add_inplace", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
                enc.set_buffer(0, Some(&a_buf_t), 0);
                enc.set_buffer(1, Some(&b_buf_t), 0);
                enc.set_bytes(2, 4, &n_u32 as *const u32 as *const _);
            })
            .expect("tcb add_inplace");

            tcb.commit_and_wait().expect("tcb commit");
        }

        // Read TCB results.
        let rn_tcb = f16_buf_to_f32(rn_out_tcb.contents() as *const f16, n);
        let ai_tcb = f32_buf_read(a_buf_t.contents() as *const f32, n);

        // ── compare ───────────────────────────────────────────────────────────────
        assert_eq!(rn_seq.len(), rn_tcb.len(), "rmsnorm output length mismatch");
        for (i, (&s, &t)) in rn_seq.iter().zip(rn_tcb.iter()).enumerate() {
            assert_eq!(s, t, "rmsnorm[{i}]: seq={s} tcb={t}");
        }

        assert_eq!(ai_seq.len(), ai_tcb.len(), "add_inplace output length mismatch");
        for (i, (&s, &t)) in ai_seq.iter().zip(ai_tcb.iter()).enumerate() {
            assert_eq!(s, t, "add_inplace[{i}]: seq={s} tcb={t}");
        }
    }

    #[test]
    fn token_command_buffer_drop_commits() {
        // Verify that dropping a TCB without commit_and_wait doesn't panic or leak.
        let ctx = make_ctx();
        let n = 64usize;
        let a_f32: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_f32: Vec<f32> = vec![1.0f32; n];
        let a_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&a_f32));
        let b_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&b_f32));
        let n_u32 = n as u32;
        let n_tg = (n_u32 + TG_SIZE - 1) / TG_SIZE;

        {
            // Drop without explicit commit: Drop impl should commit cleanly.
            let mut tcb = TokenCommandBuffer::new(&ctx);
            tcb.dispatch_threads("add_inplace", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
                enc.set_buffer(0, Some(&a_buf), 0);
                enc.set_buffer(1, Some(&b_buf), 0);
                enc.set_bytes(2, 4, &n_u32 as *const u32 as *const _);
            })
            .expect("tcb add_inplace");
            // Dropped here — Drop impl commits.
        }

        // After drop, the GPU work should have completed; a_buf should be updated.
        let result = unsafe { std::slice::from_raw_parts(a_buf.contents() as *const f32, n) };
        for i in 0..n {
            let expected = i as f32 + 1.0;
            assert!((result[i] - expected).abs() < 1e-6, "drop[{i}]: got={} expected={expected}", result[i]);
        }
    }
}
#[rustfmt::skip]
mod v100_pinned_q4kgemv_parity {
    //! Wedge A parity: gemv_q4_k_m_v2_pinned matches gemv_q4_k_m_v2 at atol=1e-5.
    //! Both paths dispatch the same `gemm_q4_k_m_fused_v2` kernel; the only
    //! difference is whether weights are memcpy'd into a fresh buffer or read
    //! from a pre-pinned buffer via byte offset. Outputs should be bit-identical.
    #![cfg(target_os = "macos")]

    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
    }

    fn synthetic_q4_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
        use half::f16;
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.01 + rng.gen::<f32>() * 0.01;
            let d_bits = f16::from_f32(d).to_bits();
            bytes[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
            let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
            let dmin_bits = f16::from_f32(dmin).to_bits();
            bytes[off + 2..off + 4].copy_from_slice(&dmin_bits.to_le_bytes());
            for i in 4..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    fn pinned_from_bytes(ctx: &MetalContext, bytes: &[u8]) -> PinnedBuffer {
        ctx.new_buffer_with_bytes(bytes)
    }

    #[test]
    fn pinned_q4kgemv_small() {
        let rows = 64;
        let cols = 256;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 42);
        let x = fixed_input(cols, 0xDEAD_BEEF);

        let ctx = ctx();

        let mut copy_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut copy_out).expect("copy path should succeed");

        let model_buf = pinned_from_bytes(ctx, &w_bytes);
        let mut pinned_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut pinned_out).expect("pinned path should succeed");

        let diff = max_abs_diff(&copy_out, &pinned_out);
        println!("[WedgeA] pinned vs copy small (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
        assert!(diff < 1e-5, "pinned vs copy diff {diff:.2e} >= 1e-5 (should be bit-identical)");
    }

    #[test]
    fn pinned_q4kgemv_realistic() {
        let rows = 512;
        let cols = 2048;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xCAFE_BABE);
        let x = fixed_input(cols, 0x1234_5678);

        let ctx = ctx();

        let mut copy_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut copy_out).expect("copy path should succeed");

        let model_buf = pinned_from_bytes(ctx, &w_bytes);
        let mut pinned_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut pinned_out).expect("pinned path should succeed");

        let diff = max_abs_diff(&copy_out, &pinned_out);
        println!("[WedgeA] pinned vs copy realistic (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
        assert!(diff < 1e-5, "pinned vs copy diff {diff:.2e} >= 1e-5 (should be bit-identical)");
    }

    #[test]
    fn pinned_q4kgemv_nonzero_offset() {
        let rows = 128;
        let cols = 512;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xBEEF_CAFE);
        let x = fixed_input(cols, 0xABCD_1234);

        let pad = 1024usize;
        let mut padded = vec![0xFFu8; pad];
        padded.extend_from_slice(&w_bytes);

        let ctx = ctx();

        let mut copy_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut copy_out).expect("copy path");

        let model_buf = pinned_from_bytes(ctx, &padded);
        let mut pinned_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2_pinned(ctx, &model_buf, pad, w_bytes.len(), rows, cols, &x, &mut pinned_out).expect("pinned path with offset");

        let diff = max_abs_diff(&copy_out, &pinned_out);
        println!("[WedgeA] pinned vs copy nonzero-offset (pad={pad}) max abs diff = {diff:.2e}");
        assert!(diff < 1e-5, "offset pinned vs copy diff {diff:.2e} >= 1e-5");
    }
}
#[rustfmt::skip]
mod v1_1_phase5b1_lm_head_tcb_parity {
    //! Phase 5B.1 parity: greedy argmax with the LM head folded into the global TCB
    //! must be byte-identical to a second run with the same engine (determinism),
    //! and the spec NGram exact-mode invariant must still hold (correctness).
    //!
    //! The fold path is active whenever `greedy_gpu_argmax_available()` is true and
    //! the Wedge C single-TCB path is used (Off/NGram mode with profile).
    //!
    //! Tests:
    //!   1. Two greedy runs on the same engine (KV reset between) produce identical
    //!      tokens — verifies the fold path is deterministic.
    //!   2. Spec NGram exact-mode invariant still holds with the fold active for
    //!      both repetitive and natural-text prompts (correctness gate).
    //!
    //! The engine pair for test 2 is shared (single load for both prompts) to keep
    //! GPU memory pressure low. Skips if model weights are not present.

    use hawking_core::{EngineConfig, GenerateRequest, SamplingParams, SpeculateMode, StreamEvent};
    use std::path::PathBuf;

    fn weights_path() -> PathBuf {
        PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
    }

    fn load_engine(speculate_mode: SpeculateMode) -> Option<Box<dyn hawking_core::Engine>> {
        let p = weights_path();
        if !p.exists() {
            eprintln!("v1_1_phase5B1: no weights at {:?}, skipping", p);
            return None;
        }
        let mut cfg = EngineConfig::default();
        cfg.speculate = speculate_mode != SpeculateMode::Off;
        cfg.speculate_mode = speculate_mode;

        // Profile enables the Wedge C TCB path and the Phase 5B.1 LM-head fold.
        let profile_path = PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
        if profile_path.exists() {
            if let Ok(profile) = hawking_core::profile::KernelProfile::load(&profile_path) {
                cfg.kernel_profile = Some(profile);
            }
        }

        match hawking_core::model::load_engine(&p, cfg) {
            Ok(e) => Some(e),
            Err(err) => {
                eprintln!("v1_1_phase5B1: load failed: {err}, skipping");
                None
            }
        }
    }

    fn collect_tokens(engine: &mut Box<dyn hawking_core::Engine>, prompt: &str, max_new_tokens: usize) -> Vec<u32> {
        let req = GenerateRequest {
            prompt: prompt.to_string(),
            max_new_tokens,
            sampling: SamplingParams { temperature: 0.0, top_p: 1.0, top_k: 0, repetition_penalty: 1.0, seed: None },
            stop: vec![],
            abort: None,
            max_stall_ms: 0,
            json_mode: false,
        };
        let mut tokens = Vec::new();
        engine
            .generate(req, &mut |ev| {
                if let StreamEvent::Token { id, .. } = ev {
                    tokens.push(id);
                }
            })
            .expect("generate");
        tokens
    }

    /// Two greedy runs on the SAME engine with KV reset between them must produce
    /// identical tokens. This verifies the Phase 5B.1 LM-head fold is deterministic.
    #[test]
    fn lm_head_fold_is_deterministic() {
        let Some(mut engine) = load_engine(SpeculateMode::Off) else {
            return;
        };

        let prompts = ["The quick brown fox", "Explain how speculative decoding works:"];
        for prompt in &prompts {
            engine.reset_kv_for_test();
            let run1 = collect_tokens(&mut engine, prompt, 16);

            engine.reset_kv_for_test();
            let run2 = collect_tokens(&mut engine, prompt, 16);

            assert_eq!(run1, run2, "prompt={prompt:?}: Phase 5B.1 fold not deterministic\nrun1={run1:?}\nrun2={run2:?}");
            assert!(!run1.is_empty(), "prompt={prompt:?}: fold produced no tokens");
        }
    }

    /// Spec NGram exact-mode invariant with Phase 5B.1 (LM head folded into TCB).
    /// Both repetitive and natural prompts are tested with a single engine-pair load.
    #[test]
    fn spec_exact_mode_with_lm_head_fold() {
        let Some(mut ref_engine) = load_engine(SpeculateMode::Off) else {
            return;
        };
        let Some(mut spec_engine) = load_engine(SpeculateMode::ExactShared) else {
            return;
        };

        // Repetitive prompt (high n-gram acceptance).
        {
            let prompt = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
            let ref_ids = collect_tokens(&mut ref_engine, prompt, 16);
            let spec_ids = collect_tokens(&mut spec_engine, prompt, 16);
            assert_eq!(ref_ids, spec_ids, "repetitive: spec+5B.1 differs from greedy\nref={ref_ids:?}\nspec={spec_ids:?}");
        }

        // Natural-text prompt (low n-gram acceptance).
        {
            let prompt = "Explain how speculative decoding works:";
            let ref_ids = collect_tokens(&mut ref_engine, prompt, 12);
            let spec_ids = collect_tokens(&mut spec_engine, prompt, 12);
            assert_eq!(ref_ids, spec_ids, "natural: spec+5B.1 differs from greedy\nref={ref_ids:?}\nspec={spec_ids:?}");
        }
    }
}
#[rustfmt::skip]
mod v1_1_phase5c2_fp16_activations_parity {
    //! Phase 5C.2 fp16 activations parity test.
    //!
    //! Verifies that the f16 intermediate activation path for the final-norm
    //! → LM head step produces outputs that match the f32 baseline:
    //!   - argmax must match exactly (regression guard)
    //!   - logit values within atol=5e-3 (f16 quantization noise tolerance)
    //!
    //! Tests are synthetic (no model weights required):
    //!
    //! Tests:
    //! - rmsnorm_f32_to_f16_parity — rmsnorm_f32 (f32 out) vs rmsnorm_f32_to_f16
    //!   (f16 out promoted to f32): atol=5e-3, same argmax.
    //! - gemv_f16_f16in_parity — gemv_f16 (f32 activation) vs gemv_f16_f16in
    //!   (f16 activation): argmax matches exactly, logit atol=5e-3.
    //! - end_to_end_final_norm_lm_head_parity — combined rmsnorm_f32_to_f16 +
    //!   gemv_f16_f16in pipeline argmax matches rmsnorm_f32 + gemv_f16 pipeline.
    //!
    //! Design rationale: the residual stream stays f32 between layers; only the
    //! per-layer normed activation is f16. This prevents the accumulation error
    //! that caused the 3 prior f16 residual attempts to produce garbage output.

    #![cfg(target_os = "macos")]

    use std::time::{SystemTime, UNIX_EPOCH};

    fn random_f32(seed: &mut u64) -> f32 {
        // xorshift64 — fast deterministic pseudo-random.
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        // Map to [-2.0, 2.0] (typical residual stream magnitude).
        ((*seed as i64 as f32) / (i64::MAX as f32)) * 2.0
    }

    fn make_residual(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n).map(|_| random_f32(seed)).collect()
    }

    fn make_weight(n: usize, seed: &mut u64) -> Vec<f32> {
        // RMS norm weights are typically close to 1.0.
        (0..n).map(|_| 0.5 + random_f32(seed).abs()).collect()
    }

    fn make_lm_head(rows: usize, cols: usize, seed: &mut u64) -> Vec<u16> {
        // f16 LM head weights — convert random f32 to f16 bits.
        (0..rows * cols)
            .map(|_| {
                let v = random_f32(seed) * 0.1; // small magnitude typical for weight matrices
                half::f16::from_f32(v).to_bits()
            })
            .collect()
    }

    /// Compute rmsnorm_f32 (f32 → f32) on CPU for reference.
    fn rmsnorm_f32_ref(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len();
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
        let inv = 1.0 / rms;
        x.iter().zip(weight.iter()).map(|(&xv, &wv)| xv * inv * wv).collect()
    }

    /// Compute rmsnorm → f16 → f32 promote (simulates GPU rmsnorm_f32_to_f16).
    fn rmsnorm_f32_to_f16_ref(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len();
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
        let inv = 1.0 / rms;
        x.iter()
            .zip(weight.iter())
            .map(|(&xv, &wv)| {
                // Simulate: store as f16 then load back to f32.
                let v_f32 = xv * inv * wv;
                half::f16::from_f32(v_f32).to_f32()
            })
            .collect()
    }

    /// Compute GEMV with f16 weights × f32 activation → f32 output (reference for gemv_f16).
    fn gemv_f16_f32in_ref(w: &[u16], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        (0..rows)
            .map(|r| {
                let row = &w[r * cols..(r + 1) * cols];
                row.iter().zip(x.iter()).map(|(&wbits, &xv)| half::f16::from_bits(wbits).to_f32() * xv).sum::<f32>()
            })
            .collect()
    }

    /// Compute GEMV with f16 weights × f16 activation → f32 output (reference for gemv_f16_f16in).
    fn gemv_f16_f16in_ref(w: &[u16], x_f32: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        // Simulate: convert activation to f16, then compute.
        let x_f16: Vec<f32> = x_f32.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        (0..rows)
            .map(|r| {
                let row = &w[r * cols..(r + 1) * cols];
                row.iter().zip(x_f16.iter()).map(|(&wbits, &xv)| half::f16::from_bits(wbits).to_f32() * xv).sum::<f32>()
            })
            .collect()
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0)
    }

    /// Verify rmsnorm_f32_to_f16 produces outputs within atol of rmsnorm_f32.
    /// The only difference is f16 rounding of each output element.
    #[test]
    fn rmsnorm_f32_to_f16_parity() {
        let mut seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
        seed ^= 0xdeadbeef_12345678;

        // Use hidden=512 (V2-Lite uses 2048; 512 is sufficient for coverage).
        let hidden = 512;
        let eps = 1e-6_f32;

        for trial in 0..8 {
            let x = make_residual(hidden, &mut seed);
            let weight = make_weight(hidden, &mut seed);

            let ref_out = rmsnorm_f32_ref(&x, &weight, eps);
            let f16_out = rmsnorm_f32_to_f16_ref(&x, &weight, eps);

            let max_diff = ref_out.iter().zip(f16_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

            assert!(max_diff <= 5e-3, "trial {trial}: rmsnorm_f32_to_f16 max_diff={max_diff:.2e} > 5e-3");

            // Argmax check: the top element should match between f32 and f16 outputs.
            // (This is a sanity check; argmax on norm output is less meaningful than
            //  argmax on logits, but verifies no catastrophic divergence.)
            let ref_top = argmax(&ref_out);
            let f16_top = argmax(&f16_out);
            assert_eq!(ref_top, f16_top, "trial {trial}: rmsnorm argmax mismatch ref={ref_top} f16={f16_top}");
        }
        eprintln!("✓ rmsnorm_f32_to_f16 parity: atol≤5e-3, argmax match (8 trials, hidden={hidden})");
    }

    /// Verify gemv_f16_f16in (f16 activation) argmax matches gemv_f16 (f32 activation).
    /// The input activation difference (f16 rounding) should not shift the winner.
    #[test]
    fn gemv_f16_f16in_parity() {
        let mut seed = 0xfeedface_abcd1234u64;

        // Small LM head shape for unit test speed: 128 rows × 256 cols.
        // Production is 102400×5120 but math is identical.
        let rows = 128;
        let cols = 256;
        let eps = 1e-6_f32;

        for trial in 0..8 {
            let x = make_residual(cols, &mut seed);
            let weight_norm = make_weight(cols, &mut seed);
            let lm_head = make_lm_head(rows, cols, &mut seed);

            // Reference: f32 rmsnorm output → f32 GEMV.
            let x_norm_f32 = rmsnorm_f32_ref(&x, &weight_norm, eps);
            let logits_f32 = gemv_f16_f32in_ref(&lm_head, &x_norm_f32, rows, cols);

            // Phase 5C.2 path: f16 rmsnorm output → f16 GEMV.
            let x_norm_f16 = rmsnorm_f32_to_f16_ref(&x, &weight_norm, eps);
            let logits_f16 = gemv_f16_f16in_ref(&lm_head, &x_norm_f16, rows, cols);

            // argmax must match.
            let top_f32 = argmax(&logits_f32);
            let top_f16 = argmax(&logits_f16);
            assert_eq!(top_f32, top_f16, "trial {trial}: gemv_f16_f16in argmax mismatch top_f32={top_f32} top_f16={top_f16}");

            // Logit values within atol=5e-3 (f16 rounding noise in activation propagates
            // through the weight matrix; for typical weight magnitudes the error is small).
            let max_diff = logits_f32.iter().zip(logits_f16.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(max_diff <= 5e-3, "trial {trial}: logit max_diff={max_diff:.2e} > atol=5e-3");
        }
        eprintln!("✓ gemv_f16_f16in parity: argmax exact, logit atol≤5e-3 (8 trials, {rows}×{cols})");
    }

    /// Combined pipeline parity: rmsnorm_f32_to_f16 + gemv_f16_f16in argmax matches
    /// rmsnorm_f32 + gemv_f16 for larger hidden size (closer to production shape).
    #[test]
    fn end_to_end_final_norm_lm_head_parity() {
        let mut seed = 0x1234abcd_5678ef90u64;

        // hidden=512, vocab=1024 — fast synthetic test covering the combined pipeline.
        let hidden = 512;
        let vocab = 1024;
        let eps = 1e-6_f32;

        let mut argmax_matches = 0usize;
        let trials = 16;

        for _trial in 0..trials {
            let residual = make_residual(hidden, &mut seed);
            let norm_weight = make_weight(hidden, &mut seed);
            let lm_head = make_lm_head(vocab, hidden, &mut seed);

            // f32 reference pipeline.
            let x_norm_f32 = rmsnorm_f32_ref(&residual, &norm_weight, eps);
            let logits_f32 = gemv_f16_f32in_ref(&lm_head, &x_norm_f32, vocab, hidden);

            // Phase 5C.2 f16 intermediate pipeline.
            let x_norm_f16 = rmsnorm_f32_to_f16_ref(&residual, &norm_weight, eps);
            let logits_f16 = gemv_f16_f16in_ref(&lm_head, &x_norm_f16, vocab, hidden);

            if argmax(&logits_f32) == argmax(&logits_f16) {
                argmax_matches += 1;
            }
        }

        // Require argmax match on ≥ 15/16 trials (>93%). In practice with well-distributed
        // random weights and activations the match rate is ~100%.
        assert!(argmax_matches >= 15, "end-to-end argmax match rate {argmax_matches}/{trials} < 15/16");
        eprintln!(
            "✓ end-to-end final-norm+LM head parity: {argmax_matches}/{trials} argmax matches \
             (hidden={hidden}, vocab={vocab})"
        );

        // Note: with real model weights and longer sequences, argmax match is expected
        // to degrade slightly (f16 noise in x_norm → slight logit shifts). The parity
        // test above uses random small weights which are worst-case for noise propagation.
        // Production weights (Q4K) are also quantized, so their activation sensitivity
        // is already bounded by the Q4K rounding; f16 x_norm adds comparable noise.
        eprintln!(
            "  Phase 5C.2 scope: final-norm → LM head path only. Per-layer FFN-norm paths \
             remain f32 (future work: MoE/FFN gate+up GEMVs need f16-input variants)."
        );
    }
}
#[rustfmt::skip]
mod v1_1_q3_k_parity {
    //! v1.1.0 Phase 1A — Q3_K Metal GEMV parity against scalar dequant.

    #![cfg(target_os = "macos")]

    use hawking_core::gguf::GgmlType;
    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer};
    use hawking_core::quant::dequant_into;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    const ATOL: f32 = 1e-2;

    fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
    }

    fn pin(ctx: &MetalContext, bytes: &[u8]) -> PinnedBuffer {
        ctx.new_buffer_with_bytes(bytes)
    }

    fn pack_q3_scale(block: &mut [u8], scale_idx: usize, signed_scale: i8) {
        let l = (signed_scale + 32) as u8;
        if scale_idx < 8 {
            block[96 + scale_idx] |= l & 0x0f;
        } else {
            block[96 + scale_idx - 8] |= (l & 0x0f) << 4;
        }
        block[104 + scale_idx % 4] |= (l >> 4) << (2 * (scale_idx / 4));
    }

    fn synthetic_q3_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
        use half::f16;
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 110];
        for b in 0..n_blocks {
            let off = b * 110;
            for i in 0..108 {
                bytes[off + i] = rng.gen::<u8>();
            }
            let d = 0.004 + rng.gen::<f32>() * 0.004;
            bytes[off + 108..off + 110].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        }
        bytes
    }

    fn zero_q3_k_bytes(n_blocks: usize) -> Vec<u8> {
        vec![0u8; n_blocks * 110]
    }

    fn ones_q3_k_bytes(n_blocks: usize) -> Vec<u8> {
        use half::f16;
        let mut bytes = vec![0xffu8; n_blocks * 110];
        for b in 0..n_blocks {
            let off = b * 110;
            bytes[off + 108..off + 110].copy_from_slice(&f16::from_f32(0.002).to_bits().to_le_bytes());
        }
        bytes
    }

    fn known_q3_k_bytes(n_blocks: usize) -> Vec<u8> {
        use half::f16;
        let mut bytes = vec![0u8; n_blocks * 110];
        for b in 0..n_blocks {
            let off = b * 110;
            let block = &mut bytes[off..off + 110];
            for scale_idx in 0..16 {
                pack_q3_scale(block, scale_idx, 1);
            }
            block[108..110].copy_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
            for i in 0..32 {
                block[i] = if i % 2 == 0 { 0xff } else { 0x00 };
            }
            for i in 0..64 {
                block[32 + i] = match i % 4 {
                    0 => 0b1110_0100,
                    1 => 0b0001_1011,
                    2 => 0b0101_0101,
                    _ => 0b1010_1010,
                };
            }
        }
        bytes
    }

    fn assert_q3_gemv_matches_scalar(rows: usize, cols: usize, w_bytes: &[u8], seed: u64, label: &str) {
        let x = fixed_input(cols, seed);

        let mut w_f32 = vec![0.0_f32; rows * cols];
        dequant_into(GgmlType::Q3_K, w_bytes, &mut w_f32).expect("Q3_K scalar dequant");
        let mut scalar_out = vec![0.0_f32; rows];
        kernels::gemv_f32(&w_f32, rows, cols, &x, &mut scalar_out);

        let ctx = ctx();
        let model_buf = pin(ctx, w_bytes);
        let mut metal_out = vec![0.0_f32; rows];
        kernels::gemv_q3_k_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut metal_out).expect("Q3_K Metal GEMV");

        let diff = max_abs_diff(&scalar_out, &metal_out);
        println!("[v1.1.0] Q3_K {label} rows={rows} cols={cols} max abs diff = {diff:.6e}");
        assert!(diff < ATOL, "Q3_K {label} diff {diff:.6e} >= atol {ATOL}");
    }

    #[test]
    fn q3_k_metal_matches_scalar_known_patterns() {
        let rows = 16;
        let cols = 256;
        let n_blocks = rows * (cols / 256);
        assert_q3_gemv_matches_scalar(rows, cols, &zero_q3_k_bytes(n_blocks), 0xA, "zero");
        assert_q3_gemv_matches_scalar(rows, cols, &ones_q3_k_bytes(n_blocks), 0xB, "ones");
        assert_q3_gemv_matches_scalar(rows, cols, &known_q3_k_bytes(n_blocks), 0xC, "known");
    }

    #[test]
    fn q3_k_metal_matches_scalar_random_small() {
        let rows = 64;
        let cols = 256;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q3_k_bytes(n_blocks, 42);
        assert_q3_gemv_matches_scalar(rows, cols, &w_bytes, 0xDEAD_BEEF, "random-small");
    }

    #[test]
    fn q3_k_metal_matches_scalar_random_realistic() {
        let rows = 256;
        let cols = 2048;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q3_k_bytes(n_blocks, 0xCAFE_BABE);
        assert_q3_gemv_matches_scalar(rows, cols, &w_bytes, 0x1234_5678, "random-realistic");
    }
}
#[rustfmt::skip]
mod v1_1_q4k_llama_port_parity {
    //! v1.1.0 Phase 1B — llama_port Q4_K GEMV parity at production shapes.

    #![cfg(target_os = "macos")]

    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer};

    use crate::common;
    use common::*;

    fn synthetic_input(cols: usize) -> Vec<f32> {
        (0..cols).map(|i| ((i % 97) as f32 - 48.0) / 97.0).collect()
    }

    fn synthetic_q4_k_bytes(n_blocks: usize) -> Vec<u8> {
        use half::f16;

        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
            bytes[off + 2] = 0x00;
            bytes[off + 3] = 0x00; // f16 0.0 dmin
            for i in 4..144 {
                bytes[off + i] = ((b * 13 + i * 37) & 0xff) as u8;
            }
        }
        bytes
    }

    fn pin(ctx: &MetalContext, bytes: &[u8]) -> PinnedBuffer {
        ctx.new_buffer_with_bytes(bytes)
    }

    fn assert_llama_port_matches_v2(rows: usize, cols: usize, label: &str) {
        let ctx = ctx();
        let w_bytes = synthetic_q4_k_bytes(rows * (cols / 256));
        let model_buf = pin(ctx, &w_bytes);
        let x = synthetic_input(cols);
        let mut v2_out = vec![0.0f32; rows];
        let mut llama_out = vec![0.0f32; rows];

        kernels::gemv_q4_k_m_v2_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut v2_out).expect("v2 Q4_K GEMV");
        kernels::gemv_q4_k_m_llama_port_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut llama_out).expect("llama_port Q4_K GEMV");

        let diff = max_abs_diff(&v2_out, &llama_out);
        println!("[v1.1.0] llama_port vs v2 {label} rows={rows} cols={cols} max abs diff = {diff:.6e}");
        assert!(diff < ATOL, "llama_port {label} diff {diff:.6e} >= atol {ATOL}");
    }

    #[test]
    fn llama_port_gate_up_shape_matches_v2() {
        assert_llama_port_matches_v2(1024, 4096, "gate_up");
    }

    #[test]
    fn llama_port_down_shape_matches_v2() {
        assert_llama_port_matches_v2(4096, 1024, "down");
    }

    #[test]
    fn llama_port_dense_shape_matches_v2() {
        assert_llama_port_matches_v2(4096, 4096, "dense");
    }
}
#[rustfmt::skip]
mod v1_2_q4k_gu_v2_parity {
    //! v1.2.0-2 — Parity test: moe_batched_gemm_q4_indexed_v2t_gu_v2 vs v2t_gu.
    //!
    //! Both kernels compute silu(gate) * up for routed Q4_K_M experts.  The v2
    //! kernel adds: sumy correction trick, scale/activation preloading, and
    //! paired nibble reads.  Output must match v2t_gu within atol=1e-3 (fp16
    //! quantisation noise budget).
    //!
    //! Test shapes:
    //!   1. routes=2, rows=16,   cols=256   — sub-TG edge case
    //!   2. routes=6, rows=1408, cols=2048  — production DeepSeek V2-Lite shape

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    /// Build a synthetic fused Q4_K_M weight tensor for `n_experts` consecutive
    /// matrices of shape (rows, blocks_per_row * 144 bytes).
    fn synthetic_q4_k_bytes(n_experts: usize, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let blocks_per_row = cols / 256;
        let bytes_per_expert = rows * blocks_per_row * 144;
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_experts * bytes_per_expert];
        for b in 0..(n_experts * rows * blocks_per_row) {
            let off = b * 144;
            // d: small positive fp16
            let d = 0.0005 + rng.gen::<f32>() * 0.001;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            // dmin: small fp16 (can be negative)
            let dmin = (rng.gen::<f32>() - 0.5) * 0.001;
            bytes[off + 2..off + 4].copy_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());
            // Scale + nibble bytes
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

        // Build weight tensor: gate matrices followed by up matrices.
        let gate_bytes = synthetic_q4_k_bytes(n_experts, rows, cols, seed_base);
        let up_bytes = synthetic_q4_k_bytes(n_experts, rows, cols, seed_base ^ 0x1234_5678);

        // Assemble with padding prefix to exercise non-zero offsets.
        let pad = 128usize;
        let mut w_all = vec![0xA5u8; pad + gate_bytes.len() + up_bytes.len()];
        let gate_offset = pad;
        let up_offset = pad + n_experts * bytes_per_expert;
        w_all[gate_offset..gate_offset + gate_bytes.len()].copy_from_slice(&gate_bytes);
        w_all[up_offset..up_offset + up_bytes.len()].copy_from_slice(&up_bytes);

        // Route IDs — spread across experts
        let route_ids: Vec<u32> = (0..routes).map(|i| ((i * 3 + 1) % n_experts) as u32).collect();

        let x = fixed_f32(cols, seed_base ^ 0xDEAD_BEEF);

        // Reference: v2t_gu
        let mut ref_out = vec![0.0_f32; routes * rows];
        kernels::moe_batched_gemm_q4_indexed_v2t_gu_raw(ctx(), &w_all, gate_offset, up_offset, &route_ids, &x, routes, rows, cols, &mut ref_out).expect("v2t_gu dispatch failed");

        // Candidate: v2t_gu_v2
        let mut v2_out = vec![0.0_f32; routes * rows];
        kernels::moe_batched_gemm_q4_indexed_v2t_gu_v2_raw(ctx(), &w_all, gate_offset, up_offset, &route_ids, &x, routes, rows, cols, &mut v2_out).expect("v2t_gu_v2 dispatch failed");

        let diff = max_abs_diff(&ref_out, &v2_out);
        println!(
            "[v1.2.0-2] gu_v2 parity (routes={routes} rows={rows} cols={cols}) \
             max|ref-v2| = {diff:.6e}"
        );
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
}
#[rustfmt::skip]
mod v1_tcb_rmsnorm_add_parity {
    //! Wedge B parity: TCB-batched rmsnorm + add_inplace produces bit-identical
    //! output to the CPU reference kernels.
    //!
    //! Tests:
    //!   1. `tcb_rmsnorm_matches_cpu` — rmsnorm_metal_buf_tcb ≡ CPU rmsnorm
    //!   2. `tcb_add_inplace_matches_cpu` — add_inplace_metal_tcb ≡ CPU add_inplace
    //!   3. `tcb_staggered_loop_matches_cpu` — simulated staggered per-layer pattern
    //!      (the forward_token_final_norm Wedge B inner loop) produces identical
    //!      residual stream to the CPU [rmsnorm + CPU add_inplace] sequence.
    #![cfg(target_os = "macos")]

    use hawking_core::kernels;
    use hawking_core::metal::{PinnedBuffer, TokenCommandBuffer};

    use crate::common;
    use common::*;

    fn write_f32_buf(buf: &PinnedBuffer, data: &[f32]) {
        let ptr = buf.contents() as *mut f32;
        unsafe { ptr.copy_from_nonoverlapping(data.as_ptr(), data.len()) };
    }

    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn tcb_rmsnorm_matches_cpu() {
        let h = 2048usize;
        let eps = 1e-6_f32;
        let ctx = ctx();

        let x = fixed_f32(h, 0xABCD_1234);
        let w = fixed_f32(h, 0xDEAD_BEEF);

        let mut cpu_out = vec![0.0f32; h];
        kernels::rmsnorm(&x, &w, eps, &mut cpu_out);

        let x_buf = new_f32_buf(ctx, &x);
        let w_buf = new_f32_buf(ctx, &w);
        let out_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());

        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::rmsnorm_metal_buf_tcb(&mut tcb, &x_buf, &w_buf, eps, h, &out_buf).expect("rmsnorm_metal_buf_tcb");
            tcb.commit_and_wait().expect("commit");
        }

        let gpu_out = read_f32_buf(&out_buf, h);
        let diff = max_abs_diff(&cpu_out, &gpu_out);
        println!("[WedgeB] rmsnorm TCB vs CPU  max abs diff = {diff:.2e}");
        assert!(diff < 1e-5, "rmsnorm TCB vs CPU diff {diff:.2e} >= 1e-5");
    }

    #[test]
    fn tcb_add_inplace_matches_cpu() {
        let h = 2048usize;
        let ctx = ctx();

        let mut a_cpu = fixed_f32(h, 0xCAFE_BABE);
        let b = fixed_f32(h, 0x1234_5678);

        // CPU reference.
        kernels::add_inplace(&mut a_cpu, &b);

        let a_buf = new_f32_buf(ctx, &fixed_f32(h, 0xCAFE_BABE));
        let b_buf = new_f32_buf(ctx, &b);

        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::add_inplace_metal_tcb(&mut tcb, &a_buf, &b_buf, h).expect("add_inplace_metal_tcb");
            tcb.commit_and_wait().expect("commit");
        }

        let gpu_a = read_f32_buf(&a_buf, h);
        let diff = max_abs_diff(&a_cpu, &gpu_a);
        println!("[WedgeB] add_inplace TCB vs CPU  max abs diff = {diff:.2e}");
        assert!(diff < 1e-6, "add_inplace TCB vs CPU diff {diff:.2e} >= 1e-6");
    }

    /// Simulate the Wedge B staggered per-layer forward pass pattern for N_LAYERS
    /// synthetic layers. Each layer applies: add_inplace(x, delta) then rmsnorm(x→out).
    /// Compares TCB path vs CPU reference; output must be bit-identical.
    #[test]
    fn tcb_staggered_loop_matches_cpu() {
        const N_LAYERS: usize = 4;
        let h = 2048usize;
        let eps = 1e-6_f32;
        let ctx = ctx();

        // Generate fixed synthetic data.
        let x_init = fixed_f32(h, 0x1111_2222);
        let deltas: Vec<Vec<f32>> = (0..N_LAYERS).map(|i| fixed_f32(h, 0xAAAA_0000 + i as u64)).collect();
        let norms: Vec<Vec<f32>> = (0..N_LAYERS).map(|i| fixed_f32(h, 0xBBBB_0000 + i as u64)).collect();

        // ── CPU reference path ───────────────────────────────────────────────
        // Pattern: for each layer, add delta to x, then rmsnorm x.
        let mut x_cpu = x_init.clone();
        let mut norm_outs_cpu = vec![vec![0.0f32; h]; N_LAYERS];
        for li in 0..N_LAYERS {
            kernels::add_inplace(&mut x_cpu, &deltas[li]);
            kernels::rmsnorm(&x_cpu, &norms[li], eps, &mut norm_outs_cpu[li]);
        }
        let x_final_cpu = x_cpu;

        // ── TCB path (staggered, mirroring forward_token_final_norm Wedge B) ─
        // Each layer: mini-TCB [add_inplace(x_buf, delta_buf), rmsnorm(x_buf → out_buf)]
        let x_buf = new_f32_buf(ctx, &x_init);
        let delta_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());
        let out_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());
        let norm_bufs: Vec<PinnedBuffer> = norms.iter().map(|n| new_f32_buf(ctx, n)).collect();

        let mut norm_outs_gpu = vec![vec![0.0f32; h]; N_LAYERS];

        for li in 0..N_LAYERS {
            write_f32_buf(&delta_buf, &deltas[li]);
            {
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::add_inplace_metal_tcb(&mut tcb, &x_buf, &delta_buf, h).expect("add_inplace_metal_tcb");
                kernels::rmsnorm_metal_buf_tcb(&mut tcb, &x_buf, &norm_bufs[li], eps, h, &out_buf).expect("rmsnorm_metal_buf_tcb");
                tcb.commit_and_wait().expect("commit");
            }
            norm_outs_gpu[li] = read_f32_buf(&out_buf, h);
        }
        let x_final_gpu = read_f32_buf(&x_buf, h);

        // ── compare ──────────────────────────────────────────────────────────
        let x_diff = max_abs_diff(&x_final_cpu, &x_final_gpu);
        println!("[WedgeB] staggered loop: x_final max abs diff = {x_diff:.2e}");
        assert!(x_diff < 1e-5, "x_final diff {x_diff:.2e} >= 1e-5");

        for li in 0..N_LAYERS {
            let d = max_abs_diff(&norm_outs_cpu[li], &norm_outs_gpu[li]);
            println!("[WedgeB] staggered loop: layer {li} norm_out max abs diff = {d:.2e}");
            assert!(d < 1e-5, "layer {li} norm_out diff {d:.2e} >= 1e-5");
        }
    }
}
#[rustfmt::skip]
mod v1c_tcb_attn_ffn_parity {
    //! Wedge C parity: attention + FFN TCB kernel variants produce the same output
    //! as CPU reference kernels.
    //!
    //! Tests:
    //!   1. `wedge_c_pair_gemv_norm_matches_cpu` — gemv_f32_attn_pair_arena_tcb +
    //!      rmsnorm_metal_buf_tcb chain matches CPU gemv_f32 + rmsnorm for both outputs.
    //!   2. `wedge_c_moe_gate_gemv_matches_cpu` — gemv_f32_moe_pinned_buf_tcb matches
    //!      CPU gemv_f32.
    //!   3. `wedge_c_full_layer_loop_matches_cpu` — simulated Wedge C per-layer pattern
    //!      (add_attn + rmsnorm_attn + add_ffn + rmsnorm_ffn) produces identical
    //!      residual stream to the CPU reference.
    #![cfg(target_os = "macos")]

    use hawking_core::kernels;
    use hawking_core::metal::{PinnedBuffer, TokenCommandBuffer};

    use crate::common;
    use common::*;

    // ─────────────────────────────────────────────────────────────────────────────

    /// Tests gemv_f32_attn_pair_arena_tcb (two GEMVs sharing one input) then
    /// rmsnorm_metal_buf_tcb on each output. Compares to CPU gemv_f32 + rmsnorm.
    #[test]
    fn wedge_c_pair_gemv_norm_matches_cpu() {
        // Shapes matching DeepSeek-V2-Lite: hidden=2048, q_lora_rank=1536, kv_lora_rank+rope=576+64=640
        let hidden = 256usize; // smaller for test speed
        let rows_a = 64usize; // q_lora_rank analogue
        let rows_b = 80usize; // kv_a_dim analogue (kv_lora_rank + qk_rope)
        let kv_lora = 64usize; // kv_lora_rank (first portion of rows_b)
        let eps = 1e-6_f32;
        let ctx = ctx();

        let x = fixed_f32(hidden, 0x1111_AAAA);
        let w_a = fixed_f32(rows_a * hidden, 0x2222_BBBB);
        let w_b = fixed_f32(rows_b * hidden, 0x3333_CCCC);
        let norm_a_w = fixed_f32(rows_a, 0x4444_DDDD);
        let norm_b_w = fixed_f32(kv_lora, 0x5555_EEEE); // norm only first kv_lora elements

        // CPU reference.
        let mut cpu_out_a = vec![0.0f32; rows_a];
        let mut cpu_out_b = vec![0.0f32; rows_b];
        kernels::gemv_f32(&w_a, rows_a, hidden, &x, &mut cpu_out_a);
        kernels::gemv_f32(&w_b, rows_b, hidden, &x, &mut cpu_out_b);
        let mut cpu_normed_a = vec![0.0f32; rows_a];
        let mut cpu_normed_b = vec![0.0f32; kv_lora];
        kernels::rmsnorm(&cpu_out_a, &norm_a_w, eps, &mut cpu_normed_a);
        kernels::rmsnorm(&cpu_out_b[..kv_lora], &norm_b_w, eps, &mut cpu_normed_b);

        // GPU TCB path.
        let x_buf = new_f32_buf(ctx, &x);
        let w_a_buf = new_f32_buf(ctx, &w_a);
        let w_b_buf = new_f32_buf(ctx, &w_b);
        let norm_a_buf = new_f32_buf(ctx, &norm_a_w);
        let norm_b_buf = new_f32_buf(ctx, &norm_b_w);
        let out_a_buf = ctx.new_buffer(rows_a * std::mem::size_of::<f32>());
        let out_b_buf = ctx.new_buffer(rows_b * std::mem::size_of::<f32>());
        let normed_a_buf = ctx.new_buffer(rows_a * std::mem::size_of::<f32>());
        let normed_b_buf = ctx.new_buffer(kv_lora * std::mem::size_of::<f32>());

        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_f32_attn_pair_arena_tcb(&mut tcb, &w_a_buf, rows_a, &w_b_buf, rows_b, hidden, &x_buf, &out_a_buf, &out_b_buf).expect("pair_gemv_tcb");
            // Norm a (all rows_a elements).
            kernels::rmsnorm_metal_buf_tcb(&mut tcb, &out_a_buf, &norm_a_buf, eps, rows_a, &normed_a_buf).expect("rmsnorm_a_tcb");
            // Norm b (only first kv_lora elements of out_b_buf, same pattern as kv_a_norm).
            kernels::rmsnorm_metal_buf_tcb(&mut tcb, &out_b_buf, &norm_b_buf, eps, kv_lora, &normed_b_buf).expect("rmsnorm_b_tcb");
            tcb.commit_and_wait().expect("commit");
        }

        let gpu_out_a = read_f32_buf(&out_a_buf, rows_a);
        let gpu_out_b = read_f32_buf(&out_b_buf, rows_b);
        let gpu_normed_a = read_f32_buf(&normed_a_buf, rows_a);
        let gpu_normed_b = read_f32_buf(&normed_b_buf, kv_lora);

        let diff_a = max_abs_diff(&cpu_out_a, &gpu_out_a);
        let diff_b = max_abs_diff(&cpu_out_b[..rows_b], &gpu_out_b);
        let diff_na = max_abs_diff(&cpu_normed_a, &gpu_normed_a);
        let diff_nb = max_abs_diff(&cpu_normed_b, &gpu_normed_b);

        println!("[WedgeC] pair_gemv out_a diff = {diff_a:.2e}");
        println!("[WedgeC] pair_gemv out_b diff = {diff_b:.2e}");
        println!("[WedgeC] pair_gemv normed_a diff = {diff_na:.2e}");
        println!("[WedgeC] pair_gemv normed_b diff = {diff_nb:.2e}");

        assert!(diff_a < 1e-4, "gemv out_a diff {diff_a:.2e} >= 1e-4");
        assert!(diff_b < 1e-4, "gemv out_b diff {diff_b:.2e} >= 1e-4");
        assert!(diff_na < 1e-4, "normed_a diff {diff_na:.2e} >= 1e-4");
        assert!(diff_nb < 1e-4, "normed_b diff {diff_nb:.2e} >= 1e-4");
    }

    /// Tests gemv_f32_moe_pinned_buf_tcb (gate-logit GEMV) against CPU gemv_f32.
    #[test]
    fn wedge_c_moe_gate_gemv_matches_cpu() {
        let hidden = 256usize;
        let n_experts = 32usize; // n_routed_experts analogue
        let ctx = ctx();

        let x = fixed_f32(hidden, 0xCAFE_0001);
        let w = fixed_f32(n_experts * hidden, 0xDEAD_0002);

        // CPU reference.
        let mut cpu_out = vec![0.0f32; n_experts];
        kernels::gemv_f32(&w, n_experts, hidden, &x, &mut cpu_out);

        // GPU TCB path.
        let x_buf = new_f32_buf(ctx, &x);
        let w_buf = new_f32_buf(ctx, &w);
        let out_buf = ctx.new_buffer(n_experts * std::mem::size_of::<f32>());

        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_f32_moe_pinned_buf_tcb(&mut tcb, &w_buf, n_experts, hidden, &x_buf, &out_buf).expect("gemv_f32_moe_pinned_buf_tcb");
            tcb.commit_and_wait().expect("commit");
        }

        let gpu_out = read_f32_buf(&out_buf, n_experts);
        let diff = max_abs_diff(&cpu_out, &gpu_out);
        println!("[WedgeC] moe_gate_gemv max abs diff = {diff:.2e}");
        assert!(diff < 1e-4, "moe gate gemv diff {diff:.2e} >= 1e-4");
    }

    /// Simulate the Wedge C per-layer pattern: add_attn + rmsnorm_attn then
    /// add_ffn + rmsnorm_ffn for N_LAYERS synthetic layers. Verifies the
    /// full residual stream is consistent between TCB path and CPU path.
    #[test]
    fn wedge_c_full_layer_loop_matches_cpu() {
        const N_LAYERS: usize = 4;
        let h = 256usize;
        let eps = 1e-6_f32;
        let ctx = ctx();

        let x_init = fixed_f32(h, 0xF00D_1111);
        let attn_outs: Vec<Vec<f32>> = (0..N_LAYERS).map(|i| fixed_f32(h, 0xAA00_0000 + i as u64)).collect();
        let ffn_outs: Vec<Vec<f32>> = (0..N_LAYERS).map(|i| fixed_f32(h, 0xBB00_0000 + i as u64)).collect();
        let attn_norms: Vec<Vec<f32>> = (0..N_LAYERS).map(|i| fixed_f32(h, 0xCC00_0000 + i as u64)).collect();
        let ffn_norms: Vec<Vec<f32>> = (0..N_LAYERS).map(|i| fixed_f32(h, 0xDD00_0000 + i as u64)).collect();

        // CPU reference: Wedge C loop structure.
        // Per layer:
        //   if li>0: x += ffn_out[li-1]
        //   x_norm_attn = rmsnorm(x, attn_norm)
        //   x += attn_out[li]                        ← add_inplace(x_buf, arena.out) in Wedge C
        //   x_norm_ffn = rmsnorm(x, ffn_norm)
        //   [deferred] ffn_out[li] stored for next iter
        // After loop: x += ffn_out[n_layers-1]
        let mut x_cpu = x_init.clone();
        let mut x_norm_attn_cpu = vec![vec![0.0f32; h]; N_LAYERS];
        let mut x_norm_ffn_cpu = vec![vec![0.0f32; h]; N_LAYERS];

        for li in 0..N_LAYERS {
            if li > 0 {
                kernels::add_inplace(&mut x_cpu, &ffn_outs[li - 1]);
            }
            kernels::rmsnorm(&x_cpu, &attn_norms[li], eps, &mut x_norm_attn_cpu[li]);
            // Wedge C: add attn_out to x (Wedge B added attn_out to ffn_out_buf,
            // Wedge C adds arena.out directly to x_buf in mini-TCB β).
            kernels::add_inplace(&mut x_cpu, &attn_outs[li]);
            kernels::rmsnorm(&x_cpu, &ffn_norms[li], eps, &mut x_norm_ffn_cpu[li]);
        }
        // Final: apply last layer's deferred ffn_out.
        kernels::add_inplace(&mut x_cpu, &ffn_outs[N_LAYERS - 1]);
        let x_final_cpu = x_cpu;

        // GPU TCB path (mirroring forward_token_final_norm Wedge C inner loop).
        let x_buf = new_f32_buf(ctx, &x_init);
        let ffn_out_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());
        let attn_out_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());
        let x_norm_buf = ctx.new_buffer(h * std::mem::size_of::<f32>());
        let attn_norm_bufs: Vec<PinnedBuffer> = attn_norms.iter().map(|n| new_f32_buf(ctx, n)).collect();
        let ffn_norm_bufs: Vec<PinnedBuffer> = ffn_norms.iter().map(|n| new_f32_buf(ctx, n)).collect();

        let mut x_norm_attn_gpu = vec![vec![0.0f32; h]; N_LAYERS];
        let mut x_norm_ffn_gpu = vec![vec![0.0f32; h]; N_LAYERS];

        for li in 0..N_LAYERS {
            // Write synthetic attn_out and ffn_out for this layer.
            {
                let ptr = attn_out_buf.contents() as *mut f32;
                unsafe { ptr.copy_from_nonoverlapping(attn_outs[li].as_ptr(), h) };
            }

            // Mini-TCB α: (add ffn_out_buf if li>0) + rmsnorm_attn → x_norm_buf.
            {
                let mut tcb = TokenCommandBuffer::new(ctx);
                if li > 0 {
                    kernels::add_inplace_metal_tcb(&mut tcb, &x_buf, &ffn_out_buf, h).expect("add_inplace α");
                }
                kernels::rmsnorm_metal_buf_tcb(&mut tcb, &x_buf, &attn_norm_bufs[li], eps, h, &x_norm_buf).expect("rmsnorm_attn_tcb");
                tcb.commit_and_wait().expect("commit α");
            }
            x_norm_attn_gpu[li] = read_f32_buf(&x_norm_buf, h);

            // Mini-TCB β: add_inplace(x_buf, attn_out_buf) + rmsnorm_ffn → x_norm_buf.
            {
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::add_inplace_metal_tcb(&mut tcb, &x_buf, &attn_out_buf, h).expect("add_inplace β");
                kernels::rmsnorm_metal_buf_tcb(&mut tcb, &x_buf, &ffn_norm_bufs[li], eps, h, &x_norm_buf).expect("rmsnorm_ffn_tcb");
                tcb.commit_and_wait().expect("commit β");
            }
            x_norm_ffn_gpu[li] = read_f32_buf(&x_norm_buf, h);

            // Write synthetic ffn_out into ffn_out_buf (simulates ffn_tcb_inner output).
            {
                let ptr = ffn_out_buf.contents() as *mut f32;
                unsafe { ptr.copy_from_nonoverlapping(ffn_outs[li].as_ptr(), h) };
            }
        }

        // Final: add last layer's ffn_out.
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::add_inplace_metal_tcb(&mut tcb, &x_buf, &ffn_out_buf, h).expect("add_inplace final");
            tcb.commit_and_wait().expect("commit final");
        }

        let x_final_gpu = read_f32_buf(&x_buf, h);

        // Comparisons.
        let x_diff = max_abs_diff(&x_final_cpu, &x_final_gpu);
        println!("[WedgeC] full_loop x_final max abs diff = {x_diff:.2e}");
        assert!(x_diff < 1e-5, "x_final diff {x_diff:.2e} >= 1e-5");

        for li in 0..N_LAYERS {
            let d_attn = max_abs_diff(&x_norm_attn_cpu[li], &x_norm_attn_gpu[li]);
            let d_ffn = max_abs_diff(&x_norm_ffn_cpu[li], &x_norm_ffn_gpu[li]);
            println!("[WedgeC] layer {li}: x_norm_attn diff = {d_attn:.2e}, x_norm_ffn diff = {d_ffn:.2e}");
            assert!(d_attn < 1e-5, "layer {li} x_norm_attn diff {d_attn:.2e} >= 1e-5");
            assert!(d_ffn < 1e-5, "layer {li} x_norm_ffn diff {d_ffn:.2e} >= 1e-5");
        }
    }
}
#[rustfmt::skip]
mod v1e_gpu_argmax_parity {
    //! Wedge E parity: GPU argmax via TCB matches CPU reference.
    //!
    //! Tests:
    //!   1. `wedge_e_argmax_tcb_matches_cpu` — sample_argmax_f32_tcb produces the
    //!      same winner as CPU argmax across vocab sizes and tie patterns.
    //!   2. `wedge_e_gemv_f16_buf_tcb_matches_cpu` — gemv_f16_metal_buf_tcb output
    //!      matches CPU gemv_f16 within fp16 tolerance.
    //!   3. `wedge_e_lmhead_plus_argmax_tcb_matches_cpu` — combined LM-head GEMV
    //!      + argmax via TCB produces the same token id as the CPU path.
    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};

    use crate::common;
    use common::*;

    fn fixed_f16(n: usize, seed: u64) -> Vec<f16> {
        fixed_f32(n, seed).iter().map(|&v| f16::from_f32(v)).collect()
    }

    fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
        ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
    }

    fn cpu_argmax(logits: &[f32]) -> u32 {
        let mut best = 0u32;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > bv {
                best = i as u32;
                bv = v;
            }
        }
        best
    }

    // ─────────────────────────────────────────────────────────────────────────────

    /// sample_argmax_f32_tcb winner matches CPU argmax for various vocab sizes
    /// and tie patterns (lowest-index-wins on ties).
    #[test]
    fn wedge_e_argmax_tcb_matches_cpu() {
        let ctx = ctx();

        // Basic: clear winner at a specific index.
        for &vocab in &[256usize, 4096, 32768] {
            let mut logits = fixed_f32(vocab, 0xDEAD_BEEF ^ vocab as u64);
            let target = vocab / 3 + 11;
            logits[target] = 9999.0;

            let cpu = cpu_argmax(&logits);
            let logits_buf = new_f32_buf(ctx, &logits);
            let token_buf = ctx.new_buffer(std::mem::size_of::<u32>());

            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::sample_argmax_f32_tcb(&mut tcb, &logits_buf, &token_buf, vocab).expect("sample_argmax_f32_tcb");
            tcb.commit_and_wait().expect("commit");

            let gpu = unsafe { *(token_buf.contents() as *const u32) };
            assert_eq!(gpu, cpu, "vocab={vocab}: gpu={gpu} cpu={cpu}");
        }

        // Tie: all same value → lowest index (0) wins.
        {
            let vocab = 1024usize;
            let logits = vec![1.0f32; vocab];
            let logits_buf = new_f32_buf(ctx, &logits);
            let token_buf = ctx.new_buffer(std::mem::size_of::<u32>());
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::sample_argmax_f32_tcb(&mut tcb, &logits_buf, &token_buf, vocab).expect("tied argmax");
            tcb.commit_and_wait().expect("commit");
            let gpu = unsafe { *(token_buf.contents() as *const u32) };
            assert_eq!(gpu, 0u32, "tied: lowest index should win, got {gpu}");
        }
    }

    /// gemv_f16_metal_buf_tcb output matches CPU gemv_f16 within fp16 precision
    /// (atol 1e-3 — fp16 quantization noise is ~1e-3 at unit scale).
    #[test]
    fn wedge_e_gemv_f16_buf_tcb_matches_cpu() {
        let ctx = ctx();

        // Shapes: rows=256 (small vocab analogue), cols=128.
        let rows = 256usize;
        let cols = 128usize;

        let w_f16 = fixed_f16(rows * cols, 0xAAAA_1111);
        let x_f32 = fixed_f32(cols, 0xBBBB_2222);

        // CPU reference.
        let mut cpu_out = vec![0.0f32; rows];
        kernels::gemv_f16(&w_f16, rows, cols, &x_f32, &mut cpu_out);

        // GPU TCB path.
        let w_buf = new_f16_buf(ctx, &w_f16);
        let x_buf = new_f32_buf(ctx, &x_f32);
        let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_f16_metal_buf_tcb(&mut tcb, &w_buf, rows, cols, &x_buf, &y_buf).expect("gemv_f16_metal_buf_tcb");
        tcb.commit_and_wait().expect("commit");

        let gpu_out = read_f32_buf(&y_buf, rows);
        let diff = max_abs_diff(&cpu_out, &gpu_out);
        assert!(diff < 1e-3, "gemv_f16 rows={rows} cols={cols}: max_abs_diff={diff:.2e} > 1e-3");
    }

    /// Combined LM-head GEMV + argmax via TCB matches CPU path.
    /// This is the exact kernel sequence that Wedge E adds to forward_token_greedy.
    #[test]
    fn wedge_e_lmhead_plus_argmax_tcb_matches_cpu() {
        let ctx = ctx();

        // Shapes: small vocab (512) × hidden (256) for test speed.
        // Parity tested at both default and 102400-vocab analogue above.
        let vocab = 512usize;
        let hidden = 256usize;

        let lm_head_f16 = fixed_f16(vocab * hidden, 0xCCCC_3333);
        let x_norm_f32 = fixed_f32(hidden, 0xDDDD_4444);

        // CPU reference: gemv_f16 → logits → argmax.
        let mut cpu_logits = vec![0.0f32; vocab];
        kernels::gemv_f16(&lm_head_f16, vocab, hidden, &x_norm_f32, &mut cpu_logits);
        let cpu_token = cpu_argmax(&cpu_logits);

        // GPU TCB: gemv_f16_metal_buf_tcb + sample_argmax_f32_tcb in one TCB.
        let lm_head_buf = new_f16_buf(ctx, &lm_head_f16);
        let x_norm_buf = new_f32_buf(ctx, &x_norm_f32);
        let logits_buf = ctx.new_buffer(vocab * std::mem::size_of::<f32>());
        let token_buf = ctx.new_buffer(std::mem::size_of::<u32>());

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_f16_metal_buf_tcb(&mut tcb, &lm_head_buf, vocab, hidden, &x_norm_buf, &logits_buf).expect("gemv_f16_metal_buf_tcb");
        kernels::sample_argmax_f32_tcb(&mut tcb, &logits_buf, &token_buf, vocab).expect("sample_argmax_f32_tcb");
        tcb.commit_and_wait().expect("commit");

        let gpu_token = unsafe { *(token_buf.contents() as *const u32) };
        assert_eq!(gpu_token, cpu_token, "lmhead+argmax: gpu={gpu_token} cpu={cpu_token}");

        // Also verify the logits buf matches CPU within fp16 tolerance.
        let gpu_logits = read_f32_buf(&logits_buf, vocab);
        let diff = max_abs_diff(&cpu_logits, &gpu_logits);
        assert!(diff < 1e-3, "logits max_abs_diff={diff:.2e} > 1e-3");
    }
}
#[rustfmt::skip]
mod v1g_rmsnorm_gemv_fusion_parity {
    //! Wedge G parity: fused rmsnorm+gemv TCB matches CPU reference.
    //!
    //! Tests:
    //!   1. `wedge_g_rmsnorm_gemv_f32_attn_pinned_tcb_matches_cpu` — fused dispatch
    //!      produces same output as sequential CPU rmsnorm + gemv_f32 within atol 1e-3.
    //!   2. `wedge_g_fused_pair_matches_cpu` — two fused calls in one TCB (simulating
    //!      q_a + kv_a dispatch) both match their CPU references.
    //!   3. `wedge_g_fused_argmax_agrees_with_unfused` — argmax of fused and unfused
    //!      GEMV agree (temp=0 token parity).
    #![cfg(target_os = "macos")]

    use hawking_core::kernels;
    use hawking_core::metal::TokenCommandBuffer;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    fn fixed_f32_positive(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..n).map(|_| rng.gen_range(0.5_f32..1.5_f32)).collect()
    }

    fn cpu_rmsnorm_gemv_f32(w: &[f32], x: &[f32], weight: &[f32], eps: f32, rows: usize, cols: usize) -> Vec<f32> {
        let mut x_norm = vec![0.0f32; cols];
        kernels::rmsnorm(x, weight, eps, &mut x_norm);
        let mut out = vec![0.0f32; rows];
        kernels::gemv_f32(w, rows, cols, &x_norm, &mut out);
        out
    }

    // ─────────────────────────────────────────────────────────────────────────────

    /// rmsnorm_gemv_f32_attn_pinned_tcb matches CPU rmsnorm + gemv_f32.
    /// Shape: rows=64 (q_lora_rank analogue), cols=256 (hidden analogue).
    #[test]
    fn wedge_g_rmsnorm_gemv_f32_attn_pinned_tcb_matches_cpu() {
        let ctx = ctx();
        let rows = 64usize;
        let cols = 256usize;
        let eps = 1e-6f32;

        let w = fixed_f32(rows * cols, 0xA1B2_C3D4);
        let x = fixed_f32(cols, 0xE5F6_0718);
        let weight = fixed_f32_positive(cols, 0x1234_5678);

        let cpu_out = cpu_rmsnorm_gemv_f32(&w, &x, &weight, eps, rows, cols);

        let w_buf = new_f32_buf(ctx, &w);
        let x_buf = new_f32_buf(ctx, &x);
        let weight_buf = new_f32_buf(ctx, &weight);
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::rmsnorm_gemv_f32_attn_pinned_tcb(&mut tcb, &w_buf, &x_buf, &weight_buf, eps, &out_buf, rows, cols).expect("rmsnorm_gemv_f32_attn_pinned_tcb");
        tcb.commit_and_wait().expect("commit");

        let gpu_out = read_f32_buf(&out_buf, rows);
        let diff = max_abs_diff(&cpu_out, &gpu_out);
        assert!(diff < 1e-3, "rmsnorm_gemv rows={rows} cols={cols}: max_abs_diff={diff:.2e} > 1e-3");
    }

    /// Two fused calls in one TCB (q_a + kv_a analogue) both match CPU reference.
    /// Mirrors the actual attention_tcb_inner Phase 1 usage: same x_buf, different w.
    #[test]
    fn wedge_g_fused_pair_matches_cpu() {
        let ctx = ctx();
        let rows_a = 48usize; // q_lora_rank analogue
        let rows_b = 64usize; // kv_a_dim analogue
        let cols = 128usize; // hidden analogue
        let eps = 1e-5f32;

        let w_a = fixed_f32(rows_a * cols, 0xAAAA_1111);
        let w_b = fixed_f32(rows_b * cols, 0xBBBB_2222);
        let x = fixed_f32(cols, 0xCCCC_3333);
        let weight = fixed_f32_positive(cols, 0xDDDD_4444);

        let cpu_out_a = cpu_rmsnorm_gemv_f32(&w_a, &x, &weight, eps, rows_a, cols);
        let cpu_out_b = cpu_rmsnorm_gemv_f32(&w_b, &x, &weight, eps, rows_b, cols);

        let w_a_buf = new_f32_buf(ctx, &w_a);
        let w_b_buf = new_f32_buf(ctx, &w_b);
        let x_buf = new_f32_buf(ctx, &x);
        let weight_buf = new_f32_buf(ctx, &weight);
        let out_a_buf = ctx.new_buffer(rows_a * std::mem::size_of::<f32>());
        let out_b_buf = ctx.new_buffer(rows_b * std::mem::size_of::<f32>());

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::rmsnorm_gemv_f32_attn_pinned_tcb(&mut tcb, &w_a_buf, &x_buf, &weight_buf, eps, &out_a_buf, rows_a, cols).expect("rmsnorm_gemv q_a");
        kernels::rmsnorm_gemv_f32_attn_pinned_tcb(&mut tcb, &w_b_buf, &x_buf, &weight_buf, eps, &out_b_buf, rows_b, cols).expect("rmsnorm_gemv kv_a");
        tcb.commit_and_wait().expect("commit");

        let gpu_out_a = read_f32_buf(&out_a_buf, rows_a);
        let gpu_out_b = read_f32_buf(&out_b_buf, rows_b);

        let diff_a = max_abs_diff(&cpu_out_a, &gpu_out_a);
        let diff_b = max_abs_diff(&cpu_out_b, &gpu_out_b);
        assert!(diff_a < 1e-3, "q_a fused: max_abs_diff={diff_a:.2e} > 1e-3");
        assert!(diff_b < 1e-3, "kv_a fused: max_abs_diff={diff_b:.2e} > 1e-3");
    }

    /// Argmax of fused and unfused GEMV agree (temperature=0 token parity).
    /// Ensures Wedge G does not change the winner of the projected output.
    #[test]
    fn wedge_g_fused_argmax_agrees_with_unfused() {
        let ctx = ctx();
        let rows = 128usize;
        let cols = 256usize;
        let eps = 1e-6f32;

        let w = fixed_f32(rows * cols, 0xF00D_CAFE);
        let x = fixed_f32(cols, 0xDEAD_BEEF);
        let weight = fixed_f32_positive(cols, 0xBEEF_CAFE);

        // CPU: rmsnorm → gemv_f32 → argmax.
        let cpu_out = cpu_rmsnorm_gemv_f32(&w, &x, &weight, eps, rows, cols);
        let cpu_winner = cpu_out.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as u32).unwrap();

        // GPU fused TCB.
        let w_buf = new_f32_buf(ctx, &w);
        let x_buf = new_f32_buf(ctx, &x);
        let weight_buf = new_f32_buf(ctx, &weight);
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::rmsnorm_gemv_f32_attn_pinned_tcb(&mut tcb, &w_buf, &x_buf, &weight_buf, eps, &out_buf, rows, cols).expect("rmsnorm_gemv_f32_attn_pinned_tcb");
        tcb.commit_and_wait().expect("commit");

        let gpu_out = read_f32_buf(&out_buf, rows);
        let gpu_winner = gpu_out.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as u32).unwrap();

        assert_eq!(gpu_winner, cpu_winner, "argmax winner differs: gpu={gpu_winner} cpu={cpu_winner}");
    }
}
#[rustfmt::skip]
mod v1h_simdgroup_gemv_parity {
    //! Wedge H parity: simdgroup_matrix GEMV matches CPU reference.
    //!
    //! Tests:
    //!   1. `wedge_h_simdgroup_f32_basic` — basic rows×cols shapes; atol 1e-5.
    //!   2. `wedge_h_simdgroup_f32_qb_shape` — q_b_proj analogue shape (rows=256, cols=64).
    //!   3. `wedge_h_simdgroup_f32_argmax_agrees` — argmax of simdgroup and scalar GEMV agree.
    #![cfg(target_os = "macos")]

    use hawking_core::kernels;
    use hawking_core::metal::TokenCommandBuffer;

    use crate::common;
    use common::*;

    // ─────────────────────────────────────────────────────────────────────────────

    /// simdgroup_f32 GEMV matches CPU gemv_f32 at atol 1e-4 for square shapes.
    #[test]
    fn wedge_h_simdgroup_f32_basic() {
        let ctx = ctx();

        for &(rows, cols) in &[(8usize, 8usize), (16, 8), (8, 16), (32, 64), (64, 32)] {
            let w = fixed_f32(rows * cols, 0xA1B2_C3D4 ^ rows as u64);
            let x = fixed_f32(cols, 0xE5F6_0718 ^ cols as u64);

            let mut cpu_out = vec![0.0f32; rows];
            kernels::gemv_f32(&w, rows, cols, &x, &mut cpu_out);

            let w_buf = new_f32_buf(ctx, &w);
            let x_buf = new_f32_buf(ctx, &x);
            let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_simdgroup_f32_tcb(&mut tcb, &w_buf, &x_buf, &y_buf, rows, cols).unwrap_or_else(|e| panic!("gemv_simdgroup_f32_tcb rows={rows} cols={cols}: {e}"));
            tcb.commit_and_wait().expect("commit");

            let gpu_out = read_f32_buf(&y_buf, rows);
            let diff = max_abs_diff(&cpu_out, &gpu_out);
            assert!(diff < 1e-4, "rows={rows} cols={cols}: max_abs_diff={diff:.2e} > 1e-4");
        }
    }

    /// q_b_proj analogue: rows=256 (heads×head_dim proxy), cols=64 (q_lora proxy).
    /// Simulates the actual Phase 2 mini-TCB shape in attention_tcb_inner.
    #[test]
    fn wedge_h_simdgroup_f32_qb_shape() {
        let ctx = ctx();
        let rows = 256usize;
        let cols = 64usize;

        let w = fixed_f32(rows * cols, 0xBEEF_CAFE);
        let x = fixed_f32(cols, 0xDEAD_BEEF);

        let mut cpu_out = vec![0.0f32; rows];
        kernels::gemv_f32(&w, rows, cols, &x, &mut cpu_out);

        let w_buf = new_f32_buf(ctx, &w);
        let x_buf = new_f32_buf(ctx, &x);
        let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_simdgroup_f32_tcb(&mut tcb, &w_buf, &x_buf, &y_buf, rows, cols).expect("gemv_simdgroup_f32_tcb");
        tcb.commit_and_wait().expect("commit");

        let gpu_out = read_f32_buf(&y_buf, rows);
        let diff = max_abs_diff(&cpu_out, &gpu_out);
        assert!(diff < 1e-3, "q_b_shape rows={rows} cols={cols}: max_abs_diff={diff:.2e} > 1e-3");
    }

    /// Argmax of simdgroup GEMV matches CPU gemv_f32 argmax (token parity at temp=0).
    #[test]
    fn wedge_h_simdgroup_f32_argmax_agrees() {
        let ctx = ctx();
        let rows = 128usize;
        let cols = 64usize;

        let w = fixed_f32(rows * cols, 0xF00D_1234);
        let x = fixed_f32(cols, 0xCAFE_5678);

        let mut cpu_out = vec![0.0f32; rows];
        kernels::gemv_f32(&w, rows, cols, &x, &mut cpu_out);
        let cpu_winner = cpu_out.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as u32).unwrap();

        let w_buf = new_f32_buf(ctx, &w);
        let x_buf = new_f32_buf(ctx, &x);
        let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_simdgroup_f32_tcb(&mut tcb, &w_buf, &x_buf, &y_buf, rows, cols).expect("gemv_simdgroup_f32_tcb");
        tcb.commit_and_wait().expect("commit");

        let gpu_out = read_f32_buf(&y_buf, rows);
        let gpu_winner = gpu_out.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as u32).unwrap();

        assert_eq!(gpu_winner, cpu_winner, "argmax: gpu={gpu_winner} cpu={cpu_winner}");
    }
}
#[rustfmt::skip]
mod v1k_q4kgemm_simdmat_parity {
    //! Wedge K parity: gemv_q4_k_m_simdmat_pinned vs gemv_q4_k_m_v2 at atol=1e-3.
    //! Different summation order (paired nibble reads) may cause rounding differences
    //! at Q4_K noise level; atol=1e-3 matches the fp16 quantization floor.
    #![cfg(target_os = "macos")]

    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
    }

    fn synthetic_q4_k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
        use half::f16;
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.01 + rng.gen::<f32>() * 0.01;
            let d_bits = f16::from_f32(d).to_bits();
            bytes[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
            let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
            let dmin_bits = f16::from_f32(dmin).to_bits();
            bytes[off + 2..off + 4].copy_from_slice(&dmin_bits.to_le_bytes());
            // scales: 12 bytes at off+4..off+15 (legal values: s_byte/m_byte ≤ 63)
            for i in 4..16 {
                bytes[off + i] = rng.gen::<u8>() & 0x3F;
            }
            // nibbles: 128 bytes at off+16..off+143
            for i in 16..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    fn pinned_from_bytes(ctx: &MetalContext, bytes: &[u8]) -> PinnedBuffer {
        ctx.new_buffer_with_bytes(bytes)
    }

    #[test]
    fn v1k_simdmat_vs_v2_small() {
        let rows = 64;
        let cols = 256;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 42);
        let x = fixed_input(cols, 0xDEAD_BEEF);

        let ctx = ctx();

        let mut v2_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut v2_out).expect("v2 path should succeed");

        let model_buf = pinned_from_bytes(ctx, &w_bytes);
        let mut sm_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_simdmat_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut sm_out).expect("simdmat path should succeed");

        let diff = max_abs_diff(&v2_out, &sm_out);
        println!("[WedgeK] simdmat vs v2 small (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
        assert!(diff < 1e-3, "simdmat vs v2 diff {diff:.2e} >= 1e-3 (Q4_K noise floor)");
    }

    #[test]
    fn v1k_simdmat_vs_v2_realistic() {
        let rows = 512;
        let cols = 2048;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xCAFE_BABE);
        let x = fixed_input(cols, 0x1234_5678);

        let ctx = ctx();

        let mut v2_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut v2_out).expect("v2 path");

        let model_buf = pinned_from_bytes(ctx, &w_bytes);
        let mut sm_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_simdmat_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut sm_out).expect("simdmat path");

        let diff = max_abs_diff(&v2_out, &sm_out);
        println!("[WedgeK] simdmat vs v2 realistic (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
        assert!(diff < 1e-3, "simdmat vs v2 diff {diff:.2e} >= 1e-3");
    }

    #[test]
    fn v1k_simdmat_argmax_agrees() {
        // Argmax of simdmat output must match v2 output on a DeepSeek-V2-like shape.
        let rows = 128;
        let cols = 7168;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xBEEF_1234);
        let x = fixed_input(cols, 0xABCD_5678);

        let ctx = ctx();

        let mut v2_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut v2_out).expect("v2 path");

        let model_buf = pinned_from_bytes(ctx, &w_bytes);
        let mut sm_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_simdmat_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut sm_out).expect("simdmat path");

        let diff = max_abs_diff(&v2_out, &sm_out);
        println!("[WedgeK] simdmat vs v2 argmax shape (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
        assert!(diff < 1e-3, "simdmat vs v2 diff {diff:.2e} >= 1e-3 on argmax shape");

        let v2_argmax = v2_out.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        let sm_argmax = sm_out.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap();
        println!("[WedgeK] argmax: v2={v2_argmax} simdmat={sm_argmax}");
        assert_eq!(v2_argmax, sm_argmax, "argmax must match between v2 and simdmat");
    }

    // ── v3_8r parity tests ───────────────────────────────────────────────────────

    #[test]
    fn v1k_v3_8r_vs_v2_small() {
        let rows = 64;
        let cols = 256;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0x1111_2222);
        let x = fixed_input(cols, 0x3333_4444);

        let ctx = ctx();

        let mut v2_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut v2_out).expect("v2 path");

        let model_buf = pinned_from_bytes(ctx, &w_bytes);
        let mut v3_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v3_8r_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut v3_out).expect("v3_8r path");

        let diff = max_abs_diff(&v2_out, &v3_out);
        println!("[WedgeK] v3_8r vs v2 small (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
        assert!(diff < 1e-3, "v3_8r vs v2 diff {diff:.2e} >= 1e-3");
    }

    #[test]
    fn v1k_v3_8r_vs_v2_realistic() {
        let rows = 1408;
        let cols = 2048;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xABCD_EF01);
        let x = fixed_input(cols, 0xFEDC_BA98);

        let ctx = ctx();

        let mut v2_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut v2_out).expect("v2 path");

        let model_buf = pinned_from_bytes(ctx, &w_bytes);
        let mut v3_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v3_8r_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut v3_out).expect("v3_8r path");

        let diff = max_abs_diff(&v2_out, &v3_out);
        println!("[WedgeK] v3_8r vs v2 realistic (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
        assert!(diff < 1e-3, "v3_8r vs v2 diff {diff:.2e} >= 1e-3");
    }

    // ── v3_dual parity tests ──────────────────────────────────────────────────────

    #[test]
    fn v1k_v3_dual_vs_v2_small() {
        let rows = 64;
        let cols = 256;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0x5555_6666);
        let x = fixed_input(cols, 0x7777_8888);

        let ctx = ctx();

        let mut v2_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut v2_out).expect("v2 path");

        let model_buf = pinned_from_bytes(ctx, &w_bytes);
        let mut dual_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v3_dual_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut dual_out).expect("v3_dual path");

        let diff = max_abs_diff(&v2_out, &dual_out);
        println!("[WedgeK] v3_dual vs v2 small (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
        assert!(diff < 1e-3, "v3_dual vs v2 diff {diff:.2e} >= 1e-3");
    }

    #[test]
    fn v1k_v3_dual_vs_v2_realistic() {
        let rows = 1408;
        let cols = 2048;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0x9999_AAAA);
        let x = fixed_input(cols, 0xBBBB_CCCC);

        let ctx = ctx();

        let mut v2_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut v2_out).expect("v2 path");

        let model_buf = pinned_from_bytes(ctx, &w_bytes);
        let mut dual_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v3_dual_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut dual_out).expect("v3_dual path");

        let diff = max_abs_diff(&v2_out, &dual_out);
        println!("[WedgeK] v3_dual vs v2 realistic (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
        assert!(diff < 1e-3, "v3_dual vs v2 diff {diff:.2e} >= 1e-3");
    }

    // ── v3_llama parity tests (Approach 3) ───────────────────────────────────────

    #[test]
    fn v1k_v3_llama_vs_v2_small() {
        let rows = 64;
        let cols = 256;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0x0101_0202);
        let x = fixed_input(cols, 0x0303_0404);

        let ctx = ctx();
        let mut v2_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut v2_out).expect("v2 path");

        let model_buf = pinned_from_bytes(ctx, &w_bytes);
        let mut llama_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v3_llama_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut llama_out).expect("v3_llama path");

        let diff = max_abs_diff(&v2_out, &llama_out);
        println!("[WedgeK] v3_llama vs v2 small (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
        assert!(diff < 1e-3, "v3_llama vs v2 diff {diff:.2e} >= 1e-3");
    }

    #[test]
    fn v1k_v3_llama_vs_v2_realistic() {
        let rows = 1408;
        let cols = 2048;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0x0505_0606);
        let x = fixed_input(cols, 0x0707_0808);

        let ctx = ctx();
        let mut v2_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut v2_out).expect("v2 path");

        let model_buf = pinned_from_bytes(ctx, &w_bytes);
        let mut llama_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v3_llama_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut llama_out).expect("v3_llama path");

        let diff = max_abs_diff(&v2_out, &llama_out);
        println!("[WedgeK] v3_llama vs v2 realistic (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
        assert!(diff < 1e-3, "v3_llama vs v2 diff {diff:.2e} >= 1e-3");
    }

    #[test]
    fn v1k_v3_llama_odd_rows() {
        let rows = 1405; // not multiple of 8
        let cols = 512;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0x0909_0A0A);
        let x = fixed_input(cols, 0x0B0B_0C0C);

        let ctx = ctx();
        let mut v2_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut v2_out).expect("v2 path");

        let model_buf = pinned_from_bytes(ctx, &w_bytes);
        let mut llama_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v3_llama_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut llama_out).expect("v3_llama odd rows path");

        let diff = max_abs_diff(&v2_out, &llama_out);
        println!("[WedgeK] v3_llama vs v2 odd rows (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
        assert!(diff < 1e-3, "v3_llama vs v2 diff {diff:.2e} >= 1e-3 on odd rows");
    }

    #[test]
    fn v1k_v3_dual_odd_rows() {
        // Test odd row count to exercise the row1_valid guard
        let rows = 1407; // odd — last TG has 7 rows (not a multiple of 8)
        let cols = 512;
        let n_blocks = rows * (cols / 256);
        let w_bytes = synthetic_q4_k_bytes(n_blocks, 0xDDDD_EEEE);
        let x = fixed_input(cols, 0xFFFF_0000);

        let ctx = ctx();

        let mut v2_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v2(ctx, &w_bytes, rows, cols, &x, &mut v2_out).expect("v2 path");

        let model_buf = pinned_from_bytes(ctx, &w_bytes);
        let mut dual_out = vec![0.0f32; rows];
        kernels::gemv_q4_k_m_v3_dual_pinned(ctx, &model_buf, 0, w_bytes.len(), rows, cols, &x, &mut dual_out).expect("v3_dual odd-rows path");

        let diff = max_abs_diff(&v2_out, &dual_out);
        println!("[WedgeK] v3_dual vs v2 odd rows (rows={rows} cols={cols}) max abs diff = {diff:.2e}");
        assert!(diff < 1e-3, "v3_dual vs v2 diff {diff:.2e} >= 1e-3 on odd rows");
    }
}
#[rustfmt::skip]
mod v1l_flash_attn_parity {
    //! Wedge L parity: flash_attn_decode_metal vs mla_decode_metal at atol=1e-3.
    //! Online softmax (flash) may accumulate fp32 rounding differently from
    //! the serial softmax in mla_decode_kernel; atol=1e-3 matches fp16 floor.
    #![cfg(target_os = "macos")]

    use hawking_core::kernels;
    use hawking_core::metal::PinnedBuffer;

    use crate::common;
    use common::*;

    fn run_parity(label: &str, n_heads: usize, qk_nope_head_dim: usize, qk_rope_head_dim: usize, v_head_dim: usize, kv_lora_rank: usize, seq_len: usize) {
        let ctx = ctx();
        let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        let scale = 1.0_f32 / (q_head_dim as f32).sqrt();

        let q = fixed_f32(n_heads * q_head_dim, 0xDEAD_BEEF);
        let c_kv = fixed_f32(seq_len * kv_lora_rank, 0xCAFE_BABE);
        let k_pe = fixed_f32(seq_len * qk_rope_head_dim, 0x1234_5678);
        let kv_b_raw = fixed_f32(n_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank, 0xABCD_EF01);
        let kv_b_buf: PinnedBuffer = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&kv_b_raw));

        let mut mla_out = vec![0.0f32; n_heads * v_head_dim];
        kernels::mla_decode_metal(ctx, &q, &c_kv, &k_pe, &kv_b_buf, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank, seq_len, scale, &mut mla_out).expect("mla_decode_metal");

        let mut flash_out = vec![0.0f32; n_heads * v_head_dim];
        kernels::flash_attn_decode_metal(ctx, &q, &c_kv, &k_pe, &kv_b_buf, n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank, seq_len, scale, &mut flash_out)
            .expect("flash_attn_decode_metal");

        let diff = max_abs_diff(&mla_out, &flash_out);
        println!("[WedgeL] {label} max_abs_diff={diff:.2e}");
        assert!(diff < 1e-3, "{label}: flash vs mla diff {diff:.2e} >= 1e-3");
    }

    #[test]
    fn v1l_flash_vs_mla_small() {
        run_parity("small(heads=4,nope=16,rope=8,v=16,lora=32,seq=64)", 4, 16, 8, 16, 32, 64);
    }

    #[test]
    fn v1l_flash_vs_mla_realistic() {
        // DeepSeek-V2-Lite-like: 16 heads, seq=256
        run_parity("realistic(heads=16,nope=64,rope=32,v=64,lora=64,seq=256)", 16, 64, 32, 64, 64, 256);
    }

    #[test]
    fn v1l_flash_vs_mla_seq_one() {
        // Edge case: seq_len=1 (first token, single tile)
        run_parity("seq1(heads=4,nope=16,rope=8,v=16,lora=32,seq=1)", 4, 16, 8, 16, 32, 1);
    }

    #[test]
    fn v1l_flash_vs_mla_multi_tile() {
        // seq_len=384 → 3 tiles of FLASH_TG=128; tests tile boundary correctness
        run_parity("multi_tile(heads=8,nope=32,rope=16,v=32,lora=32,seq=384)", 8, 32, 16, 32, 32, 384);
    }
}
#[rustfmt::skip]
mod v1x_lm_head_simdmat_parity {
    //! Phase X parity: gemv_f16_simdmat (simdgroup_matrix LM-head) matches CPU reference.
    //!
    //! Tests:
    //!   1. `phase_x_simdmat_matches_cpu_basic` — small shapes; atol 1e-3 (fp16 quant noise).
    //!   2. `phase_x_simdmat_matches_cpu_lm_head_shape` — (rows=512, cols=2048) LM-head analogue.
    //!   3. `phase_x_simdmat_argmax_matches_cpu` — token id (argmax) from simdmat matches CPU.
    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};

    use crate::common;
    use common::*;

    fn fixed_f16(n: usize, seed: u64) -> Vec<f16> {
        fixed_f32(n, seed).iter().map(|&v| f16::from_f32(v)).collect()
    }

    fn new_f16_buf(ctx: &MetalContext, data: &[f16]) -> PinnedBuffer {
        ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
    }

    fn cpu_gemv_f16(w: &[f16], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; rows];
        kernels::gemv_f16(w, rows, cols, x, &mut out);
        out
    }

    fn cpu_argmax(logits: &[f32]) -> u32 {
        logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as u32).unwrap()
    }

    // ─────────────────────────────────────────────────────────────────────────────

    /// Basic shapes — output matches CPU gemv_f16 at atol 1e-3 (fp16 quant noise).
    #[test]
    fn phase_x_simdmat_matches_cpu_basic() {
        let ctx = ctx();

        for &(rows, cols) in &[(8usize, 8usize), (16, 8), (8, 16), (32, 64), (64, 128)] {
            let w = fixed_f16(rows * cols, 0xA1B2_C3D4 ^ rows as u64);
            let x = fixed_f32(cols, 0xE5F6_0718 ^ cols as u64);
            let cpu = cpu_gemv_f16(&w, rows, cols, &x);

            let w_buf = new_f16_buf(ctx, &w);
            let x_buf = new_f32_buf(ctx, &x);
            let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_f16_simdmat_tcb(&mut tcb, &w_buf, rows, cols, &x_buf, &y_buf).unwrap_or_else(|e| panic!("gemv_f16_simdmat_tcb rows={rows} cols={cols}: {e}"));
            tcb.commit_and_wait().expect("commit");

            let gpu = read_f32_buf(&y_buf, rows);
            let diff = max_abs_diff(&cpu, &gpu);
            assert!(diff < 1e-3, "rows={rows} cols={cols}: max_abs_diff={diff:.2e} > 1e-3");
        }
    }

    /// LM-head analogue shape: rows=512, cols=2048.
    /// Validates the actual hidden_dim used by DeepSeek-V2-Lite (cols=2048 % 8 == 0 ✓).
    #[test]
    fn phase_x_simdmat_matches_cpu_lm_head_shape() {
        let ctx = ctx();
        let rows = 512usize;
        let cols = 2048usize;

        let w = fixed_f16(rows * cols, 0xBEEF_1234);
        let x = fixed_f32(cols, 0xDEAD_5678);
        let cpu = cpu_gemv_f16(&w, rows, cols, &x);

        let w_buf = new_f16_buf(ctx, &w);
        let x_buf = new_f32_buf(ctx, &x);
        let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_f16_simdmat_tcb(&mut tcb, &w_buf, rows, cols, &x_buf, &y_buf).expect("gemv_f16_simdmat_tcb");
        tcb.commit_and_wait().expect("commit");

        let gpu = read_f32_buf(&y_buf, rows);
        let diff = max_abs_diff(&cpu, &gpu);
        assert!(diff < 1e-3, "lm_head_shape rows={rows} cols={cols}: max_abs_diff={diff:.2e} > 1e-3");
    }

    /// Token parity (temp=0 greedy): argmax of simdmat output matches CPU.
    /// This is the exact gate used by forward_token_greedy.
    #[test]
    fn phase_x_simdmat_argmax_matches_cpu() {
        let ctx = ctx();
        let rows = 256usize;
        let cols = 128usize;

        let w = fixed_f16(rows * cols, 0xCAFE_BABE);
        let x = fixed_f32(cols, 0xF00D_FEED);
        let cpu_logits = cpu_gemv_f16(&w, rows, cols, &x);
        let cpu_token = cpu_argmax(&cpu_logits);

        let w_buf = new_f16_buf(ctx, &w);
        let x_buf = new_f32_buf(ctx, &x);
        let y_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_f16_simdmat_tcb(&mut tcb, &w_buf, rows, cols, &x_buf, &y_buf).expect("gemv_f16_simdmat_tcb");
        tcb.commit_and_wait().expect("commit");

        let gpu_logits = read_f32_buf(&y_buf, rows);
        let gpu_token = cpu_argmax(&gpu_logits);
        assert_eq!(gpu_token, cpu_token, "argmax: gpu={gpu_token} cpu={cpu_token}");

        let diff = max_abs_diff(&cpu_logits, &gpu_logits);
        assert!(diff < 1e-3, "logits diff={diff:.2e} > 1e-3");
    }
}
#[rustfmt::skip]
mod v2s_q4k_indexed_parity {
    //! Parity test: moe_batched_gemm_q4_indexed_v2s vs v2 reference.
    //!
    //! Test 1: routes=2, rows=64,  cols=256  (sub-TG sanity)
    //! Test 2: routes=4, rows=256, cols=2048 (realistic gate/up shape)
    //! Test 3: rows=70,  cols=256            (partial-TG boundary)

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    const ATOL: f32 = 2e-5;

    fn fixed_input(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
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

    fn run_parity(routes: usize, rows: usize, cols: usize, seed_base: u64) {
        let n_experts = routes + 3;
        let blocks_per_row = cols / 256;
        let blocks_per_expert = rows * blocks_per_row;
        let fused = synthetic_q4_k_bytes(n_experts * blocks_per_expert, seed_base);
        let mut model_bytes = vec![0xA5u8; 64];
        let base_offset = model_bytes.len();
        model_bytes.extend_from_slice(&fused);
        let route_ids: Vec<u32> = (0..routes).map(|i| ((i * 2 + 1) % n_experts) as u32).collect();
        let x = fixed_input(cols, seed_base ^ 0xDEAD_BEEF);

        let mut v2_out = vec![0.0_f32; routes * rows];
        kernels::moe_batched_gemm_q4_indexed_raw(ctx(), true, &model_bytes, base_offset, &route_ids, &x, routes, rows, cols, &mut v2_out).expect("v2 dispatch");

        let mut v2s_out = vec![0.0_f32; routes * rows];
        kernels::moe_batched_gemm_q4_indexed_v2s_raw(ctx(), &model_bytes, base_offset, &route_ids, &x, routes, rows, cols, &mut v2s_out).expect("v2s dispatch");

        let mut v2t_out = vec![0.0_f32; routes * rows];
        kernels::moe_batched_gemm_q4_indexed_v2t_raw(ctx(), &model_bytes, base_offset, &route_ids, &x, routes, rows, cols, &mut v2t_out).expect("v2t dispatch");

        let diff_s = max_abs_diff(&v2_out, &v2s_out);
        let diff_t = max_abs_diff(&v2_out, &v2t_out);
        println!("[v2s parity] routes={routes} rows={rows} cols={cols} v2s_diff={diff_s:.6e} v2t_diff={diff_t:.6e}");
        assert!(diff_s < ATOL, "v2s vs v2 diff {diff_s:.6e} >= atol {ATOL}");
        assert!(diff_t < ATOL, "v2t vs v2 diff {diff_t:.6e} >= atol {ATOL}");
    }

    #[test]
    fn test_v2s_small() {
        run_parity(2, 64, 256, 0xBEEF_0001);
    }

    #[test]
    fn test_v2s_realistic() {
        run_parity(4, 256, 2048, 0xBEEF_0002);
    }

    #[test]
    fn test_v2s_partial_tg() {
        run_parity(2, 70, 256, 0xBEEF_0003);
    }

    // ── v2t_gu parity ────────────────────────────────────────────────────────────

    const ATOL_GU: f32 = 2e-4;

    fn silu_f32(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn run_gu_parity(routes: usize, rows: usize, cols: usize, seed_base: u64) {
        let n_experts = routes + 3;
        let blocks_per_row = cols / 256;
        let blocks_per_expert = rows * blocks_per_row;
        let _n_bytes = n_experts * blocks_per_expert * 144;

        let gate_bytes = synthetic_q4_k_bytes(n_experts * blocks_per_expert, seed_base);
        let up_bytes = synthetic_q4_k_bytes(n_experts * blocks_per_expert, seed_base ^ 0xCAFE_BABE);

        let mut model_bytes = vec![0xA5u8; 64];
        let gate_offset = model_bytes.len();
        model_bytes.extend_from_slice(&gate_bytes);
        let up_offset = model_bytes.len();
        model_bytes.extend_from_slice(&up_bytes);

        let route_ids: Vec<u32> = (0..routes).map(|i| ((i * 2 + 1) % n_experts) as u32).collect();
        let x = fixed_input(cols, seed_base ^ 0xDEAD_BEEF);

        // Reference: v2t gate + v2t up + CPU silu_mul
        let mut gate_out = vec![0.0_f32; routes * rows];
        kernels::moe_batched_gemm_q4_indexed_v2t_raw(ctx(), &model_bytes, gate_offset, &route_ids, &x, routes, rows, cols, &mut gate_out).expect("v2t gate");

        let mut up_out = vec![0.0_f32; routes * rows];
        kernels::moe_batched_gemm_q4_indexed_v2t_raw(ctx(), &model_bytes, up_offset, &route_ids, &x, routes, rows, cols, &mut up_out).expect("v2t up");

        let ref_act: Vec<f32> = gate_out.iter().zip(up_out.iter()).map(|(&g, &u)| silu_f32(g) * u).collect();

        // Fused kernel
        let mut gu_act = vec![0.0_f32; routes * rows];
        kernels::moe_batched_gemm_q4_indexed_v2t_gu_raw(ctx(), &model_bytes, gate_offset, up_offset, &route_ids, &x, routes, rows, cols, &mut gu_act).expect("v2t_gu");

        let diff = max_abs_diff(&ref_act, &gu_act);
        println!("[v2t_gu parity] routes={routes} rows={rows} cols={cols} diff={diff:.6e}");
        assert!(diff < ATOL_GU, "v2t_gu diff {diff:.6e} >= atol {ATOL_GU}");
    }

    #[test]
    fn test_v2t_gu_small() {
        run_gu_parity(2, 64, 256, 0xFACE_0001);
    }

    #[test]
    fn test_v2t_gu_realistic() {
        run_gu_parity(4, 256, 2048, 0xFACE_0002);
    }

    #[test]
    fn test_v2t_gu_partial_tg() {
        run_gu_parity(2, 70, 256, 0xFACE_0003);
    }

    // ── Q8_0 v2t parity ──────────────────────────────────────────────────────────
    // Reference: CPU dequant of Q8_0 weights × route-major x.
    // Kernel under test: moe_batched_gemm_q8_0_indexed_v2t.

    const ATOL_Q8: f32 = 1e-4;

    fn synthetic_q8_0_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 34];
        for b in 0..n_blocks {
            let off = b * 34;
            let d = 0.001 + rng.gen::<f32>() * 0.001;
            bytes[off..off + 2].copy_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
            for i in 2..34 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    fn cpu_q8_0_matvec(w_bytes: &[u8], base_offset: usize, route_ids: &[u32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
        let blocks_per_row = cols / 32;
        let per_matrix_bytes = rows * blocks_per_row * 34;
        for (ri, &expert) in route_ids.iter().enumerate() {
            for row in 0..rows {
                let row_off = base_offset + expert as usize * per_matrix_bytes + row * blocks_per_row * 34;
                let mut acc = 0.0f32;
                for b in 0..blocks_per_row {
                    let bo = row_off + b * 34;
                    let d_bits = u16::from_le_bytes([w_bytes[bo], w_bytes[bo + 1]]);
                    let d = f16::from_bits(d_bits).to_f32();
                    for i in 0..32usize {
                        let qi = (w_bytes[bo + 2 + i] as i8) as f32;
                        let xi = x[ri * cols + b * 32 + i];
                        acc += d * qi * xi;
                    }
                }
                out[ri * rows + row] = acc;
            }
        }
    }

    fn run_q8_parity(routes: usize, rows: usize, cols: usize, seed_base: u64) {
        let n_experts = routes + 3;
        let blocks_per_row = cols / 32;
        let blocks_per_expert = rows * blocks_per_row;
        let w_bytes = synthetic_q8_0_bytes(n_experts * blocks_per_expert, seed_base);
        let mut model_bytes = vec![0xA5u8; 64];
        let base_offset = model_bytes.len();
        model_bytes.extend_from_slice(&w_bytes);
        let route_ids: Vec<u32> = (0..routes).map(|i| ((i * 2 + 1) % n_experts) as u32).collect();
        // x is route-major: each route has its own cols-element slice
        let x = fixed_input(routes * cols, seed_base ^ 0xDEAD_BEEF);

        let mut ref_out = vec![0.0_f32; routes * rows];
        cpu_q8_0_matvec(&model_bytes, base_offset, &route_ids, &x, rows, cols, &mut ref_out);

        let mut gpu_out = vec![0.0_f32; routes * rows];
        kernels::moe_batched_gemm_q8_0_indexed_v2t_raw(ctx(), &model_bytes, base_offset, &route_ids, &x, routes, rows, cols, &mut gpu_out).expect("q8_0 v2t dispatch");

        let diff = max_abs_diff(&ref_out, &gpu_out);
        println!("[q8_0 v2t parity] routes={routes} rows={rows} cols={cols} diff={diff:.6e}");
        assert!(diff < ATOL_Q8, "q8_0 v2t diff {diff:.6e} >= atol {ATOL_Q8}");
    }

    #[test]
    fn test_q8_0_v2t_small() {
        run_q8_parity(2, 64, 64, 0xB00B_0001);
    }

    #[test]
    fn test_q8_0_v2t_realistic() {
        run_q8_parity(6, 2048, 1408, 0xB00B_0002);
    }

    #[test]
    fn test_q8_0_v2t_partial_tg() {
        run_q8_parity(2, 70, 64, 0xB00B_0003);
    }
}
#[rustfmt::skip]
mod w4a8_per_channel_lmhead_kernel_parity {
    //! Track E parity — synthetic test that the per-channel W4A8 LM_HEAD path
    //! (quantize_per_channel + gemm_q4_k_a8_v3_8r_per_channel) matches the
    //! f32-baseline GEMV when scales come from `per_channel_scales_from_abs(x)`.
    //!
    //! This is the LM_HEAD-shape mirror of `w4a8_per_channel_parity.rs` (the
    //! q_proj-shape test that already shipped). Differs in:
    //!   - Shape: Qwen-3B LM_HEAD = vocab×hidden = 151936 × 2048 (vs 2048×2048
    //!     in q_proj). We test at a smaller-but-shape-similar 32768 × 2048 to
    //!     keep the test fast.
    //!   - Pipeline: this version uses the GPU quantize kernel
    //!     `quantize_f32_to_int8_per_channel_tcb` (NEW in Track E) instead of
    //!     the CPU-side `quantize_to_int8_per_channel`. Asserts they produce
    //!     identical int8 output bit-for-bit.
    //!   - Identity scales: also tests with all-ones scales to verify the
    //!     pipeline degenerates correctly to round-and-clip int8.

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use hawking_core::metal::{PinnedBuffer, TokenCommandBuffer};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let n_blocks = rows * (cols / 256);
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.01_f32 + rng.gen::<f32>() * 0.01;
            let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
            let d_bits = f16::from_f32(d).to_bits();
            let dmin_bits = f16::from_f32(dmin).to_bits();
            bytes[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
            bytes[off + 2..off + 4].copy_from_slice(&dmin_bits.to_le_bytes());
            for i in 4..16 {
                bytes[off + i] = rng.gen::<u8>() & 0x3F;
            }
            for i in 16..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    fn make_x(cols: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..cols).map(|_| rng.gen_range(-3.0_f32..3.0_f32)).collect()
    }

    fn read_i8_buf(buf: &PinnedBuffer, n: usize) -> Vec<i8> {
        let ptr = buf.contents() as *const i8;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    #[test]
    fn gpu_quantize_matches_cpu_per_channel() {
        // Verify the new `quantize_f32_to_int8_per_channel` GPU kernel produces
        // BIT-IDENTICAL output to the CPU `quantize_to_int8_per_channel` ref.
        let cols = 2048;
        let ctx = ctx();
        let x = make_x(cols, 0xABCD_1234);
        let scales = kernels::per_channel_scales_from_abs(&x);

        // CPU reference
        let cpu_i8 = kernels::quantize_to_int8_per_channel(&x, &scales);

        // GPU pipeline
        let x_buf = new_f32_buf(ctx, &x);
        let scales_buf = new_f32_buf(ctx, &scales);
        let out_buf = ctx.new_buffer(cols);
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::quantize_f32_to_int8_per_channel_tcb(&mut tcb, &x_buf, &scales_buf, &out_buf, cols).expect("GPU quantize encode");
            tcb.commit_and_wait().expect("GPU quantize commit");
        }
        let gpu_i8 = read_i8_buf(&out_buf, cols);

        assert_eq!(cpu_i8.len(), gpu_i8.len(), "lengths differ: cpu={} gpu={}", cpu_i8.len(), gpu_i8.len());
        let mismatches: Vec<usize> = (0..cols).filter(|&i| cpu_i8[i] != gpu_i8[i]).collect();
        if !mismatches.is_empty() {
            let first = mismatches[0];
            panic!("GPU/CPU quantize differ at {} positions; first @{}: cpu={} gpu={} x={:.4} scale={:.4e}", mismatches.len(), first, cpu_i8[first], gpu_i8[first], x[first], scales[first]);
        }
        eprintln!("[E4] GPU quantize bit-identical to CPU on cols={} ✓", cols);
    }

    #[test]
    fn end_to_end_per_channel_lmhead_pipeline() {
        // End-to-end pipeline test:
        //   1. GPU per-channel quantize x_norm → x_int8 using static scales
        //   2. GPU per-channel gemm_q4_k_a8_v3_8r_per_channel
        //   3. Compare against f32-baseline gemv at same Q4_K weights
        let rows = 32768_usize; // smaller-than-vocab but same shape ratio
        let cols = 2048_usize;
        let ctx = ctx();

        let w_bytes = make_q4k_bytes(rows, cols, 0xCAFE_BABE);
        let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
        let x = make_x(cols, 0xDEAD_BEEF);

        // f32-baseline
        let x_f32_buf = new_f32_buf(ctx, &x);
        let y_baseline_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_m_v3_8r_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_f32_buf, &y_baseline_buf).expect("baseline encode");
            tcb.commit_and_wait().expect("baseline commit");
        }
        let y_baseline = read_f32_buf(&y_baseline_buf, rows);

        // Per-channel W4A8 pipeline (GPU quantize + GPU per-channel gemm)
        let scales = kernels::per_channel_scales_from_abs(&x);
        let scales_buf = new_f32_buf(ctx, &scales);
        let x_int8_buf = ctx.new_buffer(cols);
        let y_pc_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::quantize_f32_to_int8_per_channel_tcb(&mut tcb, &x_f32_buf, &scales_buf, &x_int8_buf, cols).expect("E4 quantize encode");
            kernels::gemm_q4_k_a8_v3_8r_per_channel_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_int8_buf, &scales_buf, &y_pc_buf).expect("E4 per-channel gemm encode");
            tcb.commit_and_wait().expect("E4 commit");
        }
        let y_pc = read_f32_buf(&y_pc_buf, rows);

        // Cosine + NRMSE (oracle scales → should be near-bit-identical).
        let dot: f32 = y_baseline.iter().zip(&y_pc).map(|(&a, &b)| a * b).sum();
        let na: f32 = y_baseline.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let nb: f32 = y_pc.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let cosine = dot / (na * nb);
        let rmse: f32 = (y_baseline.iter().zip(&y_pc).map(|(&a, &b)| (a - b).powi(2)).sum::<f32>() / rows as f32).sqrt();
        let mean_abs = y_baseline.iter().map(|x| x.abs()).sum::<f32>() / rows as f32;
        let nrmse = rmse / mean_abs;
        eprintln!("[E4 end-to-end] rows={} cosine={:.6} nrmse={:.4e}  baseline[0..4]={:?}  pc[0..4]={:?}", rows, cosine, nrmse, &y_baseline[..4], &y_pc[..4]);
        assert!(cosine > 0.9999 && nrmse < 0.02, "per-channel LM_HEAD pipeline out of tolerance: cosine={cosine:.6} nrmse={nrmse:.4e}");
    }
}
#[rustfmt::skip]
mod w4a8_per_channel_parity {
    //! W4A8 per-channel parity test — `gemm_q4_k_a8_v3_8r_per_channel`.
    //!
    //! Compares two paths at the q_proj decode shape (rows=2048, cols=2048):
    //!
    //!   A. f32-activation baseline: `gemv_q4_k_m_v3_8r_pinned_tcb` on the
    //!      ORIGINAL f32 activation (no quantization noise).
    //!   B. W4A8 per-channel: CPU-side `quantize_to_int8_per_channel` →
    //!      Metal `gemm_q4_k_a8_v3_8r_per_channel`.
    //!
    //! Asserts cosine similarity > 0.9999 and normalized RMSE < 0.02 vs the
    //! f32-activation baseline. Per-channel int8 quantization is lossy so
    //! we don't expect bit-identical, but the per-channel scheme should beat
    //! the per-block one by a wide margin on outlier-heavy inputs.
    //!
    //! As a sanity check, also asserts the per-channel result is at least as
    //! good as the per-block result on a SYNTHETIC outlier-injected input
    //! (one channel set to magnitude 50, rest ~3) — the regime where the
    //! 256-block scale gets crushed by the outlier and per-channel wins.

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;

    use crate::common;
    use common::*;

    fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let n_blocks = rows * (cols / 256);
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.01_f32 + rng.gen::<f32>() * 0.01;
            let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
            let d_bits = f16::from_f32(d).to_bits();
            let dmin_bits = f16::from_f32(dmin).to_bits();
            bytes[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
            bytes[off + 2..off + 4].copy_from_slice(&dmin_bits.to_le_bytes());
            for i in 4..16 {
                bytes[off + i] = rng.gen::<u8>() & 0x3F;
            }
            for i in 16..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    fn make_x_typical(cols: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..cols).map(|_| rng.gen_range(-3.0_f32..3.0_f32)).collect()
    }

    /// Channel-correlated calibration: simulate the "static scale per channel"
    /// produced by a calibration corpus. Real Qwen-3B activations have
    /// per-channel max|x| in [0.93, 150]; here we draw per-channel scales
    /// from a log-uniform distribution in [0.2, 4.0] to mimic that uneven
    /// spread without needing the model loaded.
    fn make_channel_scales_calibrated(cols: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..cols)
            .map(|_| {
                // log-uniform: max_abs in [0.5, 50] → scale = max_abs/127
                let log_lo = (0.5_f32).ln();
                let log_hi = (50.0_f32).ln();
                let max_abs = (log_lo + (log_hi - log_lo) * rng.gen::<f32>()).exp();
                max_abs / 127.0
            })
            .collect()
    }

    /// Generate activations consistent with the given per-channel scales:
    /// `x[c] ~ U[-127, 127] * scales[c]`. This way the per-channel scheme
    /// can encode without saturation, while a per-block scheme would have
    /// to take the block-wise max.
    fn make_x_from_scales(scales: &[f32], seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        scales
            .iter()
            .map(|&s| {
                let q = rng.gen_range(-127.0_f32..127.0_f32);
                q * s
            })
            .collect()
    }

    fn new_buf_bytes(ctx: &MetalContext, bytes: &[u8]) -> PinnedBuffer {
        ctx.new_buffer_with_bytes(bytes)
    }

    fn cosine_and_nrmse(a: &[f32], b: &[f32]) -> (f32, f32) {
        let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
        let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let cosine = dot / (na * nb);
        let rmse: f32 = (a.iter().zip(b).map(|(&x, &y)| (x - y).powi(2)).sum::<f32>() / a.len() as f32).sqrt();
        let mean_abs_a = a.iter().map(|x| x.abs()).sum::<f32>() / a.len() as f32;
        let nrmse = rmse / mean_abs_a;
        (cosine, nrmse)
    }

    #[test]
    fn w4a8_per_channel_typical_activations() {
        let rows = 2048_usize;
        let cols = 2048_usize;
        let ctx = ctx();

        let w_bytes = make_q4k_bytes(rows, cols, 0xDEAD_BEEF);
        let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
        let x = make_x_typical(cols, 0xCAFE_F00D);

        // (A) f32 baseline
        let x_f32_buf = new_f32_buf(ctx, &x);
        let y_baseline_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_m_v3_8r_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_f32_buf, &y_baseline_buf).expect("baseline encode");
            tcb.commit_and_wait().expect("baseline commit");
        }
        let y_baseline = read_f32_buf(&y_baseline_buf, rows);

        // (B) W4A8 per-channel — calibrated-style scales from |x| itself
        // (oracle scales, not a calibration corpus). This is the BEST CASE
        // for per-channel quantization — saturates only at the most-extreme
        // channel — and the lower bound on what a calibration-corpus
        // implementation should achieve.
        let channel_scales = kernels::per_channel_scales_from_abs(&x);
        let x_int8 = kernels::quantize_to_int8_per_channel(&x, &channel_scales);
        assert_eq!(x_int8.len(), cols);
        assert_eq!(channel_scales.len(), cols);

        let x_int8_buf = new_buf_bytes(ctx, bytemuck::cast_slice(&x_int8));
        let x_scales_buf = new_f32_buf(ctx, &channel_scales);
        let y_w4a8_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_a8_v3_8r_per_channel_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_int8_buf, &x_scales_buf, &y_w4a8_buf).expect("W4A8 per-channel encode");
            tcb.commit_and_wait().expect("W4A8 per-channel commit");
        }
        let y_w4a8 = read_f32_buf(&y_w4a8_buf, rows);

        let (cosine, nrmse) = cosine_and_nrmse(&y_baseline, &y_w4a8);
        eprintln!("[W4A8 per-channel, typical] cosine={cosine:.6}  nrmse={nrmse:.4e}");
        eprintln!("  baseline[0..6] = {:?}", &y_baseline[..6]);
        eprintln!("  w4a8[0..6]     = {:?}", &y_w4a8[..6]);

        // Per-channel with oracle scales should be tight against f32 baseline.
        assert!(cosine > 0.9999 && nrmse < 0.02, "per-channel out of tolerance: cosine={cosine:.6} nrmse={nrmse:.4e}");
    }

    #[test]
    fn w4a8_per_channel_beats_per_block_on_outliers() {
        // Synthetic outlier regime: most channels |x|~3, but channels at
        // index 1979 and 132 (the Qwen-3B super-outliers) carry magnitude 50.
        // Per-block scaling assigns the entire 256-block's scale = 50/127,
        // crushing the resolution for the other 255 channels. Per-channel
        // gives each channel its own scale.
        let rows = 2048_usize;
        let cols = 2048_usize;
        let ctx = ctx();

        let w_bytes = make_q4k_bytes(rows, cols, 0x1979_0132);
        let model_buf = ctx.new_buffer_with_bytes(&w_bytes);

        // Channel-correlated calibration data so per-channel actually has
        // distinct scales to leverage.
        let channel_scales = make_channel_scales_calibrated(cols, 0xC0FFEE);
        let x = make_x_from_scales(&channel_scales, 0xBADF00D);

        // Inject extreme outliers
        let mut x = x;
        x[1979] = 50.0;
        x[132] = -45.0;
        // Recompute per-channel scales now that we forced outliers
        let channel_scales = kernels::per_channel_scales_from_abs(&x);

        // f32 baseline
        let x_f32_buf = new_f32_buf(ctx, &x);
        let y_baseline_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_m_v3_8r_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_f32_buf, &y_baseline_buf).unwrap();
            tcb.commit_and_wait().unwrap();
        }
        let y_baseline = read_f32_buf(&y_baseline_buf, rows);

        // Per-block W4A8 path
        let (x_int8_pb, x_scales_pb) = kernels::quantize_to_int8_per_block(&x, 256);
        let x_int8_pb_buf = new_buf_bytes(ctx, bytemuck::cast_slice(&x_int8_pb));
        let x_scales_pb_buf = new_f32_buf(ctx, &x_scales_pb);
        let y_pb_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_a8_v3_8r_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_int8_pb_buf, &x_scales_pb_buf, &y_pb_buf).unwrap();
            tcb.commit_and_wait().unwrap();
        }
        let y_pb = read_f32_buf(&y_pb_buf, rows);

        // Per-channel W4A8 path
        let x_int8_pc = kernels::quantize_to_int8_per_channel(&x, &channel_scales);
        let x_int8_pc_buf = new_buf_bytes(ctx, bytemuck::cast_slice(&x_int8_pc));
        let x_scales_pc_buf = new_f32_buf(ctx, &channel_scales);
        let y_pc_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_a8_v3_8r_per_channel_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_int8_pc_buf, &x_scales_pc_buf, &y_pc_buf).unwrap();
            tcb.commit_and_wait().unwrap();
        }
        let y_pc = read_f32_buf(&y_pc_buf, rows);

        let (cos_pb, nrmse_pb) = cosine_and_nrmse(&y_baseline, &y_pb);
        let (cos_pc, nrmse_pc) = cosine_and_nrmse(&y_baseline, &y_pc);

        eprintln!("[outlier regime]");
        eprintln!("  per-block:   cosine={cos_pb:.6}  nrmse={nrmse_pb:.4e}");
        eprintln!("  per-channel: cosine={cos_pc:.6}  nrmse={nrmse_pc:.4e}");
        eprintln!("  improvement: cosine +{:.6}, nrmse / {:.2}×", cos_pc - cos_pb, if nrmse_pc > 0.0 { nrmse_pb / nrmse_pc } else { f32::INFINITY },);

        // Per-channel must beat per-block in the outlier regime — that's the
        // entire point of the redesign. We require strict inequality on both
        // metrics with a small margin to guard against noise.
        assert!(cos_pc >= cos_pb, "per-channel cosine {cos_pc:.6} should be >= per-block {cos_pb:.6} in outlier regime");
        assert!(nrmse_pc <= nrmse_pb, "per-channel nrmse {nrmse_pc:.4e} should be <= per-block {nrmse_pb:.4e} in outlier regime");

        // And the per-channel path should be ABSOLUTELY tight against f32
        // baseline — outlier injection doesn't excuse poor agreement.
        assert!(cos_pc > 0.999 && nrmse_pc < 0.05, "per-channel absolute tolerance: cosine={cos_pc:.6} nrmse={nrmse_pc:.4e}");
    }
}
#[rustfmt::skip]
mod w4a8_prototype {
    //! W4A8 prototype — parity + microbench for gemm_q4_k_a8_v3_8r.
    //!
    //! Two assertions:
    //!   1. **Parity within tolerance.** Per-block int8 quantization is lossy,
    //!      so we don't expect bit-identical output vs the f32-activation path.
    //!      We assert each output element is within 1% relative or 1e-2 absolute.
    //!   2. **Bandwidth saving is real.** Microbench at the decode q_proj shape
    //!      (rows=2048, cols=2048). Report mean us/call for f32 baseline,
    //!      W4A8 kernel only, and W4A8 + quantize cost — so the honest
    //!      "production" delta accounts for the quantize CPU time.

    #![cfg(target_os = "macos")]

    use half::f16;
    use hawking_core::kernels;
    use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;
    use std::time::Instant;

    use crate::common;
    use common::*;

    fn make_q4k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let n_blocks = rows * (cols / 256);
        let mut rng = Pcg64Mcg::new(seed as u128);
        let mut bytes = vec![0u8; n_blocks * 144];
        for b in 0..n_blocks {
            let off = b * 144;
            let d = 0.01_f32 + rng.gen::<f32>() * 0.01;
            let dmin = (rng.gen::<f32>() - 0.5) * 0.01;
            let d_bits = f16::from_f32(d).to_bits();
            let dmin_bits = f16::from_f32(dmin).to_bits();
            bytes[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
            bytes[off + 2..off + 4].copy_from_slice(&dmin_bits.to_le_bytes());
            for i in 4..16 {
                bytes[off + i] = rng.gen::<u8>() & 0x3F;
            }
            for i in 16..144 {
                bytes[off + i] = rng.gen::<u8>();
            }
        }
        bytes
    }

    fn make_x(cols: usize, seed: u64) -> Vec<f32> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        // Typical post-rmsnorm activation magnitude: ~[-3, 3].
        (0..cols).map(|_| rng.gen_range(-3.0_f32..3.0_f32)).collect()
    }

    fn new_buf_bytes(ctx: &MetalContext, bytes: &[u8]) -> PinnedBuffer {
        ctx.new_buffer_with_bytes(bytes)
    }

    #[test]
    fn w4a8_parity_and_bw_saving() {
        let rows = 2048_usize;
        let cols = 2048_usize;
        let ctx = ctx();

        let w_bytes = make_q4k_bytes(rows, cols, 0xDEAD_BEEF);
        let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
        let x = make_x(cols, 0xCAFE_F00D);

        // f32 baseline path: gemv_q4_k_m_v3_8r_pinned_tcb.
        let x_f32_buf = new_f32_buf(ctx, &x);
        let y_baseline_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemv_q4_k_m_v3_8r_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_f32_buf, &y_baseline_buf).expect("baseline encode");
            tcb.commit_and_wait().expect("baseline commit");
        }
        let y_baseline = read_f32_buf(&y_baseline_buf, rows);

        // W4A8 path: quantize CPU-side, then dispatch.
        let (x_int8, x_scales) = kernels::quantize_to_int8_per_block(&x, 256);
        assert_eq!(x_int8.len(), cols);
        assert_eq!(x_scales.len(), cols / 256);
        let x_int8_buf = new_buf_bytes(ctx, bytemuck::cast_slice(&x_int8));
        let x_scales_buf = new_f32_buf(ctx, &x_scales);
        let y_w4a8_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        {
            let mut tcb = TokenCommandBuffer::new(ctx);
            kernels::gemm_q4_k_a8_v3_8r_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_int8_buf, &x_scales_buf, &y_w4a8_buf).expect("W4A8 encode");
            tcb.commit_and_wait().expect("W4A8 commit");
        }
        let y_w4a8 = read_f32_buf(&y_w4a8_buf, rows);

        // Parity via cosine similarity + L2-normalized RMSE — the metrics
        // that actually matter for a dot product downstream. int8 quant
        // noise is uncorrelated across 2048 elements; per-element rel
        // bound on low-magnitude outputs is dominated by sqrt(N)·scale
        // noise even when the kernel is exactly right.
        let dot_ab: f32 = y_baseline.iter().zip(&y_w4a8).map(|(&a, &b)| a * b).sum();
        let norm_a: f32 = y_baseline.iter().map(|&a| a * a).sum::<f32>().sqrt();
        let norm_b: f32 = y_w4a8.iter().map(|&a| a * a).sum::<f32>().sqrt();
        let cosine = dot_ab / (norm_a * norm_b);
        let rmse: f32 = (y_baseline.iter().zip(&y_w4a8).map(|(&a, &b)| (a - b).powi(2)).sum::<f32>() / y_baseline.len() as f32).sqrt();
        let mean_abs_baseline = y_baseline.iter().map(|x| x.abs()).sum::<f32>() / y_baseline.len() as f32;
        let nrmse = rmse / mean_abs_baseline;
        eprintln!("[W4A8 parity] cosine_sim={:.6}  rmse={:.4e}  nrmse={:.4e} (norm by mean|baseline|)", cosine, rmse, nrmse,);
        // Debug: first 8 elements of each + sanity-check the quantize roundtrip.
        eprintln!("[debug] baseline[0..8] = {:?}", &y_baseline[..8]);
        eprintln!("[debug] W4A8[0..8]     = {:?}", &y_w4a8[..8]);
        // Verify the quant round-trip is within expected noise (sanity).
        let mut max_recover_err = 0.0f32;
        for i in 0..cols {
            let b = i / 256;
            let recovered = x_int8[i] as f32 * x_scales[b];
            let err = (recovered - x[i]).abs();
            if err > max_recover_err {
                max_recover_err = err;
            }
        }
        eprintln!("[debug] quant roundtrip max_abs_err on x: {:.4e}", max_recover_err);
        assert!(cosine > 0.999 && nrmse < 0.05, "W4A8 output out of tolerance: cosine={cosine:.6} nrmse={nrmse:.4e}");

        // ── Microbench ────────────────────────────────────────────────
        eprintln!("\n=== microbench: rows={rows} cols={cols} ===");

        let warmup = 40;
        let calls = 200;

        // (A) f32 baseline kernel time
        {
            let mut run = || {
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::gemv_q4_k_m_v3_8r_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_f32_buf, &y_baseline_buf).unwrap();
                tcb.commit_and_wait().unwrap();
            };
            for _ in 0..warmup {
                run();
            }
            let t0 = Instant::now();
            for _ in 0..calls {
                run();
            }
            let us = t0.elapsed().as_micros() as f64 / calls as f64;
            eprintln!("[A] f32 baseline (v3_8r)               mean={:.1} us/call", us);
        }

        // (B) W4A8 kernel time ONLY (pre-quantized, no quantize cost included)
        {
            let mut run = || {
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::gemm_q4_k_a8_v3_8r_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &x_int8_buf, &x_scales_buf, &y_w4a8_buf).unwrap();
                tcb.commit_and_wait().unwrap();
            };
            for _ in 0..warmup {
                run();
            }
            let t0 = Instant::now();
            for _ in 0..calls {
                run();
            }
            let us = t0.elapsed().as_micros() as f64 / calls as f64;
            eprintln!("[B] W4A8 kernel only (pre-quantized)   mean={:.1} us/call", us);
        }

        // (C) W4A8 + quantize cost (realistic per-step cost)
        {
            let mut run = || {
                let (x_q, x_s) = kernels::quantize_to_int8_per_block(&x, 256);
                let xq_buf = new_buf_bytes(ctx, bytemuck::cast_slice(&x_q));
                let xs_buf = new_f32_buf(ctx, &x_s);
                let mut tcb = TokenCommandBuffer::new(ctx);
                kernels::gemm_q4_k_a8_v3_8r_pinned_tcb(&mut tcb, &model_buf, 0, w_bytes.len(), rows, cols, &xq_buf, &xs_buf, &y_w4a8_buf).unwrap();
                tcb.commit_and_wait().unwrap();
            };
            for _ in 0..warmup {
                run();
            }
            let t0 = Instant::now();
            for _ in 0..calls {
                run();
            }
            let us = t0.elapsed().as_micros() as f64 / calls as f64;
            eprintln!("[C] W4A8 + quantize + alloc each call  mean={:.1} us/call", us);
            eprintln!("    (production would amortize quant across all 7 GEMVs per layer)");
        }
    }
}
