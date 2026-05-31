//! q3k_predec — bit-identical parity between gemv_q3_k_pinned_tcb
//! (gemm_q3_k_fused_v2, inline sub-block scale decode) and
//! gemv_q3_k_v4_predec_pinned_tcb (sub-block scales pre-decoded host-side at
//! load time into an f32 table via predecode_q3_k_scale_table).
//!
//! Both kernels share the 8-row-per-TG geometry and the same widening order
//! (fp16 d -> f32, i8 6-bit scale -> f32, the (d*scale) product computed in
//! f32, then * (float)q * x), so the outputs MUST be bit-identical. Q3_K is
//! symmetric (no min term), so the pre-decoded table is 16 f32/block. Anything
//! other than exact equality is a bug in the pre-decoder or the shader.
//!
//! This is the byte-cut Stage-3 unblock validation: the fast Q3_K GEMV that a
//! Q3_K (−11% bytes) model needs to run on the predec fast path instead of the
//! generic dequant path. GPU-gated (needs a Metal device).

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use dismantle_core::quant::predecode_q3_k_scale_table;
use half::f16;
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

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

fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

fn read_f32_buf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

#[test]
fn q3k_v4_predec_bit_identical_to_fused_v2() {
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
        kernels::gemv_q3_k_pinned_tcb(
            &mut tcb, &model_buf, 0, w_bytes.len(),
            rows, cols, &x_buf, &y_fused_buf,
        ).expect("q3_k fused encode");
        tcb.commit_and_wait().expect("q3_k fused commit");
    }
    let y_fused = read_f32_buf(&y_fused_buf, rows);

    // v4_predec: build host-side scale table (16 f32/block), pin, dispatch.
    let scales = predecode_q3_k_scale_table(&w_bytes);
    let expected_scale_len = rows * (cols / 256) * 16;
    assert_eq!(scales.len(), expected_scale_len,
        "predecode_q3_k_scale_table length mismatch: got {} expected {}",
        scales.len(), expected_scale_len);
    let scales_buf = new_f32_buf(ctx, &scales);

    let y_predec_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q3_k_v4_predec_pinned_tcb(
            &mut tcb, &model_buf, 0, w_bytes.len(),
            &scales_buf, 0,
            rows, cols, &x_buf, &y_predec_buf,
        ).expect("q3_k v4_predec encode");
        tcb.commit_and_wait().expect("q3_k v4_predec commit");
    }
    let y_predec = read_f32_buf(&y_predec_buf, rows);

    // Bit-identical: every f32 bit-pattern must match. Both kernels do the
    // same fp32 operations in the same order; the only difference is whether
    // (d*scale) is decoded inline or read from the pre-decoded table.
    let mut first_diff: Option<(usize, f32, f32)> = None;
    let mut diff_count = 0usize;
    for i in 0..rows {
        if y_fused[i].to_bits() != y_predec[i].to_bits() {
            diff_count += 1;
            if first_diff.is_none() {
                first_diff = Some((i, y_fused[i], y_predec[i]));
            }
        }
    }
    if let Some((i, a, b)) = first_diff {
        panic!(
            "q3k_v4_predec NOT bit-identical to fused_v2: {diff_count}/{rows} rows differ; \
             first @ i={i}  fused={a:e} (0x{:08x})  predec={b:e} (0x{:08x})",
            a.to_bits(), b.to_bits(),
        );
    }
    eprintln!("[q3k_v4_predec parity] {} rows bit-identical to gemm_q3_k_fused_v2", rows);
}
