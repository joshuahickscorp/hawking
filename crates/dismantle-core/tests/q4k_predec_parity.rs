//! q4k_predec — bit-identical parity between gemv_q4_k_m_v3_8r_pinned_tcb
//! (inline sub-block scale decode) and gemv_q4_k_v4_predec_pinned_tcb
//! (sub-block scales pre-decoded host-side at load time into an f32 table).
//!
//! Both kernels share the v3_8r geometry and the same widening order
//! (fp16 d/dmin -> f32, uchar 6-bit sb/mb -> f32, multiply in f32), so the
//! outputs MUST be bit-identical. Anything other than exact equality is a
//! bug in the pre-decoder or the shader.

#![cfg(target_os = "macos")]

use dismantle_core::kernels;
use dismantle_core::metal::TokenCommandBuffer;
use half::f16;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

mod common;
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
        kernels::gemv_q4_k_m_v3_8r_pinned_tcb(
            &mut tcb, &model_buf, 0, w_bytes.len(),
            rows, cols, &x_buf, &y_v3_buf,
        ).expect("v3_8r encode");
        tcb.commit_and_wait().expect("v3_8r commit");
    }
    let y_v3 = read_f32_buf(&y_v3_buf, rows);

    // v4_predec: build host-side scale table, pin, dispatch.
    let scales = kernels::predecode_q4_k_scale_table(&w_bytes);
    let expected_scale_len = rows * (cols / 256) * 16;
    assert_eq!(scales.len(), expected_scale_len,
        "predecode_q4_k_scale_table length mismatch: got {} expected {}",
        scales.len(), expected_scale_len);
    let scales_buf = new_f32_buf(ctx, &scales);

    let y_v4_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q4_k_v4_predec_pinned_tcb(
            &mut tcb, &model_buf, 0, w_bytes.len(),
            &scales_buf, 0,
            rows, cols, &x_buf, &y_v4_buf,
        ).expect("v4_predec encode");
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
            a.to_bits(), b.to_bits(),
        );
    }
    eprintln!("[q4k_v4_predec parity] {} rows bit-identical to v3_8r", rows);
}
