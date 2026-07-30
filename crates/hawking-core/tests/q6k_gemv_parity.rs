#![cfg(target_os = "macos")]
use hawking_core::kernels;
use hawking_core::metal::TokenCommandBuffer;
use hawking_core::quant;
mod common;
use common::*;
#[test]
fn q6k_gemv_matches_cpu_reference() {
    let rows = 256usize;
    let cols = 2048usize;
    let w_f32 = fixed_f32(rows * cols, 0xC0DEC0DE);
    let blocks = (rows * cols) / 256;
    let mut w_q6 = vec![0u8; blocks * quant::Q6_K_BLOCK_BYTES];
    quant::quantize_q6_k(&w_f32, &mut w_q6).expect("Q6_K quant");
    let mut w_recon = vec![0.0f32; rows * cols];
    quant::dequant_into(hawking_core::gguf::GgmlType::Q6_K, &w_q6, &mut w_recon)
        .expect("Q6_K dequant");
    let x = fixed_f32(cols, 0xBEEFBEEF);
    let mut expected = vec![0.0f32; rows];
    for r in 0..rows {
        let mut acc = 0.0f32;
        let row = &w_recon[r * cols..(r + 1) * cols];
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        expected[r] = acc;
    }
    let ctx = ctx();
    let model_buf = ctx.new_buffer_with_bytes(&w_q6);
    let x_buf = new_f32_buf(ctx, &x);
    let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q6_k_pinned_tcb(
            &mut tcb,
            &model_buf,
            0,
            w_q6.len(),
            rows,
            cols,
            &x_buf,
            &out_buf,
        )
        .expect("gemv_q6_k encode");
        tcb.commit_and_wait().expect("commit");
    }
    let actual = read_f32_buf(&out_buf, rows);
    let diff = max_abs_diff(&expected, &actual);
    assert!(diff < 5e-2, "q6_k gemv max_abs_diff = {diff} (limit 5e-2)");
}
