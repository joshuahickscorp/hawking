#![cfg(target_os = "macos")]
use half::f16;
use hawking_core::kernels;
use hawking_core::metal::TokenCommandBuffer;
use rand::Rng;
use rand_pcg::Pcg64Mcg;
mod common;
use common::*;
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
    (0..cols)
        .map(|_| rng.gen_range(-3.0_f32..3.0_f32))
        .collect()
}
fn run_one(rows: usize, cols: usize, seed: u64) {
    let ctx = ctx();
    let w_bytes = make_q3k_bytes(rows, cols, seed);
    let model_buf = ctx.new_buffer_with_bytes(&w_bytes);
    let x = make_x(cols, 0xCAFE_F00D ^ seed);
    let x_buf = new_f32_buf(ctx, &x);
    let y_v2_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q3_k_pinned_tcb(
            &mut tcb,
            &model_buf,
            0,
            w_bytes.len(),
            rows,
            cols,
            &x_buf,
            &y_v2_buf,
        )
        .expect("q3_k fused_v2 encode");
        tcb.commit_and_wait().expect("q3_k fused_v2 commit");
    }
    let y_v2 = read_f32_buf(&y_v2_buf, rows);
    let y_2r_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    {
        let mut tcb = TokenCommandBuffer::new(ctx);
        kernels::gemv_q3_k_fused_2r_pinned_tcb(
            &mut tcb,
            &model_buf,
            0,
            w_bytes.len(),
            rows,
            cols,
            &x_buf,
            &y_2r_buf,
        )
        .expect("q3_k fused_2r encode");
        tcb.commit_and_wait().expect("q3_k fused_2r commit");
    }
    let y_2r = read_f32_buf(&y_2r_buf, rows);
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
    } else {
        const ATOL: f32 = 1e-3;
        assert!(
            max_abs < ATOL,
            "q3k_fused_2r exceeds fp16 tol vs fused_v2: max_abs={max_abs:e} (atol {ATOL}) \
             at i={worst}  v2={}  2r={}",
            y_v2[worst],
            y_2r[worst],
        );
    }
}
#[test]
fn q3k_fused_2r_matches_fused_v2() {
    run_one(2048, 2048, 0x3D15_8E1E);
    run_one(11008, 2048, 0x51C0_0001);
    run_one(2048, 11008, 0x7A11_BEEF);
}
#[test]
fn q3k_fused_2r_ragged_rows() {
    run_one(2056, 2048, 0x0DD0_1234);
}
