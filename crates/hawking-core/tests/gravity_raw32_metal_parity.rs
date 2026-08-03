#![cfg(target_os = "macos")]

use half::f16;
use hawking_core::gguf::GgmlType;
use hawking_core::metal::TokenCommandBuffer;
use hawking_core::quant;

mod common;
use common::*;

#[repr(C)]
#[derive(Clone, Copy)]
struct Raw32Params {
    rows: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawQ5Q5QvRopeAppendParams {
    q_rows: u32,
    kv_rows: u32,
    cols: u32,
    kv_off: u32,
    head_dim: u32,
    has_q_bias: u32,
    has_k_bias: u32,
    has_v_bias: u32,
    v_is_q8: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawQ5PairParams {
    rows: u32,
    cols: u32,
}

fn source_q8(rows: usize, cols: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows * (cols / 32) * 34);
    for block in 0..rows * (cols / 32) {
        bytes.extend_from_slice(
            &f16::from_f32(0.03125 + block as f32 / 1024.0)
                .to_bits()
                .to_le_bytes(),
        );
        for i in 0..32 {
            bytes.push(((i * 13 + block * 7) as i32 - 127) as i8 as u8);
        }
    }
    bytes
}

fn source_q5(rows: usize, cols: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows * (cols / 32) * 22);
    for block in 0..rows * (cols / 32) {
        bytes.extend_from_slice(
            &f16::from_f32(0.0625 + block as f32 / 2048.0)
                .to_bits()
                .to_le_bytes(),
        );
        let mut high = 0u32;
        let mut low = [0u8; 16];
        for i in 0..32 {
            let value = ((i * 5 + block * 3) & 31) as u8;
            high |= ((value >> 4) as u32) << i;
            let slot = &mut low[i & 15];
            if i < 16 {
                *slot |= value & 0x0f;
            } else {
                *slot |= (value & 0x0f) << 4;
            }
        }
        bytes.extend_from_slice(&high.to_le_bytes());
        bytes.extend_from_slice(&low);
    }
    bytes
}

fn cpu_matvec(dtype: GgmlType, bytes: &[u8], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let (_, block_bytes) = dtype.block_layout();
    let row_bytes = cols / dtype.block_layout().0 as usize * block_bytes as usize;
    (0..rows)
        .map(|row| {
            let mut decoded = vec![0.0; cols];
            quant::dequant_into(
                dtype,
                &bytes[row * row_bytes..(row + 1) * row_bytes],
                &mut decoded,
            )
            .expect("source decode");
            decoded.iter().zip(x).map(|(w, v)| w * v).sum()
        })
        .collect()
}

fn assert_source_packed_matvec(kernel: &str, dtype: GgmlType, bytes: Vec<u8>) {
    let rows = 11usize;
    let cols = 96usize;
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i as f32 * 0.173).sin()) - 0.25)
        .collect();
    let cpu = cpu_matvec(dtype, &bytes, rows, cols, &x);
    let ctx = ctx();
    let weights = ctx.new_buffer_with_bytes(&bytes);
    let x_buf = new_f32_buf(ctx, &x);
    let out = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let params = Raw32Params {
        rows: rows as u32,
        cols: cols as u32,
    };
    let mut tcb = TokenCommandBuffer::new(ctx);
    tcb.dispatch_threads(kernel, (512, 1, 1), (256, 1, 1), |enc| {
        enc.set_buffer(0, Some(&weights), 0);
        enc.set_buffer(1, Some(&x_buf), 0);
        enc.set_buffer(2, Some(&out), 0);
        enc.set_bytes(
            3,
            std::mem::size_of::<Raw32Params>() as u64,
            &params as *const Raw32Params as *const _,
        );
    })
    .expect("source-packed gravity dispatch");
    assert_eq!(tcb.dispatch_count(), 1);
    tcb.commit_and_wait()
        .expect("source-packed gravity completion");
    let gpu = read_f32_buf(&out, rows);
    for (row, (actual, expected)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-5,
            "{kernel} row {row}: gpu={actual} cpu={expected}"
        );
    }
}

#[test]
fn gravity_raw_q8_0_stays_packed_and_matches_cpu() {
    assert_source_packed_matvec("gravity_raw_q8_0_matvec", GgmlType::Q8_0, source_q8(11, 96));
}

#[test]
fn gravity_raw_q5_0_stays_packed_and_matches_cpu() {
    assert_source_packed_matvec("gravity_raw_q5_0_matvec", GgmlType::Q5_0, source_q5(11, 96));
}

#[test]
fn gravity_raw_q5_0_pair_stays_packed_and_matches_two_cpu_waves() {
    let rows = 11usize;
    let cols = 96usize;
    let gate = source_q5(rows, cols);
    let mut up = source_q5(rows, cols);
    // Make the second source tensor observably distinct without changing its
    // grammar, so a mistaken double-bind of gate bytes cannot pass.
    for byte in up.iter_mut().step_by(22) {
        *byte ^= 0x11;
    }
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.193).sin() - 0.35).collect();
    let gate_cpu = cpu_matvec(GgmlType::Q5_0, &gate, rows, cols, &x);
    let up_cpu = cpu_matvec(GgmlType::Q5_0, &up, rows, cols, &x);
    let ctx = ctx();
    let gate_buf = ctx.new_buffer_with_bytes(&gate);
    let up_buf = ctx.new_buffer_with_bytes(&up);
    let x_buf = new_f32_buf(ctx, &x);
    let gate_out = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let up_out = ctx.new_buffer(rows * std::mem::size_of::<f32>());
    let params = RawQ5PairParams {
        rows: rows as u32,
        cols: cols as u32,
    };
    let mut tcb = TokenCommandBuffer::new(ctx);
    tcb.dispatch_threads(
        "gravity_raw_q5_0_pair_matvec",
        (1024, 1, 1),
        (256, 1, 1),
        |enc| {
            enc.set_buffer(0, Some(&gate_buf), 0);
            enc.set_buffer(1, Some(&up_buf), 0);
            enc.set_buffer(2, Some(&x_buf), 0);
            enc.set_buffer(3, Some(&gate_out), 0);
            enc.set_buffer(4, Some(&up_out), 0);
            enc.set_bytes(
                5,
                std::mem::size_of::<RawQ5PairParams>() as u64,
                &params as *const RawQ5PairParams as *const _,
            );
        },
    )
    .expect("source-packed pair dispatch");
    assert_eq!(tcb.dispatch_count(), 1);
    tcb.commit_and_wait()
        .expect("source-packed pair completion");
    for (label, actual, expected) in [
        ("gate", read_f32_buf(&gate_out, rows), gate_cpu),
        ("up", read_f32_buf(&up_out, rows), up_cpu),
    ] {
        for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 2e-5,
                "{label} row {row}: gpu={actual} cpu={expected}"
            );
        }
    }
}

#[test]
fn gravity_raw_q5q5qv_fuses_source_bias_rope_and_kv_append() {
    const COLS: usize = 96;
    const HEAD_DIM: usize = 4;
    const Q_ROWS: usize = 8;
    const KV_ROWS: usize = 4;
    let q = source_q5(Q_ROWS, COLS);
    let k = source_q5(KV_ROWS, COLS);
    let v = source_q8(KV_ROWS, COLS);
    let x: Vec<f32> = (0..COLS)
        .map(|i| (i as f32 * 0.087).cos() - 0.125)
        .collect();
    let q_bias: Vec<f32> = (0..Q_ROWS).map(|i| (i as f32 - 3.0) * 0.01).collect();
    let k_bias: Vec<f32> = (0..KV_ROWS).map(|i| (i as f32 - 1.0) * -0.02).collect();
    let v_bias: Vec<f32> = (0..KV_ROWS).map(|i| (i as f32 - 2.0) * 0.03).collect();
    // head_dim/2 cosines followed by sines, matching Gravity's f64-built
    // runtime table contract and Qwen's split-half layout.
    let rope = [0.8f32, 0.6, 0.6, -0.8];
    let mut q_cpu = cpu_matvec(GgmlType::Q5_0, &q, Q_ROWS, COLS, &x);
    let mut k_cpu = cpu_matvec(GgmlType::Q5_0, &k, KV_ROWS, COLS, &x);
    let mut v_cpu = cpu_matvec(GgmlType::Q8_0, &v, KV_ROWS, COLS, &x);
    for (head, rows) in q_cpu.chunks_exact_mut(HEAD_DIM).enumerate() {
        for i in 0..HEAD_DIM / 2 {
            let a = rows[i] + q_bias[head * HEAD_DIM + i];
            let b = rows[i + HEAD_DIM / 2] + q_bias[head * HEAD_DIM + i + HEAD_DIM / 2];
            rows[i] = a * rope[i] - b * rope[HEAD_DIM / 2 + i];
            rows[i + HEAD_DIM / 2] = a * rope[HEAD_DIM / 2 + i] + b * rope[i];
        }
    }
    for i in 0..HEAD_DIM / 2 {
        let a = k_cpu[i] + k_bias[i];
        let b = k_cpu[i + HEAD_DIM / 2] + k_bias[i + HEAD_DIM / 2];
        k_cpu[i] = a * rope[i] - b * rope[HEAD_DIM / 2 + i];
        k_cpu[i + HEAD_DIM / 2] = a * rope[HEAD_DIM / 2 + i] + b * rope[i];
    }
    for (value, bias) in v_cpu.iter_mut().zip(&v_bias) {
        *value += bias;
    }

    let ctx = ctx();
    let q_buf = ctx.new_buffer_with_bytes(&q);
    let k_buf = ctx.new_buffer_with_bytes(&k);
    let v_buf = ctx.new_buffer_with_bytes(&v);
    let x_buf = new_f32_buf(ctx, &x);
    let rope_buf = new_f32_buf(ctx, &rope);
    let q_bias_buf = new_f32_buf(ctx, &q_bias);
    let k_bias_buf = new_f32_buf(ctx, &k_bias);
    let v_bias_buf = new_f32_buf(ctx, &v_bias);
    let q_out = ctx.new_buffer(Q_ROWS * std::mem::size_of::<f32>());
    let k_out = ctx.new_buffer(KV_ROWS * std::mem::size_of::<f32>());
    let v_out = ctx.new_buffer(KV_ROWS * std::mem::size_of::<f32>());
    let params = RawQ5Q5QvRopeAppendParams {
        q_rows: Q_ROWS as u32,
        kv_rows: KV_ROWS as u32,
        cols: COLS as u32,
        kv_off: 0,
        head_dim: HEAD_DIM as u32,
        has_q_bias: 1,
        has_k_bias: 1,
        has_v_bias: 1,
        v_is_q8: 1,
    };
    let mut tcb = TokenCommandBuffer::new(ctx);
    tcb.dispatch_threads(
        "gravity_raw_q5q5qv_rope_append",
        (768, 1, 1),
        (256, 1, 1),
        |enc| {
            enc.set_buffer(0, Some(&q_buf), 0);
            enc.set_buffer(1, Some(&k_buf), 0);
            enc.set_buffer(2, Some(&v_buf), 0);
            enc.set_buffer(3, Some(&x_buf), 0);
            enc.set_buffer(4, Some(&q_out), 0);
            enc.set_buffer(5, Some(&k_out), 0);
            enc.set_buffer(6, Some(&v_out), 0);
            enc.set_buffer(7, Some(&rope_buf), 0);
            enc.set_buffer(8, Some(&q_bias_buf), 0);
            enc.set_buffer(9, Some(&k_bias_buf), 0);
            enc.set_buffer(10, Some(&v_bias_buf), 0);
            enc.set_bytes(
                11,
                std::mem::size_of::<RawQ5Q5QvRopeAppendParams>() as u64,
                &params as *const RawQ5Q5QvRopeAppendParams as *const _,
            );
        },
    )
    .expect("fused raw Q5/Q8 dispatch");
    assert_eq!(tcb.dispatch_count(), 1);
    tcb.commit_and_wait().expect("fused raw Q5/Q8 completion");
    for (label, actual, expected) in [
        ("Q", read_f32_buf(&q_out, Q_ROWS), q_cpu),
        ("K", read_f32_buf(&k_out, KV_ROWS), k_cpu),
        ("V", read_f32_buf(&v_out, KV_ROWS), v_cpu),
    ] {
        for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 3e-5,
                "fused {label} row {row}: gpu={actual} cpu={expected}"
            );
        }
    }
}
