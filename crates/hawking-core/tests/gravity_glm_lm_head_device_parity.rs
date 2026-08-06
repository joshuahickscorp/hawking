#![cfg(target_os = "macos")]
use hawking_core::gravity::{matvec_bf16_host_accumulation, NativeBf16Accumulation};
use hawking_core::metal::MetalContext;
use hawking_core::numeric_parity::{
    format_score_line, matvec_bf16_f64_authority, score_pair, Bounds, SCHEMA,
};
fn activations(cols: usize) -> Vec<(&'static str, Vec<f32>)> {
    vec![
        ("ramp", (0..cols).map(|c| (c as f32) * 0.01 - 0.3).collect()),
        (
            "mod17",
            (0..cols)
                .map(|c| ((c * 5 + 3) % 17) as f32 * 0.05 - 0.4)
                .collect(),
        ),
        ("ones", vec![1.0; cols]),
        ("zeros", vec![0.0; cols]),
        (
            "block3",
            (0..cols)
                .map(|c| if c % 3 == 0 { 0.5 } else { -0.25 })
                .collect(),
        ),
        ("sin", (0..cols).map(|c| ((c as f32).sin()) * 0.1).collect()),
        (
            "half",
            (0..cols)
                .map(|c| if c < cols / 2 { 0.125 } else { -0.0625 })
                .collect(),
        ),
        (
            "mod31",
            (0..cols)
                .map(|c| ((c * 13 + 7) % 31) as f32 * 0.02 - 0.3)
                .collect(),
        ),
        (
            "gaussish",
            (0..cols)
                .map(|c| {
                    let u = ((c * 17 + 5) % 100) as f32 / 100.0 - 0.5;
                    (u * 3.0).tanh() // ≈ (-0.9, 0.9)
                })
                .collect(),
        ),
    ]
}
fn make_bf16_matrix_unit_scale(rows: usize, cols: usize, salt: u32) -> Vec<u8> {
    let mut bits = Vec::with_capacity(rows * cols * 2);
    for i in 0..(rows * cols) {
        let h = (i as u32).wrapping_mul(37).wrapping_add(salt);
        let sign = ((h >> 15) & 1) as u16;
        let exp = (120 + (h % 15)) as u16; // bf16 exp in [120, 134] → ~0.0078 … ~256
        let mant = (h & 0x7f) as u16;
        let u = (sign << 15) | (exp << 7) | mant;
        bits.extend_from_slice(&u.to_le_bytes());
    }
    bits
}
#[test]
fn device_bf16_lm_head_v21_against_f64_over_several_vectors() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("shader") && !msg.contains("compile"),
                "Metal is present but the shader failed to compile -- this is a real \
                 failure, not a skip: {msg}"
            );
            eprintln!("skip: no Metal device ({e})");
            return;
        }
    };
    let bounds = Bounds::logits();
    let shapes = [(64usize, 32usize), (257usize, 17usize), (16usize, 64usize)];
    let mut any_accurate_fail = false;
    let mut sequential_v21_failures = 0usize;
    let mut neumaier_v21_failures = 0usize;
    for &(rows, cols) in &shapes {
        let weight = make_bf16_matrix_unit_scale(rows, cols, (rows * cols) as u32);
        let w_buf = ctx
            .new_buffer_with_bytes_checked(&weight)
            .expect("upload weight");
        for (vi, (name, x)) in activations(cols).into_iter().enumerate() {
            let reference = matvec_bf16_f64_authority(&weight, cols, &x)
                .unwrap_or_else(|e| panic!("f64 authority failed: {e}"));
            for accumulation in NativeBf16Accumulation::ALL {
                let host = matvec_bf16_host_accumulation(&weight, cols, &x, accumulation)
                    .unwrap_or_else(|e| panic!("host {} failed: {e}", accumulation.as_str()));
                let device =
                    hawking_core::gravity_glm::gpu::dispatch_gemv_native_bf16_accumulation(
                        &ctx,
                        &w_buf,
                        rows as u32,
                        cols as u32,
                        &x,
                        accumulation,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "device {} failed (rows={rows} cols={cols} vec={vi}/{name}): {e} — \
                             shader compile failures are hard failures",
                            accumulation.as_str()
                        );
                    });
                assert_eq!(device.len(), host.len());
                assert_eq!(device.len(), reference.len());
                assert_eq!(
                    device,
                    host,
                    "Metal/host f32 comparator mismatch: accumulation={} \
                     rows={rows} cols={cols} vec={vi}/{name}",
                    accumulation.as_str()
                );
                let paired = score_pair(&host, &device, &reference, &bounds);
                match accumulation {
                    NativeBf16Accumulation::Sequential => {
                        if !paired.pass {
                            sequential_v21_failures += 1;
                        }
                    }
                    NativeBf16Accumulation::Neumaier => {
                        if !paired.pass {
                            neumaier_v21_failures += 1;
                        }
                    }
                    NativeBf16Accumulation::NeumaierCompensatedProduct => {
                        if !paired.pass {
                            any_accurate_fail = true;
                        }
                        assert!(
                            paired.pass,
                            "accurate candidate failed unchanged V2.1: rows={rows} cols={cols} \
                             vec={vi}/{name}\n  host: {:?}\n  device: {:?}",
                            paired.host.failures, paired.device.failures
                        );
                    }
                }
            }
        }
    }
    assert!(
        !any_accurate_fail,
        "internal bookkeeping: accurate failure set but asserts passed"
    );
}
