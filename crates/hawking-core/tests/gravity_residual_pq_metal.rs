#![cfg(target_os = "macos")]

use half::f16;
use hawking_core::gravity::{residual_pq_matvec_metal, ResidualPqMetalMatrix, ResidualPqTensor};
use hawking_core::metal::MetalContext;

fn u16le(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn u32le(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// A deliberately non-byte-aligned stream exercises the shared MSB-first
/// index grammar, including a tail index that crosses the final byte.
fn fixture() -> (Vec<u8>, Vec<f32>) {
    const D: usize = 4;
    const STAGES: usize = 3;
    const CARD: usize = 16;
    const ROWS: usize = 19;
    const CHUNKS: usize = 11;
    let mut payload = Vec::new();
    payload.extend_from_slice(b"LLM52RPK");
    u16le(&mut payload, D as u16);
    u16le(&mut payload, STAGES as u16);
    u16le(&mut payload, CARD as u16);
    u16le(&mut payload, 0);
    u32le(&mut payload, ROWS as u32);
    u32le(&mut payload, (CHUNKS * D) as u32);
    u32le(&mut payload, CHUNKS as u32);
    u32le(&mut payload, 0x52A1);
    u16le(&mut payload, 4);
    payload.push(0);
    payload.push(STAGES as u8);
    payload.resize(64, 0);
    for stage in 0..STAGES {
        for code in 0..CARD {
            for j in 0..D {
                let value = ((stage * 71 + code * 17 + j * 13) as f32 - 141.0) / 97.0;
                u16le(&mut payload, f16::from_f32(value).to_bits());
            }
        }
    }
    let mut byte = 0u8;
    let mut used = 0u8;
    for row in 0..ROWS {
        for chunk in 0..CHUNKS {
            for stage in 0..STAGES {
                let index = ((row * 7 + chunk * 11 + stage * 3) & 15) as u8;
                byte = (byte << 4) | index;
                used += 4;
                if used == 8 {
                    payload.push(byte);
                    byte = 0;
                    used = 0;
                }
            }
        }
    }
    if used != 0 {
        payload.push(byte << (8 - used));
    }
    let x = (0..CHUNKS * D)
        .map(|i| ((i as f32 * 0.171875).sin()) + (i as f32 - 21.0) / 64.0)
        .collect();
    (payload, x)
}

#[test]
fn residual_pq_metal_matches_direct_compact_cpu_authority() {
    let ctx = match MetalContext::new() {
        Ok(ctx) => ctx,
        Err(error) => {
            eprintln!("skipping: no Metal device available: {error}");
            return;
        }
    };
    let (payload, x) = fixture();
    let cpu = ResidualPqTensor::from_payload(&payload)
        .unwrap()
        .matvec(&x)
        .unwrap();
    let gpu = residual_pq_matvec_metal(&ctx, &payload, &x).expect("Metal residual PQ");
    assert_eq!(gpu.len(), cpu.len());
    for (row, (actual, expected)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-4,
            "row {row}: GPU {actual:?} != CPU {expected:?}"
        );
    }
}

/// Real Llama-3.1-8B FFN gate geometry.  This is intentionally ignored in
/// normal CI because it is a hardware throughput probe, not a unit test.
#[test]
#[ignore = "manual Apple-GPU roofline probe"]
fn residual_pq_four_stage_ffn_gate_geometry_roofline() {
    const ROWS: usize = 14_336;
    const D: usize = 8;
    const CHUNKS: usize = 512;
    const STAGES: usize = 4;
    const CARD: usize = 128;
    const BITS: usize = 7;
    let ctx = MetalContext::new().expect("Metal device");
    let mut payload =
        Vec::with_capacity(64 + STAGES * CARD * D * 2 + ROWS * CHUNKS * STAGES * BITS / 8);
    payload.extend_from_slice(b"LLM52RPK");
    u16le(&mut payload, D as u16);
    u16le(&mut payload, STAGES as u16);
    u16le(&mut payload, CARD as u16);
    u16le(&mut payload, 0);
    u32le(&mut payload, ROWS as u32);
    u32le(&mut payload, (D * CHUNKS) as u32);
    u32le(&mut payload, CHUNKS as u32);
    u32le(&mut payload, 0x52A1);
    u16le(&mut payload, BITS as u16);
    payload.push(0);
    payload.push(STAGES as u8);
    payload.resize(64, 0);
    for stage in 0..STAGES {
        for code in 0..CARD {
            for j in 0..D {
                u16le(
                    &mut payload,
                    f16::from_f32((stage + code + j) as f32 / 1024.0).to_bits(),
                );
            }
        }
    }
    // Index zero everywhere; 7-bit MSB stream with the canonical padded tail.
    payload.resize(
        payload.len() + (ROWS * CHUNKS * STAGES * BITS).div_ceil(8),
        0,
    );
    let matrix = ResidualPqMetalMatrix::from_payload(&ctx, &payload).expect("resident matrix");
    let x: Vec<f32> = (0..D * CHUNKS).map(|i| (i as f32 * 0.017).sin()).collect();
    for _ in 0..4 {
        matrix.matvec(&ctx, &x).unwrap();
    }
    let mut samples = Vec::new();
    for _ in 0..12 {
        let started = std::time::Instant::now();
        matrix.matvec(&ctx, &x).unwrap();
        samples.push(started.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    eprintln!("residual-PQ FFN gate D8/S4/K128 resident median_us={median:.3}, implied_max_gate_per_s={:.2}", 1e6 / median);
}

/// Same FFN gate, but one D32 stage: this separates additive-stage cost from
/// the inherent direct-codebook execution cost.
#[test]
#[ignore = "manual Apple-GPU roofline probe"]
fn residual_pq_single_stage_d32_ffn_gate_geometry_roofline() {
    const ROWS: usize = 14_336;
    const D: usize = 32;
    const CHUNKS: usize = 128;
    const STAGES: usize = 1;
    const CARD: usize = 256;
    const BITS: usize = 8;
    let ctx = MetalContext::new().expect("Metal device");
    let mut payload = Vec::with_capacity(64 + STAGES * CARD * D * 2 + ROWS * CHUNKS * STAGES);
    payload.extend_from_slice(b"LLM52RPK");
    u16le(&mut payload, D as u16);
    u16le(&mut payload, STAGES as u16);
    u16le(&mut payload, CARD as u16);
    u16le(&mut payload, 0);
    u32le(&mut payload, ROWS as u32);
    u32le(&mut payload, (D * CHUNKS) as u32);
    u32le(&mut payload, CHUNKS as u32);
    u32le(&mut payload, 0x52A1);
    u16le(&mut payload, BITS as u16);
    payload.push(0);
    payload.push(STAGES as u8);
    payload.resize(64, 0);
    for code in 0..CARD {
        for j in 0..D {
            u16le(
                &mut payload,
                f16::from_f32((code + j) as f32 / 1024.0).to_bits(),
            );
        }
    }
    payload.resize(payload.len() + ROWS * CHUNKS, 0);
    let matrix = ResidualPqMetalMatrix::from_payload(&ctx, &payload).expect("resident matrix");
    let x: Vec<f32> = (0..D * CHUNKS).map(|i| (i as f32 * 0.017).sin()).collect();
    for _ in 0..4 {
        matrix.matvec(&ctx, &x).unwrap();
    }
    let mut samples = Vec::new();
    for _ in 0..12 {
        let started = std::time::Instant::now();
        matrix.matvec(&ctx, &x).unwrap();
        samples.push(started.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    eprintln!("residual-PQ FFN gate D32/S1/K256 resident median_us={median:.3}, implied_max_gate_per_s={:.2}", 1e6 / median);
}
