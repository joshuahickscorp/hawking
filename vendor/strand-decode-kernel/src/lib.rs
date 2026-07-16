use strand_quant::decode::{decode_tensor_fixed, decode_tensor_fixed_with_lut};
use strand_quant::encode::EncodedTensor;
use strand_quant::TrellisConfig;

pub mod block_walk;

pub mod loader;

pub mod gemv;

pub mod outlier_mac;

pub mod gemv_par;

pub mod interleave;

pub mod fused;

pub mod histogram_gemv;

pub mod silence;

pub mod event_mac;

pub mod split_decode;

pub mod paired_lut;

pub mod neon_lut;

pub mod prepared;

#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(test)]
mod e2e;

pub fn decode_weights_q12(enc: &EncodedTensor, cfg: &TrellisConfig, lut: Option<&[i32]>) -> Vec<i32> {
    match lut {
        None => decode_tensor_fixed(enc, cfg),
        Some(l) => decode_tensor_fixed_with_lut(enc, cfg, l),
    }
}

pub fn matvec(enc: &EncodedTensor, cfg: &TrellisConfig, lut: Option<&[i32]>, out_features: usize, in_features: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), in_features, "x must have in_features entries");
    let w = decode_weights_q12(enc, cfg, lut);
    assert_eq!(w.len(), out_features * in_features, "decoded weight count mismatch");
    let inv = 1.0f32 / 4096.0;
    let mut y = vec![0.0f32; out_features];
    for o in 0..out_features {
        let row = &w[o * in_features..(o + 1) * in_features];
        let mut acc = 0.0f32;
        for i in 0..in_features {
            acc += (row[i] as f32) * inv * x[i];
        }
        y[o] = acc;
    }
    y
}

pub fn footprint_bytes(num_weights: usize, bpw: f64) -> usize {
    ((num_weights as f64) * bpw / 8.0).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use strand_quant::encode::encode_tensor;

    #[test]
    fn matvec_matches_manual_decode() {
        let (out, inf) = (8usize, 64usize);
        let weights: Vec<f32> = (0..out * inf).map(|i| (i as f32 * 0.123).sin()).collect();
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&weights, &cfg);
        let x: Vec<f32> = (0..inf).map(|i| (i as f32 * 0.07).cos()).collect();

        let y = matvec(&enc, &cfg, None, out, inf, &x);

        let w = decode_weights_q12(&enc, &cfg, None);
        let inv = 1.0f32 / 4096.0;
        assert_eq!(y.len(), out);
        for o in 0..out {
            let mut acc = 0.0f32;
            for i in 0..inf {
                acc += (w[o * inf + i] as f32) * inv * x[i];
            }
            assert!((y[o] - acc).abs() < 1e-6, "row {o}: {} vs {}", y[o], acc);
        }
    }

    #[test]
    fn footprint_scales_with_bpw() {
        assert_eq!(footprint_bytes(8, 16.0), 16);

        assert!(footprint_bytes(7_000_000_000, 3.0) * 5 < footprint_bytes(7_000_000_000, 16.0));
    }
}
