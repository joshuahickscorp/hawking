#![cfg(all(target_os = "macos", feature = "tq"))]
use hawking_core::gpu_decode_q12;
use hawking_core::metal::MetalContext;
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};
use strand_quant::TrellisConfig;
fn synth_w(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5)
        .collect()
}
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
fn assert_gpu_eq_cpu(
    ctx: &MetalContext,
    enc: &strand_quant::encode::EncodedTensor,
    cfg: &TrellisConfig,
    label: &str,
) {
    let got = gpu_decode_q12(ctx, enc, cfg)
        .unwrap_or_else(|| panic!("{label}: gpu_decode_q12 returned None (bake rejected?)"))
        .unwrap_or_else(|e| panic!("{label}: GPU decode error: {e}"));
    let want = decode_tensor_fixed(enc, cfg);
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length mismatch GPU {} vs CPU {}",
        got.len(),
        want.len()
    );
    if got != want {
        let first = got
            .iter()
            .zip(want.iter())
            .enumerate()
            .find(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| (i, *a, *b));
        panic!("{label}: GPU Q12 != CPU oracle bit-for-bit; first diff = {first:?}");
    }
}
#[test]
fn bitslice_gpu_decode_matches_cpu_oracle_over_matrix() {
    let Ok(ctx) = MetalContext::new() else {
        eprintln!("[tq_trellis_parity] no Metal device; skipping GPU↔CPU gate");
        return;
    };
    {
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&synth_w(256, 0), &cfg);
        let r = gpu_decode_q12(&ctx, &enc, &cfg)
            .expect("scalar bake")
            .expect("stride probe + decode");
        assert_eq!(r.len(), 256);
    }
    let lengths = [1usize, 7, 31, 32, 33, 255, 256, 257, 288, 512, 1000, 2049];
    for (cfg, cfg_label) in gate_configs() {
        for &n in &lengths {
            for seed in 0..4u64 {
                let w = synth_w(n, seed);
                let enc = encode_tensor(&w, &cfg);
                assert_gpu_eq_cpu(
                    &ctx,
                    &enc,
                    &cfg,
                    &format!("{cfg_label} n={n} seed={seed} plain"),
                );
                let enc_tb = encode_tensor_with(
                    &w,
                    &cfg,
                    &EncodeOpts {
                        tail_biting: true,
                        ..Default::default()
                    },
                );
                assert_gpu_eq_cpu(
                    &ctx,
                    &enc_tb,
                    &cfg,
                    &format!("{cfg_label} n={n} seed={seed} tail_biting"),
                );
                let enc_am = encode_tensor_with(
                    &w,
                    &cfg,
                    &EncodeOpts {
                        affine_min: true,
                        ..Default::default()
                    },
                );
                assert_gpu_eq_cpu(
                    &ctx,
                    &enc_am,
                    &cfg,
                    &format!("{cfg_label} n={n} seed={seed} affine_min"),
                );
                let enc_both = encode_tensor_with(
                    &w,
                    &cfg,
                    &EncodeOpts {
                        tail_biting: true,
                        affine_min: true,
                        ..Default::default()
                    },
                );
                assert_gpu_eq_cpu(
                    &ctx,
                    &enc_both,
                    &cfg,
                    &format!("{cfg_label} n={n} seed={seed} tail+affine"),
                );
            }
        }
    }
}
#[test]
fn bitslice_gpu_decode_matches_cpu_oracle_wide_shape() {
    let Ok(ctx) = MetalContext::new() else {
        eprintln!("[tq_trellis_parity] no Metal device; skipping wide-shape gate");
        return;
    };
    let (rows, cols) = (16usize, 2048usize); // 32768 weights, 128 blocks
    let total = rows * cols;
    for (cfg, cfg_label) in [
        (TrellisConfig::for_bpw(3.0), "k3 L7"),
        (TrellisConfig::for_bpw_l(2.0, 12), "k2 L12"),
    ] {
        let w = synth_w(total, 0xABCD);
        let enc = encode_tensor(&w, &cfg);
        assert_gpu_eq_cpu(&ctx, &enc, &cfg, &format!("{cfg_label} wide {rows}x{cols}"));
    }
}
#[test]
fn trellis_k1_l5_config_valid() {
    let cfg = TrellisConfig::for_bpw(1.0);
    assert_eq!(cfg.k_bits, 1, "for_bpw(1.0) must give k=1");
    assert_eq!(cfg.l_bits, 5, "for_bpw(1.0) must give L=k+4=5");
    assert_eq!(cfg.block_len, 256, "default block_len must be 256");
}
#[test]
fn trellis_k1_l7_explicit() {
    let cfg = TrellisConfig::new(7, 1, 256);
    assert_eq!(cfg.k_bits, 1, "explicit k=1 must be stored");
    assert_eq!(cfg.l_bits, 7, "explicit L=7 must be stored");
    assert_eq!(cfg.block_len, 256);
    assert_eq!(cfg.num_states(), 128, "2^7 = 128 trellis states");
}
#[test]
fn trellis_k1_l9_config() {
    let cfg = TrellisConfig::new(9, 1, 256);
    assert_eq!(cfg.k_bits, 1);
    assert_eq!(cfg.l_bits, 9);
    assert_eq!(cfg.num_states(), 512, "2^9 = 512 trellis states");
}
#[test]
#[ignore = "k=1 GPU path not yet validated — enable after kernel confirms k=1 coverage"]
fn trellis_k1_gpu_decode_parity() {
    let Ok(ctx) = MetalContext::new() else {
        eprintln!("[tq_trellis_parity] no Metal device; skipping k=1 GPU gate");
        return;
    };
    let cfg = TrellisConfig::new(7, 1, 256);
    let w = (0..256usize)
        .map(|i| ((i as f32) * 0.0137).sin() * 0.5)
        .collect::<Vec<_>>();
    let enc = strand_quant::encode::encode_tensor(&w, &cfg);
    assert_gpu_eq_cpu(&ctx, &enc, &cfg, "k=1 L=7 n=256 plain");
}
