//! TQ G4 bitslice GPU↔CPU bit-identity gate (Slice 3).
//!
//! The non-negotiable contract of the TQ Metal port: the GPU `strand_bitslice_decode`
//! kernel's Q12 output is byte-for-byte equal to the integer CPU oracle
//! `strand_quant::decode::decode_tensor_fixed` — the same determinism contract the
//! CPU serving reference (`crate::tq`) honours. A fast wrong kernel is worse than no
//! kernel, so perf is never measured here; only identity.
//!
//! Drives the public, `BitsliceEntry`-free entry point
//! `dismantle_core::gpu_decode_q12` (bake → pin payload+table → dispatch decode →
//! read back `Vec<i32>`), swept over the encode-lever matrix: k ∈ {2,3,4},
//! L ∈ {7,12}, tail-biting × affine-min, and edge lengths (short final block,
//! sub-block tails, 1-weight tensors). Skips cleanly when no Metal device is
//! present (never a fake pass).
//!
//! Run with:
//!   cargo test -p dismantle-core --features tq --test tq_trellis_parity -- --nocapture
//!
//! The whole file is gated on macOS + `tq` (the GPU path and the `strand_quant`
//! dep only exist there).

#![cfg(all(target_os = "macos", feature = "tq"))]

use dismantle_core::gpu_decode_q12;
use dismantle_core::metal::MetalContext;
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts};
use strand_quant::TrellisConfig;

/// Deterministic synthetic weights (a smooth signal so the encoder exercises a
/// spread of trellis states, parameterised by `seed` for edge-length coverage).
fn synth_w(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5)
        .collect()
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
    // Bit-for-bit (these are integers; == is exact).
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

    // Probe the stride contract once up front, with a clear message: the GPU
    // sizeof(BitsliceEntry) must equal the host #[repr(C)] size (84 B) or every
    // assertion below would be meaningless. The decode path also re-checks it,
    // but surfacing it here makes a stride mismatch unmistakable.
    {
        let cfg = TrellisConfig::for_bpw(3.0);
        let enc = encode_tensor(&synth_w(256, 0), &cfg);
        // A trivial decode that, if it returns Ok, proves the probe passed.
        let r = gpu_decode_q12(&ctx, &enc, &cfg)
            .expect("scalar bake")
            .expect("stride probe + decode");
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
                assert_gpu_eq_cpu(
                    &ctx,
                    &enc,
                    &cfg,
                    &format!("{cfg_label} n={n} seed={seed} plain"),
                );

                // tail-biting (the stored-vs-walked init_state branch)
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

                // affine-min (the off[8] add path)
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

                // tail-biting + affine-min together
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
    for (cfg, cfg_label) in [
        (TrellisConfig::for_bpw(3.0), "k3 L7"),
        (TrellisConfig::for_bpw_l(2.0, 12), "k2 L12"),
    ] {
        let w = synth_w(total, 0xABCD);
        let enc = encode_tensor(&w, &cfg);
        assert_gpu_eq_cpu(&ctx, &enc, &cfg, &format!("{cfg_label} wide {rows}x{cols}"));
    }
    println!("[tq_trellis_parity] wide-shape GPU decode bit-identical to oracle");
}
