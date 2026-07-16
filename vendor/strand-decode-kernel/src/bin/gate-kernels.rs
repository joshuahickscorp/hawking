use std::time::Instant;

use strand_decode_kernel::block_walk::gate_proto::{canonical_configs, machine_stamp, synth_encoded};
use strand_decode_kernel::fused::fused_gemm_with_q12;
use strand_decode_kernel::gemv::decode_q12_fast;
use strand_decode_kernel::gemv_par::{decode_q12_par, decode_q12_simd};
use strand_decode_kernel::interleave::{decode_q12_interleave, decode_q12_interleave_par};
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts, EncodedTensor};
use strand_quant::TrellisConfig;

type DecodeFn = fn(&EncodedTensor, &TrellisConfig) -> Vec<i32>;

fn k_fast(e: &EncodedTensor, c: &TrellisConfig) -> Vec<i32> {
    decode_q12_fast(e, c)
}
fn k_par(e: &EncodedTensor, c: &TrellisConfig) -> Vec<i32> {
    decode_q12_par(e, c)
}
fn k_simd(e: &EncodedTensor, c: &TrellisConfig) -> Vec<i32> {
    decode_q12_simd(e, c)
}
fn k_il2(e: &EncodedTensor, c: &TrellisConfig) -> Vec<i32> {
    decode_q12_interleave::<2>(e, c)
}
fn k_il4(e: &EncodedTensor, c: &TrellisConfig) -> Vec<i32> {
    decode_q12_interleave::<4>(e, c)
}
fn k_il_par4(e: &EncodedTensor, c: &TrellisConfig) -> Vec<i32> {
    decode_q12_interleave_par::<4>(e, c)
}

fn k_fused_b1(e: &EncodedTensor, c: &TrellisConfig) -> Vec<i32> {
    let x = vec![0.5f32; e.total];
    fused_gemm_with_q12(e, c, None, 1, e.total, &x, 1).1
}
fn k_fused_b4(e: &EncodedTensor, c: &TrellisConfig) -> Vec<i32> {
    let xs = vec![0.5f32; 4 * e.total];
    fused_gemm_with_q12(e, c, None, 1, e.total, &xs, 4).1
}

const KERNELS: &[(&str, DecodeFn)] =
    &[("fast", k_fast), ("par", k_par), ("simd", k_simd), ("interleave-s2", k_il2), ("interleave-s4", k_il4), ("interleave-par-s4", k_il_par4), ("fused-b1", k_fused_b1), ("fused-b4", k_fused_b4)];

fn identity_matrix() -> usize {
    let mut checked = 0usize;
    for (cfg, label) in canonical_configs() {
        for seed in 0..24u64 {
            let n = 1 + (seed as usize * 211) % 4096;
            let w: Vec<f32> = (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect();
            let variants = [
                encode_tensor(&w, &cfg),
                encode_tensor_with(&w, &cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
                encode_tensor_with(&w, &cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
                encode_tensor_with(&w, &cfg, &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() }),
            ];
            for enc in &variants {
                let reference = decode_tensor_fixed(enc, &cfg);
                for (name, f) in KERNELS {
                    let got = f(enc, &cfg);
                    assert_eq!(
                        got, reference,
                        "IDENTITY VIOLATION: kernel `{name}` diverged from \
                         decode_tensor_fixed at {label}, n={n}, seed={seed}, tail={}, \
                         affine={} — release blocker",
                        enc.tail_biting, enc.has_affine_min
                    );
                    checked += 1;
                }
            }
        }
    }

    {
        use strand_decode_kernel::gemv::decode_q12_fast_with_lut;
        use strand_decode_kernel::gemv_par::{decode_q12_par_with_lut, decode_q12_simd_with_lut};
        use strand_decode_kernel::interleave::decode_q12_interleave_with_lut;
        use strand_quant::decode::decode_tensor_fixed_with_lut;

        let cfg = TrellisConfig::for_bpw(3.0).with_vec_dim(2);
        let (ns, d) = (cfg.num_states(), cfg.vec_dim());
        let lut: Vec<i32> = (0..ns * d).map(|i| ((i as u32).wrapping_mul(2654435761) >> 20) as i32 - 2048).collect();
        let w: Vec<f32> = (0..700).map(|i| ((i as f32) * 0.011).cos() * 0.3).collect();
        let enc = encode_tensor(&w, &cfg);
        let want = decode_tensor_fixed_with_lut(&enc, &cfg, &lut);
        assert_eq!(decode_q12_fast_with_lut(&enc, &cfg, &lut), want, "fast vec fallback");
        assert_eq!(decode_q12_par_with_lut(&enc, &cfg, &lut), want, "par vec fallback");
        assert_eq!(decode_q12_simd_with_lut(&enc, &cfg, &lut), want, "simd vec fallback");
        assert_eq!(decode_q12_interleave_with_lut::<4>(&enc, &cfg, &lut), want, "interleave vec fallback");
        checked += 4;
    }
    checked
}

fn bench() {
    let (out_f, in_f) = (18944usize, 3584usize);
    let total = out_f * in_f;
    println!("\n== bench (best-of-3, ffn_down {out_f}x{in_f} = {:.1}M weights) ==", total as f64 / 1e6);
    println!("  {}", machine_stamp());
    for (cfg, label) in [(TrellisConfig::for_bpw(3.0), "3-bit deploy"), (TrellisConfig::for_bpw_l(2.0, 12), "2-bit reopen")] {
        println!("  -- {label} (k={} L={}) --", cfg.k_bits, cfg.l_bits);
        let enc = synth_encoded(total, cfg.k_bits, 256);
        for (name, f) in KERNELS {
            if name.starts_with("fused") {
                continue;
            }
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                let out = f(&enc, &cfg);
                let dt = t.elapsed().as_secs_f64();
                std::hint::black_box(&out);
                best = best.min(dt);
            }
            println!("  {name:<18} {:>8.1} ms   {:>6.2} Gw/s", best * 1e3, total as f64 / best / 1e9);
        }
    }
    println!("  (fused perf lives in gate-fused — the MAC is the product number there)");
}

fn main() {
    println!("gate-kernels — THE decode-path identity registry ({} kernels)", KERNELS.len());
    let t = Instant::now();
    let checked = identity_matrix();
    println!(
        "identity: {} kernel×config×variant cells byte-identical to decode_tensor_fixed \
         in {:.1}s ✓",
        checked,
        t.elapsed().as_secs_f64()
    );

    if std::env::args().any(|a| a == "--bench") {
        bench();
    } else {
        println!("(pass --bench for the machine-stamped throughput sweep)");
    }
}
