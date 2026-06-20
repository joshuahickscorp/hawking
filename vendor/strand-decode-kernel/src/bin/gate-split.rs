
use std::time::Instant;

use strand_decode_kernel::block_walk::gate_proto::{canonical_configs, machine_stamp, synth_encoded};
use strand_decode_kernel::fused::{fused_gemm, fused_gemm_factored, fused_gemm_scalar_mac};
use strand_decode_kernel::gemv::decode_q12_fast;
use strand_decode_kernel::gemv_par::decode_q12_par;
use strand_decode_kernel::split_decode::{
    decode_q12_split, decode_q12_split_par, decode_q12_split_par_with_lut,
    decode_q12_split_with_lut,
};
use strand_quant::decode::{decode_tensor_fixed, decode_tensor_fixed_with_lut};
use strand_quant::encode::{encode_tensor, encode_tensor_with, EncodeOpts, EncodedTensor};
use strand_quant::TrellisConfig;

type DecodeFn = fn(&EncodedTensor, &TrellisConfig) -> Vec<i32>;

fn k_split(e: &EncodedTensor, c: &TrellisConfig) -> Vec<i32> {
    decode_q12_split(e, c)
}
fn k_split_par(e: &EncodedTensor, c: &TrellisConfig) -> Vec<i32> {
    decode_q12_split_par(e, c)
}

const KERNELS: &[(&str, DecodeFn)] = &[("split", k_split), ("split-par", k_split_par)];

fn identity_matrix() -> usize {
    let mut checked = 0usize;
    for (cfg, label) in canonical_configs() {
        for seed in 0..24u64 {
            let n = 1 + (seed as usize * 211) % 4096;
            let w: Vec<f32> =
                (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect();
            let variants = [
                encode_tensor(&w, &cfg),
                encode_tensor_with(&w, &cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
                encode_tensor_with(&w, &cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
                encode_tensor_with(
                    &w,
                    &cfg,
                    &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
                ),
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
        let cfg = TrellisConfig::for_bpw(3.0).with_vec_dim(2);
        let (ns, d) = (cfg.num_states(), cfg.vec_dim());
        let lut: Vec<i32> = (0..ns * d)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 20) as i32 - 2048)
            .collect();
        let w: Vec<f32> = (0..700).map(|i| ((i as f32) * 0.011).cos() * 0.3).collect();
        let enc = encode_tensor(&w, &cfg);
        let want = decode_tensor_fixed_with_lut(&enc, &cfg, &lut);
        assert_eq!(decode_q12_split_with_lut(&enc, &cfg, &lut), want, "split vec fallback");
        assert_eq!(decode_q12_split_par_with_lut(&enc, &cfg, &lut), want, "split-par vec fallback");
        checked += 2;
    }
    checked
}

fn factored_tolerance_gate() -> (usize, f64) {
    const REL_SLACK: f32 = 1e-3; 
    const ABS_SLACK: f32 = 1e-4;
    let mut checked = 0usize;
    let mut worst: f64 = 0.0; 
    let configs = [
        TrellisConfig::for_bpw(3.0),
        TrellisConfig::for_bpw_l(2.0, 12),
        TrellisConfig::for_bpw_l(2.0, 5), 
    ];
    for cfg in &configs {
        for &(rows, cols) in &[(16usize, 256usize), (37, 300), (9, 1024)] {
            let n = rows * cols;
            let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0113).sin() * 0.6).collect();
            for opts in [
                EncodeOpts::default(),
                EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() },
            ] {
                let enc = encode_tensor_with(&w, cfg, &opts);
                for &batch in &[1usize, 4, 5, 16, 21, 64] {
                    let xs: Vec<f32> = (0..batch * cols)
                        .map(|i| ((i as f32) * 0.0713).cos())
                        .collect();
                    let y_ref = fused_gemm_scalar_mac(&enc, cfg, None, rows, cols, &xs, batch);
                    let y_fac = fused_gemm_factored(&enc, cfg, None, rows, cols, &xs, batch);
                    assert_eq!(y_ref.len(), y_fac.len());
                    
                    let budget: Vec<f32> = (0..batch)
                        .map(|b| {
                            xs[b * cols..(b + 1) * cols].iter().map(|v| v.abs()).sum::<f32>()
                                / 4096.0
                        })
                        .collect();
                    
                    for (i, (&r, &f)) in y_ref.iter().zip(y_fac.iter()).enumerate() {
                        let b = i % batch;
                        let tol = ABS_SLACK + REL_SLACK * r.abs() + budget[b];
                        let err = (r - f).abs();
                        worst = worst.max((err / tol) as f64);
                        assert!(
                            err <= tol,
                            "FACTORED TOLERANCE VIOLATION at flat {i}: ref={r} fac={f} \
                             err={err} tol={tol} (L={} k={} rows={rows} cols={cols} \
                             batch={batch} affine={})",
                            cfg.l_bits, cfg.k_bits, enc.has_affine_min
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    (checked, worst)
}

const EST_CLOCK_GHZ: f64 = 4.05;

fn bench_decode() {
    let (out_f, in_f) = (18944usize, 3584usize);
    let total = out_f * in_f;
    println!(
        "\n== dense decode bench (best-of-3, ffn_down {out_f}x{in_f} = {:.1}M weights) ==",
        total as f64 / 1e6
    );
    println!("  {}", machine_stamp());
    println!("  (cyc/w = estimate at {EST_CLOCK_GHZ} GHz, single-thread rows only)");
    type Row = (&'static str, DecodeFn, bool); 
    let rows: &[Row] = &[
        ("decode_q12_fast (baseline 1T)", |e, c| decode_q12_fast(e, c), true),
        ("split (1T)", k_split, true),
        ("decode_q12_par (baseline MT)", |e, c| decode_q12_par(e, c), false),
        ("split-par (MT)", k_split_par, false),
    ];
    for (cfg, label) in [
        (TrellisConfig::for_bpw(3.0), "3-bit deploy (k=3,L=7)"),
        (TrellisConfig::for_bpw_l(2.0, 12), "2-bit reopen (k=2,L=12)"),
    ] {
        println!("  -- {label} --");
        let enc = synth_encoded(total, cfg.k_bits, 256);
        for (name, f, single) in rows {
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                let out = f(&enc, &cfg);
                let dt = t.elapsed().as_secs_f64();
                std::hint::black_box(&out);
                best = best.min(dt);
            }
            let gws = total as f64 / best / 1e9;
            if *single {
                println!(
                    "  {name:<30} {:>8.1} ms   {:>6.3} Gw/s   ~{:>4.2} cyc/w",
                    best * 1e3,
                    gws,
                    EST_CLOCK_GHZ / gws
                );
            } else {
                println!("  {name:<30} {:>8.1} ms   {:>6.3} Gw/s", best * 1e3, gws);
            }
        }
    }
}

fn bench_factored() {
    let (out_f, in_f) = (18944usize, 3584usize);
    let total = out_f * in_f;
    println!("\n== fused factoring bench (best-of-3, ffn_down {out_f}x{in_f}) ==");
    println!("  {}", machine_stamp());
    for (cfg, label) in [
        (TrellisConfig::for_bpw(3.0), "3-bit deploy (k=3,L=7)"),
        (TrellisConfig::for_bpw_l(2.0, 12), "2-bit reopen (k=2,L=12)"),
    ] {
        println!("  -- {label} --");
        let enc = synth_encoded(total, cfg.k_bits, 256);
        for &batch in &[1usize, 4, 16, 64] {
            let xs: Vec<f32> =
                (0..batch * in_f).map(|i| ((i as f32) * 0.0713).cos()).collect();
            let mut t_neon = f64::INFINITY;
            let mut t_scalar = f64::INFINITY;
            let mut t_fac = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                let y = fused_gemm(&enc, &cfg, None, out_f, in_f, &xs, batch);
                t_neon = t_neon.min(t.elapsed().as_secs_f64());
                std::hint::black_box(&y);
                let t = Instant::now();
                let y = fused_gemm_scalar_mac(&enc, &cfg, None, out_f, in_f, &xs, batch);
                t_scalar = t_scalar.min(t.elapsed().as_secs_f64());
                std::hint::black_box(&y);
                let t = Instant::now();
                let y = fused_gemm_factored(&enc, &cfg, None, out_f, in_f, &xs, batch);
                t_fac = t_fac.min(t.elapsed().as_secs_f64());
                std::hint::black_box(&y);
            }
            println!(
                "  B={batch:<3} fused-NEON {:>7.1} ms | fused-scalar {:>7.1} ms | \
                 FACTORED {:>7.1} ms  ({:.2}x vs scalar, {:.2}x vs NEON)",
                t_neon * 1e3,
                t_scalar * 1e3,
                t_fac * 1e3,
                t_scalar / t_fac,
                t_neon / t_fac
            );
        }
    }
}

fn main() {
    println!("gate-split — SCHISM #1 (misplaced scale multiply): identity + tolerance gates");
    let t = Instant::now();
    let dense = identity_matrix();
    println!(
        "GATE 1 dense identity: {dense} kernel×config×variant cells byte-identical to \
         decode_tensor_fixed in {:.1}s ✓",
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let (fac, worst) = factored_tolerance_gate();
    println!(
        "GATE 2 factored tolerance: {fac} output elements within the principled \
         truncation budget vs fused_gemm_scalar_mac (worst err = {worst:.3} of budget) \
         in {:.1}s ✓ [approximate BY DESIGN — opt-in, documented in fused.rs]",
        t.elapsed().as_secs_f64()
    );

    if std::env::args().any(|a| a == "--bench") {
        
        bench_decode();
        bench_factored();
    } else {
        println!("(pass --bench for the machine-stamped perf sweep)");
    }
}
