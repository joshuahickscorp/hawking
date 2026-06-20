
use std::time::Instant;

use rayon::prelude::*;
use strand_decode_kernel::block_walk::gate_proto::{machine_stamp, synth_encoded};
use strand_decode_kernel::fused::{
    fused_gemm, fused_gemm_scalar_mac, fused_gemm_with_q12, fused_matvec,
};
use strand_decode_kernel::gemv_par::decode_q12_par;
use strand_quant::encode::EncodedTensor;
use strand_quant::TrellisConfig;

fn baseline_matvec(enc: &EncodedTensor, cfg: &TrellisConfig, out_f: usize, in_f: usize, x: &[f32]) -> Vec<f32> {
    let w = decode_q12_par(enc, cfg);
    let inv = 1.0f32 / 4096.0;
    (0..out_f)
        .into_par_iter()
        .map(|o| {
            let row = &w[o * in_f..(o + 1) * in_f];
            let mut acc = 0.0f32;
            for i in 0..in_f {
                acc += (row[i] as f32) * inv * x[i];
            }
            acc
        })
        .collect()
}

fn baseline_gemm(enc: &EncodedTensor, cfg: &TrellisConfig, out_f: usize, in_f: usize, xs: &[f32], b: usize) -> Vec<f32> {
    let w = decode_q12_par(enc, cfg);
    let inv = 1.0f32 / 4096.0;
    let mut y = vec![0.0f32; out_f * b];
    y.par_chunks_mut(b).enumerate().for_each(|(o, yo)| {
        let row = &w[o * in_f..(o + 1) * in_f];
        for (bb, slot) in yo.iter_mut().enumerate() {
            let x = &xs[bb * in_f..(bb + 1) * in_f];
            let mut acc = 0.0f32;
            for i in 0..in_f {
                acc += (row[i] as f32) * inv * x[i];
            }
            *slot = acc;
        }
    });
    y
}

fn bench<F: FnMut() -> Vec<f32>>(label: &str, reps: usize, mut f: F) -> f64 {
    
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let out = f();
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(&out);
        if dt < best {
            best = dt;
        }
    }
    println!("  {label:<34} {:>8.1} ms", best * 1e3);
    best
}

fn run_point(name: &str, cfg: &TrellisConfig, out_f: usize, in_f: usize) {
    let total = out_f * in_f;
    println!(
        "\n== {name}: k={} L={} ({} states), {out_f}x{in_f} = {:.1}M weights ==",
        cfg.k_bits,
        cfg.l_bits,
        cfg.num_states(),
        total as f64 / 1e6
    );
    let enc = synth_encoded(total, cfg.k_bits, 256);
    let x: Vec<f32> = (0..in_f).map(|i| (i as f32 * 0.07).cos()).collect();
    let max_b = 64usize;
    let mut xs = Vec::with_capacity(max_b * in_f);
    for b in 0..max_b {
        xs.extend((0..in_f).map(|i| ((i as f32 + b as f32 * 11.3) * 0.053).sin()));
    }

    let y_base = baseline_matvec(&enc, cfg, out_f, in_f, &x);
    let y_fused = fused_matvec(&enc, cfg, None, out_f, in_f, &x);
    assert_eq!(y_base.len(), y_fused.len());
    for o in 0..out_f {
        assert_eq!(
            y_fused[o].to_bits(),
            y_base[o].to_bits(),
            "fused_matvec y[{o}] != baseline (bit-equal contract)"
        );
    }
    let (_, q12) = fused_gemm_with_q12(&enc, cfg, None, out_f, in_f, &x, 1);
    let q_ref = decode_q12_par(&enc, cfg);
    assert_eq!(q12, q_ref, "hidden Q12 != decode_q12_par (byte-identity contract)");
    if cfg!(feature = "neon-fma") {
        println!(
            "  !! neon-fma build: fused_gemm B>=4 floats use vfmaq (single rounding) — \
             strict float-bit identity vs the scalar reference is WAIVED BY DESIGN \
             (documented caveat); hidden-Q12 integer identity still enforced."
        );
    } else {
        let yb4 = fused_gemm(&enc, cfg, None, out_f, in_f, &xs[..4 * in_f], 4);
        for b in 0..4 {
            let y1 = fused_matvec(&enc, cfg, None, out_f, in_f, &xs[b * in_f..(b + 1) * in_f]);
            for o in 0..out_f {
                assert_eq!(
                    yb4[o * 4 + b].to_bits(),
                    y1[o].to_bits(),
                    "fused_gemm B=4 col {b} row {o} != fused_matvec"
                );
            }
        }
        
        for &b in &[4usize, 16, 64] {
            let xb = &xs[..b * in_f];
            let y_neon = fused_gemm(&enc, cfg, None, out_f, in_f, xb, b);
            let y_scal = fused_gemm_scalar_mac(&enc, cfg, None, out_f, in_f, xb, b);
            for (i, (a, s)) in y_neon.iter().zip(y_scal.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    s.to_bits(),
                    "G2b NEON MAC != scalar MAC at flat index {i} (B={b})"
                );
            }
        }
        println!(
            "  determinism: y bit-equal, hidden-Q12 byte-identical, B=4 columns bit-equal, \
             G2b NEON==scalar MAC at B∈{{4,16,64}} ✓"
        );
    }

    let gw = |t: f64| total as f64 / t / 1e9;
    let t_dec = {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t = Instant::now();
            let w = decode_q12_par(&enc, cfg);
            let dt = t.elapsed().as_secs_f64();
            std::hint::black_box(&w);
            best = best.min(dt);
        }
        println!("  {:<34} {:>8.1} ms  ({:.2} Gw/s decode)", "decode_q12_par (decode only)", best * 1e3, gw(best));
        best
    };
    let t_base = bench("baseline decode_q12_par + matmul", 3, || {
        baseline_matvec(&enc, cfg, out_f, in_f, &x)
    });
    let t_fused = bench("fused_matvec (B=1)", 3, || fused_matvec(&enc, cfg, None, out_f, in_f, &x));
    let mut batch_lines: Vec<(usize, f64, f64, f64)> = Vec::new();
    for &b in &[4usize, 16, 64] {
        let xb = &xs[..b * in_f];
        let t_bb = bench(&format!("baseline decode + matmul (B={b})"), 3, || {
            baseline_gemm(&enc, cfg, out_f, in_f, xb, b)
        });
        let t_sc = bench(&format!("fused_gemm scalar MAC (B={b})"), 3, || {
            fused_gemm_scalar_mac(&enc, cfg, None, out_f, in_f, xb, b)
        });
        let t_fb = bench(&format!("fused_gemm NEON MAC  (B={b})"), 3, || {
            fused_gemm(&enc, cfg, None, out_f, in_f, xb, b)
        });
        batch_lines.push((b, t_bb, t_sc, t_fb));
    }

    let mb_saved = (total * 4) as f64 / 1e6;
    println!("\n  -- {name} verdict --");
    println!(
        "  Q12 traffic killed: {mb_saved:.0} MB written + {mb_saved:.0} MB re-read per matvec never happens in fused"
    );
    println!(
        "  B=1: fused {:.1} ms vs baseline {:.1} ms = {:.2}x  ({:.2} Gw/s effective; decode-only ceiling {:.2} Gw/s)",
        t_fused * 1e3,
        t_base * 1e3,
        t_base / t_fused,
        gw(t_fused),
        gw(t_dec)
    );
    for (b, t_bb, t_sc, t_fb) in &batch_lines {
        println!(
            "  B={b:<2}: NEON {:.1} ms vs scalar-MAC {:.1} ms ({:.2}x G2b) vs baseline {:.1} ms ({:.2}x) | per-column {:.2} ms | MAC rate {:.2} GMAC/s",
            t_fb * 1e3,
            t_sc * 1e3,
            t_sc / t_fb,
            t_bb * 1e3,
            t_bb / t_fb,
            t_fb * 1e3 / *b as f64,
            (total * b) as f64 / t_fb / 1e9
        );
    }
}

fn strand_delta_alive() -> bool {
    std::process::Command::new("pgrep")
        .args(["-f", "strand-delta"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn main() {
    let no_wait = std::env::args().any(|a| a == "--no-wait");
    if !no_wait {
        while strand_delta_alive() {
            println!("strand-delta measurement job is running — waiting 60 s (pass --no-wait to override)…");
            std::thread::sleep(std::time::Duration::from_secs(60));
        }
    }

    println!("gate-fused — G2/G2b: fused decode+GEMV (NEON batch MAC) vs scalar MAC vs materializing baseline");
    println!("{}", machine_stamp());

    let (out_f, in_f) = (18944usize, 3584usize);
    run_point("3-bit deploy", &TrellisConfig::for_bpw(3.0), out_f, in_f);
    run_point("2-bit reopen", &TrellisConfig::for_bpw_l(2.0, 12), out_f, in_f);
}
