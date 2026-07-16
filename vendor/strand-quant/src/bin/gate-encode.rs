use std::time::Instant;

use strand_quant::codebook::codebook_lut;
use strand_quant::decode::decode_tensor;
use strand_quant::encode::{encode_tensor_with_lut, encode_tensor_with_lut_metric, encode_tensor_with_lut_reference, vector_lut_from_scalar, EncodeOpts};
use strand_quant::gate_utils::{normal_vec, outlier_shaped, rel_rms};
use strand_quant::TrellisConfig;

const OPT_COMBOS: [(&str, EncodeOpts); 4] = [
    ("default", EncodeOpts { adaptive: true, tail_biting: false, affine_min: false, silence_bonus: 0.0, entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false }),
    ("tail", EncodeOpts { adaptive: true, tail_biting: true, affine_min: false, silence_bonus: 0.0, entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false }),
    ("affine", EncodeOpts { adaptive: true, tail_biting: false, affine_min: true, silence_bonus: 0.0, entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false }),
    ("tail+affine", EncodeOpts { adaptive: true, tail_biting: true, affine_min: true, silence_bonus: 0.0, entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false }),
];

fn run_identity() -> bool {
    println!("== BYTE-IDENTITY A/B: live relax kernel vs frozen scatter reference ==");
    let mut cases = 0usize;
    let mut fails = 0usize;
    let mut check = |label: String, w: &[f32], cfg: &TrellisConfig, opts: &EncodeOpts, lut: &[i32]| {
        let new = encode_tensor_with_lut(w, cfg, opts, lut);
        let reference = encode_tensor_with_lut_reference(w, cfg, opts, lut);
        cases += 1;
        if new != reference {
            fails += 1;
            println!("  FAIL {label}");
        }
    };

    let mut seed = 0xC0FF_EE00u64;
    for k in [2u32, 3, 4] {
        for l in [6u32, 7, 8, 12] {
            if l < k {
                continue;
            }
            let cfg = TrellisConfig::new(l, k, 256);
            let lut = codebook_lut(cfg.l_bits);
            let sizes: &[usize] = if l == 12 { &[2048, 257] } else { &[2048, 1000, 257, 17] };
            for &n in sizes {
                for (oname, opts) in &OPT_COMBOS {
                    seed += 1;
                    let w = normal_vec(n, seed);
                    check(format!("scalar k={k} L={l} n={n} opts={oname}"), &w, &cfg, opts, lut);
                }
            }
        }
    }

    {
        let cfg = TrellisConfig::new(12, 2, 256);
        let lut = codebook_lut(cfg.l_bits);
        let w = outlier_shaped(4096, 0xBADC_AB1E);
        for (oname, opts) in &OPT_COMBOS[..2] {
            check(format!("scalar k=2 L=12 outlier-shaped n=4096 opts={oname}"), &w, &cfg, opts, lut);
        }
    }

    for l in [6u32, 8] {
        let cfg = TrellisConfig::new(l, 2, 256).with_vec_dim(2);
        let lut = vector_lut_from_scalar(codebook_lut(cfg.l_bits), cfg.vec_dim());
        for &n in &[2048usize, 1000] {
            for (oname, opts) in &OPT_COMBOS[..2] {
                seed += 1;
                let w = normal_vec(n, seed);
                check(format!("vec d=2 k=2 L={l} n={n} opts={oname}"), &w, &cfg, opts, &lut);
            }
        }
    }

    println!("  {cases} cases, {fails} mismatches");
    if fails == 0 {
        println!("  PASS — stages 1+3 are byte-identical to the reference\n");
    } else {
        println!("  *** GATE FAILED — the relax kernel is NOT byte-identical ***\n");
    }

    println!("== STAGE-2 f32-METRIC PROBE (bit-CHANGING, off by default) ==");
    for (k, l) in [(2u32, 12u32), (3, 12), (4, 8)] {
        let cfg = TrellisConfig::new(l, k, 256);
        let lut = codebook_lut(cfg.l_bits);
        let w = normal_vec(65536, ((0x5EED_F320 + k as u64) << 8) | l as u64);
        let opts = EncodeOpts::default();
        let e64 = encode_tensor_with_lut_metric(&w, &cfg, &opts, lut, false);
        let e32 = encode_tensor_with_lut_metric(&w, &cfg, &opts, lut, true);
        let diff_bytes = e64.bits.iter().zip(e32.bits.iter()).filter(|(a, b)| a != b).count();
        let r64 = rel_rms(&w, &decode_tensor(&e64, &cfg));
        let r32 = rel_rms(&w, &decode_tensor(&e32, &cfg));
        println!(
            "  k={k} L={l}: payload bytes differing {}/{} ({:.2}%)  rel-RMS f64={:.6}  f32={:.6}  (Δ {:+.4}%)",
            diff_bytes,
            e64.bits.len(),
            100.0 * diff_bytes as f64 / e64.bits.len().max(1) as f64,
            r64,
            r32,
            100.0 * (r32 - r64) / r64,
        );
    }
    println!();
    fails == 0
}

fn stamp() {
    println!("== MACHINE STATE ==");
    let cmds: [(&str, &[&str]); 5] = [
        ("uname", &["uname", "-mrs"]),
        ("cpu", &["sysctl", "-n", "machdep.cpu.brand_string"]),
        ("ncpu", &["sysctl", "-n", "hw.ncpu"]),
        ("uptime/load", &["uptime"]),
        ("git", &["git", "rev-parse", "--short", "HEAD"]),
    ];
    for (label, cmd) in cmds {
        let out = std::process::Command::new(cmd[0]).args(&cmd[1..]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_else(|_| "<unavailable>".into());
        println!("  {label}: {out}");
    }
    println!("  build: release={}", !cfg!(debug_assertions));
    println!();
}

fn bench_one(label: &str, w: &[f32], cfg: &TrellisConfig, lut: &'static [i32], iters: u32, f: impl Fn(&[f32], &TrellisConfig, &[i32]) -> usize) -> f64 {
    let _ = f(&w[..1024.min(w.len())], cfg, lut);
    let t0 = Instant::now();
    let mut sink = 0usize;
    for _ in 0..iters {
        sink += f(w, cfg, lut);
    }
    let dt = t0.elapsed().as_secs_f64();
    let mw_s = (w.len() as f64 * iters as f64) / dt / 1e6;
    println!("    {label:<22} {mw_s:8.3} Mw/s   ({:.2}s, sink={sink})", dt);
    mw_s
}

fn run_bench() {
    stamp();
    println!("== ENCODE THROUGHPUT (CPU explicit-LUT path; GPU dispatch bypassed by construction) ==");
    println!("   7B-shaped slices: a 7B gate/up tensor is 18944x3584 = 67.9M w; the per-tensor");
    println!("   Viterbi work is block-local (256 w), so a fixed-N slice measures the same kernel.");
    let opts = EncodeOpts::default();
    let nthreads: usize = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);

    let configs: [(&str, u32, u32, usize, u32); 3] = [("q2_l12 (shipped 2-bit)", 2, 12, 49_152, 2), ("q3_l12", 3, 12, 24_576, 2), ("q4_l8 (4-bit default)", 4, 8, 262_144, 2)];

    for (name, k, l, n, iters) in configs {
        let cfg = TrellisConfig::new(l, k, 256);
        let lut: &'static [i32] = codebook_lut(cfg.l_bits);
        println!("  {name}  (k={k} L={l} N={n}, single-thread):");
        let w = normal_vec(n, 0xBEEF_0000 + l as u64);
        let r_ref = bench_one("reference (scatter)", &w, &cfg, lut, iters, |w, c, lu| encode_tensor_with_lut_reference(w, c, &opts, lu).bits.len());
        let r_new = bench_one("new relax kernel", &w, &cfg, lut, iters, |w, c, lu| encode_tensor_with_lut(w, c, &opts, lu).bits.len());
        let r_f32 = bench_one("new + f32 metric", &w, &cfg, lut, iters, |w, c, lu| encode_tensor_with_lut_metric(w, c, &opts, lu, true).bits.len());
        println!("    speedup: new {:.2}x   new+f32 {:.2}x\n", r_new / r_ref, r_f32 / r_ref);

        println!("  {name}  ({nthreads} threads, one tensor each):");
        for (tlabel, mode) in [("reference (scatter)", 0u8), ("new relax kernel", 1), ("new + f32 metric", 2)] {
            let t0 = Instant::now();
            let total: usize = std::thread::scope(|s| {
                let mut handles = Vec::new();
                for t in 0..nthreads {
                    handles.push(s.spawn(move || {
                        let w = normal_vec(n, 0xFEED_0000 + (l as u64) * 100 + t as u64);
                        let lu: &'static [i32] = codebook_lut(cfg.l_bits);
                        let o = EncodeOpts::default();
                        let e = match mode {
                            0 => encode_tensor_with_lut_reference(&w, &cfg, &o, lu),
                            1 => encode_tensor_with_lut(&w, &cfg, &o, lu),
                            _ => encode_tensor_with_lut_metric(&w, &cfg, &o, lu, true),
                        };
                        w.len() + e.bits.len() % 2
                    }));
                }
                handles.into_iter().map(|h| h.join().unwrap()).sum()
            });
            let dt = t0.elapsed().as_secs_f64();

            let mw_s = ((n * nthreads) as f64) / dt / 1e6;
            println!("    {tlabel:<22} {mw_s:8.3} Mw/s aggregate  ({dt:.2}s, sink={total})");
        }
        println!();
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    let mut ok = true;
    if mode == "identity" || mode == "all" {
        ok = run_identity();
    }
    if mode == "bench" || mode == "all" {
        run_bench();
    }
    if !ok {
        std::process::exit(1);
    }
}
