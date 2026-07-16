#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gate-tropical: Metal-only gate (macOS / Apple Silicon). Nothing to do here.");
}

#[cfg(target_os = "macos")]
fn main() {
    mac::main()
}

#[cfg(target_os = "macos")]
mod mac {
    use std::time::Instant;

    use strand_quant::codebook::codebook_lut;
    use strand_quant::encode::{encode_tensor_with_lut_metric, encode_tensor_with_lut_metric_search, EncodeOpts};
    use strand_quant::gate_utils::{normal_vec, outlier_shaped};
    use strand_quant::metal_encode::TropicalEncoder;
    use strand_quant::TrellisConfig;

    const OPT_COMBOS: [(&str, EncodeOpts); 4] = [
        ("default", EncodeOpts { adaptive: true, tail_biting: false, affine_min: false, silence_bonus: 0.0, entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false }),
        ("tail", EncodeOpts { adaptive: true, tail_biting: true, affine_min: false, silence_bonus: 0.0, entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false }),
        ("affine", EncodeOpts { adaptive: true, tail_biting: false, affine_min: true, silence_bonus: 0.0, entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false }),
        ("tail+affine", EncodeOpts { adaptive: true, tail_biting: true, affine_min: true, silence_bonus: 0.0, entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false }),
    ];

    fn for_each_case(mut check: impl FnMut(String, &[f32], &TrellisConfig, &EncodeOpts)) {
        let mut seed = 0x7209_1CA1u64;
        for k in [1u32, 2, 3, 4] {
            for l in [4u32, 5, 6, 7] {
                if l < k {
                    continue;
                }
                let cfg = TrellisConfig::new(l, k, 256);
                for &n in &[2048usize, 1000, 257, 17] {
                    for (oname, opts) in &OPT_COMBOS {
                        seed += 1;
                        let w = normal_vec(n, seed);
                        check(format!("k={k} L={l} n={n} opts={oname}"), &w, &cfg, opts);
                    }
                }
            }
        }

        for k in [2u32, 3] {
            for l in [8u32, 9, 10, 11, 12] {
                let cfg = TrellisConfig::new(l, k, 256);
                for &n in &[2048usize, 257] {
                    for (oname, opts) in &OPT_COMBOS[..2] {
                        seed += 1;
                        let w = normal_vec(n, seed);
                        check(format!("k={k} L={l} n={n} opts={oname}"), &w, &cfg, opts);
                    }
                }
            }
        }

        for (k, l) in [(2u32, 7u32), (3, 7), (2, 10), (2, 12)] {
            let cfg = TrellisConfig::new(l, k, 256);
            let w = outlier_shaped(4096, 0xBADC_AB1E + l as u64);
            for (oname, opts) in &OPT_COMBOS[..2] {
                check(format!("k={k} L={l} outlier-shaped n=4096 opts={oname}"), &w, &cfg, opts);
            }
        }

        for (k, l) in [(2u32, 7u32), (3, 7), (1, 6), (2, 10), (2, 12)] {
            let cfg = TrellisConfig::new(l, k, 256);
            let vals = [0.0f32, 0.5, -0.5, 0.25];
            let w: Vec<f32> = (0..2048).map(|i| vals[i % vals.len()]).collect();
            for (oname, opts) in &OPT_COMBOS[..2] {
                check(format!("k={k} L={l} tie-cyclic n=2048 opts={oname}"), &w, &cfg, opts);
            }
            let base = normal_vec(2048, 0x71E5_0000 + (k as u64) << 16 | l as u64);
            let enc = encode_tensor_with_lut_metric(&base, &cfg, &OPT_COMBOS[0].1, codebook_lut(cfg.l_bits), true);
            let snapped = strand_quant::decode::decode_tensor(&enc, &cfg);
            for (oname, opts) in &OPT_COMBOS[..2] {
                check(format!("k={k} L={l} tie-snapped n=2048 opts={oname}"), &snapped, &cfg, opts);
            }
        }

        {
            let cfg = TrellisConfig::new(7, 3, 256);
            let zeros = vec![0.0f32; 300];
            for (oname, opts) in &OPT_COMBOS {
                check(format!("k=3 L=7 all-zeros n=300 opts={oname}"), &zeros, &cfg, opts);
            }
            let consts = vec![0.37f32; 512];
            check("k=3 L=7 constant n=512 opts=tail".into(), &consts, &cfg, &OPT_COMBOS[1].1);
            let one = [1.25f32];
            check("k=3 L=7 n=1 opts=default".into(), &one, &cfg, &OPT_COMBOS[0].1);
        }
    }

    fn run_identity(gpu: &TropicalEncoder) -> bool {
        println!("== BYTE-IDENTITY A/B: both GPU encode lanes vs their CPU references ==");
        println!("   full-gpu lane  vs CPU (f32 metric + f32 search)");
        println!("   prep-cpu lane  vs CPU (f32 metric, canonical f64 search)");
        println!("   (full EncodedTensor equality: path bits + init_states + side info;");
        println!("    ties decide bits, so equality IS the tie-break proof)");
        let mut cases = 0usize;
        let mut fails = 0usize;
        let mut skipped = 0usize;
        for_each_case(|label, w, cfg, opts| {
            let lanes: [(&str, Option<_>, _); 2] = [
                ("full-gpu", gpu.encode_tensor(w, cfg, opts), encode_tensor_with_lut_metric_search(w, cfg, opts, codebook_lut(cfg.l_bits), true, true)),
                ("prep-cpu", gpu.encode_tensor_prep_cpu(w, cfg, opts), encode_tensor_with_lut_metric(w, cfg, opts, codebook_lut(cfg.l_bits), true)),
            ];
            for (lane, gpu_enc, cpu_enc) in lanes {
                let Some(gpu_enc) = gpu_enc else {
                    skipped += 1;
                    println!("  SKIP {label} [{lane}] (outside GPU envelope)");
                    continue;
                };
                cases += 1;
                if gpu_enc != cpu_enc {
                    fails += 1;

                    let bit_diff = gpu_enc.bits.iter().zip(cpu_enc.bits.iter()).position(|(a, b)| a != b);
                    let blk_diff = gpu_enc.blocks.iter().zip(cpu_enc.blocks.iter()).position(|(a, b)| a != b);
                    println!("  FAIL {label} [{lane}]  first-bit-byte-diff={bit_diff:?} first-block-diff={blk_diff:?}");
                }
            }
        });

        println!("  {cases} cases compared, {fails} mismatches, {skipped} skipped");
        if fails == 0 {
            println!("  PASS — both GPU encode lanes are byte-identical to their CPU references\n");
        } else {
            println!("  *** GATE FAILED — a GPU lane is NOT byte-identical; perf is OFF ***\n");
        }
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

    fn cpu_threads() -> usize {
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
    }

    fn cpu_mt_mws(n_per_thread: usize, cfg: &TrellisConfig, f32_metric: bool, seed: u64) -> f64 {
        let nt = cpu_threads();
        let t0 = Instant::now();
        let sink: usize = std::thread::scope(|s| {
            let mut hs = Vec::new();
            for t in 0..nt {
                let cfg = *cfg;
                hs.push(s.spawn(move || {
                    let w = normal_vec(n_per_thread, seed + t as u64);
                    let e = encode_tensor_with_lut_metric(&w, &cfg, &EncodeOpts::default(), codebook_lut(cfg.l_bits), f32_metric);
                    e.bits.len()
                }));
            }
            hs.into_iter().map(|h| h.join().unwrap()).sum()
        });
        let dt = t0.elapsed().as_secs_f64();
        let mws = (n_per_thread * nt) as f64 / dt / 1e6;
        println!("    cpu {}T {:<22} {mws:9.3} Mw/s aggregate  ({dt:.2}s, sink={sink})", nt, if f32_metric { "(f32 metric)" } else { "(canon f64)" },);
        mws
    }

    fn run_bench(gpu: &TropicalEncoder) {
        stamp();
        println!("== ENCODE THROUGHPUT: full-GPU lane vs prep-CPU lane vs relax-kernel CPU ==");
        println!("   GPU numbers are END-TO-END (upload + search kernel + Viterbi kernel +");
        println!("   readback + assembly) — the wall-clock a requant would see.");
        println!("   Kill bar (pre-registered): full-gpu < 2x the all-cores canon-f64 CPU");
        println!("   => death arithmetic.\n");

        let configs: [(&str, u32, u32, usize, usize, u32); 4] = [
            ("k=3 L=7  (3-bit product geometry)", 3, 7, 8 << 20, 1 << 19, 2),
            ("k=2 L=7", 2, 7, 8 << 20, 1 << 19, 2),
            ("k=2 L=10 (envelope edge)", 2, 10, 2 << 20, 1 << 17, 2),
            ("k=2 L=12 (2-bit op point, stretch)", 2, 12, 2 << 20, 1 << 16, 2),
        ];

        for (name, k, l, n_gpu, n_cpu1, iters) in configs {
            let cfg = TrellisConfig::new(l, k, 256);
            let opts = EncodeOpts::default();
            println!("  {name}:");

            let w = normal_vec(n_gpu, 0x90D0_0000 + l as u64);
            let mut gpu_full_mws = 0.0f64;
            for full_lane in [true, false] {
                let label = if full_lane { "full-gpu (f32 search)" } else { "prep-cpu (f64 search)" };
                let run = |ww: &[f32]| -> usize {
                    let e = if full_lane { gpu.encode_tensor(ww, &cfg, &opts) } else { gpu.encode_tensor_prep_cpu(ww, &cfg, &opts) };
                    e.map(|e| e.bits.len()).unwrap_or(0)
                };
                let _ = run(&w[..(1 << 16)]);
                let t0 = Instant::now();
                let mut sink = 0usize;
                for _ in 0..iters {
                    sink += run(&w);
                }
                let dt = t0.elapsed().as_secs_f64();
                let mws = (n_gpu as f64 * iters as f64) / dt / 1e6;
                if full_lane {
                    gpu_full_mws = mws;
                }
                println!("    gpu {label:<25} {mws:9.3} Mw/s            ({dt:.2}s, N={n_gpu}, sink={sink})");
            }

            let w1 = normal_vec(n_cpu1, 0xC0DE_0000 + l as u64);
            for (label, m) in [("(f32 metric)", true), ("(canon f64)", false)] {
                let t0 = Instant::now();
                let mut sink = 0usize;
                for _ in 0..iters {
                    sink += encode_tensor_with_lut_metric(&w1, &cfg, &opts, codebook_lut(cfg.l_bits), m).bits.len();
                }
                let dt = t0.elapsed().as_secs_f64();
                let mws = (n_cpu1 as f64 * iters as f64) / dt / 1e6;
                println!("    cpu  1T {label:<22} {mws:9.3} Mw/s            ({dt:.2}s, sink={sink})");
            }

            let mt_f32 = cpu_mt_mws(n_cpu1, &cfg, true, 0xFA57_0000 + l as u64);
            let mt_f64 = cpu_mt_mws(n_cpu1, &cfg, false, 0xFA58_0000 + l as u64);
            let vs_f32 = gpu_full_mws / mt_f32;
            let vs_f64 = gpu_full_mws / mt_f64;
            let verdict = if vs_f64 >= 2.0 { "PASS (>= 2x bar)" } else { "BELOW the 2x kill bar" };
            println!("    -> full-gpu = {vs_f32:.2}x all-cores-f32, {vs_f64:.2}x all-cores-canon-f64   [{verdict}]\n");
        }
    }

    fn run_diag(gpu: &TropicalEncoder) {
        println!("== DIAG: search vs Viterbi split (adaptive sub-scales ON vs OFF) ==");
        let no_adapt = EncodeOpts { adaptive: false, tail_biting: false, affine_min: false, silence_bonus: 0.0, entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false };
        let def = EncodeOpts::default();
        for (k, l, n_gpu, n_cpu) in [(3u32, 7u32, 8usize << 20, 1usize << 19)] {
            let cfg = TrellisConfig::new(l, k, 256);
            let w = normal_vec(n_gpu, 0xD1A6_0000 + l as u64);
            let w1 = normal_vec(n_cpu, 0xD1A6_1111 + l as u64);
            for (oname, opts) in [("adaptive", &def), ("no-adapt", &no_adapt)] {
                let t0 = Instant::now();
                let s = gpu.encode_tensor(&w, &cfg, opts).map(|e| e.bits.len()).unwrap_or(0);
                let g = n_gpu as f64 / t0.elapsed().as_secs_f64() / 1e6;
                let t0 = Instant::now();
                let sp = gpu.encode_tensor_prep_cpu(&w, &cfg, opts).map(|e| e.bits.len()).unwrap_or(0);
                let gp = n_gpu as f64 / t0.elapsed().as_secs_f64() / 1e6;
                let t0 = Instant::now();
                let s2 = encode_tensor_with_lut_metric(&w1, &cfg, opts, codebook_lut(l), true).bits.len();
                let c1 = n_cpu as f64 / t0.elapsed().as_secs_f64() / 1e6;
                println!("  k={k} L={l} {oname:<9} full-gpu {g:9.3} Mw/s   prep-cpu {gp:9.3} Mw/s   cpu-1T-f32 {c1:7.3} Mw/s   (sinks {s}/{sp}/{s2})");
            }
        }
        println!();
    }

    pub fn main() {
        let mode = std::env::args().nth(1).unwrap_or_else(|| "all".into());
        let Some(gpu) = TropicalEncoder::new() else {
            eprintln!("gate-tropical: no Metal device / pipeline — cannot gate. Exit 2.");
            std::process::exit(2);
        };
        let mut ok = true;
        if mode == "identity" || mode == "all" {
            ok = run_identity(&gpu);
        }
        if !ok {
            std::process::exit(1);
        }
        if mode == "bench" || mode == "all" {
            run_bench(&gpu);
        }
        if mode == "diag" {
            run_diag(&gpu);
        }
    }
}
