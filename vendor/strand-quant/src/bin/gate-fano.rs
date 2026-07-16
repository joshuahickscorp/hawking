use std::time::Instant;

use strand_quant::codebook::codebook_lut;
use strand_quant::decode::decode_tensor;
use strand_quant::encode::{encode_tensor_with_lut, EncodeOpts};
use strand_quant::fano::{encode_tensor_fano, encode_tensor_pruned, FanoParams, PruneReport};
use strand_quant::gate_utils::{normal_vec, outlier_shaped, rel_rms, rht_seed_for};
use strand_quant::rht::{rht_forward_rows, RhtConfig};
use strand_quant::safetensor_io::SafeTensors;
use strand_quant::TrellisConfig;

fn load_real_tensors(path: &str, max_tensors: usize) -> Vec<(String, Vec<f32>)> {
    let st = match SafeTensors::open(path) {
        Ok(st) => st,
        Err(e) => {
            eprintln!("  (real tensors unavailable: {path}: {e})");
            return Vec::new();
        }
    };
    let proj = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight"];
    let mut out = Vec::new();
    for name in &st.order {
        if out.len() >= max_tensors {
            break;
        }
        let t = &st.tensors[name];
        if t.shape.len() != 2 || !proj.iter().any(|p| name.ends_with(p)) {
            continue;
        }
        let mut w = st.to_f32(t);

        let n = w.len();
        let k = ((1.0 / 100.0) * n as f64).round() as usize;
        if k > 0 {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by(|&a, &b| w[b].abs().partial_cmp(&w[a].abs()).unwrap_or(std::cmp::Ordering::Equal));
            for &i in &order[..k] {
                w[i] = 0.0;
            }
        }
        let in_features = t.shape[1] as usize;
        let rcfg = RhtConfig::from_seed(rht_seed_for(name));
        let work = rht_forward_rows(&w, &rcfg, in_features);
        out.push((name.clone(), work));
    }
    out
}

const OPT_COMBOS: [(&str, EncodeOpts); 4] = [
    ("default", EncodeOpts { adaptive: true, tail_biting: false, affine_min: false, silence_bonus: 0.0, entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false }),
    ("tail", EncodeOpts { adaptive: true, tail_biting: true, affine_min: false, silence_bonus: 0.0, entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false }),
    ("affine", EncodeOpts { adaptive: true, tail_biting: false, affine_min: true, silence_bonus: 0.0, entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false }),
    ("tail+affine", EncodeOpts { adaptive: true, tail_biting: true, affine_min: true, silence_bonus: 0.0, entropy_bonus_scale: 0.0, entropy_bonus_two_pass: false }),
];

fn run_identity(model: &str, max_tensors: usize, max_blocks: usize) -> bool {
    println!("== BYTE-IDENTITY A/B: branch-and-bound pruned Viterbi vs live relax kernel ==");
    let mut cases = 0usize;
    let mut fails = 0usize;
    let mut exp_sum = 0.0f64;
    let mut exp_n = 0usize;
    let mut check = |label: String, w: &[f32], cfg: &TrellisConfig, opts: &EncodeOpts| {
        let lut = codebook_lut(cfg.l_bits);
        let (pruned, rep) = encode_tensor_pruned(w, cfg, opts, lut);
        let live = encode_tensor_with_lut(w, cfg, opts, lut);
        cases += 1;
        if pruned != live {
            fails += 1;
            println!("  FAIL {label}");
        }
        exp_sum += rep.expansion_ratio();
        exp_n += 1;
    };

    let mut seed = 0xFA90_0001u64;
    for k in [2u32, 3, 4] {
        for l in [6u32, 7, 8, 12] {
            if l < k {
                continue;
            }
            let cfg = TrellisConfig::new(l, k, 256);
            let sizes: &[usize] = if l == 12 { &[2048, 257] } else { &[2048, 1000, 257, 17] };
            for &n in sizes {
                for (oname, opts) in &OPT_COMBOS {
                    seed += 1;
                    let w = normal_vec(n, seed);
                    check(format!("scalar k={k} L={l} n={n} opts={oname}"), &w, &cfg, opts);
                }
            }
        }
    }

    {
        let cfg = TrellisConfig::new(12, 2, 256);
        let w = outlier_shaped(4096, 0xBADC_AB1E);
        for (oname, opts) in &OPT_COMBOS {
            check(format!("k=2 L=12 outlier-shaped n=4096 opts={oname}"), &w, &cfg, opts);
        }
    }

    for (k, l) in [(2u32, 12u32), (3, 12), (2, 8), (4, 8)] {
        let cfg = TrellisConfig::new(l, k, 256);
        let vals = [0.0f32, 0.5, -0.5, 0.25];
        let w: Vec<f32> = (0..2048).map(|i| vals[i % vals.len()]).collect();
        for (oname, opts) in &OPT_COMBOS {
            check(format!("k={k} L={l} tie-cyclic n=2048 opts={oname}"), &w, &cfg, opts);
        }
        let lut = codebook_lut(cfg.l_bits);
        let base = normal_vec(2048, 0x71E5_0000 + (k as u64) << 16 | l as u64);
        let enc = encode_tensor_with_lut(&base, &cfg, &OPT_COMBOS[0].1, lut);
        let snapped = decode_tensor(&enc, &cfg);
        for (oname, opts) in &OPT_COMBOS {
            check(format!("k={k} L={l} tie-snapped n=2048 opts={oname}"), &snapped, &cfg, opts);
        }
    }

    {
        let cfg = TrellisConfig::new(12, 2, 256);
        let w = vec![0.0f32; 512];
        check("k=2 L=12 all-zero n=512 opts=default".into(), &w, &cfg, &OPT_COMBOS[0].1);
        let w = vec![0.25f32; 300];
        check("k=2 L=12 constant n=300 opts=default".into(), &w, &cfg, &OPT_COMBOS[0].1);
    }

    let real = load_real_tensors(model, max_tensors);
    if real.is_empty() {
        println!("  (no real tensors — synthetic matrix only)");
    }
    for (name, w) in &real {
        let n_use = (max_blocks * 256).min(w.len());
        for (k, l) in [(2u32, 12u32), (3, 12)] {
            let cfg = TrellisConfig::new(l, k, 256);
            check(format!("REAL {name} [..{n_use}] k={k} L={l} default"), &w[..n_use], &cfg, &OPT_COMBOS[0].1);
        }
    }

    println!("  {cases} cases, {fails} mismatches; mean expansion ratio {:.4}", exp_sum / exp_n.max(1) as f64);
    if fails == 0 {
        println!("  PASS — pruned encoder is byte-identical to the live kernel\n");
    } else {
        println!("  *** GATE FAILED — pruning is NOT byte-identical ***\n");
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

fn bench_pair(label: &str, w: &[f32], cfg: &TrellisConfig, iters: u32) {
    let lut = codebook_lut(cfg.l_bits);
    let opts = EncodeOpts::default();

    let _ = encode_tensor_with_lut(&w[..1024.min(w.len())], cfg, &opts, lut);

    let t0 = Instant::now();
    let mut sink = 0usize;
    for _ in 0..iters {
        sink += encode_tensor_with_lut(w, cfg, &opts, lut).bits.len();
    }
    let dt_live = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let mut rep_last: Option<PruneReport> = None;
    for _ in 0..iters {
        let (e, rep) = encode_tensor_pruned(w, cfg, &opts, lut);
        sink += e.bits.len();
        rep_last = Some(rep);
    }
    let dt_pruned = t0.elapsed().as_secs_f64();

    let rep = rep_last.unwrap();
    let mw_live = (w.len() as f64 * iters as f64) / dt_live / 1e6;
    let mw_pruned = (w.len() as f64 * iters as f64) / dt_pruned / 1e6;
    println!("  {label:<44} live {mw_live:7.3} Mw/s | pruned {mw_pruned:7.3} Mw/s | speedup {:.2}x | nodes expanded {:.1}% (sink={sink})", mw_pruned / mw_live, 100.0 * rep.expansion_ratio(),);
}

fn run_bench(model: &str, max_tensors: usize, max_blocks: usize) {
    stamp();
    println!("== ENCODE WALL TIME: pruned (byte-identical) vs live relax kernel, single-thread ==");
    for (name, k, l, n, iters) in [("synthetic q2_l12", 2u32, 12u32, 49_152usize, 2u32), ("synthetic q3_l12", 3, 12, 24_576, 2), ("synthetic q4_l8", 4, 8, 262_144, 2)] {
        let cfg = TrellisConfig::new(l, k, 256);
        let w = normal_vec(n, 0xFA90_BEEF + l as u64);
        bench_pair(name, &w, &cfg, iters);
    }
    println!();
    println!("== REAL pv2 shadow tensors (production prep: 1% outlier zero + row RHT) ==");
    let real = load_real_tensors(model, max_tensors);
    for (name, w) in &real {
        let n_use = (max_blocks * 256).min(w.len());
        for (k, l) in [(2u32, 12u32), (3, 12)] {
            let cfg = TrellisConfig::new(l, k, 256);
            bench_pair(&format!("{name} [..{n_use}] k={k} L={l}"), &w[..n_use], &cfg, 1);
        }
    }
    println!();
}

fn percentiles(sorted: &[f64]) -> [f64; 7] {
    let q = |p: f64| -> f64 {
        if sorted.is_empty() {
            return f64::NAN;
        }
        let i = ((sorted.len() - 1) as f64 * p).round() as usize;
        sorted[i]
    };
    [q(0.0), q(0.10), q(0.25), q(0.50), q(0.75), q(0.90), q(1.0)]
}

fn print_pcts(label: &str, vals: &mut Vec<f64>) {
    vals.sort_by(f64::total_cmp);
    let p = percentiles(vals);
    println!("    {label:<28} min {:10.3e}  p10 {:10.3e}  p25 {:10.3e}  p50 {:10.3e}  p75 {:10.3e}  p90 {:10.3e}  max {:10.3e}", p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
}

fn run_hist(model: &str, max_tensors: usize, max_blocks: usize) {
    println!("== THE DIFFICULTY HISTOGRAM (per-block winning-path cost, real tensors) ==");
    println!("   block = 256 weights; cost = winning Viterbi SSE; floor = sum of per-step");
    println!("   nearest-level minima (the codebook floor); greedy = the memoryless bound.");
    let real = load_real_tensors(model, max_tensors);
    if real.is_empty() {
        println!("  no real tensors found at {model} — nothing to measure");
        return;
    }
    for (k, l) in [(2u32, 12u32), (3, 12)] {
        println!("  config k={k} L={l}:");
        let cfg = TrellisConfig::new(l, k, 256);
        let lut = codebook_lut(cfg.l_bits);
        let opts = EncodeOpts::default();
        let mut win_per_w: Vec<f64> = Vec::new();
        let mut win_rel: Vec<f64> = Vec::new();
        let mut greedy_gap: Vec<f64> = Vec::new();
        let mut floor_gap: Vec<f64> = Vec::new();
        let mut expansion: Vec<f64> = Vec::new();
        for (_name, w) in &real {
            let n_use = (max_blocks * 256).min(w.len());
            let (_e, rep) = encode_tensor_pruned(&w[..n_use], &cfg, &opts, lut);
            let mut off = 0usize;
            for b in &rep.blocks {
                if b.n == 0 {
                    continue;
                }
                let energy: f64 = w[off..off + b.n as usize].iter().map(|&x| (x as f64) * (x as f64)).sum();
                off += b.n as usize;
                win_per_w.push(b.win_cost / b.n as f64);
                if energy > 0.0 {
                    win_rel.push(b.win_cost / energy);
                }
                if b.win_cost > 0.0 {
                    greedy_gap.push(b.greedy_ub / b.win_cost);
                    floor_gap.push(b.win_cost / b.floor.max(1e-30));
                }
                expansion.push(b.expanded as f64 / b.total.max(1) as f64);
            }
        }
        println!("    blocks measured: {}", win_per_w.len());
        print_pcts("win cost / weight", &mut win_per_w);
        print_pcts("win cost / block energy", &mut win_rel);
        print_pcts("greedy/optimal ratio", &mut greedy_gap);
        print_pcts("optimal/floor ratio", &mut floor_gap);
        print_pcts("node-expansion ratio", &mut expansion);
        println!();
    }
}

fn run_fano(model: &str, max_tensors: usize, max_blocks: usize) {
    println!("== ⚠️ FANO/STACK MODE (BIT-CHANGING, OFF BY DEFAULT) — rel-RMS + time screening ==");
    println!("   Decision metric is the 0.5B PPL A/B (research/fano-results.md), NOT rel-RMS.");
    let real = load_real_tensors(model, max_tensors);
    let mut inputs: Vec<(String, Vec<f32>)> = Vec::new();
    if real.is_empty() {
        println!("  (no real tensors — synthetic fallback)");
        inputs.push(("synthetic-gauss-64k".into(), normal_vec(65_536, 0xFA90_0FA0)));
    } else {
        for (name, w) in real {
            let n_use = (max_blocks * 256).min(w.len());
            inputs.push((name, w[..n_use].to_vec()));
        }
    }
    let opts = EncodeOpts::default();
    for (k, l) in [(2u32, 12u32), (3, 12)] {
        let cfg = TrellisConfig::new(l, k, 256);
        let lut = codebook_lut(cfg.l_bits);
        println!("  config k={k} L={l}:");
        for (name, w) in &inputs {
            let t0 = Instant::now();
            let exact = encode_tensor_with_lut(w, &cfg, &opts, lut);
            let dt_v = t0.elapsed().as_secs_f64();
            let r_v = rel_rms(w, &decode_tensor(&exact, &cfg));
            println!("    {name} [n={}]  VITERBI: rel-RMS {:.4}%  {:.2}s", w.len(), 100.0 * r_v, dt_v);
            for (bias, budget) in [(0.5, 4.0), (1.0, 4.0), (1.0, 8.0), (1.5, 8.0), (1.0, 16.0)] {
                let params = FanoParams { bias_scale: bias, budget_mult: budget };
                let t0 = Instant::now();
                let (enc, rep) = encode_tensor_fano(w, &cfg, &opts, lut, &params);
                let dt_f = t0.elapsed().as_secs_f64();
                let r_f = rel_rms(w, &decode_tensor(&enc, &cfg));
                let pops: u64 = rep.blocks.iter().map(|b| b.pops).sum();
                let exhausted = rep.blocks.iter().filter(|b| b.budget_exhausted).count();
                println!(
                    "      fano bias={bias:.1} budget={budget:>4.0}x: rel-RMS {:.4}% (Δ {:+.2}%)  {:.2}s ({:.1}x vs viterbi)  pops/weight {:.2}  exhausted {}/{} blocks",
                    100.0 * r_f,
                    100.0 * (r_f - r_v) / r_v,
                    dt_f,
                    dt_v / dt_f.max(1e-12),
                    pops as f64 / w.len() as f64,
                    exhausted,
                    rep.blocks.len(),
                );
            }
        }
    }
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_else(|| "all".into());
    let mut model = "scratch/qwen-05b/qat-pv2-hf/model.safetensors".to_string();
    let mut max_tensors = 4usize;
    let mut max_blocks = 512usize;
    let mut it = args.iter().skip(2);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model = it.next().expect("--model PATH").clone(),
            "--tensors" => max_tensors = it.next().expect("--tensors N").parse().unwrap(),
            "--blocks" => max_blocks = it.next().expect("--blocks N").parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }

    let mut ok = true;
    match mode.as_str() {
        "identity" => ok = run_identity(&model, max_tensors, max_blocks.min(64)),
        "bench" => run_bench(&model, max_tensors, max_blocks),
        "hist" => run_hist(&model, max_tensors, max_blocks),
        "fano" | "--fano" => run_fano(&model, max_tensors, max_blocks.min(128)),
        "all" => {
            ok = run_identity(&model, max_tensors, max_blocks.min(64));
            run_bench(&model, max_tensors, max_blocks);
        }
        other => panic!("unknown mode {other} (identity|bench|hist|fano|all)"),
    }
    if !ok {
        std::process::exit(1);
    }
}
