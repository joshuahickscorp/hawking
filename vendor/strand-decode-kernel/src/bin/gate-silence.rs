use std::collections::BTreeMap;
use std::process::ExitCode;

use strand_decode_kernel::block_walk::gate_proto::machine_stamp;
use strand_decode_kernel::gemv::decode_tensor_q12;
use strand_decode_kernel::loader::StrandModel;
use strand_decode_kernel::silence::{census_tensor, decode_q12_silence, matvec_silence, matvec_silence_skip, zero_nearest_q12, SilenceMask};
use strand_quant::codebook::codebook_lut;
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor, encode_tensor_with, BlockMeta, EncodeOpts};
use strand_quant::TrellisConfig;

fn planted_weights(n: usize, block_len: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let b = i / block_len;
            let last = (n - 1) / block_len;
            if b == 0 || b == 3 || b == last {
                0.0
            } else {
                ((i as f32 + seed as f32) * 0.0137).sin() * 0.5
            }
        })
        .collect()
}

fn run_gate() -> Result<(), String> {
    let mut checked = 0usize;
    let mut detected_silent = 0usize;

    let configs = [TrellisConfig::for_bpw(3.0), TrellisConfig::for_bpw(2.0), TrellisConfig::for_bpw_l(2.0, 12), TrellisConfig::for_bpw_l(2.0, 5)];
    for cfg in &configs {
        let lut = codebook_lut(cfg.l_bits);
        for &n in &[2048usize, 1000, 257, 31] {
            for seed in 0..4u64 {
                let w = planted_weights(n, cfg.block_len, seed);
                let variants = [
                    encode_tensor(&w, cfg),
                    encode_tensor_with(&w, cfg, &EncodeOpts { tail_biting: true, ..Default::default() }),
                    encode_tensor_with(&w, cfg, &EncodeOpts { affine_min: true, ..Default::default() }),
                    encode_tensor_with(&w, cfg, &EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() }),
                ];
                for enc in &variants {
                    let mask = SilenceMask::build(enc, cfg, lut);
                    detected_silent += mask.n_silent_zero() + mask.n_silent_const();
                    let got = decode_q12_silence(enc, cfg, lut, &mask);
                    let want = decode_tensor_fixed(enc, cfg);
                    if got != want {
                        return Err(format!("DECODE IDENTITY FAIL: L={} k={} n={n} seed={seed} tail={} affine={}", cfg.l_bits, cfg.k_bits, enc.tail_biting, enc.has_affine_min));
                    }
                    checked += 1;
                }
            }
        }
    }

    {
        let cfg = TrellisConfig::for_bpw_l(2.0, 12);
        let lut = codebook_lut(cfg.l_bits);
        let mut enc = strand_decode_kernel::block_walk::gate_proto::synth_encoded(4096, cfg.k_bits, cfg.block_len);
        for b in [1usize, 2, 5] {
            let old: BlockMeta = enc.blocks[b].clone();
            enc.blocks[b] = BlockMeta { scale_q: 0, ..old };
        }
        let mask = SilenceMask::build(&enc, &cfg, lut);
        if mask.n_silent_zero() < 3 {
            return Err(format!("PLANT FAIL: 3 scale_q=0 blocks planted, only {} detected SilentZero", mask.n_silent_zero()));
        }
        detected_silent += mask.n_silent_zero();
        if decode_q12_silence(&enc, &cfg, lut, &mask) != decode_tensor_fixed(&enc, &cfg) {
            return Err("DECODE IDENTITY FAIL on hand-planted scale_q=0 tensor".into());
        }
        checked += 1;
    }

    for &(rows, cols) in &[(8usize, 896usize), (4, 256), (5, 320), (3, 1024)] {
        for cfg in [TrellisConfig::for_bpw_l(2.0, 12), TrellisConfig::for_bpw(3.0)] {
            let lut = codebook_lut(cfg.l_bits);
            let w = planted_weights(rows * cols, cfg.block_len, 11);
            let enc = encode_tensor(&w, &cfg);
            let mask = SilenceMask::build(&enc, &cfg, lut);
            let x: Vec<f32> = (0..cols)
                .map(|i| match i % 5 {
                    0 => -((i as f32) * 0.013).cos(),
                    1 => 0.0,
                    2 => -0.0,
                    3 => f32::MIN_POSITIVE / 2.0,
                    _ => ((i as f32) * 0.07).cos(),
                })
                .collect();
            let y_ref = strand_decode_kernel::matvec(&enc, &cfg, None, rows, cols, &x);
            let y_a = matvec_silence(&enc, &cfg, lut, &mask, rows, cols, &x);
            let y_b = matvec_silence_skip(&enc, &cfg, lut, &mask, rows, cols, &x);
            for o in 0..rows {
                if y_a[o].to_bits() != y_ref[o].to_bits() {
                    return Err(format!("matvec_silence BIT FAIL row {o} {rows}x{cols} L={}", cfg.l_bits));
                }
                if y_b[o].to_bits() != y_ref[o].to_bits() {
                    return Err(format!("matvec_silence_skip BIT FAIL row {o} {rows}x{cols} L={}", cfg.l_bits));
                }
            }
            checked += 1;
        }
    }

    println!("gate-silence GATE: PASS ({checked} identity cells, {detected_silent} silent blocks exercised)");
    println!("{}", machine_stamp());
    Ok(())
}

fn run_census(path: &str) -> Result<(), String> {
    let model = StrandModel::open(std::path::Path::new(path)).map_err(|e| format!("open {path}: {e}"))?;
    println!("# gate-silence census: {path}");
    println!("{}", machine_stamp());

    let names: Vec<String> = model.tensor_names().map(|s| s.to_string()).collect();
    let mut agg_weights = 0usize;
    let mut agg_blocks = 0usize;
    let mut agg_sub = 0usize;
    let mut agg_silent_zero = 0usize;
    let mut agg_silent_const = 0usize;
    let mut agg_zero_level_blocks = 0usize;
    let mut agg_strong = 0usize;
    let mut agg_sub_code0 = 0usize;
    let mut agg_sub_code_max = 0usize;
    let mut agg_scaleq_zero = 0usize;
    let mut agg_zero_level_visits = 0u64;
    let mut agg_silent_with_outlier = 0usize;
    let mut agg_state_hist: Option<Vec<u64>> = None;

    let mut kind_rollup: BTreeMap<String, (usize, usize, usize, u64, u64)> = BTreeMap::new();

    println!("{:<44} {:>9} {:>7} {:>7} {:>7} {:>8} {:>8} {:>9} {:>8}", "tensor", "blocks", "sil0", "silC", "zlvlB", "sub_c0", "sub_c63", "zlvl_vis", "occ_H");
    for name in &names {
        let hdr = model.tensor_header(name).ok_or_else(|| format!("missing header {name}"))?.clone();
        let cfg = model.config_for(&hdr);
        let enc = model.encoded_tensor(name).ok_or_else(|| format!("encoded_tensor failed for {name}"))?;
        let lut = codebook_lut(cfg.l_bits);

        let mask = SilenceMask::build(&enc, &cfg, lut);
        let got = decode_q12_silence(&enc, &cfg, lut, &mask);
        let want = decode_tensor_q12(&model, name).ok_or_else(|| format!("ref decode failed {name}"))?;
        if got != want {
            return Err(format!("CENSUS ABORT: silence decode != reference on {name}"));
        }
        drop(got);
        drop(want);

        let outlier_idx: Vec<u32> = model.outlier(name).map(|w| w.entries.iter().map(|&(i, _)| i).collect()).unwrap_or_default();
        let c = census_tensor(&enc, &cfg, lut, &outlier_idx);

        let visits: u64 = c.state_hist.iter().sum();
        let h = if visits > 0 {
            c.state_hist
                .iter()
                .filter(|&&v| v > 0)
                .map(|&v| {
                    let p = v as f64 / visits as f64;
                    -p * p.log2()
                })
                .sum::<f64>()
        } else {
            0.0
        };

        println!(
            "{:<44} {:>9} {:>7} {:>7} {:>7} {:>8} {:>8} {:>9} {:>8.3}",
            name, c.n_blocks, c.n_silent_zero, c.n_silent_const, c.n_zero_level_blocks, c.n_sub_code0, c.n_sub_code_max, c.zero_level_visits, h
        );

        agg_weights += c.n_weights;
        agg_blocks += c.n_blocks;
        agg_sub += c.n_sub;
        agg_silent_zero += c.n_silent_zero;
        agg_silent_const += c.n_silent_const;
        agg_zero_level_blocks += c.n_zero_level_blocks;
        agg_strong += c.n_strong_silent;
        agg_sub_code0 += c.n_sub_code0;
        agg_sub_code_max += c.n_sub_code_max;
        agg_scaleq_zero += c.n_scaleq_zero;
        agg_zero_level_visits += c.zero_level_visits;
        agg_silent_with_outlier += c.n_silent_with_outlier;
        match &mut agg_state_hist {
            Some(hist) if hist.len() == c.state_hist.len() => {
                for (a, b) in hist.iter_mut().zip(c.state_hist.iter()) {
                    *a += b;
                }
            }
            None => agg_state_hist = Some(c.state_hist.clone()),
            _ => {}
        }

        let kind = name.rsplit('.').nth(1).unwrap_or("other").to_string();
        let e = kind_rollup.entry(kind).or_insert((0, 0, 0, 0, 0));
        e.0 += c.n_blocks;
        e.1 += c.n_silent_zero + c.n_silent_const;
        e.2 += c.n_sub_code0;
        e.3 += c.zero_level_visits;
        e.4 += c.n_weights as u64;
    }

    println!("\n## AGGREGATE ({} tensors)", names.len());
    println!("weights                {agg_weights}");
    println!("blocks                 {agg_blocks}");
    println!("sub-blocks             {agg_sub}");
    println!("silent-zero blocks     {agg_silent_zero} ({:.4}%)", 100.0 * agg_silent_zero as f64 / agg_blocks.max(1) as f64);
    println!("silent-const blocks    {agg_silent_const} ({:.4}%)", 100.0 * agg_silent_const as f64 / agg_blocks.max(1) as f64);
    println!("zero-level blocks      {agg_zero_level_blocks}");
    println!("strong (eff==0) blocks {agg_strong}");
    println!("scale_q==0 blocks      {agg_scaleq_zero}");
    println!("sub-blocks code 0      {agg_sub_code0} ({:.4}%)", 100.0 * agg_sub_code0 as f64 / agg_sub.max(1) as f64);
    println!("sub-blocks code 63     {agg_sub_code_max} ({:.4}%)", 100.0 * agg_sub_code_max as f64 / agg_sub.max(1) as f64);
    println!("zero-level visits      {agg_zero_level_visits} ({:.4}% of weights)", 100.0 * agg_zero_level_visits as f64 / agg_weights.max(1) as f64);
    println!("silent blocks w/ OUTL  {agg_silent_with_outlier}");

    if let Some(hist) = &agg_state_hist {
        let visits: u64 = hist.iter().sum();
        let mut sorted: Vec<u64> = hist.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let topshare = |k: usize| -> f64 { 100.0 * sorted.iter().take(k).sum::<u64>() as f64 / visits.max(1) as f64 };
        let h: f64 = hist
            .iter()
            .filter(|&&v| v > 0)
            .map(|&v| {
                let p = v as f64 / visits as f64;
                -p * p.log2()
            })
            .sum();
        let nz = hist.iter().filter(|&&v| v > 0).count();
        let lut = codebook_lut((hist.len() as f64).log2() as u32);
        let zmin = zero_nearest_q12(lut);
        println!("\n## STATE OCCUPANCY (model aggregate, {} states)", hist.len());
        println!("states visited         {nz}/{}", hist.len());
        println!("occupancy entropy      {h:.3} bits (uniform = {:.3})", (hist.len() as f64).log2());
        println!("top-1/16/64 state mass {:.3}% / {:.3}% / {:.3}%", topshare(1), topshare(16), topshare(64));
        println!("zero-nearest |level|   {zmin} (Q12)");
    }

    println!("\n## BY TENSOR KIND (blocks, silent, sub_c0, zlvl_visits, weights)");
    for (k, (b, s, c0, zv, nw)) in &kind_rollup {
        println!("{:<12} blocks={:<8} silent={:<6} sub_c0={:<8} zlvl={:<10} zlvl%={:.4}", k, b, s, c0, zv, 100.0 * *zv as f64 / (*nw).max(1) as f64);
    }
    Ok(())
}

fn main() -> ExitCode {
    std::env::set_var("STRAND_NO_GPU", "1");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let res = match args.first().map(|s| s.as_str()) {
        None | Some("gate") => run_gate(),
        Some("census") => match args.get(1) {
            Some(p) => run_census(p),
            None => Err("usage: gate-silence census <artifact.strand>".into()),
        },
        Some(other) => Err(format!("unknown mode {other:?} (gate | census <path>)")),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("gate-silence FAIL: {e}");
            ExitCode::from(101)
        }
    }
}
