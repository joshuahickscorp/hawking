// gate-rht-seedsearch — Wave-4 bet 4: RHT SEED/BASIS SEARCH (moat-safe).
//
// SpinQuant's useful warning is that rotations are not all equal. Dense learned rotations
// would break STRAND's fast deterministic basis, but a seed/basis search INSIDE the existing
// fast RHT family is cheap and moat-safe (docs/STRAND-speed-moonshot-research.md §4):
//
//   for each tensor, search a small deterministic bank:
//     - the current seed (rht_seed_for(name)),
//     - alternate splitmix/FNV-salted seeds,
//     - row-sign vs column-sign RHT variants,
//   score with an OUTPUT-ERROR proxy first, then store ONLY a seed/basis id.
//
// Runtime cost is UNCHANGED for row-RHT; for col-RHT it stays one activation transform per
// tensor. The archive would carry one extra small int (seed id + a row/col bit) per tensor.
//
// Output-error proxy (the doc's activation-weighted distortion family,
//   distortion_j = (w_j - q_j)^2 * E[x_j^2]):
//   we do NOT have per-tensor activation stats here, so we use the column (input-feature)
//   ENERGY of the weight matrix itself, E_j = mean_o w[o,j]^2, as a deterministic, RHT-domain-
//   independent saliency surrogate, and report BOTH it and plain rel-RMS. The promotion bar is
//   intentionally conservative: a candidate is only "interesting" if it beats the default on
//   the activation-weighted proxy by a clear margin; the real PPL/KL confirmation is the
//   downstream eval gate, not this screen.
//
// This is the non-give-up version of "learned rotations": search the fast family STRAND can
// actually serve, not arbitrary dense matrices.
//
// Run: cargo run -p strand-quant --release --bin gate-rht-seedsearch -- \
//          --in scratch/qwen-05b/model.safetensors --bits 3 [--seeds 6] [--only down_proj]

use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor_with, EncodeOpts};
use strand_quant::gate_utils::{is_quantizable_linear, rht_seed_for};
use strand_quant::rht::{rht_forward_cols, rht_forward_rows, rht_inverse_cols, rht_inverse_rows,
    RhtConfig};
use strand_quant::safetensor_io::SafeTensors;
use strand_quant::TrellisConfig;

fn parse() -> (String, u32, u32, usize, Option<String>) {
    let mut input = String::from("scratch/qwen-05b/model.safetensors");
    let mut bits = 3u32;
    let mut l = 0u32;
    let mut seeds = 6usize;
    let mut only: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--in" | "--input" => input = it.next().expect("--in path"),
            "--bits" => bits = it.next().expect("--bits").parse().expect("int"),
            "--l" => l = it.next().expect("--l").parse().expect("int"),
            "--seeds" => seeds = it.next().expect("--seeds").parse().expect("int"),
            "--only" => only = Some(it.next().expect("--only substr")),
            "-h" | "--help" => {
                println!("gate-rht-seedsearch --in <st> --bits N [--l N] [--seeds K] [--only sub]");
                std::process::exit(0);
            }
            other => panic!("unknown arg {other}"),
        }
    }
    (input, bits, l, seeds, only)
}

// Deterministic alternate seeds: salt the FNV name-seed with splitmix64 mixes. Seed 0 (the
// default rht_seed_for) is always candidate index 0 so "no change" is representable.
fn seed_bank(name: &str, k: usize) -> Vec<u64> {
    let base = rht_seed_for(name);
    let mut out = vec![base];
    let mut s = base;
    while out.len() < k {
        // splitmix64 step
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let cand = (z ^ (z >> 31)) | 1;
        out.push(cand);
    }
    out
}

#[derive(Clone, Copy)]
struct Score {
    rel_rms: f64,
    aw_err: f64, // activation(column-energy)-weighted relative error
}

fn score_variant(
    gt: &[f32],
    in_f: usize,
    cfg: &TrellisConfig,
    opts: &EncodeOpts,
    seed: u64,
    cols: bool,
    col_energy: &[f32],
) -> Score {
    let rcfg = RhtConfig::from_seed(seed);
    let work = if cols {
        rht_forward_cols(gt, &rcfg, in_f)
    } else {
        rht_forward_rows(gt, &rcfg, in_f)
    };
    let enc = encode_tensor_with(&work, cfg, opts);
    let q12 = decode_tensor_fixed(&enc, cfg);
    let mut recon: Vec<f32> = q12.iter().map(|&q| (q as f32) * (1.0 / 4096.0)).collect();
    if cols {
        recon = rht_inverse_cols(&recon, &rcfg, in_f);
    } else {
        recon = rht_inverse_rows(&recon, &rcfg, in_f);
    }
    let rows = gt.len() / in_f;
    let (mut se, mut pw) = (0.0f64, 0.0f64);
    let (mut awe, mut awp) = (0.0f64, 0.0f64);
    for o in 0..rows {
        for j in 0..in_f {
            let idx = o * in_f + j;
            let d = gt[idx] as f64 - recon[idx] as f64;
            se += d * d;
            pw += (gt[idx] as f64) * (gt[idx] as f64);
            let e = col_energy[j] as f64;
            awe += d * d * e;
            awp += (gt[idx] as f64) * (gt[idx] as f64) * e;
        }
    }
    Score {
        rel_rms: if pw > 0.0 { (se / pw).sqrt() * 100.0 } else { 0.0 },
        aw_err: if awp > 0.0 { (awe / awp).sqrt() * 100.0 } else { 0.0 },
    }
}

fn main() {
    let (input, bits, l, k, only) = parse();
    let cfg = if l != 0 {
        TrellisConfig::for_bpw_l(bits as f64, l)
    } else {
        TrellisConfig::for_bpw(bits as f64)
    };
    let opts = EncodeOpts { adaptive: true, ..EncodeOpts::default() };
    let st = SafeTensors::open(&input).expect("open safetensors");

    // Pick one representative tensor per projection class from layer 0 (cheap, structural).
    let want_suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ];
    let mut names: Vec<String> = st
        .tensors
        .keys()
        .filter(|n| n.contains("layers.0.") && want_suffixes.iter().any(|s| n.ends_with(s)))
        .filter(|n| {
            let t = &st.tensors[*n];
            is_quantizable_linear(n, &t.shape)
        })
        .cloned()
        .collect();
    if let Some(sub) = &only {
        names.retain(|n| n.contains(sub.as_str()));
    }
    names.sort();

    println!("gate-rht-seedsearch — Wave-4 bet 4: deterministic RHT seed/basis search");
    println!(
        "  model={input}  bits={bits} (k={},L={})  seed-bank={k} x {{row,col}} = {} variants/tensor",
        cfg.k_bits, cfg.l_bits, 2 * k
    );
    println!(
        "\n{:<34} {:>9} {:>9}  {:>9} {:>9}  {:>10}",
        "tensor", "def relRMS", "def AWerr", "best relRMS", "best AWerr", "best variant"
    );
    println!("{}", "-".repeat(92));

    let mut agg_def_aw = 0.0f64;
    let mut agg_best_aw = 0.0f64;
    let mut any_gain = false;
    for name in &names {
        let t = &st.tensors[name];
        let in_f = *t.shape.last().unwrap() as usize;
        let gt = st.to_f32(t);
        let rows = gt.len() / in_f;
        // column (input-feature) energy E_j = mean_o w[o,j]^2
        let mut col_energy = vec![0.0f32; in_f];
        for o in 0..rows {
            for j in 0..in_f {
                col_energy[j] += gt[o * in_f + j] * gt[o * in_f + j];
            }
        }
        for e in col_energy.iter_mut() {
            *e /= rows as f32;
        }

        let bank = seed_bank(name, k);
        // default = seed 0, row RHT (what the encoder ships today)
        let def = score_variant(&gt, in_f, &cfg, &opts, bank[0], false, &col_energy);

        let mut best = def;
        let mut best_label = String::from("default(seed0,row)");
        for (si, &seed) in bank.iter().enumerate() {
            for &cols in &[false, true] {
                if si == 0 && !cols {
                    continue; // that's the default, already scored
                }
                let s = score_variant(&gt, in_f, &cfg, &opts, seed, cols, &col_energy);
                // promotion proxy = activation-weighted error (the loss-aligned one)
                if s.aw_err < best.aw_err {
                    best = s;
                    best_label = format!("seed{si},{}", if cols { "col" } else { "row" });
                }
            }
        }
        let short = name.replace("model.layers.0.", "L0.");
        println!(
            "{short:<34} {:>8.3}% {:>8.3}%  {:>8.3}% {:>8.3}%  {best_label:>10}",
            def.rel_rms, def.aw_err, best.rel_rms, best.aw_err
        );
        agg_def_aw += def.aw_err;
        agg_best_aw += best.aw_err;
        if best.aw_err < def.aw_err - 1e-9 && !best_label.starts_with("default") {
            any_gain = true;
        }
    }

    let n = names.len().max(1) as f64;
    let rel_gain = if agg_def_aw > 0.0 {
        100.0 * (agg_def_aw - agg_best_aw) / agg_def_aw
    } else {
        0.0
    };
    println!("\n  mean AW-err: default {:.4}%  best-of-bank {:.4}%  -> {rel_gain:+.3}% proxy gain",
        agg_def_aw / n, agg_best_aw / n);
    if any_gain && rel_gain > 0.5 {
        println!(
            "  verdict: seed/basis search shows >0.5% activation-weighted proxy gain — worth a\n\
             real PPL/KL confirmation on the winning seeds before storing a per-tensor seed id."
        );
    } else {
        println!(
            "  verdict: the default FNV seed (row-RHT) is already at/near the best of the fast\n\
             family for these tensors (proxy gain <=0.5%). RHT incoherence whitens enough that\n\
             seed choice is ~noise; no per-tensor seed id is justified by this screen."
        );
    }
}
