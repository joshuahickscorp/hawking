//! gate-debias — measures whether inner-product de-biasing reduces OUTPUT error
//! (the truth, will.md §5.5) at 2-bit, even where rel-RMS (the proxy) is flat.
//!
//! For each real 0.5B tensor it reproduces the EXACT STRAND recon path
//! (outlier-removal -> RHT-forward -> trellis encode -> integer decode ->
//! RHT-inverse -> outlier-patch), then on a batch of Gaussian activation vectors
//! compares W x vs Ŵ x vs (Ŵ x + c), with c the per-row de-bias correction (eq.3
//! in debias.rs), under both a zero-mean and a non-zero-mean activation model.
//!
//! Decision rule printed at the end:
//!   - non-zero-mean: debias must cut output-RMS by >= the kill bar to be ALIVE.
//!   - zero-mean: correction is ~0 by construction (records the dead-Hessian
//!     degeneracy); the rowsum-bias magnitude is reported to confirm it SURVIVES
//!     the RHT (estimable post-rotation) even when the zero-mean correction is moot.
//!
//! Machine stamp + config are printed for the bench ledger.

use std::time::Instant;

use strand_quant::debias::{debias_tensor, estimate_mu_bar, output_error};
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor_with, EncodeOpts};
use strand_quant::gate_utils::{normal_vec, rel_rms, rht_seed_for};
use strand_quant::rht::{rht_forward_rows, rht_inverse_rows_inplace, RhtConfig};
use strand_quant::safetensor_io::SafeTensors;
use strand_quant::TrellisConfig;

const Q12_TO_F32: f32 = 1.0 / 4096.0;

/// Real STRAND recon for one tensor, matching quantize-model's per-tensor flow.
fn strand_recon(w: &[f32], in_features: usize, name: &str, cfg: &TrellisConfig, outlier_pct: f64) -> (Vec<f32>, f64) {
    let n = w.len();
    // 1. outlier removal in WEIGHT space, pre-RHT (top-|w|).
    let mut idx_vals: Vec<(usize, f32)> = Vec::new();
    let work_gt: Vec<f32> = if outlier_pct > 0.0 {
        let k = ((outlier_pct / 100.0) * n as f64).round() as usize;
        if k > 0 {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| w[b].abs().partial_cmp(&w[a].abs()).unwrap());
            let mut b = w.to_vec();
            for &i in order.iter().take(k) {
                idx_vals.push((i, w[i]));
                b[i] = 0.0;
            }
            b
        } else {
            w.to_vec()
        }
    } else {
        w.to_vec()
    };

    // 2. RHT-forward (row-aware).
    let rcfg = RhtConfig::from_seed(rht_seed_for(name));
    let workspace = rht_forward_rows(&work_gt, &rcfg, in_features);

    // 3. trellis encode + 4. integer decode.
    let opts = EncodeOpts { adaptive: true, ..EncodeOpts::default() };
    let mut enc = encode_tensor_with(&workspace, cfg, &opts);
    enc.has_rht_seed = true;
    let q12 = decode_tensor_fixed(&enc, cfg);
    let mut recon: Vec<f32> = q12.iter().map(|&q| (q as f32) * Q12_TO_F32).collect();

    // 5. RHT-inverse.
    rht_inverse_rows_inplace(&mut recon, &rcfg, in_features);

    // 6. outlier-patch.
    let mut eff_bpw = enc.total_bpw(cfg);
    if !idx_vals.is_empty() {
        for &(i, v) in &idx_vals {
            recon[i] = v;
        }
        let f = idx_vals.len() as f64 / n.max(1) as f64;
        let idx_bits = (n as f64).log2().ceil();
        eff_bpw += f * (idx_bits + 8.0);
    }
    (recon, eff_bpw)
}

fn main() {
    let model = std::env::args().nth(1).unwrap_or_else(|| "scratch/qwen-05b/model.safetensors".to_string());
    let max_tensors: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(14);
    let n_acts = 64usize; // activation samples per tensor

    println!("== gate-debias (inner-product de-biasing, 2-bit) ==");
    println!("machine: {} | model: {} | max_tensors: {} | acts/tensor: {}", machine_stamp(), model, max_tensors, n_acts);
    // 2-bit operating point: l=12, +1% outlier (will.md canon).
    let cfg = TrellisConfig::for_bpw_l(2.0, 12);
    let outlier_pct = 1.0;
    println!("config: bits=2 l=12 outlier={}% (the 2-bit PTQ floor config)", outlier_pct);

    let st = match SafeTensors::open(&model) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot open model {model}: {e} — synthesising one Gaussian tensor instead");
            run_synthetic(&cfg, outlier_pct, n_acts);
            return;
        }
    };

    let proj = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight"];

    // Aggregate over the non-zero-mean model.
    let mut agg_rms_uncorr = 0.0f64;
    let mut agg_rms_corr = 0.0f64;
    let mut agg_bias_uncorr = 0.0f64;
    let mut agg_bias_corr = 0.0f64;
    let mut nt = 0usize;

    println!("\nNON-ZERO-MEAN activation model (mu_bar estimated from the act sample):");
    println!("{:<34} {:>7} {:>11} {:>11} {:>11} {:>11} {:>11}", "tensor", "relRMS%", "outBias_u", "outBias_c", "outRMS_u", "outRMS_c", "dRMS%");

    let t0 = Instant::now();
    for name in &st.order {
        if nt >= max_tensors {
            break;
        }
        let t = &st.tensors[name];
        if t.shape.len() != 2 || !proj.iter().any(|p| name.ends_with(p)) {
            continue;
        }
        let w = st.to_f32(t);
        let in_features = t.shape[1] as usize;

        let (recon, _bpw) = strand_recon(&w, in_features, name, &cfg, outlier_pct);
        let rr = rel_rms(&w, &recon) * 100.0;

        // Non-zero-mean activations: shifted Gaussian, mean = 0.3 (a realistic
        // residual-stream DC; RMSNorm does not centre). Deterministic seed per tensor.
        let seed = rht_seed_for(name) ^ 0xA11C_AC75;
        let act_mean = 0.3f32;
        let xs: Vec<Vec<f32>> = (0..n_acts)
            .map(|s| {
                let mut v = normal_vec(in_features, seed.wrapping_add(s as u64));
                for e in v.iter_mut() {
                    *e += act_mean;
                }
                v
            })
            .collect();
        let mu_bar = estimate_mu_bar(&xs);
        let r = debias_tensor(&w, &recon, in_features, mu_bar, 16);

        let (b_u, rms_u) = output_error(&w, &recon, in_features, &xs, None);
        let (b_c, rms_c) = output_error(&w, &recon, in_features, &xs, Some(&r.bias_correction));
        let drms = if rms_u > 0.0 { (rms_u - rms_c) / rms_u * 100.0 } else { 0.0 };

        println!("{:<34} {:>7.3} {:>11.4e} {:>11.4e} {:>11.4e} {:>11.4e} {:>+10.2}", short(name), rr, b_u, b_c, rms_u, rms_c, drms);
        agg_rms_uncorr += rms_u;
        agg_rms_corr += rms_c;
        agg_bias_uncorr += b_u.abs();
        agg_bias_corr += b_c.abs();
        nt += 1;
    }

    if nt == 0 {
        eprintln!("no matching tensors in {model}");
        return;
    }

    let mean_drms = (agg_rms_uncorr - agg_rms_corr) / agg_rms_uncorr * 100.0;
    println!("\n--- aggregate over {nt} tensors (non-zero-mean, mu_bar~0.3) ---");
    println!("  mean |output bias|  uncorrected = {:.4e}", agg_bias_uncorr / nt as f64);
    println!("  mean |output bias|  de-biased   = {:.4e}", agg_bias_corr / nt as f64);
    println!("  mean output-RMS     uncorrected = {:.4e}", agg_rms_uncorr / nt as f64);
    println!("  mean output-RMS     de-biased   = {:.4e}", agg_rms_corr / nt as f64);
    println!("  output-RMS reduction            = {:+.2}%", mean_drms);
    println!("  side-channel cost @ in=896      = {:.4} bpw (bf16 bias)", 16.0 / 896.0);

    // The honest zero-mean control: rerun output error with a centred act model.
    println!("\nZERO-MEAN control (mu_bar -> 0; eq.1 says correction is vacuous):");
    let mut zm_rowsum_mag = 0.0f64;
    let mut zm_drms = 0.0f64;
    let mut zc = 0usize;
    for name in &st.order {
        if zc >= max_tensors {
            break;
        }
        let t = &st.tensors[name];
        if t.shape.len() != 2 || !proj.iter().any(|p| name.ends_with(p)) {
            continue;
        }
        let w = st.to_f32(t);
        let in_features = t.shape[1] as usize;
        let (recon, _) = strand_recon(&w, in_features, name, &cfg, outlier_pct);
        let seed = rht_seed_for(name) ^ 0x2222_0000;
        let xs: Vec<Vec<f32>> = (0..n_acts).map(|s| normal_vec(in_features, seed.wrapping_add(s as u64))).collect();
        let mu_bar = estimate_mu_bar(&xs); // ~0
        let r = debias_tensor(&w, &recon, in_features, mu_bar, 16);
        let (_, rms_u) = output_error(&w, &recon, in_features, &xs, None);
        let (_, rms_c) = output_error(&w, &recon, in_features, &xs, Some(&r.bias_correction));
        // rowsum-bias magnitude (RMS over rows), normalised by row energy — proves
        // S_i is nonzero post-RHT (estimable; the RHT rotates, does not destroy it).
        let out = w.len() / in_features;
        let mut s2 = 0.0f64;
        for &s in &r.rowsum_bias {
            s2 += (s as f64) * (s as f64);
        }
        zm_rowsum_mag += (s2 / out as f64).sqrt();
        zm_drms += if rms_u > 0.0 { (rms_u - rms_c) / rms_u * 100.0 } else { 0.0 };
        zc += 1;
    }
    println!("  mean rowsum-bias RMS (post-RHT) = {:.4e}  (nonzero => survives RHT)", zm_rowsum_mag / zc as f64);
    println!("  mean output-RMS reduction       = {:+.4}%  (expect ~0 => dead-Hessian degeneracy)", zm_drms / zc as f64);

    // Verdict.
    let kill_bar = 0.5f64; // % output-RMS reduction to justify the side-channel
    println!("\n=== VERDICT (kill bar: >= {kill_bar}% output-RMS cut on non-zero-mean) ===");
    if mean_drms >= kill_bar {
        println!("ALIVE: de-biasing cut output-RMS {mean_drms:+.2}% on non-zero-mean acts");
        println!("  => run the 0.5B PPL A/B (protocol in research/debias-results.md).");
    } else {
        println!("DEAD (at this kill bar): output-RMS cut {mean_drms:+.2}% < {kill_bar}%");
        println!("  => the rowsum bias survives the RHT but is too small to move output");
        println!("     error materially; record as the 4th RHT-whitening kill if zero-mean");
        println!("     control also ~0 (means the only live regime is large act-mean).");
    }
    println!("\n(elapsed {:.2}s)", t0.elapsed().as_secs_f64());
}

fn run_synthetic(cfg: &TrellisConfig, outlier_pct: f64, n_acts: usize) {
    let in_features = 896usize;
    let out = 896usize;
    let w = normal_vec(in_features * out, 0xDEAD_BEEF);
    let (recon, bpw) = strand_recon(&w, in_features, "synthetic.q_proj.weight", cfg, outlier_pct);
    let rr = rel_rms(&w, &recon) * 100.0;
    let xs: Vec<Vec<f32>> = (0..n_acts)
        .map(|s| {
            let mut v = normal_vec(in_features, 0x1234 + s as u64);
            for e in v.iter_mut() {
                *e += 0.3;
            }
            v
        })
        .collect();
    let mu_bar = estimate_mu_bar(&xs);
    let r = debias_tensor(&w, &recon, in_features, mu_bar, 16);
    let (b_u, rms_u) = output_error(&w, &recon, in_features, &xs, None);
    let (b_c, rms_c) = output_error(&w, &recon, in_features, &xs, Some(&r.bias_correction));
    println!("synthetic 896x896: relRMS={rr:.3}% bpw={bpw:.4}");
    println!("  outBias u={b_u:.4e} c={b_c:.4e} | outRMS u={rms_u:.4e} c={rms_c:.4e}");
    println!("  output-RMS reduction = {:+.2}%", (rms_u - rms_c) / rms_u * 100.0);
}

fn short(name: &str) -> &str {
    // last 32 chars for readability in the table
    let n = name.len();
    if n > 32 {
        &name[n - 32..]
    } else {
        name
    }
}

fn machine_stamp() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    format!("{os}/{arch}")
}
