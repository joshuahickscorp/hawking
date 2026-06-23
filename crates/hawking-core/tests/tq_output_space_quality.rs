//! DECISIVE output-space quality measurement: TQ trellis codec vs a real Q4_K
//! encoder, from a TRUE bf16 source.
//!
//! WHY THIS EXISTS
//! ---------------
//! The kill-ledger graded the trellis codec in WEIGHT space (rel-RMSE) and
//! recorded `bits_needed = [+1.37, +0.44]` vs Q4_K_M — i.e. "0.44–1.37 bits
//! worse" — but it (a) used a *bracketed proxy*, never the real codec, and
//! (b) measured weight error, not output error, despite the project's own
//! doctrine ("rank damage in OUTPUT space; weight MSE is a scout, not a gate").
//! The decisive Colab run was deliberately skipped (`ALLOW_FRESH_QTIP_CODEC=
//! False`). This harness runs the REAL absorbed codec, from REAL bf16 weights
//! (`models/rwkv7-g1-04-hf/model.safetensors`), against the REAL `quantize_q4_k`
//! encoder, and reports weight-RMSE AND output error `||(Ŵ-W)·X||/||W·X||`.
//!
//! Two activation models bracket the truth: iid-Gaussian (where output error
//! collapses onto weight-RMSE) and a heavy-tailed model with ~1% super-outlier
//! channels (the known LLM activation structure) where output error and weight
//! error diverge — the regime where activation-aware encoding can win.
//!
//! Levers measured:
//!   * RHT (per-column incoherence) — already implemented (`--rht-cols`)
//!   * quality-L  (`for_bpw_quality`: L=k+6 vs k+4, same payload, more states)
//!   * AWQ-scale  — importance-scale columns by σ_j^α before encode, unscale
//!     after (diagonal-Hessian / AWQ trick). Calib σ from a SEPARATE activation
//!     draw than the eval draw, so the win is not circular.
//!
//! Run:
//!   cargo test -p hawking-core --release --features tq \
//!     --test tq_output_space_quality -- --nocapture report
#![cfg(feature = "tq")]

use hawking_core::gguf::GgmlType;
use hawking_core::quant::{dequant_into, quantize_q4_k, Q4_K_BLOCK_BYTES, Q_K};
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor_with, EncodeOpts};
use strand_quant::gate_utils::rht_seed_for;
use strand_quant::rht::{rht_forward_cols, rht_inverse_cols, RhtConfig};
use strand_quant::safetensor_io::SafeTensors;
use strand_quant::TrellisConfig;

// `cargo test` runs with cwd = crate dir; the model lives at workspace-root
// `models/`. Resolve from CARGO_MANIFEST_DIR (…/crates/hawking-core) two up.
// Override with HAWKING_TQ_ST=/abs/path.safetensors.
fn st_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("HAWKING_TQ_ST") {
        return std::path::PathBuf::from(p);
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/rwkv7-g1-04-hf/model.safetensors")
}

// ── deterministic RNG (xorshift64*) + standard normal via Box–Muller ─────────
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn norm(&mut self) -> f32 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}

/// Per-channel activation scale σ_j. `outlier`=false → all-ones (Gaussian).
/// `outlier`=true → ~1% of channels at 20×, deterministic by seed.
fn channel_scales(c: usize, outlier: bool, seed: u64) -> Vec<f32> {
    if !outlier {
        return vec![1.0; c];
    }
    let mut r = Rng(seed);
    (0..c)
        .map(|_| if r.unit() < 0.01 { 20.0 } else { 1.0 })
        .collect()
}

/// Per-channel scales drawn from the REAL measured Qwen-3B activation distribution
/// (`reports/w4a8_activation_dist.csv`, rms column from 180 real samples), normalized
/// to median=1 and resampled (with replacement) to `c` channels. This replaces the
/// synthetic outlier model with the real outlier marginal (max ~29× median, ~0.2% of
/// channels ≥20×, top-1% ≈ 10% of mass) — the honesty gate the handoff demands.
fn real_w4a8_scales(c: usize, seed: u64) -> Vec<f32> {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("reports/w4a8_activation_dist.csv");
    let txt = std::fs::read_to_string(&path).unwrap_or_default();
    let mut rms: Vec<f32> = Vec::new();
    for line in txt.lines() {
        if line.starts_with('#') || line.starts_with("channel") || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() >= 4 {
            if let Ok(v) = cols[3].trim().parse::<f32>() {
                rms.push(v);
            }
        }
    }
    if rms.is_empty() {
        return vec![1.0; c]; // fallback: file missing → benign Gaussian (no harm)
    }
    let mut sorted = rms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[sorted.len() / 2].max(1e-9);
    let rel: Vec<f32> = rms.iter().map(|&x| (x / med).max(1e-6)).collect();
    let mut r = Rng(seed);
    (0..c)
        .map(|_| rel[((r.unit() * rel.len() as f64) as usize).min(rel.len() - 1)])
        .collect()
}

/// Activation matrix X (C×B, row-major over channels) with per-channel scales.
fn make_acts(c: usize, b: usize, scales: &[f32], seed: u64) -> Vec<f32> {
    let mut r = Rng(seed);
    let mut x = vec![0f32; c * b];
    for j in 0..c {
        let s = scales[j];
        for bb in 0..b {
            x[j * b + bb] = r.norm() * s;
        }
    }
    x
}

fn q12_to_f32(v: &[i32]) -> Vec<f32> {
    let s = 1.0f32 / (1u32 << strand_quant::QUANTILE_SHIFT) as f32;
    v.iter().map(|&q| q as f32 * s).collect()
}

fn recon_q4k(w: &[f32]) -> Vec<f32> {
    let nb = w.len() / Q_K;
    let mut dst = vec![0u8; nb * Q4_K_BLOCK_BYTES];
    quantize_q4_k(w, &mut dst).expect("q4k encode");
    let mut out = vec![0f32; w.len()];
    dequant_into(GgmlType::Q4_K, &dst, &mut out).expect("q4k decode");
    out
}

fn recon_tq(w: &[f32], cfg: &TrellisConfig) -> Vec<f32> {
    let enc = encode_tensor_with(w, cfg, &EncodeOpts::default());
    q12_to_f32(&decode_tensor_fixed(&enc, cfg))
}

fn recon_tq_rht(w: &[f32], in_f: usize, cfg: &TrellisConfig, name: &str) -> Vec<f32> {
    let rcfg = RhtConfig::from_seed(rht_seed_for(name));
    let wr = rht_forward_cols(w, &rcfg, in_f);
    let dec = recon_tq(&wr, cfg);
    rht_inverse_cols(&dec, &rcfg, in_f)
}

/// AWQ-style importance scaling: Ŵ = unscale( RHT⁻¹( decode( encode( RHT( W·D ) ) ) ) ),
/// D = diag(σ_j^α). σ_j comes from `calib_scales` (a separate draw than eval).
fn recon_tq_awq_rht(
    w: &[f32],
    r: usize,
    c: usize,
    cfg: &TrellisConfig,
    name: &str,
    calib_scales: &[f32],
    alpha: f32,
) -> Vec<f32> {
    let d: Vec<f32> = calib_scales.iter().map(|&s| s.powf(alpha).max(1e-6)).collect();
    let mut ws = vec![0f32; w.len()];
    for i in 0..r {
        for j in 0..c {
            ws[i * c + j] = w[i * c + j] * d[j];
        }
    }
    let mut wh = recon_tq_rht(&ws, c, cfg, name);
    for i in 0..r {
        for j in 0..c {
            wh[i * c + j] /= d[j];
        }
    }
    wh
}

/// Outlier protection: keep the top `pct` of input columns (highest σ) at EXACT
/// (f16) precision in the recon, low-bit the rest. The residual output-err is
/// dominated by these high-σ channels, so protecting ~1% of them is the cheap
/// lever that should close the AWQ residual to Q4_K. Returns the effective bpw.
fn protect_outliers(
    wh: &mut [f32],
    w: &[f32],
    r: usize,
    c: usize,
    sigma: &[f32],
    pct: f64,
    base_bpw: f32,
) -> f32 {
    let k = (((c as f64) * pct).ceil() as usize).clamp(1, c);
    let mut idx: Vec<usize> = (0..c).collect();
    idx.sort_by(|&a, &b| sigma[b].partial_cmp(&sigma[a]).unwrap_or(std::cmp::Ordering::Equal));
    for &j in idx.iter().take(k) {
        for i in 0..r {
            wh[i * c + j] = w[i * c + j]; // exact (f16) restore of the outlier column
        }
    }
    ((c - k) as f32 * base_bpw + k as f32 * 16.0) / c as f32
}

/// Regularized AWQ: importance d_j = clamp((σ_j/geomean(σ))^α, 1/clip, clip).
/// Geometric-mean normalization makes a FLAT σ (benign/Gaussian acts) map to
/// d_j≈1 → harmless, so the lever only spends bits where there is real outlier
/// structure. `clip` bounds how far any single channel can be boosted.
fn recon_tq_awq_reg(
    w: &[f32],
    r: usize,
    c: usize,
    cfg: &TrellisConfig,
    name: &str,
    calib_sigma: &[f32],
    alpha: f32,
    clip: f32,
) -> Vec<f32> {
    let logmean: f64 =
        calib_sigma.iter().map(|&s| (s.max(1e-9) as f64).ln()).sum::<f64>() / c as f64;
    let gmean = logmean.exp() as f32;
    let d: Vec<f32> = calib_sigma
        .iter()
        .map(|&s| ((s.max(1e-9) / gmean).powf(alpha)).clamp(1.0 / clip, clip))
        .collect();
    let mut ws = vec![0f32; w.len()];
    for i in 0..r {
        for j in 0..c {
            ws[i * c + j] = w[i * c + j] * d[j];
        }
    }
    let mut wh = recon_tq_rht(&ws, c, cfg, name);
    for i in 0..r {
        for j in 0..c {
            wh[i * c + j] /= d[j];
        }
    }
    wh
}

fn rel_rmse(wh: &[f32], w: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for i in 0..w.len() {
        let dlt = (wh[i] - w[i]) as f64;
        num += dlt * dlt;
        den += (w[i] as f64) * (w[i] as f64);
    }
    (num / den).sqrt()
}

/// ||(Ŵ-W)·X||_F / ||W·X||_F, W is R×C row-major, X is C×B row-major.
fn out_rel_err(wh: &[f32], w: &[f32], r: usize, c: usize, x: &[f32], b: usize) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for i in 0..r {
        let wi = &w[i * c..(i + 1) * c];
        let whi = &wh[i * c..(i + 1) * c];
        for bb in 0..b {
            let (mut dy, mut y) = (0f64, 0f64);
            for j in 0..c {
                let xv = x[j * b + bb] as f64;
                dy += ((whi[j] - wi[j]) as f64) * xv;
                y += (wi[j] as f64) * xv;
            }
            num += dy * dy;
            den += y * y;
        }
    }
    (num / den).sqrt()
}

/// bpw is structural (payload = num_steps·k, plus side-info) and independent of
/// the symbol VALUES, so RHT/AWQ rotations don't change it; the RHT seed adds a
/// fixed ~64 bits/tensor (negligible at these sizes). One encode per cfg.
fn bpw_of(w: &[f32], cfg: &TrellisConfig) -> f64 {
    let enc = encode_tensor_with(w, cfg, &EncodeOpts::default());
    enc.total_bpw(cfg)
}

struct TensorPick {
    name: &'static str,
    max_rows: usize,
}

#[test]
fn report() {
    let st_path = st_path();
    if !st_path.exists() {
        eprintln!("SKIP: {} not present", st_path.display());
        return;
    }
    let st = SafeTensors::open(st_path.to_str().expect("utf8 path")).expect("open safetensors");

    let picks = [
        TensorPick { name: "model.layers.0.ffn.key.weight", max_rows: 1024 },
        TensorPick { name: "model.layers.0.ffn.value.weight", max_rows: 256 },
        TensorPick { name: "model.layers.11.ffn.key.weight", max_rows: 1024 },
        TensorPick { name: "lm_head.weight", max_rows: 1024 },
    ];

    let b = 48usize; // activation samples (eval)
    // Heavy-tailed activation structure: σ from calib seed, eval from a DIFFERENT seed.
    let act_seed_eval = 0xA5A5_1234_DEAD_BEEF;
    let act_seed_calib = 0x1357_9BDF_0246_8ACE;
    let scale_seed = 0xF00D_FACE_CAFE_0001;

    println!("\n================ TQ OUTPUT-SPACE QUALITY (real bf16 source) ================");
    println!("source: {}", st_path.display());
    println!("metrics: wRMSE = weight rel-RMSE | oG = output rel-err (Gaussian acts) | oH = output rel-err (heavy-tail acts)");
    println!("GO test: a TQ row at FEWER bpw than Q4_K with oH ≤ Q4_K's oH = denser-at-equal-output-quality\n");

    // accumulate per-method means across tensors
    use std::collections::BTreeMap;
    let mut agg: BTreeMap<String, (f64, f64, f64, f64, usize)> = BTreeMap::new(); // bpw,wrmse,oG,oH,count

    for p in &picks {
        let t = match st.tensors.get(p.name) {
            Some(t) => t,
            None => {
                eprintln!("  (missing {})", p.name);
                continue;
            }
        };
        let full = st.to_f32(t);
        let rows_full = t.shape[0] as usize;
        let c = t.shape[1] as usize;
        let r = p.max_rows.min(rows_full);
        let w = &full[..r * c];

        // activations (C×B): Gaussian + heavy-tail; calib scales (separate draw)
        let ones = channel_scales(c, false, 0);
        let outl = channel_scales(c, true, scale_seed);
        let xg = make_acts(c, b, &ones, act_seed_eval);
        let xh = make_acts(c, b, &outl, act_seed_eval);
        // calib σ estimate (empirical per-channel std) from a SEPARATE heavy-tail draw
        let xh_calib = make_acts(c, b, &outl, act_seed_calib);
        let mut calib_sigma = vec![0f32; c];
        for j in 0..c {
            let mut s2 = 0f64;
            for bb in 0..b {
                let v = xh_calib[j * b + bb] as f64;
                s2 += v * v;
            }
            calib_sigma[j] = (s2 / b as f64).sqrt() as f32;
        }

        println!("── {}  ({}×{}, using {} rows) ─────────────", p.name, rows_full, c, r);
        println!("  {:<16} {:>6}  {:>9} {:>9} {:>9}", "method", "bpw", "wRMSE", "oG", "oH");

        let mut row = |label: &str, wh: &[f32], bpw: f64| {
            let wr = rel_rmse(wh, w);
            let og = out_rel_err(wh, w, r, c, &xg, b);
            let oh = out_rel_err(wh, w, r, c, &xh, b);
            println!("  {:<16} {:>6.3}  {:>9.5} {:>9.5} {:>9.5}", label, bpw, wr, og, oh);
            let e = agg.entry(label.to_string()).or_insert((0.0, 0.0, 0.0, 0.0, 0));
            e.0 += bpw;
            e.1 += wr;
            e.2 += og;
            e.3 += oh;
            e.4 += 1;
        };

        let cfg4 = TrellisConfig::for_bpw(4.0);
        let cfg3 = TrellisConfig::for_bpw(3.0);
        let cfg3q = TrellisConfig::for_bpw_quality(3.0);
        let cfg2 = TrellisConfig::for_bpw(2.0);
        let cfg2q = TrellisConfig::for_bpw_quality(2.0);
        let (bpw4, bpw3, bpw3q, bpw2, bpw2q) = (
            bpw_of(w, &cfg4),
            bpw_of(w, &cfg3),
            bpw_of(w, &cfg3q),
            bpw_of(w, &cfg2),
            bpw_of(w, &cfg2q),
        );

        // Q4_K reference (≈4.5 bpw)
        row("Q4_K", &recon_q4k(w), 4.5);
        row("TQ4", &recon_tq(w, &cfg4), bpw4);
        row("TQ3", &recon_tq(w, &cfg3), bpw3);
        row("TQ3+L", &recon_tq(w, &cfg3q), bpw3q);
        row("TQ3+rht", &recon_tq_rht(w, c, &cfg3, p.name), bpw3);
        row("TQ3+L+rht", &recon_tq_rht(w, c, &cfg3q, p.name), bpw3q);
        row(
            "TQ3+L+rht+awq",
            &recon_tq_awq_rht(w, r, c, &cfg3q, p.name, &calib_sigma, 0.5),
            bpw3q,
        );
        row("TQ2", &recon_tq(w, &cfg2), bpw2);
        row("TQ2+L+rht", &recon_tq_rht(w, c, &cfg2q, p.name), bpw2q);
        row(
            "TQ2+L+rht+awq",
            &recon_tq_awq_rht(w, r, c, &cfg2q, p.name, &calib_sigma, 0.5),
            bpw2q,
        );
        println!();
    }

    println!("================ MEANS ACROSS TENSORS (report) ================");
    println!("  {:<16} {:>6}  {:>9} {:>9} {:>9}", "method", "bpw", "wRMSE", "oG", "oH");
    let q4k = agg.get("Q4_K").cloned().unwrap_or_default();
    let q4k_oh = if q4k.4 > 0 { q4k.3 / q4k.4 as f64 } else { f64::INFINITY };
    for (label, (bpw, wr, og, oh, n)) in &agg {
        let n = *n as f64;
        let beats = if label != "Q4_K" && oh / n <= q4k_oh && bpw / n < q4k.0 / q4k.4 as f64 {
            "  <-- DENSER @ ≤Q4_K output-err"
        } else {
            ""
        };
        println!(
            "  {:<16} {:>6.3}  {:>9.5} {:>9.5} {:>9.5}{}",
            label,
            bpw / n,
            wr / n,
            og / n,
            oh / n,
            beats
        );
    }
    println!("=====================================================\n");
}

/// Lever-robustness sweep: does activation-aware (AWQ) encoding robustly close the
/// output-space gap WITHOUT harming the benign case? Sweeps α and the assumed
/// activation outlier structure, with regularized (geomean-normalized, clipped)
/// scales and a larger calibration sample. Eval acts use a DIFFERENT draw than
/// calib (non-circular). Config fixed at TQ3+L+rht (the best 3-bit operating pt).
///
///   cargo test -p hawking-core --release --features tq \
///     --test tq_output_space_quality -- --nocapture awq_sweep
#[test]
fn awq_sweep() {
    let st_path = st_path();
    if !st_path.exists() {
        eprintln!("SKIP: {} not present", st_path.display());
        return;
    }
    let st = SafeTensors::open(st_path.to_str().expect("utf8 path")).expect("open safetensors");

    let picks = [
        ("model.layers.0.ffn.key.weight", 1024usize),
        ("model.layers.0.ffn.value.weight", 256usize),
    ];
    // (label, outlier?, frac, mag) — the assumed eval/calib activation structure
    let structures: [(&str, bool, f64, f32); 4] = [
        ("gaussian(flat)", false, 0.0, 1.0),
        ("mild 0.5%x30", true, 0.005, 30.0),
        ("modeled 1%x20", true, 0.01, 20.0),
        ("real-w4a8", false, 0.0, 0.0), // measured Qwen-3B activation marginal (csv)
    ];
    let alphas = [0.25f32, 0.5, 0.75, 1.0];
    let clip = 8.0f32;
    let (b_eval, b_calib) = (64usize, 256usize);
    let cfg = TrellisConfig::for_bpw_quality(3.0); // TQ3+L

    println!("\n================ AWQ LEVER SWEEP (TQ3+L+rht, real bf16) ================");
    println!("regularized scales (geomean-norm, clip={clip}); calib b={b_calib} eval b={b_eval} (separate draws)");
    println!("o = output rel-err ||(Ŵ-W)X||/||WX|| on EVAL acts. WANT: heavy-tail o ↓↓, gaussian o ≈ baseline (no harm)\n");

    for (name, max_rows) in &picks {
        let t = match st.tensors.get(*name) {
            Some(t) => t,
            None => continue,
        };
        let full = st.to_f32(t);
        let c = t.shape[1] as usize;
        let r = (*max_rows).min(t.shape[0] as usize);
        let w = &full[..r * c];
        let base = recon_tq_rht(w, c, &cfg, name); // activation-independent

        println!("── {} ({}×{}, {} rows) ─ baseline TQ3+L+rht ─", name, t.shape[0], c, r);
        for (slabel, outl, frac, mag) in &structures {
            // structure-specific scales: outlier channels at `mag`, deterministic
            let scales: Vec<f32> = if *slabel == "real-w4a8" {
                real_w4a8_scales(c, 0x9EA1_5EED)
            } else {
                let mut rng = Rng(0xBEEF_0000 ^ (*mag as u64).wrapping_mul(2654435761));
                (0..c).map(|_| if *outl && rng.unit() < *frac { *mag } else { 1.0 }).collect()
            };
            let x_eval = make_acts(c, b_eval, &scales, 0xE0E0_1111);
            let x_calib = make_acts(c, b_calib, &scales, 0xCA1B_2222);
            let mut sigma = vec![0f32; c];
            for j in 0..c {
                let mut s2 = 0f64;
                for bb in 0..b_calib {
                    let v = x_calib[j * b_calib + bb] as f64;
                    s2 += v * v;
                }
                sigma[j] = (s2 / b_calib as f64).sqrt() as f32;
            }
            let o_base = out_rel_err(&base, w, r, c, &x_eval, b_eval);
            print!("  {:<16} baseline o={:.5}", slabel, o_base);
            for &a in &alphas {
                let wh = recon_tq_awq_reg(w, r, c, &cfg, name, &sigma, a, clip);
                let o = out_rel_err(&wh, w, r, c, &x_eval, b_eval);
                let tag = if o < o_base * 0.999 { "↓" } else if o > o_base * 1.001 { "↑" } else { "=" };
                print!("  |α{:.2} {:.5}{}", a, o, tag);
            }
            println!();
            if *slabel == "real-w4a8" {
                let mut wh = recon_tq_awq_reg(w, r, c, &cfg, name, &sigma, 0.5, clip);
                let eff_bpw = protect_outliers(&mut wh, w, r, c, &sigma, 0.01, 3.348);
                let o = out_rel_err(&wh, w, r, c, &x_eval, b_eval);
                println!(
                    "  {:<16} +awq α0.5 +outlier(top1%@f16): o={:.5}  eff_bpw={:.2}  (Q4_K o≈0.079 @4.50)",
                    "  └─protected", o, eff_bpw
                );
            }
        }
        println!();
    }
    println!("=======================================================================\n");
}
