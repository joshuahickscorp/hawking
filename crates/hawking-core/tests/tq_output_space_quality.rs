#![cfg(feature = "tq")]
use hawking_core::gguf::GgmlType;
use hawking_core::quant::{dequant_into, quantize_q4_k, Q4_K_BLOCK_BYTES, Q_K};
use strand_quant::decode::decode_tensor_fixed;
use strand_quant::encode::{encode_tensor_with, EncodeOpts};
use strand_quant::gate_utils::rht_seed_for;
use strand_quant::rht::{rht_forward_cols, rht_inverse_cols, RhtConfig};
use strand_quant::safetensor_io::SafeTensors;
use strand_quant::TrellisConfig;
fn st_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("HAWKING_TQ_ST") {
        return std::path::PathBuf::from(p);
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/rwkv7-g1-04-hf/model.safetensors")
}
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
fn channel_scales(c: usize, outlier: bool, seed: u64) -> Vec<f32> {
    if !outlier {
        return vec![1.0; c];
    }
    let mut r = Rng(seed);
    (0..c)
        .map(|_| if r.unit() < 0.01 { 20.0 } else { 1.0 })
        .collect()
}
fn real_w4a8_scales(c: usize, seed: u64) -> Vec<f32> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../workspace/campaign/records/reports/w4a8_activation_dist.csv");
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
fn recon_tq_awq_rht(
    w: &[f32],
    r: usize,
    c: usize,
    cfg: &TrellisConfig,
    name: &str,
    calib_scales: &[f32],
    alpha: f32,
) -> Vec<f32> {
    let d: Vec<f32> = calib_scales
        .iter()
        .map(|&s| s.powf(alpha).max(1e-6))
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
    idx.sort_by(|&a, &b| {
        sigma[b]
            .partial_cmp(&sigma[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &j in idx.iter().take(k) {
        for i in 0..r {
            wh[i * c + j] = w[i * c + j]; // exact (f16) restore of the outlier column
        }
    }
    ((c - k) as f32 * base_bpw + k as f32 * 16.0) / c as f32
}
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
    let logmean: f64 = calib_sigma
        .iter()
        .map(|&s| (s.max(1e-9) as f64).ln())
        .sum::<f64>()
        / c as f64;
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
        TensorPick {
            name: "model.layers.0.ffn.key.weight",
            max_rows: 1024,
        },
        TensorPick {
            name: "model.layers.0.ffn.value.weight",
            max_rows: 256,
        },
        TensorPick {
            name: "model.layers.11.ffn.key.weight",
            max_rows: 1024,
        },
        TensorPick {
            name: "lm_head.weight",
            max_rows: 1024,
        },
    ];
    let b = 48usize; // activation samples (eval)
    let act_seed_eval = 0xA5A5_1234_DEAD_BEEF;
    let act_seed_calib = 0x1357_9BDF_0246_8ACE;
    let scale_seed = 0xF00D_FACE_CAFE_0001;
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
        let ones = channel_scales(c, false, 0);
        let outl = channel_scales(c, true, scale_seed);
        let xg = make_acts(c, b, &ones, act_seed_eval);
        let xh = make_acts(c, b, &outl, act_seed_eval);
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
        let mut row = |label: &str, wh: &[f32], bpw: f64| {
            let wr = rel_rmse(wh, w);
            let og = out_rel_err(wh, w, r, c, &xg, b);
            let oh = out_rel_err(wh, w, r, c, &xh, b);
            let e = agg
                .entry(label.to_string())
                .or_insert((0.0, 0.0, 0.0, 0.0, 0));
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
    }
    let q4k = agg.get("Q4_K").cloned().unwrap_or_default();
    let q4k_oh = if q4k.4 > 0 {
        q4k.3 / q4k.4 as f64
    } else {
        f64::INFINITY
    };
    for (label, (bpw, wr, og, oh, n)) in &agg {
        let n = *n as f64;
        let beats = if label != "Q4_K" && oh / n <= q4k_oh && bpw / n < q4k.0 / q4k.4 as f64 {
            "  <-- DENSER @ ≤Q4_K output-err"
        } else {
            ""
        };
    }
}
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
        for (slabel, outl, frac, mag) in &structures {
            let scales: Vec<f32> = if *slabel == "real-w4a8" {
                real_w4a8_scales(c, 0x9EA1_5EED)
            } else {
                let mut rng = Rng(0xBEEF_0000 ^ (*mag as u64).wrapping_mul(2654435761));
                (0..c)
                    .map(|_| {
                        if *outl && rng.unit() < *frac {
                            *mag
                        } else {
                            1.0
                        }
                    })
                    .collect()
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
                let tag = if o < o_base * 0.999 {
                    "↓"
                } else if o > o_base * 1.001 {
                    "↑"
                } else {
                    "="
                };
                print!("  |α{:.2} {:.5}{}", a, o, tag);
            }
            if *slabel == "real-w4a8" {
                let mut wh = recon_tq_awq_reg(w, r, c, &cfg, name, &sigma, 0.5, clip);
                let eff_bpw = protect_outliers(&mut wh, w, r, c, &sigma, 0.01, 3.348);
                let o = out_rel_err(&wh, w, r, c, &x_eval, b_eval);
            }
        }
    }
}
fn lowrank_heal(wh: &[f32], w: &[f32], rows: usize, cols: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rmat: Vec<f32> = (0..rows * cols).map(|i| w[i] - wh[i]).collect();
    let mut lr = vec![0f32; rows * cols];
    let mut rng = Rng(seed);
    let mut u = vec![0f32; rows];
    for _t in 0..k {
        let mut v: Vec<f32> = (0..cols).map(|_| rng.norm()).collect();
        let mut nv = (v.iter().map(|x| (x * x) as f64).sum::<f64>()).sqrt() as f32;
        if nv < 1e-12 {
            break;
        }
        for x in v.iter_mut() {
            *x /= nv;
        }
        for _it in 0..10 {
            for i in 0..rows {
                let row = &rmat[i * cols..i * cols + cols];
                let mut s = 0f32;
                for j in 0..cols {
                    s += row[j] * v[j];
                }
                u[i] = s;
            }
            let mut vn = vec![0f32; cols];
            for i in 0..rows {
                let ui = u[i];
                let row = &rmat[i * cols..i * cols + cols];
                for j in 0..cols {
                    vn[j] += row[j] * ui;
                }
            }
            nv = (vn.iter().map(|x| (x * x) as f64).sum::<f64>()).sqrt() as f32;
            if nv < 1e-12 {
                break;
            }
            for j in 0..cols {
                v[j] = vn[j] / nv;
            }
        }
        for i in 0..rows {
            let row = &rmat[i * cols..i * cols + cols];
            let mut s = 0f32;
            for j in 0..cols {
                s += row[j] * v[j];
            }
            u[i] = s;
        }
        let sigma = (u.iter().map(|x| (x * x) as f64).sum::<f64>()).sqrt() as f32;
        if sigma < 1e-8 {
            break;
        }
        for x in u.iter_mut() {
            *x /= sigma;
        }
        for i in 0..rows {
            let su = sigma * u[i];
            let ro = i * cols;
            for j in 0..cols {
                let d = su * v[j];
                rmat[ro + j] -= d;
                lr[ro + j] += d;
            }
        }
    }
    (0..rows * cols).map(|i| wh[i] + lr[i]).collect()
}
#[test]
fn recovery() {
    let st_path = st_path();
    if !st_path.exists() {
        eprintln!("SKIP: {} not present", st_path.display());
        return;
    }
    let st = SafeTensors::open(st_path.to_str().expect("utf8")).expect("open st");
    let picks = [
        ("model.layers.0.ffn.key.weight", 1024usize),
        ("model.layers.0.ffn.value.weight", 256usize),
    ];
    let clip = 8.0f32;
    let (b_eval, b_calib) = (64usize, 256usize);
    let tiers = [("TQ3", 3.0f32, 3.348f32), ("TQ2", 2.0f32, 2.344f32)];
    let ranks = [16usize, 32, 64];
    for (name, max_rows) in &picks {
        let t = match st.tensors.get(*name) {
            Some(t) => t,
            None => continue,
        };
        let full = st.to_f32(t);
        let c = t.shape[1] as usize;
        let full_rows = t.shape[0] as usize;
        let r = (*max_rows).min(full_rows);
        let w = &full[..r * c];
        let scales = real_w4a8_scales(c, 0x9EA1_5EED);
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
        let o_q4 = out_rel_err(&recon_q4k(w), w, r, c, &x_eval, b_eval);
        for (tlabel, bpw_q, base_bpw) in &tiers {
            let cfg = TrellisConfig::for_bpw_quality(*bpw_q as f64);
            let awq = recon_tq_awq_reg(w, r, c, &cfg, name, &sigma, 0.5, clip);
            let o_awq = out_rel_err(&awq, w, r, c, &x_eval, b_eval);
            print!("  {:<5} +awq o={:.5} @{:.2}bpw", tlabel, o_awq, base_bpw);
            for &k in &ranks {
                let healed = lowrank_heal(&awq, w, r, c, k, 0x5EED_0001 ^ k as u64);
                let o = out_rel_err(&healed, w, r, c, &x_eval, b_eval);
                let add_bpw =
                    (k as f32 * (full_rows + c) as f32 * 16.0) / (full_rows as f32 * c as f32);
                let tot = base_bpw + add_bpw;
                let win = if o <= o_q4 && tot < 4.5 {
                    " ✓WIN"
                } else {
                    ""
                };
                print!("  | r{:<3} o={:.5} @{:.2}bpw{}", k, o, tot, win);
            }
        }
    }
}
fn alloc_avg_bpw(lvl: &[usize], params: &[f64], lv_bpw: &[f64]) -> f64 {
    let tot: f64 = params.iter().sum();
    lvl.iter()
        .zip(params)
        .map(|(&l, &p)| lv_bpw[l] * p)
        .sum::<f64>()
        / tot
}
#[test]
fn allocate() {
    let st_path = st_path();
    if !st_path.exists() {
        eprintln!("SKIP: {} not present", st_path.display());
        return;
    }
    let st = SafeTensors::open(st_path.to_str().expect("utf8")).expect("open st");
    let mut names: Vec<String> = st
        .tensors
        .iter()
        .filter(|(_, t)| {
            t.shape.len() == 2 && t.shape[1] % 256 == 0 && t.shape[0] * t.shape[1] >= 256 * 256
        })
        .map(|(n, _)| n.clone())
        .collect();
    names.sort();
    names.truncate(8);
    if names.len() < 3 {
        eprintln!("SKIP: <3 pickable tensors");
        return;
    }
    let clip = 8.0f32;
    let (b_eval, b_calib) = (64usize, 256usize);
    let lv_bits = [2usize, 3, 4];
    let lv_bpw = [2.34f64, 3.348, 4.50];
    let mut o_all: Vec<[f64; 3]> = Vec::new();
    let mut params: Vec<f64> = Vec::new();
    for n in &names {
        let t = &st.tensors[n];
        let c = t.shape[1] as usize;
        let full_rows = t.shape[0] as usize;
        let r = full_rows.min(256);
        let full = st.to_f32(t);
        let w = &full[..r * c];
        let tseed = n.bytes().fold(0xC0FFEE_u64, |a, b| {
            a.wrapping_mul(1099511628211).wrapping_add(b as u64)
        });
        let scales = real_w4a8_scales(c, tseed ^ 0x9EA1_5EED);
        let x = make_acts(c, b_eval, &scales, tseed ^ 0xE0E0_1111);
        let xc = make_acts(c, b_calib, &scales, tseed ^ 0xCA1B_2222);
        let mut sigma = vec![0f32; c];
        for j in 0..c {
            let mut s = 0f64;
            for bb in 0..b_calib {
                let v = xc[j * b_calib + bb] as f64;
                s += v * v;
            }
            sigma[j] = (s / b_calib as f64).sqrt() as f32;
        }
        let mut o = [0f64; 3];
        for (li, bits) in lv_bits.iter().enumerate() {
            let cfg = TrellisConfig::for_bpw_quality(*bits as f64);
            let awq = recon_tq_awq_reg(w, r, c, &cfg, n, &sigma, 0.5, clip);
            o[li] = out_rel_err(&awq, w, r, c, &x, b_eval);
        }
        o_all.push(o);
        params.push((full_rows * c) as f64);
    }
    let nt = o_all.len();
    let uni_worst = o_all.iter().map(|o| o[1]).fold(0.0, f64::max);
    let uni_mean = o_all.iter().map(|o| o[1]).sum::<f64>() / nt as f64;
    let mut lvl = vec![0usize; nt];
    loop {
        let mut bi: i32 = -1;
        let mut wo = -1.0f64;
        for i in 0..nt {
            if lvl[i] < 2 && o_all[i][lvl[i]] > wo {
                wo = o_all[i][lvl[i]];
                bi = i as i32;
            }
        }
        if bi < 0 {
            break;
        }
        let mut trial = lvl.clone();
        trial[bi as usize] += 1;
        if alloc_avg_bpw(&trial, &params, &lv_bpw) > lv_bpw[1] + 1e-6 {
            break;
        }
        lvl = trial;
    }
    let mix_worst = (0..nt).map(|i| o_all[i][lvl[i]]).fold(0.0, f64::max);
    let mix_mean = (0..nt).map(|i| o_all[i][lvl[i]]).sum::<f64>() / nt as f64;
    let mix_bpw = alloc_avg_bpw(&lvl, &params, &lv_bpw);
    for i in 0..nt {}
    let verdict = if (mix_worst - uni_worst).abs() < 1e-9 {
        "TIES uniform — homogeneous sensitivity / 2-bit intolerable on all tensors => recovery (QAT/KD) is the only 2-bit path"
    } else if mix_worst < uni_worst {
        "IMPROVES on uniform"
    } else {
        "is WORSE than uniform"
    };
}
