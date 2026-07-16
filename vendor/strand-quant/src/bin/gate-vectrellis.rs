use std::fmt::Write as _;
use std::time::Instant;

use strand_quant::codebook::{codebook_lut, QUANTILE_SHIFT};
use strand_quant::decode::{decode_lean_with_lut, decode_tensor_fixed_with_lut, decode_tensor_fixed_with_lut_vec};
use strand_quant::encode::{encode_tensor_with_lut, vector_lut_from_scalar, EncodeOpts, EncodedTensor};
use strand_quant::learned_codebook::train_state_vector_lut;
use strand_quant::trellis::TrellisConfig;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    #[inline]
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    #[inline]
    fn next_student_t(&mut self, nu: u32) -> f64 {
        let z = self.next_normal();
        let mut chi2 = 0.0f64;
        for _ in 0..nu {
            let g = self.next_normal();
            chi2 += g * g;
        }
        z / (chi2 / nu as f64).sqrt()
    }
}

#[derive(Clone, Copy)]
enum Dist {
    Gaussian,

    HeavyTailed,
}

impl Dist {
    fn label(self) -> &'static str {
        match self {
            Dist::Gaussian => "gauss",
            Dist::HeavyTailed => "heavy",
        }
    }
}

fn gen_weights(dist: Dist, n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut raw = vec![0.0f64; n];
    match dist {
        Dist::Gaussian => {
            for v in raw.iter_mut() {
                *v = rng.next_normal();
            }
        }
        Dist::HeavyTailed => {
            let block = 256usize;
            let mut bi = 0usize;
            let mut bscale = 1.0f64;
            for (i, v) in raw.iter_mut().enumerate() {
                if i % block == 0 {
                    bscale = (0.5 * rng.next_normal()).exp();
                    bi += 1;
                }
                *v = rng.next_student_t(4) * bscale;
            }
            let _ = bi;
        }
    }

    let mean = raw.iter().sum::<f64>() / n as f64;
    let var = raw.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let inv_sd = if var > 0.0 { 1.0 / var.sqrt() } else { 1.0 };
    raw.iter().map(|&x| ((x - mean) * inv_sd) as f32).collect()
}

fn rel_rms_pct(reference: &[f32], approx: &[f32]) -> f64 {
    let n = reference.len().min(approx.len());
    let mut se = 0.0f64;
    let mut pw = 0.0f64;
    for i in 0..n {
        let r = reference[i] as f64;
        let a = approx[i] as f64;
        let d = r - a;
        se += d * d;
        pw += r * r;
    }
    if pw > 0.0 {
        (se / pw).sqrt() * 100.0
    } else {
        0.0
    }
}

fn decode_vec_f32(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> Vec<f32> {
    let q_to_real = 1.0f32 / (1u32 << QUANTILE_SHIFT) as f32;
    decode_tensor_fixed_with_lut_vec(enc, cfg, lut).into_iter().map(|q| q as f32 * q_to_real).collect()
}

struct Row {
    dist: &'static str,
    d: usize,
    k: u32,
    l: u32,
    payload_bpw: f64,
    total_bpw: f64,
    rel_rms: f64,

    bit_ident: bool,

    lut_ms: u128,

    enc_ms: u128,

    vecs_per_centroid: f64,

    rel_rms_broadcast: f64,
    n: usize,
}

fn measure(dist: Dist, weights: &[f32], d: usize, k: u32, l: u32, seed: u64, iters: usize, adaptive: bool) -> Row {
    let cfg = TrellisConfig::for_bpw_l(k as f64, l).with_vec_dim(d as u32);

    let k = cfg.k_bits;
    let l = cfg.l_bits;
    let d = cfg.vec_dim();

    let t_lut = Instant::now();
    let lut = train_state_vector_lut(weights, l, d, seed, iters);
    let lut_ms = t_lut.elapsed().as_millis();
    debug_assert_eq!(lut.len(), cfg.num_states() * d, "frozen LUT must be [2^L * d]");

    let opts = EncodeOpts { adaptive, ..Default::default() };
    let t_enc = Instant::now();
    let enc = encode_tensor_with_lut(weights, &cfg, &opts, &lut);
    let enc_ms = t_enc.elapsed().as_millis();

    let recon = decode_vec_f32(&enc, &cfg, &lut);
    let rel = rel_rms_pct(weights, &recon);

    let lean = decode_lean_with_lut(&enc, &cfg, &lut);
    let fixed = decode_tensor_fixed_with_lut(&enc, &cfg, &lut);
    let fixed_vec = decode_tensor_fixed_with_lut_vec(&enc, &cfg, &lut);
    let bit_ident = lean == fixed_vec && fixed == fixed_vec;

    let bcast_lut = vector_lut_from_scalar(codebook_lut(l), d);
    let enc_b = encode_tensor_with_lut(weights, &cfg, &opts, &bcast_lut);
    let recon_b = decode_vec_f32(&enc_b, &cfg, &bcast_lut);
    let rel_rms_broadcast = rel_rms_pct(weights, &recon_b);

    let n_vecs = (weights.len() / d) as f64;
    let vecs_per_centroid = n_vecs / cfg.num_states() as f64;

    Row {
        dist: dist.label(),
        d,
        k,
        l,
        payload_bpw: k as f64 / d as f64,
        total_bpw: enc.total_bpw(&cfg),
        rel_rms: rel,
        bit_ident,
        lut_ms,
        enc_ms,
        vecs_per_centroid,
        rel_rms_broadcast,
        n: weights.len(),
    }
}

fn measure_scalar(dist: Dist, weights: &[f32], k: u32, l: u32, seed: u64, iters: usize) -> Row {
    measure(dist, weights, 1, k, l, seed, iters, true)
}

fn main() {
    let t0 = Instant::now();
    let mut report = String::new();

    let n: usize = std::env::var("VT_N").ok().and_then(|s| s.parse().ok()).unwrap_or(262_144);
    let iters: usize = std::env::var("VT_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);

    let lwide: u32 = std::env::var("VT_LWIDE").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let lfix: Option<u32> = std::env::var("VT_LFIX").ok().and_then(|s| s.parse().ok());

    let l_for = |k: u32| -> u32 {
        match lfix {
            Some(l) => l.clamp(k, TrellisConfig::MAX_L),
            None => (k + lwide).min(TrellisConfig::MAX_L),
        }
    };

    let seed: u64 = 0x5654_5452_454C_4C49;
    let data_seed: u64 = 0xD474_0000_5EED_0001;

    let l_policy = match lfix {
        Some(l) => format!("L={l} PINNED (centroids 2^L fixed across the d-sweep)"),
        None => format!("L=k+{lwide} (clamped to {})", TrellisConfig::MAX_L),
    };
    writeln!(report, "GATE B1/C1 — higher-dimensional FROZEN vector trellis: does it close the sub-2-bit gap?").unwrap();
    writeln!(report, "n={n} weights/dist, Lloyd iters={iters}, {l_policy}, learned+frozen Q12 LUT, integer decode.").unwrap();
    writeln!(report, "rel-RMS = weighted (energy) NMSE-sqrt, %. payload_bpw = k/d (physical floor a packed artifact stores).").unwrap();
    writeln!(report, "total_bpw = honest realised (payload + super-scale + 6-bit sub-scales + init_state + len). bit_ident = ").unwrap();
    writeln!(report, "decode_lean_with_lut == decode_tensor_fixed_with_lut_vec for the learned LUT (the determinism contract).\n").unwrap();

    let grid: &[(usize, u32)] = &[(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5), (3, 6), (4, 4), (4, 5), (4, 6), (6, 6), (8, 6)];

    let dists = [Dist::Gaussian, Dist::HeavyTailed];
    let mut rows: Vec<Row> = Vec::new();

    for (di, &dist) in dists.iter().enumerate() {
        let weights = gen_weights(dist, n, data_seed ^ (di as u64).wrapping_mul(0x9E37_79B9));

        for k in [1u32, 2u32] {
            rows.push(measure_scalar(dist, &weights, k, l_for(k), seed, iters));
        }

        for &(d, k) in grid {
            rows.push(measure(dist, &weights, d, k, l_for(k), seed, iters, true));
        }
    }

    writeln!(report, "================ PER-CONFIG (rel-RMS is the gate) ================").unwrap();
    writeln!(report, "vec/cent = training d-vectors per centroid = (n/d)/2^L; ≫1 = well-fit, ≲1 = STARVED book (rel-RMS is then an undertraining artifact).").unwrap();
    writeln!(report, "rel_RMS% = LEARNED frozen codebook (the gate). bcast% = same geometry with the UNLEARNED broadcast-Gaussian LUT (q1). learn-gain = bcast − learned.").unwrap();
    writeln!(
        report,
        "{:>6} {:>9} {:>3} {:>3} {:>3} {:>11} {:>10} {:>9} {:>9} {:>9} {:>10} {:>9} {:>5} {:>7} {:>7}",
        "dist", "n", "d", "k", "L", "payload_bpw", "total_bpw", "rel_RMS%", "bcast%", "learn-g", "floor B/w", "vec/cent", "ident", "lut_ms", "enc_ms"
    )
    .unwrap();
    for r in &rows {
        let floor_bytes = r.payload_bpw / 8.0;
        let learn_gain = r.rel_rms_broadcast - r.rel_rms;
        writeln!(
            report,
            "{:>6} {:>9} {:>3} {:>3} {:>3} {:>11.4} {:>10.4} {:>9.3} {:>9.3} {:>+9.3} {:>10.5} {:>9.2} {:>5} {:>7} {:>7}",
            r.dist,
            r.n,
            r.d,
            r.k,
            r.l,
            r.payload_bpw,
            r.total_bpw,
            r.rel_rms,
            r.rel_rms_broadcast,
            learn_gain,
            floor_bytes,
            r.vecs_per_centroid,
            if r.bit_ident { "OK" } else { "DRIFT" },
            r.lut_ms,
            r.enc_ms
        )
        .unwrap();
    }
    writeln!(report).unwrap();

    writeln!(report, "================ ISO-BPW: rel-RMS (%) vs vector dimension d ================").unwrap();
    writeln!(report, "(lower is better; the question is whether rel-RMS FALLS MATERIALLY as d rises at fixed bpw,").unwrap();
    writeln!(report, " or plateaus. d=1 scalar shown where the bpw is reachable scalar-side.)\n").unwrap();
    let find = |dist: &str, d: usize, k: u32| -> Option<f64> { rows.iter().find(|r| r.dist == dist && r.d == d && r.k == k).map(|r| r.rel_rms) };
    for dist in [Dist::Gaussian, Dist::HeavyTailed] {
        let dl = dist.label();
        writeln!(report, "  [{dl}]").unwrap();

        writeln!(report, "    1.00 bpw : d1={:?}  d2={:?}  d3={:?}  d4={:?}  d6={:?}", find(dl, 1, 1), find(dl, 2, 2), find(dl, 3, 3), find(dl, 4, 4), find(dl, 6, 6)).unwrap();

        writeln!(report, "    1.50 bpw : d2={:?}  d4={:?}", find(dl, 2, 3), find(dl, 4, 6)).unwrap();

        writeln!(report, "    2.00 bpw : d1={:?}  d2={:?}  d3={:?}", find(dl, 1, 2), find(dl, 2, 4), find(dl, 3, 6)).unwrap();

        writeln!(report, "    0.75 bpw : d8={:?}   (sub-1-bit; no scalar analog)", find(dl, 8, 6)).unwrap();
        writeln!(report).unwrap();
    }

    let all_ident = rows.iter().all(|r| r.bit_ident);
    let n_drift = rows.iter().filter(|r| !r.bit_ident).count();
    writeln!(report, "================ DETERMINISM CONTRACT ================").unwrap();
    writeln!(
        report,
        "learned-LUT vector decode bit-identical (lean == fixed_vec == fixed) on {}/{} configs => {}",
        rows.len() - n_drift,
        rows.len(),
        if all_ident { "PASS" } else { "FAIL (REJECT the drifting path)" }
    )
    .unwrap();
    writeln!(report).unwrap();

    let wall = t0.elapsed();
    writeln!(report, "wall time: {:.2?}", wall).unwrap();

    print!("{report}");
    eprintln!("[gate-vectrellis done in {:.2?}]", wall);
}
