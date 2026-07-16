use std::path::{Path, PathBuf};

use strand_decode_kernel::block_walk::gate_proto::machine_stamp;
use strand_decode_kernel::event_mac::{salient_threshold, DeltaMac, EventMac};
use strand_decode_kernel::loader::StrandModel;
use strand_decode_kernel::outlier_mac::patched_weights;
use strand_quant::encode::encode_tensor;
use strand_quant::TrellisConfig;

fn rel_l2(y: &[f32], y_ref: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in y.iter().zip(y_ref.iter()) {
        let d = (*a as f64) - (*b as f64);
        num += d * d;
        den += (*b as f64) * (*b as f64);
    }
    (num / den.max(1e-300)).sqrt()
}

fn quantile(vals: &mut [f32], q: f64) -> f32 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if vals.is_empty() {
        return 0.0;
    }
    let pos = q * (vals.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(vals.len() - 1);
    let frac = (pos - lo as f64) as f32;
    vals[lo] * (1.0 - frac) + vals[hi] * frac
}

fn identity_gates() {
    let configs = [("3-bit deploy", TrellisConfig::for_bpw(3.0)), ("2-bit reopen", TrellisConfig::for_bpw_l(2.0, 12)), ("fold path", TrellisConfig::for_bpw_l(2.0, 5))];
    let shapes = [(8usize, 256usize), (37, 300), (5, 97)];
    let mut cells = 0usize;
    for (label, cfg) in &configs {
        for &(rows, cols) in &shapes {
            let w: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.0137).sin() * 0.5).collect();
            let enc = encode_tensor(&w, cfg);
            let q12 = strand_decode_kernel::decode_weights_q12(&enc, cfg, None);
            let em = EventMac::from_q12(&q12, rows, cols);
            let x: Vec<f32> = (0..cols)
                .map(|i| {
                    let c = ((i as f32 + 1.0) * 0.0713).cos();
                    c * c * c
                })
                .collect();

            let y_ref = strand_decode_kernel::matvec(&enc, cfg, None, rows, cols, &x);
            let y_full = em.matvec_full(&x);
            for o in 0..rows {
                assert_eq!(y_full[o].to_bits(), y_ref[o].to_bits(), "IDENTITY FAIL: full-set y[{o}] != crate::matvec ({label} {rows}x{cols})");
            }

            let (y_t0, rep) = em.matvec_threshold(&x, 0.0);
            assert_eq!(rep.fired, cols);
            for o in 0..rows {
                assert_eq!(y_t0[o].to_bits(), y_ref[o].to_bits(), "IDENTITY FAIL: tau=0");
            }

            for tau in [0.05f32, 0.3] {
                let (y, _) = em.matvec_threshold(&x, tau);
                let wq: Vec<f32> = q12.iter().map(|&q| (q as f32) * (1.0 / 4096.0)).collect();
                for o in 0..rows {
                    let mut acc = 0.0f32;
                    for i in 0..cols {
                        if x[i].abs() >= tau {
                            acc += wq[o * cols + i] * x[i];
                        }
                    }
                    assert_eq!(y[o].to_bits(), acc.to_bits(), "IDENTITY FAIL: partial S not exact ({label} tau={tau} row {o})");
                }
            }

            let mut dm = DeltaMac::new(&em);
            for t in 0..3 {
                let xt: Vec<f32> = (0..cols).map(|i| ((i as f32 + t as f32 * 7.7) * 0.0511).sin()).collect();
                let (y, rep) = dm.step(&xt, 0.123, 1);
                assert!(rep.refreshed);
                let yf = em.matvec_full(&xt);
                for o in 0..rows {
                    assert_eq!(y[o].to_bits(), yf[o].to_bits(), "IDENTITY FAIL: delta refresh");
                }
            }
            cells += 1;
        }
    }
    println!(
        "identity gates: {cells}/{cells} cells green — full-set bit-equal to crate::matvec, \
         partial-S exact, delta refresh_every=1 bit-equal ✓"
    );
}

struct ManifestEntry {
    tensor: String,
    tokens: usize,
    dim: usize,
    path: PathBuf,
}

fn read_manifest(acts_dir: &Path) -> Option<Vec<ManifestEntry>> {
    let text = std::fs::read_to_string(acts_dir.join("manifest.tsv")).ok()?;
    let mut out = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 4 {
            continue;
        }
        out.push(ManifestEntry { tensor: f[0].to_string(), tokens: f[1].parse().ok()?, dim: f[2].parse().ok()?, path: acts_dir.join(f[3]) });
    }
    Some(out)
}

fn read_f32_le(path: &Path, n: usize) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() != n * 4 {
        eprintln!("  !! {path:?}: {} bytes, expected {}", bytes.len(), n * 4);
        return None;
    }
    Some(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[derive(Default, Clone)]
struct Agg {
    sums: Vec<(f64, f64)>,
    n: usize,
}

#[allow(clippy::too_many_lines)]
fn real_curves(artifact: &Path, acts_dir: &Path, max_tensors: usize, max_tokens: usize) {
    let model = match StrandModel::open(artifact) {
        Ok(m) => m,
        Err(e) => {
            println!(
                "\nREAL-DATA SECTION SKIPPED: cannot open artifact {artifact:?}: {e}\n\
                 (identity gates above are the deliverable; re-run with --artifact)"
            );
            return;
        }
    };
    let Some(manifest) = read_manifest(acts_dir) else {
        println!(
            "\nREAL-DATA SECTION SKIPPED: no manifest at {:?}\n\
             (generate with: /usr/local/bin/python3 scripts/event-sparsity-probe.py)",
            acts_dir.join("manifest.tsv")
        );
        return;
    };

    let keeps = [0.50f64, 0.25, 0.10, 0.05, 0.02];

    let tau_ps = [0.50f64, 0.75, 0.90];

    let dtau_ps = [0.50f64, 0.75, 0.90];

    let mut agg_topk = Agg { sums: vec![(0.0, 0.0); keeps.len()], n: 0 };
    let mut agg_tau = Agg { sums: vec![(0.0, 0.0); tau_ps.len()], n: 0 };
    let mut agg_delta = Agg { sums: vec![(0.0, 0.0); dtau_ps.len()], n: 0 };
    let mut used = 0usize;

    println!(
        "\n== real-data error-vs-skip curves (ADVISORY; per-layer output distortion, \
         NOT PPL) ==\nartifact: {artifact:?}\nacts:     {acts_dir:?}"
    );

    for entry in &manifest {
        if used >= max_tensors {
            break;
        }
        let Some(hdr) = model.tensor_header(&entry.tensor) else {
            continue;
        };
        let (out_f, in_f) = (hdr.shape[0] as usize, hdr.shape[1] as usize);
        if in_f != entry.dim {
            eprintln!("  !! {}: artifact in_features {in_f} != activation dim {} — skipped", entry.tensor, entry.dim);
            continue;
        }
        let w = match patched_weights(&model, &entry.tensor) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("  !! {}: patched_weights failed: {e}", entry.tensor);
                continue;
            }
        };
        let Some(acts) = read_f32_le(&entry.path, entry.tokens * entry.dim) else {
            continue;
        };
        let em = EventMac::from_dense(&w, out_f, in_f);
        let tokens = entry.tokens.min(max_tokens);
        used += 1;

        let y_full: Vec<Vec<f32>> = (0..tokens).map(|t| em.matvec_full(&acts[t * in_f..(t + 1) * in_f])).collect();

        print!("\n{} [{}x{}] tokens={tokens}\n  top-k : ", entry.tensor, out_f, in_f);
        for (ki, &keep) in keeps.iter().enumerate() {
            let k = ((in_f as f64 * keep).round() as usize).max(1);
            let mut err = 0f64;
            let mut skip = 0f64;
            for (t, yf) in y_full.iter().enumerate() {
                let x = &acts[t * in_f..(t + 1) * in_f];
                let (y, rep) = em.matvec_topk(x, k);
                err += rel_l2(&y, yf);
                skip += rep.skip_fraction();
            }
            err /= tokens as f64;
            skip /= tokens as f64;
            agg_topk.sums[ki].0 += skip;
            agg_topk.sums[ki].1 += err;
            print!("keep {:>4.0}% → skip {:>4.1}% err {:.4}  ", keep * 100.0, skip * 100.0, err);
        }
        agg_topk.n += 1;
        println!();

        print!("  tau   : ");
        for (pi, &p) in tau_ps.iter().enumerate() {
            let mut err = 0f64;
            let mut skip = 0f64;
            for (t, yf) in y_full.iter().enumerate() {
                let x = &acts[t * in_f..(t + 1) * in_f];
                let mut mags: Vec<f32> = x.iter().map(|v| v.abs()).collect();
                let tau = quantile(&mut mags, p);
                let (y, rep) = em.matvec_threshold(x, tau);
                err += rel_l2(&y, yf);
                skip += rep.skip_fraction();
            }
            err /= tokens as f64;
            skip /= tokens as f64;
            agg_tau.sums[pi].0 += skip;
            agg_tau.sums[pi].1 += err;
            print!("|x|p{:.0} → skip {:>4.1}% err {:.4}  ", p * 100.0, skip * 100.0, err);
        }
        agg_tau.n += 1;
        println!();

        let mut dmags: Vec<f32> = Vec::with_capacity((tokens - 1) * in_f);
        for t in 1..tokens {
            let a = &acts[(t - 1) * in_f..t * in_f];
            let b = &acts[t * in_f..(t + 1) * in_f];
            dmags.extend(a.iter().zip(b.iter()).map(|(p0, p1)| (p1 - p0).abs()));
        }
        print!("  delta : ");
        for (di, &p) in dtau_ps.iter().enumerate() {
            let mut pool = dmags.clone();
            let tau_d = quantile(&mut pool, p);
            for &refresh in &[usize::MAX, 8usize] {
                let mut dm = DeltaMac::new(&em);
                let mut fired = 0f64;
                let mut steps = 0usize;
                let mut err_sum = 0f64;
                let mut err_max = 0f64;
                for (t, yf) in y_full.iter().enumerate() {
                    let x = &acts[t * in_f..(t + 1) * in_f];
                    let (y, rep) = dm.step(x, tau_d, refresh);
                    if !rep.refreshed {
                        fired += rep.fired as f64 / rep.total as f64;
                        steps += 1;
                        let e = rel_l2(y, yf);
                        err_sum += e;
                        err_max = err_max.max(e);
                    } else if t > 0 {
                        for (a, b) in y.iter().zip(yf.iter()) {
                            assert_eq!(a.to_bits(), b.to_bits(), "refresh not bit-exact");
                        }
                    }
                }
                if steps == 0 {
                    continue;
                }
                let fired = fired / steps as f64;
                let err = err_sum / steps as f64;
                if refresh == usize::MAX {
                    agg_delta.sums[di].0 += 1.0 - fired;
                    agg_delta.sums[di].1 += err;
                    print!("|dx|p{:.0} → fire {:>4.1}% err {:.4} (max {:.4})  ", p * 100.0, fired * 100.0, err, err_max);
                } else {
                    print!("[refresh8: err {err:.4}]  ");
                }
            }
        }
        agg_delta.n += 1;
        println!();
    }

    if used == 0 {
        println!("\nNo manifest tensor matched the artifact — nothing measured.");
        return;
    }

    println!("\n== AGGREGATE over {used} tensors (mean of per-tensor means; ADVISORY) ==");
    print!("  top-k : ");
    for (ki, &keep) in keeps.iter().enumerate() {
        print!("keep {:>4.0}% → skip {:>4.1}% err {:.4}  ", keep * 100.0, agg_topk.sums[ki].0 / agg_topk.n as f64 * 100.0, agg_topk.sums[ki].1 / agg_topk.n as f64);
    }
    print!("\n  tau   : ");
    for (pi, &p) in tau_ps.iter().enumerate() {
        print!("|x|p{:.0} → skip {:>4.1}% err {:.4}  ", p * 100.0, agg_tau.sums[pi].0 / agg_tau.n as f64 * 100.0, agg_tau.sums[pi].1 / agg_tau.n as f64);
    }
    print!("\n  delta : ");
    for (di, &p) in dtau_ps.iter().enumerate() {
        print!("|dx|p{:.0} → skip {:>4.1}% err {:.4}  ", p * 100.0, agg_delta.sums[di].0 / agg_delta.n as f64 * 100.0, agg_delta.sums[di].1 / agg_delta.n as f64);
    }
    println!(
        "\n\nNOTE: 'skip' here = fraction of input columns (MAC flops AND column bytes on \
         the prepared substrate) not touched. These are LAYER-LOCAL distortion curves — \
         end-to-end quality (PPL) is the science-queue protocol in \
         research/event-sparsity.md, with its kill bars. No PPL claim is made here."
    );

    if let Some(e0) = manifest.first() {
        if let Some(acts) = read_f32_le(&e0.path, e0.tokens * e0.dim) {
            let x = &acts[..e0.dim];
            let mut mags: Vec<f32> = x.iter().map(|v| v.abs()).collect();
            let tau = quantile(&mut mags, 0.5);
            let f = salient_threshold(x, tau).len() as f64 / e0.dim as f64;
            assert!((0.4..=0.6).contains(&f), "quantile/threshold disagree: p50 tau fired {f:.2} of dims");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str, default: &str| -> String { args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned()).unwrap_or_else(|| default.to_string()) };
    let artifact = PathBuf::from(get("--artifact", "scratch/artifacts/qwen05b-pv2-2bit.strand"));
    let acts_dir = PathBuf::from(get("--acts", "research/event-sparsity-probe"));
    let max_tensors: usize = get("--max-tensors", "12").parse().expect("--max-tensors");
    let max_tokens: usize = get("--tokens", "32").parse().expect("--tokens");

    println!("gate-eventmac — event-driven firing: identity gates, then error-vs-skip curves");
    println!("{}", machine_stamp());
    println!("ALL NUMBERS ADVISORY (contended box protocol); identity gates are the deliverable.\n");

    identity_gates();
    real_curves(&artifact, &acts_dir, max_tensors, max_tokens);
}
