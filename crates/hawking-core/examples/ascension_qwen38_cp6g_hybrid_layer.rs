//! CP6g -- does a HYBRID layer win, with the glue paid?
//!
//! The mixer is genuinely per-position: DeltaNet carries a recurrent state and
//! GQA appends KV, so k+1 depends on k. Batching it means threading per-position
//! addressing through many encoders -- wide, mechanical, and not done.
//!
//! The MLP has no such dependency, and it is the larger half of the step. So the
//! question this asks is whether the chunked prefill has to wait for the mixer
//! at all: run the mixer K times exactly as it runs today, scatter each result
//! into slot k of a K-wide buffer, run the MLP ONCE, gather back.
//!
//! The glue is not free and is not assumed away. Arm B pays 2K extra dispatches
//! -- a scatter per position and a gather per position -- to save K-1 MLPs. If
//! the trade is bad, this reports it as a loss rather than hiding it.
//!
//! Both arms alternate on the same quiet lane, first rep discarded for the DVFS
//! ramp (CP3b law). GPU time is GPUEndTime - GPUStartTime on the completed
//! command buffer, never a CPU wait.
//!
//! ```text
//! ./tools/gpu_lane_lock.sh cp6g \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_cp6g_hybrid_layer \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A --reps 9 \
//!   --out receipts/runtime/CP6G_HYBRID_LAYER.json
//! ```

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run()
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::qwen38_hybrid_decode::{
        Qwen38HybridDecodeSession, Qwen38MlpFusion,
    };
    use serde_json::json;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::process;

    /// min / median / max. A single median hides the thing that actually
    /// decides whether a ratio means anything: how much the ARM ITSELF moved.
    /// The first run of this harness reported the sequential baseline 36% apart
    /// between two arms of the same measurement, which is enough to manufacture
    /// a speedup out of nothing.
    fn spread(v: &mut [u64]) -> (f64, f64, f64) {
        v.sort_unstable();
        (
            v[0] as f64,
            v[v.len() / 2] as f64,
            v[v.len() - 1] as f64,
        )
    }

    pub fn run() {
        let mut root: Option<PathBuf> = None;
        let mut out: Option<PathBuf> = None;
        let mut reps = 9usize;
        let mut layers: Vec<usize> = (0..64).collect();
        let mut it = env::args().skip(1);
        while let Some(f) = it.next() {
            match f.as_str() {
                "--artifact-root" => root = it.next().map(PathBuf::from),
                "--out" => out = it.next().map(PathBuf::from),
                "--reps" => reps = it.next().unwrap_or_default().parse().unwrap_or(9),
                "--layers" => {
                    layers = it
                        .next()
                        .unwrap_or_default()
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect()
                }
                o => {
                    eprintln!("unknown flag {o}");
                    process::exit(2);
                }
            }
        }
        let root = root.unwrap_or_else(|| {
            eprintln!("--artifact-root is required");
            process::exit(2);
        });

        let mut session = match Qwen38HybridDecodeSession::open(&root, 2048) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("session open failed: {e}");
                process::exit(1);
            }
        };
        session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);

        // The noop control: if this is not near zero the instrument is lying.
        match session.measure_isolated_organ("noop_empty") {
            Ok(t) => println!("noop control gpu_ns = {:?}", t.gpu_ns),
            Err(e) => println!("noop control unavailable: {e}"),
        }

        let arms: &[(usize, usize)] = &[(4, 2), (4, 4)];
        let mut rows = Vec::new();
        // The span, not one layer. See measure_hybrid_layers: a single ~400us
        // command buffer sits below this instrument's noise floor and separates
        // nothing. All 64 layers is what CP6a/b/c used to reach 0.7%.
        let span: Vec<usize> = layers.clone();
        println!("  layers  r  k   seq ns/pos  hyb ns/pos   speedup  [conservative-generous]");
        {
            let layer = span.len();
            for &(r, k) in arms {
                let mut s = Vec::with_capacity(reps);
                let mut h = Vec::with_capacity(reps);
                let mut sd = 0usize;
                let mut hd = 0usize;
                let mut err: Option<String> = None;
                for rep in 0..reps {
                    match session.measure_hybrid_layers(&span, r, k) {
                        Ok((sn, hn, a, b)) => {
                            // First rep is the DVFS ramp, not a measurement.
                            if rep > 0 {
                                s.push(sn);
                                h.push(hn);
                            }
                            sd = a;
                            hd = b;
                        }
                        Err(e) => {
                            err = Some(e.to_string());
                            break;
                        }
                    }
                }
                if let Some(e) = err {
                    println!("  {layer:>5}  {r:>2} {k:>2}    ERROR {e}");
                    rows.push(json!({"layer": layer, "r": r, "k": k, "error": e}));
                    continue;
                }
                let kf = (k * span.len()) as f64;
                let s_ord = s.clone();
                let (slo, sm, shi) = spread(&mut s);
                let (hlo, hm, hhi) = spread(&mut h);
                let (slo, sm, shi) = (slo / kf, sm / kf, shi / kf);
                let (hlo, hm, hhi) = (hlo / kf, hm / kf, hhi / kf);
                let sp = sm / hm;
                // The honest bound: slowest sequential over fastest hybrid is
                // the most generous reading, fastest over slowest the least.
                // If those two straddle 1.0 the arms are not separated.
                let sp_hi = shi / hlo;
                let sp_lo = slo / hhi;
                let sep = sp_lo > 1.0;
                let sjit = (shi - slo) / sm;
                // Is the jitter RANDOM or MONOTONE? Both arms append to the KV
                // cache and advance the recurrent state on every rep, so the
                // work per rep can GROW -- which is drift, not noise, and a
                // median over a moving quantity is not a measurement. Compare
                // the first half of the reps against the second, in ORDER.
                let half = s_ord.len() / 2;
                let fh: f64 = s_ord[..half].iter().map(|&x| x as f64).sum::<f64>() / half as f64;
                let sh: f64 = s_ord[half..].iter().map(|&x| x as f64).sum::<f64>()
                    / (s_ord.len() - half) as f64;
                let drift = sh / fh;
                println!(
                    "  {layer:>5}  {r:>2} {k:>2}   {sm:>10.0}  {hm:>10.0}   {sp:>6.3}x  [{sp_lo:.3}-{sp_hi:.3}]  seq jitter {:>5.1}%  {sd:>3} -> {hd}  {}",
                    sjit * 100.0,
                    if sep {
                        "SEPARATED".to_string()
                    } else {
                        format!("not separated; seq drift x{drift:.2}")
                    }
                );
                rows.push(json!({
                    "layers": span.len(), "r": r, "k": k,
                    "seq_ns_per_position": {"min": slo, "median": sm, "max": shi},
                    "hybrid_ns_per_position": {"min": hlo, "median": hm, "max": hhi},
                    "speedup_median": sp,
                    "speedup_conservative": sp_lo,
                    "speedup_generous": sp_hi,
                    "arms_separated": sep,
                    "sequential_jitter_frac": sjit,
                    "sequential_drift_second_half_over_first": drift,
                    "sequential_series_ns": s_ord,
                    "dispatches_sequential": sd,
                    "dispatches_hybrid": hd,
                    "reps_kept": s.len(),
                }));
            }
        }

        if let Some(p) = out {
            let doc = json!({
                "checkpoint": "CP6G_HYBRID_LAYER",
                "artifact_root": root.display().to_string(),
                "note": "mixer sequential, MLP batched; 2K glue dispatches paid inside arm B",
                "rows": rows,
            });
            if let Some(d) = p.parent() {
                let _ = fs::create_dir_all(d);
            }
            let _ = fs::write(&p, serde_json::to_string_pretty(&doc).unwrap());
            println!("wrote {}", p.display());
        }
    }
}
