//! CP6f -- does the chunked MLP compute the same thing?
//!
//! CP6a/b/c measured the chunked organ on the real session and were **timing
//! only**. Correctness rested on CP4's bit-identical result for the gate_up
//! kernel in ISOLATION. That is a weaker claim than it sounds: an organ can be
//! individually correct and still be wrong once composed, because composition is
//! where layout assumptions meet.
//!
//! This runs the real three-dispatch MLP -- fused gate_up+swiglu, down_proj,
//! add-residual+norm -- for K positions in one pass on real catalog weights, and
//! compares against K runs of the production `encode_dense_mlp`.
//!
//! The r1k1 row is the control, and the FIRST run corrected what it controls
//! for. The bar was originally "r1k1 must be exactly zero, because at one
//! position the layouts are byte-identical". That was wrong. Production runs the
//! MATVEC family (`..._matvec_gate_up_swiglu_...`); the chunked path at K=1 runs
//! `..._matmul_..._r1k1...`. Those are different kernels with different
//! rows-per-thread accumulation, so r1k1 is NOT expected to be exact.
//!
//! What r1k1 actually isolates is better than what the original bar tested. It
//! separates two costs that would otherwise be confounded:
//!
//!   r1k1        the cost of the matvec -> matmul FAMILY change alone
//!   r4k2/r4k4   that, PLUS whatever widening K adds
//!
//! So the gate is not an absolute epsilon, it is r1k1 itself: a wider arm must
//! not exceed the family-change floor. That directly asks the question the
//! chunked path has to answer -- does batching K cost accuracy? -- instead of
//! asking whether some hand-picked constant was generous enough.
//!
//! ```text
//! ./tools/gpu_lane_lock.sh cp6f \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_cp6f_chunked_mlp_correct \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --out receipts/runtime/CP6F_CHUNKED_MLP.json
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

    pub fn run() {
        let mut root: Option<PathBuf> = None;
        let mut out: Option<PathBuf> = None;
        let mut layers: Vec<usize> = vec![0, 31, 63];
        let mut it = env::args().skip(1);
        while let Some(f) = it.next() {
            match f.as_str() {
                "--artifact-root" => root = it.next().map(PathBuf::from),
                "--out" => out = it.next().map(PathBuf::from),
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
        // The production configuration, matching CP6a/b/c.
        session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);

        // (r, k). r1k1 runs first and BECOMES the bar -- see the module note.
        let arms: &[(usize, usize)] = &[(1, 1), (4, 2), (4, 4)];
        // Slack over the family-change floor. 2x, not 1x, because the floor is
        // itself a max over 5120*K fp32 dots and one extra ulp of jitter between
        // arms is not evidence that batching costs accuracy. Anything genuinely
        // caused by widening K would move this by orders, not by a factor.
        const SLACK: f64 = 2.0;
        let mut rows = Vec::new();
        let mut failures = 0usize;

        println!("  layer   r  k    max_abs_resid    max_abs_norm   disp seq -> chunked   bar");
        for &layer in &layers {
            let mut floor: Option<f64> = None;
            for &(r, k) in arms {
                match session.verify_chunked_dense_mlp(layer, r, k) {
                    Ok((mr, mn, ds, dc)) => {
                        // The head norm is bit-identical by construction (CP6e),
                        // so it is gated at exactly zero in every arm. Any drift
                        // there is a real defect, not reassociation.
                        let norm_ok = mn == 0.0;
                        let (bar, resid_ok) = match floor {
                            None => {
                                floor = Some(mr);
                                // The control has no bar of its own; it only has
                                // to be small enough to be fp32 noise at all.
                                (f64::NAN, mr < 1e-3)
                            }
                            Some(f) => {
                                let b = (f * SLACK).max(1e-9);
                                (b, mr <= b)
                            }
                        };
                        let ok = norm_ok && resid_ok;
                        if !ok {
                            failures += 1;
                        }
                        println!(
                            "  {layer:>5}  {r:>2} {k:>2}    {mr:>12.4e}    {mn:>12.4e}   {ds:>3} -> {dc:<3}   {}  {}",
                            if bar.is_nan() {
                                "floor".to_string()
                            } else {
                                format!("{bar:.2e}")
                            },
                            if ok { "PASS" } else { "FAIL" }
                        );
                        rows.push(json!({
                            "layer": layer, "r": r, "k": k,
                            "max_abs_resid": mr, "max_abs_norm": mn,
                            "bar": if bar.is_nan() { serde_json::Value::Null } else { json!(bar) },
                            "is_family_change_floor": bar.is_nan(),
                            "pass": ok,
                            "dispatches_sequential": ds,
                            "dispatches_chunked": dc,
                        }));
                    }
                    Err(e) => {
                        failures += 1;
                        println!("  {layer:>5}  {r:>2} {k:>2}    ERROR {e}");
                        rows.push(json!({
                            "layer": layer, "r": r, "k": k, "error": e.to_string(),
                        }));
                    }
                }
            }
        }

        if let Some(p) = out {
            let doc = json!({
                "checkpoint": "CP6F_CHUNKED_MLP_CORRECTNESS",
                "artifact_root": root.display().to_string(),
                "rows": rows,
                "failures": failures,
            });
            if let Some(d) = p.parent() {
                let _ = fs::create_dir_all(d);
            }
            let _ = fs::write(&p, serde_json::to_string_pretty(&doc).unwrap());
            println!("wrote {}", p.display());
        }
        if failures > 0 {
            eprintln!("{failures} arm(s) FAILED");
            process::exit(1);
        }
        println!("all arms PASS");
    }
}
