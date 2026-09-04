//! CP6h -- prefill pays the LM head at every position and throws it away.
//!
//! `encode_full_token` is embed -> layers -> terminal, and `terminal` is the
//! final norm plus a **full-vocab matvec** plus argmax. Prefill drives one
//! `step()` per prompt token, so every prefill position computes logits over the
//! entire vocabulary and every one but the last is discarded.
//!
//! That is orthogonal to chunking. If it is a large share of the prefill wall it
//! is a cheaper win than batching anything, and it is measured here BEFORE any
//! more chunking work, per the frontier rule: recompute the bottleneck after
//! every result rather than continuing down the path already started.
//!
//! Reports the lm_head organ against the whole-token wall, both from the same
//! session on the same quiet lane, alternated, first rep dropped.
//!
//! ```text
//! ./tools/gpu_lane_lock.sh cp6h \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_cp6h_lm_head_waste \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A --reps 9 \
//!   --out receipts/runtime/CP6H_LM_HEAD_WASTE.json
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

    fn stats(v: &mut Vec<u64>) -> (f64, f64, f64) {
        v.sort_unstable();
        (v[0] as f64, v[v.len() / 2] as f64, v[v.len() - 1] as f64)
    }

    pub fn run() {
        let mut root: Option<PathBuf> = None;
        let mut out: Option<PathBuf> = None;
        let mut reps = 9usize;
        let mut it = env::args().skip(1);
        while let Some(f) = it.next() {
            match f.as_str() {
                "--artifact-root" => root = it.next().map(PathBuf::from),
                "--out" => out = it.next().map(PathBuf::from),
                "--reps" => reps = it.next().unwrap_or_default().parse().unwrap_or(9),
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

        match session.measure_isolated_organ("noop_empty") {
            Ok(t) => println!("noop control gpu_ns = {:?}", t.gpu_ns),
            Err(e) => println!("noop control unavailable: {e}"),
        }

        let mut head = Vec::new();
        let mut whole = Vec::new();
        for rep in 0..reps {
            let h = session
                .measure_isolated_organ("lm_head")
                .and_then(|t| {
                    t.gpu_ns
                        .ok_or_else(|| hawking_core::Error::Model("no gpu_ns on lm_head".into()))
                });
            // The whole-token wall through the ordinary production path.
            let w = session.step(1).map(|(_, t)| t.gpu_ns);
            match (h, w) {
                (Ok(hn), Ok(Some(wn))) => {
                    if rep > 0 {
                        head.push(hn);
                        whole.push(wn);
                    }
                }
                (Err(e), _) => {
                    eprintln!("lm_head organ failed: {e}");
                    process::exit(1);
                }
                (_, Ok(None)) => {
                    eprintln!("step returned no GPU timestamp");
                    process::exit(1);
                }
                (_, Err(e)) => {
                    eprintln!("step failed: {e}");
                    process::exit(1);
                }
            }
        }

        let (hlo, hmd, hhi) = stats(&mut head);
        let (wlo, wmd, whi) = stats(&mut whole);
        // The conservative reading of the waste: smallest head over largest wall.
        let frac_lo = hlo / whi;
        let frac_md = hmd / wmd;
        let frac_hi = hhi / wlo;
        println!("  lm_head  ns  min {hlo:.0}  med {hmd:.0}  max {hhi:.0}");
        println!("  token    ns  min {wlo:.0}  med {wmd:.0}  max {whi:.0}");
        println!(
            "  lm_head share of the token wall: {:.1}%  [{:.1}% - {:.1}%]",
            frac_md * 100.0,
            frac_lo * 100.0,
            frac_hi * 100.0
        );
        // What removing it from all but the last position would buy, as a bound.
        println!(
            "  a prefill that computed logits ONCE would be at most {:.3}x faster",
            1.0 / (1.0 - frac_md)
        );
        println!("  (upper bound: it assumes the rest of the token wall is unchanged)");

        if let Some(p) = out {
            let doc = json!({
                "checkpoint": "CP6H_LM_HEAD_WASTE",
                "artifact_root": root.display().to_string(),
                "lm_head_ns": {"min": hlo, "median": hmd, "max": hhi},
                "token_wall_ns": {"min": wlo, "median": wmd, "max": whi},
                "lm_head_share": {"conservative": frac_lo, "median": frac_md, "generous": frac_hi},
                "upper_bound_speedup_if_removed": 1.0 / (1.0 - frac_md),
                "reps_kept": head.len(),
                "note": "encode_full_token always runs encode_terminal; prefill drives one step() \
                         per prompt token, so every position computes full-vocab logits and all \
                         but the last are discarded",
            });
            if let Some(d) = p.parent() {
                let _ = fs::create_dir_all(d);
            }
            let _ = fs::write(&p, serde_json::to_string_pretty(&doc).unwrap());
            println!("wrote {}", p.display());
        }
    }
}
