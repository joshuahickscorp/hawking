//! CP6i -- the second row of the CP6d table.
//!
//! CP6d measured the prompt wall the sequential way, one `step()` per prompt
//! token: 41.67 tok/s at 128, 708 dispatches per position. G019's acceptance is
//! capability equivalence AND a lower complete prompt wall against that row.
//! Everything since has been organs, kernels and layers. This is the wall.
//!
//! Both arms run in ONE session, `reset()` between them, so only one 10.5 GB
//! resident is ever live (directive SS XXVII). Arms alternate and the first pair
//! is discarded for the DVFS ramp.
//!
//! The equivalence check is the sampled token after the prompt. If the chunked
//! path is faster but predicts something else, it has not done the same work and
//! the speed is worthless -- so the token is compared before any ratio is
//! reported, and a mismatch is a FAIL regardless of timing.
//!
//! ```text
//! ./tools/gpu_lane_lock.sh cp6i \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_cp6i_chunked_prompt_wall \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A --tokens 128 --chunk 4 --reps 3 \
//!   --out receipts/runtime/CP6I_CHUNKED_PROMPT_WALL.json
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
    use std::time::Instant;

    /// A pinned, deterministic prompt. Content does not matter to the physics --
    /// what matters is that both arms see the SAME token ids.
    fn prompt(n: usize) -> Vec<u32> {
        let mut s = 0xA5C2_E326u64;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) % 100_000) as u32 + 10
            })
            .collect()
    }

    fn stats(v: &mut Vec<f64>) -> (f64, f64, f64) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[0], v[v.len() / 2], v[v.len() - 1])
    }

    pub fn run() {
        let mut root: Option<PathBuf> = None;
        let mut out: Option<PathBuf> = None;
        let mut n = 128usize;
        let mut chunk = 4usize;
        let mut reps = 3usize;
        let mut r = 4usize;
        let mut it = env::args().skip(1);
        while let Some(f) = it.next() {
            match f.as_str() {
                "--artifact-root" => root = it.next().map(PathBuf::from),
                "--out" => out = it.next().map(PathBuf::from),
                "--tokens" => n = it.next().unwrap_or_default().parse().unwrap_or(128),
                "--chunk" => chunk = it.next().unwrap_or_default().parse().unwrap_or(4),
                "--reps" => reps = it.next().unwrap_or_default().parse().unwrap_or(3),
                "--r" => r = it.next().unwrap_or_default().parse().unwrap_or(4),
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
        if n % chunk != 0 {
            eprintln!("--tokens must be a multiple of --chunk");
            process::exit(2);
        }

        let mut session = match Qwen38HybridDecodeSession::open(&root, 2048) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("session open failed: {e}");
                process::exit(1);
            }
        };
        session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
        let toks = prompt(n);

        let mut seq_ms: Vec<f64> = Vec::new();
        let mut chk_ms: Vec<f64> = Vec::new();
        let mut seq_tok: Option<u32> = None;
        let mut chk_tok: Option<u32> = None;

        // ORDER IS ALTERNATED. Running arm A first in every rep would hand arm
        // B a warmer GPU, and the sequential arm is the longer one, so that bias
        // points the wrong way -- toward the result being claimed. On even reps
        // A precedes B, on odd reps B precedes A.
        let run_seq = |session: &mut Qwen38HybridDecodeSession, toks: &[u32]| -> (f64, u32) {
            session.reset();
            let t0 = Instant::now();
            let mut last = 0u32;
            for &t in toks {
                match session.step_unmeasured(t) {
                    Ok(s) => last = s,
                    Err(e) => {
                        eprintln!("sequential step failed: {e}");
                        process::exit(1);
                    }
                }
            }
            (t0.elapsed().as_secs_f64() * 1000.0, last)
        };
        let run_chunked = |session: &mut Qwen38HybridDecodeSession, toks: &[u32]| -> (f64, u32) {
            session.reset();
            let t0 = Instant::now();
            let mut last = 0u32;
            for c in toks.chunks(chunk) {
                match session.prefill_chunk(c, r) {
                    Ok((s, _)) => last = s,
                    Err(e) => {
                        eprintln!("chunked prefill failed: {e}");
                        process::exit(1);
                    }
                }
            }
            (t0.elapsed().as_secs_f64() * 1000.0, last)
        };

        for rep in 0..=reps {
            let (a, la, b, lb) = if rep % 2 == 0 {
                let (a, la) = run_seq(&mut session, &toks);
                let (b, lb) = run_chunked(&mut session, &toks);
                (a, la, b, lb)
            } else {
                let (b, lb) = run_chunked(&mut session, &toks);
                let (a, la) = run_seq(&mut session, &toks);
                (a, la, b, lb)
            };
            if rep > 0 {
                seq_ms.push(a);
                chk_ms.push(b);
            }
            seq_tok = Some(la);
            chk_tok = Some(lb);
        }

        let equivalent = seq_tok == chk_tok;
        println!("  prompt {n} tokens, chunk {chunk}, r{r}");
        println!(
            "  sampled token   sequential {:?}   chunked {:?}   {}",
            seq_tok,
            chk_tok,
            if equivalent { "EQUIVALENT" } else { "DIVERGED" }
        );

        let (alo, amd, ahi) = stats(&mut seq_ms);
        let (blo, bmd, bhi) = stats(&mut chk_ms);
        let tps_a = n as f64 / (amd / 1000.0);
        let tps_b = n as f64 / (bmd / 1000.0);
        let sp = amd / bmd;
        let sp_lo = alo / bhi;
        println!("  sequential  {amd:>9.1} ms  [{alo:.1}-{ahi:.1}]   {tps_a:>6.2} tok/s");
        println!("  chunked     {bmd:>9.1} ms  [{blo:.1}-{bhi:.1}]   {tps_b:>6.2} tok/s");
        println!("  speedup {sp:.3}x   conservative {sp_lo:.3}x");

        // Acceptance is BOTH. A faster arm that predicts a different token has
        // not done the same work.
        let pass = equivalent && sp_lo > 1.0;
        println!(
            "  ACCEPTANCE: {}",
            if pass {
                "PASS -- equivalent AND lower wall"
            } else if !equivalent {
                "FAIL -- not capability-equivalent"
            } else {
                "FAIL -- wall not separated"
            }
        );

        if let Some(p) = out {
            let doc = json!({
                "checkpoint": "CP6I_CHUNKED_PROMPT_WALL",
                "artifact_root": root.display().to_string(),
                "tokens": n, "chunk": chunk, "r": r, "reps_kept": seq_ms.len(),
                "sequential_ms": {"min": alo, "median": amd, "max": ahi},
                "chunked_ms": {"min": blo, "median": bmd, "max": bhi},
                "sequential_tok_s": tps_a,
                "chunked_tok_s": tps_b,
                "speedup_median": sp,
                "speedup_conservative": sp_lo,
                "sampled_sequential": seq_tok,
                "sampled_chunked": chk_tok,
                "capability_equivalent": equivalent,
                "acceptance_pass": pass,
            });
            if let Some(d) = p.parent() {
                let _ = fs::create_dir_all(d);
            }
            let _ = fs::write(&p, serde_json::to_string_pretty(&doc).unwrap());
            println!("wrote {}", p.display());
        }
        if !pass {
            process::exit(1);
        }
    }
}
