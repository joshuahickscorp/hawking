//! CP6a — the multi-position gate_up, on the REAL session's REAL weights, all 64 layers.
//!
//! Everything before this measured buffers this crate filled. CP3, CP4, CP4b,
//! CP5a/b/c all ran standalone harnesses at real SHAPES with deterministic or
//! artifact-sliced bytes. `CP5C_RESULT.md` lists "real artifact weight bytes at
//! these shapes" as untested, and this is the test: the weights come from the
//! session's own catalog via `affine()`, every one of the 64 layers is encoded,
//! and both arms sit in one command buffer exactly as `measure_isolated_organ`
//! does for the production number.
//!
//! FUSION-MATCHED, which is the trap this avoids. Production fuses SwiGLU into
//! gate_up; the multi-position kernel is the non-fused pair. Dividing by the
//! FUSED baseline would credit batching with a fusion difference, so the
//! baseline here is `measure_isolated_organ_pair_baseline`, which encodes the
//! same organ with `with_swiglu = false`.
//!
//! Opens ONE catalog. Does not load a second 27B. Does not mutate the artifact.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core \
//!   --example ascension_qwen38_cp6a_real_session_chunked_organ
//! ./tools/gpu_lane_lock.sh cp6a \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_cp6a_real_session_chunked_organ \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A --reps 9 \
//!   --out receipts/runtime/CP6A_REAL_SESSION_ORGAN.json
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

    fn fail(m: impl std::fmt::Display) -> ! {
        eprintln!("cp6a: {m}");
        process::exit(2);
    }

    fn median(mut v: Vec<u64>) -> u64 {
        v.sort_unstable();
        v[v.len() / 2]
    }

    pub fn run() {
        let mut artifact_root = PathBuf::from(env::var("HOME").unwrap_or_default())
            .join("noetic")
            .join("NOETIC_PARENT_A");
        let mut out: Option<PathBuf> = None;
        let mut reps = 9usize;
        let mut chunk = 4usize;
        let mut max_seq_len = 2048usize;
        let mut it = env::args().skip(1);
        while let Some(f) = it.next() {
            match f.as_str() {
                "--artifact-root" => artifact_root = PathBuf::from(it.next().unwrap_or_default()),
                "--out" => out = it.next().map(PathBuf::from),
                "--reps" => reps = it.next().unwrap_or_default().parse().unwrap_or(9),
                "--chunk" => chunk = it.next().unwrap_or_default().parse().unwrap_or(4),
                "--max-seq-len" => {
                    max_seq_len = it.next().unwrap_or_default().parse().unwrap_or(2048)
                }
                o => fail(format!("unknown flag {o}")),
            }
        }

        let t0 = Instant::now();
        let mut session = Qwen38HybridDecodeSession::open(&artifact_root, max_seq_len)
            .unwrap_or_else(|e| fail(e));
        let open_s = t0.elapsed().as_secs_f64();
        eprintln!("session open {open_s:.3}s");
        session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);

        // A no-op command buffer, so the instrument is shown not to be measuring
        // nothing. The organ census carries the same control at 125 ns.
        // What the catalog actually resolved to, before any timing is trusted.
        eprint!("{}", session.probe_gate_up_geometry(0));

        let noop = session
            .measure_isolated_organ("noop_empty")
            .unwrap_or_else(|e| fail(e));
        eprintln!("noop control gpu_ns = {:?}", noop.gpu_ns);

        let mut seq: Vec<u64> = Vec::new();
        let mut bat: Vec<u64> = Vec::new();
        let mut fused: Vec<u64> = Vec::new();
        let mut chunked_fused: Vec<u64> = Vec::new();
        for rep in 0..(reps + 1) {
            // Alternated, so a clock ramp hits both arms. CP3b established that
            // the first cells of any sweep sit on a DVFS ramp and that the ratio
            // survives it only because the arms are adjacent.
            let a = session
                .measure_isolated_organ_pair_baseline("mlp_gate_up")
                .unwrap_or_else(|e| fail(e));
            let b = session
                .measure_isolated_organ_chunked("mlp_gate_up", chunk)
                .unwrap_or_else(|e| fail(e));
            let f = session
                .measure_isolated_organ("mlp_gate_up")
                .unwrap_or_else(|e| fail(e));
            // Batching AND fusion AND bitcast together -- the kernel CP6a said
            // was the lever, measured against the same fused production arm.
            let cf = session
                .measure_isolated_organ_chunked("mlp_gate_up_swiglu", chunk)
                .unwrap_or_else(|e| fail(e));
            if rep == 0 {
                continue;
            }
            match (a.gpu_ns, b.gpu_ns, f.gpu_ns, cf.gpu_ns) {
                (Some(x), Some(y), Some(z), Some(w)) => {
                    seq.push(x);
                    bat.push(y);
                    fused.push(z);
                    chunked_fused.push(w);
                }
                _ => fail("driver gave no GPU timestamp"),
            }
        }

        let sm = median(seq.clone()) as f64;
        let bm = median(bat.clone()) as f64;
        let fm = median(fused.clone()) as f64;
        let cfm = median(chunked_fused.clone()) as f64;
        // The baseline runs ONE position over 64 layers; the chunked arm runs
        // `chunk` positions over the same 64 layers in one dispatch each.
        let speedup = (sm * chunk as f64) / bm;
        eprintln!("pair baseline  (1 position, 64 layers) median {sm:.0} ns");
        eprintln!("chunked r4k{chunk} ({chunk} positions, 64 layers) median {bm:.0} ns");
        eprintln!("swiglu-fused production baseline        median {fm:.0} ns");
        eprintln!("SPEEDUP vs the fusion-matched pair baseline: {speedup:.3}x");
        eprintln!(
            "  (against the FUSED production baseline it would read {:.3}x -- not claimed, \
different fusion)",
            (fm * chunk as f64) / bm
        );
        eprintln!("chunked FUSED+bitcast r4k{chunk}                median {cfm:.0} ns");
        eprintln!(
            "  vs the fused production path, LIKE FOR LIKE: {:.3}x",
            (fm * chunk as f64) / cfm
        );

        let doc = json!({
            "schema": "hawking.cp6a.real_session_chunked_organ.v1",
            "organ": "mlp_gate_up",
            "what_is_real_here": "the session's own catalog weights via affine(), all 64 layers, \
one command buffer per arm -- the same instrument measure_isolated_organ uses for the production \
organ number. Prior CP receipts measured real SHAPES with buffers this crate filled.",
            "artifact_root": artifact_root.to_string_lossy(),
            "max_seq_len": max_seq_len,
            "chunk": chunk,
            "reps": reps,
            "warmup_discarded": 1,
            "arms_alternated": true,
            "noop_control_gpu_ns": noop.gpu_ns,
            "gpu_timestamp_authority": "TokenCommandBuffer::commit_and_wait_timed, GPUEndTime-GPUStartTime on the completed command buffer",
            "fusion_matched_baseline": "measure_isolated_organ_pair_baseline encodes gate_up with \
with_swiglu=false, matching the multi-position kernel. Dividing by the SwiGLU-fused production \
baseline would credit batching with a fusion difference.",
            "pair_baseline_gpu_ns_reps": seq,
            "chunked_gpu_ns_reps": bat,
            "fused_production_gpu_ns_reps": fused,
            "pair_baseline_gpu_ns_median": sm as u64,
            "chunked_gpu_ns_median": bm as u64,
            "fused_production_gpu_ns_median": fm as u64,
            "speedup_x_vs_pair": speedup,
            "speedup_x_vs_fused_NOT_CLAIMED": (fm * chunk as f64) / bm,
            "chunked_fused_bitcast_gpu_ns_reps": chunked_fused,
            "chunked_fused_bitcast_gpu_ns_median": cfm as u64,
            "speedup_x_fused_vs_chunked_fused_LIKE_FOR_LIKE": (fm * chunk as f64) / cfm,
            "session_open_s": open_s,
        });
        match out {
            Some(p) => {
                if let Some(d) = p.parent() {
                    fs::create_dir_all(d).ok();
                }
                fs::write(&p, serde_json::to_string_pretty(&doc).unwrap() + "\n")
                    .unwrap_or_else(|e| fail(e));
                eprintln!("wrote {}", p.display());
            }
            None => println!("{}", serde_json::to_string_pretty(&doc).unwrap()),
        }
    }
}
