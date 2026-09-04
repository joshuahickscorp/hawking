//! CP6c — what actually binds the chunked organ at 30% of roof?
//!
//! `CP6B_RESULT.md` ruled out one candidate by experiment: the fused+bitcast
//! multi-position kernel bought 0.35%, so the int->float unpack is not what
//! binds it. Batching had already amortised that -- 16 FMAs per unpacked weight
//! at r4k4 against 1 at a single position.
//!
//! Three candidates are left, and at fixed R they move together, which is why
//! this sweeps R and K independently on the REAL session:
//!
//!   register pressure   2*R*K accumulators live per thread
//!   reduction cost      R*K simd_sums into an 8*R*K threadgroup array
//!   occupancy           ceil(17408 / 2R) threadgroups against 60 GPU cores
//!
//! The discriminating pairs:
//!
//!   r2k4 vs r4k2   SAME R*K -- same accumulators, same reduction -- but 4352
//!                  threadgroups against 2176. Any difference is occupancy.
//!   r4k4 vs r2k4   double the accumulators AND half the threadgroups. If r4k4
//!                  wins, occupancy at 2176 TGs is not the constraint.
//!   r1k1           the degenerate control: one position, no batching, so its
//!                  per-position cost should land near the production baseline.
//!
//! Real catalog weights, all 64 layers, one command buffer per arm, alternated,
//! first rep discarded. Same instrument as CP6a/CP6b.
//!
//! ```text
//! ./tools/gpu_lane_lock.sh cp6c \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_cp6c_what_binds_the_chunk \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A --reps 9 \
//!   --out receipts/runtime/CP6C_WHAT_BINDS.json
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

    const ROWS: usize = 17_408;
    const CORES: usize = 60;
    // gate + up, codes + scales + biases, 64 layers.
    const WEIGHT_BYTES: f64 = 64.0 * 2.0 * (22_282_240.0 + 2_785_280.0 + 2_785_280.0);
    const ROOF_GBS: f64 = 778.8;

    fn fail(m: impl std::fmt::Display) -> ! {
        eprintln!("cp6c: {m}");
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
        let mut it = env::args().skip(1);
        while let Some(f) = it.next() {
            match f.as_str() {
                "--artifact-root" => artifact_root = PathBuf::from(it.next().unwrap_or_default()),
                "--out" => out = it.next().map(PathBuf::from),
                "--reps" => reps = it.next().unwrap_or_default().parse().unwrap_or(9),
                o => fail(format!("unknown flag {o}")),
            }
        }

        let mut session =
            Qwen38HybridDecodeSession::open(&artifact_root, 2048).unwrap_or_else(|e| fail(e));
        session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
        let noop = session
            .measure_isolated_organ("noop_empty")
            .unwrap_or_else(|e| fail(e));

        // Only cells the shader instantiates for the fused bitcast family.
        let grid: &[(usize, usize)] = &[(1, 1), (2, 2), (2, 4), (4, 2), (4, 4)];
        let mut prod: Vec<u64> = Vec::new();
        let mut cells: Vec<(usize, usize, Vec<u64>)> =
            grid.iter().map(|&(r, k)| (r, k, Vec::new())).collect();

        for rep in 0..(reps + 1) {
            let p = session
                .measure_isolated_organ("mlp_gate_up")
                .unwrap_or_else(|e| fail(e));
            if rep > 0 {
                prod.push(p.gpu_ns.unwrap_or_else(|| fail("no GPU timestamp")));
            }
            for (r, k, acc) in cells.iter_mut() {
                let t = session
                    .measure_isolated_organ_chunked_rk("mlp_gate_up_swiglu", *r, *k)
                    .unwrap_or_else(|e| fail(e));
                if rep > 0 {
                    acc.push(t.gpu_ns.unwrap_or_else(|| fail("no GPU timestamp")));
                }
            }
        }

        let pm = median(prod.clone()) as f64;
        eprintln!("production swiglu-fused baseline: {pm:.0} ns for 1 position, 64 layers");
        eprintln!(
            "{:>3}{:>3} {:>8} {:>6} {:>13} {:>9} {:>8} {:>8}",
            "r", "k", "TGs", "acc", "ns/position", "speedup", "GB/s", "%roof"
        );
        let mut rows_json = Vec::new();
        for (r, k, acc) in &cells {
            let m = median(acc.clone()) as f64;
            let per_pos = m / *k as f64;
            let tgs = ROWS.div_ceil(2 * r);
            let gbs = WEIGHT_BYTES / (m / 1e9) / 1e9;
            eprintln!(
                "{r:>3}{k:>3} {tgs:>8} {:>6} {per_pos:>13.0} {:>9.3} {gbs:>8.1} {:>7.1}%",
                2 * r * k,
                pm / per_pos,
                100.0 * gbs / ROOF_GBS
            );
            rows_json.push(json!({
                "r": r, "k": k,
                "threadgroups": tgs,
                "threadgroups_per_core": tgs as f64 / CORES as f64,
                "accumulators_per_thread": 2 * r * k,
                "threadgroup_floats": 8 * r * k,
                "simd_sums_per_thread": r * k,
                "gpu_ns_reps": acc,
                "gpu_ns_median": m as u64,
                "ns_per_position": per_pos,
                "speedup_vs_production": pm / per_pos,
                "achieved_gb_s": gbs,
                "pct_of_roof": 100.0 * gbs / ROOF_GBS,
            }));
        }

        let doc = json!({
            "schema": "hawking.cp6c.what_binds_the_chunk.v1",
            "question": "CP6b ruled out unpack cost. Of register pressure, reduction cost and \
occupancy, which binds the chunked gate_up at 30% of roof?",
            "discriminators": {
                "r2k4_vs_r4k2": "same R*K, so identical accumulators and identical reduction; \
4352 threadgroups against 2176. Any difference is OCCUPANCY.",
                "r4k4_vs_r2k4": "double the accumulators AND half the threadgroups. If r4k4 wins, \
2176 threadgroups is not starving the grid.",
                "r1k1": "degenerate control -- one position, no batching; its per-position cost \
should land near the production baseline."
            },
            "organ": "mlp_gate_up", "rows": ROWS, "gpu_cores": CORES,
            "weight_bytes_64_layers": WEIGHT_BYTES,
            "roof_gb_s": ROOF_GBS,
            "noop_control_gpu_ns": noop.gpu_ns,
            "reps": reps, "warmup_discarded": 1,
            "production_gpu_ns_reps": prod,
            "production_gpu_ns_median": pm as u64,
            "cells": rows_json,
        });
        match out {
            Some(p) => {
                if let Some(d) = p.parent() { fs::create_dir_all(d).ok(); }
                fs::write(&p, serde_json::to_string_pretty(&doc).unwrap() + "\n")
                    .unwrap_or_else(|e| fail(e));
                eprintln!("wrote {}", p.display());
            }
            None => println!("{}", serde_json::to_string_pretty(&doc).unwrap()),
        }
    }
}
