//! RWKV-7 GPU decode throughput + flatness measurement (manual; `--ignored`).
//!
//! Not a correctness gate — `rwkv7_metal_parity.rs` owns that. This reports:
//!   * GPU decode tok/s and the GPU-vs-CPU ratio (the ratio is robust to a live
//!     Claude session perturbing absolute tps; absolute is indicative only).
//!   * decode tok/s vs CONTEXT DEPTH (0 / 4k / 16k / 64k): RWKV-7 has no KV
//!     cache, so per-step work is constant and the curve must stay FLAT — the
//!     whole point of the slice.
//!
//! Run (single-threaded so the two models don't contend for the GPU):
//!   DISMANTLE_RWKV7_GGUF=/abs/rwkv7-0.4B-world.Q4_K_M.gguf \
//!   cargo test -p dismantle-core --test rwkv7_metal_bench -- --ignored --nocapture --test-threads=1
//!
//! Depth sweep is capped by `DISMANTLE_RWKV7_MAX_DEPTH` (default 64000); set
//! lower for a quick run.

#![cfg(target_os = "macos")]

use dismantle_core::model::rwkv7::RwkvSeven;
use dismantle_core::{Engine, EngineConfig};
use std::path::PathBuf;
use std::time::Instant;

fn locate(rel: &str, env_key: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env_key) {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let cand = dir.join(rel);
        if cand.exists() {
            return Some(cand);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn load(rel: &str, env_key: &str) -> Option<RwkvSeven> {
    let path = locate(rel, env_key)?;
    let engine = RwkvSeven::load(&path, EngineConfig::default()).expect("load rwkv7");
    if !engine.has_gpu() {
        eprintln!("skip: no Metal GPU");
        return None;
    }
    eprintln!("loaded {path:?}");
    Some(engine)
}

/// Median per-step GPU decode tok/s over `iters` steps after `warmup` steps.
/// Feeds a fixed token id so timing reflects pure per-step cost.
fn time_gpu_decode(engine: &mut RwkvSeven, tok: u32, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        let _ = engine.forward_token_gpu(tok).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = engine.forward_token_gpu(tok).unwrap();
    }
    let secs = t0.elapsed().as_secs_f64();
    iters as f64 / secs
}

fn time_cpu_decode(engine: &mut RwkvSeven, tok: u32, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        let _ = engine.forward_token(tok).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = engine.forward_token(tok).unwrap();
    }
    let secs = t0.elapsed().as_secs_f64();
    iters as f64 / secs
}

fn bench_model(label: &str, rel: &str, env_key: &str) {
    let Some(mut engine) = load(rel, env_key) else {
        eprintln!("== {label}: skipped (no model / no GPU) ==");
        return;
    };
    let tok = 33u32; // arbitrary in-vocab id

    // ── headline GPU tps + GPU/CPU ratio at depth 0 ──
    engine.reset_kv_for_test();
    let gpu_tps = time_gpu_decode(&mut engine, tok, 8, 64);
    engine.reset_kv_for_test();
    let cpu_tps = time_cpu_decode(&mut engine, tok, 2, 32);
    eprintln!(
        "== {label}: GPU {gpu_tps:.1} tok/s | CPU {cpu_tps:.1} tok/s | GPU/CPU = {:.2}x ==",
        gpu_tps / cpu_tps
    );

    // ── flatness vs context depth ──
    let max_depth: usize = std::env::var("DISMANTLE_RWKV7_MAX_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64_000);
    let depths: Vec<usize> = [0usize, 4_000, 16_000, 64_000]
        .into_iter()
        .filter(|&d| d <= max_depth)
        .collect();

    eprintln!("-- {label}: decode tok/s vs context depth (RWKV-7 = no KV cache → expect FLAT) --");
    let mut base = None;
    for &d in &depths {
        engine.reset_kv_for_test();
        // Advance the recurrent state to depth `d` on the GPU path (untimed).
        for _ in 0..d {
            let _ = engine.forward_token_gpu(tok).unwrap();
        }
        // Time a fixed decode window at this depth.
        let tps = time_gpu_decode(&mut engine, tok, 4, 64);
        let base = *base.get_or_insert(tps);
        eprintln!(
            "   depth {d:>6}: {tps:8.1} tok/s   ({:+.1}% vs depth 0)",
            (tps / base - 1.0) * 100.0
        );
    }
}

#[test]
#[ignore = "manual throughput/flatness measurement; run with --ignored"]
fn rwkv7_gpu_decode_tps_and_flatness() {
    bench_model(
        "rwkv7-0.4B",
        "models/rwkv7-04/rwkv7-0.4B-world.Q4_K_M.gguf",
        "DISMANTLE_RWKV7_GGUF",
    );
    bench_model(
        "rwkv7-191M",
        "models/rwkv7-191m/rwkv7-191M-world.Q4_K_M.gguf",
        "DISMANTLE_RWKV7_191M_GGUF",
    );
}
