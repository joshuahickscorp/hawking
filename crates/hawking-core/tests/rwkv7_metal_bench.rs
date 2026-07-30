#![cfg(target_os = "macos")]
use hawking_core::model::rwkv7::RwkvSeven;
use hawking_core::{Engine, EngineConfig};
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
    Some(engine)
}
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
    engine.reset_kv_for_test();
    let gpu_tps = time_gpu_decode(&mut engine, tok, 8, 64);
    engine.reset_kv_for_test();
    let cpu_tps = time_cpu_decode(&mut engine, tok, 2, 32);
    let max_depth: usize = std::env::var("HAWKING_RWKV7_MAX_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64_000);
    let depths: Vec<usize> = [0usize, 4_000, 16_000, 64_000]
        .into_iter()
        .filter(|&d| d <= max_depth)
        .collect();
    let mut base = None;
    for &d in &depths {
        engine.reset_kv_for_test();
        for _ in 0..d {
            let _ = engine.forward_token_gpu(tok).unwrap();
        }
        let tps = time_gpu_decode(&mut engine, tok, 4, 64);
        let base = *base.get_or_insert(tps);
    }
}
#[test]
#[ignore = "manual throughput/flatness measurement; run with --ignored"]
fn rwkv7_gpu_decode_tps_and_flatness() {
    bench_model(
        "rwkv7-0.4B",
        "models/rwkv7-04/rwkv7-0.4B-world.Q4_K_M.gguf",
        "HAWKING_RWKV7_GGUF",
    );
    bench_model(
        "rwkv7-191M",
        "models/rwkv7-191m/rwkv7-191M-world.Q4_K_M.gguf",
        "HAWKING_RWKV7_191M_GGUF",
    );
}
