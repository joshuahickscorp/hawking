#![cfg(target_os = "macos")]
//! Continuous-batching AGGREGATE-tps demonstration (build #6).
//!
//! Times `forward_tokens_multiseq` at B = 1 / 4 / 8 in lockstep and prints
//! aggregate tokens/sec + speedup vs B=1. The SPEEDUP RATIO is contamination-
//! robust (the ~4-5x Claude inflation cancels in the ratio), so it is valid with
//! Claude open. The ABSOLUTE tps needs a clean room.
//!
//! `#[ignore]` so it never runs (or flakes) the normal suite. Invoke explicitly:
//!   cargo test --release -p dismantle-core --test multiseq_aggregate_bench -- --ignored --nocapture
//! (or via tools/bench/batch_aggregate_bench.sh, which the clean_bench_queue picks up).

use std::path::PathBuf;
use std::time::Instant;

use dismantle_core::{model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig};

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

#[test]
#[ignore]
fn multiseq_aggregate_speedup() {
    let w = weights_path();
    if !w.exists() {
        eprintln!("skipping multiseq_aggregate_bench: weights missing at {w:?}");
        return;
    }
    for v in [
        "DISMANTLE_QWEN_VOCAB_PRUNE",
        "DISMANTLE_QWEN_Q4K_LMHEAD",
        "DISMANTLE_QWEN_F16_KV",
    ] {
        std::env::remove_var(v);
    }
    let profile = fresh_test_profile(&w).expect("fresh test profile");
    let cfg = EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    let mut engine = QwenDense::load(&w, cfg).expect("load qwen-3b");

    let max_seq = 64usize;
    let n_steps = 24usize;
    let warmup = 4usize;
    let mut base_tps = 0.0f64;

    println!(
        "\n[multiseq-aggregate] B | per-step ms | aggregate tps | speedup  \
         (ratio is contamination-robust; absolute needs a clean room)"
    );
    for &bsz in &[1usize, 4, 8] {
        engine.multiseq_arena = None;
        let mut cur: Vec<u32> = (0..bsz).map(|i| 100 + i as u32 * 50).collect();
        for pos in 0..warmup {
            let positions = vec![pos; bsz];
            cur = engine
                .forward_tokens_multiseq(&cur, &positions, max_seq)
                .expect("warmup step");
        }
        let t0 = Instant::now();
        for step in 0..n_steps {
            let positions = vec![warmup + step; bsz];
            cur = engine
                .forward_tokens_multiseq(&cur, &positions, max_seq)
                .expect("timed step");
        }
        let dt = t0.elapsed().as_secs_f64();
        let per_step_ms = dt / n_steps as f64 * 1000.0;
        let agg_tps = (bsz * n_steps) as f64 / dt;
        if bsz == 1 {
            base_tps = agg_tps;
        }
        let speedup = agg_tps / base_tps;
        println!("  {bsz:>2} | {per_step_ms:>10.2} | {agg_tps:>12.2} | {speedup:.2}x");
    }
    println!("(batch_ceiling.py predicts ~3.5-5.6x realistic aggregate @ B=8)\n");
}
