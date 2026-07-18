#![cfg(target_os = "macos")]
//! Continuous-batching AGGREGATE-tps demonstration (build #6; R1/R2/R3 aware).
//!
//! Times `forward_tokens_multiseq` at B = 1 / 4 / 8 in lockstep and prints
//! aggregate tokens/sec + speedup vs B=1, for TWO configs:
//!   - default (flag-OFF): per-slot CPU f16 full-vocab LM head. Reflects R2
//!     (batched RoPE) + R3 (batched KV-append), which are UNCONDITIONAL, vs the
//!     historical pre-R2/R3 2.57x baseline.
//!   - R1 ON (HAWKING_QWEN_Q4K_LMHEAD=1): GPU-batched Q4_K LM head — one
//!     weight-amortizing GEMM over B columns instead of B sequential CPU matmuls.
//!
//! The SPEEDUP RATIO and the flag-ON/OFF DELTA are contamination-robust (the
//! ~4-5x agent inflation cancels), so they are valid with the agent open. The
//! ABSOLUTE tps needs a clean room. R2+R3 are baked into BOTH configs; to isolate
//! their delta, A/B this bench vs commit 8aba79e (R1-only, pre-R2/R3).
//!
//! `#[ignore]` so it never runs (or flakes) the normal suite. Invoke explicitly:
//!   cargo test --release -p hawking-core --test multiseq_aggregate_bench -- --ignored --nocapture
//! (or via tools/bench/batch_aggregate_bench.sh, which the clean_bench_queue picks up).

use std::path::PathBuf;
use std::time::Instant;

use hawking_core::{
    model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig,
};

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

/// Load fresh and time B=1/4/8 under whatever env is currently set. Returns the
/// per-B aggregate tps ([B1, B4, B8]). The Q4_K LM-head buffer is built at LOAD
/// time from the env, so the caller sets HAWKING_QWEN_Q4K_LMHEAD before calling.
fn time_configs(label: &str) -> [f64; 3] {
    let w = weights_path();
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
    let mut out = [0.0f64; 3];

    println!("\n[multiseq-aggregate] {label}");
    println!("  B | per-step ms | aggregate tps | speedup");
    for (idx, &bsz) in [1usize, 4, 8].iter().enumerate() {
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
        out[idx] = agg_tps;
    }
    out
}

#[test]
#[ignore]
fn multiseq_aggregate_speedup() {
    let w = weights_path();
    if !w.exists() {
        eprintln!("skipping multiseq_aggregate_bench: weights missing at {w:?}");
        return;
    }
    for v in ["HAWKING_QWEN_VOCAB_PRUNE", "HAWKING_QWEN_F16_KV"] {
        std::env::remove_var(v);
    }

    // Config A — default (flag-OFF): per-slot CPU f16 full-vocab LM head. This run
    // reflects R2 (batched RoPE) + R3 (batched KV-append) vs the pre-R2/R3 2.57x.
    std::env::remove_var("HAWKING_QWEN_Q4K_LMHEAD");
    let off = time_configs("default (flag-OFF: per-slot CPU f16 LM head; reflects R2+R3)");

    // Config B — R1 ON: GPU-batched Q4_K LM head (one GEMM over B columns).
    std::env::set_var("HAWKING_QWEN_Q4K_LMHEAD", "1");
    let on = time_configs("R1 ON (HAWKING_QWEN_Q4K_LMHEAD=1: GPU-batched Q4_K LM head)");

    // R1 delta (flag-ON vs flag-OFF, SAME binary): contamination-robust AND, in a
    // clean room, the absolute aggregate. R2+R3 are baked into both sides.
    println!("\n[multiseq-aggregate] R1 delta (flag-ON / flag-OFF, contamination-robust):");
    for (idx, b) in [1usize, 4, 8].iter().enumerate() {
        let r = if off[idx] > 0.0 {
            on[idx] / off[idx]
        } else {
            0.0
        };
        println!(
            "  B={b}: {:.2} -> {:.2} aggregate tps  (x{r:.3})",
            off[idx], on[idx]
        );
    }
    println!("(ratio/delta cancel the agent's ~4-5x inflation; ABSOLUTE tps needs a clean room.)");
    println!("(batch_ceiling.py predicts ~3.5-5.6x realistic aggregate @ B=8.)");
    println!(
        "(R2+R3 are unconditional; to isolate their delta, A/B this bench vs commit 8aba79e.)\n"
    );
}
