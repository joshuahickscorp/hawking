#![cfg(target_os = "macos")]
use hawking_core::{
    model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig,
};
use std::time::Instant;
mod common;
use common::weights_path_qwen as weights_path;
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
    std::env::remove_var("HAWKING_QWEN_Q4K_LMHEAD");
    let off = time_configs("default (flag-OFF: per-slot CPU f16 LM head; reflects R2+R3)");
    std::env::set_var("HAWKING_QWEN_Q4K_LMHEAD", "1");
    let on = time_configs("R1 ON (HAWKING_QWEN_Q4K_LMHEAD=1: GPU-batched Q4_K LM head)");
    for (idx, b) in [1usize, 4, 8].iter().enumerate() {
        let r = if off[idx] > 0.0 {
            on[idx] / off[idx]
        } else {
            0.0
        };
    }
}
