#![cfg(target_os = "macos")]
use hawking_core::{
    model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig,
};
mod common;
use common::argmax;
use common::weights_path_qwen as weights_path;
fn load_q4k_lmhead() -> Option<QwenDense> {
    let w = weights_path();
    if !w.exists() {
        eprintln!("skipping multiseq_q4k_lmhead_parity: weights missing at {w:?}");
        return None;
    }
    for v in [
        "HAWKING_QWEN_VOCAB_PRUNE",
        "HAWKING_QWEN_VOCAB_PRUNE_CORPUS",
        "HAWKING_QWEN_F16_KV",
        "HAWKING_QWEN_FLASH_ATTN",
        "HAWKING_QWEN_W4A8",
    ] {
        std::env::remove_var(v);
    }
    std::env::set_var("HAWKING_QWEN_Q4K_LMHEAD", "1"); // env_on => exact "1"; full-vocab Q4_K head
    let profile = fresh_test_profile(&w).expect("fresh test profile");
    let cfg = EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    Some(QwenDense::load(&w, cfg).expect("load qwen-3b (q4k lm-head)"))
}
fn solo_tokens(engine: &mut QwenDense, seed: u32, n: usize, max_seq: usize) -> Vec<u32> {
    engine.multiseq_arena = None;
    let mut cur = seed;
    let mut out = Vec::with_capacity(n);
    for pos in 0..n {
        let next = engine
            .forward_tokens_multiseq(&[cur], &[pos], max_seq)
            .expect("solo q4k lm-head");
        out.push(next[0]);
        cur = next[0];
    }
    out
}
#[test]
#[ignore] // loads 1.93 GB model + sets a process-global env var; run explicitly, single-threaded
fn multiseq_q4k_lmhead_batched_equals_solo() {
    let mut engine = match load_q4k_lmhead() {
        Some(e) => e,
        None => return, // weights missing — clean skip (no failure)
    };
    let seeds: Vec<u32> = vec![9707, 374, 100, 151643];
    let b = seeds.len();
    let n = 4usize;
    let max_seq = 16usize;
    assert!(b <= 8, "B must be <= MAX_MULTISEQ_SLOTS (8)");
    let solo: Vec<Vec<u32>> = seeds
        .iter()
        .map(|&s| solo_tokens(&mut engine, s, n, max_seq))
        .collect();
    engine.multiseq_arena = None;
    let mut cur = seeds.clone();
    let mut batched: Vec<Vec<u32>> = vec![Vec::new(); b];
    for pos in 0..n {
        let positions = vec![pos; b];
        let next = engine
            .forward_tokens_multiseq(&cur, &positions, max_seq)
            .expect("batched q4k lm-head");
        for bi in 0..b {
            batched[bi].push(next[bi]);
        }
        cur = next;
    }
    // GATE: per-slot argmax token byte-identical (batched column == solo).
    for bi in 0..b {
        assert_eq!(
            solo[bi], batched[bi],
            "slot {bi}: batched Q4_K LM-head argmax {:?} != solo {:?} (seed {})",
            batched[bi], solo[bi], seeds[bi]
        );
    }
}
#[test]
#[ignore]
fn multiseq_q4k_lmhead_divergent_positions_equal_solo() {
    let mut engine = match load_q4k_lmhead() {
        Some(e) => e,
        None => return,
    };
    let max_seq = 2048usize; // accommodate the long-context slot
    let seeds = [9707u32, 374, 100, 151643];
    let start_pos = [2047usize, 1024, 512, 100]; // divergent positions
    let b = seeds.len();
    let mut solo: Vec<u32> = Vec::with_capacity(b);
    for bi in 0..b {
        engine.multiseq_arena = None;
        let logits = engine
            .forward_tokens_multiseq_logits(&[seeds[bi]], &[start_pos[bi]], &[0], max_seq)
            .expect("solo divergent logits");
        assert_eq!(
            logits[0].len(),
            engine.config.vocab_size,
            "solo logits must be full-vocab"
        );
        solo.push(argmax(&logits[0]));
    }
    engine.multiseq_arena = None;
    let regions: Vec<usize> = (0..b).collect();
    let logits = engine
        .forward_tokens_multiseq_logits(&seeds, &start_pos, &regions, max_seq)
        .expect("batched divergent logits");
    for l in &logits {
        assert_eq!(
            l.len(),
            engine.config.vocab_size,
            "batched logits must be full-vocab"
        );
    }
    let batched: Vec<u32> = logits.iter().map(|l| argmax(l)).collect();
    for bi in 0..b {
        assert_eq!(
            solo[bi], batched[bi],
            "slot {bi} @pos {}: batched Q4_K LM-head argmax {} != solo {}",
            start_pos[bi], batched[bi], solo[bi]
        );
    }
}
