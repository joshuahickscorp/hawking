#![cfg(target_os = "macos")]
use hawking_core::{
    model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig,
};
mod common;
use common::weights_path_qwen as weights_path;
fn load() -> Option<QwenDense> {
    let w = weights_path();
    if !w.exists() {
        eprintln!("skipping multiseq_decode_parity: weights missing at {w:?}");
        return None;
    }
    for v in [
        "HAWKING_QWEN_VOCAB_PRUNE",
        "HAWKING_QWEN_Q4K_LMHEAD",
        "HAWKING_QWEN_F16_KV",
        "HAWKING_QWEN_FLASH_ATTN",
        "HAWKING_QWEN_W4A8",
    ] {
        std::env::remove_var(v);
    }
    let profile = fresh_test_profile(&w).expect("fresh test profile");
    let cfg = EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    Some(QwenDense::load(&w, cfg).expect("load qwen-3b"))
}
fn ms_solo(engine: &mut QwenDense, seed: u32, n: usize, max_seq: usize) -> Vec<u32> {
    engine.multiseq_arena = None;
    let mut cur = seed;
    let mut seq = Vec::with_capacity(n);
    for pos in 0..n {
        let next = engine
            .forward_tokens_multiseq(&[cur], &[pos], max_seq)
            .expect("ms solo");
        seq.push(next[0]);
        cur = next[0];
    }
    seq
}
#[test]
fn multiseq_batched_equals_solo_and_anchors_single() {
    let mut engine = match load() {
        Some(e) => e,
        None => return,
    };
    let seeds: Vec<u32> = vec![9707, 374, 100];
    let b = seeds.len();
    let n = 4usize;
    let max_seq = 16usize;
    for &s in &seeds {
        engine.kv.reset();
        let single = engine.forward_token_greedy_tcb(s, 0).expect("single fwd");
        engine.multiseq_arena = None;
        let ms = engine
            .forward_tokens_multiseq(&[s], &[0], max_seq)
            .expect("ms anchor")[0];
        assert_eq!(
            single, ms,
            "anchor: B=1 multiseq token {ms} != single-stream {single} (seed {s})"
        );
    }
    let solo: Vec<Vec<u32>> = seeds
        .iter()
        .map(|&s| ms_solo(&mut engine, s, n, max_seq))
        .collect();
    engine.multiseq_arena = None;
    let mut cur = seeds.clone();
    let mut batched: Vec<Vec<u32>> = vec![Vec::new(); b];
    for pos in 0..n {
        let positions = vec![pos; b];
        let next = engine
            .forward_tokens_multiseq(&cur, &positions, max_seq)
            .expect("ms batched");
        for bi in 0..b {
            batched[bi].push(next[bi]);
        }
        cur = next;
    }
    for bi in 0..b {
        assert_eq!(
            solo[bi], batched[bi],
            "slot {bi}: batched {:?} != solo {:?}",
            batched[bi], solo[bi]
        );
    }
}
