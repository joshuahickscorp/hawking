#![cfg(target_os = "macos")]
//! Continuous-batching multi-seq DECODE equivalence (build #4 gate).
//!
//! `forward_tokens_multiseq` decodes B INDEPENDENT sequences in one batched GPU
//! pass (one weight read across B columns via the v3w GEMM + the multi-seq MHA +
//! per-slot KV). Two checks:
//!
//!   (1) CONSISTENCY — B sequences decoded TOGETHER give each slot byte-identical
//!       tokens to decoding that sequence ALONE (B=1). This is the serving-
//!       correctness guarantee: no cross-slot contamination, correct per-slot KV.
//!       Uses the SAME LM-head path on both sides (no GPU-vs-CPU argmax mismatch).
//!   (2) ANCHOR — B=1 multi-seq's first token == the proven single-decode path
//!       `forward_token_greedy_tcb`, anchoring the new path to production.
//!
//! Skipped if weights are missing.

use std::path::PathBuf;

use dismantle_core::{model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig};

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

fn load() -> Option<QwenDense> {
    let w = weights_path();
    if !w.exists() {
        eprintln!("skipping multiseq_decode_parity: weights missing at {w:?}");
        return None;
    }
    // Clean env so single + multi-seq both use the default full-vocab predec path.
    for v in [
        "DISMANTLE_QWEN_VOCAB_PRUNE",
        "DISMANTLE_QWEN_Q4K_LMHEAD",
        "DISMANTLE_QWEN_F16_KV",
        "DISMANTLE_QWEN_FLASH_ATTN",
        "DISMANTLE_QWEN_W4A8",
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

/// Decode `n` steps of ONE sequence via the multi-seq path with B=1 (fresh KV).
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

    // (2) ANCHOR: B=1 multi-seq first token == single-stream first token.
    for &s in &seeds {
        engine.kv.reset();
        let single = engine.forward_token_greedy_tcb(s, 0).expect("single fwd");
        engine.multiseq_arena = None;
        let ms = engine.forward_tokens_multiseq(&[s], &[0], max_seq).expect("ms anchor")[0];
        assert_eq!(
            single, ms,
            "anchor: B=1 multiseq token {ms} != single-stream {single} (seed {s})"
        );
    }

    // Solo references: each seed decoded ALONE via the multi-seq path.
    let solo: Vec<Vec<u32>> = seeds.iter().map(|&s| ms_solo(&mut engine, s, n, max_seq)).collect();

    // (1) CONSISTENCY: all B seeds decoded TOGETHER, lockstep positions.
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
    println!("[multiseq-decode] anchored + B={b} batched==solo over {n} steps: {batched:?}");
}
