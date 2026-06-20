#![cfg(target_os = "macos")]
//! Parity gate for the greedy token-only multiseq serving lane (Phase A).
//!
//! `forward_tokens_multiseq_greedy` must return the SAME token ids as
//! `forward_tokens_multiseq_logits` → CPU argmax for every (B, seed) pair.
//!
//! Two checks:
//!   (1) Token ids from the GPU argmax path == token ids from the CPU argmax
//!       path, for B = 1, 2, 4, 8, over 4 decode steps each.
//!   (2) The Engine trait dispatch (`forward_multiseq_greedy_tokens`) returns
//!       the same ids as direct QwenDense calls, confirming the seam is wired.
//!
//! Skipped if weights are absent.

use std::path::PathBuf;

use hawking_core::{
    model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig,
};

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

fn load() -> Option<QwenDense> {
    let w = weights_path();
    if !w.exists() {
        eprintln!("skipping greedy_token_only_parity: weights missing at {w:?}");
        return None;
    }
    // Enable the Q4K LM head so the GPU argmax path is active.
    std::env::set_var("HAWKING_QWEN_Q4K_LMHEAD", "1");
    for v in [
        "HAWKING_QWEN_VOCAB_PRUNE",
        "HAWKING_QWEN_F16_KV",
        "HAWKING_QWEN_W4A8",
    ] {
        std::env::remove_var(v);
    }
    let profile = fresh_test_profile(&w).expect("fresh test profile");
    let cfg = EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    Some(QwenDense::load(&w, cfg).expect("load"))
}

/// Run n decode steps via the FULL logits path → CPU argmax.
fn logits_path(engine: &mut QwenDense, seeds: &[u32], n: usize, max_seq: usize) -> Vec<Vec<u32>> {
    let b = seeds.len();
    // one sequence per slot, stable region = slot index
    let regions: Vec<usize> = (0..b).collect();
    let mut cur = seeds.to_vec();
    let mut seqs = vec![Vec::with_capacity(n); b];
    // Fresh arena so both paths start from the same state.
    engine.multiseq_arena = None;
    for step in 0..n {
        let positions: Vec<usize> = vec![step; b];
        let logits = engine
            .forward_tokens_multiseq_logits(&cur, &positions, &regions, max_seq)
            .expect("logits path");
        let tokens: Vec<u32> = logits
            .into_iter()
            .map(|l| {
                l.iter()
                    .copied()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0)
            })
            .collect();
        for (i, &t) in tokens.iter().enumerate() {
            seqs[i].push(t);
        }
        cur = tokens;
    }
    seqs
}

/// Run n decode steps via the GPU ARGMAX token-only path.
fn greedy_path(engine: &mut QwenDense, seeds: &[u32], n: usize, max_seq: usize) -> Vec<Vec<u32>> {
    let b = seeds.len();
    let regions: Vec<usize> = (0..b).collect();
    let mut cur = seeds.to_vec();
    let mut seqs = vec![Vec::with_capacity(n); b];
    engine.multiseq_arena = None;
    for step in 0..n {
        let positions: Vec<usize> = vec![step; b];
        let tokens = engine
            .forward_tokens_multiseq_greedy(&cur, &positions, &regions, max_seq)
            .expect("greedy path");
        for (i, &t) in tokens.iter().enumerate() {
            seqs[i].push(t);
        }
        cur = tokens;
    }
    seqs
}

#[test]
fn greedy_token_only_matches_logits_argmax() {
    let mut engine = match load() {
        Some(e) => e,
        None => return,
    };
    let n = 4usize;
    let max_seq = 32usize;
    let seed_sets: &[&[u32]] = &[
        &[9707],                               // B=1
        &[9707, 374],                          // B=2
        &[9707, 374, 100, 151643],             // B=4
        &[9707, 374, 100, 151643, 1, 2, 3, 4], // B=8
    ];

    for seeds in seed_sets {
        let b = seeds.len();
        let expected = logits_path(&mut engine, seeds, n, max_seq);
        let got = greedy_path(&mut engine, seeds, n, max_seq);
        for slot in 0..b {
            assert_eq!(
                expected[slot], got[slot],
                "B={b} slot={slot} seeds={seeds:?}: \
                 greedy GPU argmax != logits CPU argmax\n  expected: {:?}\n  got: {:?}",
                expected[slot], got[slot]
            );
        }
        eprintln!("B={b}: {} steps parity OK", n);
    }
}

/// Engine trait dispatch parity: forward_multiseq_greedy_tokens (the trait
/// method QwenDense overrides) must return the same ids as direct calls.
#[test]
fn engine_trait_dispatch_matches_direct() {
    let mut engine = match load() {
        Some(e) => e,
        None => return,
    };
    let seeds: &[u32] = &[9707, 374, 100, 151643];
    let b = seeds.len();
    let max_seq = 32usize;
    let regions: Vec<usize> = (0..b).collect();
    let positions = vec![0usize; b];

    engine.multiseq_arena = None;
    let direct = engine
        .forward_tokens_multiseq_greedy(seeds, &positions, &regions, max_seq)
        .expect("direct");

    engine.multiseq_arena = None;
    let via_trait = engine
        .forward_multiseq_greedy_tokens(seeds, &positions, &regions)
        .expect("via trait");

    assert_eq!(direct, via_trait, "trait dispatch mismatch");
    eprintln!("engine trait dispatch parity OK: {:?}", direct);
}
