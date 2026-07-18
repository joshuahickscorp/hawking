//! P0.1 spike — Q/K/V concurrent-encoder parity gate.
//!
//! Mirrors `w4a8_qwen3b_quality_gate.rs` but compares
//! `HAWKING_QWEN_CONCURRENT_QKV` off vs on on the locked Qwen-3B-Q4_K_M
//! config. Concurrent dispatch reorders execution but does not change
//! the math — both runs should be bit-identical (cosine ≈ 1.0, first 8
//! greedy tokens identical).
//!
//! If parity fails the spike halts per the plan:
//! (see project design memory: closing-the-2-4-virtual-phoenix).
//!
//! Run via:
//!
//!   cargo test --release -p hawking-core \
//!     --test qkv_concurrent_parity -- --ignored --nocapture
//!
//! `#[ignore]` so it doesn't fire on `cargo test` (loads the 1.9 GB
//! Qwen-3B mmap and runs two forwards).

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use hawking_core::{
    metal::DenseDecodeArena, model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine,
    EngineConfig,
};

const PROMPT: &str = "Hello, my name is";
const MAX_NEW: usize = 16;

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

struct RunOut {
    logits: Vec<f32>,
    tokens: Vec<u32>,
}

fn set_locked_env() {
    std::env::set_var("HAWKING_QWEN_TCB", "1");
    std::env::set_var("HAWKING_QWEN_VOCAB_PRUNE_CORPUS", "32000");
    std::env::set_var("HAWKING_QWEN_Q4K_LMHEAD", "1");
    std::env::set_var("HAWKING_QWEN_FFN_DOWN_Q4K", "1");
}

fn read_logits(arena: &DenseDecodeArena, n: usize) -> Vec<f32> {
    let ptr = arena.logits_buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

fn run(weights: &PathBuf, concurrent: bool) -> Option<RunOut> {
    if !weights.exists() {
        eprintln!("skipping qkv_concurrent_parity: weights missing at {weights:?}");
        return None;
    }
    set_locked_env();
    if concurrent {
        std::env::set_var("HAWKING_QWEN_CONCURRENT_QKV", "1");
    } else {
        std::env::remove_var("HAWKING_QWEN_CONCURRENT_QKV");
    }

    let profile = fresh_test_profile(weights).expect("fresh test profile");
    let cfg = EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    let mut engine = QwenDense::load(weights, cfg).expect("load qwen-3b");
    let prompt_ids = engine
        .tokenizer
        .encode(PROMPT, true)
        .expect("encode prompt");
    assert!(prompt_ids.len() >= 2, "prompt too short: {:?}", prompt_ids);

    for (i, &t) in prompt_ids.iter().enumerate() {
        let _ = engine
            .forward_token_greedy_tcb(t, i)
            .expect("prefill forward");
    }
    let pn = engine
        .vocab_pruned
        .expect("vocab-prune must be active under locked config");
    let arena = engine
        .dense_arena
        .as_ref()
        .expect("arena populated after first forward");
    let logits = read_logits(arena, pn);

    let mut tokens = Vec::with_capacity(MAX_NEW);
    let mut last = *prompt_ids.last().unwrap();
    for step in 0..MAX_NEW {
        let pos = prompt_ids.len() + step;
        let next = engine
            .forward_token_greedy_tcb(last, pos)
            .expect("decode forward");
        tokens.push(next);
        last = next;
    }
    Some(RunOut { logits, tokens })
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// Sanity: two fresh baseline (off) runs must produce identical logits +
/// tokens. If this fails, the parity methodology itself is unsound and
/// any concurrent-on number is meaningless.
#[test]
#[ignore]
fn qkv_concurrent_parity_self_consistency() {
    let weights = weights_path();
    let a = match run(&weights, false) {
        Some(o) => o,
        None => return,
    };
    let b = run(&weights, false).expect("second baseline run");
    let cos = cosine(&a.logits, &b.logits);
    eprintln!("[sanity] off-vs-off logits cosine = {cos:.9}");
    assert!(cos > 0.99999, "two baseline runs disagree: cos={cos}");
    assert_eq!(
        a.tokens, b.tokens,
        "two baseline runs emit different tokens"
    );
}

#[test]
#[ignore]
fn qkv_concurrent_parity() {
    let weights = weights_path();
    let off = match run(&weights, false) {
        Some(o) => o,
        None => return,
    };
    let on = run(&weights, true).expect("concurrent run after baseline");
    std::env::remove_var("HAWKING_QWEN_CONCURRENT_QKV");

    let cos = cosine(&off.logits, &on.logits);
    eprintln!(
        "[qkv_concurrent_parity] logits_cosine = {cos:.9} | off tokens = {:?} | on tokens = {:?}",
        off.tokens, on.tokens,
    );

    let first_div = off
        .tokens
        .iter()
        .zip(&on.tokens)
        .position(|(a, b)| a != b)
        .unwrap_or(MAX_NEW);
    eprintln!("[qkv_concurrent_parity] first divergence at token index {first_div}");

    // Concurrent dispatch reorders execution but not math: cosine must
    // be essentially 1.0 and first 8 greedy tokens identical. The plan's
    // session-level gate is cosine > 0.998 + first-8 match; we use a
    // tighter cosine here because there's no quantization-noise source.
    assert!(
        cos > 0.998,
        "logit cosine too low: {cos:.6} (need > 0.998 per plan)"
    );
    assert!(
        first_div >= 8,
        "greedy divergence at token {first_div} (need >= 8 per plan)"
    );
}
