#![cfg(target_os = "macos")]
use hawking_core::{
    metal::DenseDecodeArena, model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine,
    EngineConfig,
};
use std::path::PathBuf;
mod common;
use common::weights_path_qwen as weights_path;
const PROMPT: &str = "Hello, my name is";
const MAX_NEW: usize = 16;
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
    let first_div = off
        .tokens
        .iter()
        .zip(&on.tokens)
        .position(|(a, b)| a != b)
        .unwrap_or(MAX_NEW);
    assert!(
        cos > 0.998,
        "logit cosine too low: {cos:.6} (need > 0.998 per plan)"
    );
    assert!(
        first_div >= 8,
        "greedy divergence at token {first_div} (need >= 8 per plan)"
    );
}
