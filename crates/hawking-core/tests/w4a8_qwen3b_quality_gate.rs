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
    /// Logits at the final-prompt position (vocab_pruned dim when prune is active).
    logits: Vec<f32>,
    /// Greedy tokens produced after the prompt.
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
fn run(weights: &PathBuf, w4a8: bool) -> Option<RunOut> {
    if !weights.exists() {
        eprintln!("skipping w4a8_qwen3b_quality_gate: weights missing at {weights:?}");
        return None;
    }
    set_locked_env();
    if w4a8 {
        std::env::set_var("HAWKING_QWEN_W4A8", "1");
    } else {
        std::env::remove_var("HAWKING_QWEN_W4A8");
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
    assert!(
        prompt_ids.len() >= 2,
        "prompt tokenized too short: {:?}",
        prompt_ids
    );
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
fn w4a8_quality_gate_self_consistency_f32() {
    let weights = weights_path();
    let a = match run(&weights, false) {
        Some(o) => o,
        None => return,
    };
    let b = run(&weights, false).expect("second f32 run");
    let cos = cosine(&a.logits, &b.logits);
    assert!(cos > 0.99999, "two fresh f32 runs disagree: cos={cos}");
    assert_eq!(
        a.tokens, b.tokens,
        "two fresh f32 runs emit different tokens"
    );
}
#[test]
#[ignore]
fn w4a8_quality_gate() {
    let weights = weights_path();
    let f32_out = match run(&weights, false) {
        Some(o) => o,
        None => return,
    };
    let w4a8_out = run(&weights, true).expect("w4a8 run after f32 run");
    std::env::remove_var("HAWKING_QWEN_W4A8");
    let cos = cosine(&f32_out.logits, &w4a8_out.logits);
    let first_div = f32_out
        .tokens
        .iter()
        .zip(&w4a8_out.tokens)
        .position(|(a, b)| a != b)
        .unwrap_or(MAX_NEW);
    assert!(cos > 0.998, "logit cosine too low: {cos:.6} (need > 0.998)");
    assert!(
        first_div >= 8,
        "greedy divergence at token {first_div} (need >= 8)"
    );
}
