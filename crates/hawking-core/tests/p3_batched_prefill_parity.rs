#![cfg(target_os = "macos")]
use hawking_core::{
    profile::fresh_test_profile, EngineConfig, GenerateRequest, SamplingParams, StreamEvent,
};
mod common;
use common::weights_path_qwen as weights_path;
const PROMPT: &str = "Write a detailed explanation of how the attention mechanism in transformer \
     neural networks computes scaled dot-product attention scores using query, \
     key, and value matrices.";
const MAX_NEW_TOKENS: usize = 16;
fn run_greedy() -> Vec<u32> {
    let weights = weights_path();
    let profile = fresh_test_profile(&weights).expect("fresh test profile");
    let cfg = EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    let mut engine = hawking_core::model::load_engine(&weights, cfg).expect("load engine");
    let req = GenerateRequest {
        prompt: PROMPT.into(),
        max_new_tokens: MAX_NEW_TOKENS,
        sampling: SamplingParams {
            temperature: 0.0,
            seed: Some(42),
            ..Default::default()
        },
        stop: vec![],
        abort: None,
        max_stall_ms: 0,
        json_mode: false,
    };
    let mut ids: Vec<u32> = Vec::new();
    engine
        .generate(req, &mut |ev| {
            if let StreamEvent::Token { id, .. } = ev {
                ids.push(id);
            }
        })
        .expect("generate");
    ids
}
#[test]
fn batched_prefill_matches_single_token_prefill() {
    let weights = weights_path();
    if !weights.exists() {
        eprintln!(
            "skipping p3_batched_prefill_parity: weights missing at {:?}",
            weights
        );
        return;
    }
    std::env::set_var("HAWKING_QWEN_TCB", "1");
    std::env::remove_var("HAWKING_QWEN_BATCH_PREFILL");
    let baseline = run_greedy();
    std::env::set_var("HAWKING_QWEN_BATCH_PREFILL", "1");
    let batched = run_greedy();
    std::env::remove_var("HAWKING_QWEN_BATCH_PREFILL");
    std::env::remove_var("HAWKING_QWEN_TCB");
    assert_eq!(
        baseline.len(),
        batched.len(),
        "token count mismatch: baseline={} batched={}",
        baseline.len(),
        batched.len(),
    );
    assert_eq!(
        baseline, batched,
        "batched prefill must produce identical greedy tokens to single-token prefill"
    );
}
