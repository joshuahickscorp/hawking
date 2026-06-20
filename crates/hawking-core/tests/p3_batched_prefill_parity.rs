//! P3 — batched prefill parity test.
//!
//! End-to-end check that `forward_tokens_batch_tcb` (gated via
//! `HAWKING_QWEN_BATCH_PREFILL=1`) produces the same token sequence
//! as the single-token TCB prefill path on a greedy generation.
//!
//! Skipped if `models/qwen2.5-3b-instruct-q4_k_m.gguf` is not present
//! on disk (CI without weights downloads, etc).

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use hawking_core::{
    profile::fresh_test_profile, EngineConfig, GenerateRequest, SamplingParams, StreamEvent,
};

const PROMPT: &str = "Write a detailed explanation of how the attention mechanism in transformer \
     neural networks computes scaled dot-product attention scores using query, \
     key, and value matrices.";
const MAX_NEW_TOKENS: usize = 16;

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

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

    // Path A: single-token TCB prefill.
    std::env::set_var("HAWKING_QWEN_TCB", "1");
    std::env::remove_var("HAWKING_QWEN_BATCH_PREFILL");
    let baseline = run_greedy();

    // Path B: batched B=4 TCB prefill.
    std::env::set_var("HAWKING_QWEN_BATCH_PREFILL", "1");
    let batched = run_greedy();

    // Clean up so we don't leak into other tests.
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
