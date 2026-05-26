//! Lookahead n-gram parity test — re-port from feat/qwen-lookahead-decoding.
//!
//! Greedy temp=0 with DISMANTLE_LOOKAHEAD=N must produce bit-identical
//! tokens to greedy without lookahead. Lookahead is supposed to be an
//! optimization over the same argmax-of-forward; on every verify
//! mismatch the correction IS the model's argmax, so the emitted
//! sequence should be identical.
//!
//! Branch's prior failure on the repetitive prompt motivated this
//! re-port + investigation.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use dismantle_core::{
    profile::fresh_test_profile, EngineConfig, GenerateRequest, SamplingParams,
    StreamEvent,
};

const PROMPT_CHAT: &str = "Briefly explain what a transformer attention head does.";
const PROMPT_REPEAT: &str =
    "List the first 20 prime numbers in order: 2, 3, 5, 7, 11,";
const MAX_NEW_TOKENS: usize = 24;

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

fn run_greedy(prompt: &str) -> Vec<u32> {
    let weights = weights_path();
    let profile = fresh_test_profile(&weights).expect("fresh test profile");
    let cfg = EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    let mut engine = dismantle_core::model::load_engine(&weights, cfg).expect("load engine");
    let req = GenerateRequest {
        prompt: prompt.into(),
        max_new_tokens: MAX_NEW_TOKENS,
        sampling: SamplingParams {
            temperature: 0.0,
            seed: Some(42),
            ..Default::default()
        },
        stop: vec![],
        abort: None,
        max_stall_ms: 0,
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

fn assert_parity(prompt: &str, label: &str) {
    if !weights_path().exists() {
        eprintln!("skip ({label}): weights missing");
        return;
    }

    // Baseline: TCB greedy, no lookahead.
    std::env::set_var("DISMANTLE_QWEN_TCB", "1");
    std::env::remove_var("DISMANTLE_LOOKAHEAD");
    std::env::remove_var("DISMANTLE_LOOKAHEAD_K");
    let baseline = run_greedy(prompt);

    // Lookahead n=4, k=4.
    std::env::set_var("DISMANTLE_LOOKAHEAD", "4");
    std::env::set_var("DISMANTLE_LOOKAHEAD_K", "4");
    let lookahead = run_greedy(prompt);

    // Restore env so we don't leak into other tests.
    std::env::remove_var("DISMANTLE_LOOKAHEAD");
    std::env::remove_var("DISMANTLE_LOOKAHEAD_K");

    eprintln!("[{label}] baseline ({} tok):  {:?}", baseline.len(), baseline);
    eprintln!("[{label}] lookahead ({} tok): {:?}", lookahead.len(), lookahead);

    let mut first_div = baseline.len().min(lookahead.len());
    for i in 0..first_div {
        if baseline[i] != lookahead[i] {
            first_div = i;
            break;
        }
    }
    eprintln!("[{label}] first divergence at index {first_div}");

    assert_eq!(baseline, lookahead, "{label} parity failed");
}

#[test]
#[ignore]
fn lookahead_parity_chat() {
    assert_parity(PROMPT_CHAT, "chat");
}

#[test]
#[ignore]
fn lookahead_parity_repetitive() {
    assert_parity(PROMPT_REPEAT, "repetitive");
}
