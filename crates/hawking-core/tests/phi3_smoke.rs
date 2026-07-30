#![cfg(target_os = "macos")]
use std::path::{Path, PathBuf};
mod common;
use common::*;
const PROMPT: &str = "Once upon a time";
const MAX_NEW_TOKENS: usize = 32;
fn run_greedy(weights: &PathBuf) -> Vec<u32> {
    let cfg = hawking_core::EngineConfig::default();
    let mut engine = hawking_core::model::load_engine(weights, cfg).expect("load engine");
    assert_eq!(engine.model_arch(), "phi3", "dispatcher must route to phi3");
    let req = hawking_core::GenerateRequest {
        prompt: PROMPT.into(),
        max_new_tokens: MAX_NEW_TOKENS,
        sampling: hawking_core::SamplingParams {
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
            if let hawking_core::StreamEvent::Token { id, .. } = ev {
                ids.push(id);
            }
        })
        .expect("generate");
    assert!(!ids.is_empty(), "must produce at least one token");
    ids
}
#[test]
fn phi35_mini_greedy_smoke() {
    let Some(weights) = find_gguf_with_tags(&["phi-3.5-mini", "phi-3-mini", "phi3.5-mini"]) else {
        eprintln!("skipping phi3.5-mini: no models/*phi-3*mini*.gguf present");
        return;
    };
    let ids = run_greedy(&weights);
    let ids2 = run_greedy(&weights);
    assert_eq!(ids, ids2, "greedy not deterministic");
    check_or_pin_hash(
        Path::new("tests/golden/_phi3_token_baseline.hashes"),
        "phi3",
        &hash16_tokens(&ids),
    );
}
