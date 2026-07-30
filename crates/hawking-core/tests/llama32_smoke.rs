#![cfg(target_os = "macos")]
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
mod common;
use common::*;
const PROMPT: &str = "Once upon a time";
const MAX_NEW_TOKENS: usize = 32;
fn find_llama_gguf(size_tag: &str) -> Option<PathBuf> {
    let dir = PathBuf::from("../../models");
    let entries = std::fs::read_dir(&dir).ok()?;
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("gguf") {
            continue;
        }
        let name = p.file_name()?.to_str()?.to_lowercase();
        if name.contains(size_tag) {
            return Some(p);
        }
    }
    None
}
fn run_greedy(weights: &PathBuf, expect_arch: &str) -> Vec<u32> {
    let cfg = hawking_core::EngineConfig::default();
    let mut engine = hawking_core::model::load_engine(weights, cfg).expect("load llama engine");
    assert_eq!(
        engine.model_arch(),
        expect_arch,
        "dispatcher routed to the wrong engine"
    );
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
fn smoke_for(size_tag: &str, label: &str, expect_arch: &str) {
    let Some(weights) = find_llama_gguf(size_tag) else {
        eprintln!("skipping {label}: no models/*{size_tag}*.gguf present");
        return;
    };
    let ids = run_greedy(&weights, expect_arch);
    let ids2 = run_greedy(&weights, expect_arch);
    assert_eq!(ids, ids2, "{label}: greedy temp=0 output not deterministic");
    // Pin the *input* alongside the output. Without this the baseline cannot
    // tell a code regression from a different GGUF build: the finder accepts
    // any models/*<size_tag>*.gguf, and two publishers' Q4_K_M of the same
    // model produce different tokens for reasons that are nobody's bug.
    let bytes = std::fs::metadata(&weights).map(|m| m.len()).unwrap_or(0);
    let mut h = Sha256::new();
    h.update(std::fs::read(&weights).expect("read weights"));
    let weights_sha = format!("{:x}", h.finalize());
    check_or_pin_hash(
        Path::new("tests/golden/_llama32_token_baseline.hashes"),
        &format!("{label}/in:{}:{}", &weights_sha[..16], bytes),
        &hash16_tokens(&ids),
    );
}
#[test]
fn llama32_1b_greedy_smoke() {
    smoke_for("llama-3.2-1b", "llama-3.2-1b-instruct", "llama");
}
#[test]
fn llama32_3b_greedy_smoke() {
    smoke_for("llama-3.2-3b", "llama-3.2-3b-instruct", "llama");
}
#[test]
fn llama31_8b_greedy_smoke() {
    smoke_for("llama-3.1-8b", "llama-3.1-8b-instruct", "llama");
}
#[test]
fn mistral_7b_v03_greedy_smoke() {
    smoke_for("mistral-7b", "mistral-7b-instruct-v0.3", "llama");
}
