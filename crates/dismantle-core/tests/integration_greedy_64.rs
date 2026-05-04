// Master integration test: pins the greedy 64-token output of the
// production profile so any architectural change (multi-token forward,
// GPU-resident residual, fp16, etc.) can prove it didn't drift output
// at temp=0.
//
// Hash file: tests/golden/_phase0_token_baseline_64.hashes
//   Format: one line per pin, "<phase_label>: <sha256_prefix>"
//   On first run (no pin), writes the current hash and PASSES.
//   On later runs, compares and FAILS on drift.

#![cfg(target_os = "macos")]

use sha2::{Digest, Sha256};
use std::path::PathBuf;

const PROMPT: &str = "Once upon a time";
const MAX_NEW_TOKENS: usize = 64;
const PIN_LABEL: &str = "v0.5.0-phase0";

#[test]
fn greedy_64_matches_pinned_hash() {
    let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
    if !weights.exists() {
        eprintln!("skipping greedy_64: no weights at {:?}", weights);
        return;
    }
    let profile_path = PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");

    let profile = dismantle_core::profile::KernelProfile::load(&profile_path)
        .expect("load profile");
    let cfg = dismantle_core::EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    let mut engine = dismantle_core::model::load_engine(&weights, cfg)
        .expect("load engine");

    let req = dismantle_core::GenerateRequest {
        prompt: PROMPT.into(),
        max_new_tokens: MAX_NEW_TOKENS,
        sampling: dismantle_core::SamplingParams {
            temperature: 0.0,
            seed: Some(42),
            ..Default::default()
        },
        stop: vec![],
        abort: None,
        max_stall_ms: 0,
    };

    let mut token_ids: Vec<u32> = Vec::new();
    engine
        .generate(req, &mut |ev| {
            if let dismantle_core::StreamEvent::Token { id, .. } = ev {
                token_ids.push(id);
            }
        })
        .expect("generate");

    assert!(!token_ids.is_empty(), "must produce at least one token");

    let mut hasher = Sha256::new();
    for id in &token_ids {
        hasher.update(id.to_le_bytes());
    }
    let actual_hash: String = format!("{:x}", hasher.finalize())[..16].to_string();
    let actual_line = format!("{}: {}\n", PIN_LABEL, actual_hash);

    let pin_path = PathBuf::from("../../tests/golden/_phase0_token_baseline_64.hashes");
    if !pin_path.exists() {
        std::fs::write(&pin_path, &actual_line).expect("write pin");
        eprintln!("PINNED first hash for {}: {}", PIN_LABEL, actual_hash);
        return;
    }

    let pinned = std::fs::read_to_string(&pin_path).expect("read pin");
    if !pinned.contains(actual_line.trim()) {
        let pinned_for_label = pinned
            .lines()
            .find(|l| l.starts_with(&format!("{}:", PIN_LABEL)))
            .unwrap_or("(no prior pin for this label)");
        panic!(
            "greedy 64 hash drift for {}:\n  pinned: {}\n  actual: {}\n  token_ids: {:?}",
            PIN_LABEL,
            pinned_for_label,
            actual_line.trim(),
            token_ids
        );
    }
}
