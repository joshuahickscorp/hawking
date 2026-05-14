// Master integration test: pins the greedy 64-token output of the
// production profile so any architectural change proves it didn't drift.

#![cfg(target_os = "macos")]

use sha2::{Digest, Sha256};
use std::path::PathBuf;

const PROMPT: &str = "Once upon a time";
const MAX_NEW_TOKENS: usize = 64;

fn run_greedy_64(
    weights: &PathBuf,
    cfg: dismantle_core::EngineConfig,
) -> Vec<u32> {
    let mut engine =
        dismantle_core::model::load_engine(weights, cfg).expect("load engine");
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
    let mut ids: Vec<u32> = Vec::new();
    engine
        .generate(req, &mut |ev| {
            if let dismantle_core::StreamEvent::Token { id, .. } = ev {
                ids.push(id);
            }
        })
        .expect("generate");
    assert!(!ids.is_empty(), "must produce at least one token");
    ids
}

fn hash16(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for &id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())[..16].to_string()
}

fn check_or_pin(pin_path: &PathBuf, label: &str, actual_hash: &str) {
    let actual_line = format!("{}: {}\n", label, actual_hash);
    if !pin_path.exists() {
        std::fs::write(pin_path, &actual_line).expect("write pin");
        eprintln!("PINNED first hash for {}: {}", label, actual_hash);
        return;
    }
    let pinned = std::fs::read_to_string(pin_path).expect("read pin");
    if !pinned.contains(actual_line.trim()) {
        let prior = pinned
            .lines()
            .find(|l| l.starts_with(&format!("{}:", label)))
            .unwrap_or("(no prior pin for this label)");
        panic!(
            "greedy 64 hash drift for {}:\n  pinned: {}\n  actual: {}",
            label,
            prior,
            actual_line.trim()
        );
    }
}

// ── F32 regression guard (original c30b9e036d578ae2 must hold) ──────────────

#[test]
fn greedy_64_f32_regression() {
    let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
    if !weights.exists() {
        eprintln!("skipping greedy_64_f32_regression: no weights");
        return;
    }
    let profile_path = PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
    let profile = dismantle_core::profile::KernelProfile::load(&profile_path)
        .expect("load profile");

    let cfg = dismantle_core::EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    let ids = run_greedy_64(&weights, cfg);
    let hash = hash16(&ids);

    check_or_pin(
        &PathBuf::from("../../tests/golden/_phase0_token_baseline_64.hashes"),
        "v0.5.0-phase0",
        &hash,
    );
}
