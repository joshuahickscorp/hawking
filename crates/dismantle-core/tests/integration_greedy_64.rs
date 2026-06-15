// Master integration test: pins the greedy 64-token output of the
// production profile so any architectural change proves it didn't drift.

#![cfg(target_os = "macos")]

use sha2::{Digest, Sha256};
use std::path::PathBuf;

const PROMPT: &str = "Once upon a time";
const MAX_NEW_TOKENS: usize = 64;

fn run_greedy_64(weights: &PathBuf, cfg: dismantle_core::EngineConfig) -> Vec<u32> {
    let mut engine = dismantle_core::model::load_engine(weights, cfg).expect("load engine");
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
    let actual_line = format!("{label}: {actual_hash}");
    let pinned = std::fs::read_to_string(pin_path).unwrap_or_default();
    match pinned.lines().find(|l| l.starts_with(&format!("{label}:"))) {
        // This label has a prior pin and it matches — the guard holds.
        Some(prior) if prior.trim() == actual_line => {}
        // This label drifted — a real regression.
        Some(prior) => panic!(
            "greedy 64 hash drift for {label}:\n  pinned: {prior}\n  actual: {actual_line}"
        ),
        // First sight of this label — pin it (append; create the file if absent).
        None => {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(pin_path)
                .expect("open pin for append");
            writeln!(f, "{actual_line}").expect("append pin");
            eprintln!("PINNED first hash for {label}: {actual_hash}");
        }
    }
}

// ── Greedy 64-token regression guard ────────────────────────────────────────
//
// Runs against the first model present on disk — the primary Qwen-3B target,
// then DeepSeek-V2-Lite. The hash self-pins on first run and guards drift
// afterward. It skips ONLY when no model is available; previously it pinned a
// model that wasn't on disk, so it skipped silently and guarded nothing.

/// `(weights, profile, pin-label)` for the first model that exists on disk.
fn first_available_model() -> Option<(PathBuf, PathBuf, &'static str)> {
    const CANDIDATES: &[(&str, &str, &str)] = &[
        (
            "../../models/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
            "../../profiles/qwen3b-instruct-q4k.m3pro18.json",
            "qwen3b-q4k-greedy64",
        ),
        (
            "../../models/deepseek-v2-lite-q4.gguf",
            "../../profiles/deepseek-v2-lite-q4.m3pro18.json",
            "deepseek-v2-lite-q4-greedy64",
        ),
    ];
    CANDIDATES.iter().find_map(|&(w, p, label)| {
        let wp = PathBuf::from(w);
        if wp.exists() {
            Some((wp, PathBuf::from(p), label))
        } else {
            None
        }
    })
}

#[test]
fn greedy_64_regression() {
    let Some((weights, profile_path, label)) = first_available_model() else {
        eprintln!(
            "skipping greedy_64_regression: no model on disk (tried Qwen-3B, DeepSeek-V2-Lite)"
        );
        return;
    };
    let profile =
        dismantle_core::profile::KernelProfile::load(&profile_path).expect("load profile");
    let cfg = dismantle_core::EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    let ids = run_greedy_64(&weights, cfg);
    let hash = hash16(&ids);
    check_or_pin(
        &PathBuf::from("tests/golden/_phase0_token_baseline_64.hashes"),
        label,
        &hash,
    );
}
