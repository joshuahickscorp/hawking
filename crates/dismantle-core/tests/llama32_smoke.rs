//! Llama-3.2 smoke + greedy-output regression gate.
//!
//! Auto-activates when a matching GGUF is present under `models/`;
//! skips cleanly otherwise (so CI without weights stays green). On first
//! run it PINS the greedy token-id hash to
//! `tests/golden/_llama32_token_baseline.hashes`; subsequent runs guard
//! against drift. Mirrors `integration_greedy_64.rs`.
//!
//! Pull a GGUF into `models/` to enable, e.g.:
//!   models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
//!   models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
//! The matcher is case-insensitive and keys on the size token
//! ("1b" / "3b" / "8b") + ".gguf", so the exact HF filename is fine.

#![cfg(target_os = "macos")]

use sha2::{Digest, Sha256};
use std::path::PathBuf;

const PROMPT: &str = "Once upon a time";
const MAX_NEW_TOKENS: usize = 32;

/// Find the first `models/*.gguf` whose lowercased name contains
/// `size_tag` (e.g. "llama-3.2-1b"). Returns None if absent.
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
    // No kernel profile: the engine runs with default kernel selections.
    // Profiles are model-specific (generated via `dismantle autotune`);
    // the smoke gate intentionally exercises the no-profile load path.
    let cfg = dismantle_core::EngineConfig::default();
    let mut engine = dismantle_core::model::load_engine(weights, cfg).expect("load llama engine");
    assert_eq!(
        engine.model_arch(),
        expect_arch,
        "dispatcher routed to the wrong engine"
    );

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
        json_mode: false,
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

fn check_or_pin(label: &str, actual_hash: &str) {
    let pin_path = PathBuf::from("tests/golden/_llama32_token_baseline.hashes");
    let actual_line = format!("{}: {}\n", label, actual_hash);
    let existing = std::fs::read_to_string(&pin_path).unwrap_or_default();
    let prior = existing
        .lines()
        .find(|l| l.starts_with(&format!("{}:", label)));
    match prior {
        None => {
            // Append-pin first hash for this label.
            let mut all = existing;
            all.push_str(&actual_line);
            std::fs::write(&pin_path, all).expect("write pin");
            eprintln!("PINNED first hash for {}: {}", label, actual_hash);
        }
        Some(prior_line) => {
            assert_eq!(
                prior_line.trim(),
                actual_line.trim(),
                "llama greedy hash drift for {label}:\n  pinned: {prior_line}\n  actual: {}",
                actual_line.trim()
            );
        }
    }
}

fn smoke_for(size_tag: &str, label: &str, expect_arch: &str) {
    let Some(weights) = find_llama_gguf(size_tag) else {
        eprintln!("skipping {label}: no models/*{size_tag}*.gguf present");
        return;
    };
    eprintln!("running {label} against {}", weights.display());
    let ids = run_greedy(&weights, expect_arch);
    // Sanity: greedy at temp=0 must be deterministic across two runs.
    let ids2 = run_greedy(&weights, expect_arch);
    assert_eq!(ids, ids2, "{label}: greedy temp=0 output not deterministic");
    check_or_pin(label, &hash16(&ids));
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
    // Llama-3.1-8B is the larger coverage target; matcher keys on "8b".
    smoke_for("llama-3.1-8b", "llama-3.1-8b-instruct", "llama");
}

#[test]
fn mistral_7b_v03_greedy_smoke() {
    // Mistral-7B-Instruct-v0.3 reports arch "llama" and runs through the
    // same dense engine (GQA + SwiGLU + RoPE θ=1e6, no biases, no SWA).
    smoke_for("mistral-7b", "mistral-7b-instruct-v0.3", "llama");
}
