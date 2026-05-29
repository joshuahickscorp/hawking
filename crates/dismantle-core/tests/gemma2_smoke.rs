//! Gemma-2 smoke + greedy-output regression gate.
//!
//! Auto-activates when a `models/*gemma-2*.gguf` is present; skips
//! cleanly otherwise. Pins the greedy token-id hash on first run to
//! `tests/golden/_gemma2_token_baseline.hashes`, guards drift after.
//! Mirrors llama32_smoke.rs.
//!
//! Pull a GGUF into models/, e.g. models/gemma-2-2b-it-Q4_K_M.gguf.

#![cfg(target_os = "macos")]

use sha2::{Digest, Sha256};
use std::path::PathBuf;

const PROMPT: &str = "Once upon a time";
const MAX_NEW_TOKENS: usize = 32;

fn find_gguf(size_tag: &str) -> Option<PathBuf> {
    let dir = PathBuf::from("../../models");
    for e in std::fs::read_dir(&dir).ok()?.flatten() {
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

fn run_greedy(weights: &PathBuf) -> Vec<u32> {
    let cfg = dismantle_core::EngineConfig::default();
    let mut engine =
        dismantle_core::model::load_engine(weights, cfg).expect("load gemma2 engine");
    assert_eq!(engine.model_arch(), "gemma2", "dispatcher must route to gemma2");
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

fn check_or_pin(label: &str, actual_hash: &str) {
    let pin_path = PathBuf::from("../../tests/golden/_gemma2_token_baseline.hashes");
    let actual_line = format!("{}: {}\n", label, actual_hash);
    let existing = std::fs::read_to_string(&pin_path).unwrap_or_default();
    match existing
        .lines()
        .find(|l| l.starts_with(&format!("{}:", label)))
    {
        None => {
            let mut all = existing;
            all.push_str(&actual_line);
            std::fs::write(&pin_path, all).expect("write pin");
            eprintln!("PINNED first hash for {}: {}", label, actual_hash);
        }
        Some(prior) => assert_eq!(
            prior.trim(),
            actual_line.trim(),
            "gemma2 greedy hash drift for {label}"
        ),
    }
}

#[test]
fn gemma2_2b_greedy_smoke() {
    let Some(weights) = find_gguf("gemma-2-2b") else {
        eprintln!("skipping gemma2-2b: no models/*gemma-2-2b*.gguf present");
        return;
    };
    eprintln!("running gemma2-2b against {}", weights.display());
    let ids = run_greedy(&weights);
    let ids2 = run_greedy(&weights);
    assert_eq!(ids, ids2, "gemma2-2b: greedy temp=0 not deterministic");
    check_or_pin("gemma-2-2b-it", &hash16(&ids));
}
