#![cfg(target_os = "macos")]
use sha2::{Digest, Sha256};
use std::path::PathBuf;
const PROMPT: &str = "Once upon a time";
const MAX_NEW_TOKENS: usize = 64;
fn run_greedy_64(weights: &PathBuf, cfg: hawking_core::EngineConfig) -> Vec<u32> {
    let mut engine = hawking_core::model::load_engine(weights, cfg).expect("load engine");
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
        Some(prior) if prior.trim() == actual_line => {}
        Some(prior) => {
            panic!("greedy 64 hash drift for {label}:\n  pinned: {prior}\n  actual: {actual_line}")
        }
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
fn first_available_model() -> Option<(PathBuf, PathBuf, &'static str)> {
    const CANDIDATES: &[(&str, &str, &str)] = &[
        (
            "../../models/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
            "../../workspace/campaign/config/profiles/qwen/qwen3b-instruct-q4k.m3pro18.json",
            "qwen3b-q4k-greedy64",
        ),
        (
            "../../models/deepseek-v2-lite-q4.gguf",
            "../../workspace/campaign/config/profiles/deepseek-v2-lite/baseline/deepseek-v2-lite-q4.m3pro18.json",
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
    let profile = hawking_core::profile::KernelProfile::load(&profile_path).expect("load profile");
    let cfg = hawking_core::EngineConfig {
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
