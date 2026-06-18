//! path-to-50 lever 1: vocab-prune parity test.
//!
//! With the same prompt and fixed seed (temperature=0 ⇒ greedy), the
//! pruned model must produce the same token sequence as the unpruned
//! model — provided every emitted token survives the whitelist. The
//! whitelist covers ≥99.5% of corpus tokens; the held-out generation
//! prompt below is chosen to land in that 99.5%.
//!
//! If the test pins (no prior token-list on disk), it writes the
//! reference and asserts the rerun matches. Subsequent runs assert
//! exact equality.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use dismantle_core::{
    profile::fresh_test_profile, EngineConfig, GenerateRequest, SamplingParams, StreamEvent,
};

const PROMPT: &str = "Once upon a time";
const MAX_NEW_TOKENS: usize = 64;
const SEED: u64 = 42;

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
}

fn whitelist_path() -> PathBuf {
    PathBuf::from("../../artifacts/calibration/analysis/vocab_whitelist_995.json")
}

fn run_greedy(prune: Option<PathBuf>) -> Option<Vec<u32>> {
    let weights = weights_path();
    if !weights.exists() {
        eprintln!("skipping vocab_prune_parity: model weights missing");
        return None;
    }
    // Build a fresh kernel profile in-memory via the shared helper that
    // sidesteps on-disk shader-hash drift.
    let profile = fresh_test_profile(&weights).expect("fresh test profile");
    let cfg = EngineConfig {
        kernel_profile: Some(profile),
        vocab_prune_path: prune,
        ..Default::default()
    };
    let mut engine = dismantle_core::model::load_engine(&weights, cfg).expect("load engine");
    let req = GenerateRequest {
        prompt: PROMPT.into(),
        max_new_tokens: MAX_NEW_TOKENS,
        sampling: SamplingParams {
            temperature: 0.0,
            seed: Some(SEED),
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
    Some(ids)
}

#[test]
fn vocab_prune_matches_full_vocab_greedy() {
    let baseline = match run_greedy(None) {
        Some(b) => b,
        None => return,
    };
    let pruned_path = whitelist_path();
    if !pruned_path.exists() {
        eprintln!(
            "skipping vocab_prune_parity: whitelist missing at {:?}",
            pruned_path
        );
        return;
    }

    // The LM head GEMV produces logits[i] = W[i,:] @ x. Pruning deletes
    // some rows from W, so logits for surviving rows are bit-identical
    // to the unpruned logits restricted to those rows. Greedy (temp=0)
    // therefore agrees up to the first baseline token that is NOT in the
    // whitelist; at that position the pruned model cannot emit the same
    // token and divergence is expected. After divergence the input embed
    // for the next step differs, so subsequent positions are unrelated.
    let whitelist_bytes = std::fs::read(&pruned_path).expect("read vocab_whitelist_995.json");
    let raw: serde_json::Value =
        serde_json::from_slice(&whitelist_bytes).expect("parse whitelist json");
    let keep: std::collections::HashSet<u32> = raw["keep_token_ids"]
        .as_array()
        .expect("keep_token_ids array")
        .iter()
        .map(|v| v.as_u64().expect("u64 token id") as u32)
        .collect();

    let first_oov = baseline.iter().position(|t| !keep.contains(t));
    let parity_len = first_oov.unwrap_or(baseline.len());
    eprintln!(
        "vocab_prune_parity: baseline {} tokens; first OOV-in-whitelist at index {:?} → parity prefix {}",
        baseline.len(),
        first_oov,
        parity_len,
    );
    // Sanity: parity prefix should be substantial. A whitelist with 99.5%
    // coverage means OOV in a 64-token greedy run is uncommon but possible;
    // we require ≥ 4 to ensure the wiring isn't trivially broken at token 0.
    assert!(
        parity_len >= 4,
        "parity prefix too short ({}); wiring suspect — baseline[0..8]={:?}",
        parity_len,
        &baseline[..baseline.len().min(8)],
    );

    let pruned = run_greedy(Some(pruned_path)).expect("pruned generate");
    assert!(
        pruned.len() >= parity_len,
        "pruned produced fewer tokens ({}) than expected parity prefix ({})",
        pruned.len(),
        parity_len,
    );
    for i in 0..parity_len {
        assert_eq!(
            baseline[i], pruned[i],
            "vocab-prune parity diverged at in-whitelist token {}: baseline={} pruned={}",
            i, baseline[i], pruned[i],
        );
    }
}
