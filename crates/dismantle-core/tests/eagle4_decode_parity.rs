//! Path-to-90 step 9 — `SpeculateMode::Eagle4` should produce output
//! bit-identical to `SpeculateMode::Off` greedy.
//!
//! **CURRENTLY FAILS.** See `reports/path_to_90/foundation_halt.md`.
//! The CPU walk used by `forward_token_eagle4_capture_with_argmax`
//! diverges from the GPU Wedge C `forward_token` at *argmax* level —
//! not subtle fp drift, completely different tokens from step 0:
//!
//! ```text
//! Off:    33747, 855,   254,   24547, 5025,  5025,  ...
//! Eagle4: 257,   9442,  78887, 71199, 21700, 28542, ...
//! ```
//!
//! Test landed for the audit trail; fixing the divergence is the
//! next attended session's job per CLAUDE.md § Halt rule. Once
//! fixed, expected behavior: zero mismatches across the full token
//! window for any prompt.
//!
//! Gating: `#[ignore]`'d AND short-circuits unless `EAGLE4_PARITY_TEST=1`
//! is set. With the env var set + the trained head + frozen NPZs
//! present, currently panics with "Eagle4 greedy not bit-identical to
//! Off". That's the documented halt state.
//!
//! ## K-window note
//!
//! Step 8 ships K=1 verify-by-comparison: the decode loop always
//! emits the CPU walk's V2-Lite argmax (currently divergent from GPU
//! Wedge C's argmax — hence the halt). When the divergence is fixed
//! AND the loop is switched to GPU-side emission, K∈{1,2,4,8} matrix
//! sweep becomes meaningful. Stage 2 Path B kernels (execution_plan
//! steps 12–17) land K-batched verify.
//!
//! ## Runtime
//!
//! Single prompt × 16 tokens × 2 modes ≈ 1–2 min on M3 Pro 18 GB
//! (Eagle4 mode is the CPU-walk path, ~3.7s/token; Off is GPU
//! Wedge C, ~40ms/token; prefill amortizes once per mode). Set
//! DISMANTLE_EAGLE4_GREEDY_TOKENS to override the per-run token budget.
//!
//! Run:
//!
//! ```bash
//! EAGLE4_PARITY_TEST=1 cargo test -p dismantle-core --release \
//!     --test eagle4_decode_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use dismantle_core::{
    profile::KernelProfile, EngineConfig, GenerateRequest, SamplingParams, SpeculateMode,
    StreamEvent,
};
use std::env;
use std::path::PathBuf;

const PROMPT: &str = "The quick brown fox";
const DEFAULT_TOKENS: usize = 16;

fn greedy_request(prompt: &str, max_new_tokens: usize) -> GenerateRequest {
    GenerateRequest {
        prompt: prompt.to_string(),
        max_new_tokens,
        sampling: SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
        },
        stop: Vec::new(),
        abort: None,
        max_stall_ms: 0,
    }
}

fn collect_tokens(
    weights: &PathBuf,
    profile_path: &PathBuf,
    mode: SpeculateMode,
    eagle4_head: Option<&PathBuf>,
    eagle4_frozen: Option<&PathBuf>,
    max_new_tokens: usize,
) -> Vec<u32> {
    let profile = KernelProfile::load(profile_path).expect("load kernel profile");
    let cfg = EngineConfig {
        max_seq_len: 4096,
        max_batch_size: 1,
        speculate: mode != SpeculateMode::Off,
        speculate_mode: mode,
        verify_window: 1, // K=1 — see header note
        prefill_cache_dir: None,
        kernel_profile: Some(profile),
        trace_dispatch: false,
        max_routed_expert_ram_mb: None,
        memory_limit_mb: None,
        eagle4_head_path: eagle4_head.cloned(),
        eagle4_frozen_path: eagle4_frozen.cloned(),
        eagle4_calib_threshold: 0.5,
    };
    let mut engine = dismantle_core::model::load_engine(weights, cfg)
        .unwrap_or_else(|e| panic!("load_engine({mode:?}): {e}"));
    let req = greedy_request(PROMPT, max_new_tokens);
    let mut tokens = Vec::new();
    engine
        .generate(req, &mut |ev| match ev {
            StreamEvent::Token { id, .. } => tokens.push(id),
            StreamEvent::Done { .. } => {}
        })
        .unwrap_or_else(|e| panic!("generate({mode:?}): {e}"));
    tokens
}

#[test]
#[ignore = "requires eagle4 artifacts + V2-Lite weights; gate via EAGLE4_PARITY_TEST=1"]
fn eagle4_greedy_bit_identical_with_off() {
    if env::var("EAGLE4_PARITY_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping eagle4 greedy parity; set EAGLE4_PARITY_TEST=1");
        return;
    }

    let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
    let profile = PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
    let head = PathBuf::from("../../eagle4/checkpoints/eagle4_v3/best.npz");
    let frozen = PathBuf::from("../../eagle4/v2lite_frozen.npz");
    for (label, p) in [
        ("weights", &weights),
        ("profile", &profile),
        ("head NPZ", &head),
        ("frozen NPZ", &frozen),
    ] {
        assert!(
            p.exists(),
            "{} not found at {} — copy / regenerate before running",
            label,
            p.display()
        );
    }

    let n_tokens: usize = env::var("DISMANTLE_EAGLE4_GREEDY_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TOKENS);

    eprintln!(
        "[eagle4 greedy parity] {n_tokens} tokens × greedy × {{Off, Eagle4}}; prompt={:?}",
        PROMPT
    );

    let off_tokens = collect_tokens(
        &weights,
        &profile,
        SpeculateMode::Off,
        None,
        None,
        n_tokens,
    );
    let eagle4_tokens = collect_tokens(
        &weights,
        &profile,
        SpeculateMode::Eagle4,
        Some(&head),
        Some(&frozen),
        n_tokens,
    );

    eprintln!(
        "[eagle4 greedy parity] Off:    {}",
        off_tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    eprintln!(
        "[eagle4 greedy parity] Eagle4: {}",
        eagle4_tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    assert_eq!(
        off_tokens.len(),
        eagle4_tokens.len(),
        "Off and Eagle4 produced different token counts ({} vs {})",
        off_tokens.len(),
        eagle4_tokens.len()
    );
    let mut mismatches = Vec::new();
    for (i, (a, b)) in off_tokens.iter().zip(eagle4_tokens.iter()).enumerate() {
        if a != b {
            mismatches.push((i, *a, *b));
        }
    }
    assert!(
        mismatches.is_empty(),
        "Eagle4 greedy not bit-identical to Off: {} mismatch(es), first 5: {:?}",
        mismatches.len(),
        &mismatches[..mismatches.len().min(5)]
    );
}

#[test]
fn module_compiles() {
    assert_eq!(2 + 2, 4);
}
