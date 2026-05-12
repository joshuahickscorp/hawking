//! Smoke test: forward_token with activation_dtype=F16 produces at least one
//! token and does not panic. Output drift vs the F32 path is expected and
//! accepted in v0.8.3; correctness is pinned in v0.8.4.
#![cfg(target_os = "macos")]

use std::path::PathBuf;

#[test]
fn forward_token_f16_doesnt_panic() {
    let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
    if !weights.exists() {
        eprintln!("skipping: no weights at {:?}", weights);
        return;
    }
    let profile_path =
        PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
    let profile = dismantle_core::profile::KernelProfile::load(&profile_path)
        .expect("load profile");
    let cfg = dismantle_core::EngineConfig {
        kernel_profile: Some(profile),
        activation_dtype: dismantle_core::ActivationDtype::F16,
        ..Default::default()
    };
    let mut engine =
        dismantle_core::model::load_engine(&weights, cfg).expect("load engine");

    let req = dismantle_core::GenerateRequest {
        prompt: "Once upon".into(),
        max_new_tokens: 4,
        sampling: dismantle_core::SamplingParams {
            temperature: 0.0,
            seed: Some(42),
            ..Default::default()
        },
        stop: vec![],
        abort: None,
        max_stall_ms: 0,
    };

    let mut got_tokens = 0usize;
    engine
        .generate(req, &mut |ev| {
            if matches!(ev, dismantle_core::StreamEvent::Token { .. }) {
                got_tokens += 1;
            }
        })
        .expect("generate did not error");

    assert!(got_tokens >= 1, "produced no tokens");
}
