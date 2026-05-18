//! Diagnostic A/B test for the CPU-walk vs GPU Wedge C divergence
//! documented in `reports/path_to_90/foundation_halt.md`.
//!
//! Runs both forwards on the SAME single token at pos=0 with reset
//! KV cache. Compares:
//!   - forward_token (GPU Wedge C): canonical V2-Lite forward
//!   - forward_token_eagle4_capture_with_argmax (CPU walk): the path
//!     step 8 uses for Eagle4 spec decode
//!
//! If both agree on argmax → divergence is in KV state management
//! across prefill→decode transitions. If they differ → divergence is
//! within the per-step forward itself.
//!
//! Gating: `#[ignore]`'d unless `EAGLE4_PARITY_TEST=1` is set. Needs
//! V2-Lite weights at `models/deepseek-v2-lite-q4.gguf`.

#![cfg(target_os = "macos")]

use dismantle_core::{
    profile::KernelProfile, EngineConfig, GenerateRequest, SamplingParams, SpeculateMode,
    StreamEvent,
};
use std::env;
use std::path::PathBuf;

#[test]
#[ignore = "requires V2-Lite weights + EAGLE4_PARITY_TEST=1"]
fn cpu_walk_vs_gpu_wedge_c_same_pos_zero() {
    if env::var("EAGLE4_PARITY_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping cpu/gpu A/B; set EAGLE4_PARITY_TEST=1");
        return;
    }

    let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
    let profile_path = PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
    assert!(weights.exists(), "weights missing");
    assert!(profile_path.exists(), "profile missing");

    // We need access to the inherent methods on DeepSeekV2 — load through
    // the dynamic Engine trait so both forward_token_with_hidden_for_test
    // (GPU path) and a CPU-walk equivalent are reachable.
    let profile = KernelProfile::load(&profile_path).expect("load profile");
    let cfg = EngineConfig {
        kernel_profile: Some(profile.clone()),
        ..Default::default()
    };
    let mut engine = dismantle_core::model::load_engine(&weights, cfg)
        .expect("load engine for A/B");

    // Some BOS-ish token id. Pick something common; the specific token
    // doesn't matter — we want to compare GPU vs CPU argmax.
    let test_token = 1u32;
    let test_pos = 0usize;

    // Path A — GPU Wedge C via forward_token_with_hidden_for_test (which
    // wraps forward_token + argmax).
    engine.reset_kv_for_test();
    let (gpu_hidden, gpu_argmax) = engine
        .forward_token_with_hidden_for_test(test_token, test_pos)
        .expect("GPU forward_token_with_hidden_for_test");
    let gpu_norm: f32 = gpu_hidden.iter().map(|v| v * v).sum::<f32>().sqrt();

    // Path B — CPU walk via forward_token_eagle4_for_test, which returns
    // the Eagle4Inputs bundle (we'll inspect h_high vs the GPU
    // post-final-norm hidden for sanity).
    engine.reset_kv_for_test();
    let eagle4_inputs = engine
        .forward_token_eagle4_for_test(test_token, test_pos)
        .expect("CPU walk forward_token_eagle4_for_test");
    let cpu_h_high_norm: f32 = eagle4_inputs
        .h_high
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();
    let cpu_h_low_norm: f32 = eagle4_inputs
        .h_low
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();
    let cpu_h_mid_norm: f32 = eagle4_inputs
        .h_mid
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();
    let cpu_h_shared_norm: f32 = eagle4_inputs
        .h_shared
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();

    // Path C — shared-only CPU walk: same attention() helper as Path B,
    // but ffn_shared_only (currently zero-output for shared-expert leg,
    // per step 3's latent-bug finding). Compares CPU attention()'s
    // contribution in isolation; ffn() effectively neutralized.
    engine.reset_kv_for_test();
    let shared_only_logits = engine
        .forward_token_shared_only_for_test(test_token, test_pos)
        .expect("forward_token_shared_only_for_test");
    let shared_only_argmax = dismantle_core::kernels::argmax_f32(&shared_only_logits);
    let shared_only_l2: f32 = shared_only_logits
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();

    eprintln!(
        "[A/B] pos=0 token={test_token}\n  \
         GPU Wedge C       | argmax={gpu_argmax:>6}  post-norm-hidden L2={gpu_norm:.4}\n  \
         CPU walk (full)   | h_low  L2={cpu_h_low_norm:.4}  h_mid L2={cpu_h_mid_norm:.4}\n  \
         CPU walk (full)   | h_high L2={cpu_h_high_norm:.4}  h_shared L2={cpu_h_shared_norm:.4}\n  \
         CPU shared_only   | argmax={shared_only_argmax:>6}  logits L2={shared_only_l2:.4}\n\n  \
         → if GPU argmax == shared_only argmax: CPU attention() agrees with GPU; divergence is in ffn() / MoE\n  \
         → if GPU argmax != shared_only argmax: divergence is already in CPU attention()"
    );

    // The CPU walk doesn't currently return the V2-Lite argmax through
    // the public Engine trait — the inherent
    // forward_token_eagle4_capture_with_argmax does, but it's not on
    // the trait. So this A/B test surfaces the hidden norms; matching
    // argmax requires either exposing the inherent method via the trait
    // or comparing post-final-norm hiddens directly.
    //
    // If h_high (CPU layer 25 output) is wildly different from
    // gpu_hidden (GPU post-final-norm-of-layer-26), the divergence is
    // already visible at the residual-stream level. Note these aren't
    // directly comparable (h_high is pre-final-norm at layer 25; GPU
    // hidden is post-final-norm at layer 26) but their L2 should be
    // within an order of magnitude on a well-trained model.

    // Generate using both modes for one step to compare emitted tokens.
    let req = GenerateRequest {
        prompt: "Hello".to_string(),
        max_new_tokens: 1,
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
    };

    let mut off_cfg = EngineConfig {
        kernel_profile: Some(profile.clone()),
        speculate_mode: SpeculateMode::Off,
        ..Default::default()
    };
    off_cfg.speculate = false;
    let mut off_engine =
        dismantle_core::model::load_engine(&weights, off_cfg).expect("load Off engine");
    let mut off_tok: Option<u32> = None;
    off_engine
        .generate(req.clone(), &mut |ev| {
            if let StreamEvent::Token { id, .. } = ev {
                off_tok = Some(id);
            }
        })
        .expect("Off generate");

    eprintln!("[A/B] Off generate one-token: {:?}", off_tok);
}
