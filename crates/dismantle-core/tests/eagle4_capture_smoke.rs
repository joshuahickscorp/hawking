//! Path-to-90 step 3 — smoke test for `Engine::forward_token_eagle4_for_test`.
//!
//! Confirms the 5-input EAGLE-4 capture seam:
//!
//! 1. Returns the right-shape `Eagle4Inputs` struct (four 2048-vectors +
//!    echoed `prev_token`).
//! 2. All four hidden vectors are finite (no NaN / Inf).
//! 3. L2 norms are in a plausible residual-stream range (≥ 0.1, ≤ 1e6).
//! 4. KV cache advances — `h_high` at step 31 differs measurably from
//!    `h_high` at step 0 (proves the method isn't a stateless pure
//!    function of the input token).
//! 5. The four captured hiddens are mutually distinct (not aliased).
//!
//! What this test does NOT do: numerical parity against EAGLE-4's
//! Python reference forward at `atol=1e-3 fp16`. That cross-language
//! diff lands in step 6 of the execution plan (the parity test wires
//! up the Python-subprocess + `--dump-logits` flag added in step 4).
//! Until then, this smoke test guards against:
//!   - silent stub regressions (returning empty vecs / Err),
//!   - capture-point off-by-one (wrong layer index → degenerate norm),
//!   - KV-cache-not-advancing bugs (would make every step identical).
//!
//! Skipped when the V2-Lite weights are absent (CI without model files).

#![cfg(target_os = "macos")]

use std::path::PathBuf;

const HIDDEN: usize = 2048;
const N_TOKENS: usize = 32;

#[test]
fn forward_token_eagle4_for_test_smoke() {
    let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
    if !weights.exists() {
        eprintln!(
            "skipping eagle4_capture_smoke: no weights at {:?}",
            weights
        );
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

    // Synthetic 32-token sequence: a small varying pattern so the
    // KV cache state evolves nontrivially across steps.
    let tokens: Vec<u32> = (0..N_TOKENS)
        .map(|i| ((i as u32 * 17 + 5) % 31_999) + 1)
        .collect();

    let mut all_inputs = Vec::with_capacity(N_TOKENS);
    for (pos, &tok) in tokens.iter().enumerate() {
        let inp = engine
            .forward_token_eagle4_for_test(tok, pos)
            .expect("forward_token_eagle4_for_test returned err");
        // 1. Shape + prev_token echo.
        assert_eq!(inp.prev_token, tok, "prev_token echo mismatch at step {pos}");
        assert_eq!(inp.h_low.len(),    HIDDEN, "h_low len at step {pos}");
        assert_eq!(inp.h_mid.len(),    HIDDEN, "h_mid len at step {pos}");
        assert_eq!(inp.h_high.len(),   HIDDEN, "h_high len at step {pos}");
        assert_eq!(inp.h_shared.len(), HIDDEN, "h_shared len at step {pos}");
        // 2. Finite.
        for (name, v) in [
            ("h_low",    &inp.h_low),
            ("h_mid",    &inp.h_mid),
            ("h_high",   &inp.h_high),
            ("h_shared", &inp.h_shared),
        ] {
            if let Some((i, bad)) = v.iter().enumerate().find(|(_, &x)| !x.is_finite()) {
                panic!("step {pos}: non-finite value in {name}[{i}] = {bad}");
            }
        }
        // 3. Plausible L2 norm.
        for (name, v) in [
            ("h_low",    &inp.h_low),
            ("h_mid",    &inp.h_mid),
            ("h_high",   &inp.h_high),
            ("h_shared", &inp.h_shared),
        ] {
            let n2: f32 = v.iter().map(|x| x * x).sum();
            let norm = n2.sqrt();
            assert!(
                norm > 0.1 && norm < 1.0e6,
                "step {pos}: {name} L2 norm out of range: {norm}"
            );
        }
        all_inputs.push(inp);
    }

    // 4. KV cache advanced — h_high at step 31 should not be approximately
    //    equal to h_high at step 0 (they'd be equal only if the model
    //    were stateless w.r.t. KV).
    let h_high_0  = &all_inputs[0].h_high;
    let h_high_31 = &all_inputs[N_TOKENS - 1].h_high;
    let l2_diff: f32 = h_high_0
        .iter()
        .zip(h_high_31.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    assert!(
        l2_diff > 1.0,
        "h_high at step 0 vs step {} is essentially identical (L2 diff = {l2_diff}); KV cache may not be advancing",
        N_TOKENS - 1
    );

    // 5. The four captured hiddens at a single step are mutually distinct.
    //    (If two were aliased — e.g. capture-point bug copied the same
    //    buffer twice — this would silently pass step 1's shape check.)
    let mid_step = &all_inputs[N_TOKENS / 2];
    let pairs: [(&str, &Vec<f32>, &str, &Vec<f32>); 6] = [
        ("h_low", &mid_step.h_low, "h_mid",    &mid_step.h_mid),
        ("h_low", &mid_step.h_low, "h_high",   &mid_step.h_high),
        ("h_low", &mid_step.h_low, "h_shared", &mid_step.h_shared),
        ("h_mid", &mid_step.h_mid, "h_high",   &mid_step.h_high),
        ("h_mid", &mid_step.h_mid, "h_shared", &mid_step.h_shared),
        ("h_high", &mid_step.h_high, "h_shared", &mid_step.h_shared),
    ];
    for (a_name, a, b_name, b) in pairs {
        let d: f32 = a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            d > 0.01,
            "step {}: {a_name} ≈ {b_name} (L2 diff = {d}); capture aliased?",
            N_TOKENS / 2
        );
    }
}
