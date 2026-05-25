//! W4A8 per-channel calibration — LM_HEAD site only.
//!
//! Produces a calibrated per-channel scale vector at the post-final-norm
//! activation site (input to the lm_head Q4_K projection). The scale at
//! channel c is `max|x_c| / 127`, taken over a corpus of N prompts × M
//! decoded tokens.
//!
//! Output (gitignored): `reports/w4a8_lmhead_calibration_2026_05_26.json`
//! with schema:
//!   {
//!     "model": "qwen2.5-3b-instruct-q4_k_m",
//!     "site": "lm_head_input_post_final_norm",
//!     "hidden": 2048,
//!     "n_samples": <int>,
//!     "n_prompts": <int>,
//!     "max_abs_per_channel": [f32; hidden],
//!     "scales_per_channel":  [f32; hidden]   // = max_abs[c] / 127
//!   }
//!
//! This is the BASIC calibration needed to wire per-channel W4A8 at
//! the LM_HEAD site only. The other 6 W4A8 sites (q/k/v/o/ffn_gate/
//! ffn_up/ffn_down × 36 layers) need separate calibration; that's a
//! follow-up requiring per-site capture hooks in qwen_dense.rs.
//!
//! Test is `#[ignore]` — loads Qwen-3B-Q4_K_M and runs ~250 forwards,
//! ~10 sec wall. Run via:
//!   cargo test --release -p dismantle-core --test w4a8_per_channel_calibrate \
//!     -- --ignored --nocapture --test-threads=1

#![cfg(target_os = "macos")]

use std::fs;
use std::path::PathBuf;

use dismantle_core::{
    model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig,
};

/// 12 diverse prompts × 20 steps = 240 samples. Tighter outlier
/// estimates than the 6×16=96 used in `w4a8_activation_distribution`
/// because the calibration scales the production W4A8 path directly
/// (any under-estimation of max|x| → saturation → quality regression).
const PROMPTS: &[&str] = &[
    "Briefly explain what a transformer attention head does.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Translate to French: I would like a cup of coffee and a croissant, please.",
    "List the first 10 prime numbers in order: 2, 3, 5, 7,",
    "What is the capital of Japan?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Explain how RAM differs from disk storage in one paragraph.",
    "What year did the Berlin Wall fall and what was its significance?",
    "Provide a recipe for pancakes with measurements in cups and tablespoons.",
    "Describe the difference between mitosis and meiosis in three points.",
    "Write a haiku about an autumn evening.",
    "What does the word 'quintessential' mean? Use it in a sentence.",
];

const STEPS_PER_PROMPT: usize = 20;

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

fn report_path() -> PathBuf {
    PathBuf::from("../../reports/w4a8_lmhead_calibration_2026_05_26.json")
}

#[test]
#[ignore]
fn w4a8_per_channel_calibrate_lmhead() {
    let weights = weights_path();
    if !weights.exists() {
        eprintln!("skip: weights missing at {:?}", weights);
        return;
    }

    let profile = fresh_test_profile(&weights).expect("fresh test profile");
    let cfg = EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    let mut engine = QwenDense::load(&weights, cfg).expect("load engine");
    let hidden = engine.config.hidden;

    let mut max_abs = vec![0.0f32; hidden];
    let mut n_samples = 0usize;

    for (pi, prompt) in PROMPTS.iter().enumerate() {
        eprintln!(
            "[calibrate] prompt {}/{}: {}",
            pi + 1,
            PROMPTS.len(),
            prompt
        );
        engine.kv.reset();
        let tokens = engine.tokenizer.encode(prompt, true).expect("encode");

        // Prefill: dump x_norm at each prompt position
        for (i, &tok) in tokens.iter().enumerate() {
            let x_norm = engine
                .dump_x_norm_after_forward(tok, i)
                .expect("forward prefill");
            for (c, &v) in x_norm.iter().enumerate() {
                let a = v.abs();
                if a > max_abs[c] {
                    max_abs[c] = a;
                }
            }
            n_samples += 1;
        }

        // Decode: STEPS_PER_PROMPT additional samples
        let mut last_tok = *tokens.last().unwrap();
        let pos_start = tokens.len();
        for step in 0..STEPS_PER_PROMPT {
            let pos = pos_start + step;
            let x_norm = engine
                .dump_x_norm_after_forward(last_tok, pos)
                .expect("forward decode");
            // proxy next token = argmax|x_norm| % vocab (same as activation dumper)
            let mut argmax_idx = 0u32;
            let mut argmax_val = 0.0f32;
            for (i, &v) in x_norm.iter().enumerate() {
                let a = v.abs();
                if a > argmax_val {
                    argmax_val = a;
                    argmax_idx = i as u32;
                }
            }
            for (c, &v) in x_norm.iter().enumerate() {
                let a = v.abs();
                if a > max_abs[c] {
                    max_abs[c] = a;
                }
            }
            n_samples += 1;
            last_tok = argmax_idx % (engine.config.vocab_size as u32).max(1);
        }
    }

    // Compute per-channel scales (clamped >0 so quantize is well-defined
    // for never-seen-active channels). Add a 5% safety margin per
    // SmoothQuant practice — runtime tokens can produce magnitudes
    // slightly above calibration max; saturating clamps to ±127 are
    // recoverable but underflow to 0 is not.
    let safety = 1.05_f32;
    let mut scales = vec![0.0f32; hidden];
    let mut max_seen = 0.0f32;
    let mut min_nonzero = f32::INFINITY;
    for c in 0..hidden {
        let m = (max_abs[c] * safety).max(1e-6);
        scales[c] = m / 127.0;
        if max_abs[c] > max_seen {
            max_seen = max_abs[c];
        }
        if max_abs[c] > 0.0 && max_abs[c] < min_nonzero {
            min_nonzero = max_abs[c];
        }
    }

    // Top 10 outlier channels for memo cross-check
    let mut idx: Vec<usize> = (0..hidden).collect();
    idx.sort_by(|&a, &b| {
        max_abs[b]
            .partial_cmp(&max_abs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("\n[calibrate] n_samples={} hidden={}", n_samples, hidden);
    eprintln!(
        "[calibrate] max_abs: max={:.2} min_nonzero={:.4} mean={:.2}",
        max_seen,
        min_nonzero,
        max_abs.iter().sum::<f32>() / hidden as f32
    );
    eprintln!("[calibrate] top 10 outlier channels:");
    for &c in idx.iter().take(10) {
        eprintln!("  ch[{:4}] = {:7.2}  scale={:.4}", c, max_abs[c], scales[c]);
    }

    // Write JSON (raw, hand-rolled — serde_json isn't a workspace dep
    // here and we don't want to add one for one file).
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str("  \"model\": \"qwen2.5-3b-instruct-q4_k_m\",\n");
    json.push_str("  \"site\": \"lm_head_input_post_final_norm\",\n");
    json.push_str(&format!("  \"hidden\": {hidden},\n"));
    json.push_str(&format!("  \"n_samples\": {n_samples},\n"));
    json.push_str(&format!("  \"n_prompts\": {},\n", PROMPTS.len()));
    json.push_str(&format!("  \"steps_per_prompt\": {STEPS_PER_PROMPT},\n"));
    json.push_str(&format!("  \"safety_margin\": {safety},\n"));
    json.push_str("  \"max_abs_per_channel\": [");
    for (i, v) in max_abs.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!("{:.6}", v));
    }
    json.push_str("],\n  \"scales_per_channel\": [");
    for (i, v) in scales.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!("{:.8e}", v));
    }
    json.push_str("]\n}\n");

    let out = report_path();
    if let Some(parent) = out.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&out, json).expect("write calibration JSON");
    eprintln!("[calibrate] wrote {}", out.display());

    // Sanity: confirm at least one outlier channel was detected. If
    // max_seen < 10 the model probably failed to load tokens.
    assert!(
        max_seen > 5.0,
        "calibration max_abs too low ({max_seen:.2}) — expected outlier channel >5 magnitude"
    );
}
