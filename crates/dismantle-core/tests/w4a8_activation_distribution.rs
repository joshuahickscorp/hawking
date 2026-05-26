//! W4A8 activation distribution dumper — per `memory/w4a8_quality_redesign_2026_05_26.md`
//! Approach 0 (instrumentation-first).
//!
//! Dumps post-final-norm `x_norm` activation across many forward steps
//! on Qwen-3B, computes per-channel statistics, and answers ONE
//! question: **does the top 5% of channels carry >50% of the activation
//! variance / peak magnitude?**
//!
//! If yes, the outlier-channel hypothesis is confirmed and per-channel
//! W4A8 scaling (Approach A in the redesign memo) is the right next
//! attempt for unblocking W4A8 default-on.
//!
//! If no, the W4A8 quality failure is from a different mechanism (e.g.,
//! token-position-dependent activation drift, accumulated quant noise
//! across layers, etc.) and the redesign needs a different approach.
//!
//! Test is `#[ignore]` — loads Qwen-3B-Q4_K_M and runs ~50-100 forwards,
//! takes ~5 sec. Run via:
//!   cargo test --release -p dismantle-core --test w4a8_activation_distribution \
//!     -- --ignored --nocapture --test-threads=1

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use dismantle_core::{
    model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig,
};

const PROMPTS: &[&str] = &[
    "Briefly explain what a transformer attention head does.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Translate to French: I would like a cup of coffee and a croissant, please.",
    "List the first 10 prime numbers in order: 2, 3, 5, 7,",
    "What is the capital of Japan?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
];

const STEPS_PER_PROMPT: usize = 16;

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

#[test]
#[ignore]
fn w4a8_activation_distribution() {
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

    // Each prompt: tokenize, prefill via single-step calls, then dump
    // x_norm at each of STEPS_PER_PROMPT decoded positions.
    //
    // Total samples: PROMPTS.len() × STEPS_PER_PROMPT = ~96
    let mut samples: Vec<Vec<f32>> = Vec::with_capacity(PROMPTS.len() * STEPS_PER_PROMPT);

    for (pi, prompt) in PROMPTS.iter().enumerate() {
        eprintln!("[w4a8-act] prompt {}/{}: {}", pi + 1, PROMPTS.len(), prompt);
        // Reset KV cache so each prompt is independent.
        engine.kv.reset();

        let tokens = engine.tokenizer.encode(prompt, true).expect("encode");
        // Prefill: dump x_norm at each prompt position so we get a
        // diverse activation sample even during prompt processing.
        for (i, &tok) in tokens.iter().enumerate() {
            let x_norm = engine
                .dump_x_norm_after_forward(tok, i)
                .expect("forward prefill");
            samples.push(x_norm);
        }
        // Decode: run greedy continuation, dumping x_norm at each step.
        // dump_x_norm_after_forward returns x_norm but consumes the
        // forward (we don't get the predicted token back from it).
        // For diversity we just use a fixed proxy token to extend
        // the sequence — what matters is that we get N forwards at
        // different positions producing different activations.
        //
        // Note: this means we're not running "real" greedy decode;
        // we're just sampling x_norm at N decoded positions with
        // sometimes-noisy inputs. That's fine for activation
        // distribution measurement — the distribution shape doesn't
        // depend on whether we got the "right" next token.
        let mut last_tok = *tokens.last().unwrap();
        let pos_start = tokens.len();
        for step in 0..STEPS_PER_PROMPT {
            let pos = pos_start + step;
            let x_norm = engine
                .dump_x_norm_after_forward(last_tok, pos)
                .expect("forward decode");
            // Pick a deterministic next token from the activation
            // (cheap proxy: argmax of |x_norm|). The exact choice
            // doesn't matter — we just need a varying token id.
            let mut argmax_idx = 0u32;
            let mut argmax_val = 0.0f32;
            for (i, &v) in x_norm.iter().enumerate() {
                if v.abs() > argmax_val {
                    argmax_val = v.abs();
                    argmax_idx = i as u32;
                }
            }
            last_tok = argmax_idx % (engine.config.vocab_size as u32).max(1);
            samples.push(x_norm);
        }
    }

    let hidden = samples[0].len();
    let n_samples = samples.len();
    eprintln!(
        "\n[w4a8-act] collected {} samples × {} channels",
        n_samples, hidden
    );

    // Per-channel statistics:
    // - max_abs[c] = max over all samples of |x[c]|
    // - mean_abs[c] = mean over all samples of |x[c]|
    // - rms[c] = sqrt(mean(x[c]²))
    let mut max_abs = vec![0.0f32; hidden];
    let mut sum_sq = vec![0.0f64; hidden];
    let mut sum_abs = vec![0.0f64; hidden];

    for sample in &samples {
        for c in 0..hidden {
            let v = sample[c];
            let av = v.abs();
            if av > max_abs[c] {
                max_abs[c] = av;
            }
            sum_abs[c] += av as f64;
            sum_sq[c] += (v as f64) * (v as f64);
        }
    }

    // Global statistics.
    let total_max_abs: f32 = max_abs.iter().sum();
    let mean_of_max: f32 = total_max_abs / hidden as f32;
    let max_of_max: f32 = max_abs.iter().cloned().fold(0.0f32, f32::max);
    let min_of_max: f32 = max_abs.iter().cloned().fold(f32::INFINITY, f32::min);

    eprintln!(
        "\n[w4a8-act] global max_abs distribution:"
    );
    eprintln!("  channel-wise max|x|:");
    eprintln!("    min:  {:.4}", min_of_max);
    eprintln!("    mean: {:.4}", mean_of_max);
    eprintln!("    max:  {:.4}", max_of_max);
    eprintln!("    ratio max/mean: {:.2}×", max_of_max / mean_of_max);

    // Outlier test: sort channels by max_abs descending, check if top-K
    // accounts for disproportionate share of total.
    let mut sorted_max: Vec<(usize, f32)> =
        max_abs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    sorted_max.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let percentile_thresholds = [1, 5, 10, 25, 50];
    eprintln!("\n[w4a8-act] cumulative max_abs share by top-N% of channels:");
    for &p in &percentile_thresholds {
        let n_top = (hidden * p / 100).max(1);
        let sum_top: f32 = sorted_max.iter().take(n_top).map(|&(_, v)| v).sum();
        let share = sum_top / total_max_abs * 100.0;
        eprintln!("  top {:>2}% ({:>4} channels): {:.1}% of total max_abs", p, n_top, share);
    }

    // The HYPOTHESIS: top 5% of channels carry >50% of max_abs.
    let n_top_5 = (hidden * 5 / 100).max(1);
    let sum_top_5: f32 = sorted_max.iter().take(n_top_5).map(|&(_, v)| v).sum();
    let share_top_5 = sum_top_5 / total_max_abs * 100.0;

    eprintln!(
        "\n[w4a8-act] OUTLIER-CHANNEL HYPOTHESIS: top 5% of channels carry {:.1}% of max_abs",
        share_top_5
    );
    if share_top_5 > 50.0 {
        eprintln!("  → CONFIRMED. Per-channel W4A8 scaling (Approach A in redesign memo) is the right next attempt.");
    } else if share_top_5 > 30.0 {
        eprintln!("  → PARTIAL. Outliers exist but don't dominate; per-channel scaling would help but won't fully fix W4A8 quality.");
    } else {
        eprintln!("  → REFUTED. Channels are roughly uniform; W4A8 quality failure is from a different mechanism (token-position-dependent? accumulated quant noise?).");
    }

    // Also report top-10 outlier channel indices for the next session.
    eprintln!("\n[w4a8-act] top 10 outlier channels (idx: max_abs):");
    for &(idx, v) in sorted_max.iter().take(10) {
        eprintln!("  ch[{:>4}] = {:.4}", idx, v);
    }

    // Save raw stats to a file for later analysis.
    let mut report = String::new();
    report.push_str("# W4A8 activation distribution report\n");
    report.push_str(&format!("samples={} hidden={}\n", n_samples, hidden));
    report.push_str("\nchannel,max_abs,mean_abs,rms\n");
    for c in 0..hidden {
        let mean_a = sum_abs[c] / n_samples as f64;
        let rms = (sum_sq[c] / n_samples as f64).sqrt();
        report.push_str(&format!("{},{:.6},{:.6},{:.6}\n", c, max_abs[c], mean_a, rms));
    }
    let out_path = PathBuf::from("reports/w4a8_activation_dist.csv");
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&out_path, &report);
    eprintln!("\n[w4a8-act] full per-channel CSV written to {}", out_path.display());

    // ── Per-block vs per-channel reconstruction-error analysis ──────────────
    //
    // Now that we have 180 samples in memory, simulate both
    // quantization approaches and compare reconstruction RMS error.
    // This answers: "is per-channel actually worth the 1-week kernel
    // rebuild, OR does it only help a few outlier blocks?"
    //
    // Per-block: existing approach. scale_b = max(|x[b*256..(b+1)*256]|)/127
    // Per-channel: proposed. scale_c = max_over_samples(|x[c]|)/127
    //
    // For each sample, quantize then dequantize via both approaches.
    // Compute the L2 norm of the reconstruction error.
    eprintln!("\n[w4a8-act] reconstruction error analysis (per-block vs per-channel)");

    // Per-channel scales — computed from the global max_abs we already have.
    let per_channel_scale: Vec<f32> = max_abs
        .iter()
        .map(|&m| if m > 0.0 { m / 127.0 } else { 1.0 })
        .collect();

    let block_size = 256usize;
    let mut sum_sq_err_block = 0.0f64;
    let mut sum_sq_err_channel = 0.0f64;
    let mut sum_sq_signal = 0.0f64;
    let mut sum_abs_err_block_per_outlier_block = 0.0f64;
    let mut sum_abs_err_channel_per_outlier_block = 0.0f64;
    let mut outlier_block_elem_count = 0usize;

    // Identify which 256-blocks contain a super-outlier channel
    // (max_abs > 30 — captures the top 10 outliers in the dataset).
    let outlier_blocks: Vec<usize> = {
        let nb = hidden / block_size;
        (0..nb)
            .filter(|&b| {
                (b * block_size..(b + 1) * block_size)
                    .any(|c| max_abs[c] > 30.0)
            })
            .collect()
    };
    eprintln!(
        "  {} of {} blocks contain a super-outlier channel (max|x|>30)",
        outlier_blocks.len(),
        hidden / block_size
    );

    for sample in &samples {
        // Per-block quantize+dequantize.
        let nb = hidden / block_size;
        let mut recon_block = vec![0.0f32; hidden];
        for b in 0..nb {
            let lo = b * block_size;
            let hi = lo + block_size;
            let mut bmax = 0.0f32;
            for c in lo..hi {
                let a = sample[c].abs();
                if a > bmax { bmax = a; }
            }
            let scale = if bmax > 0.0 { bmax / 127.0 } else { 1.0 };
            let inv = 1.0 / scale;
            for c in lo..hi {
                let q = (sample[c] * inv).round().clamp(-127.0, 127.0) as i32;
                recon_block[c] = (q as f32) * scale;
            }
        }

        // Per-channel quantize+dequantize.
        let mut recon_channel = vec![0.0f32; hidden];
        for c in 0..hidden {
            let scale = per_channel_scale[c];
            let inv = 1.0 / scale;
            let q = (sample[c] * inv).round().clamp(-127.0, 127.0) as i32;
            recon_channel[c] = (q as f32) * scale;
        }

        // Accumulate errors.
        for c in 0..hidden {
            let signal = sample[c];
            let err_b = recon_block[c] - signal;
            let err_c = recon_channel[c] - signal;
            sum_sq_err_block += (err_b as f64) * (err_b as f64);
            sum_sq_err_channel += (err_c as f64) * (err_c as f64);
            sum_sq_signal += (signal as f64) * (signal as f64);
        }

        // Outlier-block focused analysis: errors only inside the
        // blocks that contain a super-outlier.
        for &b in &outlier_blocks {
            for c in b * block_size..(b + 1) * block_size {
                sum_abs_err_block_per_outlier_block +=
                    (recon_block[c] - sample[c]).abs() as f64;
                sum_abs_err_channel_per_outlier_block +=
                    (recon_channel[c] - sample[c]).abs() as f64;
                outlier_block_elem_count += 1;
            }
        }
    }

    let total_elems = (samples.len() * hidden) as f64;
    let rmse_block = (sum_sq_err_block / total_elems).sqrt();
    let rmse_channel = (sum_sq_err_channel / total_elems).sqrt();
    let signal_rms = (sum_sq_signal / total_elems).sqrt();

    eprintln!("\n[w4a8-act] global reconstruction RMSE (all blocks, all samples):");
    eprintln!("  signal RMS:        {:.4}", signal_rms);
    eprintln!("  per-block RMSE:    {:.4e}   ({:.4}% of signal RMS)", rmse_block, rmse_block / signal_rms as f64 * 100.0);
    eprintln!("  per-channel RMSE:  {:.4e}   ({:.4}% of signal RMS)", rmse_channel, rmse_channel / signal_rms as f64 * 100.0);
    eprintln!("  improvement:       {:.2}×  (per-channel is {:.2}× lower error)", rmse_block / rmse_channel, rmse_block / rmse_channel);

    let mae_block_outlier = sum_abs_err_block_per_outlier_block / outlier_block_elem_count as f64;
    let mae_channel_outlier = sum_abs_err_channel_per_outlier_block / outlier_block_elem_count as f64;
    eprintln!("\n[w4a8-act] reconstruction MAE on OUTLIER blocks only:");
    eprintln!("  per-block MAE:     {:.4e}", mae_block_outlier);
    eprintln!("  per-channel MAE:   {:.4e}", mae_channel_outlier);
    eprintln!("  improvement:       {:.2}× on outlier blocks", mae_block_outlier / mae_channel_outlier);

    let global_ratio = rmse_block / rmse_channel;
    if global_ratio > 3.0 {
        eprintln!("\n[w4a8-act] VERDICT: per-channel recovers significantly more activation precision.");
        eprintln!("  Recommendation: per-channel W4A8 kernel work is JUSTIFIED ({:.1}× lower error).", global_ratio);
    } else if global_ratio > 1.5 {
        eprintln!("\n[w4a8-act] VERDICT: per-channel modestly better.");
        eprintln!("  Recommendation: per-channel W4A8 is WORTH TRYING but expected quality gain is modest.");
    } else {
        eprintln!("\n[w4a8-act] VERDICT: per-channel barely beats per-block.");
        eprintln!("  Recommendation: skip per-channel; investigate other mechanisms.");
    }
}
