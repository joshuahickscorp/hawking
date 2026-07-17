//! Parity scaffold for RWKV-7 TQ (trellis-quant) vs Q4_K_M and vs Metal decode.
//!
//! All tests in this file are `#[ignore]` stubs — they hold the signatures and
//! the env-var loading pattern for the three parity gates that must pass before
//! TQ can be enabled in the RWKV-7 serving path:
//!
//! 1. CPU greedy trajectory matches Metal greedy trajectory token-for-token.
//! 2. TQ perplexity is within the "silver band" of Q4_K_M (≤ +0.3 PPL on the
//!    calibration set).
//! 3. Two runs with identical seed produce identical greedy output (determinism
//!    gate, required before any A/B bench is meaningful).
//!
//! Enable a test by setting the required env vars and passing `--ignored`:
//!
//! ```sh
//! RWKV7_TQ_MODEL=/path/to/model.tq \
//! RWKV7_Q4K_MODEL=/path/to/model.gguf \
//!   cargo test -p hawking-core --features tq --test rwkv7_tq_parity -- --nocapture --ignored
//! ```

#![cfg(feature = "tq")]
#![allow(dead_code)]

/// Prompts used as fixtures across all parity tests. Short enough to run in
/// seconds on a dev machine, varied enough to exercise both the attention branch
/// and the channel-mix branch across multiple layer types.
const FIXTURE_PROMPTS: [&str; 3] = [
    "The quick brown fox jumps over the lazy dog.",
    "In mathematics, a prime number is a natural number greater than 1",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
];

/// Compare RWKV-7 TQ CPU greedy output against Metal (GPU) greedy output
/// token-for-token for each fixture prompt.
///
/// Requires:
/// - `RWKV7_TQ_MODEL`: path to the `.tq` artifact.
/// - `RWKV7_TOKENIZER`: path to the RWKV World tokenizer vocab file.
///
/// Pass length: 32 tokens per prompt (enough to surface divergence without
/// slow CI wall-clock).
#[test]
#[ignore = "stub — implement after TqPreparedGpu dispatch and CPU serving reference are wired"]
fn rwkv7_tq_cpu_vs_metal_greedy_trajectory() {
    let model_path = std::env::var("RWKV7_TQ_MODEL").expect("RWKV7_TQ_MODEL must be set");
    let _tokenizer_path = std::env::var("RWKV7_TOKENIZER")
        .unwrap_or_else(|_| "tokenizer/rwkv_vocab_v20230424.txt".to_string());

    println!("STUB: not yet implemented");
    println!("  model_path = {model_path}");
    println!("  fixture_prompts = {}", FIXTURE_PROMPTS.len());
    println!("  Would compare CPU vs Metal greedy trajectories for each prompt.");
    println!("  Wire hawking_core::model::rwkv7 CPU and Metal forward paths, then");
    println!("  assert token-for-token equality for 32-token generations.");
}

/// Verify that TQ perplexity on the calibration set is within the silver band
/// of Q4_K_M: `ppl_tq <= ppl_q4k + 0.3`.
///
/// Requires:
/// - `RWKV7_TQ_MODEL`: path to the `.tq` artifact.
/// - `RWKV7_Q4K_MODEL`: path to the baseline Q4_K_M GGUF.
/// - `RWKV7_CALIB_CORPUS`: path to calibration corpus (JSON lines of text).
///
/// N=100 sequences from the calibration corpus, max 512 tokens each.
#[test]
#[ignore = "stub — implement after TQ serving reference is wired and calibration corpus is available"]
fn rwkv7_tq_vs_q4k_ppl_within_silver() {
    let tq_path = std::env::var("RWKV7_TQ_MODEL").expect("RWKV7_TQ_MODEL must be set");
    let q4k_path = std::env::var("RWKV7_Q4K_MODEL").expect("RWKV7_Q4K_MODEL must be set");
    let _corpus_path = std::env::var("RWKV7_CALIB_CORPUS").expect("RWKV7_CALIB_CORPUS must be set");

    println!("STUB: not yet implemented");
    println!("  tq_path   = {tq_path}");
    println!("  q4k_path  = {q4k_path}");
    println!("  Would compute PPL on N=100 corpus sequences and assert:");
    println!("    ppl_tq <= ppl_q4k + 0.3  (silver gate)");
}

/// Verify that two greedy decode runs with the same model and the same prompt
/// produce identical token sequences (determinism gate).
///
/// Requires:
/// - `RWKV7_TQ_MODEL`: path to the `.tq` artifact.
///
/// Run 3 times per fixture prompt, compare all pairs. Length: 64 tokens.
#[test]
#[ignore = "stub — implement after TqPreparedGpu dispatch is wired"]
fn rwkv7_tq_deterministic_across_runs() {
    let model_path = std::env::var("RWKV7_TQ_MODEL").expect("RWKV7_TQ_MODEL must be set");

    println!("STUB: not yet implemented");
    println!("  model_path = {model_path}");
    println!("  Would run each fixture prompt 3 times and assert identical 64-token outputs.");
    println!("  Failure here means the kernel has non-deterministic memory reads or");
    println!("  the Metal pipeline cache is not stable across command buffer submissions.");
}
