#![cfg(feature = "tq")]
#![allow(dead_code)]
const FIXTURE_PROMPTS: [&str; 3] = [
    "The quick brown fox jumps over the lazy dog.",
    "In mathematics, a prime number is a natural number greater than 1",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
];
#[test]
#[ignore = "stub — implement after TqPreparedGpu dispatch and CPU serving reference are wired"]
fn rwkv7_tq_cpu_vs_metal_greedy_trajectory() {
    let model_path = std::env::var("RWKV7_TQ_MODEL").expect("RWKV7_TQ_MODEL must be set");
    let _tokenizer_path = std::env::var("RWKV7_TOKENIZER")
        .unwrap_or_else(|_| "tokenizer/rwkv_vocab_v20230424.txt".to_string());
}
#[test]
#[ignore = "stub — implement after TQ serving reference is wired and calibration corpus is available"]
fn rwkv7_tq_vs_q4k_ppl_within_silver() {
    let tq_path = std::env::var("RWKV7_TQ_MODEL").expect("RWKV7_TQ_MODEL must be set");
    let q4k_path = std::env::var("RWKV7_Q4K_MODEL").expect("RWKV7_Q4K_MODEL must be set");
    let _corpus_path = std::env::var("RWKV7_CALIB_CORPUS").expect("RWKV7_CALIB_CORPUS must be set");
}
#[test]
#[ignore = "stub — implement after TqPreparedGpu dispatch is wired"]
fn rwkv7_tq_deterministic_across_runs() {
    let model_path = std::env::var("RWKV7_TQ_MODEL").expect("RWKV7_TQ_MODEL must be set");
}
