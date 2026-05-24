//! 2-layer megakernel parity test — POC skeleton (2026-05-25, build/megakernel).
//!
//! GATED OFF by default — the test compiles but is `#[ignore]`d because the
//! megakernel dispatcher is a skeleton. See
//! `~/.claude/projects/-Users-scammermike-Downloads-dismantle/memory/build_megakernel_design_2026_05_25.md`
//! § "What attended work unblocks" for the path to making this functional.
//!
//! Intended flow once unblocked:
//!   1. Load Qwen-3B from disk (skip if model missing).
//!   2. Call `QwenDense::forward_layers_subset(token, pos, last_layer=1)` —
//!      a new debug-only API on QwenDense that runs only the first
//!      `last_layer+1` layers via the existing forward_token_greedy_tcb
//!      path, returning the residual stream (x_norm after layer 1).
//!   3. Pre-dequantize layer 0 + layer 1 weights from Q4_K → f16.
//!   4. Call megakernel_2layer_dispatch with the same inputs.
//!   5. Compare both residual streams.
//!   6. Assert max abs diff < 1e-3 fp32 (memo § "Verification rule"
//!      relaxation — fp16 noise inside the megakernel makes bit-identical
//!      unrealistic).

#![cfg(target_os = "macos")]

#[test]
#[ignore = "megakernel POC: dispatcher is skeleton only; see build_megakernel_design_2026_05_25.md"]
fn megakernel_2layer_parity_qwen3b() {
    // Skip gracefully if model isn't present.
    let model_path = std::env::var("DISMANTLE_QWEN_GGUF")
        .ok()
        .unwrap_or_else(|| "models/qwen2.5-3b-instruct-q4_k_m.gguf".to_string());
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("SKIP: model not at {model_path}");
        return;
    }

    // TODO(megakernel-poc): once forward_layers_subset lands, replace this
    // panic with the real flow described in the module doc.
    panic!(
        "megakernel_2layer_parity_qwen3b: not yet runnable — \
         forward_layers_subset() and megakernel_2layer_dispatch() are stubs"
    );
}
