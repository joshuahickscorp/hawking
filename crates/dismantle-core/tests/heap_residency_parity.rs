//! Parity gate for the `MTLHeap`-backed weight residency POC.
//!
//! Loads Qwen-3B-Q4_K_M two ways:
//!   1. via the established `QwenDense::load` (Engine trait) path
//!   2. via the new `QwenDense::load_heap_resident` POC entry point
//!
//! Both runs do greedy decode on a fixed prompt and must emit
//! bit-identical token sequences. The two paths compute the same math
//! over the same bytes; the only difference is the allocator backing
//! each weight buffer (per-buffer `new_buffer_with_bytes` vs one
//! `MTLHeap`-suballocated `new_buffer`). Anything other than identical
//! tokens here is a memcpy / allocation bug in the heap module.
//!
//! Skip behavior: if the Qwen-3B GGUF can't be located under any of the
//! probed paths, the test prints a SKIP message and returns Ok. We do
//! NOT mark it `#[ignore]` — the brief asks for the test to be
//! invocable directly by `cargo test --release -p dismantle-core
//! --test heap_residency_parity`; the runtime skip keeps it harmless
//! in CI where the model isn't present.
//!
//! Run:
//!   cargo test --release -p dismantle-core --test heap_residency_parity -- --nocapture

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use dismantle_core::{model::qwen_dense::QwenDense, Engine, EngineConfig};

const PROMPT: &str = "The future of";
const MAX_NEW: usize = 16;

fn locate_weights() -> Option<PathBuf> {
    // Probe order:
    //   1. DISMANTLE_QWEN3B_GGUF env (escape hatch for arbitrary location)
    //   2. ../../models/... (worktree-root relative; matches qkv_concurrent_parity)
    //   3. ../../../models/... (build-heap is a nested worktree under
    //      dismantle/dismantle-build-heap/; tests run from crates/
    //      dismantle-core/, so 3 levels up reaches the main worktree
    //      where the GGUF actually lives)
    //   4. absolute fallback the test author knows about
    if let Ok(p) = std::env::var("DISMANTLE_QWEN3B_GGUF") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let candidates = [
        "../../models/qwen2.5-3b-instruct-q4_k_m.gguf",
        "../../../models/qwen2.5-3b-instruct-q4_k_m.gguf",
        "/Users/scammermike/Downloads/dismantle/models/qwen2.5-3b-instruct-q4_k_m.gguf",
    ];
    for c in candidates.iter() {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn run_path(weights: &Path, via_heap: bool) -> Vec<u32> {
    // Use the same flag set the production decode path uses (greedy
    // TCB). We deliberately do NOT enable the optional opt-ins
    // (vocab-prune, Q4_K LM-head, FFN-down requant) — the brief notes
    // those are out of scope for the POC.
    std::env::set_var("DISMANTLE_QWEN_TCB", "1");
    std::env::remove_var("DISMANTLE_QWEN_VOCAB_PRUNE_CORPUS");
    std::env::remove_var("DISMANTLE_QWEN_VOCAB_PRUNE");
    std::env::remove_var("DISMANTLE_QWEN_Q4K_LMHEAD");
    std::env::remove_var("DISMANTLE_QWEN_FFN_DOWN_Q4K");
    std::env::remove_var("DISMANTLE_QWEN_CONCURRENT_QKV");
    std::env::remove_var("DISMANTLE_QWEN_W4A8");

    let cfg = EngineConfig::default();
    let mut engine = if via_heap {
        QwenDense::load_heap_resident(weights, cfg).expect("load_heap_resident")
    } else {
        <QwenDense as Engine>::load(weights, cfg).expect("Engine::load")
    };

    let prompt_ids = engine
        .tokenizer
        .encode(PROMPT, true)
        .expect("encode prompt");
    assert!(
        prompt_ids.len() >= 2,
        "tokenizer returned empty/short prompt: {:?}",
        prompt_ids
    );

    // Prefill via the same TCB path used in production.
    for (i, &t) in prompt_ids.iter().enumerate() {
        let _ = engine
            .forward_token_greedy_tcb(t, i)
            .expect("prefill forward");
    }

    // Greedy decode loop.
    let mut tokens = Vec::with_capacity(MAX_NEW);
    let mut last = *prompt_ids.last().unwrap();
    for step in 0..MAX_NEW {
        let pos = prompt_ids.len() + step;
        let next = engine
            .forward_token_greedy_tcb(last, pos)
            .expect("decode forward");
        tokens.push(next);
        last = next;
    }
    tokens
}

#[test]
fn heap_residency_greedy_parity_16tok() {
    let weights = match locate_weights() {
        Some(p) => p,
        None => {
            eprintln!(
                "SKIP heap_residency_greedy_parity_16tok: \
                 qwen2.5-3b-instruct-q4_k_m.gguf not found in any probed location"
            );
            return;
        }
    };
    eprintln!(
        "[heap_residency_parity] using weights at {}",
        weights.display()
    );

    let baseline = run_path(&weights, false);
    let heap_run = run_path(&weights, true);

    eprintln!(
        "[heap_residency_parity] baseline tokens = {:?}",
        baseline
    );
    eprintln!(
        "[heap_residency_parity] heap     tokens = {:?}",
        heap_run
    );

    let first_div = baseline
        .iter()
        .zip(&heap_run)
        .position(|(a, b)| a != b)
        .unwrap_or(MAX_NEW);
    eprintln!(
        "[heap_residency_parity] first divergence at index {first_div} (need {MAX_NEW})"
    );

    assert_eq!(
        baseline, heap_run,
        "heap-resident decode diverged from baseline"
    );
}
