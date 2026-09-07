//! Greedy identity: sequential per-token prefill vs batched GEMM prefill.
//!
//! Requires the sealed-3.14 artifact. Skips when it is absent so
//! `cargo test -p hawking-core` stays green on machines without the body.
//!
//! Mutation check: `batched_prefill` must be true and prefill dispatches must
//! drop versus the sequential path. With `HAWKING_QWEN38_BATCH_PREFILL=0` the
//! identity comparison would pass vacuously (both arms are `step()`); the
//! flag + dispatch assertions fail that cheat.

#![cfg(target_os = "macos")]

use hawking_core::model::qwen38_hybrid_decode::{
    generate_greedy, load_qwen38_tokenizer, Qwen38HybridDecodeSession, Qwen38HybridWeights,
};
use std::sync::Arc;

const ARTIFACT: &str = "/Users/scammermike/noetic/NOETIC_PARENT_A";
const TOKENIZER: &str = "/Users/scammermike/noetic/NOETIC_PARENT_A/tokenizer.json";
const PROMPT: &str = "The capital of France is";
const MAX_NEW: usize = 64;
const MAX_SEQ: usize = 2048;

fn fusion_env() {
    std::env::set_var("HAWKING_QWEN38_FUSE_ADD_RMSNORM", "1");
    std::env::set_var("HAWKING_QWEN38_FUSE_GQA_QKV", "1");
    std::env::set_var("HAWKING_QWEN38_FUSE_DN_INPROJ", "1");
    std::env::set_var("HAWKING_QWEN38_FUSE_MLP", "swiglu");
    std::env::set_var("HAWKING_QWEN38_IGNORE_EOS", "1");
}

fn artifact_present() -> bool {
    std::path::Path::new(ARTIFACT)
        .join("catalog.hq38m20")
        .is_file()
        && std::path::Path::new(TOKENIZER).is_file()
}

#[test]
fn batched_prefill_greedy_64_matches_sequential() {
    if !artifact_present() {
        eprintln!("skipping qwen38_batched_prefill_greedy: artifact missing at {ARTIFACT}");
        return;
    }
    if hawking_core::metal::MetalContext::new().is_err() {
        eprintln!("skipping qwen38_batched_prefill_greedy: no Metal GPU");
        return;
    }
    fusion_env();
    let weights = match Qwen38HybridWeights::load(ARTIFACT) {
        Ok(w) => Arc::new(w),
        Err(err) => {
            eprintln!("skipping qwen38_batched_prefill_greedy: load failed: {err}");
            return;
        }
    };
    let tokenizer = load_qwen38_tokenizer(TOKENIZER).expect("tokenizer");
    let prompt = tokenizer.encode(PROMPT, false).expect("encode");
    assert!(!prompt.is_empty(), "prompt tokenized empty");

    std::env::set_var("HAWKING_QWEN38_BATCH_PREFILL", "0");
    let mut sequential = Qwen38HybridDecodeSession::attach(Arc::clone(&weights), MAX_SEQ)
        .expect("attach sequential");
    let seq = generate_greedy(&mut sequential, &prompt, MAX_NEW).expect("sequential greedy");
    let seq_ids = seq.new_tokens().to_vec();
    assert_eq!(
        seq_ids.len(),
        MAX_NEW,
        "sequential path produced {} tokens, want {MAX_NEW}",
        seq_ids.len()
    );
    assert!(
        !seq.batched_prefill,
        "sequential arm set batched_prefill; the disable flag is broken"
    );

    std::env::set_var("HAWKING_QWEN38_BATCH_PREFILL", "1");
    let mut batched = Qwen38HybridDecodeSession::attach(weights, MAX_SEQ).expect("attach batched");
    let bat = generate_greedy(&mut batched, &prompt, MAX_NEW).expect("batched greedy");
    let bat_ids = bat.new_tokens().to_vec();

    assert!(
        bat.batched_prefill,
        "batched path was not taken; identity against sequential would be vacuous"
    );
    assert!(
        bat.prefill_dispatches > 0,
        "batched prefill recorded zero dispatches"
    );
    assert!(
        bat.prefill_dispatches < seq.prefill_dispatches / 4,
        "prefill dispatches did not drop (seq={}, bat={}); GEMM path not engaged",
        seq.prefill_dispatches,
        bat.prefill_dispatches
    );
    assert_eq!(
        seq_ids, bat_ids,
        "greedy token ids must match for {MAX_NEW} new tokens\nseq={seq_ids:?}\nbat={bat_ids:?}"
    );

    std::env::remove_var("HAWKING_QWEN38_BATCH_PREFILL");
}

#[test]
fn batched_prefill_flag_is_load_bearing() {
    // Mutation check: disabling the kernel must fail the engagement
    // assertions of the sibling test, not just the token compare.
    assert!(
        hawking_core::model::qwen38_hybrid_decode::qwen38_batched_prefill_enabled()
            || std::env::var("HAWKING_QWEN38_BATCH_PREFILL")
                .ok()
                .as_deref()
                == Some("0"),
        "default must be on unless the operator set =0"
    );
    let saved = std::env::var("HAWKING_QWEN38_BATCH_PREFILL").ok();
    std::env::set_var("HAWKING_QWEN38_BATCH_PREFILL", "0");
    assert!(
        !hawking_core::model::qwen38_hybrid_decode::qwen38_batched_prefill_enabled(),
        "HAWKING_QWEN38_BATCH_PREFILL=0 must select the sequential path"
    );
    match saved {
        Some(v) => std::env::set_var("HAWKING_QWEN38_BATCH_PREFILL", v),
        None => std::env::remove_var("HAWKING_QWEN38_BATCH_PREFILL"),
    }
}
