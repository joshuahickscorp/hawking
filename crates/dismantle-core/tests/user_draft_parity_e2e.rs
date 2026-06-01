//! End-to-end LOSSLESS gate for the per-user n-gram draft (L3.1 §2.1b,
//! `DISMANTLE_QWEN_USER_DRAFT`). Runs on the real Qwen-3B model; skips
//! silently when the weights are missing so CI stays green without them.
//!
//! ## THE GATE (bit-identical output)
//!
//! The draft is lossless **by construction**: it is only a draft *source*
//! for a propose→batched-verify→accept loop that reuses the landed verifier
//! (`forward_tokens_verify`) unchanged, so the verifier emits every token.
//! A request with `DISMANTLE_QWEN_USER_DRAFT=1` must therefore produce the
//! EXACT same greedy (temp=0) output as the same request with the draft
//! OFF — drafts change only how many tokens land per verify forward, never
//! which tokens. We assert full token-vector equality (`==`) on the first
//! 16 tokens (the project's token-parity window is 3; 16 is stricter).
//!
//! ## Drafting is actually exercised
//!
//! The prompt is repetitive code (the regime the n-gram draft targets), so
//! the index proposes and the verifier accepts a non-zero number of drafts
//! (`stats.draft_accepted > 0` is reported, not asserted — acceptance is a
//! speed property, and the gate is correctness). If the draft path silently
//! no-op'd, the output would still match, but the accept count exposes it.

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

const WEIGHTS: &str = "../../models/qwen2.5-3b-instruct-q4_k_m.gguf";

// Repetitive code-y prompt: literal token runs (repeated identifiers /
// boilerplate) are exactly what the user n-gram index learns to draft.
const PROMPT: &str =
    "fn add(a: i32, b: i32) -> i32 { a + b }\nfn add(a: i32, b: i32) -> i32 { a + b }\nfn add(a: i32, b: i32) -> i32 {";

const MAX_NEW_TOKENS: usize = 16;

static SERIAL_GATE: OnceLock<Mutex<()>> = OnceLock::new();

fn weights_path() -> Option<PathBuf> {
    let p = PathBuf::from(WEIGHTS);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("user_draft_parity_e2e: skipping — no weights at {WEIGHTS}");
        None
    }
}

fn make_engine(weights: &PathBuf) -> Box<dyn dismantle_core::Engine> {
    // The user-draft path requires the production TCB (full-Metal, greedy)
    // decode path: it uses forward_token_greedy_tcb for the bonus + the
    // batched forward_tokens_verify primitive.
    std::env::set_var("DISMANTLE_QWEN_TCB", "1");
    // Keep the prefix cache out of this gate so the only variable is the
    // draft flag (default-on prefix cache is bit-identical anyway, but a
    // single-request gate has no prior session to hit, so it is inert).
    std::env::set_var("DISMANTLE_QWEN_PREFIX_CACHE", "0");
    let cfg = dismantle_core::EngineConfig::default();
    dismantle_core::model::load_engine(weights, cfg).expect("load engine")
}

fn gen_on(engine: &mut dyn dismantle_core::Engine, prompt: &str) -> (Vec<u32>, usize) {
    let req = dismantle_core::GenerateRequest {
        prompt: prompt.into(),
        max_new_tokens: MAX_NEW_TOKENS,
        sampling: dismantle_core::SamplingParams {
            temperature: 0.0,
            seed: Some(42),
            ..Default::default()
        },
        stop: vec![],
        abort: None,
        max_stall_ms: 0,
    };
    let mut ids: Vec<u32> = Vec::new();
    let mut accepted = 0usize;
    engine
        .generate(req, &mut |ev| match ev {
            dismantle_core::StreamEvent::Token { id, .. } => ids.push(id),
            dismantle_core::StreamEvent::Done { stats, .. } => {
                accepted = stats.draft_accepted;
            }
        })
        .expect("generate");
    (ids, accepted)
}

/// THE GATE: `DISMANTLE_QWEN_USER_DRAFT=1` greedy output must be
/// byte-identical to the draft-OFF greedy output on the same prompt.
#[test]
fn user_draft_is_bit_identical() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap();

    // --- Reference: draft OFF. ---
    std::env::set_var("DISMANTLE_QWEN_USER_DRAFT", "0");
    let (ref_ids, _) = {
        let mut e = make_engine(&weights);
        gen_on(e.as_mut(), PROMPT)
    };

    // --- Draft ON. ---
    std::env::set_var("DISMANTLE_QWEN_USER_DRAFT", "1");
    let (draft_ids, accepted) = {
        let mut e = make_engine(&weights);
        gen_on(e.as_mut(), PROMPT)
    };
    std::env::set_var("DISMANTLE_QWEN_USER_DRAFT", "0");

    assert_eq!(draft_ids.len(), MAX_NEW_TOKENS, "draft-ON produced wrong token count");
    assert_eq!(ref_ids.len(), MAX_NEW_TOKENS, "draft-OFF produced wrong token count");

    // First 3 tokens (the project's token-parity window) — checked
    // explicitly so a failure message pinpoints the window.
    assert_eq!(
        &ref_ids[..3],
        &draft_ids[..3],
        "GATE FAILED (first 3 tokens): user-draft changed greedy output.\n \
         off={:?}\n  on={:?}",
        &ref_ids[..3],
        &draft_ids[..3],
    );
    // Full 16-token vector — the strict gate.
    assert_eq!(
        ref_ids, draft_ids,
        "GATE FAILED (16 tokens): user-draft changed greedy output.\n \
         off={ref_ids:?}\n  on={draft_ids:?}"
    );

    eprintln!("\n=== user-draft parity gate ===");
    eprintln!("draft OFF: {ref_ids:?}");
    eprintln!("draft ON : {draft_ids:?}  (draft_accepted={accepted})");
    eprintln!("bit-identical: YES");
    eprintln!("==============================\n");
}

/// THE GATE on the **fast pruned-Q4K verify path** (the one production decode
/// actually runs, and the one the sibling gate above does NOT cover).
///
/// With the shipped vocab-pruned Q4_K LM head active (`DISMANTLE_QWEN_VOCAB_PRUNE`
/// + `DISMANTLE_QWEN_Q4K_LMHEAD`), `forward_tokens_verify` takes its GPU fast path
/// — ONE `gemm_q4_k_m_batched_v3w` over the pruned head + a CPU argmax over the
/// pruned logits — instead of the CPU fp16 full-vocab fallback. Commit `010827b`
/// introduced that path as bit-identical to greedy "vs the CPU fp16 full-vocab
/// path which diverged", but nothing gated it: the clean-room fast-decode env
/// (`tools/bench/clean_room_batch.sh`) sets `_Q4K_LMHEAD`, while the sibling gate
/// runs the plain f16 head. This closes that coverage gap.
///
/// Output must be byte-identical to draft-OFF on the same pruned-Q4K decode, and
/// the draft must actually fire (`draft_accepted > 0`) so the fast verify path is
/// genuinely exercised — otherwise a silent no-op draft would pass vacuously.
#[test]
fn user_draft_bit_identical_fast_pruned_q4k() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap();

    // Activate the shipped pruned-Q4K LM head so the verify takes its GPU fast
    // path (`vocab_pruned_is_q4k == true`). Restored after so env does not leak
    // into the sibling (CPU-fallback) gate, whatever order the two run in.
    std::env::set_var("DISMANTLE_QWEN_VOCAB_PRUNE", "32000");
    std::env::set_var("DISMANTLE_QWEN_Q4K_LMHEAD", "1");

    // --- Reference: draft OFF (still on the pruned-Q4K decode). ---
    std::env::set_var("DISMANTLE_QWEN_USER_DRAFT", "0");
    let (ref_ids, _) = {
        let mut e = make_engine(&weights);
        gen_on(e.as_mut(), PROMPT)
    };

    // --- Draft ON → GPU pruned-Q4K batched verify. ---
    std::env::set_var("DISMANTLE_QWEN_USER_DRAFT", "1");
    let (draft_ids, accepted) = {
        let mut e = make_engine(&weights);
        gen_on(e.as_mut(), PROMPT)
    };

    std::env::set_var("DISMANTLE_QWEN_USER_DRAFT", "0");
    std::env::remove_var("DISMANTLE_QWEN_VOCAB_PRUNE");
    std::env::remove_var("DISMANTLE_QWEN_Q4K_LMHEAD");

    assert_eq!(draft_ids.len(), MAX_NEW_TOKENS, "draft-ON produced wrong token count");
    assert_eq!(ref_ids.len(), MAX_NEW_TOKENS, "draft-OFF produced wrong token count");
    assert_eq!(
        &ref_ids[..3],
        &draft_ids[..3],
        "GATE FAILED (first 3 tokens, fast pruned-Q4K verify): user-draft changed greedy output.\n \
         off={:?}\n  on={:?}",
        &ref_ids[..3],
        &draft_ids[..3],
    );
    assert_eq!(
        ref_ids, draft_ids,
        "GATE FAILED (16 tokens, fast pruned-Q4K verify): user-draft changed greedy output.\n \
         off={ref_ids:?}\n  on={draft_ids:?}"
    );
    // The fast verify path must actually have run, or the gate is vacuous.
    assert!(
        accepted > 0,
        "fast pruned-Q4K verify not exercised: draft_accepted=0 (the n-gram never \
         proposed an accepted token, so forward_tokens_verify's GPU path never ran). \
         Check the repetitive PROMPT and the VOCAB_PRUNE/Q4K_LMHEAD env."
    );

    eprintln!("\n=== user-draft parity gate (FAST pruned-Q4K verify) ===");
    eprintln!("draft OFF: {ref_ids:?}");
    eprintln!("draft ON : {draft_ids:?}  (draft_accepted={accepted})");
    eprintln!("bit-identical: YES");
    eprintln!("=======================================================\n");
}
