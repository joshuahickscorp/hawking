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

fn gen_on_n(
    engine: &mut dyn dismantle_core::Engine,
    prompt: &str,
    max_new_tokens: usize,
) -> (Vec<u32>, usize) {
    let req = dismantle_core::GenerateRequest {
        prompt: prompt.into(),
        max_new_tokens,
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

fn gen_on(engine: &mut dyn dismantle_core::Engine, prompt: &str) -> (Vec<u32>, usize) {
    gen_on_n(engine, prompt, MAX_NEW_TOKENS)
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

// ── Env helpers for the full shipped fast-decode recipe. The production
// decode path (tools/bench/clean_room_batch.sh) runs VOCAB_PRUNE + Q4K_LMHEAD
// + FFN_DOWN_Q4K + Q4K_PREDEC; the two gates above cover only the first two.
// Set/clear them together so they never leak into a sibling serialized test.
fn set_full_fast_env() {
    std::env::set_var("DISMANTLE_QWEN_VOCAB_PRUNE", "32000");
    std::env::set_var("DISMANTLE_QWEN_Q4K_LMHEAD", "1");
    std::env::set_var("DISMANTLE_QWEN_FFN_DOWN_Q4K", "1");
    std::env::set_var("DISMANTLE_QWEN_Q4K_PREDEC", "1");
}
fn clear_full_fast_env() {
    std::env::remove_var("DISMANTLE_QWEN_VOCAB_PRUNE");
    std::env::remove_var("DISMANTLE_QWEN_Q4K_LMHEAD");
    std::env::remove_var("DISMANTLE_QWEN_FFN_DOWN_Q4K");
    std::env::remove_var("DISMANTLE_QWEN_Q4K_PREDEC");
}

/// (a) THE GATE on the **full shipped fast-decode env** — the recipe the
/// production CLI / clean-room bench actually runs (VOCAB_PRUNE + Q4K_LMHEAD +
/// FFN_DOWN_Q4K + Q4K_PREDEC), which the two gates above do NOT cover. This is
/// the env the failing CLI run in reports/move2_user_draft_diagnosis.md used
/// (with the draft silently OFF because the flag was unset).
///
/// Draft ON must be byte-identical to draft OFF under this env. We do NOT
/// assert `draft_accepted > 0`: FFN_DOWN_Q4K perturbs the verifier's logits and
/// can legitimately lower acceptance (measured 7 → 3, contaminated), so the
/// accept count is REPORTED, not gated — the gate here is correctness, not
/// speed (diagnosis §5 test 1).
#[test]
fn user_draft_bit_identical_full_fast_env() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap();

    set_full_fast_env();

    // --- Reference: draft OFF (full fast decode). ---
    std::env::set_var("DISMANTLE_QWEN_USER_DRAFT", "0");
    let (ref_ids, _) = {
        let mut e = make_engine(&weights);
        gen_on(e.as_mut(), PROMPT)
    };

    // --- Draft ON (full fast decode). ---
    std::env::set_var("DISMANTLE_QWEN_USER_DRAFT", "1");
    let (draft_ids, accepted) = {
        let mut e = make_engine(&weights);
        gen_on(e.as_mut(), PROMPT)
    };

    std::env::set_var("DISMANTLE_QWEN_USER_DRAFT", "0");
    clear_full_fast_env();

    assert_eq!(draft_ids.len(), MAX_NEW_TOKENS, "draft-ON produced wrong token count");
    assert_eq!(ref_ids.len(), MAX_NEW_TOKENS, "draft-OFF produced wrong token count");
    assert_eq!(
        &ref_ids[..3],
        &draft_ids[..3],
        "GATE FAILED (first 3 tokens, full fast env): user-draft changed greedy output.\n \
         off={:?}\n  on={:?}",
        &ref_ids[..3],
        &draft_ids[..3],
    );
    assert_eq!(
        ref_ids, draft_ids,
        "GATE FAILED (16 tokens, full fast env): user-draft changed greedy output.\n \
         off={ref_ids:?}\n  on={draft_ids:?}"
    );

    eprintln!("\n=== user-draft parity gate (FULL fast env) ===");
    eprintln!("draft OFF: {ref_ids:?}");
    eprintln!("draft ON : {draft_ids:?}  (draft_accepted={accepted}, reported not gated)");
    eprintln!("bit-identical: YES");
    eprintln!("==============================================\n");
}

/// Shared body for (b): propose-first vs bonus-first must emit the SAME tokens.
/// Both are bit-identical to plain greedy by construction; comparing them to
/// EACH OTHER pins that the propose-first restructure changed only the
/// forward-count schedule, not which tokens are emitted. `pruned` selects the
/// GPU fast verify path (VOCAB_PRUNE + Q4K_LMHEAD) vs the CPU fp16 fallback, so
/// the new loop is exercised on both. Returns (bonus_first_ids, accepted)
/// alongside the asserted-equal propose_first_ids for the caller to log.
fn propose_first_matches_bonus_first(weights: &PathBuf, pruned: bool, n: usize) -> (Vec<u32>, Vec<u32>, usize, usize) {
    if pruned {
        std::env::set_var("DISMANTLE_QWEN_VOCAB_PRUNE", "32000");
        std::env::set_var("DISMANTLE_QWEN_Q4K_LMHEAD", "1");
    }
    // Both arms enable the draft; they differ only in the loop variant.
    std::env::set_var("DISMANTLE_QWEN_USER_DRAFT", "1");

    // --- Reference: bonus-first ('ud_loop), propose-first OFF. ---
    std::env::remove_var("DISMANTLE_QWEN_USER_DRAFT_PROPOSE_FIRST");
    let (bonus_ids, bonus_acc) = {
        let mut e = make_engine(weights);
        gen_on_n(e.as_mut(), PROMPT, n)
    };

    // --- Propose-first ('udpf_loop), propose-first ON. ---
    std::env::set_var("DISMANTLE_QWEN_USER_DRAFT_PROPOSE_FIRST", "1");
    let (pf_ids, pf_acc) = {
        let mut e = make_engine(weights);
        gen_on_n(e.as_mut(), PROMPT, n)
    };

    // Restore.
    std::env::remove_var("DISMANTLE_QWEN_USER_DRAFT_PROPOSE_FIRST");
    std::env::set_var("DISMANTLE_QWEN_USER_DRAFT", "0");
    if pruned {
        std::env::remove_var("DISMANTLE_QWEN_VOCAB_PRUNE");
        std::env::remove_var("DISMANTLE_QWEN_Q4K_LMHEAD");
    }

    assert_eq!(bonus_ids.len(), n, "bonus-first produced wrong token count");
    assert_eq!(pf_ids.len(), n, "propose-first produced wrong token count");
    assert_eq!(
        bonus_ids, pf_ids,
        "GATE FAILED (propose-first vs bonus-first, pruned={pruned}, n={n}): the \
         propose-first loop changed the emitted token stream.\n \
         bonus-first={bonus_ids:?}\n propose-first={pf_ids:?}"
    );
    (bonus_ids, pf_ids, bonus_acc, pf_acc)
}

/// (b/default) Propose-first ≡ bonus-first on the DEFAULT (CPU fp16 fallback)
/// verify path. Diagnosis §5 test 2.
#[test]
fn user_draft_propose_first_bit_identical_default() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap();
    let (bonus_ids, pf_ids, bonus_acc, pf_acc) =
        propose_first_matches_bonus_first(&weights, false, MAX_NEW_TOKENS);
    eprintln!("\n=== propose-first vs bonus-first (default cfg, 16 tok) ===");
    eprintln!("bonus-first : {bonus_ids:?}  (draft_accepted={bonus_acc})");
    eprintln!("propose-first: {pf_ids:?}  (draft_accepted={pf_acc})");
    eprintln!("token-identical: YES");
    eprintln!("=========================================================\n");
}

/// (b/pruned) Propose-first ≡ bonus-first on the GPU pruned-Q4K fast verify
/// path (the one production decode runs). Diagnosis §5 test 2.
#[test]
fn user_draft_propose_first_bit_identical_pruned_q4k() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap();
    let (bonus_ids, pf_ids, bonus_acc, pf_acc) =
        propose_first_matches_bonus_first(&weights, true, MAX_NEW_TOKENS);
    eprintln!("\n=== propose-first vs bonus-first (pruned-Q4K cfg, 16 tok) ===");
    eprintln!("bonus-first : {bonus_ids:?}  (draft_accepted={bonus_acc})");
    eprintln!("propose-first: {pf_ids:?}  (draft_accepted={pf_acc})");
    eprintln!("token-identical: YES");
    eprintln!("============================================================\n");
}

/// (c) A 64-token lossless run (propose-first vs bonus-first, pruned-Q4K fast
/// verify). A 16-token window can miss a KV-rewind off-by-one in the new loop's
/// accept/advance bookkeeping (the highest-risk part of the port — the eagle5
/// 'pf_loop advance analog); 64 tokens exercises many more accept/reject
/// boundaries and KV rewinds. Diagnosis §5 test 3.
#[test]
fn user_draft_propose_first_lossless_long() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap();
    let (bonus_ids, pf_ids, bonus_acc, pf_acc) =
        propose_first_matches_bonus_first(&weights, true, 64);
    eprintln!("\n=== propose-first vs bonus-first (pruned-Q4K cfg, 64 tok) ===");
    eprintln!("bonus-first  draft_accepted={bonus_acc}");
    eprintln!("propose-first draft_accepted={pf_acc}");
    eprintln!("token-identical (64 tok): YES");
    eprintln!("len bonus={} pf={}", bonus_ids.len(), pf_ids.len());
    eprintln!("============================================================\n");
}
