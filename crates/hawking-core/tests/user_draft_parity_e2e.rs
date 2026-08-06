#![cfg(target_os = "macos")]
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
const WEIGHTS: &str = "../../models/qwen2.5-3b-instruct-q4_k_m.gguf";
const PROMPT: &str = "fn add(a: i32, b: i32) -> i32 { a + b }\nfn add(a: i32, b: i32) -> i32 { a + b }\nfn add(a: i32, b: i32) -> i32 {";
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
fn make_engine(weights: &PathBuf) -> Box<dyn hawking_core::Engine> {
    std::env::set_var("HAWKING_QWEN_TCB", "1");
    std::env::set_var("HAWKING_QWEN_PREFIX_CACHE", "0");
    std::env::set_var("HAWKING_QWEN_PAIR_2R_INLINE", "0");
    hawking_core::model::load_engine(weights, hawking_core::EngineConfig::default())
        .expect("load engine")
}
fn gen_on_n(
    engine: &mut dyn hawking_core::Engine,
    prompt: &str,
    max_new_tokens: usize,
) -> (Vec<u32>, usize) {
    let req = hawking_core::GenerateRequest {
        prompt: prompt.into(),
        max_new_tokens,
        sampling: hawking_core::SamplingParams {
            temperature: 0.0,
            seed: Some(42),
            ..Default::default()
        },
        stop: vec![],
        abort: None,
        max_stall_ms: 0,
        json_mode: false,
    };
    let mut ids: Vec<u32> = Vec::new();
    let mut accepted = 0usize;
    engine
        .generate(req, &mut |ev| match ev {
            hawking_core::StreamEvent::Token { id, .. } => ids.push(id),
            hawking_core::StreamEvent::Done { stats, .. } => accepted = stats.draft_accepted,
            _ => {}
        })
        .expect("generate");
    (ids, accepted)
}
fn gen_on(engine: &mut dyn hawking_core::Engine, prompt: &str) -> (Vec<u32>, usize) {
    gen_on_n(engine, prompt, MAX_NEW_TOKENS)
}
fn lock_gate() -> std::sync::MutexGuard<'static, ()> {
    SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap()
}
fn draft_on_vs_off(
    weights: &PathBuf,
    label: &str,
    setup: impl FnOnce(),
    teardown: impl FnOnce(),
    require_accept: bool,
) {
    setup();
    std::env::set_var("HAWKING_QWEN_USER_DRAFT", "0");
    let (ref_ids, _) = {
        let mut e = make_engine(weights);
        gen_on(e.as_mut(), PROMPT)
    };
    std::env::set_var("HAWKING_QWEN_USER_DRAFT", "1");
    let (draft_ids, accepted) = {
        let mut e = make_engine(weights);
        gen_on(e.as_mut(), PROMPT)
    };
    std::env::set_var("HAWKING_QWEN_USER_DRAFT", "0");
    teardown();
    assert_eq!(
        draft_ids.len(),
        MAX_NEW_TOKENS,
        "{label}: draft-ON wrong token count"
    );
    assert_eq!(
        ref_ids.len(),
        MAX_NEW_TOKENS,
        "{label}: draft-OFF wrong token count"
    );
    assert_eq!(
        &ref_ids[..3],
        &draft_ids[..3],
        "GATE FAILED (first 3, {label})"
    );
    assert_eq!(ref_ids, draft_ids, "GATE FAILED (16 tokens, {label})");
    if require_accept {
        assert!(accepted > 0, "{label}: draft_accepted=0 — vacuous gate");
    }
}
fn set_pruned_q4k() {
    std::env::set_var("HAWKING_QWEN_VOCAB_PRUNE", "32000");
    std::env::set_var("HAWKING_QWEN_Q4K_LMHEAD", "1");
}
fn clear_pruned_q4k() {
    std::env::remove_var("HAWKING_QWEN_VOCAB_PRUNE");
    std::env::remove_var("HAWKING_QWEN_Q4K_LMHEAD");
}
fn set_full_fast_env() {
    set_pruned_q4k();
    std::env::set_var("HAWKING_QWEN_FFN_DOWN_Q4K", "1");
    std::env::set_var("HAWKING_QWEN_Q4K_PREDEC", "1");
}
fn clear_full_fast_env() {
    clear_pruned_q4k();
    std::env::remove_var("HAWKING_QWEN_FFN_DOWN_Q4K");
    std::env::remove_var("HAWKING_QWEN_Q4K_PREDEC");
}
#[test]
fn user_draft_is_bit_identical() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = lock_gate();
    draft_on_vs_off(&weights, "default", || {}, || {}, false);
}
#[test]
fn user_draft_bit_identical_fast_pruned_q4k() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = lock_gate();
    draft_on_vs_off(
        &weights,
        "fast pruned-Q4K",
        set_pruned_q4k,
        clear_pruned_q4k,
        true,
    );
}
#[test]
fn user_draft_bit_identical_full_fast_env() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = lock_gate();
    draft_on_vs_off(
        &weights,
        "full fast env",
        set_full_fast_env,
        clear_full_fast_env,
        false,
    );
}
fn propose_first_matches_bonus_first(
    weights: &PathBuf,
    pruned: bool,
    n: usize,
) -> (Vec<u32>, Vec<u32>, usize, usize) {
    if pruned {
        set_pruned_q4k();
    }
    std::env::set_var("HAWKING_QWEN_USER_DRAFT", "1");
    std::env::remove_var("HAWKING_QWEN_USER_DRAFT_PROPOSE_FIRST");
    let (bonus_ids, bonus_acc) = {
        let mut e = make_engine(weights);
        gen_on_n(e.as_mut(), PROMPT, n)
    };
    std::env::set_var("HAWKING_QWEN_USER_DRAFT_PROPOSE_FIRST", "1");
    let (pf_ids, pf_acc) = {
        let mut e = make_engine(weights);
        gen_on_n(e.as_mut(), PROMPT, n)
    };
    std::env::remove_var("HAWKING_QWEN_USER_DRAFT_PROPOSE_FIRST");
    std::env::set_var("HAWKING_QWEN_USER_DRAFT", "0");
    if pruned {
        clear_pruned_q4k();
    }
    assert_eq!(bonus_ids.len(), n, "bonus-first wrong token count");
    assert_eq!(pf_ids.len(), n, "propose-first wrong token count");
    assert_eq!(
        bonus_ids, pf_ids,
        "GATE FAILED propose-first vs bonus-first pruned={pruned} n={n}"
    );
    (bonus_ids, pf_ids, bonus_acc, pf_acc)
}
#[test]
fn user_draft_propose_first_bit_identical_default() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = lock_gate();
    let _ = propose_first_matches_bonus_first(&weights, false, MAX_NEW_TOKENS);
}
#[test]
fn user_draft_propose_first_bit_identical_pruned_q4k() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = lock_gate();
    let _ = propose_first_matches_bonus_first(&weights, true, MAX_NEW_TOKENS);
}
#[test]
fn user_draft_propose_first_lossless_long() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = lock_gate();
    let _ = propose_first_matches_bonus_first(&weights, true, 64);
}
