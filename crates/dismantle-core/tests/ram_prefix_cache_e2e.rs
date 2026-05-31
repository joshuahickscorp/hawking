//! End-to-end bit-identical-reuse gate for the in-RAM prefix cache (B1,
//! Bible §8 L1.2). Runs on the real Qwen-3B model; skips silently when
//! the weights are missing so CI stays green without them.
//!
//! ## THE GATE (bit-identical reuse)
//!
//! A request whose prefix was served from the in-RAM cache must produce
//! the EXACT same greedy output as the same request with the cache OFF.
//! A matched prefix is a pure function of (model, tokenizer, tokens), so
//! the restored KV equals a cold prefill's → identical logits → identical
//! greedy argmax. We assert full token-vector equality (`==`).
//!
//! The scenario is a real two-request *session* against ONE persistent
//! engine (the in-RAM tier's whole purpose): request 1 establishes the
//! prefix, request 2 shares it and must hit + stay bit-identical.
//!
//! ## Bench (informational)
//!
//! Reports request-2 prefill_ms with the cache ON vs OFF — the win is
//! prefill / first-token latency, not steady-state dec_tps.

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

const WEIGHTS: &str = "../../models/qwen2.5-3b-instruct-q4_k_m.gguf";

// A long "file preamble" that gets re-sent (the redundant majority of a
// coding / multi-turn session). Request 2's prompt has request 1's prompt
// as a strict token prefix — exactly the case prefix caching accelerates:
// the retained KV for the shared leading tokens is reused, so request 2
// only prefills the *new* turn.
const PREAMBLE: &str = "Here is a Rust file I am working on:\n\nfn fib(n: u64) -> u64 {\n    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }\n}\n\nfn factorial(n: u64) -> u64 {\n    (1..=n).product()\n}\n\nfn main() {\n    for i in 0..10 {\n        println!(\"fib({}) = {}\", i, fib(i));\n    }\n    for i in 0..6 {\n        println!(\"fact({}) = {}\", i, factorial(i));\n    }\n}\n\n";
// Turn 1: preamble + first question.
const TURN1_TAIL: &str = "User: What does this program print?\nAssistant:";
// Turn 2 = preamble + first Q&A + a follow-up question. Its tokenization
// begins with turn 1's exact tokens, so the cache stored under turn 1's
// full-prompt key is a strict prefix of turn 2 → HIT.
const TURN2_TAIL: &str = "User: What does this program print?\nAssistant: It prints the first ten Fibonacci numbers and the first six factorials.\nUser: What is the time complexity of fib here?\nAssistant:";

const MAX_NEW_TOKENS: usize = 32;

static SERIAL_GATE: OnceLock<Mutex<()>> = OnceLock::new();

fn weights_path() -> Option<PathBuf> {
    let p = PathBuf::from(WEIGHTS);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("ram_prefix_cache_e2e: skipping — no weights at {WEIGHTS}");
        None
    }
}

fn make_engine(weights: &PathBuf) -> Box<dyn dismantle_core::Engine> {
    // Exercise the production TCB (full-Metal, greedy) decode path. This
    // is both the fast path AND the harder correctness case: the GPU
    // decode arena caches K/V in GPU buffers and only bridges the CPU
    // prefix on a fresh arena, so a same-session prefix-cache hit must
    // force a re-bridge. If the gate is bit-identical here, it is
    // bit-identical on the CPU/Metal-hybrid path too (that path keeps
    // self.kv authoritative throughout).
    std::env::set_var("DISMANTLE_QWEN_TCB", "1");
    let cfg = dismantle_core::EngineConfig::default();
    dismantle_core::model::load_engine(weights, cfg).expect("load engine")
}

fn gen_on(engine: &mut dyn dismantle_core::Engine, prompt: &str) -> (Vec<u32>, f64) {
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
    let mut prefill_ms = 0.0f64;
    engine
        .generate(req, &mut |ev| match ev {
            dismantle_core::StreamEvent::Token { id, .. } => ids.push(id),
            dismantle_core::StreamEvent::Done { stats, .. } => prefill_ms = stats.prefill_ms,
        })
        .expect("generate");
    (ids, prefill_ms)
}

/// THE GATE: a cache HIT on request 2 (same persistent engine/session)
/// must produce byte-identical tokens to request 2 with the cache OFF.
#[test]
fn ram_cache_hit_is_bit_identical() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap();

    // Turn 2's prompt has turn 1's prompt as a strict token prefix.
    let prompt1 = format!("{PREAMBLE}{TURN1_TAIL}");
    let prompt2 = format!("{PREAMBLE}{TURN2_TAIL}");

    // --- Reference: cache OFF. Fresh engine, request 2 cold. ---
    std::env::remove_var("DISMANTLE_QWEN_PREFIX_CACHE");
    std::env::remove_var("DISMANTLE_PREFIX_CACHE_DIR");
    let (ref_ids, off_ms) = {
        let mut e = make_engine(&weights);
        // Run request 1 first so the engine state (arena, etc.) matches
        // the cache-on path's history as closely as possible.
        let _ = gen_on(e.as_mut(), &prompt1);
        gen_on(e.as_mut(), &prompt2)
    };

    // --- Cache ON: same persistent engine across both requests. ---
    std::env::set_var("DISMANTLE_QWEN_PREFIX_CACHE", "1");
    let (hit_ids, on_ms) = {
        let mut e = make_engine(&weights);
        let _ = gen_on(e.as_mut(), &prompt1); // populates the RAM prefix cache
        gen_on(e.as_mut(), &prompt2) // turn 2 extends turn 1 → HIT
    };
    std::env::remove_var("DISMANTLE_QWEN_PREFIX_CACHE");

    assert_eq!(hit_ids.len(), MAX_NEW_TOKENS);
    // THE GATE: bit-identical reuse.
    assert_eq!(
        ref_ids, hit_ids,
        "GATE FAILED: RAM prefix-cache hit changed greedy output.\n off={ref_ids:?}\n  on={hit_ids:?}"
    );
    // The win: a real hit must skip the shared prefix's prefill. The
    // shared prefix is the dominant majority of turn 2, so prefill should
    // drop substantially. A loose >15% bound catches a wire-up regression
    // (hit detected but prefill not skipped) without being timing-flaky.
    assert!(
        on_ms < 0.85 * off_ms,
        "expected prefix-cache HIT to cut turn-2 prefill >15%; got on={on_ms:.1} vs off={off_ms:.1} \
         (no hit ⇒ check lookup/extends-prefix scenario)"
    );

    eprintln!("\n=== RAM prefix-cache request-2 prefill ===");
    eprintln!("cache OFF (cold req 2): prefill_ms = {off_ms:>7.1}");
    eprintln!(
        "cache ON  (HIT req 2):  prefill_ms = {on_ms:>7.1}  ({:>5.1}% of off)",
        100.0 * on_ms / off_ms.max(1e-9)
    );
    eprintln!("==========================================\n");
}

/// A cache MISS (cache on, but request 2 shares no prefix) must stay
/// bit-identical to the cache-OFF path. To isolate the cache's effect
/// from any persistent-engine state, BOTH arms run the identical
/// two-request sequence on a persistent engine; only the flag differs.
#[test]
fn ram_cache_miss_is_bit_identical() {
    let Some(weights) = weights_path() else {
        return;
    };
    let _g = SERIAL_GATE.get_or_init(|| Mutex::new(())).lock().unwrap();

    // Two prompts that share no leading tokens.
    let prompt_a = format!("Apples are red. {TURN1_TAIL}");
    let prompt_b = format!("Bananas are yellow. {TURN1_TAIL}");

    // Reference: cache OFF, same persistent engine, same 2-request order.
    std::env::remove_var("DISMANTLE_QWEN_PREFIX_CACHE");
    std::env::remove_var("DISMANTLE_PREFIX_CACHE_DIR");
    let ref_ids = {
        let mut e = make_engine(&weights);
        let _ = gen_on(e.as_mut(), &prompt_a);
        gen_on(e.as_mut(), &prompt_b).0
    };

    // Cache ON: prompt_b shares no prefix with prompt_a → miss → plain
    // prefill. Must match the cache-OFF reference exactly.
    std::env::set_var("DISMANTLE_QWEN_PREFIX_CACHE", "1");
    let miss_ids = {
        let mut e = make_engine(&weights);
        let _ = gen_on(e.as_mut(), &prompt_a); // unrelated prefix in cache
        gen_on(e.as_mut(), &prompt_b).0 // no shared prefix → miss
    };
    std::env::remove_var("DISMANTLE_QWEN_PREFIX_CACHE");

    assert_eq!(
        ref_ids, miss_ids,
        "cache-miss path must match the no-cache path on the same engine"
    );
}
